//! [`PassScheduler`] - the core engine that orchestrates [`SsaPass`]
//! execution across methods.
//!
//! The scheduler organizes registered passes into execution layers
//! computed from capability dependencies, runs each layer to fixpoint
//! with normalize passes interleaved, dispatches per-method work in
//! parallel via rayon, and applies modification-scope-driven SSA repair
//! after each pass.
//!
//! # Layer computation
//!
//! Passes that do not declare capabilities fall back to their
//! [`SsaPass::fallback_layer`]. Passes with declared capabilities are
//! topologically sorted: if pass B requires a capability that pass A
//! provides, B is placed in a strictly later layer than A.
//!
//! # Normalization
//!
//! Normalize passes (DCE, GVN, copy propagation, etc.) are separate from
//! the layered passes. They run between every layer's fixpoint iterations,
//! cleaning up after each round of structural changes to expose new
//! optimization opportunities.
//!
//! # Dirty tracking
//!
//! The scheduler seeds the dirty set with all methods at pipeline start.
//! After each fixpoint iteration, the dirty set is updated: methods
//! modified during the iteration remain dirty; unmodified methods are
//! cleared. Passes that declare `requires_full_scan()` bypass dirty
//! tracking entirely.

use std::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use dashmap::DashSet;
use log::{Level, debug, log_enabled};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    analysis::{
        budget::{Budget, Rounds},
        verifier::VerifyLevel,
    },
    error::{Error, Result},
    events::{Event, EventKind},
    graph::IndexedGraph,
    ir::function::SsaFunction,
    parallel::MapCarried,
    passes::{
        AlgebraicSimplificationPass, ArmMergingPass, BlockMergingPass,
        ControlFlowSimplificationPass, CopyPropagationPass, DeadCodeEliminationPass,
        DeadMethodEliminationPass, DuplicateMergingPass, GlobalValueNumberingPass,
        JumpThreadingPass, LicmPass, LoopCanonicalizationPass, LoopRejoiningPass,
        MemoryOptimizationPass, MootTestPass, OpaquePredicatePass, ReassociationPass,
        StrengthReductionPass, ValueRangePropagationPass,
    },
    scheduling::{
        AsGlobalPass, AsSsaPass, Declared, DeobfuscationCapability,
        pass::{SsaPass, SsaPassHost},
        transaction::{Guarded, PassTransaction, Verify},
    },
    target::Target,
};

/// A registered pass paired with its assigned fallback layer number.
type LayeredPass<T, H> = (Box<dyn SsaPass<T, H>>, usize);

/// What the scheduler does when a pass fails on one method.
///
/// Either way the method is first restored to its state before the pass, so
/// the store never holds a half-edited body.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FailurePolicy {
    /// The run stops with the failure as its error.
    #[default]
    Abort,
    /// The method keeps its restored body, the failure is recorded as an
    /// [`EventKind::PassFailed`] event, and the run goes on: one method a
    /// pass cannot handle does not cost every other method its optimization.
    RollbackAndContinue,
}

/// What the scheduler checks after each pass, and what a failure costs.
#[derive(Debug, Clone, Copy)]
struct RunChecks {
    /// Verify each method after every pass.
    verify_hard: bool,
    /// What a pass failing on one method does to the run.
    on_failure: FailurePolicy,
}

/// Built-in pass-pipeline configuration.
///
/// What the pipeline runs and checks. How many rounds it and its passes may
/// take is the work budget of the host's context ([`SsaPassHost::context`]):
/// [`Budget::PipelineRounds`], [`Budget::StableRounds`] and
/// [`Budget::PhaseRounds`] for the scheduler, each pass's own rounds for its
/// fixpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineConfig {
    /// Include interprocedural dead-method elimination in scheduler presets.
    pub include_dead_method_elimination: bool,
    /// Verify each method after every pass and fail fast on invalid SSA.
    pub verify_hard: bool,
    /// What a pass failing on one method does to the run.
    pub on_method_failure: FailurePolicy,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            include_dead_method_elimination: true,
            verify_hard: false,
            on_method_failure: FailurePolicy::Abort,
        }
    }
}

/// Orchestrates SSA pass execution using capability-based scheduling.
///
/// Passes are organized into execution layers computed from their declared
/// capabilities. Each layer runs all its passes to fixpoint with
/// normalization between iterations. The entire pipeline repeats until
/// global fixpoint or the host's [`Budget::PipelineRounds`].
///
/// # Type parameters
///
/// * `T` - The host's target type.
/// * `H` - The host adapter (must implement [`SsaPassHost<T>`]).
///   Typically a single host type implements [`crate::world::World`],
///   [`crate::host::SsaStore`], [`crate::host::DirtySet`], and
///   [`SsaPassHost<T>`] together.
pub struct PassScheduler<T, H>
where
    T: Target,
    T::MethodRef: Send + Sync,
    H: SsaPassHost<T>,
{
    /// Verify each method after every pass and return an error on invalid SSA.
    verify_hard: bool,
    /// What a pass failing on one method does to the run.
    on_failure: FailurePolicy,
    /// All non-normalize passes paired with their fallback layer number.
    passes: Vec<LayeredPass<T, H>>,
    /// Normalization passes. Run between every layer's fixpoint iterations.
    normalize: Vec<Box<dyn SsaPass<T, H>>>,
    /// Phantom marker; the host type appears only in trait bounds at the
    /// `run_pipeline` entry point, but we record it here to keep the
    /// scheduler instance keyed to a specific host.
    _host: PhantomData<fn(&H)>,
}

impl<T, H> PassScheduler<T, H>
where
    T: Target,
    T::MethodRef: Send + Sync,
    SsaFunction<T>: Send,
    H: SsaPassHost<T>,
{
    /// Creates a scheduler that registers no built-in passes.
    ///
    /// Only the iteration limits and `verify_hard` flag from `config` are
    /// applied; the pass-tuning fields (e.g. `copy_iterations`,
    /// `include_dead_method_elimination`) are ignored because no built-in
    /// passes are added. Hosts that organize their own pipeline - adding
    /// every pass explicitly via [`Self::add`], [`Self::add_at_layer`], or
    /// [`Self::add_normalize`] - should start here instead of [`Self::new`]
    /// to avoid running the default pipeline alongside their own passes.
    ///
    /// # Arguments
    ///
    /// * `config` - Supplies the `verify_hard` flag and the failure policy.
    ///
    /// # Returns
    ///
    /// A scheduler with empty layered and normalize pass lists.
    #[must_use]
    pub fn empty(config: PipelineConfig) -> Self {
        Self {
            verify_hard: config.verify_hard,
            on_failure: config.on_method_failure,
            passes: Vec::new(),
            normalize: Vec::new(),
            _host: PhantomData,
        }
    }

    /// Creates a scheduler from a pipeline configuration.
    ///
    /// The default configuration registers all built-in scheduler-backed
    /// passes in a deterministic order. Value-cleanup passes are registered
    /// as normalize passes so they run between layered fixpoint iterations.
    /// Structural and value/enhancement passes are registered at their
    /// conventional fallback layers. Interprocedural dead-method elimination
    /// is included by default and can be disabled with
    /// [`PipelineConfig::include_dead_method_elimination`].
    ///
    /// Hosts still provide storage and program context through
    /// [`SsaPassHost`], [`crate::host::SsaStore`], [`crate::host::DirtySet`],
    /// and [`crate::world::World`]; they do not need to hand-roll the pass
    /// order. Hosts that supply their own complete pipeline should use
    /// [`Self::empty`] instead.
    ///
    /// # Arguments
    ///
    /// * `config` - Which passes it registers and how it checks them; the
    ///   rounds it and they take are the host's budget at
    ///   [`Self::run_pipeline`].
    ///
    /// # Returns
    ///
    /// A scheduler pre-loaded with the default normalize and layered passes.
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self
    where
        T: Send + Sync,
    {
        let mut scheduler = Self::empty(config);

        scheduler.add_normalize(Box::new(AsSsaPass(GlobalValueNumberingPass::new())));
        scheduler.add_normalize(Box::new(AsSsaPass(CopyPropagationPass)));
        scheduler.add_normalize(Box::new(AsSsaPass(DeadCodeEliminationPass)));

        if config.include_dead_method_elimination {
            scheduler.add(Box::new(AsGlobalPass(DeadMethodEliminationPass)));
        }

        scheduler.add(Box::new(AsSsaPass(ControlFlowSimplificationPass)));
        scheduler.add(Box::new(AsSsaPass(JumpThreadingPass::new())));
        scheduler.add(Box::new(AsSsaPass(LoopCanonicalizationPass::new())));
        scheduler.add(Box::new(AsSsaPass(BlockMergingPass)));
        scheduler.add(Box::new(AsSsaPass(DuplicateMergingPass::new())));
        scheduler.add(Box::new(AsSsaPass(LoopRejoiningPass::new())));
        scheduler.add(Box::new(AsSsaPass(ArmMergingPass::new())));
        scheduler.add(Box::new(AsSsaPass(MootTestPass::new())));
        scheduler.add(Box::new(AsSsaPass(AlgebraicSimplificationPass::new())));
        scheduler.add(Box::new(AsSsaPass(ReassociationPass::new())));
        scheduler.add(Box::new(AsSsaPass(StrengthReductionPass::new())));
        scheduler.add(Box::new(AsSsaPass(ValueRangePropagationPass::new())));
        // The one analyssa pass whose effect the shared vocabulary names, said
        // here rather than left to each host to remember. A host whose
        // capability enum does not carry the milestone declares nothing, which
        // is what the pass provides there.
        scheduler.add(Box::new(
            Declared::new(AsSsaPass(OpaquePredicatePass::<T>::new()))
                .providing([DeobfuscationCapability::SimplifiedPredicates]),
        ));
        scheduler.add(Box::new(AsSsaPass(LicmPass::new())));
        // Runs after the value passes: forwarding a store to a load exposes the
        // stored value to constant folding and GVN on the next iteration.
        scheduler.add(Box::new(AsSsaPass(MemoryOptimizationPass)));

        scheduler
    }

    /// Returns the number of non-normalize (layered) passes registered.
    ///
    /// # Returns
    ///
    /// The count of passes subject to capability-based layer assignment.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Returns the number of normalization passes registered.
    ///
    /// # Returns
    ///
    /// The count of passes run between layer fixpoint iterations.
    #[must_use]
    pub fn normalize_count(&self) -> usize {
        self.normalize.len()
    }

    /// Register a layered pass using its own `fallback_layer()` return
    /// value.
    ///
    /// The pass's [`fallback_layer`](SsaPass::fallback_layer) is read once at
    /// registration and stored alongside it; capability relaxation may later
    /// push it to a higher layer, but never below this seed.
    ///
    /// # Arguments
    ///
    /// * `pass` - The layered pass to register.
    pub fn add(&mut self, pass: Box<dyn SsaPass<T, H>>) {
        let layer = pass.fallback_layer();
        self.passes.push((pass, layer));
    }

    /// Register a layered pass at an explicit fallback layer, overriding
    /// the pass's own `fallback_layer()`.
    ///
    /// Useful for hosts that organize passes via a phase enum (e.g.,
    /// CIL: Structure=0, Value=1, Simplify=2, Inline=3) rather than
    /// per-pass-type defaults.
    ///
    /// # Arguments
    ///
    /// * `pass` - The layered pass to register.
    /// * `layer` - Seed fallback layer, used in place of the pass's own
    ///   [`fallback_layer`](SsaPass::fallback_layer).
    pub fn add_at_layer(&mut self, pass: Box<dyn SsaPass<T, H>>, layer: usize) {
        self.passes.push((pass, layer));
    }

    /// Register a normalization pass.
    ///
    /// Normalize passes run between every layer's fixpoint iterations
    /// and are excluded from the capability dependency graph.
    ///
    /// # Arguments
    ///
    /// * `pass` - The normalization pass to register.
    pub fn add_normalize(&mut self, pass: Box<dyn SsaPass<T, H>>) {
        self.normalize.push(pass);
    }

    /// Compute execution layer assignments from capability dependencies.
    ///
    /// Uses Bellman-Ford-style relaxation: each pass starts at its
    /// fallback layer; if pass A provides a capability that pass B
    /// requires, B is pushed to at least `layer(A) + 1`.
    ///
    /// # Returns
    ///
    /// A `Vec<usize>` where element `i` is the computed layer number
    /// for `self.passes[i]`.
    ///
    /// # Errors
    ///
    /// Returns an error if a cycle is detected in the capability
    /// dependency graph.
    fn compute_layer_assignment(&self) -> Result<Vec<usize>> {
        let n = self.passes.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Build a map from capability to provider indices.
        let mut providers: FxHashMap<T::Capability, Vec<usize>> = FxHashMap::default();
        for (i, (pass, _)) in self.passes.iter().enumerate() {
            for cap in pass.provides() {
                providers.entry(*cap).or_default().push(i);
            }
        }

        // Build a dependency graph: edge provider -> dependent.
        let mut graph: IndexedGraph<usize, ()> = IndexedGraph::with_capacity(n, n);
        for i in 0..n {
            graph.add_node(i);
        }

        let mut deps: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, (pass, _)) in self.passes.iter().enumerate() {
            for cap in pass.requires() {
                if let Some(provider_indices) = providers.get(cap) {
                    for &j in provider_indices {
                        if j != i {
                            if let Some(slot) = deps.get_mut(i) {
                                slot.push(j);
                            }
                            let _ = graph.add_edge(j, i, ());
                        }
                    }
                }
            }
        }

        // Check for cycles via topological sort.
        if graph.topological_sort().is_none() {
            if let Some(cycle) = graph.find_any_cycle() {
                let names: Vec<&str> = cycle
                    .iter()
                    .filter_map(|&i| self.passes.get(i).map(|p| p.0.name()))
                    .collect();
                return Err(Error::new(format!(
                    "Cycle detected in pass capability dependencies: {}",
                    names.join(" → ")
                )));
            }
            return Err(Error::new("Cycle detected in pass capability dependencies"));
        }

        // Bellman-Ford relaxation: push layers forward to satisfy deps.
        let mut layer: Vec<usize> = self.passes.iter().map(|(_, fallback)| *fallback).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..n {
                // `deps` is read-only here and `layer` is a separate buffer, so
                // no clone is needed to satisfy the borrow checker.
                let Some(dep_list) = deps.get(i) else {
                    continue;
                };
                for &dep in dep_list {
                    let layer_i = layer.get(i).copied().unwrap_or(0);
                    let layer_dep = layer.get(dep).copied().unwrap_or(0);
                    if layer_i <= layer_dep {
                        if let Some(slot) = layer.get_mut(i) {
                            *slot = layer_dep.saturating_add(1);
                        }
                        changed = true;
                    }
                }
            }
        }

        // Log layer assignments if any were moved from fallback.
        if !deps.iter().all(Vec::is_empty) {
            let max_layer = layer.iter().copied().max().unwrap_or(0);
            debug!(
                "Capability scheduling: {} passes across {} layers",
                n,
                max_layer.saturating_add(1)
            );
            for (i, (pass, fallback)) in self.passes.iter().enumerate() {
                let layer_i = layer.get(i).copied().unwrap_or(*fallback);
                if layer_i != *fallback {
                    debug!(
                        "  pass '{}': layer {} (moved from fallback {})",
                        pass.name(),
                        layer_i,
                        fallback
                    );
                }
            }
        }

        Ok(layer)
    }

    /// Run the complete pipeline across all layers.
    ///
    /// Seeds the dirty set with all methods, then iterates layers until
    /// convergence or the host's [`Budget::PipelineRounds`], stopping once
    /// [`Budget::StableRounds`] rounds change nothing and giving each layer
    /// [`Budget::PhaseRounds`].
    ///
    /// # Arguments
    ///
    /// * `host` - The host adapter providing method storage, dirty
    ///   tracking, and event recording.
    ///
    /// # Returns
    ///
    /// The number of outer iterations completed.
    ///
    /// # Errors
    ///
    /// Returns an error if a cycle is detected in the capability
    /// dependency graph or if any pass fails.
    pub fn run_pipeline(&mut self, host: &H) -> Result<usize> {
        // Seed the dirty set with all methods that have SSA. On the
        // first iteration every method qualifies. Hosts that pre-seed
        // their own dirty set (typically by marking newly-built methods
        // as dirty when SSA is constructed) are unaffected - `mark_dirty`
        // is idempotent.
        for method in host.iter_methods() {
            host.mark_dirty(&method);
        }

        let layer_assignment = self.compute_layer_assignment()?;

        let num_layers = layer_assignment
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m.saturating_add(1));
        let mut layer_indices: Vec<Vec<usize>> = vec![vec![]; num_layers];
        for (i, &layer) in layer_assignment.iter().enumerate() {
            if let Some(slot) = layer_indices.get_mut(layer) {
                slot.push(i);
            }
        }
        layer_indices.retain(|layer| !layer.is_empty());

        let mut stable_count: usize = 0;
        let mut iterations: usize = 0;
        let budget = *host.context().budget();
        let max_phase = budget.limit(Budget::PhaseRounds);
        let max_iterations = budget.limit(Budget::PipelineRounds);
        let stable_iterations = budget.limit(Budget::StableRounds);
        let checks = RunChecks {
            verify_hard: self.verify_hard,
            on_failure: self.on_failure,
        };

        // Whether the last round taken changed something: a run whose rounds
        // all ran out with it set stopped short of the fixpoint.
        let mut last_changed = false;
        let mut rounds = Rounds::new(Budget::PipelineRounds, max_iterations);
        while rounds.next_round() {
            let iteration = iterations;
            iterations = iteration.saturating_add(1);
            debug!("Pipeline iteration {iterations}/{max_iterations}");

            let iteration_modified: DashSet<T::MethodRef> = DashSet::new();
            let run = PipelineRun {
                host,
                iteration_modified: &iteration_modified,
                checks,
                max_phase_iterations: max_phase,
            };
            let mut iteration_changed = false;

            for layer in &layer_indices {
                if run.layer_to_fixpoint(&mut self.passes, layer, &mut self.normalize)? {
                    iteration_changed = true;
                }
            }

            // Ensure normalize runs at least once on iteration 0 even if no
            // layer pass made changes.
            if iteration == 0 && !iteration_changed && !self.normalize.is_empty() {
                iteration_changed = run.normalize_to_fixpoint(&mut self.normalize)?;
            }

            // Update dirty/stable tracking at iteration boundary.
            if iteration_changed {
                let dirty = host.dirty_snapshot();
                for m in dirty {
                    if !iteration_modified.contains(&m) {
                        host.clear_dirty_for(&m);
                    }
                }
                for entry in iteration_modified.iter() {
                    host.mark_dirty(&entry);
                }
            } else {
                let dirty = host.dirty_snapshot();
                for m in dirty {
                    host.clear_dirty_for(&m);
                }
            }

            last_changed = iteration_changed;
            if iteration_changed {
                stable_count = 0;
            } else {
                stable_count = stable_count.saturating_add(1);
                if stable_count >= stable_iterations {
                    debug!("Pipeline stable after {iterations} iterations");
                    rounds.settle();
                }
            }
        }
        if let Some(exhaustion) = rounds.exhausted().filter(|_| last_changed) {
            host.events().push(Event::exhausted(None, exhaustion));
        }

        Ok(iterations)
    }

    /// Runs `pass` on one function exactly as a scheduled run does, and
    /// reports a rejection as an error.
    ///
    /// The pass runs inside a transaction ([`PassTransaction::run_guarded`])
    /// with the scheduler's repair after it: a failed edit the pass left
    /// applied is an error, and a change is repaired as the pass's
    /// [`modification_scope`](SsaPass::modification_scope) asks. `verify`
    /// says which results are checked and how hard;
    /// `Verify::Always(VerifyLevel::Standard)` is what a run under
    /// `verify_hard` does. Whatever the pass does wrong - it returns an error,
    /// panics, leaves a rebuild that fails or SSA the verifier rejects -
    /// `ssa` is restored to the function it was before the pass.
    ///
    /// This is the scheduler's per-function step and nothing around it: it
    /// does not ask [`SsaPass::should_run`], which decides whether a scheduled
    /// run hands the pass a method at all, and it touches no host store, so
    /// it marks nothing processed and invalidates no analyses. It is the way
    /// to test a pass on a function of its own.
    ///
    /// # Arguments
    ///
    /// * `pass` - The pass to run.
    /// * `ssa` - The function, transformed in place.
    /// * `method` - The method the function is, handed to the pass.
    /// * `host` - Host adapter handed to the pass.
    /// * `verify` - Which results to verify, and at what level.
    ///
    /// # Returns
    ///
    /// Whether the pass changed the function and the change was kept.
    ///
    /// # Errors
    ///
    /// The pass failed, panicked, needed a rebuild that failed, or left SSA
    /// the verifier rejected (its errors listed), naming the pass and the
    /// method; `ssa` is then as it was before the pass.
    pub fn run_once(
        pass: &dyn SsaPass<T, H>,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
        verify: Verify,
    ) -> Result<bool> {
        let outcome = PassTransaction::new().run_guarded(ssa, verify, |ssa| {
            Self::run_and_repair(pass, ssa, method, host)
        });
        Self::settled(pass, method, outcome).map_err(Error::new)
    }

    /// How a run of `pass` on `method` ended: whether it changed the function
    /// and kept the change, or why it was rejected, naming the pass and the
    /// method, with the verifier's errors for SSA it rejected.
    fn settled(
        pass: &dyn SsaPass<T, H>,
        method: &T::MethodRef,
        outcome: Guarded<Error>,
    ) -> Result<bool, String> {
        match outcome {
            Guarded::Unchanged => Ok(false),
            Guarded::Applied => Ok(true),
            Guarded::Failed(error) => Err(format!(
                "pass '{}' failed for method {method:?}: {error}",
                pass.name()
            )),
            Guarded::Panicked(detail) => Err(format!(
                "pass '{}' panicked for method {method:?}: {detail}",
                pass.name()
            )),
            Guarded::Invalid(verifier_errors) => Err(format!(
                "pass '{}' produced invalid SSA for method {method:?}: {}",
                pass.name(),
                verifier_errors
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            )),
        }
    }

    /// Runs `pass` on one method and applies the SSA repair its change needs
    /// ([`crate::scheduling::ModificationScope::settle`]).
    ///
    /// # Arguments
    ///
    /// * `pass` - The pass to run.
    /// * `ssa` - The method's function, transformed in place.
    /// * `method` - The method being processed.
    ///
    /// # Returns
    ///
    /// Whether the function changed.
    ///
    /// # Errors
    ///
    /// The pass's error when it fails, and an error when it left an edit
    /// half applied or a graph change it did not repair.
    fn run_and_repair(
        pass: &dyn SsaPass<T, H>,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool> {
        let reported = pass.run_on_method(ssa, method, host)?;
        pass.modification_scope()
            .settle(ssa, pass.repairs_ssa(), reported)
    }
}

/// Format an event-kind delta map into a compact summary string.
///
/// Keeps only transformation kinds (skipping bookkeeping events), renders each
/// as `"<count> <description>"`, and sorts the parts so the summary is
/// deterministic across runs - it feeds a debug log line, where a stable
/// ordering makes runs comparable. Example: `"93 strings decrypted, 115
/// constants folded"`.
///
/// # Arguments
///
/// * `delta` - Per-kind event counts recorded during one pass, as produced by
///   [`EventListener::count_by_kind_since`](crate::events::EventListener::count_by_kind_since).
///
/// # Returns
///
/// A comma-separated summary, or the empty string if no transformation events
/// are present.
fn format_event_delta(delta: &FxHashMap<EventKind, usize>) -> String {
    let mut parts: Vec<String> = delta
        .iter()
        .filter(|(kind, _)| kind.is_transformation())
        .map(|(kind, count)| format!("{} {}", count, kind.description()))
        .collect();
    parts.sort();
    parts.join(", ")
}

/// One iteration of a pipeline run: the host it runs on, the methods it has
/// changed so far, what it checks after each pass, and the rounds each phase
/// may take.
struct PipelineRun<'a, T: Target, H> {
    /// The host the passes run on.
    host: &'a H,
    /// The methods any pass changed this iteration.
    iteration_modified: &'a DashSet<T::MethodRef>,
    /// What is checked after each pass, and what a failure costs.
    checks: RunChecks,
    /// The rounds one phase may run to its fixpoint.
    max_phase_iterations: usize,
}

impl<T, H> PipelineRun<'_, T, H>
where
    T: Target,
    T::MethodRef: Send + Sync,
    SsaFunction<T>: Send,
    H: SsaPassHost<T>,
{
    /// Run normalize passes to fixpoint.
    ///
    /// Repeats a full sweep of `passes` until one sweep reports no change or
    /// `max_phase_iterations` is reached. This is the convergence loop
    /// [`layer_to_fixpoint`](Self::layer_to_fixpoint) invokes after each round
    /// of structural changes, letting cleanup (DCE, GVN, copy propagation)
    /// cascade before the next layer sweep.
    ///
    /// # Arguments
    ///
    /// * `passes` - The normalize passes to drive to fixpoint.
    ///
    /// # Returns
    ///
    /// `true` if any sweep changed a method.
    ///
    /// # Errors
    ///
    /// Propagates any pass failure surfaced by
    /// [`run_passes_once`](Self::run_passes_once).
    fn normalize_to_fixpoint(&self, passes: &mut [Box<dyn SsaPass<T, H>>]) -> Result<bool> {
        let mut any_changed = false;
        let mut rounds = Rounds::new(Budget::PhaseRounds, self.max_phase_iterations);
        while rounds.next_round() {
            if self.run_passes_once(passes)? {
                any_changed = true;
            } else {
                rounds.settle();
            }
        }
        if let Some(exhaustion) = rounds.exhausted() {
            self.host.events().push(Event::exhausted(None, exhaustion));
        }
        Ok(any_changed)
    }
    /// Run all passes in a single layer to fixpoint, running normalize
    /// passes between iterations.
    ///
    /// Each round runs the layer's passes once; after every round that changed
    /// something - and once more when the layer finally settles - the normalize
    /// passes are driven to their own fixpoint, so structural edits are cleaned
    /// up and expose new opportunities before the next round. An empty layer
    /// returns immediately.
    ///
    /// # Arguments
    ///
    /// * `all_passes` - The full layered-pass vector; `layer_indices` selects
    ///   into it.
    /// * `layer_indices` - Indices of the passes belonging to this layer.
    /// * `normalize_passes` - Normalize passes interleaved between rounds.
    ///
    /// # Returns
    ///
    /// `true` if any layer pass fired during the phase.
    ///
    /// # Errors
    ///
    /// Propagates any pass or normalize failure.
    fn layer_to_fixpoint(
        &self,
        all_passes: &mut [LayeredPass<T, H>],
        layer_indices: &[usize],
        normalize_passes: &mut [Box<dyn SsaPass<T, H>>],
    ) -> Result<bool> {
        if layer_indices.is_empty() {
            return Ok(false);
        }

        let mut phase_changed = false;

        let mut rounds = Rounds::new(Budget::PhaseRounds, self.max_phase_iterations);
        while rounds.next_round() {
            let pass_changed = self.run_layer_passes_once(all_passes, layer_indices)?;

            if !pass_changed {
                if phase_changed && !normalize_passes.is_empty() {
                    self.normalize_to_fixpoint(normalize_passes)?;
                }
                rounds.settle();
                continue;
            }

            phase_changed = true;

            if !normalize_passes.is_empty() {
                self.normalize_to_fixpoint(normalize_passes)?;
            }
        }
        if let Some(exhaustion) = rounds.exhausted() {
            self.host.events().push(Event::exhausted(None, exhaustion));
        }

        Ok(phase_changed)
    }
    /// Run all normalize passes once across all eligible methods.
    ///
    /// One full sweep: [`initialize`](SsaPass::initialize) every pass, run the
    /// global passes, then dispatch each per-method pass over either all
    /// methods or just the dirty set depending on
    /// [`requires_full_scan`](SsaPass::requires_full_scan), and finally
    /// [`finalize`](SsaPass::finalize) every pass. The two-phase split (globals
    /// before per-method) keeps whole-program passes ordered ahead of the
    /// parallel per-method work within the sweep.
    ///
    /// # Arguments
    ///
    /// * `passes` - The normalize passes to run one sweep of.
    ///
    /// # Returns
    ///
    /// `true` if any pass changed a method during the sweep.
    ///
    /// # Errors
    ///
    /// Returns the first pass failure from initialize, run, repair, or finalize.
    fn run_passes_once(&self, passes: &mut [Box<dyn SsaPass<T, H>>]) -> Result<bool> {
        for pass in passes.iter_mut() {
            pass.initialize(self.host)?;
        }

        let (all_methods, dirty_methods) = self.method_orders();
        let any_changed = AtomicBool::new(false);

        for pass in passes.iter() {
            if pass.is_global() && pass.run_global(self.host)? {
                any_changed.store(true, Ordering::Relaxed);
            }
        }

        for pass in passes.iter() {
            if pass.is_global() {
                continue;
            }
            let methods = if pass.requires_full_scan() {
                &all_methods
            } else {
                &dirty_methods
            };
            self.run_single_pass(pass.as_ref(), methods, &any_changed)?;
        }

        for pass in passes.iter_mut() {
            pass.finalize(self.host)?;
        }

        Ok(any_changed.load(Ordering::Relaxed))
    }
    /// Run all passes in a specific layer once across eligible methods.
    ///
    /// The layered analogue of [`run_passes_once`](Self::run_passes_once):
    /// initialize, run globals, dispatch per-method work, then finalize - but
    /// the passes are reached by index into the shared `all_passes` vector
    /// rather than owned directly, so each lookup is bounds-checked and an
    /// out-of-range index becomes a scheduler error instead of a panic.
    ///
    /// # Arguments
    ///
    /// * `all_passes` - The full layered-pass vector.
    /// * `indices` - Indices of the passes to run this round.
    ///
    /// # Returns
    ///
    /// `true` if any pass changed a method during the round.
    ///
    /// # Errors
    ///
    /// Returns an error on an out-of-bounds index or any pass failure.
    fn run_layer_passes_once(
        &self,
        all_passes: &mut [LayeredPass<T, H>],
        indices: &[usize],
    ) -> Result<bool> {
        for &idx in indices {
            let pass_entry = all_passes
                .get_mut(idx)
                .ok_or_else(|| Error::new(format!("scheduler: pass index {idx} out of bounds")))?;
            pass_entry.0.initialize(self.host)?;
        }

        let (all_methods, dirty_methods) = self.method_orders();
        let any_changed = AtomicBool::new(false);

        for &idx in indices {
            let pass_entry = all_passes
                .get(idx)
                .ok_or_else(|| Error::new(format!("scheduler: pass index {idx} out of bounds")))?;
            let pass = &pass_entry.0;
            if pass.is_global() && pass.run_global(self.host)? {
                any_changed.store(true, Ordering::Relaxed);
            }
        }

        for &idx in indices {
            let pass_entry = all_passes
                .get(idx)
                .ok_or_else(|| Error::new(format!("scheduler: pass index {idx} out of bounds")))?;
            let pass = &pass_entry.0;
            if pass.is_global() {
                continue;
            }
            let methods = if pass.requires_full_scan() {
                &all_methods
            } else {
                &dirty_methods
            };
            self.run_single_pass(pass.as_ref(), methods, &any_changed)?;
        }

        for &idx in indices {
            let pass_entry = all_passes
                .get_mut(idx)
                .ok_or_else(|| Error::new(format!("scheduler: pass index {idx} out of bounds")))?;
            pass_entry.0.finalize(self.host)?;
        }

        Ok(any_changed.load(Ordering::Relaxed))
    }
    /// Execute a single pass across all eligible methods in parallel.
    ///
    /// For each method (via rayon): takes - or, for a peer-reading pass,
    /// clones - the SSA from the store and runs the pass inside a
    /// [`PassTransaction`], one reusable snapshot buffer per worker. Inside the
    /// transaction the pass runs, and when it changed the function (by its own
    /// report, or by the function's edit flag when it claims otherwise) the
    /// scheduler applies the SSA repair its
    /// [`modification_scope`](SsaPass::modification_scope) calls for, unless
    /// the pass [repairs its own SSA](SsaPass::repairs_ssa) and says it
    /// changed. Under `verify_hard` the result is then verified, whether or not
    /// the pass reported a change.
    ///
    /// A pass error, a caught panic, a failed rebuild, or a verifier rejection
    /// restores the function to its state before the pass, and the restored
    /// function is what goes back into the store - the store never holds a
    /// half-edited body, whether the SSA was taken or cloned. A kept change
    /// marks the method processed, records it as modified this iteration, and
    /// asks the host to [invalidate its analyses](SsaPassHost::invalidate_analyses).
    /// Per-method failures are collected and the first is surfaced after the
    /// parallel region joins.
    ///
    /// # Arguments
    ///
    /// * `pass` - The pass to execute against each method.
    /// * `methods` - The methods to dispatch across, already filtered for
    ///   eligibility by the caller.
    /// * `any_changed` - Set to `true` if any method was changed.
    ///
    /// # Errors
    ///
    /// Returns the first collected per-method failure - a pass error, a caught
    /// panic, a failed SSA rebuild, or (under `verify_hard`) a verifier
    /// rejection.
    fn run_single_pass(
        &self,
        pass: &dyn SsaPass<T, H>,
        methods: &[T::MethodRef],
        any_changed: &AtomicBool,
    ) -> Result<()> {
        let event_snapshot = self.host.events().recorded_count();
        let pass_change_count = AtomicUsize::new(0);
        let verify = if self.checks.verify_hard {
            Verify::Always(VerifyLevel::Standard)
        } else {
            Verify::Never
        };

        // Passes that read other methods' SSA need peer SSAs to remain
        // visible during parallel execution. Clone the SSA before
        // processing so the original stays readable.
        let clone_for_visibility = pass.reads_peer_ssa();

        // Nothing goes back into the store until every method has run. A peer
        // reader then sees each peer as it was when the pass began, whatever
        // order the workers happen to finish in; writing results back as they
        // came in let a method inline a callee before or after that callee's
        // own rewrite depending on thread timing, so the same input produced
        // different output under different machine load. The bookkeeping that
        // other methods' passes consult (processed marks, analysis
        // invalidation) waits for the same reason, and the results are
        // committed in method order.
        let outcomes: Vec<_> = methods
            .par_iter()
            .map_init_carried(PassTransaction::new, |transaction, method| {
                if !pass.should_run(method, self.host) {
                    return None;
                }

                let taken = if clone_for_visibility {
                    self.host.clone_ssa(method)
                } else {
                    self.host.take_ssa(method)
                };
                let mut ssa = taken?;

                let outcome = transaction.run_guarded(&mut ssa, verify, |ssa| {
                    PassScheduler::<T, H>::run_and_repair(pass, ssa, method, self.host)
                });
                Some((method, ssa, outcome))
            })
            .collect();

        let mut failures = Vec::new();
        for (method, ssa, outcome) in outcomes.into_iter().flatten() {
            // Restored or rewritten, the function goes back: the SSA was
            // taken out of the store (or, for a peer reader, the store copy
            // is overwritten with an identical restore).
            self.host.insert_ssa(method.clone(), ssa);

            match PassScheduler::<T, H>::settled(pass, method, outcome) {
                Ok(false) => {}
                Ok(true) => {
                    any_changed.store(true, Ordering::Relaxed);
                    pass_change_count.fetch_add(1, Ordering::Relaxed);
                    self.host.mark_processed(method);
                    self.host.invalidate_analyses(method);
                    self.iteration_modified.insert(method.clone());
                }
                Err(failure) => {
                    // Rolled back, but the pass may have computed facts about
                    // the function while it was mid-edit.
                    self.host.invalidate_analyses(method);
                    failures.push((method, failure));
                }
            }
        }

        match self.checks.on_failure {
            FailurePolicy::Abort => {
                if let Some((_, first)) = failures.first() {
                    return Err(Error::new(first.clone()));
                }
            }
            FailurePolicy::RollbackAndContinue => {
                for (method, failure) in failures {
                    self.host.events().push(Event {
                        kind: EventKind::PassFailed,
                        method: Some(method.clone()),
                        location: None,
                        message: failure,
                        pass: Some(pass.name().to_string()),
                    });
                }
            }
        }

        if log_enabled!(Level::Debug) {
            let count = pass_change_count.load(Ordering::Relaxed);
            if count > 0 {
                let event_delta = self.host.events().count_by_kind_since(event_snapshot);
                if event_delta.is_empty() {
                    debug!("  pass '{}' changed {} methods", pass.name(), count);
                } else {
                    let summary = format_event_delta(&event_delta);
                    if summary.is_empty() {
                        debug!("  pass '{}' changed {} methods", pass.name(), count);
                    } else {
                        debug!(
                            "  pass '{}' changed {} methods ({})",
                            pass.name(),
                            count,
                            summary
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Determine the two method processing orders in one pass.
    ///
    /// Returns `(all_methods, dirty_methods)`. Both derive from the same
    /// traversal order, so the call graph behind
    /// [`World::methods_reverse_topological`] is built **once** per pass
    /// batch rather than once per list - and the batch itself sits inside the
    /// scheduler's fixpoint, so a per-list rebuild is paid on every iteration.
    ///
    /// Ordering follows reverse topological order if the host supplies one, or
    /// falls back to `host.iter_methods()`. Both lists are filtered to methods
    /// the host still holds SSA for, and the dirty list is derived by
    /// intersecting against a `HashSet` so the filter stays O(methods).
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// `(all_methods, dirty_methods)`: every method with SSA, and the subset of
    /// those currently marked dirty, both in the same order.
    ///
    /// [`World::methods_reverse_topological`]: crate::world::World::methods_reverse_topological
    fn method_orders(&self) -> (Vec<T::MethodRef>, Vec<T::MethodRef>) {
        // Components arrive callee-first and grouped by mutual recursion.
        // Flattening discards the grouping, which is correct here: this order
        // only decides *membership and sequence* for a per-method pass, and
        // `run_single_pass` dispatches those in parallel anyway. An analysis
        // that genuinely needs callee-before-caller must be a global pass and
        // drive `interproc::solve`, which consumes the components ungrouped.
        let topo: Vec<T::MethodRef> = self
            .host
            .methods_reverse_topological()
            .into_iter()
            .flatten()
            .collect();
        let order: Vec<_> = if topo.is_empty() {
            self.host.iter_methods()
        } else {
            topo
        };

        let all: Vec<T::MethodRef> = order
            .into_iter()
            .filter(|m| self.host.contains(m))
            .collect();

        // Build the dirty set as a HashSet so membership is O(1); a `Vec`
        // made the per-method filter below O(methods * dirty).
        let dirty_set: FxHashSet<T::MethodRef> = self.host.dirty_snapshot().into_iter().collect();
        let dirty: Vec<T::MethodRef> = all
            .iter()
            .filter(|m| dirty_set.contains(*m))
            .cloned()
            .collect();

        (all, dirty)
    }
}
