//! [`SsaPass`] trait and supporting types ([`ModificationScope`],
//! [`SsaPassHost`]) for declaring and implementing SSA transformation
//! passes.
//!
//! A pass is a target-agnostic transformation that runs on a single
//! method's SSA (or globally across all methods) and reports whether
//! it made changes. The scheduler in [`crate::scheduling::scheduler`] orchestrates
//! execution: layer assignment from capability dependencies, per-layer
//! fixpoint, parallel per-method dispatch via rayon, and
//! modification-scope-driven repair after each pass.

use crate::{
    analysis::{bindingtime::BindingOracle, context::AnalysisContext},
    error::{Error, Result},
    events::EventListener,
    host::{DirtySet, SsaStore},
    ir::function::SsaFunction,
    scheduling::env::PassEnv,
    target::Target,
    world::World,
};

/// Combined host surface that the scheduler exposes to passes.
///
/// Scheduler hosts implement [`World<T>`], [`SsaStore<T>`], and
/// [`DirtySet<T>`] separately, plus this trait to surface the event
/// sink and pointer size. The scheduler hands passes a
/// `&dyn SsaPassHost<T>` so they can read the call graph, look up peer
/// methods' SSA, mark methods dirty, and record events without knowing
/// the concrete host type.
pub trait SsaPassHost<T: Target>: World<T> + SsaStore<T> + DirtySet<T> + Sync {
    /// Returns the event sink for transformation events recorded by passes.
    ///
    /// A trait object rather than a concrete [`EventLog<T>`], so a host can
    /// choose its sink. [`EventLog<T>`] is an append-only `boxcar::Vec` with no
    /// bound and no way to drain it, so naming it concretely would force a host
    /// that never consumes events to retain every one for the process lifetime.
    /// [`NullListener`] discards them instead.
    ///
    /// [`EventLog<T>`]: crate::events::EventLog
    /// [`NullListener`]: crate::events::NullListener
    fn events(&self) -> &dyn EventListener<T>;

    /// Returns what the host's passes and their analyses run under
    /// ([`AnalysisContext`]): the machine the program runs on, what the host
    /// knows of its image, the work budget, what the functions it calls do,
    /// and the interrupt a long pass asks.
    ///
    /// The machine has no default: a host that stated none would have its
    /// passes fold by a machine it is not. Everything else a host does not
    /// know it leaves out ([`AnalysisContext::new`]): no image is memory
    /// unknown, no models is every call opaque.
    fn context(&self) -> AnalysisContext<'_>;

    /// Returns what the host knows about binding time that one function does
    /// not say: what its calls do, and what a location holds on entry.
    ///
    /// The default knows nothing, which leaves every call opaque.
    fn binding_oracle(&self) -> &dyn BindingOracle<T> {
        &()
    }

    /// Returns this host as the per-function surface, for one method.
    ///
    /// What a [`FunctionPass`](crate::scheduling::FunctionPass) is handed. A host with a program to offer builds
    /// it from itself; a caller with only a function builds one directly with
    /// [`PassEnv::new`].
    ///
    /// # Arguments
    ///
    /// * `method` - The method the environment describes.
    fn env<'a>(&'a self, method: &'a T::MethodRef) -> PassEnv<'a, T>
    where
        Self: Sized,
    {
        PassEnv::from_host(method, self)
    }

    /// Discards any cached analyses of `method`, whose SSA a pass just changed.
    ///
    /// The scheduler calls this after every per-method pass run that changed
    /// `method` and kept the change, and after every run it rolled back (the
    /// pass may have read the function mid-edit), before the next pass runs -
    /// so a host that caches analyses per method (an
    /// [`AnalysisCache`](crate::scheduling::AnalysisCache), say) never serves a
    /// later pass in the same iteration facts about the function as it was.
    /// [`DirtySet::mark_dirty`] is not that signal: the scheduler marks dirty
    /// only at iteration boundaries. A global pass reports no per-method
    /// change, so one that rewrites a method's SSA calls this itself.
    ///
    /// The default does nothing, which is correct for a host that caches no
    /// analyses.
    ///
    /// # Arguments
    ///
    /// * `method` - The method whose SSA changed.
    fn invalidate_analyses(&self, _method: &T::MethodRef) {}
}

/// Describes the extent of modifications a pass makes to the SSA function.
///
/// The scheduler uses this to select the repair a pass's change needs
/// ([`Self::settle`]). Passes should declare the **tightest** scope that
/// covers all their modifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModificationScope {
    /// The pass only replaces uses of variables with other existing
    /// variables.
    ///
    /// SSA invariants are preserved automatically - no repair is needed
    /// after the pass. Example: forwarding redundant uses to earlier
    /// definitions without removing the redundant instructions.
    UsesOnly,

    /// The pass replaces or removes instructions but does not change
    /// the CFG.
    ///
    /// After this scope, the scheduler runs a lightweight repair to strip
    /// `Nop`s, recompute variable metadata, and eliminate trivial phis.
    /// No dominator or dominance-frontier recomputation is needed.
    /// Examples: constant propagation, copy propagation, global value
    /// numbering, algebraic simplification, strength reduction.
    InstructionsOnly,

    /// The pass may add or remove blocks, change predecessors or
    /// successors, or otherwise modify the control-flow graph.
    ///
    /// Such a pass repairs the SSA it leaves itself, and says so
    /// ([`SsaPass::repairs_ssa`]): an edge it adds carries values only it
    /// knows, which it states in its own edit sessions and declares repaired
    /// ([`SsaEditor::declare_repaired`]). One that leaves the repair to the
    /// scheduler is refused when it changes a function. Examples: control
    /// flow unflattening, jump threading, block merging, loop
    /// canonicalization, dead-code elimination (which clears unreachable
    /// blocks), inlining.
    ///
    /// [`SsaEditor::declare_repaired`]: crate::ir::function::SsaEditor::declare_repaired
    CfgModifying,
}

impl ModificationScope {
    /// Applies the repair a pass's run calls for, and returns whether the
    /// run changed the function.
    ///
    /// The change is the pass's own report or the function's edit flag,
    /// whichever says "changed": a pass that edited through a checked session
    /// and then reported `Ok(false)` still gets repaired. The repair is
    /// skipped only for a pass that repairs its own SSA *and* reported the
    /// change - a pass that claims no change over an edited function cannot
    /// be trusted to have finished its boundary repair. A graph change is
    /// never repaired here: its edit sessions repaired their edges, which
    /// leaves an under-reporting pass only the instruction-level clean-up,
    /// and a pass that does not repair its own SSA is refused.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function the pass ran on.
    /// * `repairs_ssa` - Whether the pass repairs its own SSA.
    /// * `reported` - Whether the pass reported a change.
    ///
    /// # Returns
    ///
    /// Whether the function changed.
    ///
    /// # Errors
    ///
    /// Returns an error when an edit session of the pass failed and left its
    /// edits applied, and when a pass that does not repair its own SSA changed
    /// the control-flow graph. Either way only the caller's snapshot of the
    /// function from before the pass is known to be sound.
    pub fn settle<T: Target>(
        self,
        ssa: &mut SsaFunction<T>,
        repairs_ssa: bool,
        reported: bool,
    ) -> Result<bool> {
        if let Some(failure) = ssa.take_edit_failure() {
            return Err(Error::new(format!("left a failed edit applied: {failure}")));
        }
        // Taking the flag here is safe: the returned `changed` carries it to
        // the caller, which only needs one of the two to say "changed".
        let changed = ssa.take_edit_dirty() || reported;
        if changed && !(reported && repairs_ssa) {
            match self {
                Self::UsesOnly | Self::InstructionsOnly => ssa.repair_ssa(),
                Self::CfgModifying if repairs_ssa => ssa.repair_ssa(),
                Self::CfgModifying => {
                    return Err(Error::new(
                        "changed the control-flow graph and did not repair the SSA it left",
                    ));
                }
            }
        }
        Ok(changed)
    }
}

/// An SSA transformation pass.
///
/// Generic over both the target `T` and the host adapter `H`. The host
/// type must implement [`SsaPassHost<T>`] and is fixed by the
/// [`PassScheduler`](crate::scheduling::PassScheduler) instance - all
/// passes registered with that scheduler see the same host type.
///
/// # Concurrency
///
/// Passes must be `Send + Sync` so the scheduler can run them in
/// parallel across methods via rayon. Mutation of pass state happens
/// only in [`initialize`](Self::initialize) /
/// [`finalize`](Self::finalize) (single-threaded boundary calls);
/// per-method work uses `&self` and must rely on interior mutability
/// for any cross-method state.
///
/// # Implementing a pass
///
/// Most analyssa-side passes have a pure-function body in `crate::passes`
/// and a one-line trait impl in the scheduling sub-module:
///
/// ```text
/// impl<T: Target, H: SsaPassHost<T>> SsaPass<T, H> for MyPass {
///     fn name(&self) -> &'static str { "my-pass" }
///     fn run_on_method(&self, ssa: &mut SsaFunction<T>, method: &T::MethodRef, host: &H) -> Result<bool> {
///         Ok(passes::my_pass::run(ssa, method, host.events()))
///     }
/// }
/// ```
///
/// Hosts that ship target-specific passes (e.g., CIL inlining) write
/// impls bounded on a host extension trait, giving the impl access to
/// host-specific methods while still being storable in a
/// `Box<dyn SsaPass<CilTarget, ConcreteCilHost>>`.
pub trait SsaPass<T: Target, H: SsaPassHost<T>>: Send + Sync {
    /// Returns a unique short name for logging and debugging.
    ///
    /// The scheduler uses it verbatim in debug output and in the
    /// capability-cycle error message, so it should be stable and identify the
    /// pass unambiguously among its peers.
    ///
    /// # Returns
    ///
    /// A `'static` identifier for the pass.
    fn name(&self) -> &'static str;

    /// Returns a human-readable description of the pass.
    ///
    /// Purely informational - the scheduler never branches on it. The default
    /// is a placeholder; override it to explain what the pass does for hosts
    /// that surface pass metadata.
    ///
    /// # Returns
    ///
    /// A `'static` description, or `"No description available"` if not
    /// overridden.
    fn description(&self) -> &'static str {
        "No description available"
    }

    /// Determines whether this pass should run on a specific method.
    ///
    /// Called before [`run_on_method`](Self::run_on_method). Override to
    /// skip methods that do not need this pass (e.g., already processed,
    /// too small to be interesting). Default implementation returns `true`.
    fn should_run(&self, _method: &T::MethodRef, _host: &H) -> bool {
        true
    }

    /// Run the pass on a single method's SSA.
    ///
    /// Events are recorded directly to `host.events()`.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to transform in place.
    /// * `method` - Opaque reference identifying the method being processed.
    /// * `host` - The host adapter for reading the call graph, looking up
    ///   peer SSA, marking methods dirty, and recording events.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if any changes were made, `Ok(false)` if no changes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the pass fails to process the method (e.g.,
    /// malformed SSA, unexpected opcode).
    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<T>,
        method: &T::MethodRef,
        host: &H,
    ) -> Result<bool>;

    /// Run the pass on the entire program (interprocedural passes).
    ///
    /// Override for passes that need to see all methods simultaneously,
    /// such as dead-method elimination or whole-program constant
    /// propagation.
    ///
    /// # Arguments
    ///
    /// * `host` - The host adapter providing program-wide state.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if any changes were made, `Ok(false)` if no changes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the pass fails.
    fn run_global(&self, _host: &H) -> Result<bool> {
        Ok(false)
    }

    /// Returns whether this pass operates globally (across all methods).
    ///
    /// When `true`, the scheduler invokes [`run_global`](Self::run_global)
    /// instead of iterating per-method with
    /// [`run_on_method`](Self::run_on_method).
    fn is_global(&self) -> bool {
        false
    }

    /// Called once before the pass runs in a phase.
    ///
    /// Use this to initialize pass-specific state or caches. The default
    /// implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    fn initialize(&mut self, _host: &H) -> Result<()> {
        Ok(())
    }

    /// Called once after the pass completes in a phase.
    ///
    /// Use this to clean up pass-specific state. The default
    /// implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails.
    fn finalize(&mut self, _host: &H) -> Result<()> {
        Ok(())
    }

    /// Declares the extent of modifications this pass makes.
    ///
    /// It has no default: the scope decides the repair the scheduler applies
    /// after the pass, and a pass that changes the control-flow graph must
    /// repair its own SSA ([`ModificationScope::CfgModifying`]).
    fn modification_scope(&self) -> ModificationScope;

    /// Returns whether the pass performs its own SSA boundary repair.
    ///
    /// Passes that wrap their mutation body in
    /// [`SsaFunction::edit`](crate::ir::function::SsaFunction::edit) should
    /// return `true` so the scheduler does not run a second repair step on
    /// the same method. The pass must still report an accurate
    /// [`modification_scope`](Self::modification_scope) for ordering,
    /// documentation, and hosts that inspect pass metadata.
    fn repairs_ssa(&self) -> bool {
        false
    }

    /// Returns the capabilities this pass provides after execution.
    ///
    /// The scheduler ensures that consumers of a capability are placed in
    /// later layers than providers.
    fn provides(&self) -> &[T::Capability] {
        &[]
    }

    /// Returns the capabilities this pass requires before it can run.
    ///
    /// Unsatisfied requirements (no provider registered) fall back to
    /// phase-based ordering rather than blocking the pass.
    fn requires(&self) -> &[T::Capability] {
        &[]
    }

    /// Returns whether this pass reads other methods' SSA during
    /// [`run_on_method`](Self::run_on_method).
    ///
    /// During parallel execution, the scheduler must keep each method's
    /// SSA visible in the store so other threads can read it. When this
    /// returns `true`, the scheduler clones the SSA before processing
    /// (instead of removing it) so concurrent visibility is preserved.
    /// Passes that only modify their own method should keep the default
    /// `false` to avoid the clone overhead.
    fn reads_peer_ssa(&self) -> bool {
        false
    }

    /// Returns whether this pass requires a full scan of all methods
    /// every iteration.
    ///
    /// If `true`, the scheduler dispatches to every method with SSA,
    /// regardless of dirty-tracking state. If `false` (default), only
    /// dirty methods are processed.
    fn requires_full_scan(&self) -> bool {
        false
    }

    /// Returns the fallback execution layer for this pass.
    ///
    /// Used when capability dependencies do not constrain the pass's
    /// position. Hosts assign meaningful numbers (e.g., CIL uses
    /// Structure=0, Value=1, Simplify=2, Inline=3 by convention).
    ///
    /// Normalize-style passes (DCE, GVN, constant propagation) typically
    /// use layer 0 and are registered via the scheduler's `add_normalize`
    /// API, which takes them out of the layered execution and runs them
    /// between every layer's fixpoint iterations.
    fn fallback_layer(&self) -> usize {
        0
    }
}
