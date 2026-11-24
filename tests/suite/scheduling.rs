//! Pass scheduler configuration tests.

use std::{
    collections::{BTreeMap, BTreeSet},
    process,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use analyssa::{
    Error, PipelineConfig, Result,
    analysis::{
        AnalysisContext, Machine, VerifyLevel,
        budget::{Budget, WorkBudget},
    },
    events::{EventKind, EventListener, EventLog, NullListener},
    host::{DirtySet, SsaStore},
    interproc::CallGraph,
    ir::{
        ConstValue, DefaultNan, FloatRules, NanEncoding,
        function::{SsaEditOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::SsaOp,
        variable::SsaVarId,
    },
    passes::{
        AlgebraicSimplificationPass, BlockMergingPass, ControlFlowSimplificationPass,
        CopyPropagationPass, DeadCodeEliminationPass, DeadMethodEliminationPass,
        GlobalValueNumberingPass, JumpThreadingPass, LicmPass, LoopCanonicalizationPass,
        MemoryOptimizationPass, OpaquePredicatePass, ReassociationPass, StrengthReductionPass,
        ValueRangePropagationPass,
    },
    scheduling::{
        AsGlobalPass, AsSsaPass, FailurePolicy, FunctionPass, ModificationScope, ModulePass,
        PassEnv, PassScheduler, SsaPass, SsaPassHost, Verify,
    },
    testing::{self, MACHINE, MockTarget},
    world::World,
};

use crate::common::assert_valid_full;

/// The crate's own passes are [`FunctionPass`]es: they rewrite the function in
/// front of them and name no host type. These read the declarations the
/// scheduler orders and repairs by.
/// A whole-program pass names a host type, because reading the program is what
/// it is for.
fn whole_program_pass_name(pass: &dyn ModulePass<MockTarget>) -> &'static str {
    pass.name()
}

fn pass_name(pass: &dyn FunctionPass<MockTarget>) -> &'static str {
    pass.name()
}

fn pass_scope(pass: &dyn FunctionPass<MockTarget>) -> ModificationScope {
    pass.modification_scope()
}

fn pass_repairs_ssa(pass: &dyn FunctionPass<MockTarget>) -> bool {
    pass.repairs_ssa()
}

fn pass_is_global(pass: &dyn SsaPass<MockTarget, MockHost>) -> bool {
    pass.is_global()
}

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| process::abort())
}

fn result_or_abort<T>(result: Result<T>) -> T {
    result.unwrap_or_else(|_| process::abort())
}

fn err_or_abort<T>(result: Result<T>) -> Error {
    match result {
        Ok(_) => process::abort(),
        Err(error) => error,
    }
}

#[derive(Default)]
struct MockHost {
    ssa: Mutex<BTreeMap<u32, SsaFunction<MockTarget>>>,
    dirty: Mutex<BTreeSet<u32>>,
    processed: Mutex<BTreeSet<u32>>,
    events: EventLog<MockTarget>,
    /// Counts `methods_reverse_topological` calls. That default implementation
    /// builds the whole call graph, so a scheduler that asked for it once per
    /// method list would pay for it twice per pass batch, inside its own
    /// fixpoint.
    topo_calls: AtomicUsize,
    /// Every `invalidate_analyses` call, in order.
    invalidated: Mutex<Vec<u32>>,
    /// The work budget the host hands the scheduler and its passes.
    budget: WorkBudget,
}

impl MockHost {
    /// A host whose pipelines run at most `pipeline` rounds, and `phase` a
    /// layer where given.
    fn with_rounds(pipeline: usize, phase: Option<usize>) -> Self {
        let budget = WorkBudget::DEFAULT.with_limit(Budget::PipelineRounds, pipeline);
        Self {
            budget: phase.map_or(budget, |phase| {
                budget.with_limit(Budget::PhaseRounds, phase)
            }),
            ..Self::default()
        }
    }
}

impl World<MockTarget> for MockHost {
    fn methods_reverse_topological(&self) -> Vec<Vec<u32>> {
        self.topo_calls.fetch_add(1, Ordering::Relaxed);
        CallGraph::from_world(self).components_callee_first()
    }

    fn all_methods(&self) -> Vec<u32> {
        self.iter_methods()
    }

    fn entry_points(&self) -> Vec<u32> {
        self.iter_methods()
    }

    fn callees(&self, _method: &u32) -> Vec<u32> {
        Vec::new()
    }

    fn is_dead(&self, _method: &u32) -> bool {
        false
    }

    fn mark_dead(&self, _method: &u32) {}
}

impl SsaStore<MockTarget> for MockHost {
    fn contains(&self, method: &u32) -> bool {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .contains_key(method)
    }

    fn take_ssa(&self, method: &u32) -> Option<SsaFunction<MockTarget>> {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .remove(method)
    }

    fn insert_ssa(&self, method: u32, ssa: SsaFunction<MockTarget>) {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .insert(method, ssa);
    }

    fn clone_ssa(&self, method: &u32) -> Option<SsaFunction<MockTarget>> {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .get(method)
            .cloned()
    }

    fn iter_methods(&self) -> Vec<u32> {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .keys()
            .copied()
            .collect()
    }
}

impl DirtySet<MockTarget> for MockHost {
    fn mark_dirty(&self, method: &u32) {
        self.dirty
            .lock()
            .unwrap_or_else(|_| process::abort())
            .insert(*method);
    }

    fn is_dirty(&self, method: &u32) -> bool {
        self.dirty
            .lock()
            .unwrap_or_else(|_| process::abort())
            .contains(method)
    }

    fn dirty_snapshot(&self) -> Vec<u32> {
        self.dirty
            .lock()
            .unwrap_or_else(|_| process::abort())
            .iter()
            .copied()
            .collect()
    }

    fn clear_dirty_for(&self, method: &u32) {
        self.dirty
            .lock()
            .unwrap_or_else(|_| process::abort())
            .remove(method);
    }

    fn mark_processed(&self, method: &u32) {
        self.processed
            .lock()
            .unwrap_or_else(|_| process::abort())
            .insert(*method);
    }

    fn is_processed(&self, method: &u32) -> bool {
        self.processed
            .lock()
            .unwrap_or_else(|_| process::abort())
            .contains(method)
    }
}

impl SsaPassHost<MockTarget> for MockHost {
    fn events(&self) -> &dyn EventListener<MockTarget> {
        &self.events
    }

    fn context(&self) -> AnalysisContext<'_> {
        AnalysisContext::new(MACHINE).with_budget(self.budget)
    }

    fn invalidate_analyses(&self, method: &u32) {
        self.invalidated
            .lock()
            .unwrap_or_else(|_| process::abort())
            .push(*method);
    }
}

struct BreakingPass;

impl SsaPass<MockTarget, MockHost> for BreakingPass {
    fn name(&self) -> &'static str {
        "breaking"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        ssa.replace_instruction_op(0, 1, SsaOp::returns(Some(SsaVarId::from_index(99))));
        Ok(true)
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }
}

struct FailingPass;

impl SsaPass<MockTarget, MockHost> for FailingPass {
    fn name(&self) -> &'static str {
        "failing"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }

    fn run_on_method(
        &self,
        _ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        Err(Error::new("pass failed"))
    }
}

#[test]
fn default_pipeline_config_registers_all_builtin_passes() {
    let config = PipelineConfig::default();
    let scheduler = PassScheduler::<MockTarget, MockHost>::new(config);

    assert_eq!(scheduler.normalize_count(), 3);
    assert_eq!(scheduler.pass_count(), 16);

    let without_global = PipelineConfig {
        include_dead_method_elimination: false,
        ..PipelineConfig::default()
    };
    let scheduler = PassScheduler::<MockTarget, MockHost>::new(without_global);

    assert_eq!(scheduler.normalize_count(), 3);
    assert_eq!(scheduler.pass_count(), 15);
}

#[test]
fn verify_hard_reports_invalid_pass_output_and_rolls_back() {
    let host = MockHost::with_rounds(1, None);
    let method = 1u32;
    let original = testing::const_i32_return(7);
    host.insert_ssa(method, original.clone());

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        verify_hard: true,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::new(config);
    scheduler.add_at_layer(Box::new(BreakingPass), 0);

    let error = err_or_abort(scheduler.run_pipeline(&host));

    assert!(error.to_string().contains("breaking"));
    assert!(error.to_string().contains("invalid SSA"));
    let stored = some_or_abort(host.clone_ssa(&method));
    assert_eq!(format!("{stored}"), format!("{original}"));
}

#[test]
fn scheduler_propagates_pass_errors() {
    let host = MockHost::with_rounds(1, None);
    let method = 1u32;
    host.insert_ssa(method, testing::const_i32_return(7));

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::new(config);
    scheduler.add_at_layer(Box::new(FailingPass), 0);

    let error = err_or_abort(scheduler.run_pipeline(&host));

    assert!(error.to_string().contains("failing"));
    assert!(error.to_string().contains("pass failed"));
}

/// Folds the diamond's branch by writing the jump straight into the function,
/// leaving the join's phi an operand for an edge that is gone.
struct UnrepairedFoldPass;

impl SsaPass<MockTarget, MockHost> for UnrepairedFoldPass {
    fn name(&self) -> &'static str {
        "unrepaired-fold"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        ssa.replace_instruction_op(0, 1, SsaOp::Jump { target: 1 });
        Ok(true)
    }
}

/// **A pass that changes the graph and leaves the repair to the scheduler is
/// refused.** The values its edges carry are the pass's to state; the
/// scheduler has nothing to reconstruct them from, so the method keeps its body
/// from before the pass.
#[test]
fn a_graph_change_left_unrepaired_is_refused() {
    let host = MockHost::with_rounds(1, None);
    let method = 1u32;
    let original = testing::diamond_phi_fixture();
    host.insert_ssa(method, original.clone());

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::empty(config);
    scheduler.add_at_layer(Box::new(UnrepairedFoldPass), 0);

    let error = err_or_abort(scheduler.run_pipeline(&host));

    assert!(error.to_string().contains("unrepaired-fold"), "{error}");
    assert!(
        error.to_string().contains("did not repair the SSA"),
        "{error}"
    );
    let stored = some_or_abort(host.clone_ssa(&method));
    assert_eq!(format!("{stored:?}"), format!("{original:?}"));
}

/// Folds the diamond's branch through an edit session that prunes the join's
/// phi operand, then reports no change.
struct UnderReportingFoldPass;

impl SsaPass<MockTarget, MockHost> for UnderReportingFoldPass {
    fn name(&self) -> &'static str {
        "under-reporting-fold"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    fn repairs_ssa(&self) -> bool {
        true
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        ssa.edit(SsaEditOptions::new(), |editor| {
            editor.fold_terminator_pruning_phis(0, SsaOp::Jump { target: 1 })?;
            Ok(())
        })?;
        // The claim under test: the function was edited above.
        Ok(false)
    }
}

/// **A pass that repairs its own graph edits is kept when it under-reports.**
/// Its edit session repaired the edges at its boundary, so the scheduler adds
/// only the instruction-level clean-up and treats the method as changed.
#[test]
fn a_self_repairing_graph_change_is_kept_when_under_reported() {
    let host = MockHost::with_rounds(1, None);
    let method = 1u32;
    host.insert_ssa(method, testing::diamond_phi_fixture());

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        verify_hard: true,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::empty(config);
    scheduler.add_at_layer(Box::new(UnderReportingFoldPass), 0);

    result_or_abort(scheduler.run_pipeline(&host));

    assert!(host.is_processed(&method));
    let stored = some_or_abort(host.clone_ssa(&method));
    assert!(
        matches!(
            stored.block(0).and_then(|block| block.control_terminator()),
            Some(SsaOp::Jump { target: 1 })
        ),
        "the fold was kept\n{stored}"
    );
    assert_valid_full(&stored, "under-reporting-fold");
}

/// Under `RollbackAndContinue` a pass failing on a method does not fail the
/// run: the method keeps its body from before the pass, the failure is a
/// `PassFailed` event naming the method and the pass, and a broken result is
/// rolled back the same way.
#[test]
fn a_failed_pass_is_rolled_back_and_recorded_when_the_run_continues() {
    let host = MockHost::with_rounds(1, None);
    let original = testing::const_i32_return(7);
    host.insert_ssa(1, original.clone());
    host.insert_ssa(2, original.clone());

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        verify_hard: true,
        on_method_failure: FailurePolicy::RollbackAndContinue,
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::new(config);
    scheduler.add_at_layer(Box::new(FailingPass), 0);
    scheduler.add_at_layer(Box::new(BreakingPass), 0);

    result_or_abort(scheduler.run_pipeline(&host));

    for method in [1, 2] {
        let stored = some_or_abort(host.clone_ssa(&method));
        assert_eq!(format!("{stored}"), format!("{original}"));
    }
    let failed: Vec<_> = host.events.filter_kind(EventKind::PassFailed).collect();
    assert!(
        failed
            .iter()
            .any(|event| event.pass.as_deref() == Some("failing")
                && event.method == Some(1)
                && event.message.contains("pass failed"))
    );
    assert!(
        failed
            .iter()
            .any(|event| event.pass.as_deref() == Some("breaking")
                && event.message.contains("invalid SSA"))
    );
}

#[test]
fn default_scheduler_verify_hard_handles_mixed_builder_fixtures() {
    let host = MockHost::with_rounds(2, Some(3));
    host.insert_ssa(1, testing::scalar_rewrite_fixture());
    host.insert_ssa(2, testing::diamond_phi_fixture());
    host.insert_ssa(3, testing::memory_effect_fixture());
    host.insert_ssa(4, testing::native_effect_fixture());
    host.insert_ssa(5, testing::vector_simd_fixture());

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        verify_hard: true,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::new(config);

    let changes = result_or_abort(scheduler.run_pipeline(&host));
    assert!(changes > 0, "mixed fixtures should expose scheduler work");

    for method in host.iter_methods() {
        let ssa = some_or_abort(host.clone_ssa(&method));
        assert_valid_full(&ssa, &format!("method {method} after scheduler"));
    }
}

#[test]
fn built_in_pass_wrappers_report_expected_metadata() {
    let instructions_only = ModificationScope::InstructionsOnly;
    let cfg = ModificationScope::CfgModifying;

    let algebraic = AlgebraicSimplificationPass::new();
    assert_eq!(pass_name(&algebraic), "algebraic-simplification");
    assert_eq!(pass_scope(&algebraic), instructions_only);
    assert!(pass_repairs_ssa(&algebraic));
    assert!(!pass_is_global(&AsSsaPass(algebraic)));

    let blockmerge = BlockMergingPass;
    assert_eq!(pass_name(&blockmerge), "block-merging");
    assert_eq!(pass_scope(&blockmerge), cfg);
    assert!(pass_repairs_ssa(&blockmerge));

    let controlflow = ControlFlowSimplificationPass;
    assert_eq!(pass_name(&controlflow), "control-flow-simplification");
    assert_eq!(pass_scope(&controlflow), cfg);
    assert!(pass_repairs_ssa(&controlflow));

    let copying = CopyPropagationPass;
    assert_eq!(pass_name(&copying), "copy-propagation");
    assert_eq!(pass_scope(&copying), instructions_only);
    assert!(pass_repairs_ssa(&copying));

    let memory_opt = MemoryOptimizationPass;
    assert_eq!(pass_name(&memory_opt), "memory-optimization");
    assert_eq!(pass_scope(&memory_opt), instructions_only);
    assert!(pass_repairs_ssa(&memory_opt));
    assert!(!pass_is_global(&AsSsaPass(memory_opt)));

    let dce = DeadCodeEliminationPass;
    assert_eq!(pass_name(&dce), "dead-code-elimination");
    assert_eq!(pass_scope(&dce), cfg);
    assert!(pass_repairs_ssa(&dce));

    // A whole-program pass says so by being a `ModulePass`: it has no
    // per-function entry point to leave unimplemented, and the adapter is what
    // presents it to a scheduler that drives `SsaPass`.
    let global_dce = DeadMethodEliminationPass;
    assert_eq!(
        whole_program_pass_name(&global_dce),
        "dead-method-elimination"
    );
    assert!(pass_is_global(&AsGlobalPass(global_dce)));

    let gvn = GlobalValueNumberingPass::new();
    assert_eq!(pass_name(&gvn), "global-value-numbering");
    assert_eq!(pass_scope(&gvn), instructions_only);
    assert!(pass_repairs_ssa(&gvn));

    let licm = LicmPass::new();
    assert_eq!(pass_name(&licm), "licm");
    assert_eq!(pass_scope(&licm), cfg);
    assert!(pass_repairs_ssa(&licm));

    let loopcanon = LoopCanonicalizationPass::new();
    assert_eq!(pass_name(&loopcanon), "loop-canonicalization");
    assert_eq!(pass_scope(&loopcanon), cfg);
    assert!(pass_repairs_ssa(&loopcanon));

    let predicates = OpaquePredicatePass::<MockTarget>::new();
    assert_eq!(pass_name(&predicates), "opaque-predicate");
    assert_eq!(pass_scope(&predicates), cfg);
    assert!(pass_repairs_ssa(&predicates));

    let ranges = ValueRangePropagationPass::new();
    assert_eq!(pass_name(&ranges), "value-range-propagation");
    assert_eq!(pass_scope(&ranges), cfg);
    assert!(pass_repairs_ssa(&ranges));

    let reassociation = ReassociationPass::new();
    assert_eq!(pass_name(&reassociation), "reassociation");
    assert_eq!(pass_scope(&reassociation), instructions_only);
    assert!(pass_repairs_ssa(&reassociation));

    let strength = StrengthReductionPass::new();
    assert_eq!(pass_name(&strength), "strength-reduction");
    assert_eq!(pass_scope(&strength), instructions_only);
    assert!(pass_repairs_ssa(&strength));

    let threading = JumpThreadingPass::new();
    assert_eq!(pass_name(&threading), "jump-threading");
    assert_eq!(pass_scope(&threading), cfg);
    assert!(pass_repairs_ssa(&threading));
}

/// The scheduler needs two method orders per pass batch - every method, and the
/// dirty subset. Both must come from one call to `methods_reverse_topological`,
/// whose default implementation builds the entire call graph: a call per list
/// would double the call-graph cost of every batch, and batches run inside the
/// scheduler's own fixpoint, so it would be paid per iteration too.
#[test]
fn a_pass_batch_builds_the_call_graph_once() {
    let host = MockHost::with_rounds(1, None);
    for method in 1u32..=4 {
        host.insert_ssa(method, testing::const_i32_return(method as i32));
    }

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::new(config);
    scheduler.add_at_layer(Box::new(CountingPass), 0);

    host.topo_calls.store(0, Ordering::Relaxed);
    result_or_abort(scheduler.run_pipeline(&host));

    // This configuration runs two batches. Each must build the call graph once,
    // for two total rather than four (one per method list per batch).
    let calls = host.topo_calls.load(Ordering::Relaxed);
    assert_eq!(
        calls, 2,
        "each pass batch must build the call graph once, not once per method list"
    );
}

/// A pass that touches nothing, so the pipeline runs exactly one batch.
struct CountingPass;

impl SsaPass<MockTarget, MockHost> for CountingPass {
    fn name(&self) -> &'static str {
        "counting"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }

    fn run_on_method(
        &self,
        _ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        Ok(false)
    }
}

/// A pass that reports a change on every method every time it runs, so no
/// fixpoint is ever reached.
struct ChurningPass;

impl SsaPass<MockTarget, MockHost> for ChurningPass {
    fn name(&self) -> &'static str {
        "churning"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }

    fn run_on_method(
        &self,
        _ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        Ok(true)
    }
}

/// **A scheduler that runs out of rounds says which ones.** A layer whose pass
/// changes something every round exhausts its phase rounds in each pipeline
/// round, and the pipeline its own; a pass that settles exhausts neither.
#[test]
fn a_scheduler_out_of_rounds_reports_its_budgets() {
    let run = |pass: Box<dyn SsaPass<MockTarget, MockHost>>| {
        let host = MockHost::with_rounds(3, Some(2));
        host.insert_ssa(1, testing::const_i32_return(1));
        let mut scheduler = PassScheduler::<MockTarget, MockHost>::empty(PipelineConfig {
            include_dead_method_elimination: false,
            ..PipelineConfig::default()
        });
        scheduler.add_at_layer(pass, 0);
        result_or_abort(scheduler.run_pipeline(&host));
        let exhausted: Vec<String> = host
            .events
            .filter_kind(EventKind::BudgetExhausted)
            .map(|event| event.message.clone())
            .collect();
        exhausted
    };

    let churning = run(Box::new(ChurningPass));
    let phases = churning
        .iter()
        .filter(|message| message.starts_with("phase rounds"))
        .count();
    let pipelines = churning
        .iter()
        .filter(|message| message.starts_with("pipeline rounds"))
        .count();
    assert_eq!(
        phases, 3,
        "one exhausted layer per pipeline round: {churning:?}"
    );
    assert_eq!(pipelines, 1, "and the pipeline once: {churning:?}");

    assert!(
        run(Box::new(CountingPass)).is_empty(),
        "a settled run spends no budget"
    );
}

/// A host that discards events entirely.
///
/// This is the shape that `SsaPassHost::events()` returning a concrete
/// `&EventLog<T>` made impossible: `EventLog` is an append-only `boxcar::Vec`
/// with no bound and no drain, so a host that never consumes events still had to
/// retain every one of them for the process lifetime.
#[derive(Default)]
struct SilentHost {
    ssa: Mutex<BTreeMap<u32, SsaFunction<MockTarget>>>,
    dirty: Mutex<BTreeSet<u32>>,
    sink: NullListener,
    /// The float rules of the machine the host states.
    float: FloatRules,
}

impl World<MockTarget> for SilentHost {
    fn all_methods(&self) -> Vec<u32> {
        self.iter_methods()
    }

    fn entry_points(&self) -> Vec<u32> {
        self.iter_methods()
    }

    fn callees(&self, _method: &u32) -> Vec<u32> {
        Vec::new()
    }

    fn is_dead(&self, _method: &u32) -> bool {
        false
    }

    fn mark_dead(&self, _method: &u32) {}
}

impl SsaStore<MockTarget> for SilentHost {
    fn take_ssa(&self, method: &u32) -> Option<SsaFunction<MockTarget>> {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .remove(method)
    }

    fn clone_ssa(&self, method: &u32) -> Option<SsaFunction<MockTarget>> {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .get(method)
            .cloned()
    }

    fn insert_ssa(&self, method: u32, ssa: SsaFunction<MockTarget>) {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .insert(method, ssa);
    }

    fn contains(&self, method: &u32) -> bool {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .contains_key(method)
    }

    fn iter_methods(&self) -> Vec<u32> {
        self.ssa
            .lock()
            .unwrap_or_else(|_| process::abort())
            .keys()
            .copied()
            .collect()
    }
}

impl DirtySet<MockTarget> for SilentHost {
    fn mark_dirty(&self, method: &u32) {
        self.dirty
            .lock()
            .unwrap_or_else(|_| process::abort())
            .insert(*method);
    }

    fn is_dirty(&self, method: &u32) -> bool {
        self.dirty
            .lock()
            .unwrap_or_else(|_| process::abort())
            .contains(method)
    }

    fn mark_processed(&self, _method: &u32) {}

    fn is_processed(&self, _method: &u32) -> bool {
        false
    }

    fn dirty_snapshot(&self) -> Vec<u32> {
        self.dirty
            .lock()
            .unwrap_or_else(|_| process::abort())
            .iter()
            .copied()
            .collect()
    }

    fn clear_dirty_for(&self, method: &u32) {
        self.dirty
            .lock()
            .unwrap_or_else(|_| process::abort())
            .remove(method);
    }
}

impl SsaPassHost<MockTarget> for SilentHost {
    fn events(&self) -> &dyn EventListener<MockTarget> {
        &self.sink
    }

    fn context(&self) -> AnalysisContext<'_> {
        AnalysisContext::new(Machine {
            float: self.float,
            ..MACHINE
        })
    }
}

/// Records the machine of every environment it runs in.
struct MachineProbe(Arc<Mutex<Vec<Machine>>>);

impl FunctionPass<MockTarget> for MachineProbe {
    fn name(&self) -> &'static str {
        "machine probe"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }

    fn run(
        &self,
        _ssa: &mut SsaFunction<MockTarget>,
        env: &PassEnv<'_, MockTarget>,
    ) -> Result<bool> {
        self.0
            .lock()
            .unwrap_or_else(|_| process::abort())
            .push(env.context().machine());
        Ok(false)
    }
}

/// **A pass the scheduler runs computes on the host's machine.** The
/// environment it is handed states the host's float rules, so a constant
/// invalid operation folds to the target's default NaN there as it does in a
/// pass run directly.
#[test]
fn a_scheduled_pass_computes_on_the_hosts_machine() {
    let float = FloatRules {
        out_of_range: None,
        default_nan: Some(DefaultNan::Positive),
        nan_conversion: None,
        nan_operands: None,
        extended_nan_operands: None,

        program_flushes_subnormals: false,

        nan_encoding: NanEncoding::Ieee,
    };
    let host = SilentHost {
        float,
        ..SilentHost::default()
    };
    host.insert_ssa(1, testing::const_i32_return(7));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let config = PipelineConfig {
        include_dead_method_elimination: false,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, SilentHost>::new(config);
    scheduler.add_at_layer(Box::new(AsSsaPass(MachineProbe(Arc::clone(&seen)))), 0);
    result_or_abort(scheduler.run_pipeline(&host));

    let seen = seen.lock().unwrap_or_else(|_| process::abort());
    assert!(!seen.is_empty(), "the probe ran");
    assert!(
        seen.iter()
            .all(|machine| *machine == host.context().machine())
    );
}

#[test]
fn a_host_can_discard_events_entirely() {
    let host = SilentHost::default();
    host.insert_ssa(1, testing::const_i32_return(7));

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, SilentHost>::new(config);
    scheduler.add_at_layer(Box::new(CountingPass2), 0);

    result_or_abort(scheduler.run_pipeline(&host));

    // A discarding sink reports no history, and the scheduler's debug summary
    // degrades to "no detail" rather than failing to compile or panicking.
    assert_eq!(host.events().recorded_count(), 0);
    assert!(host.events().count_by_kind_since(0).is_empty());
    assert!(!host.events().is_enabled());
}

struct CountingPass2;

impl SsaPass<MockTarget, SilentHost> for CountingPass2 {
    fn name(&self) -> &'static str {
        "silent"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }

    fn run_on_method(
        &self,
        _ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &SilentHost,
    ) -> Result<bool> {
        Ok(false)
    }
}

/// A pass that panics.
struct PanickingPass;

impl SsaPass<MockTarget, MockHost> for PanickingPass {
    fn name(&self) -> &'static str {
        "panicking"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }

    // The crate denies `clippy::panic`, but this test exists to prove the
    // scheduler survives a real unwind, which needs a real panic.
    #[allow(clippy::panic)]
    fn run_on_method(
        &self,
        _ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        panic!("deliberate pass panic");
    }
}

/// The scheduler *takes* a method's SSA from the store before running a pass, so
/// an unwind through the pass skips every reinsertion and the method loses its
/// body permanently - a panicking pass would silently delete code rather than
/// fail. The panic must be reported and the function restored.
#[test]
fn a_panicking_pass_does_not_destroy_the_method() {
    let host = MockHost::with_rounds(1, None);
    let method = 1u32;
    let original = testing::const_i32_return(7);
    host.insert_ssa(method, original.clone());

    let config = PipelineConfig {
        include_dead_method_elimination: false,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::new(config);
    scheduler.add_at_layer(Box::new(PanickingPass), 0);

    let error = err_or_abort(scheduler.run_pipeline(&host));

    assert!(
        error.to_string().contains("panicked"),
        "the panic must be reported as an error; got {error}"
    );
    assert!(
        error.to_string().contains("panicking"),
        "and must name the pass; got {error}"
    );

    let stored = some_or_abort(host.clone_ssa(&method));
    assert_eq!(
        format!("{stored}"),
        format!("{original}"),
        "the method's SSA must survive the panic"
    );
}

/// Rewrites the method's return to read an undefined variable, then fails.
struct EditThenFailPass {
    /// Whether to declare `reads_peer_ssa`, so the scheduler clones the SSA
    /// instead of taking it.
    peer: bool,
}

impl SsaPass<MockTarget, MockHost> for EditThenFailPass {
    fn name(&self) -> &'static str {
        "edit-then-fail"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        ssa.replace_instruction_op(0, 1, SsaOp::returns(Some(SsaVarId::from_index(99))));
        Err(Error::new("failed after editing"))
    }

    fn reads_peer_ssa(&self) -> bool {
        self.peer
    }
}

/// A pass that edits the function and then fails must leave the stored SSA
/// exactly as it was, with or without `verify_hard`, whether the scheduler took
/// the SSA out of the store or cloned it for a peer-reading pass.
#[test]
fn a_pass_that_edits_then_fails_leaves_the_stored_ssa_untouched() {
    for peer in [false, true] {
        let host = MockHost::with_rounds(1, None);
        let method = 1u32;
        let original = testing::const_i32_return(7);
        host.insert_ssa(method, original.clone());

        let config = PipelineConfig::default();
        let mut scheduler = PassScheduler::<MockTarget, MockHost>::empty(config);
        scheduler.add_at_layer(Box::new(EditThenFailPass { peer }), 0);

        let error = err_or_abort(scheduler.run_pipeline(&host));
        assert!(
            error.to_string().contains("failed after editing"),
            "the pass error is reported; got {error}"
        );

        let stored = some_or_abort(host.clone_ssa(&method));
        assert_eq!(
            format!("{stored}"),
            format!("{original}"),
            "the half-edited function must not reach the store (peer = {peer})"
        );
    }
}

/// Edits through a checked session once, then reports `Ok(false)`.
struct SilentEditPass {
    /// Set after the first edit so the pass settles.
    fired: AtomicBool,
}

impl SsaPass<MockTarget, MockHost> for SilentEditPass {
    fn name(&self) -> &'static str {
        "silent-edit"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        if !self.fired.swap(true, Ordering::Relaxed) {
            ssa.edit(SsaEditOptions::new(), |editor| {
                editor.insert_before_terminator(0, SsaInstruction::new((), SsaOp::Nop))?;
                Ok(())
            })?;
        }
        // The claim under test: the function was edited above.
        Ok(false)
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }
}

/// A pass that edits the function but reports "unchanged" is still treated as
/// having changed it: the method is marked processed, its analyses are
/// invalidated, the repair runs, and the edit flag does not linger on the
/// stored SSA.
#[test]
fn a_pass_that_edits_but_reports_unchanged_is_still_treated_as_changed() {
    let host = MockHost::with_rounds(1, None);
    let method = 1u32;
    host.insert_ssa(method, testing::const_i32_return(7));

    let config = PipelineConfig {
        verify_hard: true,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::empty(config);
    scheduler.add_at_layer(
        Box::new(SilentEditPass {
            fired: AtomicBool::new(false),
        }),
        0,
    );

    result_or_abort(scheduler.run_pipeline(&host));

    assert!(
        host.is_processed(&method),
        "the edited method must be marked processed"
    );
    let mut stored = some_or_abort(host.clone_ssa(&method));
    assert!(
        !stored.take_edit_dirty(),
        "the edit flag must be consumed, not stored with the function"
    );
    assert_valid_full(&stored, "silent-edit");
    assert_eq!(
        *host.invalidated.lock().unwrap_or_else(|_| process::abort()),
        vec![method]
    );
}

/// Reports a change for one method, once.
struct ChangeOnePass {
    /// The method the pass changes.
    target: u32,
    /// Set after the change so the pass settles.
    fired: AtomicBool,
}

impl SsaPass<MockTarget, MockHost> for ChangeOnePass {
    fn name(&self) -> &'static str {
        "change-one"
    }

    fn run_on_method(
        &self,
        _ssa: &mut SsaFunction<MockTarget>,
        method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        Ok(*method == self.target && !self.fired.swap(true, Ordering::Relaxed))
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }
}

/// The scheduler asks the host to invalidate analyses for exactly the methods a
/// pass changed, once per changing run - not for the methods it left alone.
#[test]
fn analyses_are_invalidated_exactly_for_changed_methods() {
    let host = MockHost::with_rounds(1, None);
    for method in 1u32..=3 {
        host.insert_ssa(method, testing::const_i32_return(method as i32));
    }

    let config = PipelineConfig::default();
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::empty(config);
    scheduler.add_at_layer(
        Box::new(ChangeOnePass {
            target: 2,
            fired: AtomicBool::new(false),
        }),
        0,
    );

    result_or_abort(scheduler.run_pipeline(&host));

    assert_eq!(
        *host.invalidated.lock().unwrap_or_else(|_| process::abort()),
        vec![2]
    );
}

/// The constant a `const_i32_return` fixture returns.
fn returned_constant(ssa: &SsaFunction<MockTarget>) -> Option<i32> {
    ssa.blocks()
        .first()?
        .instructions()
        .iter()
        .find_map(|instr| match instr.op() {
            SsaOp::Const {
                value: ConstValue::I32(value),
                ..
            } => Some(*value),
            _ => None,
        })
}

/// How many methods [`ReadPeerPass`] runs over.
const PEERS: u32 = 64;

/// Rewrites each method to return its peer's constant plus 1000, reading the
/// peer from the store, once per method.
struct ReadPeerPass {
    /// Methods already rewritten, so the pass settles.
    fired: Mutex<BTreeSet<u32>>,
}

impl SsaPass<MockTarget, MockHost> for ReadPeerPass {
    fn name(&self) -> &'static str {
        "read-peer"
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<MockTarget>,
        method: &u32,
        host: &MockHost,
    ) -> Result<bool> {
        let mut fired = self.fired.lock().map_err(|_| Error::new("poisoned"))?;
        if !fired.insert(*method) {
            return Ok(false);
        }
        drop(fired);

        // Give the other workers time to finish and store their results, so a
        // store that changes under a running pass is seen doing so.
        thread::sleep(Duration::from_millis(2));
        let peer = (method % PEERS).saturating_add(1);
        let peer_ssa = host
            .clone_ssa(&peer)
            .ok_or_else(|| Error::new("peer missing"))?;
        let peer_value = returned_constant(&peer_ssa).ok_or_else(|| Error::new("no constant"))?;

        let Some(SsaOp::Const { dest, .. }) = ssa
            .block(0)
            .and_then(|block| block.instructions().first())
            .map(|i| i.op().clone())
        else {
            return Err(Error::new("no constant to rewrite"));
        };
        ssa.replace_instruction_op(
            0,
            0,
            SsaOp::Const {
                dest,
                value: ConstValue::I32(peer_value.saturating_add(1000)),
            },
        );
        Ok(true)
    }

    fn reads_peer_ssa(&self) -> bool {
        true
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::UsesOnly
    }
}

/// A pass that reads other methods sees each of them as it was when the pass
/// began. When results went back into the store as each worker finished, a
/// method read its peer before or after the peer's own rewrite depending on
/// thread timing, and the same program came out differently under different
/// load.
#[test]
fn a_peer_reading_pass_sees_its_peers_as_they_were() {
    let host = MockHost::with_rounds(1, None);
    for method in 1..=PEERS {
        host.insert_ssa(method, testing::const_i32_return(method as i32));
    }

    let config = PipelineConfig::default();
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::empty(config);
    scheduler.add_at_layer(
        Box::new(ReadPeerPass {
            fired: Mutex::new(BTreeSet::new()),
        }),
        0,
    );
    result_or_abort(scheduler.run_pipeline(&host));

    for method in 1..=PEERS {
        let stored = some_or_abort(host.clone_ssa(&method));
        let peer = method % PEERS + 1;
        assert_eq!(
            returned_constant(&stored),
            Some(peer as i32 + 1000),
            "method {method} read peer {peer} after the peer had been rewritten"
        );
    }
}

/// Corrupts the function directly, bypassing any edit session, and reports no
/// change.
struct SilentBreakingPass;

impl SsaPass<MockTarget, MockHost> for SilentBreakingPass {
    fn name(&self) -> &'static str {
        "silent-breaking"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    fn run_on_method(
        &self,
        ssa: &mut SsaFunction<MockTarget>,
        _method: &u32,
        _host: &MockHost,
    ) -> Result<bool> {
        ssa.replace_instruction_op(0, 1, SsaOp::returns(Some(SsaVarId::from_index(99))));
        Ok(false)
    }
}

/// `verify_hard` verifies every method after every pass, not only the ones a
/// pass reports changing: a direct mutation leaves no edit flag behind, so the
/// report is the only other signal, and it can be wrong.
#[test]
fn verify_hard_checks_a_pass_that_reports_no_change() {
    let host = MockHost::with_rounds(1, None);
    let method = 1u32;
    let original = testing::const_i32_return(7);
    host.insert_ssa(method, original.clone());

    let config = PipelineConfig {
        verify_hard: true,
        ..PipelineConfig::default()
    };
    let mut scheduler = PassScheduler::<MockTarget, MockHost>::empty(config);
    scheduler.add_at_layer(Box::new(SilentBreakingPass), 0);

    let error = err_or_abort(scheduler.run_pipeline(&host));

    assert!(error.to_string().contains("invalid SSA"), "got {error}");
    let stored = some_or_abort(host.clone_ssa(&method));
    assert_eq!(format!("{stored}"), format!("{original}"));
}

/// `run_once` runs one pass on one function as a scheduled run does and
/// reports a rejection as an error, restoring the function: SSA the verifier
/// rejects names the pass and the verifier's complaint, a pass error carries
/// the pass's message, and a panic is caught. Unverified, the broken result is
/// kept, which is why the level is the caller's to choose; a pass that changes
/// nothing reports so.
#[test]
fn run_once_runs_a_pass_as_the_scheduler_does_and_reports_a_rejection() {
    let host = MockHost::default();
    let method = 1u32;
    let original = testing::const_i32_return(7);
    let verified = Verify::Always(VerifyLevel::Standard);
    type Scheduler = PassScheduler<MockTarget, MockHost>;

    let mut ssa = original.clone();
    let error = err_or_abort(Scheduler::run_once(
        &BreakingPass,
        &mut ssa,
        &method,
        &host,
        verified,
    ));
    assert!(error.to_string().contains("breaking"), "{error}");
    assert!(error.to_string().contains("invalid SSA"), "{error}");
    assert_eq!(format!("{ssa}"), format!("{original}"));

    let mut ssa = original.clone();
    let changed = result_or_abort(Scheduler::run_once(
        &BreakingPass,
        &mut ssa,
        &method,
        &host,
        Verify::Never,
    ));
    assert!(changed);
    assert_ne!(format!("{ssa}"), format!("{original}"));

    let mut ssa = original.clone();
    let error = err_or_abort(Scheduler::run_once(
        &FailingPass,
        &mut ssa,
        &method,
        &host,
        verified,
    ));
    assert!(error.to_string().contains("pass failed"), "{error}");
    assert_eq!(format!("{ssa}"), format!("{original}"));

    let mut ssa = original.clone();
    let error = err_or_abort(Scheduler::run_once(
        &PanickingPass,
        &mut ssa,
        &method,
        &host,
        verified,
    ));
    assert!(error.to_string().contains("panicked"), "{error}");
    assert_eq!(format!("{ssa}"), format!("{original}"));

    let mut ssa = original.clone();
    let changed = result_or_abort(Scheduler::run_once(
        &CountingPass,
        &mut ssa,
        &method,
        &host,
        verified,
    ));
    assert!(!changed);
    assert_eq!(format!("{ssa}"), format!("{original}"));
}

/// A pass that edits the function but reports "unchanged" is a change to
/// `run_once` too, repaired and verified, as in a scheduled run.
#[test]
fn run_once_treats_a_silent_edit_as_a_change() {
    let host = MockHost::default();
    let mut ssa = testing::const_i32_return(7);
    let pass = SilentEditPass {
        fired: AtomicBool::new(false),
    };
    let changed = result_or_abort(PassScheduler::<MockTarget, MockHost>::run_once(
        &pass,
        &mut ssa,
        &1u32,
        &host,
        Verify::Changed(VerifyLevel::Standard),
    ));
    assert!(changed);
    assert_valid_full(&ssa, "a silently edited function");
}
