//! Running a list of passes over one function, to a fixpoint, and saying what
//! happened.
//!
//! [`PassScheduler`](crate::scheduling::PassScheduler) is the whole-program
//! driver: it orders passes by capability, dispatches across methods and tracks
//! which are dirty. A host that has one function and a list of passes needs
//! none of that, and every one that tried wrote the same loop instead - run the
//! group, catch the rollback, iterate until nothing changes, give up at a cap.
//! Written twice it came out twice, with different answers for the same
//! questions: whether an unconverged run is an error, whether a rolled-back
//! group counts as a change, and whether anyone is told at all.
//!
//! The loop is here, the list is data ([`PipelineSpec`]), and what happened
//! comes back as a [`PipelineReport`] rather than a log line. That last part is
//! the one that matters: a group that fails verification is **rolled back and
//! the pass reports no change**, so a pipeline that only returns "changed"
//! cannot tell "there was nothing to do" from "everything it did was thrown
//! away". One of those is a healthy function and the other is a defect in a
//! pass, and a host that cannot see the difference finds out much later.

use std::fmt;

use crate::{
    analysis::{VerifyLevel, budget::Budget},
    ir::function::SsaFunction,
    scheduling::{
        env::{FunctionPass, PassEnv, run_pass_with_repair},
        transaction::{GroupOutcome, PassTransaction},
    },
    target::Target,
};

/// How much a group's failure is allowed to cost.
///
/// Two questions, and they are worth separating. *Whether* a group's work was
/// discarded is known the moment it happens and costs nothing to record.
/// *Which* pass discarded it is only answerable by running the passes again,
/// one at a time - and the answer is for whoever is reading the report, not for
/// the run producing it, which can do nothing with a name.
///
/// So the default records and moves on, and naming the pass is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupPolicy {
    /// One snapshot for the group; if it fails, restore and record it.
    ///
    /// The default. A rollback appears in the report against the group, with
    /// no pass named, and the run continues - the function is back as it was,
    /// which is the guarantee the snapshot exists for. Nothing is re-run, so a
    /// failure costs one restore.
    #[default]
    Group,
    /// [`Group`](Self::Group), and on a failure re-run the passes one at a
    /// time under their own snapshots.
    ///
    /// For diagnosis. It buys two things the cheap policy cannot: the name of
    /// the pass whose own run is rejected, and the work of the passes that
    /// would have succeeded, which the group-level rollback discarded along
    /// with the rest. Both are paid for with a second run of every pass in the
    /// group, on a path that should not be taken at all.
    ///
    /// [`PipelineSpec::diagnose`] turns this on for every group at once, which
    /// is how a host re-runs something its report flagged.
    GroupThenIsolate,
    /// A snapshot per pass, always.
    ///
    /// For a group whose passes are known to be able to break each other, where
    /// the group-level attempt would usually be wasted work.
    PerPass,
    /// No snapshot, no verification, no rollback.
    ///
    /// For passes a host trusts absolutely, or a stage where a later
    /// reconstruction will repair whatever happens. Nothing is checked, so
    /// nothing is reported: a rollback cannot appear in the report for a group
    /// that cannot roll back.
    Unguarded,
}

/// One named group of passes, and what protects it.
pub struct PassGroup<T: Target> {
    /// The group's name, as it appears in the report.
    pub name: &'static str,
    /// The passes, in the order they run.
    pub passes: Vec<Box<dyn FunctionPass<T>>>,
    /// How much its failure is allowed to cost.
    pub policy: GroupPolicy,
}

impl<T: Target> PassGroup<T> {
    /// Builds a group with the default policy.
    ///
    /// # Arguments
    ///
    /// * `name` - The group's name, as it appears in the report.
    /// * `passes` - The passes, in the order they run.
    #[must_use]
    pub fn new(name: &'static str, passes: Vec<Box<dyn FunctionPass<T>>>) -> Self {
        Self {
            name,
            passes,
            policy: GroupPolicy::default(),
        }
    }

    /// Sets the group's policy.
    ///
    /// # Arguments
    ///
    /// * `policy` - How much the group's failure is allowed to cost.
    #[must_use]
    pub fn with_policy(mut self, policy: GroupPolicy) -> Self {
        self.policy = policy;
        self
    }
}

impl<T: Target> fmt::Debug for PassGroup<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PassGroup")
            .field("name", &self.name)
            .field("passes", &self.passes.len())
            .field("policy", &self.policy)
            .finish()
    }
}

/// A pipeline, as data.
///
/// Stated rather than coded, so the *order passes run in* is a value a host can
/// read, print, compare against another host's, and reuse. Two hosts running
/// "the normalization pipeline" should be running the same one, and this is
/// what makes that checkable.
pub struct PipelineSpec<T: Target> {
    /// The groups, in the order they run within one iteration.
    ///
    /// The most iterations of the whole list are the host's
    /// [`Budget::PipelineRounds`], and a run that reaches them without
    /// settling is reported as unconverged rather than treated as finished.
    pub groups: Vec<PassGroup<T>>,
    /// The level a guarded group's result is verified at, or `None` to accept
    /// results unverified.
    pub verify: Option<VerifyLevel>,
    /// Whether to record a per-pass trace in the report.
    ///
    /// Off by default: the trace is one entry per pass per iteration, which a
    /// host wants while diagnosing and not while running.
    pub telemetry: bool,
}

impl<T: Target> PipelineSpec<T> {
    /// Builds a spec from its groups, with no telemetry.
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups, in the order they run.
    #[must_use]
    pub const fn new(groups: Vec<PassGroup<T>>) -> Self {
        Self {
            groups,
            verify: Some(VerifyLevel::Standard),
            telemetry: false,
        }
    }

    /// Records a per-pass trace in the report.
    #[must_use]
    pub const fn with_telemetry(mut self) -> Self {
        self.telemetry = true;
        self
    }

    /// Sets the level guarded groups are verified at.
    ///
    /// # Arguments
    ///
    /// * `verify` - The level, or `None` to accept results unverified.
    #[must_use]
    pub const fn verifying(mut self, verify: Option<VerifyLevel>) -> Self {
        self.verify = verify;
        self
    }

    /// Turns the spec into the one a host re-runs to explain a report.
    ///
    /// Records the per-pass trace, and upgrades every
    /// [`GroupPolicy::Group`] to [`GroupPolicy::GroupThenIsolate`] so a
    /// rollback names the pass responsible instead of only the group.
    ///
    /// Both cost, which is why neither is on for an ordinary run: the trace is
    /// an entry per pass per iteration, and isolation runs every pass in a
    /// failing group a second time. A host reaches for this once its report
    /// says something was discarded - not before, when there is nothing to
    /// explain.
    ///
    /// [`GroupPolicy::PerPass`] and [`GroupPolicy::Unguarded`] are left as they
    /// are: both are deliberate choices about a particular group, and neither
    /// hides which pass did what.
    #[must_use]
    pub fn diagnose(mut self) -> Self {
        self.telemetry = true;
        for group in &mut self.groups {
            if matches!(group.policy, GroupPolicy::Group) {
                group.policy = GroupPolicy::GroupThenIsolate;
            }
        }
        self
    }
}

impl<T: Target> fmt::Debug for PipelineSpec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineSpec")
            .field("groups", &self.groups)
            .field("verify", &self.verify)
            .field("telemetry", &self.telemetry)
            .finish()
    }
}

/// Why a group's or a pass's work was discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollbackReason {
    /// The result did not verify, or a pass panicked.
    ///
    /// The transaction does not distinguish the two at its boundary: both
    /// restore the function, and both are a defect in a pass rather than a
    /// property of the input.
    Rejected,
    /// A pass returned an error.
    ///
    /// Under every policy but [`GroupPolicy::Unguarded`] the work was
    /// discarded with it; under that one nothing is discarded, and this is the
    /// record that something went wrong anyway.
    Failed,
}

/// One discarded piece of work.
///
/// The record a host needs to answer "why did this method not improve": a
/// pipeline that reports only `changed` cannot tell a function with nothing to
/// do from one where every rewrite was thrown away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rollback {
    /// The group whose work was discarded.
    pub group: &'static str,
    /// The pass, when isolation identified one.
    ///
    /// `None` for a group rolled back as a whole, which is the
    /// [`GroupPolicy::GroupThenIsolate`] case where re-running the passes
    /// individually found none that failed on its own.
    pub pass: Option<&'static str>,
    /// Which iteration of the fixpoint it happened in, counting from one.
    pub iteration: usize,
    /// Why the work was discarded.
    pub reason: RollbackReason,
}

/// One pass run, when telemetry is on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassRun {
    /// The group the pass belongs to.
    pub group: &'static str,
    /// The pass.
    pub pass: &'static str,
    /// Which iteration it ran in, counting from one.
    pub iteration: usize,
    /// Whether it rewrote the function and the rewrite was kept.
    pub changed: bool,
}

/// What a pipeline run did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PipelineReport {
    /// Iterations entered, counting from one. The last one stops as soon as
    /// every pass has seen the function as it stands, so it need not have run
    /// the whole list.
    pub iterations: usize,
    /// Whether the run settled rather than hitting the cap.
    pub converged: bool,
    /// Whether any group rewrote the function and the rewrite was kept.
    pub changed: bool,
    /// One entry per pass run, when the spec asked for telemetry.
    pub trace: Vec<PassRun>,
    /// Every piece of work that was discarded.
    ///
    /// Empty is the healthy answer. A non-empty list on a corpus is the signal
    /// that a pass is producing IR the verifier rejects - which otherwise shows
    /// up only as a pipeline that quietly improves nothing.
    pub rollbacks: Vec<Rollback>,
    /// Whether the host's interrupt stopped the run before it settled
    /// ([`AnalysisContext::with_interrupt`](crate::analysis::AnalysisContext::with_interrupt)):
    /// the function is then part rewritten, and the caller is to discard
    /// it rather than use it.
    pub interrupted: bool,
}

impl PipelineReport {
    /// Whether the run settled with nothing discarded.
    #[must_use]
    pub fn clean(&self) -> bool {
        self.converged && self.rollbacks.is_empty()
    }
}

/// A pipeline run under way: what it has reported and how far it has settled.
struct Progress {
    /// The report being built.
    report: PipelineReport,
    /// Whether to record one trace entry per pass run.
    telemetry: bool,
    /// The iteration running, counting from one.
    iteration: usize,
    /// The passes in the pipeline.
    passes: usize,
    /// Passes run since the function last changed, each having left it alone.
    quiet: usize,
}

impl Progress {
    /// Whether every pass has run on the function as it stands and left it
    /// alone. A pass is a function of what it is handed, so running any of them
    /// again would repeat an answer already in hand.
    const fn settled(&self) -> bool {
        self.quiet >= self.passes
    }

    /// Whether the run must stop: `env`'s interrupt has tripped, which marks
    /// the run interrupted. Asked before a pass, and after one, whose outcome
    /// a tripped interrupt explains: a pass stopped part way is no failure to
    /// record, and the run's function is discarded with it.
    fn stopped<T: Target>(&mut self, env: &PassEnv<'_, T>) -> bool {
        if !self.report.interrupted && env.context().interrupted() {
            self.report.interrupted = true;
            self.report.converged = false;
        }
        self.report.interrupted
    }

    /// Notes one pass run and whether it changed the function.
    const fn note(&mut self, changed: bool) {
        self.quiet = if changed {
            0
        } else {
            self.quiet.saturating_add(1)
        };
    }

    /// Records one pass run, when telemetry is on.
    fn trace(&mut self, group: &'static str, pass: &'static str, changed: bool) {
        if self.telemetry {
            self.report.trace.push(PassRun {
                group,
                pass,
                iteration: self.iteration,
                changed,
            });
        }
    }

    /// Records a piece of discarded work in the current iteration.
    fn rollback(
        &mut self,
        group: &'static str,
        pass: Option<&'static str>,
        reason: RollbackReason,
    ) {
        self.report.rollbacks.push(Rollback {
            group,
            pass,
            iteration: self.iteration,
            reason,
        });
    }
}

/// Runs a [`PipelineSpec`] over one function at a time.
///
/// Holds the snapshot buffer between runs, which is the point: a worker makes
/// one of these and reuses it for every function it processes, and the buffer
/// grows once to the largest it has seen.
#[derive(Debug)]
pub struct FunctionPipeline<T: Target> {
    /// The pipeline to run.
    spec: PipelineSpec<T>,
    /// The reusable snapshot buffer.
    transaction: PassTransaction<T>,
}

impl<T: Target> FunctionPipeline<T> {
    /// Builds a pipeline from a spec.
    ///
    /// # Arguments
    ///
    /// * `spec` - The groups, the cap, and what to verify.
    #[must_use]
    pub fn new(spec: PipelineSpec<T>) -> Self {
        Self {
            spec,
            transaction: PassTransaction::new(),
        }
    }

    /// Returns the spec this pipeline runs.
    #[must_use]
    pub const fn spec(&self) -> &PipelineSpec<T> {
        &self.spec
    }

    /// Runs the pipeline over `ir` until it settles or the host's
    /// [`Budget::PipelineRounds`] are spent.
    ///
    /// # Arguments
    ///
    /// * `ir` - The function to rewrite.
    /// * `env` - Events, pointer width, the method, and the program when there
    ///   is one.
    ///
    /// # Returns
    ///
    /// What the run did, including everything it discarded.
    pub fn run(&mut self, ir: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> PipelineReport {
        // Whatever the environment holds was computed before this run, of a
        // function the caller may have edited since.
        env.ranges().invalidate();
        let mut progress = Progress {
            report: PipelineReport {
                converged: true,
                ..PipelineReport::default()
            },
            telemetry: self.spec.telemetry,
            iteration: 0,
            passes: self
                .spec
                .groups
                .iter()
                .map(|group| group.passes.len())
                .sum(),
            quiet: 0,
        };
        let rounds = env.context().budget().limit(Budget::PipelineRounds).max(1);
        for iteration in 1..=rounds {
            progress.iteration = iteration;
            progress.report.iterations = iteration;
            let mut changed_this_round = false;
            for group in &self.spec.groups {
                if progress.settled() {
                    break;
                }
                // Asked before every group, and before and after every pass:
                // a large function's run is the longest stretch a host's lift
                // has between its own asks.
                if progress.stopped(env) {
                    return progress.report;
                }
                changed_this_round |= GroupRun {
                    transaction: &mut self.transaction,
                    progress: &mut progress,
                    env,
                }
                .run_group(group, ir);
                if progress.stopped(env) {
                    return progress.report;
                }
            }
            progress.report.changed |= changed_this_round;
            if !changed_this_round {
                return progress.report;
            }
            // The last allowed iteration still changed something, so the run
            // was cut short rather than finished.
            if iteration == rounds {
                progress.report.converged = false;
            }
        }
        progress.report
    }
}

/// One group's run: the transaction its rewrites go through, the progress of
/// the whole run it adds to, and the environment the passes run under.
struct GroupRun<'a, 'e, T: Target> {
    /// The snapshot and verification each rewrite is kept or undone by.
    transaction: &'a mut PassTransaction<T>,
    /// What the run has done so far.
    progress: &'a mut Progress,
    /// What the passes run under.
    env: &'a PassEnv<'e, T>,
}

impl<T: Target> GroupRun<'_, '_, T> {
    /// Runs one group under its policy, returning whether anything was kept.
    fn run_group(&mut self, group: &PassGroup<T>, ir: &mut SsaFunction<T>) -> bool {
        match group.policy {
            GroupPolicy::Unguarded => {
                let mut changed = false;
                for pass in &group.passes {
                    if self.progress.settled() || self.progress.stopped(self.env) {
                        break;
                    }
                    let outcome = run_pass_with_repair(pass.as_ref(), ir, self.env);
                    if self.progress.stopped(self.env) {
                        break;
                    }
                    // Nothing is discarded here - that is what the policy
                    // means - but a pass that failed is still recorded, because
                    // silence is how a host ends up reading a function no pass
                    // finished rewriting as one no pass needed to.
                    if outcome.is_err() {
                        self.progress.rollback(
                            group.name,
                            Some(pass.name()),
                            RollbackReason::Failed,
                        );
                    }
                    let ran = outcome.unwrap_or(false);
                    changed |= ran;
                    self.progress.note(ran);
                    self.progress.trace(group.name, pass.name(), ran);
                }
                changed
            }
            GroupPolicy::PerPass => self.run_each(group, ir),
            GroupPolicy::Group | GroupPolicy::GroupThenIsolate => {
                let isolate = matches!(group.policy, GroupPolicy::GroupThenIsolate);
                let mut failed = false;
                let quiet_before = self.progress.quiet;
                // What each pass reported, for the trace: a group kept whole
                // still has passes that rewrote nothing, and naming those as
                // changed would point a host at the wrong pass.
                let mut rewrites = Vec::new();
                let outcome = self.transaction.run_group(ir, |ir| {
                    let mut changed = false;
                    for pass in &group.passes {
                        if self.progress.settled() {
                            break;
                        }
                        let rewrote = match run_pass_with_repair(pass.as_ref(), ir, self.env) {
                            Ok(ran) => ran,
                            // A failure must not read as "changed nothing". An
                            // unchanged group is accepted *without
                            // verification* - a group that rewrote nothing
                            // cannot have broken anything - so reporting false
                            // here would keep whatever the failed pass left
                            // behind, unchecked. Reporting the change is what
                            // makes the self.transaction look, and roll back.
                            Err(_) => {
                                failed = true;
                                true
                            }
                        };
                        changed |= rewrote;
                        self.progress.note(rewrote);
                        if self.progress.telemetry {
                            rewrites.push(rewrote);
                        }
                    }
                    changed
                });
                if self.progress.stopped(self.env) {
                    return false;
                }
                match outcome {
                    GroupOutcome::Applied | GroupOutcome::Unchanged => {
                        for (pass, rewrote) in group.passes.iter().zip(rewrites) {
                            self.progress.trace(group.name, pass.name(), rewrote);
                        }
                        matches!(outcome, GroupOutcome::Applied)
                    }
                    // The group as a whole produced something the verifier
                    // rejected, and the function is back as it was. Re-running
                    // the passes one at a time keeps the ones that are fine and
                    // names the one that is not - which is the whole reason to
                    // pay for isolation, and only here, where something has
                    // already gone wrong.
                    GroupOutcome::RolledBack => {
                        // The function is back as it was before the group,
                        // and a pass inside it may have computed facts about
                        // what the group left.
                        self.env.ranges().invalidate();
                        // The passes before the group saw the function it is
                        // back to.
                        self.progress.quiet = quiet_before;
                        let reason = if failed {
                            RollbackReason::Failed
                        } else {
                            RollbackReason::Rejected
                        };

                        // The cheap answer: the group's work is gone and the
                        // function is back as it was. That is the whole fact,
                        // and it is already in hand.
                        if !isolate {
                            self.progress.rollback(group.name, None, reason);
                            for pass in &group.passes {
                                self.progress.note(false);
                                self.progress.trace(group.name, pass.name(), false);
                            }
                            return false;
                        }

                        let before = self.progress.report.rollbacks.len();
                        let changed = self.run_each(group, ir);
                        // Every pass succeeded alone, so what failed was the
                        // combination. Recorded against the group, with no pass
                        // named, because naming one would be a guess.
                        if self.progress.report.rollbacks.len() == before {
                            self.progress.rollback(group.name, None, reason);
                        }
                        changed
                    }
                }
            }
        }
    }
    /// Runs a group's passes each under its own snapshot, until the function
    /// has settled.
    fn run_each(&mut self, group: &PassGroup<T>, ir: &mut SsaFunction<T>) -> bool {
        let mut changed = false;
        for pass in &group.passes {
            if self.progress.settled() || self.progress.stopped(self.env) {
                break;
            }
            let rewrote = self.run_isolated(group, pass.as_ref(), ir);
            changed |= rewrote;
            self.progress.note(rewrote);
        }
        changed
    }
    /// Runs one pass under its own snapshot.
    fn run_isolated(
        &mut self,
        group: &PassGroup<T>,
        pass: &dyn FunctionPass<T>,
        ir: &mut SsaFunction<T>,
    ) -> bool {
        let mut failed = false;
        let outcome =
            self.transaction
                .run_group(ir, |ir| match run_pass_with_repair(pass, ir, self.env) {
                    Ok(changed) => changed,
                    Err(_) => {
                        failed = true;
                        // Report the change so the self.transaction verifies and restores
                        // rather than keeping whatever a failed pass left behind.
                        true
                    }
                });
        if matches!(outcome, GroupOutcome::RolledBack) {
            self.env.ranges().invalidate();
        }
        if self.progress.stopped(self.env) {
            return false;
        }
        let changed = outcome.changed();
        self.progress.trace(group.name, pass.name(), changed);
        if !changed && matches!(outcome, GroupOutcome::RolledBack) {
            self.progress.rollback(
                group.name,
                Some(pass.name()),
                if failed {
                    RollbackReason::Failed
                } else {
                    RollbackReason::Rejected
                },
            );
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::{
        Error, Result,
        analysis::{budget::WorkBudget, context::AnalysisContext},
        events::NullListener,
        interrupt::Interrupt,
        ir::{instruction::SsaInstruction, ops::SsaOp},
        passes::ranges::ValueRanges,
        scheduling::{ModificationScope, PassEnv},
        testing::{MACHINE, MockTarget, const_i32_return},
    };

    /// Every answer the readers of one pipeline run were given, in order.
    type Reads = Arc<Mutex<Vec<Arc<ValueRanges>>>>;

    /// Reads the environment's range facts and records the answer, then
    /// panics when asked to - a pass that computed facts mid-group and never
    /// finished.
    struct ReadRanges {
        reads: Reads,
        panics: bool,
    }

    impl FunctionPass<MockTarget> for ReadRanges {
        fn name(&self) -> &'static str {
            "read-ranges"
        }

        fn modification_scope(&self) -> ModificationScope {
            ModificationScope::UsesOnly
        }

        fn run(
            &self,
            ssa: &mut SsaFunction<MockTarget>,
            env: &PassEnv<'_, MockTarget>,
        ) -> Result<bool> {
            let ranges = env.ranges().get(ssa).expect("a constant return converges");
            self.reads
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(ranges);
            assert!(!self.panics, "read the ranges, then failed");
            Ok(false)
        }
    }

    /// A reader recording into `reads`.
    fn reader(reads: &Reads, panics: bool) -> Box<dyn FunctionPass<MockTarget>> {
        Box::new(ReadRanges {
            reads: Arc::clone(reads),
            panics,
        })
    }

    /// The answers recorded so far.
    fn recorded(reads: &Reads) -> Vec<Arc<ValueRanges>> {
        reads.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Reports a rewrite for its first `limit` runs and nothing after, so a
    /// pipeline over it settles after a known number of iterations.
    ///
    /// The count lives in the pass rather than in the IR because every repair
    /// the driver runs strips nops: a pass that measured its own progress by
    /// what it left in the function would be undone between iterations and
    /// never settle, which is a property of this fixture and not of the
    /// pipeline.
    struct AddNop {
        limit: usize,
        runs: AtomicUsize,
    }

    impl AddNop {
        fn new(limit: usize) -> Self {
            Self {
                limit,
                runs: AtomicUsize::new(0),
            }
        }
    }

    impl FunctionPass<MockTarget> for AddNop {
        fn name(&self) -> &'static str {
            "add-nop"
        }

        fn modification_scope(&self) -> ModificationScope {
            ModificationScope::InstructionsOnly
        }

        fn run(
            &self,
            ssa: &mut SsaFunction<MockTarget>,
            _env: &PassEnv<'_, MockTarget>,
        ) -> Result<bool> {
            if self.runs.fetch_add(1, Ordering::Relaxed) >= self.limit {
                return Ok(false);
            }
            let Some(block) = ssa.block_mut(0) else {
                return Ok(false);
            };
            block
                .instructions_mut()
                .insert(0, SsaInstruction::synthetic(SsaOp::Nop));
            Ok(true)
        }
    }

    /// Leaves the function naming a variable that does not exist, which the
    /// verifier rejects and the transaction therefore restores.
    struct BreakIt;

    impl FunctionPass<MockTarget> for BreakIt {
        fn name(&self) -> &'static str {
            "break-it"
        }

        fn modification_scope(&self) -> ModificationScope {
            ModificationScope::InstructionsOnly
        }

        fn run(
            &self,
            ssa: &mut SsaFunction<MockTarget>,
            _env: &PassEnv<'_, MockTarget>,
        ) -> Result<bool> {
            let Some(block) = ssa.block_mut(0) else {
                return Ok(false);
            };
            block
                .instructions_mut()
                .insert(0, SsaInstruction::synthetic(SsaOp::Jump { target: 99 }));
            Ok(true)
        }
    }

    /// Never rewrites anything.
    struct Idle;

    impl FunctionPass<MockTarget> for Idle {
        fn name(&self) -> &'static str {
            "idle"
        }

        fn modification_scope(&self) -> ModificationScope {
            ModificationScope::UsesOnly
        }

        fn run(
            &self,
            _ssa: &mut SsaFunction<MockTarget>,
            _env: &PassEnv<'_, MockTarget>,
        ) -> Result<bool> {
            Ok(false)
        }
    }

    /// An environment whose pipelines run at most `value` rounds.
    fn rounds(value: usize) -> PassEnv<'static, MockTarget> {
        PassEnv::<MockTarget>::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::PipelineRounds, value))
    }

    /// Passes reading the ranges of a function nothing changed between them
    /// read one computation, within an iteration and across iterations; a pass
    /// that changes the function makes the next reader compute again.
    #[test]
    fn readers_share_the_ranges_until_a_pass_changes_the_function() {
        let reads = Reads::default();
        let spec = PipelineSpec::new(vec![PassGroup::new(
            "normalize",
            vec![
                reader(&reads, false),
                Box::new(AddNop::new(1)),
                reader(&reads, false),
            ],
        )]);
        let mut ssa = const_i32_return(1);
        let env = rounds(4);

        let report = FunctionPipeline::new(spec).run(&mut ssa, &env);
        assert_eq!(report.iterations, 2, "{report:?}");
        let reads = recorded(&reads);
        let [before, after, second] = reads.as_slice() else {
            panic!(
                "the second iteration stops once the nop pass has seen the function: {}",
                reads.len()
            );
        };
        assert!(
            !Arc::ptr_eq(before, after),
            "the nop pass changed the function between the two reads"
        );
        assert!(
            Arc::ptr_eq(after, second),
            "nothing changed the function between the iterations"
        );
    }

    /// A group rolled back after a pass computed the ranges of what the group
    /// had left leaves no trace of them: the next reader computes again, over
    /// the function as it was restored.
    #[test]
    fn a_rolled_back_group_leaves_no_ranges_behind() {
        let reads = Reads::default();
        let spec = PipelineSpec::new(vec![
            PassGroup::new(
                "breaks",
                vec![Box::new(AddNop::new(1)), reader(&reads, true)],
            ),
            PassGroup::new("reads", vec![reader(&reads, false)]),
        ]);
        let mut ssa = const_i32_return(1);
        let env = rounds(4);

        let report = FunctionPipeline::new(spec).run(&mut ssa, &env);
        assert!(!report.rollbacks.is_empty(), "{report:?}");
        let reads = recorded(&reads);
        let [mid_group, restored, ..] = reads.as_slice() else {
            panic!("both readers ran: {}", reads.len());
        };
        assert!(
            !Arc::ptr_eq(mid_group, restored),
            "the second group read the ranges of a function the first group's rollback discarded"
        );
    }

    /// Ranges an environment held before the run describe a function the
    /// caller may have changed since; the run starts without them.
    #[test]
    fn a_run_does_not_trust_ranges_from_before_it() {
        let reads = Reads::default();
        let spec = PipelineSpec::new(vec![PassGroup::new("reads", vec![reader(&reads, false)])]);
        let mut ssa = const_i32_return(1);
        let env = rounds(1);
        let held = env.ranges().get(&ssa).expect("a constant return converges");

        let _ = FunctionPipeline::new(spec).run(&mut ssa, &env);
        let reads = recorded(&reads);
        assert!(!Arc::ptr_eq(&held, &reads[0]));
    }

    #[test]
    fn a_pipeline_iterates_until_nothing_changes() {
        let spec = PipelineSpec::new(vec![PassGroup::new(
            "normalize",
            vec![Box::new(AddNop::new(3))],
        )]);
        let mut pipeline = FunctionPipeline::new(spec);
        let mut ssa = const_i32_return(1);
        let env = rounds(16);

        let report = pipeline.run(&mut ssa, &env);
        assert!(report.changed);
        assert!(report.converged);
        assert!(report.rollbacks.is_empty());
        // Three iterations add a nop, the fourth finds nothing left to do.
        assert_eq!(report.iterations, 4);
    }

    /// An interrupt that trips on its `at`th ask, counting from one, and
    /// stays tripped.
    struct TripAt {
        /// The ask it trips on.
        at: usize,
        /// Asks so far.
        asks: AtomicUsize,
    }

    impl Interrupt for TripAt {
        fn interrupted(&self) -> bool {
            self.asks.fetch_add(1, Ordering::Relaxed).saturating_add(1) >= self.at
        }
    }

    /// **An interrupted run stops at the pass it was asked before, and says
    /// so.** A function that would change for a hundred rounds, under an
    /// interrupt that trips on its `at`th ask: the run asks no more after
    /// that, reports itself interrupted and not settled, and runs fewer
    /// passes the earlier it is stopped. With no interrupt it runs to its
    /// cap.
    #[test]
    fn an_interrupted_run_stops_at_once_and_says_so() {
        let spec = || {
            PipelineSpec::new(vec![PassGroup::new(
                "normalize",
                vec![Box::new(AddNop::new(100))],
            )])
            .with_telemetry()
        };
        let mut passes = Vec::new();
        for at in 1..=12 {
            let probe = TripAt {
                at,
                asks: AtomicUsize::new(0),
            };
            let env = rounds(50).with_interrupt(&probe);
            let mut ssa = const_i32_return(1);
            let report = FunctionPipeline::new(spec()).run(&mut ssa, &env);
            assert!(report.interrupted, "tripped on ask {at}");
            assert!(!report.converged, "tripped on ask {at}");
            assert_eq!(
                probe.asks.load(Ordering::Relaxed),
                at,
                "nothing is asked after the ask that tripped"
            );
            passes.push(report.trace.len());
        }
        assert!(
            passes.windows(2).all(|pair| pair[0] <= pair[1]) && passes[0] < passes[11],
            "a later trip lets more passes run: {passes:?}"
        );
        let mut ssa = const_i32_return(1);
        let report = FunctionPipeline::new(spec()).run(&mut ssa, &rounds(50));
        assert!(!report.interrupted, "no interrupt, no stop");
        assert_eq!(report.iterations, 50, "it runs to its cap");
    }

    /// Adds a nop, then asks its environment's interrupt, as a long pass asks
    /// inside its own loops, and gives up with an error once it has tripped.
    struct StopsPartWay;

    impl FunctionPass<MockTarget> for StopsPartWay {
        fn name(&self) -> &'static str {
            "stops-part-way"
        }

        fn modification_scope(&self) -> ModificationScope {
            ModificationScope::InstructionsOnly
        }

        fn run(
            &self,
            ssa: &mut SsaFunction<MockTarget>,
            env: &PassEnv<'_, MockTarget>,
        ) -> Result<bool> {
            if let Some(block) = ssa.block_mut(0) {
                block
                    .instructions_mut()
                    .insert(0, SsaInstruction::synthetic(SsaOp::Nop));
            }
            if env.context().interrupted() {
                return Err(Error::new("stopped part way"));
            }
            Ok(true)
        }
    }

    /// **A pass its interrupt stops part way is no failure.** Under every
    /// group policy, a pass that asks the interrupt itself and gives up once
    /// it has tripped leaves the run reported stopped and not settled, with no
    /// rollback recorded against it: what it left is discarded with the run.
    #[test]
    fn a_pass_its_interrupt_stops_is_no_failure() {
        for policy in [
            GroupPolicy::Unguarded,
            GroupPolicy::PerPass,
            GroupPolicy::Group,
            GroupPolicy::GroupThenIsolate,
        ] {
            let spec = PipelineSpec::new(vec![
                PassGroup::new("stopping", vec![Box::new(StopsPartWay)]).with_policy(policy),
            ]);
            // The run asks before the group (the first ask) and the pass asks
            // inside it (the second), which trips.
            let probe = TripAt {
                at: 2,
                asks: AtomicUsize::new(0),
            };
            let env = rounds(4).with_interrupt(&probe);
            let mut ssa = const_i32_return(1);
            let report = FunctionPipeline::new(spec).run(&mut ssa, &env);
            assert!(
                report.interrupted && !report.converged,
                "{policy:?}: reported stopped"
            );
            assert!(
                report.rollbacks.is_empty(),
                "{policy:?}: no failure recorded, {:?}",
                report.rollbacks
            );
        }
    }

    #[test]
    fn a_run_that_hits_the_cap_says_so() {
        let spec = PipelineSpec::new(vec![PassGroup::new(
            "normalize",
            vec![Box::new(AddNop::new(100))],
        )]);
        let mut pipeline = FunctionPipeline::new(spec);
        let mut ssa = const_i32_return(1);
        let env = rounds(3);

        let report = pipeline.run(&mut ssa, &env);
        assert_eq!(report.iterations, 3);
        assert!(!report.converged, "it was still changing when it stopped");
        assert!(!report.clean());
    }

    #[test]
    fn a_discarded_rewrite_is_reported_without_a_second_run() {
        // The failure this exists for: the bad pass is rolled back, so the
        // pipeline looks like it simply had nothing to do. Reporting that the
        // work was discarded needs no re-run and so is always done.
        let spec = PipelineSpec::new(vec![PassGroup::new(
            "normalize",
            vec![Box::new(AddNop::new(4)), Box::new(BreakIt)],
        )]);
        let mut pipeline = FunctionPipeline::new(spec);
        let mut ssa = const_i32_return(1);
        let env = rounds(4);

        let report = pipeline.run(&mut ssa, &env);
        assert!(!report.rollbacks.is_empty(), "{report:?}");
        assert!(
            report
                .rollbacks
                .iter()
                .all(|rollback| rollback.pass.is_none()),
            "naming a pass costs a second run, which this policy does not pay: {report:?}"
        );
        assert!(
            report
                .rollbacks
                .iter()
                .all(|rollback| rollback.group == "normalize"),
            "{report:?}"
        );
        // The function is back as it was, which is what the snapshot is for.
        assert!(ssa.validate().is_ok(), "{ssa}");
    }

    #[test]
    fn diagnosing_a_discarded_rewrite_names_the_pass() {
        // Re-run with diagnostics, which is what a host does once its report
        // says something was discarded. Isolation both names the pass and
        // keeps the work of the one that was fine.
        let spec = PipelineSpec::new(vec![PassGroup::new(
            "normalize",
            vec![Box::new(AddNop::new(4)), Box::new(BreakIt)],
        )])
        .diagnose();
        let mut pipeline = FunctionPipeline::new(spec);
        let mut ssa = const_i32_return(1);
        let env = rounds(4);

        let report = pipeline.run(&mut ssa, &env);
        assert!(!report.rollbacks.is_empty(), "{report:?}");
        assert!(
            report
                .rollbacks
                .iter()
                .any(|rollback| rollback.pass == Some("break-it")),
            "isolation names the pass that broke it: {report:?}"
        );
        assert!(
            report
                .rollbacks
                .iter()
                .all(|rollback| rollback.group == "normalize"),
            "{report:?}"
        );
        // And the good pass's work survived.
        assert!(report.changed);
        assert!(ssa.validate().is_ok(), "{ssa}");
    }

    #[test]
    fn telemetry_records_one_entry_per_pass_run() {
        let spec = PipelineSpec::new(vec![PassGroup::new(
            "normalize",
            vec![Box::new(AddNop::new(1))],
        )])
        .with_telemetry();
        let mut pipeline = FunctionPipeline::new(spec);
        let mut ssa = const_i32_return(1);
        let env = rounds(8);

        let report = pipeline.run(&mut ssa, &env);
        assert_eq!(report.trace.len(), report.iterations);
        assert!(report.trace.iter().all(|run| run.pass == "add-nop"));
    }

    /// A group kept whole records what each of its passes did: the pass that
    /// rewrote the function, and not the one beside it that did nothing.
    #[test]
    fn telemetry_names_only_the_pass_that_rewrote() {
        let spec = PipelineSpec::new(vec![PassGroup::new(
            "normalize",
            vec![Box::new(Idle), Box::new(AddNop::new(1)), Box::new(Idle)],
        )])
        .with_telemetry();
        let mut pipeline = FunctionPipeline::new(spec);
        let mut ssa = const_i32_return(1);
        let env = rounds(8);

        let report = pipeline.run(&mut ssa, &env);
        let changed: Vec<(&str, usize)> = report
            .trace
            .iter()
            .filter(|run| run.changed)
            .map(|run| (run.pass, run.iteration))
            .collect();
        assert_eq!(changed, [("add-nop", 1)], "{report:?}");
        assert_eq!(report.trace.len(), 5, "{report:?}");
    }

    /// A pass that has seen the function as it stands is not asked again: the
    /// last rewrite came from the first pass, so the second iteration stops
    /// as soon as that pass has seen what it left. Under every policy.
    #[test]
    fn a_run_stops_once_every_pass_has_seen_the_function() {
        for policy in [
            GroupPolicy::Group,
            GroupPolicy::GroupThenIsolate,
            GroupPolicy::PerPass,
            GroupPolicy::Unguarded,
        ] {
            let spec = PipelineSpec::new(vec![
                PassGroup::new("first", vec![Box::new(AddNop::new(1))]).with_policy(policy),
                PassGroup::new("rest", vec![Box::new(Idle), Box::new(Idle)]).with_policy(policy),
            ])
            .with_telemetry();
            let mut ssa = const_i32_return(1);
            let env = rounds(8);

            let report = FunctionPipeline::new(spec).run(&mut ssa, &env);
            let runs: Vec<(&str, usize)> = report
                .trace
                .iter()
                .map(|run| (run.pass, run.iteration))
                .collect();
            assert_eq!(
                runs,
                [("add-nop", 1), ("idle", 1), ("idle", 1), ("add-nop", 2)],
                "{policy:?}"
            );
            assert!(report.converged && report.changed, "{report:?}");
            assert_eq!(report.iterations, 2, "{report:?}");
        }
    }

    /// The last pass rewriting leaves nothing to skip: every other pass has
    /// to see what it left, and it has to see that nobody changed it.
    #[test]
    fn a_rewrite_by_the_last_pass_is_seen_by_every_pass() {
        let spec = PipelineSpec::new(vec![PassGroup::new(
            "normalize",
            vec![Box::new(Idle), Box::new(AddNop::new(1))],
        )])
        .with_telemetry();
        let mut ssa = const_i32_return(1);
        let env = rounds(8);

        let report = FunctionPipeline::new(spec).run(&mut ssa, &env);
        assert_eq!(report.trace.len(), 4, "{report:?}");
    }

    /// A group the verifier rejects counts as having changed nothing, as the
    /// round it happens in always has: the run ends, rather than repeating a
    /// rejection until the cap.
    #[test]
    fn a_rejected_group_does_not_keep_the_run_going() {
        let spec = PipelineSpec::new(vec![
            PassGroup::new("breaks", vec![Box::new(BreakIt)]),
            PassGroup::new("rest", vec![Box::new(Idle)]),
        ])
        .with_telemetry();
        let mut ssa = const_i32_return(1);
        let env = rounds(8);

        let report = FunctionPipeline::new(spec).run(&mut ssa, &env);
        assert_eq!(report.iterations, 1, "{report:?}");
        assert!(report.converged, "{report:?}");
        assert_eq!(report.rollbacks.len(), 1, "{report:?}");
    }
}
