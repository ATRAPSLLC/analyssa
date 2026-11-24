//! Transactional pass execution - run a group of passes under one reusable
//! snapshot, verify once, and roll back on failure.
//!
//! # Why a group and not a pass
//!
//! Snapshotting before *every* pass so a verifier rejection can be undone costs
//! one whole-IR deep copy per pass per fixpoint iteration, and the terminating
//! iteration - by definition the one where nothing changes - pays all of them
//! for no output.
//!
//! Running a whole group against a single snapshot and verifying once at the end
//! costs one copy instead of N. On the rare failure the snapshot is restored and
//! the group re-runs pass-by-pass, so a single bad pass is still isolated and the
//! innocent ones still land: the failure path is no worse than per-pass
//! snapshotting, and the common path is N× cheaper.
//!
//! The trade is only sound because rollbacks are rare: a group that rolls back
//! re-runs every pass in it, so a pipeline whose passes routinely produce
//! invalid IR pays for the group and then for each pass again, and per-pass
//! snapshotting would be the better choice for it.
//!
//! # Why the buffer is owned rather than thread-local
//!
//! The snapshot buffer is reused across functions, which is where the allocation
//! saving comes from - [`SsaFunction`]'s `clone_from` reuses capacity all the way
//! into the per-block instruction vectors and per-variable use lists. A
//! thread-local cannot hold a `SsaFunction<T>` without erasing `T`, so the buffer
//! is owned by [`PassTransaction`] and the caller keeps one per worker (rayon's
//! `for_each_init` supplies exactly that shape).

use std::{
    convert::Infallible,
    mem,
    panic::{AssertUnwindSafe, catch_unwind},
};

use log::warn;

use crate::{
    analysis::{SsaVerifier, VerifierError, VerifyLevel},
    ir::function::SsaFunction,
    target::Target,
};

/// Verification level used after a group runs.
///
/// [`VerifyLevel::Quick`] is an O(n) tripwire, enough to catch a pass that
/// corrupted the graph - including a dangling read, a definition deleted while
/// an instruction or phi operand still names it. That one has to be here and
/// not only at Standard: a result the release tripwire accepts and the debug one
/// rejects makes the two profiles recover the same bytes differently. Debug and
/// test builds promote it to [`VerifyLevel::Standard`] so a failing suite
/// localizes the offending pass rather than surfacing it at the next boundary
/// check.
///
/// The same level in every build, because this one decides **output**.
///
/// A group the verifier rejects is rolled back, so a check that runs in one
/// profile and not another makes the optimized IR - and every listing, hash and
/// recovered source derived from it - a function of how the compiler was
/// invoked. A level that varied by profile would keep a group in one build that
/// the other rolls back, and the two agreeing on any given corpus would be an
/// accident nothing checks. `scripts/check-conventions.sh` enforces the rule
/// over the whole crate.
///
/// `Standard` rather than `Quick` because rolling a bad group back is the safe
/// side of the choice, and the cost is not the reason it was ever split:
/// measured over the recorded-baseline suite in release, uniform `Standard`
/// runs within noise of the split version (18.1s against 17.8s), and no group
/// in that corpus is rejected by a `Standard`-only check - the release output
/// is byte-identical with it turned on.
///
/// Deliberately **not** [`VerifyLevel::Full`]. `Full` adds a dominance check
/// that does not hold for lifted native code: on real binaries it rejects the
/// output of five unrelated passes (dead-code, GVN, LICM, copy propagation,
/// block merging) hundreds of times. A defect that uniform across unrelated
/// passes is structural rather than per-pass - the likely cause is that
/// dominators are computed from the CFG entry while reachability also admits
/// exception-handler entries, so a block reached only via a handler edge is
/// dominated by nothing. Turning it on before that is understood would strip
/// optimization from every build.
const fn post_group_verify_level() -> VerifyLevel {
    VerifyLevel::Standard
}

/// How a transactional group ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupOutcome {
    /// No pass reported a rewrite. The IR is untouched and was not verified -
    /// a pass that changed nothing cannot have corrupted valid IR.
    Unchanged,
    /// The group rewrote the IR and the result verified.
    Applied,
    /// The group panicked or produced invalid IR; the pre-group state was
    /// restored. The caller should re-run the passes individually to isolate
    /// the culprit.
    RolledBack,
}

impl GroupOutcome {
    /// Returns `true` when the IR was rewritten and kept.
    ///
    /// Only [`Applied`](Self::Applied) counts. [`Unchanged`](Self::Unchanged)
    /// made no edits and [`RolledBack`](Self::RolledBack) discarded whatever it
    /// made, so from a caller's fixpoint-loop perspective neither advanced the
    /// program - this is the signal that another iteration may be worthwhile.
    ///
    /// # Returns
    ///
    /// `true` if the group's rewrite was verified and retained.
    #[must_use]
    pub fn changed(self) -> bool {
        matches!(self, Self::Applied)
    }
}

/// How a [`PassTransaction::run_guarded`] run ended.
///
/// Every variant other than [`Unchanged`](Self::Unchanged) and
/// [`Applied`](Self::Applied) has restored the pre-group function.
#[derive(Debug)]
pub enum Guarded<E> {
    /// Nothing changed; the function was not verified.
    Unchanged,
    /// The function changed and was kept (verified, when a level was given).
    Applied,
    /// The group returned this error; the function was restored.
    Failed(E),
    /// The group panicked with this message; the function was restored.
    Panicked(String),
    /// The changed function failed verification with these errors; the
    /// function was restored.
    Invalid(Vec<VerifierError>),
}

/// Which results [`PassTransaction::run_guarded`] verifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verify {
    /// Accept every result unverified.
    Never,
    /// Verify a changed result at this level; an unchanged one is accepted
    /// unverified, since a group that changed nothing cannot have corrupted
    /// valid IR.
    Changed(VerifyLevel),
    /// Verify every result at this level, changed or not - for a caller that
    /// does not trust a group's report of having changed nothing.
    Always(VerifyLevel),
}

/// Reusable snapshot buffer for transactional pass execution.
///
/// Hold one per worker thread and reuse it across functions; the capacity it
/// accumulates is the point.
#[derive(Debug)]
pub struct PassTransaction<T: Target> {
    /// Reused snapshot buffer. Starts empty and grows to the largest function
    /// this worker has seen.
    snapshot: SsaFunction<T>,
}

impl<T: Target> Default for PassTransaction<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Target> PassTransaction<T> {
    /// Creates a transaction with an empty buffer.
    ///
    /// The snapshot starts at zero capacity and grows on its first
    /// [`run_group`](Self::run_group) to fit the function being copied;
    /// thereafter `clone_from` reuses that capacity. A fresh transaction
    /// therefore pays one allocation on its first real function and none after,
    /// which is why a worker holds one for its whole run rather than making one
    /// per function.
    ///
    /// # Returns
    ///
    /// A transaction whose reusable snapshot buffer is empty.
    #[must_use]
    pub fn new() -> Self {
        Self {
            snapshot: SsaFunction::with_capacity(0, 0, 0, 0),
        }
    }

    /// Runs `group` against `ssa` under one snapshot.
    ///
    /// `group` returns whether any pass in it rewrote the IR. It may panic; the
    /// panic is caught and treated as a rollback, so one bad pass cannot abandon
    /// the whole function.
    ///
    /// # Arguments
    ///
    /// * `ssa` - Function to transform in place.
    /// * `group` - Runs the passes, returning whether anything changed.
    ///
    /// # Returns
    ///
    /// How the group ended; see [`GroupOutcome`].
    ///
    /// # Events are not rolled back
    ///
    /// A rollback restores the IR but **not** the event log: events a passing
    /// group recorded before the rollback remain. [`EventLog`] is append-only by
    /// construction - it takes `&self` so passes can record concurrently - so
    /// there is no watermark to truncate to. Nothing about the emitted IR is
    /// affected; a host reading events for diagnostics should treat them as "what
    /// was attempted", not "what was kept", and correlate against the returned
    /// [`GroupOutcome`].
    ///
    /// [`EventLog`]: crate::events::EventLog
    ///
    /// `#[inline]` because this is a cross-crate generic on the normalization
    /// hot path and neither crate builds with LTO; without it the pass loop
    /// stays behind a call boundary.
    #[inline]
    pub fn run_group<F>(&mut self, ssa: &mut SsaFunction<T>, group: F) -> GroupOutcome
    where
        F: FnOnce(&mut SsaFunction<T>) -> bool,
    {
        let guarded = self.run_guarded::<Infallible, _>(
            ssa,
            Verify::Changed(post_group_verify_level()),
            |ssa| Ok(group(ssa)),
        );
        match guarded {
            Guarded::Unchanged => GroupOutcome::Unchanged,
            Guarded::Applied => GroupOutcome::Applied,
            Guarded::Failed(never) => match never {},
            Guarded::Panicked(detail) => {
                // Surface the panic rather than discarding it. Rolling back
                // silently makes a panicking pass look like a pass that simply
                // did nothing, which is the hardest possible failure to trace.
                warn!("pass group panicked ({detail}); rolling back");
                GroupOutcome::RolledBack
            }
            Guarded::Invalid(_) => {
                warn!("pass group left the SSA verifier-invalid; rolling back");
                GroupOutcome::RolledBack
            }
        }
    }

    /// Runs a fallible `group` against `ssa` under one snapshot, rolling back
    /// on an error, a panic, or (as `verify` asks) a verifier rejection.
    ///
    /// This is the one implementation of the transaction: [`run_group`]
    /// wraps it for infallible groups, and the scheduler runs every per-method
    /// pass through it, so a pass that edits and then fails leaves the function
    /// exactly as it found it whichever driver ran it.
    ///
    /// The group's `Ok(changed)` is combined with
    /// [`SsaFunction::take_edit_dirty`], so a group that edited through a
    /// checked session and then reported "unchanged" is still treated as
    /// changed and still verified. The flag is cleared before the snapshot is
    /// taken, so only this group's edits count.
    ///
    /// # Arguments
    ///
    /// * `ssa` - Function to transform in place.
    /// * `verify` - Which results to verify, and at what level.
    /// * `group` - Runs the passes, returning whether anything changed.
    ///
    /// # Returns
    ///
    /// How the group ended; every variant other than [`Guarded::Unchanged`]
    /// and [`Guarded::Applied`] has restored the pre-group function.
    ///
    /// [`run_group`]: Self::run_group
    pub fn run_guarded<E, F>(
        &mut self,
        ssa: &mut SsaFunction<T>,
        verify: Verify,
        group: F,
    ) -> Guarded<E>
    where
        F: FnOnce(&mut SsaFunction<T>) -> Result<bool, E>,
    {
        // A flag left set by an earlier, unrelated edit would make this group
        // look changed; clear it so the snapshot and the check below see only
        // what the group does.
        let _ = ssa.take_edit_dirty();
        let _ = ssa.take_edit_failure();
        // Capacity-reusing copy: `clone_from` propagates into the per-block and
        // per-variable buffers rather than reallocating them.
        self.snapshot.clone_from(ssa);

        let guarded = match catch_unwind(AssertUnwindSafe(|| group(ssa))) {
            Ok(Ok(changed)) => {
                // `changed` is what the group *claims*. A pass whose edit
                // session ran under `SsaRollbackPolicy::Never` and whose
                // boundary repair failed has left its edits applied, and a pass
                // can report "unchanged" in exactly that case - which would skip
                // both verification and rollback, keeping damaged IR unchecked.
                // `take_edit_dirty` reports what actually happened to the
                // function, so the claim cannot suppress the check.
                let changed = ssa.take_edit_dirty() || changed;
                let level = match verify {
                    Verify::Always(level) => Some(level),
                    Verify::Changed(level) if changed => Some(level),
                    Verify::Changed(_) | Verify::Never => None,
                };
                let errors = level
                    .map(|level| SsaVerifier::new(ssa).verify(level))
                    .unwrap_or_default();
                if !errors.is_empty() {
                    Guarded::Invalid(errors)
                } else if changed {
                    return Guarded::Applied;
                } else {
                    return Guarded::Unchanged;
                }
            }
            Ok(Err(error)) => Guarded::Failed(error),
            Err(payload) => Guarded::Panicked(
                payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string()),
            ),
        };

        // The group failed, panicked, or left the IR invalid. Swap rather than
        // assign so the rejected IR becomes the reusable buffer and neither
        // allocation is freed.
        mem::swap(ssa, &mut self.snapshot);
        guarded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{function::SsaEditOptions, instruction::SsaInstruction, ops::SsaOp},
        testing::{self, MockTarget},
    };

    /// Corrupts `ssa` in a way the verifier rejects: appending an instruction
    /// after the block's terminator, so the terminator is no longer last.
    fn corrupt(ssa: &mut SsaFunction<MockTarget>) -> bool {
        if let Some(block) = ssa.block_mut(0) {
            block.add_instruction(SsaInstruction::new((), SsaOp::Nop));
        }
        true
    }

    /// A group that rewrites nothing is accepted without verification and
    /// leaves the IR untouched.
    #[test]
    fn an_unchanged_group_is_accepted_without_verifying() {
        let mut ssa = testing::const_i32_return(7);
        let before = ssa.block_count();
        let mut transaction = PassTransaction::<MockTarget>::new();

        let outcome = transaction.run_group(&mut ssa, |_| false);

        assert_eq!(outcome, GroupOutcome::Unchanged);
        assert!(!outcome.changed());
        assert_eq!(ssa.block_count(), before);
        assert_eq!(ssa.validate(), Ok(()));
    }

    /// A group that rewrites the IR validly is kept.
    #[test]
    fn a_valid_group_is_applied() {
        let mut ssa = testing::const_i32_return(7);
        let mut transaction = PassTransaction::<MockTarget>::new();

        // Reports a rewrite without corrupting anything. What is under test is
        // the transaction's handling of a changed-and-valid group, not any
        // particular pass's edit.
        let outcome = transaction.run_group(&mut ssa, |_| true);

        assert_eq!(outcome, GroupOutcome::Applied);
        assert!(outcome.changed());
        assert_eq!(ssa.validate(), Ok(()));
    }

    /// A pass that mutates through a checked edit and then reports "unchanged"
    /// must not escape verification.
    ///
    /// `SsaEditOptions::new()` defaults to `SsaRollbackPolicy::Never`, so the
    /// edits stay applied when the session fails. A pass that returns `false`
    /// in that case would take the `Unchanged` path and keep the damaged IR
    /// unverified if the transaction trusted the claim instead of the
    /// function's own edit flag.
    #[test]
    fn a_pass_that_mutates_then_claims_unchanged_is_still_verified() {
        let mut ssa = testing::const_i32_return(7);
        let original_blocks = ssa.block_count();
        let mut transaction = PassTransaction::<MockTarget>::new();

        let outcome = transaction.run_group(&mut ssa, |ssa| {
            let _ = ssa.edit(SsaEditOptions::new(), |editor| {
                editor.insert_before_terminator(0, SsaInstruction::new((), SsaOp::Nop))?;
                Ok(())
            });
            // The lie under test.
            false
        });

        assert_ne!(
            outcome,
            GroupOutcome::Unchanged,
            "a mutated function must not be reported as unchanged"
        );
        assert_eq!(ssa.block_count(), original_blocks);
        assert_eq!(ssa.validate(), Ok(()));
    }

    /// A group that corrupts the IR is rolled back to its pre-group state.
    #[test]
    fn an_invalid_group_is_rolled_back() {
        let mut ssa = testing::const_i32_return(7);
        let original_blocks = ssa.block_count();
        let mut transaction = PassTransaction::<MockTarget>::new();

        let outcome = transaction.run_group(&mut ssa, corrupt);

        assert_eq!(outcome, GroupOutcome::RolledBack);
        assert!(!outcome.changed());
        assert_eq!(
            ssa.block_count(),
            original_blocks,
            "the pre-group IR was restored"
        );
        assert_eq!(ssa.validate(), Ok(()), "and it is still valid");
    }

    /// A panicking pass is caught and rolled back rather than abandoning the
    /// function.
    #[test]
    fn a_panicking_group_is_rolled_back() {
        let mut ssa = testing::const_i32_return(7);
        let original_blocks = ssa.block_count();
        let mut transaction = PassTransaction::<MockTarget>::new();

        // The panic hook is process-global and other tests run concurrently,
        // so it is left alone: the expected panic message appears in the
        // captured output of this test only.
        let outcome = transaction.run_group(&mut ssa, |_| {
            panic!("pass exploded");
        });

        assert_eq!(outcome, GroupOutcome::RolledBack);
        assert_eq!(ssa.block_count(), original_blocks);
        assert_eq!(ssa.validate(), Ok(()));
    }

    /// The buffer is reused across runs, which is where the allocation saving
    /// comes from: a second run on a larger function must still be correct.
    #[test]
    fn the_buffer_is_reusable_across_functions() {
        let mut transaction = PassTransaction::<MockTarget>::new();

        let mut small = testing::const_i32_return(1);
        assert_eq!(
            transaction.run_group(&mut small, |_| false),
            GroupOutcome::Unchanged
        );

        let mut larger = testing::diamond_phi_fixture();
        let blocks = larger.block_count();
        let outcome = transaction.run_group(&mut larger, corrupt);

        assert_eq!(outcome, GroupOutcome::RolledBack);
        assert_eq!(larger.block_count(), blocks, "restored after reuse");
        assert_eq!(larger.validate(), Ok(()));
    }
}
