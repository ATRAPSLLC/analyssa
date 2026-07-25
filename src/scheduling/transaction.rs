//! Transactional pass execution — run a group of passes under one reusable
//! snapshot, verify once, and roll back on failure.
//!
//! # Why a group and not a pass
//!
//! Snapshotting before *every* pass so a verifier rejection can be undone costs
//! one whole-IR deep copy per pass per fixpoint iteration, and the terminating
//! iteration — by definition the one where nothing changes — pays all of them
//! for no output.
//!
//! Running a whole group against a single snapshot and verifying once at the end
//! costs one copy instead of N. On the rare failure the snapshot is restored and
//! the group re-runs pass-by-pass, so a single bad pass is still isolated and the
//! innocent ones still land: the failure path is no worse than per-pass
//! snapshotting, and the common path is N× cheaper.
//!
//! The trade is only sound because rollbacks are rare. Measured at **0 across
//! ~36k functions** of a native corpus once a phi-pruning defect was fixed —
//! before that fix one binary alone rolled back 1,632 times, and this would have
//! been a poor trade.
//!
//! # Why the buffer is owned rather than thread-local
//!
//! The snapshot buffer is reused across functions, which is where the allocation
//! saving comes from — [`SsaFunction`]'s `clone_from` reuses capacity all the way
//! into the per-block instruction vectors and per-variable use lists. A
//! thread-local cannot hold a `SsaFunction<T>` without erasing `T`, so the buffer
//! is owned by [`PassTransaction`] and the caller keeps one per worker (rayon's
//! `for_each_init` supplies exactly that shape).

use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::{
    analysis::{SsaVerifier, VerifyLevel},
    ir::function::SsaFunction,
    target::Target,
};

/// Verification level used after a group runs.
///
/// [`VerifyLevel::Quick`] is an O(n) tripwire, enough to catch a pass that
/// corrupted the graph. Debug and test builds promote it to
/// [`VerifyLevel::Standard`] so a failing suite localizes the offending pass
/// rather than surfacing it at the next boundary check.
///
/// Deliberately **not** [`VerifyLevel::Full`], even in debug. `Full` adds a
/// dominance check that has never held for lifted native code: enabling it was
/// measured to reject hundreds of passes across five unrelated ones
/// (dead-code, GVN, LICM, copy propagation, block merging) on real binaries. A
/// defect that uniform across unrelated passes is structural rather than
/// per-pass — the likely cause is that dominators are computed from the CFG
/// entry while reachability also admits exception-handler entries, so a block
/// reached only via a handler edge is dominated by nothing. Turning it on
/// before that is understood would strip optimization from debug builds and
/// make the suite disagree with release.
const fn post_group_verify_level() -> VerifyLevel {
    if cfg!(debug_assertions) {
        VerifyLevel::Standard
    } else {
        VerifyLevel::Quick
    }
}

/// How a transactional group ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupOutcome {
    /// No pass reported a rewrite. The IR is untouched and was not verified —
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
    #[must_use]
    pub fn changed(self) -> bool {
        matches!(self, Self::Applied)
    }
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
    /// * `ssa` — Function to transform in place.
    /// * `group` — Runs the passes, returning whether anything changed.
    ///
    /// # Returns
    ///
    /// How the group ended; see [`GroupOutcome`].
    ///
    /// # Events are not rolled back
    ///
    /// A rollback restores the IR but **not** the event log: events a passing
    /// group recorded before the rollback remain. [`EventLog`] is append-only by
    /// construction — it takes `&self` so passes can record concurrently — so
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
        // Capacity-reusing copy: `clone_from` propagates into the per-block and
        // per-variable buffers rather than reallocating them.
        self.snapshot.clone_from(ssa);

        let outcome = catch_unwind(AssertUnwindSafe(|| group(ssa)));

        match outcome {
            Ok(changed) => {
                if !changed {
                    return GroupOutcome::Unchanged;
                }
                if SsaVerifier::new(ssa)
                    .verify(post_group_verify_level())
                    .is_empty()
                {
                    return GroupOutcome::Applied;
                }
                log::warn!("pass group left the SSA verifier-invalid; rolling back");
            }
            Err(payload) => {
                // Surface the panic rather than discarding it. Rolling back
                // silently makes a panicking pass look like a pass that simply
                // did nothing, which is the hardest possible failure to trace.
                let detail = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                log::warn!("pass group panicked ({detail}); rolling back");
            }
        }

        // Either a pass panicked or the group left the IR invalid. Swap rather
        // than assign so the rejected IR becomes the reusable buffer and neither
        // allocation is freed.
        std::mem::swap(ssa, &mut self.snapshot);
        GroupOutcome::RolledBack
    }

    /// Runs one pass under its own snapshot, for isolating a culprit after a
    /// group rolled back.
    ///
    /// # Arguments
    ///
    /// * `ssa` — Function to transform in place.
    /// * `pass` — Runs the pass, returning whether it rewrote anything.
    ///
    /// # Returns
    ///
    /// How the pass ended.
    #[inline]
    pub fn run_one<F>(&mut self, ssa: &mut SsaFunction<T>, pass: F) -> GroupOutcome
    where
        F: FnOnce(&mut SsaFunction<T>) -> bool,
    {
        self.run_group(ssa, pass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{instruction::SsaInstruction, ops::SsaOp},
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

        // Silence the panic hook for the duration so the suite output stays
        // readable; the panic is expected.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = transaction.run_group(&mut ssa, |_| {
            panic!("pass exploded");
        });
        std::panic::set_hook(hook);

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
