//! Control-flow queries and branch-target rewriting for [`SsaOp`].
//!
//! Terminator recognition, successor enumeration, and the target remapping the
//! CFG-shaping passes (block merging, jump threading, control-flow
//! simplification) use when they splice blocks.

use std::convert::Infallible;

use crate::{
    ir::{
        ops::{def::SsaOp, execution::Computation, kinds::KindedVecData, vector::Contraction},
        variable::SsaVarId,
    },
    target::Target,
};

impl<T: Target> SsaOp<T> {
    /// Builds an integer conversion that cannot overflow.
    ///
    /// A representation change - truncating a register to a subregister,
    /// widening one, relabelling a sign - never traps, so the only
    /// [`SsaOp::IntConv`] a lifter builds for one carries no overflow check.
    /// Naming that construction here keeps every caller from restating the same
    /// five fields, and keeps the question "which conversions does the lift
    /// emit?" answerable in one place.
    ///
    /// `unsigned` selects zero- over sign-extension; it describes the
    /// *conversion*, not the operand.
    #[must_use]
    pub fn int_conv(dest: SsaVarId, operand: SsaVarId, target: T::Type, unsigned: bool) -> Self {
        Self::IntConv {
            dest,
            operand,
            target,
            overflow_check: false,
            unsigned,
        }
    }

    /// Builds the contraction `kind` defining `outputs` from `inputs`
    /// ([`SsaOp::Contract`]).
    #[must_use]
    pub fn contract(kind: Contraction, outputs: Vec<SsaVarId>, inputs: Vec<SsaVarId>) -> Self {
        Self::Contract(Box::new(KindedVecData {
            kind,
            outputs,
            inputs,
        }))
    }

    /// Builds the computation `kind` defining `outputs` from `inputs`
    /// ([`SsaOp::Compute`]).
    #[must_use]
    pub fn compute(kind: Computation, outputs: Vec<SsaVarId>, inputs: Vec<SsaVarId>) -> Self {
        Self::Compute(Box::new(KindedVecData {
            kind,
            outputs,
            inputs,
        }))
    }

    /// Returns `true` if this operation is a terminator (ends a basic block).
    #[must_use]
    pub const fn is_terminator(&self) -> bool {
        matches!(
            self,
            Self::Jump { .. }
                | Self::Branch { .. }
                | Self::BranchCmp { .. }
                | Self::BranchFlags { .. }
                | Self::IndirectBranch { .. }
                | Self::Switch { .. }
                | Self::Return { .. }
                | Self::Throw { .. }
                | Self::Rethrow
                | Self::Leave { .. }
                | Self::EndFinally
                | Self::EndFilter { .. }
                | Self::InterruptReturn
                | Self::Unreachable
        )
    }

    /// Returns the successor block indices for this operation.
    ///
    /// For control flow operations (terminators), this returns the indices of
    /// all possible successor blocks:
    /// - `Jump` and `Leave`: single target block
    /// - `Branch`: true and false target blocks
    /// - `Switch`: all case targets plus the default target
    ///
    /// For non-terminator operations, returns an empty vector.
    ///
    /// # Returns
    ///
    /// A vector of successor block indices. Empty for non-branching operations.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{SsaOp, SsaVarId}};
    ///
    /// let var = SsaVarId::from_index(0);
    /// let op = SsaOp::<MockTarget>::Branch {
    ///     condition: var,
    ///     true_target: 1,
    ///     false_target: 2,
    /// };
    /// assert_eq!(op.successors(), vec![1, 2]);
    /// ```
    #[must_use]
    pub fn successors(&self) -> Vec<usize> {
        match self {
            Self::Jump { target } | Self::Leave { target } => vec![*target],
            Self::Branch {
                true_target,
                false_target,
                ..
            }
            | Self::BranchCmp {
                true_target,
                false_target,
                ..
            }
            | Self::BranchFlags {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],
            Self::Switch {
                targets, default, ..
            } => {
                let mut succs = targets.clone();
                succs.push(*default);
                succs
            }
            Self::IndirectBranch {
                resolved_targets, ..
            } => resolved_targets.clone(),
            // Return, Throw, Rethrow, EndFinally, EndFilter have no successors
            _ => vec![],
        }
    }

    /// The `(true_target, false_target)` of a two-way conditional terminator.
    ///
    /// **There are three of them**, and that is the whole reason this exists.
    /// `Branch` names a condition variable, `BranchCmp` states a relation
    /// inline, and `BranchFlags` tests a bit of a flags value; they are the same
    /// shape - one predicate, two arms - and a consumer that reasons about "a
    /// conditional branch" wants all three or none. Enumerating them by hand
    /// invites handling two of the three, and the omission is silent: the
    /// missing kind falls to a conservative default and simply stops being
    /// simplified.
    ///
    /// `None` for every other op, including `Switch` and `IndirectBranch` -
    /// those have arms but not *two* of them, and a caller reasoning about a
    /// predicate's two outcomes cannot use them. The arms are returned as they
    /// are written, so a branch naming one block twice comes back as
    /// `Some((n, n))`: whether that is a degenerate branch to collapse or a
    /// condition that decides nothing is the caller's question, and both callers
    /// want a different answer to it. Callers wanting the successor *set* want
    /// [`SsaOp::successors`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{SsaOp, ops::native::FlagCondition, variable::SsaVarId}};
    ///
    /// let op: SsaOp<MockTarget> = SsaOp::BranchFlags {
    ///     flags: SsaVarId::from_index(0),
    ///     condition: FlagCondition::Zero,
    ///     true_target: 1,
    ///     false_target: 2,
    /// };
    /// assert_eq!(op.two_way_targets(), Some((1, 2)));
    ///
    /// let unconditional: SsaOp<MockTarget> = SsaOp::Jump { target: 1 };
    /// assert_eq!(unconditional.two_way_targets(), None);
    /// ```
    #[must_use]
    pub fn two_way_targets(&self) -> Option<(usize, usize)> {
        match self {
            Self::Branch {
                true_target,
                false_target,
                ..
            }
            | Self::BranchCmp {
                true_target,
                false_target,
                ..
            }
            | Self::BranchFlags {
                true_target,
                false_target,
                ..
            } => Some((*true_target, *false_target)),
            _ => None,
        }
    }

    /// Calls `f` for every successor block index of this operation.
    ///
    /// Allocation-free equivalent of iterating [`SsaOp::successors`]; preferred
    /// in CFG-construction and traversal hot paths.
    pub fn for_each_successor<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        match self.try_for_each_successor(|target| {
            f(target);
            Ok::<(), Infallible>(())
        }) {
            Ok(()) => (),
            Err(never) => match never {},
        }
    }

    /// Visits successors in operand order, stopping immediately on a visitor error.
    ///
    /// # Errors
    ///
    /// Returns the first visitor error without invoking it for any remaining target.
    pub fn try_for_each_successor<E>(
        &self,
        mut f: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(), E> {
        match self {
            Self::Jump { target } | Self::Leave { target } => f(*target)?,
            Self::Branch {
                true_target,
                false_target,
                ..
            }
            | Self::BranchCmp {
                true_target,
                false_target,
                ..
            }
            | Self::BranchFlags {
                true_target,
                false_target,
                ..
            } => {
                f(*true_target)?;
                f(*false_target)?;
            }
            Self::Switch {
                targets, default, ..
            } => {
                for target in targets {
                    f(*target)?;
                }
                f(*default)?;
            }
            Self::IndirectBranch {
                resolved_targets, ..
            } => {
                for target in resolved_targets {
                    f(*target)?;
                }
            }
            // Return, Throw, Rethrow, EndFinally, EndFilter have no successors
            _ => {}
        }
        Ok(())
    }

    /// Returns `true` if `block` is a successor of this operation.
    ///
    /// Allocation-free and short-circuiting; preferred over
    /// `self.successors().contains(&block)` in hot paths.
    #[must_use]
    pub fn has_successor(&self, block: usize) -> bool {
        match self {
            Self::Jump { target } | Self::Leave { target } => *target == block,
            Self::Branch {
                true_target,
                false_target,
                ..
            }
            | Self::BranchCmp {
                true_target,
                false_target,
                ..
            }
            | Self::BranchFlags {
                true_target,
                false_target,
                ..
            } => *true_target == block || *false_target == block,
            Self::Switch {
                targets, default, ..
            } => *default == block || targets.contains(&block),
            Self::IndirectBranch {
                resolved_targets, ..
            } => resolved_targets.contains(&block),
            _ => false,
        }
    }

    /// Redirects control flow targets from `old_target` to `new_target`.
    ///
    /// This method modifies branch/jump targets in-place. It handles all control
    /// flow operations: `Jump`, `Leave`, `Branch`, `BranchCmp`, and `Switch`.
    ///
    /// # Arguments
    ///
    /// * `old_target` - The block index to redirect from
    /// * `new_target` - The block index to redirect to
    ///
    /// # Returns
    ///
    /// `true` if any target was changed, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{MockTarget, ir::{SsaOp, SsaVarId}};
    ///
    /// let var = SsaVarId::from_index(0);
    /// let mut op = SsaOp::<MockTarget>::Branch {
    ///     condition: var,
    ///     true_target: 2,
    ///     false_target: 3,
    /// };
    /// // Redirect all jumps to block 2 to instead go to block 5
    /// if op.redirect_target(2, 5) {
    ///     println!("Target redirected");
    /// }
    /// assert_eq!(op.successors(), vec![5, 3]);
    /// ```
    pub fn redirect_target(&mut self, old_target: usize, new_target: usize) -> bool {
        if old_target == new_target {
            return false;
        }
        let mut changed = false;
        self.for_each_target_mut(|target| {
            if *target == old_target {
                *target = new_target;
                changed = true;
            }
        });
        changed
    }

    /// Remaps branch target block indices using the provided mapping function.
    ///
    /// This is used to translate RVA-based targets (from CIL instructions) to
    /// sequential block indices (used by the SSA representation).
    ///
    /// # Arguments
    ///
    /// * `remap` - A function that maps old block indices to new block indices.
    ///   Returns `None` if the target should remain unchanged.
    pub fn remap_branch_targets<F>(&mut self, remap: F)
    where
        F: Fn(usize) -> Option<usize>,
    {
        self.for_each_target_mut(|target| {
            if let Some(new_target) = remap(*target) {
                *target = new_target;
            }
        });
    }

    /// Calls `visit` with every block index this terminator transfers to, in
    /// place: a jump's or leave's target, both arms of a branch, a switch's
    /// cases and default, and an indirect branch's resolved targets. Any other
    /// operation has none.
    fn for_each_target_mut(&mut self, mut visit: impl FnMut(&mut usize)) {
        match self {
            Self::Jump { target } | Self::Leave { target } => visit(target),
            Self::Branch {
                true_target,
                false_target,
                ..
            }
            | Self::BranchCmp {
                true_target,
                false_target,
                ..
            }
            | Self::BranchFlags {
                true_target,
                false_target,
                ..
            } => {
                visit(true_target);
                visit(false_target);
            }
            Self::Switch {
                targets, default, ..
            } => {
                targets.iter_mut().for_each(&mut visit);
                visit(default);
            }
            Self::IndirectBranch {
                resolved_targets, ..
            } => resolved_targets.iter_mut().for_each(visit),
            _ => {}
        }
    }
}
