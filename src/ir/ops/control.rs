//! Control-flow queries and branch-target rewriting for [`SsaOp`].
//!
//! Terminator recognition, successor enumeration, and the target remapping the
//! CFG-shaping passes (block merging, jump threading, control-flow
//! simplification) use when they splice blocks.

use super::*;
use crate::target::Target;

impl<T: Target> SsaOp<T> {
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

    /// Calls `f` for every successor block index of this operation.
    ///
    /// Allocation-free equivalent of iterating [`SsaOp::successors`]; preferred
    /// in CFG-construction and traversal hot paths.
    pub fn for_each_successor<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        match self {
            Self::Jump { target } | Self::Leave { target } => f(*target),
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
                f(*true_target);
                f(*false_target);
            }
            Self::Switch {
                targets, default, ..
            } => {
                for target in targets {
                    f(*target);
                }
                f(*default);
            }
            Self::IndirectBranch {
                resolved_targets, ..
            } => {
                for target in resolved_targets {
                    f(*target);
                }
            }
            // Return, Throw, Rethrow, EndFinally, EndFilter have no successors
            _ => {}
        }
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

        match self {
            Self::Jump { target } | Self::Leave { target } if *target == old_target => {
                *target = new_target;
                true
            }
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
                let mut changed = false;
                if *true_target == old_target {
                    *true_target = new_target;
                    changed = true;
                }
                if *false_target == old_target {
                    *false_target = new_target;
                    changed = true;
                }
                changed
            }
            Self::Switch {
                targets, default, ..
            } => {
                let mut changed = false;
                if *default == old_target {
                    *default = new_target;
                    changed = true;
                }
                for target in targets.iter_mut() {
                    if *target == old_target {
                        *target = new_target;
                        changed = true;
                    }
                }
                changed
            }
            Self::IndirectBranch {
                resolved_targets, ..
            } => {
                let mut changed = false;
                for target in resolved_targets.iter_mut() {
                    if *target == old_target {
                        *target = new_target;
                        changed = true;
                    }
                }
                changed
            }
            _ => false,
        }
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
        match self {
            Self::Jump { target } | Self::Leave { target } => {
                if let Some(new_target) = remap(*target) {
                    *target = new_target;
                }
            }
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
                if let Some(new_target) = remap(*true_target) {
                    *true_target = new_target;
                }
                if let Some(new_target) = remap(*false_target) {
                    *false_target = new_target;
                }
            }
            Self::Switch {
                targets, default, ..
            } => {
                for target in targets.iter_mut() {
                    if let Some(new_target) = remap(*target) {
                        *target = new_target;
                    }
                }
                if let Some(new_target) = remap(*default) {
                    *default = new_target;
                }
            }
            Self::IndirectBranch {
                resolved_targets, ..
            } => {
                for target in resolved_targets.iter_mut() {
                    if let Some(new_target) = remap(*target) {
                        *target = new_target;
                    }
                }
            }
            // All other operations don't have branch targets
            _ => {}
        }
    }
}
