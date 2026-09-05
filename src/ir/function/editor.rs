//! Checked SSA editing utilities for optimization passes.
//!
//! This module provides [`SsaEditor`], a lightweight mutation facade for
//! [`SsaFunction`]. The editor is designed for pass bodies
//! that want local safety checks while avoiding verifier or repair work after
//! every individual edit.
//!
//! # Design
//!
//! Editing follows a two-stage model:
//!
//! 1. Individual editor operations perform cheap local checks before mutating
//!    the function.
//! 2. [`SsaFunction::edit`] performs one repair or rebuild step at the edit
//!    boundary, followed by optional validation.
//!
//! This keeps common optimization passes fast while moving correctness checks
//! such as type compatibility, self-reference prevention, and same-block
//! use-before-def prevention into shared infrastructure.

use std::collections::BTreeMap;

use crate::{
    BitSet,
    analysis::{
        exceptions::EhCfg,
        verifier::{SsaVerifier, VerifierError, VerifyLevel},
    },
    error::{Error, Result},
    graph::{
        NodeId,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        block::{ReplaceResult, SsaBlock},
        function::{CopyPropagationResult, SsaFunction},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId, UseSite, VariableOrigin},
    },
    target::Target,
};

/// Required cleanup after a sequence of SSA edits.
///
/// Scopes are ordered from cheapest to most invasive. [`SsaEditor`] widens the
/// current scope automatically as edits are performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SsaEditScope {
    /// No mutation has occurred.
    None,
    /// Only instruction operands were changed.
    ///
    /// Finishing this scope recomputes use-site metadata but does not strip
    /// instructions or compact variables.
    UsesOnly,
    /// Instructions were replaced, nopped, inserted, or removed without
    /// changing the control-flow graph.
    ///
    /// Finishing this scope runs [`SsaFunction::repair_ssa`].
    InstructionsOnly,
    /// The control-flow graph may have changed.
    ///
    /// Finishing this scope runs [`SsaFunction::rebuild_ssa`].
    ///
    /// This subsumes what was previously a separate `StructuredCfg` scope for
    /// "the CFG changed but the caller maintained phi edges". That variant ran
    /// `rebuild_ssa` too, so it was behaviourally identical while its
    /// documentation promised a cheaper boundary — an edge edit can invalidate
    /// dominance, liveness, and definition placement regardless of whether phi
    /// operands were kept consistent, so there was nothing cheaper to do.
    CfgModifying,
}

/// Rollback behavior for [`SsaFunction::edit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsaRollbackPolicy {
    /// Never clone the original function and never roll back automatically.
    Never,
    /// Clone the original function and restore it if the edit, repair, rebuild,
    /// or optional verifier step fails.
    OnFailure,
}

/// Options controlling a checked SSA edit session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsaEditOptions {
    /// Whether to run the verifier after boundary repair.
    pub verify: bool,
    /// Which invariants boundary verification checks.
    ///
    /// Defaults to [`VerifyLevel::Full`]. The cheaper [`VerifyLevel::Standard`]
    /// omits the dominance check, and dominance is exactly what a mis-scoped
    /// edit breaks: `check_defined_before_use` only asks whether a definition
    /// *exists*, so IR with a use above its own definition passes. A caller that
    /// has already opted into verification is unlikely to want the one check
    /// that catches the most common way an edit goes wrong; opt down explicitly
    /// with [`Self::with_verify_level`] where the O(n²) dominance pass is too
    /// expensive.
    pub verify_level: VerifyLevel,
    /// Whether to restore the original function when the edit fails.
    pub rollback: SsaRollbackPolicy,
}

impl SsaEditOptions {
    /// Creates edit options with no verification and no rollback.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            verify: false,
            verify_level: VerifyLevel::Full,
            rollback: SsaRollbackPolicy::Never,
        }
    }

    /// Sets which invariants boundary verification checks.
    ///
    /// Only consulted when verification is enabled via [`Self::with_verify`].
    #[must_use]
    pub const fn with_verify_level(mut self, level: VerifyLevel) -> Self {
        self.verify_level = level;
        self
    }

    /// Enables or disables boundary verification.
    #[must_use]
    pub const fn with_verify(mut self, verify: bool) -> Self {
        self.verify = verify;
        self
    }

    /// Sets rollback behavior for the edit session.
    #[must_use]
    pub const fn with_rollback(mut self, rollback: SsaRollbackPolicy) -> Self {
        self.rollback = rollback;
        self
    }
}

impl Default for SsaEditOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary returned after a checked SSA edit session finishes successfully.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaEditReport {
    /// Whether any editor operation changed the function.
    pub changed: bool,
    /// The strongest cleanup scope required by the operations that ran.
    pub scope: SsaEditScope,
}

/// Reason a checked replacement was skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplacementSkipReason {
    /// The old variable and replacement variable are identical.
    SameVariable,
    /// The variable being replaced is not registered in the function.
    MissingOldVariable,
    /// The replacement variable is not registered in the function.
    MissingNewVariable,
    /// The variable types are incompatible.
    TypeMismatch,
    /// The requested instruction does not use the variable being replaced.
    UseNotFound,
    /// The replacement would make an instruction read its own destination.
    SelfReference,
    /// The replacement source is defined later than the use in the same block.
    SameBlockUseBeforeDef,
    /// The replacement source does not dominate the instruction use.
    DominanceViolation,
    /// The requested use is a phi operand and must use phi-specific APIs.
    PhiOperand,
    /// The requested block or instruction index does not exist.
    MissingInstruction,
}

/// A skipped checked replacement at a concrete use site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedReplacement {
    /// The use site where replacement was not performed.
    pub site: UseSite,
    /// The reason replacement was not performed.
    pub reason: ReplacementSkipReason,
}

/// Result of a checked replacement operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedReplaceResult {
    /// Number of operand occurrences replaced.
    pub replaced: usize,
    /// Use sites skipped by the shared safety checks.
    pub skipped: Vec<SkippedReplacement>,
}

impl CheckedReplaceResult {
    /// Returns `true` if no replacements were skipped.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.skipped.is_empty()
    }

    /// Converts this result into the legacy aggregate replacement result.
    #[must_use]
    pub fn as_replace_result(&self) -> ReplaceResult {
        ReplaceResult {
            replaced: self.replaced,
            skipped: self.skipped.len(),
        }
    }
}

/// Checked mutation facade for an [`SsaFunction`].
///
/// The editor performs local checks before mutating and records the strongest
/// repair scope required by the performed edits. It does not repair or verify
/// after each operation; [`SsaFunction::edit`] performs that boundary work once.
pub struct SsaEditor<'a, T: Target> {
    /// Function being edited.
    ssa: &'a mut SsaFunction<T>,
    /// Whether any operation changed the function.
    changed: bool,
    /// Strongest repair scope required so far.
    scope: SsaEditScope,
    /// Whether every variable's use list is currently exact.
    ///
    /// Cleared by any mutation that does not maintain them, and re-established
    /// by a single `recompute_uses` before the first indexed replacement. This
    /// is what lets [`Self::replace_uses_checked_with`] consult the use index
    /// instead of scanning the function on every call.
    uses_indexed: bool,
}

impl<T: Target> SsaFunction<T> {
    /// Runs a checked edit session and repairs the function once at the end.
    ///
    /// The closure receives an [`SsaEditor`] that exposes checked mutation
    /// helpers. If the closure succeeds and changed the function, this method
    /// applies the repair required by the editor's accumulated
    /// [`SsaEditScope`]. Optional verification runs after repair.
    ///
    /// # Errors
    ///
    /// Returns an error if the edit closure fails, SSA rebuild fails, or
    /// verification is enabled and detects malformed SSA. If
    /// [`SsaRollbackPolicy::OnFailure`] is selected, the original function is
    /// restored before returning the error.
    pub fn edit<F>(&mut self, options: SsaEditOptions, edit: F) -> Result<SsaEditReport>
    where
        F: FnOnce(&mut SsaEditor<T>) -> Result<()>,
    {
        let original = if matches!(options.rollback, SsaRollbackPolicy::OnFailure) {
            Some(self.clone())
        } else {
            None
        };

        let edit_result = {
            let mut editor = SsaEditor {
                ssa: self,
                changed: false,
                scope: SsaEditScope::None,
                // Pessimistic: a caller may have mutated the function directly
                // before opening this session, so the first indexed
                // replacement pays one `recompute_uses` to establish the index.
                uses_indexed: false,
            };
            let result = edit(&mut editor);
            result.map(|_| SsaEditReport {
                changed: editor.changed,
                scope: editor.scope,
            })
        };

        let report = match edit_result {
            Ok(report) => report,
            Err(error) => {
                restore_on_failure(self, original);
                return Err(error);
            }
        };

        if report.changed {
            // Set before boundary repair can fail. Under
            // `SsaRollbackPolicy::Never` a failed repair leaves these edits
            // applied, and the caller's own "changed" return has repeatedly
            // been written to say otherwise — so the pass group reads this
            // instead of trusting it. A rollback below replaces the whole
            // function with its pre-edit clone, which clears the flag.
            self.mark_edit_dirty();
            if let Err(error) = finish_edit_scope(self, report.scope) {
                restore_on_failure(self, original);
                return Err(error);
            }
        }

        if options.verify {
            let errors = SsaVerifier::new(self).verify(options.verify_level);
            if !errors.is_empty() {
                restore_on_failure(self, original);
                return Err(Error::new(format_verifier_errors(&errors)));
            }
        }

        Ok(report)
    }

    /// Replaces instruction uses of `old_var` with `new_var` after shared checks.
    ///
    /// This method mutates instruction operands directly and does not run repair
    /// or recompute use metadata. Callers that use it outside [`SsaFunction::edit`]
    /// should call [`SsaFunction::recompute_uses`] or the appropriate repair
    /// method before relying on def-use metadata.
    #[must_use]
    pub fn replace_uses_checked(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
    ) -> CheckedReplaceResult {
        let needs_dominance = self
            .variable(new_var)
            .map(|new| {
                let def_block = new.def_site().block;
                self.blocks().iter().enumerate().any(|(block_idx, block)| {
                    block_idx != def_block
                        && block
                            .instructions()
                            .iter()
                            .any(|instr| instr.op().uses_var(old_var))
                })
            })
            .unwrap_or(false);
        let dominators = if needs_dominance && self.block_count() > 0 {
            let eh = EhCfg::from_ssa(self);
            Some(compute_dominators(&eh, NodeId::new(0)))
        } else {
            None
        };

        self.replace_uses_checked_with(old_var, new_var, dominators.as_ref())
    }

    /// Performs a checked use replacement against a caller-supplied dominator
    /// tree, skipping the per-call CFG/dominator construction.
    ///
    /// Replacing instruction uses never changes any terminator, so a single
    /// dominator tree stays valid across a batch of replacements. Callers that
    /// rewrite many copies (e.g. copy propagation) should build the tree once
    /// and reuse it via this method instead of calling
    /// [`replace_uses_checked`](Self::replace_uses_checked) per pair.
    #[must_use]
    /// Checked use replacement driven by `old_var`'s recorded use sites rather
    /// than a scan of the whole function.
    ///
    /// Same result as [`Self::replace_uses_checked_with`], at O(uses of
    /// `old_var`) instead of O(function). The scanning form is called once per
    /// redundant pair by GVN and once per copy by copy propagation, which makes
    /// those passes O(pairs x function) — the dominant cost on large functions.
    ///
    /// # Correctness
    ///
    /// Requires `old_var`'s use list to be current; [`SsaEditor`] guarantees
    /// that by recomputing once before its first indexed replacement and then
    /// keeping the lists exact, which this method does by moving each rewritten
    /// site from `old_var` to `new_var`.
    pub(in crate::ir::function) fn replace_uses_checked_indexed(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        dominators: Option<&DominatorTree>,
    ) -> CheckedReplaceResult {
        // Instruction uses only, mirroring the scanning form; phi operands are
        // handled by the phi-specific paths.
        let candidates: Vec<UseSite> = self
            .variable(old_var)
            .map(|var| {
                var.uses()
                    .iter()
                    .filter(|site| !site.is_phi_operand)
                    .copied()
                    .collect()
            })
            .unwrap_or_default();

        let mut allowed = Vec::new();
        let mut skipped = Vec::new();
        for site in candidates {
            // The list can name a site whose op no longer reads `old_var` (a
            // previous rewrite in this session); skip those rather than trust it
            // blindly.
            let still_uses = self
                .block(site.block)
                .and_then(|block| block.instruction(site.instruction))
                .is_some_and(|instr| instr.op().uses_var(old_var));
            if !still_uses {
                continue;
            }
            match can_replace_instruction_use_with_dominators(
                self,
                old_var,
                new_var,
                site.block,
                site.instruction,
                dominators,
            ) {
                Ok(()) => allowed.push(site),
                Err(reason) => skipped.push(SkippedReplacement { site, reason }),
            }
        }

        let mut replaced = 0usize;
        let mut moved: Vec<UseSite> = Vec::new();
        for site in allowed {
            if let Some(instr) = self
                .block_mut(site.block)
                .and_then(|block| block.instruction_mut(site.instruction))
            {
                let n = instr.op_mut().replace_uses(old_var, new_var);
                if n > 0 {
                    replaced = replaced.saturating_add(n);
                    moved.push(site);
                }
            }
        }

        // Keep both use lists exact so the next indexed replacement stays valid.
        if !moved.is_empty() {
            if let Some(old) = self.variable_mut(old_var) {
                old.retain_uses(|site| {
                    !moved
                        .iter()
                        .any(|m| m.block == site.block && m.instruction == site.instruction)
                });
            }
            if let Some(new) = self.variable_mut(new_var) {
                for site in moved {
                    new.add_use(site);
                }
            }
        }

        CheckedReplaceResult { replaced, skipped }
    }

    pub(in crate::ir::function) fn replace_uses_checked_with(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        dominators: Option<&DominatorTree>,
    ) -> CheckedReplaceResult {
        let mut allowed = Vec::new();
        let mut skipped = Vec::new();

        for (block_idx, block) in self.blocks().iter().enumerate() {
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if !instr.op().uses_var(old_var) {
                    continue;
                }

                match can_replace_instruction_use_with_dominators(
                    self, old_var, new_var, block_idx, instr_idx, dominators,
                ) {
                    Ok(()) => allowed.push((block_idx, instr_idx)),
                    Err(reason) => skipped.push(SkippedReplacement {
                        site: UseSite::instruction(block_idx, instr_idx),
                        reason,
                    }),
                }
            }
        }

        let mut replaced = 0usize;
        for (block_idx, instr_idx) in allowed {
            if let Some(instr) = self
                .block_mut(block_idx)
                .and_then(|block| block.instruction_mut(instr_idx))
            {
                replaced = replaced.saturating_add(instr.op_mut().replace_uses(old_var, new_var));
            }
        }

        CheckedReplaceResult { replaced, skipped }
    }

    /// Checks whether one instruction use can be replaced safely.
    ///
    /// This method is intentionally cheap enough for optimization passes. It
    /// validates variable existence, type compatibility, self-reference, same
    /// block ordering, and cross-block dominance for instruction uses.
    ///
    /// # Errors
    ///
    /// Returns a [`ReplacementSkipReason`] describing why the replacement would
    /// be unsafe or inapplicable.
    pub fn can_replace_instruction_use(
        &self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        block_idx: usize,
        instr_idx: usize,
    ) -> std::result::Result<(), ReplacementSkipReason> {
        let needs_dominance = self
            .variable(new_var)
            .is_some_and(|new| new.def_site().block != block_idx);
        let dominators = if needs_dominance && self.block_count() > 0 {
            let eh = EhCfg::from_ssa(self);
            Some(compute_dominators(&eh, NodeId::new(0)))
        } else {
            None
        };
        can_replace_instruction_use_with_dominators(
            self,
            old_var,
            new_var,
            block_idx,
            instr_idx,
            dominators.as_ref(),
        )
    }
}

impl<'a, T: Target> SsaEditor<'a, T> {
    /// Returns an immutable view of the function being edited.
    #[must_use]
    pub const fn function(&self) -> &SsaFunction<T> {
        self.ssa
    }

    /// Returns whether this edit session has changed the function.
    #[must_use]
    pub const fn changed(&self) -> bool {
        self.changed
    }

    /// Returns the strongest repair scope required so far.
    #[must_use]
    pub const fn scope(&self) -> SsaEditScope {
        self.scope
    }

    /// Replaces instruction uses after shared safety checks.
    ///
    /// The edit scope is widened to [`SsaEditScope::UsesOnly`] when at least
    /// one operand occurrence is replaced.
    pub fn replace_uses_checked(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
    ) -> CheckedReplaceResult {
        let result = self.ssa.replace_uses_checked(old_var, new_var);
        if result.replaced > 0 {
            self.mark_changed(SsaEditScope::UsesOnly);
        }
        result
    }

    /// Replaces instruction uses of `old_var` with `new_var` against a
    /// caller-supplied dominator tree, skipping the per-call CFG + dominator
    /// construction that [`replace_uses_checked`](Self::replace_uses_checked)
    /// performs internally.
    ///
    /// Replacing instruction uses never rewrites a terminator, so a single
    /// dominator tree stays valid across an entire batch of replacements. A
    /// pass that rewrites many variables (e.g. GVN) should build the tree once
    /// from [`function`](Self::function) and reuse it here, turning a per-pair
    /// O(blocks) dominator rebuild into one build for the whole batch.
    ///
    /// The edit scope is widened to [`SsaEditScope::UsesOnly`] when at least one
    /// operand occurrence is replaced.
    pub fn replace_uses_checked_with(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        dominators: Option<&DominatorTree>,
    ) -> CheckedReplaceResult {
        // One `recompute_uses` per edit session, then O(uses) per replacement.
        // The scanning form is O(function) *per call*, and callers such as GVN
        // and copy propagation invoke this once per candidate pair.
        if !self.uses_indexed {
            self.ssa.recompute_uses();
            self.uses_indexed = true;
        }
        let result = self
            .ssa
            .replace_uses_checked_indexed(old_var, new_var, dominators);
        if result.replaced > 0 {
            self.mark_changed(SsaEditScope::UsesOnly);
            // `replace_uses_checked_indexed` moved the rewritten sites between
            // the two variables, so the lists are still exact.
            self.uses_indexed = true;
        }
        result
    }

    /// Propagates copy mappings through instruction operands.
    ///
    /// The edit scope is widened to [`SsaEditScope::UsesOnly`] when at least
    /// one operand occurrence is replaced.
    pub fn propagate_copies(
        &mut self,
        copies: &BTreeMap<SsaVarId, SsaVarId>,
    ) -> CopyPropagationResult {
        let result = self.ssa.propagate_copies(copies);
        if result.total_replaced > 0 {
            self.mark_changed(SsaEditScope::UsesOnly);
        }
        result
    }

    /// Replaces the copy instruction defining `dest` with [`SsaOp::Nop`].
    ///
    /// Boundary repair strips the nop and compacts variables once the edit
    /// session finishes.
    pub fn nop_copy_defining(&mut self, dest: SsaVarId) -> bool {
        let changed = self.ssa.nop_copy_defining(dest);
        if changed {
            self.mark_changed(SsaEditScope::InstructionsOnly);
        }
        changed
    }

    /// Replaces an instruction operation and marks instruction repair required.
    ///
    /// If the new operation defines a variable, the variable's definition site
    /// is updated to the instruction location. The final edit boundary repair
    /// handles stale use metadata and any variables orphaned by the old
    /// operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the target instruction does not exist.
    pub fn replace_instruction_op(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        new_op: SsaOp<T>,
    ) -> Result<()> {
        let defs: Vec<SsaVarId> = new_op.defs().collect();
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };
        let Some(instr) = block.instruction_mut(instr_idx) else {
            return Err(Error::new(format!(
                "missing instruction B{block_idx}:I{instr_idx}"
            )));
        };

        instr.set_op_preserving_type(new_op);
        for dest in defs {
            if let Some(var) = self.ssa.variable_mut(dest) {
                var.set_def_site(DefSite::instruction(block_idx, instr_idx));
            }
        }
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(())
    }

    /// Replaces an instruction with [`SsaOp::Nop`].
    ///
    /// This is the checked editor equivalent of instruction removal. Boundary
    /// repair strips the nop and compacts variables once the edit session
    /// finishes.
    ///
    /// # Errors
    ///
    /// Returns an error if the target instruction does not exist.
    pub fn nop_instruction(&mut self, block_idx: usize, instr_idx: usize) -> Result<()> {
        self.replace_instruction_op(block_idx, instr_idx, SsaOp::Nop)
    }

    /// Replaces the terminator operation for a block.
    ///
    /// The terminator is represented by the block's last instruction. This
    /// helper marks the edit as [`SsaEditScope::CfgModifying`] because changing
    /// a terminator can change successor edges even when the instruction count
    /// stays fixed.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist or has no instructions.
    pub fn replace_terminator_op(&mut self, block_idx: usize, new_op: SsaOp<T>) -> Result<()> {
        let Some(block) = self.ssa.block(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };
        // The control question, not the positional one. Overwriting a last
        // instruction that is an ordinary computation would delete a definition
        // the block still needs, so such a block is reported as having no
        // terminator rather than silently retargeted.
        if block.control_terminator().is_none() {
            return Err(Error::new(format!("block B{block_idx} has no terminator")));
        }
        let Some(instr_idx) = block.instructions().len().checked_sub(1) else {
            return Err(Error::new(format!("block B{block_idx} has no terminator")));
        };

        self.replace_instruction_op(block_idx, instr_idx, new_op)?;
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(())
    }

    /// Removes all instructions in a block starting at `start_idx`.
    ///
    /// This is intended for dead-tail cleanup after a terminator. The function
    /// truncates the instruction list and marks instruction repair required so
    /// shifted or removed definition sites are reconciled at the edit boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn remove_instruction_tail(&mut self, block_idx: usize, start_idx: usize) -> Result<usize> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let original_len = block.instructions().len();
        if start_idx >= original_len {
            return Ok(0);
        }
        block.instructions_mut().truncate(start_idx);
        let removed = original_len.saturating_sub(start_idx);
        if removed > 0 {
            self.mark_changed(SsaEditScope::InstructionsOnly);
        }
        Ok(removed)
    }

    /// Clears all phis and instructions from a block.
    ///
    /// Clearing a block changes CFG structure by removing its terminator and
    /// definitions, so boundary cleanup uses full SSA rebuild.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn clear_block(&mut self, block_idx: usize) -> Result<bool> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        if block.is_empty() {
            return Ok(false);
        }

        block.clear();
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(true)
    }

    /// Folds a block's terminator to a new op and repairs the phi operands of
    /// any successor whose incoming edge that removes.
    ///
    /// This is the cheap path for branch folding — replacing
    /// `Branch(cond, A, B)` with `Jump(A)`. Such an edit is *not* a general CFG
    /// mutation and does not need [`SsaFunction::rebuild_ssa`]:
    ///
    /// - Removing an edge only ever **increases** dominance, because it removes
    ///   paths. Every definition that dominated a use still does, so no
    ///   def-use violation can be introduced and **no new phi can ever be
    ///   required** — existing phis can only become redundant.
    /// - The one thing that *does* need fixing is each dropped successor's phi
    ///   operands for this predecessor, which this method prunes directly.
    ///
    /// The edit therefore settles at [`SsaEditScope::InstructionsOnly`]
    /// (`repair_ssa`) rather than escalating the whole function to a 19-phase
    /// Cytron reconstruction. Measured, `rebuild_ssa` costs 23-54x a full IR
    /// clone, so avoiding one is worth more than eliminating every per-pass
    /// snapshot in the same sweep.
    ///
    /// Returns the number of phi operands pruned.
    ///
    /// # Errors
    ///
    /// Returns an error if the block or its terminator does not exist, or if
    /// `new_op` would **add** a successor edge — the reasoning above holds only
    /// for edge removal, so an edge-adding fold is rejected rather than
    /// under-repaired.
    pub fn fold_terminator_pruning_phis(
        &mut self,
        block_idx: usize,
        new_op: SsaOp<T>,
    ) -> Result<usize> {
        let Some(block) = self.ssa.block(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };
        let Some(last_idx) = block.instructions().len().checked_sub(1) else {
            return Err(Error::new(format!("block B{block_idx} has no terminator")));
        };

        let before: Vec<usize> = block.successors();
        let after: Vec<usize> = new_op.successors();

        // The whole justification for staying at `InstructionsOnly` is that this
        // edit only *removes* edges: removing paths can only increase dominance,
        // so no new phi can ever be required. An edit that adds an edge breaks
        // that argument — it can make a definition stop dominating a use, and it
        // can require a phi that does not exist — so it is rejected here rather
        // than silently under-repaired. Use
        // [`SsaEditor::redirect_terminator_target`] or
        // [`SsaEditor::replace_terminator_op`] for the general case; both record
        // [`SsaEditScope::CfgModifying`].
        if let Some(added) = after.iter().find(|target| !before.contains(target)) {
            return Err(Error::new(format!(
                "fold_terminator_pruning_phis on B{block_idx} would add edge to B{added}; \
                 this path only prunes removed edges — use `replace_terminator_op` \
                 for an edit that adds one"
            )));
        }

        self.replace_instruction_op(block_idx, last_idx, new_op)?;

        // Prune only the edges this fold actually removed. A successor that
        // still has another edge from this block (e.g. `Branch(c, A, A)`)
        // keeps its operand.
        let mut pruned = 0usize;
        for successor in before {
            if after.contains(&successor) {
                continue;
            }
            if let Some(target) = self.ssa.block_mut(successor) {
                for phi in target.phi_nodes_mut() {
                    let operands = phi.operands_mut();
                    let original = operands.len();
                    // Never leave a phi empty; a single-operand phi is handled
                    // by trivial-phi simplification, not here.
                    if operands
                        .iter()
                        .filter(|op| op.predecessor() != block_idx)
                        .count()
                        == 0
                    {
                        continue;
                    }
                    operands.retain(|op| op.predecessor() != block_idx);
                    pruned = pruned.saturating_add(original.saturating_sub(operands.len()));
                }
            }
        }

        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(pruned)
    }

    /// Redirects one successor edge in a block terminator.
    ///
    /// The block's last instruction is treated as its terminator. If that
    /// operation references `old_target`, all matching target fields are
    /// rewritten to `new_target` through [`SsaOp::redirect_target`].
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist or has no instructions.
    pub fn redirect_terminator_target(
        &mut self,
        block_idx: usize,
        old_target: usize,
        new_target: usize,
    ) -> Result<bool> {
        let Some(mut op) = self
            .ssa
            .block(block_idx)
            .and_then(SsaBlock::control_terminator)
            .cloned()
        else {
            return Err(Error::new(format!("block B{block_idx} has no terminator")));
        };

        if !op.redirect_target(old_target, new_target) {
            return Ok(false);
        }

        self.replace_terminator_op(block_idx, op)?;
        Ok(true)
    }

    /// Redirects one terminator target, for a caller that also fixes up the
    /// affected successor phi operands in the same edit session.
    ///
    /// Behaviourally identical to [`SsaEditor::redirect_terminator_target`] —
    /// both record [`SsaEditScope::CfgModifying`], because an edge edit
    /// invalidates dominance and definition placement whether or not phi
    /// operands were kept consistent. Retained as a distinct name so call sites
    /// that *do* maintain phis stay self-documenting.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist or has no instructions.
    pub fn redirect_terminator_target_structured(
        &mut self,
        block_idx: usize,
        old_target: usize,
        new_target: usize,
    ) -> Result<bool> {
        let Some(mut op) = self
            .ssa
            .block(block_idx)
            .and_then(SsaBlock::control_terminator)
            .cloned()
        else {
            return Err(Error::new(format!("block B{block_idx} has no terminator")));
        };

        if !op.redirect_target(old_target, new_target) {
            return Ok(false);
        }

        let instr_idx = self
            .ssa
            .block(block_idx)
            .and_then(|block| block.instructions().len().checked_sub(1))
            .ok_or_else(|| Error::new(format!("block B{block_idx} has no terminator")))?;
        self.replace_instruction_op(block_idx, instr_idx, op)?;
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(true)
    }

    /// Replaces one predecessor in phi operands with one or more predecessors.
    ///
    /// For each phi in `block_idx`, the first operand whose predecessor is
    /// `old_pred` is rewritten to the first value in `new_preds`. Additional
    /// predecessors receive duplicate operands carrying the same value. This
    /// preserves phi value attribution when multiple incoming edges are
    /// redirected through a single removed block.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn expand_phi_predecessor(
        &mut self,
        block_idx: usize,
        old_pred: usize,
        new_preds: &[usize],
    ) -> Result<usize> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };
        if new_preds.is_empty() {
            return Ok(0);
        }

        let mut updated = 0usize;
        for phi in block.phi_nodes_mut() {
            let Some(value) = phi
                .operands()
                .iter()
                .find(|operand| operand.predecessor() == old_pred)
                .map(|operand| operand.value())
            else {
                continue;
            };

            // Drop the edge being replaced, then re-add one operand per new
            // predecessor — but never for a predecessor the phi already names.
            //
            // A predecessor can reach this block *both* directly and through the
            // block being bypassed; redirecting then collapses the two paths onto
            // one edge. Adding an operand regardless leaves two operands for that
            // single edge, which has no defined meaning and which consumers
            // disagree about: `PhiNode::operand_from` returns the first and
            // discards the rest, while SCCP meets them all and yields Bottom.
            // `retarget_phi_predecessor` already guards this case; this is the
            // same guard for the one-to-many form.
            //
            // When an operand for that predecessor already exists it is kept as
            // it stands: it describes the direct edge, which is the edge that
            // survives.
            phi.operands_mut()
                .retain(|operand| operand.predecessor() != old_pred);
            updated = updated.saturating_add(1);

            for &pred in new_preds {
                if phi
                    .operands()
                    .iter()
                    .any(|operand| operand.predecessor() == pred)
                {
                    continue;
                }
                phi.add_operand(PhiOperand::new(value, pred));
                updated = updated.saturating_add(1);
            }
        }

        if updated > 0 {
            self.mark_changed(SsaEditScope::CfgModifying);
        }
        Ok(updated)
    }

    /// Rewrites phi operands that name one predecessor to another predecessor.
    ///
    /// This is useful after block coalescing, where successor phis that pointed
    /// at the removed block must instead point at the block that absorbed it.
    ///
    /// When the phi already carries an operand for `new_pred`, retargeting would
    /// leave two operands on one edge — which has no defined meaning, and which
    /// consumers disagree about: [`PhiNode::operand_from`] returns the first and
    /// discards the rest, while SCCP meets them all and yields Bottom. Both
    /// edges merging into one carry the same value only if their operands agree;
    /// when they do, the redundant operand is dropped, and when they do not the
    /// merge is not semantics-preserving and is rejected.
    ///
    /// [`PhiNode::operand_from`]: crate::ir::phi::PhiNode::operand_from
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist, or if the two edges being
    /// merged carry different values for the same phi.
    pub fn replace_phi_predecessor(
        &mut self,
        block_idx: usize,
        old_pred: usize,
        new_pred: usize,
    ) -> Result<usize> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let mut updated = 0usize;
        for (phi_idx, phi) in block.phi_nodes_mut().iter_mut().enumerate() {
            let value_on = |pred: usize| {
                phi.operands()
                    .iter()
                    .find(|operand| operand.predecessor() == pred)
                    .map(PhiOperand::value)
            };
            match (value_on(new_pred), value_on(old_pred)) {
                (Some(existing), Some(retargeted)) if existing != retargeted => {
                    return Err(Error::new(format!(
                        "B{block_idx} phi {phi_idx}: cannot redirect B{old_pred} to B{new_pred},                          the edges carry different values ({retargeted} and {existing})"
                    )));
                }
                (Some(_), Some(_)) => {
                    // Both edges already agree, so the retargeted operand is
                    // redundant rather than a second value for the edge.
                    phi.operands_mut()
                        .retain(|operand| operand.predecessor() != old_pred);
                    updated = updated.saturating_add(1);
                }
                (None, Some(_)) => {
                    for operand in phi.operands_mut() {
                        if operand.predecessor() == old_pred {
                            operand.set_predecessor(new_pred);
                            updated = updated.saturating_add(1);
                        }
                    }
                }
                (_, None) => {}
            }
        }

        if updated > 0 {
            self.mark_changed(SsaEditScope::CfgModifying);
        }
        Ok(updated)
    }

    /// Coalesces a block into its unconditional predecessor.
    ///
    /// `predecessor_idx` must end with `Jump { target: successor_idx }`.
    /// The predecessor's jump is removed, successor phis are converted into
    /// `Copy` instructions, successor instructions are appended, self-targets
    /// inside the moved instructions are redirected to the predecessor, and
    /// the successor block is cleared. Phi operands in other blocks that
    /// referenced the successor are rewritten to the predecessor.
    ///
    /// # Errors
    ///
    /// Returns an error if either block is missing, or if the predecessor does
    /// not end in an unconditional jump to the successor.
    pub fn coalesce_unconditional_successor(
        &mut self,
        predecessor_idx: usize,
        successor_idx: usize,
    ) -> Result<bool> {
        let Some(predecessor) = self.ssa.block(predecessor_idx) else {
            return Err(Error::new(format!("missing block B{predecessor_idx}")));
        };
        if !matches!(
            predecessor.control_terminator(),
            Some(SsaOp::Jump { target }) if *target == successor_idx
        ) {
            return Err(Error::new(format!(
                "block B{predecessor_idx} does not jump to B{successor_idx}"
            )));
        }

        let Some(successor) = self.ssa.block(successor_idx) else {
            return Err(Error::new(format!("missing block B{successor_idx}")));
        };
        if successor.instructions().is_empty() {
            return Ok(false);
        }

        let phi_copies: Vec<SsaInstruction<T>> = successor
            .phi_nodes()
            .iter()
            .filter_map(|phi| {
                let operand = phi.operands().first()?;
                let dest = phi.result();
                let src = operand.value();
                if dest == src {
                    return None;
                }
                Some(SsaInstruction::synthetic(SsaOp::Copy { dest, src }))
            })
            .collect();
        let successor_instrs = successor.instructions().to_vec();

        let Some(predecessor) = self.ssa.block_mut(predecessor_idx) else {
            return Err(Error::new(format!("missing block B{predecessor_idx}")));
        };
        let instrs = predecessor.instructions_mut();
        instrs.pop();
        instrs.extend(phi_copies);
        instrs.extend(successor_instrs);
        for instr in instrs {
            instr
                .op_mut()
                .redirect_target(successor_idx, predecessor_idx);
        }

        self.clear_block(successor_idx)?;

        let block_count = self.ssa.block_count();
        for block_idx in 0..block_count {
            if block_idx == predecessor_idx || block_idx == successor_idx {
                continue;
            }
            self.replace_phi_predecessor(block_idx, successor_idx, predecessor_idx)?;
        }

        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(true)
    }

    /// Appends a basic block to the function.
    ///
    /// The block id must match the next dense block index. This keeps block
    /// creation explicit and avoids silently introducing an inconsistent block
    /// id that would later require canonicalization to interpret.
    ///
    /// # Errors
    ///
    /// Returns an error if the block id is not equal to the current block
    /// count.
    pub fn append_block(&mut self, block: SsaBlock<T>) -> Result<usize> {
        let block_idx = self.ssa.block_count();
        if block.id() != block_idx {
            return Err(Error::new(format!(
                "new block id B{} does not match next block B{block_idx}",
                block.id()
            )));
        }

        self.ssa.add_block(block);
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(block_idx)
    }

    /// Creates a variable using the function's registered origin type.
    ///
    /// This is the editor-facing equivalent of
    /// [`SsaFunction::create_variable_for_origin`]. Creating a variable does
    /// not by itself require repair, because the caller must still insert the
    /// corresponding definition before the edit boundary.
    #[must_use]
    pub fn create_variable_for_origin(
        &mut self,
        origin: VariableOrigin,
        version: u32,
        def_site: DefSite,
    ) -> SsaVarId {
        self.ssa
            .create_variable_for_origin(origin, version, def_site)
    }

    /// Appends a phi node to a block.
    ///
    /// If the phi result already exists in the function variable table, its
    /// definition site is updated to the destination block. Boundary repair
    /// refreshes uses and removes any stale definitions if later edits discard
    /// the phi.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn add_phi(&mut self, block_idx: usize, phi: PhiNode) -> Result<usize> {
        let result = phi.result();
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let phi_idx = block.phi_nodes().len();
        block.phi_nodes_mut().push(phi);
        if let Some(var) = self.ssa.variable_mut(result) {
            var.set_def_site(DefSite::phi(block_idx));
        }
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(phi_idx)
    }

    /// Replaces phi operands from a predecessor group with one operand.
    ///
    /// Every operand whose predecessor appears in `old_preds` is removed. If
    /// at least one operand was removed, `new_operand` is appended. This models
    /// canonical CFG insertion such as replacing several incoming header edges
    /// with a single preheader or unified latch edge.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn replace_phi_predecessor_group(
        &mut self,
        block_idx: usize,
        old_preds: &[usize],
        new_operand: PhiOperand,
    ) -> Result<bool> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let mut changed = false;
        for phi in block.phi_nodes_mut() {
            let before = phi.operands().len();
            phi.operands_mut()
                .retain(|operand| !old_preds.contains(&operand.predecessor()));
            if phi.operands().len() != before {
                phi.add_operand(new_operand);
                changed = true;
            }
        }

        if changed {
            self.mark_changed(SsaEditScope::CfgModifying);
        }
        Ok(changed)
    }

    /// Replaces the operands of **one** phi — identified by its result — that
    /// come from `old_preds`, with a single `new_operand`.
    ///
    /// Prefer this over
    /// [`replace_phi_predecessor_group_for_origin`](Self::replace_phi_predecessor_group_for_origin)
    /// whenever the caller means a specific phi. A [`VariableOrigin`] does not
    /// identify one: [`VariableOrigin::Phi`] is a unit variant, so every phi
    /// merging earlier phis — the common shape in lifted native code — shares
    /// it, and the origin-keyed form rewrites all of them to the same value.
    ///
    /// # Errors
    ///
    /// Returns an error if `block_idx` names no block.
    pub fn replace_phi_predecessor_group_for_result(
        &mut self,
        block_idx: usize,
        phi_result: SsaVarId,
        old_preds: &[usize],
        new_operand: PhiOperand,
    ) -> Result<bool> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let mut changed = false;
        if let Some(phi) = block
            .phi_nodes_mut()
            .iter_mut()
            .find(|phi| phi.result() == phi_result)
        {
            let before = phi.operands().len();
            phi.operands_mut()
                .retain(|operand| !old_preds.contains(&operand.predecessor()));
            if phi.operands().len() != before {
                phi.add_operand(new_operand);
                changed = true;
            }
        }

        if changed {
            self.mark_changed(SsaEditScope::CfgModifying);
        }
        Ok(changed)
    }

    /// Replaces one origin's phi operands from a predecessor group.
    ///
    /// Only the phi node whose [`PhiNode::origin`] equals `origin` is updated.
    /// This lets loop canonicalization rewrite each header phi to the value
    /// produced by the matching inserted preheader or latch phi.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn replace_phi_predecessor_group_for_origin(
        &mut self,
        block_idx: usize,
        origin: VariableOrigin,
        old_preds: &[usize],
        new_operand: PhiOperand,
    ) -> Result<bool> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let mut changed = false;
        for phi in block.phi_nodes_mut() {
            if phi.origin() != origin {
                continue;
            }
            let before = phi.operands().len();
            phi.operands_mut()
                .retain(|operand| !old_preds.contains(&operand.predecessor()));
            if phi.operands().len() != before {
                phi.add_operand(new_operand);
                changed = true;
            }
        }

        if changed {
            self.mark_changed(SsaEditScope::CfgModifying);
        }
        Ok(changed)
    }

    /// Prunes phi operands that refer to unreachable predecessors.
    ///
    /// This is the editor-facing wrapper around
    /// [`SsaFunction::prune_phi_operands`]. It marks the edit as
    /// [`SsaEditScope::UsesOnly`] when operands are removed, so the boundary
    /// refreshes use metadata without simplifying phis before the calling pass
    /// can account for them.
    #[must_use]
    pub fn prune_phi_operands(&mut self, reachable: &BitSet) -> usize {
        let pruned = self.ssa.prune_phi_operands(reachable);
        if pruned > 0 {
            self.mark_changed(SsaEditScope::UsesOnly);
        }
        pruned
    }

    /// Simplifies a phi node by replacing its result with a source variable.
    ///
    /// This is intended for phi nodes already proven trivial by analysis.
    /// Instruction uses are replaced through dominance-aware checks, and the
    /// phi is removed only when no remaining uses of its result exist outside
    /// the phi being removed.
    ///
    /// # Errors
    ///
    /// Returns an error if the block or phi node does not exist.
    pub fn simplify_phi_to_copy(
        &mut self,
        block_idx: usize,
        phi_idx: usize,
        source: SsaVarId,
    ) -> Result<()> {
        if self
            .ssa
            .block(block_idx)
            .and_then(|block| block.phi_nodes().get(phi_idx))
            .is_none()
        {
            return Err(Error::new(format!("missing phi B{block_idx}:P{phi_idx}")));
        }

        if self.ssa.simplify_phi_to_copy(block_idx, phi_idx, source) {
            self.mark_changed(SsaEditScope::InstructionsOnly);
            Ok(())
        } else {
            Err(Error::new(format!(
                "failed to simplify phi B{block_idx}:P{phi_idx}"
            )))
        }
    }

    /// Inserts an instruction before the block terminator.
    ///
    /// If the block has no terminator, the instruction is appended. If the
    /// inserted instruction defines a variable, its definition site is updated
    /// to the inserted location.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn insert_before_terminator(
        &mut self,
        block_idx: usize,
        instr: SsaInstruction<T>,
    ) -> Result<usize> {
        let defs: Vec<SsaVarId> = instr.defs().collect();
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let insert_idx = block
            .instructions()
            .iter()
            .position(SsaInstruction::is_terminator)
            .unwrap_or_else(|| block.instructions().len());
        block.instructions_mut().insert(insert_idx, instr);

        for dest in defs {
            if let Some(var) = self.ssa.variable_mut(dest) {
                var.set_def_site(DefSite::instruction(block_idx, insert_idx));
            }
        }
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(insert_idx)
    }

    /// Inserts an instruction at a specific index within a block.
    ///
    /// The index is clamped to the end of the instruction list, so callers can
    /// use a large index to append. If the inserted instruction defines a
    /// variable, that variable's definition site is updated to the insertion
    /// location. Boundary repair fixes definition sites for later instructions
    /// shifted by the insertion.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn insert_instruction(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        instr: SsaInstruction<T>,
    ) -> Result<usize> {
        let defs: Vec<SsaVarId> = instr.defs().collect();
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let insert_idx = instr_idx.min(block.instructions().len());
        block.instructions_mut().insert(insert_idx, instr);

        for dest in defs {
            if let Some(var) = self.ssa.variable_mut(dest) {
                var.set_def_site(DefSite::instruction(block_idx, insert_idx));
            }
        }
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(insert_idx)
    }

    /// Removes a phi node at a specific block-local index.
    ///
    /// Boundary repair removes any orphaned phi result variables and refreshes
    /// def-use metadata. This helper intentionally does not replace uses of the
    /// phi result; callers must insert or otherwise provide a replacement
    /// definition when uses remain.
    ///
    /// # Errors
    ///
    /// Returns an error if the block or phi index does not exist.
    pub fn remove_phi(&mut self, block_idx: usize, phi_idx: usize) -> Result<()> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };
        if phi_idx >= block.phi_nodes().len() {
            return Err(Error::new(format!("missing phi B{block_idx}:P{phi_idx}")));
        }

        block.phi_nodes_mut().remove(phi_idx);
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(())
    }

    /// Marks that the edit session changed the control-flow graph.
    ///
    /// Low-level CFG transformations can call this after performing block or
    /// edge edits through existing APIs. The edit boundary will run
    /// [`SsaFunction::rebuild_ssa`].
    pub fn mark_cfg_changed(&mut self) {
        self.mark_changed(SsaEditScope::CfgModifying);
    }

    /// Widens the accumulated edit scope and records that a mutation occurred.
    fn mark_changed(&mut self, scope: SsaEditScope) {
        self.changed = true;
        self.scope = self.scope.max(scope);
        // Conservative: only the replacement path below knows how to keep the
        // use lists exact, and it re-sets this immediately afterwards.
        self.uses_indexed = false;
    }
}

/// Restores the original function when a rollback snapshot exists.
fn restore_on_failure<T: Target>(ssa: &mut SsaFunction<T>, original: Option<SsaFunction<T>>) {
    if let Some(original) = original {
        *ssa = original;
    }
}

/// Applies boundary cleanup for an edit scope.
///
/// Every scope leaves the def sites and the use index **exact**. `repair_ssa`
/// and `rebuild_ssa` rewrite instructions and variables without restoring the
/// index, so an edit that declared anything above `UsesOnly` used to end with a
/// use index that no longer described the function.
///
/// That matters because the index is not merely metadata: it is the candidate
/// list `replace_uses_checked_indexed` iterates, so the next pass's rewrites are
/// chosen from it.
///
/// The scope-specific work below therefore does *not* recompute uses itself —
/// the unconditional pair afterwards supersedes it, and doing both would scan
/// the function twice for one edit.
fn finish_edit_scope<T: Target>(ssa: &mut SsaFunction<T>, scope: SsaEditScope) -> Result<()> {
    match scope {
        SsaEditScope::None => return Ok(()),
        SsaEditScope::UsesOnly => {}
        SsaEditScope::InstructionsOnly => ssa.repair_ssa(),
        SsaEditScope::CfgModifying => ssa.rebuild_ssa()?,
    }

    ssa.refresh_def_sites();
    ssa.recompute_uses();
    Ok(())
}

/// Formats verifier errors for an edit-session failure.
fn format_verifier_errors(errors: &[VerifierError]) -> String {
    let details = errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    format!("checked SSA edit produced invalid SSA: {details}")
}

/// Returns whether two SSA types can be treated as compatible for replacement.
fn types_compatible<T: Target>(old_type: &T::Type, new_type: &T::Type) -> bool {
    old_type == new_type || T::is_unknown(old_type) || T::is_unknown(new_type)
}

/// Checks one instruction replacement using optional precomputed dominators.
fn can_replace_instruction_use_with_dominators<T: Target>(
    ssa: &SsaFunction<T>,
    old_var: SsaVarId,
    new_var: SsaVarId,
    block_idx: usize,
    instr_idx: usize,
    dominators: Option<&DominatorTree>,
) -> std::result::Result<(), ReplacementSkipReason> {
    if old_var == new_var {
        return Err(ReplacementSkipReason::SameVariable);
    }

    let Some(old) = ssa.variable(old_var) else {
        return Err(ReplacementSkipReason::MissingOldVariable);
    };
    let Some(new) = ssa.variable(new_var) else {
        return Err(ReplacementSkipReason::MissingNewVariable);
    };

    if !types_compatible::<T>(old.var_type(), new.var_type()) {
        return Err(ReplacementSkipReason::TypeMismatch);
    }

    let Some(instr) = ssa
        .block(block_idx)
        .and_then(|block| block.instruction(instr_idx))
    else {
        return Err(ReplacementSkipReason::MissingInstruction);
    };

    if !instr.op().uses_var(old_var) {
        return Err(ReplacementSkipReason::UseNotFound);
    }

    if instr.op().defs().any(|def| def == new_var) {
        return Err(ReplacementSkipReason::SelfReference);
    }

    let def_site = new.def_site();
    if def_site.block == block_idx {
        if def_site
            .instruction
            .is_some_and(|def_idx| def_idx >= instr_idx)
        {
            return Err(ReplacementSkipReason::SameBlockUseBeforeDef);
        }
        return Ok(());
    }

    if definition_dominates_block(ssa, def_site, block_idx, dominators) {
        Ok(())
    } else {
        Err(ReplacementSkipReason::DominanceViolation)
    }
}

/// Checks whether a definition site dominates an instruction-use block.
fn definition_dominates_block<T: Target>(
    ssa: &SsaFunction<T>,
    def_site: DefSite,
    use_block: usize,
    dominators: Option<&DominatorTree>,
) -> bool {
    if def_site.block == use_block {
        return true;
    }
    if def_site.block >= ssa.block_count() || use_block >= ssa.block_count() {
        return false;
    }

    dominators
        .is_some_and(|tree| tree.dominates(NodeId::new(def_site.block), NodeId::new(use_block)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            ops::SsaOp,
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// Returns a test variable ID by dense index.
    fn var(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    /// Builds a one-block function where `v1` is used before `v2` is defined.
    fn same_block_late_def_function() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 0);
        let v0 = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let v1 = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let v2 = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 3),
            MockType::I32,
        );
        let v3 = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy { dest: v1, src: v0 }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: v3,
            left: v1,
            right: v1,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v2,
            value: ConstValue::I32(2),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v3) }));
        ssa.add_block(block);
        ssa.recompute_uses();
        ssa
    }

    /// Builds a branching function where a definition in B1 does not dominate B2.
    fn non_dominating_definition_function() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 0);
        let cond = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let branch_value = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let old = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(2, 0),
            MockType::I32,
        );
        let dest = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(2, 1),
            MockType::I32,
        );

        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));

        let mut left = SsaBlock::new(1);
        left.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: branch_value,
            value: ConstValue::I32(10),
        }));
        left.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));

        let mut right = SsaBlock::new(2);
        right.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: old,
            value: ConstValue::I32(20),
        }));
        right.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest,
            left: old,
            right: old,
            flags: None,
        }));
        right.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));

        let mut exit = SsaBlock::new(3);
        exit.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));

        ssa.add_block(entry);
        ssa.add_block(left);
        ssa.add_block(right);
        ssa.add_block(exit);
        ssa.recompute_uses();
        ssa
    }

    /// Builds a function with an otherwise-safe replacement of incompatible types.
    fn type_mismatch_function() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 0);
        let old = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let new = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I64,
        );
        let dest = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: old,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: new,
            value: ConstValue::I64(2),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest,
            left: old,
            right: old,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(dest),
        }));
        ssa.add_block(block);
        ssa.recompute_uses();
        ssa
    }

    /// `fold_terminator_pruning_phis` stays at `InstructionsOnly` because
    /// removing an edge can only *increase* dominance, so no new phi can be
    /// required. An edit that adds an edge breaks that argument, so it must be
    /// rejected rather than under-repaired.
    #[test]
    fn folding_a_terminator_rejects_an_edit_that_adds_an_edge() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let cond = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        for id in [1usize, 2] {
            let mut block = SsaBlock::new(id);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let outcome = ssa.edit(SsaEditOptions::new(), |editor| {
            // `Jump B1` -> `Branch B1, B2` adds the edge to B2.
            let result = editor.fold_terminator_pruning_phis(
                0,
                SsaOp::Branch {
                    condition: cond,
                    true_target: 1,
                    false_target: 2,
                },
            );
            assert!(
                result.is_err(),
                "an edge-adding fold must be rejected on this path"
            );
            Ok(())
        });
        assert!(outcome.is_ok());

        // Pruning a real edge still works.
        let outcome = ssa.edit(SsaEditOptions::new(), |editor| {
            editor.fold_terminator_pruning_phis(0, SsaOp::Jump { target: 1 })?;
            Ok(())
        });
        assert!(outcome.is_ok());
    }

    #[test]
    fn checked_replacement_rejects_same_block_late_definition() {
        let mut ssa = same_block_late_def_function();

        let result = ssa.replace_uses_checked(var(1), var(2));

        assert_eq!(result.replaced, 0);
        assert_eq!(result.skipped.len(), 1);
        assert_eq!(
            result.skipped[0].reason,
            ReplacementSkipReason::SameBlockUseBeforeDef
        );
    }

    #[test]
    fn checked_replacement_rejects_self_reference() {
        let mut ssa = same_block_late_def_function();

        let result = ssa.replace_uses_checked(var(1), var(3));

        assert_eq!(result.replaced, 0);
        assert_eq!(result.skipped.len(), 1);
        assert_eq!(
            result.skipped[0].reason,
            ReplacementSkipReason::SelfReference
        );
    }

    #[test]
    fn checked_replacement_rejects_non_dominating_definition() {
        let mut ssa = non_dominating_definition_function();

        let result = ssa.replace_uses_checked(var(2), var(1));

        assert_eq!(result.replaced, 0);
        assert_eq!(result.skipped.len(), 1);
        assert_eq!(
            result.skipped[0].reason,
            ReplacementSkipReason::DominanceViolation
        );
    }

    #[test]
    fn checked_replacement_rejects_type_mismatch() {
        let mut ssa = type_mismatch_function();

        let result = ssa.replace_uses_checked(var(0), var(1));

        assert_eq!(result.replaced, 0);
        assert_eq!(result.skipped.len(), 1);
        assert_eq!(
            result.skipped[0].reason,
            ReplacementSkipReason::TypeMismatch
        );
    }

    #[test]
    fn editor_repairs_once_after_instruction_edits() {
        let mut ssa = same_block_late_def_function();

        let report = ssa
            .edit(SsaEditOptions::new(), |editor| {
                editor.nop_instruction(0, 1)?;
                Ok(())
            })
            .unwrap();

        assert!(report.changed);
        assert_eq!(report.scope, SsaEditScope::InstructionsOnly);
        assert!(
            ssa.block(0)
                .unwrap()
                .instructions()
                .iter()
                .all(|instr| !matches!(instr.op(), SsaOp::Nop))
        );
    }

    #[test]
    fn editor_rolls_back_when_verification_fails() {
        let mut ssa = same_block_late_def_function();
        let original = ssa.clone();

        let result = ssa.edit(
            SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| {
                editor.replace_instruction_op(
                    0,
                    4,
                    SsaOp::Return {
                        value: Some(SsaVarId::from_index(999)),
                    },
                )?;
                Ok(())
            },
        );

        assert!(result.is_err());
        assert_eq!(ssa.variable_count(), original.variable_count());
        assert_eq!(
            ssa.block(0).unwrap().instructions().len(),
            original.block(0).unwrap().instructions().len()
        );
        assert_eq!(
            ssa.block(0).unwrap().instructions()[4].uses(),
            original.block(0).unwrap().instructions()[4].uses()
        );
    }

    /// The indexed replacement must be indistinguishable from the scanning one.
    ///
    /// `replace_uses_checked_indexed` exists purely to avoid the O(function)
    /// scan; if it ever diverges — by consulting a stale use list, by missing a
    /// site, or by replacing one the checks should have rejected — it silently
    /// changes optimizer output. This runs both forms against identical copies
    /// of every fixture, for every variable pair, and requires the replacement
    /// count, the skip reasons, and the resulting IR to match exactly.
    #[test]
    fn indexed_replacement_matches_scanning_replacement() {
        let fixtures: Vec<(&str, SsaFunction<MockTarget>)> = vec![
            ("scalar", crate::testing::scalar_rewrite_fixture()),
            ("diamond", crate::testing::diamond_phi_fixture()),
            ("loop", crate::testing::loop_counter_fixture()),
            ("memory", crate::testing::memory_effect_fixture()),
            ("straight_32", crate::testing::straight_line_fixture(32)),
            ("diamonds_8", crate::testing::diamond_chain_fixture(8)),
        ];

        for (name, base) in fixtures {
            let ids: Vec<SsaVarId> = base.variables().iter().map(|v| v.id()).collect();
            for &old_var in &ids {
                for &new_var in &ids {
                    if old_var == new_var {
                        continue;
                    }

                    let mut scanned = base.clone();
                    scanned.recompute_uses();
                    let scan_result = scanned.replace_uses_checked_with(old_var, new_var, None);

                    let mut indexed = base.clone();
                    indexed.recompute_uses();
                    let index_result = indexed.replace_uses_checked_indexed(old_var, new_var, None);

                    assert_eq!(
                        scan_result.replaced, index_result.replaced,
                        "{name}: replaced count differs for {old_var} -> {new_var}"
                    );
                    assert_eq!(
                        scan_result.skipped.len(),
                        index_result.skipped.len(),
                        "{name}: skip count differs for {old_var} -> {new_var}"
                    );

                    // Resulting IR must be identical instruction for instruction.
                    assert_eq!(
                        scanned.block_count(),
                        indexed.block_count(),
                        "{name}: block count differs"
                    );
                    for block_idx in 0..scanned.block_count() {
                        let a = scanned.block(block_idx).map(SsaBlock::instructions);
                        let b = indexed.block(block_idx).map(SsaBlock::instructions);
                        match (a, b) {
                            (Some(a), Some(b)) => {
                                assert_eq!(a.len(), b.len(), "{name}: block {block_idx} length");
                                for (i, (ia, ib)) in a.iter().zip(b.iter()).enumerate() {
                                    assert_eq!(
                                        ia.op(),
                                        ib.op(),
                                        "{name}: block {block_idx} instr {i} differs for {old_var} -> {new_var}"
                                    );
                                }
                            }
                            (None, None) => {}
                            _ => panic!("{name}: block {block_idx} presence differs"),
                        }
                    }
                }
            }
        }
    }

    /// A *sequence* of replacements through one edit session must match the
    /// scanning form applied in the same order.
    ///
    /// This is the invariant the indexed path actually risks. The scanning form
    /// re-derives uses from the IR on every call, so it is inherently immune to
    /// staleness; the indexed form reads a cached list and must therefore keep
    /// that list exact as it rewrites. Chained pairs (`a -> b` then `b -> c`)
    /// are the case that breaks a naive implementation: the second replacement
    /// must see the uses the first one *created*.
    ///
    /// Verified to fail if the use-list maintenance in
    /// `replace_uses_checked_indexed` is removed. The stale-site guard there is
    /// defensive rather than load-bearing — with maintenance correct no stale
    /// site arises, so removing it does not fail this test; it protects against
    /// a future caller mutating the IR mid-session without clearing
    /// `uses_indexed`.
    #[test]
    fn chained_indexed_replacements_match_scanning() {
        let fixtures: Vec<(&str, SsaFunction<MockTarget>)> = vec![
            ("scalar", crate::testing::scalar_rewrite_fixture()),
            ("diamond", crate::testing::diamond_phi_fixture()),
            ("loop", crate::testing::loop_counter_fixture()),
            ("straight_16", crate::testing::straight_line_fixture(16)),
            ("diamonds_4", crate::testing::diamond_chain_fixture(4)),
        ];

        for (name, base) in fixtures {
            let ids: Vec<SsaVarId> = base.variables().iter().map(|v| v.id()).collect();
            if ids.len() < 3 {
                continue;
            }
            // Overlapping chains: each pair's target is the next pair's source,
            // so every step depends on uses the previous step introduced.
            let mut pairs: Vec<(SsaVarId, SsaVarId)> = Vec::new();
            for window in ids.windows(2) {
                if let [a, b] = window {
                    pairs.push((*a, *b));
                }
            }

            // Reference: the scanning form, which never consults a use list.
            let mut scanned = base.clone();
            let mut scan_total = 0usize;
            for &(old_var, new_var) in &pairs {
                scan_total = scan_total.saturating_add(
                    scanned
                        .replace_uses_checked_with(old_var, new_var, None)
                        .replaced,
                );
            }

            // Subject: the indexed form, all inside one editor session so the
            // cached lists must survive every step.
            let mut indexed = base.clone();
            let mut index_total = 0usize;
            let report = indexed.edit(SsaEditOptions::new(), |editor| {
                for &(old_var, new_var) in &pairs {
                    index_total = index_total.saturating_add(
                        editor
                            .replace_uses_checked_with(old_var, new_var, None)
                            .replaced,
                    );
                }
                Ok(())
            });
            assert!(report.is_ok(), "{name}: edit session failed");

            assert_eq!(
                scan_total, index_total,
                "{name}: total replacements differ across the chain"
            );
            for block_idx in 0..scanned.block_count() {
                let a = scanned.block(block_idx).map(SsaBlock::instructions);
                let b = indexed.block(block_idx).map(SsaBlock::instructions);
                if let (Some(a), Some(b)) = (a, b) {
                    for (i, (ia, ib)) in a.iter().zip(b.iter()).enumerate() {
                        assert_eq!(
                            ia.op(),
                            ib.op(),
                            "{name}: block {block_idx} instr {i} diverged after chained replacement"
                        );
                    }
                }
            }
        }
    }
}
