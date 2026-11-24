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

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, DerefMut},
};

use crate::{
    analysis::{
        cfg::SsaCfg,
        exceptions::FunctionRoots,
        verifier::{SsaVerifier, VerifierError, VerifyLevel},
    },
    bitset::BitSet,
    error::{Error, Result},
    facts::{Substitution, VariableFacts},
    graph::{
        NodeId,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        block::ReplaceResult,
        function::SsaFunction,
        identity::Identity,
        instruction::SsaInstruction,
        phi::{PhiEdge, PhiOperand},
        variable::{DefSite, SsaVarId, UseSite},
    },
    target::Target,
};

mod blocks;
mod instructions;
mod merges;
mod phis;
mod routes;
mod terminators;
mod uses;
mod variables;

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
    /// Edges were given blocks of their own that jump on to where the edges
    /// went ([`SsaEditor::split_edges`]), and nothing else changed: every
    /// value means what it meant and is read where it was.
    ///
    /// Finishing this scope restores definition sites, use-site metadata and
    /// instruction identities, and nothing more: there is nothing to tidy, and
    /// tidying would renumber values a caller holding facts about them still
    /// names.
    EdgesSplit,
    /// Instructions were replaced, nopped, inserted, or removed without
    /// changing the control-flow graph.
    ///
    /// Finishing this scope runs [`SsaFunction::repair_ssa`].
    InstructionsOnly,
    /// The control-flow graph changed, and the caller restored every value
    /// the change disturbed (see [`SsaEditor::declare_repaired`]).
    ///
    /// Finishing this scope empties the blocks the edits cut off, drops the phi
    /// operands of edges that no longer exist, and runs
    /// [`SsaFunction::repair_ssa`].
    CfgRepaired,
    /// The control-flow graph changed, and the caller has not said that the
    /// values the change disturbed are restored.
    ///
    /// Finishing this scope fails, and the session's rollback policy decides
    /// what becomes of its edits: an edge carries values only the editing
    /// caller knows, so the caller restores them and declares it
    /// ([`SsaEditor::declare_repaired`]), which makes the scope
    /// [`Self::CfgRepaired`].
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
    /// Whether the substitution took: it rewrote a use, or refused none,
    /// which is so for a value nothing read.
    #[must_use]
    pub fn took(&self) -> bool {
        self.replaced > 0 || self.skipped.is_empty()
    }

    /// Returns `true` if no replacements were skipped.
    ///
    /// Every use was considered - instruction operands and phi operands alike -
    /// so a complete result means the replaced variable is read nowhere any more
    /// and its definition may be removed. A phi operand the replacement cannot
    /// reach is skipped, never silently left behind: a caller that deleted the
    /// definition on a result that ignored phi operands left the operand naming
    /// a value nothing defines.
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

/// The function an edit session changes, and the rollback copy of it.
///
/// Reads go straight through. The first access that can change the function
/// takes the copy a rollback would restore, when the session asked for one, so
/// a session that changes nothing never pays for it. Every mutation of the
/// function goes through [`DerefMut`], and recording a change without mutating
/// (see [`SsaEditor::mark_cfg_changed`]) takes the copy too, so the copy exists
/// before the first change of either kind and holds the function exactly as the
/// session found it.
struct EditTarget<'a, T: Target> {
    /// Function being edited.
    ssa: &'a mut SsaFunction<T>,
    /// Whether the session wants a copy to restore on failure.
    keeps_original: bool,
    /// The function as the session found it, once something asked to change it.
    original: Option<SsaFunction<T>>,
}

impl<T: Target> EditTarget<'_, T> {
    /// Takes the rollback copy if the session wants one and has none yet.
    fn keep_original(&mut self) {
        if self.keeps_original && self.original.is_none() {
            self.original = Some(self.ssa.clone());
        }
    }
}

impl<T: Target> Deref for EditTarget<'_, T> {
    type Target = SsaFunction<T>;

    fn deref(&self) -> &Self::Target {
        self.ssa
    }
}

impl<T: Target> DerefMut for EditTarget<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.keep_original();
        self.ssa
    }
}

/// Checked mutation facade for an [`SsaFunction`].
///
/// The editor performs local checks before mutating and records the strongest
/// repair scope required by the performed edits. It does not repair or verify
/// after each operation; [`SsaFunction::edit`] performs that boundary work once.
pub struct SsaEditor<'a, T: Target> {
    /// Function being edited.
    ssa: EditTarget<'a, T>,
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
    /// Whether the caller restored every value its CFG edits disturbed; see
    /// [`Self::declare_repaired`].
    repaired: bool,
}

impl<T: Target> SsaFunction<T> {
    /// Runs a checked edit session and repairs the function once at the end.
    ///
    /// The closure receives an [`SsaEditor`] that exposes checked mutation
    /// helpers. If the closure succeeds and changed the function, this method
    /// applies the repair required by the editor's accumulated
    /// [`SsaEditScope`]. Optional verification runs after repair.
    ///
    /// # Arguments
    ///
    /// * `options` - Verification toggle, verify level, and rollback policy for
    ///   the session.
    /// * `edit` - Closure that performs the mutations through the editor.
    ///
    /// # Returns
    ///
    /// An [`SsaEditReport`] recording whether the function changed and the
    /// strongest repair scope that ran.
    ///
    /// # Errors
    ///
    /// Returns an error if the edit closure fails, SSA rebuild fails, or
    /// verification is enabled and detects malformed SSA. If
    /// [`SsaRollbackPolicy::OnFailure`] is selected, the original function is
    /// restored before returning the error. Otherwise any edit made before the
    /// failure stays applied, and the function is marked dirty (see
    /// [`SsaFunction::take_edit_dirty`]).
    pub fn edit<F>(&mut self, options: SsaEditOptions, edit: F) -> Result<SsaEditReport>
    where
        F: FnOnce(&mut SsaEditor<T>) -> Result<()>,
    {
        let (edit_result, original) = {
            let mut editor = SsaEditor {
                ssa: EditTarget {
                    ssa: self,
                    keeps_original: matches!(options.rollback, SsaRollbackPolicy::OnFailure),
                    original: None,
                },
                changed: false,
                scope: SsaEditScope::None,
                // Pessimistic: a caller may have mutated the function directly
                // before opening this session, so the first indexed
                // replacement pays one `recompute_uses` to establish the index.
                uses_indexed: false,
                repaired: false,
            };
            let result = edit(&mut editor);
            let changed = editor.changed;
            let scope = if editor.repaired && editor.scope == SsaEditScope::CfgModifying {
                SsaEditScope::CfgRepaired
            } else {
                editor.scope
            };
            let original = editor.ssa.original.take();
            let result = result
                .map(|_| SsaEditReport { changed, scope })
                .map_err(|error| (error, changed));
            (result, original)
        };

        let report = match edit_result {
            Ok(report) => report,
            Err((error, changed)) => {
                // A closure that mutated and then failed leaves its edits in
                // place unless the rollback below restores the snapshot. The
                // caller's "did anything change" cannot be trusted on an error
                // path any more than on a success path, so the flag records it
                // here; a restored snapshot carries its own pre-edit flag.
                if changed {
                    self.mark_edit_dirty();
                }
                restore_on_failure(self, original, &error, changed);
                return Err(error);
            }
        };

        if report.changed {
            // Set before boundary repair can fail. Under
            // `SsaRollbackPolicy::Never` a failed repair leaves these edits
            // applied, and the caller's own "changed" return has repeatedly
            // been written to say otherwise - so the pass group reads this
            // instead of trusting it. A rollback below replaces the whole
            // function with its pre-edit clone, which clears the flag.
            self.mark_edit_dirty();
            if let Err(error) = finish_edit_scope(self, report.scope) {
                restore_on_failure(self, original, &error, true);
                return Err(error);
            }
        }

        if options.verify {
            let errors = SsaVerifier::new(self).verify(options.verify_level);
            if !errors.is_empty() {
                let error = Error::new(format_verifier_errors(&errors));
                restore_on_failure(self, original, &error, report.changed);
                return Err(error);
            }
        }

        Ok(report)
    }

    /// Replaces every use of `old_var` - instruction operands and phi operands -
    /// with `new_var` after shared checks.
    ///
    /// A phi operand is read on its incoming edge, so it is rewritten when
    /// `new_var`'s definition reaches the end of the operand's predecessor, and
    /// reported as skipped when it does not.
    ///
    /// This method mutates operands directly and does not run repair
    /// or recompute use metadata. Callers that use it outside [`SsaFunction::edit`]
    /// should call [`SsaFunction::recompute_uses`] or the appropriate repair
    /// method before relying on def-use metadata.
    ///
    /// A dominator tree is built here only when `new_var` is defined outside at
    /// least one block that uses `old_var`; a purely same-block rewrite needs no
    /// dominance check and skips the CFG construction entirely.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable whose instruction uses are rewritten.
    /// * `new_var` - The variable substituted at each accepted use site.
    ///
    /// # Returns
    ///
    /// The count of operand occurrences replaced together with the sites the
    /// shared safety checks skipped.
    #[must_use]
    pub fn replace_uses_checked(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
    ) -> CheckedReplaceResult {
        let needs_dominance = self
            .variable(new_var)
            .map(|new| {
                let def_block = new.def_site().block();
                self.blocks().iter().enumerate().any(|(block_idx, block)| {
                    (block_idx != def_block
                        && block
                            .instructions()
                            .iter()
                            .any(|instr| instr.op().uses_var(old_var)))
                        || block.phi_nodes().iter().any(|phi| {
                            phi.operands().iter().any(|operand| {
                                operand.value() == old_var && operand.predecessor() != def_block
                            })
                        })
                })
            })
            .unwrap_or(false);
        let dominators = if needs_dominance && self.block_count() > 0 {
            let eh = SsaCfg::from_ssa(self);
            Some(compute_dominators(&eh, NodeId::new(0)))
        } else {
            None
        };

        let result = self.replace_uses_checked_with(old_var, new_var, dominators.as_ref());
        self.inherit_rename_group(old_var, new_var, result.took(), Substitution::Host);
        result
    }

    /// Checks whether one instruction use can be replaced safely.
    ///
    /// This method is intentionally cheap enough for optimization passes. It
    /// validates variable existence, type compatibility, self-reference, same
    /// block ordering, and cross-block dominance for instruction uses.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable currently read at the use site.
    /// * `new_var` - The candidate replacement variable.
    /// * `block_idx` - Block containing the instruction whose use is checked.
    /// * `instr_idx` - Index of that instruction within the block.
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
    ) -> Result<(), ReplacementSkipReason> {
        let needs_dominance = self
            .variable(new_var)
            .is_some_and(|new| new.def_site().block() != block_idx);
        let dominators = if needs_dominance && self.block_count() > 0 {
            let eh = SsaCfg::from_ssa(self);
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
    /// Declares that every value this session's CFG edits disturbed has been
    /// restored.
    ///
    /// For a caller that repairs value by value - with
    /// [`Self::reconstruct_values`] and phi operands it sets itself - after
    /// editing edges. Without this a session that changed the graph is
    /// refused ([`SsaEditScope::CfgModifying`]). A caller that declares this
    /// should also verify the session, so an incomplete repair is rejected
    /// rather than kept.
    pub fn declare_repaired(&mut self) {
        self.repaired = true;
    }

    /// Returns an immutable view of the function being edited.
    #[must_use]
    pub const fn function(&self) -> &SsaFunction<T> {
        self.ssa.ssa
    }

    /// Notes that `whole` is read from `start`, the part at its lowest
    /// address, and `rest`, its other half where it is composed of two
    /// values, so the host's facts can keep `whole` where they were kept
    /// ([`VariableFacts::hold_whole`]).
    ///
    /// A fact, not an edit: nothing in the IR changes, and a rolled-back
    /// session restores it with the function.
    ///
    /// # Arguments
    ///
    /// * `whole` - The value kept whole.
    /// * `start` - The part at its lowest address.
    /// * `rest` - Its other half, where it is two values.
    pub fn note_whole(&mut self, whole: SsaVarId, start: SsaVarId, rest: Option<SsaVarId>) {
        if whole == start || rest == Some(whole) {
            return;
        }
        let facts = |var: SsaVarId| self.ssa.variable(var).map(|var| var.facts().clone());
        let Some(start) = facts(start) else {
            return;
        };
        let rest = match rest {
            Some(rest) => match facts(rest) {
                Some(facts) => Some(facts),
                None => return,
            },
            None => None,
        };
        if let Some(kept) = self.ssa.variable_mut(whole) {
            kept.facts_mut().hold_whole(&start, rest.as_ref());
        }
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

    /// [`Self::duplicate_phi_edge`] without recording a scope.
    fn add_edge_like(
        &mut self,
        block_idx: usize,
        old_pred: usize,
        new_pred: usize,
    ) -> Result<usize> {
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };
        let mut added = 0usize;
        for (phi_idx, phi) in block.phi_nodes_mut().iter_mut().enumerate() {
            let Some(value) = phi
                .normal_operands()
                .find(|operand| operand.predecessor() == old_pred)
                .map(PhiOperand::value)
            else {
                return Err(Error::new(format!(
                    "B{block_idx} phi {phi_idx} has no operand for B{old_pred}"
                )));
            };
            let existing = phi
                .normal_operands()
                .find(|operand| operand.predecessor() == new_pred)
                .map(PhiOperand::value);
            match existing {
                Some(existing) if existing == value => {}
                Some(existing) => {
                    return Err(Error::new(format!(
                        "B{block_idx} phi {phi_idx}: B{new_pred} already carries {existing}, \
                         not {value}"
                    )));
                }
                None => {
                    phi.add_operand(PhiOperand::new(value, new_pred));
                    added = added.saturating_add(1);
                }
            }
        }
        Ok(added)
    }

    /// [`Self::replace_phi_predecessor`] without recording a scope: for an
    /// operation that retargets as part of an edit whose scope it states itself.
    fn retarget_phi_operands(
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
            // Only the terminator edge is one per predecessor. Operands for
            // throws name program points inside a block, and a block has as
            // many as it has points a throw can leave from - so an exceptional
            // operand from `new_pred` is no second value for its edge, and one
            // from `old_pred` simply moves with the code it describes.
            let normal_value_on = |pred: usize| {
                phi.normal_operands()
                    .find(|operand| operand.predecessor() == pred)
                    .map(PhiOperand::value)
            };
            match (normal_value_on(new_pred), normal_value_on(old_pred)) {
                (Some(existing), Some(retargeted)) if existing != retargeted => {
                    return Err(Error::new(format!(
                        "B{block_idx} phi {phi_idx}: cannot redirect B{old_pred} to \
                         B{new_pred}, the edges carry different values ({retargeted} and \
                         {existing})"
                    )));
                }
                (Some(_), Some(_)) => {
                    // Both edges already agree, so the retargeted operand is
                    // redundant rather than a second value for the edge.
                    phi.operands_mut().retain(|operand| {
                        operand.is_exceptional() || operand.predecessor() != old_pred
                    });
                    updated = updated.saturating_add(1);
                }
                (None, Some(_)) => {
                    for operand in phi.operands_mut() {
                        if !operand.is_exceptional() && operand.predecessor() == old_pred {
                            operand.set_predecessor(new_pred);
                            updated = updated.saturating_add(1);
                        }
                    }
                }
                (_, None) => {}
            }
            // The throws move with the code: each keeps its kind and value and
            // now names the block that holds it. One that becomes an exact copy
            // of an operand already there says nothing new.
            let mut retargeted = false;
            for operand in phi.operands_mut() {
                if operand.is_exceptional() && operand.predecessor() == old_pred {
                    operand.set_predecessor(new_pred);
                    retargeted = true;
                    updated = updated.saturating_add(1);
                }
            }
            if retargeted {
                let mut seen: Vec<(usize, SsaVarId)> = Vec::new();
                phi.operands_mut().retain(|operand| {
                    if !operand.is_exceptional() {
                        return true;
                    }
                    let key = (operand.predecessor(), operand.value());
                    if seen.contains(&key) {
                        false
                    } else {
                        seen.push(key);
                        true
                    }
                });
            }
        }

        Ok(updated)
    }

    /// Gives an instruction arriving without an identity one, so it is
    /// identified on the way in rather than at the next repair: a pass that
    /// inserts and then reads back the id gets the same answer whichever
    /// insertion it used. An instruction that already has one (a move) keeps it.
    fn identify(&mut self, instr: &mut SsaInstruction<T>) {
        if instr.id().is_none() {
            instr.set_id(self.ssa.mint_instr_id());
        }
    }

    /// Sets the def site of every instruction definition in `block_idx` from
    /// index `from` onward to where it now is.
    ///
    /// Costs one pass over the block's tail, which an insertion has just paid
    /// for anyway.
    fn refresh_def_sites_from(&mut self, block_idx: usize, from: usize) {
        let moved: Vec<(SsaVarId, usize)> = self
            .ssa
            .block(block_idx)
            .map(|block| {
                block
                    .instructions()
                    .iter()
                    .enumerate()
                    .skip(from)
                    .flat_map(|(index, instr)| instr.defs().map(move |dest| (dest, index)))
                    .collect()
            })
            .unwrap_or_default();
        for (dest, index) in moved {
            if let Some(var) = self.ssa.variable_mut(dest) {
                var.set_def_site(DefSite::instruction(block_idx, index));
            }
        }
    }

    /// Marks that the edit session changed the control-flow graph.
    ///
    /// Low-level CFG transformations can call this after performing block or
    /// edge edits through existing APIs. The boundary refuses the session
    /// unless the caller also restored the values the edits disturbed and said
    /// so with [`Self::declare_repaired`].
    pub fn mark_cfg_changed(&mut self) {
        self.mark_changed(SsaEditScope::CfgModifying);
    }

    /// Widens the accumulated edit scope and records that a mutation occurred.
    ///
    /// The session's rollback copy exists from here on, so a session that only
    /// records a change still restores the function it found.
    fn mark_changed(&mut self, scope: SsaEditScope) {
        self.ssa.keep_original();
        self.changed = true;
        self.scope = self.scope.max(scope);
        // Conservative: only the replacement path below knows how to keep the
        // use lists exact, and it re-sets this immediately afterwards.
        self.uses_indexed = false;
    }
}

/// A use site as an ordered key; `UseSite` itself has no ordering.
fn use_site_key(site: UseSite) -> (usize, usize, bool) {
    (site.block(), site.index(), site.is_phi_operand())
}

/// Restores the original function when a rollback snapshot exists, and
/// otherwise records the failure on it when the session had changed it (see
/// [`SsaFunction::take_edit_failure`]).
fn restore_on_failure<T: Target>(
    ssa: &mut SsaFunction<T>,
    original: Option<SsaFunction<T>>,
    error: &Error,
    mutated: bool,
) {
    match original {
        Some(original) => *ssa = original,
        // No snapshot to restore: whatever the session changed stays, and the
        // function says so for whoever runs the pass.
        None if mutated => ssa.mark_edit_failed(error),
        None => {}
    }
}

/// Applies boundary cleanup for an edit scope.
///
/// Every scope leaves the def sites and the use index **exact**.
///
/// That matters because the index is not merely metadata: it is the candidate
/// list `replace_uses_checked_indexed` iterates, so the next pass's rewrites are
/// chosen from it, and the sparse analyses revisit a value's consumers only by
/// walking it.
///
/// `repair_ssa` restores def sites and the index as its own last step, because
/// it is called directly and not only from here. Only `UsesOnly` - which runs
/// no repair - needs the pair applied here, so one edit costs one scan rather
/// than two.
///
/// A session that changed the graph and did not declare its values restored
/// ([`SsaEditor::declare_repaired`]) is refused: an edge a session adds
/// carries values only the session knows, and no reconstruction of the whole
/// function can recover them without inventing them from rename groups the
/// edits may have invalidated.
///
/// # Errors
///
/// Returns an error for a [`SsaEditScope::CfgModifying`] session.
fn finish_edit_scope<T: Target>(ssa: &mut SsaFunction<T>, scope: SsaEditScope) -> Result<()> {
    match scope {
        SsaEditScope::None => {}
        SsaEditScope::UsesOnly | SsaEditScope::EdgesSplit => {
            ssa.refresh_def_sites();
            ssa.recompute_uses();
            ssa.assign_instr_ids();
        }
        SsaEditScope::InstructionsOnly => ssa.repair_ssa(),
        SsaEditScope::CfgRepaired => {
            // What the edits cut off goes: the caller restored every value
            // reachable code reads, so nothing it reads is defined there, and
            // the verification a repairing caller runs says so if not.
            let cfg = SsaCfg::from_ssa(ssa);
            let live = FunctionRoots::of(ssa).live_blocks(&cfg);
            drop(cfg);
            for block_idx in 1..ssa.block_count() {
                if !live.contains_checked(block_idx)
                    && let Some(block) = ssa.block_mut(block_idx)
                {
                    block.clear();
                }
            }
            ssa.prune_phi_operands(&live);
            ssa.repair_ssa();
        }
        SsaEditScope::CfgModifying => {
            return Err(Error::new(
                "an SSA edit changed the control-flow graph without restoring the values its \
                 edges carry; state them (phi operands, reconstruct_values) and declare the \
                 session repaired",
            ));
        }
    }
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
) -> Result<(), ReplacementSkipReason> {
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
    if def_site.block() == block_idx {
        if def_site.index().is_some_and(|def_idx| def_idx >= instr_idx) {
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

/// Checks whether a phi operand's read of `old_var`, arriving from
/// `predecessor`, can read `new_var` instead.
///
/// A phi operand is read on its incoming edge - at the end of the predecessor,
/// not in the phi's own block - so the replacement's definition has to reach
/// the end of `predecessor`: defined inside it (a phi of it included), or in a
/// block that dominates it. Judging the phi's own block instead would accept a
/// value that is not available on the edge.
fn can_replace_phi_operand_use_with_dominators<T: Target>(
    ssa: &SsaFunction<T>,
    old_var: SsaVarId,
    new_var: SsaVarId,
    block_idx: usize,
    phi_idx: usize,
    predecessor: usize,
    dominators: Option<&DominatorTree>,
) -> Result<(), ReplacementSkipReason> {
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
    let Some(phi) = ssa
        .block(block_idx)
        .and_then(|block| block.phi_nodes().get(phi_idx))
    else {
        return Err(ReplacementSkipReason::MissingInstruction);
    };
    if !phi
        .operands()
        .iter()
        .any(|operand| operand.predecessor() == predecessor && operand.value() == old_var)
    {
        return Err(ReplacementSkipReason::UseNotFound);
    }
    if definition_dominates_block(ssa, new.def_site(), predecessor, dominators) {
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
    if def_site.block() == use_block {
        return true;
    }
    if def_site.block() >= ssa.block_count() || use_block >= ssa.block_count() {
        return false;
    }

    if !dominators
        .is_some_and(|tree| tree.dominates(NodeId::new(def_site.block()), NodeId::new(use_block)))
    {
        return false;
    }
    // A definition at an instruction is missing wherever an exception raised
    // in its block before it can go: the handler and what follows it, even
    // where the block dominates them because the region also enters them
    // normally.
    !(def_site.index().is_some()
        && !ssa.exception_handlers().is_empty()
        && FunctionRoots::of(ssa).raise_reaches(ssa, def_site.block(), use_block))
}

/// Rewiring a value's uses to another, the way an edit session does it.
///
/// The session's replacements check each use against dominance and the types,
/// and carry the rename group a value is in to the one replacing it; the
/// function's own repairs rewire through the same two steps. Not a caller's:
/// a pass rewires through an edit session.
pub trait Rewiring<T: Target> {
    /// Carries a rename group across a substitution.
    ///
    /// A **rename group is one variable**: the set of SSA values that are
    /// versions of one storage location, and every reconstruction the rebuild
    /// performs is per group. The group is therefore only sound while it is
    /// **complete** - a definition the group cannot see is one the rename walks
    /// a read straight past, resolving it to a value that never reached it.
    ///
    /// A pass that mints a value to stand in for another is exactly where a
    /// definition would go missing: the new value is the storage's value
    /// wherever the old one was, and it is born with no group. Substitution is
    /// the one place the relationship is stated, so it is the one place that
    /// can carry the group - a hundred-odd `create_variable` calls across the
    /// passes cannot each be asked to remember.
    ///
    /// An existing group on the new value is never overwritten: it already
    /// names the storage it is a version of, and the substitution says nothing
    /// about which of the two names is right.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable that was substituted out.
    /// * `new_var` - The variable substituted in.
    /// * `took` - Whether the substitution took: it rewrote a use of
    ///   `old_var`, or refused none, as for a value nothing reads any more,
    ///   which stands for `new_var` as wholly as one whose readers it
    ///   rewrote. Nothing is carried when it was refused everywhere.
    fn inherit_rename_group(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        took: bool,
        cause: Substitution,
    );

    /// Performs a checked use replacement against a caller-supplied dominator
    /// tree, skipping the per-call CFG/dominator construction.
    ///
    /// Replacing instruction uses never changes any terminator, so a single
    /// dominator tree stays valid across a batch of replacements. Callers that
    /// rewrite many copies (e.g. copy propagation) should build the tree once
    /// and reuse it via this method instead of calling
    /// [`SsaFunction::replace_uses_checked`] per pair.
    ///
    /// Unlike [`SsaFunction::replace_uses_checked_indexed`], this form rescans every
    /// instruction in the function rather than consulting `old_var`'s use list,
    /// so it stays correct even when no use index has been established.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable whose instruction uses are rewritten.
    /// * `new_var` - The variable substituted at each accepted use site.
    /// * `dominators` - Optional precomputed dominator tree reused across a
    ///   batch; when `None`, a cross-block use that would need one is skipped.
    ///
    /// # Returns
    ///
    /// The count of operand occurrences replaced together with the sites the
    /// shared safety checks skipped.
    fn replace_uses_checked_with(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        dominators: Option<&DominatorTree>,
    ) -> CheckedReplaceResult;

    #[must_use]
    /// Checked use replacement driven by `old_var`'s recorded use sites rather
    /// than a scan of the whole function.
    ///
    /// Same result as [`Self::replace_uses_checked_with`], at O(uses of
    /// `old_var`) instead of O(function). The scanning form is called once per
    /// redundant pair by GVN and once per copy by copy propagation, which makes
    /// those passes O(pairs x function) - the dominant cost on large functions.
    ///
    /// # Correctness
    ///
    /// Requires `old_var`'s use list to be current; [`SsaEditor`] guarantees
    /// that by recomputing once before its first indexed replacement and then
    /// keeping the lists exact, which this method does by moving each rewritten
    /// site from `old_var` to `new_var`.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable whose recorded instruction uses are rewritten.
    /// * `new_var` - The variable substituted at each accepted use site.
    /// * `dominators` - Optional precomputed dominator tree reused across a
    ///   batch; when `None`, a cross-block use that would need one is skipped.
    /// * `within` - The blocks whose uses may be rewritten, an instruction's
    ///   where it lies in one and a phi operand's where its edge leaves one;
    ///   `None` for every block.
    ///
    /// # Returns
    ///
    /// The count of operand occurrences replaced together with the sites the
    /// shared safety checks skipped.
    fn replace_uses_checked_indexed(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        dominators: Option<&DominatorTree>,
        within: Option<&BitSet>,
    ) -> CheckedReplaceResult;
}

impl<T: Target> Rewiring<T> for SsaFunction<T> {
    fn replace_uses_checked_indexed(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        dominators: Option<&DominatorTree>,
        within: Option<&BitSet>,
    ) -> CheckedReplaceResult {
        let inside = |block: usize| within.is_none_or(|blocks| blocks.contains(block));
        // What each reader takes in where it then computes in `old_var`'s
        // place ([`Self::substituted_facts`]).
        let carried = self.substituted_facts(old_var, new_var);
        let candidates: Vec<UseSite> = self
            .variable(old_var)
            .map(|var| var.uses().to_vec())
            .unwrap_or_default();

        let mut allowed = Vec::new();
        let mut allowed_edges: Vec<(UseSite, usize, PhiEdge)> = Vec::new();
        let mut skipped = Vec::new();
        // The list holds one entry per operand occurrence, so a site reading
        // `old_var` twice appears twice; each is judged once.
        let mut sites_seen: BTreeSet<(usize, usize, bool)> = BTreeSet::new();
        for site in candidates {
            if !sites_seen.insert(use_site_key(site)) {
                continue;
            }
            if site.is_phi_operand() {
                // Every operand of the phi naming `old_var` is judged on its
                // own edge.
                let edges: Vec<(usize, PhiEdge)> = self
                    .block(site.block())
                    .and_then(|block| block.phi_nodes().get(site.index()))
                    .map(|phi| {
                        phi.operands()
                            .iter()
                            .filter(|operand| operand.value() == old_var)
                            .map(|operand| (operand.predecessor(), operand.edge()))
                            .collect()
                    })
                    .unwrap_or_default();
                // A predecessor listed twice is judged per operand, as the
                // scanning form judges it; the first `replace_operand_value`
                // rewrites both, and the second finds nothing left to move.
                for (predecessor, edge) in edges
                    .into_iter()
                    .filter(|(predecessor, _)| inside(*predecessor))
                {
                    match can_replace_phi_operand_use_with_dominators(
                        self,
                        old_var,
                        new_var,
                        site.block(),
                        site.index(),
                        predecessor,
                        dominators,
                    ) {
                        Ok(()) => allowed_edges.push((site, predecessor, edge)),
                        Err(reason) => skipped.push(SkippedReplacement { site, reason }),
                    }
                }
                continue;
            }
            // The list can name a site whose op no longer reads `old_var` (a
            // previous rewrite in this session); skip those rather than trust it
            // blindly.
            let still_uses = self
                .block(site.block())
                .and_then(|block| block.instruction(site.index()))
                .is_some_and(|instr| instr.op().uses_var(old_var));
            if !still_uses || !inside(site.block()) {
                continue;
            }
            match can_replace_instruction_use_with_dominators(
                self,
                old_var,
                new_var,
                site.block(),
                site.index(),
                dominators,
            ) {
                Ok(()) => allowed.push(site),
                Err(reason) => skipped.push(SkippedReplacement { site, reason }),
            }
        }

        // Each rewritten occurrence, by site: how many entries leave
        // `old_var`'s list, and how many of them join `new_var`'s. An
        // exceptional operand merged into one already naming `new_var` leaves
        // the first and joins nothing.
        let mut moved: BTreeMap<(usize, usize, bool), (UseSite, usize, usize)> = BTreeMap::new();
        let mut replaced = 0usize;
        for site in allowed {
            if let Some(instr) = self
                .block_mut(site.block())
                .and_then(|block| block.instruction_mut(site.index()))
            {
                let n = instr.replace_uses(old_var, new_var, &carried);
                if n > 0 {
                    replaced = replaced.saturating_add(n);
                    let slot = moved.entry(use_site_key(site)).or_insert((site, 0, 0));
                    slot.1 = slot.1.saturating_add(n);
                    slot.2 = slot.2.saturating_add(n);
                }
            }
        }
        for (site, predecessor, edge) in allowed_edges {
            if let Some(phi) = self
                .block_mut(site.block())
                .and_then(|block| block.phi_mut(site.index()))
            {
                let replacement = phi.replace_operand_value(predecessor, edge, old_var, new_var);
                if replacement.total() > 0 {
                    replaced = replaced.saturating_add(replacement.total());
                    let slot = moved.entry(use_site_key(site)).or_insert((site, 0, 0));
                    slot.1 = slot.1.saturating_add(replacement.total());
                    slot.2 = slot.2.saturating_add(replacement.rewritten);
                }
            }
        }

        // Keep both use lists exact so the next indexed replacement stays valid.
        // Sites are compared whole: a phi index and an instruction index can be
        // the same number. Entries are moved by count, so an operand the
        // replacement could not reach keeps its entry.
        if !moved.is_empty() {
            if let Some(old) = self.variable_mut(old_var) {
                let mut to_remove: BTreeMap<(usize, usize, bool), usize> = moved
                    .iter()
                    .map(|(key, &(_, removed, _))| (*key, removed))
                    .collect();
                old.retain_uses(|site| match to_remove.get_mut(&use_site_key(*site)) {
                    Some(remaining) if *remaining > 0 => {
                        *remaining = remaining.saturating_sub(1);
                        false
                    }
                    _ => true,
                });
            }
            if let Some(new) = self.variable_mut(new_var) {
                for (site, _, added) in moved.into_values() {
                    for _ in 0..added {
                        new.add_use(site);
                    }
                }
            }
        }

        CheckedReplaceResult { replaced, skipped }
    }

    fn inherit_rename_group(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        took: bool,
        cause: Substitution,
    ) {
        if !took {
            return;
        }
        // The host's facts fold in wherever a substitution took, including when
        // the new value already names a group: either way the two describe one
        // value from here on.
        self.note_substitution(old_var, new_var, cause);
        self.machine.substituted(old_var, new_var);
        if self.rename_group(new_var) != u32::MAX {
            return;
        }
        let group = self.rename_group(old_var);
        if group != u32::MAX {
            self.set_rename_group(new_var, group);
        }
    }

    fn replace_uses_checked_with(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        dominators: Option<&DominatorTree>,
    ) -> CheckedReplaceResult {
        // What each reader takes in where it then computes in `old_var`'s
        // place ([`SsaFunction::substituted_facts`]).
        let carried = self.substituted_facts(old_var, new_var);
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

        let mut allowed_edges = Vec::new();
        for (block_idx, block) in self.blocks().iter().enumerate() {
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                for operand in phi.operands() {
                    if operand.value() != old_var {
                        continue;
                    }
                    match can_replace_phi_operand_use_with_dominators(
                        self,
                        old_var,
                        new_var,
                        block_idx,
                        phi_idx,
                        operand.predecessor(),
                        dominators,
                    ) {
                        Ok(()) => allowed_edges.push((
                            block_idx,
                            phi_idx,
                            operand.predecessor(),
                            operand.edge(),
                        )),
                        Err(reason) => skipped.push(SkippedReplacement {
                            site: UseSite::phi_operand(block_idx, phi_idx),
                            reason,
                        }),
                    }
                }
            }
        }

        let mut replaced = 0usize;
        for (block_idx, instr_idx) in allowed {
            if let Some(instr) = self
                .block_mut(block_idx)
                .and_then(|block| block.instruction_mut(instr_idx))
            {
                replaced = replaced.saturating_add(instr.replace_uses(old_var, new_var, &carried));
            }
        }
        for (block_idx, phi_idx, predecessor, edge) in allowed_edges {
            if let Some(phi) = self
                .block_mut(block_idx)
                .and_then(|block| block.phi_mut(phi_idx))
            {
                replaced = replaced.saturating_add(
                    phi.replace_operand_value(predecessor, edge, old_var, new_var)
                        .total(),
                );
            }
        }

        CheckedReplaceResult { replaced, skipped }
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;
    use crate::{
        BitSet,
        analysis::{SsaEvaluator, SsaVerifier, VerifyLevel},
        ir::{
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            function::{SsaDefSpec, SsaFunctionBuilder},
            instruction::SsaInstruction,
            ops::SsaOp,
            phi::PhiNode,
            value::ConstValue,
            variable::{DefSite, SsaVariable, VariableOrigin},
        },
        testing::{
            MACHINE, MockTarget, MockType, assert_mock_valid_full, diamond_chain_fixture,
            diamond_phi_fixture, loop_counter_fixture, memory_effect_fixture, mock_i32,
            scalar_rewrite_fixture, straight_line_fixture,
        },
    };

    /// Returns a test variable ID by dense index.
    fn var(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    /// `B0` defines `early` and enters the protected `B1`, which defines `late`
    /// and returns it from `B2`; the handler `B3` merges what a throw in `B1`
    /// may see: `late`, and `early` too when `both`. Returns the function with
    /// `early` and `late`.
    fn handler_reading(both: bool) -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let early = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let late = ssa.create_variable(
            VariableOrigin::Local(0),
            1,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let seen = ssa.create_variable(VariableOrigin::Local(0), 2, DefSite::phi(3), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: early,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: late,
            value: ConstValue::I32(7),
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(late))));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(seen, VariableOrigin::Local(0));
        if both {
            phi.add_exceptional_operand(1, early);
        }
        phi.add_exceptional_operand(1, late);
        b3.add_phi(phi);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(seen))));
        ssa.add_block(b3);
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 0,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(1, 2),
            handler_range: BlockRange::new(3, 4),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();
        (ssa, early, late)
    }

    /// The handler's operands, as `(block, value)` pairs naming a throw.
    fn thrown(ssa: &SsaFunction<MockTarget>) -> Vec<(usize, SsaVarId)> {
        ssa.block(3)
            .and_then(|block| block.phi_nodes().first())
            .map(|phi| {
                phi.exceptional_operands()
                    .map(|operand| (operand.predecessor(), operand.value()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// **A substitution reaches what a throw may bring a handler.** `late`
    /// equals `early`, which dominates it; substituting `early` for `late`
    /// rewrites the return and the handler's operand naming `late`, so the
    /// definition can go, and the function verifies. Before, the handler kept
    /// naming `late`, the result still claimed every use replaced, and the
    /// caller removed a definition the handler read: analysir's memory pass
    /// forwarded a load so, on a Delphi x64 routine. Both forms, the scan and
    /// the use-list walk, are asked; where the handler already lists `early`
    /// for the block the operand naming `late` merges into it, and the use
    /// lists stay exact.
    #[test]
    fn a_substitution_reaches_the_values_a_throw_brings_a_handler() {
        for both in [false, true] {
            for indexed in [false, true] {
                let (mut ssa, early, late) = handler_reading(both);
                let result = if indexed {
                    let dominators = compute_dominators(&SsaCfg::from_ssa(&ssa), NodeId::new(0));
                    ssa.replace_uses_checked_indexed(late, early, Some(&dominators), None)
                } else {
                    ssa.replace_uses_checked(late, early)
                };
                assert!(result.is_complete(), "both {both} indexed {indexed}");
                assert_eq!(
                    thrown(&ssa),
                    vec![(1, early)],
                    "both {both} indexed {indexed}"
                );
                if indexed {
                    let phi_uses = |value: SsaVarId| {
                        ssa.variable(value).map_or(0, |variable| {
                            variable
                                .uses()
                                .iter()
                                .filter(|site| site.is_phi_operand())
                                .count()
                        })
                    };
                    assert_eq!(phi_uses(late), 0, "both {both}");
                    assert_eq!(phi_uses(early), 1, "both {both}");
                }
                let outcome =
                    ssa.edit(SsaEditOptions::new(), |editor| editor.nop_instruction(1, 0));
                assert!(outcome.is_ok(), "both {both} indexed {indexed}");
                let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
                assert!(
                    errors.is_empty(),
                    "both {both} indexed {indexed}: {errors:?}"
                );
            }
        }
    }

    /// A block nothing reaches is dead code, and clearing it forgets the host
    /// block it stood for; the live one keeps its own.
    #[test]
    fn a_cleared_unreached_block_stands_for_nothing() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        for (id, host) in [(0, 0x10), (1, 0x20)] {
            let mut block = SsaBlock::new(id);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            block.provenance_mut().carry(host);
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        let mut reachable = BitSet::new(2);
        reachable.insert(0);
        let outcome = ssa.edit(SsaEditOptions::new(), |editor| {
            assert_eq!(editor.clear_unreached_blocks(&reachable), vec![1]);
            Ok(())
        });
        assert!(outcome.is_ok());
        assert_eq!(
            ssa.block(0).map(|block| block.provenance().carried()),
            Some(&[0x10][..])
        );
        assert!(
            ssa.block(1)
                .is_some_and(|block| block.provenance().is_empty())
        );
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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v3))));
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
        exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));

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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(dest))));
        ssa.add_block(block);
        ssa.recompute_uses();
        ssa
    }

    /// A two-way function: `B0` branches on a local to `B1` or `B2`, each of
    /// which returns.
    fn two_way() -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
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
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        for id in [1usize, 2] {
            let mut block = SsaBlock::new(id);
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        (ssa, cond)
    }

    /// **A session folds a branch and removes what the fold left unread, and
    /// its boundary repairs both.** The condition is read until the branch
    /// becomes a jump; removed in the same session, it is gone from the
    /// function the session leaves, and the session needs no declared repair.
    #[test]
    fn a_session_removes_what_its_fold_left_unread() {
        let (mut ssa, _) = two_way();
        let report = ssa
            .edit(
                SsaEditOptions::new()
                    .with_verify(true)
                    .with_rollback(SsaRollbackPolicy::OnFailure),
                |editor| {
                    editor.fold_terminator_pruning_phis(0, SsaOp::Jump { target: 1 })?;
                    let removed = editor.remove_unread_instructions([(0, 0)]);
                    assert_eq!(removed.len(), 1, "the condition is no longer read");
                    Ok(())
                },
            )
            .unwrap_or_else(|_| process::abort());
        assert_eq!(report.scope, SsaEditScope::InstructionsOnly);
        let ops: Vec<&SsaOp<MockTarget>> = ssa
            .block(0)
            .map(|block| {
                block
                    .instructions()
                    .iter()
                    .map(SsaInstruction::op)
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(ops, vec![&SsaOp::Jump { target: 1 }], "{ssa}");

        // A removal alone is a change the session repairs.
        let (mut ssa, _) = two_way();
        ssa.edit(SsaEditOptions::new(), |editor| {
            editor.fold_terminator_pruning_phis(0, SsaOp::Jump { target: 1 })?;
            Ok(())
        })
        .unwrap_or_else(|_| process::abort());
        ssa.edit(SsaEditOptions::new(), |editor| {
            assert_eq!(editor.remove_unread_instructions([(0, 0)]).len(), 1);
            Ok(())
        })
        .unwrap_or_else(|_| process::abort());
        let ops: Vec<&SsaOp<MockTarget>> = ssa
            .block(0)
            .map(|block| {
                block
                    .instructions()
                    .iter()
                    .map(SsaInstruction::op)
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(ops, vec![&SsaOp::Jump { target: 1 }], "{ssa}");

        // A definition still read stays.
        let (mut ssa, _) = two_way();
        ssa.edit(SsaEditOptions::new(), |editor| {
            assert!(editor.remove_unread_instructions([(0, 0)]).is_empty());
            Ok(())
        })
        .unwrap_or_else(|_| process::abort());
        assert!(matches!(
            ssa.block(0)
                .and_then(|block| block.instructions().first())
                .map(SsaInstruction::op),
            Some(SsaOp::Const { .. })
        ));
    }

    /// **A terminator rewritten with the same edges is not a change to the
    /// control flow.** Swapping the branch's targets keeps both edges and needs
    /// no repair; replacing it with a jump drops one, and a session that does
    /// not say what that leaves of the values is refused and rolled back.
    #[test]
    fn a_terminator_keeping_its_edges_needs_no_reconstruction() {
        let (mut ssa, cond) = two_way();
        let report = ssa
            .edit(SsaEditOptions::new(), |editor| {
                editor.replace_terminator_op(
                    0,
                    SsaOp::Branch {
                        condition: cond,
                        true_target: 2,
                        false_target: 1,
                    },
                )
            })
            .unwrap_or_else(|_| process::abort());
        assert_eq!(report.scope, SsaEditScope::InstructionsOnly);

        let (mut ssa, _) = two_way();
        let before = format!("{ssa:?}");
        let refused = ssa.edit(
            SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| editor.replace_terminator_op(0, SsaOp::Jump { target: 1 }),
        );
        assert!(refused.is_err(), "an unrepaired graph change is refused");
        assert_eq!(format!("{ssa:?}"), before, "and rolled back");
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
            block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
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

    /// B0 branches to B1 or B2, each of which branches to the merge B3 or to
    /// the exit B4; B3 returns `phi(a from B1, b from B2)`.
    fn two_edges_into_a_merge() -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let cond = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let a = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let b = ssa.create_variable(
            VariableOrigin::Local(1),
            1,
            DefSite::instruction(2, 0),
            MockType::I32,
        );
        let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);
        let branch = || {
            SsaInstruction::synthetic(SsaOp::Branch {
                condition: cond,
                true_target: 3,
                false_target: 4,
            })
        };
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(7),
        }));
        b1.add_instruction(branch());
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: b,
            value: ConstValue::I32(8),
        }));
        b2.add_instruction(branch());
        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(a, 1));
        phi.add_operand(PhiOperand::new(b, 2));
        b3.add_phi(phi);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(merged))));
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        for block in [b0, b1, b2, b3, b4] {
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        ssa
    }

    /// **Folding away a merge's last incoming edge leaves valid SSA.** Each
    /// fold keeps a phi's last operand rather than empty it, and once both of
    /// B3's edges are gone that operand names an edge that does not exist: the
    /// verifier rejected the phi, and trivial-phi simplification could not
    /// retire it, because nothing dominates a block the entry cannot reach.
    /// The phi becomes a copy of the value it still named.
    #[test]
    fn folding_away_a_merges_last_edge_leaves_valid_ssa() {
        let mut ssa = two_edges_into_a_merge();
        ssa.edit(SsaEditOptions::new(), |editor| {
            editor.fold_terminator_pruning_phis(1, SsaOp::Jump { target: 4 })?;
            editor.fold_terminator_pruning_phis(2, SsaOp::Jump { target: 4 })?;
            Ok(())
        })
        .unwrap_or_else(|_| process::abort());
        assert_eq!(ssa.validate(), Ok(()));
        let merge = ssa.block(3).unwrap_or_else(|| process::abort());
        assert!(
            merge.phi_nodes().is_empty(),
            "a stranded block keeps no phi"
        );
        assert!(
            matches!(
                merge.instructions().first().map(SsaInstruction::op),
                Some(SsaOp::Copy { dest, src }) if *dest == var(3) && *src == var(2)
            ),
            "the phi is a copy of the operand it last kept"
        );
    }

    /// The control: a merge that keeps an edge keeps its phi's meaning. One
    /// fold leaves B3 reached from B2, so nothing is stranded and no copy is
    /// placed; the one-operand phi is simplification's to settle.
    #[test]
    fn folding_away_one_of_a_merges_edges_strands_nothing() {
        let mut ssa = two_edges_into_a_merge();
        ssa.edit(SsaEditOptions::new(), |editor| {
            editor.fold_terminator_pruning_phis(1, SsaOp::Jump { target: 4 })?;
            Ok(())
        })
        .unwrap_or_else(|_| process::abort());
        assert_eq!(ssa.validate(), Ok(()));
        let merge = ssa.block(3).unwrap_or_else(|| process::abort());
        assert!(
            !merge
                .instructions()
                .iter()
                .any(|instr| matches!(instr.op(), SsaOp::Copy { .. })),
            "a reachable merge is not settled as a stranded one"
        );
    }

    /// A source defined later than its use in the same block is rejected.
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

    /// A replacement that would make an instruction read its own destination is
    /// rejected.
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

    /// A source whose definition does not dominate the use block is rejected.
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

    /// A merge that reads `old` only as a phi operand.
    ///
    /// ```text
    /// b0: replacement = 7; cond = 1; branch cond -> b1, b2
    /// b1: old = 7;   jump b3
    /// b2: other = 8; jump b3
    /// b3: merged = phi(old from b1, other from b2); return merged
    /// ```
    ///
    /// `replacement` (v0) dominates b1; `other` (v3) does not.
    fn phi_operand_function() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 0);
        let replacement = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let cond = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let old = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );
        let other = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(2, 0),
            MockType::I32,
        );
        let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: replacement,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: old,
            value: ConstValue::I32(7),
        }));
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: other,
            value: ConstValue::I32(8),
        }));
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(old, 1));
        phi.add_operand(PhiOperand::new(other, 2));
        b3.add_phi(phi);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(merged))));
        for block in [b0, b1, b2, b3] {
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        ssa
    }

    /// The value merged in from `b1` in [`phi_operand_function`].
    fn operand_from_b1(ssa: &SsaFunction<MockTarget>) -> Option<SsaVarId> {
        ssa.block(3)
            .and_then(|block| block.phi_nodes().first())
            .and_then(|phi| phi.operand_from(1))
            .map(PhiOperand::value)
    }

    /// A phi operand is a use. A replacement whose definition reaches the
    /// operand's predecessor rewrites it, and only then is the result complete -
    /// which is what lets a caller delete the old definition.
    #[test]
    fn checked_replacement_rewrites_a_phi_operand_its_definition_reaches() {
        let mut ssa = phi_operand_function();

        let result = ssa.replace_uses_checked(var(2), var(0));

        assert_eq!(result.replaced, 1, "the phi operand is the one use of v2");
        assert!(result.is_complete());
        assert_eq!(operand_from_b1(&ssa), Some(var(0)));
    }

    /// A phi operand the replacement cannot reach is reported as skipped, so the
    /// result is incomplete and no caller deletes a definition it still needs.
    #[test]
    fn checked_replacement_reports_a_phi_operand_it_cannot_reach() {
        let mut ssa = phi_operand_function();

        let result = ssa.replace_uses_checked(var(2), var(3));

        assert_eq!(result.replaced, 0);
        assert!(
            !result.is_complete(),
            "a skipped phi operand is still a use"
        );
        assert_eq!(result.skipped.len(), 1);
        assert!(result.skipped[0].site.is_phi_operand());
        assert_eq!(
            result.skipped[0].reason,
            ReplacementSkipReason::DominanceViolation
        );
        assert_eq!(operand_from_b1(&ssa), Some(var(2)));
    }

    /// The indexed form treats phi operands exactly as the scanning form does,
    /// and moves the use site between the two variables' use lists.
    #[test]
    fn indexed_replacement_rewrites_phi_operands_like_scanning() {
        let mut scanned = phi_operand_function();
        let mut indexed = phi_operand_function();
        let tree = compute_dominators(&SsaCfg::from_ssa(&indexed), NodeId::new(0));

        let by_scan = scanned.replace_uses_checked(var(2), var(0));
        let by_index = indexed.replace_uses_checked_indexed(var(2), var(0), Some(&tree), None);

        assert_eq!(by_scan, by_index);
        assert_eq!(operand_from_b1(&indexed), Some(var(0)));
        let phi_site = UseSite::phi_operand(3, 0);
        assert!(
            !indexed
                .variable(var(2))
                .is_some_and(|v| v.uses().contains(&phi_site)),
            "the rewritten operand no longer reads v2"
        );
        assert!(
            indexed
                .variable(var(0))
                .is_some_and(|v| v.uses().contains(&phi_site)),
            "and now reads v0"
        );
    }

    /// A replacement between incompatible variable types is rejected.
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

    /// Instruction edits settle at `InstructionsOnly` and one boundary repair
    /// strips the resulting nops.
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

    /// With `OnFailure` rollback, a verification failure restores the original
    /// function unchanged.
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
                    SsaOp::returns(Some(SsaVarId::from_index(999))),
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
    /// scan; if it ever diverges - by consulting a stale use list, by missing a
    /// site, or by replacing one the checks should have rejected - it silently
    /// changes optimizer output. This runs both forms against identical copies
    /// of every fixture, for every variable pair, and requires the replacement
    /// count, the skip reasons, and the resulting IR to match exactly.
    #[test]
    fn indexed_replacement_matches_scanning_replacement() {
        let fixtures: Vec<(&str, SsaFunction<MockTarget>)> = vec![
            ("scalar", scalar_rewrite_fixture()),
            ("diamond", diamond_phi_fixture()),
            ("loop", loop_counter_fixture()),
            ("memory", memory_effect_fixture()),
            ("straight_32", straight_line_fixture(32)),
            ("diamonds_8", diamond_chain_fixture(8)),
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
                    let index_result =
                        indexed.replace_uses_checked_indexed(old_var, new_var, None, None);

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
    /// defensive rather than load-bearing - with maintenance correct no stale
    /// site arises, so removing it does not fail this test; it protects against
    /// a future caller mutating the IR mid-session without clearing
    /// `uses_indexed`.
    #[test]
    fn chained_indexed_replacements_match_scanning() {
        let fixtures: Vec<(&str, SsaFunction<MockTarget>)> = vec![
            ("scalar", scalar_rewrite_fixture()),
            ("diamond", diamond_phi_fixture()),
            ("loop", loop_counter_fixture()),
            ("straight_16", straight_line_fixture(16)),
            ("diamonds_4", diamond_chain_fixture(4)),
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
                            .replace_uses_checked_with(old_var, new_var, None, Substitution::Host)
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

    /// Runs the function from its entry and returns the value it returns.
    fn returned(ssa: &SsaFunction<MockTarget>) -> Option<ConstValue<MockTarget>> {
        let mut evaluator = SsaEvaluator::new(ssa, MACHINE);
        let trace = evaluator.execute_from_entry(1_000);
        let last = trace.last_block()?;
        match ssa.block(last)?.control_terminator()? {
            SsaOp::Return {
                value: Some(value), ..
            } => evaluator.get_concrete(*value).cloned(),
            _ => None,
        }
    }

    /// Every variable's recorded uses, as a sorted multiset of sites.
    fn use_lists(ssa: &SsaFunction<MockTarget>) -> Vec<Vec<(usize, usize, bool)>> {
        ssa.variables()
            .iter()
            .map(|variable| {
                let mut sites: Vec<(usize, usize, bool)> = variable
                    .uses()
                    .iter()
                    .map(|site| (site.block(), site.index(), site.is_phi_operand()))
                    .collect();
                sites.sort_unstable();
                sites
            })
            .collect()
    }

    /// Asserts the use lists are what a full recomputation would record.
    fn assert_uses_exact(ssa: &SsaFunction<MockTarget>, context: &str) {
        let mut recomputed = ssa.clone();
        recomputed.recompute_uses();
        assert_eq!(use_lists(ssa), use_lists(&recomputed), "{context}\n{ssa}");
    }

    /// `old = 1; u = add old, old; new = 5; s = add u, new; ret s`, returning 7.
    fn late_replacement_function() -> (SsaFunction<MockTarget>, [SsaVarId; 4]) {
        let mut ssa = SsaFunction::new(0, 0);
        let [old, u, new, s] = [0, 1, 2, 3].map(|index| {
            ssa.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::instruction(0, index),
                MockType::I32,
            )
        });
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: old,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: u,
            left: old,
            right: old,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: new,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Add {
            dest: s,
            left: u,
            right: new,
            flags: None,
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(s))));
        ssa.add_block(block);
        ssa.recompute_uses();
        (ssa, [old, u, new, s])
    }

    /// An insertion moves the definitions below it, and a later check in the
    /// same session judges order by where they now are.
    ///
    /// The same-block order check reads the replacement's def site. Two
    /// insertions at the top push `new` from index 2 to 4; a def site still
    /// saying 2 lets `u = add old, old` at index 3 read `new` before it is
    /// defined, and boundary repair then papers over that use with a value
    /// nothing defines.
    #[test]
    fn an_insertion_keeps_the_def_sites_it_shifts_exact() {
        let (mut ssa, [old, _, new, _]) = late_replacement_function();
        assert_eq!(
            returned(&ssa),
            Some(ConstValue::I32(7)),
            "the fixture itself"
        );
        let padding = [0, 1]
            .map(|_| ssa.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::I32));

        let mut replaced = None;
        ssa.edit(
            SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| {
                for (value, dest) in (100..).zip(padding) {
                    editor.insert_instruction(
                        0,
                        0,
                        SsaInstruction::synthetic(SsaOp::Const {
                            dest,
                            value: ConstValue::I32(value),
                        }),
                    )?;
                }
                replaced =
                    Some(editor.replace_uses_checked_with(old, new, None, Substitution::Host));
                Ok(())
            },
        )
        .expect("the edit is valid");

        let result = replaced.expect("the edit ran");
        assert_eq!(result.replaced, 0, "{ssa}");
        assert!(
            result
                .skipped
                .iter()
                .all(|skip| skip.reason == ReplacementSkipReason::SameBlockUseBeforeDef)
        );
        assert_eq!(returned(&ssa), Some(ConstValue::I32(7)), "{ssa}");
    }

    /// An insertion mid-block moves the def site of every definition below it.
    #[test]
    fn an_insertion_shifts_every_definition_below_it() {
        let (mut ssa, [_, _, new, s]) = late_replacement_function();
        let padding = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::I32);
        ssa.edit(SsaEditOptions::new(), |editor| {
            editor.insert_instruction(
                0,
                2,
                SsaInstruction::synthetic(SsaOp::Const {
                    dest: padding,
                    value: ConstValue::I32(0),
                }),
            )?;
            let function = editor.function();
            assert_eq!(
                function.variable(new).map(SsaVariable::def_site),
                Some(DefSite::instruction(0, 3))
            );
            assert_eq!(
                function.variable(s).map(SsaVariable::def_site),
                Some(DefSite::instruction(0, 4))
            );
            Ok(())
        })
        .expect("the edit is valid");
    }

    /// An indexed replacement leaves every use list exactly as a full
    /// recomputation would.
    ///
    /// The lists hold one entry per operand occurrence, and the next indexed
    /// replacement in the session iterates them: an instruction reading the
    /// value twice moves two entries, and a phi rewritten on one edge but not
    /// another keeps exactly one.
    #[test]
    fn an_indexed_replacement_keeps_the_use_lists_exact() {
        // `add old, old` moves two occurrences.
        let (mut ssa, [old, ..]) = late_replacement_function();
        let earlier = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::I32);
        ssa.block_mut(0)
            .expect("the block")
            .instructions_mut()
            .insert(
                0,
                SsaInstruction::synthetic(SsaOp::Const {
                    dest: earlier,
                    value: ConstValue::I32(3),
                }),
            );
        ssa.refresh_def_sites();
        ssa.edit(SsaEditOptions::new(), |editor| {
            let result = editor.replace_uses_checked_with(old, earlier, None, Substitution::Host);
            assert_eq!(result.replaced, 2);
            assert_uses_exact(editor.function(), "after replacing both operands of an add");
            Ok(())
        })
        .expect("the edit is valid");

        // A phi rewritten on the edge `new` reaches, and not on the other.
        let mut ssa = non_dominating_definition_function();
        let reaching = var(1);
        let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);
        let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(var(0), 1));
        phi.add_operand(PhiOperand::new(var(0), 2));
        ssa.block_mut(3).expect("the exit").add_phi(phi);
        ssa.recompute_uses();
        let eh = SsaCfg::from_ssa(&ssa);
        let dominators = compute_dominators(&eh, NodeId::new(0));
        ssa.edit(SsaEditOptions::new(), |editor| {
            let result = editor.replace_uses_checked_with(
                var(0),
                reaching,
                Some(&dominators),
                Substitution::Host,
            );
            assert_eq!(result.replaced, 1, "only the edge from B1 sees `reaching`");
            assert_uses_exact(editor.function(), "after rewriting one edge of a phi");
            Ok(())
        })
        .expect("the edit is valid");

        // Two operands on one predecessor, both rewritten by one edge.
        let mut ssa = non_dominating_definition_function();
        let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);
        let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(var(2), 2));
        phi.add_operand(PhiOperand::new(var(2), 2));
        ssa.block_mut(3).expect("the exit").add_phi(phi);
        ssa.recompute_uses();
        ssa.edit(SsaEditOptions::new(), |editor| {
            let result = editor.replace_uses_checked_with(var(2), var(3), None, Substitution::Host);
            assert_eq!(
                result.replaced, 2,
                "both phi operands; the add defines var(3)"
            );
            assert_uses_exact(editor.function(), "after rewriting a doubled phi edge");
            Ok(())
        })
        .expect("the edit is valid");
    }

    /// A closure that edits and then fails leaves the function marked dirty
    /// when its edits stay applied, and clean when they are rolled back.
    ///
    /// The scheduler reads the flag rather than the pass's own "changed"
    /// return, so an edit that survives a failed session has to be visible
    /// there, and one that was undone must not be.
    #[test]
    fn a_failed_edit_session_marks_the_function_dirty_only_when_its_edits_stay() {
        for (policy, expect_dirty) in [
            (SsaRollbackPolicy::Never, true),
            (SsaRollbackPolicy::OnFailure, false),
        ] {
            let (mut ssa, _) = late_replacement_function();
            let padding =
                ssa.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::I32);
            let _ = ssa.take_edit_dirty();
            let outcome = ssa.edit(SsaEditOptions::new().with_rollback(policy), |editor| {
                editor.insert_instruction(
                    0,
                    0,
                    SsaInstruction::synthetic(SsaOp::Const {
                        dest: padding,
                        value: ConstValue::I32(0),
                    }),
                )?;
                Err(Error::new("the pass gives up after editing"))
            });

            assert!(outcome.is_err());
            assert_eq!(
                ssa.block(0).map(|block| block.instructions().len()),
                Some(if expect_dirty { 6 } else { 5 }),
                "{policy:?}: the insertion stays only without rollback"
            );
            assert_eq!(ssa.take_edit_dirty(), expect_dirty, "{policy:?}");
        }
    }

    /// A session that changes nothing has nothing to restore, so a failure at
    /// its boundary leaves the function exactly as it was and records no failed
    /// edit.
    #[test]
    fn a_failed_session_that_changed_nothing_leaves_the_function_alone() {
        for policy in [SsaRollbackPolicy::Never, SsaRollbackPolicy::OnFailure] {
            let (mut ssa, _) = late_replacement_function();
            // A defect the session did not cause, which its verification finds.
            let ghost = ssa.create_variable(
                VariableOrigin::Local(40),
                1,
                DefSite::instruction(0, 3),
                MockType::I32,
            );
            let sink = ssa.create_variable(
                VariableOrigin::Local(41),
                0,
                DefSite::instruction(0, 0),
                MockType::I32,
            );
            if let Some(block) = ssa.block_mut(0) {
                block.instructions_mut().insert(
                    0,
                    SsaInstruction::synthetic(SsaOp::Copy {
                        dest: sink,
                        src: ghost,
                    }),
                );
            }
            ssa.refresh_def_sites();
            ssa.recompute_uses();
            let _ = ssa.take_edit_dirty();
            let before = format!("{ssa:?}");

            let outcome = ssa.edit(
                SsaEditOptions::new()
                    .with_verify(true)
                    .with_rollback(policy),
                |_| Ok(()),
            );

            assert!(outcome.is_err(), "{policy:?}: the defect is found");
            assert_eq!(format!("{ssa:?}"), before, "{policy:?}");
            assert!(!ssa.take_edit_dirty(), "{policy:?}");
            assert!(ssa.take_edit_failure().is_none(), "{policy:?}");
        }
    }

    /// A session whose only change is the declaration that the control-flow
    /// graph changed still has the function it found restored when its
    /// boundary fails: the declaration takes the rollback copy, as a mutation
    /// does. Without rollback the failure stays on the function, which the
    /// restored case must not carry.
    #[test]
    fn a_declared_cfg_change_alone_is_rolled_back() {
        for (policy, restored) in [
            (SsaRollbackPolicy::OnFailure, true),
            (SsaRollbackPolicy::Never, false),
        ] {
            let mut ssa = diamond_phi_fixture();
            // A defect the session did not cause: block 1 reads a variable
            // nothing defines.
            let ghost = ssa.create_variable(
                VariableOrigin::Local(40),
                1,
                DefSite::instruction(1, 5),
                MockType::I32,
            );
            let sink = ssa.create_variable(
                VariableOrigin::Local(41),
                0,
                DefSite::instruction(1, 0),
                MockType::I32,
            );
            if let Some(block) = ssa.block_mut(1) {
                block.instructions_mut().insert(
                    0,
                    SsaInstruction::synthetic(SsaOp::Copy {
                        dest: sink,
                        src: ghost,
                    }),
                );
            }
            ssa.refresh_def_sites();
            ssa.recompute_uses();
            let _ = ssa.take_edit_dirty();
            let _ = ssa.take_edit_failure();
            let before = format!("{ssa:?}");

            let outcome = ssa.edit(
                SsaEditOptions::new()
                    .with_verify(true)
                    .with_rollback(policy),
                |editor| {
                    editor.mark_cfg_changed();
                    Ok(())
                },
            );

            assert!(outcome.is_err(), "{policy:?}: the defect is found");
            assert_eq!(format!("{ssa:?}") == before, restored, "{policy:?}");
            assert_eq!(ssa.take_edit_dirty(), !restored, "{policy:?}");
            assert_eq!(ssa.take_edit_failure().is_some(), !restored, "{policy:?}");
        }
    }

    /// The function a rollback restores is the one the session found, however
    /// late in the session the first change comes and however many follow it.
    #[test]
    fn a_rollback_restores_the_function_as_the_session_found_it() {
        let (mut ssa, _) = late_replacement_function();
        let before = format!("{ssa:?}");
        let padding = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::I32);
        let before_padding = format!("{ssa:?}");
        assert_ne!(before, before_padding);

        let outcome = ssa.edit(
            SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| {
                // Reads first: nothing has changed yet.
                assert!(editor.function().block(0).is_some());
                for _ in 0..3 {
                    editor.insert_instruction(
                        0,
                        0,
                        SsaInstruction::synthetic(SsaOp::Const {
                            dest: padding,
                            value: ConstValue::I32(0),
                        }),
                    )?;
                }
                Err(Error::new("the pass gives up after several edits"))
            },
        );

        assert!(outcome.is_err());
        assert_eq!(format!("{ssa:?}"), before_padding);
    }

    /// An edit that changes the function and then fails verification is undone
    /// under rollback, and stays, flagged, without it.
    #[test]
    fn an_edit_that_fails_verification_is_undone_only_under_rollback() {
        for (policy, undone) in [
            (SsaRollbackPolicy::OnFailure, true),
            (SsaRollbackPolicy::Never, false),
        ] {
            let (mut ssa, _) = late_replacement_function();
            let before = format!("{ssa:?}");
            let dangling = SsaVarId::from_index(ssa.variables().len().saturating_add(5));

            let outcome = ssa.edit(
                SsaEditOptions::new()
                    .with_verify(true)
                    .with_rollback(policy),
                |editor| {
                    editor
                        .insert_instruction(
                            0,
                            0,
                            SsaInstruction::synthetic(SsaOp::Copy {
                                dest: SsaVarId::from_index(0),
                                src: dangling,
                            }),
                        )
                        .map(|_| ())
                },
            );

            assert!(outcome.is_err(), "{policy:?}: the dangling read is found");
            assert_eq!(format!("{ssa:?}") == before, undone, "{policy:?}");
        }
    }

    /// `B0` tests and goes to `B1` or straight to the join `B2`, whose phi
    /// takes `B0`'s value on that edge and `B1`'s on the other: the edge
    /// `B0 -> B2` is critical. Returns the function and `B0`'s value.
    fn triangle() -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 4);
        builder.ensure_block(2);
        let early = builder
            .in_block(0, |block| {
                let value = block.const_i32(mock_i32(), 10)?;
                let condition = block.const_bool(mock_i32(), true)?;
                block.branch(condition, 1, 2)?;
                Ok(value)
            })
            .expect("B0 builds");
        let late = builder
            .in_block(1, |block| {
                let value = block.const_i32(mock_i32(), 20)?;
                block.jump(2)?;
                Ok(value)
            })
            .expect("B1 builds");
        builder
            .in_block(2, |block| {
                let merged =
                    block.phi(SsaDefSpec::local(0, MockType::I32), [(0, early), (1, late)])?;
                block.ret(Some(merged))?;
                Ok(())
            })
            .expect("B2 builds");
        (builder.finish().expect("the triangle builds"), early)
    }

    /// **A split edge passes through a block of its own and carries what it
    /// carried.** `B0 -> B2` goes through the new `B3`, which jumps to `B2`;
    /// the phi takes `B0`'s value from `B3`, and the function verifies.
    #[test]
    fn a_split_edge_carries_its_value_through_a_block_of_its_own() {
        let (mut ssa, early) = triangle();
        let mut split = None;
        ssa.edit(SsaEditOptions::new(), |editor| {
            split = editor.split_edges(&[0], 2)?;
            Ok(())
        })
        .expect("the split commits");
        assert_eq!(split, Some(3));
        assert_eq!(
            ssa.block(0).map(SsaBlock::successors),
            Some(vec![1, 3]),
            "B0 branches to the split block"
        );
        assert_eq!(ssa.block(3).map(SsaBlock::successors), Some(vec![2]));
        let operands: Vec<(usize, SsaVarId)> = ssa
            .block(2)
            .and_then(|block| block.phi_nodes().first())
            .map(|phi| {
                phi.operands()
                    .iter()
                    .map(|operand| (operand.predecessor(), operand.value()))
                    .collect()
            })
            .unwrap_or_default();
        assert!(operands.contains(&(3, early)), "{operands:?}");
        assert!(!operands.iter().any(|(from, _)| *from == 0), "{operands:?}");
        assert_mock_valid_full(&ssa, "split edge");
    }

    /// **An edge the terminator does not take, or one inside a protected
    /// region, is not split.** `B1` does not branch to `B0`; and with `B0` to
    /// `B2` all inside a `try`, the new block would lie outside it and the
    /// edge through it would leave the region it stayed in.
    #[test]
    fn an_edge_that_is_not_there_or_stays_in_a_region_is_not_split() {
        let (mut ssa, _) = triangle();
        let mut split = Some(0);
        ssa.edit(SsaEditOptions::new(), |editor| {
            split = editor.split_edges(&[1], 0)?;
            Ok(())
        })
        .expect("nothing to commit");
        assert_eq!(split, None);
        assert_eq!(ssa.block_count(), 3);

        let (mut ssa, _) = triangle();
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 0,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(0, 3),
            handler_range: BlockRange::new(3, 3),
            filter_range: None,
            selector: None,
        }]);
        let mut split = Some(0);
        ssa.edit(SsaEditOptions::new(), |editor| {
            split = editor.split_edges(&[0], 2)?;
            Ok(())
        })
        .expect("nothing to commit");
        assert_eq!(split, None);
        assert_eq!(ssa.block_count(), 3);
    }

    /// `B0` tests and goes to `B1` or to the join `B2`; `B1` tests again and
    /// goes to `B2` or to `B3`, which enters `B2` with a value of its own.
    /// The join's phi takes `B0`'s value from `B0`, and from `B1` that value
    /// too or, where `differ`, one of `B1`'s own.
    fn chain(differ: bool) -> SsaFunction<MockTarget> {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 4);
        builder.ensure_block(3);
        let (early, condition) = builder
            .in_block(0, |block| {
                let value = block.const_i32(mock_i32(), 10)?;
                let condition = block.const_bool(mock_i32(), true)?;
                block.branch(condition, 1, 2)?;
                Ok((value, condition))
            })
            .expect("B0 builds");
        let late = builder
            .in_block(1, |block| {
                let value = block.const_i32(mock_i32(), 20)?;
                block.branch(condition, 2, 3)?;
                Ok(value)
            })
            .expect("B1 builds");
        let other = builder
            .in_block(3, |block| {
                let value = block.const_i32(mock_i32(), 30)?;
                block.jump(2)?;
                Ok(value)
            })
            .expect("B3 builds");
        builder
            .in_block(2, |block| {
                let from_one = if differ { late } else { early };
                let merged = block.phi(
                    SsaDefSpec::local(0, MockType::I32),
                    [(0, early), (1, from_one), (3, other)],
                )?;
                block.ret(Some(merged))?;
                Ok(())
            })
            .expect("B2 builds");
        builder.finish().expect("the chain builds")
    }

    /// **Edges carrying the same values share one block.** Both edges into
    /// `B2` carry `B0`'s value, so one block takes both: `B0` and `B1` branch
    /// to it, and the phi takes the value from it once. Edges carrying
    /// different values are not split.
    #[test]
    fn edges_carrying_the_same_values_share_one_block() {
        let mut ssa = chain(false);
        // Read by nothing: a session that tidied would remove it, and with it
        // a value a caller may hold facts about.
        let unread =
            ssa.create_variable(VariableOrigin::Local(9), 0, DefSite::phi(2), MockType::I32);
        let (early, other) = ssa
            .block(2)
            .and_then(|block| block.phi_nodes().first())
            .map(|phi| (phi.operands()[0].value(), phi.operands()[2].value()))
            .expect("the join's phi");
        let mut dead = PhiNode::new(unread, VariableOrigin::Local(9));
        for (value, from) in [(early, 0), (early, 1), (other, 3)] {
            dead.add_operand(PhiOperand::new(value, from));
        }
        if let Some(block) = ssa.block_mut(2) {
            block.phi_nodes_mut().push(dead);
        }
        ssa.recompute_uses();
        let bound = ssa.var_id_bound();
        let mut split = None;
        ssa.edit(SsaEditOptions::new(), |editor| {
            split = editor.split_edges(&[0, 1], 2)?;
            Ok(())
        })
        .expect("the split commits with no repair declared");
        assert_eq!(ssa.var_id_bound(), bound, "no value is renumbered");
        assert_eq!(
            ssa.block(2).map(|block| block.phi_nodes().len()),
            Some(2),
            "the unread phi is still there"
        );
        assert_eq!(split, Some(4));
        assert!(
            ssa.block(0)
                .is_some_and(|block| block.successors().contains(&4))
        );
        assert!(
            ssa.block(1)
                .is_some_and(|block| block.successors().contains(&4))
        );
        let operands: Vec<usize> = ssa
            .block(2)
            .and_then(|block| block.phi_nodes().first())
            .map(|phi| phi.operands().iter().map(PhiOperand::predecessor).collect())
            .unwrap_or_default();
        assert_eq!(operands, vec![3, 4]);
        assert_mock_valid_full(&ssa, "shared split");

        let mut ssa = chain(true);
        let mut split = Some(0);
        ssa.edit(SsaEditOptions::new(), |editor| {
            split = editor.split_edges(&[0, 1], 2)?;
            Ok(())
        })
        .expect("nothing to commit");
        assert_eq!(split, None);
        assert_eq!(ssa.block_count(), 4);
    }
}
