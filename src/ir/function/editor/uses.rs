//! Rewriting reads: replacing a variable's uses, checked against types and
//! dominance, and propagating copies.

use std::collections::BTreeMap;

use crate::{
    bitset::BitSet,
    error::Result,
    facts::Substitution,
    graph::algorithms::DominatorTree,
    ir::{
        function::{
            CopyPropagationResult, Reconstruction, ValueUpdate,
            editor::{CheckedReplaceResult, Rewiring, SsaEditScope, SsaEditor},
            transforms::Cleanup,
        },
        variable::SsaVarId,
    },
    target::Target,
};

impl<'a, T: Target> SsaEditor<'a, T> {
    /// Replaces instruction uses after shared safety checks.
    ///
    /// The edit scope is widened to [`SsaEditScope::UsesOnly`] when at least
    /// one operand occurrence is replaced.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable whose instruction uses are rewritten.
    /// * `new_var` - The variable substituted at each accepted use site.
    ///
    /// # Returns
    ///
    /// The replacement count and the use sites the safety checks skipped.
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

    /// Reconstructs the values an edge this session added cut off from their
    /// definition.
    ///
    /// See [`crate::ir::function::ValueUpdate`] and its module. A transform
    /// that reroutes an edge names the values it disturbed and what the new
    /// edge carries for each, and gets back the phis that answer for them,
    /// without a reconstruction of the whole function and without the rename
    /// groups one would need.
    ///
    /// # Arguments
    ///
    /// * `updates` - The values to reconstruct and the values the new edges
    ///   carry.
    ///
    /// # Returns
    ///
    /// The phis placed and the reads rewritten.
    ///
    /// # Errors
    ///
    /// Returns an error when a phi the reconstruction needs has a predecessor
    /// the value does not reach; the session's rollback policy then decides
    /// what happens to the edit that asked.
    pub fn reconstruct_values(&mut self, updates: &[ValueUpdate]) -> Result<Reconstruction> {
        let reconstruction = self.ssa.reconstruct_values(updates)?;
        if reconstruction.changed() {
            self.mark_changed(SsaEditScope::InstructionsOnly);
        }
        Ok(reconstruction)
    }

    /// Carries a rename group and the host's facts across a substitution; see
    /// [`SsaFunction::inherit_rename_group`](crate::ir::function::SsaFunction::inherit_rename_group).
    fn carry_rename_group(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        took: bool,
        cause: Substitution,
    ) {
        self.ssa.inherit_rename_group(old_var, new_var, took, cause);
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
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable whose instruction uses are rewritten.
    /// * `new_var` - The variable substituted at each accepted use site.
    /// * `dominators` - Dominator tree reused across the batch, or `None` to
    ///   skip cross-block uses that would require one.
    /// * `cause` - What substituted the value; [`Substitution::Host`] for a
    ///   host's own substitution. It reaches the host's facts, which is how a
    ///   host tells a value numbering apart from a copy it forwarded itself.
    ///
    /// # Returns
    ///
    /// The replacement count and the use sites the safety checks skipped.
    pub fn replace_uses_checked_with(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        dominators: Option<&DominatorTree>,
        cause: Substitution,
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
            .replace_uses_checked_indexed(old_var, new_var, dominators, None);
        if result.replaced > 0 {
            self.mark_changed(SsaEditScope::UsesOnly);
            // `replace_uses_checked_indexed` moved the rewritten sites between
            // the two variables, so the lists are still exact.
            self.uses_indexed = true;
        }
        self.carry_rename_group(old_var, new_var, result.took(), cause);
        result
    }

    /// Replaces the uses of `old_var` with `new_var` in the blocks `within`
    /// alone, after the shared checks: an instruction's use where it lies in
    /// one, and a phi operand's where its edge leaves one.
    ///
    /// For a caller that knows two values are one only in part of the
    /// function: a branch on their equality decides they are on the edge it
    /// takes, and in every block that edge dominates. Uses elsewhere keep
    /// `old_var`, which therefore keeps its definition and its facts. The
    /// readers rewritten take in what they read of `old_var`
    /// ([`crate::ir::function::SsaFunction::substituted_facts`]), but `new_var` takes nothing of
    /// it, neither its host facts nor its rename group: the two are one value
    /// only in `within`, and `new_var` is live outside it, at the branch that
    /// decided the equality if nowhere else. Naming it after `old_var`'s
    /// storage would name it so where that storage holds another value.
    ///
    /// # Arguments
    ///
    /// * `old_var` - The variable whose uses in `within` are rewritten.
    /// * `new_var` - The variable substituted at each accepted use site.
    /// * `within` - The blocks the rewrite reaches.
    /// * `dominators` - Dominator tree reused across the batch, or `None` to
    ///   skip cross-block uses that would require one.
    ///
    /// # Returns
    ///
    /// The replacement count and the use sites the safety checks skipped.
    pub fn replace_uses_within(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
        within: &BitSet,
        dominators: Option<&DominatorTree>,
    ) -> CheckedReplaceResult {
        if !self.uses_indexed {
            self.ssa.recompute_uses();
            self.uses_indexed = true;
        }
        let result =
            self.ssa
                .replace_uses_checked_indexed(old_var, new_var, dominators, Some(within));
        if result.replaced > 0 {
            self.mark_changed(SsaEditScope::UsesOnly);
            self.uses_indexed = true;
        }
        result
    }

    /// Propagates copy mappings through instruction operands.
    ///
    /// The edit scope is widened to [`SsaEditScope::UsesOnly`] when at least
    /// one operand occurrence is replaced.
    ///
    /// # Arguments
    ///
    /// * `copies` - Map from each copy destination to the source it should be
    ///   rewritten to.
    ///
    /// # Returns
    ///
    /// The total occurrences replaced plus the sets of fully and partially
    /// propagated copy destinations.
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
}
