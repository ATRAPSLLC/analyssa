//! Editing phi nodes: their operands, predecessors and removal.

use crate::{
    BitSet,
    error::{Error, Result},
    ir::{
        function::editor::{SsaEditScope, SsaEditor},
        phi::{PhiEdge, PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
};

impl<'a, T: Target> SsaEditor<'a, T> {
    /// Adds a terminator-edge operand to one phi.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block holding the phi.
    /// * `phi_idx` - Index of the phi in the block.
    /// * `value` - The value the edge carries.
    /// * `pred` - The predecessor of the edge.
    ///
    /// # Errors
    ///
    /// Returns an error if the block or phi does not exist, or if the phi
    /// already has an operand for `pred`'s edge holding a different value.
    pub fn add_phi_operand(
        &mut self,
        block_idx: usize,
        phi_idx: usize,
        value: SsaVarId,
        pred: usize,
    ) -> Result<()> {
        let Some(phi) = self
            .ssa
            .block_mut(block_idx)
            .and_then(|block| block.phi_nodes_mut().get_mut(phi_idx))
        else {
            return Err(Error::new(format!("missing phi {phi_idx} in B{block_idx}")));
        };
        let existing = phi
            .normal_operands()
            .find(|operand| operand.predecessor() == pred)
            .map(PhiOperand::value);
        match existing {
            Some(existing) if existing == value => return Ok(()),
            Some(existing) => {
                return Err(Error::new(format!(
                    "B{block_idx} phi {phi_idx}: B{pred} already carries {existing}, not {value}"
                )));
            }
            None => phi.add_operand(PhiOperand::new(value, pred)),
        }
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(())
    }

    /// Makes the terminator-edge operand `pred` gives one phi read `value`.
    ///
    /// For a caller that knows what the edge carries better than the value it
    /// names: a select whose condition the edge decides is, on that edge, the
    /// arm the answer chooses. `value` must be available at the end of `pred`;
    /// the edit is checked when the session finishes. Only a terminator edge
    /// is touched: an operand for a throw describes a program point, not an
    /// edge.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block holding the phi.
    /// * `phi_idx` - Index of the phi in the block.
    /// * `pred` - The predecessor of the edge.
    /// * `value` - The value the edge carries from here on.
    ///
    /// # Returns
    ///
    /// Whether an operand changed.
    ///
    /// # Errors
    ///
    /// Returns an error if the block or phi does not exist.
    pub fn set_phi_operand(
        &mut self,
        block_idx: usize,
        phi_idx: usize,
        pred: usize,
        value: SsaVarId,
    ) -> Result<bool> {
        let Some(phi) = self
            .ssa
            .block_mut(block_idx)
            .and_then(|block| block.phi_nodes_mut().get_mut(phi_idx))
        else {
            return Err(Error::new(format!("missing phi {phi_idx} in B{block_idx}")));
        };
        let mut changed = false;
        for operand in phi.operands_mut() {
            if operand.edge() == PhiEdge::Normal
                && operand.predecessor() == pred
                && operand.value() != value
            {
                operand.set_value(value);
                changed = true;
            }
        }
        if changed {
            self.mark_changed(SsaEditScope::UsesOnly);
        }
        Ok(changed)
    }

    /// Gives `new_pred` the values `old_pred` carries into `block_idx`'s phis,
    /// keeping `old_pred`'s own.
    ///
    /// For a caller adding an edge that arrives with what an existing edge
    /// brings - a predecessor redirected past a forwarding block that other
    /// predecessors still use. Only terminator edges are touched: an operand
    /// for a throw describes a program point, not an edge.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block whose phis gain the edge.
    /// * `old_pred` - The predecessor whose values the new edge carries.
    /// * `new_pred` - The predecessor of the new edge.
    ///
    /// # Returns
    ///
    /// The number of operands added.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist, if a phi has no operand
    /// for `old_pred`, or if one already has an operand for `new_pred` holding
    /// a different value - two values for one edge.
    pub fn duplicate_phi_edge(
        &mut self,
        block_idx: usize,
        old_pred: usize,
        new_pred: usize,
    ) -> Result<usize> {
        let added = self.add_edge_like(block_idx, old_pred, new_pred)?;
        if added > 0 {
            self.mark_changed(SsaEditScope::InstructionsOnly);
        }
        Ok(added)
    }

    /// Rewrites phi operands that name one predecessor to another predecessor.
    ///
    /// This is useful after block coalescing, where successor phis that pointed
    /// at the removed block must instead point at the block that absorbed it.
    ///
    /// Operands for throws (see [`PhiOperand::is_exceptional`]) are retargeted
    /// with the code they describe, keeping their kind; only the terminator edge
    /// is checked for a second value.
    ///
    /// When the phi already carries an operand for `new_pred`, retargeting would
    /// leave two operands on one edge - which has no defined meaning, and which
    /// consumers disagree about: [`PhiNode::operand_from`] returns the first and
    /// discards the rest, while SCCP meets them all and yields Bottom. Both
    /// edges merging into one carry the same value only if their operands agree;
    /// when they do, the redundant operand is dropped, and when they do not the
    /// merge is not semantics-preserving and is rejected.
    ///
    /// [`PhiNode::operand_from`]: crate::ir::phi::PhiNode::operand_from
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block whose phis are rewritten.
    /// * `old_pred` - The predecessor operands currently name.
    /// * `new_pred` - The predecessor they should name instead.
    ///
    /// # Returns
    ///
    /// The number of phi operands retargeted or merged away.
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
        let updated = self.retarget_phi_operands(block_idx, old_pred, new_pred)?;
        if updated > 0 {
            self.mark_changed(SsaEditScope::CfgModifying);
        }
        Ok(updated)
    }

    /// Appends a phi node to a block.
    ///
    /// If the phi result already exists in the function variable table, its
    /// definition site is updated to the destination block. Boundary repair
    /// refreshes uses and removes any stale definitions if later edits discard
    /// the phi.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block the phi is appended to.
    /// * `phi` - The phi node to add.
    ///
    /// # Returns
    ///
    /// The block-local index of the appended phi.
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
        // A phi the caller states whole defines a value; no edge changes.
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(phi_idx)
    }

    /// Replaces phi operands from a predecessor group with one operand.
    ///
    /// Every operand whose predecessor appears in `old_preds` is removed. If
    /// at least one operand was removed, `new_operand` is appended. This models
    /// canonical CFG insertion such as replacing several incoming header edges
    /// with a single preheader or unified latch edge.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block whose phis are rewritten.
    /// * `old_preds` - Predecessors whose operands are removed.
    /// * `new_operand` - Operand appended once per phi that lost an operand.
    ///
    /// # Returns
    ///
    /// `true` if any phi was rewritten.
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

    /// Replaces the operands of **one** phi - identified by its result - that
    /// come from `old_preds`, with a single `new_operand`.
    ///
    /// Prefer this over
    /// [`replace_phi_predecessor_group_for_origin`](Self::replace_phi_predecessor_group_for_origin)
    /// whenever the caller means a specific phi. A [`VariableOrigin`] does not
    /// identify one: [`VariableOrigin::Phi`] is a unit variant, so every phi
    /// merging earlier phis - the common shape in lifted native code - shares
    /// it, and the origin-keyed form rewrites all of them to the same value.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block containing the target phi.
    /// * `phi_result` - Result variable identifying the single phi to rewrite.
    /// * `old_preds` - Predecessors whose operands are removed from that phi.
    /// * `new_operand` - Operand appended if any operand was removed.
    ///
    /// # Returns
    ///
    /// `true` if the phi was found and rewritten.
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
    /// # Arguments
    ///
    /// * `block_idx` - Block whose phis are scanned.
    /// * `origin` - Only the phi with this origin is rewritten.
    /// * `old_preds` - Predecessors whose operands are removed from that phi.
    /// * `new_operand` - Operand appended if any operand was removed.
    ///
    /// # Returns
    ///
    /// `true` if the matching phi was rewritten.
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
    /// [`SsaFunction::prune_phi_operands`](crate::ir::function::SsaFunction::prune_phi_operands). It marks the edit as
    /// [`SsaEditScope::UsesOnly`] when operands are removed, so the boundary
    /// refreshes use metadata without simplifying phis before the calling pass
    /// can account for them.
    ///
    /// # Arguments
    ///
    /// * `reachable` - Set of reachable block indices; operands from any block
    ///   not in this set are pruned.
    ///
    /// # Returns
    ///
    /// The number of phi operands removed.
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
    /// # Arguments
    ///
    /// * `block_idx` - Block containing the phi.
    /// * `phi_idx` - Block-local index of the phi to simplify.
    /// * `source` - Variable the phi result is replaced with.
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

    /// Removes a phi node at a specific block-local index.
    ///
    /// Boundary repair removes any orphaned phi result variables and refreshes
    /// def-use metadata. This helper intentionally does not replace uses of the
    /// phi result; callers must insert or otherwise provide a replacement
    /// definition when uses remain.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block containing the phi.
    /// * `phi_idx` - Block-local index of the phi to remove.
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
}
