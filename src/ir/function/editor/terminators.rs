//! Editing terminators: replacing them, folding them and redirecting their
//! targets, with the phis the folds strand.

use crate::{
    analysis::exceptions::FunctionRoots,
    error::{Error, Result},
    ir::{
        block::SsaBlock,
        function::editor::{SsaEditScope, SsaEditor},
        instruction::SsaInstruction,
        ops::SsaOp,
        variable::SsaVarId,
    },
    target::Target,
};

impl<'a, T: Target> SsaEditor<'a, T> {
    /// Replaces the terminator operation for a block.
    ///
    /// The terminator is represented by the block's last instruction. This
    /// helper marks the edit as [`SsaEditScope::CfgModifying`] when the new
    /// terminator's successor edges differ from the old one's - changing a
    /// terminator can change them even when the instruction count stays fixed -
    /// and as an instruction edit when they are the same edges.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block whose terminator is replaced.
    /// * `new_op` - The operation installed as the block's last instruction.
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
        // A terminator that keeps every edge - `branch` rewritten as the
        // `branchcmp` it tests, a condition swapped for an equivalent one -
        // changes what decides the flow and not the flow. The graph every phi is
        // keyed by is the one it was, so the edit needs no reconstruction.
        let mut before = block.successors();
        let mut after = new_op.successors();
        before.sort_unstable();
        after.sort_unstable();

        self.replace_instruction_op(block_idx, instr_idx, new_op)?;
        if before != after {
            self.mark_changed(SsaEditScope::CfgModifying);
        }
        Ok(())
    }

    /// Folds a block's terminator to a new op and repairs the phi operands of
    /// any successor whose incoming edge that removes.
    ///
    /// This is the cheap path for branch folding - replacing
    /// `Branch(cond, A, B)` with `Jump(A)`. Such an edit is *not* a general CFG
    /// mutation and needs no [`SsaFunction::reconstruct_values`](crate::ir::function::SsaFunction::reconstruct_values):
    ///
    /// - Removing an edge only ever **increases** dominance, because it removes
    ///   paths. Every definition that dominated a use still does, so no
    ///   def-use violation can be introduced and **no new phi can ever be
    ///   required** - existing phis can only become redundant.
    /// - The one thing that *does* need fixing is each dropped successor's phi
    ///   operands for this predecessor, which this method prunes directly.
    ///
    /// The edit therefore settles at [`SsaEditScope::InstructionsOnly`]
    /// (`repair_ssa`): a session that only folds terminators this way needs
    /// no value reconstruction and no repair declaration.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block whose terminator is folded.
    /// * `new_op` - The folded terminator; it may only drop successor edges,
    ///   never add one.
    ///
    /// # Returns
    ///
    /// The number of phi operands pruned from the dropped successors.
    ///
    /// # Errors
    ///
    /// Returns an error if the block or its terminator does not exist, or if
    /// `new_op` would **add** a successor edge - the reasoning above holds only
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
        // that argument - it can make a definition stop dominating a use, and it
        // can require a phi that does not exist - so it is rejected here rather
        // than silently under-repaired. Use
        // [`SsaEditor::redirect_terminator_target`] or
        // [`SsaEditor::replace_terminator_op`] for the general case; both record
        // [`SsaEditScope::CfgModifying`].
        if let Some(added) = after.iter().find(|target| !before.contains(target)) {
            return Err(Error::new(format!(
                "fold_terminator_pruning_phis on B{block_idx} would add edge to B{added}; \
                 this path only prunes removed edges - use `replace_terminator_op` \
                 for an edit that adds one"
            )));
        }

        self.replace_instruction_op(block_idx, last_idx, new_op)?;

        // Prune only the edges this fold actually removed. A successor that
        // still has another edge from this block (e.g. `Branch(c, A, A)`)
        // keeps its operand.
        let mut pruned = 0usize;
        let mut dropped = Vec::new();
        for successor in before {
            if after.contains(&successor) || dropped.contains(&successor) {
                continue;
            }
            dropped.push(successor);
            if let Some(target) = self.ssa.block_mut(successor) {
                for phi in target.phi_nodes_mut() {
                    let operands = phi.operands_mut();
                    let original = operands.len();
                    // Never leave a phi empty; a single-operand phi is handled
                    // by trivial-phi simplification, not here - unless this was
                    // the block's last incoming edge, which is settled below.
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
        for successor in dropped {
            self.settle_stranded_phis(successor);
        }

        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(pruned)
    }

    /// Turns the phis of `block_idx` into copies of the value each still
    /// names, when a fold has removed the block's last incoming edge.
    ///
    /// A phi operand names the edge its value arrives along, and a block that
    /// nothing transfers or throws to has no edges: the verifier rejects a phi
    /// there. The fold keeps one operand rather than empty a phi, expecting
    /// trivial-phi simplification to retire the rest, and in a stranded block
    /// that expectation fails - the simplification substitutes under
    /// dominance, and nothing dominates a block the entry cannot reach, so
    /// every substitution is refused and the phi stays. Two folds that remove
    /// both of a block's edges were enough to leave the function unverifiable,
    /// and the pass group that ran them was rolled back with every other
    /// pass's work in it.
    ///
    /// The block is dead code, and deleting it belongs to the pass that owns
    /// reachability. Until then each of its phis is what it last merged: a copy
    /// of the surviving operand, in phi order ahead of the block's
    /// instructions, so every read of the result is still defined.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - A block the fold removed an edge to.
    fn settle_stranded_phis(&mut self, block_idx: usize) {
        let has_phis = self
            .ssa
            .block(block_idx)
            .is_some_and(|block| !block.phi_nodes().is_empty());
        if !has_phis
            || self
                .ssa
                .blocks()
                .iter()
                .any(|block| block.successors().contains(&block_idx))
            || !FunctionRoots::of(&self.ssa)
                .exceptional_predecessors(block_idx)
                .is_empty()
        {
            return;
        }
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return;
        };
        // The fold never empties a phi, so each has the operand it last kept;
        // one without (never the fold's doing) is left for the verifier to name.
        let Some(copies) = block
            .phi_nodes()
            .iter()
            .map(|phi| Some((phi.result(), phi.operands().first()?.value())))
            .collect::<Option<Vec<(SsaVarId, SsaVarId)>>>()
        else {
            return;
        };
        block.phi_nodes_mut().clear();
        for (at, (dest, src)) in copies.into_iter().enumerate() {
            block
                .instructions_mut()
                .insert(at, SsaInstruction::synthetic(SsaOp::Copy { dest, src }));
        }
        self.refresh_def_sites_from(block_idx, 0);
    }

    /// Redirects one successor edge in a block terminator.
    ///
    /// The block's last instruction is treated as its terminator. If that
    /// operation references `old_target`, all matching target fields are
    /// rewritten to `new_target` through [`SsaOp::redirect_target`].
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block whose terminator is retargeted.
    /// * `old_target` - The successor block currently referenced.
    /// * `new_target` - The successor block to point at instead.
    ///
    /// # Returns
    ///
    /// `true` if the terminator referenced `old_target` and was rewritten.
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
        // The edge keeps what it passed, which runs on the way out of the
        // block ([`BlockProvenance::retarget`]).
        if let Some(block) = self.ssa.block_mut(block_idx) {
            block
                .provenance_mut()
                .retarget(|edge| Some(if edge == old_target { new_target } else { edge }));
        }
        Ok(true)
    }
}
