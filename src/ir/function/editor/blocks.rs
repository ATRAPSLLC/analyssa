//! Clearing, removing and appending blocks.

use std::collections::BTreeSet;

use crate::{
    BitSet,
    analysis::cfg::SsaCfg,
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
    /// Clears all phis and instructions from a block.
    ///
    /// Clearing a block changes CFG structure by removing its terminator and
    /// definitions, so boundary cleanup uses full SSA rebuild.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block to empty of all phis and instructions.
    ///
    /// # Returns
    ///
    /// `true` if the block had content that was cleared, `false` if it was
    /// already empty.
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

    /// Empties every block outside `reachable` and drops the phi operands that
    /// named them.
    ///
    /// **Reconstruction only when something still needs it.** A block nothing
    /// reaches dominates nothing reachable, so clearing it moves no definition
    /// away from a use - except where reachable code still reads what such a
    /// block defined, which a CFG edit upstream can leave behind. The
    /// side-effect-free definitions among those (`Const`, `LoadToken`,
    /// `LoadStaticField`) are copied into the entry block first, exactly as
    /// the rebuild would; any other one still read means the function needs
    /// the rebuild, and the edit is recorded as
    /// [`SsaEditScope::CfgModifying`] as before. Otherwise it settles without
    /// one: a rebuild recomputes every use from its rename group, which on
    /// optimised SSA - where a group can hold unrelated values - reads some of
    /// them as undefined.
    ///
    /// # Arguments
    ///
    /// * `reachable` - The blocks that stay; every other one is cleared.
    ///
    /// # Returns
    ///
    /// The blocks cleared.
    pub fn clear_unreached_blocks(&mut self, reachable: &BitSet) -> Vec<usize> {
        let block_count = self.ssa.block_count();
        let mut read_from_reachable: BTreeSet<SsaVarId> = BTreeSet::new();
        for block_idx in reachable.iter() {
            let Some(block) = self.ssa.block(block_idx) else {
                continue;
            };
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    if reachable.contains_checked(operand.predecessor()) {
                        read_from_reachable.insert(operand.value());
                    }
                }
            }
            for instr in block.instructions() {
                instr.op().for_each_use(|used| {
                    read_from_reachable.insert(used);
                });
            }
        }

        let mut rescued: Vec<SsaInstruction<T>> = Vec::new();
        let mut needs_rebuild = false;
        let mut cleared = Vec::new();
        for block_idx in 0..block_count {
            if reachable.contains_checked(block_idx) {
                continue;
            }
            let Some(block) = self.ssa.block(block_idx) else {
                continue;
            };
            if block.is_empty() {
                continue;
            }
            for phi in block.phi_nodes() {
                if read_from_reachable.contains(&phi.result()) {
                    needs_rebuild = true;
                }
            }
            for instr in block.instructions() {
                if !instr.defs().any(|dest| read_from_reachable.contains(&dest)) {
                    continue;
                }
                if matches!(
                    instr.op(),
                    SsaOp::Const { .. } | SsaOp::LoadToken { .. } | SsaOp::LoadStaticField { .. }
                ) {
                    rescued.push(instr.clone());
                } else {
                    needs_rebuild = true;
                }
            }
            cleared.push(block_idx);
        }
        if cleared.is_empty() {
            return cleared;
        }

        if !rescued.is_empty()
            && let Some(entry) = self.ssa.block_mut(0)
        {
            let at = entry
                .instructions()
                .iter()
                .position(SsaInstruction::is_terminator)
                .unwrap_or(entry.instructions().len());
            for (offset, instr) in rescued.into_iter().enumerate() {
                entry
                    .instructions_mut()
                    .insert(at.saturating_add(offset), instr);
            }
        }
        for &block_idx in &cleared {
            if let Some(block) = self.ssa.block_mut(block_idx) {
                block.clear();
            }
        }
        self.ssa.prune_phi_operands(reachable);
        self.mark_changed(if needs_rebuild {
            SsaEditScope::CfgModifying
        } else {
            SsaEditScope::InstructionsOnly
        });
        cleared
    }

    /// Empties a block nothing reaches and drops the phi operands naming it.
    ///
    /// **No reconstruction.** A block with no way in dominates nothing
    /// reachable and brings nothing anywhere; the operands that named it were
    /// for paths the function no longer has.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block to remove.
    ///
    /// # Returns
    ///
    /// `true` if the block held anything.
    ///
    /// # Errors
    ///
    /// Returns an error if the block is the entry, has a predecessor, or is
    /// the only way into some phi's block.
    pub fn remove_unreached_block(&mut self, block_idx: usize) -> Result<bool> {
        if block_idx == 0 {
            return Err(Error::new("the entry block is always reached"));
        }
        let cfg = SsaCfg::from_ssa(&self.ssa);
        if cfg.predecessor_blocks(block_idx).next().is_some() {
            return Err(Error::new(format!("B{block_idx} still has a predecessor")));
        }
        drop(cfg);
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };
        if block.is_empty() {
            return Ok(false);
        }
        block.clear();
        let block_count = self.ssa.block_count();
        for other in 0..block_count {
            let Some(block) = self.ssa.block_mut(other) else {
                continue;
            };
            for phi in block.phi_nodes_mut() {
                if phi
                    .operands()
                    .iter()
                    .all(|operand| operand.predecessor() == block_idx)
                    && !phi.operands().is_empty()
                {
                    return Err(Error::new(format!(
                        "B{other} is reached only through B{block_idx}"
                    )));
                }
                phi.operands_mut()
                    .retain(|operand| operand.predecessor() != block_idx);
            }
        }
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(true)
    }

    /// Appends a basic block to the function.
    ///
    /// The block id must match the next dense block index. This keeps block
    /// creation explicit and avoids silently introducing an inconsistent block
    /// id that would later require canonicalization to interpret.
    ///
    /// # Arguments
    ///
    /// * `block` - The block to append; its id must equal the current block
    ///   count.
    ///
    /// # Returns
    ///
    /// The index at which the block was appended.
    ///
    /// # Errors
    ///
    /// Returns an error if the block id is not equal to the current block
    /// count.
    pub fn append_block(&mut self, mut block: SsaBlock<T>) -> Result<usize> {
        let block_idx = self.ssa.block_count();
        if block.id() != block_idx {
            return Err(Error::new(format!(
                "new block id B{} does not match next block B{block_idx}",
                block.id()
            )));
        }
        for instr in block.instructions_mut() {
            self.identify(instr);
        }

        self.ssa.add_block(block);
        self.mark_changed(SsaEditScope::CfgModifying);
        Ok(block_idx)
    }
}
