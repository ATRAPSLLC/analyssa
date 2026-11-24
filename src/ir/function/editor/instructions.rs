//! Editing instructions: replacing, inserting, moving, duplicating and removing
//! them, and the definition sites they record.

use crate::{
    error::{Error, Result},
    facts::InstructionFacts,
    ir::{
        function::{
            editor::{SsaEditScope, SsaEditor},
            transforms::Cleanup,
        },
        identity::Identity,
        instruction::{InstrId, SsaInstruction},
        ops::SsaOp,
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
};

impl<'a, T: Target> SsaEditor<'a, T> {
    /// Replaces the copy instruction defining `dest` with [`SsaOp::Nop`].
    ///
    /// Boundary repair strips the nop and compacts variables once the edit
    /// session finishes.
    ///
    /// # Arguments
    ///
    /// * `dest` - The variable whose defining copy instruction is nopped.
    ///
    /// # Returns
    ///
    /// `true` if a copy defining `dest` was found and replaced.
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
    /// # Arguments
    ///
    /// * `block_idx` - Block containing the instruction to rewrite.
    /// * `instr_idx` - Index of the instruction within that block.
    /// * `new_op` - The operation installed in place of the current one.
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
        let Some(block) = self.ssa.block(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };
        if block.instructions().get(instr_idx).is_none() {
            return Err(Error::new(format!(
                "missing instruction B{block_idx}:I{instr_idx}"
            )));
        }
        // The function's rewrite, which folds in the facts of the operands
        // the new operation no longer reads.
        self.ssa
            .replace_instruction_op(block_idx, instr_idx, new_op);
        for dest in defs {
            if let Some(var) = self.ssa.variable_mut(dest) {
                var.set_def_site(DefSite::instruction(block_idx, instr_idx));
            }
        }
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(())
    }

    /// Replaces an instruction operation and retypes the variable it defines.
    ///
    /// [`Self::replace_instruction_op`] keeps the destination's type, which is
    /// right for a rewrite that computes the same value. This is for one that
    /// computes the same bits as a different kind of value: an address sum
    /// proved to be integer addition keeps its bits and stops being an address.
    /// The type is stated on both the instruction's result and the variable, so
    /// no reader of either takes the old one at its word.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block containing the instruction to rewrite.
    /// * `instr_idx` - Index of the instruction within that block.
    /// * `new_op` - The operation installed in place of the current one.
    /// * `dest_type` - The type of the value `new_op` defines.
    ///
    /// # Errors
    ///
    /// Returns an error if the target instruction does not exist, or if
    /// `new_op` defines no variable to retype.
    pub fn replace_instruction_op_as(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        new_op: SsaOp<T>,
        dest_type: T::Type,
    ) -> Result<()> {
        let Some(dest) = new_op.dest() else {
            return Err(Error::new(format!(
                "B{block_idx}:I{instr_idx}: a retyped replacement defines no variable"
            )));
        };
        self.replace_instruction_op(block_idx, instr_idx, new_op)?;
        if let Some(instr) = self
            .ssa
            .block_mut(block_idx)
            .and_then(|block| block.instruction_mut(instr_idx))
        {
            instr.set_result_type(Some(dest_type.clone()));
        }
        if let Some(var) = self.ssa.variable_mut(dest) {
            var.set_type(dest_type);
        }
        Ok(())
    }

    /// Folds `twin` into the facts of the instruction at `instr_idx` of
    /// `block_idx`: the facts of an instruction doing the same work at
    /// another point, which goes as this one does it there too
    /// ([`InstructionFacts::absorb_twin`]).
    ///
    /// Only the facts change, so the session repairs nothing for it.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block containing the instruction that stands for it.
    /// * `instr_idx` - Index of the instruction within that block.
    /// * `twin` - The facts of the instruction it stands for.
    ///
    /// # Errors
    ///
    /// Returns an error if the target instruction does not exist.
    pub fn absorb_twin_facts(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        twin: &T::InstructionFacts,
    ) -> Result<()> {
        if self
            .ssa
            .block(block_idx)
            .and_then(|block| block.instructions().get(instr_idx))
            .is_none()
        {
            return Err(Error::new(format!(
                "missing instruction {instr_idx} in block B{block_idx}"
            )));
        }
        self.mark_changed(SsaEditScope::None);
        if let Some(instr) = self
            .ssa
            .block_mut(block_idx)
            .and_then(|block| block.instructions_mut().get_mut(instr_idx))
        {
            instr.facts_mut().absorb_twin(twin);
        }
        Ok(())
    }

    /// Replaces an instruction with [`SsaOp::Nop`].
    ///
    /// This is the checked editor equivalent of instruction removal. Boundary
    /// repair strips the nop and compacts variables once the edit session
    /// finishes.
    ///
    /// The instruction is destroyed, so its [`InstrId`] goes with it: what is
    /// left behind is a placeholder holding a position, not the instruction
    /// that was there. The id is never reissued, so a host table still keyed by
    /// it reads as absent rather than as some later instruction. A pass moving
    /// an instruction rather than destroying it takes it out with
    /// [`detach_instruction`](Self::detach_instruction), which hands the
    /// identity over with the instruction.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block containing the instruction to remove.
    /// * `instr_idx` - Index of the instruction within that block.
    ///
    /// # Errors
    ///
    /// Returns an error if the target instruction does not exist.
    pub fn nop_instruction(&mut self, block_idx: usize, instr_idx: usize) -> Result<()> {
        self.replace_instruction_op(block_idx, instr_idx, SsaOp::Nop)?;
        if let Some(instr) = self
            .ssa
            .block_mut(block_idx)
            .and_then(|block| block.instructions_mut().get_mut(instr_idx))
        {
            instr.clear_identity();
        }
        Ok(())
    }

    /// Removes the listed instructions whose results nothing else reads, within
    /// the session.
    ///
    /// The set is settled as [`SsaFunction::remove_unread_instructions`](crate::ir::function::SsaFunction::remove_unread_instructions)
    /// settles it, and the session repairs what the removals leave at its
    /// boundary, so a pass that also edits the graph in the same session
    /// leaves the function repaired whole.
    ///
    /// # Arguments
    ///
    /// * `sites` - `(block, instruction)` locations of the instructions to
    ///   remove.
    ///
    /// # Returns
    ///
    /// The removed instructions in location order, each with the operation it
    /// held.
    pub fn remove_unread_instructions(
        &mut self,
        sites: impl IntoIterator<Item = (usize, usize)>,
    ) -> Vec<(usize, usize, SsaOp<T>)> {
        let removed = self.ssa.remove_unread_instructions(sites);
        if !removed.is_empty() {
            self.mark_changed(SsaEditScope::InstructionsOnly);
        }
        removed
    }

    /// Takes an instruction out, leaving a [`SsaOp::Nop`] placeholder in its
    /// slot.
    ///
    /// The returned instruction keeps its identity, its host payload and its
    /// resolved result type, so re-inserting it elsewhere is a *move*: the same
    /// instruction, somewhere else. The placeholder keeps every later index in
    /// the block where it was, which is what lets a pass plan a set of moves
    /// over positions it computed in advance; boundary repair strips it.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block holding the instruction.
    /// * `instr_idx` - Index of the instruction within that block.
    ///
    /// # Errors
    ///
    /// Returns an error when the instruction does not exist.
    pub fn detach_instruction(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
    ) -> Result<SsaInstruction<T>> {
        let taken = self
            .ssa
            .block(block_idx)
            .and_then(|block| block.instructions().get(instr_idx))
            .cloned()
            .ok_or_else(|| Error::new(format!("B{block_idx} holds no instruction {instr_idx}")))?;
        self.nop_instruction(block_idx, instr_idx)?;
        Ok(taken)
    }

    /// Removes all instructions in a block starting at `start_idx`.
    ///
    /// This is intended for dead-tail cleanup after a terminator. The function
    /// truncates the instruction list and marks instruction repair required so
    /// shifted or removed definition sites are reconciled at the edit boundary.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block to truncate.
    /// * `start_idx` - First instruction index to drop; indices at or beyond the
    ///   block length remove nothing.
    ///
    /// # Returns
    ///
    /// The number of instructions removed.
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

    /// Defines `value`, of `ty`, at the end of `block`, before its
    /// terminator: a constant a phi joins from that predecessor.
    ///
    /// # Errors
    ///
    /// Returns an error if `block` does not exist.
    pub fn constant_at_exit(
        &mut self,
        block: usize,
        ty: T::Type,
        value: ConstValue<T>,
    ) -> Result<SsaVarId> {
        let at = self
            .ssa
            .block(block)
            .map_or(0, |block| block.instructions().len());
        let constant =
            self.create_fresh_variable(VariableOrigin::Phi, 0, DefSite::instruction(block, at), ty);
        self.insert_before_terminator(
            block,
            SsaInstruction::synthetic(SsaOp::Const {
                dest: constant,
                value,
            }),
        )?;
        Ok(constant)
    }

    /// Inserts an instruction before the block terminator.
    ///
    /// If the block has no terminator, the instruction is appended. The
    /// inserted instruction's definitions get its location as their def site,
    /// and every definition it pushes down is moved with it, so later checks in
    /// the same session judge order by where instructions now are.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block to insert into.
    /// * `instr` - The instruction to insert.
    ///
    /// # Returns
    ///
    /// The index at which the instruction was inserted.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn insert_before_terminator(
        &mut self,
        block_idx: usize,
        mut instr: SsaInstruction<T>,
    ) -> Result<usize> {
        self.identify(&mut instr);
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let insert_idx = block
            .instructions()
            .iter()
            .position(SsaInstruction::is_terminator)
            .unwrap_or_else(|| block.instructions().len());
        block.instructions_mut().insert(insert_idx, instr);

        self.refresh_def_sites_from(block_idx, insert_idx);
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(insert_idx)
    }

    /// Inserts an instruction at a specific index within a block.
    ///
    /// The index is clamped to the end of the instruction list, so callers can
    /// use a large index to append. The inserted instruction's definitions get
    /// the insertion location as their def site, and every definition below it
    /// is moved down by one. The same-block order check of a later
    /// replacement in this session reads those def sites, so a stale one would
    /// let a use be rewritten to a value defined after it.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Block to insert into.
    /// * `instr_idx` - Requested insertion index, clamped to the block length.
    /// * `instr` - The instruction to insert.
    ///
    /// # Returns
    ///
    /// The index at which the instruction was actually inserted.
    ///
    /// # Errors
    ///
    /// Returns an error if the block does not exist.
    pub fn insert_instruction(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        mut instr: SsaInstruction<T>,
    ) -> Result<usize> {
        self.identify(&mut instr);
        let Some(block) = self.ssa.block_mut(block_idx) else {
            return Err(Error::new(format!("missing block B{block_idx}")));
        };

        let insert_idx = instr_idx.min(block.instructions().len());
        block.instructions_mut().insert(insert_idx, instr);

        self.refresh_def_sites_from(block_idx, insert_idx);
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(insert_idx)
    }

    /// Moves one instruction to another position, keeping its identity.
    ///
    /// The instruction is the same instruction somewhere else, so it keeps its
    /// [`InstrId`], its host payload and its resolved result type. A pass that
    /// relocates code - hoisting an invariant out of a loop, folding a block
    /// into its predecessor - uses this rather than rebuilding the operation as
    /// a synthetic instruction, which would discard all three.
    ///
    /// # Arguments
    ///
    /// * `from` - Where the instruction is now, as `(block, index)`.
    /// * `to` - Where it goes, as `(block, index)`. The index is clamped to the
    ///   destination block's length, and is read *after* the instruction is
    ///   removed from its old position.
    ///
    /// # Returns
    ///
    /// The identity of the moved instruction.
    ///
    /// # Errors
    ///
    /// Returns an error when either block is missing or `from` names no
    /// instruction.
    pub fn move_instruction(
        &mut self,
        from: (usize, usize),
        to: (usize, usize),
    ) -> Result<InstrId> {
        let (from_block, from_idx) = from;
        let (to_block, to_idx) = to;
        if self.ssa.block(to_block).is_none() {
            return Err(Error::new(format!("missing block B{to_block}")));
        }
        let Some(block) = self.ssa.block_mut(from_block) else {
            return Err(Error::new(format!("missing block B{from_block}")));
        };
        if from_idx >= block.instructions().len() {
            return Err(Error::new(format!(
                "B{from_block} holds no instruction {from_idx}"
            )));
        }
        let mut instr = block.instructions_mut().remove(from_idx);
        let id = match instr.id() {
            Some(id) => id,
            None => {
                let id = self.ssa.mint_instr_id();
                instr.set_id(id);
                id
            }
        };
        let Some(destination) = self.ssa.block_mut(to_block) else {
            return Err(Error::new(format!("missing block B{to_block}")));
        };
        let insert_idx = to_idx.min(destination.instructions().len());
        destination.instructions_mut().insert(insert_idx, instr);

        self.refresh_def_sites_from(from_block, from_idx);
        self.refresh_def_sites_from(to_block, insert_idx);
        self.mark_changed(SsaEditScope::InstructionsOnly);
        Ok(id)
    }

    /// Inserts a copy of one instruction under a fresh identity.
    ///
    /// The copy is a *different* instruction - a block duplicated along a
    /// threaded path holds its own copy of each operation - so it gets its own
    /// [`InstrId`], and the function records where it came from, which
    /// [`SsaFunction::instruction_provenance`](crate::ir::function::SsaFunction::instruction_provenance) answers.
    ///
    /// # Arguments
    ///
    /// * `source` - The instruction to copy, as `(block, index)`.
    /// * `to` - Where the copy goes, as `(block, index)`.
    ///
    /// # Returns
    ///
    /// The identity of the copy.
    ///
    /// # Errors
    ///
    /// Returns an error when either block is missing or `source` names no
    /// instruction.
    pub fn duplicate_instruction(
        &mut self,
        source: (usize, usize),
        to: (usize, usize),
    ) -> Result<InstrId> {
        let (source_block, source_idx) = source;
        let copy = self
            .ssa
            .block(source_block)
            .and_then(|block| block.instructions().get(source_idx))
            .map(SsaInstruction::duplicated)
            .ok_or_else(|| {
                Error::new(format!("B{source_block} holds no instruction {source_idx}"))
            })?;
        let source_id = self
            .ssa
            .block(source_block)
            .and_then(|block| block.instructions().get(source_idx))
            .and_then(SsaInstruction::id);
        let (to_block, to_idx) = to;
        let index = self.insert_instruction(to_block, to_idx, copy)?;
        let id = self
            .ssa
            .block(to_block)
            .and_then(|block| block.instructions().get(index))
            .and_then(SsaInstruction::id)
            .ok_or_else(|| Error::new("duplicated instruction lost its identity"))?;
        if let Some(source_id) = source_id {
            self.ssa.record_instruction_provenance(id, source_id);
        }
        Ok(id)
    }
}
