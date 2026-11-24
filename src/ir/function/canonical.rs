//! SSA function canonicalization: final cleanup before code generation.
//!
//! After all deobfuscation and optimization passes complete, [`SsaFunction::canonicalize`]
//! performs final cleanup to produce clean, valid SSA for code generation:
//!
//! # Canonicalization Pipeline
//!
//! | Phase | Step | Description |
//! |-------|------|-------------|
//! | 1 | Strip Nops | Remove all `Nop` instructions |
//! | 2 | Protect blocks | Mark exception handler entries and Leave targets as protected |
//! | 3 | Identify empties | Mark blocks with no instructions/phis for removal |
//! | 4 | Build redirect map | For removed empty blocks, find ultimate jump targets |
//! | 5 | Update branch targets | Retarget jumps to skip removed blocks |
//! | 6 | Update phi predecessors | Fix phi operands for removed predecessors |
//! | 7 | Compact blocks | Remove empty blocks and renumber survivors |
//! | 8 | Remap exception handlers | Update handler block indices |
//! | 9 | Ensure valid terminator | Replace dead code with valid Return |
//!
//! # Edge Cases
//!
//! - **Protected blocks**: Exception handler entry blocks and Leave targets are
//!   never removed, even if empty, to preserve valid exception handling.
//! - **Orphaned phi operands**: When a removed block's kept predecessors
//!   can't be determined, orphaned operands are distributed to unaccounted-for
//!   predecessors as a best-effort recovery.
//! - **All-code-removed**: If after stripping, the method has no useful code
//!   (only jumps between empty blocks), the entry block is replaced with a
//!   simple `Return` to produce valid IL.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    BitSet,
    facts::Derivation,
    ir::{
        block::{BlockProvenance, SsaBlock},
        function::{SsaFunction, transforms::Cleanup},
        identity::Identity,
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::PhiOperand,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
};

/// Finds the surviving predecessors that flow into a removed block.
///
/// When canonicalization deletes an empty block, any phi that named it as a
/// predecessor must instead name the still-live blocks that reach it. This
/// walks predecessor chains through other removed blocks (those present in
/// `redirect_map`) until it reaches blocks that survive compaction (those with
/// a `Some` entry in `block_remap`), returning their post-compaction indices.
/// The `BitSet` visited-guard makes the walk terminate on cyclic chains of
/// removed blocks.
///
/// # Arguments
///
/// * `removed_block` - Old index of the block being deleted whose live
///   predecessors are sought.
/// * `predecessors` - Map from each block to the old indices of the blocks with
///   an edge into it.
/// * `block_remap` - Per old block, its new index if kept or `None` if removed.
/// * `redirect_map` - Removed blocks that have a known ultimate jump target.
///
/// # Returns
///
/// The new (post-compaction) indices of the kept blocks that flow into
/// `removed_block`.
fn find_kept_predecessors(
    removed_block: usize,
    predecessors: &BTreeMap<usize, Vec<usize>>,
    block_remap: &[Option<usize>],
    redirect_map: &BTreeMap<usize, usize>,
) -> Vec<usize> {
    let mut result = Vec::new();
    let block_count = block_remap.len();
    let mut visited = BitSet::new(block_count);
    let mut queue = vec![removed_block];

    while let Some(current) = queue.pop() {
        if current >= block_count || !visited.insert(current) {
            continue;
        }

        if let Some(preds) = predecessors.get(&current) {
            for &pred in preds {
                if let Some(Some(new_idx)) = block_remap.get(pred) {
                    // This predecessor is kept - add its new index
                    result.push(*new_idx);
                } else if redirect_map.contains_key(&pred) {
                    // This predecessor is also removed - follow the chain
                    queue.push(pred);
                }
            }
        }
    }

    result
}

impl<T: Target> SsaFunction<T> {
    /// Canonicalizes the SSA function for clean code generation.
    ///
    /// This method performs final cleanup after deobfuscation passes:
    ///
    /// 1. **Strip Nop instructions**: Removes all `SsaOp::Nop` instructions
    /// 2. **Identify empty blocks**: Marks blocks with no instructions or phi nodes for removal
    /// 3. **Build redirect map**: For removed blocks, finds their ultimate jump targets
    /// 4. **Update branch targets**: Retargets jumps to skip removed empty blocks
    /// 5. **Update PHI predecessors**: Fixes PHI node operands when predecessor blocks are removed
    /// 6. **Compact blocks**: Removes empty blocks and renumbers remaining blocks contiguously
    ///
    /// This should be called after all deobfuscation passes complete, before
    /// code generation. The resulting SSA is cleaner and easier to convert to IL.
    pub fn canonicalize(&mut self) {
        // Phase 1: Strip Nop instructions from all blocks and remap instruction DefSites.
        self.strip_nops();

        // Collect blocks that must be preserved:
        // - Exception handler entry blocks
        // - Leave targets (exception handler exit blocks)
        let block_count = self.blocks.len();
        let mut protected_blocks = BitSet::new(block_count);

        // Protect the first block of every clause part. Read through the total
        // `parts()` view, so a clause the verifier would reject still keeps its
        // boundaries alive.
        for handler in &self.exception_handlers {
            for (_, range) in handler.parts() {
                if range.start() < block_count {
                    protected_blocks.insert(range.start());
                }
            }
        }

        // Protect Leave targets (exception handler exit blocks)
        for block in self.blocks.iter() {
            if let Some(SsaOp::Leave { target }) = block.control_terminator()
                && *target < block_count
            {
                protected_blocks.insert(*target);
            }
        }

        // Phase 2: Identify empty blocks and build remapping.
        // An empty block has no instructions AND no phi nodes.
        // Exception: Keep block 0 (entry) and protected exception handler blocks even if empty.
        let mut block_remap: Vec<Option<usize>> = Vec::with_capacity(block_count);
        let mut new_index = 0usize;

        for (old_index, block) in self.blocks.iter().enumerate() {
            let is_empty = block.instructions().is_empty() && block.phi_nodes().is_empty();
            let is_entry = old_index == 0;
            let is_protected = protected_blocks.contains(old_index);

            if is_empty && !is_entry && !is_protected {
                block_remap.push(None); // This block will be removed
            } else {
                block_remap.push(Some(new_index));
                new_index = new_index.saturating_add(1);
            }
        }

        // Phase 3: Build redirect map for removed blocks.
        // For each removed block, find its ultimate jump target (following jump chains).
        // If we can't find a redirect for a block, we must keep it instead of removing it.
        let mut redirect_map: BTreeMap<usize, usize> = BTreeMap::new();
        // What an edge redirected past each removed block runs through: the
        // host blocks the removed chain stood for.
        let mut passed: BTreeMap<usize, BlockProvenance> = BTreeMap::new();

        for old_index in 0..self.blocks.len() {
            if matches!(block_remap.get(old_index), Some(None)) {
                // This block is being removed - find where it would jump to
                if let Some((target, chain)) = self.find_ultimate_target(old_index, &block_remap) {
                    redirect_map.insert(old_index, target);
                    let mut through = BlockProvenance::default();
                    for removed in chain.iter().filter_map(|&index| self.blocks.get(index)) {
                        through.pass_through(removed.provenance());
                    }
                    if !through.is_empty() {
                        passed.insert(old_index, through);
                    }
                } else {
                    // Can't find a redirect target - we must keep this block.
                    // Reassign it a new index.
                    if let Some(slot) = block_remap.get_mut(old_index) {
                        *slot = Some(new_index);
                        new_index = new_index.saturating_add(1);
                    }
                }
            }
        }

        // Build predecessor map for PHI updates (needed for Phase 5).
        // For each block, collect all blocks that have edges TO it.
        let mut predecessors: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            block.for_each_successor(|target| {
                predecessors.entry(target).or_default().push(block_idx);
            });
        }

        // Phase 4: Update all branch targets in remaining blocks. An edge
        // redirected past a removed chain passes what the chain stood for, on
        // that edge, and every edge a block's record names is restated as its
        // branch is: the redirect a removed block's edge takes, or a kept
        // block's new index.
        let moved = |target: usize| {
            redirect_map
                .get(&target)
                .copied()
                .or_else(|| block_remap.get(target).copied().flatten())
        };
        for block in self.blocks.iter_mut() {
            let mut through: Vec<(usize, &BlockProvenance)> = Vec::new();
            block.for_each_successor(|target| {
                if let Some(chain) = passed.get(&target)
                    && let Some(onward) = moved(target)
                {
                    through.push((onward, chain));
                }
            });
            block.provenance_mut().retarget(moved);
            for (onward, chain) in through {
                block.provenance_mut().pass_through_on(onward, chain);
            }
            // A removed block whose control was forwarded resolves to where it
            // went; a kept one to its new index; any other target is left.
            for instr in block.instructions_mut() {
                instr.op_mut().remap_branch_targets(|target| {
                    redirect_map
                        .get(&target)
                        .copied()
                        .or_else(|| block_remap.get(target).copied().flatten())
                });
            }
        }

        // Phase 5: Update PHI node predecessors.
        // When a predecessor block is removed, we find the kept blocks that would have
        // flowed into the removed block and use those as the new predecessors.
        //
        // Special case: Some PHI operands may reference orphaned blocks (blocks with no
        // predecessors). This happens when deobfuscation passes modify the CFG without
        // properly updating PHI predecessors. We try to recover these by assigning
        // orphaned values to unaccounted-for predecessors.

        // Process each block's PHI nodes
        for block_idx in 0..self.blocks.len() {
            // Get the predecessors of THIS block (the one containing the PHI)
            // These are the OLD indices of blocks that jump to this block.
            let phi_block_preds: Vec<usize> =
                predecessors.get(&block_idx).cloned().unwrap_or_default();

            // Also compute the NEW indices of kept predecessors
            let kept_phi_block_preds: Vec<usize> = phi_block_preds
                .iter()
                .filter_map(|&old_pred| block_remap.get(old_pred).and_then(|opt| *opt))
                .collect();

            let Some(block) = self.blocks.get_mut(block_idx) else {
                continue;
            };
            for phi in block.phi_nodes_mut() {
                // Collect changes first (to avoid borrow issues)
                let mut changes: Vec<(usize, Option<PhiOperand>, Vec<PhiOperand>)> = Vec::new();
                // Track orphaned values (removed operands with no replacement)
                let mut orphaned_values: Vec<SsaVarId> = Vec::new();

                for (op_idx, operand) in phi.operands().iter().enumerate() {
                    let old_pred = operand.predecessor();
                    let value = operand.value();

                    // An operand for a throw names the block the throw leaves
                    // from, not an edge a terminator takes, and keeps being one:
                    // renumbered with its block, or dropped with it. A removed
                    // block is empty, and an empty block cannot throw.
                    if operand.is_exceptional() {
                        match block_remap.get(old_pred) {
                            Some(Some(new_pred)) => {
                                let mut moved = *operand;
                                moved.set_predecessor(*new_pred);
                                changes.push((op_idx, Some(moved), Vec::new()));
                            }
                            _ => changes.push((op_idx, None, Vec::new())),
                        }
                        continue;
                    }

                    if redirect_map.contains_key(&old_pred) {
                        // This predecessor was removed. Find all kept blocks that flow into it.
                        let kept_preds = find_kept_predecessors(
                            old_pred,
                            &predecessors,
                            &block_remap,
                            &redirect_map,
                        );

                        if kept_preds.is_empty() {
                            // Orphaned operand - track the value for potential recovery below
                            orphaned_values.push(value);
                        }

                        let replacements: Vec<PhiOperand> = kept_preds
                            .into_iter()
                            .map(|new_pred| PhiOperand::new(value, new_pred))
                            .collect();

                        // None = remove this operand, replacements = add these instead
                        changes.push((op_idx, None, replacements));
                    } else if let Some(Some(new_pred)) = block_remap.get(old_pred) {
                        // Predecessor was kept but renumbered - update in place
                        let mut moved = *operand;
                        moved.set_predecessor(*new_pred);
                        changes.push((op_idx, Some(moved), Vec::new()));
                    }
                }

                // Apply changes in reverse order (to preserve indices when removing)
                for (op_idx, replacement, additions) in changes.into_iter().rev() {
                    if let Some(new_op) = replacement {
                        // Update in place
                        if let Some(operand) = phi.operands_mut().get_mut(op_idx) {
                            *operand = new_op;
                        }
                    } else {
                        // Remove the operand
                        phi.operands_mut().remove(op_idx);
                        // Add replacement operands
                        for op in additions {
                            phi.add_operand(op);
                        }
                    }
                }

                // Post-processing: try to recover orphaned values by assigning them
                // to unaccounted-for predecessors.
                if !orphaned_values.is_empty() {
                    // Get the predecessors that are currently accounted for in the
                    // PHI, by the terminator edges its normal operands name.
                    let accounted_preds: BTreeSet<usize> =
                        phi.normal_operands().map(PhiOperand::predecessor).collect();

                    // Find predecessors that are NOT accounted for
                    let unaccounted_preds: Vec<usize> = kept_phi_block_preds
                        .iter()
                        .copied()
                        .filter(|pred| !accounted_preds.contains(pred))
                        .collect();

                    // Assign orphaned values to unaccounted predecessors
                    for (orphan_val, &unaccounted_pred) in
                        orphaned_values.iter().zip(unaccounted_preds.iter())
                    {
                        phi.add_operand(PhiOperand::new(*orphan_val, unaccounted_pred));
                    }
                }
            }
        }

        // Phase 6: Remove empty blocks and compact block indices.
        let mut kept_blocks: Vec<SsaBlock<T>> = Vec::with_capacity(new_index);
        for (old_index, block) in self.blocks.drain(..).enumerate() {
            if matches!(block_remap.get(old_index), Some(Some(_))) {
                kept_blocks.push(block);
            }
        }

        // Update block indices in kept blocks
        for (new_idx, block) in kept_blocks.iter_mut().enumerate() {
            block.set_id(new_idx);
        }

        *self.blocks = kept_blocks;
        self.remap_variable_def_sites_after_block_compaction(&block_remap);

        // Phase 7: Remap exception handler block indices.
        for handler in &mut self.exception_handlers {
            handler.remap_block_indices(&block_remap);
        }

        // Phase 8: Ensure the method has a valid terminator.
        self.ensure_valid_terminator();

        // Use sites name blocks by position, and every position after a removed
        // block moved.
        self.recompute_uses();
    }

    /// Gives the function an entry block no terminator transfers to.
    ///
    /// The entry block is where a call enters, and nothing else may: forward
    /// analyses seed their boundary state there, and a block with one incoming
    /// edge is taken to know that edge's condition. A lifter produces the
    /// violating shape when a loop starts at the function's first instruction
    /// (a self tail-call lowered to a jump, or a `while` compiled without a
    /// preamble). The verifier rejects it with
    /// [`VerifierError::EntryHasPredecessors`](crate::analysis::VerifierError::EntryHasPredecessors);
    /// this repairs it.
    ///
    /// Every block moves up by one index and a new, empty block 0 jumps to the
    /// old entry. Shifting rather than relocating keeps every exception clause's
    /// block range contiguous. Terminator targets, phi predecessors, def sites
    /// and clause ranges are renumbered; a variable defined at function entry
    /// (an argument or initialised local, [`DefSite::entry`]) stays on the new
    /// entry block.
    ///
    /// A phi in the old entry block has no operand for the new entry edge -
    /// there was no such edge to name - so each gains one: the value its
    /// variable holds on entry. That is the argument, initialised local or
    /// entry live-in the function already has for it, and otherwise a new entry
    /// live-in, the slot's value on entry, which nothing in the function
    /// computed.
    ///
    /// # Returns
    ///
    /// `true` if the entry block had predecessors and was split off, `false`
    /// if the function already satisfied the invariant.
    pub fn ensure_entry_has_no_predecessors(&mut self) -> bool {
        let mut entry_has_predecessor = false;
        for block in self.blocks.iter() {
            block.for_each_successor(|succ| entry_has_predecessor |= succ == 0);
        }
        if !entry_has_predecessor {
            return false;
        }

        let block_count = self.blocks.len();
        let block_remap: Vec<Option<usize>> = (1..=block_count).map(Some).collect();

        // `DefSite::entry()` and `DefSite::phi(0)` are the same value, so the
        // old entry's phi results are what tells them apart.
        let old_entry_phis: BTreeSet<SsaVarId> = self
            .blocks
            .first()
            .map(|block| block.phi_nodes().iter().map(|phi| phi.result()).collect())
            .unwrap_or_default();

        for block in self.blocks.iter_mut() {
            if let Some(op) = block
                .instructions_mut()
                .last_mut()
                .map(SsaInstruction::op_mut)
            {
                op.remap_branch_targets(|target| block_remap.get(target).copied().flatten());
            }
            for phi in block.phi_nodes_mut() {
                for operand in phi.operands_mut() {
                    let shifted = operand.predecessor().saturating_add(1);
                    operand.set_predecessor(shifted);
                }
            }
        }

        for var in &mut self.variables {
            let site = var.def_site();
            let is_function_entry = site == DefSite::entry() && !old_entry_phis.contains(&var.id());
            if !is_function_entry {
                var.set_def_site(site.with_block(site.block().saturating_add(1)));
            }
        }

        for handler in &mut self.exception_handlers {
            handler.remap_block_indices(&block_remap);
        }

        let mut entry = SsaBlock::new(0);
        let jump = self.instr_allocator.alloc();
        let mut instruction = SsaInstruction::synthetic(SsaOp::Jump { target: 1 });
        instruction.set_id(jump);
        entry.add_instruction(instruction);
        self.blocks.insert(0, entry);
        for (index, block) in self.blocks.iter_mut().enumerate() {
            block.set_id(index);
        }

        // Each old entry phi reads its variable's entry value on the new edge
        // from block 0, as the function's first pass through the old entry did.
        let entry_operands: Vec<(SsaVarId, SsaVarId)> = old_entry_phis
            .iter()
            .map(|phi| (*phi, self.entry_value_of(*phi, &old_entry_phis)))
            .collect();
        if let Some(old_entry) = self.blocks.get_mut(1) {
            for phi in old_entry.phi_nodes_mut() {
                if let Some((_, value)) = entry_operands
                    .iter()
                    .find(|(result, _)| *result == phi.result())
                {
                    phi.add_operand(PhiOperand::new(*value, 0));
                }
            }
        }
        self.recompute_uses();
        true
    }

    /// The value the variable the phi `phi` merges holds when the function is
    /// entered.
    ///
    /// The variable's version 0 defined at entry - an argument, an initialised
    /// local or an entry live-in of the phi's rename group, or, in no group,
    /// of the phi's origin - and otherwise a new entry live-in derived from
    /// the phi, in its group: the slot's value on entry, which nothing in the
    /// function computed.
    ///
    /// # Arguments
    ///
    /// * `phi` - The result of a phi of the old entry block.
    /// * `entry_phis` - Every such result; none of them is an entry value,
    ///   though each shares the entry's def site.
    fn entry_value_of(&mut self, phi: SsaVarId, entry_phis: &BTreeSet<SsaVarId>) -> SsaVarId {
        let group = self.rename_group(phi);
        let origin = self.variable(phi).map(|var| var.origin());
        let held = self.variables.iter().find(|var| {
            let same = if group == u32::MAX {
                Some(var.origin()) == origin && self.rename_group(var.id()) == u32::MAX
            } else {
                self.rename_group(var.id()) == group
            };
            same && var.version() == 0
                && var.def_site() == DefSite::entry()
                && !entry_phis.contains(&var.id())
                && matches!(
                    var.origin(),
                    VariableOrigin::Argument(_)
                        | VariableOrigin::Local(_)
                        | VariableOrigin::EntryLiveIn
                )
        });
        if let Some(held) = held {
            return held.id();
        }
        let var_type = self
            .variable(phi)
            .map_or_else(T::unknown_type, |var| var.var_type().clone());
        let value = self.create_variable_derived(
            phi,
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            var_type,
            Derivation::EntryValue,
        );
        if group != u32::MAX {
            self.set_rename_group(value, group);
        }
        value
    }

    /// Rewrites every variable's def site to its post-compaction location.
    ///
    /// Block removal and renumbering invalidate the block and instruction
    /// indices stored on each variable's [`DefSite`]. A variable whose defining
    /// block or instruction no longer exists is rebased onto an entry def site,
    /// but only when nothing still reads it: stamping `DefSite::entry()` on a
    /// still-read variable would make a dangling read indistinguishable from a
    /// function argument, the exact disguise the verifier must be able to see
    /// through. Phi and instruction sites that survive are translated to their
    /// new block index in place.
    ///
    /// # Arguments
    ///
    /// * `block_remap` - Per old block, its new index if kept or `None` if
    ///   removed.
    fn remap_variable_def_sites_after_block_compaction(&mut self, block_remap: &[Option<usize>]) {
        let block_instr_counts: Vec<usize> = self
            .blocks
            .iter()
            .map(|block| block.instructions().len())
            .collect();

        // A variable whose defining block or instruction no longer exists falls
        // back to an entry def site - but only when nothing reads it. Stamping
        // `DefSite::entry()` on a still-read variable makes a dangling read
        // indistinguishable from an argument, which is precisely the disguise
        // the verifier must be able to see through. See
        // `SsaFunction::collect_read_variables`.
        let still_read = self.collect_read_variables();
        for var in &mut self.variables {
            let site = var.def_site();
            let readable = still_read.contains(var.id());
            let Some(Some(new_block)) = block_remap.get(site.block()) else {
                if !readable {
                    var.set_def_site(DefSite::entry());
                }
                continue;
            };

            if let Some(instr_idx) = site.index() {
                if block_instr_counts
                    .get(*new_block)
                    .is_some_and(|count| instr_idx < *count)
                {
                    var.set_def_site(DefSite::instruction(*new_block, instr_idx));
                } else if !readable {
                    var.set_def_site(DefSite::entry());
                }
            } else {
                var.set_def_site(DefSite::phi(*new_block));
            }
        }
    }

    /// Ensures the function has a valid terminator path from the entry block.
    ///
    /// This handles the case where all meaningful code has been removed (e.g., after
    /// neutralizing 100% protection code in a module .cctor), leaving only Jumps to
    /// empty blocks. In such cases, we replace the entry block's terminator with a
    /// Return instruction to produce valid IL.
    fn ensure_valid_terminator(&mut self) {
        // Check if the method effectively does nothing useful:
        // - Only has Jump instructions (no actual code)
        // - All Jump targets lead to empty blocks or more Jumps
        let has_useful_code = self.blocks.iter().any(|block| {
            block.instructions().iter().any(|instr| {
                match instr.op() {
                    // Jumps and Nops don't count as useful - they're just control flow
                    SsaOp::Jump { .. } | SsaOp::Nop => false,
                    // Any other instruction (including returns, throws) is useful code
                    _ => true,
                }
            })
        });

        // If there's no useful code, replace entry block with just a Return
        if !has_useful_code {
            let id = self.instr_allocator.alloc();
            if let Some(entry_block) = self.blocks.first_mut() {
                entry_block.instructions_mut().clear();
                entry_block.phi_nodes_mut().clear();
                let mut instruction = SsaInstruction::synthetic(SsaOp::returns(None));
                instruction.set_id(id);
                entry_block.add_instruction(instruction);
            }
        }
    }

    /// Finds where a block transfers control once empty blocks collapse.
    ///
    /// Follows the block's jump chain (and, for a truly empty terminator-less
    /// block, CIL fall-through to the next block) across other removed blocks
    /// until it lands on a block that survives compaction, returning that
    /// survivor's new index. Branches, switches, and other multi-target
    /// terminators stop the walk: such a block is not genuinely empty and cannot
    /// be redirected. The `BitSet` visited-guard yields `None` on a cycle rather
    /// than looping forever.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Old index of the removed block whose redirect target is
    ///   sought.
    /// * `block_remap` - Per old block, its new index if kept or `None` if
    ///   removed.
    ///
    /// # Returns
    ///
    /// The new index of the surviving block control reaches, with the removed
    /// blocks walked to get there, or `None` when no single redirect target
    /// exists (a branch, a dead end, or a cycle).
    fn find_ultimate_target(
        &self,
        block_idx: usize,
        block_remap: &[Option<usize>],
    ) -> Option<(usize, Vec<usize>)> {
        let mut visited = BitSet::new(self.blocks.len());
        let mut current = block_idx;
        let mut chain = Vec::new();

        while visited.insert(current) {
            let block = self.blocks.get(current)?;
            chain.push(current);

            // Get the terminator's target
            let terminator = block.control_terminator();
            let target = terminator.and_then(|op| match op {
                SsaOp::Jump { target } => Some(*target),
                // For branches, we can't simplify - the block isn't truly empty
                _ => None,
            });

            // Handle the target
            match target {
                Some(t) if block_remap.get(t).copied().flatten().is_some() => {
                    // Target exists in new layout
                    return block_remap
                        .get(t)
                        .copied()
                        .flatten()
                        .map(|target| (target, chain));
                }
                Some(t) => {
                    // Target is also being removed, follow the chain
                    current = t;
                }
                None => {
                    // No explicit jump target. Check if block is truly empty (no terminator).
                    // In CIL semantics, empty blocks fall through to the next block.
                    if terminator.is_none() && block.instructions().is_empty() {
                        // Try to fall through to the next block
                        let next_block = current.saturating_add(1);
                        if next_block < self.blocks.len() {
                            if let Some(Some(new_idx)) = block_remap.get(next_block) {
                                // Next block exists in new layout
                                return Some((*new_idx, chain));
                            } else if block_remap.get(next_block).is_some() {
                                // Next block is also being removed, follow the chain
                                current = next_block;
                                continue;
                            }
                        }
                    }
                    // No simple jump target and no fall-through, can't redirect
                    return None;
                }
            }
        }

        None // Cycle detected
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ir::{block::SsaBlock, function::SsaFunction, instruction::SsaInstruction, ops::SsaOp},
        testing::MockTarget,
    };

    /// An edge redirected past an empty block that still stands for a host
    /// block - a host whose blocks fall through, emptied - passes through it.
    #[test]
    fn an_edge_redirected_past_an_empty_block_passes_through_it() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        entry.provenance_mut().carry(0x10);
        ssa.add_block(entry);
        let mut empty = SsaBlock::new(1);
        empty.provenance_mut().carry(0x20);
        ssa.add_block(empty);
        let mut exit = SsaBlock::new(2);
        exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
        ssa.add_block(exit);
        ssa.recompute_uses();

        ssa.canonicalize();

        assert_eq!(ssa.block_count(), 2, "the empty block is gone");
        let entry = ssa.block(0).unwrap_or_else(|| panic!("the entry stays"));
        assert!(matches!(
            entry.control_terminator(),
            Some(SsaOp::Jump { target: 1 })
        ));
        assert_eq!(entry.provenance().carried(), &[0x10]);
        assert_eq!(entry.provenance().passed(), &[0x20]);
        assert_eq!(
            entry.provenance().passed_edges().collect::<Vec<_>>(),
            [Some(1)],
            "on the edge redirected past the removed block, renumbered"
        );
    }
}
