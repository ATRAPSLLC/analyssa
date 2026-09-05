//! Mutation and transform methods for SSA functions.
//!
//! Provides all operations that modify an [`SsaFunction`] — replacing uses,
//! eliminating phi nodes, folding constants, compacting variables, and
//! optimizing local variable layout.
//!
//! # Variable Replacement Architecture
//!
//! Two primitives with different safety profiles:
//!
//! | Primitive | Scope | Safety |
//! |-----------|-------|--------|
//! | [`replace_uses(old, new)`] | Instructions only | Safe for passes |
//! | [`replace_uses_including_phis(old, new)`] | Instructions + phi operands | Internal only |
//!
//! **`replace_uses`** (instruction uses only) is the safe default for compiler passes.
//! It avoids creating cross-origin phi operand references, which can break `rebuild_ssa`'s
//! assumption that each variable flows to at most one phi origin.
//!
//! **`replace_uses_including_phis`** (`pub(crate)`) also replaces phi operands.
//! Needed for infrastructure operations like trivial phi elimination where the
//! eliminated phi and its forwarding target share the same origin context.
//!
//! ## Self-Referential Guard
//!
//! Both methods skip replacements where the instruction's destination equals `new_var`,
//! preventing self-referential instructions like `v0 = add(v0, v1)`. The
//! [`ReplaceResult`] reports both successful replacements and skips.
//!
//! ## High-Level Operations
//!
//! | Operation | Description |
//! |-----------|-------------|
//! | `propagate_copies` | Batch copy propagation with completion tracking |
//! | `eliminate_trivial_phis` | Iterative trivial phi removal to fixpoint |
//! | `prune_phi_operands` | Remove stale operands after CFG changes |
//! | `fold_constant` | Replace an instruction with its constant result |
//! | `compact_variables` | Remove orphaned variables and reindex |
//! | `strip_nops` | Remove Nop instructions and fix DefSites |
//! | `recompute_uses` | Rebuild use-site tracking from scratch |

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    BitSet,
    analysis::{cfg::SsaCfg, exceptions::EhCfg},
    graph::{NodeId, algorithms::compute_dominators},
    ir::{
        block::ReplaceResult,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::PhiOperand,
        value::ConstValue,
        variable::{DefSite, SsaVarId, UseSite, VariableOrigin},
    },
    target::Target,
};

/// Options for trivial phi elimination.
pub struct TrivialPhiOptions<'a> {
    /// If set, only consider phis in reachable blocks and use reachability-aware
    /// self-referential checks. Unreachable predecessor operands are filtered out
    /// as a second-pass check. All trivial phis are removed unconditionally.
    ///
    /// If `None`, all blocks are considered. Chain resolution is applied, and only
    /// fully propagated phis (no skipped uses from the self-ref guard) are removed.
    pub reachable: Option<&'a BitSet>,
}

/// Result of batch copy propagation.
pub struct CopyPropagationResult {
    /// Total number of uses replaced across all copies.
    pub total_replaced: usize,
    /// Set of copy destinations that were fully propagated (all uses replaced).
    /// These copies can safely be Nop'd by the caller.
    /// Stored as a BitSet indexed by `SsaVarId::index()`.
    pub fully_propagated: BitSet,
    /// Set of copy destinations that still have remaining instruction uses
    /// (due to self-referential guard). These copies must be kept alive.
    /// Stored as a BitSet indexed by `SsaVarId::index()`.
    pub partially_propagated: BitSet,
}

fn has_remaining_uses_including_phis<T: Target>(
    ssa: &SsaFunction<T>,
    var: SsaVarId,
    skip_phi: Option<(usize, usize)>,
) -> bool {
    ssa.blocks().iter().any(|block| {
        block.instructions().iter().any(|instr| {
            let mut found = false;
            instr.op().for_each_use(|used| found |= used == var);
            found
        }) || block
            .phi_nodes()
            .iter()
            .enumerate()
            .filter(|(phi_idx, _)| skip_phi != Some((block.id(), *phi_idx)))
            .any(|(_, phi)| phi.operands().iter().any(|operand| operand.value() == var))
    })
}

impl<T: Target> SsaFunction<T> {
    /// Removes every phi from a block the runtime dispatches to, declaring its
    /// result runtime-supplied instead.
    ///
    /// A phi operand names the terminator edge it arrived along. A handler or
    /// filter entry that no terminator transfers to has no such edges, so a phi
    /// there is unrepresentable — its operands would name edges that do not
    /// exist, and the four consumers that walk operands by predecessor would
    /// read them as if they did.
    ///
    /// An entry that *is* also a branch or loop target keeps its phis: the rule
    /// is about having no terminator predecessors, not about being a handler.
    ///
    /// When a removed phi's result is still used, an [`SsaOp::Pop`] is
    /// prepended in its place. That is the shape the verifier already reads as
    /// "the runtime supplies this value", so the value stays defined without
    /// asserting a merge over edges that cannot exist.
    ///
    /// # Returns
    ///
    /// The number of phis removed.
    pub fn demote_runtime_entry_phis(&mut self) -> usize {
        let exception_entries: Vec<usize> = {
            let blocks = self.exception_blocks();
            if blocks.is_empty() {
                return 0;
            }
            blocks.runtime_entries().to_vec()
        };

        let unreachable_entries: Vec<usize> = {
            let cfg = SsaCfg::from_ssa(self);
            exception_entries
                .into_iter()
                .filter(|entry| cfg.block_predecessors(*entry).is_empty())
                .collect()
        };

        let mut demoted: usize = 0;
        for entry in unreachable_entries {
            let results: Vec<SsaVarId> = self
                .block(entry)
                .map(|block| block.phi_nodes().iter().map(|phi| phi.result()).collect())
                .unwrap_or_default();
            if results.is_empty() {
                continue;
            }

            if let Some(block) = self.block_mut(entry) {
                block.phi_nodes_mut().clear();
            }
            demoted = demoted.saturating_add(results.len());

            // One use census per entry: `count_uses` is a whole-function walk.
            let use_counts = self.count_uses();
            for (offset, result) in results.into_iter().enumerate() {
                if use_counts.get(&result).copied().unwrap_or(0) == 0 {
                    continue;
                }
                if let Some(block) = self.block_mut(entry) {
                    let at = offset.min(block.instructions().len());
                    block
                        .instructions_mut()
                        .insert(at, SsaInstruction::synthetic(SsaOp::Pop { value: result }));
                }
                if let Some(variable) = self.variable_mut(result) {
                    variable.set_def_site(DefSite::phi(entry));
                }
            }
        }

        demoted
    }

    /// Replaces all uses of `old_var` with `new_var` throughout the function.
    ///
    /// This is the core operation for copy propagation - when we know that
    /// `v1 = v0` (a copy), we can replace all uses of `v1` with `v0`.
    ///
    /// # Note
    ///
    /// This method only replaces uses in instructions, not in PHI operands.
    /// For internal operations that need to also replace PHI operands, use
    /// `replace_uses_including_phis`.
    pub fn replace_uses(&mut self, old_var: SsaVarId, new_var: SsaVarId) -> ReplaceResult {
        self.blocks
            .iter_mut()
            .map(|block| block.replace_uses(old_var, new_var))
            .fold(ReplaceResult::default(), |acc, r| ReplaceResult {
                replaced: acc.replaced.saturating_add(r.replaced),
                skipped: acc.skipped.saturating_add(r.skipped),
            })
    }

    /// Replaces all uses of `old_var` with `new_var`, including in PHI operands.
    ///
    /// Unlike [`replace_uses`](Self::replace_uses), this method also replaces uses
    /// in PHI node operands across all blocks. This is necessary for internal SSA
    /// operations that eliminate PHI nodes and need to forward their values through
    /// other PHIs.
    ///
    /// # Safety
    ///
    /// This method is `pub(crate)` because it can create cross-origin PHI operand
    /// references if misused.
    ///
    /// # When to Use
    ///
    /// Only use this method for:
    /// - **Trivial PHI elimination**: When removing a PHI like `v10 = phi(v5, v5)`,
    ///   we need to replace uses of `v10` with `v5` everywhere, including in other
    ///   PHI operands.
    /// - **Copy propagation within PHIs**: When a copy's destination is a PHI result
    ///   and we're eliminating that PHI.
    pub fn replace_uses_including_phis(
        &mut self,
        old_var: SsaVarId,
        new_var: SsaVarId,
    ) -> ReplaceResult {
        self.blocks
            .iter_mut()
            .map(|block| block.replace_uses_including_phis(old_var, new_var))
            .fold(ReplaceResult::default(), |acc, r| ReplaceResult {
                replaced: acc.replaced.saturating_add(r.replaced),
                skipped: acc.skipped.saturating_add(r.skipped),
            })
    }

    /// Replaces all uses of `old_var` with `new_var` within a specific block.
    ///
    /// This is a targeted version of `replace_uses` that only affects instructions
    /// within the specified block (not PHI operands).
    pub fn replace_uses_in_block(
        &mut self,
        block_idx: usize,
        old_var: SsaVarId,
        new_var: SsaVarId,
    ) -> ReplaceResult {
        self.block_mut(block_idx)
            .map_or(ReplaceResult::default(), |block| {
                block.replace_uses(old_var, new_var)
            })
    }

    /// Propagates a batch of copy mappings (dest → src) through all instructions.
    ///
    /// For each mapping, replaces all uses of `dest` with `src` in instructions
    /// (NOT in phi operands — this is the safe default that avoids cross-origin
    /// phi references). Reports which copies were fully propagated vs. which
    /// still have remaining uses due to the self-referential guard.
    ///
    /// # Usage
    ///
    /// This is a crate-internal method used by the copy-propagation pass. It is
    /// reachable from outside the crate through the checked edit session on
    /// [`SsaEditor::propagate_copies`](crate::ir::function::SsaEditor::propagate_copies):
    ///
    /// ```rust
    /// use std::collections::BTreeMap;
    /// use analyssa::{
    ///     ir::{function::SsaEditOptions, SsaVarId},
    ///     testing,
    /// };
    ///
    /// // The fixture ends block 0 with `v9 = copy v8`, and block 1 returns v9.
    /// let mut ssa = testing::scalar_rewrite_fixture();
    /// let dest = SsaVarId::from_index(9);
    /// let src = SsaVarId::from_index(8);
    ///
    /// let copies = BTreeMap::from([(dest, src)]);
    /// ssa.edit(SsaEditOptions::new(), |editor| {
    ///     let result = editor.propagate_copies(&copies);
    ///
    ///     // The single use of `dest` (the return in block 1) now reads `src`.
    ///     assert_eq!(result.total_replaced, 1);
    ///     assert!(result.fully_propagated.contains(dest.index()));
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    pub(in crate::ir::function) fn propagate_copies(
        &mut self,
        copies: &BTreeMap<SsaVarId, SsaVarId>,
    ) -> CopyPropagationResult {
        let variable_count = self.var_id_capacity();
        let mut total_replaced: usize = 0;
        let mut fully_propagated = BitSet::new(variable_count);
        let mut partially_propagated = BitSet::new(variable_count);

        // Build the dominator tree once: rewriting instruction uses never
        // changes any terminator, so it stays valid across every replacement
        // below; rebuilding it per copy would repeat the whole CFG walk.
        let dominators = if self.block_count() > 0 {
            let eh = EhCfg::from_ssa(self);
            Some(compute_dominators(&eh, NodeId::new(0)))
        } else {
            None
        };

        for (dest, src) in copies {
            if dest == src {
                continue;
            }

            let result = self
                .replace_uses_checked_with(*dest, *src, dominators.as_ref())
                .as_replace_result();

            if result.replaced > 0 {
                if result.is_complete() {
                    fully_propagated.insert(dest.index());
                } else {
                    partially_propagated.insert(dest.index());
                }
                total_replaced = total_replaced.saturating_add(result.replaced);
            }
        }

        CopyPropagationResult {
            total_replaced,
            fully_propagated,
            partially_propagated,
        }
    }

    /// Neutralizes Copy instructions that define the given variable by
    /// replacing them with Nop.
    ///
    /// This is used after copy propagation to eliminate dead copy instructions
    /// whose destination has been fully propagated to all use sites. Without
    /// this, rebuild_ssa's rename would re-create versions for the Copy's origin,
    /// shadowing the source variable and undoing the propagation.
    pub(in crate::ir::function) fn nop_copy_defining(&mut self, dest: SsaVarId) -> bool {
        for block in &mut self.blocks {
            for instr in block.instructions_mut() {
                if let SsaOp::Copy { dest: d, .. } = instr.op()
                    && *d == dest
                {
                    instr.set_op(SsaOp::Nop);
                    return true;
                }
            }
        }
        false
    }

    /// Prunes phi operands from non-existent or unreachable predecessors.
    ///
    /// After block removal or CFG changes, phi nodes may reference predecessors
    /// that no longer exist or are unreachable. This method removes those stale
    /// operands, ensuring phi nodes reference exactly the block's real
    /// predecessors.
    ///
    /// Pruning is deliberately **structural only**: an operand is dropped solely
    /// because its predecessor edge is gone, never because its *value* looks
    /// undefined. A phi's operand list is a per-incoming-edge mapping, so
    /// dropping the operand for a live edge breaks the phi/CFG invariant that
    /// [`MissingPhiOperand`](crate::analysis::VerifierError::MissingPhiOperand)
    /// guards — and, worse, can leave a
    /// two-operand phi with a single operand, which trivial-phi simplification
    /// then collapses into a plain copy of the surviving value. That silently
    /// discards one of the merged values. An operand naming a variable with no
    /// reachable definition is an upstream defect to fix at its source; keeping
    /// it is strictly safer than deleting a live edge to hide it.
    ///
    /// Returns the number of operands pruned.
    pub fn prune_phi_operands(&mut self, reachable: &BitSet) -> usize {
        // Compute actual predecessors from the CFG
        let block_count = self.blocks.len();
        let mut actual_predecessors: BTreeMap<usize, BitSet> = BTreeMap::new();
        for block_idx in reachable.iter() {
            if let Some(block) = self.block(block_idx) {
                block.for_each_successor(|successor| {
                    actual_predecessors
                        .entry(successor)
                        .or_insert_with(|| BitSet::new(block_count))
                        .insert(block_idx);
                });
            }
        }

        let mut pruned: usize = 0;

        for block_idx in reachable.iter() {
            if let Some(block) = self.block_mut(block_idx) {
                let preds = actual_predecessors.get(&block_idx);

                for phi in block.phi_nodes_mut() {
                    let operands = phi.operands_mut();
                    let original_len = operands.len();

                    if original_len == 0 {
                        continue;
                    }

                    // Predicate for operands worth keeping; evaluated without
                    // materializing a per-phi `Vec<bool>`.
                    let keeps = |op: &PhiOperand| -> bool {
                        let pred = op.predecessor();
                        pred < block_count && preds.is_some_and(|p| p.contains(pred))
                    };

                    // Never leave a PHI completely empty.
                    let keep_count = operands.iter().filter(|op| keeps(op)).count();
                    if keep_count == 0 {
                        continue;
                    }

                    operands.retain(keeps);

                    pruned = pruned.saturating_add(original_len.saturating_sub(operands.len()));
                }
            }
        }

        pruned
    }

    /// Recomputes all use information from scratch.
    ///
    /// This should be called after SSA transformations that may have invalidated
    /// the use tracking.
    pub fn recompute_uses(&mut self) {
        let variables = &mut self.variables;

        // Step 1: Clear all existing uses
        for var in variables.iter_mut() {
            var.clear_uses();
        }

        // Step 2: Scan instructions to record uses
        for (block_idx, block) in self.blocks.iter().enumerate() {
            // Record uses from instructions
            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                instr.op().for_each_use(|use_var| {
                    let var = use_var.index();
                    if let Some(slot) = variables.get_mut(var) {
                        let use_site = UseSite::instruction(block_idx, instr_idx);
                        slot.add_use(use_site);
                    }
                });
            }

            // Record uses from phi nodes
            for (phi_idx, phi) in block.phi_nodes().iter().enumerate() {
                for operand in phi.operands() {
                    let var = operand.value().index();
                    if let Some(slot) = variables.get_mut(var) {
                        let use_site = UseSite::phi_operand(block_idx, phi_idx);
                        slot.add_use(use_site);
                    }
                }
            }
        }
    }

    /// Replaces the operation of an instruction at a specific location.
    pub fn replace_instruction_op(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        new_op: SsaOp<T>,
    ) -> bool {
        if let Some(block) = self.blocks.get_mut(block_idx)
            && let Some(instr) = block.instructions_mut().get_mut(instr_idx)
        {
            instr.set_op_preserving_type(new_op);
            return true;
        }
        false
    }

    /// Simplifies a phi node by converting it to a copy operation.
    ///
    /// When a phi node has all identical operands (excluding self-references),
    /// instruction uses of the phi result can be replaced with `source`. The
    /// phi is removed only when no remaining use of its result exists outside
    /// the phi being removed.
    pub fn simplify_phi_to_copy(
        &mut self,
        block_idx: usize,
        phi_idx: usize,
        source: SsaVarId,
    ) -> bool {
        let Some(block) = self.blocks.get(block_idx) else {
            return false;
        };

        let Some(phi) = block.phi_nodes().get(phi_idx) else {
            return false;
        };

        let dest = phi.result();

        if dest != source {
            let _ = self.replace_uses_checked(dest, source);
        }

        if has_remaining_uses_including_phis(self, dest, Some((block_idx, phi_idx))) {
            return false;
        }

        let Some(block) = self.blocks.get_mut(block_idx) else {
            return false;
        };
        if phi_idx >= block.phi_nodes().len() {
            return false;
        }
        block.phi_nodes_mut().remove(phi_idx);

        true
    }

    /// Removes a phi node by index without any validation.
    pub fn remove_phi_unchecked(&mut self, block_idx: usize, phi_idx: usize) -> bool {
        if let Some(block) = self.blocks.get_mut(block_idx)
            && phi_idx < block.phi_nodes().len()
        {
            block.phi_nodes_mut().remove(phi_idx);
            return true;
        }
        false
    }

    /// Eliminates trivial phi nodes where all non-self operands resolve to a
    /// single value. Iterates to fixpoint (cascading simplification).
    ///
    /// A phi is trivial when, excluding self-references, all operands provide
    /// the same value. The phi result is replaced by that value everywhere
    /// (including other phi operands).
    ///
    /// # Modes
    ///
    /// When `options.reachable` is `Some`:
    /// - Uses reachability-aware self-referential checks (definitions in unreachable
    ///   blocks don't count as creating cycles).
    /// - Performs a second-pass check filtering operands from unreachable predecessors.
    /// - All trivial phis are removed unconditionally (suitable for rebuild_ssa).
    ///
    /// When `options.reachable` is `None`:
    /// - Uses basic self-referential checks.
    /// - Resolves chains among trivial phis to avoid stale references.
    /// - Only fully propagated phis (no skipped uses) are removed (suitable for repair_ssa).
    ///
    /// # Returns
    ///
    /// The number of phis eliminated.
    pub fn eliminate_trivial_phis(&mut self, options: &TrivialPhiOptions) -> usize {
        let mut total_eliminated: usize = 0;
        let block_count = self.blocks.len();

        // Precompute reachability data if in reachable mode. Build the full
        // predecessor relation in one O(V+E) pass instead of calling
        // `block_predecessors` per block (which is O(V) each → O(V²)).
        let reachable_preds: Option<BTreeMap<usize, BitSet>> = options.reachable.map(|reachable| {
            let all_preds = SsaCfg::from_ssa(self).to_predecessor_sets();
            let mut map = BTreeMap::new();
            for block in &self.blocks {
                let block_idx = block.id();
                if !reachable.contains(block_idx) {
                    continue;
                }
                let mut preds = BitSet::new(block_count);
                if let Some(plist) = all_preds.get(block_idx) {
                    for &p in plist {
                        if reachable.contains(p) {
                            preds.insert(p);
                        }
                    }
                }
                map.insert(block_idx, preds);
            }
            map
        });

        let var_def_block: Option<BTreeMap<SsaVarId, usize>> = options.reachable.map(|_| {
            let mut map = BTreeMap::new();
            for block in &self.blocks {
                let block_idx = block.id();
                for instr in block.instructions() {
                    for dest in instr.op().defs() {
                        map.insert(dest, block_idx);
                    }
                }
            }
            map
        });

        loop {
            let mut trivial_phis: Vec<(SsaVarId, SsaVarId)> = Vec::new();

            for block in &self.blocks {
                let block_idx = block.id();
                let block_reachable_preds =
                    reachable_preds.as_ref().and_then(|rp| rp.get(&block_idx));

                for phi in block.phi_nodes() {
                    let result = phi.result();

                    // Collect unique non-self operands
                    let unique_sources: BTreeSet<SsaVarId> = phi
                        .operands()
                        .iter()
                        .map(|op| op.value())
                        .filter(|&v| v != result)
                        .collect();

                    if let Some(&source) = unique_sources
                        .iter()
                        .next()
                        .filter(|_| unique_sources.len() == 1)
                    {
                        let is_self_ref = match (&var_def_block, options.reachable) {
                            (Some(vdb), Some(reachable)) => self
                                .would_create_self_reference_reachable(
                                    source, result, vdb, reachable,
                                ),
                            _ => self.would_create_self_reference(source, result),
                        };

                        if !is_self_ref {
                            trivial_phis.push((result, source));
                            continue;
                        }
                    } else if unique_sources.is_empty() && !phi.operands().is_empty() {
                        // Fully self-referential phi
                        trivial_phis.push((result, result));
                        continue;
                    }

                    // Reachable-only second pass: filter out operands from
                    // unreachable predecessors and check triviality again
                    if unique_sources.len() > 1
                        && let Some(rpreds) = block_reachable_preds
                    {
                        let unique_reachable: BTreeSet<SsaVarId> = phi
                            .operands()
                            .iter()
                            .filter(|op| {
                                let pred = op.predecessor();
                                pred < block_count && rpreds.contains(pred)
                            })
                            .map(|op| op.value())
                            .filter(|&v| v != result)
                            .collect();

                        if let Some(&source) = unique_reachable
                            .iter()
                            .next()
                            .filter(|_| unique_reachable.len() == 1)
                        {
                            let is_self_ref = match (&var_def_block, options.reachable) {
                                (Some(vdb), Some(reachable)) => self
                                    .would_create_self_reference_reachable(
                                        source, result, vdb, reachable,
                                    ),
                                _ => self.would_create_self_reference(source, result),
                            };
                            if !is_self_ref {
                                trivial_phis.push((result, source));
                            }
                        } else if unique_reachable.is_empty()
                            && phi.operands().iter().any(|op| {
                                let pred = op.predecessor();
                                pred < block_count && rpreds.contains(pred)
                            })
                        {
                            trivial_phis.push((result, result));
                        }
                    }
                }
            }

            if trivial_phis.is_empty() {
                break;
            }

            let variable_count = self.var_id_capacity();

            if options.reachable.is_none() {
                // Repair mode: resolve chains among trivial phis.
                let trivial_map: BTreeMap<SsaVarId, SsaVarId> =
                    trivial_phis.iter().copied().collect();
                for entry in &mut trivial_phis {
                    if entry.0 == entry.1 {
                        continue;
                    }
                    let mut current = entry.1;
                    let mut visited = BTreeSet::new();
                    while let Some(&next) = trivial_map.get(&current) {
                        if next == current || !visited.insert(current) {
                            break;
                        }
                        current = next;
                    }
                    entry.1 = current;
                }

                // Replace instruction uses through the checked path and only
                // remove phis whose result is completely unused afterward. The
                // dominator tree is built once for the whole batch — use
                // replacement leaves the CFG (and therefore dominance) unchanged.
                let dominators = if self.block_count() > 0 {
                    let eh = EhCfg::from_ssa(self);
                    Some(compute_dominators(&eh, NodeId::new(0)))
                } else {
                    None
                };
                let mut trivial_set = BitSet::new(variable_count);
                // All replacements first, then *one* scan for what is still
                // read. Interleaving them meant a whole-function scan per
                // trivial phi — O(trivial_phis × function) per fixpoint round,
                // and a rebuild produces trivial phis in proportion to the
                // function, so the round was quadratic.
                //
                // Batching is equivalent: `replace_uses_checked_with(p, s)` only
                // removes uses of `p` and adds uses of `s`, and `s` is never
                // another phi result being tested in this round (chains were
                // already resolved above), so no replacement can resurrect a use
                // of a phi an earlier one retired.
                for (phi_result, source) in &trivial_phis {
                    if *phi_result != *source {
                        let _ = self.replace_uses_checked_with(
                            *phi_result,
                            *source,
                            dominators.as_ref(),
                        );
                    }
                }
                // Collapse phi-operand references too.
                //
                // `replace_uses_checked_with` walks only `block.instructions()`,
                // so a trivial phi whose result is another phi's operand stays
                // "read" and survives the round. In a chain
                // `p1 = phi(x); p2 = phi(p1); ...` that retires exactly one phi
                // per round — the fixpoint runs as many rounds as there are
                // phis, and since each round is already O(phis x function),
                // the whole thing would be cubic — seconds on a chain of a few
                // hundred phis.
                //
                // Rewriting the operand is the sanctioned case for touching phi
                // operands directly (see the module header on
                // `replace_uses_including_phis`): a trivial phi's value *is* its
                // source, so an operand naming it and an operand naming the
                // source denote the same value on the same edge. Chains were
                // resolved into `trivial_phis` above, so `source` is already the
                // end of the chain.
                let resolved: BTreeMap<SsaVarId, SsaVarId> = trivial_phis
                    .iter()
                    .filter(|(result, source)| result != source)
                    .copied()
                    .collect();
                if !resolved.is_empty() {
                    for block in &mut self.blocks {
                        for phi in block.phi_nodes_mut() {
                            let result = phi.result();
                            for operand in phi.operands_mut() {
                                if let Some(&target) = resolved.get(&operand.value())
                                    // Never rewrite an operand into a reference
                                    // to its own phi — that would manufacture the
                                    // self-referential shape this pass removes.
                                    && target != result
                                {
                                    *operand = PhiOperand::new(target, operand.predecessor());
                                }
                            }
                        }
                    }
                }

                let still_read = self.collect_read_variables();
                for (phi_result, _) in &trivial_phis {
                    if !still_read.contains_checked(phi_result.index()) {
                        trivial_set.insert(phi_result.index());
                    }
                }
                if trivial_set.is_empty() {
                    break;
                }

                total_eliminated = total_eliminated.saturating_add(trivial_set.count());
                for block in &mut self.blocks {
                    block.phi_nodes_mut().retain(|phi| {
                        let idx = phi.result().index();
                        idx >= variable_count || !trivial_set.contains(idx)
                    });
                }
                // The variable rows are deliberately left in place. `variables`
                // is indexed by id (`variables[i].id().index() == i`), and
                // dropping a row without renumbering makes every higher id
                // resolve to the wrong variable. Renumbering here is not an
                // option either: `var_def_block` above is built once and keyed
                // by id, and the next fixpoint iteration reads it. Removing a
                // phi removes a *definition*; deleting the now-orphaned row is
                // `compact_variables`' job, and it already recognises variables
                // with no remaining definition.
            } else {
                // Rebuild mode: replace uses and remove unconditionally.
                //
                // The substitutions are composed into one map and applied in a
                // single pass. Doing them one at a time meant a whole-function
                // scan per trivial phi, and a rebuild produces trivial phis in
                // proportion to the function, so every fixpoint round was
                // quadratic — the same defect the repair branch above was
                // already corrected for.
                //
                // Sequential application is order-sensitive: each substitution
                // only sees the values left by the ones before it, so `p2 -> p1`
                // applied after `p1 -> x` leaves `p1`, not `x`. Composing the
                // map back-to-front reproduces that exactly — when an entry is
                // resolved every later entry is already final, and earlier ones
                // must not be followed.
                let resolved = {
                    // Every substitution target must be a variable that survives
                    // this round. Composing back-to-front only resolves a source
                    // through entries already inserted, so a chain
                    // `p1 -> p2 -> x` can leave `p1 -> p2` while `p2` is retired
                    // in the same round — the rewrite then points uses of `p1` at
                    // `p2`, which is deleted moments later, and those uses are
                    // stranded on a variable nothing defines. Walk each chain to
                    // its end instead, exactly as the repair branch above does.
                    let direct: BTreeMap<SsaVarId, SsaVarId> = trivial_phis
                        .iter()
                        .filter(|(result, source)| result != source)
                        .copied()
                        .collect();
                    let mut resolved: BTreeMap<SsaVarId, SsaVarId> = BTreeMap::new();
                    for (result, source) in &direct {
                        let mut current = *source;
                        let mut visited: BTreeSet<SsaVarId> = BTreeSet::new();
                        visited.insert(*result);
                        while let Some(&next) = direct.get(&current) {
                            if !visited.insert(current) {
                                break;
                            }
                            current = next;
                        }
                        if current != *result {
                            resolved.insert(*result, current);
                        }
                    }
                    resolved
                };
                if !resolved.is_empty() {
                    for block in &mut self.blocks {
                        for instr in block.instructions_mut() {
                            instr
                                .op_mut()
                                .replace_uses_with(|var| resolved.get(&var).copied());
                        }
                        for phi in block.phi_nodes_mut() {
                            for operand in phi.operands_mut() {
                                if let Some(&target) = resolved.get(&operand.value()) {
                                    *operand = PhiOperand::new(target, operand.predecessor());
                                }
                            }
                        }
                    }
                }

                // A self-referential phi is recorded as `(result, result)` and is
                // deliberately absent from `resolved` — there is no other value
                // to rewrite its uses to. Removing it regardless strands every
                // one of those uses on a variable nothing defines. Retire it
                // only once nothing reads it; a later fixpoint round collects it
                // after its readers go away. The repair branch already applies
                // exactly this condition.
                let still_read = self.collect_read_variables();
                let mut trivial_set = BitSet::new(variable_count);
                for (result, source) in &trivial_phis {
                    if result == source && still_read.contains_checked(result.index()) {
                        continue;
                    }
                    trivial_set.insert(result.index());
                }
                if trivial_set.is_empty() {
                    break;
                }
                total_eliminated = total_eliminated.saturating_add(trivial_set.count());
                for block in &mut self.blocks {
                    block.phi_nodes_mut().retain(|phi| {
                        let idx = phi.result().index();
                        idx >= variable_count || !trivial_set.contains(idx)
                    });
                }
                // Left in place for the same reason as the repair-mode branch
                // above: `compact_variables` owns row removal, because it is the
                // only path that renumbers afterwards.
            }
        }

        total_eliminated
    }

    /// Folds a constant operation, replacing its uses with the computed value.
    pub fn fold_constant(
        &mut self,
        block_idx: usize,
        instr_idx: usize,
        value: ConstValue<T>,
    ) -> bool {
        if let Some(block) = self.blocks.get_mut(block_idx)
            && let Some(instr) = block.instructions_mut().get_mut(instr_idx)
            && let Some(dest) = instr.op().dest()
        {
            instr.set_op_preserving_type(SsaOp::Const { dest, value });
            return true;
        }
        false
    }

    /// Returns the variables read by any live instruction or phi operand.
    ///
    /// Used by every path that would otherwise rewrite a variable's definition
    /// site to [`DefSite::entry`] after its defining instruction disappeared.
    /// That stamp is how an argument or a default-initialized local says "the
    /// caller supplies this", so applying it to a variable something still
    /// *reads* turns a dangling read into IR that looks legitimate — and
    /// [`SsaVerifier`](crate::analysis::verifier::SsaVerifier), whose job is to
    /// catch the pass that destroyed the definition, is left with nothing to
    /// see. Leaving the stale site in place is the honest answer: it still names
    /// an instruction, so the read is reported as an `UndefinedUse`.
    ///
    /// `Nop` instructions are skipped: a nopped instruction reads nothing.
    pub(in crate::ir::function) fn collect_read_variables(&self) -> BitSet {
        let mut read = BitSet::new(self.var_id_capacity());
        for block in &self.blocks {
            for instr in block.instructions() {
                if matches!(instr.op(), SsaOp::Nop) {
                    continue;
                }
                instr.op().for_each_use(|used| {
                    read.insert_checked(used.index());
                });
            }
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    read.insert_checked(operand.value().index());
                }
            }
        }
        read
    }

    /// Recomputes every variable's definition site from its current position.
    ///
    /// A definition site names the block and the index of the phi or
    /// instruction that produces the variable. Any edit that inserts, removes,
    /// or reorders instructions invalidates those indices, and a stale index
    /// that runs past the end of its block is rejected by index-bounds
    /// verification. This restores them from the IR as it actually stands.
    ///
    /// Variables with no remaining definition are reset to an entry site unless
    /// something still reads them, so a destroyed definition is not disguised as
    /// a legitimate entry value.
    pub fn refresh_def_sites(&mut self) {
        let variable_count = self.var_id_capacity();
        let mut active_defs = BitSet::new(variable_count);

        for (block_idx, block) in self.blocks.iter().enumerate() {
            for phi in block.phi_nodes() {
                let result = phi.result();
                let idx = result.index();
                if idx < variable_count {
                    active_defs.insert(idx);
                    if let Some(var) = self.variables.get_mut(idx) {
                        var.set_def_site(DefSite::phi(block_idx));
                    }
                }
            }

            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if matches!(instr.op(), SsaOp::Nop) {
                    continue;
                }
                for dest in instr.op().defs() {
                    let idx = dest.index();
                    if idx < variable_count {
                        active_defs.insert(idx);
                        if let Some(var) = self.variables.get_mut(idx) {
                            var.set_def_site(DefSite::instruction(block_idx, instr_idx));
                        }
                    }
                }
            }
        }

        let still_read = self.collect_read_variables();

        for var in &mut self.variables {
            let idx = var.id().index();
            if idx < variable_count
                && !active_defs.contains(idx)
                && !still_read.contains_checked(idx)
                && var.def_site().block != 0
            {
                var.set_def_site(DefSite::entry());
            }
        }
    }

    /// Compacts the variable table by removing orphaned variables.
    ///
    /// A variable is considered orphaned if:
    /// - It's not defined by any instruction in any block
    /// - It's not defined by any phi node in any block
    ///
    /// # Returns
    ///
    /// The number of variables that were removed.
    pub(in crate::ir::function) fn compact_variables(&mut self) -> usize {
        let variable_count = self.var_id_capacity();

        // Phase 1: Collect all variables that still have active definitions
        let mut defined_vars = BitSet::new(variable_count);

        for block in &self.blocks {
            // From instructions
            for instr in block.instructions() {
                let op = instr.op();
                // Skip Nop instructions - they have no definition
                if matches!(op, SsaOp::Nop) {
                    continue;
                }
                for dest in op.defs() {
                    let idx = dest.index();
                    if idx < variable_count {
                        defined_vars.insert(idx);
                    }
                }
            }

            // From phi nodes
            for phi in block.phi_nodes() {
                let idx = phi.result().index();
                if idx < variable_count {
                    defined_vars.insert(idx);
                }
            }
        }

        // Also keep version-0 entry-point variables. These have no instruction
        // def but are implicitly defined at function entry:
        // - Argument/Local v0: method parameters and default-initialized locals
        // - Phi v0 with entry def_site: placeholder reaching defs for stack temp
        //   groups created during SSA rebuild
        for var in &self.variables {
            if var.version() == 0 && var.def_site().instruction.is_none() {
                let idx = var.id().index();
                if idx < variable_count {
                    defined_vars.insert(idx);
                }
            }
        }

        // Also keep variables that are still referenced by non-Nop instructions.
        // This can happen when replace_uses skips replacements due to the
        // self-referential guard (dest == new_var), leaving uses behind after
        // the definition was Nop'd or eliminated.
        for block in &self.blocks {
            for instr in block.instructions() {
                if matches!(instr.op(), SsaOp::Nop) {
                    continue;
                }
                instr.for_each_use(|u| {
                    let idx = u.index();
                    if idx < variable_count {
                        defined_vars.insert(idx);
                    }
                });
            }

            // Also keep variables referenced by phi operands. A phi may
            // reference a variable whose defining instruction was Nop'd by
            // an optimization pass — without this, compact would remove the
            // variable and the phi operand would become a dangling reference.
            for phi in block.phi_nodes() {
                for op in phi.operands() {
                    let idx = op.value().index();
                    if idx < variable_count {
                        defined_vars.insert(idx);
                    }
                }
            }
        }

        // Phase 2: Remove orphaned variables
        let original_count = self.variables.len();
        self.variables.retain(|v| {
            let idx = v.id().index();
            idx < variable_count && defined_vars.contains(idx)
        });
        // Reassign dense IDs and rebuild registries
        let remap = self.reassign_dense_ids();
        self.remap_var_ids_in_blocks(&remap);
        self.rebuild_origin_versions();
        original_count.saturating_sub(self.variables.len())
    }

    /// Reassigns all variable IDs to dense contiguous indices (0..N-1) and
    /// remaps all references in blocks.
    ///
    /// **Warning**: This invalidates any externally-held `SsaVarId` references.
    pub fn reindex_variables(&mut self) -> usize {
        let remap = self.reassign_dense_ids();
        let remapped = remap.len();
        self.remap_var_ids_in_blocks(&remap);
        self.rebuild_origin_versions();
        remapped
    }

    /// Strips Nop instructions from all blocks and reindexes variable DefSites.
    ///
    /// This is the shared implementation used by both `repair_ssa` and
    /// `rebuild_ssa`. After stripping Nops:
    ///
    /// 1. Non-Nop instructions that shifted get their DefSites remapped
    /// 2. Variables whose defining instruction was a Nop get reset to entry DefSite
    /// 3. Any remaining out-of-bounds DefSites are reset to entry DefSite
    pub(in crate::ir::function) fn strip_nops(&mut self) {
        let mut remap: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut nop_sites: BTreeSet<(usize, usize)> = BTreeSet::new();

        for (block_idx, block) in self.blocks.iter_mut().enumerate() {
            let instructions = block.instructions_mut();

            if !instructions.iter().any(|i| matches!(i.op(), SsaOp::Nop)) {
                continue;
            }

            let mut new_idx = 0usize;
            for (old_idx, instr) in instructions.iter().enumerate() {
                if matches!(instr.op(), SsaOp::Nop) {
                    nop_sites.insert((block_idx, old_idx));
                } else {
                    if old_idx != new_idx {
                        remap.insert((block_idx, old_idx), new_idx);
                    }
                    new_idx = new_idx.saturating_add(1);
                }
            }

            instructions.retain(|instr| !matches!(instr.op(), SsaOp::Nop));
        }

        // Update variable DefSites to reflect new instruction positions.
        // A variable whose defining instruction was a Nop is reset to entry —
        // unless something still reads it, in which case the reset would
        // disguise a dangling read as a legitimate entry definition. See
        // `collect_read_variables`.
        if !remap.is_empty() || !nop_sites.is_empty() {
            let still_read = self.collect_read_variables();
            for var in &mut self.variables {
                let site = var.def_site();
                if let Some(old_instr) = site.instruction {
                    if nop_sites.contains(&(site.block, old_instr)) {
                        if !still_read.contains_checked(var.id().index()) {
                            var.set_def_site(DefSite::entry());
                        }
                    } else if let Some(&new_instr) = remap.get(&(site.block, old_instr)) {
                        var.set_def_site(DefSite::instruction(site.block, new_instr));
                    }
                }
            }
        }

        // Validate remaining DefSites are in-bounds. Catches stale DefSites
        // that existed before strip_nops was called (e.g., from passes that
        // modified instructions without updating DefSites).
        let block_instr_counts: Vec<usize> =
            self.blocks.iter().map(|b| b.instructions().len()).collect();

        for var in &mut self.variables {
            let site = var.def_site();
            if let Some(instr_idx) = site.instruction {
                let out_of_bounds = match block_instr_counts.get(site.block) {
                    Some(&count) => instr_idx >= count,
                    None => true,
                };
                if out_of_bounds {
                    var.set_def_site(DefSite::entry());
                }
            }
        }
    }

    /// Eliminates dead phi nodes whose result is never used.
    ///
    /// A phi is dead if its result variable has no consumers (no instruction
    /// or other phi uses it). Handles dead phi cycles (A uses B, B uses A,
    /// neither used elsewhere) via liveness propagation.
    ///
    /// Also bridges implicit uses from `LoadLocal`/`LoadArg` instructions
    /// to the corresponding phi nodes for that local/arg origin, ensuring
    /// phis that are read by index-based loads are not incorrectly eliminated.
    pub fn eliminate_dead_phis(&mut self) {
        let variable_count = self.var_id_capacity();
        let mut all_phi_results = BitSet::new(variable_count);
        for block in &self.blocks {
            for phi in block.phi_nodes() {
                let idx = phi.result().index();
                if idx < variable_count {
                    all_phi_results.insert(idx);
                }
            }
        }

        if all_phi_results.is_empty() {
            return;
        }

        // Build map from phi origin to phi result IDs for LoadLocal/LoadArg bridging.
        let mut origin_to_phi_results: BTreeMap<VariableOrigin, Vec<SsaVarId>> = BTreeMap::new();
        for block in &self.blocks {
            for phi in block.phi_nodes() {
                origin_to_phi_results
                    .entry(phi.origin())
                    .or_default()
                    .push(phi.result());
            }
        }

        // Phase 1: Mark phis as live if used by any non-phi instruction
        let mut live_phis = BitSet::new(variable_count);
        for block in &self.blocks {
            for instr in block.instructions() {
                // Direct SSA uses
                instr.for_each_use(|u| {
                    let idx = u.index();
                    if idx < variable_count && all_phi_results.contains(idx) {
                        live_phis.insert(idx);
                    }
                });

                // Implicit uses via LoadLocal/LoadArg (index-based reads).
                // These don't appear in uses() but create a dependency on
                // the corresponding PHI node for that local/arg origin.
                match instr.op() {
                    SsaOp::LoadLocal { local_index, .. } => {
                        let origin = VariableOrigin::Local(*local_index);
                        if let Some(phi_results) = origin_to_phi_results.get(&origin) {
                            for &phi_result in phi_results {
                                let idx = phi_result.index();
                                if idx < variable_count {
                                    live_phis.insert(idx);
                                }
                            }
                        }
                    }
                    SsaOp::LoadArg { arg_index, .. } => {
                        let origin = VariableOrigin::Argument(*arg_index);
                        if let Some(phi_results) = origin_to_phi_results.get(&origin) {
                            for &phi_result in phi_results {
                                let idx = phi_result.index();
                                if idx < variable_count {
                                    live_phis.insert(idx);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Phase 2: Propagate liveness through phi operands
        let mut changed = true;
        while changed {
            changed = false;
            for block in &self.blocks {
                for phi in block.phi_nodes() {
                    let result_idx = phi.result().index();
                    if result_idx < variable_count && live_phis.contains(result_idx) {
                        for op in phi.operands() {
                            let val_idx = op.value().index();
                            if val_idx < variable_count
                                && all_phi_results.contains(val_idx)
                                && live_phis.insert(val_idx)
                            {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Phase 3: Remove dead phis (all_phi_results - live_phis)
        let mut dead_phis = all_phi_results.clone();
        dead_phis.difference_with(&live_phis);

        if dead_phis.is_empty() {
            return;
        }

        for block in &mut self.blocks {
            block.phi_nodes_mut().retain(|phi| {
                let idx = phi.result().index();
                idx >= variable_count || !dead_phis.contains(idx)
            });
        }

        // As in `eliminate_trivial_phis`: the orphaned rows stay until
        // `compact_variables` removes them, because that is the only path that
        // renumbers `variables` and remaps the ids in blocks afterwards.
        // Dropping them here would break `variables[i].id().index() == i`.
    }

    /// Shrinks `num_locals` to the actual maximum local index in use.
    ///
    /// After `compact_variables()` removes unused variables, `num_locals` may
    /// exceed the actual maximum local index referenced. This scans all
    /// `VariableOrigin::Local(idx)` references (variables, phi nodes, and
    /// `LoadLocal`/`LoadLocalAddr` instructions) to find the true maximum, then
    /// sets `num_locals = max(max_used + 1, original_num_locals)`.
    ///
    /// The `original_num_locals` floor ensures we never drop below the method's
    /// declared local count (those locals have default-initialization semantics).
    pub fn shrink_num_locals(&mut self) {
        let mut max_local_idx: Option<u16> = None;

        // From variables
        for var in &self.variables {
            if let VariableOrigin::Local(idx) = var.origin() {
                max_local_idx = Some(max_local_idx.map_or(idx, |cur| cur.max(idx)));
            }
        }

        // From phi nodes
        for block in &self.blocks {
            for phi in block.phi_nodes() {
                if let VariableOrigin::Local(idx) = phi.origin() {
                    max_local_idx = Some(max_local_idx.map_or(idx, |cur| cur.max(idx)));
                }
            }
        }

        // From LoadLocal and LoadLocalAddr instructions
        for block in &self.blocks {
            for instr in block.instructions() {
                match instr.op() {
                    SsaOp::LoadLocal { local_index, .. }
                    | SsaOp::LoadLocalAddr { local_index, .. } => {
                        max_local_idx =
                            Some(max_local_idx.map_or(*local_index, |cur| cur.max(*local_index)));
                    }
                    _ => {}
                }
            }
        }

        let needed = max_local_idx.map_or(0, |idx| (idx as usize).saturating_add(1));
        self.num_locals = needed.max(self.original_num_locals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{block::SsaBlock, instruction::SsaInstruction, phi::PhiNode},
        testing::{MockTarget, MockType},
    };

    /// Builds `B0 -> B2`, `B1 -> B2`, where `B2` carries a phi merging one value
    /// defined in `B0` with a second operand whose value has no definition in any
    /// block. Both incoming edges are live.
    fn merge_with_one_undefined_operand() -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);

        let phi_res = SsaVarId::from_index(0);
        ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::phi(2), MockType::I32);
        let defined = SsaVarId::from_index(1);
        ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        // Registered but never defined by any instruction or phi.
        let undefined = SsaVarId::from_index(2);
        ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(9, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: defined,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_res, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(defined, 0));
        phi.add_operand(PhiOperand::new(undefined, 1));
        b2.add_phi(phi);
        b2.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(phi_res),
        }));
        ssa.add_block(b2);

        ssa.recompute_uses();
        ssa
    }

    /// Builds `B0 -> {B1, B2} -> B3` where `B3` carries a phi whose result is
    /// never used, and where a *later* variable (`kept`) is allocated after the
    /// dead phi's result. Removing the phi's row from `variables` without
    /// renumbering shifts `kept` down a slot, so `variables[i].id().index() == i`
    /// no longer holds.
    fn dead_phi_before_a_live_variable() -> SsaFunction<MockTarget> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);

        // Slot 0: the branch condition, defined in B0.
        let cond = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        // Slot 1: the dead phi's result, defined by the phi in B3.
        let dead = SsaVarId::from_index(1);
        ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(3), MockType::I32);
        // Slot 2: allocated *after* the dead phi — this is the one that shifts.
        let kept = SsaVarId::from_index(2);
        ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(3, 0),
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
            block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
            ssa.add_block(block);
        }

        let mut b3 = SsaBlock::new(3);
        // Dead: `dead` is never read by any instruction or phi operand.
        let mut phi = PhiNode::new(dead, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(cond, 1));
        phi.add_operand(PhiOperand::new(cond, 2));
        b3.add_phi(phi);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: kept,
            value: ConstValue::I32(7),
        }));
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
            value: Some(kept),
        }));
        ssa.add_block(b3);

        ssa.recompute_uses();
        ssa
    }

    /// A chain `p1 = phi(x); p2 = phi(p1); ...` is entirely trivial and must
    /// collapse in a bounded number of fixpoint rounds.
    ///
    /// `replace_uses_checked_with` rewrites only instruction uses, so without the
    /// operand collapse each phi in the chain stays "read" by the next phi's
    /// operand and exactly one phi retires per round. The fixpoint then runs once
    /// per phi, and each round is already O(phis x function) — cubic, and
    /// measurably so: this shape reaches seconds at a few hundred phis and tens
    /// of seconds at 1,600.
    #[test]
    fn a_chain_of_trivial_phis_collapses_completely() {
        const CHAIN: usize = 512;

        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, CHAIN + 1);
        let seed = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: seed,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut prev = seed;
        for id in 1..=CHAIN {
            let origin = VariableOrigin::Local(u16::try_from(id % 1000).unwrap_or(0));
            let result = ssa.create_variable(origin, 0, DefSite::phi(id), MockType::I32);
            let mut block = SsaBlock::new(id);
            let mut phi = PhiNode::new(result, origin);
            phi.add_operand(PhiOperand::new(prev, id.saturating_sub(1)));
            block.add_phi(phi);
            if id == CHAIN {
                block.add_instruction(SsaInstruction::synthetic(SsaOp::Return {
                    value: Some(result),
                }));
            } else {
                block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump {
                    target: id.saturating_add(1),
                }));
            }
            ssa.add_block(block);
            prev = result;
        }
        ssa.recompute_uses();

        ssa.repair_ssa();

        let remaining: usize = ssa.blocks().iter().map(|b| b.phi_nodes().len()).sum();
        assert_eq!(
            remaining, 0,
            "every phi in the chain is trivial and must be eliminated"
        );
        assert_dense_variable_table(&ssa, "after collapsing a trivial phi chain");
        // The return must now read the value the chain forwarded.
        let returned = ssa
            .blocks()
            .iter()
            .find_map(|block| match block.control_terminator() {
                Some(SsaOp::Return { value: Some(v) }) => Some(*v),
                _ => None,
            });
        assert_eq!(
            returned,
            Some(seed),
            "the chain forwarded the seed, so the return must read it directly"
        );
    }

    /// `SsaFunction::variable(id)` is a raw index into `variables`, so the whole
    /// IR depends on `variables[i].id().index() == i`. Any method that drops a
    /// row must renumber, or every id above the hole silently resolves to a
    /// *different* variable's `def_site`/`var_type`/use list — which
    /// `can_replace_instruction_use_with_dominators` then reads to approve
    /// replacements that violate dominance.
    fn assert_dense_variable_table(ssa: &SsaFunction<MockTarget>, label: &str) {
        for (slot, var) in ssa.variables().iter().enumerate() {
            assert_eq!(
                var.id().index(),
                slot,
                "{label}: variable table density broken at slot {slot} \
                 (found id {}), so variable(id) now resolves to the wrong row",
                var.id().index()
            );
        }
    }

    #[test]
    fn eliminate_dead_phis_preserves_dense_variable_table() {
        let mut ssa = dead_phi_before_a_live_variable();
        assert_dense_variable_table(&ssa, "before");

        ssa.eliminate_dead_phis();

        assert!(
            ssa.block(3).unwrap().phi_nodes().is_empty(),
            "the dead phi should have been removed"
        );
        assert_dense_variable_table(&ssa, "after eliminate_dead_phis");
    }

    #[test]
    fn eliminate_trivial_phis_preserves_dense_variable_table() {
        let mut ssa = dead_phi_before_a_live_variable();

        // The phi merges the same value on both edges, so it is trivial as well
        // as dead; repair mode removes it through the trivial-phi path.
        ssa.eliminate_trivial_phis(&TrivialPhiOptions { reachable: None });

        assert_dense_variable_table(&ssa, "after eliminate_trivial_phis");
    }

    /// An operand on a live incoming edge must survive pruning even when its
    /// value has no reachable definition. Dropping it leaves the phi without an
    /// operand for a real predecessor (`MissingPhiOperand`), and shrinks the phi
    /// to a single operand that trivial-phi simplification then collapses into a
    /// copy — silently discarding the other merged value.
    #[test]
    fn prune_phi_operands_keeps_live_edges_with_undefined_values() {
        let mut ssa = merge_with_one_undefined_operand();
        let mut reachable = BitSet::new(3);
        for block in 0..3 {
            reachable.insert(block);
        }

        let pruned = ssa.prune_phi_operands(&reachable);

        assert_eq!(pruned, 0, "no live-edge operand may be pruned");
        let operands = ssa.block(2).unwrap().phi_nodes()[0].operands();
        assert_eq!(
            operands.len(),
            2,
            "both incoming edges must retain an operand"
        );
        let mut preds: Vec<usize> = operands.iter().map(PhiOperand::predecessor).collect();
        preds.sort_unstable();
        assert_eq!(preds, vec![0, 1]);
    }

    /// Control: an operand naming a block that is not a predecessor is still
    /// pruned. Structural staleness remains the sole removal criterion.
    #[test]
    fn prune_phi_operands_drops_operands_from_non_predecessors() {
        let mut ssa = merge_with_one_undefined_operand();
        let defined = SsaVarId::from_index(1);
        // B1 no longer reaches B2, so its operand is stale.
        ssa.block_mut(1)
            .unwrap()
            .instructions_mut()
            .last_mut()
            .unwrap()
            .set_op(SsaOp::Return { value: None });

        let mut reachable = BitSet::new(3);
        for block in 0..3 {
            reachable.insert(block);
        }

        let pruned = ssa.prune_phi_operands(&reachable);

        assert_eq!(pruned, 1, "the stale operand must be pruned");
        let operands = ssa.block(2).unwrap().phi_nodes()[0].operands();
        assert_eq!(operands.len(), 1);
        assert_eq!(operands[0].predecessor(), 0);
        assert_eq!(operands[0].value(), defined);
    }
}
