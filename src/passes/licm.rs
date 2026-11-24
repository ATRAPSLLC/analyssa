//! Loop Invariant Code Motion (LICM) pass - moves loop-invariant
//! computations into the loop preheader.
//!
//! # Algorithm
//!
//! 1. Analyze the loop structure using [`LoopAnalyzer`].
//! 2. Process loops innermost-first so hoisted values are available to
//!    outer loops in subsequent passes.
//! 3. For each loop with a valid preheader:
//!    a. Find invariant instructions - those whose operands are all
//!    defined outside the loop or by other invariants, excluding
//!    header phi defs.
//!    b. Filter hoistable candidates: reject instructions that feed a phi
//!    operand on an intra-loop back-edge (would collapse per-edge
//!    attribution), and reject instructions whose removal would leave
//!    a trampoline block whose successor has phis.
//!    c. Hoist the whole invariant dependency chain in a single pass,
//!    inserting instructions just before the preheader's terminator in
//!    topological (data-dependency) order so a hoisted def always precedes
//!    its hoisted uses.
//!    d. Replace original instructions with `Nop`.
//!
//! # Conservative Guards
//!
//! - Requires a loop preheader that is a CFG predecessor of the header.
//! - Skips loops whose header terminator is a `Switch`: the CFF dispatcher
//!   pattern, left as the obfuscator wrote it for unflattening.
//! - Rejects instructions whose result transitively feeds a phi on an
//!   intra-loop edge (would erase per-edge value distinctions).
//! - Avoids creating trampoline blocks whose successor has phis (phi
//!   stability across subsequent block merging).

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, VecDeque},
    mem,
};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    Result,
    analysis::{loopanalyzer::LoopAnalyzer, loops::LoopInfo},
    bitset::BitSet,
    events::EventKind,
    facts::Derivation,
    graph::NodeId,
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        instruction::SsaInstruction,
        ops::SsaOp,
        variable::{DefSite, SsaVarId, VariableOrigin},
        varstore::VarSet,
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// One loop's planned hoist: where the instructions go and which ones move.
struct LoopHoist<T: Target> {
    /// Preheader block receiving the hoisted instructions.
    preheader: usize,
    /// Index in the preheader at which to begin inserting.
    insert_base: usize,
    /// `(source block, source instruction, op)` for each instruction to move,
    /// already ordered so every producer precedes its consumers.
    to_hoist: Vec<(usize, usize, SsaOp<T>)>,
}

/// The hoisting analysis of one loop: the function and the loop it plans to
/// move invariant instructions out of.
struct HoistPlanner<'a, T: Target> {
    /// The function the loop is in.
    ssa: &'a SsaFunction<T>,
    /// The loop.
    loop_info: &'a LoopInfo,
}

impl<T: Target> HoistPlanner<'_, T> {
    /// Decides what `loop_info` can contribute, without mutating anything.
    ///
    /// Split out of [`LicmPass`] so the whole forest can be processed inside a single
    /// edit scope: the analysis needs to see the IR as previous loops left it, so it
    /// runs against `ssa` - the editor's current function - rather than a snapshot
    /// taken before the pass began. Bails early on the conservative guards (no
    /// preheader, a preheader that is not a CFG predecessor of the header, or a
    /// `Switch` header), then narrows the invariant set through the availability and
    /// trampoline filters run to a *joint* fixpoint - each can invalidate the
    /// other's conclusion - and finally topologically orders the survivors so a
    /// producer never lands after its consumer in the preheader.
    ///
    /// # Returns
    ///
    /// A [`LoopHoist`] describing the preheader, insertion point, and ordered
    /// instructions to move, or `None` when the loop contributes nothing.
    fn plan(&self) -> Option<LoopHoist<T>> {
        let preheader = self.loop_info.preheader?;

        let header_idx = self.loop_info.header.index();
        let preheader_is_pred = self
            .ssa
            .block(preheader.index())
            .map(|b| {
                b.instructions()
                    .last()
                    .map(|i| i.op().has_successor(header_idx))
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        if !preheader_is_pred {
            return None;
        }

        let header_has_switch = self
            .ssa
            .block(header_idx)
            .and_then(|b| b.control_terminator())
            .is_some_and(|op| matches!(op, SsaOp::Switch { .. }));
        if header_has_switch {
            return None;
        }

        let inside_defs = self.inside_defs();
        let invariants = self.invariants(&inside_defs);
        if invariants.is_empty() {
            return None;
        }

        let phi_back_edge_operands = self.phi_back_edge_operands();
        let back_edge_tainted = self.back_edge_tainted(&phi_back_edge_operands);
        let mut hoistable: Vec<_> = invariants
            .into_iter()
            .filter(|(block_idx, instr_idx)| {
                self.can_hoist(&back_edge_tainted, *block_idx, *instr_idx)
            })
            .collect();

        let preheader_idx = preheader.index();
        let insert_base = if let Some(preheader_block) = self.ssa.block(preheader_idx) {
            let instrs = preheader_block.instructions();
            if instrs.is_empty() {
                0
            } else if instrs.last().is_some_and(SsaInstruction::is_terminator) {
                instrs.len().saturating_sub(1)
            } else {
                instrs.len()
            }
        } else {
            0
        };

        // A hoisted instruction's operands must all already be available in the
        // preheader: defined outside the loop body (i.e. not in `inside_defs`)
        // and not produced by a preheader instruction at or after the insertion
        // point. `preheader_after` is that small exclusion set, collected in
        // O(preheader). Re-deriving "outside" instead would need an
        // all-variables scan per loop.
        let mut preheader_after = VarSet::new(self.ssa.var_id_bound());
        if let Some(preheader_block) = self.ssa.block(preheader_idx) {
            for (instr_idx, instr) in preheader_block.instructions().iter().enumerate() {
                if instr_idx >= insert_base {
                    for def in instr.op().defs() {
                        preheader_after.insert(def);
                    }
                }
            }
        }

        // Both filters below can invalidate the other's conclusion, so they run to a
        // joint fixpoint. The availability filter keeps an instruction only because
        // its producer is *also* being hoisted; the trampoline filter then drops
        // whole blocks, which can remove that producer and leave the dependent
        // hoisted above its own definition. Running availability once and the
        // trampoline filter afterwards would produce exactly that, and nothing
        // downstream would catch it: the edit stays at
        // `SsaEditScope::InstructionsOnly`, so the boundary only runs `repair_ssa`,
        // and it verifies at `VerifyLevel::Standard`, whose
        // `check_defined_before_use` tests existence rather than dominance.
        loop {
            let joint_before = hoistable.len();

            loop {
                let before = hoistable.len();
                // Defs produced by the instructions still slated for hoisting. An
                // invariant whose operands are all either defined outside the loop
                // OR by another hoisted instruction is itself hoistable - the whole
                // dependency chain moves together in one pass (ordered
                // topologically below). Recomputed each round because dropping an
                // instruction can make its dependents unhoistable.
                let hoistable_defs: FxHashSet<SsaVarId> = hoistable
                    .iter()
                    .filter_map(|(b, i)| self.ssa.block(*b).and_then(|blk| blk.instruction(*i)))
                    .flat_map(|instr| instr.op().defs())
                    .collect();
                hoistable.retain(|(block_idx, instr_idx)| {
                    let Some(block) = self.ssa.block(*block_idx) else {
                        return false;
                    };
                    let Some(instr) = block.instruction(*instr_idx) else {
                        return false;
                    };
                    let def_used_before_insert = instr.op().defs().any(|def| {
                        self.ssa.variable(def).is_some_and(|var| {
                            var.uses().iter().any(|site| {
                                site.block() == preheader_idx
                                    && !site.is_phi_operand()
                                    && site.index() < insert_base
                            })
                        })
                    });
                    if def_used_before_insert {
                        return false;
                    }
                    let mut operands_are_available = true;
                    instr.op().for_each_use(|operand| {
                        let available_outside =
                            !inside_defs.contains(operand) && !preheader_after.contains(operand);
                        operands_are_available &=
                            available_outside || hoistable_defs.contains(&operand);
                    });
                    operands_are_available
                });
                if hoistable.len() == before {
                    break;
                }
            }

            // Skip blocks that would become trampolines whose successor has phis.
            {
                let mut hoist_count_per_block: FxHashMap<usize, usize> = FxHashMap::default();
                for (block_idx, _) in &hoistable {
                    let entry = hoist_count_per_block.entry(*block_idx).or_insert(0);
                    *entry = entry.saturating_add(1);
                }
                let mut trampoline_blocks = BitSet::new(self.ssa.block_count());
                for (&block_idx, &hoist_count) in &hoist_count_per_block {
                    if let Some(block) = self.ssa.block(block_idx) {
                        let non_term = block
                            .instructions()
                            .iter()
                            .filter(|i| !i.is_terminator() && !matches!(i.op(), SsaOp::Nop))
                            .count();
                        if hoist_count >= non_term
                            && let Some(term) = block.control_terminator()
                        {
                            term.for_each_successor(|succ| {
                                if let Some(succ_block) = self.ssa.block(succ)
                                    && !succ_block.phi_nodes().is_empty()
                                {
                                    trampoline_blocks.insert(block_idx);
                                }
                            });
                        }
                    }
                }
                if !trampoline_blocks.is_empty() {
                    hoistable.retain(|(block_idx, _)| !trampoline_blocks.contains(*block_idx));
                }
            }

            // Neither filter removed anything this round, so the set is stable under
            // both and every retained instruction's operands really are available.
            if hoistable.len() == joint_before {
                break;
            }
        }

        if hoistable.is_empty() {
            return None;
        }

        let mut to_hoist: Vec<(usize, usize, SsaOp<T>)> = Vec::new();
        for (block_idx, instr_idx) in &hoistable {
            if let Some(block) = self.ssa.block(*block_idx)
                && let Some(instr) = block.instruction(*instr_idx)
            {
                to_hoist.push((*block_idx, *instr_idx, instr.op().clone()));
            }
        }

        // Order hoisted instructions so every producer precedes its consumers
        // in the preheader. Original block/instruction position is NOT a valid
        // order across blocks (a def can sit in a later-indexed block than its
        // use), which is why hoisting a full dependency chain requires a real
        // topological sort rather than a positional one.
        topological_hoist_order(&mut to_hoist);

        Some(LoopHoist {
            preheader: preheader_idx,
            insert_base,
            to_hoist,
        })
    }

    /// Collects the variables whose definition site lies inside the loop body -
    /// instruction results and phi results in body blocks.
    ///
    /// Invariance is decided against this set rather than its complement (the
    /// definitions outside the loop): scanning only the loop body makes it
    /// O(loop-body), where deriving the complement means scanning every variable
    /// in the whole function once per loop (O(loops × variables)). Consumed by [`is_instruction_invariant`], where an
    /// operand in this set that is not yet proven invariant breaks invariance.
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// The set of variables defined inside `loop_info`'s body.
    fn inside_defs(&self) -> VarSet {
        let mut inside = VarSet::new(self.ssa.var_id_bound());
        for block_idx in self.loop_info.body.iter() {
            if let Some(block) = self.ssa.block(block_idx) {
                for phi in block.phi_nodes() {
                    inside.insert(phi.result());
                }
                for instr in block.instructions() {
                    for def in instr.op().defs() {
                        inside.insert(def);
                    }
                }
            }
        }
        inside
    }

    /// Finds the loop-body instructions that are loop-invariant.
    ///
    /// Runs a monotone worklist fixpoint: every non-terminator, non-`Nop` body
    /// instruction is seeded, and an instruction is admitted once
    /// [`is_instruction_invariant`] holds - its operands are all defined outside the
    /// loop or by an already-admitted invariant, and none is a header phi result.
    /// When an instruction is admitted its defs join `invariant_defs` and only the
    /// instructions that *use* those defs are re-queued, through a scoped def→users
    /// map, so the pass is O(loop) rather than the O(loop²) of a
    /// `while changed { rescan body }` loop.
    ///
    /// # Arguments
    ///
    /// * `inside_defs` - Variables defined inside the loop body, from
    ///   [`Self::inside_defs`].
    ///
    /// # Returns
    ///
    /// The `(block, instruction)` positions of the invariant instructions.
    fn invariants(&self, inside_defs: &VarSet) -> Vec<(usize, usize)> {
        let mut invariants: FxHashSet<(usize, usize)> = FxHashSet::default();
        let mut invariant_defs = VarSet::new(self.ssa.var_id_bound());

        let mut header_phi_defs = VarSet::new(self.ssa.var_id_bound());
        if let Some(header_block) = self.ssa.block(self.loop_info.header.index()) {
            for phi in header_block.phi_nodes() {
                header_phi_defs.insert(phi.result());
            }
        }

        // Worklist over loop-body instructions: seed with all of them, and when an
        // instruction becomes invariant, only re-examine the instructions that use
        // its definitions (via a scoped def→users map). A
        // `while changed { rescan whole loop body }` fixpoint would be O(loop^2).
        let mut users: FxHashMap<SsaVarId, Vec<(usize, usize)>> = FxHashMap::default();
        let mut worklist: VecDeque<(usize, usize)> = VecDeque::new();
        let mut queued: FxHashSet<(usize, usize)> = FxHashSet::default();
        for block_idx in self.loop_info.body.iter() {
            if let Some(block) = self.ssa.block(block_idx) {
                for (instr_idx, instr) in block.instructions().iter().enumerate() {
                    if instr.is_terminator() || matches!(instr.op(), SsaOp::Nop) {
                        continue;
                    }
                    instr.op().for_each_use(|u| {
                        users.entry(u).or_default().push((block_idx, instr_idx));
                    });
                    worklist.push_back((block_idx, instr_idx));
                    queued.insert((block_idx, instr_idx));
                }
            }
        }

        while let Some((block_idx, instr_idx)) = worklist.pop_front() {
            queued.remove(&(block_idx, instr_idx));
            if invariants.contains(&(block_idx, instr_idx)) {
                continue;
            }
            let Some(instr) = self
                .ssa
                .block(block_idx)
                .and_then(|b| b.instructions().get(instr_idx))
            else {
                continue;
            };
            if is_instruction_invariant(instr, inside_defs, &invariant_defs, &header_phi_defs) {
                invariants.insert((block_idx, instr_idx));
                for def in instr.defs() {
                    invariant_defs.insert(def);
                    if let Some(dependents) = users.get(&def) {
                        for &(ub, ui) in dependents {
                            if !invariants.contains(&(ub, ui)) && queued.insert((ub, ui)) {
                                worklist.push_back((ub, ui));
                            }
                        }
                    }
                }
            }
        }

        invariants.into_iter().collect()
    }

    /// Tests whether an invariant instruction is *safe* to hoist out of the loop.
    ///
    /// Invariance is necessary but not sufficient: this additionally requires the
    /// instruction to define a value (nothing to move otherwise), to be pure (a
    /// side-effecting op cannot be pulled out of the loop's execution count), to
    /// belong to a loop that has a preheader to hoist into, and to produce no value
    /// in `back_edge_tainted` - hoisting a value that feeds a phi operand on a
    /// back-edge would erase the per-edge value distinction. The taint set is
    /// precomputed once per loop, making the last check an O(1) membership test.
    ///
    /// # Arguments
    ///
    /// * `back_edge_tainted` - Values that transitively feed a back-edge phi
    ///   operand, from [`Self::back_edge_tainted`].
    /// * `block_idx` - Block holding the candidate instruction.
    /// * `instr_idx` - Index of the candidate instruction within the block.
    ///
    /// # Returns
    ///
    /// `true` if the instruction may be hoisted.
    fn can_hoist(&self, back_edge_tainted: &VarSet, block_idx: usize, instr_idx: usize) -> bool {
        let Some(block) = self.ssa.block(block_idx) else {
            return false;
        };
        let Some(instr) = block.instruction(instr_idx) else {
            return false;
        };
        if !instr.has_def() {
            return false;
        }
        if !instr.op().effects().is_pure() {
            return false;
        }
        // A constant or a copy into a storage location only writes it: there is no
        // computation to take out of the loop. Hoisted, it leaves a copy behind (see
        // the copy `run` leaves for a hoisted location version), which is the same
        // write - and constant propagation turns a copy of a constant back into the
        // constant, for the next run to hoist again.
        if let SsaOp::Copy { dest, .. } | SsaOp::Const { dest, .. } = instr.op()
            && self.ssa.rename_group(*dest) != u32::MAX
        {
            return false;
        }
        if self.loop_info.preheader.is_none() {
            return false;
        }
        // Reject instructions whose result transitively feeds a phi operand on a
        // loop back-edge (hoisting would erase per-edge value distinctions). The
        // taint set is precomputed once per loop, so this is an O(1) membership
        // test rather than a per-candidate def-use traversal.
        for dest in instr.defs() {
            if back_edge_tainted.contains(dest) {
                return false;
            }
        }
        true
    }

    /// Computes, in one backward pass, the set of in-loop values that transitively
    /// feed a phi operand on a loop back-edge.
    ///
    /// [`Self::can_hoist`] must reject any instruction whose result feeds such a phi.
    /// Answering that per candidate with an independent forward def-use walk is
    /// `O(candidates × loop)`; seeding a worklist from the back-edge phi operands
    /// and propagating backward through in-loop defining instructions answers it
    /// for *every* value in `O(loop)` once. A value is tainted when it is itself a
    /// back-edge operand, or when it is an operand of an in-loop instruction whose
    /// result is tainted.
    ///
    /// # Arguments
    ///
    /// * `phi_back_edge_operands` - The seed set: phi operands arriving on an
    ///   intra-loop back-edge, from [`Self::phi_back_edge_operands`].
    ///
    /// # Returns
    ///
    /// The set of in-loop values transitively feeding a back-edge phi operand.
    fn back_edge_tainted(&self, phi_back_edge_operands: &VarSet) -> VarSet {
        let mut tainted = phi_back_edge_operands.clone();
        let mut worklist: VecDeque<SsaVarId> = phi_back_edge_operands.iter().collect();
        while let Some(value) = worklist.pop_front() {
            // This set is a *safety* set - `can_hoist` refuses every instruction
            // whose result is in it - so the two lookups fail in opposite
            // directions. Dropping a link shrinks the set, and a value that should
            // have been tainted is one that gets hoisted out of the loop: the
            // fail-open direction, and a miscompile. Over-tainting only costs a
            // hoist. So a stale def site is *recovered* by scan here rather than
            // refused, and a definition the scan finds but cannot place is treated
            // as in-loop.
            //
            // A def site is also variable *metadata*, not a fact about the current
            // CFG: it can name a block this function no longer has, or never had.
            // The membership test below therefore has to be as tolerant as the
            // lookup, rather than asserting one line earlier.
            //
            // It asserted, and a `BitSet` bounds assert is a **panic**: measured over
            // 21,600 binaries it fired 8 times, each one unwinding out of LICM into
            // `guard_pass`, which rolled the whole pass back and left the function
            // unoptimised with nothing upstream the wiser.
            let op = match self.ssa.recorded_definition(value) {
                Some(recorded) => {
                    if !self.loop_info.contains(NodeId::new(recorded.block_index())) {
                        continue;
                    }
                    recorded.op()
                }
                None => {
                    let Some(instr) = self.ssa.get_definition_instruction(value) else {
                        continue;
                    };
                    instr.op()
                }
            };
            op.for_each_use(|operand| {
                if tainted.insert(operand) {
                    worklist.push_back(operand);
                }
            });
        }
        tainted
    }

    /// Collects phi operands that arrive on an intra-loop back-edge.
    ///
    /// Walks every phi in the loop body and keeps each operand whose predecessor
    /// block is itself inside the loop - those are the values carried around the
    /// loop rather than in from outside. This is the seed set for
    /// [`Self::back_edge_tainted`], which propagates the taint backward through
    /// defining instructions. A phi operand's predecessor is raw metadata and may
    /// name a block the function no longer has, which is trivially not in the loop
    /// body and so simply excluded.
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    /// The set of phi operand values incoming on an intra-loop back-edge.
    fn phi_back_edge_operands(&self) -> VarSet {
        let mut operands = VarSet::new(self.ssa.var_id_bound());
        for phi_block_idx in self.loop_info.body.iter() {
            let Some(phi_block) = self.ssa.block(phi_block_idx) else {
                continue;
            };
            for phi in phi_block.phi_nodes() {
                for operand in phi.operands() {
                    // Raw phi-operand predecessor: may name a block that does not
                    // exist, which is trivially not in the loop body.
                    if self.loop_info.contains(NodeId::new(operand.predecessor())) {
                        operands.insert(operand.value());
                    }
                }
            }
        }
        operands
    }
}

/// Reorders `items` so every producer precedes its consumers.
///
/// All hoisted instructions move into the same preheader, so a use must never
/// be inserted before its definition. Original block/instruction position is not
/// a valid order across blocks (a def can live in a higher-indexed block than
/// its use), so this performs a deterministic topological sort - Kahn's
/// algorithm with ties broken by original `(block, instruction)` position.
/// Loop-invariant instructions cannot form a data-dependency cycle; if one
/// somehow survives, the remainder is appended in positional order as a
/// defensive fallback rather than dropped.
///
/// # Arguments
///
/// * `items` - The `(source block, source instruction, op)` triples to move,
///   reordered in place into producer-before-consumer order.
fn topological_hoist_order<T: Target>(items: &mut Vec<(usize, usize, SsaOp<T>)>) {
    let n = items.len();
    if n < 2 {
        return;
    }

    let positions: Vec<(usize, usize)> = items.iter().map(|(b, i, _)| (*b, *i)).collect();
    let pos_of = |idx: usize| positions.get(idx).copied().unwrap_or_default();

    // Variable -> index of the hoisted instruction that defines it.
    let mut def_owner: FxHashMap<SsaVarId, usize> = FxHashMap::default();
    for (idx, (_, _, op)) in items.iter().enumerate() {
        for def in op.defs() {
            def_owner.insert(def, idx);
        }
    }

    let mut indegree: Vec<usize> = vec![0; n];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, (_, _, op)) in items.iter().enumerate() {
        let mut producers: FxHashSet<usize> = FxHashSet::default();
        op.for_each_use(|operand| {
            if let Some(&producer) = def_owner.get(&operand)
                && producer != idx
            {
                producers.insert(producer);
            }
        });
        for producer in producers {
            if let Some(list) = consumers.get_mut(producer) {
                list.push(idx);
            }
            if let Some(deg) = indegree.get_mut(idx) {
                *deg = deg.saturating_add(1);
            }
        }
    }

    // Min-heap on original position keeps the order deterministic and keeps
    // independent instructions in their original relative order.
    let mut ready: BinaryHeap<Reverse<((usize, usize), usize)>> = (0..n)
        .filter(|idx| indegree.get(*idx).copied().unwrap_or(0) == 0)
        .map(|idx| Reverse((pos_of(idx), idx)))
        .collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(Reverse((_, idx))) = ready.pop() {
        order.push(idx);
        let Some(consumer_list) = consumers.get(idx) else {
            continue;
        };
        for &consumer in consumer_list {
            if let Some(deg) = indegree.get_mut(consumer) {
                *deg = deg.saturating_sub(1);
                if *deg == 0 {
                    ready.push(Reverse((pos_of(consumer), consumer)));
                }
            }
        }
    }
    if order.len() < n {
        let mut placed = vec![false; n];
        for &idx in &order {
            if let Some(slot) = placed.get_mut(idx) {
                *slot = true;
            }
        }
        let mut rest: Vec<usize> = (0..n)
            .filter(|idx| !placed.get(*idx).copied().unwrap_or(false))
            .collect();
        rest.sort_by_key(|idx| pos_of(*idx));
        order.extend(rest);
    }

    let mut taken: Vec<Option<(usize, usize, SsaOp<T>)>> =
        mem::take(items).into_iter().map(Some).collect();
    *items = order
        .into_iter()
        .filter_map(|idx| taken.get_mut(idx).and_then(Option::take))
        .collect();
}

/// Tests whether a single instruction is loop-invariant given what is already
/// known.
///
/// An instruction is invariant when none of its operands is a header phi result
/// (those change every iteration) and every operand defined inside the loop has
/// already been proven invariant. The header-phi check is separate from
/// `inside_defs` because a header phi's result must never be treated as
/// invariant even transitively, whereas an ordinary in-loop def can become
/// invariant once its own producer does.
///
/// # Arguments
///
/// * `instr` - The instruction to classify.
/// * `inside_defs` - Variables defined inside the loop body.
/// * `invariant_defs` - Variables already proven invariant this fixpoint.
/// * `header_phi_defs` - Results of the loop header's phi nodes.
///
/// # Returns
///
/// `true` if `instr` is loop-invariant.
fn is_instruction_invariant<T: Target>(
    instr: &SsaInstruction<T>,
    inside_defs: &VarSet,
    invariant_defs: &VarSet,
    header_phi_defs: &VarSet,
) -> bool {
    let mut invariant = true;
    instr.op().for_each_use(|operand| {
        if header_phi_defs.contains(operand) {
            invariant = false;
        }
        // An operand defined inside the loop and not yet proven invariant breaks
        // invariance. `inside_defs` is computed once by the caller in
        // O(loop-body), where the complementary "defined outside" set would
        // need a per-loop O(all-variables) scan.
        if inside_defs.contains(operand) && !invariant_defs.contains(operand) {
            invariant = false;
        }
    });
    invariant
}

/// Pass that hoists loop-invariant computations out of loop bodies.
///
/// Processes loops
/// innermost-first, conservatively guarding against hoisting that would
/// corrupt phi nodes or create unstable trampoline blocks.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - adds instructions to the
/// preheader and may modify phi predecessor references.
#[derive(Debug, Default, Clone, Copy)]
pub struct LicmPass;

impl LicmPass {
    /// Creates a new LICM pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for LicmPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "licm"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Hoist loop-invariant computations out of loops"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Run Loop Invariant Code Motion on `ssa`.
    ///
    /// Hoists pure, invariant instructions from loop bodies into their
    /// preheaders. Processes loops innermost-first.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any instruction was hoisted.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let forest = LoopAnalyzer::new(ssa).analyze();
        if forest.is_empty() {
            return Ok(false);
        }

        // One edit scope for the whole forest, not one per loop.
        //
        // Closing a scope costs a whole-function snapshot (the rollback buffer),
        // an SSA repair, and a verification - all O(function). Paying that per loop
        // would make the pass O(loops x function): a function with thousands of
        // natural loops would take thousands of full-function walks per run.
        //
        // Per-loop *ordering* is preserved: `HoistPlanner::plan` runs against
        // `editor.function()`, so each loop sees the IR as the loops before it
        // left it, exactly as if each had its own scope. The cost is on the
        // failure path - a rejected result discards the whole forest's hoists
        // rather than one loop's. Verifier rejections here are rare, and the
        // normalization fixpoint re-runs the pass, so the lost work is recovered on
        // the next iteration.
        let mut total_hoisted: usize = 0;
        let result = ssa.edit(
            SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| {
                for loop_info in forest.by_depth_descending() {
                    let planner = HoistPlanner {
                        ssa: editor.function(),
                        loop_info,
                    };
                    let Some(plan) = planner.plan() else {
                        continue;
                    };

                    // Destinations moved to a fresh name, so the instructions
                    // hoisted after them read the moved value, not the copy left
                    // in the loop.
                    let mut moved: BTreeMap<SsaVarId, SsaVarId> = BTreeMap::new();

                    for (i, (block_idx, instr_idx, _)) in plan.to_hoist.iter().enumerate() {
                        let position = plan.insert_base.saturating_add(i);

                        // Taken out and re-inserted rather than rebuilt from the
                        // operation: the hoisted instruction is the same
                        // instruction in a different block, so it keeps its
                        // identity, the host payload that names where it came from,
                        // and its resolved result type. The placeholder left behind
                        // keeps the plan's remaining indices valid.
                        let mut hoisted = editor.detach_instruction(*block_idx, *instr_idx)?;

                        // A definition that is a version of a storage location
                        // stays where the program writes the location: the
                        // computation moves under a fresh name, and a copy of it
                        // takes the definition's place. Moved outright, the
                        // location would be written in the preheader - where it
                        // may already be written, and a later reconstruction
                        // reads the last of those writes - instead of in the
                        // loop, where the program's order puts it.
                        let defs: Vec<SsaVarId> = hoisted.op().defs().collect();
                        if let [dest] = defs.as_slice()
                            && editor.function().rename_group(*dest) != u32::MAX
                        {
                            let fresh = editor.derive_variable_for_origin(
                                *dest,
                                VariableOrigin::Phi,
                                0,
                                DefSite::instruction(plan.preheader, position),
                                Derivation::Rename,
                            );
                            editor.replace_instruction_op(
                                *block_idx,
                                *instr_idx,
                                SsaOp::Copy {
                                    dest: *dest,
                                    src: fresh,
                                },
                            )?;
                            moved.insert(*dest, fresh);
                        }
                        if !moved.is_empty() {
                            let op = hoisted.op().remap_variables(|var| moved.get(&var).copied());
                            hoisted.set_op(op);
                        }
                        editor.insert_instruction(plan.preheader, position, hoisted)?;
                        total_hoisted = total_hoisted.saturating_add(1);
                    }
                }

                Ok(())
            },
        );
        if result.is_err() {
            total_hoisted = 0;
        }

        if total_hoisted > 0 {
            env.record(EventKind::InstructionRemoved)
                .location(0)
                .message(format!(
                    "LICM: hoisted {total_hoisted} loop-invariant instructions"
                ));
        }
        Ok(total_hoisted > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType, assert_mock_valid_full, run_mock_pass_boundary},
    };

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    fn local_at(
        ssa: &mut SsaFunction<MockTarget>,
        idx: u16,
        block: usize,
        instr: usize,
    ) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::instruction(block, instr),
            MockType::I32,
        )
    }

    /// The availability filter and the trampoline filter can each invalidate the
    /// other's conclusion, so both must run to a joint fixpoint.
    ///
    /// `B2` holds the producer as its only non-terminator instruction, so
    /// hoisting it turns `B2` into a trampoline into `B3` - which has phis, so
    /// the trampoline filter drops `B2`'s entry. `B3`'s consumer was retained
    /// only because that producer was being hoisted with it. Without re-running
    /// availability, the consumer is hoisted into the preheader while its
    /// operand stays behind in the loop: a use above its own definition.
    ///
    /// `VerifyLevel::Standard` cannot catch that (`check_defined_before_use`
    /// tests existence, not dominance), so this asserts at `Full`.
    #[test]
    fn a_hoist_dropped_by_the_trampoline_filter_also_drops_its_dependents() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 6);
        let base = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);
        let cond = local_at(&mut ssa, 2, 0, 2);
        let producer = local_at(&mut ssa, 3, 2, 0);
        let consumer = local_at(&mut ssa, 4, 3, 0);
        let merged =
            ssa.create_variable(VariableOrigin::Local(5), 0, DefSite::phi(3), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        for (dest, value) in [(base, 10), (one, 1), (cond, 1)] {
            b0.add_instruction(instr(SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            }));
        }
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        // Header: loops back from B3, exits to B4.
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 4,
        }));
        ssa.add_block(b1);

        // The producer is B2's only non-terminator instruction.
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: producer,
            left: base,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b2);

        // B3 has a phi, so emptying B2 is penalised by the trampoline filter.
        let mut b3 = SsaBlock::new(3);
        let mut phi = PhiNode::new(merged, VariableOrigin::Local(5));
        phi.add_operand(PhiOperand::new(base, 2));
        b3.add_phi(phi);
        // Uses the producer and a value from outside the loop - so it is
        // invariant, and hoistable *only* because the producer is too. It must
        // not use `merged`, which is loop-variant and would make it unhoistable.
        b3.add_instruction(instr(SsaOp::Add {
            dest: consumer,
            left: producer,
            right: one,
            flags: None,
        }));
        b3.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        // Returns `cond`, not `consumer`: B3 is the latch and does not dominate
        // the exit, so reading `consumer` here would be invalid input rather
        // than a pass bug.
        b4.add_instruction(instr(SsaOp::returns(Some(cond))));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            LicmPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );

        assert_mock_valid_full(&ssa, "LICM joint filter fixpoint");
    }

    /// Builds: preheader (B0) → header (B1: branch cond → body or exit)
    /// Body (B2) contains invariant expression, latches back to header
    fn build_simple_loop() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 5);
        let base = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);
        let cond = local_at(&mut ssa, 2, 0, 2);
        let invariant = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(2, 0),
            MockType::I32,
        );
        let iv = local_at(&mut ssa, 4, 1, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: base,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: iv,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: invariant,
            left: base,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);

        ssa.recompute_uses();
        ssa
    }

    /// A pure loop-invariant expression is hoisted into the preheader.
    #[test]
    fn licm_hoists_invariant_to_preheader() {
        let mut ssa = build_simple_loop();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "simple LICM hoist", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "LICM should hoist invariant expression");
        assert!(log.has(EventKind::InstructionRemoved));
    }

    /// What a call leaves in its registers is defined by that call, each
    /// time it runs: a clobber in a loop's body reads no operand the loop
    /// varies, and still stays where its call is.
    #[test]
    fn a_calls_clobber_is_not_hoisted() {
        let mut ssa = build_simple_loop();
        let clobbered = SsaVarId::from_index(3);
        if let Some(body) = ssa.block_mut(2)
            && let Some(first) = body.instructions_mut().first_mut()
        {
            *first = instr(SsaOp::CallClobber {
                outputs: vec![clobbered],
                machine: None,
            });
        }
        ssa.assign_instr_ids();
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let _ = run_mock_pass_boundary(&mut ssa, "clobber LICM", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(
            ssa.block(2).is_some_and(|body| body
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.op(), SsaOp::CallClobber { .. }))),
            "the clobber stays in the loop: {ssa}"
        );
    }

    /// A hoisted value that is a version of a local keeps its definition in
    /// the loop. The computation moves to the preheader under a fresh name,
    /// and the loop writes the local from it where the program does - moved
    /// outright, the local would be written in the preheader instead, next to
    /// whatever else writes it there, and a reconstruction would take the
    /// last of those writes for the value the loop is entered with.
    #[test]
    fn a_hoisted_local_version_stays_written_in_the_loop() {
        let mut ssa = build_simple_loop();
        let invariant = ssa
            .variables()
            .iter()
            .find(|var| var.origin() == VariableOrigin::Local(3))
            .map(|var| var.id())
            .expect("the fixture defines local 3");
        ssa.set_rename_group(invariant, 3);
        let log: EventLog<MockTarget> = EventLog::new();
        assert!(run_mock_pass_boundary(
            &mut ssa,
            "LICM of a local version",
            |ssa| LicmPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ) == Ok(true)
        ));

        let hoisted = ssa.block(0).and_then(|block| {
            block
                .instructions()
                .iter()
                .find_map(|instr| match instr.op() {
                    SsaOp::Add { dest, .. } => Some(*dest),
                    _ => None,
                })
        });
        let hoisted = hoisted.expect("the addition moved to the preheader");
        assert_ne!(hoisted, invariant, "under a fresh name");
        assert_eq!(
            ssa.rename_group(hoisted),
            u32::MAX,
            "which is no location's"
        );
        let written_in_loop = ssa.block(2).is_some_and(|block| {
            block.instructions().iter().any(|instr| {
                matches!(instr.op(), SsaOp::Copy { dest, src } if *dest == invariant && *src == hoisted)
            })
        });
        assert!(
            written_in_loop,
            "and the loop still writes local 3: {ssa:?}"
        );
    }

    /// A constant written to a local in a loop stays there. It computes
    /// nothing, and hoisted it would leave a copy of itself that constant
    /// propagation folds back into the same constant, for the next run to
    /// hoist again - a cycle that ran a NetReactor method's pipeline ten rounds
    /// longer than it converged in.
    #[test]
    fn a_constant_written_to_a_local_is_not_hoisted() {
        let mut ssa = build_simple_loop();
        let written = ssa
            .variables()
            .iter()
            .find(|var| var.origin() == VariableOrigin::Local(4))
            .map(|var| var.id())
            .expect("the fixture writes local 4 in the loop");
        ssa.set_rename_group(written, 4);
        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "LICM of a local constant", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let still_in_loop = ssa.block(1).is_some_and(|block| {
            block
                .instructions()
                .iter()
                .any(|instr| matches!(instr.op(), SsaOp::Const { dest, .. } if *dest == written))
        });
        assert!(still_in_loop, "{ssa:?}");
    }

    /// The copy a hoist leaves in the loop is not itself hoisted: it computes
    /// nothing, and moving it would leave the same copy behind, so a second run
    /// has nothing to do.
    #[test]
    fn the_copy_a_hoist_leaves_is_not_hoisted_again() {
        let mut ssa = build_simple_loop();
        let invariant = ssa
            .variables()
            .iter()
            .find(|var| var.origin() == VariableOrigin::Local(3))
            .map(|var| var.id())
            .expect("the fixture defines local 3");
        ssa.set_rename_group(invariant, 3);
        let log: EventLog<MockTarget> = EventLog::new();
        assert!(run_mock_pass_boundary(&mut ssa, "first LICM run", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        }));
        assert!(
            !run_mock_pass_boundary(&mut ssa, "second LICM run", |ssa| LicmPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ) == Ok(true)),
            "a second run finds nothing to hoist: {ssa:?}"
        );
    }

    /// A function with no loops hoists nothing and reports no change.
    #[test]
    fn no_loops_nothing_hoisted() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::returns(Some(v0))));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "no-loop LICM", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "no loops should mean nothing to hoist");
    }

    /// An instruction reading a header phi (loop-variant) result is not hoisted.
    #[test]
    fn invariant_with_loop_variant_operand_not_hoisted() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 5);
        let base = local_at(&mut ssa, 0, 0, 0);
        let cond = local_at(&mut ssa, 1, 0, 1);
        let iv = local_at(&mut ssa, 2, 0, 2);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(1), MockType::I32);
        let result = local_at(&mut ssa, 4, 2, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: base,
            value: ConstValue::I32(5),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: iv,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        // phi_var is loop-variant (defined by phi in header, fed by back-edge)
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(iv, 0));
        phi.add_operand(PhiOperand::new(result, 2));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        // result depends on phi_var which is loop-variant -> not invariant
        b2.add_instruction(instr(SsaOp::Add {
            dest: result,
            left: base,
            right: phi_var,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "variant-operand LICM", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        // phi_var is defined by header phi which is part of loop body, so Add should NOT be hoisted
        assert!(
            !changed,
            "loop-variant operand (header phi) should prevent hoisting"
        );
    }

    /// An empty function is a no-op.
    #[test]
    fn empty_function_is_noop() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty LICM", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed);
    }

    /// A loop with no preheader (self-loop) hoists nothing.
    #[test]
    fn loop_without_preheader_not_hoisted() {
        // The header is entered from two outside blocks, so no single block
        // dominates the loop entry to hoist into.
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let cond = local_at(&mut ssa, 0, 0, 0);
        let v0 = local_at(&mut ssa, 1, 2, 0);
        let sum = local_at(&mut ssa, 2, 2, 1);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        b2.add_instruction(instr(SsaOp::Add {
            dest: sum,
            left: v0,
            right: v0,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "no-preheader LICM", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "loop without preheader should not hoist");
    }

    /// Multiple invariants in one loop body are all hoisted in a single run.
    #[test]
    fn hoist_multiple_invariants() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 5);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let cond = local_at(&mut ssa, 2, 0, 2);
        let inv1 = local_at(&mut ssa, 3, 2, 0);
        let inv2 = local_at(&mut ssa, 4, 2, 1);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(3),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: b,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: inv1,
            left: a,
            right: b,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Mul {
            dest: inv2,
            left: a,
            right: b,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "multiple-invariant LICM", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "multiple invariants should be hoisted");
    }

    /// Two independent loops must both hoist in a single `run`.
    ///
    /// The pass processes the whole loop forest inside one edit scope, so this
    /// is the case that would regress if a later loop were planned against a
    /// stale view of the function, or if the shared scope dropped work from all
    /// but the first loop. Each loop's invariant must land in *its own*
    /// preheader.
    #[test]
    fn independent_loops_all_hoist_in_one_run() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 6);
        let a = local_at(&mut ssa, 0, 0, 0);
        let b = local_at(&mut ssa, 1, 0, 1);
        let cond = local_at(&mut ssa, 2, 0, 2);
        let inv1 = local_at(&mut ssa, 3, 2, 0);
        let inv2 = local_at(&mut ssa, 4, 5, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: a,
            value: ConstValue::I32(3),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: b,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        // Loop 1: header 1, body 2, preheader 0.
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: inv1,
            left: a,
            right: b,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        // Loop 2: header 4, body 5, preheader 3.
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Jump { target: 4 }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 5,
            false_target: 6,
        }));
        ssa.add_block(b4);

        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::Mul {
            dest: inv2,
            left: a,
            right: b,
            flags: None,
        }));
        b5.add_instruction(instr(SsaOp::Jump { target: 4 }));
        ssa.add_block(b5);

        let mut b6 = SsaBlock::new(6);
        b6.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b6);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "two-loop LICM", |ssa| {
            LicmPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "both loops have a hoistable invariant");

        let has_add = |block: usize| {
            ssa.block(block).is_some_and(|b| {
                b.instructions()
                    .iter()
                    .any(|i| matches!(i.op(), SsaOp::Add { .. }))
            })
        };
        let has_mul = |block: usize| {
            ssa.block(block).is_some_and(|b| {
                b.instructions()
                    .iter()
                    .any(|i| matches!(i.op(), SsaOp::Mul { .. }))
            })
        };
        assert!(has_add(0), "loop 1's invariant should reach preheader 0");
        assert!(has_mul(3), "loop 2's invariant should reach preheader 3");
    }
}
