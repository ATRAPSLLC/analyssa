//! Loop Invariant Code Motion (LICM) pass — moves loop-invariant
//! computations into the loop preheader.
//!
//! # Algorithm
//!
//! 1. Analyze the loop structure using [`LoopAnalyzer`].
//! 2. Process loops innermost-first so hoisted values are available to
//!    outer loops in subsequent passes.
//! 3. For each loop with a valid preheader:
//!    a. Find invariant instructions — those whose operands are all
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
//!    e. If a source block is now a trampoline, redirect successor phi
//!    operands from the source block to the preheader.
//!
//! # Conservative Guards
//!
//! - Requires a loop preheader that is a CFG predecessor of the header.
//! - Skips loops whose header terminator is a `Switch` (CFF dispatcher
//!   pattern — hoisting corrupts SSA rebuild).
//! - Rejects instructions whose result transitively feeds a phi on an
//!   intra-loop edge (would erase per-edge value distinctions).
//! - Avoids creating trampoline blocks whose successor has phis (phi
//!   stability across subsequent block merging).

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
};

use crate::{
    analysis::{loop_analyzer::LoopAnalyzer, loops::LoopInfo},
    bitset::BitSet,
    events::{EventKind, EventListener},
    ir::{
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        instruction::SsaInstruction,
        ops::SsaOp,
        variable::SsaVarId,
        varstore::VarSet,
    },
    target::Target,
};

/// Run Loop Invariant Code Motion on `ssa`.
///
/// Hoists pure, invariant instructions from loop bodies into their
/// preheaders. Processes loops innermost-first.
///
/// # Arguments
///
/// * `ssa` — The SSA function to optimize in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::InstructionRemoved`] events.
///
/// # Returns
///
/// `true` if any instruction was hoisted.
pub fn run<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let forest = LoopAnalyzer::new(ssa).analyze();
    if forest.is_empty() {
        return false;
    }

    // One edit scope for the whole forest, not one per loop.
    //
    // Closing a scope costs a whole-function snapshot (the rollback buffer),
    // an SSA repair, and a verification — all O(function). Paying that per loop
    // made the pass O(loops x function): on a 27,384-block function from a real
    // binary with 3,633 natural loops, that was thousands of full-function
    // walks per run, and the pass accounted for the overwhelming majority of
    // the binary's lift time.
    //
    // Per-loop *ordering* is preserved: `plan_loop_hoist` runs against
    // `editor.function()`, so each loop still sees the IR as the previous loops
    // left it, exactly as when each had its own scope. What changes is the
    // failure path — a rejected result now discards the whole forest's hoists
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
                let Some(plan) = plan_loop_hoist(editor.function(), loop_info) else {
                    continue;
                };

                let mut hoisted_from = BitSet::new(editor.function().block_count());

                for (i, (block_idx, instr_idx, op)) in plan.to_hoist.iter().enumerate() {
                    hoisted_from.insert(*block_idx);

                    editor.insert_instruction(
                        plan.preheader,
                        plan.insert_base.saturating_add(i),
                        SsaInstruction::synthetic(op.clone()),
                    )?;
                    editor.nop_instruction(*block_idx, *instr_idx)?;
                    total_hoisted = total_hoisted.saturating_add(1);
                }

                // If a source block is now a trampoline, redirect successor phis
                // from the source block to the preheader where the hoisted defs live.
                for source_block in hoisted_from.iter() {
                    let is_trampoline = editor.function().block(source_block).is_some_and(|b| {
                        b.instructions()
                            .iter()
                            .all(|i| i.is_terminator() || matches!(i.op(), SsaOp::Nop))
                    });
                    if !is_trampoline {
                        continue;
                    }
                    let successors: Vec<usize> = editor
                        .function()
                        .block(source_block)
                        .map(|b| {
                            b.instructions()
                                .last()
                                .map(|i| i.op().successors())
                                .unwrap_or_default()
                        })
                        .unwrap_or_default();
                    for succ in successors {
                        editor.replace_phi_predecessor(succ, source_block, plan.preheader)?;
                    }
                }
            }

            Ok(())
        },
    );
    if result.is_err() {
        total_hoisted = 0;
    }

    if total_hoisted > 0 {
        let event = crate::events::Event {
            kind: EventKind::InstructionRemoved,
            method: Some(method.clone()),
            location: Some(0),
            message: format!("LICM: hoisted {total_hoisted} loop-invariant instructions"),
            pass: None,
        };
        events.push(event);
    }

    total_hoisted > 0
}

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

/// Decides what `loop_info` can contribute, without mutating anything.
///
/// Split out of [`run`] so the whole forest can be processed inside a single
/// edit scope: the analysis needs to see the IR as previous loops left it, so it
/// runs against `ssa` — the editor's current function — rather than a snapshot
/// taken before the pass began.
///
/// Returns `None` when the loop contributes nothing.
fn plan_loop_hoist<T: Target>(ssa: &SsaFunction<T>, loop_info: &LoopInfo) -> Option<LoopHoist<T>> {
    let preheader = loop_info.preheader?;

    let header_idx = loop_info.header.index();
    let preheader_is_pred = ssa
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

    let header_has_switch = ssa
        .block(header_idx)
        .and_then(|b| b.terminator_op())
        .is_some_and(|op| matches!(op, SsaOp::Switch { .. }));
    if header_has_switch {
        return None;
    }

    let inside_defs = loop_inside_defs(ssa, loop_info);
    let invariants = find_loop_invariants(ssa, loop_info, &inside_defs);
    if invariants.is_empty() {
        return None;
    }

    let phi_back_edge_operands = phi_back_edge_operands(ssa, loop_info);
    let back_edge_tainted = back_edge_tainted_vars(ssa, loop_info, &phi_back_edge_operands);
    let mut hoistable: Vec<_> = invariants
        .into_iter()
        .filter(|(block_idx, instr_idx)| {
            can_hoist(ssa, loop_info, &back_edge_tainted, *block_idx, *instr_idx)
        })
        .collect();

    let preheader_idx = preheader.index();
    let insert_base = if let Some(preheader_block) = ssa.block(preheader_idx) {
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
    let mut preheader_after = VarSet::new(ssa.var_id_bound());
    if let Some(preheader_block) = ssa.block(preheader_idx) {
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
    // trampoline filter afterwards — as this did — produces exactly that, and
    // nothing downstream catches it: the edit stays at
    // `SsaEditScope::InstructionsOnly`, so `repair_ssa` runs rather than
    // `rebuild_ssa`, and the boundary verifies at `VerifyLevel::Standard`, whose
    // `check_defined_before_use` tests existence rather than dominance.
    loop {
        let joint_before = hoistable.len();

        loop {
            let before = hoistable.len();
            // Defs produced by the instructions still slated for hoisting. An
            // invariant whose operands are all either defined outside the loop
            // OR by another hoisted instruction is itself hoistable — the whole
            // dependency chain moves together in one pass (ordered
            // topologically below). Recomputed each round because dropping an
            // instruction can make its dependents unhoistable.
            let hoistable_defs: HashSet<SsaVarId> = hoistable
                .iter()
                .filter_map(|(b, i)| ssa.block(*b).and_then(|blk| blk.instruction(*i)))
                .flat_map(|instr| instr.op().defs())
                .collect();
            hoistable.retain(|(block_idx, instr_idx)| {
                let Some(block) = ssa.block(*block_idx) else {
                    return false;
                };
                let Some(instr) = block.instruction(*instr_idx) else {
                    return false;
                };
                let def_used_before_insert = instr.op().defs().any(|def| {
                    ssa.variable(def).is_some_and(|var| {
                        var.uses().iter().any(|site| {
                            site.block == preheader_idx
                                && !site.is_phi_operand
                                && site.instruction < insert_base
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
            let mut hoist_count_per_block: HashMap<usize, usize> = HashMap::new();
            for (block_idx, _) in &hoistable {
                let entry = hoist_count_per_block.entry(*block_idx).or_insert(0);
                *entry = entry.saturating_add(1);
            }
            let mut trampoline_blocks = BitSet::new(ssa.block_count());
            for (&block_idx, &hoist_count) in &hoist_count_per_block {
                if let Some(block) = ssa.block(block_idx) {
                    let non_term = block
                        .instructions()
                        .iter()
                        .filter(|i| !i.is_terminator() && !matches!(i.op(), SsaOp::Nop))
                        .count();
                    if hoist_count >= non_term
                        && let Some(term) = block.terminator_op()
                    {
                        term.for_each_successor(|succ| {
                            if let Some(succ_block) = ssa.block(succ)
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
        if let Some(block) = ssa.block(*block_idx)
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

/// Reorders `items` so every producer precedes its consumers.
///
/// All hoisted instructions move into the same preheader, so a use must never
/// be inserted before its definition. Original block/instruction position is not
/// a valid order across blocks (a def can live in a higher-indexed block than
/// its use), so this performs a deterministic topological sort — Kahn's
/// algorithm with ties broken by original `(block, instruction)` position.
/// Loop-invariant instructions cannot form a data-dependency cycle; if one
/// somehow survives, the remainder is appended in positional order as a
/// defensive fallback rather than dropped.
fn topological_hoist_order<T: Target>(items: &mut Vec<(usize, usize, SsaOp<T>)>) {
    let n = items.len();
    if n < 2 {
        return;
    }

    let positions: Vec<(usize, usize)> = items.iter().map(|(b, i, _)| (*b, *i)).collect();
    let pos_of = |idx: usize| positions.get(idx).copied().unwrap_or_default();

    // Variable -> index of the hoisted instruction that defines it.
    let mut def_owner: HashMap<SsaVarId, usize> = HashMap::new();
    for (idx, (_, _, op)) in items.iter().enumerate() {
        for def in op.defs() {
            def_owner.insert(def, idx);
        }
    }

    let mut indegree: Vec<usize> = vec![0; n];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, (_, _, op)) in items.iter().enumerate() {
        let mut producers: HashSet<usize> = HashSet::new();
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

    // Min-heap on original position keeps the order deterministic and close to
    // the previous positional behaviour for independent instructions.
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
        std::mem::take(items).into_iter().map(Some).collect();
    *items = order
        .into_iter()
        .filter_map(|idx| taken.get_mut(idx).and_then(Option::take))
        .collect();
}

/// Returns the variables whose definition site lies inside the loop body —
/// instruction results and phi results in body blocks.
///
/// This is the complement of the former per-loop "outside defs" set. Scanning
/// only the loop body makes it O(loop-body); the previous code derived the same
/// information by scanning every variable in the whole function once per loop
/// (O(loops × variables)).
fn loop_inside_defs<T: Target>(ssa: &SsaFunction<T>, loop_info: &LoopInfo) -> VarSet {
    let mut inside = VarSet::new(ssa.var_id_bound());
    for block_idx in loop_info.body.iter() {
        if let Some(block) = ssa.block(block_idx) {
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

fn find_loop_invariants<T: Target>(
    ssa: &SsaFunction<T>,
    loop_info: &LoopInfo,
    inside_defs: &VarSet,
) -> Vec<(usize, usize)> {
    let mut invariants: HashSet<(usize, usize)> = HashSet::new();
    let mut invariant_defs = VarSet::new(ssa.var_id_bound());

    let mut header_phi_defs = VarSet::new(ssa.var_id_bound());
    if let Some(header_block) = ssa.block(loop_info.header.index()) {
        for phi in header_block.phi_nodes() {
            header_phi_defs.insert(phi.result());
        }
    }

    // Worklist over loop-body instructions: seed with all of them, and when an
    // instruction becomes invariant, only re-examine the instructions that use
    // its definitions (via a scoped def→users map). A
    // `while changed { rescan whole loop body }` fixpoint would be O(loop^2).
    let mut users: HashMap<SsaVarId, Vec<(usize, usize)>> = HashMap::new();
    let mut worklist: VecDeque<(usize, usize)> = VecDeque::new();
    let mut queued: HashSet<(usize, usize)> = HashSet::new();
    for block_idx in loop_info.body.iter() {
        if let Some(block) = ssa.block(block_idx) {
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
        let Some(instr) = ssa
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
        // invariance. `inside_defs` is the complement of the former `outside_defs`
        // set, computed once by the caller in O(loop-body) instead of a per-loop
        // O(all-variables) scan.
        if inside_defs.contains(operand) && !invariant_defs.contains(operand) {
            invariant = false;
        }
    });
    invariant
}

fn can_hoist<T: Target>(
    ssa: &SsaFunction<T>,
    loop_info: &LoopInfo,
    back_edge_tainted: &VarSet,
    block_idx: usize,
    instr_idx: usize,
) -> bool {
    let Some(block) = ssa.block(block_idx) else {
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
    if loop_info.preheader.is_none() {
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
/// [`can_hoist`] must reject any instruction whose result feeds such a phi.
/// Answering that per candidate with an independent forward def-use walk is
/// `O(candidates × loop)`; seeding a worklist from the back-edge phi operands
/// and propagating backward through in-loop defining instructions answers it
/// for *every* value in `O(loop)` once. A value is tainted when it is itself a
/// back-edge operand, or when it is an operand of an in-loop instruction whose
/// result is tainted.
fn back_edge_tainted_vars<T: Target>(
    ssa: &SsaFunction<T>,
    loop_info: &LoopInfo,
    phi_back_edge_operands: &VarSet,
) -> VarSet {
    let mut tainted = phi_back_edge_operands.clone();
    let mut worklist: VecDeque<SsaVarId> = phi_back_edge_operands.iter().collect();
    while let Some(value) = worklist.pop_front() {
        let Some(var) = ssa.variable(value) else {
            continue;
        };
        let site = var.def_site();
        if !loop_info.body.contains(site.block) {
            continue;
        }
        let Some(instr_idx) = site.instruction else {
            continue;
        };
        let Some(instr) = ssa
            .block(site.block)
            .and_then(|block| block.instruction(instr_idx))
        else {
            continue;
        };
        instr.op().for_each_use(|operand| {
            if tainted.insert(operand) {
                worklist.push_back(operand);
            }
        });
    }
    tainted
}

fn phi_back_edge_operands<T: Target>(ssa: &SsaFunction<T>, loop_info: &LoopInfo) -> VarSet {
    let mut operands = VarSet::new(ssa.var_id_bound());
    for phi_block_idx in loop_info.body.iter() {
        let Some(phi_block) = ssa.block(phi_block_idx) else {
            continue;
        };
        for phi in phi_block.phi_nodes() {
            for operand in phi.operands() {
                // Raw phi-operand predecessor: may name a block that does not
                // exist, which is trivially not in the loop body.
                if loop_info.body.contains_checked(operand.predecessor()) {
                    operands.insert(operand.value());
                }
            }
        }
    }
    operands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockTarget, MockType, run_mock_pass_boundary},
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
    /// hoisting it turns `B2` into a trampoline into `B3` — which has phis, so
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
        // Uses the producer and a value from outside the loop — so it is
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
        b4.add_instruction(instr(SsaOp::Return { value: Some(cond) }));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run(&mut ssa, &0u32, &log);

        crate::testing::assert_mock_valid_full(&ssa, "LICM joint filter fixpoint");
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
        b3.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b3);

        ssa.recompute_uses();
        ssa
    }

    #[test]
    fn licm_hoists_invariant_to_preheader() {
        let mut ssa = build_simple_loop();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "simple LICM hoist", |ssa| run(ssa, &method, &log));
        assert!(changed, "LICM should hoist invariant expression");
        assert!(log.has(EventKind::InstructionRemoved));
    }

    #[test]
    fn no_loops_nothing_hoisted() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        b0.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "no-loop LICM", |ssa| run(ssa, &method, &log));
        assert!(!changed, "no loops should mean nothing to hoist");
    }

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
        b3.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "variant-operand LICM", |ssa| {
            run(ssa, &method, &log)
        });
        // phi_var is defined by header phi which is part of loop body, so Add should NOT be hoisted
        assert!(
            !changed,
            "loop-variant operand (header phi) should prevent hoisting"
        );
    }

    #[test]
    fn empty_function_is_noop() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty LICM", |ssa| run(ssa, &method, &log));
        assert!(!changed);
    }

    #[test]
    fn loop_without_preheader_not_hoisted() {
        // A self-loop where the only block branches to itself — no preheader
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let cond = local_at(&mut ssa, 1, 0, 1);
        let sum = local_at(&mut ssa, 2, 0, 2);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Add {
            dest: sum,
            left: v0,
            right: v0,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 0,
            false_target: 1,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "no-preheader LICM", |ssa| run(ssa, &method, &log));
        assert!(!changed, "loop without preheader should not hoist");
    }

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
        b3.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "multiple-invariant LICM", |ssa| {
            run(ssa, &method, &log)
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
        b6.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b6);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "two-loop LICM", |ssa| run(ssa, &method, &log));
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
