//! Loop canonicalization pass — transforms loops into well-formed
//! canonical form with single preheaders and single latches.
//!
//! # Transformations
//!
//! 1. **Preheader insertion**: When a loop header has multiple non-loop
//!    predecessors, a fresh preheader block is created. All non-loop
//!    predecessors are redirected to it, and its terminator is a `Jump`
//!    to the header. Phi nodes in the header are updated: multiple non-loop
//!    operands are merged through a new phi in the preheader; a single
//!    non-loop operand is forwarded directly.
//!
//! 2. **Latch unification**: When a loop has multiple back-edges (latches),
//!    a single unified latch block is created. All original latches are
//!    redirected to it, and its terminator is a `Jump` to the header.
//!    Phi nodes in the header are rewritten: multiple latch operands are
//!    merged through a new phi in the unified latch.
//!
//! # Impact
//!
//! Canonical loop structure is required by subsequent analyses:
//! - LICM needs a single preheader to insert hoisted code.
//! - Induction variable detection assumes a single latch.
//! - Loop interchange and unrolling benefit from canonical form.
//!
//! # Algorithm
//!
//! Repeatedly analyzes loop structure via [`SsaFunction::analyze_loops`]. Each
//! pass canonicalizes *every* loop the forest reports, innermost-first, applying
//! at most one transformation per loop (preheader insertion takes priority over
//! latch unification) so phi management stays simple; a loop needing both is
//! finished on the next pass. Because every transformation is local to its own
//! loop, the forest is re-analyzed once per pass rather than once per individual
//! transformation. After all loops are canonical, [`SsaFunction::canonicalize`]
//! is called.

use std::collections::HashMap;

use crate::{
    analysis::{cfg::SsaCfg, loop_analyzer::SsaLoopAnalysis, loops::LoopInfo},
    events::{EventKind, EventListener},
    graph::NodeId,
    ir::{
        block::SsaBlock,
        function::{SsaEditOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::Target,
};

/// Run loop canonicalization on `ssa`.
///
/// Inserts missing preheaders and unifies multiple latches so every loop
/// has well-formed canonical structure. After convergence,
/// [`SsaFunction::canonicalize`] is called on the function.
///
/// # Arguments
///
/// * `ssa` — The SSA function to canonicalize in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::ControlFlowRestructured`] events.
///
/// # Returns
///
/// `true` if any preheader was inserted or any latch unified.
pub fn run<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    if ssa.block_count() < 2 {
        return false;
    }

    let modified = canonicalize_loops(ssa, method, events);
    if modified > 0 {
        ssa.canonicalize();
        true
    } else {
        false
    }
}

/// Cap on loop-canonicalization rounds.
///
/// Each round restructures the CFG — inserting preheaders, splitting latches —
/// and then re-analyses the loop forest, so unlike the monotone worklists in
/// `licm` and `predicates` there is no measure that provably decreases. A round
/// that undoes a previous round's change would spin forever. Progress still
/// terminates the loop normally; this bound only stops pathological input from
/// hanging the pass.
const MAX_CANONICALIZATION_ROUNDS: usize = 16;

fn canonicalize_loops<T, L>(ssa: &mut SsaFunction<T>, method: &T::MethodRef, events: &L) -> usize
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut total_modified: usize = 0;

    for _ in 0..MAX_CANONICALIZATION_ROUNDS {
        let forest = ssa.analyze_loops();
        if forest.is_empty() {
            break;
        }

        let mut modified_this_iteration: usize = 0;

        // One predecessor relation per round, shared by every loop. Deriving it
        // per loop would be O(blocks) each, so a function with many loops would
        // pay O(loops x blocks).
        let predecessors = SsaCfg::from_ssa(ssa).to_predecessor_sets();

        // A loop whose header the runtime dispatches to cannot be canonicalised
        // by splicing a block in front of it. A preheader works by redirecting
        // the terminators that name the header, and no terminator names this
        // one -- so the preheader would sit between nothing and the header
        // while the runtime kept entering the header directly.
        //
        // Loops inside handlers are now detected, so this is reachable where it
        // previously was not. LICM needs no matching guard: `plan_loop_hoist`
        // already returns `None` unless the preheader's terminator names the
        // header, which is false by construction for such a loop.
        let exception_entries = ssa.exception_blocks();
        let is_runtime_entry: Vec<bool> = (0..ssa.block_count())
            .map(|block| exception_entries.is_runtime_entry(block))
            .collect();
        drop(exception_entries);

        for loop_info in forest.by_depth_descending() {
            if is_runtime_entry
                .get(loop_info.header.index())
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            if !loop_info.has_preheader() {
                let non_loop_preds = get_non_loop_predecessors(&predecessors, loop_info);
                if non_loop_preds.len() > 1 {
                    insert_preheader(ssa, loop_info, &non_loop_preds, method, events);
                    modified_this_iteration = modified_this_iteration.saturating_add(1);
                    continue;
                }
            }
            if !loop_info.has_single_latch() && loop_info.latches.len() > 1 {
                unify_latches(ssa, loop_info, method, events);
                modified_this_iteration = modified_this_iteration.saturating_add(1);
            }
        }

        total_modified = total_modified.saturating_add(modified_this_iteration);
        if modified_this_iteration == 0 {
            break;
        }
    }

    total_modified
}

/// Returns the loop header's predecessors that lie outside the loop body.
///
/// Reads the precomputed predecessor relation rather than re-deriving it by
/// scanning every block's terminator, which is O(preds) instead of O(blocks).
///
/// These are the blocks a preheader would have to be spliced in front of, so
/// they must be blocks whose terminator really does transfer to the header —
/// exactly what the relation records. A block the runtime dispatches to
/// carries no such transfer and must not be counted here, or the preheader is
/// inserted for an edge no terminator can be redirected along.
fn get_non_loop_predecessors(predecessors: &[Vec<usize>], loop_info: &LoopInfo) -> Vec<usize> {
    let header_idx = loop_info.header.index();
    predecessors
        .get(header_idx)
        .into_iter()
        .flatten()
        .copied()
        .filter(|block_idx| !loop_info.contains(NodeId::new(*block_idx)))
        .collect()
}

fn insert_preheader<T, L>(
    ssa: &mut SsaFunction<T>,
    loop_info: &LoopInfo,
    non_loop_preds: &[usize],
    method: &T::MethodRef,
    events: &L,
) where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let header_idx = loop_info.header.index();
    let preheader_idx = ssa.block_count();
    // Keyed by the header phi's *result*, not its origin. `VariableOrigin::Phi`
    // is a unit variant, so every phi that merges earlier phis — the ordinary
    // shape in lifted native code — carries the same origin. Keying on it made
    // one preheader phi stand for all of them and rewrote every header phi to
    // that single value, which is a dominance violation for all but one.
    let phi_info: Vec<(SsaVarId, VariableOrigin, Vec<PhiOperand>)> = ssa
        .block(header_idx)
        .map(|header| {
            header
                .phi_nodes()
                .iter()
                .filter_map(|phi| {
                    let non_loop_operands: Vec<_> = phi
                        .operands()
                        .iter()
                        .filter(|op| non_loop_preds.contains(&op.predecessor()))
                        .copied()
                        .collect();
                    if non_loop_operands.len() > 1 {
                        Some((phi.result(), phi.origin(), non_loop_operands))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let header_phi_operands: Vec<(SsaVarId, Vec<PhiOperand>)> = ssa
        .block(header_idx)
        .map(|header| {
            header
                .phi_nodes()
                .iter()
                .map(|phi| {
                    let non_loop_operands = phi
                        .operands()
                        .iter()
                        .filter(|op| non_loop_preds.contains(&op.predecessor()))
                        .copied()
                        .collect();
                    (phi.result(), non_loop_operands)
                })
                .collect()
        })
        .unwrap_or_default();

    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        let mut preheader = SsaBlock::new(preheader_idx);
        let mut preheader_phi_map: HashMap<SsaVarId, SsaVarId> = HashMap::new();

        for (header_result, origin, operands) in &phi_info {
            let new_var =
                editor.create_variable_for_origin(*origin, 0, DefSite::phi(preheader_idx));
            let mut preheader_phi = PhiNode::new(new_var, *origin);
            for op in operands {
                preheader_phi.add_operand(*op);
            }
            preheader_phi_map.insert(*header_result, new_var);
            preheader.phi_nodes_mut().push(preheader_phi);
        }

        preheader.add_instruction(SsaInstruction::synthetic(SsaOp::Jump {
            target: header_idx,
        }));
        editor.append_block(preheader)?;

        for &pred_idx in non_loop_preds {
            editor.redirect_terminator_target_structured(pred_idx, header_idx, preheader_idx)?;
        }

        for (header_result, non_loop_values) in &header_phi_operands {
            if non_loop_values.is_empty() {
                continue;
            }
            let replacement = if let [single] = non_loop_values.as_slice() {
                single.value()
            } else if let Some(&preheader_var) = preheader_phi_map.get(header_result) {
                preheader_var
            } else {
                continue;
            };
            editor.replace_phi_predecessor_group_for_result(
                header_idx,
                *header_result,
                non_loop_preds,
                PhiOperand::new(replacement, preheader_idx),
            )?;
        }

        Ok(())
    });
    if result.is_err() {
        return;
    }

    let event = crate::events::Event {
        kind: EventKind::ControlFlowRestructured,
        method: Some(method.clone()),
        location: Some(preheader_idx),
        message: format!("Inserted preheader B{preheader_idx} for loop at B{header_idx}"),
        pass: None,
    };
    events.push(event);
}

fn unify_latches<T, L>(
    ssa: &mut SsaFunction<T>,
    loop_info: &LoopInfo,
    method: &T::MethodRef,
    events: &L,
) where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let header_idx = loop_info.header.index();
    let latches: Vec<usize> = loop_info.latches.iter().map(|n| n.index()).collect();
    let unified_latch_idx = ssa.block_count();
    // Keyed by the header phi's *result*, not its origin. `VariableOrigin::Phi`
    // is a unit variant, so every phi that merges earlier phis — the ordinary
    // shape in lifted native code — carries the same origin. Keying on it made
    // one unified-latch phi stand for all of them and rewrote every header phi
    // to that single value, which is a dominance violation for all but one.
    let phi_info: Vec<(SsaVarId, VariableOrigin, Vec<PhiOperand>)> = ssa
        .block(header_idx)
        .map(|header| {
            header
                .phi_nodes()
                .iter()
                .map(|phi| {
                    let latch_operands: Vec<_> = phi
                        .operands()
                        .iter()
                        .filter(|op| latches.contains(&op.predecessor()))
                        .copied()
                        .collect();
                    (phi.result(), phi.origin(), latch_operands)
                })
                .collect()
        })
        .unwrap_or_default();

    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        let mut unified_latch = SsaBlock::new(unified_latch_idx);
        let mut latch_phi_vars: HashMap<SsaVarId, SsaVarId> = HashMap::new();

        for (header_result, origin, latch_operands) in &phi_info {
            if latch_operands.len() > 1 {
                let new_var =
                    editor.create_variable_for_origin(*origin, 0, DefSite::phi(unified_latch_idx));
                let mut latch_phi = PhiNode::new(new_var, *origin);
                for op in latch_operands {
                    latch_phi.add_operand(*op);
                }
                latch_phi_vars.insert(*header_result, new_var);
                unified_latch.phi_nodes_mut().push(latch_phi);
            } else if let [single] = latch_operands.as_slice() {
                latch_phi_vars.insert(*header_result, single.value());
            }
        }

        unified_latch.add_instruction(SsaInstruction::synthetic(SsaOp::Jump {
            target: header_idx,
        }));
        editor.append_block(unified_latch)?;

        for &latch_idx in &latches {
            editor.redirect_terminator_target_structured(
                latch_idx,
                header_idx,
                unified_latch_idx,
            )?;
        }

        for (header_result, _, _) in &phi_info {
            if let Some(&var) = latch_phi_vars.get(header_result) {
                editor.replace_phi_predecessor_group_for_result(
                    header_idx,
                    *header_result,
                    &latches,
                    PhiOperand::new(var, unified_latch_idx),
                )?;
            }
        }

        Ok(())
    });
    if result.is_err() {
        return;
    }

    let event = crate::events::Event {
        kind: EventKind::ControlFlowRestructured,
        method: Some(method.clone()),
        location: Some(unified_latch_idx),
        message: format!(
            "Unified {} latches into B{} for loop at B{}",
            latches.len(),
            unified_latch_idx,
            header_idx
        ),
        pass: None,
    };
    events.push(event);
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
        testing::{MockTarget, MockType, assert_mock_valid_full, run_mock_pass_boundary},
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

    fn phi_local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::phi(block),
            MockType::I32,
        )
    }

    fn build_loop_with_multiple_latches() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 8);
        let init = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);
        let limit = local_at(&mut ssa, 2, 0, 2);
        let i_phi = phi_local(&mut ssa, 3, 1);
        let cond = local_at(&mut ssa, 4, 1, 0);
        let next = local_at(&mut ssa, 5, 2, 0);
        let alt_next = local_at(&mut ssa, 6, 3, 0);
        let branch_cond = local_at(&mut ssa, 7, 5, 0);

        // Preheader (B0)
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: init,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(100),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        // Header (B1): both latches feed the induction phi.
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(init, 0));
        phi.add_operand(PhiOperand::new(next, 2));
        phi.add_operand(PhiOperand::new(alt_next, 3));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Clt {
            dest: cond,
            left: i_phi,
            right: limit,
            unsigned: false,
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 5,
            false_target: 4,
        }));
        ssa.add_block(b1);

        // Latch 1 (B2)
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        // Latch 2 (B3)
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Add {
            dest: alt_next,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b3.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b3);

        // Exit (B4)
        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Return { value: Some(i_phi) }));
        ssa.add_block(b4);

        // Body (B5): conditionally reaches either latch, so both are genuine
        // reachable back-edges to the header.
        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::Clt {
            dest: branch_cond,
            left: i_phi,
            right: one,
            unsigned: false,
        }));
        b5.add_instruction(instr(SsaOp::Branch {
            condition: branch_cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b5);

        ssa.recompute_uses();
        ssa
    }

    #[test]
    fn multiple_latches_unified() {
        let mut ssa = build_loop_with_multiple_latches();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;

        let original_blocks = ssa.block_count();
        let changed =
            run_mock_pass_boundary(&mut ssa, "multiple-latch loop canonicalization", |ssa| {
                run(ssa, &method, &log)
            });
        assert!(changed, "multiple latches should be unified");
        assert!(
            ssa.block_count() > original_blocks,
            "should add a unified latch block"
        );

        assert_mock_valid_full(&ssa, "multiple-latch canonical loop");
    }

    #[test]
    fn single_latch_already_canonical() {
        let mut ssa = SsaFunction::new(0, 5);
        let init = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);
        let limit = local_at(&mut ssa, 2, 0, 2);
        let i_phi = phi_local(&mut ssa, 3, 1);
        let cond = local_at(&mut ssa, 4, 1, 0);
        let next = local_at(&mut ssa, 5, 2, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: init,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(init, 0));
        phi.add_operand(PhiOperand::new(next, 2));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Clt {
            dest: cond,
            left: i_phi,
            right: limit,
            unsigned: false,
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Return { value: Some(i_phi) }));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "single-latch loop canonicalization", |ssa| {
                run(ssa, &method, &log)
            });
        assert!(!changed, "single latch should already be canonical");
    }

    #[test]
    fn no_loops_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty loop canonicalization", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(!changed);
    }

    #[test]
    fn single_block_no_loop() {
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
            run_mock_pass_boundary(&mut ssa, "single-block loop canonicalization", |ssa| {
                run(ssa, &method, &log)
            });
        assert!(!changed, "single block cannot have a loop");
    }

    #[test]
    fn idempotent_after_canonicalization() {
        let mut ssa = build_loop_with_multiple_latches();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;

        let first_changed =
            run_mock_pass_boundary(&mut ssa, "first loop canonicalization", |ssa| {
                run(ssa, &method, &log)
            });
        assert!(
            first_changed,
            "first canonicalization should rewrite the loop"
        );
        let after_first = ssa.block_count();

        // Second run should be idempotent
        let changed = run_mock_pass_boundary(&mut ssa, "idempotent loop canonicalization", |ssa| {
            run(ssa, &method, &log)
        });
        assert!(!changed, "second canonicalization should be a no-op");
        assert_eq!(ssa.block_count(), after_first);
    }

    #[test]
    fn loop_with_preheader_insertion_and_latch_unification() {
        // Build a loop with BOTH: no preheader (multiple entries) AND multiple latches
        let mut ssa = SsaFunction::new(0, 6);
        let init_a = local_at(&mut ssa, 0, 0, 0);
        let init_b = local_at(&mut ssa, 1, 1, 0);
        let one = local_at(&mut ssa, 2, 0, 1);
        let limit = local_at(&mut ssa, 3, 0, 2);
        let i_phi = phi_local(&mut ssa, 4, 2);
        let cond = local_at(&mut ssa, 5, 2, 0);
        let next1 = local_at(&mut ssa, 6, 3, 0);
        let next2 = local_at(&mut ssa, 7, 4, 0);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: init_a,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: limit,
            value: ConstValue::I32(50),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: init_b,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(4));
        phi.add_operand(PhiOperand::new(init_a, 0));
        phi.add_operand(PhiOperand::new(init_b, 1));
        phi.add_operand(PhiOperand::new(next1, 3));
        phi.add_operand(PhiOperand::new(next2, 4));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Clt {
            dest: cond,
            left: i_phi,
            right: limit,
            unsigned: false,
        }));
        b2.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 3,
            false_target: 5,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Add {
            dest: next1,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b3.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Add {
            dest: next2,
            left: i_phi,
            right: one,
            flags: None,
        }));
        b4.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b4);

        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::Return { value: Some(i_phi) }));
        ssa.add_block(b5);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed =
            run_mock_pass_boundary(&mut ssa, "preheader-and-latch canonicalization", |ssa| {
                run(ssa, &method, &log)
            });
        assert!(changed, "loop with both issues should be canonicalized");

        assert_mock_valid_full(&ssa, "preheader-and-latch canonical loop");
    }
}
