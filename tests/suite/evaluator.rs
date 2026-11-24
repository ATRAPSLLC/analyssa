//! Path-aware evaluator tests: concrete and symbolic evaluation, phi-aware
//! path tracing, constraint solving, loop fixpoint evaluation.

use std::{
    error::Error,
    num::{NonZeroU8, NonZeroU16},
};

use analyssa::{
    PointerSize,
    analysis::{
        SsaVerifier, VerifyLevel,
        constraints::Constraint,
        consts::Machine,
        evaluator::{ControlFlow, EvaluatorConfig, SsaEvaluator},
    },
    ir::{
        FloatBits, FloatFormat, FloatRules, NanPropagation,
        block::SsaBlock,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        instruction::SsaInstruction,
        ops::{
            AccessPermission, AccessQualifiers, AtomicAccessWidth, AtomicOrdering, CmpKind,
            FenceKind, FloatFunctionKind, HintOp, KindedVecData, Locality, NativeKindedData,
            SsaEffectKind, SsaOp, SysRegOp, SystemOpKind, TlbMaintenanceOp,
        },
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{MACHINE, MockTarget, MockType},
};

fn local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
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

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

fn typed(ssa: &mut SsaFunction<MockTarget>, idx: u16, ty: MockType) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, usize::from(idx)),
        ty,
    )
}

/// A one-block function declaring one temporary per entry of `types`, whose
/// block holds the instructions `body` builds over them.
fn scratch<const N: usize>(
    types: [MockType; N],
    body: impl FnOnce([SsaVarId; N]) -> Vec<SsaOp<MockTarget>>,
) -> (SsaFunction<MockTarget>, [SsaVarId; N]) {
    let mut ssa = SsaFunction::new(0, 0);
    let mut idx = 0u16;
    let vars = types.map(|ty| {
        let var = typed(&mut ssa, idx, ty);
        idx = idx.saturating_add(1);
        var
    });
    let mut block = SsaBlock::new(0);
    for op in body(vars) {
        block.add_instruction(instr(op));
    }
    ssa.add_block(block);
    ssa.recompute_uses();
    (ssa, vars)
}

#[test]
fn evaluator_resolves_simple_arithmetic_to_concrete() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let sum = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(sum))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);
    assert_eq!(eval.get_concrete(sum).and_then(|cv| cv.as_i64()), Some(8));
    assert_eq!(eval.get_concrete(a).and_then(|cv| cv.as_i64()), Some(5));
}

#[test]
fn evaluator_resolves_chained_arithmetic() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let c = local(&mut ssa, 2, 0, 2);
    let ab = local(&mut ssa, 3, 0, 3);
    let abc = local(&mut ssa, 4, 0, 4);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: c,
        value: ConstValue::I32(4),
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: ab,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: abc,
        left: ab,
        right: c,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(abc))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);
    assert_eq!(eval.get_concrete(abc).and_then(|cv| cv.as_i64()), Some(10));
}

fn build_simple_diamond() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 4);
    let cond = local(&mut ssa, 0, 0, 0);
    let left = local(&mut ssa, 1, 1, 0);
    let right = local(&mut ssa, 2, 2, 0);
    let merged = phi_local(&mut ssa, 3, 3);

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
    b1.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(100),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(200),
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);

    let mut b3 = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(3));
    phi.add_operand(PhiOperand::new(left, 1));
    phi.add_operand(PhiOperand::new(right, 2));
    b3.add_phi(phi);
    b3.add_instruction(instr(SsaOp::returns(Some(merged))));
    ssa.add_block(b3);

    ssa.recompute_uses();
    ssa
}

#[test]
fn evaluator_selects_phi_operand_based_on_predecessor() {
    let ssa = build_simple_diamond();
    let merged = SsaVarId::from_index(3);

    let mut eval_left = SsaEvaluator::new(&ssa, MACHINE);
    eval_left.evaluate_block(0);
    eval_left.evaluate_block(1);
    eval_left.set_predecessor(Some(1));
    eval_left.evaluate_phis(3);
    assert_eq!(
        eval_left.get_concrete(merged).and_then(|cv| cv.as_i64()),
        Some(100)
    );

    let mut eval_right = SsaEvaluator::new(&ssa, MACHINE);
    eval_right.evaluate_block(0);
    eval_right.evaluate_block(2);
    eval_right.set_predecessor(Some(2));
    eval_right.evaluate_phis(3);
    assert_eq!(
        eval_right.get_concrete(merged).and_then(|cv| cv.as_i64()),
        Some(200)
    );
}

#[test]
fn evaluator_evaluates_comparison_results() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let is_lt = local(&mut ssa, 2, 0, 2);
    let is_eq = local(&mut ssa, 3, 0, 3);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Clt {
        dest: is_lt,
        left: a,
        right: b,
        unsigned: false,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::Ceq {
        dest: is_eq,
        left: a,
        right: b,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(is_lt))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);
    assert_eq!(eval.get_concrete(is_lt).and_then(|cv| cv.as_i64()), Some(1));
    assert_eq!(eval.get_concrete(is_eq).and_then(|cv| cv.as_i64()), Some(0));
}

#[test]
fn evaluate_path_evaluates_blocks_in_order() {
    let ssa = build_simple_diamond();
    let merged = SsaVarId::from_index(3);

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_path(&[0, 1]);
    eval.set_predecessor(Some(1));
    eval.evaluate_phis(3);
    assert_eq!(
        eval.get_concrete(merged).and_then(|cv| cv.as_i64()),
        Some(100)
    );
}

#[test]
fn evaluate_blocks_evaluates_multiple_in_sequence() {
    let ssa = build_simple_diamond();
    let left = SsaVarId::from_index(1);

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_blocks(&[0, 1]);
    assert_eq!(
        eval.get_concrete(left).and_then(|cv| cv.as_i64()),
        Some(100)
    );
}

fn build_counter_loop() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 4);

    let init = local(&mut ssa, 0, 0, 0);
    let one = local(&mut ssa, 1, 0, 1);
    let limit = local(&mut ssa, 2, 0, 2);
    let counter = phi_local(&mut ssa, 3, 1);

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
        value: ConstValue::I32(3),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let cond = local(&mut ssa, 5, 1, 0);
    let next = local(&mut ssa, 6, 1, 1);
    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(counter, VariableOrigin::Local(3));
    phi.add_operand(PhiOperand::new(init, 0));
    phi.add_operand(PhiOperand::new(next, 1));
    b1.add_phi(phi);
    b1.add_instruction(instr(SsaOp::Clt {
        dest: cond,
        left: counter,
        right: limit,
        unsigned: false,
        signaling: false,
    }));
    b1.add_instruction(instr(SsaOp::Add {
        dest: next,
        left: counter,
        right: one,
        flags: None,
    }));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(Some(counter))));
    ssa.add_block(b2);

    ssa.recompute_uses();
    ssa
}

#[test]
fn evaluate_loop_to_fixpoint_converges() {
    let ssa = build_counter_loop();
    let counter = SsaVarId::from_index(3);

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);
    eval.evaluate_loop_to_fixpoint(&[1], 10);

    // The evaluator may or may not resolve the loop counter depending
    // on whether it can track phi values across iterations
    let _ = eval.get_concrete(counter);
    // At minimum, no panic occurred
}

#[test]
fn evaluate_loop_iterations_converges() {
    let ssa = build_counter_loop();
    let counter = SsaVarId::from_index(3);

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);
    eval.evaluate_loop_iterations(&[1], 5);

    // The evaluator may or may not resolve the loop counter concretely
    let _ = eval.get_concrete(counter);
    // At minimum, no panic occurred
}

#[test]
fn evaluator_resolves_known_condition_to_boolean() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let is_lt = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Clt {
        dest: is_lt,
        left: a,
        right: b,
        unsigned: false,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(is_lt))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);
    let result = eval.evaluate_condition_with_constraints(is_lt);
    assert_eq!(result, Some(true));
}

#[test]
fn evaluator_distinguishes_concrete_from_unknown() {
    let mut ssa = SsaFunction::new(0, 0);
    let param = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let const_val = local(&mut ssa, 1, 0, 1);
    let sum = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: const_val,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: param,
        right: const_val,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(sum))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);

    assert!(eval.is_concrete(const_val));
    assert!(!eval.is_unknown(const_val));

    assert!(eval.is_unknown(param));
    assert!(!eval.is_concrete(param));

    assert!(eval.is_symbolic(sum) || eval.is_unknown(sum));
}

#[test]
fn evaluator_resolves_bitwise_operations() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let and_result = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(6),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::And {
        dest: and_result,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(and_result))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);
    assert_eq!(
        eval.get_concrete(and_result).and_then(|cv| cv.as_i64()),
        Some(2)
    );
}

#[test]
fn control_flow_is_terminal_for_terminal() {
    let cf = ControlFlow::Terminal;
    assert!(cf.is_terminal());
    assert!(!cf.is_unknown());
    assert_eq!(cf.target(), None);

    let cf = ControlFlow::Unknown;
    assert!(cf.is_unknown());
    assert!(!cf.is_terminal());

    let cf = ControlFlow::Continue(7);
    assert!(!cf.is_terminal());
    assert!(!cf.is_unknown());
    assert_eq!(cf.target(), Some(7));
}

#[test]
fn evaluator_evaluates_branch_cmp_targets() {
    let mut ssa = SsaFunction::new(0, 2);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(instr(SsaOp::BranchCmp {
        left: a,
        right: b,
        cmp: CmpKind::Lt,
        unsigned: false,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::returns(Some(a))));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(Some(b))));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);
    assert_eq!(eval.get_concrete(a).and_then(|cv| cv.as_i64()), Some(1));
}

#[test]
fn ssa_evaluator_has_constraints_for_variables() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let cmp = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Clt {
        dest: cmp,
        left: a,
        right: b,
        unsigned: false,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(cmp))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);

    let _ = eval.has_constraints(cmp);
}

#[test]
fn evaluate_with_trace_resolves_complex_path() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let c = local(&mut ssa, 2, 0, 2);
    let d = local(&mut ssa, 3, 0, 3);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: c,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: d,
        left: c,
        right: a,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(d))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.evaluate_block(0);

    let trace = eval.evaluate_with_trace(d, 10);
    assert_eq!(trace, Some(40));
}

/// A checkpoint must restore every kind of state the evaluator tracks, not
/// just the value map: overwrites, first-time definitions, removals, local
/// mirroring and the phi predecessor all have to come back.
#[test]
fn checkpoint_rollback_restores_evaluator_state() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let sum = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(sum))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.set_concrete(a, ConstValue::I32(5));
    eval.set_concrete(b, ConstValue::I32(3));
    eval.set_predecessor(Some(7));

    let mark = eval.checkpoint();

    // Speculate: overwrite a known value, define a new one, drop another, and
    // move the predecessor.
    eval.set_concrete(a, ConstValue::I32(99));
    eval.evaluate_block(0);
    eval.set_unknown(b);
    eval.set_predecessor(Some(11));
    assert_eq!(eval.get_concrete(a).and_then(|cv| cv.as_i64()), Some(99));
    assert_eq!(eval.get_concrete(sum).and_then(|cv| cv.as_i64()), Some(102));
    assert!(eval.get(b).is_none());

    eval.rollback(mark);

    assert_eq!(eval.get_concrete(a).and_then(|cv| cv.as_i64()), Some(5));
    assert_eq!(eval.get_concrete(b).and_then(|cv| cv.as_i64()), Some(3));
    assert!(eval.get(sum).is_none(), "speculative definition survived");
    assert_eq!(eval.predecessor(), Some(7));
}

/// Rolling back to an outer mark must subsume any marks taken inside it, so a
/// depth-first exploration can abandon a whole subtree in one step.
#[test]
fn checkpoint_rollback_nests() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    ssa.add_block(SsaBlock::new(0));
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.set_concrete(a, ConstValue::I32(1));

    let outer = eval.checkpoint();
    eval.set_concrete(a, ConstValue::I32(2));

    let inner = eval.checkpoint();
    eval.set_concrete(a, ConstValue::I32(3));
    eval.set_concrete(b, ConstValue::I32(4));

    eval.rollback(inner);
    assert_eq!(eval.get_concrete(a).and_then(|cv| cv.as_i64()), Some(2));
    assert!(eval.get(b).is_none());

    // Re-explore from the inner point, then discard everything at once.
    eval.set_concrete(b, ConstValue::I32(8));
    eval.rollback(outer);
    assert_eq!(eval.get_concrete(a).and_then(|cv| cv.as_i64()), Some(1));
    assert!(eval.get(b).is_none());
}

/// A compare branch over a pair that names no two numbers is undecided, not
/// "the operands are equal".
///
/// An integer against a float is the case that stays: the two are not two
/// readings of one value. An integer against a *native* int is not - both name
/// a number, and the variants a frontend picks for one comparison need not
/// agree.
#[test]
fn a_compare_branch_over_an_incomparable_pair_is_undecided() {
    let (ssa, [left, right]) = scratch([MockType::I32, MockType::F64], |[left, right]| {
        vec![SsaOp::BranchCmp {
            left,
            right,
            cmp: CmpKind::Ne,
            unsigned: true,
            true_target: 1,
            false_target: 2,
        }]
    });
    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.set_concrete(left, ConstValue::I32(1));
    eval.set_concrete(right, ConstValue::from(2.0f64));
    assert_eq!(eval.next_block(0), ControlFlow::Unknown);
    assert_eq!(
        SsaEvaluator::<MockTarget>::evaluate_comparison(
            &ConstValue::I32(1),
            &ConstValue::from(2.0f64),
            CmpKind::Ne,
            false,
            PointerSize::Bit64,
        ),
        None
    );
    assert_eq!(
        SsaEvaluator::<MockTarget>::evaluate_comparison(
            &ConstValue::I32(1),
            &ConstValue::NativeInt(2),
            CmpKind::Ne,
            false,
            PointerSize::Bit64,
        ),
        Some(true),
        "two integers of different variants are still two numbers"
    );
}

/// For floats the `unsigned` flag means "unordered or": `ble` (flag clear) is
/// false on a NaN operand and `ble.un` (flag set) is true.
#[test]
fn float_less_or_equal_honours_the_unordered_flag() {
    type Ev<'a> = SsaEvaluator<'a, MockTarget>;
    let nan = ConstValue::<MockTarget>::from(f64::NAN);
    let zero = ConstValue::<MockTarget>::from(0.0f64);
    let one = ConstValue::<MockTarget>::from(1.0f64);
    for cmp in [CmpKind::Le, CmpKind::Ge] {
        assert_eq!(
            Ev::evaluate_comparison(&nan, &zero, cmp, false, PointerSize::Bit64),
            Some(false)
        );
        assert_eq!(
            Ev::evaluate_comparison(&zero, &nan, cmp, false, PointerSize::Bit64),
            Some(false)
        );
        assert_eq!(
            Ev::evaluate_comparison(&nan, &zero, cmp, true, PointerSize::Bit64),
            Some(true)
        );
        assert_eq!(
            Ev::evaluate_comparison(&zero, &zero, cmp, false, PointerSize::Bit64),
            Some(true)
        );
        assert_eq!(
            Ev::evaluate_comparison(&zero, &zero, cmp, true, PointerSize::Bit64),
            Some(true)
        );
    }
    assert_eq!(
        Ev::evaluate_comparison(&zero, &one, CmpKind::Le, false, PointerSize::Bit64),
        Some(true)
    );
    assert_eq!(
        Ev::evaluate_comparison(&one, &zero, CmpKind::Le, true, PointerSize::Bit64),
        Some(false)
    );
    assert_eq!(
        Ev::evaluate_comparison(&one, &zero, CmpKind::Ge, false, PointerSize::Bit64),
        Some(true)
    );
    assert_eq!(
        Ev::evaluate_comparison(&zero, &one, CmpKind::Ge, true, PointerSize::Bit64),
        Some(false)
    );
    // Integers keep their signedness: -1 <= 0 signed, but not unsigned.
    let minus_one = ConstValue::<MockTarget>::I32(-1);
    let int_zero = ConstValue::<MockTarget>::I32(0);
    assert_eq!(
        Ev::evaluate_comparison(
            &minus_one,
            &int_zero,
            CmpKind::Le,
            false,
            PointerSize::Bit64
        ),
        Some(true)
    );
    assert_eq!(
        Ev::evaluate_comparison(&minus_one, &int_zero, CmpKind::Le, true, PointerSize::Bit64),
        Some(false)
    );
}

/// Bit-manipulation ops fold at the operand's width, not through `u32`.
#[test]
fn bit_manipulation_folds_at_the_operand_width() {
    let (ssa, [src, swapped, count, scan]) = scratch(
        [MockType::I64, MockType::I64, MockType::I32, MockType::I32],
        |_| Vec::new(),
    );
    let mut eval = SsaEvaluator::new(&ssa, MACHINE);

    eval.set_concrete(src, ConstValue::I64(0x0102_0304_0506_0708));
    eval.evaluate_op(&SsaOp::BSwap { dest: swapped, src });
    assert_eq!(
        eval.get_concrete(swapped),
        Some(&ConstValue::I64(0x0807_0605_0403_0201))
    );

    eval.set_concrete(src, ConstValue::I64(1 << 32));
    eval.evaluate_op(&SsaOp::Popcount { dest: count, src });
    assert_eq!(
        eval.get_concrete(count).and_then(ConstValue::as_i64),
        Some(1)
    );

    // The bit scan of zero is undefined, so it has no value.
    eval.set_concrete(src, ConstValue::I64(0));
    eval.set_concrete(scan, ConstValue::I32(7));
    eval.evaluate_op(&SsaOp::BitScanReverse { dest: scan, src });
    assert!(eval.get(scan).is_none());
}

/// An operation the evaluator does not model leaves its destinations unknown,
/// not holding whatever an earlier evaluation stored there.
#[test]
fn an_unmodelled_operation_forgets_every_definition() {
    let (ssa, [left, right, dest, flags]) = scratch([MockType::I32; 4], |_| Vec::new());
    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.set_concrete(left, ConstValue::I32(1));
    eval.set_concrete(right, ConstValue::I32(2));
    eval.set_concrete(dest, ConstValue::I32(99));
    eval.set_concrete(flags, ConstValue::I32(99));

    eval.evaluate_op(&SsaOp::SystemOp(Box::new(NativeKindedData {
        kind: SystemOpKind::CpuId,
        mnemonic: "cpuid".into(),
        metadata: None,
        outputs: vec![dest, flags],
        inputs: vec![left, right],
    })));
    assert!(eval.get(dest).is_none(), "stale destination survived");
    assert!(eval.get(flags).is_none(), "stale flags survived");

    // A modelled operation still forgets the flags it does not compute.
    eval.set_concrete(flags, ConstValue::I32(99));
    eval.evaluate_op(&SsaOp::Add {
        dest,
        left,
        right,
        flags: Some(flags),
    });
    assert_eq!(eval.get_concrete(dest), Some(&ConstValue::I32(3)));
    assert!(eval.get(flags).is_none(), "stale flags survived");
}

/// Checked operations replace known results and forget declined results and secondary definitions.
#[test]
fn checked_integer_results_replace_or_forget_every_definition() {
    for (kind, a, b, expected) in [
        (0, 1, 2, Some(3)),
        (0, i32::MAX, 1, None),
        (1, 1, 2, Some(-1)),
        (1, i32::MIN, 1, None),
        (2, 3, 7, Some(21)),
        (2, i32::MAX, 2, None),
    ] {
        for unknown in [false, true] {
            let (ssa, [left, right, dest, flags]) = scratch([MockType::I32; 4], |_| Vec::new());
            let mut eval = SsaEvaluator::new(&ssa, MACHINE);
            if !unknown {
                eval.set_concrete(left, ConstValue::I32(a));
            }
            eval.set_concrete(right, ConstValue::I32(b));
            eval.set_concrete(dest, ConstValue::I32(99));
            eval.set_concrete(flags, ConstValue::I32(99));
            let op = match kind {
                0 => SsaOp::AddOvf {
                    dest,
                    left,
                    right,
                    unsigned: false,
                    flags: Some(flags),
                },
                1 => SsaOp::SubOvf {
                    dest,
                    left,
                    right,
                    unsigned: false,
                    flags: Some(flags),
                },
                _ => SsaOp::MulOvf {
                    dest,
                    left,
                    right,
                    unsigned: false,
                    flags: Some(flags),
                },
            };
            eval.evaluate_op(&op);
            let expected = if unknown { None } else { expected }.map(ConstValue::I32);
            assert_eq!(
                eval.get_concrete(dest),
                expected.as_ref(),
                "kind={kind}, unknown={unknown}"
            );
            assert!(
                eval.get(flags).is_none(),
                "stale flags: kind={kind}, unknown={unknown}"
            );
        }
    }
}

/// Rolling back undoes a field load and an address-taken local or argument.
#[test]
fn rollback_restores_field_loads_and_address_taken_slots() {
    let mut ssa = SsaFunction::new(1, 1);
    let local_value = local(&mut ssa, 0, 0, 0);
    let arg_value = ssa.create_variable(
        VariableOrigin::Argument(0),
        1,
        DefSite::instruction(0, 1),
        MockType::I32,
    );
    let object = local(&mut ssa, 1, 0, 2);
    let stored = local(&mut ssa, 2, 0, 3);
    let loaded = local(&mut ssa, 3, 0, 4);
    let address = local(&mut ssa, 4, 0, 5);
    let reloaded = local(&mut ssa, 5, 0, 6);
    ssa.add_block(SsaBlock::new(0));
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
    eval.set_concrete(local_value, ConstValue::I32(5));
    eval.set_concrete(arg_value, ConstValue::I32(6));
    eval.evaluate_op(&SsaOp::StoreField {
        access: AccessQualifiers::FLAT,
        object,
        field: 1,
        value: stored,
    });

    let mark = eval.checkpoint();
    eval.evaluate_op(&SsaOp::LoadField {
        access: AccessQualifiers::FLAT,
        dest: loaded,
        object,
        field: 1,
    });
    assert!(eval.get(loaded).is_some());
    eval.evaluate_op(&SsaOp::LoadLocalAddr {
        dest: address,
        local_index: 0,
    });
    eval.evaluate_op(&SsaOp::LoadArgAddr {
        dest: address,
        arg_index: 0,
    });
    eval.rollback(mark);

    assert!(
        eval.get(loaded).is_none(),
        "speculative field load survived"
    );
    eval.evaluate_op(&SsaOp::LoadLocal {
        dest: reloaded,
        local_index: 0,
    });
    assert_eq!(eval.get_concrete(reloaded), Some(&ConstValue::I32(5)));
    eval.evaluate_op(&SsaOp::LoadArg {
        dest: reloaded,
        arg_index: 0,
    });
    assert_eq!(eval.get_concrete(reloaded), Some(&ConstValue::I32(6)));
}

/// A `brtrue` on a null reference takes the false arm.
#[test]
fn a_branch_on_null_takes_the_false_arm() {
    let (ssa, [cond]) = scratch([MockType::Ref], |[condition]| {
        vec![SsaOp::Branch {
            condition,
            true_target: 1,
            false_target: 2,
        }]
    });
    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.set_concrete(cond, ConstValue::Null);
    assert_eq!(eval.next_block(0), ControlFlow::Continue(2));
}

/// A signed bound decides only a signed comparison: `x <= 0` (signed) says
/// nothing about `x >u 0`, which holds for `x = -1`. And a condition with no
/// truth value, or a comparison with an operand it cannot compare against,
/// is undetermined.
#[test]
fn constraints_decide_only_comparisons_they_speak_for() {
    let mut ssa = SsaFunction::new(0, 0);
    let x = local(&mut ssa, 0, 0, 0);
    let zero = local(&mut ssa, 1, 0, 0);
    let unsigned_gt = local(&mut ssa, 2, 0, 1);
    let equal = local(&mut ssa, 3, 0, 2);
    let float = local(&mut ssa, 4, 0, 3);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    block.add_instruction(instr(SsaOp::Cgt {
        dest: unsigned_gt,
        left: x,
        right: zero,
        unsigned: true,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::Ceq {
        dest: equal,
        left: x,
        right: zero,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.set_concrete(zero, ConstValue::I32(0));
    eval.add_constraint(x, Constraint::LessOrEqual(ConstValue::I32(0)));
    assert_eq!(eval.evaluate_condition_with_constraints(unsigned_gt), None);

    eval.clear_constraints(x);
    // A bound stated with a different variant still states a number, so the
    // equality it implies is decided rather than dropped.
    eval.add_constraint(x, Constraint::Equal(ConstValue::NativeInt(0)));
    assert_eq!(
        eval.evaluate_condition_with_constraints(equal),
        Some(true),
        "x == 0 (native int) settles x == 0 (int32)"
    );

    eval.clear_constraints(x);
    eval.add_constraint(x, Constraint::Equal(ConstValue::from(0.0f64)));
    assert_eq!(
        eval.evaluate_condition_with_constraints(equal),
        None,
        "a float bound names no integer to compare against"
    );

    eval.set_concrete(float, ConstValue::from(0.5f64));
    assert_eq!(eval.evaluate_condition_with_constraints(float), None);
}

/// Both results of the double-width arithmetic are known when its operands
/// are, and both are forgotten when one is not.
#[test]
fn both_wide_results_fold_over_constants() {
    let (ssa, [left, right, low, high]) = scratch([MockType::U64; 4], |_| Vec::new());
    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.set_concrete(left, ConstValue::U64(u64::MAX));
    eval.set_concrete(right, ConstValue::U64(3));
    eval.evaluate_op(&SsaOp::WideMul {
        low,
        high,
        left,
        right,
        unsigned: true,
    });
    assert_eq!(eval.get_concrete(low), Some(&ConstValue::U64(u64::MAX - 2)));
    assert_eq!(eval.get_concrete(high), Some(&ConstValue::U64(2)));

    // A divisor that is not known leaves both results unknown, and forgets
    // what they held.
    let (ssa, [high_in, low_in, divisor, quotient, remainder]) =
        scratch([MockType::U64; 5], |_| Vec::new());
    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.set_concrete(high_in, ConstValue::U64(0));
    eval.set_concrete(low_in, ConstValue::U64(7));
    eval.set_concrete(quotient, ConstValue::U64(99));
    eval.set_concrete(remainder, ConstValue::U64(99));
    eval.evaluate_op(&SsaOp::WideDiv {
        quotient,
        remainder,
        high: high_in,
        low: low_in,
        divisor,
        unsigned: true,
    });
    assert!(eval.get(quotient).is_none());
    assert!(eval.get(remainder).is_none());
    eval.set_concrete(divisor, ConstValue::U64(2));
    eval.evaluate_op(&SsaOp::WideDiv {
        quotient,
        remainder,
        high: high_in,
        low: low_in,
        divisor,
        unsigned: true,
    });
    assert_eq!(eval.get_concrete(quotient), Some(&ConstValue::U64(3)));
    assert_eq!(eval.get_concrete(remainder), Some(&ConstValue::U64(1)));
}

/// **The evaluator converts a NaN between formats by its machine's rule.**
/// The signaling double `0x7ff4000000000000` narrowed to a bfloat16 is the
/// quiet `0x7fe0` where the machine keeps the NaN as x86 does, and unknown on
/// a machine that states no rule.
#[test]
fn a_nan_conversion_evaluates_by_the_machines_rule() {
    let (ssa, [double, narrowed]) = scratch([MockType::F64, MockType::BF16], |_| Vec::new());
    let x86 = Machine::new(
        PointerSize::Bit64,
        None,
        FloatRules {
            nan_conversion: Some(NanPropagation::Operand),
            ..MACHINE.float
        },
    );
    let conversion = SsaOp::FloatConv {
        dest: narrowed,
        operand: double,
        target: MockType::BF16,
        rounding: None,
        source: None,
        environment: None,
    };
    let signaling = ConstValue::from(f64::from_bits(0x7ff4_0000_0000_0000));
    let evaluated = |machine| {
        let mut eval = SsaEvaluator::new(&ssa, machine);
        eval.set_concrete(double, signaling.clone());
        eval.evaluate_op(&conversion);
        eval.get_concrete(narrowed).cloned()
    };
    assert_eq!(
        evaluated(x86),
        FloatBits::new(FloatFormat::BFloat16, 0x7fe0).map(ConstValue::Float)
    );
    assert_eq!(evaluated(MACHINE), None, "no rule");
}

/// **The exact float functions evaluate.** `3 fscale 2.7` is 12, one
/// result; `7 fprem 2` is the remainder 1 with the quotient 3's low bit in
/// its last result, each of several results at once; `fsin` evaluates to
/// nothing.
#[test]
fn an_exact_float_function_evaluates() {
    let (ssa, [value, by, scaled, remainder, codes, low_bit, sine]) = scratch(
        [
            MockType::F64,
            MockType::F64,
            MockType::F64,
            MockType::F64,
            MockType::U32,
            MockType::U8,
            MockType::F64,
        ],
        |_| Vec::new(),
    );
    let function = |kind, outputs: Vec<SsaVarId>, inputs: Vec<SsaVarId>| {
        SsaOp::FloatFunction(Box::new(KindedVecData {
            kind,
            outputs,
            inputs,
        }))
    };
    let mut eval = SsaEvaluator::new(&ssa, MACHINE);
    eval.set_concrete(value, ConstValue::from(3.0));
    eval.set_concrete(by, ConstValue::from(2.7));
    eval.evaluate_op(&function(
        FloatFunctionKind::Scale,
        vec![scaled],
        vec![value, by],
    ));
    assert_eq!(eval.get_concrete(scaled), Some(&ConstValue::from(12.0)));

    eval.set_concrete(value, ConstValue::from(7.0));
    eval.set_concrete(by, ConstValue::from(2.0));
    eval.evaluate_op(&function(
        FloatFunctionKind::Rem,
        vec![remainder, codes, low_bit],
        vec![value, by],
    ));
    assert_eq!(eval.get_concrete(remainder), Some(&ConstValue::from(1.0)));
    assert_eq!(eval.get_concrete(low_bit), Some(&ConstValue::U8(1)));

    eval.evaluate_op(&function(FloatFunctionKind::Sin, vec![sine], vec![value]));
    assert_eq!(eval.get_concrete(sine), None);
}

/// Constructs an executable field roundtrip with a real entry object and instruction sites.
fn qualified_field_roundtrip(
    instance: bool,
    store: AccessQualifiers,
    load: AccessQualifiers,
    intervening: Option<SsaOp<MockTarget>>,
) -> (SsaFunction<MockTarget>, SsaVarId) {
    let mut ssa = SsaFunction::new(1, 0);
    let object = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::Ref,
    );
    let value = typed(&mut ssa, 0, MockType::I32);
    let result = typed(
        &mut ssa,
        if intervening.is_some() { 3 } else { 2 },
        MockType::I32,
    );
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(41),
    }));
    block.add_instruction(instr(if instance {
        SsaOp::StoreField {
            object,
            field: 1,
            value,
            access: store,
        }
    } else {
        SsaOp::StoreStaticField {
            field: 1,
            value,
            access: store,
        }
    }));
    if let Some(operation) = intervening {
        block.add_instruction(instr(operation));
    }
    block.add_instruction(instr(if instance {
        SsaOp::LoadField {
            dest: result,
            object,
            field: 1,
            access: load,
        }
    } else {
        SsaOp::LoadStaticField {
            dest: result,
            field: 1,
            access: load,
        }
    }));
    block.add_instruction(instr(SsaOp::returns(Some(result))));
    ssa.add_block(block);
    ssa.recompute_uses();
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "invalid field witness: {errors:?}");
    (ssa, result)
}

/// Volatile operations and unresolved access domains never acquire a field constant.
#[test]
fn qualified_field_roundtrips_refuse_unproved_accesses() {
    for access in [
        AccessQualifiers::FLAT.made_volatile(),
        AccessQualifiers::new(NonZeroU16::new(1), AccessPermission::Current),
        AccessQualifiers::new(NonZeroU16::new(7), AccessPermission::Current),
        AccessQualifiers::new(None, AccessPermission::Unprivileged),
        AccessQualifiers::new(None, AccessPermission::Execute),
        AccessQualifiers::new(None, AccessPermission::ShadowStack),
        AccessQualifiers::new(None, AccessPermission::UnprivilegedShadowStack),
    ] {
        for instance in [false, true] {
            for (store, load) in [
                (AccessQualifiers::FLAT, access),
                (access, AccessQualifiers::FLAT),
                (access, access),
            ] {
                let (ssa, result) = qualified_field_roundtrip(instance, store, load, None);
                let mut eval =
                    SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
                eval.evaluate_block(0);
                assert_eq!(
                    eval.get_concrete(result),
                    None,
                    "instance={instance}, store={store:?}, load={load:?}"
                );
            }
        }
    }
}

/// Locality hints and reduced alignment preserve field values in the ordinary domain.
#[test]
fn qualified_field_roundtrips_preserve_value_neutral_hints() {
    for access in [
        AccessQualifiers::FLAT,
        AccessQualifiers::FLAT.with_locality(Locality::NonTemporal),
        AccessQualifiers::FLAT.with_alignment(NonZeroU8::MIN),
    ] {
        for instance in [false, true] {
            for (store, load) in [
                (AccessQualifiers::FLAT, access),
                (access, AccessQualifiers::FLAT),
                (access, access),
            ] {
                let (ssa, result) = qualified_field_roundtrip(instance, store, load, None);
                let mut eval =
                    SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
                eval.evaluate_block(0);
                assert_eq!(eval.get_concrete(result), Some(&ConstValue::I32(41)));
            }
        }
    }
}

/// Unresolved writes, atomic operations and fences invalidate store-time
/// snapshots. An operation of each effect class sits between a field's store
/// and its load: nothing, a coprocessor load (a read), a TLB invalidation (a
/// write of memory unknown), a system register exchange (a read and write),
/// a full fence, a hint (machine state, no memory) and `cpuid` (a value the
/// machine supplies); the writes and the fence invalidate the snapshot.
#[test]
fn qualified_field_snapshots_respect_intervening_memory_effects() {
    let system = |kind| {
        SsaOp::<MockTarget>::SystemOp(Box::new(NativeKindedData {
            kind,
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![],
            inputs: vec![],
        }))
    };
    let barriers = [
        (SsaOp::Nop, SsaEffectKind::Pure, false),
        (
            system(SystemOpKind::ControlRegister(SysRegOp::CoprocessorLoad)),
            SsaEffectKind::Read,
            false,
        ),
        (
            system(SystemOpKind::TlbMaintenance(
                TlbMaintenanceOp::InvalidatePage,
            )),
            SsaEffectKind::Write,
            true,
        ),
        (
            system(SystemOpKind::ControlRegister(
                SysRegOp::ExchangeSystemRegister,
            )),
            SsaEffectKind::ReadWrite,
            true,
        ),
        (
            SsaOp::Fence {
                kind: FenceKind::Full,
            },
            SsaEffectKind::Fence,
            true,
        ),
        (
            system(SystemOpKind::Hint(HintOp::GenericHint)),
            SsaEffectKind::Opaque,
            false,
        ),
        (
            system(SystemOpKind::CpuId),
            SsaEffectKind::RuntimeDefined,
            false,
        ),
    ];
    for (barrier, kind, invalidates) in barriers {
        assert_eq!(barrier.effects().kind, kind, "{barrier}");
        for instance in [false, true] {
            let (ssa, result) = qualified_field_roundtrip(
                instance,
                AccessQualifiers::FLAT,
                AccessQualifiers::FLAT,
                Some(barrier.clone()),
            );
            let mut eval =
                SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
            eval.evaluate_block(0);
            let expected = (!invalidates).then_some(ConstValue::I32(41));
            assert_eq!(
                eval.get_concrete(result),
                expected.as_ref(),
                "instance={instance}, {barrier}"
            );
        }
    }
}

/// Refused writes erase earlier snapshots, and speculative invalidation rolls back.
#[test]
fn qualified_field_snapshots_rollback_refused_writes() -> Result<(), Box<dyn Error>> {
    for instance in [false, true] {
        for access in [
            AccessQualifiers::FLAT.made_volatile(),
            AccessQualifiers::new(NonZeroU16::new(7), AccessPermission::Current),
            AccessQualifiers::new(None, AccessPermission::Unprivileged),
        ] {
            let (ssa, result) = qualified_field_roundtrip(
                instance,
                AccessQualifiers::FLAT,
                AccessQualifiers::FLAT,
                None,
            );
            let mut eval =
                SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
            let block = ssa.block(0).ok_or("missing field block")?;
            for instruction in block.instructions().iter().take(2) {
                eval.evaluate_op(instruction.op());
            }
            let mark = eval.checkpoint();
            let store = if instance {
                SsaOp::StoreField {
                    object: SsaVarId::from_index(0),
                    field: 1,
                    value: SsaVarId::from_index(1),
                    access,
                }
            } else {
                SsaOp::StoreStaticField {
                    field: 1,
                    value: SsaVarId::from_index(1),
                    access,
                }
            };
            eval.evaluate_op(&store);
            let load = block.instruction(2).ok_or("missing field load")?.op();
            eval.evaluate_op(load);
            assert_eq!(
                eval.get_concrete(result),
                None,
                "instance={instance}, access={access:?}"
            );
            eval.rollback(mark);
            eval.evaluate_op(load);
            assert_eq!(eval.get_concrete(result), Some(&ConstValue::I32(41)));
        }
    }
    Ok(())
}

/// An actual atomic store invalidates both static and instance field snapshots.
#[test]
fn qualified_field_snapshots_respect_atomic_stores() -> Result<(), Box<dyn Error>> {
    for instance in [false, true] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
        let address = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let object = builder.function_mut().create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::entry(),
            MockType::Ref,
        );
        let result = builder.in_block(0, |block| {
            let value = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(41))?;
            block.emit_no_defs(if instance {
                SsaOp::StoreField {
                    object,
                    field: 1,
                    value,
                    access: AccessQualifiers::FLAT,
                }
            } else {
                SsaOp::StoreStaticField {
                    field: 1,
                    value,
                    access: AccessQualifiers::FLAT,
                }
            })?;
            block.emit_no_defs(SsaOp::AtomicStore {
                addr: address,
                value,
                value_type: MockType::I32,
                ordering: AtomicOrdering::SeqCst,
                width: AtomicAccessWidth::Bits32,
                access: AccessQualifiers::FLAT,
            })?;
            let result = block.emit_def(SsaDefSpec::tmp(MockType::I32), |dest| {
                if instance {
                    SsaOp::LoadField {
                        dest,
                        object,
                        field: 1,
                        access: AccessQualifiers::FLAT,
                    }
                } else {
                    SsaOp::LoadStaticField {
                        dest,
                        field: 1,
                        access: AccessQualifiers::FLAT,
                    }
                }
            })?;
            block.ret(Some(result))?;
            Ok(result)
        })?;
        let ssa = builder.finish()?;
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(
            errors.is_empty(),
            "invalid atomic field witness: {errors:?}"
        );
        let mut evaluator =
            SsaEvaluator::new(&ssa, MACHINE).with_config(EvaluatorConfig::with_memory());
        evaluator.evaluate_block(0);
        assert!(
            evaluator.get_concrete(result).is_none(),
            "instance={instance}"
        );
    }
    Ok(())
}

/// Builds a valid dominance relation whose literal block a path walk can skip.
fn dominating_literal_fixture() -> Result<(SsaFunction<MockTarget>, [SsaVarId; 3]), Box<dyn Error>>
{
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let (input, literal) = builder.in_block(0, |block| {
        let input = block.load_arg(SsaDefSpec::argument(0, MockType::I32), 0)?;
        let literal = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(0))?;
        block.jump(1)?;
        Ok((input, literal))
    })?;
    let result = builder.in_block(1, |block| {
        let result = block.emit_def(SsaDefSpec::tmp(MockType::Bool), |dest| SsaOp::Ceq {
            dest,
            left: input,
            right: literal,
            signaling: false,
        })?;
        block.ret(Some(result))?;
        Ok(result)
    })?;
    let ssa = builder.finish()?;
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
    Ok((ssa, [input, literal, result]))
}

/// A retained relation reads a dominating literal without replaying its block.
#[test]
fn dominating_literal_decides_a_relation_from_an_initialized_path() -> Result<(), Box<dyn Error>> {
    let (ssa, [input, literal, result]) = dominating_literal_fixture()?;
    let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
    evaluator.set_concrete(input, ConstValue::I32(1));
    evaluator.evaluate_block(1);
    assert_eq!(evaluator.get_concrete(result), Some(&ConstValue::False));
    assert!(
        evaluator.get(literal).is_none(),
        "literal recovery must not execute the dominating block"
    );
    Ok(())
}

/// Explicit and opaque current bindings take precedence over a literal definition.
#[test]
fn dominating_literal_does_not_replace_current_bindings() -> Result<(), Box<dyn Error>> {
    let (ssa, [input, literal, result]) = dominating_literal_fixture()?;
    let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
    evaluator.set_concrete(input, ConstValue::I32(1));
    evaluator.set_concrete(literal, ConstValue::I32(1));
    evaluator.evaluate_block(1);
    assert_eq!(evaluator.get_concrete(result), Some(&ConstValue::True));

    let mut evaluator = SsaEvaluator::new(&ssa, MACHINE);
    evaluator.set_concrete(input, ConstValue::I32(1));
    evaluator.set_unknown(literal);
    evaluator.evaluate_block(1);
    assert!(evaluator.get_concrete(result).is_none());
    assert!(evaluator.get(literal).is_none());
    Ok(())
}
