//! Dataflow analysis tests: SCCP, liveness, reaching definitions.

use analyssa::{
    analysis::{
        cfg::SsaCfg,
        dataflow::{
            liveness::{LiveVariables, LivenessResult},
            reaching::ReachingDefinitions,
            sccp::{ConstantPropagation, ScalarValue, SccpResult},
            solver::DataFlowSolver,
        },
    },
    ir::{
        block::SsaBlock,
        exception::{BlockRange, SsaExceptionHandler},
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{self, MACHINE, MockTarget, MockType},
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

#[test]
fn sccp_propagates_simple_constants() {
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

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(MACHINE);
    let result = cp.analyze(&ssa, &cfg);

    assert!(result.is_constant(a));
    assert!(result.is_constant(b));
    assert!(result.is_constant(sum));
    assert_eq!(result.constant_value(sum).and_then(|c| c.as_i64()), Some(8));
    assert_eq!(result.constant_count(), 3);
}

#[test]
fn sccp_propagates_through_copies() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
    block.add_instruction(instr(SsaOp::returns(Some(b))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(MACHINE);
    let result = cp.analyze(&ssa, &cfg);

    assert!(result.is_constant(b));
    assert_eq!(result.constant_value(b).and_then(|c| c.as_i64()), Some(42));
}

#[test]
fn sccp_respects_control_flow_with_unexecutable_blocks() {
    let mut ssa = SsaFunction::new(0, 3);
    let cond = local(&mut ssa, 0, 0, 0);
    let val = local(&mut ssa, 1, 1, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(0), // Always false
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: val,
        value: ConstValue::I32(99),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(MACHINE);
    let result = cp.analyze(&ssa, &cfg);

    // SCCP may or may not prune the false branch; at minimum block 0 is
    // executable and the constants are known
    assert!(result.executable_blocks().any(|b| b == 0));
    assert!(result.constant_count() > 0);
}

#[test]
fn sccp_empty_function_is_handled() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    // Empty function has no blocks, so SCCP should not panic
    if ssa.block_count() > 0 {
        let cfg = SsaCfg::from_ssa(&ssa);
        let mut cp = ConstantPropagation::new(MACHINE);
        let result = cp.analyze(&ssa, &cfg);
        assert_eq!(result.constant_count(), 0);
    }
}

#[test]
fn scalar_value_enumeration_is_correct() {
    assert!(matches!(
        ScalarValue::<MockTarget>::Bottom,
        ScalarValue::Bottom
    ));
    let result = SccpResult::<MockTarget>::empty();
    assert!(result.constants().next().is_none());
    assert_eq!(result.constant_count(), 0);
}

#[test]
fn sccp_propagates_through_phi_in_diamond() {
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
        value: ConstValue::I32(10),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(20),
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

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(MACHINE);
    let result = cp.analyze(&ssa, &cfg);

    // The entry block branches on the constant 1, so only the true arm is
    // executable - that is the whole point of *conditional* constant
    // propagation. B2 is never entered, so `right` is never evaluated.
    assert!(result.executable_blocks().any(|b| b == 0));
    assert!(
        result.executable_blocks().any(|b| b == 1),
        "the taken arm must be executable"
    );
    assert!(
        !result.executable_blocks().any(|b| b == 2),
        "a constant-true branch must leave the false arm unreachable"
    );

    assert!(result.is_constant(left));
    assert_eq!(
        result.constant_value(left).and_then(|c| c.as_i64()),
        Some(10)
    );
    assert!(
        !result.is_constant(right),
        "`right` is defined in the pruned arm and is never evaluated"
    );

    // With one executable incoming edge, the merge resolves to that operand.
    assert_eq!(
        result.constant_value(merged).and_then(|c| c.as_i64()),
        Some(10),
        "the phi has a single executable operand, so it is that constant"
    );
}

#[test]
fn sccp_converges_in_loop_to_fixpoint() {
    let mut ssa = SsaFunction::new(0, 3);
    let init = local(&mut ssa, 0, 0, 0);
    let limit = local(&mut ssa, 1, 0, 1);
    let counter = phi_local(&mut ssa, 2, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: init,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: limit,
        value: ConstValue::I32(3),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let cond = local(&mut ssa, 3, 1, 0);
    let next = local(&mut ssa, 4, 1, 1);
    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(counter, VariableOrigin::Local(2));
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
        right: init,
        flags: None,
    }));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let mut cp = ConstantPropagation::new(MACHINE);
    let result = cp.analyze(&ssa, &cfg);

    assert!(result.is_constant(init));
    assert!(result.is_constant(limit));
    assert_eq!(
        result.constant_value(init).and_then(|c| c.as_i64()),
        Some(0)
    );
    assert_eq!(
        result.constant_value(limit).and_then(|c| c.as_i64()),
        Some(3)
    );
}

#[test]
fn live_variables_analysis_runs_without_panicking() {
    let mut ssa = SsaFunction::new(0, 3);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);
    let c = local(&mut ssa, 2, 1, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: a,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: c,
        value: ConstValue::I32(3),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(Some(c))));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let analysis = LiveVariables::new(&ssa);
    let solver = DataFlowSolver::new(analysis);
    let _result = solver.solve(&ssa, &cfg);
}

#[test]
fn liveness_result_empty_and_live_queries() {
    let empty = LivenessResult::new(0);
    assert!(empty.is_empty());

    let state = LivenessResult::new(10);
    assert!(!state.is_live(SsaVarId::from_index(0)));
}

#[test]
fn reaching_definitions_runs_without_panicking() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(a))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let analysis = ReachingDefinitions::new(&ssa);
    let solver = DataFlowSolver::new(analysis);
    let _result = solver.solve(&ssa, &cfg);
}

#[test]
fn reaching_definitions_empty_function_handled() {
    let ssa = SsaFunction::<MockTarget>::new(0, 0);
    let cfg = SsaCfg::from_ssa(&ssa);
    let analysis = ReachingDefinitions::new(&ssa);
    let solver = DataFlowSolver::new(analysis);
    let _result = solver.solve(&ssa, &cfg);
}

/// A definition reaches the entry block along a back edge into it.
///
/// The entry block's boundary value is where its in-state starts, not all it
/// is: a function whose first block is a loop header has a predecessor there,
/// and replacing that edge's contribution with the boundary would report the
/// loop's own definitions as not reaching its header.
#[test]
fn reaching_definitions_cross_a_back_edge_into_the_entry() {
    let mut ssa = SsaFunction::new(0, 2);
    let cond = local(&mut ssa, 0, 1, 0);
    // Version 1: the boundary already holds every version-0 local, so only the
    // back edge can deliver this definition to the entry.
    let x = ssa.create_variable(
        VariableOrigin::Local(1),
        1,
        DefSite::instruction(1, 1),
        MockType::I32,
    );

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    b1.add_instruction(instr(SsaOp::Const {
        dest: x,
        value: ConstValue::I32(7),
    }));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 0,
        false_target: 2,
    }));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b2);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let results = DataFlowSolver::new(ReachingDefinitions::new(&ssa)).solve(&ssa, &cfg);
    let entry_in = results.in_state(0).map(|state| state.reaches(x));
    assert_eq!(
        entry_in,
        Some(true),
        "x is defined in B1 and B1 jumps back to B0"
    );
}

/// Builds `B0: c = value; <terminator(c)>` followed by `arms` returning blocks,
/// runs SCCP, and returns which of the arm blocks it found executable.
fn executable_arms(
    value: ConstValue<MockTarget>,
    arms: usize,
    terminator: impl FnOnce(SsaVarId) -> SsaOp<MockTarget>,
) -> Vec<bool> {
    let mut ssa = SsaFunction::new(0, 1);
    // Untyped, so the constant's own width is never at odds with the slot's.
    let cond = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::Unknown,
    );
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const { dest: cond, value }));
    b0.add_instruction(instr(terminator(cond)));
    ssa.add_block(b0);
    for arm in 1..=arms {
        let mut block = SsaBlock::new(arm);
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
    }
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let result = ConstantPropagation::new(MACHINE).analyze(&ssa, &cfg);
    (1..=arms)
        .map(|arm| result.is_block_executable(arm))
        .collect()
}

fn branch(condition: SsaVarId) -> SsaOp<MockTarget> {
    SsaOp::Branch {
        condition,
        true_target: 1,
        false_target: 2,
    }
}

fn switch(value: SsaVarId) -> SsaOp<MockTarget> {
    SsaOp::Switch {
        value,
        targets: vec![1, 2],
        default: 3,
    }
}

/// A constant branch condition takes exactly the arm its truth value names;
/// one with no truth value (a float) leaves both arms live.
#[test]
fn sccp_branch_on_a_constant_follows_its_truth_value() {
    assert_eq!(
        executable_arms(ConstValue::NativeInt(1), 2, branch),
        [true, false]
    );
    assert_eq!(
        executable_arms(ConstValue::NativeUInt(0), 2, branch),
        [false, true]
    );
    assert_eq!(
        executable_arms(ConstValue::from(1.0f64), 2, branch),
        [true, true]
    );
}

/// A constant switch selector of any integer type picks its arm; a valid
/// integer past the table picks the default; a non-integer leaves every arm
/// live.
#[test]
fn sccp_switch_on_a_constant_selects_by_integer_value() {
    assert_eq!(
        executable_arms(ConstValue::U32(1), 3, switch),
        [false, true, false]
    );
    assert_eq!(
        executable_arms(ConstValue::I64(0), 3, switch),
        [true, false, false]
    );
    assert_eq!(
        executable_arms(ConstValue::NativeUInt(1), 3, switch),
        [false, true, false]
    );
    assert_eq!(
        executable_arms(ConstValue::U64(7), 3, switch),
        [false, false, true]
    );
    assert_eq!(
        executable_arms(ConstValue::I32(-1), 3, switch),
        [false, false, true]
    );
    assert_eq!(
        executable_arms(ConstValue::from(1.0f64), 3, switch),
        [true, true, true]
    );
}

/// A protected block that ends in a jump can still throw, so its handler is
/// executable.
#[test]
fn sccp_marks_the_handler_of_a_jumping_protected_block_executable() {
    let ssa = testing::try_catch_fixture();
    let cfg = SsaCfg::from_ssa(&ssa);
    let result = ConstantPropagation::new(MACHINE).analyze(&ssa, &cfg);
    assert!(result.is_block_executable(1));
    assert!(result.is_block_executable(2), "the handler is reachable");
    assert!(result.is_block_executable(3));
}

/// A value the handler reads is live across every block of the protected
/// region, including one that returns: a throw from any of them lands in the
/// handler.
///
/// ```text
/// B0: v = 1; jump B1          try [1, 3), handler B3
/// B1: jump B2                 (guard: the exceptional edge B1 -> B3)
/// B2: return                  (protected, and an exit)
/// B3: return v                (handler)
/// ```
#[test]
fn a_value_the_handler_reads_is_live_through_the_protected_region() {
    let mut ssa = SsaFunction::new(0, 1);
    let v = local(&mut ssa, 0, 0, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: v,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b2);
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(Some(v))));
    ssa.add_block(b3);
    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(1, 3),
        handler_range: BlockRange::new(3, 4),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();

    let cfg = SsaCfg::from_ssa(&ssa);
    let results = DataFlowSolver::new(LiveVariables::new(&ssa)).solve(&ssa, &cfg);
    for block in [1, 2] {
        assert!(
            results.out_state(block).is_some_and(|out| out.is_live(v)),
            "v is live out of protected block {block}"
        );
        assert!(
            results.in_state(block).is_some_and(|live| live.is_live(v)),
            "v is live into protected block {block}"
        );
    }
    assert!(results.out_state(3).is_some_and(|out| !out.is_live(v)));
}
