//! SSA reconstruction preserves what the function computes.
//!
//! A renaming `construct_ssa` renames every grouped variable and re-places
//! every phi. Its output is checked two ways: by the verifier, which catches
//! malformed SSA, and here by running the function before and after, which
//! catches SSA that is well-formed and computes something else - the failure
//! the verifier cannot see.

use std::process;

use analyssa::{
    analysis::{SsaEvaluator, SsaVerifier, VerifyLevel},
    ir::{
        CallData, Callee,
        block::SsaBlock,
        function::{SsaConstructOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    testing::{self, MACHINE, MockTarget, MockType},
};

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

fn ok_or_abort<T, E>(value: Result<T, E>) -> T {
    value.unwrap_or_else(|_| process::abort())
}

/// Runs the function from its entry and returns the value it returns.
fn returned(ssa: &SsaFunction<MockTarget>) -> Option<ConstValue<MockTarget>> {
    let mut evaluator = SsaEvaluator::new(ssa, MACHINE);
    let trace = evaluator.execute_from_entry(1_000);
    let last = trace.last_block()?;
    match ssa.block(last)?.control_terminator()? {
        SsaOp::Return {
            value: Some(value), ..
        } => evaluator.get_concrete(*value).cloned(),
        _ => None,
    }
}

/// Asserts the function verifies at `Full`.
fn assert_verifies(ssa: &SsaFunction<MockTarget>) {
    let errors = SsaVerifier::new(ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}\n{ssa}");
}

/// A loop that returns its counter's value from *before* the increment.
///
/// The counter phi and the increment are joined by the phi, so the rebuild
/// groups them, and both are live at the exit: the exit reads the phi after
/// the increment is defined. Renaming the exit's use to the group's newest
/// version returns the incremented value instead - one more than the loop
/// computes. And a loop phi whose back edge is renamed to the local's entry
/// value forgets its iterations entirely.
#[test]
fn a_rebuild_keeps_a_loop_and_the_value_it_returns() {
    let mut ssa = testing::loop_counter_fixture();
    let expected = returned(&ssa);
    assert_eq!(expected, Some(ConstValue::I32(3)), "the fixture itself");

    for round in 1..=3 {
        ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
        assert_verifies(&ssa);
        assert_eq!(returned(&ssa), expected, "after rebuild {round}:\n{ssa}");
    }
}

/// A phi with no operand for one of its incoming edges is not trivial.
///
/// A CFG edit that adds a predecessor leaves the phi short an operand until
/// the rebuild supplies it. Collapsing the phi onto the one operand it does
/// name would claim the new edge carries that value too - here, that the
/// loop's counter enters already incremented.
#[test]
fn a_phi_missing_an_edge_is_not_collapsed_onto_its_other_operand() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
    let counter = ssa.create_variable(VariableOrigin::Local(0), 1, DefSite::phi(1), MockType::I32);
    let limit = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(1, 0),
        MockType::I32,
    );
    let cond = ssa.create_variable(
        VariableOrigin::Local(2),
        0,
        DefSite::instruction(1, 1),
        MockType::I32,
    );
    let one = ssa.create_variable(
        VariableOrigin::Local(3),
        0,
        DefSite::instruction(2, 0),
        MockType::I32,
    );
    let next = ssa.create_variable(
        VariableOrigin::Local(0),
        2,
        DefSite::instruction(2, 1),
        MockType::I32,
    );

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);
    // The phi names the back edge only; B0's edge has no operand yet.
    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(counter, VariableOrigin::Local(0));
    phi.add_operand(PhiOperand::new(next, 2));
    b1.add_phi(phi);
    b1.add_instruction(instr(SsaOp::Const {
        dest: limit,
        value: ConstValue::I32(4),
    }));
    b1.add_instruction(instr(SsaOp::Clt {
        dest: cond,
        left: counter,
        right: limit,
        unsigned: false,
        signaling: false,
    }));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    b2.add_instruction(instr(SsaOp::Add {
        dest: next,
        left: counter,
        right: one,
        flags: None,
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b2);
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(Some(counter))));
    ssa.add_block(b3);
    ssa.recompute_uses();

    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
    assert_verifies(&ssa);

    let header = ssa.block(1).map(SsaBlock::phi_nodes).unwrap_or_default();
    assert_eq!(header.len(), 1, "the header keeps its phi\n{ssa}");
    let Some(phi) = header.first() else {
        return;
    };
    let mut preds: Vec<usize> = phi.operands().iter().map(PhiOperand::predecessor).collect();
    preds.sort_unstable();
    assert_eq!(preds, vec![0, 2], "one operand per incoming edge\n{ssa}");
    let returns_phi = matches!(
        ssa.block(3).and_then(SsaBlock::control_terminator),
        Some(SsaOp::Return { value: Some(v), .. }) if *v == phi.result()
    );
    assert!(
        returns_phi,
        "the exit returns the counter, not its increment\n{ssa}"
    );
}

/// A value read after a later member of its group is defined keeps its own
/// definition.
///
/// `a` and `b` are two versions of one local slot, so the rebuild groups them.
/// `a` is read after `b` is defined; renaming that read to the group's newest
/// version would read `b`.
#[test]
fn a_use_keeps_its_own_definition_when_a_newer_group_member_is_live() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let a = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let b = ssa.create_variable(
        VariableOrigin::Local(0),
        1,
        DefSite::instruction(0, 1),
        MockType::I32,
    );
    let sum = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 2),
        MockType::I32,
    );
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(5),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(40),
    }));
    b0.add_instruction(instr(SsaOp::Sub {
        dest: sum,
        left: b,
        right: a,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(sum))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    let expected = returned(&ssa);
    assert_eq!(expected, Some(ConstValue::I32(35)));
    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
    assert_verifies(&ssa);
    assert_eq!(returned(&ssa), expected, "{ssa}");
}

/// `u = add x, x; x = 5; ret u`, with `x` in no rename group.
fn ungrouped_read_before_its_definition() -> SsaFunction<MockTarget> {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
    let x = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 1),
        MockType::I32,
    );
    let u = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Add {
        dest: u,
        left: x,
        right: x,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: x,
        value: ConstValue::I32(5),
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(u))));
    ssa.add_block(b0);
    ssa.recompute_uses();
    ssa
}

/// A read above its own definition, of a value in no rename group, is left
/// for the verifier to report rather than given a value from nowhere.
///
/// With no group there is no other version of the value to read, so the only
/// "repair" is a fresh entry live-in - a value the caller never supplies. The
/// function then verifies at `Full` while computing something undefined, and
/// an edit session that relies on verification keeps it.
#[test]
fn repair_does_not_invent_a_value_for_an_ungrouped_read() {
    let mut ssa = ungrouped_read_before_its_definition();

    ssa.repair_ssa();

    assert!(
        ssa.variables()
            .iter()
            .all(|variable| variable.origin() != VariableOrigin::EntryLiveIn),
        "{ssa}"
    );
    assert!(
        !SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty(),
        "the read above its definition is still reported\n{ssa}"
    );

    let mut rebuilt = ungrouped_read_before_its_definition();
    assert!(
        rebuilt
            .construct_ssa(&SsaConstructOptions::renaming_only())
            .is_err(),
        "the rebuild reports what it cannot repair\n{rebuilt}"
    );
}

/// A read above its own definition is redirected only to a group member whose
/// definition dominates it.
///
/// `x` is read in B2 before B2 defines it. The group's other entry-less
/// member is the phi joining B1 and B2 in B3, which does not dominate B2:
/// reading it there reads a value from a block that has not run yet.
#[test]
fn repair_adopts_only_a_dominating_group_member() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
    let cond = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let a = ssa.create_variable(
        VariableOrigin::Phi,
        1,
        DefSite::instruction(1, 0),
        MockType::I32,
    );
    let u = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(2, 0),
        MockType::I32,
    );
    let x = ssa.create_variable(
        VariableOrigin::Phi,
        2,
        DefSite::instruction(2, 1),
        MockType::I32,
    );
    let merged = ssa.create_variable(VariableOrigin::Phi, 3, DefSite::phi(3), MockType::I32);
    let group = 7;
    for member in [a, x, merged] {
        ssa.set_rename_group(member, group);
    }

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::True,
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Add {
        dest: u,
        left: x,
        right: x,
        flags: None,
    }));
    b2.add_instruction(instr(SsaOp::Const {
        dest: x,
        value: ConstValue::I32(2),
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b2);
    let mut b3 = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
    phi.add_operand(PhiOperand::new(a, 1));
    phi.add_operand(PhiOperand::new(x, 2));
    b3.add_phi(phi);
    b3.add_instruction(instr(SsaOp::returns(Some(merged))));
    ssa.add_block(b3);
    ssa.recompute_uses();

    ssa.repair_ssa();

    let reads_in_b2: Vec<_> = ssa
        .block(2)
        .map(|block| block.instructions().iter().flat_map(|i| i.uses()).collect())
        .unwrap_or_default();
    let merged_now = ssa
        .block(3)
        .and_then(|block| block.phi_nodes().first())
        .map(PhiNode::result);
    assert!(
        merged_now.is_some_and(|merged| !reads_in_b2.contains(&merged)),
        "B2 reads the join's phi, which runs after it\n{ssa}"
    );
    assert_verifies(&ssa);
}

/// A group above the arguments and locals is a stack depth, not a variable:
/// the values pushed there are unrelated. Construction keeps a front end's
/// groups as given, so renaming an untyped one must give it no type rather
/// than the type of whatever the depth held first.
#[test]
fn a_stack_group_lends_no_type_to_an_untyped_member() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
    let wide = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 0),
        MockType::I64,
    );
    let untyped = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 1),
        MockType::Unknown,
    );
    ssa.set_rename_group(wide, 0);
    ssa.set_rename_group(untyped, 0);
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: wide,
        value: ConstValue::I64(7),
    }));
    b0.add_instruction(instr(SsaOp::from(CallData::new(
        Some(untyped),
        Callee::Direct(1),
        vec![wide],
    ))));
    b0.add_instruction(instr(SsaOp::returns(Some(untyped))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::unverified()));
    let returned_type = ssa
        .block(0)
        .and_then(SsaBlock::control_terminator)
        .and_then(|op| match op {
            SsaOp::Return {
                value: Some(value), ..
            } => ssa.variable(*value),
            _ => None,
        })
        .map(|variable| *variable.var_type());
    assert_eq!(returned_type, Some(MockType::Unknown), "{ssa}");
}

/// A version of local 0 defined at `site`.
fn l0(ssa: &mut SsaFunction<MockTarget>, version: u32, site: DefSite) -> SsaVarId {
    ssa.create_variable(VariableOrigin::Local(0), version, site, MockType::I32)
}

/// `B0: x = 5; x = 9; jump B2` / `B1` (no longer reached): `m = phi(x)` /
/// `B2: ret m`.
///
/// The read names a merge in a block an edit cut off, so the rebuild resolves
/// it as a read of local 0 at `B2`: whatever `B0` left there, the second
/// write. A rebuild that kept the first definition of a group per block and
/// split the rest off answered `5` - a program that assigns a local, uses it,
/// and assigns it again read the stale value after unflattening.
#[test]
fn the_last_write_in_a_block_is_what_it_leaves() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let first = l0(&mut ssa, 1, DefSite::instruction(0, 0));
    let second = l0(&mut ssa, 2, DefSite::instruction(0, 1));
    let merged = l0(&mut ssa, 3, DefSite::phi(1));
    for var in [first, second, merged] {
        ssa.set_rename_group(var, 0);
    }
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: first,
        value: ConstValue::I32(5),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: second,
        value: ConstValue::I32(9),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(0));
    phi.add_operand(PhiOperand::new(second, 0));
    b1.add_phi(phi);
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(Some(merged))));
    ssa.add_block(b2);
    ssa.recompute_uses();

    ok_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));
    assert_verifies(&ssa);
    assert_eq!(returned(&ssa), Some(ConstValue::I32(9)), "{ssa}");
}

/// A block table whose blocks do not record their own positions has no meaning
/// to construction, which addresses blocks by both, and it is refused with an
/// error, the function untouched, rather than indexing a set sized for the
/// positions with an id past it.
#[test]
fn construction_refuses_a_block_table_that_is_not_dense() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let value = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let mut block = SsaBlock::new(2);
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(value))));
    ssa.add_block(block);
    ssa.recompute_uses();
    let before = format!("{ssa:?}");

    let error = match ssa.construct_ssa(&SsaConstructOptions::renaming_only()) {
        Ok(_) => process::abort(),
        Err(error) => error.to_string(),
    };

    assert!(error.contains("dense block table"), "{error}");
    assert_eq!(format!("{ssa:?}"), before, "nothing was changed");
}
