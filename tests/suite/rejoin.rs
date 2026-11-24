//! Rejoining a loop a compiler split on a guard into the loop it was split
//! from.
//!
//! The positive case runs the function before and after on every count it
//! could be asked about, so the rejoin is held to computing what the split
//! loops computed; each negative breaks one condition the rejoin rests on.

use analyssa::{
    analysis::{SsaEvaluator, SsaVerifier, VerifyLevel, context::AnalysisContext},
    events::NullListener,
    ir::{
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{AccessQualifiers, SsaOp, kinds::CmpKind},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::LoopRejoiningPass,
    scheduling::{FunctionPass, PassEnv},
    testing::{MACHINE, MockTarget, MockType},
};

/// Declares a variable defined at `(block, instr)`.
fn var(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        1,
        DefSite::instruction(block, instr),
        MockType::I32,
    )
}

/// Declares a variable a phi of `block` defines.
fn phi_var(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        1,
        DefSite::phi(block),
        MockType::I32,
    )
}

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

fn add(dest: SsaVarId, left: SsaVarId, right: SsaVarId) -> SsaInstruction<MockTarget> {
    instr(SsaOp::Add {
        dest,
        left,
        right,
        flags: None,
    })
}

fn constant(dest: SsaVarId, value: i32) -> SsaInstruction<MockTarget> {
    instr(SsaOp::Const {
        dest,
        value: ConstValue::I32(value),
    })
}

fn phi(result: SsaVarId, idx: u16, operands: &[(SsaVarId, usize)]) -> PhiNode {
    let mut node = PhiNode::new(result, VariableOrigin::Local(idx));
    for (value, from) in operands {
        node.add_operand(PhiOperand::new(*value, *from));
    }
    node
}

/// What breaks the split, if anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Breaks {
    /// Nothing: the second loop never touches the guard.
    Nothing,
    /// The second loop steps the guard's counter on, so the guard read again
    /// is another.
    Guard,
    /// The path to the second loop reads memory, which re-taken may read
    /// another value.
    Path,
}

/// `v_sum` as GCC splits its `va_arg` loop: a first loop reading the register
/// area while `gp <= 47`, whose test leaves for a second loop reading the
/// stack from `o`.
///
/// ```text
/// 0: goto 1
/// 1: s = phi(0: 0, 2: s1); g = phi(0: 8, 2: g1); i = phi(0: 0, 2: i1)
///    if (g <= 47) goto 2 else goto 3
/// 2: i1 = i + 1; g1 = g + 8; s1 = s + (area + g); if (i1 == n) goto 5 else goto 1
/// 3: goto 4
/// 4: o = phi(3: o0, 4: o1); t = phi(3: s, 4: t1); j = phi(3: i, 4: j1)
///    j1 = j + 1; o1 = o + 8; t1 = t + o; if (j1 != n) goto 4 else goto 5
/// 5: r = phi(2: s1, 4: t1); return r
/// ```
///
/// Returns the function with `n`, `area` and `o0`.
fn split(breaks: Breaks) -> (SsaFunction<MockTarget>, [SsaVarId; 3]) {
    let mut ssa = SsaFunction::new(3, 8);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let n = argument(&mut ssa, 0);
    let area = argument(&mut ssa, 1);
    let o0 = argument(&mut ssa, 2);
    let zero = var(&mut ssa, 0, 0, 0);
    let eight = var(&mut ssa, 1, 0, 1);
    let limit = var(&mut ssa, 2, 0, 2);
    let one = var(&mut ssa, 3, 0, 3);
    let s = phi_var(&mut ssa, 4, 1);
    let g = phi_var(&mut ssa, 5, 1);
    let i = phi_var(&mut ssa, 6, 1);
    let i1 = var(&mut ssa, 7, 2, 0);
    let g1 = var(&mut ssa, 8, 2, 1);
    let x = var(&mut ssa, 9, 2, 2);
    let s1 = var(&mut ssa, 10, 2, 3);
    let read = var(&mut ssa, 11, 3, 0);
    let o = phi_var(&mut ssa, 12, 4);
    let t = phi_var(&mut ssa, 13, 4);
    let j = phi_var(&mut ssa, 14, 4);
    let guard = phi_var(&mut ssa, 15, 4);
    let j1 = var(&mut ssa, 16, 4, 0);
    let o1 = var(&mut ssa, 17, 4, 1);
    let t1 = var(&mut ssa, 18, 4, 2);
    let guard1 = var(&mut ssa, 19, 4, 3);
    let r = phi_var(&mut ssa, 20, 5);

    let mut entry = SsaBlock::new(0);
    for (dest, value) in [(zero, 0), (eight, 8), (limit, 47), (one, 1)] {
        entry.add_instruction(constant(dest, value));
    }
    entry.add_instruction(instr(SsaOp::Jump { target: 1 }));

    let mut header = SsaBlock::new(1);
    header.add_phi(phi(s, 4, &[(zero, 0), (s1, 2)]));
    header.add_phi(phi(g, 5, &[(eight, 0), (g1, 2)]));
    header.add_phi(phi(i, 6, &[(zero, 0), (i1, 2)]));
    header.add_instruction(instr(SsaOp::BranchCmp {
        left: g,
        right: limit,
        cmp: CmpKind::Le,
        unsigned: true,
        true_target: 2,
        false_target: 3,
    }));

    let mut first = SsaBlock::new(2);
    first.add_instruction(add(i1, i, one));
    first.add_instruction(add(g1, g, eight));
    first.add_instruction(add(x, area, g));
    first.add_instruction(add(s1, s, x));
    first.add_instruction(instr(SsaOp::BranchCmp {
        left: i1,
        right: n,
        cmp: CmpKind::Eq,
        unsigned: false,
        true_target: 5,
        false_target: 1,
    }));

    let mut path = SsaBlock::new(3);
    if breaks == Breaks::Path {
        path.add_instruction(instr(SsaOp::LoadIndirect {
            dest: read,
            addr: area,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
    }
    path.add_instruction(instr(SsaOp::Jump { target: 4 }));

    let mut second = SsaBlock::new(4);
    second.add_phi(phi(o, 12, &[(o0, 3), (o1, 4)]));
    second.add_phi(phi(t, 13, &[(s, 3), (t1, 4)]));
    second.add_phi(phi(j, 14, &[(i, 3), (j1, 4)]));
    if breaks == Breaks::Guard {
        second.add_phi(phi(guard, 15, &[(g, 3), (guard1, 4)]));
    }
    second.add_instruction(add(j1, j, one));
    second.add_instruction(add(o1, o, eight));
    second.add_instruction(add(t1, t, o));
    if breaks == Breaks::Guard {
        second.add_instruction(add(guard1, guard, eight));
    }
    second.add_instruction(instr(SsaOp::BranchCmp {
        left: j1,
        right: n,
        cmp: CmpKind::Ne,
        unsigned: false,
        true_target: 4,
        false_target: 5,
    }));

    let mut exit = SsaBlock::new(5);
    exit.add_phi(phi(r, 20, &[(s1, 2), (t1, 4)]));
    exit.add_instruction(instr(SsaOp::returns(Some(r))));

    for block in [entry, header, first, path, second, exit] {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    (ssa, [n, area, o0])
}

/// What `ssa` returns for the count `count`, with the area at 1000 and the
/// stack at 5000: the value its `return` reads, which an edit may renumber.
fn returns(ssa: &SsaFunction<MockTarget>, inputs: [SsaVarId; 3], count: i32) -> Option<i64> {
    let mut eval = SsaEvaluator::new(ssa, MACHINE);
    for (input, value) in inputs.into_iter().zip([count, 1000, 5000]) {
        eval.set_concrete(input, ConstValue::I32(value));
    }
    let trace = eval.execute(0, None, 10_000);
    assert!(!trace.hit_limit(), "the run ends");
    let returned = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .find_map(|row| match row.op() {
            SsaOp::Return { value, .. } => *value,
            _ => None,
        })?;
    eval.get_concrete(returned).and_then(ConstValue::as_i64)
}

/// **A loop split where its guard turns is one loop again.** The second
/// loop's latch goes back to the first loop's header, which takes its values
/// through its phis, the stack pointer `o` among them, and leaves its test
/// as it was: one loop, and every count from one to twelve, past where the
/// guard turns, returns what the split loops returned.
#[test]
fn a_loop_split_where_its_guard_turns_is_one_loop_again() {
    let (mut ssa, inputs) = split(Breaks::Nothing);
    let before: Vec<Option<i64>> = (1..=12).map(|count| returns(&ssa, inputs, count)).collect();
    assert!(matches!(
        LoopRejoiningPass.run(
            &mut ssa,
            &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
        ),
        Ok(true)
    ));
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        errors.is_empty(),
        "the rejoined function verifies: {errors:?}"
    );
    let second = ssa.block(4);
    assert_eq!(
        second.map(|block| block.phi_nodes().len()),
        Some(0),
        "the second body's phis are the header's"
    );
    assert_eq!(
        second
            .and_then(SsaBlock::control_terminator)
            .map(SsaOp::successors),
        Some(vec![1, 5]),
        "the second body goes back to the header"
    );
    assert_eq!(ssa.block(1).map(|header| header.phi_nodes().len()), Some(4));
    let after: Vec<Option<i64>> = (1..=12).map(|count| returns(&ssa, inputs, count)).collect();
    assert_eq!(before, after);
    assert!(before.iter().all(Option::is_some));
}

/// **A guard the second loop changes, or a path that reads memory, keeps
/// the loops apart**: re-taken after a second trip, the test or the path
/// would compute something else.
#[test]
fn a_changed_guard_or_a_reading_path_keeps_the_loops_apart() {
    for breaks in [Breaks::Guard, Breaks::Path] {
        let (mut ssa, _) = split(breaks);
        assert!(
            matches!(
                LoopRejoiningPass.run(
                    &mut ssa,
                    &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                ),
                Ok(false)
            ),
            "{breaks:?}"
        );
    }
}
