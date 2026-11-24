//! Merging the two arms of a branch that compute the same thing.
//!
//! Each merged case runs the function before and after on every input it
//! could be asked about, so the merge is held to computing what either arm
//! computed; each kept case breaks one condition the merge rests on.

use analyssa::{
    analysis::{SsaEvaluator, SsaVerifier, VerifyLevel, context::AnalysisContext},
    events::NullListener,
    ir::{
        CallData, Callee,
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{SsaOp, kinds::CmpKind},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::ArmMergingPass,
    scheduling::{FunctionPass, PassEnv},
    testing::{FactfulTarget, MACHINE, MockSource, MockTarget, MockType},
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

fn phi(result: SsaVarId, idx: u16, operands: &[(SsaVarId, usize)]) -> PhiNode {
    let mut node = PhiNode::new(result, VariableOrigin::Local(idx));
    for (value, from) in operands {
        node.add_operand(PhiOperand::new(*value, *from));
    }
    node
}

fn branch(
    left: SsaVarId,
    cmp: CmpKind,
    right: SsaVarId,
    true_target: usize,
    false_target: usize,
) -> SsaInstruction<MockTarget> {
    instr(SsaOp::BranchCmp {
        left,
        right,
        cmp,
        unsigned: false,
        true_target,
        false_target,
    })
}

/// How the second arm is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Second {
    /// As the first: `x2 = a + b; if (x2 == n) goto 3 else goto 4`.
    Alike,
    /// Its test negated, its targets swapped: `if (x2 != n) goto 4 else
    /// goto 3`.
    Negated,
    /// Its test's operands swapped: `if (n == x2) goto 3 else goto 4`.
    Swapped,
    /// Another operation: `x2 = a - b`.
    OtherOperation,
    /// Another operand: `x2 = a + a`.
    OtherOperand,
    /// Another test: `if (x2 < n) goto 3 else goto 4`.
    OtherTest,
    /// Alike, but the block after takes `b` from it where it takes `x1` from
    /// the first.
    OtherPhi,
}

/// A test on `c` whose arms each add `a` and `b` and test the sum against
/// `n`.
///
/// ```text
/// 0: zero = 0; one = 1; if (c == zero) goto 1 else goto 2
/// 1: x1 = a + b; if (x1 == n) goto 3 else goto 4
/// 2: x2 = a + b; if (x2 == n) goto 3 else goto 4
/// 3: r = phi(1: x1, 2: x2); return r
/// 4: s = phi(1: x1, 2: x2); t = s + one; return t
/// ```
///
/// Returns the function with `c`, `a`, `b` and `n`.
fn arms(second: Second) -> (SsaFunction<MockTarget>, [SsaVarId; 4]) {
    let mut ssa = SsaFunction::new(4, 4);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let c = argument(&mut ssa, 0);
    let a = argument(&mut ssa, 1);
    let b = argument(&mut ssa, 2);
    let n = argument(&mut ssa, 3);
    let zero = var(&mut ssa, 0, 0, 0);
    let one = var(&mut ssa, 1, 0, 1);
    let x1 = var(&mut ssa, 2, 1, 0);
    let x2 = var(&mut ssa, 3, 2, 0);
    let r = phi_var(&mut ssa, 4, 3);
    let s = phi_var(&mut ssa, 5, 4);
    let t = var(&mut ssa, 6, 4, 0);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(branch(c, CmpKind::Eq, zero, 1, 2));

    let mut first = SsaBlock::new(1);
    first.add_instruction(instr(SsaOp::Add {
        dest: x1,
        left: a,
        right: b,
        flags: None,
    }));
    first.add_instruction(branch(x1, CmpKind::Eq, n, 3, 4));

    let mut other = SsaBlock::new(2);
    other.add_instruction(instr(match second {
        Second::OtherOperation => SsaOp::Sub {
            dest: x2,
            left: a,
            right: b,
            flags: None,
        },
        Second::OtherOperand => SsaOp::Add {
            dest: x2,
            left: a,
            right: a,
            flags: None,
        },
        _ => SsaOp::Add {
            dest: x2,
            left: a,
            right: b,
            flags: None,
        },
    }));
    other.add_instruction(match second {
        Second::Negated => branch(x2, CmpKind::Ne, n, 4, 3),
        Second::Swapped => branch(n, CmpKind::Eq, x2, 3, 4),
        Second::OtherTest => branch(x2, CmpKind::Lt, n, 3, 4),
        _ => branch(x2, CmpKind::Eq, n, 3, 4),
    });

    let mut equal = SsaBlock::new(3);
    let from_other = if second == Second::OtherPhi { b } else { x2 };
    equal.add_phi(phi(r, 4, &[(x1, 1), (from_other, 2)]));
    equal.add_instruction(instr(SsaOp::returns(Some(r))));

    let mut unequal = SsaBlock::new(4);
    unequal.add_phi(phi(s, 5, &[(x1, 1), (x2, 2)]));
    unequal.add_instruction(instr(SsaOp::Add {
        dest: t,
        left: s,
        right: one,
        flags: None,
    }));
    unequal.add_instruction(instr(SsaOp::returns(Some(t))));

    for block in [entry, first, other, equal, unequal] {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    (ssa, [c, a, b, n])
}

/// What `ssa` returns for `values` of its inputs: the value the `return` of
/// the block it ends in reads.
fn returns(ssa: &SsaFunction<MockTarget>, inputs: [SsaVarId; 4], values: [i32; 4]) -> Option<i64> {
    let mut eval = SsaEvaluator::new(ssa, MACHINE);
    for (input, value) in inputs.into_iter().zip(values) {
        eval.set_concrete(input, ConstValue::I32(value));
    }
    let trace = eval.execute(0, None, 1_000);
    assert!(!trace.hit_limit(), "the run ends");
    let returned = ssa
        .block(trace.last_block()?)?
        .instructions()
        .iter()
        .find_map(|row| match row.op() {
            SsaOp::Return { value, .. } => *value,
            _ => None,
        })?;
    eval.get_concrete(returned).and_then(ConstValue::as_i64)
}

/// Every input the functions are asked about: either arm, and a sum equal to
/// `n` or not.
fn inputs() -> Vec<[i32; 4]> {
    let mut all = Vec::new();
    for c in [0, 1] {
        for a in [-2, 0, 3] {
            for b in [-1, 0, 4] {
                for n in [-3, 0, 2, 3, 7] {
                    all.push([c, a, b, n]);
                }
            }
        }
    }
    all
}

/// **Arms that compute the same thing are one**: whether they leave by the
/// same test, its negation with the targets swapped or its operands swapped,
/// the test jumps to the first arm, the second goes, and every input returns
/// what it returned.
#[test]
fn arms_that_compute_the_same_thing_are_one() {
    for second in [Second::Alike, Second::Negated, Second::Swapped] {
        let (mut ssa, vars) = arms(second);
        let before: Vec<Option<i64>> = inputs()
            .into_iter()
            .map(|values| returns(&ssa, vars, values))
            .collect();
        assert!(
            matches!(
                ArmMergingPass.run(
                    &mut ssa,
                    &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                ),
                Ok(true)
            ),
            "{second:?}"
        );
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(errors.is_empty(), "{second:?} verifies: {errors:?}");
        assert!(
            matches!(
                ssa.block(0).and_then(SsaBlock::control_terminator),
                Some(SsaOp::Jump { target: 1 })
            ),
            "{second:?}: the test jumps to the first arm"
        );
        assert_eq!(
            ssa.block(2).map(SsaBlock::is_empty),
            Some(true),
            "{second:?}: the second arm goes"
        );
        let from_second = ssa
            .blocks()
            .iter()
            .flat_map(SsaBlock::phi_nodes)
            .any(|phi| phi.operand_from(2).is_some());
        assert!(!from_second, "{second:?}: no phi takes from the second arm");
        let after: Vec<Option<i64>> = inputs()
            .into_iter()
            .map(|values| returns(&ssa, vars, values))
            .collect();
        assert_eq!(before, after, "{second:?}");
        assert!(before.iter().all(Option::is_some), "{second:?}");
    }
}

/// **Arms that differ in one operation, one operand, their test or what the
/// block after takes from them stay apart**: the test chooses between them.
#[test]
fn arms_that_differ_stay_apart() {
    for second in [
        Second::OtherOperation,
        Second::OtherOperand,
        Second::OtherTest,
        Second::OtherPhi,
    ] {
        let (mut ssa, _) = arms(second);
        assert!(
            matches!(
                ArmMergingPass.run(
                    &mut ssa,
                    &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                ),
                Ok(false)
            ),
            "{second:?}"
        );
    }
}

/// **The arm kept stands for the arm merged into it.** A test of `g` picks
/// between two arms that each call service `7` with `g` and jump to one
/// return: the arms merge, and each row of the kept arm takes in where its
/// counterpart came from, so a host keeping a call site per row loses
/// neither.
#[test]
fn the_kept_arm_takes_in_where_the_merged_arm_came_from() {
    let mut ssa: SsaFunction<FactfulTarget> = SsaFunction::new(1, 1);
    let g = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let zero = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let row = |at: u64, op: SsaOp<FactfulTarget>| {
        let mut row = SsaInstruction::synthetic(op);
        row.set_facts(MockSource::at(at));
        row
    };
    let mut test = SsaBlock::new(0);
    test.add_instruction(row(
        0x00,
        SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        },
    ));
    test.add_instruction(row(
        0x04,
        SsaOp::BranchCmp {
            left: g,
            right: zero,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        },
    ));
    ssa.add_block(test);
    for (index, at) in [(1, 0x10), (2, 0x20)] {
        let mut arm = SsaBlock::new(index);
        arm.add_instruction(row(
            at,
            CallData::new(None, Callee::Direct(7), vec![g]).into(),
        ));
        arm.add_instruction(row(at + 4, SsaOp::Jump { target: 3 }));
        ssa.add_block(arm);
    }
    let mut exit = SsaBlock::new(3);
    exit.add_instruction(row(0x30, SsaOp::returns(None)));
    ssa.add_block(exit);
    ssa.recompute_uses();
    assert!(
        matches!(
            ArmMergingPass.run(
                &mut ssa,
                &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
            ),
            Ok(true)
        ),
        "{ssa}"
    );
    let twins = |at: u64| -> Vec<u64> {
        ssa.blocks()
            .iter()
            .flat_map(SsaBlock::instructions)
            .find(|row| row.facts().at == at)
            .map(|row| row.facts().twins.iter().copied().collect())
            .unwrap_or_default()
    };
    assert_eq!(twins(0x10), [0x20], "{ssa}");
    assert_eq!(twins(0x14), [0x24], "{ssa}");
    assert!(twins(0x30).is_empty(), "{ssa}");
}
