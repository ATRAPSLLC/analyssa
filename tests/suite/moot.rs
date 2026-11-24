//! Dropping a test whose outcome changes nothing.
//!
//! The function chooses, by a test of a value nothing else reads, between two
//! loops; each case dropped runs the function before and after on every input
//! it could be asked about, so the drop is held to computing what either loop
//! computed, and each kept case breaks one thing the loops shared.

use std::mem;

use analyssa::{
    Result,
    analysis::{
        SsaEvaluator, SsaVerifier, VerifyLevel,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
    },
    events::NullListener,
    interrupt::{Interrupt, NeverInterrupted},
    ir::{
        CallData, Callee, SsaBlockBuilder, SsaDefSpec, SsaFunctionBuilder,
        block::SsaBlock,
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{AccessQualifiers, SsaOp, kinds::CmpKind},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::MootTestPass,
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

fn add(dest: SsaVarId, left: SsaVarId, right: SsaVarId) -> SsaInstruction<MockTarget> {
    instr(SsaOp::Add {
        dest,
        left,
        right,
        flags: None,
    })
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

/// How the second loop is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Second {
    /// As the first: `j1 = j + 1; t1 = t + j1; field = t1; if (j1 != n)
    /// loop`.
    Alike,
    /// Its rows in another order, and its test negated with its targets
    /// swapped: `t1 = t + (j + 1)` computed after the step, `if (j1 == n)
    /// leave`.
    Reordered,
    /// Another sum: `t1 = t + j`.
    OtherSum,
    /// The tested value read: `t1 = t + g`.
    ReadsTheTest,
    /// No store.
    NoStore,
}

/// A test of `g` choosing between two loops summing `1..=n`, each storing
/// its running sum to a static field.
///
/// ```text
/// 0: zero = 0; if (g == zero) goto 1 else goto 2
/// 1: s = phi(0: zero, 1: s1); i = phi(0: zero, 1: i1)
///    i1 = i + one; s1 = s + i1; field = s1; if (i1 != n) goto 1 else goto 3
/// 2: t = phi(0: zero, 2: t1); j = phi(0: zero, 2: j1)
///    j1 = j + one; t1 = t + j1; field = t1; if (j1 != n) goto 2 else goto 4
/// 3: return s1
/// 4: return t1
/// ```
///
/// Without `stored`, neither loop stores. Returns the function with `g` and
/// `n`.
fn loops(second: Second, stored: bool) -> (SsaFunction<MockTarget>, [SsaVarId; 2]) {
    let mut ssa = SsaFunction::new(2, 4);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let g = argument(&mut ssa, 0);
    let n = argument(&mut ssa, 1);
    let zero = var(&mut ssa, 0, 0, 0);
    let one = var(&mut ssa, 1, 0, 1);
    let s = phi_var(&mut ssa, 2, 1);
    let i = phi_var(&mut ssa, 3, 1);
    let i1 = var(&mut ssa, 4, 1, 0);
    let s1 = var(&mut ssa, 5, 1, 1);
    let t = phi_var(&mut ssa, 6, 2);
    let j = phi_var(&mut ssa, 7, 2);
    let j1 = var(&mut ssa, 8, 2, 0);
    let t1 = var(&mut ssa, 9, 2, 1);
    let jj = var(&mut ssa, 10, 2, 2);

    let mut entry = SsaBlock::new(0);
    for (dest, value) in [(zero, 0), (one, 1)] {
        entry.add_instruction(instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        }));
    }
    entry.add_instruction(branch(g, CmpKind::Eq, zero, 1, 2));

    let store = |value: SsaVarId| {
        instr(SsaOp::StoreStaticField {
            field: 1,
            value,
            access: AccessQualifiers::FLAT,
        })
    };

    let mut first = SsaBlock::new(1);
    first.add_phi(phi(s, 2, &[(zero, 0), (s1, 1)]));
    first.add_phi(phi(i, 3, &[(zero, 0), (i1, 1)]));
    first.add_instruction(add(i1, i, one));
    first.add_instruction(add(s1, s, i1));
    if stored {
        first.add_instruction(store(s1));
    }
    first.add_instruction(branch(i1, CmpKind::Ne, n, 1, 3));

    let mut other = SsaBlock::new(2);
    other.add_phi(phi(t, 6, &[(zero, 0), (t1, 2)]));
    other.add_phi(phi(j, 7, &[(zero, 0), (j1, 2)]));
    match second {
        Second::Reordered => {
            other.add_instruction(add(jj, j, one));
            other.add_instruction(add(t1, t, jj));
            other.add_instruction(add(j1, j, one));
        }
        Second::OtherSum => {
            other.add_instruction(add(j1, j, one));
            other.add_instruction(add(t1, t, j));
        }
        Second::ReadsTheTest => {
            other.add_instruction(add(j1, j, one));
            other.add_instruction(add(t1, t, g));
        }
        Second::Alike | Second::NoStore => {
            other.add_instruction(add(j1, j, one));
            other.add_instruction(add(t1, t, j1));
        }
    }
    if stored && second != Second::NoStore {
        other.add_instruction(store(t1));
    }
    other.add_instruction(if second == Second::Reordered {
        branch(j1, CmpKind::Eq, n, 4, 2)
    } else {
        branch(j1, CmpKind::Ne, n, 2, 4)
    });

    let mut left_first = SsaBlock::new(3);
    left_first.add_instruction(instr(SsaOp::returns(Some(s1))));
    let mut left_other = SsaBlock::new(4);
    left_other.add_instruction(instr(SsaOp::returns(Some(t1))));

    for block in [entry, first, other, left_first, left_other] {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    (ssa, [g, n])
}

/// What `ssa` returns for `g` and `n`: the value the `return` of the block
/// it ends in reads.
fn returns(ssa: &SsaFunction<MockTarget>, inputs: [SsaVarId; 2], g: i32, n: i32) -> Option<i64> {
    let mut eval = SsaEvaluator::new(ssa, MACHINE);
    for (input, value) in inputs.into_iter().zip([g, n]) {
        eval.set_concrete(input, ConstValue::I32(value));
    }
    let trace = eval.execute(0, None, 10_000);
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

/// What every input returns: either loop, counts one to six.
fn outcomes(ssa: &SsaFunction<MockTarget>, inputs: [SsaVarId; 2]) -> Vec<Option<i64>> {
    let mut all = Vec::new();
    for g in [0, 1, 7] {
        for n in 1..=6 {
            all.push(returns(ssa, inputs, g, n));
        }
    }
    all
}

/// **A test between two loops that do the same is no test**: whether the
/// second is written as the first or in another order with its test negated,
/// the entry jumps to one loop, the other goes, and every input returns what
/// it returned. With a store on each trip, the stores are the same in the same
/// order, and the test goes too.
#[test]
fn a_test_between_loops_that_do_the_same_is_no_test() {
    for (second, stored) in [
        (Second::Alike, false),
        (Second::Reordered, false),
        (Second::Alike, true),
        (Second::Reordered, true),
    ] {
        let (mut ssa, inputs) = loops(second, stored);
        let before = outcomes(&ssa, inputs);
        assert!(
            matches!(
                MootTestPass.run(
                    &mut ssa,
                    &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                        .with_interrupt(&NeverInterrupted)
                ),
                Ok(true)
            ),
            "{second:?} {stored}"
        );
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(
            errors.is_empty(),
            "{second:?} {stored} verifies: {errors:?}"
        );
        assert!(
            matches!(
                ssa.block(0).and_then(SsaBlock::control_terminator),
                Some(SsaOp::Jump { .. })
            ),
            "{second:?} {stored}: the entry jumps"
        );
        // The evaluator stops at a store, so the stored loops are held to
        // their shape and the others to what they return.
        if !stored {
            assert_eq!(before, outcomes(&ssa, inputs), "{second:?}");
            assert!(before.iter().all(Option::is_some), "{second:?}");
        }
    }
}

/// An interrupt that is always tripped.
struct AlwaysInterrupted;

impl Interrupt for AlwaysInterrupted {
    fn interrupted(&self) -> bool {
        true
    }
}

/// **A stopped search drops nothing**: the test between two loops that do
/// the same goes when the search runs, and stays when its interrupt has
/// tripped, which proves nothing rather than deciding on part of a search.
#[test]
fn a_stopped_search_drops_nothing() {
    let (mut ssa, _) = loops(Second::Alike, false);
    assert!(
        matches!(
            MootTestPass.run(
                &mut ssa,
                &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                    .with_interrupt(&AlwaysInterrupted)
            ),
            Ok(false)
        ),
        "a stopped search drops no test"
    );
    assert!(
        matches!(
            ssa.block(0).and_then(SsaBlock::control_terminator),
            Some(SsaOp::BranchCmp { .. })
        ),
        "the entry still tests"
    );
    assert!(
        matches!(
            MootTestPass.run(
                &mut ssa,
                &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                    .with_interrupt(&NeverInterrupted)
            ),
            Ok(true)
        ),
        "run to the end, the same search drops it"
    );
}

/// **Loops that differ in what they sum, in reading the tested value or in
/// storing stay apart**: the test chooses between them.
#[test]
fn a_test_between_loops_that_differ_stays() {
    for (second, stored) in [
        (Second::OtherSum, false),
        (Second::ReadsTheTest, false),
        (Second::NoStore, true),
    ] {
        let (mut ssa, _) = loops(second, stored);
        assert!(
            matches!(
                MootTestPass.run(
                    &mut ssa,
                    &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                        .with_interrupt(&NeverInterrupted)
                ),
                Ok(false)
            ),
            "{second:?}"
        );
    }
}

/// **A store only one loop makes is never left behind at a join**: an effect
/// is matched only with the other side's, so the loop that stores each trip
/// waits at its store while the other goes round, and the two never meet at
/// their heads with the store outstanding. Whichever arm the store is on, the
/// test stays.
#[test]
fn a_store_only_one_loop_makes_keeps_the_test() {
    for swapped in [false, true] {
        let (mut ssa, _) = loops(Second::NoStore, true);
        if swapped {
            let mut turned = false;
            if let Some(SsaOp::BranchCmp {
                true_target,
                false_target,
                ..
            }) = ssa
                .block_mut(0)
                .and_then(|block| block.instructions_mut().last_mut())
                .map(SsaInstruction::op_mut)
            {
                mem::swap(true_target, false_target);
                turned = true;
            }
            assert!(turned, "the entry's arms are swapped");
        }
        assert!(
            matches!(
                MootTestPass.run(
                    &mut ssa,
                    &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                        .with_interrupt(&NeverInterrupted)
                ),
                Ok(false)
            ),
            "store on the {} arm",
            if swapped { "second" } else { "first" }
        );
        assert!(
            matches!(
                ssa.block(0).and_then(SsaBlock::control_terminator),
                Some(SsaOp::BranchCmp { .. })
            ),
            "the entry still tests"
        );
    }
}

/// **The search stops where its budget does**: with none left, nothing is
/// proven and the test stays.
#[test]
fn a_spent_budget_proves_nothing() {
    let (mut ssa, _) = loops(Second::Alike, false);
    assert!(matches!(
        MootTestPass.run(
            &mut ssa,
            &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 0))
                .with_interrupt(&NeverInterrupted)
        ),
        Ok(false)
    ));
}

/// A loop whose header tests `g` each trip, going on in its own body or
/// leaving for a copy of the loop that goes on with what it counted.
///
/// ```text
/// 0: zero = 0; one = 1; goto 1
/// 1: s = phi(0: zero, 2: s1); i = phi(0: zero, 2: i1)
///    if (g == zero) goto 3 else goto 2
/// 2: i1 = i + one; s1 = s + i1; if (i1 != n) goto 1 else goto 4
/// 3: t = phi(1: s, 3: t1); j = phi(1: i, 3: j1)
///    j1 = j + one; t1 = t + j1; if (j1 != n) goto 3 else goto 5
/// 4: return s1
/// 5: return t1
/// ```
///
/// With `other_step`, the copy steps by two. Returns the function with `g`
/// and `n`.
fn split_by_a_test(other_step: bool) -> (SsaFunction<MockTarget>, [SsaVarId; 2]) {
    let mut ssa = SsaFunction::new(2, 4);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let g = argument(&mut ssa, 0);
    let n = argument(&mut ssa, 1);
    let zero = var(&mut ssa, 0, 0, 0);
    let one = var(&mut ssa, 1, 0, 1);
    let two = var(&mut ssa, 2, 0, 2);
    let s = phi_var(&mut ssa, 3, 1);
    let i = phi_var(&mut ssa, 4, 1);
    let i1 = var(&mut ssa, 5, 2, 0);
    let s1 = var(&mut ssa, 6, 2, 1);
    let t = phi_var(&mut ssa, 7, 3);
    let j = phi_var(&mut ssa, 8, 3);
    let j1 = var(&mut ssa, 9, 3, 0);
    let t1 = var(&mut ssa, 10, 3, 1);

    let mut entry = SsaBlock::new(0);
    for (dest, value) in [(zero, 0), (one, 1), (two, 2)] {
        entry.add_instruction(instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        }));
    }
    entry.add_instruction(instr(SsaOp::Jump { target: 1 }));

    let mut header = SsaBlock::new(1);
    header.add_phi(phi(s, 3, &[(zero, 0), (s1, 2)]));
    header.add_phi(phi(i, 4, &[(zero, 0), (i1, 2)]));
    header.add_instruction(branch(g, CmpKind::Eq, zero, 3, 2));

    let mut body = SsaBlock::new(2);
    body.add_instruction(add(i1, i, one));
    body.add_instruction(add(s1, s, i1));
    body.add_instruction(branch(i1, CmpKind::Ne, n, 1, 4));

    let mut copy = SsaBlock::new(3);
    copy.add_phi(phi(t, 7, &[(s, 1), (t1, 3)]));
    copy.add_phi(phi(j, 8, &[(i, 1), (j1, 3)]));
    copy.add_instruction(add(j1, j, if other_step { two } else { one }));
    copy.add_instruction(add(t1, t, j1));
    copy.add_instruction(branch(j1, CmpKind::Ne, n, 3, 5));

    let mut left_body = SsaBlock::new(4);
    left_body.add_instruction(instr(SsaOp::returns(Some(s1))));
    let mut left_copy = SsaBlock::new(5);
    left_copy.add_instruction(instr(SsaOp::returns(Some(t1))));

    for block in [entry, header, body, copy, left_body, left_copy] {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    (ssa, [g, n])
}

/// **A loop a test sends into a copy of itself is one loop**: the header's
/// test goes, its own values held alike on both ways out of it, and every
/// input returns what it returned; a copy that steps by two keeps the test.
#[test]
fn a_loop_a_test_sends_into_its_copy_is_one_loop() {
    let (mut ssa, inputs) = split_by_a_test(false);
    let before: Vec<Option<i64>> = [0, 1]
        .into_iter()
        .flat_map(|g| (1..=6).map(move |n| (g, n)))
        .map(|(g, n)| returns(&ssa, inputs, g, n))
        .collect();
    assert!(matches!(
        MootTestPass.run(
            &mut ssa,
            &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                .with_interrupt(&NeverInterrupted)
        ),
        Ok(true)
    ));
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "verifies: {errors:?}");
    assert!(
        matches!(
            ssa.block(1).and_then(SsaBlock::control_terminator),
            Some(SsaOp::Jump { .. })
        ),
        "the header jumps"
    );
    let after: Vec<Option<i64>> = [0, 1]
        .into_iter()
        .flat_map(|g| (1..=6).map(move |n| (g, n)))
        .map(|(g, n)| returns(&ssa, inputs, g, n))
        .collect();
    assert_eq!(before, after);
    assert!(before.iter().all(Option::is_some));

    let (mut ssa, _) = split_by_a_test(true);
    assert!(matches!(
        MootTestPass.run(
            &mut ssa,
            &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                .with_interrupt(&NeverInterrupted)
        ),
        Ok(false)
    ));
}

/// A `va_list` walk GCC's `-O2` split by the list's count, once each read is
/// the `va_arg` it is: `v_pick(n, ap)` reads `n + 1` arguments and returns the
/// last. A register loop tests the count stepped before it reads and the
/// count again after, a read between the files leaves for a stack loop, and
/// the entry chooses by the count's sign, each loop's counter entering from a
/// zero of its own; every way reads one argument a trip and leaves with the
/// last read once the trip counter reaches `n + 1`. The
/// stack loop reads each argument as `stacked`.
fn split_walk(stacked: MockType) -> Result<SsaFunction<MockTarget>> {
    let i32d = || SsaDefSpec::<MockTarget>::tmp(MockType::I32);
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
    let (list, n, count_at) = builder.in_block(0, |b| {
        let list = b.load_arg(SsaDefSpec::tmp(MockType::Ptr), 0)?;
        let n = b.load_arg(i32d(), 1)?;
        let count_at = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: list,
            index: None,
            stride: 0,
            offset: 24,
            result_type: MockType::Ptr,
        })?;
        let zero = b.const_i32(i32d(), 0)?;
        b.branch_cmp(n, zero, CmpKind::Lt, false, 10, 1)?;
        Ok((list, n, count_at))
    })?;
    let read = |b: &mut SsaBlockBuilder<'_, MockTarget>, ty: MockType| {
        b.emit_def(i32d(), |dest| SsaOp::ArgNext { dest, list, ty })
    };
    // Each loop's counter enters from a zero of its own.
    let (last, zero, stack_zero) = builder.in_block(1, |b| {
        let one = b.const_i32(i32d(), 1)?;
        let last = b.add(i32d(), n, one)?;
        let zero = b.const_i32(i32d(), 0)?;
        let stack_zero = b.const_i32(i32d(), 0)?;
        let count = b.load_indirect(i32d(), count_at, MockType::I32)?;
        b.branch_cmp(count, zero, CmpKind::Ge, false, 9, 2)?;
        Ok((last, zero, stack_zero))
    })?;
    let trips = builder.in_block(2, |b| b.empty_phi(i32d()))?;
    let stepped = builder.in_block(2, |b| {
        let one = b.const_i32(i32d(), 1)?;
        let stepped = b.add(i32d(), trips, one)?;
        let count = b.load_indirect(i32d(), count_at, MockType::I32)?;
        let eight = b.const_i32(i32d(), 8)?;
        let moved = b.add(i32d(), count, eight)?;
        let zero = b.const_i32(i32d(), 0)?;
        b.branch_cmp(moved, zero, CmpKind::Le, false, 7, 3)?;
        Ok(stepped)
    })?;
    let turning = builder.in_block(3, |b| {
        let value = read(b, MockType::I32)?;
        b.branch_cmp(last, stepped, CmpKind::Eq, false, 5, 11)?;
        Ok(value)
    })?;
    let stack = builder.in_block(4, |b| b.empty_phi(i32d()))?;
    let (stacked, stack_next) = builder.in_block(4, |b| {
        let one = b.const_i32(i32d(), 1)?;
        let next = b.add(i32d(), stack, one)?;
        let value = read(b, stacked)?;
        b.branch_cmp(last, next, CmpKind::Ne, false, 4, 5)?;
        Ok((value, next))
    })?;
    let left = builder.in_block(5, |b| {
        let left = b.phi(i32d(), [(3, turning), (4, stacked)])?;
        b.jump(6)?;
        Ok(left)
    })?;
    let saved = builder.in_block(7, |b| {
        let value = read(b, MockType::I32)?;
        b.branch_cmp(last, stepped, CmpKind::Eq, false, 6, 8)?;
        Ok(value)
    })?;
    builder.in_block(6, |b| {
        let returned = b.phi(i32d(), [(5, left), (7, saved)])?;
        b.ret(Some(returned))
    })?;
    builder.in_block(8, |b| {
        let count = b.load_indirect(i32d(), count_at, MockType::I32)?;
        let zero = b.const_i32(i32d(), 0)?;
        b.branch_cmp(count, zero, CmpKind::Lt, false, 2, 9)
    })?;
    let entering = builder.in_block(9, |b| {
        let entering = b.phi(i32d(), [(1, stack_zero), (8, stepped)])?;
        b.jump(11)?;
        Ok(entering)
    })?;
    builder.in_block(10, |b| {
        let none = b.const_i32(i32d(), 0)?;
        b.ret(Some(none))
    })?;
    let bridged = builder.in_block(11, |b| {
        let bridged = b.phi(i32d(), [(3, stepped), (9, entering)])?;
        b.jump(4)?;
        Ok(bridged)
    })?;
    builder.add_phi_operand(2, trips, 1, zero)?;
    builder.add_phi_operand(2, trips, 8, stepped)?;
    builder.add_phi_operand(4, stack, 11, bridged)?;
    builder.add_phi_operand(4, stack, 4, stack_next)?;
    builder.finish()
}

/// **A `va_list` walk split by the list's count is one walk**: every test of
/// the count chooses between copies of one trip, a read and the trip
/// counter's test, so each goes and one loop reads the list. The control: a
/// stack loop reading another type than the register loop is another walk,
/// and the tests choosing it stay.
#[test]
fn a_walk_split_by_its_count_is_one_walk() -> Result<()> {
    for (stacked, merged) in [(MockType::I32, true), (MockType::I64, false)] {
        let mut ssa = split_walk(stacked)?;
        let dropped = MootTestPass.run(
            &mut ssa,
            &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 1 << 14))
                .with_interrupt(&NeverInterrupted),
        );
        assert!(dropped.is_ok(), "{stacked:?}: {dropped:?}");
        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(errors.is_empty(), "{stacked:?} verifies: {errors:?}");
        let reads = ssa
            .blocks()
            .iter()
            .flat_map(SsaBlock::instructions)
            .filter(|row| matches!(row.op(), SsaOp::ArgNext { .. }))
            .count();
        if merged {
            assert_eq!(reads, 1, "{ssa}");
        } else {
            assert_eq!(reads, 3, "{ssa}");
        }
    }
    Ok(())
}

/// **The arm a moot test keeps stands for the one that goes.** A test of `g`
/// picks between two arms that each call service `7` with `g` and return
/// nothing, as an `int 2Eh` and a `syscall` enter one system service: the
/// test goes, and the kept arm's call and return stand for the dropped arm's
/// as twins, apart from any computation they absorbed, so a host keeping a
/// call site per row loses neither. The
/// test row's own facts stay its own.
#[test]
fn the_kept_arm_takes_in_where_the_dropped_arm_came_from() {
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
    let rows = |at: u64, op: SsaOp<FactfulTarget>| {
        let mut row = SsaInstruction::synthetic(op);
        row.set_facts(MockSource::at(at));
        row
    };
    let mut test = SsaBlock::new(0);
    test.add_instruction(rows(
        0x00,
        SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        },
    ));
    test.add_instruction(rows(
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
        arm.add_instruction(rows(
            at,
            CallData::new(None, Callee::Direct(7), vec![g]).into(),
        ));
        arm.add_instruction(rows(at + 4, SsaOp::returns(None)));
        ssa.add_block(arm);
    }
    ssa.recompute_uses();
    assert!(
        matches!(
            MootTestPass.run(
                &mut ssa,
                &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                    .with_interrupt(&NeverInterrupted)
            ),
            Ok(true)
        ),
        "{ssa}"
    );
    let facts: Vec<MockSource> = ssa
        .blocks()
        .iter()
        .flat_map(SsaBlock::instructions)
        .map(|row| row.facts().clone())
        .collect();
    let twins = |at: u64| -> Vec<u64> {
        facts
            .iter()
            .find(|facts| facts.at == at)
            .map(|facts| facts.twins.iter().copied().collect())
            .unwrap_or_default()
    };
    assert_eq!(twins(0x10), [0x20], "{ssa}");
    assert!(
        facts.iter().all(|facts| facts.absorbed.is_empty()),
        "a twin is no absorbed computation: {ssa}"
    );
    assert_eq!(twins(0x14), [0x24], "{ssa}");
    assert!(twins(0x04).is_empty(), "{ssa}");
    assert!(!facts.iter().any(|facts| facts.at == 0x20), "{ssa}");
}

/// **Values the test's block builds from each other twice over are decided
/// at once.** `x1 = a + a; x2 = x1 + x1; ... x60 = x59 + x59` before a test of
/// `g` against `x60` whose arms both store `x60` and return it: each value
/// reaches the argument by 2^i ways, and whether it is one value on every
/// trip is read from its operands' answers, not by walking every way. The
/// arms do the same, and the test goes.
#[test]
fn a_test_after_values_shared_many_ways_is_decided_at_once() {
    const DEPTH: u16 = 60;
    let mut ssa = SsaFunction::new(2, usize::from(DEPTH).saturating_add(1));
    let argument = |ssa: &mut SsaFunction<MockTarget>, index| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let a = argument(&mut ssa, 0);
    let g = argument(&mut ssa, 1);
    let mut entry = SsaBlock::new(0);
    let mut value = a;
    for index in 0..DEPTH {
        let next = var(&mut ssa, index, 0, usize::from(index));
        entry.add_instruction(add(next, value, value));
        value = next;
    }
    entry.add_instruction(branch(g, CmpKind::Eq, value, 1, 2));
    let arm = |block| {
        let mut arm = SsaBlock::new(block);
        arm.add_instruction(instr(SsaOp::StoreStaticField {
            field: 1,
            value,
            access: AccessQualifiers::FLAT,
        }));
        arm.add_instruction(instr(SsaOp::returns(Some(value))));
        arm
    };
    for block in [entry, arm(1), arm(2)] {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert!(matches!(
        MootTestPass.run(
            &mut ssa,
            &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                .with_interrupt(&NeverInterrupted)
        ),
        Ok(true)
    ));
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}");
    assert!(matches!(
        ssa.block(0).and_then(SsaBlock::control_terminator),
        Some(SsaOp::Jump { .. })
    ));
}

/// **A loop whose trips do nothing keeps its way out.** libopencm3's
/// `dwc_poll` drains a FIFO it never reads the words of, `for (i = 0; i <
/// rxbcnt; i += 4) (void)fifo;`, so a trip does nothing the way out does
/// not, and the loop's test is moot:
///
/// ```text
/// 0: zero = 0; one = 1; if (n == zero) goto 3 else goto 1
/// 1: goto 2
/// 2: i = phi(1: zero, 2: i1); four = 4; i1 = i + four; word = *fifo;
///    if (i1 < n) goto 2 else goto 3
/// 3: stored = phi(0: one, 2: zero); *field = stored; return n
/// ```
///
/// The entry's test is no moot one: its ways store 1 and 0. Dropped, the
/// loop's test jumps to the way out: kept on the trip, which comes back only
/// through the test, the loop would go round for ever.
#[test]
fn a_moot_loop_test_keeps_the_way_out() {
    let mut ssa = SsaFunction::new(2, 4);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let fifo = argument(&mut ssa, 0);
    let n = argument(&mut ssa, 1);
    let zero = var(&mut ssa, 0, 0, 0);
    let one = var(&mut ssa, 1, 0, 1);
    let i = phi_var(&mut ssa, 2, 2);
    let four = var(&mut ssa, 3, 2, 0);
    let i1 = var(&mut ssa, 4, 2, 1);
    let word = var(&mut ssa, 5, 2, 2);
    let stored = phi_var(&mut ssa, 6, 3);
    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(branch(n, CmpKind::Eq, zero, 3, 1));
    let mut preheader = SsaBlock::new(1);
    preheader.add_instruction(instr(SsaOp::Jump { target: 2 }));
    let mut trip = SsaBlock::new(2);
    trip.add_phi(phi(i, 2, &[(zero, 1), (i1, 2)]));
    trip.add_instruction(instr(SsaOp::Const {
        dest: four,
        value: ConstValue::I32(4),
    }));
    trip.add_instruction(add(i1, i, four));
    trip.add_instruction(instr(SsaOp::LoadIndirect {
        dest: word,
        addr: fifo,
        value_type: MockType::I32,
        access: AccessQualifiers::FLAT,
    }));
    trip.add_instruction(branch(i1, CmpKind::Lt, n, 2, 3));
    let mut out = SsaBlock::new(3);
    out.add_phi(phi(stored, 6, &[(one, 0), (zero, 2)]));
    out.add_instruction(instr(SsaOp::StoreStaticField {
        field: 1,
        value: stored,
        access: AccessQualifiers::FLAT,
    }));
    out.add_instruction(instr(SsaOp::returns(Some(n))));
    for block in [entry, preheader, trip, out] {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert!(
        matches!(
            MootTestPass.run(
                &mut ssa,
                &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                    .with_interrupt(&NeverInterrupted)
            ),
            Ok(true)
        ),
        "the loop's test is moot"
    );
    assert!(
        matches!(
            ssa.block(2).and_then(SsaBlock::control_terminator),
            Some(SsaOp::Jump { target: 3 })
        ),
        "the loop leaves: {:?}",
        ssa.block(2).and_then(SsaBlock::control_terminator)
    );
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
    assert!(
        matches!(
            ssa.block(0).and_then(SsaBlock::control_terminator),
            Some(SsaOp::BranchCmp { .. })
        ),
        "the entry still tests"
    );
}

/// **A moot test in a loop that never ends keeps the way on to the rest of
/// it.** A firmware's main loop drains a FIFO it never reads the words of,
/// then records the pass and goes round again, never returning:
///
/// ```text
/// 0: zero = 0; one = 1; goto 1
/// 1: if (n == zero) goto 3 else goto 4
/// 4: goto 2
/// 2: i = phi(4: zero, 2: i1); four = 4; i1 = i + four; word = *fifo;
///    if (i1 < n) goto 2 else goto 3
/// 3: stored = phi(1: one, 2: zero); *field = stored; goto 1
/// ```
///
/// Neither of the drain's ways leaves the function. Kept on the trip, which
/// comes back only through the test, the drain would go round for ever and
/// the pass would never be recorded again; the way on is kept. The same with
/// the trip's back edge a block of its own (`if (i1 < n) goto 5`, `5: goto
/// 2`), where everything the way on reaches, the trip included, is ahead of
/// it only through the test.
#[test]
fn a_moot_test_in_an_endless_loop_keeps_the_way_on() {
    for own_back_edge in [false, true] {
        let mut ssa = SsaFunction::new(2, 4);
        let argument = |ssa: &mut SsaFunction<MockTarget>, index| {
            ssa.create_variable(
                VariableOrigin::Argument(index),
                0,
                DefSite::entry(),
                MockType::I32,
            )
        };
        let fifo = argument(&mut ssa, 0);
        let n = argument(&mut ssa, 1);
        let zero = var(&mut ssa, 0, 0, 0);
        let one = var(&mut ssa, 1, 0, 1);
        let i = phi_var(&mut ssa, 2, 2);
        let four = var(&mut ssa, 3, 2, 0);
        let i1 = var(&mut ssa, 4, 2, 1);
        let word = var(&mut ssa, 5, 2, 2);
        let stored = phi_var(&mut ssa, 6, 3);
        let (trip, back) = if own_back_edge { (5, 5) } else { (2, 2) };
        let mut entry = SsaBlock::new(0);
        entry.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        entry.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
        let mut head = SsaBlock::new(1);
        head.add_instruction(branch(n, CmpKind::Eq, zero, 3, 4));
        let mut drain = SsaBlock::new(2);
        drain.add_phi(phi(i, 2, &[(zero, 4), (i1, back)]));
        drain.add_instruction(instr(SsaOp::Const {
            dest: four,
            value: ConstValue::I32(4),
        }));
        drain.add_instruction(add(i1, i, four));
        drain.add_instruction(instr(SsaOp::LoadIndirect {
            dest: word,
            addr: fifo,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
        drain.add_instruction(branch(i1, CmpKind::Lt, n, trip, 3));
        let mut pass = SsaBlock::new(3);
        pass.add_phi(phi(stored, 6, &[(one, 1), (zero, 2)]));
        pass.add_instruction(instr(SsaOp::StoreStaticField {
            field: 1,
            value: stored,
            access: AccessQualifiers::FLAT,
        }));
        pass.add_instruction(instr(SsaOp::Jump { target: 1 }));
        let mut preheader = SsaBlock::new(4);
        preheader.add_instruction(instr(SsaOp::Jump { target: 2 }));
        for block in [entry, head, drain, pass, preheader] {
            ssa.add_block(block);
        }
        if own_back_edge {
            let mut latch = SsaBlock::new(5);
            latch.add_instruction(instr(SsaOp::Jump { target: 2 }));
            ssa.add_block(latch);
        }
        ssa.recompute_uses();
        assert!(
            matches!(
                MootTestPass.run(
                    &mut ssa,
                    &PassEnv::new(&0, &NullListener, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::MootSteps, 10_000))
                        .with_interrupt(&NeverInterrupted)
                ),
                Ok(true)
            ),
            "{own_back_edge}: the drain's test is moot"
        );
        assert!(
            matches!(
                ssa.block(2).and_then(SsaBlock::control_terminator),
                Some(SsaOp::Jump { target: 3 })
            ),
            "{own_back_edge}: the drain goes on: {:?}",
            ssa.block(2).and_then(SsaBlock::control_terminator)
        );
        assert!(
            SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty(),
            "{own_back_edge}"
        );
    }
}
