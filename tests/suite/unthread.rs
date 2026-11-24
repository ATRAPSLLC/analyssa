//! Merging back a block a compiler duplicated into the block it copies.
//!
//! The positive cases are paired with negatives that prove each condition is
//! doing the work: a merge that ran on blocks merely shaped alike would pass
//! the positives alone.

use analyssa::{
    analysis::{
        AnalysisContext, SsaCfg, SsaEvaluator, SsaVerifier, VerifyLevel, WorkBudget,
        guards::{Definitions, GuardFacts, Question},
        memory::MemorySsa,
    },
    events::NullListener,
    ir::{
        CallData, Callee,
        block::{BlockProvenance, SsaBlock},
        function::{SsaEditOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::{AccessQualifiers, SsaOp, kinds::CmpKind},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::{DuplicateMergingPass, JumpThreadingPass},
    scheduling::{FunctionPass, PassEnv},
    testing::{FactfulTarget, MACHINE, MockSource, MockTarget, MockType},
};

/// Declares a variable defined at `(block, instr)`, as the first version of
/// its local: version 0 is what the local holds on entry.
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

fn less(dest: SsaVarId, left: SsaVarId, right: SsaVarId) -> SsaInstruction<MockTarget> {
    instr(SsaOp::Clt {
        dest,
        left,
        right,
        unsigned: false,
        signaling: false,
    })
}

fn load(dest: SsaVarId, addr: SsaVarId) -> SsaInstruction<MockTarget> {
    instr(SsaOp::LoadIndirect {
        dest,
        addr,
        value_type: MockType::I32,
        access: AccessQualifiers::FLAT,
    })
}

/// What the block ahead of the loop computes, against the loop's test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ahead {
    /// The test, `n + 1 < 10` where the loop tests `m + step < 10`.
    Test,
    /// The test against a limit the loop reads from memory once, before
    /// everything: the copy reads the cell again, unchanged.
    Reread,
    /// As [`Self::Reread`], with `n` stored to the cell before the copy reads
    /// it: the copy's limit is another value.
    Rewritten,
    /// `a = step + 10; t1 = a + a`, where the loop computes
    /// `a = step + 10; t2 = m + a`: the copy reads in the place of the loop's
    /// phi a value it defines itself, so no value enters the phi from it.
    ReadsItself,
    /// The test, whose sum the loop's body `3` reads, `w = t1 + 1`: the copy
    /// dominates the loop it is ahead of, so the body may.
    ReadBeyond,
}

/// A loop a compiler rotated: its test was copied ahead of it into `1`, so
/// the loop is entered at its test `2` and at its body `3`.
///
/// ```text
/// 0: limit = *cell; goto 1
/// 1: t1 = n + 1; if (t1 < 10) goto 3 else goto 4      (the copy)
/// 2: m = phi(3: k); t2 = m + step; if (t2 < 10) goto 3 else goto 4
/// 3: k = phi(1: t1, 2: t2); goto 2
/// 4: e = phi(1: t1, 2: t2); return e
/// ```
///
/// `ahead` says what `1` computes, and with it what the test compares
/// against ([`Ahead`]). With `step` other than `1` the copy computes something
/// else. Without `looped` nothing goes round: `0` branches to `1` or `2` on
/// `n < 1`, `2`'s phi takes `n` from `0`, and `3` runs on to `4`.
fn rotated(step: i32, looped: bool, ahead: Ahead) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(1, 8);
    let n = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let one = var(&mut ssa, 1, 0, 0);
    let ten = var(&mut ssa, 2, 0, 1);
    let stride = var(&mut ssa, 3, 0, 2);
    let cell = var(&mut ssa, 4, 0, 3);
    let limit = var(&mut ssa, 5, 0, 4);
    let low = (!looped).then(|| var(&mut ssa, 16, 0, 5));
    let reread = matches!(ahead, Ahead::Reread | Ahead::Rewritten);
    // Where the copy's sum lands in `1`, past what it reads first.
    let sum_at: usize = match ahead {
        Ahead::Test | Ahead::ReadBeyond => 0,
        Ahead::Reread | Ahead::ReadsItself => 1,
        Ahead::Rewritten => 2,
    };
    let again = reread.then(|| var(&mut ssa, 6, 1, sum_at.saturating_sub(1)));
    let itself = ahead == Ahead::ReadsItself;
    let a1 = itself.then(|| var(&mut ssa, 7, 1, 0));
    let t1 = var(&mut ssa, 8, 1, sum_at);
    let c1 = var(&mut ssa, 9, 1, sum_at.saturating_add(1));
    let m = phi_var(&mut ssa, 10, 2);
    let a2 = itself.then(|| var(&mut ssa, 11, 2, 0));
    let itself = usize::from(itself);
    let t2 = var(&mut ssa, 12, 2, itself);
    let c2 = var(&mut ssa, 13, 2, itself.saturating_add(1));
    let k = phi_var(&mut ssa, 14, 3);
    let beyond = (ahead == Ahead::ReadBeyond).then(|| var(&mut ssa, 17, 3, 0));
    let e = phi_var(&mut ssa, 15, 4);

    let mut b0 = SsaBlock::new(0);
    for (dest, value) in [
        (one, ConstValue::I32(1)),
        (ten, ConstValue::I32(10)),
        (stride, ConstValue::I32(step)),
        (cell, ConstValue::I32(0x1000)),
    ] {
        b0.add_instruction(instr(SsaOp::Const { dest, value }));
    }
    b0.add_instruction(load(limit, cell));
    if let Some(low) = low {
        b0.add_instruction(less(low, n, one));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: low,
            true_target: 1,
            false_target: 2,
        }));
    } else {
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    }

    let mut b1 = SsaBlock::new(1);
    if ahead == Ahead::Rewritten {
        b1.add_instruction(instr(SsaOp::StoreIndirect {
            addr: cell,
            value: n,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
    }
    if let Some(again) = again {
        b1.add_instruction(load(again, cell));
    }
    if let Some(a1) = a1 {
        b1.add_instruction(add(a1, stride, ten));
        b1.add_instruction(add(t1, a1, a1));
    } else {
        b1.add_instruction(add(t1, n, one));
    }
    let bound = again.unwrap_or(ten);
    b1.add_instruction(less(c1, t1, bound));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: c1,
        true_target: 3,
        false_target: 4,
    }));

    let mut b2 = SsaBlock::new(2);
    let mut header = PhiNode::new(m, VariableOrigin::Local(10));
    header.add_operand(if looped {
        PhiOperand::new(k, 3)
    } else {
        PhiOperand::new(n, 0)
    });
    b2.add_phi(header);
    if let Some(a2) = a2 {
        b2.add_instruction(add(a2, stride, ten));
        b2.add_instruction(add(t2, m, a2));
    } else {
        b2.add_instruction(add(t2, m, stride));
    }
    b2.add_instruction(less(c2, t2, if reread { limit } else { ten }));
    b2.add_instruction(instr(SsaOp::Branch {
        condition: c2,
        true_target: 3,
        false_target: 4,
    }));

    let mut b3 = SsaBlock::new(3);
    let mut body = PhiNode::new(k, VariableOrigin::Local(14));
    body.add_operand(PhiOperand::new(t1, 1));
    body.add_operand(PhiOperand::new(t2, 2));
    b3.add_phi(body);
    if let Some(beyond) = beyond {
        b3.add_instruction(add(beyond, t1, one));
    }
    b3.add_instruction(instr(SsaOp::Jump {
        target: if looped { 2 } else { 4 },
    }));

    let mut b4 = SsaBlock::new(4);
    let mut exit = PhiNode::new(e, VariableOrigin::Local(15));
    exit.add_operand(PhiOperand::new(t1, 1));
    exit.add_operand(PhiOperand::new(t2, 2));
    if !looped {
        exit.add_operand(PhiOperand::new(k, 3));
    }
    b4.add_phi(exit);
    b4.add_instruction(instr(SsaOp::returns(Some(e))));

    for block in [b0, b1, b2, b3, b4] {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// Runs the merge over `ssa`: whether it merged anything, or how it failed.
fn merge(ssa: &mut SsaFunction<MockTarget>) -> analyssa::Result<bool> {
    DuplicateMergingPass.run(
        ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE)),
    )
}

/// The loop's phi in `2`.
fn header(ssa: &SsaFunction<MockTarget>) -> Option<&PhiNode> {
    ssa.block(2).and_then(|block| block.phi_nodes().first())
}

/// Asserts `1` merged into `2`: it keeps what it computes ahead of the copy,
/// `kept` rows, and jumps to `2`, whose phi takes a value from it; the blocks
/// both branched to take nothing from it any more, and the function verifies
/// as the pipeline verifies it after a pass.
fn assert_merged(ssa: &SsaFunction<MockTarget>, kept: usize) {
    let copy = ssa.block(1);
    assert_eq!(
        copy.map(|block| block.instructions().len()),
        Some(kept.saturating_add(1)),
        "{ssa:?}"
    );
    assert!(
        copy.and_then(|block| block.instructions().get(kept))
            .is_some_and(|row| matches!(row.op(), SsaOp::Jump { target: 2 })),
        "{ssa:?}"
    );
    assert!(
        header(ssa).is_some_and(|phi| phi.operand_from(1).is_some()),
        "{ssa:?}"
    );
    for target in [3, 4] {
        for phi in ssa
            .block(target)
            .into_iter()
            .flat_map(|block| block.phi_nodes())
        {
            assert!(phi.operand_from(1).is_none(), "{ssa:?}");
        }
    }
    assert_eq!(ssa.validate(), Ok(()));
    // What the copy defined goes with it: the verifier the pipeline runs
    // after a pass finds no value left behind.
    assert_eq!(
        SsaVerifier::new(ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// **A block a compiler copied ahead of a loop is the loop's test again.**
/// The copy in `1` computes the test `2` computes with `n` in its phi's
/// place, and both branch to `3` and `4`: `1` jumps to `2`, whose phi takes
/// `n` from it, and the loop is entered at `2` alone.
#[test]
fn a_copied_test_ahead_of_a_loop_merges_into_the_test() {
    let mut ssa = rotated(1, true, Ahead::Test);
    assert!(
        matches!(merge(&mut ssa), Ok(true)),
        "the copy merges into the test it copies"
    );
    assert_merged(&ssa, 0);
    assert_eq!(header(&ssa).map(PhiNode::operand_count), Some(2), "{ssa:?}");
}

/// **A block that computes something else is no copy.** The control of
/// [`a_copied_test_ahead_of_a_loop_merges_into_the_test`]: the test in `2`
/// steps by 2, the block in `1` by 1, so nothing is merged.
#[test]
fn a_block_computing_something_else_is_no_copy() {
    let mut ssa = rotated(2, true, Ahead::Test);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **Blocks alike outside any loop are the source's own.** The control of
/// [`a_copied_test_ahead_of_a_loop_merges_into_the_test`] on the other rule:
/// `0` enters `1` or `2`, and `3` runs on to `4` instead of back to `2`, so
/// no loop is entered at two blocks, and the two tests alike are two the
/// program writes.
#[test]
fn blocks_alike_outside_a_loop_stay() {
    let mut ssa = rotated(1, false, Ahead::Test);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **A cell read again, unchanged, is the value read before.** The loop
/// compares against a limit read once ahead of it; the copy reads the cell
/// again and compares against that. Nothing wrote the cell in between, so the
/// two reads are one value, and the copy is the test: it keeps its read and
/// jumps on.
#[test]
fn a_copy_rereading_an_unchanged_cell_merges() {
    let mut ssa = rotated(1, true, Ahead::Reread);
    assert!(matches!(merge(&mut ssa), Ok(true)), "{ssa:?}");
    assert_merged(&ssa, 1);
}

/// **A cell read after a write is another value.** The control of
/// [`a_copy_rereading_an_unchanged_cell_merges`]: the copy's block stores `n`
/// to the cell before it reads it, so its limit is not the loop's, and nothing
/// is merged.
#[test]
fn a_copy_reading_a_rewritten_cell_stays() {
    let mut ssa = rotated(1, true, Ahead::Rewritten);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **No value enters the original that is not there before the copy.** The
/// copy reads `a` where the loop reads its phi `m`, and `a` is the copy's
/// own: it does not exist where the copy would jump, so the phi has nothing
/// to take, and nothing is merged.
#[test]
fn a_copy_reading_its_own_value_in_a_phis_place_stays() {
    let mut ssa = rotated(1, true, Ahead::ReadsItself);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **A value the copy defines stays where a block past it reads it.** The
/// copy dominates the loop it is ahead of, and the loop's body reads the
/// sum it computes: with the copy gone, the body would read nothing, so
/// nothing is merged.
#[test]
fn a_copy_whose_value_the_loop_reads_stays() {
    let mut ssa = rotated(1, true, Ahead::ReadBeyond);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// How [`alike_arms`] ends its arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arms {
    /// Each tests `sum < 10`, the loop's test.
    Alike,
    /// `3` tests `r == 10` where `2` tests `s < 10`.
    OtherQuestion,
    /// `4` returns `j` rather than going round: no loop.
    NoLoop,
    /// Each branches on a value its definition asks nothing of, the
    /// function's argument read again.
    Opaque,
    /// As [`Self::Alike`], with the way out of `3` to `5` too running
    /// through code that folded away, host block 0x35.
    BothFold,
}

/// Two arms of a `switch` in a loop that compute the same thing.
///
/// ```text
/// 0 -> 1
/// 1: i = phi(0: 0, 4: j); if (i < 10) goto 2 else goto 3
/// 2: s = i + 1; if (s < 10) goto 4 else goto 5
/// 3: r = i + 1; if (r < 10) goto 4 else goto 5
/// 4: j = phi(2: s, 3: r); goto 1
/// 5: e = phi(2: s, 3: r); return e
/// ```
///
/// `arms` says how the arms test ([`Arms`]).
fn alike_arms(arms: Arms) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(1, 8);
    let zero = var(&mut ssa, 0, 0, 0);
    let one = var(&mut ssa, 1, 0, 1);
    let ten = var(&mut ssa, 2, 0, 2);
    let i = phi_var(&mut ssa, 3, 1);
    let ci = var(&mut ssa, 4, 1, 0);
    let s = var(&mut ssa, 5, 2, 0);
    let cs = var(&mut ssa, 6, 2, 1);
    let r = var(&mut ssa, 7, 3, 0);
    let cr = var(&mut ssa, 8, 3, 1);
    let j = phi_var(&mut ssa, 9, 4);
    let e = phi_var(&mut ssa, 10, 5);

    let mut b0 = SsaBlock::new(0);
    for (dest, value) in [(zero, 0), (one, 1), (ten, 10)] {
        b0.add_instruction(instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        }));
    }
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));

    let mut b1 = SsaBlock::new(1);
    let mut head = PhiNode::new(i, VariableOrigin::Local(3));
    head.add_operand(PhiOperand::new(zero, 0));
    if arms != Arms::NoLoop {
        head.add_operand(PhiOperand::new(j, 4));
    }
    b1.add_phi(head);
    b1.add_instruction(less(ci, i, ten));
    b1.add_instruction(instr(SsaOp::Branch {
        condition: ci,
        true_target: 2,
        false_target: 3,
    }));

    let mut blocks = Vec::new();
    for (index, sum, test) in [(2, s, cs), (3, r, cr)] {
        let mut arm = SsaBlock::new(index);
        arm.add_instruction(add(sum, i, one));
        arm.add_instruction(match arms {
            Arms::OtherQuestion if index == 3 => instr(SsaOp::Ceq {
                dest: test,
                left: sum,
                right: ten,
                signaling: false,
            }),
            Arms::Opaque => instr(SsaOp::LoadArg {
                dest: test,
                arg_index: 0,
            }),
            _ => less(test, sum, ten),
        });
        arm.add_instruction(instr(SsaOp::Branch {
            condition: test,
            true_target: 4,
            false_target: 5,
        }));
        blocks.push(arm);
    }

    let mut b4 = SsaBlock::new(4);
    let mut latch = PhiNode::new(j, VariableOrigin::Local(9));
    latch.add_operand(PhiOperand::new(s, 2));
    latch.add_operand(PhiOperand::new(r, 3));
    b4.add_phi(latch);
    b4.add_instruction(instr(if arms == Arms::NoLoop {
        SsaOp::returns(Some(j))
    } else {
        SsaOp::Jump { target: 1 }
    }));

    let mut b5 = SsaBlock::new(5);
    let mut exit = PhiNode::new(e, VariableOrigin::Local(10));
    exit.add_operand(PhiOperand::new(s, 2));
    exit.add_operand(PhiOperand::new(r, 3));
    b5.add_phi(exit);
    b5.add_instruction(instr(SsaOp::returns(Some(e))));

    // Each block stands for the host block its index names, times 0x10.
    let mut all = vec![b0, b1];
    all.extend(blocks);
    all.extend([b4, b5]);
    // and arm 2's way out to `5` runs through host block 0x25, whose code
    // folded away on that edge, as a branch whose way out is a jump runs.
    for (index, mut block) in all.into_iter().enumerate() {
        block
            .provenance_mut()
            .carry(u64::try_from(index).unwrap_or(0).saturating_mul(0x10));
        let folded = match index {
            2 => Some(0x25),
            3 if arms == Arms::BothFold => Some(0x35),
            _ => None,
        };
        if let Some(host) = folded {
            let mut passed = BlockProvenance::default();
            passed.carry(host);
            block.provenance_mut().pass_through_on(5, &passed);
        }
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **Arms alike that the loop enters are the source's own.** `2` and `3`
/// compute the same thing and branch to the same blocks, each on the loop
/// through `1`: neither is entered from outside a loop the other lies on, so
/// they are two arms the program writes, as `switch` arms each opening a file
/// are, and each keeps its sum. The loop's test both end in is the loop's
/// own, which the source writes once past the arms, and is one block.
#[test]
fn arms_alike_that_a_loop_enters_stay() {
    let mut ssa = alike_arms(Arms::Alike);
    assert!(matches!(merge(&mut ssa), Ok(true)), "{ssa:?}");
    for arm in [2, 3] {
        let block = ssa.block(arm);
        assert!(
            block.is_some_and(|block| block
                .instructions()
                .iter()
                .any(|row| matches!(row.op(), SsaOp::Add { .. }))),
            "B{arm} keeps its sum: {ssa:?}"
        );
        assert!(
            block.is_some_and(
                |block| !block.successors().contains(&2) && !block.successors().contains(&3)
            ),
            "B{arm} runs into no other arm: {ssa:?}"
        );
    }
    let tests: Vec<usize> = ssa
        .blocks()
        .iter()
        .enumerate()
        .filter(|(_, block)| {
            block.instructions().last().is_some_and(|row| {
                matches!(
                    row.op(),
                    SsaOp::Clt { .. } | SsaOp::BranchCmp { .. } | SsaOp::Branch { .. }
                )
            }) && block.successors().contains(&4)
        })
        .map(|(index, _)| index)
        .collect();
    assert_eq!(tests.len(), 1, "one block tests the loop: {ssa:?}");
    // The one test is arm 2's branch: it holds the host block that branch
    // lies in, and its edge to `5` runs through what arm 2's did, its edge to
    // `4` through nothing. Arm 3 keeps its own host block, its branch's code,
    // which ran on its way to the one test.
    let merged = tests.first().and_then(|test| ssa.block(*test));
    assert_eq!(
        merged.map(|block| block.provenance().carried().to_vec()),
        Some(vec![0x20]),
        "{ssa:?}"
    );
    assert_eq!(
        merged.map(|block| (block.passed_on(4), block.passed_on(5))),
        Some((Vec::new(), vec![0x25])),
        "{ssa:?}"
    );
    // The sum `4`'s and `5`'s phis take from the one test is one value of the
    // variable the first of those phis stands for, not of one arm's sum; the
    // condition it branches on is a phi of the arms' conditions.
    let mut joined: Vec<VariableOrigin> = merged
        .map(|block| block.phi_nodes().iter().map(PhiNode::origin).collect())
        .unwrap_or_default();
    joined.sort_unstable();
    assert_eq!(
        joined,
        [VariableOrigin::Local(6), VariableOrigin::Local(9)],
        "{ssa:?}"
    );
    assert!(
        ssa.block(2)
            .is_some_and(|block| block.provenance().passed().is_empty()),
        "{ssa:?}"
    );
    assert_eq!(
        ssa.block(3)
            .map(|block| block.provenance().carried().to_vec()),
        Some(vec![0x30]),
        "{ssa:?}"
    );
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// **Arms asking different questions keep their tests.** The control of
/// [`arms_alike_that_a_loop_enters_stay`]'s merge: `3` tests `r == 10` where
/// `2` tests `s < 10`, two questions, and each stays its own.
#[test]
fn loop_tests_asking_different_questions_stay() {
    let mut ssa = alike_arms(Arms::OtherQuestion);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **Tests alike outside a loop stay.** `4` returns: no loop goes on through
/// the tests, so they are two the program writes.
#[test]
fn tests_alike_outside_a_loop_stay() {
    let mut ssa = alike_arms(Arms::NoLoop);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **Code folded on two ways out stays on each.** Each arm's way out to `5`
/// runs through code that folded away, a jump of its own. The one test is arm
/// 2's branch, whose edge to `5` runs through arm 2's code; arm 3's ran on its
/// way out, and its way out is its jump to the one test now, which runs
/// through it. Every host block is still named, each on the path its code ran
/// on.
#[test]
fn loop_tests_whose_ways_out_both_fold_code_merge() {
    let mut ssa = alike_arms(Arms::BothFold);
    assert!(matches!(merge(&mut ssa), Ok(true)), "{ssa:?}");
    let merged = ssa.block_count().saturating_sub(1);
    assert_eq!(
        ssa.block(merged)
            .map(|block| (block.passed_on(4), block.passed_on(5))),
        Some((Vec::new(), vec![0x25])),
        "{ssa:?}"
    );
    assert_eq!(
        ssa.block(2).map(|block| block.passed_on(merged)),
        Some(Vec::new()),
        "{ssa:?}"
    );
    assert_eq!(
        ssa.block(3).map(|block| block.passed_on(merged)),
        Some(vec![0x35]),
        "{ssa:?}"
    );
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// **Branches on values that ask nothing stay.** Each arm branches on a value
/// no definition makes a question of: two such branches are a `break` on one
/// condition and on another, and nothing reads them as one test.
#[test]
fn loop_branches_on_values_that_ask_nothing_stay() {
    let mut ssa = alike_arms(Arms::Opaque);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// How [`threaded`] differs from the threaded dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Threaded {
    /// `1` tests `x == 3`, so `7` is entered knowing `x != 3` and skips the
    /// dispatch's `x != 3`.
    Skipped,
    /// `1` tests `x == 4`, which answers nothing about `x != 3`.
    Unanswered,
    /// `5`'s phi takes `y` from `7` where it takes `x` through `3`.
    OtherValue,
    /// `3` stores `x` before its test, so going through it does something.
    Writes,
}

/// A dispatch on `x` the compiler threaded a path into: the path, knowing
/// `x != 3`, copied the dispatch's first test and jumps past its second.
///
/// ```text
/// 0: if (y == 0) goto 1 else goto 2
/// 1: if (x == 3) goto 6 else goto 7
/// 7: if (x == 2) goto 4 else goto 5      (the copy)
/// 2: if (x == 2) goto 4 else goto 3      (the dispatch)
/// 3: if (x != 3) goto 5 else goto 6
/// 4: return 20
/// 5: r = phi(3: x, 7: x); return r
/// 6: return 60
/// ```
fn threaded(shape: Threaded) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(2, 8);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let x = argument(&mut ssa, 0);
    let y = argument(&mut ssa, 1);
    let zero = var(&mut ssa, 0, 0, 0);
    let cell = var(&mut ssa, 1, 0, 1);
    let tested = var(&mut ssa, 2, 1, 0);
    let two_ahead = var(&mut ssa, 3, 7, 0);
    let two = var(&mut ssa, 4, 2, 0);
    let writes = shape == Threaded::Writes;
    let three = var(&mut ssa, 5, 3, usize::from(writes));
    let r = phi_var(&mut ssa, 6, 5);
    let twenty = var(&mut ssa, 7, 4, 0);
    let sixty = var(&mut ssa, 8, 6, 0);

    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..8).map(SsaBlock::new).collect();
    let mut fill = |index: usize, rows: Vec<SsaInstruction<MockTarget>>| {
        if let Some(block) = blocks.get_mut(index) {
            for row in rows {
                block.add_instruction(row);
            }
        }
    };
    fill(
        0,
        vec![
            constant(zero, 0),
            constant(cell, 0x1000),
            compare(y, zero, CmpKind::Eq, (1, 2)),
        ],
    );
    let limit = if shape == Threaded::Unanswered { 4 } else { 3 };
    fill(
        1,
        vec![
            constant(tested, limit),
            compare(x, tested, CmpKind::Eq, (6, 7)),
        ],
    );
    fill(
        7,
        vec![
            constant(two_ahead, 2),
            compare(x, two_ahead, CmpKind::Eq, (4, 5)),
        ],
    );
    fill(
        2,
        vec![constant(two, 2), compare(x, two, CmpKind::Eq, (4, 3))],
    );
    let mut dispatch = Vec::new();
    if writes {
        dispatch.push(instr(SsaOp::StoreIndirect {
            addr: cell,
            value: x,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
    }
    dispatch.push(constant(three, 3));
    dispatch.push(compare(x, three, CmpKind::Ne, (5, 6)));
    fill(3, dispatch);
    fill(
        4,
        vec![constant(twenty, 20), instr(SsaOp::returns(Some(twenty)))],
    );
    fill(5, vec![instr(SsaOp::returns(Some(r)))]);
    fill(
        6,
        vec![constant(sixty, 60), instr(SsaOp::returns(Some(sixty)))],
    );
    let mut joined = PhiNode::new(r, VariableOrigin::Local(6));
    joined.add_operand(PhiOperand::new(x, 3));
    joined.add_operand(PhiOperand::new(
        if shape == Threaded::OtherValue { y } else { x },
        7,
    ));
    if let Some(block) = blocks.get_mut(5) {
        block.add_phi(joined);
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **A copy that jumps past a test its path answered is the dispatch
/// again.** `7` tests `x == 2` as `2` does, and where `2` goes on to `3`'s
/// `x != 3`, `7` goes straight to where that test leads: on its way in,
/// `1` found `x != 3`. `7` jumps to `2`, and the dispatch is entered at one
/// block.
#[test]
fn a_copy_jumping_past_an_answered_test_merges_into_the_dispatch() {
    let mut ssa = threaded(Threaded::Skipped);
    assert!(matches!(merge(&mut ssa), Ok(true)), "{ssa:?}");
    assert!(
        ssa.block(7)
            .and_then(|block| block.instructions().last())
            .is_some_and(|row| matches!(row.op(), SsaOp::Jump { target: 2 })),
        "{ssa:?}"
    );
    assert!(
        ssa.block(5)
            .and_then(|block| block.phi_nodes().first())
            .is_none_or(|phi| phi.operand_from(7).is_none()),
        "{ssa:?}"
    );
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// **A test the path did not answer is not skipped.** The control of
/// [`a_copy_jumping_past_an_answered_test_merges_into_the_dispatch`]: `1`
/// tests `x == 4`, so `7` does not know where `3` would go, and nothing is
/// merged.
#[test]
fn a_copy_past_an_unanswered_test_stays() {
    let mut ssa = threaded(Threaded::Unanswered);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **A copy carrying another value where the paths join stays.** `5` takes
/// `y` from `7` and `x` through `3`, so the two ways in are not one.
#[test]
fn a_copy_carrying_another_value_past_the_test_stays() {
    let mut ssa = threaded(Threaded::OtherValue);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **A block that does something is not jumped past.** `3` stores `x`
/// before its test, so going through it is not going straight on, and
/// nothing is merged.
#[test]
fn a_test_behind_a_write_is_not_skipped() {
    let mut ssa = threaded(Threaded::Writes);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// What the way into [`skipped_phi`]'s copy knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkippedPhi {
    /// `0` sends the copy on knowing `lim >=u new`, as `cdcmd` in dash does.
    Answered,
    /// `0` sends it on knowing `lim >=u y`, which says nothing of `new`.
    Unanswered,
}

/// A copy of a block whose phi only the test past it reads: `cdcmd` in
/// dash, where `p = strtok(cdcomppath, "/"); if (p)` is copied into the arm
/// that knows `new <= lim`, the copy's way out sent past `if (new > lim)`,
/// which reads the original's `lim` and nothing of the copied rows.
///
/// ```text
/// 0: if (lim >=u new) goto 7 else goto 1
/// 1: if (y == 0) goto 3 else goto 2
/// 2: goto 3
/// 3: q = phi(1: lim, 2: y); k = 9; if (y == k) goto 4 else goto 5   (the original)
/// 5: if (q <u new) goto 6 else goto 8
/// 6: d = new - 1; goto 8
/// 7: k7 = 9; if (y == k7) goto 4 else goto 8                         (the copy)
/// 4: return 40
/// 8: r = phi(5: new; 6: d; 7: new); return r
/// ```
///
/// On `7`'s way, `q` would be `lim`, its one operand defined there, and `5`
/// would then find `lim >=u new` and go on to `8`, where `7` goes.
fn skipped_phi(shape: SkippedPhi) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(3, 8);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let lim = argument(&mut ssa, 0);
    let new = argument(&mut ssa, 1);
    let y = argument(&mut ssa, 2);
    let zero = var(&mut ssa, 0, 1, 0);
    let q = phi_var(&mut ssa, 1, 3);
    let k = var(&mut ssa, 2, 3, 0);
    let one = var(&mut ssa, 3, 6, 0);
    let d = var(&mut ssa, 4, 6, 1);
    let k7 = var(&mut ssa, 5, 7, 0);
    let forty = var(&mut ssa, 6, 4, 0);
    let r = phi_var(&mut ssa, 7, 8);

    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: matches!(cmp, CmpKind::Ge | CmpKind::Lt),
            true_target: to.0,
            false_target: to.1,
        })
    };
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..9).map(SsaBlock::new).collect();
    let mut fill = |index: usize, rows: Vec<SsaInstruction<MockTarget>>| {
        if let Some(block) = blocks.get_mut(index) {
            for row in rows {
                block.add_instruction(row);
            }
        }
    };
    let known = match shape {
        SkippedPhi::Answered => new,
        SkippedPhi::Unanswered => y,
    };
    fill(0, vec![compare(lim, known, CmpKind::Ge, (7, 1))]);
    fill(
        1,
        vec![constant(zero, 0), compare(y, zero, CmpKind::Eq, (3, 2))],
    );
    fill(2, vec![instr(SsaOp::Jump { target: 3 })]);
    fill(3, vec![constant(k, 9), compare(y, k, CmpKind::Eq, (4, 5))]);
    fill(5, vec![compare(q, new, CmpKind::Lt, (6, 8))]);
    fill(
        6,
        vec![
            constant(one, 1),
            instr(SsaOp::Sub {
                dest: d,
                left: new,
                right: one,
                flags: None,
            }),
            instr(SsaOp::Jump { target: 8 }),
        ],
    );
    fill(
        7,
        vec![constant(k7, 9), compare(y, k7, CmpKind::Eq, (4, 8))],
    );
    fill(
        4,
        vec![constant(forty, 40), instr(SsaOp::returns(Some(forty)))],
    );
    fill(8, vec![instr(SsaOp::returns(Some(r)))]);
    let mut entered = PhiNode::new(q, VariableOrigin::Local(1));
    entered.add_operand(PhiOperand::new(lim, 1));
    entered.add_operand(PhiOperand::new(y, 2));
    let mut joined = PhiNode::new(r, VariableOrigin::Local(7));
    joined.add_operand(PhiOperand::new(new, 5));
    joined.add_operand(PhiOperand::new(d, 6));
    joined.add_operand(PhiOperand::new(new, 7));
    if let Some(block) = blocks.get_mut(3) {
        block.add_phi(entered);
    }
    if let Some(block) = blocks.get_mut(8) {
        block.add_phi(joined);
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "the fixture verifies: {ssa:?}"
    );
    ssa
}

/// **A copy past a test that reads only the original's phi merges.** `7`
/// copies `3`'s rows and goes straight to `8` where `3` goes through `5`'s
/// `q <u new`; `q` is read by none of the copied rows, and on `7`'s way it
/// would be `lim`, which `0` found `>=u new`. `7` jumps to `3`, and `q` takes
/// `lim` from it.
#[test]
fn a_copy_past_a_test_reading_the_originals_phi_merges() {
    let mut ssa = skipped_phi(SkippedPhi::Answered);
    assert!(matches!(merge(&mut ssa), Ok(true)), "{ssa:?}");
    assert!(
        ssa.block(7)
            .and_then(|block| block.instructions().last())
            .is_some_and(|row| matches!(row.op(), SsaOp::Jump { target: 3 })),
        "{ssa:?}"
    );
    let lim = ssa
        .block(3)
        .and_then(|block| block.phi_nodes().first())
        .and_then(|phi| phi.operand_from(1))
        .map(PhiOperand::value);
    assert_eq!(
        ssa.block(3)
            .and_then(|block| block.phi_nodes().first())
            .and_then(|phi| phi.operand_from(7))
            .map(PhiOperand::value),
        lim,
        "{ssa:?}"
    );
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// **A phi the copy's way says nothing of keeps the copy.** The control of
/// [`a_copy_past_a_test_reading_the_originals_phi_merges`]: `0` knows
/// `lim >=u y`, so with `q` as `lim`, `5` is not decided, and nothing is
/// merged.
#[test]
fn a_copy_past_a_test_of_a_phi_its_way_leaves_open_stays() {
    let mut ssa = skipped_phi(SkippedPhi::Unanswered);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// Which way a copy of [`tail_copies`]' join is made to differ from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailCopies {
    /// As `cdcmd` in dash has them.
    Copied,
    /// `1` tests `lim < s` instead of `lim >= new`: no copy's way decides
    /// `new <= lim`.
    Unanswered,
    /// `7` calls `g` where the join calls `f`.
    OtherCall,
    /// `5` tests its call's result against `1`, where the join tests `0`.
    OtherConstant,
}

/// A join the compiler copied into the ways that reach it, each copy's exit
/// sent past the test after the join where the copy's way decides it, as
/// `cdcmd` in dash has `p = strtok(cdcomppath, "/")` and its `if (p)` in
/// each arm of `if (*dir != '/')`, and `if (new > lim) new--` after the loop:
///
/// ```text
///  0: if (s == 0) goto 11 else goto 1
/// 11: s1 = s + 1; n1 = new + 1; goto 6
///  1: if (lim >=u new) goto 7 else goto 2
///  2: lim2 = lim + 1; if (c != 0) goto 3 else goto 4
///  3: goto 4
///  4: l = phi(2: lim, 3: lim2); if (c != 0) goto 6 else goto 5
///  5: a = f(s); if (a == 0u) goto 10 else goto 8          (a copy)
///  6: s6, n6, l6 = phi(4: s, new, l; 11: s1, n1, lim)
///     j = f(s6); if (j != 0) goto 8 else goto 9           (the join)
///  7: b = f(s); if (b != 0) goto 8 else goto 12           (a copy)
///  8: p8, n8, l8 = phi(5: a, new, l; 6: j, n6, l6; 7: b, new, lim)
///     if (p8 == 0) goto 9 else goto 13
///  9: n9, l9 = phi(6: n6, l6; 8: n8, l8); if (n9 <=u l9) goto 12 else goto 10
/// 10: m = phi(5: new; 9: n9); m1 = m - 1; goto 12
/// 12: r = phi(7: new; 9: n9; 10: m1); return r
/// 13: return p8
/// ```
///
/// `7` is entered knowing `lim >= new`, so `9` would find `new <= lim` and
/// go on to `12`; `5` is entered from `4` knowing `c == 0`, so from `2`
/// with `l` its `lim`, and knowing `lim < new`, so `9` would go on to `10`.
/// Each is the join entered from its own way.
fn tail_copies(shape: TailCopies) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(4, 32);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let new = argument(&mut ssa, 0);
    let lim = argument(&mut ssa, 1);
    let s = argument(&mut ssa, 2);
    let c = argument(&mut ssa, 3);
    let zero = var(&mut ssa, 0, 0, 0);
    let one = var(&mut ssa, 1, 11, 0);
    let s1 = var(&mut ssa, 2, 11, 1);
    let n1 = var(&mut ssa, 3, 11, 2);
    let one2 = var(&mut ssa, 4, 2, 0);
    let lim2 = var(&mut ssa, 5, 2, 1);
    let l = phi_var(&mut ssa, 6, 4);
    let a = var(&mut ssa, 7, 5, 0);
    let za = var(&mut ssa, 8, 5, 1);
    let s6 = phi_var(&mut ssa, 9, 6);
    let n6 = phi_var(&mut ssa, 10, 6);
    let l6 = phi_var(&mut ssa, 11, 6);
    let j = var(&mut ssa, 12, 6, 0);
    let zj = var(&mut ssa, 13, 6, 1);
    let b = var(&mut ssa, 14, 7, 0);
    let zb = var(&mut ssa, 15, 7, 1);
    let p8 = phi_var(&mut ssa, 16, 8);
    let n8 = phi_var(&mut ssa, 17, 8);
    let l8 = phi_var(&mut ssa, 18, 8);
    let n9 = phi_var(&mut ssa, 19, 9);
    let l9 = phi_var(&mut ssa, 20, 9);
    let m = phi_var(&mut ssa, 21, 10);
    let minus = var(&mut ssa, 22, 10, 0);
    let m1 = var(&mut ssa, 23, 10, 1);
    let r = phi_var(&mut ssa, 24, 12);

    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let compare =
        |left: SsaVarId, right: SsaVarId, cmp: CmpKind, unsigned: bool, to: (usize, usize)| {
            instr(SsaOp::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target: to.0,
                false_target: to.1,
            })
        };
    let call = |dest: SsaVarId, callee: u32, argument: SsaVarId| {
        instr(CallData::new(Some(dest), Callee::Direct(callee), vec![argument]).into())
    };
    let jump = |target: usize| instr(SsaOp::Jump { target });
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..14).map(SsaBlock::new).collect();
    let mut fill = |index: usize, rows: Vec<SsaInstruction<MockTarget>>| {
        if let Some(block) = blocks.get_mut(index) {
            for row in rows {
                block.add_instruction(row);
            }
        }
    };
    fill(
        0,
        vec![
            constant(zero, 0),
            compare(s, zero, CmpKind::Eq, false, (11, 1)),
        ],
    );
    fill(
        11,
        vec![
            constant(one, 1),
            add(s1, s, one),
            add(n1, new, one),
            jump(6),
        ],
    );
    let decider = if shape == TailCopies::Unanswered {
        compare(lim, s, CmpKind::Lt, true, (7, 2))
    } else {
        compare(lim, new, CmpKind::Ge, true, (7, 2))
    };
    fill(1, vec![decider]);
    fill(
        2,
        vec![
            constant(one2, 1),
            add(lim2, lim, one2),
            compare(c, zero, CmpKind::Ne, false, (3, 4)),
        ],
    );
    fill(3, vec![jump(4)]);
    fill(4, vec![compare(c, zero, CmpKind::Ne, false, (6, 5))]);
    // The copy writes its `0` unsigned, the join signed: one word.
    let tested = if shape == TailCopies::OtherConstant {
        ConstValue::U32(1)
    } else {
        ConstValue::U32(0)
    };
    fill(
        5,
        vec![
            call(a, 7, s),
            instr(SsaOp::Const {
                dest: za,
                value: tested,
            }),
            compare(a, za, CmpKind::Eq, false, (10, 8)),
        ],
    );
    fill(
        6,
        vec![
            call(j, 7, s6),
            constant(zj, 0),
            compare(j, zj, CmpKind::Ne, false, (8, 9)),
        ],
    );
    let callee_b = if shape == TailCopies::OtherCall { 8 } else { 7 };
    fill(
        7,
        vec![
            call(b, callee_b, s),
            constant(zb, 0),
            compare(b, zb, CmpKind::Ne, false, (8, 12)),
        ],
    );
    fill(8, vec![compare(p8, zero, CmpKind::Eq, false, (9, 13))]);
    fill(9, vec![compare(n9, l9, CmpKind::Le, true, (12, 10))]);
    fill(10, vec![constant(minus, -1), add(m1, m, minus), jump(12)]);
    fill(12, vec![instr(SsaOp::returns(Some(r)))]);
    fill(13, vec![instr(SsaOp::returns(Some(p8)))]);
    let phis = [
        (4, l, 6, vec![(lim, 2), (lim2, 3)]),
        (6, s6, 9, vec![(s, 4), (s1, 11)]),
        (6, n6, 10, vec![(new, 4), (n1, 11)]),
        (6, l6, 11, vec![(l, 4), (lim, 11)]),
        (8, p8, 16, vec![(a, 5), (j, 6), (b, 7)]),
        (8, n8, 17, vec![(new, 5), (n6, 6), (new, 7)]),
        (8, l8, 18, vec![(l, 5), (l6, 6), (lim, 7)]),
        (9, n9, 19, vec![(n6, 6), (n8, 8)]),
        (9, l9, 20, vec![(l6, 6), (l8, 8)]),
        (10, m, 21, vec![(new, 5), (n9, 9)]),
        (12, r, 24, vec![(new, 7), (n9, 9), (m1, 10)]),
    ];
    for (at, result, local, operands) in phis {
        let mut phi = PhiNode::new(result, VariableOrigin::Local(local));
        for (value, from) in operands {
            phi.add_operand(PhiOperand::new(value, from));
        }
        if let Some(block) = blocks.get_mut(at) {
            block.add_phi(phi);
        }
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// The block `copy` jumps to, where it is a jump.
fn jumps_to(ssa: &SsaFunction<MockTarget>, copy: usize) -> Option<usize> {
    match ssa.block(copy)?.instructions().last()?.op() {
        SsaOp::Jump { target } => Some(*target),
        _ => None,
    }
}

/// **Copies of a join, each sent past the test after it, are the join
/// again.** `5` and `7` call `f(s)` and test the result as `6` does, `6`
/// entered from `4` or `1` with `s` for `s6`; where `6`'s `j == 0` goes on
/// to `9`, each goes straight to where `9` would send it, its way having
/// decided `new <= lim`. Both jump to `6`, whose phis take each one's
/// values, and the function verifies.
#[test]
fn copies_of_a_join_sent_past_its_next_test_merge_into_it() {
    let mut ssa = tail_copies(TailCopies::Copied);
    while matches!(merge(&mut ssa), Ok(true)) {}
    assert_eq!(jumps_to(&ssa, 7), Some(6), "{ssa:?}");
    assert_eq!(jumps_to(&ssa, 5), Some(6), "{ssa:?}");
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// **A copy whose way decides nothing stays.** The control of
/// [`copies_of_a_join_sent_past_its_next_test_merge_into_it`]: `1` asks
/// `lim < s`, so neither copy knows where `9` would go.
#[test]
fn copies_of_a_join_past_an_unanswered_test_stay() {
    let mut ssa = tail_copies(TailCopies::Unanswered);
    while matches!(merge(&mut ssa), Ok(true)) {}
    assert_eq!(jumps_to(&ssa, 7), None, "{ssa:?}");
    assert_eq!(jumps_to(&ssa, 5), None, "{ssa:?}");
}

/// **A copy testing another constant is no copy.** `5` tests `a == 1`
/// where the join tests `j != 0`, and stays; `7` merges as before.
#[test]
fn a_join_copy_testing_another_constant_stays() {
    let mut ssa = tail_copies(TailCopies::OtherConstant);
    while matches!(merge(&mut ssa), Ok(true)) {}
    assert_eq!(jumps_to(&ssa, 7), Some(6), "{ssa:?}");
    assert_eq!(jumps_to(&ssa, 5), None, "{ssa:?}");
}

/// **A block calling another function is no copy.** `7` calls `g(s)` where
/// the join calls `f(s)`, and stays.
#[test]
fn a_join_copy_calling_otherwise_stays() {
    let mut ssa = tail_copies(TailCopies::OtherCall);
    while matches!(merge(&mut ssa), Ok(true)) {}
    assert_eq!(jumps_to(&ssa, 7), None, "{ssa:?}");
}

/// **A merged copy's rows are points of the original's.** Each row of the
/// original does the copy's work on the copy's path once the copy jumps to
/// it, so it takes in the address of the row it stands for, as a host keeping
/// a call site per row needs: `cdcmd`'s copied `strtok` call stays a call
/// site of the one call left. The row the copy keeps ahead of the copy is no
/// twin of anything.
#[test]
fn a_merged_copys_rows_are_points_of_the_originals() {
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
        if index == 1 {
            arm.add_instruction(row(
                0x0c,
                CallData::new(None, Callee::Direct(8), vec![g]).into(),
            ));
        }
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
    let merged = ssa.edit(SsaEditOptions::new(), |editor| {
        editor.merge_duplicate(1, 2, 1, &[])?;
        editor.declare_repaired();
        Ok(())
    });
    assert!(merged.is_ok(), "{merged:?} {ssa}");
    let twins = |at: u64| -> Vec<u64> {
        ssa.blocks()
            .iter()
            .flat_map(SsaBlock::instructions)
            .find(|row| row.facts().at == at)
            .map(|row| row.facts().twins.iter().copied().collect())
            .unwrap_or_default()
    };
    assert_eq!(twins(0x20), [0x10], "{ssa}");
    assert_eq!(twins(0x24), [0x14], "{ssa}");
    assert!(twins(0x0c).is_empty(), "{ssa}");
}

/// Two arms of a test calling `f(g)` and going on to `3`: `1`, holding
/// host block `0x10` and, where `keeps` says, `0x0c` ahead of its call, and
/// `2`, holding `0x20`. Each arm's edge to `3` runs through host code of its
/// own, `0x18` and `0x28`; every edge out of `1` runs through `0x08`.
fn arms_holding_code(keeps: bool) -> SsaFunction<FactfulTarget> {
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
    let mut test = SsaBlock::new(0);
    test.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    test.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
        left: g,
        right: zero,
        cmp: CmpKind::Eq,
        unsigned: false,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(test);
    let hosting = |host: u64| {
        let mut held = BlockProvenance::default();
        held.carry(host);
        held
    };
    for (index, at, exit) in [(1usize, 0x10u64, 0x18u64), (2, 0x20, 0x28)] {
        let mut arm = SsaBlock::new(index);
        if index == 1 && keeps {
            arm.provenance_mut().carry(0x0c);
            arm.add_instruction(SsaInstruction::synthetic(
                CallData::new(None, Callee::Direct(8), vec![g]).into(),
            ));
        }
        arm.provenance_mut().carry(at);
        arm.provenance_mut().pass_through_on(3, &hosting(exit));
        if index == 1 {
            arm.provenance_mut().pass_through(&hosting(0x08));
        }
        arm.add_instruction(SsaInstruction::synthetic(
            CallData::new(None, Callee::Direct(7), vec![g]).into(),
        ));
        arm.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        ssa.add_block(arm);
    }
    let mut exit = SsaBlock::new(3);
    exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
    ssa.add_block(exit);
    ssa.recompute_uses();
    ssa
}

/// `1` merged into `2` from its row `at`: what `1` then carries, and what
/// it passes on every edge and on its edge to `3` and to `2`.
fn merged_record(keeps: bool, at: usize) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    let mut ssa = arms_holding_code(keeps);
    let merged = ssa.edit(SsaEditOptions::new(), |editor| {
        editor.merge_duplicate(1, 2, at, &[])?;
        editor.declare_repaired();
        Ok(())
    });
    assert!(merged.is_ok(), "{merged:?} {ssa}");
    let Some(block) = ssa.block(1) else {
        return Default::default();
    };
    let provenance = block.provenance();
    let everywhere: Vec<u64> = provenance
        .passed()
        .into_iter()
        .filter(|host| {
            provenance.passed_on(2).contains(host) && provenance.passed_on(3).contains(host)
        })
        .collect();
    (
        provenance.carried().to_vec(),
        everywhere,
        provenance.passed_on(3),
        provenance.passed_on(2),
    )
}

/// **A merged copy's ways out run nothing of its own.** `1` keeps its first
/// call, and what it holds for it, `0x0c`, and what every edge out of it
/// runs through, `0x08`; the code its edge to `3` ran after the copy,
/// `0x18`, runs on no way now, its jump to `2` included. The control is `2`,
/// which keeps its own exit.
#[test]
fn a_merged_copys_ways_out_run_nothing_of_its_own() {
    let (carried, everywhere, to_exit, to_original) = merged_record(true, 1);
    assert_eq!(carried, [0x0c, 0x10]);
    assert_eq!(everywhere, [0x08]);
    assert!(!to_exit.contains(&0x18) && !to_original.contains(&0x18));
    let ssa = arms_holding_code(true);
    assert_eq!(
        ssa.block(2).map(|block| block.provenance().passed_on(3)),
        Some(vec![0x28]),
        "control"
    );
}

/// **A copy that was the whole block leaves it standing for nothing.** Its
/// rows live on as the original's twins, so the jump left in its place
/// holds and passes nothing, and an edge sent past it later runs past
/// nothing either.
#[test]
fn a_whole_block_copy_leaves_its_block_standing_for_nothing() {
    let (carried, _, to_exit, to_original) = merged_record(false, 0);
    assert!(carried.is_empty() && to_exit.is_empty() && to_original.is_empty());
}

/// **A copy that was the whole block leaves it holding no code.** The
/// original still holds its own: the control.
#[test]
fn a_whole_block_copy_leaves_its_block_holding_nothing() {
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
    let mut test = SsaBlock::new(0);
    test.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    test.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
        left: g,
        right: zero,
        cmp: CmpKind::Eq,
        unsigned: false,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(test);
    for (index, at) in [(1u64, 0x10u64), (2, 0x20)] {
        let mut arm = SsaBlock::new(usize::try_from(index).unwrap_or_default());
        arm.provenance_mut().carry(at);
        arm.add_instruction(SsaInstruction::synthetic(
            CallData::new(None, Callee::Direct(7), vec![g]).into(),
        ));
        arm.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 3 }));
        ssa.add_block(arm);
    }
    let mut exit = SsaBlock::new(3);
    exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
    ssa.add_block(exit);
    ssa.recompute_uses();
    let merged = ssa.edit(SsaEditOptions::new(), |editor| {
        editor.merge_duplicate(1, 2, 0, &[])?;
        editor.declare_repaired();
        Ok(())
    });
    assert!(merged.is_ok(), "{merged:?} {ssa}");
    let carried = |block: usize| {
        ssa.block(block)
            .map(|block| block.provenance().carried().to_vec())
    };
    assert_eq!(carried(1), Some(Vec::new()), "{ssa}");
    assert_eq!(carried(2), Some(vec![0x20]), "control: {ssa}");
}

/// A copy inside the loop its original heads, entered knowing a fact about
/// the original's phi as it was on this iteration.
///
/// ```text
/// 0: goto 1
/// 1: p = phi(0: 0); two = 2; if (p == two) goto 4 else goto 2  (the original)
/// 2: if (p != 3) goto 5 else goto 6
/// 5: if (p == 3) goto 6 else goto 3
/// 3: q = p + 2; two = 2; if (q == two) goto 4 else goto 5     (the copy)
/// ```
///
/// `3` copies `1`'s test with `q` in `p`'s place and goes to `5` where `1`
/// goes on to `2`: on its way in, `p != 3`. Merged, `1` would take `q` for
/// `p`, and `2` would test `q`, of which nothing is known.
fn copy_beneath_its_original() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 8);
    let zero = var(&mut ssa, 0, 0, 0);
    let two = var(&mut ssa, 1, 0, 1);
    let p = phi_var(&mut ssa, 2, 1);
    let three_f = var(&mut ssa, 3, 2, 0);
    let three_d = var(&mut ssa, 4, 5, 0);
    let q = var(&mut ssa, 5, 3, 0);
    let two_o = var(&mut ssa, 6, 1, 0);
    let two_c = var(&mut ssa, 7, 3, 1);
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..7).map(SsaBlock::new).collect();
    let rows: [Vec<SsaInstruction<MockTarget>>; 7] = [
        vec![
            constant(zero, 0),
            constant(two, 2),
            instr(SsaOp::Jump { target: 1 }),
        ],
        vec![constant(two_o, 2), compare(p, two_o, CmpKind::Eq, (4, 2))],
        vec![
            constant(three_f, 3),
            compare(p, three_f, CmpKind::Ne, (5, 6)),
        ],
        vec![
            add(q, p, two),
            constant(two_c, 2),
            compare(q, two_c, CmpKind::Eq, (4, 5)),
        ],
        vec![instr(SsaOp::returns(None))],
        vec![
            constant(three_d, 3),
            compare(p, three_d, CmpKind::Eq, (6, 3)),
        ],
        vec![instr(SsaOp::returns(None))],
    ];
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    let mut header = PhiNode::new(p, VariableOrigin::Local(2));
    header.add_operand(PhiOperand::new(zero, 0));
    if let Some(block) = blocks.get_mut(1) {
        block.add_phi(header);
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **A fact about the original's own values is no fact on the merged
/// path.** The copy lies beneath the original, which dominates it: what the
/// copy's way in proved of the original's phi held of this iteration's value,
/// and merged, the test the copy jumped past would read the value the copy
/// carries in. Nothing is merged.
#[test]
fn a_copy_beneath_its_original_stays() {
    let mut ssa = copy_beneath_its_original();
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// How [`skipped`] differs from an edge threaded past a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Skipped {
    /// `0` proves `x == 45`, so `1`'s edge to `4` skips `2`'s `x != 0`.
    Answered,
    /// `0` tests `y`, and nothing is known of `x` where `1` branches.
    Unanswered,
    /// `4` takes `y` from `1` and `x` from `2`.
    OtherValue,
    /// `5` goes back to `2`, so `2` lies on a loop `1` is not on.
    IntoLoop,
    /// `1`'s edge to `4` runs through host code of its own that ends in a
    /// test.
    Passing,
    /// `1`'s edge to `4` runs through host code that only moves values.
    PassingMoves,
}

/// An edge a compiler threaded past a test, as `trapcmd` in dash writes
/// `if (action[0] == '-' && action[1] == '\0') ... else if (*action) ...`.
///
/// ```text
/// 0: if (x == 45) goto 1 else goto 2
/// 1: if (y != 0) goto 4 else goto 3      (straight to 4)
/// 2: if (x != 0) goto 4 else goto 5      (the test skipped)
/// 3: return
/// 4: r = phi(1: x, 2: x); return r
/// 5: return
/// ```
fn skipped(shape: Skipped) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(2, 8);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let x = argument(&mut ssa, 0);
    let y = argument(&mut ssa, 1);
    let dash = var(&mut ssa, 0, 0, 0);
    let zero_1 = var(&mut ssa, 1, 1, 0);
    let zero_2 = var(&mut ssa, 2, 2, 0);
    let r = phi_var(&mut ssa, 3, 4);
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let tested = if shape == Skipped::Unanswered { y } else { x };
    let rows: [Vec<SsaInstruction<MockTarget>>; 6] = [
        vec![
            constant(dash, 45),
            compare(tested, dash, CmpKind::Eq, (1, 2)),
        ],
        vec![constant(zero_1, 0), compare(y, zero_1, CmpKind::Ne, (4, 3))],
        vec![constant(zero_2, 0), compare(x, zero_2, CmpKind::Ne, (4, 5))],
        vec![instr(SsaOp::returns(None))],
        vec![instr(SsaOp::returns(Some(r)))],
        vec![if shape == Skipped::IntoLoop {
            instr(SsaOp::Jump { target: 2 })
        } else {
            instr(SsaOp::returns(None))
        }],
    ];
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..6).map(SsaBlock::new).collect();
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    let mut joined = PhiNode::new(r, VariableOrigin::Local(3));
    joined.add_operand(PhiOperand::new(
        if shape == Skipped::OtherValue { y } else { x },
        1,
    ));
    joined.add_operand(PhiOperand::new(x, 2));
    if let Some(block) = blocks.get_mut(4) {
        block.add_phi(joined);
    }
    if matches!(shape, Skipped::Passing | Skipped::PassingMoves)
        && let Some(block) = blocks.get_mut(1)
    {
        let mut own = BlockProvenance::default();
        if shape == Skipped::Passing {
            own.carry_test(0x40);
        } else {
            own.carry(0x40);
        }
        block.provenance_mut().pass_through_on(4, &own);
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// Whether `block`'s terminator branches to `target`.
fn branches_to(ssa: &SsaFunction<MockTarget>, block: usize, target: usize) -> bool {
    ssa.block(block)
        .is_some_and(|block| block.successors().contains(&target))
}

/// **An edge threaded past a test goes through the test again.** `1` is
/// entered knowing `x == 45` and goes straight to `4` where `2`'s `x != 0`
/// would send it: its edge goes to `2`, and `4` is entered from `2` alone, as
/// the source's `else if` enters it.
#[test]
fn an_edge_threaded_past_an_answered_test_goes_through_it() {
    let mut ssa = skipped(Skipped::Answered);
    let outcome = merge(&mut ssa);
    assert!(matches!(outcome, Ok(true)), "{outcome:?}");
    assert!(branches_to(&ssa, 1, 2), "{ssa:?}");
    assert!(!branches_to(&ssa, 1, 4), "{ssa:?}");
    assert!(
        ssa.block(4)
            .and_then(|block| block.phi_nodes().first())
            .is_none_or(|phi| phi.operand_from(1).is_none()),
        "{ssa:?}"
    );
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// **An edge past a test nothing answered stays.** The control of
/// [`an_edge_threaded_past_an_answered_test_goes_through_it`]: `0` tests
/// `y`, so where `1` branches nothing says where `2` would go.
#[test]
fn an_edge_past_an_unanswered_test_stays() {
    let mut ssa = skipped(Skipped::Unanswered);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **An edge carrying another value stays.** `4` takes `y` from `1` and `x`
/// from `2`: through `2`, `1`'s path would carry `x`.
#[test]
fn an_edge_carrying_another_value_stays() {
    let mut ssa = skipped(Skipped::OtherValue);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **An edge that runs through a test of its own skipped none.** `1`'s
/// edge to `4` is [`an_edge_threaded_past_an_answered_test_goes_through_it`]'s,
/// but the host runs code on it that ends in a test, which a fold decided
/// and the host executes there rather than `2`'s. The control passes code
/// that only moves values, as a coalesced copy does, and goes through `2`.
#[test]
fn an_edge_through_a_test_of_its_own_stays() {
    let mut ssa = skipped(Skipped::Passing);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
    let mut moves = skipped(Skipped::PassingMoves);
    assert!(matches!(merge(&mut moves), Ok(true)), "{moves:?}");
    assert!(branches_to(&moves, 1, 2), "{moves:?}");
}

/// **An edge enters no loop it did not.** `2` lies on the loop through `5`,
/// which `1` is not on: through `2`, `1` would enter the loop away from where
/// it is entered now.
#[test]
fn an_edge_into_a_loop_stays() {
    let mut ssa = skipped(Skipped::IntoLoop);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// A test that dominates the edge it would take back.
///
/// ```text
/// 0: goto 1
/// 1: if (x != 0) goto 2 else goto 5      (the test)
/// 2: goto 3
/// 3: if (x == 45) goto 4 else goto 5
/// 4: if (y != 0) goto 2 else goto 5      (back to 2, past 1's test)
/// 5: return
/// ```
fn skipped_beneath() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(2, 8);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let x = argument(&mut ssa, 0);
    let y = argument(&mut ssa, 1);
    let zero_1 = var(&mut ssa, 0, 1, 0);
    let dash = var(&mut ssa, 1, 3, 0);
    let zero_4 = var(&mut ssa, 2, 4, 0);
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let rows: [Vec<SsaInstruction<MockTarget>>; 6] = [
        vec![instr(SsaOp::Jump { target: 1 })],
        vec![constant(zero_1, 0), compare(x, zero_1, CmpKind::Ne, (2, 5))],
        vec![instr(SsaOp::Jump { target: 3 })],
        vec![constant(dash, 45), compare(x, dash, CmpKind::Eq, (4, 5))],
        vec![constant(zero_4, 0), compare(y, zero_4, CmpKind::Ne, (2, 5))],
        vec![instr(SsaOp::returns(None))],
    ];
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..6).map(SsaBlock::new).collect();
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **A test that dominates the edge is not taken again.** `4` knows
/// `x == 45`, so `1`'s `x != 0` would send it to `2` as it goes, but `1`
/// dominates `4`: through `1`, the edge would make a loop of `1` the source
/// never wrote. Nothing moves.
#[test]
fn an_edge_beneath_its_test_stays() {
    let mut ssa = skipped_beneath();
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// A `switch` compiled as a compare tree, each case breaking to its end.
///
/// ```text
/// 0: if (x == 0x69) goto 1 else goto 2
/// 1: *cell = 1; goto 4                     (case 'i': break)
/// 2: if (x == 0x61) goto 3 else goto 4     (decided false on 1's way in)
/// 3: *cell = 2; goto 4                     (case 'a': break)
/// 4: return                                (the end of the switch)
/// ```
fn compare_tree() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(1, 8);
    let x = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let i = var(&mut ssa, 0, 0, 0);
    let cell = var(&mut ssa, 1, 0, 1);
    let one = var(&mut ssa, 2, 1, 0);
    let a = var(&mut ssa, 3, 2, 0);
    let two = var(&mut ssa, 4, 3, 0);
    let compare = |right: SsaVarId, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left: x,
            right,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let store = |value: SsaVarId| {
        instr(SsaOp::StoreIndirect {
            addr: cell,
            value,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        })
    };
    let rows: [Vec<SsaInstruction<MockTarget>>; 5] = [
        vec![
            constant(i, 0x69),
            constant(cell, 0x1000),
            compare(i, (1, 2)),
        ],
        vec![
            constant(one, 1),
            store(one),
            instr(SsaOp::Jump { target: 4 }),
        ],
        vec![constant(a, 0x61), compare(a, (3, 4))],
        vec![
            constant(two, 2),
            store(two),
            instr(SsaOp::Jump { target: 4 }),
        ],
        vec![instr(SsaOp::returns(None))],
    ];
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..5).map(SsaBlock::new).collect();
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **A case's `break` is no test skipped.** Case `'i'` knows `x == 0x69`,
/// which decides the compare tree's next test to go where the case goes,
/// the end of the switch; but every case and the tree's default reach that
/// end, a join. The `break` stays a `break`.
#[test]
fn a_break_to_a_switchs_end_stays() {
    let mut ssa = compare_tree();
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// How [`exited`]'s loop breaks into the block past it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Exit {
    /// `3` reads `*cell` again and asks what `2`'s break answered.
    Retested,
    /// `2` writes `*cell` after it reads it: `3` reads another value on the
    /// break's way.
    Rewritten,
    /// `3` asks of `*other`, which nothing on the break's way answered.
    Unanswered,
    /// `4` takes from `3` what `3` computes, `w + 1`, and from `2` `v + 1`.
    Computed,
    /// As [`Self::Retested`], `3` reaching `4` through `6`, a block that only
    /// jumps there: the edge from `3` to `4` split, as dash's `expmeta` on
    /// RISC-V reaches what follows `out:`.
    Forwarded,
    /// As [`Self::Forwarded`], `6` defining the `7` that `4`'s `l` takes on
    /// its edge, as `expmeta`'s `reg_22 = 0`.
    ForwardedConstant,
}

/// A loop that breaks straight into the block past the test after it, as
/// dash's `evalcommand` inlines `fill_arglist`: `while ((argp = *argpp)) {
/// expandarg(argp, ...); *argpp = argp->next; if (*lastp) break; }` and then
/// `if (*lastp) ...`, the break sent past the second test.
///
/// ```text
/// 0: goto 1
/// 1: p = phi(0: first, 2: next); if (p == 0) goto 3 else goto 2
/// 2: *p = p; v = *cell; next = *p; if (v == 0) goto 1 else goto 4
/// 3: w = *cell; if (w != 0) goto 4 else goto 5      (the test skipped)
/// 4: a = phi(2: next, 3: p); l = phi(2: v, 3: w); return a
/// 5: return
/// ```
fn exited(shape: Exit) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(3, 8);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let first = argument(&mut ssa, 0);
    let cell = argument(&mut ssa, 1);
    let other = argument(&mut ssa, 2);
    let p = phi_var(&mut ssa, 0, 1);
    let zero_1 = var(&mut ssa, 1, 1, 0);
    let v = var(&mut ssa, 2, 2, 1);
    let next = var(&mut ssa, 3, 2, 2);
    let zero_2 = var(&mut ssa, 4, 2, 3);
    let w = var(&mut ssa, 5, 3, 0);
    let zero_3 = var(&mut ssa, 6, 3, 1);
    let a = phi_var(&mut ssa, 7, 4);
    let l = phi_var(&mut ssa, 8, 4);
    let one = var(&mut ssa, 9, 0, 0);
    let computed = shape == Exit::Computed;
    let (v_1, w_1) = if computed {
        (var(&mut ssa, 10, 2, 4), var(&mut ssa, 11, 3, 2))
    } else {
        (v, w)
    };
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let store = |addr: SsaVarId, value: SsaVarId| {
        instr(SsaOp::StoreIndirect {
            addr,
            value,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        })
    };
    let mut body = vec![
        store(p, p),
        load(v, cell),
        load(next, p),
        constant(zero_2, 0),
    ];
    if shape == Exit::Rewritten {
        body.push(store(cell, p));
    }
    if computed {
        body.push(add(v_1, v, one));
    }
    body.push(compare(v, zero_2, CmpKind::Eq, (1, 4)));
    let mut past = vec![
        load(
            w,
            if shape == Exit::Unanswered {
                other
            } else {
                cell
            },
        ),
        constant(zero_3, 0),
    ];
    if computed {
        past.push(add(w_1, w, one));
    }
    let constant_arm = shape == Exit::ForwardedConstant;
    let forwarded = shape == Exit::Forwarded || constant_arm;
    let seven = constant_arm.then(|| var(&mut ssa, 12, 6, 0));
    let arm = if forwarded { 6 } else { 4 };
    // The block `4` is entered from on `3`'s way.
    let tested = if forwarded { 6 } else { 3 };
    past.push(compare(w, zero_3, CmpKind::Ne, (arm, 5)));
    let mut rows: Vec<Vec<SsaInstruction<MockTarget>>> = vec![
        vec![constant(one, 1), instr(SsaOp::Jump { target: 1 })],
        vec![constant(zero_1, 0), compare(p, zero_1, CmpKind::Eq, (3, 2))],
        body,
        past,
        vec![instr(SsaOp::returns(Some(a)))],
        vec![instr(SsaOp::returns(None))],
    ];
    if forwarded {
        let mut arm = Vec::new();
        if let Some(seven) = seven {
            arm.push(constant(seven, 7));
        }
        arm.push(instr(SsaOp::Jump { target: 4 }));
        rows.push(arm);
    }
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..rows.len()).map(SsaBlock::new).collect();
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    let mut header = PhiNode::new(p, VariableOrigin::Local(0));
    header.add_operand(PhiOperand::new(first, 0));
    header.add_operand(PhiOperand::new(next, 2));
    let mut taken = PhiNode::new(a, VariableOrigin::Local(7));
    taken.add_operand(PhiOperand::new(next, 2));
    taken.add_operand(PhiOperand::new(p, tested));
    let mut read = PhiNode::new(l, VariableOrigin::Local(8));
    read.add_operand(PhiOperand::new(v_1, 2));
    read.add_operand(PhiOperand::new(seven.unwrap_or(w_1), tested));
    if let Some(block) = blocks.get_mut(1) {
        block.add_phi(header);
    }
    if let Some(block) = blocks.get_mut(4) {
        block.add_phi(taken);
        block.add_phi(read);
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **A loop's break sent past the test after the loop goes through it.**
/// `2` breaks to `4` where `v != 0`, and `3`, past the loop, asks `w != 0` of
/// `*cell` again: on the break's way nothing wrote the cell since `2` read
/// it, so `3` would send the break on to `4`. The break goes to `3`, which
/// joins the `p` it passed `4` with the break's `next`, and `4` is entered
/// from `3` alone, as the source's `if` after the loop enters it. So it is
/// where `3` reaches `4` through a block that only jumps there.
#[test]
fn a_break_past_the_test_after_its_loop_goes_through_it() {
    for shape in [Exit::Retested, Exit::Forwarded, Exit::ForwardedConstant] {
        let mut ssa = exited(shape);
        let outcome = merge(&mut ssa);
        assert!(matches!(outcome, Ok(true)), "{shape:?}: {outcome:?}");
        assert!(branches_to(&ssa, 2, 3), "{shape:?}: {ssa:?}");
        assert!(!branches_to(&ssa, 2, 4), "{shape:?}: {ssa:?}");
        let joined = ssa
            .block(3)
            .and_then(|block| block.phi_nodes().first())
            .cloned();
        let operands = joined.as_ref().map(|phi| {
            let mut operands: Vec<(usize, SsaVarId)> = phi
                .operands()
                .iter()
                .map(|operand| (operand.predecessor(), operand.value()))
                .collect();
            operands.sort_unstable();
            operands
        });
        // The loop's `p`, and the `next` it goes round with.
        let header = ssa.block(1).and_then(|block| block.phi_nodes().first());
        let p = header.map(PhiNode::result);
        let next = header
            .and_then(|phi| phi.operand_from(2))
            .map(PhiOperand::value);
        assert_eq!(
            operands,
            p.zip(next).map(|(p, next)| vec![(1, p), (2, next)]),
            "{shape:?}: {ssa:?}"
        );
        // Entered from `3` alone, `4` returns what `3` joined, its phis gone.
        let returned = ssa
            .block(4)
            .and_then(SsaBlock::control_terminator)
            .and_then(|op| match op {
                SsaOp::Return { value, .. } => *value,
                _ => None,
            });
        assert_eq!(
            returned,
            joined.map(|phi| phi.result()),
            "{shape:?}: {ssa:?}"
        );
        assert!(
            ssa.block(4).is_some_and(|block| block
                .phi_nodes()
                .iter()
                .all(|phi| phi.operand_from(2).is_none())),
            "{shape:?}: {ssa:?}"
        );
        assert_eq!(ssa.validate(), Ok(()));
        assert_eq!(
            SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
            Vec::new(),
            "{shape:?}: {ssa:?}"
        );
    }
}

/// **Past the test the break goes through, the loop's exit is still known.**
/// Once `2`'s break goes through `3`, `3` is entered from the loop's exit,
/// where `p == 0`, and from the break, where `v != 0`. The edge from `3` to `5`
/// answers `w == 0`, and on the break's way `w` reads what `v` read: nothing
/// wrote the cell since. So the edge is never taken from the break, and `5`
/// knows `p == 0`, which a null `p` joined into a variable that is no pointer
/// relies on. Read without the memory, `w` is another value, the break's way
/// stays possible, and `5` knows nothing of `p`.
#[test]
fn past_the_rerouted_test_the_loops_exit_is_known() {
    let mut ssa = exited(Exit::Retested);
    assert!(matches!(merge(&mut ssa), Ok(true)), "{ssa:?}");
    let context = AnalysisContext::new(MACHINE);
    let memory = MemorySsa::build(&ssa, &context);
    let cfg = SsaCfg::from_ssa(&ssa);
    let exception_blocks = ssa.exception_blocks();
    // The loop's `p`, and the zero its header compares it with.
    let p = ssa
        .block(1)
        .and_then(|block| block.phi_nodes().first())
        .map(PhiNode::result);
    let zero = ssa.block(1).and_then(|block| {
        block.instructions().iter().find_map(|row| match row.op() {
            SsaOp::Const { dest, .. } => Some(*dest),
            _ => None,
        })
    });
    let asked = p.zip(zero).map(|(p, zero)| Question::Compare {
        left: p,
        right: zero,
        cmp: CmpKind::Eq,
        unsigned: false,
    });
    assert!(asked.is_some(), "{ssa:?}");
    let reading = Definitions::of(&ssa).reading(&memory);
    let facts = GuardFacts::of(
        &ssa,
        &cfg,
        &exception_blocks,
        &reading,
        MACHINE.ptr_size,
        Some(&memory),
    );
    assert_eq!(
        asked.and_then(|asked| facts.answer(5, asked, &reading)),
        Some(true),
        "{ssa:?}"
    );
    let plain = Definitions::of(&ssa);
    let facts = GuardFacts::of(
        &ssa,
        &cfg,
        &exception_blocks,
        &plain,
        MACHINE.ptr_size,
        None,
    );
    assert_eq!(
        asked.and_then(|asked| facts.answer(5, asked, &plain)),
        None,
        "{ssa:?}"
    );
}

/// **A break past a cell written since it was read stays.** The control of
/// [`a_break_past_the_test_after_its_loop_goes_through_it`]: `2` writes
/// `*cell` after it reads it, so on the break's way `3` reads another value,
/// which nothing answered.
#[test]
fn a_break_past_a_rewritten_cell_stays() {
    let mut ssa = exited(Exit::Rewritten);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **A break past a test of another cell stays.** `3` asks of `*other`,
/// which `2`'s branch says nothing of.
#[test]
fn a_break_past_an_unanswered_test_stays() {
    let mut ssa = exited(Exit::Unanswered);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// **A break past a test whose value the block after it takes stays.** `4`
/// takes `w + 1` from `3`, which `3` computes: no value of `3`'s can be
/// joined with the break's.
#[test]
fn a_break_past_a_test_computing_what_it_passes_stays() {
    let mut ssa = exited(Exit::Computed);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}

/// The test a loop goes round by, copied ahead of the loop without the step
/// its latch makes first, as dash's `pmatch` peels the `?` of its pattern:
///
/// ```text
/// 0: c0 = *p0; p1 = p0 + 1; if (c0 == 63) goto 3 else goto 1       (the copy)
/// 1: p = phi(0: p1, 2: p2); q = phi(0: q0, 2: q2); if (*q == 0) goto 4 else goto 2
/// 2: pl = phi(1: p, 3: p3); ql = phi(1: q, 3: q3);
///    q2 = ql + 1; c2 = *pl; p2 = pl + 1; if (c2 != 63) goto 1 else goto 3   (the original)
/// 3: p3 = phi(0: p1, 2: p2); q3 = phi(0: q0, 2: q2); if (*q3 == 0) goto 4 else goto 2
/// 4: return
/// ```
///
/// `controls` makes the step `q2` the value the copied rows read, which the
/// copy computed otherwise.
fn peeled(controls: bool) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(2, 16);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let p0 = argument(&mut ssa, 0);
    let q0 = argument(&mut ssa, 1);
    let c0 = var(&mut ssa, 0, 0, 0);
    let one_0 = var(&mut ssa, 1, 0, 1);
    let p1 = var(&mut ssa, 2, 0, 2);
    let mark_0 = var(&mut ssa, 3, 0, 3);
    let p = phi_var(&mut ssa, 4, 1);
    let q = phi_var(&mut ssa, 5, 1);
    let at_1 = var(&mut ssa, 6, 1, 0);
    let zero_1 = var(&mut ssa, 7, 1, 1);
    let pl = phi_var(&mut ssa, 8, 2);
    let ql = phi_var(&mut ssa, 9, 2);
    let one_2 = var(&mut ssa, 10, 2, 0);
    let q2 = var(&mut ssa, 11, 2, 1);
    let c2 = var(&mut ssa, 12, 2, 2);
    let one_3 = var(&mut ssa, 13, 2, 3);
    let p2 = var(&mut ssa, 14, 2, 4);
    let mark_2 = var(&mut ssa, 15, 2, 5);
    let p3 = phi_var(&mut ssa, 16, 3);
    let q3 = phi_var(&mut ssa, 17, 3);
    let at_3 = var(&mut ssa, 18, 3, 0);
    let zero_3 = var(&mut ssa, 19, 3, 1);
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let rows: [Vec<SsaInstruction<MockTarget>>; 5] = [
        vec![
            load(c0, p0),
            constant(one_0, 1),
            add(p1, p0, one_0),
            constant(mark_0, 63),
            compare(c0, mark_0, CmpKind::Eq, (3, 1)),
        ],
        vec![
            load(at_1, q),
            constant(zero_1, 0),
            compare(at_1, zero_1, CmpKind::Eq, (4, 2)),
        ],
        vec![
            constant(one_2, 1),
            add(q2, ql, one_2),
            load(c2, if controls { q2 } else { pl }),
            constant(one_3, 1),
            add(p2, pl, one_3),
            constant(mark_2, 63),
            compare(c2, mark_2, CmpKind::Ne, (1, 3)),
        ],
        vec![
            load(at_3, q3),
            constant(zero_3, 0),
            compare(at_3, zero_3, CmpKind::Eq, (4, 2)),
        ],
        vec![instr(SsaOp::returns(None))],
    ];
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..5).map(SsaBlock::new).collect();
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    let joins = [
        (1, p, VariableOrigin::Local(4), [(p1, 0), (p2, 2)]),
        (1, q, VariableOrigin::Local(5), [(q0, 0), (q2, 2)]),
        (2, pl, VariableOrigin::Local(8), [(p, 1), (p3, 3)]),
        (2, ql, VariableOrigin::Local(9), [(q, 1), (q3, 3)]),
        (3, p3, VariableOrigin::Local(16), [(p1, 0), (p2, 2)]),
        (3, q3, VariableOrigin::Local(17), [(q0, 0), (q2, 2)]),
    ];
    for (at, result, origin, operands) in joins {
        let mut phi = PhiNode::new(result, origin);
        for (value, from) in operands {
            phi.add_operand(PhiOperand::new(value, from));
        }
        if let Some(block) = blocks.get_mut(at) {
            block.add_phi(phi);
        }
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **A loop's test copied ahead of it without its latch's step is the
/// latch's test.** `0` reads the pattern's next character and asks `c ==
/// 63`; `2`, the latch, steps `q` first, then reads it and asks `c != 63`,
/// the same test the other way round. `2` is split where its copied rows
/// begin, and `0` jumps there: the split takes `q`'s step from `2` and `q0`
/// from `0`, and the blocks the test leads to take `q` from the split. The
/// control reads the character through the step, which `0` has not taken:
/// no row of `0` is a copy, and `0` keeps its read. Its test and the
/// latch's still ask one question and pick between `1` and `3`, the two
/// blocks the loop is entered at, and become one test of their own
/// ([`a_loop_entered_at_two_blocks_by_one_copied_test_is_entered_at_it`]).
#[test]
fn a_test_copied_without_its_latchs_step_is_the_latchs_test() {
    let mut ssa = peeled(false);
    let outcome = merge(&mut ssa);
    assert!(matches!(outcome, Ok(true)), "{outcome:?}");
    let split = ssa
        .block(0)
        .and_then(|block| block.instructions().last())
        .and_then(|row| match row.op() {
            SsaOp::Jump { target } => Some(*target),
            _ => None,
        });
    assert!(split.is_some_and(|split| split >= 5), "{ssa:?}");
    let split = split.unwrap_or(0);
    assert!(branches_to(&ssa, 2, split), "{ssa:?}");
    for target in [1, 3] {
        assert!(branches_to(&ssa, split, target), "{ssa:?}");
        assert!(
            ssa.block(target).is_some_and(|block| block
                .phi_nodes()
                .iter()
                .all(|phi| phi.operand_from(0).is_none() && phi.operand_from(2).is_none())),
            "B{target} takes from the split alone: {ssa:?}"
        );
    }
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );

    let mut control = peeled(true);
    assert!(matches!(merge(&mut control), Ok(true)), "{control:?}");
    assert!(
        control.block(0).is_some_and(|block| block
            .instructions()
            .iter()
            .any(|row| matches!(row.op(), SsaOp::LoadIndirect { .. }))),
        "{control:?}"
    );
    let test = control.block_count().saturating_sub(1);
    assert_eq!(entered_from(&control, test), [0, 2], "{control:?}");
    for target in [1, 3] {
        assert_eq!(entered_from(&control, target), [test], "{control:?}");
    }
    assert_eq!(
        SsaVerifier::new(&control).verify(VerifyLevel::Standard),
        Vec::new(),
        "{control:?}"
    );
}

/// A loop with two ways out past the test after it, as dash's `expmeta`
/// leaves its scan of a pattern: `if (metaflag) break;` asks the flag the test
/// past the loop asks, and the way that sets `metaflag = 1` leaves at its
/// latch. The compiler sends both straight into `out:`'s arm.
///
/// ```text
/// 0: zero = 0; goto 1
/// 1: flag = phi(0: zero, 5: carried); if (m < 10) goto 2 else goto 3
/// 2: if (flag != 0u) goto 7 else goto 4              (asks the flag)
/// 3: set = value; if (n == 3) goto 5 else goto 7     (sets it, leaves)
/// 4: if (m < 20) goto 5 else goto 6
/// 5: carried = phi(3: set, 4: flag); goto 1
/// 6: if (flag != 0) goto 7 else goto 8               (the test skipped)
/// 7: return n
/// 8: return n
/// ```
fn flagged(value: i32) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(2, 8);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let n = argument(&mut ssa, 0);
    let m = argument(&mut ssa, 1);
    let zero = var(&mut ssa, 0, 0, 0);
    let flag = phi_var(&mut ssa, 0, 1);
    let ten = var(&mut ssa, 1, 1, 0);
    let unsigned_zero = var(&mut ssa, 2, 2, 0);
    let set = var(&mut ssa, 0, 3, 0);
    let three = var(&mut ssa, 3, 3, 1);
    let twenty = var(&mut ssa, 4, 4, 0);
    let carried = phi_var(&mut ssa, 0, 5);
    let zero_6 = var(&mut ssa, 5, 6, 0);
    let constant =
        |dest: SsaVarId, value: ConstValue<MockTarget>| instr(SsaOp::Const { dest, value });
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let rows: [Vec<SsaInstruction<MockTarget>>; 9] = [
        vec![
            constant(zero, ConstValue::I32(0)),
            instr(SsaOp::Jump { target: 1 }),
        ],
        vec![
            constant(ten, ConstValue::I32(10)),
            compare(m, ten, CmpKind::Lt, (2, 3)),
        ],
        vec![
            constant(unsigned_zero, ConstValue::U32(0)),
            compare(flag, unsigned_zero, CmpKind::Ne, (7, 4)),
        ],
        vec![
            constant(set, ConstValue::I32(value)),
            constant(three, ConstValue::I32(3)),
            compare(n, three, CmpKind::Eq, (5, 7)),
        ],
        vec![
            constant(twenty, ConstValue::I32(20)),
            compare(m, twenty, CmpKind::Lt, (5, 6)),
        ],
        vec![instr(SsaOp::Jump { target: 1 })],
        vec![
            constant(zero_6, ConstValue::I32(0)),
            compare(flag, zero_6, CmpKind::Ne, (7, 8)),
        ],
        vec![instr(SsaOp::returns(Some(n)))],
        vec![instr(SsaOp::returns(Some(n)))],
    ];
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..9).map(SsaBlock::new).collect();
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    let mut header = PhiNode::new(flag, VariableOrigin::Local(0));
    header.add_operand(PhiOperand::new(zero, 0));
    header.add_operand(PhiOperand::new(carried, 5));
    let mut latch = PhiNode::new(carried, VariableOrigin::Local(0));
    latch.add_operand(PhiOperand::new(set, 3));
    latch.add_operand(PhiOperand::new(flag, 4));
    if let Some(block) = blocks.get_mut(1) {
        block.add_phi(header);
    }
    if let Some(block) = blocks.get_mut(5) {
        block.add_phi(latch);
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **The ways out of a loop past the test after it go through it
/// together.** `2` breaks to `7` where the flag is set, and `3` leaves for
/// `7` past setting it to `1`, the version its way on to the header carries:
/// `6`, past the loop, would send each of them on to `7`. Both go to `6`,
/// which joins the flag with the `1` `3` set, and `7` is entered from `6`
/// alone, as `out:`'s `if (metaflag)` enters its arm. `2` compares with an
/// unsigned zero where `6` compares with a signed one: a comparison reads
/// its own signedness, and the two ask one question.
#[test]
fn the_ways_out_past_the_test_after_a_loop_go_through_it_together() {
    let mut ssa = flagged(1);
    let flag = ssa
        .block(1)
        .and_then(|block| block.phi_nodes().first())
        .map(PhiNode::result);
    let set = ssa
        .block(3)
        .and_then(|block| block.instructions().first())
        .and_then(|row| row.op().dest());
    let outcome = merge(&mut ssa);
    assert!(matches!(outcome, Ok(true)), "{outcome:?}");
    for exit in [2, 3] {
        assert!(branches_to(&ssa, exit, 6), "{ssa:?}");
        assert!(!branches_to(&ssa, exit, 7), "{ssa:?}");
    }
    let joined = ssa.block(6).and_then(|block| block.phi_nodes().first());
    let operands = joined.map(|phi| {
        let mut operands: Vec<(usize, Option<SsaVarId>)> = phi
            .operands()
            .iter()
            .map(|operand| (operand.predecessor(), Some(operand.value())))
            .collect();
        operands.sort_unstable();
        operands
    });
    assert_eq!(
        operands,
        Some(vec![(2, flag), (3, set), (4, flag)]),
        "{ssa:?}"
    );
    let asked = ssa
        .block(6)
        .and_then(SsaBlock::control_terminator)
        .is_some_and(|op| {
            joined
                .map(PhiNode::result)
                .is_some_and(|result| op.uses().contains(&result))
        });
    assert!(asked, "6 asks the joined flag: {ssa:?}");
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// [`flagged`]'s loop with its latch testing whether to go round, as
/// `expmeta`'s is once its copies are one test: the test past the loop reads
/// the latch's flag, and the way on from `2`'s `if (flag) break` steps `n`
/// before it reaches the latch.
///
/// ```text
/// 0: zero = 0; goto 1
/// 1: flag = phi(0: zero, 5: kept); if (n < 10) goto 2 else goto 3
/// 2: if (flag != 0u) goto 7 else goto 4              (asks the flag)
/// 3: set = 1; if (n == 3) goto 5 else goto 7         (sets it, leaves)
/// 4: m = n + n; goto 5
/// 5: kept = phi(3: set, 4: flag); if (n < 20) goto 1 else goto 6
/// 6: if (kept != 0) goto 7 else goto 8               (the test skipped)
/// 7: return n
/// 8: return n
/// ```
fn flagged_at_latch() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(1, 8);
    let n = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let zero = var(&mut ssa, 0, 0, 0);
    let flag = phi_var(&mut ssa, 0, 1);
    let ten = var(&mut ssa, 1, 1, 0);
    let unsigned_zero = var(&mut ssa, 2, 2, 0);
    let set = var(&mut ssa, 0, 3, 0);
    let three = var(&mut ssa, 3, 3, 1);
    let m = var(&mut ssa, 6, 4, 0);
    let kept = phi_var(&mut ssa, 0, 5);
    let twenty = var(&mut ssa, 4, 5, 0);
    let zero_6 = var(&mut ssa, 5, 6, 0);
    let constant =
        |dest: SsaVarId, value: ConstValue<MockTarget>| instr(SsaOp::Const { dest, value });
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let rows: [Vec<SsaInstruction<MockTarget>>; 9] = [
        vec![
            constant(zero, ConstValue::I32(0)),
            instr(SsaOp::Jump { target: 1 }),
        ],
        vec![
            constant(ten, ConstValue::I32(10)),
            compare(n, ten, CmpKind::Lt, (2, 3)),
        ],
        vec![
            constant(unsigned_zero, ConstValue::U32(0)),
            compare(flag, unsigned_zero, CmpKind::Ne, (7, 4)),
        ],
        vec![
            constant(set, ConstValue::I32(1)),
            constant(three, ConstValue::I32(3)),
            compare(n, three, CmpKind::Eq, (5, 7)),
        ],
        vec![add(m, n, n), instr(SsaOp::Jump { target: 5 })],
        vec![
            constant(twenty, ConstValue::I32(20)),
            compare(n, twenty, CmpKind::Lt, (1, 6)),
        ],
        vec![
            constant(zero_6, ConstValue::I32(0)),
            compare(kept, zero_6, CmpKind::Ne, (7, 8)),
        ],
        vec![instr(SsaOp::returns(Some(n)))],
        vec![instr(SsaOp::returns(Some(n)))],
    ];
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..9).map(SsaBlock::new).collect();
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    let mut header = PhiNode::new(flag, VariableOrigin::Local(0));
    header.add_operand(PhiOperand::new(zero, 0));
    header.add_operand(PhiOperand::new(kept, 5));
    let mut latch = PhiNode::new(kept, VariableOrigin::Local(0));
    latch.add_operand(PhiOperand::new(set, 3));
    latch.add_operand(PhiOperand::new(flag, 4));
    if let Some(block) = blocks.get_mut(1) {
        block.add_phi(header);
    }
    if let Some(block) = blocks.get_mut(5) {
        block.add_phi(latch);
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **A way out holds the version its way on to the latch keeps.** In
/// [`flagged_at_latch`] the test past the loop reads the latch's flag. On
/// `2`'s way on, through `4`, which steps `n` and joins nothing, the latch
/// takes the header's flag, which `2`'s own `flag != 0` answered; on `3`'s,
/// the `1` it set. Both go through `6`, which joins the latch's flag with
/// those two.
#[test]
fn a_way_out_holds_what_its_way_on_to_the_latch_keeps() {
    let mut ssa = flagged_at_latch();
    let flag = ssa
        .block(1)
        .and_then(|block| block.phi_nodes().first())
        .map(PhiNode::result);
    let kept = ssa
        .block(5)
        .and_then(|block| block.phi_nodes().first())
        .map(PhiNode::result);
    let set = ssa
        .block(3)
        .and_then(|block| block.instructions().first())
        .and_then(|row| row.op().dest());
    let outcome = merge(&mut ssa);
    assert!(matches!(outcome, Ok(true)), "{outcome:?}");
    for exit in [2, 3] {
        assert!(branches_to(&ssa, exit, 6), "{ssa:?}");
    }
    let operands = ssa
        .block(6)
        .and_then(|block| block.phi_nodes().first())
        .map(|phi| {
            let mut operands: Vec<(usize, Option<SsaVarId>)> = phi
                .operands()
                .iter()
                .map(|operand| (operand.predecessor(), Some(operand.value())))
                .collect();
            operands.sort_unstable();
            operands
        });
    assert_eq!(
        operands,
        Some(vec![(2, flag), (3, set), (5, kept)]),
        "{ssa:?}"
    );
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// **Jump threading leaves the ways out going through the test.** Once
/// [`flagged`]'s ways out go through `6`, the flag `3` brings decides `6` on
/// `3`'s edge, and `2`'s own branch decides it on `2`'s: threading either
/// past it would build again the join unthreading undid, every round of the
/// pipeline. `6` is the source's test after its loop, and asks one thing; a
/// flattener's dispatcher asks several.
#[test]
fn threading_leaves_the_ways_out_going_through_the_test() {
    let mut ssa = flagged(1);
    let outcome = merge(&mut ssa);
    assert!(matches!(outcome, Ok(true)), "{outcome:?}");
    let changed = JumpThreadingPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT),
    ) == Ok(true);
    assert!(!changed, "{ssa:?}");
    for exit in [2, 3] {
        assert!(branches_to(&ssa, exit, 6), "{ssa:?}");
    }
}

/// A loop whose one way out carries the state the test after it asks, as a
/// flattener's dispatcher leaves for the case past its loop:
///
/// ```text
/// 0: zero = 0; three = 3; goto 1
/// 1: i = phi(0: zero, 2: next); if (i < n) goto 2 else goto 3
/// 2: one = 1; next = i + one; goto 1
/// 3: state = phi(1: three); asked = 3; if (state == asked) goto 4 else goto 5
/// 4: return i
/// 5: return n
/// ```
fn dispatched_exit() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(1, 8);
    let n = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let zero = var(&mut ssa, 0, 0, 0);
    let three = var(&mut ssa, 1, 0, 1);
    let i = phi_var(&mut ssa, 0, 1);
    let one = var(&mut ssa, 2, 2, 0);
    let next = var(&mut ssa, 0, 2, 1);
    let state = phi_var(&mut ssa, 1, 3);
    let asked = var(&mut ssa, 3, 3, 0);
    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let rows: [Vec<SsaInstruction<MockTarget>>; 6] = [
        vec![
            constant(zero, 0),
            constant(three, 3),
            instr(SsaOp::Jump { target: 1 }),
        ],
        vec![compare(i, n, CmpKind::Lt, (2, 3))],
        vec![
            constant(one, 1),
            add(next, i, one),
            instr(SsaOp::Jump { target: 1 }),
        ],
        vec![
            constant(asked, 3),
            compare(state, asked, CmpKind::Eq, (4, 5)),
        ],
        vec![instr(SsaOp::returns(Some(i)))],
        vec![instr(SsaOp::returns(Some(n)))],
    ];
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..6).map(SsaBlock::new).collect();
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    let mut header = PhiNode::new(i, VariableOrigin::Local(0));
    header.add_operand(PhiOperand::new(zero, 0));
    header.add_operand(PhiOperand::new(next, 2));
    let mut carried = PhiNode::new(state, VariableOrigin::Local(1));
    carried.add_operand(PhiOperand::new(three, 1));
    if let Some(block) = blocks.get_mut(1) {
        block.add_phi(header);
    }
    if let Some(block) = blocks.get_mut(3) {
        block.add_phi(carried);
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// **A way out past a test that runs no more is threaded past it.** The
/// counterpart of [`threading_leaves_the_ways_out_going_through_the_test`]:
/// in [`dispatched_exit`] the loop's one way out carries the state `3` asks,
/// so threaded past it the test is entered from nowhere, and its other case
/// is gone with it, as a flattened function's dead default is.
#[test]
fn threading_takes_a_way_out_past_a_test_it_alone_enters() {
    let mut ssa = dispatched_exit();
    let changed = JumpThreadingPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &NullListener, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT),
    ) == Ok(true);
    assert!(changed, "{ssa:?}");
    assert!(branches_to(&ssa, 1, 4), "{ssa:?}");
    assert!(!branches_to(&ssa, 1, 3), "{ssa:?}");
}

/// **A way out the test would send elsewhere keeps every way where it is.**
/// The control of
/// [`the_ways_out_past_the_test_after_a_loop_go_through_it_together`]: `3`
/// sets the flag to `0`, so `6` would send it to `8`. `7` is then a join of
/// ways the test does not decide, and neither edge moves.
#[test]
fn a_way_out_the_test_would_send_elsewhere_keeps_both_ways() {
    let mut ssa = flagged(0);
    let outcome = merge(&mut ssa);
    assert!(matches!(outcome, Ok(false)), "{outcome:?}: {ssa:?}");
    assert!(branches_to(&ssa, 2, 7), "{ssa:?}");
    assert!(branches_to(&ssa, 3, 7), "{ssa:?}");
}

/// How [`joined`] is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Joined {
    /// Both ways into the join test `n == 0`, each sending its other answer
    /// where the test the join goes on to sends it on that way.
    Copied,
    /// The second way tests `n == 1`.
    OtherQuestion,
    /// The second way's other answer goes where the onward test would not
    /// send it on that way.
    WrongArm,
    /// The way from the onward test to the first copy's other answer loads a
    /// cell the copy's way loaded too, unchanged.
    Loaded,
    /// As [`Self::Loaded`], the first copy storing to the cell first.
    Stored,
    /// As [`Self::Copied`], each copy branching on a comparison of its own:
    /// `c = n == zero; if (c) ...`.
    OwnConditions,
}

/// A test copied into both ways into a join, as `binop2` in dash computes
/// `a = noeval ? b : do_binop(op, a, b)` and then tests the next operator:
///
/// ```text
/// 0: zero = 0; one = 1; ten = 10; [cell = 0x1000; m0 = *cell;]
///    if (k < ten) goto 1 else goto 2
/// 1: [*cell = x;] if (n == zero) goto 3 else goto 5
/// 2: if (n == zero) goto 3 else goto 6
/// 3: s = x + one; goto 4
/// 4: a = phi(3: s); if (k < ten) goto 5 [7] else goto 6
/// 7: m7 = *cell; goto 5
/// 5: r = phi(4 [7]: a, 1: x); [m = phi(7: m7, 1: m0); u = r + m;] return r [u]
/// 6: q = phi(4: a, 2: x); t = q + one; return t
/// ```
///
/// The copies are `1` and `2`; `1` is entered knowing `k < ten`, so its
/// `n != 0` goes straight to where `4` would send it, and `2` likewise.
/// Returns the function with `n`, `k` and `x`.
fn joined(shape: Joined) -> (SsaFunction<MockTarget>, [SsaVarId; 3]) {
    let loaded = matches!(shape, Joined::Loaded | Joined::Stored);
    let mut ssa = SsaFunction::new(3, 8);
    let argument = |ssa: &mut SsaFunction<MockTarget>, index: u16| {
        ssa.create_variable(
            VariableOrigin::Argument(index),
            0,
            DefSite::entry(),
            MockType::I32,
        )
    };
    let n = argument(&mut ssa, 0);
    let k = argument(&mut ssa, 1);
    let x = argument(&mut ssa, 2);
    let zero = var(&mut ssa, 0, 0, 0);
    let one = var(&mut ssa, 1, 0, 1);
    let ten = var(&mut ssa, 2, 0, 2);
    // The loaded shapes' cell and the entry's load of it.
    let entry_read = loaded.then(|| (var(&mut ssa, 3, 0, 3), var(&mut ssa, 4, 0, 4)));
    let s = var(&mut ssa, 5, 3, 0);
    let a = phi_var(&mut ssa, 6, 4);
    let r = phi_var(&mut ssa, 8, 5);
    // The loaded shapes' load on the way, the phi taking it and the sum.
    let read = loaded.then(|| {
        (
            var(&mut ssa, 7, 7, 0),
            phi_var(&mut ssa, 9, 5),
            var(&mut ssa, 10, 5, 0),
        )
    });
    let q = phi_var(&mut ssa, 11, 6);
    let t = var(&mut ssa, 12, 6, 0);
    // The copies' own comparisons.
    let own = (shape == Joined::OwnConditions)
        .then(|| (var(&mut ssa, 13, 1, 0), var(&mut ssa, 14, 2, 0)));

    let constant = |dest: SsaVarId, value: i32| {
        instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        })
    };
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..8).map(SsaBlock::new).collect();
    let mut fill = |index: usize, rows: Vec<SsaInstruction<MockTarget>>| {
        if let Some(block) = blocks.get_mut(index) {
            for row in rows {
                block.add_instruction(row);
            }
        }
    };
    let mut entry = vec![constant(zero, 0), constant(one, 1), constant(ten, 10)];
    if let Some((cell, m0)) = entry_read {
        entry.push(constant(cell, 0x1000));
        entry.push(load(m0, cell));
    }
    entry.push(compare(k, ten, CmpKind::Lt, (1, 2)));
    fill(0, entry);
    let mut first = Vec::new();
    if shape == Joined::Stored
        && let Some((cell, _)) = entry_read
    {
        first.push(instr(SsaOp::StoreIndirect {
            addr: cell,
            value: x,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
    }
    let own_branch = |condition: SsaVarId, to: (usize, usize)| {
        vec![
            instr(SsaOp::Ceq {
                dest: condition,
                left: n,
                right: zero,
                signaling: false,
            }),
            instr(SsaOp::Branch {
                condition,
                true_target: to.0,
                false_target: to.1,
            }),
        ]
    };
    match own {
        Some((first_condition, _)) => first.extend(own_branch(first_condition, (3, 5))),
        None => first.push(compare(n, zero, CmpKind::Eq, (3, 5))),
    }
    fill(1, first);
    let (second_asks, second_other) = match shape {
        Joined::OtherQuestion => (one, 6),
        Joined::WrongArm => (zero, 5),
        _ => (zero, 6),
    };
    match own {
        Some((_, second_condition)) => fill(2, own_branch(second_condition, (3, second_other))),
        None => fill(
            2,
            vec![compare(n, second_asks, CmpKind::Eq, (3, second_other))],
        ),
    }
    fill(3, vec![add(s, x, one), instr(SsaOp::Jump { target: 4 })]);
    fill(
        4,
        vec![compare(
            k,
            ten,
            CmpKind::Lt,
            (if loaded { 7 } else { 5 }, 6),
        )],
    );
    if let (Some((m7, m, u)), Some((cell, _))) = (read, entry_read) {
        fill(7, vec![load(m7, cell), instr(SsaOp::Jump { target: 5 })]);
        fill(5, vec![add(u, r, m), instr(SsaOp::returns(Some(u)))]);
    } else {
        fill(5, vec![instr(SsaOp::returns(Some(r)))]);
    }
    fill(6, vec![add(t, q, one), instr(SsaOp::returns(Some(t)))]);
    let phi = |result: SsaVarId, idx: u16, operands: &[(SsaVarId, usize)]| {
        let mut node = PhiNode::new(result, VariableOrigin::Local(idx));
        for (value, from) in operands {
            node.add_operand(PhiOperand::new(*value, *from));
        }
        node
    };
    let onward_to_five = if loaded { 7 } else { 4 };
    let mut into_five = vec![(a, onward_to_five), (x, 1)];
    let mut into_six = vec![(a, 4)];
    if second_other == 5 {
        into_five.push((x, 2));
    } else {
        into_six.push((x, 2));
    }
    if let Some(block) = blocks.get_mut(4) {
        block.add_phi(phi(a, 6, &[(s, 3)]));
    }
    if let Some(block) = blocks.get_mut(5) {
        block.add_phi(phi(r, 8, &into_five));
        if let (Some((m7, m, _)), Some((_, m0))) = (read, entry_read) {
            block.add_phi(phi(m, 9, &[(m7, 7), (m0, 1)]));
        }
    }
    if let Some(block) = blocks.get_mut(6) {
        block.add_phi(phi(q, 11, &into_six));
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    (ssa, [n, k, x])
}

/// The blocks of `ssa` branching on `n`.
fn tests_of(ssa: &SsaFunction<MockTarget>, n: SsaVarId) -> usize {
    ssa.blocks()
        .iter()
        .filter(|block| {
            matches!(block.control_terminator(), Some(SsaOp::BranchCmp { left, .. }) if *left == n)
        })
        .count()
}

/// What `ssa` returns for `values` of `n`, `k` and `x`.
fn joined_returns(
    ssa: &SsaFunction<MockTarget>,
    inputs: [SsaVarId; 3],
    values: [i32; 3],
) -> Option<i64> {
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

/// **A test copied into each way into a join is one test.** `1` and `2` both
/// test `n == 0`, and each sends `n != 0` straight to where `4` sends it on
/// that way: they become one test, entered from each, that goes to the join
/// or to `4`, and every input returns what it returned. Copies that each
/// compute their own comparison are one test too, of a phi of what each
/// computed.
#[test]
fn a_test_copied_into_each_way_into_a_join_is_one_test() {
    for shape in [Joined::Copied, Joined::OwnConditions] {
        copied_tests_merge(shape);
    }
}

/// Asserts the copies in `shape` merge into one test and the function
/// returns what it returned for every input.
fn copied_tests_merge(shape: Joined) {
    let (mut ssa, inputs) = joined(shape);
    let mut values = Vec::new();
    for n in [0, 1] {
        for k in [3, 12] {
            for x in [-4, 5] {
                values.push([n, k, x]);
            }
        }
    }
    let before: Vec<Option<i64>> = values
        .iter()
        .map(|values| joined_returns(&ssa, inputs, *values))
        .collect();
    assert!(before.iter().all(Option::is_some), "{shape:?}: {before:?}");
    assert!(matches!(merge(&mut ssa), Ok(true)), "{shape:?}: {ssa:?}");
    let branches = ssa
        .blocks()
        .iter()
        .filter(|block| {
            block
                .control_terminator()
                .and_then(SsaOp::two_way_targets)
                .is_some()
        })
        .count();
    // `0`'s and `4`'s tests of `k`, and one of `n`.
    assert_eq!(branches, 3, "{shape:?}: one test of n: {ssa:?}");
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
    let after: Vec<Option<i64>> = values
        .iter()
        .map(|values| joined_returns(&ssa, inputs, *values))
        .collect();
    assert_eq!(before, after, "{shape:?}");
}

/// **A load on the way that reads what the copy's way read merges; one past
/// a store does not.** On the way from `4` to `5`, `7` loads the cell `0`
/// loaded: unchanged on `1`'s way, it is the same value, and the copies
/// merge. Where `1` stores to the cell first, the load reads another value
/// than the one `1` carries, and nothing merges.
#[test]
fn a_load_on_the_way_merges_only_where_the_cell_is_unchanged() {
    let (mut ssa, inputs) = joined(Joined::Loaded);
    assert!(matches!(merge(&mut ssa), Ok(true)), "{ssa:?}");
    assert_eq!(tests_of(&ssa, inputs[0]), 1, "{ssa:?}");
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
    let (mut stored, inputs) = joined(Joined::Stored);
    assert!(merge(&mut stored).is_ok(), "{stored:?}");
    assert_eq!(tests_of(&stored, inputs[0]), 2, "{stored:?}");
}

/// **Copies asking another question, or sending an answer elsewhere, stay.**
/// The controls of [`a_test_copied_into_each_way_into_a_join_is_one_test`]:
/// `2` asking `n == 1` is no copy of `1`, and `2` sending `n != 0` to `5`,
/// where `4` would send its way to `6`, is no jump past `4`.
#[test]
fn copies_asking_otherwise_or_going_elsewhere_stay() {
    for shape in [Joined::OtherQuestion, Joined::WrongArm] {
        let (mut ssa, inputs) = joined(shape);
        assert!(merge(&mut ssa).is_ok(), "{shape:?}");
        assert_eq!(tests_of(&ssa, inputs[0]), 2, "{shape:?}: {ssa:?}");
    }
}

/// How [`guarded`] varies from the loop a compiler rotated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Guarded {
    /// The guard copies the loop's test: `c == 0` of the byte where the
    /// loop asks `ch != 0` of the word it widened.
    Copy,
    /// The guard asks `c == 1`, another question.
    OtherConstant,
    /// The test after the loop stores the flag first: the guard's way past
    /// the loop never ran it.
    Effect,
    /// The test after the loop reads the loop's `x` outside a phi: the
    /// guard's way past the loop holds none.
    ReadsTheLoop,
}

/// A loop a compiler rotated to test at its foot, its test copied ahead of
/// it, as dash's `expmeta` scans its name: `for (p = name; *p; p++) if (*p ==
/// 3) metaflag = 1;` and then `if (metaflag == 0) ...`, the copy's way past
/// the loop sent straight past that test.
///
/// ```text
/// 0: c32 = zext c; if (c == 0) goto 6 else goto 1    (the copy)
/// 1: zero = 0; goto 2
/// 2: f = phi(1: zero, 4: fl); x = phi(1: c32, 4: ch); d = x - 1;
///    if (x == 3) goto 3 else goto 4
/// 3: one = 1; goto 4
/// 4: fl = phi(2: f, 3: one); ch = phi(2: d, 3: d); if (ch != 0) goto 2 else goto 5
/// 5: g = phi(4: fl); if (g != 0) goto 7 else goto 6
/// 6: return c32
/// 7: return g
/// ```
fn guarded(shape: Guarded) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(2, 8);
    let c = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::U8,
    );
    let cell = ssa.create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let c32 = var(&mut ssa, 0, 0, 0);
    let zero_8 = ssa.create_variable(
        VariableOrigin::Local(1),
        1,
        DefSite::instruction(0, 1),
        MockType::U8,
    );
    let zero = var(&mut ssa, 2, 1, 0);
    let f = phi_var(&mut ssa, 3, 2);
    let x = phi_var(&mut ssa, 4, 2);
    let minus_one = var(&mut ssa, 5, 2, 0);
    let d = var(&mut ssa, 6, 2, 1);
    let three = var(&mut ssa, 7, 2, 2);
    let one = var(&mut ssa, 8, 3, 0);
    let fl = phi_var(&mut ssa, 9, 4);
    let ch = phi_var(&mut ssa, 10, 4);
    let zero_4 = var(&mut ssa, 11, 4, 0);
    let g = phi_var(&mut ssa, 12, 5);
    let reads = shape == Guarded::ReadsTheLoop;
    let stores = shape == Guarded::Effect;
    let zero_5 = var(&mut ssa, 13, 5, usize::from(reads || stores));
    let read = reads.then(|| var(&mut ssa, 14, 5, 0));
    let constant =
        |dest: SsaVarId, value: ConstValue<MockTarget>| instr(SsaOp::Const { dest, value });
    let compare = |left: SsaVarId, right: SsaVarId, cmp: CmpKind, to: (usize, usize)| {
        instr(SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned: false,
            true_target: to.0,
            false_target: to.1,
        })
    };
    let asked = if shape == Guarded::OtherConstant {
        ConstValue::U8(1)
    } else {
        ConstValue::U8(0)
    };
    let mut past = Vec::new();
    if let Some(read) = read {
        past.push(add(read, x, x));
    }
    if stores {
        past.push(instr(SsaOp::StoreIndirect {
            addr: cell,
            value: g,
            value_type: MockType::I32,
            access: AccessQualifiers::FLAT,
        }));
    }
    past.push(constant(zero_5, ConstValue::I32(0)));
    past.push(compare(g, zero_5, CmpKind::Ne, (7, 6)));
    let rows: [Vec<SsaInstruction<MockTarget>>; 8] = [
        vec![
            instr(SsaOp::IntConv {
                dest: c32,
                operand: c,
                target: MockType::I32,
                overflow_check: false,
                unsigned: true,
            }),
            constant(zero_8, asked),
            compare(c, zero_8, CmpKind::Eq, (6, 1)),
        ],
        vec![
            constant(zero, ConstValue::I32(0)),
            instr(SsaOp::Jump { target: 2 }),
        ],
        vec![
            constant(minus_one, ConstValue::I32(-1)),
            add(d, x, minus_one),
            constant(three, ConstValue::I32(3)),
            compare(x, three, CmpKind::Eq, (3, 4)),
        ],
        vec![
            constant(one, ConstValue::I32(1)),
            instr(SsaOp::Jump { target: 4 }),
        ],
        vec![
            constant(zero_4, ConstValue::I32(0)),
            compare(ch, zero_4, CmpKind::Ne, (2, 5)),
        ],
        past,
        vec![instr(SsaOp::returns(Some(c32)))],
        vec![instr(SsaOp::returns(Some(g)))],
    ];
    let mut blocks: Vec<SsaBlock<MockTarget>> = (0..8).map(SsaBlock::new).collect();
    for (block, rows) in blocks.iter_mut().zip(rows) {
        for row in rows {
            block.add_instruction(row);
        }
    }
    let phi = |result: SsaVarId, local: u16, operands: &[(SsaVarId, usize)]| {
        let mut phi = PhiNode::new(result, VariableOrigin::Local(local));
        for &(value, from) in operands {
            phi.add_operand(PhiOperand::new(value, from));
        }
        phi
    };
    let placed = [
        (2, phi(f, 3, &[(zero, 1), (fl, 4)])),
        (2, phi(x, 4, &[(c32, 1), (ch, 4)])),
        (4, phi(fl, 9, &[(f, 2), (one, 3)])),
        (4, phi(ch, 10, &[(d, 2), (d, 3)])),
        (5, phi(g, 12, &[(fl, 4)])),
    ];
    for (at, phi) in placed {
        if let Some(block) = blocks.get_mut(at) {
            block.add_phi(phi);
        }
    }
    for block in blocks {
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    ssa
}

/// What [`guarded`] returns for the byte `c`, run by the evaluator.
fn guarded_returns(ssa: &SsaFunction<MockTarget>, c: u8) -> Option<i64> {
    let mut eval = SsaEvaluator::new(ssa, MACHINE);
    let input = ssa
        .variables()
        .iter()
        .find(|var| matches!(var.origin(), VariableOrigin::Argument(0)))?
        .id();
    eval.set_concrete(input, ConstValue::U8(c));
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

/// **A test copied ahead of a loop a compiler rotated is the loop's test.**
/// `0` asks `c == 0` of the byte; the loop goes round on `ch != 0` of the
/// word it widened, which `x` starts as. `0` enters the loop through `1` at
/// its test `4`, whose phis take what the loop starts with (`0` and the
/// widened byte), and the loop is entered at its test alone; the way out
/// with nothing done runs through `5`, which `metaflag`'s `0` decides, so
/// `6` is entered from `5` alone. Every byte returns what it returned.
#[test]
fn a_test_copied_ahead_of_a_rotated_loop_is_its_test() {
    let original = guarded(Guarded::Copy);
    let mut ssa = original.clone();
    let outcome = merge(&mut ssa);
    assert!(matches!(outcome, Ok(true)), "{outcome:?}");
    let jumps = |block: usize| {
        ssa.block(block)
            .and_then(SsaBlock::control_terminator)
            .and_then(|op| match op {
                SsaOp::Jump { target } => Some(*target),
                _ => None,
            })
    };
    assert_eq!(jumps(0), Some(1), "{ssa:?}");
    assert_eq!(jumps(1), Some(4), "{ssa:?}");
    // What the loop starts with enters its test from `1`: `metaflag`'s `0`,
    // defined in `1`, and the byte widened in `0`.
    let entering: Vec<Option<SsaOp<MockTarget>>> = ssa
        .block(4)
        .map(|block| {
            block
                .phi_nodes()
                .iter()
                .map(|phi| {
                    phi.operand_from(1)
                        .and_then(|operand| ssa.get_definition(operand.value()))
                        .cloned()
                })
                .collect()
        })
        .unwrap_or_default();
    assert!(
        matches!(
            entering.as_slice(),
            [
                Some(SsaOp::Const {
                    value: ConstValue::I32(0),
                    ..
                }),
                Some(SsaOp::IntConv { unsigned: true, .. })
            ]
        ),
        "{entering:?}"
    );
    assert!(
        ssa.block(2).is_some_and(|block| block
            .phi_nodes()
            .iter()
            .all(|phi| phi.operand_from(1).is_none())),
        "{ssa:?}"
    );
    assert!(!branches_to(&ssa, 0, 6), "{ssa:?}");
    assert_eq!(ssa.validate(), Ok(()));
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
    for c in [0, 1, 2, 3, 4, 5, 200] {
        let before = guarded_returns(&original, c);
        assert!(before.is_some(), "the fixture returns for {c}");
        assert_eq!(guarded_returns(&ssa, c), before, "{c}: {ssa:?}");
    }
}

/// **A guard asking something else, or whose way past the loop would run
/// or read what it never did, stays.** The controls of
/// [`a_test_copied_ahead_of_a_rotated_loop_is_its_test`]: `0` asks `c == 1`;
/// `5` stores the flag first; `5` reads the loop's `x`, which the guard's
/// way past the loop never computed.
#[test]
fn a_guard_that_is_no_copy_stays() {
    for shape in [
        Guarded::OtherConstant,
        Guarded::Effect,
        Guarded::ReadsTheLoop,
    ] {
        let mut ssa = guarded(shape);
        let outcome = merge(&mut ssa);
        assert!(outcome.is_ok(), "{shape:?}: {outcome:?}");
        assert!(branches_to(&ssa, 0, 6), "{shape:?}: {ssa:?}");
    }
}

/// What a loop entered at two blocks tests on its ways in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Entries {
    /// Each way in asks `x == K` of the character it read.
    Alike,
    /// `4` asks `x2 == 10`: another question.
    OtherConstant,
    /// Nothing goes round: `1` and `2` run on to `5`, and `0` sends its way
    /// on to `3` or `4` on `x0 < 10`, two tests the program writes.
    NoCycle,
    /// The loop has a header of its own, `6`, which `1` and `2` go back to
    /// and which picks `3` or `4` on `x < 10`: `1` and `2` are two arms of an
    /// `if` in the loop, which reach each other only through `6`.
    Arms,
}

/// A loop a compiler wrote as two versions, one entered past a backslash,
/// each ending in its own copy of the test that picks the next version.
///
/// ```text
/// 0: x0 = *p; if (x0 == K) goto 1 else goto 2
/// 1: a = phi(0: p, 3: q1, 4: q2); q1 = a + 2; if (q1 < 10) goto 3 else goto 5
/// 2: b = phi(0: p, 3: q1, 4: q2); q2 = b + 1; if (q2 < 10) goto 4 else goto 5
/// 3: x1 = *q1; if (x1 == K) goto 1 else goto 2
/// 4: x2 = *q2; if (x2 == K) goto 1 else goto 2
/// 5: r = phi(1: q1, 2: q2); return r
/// ```
///
/// The loop is entered at `1` and at `2`, and neither dominates the other.
/// `entries` says what the ways in ask ([`Entries`]). Where `3` and `4` run
/// ahead of both versions, with [`Entries::NoCycle`] and [`Entries::Arms`],
/// they read `h`: `p`, or the header `6`'s `h = phi(0: p, 1: q1, 2: q2)`.
fn two_entries(entries: Entries) -> SsaFunction<MockTarget> {
    let cycle = matches!(entries, Entries::Alike | Entries::OtherConstant);
    let arms = entries == Entries::Arms;
    let mut ssa = SsaFunction::new(1, 8);
    let p = var(&mut ssa, 0, 0, 0);
    let key = var(&mut ssa, 1, 0, 1);
    let one = var(&mut ssa, 2, 0, 2);
    let two = var(&mut ssa, 3, 0, 3);
    let ten = var(&mut ssa, 4, 0, 4);
    let x0 = var(&mut ssa, 5, 0, 5);
    let c0 = var(&mut ssa, 6, 0, 6);
    let a = phi_var(&mut ssa, 7, 1);
    let q1 = var(&mut ssa, 8, 1, 0);
    let d1 = var(&mut ssa, 9, 1, 1);
    let b = phi_var(&mut ssa, 10, 2);
    let q2 = var(&mut ssa, 11, 2, 0);
    let d2 = var(&mut ssa, 12, 2, 1);
    let x1 = var(&mut ssa, 13, 3, 0);
    let c1 = var(&mut ssa, 14, 3, 1);
    let x2 = var(&mut ssa, 15, 4, 0);
    let c2 = var(&mut ssa, 16, 4, 1);
    let r = phi_var(&mut ssa, 17, 5);
    // The header's phi, read and test, where the loop has one.
    let header = arms.then(|| {
        (
            phi_var(&mut ssa, 18, 6),
            var(&mut ssa, 19, 6, 0),
            var(&mut ssa, 20, 6, 1),
        )
    });
    // What `3` and `4` read, ahead of both versions.
    let ahead = header.map_or(p, |(h, _, _)| h);

    let equals = |dest, left, right| {
        instr(SsaOp::Ceq {
            dest,
            left,
            right,
            signaling: false,
        })
    };
    let branch = |condition, true_target, false_target| {
        instr(SsaOp::Branch {
            condition,
            true_target,
            false_target,
        })
    };

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::LoadArg {
        dest: p,
        arg_index: 0,
    }));
    for (dest, value) in [(key, 0x5c), (one, 1), (two, 2), (ten, 10)] {
        b0.add_instruction(instr(SsaOp::Const {
            dest,
            value: ConstValue::I32(value),
        }));
    }
    b0.add_instruction(load(x0, p));
    match entries {
        Entries::Alike | Entries::OtherConstant => {
            b0.add_instruction(equals(c0, x0, key));
            b0.add_instruction(branch(c0, 1, 2));
        }
        Entries::NoCycle => {
            b0.add_instruction(less(c0, x0, ten));
            b0.add_instruction(branch(c0, 3, 4));
        }
        Entries::Arms => {
            b0.add_instruction(less(c0, x0, ten));
            b0.add_instruction(instr(SsaOp::Jump { target: 6 }));
        }
    }

    let mut versions = Vec::new();
    for (index, phi, step, next, test, onward) in [(1, a, two, q1, d1, 3), (2, b, one, q2, d2, 4)] {
        let mut version = SsaBlock::new(index);
        let mut entered = PhiNode::new(phi, VariableOrigin::Local(if index == 1 { 7 } else { 10 }));
        if cycle {
            entered.add_operand(PhiOperand::new(p, 0));
            entered.add_operand(PhiOperand::new(q1, 3));
            entered.add_operand(PhiOperand::new(q2, 4));
        } else {
            entered.add_operand(PhiOperand::new(ahead, 3));
            entered.add_operand(PhiOperand::new(ahead, 4));
        }
        version.add_phi(entered);
        version.add_instruction(add(next, phi, step));
        version.add_instruction(less(test, next, ten));
        version.add_instruction(match entries {
            Entries::Alike | Entries::OtherConstant => branch(test, onward, 5),
            Entries::NoCycle => instr(SsaOp::Jump { target: 5 }),
            Entries::Arms => branch(test, 6, 5),
        });
        versions.push(version);
    }

    let mut tests = Vec::new();
    for (index, read, from, condition) in [(3, x1, q1, c1), (4, x2, q2, c2)] {
        let mut test = SsaBlock::new(index);
        test.add_instruction(load(read, if cycle { from } else { ahead }));
        let against = if entries == Entries::OtherConstant && index == 4 {
            ten
        } else {
            key
        };
        test.add_instruction(equals(condition, read, against));
        test.add_instruction(branch(condition, 1, 2));
        tests.push(test);
    }

    let mut b5 = SsaBlock::new(5);
    let mut exit = PhiNode::new(r, VariableOrigin::Local(17));
    exit.add_operand(PhiOperand::new(q1, 1));
    exit.add_operand(PhiOperand::new(q2, 2));
    b5.add_phi(exit);
    b5.add_instruction(instr(SsaOp::returns(Some(r))));

    let mut all = vec![b0];
    all.extend(versions);
    all.extend(tests);
    all.push(b5);
    if let Some((h, xh, ch)) = header {
        let mut b6 = SsaBlock::new(6);
        let mut entered = PhiNode::new(h, VariableOrigin::Local(18));
        entered.add_operand(PhiOperand::new(p, 0));
        entered.add_operand(PhiOperand::new(q1, 1));
        entered.add_operand(PhiOperand::new(q2, 2));
        b6.add_phi(entered);
        b6.add_instruction(load(xh, h));
        b6.add_instruction(less(ch, xh, ten));
        b6.add_instruction(branch(ch, 3, 4));
        all.push(b6);
    }
    for (index, mut block) in all.into_iter().enumerate() {
        block
            .provenance_mut()
            .carry(u64::try_from(index).unwrap_or(0).saturating_mul(0x10));
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "the fixture verifies: {ssa:?}"
    );
    ssa
}

/// The blocks whose edges lead to `block`.
fn entered_from(ssa: &SsaFunction<MockTarget>, block: usize) -> Vec<usize> {
    (0..ssa.block_count())
        .filter(|from| {
            ssa.block(*from)
                .is_some_and(|candidate| candidate.successors().contains(&block))
        })
        .collect()
}

/// **A loop entered at two blocks by one test, copied, is entered at that
/// test.** `0`, `3` and `4` each ask `x == K` of a character and pick `1` or
/// `2` by it: a compiler that wrote the loop as two versions copied the test
/// that picks one into every way in. No block of the loop dominates it, and
/// the source writes no such loop. The copies become one test of their own,
/// reading a phi of each character, and the loop is entered at it alone.
#[test]
fn a_loop_entered_at_two_blocks_by_one_copied_test_is_entered_at_it() {
    let mut ssa = two_entries(Entries::Alike);
    assert!(matches!(merge(&mut ssa), Ok(true)), "{ssa:?}");
    let merged = ssa.block_count().saturating_sub(1);
    for version in [1, 2] {
        assert_eq!(entered_from(&ssa, version), [merged], "{ssa:?}");
    }
    assert_eq!(entered_from(&ssa, merged), [0, 3, 4], "{ssa:?}");
    assert_eq!(
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard),
        Vec::new(),
        "{ssa:?}"
    );
}

/// **Ways in asking other questions keep their tests.** The control of
/// [`a_loop_entered_at_two_blocks_by_one_copied_test_is_entered_at_it`]:
/// `4` compares with `10`, not `K`, and the loop stays entered at two
/// blocks.
#[test]
fn a_loop_entered_by_two_questions_keeps_its_tests() {
    let mut ssa = two_entries(Entries::OtherConstant);
    let _ = merge(&mut ssa);
    assert!(entered_from(&ssa, 1).contains(&4), "{ssa:?}");
}

/// **Alike tests picking between two arms of a loop stay.** `3` and `4`
/// each ask `x == K` and pick `1` or `2`, which reach each other only
/// through the loop's header `6`, the block dominating both: two arms of an
/// `if` in the loop, entered by the tests the program writes, and the loop
/// is entered at its header alone.
#[test]
fn alike_tests_picking_between_two_arms_of_a_loop_stay() {
    let mut ssa = two_entries(Entries::Arms);
    let _ = merge(&mut ssa);
    for (arm, tests) in [(1, [3, 4]), (2, [3, 4])] {
        assert_eq!(entered_from(&ssa, arm), tests, "{ssa:?}");
    }
}

/// **Alike tests outside a loop stay.** `1` and `2` run on to `5`: `3` and
/// `4` pick between two blocks nothing goes round, two tests the program
/// writes.
#[test]
fn alike_tests_picking_between_blocks_no_loop_holds_stay() {
    let mut ssa = two_entries(Entries::NoCycle);
    assert!(matches!(merge(&mut ssa), Ok(false)), "{ssa:?}");
}
