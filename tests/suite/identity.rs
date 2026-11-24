//! Instruction identity: what it survives, and what retires it.
//!
//! A host keys its per-instruction facts - a call's argument places, a decrypted
//! constant, the address an instruction was lifted from - by
//! [`InstrId`](analyssa::ir::instruction::InstrId), because the position an
//! instruction sits at changes on the first edit before it. These tests pin the
//! properties that makes possible: an id names one instruction, it travels with
//! the instruction when a pass relocates it, a copy is a different instruction
//! with its own id, and a retired id is never handed out again.

use std::process;

use analyssa::{
    analysis::{SsaVerifier, VerifierError, VerifyLevel, WorkBudget, context::AnalysisContext},
    events::{EventLog, NullListener},
    ir::{
        block::SsaBlock,
        function::{SsaEditOptions, SsaFunction},
        instruction::{InstrId, SsaInstruction},
        ops::SsaOp,
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::{
        BlockMergingPass, ControlFlowSimplificationPass, DeadCodeEliminationPass,
        JumpThreadingPass, LicmPass,
    },
    scheduling::{FunctionPass, PassEnv},
    testing::{self, MACHINE, MockTarget, MockType},
};
#[cfg(feature = "serde")]
use serde_json::{from_str, to_string};

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| process::abort())
}

fn result_or_abort<T, E>(result: Result<T, E>) -> T {
    result.unwrap_or_else(|_| process::abort())
}

/// Every identity the function currently holds, in block order.
fn identities(ssa: &SsaFunction<MockTarget>) -> Vec<InstrId> {
    ssa.blocks()
        .iter()
        .flat_map(|block| block.instructions().iter().filter_map(SsaInstruction::id))
        .collect()
}

/// A loop whose body holds one invariant addition, which LICM hoists.
fn loop_with_invariant() -> (SsaFunction<MockTarget>, SsaVarId) {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 5);
    let left = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let right = ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(0, 1),
        MockType::I32,
    );
    let cond = ssa.create_variable(
        VariableOrigin::Local(2),
        0,
        DefSite::instruction(0, 2),
        MockType::I32,
    );
    let sum = ssa.create_variable(
        VariableOrigin::Local(3),
        0,
        DefSite::instruction(2, 0),
        MockType::I32,
    );

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(2),
    }));
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(3),
    }));
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
    ssa.add_block(entry);

    let mut header = SsaBlock::new(1);
    header.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
        condition: cond,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(header);

    let mut body = SsaBlock::new(2);
    body.add_instruction(
        SsaInstruction::synthetic(SsaOp::Add {
            dest: sum,
            left,
            right,
            flags: None,
        })
        .with_result_type(MockType::I32),
    );
    body.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
    ssa.add_block(body);

    let mut exit = SsaBlock::new(3);
    exit.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(sum))));
    ssa.add_block(exit);

    ssa.recompute_uses();
    (ssa, sum)
}

/// Where the instruction defining `var` sits, and what identity it carries.
fn definition_of(ssa: &SsaFunction<MockTarget>, var: SsaVarId) -> Option<(usize, Option<InstrId>)> {
    ssa.blocks()
        .iter()
        .enumerate()
        .find_map(|(block_idx, block)| {
            block
                .instructions()
                .iter()
                .find(|instr| instr.op().defs().any(|dest| dest == var))
                .map(|instr| (block_idx, instr.id()))
        })
}

#[test]
fn a_hoisted_instruction_is_the_same_instruction_in_another_block() {
    let (mut ssa, sum) = loop_with_invariant();
    let (body_block, before) = some_or_abort(definition_of(&ssa, sum));
    assert_eq!(body_block, 2, "the addition starts in the loop body");
    let before = some_or_abort(before);

    let log: EventLog<MockTarget> = EventLog::new();
    let method = 0u32;
    assert!(
        LicmPass.run(
            &mut ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true),
        "the invariant addition is hoistable"
    );
    ssa.repair_ssa();

    let (hoisted_block, after) = some_or_abort(definition_of(&ssa, sum));
    assert_ne!(hoisted_block, 2, "the addition left the loop body");
    assert_eq!(
        after,
        Some(before),
        "a hoist moves an instruction; it does not build a new one"
    );
    assert_eq!(
        ssa.locate_instruction(before).map(|(block, _)| block),
        Some(hoisted_block),
        "the identity resolves to where the instruction now is"
    );
}

/// **A moved instruction is the same instruction elsewhere.** `B0` defines
/// `x` and jumps to `B1`, which returns it; moving the definition to the top of
/// `B1` keeps its identity, which now resolves there, and leaves the function
/// valid.
#[test]
fn a_move_keeps_the_instructions_identity() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let x = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: x,
        value: ConstValue::I32(5),
    }));
    b0.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(x))));
    ssa.add_block(b1);
    ssa.recompute_uses();

    let moved = {
        let mut moved = None;
        result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
            moved = Some(editor.move_instruction((0, 0), (1, 0))?);
            Ok(())
        }));
        some_or_abort(moved)
    };
    assert_eq!(
        ssa.locate_instruction(moved),
        Some((1, 0)),
        "the identity resolves to where the instruction now is"
    );
    assert_eq!(
        identities(&ssa).iter().filter(|id| **id == moved).count(),
        1,
        "a move leaves one instruction of that identity"
    );
    assert!(
        ssa.block(0)
            .is_some_and(|block| block.instructions().len() == 1),
        "the old block keeps only its jump"
    );
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}");
}

#[test]
fn a_copy_is_a_new_instruction_that_names_its_source() {
    let mut ssa = testing::const_i32_return(11);
    let source_id = some_or_abort(some_or_abort(
        ssa.block(0)
            .and_then(|block| block.instructions().first())
            .map(SsaInstruction::id),
    ));

    let copy_id = {
        let mut copied = None;
        result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
            copied = Some(editor.duplicate_instruction((0, 0), (0, 1))?);
            Ok(())
        }));
        some_or_abort(copied)
    };

    assert_ne!(copy_id, source_id, "a copy is a different instruction");
    assert_eq!(
        ssa.instruction_provenance(copy_id),
        Some(source_id),
        "the copy names the instruction it was made from"
    );
    assert_eq!(
        ssa.instruction_provenance(source_id),
        None,
        "the source is nobody's copy"
    );
}

#[test]
fn an_identity_is_never_reissued_after_its_instruction_goes() {
    let (mut ssa, sum) = loop_with_invariant();
    let retired = some_or_abort(some_or_abort(definition_of(&ssa, sum)).1);

    result_or_abort(ssa.edit(SsaEditOptions::new(), |editor| {
        editor.nop_instruction(2, 0)?;
        Ok(())
    }));
    assert_eq!(
        ssa.locate_instruction(retired),
        None,
        "the instruction is gone, so its identity names nothing"
    );

    // Everything minted from here on is new, however many instructions come and
    // go in between.
    for _ in 0..4 {
        assert_ne!(ssa.mint_instr_id(), retired);
    }
}

#[test]
fn assigning_identities_is_idempotent() {
    let mut ssa = testing::diamond_phi_fixture();
    assert_eq!(
        ssa.assign_instr_ids(),
        0,
        "a finished function has nothing left to stamp"
    );
    let before = identities(&ssa);

    // An instruction added through the raw surface is unidentified until a
    // boundary stamps it; stamping twice changes nothing further.
    some_or_abort(ssa.block_mut(0))
        .instructions_mut()
        .insert(0, SsaInstruction::synthetic(SsaOp::Nop));
    assert_eq!(ssa.assign_instr_ids(), 1);
    assert_eq!(ssa.assign_instr_ids(), 0);

    let after = identities(&ssa);
    assert_eq!(
        after.len(),
        before.len().saturating_add(1),
        "exactly the new instruction gained an identity"
    );
    assert!(
        before.iter().all(|id| after.contains(id)),
        "stamping leaves existing identities alone"
    );
}

#[test]
fn identities_stay_unique_across_the_passes_that_move_instructions() {
    for (label, mut ssa) in [
        ("diamond", testing::diamond_phi_fixture()),
        ("loop", testing::loop_counter_fixture()),
        ("try/catch", testing::try_catch_fixture()),
        ("scalar", testing::scalar_rewrite_fixture()),
    ] {
        let method = 0u32;
        let _ = BlockMergingPass.run(
            &mut ssa,
            &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
        );
        let _ = ControlFlowSimplificationPass.run(
            &mut ssa,
            &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
        );
        let _ = LicmPass.run(
            &mut ssa,
            &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
        );
        let _ = JumpThreadingPass.run(
            &mut ssa,
            &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE))
                .with_budget(WorkBudget::DEFAULT),
        );
        let _ = DeadCodeEliminationPass.run(
            &mut ssa,
            &PassEnv::new(&method, &NullListener, AnalysisContext::new(MACHINE)),
        );

        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
        assert!(
            !errors
                .iter()
                .any(|error| matches!(error, VerifierError::DuplicateInstructionId { .. })),
            "{label}: {errors:?}"
        );
        let mut ids = identities(&ssa);
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "{label}: identities are unique");
    }
}

#[test]
fn a_function_built_by_hand_reports_what_it_has_not_finished() {
    let mut ssa = testing::const_i32_return(1);
    some_or_abort(ssa.block_mut(0))
        .instructions_mut()
        .insert(0, SsaInstruction::synthetic(SsaOp::Nop));
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        errors
            .iter()
            .any(|error| matches!(error, VerifierError::UnassignedInstructionId { .. })),
        "{errors:?}"
    );

    ssa.assign_instr_ids();
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        !errors
            .iter()
            .any(|error| matches!(error, VerifierError::UnassignedInstructionId { .. })),
        "{errors:?}"
    );
}

#[test]
fn adopting_a_foreign_body_restamps_it() {
    let donor = testing::diamond_phi_fixture();
    let mut host = testing::const_i32_return(3);
    let host_ids = identities(&host);

    let mut imported: Vec<SsaInstruction<MockTarget>> =
        some_or_abort(donor.block(0)).instructions().to_vec();
    host.adopt_instructions(&mut imported);

    for instr in &imported {
        let id = some_or_abort(instr.id());
        assert!(
            !host_ids.contains(&id),
            "an adopted instruction does not reuse an identity the host already issued"
        );
    }
}

#[cfg(feature = "serde")]
#[test]
fn identities_survive_a_round_trip() {
    let ssa = testing::loop_counter_fixture();
    let before = identities(&ssa);
    let encoded = result_or_abort(to_string(&ssa));
    let decoded: SsaFunction<MockTarget> = result_or_abort(from_str(&encoded));
    assert_eq!(identities(&decoded), before);

    // The allocator round-trips too, so a decoded function does not reissue an
    // identity it already handed out.
    let mut decoded = decoded;
    let next = decoded.mint_instr_id();
    assert!(!before.contains(&next));
}
