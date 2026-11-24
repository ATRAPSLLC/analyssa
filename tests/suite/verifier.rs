//! Focused SSA verifier failure cases.

use std::process;

use analyssa::{
    Result,
    analysis::{SsaVerifier, VerifierError, VerifyLevel},
    ir::{
        block::SsaBlock,
        exception::{BlockRange, ClausePart, ExceptionTableError, SsaExceptionHandler},
        float::{FloatBits, FloatExceptionLayout, FloatExceptions, FloatRounding},
        function::{SsaConstructOptions, SsaFunction},
        instruction::SsaInstruction,
        lanes::VectorConst,
        ops::{
            AccessQualifiers, AtomicAccessWidth, AtomicOrdering, AtomicRmwOp, FlagsMask,
            FpuControlKind, KindedVecData, RuntimeValueKind, SsaOp, VectorBinaryKind,
            VectorBitmaskKind, VectorCastKind, VectorCompareKind, VectorElement, VectorFaultMode,
            VectorMaskMode, VectorReduceKind, VectorSegmentLayout, VectorShuffleSelector,
        },
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, UseSite, VariableOrigin},
    },
    target::{Target, VectorDescriptor, VectorLaneKind, VectorShuffleLane, VectorShuffleMask},
    testing::{MockTarget, MockType, diamond_chain_fixture, diamond_phi_fixture},
};

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| process::abort())
}

fn result_or_abort<T>(result: Result<T>) -> T {
    result.unwrap_or_else(|_| process::abort())
}

fn err_or_abort<T>(result: Result<T, Vec<VerifierError>>) -> Vec<VerifierError> {
    match result {
        Ok(_) => process::abort(),
        Err(errors) => errors,
    }
}

fn local(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(block, instr),
        MockType::I32,
    )
}

fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
    SsaInstruction::synthetic(op)
}

fn typed_local(
    ssa: &mut SsaFunction<MockTarget>,
    idx: u16,
    block: usize,
    instr: usize,
    ty: MockType,
) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(block, instr),
        ty,
    )
}

#[test]
fn verifier_rejects_duplicate_instruction_definitions() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(v0))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(
        errors
            .iter()
            .any(|err| matches!(err, VerifierError::DuplicateDefinition { var, .. } if *var == v0))
    );
}

#[test]
fn verifier_rejects_missing_phi_operand_for_predecessor() {
    let mut ssa = SsaFunction::new(0, 3);
    let from_left = local(&mut ssa, 0, 1, 0);
    let from_right = local(&mut ssa, 1, 2, 0);
    let merged = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(3), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Branch {
        condition: from_left,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    let mut left = SsaBlock::new(1);
    left.add_instruction(instr(SsaOp::Const {
        dest: from_left,
        value: ConstValue::I32(1),
    }));
    left.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(left);

    let mut right = SsaBlock::new(2);
    right.add_instruction(instr(SsaOp::Const {
        dest: from_right,
        value: ConstValue::I32(2),
    }));
    right.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(right);

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(from_left, 1));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::returns(Some(merged))));
    ssa.add_block(join);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::MissingPhiOperand {
            block: 3,
            phi_idx: 0,
            missing_pred: 2,
        }
    )));
}

#[test]
fn verifier_rejects_terminator_before_block_end() {
    let mut ssa = SsaFunction::new(0, 2);
    let v0 = local(&mut ssa, 0, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Jump { target: 1 }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    ssa.add_block(block);

    let mut exit = SsaBlock::new(1);
    exit.add_instruction(instr(SsaOp::returns(Some(v0))));
    ssa.add_block(exit);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::TerminatorNotLast {
            block: 0,
            instr_idx: 0,
            instr_count: 2,
        }
    )));
}

#[test]
fn verifier_rejects_self_referential_instruction() {
    let mut ssa = SsaFunction::new(0, 2);
    let v0 = local(&mut ssa, 0, 0, 0);
    let v1 = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: v1,
        left: v0,
        right: v1,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(v1))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::SelfReferentialInstruction { var, .. } if *var == v1
    )));
}

#[test]
fn verifier_rejects_extra_phi_operand_for_non_predecessor() {
    let mut ssa = SsaFunction::new(0, 3);
    let left_value = local(&mut ssa, 0, 1, 0);
    let stale_value = local(&mut ssa, 1, 2, 0);
    let merged = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(3), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(entry);

    let mut left = SsaBlock::new(1);
    left.add_instruction(instr(SsaOp::Const {
        dest: left_value,
        value: ConstValue::I32(1),
    }));
    left.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(left);

    let mut disconnected = SsaBlock::new(2);
    disconnected.add_instruction(instr(SsaOp::Const {
        dest: stale_value,
        value: ConstValue::I32(2),
    }));
    disconnected.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(disconnected);

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(left_value, 1));
    phi.add_operand(PhiOperand::new(stale_value, 2));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::returns(Some(merged))));
    ssa.add_block(join);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::ExtraPhiOperand {
            block: 3,
            phi_idx: 0,
            extra_pred: 2,
        }
    )));
}

#[test]
fn verifier_rejects_dominance_violation_across_branches() {
    let mut ssa = SsaFunction::new(0, 2);
    let condition = local(&mut ssa, 0, 0, 0);
    let branch_only = local(&mut ssa, 1, 1, 0);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: condition,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Branch {
        condition,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    let mut left = SsaBlock::new(1);
    left.add_instruction(instr(SsaOp::Const {
        dest: branch_only,
        value: ConstValue::I32(42),
    }));
    left.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(left);

    let mut right = SsaBlock::new(2);
    right.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(right);

    let mut join = SsaBlock::new(3);
    join.add_instruction(instr(SsaOp::returns(Some(branch_only))));
    ssa.add_block(join);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::DominanceViolation {
            var,
            def_block: 1,
            use_block: 3,
        } if *var == branch_only
    )));
}

#[test]
fn verifier_rejects_undefined_use_in_instruction() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = SsaVarId::from_index(0);
    let v1 = SsaVarId::from_index(1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: v1,
        left: v0,
        right: SsaVarId::from_index(99),
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(v1))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|err| matches!(err, VerifierError::UndefinedUse { .. }))
    );
}

#[test]
fn verifier_rejects_orphan_variable() {
    let mut ssa = SsaFunction::new(0, 1);
    let orphan = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    // Don't use `orphan` - it stays in the variables vec but is never defined in a block
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::OrphanVariable { var } if *var == orphan
    )));
}

#[test]
fn validate_returns_structured_verifier_errors() {
    let mut ssa = SsaFunction::new(0, 1);
    let orphan = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = err_or_abort(ssa.validate());
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::OrphanVariable { var } if *var == orphan
    )));
}

/// `SsaFunction::variable(id)` indexes the variables table directly, so a row
/// removed without renumbering makes every higher id resolve to a different
/// variable's def site, type, and uses. Nothing else detects this: the IR stays
/// structurally well-formed and merely describes the wrong variables.
#[test]
fn verifier_rejects_a_non_dense_variable_table() {
    let mut ssa = SsaFunction::new(0, 2);
    let a = local(&mut ssa, 0, 0, 0);
    let b = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(b))));
    ssa.add_block(block);
    ssa.recompute_uses();

    // Dense to begin with.
    assert!(
        !SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Quick)
            .iter()
            .any(|e| matches!(e, VerifierError::VariableTableNotDense { .. })),
        "the fixture must start dense"
    );

    // Punch a hole the way an unpaired `variables.retain(..)` would.
    ssa.variables_mut().remove(0);

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(
        errors.iter().any(|err| matches!(
            err,
            VerifierError::VariableTableNotDense { slot: 0, found } if *found == b
        )),
        "a hole in the variables table must be rejected; got {errors:?}"
    );
}

/// A variable whose defining instruction was removed while a use survived is a
/// dangling read, and the verifier must reject it.
///
/// This is the shape a pass produces when it deletes a multi-definition
/// instruction on the strength of its primary result alone: the secondary
/// (`flags`) definition disappears but its consumer does not. Repair then resets
/// the now-undefined variable's def site to `DefSite::entry()`, making it look
/// like an implicitly-defined argument, which a check keyed on the def site
/// alone would accept.
#[test]
fn verifier_rejects_a_use_whose_definition_was_removed() {
    let mut ssa = SsaFunction::new(0, 1);
    let defined = local(&mut ssa, 0, 0, 0);
    // A temporary beyond the declared local count, with an entry-shaped def site
    // and no defining instruction anywhere - exactly what repair leaves behind.
    let dangling =
        ssa.create_variable(VariableOrigin::Local(7), 0, DefSite::entry(), MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: defined,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: SsaVarId::from_index(2),
        left: defined,
        right: dangling,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors.iter().any(|err| matches!(
            err,
            VerifierError::UndefinedUse { var, block: 0, instr_idx: 1 } if *var == dangling
        )),
        "a read of a variable nothing defines must be rejected, and must name the \
         reading instruction rather than being filtered as a cosmetic orphan; got {errors:?}"
    );
}

/// A `VariableOrigin::EntryLiveIn` variable is caller-supplied, so reading it
/// without any defining instruction is legal.
///
/// This is the contract machine-code front ends depend on: a register read
/// before the function body ever writes it has no definition to point at. It is
/// deliberately a *distinct* origin from `Argument` - the lifter has not
/// recovered the calling convention, so it cannot honestly claim a signature
/// position - and distinct from `Phi`, which stays rejected so that a variable
/// whose definition was destroyed cannot masquerade as an entry value.
///
/// This test exists because the entry-defined origin set is a public contract,
/// not a verifier implementation detail: tightening it would break a
/// downstream lifter with no failure in this crate's own suite.
#[test]
fn verifier_accepts_a_read_of_an_entry_live_in() {
    let mut ssa = SsaFunction::new(0, 1);
    let defined = local(&mut ssa, 0, 0, 0);
    let live_in = ssa.create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::I32,
    );

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: defined,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: SsaVarId::from_index(2),
        left: defined,
        right: live_in,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        !errors.iter().any(|err| matches!(
            err,
            VerifierError::UndefinedUse { var, .. } | VerifierError::OrphanVariable { var }
                if *var == live_in
        )),
        "an entry live-in needs no definition in the body; got {errors:?}"
    );
}

/// `EntryLiveIn` must not widen the hole that rejecting `Phi` closed: a
/// phi-origin variable with an entry-shaped def site and no definition is still
/// a destroyed definition, and still rejected.
#[test]
fn verifier_still_rejects_a_phi_origin_variable_with_no_definition() {
    let mut ssa = SsaFunction::new(0, 1);
    let defined = local(&mut ssa, 0, 0, 0);
    let destroyed = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::entry(), MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: defined,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: SsaVarId::from_index(2),
        left: defined,
        right: destroyed,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors.iter().any(|err| matches!(
            err,
            VerifierError::UndefinedUse { var, .. } if *var == destroyed
        )),
        "a phi-origin variable with no definition is a destroyed definition and \
         must still be rejected; got {errors:?}"
    );
}

/// The verifier is the safety net for malformed IR, so it must *report* a
/// dangling phi predecessor rather than abort on it. `BitSet::contains` asserts
/// its index is in range, and `check_dominance` fed it raw block indices taken
/// straight from the IR - so `VerifyLevel::Full` aborted the process on one of
/// the shapes it exists to catch.
#[test]
fn verifier_tolerates_a_phi_operand_naming_a_nonexistent_block() {
    let mut ssa = SsaFunction::new(0, 2);
    let value = local(&mut ssa, 0, 0, 0);
    let merged = ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(1), MockType::I32);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(1));
    phi.add_operand(PhiOperand::new(value, 0));
    // Block 99 does not exist. The IR permits dangling successors, and the
    // verifier documents that it tolerates them.
    phi.add_operand(PhiOperand::new(value, 99));
    b1.add_phi(phi);
    b1.add_instruction(instr(SsaOp::returns(Some(merged))));
    ssa.add_block(b1);
    ssa.recompute_uses();

    // Every level must complete. Each returns findings; none may abort.
    for level in [VerifyLevel::Quick, VerifyLevel::Standard, VerifyLevel::Full] {
        let errors = SsaVerifier::new(&ssa).verify(level);
        // At Standard and above the bogus edge is reported rather than ignored.
        if level >= VerifyLevel::Standard {
            assert!(
                errors.iter().any(|err| matches!(
                    err,
                    VerifierError::ExtraPhiOperand { extra_pred: 99, .. }
                )),
                "the dangling predecessor must be reported at {level:?}; got {errors:?}"
            );
        }
    }
}

/// A variable whose recorded definition site names a block that does not exist
/// reaches the same asserting bounds check from the other direction.
#[test]
fn verifier_tolerates_a_definition_site_naming_a_nonexistent_block() {
    let mut ssa = SsaFunction::new(0, 1);
    // Claims to be defined in block 42, which does not exist.
    let ghost = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(42, 0),
        MockType::I32,
    );

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::returns(Some(ghost))));
    ssa.add_block(b0);
    ssa.recompute_uses();

    // Must not abort.
    let _ = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
}

/// One CFG edge carries one value, so two operands naming the same predecessor
/// have no defined meaning - and consumers disagree about them: `operand_from`
/// returns the first and drops the rest, while SCCP meets them all and yields
/// Bottom. Comparing operand predecessors as a *set* cannot see the duplicate,
/// which is how the advertised "exactly one operand per predecessor" invariant
/// went unchecked.
#[test]
fn verifier_rejects_two_phi_operands_for_one_predecessor() {
    let mut ssa = SsaFunction::new(0, 3);
    let left = local(&mut ssa, 0, 1, 0);
    let right = local(&mut ssa, 1, 2, 0);
    let merged = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(3), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Branch {
        condition: left,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    for (id, var) in [(1usize, left), (2usize, right)] {
        let mut block = SsaBlock::new(id);
        block.add_instruction(instr(SsaOp::Const {
            dest: var,
            value: ConstValue::I32(id as i32),
        }));
        block.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(block);
    }

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(left, 1));
    // A second, conflicting operand for the *same* edge - what an edge redirect
    // produces when it retargets onto a predecessor that already has one.
    phi.add_operand(PhiOperand::new(right, 1));
    phi.add_operand(PhiOperand::new(right, 2));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::returns(Some(merged))));
    ssa.add_block(join);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors.iter().any(|err| matches!(
            err,
            VerifierError::DuplicatePhiOperand {
                block: 3,
                phi_idx: 0,
                pred: 1
            }
        )),
        "two operands on edge B1->B3 must be rejected; got {errors:?}"
    );
}

/// A well-formed phi must not trip the duplicate check.
#[test]
fn verifier_accepts_one_operand_per_predecessor() {
    let mut ssa = SsaFunction::new(0, 3);
    let left = local(&mut ssa, 0, 1, 0);
    let right = local(&mut ssa, 1, 2, 0);
    let merged = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(3), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Branch {
        condition: left,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    for (id, var) in [(1usize, left), (2usize, right)] {
        let mut block = SsaBlock::new(id);
        block.add_instruction(instr(SsaOp::Const {
            dest: var,
            value: ConstValue::I32(id as i32),
        }));
        block.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(block);
    }

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(left, 1));
    phi.add_operand(PhiOperand::new(right, 2));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::returns(Some(merged))));
    ssa.add_block(join);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        !errors
            .iter()
            .any(|err| matches!(err, VerifierError::DuplicatePhiOperand { .. })),
        "a well-formed phi must not be reported; got {errors:?}"
    );
}

/// `strip_nops` resets a variable to `DefSite::entry()` when its defining
/// instruction became a `Nop` - which is exactly what a pass leaves behind when
/// it deletes a definition whose value is still read. That reset must not fire
/// while something reads the variable, or the dangling read is disguised as an
/// argument and `validate` returns `Ok`.
#[test]
fn stripping_nops_does_not_disguise_a_dangling_read_as_an_argument() {
    let mut ssa = SsaFunction::new(0, 1);
    let defined = local(&mut ssa, 0, 0, 0);
    let killed = local(&mut ssa, 1, 0, 1);
    let sum = local(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: defined,
        value: ConstValue::I32(1),
    }));
    // The instruction that defined `killed`, deleted by a pass.
    block.add_instruction(instr(SsaOp::Nop));
    // ...but its value is still read here.
    block.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: defined,
        right: killed,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(sum))));
    ssa.add_block(block);
    ssa.recompute_uses();

    // `repair_ssa` runs `strip_nops` and then `refresh_def_sites`; neither may
    // rewrite `killed` to look entry-defined.
    ssa.repair_ssa();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|err| matches!(err, VerifierError::UndefinedUse { .. })),
        "the read of a variable whose definition was nopped must survive repair \
         as a reported error; got {errors:?}"
    );
}

/// The counterpart: a real argument has no defining instruction either, and must
/// not be reported. Without this, the check above would just be noise.
#[test]
fn verifier_accepts_an_argument_with_no_defining_instruction() {
    let mut ssa = SsaFunction::new(1, 0);
    let arg = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::returns(Some(arg))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        !errors
            .iter()
            .any(|err| matches!(err, VerifierError::OrphanVariable { .. })),
        "an argument is defined at entry; got {errors:?}"
    );
}

#[test]
fn verifier_rejects_unregistered_variable() {
    let mut ssa = SsaFunction::new(0, 0);
    let ghost = SsaVarId::from_index(42);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: ghost,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(ghost))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.iter().any(|err| matches!(
        err,
        VerifierError::UnregisteredVariable { var } if *var == ghost
    )));
}

#[test]
fn verifier_rejects_phi_in_entry_block() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);
    let phi_var = ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(0), MockType::I32);

    let mut entry = SsaBlock::new(0);
    let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(1));
    phi.add_operand(PhiOperand::new(v0, 0));
    entry.add_phi(phi);
    entry.add_instruction(instr(SsaOp::returns(Some(phi_var))));
    ssa.add_block(entry);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|err| matches!(err, VerifierError::PhiWithoutPredecessors { block: 0, .. }))
    );
}

#[test]
fn verifier_rejects_intra_block_cycle() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);
    let v1 = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Add {
        dest: v0,
        left: SsaVarId::from_index(2),
        right: SsaVarId::from_index(2),
        flags: None,
    }));
    let v2_instr = SsaInstruction::synthetic(SsaOp::Const {
        dest: SsaVarId::from_index(2),
        value: ConstValue::I32(1),
    });
    block.add_instruction(v2_instr);
    block.add_instruction(instr(SsaOp::returns(Some(v1))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(
        errors
            .iter()
            .any(|err| matches!(err, VerifierError::IntraBlockCycle { .. }))
    );
}

#[test]
fn verifier_rejects_placeholder_variable_in_instruction() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);
    let sentinel = SsaVarId::from_index(usize::MAX);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: sentinel,
        left: v0,
        right: sentinel,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(v0))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(
        errors
            .iter()
            .any(|err| matches!(err, VerifierError::PlaceholderVariable { .. }))
    );
}

#[test]
fn verifier_all_levels_accept_well_formed_minimal_function() {
    let mut ssa = SsaFunction::new(0, 0);
    let v0 = local(&mut ssa, 0, 0, 0);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(v0))));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Quick).is_empty());
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
}

#[test]
fn verifier_all_levels_accept_diamond_with_phis() {
    let mut ssa = SsaFunction::new(0, 4);
    let cond = local(&mut ssa, 0, 0, 0);
    let left_val = local(&mut ssa, 1, 1, 0);
    let right_val = local(&mut ssa, 2, 2, 0);
    let phi_var = ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(3), MockType::I32);

    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    let mut left = SsaBlock::new(1);
    left.add_instruction(instr(SsaOp::Const {
        dest: left_val,
        value: ConstValue::I32(10),
    }));
    left.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(left);

    let mut right = SsaBlock::new(2);
    right.add_instruction(instr(SsaOp::Const {
        dest: right_val,
        value: ConstValue::I32(20),
    }));
    right.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(right);

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(3));
    phi.add_operand(PhiOperand::new(left_val, 1));
    phi.add_operand(PhiOperand::new(right_val, 2));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::returns(Some(phi_var))));
    ssa.add_block(join);
    ssa.recompute_uses();

    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Quick).is_empty());
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
}

#[test]
fn verifier_reports_multiple_error_kinds_at_once() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local(&mut ssa, 0, 0, 0);
    let v1 = local(&mut ssa, 1, 0, 1);
    let undefined = SsaVarId::from_index(99);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: v1,
        left: v0,
        right: v1,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Copy {
        dest: v1,
        src: undefined,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(v0))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors.len() >= 3,
        "expected at least 3 errors, got {}",
        errors.len()
    );
    let has_duplicate = errors
        .iter()
        .any(|e| matches!(e, VerifierError::DuplicateDefinition { .. }));
    let has_self_ref = errors
        .iter()
        .any(|e| matches!(e, VerifierError::SelfReferentialInstruction { .. }));
    let has_undef = errors
        .iter()
        .any(|e| matches!(e, VerifierError::UndefinedUse { .. }));
    assert!(has_duplicate, "missing DuplicateDefinition");
    assert!(has_self_ref, "missing SelfReferentialInstruction");
    assert!(has_undef, "missing UndefinedUse");
}

#[test]
fn verifier_registers_secondary_flag_definitions() {
    let mut ssa = SsaFunction::new(0, 2);
    let value = typed_local(&mut ssa, 0, 0, 0, MockType::I32);
    let flags = typed_local(&mut ssa, 1, 0, 0, MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Add {
        dest: value,
        left: value,
        right: value,
        flags: Some(flags),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: flags,
        value: ConstValue::I32(1),
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::DuplicateDefinition { var, .. } if *var == flags))
    );
}

#[test]
fn rebuild_remaps_secondary_flag_definitions_and_uses() {
    let mut ssa = SsaFunction::new(0, 5);
    let left = typed_local(&mut ssa, 0, 0, 0, MockType::I32);
    let right = typed_local(&mut ssa, 1, 0, 1, MockType::I32);
    let value = typed_local(&mut ssa, 2, 0, 2, MockType::I32);
    let flags = typed_local(&mut ssa, 3, 0, 2, MockType::I32);
    let read = typed_local(&mut ssa, 4, 0, 3, MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: value,
        left,
        right,
        flags: Some(flags),
    }));
    block.add_instruction(instr(SsaOp::ReadFlags {
        dest: read,
        flags,
        mask: FlagsMask::ZERO,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(read))));
    ssa.add_block(block);
    ssa.recompute_uses();

    result_or_abort(ssa.construct_ssa(&SsaConstructOptions::renaming_only()));

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "verifier errors: {errors:?}");
    assert!(ssa.iter_instructions().any(|(_, _, instr)| {
        matches!(instr.op(), SsaOp::ReadFlags { flags, .. } if ssa.variable(*flags).is_some())
    }));
}

#[test]
fn verifier_accepts_matching_vector_binary_shapes() {
    let mut ssa = SsaFunction::new(0, 3);
    let left = typed_local(&mut ssa, 0, 0, 0, MockType::V4I32);
    let right = typed_local(&mut ssa, 1, 0, 1, MockType::V4I32);
    let dest = typed_local(&mut ssa, 2, 0, 2, MockType::V4I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::Vector(some_or_abort(VectorConst::from_lanes(32, [1, 2, 3, 4]))),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::Vector(some_or_abort(VectorConst::from_lanes(32, [5, 6, 7, 8]))),
    }));
    block.add_instruction(instr(SsaOp::VectorBinary {
        dest,
        left,
        right,
        kind: VectorBinaryKind::Add,
        element: VectorElement::default(),
        environment: None,
    }));
    // Shape tests still have to be valid functions: control leaves a
    // non-empty block through its last instruction, or every successor
    // model sees a block with no edges (`VerifierError::MissingTerminator`).
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
}

/// A non-empty block whose last instruction is not a terminator is rejected.
///
/// Control falls off the end of such a block, and every successor model in the
/// crate reads edges from the last instruction alone - so the block contributes
/// none, and whatever it falls through to silently loses a predecessor. The
/// damage is downstream and quiet: a pass asking "is `A` `B`'s only
/// predecessor" gets `true` from the under-count exactly as it would from the
/// truth, and folds on it.
#[test]
fn verifier_rejects_a_block_control_falls_off_the_end_of() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
    let dest = typed_local(&mut ssa, 0, 0, 0, MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest,
        value: ConstValue::I32(1),
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .iter()
            .any(|error| matches!(error, VerifierError::MissingTerminator { block: 0 })),
        "a block ending in a `Const` never transfers control"
    );
}

/// An *empty* block is not that defect, and is accepted.
///
/// It carries no instruction to fall off the end of. Passes leave them behind
/// as placeholders with no predecessor, so no edge is lost by their having no
/// terminator.
#[test]
fn verifier_accepts_an_empty_placeholder_block() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
    let mut entry = SsaBlock::new(0);
    entry.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(entry);
    ssa.add_block(SsaBlock::new(1));
    ssa.recompute_uses();

    assert!(
        !SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .iter()
            .any(|error| matches!(error, VerifierError::MissingTerminator { .. })),
        "an empty block has no instruction for control to fall off the end of"
    );
}

#[test]
fn verifier_rejects_mismatched_vector_binary_shapes() {
    let mut ssa = SsaFunction::new(0, 3);
    let left = typed_local(&mut ssa, 0, 0, 0, MockType::V4I32);
    let right = typed_local(&mut ssa, 1, 0, 1, MockType::V2F64);
    let dest = typed_local(&mut ssa, 2, 0, 2, MockType::V4I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::Vector(some_or_abort(VectorConst::splat(32, 4, 1))),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::Vector(some_or_abort(VectorConst::splat(
            64,
            2,
            u128::from(1.0f64.to_bits()),
        ))),
    }));
    block.add_instruction(instr(SsaOp::VectorBinary {
        dest,
        left,
        right,
        kind: VectorBinaryKind::Add,
        element: VectorElement::default(),
        environment: None,
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidVectorOperation { .. }))
    );
}

/// A cast, saturating or not, converts each lane into a lane: one that changes
/// the lane count (four lanes into two) is rejected, the same lane count
/// accepted. Saturation clamps to an integer range, so a saturating cast to
/// floats is rejected too.
#[test]
fn verifier_rejects_a_conversion_that_changes_the_lane_count() {
    let rejected = |op: fn(SsaVarId, SsaVarId) -> SsaOp<MockTarget>, target: MockType| {
        // `target` types the destination the operation writes.
        let mut ssa = SsaFunction::new(0, 2);
        let source = typed_local(&mut ssa, 0, 0, 0, MockType::V4I32);
        let dest = typed_local(&mut ssa, 1, 0, 1, target);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: source,
            value: ConstValue::Vector(some_or_abort(VectorConst::splat(32, 4, 1))),
        }));
        block.add_instruction(instr(op(dest, source)));
        ssa.add_block(block);
        ssa.recompute_uses();
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidVectorOperation { .. }))
    };
    let cast_to_doubles = |dest, value| SsaOp::VectorCast {
        dest,
        value,
        target_type: MockType::V2F64,
        from: VectorCastKind::Signed,
        to: VectorCastKind::Float,
        rounding: None,
        saturate: false,
        environment: None,
        format: None,
    };
    let cast_in_place = |dest, value| SsaOp::VectorCast {
        dest,
        value,
        target_type: MockType::V4I32,
        from: VectorCastKind::Signed,
        to: VectorCastKind::Signed,
        rounding: None,
        saturate: false,
        environment: None,
        format: None,
    };
    let narrow = |dest, value| SsaOp::VectorCast {
        dest,
        value,
        target_type: MockType::V4I16,
        from: VectorCastKind::Signed,
        to: VectorCastKind::Signed,
        rounding: None,
        saturate: true,
        environment: None,
        format: None,
    };
    let narrow_into_pairs = |dest, value| SsaOp::VectorCast {
        dest,
        value,
        target_type: MockType::V2I256,
        from: VectorCastKind::Signed,
        to: VectorCastKind::Signed,
        rounding: None,
        saturate: true,
        environment: None,
        format: None,
    };
    let saturate_to_floats = |dest, value| SsaOp::VectorCast {
        dest,
        value,
        target_type: MockType::V4I32,
        from: VectorCastKind::Signed,
        to: VectorCastKind::Float,
        rounding: None,
        saturate: true,
        environment: None,
        format: None,
    };
    assert!(rejected(cast_to_doubles, MockType::V2F64));
    assert!(!rejected(cast_in_place, MockType::V4I32));
    assert!(!rejected(narrow, MockType::V4I16));
    assert!(rejected(narrow_into_pairs, MockType::V2I256));
    assert!(rejected(saturate_to_floats, MockType::V4I32));
}

#[test]
fn verifier_rejects_invalid_vector_shuffle_lane() {
    let mut ssa = SsaFunction::new(0, 2);
    let source = typed_local(&mut ssa, 0, 0, 0, MockType::V4I32);
    let dest = typed_local(&mut ssa, 1, 0, 1, MockType::V4I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: source,
        value: ConstValue::Vector(some_or_abort(VectorConst::splat(32, 4, 1))),
    }));
    block.add_instruction(instr(SsaOp::VectorShuffle {
        dest,
        left: source,
        right: None,
        selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
            VectorShuffleLane::Left(0),
            VectorShuffleLane::Left(1),
            VectorShuffleLane::Left(2),
            VectorShuffleLane::Left(4),
        ])),
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidVectorOperation { .. }))
    );
}

/// A shuffle by an index vector has a lane per index lane, as wide as its
/// inputs' lanes, and its index lanes are integers: one of four 32-bit
/// lanes indexed by two lanes, of 16-bit lanes from a 32-bit table, or of
/// doubles indexed by doubles is no shuffle, nor is one of tables of two lane shapes.
#[test]
fn verifier_holds_an_index_shuffle_to_its_index_and_tables() {
    let rejected = |left: MockType, right: Option<MockType>, index: MockType, dest: MockType| {
        let mut ssa = SsaFunction::new(3, 0);
        let argument = |ssa: &mut SsaFunction<MockTarget>, at: u16, ty: MockType| {
            ssa.create_variable(VariableOrigin::Argument(at), 0, DefSite::entry(), ty)
        };
        let left = argument(&mut ssa, 0, left);
        let right = right.map(|right| argument(&mut ssa, 1, right));
        let index = argument(&mut ssa, 2, index);
        let dest = typed_local(&mut ssa, 0, 0, 0, dest);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::VectorShuffle {
            dest,
            left,
            right,
            selector: VectorShuffleSelector::Index(index),
        }));
        block.add_instruction(instr(SsaOp::returns(Some(dest))));
        ssa.add_block(block);
        ssa.refresh_def_sites();
        ssa.recompute_uses();
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .iter()
            .any(|error| matches!(error, VerifierError::InvalidVectorOperation { .. }))
    };
    let words = MockType::V4I32;
    assert!(!rejected(words, None, words, words));
    assert!(!rejected(words, Some(words), words, words));
    // Bytes indexed by bytes, four of each.
    assert!(!rejected(
        MockType::V4I8,
        None,
        MockType::V4I8,
        MockType::V4I8
    ));
    // Words indexed by halfwords: a lane per index lane, as wide as a word.
    assert!(!rejected(words, None, MockType::V4I16, words));
    assert!(rejected(words, None, MockType::V2I256, words));
    assert!(rejected(words, None, words, MockType::V4I16));
    let doubles = MockType::V2F64;
    assert!(rejected(doubles, None, doubles, doubles));
    assert!(rejected(words, Some(MockType::V4I16), words, words));
}

#[test]
fn mock_target_exposes_vector_descriptors() {
    let shape = some_or_abort(MockTarget::vector_shape(&MockType::V4I32));
    assert_eq!(shape.lane_count, 4);
    assert_eq!(shape.lane_kind, VectorLaneKind::Integer);
    assert_eq!(shape.lane_bits, 32);
    assert_eq!(MockTarget::vector_type(shape), Some(MockType::V4I32));
    assert_eq!(MockTarget::vector_lane_type(shape), Some(MockType::I32));

    let scalable = some_or_abort(MockTarget::scalable_vector_shape(&MockType::NxV4I32));
    assert_eq!(scalable.min_lane_count, 4);
    assert_eq!(scalable.lane_kind, VectorLaneKind::Integer);
    assert_eq!(scalable.lane_bits, 32);
    assert_eq!(
        MockTarget::vector_descriptor(&MockType::NxV4I32),
        Some(VectorDescriptor::Scalable(scalable))
    );
    assert_eq!(
        MockTarget::scalable_vector_type(scalable),
        Some(MockType::NxV4I32)
    );
    assert_eq!(
        MockTarget::scalable_vector_lane_type(scalable),
        Some(MockType::I32)
    );
}

#[test]
fn verifier_accepts_scalable_vector_binary_and_compare() {
    let mut ssa = SsaFunction::new(0, 5);
    let scalar = typed_local(&mut ssa, 0, 0, 0, MockType::I32);
    let left = typed_local(&mut ssa, 1, 0, 1, MockType::NxV4I32);
    let right = typed_local(&mut ssa, 2, 0, 2, MockType::NxV4I32);
    let sum = typed_local(&mut ssa, 3, 0, 3, MockType::NxV4I32);
    let mask = typed_local(&mut ssa, 4, 0, 4, MockType::NxMask4);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: scalar,
        value: ConstValue::I32(7),
    }));
    block.add_instruction(instr(SsaOp::VectorSplat {
        dest: left,
        value: scalar,
        vector_type: MockType::NxV4I32,
    }));
    block.add_instruction(instr(SsaOp::VectorSplat {
        dest: right,
        value: scalar,
        vector_type: MockType::NxV4I32,
    }));
    block.add_instruction(instr(SsaOp::VectorBinary {
        dest: sum,
        left,
        right,
        kind: VectorBinaryKind::Add,
        element: VectorElement::default(),
        environment: None,
    }));
    block.add_instruction(instr(SsaOp::VectorCompare {
        dest: mask,
        left: sum,
        right,
        kind: VectorCompareKind::Eq,
        element: VectorElement::default(),
        signaling: false,
    }));
    // Shape tests still have to be valid functions: control leaves a
    // non-empty block through its last instruction, or every successor
    // model sees a block with no edges (`VerifierError::MissingTerminator`).
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
}

#[test]
fn verifier_rejects_fixed_and_scalable_vector_mix() {
    let mut ssa = SsaFunction::new(0, 4);
    let scalar = typed_local(&mut ssa, 0, 0, 0, MockType::I32);
    let fixed = typed_local(&mut ssa, 1, 0, 1, MockType::V4I32);
    let scalable = typed_local(&mut ssa, 2, 0, 2, MockType::NxV4I32);
    let dest = typed_local(&mut ssa, 3, 0, 3, MockType::NxV4I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: scalar,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: fixed,
        value: ConstValue::Vector(some_or_abort(VectorConst::splat(32, 4, 1))),
    }));
    block.add_instruction(instr(SsaOp::VectorSplat {
        dest: scalable,
        value: scalar,
        vector_type: MockType::NxV4I32,
    }));
    block.add_instruction(instr(SsaOp::VectorBinary {
        dest,
        left: fixed,
        right: scalable,
        kind: VectorBinaryKind::Add,
        element: VectorElement::default(),
        environment: None,
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidVectorOperation { .. }))
    );
}

#[test]
fn verifier_accepts_modern_simd_masked_memory_and_bitmask_ops() {
    let mut ssa = SsaFunction::new(0, 7);
    let addr = typed_local(&mut ssa, 0, 0, 0, MockType::Ptr);
    let mask_a = typed_local(&mut ssa, 1, 0, 1, MockType::Mask4);
    let mask_b = typed_local(&mut ssa, 2, 0, 2, MockType::Mask4);
    let mask_merged = typed_local(&mut ssa, 3, 0, 3, MockType::Mask4);
    let vector = typed_local(&mut ssa, 4, 0, 4, MockType::V4I32);
    let bits = typed_local(&mut ssa, 5, 0, 5, MockType::I32);
    let reduced = typed_local(&mut ssa, 6, 0, 6, MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: addr,
        arg_index: 0,
    }));
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: mask_a,
        arg_index: 1,
    }));
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: mask_b,
        arg_index: 2,
    }));
    block.add_instruction(instr(SsaOp::And {
        dest: mask_merged,
        left: mask_a,
        right: mask_b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::VectorMaskedLoad {
        dest: vector,
        addr,
        mask: mask_merged,
        passthrough: None,
        vector_type: MockType::V4I32,
        mode: VectorMaskMode::Zero,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(instr(SsaOp::VectorBitmask {
        dest: bits,
        value: vector,
        kind: VectorBitmaskKind::LaneMostSignificantBits,
    }));
    block.add_instruction(instr(SsaOp::VectorReduce {
        dest: reduced,
        value: vector,
        kind: VectorReduceKind::Add,
        element: VectorElement::default(),
        initial: None,
        ordered: false,
        groups: None,
        environment: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(reduced))));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
}

#[test]
fn verifier_rejects_masked_vector_memory_with_wrong_mask_shape() {
    let mut ssa = SsaFunction::new(0, 4);
    let addr = typed_local(&mut ssa, 0, 0, 0, MockType::Ptr);
    let bad_mask = typed_local(&mut ssa, 1, 0, 1, MockType::I32);
    let vector = typed_local(&mut ssa, 2, 0, 2, MockType::V4I32);
    let passthrough = typed_local(&mut ssa, 3, 0, 3, MockType::V4I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: addr,
        arg_index: 0,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: bad_mask,
        value: ConstValue::I32(0xf),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: passthrough,
        value: ConstValue::Vector(some_or_abort(VectorConst::splat(32, 4, 0))),
    }));
    block.add_instruction(instr(SsaOp::VectorMaskedLoad {
        dest: vector,
        addr,
        mask: bad_mask,
        passthrough: Some(passthrough),
        vector_type: MockType::V4I32,
        mode: VectorMaskMode::Merge,
        access: AccessQualifiers::FLAT,
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidVectorOperation { .. }))
    );
}

#[test]
fn verifier_accepts_gather_and_scatter_shapes() {
    let mut ssa = SsaFunction::new(0, 5);
    let base = typed_local(&mut ssa, 0, 0, 0, MockType::Ptr);
    let indices = typed_local(&mut ssa, 1, 0, 1, MockType::V4I32);
    let mask = typed_local(&mut ssa, 2, 0, 2, MockType::Mask4);
    let gathered = typed_local(&mut ssa, 3, 0, 3, MockType::V4I32);
    let broadcast = typed_local(&mut ssa, 4, 0, 4, MockType::V4I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: base,
        arg_index: 0,
    }));
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: indices,
        arg_index: 1,
    }));
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: mask,
        arg_index: 2,
    }));
    block.add_instruction(instr(SsaOp::VectorGather {
        dest: gathered,
        base,
        indices,
        mask,
        passthrough: None,
        vector_type: MockType::V4I32,
        mode: VectorMaskMode::Zero,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(instr(SsaOp::VectorBroadcastLoad {
        dest: broadcast,
        addr: base,
        vector_type: MockType::V4I32,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(instr(SsaOp::VectorScatter {
        base,
        indices,
        value: gathered,
        mask,
        vector_type: MockType::V4I32,
        access: AccessQualifiers::FLAT,
    }));
    // Shape tests still have to be valid functions: control leaves a
    // non-empty block through its last instruction, or every successor
    // model sees a block with no edges (`VerifierError::MissingTerminator`).
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
}

#[test]
fn verifier_accepts_faulting_and_segment_vector_memory_shapes() {
    let mut ssa = SsaFunction::new(0, 8);
    let base = typed_local(&mut ssa, 0, 0, 0, MockType::Ptr);
    let mask = typed_local(&mut ssa, 1, 0, 1, MockType::Mask4);
    let passthrough = typed_local(&mut ssa, 2, 0, 2, MockType::V4I32);
    let faulting = typed_local(&mut ssa, 3, 0, 3, MockType::V4I32);
    let fault = typed_local(&mut ssa, 4, 0, 3, MockType::Mask4);
    let seg0 = typed_local(&mut ssa, 5, 0, 4, MockType::V4I32);
    let seg1 = typed_local(&mut ssa, 6, 0, 4, MockType::V4I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: base,
        arg_index: 0,
    }));
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: mask,
        arg_index: 1,
    }));
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: passthrough,
        arg_index: 2,
    }));
    block.add_instruction(instr(SsaOp::VectorFaultingLoad {
        dest: faulting,
        fault: Some(fault),
        addr: base,
        mask: Some(mask),
        passthrough: Some(passthrough),
        vector_type: MockType::V4I32,
        fault_mode: VectorFaultMode::FirstFault,
        mask_mode: VectorMaskMode::Merge,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(instr(SsaOp::VectorSegmentLoad {
        dests: vec![seg0, seg1],
        base,
        mask: Some(mask),
        vector_type: MockType::V4I32,
        segments: 2,
        layout: VectorSegmentLayout::Interleaved,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(instr(SsaOp::VectorSegmentStore {
        base,
        values: vec![seg0, seg1],
        mask: Some(mask),
        vector_type: MockType::V4I32,
        segments: 2,
        layout: VectorSegmentLayout::Interleaved,
        access: AccessQualifiers::FLAT,
    }));
    // Shape tests still have to be valid functions: control leaves a
    // non-empty block through its last instruction, or every successor
    // model sees a block with no edges (`VerifierError::MissingTerminator`).
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
}

#[test]
fn verifier_rejects_vector_segment_count_mismatch() {
    let mut ssa = SsaFunction::new(0, 4);
    let base = typed_local(&mut ssa, 0, 0, 0, MockType::Ptr);
    let seg0 = typed_local(&mut ssa, 1, 0, 1, MockType::V4I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: base,
        arg_index: 0,
    }));
    block.add_instruction(instr(SsaOp::VectorSegmentLoad {
        dests: vec![seg0],
        base,
        mask: None,
        vector_type: MockType::V4I32,
        segments: 2,
        layout: VectorSegmentLayout::Consecutive,
        access: AccessQualifiers::FLAT,
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidVectorOperation { .. }))
    );
}

#[test]
fn verifier_accepts_native_atomic_cmpxchg_with_status_output() {
    let mut ssa = SsaFunction::new(0, 5);
    let addr = typed_local(&mut ssa, 0, 0, 0, MockType::Ptr);
    let expected = typed_local(&mut ssa, 1, 0, 1, MockType::I32);
    let desired = typed_local(&mut ssa, 2, 0, 2, MockType::I32);
    let old = typed_local(&mut ssa, 3, 0, 3, MockType::I32);
    let success = typed_local(&mut ssa, 4, 0, 3, MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: addr,
        arg_index: 0,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: expected,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: desired,
        value: ConstValue::I32(20),
    }));
    block.add_instruction(instr(SsaOp::AtomicCmpXchg {
        old,
        success: Some(success),
        addr,
        expected,
        desired,
        success_ordering: AtomicOrdering::SeqCst,
        failure_ordering: AtomicOrdering::Acquire,
        width: AtomicAccessWidth::Bits32,
        weak: false,
        access: AccessQualifiers::FLAT.made_volatile(),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(old))));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
}

#[test]
fn verifier_rejects_native_atomic_width_mismatch() {
    let mut ssa = SsaFunction::new(0, 3);
    let addr = typed_local(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = typed_local(&mut ssa, 1, 0, 1, MockType::I64);
    let old = typed_local(&mut ssa, 2, 0, 2, MockType::I64);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: addr,
        arg_index: 0,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I64(10),
    }));
    block.add_instruction(instr(SsaOp::AtomicExchange {
        dest: old,
        addr,
        value,
        ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits32,
        access: AccessQualifiers::FLAT,
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidAtomicOperation { .. }))
    );
}

/// A read-modify-write may release: RISC-V `amoadd.w.rl`, AArch64 `ldaddl`
/// and a C++ `fetch_add` at `memory_order_release` are all one.
#[test]
fn verifier_accepts_a_release_read_modify_write() {
    let mut ssa = SsaFunction::new(0, 3);
    let addr = typed_local(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = typed_local(&mut ssa, 1, 0, 1, MockType::I32);
    let old = typed_local(&mut ssa, 2, 0, 2, MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: addr,
        arg_index: 0,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::AtomicLockRmw {
        dest: old,
        addr,
        value,
        op: AtomicRmwOp::Add,
        ordering: AtomicOrdering::Release,
        width: AtomicAccessWidth::Bits32,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(old))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(errors.is_empty(), "{errors:?}");
}

/// A compare-exchange that fails stores nothing, so its failure path cannot
/// release, a pair one's no more than a single one's; and it cannot be
/// stronger than the success path.
#[test]
fn verifier_rejects_illegal_compare_exchange_ordering() {
    let verify = |pair: bool, success: AtomicOrdering, failure: AtomicOrdering| {
        let mut ssa = SsaFunction::new(0, 6);
        let addr = typed_local(&mut ssa, 0, 0, 0, MockType::Ptr);
        let value = typed_local(&mut ssa, 1, 0, 1, MockType::I32);
        let old = typed_local(&mut ssa, 2, 0, 2, MockType::I32);
        let old_second = typed_local(&mut ssa, 3, 0, 3, MockType::I32);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::LoadArg {
            dest: addr,
            arg_index: 0,
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(if pair {
            SsaOp::AtomicPairCmpXchg {
                old_first: old,
                old_second,
                addr,
                expected_first: value,
                expected_second: value,
                desired_first: value,
                desired_second: value,
                success_ordering: success,
                failure_ordering: failure,
                width: AtomicAccessWidth::Bits64,
                weak: false,
                access: AccessQualifiers::FLAT,
            }
        } else {
            SsaOp::AtomicCmpXchg {
                old,
                success: None,
                addr,
                expected: value,
                desired: value,
                success_ordering: success,
                failure_ordering: failure,
                width: AtomicAccessWidth::Bits32,
                weak: false,
                access: AccessQualifiers::FLAT,
            }
        }));
        block.add_instruction(instr(SsaOp::returns(Some(old))));
        ssa.add_block(block);
        ssa.recompute_uses();
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidAtomicOperation { .. }))
    };
    for pair in [false, true] {
        assert!(
            !verify(pair, AtomicOrdering::Release, AtomicOrdering::Relaxed),
            "pair {pair}"
        );
        assert!(
            !verify(pair, AtomicOrdering::SeqCst, AtomicOrdering::SeqCst),
            "pair {pair}"
        );
        assert!(
            verify(pair, AtomicOrdering::SeqCst, AtomicOrdering::Release),
            "pair {pair}"
        );
        assert!(
            verify(pair, AtomicOrdering::AcqRel, AtomicOrdering::AcqRel),
            "pair {pair}"
        );
        assert!(
            verify(pair, AtomicOrdering::Relaxed, AtomicOrdering::Acquire),
            "pair {pair}"
        );
    }
}

#[test]
fn verifier_accepts_wide_multiply_matching_half_widths() {
    let mut ssa = SsaFunction::new(0, 4);
    let left = typed_local(&mut ssa, 0, 0, 0, MockType::I32);
    let right = typed_local(&mut ssa, 1, 0, 1, MockType::I32);
    let low = typed_local(&mut ssa, 2, 0, 2, MockType::I32);
    let high = typed_local(&mut ssa, 3, 0, 2, MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(7),
    }));
    block.add_instruction(instr(SsaOp::WideMul {
        low,
        high,
        left,
        right,
        unsigned: true,
    }));
    // Shape tests still have to be valid functions: control leaves a
    // non-empty block through its last instruction, or every successor
    // model sees a block with no edges (`VerifierError::MissingTerminator`).
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
}

#[test]
fn verifier_rejects_wide_divide_width_mismatch() {
    let mut ssa = SsaFunction::new(0, 5);
    let high = typed_local(&mut ssa, 0, 0, 0, MockType::I32);
    let low = typed_local(&mut ssa, 1, 0, 1, MockType::I32);
    let divisor = typed_local(&mut ssa, 2, 0, 2, MockType::I64);
    let quotient = typed_local(&mut ssa, 3, 0, 3, MockType::I32);
    let remainder = typed_local(&mut ssa, 4, 0, 3, MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: high,
        value: ConstValue::I32(0),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: low,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: divisor,
        value: ConstValue::I64(3),
    }));
    block.add_instruction(instr(SsaOp::WideDiv {
        quotient,
        remainder,
        high,
        low,
        divisor,
        unsigned: true,
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidWideArithmetic { .. }))
    );
}

/// The legacy shape - a handler's leading `Pop` of a variable nothing defines -
/// is reported, and `adopt_handler_entry_values` is what repairs it.
///
/// The older model left the exception object undefined and had the verifier
/// carve the shape out by position. That made the IR's correctness depend on an
/// instruction staying first in a block any pass may rewrite. The object now
/// has a definition, so the carve-out is gone and the same function reads a
/// value nothing produces - until the migration gives it one.
#[test]
fn a_handler_entry_value_is_adopted_into_a_definition() {
    let mut ssa = SsaFunction::new(0, 1);
    let guarded = local(&mut ssa, 0, 0, 0);
    // The exception object as a frontend built it before `RuntimeValue`: a
    // stack temporary with no defining instruction.
    let exception = ssa.create_variable(VariableOrigin::Phi, 1, DefSite::entry(), MockType::I32);

    let mut try_block = SsaBlock::new(0);
    try_block.add_instruction(instr(SsaOp::Const {
        dest: guarded,
        value: ConstValue::I32(1),
    }));
    try_block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(try_block);

    let mut handler = SsaBlock::new(1);
    handler.add_instruction(instr(SsaOp::Pop { value: exception }));
    handler.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(handler);

    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 1,
        handler_offset: 1,
        handler_length: 1,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(0, 1),
        handler_range: BlockRange::new(1, 2),
        filter_range: None,
        selector: None,
    }]);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::UndefinedUse { var, .. } if *var == exception)),
        "the legacy shape reads a value nothing defines; got {errors:?}"
    );

    assert_eq!(ssa.adopt_handler_entry_values(), 1);
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        errors.is_empty(),
        "adoption turns the implied definition into a real one; got {errors:?}"
    );

    // Adoption is keyed on that one leading `Pop`. Any other undefined read in
    // the handler stays a dangling read, and running it again finds nothing
    // more to adopt.
    let stray = ssa.create_variable(VariableOrigin::Phi, 1, DefSite::entry(), MockType::I32);
    if let Some(block) = ssa.block_mut(1) {
        block.add_instruction(instr(SsaOp::Throw { exception: stray }));
    }
    ssa.recompute_uses();
    assert_eq!(ssa.adopt_handler_entry_values(), 0);
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::UndefinedUse { var, .. } if *var == stray)),
        "only the object the handler is entered with is runtime-supplied; got {errors:?}"
    );
}

/// A mask naming a flag the crate does not define is reported rather than
/// silently narrowed.
///
/// `FlagsMask::from_bits` is unmasked so a persisted mask round-trips, which
/// means a blob written by an older or a buggy producer can carry a bit nothing
/// downstream can interpret. Reporting it as `InvalidNativeOperation` is what
/// turns that into a named failure instead of a mask that quietly selects
/// fewer flags than it says.
#[test]
fn verifier_rejects_a_read_flags_mask_with_undefined_bits() {
    let mut ssa = SsaFunction::new(0, 1);
    let flags = local(&mut ssa, 0, 0, 0);
    let dest = local(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: flags,
        value: ConstValue::I32(0),
    }));
    block.add_instruction(instr(SsaOp::ReadFlags {
        dest,
        flags,
        mask: FlagsMask::from_bits(FlagsMask::CARRY.bits() | (1 << 15)),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(dest))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidNativeOperation { .. })),
        "an undefined flag bit must be reported; got {errors:?}"
    );

    // Every defined bit together is still a legal mask, so the check reports
    // the malformed shape rather than flag masks in general.
    if let Some(block) = ssa.block_mut(0)
        && let Some(op) = block.instructions_mut().get_mut(1)
    {
        *op = instr(SsaOp::ReadFlags {
            dest,
            flags,
            mask: FlagsMask::ALL,
        });
    }
    ssa.recompute_uses();
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidNativeOperation { .. })),
        "FlagsMask::ALL names only defined flags; got {errors:?}"
    );
}

/// A two-block function whose single clause is `handlers`.
fn function_with_clauses(
    handlers: Vec<SsaExceptionHandler<MockTarget>>,
) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    for index in 0..2 {
        let mut block = SsaBlock::new(index);
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
    }
    ssa.set_exception_handlers(handlers);
    ssa.recompute_uses();
    ssa
}

/// One clause, three verdicts.
///
/// A clause naming a block the function does not have is reported at Quick --
/// it is corruption every later check would then reason from. A clause whose
/// handler lives in another function is legal and silent: on Windows x64 a
/// handler is a funclet, as nearly every clause of a Windows x64 binary is.
/// And a clause that fits is silent too.
#[test]
fn verifier_reports_a_malformed_exception_clause() {
    let mut clause = SsaExceptionHandler::<MockTarget> {
        flags: 0,
        try_offset: 0,
        try_length: 0,
        handler_offset: 0,
        handler_length: 0,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(0, 1),
        handler_range: BlockRange::new(1, 9),
        filter_range: None,
        selector: None,
    };

    let errors =
        SsaVerifier::new(&function_with_clauses(vec![clause.clone()])).verify(VerifyLevel::Quick);
    assert!(
        errors.iter().any(|e| matches!(
            e,
            VerifierError::MalformedExceptionClause {
                clause: 0,
                reason: ExceptionTableError::OutOfRange {
                    part: ClausePart::Handler,
                    end: 9,
                    block_count: 2,
                },
            }
        )),
        "a handler range past the last block is reportable at Quick; got {errors:?}"
    );

    clause.handler_range = None;
    let errors =
        SsaVerifier::new(&function_with_clauses(vec![clause.clone()])).verify(VerifyLevel::Full);
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e, VerifierError::MalformedExceptionClause { .. })),
        "a handler in another function is legal at every level; got {errors:?}"
    );

    clause.handler_range = BlockRange::new(1, 2);
    let errors = SsaVerifier::new(&function_with_clauses(vec![clause])).verify(VerifyLevel::Full);
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e, VerifierError::MalformedExceptionClause { .. })),
        "and a clause that fits raises nothing; got {errors:?}"
    );
}

/// A phi operand naming a value whose definition was removed is a dangling read,
/// and the Quick tripwire rejects it.
///
/// Quick is what a release build verifies each pass group at. Without the read
/// check there, a pass that deletes a definition under a live phi operand is
/// accepted in release and rolled back in debug - the same bytes recovered two
/// ways.
#[test]
fn quick_verification_rejects_a_dangling_phi_operand() {
    let mut ssa = SsaFunction::new(0, 0);
    let cond = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let gone = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(1, 0),
        MockType::I32,
    );
    let kept = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(2, 0),
        MockType::I32,
    );
    let merged = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(3), MockType::I32);

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
    let mut b1 = SsaBlock::new(1);
    // `gone`'s definition, already removed: only the phi operand still names it.
    b1.add_instruction(instr(SsaOp::Nop));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: kept,
        value: ConstValue::I32(2),
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
    let mut b3 = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Phi);
    phi.add_operand(PhiOperand::new(gone, 1));
    phi.add_operand(PhiOperand::new(kept, 2));
    b3.add_phi(phi);
    b3.add_instruction(instr(SsaOp::returns(Some(merged))));
    for block in [b0, b1, b2, b3] {
        ssa.add_block(block);
    }
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(
        errors
            .iter()
            .any(|error| matches!(error, VerifierError::UndefinedUse { var, .. } if *var == gone)),
        "Quick must reject the dangling phi operand; got {errors:?}"
    );
}

/// Builds a loop whose header is the entry block: B0 tests, B1 is the body
/// and jumps back to B0, B2 returns. With `with_phi` the header carries a
/// counter phi fed only by the back edge.
fn loop_at_entry(with_phi: bool) -> (SsaFunction<MockTarget>, SsaVarId) {
    let mut ssa = SsaFunction::new(1, 1);
    let arg = ssa.create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::I32,
    );
    let counter = with_phi
        .then(|| ssa.create_variable(VariableOrigin::Local(0), 1, DefSite::phi(0), MockType::I32));
    let limit = local(&mut ssa, 1, 0, 0);
    let cond = local(&mut ssa, 2, 0, 1);
    let one = local(&mut ssa, 3, 1, 0);
    let next = ssa.create_variable(
        VariableOrigin::Local(0),
        2,
        DefSite::instruction(1, 1),
        MockType::I32,
    );
    let tested = counter.unwrap_or(arg);

    let mut b0 = SsaBlock::new(0);
    if let Some(counter) = counter {
        let mut phi = PhiNode::new(counter, VariableOrigin::Local(0));
        phi.add_operand(PhiOperand::new(next, 1));
        b0.add_phi(phi);
    }
    b0.add_instruction(instr(SsaOp::Const {
        dest: limit,
        value: ConstValue::I32(4),
    }));
    b0.add_instruction(instr(SsaOp::Clt {
        dest: cond,
        left: tested,
        right: limit,
        unsigned: false,
        signaling: false,
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    b1.add_instruction(instr(SsaOp::Add {
        dest: next,
        left: tested,
        right: one,
        flags: None,
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 0 }));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(Some(tested))));
    ssa.add_block(b2);
    ssa.recompute_uses();
    (ssa, arg)
}

/// A back edge into the entry block is reported, from the block that owns it.
#[test]
fn a_back_edge_into_the_entry_block_is_rejected() {
    let (ssa, _) = loop_at_entry(false);
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);
    assert!(
        errors.contains(&VerifierError::EntryHasPredecessors { predecessor: 1 }),
        "B1 jumps to B0: {errors:?}"
    );
}

/// Splitting off a fresh entry leaves a function the verifier accepts, with
/// every block reference shifted and the argument still defined at entry.
#[test]
fn a_loop_at_the_entry_gets_a_fresh_entry_block() {
    let (mut ssa, arg) = loop_at_entry(false);
    assert!(ssa.ensure_entry_has_no_predecessors());

    assert_eq!(ssa.block_count(), 4);
    assert!(matches!(
        ssa.block(0).and_then(SsaBlock::control_terminator),
        Some(SsaOp::Jump { target: 1 })
    ));
    assert!(matches!(
        ssa.block(2).and_then(SsaBlock::control_terminator),
        Some(SsaOp::Jump { target: 1 })
    ));
    assert!(matches!(
        ssa.block(1).and_then(SsaBlock::control_terminator),
        Some(SsaOp::Branch {
            true_target: 2,
            false_target: 3,
            ..
        })
    ));
    assert_eq!(
        ssa.variable(arg).map(|v| v.def_site()),
        Some(DefSite::entry()),
        "an argument is defined where the call enters, which is the new block 0"
    );
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}");
    assert!(
        !ssa.ensure_entry_has_no_predecessors(),
        "a second call finds nothing to do"
    );
}

/// A phi in the old entry block gains an operand for the new entry edge.
///
/// The phi had no operand for "control arrives from the caller" because the
/// entry block had no edge to name it; once that edge exists, the phi needs
/// the entry value of its variable.
#[test]
fn an_entry_phi_gains_an_operand_for_the_new_entry_edge() {
    let (mut ssa, _) = loop_at_entry(true);
    assert!(ssa.ensure_entry_has_no_predecessors());
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}\n{ssa}");
    let header_phi_preds: Vec<Vec<usize>> = some_or_abort(ssa.block(1))
        .phi_nodes()
        .iter()
        .map(|phi| {
            let mut preds: Vec<usize> =
                phi.operands().iter().map(PhiOperand::predecessor).collect();
            preds.sort_unstable();
            preds
        })
        .collect();
    assert_eq!(header_phi_preds, vec![vec![0, 2]]);
    // No value of the counter is defined at entry, so it reads a new entry
    // live-in there: whatever the slot held, which nothing computed.
    let entered = entry_operand(&ssa);
    let entered = some_or_abort(ssa.variable(entered));
    assert_eq!(entered.origin(), VariableOrigin::EntryLiveIn);
    assert_eq!(entered.def_site(), DefSite::entry());
}

/// The value the old entry phi reads on the new entry edge.
fn entry_operand(ssa: &SsaFunction<MockTarget>) -> SsaVarId {
    let phi = some_or_abort(some_or_abort(ssa.block(1)).phi_nodes().first());
    some_or_abort(
        phi.operands()
            .iter()
            .find(|operand| operand.predecessor() == 0)
            .map(PhiOperand::value),
    )
}

/// **An entry phi reads its variable's own entry value where one exists.** A
/// counter initialised at entry (version 0 of its local, defined there) is
/// what the first pass through the old entry block read, so the new entry
/// edge carries it rather than a value made up for the slot.
#[test]
fn an_entry_phi_reads_the_initialised_value_on_the_new_edge() {
    let (mut ssa, _) = loop_at_entry(true);
    let initial = ssa.create_variable(VariableOrigin::Local(0), 0, DefSite::entry(), MockType::I32);
    ssa.recompute_uses();
    assert!(ssa.ensure_entry_has_no_predecessors());
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}\n{ssa}");
    assert_eq!(entry_operand(&ssa), initial);
}

/// Exception clauses move with the blocks they cover.
#[test]
fn splitting_the_entry_shifts_exception_ranges() {
    let (mut ssa, _) = loop_at_entry(false);
    ssa.set_exception_handlers(vec![SsaExceptionHandler {
        flags: 0,
        try_offset: 0,
        try_length: 1,
        handler_offset: 1,
        handler_length: 1,
        class_token_or_filter: 0,
        protected_range: BlockRange::new(0, 2),
        handler_range: BlockRange::new(2, 3),
        filter_range: None,
        selector: None,
    }]);
    assert!(ssa.ensure_entry_has_no_predecessors());
    let handler = some_or_abort(ssa.exception_handlers().first());
    assert_eq!(handler.protected_range, BlockRange::new(1, 3));
    assert_eq!(handler.handler_range, BlockRange::new(3, 4));
}

/// A use list whose sites have shifted is stale even when its count is right.
///
/// Inserting an instruction above a use moves it one index down while every
/// variable keeps the same number of uses. The indexed replacement iterates
/// these sites, so a shifted one names an instruction that does not read the
/// value, and the one that does is never rewritten.
#[test]
fn a_use_list_with_shifted_sites_is_stale() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
    let value = local(&mut ssa, 0, 0, 1);
    let padding = local(&mut ssa, 1, 0, 0);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(value))));
    ssa.add_block(block);
    ssa.recompute_uses();

    some_or_abort(ssa.block_mut(0)).instructions_mut().insert(
        0,
        instr(SsaOp::Const {
            dest: padding,
            value: ConstValue::I32(0),
        }),
    );
    ssa.refresh_def_sites();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors.iter().any(|error| matches!(
            error,
            VerifierError::StaleUseList { var, recorded: 1, actual: 1 } if *var == value
        )),
        "{errors:?}"
    );

    ssa.recompute_uses();
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
}

/// A bitcast reads its operand's bits as a type of the same width: one of a
/// 128-bit vector read as a 64-bit float is a narrowing no bitcast states,
/// and the bitcast of a 64-bit integer to it is the control.
#[test]
fn verifier_rejects_a_bitcast_that_changes_the_width() {
    let build = |source: MockType| {
        let mut ssa = SsaFunction::new(0, 1);
        let value = ssa.create_variable(VariableOrigin::EntryLiveIn, 0, DefSite::entry(), source);
        let cast = typed_local(&mut ssa, 0, 0, 0, MockType::F64);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Bitcast {
            dest: cast,
            operand: value,
            target: MockType::F64,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(cast))));
        ssa.add_block(block);
        ssa.recompute_uses();
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard)
    };
    assert!(
        build(MockType::V2F64)
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidConversion { .. }))
    );
    assert!(build(MockType::I64).is_empty());
}

/// A float-to-integer conversion whose flag contradicts its integer target's
/// sign states two different conversions; the one agreeing with it is the
/// control.
#[test]
fn verifier_rejects_a_float_to_integer_flag_contradicting_its_target() {
    let build = |unsigned: bool| {
        let mut ssa = SsaFunction::new(0, 2);
        let value = typed_local(&mut ssa, 0, 0, 0, MockType::F64);
        let converted = typed_local(&mut ssa, 1, 0, 1, MockType::U32);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: value,
            value: ConstValue::from(2.0f64),
        }));
        block.add_instruction(instr(SsaOp::FloatToInt {
            dest: converted,
            operand: value,
            target: MockType::U32,
            overflow_check: false,
            unsigned,
            source: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(converted))));
        ssa.add_block(block);
        ssa.recompute_uses();
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard)
    };
    assert!(
        build(false)
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidConversion { .. }))
    );
    assert!(
        !build(true)
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidConversion { .. }))
    );
}

/// An integer conversion reads an integer and gives one. A float or a vector
/// on either side is refused; the integer, bool-width, pointer and unknown
/// sides are the controls, and so is a type the target classifies as
/// neither integer nor float (a managed reference), which is not judged.
#[test]
fn verifier_rejects_an_integer_conversion_of_or_to_a_non_integer() {
    let build = |source: MockType, target: MockType| {
        let mut ssa = SsaFunction::new(0, 1);
        let value = ssa.create_variable(VariableOrigin::EntryLiveIn, 0, DefSite::entry(), source);
        let converted = typed_local(&mut ssa, 0, 0, 0, target);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::IntConv {
            dest: converted,
            operand: value,
            target,
            unsigned: true,
            overflow_check: false,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(converted))));
        ssa.add_block(block);
        ssa.recompute_uses();
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .into_iter()
            .any(|e| matches!(e, VerifierError::InvalidConversion { .. }))
    };
    for (source, target) in [
        (MockType::U32, MockType::F64),
        (MockType::F64, MockType::U32),
        (MockType::BF16, MockType::U16),
        (MockType::U64, MockType::V2F64),
        (MockType::V4I32, MockType::U32),
    ] {
        assert!(build(source, target), "{source:?} -> {target:?}");
    }
    for (source, target) in [
        (MockType::Ref, MockType::U64),
        (MockType::U8, MockType::U32),
        (MockType::I64, MockType::I32),
        (MockType::I1, MockType::U32),
        (MockType::Ptr, MockType::U64),
        (MockType::U64, MockType::Ptr),
        (MockType::Unknown, MockType::U32),
        (MockType::U32, MockType::Unknown),
    ] {
        assert!(!build(source, target), "{source:?} -> {target:?}");
    }
}

// The checks share one scan of the function, so a check that stopped looking
// past some point of it would be invisible to a test that breaks the first
// block only. Each defect below is planted in the entry block, in the middle
// and in the last block of a long chain, and must be reported at the block it
// was planted in, at its level and not below it. The clean chain is the
// control: it verifies at every level, so the defects are the only difference.

/// How many blocks the chain holds; the defects are planted among them.
const CHAIN: usize = 9;

/// The index of the last block of the chain.
const LAST: usize = 8;

/// A chain of blocks, each defining one value and jumping to the next; the
/// last returns its own. Returns the function and the value of each block.
fn chain() -> (SsaFunction<MockTarget>, Vec<SsaVarId>) {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, CHAIN);
    let mut values: Vec<SsaVarId> = Vec::new();
    for block in 0..CHAIN {
        values.push(local(
            &mut ssa,
            u16::try_from(block).unwrap_or(u16::MAX),
            block,
            0,
        ));
    }
    for (block_idx, &value) in values.iter().enumerate() {
        let mut block = SsaBlock::new(block_idx);
        block.add_instruction(instr(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(i32::from(u16::try_from(block_idx).unwrap_or(u16::MAX))),
        }));
        if block_idx == LAST {
            block.add_instruction(instr(SsaOp::returns(Some(value))));
        } else {
            block.add_instruction(instr(SsaOp::Jump {
                target: block_idx.saturating_add(1),
            }));
        }
        ssa.add_block(block);
    }
    ssa.recompute_uses();
    ssa.assign_instr_ids();
    (ssa, values)
}

/// Replaces the last instruction of block `block`.
fn replace_terminator(ssa: &mut SsaFunction<MockTarget>, block: usize, op: SsaOp<MockTarget>) {
    *some_or_abort(
        some_or_abort(ssa.block_mut(block))
            .instructions_mut()
            .last_mut(),
    ) = instr(op);
}

/// Inserts `op` at the front of block `block`.
fn plant(ssa: &mut SsaFunction<MockTarget>, block: usize, op: SsaOp<MockTarget>) {
    some_or_abort(ssa.block_mut(block))
        .instructions_mut()
        .insert(0, instr(op));
}

/// A fresh registered value no block defines yet, for a planted instruction to
/// define.
fn scratch(ssa: &mut SsaFunction<MockTarget>) -> SsaVarId {
    let next = u16::try_from(ssa.variables().len()).unwrap_or(u16::MAX);
    local(ssa, next, 0, 0)
}

/// Where in the chain a defect is planted: the entry, the middle and the end.
const SITES: [usize; 3] = [0, 4, LAST];

/// Where in the chain a defect that needs a later block to read from is
/// planted.
const EARLY_SITES: [usize; 2] = [0, 4];

/// One planted defect: how to plant it in a block, the level it is reported
/// from, and how to recognise its report at that block.
struct Defect {
    name: &'static str,
    level: VerifyLevel,
    sites: &'static [usize],
    plant: fn(&mut SsaFunction<MockTarget>, &[SsaVarId], usize),
    reported: fn(&VerifierError, &[SsaVarId], usize) -> bool,
}

fn defects() -> Vec<Defect> {
    vec![
        Defect {
            name: "a second definition of an earlier value",
            level: VerifyLevel::Quick,
            sites: &SITES,
            plant: |ssa, values, block| {
                plant(
                    ssa,
                    block,
                    SsaOp::Const {
                        dest: *some_or_abort(values.first()),
                        value: ConstValue::I32(99),
                    },
                );
            },
            reported: |error, values, block| {
                matches!(
                    error,
                    VerifierError::DuplicateDefinition { var, def2, .. }
                        if *var == *some_or_abort(values.first()) && def2.block == block
                )
            },
        },
        Defect {
            name: "a use above its own definition in the block",
            level: VerifyLevel::Quick,
            sites: &SITES,
            plant: |ssa, values, block| {
                let sink = scratch(ssa);
                plant(
                    ssa,
                    block,
                    SsaOp::Copy {
                        dest: sink,
                        src: *some_or_abort(values.get(block)),
                    },
                );
            },
            reported: |error, values, block| {
                matches!(
                    error,
                    VerifierError::IntraBlockCycle { block: b, var, .. }
                        if *b == block && *var == *some_or_abort(values.get(block))
                )
            },
        },
        Defect {
            name: "an instruction reading its own result",
            level: VerifyLevel::Quick,
            sites: &SITES,
            plant: |ssa, values, block| {
                let own = *some_or_abort(values.get(block));
                *some_or_abort(
                    some_or_abort(ssa.block_mut(block))
                        .instructions_mut()
                        .first_mut(),
                ) = instr(SsaOp::Copy {
                    dest: own,
                    src: own,
                });
            },
            reported: |error, values, block| {
                matches!(
                    error,
                    VerifierError::SelfReferentialInstruction { block: b, var, .. }
                        if *b == block && *var == *some_or_abort(values.get(block))
                )
            },
        },
        Defect {
            name: "a placeholder operand",
            level: VerifyLevel::Quick,
            sites: &SITES,
            plant: |ssa, _, block| {
                let sink = scratch(ssa);
                plant(
                    ssa,
                    block,
                    SsaOp::Copy {
                        dest: sink,
                        src: SsaVarId::PLACEHOLDER,
                    },
                );
            },
            reported: |error, _, block| matches!(error, VerifierError::PlaceholderVariable { block: b, .. } if *b == block),
        },
        Defect {
            name: "a read of a value whose definition is gone",
            level: VerifyLevel::Quick,
            sites: &SITES,
            plant: |ssa, _, block| {
                let sink = scratch(ssa);
                let ghost = ssa.create_variable(
                    VariableOrigin::Local(40),
                    1,
                    DefSite::instruction(block, 5),
                    MockType::I32,
                );
                plant(
                    ssa,
                    block,
                    SsaOp::Copy {
                        dest: sink,
                        src: ghost,
                    },
                );
            },
            reported: |error, _, block| matches!(error, VerifierError::UndefinedUse { block: b, .. } if *b == block),
        },
        Defect {
            name: "a terminator that is not last",
            level: VerifyLevel::Quick,
            sites: &SITES,
            plant: |ssa, _, block| plant(ssa, block, SsaOp::Jump { target: 0 }),
            reported: |error, _, block| matches!(error, VerifierError::TerminatorNotLast { block: b, .. } if *b == block),
        },
        Defect {
            name: "a branch to a block that does not exist",
            level: VerifyLevel::Quick,
            sites: &SITES,
            plant: |ssa, _, block| {
                replace_terminator(ssa, block, SsaOp::Jump { target: 99 });
            },
            reported: |error, _, block| matches!(error, VerifierError::DanglingBranchTarget { block: b, .. } if *b == block),
        },
        Defect {
            name: "a runtime value where no runtime enters",
            level: VerifyLevel::Quick,
            sites: &SITES,
            plant: |ssa, _, block| {
                let sink = scratch(ssa);
                plant(
                    ssa,
                    block,
                    SsaOp::RuntimeValue {
                        dest: sink,
                        kind: RuntimeValueKind::ExceptionObject,
                    },
                );
            },
            reported: |error, _, block| matches!(error, VerifierError::MisplacedRuntimeValue { block: b, .. } if *b == block),
        },
        Defect {
            name: "an instruction identity used twice",
            level: VerifyLevel::Quick,
            sites: &SITES,
            plant: |ssa, _, block| {
                // Block 0 holds the original; planting there duplicates within
                // the block, anywhere else duplicates across blocks.
                let donor =
                    some_or_abort(ssa.block(0).and_then(|b| b.instructions().first())).clone();
                some_or_abort(ssa.block_mut(block))
                    .instructions_mut()
                    .insert(1, donor);
            },
            reported: |error, _, block| {
                matches!(
                    error,
                    VerifierError::DuplicateInstructionId { second, .. } if second.0 == block
                )
            },
        },
        Defect {
            name: "a definition no table row registers",
            level: VerifyLevel::Standard,
            sites: &SITES,
            plant: |ssa, _, block| {
                let stranger = SsaVarId::from_index(ssa.variables().len().saturating_add(3));
                plant(
                    ssa,
                    block,
                    SsaOp::Const {
                        dest: stranger,
                        value: ConstValue::I32(0),
                    },
                );
            },
            reported: |error, _, _| matches!(error, VerifierError::UnregisteredVariable { .. }),
        },
        Defect {
            name: "a use list naming a reader that is not there",
            level: VerifyLevel::Standard,
            sites: &SITES,
            plant: |ssa, values, block| {
                some_or_abort(ssa.variable_mut(*some_or_abort(values.get(block))))
                    .add_use(UseSite::instruction(block, 1));
            },
            reported: |error, values, block| matches!(error, VerifierError::StaleUseList { var, .. } if *var == *some_or_abort(values.get(block))),
        },
        Defect {
            name: "a float-to-integer flag contradicting its target",
            level: VerifyLevel::Standard,
            sites: &SITES,
            plant: |ssa, values, block| {
                let sink = typed_local(ssa, 60, 0, 0, MockType::U32);
                plant(
                    ssa,
                    block,
                    SsaOp::FloatToInt {
                        dest: sink,
                        operand: *some_or_abort(values.first()),
                        target: MockType::U32,
                        overflow_check: false,
                        unsigned: false,
                        source: None,
                    },
                );
            },
            reported: |error, _, block| matches!(error, VerifierError::InvalidConversion { block: b, .. } if *b == block),
        },
        Defect {
            name: "an integer conversion to a float",
            level: VerifyLevel::Standard,
            sites: &SITES,
            plant: |ssa, values, block| {
                let sink = typed_local(ssa, 60, 0, 0, MockType::F64);
                plant(
                    ssa,
                    block,
                    SsaOp::IntConv {
                        dest: sink,
                        operand: *some_or_abort(values.first()),
                        target: MockType::F64,
                        unsigned: true,
                        overflow_check: false,
                    },
                );
            },
            reported: |error, _, block| matches!(error, VerifierError::InvalidConversion { block: b, .. } if *b == block),
        },
        Defect {
            name: "a flag mask naming no defined flag",
            level: VerifyLevel::Standard,
            sites: &SITES,
            plant: |ssa, values, block| {
                let sink = scratch(ssa);
                plant(
                    ssa,
                    block,
                    SsaOp::ReadFlags {
                        dest: sink,
                        flags: *some_or_abort(values.first()),
                        mask: FlagsMask::from_bits(FlagsMask::CARRY.bits() | (1 << 15)),
                    },
                );
            },
            reported: |error, _, block| matches!(error, VerifierError::InvalidNativeOperation { block: b, .. } if *b == block),
        },
        Defect {
            name: "a use of a value a later block defines",
            level: VerifyLevel::Full,
            sites: &EARLY_SITES,
            plant: |ssa, values, block| {
                let sink = scratch(ssa);
                plant(
                    ssa,
                    block,
                    SsaOp::Copy {
                        dest: sink,
                        src: *some_or_abort(values.get(LAST)),
                    },
                );
            },
            reported: |error, values, block| {
                matches!(
                    error,
                    VerifierError::DominanceViolation { var, use_block, .. }
                        if *var == *some_or_abort(values.get(LAST)) && *use_block == block
                )
            },
        },
        Defect {
            name: "an instruction with no identity",
            level: VerifyLevel::Full,
            sites: &SITES,
            plant: |ssa, _, block| {
                let sink = scratch(ssa);
                plant(
                    ssa,
                    block,
                    SsaOp::Const {
                        dest: sink,
                        value: ConstValue::I32(0),
                    },
                );
            },
            reported: |error, _, block| matches!(error, VerifierError::UnassignedInstructionId { block: b, .. } if *b == block),
        },
    ]
}

const LEVELS: [VerifyLevel; 3] = [VerifyLevel::Quick, VerifyLevel::Standard, VerifyLevel::Full];

/// The control: with no defect planted the chain verifies at every level.
#[test]
fn the_unbroken_chain_verifies_at_every_level() {
    let (ssa, _) = chain();
    for level in LEVELS {
        assert_eq!(
            SsaVerifier::new(&ssa).verify(level),
            Vec::new(),
            "{level:?}"
        );
    }
}

/// Every defect is reported where it was planted, from its level up, wherever
/// in the function that is, and is not reported below its level.
#[test]
fn a_defect_is_reported_at_its_block_and_its_level_wherever_it_stands() {
    for defect in defects() {
        for &block in defect.sites {
            let (mut ssa, values) = chain();
            (defect.plant)(&mut ssa, &values, block);
            // Planted instructions do not disturb the bookkeeping the other
            // checks read, except where the defect is the bookkeeping.
            if defect.level != VerifyLevel::Standard || defect.name.contains("registers") {
                ssa.refresh_def_sites();
            }
            if !defect.name.contains("use list") {
                ssa.recompute_uses();
            }
            if defect.level != VerifyLevel::Full {
                ssa.assign_instr_ids();
            }
            for level in LEVELS {
                let errors = SsaVerifier::new(&ssa).verify(level);
                let seen = errors
                    .iter()
                    .any(|error| (defect.reported)(error, &values, block));
                assert_eq!(
                    seen,
                    level >= defect.level,
                    "{} in B{block} at {level:?}: {errors:?}",
                    defect.name
                );
            }
        }
    }
}

/// The report reads as if each check had made its own pass: a later check's
/// finding in an early block still follows an earlier check's finding in a
/// late one, and findings of one check keep the order they were found in.
#[test]
fn findings_are_reported_check_by_check() {
    let (mut ssa, _) = chain();
    // Block-structure finding early, index-bounds findings later: the
    // index-bounds check is listed first, so its findings come first.
    plant(&mut ssa, 1, SsaOp::Jump { target: 0 });
    for block in [6, 4] {
        replace_terminator(&mut ssa, block, SsaOp::Jump { target: 77 });
    }
    ssa.recompute_uses();
    ssa.refresh_def_sites();
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Quick);

    let dangling: Vec<usize> = errors
        .iter()
        .filter_map(|error| match error {
            VerifierError::DanglingBranchTarget { block, .. } => Some(*block),
            _ => None,
        })
        .collect();
    assert_eq!(
        dangling,
        vec![4, 6],
        "one check's findings keep block order"
    );

    let position = |wanted: fn(&VerifierError) -> bool| errors.iter().position(wanted);
    let last_dangling =
        position(|e| matches!(e, VerifierError::DanglingBranchTarget { block: 6, .. }));
    let not_last = position(|e| matches!(e, VerifierError::TerminatorNotLast { block: 1, .. }));
    assert!(last_dangling < not_last, "{errors:?}");
}

/// A predecessor an operand names twice is reported once, where the second
/// naming stands, so the findings follow operand order and not predecessor
/// order.
#[test]
fn repeated_phi_operands_are_reported_in_operand_order() {
    let mut ssa = diamond_phi_fixture();
    let phi = some_or_abort(some_or_abort(ssa.block_mut(3)).phi_mut(0));
    let left = *some_or_abort(phi.operands().first());
    let right = *some_or_abort(phi.operands().get(1));
    // Predecessors 2, 1, 2, 1: the second 2 stands before the second 1.
    *phi.operands_mut() = vec![right, left, right, left];
    ssa.recompute_uses();

    let duplicates: Vec<usize> = SsaVerifier::new(&ssa)
        .verify(VerifyLevel::Standard)
        .iter()
        .filter_map(|error| match error {
            VerifierError::DuplicatePhiOperand { pred, .. } => Some(*pred),
            _ => None,
        })
        .collect();
    assert_eq!(duplicates, vec![2, 1]);

    // The control: each predecessor named once is clean.
    let mut clean = diamond_phi_fixture();
    clean.recompute_uses();
    assert!(
        SsaVerifier::new(&clean)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
}

/// Phi checks reach the last join of a long function, and a missing, an extra
/// and a repeated operand there are each reported.
#[test]
fn phi_operands_are_checked_at_the_last_join() {
    let diamonds: usize = 6;
    let join = diamonds.saturating_mul(3);
    let build = || {
        let mut ssa = diamond_chain_fixture(diamonds);
        ssa.recompute_uses();
        ssa
    };
    assert!(
        SsaVerifier::new(&build())
            .verify(VerifyLevel::Full)
            .is_empty()
    );

    let mut missing = build();
    let removed = some_or_abort(some_or_abort(missing.block_mut(join)).phi_mut(0))
        .operands_mut()
        .pop();
    assert!(removed.is_some());
    missing.recompute_uses();
    assert!(
        SsaVerifier::new(&missing)
            .verify(VerifyLevel::Standard)
            .iter()
            .any(|e| matches!(e, VerifierError::MissingPhiOperand { block, .. } if *block == join))
    );

    let mut extra = build();
    let any_value = some_or_abort(
        some_or_abort(some_or_abort(extra.block(join)).phi(0))
            .operands()
            .first(),
    )
    .value();
    some_or_abort(some_or_abort(extra.block_mut(join)).phi_mut(0))
        .operands_mut()
        .push(PhiOperand::new(any_value, 0));
    extra.recompute_uses();
    assert!(
        SsaVerifier::new(&extra)
            .verify(VerifyLevel::Standard)
            .iter()
            .any(|e| matches!(e, VerifierError::ExtraPhiOperand { block, extra_pred: 0, .. } if *block == join))
    );
}

/// A use list is compared as a multiset: an instruction that reads a value
/// twice is two uses, and a list that records it once is stale.
#[test]
fn a_value_read_twice_by_one_instruction_is_two_uses() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
    let value = local(&mut ssa, 0, 0, 0);
    let sum = local(&mut ssa, 1, 0, 1);
    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(2),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: value,
        right: value,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(sum))));
    ssa.add_block(block);
    ssa.recompute_uses();
    assert_eq!(some_or_abort(ssa.variable(value)).uses().len(), 2);
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );

    some_or_abort(ssa.variable_mut(value)).retain_uses({
        let mut kept = false;
        move |_| !std::mem::replace(&mut kept, true)
    });
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .iter()
            .any(|e| matches!(
                e,
                VerifierError::StaleUseList { var, recorded: 1, actual: 2 } if *var == value
            ))
    );
}

/// A float raise is of a computation and sets only flags its layout keeps: one
/// whose operation defines another variable, or whose flags have no bit, is
/// reported, and the raise the constructor builds is not.
#[test]
fn verifier_checks_a_float_raise() {
    let build = |raise: &dyn Fn(SsaVarId, SsaVarId, SsaVarId) -> SsaOp<MockTarget>| {
        let mut ssa = SsaFunction::new(0, 3);
        let (left, right, flags) = (
            local(&mut ssa, 0, 0, 0),
            local(&mut ssa, 1, 0, 1),
            local(&mut ssa, 2, 0, 2),
        );
        let mut block = SsaBlock::new(0);
        for var in [left, right] {
            block.add_instruction(instr(SsaOp::Const {
                dest: var,
                value: ConstValue::Float(FloatBits::from(1.5f64)),
            }));
        }
        block.add_instruction(instr(raise(left, right, flags)));
        block.add_instruction(instr(SsaOp::returns(Some(flags))));
        ssa.add_block(block);
        ssa.recompute_uses();
        SsaVerifier::new(&ssa).verify(VerifyLevel::Standard)
    };
    let add = |dest, left, right| SsaOp::Add {
        dest,
        left,
        right,
        flags: None,
    };
    let rejected = |errors: &[VerifierError]| {
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidNativeOperation { .. }))
    };

    let errors = build(&|left, right, flags| {
        SsaOp::float_raise(
            flags,
            add(left, left, right),
            None,
            FloatExceptions::INEXACT.union(FloatExceptions::INVALID),
            FloatExceptionLayout::IEEE,
        )
    });
    assert!(!rejected(&errors), "a well-formed raise; got {errors:?}");

    let errors = build(&|left, right, flags| SsaOp::FloatRaise {
        dest: flags,
        op: Box::new(add(left, left, right)),
        lanes: None,
        raises: FloatExceptions::INEXACT,
        layout: FloatExceptionLayout::IEEE,
    });
    assert!(
        rejected(&errors),
        "an operation defining its operand; got {errors:?}"
    );

    let errors = build(&|left, right, flags| {
        SsaOp::float_raise(
            flags,
            add(left, left, right),
            None,
            FloatExceptions::INPUT_DENORMAL,
            FloatExceptionLayout::IEEE,
        )
    });
    assert!(
        rejected(&errors),
        "a flag the layout keeps nowhere; got {errors:?}"
    );
}

/// **An FPU control operation of the SSA moves a state image.** A save of the
/// extended state at the address it names first is accepted; a control word
/// load, which a host lifts as a move of the control register, is rejected as
/// an operation of its own, and so is an image move that names no address.
#[test]
fn verifier_accepts_only_a_state_image_move_as_an_fpu_control_operation() {
    let errors_of = |kind: FpuControlKind, addressed: bool| {
        let mut ssa = SsaFunction::new(0, 1);
        let address = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::FpuControl(Box::new(KindedVecData {
            kind,
            outputs: Vec::new(),
            inputs: if addressed { vec![address] } else { Vec::new() },
        }))));
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.recompute_uses();
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .into_iter()
            .filter(|error| matches!(error, VerifierError::InvalidNativeOperation { .. }))
            .count()
    };
    assert_eq!(errors_of(FpuControlKind::SaveExtended, true), 0);
    assert_eq!(errors_of(FpuControlKind::LoadControlWord, true), 1);
    assert_eq!(errors_of(FpuControlKind::SaveExtended, false), 1);
}

/// **The environment's rounding mode is read by an operation that rounds by
/// it.** A lane division of doubles stating no rounding may read the mode; an
/// integer addition, which rounds nothing, may not, and neither may a float
/// division stating its rounding.
#[test]
fn verifier_accepts_the_environments_mode_only_where_an_operation_rounds_by_it() {
    let errors_of = |kind: VectorBinaryKind, element: VectorElement| {
        let mut ssa = SsaFunction::new(0, 1);
        let entry = |ssa: &mut SsaFunction<MockTarget>, ty| {
            ssa.create_variable(VariableOrigin::EntryLiveIn, 0, DefSite::entry(), ty)
        };
        let left = entry(&mut ssa, MockType::F64);
        let right = entry(&mut ssa, MockType::F64);
        let mode = entry(&mut ssa, MockType::I32);
        let dest = typed_local(&mut ssa, 0, 0, 0, MockType::F64);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::VectorBinary {
            dest,
            left,
            right,
            kind,
            element,
            environment: Some(mode),
        }));
        block.add_instruction(instr(SsaOp::returns(Some(dest))));
        ssa.add_block(block);
        ssa.recompute_uses();
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .into_iter()
            .filter(|error| matches!(error, VerifierError::InvalidVectorOperation { .. }))
            .count()
    };
    let double = VectorElement::float(64).as_scalar();
    assert_eq!(errors_of(VectorBinaryKind::Div, double), 0);
    assert_eq!(
        errors_of(
            VectorBinaryKind::Add,
            VectorElement::integer(64, None).as_scalar()
        ),
        1
    );
    assert_eq!(
        errors_of(
            VectorBinaryKind::Div,
            double.rounded(FloatRounding::TowardZero)
        ),
        1
    );
}
