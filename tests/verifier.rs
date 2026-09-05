//! Focused SSA verifier failure cases.

use analyssa::{
    analysis::{SsaVerifier, VerifierError, VerifyLevel},
    ir::{
        block::SsaBlock,
        exception::{BlockRange, ClausePart, ExceptionTableError, SsaExceptionHandler},
        function::SsaFunction,
        instruction::SsaInstruction,
        ops::{
            AtomicAccessWidth, AtomicOrdering, AtomicRmwOp, ControlEffect, FlagAdjustKind,
            FlagsMask, KindedVecData, MemoryAccessSemantics, MemoryEffectLocation,
            NativeOpaqueData, SsaEffectKind, SsaEffects, SsaOp, TrapClass, VectorBinaryKind,
            VectorBitmaskKind, VectorCompareKind, VectorElement, VectorFaultMode,
            VectorMaskBinaryKind, VectorMaskMode, VectorReduceKind, VectorSegmentLayout,
        },
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    target::{Target, VectorDescriptor, VectorLaneKind, VectorShuffleLane, VectorShuffleMask},
    testing::{MockTarget, MockType},
};

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| std::process::abort())
}

fn result_or_abort<T>(result: analyssa::Result<T>) -> T {
    result.unwrap_or_else(|_| std::process::abort())
}

fn err_or_abort<T>(result: std::result::Result<T, Vec<VerifierError>>) -> Vec<VerifierError> {
    match result {
        Ok(_) => std::process::abort(),
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
    block.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
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
    join.add_instruction(instr(SsaOp::Return {
        value: Some(merged),
    }));
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
    exit.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(v1) }));
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
    disconnected.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(disconnected);

    let mut join = SsaBlock::new(3);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(left_value, 1));
    phi.add_operand(PhiOperand::new(stale_value, 2));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::Return {
        value: Some(merged),
    }));
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
    join.add_instruction(instr(SsaOp::Return {
        value: Some(branch_only),
    }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(v1) }));
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
    // Don't use `orphan` — it stays in the variables vec but is never defined in a block
    block.add_instruction(instr(SsaOp::Return { value: None }));
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
    block.add_instruction(instr(SsaOp::Return { value: None }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(b) }));
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
/// like an implicitly-defined argument — which is what previously let this pass
/// verification clean.
#[test]
fn verifier_rejects_a_use_whose_definition_was_removed() {
    let mut ssa = SsaFunction::new(0, 1);
    let defined = local(&mut ssa, 0, 0, 0);
    // A temporary beyond the declared local count, with an entry-shaped def site
    // and no defining instruction anywhere — exactly what repair leaves behind.
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
    block.add_instruction(instr(SsaOp::Return { value: None }));
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
/// deliberately a *distinct* origin from `Argument` — the lifter has not
/// recovered the calling convention, so it cannot honestly claim a signature
/// position — and distinct from `Phi`, which stays rejected so that a variable
/// whose definition was destroyed cannot masquerade as an entry value.
///
/// This test exists because the entry-defined origin set is a public contract,
/// not a verifier implementation detail: tightening it previously broke a
/// downstream lifter silently, with no failure in this suite.
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
    block.add_instruction(instr(SsaOp::Return { value: None }));
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
    block.add_instruction(instr(SsaOp::Return { value: None }));
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
/// straight from the IR — so `VerifyLevel::Full` aborted the process on one of
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
    b1.add_instruction(instr(SsaOp::Return {
        value: Some(merged),
    }));
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
    b0.add_instruction(instr(SsaOp::Return { value: Some(ghost) }));
    ssa.add_block(b0);
    ssa.recompute_uses();

    // Must not abort.
    let _ = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
}

/// One CFG edge carries one value, so two operands naming the same predecessor
/// have no defined meaning — and consumers disagree about them: `operand_from`
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
    // A second, conflicting operand for the *same* edge — what an edge redirect
    // produces when it retargets onto a predecessor that already has one.
    phi.add_operand(PhiOperand::new(right, 1));
    phi.add_operand(PhiOperand::new(right, 2));
    join.add_phi(phi);
    join.add_instruction(instr(SsaOp::Return {
        value: Some(merged),
    }));
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
    join.add_instruction(instr(SsaOp::Return {
        value: Some(merged),
    }));
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
/// instruction became a `Nop` — which is exactly what a pass leaves behind when
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
    block.add_instruction(instr(SsaOp::Return { value: Some(sum) }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(arg) }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(ghost) }));
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
    entry.add_instruction(instr(SsaOp::Return {
        value: Some(phi_var),
    }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(v1) }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
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
    join.add_instruction(instr(SsaOp::Return {
        value: Some(phi_var),
    }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(v0) }));
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
    block.add_instruction(instr(SsaOp::Return { value: Some(read) }));
    ssa.add_block(block);
    ssa.recompute_uses();

    result_or_abort(ssa.rebuild_ssa());

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
        value: ConstValue::Vector(
            vec![
                ConstValue::I32(1),
                ConstValue::I32(2),
                ConstValue::I32(3),
                ConstValue::I32(4),
            ]
            .into(),
        ),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::Vector(
            vec![
                ConstValue::I32(5),
                ConstValue::I32(6),
                ConstValue::I32(7),
                ConstValue::I32(8),
            ]
            .into(),
        ),
    }));
    block.add_instruction(instr(SsaOp::VectorBinary {
        dest,
        left,
        right,
        kind: VectorBinaryKind::Add,
        element: VectorElement::default(),
    }));
    ssa.add_block(block);
    ssa.recompute_uses();

    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
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
        value: ConstValue::Vector(vec![ConstValue::I32(1); 4].into()),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::Vector(vec![ConstValue::F64(1.0); 2].into()),
    }));
    block.add_instruction(instr(SsaOp::VectorBinary {
        dest,
        left,
        right,
        kind: VectorBinaryKind::Add,
        element: VectorElement::default(),
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
fn verifier_rejects_invalid_vector_shuffle_lane() {
    let mut ssa = SsaFunction::new(0, 2);
    let source = typed_local(&mut ssa, 0, 0, 0, MockType::V4I32);
    let dest = typed_local(&mut ssa, 1, 0, 1, MockType::V4I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: source,
        value: ConstValue::Vector(vec![ConstValue::I32(1); 4].into()),
    }));
    block.add_instruction(instr(SsaOp::VectorShuffle {
        dest,
        left: source,
        right: None,
        mask: VectorShuffleMask::new(vec![
            VectorShuffleLane::Left(0),
            VectorShuffleLane::Left(1),
            VectorShuffleLane::Left(2),
            VectorShuffleLane::Left(4),
        ]),
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
    }));
    block.add_instruction(instr(SsaOp::VectorCompare {
        dest: mask,
        left: sum,
        right,
        kind: VectorCompareKind::Eq,
        unsigned: false,
    }));
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
        value: ConstValue::Vector(vec![ConstValue::I32(1); 4].into()),
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
    block.add_instruction(instr(SsaOp::VectorMaskBinary {
        dest: mask_merged,
        left: mask_a,
        right: mask_b,
        kind: VectorMaskBinaryKind::And,
    }));
    block.add_instruction(instr(SsaOp::VectorMaskedLoad {
        dest: vector,
        addr,
        mask: mask_merged,
        passthrough: None,
        vector_type: MockType::V4I32,
        mode: VectorMaskMode::Zero,
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
    }));
    block.add_instruction(instr(SsaOp::Return {
        value: Some(reduced),
    }));
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
        value: ConstValue::Vector(vec![ConstValue::I32(0); 4].into()),
    }));
    block.add_instruction(instr(SsaOp::VectorMaskedLoad {
        dest: vector,
        addr,
        mask: bad_mask,
        passthrough: Some(passthrough),
        vector_type: MockType::V4I32,
        mode: VectorMaskMode::Merge,
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
    }));
    block.add_instruction(instr(SsaOp::VectorBroadcastLoad {
        dest: broadcast,
        addr: base,
        vector_type: MockType::V4I32,
    }));
    block.add_instruction(instr(SsaOp::VectorScatter {
        base,
        indices,
        value: gathered,
        mask,
        vector_type: MockType::V4I32,
    }));
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
    }));
    block.add_instruction(instr(SsaOp::VectorSegmentLoad {
        dests: vec![seg0, seg1],
        base,
        mask: Some(mask),
        vector_type: MockType::V4I32,
        segments: 2,
        layout: VectorSegmentLayout::Interleaved,
    }));
    block.add_instruction(instr(SsaOp::VectorSegmentStore {
        base,
        values: vec![seg0, seg1],
        mask: Some(mask),
        vector_type: MockType::V4I32,
        segments: 2,
        layout: VectorSegmentLayout::Interleaved,
    }));
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
        volatile: true,
    }));
    block.add_instruction(instr(SsaOp::Return { value: Some(old) }));
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
        volatile: false,
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

#[test]
fn verifier_rejects_illegal_native_atomic_ordering() {
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
        volatile: false,
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

#[test]
fn verifier_rejects_inconsistent_native_effect_summary() {
    let mut ssa = SsaFunction::new(0, 1);
    let effects = SsaEffects {
        kind: SsaEffectKind::Atomic,
        may_throw: false,
        memory: MemoryEffectLocation::Unknown,
        memory_semantics: MemoryAccessSemantics::Normal,
        volatile: false,
        ordering: None,
        trap: TrapClass::None,
        control: ControlEffect::None,
    };

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::NativeOpaque(Box::new(NativeOpaqueData {
        mnemonic: "bad_atomic_effect".to_string(),
        metadata: None,
        outputs: Vec::new(),
        inputs: Vec::new(),
        effects,
    }))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::InvalidNativeOperation { .. }))
    );
}

/// A catch or filter handler receives its exception object from the runtime:
/// nothing in the IR defines it, and the handler's leading `Pop` consumes it.
/// `SsaRebuilder::strip_undefined_pops` already carves out exactly this shape,
/// so the verifier must too — otherwise it rejects the IR the rebuilder is
/// built to produce, and every method with a handler fails as soon as a
/// CFG-modifying pass triggers a rebuild.
#[test]
fn verifier_accepts_runtime_supplied_exception_object() {
    let mut ssa = SsaFunction::new(0, 1);
    let guarded = local(&mut ssa, 0, 0, 0);
    // The exception object: a stack temporary with no defining instruction,
    // which is how a frontend models the runtime-pushed value.
    let exception = ssa.create_variable(VariableOrigin::Phi, 1, DefSite::entry(), MockType::I32);

    let mut try_block = SsaBlock::new(0);
    try_block.add_instruction(instr(SsaOp::Const {
        dest: guarded,
        value: ConstValue::I32(1),
    }));
    try_block.add_instruction(instr(SsaOp::Return { value: None }));
    ssa.add_block(try_block);

    let mut handler = SsaBlock::new(1);
    handler.add_instruction(instr(SsaOp::Pop { value: exception }));
    handler.add_instruction(instr(SsaOp::Return { value: None }));
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
    }]);
    ssa.recompute_uses();

    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e, VerifierError::UndefinedUse { var, .. } if *var == exception)),
        "the handler's runtime-supplied exception object must not be reported \
         as a dangling read; got {errors:?}"
    );

    // The carve-out is keyed on the handler's `Pop`: an undefined variable
    // that no handler-entry `Pop` consumes is still a dangling read, even in
    // the same block.
    let stray = ssa.create_variable(VariableOrigin::Phi, 1, DefSite::entry(), MockType::I32);
    if let Some(block) = ssa.block_mut(1) {
        block.add_instruction(instr(SsaOp::Throw { exception: stray }));
    }
    ssa.recompute_uses();
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, VerifierError::UndefinedUse { var, .. } if *var == stray)),
        "only the exception object the handler pops is runtime-supplied; got {errors:?}"
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
    block.add_instruction(instr(SsaOp::Return { value: Some(dest) }));
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

/// A flag adjustment carries exactly the outputs its kind declares, for every
/// kind.
///
/// The dangerous direction is a kind that declares a result and carries none:
/// the architectural write then has no SSA value modelling it, and an effect
/// summary derived from the kind alone calls the operation pure — which, with
/// no definitions, is what dead-code elimination deletes.
#[test]
fn flag_adjust_defining_a_register_has_exactly_one_output() {
    for (kind, outputs) in [
        (FlagAdjustKind::StoreStatusToReg, 0usize),
        (FlagAdjustKind::LoadStatusFromReg, 0),
        (FlagAdjustKind::ClearInterrupt, 1),
        (FlagAdjustKind::SetDirection, 1),
    ] {
        let mut ssa = SsaFunction::new(0, 1);
        let value = local(&mut ssa, 0, 0, 0);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(0),
        }));
        block.add_instruction(instr(SsaOp::FlagAdjust(Box::new(KindedVecData {
            kind,
            outputs: vec![value; outputs],
            inputs: Vec::new(),
        }))));
        block.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, VerifierError::InvalidNativeOperation { .. })),
            "{kind:?} with {outputs} output(s) must be reported; got {errors:?}"
        );
    }

    // The well-formed shapes pass: one output where the kind declares a result,
    // none where it declares `FlagAdjustResult::None`.
    for (kind, outputs) in [
        (FlagAdjustKind::StoreStatusToReg, 1usize),
        (FlagAdjustKind::ClearInterrupt, 0),
    ] {
        let mut ssa = SsaFunction::new(0, 1);
        let value = local(&mut ssa, 0, 0, 1);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Nop));
        block.add_instruction(instr(SsaOp::FlagAdjust(Box::new(KindedVecData {
            kind,
            outputs: vec![value; outputs],
            inputs: Vec::new(),
        }))));
        block.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e, VerifierError::InvalidNativeOperation { .. })),
            "{kind:?} with {outputs} output(s) is well formed; got {errors:?}"
        );
    }
}

/// A two-block function whose single clause is `handlers`.
fn function_with_clauses(
    handlers: Vec<SsaExceptionHandler<MockTarget>>,
) -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    for index in 0..2 {
        let mut block = SsaBlock::new(index);
        block.add_instruction(instr(SsaOp::Return { value: None }));
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
/// handler is a funclet, which is 28 of the 29 clauses on the corpus fixture.
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
