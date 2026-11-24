//! Artificial SSA pass pipeline tests.

use std::process;

use analyssa::{
    PointerSize,
    analysis::{
        Budget, Exhaustion, Machine, WorkBudget,
        context::AnalysisContext,
        taint::{TaintAnalysis, TaintConfig},
    },
    events::{EventKind, EventLog},
    ir::{
        CallData, Callee,
        block::SsaBlock,
        function::{SsaEditOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::{
            AccessQualifiers, AtomicAccessWidth, AtomicOrdering, FenceKind, FlagsMask,
            NativeKindedData, SsaOp, SysRegOp, SystemOpKind, VectorFaultMode, VectorMaskMode,
            VectorSegmentLayout,
        },
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::{
        AlgebraicSimplificationPass, BlockMergingPass, ControlFlowSimplificationPass,
        CopyPropagationPass, DeadCodeEliminationPass, GlobalValueNumberingPass, JumpThreadingPass,
        LicmPass, LoopCanonicalizationPass, OpaquePredicatePass, PredicateResult,
        ReassociationPass, StrengthReductionPass, ValueRangePropagationPass,
    },
    scheduling::{FunctionPass, PassEnv},
    testing::{MACHINE, MockTarget, MockType},
};

use crate::common::{
    assert_event, assert_has_op, assert_valid_full, op_at, run_pass_boundary,
    run_pass_repaired_boundary, terminator_at,
};

fn some_or_abort<T>(value: Option<T>) -> T {
    value.unwrap_or_else(|| process::abort())
}

fn local(ssa: &mut SsaFunction<MockTarget>, idx: u16, instr: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(0, instr),
        MockType::I32,
    )
}

fn local_at(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
    ssa.create_variable(
        VariableOrigin::Local(idx),
        0,
        DefSite::instruction(block, instr),
        MockType::I32,
    )
}

fn typed_local_at(
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

fn assert_valid_ssa(ssa: &SsaFunction<MockTarget>, label: &str) {
    assert_valid_full(ssa, label);
}

fn build_rewrite_fixture() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 11);
    let x = local(&mut ssa, 0, 0);
    let zero = local(&mut ssa, 1, 1);
    let scale = local(&mut ssa, 2, 2);
    let c2 = local(&mut ssa, 3, 3);
    let sum = local(&mut ssa, 4, 4);
    let duplicate_sum = local(&mut ssa, 5, 5);
    let reassoc_inner = local(&mut ssa, 6, 6);
    let reassoc_outer = local(&mut ssa, 7, 7);
    let shifted = local(&mut ssa, 8, 8);
    let copied = local(&mut ssa, 9, 9);
    let unused = local(&mut ssa, 10, 10);
    let branch_const = local(&mut ssa, 11, 11);

    let mut entry = SsaBlock::new(0);
    // An argument, so no pass folds the chain built on it and each later
    // pass is left the rewrite it is here to make.
    entry.add_instruction(instr(SsaOp::LoadArg {
        dest: x,
        arg_index: 0,
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: scale,
        value: ConstValue::I32(8),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: c2,
        value: ConstValue::I32(2),
    }));
    entry.add_instruction(instr(SsaOp::Add {
        dest: sum,
        left: x,
        right: zero,
        flags: None,
    }));
    entry.add_instruction(instr(SsaOp::Add {
        dest: duplicate_sum,
        left: zero,
        right: x,
        flags: None,
    }));
    entry.add_instruction(instr(SsaOp::Add {
        dest: reassoc_inner,
        left: duplicate_sum,
        right: c2,
        flags: None,
    }));
    entry.add_instruction(instr(SsaOp::Add {
        dest: reassoc_outer,
        left: reassoc_inner,
        right: scale,
        flags: None,
    }));
    entry.add_instruction(instr(SsaOp::Mul {
        dest: shifted,
        left: reassoc_outer,
        right: scale,
        flags: None,
    }));
    entry.add_instruction(instr(SsaOp::Copy {
        dest: copied,
        src: shifted,
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: unused,
        value: ConstValue::I32(99),
    }));
    entry.add_instruction(instr(SsaOp::Const {
        dest: branch_const,
        value: ConstValue::I32(1),
    }));
    entry.add_instruction(instr(SsaOp::Branch {
        condition: branch_const,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(entry);

    let mut true_block = SsaBlock::new(1);
    true_block.add_instruction(instr(SsaOp::returns(Some(copied))));
    ssa.add_block(true_block);

    let mut false_block = SsaBlock::new(2);
    false_block.add_instruction(instr(SsaOp::returns(Some(sum))));
    ssa.add_block(false_block);

    ssa.recompute_uses();
    ssa
}

fn build_trampoline_phi_fixture() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 3);
    let v0 = local_at(&mut ssa, 0, 0, 0);
    let cond = local_at(&mut ssa, 1, 0, 1);
    let v1 = local_at(&mut ssa, 2, 1, 0);
    let phi_var = phi_local(&mut ssa, 3, 3);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::LoadArg {
        dest: cond,
        arg_index: 0,
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 2,
        false_target: 1,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: v1,
        value: ConstValue::I32(2),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(b1);

    let mut trampoline = SsaBlock::new(2);
    trampoline.add_instruction(instr(SsaOp::Jump { target: 3 }));
    ssa.add_block(trampoline);

    let mut merge = SsaBlock::new(3);
    let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(3));
    phi.add_operand(PhiOperand::new(v0, 2));
    phi.add_operand(PhiOperand::new(v1, 1));
    merge.add_phi(phi);
    merge.add_instruction(instr(SsaOp::returns(Some(phi_var))));
    ssa.add_block(merge);

    ssa.recompute_uses();
    ssa
}

fn build_threading_phi_fixture() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 3);
    let true_cond = local(&mut ssa, 0, 0);
    let false_cond = local(&mut ssa, 1, 0);
    let merged_cond = phi_local(&mut ssa, 2, 2);

    let mut true_pred = SsaBlock::new(0);
    true_pred.add_instruction(instr(SsaOp::Const {
        dest: true_cond,
        value: ConstValue::I32(1),
    }));
    true_pred.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(true_pred);

    let mut false_pred = SsaBlock::new(1);
    false_pred.add_instruction(instr(SsaOp::Const {
        dest: false_cond,
        value: ConstValue::I32(0),
    }));
    false_pred.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(false_pred);

    let mut branch = SsaBlock::new(2);
    let mut phi = PhiNode::new(merged_cond, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(true_cond, 0));
    phi.add_operand(PhiOperand::new(false_cond, 1));
    branch.add_phi(phi);
    branch.add_instruction(instr(SsaOp::Branch {
        condition: merged_cond,
        true_target: 3,
        false_target: 4,
    }));
    ssa.add_block(branch);

    for block_idx in 3..=4 {
        let mut block = SsaBlock::new(block_idx);
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
    }

    ssa.recompute_uses();
    ssa
}

fn build_licm_invariant_fixture() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 5);
    let base = local_at(&mut ssa, 0, 0, 0);
    let one = local_at(&mut ssa, 1, 0, 1);
    let cond = local_at(&mut ssa, 2, 0, 2);
    let invariant = local_at(&mut ssa, 3, 2, 0);

    let mut preheader = SsaBlock::new(0);
    preheader.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(10),
    }));
    preheader.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    preheader.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    preheader.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(preheader);

    let mut header = SsaBlock::new(1);
    header.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(header);

    let mut body = SsaBlock::new(2);
    body.add_instruction(instr(SsaOp::Add {
        dest: invariant,
        left: base,
        right: one,
        flags: None,
    }));
    body.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(body);

    let mut exit = SsaBlock::new(3);
    exit.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(exit);

    ssa.recompute_uses();
    ssa
}

fn build_native_side_effect_fixture() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 12);
    let addr = typed_local_at(&mut ssa, 0, 0, 0, MockType::Ptr);
    let value = typed_local_at(&mut ssa, 1, 0, 1, MockType::I32);
    let mask = typed_local_at(&mut ssa, 2, 0, 2, MockType::Mask4);
    let vector = typed_local_at(&mut ssa, 3, 0, 3, MockType::V4I32);
    let native_out = typed_local_at(&mut ssa, 4, 0, 4, MockType::I32);
    let old = typed_local_at(&mut ssa, 5, 0, 5, MockType::I32);
    let faulting = typed_local_at(&mut ssa, 6, 0, 8, MockType::V4I32);
    let fault = typed_local_at(&mut ssa, 7, 0, 8, MockType::Mask4);
    let ret = typed_local_at(&mut ssa, 8, 0, 10, MockType::I32);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: addr,
        arg_index: 0,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: mask,
        arg_index: 1,
    }));
    block.add_instruction(instr(SsaOp::LoadArg {
        dest: vector,
        arg_index: 2,
    }));
    block.add_instruction(instr(SsaOp::SystemOp(Box::new(NativeKindedData {
        kind: SystemOpKind::ControlRegister(SysRegOp::ExchangeSystemRegister),
        mnemonic: "csrrw".to_string(),
        metadata: None,
        outputs: vec![native_out],
        inputs: vec![value],
    }))));
    block.add_instruction(instr(SsaOp::AtomicExchange {
        dest: old,
        addr,
        value,
        ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits32,
        access: AccessQualifiers::FLAT.made_volatile(),
    }));
    block.add_instruction(instr(SsaOp::Fence {
        kind: FenceKind::SeqCst,
    }));
    block.add_instruction(instr(SsaOp::from(CallData::new(
        None,
        Callee::Direct(0xCA11),
        vec![value],
    ))));
    block.add_instruction(instr(SsaOp::VectorFaultingLoad {
        dest: faulting,
        fault: Some(fault),
        addr,
        mask: Some(mask),
        passthrough: Some(vector),
        vector_type: MockType::V4I32,
        fault_mode: VectorFaultMode::FaultOnlyFirst,
        mask_mode: VectorMaskMode::Merge,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(instr(SsaOp::VectorSegmentStore {
        base: addr,
        values: vec![faulting, vector],
        mask: Some(mask),
        vector_type: MockType::V4I32,
        segments: 2,
        layout: VectorSegmentLayout::Interleaved,
        access: AccessQualifiers::FLAT,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: ret,
        value: ConstValue::I32(1),
    }));
    block.add_instruction(instr(SsaOp::returns(Some(ret))));
    ssa.add_block(block);
    ssa.recompute_uses();
    ssa
}

#[test]
fn artificial_pass_pipeline_rewrites_and_preserves_valid_ssa() {
    let mut ssa = build_rewrite_fixture();
    let log = EventLog::<MockTarget>::new();
    let method = 0xC0FFEEu32;

    assert!(
        AlgebraicSimplificationPass.run(
            &mut ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );
    ssa.recompute_uses();
    run_pass_boundary(&mut ssa, "gvn", |ssa| {
        GlobalValueNumberingPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    assert!(
        ReassociationPass.run(
            &mut ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );
    ssa.recompute_uses();
    assert!(
        StrengthReductionPass.run(
            &mut ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );
    ssa.recompute_uses();
    assert!(CopyPropagationPass.propagate(
        &mut ssa,
        &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        |_, _| {}
    ));
    ssa.recompute_uses();
    assert!(
        ValueRangePropagationPass.run(
            &mut ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );
    ssa.recompute_uses();
    assert!(
        DeadCodeEliminationPass.run(
            &mut ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );
    ssa.recompute_uses();

    assert_event(&log, EventKind::ConstantFolded);
    assert_event(&log, EventKind::StrengthReduced);
    assert_event(&log, EventKind::CopyPropagated);
    assert_event(&log, EventKind::BranchSimplified);
    assert_event(&log, EventKind::InstructionRemoved);

    let entry_term = terminator_at(&ssa, 0);
    assert!(matches!(entry_term, SsaOp::Jump { target: 1 }));
    assert_has_op(&ssa, "strength-reduced shift", |op| {
        matches!(op, SsaOp::Shl { .. })
    });

    assert_valid_ssa(&ssa, "rewrite pipeline");
}

#[test]
fn artificial_rewrite_passes_preserve_valid_ssa_after_each_boundary() {
    let mut ssa = build_rewrite_fixture();
    let log = EventLog::<MockTarget>::new();
    let method = 0xA11CEu32;

    run_pass_boundary(&mut ssa, "algebraic", |ssa| {
        AlgebraicSimplificationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "gvn", |ssa| {
        GlobalValueNumberingPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "reassociate", |ssa| {
        ReassociationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "strength", |ssa| {
        StrengthReductionPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "copying", |ssa| {
        CopyPropagationPass.propagate(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            |_, _| {},
        )
    });
    run_pass_boundary(&mut ssa, "ranges", |ssa| {
        ValueRangePropagationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "predicates", |ssa| {
        OpaquePredicatePass::new().run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "deadcode", |ssa| {
        DeadCodeEliminationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
}

#[test]
fn native_side_effect_passes_preserve_barriers_after_each_boundary() {
    let mut ssa = build_native_side_effect_fixture();
    let log = EventLog::<MockTarget>::new();
    let method = 0xA11CE5u32;

    run_pass_boundary(&mut ssa, "native algebraic", |ssa| {
        AlgebraicSimplificationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "native gvn", |ssa| {
        GlobalValueNumberingPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "native reassociate", |ssa| {
        ReassociationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "native strength", |ssa| {
        StrengthReductionPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "native copying", |ssa| {
        CopyPropagationPass.propagate(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            |_, _| {},
        )
    });
    run_pass_boundary(&mut ssa, "native ranges", |ssa| {
        ValueRangePropagationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "native predicates", |ssa| {
        OpaquePredicatePass::new().run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "native deadcode", |ssa| {
        DeadCodeEliminationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert!(
        ssa.iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::SystemOp(_)))
    );
    assert!(
        ssa.iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::AtomicExchange { .. }))
    );
    assert!(
        ssa.iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::Fence { .. }))
    );
    assert!(
        ssa.iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::Call(_)))
    );
    assert!(
        ssa.iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::VectorFaultingLoad { .. }))
    );
    assert!(
        ssa.iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::VectorSegmentStore { .. }))
    );
}

#[test]
fn artificial_cfg_passes_preserve_phi_edges_after_each_boundary() {
    let log = EventLog::<MockTarget>::new();
    let method = 0xCF6u32;

    let mut blockmerge_ssa = build_trampoline_phi_fixture();
    run_pass_boundary(&mut blockmerge_ssa, "blockmerge", |ssa| {
        BlockMergingPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    let mut controlflow_ssa = build_rewrite_fixture();
    run_pass_boundary(&mut controlflow_ssa, "controlflow", |ssa| {
        ControlFlowSimplificationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    let mut threading_ssa = build_threading_phi_fixture();
    run_pass_boundary(&mut threading_ssa, "threading", |ssa| {
        JumpThreadingPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
}

#[test]
fn artificial_loop_passes_preserve_valid_ssa_after_each_boundary() {
    let log = EventLog::<MockTarget>::new();
    let method = 0x1009u32;

    let mut loopcanon_ssa = build_loop_without_preheader();
    run_pass_boundary(&mut loopcanon_ssa, "loopcanon", |ssa| {
        LoopCanonicalizationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    let mut licm_ssa = build_licm_invariant_fixture();
    run_pass_boundary(&mut licm_ssa, "licm", |ssa| {
        LicmPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
}

#[test]
fn gvn_canonicalizes_commutative_duplicate_into_original_value() {
    let mut ssa = SsaFunction::new(0, 6);
    let a = local(&mut ssa, 0, 0);
    let b = local(&mut ssa, 1, 1);
    let first = local(&mut ssa, 2, 2);
    let duplicate = local(&mut ssa, 3, 3);
    let out = local(&mut ssa, 4, 4);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(4),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: first,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: duplicate,
        left: b,
        right: a,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Copy {
        dest: out,
        src: duplicate,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(out))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let log = EventLog::<MockTarget>::new();
    assert!(
        GlobalValueNumberingPass.run(
            &mut ssa,
            &PassEnv::new(&7u32, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );
    ssa.recompute_uses();

    // The commutative duplicate `b + a` is recognized as equal to `first`
    // (`a + b`): its uses are rewired to `first`, and the now-dead duplicate is
    // eliminated (boundary repair strips the resulting Nop, compacting the
    // block). So exactly one `Add` survives and the `Copy` reads `first`.
    let ops: Vec<_> = some_or_abort(ssa.block(0))
        .instructions()
        .iter()
        .map(SsaInstruction::op)
        .collect();
    let add_count = ops
        .iter()
        .filter(|op| matches!(op, SsaOp::Add { .. }))
        .count();
    assert_eq!(
        add_count, 1,
        "the commutative duplicate add should be eliminated"
    );
    assert!(
        ops.iter()
            .any(|op| matches!(op, SsaOp::Copy { src, .. } if *src == first)),
        "the copy should read the canonical `first` value"
    );
}

#[test]
fn jump_threading_uses_incoming_phi_values_to_redirect_predecessors() {
    let mut ssa = SsaFunction::new(0, 3);
    let true_cond = local(&mut ssa, 0, 0);
    let false_cond = local(&mut ssa, 1, 0);
    let merged_cond =
        ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

    let mut true_pred = SsaBlock::new(0);
    true_pred.add_instruction(instr(SsaOp::Const {
        dest: true_cond,
        value: ConstValue::I32(1),
    }));
    true_pred.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(true_pred);

    let mut false_pred = SsaBlock::new(1);
    false_pred.add_instruction(instr(SsaOp::Const {
        dest: false_cond,
        value: ConstValue::I32(0),
    }));
    false_pred.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(false_pred);

    let mut branch = SsaBlock::new(2);
    let mut phi = PhiNode::new(merged_cond, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(true_cond, 0));
    phi.add_operand(PhiOperand::new(false_cond, 1));
    branch.add_phi(phi);
    branch.add_instruction(instr(SsaOp::Branch {
        condition: merged_cond,
        true_target: 3,
        false_target: 4,
    }));
    ssa.add_block(branch);

    for block_idx in 3..=4 {
        let mut block = SsaBlock::new(block_idx);
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
    }
    ssa.recompute_uses();

    let log = EventLog::<MockTarget>::new();
    assert!(
        JumpThreadingPass.run(
            &mut ssa,
            &PassEnv::new(&0x1234u32, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );

    assert!(matches!(terminator_at(&ssa, 0), SsaOp::Jump { target: 3 }));
    if let Some(block) = ssa.block(1)
        && let Some(terminator) = block.control_terminator()
    {
        assert!(matches!(terminator, SsaOp::Jump { target: 4 }));
    }
    assert_event(&log, EventKind::ControlFlowRestructured);
}

#[test]
fn licm_hoists_loop_invariant_expression_to_preheader() {
    let mut ssa = SsaFunction::new(0, 5);
    let base = local(&mut ssa, 0, 0);
    let one = local(&mut ssa, 1, 1);
    let cond = local(&mut ssa, 2, 2);
    let invariant = ssa.create_variable(
        VariableOrigin::Local(3),
        0,
        DefSite::instruction(2, 0),
        MockType::I32,
    );

    let mut preheader = SsaBlock::new(0);
    preheader.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(10),
    }));
    preheader.add_instruction(instr(SsaOp::Const {
        dest: one,
        value: ConstValue::I32(1),
    }));
    preheader.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    preheader.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(preheader);

    let mut header = SsaBlock::new(1);
    header.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 2,
        false_target: 3,
    }));
    ssa.add_block(header);

    let mut body = SsaBlock::new(2);
    body.add_instruction(instr(SsaOp::Add {
        dest: invariant,
        left: base,
        right: one,
        flags: None,
    }));
    body.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(body);

    let mut exit = SsaBlock::new(3);
    exit.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(exit);
    ssa.recompute_uses();

    let log = EventLog::<MockTarget>::new();
    assert!(
        LicmPass.run(
            &mut ssa,
            &PassEnv::new(&0x5678u32, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );

    let preheader_ops: Vec<_> = some_or_abort(ssa.block(0))
        .instructions()
        .iter()
        .map(SsaInstruction::op)
        .collect();
    assert!(
        preheader_ops
            .get(3)
            .is_some_and(|op| matches!(op, SsaOp::Add { dest, .. } if *dest == invariant))
    );
    assert!(
        some_or_abort(ssa.block(2))
            .instructions()
            .iter()
            .all(|instr| !matches!(instr.op(), SsaOp::Add { dest, .. } if *dest == invariant))
    );
    assert_valid_ssa(&ssa, "licm hoist");
    assert!(log.has(EventKind::InstructionRemoved));
}

// Loop Canonicalization

/// Builds: preheader (B0) → header (B1: phi for i, condition, then body and
/// latch). This loop already has a single preheader and single latch, so it
/// should already be canonical.
fn build_canonical_loop() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 5);

    let init = local_at(&mut ssa, 0, 0, 0);
    let one = local_at(&mut ssa, 1, 0, 1);
    let limit = local_at(&mut ssa, 2, 0, 2);
    let i_phi = phi_local(&mut ssa, 3, 1);
    let cond = local_at(&mut ssa, 4, 1, 0);
    let next = local_at(&mut ssa, 5, 2, 0);

    // Preheader B0
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
        value: ConstValue::I32(100),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    // Header B1
    let mut b1 = SsaBlock::new(1);
    let mut phi = PhiNode::new(i_phi, VariableOrigin::Local(3));
    phi.add_operand(PhiOperand::new(init, 0));
    phi.add_operand(PhiOperand::new(next, 2));
    b1.add_phi(phi);
    b1.add_instruction(instr(SsaOp::Clt {
        dest: cond,
        left: i_phi,
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

    // Body/latch B2
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Add {
        dest: next,
        left: i_phi,
        right: one,
        flags: None,
    }));
    b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b2);

    // Exit B3
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::returns(Some(i_phi))));
    ssa.add_block(b3);

    ssa.recompute_uses();
    ssa
}

#[test]
fn loop_canonicalization_does_not_modify_already_canonical_loop() {
    let mut ssa = build_canonical_loop();
    let log: EventLog<MockTarget> = EventLog::new();
    let method = 0u32;

    let changed = LoopCanonicalizationPass.run(
        &mut ssa,
        &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    // Already canonical → no changes needed
    assert!(!changed, "should not modify already-canonical loop");

    assert_valid_ssa(&ssa, "already canonical loop");
}

/// **A rewriting loop out of rounds says so.** Allowed no round, each loop
/// reports its budget exhausted - block merging twice, its trampoline and
/// coalescing loops counting their rounds apart - and allowed its default,
/// the same loops over the same function report nothing.
#[test]
fn a_rewriting_loop_out_of_rounds_reports_its_budget() {
    let run = |rounds: Option<usize>| {
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let budget = WorkBudget::DEFAULT;
        let limit = |kind: Budget| rounds.unwrap_or_else(|| budget.limit(kind));
        let mut ssa = build_rewrite_fixture();
        assert!(
            DeadCodeEliminationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)).with_budget(
                        WorkBudget::DEFAULT
                            .with_limit(Budget::DeadCodeRounds, limit(Budget::DeadCodeRounds))
                    )
                )
                .is_ok()
        );
        CopyPropagationPass.propagate(
            &mut ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)).with_budget(
                WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, limit(Budget::CopyRounds)),
            ),
            |_, _| {},
        );
        assert!(
            ControlFlowSimplificationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)).with_budget(
                        WorkBudget::DEFAULT.with_limit(
                            Budget::ControlFlowRounds,
                            limit(Budget::ControlFlowRounds)
                        )
                    )
                )
                .is_ok()
        );
        assert!(
            BlockMergingPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)).with_budget(
                        WorkBudget::DEFAULT
                            .with_limit(Budget::BlockMergeRounds, limit(Budget::BlockMergeRounds))
                    )
                )
                .is_ok()
        );
        let mut looping = build_loop_without_preheader();
        assert!(
            LoopCanonicalizationPass
                .run(
                    &mut looping,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)).with_budget(
                        WorkBudget::DEFAULT.with_limit(
                            Budget::CanonicalizationRounds,
                            limit(Budget::CanonicalizationRounds)
                        )
                    )
                )
                .is_ok()
        );
        log.filter_kind(EventKind::BudgetExhausted)
            .map(|event| event.message.clone())
            .collect::<Vec<_>>()
    };
    let starved = run(Some(0));
    assert_eq!(starved.len(), 6, "{starved:?}");
    for name in [
        "dead-code elimination rounds",
        "copy propagation rounds",
        "control-flow simplification rounds",
        "loop canonicalization rounds",
    ] {
        assert_eq!(
            starved
                .iter()
                .filter(|message| message.starts_with(name))
                .count(),
            1,
            "{name}: {starved:?}"
        );
    }
    assert_eq!(
        starved
            .iter()
            .filter(|message| message.starts_with("block merging rounds"))
            .count(),
        2,
        "{starved:?}"
    );
    assert!(run(None).is_empty(), "{:?}", run(None));
}

/// **A loop that changed something on its last round has not settled.** The
/// loop without a preheader gets one in its first round: one round is spent
/// changing it and the budget says the fixpoint needed another, while two
/// rounds see the second change nothing.
#[test]
fn canonicalization_given_one_round_too_few_reports_the_round_it_needed() {
    let canonicalize = |limit: usize| {
        let log: EventLog<MockTarget> = EventLog::new();
        let mut ssa = build_loop_without_preheader();
        assert!(
            LoopCanonicalizationPass.run(
                &mut ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)).with_budget(
                    WorkBudget::DEFAULT.with_limit(Budget::CanonicalizationRounds, limit)
                )
            ) == Ok(true)
        );
        log.filter_kind(EventKind::BudgetExhausted)
            .map(|event| event.message.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        canonicalize(1),
        vec![
            "loop canonicalization rounds budget exhausted: needed at least 2, limit 1".to_string()
        ]
    );
    assert!(canonicalize(2).is_empty());
}

/// **Taint out of rounds says its answer is partial.** Allowed no round,
/// propagation records the exhaustion of its budget; allowed the default, it
/// settles and records none.
#[test]
fn taint_out_of_rounds_records_its_exhaustion() {
    let ssa = build_rewrite_fixture();
    let propagate = |budget: WorkBudget| {
        let mut taint = TaintAnalysis::new(TaintConfig::forward_only(), &budget);
        taint.add_tainted_var(SsaVarId::from_index(0));
        taint.propagate(&ssa);
        taint.stats().exhausted
    };
    assert_eq!(
        propagate(WorkBudget::DEFAULT.with_limit(Budget::TaintRounds, 0)),
        Some(Exhaustion {
            budget: Budget::TaintRounds,
            limit: 0,
            demand: 1,
        })
    );
    assert_eq!(propagate(WorkBudget::DEFAULT), None);
}

/// Builds a loop where the header has multiple non-loop predecessors (no
/// dedicated preheader). Canonicalization should insert a new preheader.
fn build_loop_without_preheader() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 6);

    let init_a = local_at(&mut ssa, 0, 0, 0);
    let init_b = local_at(&mut ssa, 1, 1, 0);
    let phi_i = phi_local(&mut ssa, 2, 2);
    let cond = local_at(&mut ssa, 3, 2, 0);
    let next = local_at(&mut ssa, 4, 3, 0);
    let entry_cond = local_at(&mut ssa, 5, 0, 1);

    // Entry (B0) - defines `init_a` and branches to the two entry paths so that
    // both reach the header directly (B0) and via B1. This gives the header two
    // distinct non-loop predecessors with no dedicated preheader.
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: init_a,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: entry_cond,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: entry_cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    // Entry path B (B1) - also jumps to header (no single preheader)
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Const {
        dest: init_b,
        value: ConstValue::I32(10),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    // Header (B2) - has two non-loop predecessors
    let mut b2 = SsaBlock::new(2);
    let mut phi = PhiNode::new(phi_i, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(init_a, 0));
    phi.add_operand(PhiOperand::new(init_b, 1));
    phi.add_operand(PhiOperand::new(next, 3));
    b2.add_phi(phi);
    // Compare against `init_a` (defined in B0, which dominates the header) so
    // the condition is valid on every entry path. `init_b` is defined only on
    // the B1 path and is used solely as that edge's phi operand.
    b2.add_instruction(instr(SsaOp::Clt {
        dest: cond,
        left: phi_i,
        right: init_a,
        unsigned: false,
        signaling: false,
    }));
    b2.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 3,
        false_target: 4,
    }));
    ssa.add_block(b2);

    // Latch (B3)
    let mut b3 = SsaBlock::new(3);
    b3.add_instruction(instr(SsaOp::Add {
        dest: next,
        left: phi_i,
        right: init_a,
        flags: None,
    }));
    b3.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b3);

    // Exit (B4)
    let mut b4 = SsaBlock::new(4);
    b4.add_instruction(instr(SsaOp::returns(Some(phi_i))));
    ssa.add_block(b4);

    ssa.recompute_uses();
    ssa
}

#[test]
fn loop_canonicalization_inserts_preheader_for_multi_predecessor_header() {
    let mut ssa = build_loop_without_preheader();
    let log: EventLog<MockTarget> = EventLog::new();
    let method = 1u32;

    let original_blocks = ssa.block_count();
    let changed = LoopCanonicalizationPass.run(
        &mut ssa,
        &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(changed, "should create a preheader");

    // After canonicalization, verify the IR is still well-formed
    // It should have at least one more block (the new preheader)
    assert!(
        ssa.block_count() > original_blocks,
        "block count should increase after preheader insertion"
    );

    assert_valid_ssa(&ssa, "loop preheader insertion");
}

#[test]
fn loop_canonicalization_idempotent() {
    let mut ssa = build_loop_without_preheader();
    let log: EventLog<MockTarget> = EventLog::new();
    let method = 2u32;

    // First run inserts preheader.
    run_pass_boundary(&mut ssa, "first loopcanon", |ssa| {
        LoopCanonicalizationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    let after_first = ssa.block_count();

    // Second run should make no changes
    let changed = LoopCanonicalizationPass.run(
        &mut ssa,
        &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(!changed, "second run should be a no-op");
    assert_eq!(ssa.block_count(), after_first);
}

#[test]
fn loop_canonicalization_on_empty_function_is_noop() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
    let log: EventLog<MockTarget> = EventLog::new();
    let changed = LoopCanonicalizationPass.run(
        &mut ssa,
        &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(!changed);
}

// Opaque Predicates

#[test]
fn predicates_detects_self_equality_always_true() {
    let mut ssa = SsaFunction::new(0, 3);
    let v0 = local_at(&mut ssa, 0, 0, 0);
    let v1 = local_at(&mut ssa, 1, 0, 1);
    let _v_bool = local_at(&mut ssa, 2, 0, 2);

    // Blocks are added in index order: a block's position in the list is what
    // every terminator target and definition site names.
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(42),
    }));
    b0.add_instruction(instr(SsaOp::Ceq {
        dest: v1,
        left: v0,
        right: v0,
        signaling: false,
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: v1,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    // B1 and B2 are the branch targets.
    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    let method = 3u32;
    let changed = OpaquePredicatePass::new().run(
        &mut ssa,
        &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
    ) == Ok(true);

    if changed {
        // After predicate simplification, the branch should be simplified
        assert!(log.has(EventKind::BranchSimplified));
    } else {
        // Even if the pass didn't fire, the function should still be valid
        assert_valid_ssa(&ssa, "self equality predicate no-op");
    }
}

#[test]
fn predicates_detects_xor_self_is_zero() {
    let mut ssa = SsaFunction::new(0, 3);
    let v0 = local_at(&mut ssa, 0, 0, 0);
    let v_xor = local_at(&mut ssa, 1, 0, 1);
    let v_zero = local_at(&mut ssa, 2, 0, 2);

    // Blocks are added in index order: a block's position in the list is what
    // every terminator target and definition site names.
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(7),
    }));
    b0.add_instruction(instr(SsaOp::Xor {
        dest: v_xor,
        left: v0,
        right: v0,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::Ceq {
        dest: v_zero,
        left: v_xor,
        right: SsaVarId::from_index(0),
        signaling: false,
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: v_zero,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b1);
    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    let changed = run_pass_boundary(&mut ssa, "xor predicate", |ssa| {
        OpaquePredicatePass::new().run(
            ssa,
            &PassEnv::new(&4u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert!(changed, "xor-self predicate should be simplified");
    assert_event(&log, EventKind::BranchSimplified);
}

/// A signed `(x - y) < 0` tests the sign of a difference that wraps: with
/// `x = i32::MIN + 5` and `y = 1 << 20` the difference is positive while `x <
/// y`, so the comparison stays on the difference.
#[test]
fn predicates_keep_an_order_of_a_wrapping_difference() {
    let mut ssa = SsaFunction::new(0, 5);
    let left = local_at(&mut ssa, 0, 0, 0);
    let right = local_at(&mut ssa, 1, 0, 1);
    let zero = local_at(&mut ssa, 2, 0, 2);
    let diff = local_at(&mut ssa, 3, 0, 3);
    let cmp = local_at(&mut ssa, 4, 0, 4);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(i32::MIN + 5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(1 << 20),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    block.add_instruction(instr(SsaOp::Sub {
        dest: diff,
        left,
        right,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Clt {
        dest: cmp,
        left: diff,
        right: zero,
        unsigned: false,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(cmp))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    assert!(
        OpaquePredicatePass::new()
            .run(
                &mut ssa,
                &PassEnv::new(&5u32, &log, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );

    let cmp_op = op_at(&ssa, 0, 4);
    assert!(
        matches!(
            cmp_op,
            SsaOp::Clt {
                dest,
                left: actual_left,
                right: actual_right,
                unsigned: false,
                ..
            } if *dest == cmp && *actual_left == diff && *actual_right == zero
        ),
        "{cmp_op:?}"
    );
    assert_valid_ssa(&ssa, "wrapping difference order");
}

#[test]
fn predicates_fold_nested_compare_equal_false_branch() {
    let mut ssa = SsaFunction::new(0, 5);
    let value = local_at(&mut ssa, 0, 0, 0);
    let inner = local_at(&mut ssa, 1, 0, 1);
    let false_const = local_at(&mut ssa, 2, 0, 2);
    let inverted = local_at(&mut ssa, 3, 0, 3);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(42),
    }));
    block.add_instruction(instr(SsaOp::Ceq {
        dest: inner,
        left: value,
        right: value,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: false_const,
        value: ConstValue::False,
    }));
    block.add_instruction(instr(SsaOp::Ceq {
        dest: inverted,
        left: inner,
        right: false_const,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::Branch {
        condition: inverted,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(block);

    let mut true_block = SsaBlock::new(1);
    true_block.add_instruction(instr(SsaOp::returns(Some(value))));
    ssa.add_block(true_block);

    let mut false_block = SsaBlock::new(2);
    false_block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(false_block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    assert!(
        OpaquePredicatePass::new().run(
            &mut ssa,
            &PassEnv::new(&6u32, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );

    assert!(matches!(terminator_at(&ssa, 0), SsaOp::Jump { target: 2 }));
    assert!(log.has(EventKind::BranchSimplified));
    assert_valid_ssa(&ssa, "nested compare inversion");
}

#[test]
fn predicates_preserve_unknown_flag_read_condition() {
    let mut ssa = SsaFunction::new(0, 6);
    let left = local_at(&mut ssa, 0, 0, 0);
    let right = local_at(&mut ssa, 1, 0, 1);
    let value = local_at(&mut ssa, 2, 0, 2);
    let flags = local_at(&mut ssa, 3, 0, 2);
    let zero_flag = local_at(&mut ssa, 4, 0, 3);
    let false_const = local_at(&mut ssa, 5, 0, 4);
    let inverted = local_at(&mut ssa, 6, 0, 5);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(20),
    }));
    block.add_instruction(instr(SsaOp::Sub {
        dest: value,
        left,
        right,
        flags: Some(flags),
    }));
    block.add_instruction(instr(SsaOp::ReadFlags {
        dest: zero_flag,
        flags,
        mask: FlagsMask::ZERO,
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: false_const,
        value: ConstValue::False,
    }));
    block.add_instruction(instr(SsaOp::Ceq {
        dest: inverted,
        left: zero_flag,
        right: false_const,
        signaling: false,
    }));
    block.add_instruction(instr(SsaOp::Branch {
        condition: inverted,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(block);

    let mut true_block = SsaBlock::new(1);
    true_block.add_instruction(instr(SsaOp::returns(Some(value))));
    ssa.add_block(true_block);

    let mut false_block = SsaBlock::new(2);
    false_block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(false_block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    run_pass_boundary(&mut ssa, "unknown flag predicate", |ssa| {
        OpaquePredicatePass::new().run(
            ssa,
            &PassEnv::new(&7u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert!(matches!(
        terminator_at(&ssa, 0),
        SsaOp::Branch { condition, .. } if *condition == inverted
    ));
    assert!(ssa.iter_instructions().any(
        |(_, _, instr)| matches!(instr.op(), SsaOp::ReadFlags { flags: f, .. } if *f == flags)
    ));
    assert_valid_ssa(&ssa, "unknown native flag condition");
}

#[test]
fn predicate_result_always_true_always_false() {
    assert_eq!(PredicateResult::AlwaysTrue.as_bool(), Some(true));
    assert_eq!(PredicateResult::AlwaysFalse.as_bool(), Some(false));
    assert_eq!(PredicateResult::Unknown.as_bool(), None);

    assert_eq!(
        PredicateResult::AlwaysTrue.negate(),
        PredicateResult::AlwaysFalse
    );
    assert_eq!(
        PredicateResult::AlwaysFalse.negate(),
        PredicateResult::AlwaysTrue
    );
    assert_eq!(PredicateResult::Unknown.negate(), PredicateResult::Unknown);
    assert_eq!(PredicateResult::AlwaysTrue, PredicateResult::AlwaysTrue);
    assert_ne!(PredicateResult::AlwaysTrue, PredicateResult::AlwaysFalse);
}

// Control-flow simplification edge cases

#[test]
fn control_flow_removes_dead_tail_after_return() {
    let mut ssa = SsaFunction::new(0, 1);
    let v0 = local_at(&mut ssa, 0, 0, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(v0))));
    // Add dead instruction after terminator
    b0.add_instruction(instr(SsaOp::Const {
        dest: v0,
        value: ConstValue::I32(999),
    }));
    ssa.add_block(b0);

    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    let changed = ControlFlowSimplificationPass.run(
        &mut ssa,
        &PassEnv::new(&5u32, &log, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    ssa.recompute_uses();
    assert_valid_ssa(&ssa, "dead-tail controlflow");
    assert!(changed, "controlflow should remove the dead tail");
}

// Copy propagation edge cases: chain elimination

#[test]
fn copy_propagation_collapses_three_element_chain() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local_at(&mut ssa, 0, 0, 0);
    let b = local_at(&mut ssa, 1, 0, 1);
    let c = local_at(&mut ssa, 2, 0, 2);
    let out = local_at(&mut ssa, 3, 0, 3);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
    block.add_instruction(instr(SsaOp::Copy { dest: c, src: b }));
    block.add_instruction(instr(SsaOp::Copy { dest: out, src: c }));
    block.add_instruction(instr(SsaOp::returns(Some(out))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    run_pass_boundary(&mut ssa, "copy chain propagation", |ssa| {
        CopyPropagationPass.propagate(
            ssa,
            &PassEnv::new(&6u32, &log, AnalysisContext::new(MACHINE)),
            |_, _| {},
        )
    });
    // Copy propagation repairs its own nopped definitions and should preserve
    // the standard verifier invariants without waiting for a later DCE pass.
    assert_valid_ssa(&ssa, "copy chain propagation");
}

// Dead code elimination: inter-block deadness

#[test]
fn dead_code_elimination_removes_inter_block_dead_variable() {
    let mut ssa = SsaFunction::new(0, 3);
    let cond = local_at(&mut ssa, 0, 0, 0);
    let dead_on_left = local_at(&mut ssa, 1, 1, 0);
    let live_on_right = local_at(&mut ssa, 2, 2, 0);

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
        dest: dead_on_left,
        value: ConstValue::I32(42),
    }));
    b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::Const {
        dest: live_on_right,
        value: ConstValue::I32(99),
    }));
    b2.add_instruction(instr(SsaOp::returns(Some(live_on_right))));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    let changed = run_pass_boundary(&mut ssa, "inter-block dce", |ssa| {
        DeadCodeEliminationPass.run(
            ssa,
            &PassEnv::new(&7u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    assert!(
        changed,
        "DCE should remove the dead branch-local definition"
    );
}

#[test]
fn dead_code_elimination_keeps_instruction_when_secondary_def_is_live() {
    let mut ssa = SsaFunction::new(0, 5);
    let left = local_at(&mut ssa, 0, 0, 0);
    let right = local_at(&mut ssa, 1, 0, 1);
    let value = local_at(&mut ssa, 2, 0, 2);
    let flags = local_at(&mut ssa, 3, 0, 2);
    let flag_value = local_at(&mut ssa, 4, 0, 3);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: left,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: right,
        value: ConstValue::I32(20),
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: value,
        left,
        right,
        flags: Some(flags),
    }));
    block.add_instruction(instr(SsaOp::ReadFlags {
        dest: flag_value,
        flags,
        mask: FlagsMask::ZERO,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(flag_value))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    run_pass_boundary(&mut ssa, "secondary def dce", |ssa| {
        DeadCodeEliminationPass.run(
            ssa,
            &PassEnv::new(&70u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert_valid_ssa(&ssa, "secondary def dce");
    assert!(ssa.iter_instructions().any(
        |(_, _, instr)| matches!(instr.op(), SsaOp::Add { flags: Some(f), .. } if *f == flags)
    ));
}

#[test]
fn dead_code_elimination_keeps_an_unread_system_register_exchange() {
    let mut ssa = SsaFunction::new(0, 2);
    let input = local_at(&mut ssa, 0, 0, 0);
    let output = local_at(&mut ssa, 1, 0, 1);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: input,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::SystemOp(Box::new(NativeKindedData {
        kind: SystemOpKind::ControlRegister(SysRegOp::ExchangeSystemRegister),
        mnemonic: "csrrw".to_string(),
        metadata: None,
        outputs: vec![output],
        inputs: vec![input],
    }))));
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    run_pass_boundary(&mut ssa, "system exchange dce", |ssa| {
        DeadCodeEliminationPass.run(
            ssa,
            &PassEnv::new(&71u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert_valid_ssa(&ssa, "system exchange dce");
    assert!(
        ssa.iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::SystemOp(_)))
    );
}

#[test]
fn dead_code_elimination_keeps_atomic_and_fence_barriers_when_unused() {
    let mut ssa = SsaFunction::new(0, 3);
    let addr = ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::Ptr,
    );
    let value = local_at(&mut ssa, 1, 0, 1);
    let old = local_at(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: addr,
        value: ConstValue::I32(0),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: value,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::AtomicExchange {
        dest: old,
        addr,
        value,
        ordering: AtomicOrdering::SeqCst,
        width: AtomicAccessWidth::Bits32,
        access: AccessQualifiers::FLAT.made_volatile(),
    }));
    block.add_instruction(instr(SsaOp::Fence {
        kind: FenceKind::SeqCst,
    }));
    block.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    run_pass_boundary(&mut ssa, "atomic fence dce", |ssa| {
        DeadCodeEliminationPass.run(
            ssa,
            &PassEnv::new(&72u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert_valid_ssa(&ssa, "atomic fence dce");
    assert!(
        ssa.iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::AtomicExchange { .. }))
    );
    assert!(
        ssa.iter_instructions()
            .any(|(_, _, instr)| matches!(instr.op(), SsaOp::Fence { .. }))
    );
}

// Block merge: coalescing adjacent blocks

#[test]
fn block_merge_coalesces_two_blocks_with_single_edge() {
    let mut ssa = SsaFunction::new(0, 2);
    let a = local_at(&mut ssa, 0, 0, 0);
    let b = local_at(&mut ssa, 1, 1, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(10),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Copy { dest: b, src: a }));
    b1.add_instruction(instr(SsaOp::returns(Some(b))));
    ssa.add_block(b1);

    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    let changed = run_pass_boundary(&mut ssa, "block merge coalescing", |ssa| {
        BlockMergingPass.run(
            ssa,
            &PassEnv::new(&8u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    assert!(changed, "block merge should coalesce the single-edge pair");
}

// LICM: blocked hoisting (cannot hoist if it creates phi operand issues)

#[test]
fn licm_does_not_hoist_when_loop_has_no_preheader() {
    // A loop without clean preheader - LICM should handle gracefully
    let mut ssa = SsaFunction::new(0, 3);
    let base = local_at(&mut ssa, 0, 0, 0);
    let cond = local_at(&mut ssa, 1, 0, 1);
    let invariant = local_at(&mut ssa, 2, 1, 0);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: base,
        value: ConstValue::I32(10),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::Add {
        dest: invariant,
        left: base,
        right: base,
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

    let log: EventLog<MockTarget> = EventLog::new();
    run_pass_boundary(&mut ssa, "licm no preheader", |ssa| {
        LicmPass.run(
            ssa,
            &PassEnv::new(&9u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert_valid_ssa(&ssa, "licm no preheader");
}

// GVN with complex expressions

#[test]
fn gvn_eliminates_duplicate_multi_level_expression() {
    let mut ssa = SsaFunction::new(0, 0);
    let a = local_at(&mut ssa, 0, 0, 0);
    let b = local_at(&mut ssa, 1, 0, 1);
    let c = local_at(&mut ssa, 2, 0, 2);
    let expr1 = local_at(&mut ssa, 3, 0, 3);
    let expr2 = local_at(&mut ssa, 4, 0, 4);
    let result1 = local_at(&mut ssa, 5, 0, 5);
    let result2 = local_at(&mut ssa, 6, 0, 6);

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
    // (a + b) * c
    block.add_instruction(instr(SsaOp::Add {
        dest: expr1,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: result1,
        left: expr1,
        right: c,
        flags: None,
    }));
    // Duplicate: (a + b) * c again
    block.add_instruction(instr(SsaOp::Add {
        dest: expr2,
        left: a,
        right: b,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: result2,
        left: expr2,
        right: c,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(result2))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    run_pass_repaired_boundary(&mut ssa, "multi-level gvn", |ssa| {
        GlobalValueNumberingPass.run(
            ssa,
            &PassEnv::new(&10u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    assert_valid_ssa(&ssa, "multi-level gvn");
}

// Reassociation with constants on both sides

#[test]
fn reassociate_combines_adjacent_constants() {
    let mut ssa = SsaFunction::new(0, 0);
    let x = local_at(&mut ssa, 0, 0, 0);
    let c1 = local_at(&mut ssa, 1, 0, 1);
    let c2 = local_at(&mut ssa, 2, 0, 2);
    let inner = local_at(&mut ssa, 3, 0, 3);
    let outer = local_at(&mut ssa, 4, 0, 4);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: x,
        value: ConstValue::I32(5),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: c1,
        value: ConstValue::I32(3),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: c2,
        value: ConstValue::I32(7),
    }));
    // (x + 3) + 7  →  reassociate to x + (3 + 7) = x + 10
    block.add_instruction(instr(SsaOp::Add {
        dest: inner,
        left: x,
        right: c1,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::Add {
        dest: outer,
        left: inner,
        right: c2,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(outer))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    let changed = ReassociationPass.run(
        &mut ssa,
        &PassEnv::new(&11u32, &log, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(changed, "reassociation should fire");

    assert_valid_ssa(&ssa, "reassociation");
}

// Strength reduction: multi-constant guarding

#[test]
fn strength_reduces_power_of_two_multiplication() {
    let mut ssa = SsaFunction::new(0, 0);
    let x = local_at(&mut ssa, 0, 0, 0);
    let pow2 = local_at(&mut ssa, 1, 0, 1);
    let result = local_at(&mut ssa, 2, 0, 2);

    let mut block = SsaBlock::new(0);
    block.add_instruction(instr(SsaOp::Const {
        dest: x,
        value: ConstValue::I32(10),
    }));
    block.add_instruction(instr(SsaOp::Const {
        dest: pow2,
        value: ConstValue::I32(8), // 2^3
    }));
    block.add_instruction(instr(SsaOp::Mul {
        dest: result,
        left: x,
        right: pow2,
        flags: None,
    }));
    block.add_instruction(instr(SsaOp::returns(Some(result))));
    ssa.add_block(block);
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    let changed = StrengthReductionPass.run(
        &mut ssa,
        &PassEnv::new(&12u32, &log, AnalysisContext::new(MACHINE)),
    ) == Ok(true);
    assert!(changed, "strength reduction should fire for x * 8 → x << 3");
    assert!(log.has(EventKind::StrengthReduced));

    assert_valid_ssa(&ssa, "strength reduction");
}

// Jump threading: conditional simplification via evaluator

#[test]
fn jump_threading_leaves_direct_constant_branch_to_controlflow() {
    let mut ssa = SsaFunction::new(0, 3);
    let true_val = local_at(&mut ssa, 0, 0, 0);
    let cond = local_at(&mut ssa, 1, 0, 1);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: true_val,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: cond,
        value: ConstValue::I32(1), // Always truthy
    }));
    b0.add_instruction(instr(SsaOp::Branch {
        condition: cond,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(instr(SsaOp::returns(Some(true_val))));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    b2.add_instruction(instr(SsaOp::returns(None)));
    ssa.add_block(b2);

    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    let changed = run_pass_boundary(&mut ssa, "direct constant jump threading", |ssa| {
        JumpThreadingPass.run(
            ssa,
            &PassEnv::new(&13u32, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert!(
        !changed,
        "jump threading should leave direct constant branches to controlflow/ranges"
    );
    assert!(!log.has(EventKind::ControlFlowRestructured));
}

// Full pass pipeline on a moderately complex function

fn build_complex_mixed_function() -> SsaFunction<MockTarget> {
    let mut ssa = SsaFunction::new(0, 15);
    let a = local_at(&mut ssa, 0, 0, 0);
    let b = local_at(&mut ssa, 1, 0, 1);
    let zero = local_at(&mut ssa, 2, 0, 2);
    let four = local_at(&mut ssa, 3, 0, 3);
    let sum1 = local_at(&mut ssa, 4, 0, 4);
    let sum2 = local_at(&mut ssa, 5, 0, 5);
    let mul1 = local_at(&mut ssa, 6, 0, 6);
    let mul2 = local_at(&mut ssa, 7, 0, 7);
    let copy = local_at(&mut ssa, 8, 0, 8);
    let _cmp = local_at(&mut ssa, 9, 0, 9);
    let unused = local_at(&mut ssa, 10, 0, 10);
    let self_eq = local_at(&mut ssa, 11, 0, 11);
    let c1 = local_at(&mut ssa, 12, 0, 12);
    let c2 = local_at(&mut ssa, 13, 0, 13);
    let final_val = local_at(&mut ssa, 14, 0, 14);

    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(instr(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(6),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(2),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: zero,
        value: ConstValue::I32(0),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: four,
        value: ConstValue::I32(4),
    }));
    b0.add_instruction(instr(SsaOp::Add {
        dest: sum1,
        left: a,
        right: zero,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::Add {
        dest: sum2,
        left: a,
        right: zero,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::Mul {
        dest: mul1,
        left: sum1,
        right: four,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::Mul {
        dest: mul2,
        left: sum2,
        right: four,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::Copy {
        dest: copy,
        src: mul1,
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: unused,
        value: ConstValue::I32(999),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: c1,
        value: ConstValue::I32(3),
    }));
    b0.add_instruction(instr(SsaOp::Const {
        dest: c2,
        value: ConstValue::I32(5),
    }));
    b0.add_instruction(instr(SsaOp::Ceq {
        dest: self_eq,
        left: c1,
        right: c1,
        signaling: false,
    }));
    b0.add_instruction(instr(SsaOp::Add {
        dest: final_val,
        left: copy,
        right: c1,
        flags: None,
    }));
    b0.add_instruction(instr(SsaOp::returns(Some(final_val))));
    ssa.add_block(b0);

    ssa.recompute_uses();
    ssa
}

#[test]
fn full_pass_pipeline_on_mixed_function_preserves_valid_ssa() {
    let mut ssa = build_complex_mixed_function();
    let log: EventLog<MockTarget> = EventLog::new();
    let method = 0xBADF00Du32;
    let ptr_size = PointerSize::Bit64;

    // Run the full complement of passes
    assert!(
        AlgebraicSimplificationPass
            .run(
                &mut ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    ssa.recompute_uses();
    assert!(
        GlobalValueNumberingPass
            .run(
                &mut ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    ssa.recompute_uses();
    assert!(
        ReassociationPass
            .run(
                &mut ssa,
                &PassEnv::new(
                    &method,
                    &log,
                    AnalysisContext::new(Machine {
                        ptr_size,
                        ..MACHINE
                    })
                )
            )
            .is_ok()
    );
    ssa.recompute_uses();
    assert!(
        StrengthReductionPass
            .run(
                &mut ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    ssa.recompute_uses();
    CopyPropagationPass.propagate(
        &mut ssa,
        &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        |_, _| {},
    );
    ssa.recompute_uses();
    assert!(
        ValueRangePropagationPass
            .run(
                &mut ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    ssa.recompute_uses();
    assert!(
        OpaquePredicatePass::new()
            .run(
                &mut ssa,
                &PassEnv::new(
                    &method,
                    &log,
                    AnalysisContext::new(Machine {
                        ptr_size,
                        ..MACHINE
                    })
                )
            )
            .is_ok()
    );
    ssa.recompute_uses();
    assert!(
        DeadCodeEliminationPass
            .run(
                &mut ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );
    ssa.recompute_uses();

    assert_valid_ssa(&ssa, "full mixed pipeline");

    // All events should reference our method
    for ev in &log {
        assert_eq!(ev.method, Some(method));
    }
}

/// Folding `Branch(cond, A, B)` to `Jump(A)` must prune `B`'s phi operand for
/// the folded block and settle at the cheap repair tier.
///
/// Removing an edge only *increases* dominance - it removes paths - so no new
/// phi can ever be required and no value needs reconstructing. That is why the
/// branch-folding passes use `fold_terminator_pruning_phis` rather than
/// `mark_cfg_changed`, which leaves the repair to the caller.
#[test]
fn folding_a_branch_prunes_the_dropped_edges_phi_operand() {
    let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
    let a = SsaVarId::from_index(0);
    ssa.create_variable(
        VariableOrigin::Local(0),
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let b = SsaVarId::from_index(1);
    ssa.create_variable(
        VariableOrigin::Local(1),
        0,
        DefSite::instruction(1, 0),
        MockType::I32,
    );
    let merged = SsaVarId::from_index(2);
    ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

    // B0: cond -> Branch(B1, B2).  B1 -> B2.  B2: phi(B0, B1).
    let mut b0 = SsaBlock::new(0);
    b0.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: a,
        value: ConstValue::I32(1),
    }));
    b0.add_instruction(SsaInstruction::synthetic(SsaOp::Branch {
        condition: a,
        true_target: 1,
        false_target: 2,
    }));
    ssa.add_block(b0);

    let mut b1 = SsaBlock::new(1);
    b1.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
        dest: b,
        value: ConstValue::I32(2),
    }));
    b1.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
    ssa.add_block(b1);

    let mut b2 = SsaBlock::new(2);
    let mut phi = PhiNode::new(merged, VariableOrigin::Local(2));
    phi.add_operand(PhiOperand::new(a, 0));
    phi.add_operand(PhiOperand::new(b, 1));
    b2.add_phi(phi);
    b2.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(merged))));
    ssa.add_block(b2);
    ssa.recompute_uses();

    // Fold B0's branch so it always goes to B1 - dropping the B0 -> B2 edge.
    let mut pruned = 0;
    let result = ssa.edit(SsaEditOptions::new(), |editor| {
        pruned = editor.fold_terminator_pruning_phis(0, SsaOp::Jump { target: 1 })?;
        Ok(())
    });
    assert!(
        result.is_ok(),
        "fold_terminator_pruning_phis must succeed on a well-formed branch"
    );
    let changed = result.map(|report| report.changed).unwrap_or(false);

    assert_eq!(
        pruned, 1,
        "the dropped B0 -> B2 edge's operand must be pruned"
    );

    // No surviving phi may still name B0 as a predecessor. The phi itself may
    // be gone entirely: once reduced to one operand it is trivial, and the
    // boundary repair collapses it into a copy - which is the correct outcome,
    // not a lost merge.
    for block in ssa.blocks() {
        for phi in block.phi_nodes() {
            assert!(
                phi.operands().iter().all(|op| op.predecessor() != 0),
                "no phi may reference the folded-away B0 edge"
            );
        }
    }

    assert!(changed);
    assert_eq!(ssa.validate(), Ok(()), "the fold must leave valid SSA");
}

/// Strength reduction turns a signed division into a shift when the range
/// analysis proves the dividend cannot be negative.
///
/// The signed arms were gated on a caller-supplied predicate, and every caller
/// in the crate and in both host crates supplied one that answered `false` - so
/// this rewrite existed and had never once fired. The gate reads the proved
/// ranges now, which is the one thing that knows.
#[test]
fn a_signed_division_reduces_when_the_range_proves_the_sign() {
    // A constant dividend is proved non-negative; an argument is proved
    // nothing.
    let build = |known_dividend: bool| {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 3);
        let dividend = if known_dividend {
            ssa.create_variable(
                VariableOrigin::Local(0),
                0,
                DefSite::instruction(0, 0),
                MockType::I32,
            )
        } else {
            ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::I32,
            )
        };
        let divisor = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let quotient = ssa.create_variable(
            VariableOrigin::Local(2),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );

        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: if known_dividend {
                dividend
            } else {
                ssa.create_variable(
                    VariableOrigin::Local(0),
                    0,
                    DefSite::instruction(0, 0),
                    MockType::I32,
                )
            },
            value: ConstValue::I32(40),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: divisor,
            value: ConstValue::I32(8),
        }));
        block.add_instruction(
            SsaInstruction::synthetic(SsaOp::Div {
                dest: quotient,
                left: dividend,
                right: divisor,
                unsigned: false,
                flags: None,
            })
            .with_result_type(MockType::I32),
        );
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(quotient))));
        ssa.add_block(block);
        ssa.recompute_uses();
        ssa
    };

    let shifted = |ssa: &SsaFunction<MockTarget>| {
        ssa.blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .any(|instr| matches!(instr.op(), SsaOp::Shr { .. }))
    };

    let log: EventLog<MockTarget> = EventLog::new();

    // With the facts, the signed division becomes a shift.
    let mut with_facts = build(true);
    assert!(
        StrengthReductionPass.run(
            &mut with_facts,
            &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
        ) == Ok(true)
    );
    assert!(
        shifted(&with_facts),
        "a dividend proved non-negative divides by shifting: {with_facts}"
    );

    // Without them it is left alone, because a negative dividend rounds the
    // other way and the shift would be wrong.
    let mut without_facts = build(false);
    let _ = StrengthReductionPass.run(
        &mut without_facts,
        &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
    );
    assert!(
        !shifted(&without_facts),
        "nothing is proved, so the signed arm stays: {without_facts}"
    );
}

/// A load of a slot whose address was handed to a call is not what the
/// function last stored there.
///
/// `lockTaken = false; Monitor.Enter(obj, ref lockTaken); if (lockTaken) ...`
/// with the address computed first, as it is once the address computations are
/// shared: the callee writes the slot through the pointer, so the branch is
/// undecided. Deciding it from the `false` stored before the call deletes the
/// `Monitor.Exit` in the `finally` that tests it.
#[test]
fn predicates_does_not_decide_a_load_of_a_slot_a_call_may_write() {
    let mut ssa = SsaFunction::<MockTarget>::new(0, 1);
    let address = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 0),
        MockType::I32,
    );
    let flag = local_at(&mut ssa, 0, 0, 1);
    let loaded = ssa.create_variable(
        VariableOrigin::Phi,
        0,
        DefSite::instruction(0, 3),
        MockType::I32,
    );

    let mut b0 = SsaBlock::new(0);
    for op in [
        SsaOp::LoadLocalAddr {
            dest: address,
            local_index: 0,
        },
        SsaOp::Const {
            dest: flag,
            value: ConstValue::I32(0),
        },
        SsaOp::from(CallData::new(None, Callee::Direct(7), vec![address])),
        SsaOp::LoadLocal {
            dest: loaded,
            local_index: 0,
        },
        SsaOp::Branch {
            condition: loaded,
            true_target: 1,
            false_target: 2,
        },
    ] {
        b0.add_instruction(instr(op));
    }
    ssa.add_block(b0);
    for index in [1, 2] {
        let mut block = SsaBlock::new(index);
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
    }
    ssa.recompute_uses();

    let log: EventLog<MockTarget> = EventLog::new();
    assert!(
        OpaquePredicatePass::new()
            .run(
                &mut ssa,
                &PassEnv::new(&3u32, &log, AnalysisContext::new(MACHINE))
            )
            .is_ok()
    );

    assert!(
        matches!(
            ssa.block(0)
                .and_then(|block| block.instructions().last())
                .map(|i| i.op()),
            Some(SsaOp::Branch { .. })
        ),
        "the callee may have set the flag:\n{ssa}"
    );
}
