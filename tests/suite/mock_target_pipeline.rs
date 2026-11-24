//! Integration tests for the target-agnostic pass pipeline with `MockTarget`.

use std::process;

use analyssa::{
    Result,
    analysis::{
        VerifyLevel,
        budget::{Budget, WorkBudget},
        context::AnalysisContext,
    },
    events::{EventKind, EventLog},
    ir::{SsaFunction, function::SsaFunctionBuilder, ops::SsaOp},
    passes::{
        AlgebraicSimplificationPass, BlockMergingPass, ControlFlowSimplificationPass,
        CopyPropagationPass, DeadCodeEliminationPass, GlobalValueNumberingPass,
    },
    scheduling::{FunctionPass, PassEnv},
    testing::{MACHINE, MockTarget, mock_i32, scalar_rewrite_fixture},
};

use crate::common::{
    assert_event, assert_has_op, assert_pass_changes, run_pass_boundary, terminator_at,
};

fn result_or_abort<T>(result: Result<T>) -> T {
    result.unwrap_or_else(|_| process::abort())
}

fn trampoline_pipeline_fixture() -> SsaFunction<MockTarget> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.ensure_block(2);
    result_or_abort(builder.in_block(0, |block| {
        block.jump(1)?;
        Ok(())
    }));
    result_or_abort(builder.in_block(1, |block| {
        block.jump(2)?;
        Ok(())
    }));
    result_or_abort(builder.in_block(2, |block| {
        let value = block.const_i32(mock_i32(), 1)?;
        block.ret(Some(value))?;
        Ok(())
    }));
    result_or_abort(builder.finish_verified(VerifyLevel::Full))
}

#[test]
fn mock_target_drives_scalar_cleanup_pipeline() {
    let mut ssa = scalar_rewrite_fixture();
    let log: EventLog<MockTarget> = EventLog::new();
    let method = 0xDEAD_BEEFu32;

    assert_pass_changes(&mut ssa, "algebraic", |ssa| {
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
    assert_pass_changes(&mut ssa, "copy propagation", |ssa| {
        CopyPropagationPass.propagate(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::CopyRounds, 5)),
            |_, _| {},
        )
    });
    assert_pass_changes(&mut ssa, "dead code elimination", |ssa| {
        DeadCodeEliminationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert_event(&log, EventKind::ConstantFolded);
    assert_event(&log, EventKind::CopyPropagated);
    assert_event(&log, EventKind::InstructionRemoved);
    assert!(matches!(
        terminator_at(&ssa, 1),
        SsaOp::Return { value: Some(_), .. }
    ));
    for event in &log {
        assert_eq!(event.method, Some(method));
    }
}

#[test]
fn mock_target_runs_structural_pipeline() {
    let mut ssa = trampoline_pipeline_fixture();
    let log: EventLog<MockTarget> = EventLog::new();
    let method = 1u32;

    assert_pass_changes(&mut ssa, "block merging", |ssa| {
        BlockMergingPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });
    run_pass_boundary(&mut ssa, "controlflow", |ssa| {
        ControlFlowSimplificationPass.run(
            ssa,
            &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
        ) == Ok(true)
    });

    assert!(matches!(
        terminator_at(&ssa, 0),
        SsaOp::Jump { target: 2 } | SsaOp::Return { .. }
    ));
    assert_has_op(&ssa, "pipeline return", |op| {
        matches!(op, SsaOp::Return { .. })
    });
}
