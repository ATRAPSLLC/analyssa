//! Independent output dimensions bound whole residuals before byte expansion.

use std::{
    cell::{Cell, RefCell},
    ops::Range,
};

use analyssa::{
    Error, Result,
    analysis::{
        SsaVerifier, VerifyLevel,
        budget::{Budget, Exhaustion, WorkBudget},
        context::AnalysisContext,
        execution::{CallModel, ExecutionHost, ModelCall, StaticExecution},
    },
    ir::{
        Callee,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::{CmpKind, SsaOp},
        value::ConstValue,
        variable::{DefSite, VariableOrigin, VariableRole},
    },
    passes::residual::{ResidualRefusal, ResidualRewrite, ResidualSize, rewrite_static_facts},
    testing::{MACHINE, MockTarget, MockType},
};

#[derive(Default)]
struct Host {
    observed: RefCell<Vec<Option<Vec<u8>>>>,
}
impl ExecutionHost<MockTarget> for Host {
    fn stack_extent(&self, _: &SsaFunction<MockTarget>, _: Option<&u32>) -> Option<Range<i64>> {
        Some(-8192..0)
    }
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 9).then_some(self)
    }
}
impl CallModel<MockTarget> for Host {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        let bytes = call
            .integer(1)
            .and_then(|count| usize::try_from(count).ok())
            .and_then(|count| {
                call.address(0)
                    .and_then(|address| call.read(address, count))
            });
        self.observed.borrow_mut().push(bytes);
    }
}

fn body(bytes: usize) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let stack = builder.function_mut().create_variable(
        VariableOrigin::EntryLiveIn,
        0,
        DefSite::entry(),
        MockType::Ptr,
    );
    builder
        .function_mut()
        .variable_mut(stack)
        .ok_or_else(|| Error::new("stack"))?
        .set_role(VariableRole::EntryStackPointer);
    let offset = i64::try_from(bytes)
        .ok()
        .and_then(i64::checked_neg)
        .ok_or_else(|| Error::new("fixture extent"))?;
    let (address, zero, one, two, size) = builder.in_block(0, |block| {
        let address = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: stack,
            index: None,
            stride: 1,
            offset,
            result_type: MockType::Ptr,
        })?;
        let zero = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
        let one = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(1))?;
        let two = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(2))?;
        let size = block.const_value(
            SsaDefSpec::tmp(MockType::U64),
            ConstValue::U64(u64::try_from(bytes).map_err(|_| Error::new("fixture size"))?),
        )?;
        let fill = block.add(SsaDefSpec::tmp(MockType::U8), zero, one)?;
        block.jump(1)?;
        Ok((address, zero, fill, two, size))
    })?;
    let counter = builder.in_block(1, |block| block.empty_phi(SsaDefSpec::tmp(MockType::U8)))?;
    let next = builder.in_block(1, |block| {
        block.init_blk(address, one, size, None)?;
        let next = block.add(SsaDefSpec::tmp(MockType::U8), counter, one)?;
        block.branch_cmp(next, two, CmpKind::Lt, true, 1, 2)?;
        Ok(next)
    })?;
    builder.add_phi_operand(1, counter, 0, zero)?;
    builder.add_phi_operand(1, counter, 1, next)?;
    builder.in_block(2, |block| {
        block.call_void(Callee::Direct(9), vec![address, size])?;
        block.ret(Some(next))
    })?;
    let function = builder.finish()?;
    assert!(
        SsaVerifier::new(&function)
            .verify(VerifyLevel::Full)
            .is_empty()
    );
    Ok(function)
}

#[test]
fn exact_peak_bounds_preserve_thirty_one_to_thirty_three_byte_consumers() -> Result<()> {
    for bytes in [31usize, 32, 33] {
        let function = body(bytes)?;
        let original = format!("{function:?}");
        let context = AnalysisContext::new(MACHINE);
        let host = Host::default();
        let facts = StaticExecution::new(&function, &context)
            .with_host(&host)
            .run();
        assert!(facts.is_complete());
        assert_eq!(facts.decoded_loops(), vec![1]);
        assert_eq!(*host.observed.borrow(), vec![Some(vec![1; bytes])]);
        let initial = ResidualSize::measure(&function, &WorkBudget::DEFAULT, None)
            .map_err(|e| Error::new(format!("{e:?}")))?;
        // One nonzero stack base, n byte constants, n-1 offset definitions,
        // n stores, and one returned counter materialized in the preheader.
        let operations = initial
            .operations
            .checked_add(
                bytes
                    .checked_mul(3)
                    .and_then(|n| n.checked_add(1))
                    .ok_or_else(|| Error::new("ops"))?,
            )
            .ok_or_else(|| Error::new("ops"))?;
        let variables = initial
            .variables
            .checked_add(
                bytes
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(1))
                    .ok_or_else(|| Error::new("vars"))?,
            )
            .ok_or_else(|| Error::new("vars"))?;
        let exact = WorkBudget::DEFAULT
            .with_limit(Budget::ResidualOperations, operations)
            .with_limit(Budget::ResidualVariables, variables)
            .with_limit(Budget::ResidualBlocks, initial.blocks)
            .with_limit(Budget::ResidualPhiOperands, initial.phi_operands);
        for (kind, limit) in [
            (Budget::ResidualOperations, operations),
            (Budget::ResidualVariables, variables),
            (Budget::ResidualBlocks, initial.blocks),
            (Budget::ResidualPhiOperands, initial.phi_operands),
        ] {
            assert!(
                matches!(rewrite_static_facts(&function, &facts, &exact.with_limit(kind, limit.saturating_sub(1)), &|| false)?, ResidualRewrite::Refused(ResidualRefusal::Exhausted(Exhaustion { budget, .. })) if budget == kind)
            );
            assert_eq!(format!("{function:?}"), original);
            assert!(facts.is_complete());
        }
        for limits in [exact, WorkBudget::DEFAULT] {
            let ResidualRewrite::Rewritten {
                function: rewritten,
                replacements,
                runs,
            } = rewrite_static_facts(&function, &facts, &limits, &|| false)?
            else {
                return Err(Error::new("exact bound refused"));
            };
            // The one run the loop decoded, written back one store per byte in
            // address order: each store in the rewritten body is the byte.
            let [run] = runs.as_slice() else {
                return Err(Error::new(format!("one run expected: {runs:?}")));
            };
            assert_eq!((run.header, run.run, run.stores.len()), (1, 0, bytes));
            let stored: Vec<u8> = run
                .stores
                .iter()
                .map(|id| {
                    let (block, index) = rewritten
                        .locate_instruction(*id)
                        .ok_or_else(|| Error::new("store not in the body"))?;
                    let instruction = rewritten
                        .block(block)
                        .and_then(|block| block.instruction(index))
                        .ok_or_else(|| Error::new("store location"))?;
                    match instruction.op() {
                        SsaOp::StoreIndirect { value, .. } => rewritten
                            .get_var_constant(*value)
                            .and_then(ConstValue::as_u64)
                            .and_then(|byte| u8::try_from(byte).ok())
                            .ok_or_else(|| Error::new("stored byte")),
                        _ => Err(Error::new("not a store")),
                    }
                })
                .collect::<Result<_>>()?;
            assert_eq!(stored, vec![1; bytes]);
            assert!(
                replacements >= 2,
                "scalar and loop edits must share the transaction"
            );
            assert!(
                SsaVerifier::new(&rewritten)
                    .verify(VerifyLevel::Full)
                    .is_empty()
            );
            assert_eq!(format!("{function:?}"), original);
            let host = Host::default();
            let after = StaticExecution::new(&rewritten, &context)
                .with_host(&host)
                .run();
            assert!(after.is_complete());
            assert_eq!(
                after.returned().and_then(|value| value.constant()),
                Some(&ConstValue::U8(2))
            );
            assert_eq!(*host.observed.borrow(), vec![Some(vec![1; bytes])]);
        }
    }
    Ok(())
}

#[test]
fn large_fill_is_refused_before_emission_despite_small_execution() -> Result<()> {
    let function = body(4096)?;
    let original = format!("{function:?}");
    let context = AnalysisContext::new(MACHINE);
    let host = Host::default();
    let facts = StaticExecution::new(&function, &context)
        .with_host(&host)
        .run();
    assert!(facts.is_complete());
    // The loop's two fills and the observing call's read cost their 12288
    // bytes; the instructions around them are few.
    assert!(
        (12288..12388).contains(&facts.usage().steps),
        "{:?}",
        facts.usage()
    );
    let limits = WorkBudget::DEFAULT.with_limit(Budget::ResidualOperations, 1000);
    assert!(matches!(
        rewrite_static_facts(&function, &facts, &limits, &|| false)?,
        ResidualRewrite::Refused(ResidualRefusal::Exhausted(Exhaustion {
            budget: Budget::ResidualOperations,
            ..
        }))
    ));
    assert_eq!(format!("{function:?}"), original);
    assert_eq!(*host.observed.borrow(), vec![Some(vec![1; 4096])]);
    Ok(())
}

#[test]
fn cancellation_during_expansion_and_after_repair_discards_the_whole_copy() -> Result<()> {
    let function = body(4096)?;
    let original = format!("{function:?}");
    let context = AnalysisContext::new(MACHINE);
    let host = Host::default();
    let facts = StaticExecution::new(&function, &context)
        .with_host(&host)
        .run();
    assert!(facts.is_complete());
    let polls = Cell::new(0usize);
    let count = || {
        polls.set(polls.get().saturating_add(1));
        false
    };
    assert!(matches!(
        rewrite_static_facts(&function, &facts, &WorkBudget::DEFAULT, &count)?,
        ResidualRewrite::Rewritten { .. }
    ));
    let total = polls.get();
    assert!(total >= 128, "4096 bytes require 128 emission checkpoints");
    // Planning this three-block body uses far fewer checkpoints than expansion.
    // The final three checkpoints follow editor repair and verification.
    for stop in [
        1,
        total.checked_div(2).ok_or_else(|| Error::new("polls"))?,
        total.saturating_sub(2),
        total,
    ] {
        polls.set(0);
        let cancelled = || {
            polls.set(polls.get().saturating_add(1));
            polls.get() == stop
        };
        assert!(matches!(
            rewrite_static_facts(&function, &facts, &WorkBudget::DEFAULT, &cancelled)?,
            ResidualRewrite::Refused(ResidualRefusal::Cancelled)
        ));
        assert_eq!(polls.get(), stop, "cancellation must remain sticky");
        assert_eq!(format!("{function:?}"), original);
        assert!(facts.is_complete());
    }
    Ok(())
}
