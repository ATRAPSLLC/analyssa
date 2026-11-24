//! Loop residuals must repair exit-edge phi inputs without changing bypass paths.
use std::{cell::RefCell, ops::Range};

use analyssa::{
    Error, Result,
    analysis::{
        SsaVerifier, VerifyLevel,
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
    passes::residual::{apply_static_facts, residualize_loops},
    testing::{MACHINE, MockTarget, MockType},
};

#[derive(Clone, Copy)]
enum ExitPhis {
    None,
    Single,
    Multiple,
    Outside,
}

#[derive(Default)]
struct Consumer {
    observed: RefCell<Vec<Option<Vec<u8>>>>,
}

impl CallModel<MockTarget> for Consumer {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        let bytes = call.address(0).and_then(|address| call.read(address, 16));
        self.observed.borrow_mut().push(bytes);
    }
}

#[derive(Default)]
struct Host(Consumer);
impl ExecutionHost<MockTarget> for Host {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 9).then_some(&self.0)
    }
    fn stack_extent(&self, _: &SsaFunction<MockTarget>, _: Option<&u32>) -> Option<Range<i64>> {
        Some(-16..0)
    }
}

fn body(phis: ExitPhis, bypass: bool) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(2, 0);
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
    let selector = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::U8,
    );
    let dynamic = builder.function_mut().create_variable(
        VariableOrigin::Argument(1),
        0,
        DefSite::entry(),
        MockType::U8,
    );
    let (address, zero, one, two, size, other) = builder.in_block(0, |block| {
        let address = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
            dest,
            base: stack,
            index: None,
            stride: 1,
            offset: -16,
            result_type: MockType::Ptr,
        })?;
        let zero = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
        let one = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(1))?;
        let two = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(2))?;
        let size = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(16))?;
        let other = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(42))?;
        // A separate scalar replacement shares the original-snapshot transaction.
        block.add(SsaDefSpec::tmp(MockType::U8), zero, one)?;
        if bypass {
            block.branch_cmp(selector, zero, CmpKind::Eq, true, 1, 3)?;
        } else {
            block.jump(1)?;
        }
        Ok((address, zero, one, two, size, other))
    })?;
    builder.in_block(1, |block| block.jump(2))?;
    let counter = builder.in_block(2, |block| block.empty_phi(SsaDefSpec::tmp(MockType::U8)))?;
    let next = builder.in_block(2, |block| {
        block.init_blk(address, one, size, None)?;
        let next = block.add(SsaDefSpec::tmp(MockType::U8), counter, one)?;
        block.branch_cmp(next, two, CmpKind::Lt, true, 2, 3)?;
        Ok(next)
    })?;
    builder.add_phi_operand(2, counter, 1, zero)?;
    builder.add_phi_operand(2, counter, 2, next)?;
    let returned = if matches!(phis, ExitPhis::None) {
        next
    } else {
        let result = builder.in_block(3, |block| block.empty_phi(SsaDefSpec::tmp(MockType::U8)))?;
        let from_loop = if matches!(phis, ExitPhis::Outside) {
            dynamic
        } else {
            next
        };
        builder.add_phi_operand(3, result, 2, from_loop)?;
        if bypass {
            builder.add_phi_operand(3, result, 0, other)?;
        }
        if matches!(phis, ExitPhis::Multiple) {
            let second =
                builder.in_block(3, |block| block.empty_phi(SsaDefSpec::tmp(MockType::U8)))?;
            builder.add_phi_operand(3, second, 2, dynamic)?;
            if bypass {
                builder.add_phi_operand(3, second, 0, one)?;
            }
            builder.in_block(3, |block| {
                block.add(SsaDefSpec::tmp(MockType::U8), result, second)
            })?
        } else {
            result
        }
    };
    builder.in_block(3, |block| {
        block.call_void(Callee::Direct(9), vec![address])?;
        block.ret(Some(returned))
    })?;
    let ssa = builder.finish()?;
    let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Full);
    assert!(errors.is_empty(), "{errors:?}");
    Ok(ssa)
}

struct BoundHost<'a>(&'a SsaFunction<MockTarget>, Consumer);
impl ExecutionHost<MockTarget> for BoundHost<'_> {
    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 9).then_some(&self.1)
    }
    fn body(&self, method: &u32) -> Option<&SsaFunction<MockTarget>> {
        (*method == 7).then_some(self.0)
    }
    fn stack_extent(&self, _: &SsaFunction<MockTarget>, _: Option<&u32>) -> Option<Range<i64>> {
        Some(-16..0)
    }
}
fn bound_return(
    callee: &SsaFunction<MockTarget>,
    selector: u8,
    dynamic: u8,
    expected_bytes: Option<Vec<u8>>,
) -> Result<Option<ConstValue<MockTarget>>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    builder.in_block(0, |block| {
        let argument =
            block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(selector))?;
        let payload = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(dynamic))?;
        let result = block.call(
            SsaDefSpec::tmp(MockType::U8),
            Callee::Direct(7),
            vec![argument, payload],
        )?;
        block.ret(Some(result))
    })?;
    let caller = builder.finish()?;
    assert!(
        SsaVerifier::new(&caller)
            .verify(VerifyLevel::Full)
            .is_empty()
    );
    let context = AnalysisContext::new(MACHINE);
    let host = BoundHost(callee, Consumer::default());
    let facts = StaticExecution::new(&caller, &context)
        .with_host(&host)
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(*host.1.observed.borrow(), vec![expected_bytes]);
    Ok(facts.returned().and_then(|v| v.constant()).cloned())
}

/// Computes byte arithmetic independently of the original or residual execution.
fn expected_return(phis: ExitPhis, bypass: bool, selector: u8, dynamic: u8) -> u8 {
    match phis {
        ExitPhis::None => 2,
        ExitPhis::Single => {
            if bypass && selector != 0 {
                42
            } else {
                2
            }
        }
        ExitPhis::Multiple => {
            if bypass && selector != 0 {
                43
            } else {
                dynamic.wrapping_add(2)
            }
        }
        ExitPhis::Outside => {
            if bypass && selector != 0 {
                42
            } else {
                dynamic
            }
        }
    }
}

fn check(phis: ExitPhis, bypass: bool, combined: bool) -> Result<()> {
    let mut ssa = body(phis, bypass)?;
    let original = ssa.clone();
    for dynamic in [9u8, 17] {
        for selector in [0, 1] {
            let expected = expected_return(phis, bypass, selector, dynamic);
            assert_eq!(
                bound_return(
                    &original,
                    selector,
                    dynamic,
                    if bypass && selector != 0 {
                        None
                    } else {
                        Some(vec![1; 16])
                    }
                )?,
                Some(ConstValue::U8(expected))
            );
        }
    }
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context)
        .with_host(&Host::default())
        .run();
    assert!(facts.is_complete(), "{:?}", facts.completion());
    assert_eq!(facts.decoded_loops(), vec![2]);
    let plans = if combined {
        apply_static_facts(&mut ssa, &facts)?
    } else {
        residualize_loops(&mut ssa, &facts)?
    };
    if combined {
        assert!(plans > 1, "both scalar and loop plans must commit");
    } else {
        assert_eq!(plans, 1);
    }
    assert!(SsaVerifier::new(&ssa).verify(VerifyLevel::Full).is_empty());
    assert!(
        ssa.blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .all(|instruction| !matches!(instruction.op(), SsaOp::InitBlk { .. }))
    );
    let stores: Vec<_> = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.instructions())
        .filter_map(|instruction| {
            if let SsaOp::StoreIndirect { value, .. } = instruction.op() {
                Some(ssa.get_var_constant(*value).cloned())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(stores, vec![Some(ConstValue::U8(1)); 16]);
    let after = StaticExecution::new(&ssa, &context)
        .with_host(&Host::default())
        .run();
    assert!(after.is_complete());
    assert!(after.decoded_loops().is_empty());
    // Opaque identities belong to one execution and SSA namespace. Prove the
    // preserved relation with every byte input and both selector equivalence classes.
    assert_eq!(after.returned().is_some(), facts.returned().is_some());
    for dynamic in u8::MIN..=u8::MAX {
        for selector in [0, 1] {
            let expected = Some(ConstValue::U8(expected_return(
                phis, bypass, selector, dynamic,
            )));
            assert_eq!(
                bound_return(
                    &ssa,
                    selector,
                    dynamic,
                    if bypass && selector != 0 {
                        None
                    } else {
                        Some(vec![1; 16])
                    }
                )?,
                expected,
                "residual selector={selector} dynamic={dynamic}"
            );
            assert_eq!(
                bound_return(
                    &original,
                    selector,
                    dynamic,
                    if bypass && selector != 0 {
                        None
                    } else {
                        Some(vec![1; 16])
                    }
                )?,
                expected,
                "original selector={selector} dynamic={dynamic}"
            );
        }
    }
    Ok(())
}

#[test]
fn no_exit_phi_control() -> Result<()> {
    check(ExitPhis::None, false, false)
}

#[test]
fn single_predecessor_exit_phi_is_repaired() -> Result<()> {
    check(ExitPhis::Single, false, false)
}

#[test]
fn exit_phi_preserves_exact_two_and_forty_two_bypass() -> Result<()> {
    check(ExitPhis::Single, true, false)
}

#[test]
fn multiple_exit_phis_preserve_dynamic_outside_operand_and_bypass() -> Result<()> {
    for bypass in [false, true] {
        check(ExitPhis::Multiple, bypass, false)?;
    }
    Ok(())
}

#[test]
fn exit_phi_can_carry_only_an_outside_dynamic_value() -> Result<()> {
    for bypass in [false, true] {
        check(ExitPhis::Outside, bypass, false)?;
    }
    Ok(())
}

#[test]
fn exit_phis_share_the_scalar_and_loop_transaction() -> Result<()> {
    check(ExitPhis::Multiple, true, true)
}
