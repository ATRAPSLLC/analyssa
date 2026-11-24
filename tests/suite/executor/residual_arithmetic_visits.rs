//! Loop replacement requires every arithmetic visit to prove normal completion.

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
    passes::residual::residualize_loops,
    testing::{MACHINE, MockTarget, MockType},
};

#[derive(Clone, Copy, Debug)]
enum Arithmetic {
    Plain,
    Div,
    Rem,
    SignedDiv,
    SignedRem,
    Wide,
    Checked,
    CheckedAdd,
    Zero,
    Unknown,
    LaterZero,
    SignedOverflow,
    RemainderOverflowUnknown,
    CheckedOverflow,
    CheckedUnknown,
    CheckedLaterOverflow,
    WideZero,
    WideUnknown,
    ConsumerInside,
}

#[derive(Default)]
struct Host {
    observed: RefCell<Vec<Option<Vec<u8>>>>,
}

impl ExecutionHost<MockTarget> for Host {
    fn stack_extent(&self, _: &SsaFunction<MockTarget>, _: Option<&u32>) -> Option<Range<i64>> {
        Some(-16..0)
    }

    fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        (*method == 9).then_some(self)
    }
}

impl CallModel<MockTarget> for Host {
    fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
        let bytes = call.address(0).and_then(|address| call.read(address, 16));
        self.observed.borrow_mut().push(bytes);
    }
}

fn body(arithmetic: Arithmetic) -> Result<SsaFunction<MockTarget>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    let dynamic = builder.function_mut().create_variable(
        VariableOrigin::Argument(0),
        0,
        DefSite::entry(),
        MockType::U16,
    );
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
    let (address, zero, one, two, byte, size, max, too_big, signed_two, min, minus_one) =
        builder.in_block(0, |block| {
            let address = block.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                dest,
                base: stack,
                index: None,
                stride: 1,
                offset: -16,
                result_type: MockType::Ptr,
            })?;
            let zero = block.const_value(SsaDefSpec::tmp(MockType::U16), ConstValue::U16(0))?;
            let one = block.const_value(SsaDefSpec::tmp(MockType::U16), ConstValue::U16(1))?;
            let two = block.const_value(SsaDefSpec::tmp(MockType::U16), ConstValue::U16(2))?;
            let byte = block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(1))?;
            let size = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(16))?;
            let max = block.const_value(SsaDefSpec::tmp(MockType::U16), ConstValue::U16(255))?;
            let too_big =
                block.const_value(SsaDefSpec::tmp(MockType::U16), ConstValue::U16(256))?;
            let signed_two =
                block.const_value(SsaDefSpec::tmp(MockType::I16), ConstValue::I16(2))?;
            let min =
                block.const_value(SsaDefSpec::tmp(MockType::I16), ConstValue::I16(i16::MIN))?;
            let minus_one =
                block.const_value(SsaDefSpec::tmp(MockType::I16), ConstValue::I16(-1))?;
            block.jump(1)?;
            Ok((
                address, zero, one, two, byte, size, max, too_big, signed_two, min, minus_one,
            ))
        })?;
    let counter = builder.in_block(1, |block| block.empty_phi(SsaDefSpec::tmp(MockType::U16)))?;
    let next = builder.in_block(1, |block| {
        block.init_blk(address, byte, size, None)?;
        match arithmetic {
            Arithmetic::Plain => {}
            Arithmetic::Checked
            | Arithmetic::CheckedOverflow
            | Arithmetic::CheckedUnknown
            | Arithmetic::CheckedLaterOverflow => {
                let operand = match arithmetic {
                    Arithmetic::CheckedOverflow => too_big,
                    Arithmetic::CheckedUnknown => dynamic,
                    Arithmetic::CheckedLaterOverflow => {
                        block.add(SsaDefSpec::tmp(MockType::U16), max, counter)?
                    }
                    _ => counter,
                };
                block.conv(
                    SsaDefSpec::tmp(MockType::U8),
                    operand,
                    MockType::U8,
                    true,
                    true,
                )?;
            }
            Arithmetic::CheckedAdd => {
                block.add_ovf(SsaDefSpec::tmp(MockType::U16), counter, one, true)?;
            }
            Arithmetic::SignedOverflow | Arithmetic::RemainderOverflowUnknown => {
                if matches!(arithmetic, Arithmetic::SignedOverflow) {
                    block.div(SsaDefSpec::tmp(MockType::I16), min, minus_one, false)?;
                } else {
                    block.rem(SsaDefSpec::tmp(MockType::I16), min, minus_one, false)?;
                }
            }
            Arithmetic::SignedDiv | Arithmetic::SignedRem => {
                let signed = block.conv(
                    SsaDefSpec::tmp(MockType::I16),
                    counter,
                    MockType::I16,
                    false,
                    true,
                )?;
                if matches!(arithmetic, Arithmetic::SignedDiv) {
                    block.div(SsaDefSpec::tmp(MockType::I16), signed, minus_one, false)?;
                } else {
                    block.rem(SsaDefSpec::tmp(MockType::I16), signed, signed_two, false)?;
                }
            }
            Arithmetic::Wide | Arithmetic::WideZero | Arithmetic::WideUnknown => {
                let divisor = match arithmetic {
                    Arithmetic::WideZero => zero,
                    Arithmetic::WideUnknown => dynamic,
                    _ => two,
                };
                block.wide_div(
                    SsaDefSpec::tmp(MockType::U16),
                    SsaDefSpec::tmp(MockType::U16),
                    zero,
                    counter,
                    divisor,
                    true,
                )?;
            }
            other => {
                let divisor = match other {
                    Arithmetic::Div => one,
                    Arithmetic::Zero => zero,
                    Arithmetic::Unknown => dynamic,
                    Arithmetic::LaterZero => {
                        block.sub(SsaDefSpec::tmp(MockType::U16), one, counter)?
                    }
                    _ => two,
                };
                if matches!(other, Arithmetic::Div) {
                    block.div(SsaDefSpec::tmp(MockType::U16), counter, divisor, true)?;
                } else {
                    block.rem(SsaDefSpec::tmp(MockType::U16), counter, divisor, true)?;
                }
                if matches!(other, Arithmetic::ConsumerInside) {
                    block.call_void(Callee::Direct(9), vec![address])?;
                }
            }
        }
        let next = block.add(SsaDefSpec::tmp(MockType::U16), counter, one)?;
        block.branch_cmp(next, two, CmpKind::Lt, true, 1, 2)?;
        Ok(next)
    })?;
    builder.add_phi_operand(1, counter, 0, zero)?;
    builder.add_phi_operand(1, counter, 1, next)?;
    builder.in_block(2, |block| {
        block.call_void(Callee::Direct(9), vec![address])?;
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

fn check(arithmetic: Arithmetic, replacement: usize, traps: usize) -> Result<()> {
    let mut function = body(arithmetic)?;
    let original = format!("{function:?}");
    let context = AnalysisContext::new(MACHINE);
    let before_host = Host::default();
    let before = StaticExecution::new(&function, &context)
        .with_host(&before_host)
        .run();
    assert!(
        before.is_complete(),
        "{arithmetic:?}: {:?}",
        before.completion()
    );
    assert_eq!(before.usage().traps, traps, "{arithmetic:?}");
    assert_eq!(
        before.decoded_loops(),
        if replacement == 1 { vec![1] } else { vec![] },
        "{arithmetic:?}"
    );
    assert_eq!(
        residualize_loops(&mut function, &before)?,
        replacement,
        "{arithmetic:?}"
    );
    if replacement == 0 {
        assert_eq!(
            format!("{function:?}"),
            original,
            "refusal must preserve all IR"
        );
    }
    assert!(
        SsaVerifier::new(&function)
            .verify(VerifyLevel::Full)
            .is_empty()
    );
    let after_host = Host::default();
    let after = StaticExecution::new(&function, &context)
        .with_host(&after_host)
        .run();
    assert!(after.is_complete());
    assert_eq!(after.usage().traps, traps);
    assert_eq!(after.returned(), before.returned());
    assert_eq!(
        *after_host.observed.borrow(),
        *before_host.observed.borrow()
    );
    if traps == 0 {
        assert_eq!(
            after.returned().and_then(|value| value.constant()),
            Some(&ConstValue::U16(2))
        );
        let count = if matches!(arithmetic, Arithmetic::ConsumerInside) {
            3
        } else {
            1
        };
        assert_eq!(
            *after_host.observed.borrow(),
            vec![Some(vec![1; 16]); count]
        );
    } else {
        assert!(after_host.observed.borrow().is_empty());
    }
    Ok(())
}

#[test]
fn varying_scalar_and_wide_arithmetic_can_rewrite() -> Result<()> {
    for arithmetic in [
        Arithmetic::Plain,
        Arithmetic::Div,
        Arithmetic::Rem,
        Arithmetic::SignedDiv,
        Arithmetic::SignedRem,
        Arithmetic::Wide,
        Arithmetic::Checked,
        Arithmetic::CheckedAdd,
    ] {
        check(arithmetic, 1, 0)?;
    }
    Ok(())
}

#[test]
fn zero_overflow_and_later_faults_cannot_disappear() -> Result<()> {
    for arithmetic in [
        Arithmetic::Zero,
        Arithmetic::LaterZero,
        Arithmetic::SignedOverflow,
        Arithmetic::CheckedOverflow,
        Arithmetic::CheckedLaterOverflow,
        Arithmetic::WideZero,
    ] {
        check(arithmetic, 0, 1)?;
    }
    Ok(())
}

#[test]
fn unknown_inputs_and_unstated_exception_rules_refuse() -> Result<()> {
    for arithmetic in [
        Arithmetic::Unknown,
        Arithmetic::CheckedUnknown,
        Arithmetic::WideUnknown,
        Arithmetic::RemainderOverflowUnknown,
    ] {
        check(arithmetic, 0, 0)?;
    }
    Ok(())
}

#[test]
fn effectful_consumer_inside_loop_still_refuses() -> Result<()> {
    check(Arithmetic::ConsumerInside, 0, 0)
}
