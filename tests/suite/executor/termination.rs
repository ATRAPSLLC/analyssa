//! Guaranteed stops exclude suffix definitions and preserve surviving normal returns.

use std::error::Error;

use analyssa::{
    analysis::{
        ConstEvaluator, SsaEvaluator, SsaVerifier, VerifyLevel,
        context::AnalysisContext,
        execution::{CallModel, ExecutionHost, LibraryModel, StaticExecution, Value},
    },
    ir::{
        CallData, Callee, SsaOp, SsaVarId,
        division::DivisionRules,
        float::FloatFormat,
        function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
        ops::{CallEffects, CallTermination, CmpKind},
        value::ConstValue,
    },
    testing::{MACHINE, MockTarget, MockType},
};

/// One immutable body with its suffix and arithmetic result identities.
type Trial = (SsaFunction<MockTarget>, SsaVarId, Vec<SsaVarId>);

/// Arithmetic operations with one or two numeric outputs.
#[derive(Clone, Copy, Debug)]
enum Division {
    Quotient,
    Remainder,
    Wide,
}

/// Builds a division followed by a return, optionally beside an independent return path.
fn division(
    kind: Division,
    unsigned: bool,
    values: [u32; 3],
    dynamic: bool,
    branch: bool,
    unknown_left: bool,
) -> Result<Trial, Box<dyn Error>> {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
    if branch {
        builder.in_block(0, |block| {
            let condition = block.load_arg(SsaDefSpec::argument(0, MockType::Bool), 0)?;
            block.branch(condition, 1, 2)
        })?;
    }
    let ty = if unsigned {
        MockType::U32
    } else {
        MockType::I32
    };
    let constant = |bits: u32| {
        if unsigned {
            ConstValue::U32(bits)
        } else {
            ConstValue::I32(bits.cast_signed())
        }
    };
    let (suffix, outputs) = builder.in_block(usize::from(branch), |block| {
        let [low, high, divisor] = values;
        let low = if unknown_left {
            block.load_arg(SsaDefSpec::argument(0, ty), 0)?
        } else {
            block.const_value(SsaDefSpec::tmp(ty), constant(low))?
        };
        let high = block.const_value(SsaDefSpec::tmp(ty), constant(high))?;
        let divisor = if dynamic {
            block.load_arg(SsaDefSpec::argument(0, ty), 0)?
        } else {
            block.const_value(SsaDefSpec::tmp(ty), constant(divisor))?
        };
        let outputs = match kind {
            Division::Quotient => {
                vec![block.div(SsaDefSpec::tmp(ty), low, divisor, unsigned)?]
            }
            Division::Remainder => {
                vec![block.rem(SsaDefSpec::tmp(ty), low, divisor, unsigned)?]
            }
            Division::Wide => {
                let (quotient, remainder) = block.wide_div(
                    SsaDefSpec::tmp(ty),
                    SsaDefSpec::tmp(ty),
                    high,
                    low,
                    divisor,
                    unsigned,
                )?;
                vec![quotient, remainder]
            }
        };
        let suffix = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(0))?;
        block.ret(Some(suffix))?;
        Ok((suffix, outputs))
    })?;
    if branch {
        builder.in_block(2, |block| {
            let value = block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(42))?;
            block.ret(Some(value))
        })?;
    }
    let function = builder.finish()?;
    assert!(
        SsaVerifier::new(&function)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    Ok((function, suffix, outputs))
}

/// Zero divisors and signed or wide quotient overflow stop only their own execution paths.
#[test]
fn division_faults_exclude_suffixes_and_keep_other_returns() -> Result<(), Box<dyn Error>> {
    for kind in [Division::Quotient, Division::Remainder, Division::Wide] {
        for unsigned in [false, true] {
            for branch in [false, true] {
                let mut cases = vec![([8, 0, 0], true), ([8, 0, 2], false)];
                if matches!(kind, Division::Wide) {
                    cases.push(([0, 1, 1], true));
                    cases.push(([0, 0x8000_0000, u32::MAX], !unsigned));
                    cases.push(([u32::MAX.wrapping_sub(7), u32::MAX, 2], unsigned));
                } else {
                    cases.push((
                        [0x8000_0000, 0, u32::MAX],
                        !unsigned && !matches!(kind, Division::Remainder),
                    ));
                }
                for (values, stops) in cases {
                    let (function, suffix, outputs) =
                        division(kind, unsigned, values, false, branch, false)?;
                    let context = AnalysisContext::new(MACHINE);
                    let facts = StaticExecution::new(&function, &context).run();
                    assert!(facts.is_complete());
                    assert_eq!(
                        facts.usage().traps,
                        usize::from(stops),
                        "{kind:?} {unsigned} {values:?}"
                    );
                    assert_eq!(
                        facts.constant(suffix),
                        (!stops).then_some(&ConstValue::I32(0))
                    );
                    let expected = if branch && !stops {
                        None
                    } else if branch {
                        Some(ConstValue::I32(42))
                    } else if stops {
                        None
                    } else {
                        Some(ConstValue::I32(0))
                    };
                    assert_eq!(
                        facts.returned().and_then(Value::constant),
                        expected.as_ref()
                    );
                    if !stops {
                        let known = unsigned
                            || !matches!(kind, Division::Remainder)
                            || values != [0x8000_0000, 0, u32::MAX];
                        assert!(
                            outputs
                                .iter()
                                .all(|value| facts.constant(*value).is_some() == known)
                        );
                    }
                }
            }
            let (function, suffix, outputs) =
                division(kind, unsigned, [8, 0, 0], true, false, false)?;
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&function, &context).run();
            assert!(facts.is_complete());
            assert_eq!(facts.usage().traps, 0);
            assert_eq!(facts.constant(suffix), Some(&ConstValue::I32(0)));
            assert!(outputs.iter().all(|value| facts.constant(*value).is_none()));
        }
    }
    Ok(())
}

/// A known model supplies an independent control for declared termination.
struct Host;

impl ExecutionHost<MockTarget> for Host {
    fn model(&self, _method: &u32) -> Option<&dyn CallModel<MockTarget>> {
        Some(&LibraryModel::NoWrites)
    }
}

/// Declared non-returning calls have no suffix even when another branch returns normally.
#[test]
fn declared_termination_preserves_normal_return_alternatives() -> Result<(), Box<dyn Error>> {
    for termination in [
        CallTermination::Returns,
        CallTermination::MayNotReturn,
        CallTermination::NeverReturns,
    ] {
        for branch in [false, true] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            if branch {
                builder.in_block(0, |block| {
                    let condition = block.load_arg(SsaDefSpec::argument(0, MockType::Bool), 0)?;
                    block.branch(condition, 1, 2)
                })?;
            }
            let suffix = builder.in_block(usize::from(branch), |block| {
                let mut effects = CallEffects::pure();
                effects.termination = termination;
                block.emit_no_defs(SsaOp::Call(Box::new(CallData {
                    dest: None,
                    callee: Callee::Direct(77),
                    args: vec![],
                    effects: Some(effects),
                    tail: false,
                    implicit_inputs: Default::default(),
                    results: Default::default(),
                    staged: vec![],
                })))?;
                let suffix =
                    block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(0))?;
                block.ret(Some(suffix))?;
                Ok(suffix)
            })?;
            if branch {
                builder.in_block(2, |block| {
                    let value =
                        block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(42))?;
                    block.ret(Some(value))
                })?;
            }
            let function = builder.finish()?;
            assert!(
                SsaVerifier::new(&function)
                    .verify(VerifyLevel::Standard)
                    .is_empty()
            );
            for host in [&() as &dyn ExecutionHost<MockTarget>, &Host] {
                let context = AnalysisContext::new(MACHINE);
                let facts = StaticExecution::new(&function, &context)
                    .with_host(host)
                    .run();
                assert!(facts.is_complete());
                let stops = termination == CallTermination::NeverReturns;
                assert_eq!(
                    facts.constant(suffix),
                    (!stops).then_some(&ConstValue::I32(0))
                );
                let expected = if branch && !stops {
                    None
                } else if branch {
                    Some(ConstValue::I32(42))
                } else if stops {
                    None
                } else {
                    Some(ConstValue::I32(0))
                };
                assert_eq!(
                    facts.returned().and_then(Value::constant),
                    expected.as_ref()
                );
            }
        }
    }
    Ok(())
}

/// Machine policy decides exceptional values in both execution and ordinary constant folding.
#[test]
fn scalar_division_uses_attested_machine_rules() -> Result<(), Box<dyn Error>> {
    for rules in [
        DivisionRules::CHECKED,
        DivisionRules::ARM,
        DivisionRules::RISCV,
        DivisionRules::UNSTATED,
    ] {
        for kind in [Division::Quotient, Division::Remainder] {
            for unsigned in [false, true] {
                for values in [[8, 0, 0], [0x8000_0000, 0, u32::MAX], [8, 0, 2]] {
                    let (function, suffix, outputs) =
                        division(kind, unsigned, values, false, false, false)?;
                    let machine = MACHINE.with_division_rules(rules);
                    let context = AnalysisContext::new(machine);
                    let facts = StaticExecution::new(&function, &context).run();
                    assert!(facts.is_complete());
                    let remainder = matches!(kind, Division::Remainder);
                    let zero = values[2] == 0;
                    let overflow = !unsigned && values[0] == 0x8000_0000;
                    let traps = rules == DivisionRules::CHECKED && (zero || overflow && !remainder);
                    let expected = if zero {
                        if rules == DivisionRules::ARM {
                            Some(if remainder { 8 } else { 0 })
                        } else if rules == DivisionRules::RISCV {
                            Some(if remainder { 8 } else { u32::MAX })
                        } else {
                            None
                        }
                    } else if overflow {
                        if rules == DivisionRules::ARM || rules == DivisionRules::RISCV {
                            Some(if remainder { 0 } else { 0x8000_0000 })
                        } else {
                            None
                        }
                    } else if values[0] == 8 {
                        Some(if remainder { 0 } else { 4 })
                    } else {
                        Some(if remainder { 0x8000_0000 } else { 0 })
                    };
                    assert_eq!(facts.usage().traps, usize::from(traps));
                    assert_eq!(facts.constant(suffix).is_none(), traps);
                    let output = *outputs.first().ok_or("numeric result")?;
                    let pattern = |value: &ConstValue<MockTarget>| {
                        value
                            .scalar_integer_pattern(machine.ptr_size)
                            .map(|(value, _)| value as u32)
                    };
                    assert_eq!(
                        facts.constant(output).and_then(pattern),
                        expected,
                        "{rules:?} {kind:?} {unsigned} {values:?}"
                    );
                    assert_eq!(
                        ConstEvaluator::new(&function, machine)
                            .evaluate_var(output)
                            .as_ref()
                            .and_then(pattern),
                        expected
                    );
                }
            }
        }
    }
    Ok(())
}

/// Scalar float arithmetic honors flushing while exact normal controls retain their values.
#[test]
fn floating_arithmetic_keeps_machine_rules() -> Result<(), Box<dyn Error>> {
    for flush in [false, true] {
        for (kind, left, right, expected) in [
            (0, 1, 0, Some(1)),
            (1, 1, 0, Some(1)),
            (2, 1, 0x3ff0_0000_0000_0000, Some(1)),
            (3, 1, 0x3ff0_0000_0000_0000, Some(1)),
            (4, 1, 0x3ff0_0000_0000_0000, Some(1)),
            (
                2,
                0x0010_0000_0000_0000,
                0x3fe0_0000_0000_0000,
                Some(0x0008_0000_0000_0000),
            ),
            (
                3,
                0x0010_0000_0000_0000,
                0x4000_0000_0000_0000,
                Some(0x0008_0000_0000_0000),
            ),
            (3, 0x3ff0_0000_0000_0000, 0, Some(0x7ff0_0000_0000_0000)),
            (3, 0, 0, None),
        ] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
            let result = builder.in_block(0, |block| {
                let left = block.const_value(
                    SsaDefSpec::tmp(MockType::F64),
                    ConstValue::from(f64::from_bits(left)),
                )?;
                let right = block.const_value(
                    SsaDefSpec::tmp(MockType::F64),
                    ConstValue::from(f64::from_bits(right)),
                )?;
                let result = match kind {
                    0 => block.add(SsaDefSpec::tmp(MockType::F64), left, right)?,
                    1 => block.sub(SsaDefSpec::tmp(MockType::F64), left, right)?,
                    2 => block.mul(SsaDefSpec::tmp(MockType::F64), left, right)?,
                    3 => block.div(SsaDefSpec::tmp(MockType::F64), left, right, false)?,
                    _ => block.rem(SsaDefSpec::tmp(MockType::F64), left, right, false)?,
                };
                block.ret(Some(result))?;
                Ok(result)
            })?;
            let function = builder.finish()?;
            let mut machine = MACHINE;
            machine.float.program_flushes_subnormals = flush;
            let context = AnalysisContext::new(machine);
            let facts = StaticExecution::new(&function, &context).run();
            assert!(facts.is_complete());
            assert_eq!(facts.usage().traps, 0);
            let expected = expected
                .filter(|bits| !flush || *bits == 0x7ff0_0000_0000_0000)
                .map(|bits| ConstValue::from(f64::from_bits(bits)));
            assert_eq!(
                facts.constant(result),
                expected.as_ref(),
                "kind={kind} flush={flush}"
            );
            assert_eq!(
                ConstEvaluator::new(&function, machine)
                    .evaluate_var(result)
                    .as_ref(),
                expected.as_ref()
            );
        }
    }
    Ok(())
}

/// The x87 extended format is unaffected by interchange-format flush modes.
#[test]
fn extended_division_retains_subnormal_values() -> Result<(), Box<dyn Error>> {
    let tiny = ConstValue::float(FloatFormat::X87Extended, 1).ok_or("extended tiny")?;
    let one = ConstValue::float(FloatFormat::X87Extended, 0x3fff_8000_0000_0000_0000)
        .ok_or("extended one")?;
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let result = builder.in_block(0, |block| {
        let left = block.const_value(SsaDefSpec::tmp(MockType::F80), tiny.clone())?;
        let right = block.const_value(SsaDefSpec::tmp(MockType::F80), one)?;
        let result = block.div(SsaDefSpec::tmp(MockType::F80), left, right, false)?;
        block.ret(Some(result))?;
        Ok(result)
    })?;
    let function = builder.finish()?;
    let mut machine = MACHINE;
    machine.float.program_flushes_subnormals = true;
    let context = AnalysisContext::new(machine);
    let facts = StaticExecution::new(&function, &context).run();
    assert!(facts.is_complete());
    assert_eq!(facts.constant(result), Some(&tiny));
    Ok(())
}

/// A zero divisor decides checked faults and fixed native quotients without a known dividend.
#[test]
fn zero_divisors_need_no_dividend_constant() -> Result<(), Box<dyn Error>> {
    for rules in [
        DivisionRules::CHECKED,
        DivisionRules::ARM,
        DivisionRules::RISCV,
        DivisionRules::UNSTATED,
    ] {
        for kind in [Division::Quotient, Division::Remainder, Division::Wide] {
            let (function, suffix, outputs) = division(kind, false, [0, 0, 0], false, false, true)?;
            let machine = MACHINE.with_division_rules(rules);
            let context = AnalysisContext::new(machine);
            let facts = StaticExecution::new(&function, &context).run();
            assert!(facts.is_complete());
            let traps = rules == DivisionRules::CHECKED || matches!(kind, Division::Wide);
            let expected = match kind {
                Division::Quotient if rules == DivisionRules::ARM => Some(0),
                Division::Quotient if rules == DivisionRules::RISCV => Some(u32::MAX),
                _ => None,
            };
            assert_eq!(
                facts.usage().traps,
                usize::from(traps),
                "{rules:?} {kind:?}"
            );
            assert_eq!(facts.constant(suffix).is_none(), traps);
            let output = *outputs.first().ok_or("numeric result")?;
            let pattern = |value: &ConstValue<MockTarget>| {
                value
                    .scalar_integer_pattern(machine.ptr_size)
                    .map(|(value, _)| value as u32)
            };
            assert_eq!(facts.constant(output).and_then(pattern), expected);
            assert_eq!(
                ConstEvaluator::new(&function, machine)
                    .evaluate_var(output)
                    .as_ref()
                    .and_then(pattern),
                expected
            );
        }
    }
    Ok(())
}

/// Both comparison forms keep all float-mode outcomes and retain normal and x87 decisions.
#[test]
fn floating_comparisons_preserve_flush_alternatives() -> Result<(), Box<dyn Error>> {
    for extended in [false, true] {
        for subnormal in [false, true] {
            for flush in [false, true] {
                for direct in [false, true] {
                    for cmp in [CmpKind::Eq, CmpKind::Gt] {
                        let ty = if extended {
                            MockType::F80
                        } else {
                            MockType::F64
                        };
                        let left = if extended {
                            ConstValue::float(
                                FloatFormat::X87Extended,
                                if subnormal {
                                    1
                                } else {
                                    0x4000_8000_0000_0000_0000
                                },
                            )
                            .ok_or("extended input")?
                        } else {
                            ConstValue::from(if subnormal { f64::from_bits(1) } else { 2.0f64 })
                        };
                        let right = if extended {
                            ConstValue::float(FloatFormat::X87Extended, 0).ok_or("extended zero")?
                        } else {
                            ConstValue::from(0.0f64)
                        };
                        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
                        let comparison = builder.in_block(0, |block| {
                            let left = block.const_value(SsaDefSpec::tmp(ty), left)?;
                            let right = block.const_value(SsaDefSpec::tmp(ty), right)?;
                            if direct {
                                block.branch_cmp(left, right, cmp, false, 1, 2)?;
                                Ok(None)
                            } else {
                                let comparison = if cmp == CmpKind::Eq {
                                    block.ceq(SsaDefSpec::tmp(MockType::Bool), left, right)?
                                } else {
                                    block.cgt(
                                        SsaDefSpec::tmp(MockType::Bool),
                                        left,
                                        right,
                                        false,
                                    )?
                                };
                                block.branch(comparison, 1, 2)?;
                                Ok(Some(comparison))
                            }
                        })?;
                        for (block, value) in [(1, 42), (2, 13)] {
                            builder.in_block(block, |block| {
                                let value = block.const_value(
                                    SsaDefSpec::tmp(MockType::I32),
                                    ConstValue::I32(value),
                                )?;
                                block.ret(Some(value))
                            })?;
                        }
                        let function = builder.finish()?;
                        assert!(
                            SsaVerifier::new(&function)
                                .verify(VerifyLevel::Standard)
                                .is_empty()
                        );
                        let mut machine = MACHINE;
                        machine.float.program_flushes_subnormals = flush;
                        let context = AnalysisContext::new(machine);
                        let facts = StaticExecution::new(&function, &context).run();
                        assert!(facts.is_complete());
                        let ambiguous = subnormal && flush && !extended;
                        let expected = (!ambiguous)
                            .then_some(ConstValue::I32(if cmp == CmpKind::Eq { 13 } else { 42 }));
                        assert_eq!(
                            facts.returned().and_then(Value::constant),
                            expected.as_ref(),
                            "{extended} {subnormal} {flush} {direct} {cmp:?}"
                        );
                        let mut evaluator = SsaEvaluator::new(&function, machine);
                        evaluator.evaluate_block(0);
                        assert_eq!(
                            evaluator.next_block(0).target(),
                            (!ambiguous).then_some(if cmp == CmpKind::Eq { 2 } else { 1 })
                        );
                        if let Some(comparison) = comparison {
                            assert_eq!(
                                ConstEvaluator::new(&function, machine)
                                    .evaluate_var(comparison)
                                    .is_none(),
                                ambiguous
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Direct floating branches preserve unordered relations rather than complementing an ordered test.
#[test]
fn floating_nan_branches_keep_unordered_semantics() -> Result<(), Box<dyn Error>> {
    for unsigned in [false, true] {
        for cmp in [
            CmpKind::Eq,
            CmpKind::Ne,
            CmpKind::Lt,
            CmpKind::Le,
            CmpKind::Gt,
            CmpKind::Ge,
        ] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
            builder.in_block(0, |block| {
                let left = block
                    .const_value(SsaDefSpec::tmp(MockType::F64), ConstValue::from(f64::NAN))?;
                let right =
                    block.const_value(SsaDefSpec::tmp(MockType::F64), ConstValue::from(0.0f64))?;
                block.branch_cmp(left, right, cmp, unsigned, 1, 2)
            })?;
            for (block, value) in [(1, 42), (2, 13)] {
                builder.in_block(block, |block| {
                    let value = block
                        .const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(value))?;
                    block.ret(Some(value))
                })?;
            }
            let function = builder.finish()?;
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&function, &context).run();
            assert!(facts.is_complete());
            let taken = match cmp {
                CmpKind::Eq => false,
                CmpKind::Ne => true,
                _ => unsigned,
            };
            assert_eq!(
                facts.returned().and_then(Value::constant),
                Some(&ConstValue::I32(if taken { 42 } else { 13 }))
            );
        }
    }
    Ok(())
}

/// Equality observations cannot replace a float input with one sign of zero or a flushable value.
#[test]
fn floating_constraints_preserve_original_input_bits() -> Result<(), Box<dyn Error>> {
    for flush in [false, true] {
        for bound in [0.0f64, -0.0f64, f64::from_bits(1), 2.0f64] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
            let (input, comparison) = builder.in_block(0, |block| {
                let input = block.load_arg(SsaDefSpec::argument(0, MockType::F64), 0)?;
                let bound =
                    block.const_value(SsaDefSpec::tmp(MockType::F64), ConstValue::from(bound))?;
                let comparison = block.ceq(SsaDefSpec::tmp(MockType::Bool), input, bound)?;
                block.ret(Some(comparison))?;
                Ok((input, comparison))
            })?;
            let function = builder.finish()?;
            let mut machine = MACHINE;
            machine.float.program_flushes_subnormals = flush;
            let mut evaluator = SsaEvaluator::new(&function, machine);
            evaluator.evaluate_block(0);
            let expected = bound != 0.0 && (!flush || bound == 2.0);
            assert_eq!(
                evaluator.apply_branch_constraint(comparison, true),
                expected,
                "flush={flush} bound={bound:?}"
            );
            assert_eq!(
                evaluator.get_concrete(input),
                expected.then_some(&ConstValue::from(bound))
            );
        }
    }
    Ok(())
}
