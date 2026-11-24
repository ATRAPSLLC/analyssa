//! Exceptional integer division retains attested scalar widths and representations.

use std::error::Error;

use analyssa::{
    PointerSize,
    analysis::consts::{Destination, Machine, OperandLayout},
    ir::{
        division::DivisionRules, evaluation::Evaluation, float::FloatRules, integer::IntegerType,
        ops::SsaOp, value::ConstValue, variable::SsaVarId,
    },
    testing::MockTarget,
};

type Value = ConstValue<MockTarget>;
type TestResult<T = ()> = Result<T, Box<dyn Error>>;

struct DivisionCase {
    layout: IntegerType,
    pointer: PointerSize,
    rules: DivisionRules,
    unsigned: bool,
    remainder: bool,
}

impl DivisionCase {
    fn evaluate(&self, dividend: Option<Value>, divisor: Value) -> TestResult<Evaluation<Value>> {
        let left = SsaVarId::from_index(0);
        let right = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op = if self.remainder {
            SsaOp::Rem {
                dest,
                left,
                right,
                unsigned: self.unsigned,
                flags: None,
            }
        } else {
            SsaOp::Div {
                dest,
                left,
                right,
                unsigned: self.unsigned,
                flags: None,
            }
        };
        let destination = Destination {
            bits: Some(self.layout.bits()),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: Some(self.layout),
        };
        Machine::new(self.pointer, None, FloatRules::NONE)
            .with_division_rules(self.rules)
            .evaluate_control_with_operand_layouts(
                &op,
                destination,
                |id| {
                    if id == left {
                        dividend.clone()
                    } else {
                        Some(divisor.clone())
                    }
                },
                |_| OperandLayout::Register(destination),
            )
            .ok_or_else(|| "scalar division has no control outcome".into())
    }
}

fn pattern(outcome: Evaluation<Value>, pointer: PointerSize) -> TestResult<(u128, u32)> {
    let Evaluation::Value(value) = outcome else {
        return Err(format!("expected established scalar value, got {outcome:?}").into());
    };
    match value {
        Value::False => Ok((0, 1)),
        Value::True => Ok((1, 1)),
        _ => value
            .scalar_integer_pattern(pointer)
            .ok_or_else(|| "scalar integer result missing".into()),
    }
}

#[test]
fn known_and_unknown_dividends_produce_zero_and_all_ones_at_scalar_lane_width() -> TestResult {
    for width in [1, 128] {
        let maximum = u128::MAX >> (128 - width);
        for signed in [false, true] {
            for (rules, expected) in [(DivisionRules::ARM, 0), (DivisionRules::RISCV, maximum)] {
                let case = DivisionCase {
                    layout: IntegerType::new(width, signed).ok_or("integer type")?,
                    pointer: PointerSize::Bit64,
                    rules,
                    unsigned: !signed,
                    remainder: false,
                };
                for known in [false, true] {
                    let dividend = if known {
                        Some(Value::of_lane(width, maximum).ok_or("dividend lane")?)
                    } else {
                        None
                    };
                    let result =
                        case.evaluate(dividend, Value::of_lane(width, 0).ok_or("scalar lane")?)?;
                    assert_eq!(pattern(result, case.pointer)?, (expected, width));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn signed_minimum_remainder_rules_preserve_scalar_lane_width() -> TestResult {
    for width in [1, 128] {
        let minimum = 1u128 << (width - 1);
        let minus_one = u128::MAX >> (128 - width);
        for rules in [
            DivisionRules::ARM,
            DivisionRules::RISCV,
            DivisionRules::CHECKED,
        ] {
            let case = DivisionCase {
                layout: IntegerType::new(width, true).ok_or("integer type")?,
                pointer: PointerSize::Bit64,
                rules,
                unsigned: false,
                remainder: true,
            };
            let result = case.evaluate(
                Value::of_lane(width, minimum),
                Value::of_lane(width, minus_one).ok_or("scalar lane")?,
            )?;
            if rules == DivisionRules::CHECKED {
                assert_eq!(result, Evaluation::Unknown);
            } else {
                assert_eq!(pattern(result, case.pointer)?, (0, width));
            }
        }
    }
    Ok(())
}

#[test]
fn primitive_and_native_exception_values_preserve_declared_carriers() -> TestResult {
    for (layout, pointer, dividend, divisor, zero, all_ones) in [
        (
            IntegerType::new(8, true).ok_or("integer type")?,
            PointerSize::Bit64,
            Value::I8(7),
            Value::I8(0),
            Value::I8(0),
            Value::I8(-1),
        ),
        (
            IntegerType::new(8, false).ok_or("integer type")?,
            PointerSize::Bit64,
            Value::U8(7),
            Value::U8(0),
            Value::U8(0),
            Value::U8(u8::MAX),
        ),
        (
            IntegerType::new(32, true).ok_or("integer type")?.native(),
            PointerSize::Bit32,
            Value::NativeInt(7),
            Value::NativeInt(0),
            Value::NativeInt(0),
            Value::NativeInt(-1),
        ),
        (
            IntegerType::new(32, false).ok_or("integer type")?.native(),
            PointerSize::Bit32,
            Value::NativeUInt(7),
            Value::NativeUInt(0),
            Value::NativeUInt(0),
            Value::NativeUInt(u64::from(u32::MAX)),
        ),
        (
            IntegerType::new(64, true).ok_or("integer type")?.native(),
            PointerSize::Bit64,
            Value::NativeInt(7),
            Value::NativeInt(0),
            Value::NativeInt(0),
            Value::NativeInt(-1),
        ),
        (
            IntegerType::new(64, false).ok_or("integer type")?.native(),
            PointerSize::Bit64,
            Value::NativeUInt(7),
            Value::NativeUInt(0),
            Value::NativeUInt(0),
            Value::NativeUInt(u64::MAX),
        ),
    ] {
        for (rules, expected) in [(DivisionRules::ARM, zero), (DivisionRules::RISCV, all_ones)] {
            let case = DivisionCase {
                layout,
                pointer,
                rules,
                unsigned: !layout.signed(),
                remainder: false,
            };
            // Enum equality checks signed/native boxing, not just bit patterns.
            assert_eq!(
                case.evaluate(Some(dividend.clone()), divisor.clone())?,
                Evaluation::Value(expected)
            );
        }
    }
    Ok(())
}
