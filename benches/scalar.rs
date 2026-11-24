//! Isolated scalar register kernels through the shared machine evaluator.
//!
//! Each case checks bytes against an integer oracle before timing. An optional
//! group selects permutations, counts, division, comparisons or wide scalars.

use std::{env, error::Error, hint::black_box, io, time::Instant};

use analyssa::{
    PointerSize,
    analysis::{Machine, consts::Destination},
    ir::{
        evaluation::NumericOp, lanes::VectorConst, ops::SsaOp, value::ConstValue,
        variable::SsaVarId,
    },
    target::Endianness,
    testing::{MACHINE, MockTarget},
};

/// One operation and its independently computed expected byte pattern.
struct Case {
    /// The measured kernel family.
    group: &'static str,
    /// A printable operation name.
    name: &'static str,
    /// The operation to dispatch.
    op: SsaOp<MockTarget>,
    /// Known variable values, in dense variable order.
    inputs: Vec<ConstValue<MockTarget>>,
    /// The operation's output width.
    width: u32,
    /// Expected result, encoded at the output width.
    expected: Vec<u8>,
}

impl Case {
    /// Evaluates one kernel with the normal shared dispatch and operand resolver.
    fn evaluate(&self, machine: Machine) -> Option<ConstValue<MockTarget>> {
        machine.evaluate(
            black_box(&self.op),
            Destination {
                bits: Some(self.width),
                lanes: None,
                lane_format: None,
                integer: true,
                integer_type: None,
            },
            |var| black_box(&self.inputs).get(var.index()).cloned(),
        )
    }
}

/// Returns a packed scalar fixture with its complete register pattern.
fn wide(value: u128) -> Result<ConstValue<MockTarget>, Box<dyn Error>> {
    VectorConst::from_lanes(128, [value])
        .map(ConstValue::Vector)
        .ok_or_else(|| io::Error::other("invalid wide fixture").into())
}

/// Adds one primitive or wide kernel with a byte oracle independent of dispatch.
fn case(
    cases: &mut Vec<Case>,
    group: &'static str,
    name: &'static str,
    op: SsaOp<MockTarget>,
    inputs: Vec<ConstValue<MockTarget>>,
    width: u32,
    expected: u128,
) {
    cases.push(Case {
        group,
        name,
        op,
        inputs,
        width,
        expected: expected
            .to_le_bytes()
            .into_iter()
            .take(width.div_ceil(8) as usize)
            .collect(),
    });
}

/// Builds primitive comparison cases and wide cases with complete integer oracles.
fn cases() -> Result<Vec<Case>, Box<dyn Error>> {
    let mut cases = Vec::new();
    let left = SsaVarId::from_index(0);
    let right = SsaVarId::from_index(1);
    let dest = SsaVarId::from_index(2);
    for width in [32u32, 64, 128] {
        let value: u128 = match width {
            32 => 0x1234_5678,
            64 => 0x1234_5678_9abc_def0,
            _ => 0x1234_5678_9abc_def0_1357_9bdf_0246_8ace,
        };
        let carrier = |value: u128| -> Result<ConstValue<MockTarget>, Box<dyn Error>> {
            Ok(match width {
                32 => ConstValue::U32(value as u32),
                64 => ConstValue::U64(value as u64),
                _ => return wide(value),
            })
        };
        let sign = 1u128 << width.saturating_sub(1);
        for (name, op, right_value, expected) in [
            (
                "eq-true",
                SsaOp::Ceq {
                    dest,
                    left,
                    right,
                    signaling: false,
                },
                sign,
                true,
            ),
            (
                "eq-false",
                SsaOp::Ceq {
                    dest,
                    left,
                    right,
                    signaling: false,
                },
                17,
                false,
            ),
            (
                "lt-signed",
                SsaOp::Clt {
                    dest,
                    left,
                    right,
                    unsigned: false,
                    signaling: false,
                },
                17,
                true,
            ),
            (
                "lt-unsigned",
                SsaOp::Clt {
                    dest,
                    left,
                    right,
                    unsigned: true,
                    signaling: false,
                },
                17,
                false,
            ),
            (
                "gt-signed",
                SsaOp::Cgt {
                    dest,
                    left,
                    right,
                    unsigned: false,
                    signaling: false,
                },
                17,
                false,
            ),
            (
                "gt-unsigned",
                SsaOp::Cgt {
                    dest,
                    left,
                    right,
                    unsigned: true,
                    signaling: false,
                },
                17,
                true,
            ),
        ] {
            case(
                &mut cases,
                if width == 128 {
                    "wide-comparisons"
                } else {
                    "comparisons"
                },
                name,
                op,
                vec![carrier(sign)?, carrier(right_value)?],
                1,
                u128::from(expected),
            );
        }
        let permute = if width == 128 { "wide" } else { "permute" };
        let counts = if width == 128 { "wide" } else { "counts" };
        let division = if width == 128 { "wide" } else { "division" };
        let swap = match width {
            32 => u128::from((value as u32).swap_bytes()),
            64 => u128::from((value as u64).swap_bytes()),
            _ => value.swap_bytes(),
        };
        let reverse = match width {
            32 => u128::from((value as u32).reverse_bits()),
            64 => u128::from((value as u64).reverse_bits()),
            _ => value.reverse_bits(),
        };
        case(
            &mut cases,
            permute,
            "swap",
            SsaOp::BSwap { dest, src: left },
            vec![carrier(value)?],
            width,
            swap,
        );
        case(
            &mut cases,
            permute,
            "reverse",
            SsaOp::BRev { dest, src: left },
            vec![carrier(value)?],
            width,
            reverse,
        );
        for (name, op, expected) in [
            (
                "bsf",
                SsaOp::BitScanForward { dest, src: left },
                value.trailing_zeros(),
            ),
            (
                "bsr",
                SsaOp::BitScanReverse { dest, src: left },
                127u32.saturating_sub(value.leading_zeros()),
            ),
            (
                "popcount",
                SsaOp::Popcount { dest, src: left },
                value.count_ones(),
            ),
            (
                "parity",
                SsaOp::Parity { dest, src: left },
                value.count_ones() % 2,
            ),
        ] {
            case(
                &mut cases,
                counts,
                name,
                op,
                vec![carrier(value)?],
                32,
                u128::from(expected),
            );
        }
        let arithmetic = if width == 128 {
            "wide-arithmetic"
        } else {
            "arithmetic"
        };
        let mask = u128::MAX >> 128u32.saturating_sub(width);
        let back = width.saturating_sub(17);
        for (name, op, expected) in [
            (
                "add",
                SsaOp::Add {
                    dest,
                    left,
                    right,
                    flags: None,
                },
                value.wrapping_add(17) & mask,
            ),
            (
                "sub",
                SsaOp::Sub {
                    dest,
                    left,
                    right,
                    flags: None,
                },
                value.wrapping_sub(17) & mask,
            ),
            (
                "mul",
                SsaOp::Mul {
                    dest,
                    left,
                    right,
                    flags: None,
                },
                value.wrapping_mul(17) & mask,
            ),
            (
                "and",
                SsaOp::And {
                    dest,
                    left,
                    right,
                    flags: None,
                },
                value & 17,
            ),
            (
                "or",
                SsaOp::Or {
                    dest,
                    left,
                    right,
                    flags: None,
                },
                value | 17,
            ),
            (
                "xor",
                SsaOp::Xor {
                    dest,
                    left,
                    right,
                    flags: None,
                },
                value ^ 17,
            ),
            (
                "shl",
                SsaOp::Shl {
                    dest,
                    value: left,
                    amount: right,
                    flags: None,
                },
                value.wrapping_shl(17) & mask,
            ),
            (
                "shr",
                SsaOp::Shr {
                    dest,
                    value: left,
                    amount: right,
                    unsigned: true,
                    flags: None,
                },
                value >> 17,
            ),
            (
                "sar",
                SsaOp::Shr {
                    dest,
                    value: left,
                    amount: right,
                    unsigned: false,
                    flags: None,
                },
                value >> 17,
            ),
            (
                "rol",
                SsaOp::Rol {
                    dest,
                    value: left,
                    amount: right,
                },
                ((value << 17) | (value >> back)) & mask,
            ),
            (
                "ror",
                SsaOp::Ror {
                    dest,
                    value: left,
                    amount: right,
                },
                ((value >> 17) | (value << back)) & mask,
            ),
            (
                "neg",
                SsaOp::Neg {
                    dest,
                    operand: left,
                    flags: None,
                },
                value.wrapping_neg() & mask,
            ),
            (
                "not",
                SsaOp::Not {
                    dest,
                    operand: left,
                    flags: None,
                },
                !value & mask,
            ),
        ] {
            case(
                &mut cases,
                arithmetic,
                name,
                op,
                vec![carrier(value)?, carrier(17)?],
                width,
                expected,
            );
        }
        for unsigned in [false, true] {
            case(
                &mut cases,
                division,
                if unsigned { "udiv" } else { "sdiv" },
                SsaOp::Div {
                    dest,
                    left,
                    right,
                    unsigned,
                    flags: None,
                },
                vec![carrier(value)?, carrier(17)?],
                width,
                value / 17,
            );
            case(
                &mut cases,
                division,
                if unsigned { "urem" } else { "srem" },
                SsaOp::Rem {
                    dest,
                    left,
                    right,
                    unsigned,
                    flags: None,
                },
                vec![carrier(value)?, carrier(17)?],
                width,
                value % 17,
            );
        }
    }
    for (name, op, first, second, expected) in [
        (
            "add",
            SsaOp::Add {
                dest,
                left,
                right,
                flags: None,
            },
            u64::MAX,
            1,
            1u128 << 64,
        ),
        (
            "sub",
            SsaOp::Sub {
                dest,
                left,
                right,
                flags: None,
            },
            0,
            1,
            u128::MAX,
        ),
        (
            "mul",
            SsaOp::Mul {
                dest,
                left,
                right,
                flags: None,
            },
            u64::MAX,
            2,
            (1u128 << 65) - 2,
        ),
        (
            "shl",
            SsaOp::Shl {
                dest,
                value: left,
                amount: right,
                flags: None,
            },
            1,
            64,
            1u128 << 64,
        ),
        (
            "ror",
            SsaOp::Ror {
                dest,
                value: left,
                amount: right,
            },
            1,
            1,
            1u128 << 127,
        ),
        (
            "not",
            SsaOp::Not {
                dest,
                operand: left,
                flags: None,
            },
            0,
            0,
            u128::MAX,
        ),
        (
            "neg",
            SsaOp::Neg {
                dest,
                operand: left,
                flags: None,
            },
            1,
            0,
            u128::MAX,
        ),
    ] {
        case(
            &mut cases,
            "native128",
            name,
            op,
            vec![
                ConstValue::NativeUInt(first),
                ConstValue::NativeUInt(second),
            ],
            128,
            expected,
        );
    }
    Ok(cases)
}

/// Measures dispatch adapters against the direct carrier kernel with identical inputs.
///
/// The direct path reproduces carrier arithmetic at the expression boundary.
/// Fixed independent oracles validate both paths before timed evaluation.
fn symbolic_dispatch(shared: bool) -> Result<(), Box<dyn Error>> {
    let left = ConstValue::<MockTarget>::U32(0x1234_5678);
    let right = ConstValue::U32(7);
    for (op, expected) in [
        (NumericOp::Add, 0x1234_567fu32),
        (NumericOp::Sub, 0x1234_5671),
        (NumericOp::Mul, 0x7f6e_5d48),
        (NumericOp::Xor, 0x1234_567f),
        (NumericOp::Shl, 0x1a2b_3c00),
    ] {
        let evaluate =
            |op: NumericOp, left: &ConstValue<MockTarget>, right: &ConstValue<MockTarget>| {
                if shared {
                    op.evaluate_binary(left, right, PointerSize::Bit64)
                } else {
                    match op {
                        NumericOp::Add => left.add(right, PointerSize::Bit64),
                        NumericOp::Sub => left.sub(right, PointerSize::Bit64),
                        NumericOp::Mul => left.mul(right, PointerSize::Bit64),
                        NumericOp::Xor => left.bitwise_xor(right, PointerSize::Bit64),
                        NumericOp::Shl => left.shl(right, PointerSize::Bit64),
                        _ => None,
                    }
                }
            };
        if evaluate(op, &left, &right).and_then(|value| value.as_u64()) != Some(u64::from(expected))
        {
            return Err(io::Error::other("incorrect symbolic dispatch fixture").into());
        }
        let started = Instant::now();
        for _ in 0..2_000_000 {
            black_box(evaluate(black_box(op), black_box(&left), black_box(&right)));
        }
        println!(
            "symbolic,{shared},{op:?},2000000,{:.2}",
            started.elapsed().as_nanos() as f64 / 2_000_000.0
        );
    }
    Ok(())
}

/// Checks each oracle and times complete machine evaluations for a selected family.
fn main() -> Result<(), Box<dyn Error>> {
    let group = env::args()
        .skip(1)
        .find(|arg| arg != "--bench")
        .unwrap_or_else(|| "all".into());
    if group == "symbolic-kernel" || group == "symbolic-shared" {
        return symbolic_dispatch(group == "symbolic-shared");
    }
    if ![
        "all",
        "permute",
        "counts",
        "division",
        "wide",
        "arithmetic",
        "wide-arithmetic",
        "native128",
        "comparisons",
        "wide-comparisons",
    ]
    .contains(&group.as_str())
    {
        return Err(io::Error::other("unknown scalar benchmark group").into());
    }
    println!("group,kernel,input_bits,result_bits,iterations,ns_per_evaluation");
    for case in cases()?
        .into_iter()
        .filter(|case| group == "all" || group == case.group)
    {
        let machine = if case.group == "native128" {
            Machine {
                ptr_size: PointerSize::Bit128,
                ..MACHINE
            }
        } else {
            MACHINE
        };
        let input_bits = case
            .inputs
            .first()
            .and_then(|value| match value {
                ConstValue::Vector(vector) => Some(vector.bits()),
                _ => value.bit_width(machine.ptr_size),
            })
            .ok_or_else(|| io::Error::other("missing scalar input width"))?;
        let actual = case.evaluate(machine).and_then(|value| {
            if case.width == 1 {
                value.as_bool().map(|value| vec![u8::from(value)])
            } else {
                value.to_bytes(Endianness::Little, machine.ptr_size)
            }
        });
        if actual.as_ref() != Some(&case.expected) {
            return Err(io::Error::other(format!(
                "incorrect scalar kernel {} width {}: {actual:?}",
                case.name, case.width
            ))
            .into());
        }
        let started = Instant::now();
        for _ in 0..200_000 {
            black_box(case.evaluate(black_box(machine)));
        }
        println!(
            "{},{},{},{},200000,{:.2}",
            case.group,
            case.name,
            input_bits,
            case.width,
            started.elapsed().as_nanos() as f64 / 200_000.0
        );
    }
    Ok(())
}
