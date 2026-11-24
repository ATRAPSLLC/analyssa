//! Exact declared-type reads do not clone arbitrary held payloads.

use std::iter::repeat_n;

use crate::{
    PointerSize,
    analysis::{
        Machine,
        execution::{edgescalar::ScalarLayout, value::Value},
    },
    ir::{
        float::FloatFormat,
        integer::{IntegerBits, IntegerType},
        lanes::{MAX_VECTOR_BITS, VectorConst},
        value::ConstValue,
    },
    target::TypeLayout,
    testing::{MACHINE, MockTarget, MockType},
};

#[test]
fn all_attested_integer_widths_materialize_exact_bits_without_a_64_bit_cutoff() {
    for width in 1..=128 {
        for signed in [false, true] {
            let mask = u128::MAX >> (128 - width);
            let pattern = (0x8000000000000001_0123456789abcdefu128 | (1u128 << (width - 1))) & mask;
            let ty = IntegerType::new(width, signed).unwrap();
            let layout = ScalarLayout::of(TypeLayout::integer(ty)).unwrap();
            let held = Value::<MockTarget>::partial_integer(
                IntegerBits::constant(width, pattern).unwrap(),
                false,
            );
            let constant = layout
                .read(&held, MACHINE)
                .expect("fully known supported width");
            if width == 1 && !signed {
                assert_eq!(constant, ConstValue::from_bool(pattern != 0));
            } else {
                assert_eq!(
                    constant.scalar_pattern_at_width(width, MACHINE.ptr_size),
                    Some(pattern)
                );
            }
            assert_eq!(
                constant.payload_bytes(),
                if [1, 8, 16, 32, 64].contains(&width) && !(width == 1 && signed) {
                    0
                } else {
                    width.div_ceil(8) as usize
                }
            );
        }
    }
}

#[test]
fn native32_read_masks_high_carrier_bits_and_keeps_native_representation() {
    let machine = Machine {
        ptr_size: PointerSize::Bit32,
        ..MACHINE
    };
    for signed in [false, true] {
        let layout = ScalarLayout::of(TypeLayout::integer(
            IntegerType::new(32, signed).unwrap().native(),
        ))
        .unwrap();
        let value = Value::<MockTarget>::Const(ConstValue::NativeUInt(0xdeadbeef80000001));
        assert_eq!(
            layout.read(&value, machine),
            Some(if signed {
                ConstValue::NativeInt(-2147483647)
            } else {
                ConstValue::NativeUInt(0x80000001)
            })
        );
    }
}

#[test]
fn boolean_reads_the_value_bit_instead_of_nonzero_truth() {
    let layout = ScalarLayout::of(TypeLayout::of::<MockTarget>(
        &MockType::Bool,
        MACHINE.ptr_size,
    ))
    .unwrap();
    assert_eq!(
        layout.read(&Value::<MockTarget>::Const(ConstValue::U8(2)), MACHINE),
        Some(ConstValue::False)
    );
    assert_eq!(
        layout.read(&Value::<MockTarget>::Const(ConstValue::U8(3)), MACHINE),
        Some(ConstValue::True)
    );
}

#[test]
fn float_reads_keep_exact_format_nan_payload_and_signed_zero() {
    let layout = ScalarLayout::of(TypeLayout::of::<MockTarget>(
        &MockType::F64,
        MACHINE.ptr_size,
    ))
    .unwrap();
    for bits in [0u64, 1u64 << 63, 0x7ff8000000000042] {
        let constant = layout
            .read(&Value::<MockTarget>::Const(ConstValue::U64(bits)), MACHINE)
            .unwrap();
        assert_eq!(
            constant
                .as_float(FloatFormat::Double, MACHINE.ptr_size)
                .unwrap()
                .bits(),
            u128::from(bits)
        );
    }
    assert!(
        layout
            .read(&Value::<MockTarget>::Const(ConstValue::U32(1)), MACHINE)
            .is_none()
    );
    let integer =
        ScalarLayout::of(TypeLayout::integer(IntegerType::new(64, false).unwrap())).unwrap();
    assert!(
        integer
            .read(
                &Value::<MockTarget>::Const(ConstValue::float(FloatFormat::Double, 0).unwrap()),
                MACHINE
            )
            .is_none()
    );
}

#[test]
fn partial_stack_and_malformed_carriers_never_become_exact() {
    let layout =
        ScalarLayout::of(TypeLayout::integer(IntegerType::new(64, false).unwrap())).unwrap();
    let maximum_lanes = usize::try_from(MAX_VECTOR_BITS / 8).unwrap();
    assert!(VectorConst::from_lanes(8, repeat_n(1u128, maximum_lanes + 1)).is_none());
    for held in [
        Value::<MockTarget>::Unknown,
        Value::StackDerived,
        Value::partial_integer(IntegerBits::new(64, 0xff00, 1).unwrap(), false),
        Value::partial_integer(IntegerBits::constant(64, 7).unwrap(), true),
        Value::partial_integer(IntegerBits::constant(32, 7).unwrap(), false),
        Value::Const(ConstValue::DecryptedString("x".repeat(1 << 20).into())),
        Value::Const(ConstValue::Vector(
            VectorConst::from_lanes(8, repeat_n(1u128, maximum_lanes)).unwrap(),
        )),
    ] {
        assert!(layout.read(&held, MACHINE).is_none());
    }
    assert!(
        ScalarLayout::of(TypeLayout::of::<MockTarget>(
            &MockType::V4I32,
            MACHINE.ptr_size
        ))
        .is_none()
    );
    assert!(
        ScalarLayout::of(TypeLayout::of::<MockTarget>(
            &MockType::Ptr,
            MACHINE.ptr_size
        ))
        .is_none()
    );
    let wide = ScalarLayout::of(TypeLayout::of::<MockTarget>(
        &MockType::U128,
        MACHINE.ptr_size,
    ))
    .unwrap();
    let bits = 1u128 << 100;
    let carrier = Value::<MockTarget>::Const(ConstValue::Vector(
        VectorConst::from_lanes(128, [bits]).unwrap(),
    ));
    assert_eq!(
        wide.read(&carrier, MACHINE)
            .unwrap()
            .scalar_pattern_at_width(128, MACHINE.ptr_size),
        Some(bits)
    );
}

#[test]
fn bfloat_and_extended_float_require_the_declared_format() {
    for (ty, format, bits) in [
        (MockType::BF16, FloatFormat::BFloat16, 0x7fc2u128),
        (MockType::F80, FloatFormat::X87Extended, 1u128 << 79),
    ] {
        let layout = ScalarLayout::of(TypeLayout::of::<MockTarget>(&ty, MACHINE.ptr_size)).unwrap();
        let value = Value::<MockTarget>::Const(ConstValue::float(format, bits).unwrap());
        assert_eq!(
            layout
                .read(&value, MACHINE)
                .unwrap()
                .as_float(format, MACHINE.ptr_size)
                .unwrap()
                .bits(),
            bits
        );
    }
    let layout = ScalarLayout::of(TypeLayout::of::<MockTarget>(
        &MockType::BF16,
        MACHINE.ptr_size,
    ))
    .unwrap();
    assert!(
        layout
            .read(
                &Value::<MockTarget>::Const(ConstValue::float(FloatFormat::Half, 0).unwrap()),
                MACHINE
            )
            .is_none()
    );
}
