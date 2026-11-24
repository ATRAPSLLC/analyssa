use super::*;
use crate::{
    ir::ops::{
        LaneControl, VectorBinaryKind, VectorCastKind, VectorCompareKind, VectorInactivePolicy,
        VectorMaskBinaryKind, VectorMaskPrefix, VectorPredicateLayout, VectorReduceKind,
        VectorStatus, VectorUnaryKind,
    },
    target::VectorTailPolicy,
};

/// A computation of `kernel` on `element` under lane control: the tail
/// `tail` fills, inactive elements governed by `inactive` where it is given.
fn controlled(
    kernel: Kernel,
    element: VectorElement,
    rounding: VectorRounding,
    tail: VectorTailPolicy,
    inactive: Option<VectorInactivePolicy>,
) -> Computation {
    Computation {
        kernel,
        element,
        predicate: VectorPredicateLayout::Packed,
        rounding,
        status: VectorStatus::None,
        canonical_nan: false,
        lanes: Some(LaneControl {
            mask_tail: false,
            tail,
            inactive,
            no_passthrough: false,
        }),
    }
}

/// A scalar control or data input.
fn scalar(value: u128, bits: u32) -> Option<LaneOperand> {
    Some(LaneOperand::Scalar {
        value,
        bits,
        format: None,
    })
}

/// A vector input of `lanes` elements `bits` wide, declared so.
fn vector(bits: u32, lanes: &[u128]) -> Option<LaneOperand> {
    declared(bits, lanes, Some(bits))
}

/// A vector input of `lanes` elements `bits` wide whose type declares
/// elements `lane_bits` wide.
fn declared(bits: u32, lanes: &[u128], lane_bits: Option<u32>) -> Option<LaneOperand> {
    Some(LaneOperand::Vector {
        value: VectorConst::from_lanes(bits, lanes.iter().copied()).unwrap(),
        lane_bits,
        lane_format: None,
    })
}

/// The elements an execution writes, `result_bits` wide, as `bits`-wide lanes.
fn run(
    computation: Computation,
    inputs: &[Option<LaneOperand>],
    result_bits: u32,
) -> Option<Vec<u128>> {
    let execution = LaneExecution::of(computation, inputs, None, FloatRules::default())?;
    Some(execution.evaluate(result_bits)?.lane_values().collect())
}

/// Byte addition over four elements, wrapping at the element.
fn add8(tail: VectorTailPolicy, inactive: Option<VectorInactivePolicy>) -> Computation {
    controlled(
        Kernel::Binary(VectorBinaryKind::Add),
        VectorElement::integer(8, None),
        VectorRounding::None,
        tail,
        inactive,
    )
}

/// **Every element executed folds without the old value.** `vadd.vv` over
/// four bytes, unmasked, at the full length writes every element, so the
/// passthrough (here not known) is never read; `250 + 10` wraps to `4`.
#[test]
fn an_unmasked_full_length_add_folds_without_the_passthrough() {
    let inputs = [
        scalar(0, 64),
        scalar(4, 64),
        None,
        vector(8, &[1, 2, 3, 250]),
        vector(8, &[1, 1, 1, 10]),
    ];
    assert_eq!(
        run(add8(VectorTailPolicy::Agnostic, None), &inputs, 32),
        Some(vec![2, 3, 4, 4])
    );
}

/// **A masked element keeps, zeroes or folds nothing, as its policy says.**
/// Mask `0b0101` executes elements 0 and 2: undisturbed keeps the
/// passthrough's 9s, zeroing writes 0, agnostic has no single value.
#[test]
fn an_inactive_element_follows_its_policy() {
    let inputs = |passthrough| {
        [
            scalar(0, 64),
            scalar(4, 64),
            vector(8, &[0b0101]),
            passthrough,
            vector(8, &[1, 2, 3, 4]),
            vector(8, &[1, 1, 1, 1]),
        ]
    };
    let nines = vector(8, &[9, 9, 9, 9]);
    assert_eq!(
        run(
            add8(
                VectorTailPolicy::Agnostic,
                Some(VectorInactivePolicy::Undisturbed)
            ),
            &inputs(nines.clone()),
            32
        ),
        Some(vec![2, 9, 4, 9])
    );
    assert_eq!(
        run(
            add8(VectorTailPolicy::Agnostic, Some(VectorInactivePolicy::Zero)),
            &inputs(None),
            32
        ),
        Some(vec![2, 0, 4, 0])
    );
    assert_eq!(
        run(
            add8(
                VectorTailPolicy::Agnostic,
                Some(VectorInactivePolicy::Agnostic)
            ),
            &inputs(nines),
            32
        ),
        None
    );
    // An undisturbed element whose old value is not known is not known.
    assert_eq!(
        run(
            add8(
                VectorTailPolicy::Agnostic,
                Some(VectorInactivePolicy::Undisturbed)
            ),
            &inputs(None),
            32
        ),
        None
    );
}

/// **The tail and the prestart keep their old elements.** At length 3 and
/// restart 1, element 0 (prestart) and element 3 (tail, undisturbed) keep
/// the passthrough's; an agnostic tail has no single value; a restart at or
/// past the length changes nothing.
#[test]
fn the_tail_and_the_prestart_keep_their_elements() {
    let inputs = |start, length| {
        [
            scalar(start, 64),
            scalar(length, 64),
            vector(8, &[7, 7, 7, 7]),
            vector(8, &[1, 2, 3, 4]),
            vector(8, &[10, 10, 10, 10]),
        ]
    };
    assert_eq!(
        run(add8(VectorTailPolicy::Undisturbed, None), &inputs(1, 3), 32),
        Some(vec![7, 12, 13, 7])
    );
    assert_eq!(
        run(add8(VectorTailPolicy::Agnostic, None), &inputs(1, 3), 32),
        None
    );
    assert_eq!(
        run(add8(VectorTailPolicy::Agnostic, None), &inputs(3, 3), 32),
        Some(vec![7, 7, 7, 7])
    );
    assert_eq!(
        run(add8(VectorTailPolicy::Agnostic, None), &inputs(0, 0), 32),
        Some(vec![7, 7, 7, 7])
    );
}

/// **A scalar operand is broadcast at the element's width.** `vadd.vx` adds
/// the low byte of `0x105`, 5, to every element.
#[test]
fn a_scalar_operand_is_broadcast() {
    let inputs = [
        scalar(0, 64),
        scalar(4, 64),
        None,
        vector(8, &[1, 2, 3, 4]),
        scalar(0x105, 64),
    ];
    assert_eq!(
        run(add8(VectorTailPolicy::Agnostic, None), &inputs, 32),
        Some(vec![6, 7, 8, 9])
    );
}

/// **Division is total.** RISC-V `vdivu` by zero gives all ones and `vremu`
/// the dividend; `vdiv` of the least byte by minus one gives the least byte
/// and `vrem` zero; `-7 / 2` truncates to `-3`, remainder `-1`.
#[test]
fn integer_division_is_total() {
    let divide = |remainder, unsigned| {
        controlled(
            Kernel::IntegerDivide {
                remainder,
                unsigned,
            },
            VectorElement::integer(8, Some(!unsigned)),
            VectorRounding::None,
            VectorTailPolicy::Agnostic,
            None,
        )
    };
    let inputs = |left: [u128; 2], right: [u128; 2]| {
        [
            scalar(0, 64),
            scalar(2, 64),
            None,
            vector(8, &left),
            vector(8, &right),
        ]
    };
    assert_eq!(
        run(divide(false, true), &inputs([9, 200], [0, 7]), 16),
        Some(vec![0xff, 28])
    );
    assert_eq!(
        run(divide(true, true), &inputs([9, 200], [0, 7]), 16),
        Some(vec![9, 4])
    );
    assert_eq!(
        run(divide(false, false), &inputs([0x80, 0xf9], [0xff, 2]), 16),
        Some(vec![0x80, 0xfd])
    );
    assert_eq!(
        run(divide(true, false), &inputs([0x80, 0xf9], [0xff, 2]), 16),
        Some(vec![0, 0xff])
    );
}

/// **The high half of a product is read by each factor's signedness.**
/// `vmulhu` of two all-ones quadwords is `2^64 - 2` (the product,
/// `2^128 - 2^65 + 1`, overflows an i128 but not its unsigned half);
/// `vmulh` of `-1` by `-1` is 0; `vmulhsu` of `-1` by all ones is all ones.
#[test]
fn a_high_product_reads_each_factors_signedness() {
    let high = |signed_left, signed_right| {
        controlled(
            Kernel::MultiplyHigh {
                signed_left,
                signed_right,
            },
            VectorElement::integer(64, Some(signed_left)),
            VectorRounding::None,
            VectorTailPolicy::Agnostic,
            None,
        )
    };
    let ones = u128::from(u64::MAX);
    let inputs = [
        scalar(0, 64),
        scalar(1, 64),
        None,
        vector(64, &[ones]),
        vector(64, &[ones]),
    ];
    assert_eq!(run(high(false, false), &inputs, 64), Some(vec![ones - 1]));
    assert_eq!(run(high(true, true), &inputs, 64), Some(vec![0]));
    assert_eq!(run(high(true, false), &inputs, 64), Some(vec![ones]));
}

/// **A merge takes the right element where its selector bit is set.**
/// `vmerge.vvm` with selector `0b10`.
#[test]
fn a_merge_reads_its_selector() {
    let merge = controlled(
        Kernel::Select,
        VectorElement::integer(16, None),
        VectorRounding::None,
        VectorTailPolicy::Agnostic,
        None,
    );
    let inputs = [
        scalar(0, 64),
        scalar(2, 64),
        None,
        vector(16, &[1, 2]),
        vector(16, &[30, 40]),
        vector(8, &[0b10]),
    ];
    assert_eq!(run(merge, &inputs, 32), Some(vec![1, 40]));
}

/// **A fixed-point average rounds by the mode it reads.** `vaadd` of 5 and
/// 2 is 3.5: up and even give 4, down 3, odd 3.
#[test]
fn a_fixed_point_average_rounds_by_its_mode() {
    let average = controlled(
        Kernel::Fixed {
            kind: VectorFixedKind::AverageAdd,
        },
        VectorElement::integer(8, Some(true)),
        VectorRounding::FixedEnvironment,
        VectorTailPolicy::Agnostic,
        None,
    );
    let inputs = |mode| {
        [
            scalar(0, 64),
            scalar(1, 64),
            scalar(mode, 64),
            None,
            vector(8, &[5]),
            vector(8, &[2]),
        ]
    };
    for (mode, expected) in [(0, 4), (1, 4), (2, 3), (3, 3)] {
        assert_eq!(
            run(average, &inputs(mode), 8),
            Some(vec![expected]),
            "mode {mode}"
        );
    }
}

/// **A float element computes as IEEE's, by the mode the environment input
/// names.** `vfadd` of 1.5 and 2.25 is 3.75; `vfsgnjn` of 1.0 with the sign
/// of 2.0 negated is -1.0.
#[test]
fn float_elements_compute_by_the_named_mode() {
    let f32s = |values: &[f32]| {
        vector(
            32,
            &values
                .iter()
                .map(|value| u128::from(value.to_bits()))
                .collect::<Vec<_>>(),
        )
    };
    let add = controlled(
        Kernel::Binary(VectorBinaryKind::Add),
        VectorElement::float(32),
        VectorRounding::FloatingEnvironment,
        VectorTailPolicy::Agnostic,
        None,
    );
    let inputs = [
        scalar(0, 64),
        scalar(1, 64),
        scalar(0, 64),
        None,
        f32s(&[1.5]),
        f32s(&[2.25]),
    ];
    assert_eq!(
        run(add, &inputs, 32),
        Some(vec![u128::from(3.75f32.to_bits())])
    );
    let negated = controlled(
        Kernel::SignInject {
            negate: true,
            xor: false,
        },
        VectorElement::float(32),
        VectorRounding::None,
        VectorTailPolicy::Agnostic,
        None,
    );
    let inputs = [
        scalar(0, 64),
        scalar(1, 64),
        None,
        f32s(&[1.0]),
        f32s(&[2.0]),
    ];
    assert_eq!(
        run(negated, &inputs, 32),
        Some(vec![u128::from((-1.0f32).to_bits())])
    );
}

/// **A computed NaN is the canonical one where the computation says.** RISC-V
/// float arithmetic writes the canonical NaN `0x7fc00000` for any NaN it
/// computes; a NaN operand otherwise propagates as the target decides, which
/// the default rules leave unknown.
#[test]
fn a_computed_nan_is_canonical_where_stated() {
    let mut add = controlled(
        Kernel::Unary(VectorUnaryKind::Neg),
        VectorElement::float(32),
        VectorRounding::None,
        VectorTailPolicy::Agnostic,
        None,
    );
    let inputs = [
        scalar(0, 64),
        scalar(1, 64),
        None,
        vector(32, &[0xffc0_0001]),
    ];
    // Negation changes only the sign bit: of a NaN, a NaN.
    assert_eq!(run(add, &inputs, 32), Some(vec![0x7fc0_0001]));
    add.canonical_nan = true;
    assert_eq!(run(add, &inputs, 32), Some(vec![0x7fc0_0000]));
}

/// **An element not executed is never computed.** The float sum of a NaN
/// and 1.0 is the target's to decide, so it folds nothing executed; in an
/// element the mask zeroes, the execution folds all the same.
#[test]
fn an_element_not_executed_is_never_computed() {
    let add = |inactive| {
        controlled(
            Kernel::Binary(VectorBinaryKind::Add),
            VectorElement::float(32),
            VectorRounding::None,
            VectorTailPolicy::Agnostic,
            inactive,
        )
    };
    let one = u128::from(1.0f32.to_bits());
    let nan = 0x7fc0_0001;
    let inputs = |mask| {
        [
            scalar(0, 64),
            scalar(2, 64),
            vector(8, &[mask]),
            None,
            vector(32, &[one, nan]),
            vector(32, &[one, one]),
        ]
    };
    assert_eq!(
        run(add(Some(VectorInactivePolicy::Zero)), &inputs(0b11), 64),
        None
    );
    assert_eq!(
        run(add(Some(VectorInactivePolicy::Zero)), &inputs(0b01), 64),
        Some(vec![u128::from(2.0f32.to_bits()), 0])
    );
}

/// **A vector is read only at the width its type declares.** The bits of
/// two words read as bytes are other values, so a byte addition over a
/// vector declared of words, or of no declared width, folds nothing; a
/// fused multiply-add of f32 whose factors are declared 16 bits wide
/// (`vfwmaccbf16`'s bfloat16) folds nothing rather than read them as f32.
#[test]
fn a_vector_is_read_at_its_declared_width() {
    let inputs = |lane_bits| {
        [
            scalar(0, 64),
            scalar(4, 64),
            None,
            declared(8, &[1, 2, 3, 4], lane_bits),
            vector(8, &[1, 1, 1, 1]),
        ]
    };
    let add = add8(VectorTailPolicy::Agnostic, None);
    assert_eq!(run(add, &inputs(Some(8)), 32), Some(vec![2, 3, 4, 5]));
    assert_eq!(run(add, &inputs(Some(16)), 32), None);
    assert_eq!(run(add, &inputs(None), 32), None);
    let fused = controlled(
        Kernel::FusedMultiplyAdd {
            negate_product: false,
            negate_addend: false,
        },
        VectorElement::float(32),
        VectorRounding::FloatingEnvironment,
        VectorTailPolicy::Agnostic,
        None,
    );
    let one = u128::from(1.0f32.to_bits());
    let inputs = [
        scalar(0, 64),
        scalar(1, 64),
        scalar(0, 64),
        None,
        declared(32, &[one], Some(16)),
        declared(32, &[one], Some(16)),
        vector(32, &[one]),
    ];
    assert_eq!(run(fused, &inputs, 32), None);
}

/// One execution of `kernel` on `element`, unmasked over `length` elements
/// with an agnostic tail, of `data`.
fn unmasked(
    kernel: Kernel,
    element: VectorElement,
    length: u128,
    data: &[Option<LaneOperand>],
    result_bits: u32,
) -> Option<Vec<u128>> {
    let computation = controlled(
        kernel,
        element,
        VectorRounding::None,
        VectorTailPolicy::Agnostic,
        None,
    );
    let mut inputs = vec![scalar(0, 64), scalar(length, 64), None];
    inputs.extend_from_slice(data);
    run(computation, &inputs, result_bits)
}

/// **An extension reads its source at the width its type declares.**
/// `vzext.vf2` of bytes `0xff, 1` to halfwords is `0x00ff, 1`; `vsext.vf4`
/// of `0x80` to a word is `0xffff_ff80`; the same bytes declared of no width
/// fold nothing.
#[test]
fn an_extension_reads_its_declared_source() {
    let extend = |signed| Kernel::Extend { signed };
    assert_eq!(
        unmasked(
            extend(false),
            VectorElement::integer(16, None),
            2,
            &[vector(8, &[0xff, 1])],
            32
        ),
        Some(vec![0xff, 1])
    );
    assert_eq!(
        unmasked(
            extend(true),
            VectorElement::integer(32, None),
            1,
            &[vector(8, &[0x80])],
            32
        ),
        Some(vec![0xffff_ff80])
    );
    assert_eq!(
        unmasked(
            extend(true),
            VectorElement::integer(32, None),
            1,
            &[declared(8, &[0x80], None)],
            32
        ),
        None
    );
}

/// **A widening operation extends its narrow sources first.** `vwaddu.vv`
/// of bytes 200 and 100 is 300; `vwadd.vv` of `-1` and `-1` is `-2`, the
/// halfword `0xfffe`; `vwadd.wv` of the halfword 1000 and the byte `-1` is
/// 999.
#[test]
fn a_widening_operation_extends_its_sources() {
    let add = |wide_left, signed| Kernel::WidenBinary {
        wide_left,
        kind: VectorBinaryKind::Add,
        signed_left: signed,
        signed_right: signed,
    };
    let halfwords = VectorElement::integer(16, None);
    assert_eq!(
        unmasked(
            add(false, false),
            halfwords,
            2,
            &[vector(8, &[200, 1]), vector(8, &[100, 2])],
            32
        ),
        Some(vec![300, 3])
    );
    assert_eq!(
        unmasked(
            add(false, true),
            halfwords,
            1,
            &[vector(8, &[0xff]), vector(8, &[0xff])],
            16
        ),
        Some(vec![0xfffe])
    );
    assert_eq!(
        unmasked(
            add(true, true),
            halfwords,
            1,
            &[vector(16, &[1000]), vector(8, &[0xff])],
            16
        ),
        Some(vec![999])
    );
}

/// **A narrowing shift shifts the wide source by the count's low bits.**
/// `vnsrl` of `0x1234` by 4 is `0x23`, and by 20 (20 mod 16) the same;
/// `vnsra` of `0x8000` by 8 is `0x80`, the sign kept.
#[test]
fn a_narrowing_shift_reads_its_counts_low_bits() {
    let bytes = VectorElement::integer(8, None);
    for count in [4, 20] {
        assert_eq!(
            unmasked(
                Kernel::NarrowShift { unsigned: true },
                bytes,
                1,
                &[vector(16, &[0x1234]), scalar(count, 64)],
                8
            ),
            Some(vec![0x23]),
            "count {count}"
        );
    }
    assert_eq!(
        unmasked(
            Kernel::NarrowShift { unsigned: false },
            bytes,
            1,
            &[vector(16, &[0x8000]), scalar(8, 64)],
            8
        ),
        Some(vec![0x80])
    );
}

/// **A widening multiply-add accumulates the whole product.** `vwmaccu` of
/// bytes 200 and 200 into 1 is 40001; `vwmacc` of `-2` and 3 into 10 is 4.
#[test]
fn a_widening_multiply_add_accumulates_the_whole_product() {
    let fused = |signed| Kernel::WidenMultiplyAdd {
        signed_left: signed,
        signed_right: signed,
    };
    let halfwords = VectorElement::integer(16, None);
    assert_eq!(
        unmasked(
            fused(false),
            halfwords,
            1,
            &[vector(8, &[200]), vector(8, &[200]), vector(16, &[1])],
            16
        ),
        Some(vec![40001])
    );
    assert_eq!(
        unmasked(
            fused(true),
            halfwords,
            1,
            &[vector(8, &[0xfe]), vector(8, &[3]), vector(16, &[10])],
            16
        ),
        Some(vec![4])
    );
}

/// **An add with carry reads each element's carry bit.** `vadc.vvm` of
/// 255 and 1 with carry 1 wraps to 1, of 1 and 1 with carry 0 is 2;
/// `vsbc.vvm` of 0 and 1 with no borrow is 255, of 5 and 1 with borrow 1 is 3.
#[test]
fn an_add_with_carry_reads_each_carry_bit() {
    let carry = |subtract| Kernel::IntegerCarry {
        subtract,
        mask_result: false,
    };
    let bytes = VectorElement::integer(8, None);
    assert_eq!(
        unmasked(
            carry(false),
            bytes,
            2,
            &[
                vector(8, &[255, 1]),
                vector(8, &[1, 1]),
                vector(8, &[0b01, 0])
            ],
            16
        ),
        Some(vec![1, 2])
    );
    assert_eq!(
        unmasked(
            carry(true),
            bytes,
            2,
            &[
                vector(8, &[0, 5]),
                vector(8, &[1, 1]),
                vector(8, &[0b10, 0])
            ],
            16
        ),
        Some(vec![255, 3])
    );
}

/// **A conversion reads its source at its declared width and kind.**
/// `vfcvt.f.xu.v` of 3 is 3.0; `vfwcvt.f.f.v` of the single 1.5 is the
/// double 1.5; a float converted to a signed integer truncates 2.75 to 2.
#[test]
fn a_conversion_reads_its_declared_source() {
    let cast = |from, to| Kernel::Cast { from, to };
    assert_eq!(
        unmasked(
            cast(VectorCastKind::Unsigned, VectorCastKind::Float),
            VectorElement::float(32),
            1,
            &[vector(32, &[3])],
            32
        ),
        Some(vec![u128::from(3.0f32.to_bits())])
    );
    assert_eq!(
        unmasked(
            cast(VectorCastKind::Float, VectorCastKind::Float),
            VectorElement::float(64),
            1,
            &[vector(32, &[u128::from(1.5f32.to_bits())])],
            64
        ),
        Some(vec![u128::from(1.5f64.to_bits())])
    );
    assert_eq!(
        unmasked(
            cast(VectorCastKind::Float, VectorCastKind::Signed),
            VectorElement::integer(32, Some(true)),
            1,
            &[vector(32, &[u128::from(2.75f32.to_bits())])],
            32
        ),
        Some(vec![2])
    );
}

/// The packed predicate an execution writes over `result_bits` elements,
/// with `old` as its passthrough and an undisturbed tail past `length`.
fn predicate(
    kernel: Kernel,
    element: VectorElement,
    length: u128,
    old: u128,
    data: &[Option<LaneOperand>],
    result_bits: u32,
) -> Option<u128> {
    let computation = controlled(
        kernel,
        element,
        VectorRounding::None,
        VectorTailPolicy::Undisturbed,
        None,
    );
    let mut inputs = vec![
        scalar(0, 64),
        scalar(length, 64),
        Some(LaneOperand::Scalar {
            value: old,
            bits: result_bits,
            format: None,
        }),
    ];
    inputs.extend_from_slice(data);
    let bits = run(computation, &inputs, result_bits)?;
    Some(
        bits.iter()
            .enumerate()
            .map(|(index, bit)| bit << index)
            .sum(),
    )
}

/// **A comparison writes one bit per element, the tail keeping its own.**
/// `vmseq` of `[1, 5, 3, 7]` and `[1, 4, 4, 7]` is `0b1001`, the four tail
/// bits of `0xf0` kept; `vmsltu` is `0b0100`; an agnostic tail has no single
/// value.
#[test]
fn a_comparison_writes_a_bit_per_element() {
    let compare = |relation| Kernel::Compare {
        relation,
        signaling: false,
    };
    let data = [vector(8, &[1, 5, 3, 7]), vector(8, &[1, 4, 4, 7])];
    let bytes = VectorElement::integer(8, Some(false));
    assert_eq!(
        predicate(compare(VectorCompareKind::Eq), bytes, 4, 0xf0, &data, 8),
        Some(0xf9)
    );
    assert_eq!(
        predicate(compare(VectorCompareKind::Lt), bytes, 4, 0, &data, 8),
        Some(0b0100)
    );
    let agnostic = controlled(
        compare(VectorCompareKind::Eq),
        bytes,
        VectorRounding::None,
        VectorTailPolicy::Agnostic,
        None,
    );
    let mut inputs = vec![scalar(0, 64), scalar(4, 64), scalar(0, 8)];
    inputs.extend_from_slice(&data);
    assert_eq!(run(agnostic, &inputs, 8), None);
}

/// **Mask logic and carries out write one bit per element.** `vmand.mm` of
/// `0b1100` and `0b1010` is `0b1000`; `vmadc.vvm` of `[255, 1]` and `[1, 1]`
/// carries out of the first only; `vmsbc.vvm` of `[0, 5]` and `[1, 1]`
/// borrows out of the first only.
#[test]
fn mask_logic_and_carries_out_write_a_bit_per_element() {
    let mask = VectorElement::integer(1, None);
    assert_eq!(
        predicate(
            Kernel::MaskBinary(VectorMaskBinaryKind::And),
            mask,
            4,
            0,
            &[scalar(0b1100, 8), scalar(0b1010, 8)],
            4
        ),
        Some(0b1000)
    );
    let carry = |subtract| Kernel::IntegerCarry {
        subtract,
        mask_result: true,
    };
    let bytes = VectorElement::integer(8, None);
    assert_eq!(
        predicate(
            carry(false),
            bytes,
            2,
            0,
            &[vector(8, &[255, 1]), vector(8, &[1, 1]), scalar(0, 8)],
            2
        ),
        Some(0b01)
    );
    assert_eq!(
        predicate(
            carry(true),
            bytes,
            2,
            0,
            &[vector(8, &[0, 5]), vector(8, &[1, 1]), scalar(0, 8)],
            2
        ),
        Some(0b01)
    );
}

/// **A prefix and an iota scan the selected source bits.** Of the source
/// `0b0100` over four elements, `vmsbf` is `0b0011`, `vmsif` `0b0111` and
/// `vmsof` `0b0100`; of no set bit, `vmsbf` sets every element and `vmsof`
/// none. `viota.m` of `0b1011` counts `[0, 1, 2, 2]`.
#[test]
fn a_prefix_and_an_iota_scan_the_source() {
    let mask = VectorElement::integer(1, None);
    let prefix = |kind, source| {
        predicate(
            Kernel::MaskPrefix(kind),
            mask,
            4,
            0,
            &[scalar(source, 8)],
            4,
        )
    };
    assert_eq!(prefix(VectorMaskPrefix::Before, 0b0100), Some(0b0011));
    assert_eq!(prefix(VectorMaskPrefix::Including, 0b0100), Some(0b0111));
    assert_eq!(prefix(VectorMaskPrefix::Only, 0b0100), Some(0b0100));
    assert_eq!(prefix(VectorMaskPrefix::Before, 0), Some(0b1111));
    assert_eq!(prefix(VectorMaskPrefix::Only, 0), Some(0));
    assert_eq!(
        unmasked(
            Kernel::MaskIota,
            VectorElement::integer(8, None),
            4,
            &[scalar(0b1011, 8)],
            32
        ),
        Some(vec![0, 1, 2, 2])
    );
}

/// One execution over four bytes with an undisturbed tail and the old
/// elements `[9, 9, 9, 9]`, of `data`, at active length `length`.
fn kept(kernel: Kernel, length: u128, data: &[Option<LaneOperand>]) -> Option<Vec<u128>> {
    let computation = controlled(
        kernel,
        VectorElement::integer(8, None),
        VectorRounding::None,
        VectorTailPolicy::Undisturbed,
        None,
    );
    let mut inputs = vec![scalar(0, 64), scalar(length, 64), vector(8, &[9, 9, 9, 9])];
    inputs.extend_from_slice(data);
    run(computation, &inputs, 32)
}

/// **An index, an insertion and a gather read elements by position.**
/// `vid.v` is `[0, 1, 2, 3]`; `vmv.s.x` of `0x105` writes 5 to element 0
/// alone; `vrgather.vv` of `[10, 20, 30, 40]` by `[3, 0, 9, 1]` is
/// `[40, 10, 0, 20]`, an index past the source reading zero, and
/// `vrgatherei16` reads its indices as the halfwords they are declared.
#[test]
fn an_index_insertion_and_gather_read_by_position() {
    assert_eq!(kept(Kernel::Index, 4, &[]), Some(vec![0, 1, 2, 3]));
    assert_eq!(
        kept(Kernel::InsertScalar, 4, &[scalar(0x105, 64)]),
        Some(vec![5, 9, 9, 9])
    );
    let source = vector(8, &[10, 20, 30, 40]);
    assert_eq!(
        kept(
            Kernel::Gather,
            4,
            &[source.clone(), vector(8, &[3, 0, 9, 1])]
        ),
        Some(vec![40, 10, 0, 20])
    );
    assert_eq!(
        kept(Kernel::Gather, 4, &[source, vector(16, &[3, 0, 9, 1])]),
        Some(vec![40, 10, 0, 20])
    );
}

/// **A slide moves elements by its offset or one fill.** Of `[1, 2, 3, 4]`:
/// `vslideup` by 2 keeps the old 9s below the offset, `[9, 9, 1, 2]`;
/// `vslidedown` by 1 reads zero past the source, `[2, 3, 4, 0]`;
/// `vslide1up` and `vslide1down` of 7 are `[7, 1, 2, 3]` and `[2, 3, 4, 7]`.
#[test]
fn a_slide_moves_elements() {
    let source = vector(8, &[1, 2, 3, 4]);
    let slide = |up, scalar_fill| Kernel::Slide { up, scalar_fill };
    assert_eq!(
        kept(slide(true, false), 4, &[source.clone(), scalar(2, 64)]),
        Some(vec![9, 9, 1, 2])
    );
    assert_eq!(
        kept(slide(false, false), 4, &[source.clone(), scalar(1, 64)]),
        Some(vec![2, 3, 4, 0])
    );
    assert_eq!(
        kept(slide(true, true), 4, &[source.clone(), scalar(7, 64)]),
        Some(vec![7, 1, 2, 3])
    );
    assert_eq!(
        kept(slide(false, true), 4, &[source, scalar(7, 64)]),
        Some(vec![2, 3, 4, 7])
    );
}

/// **A compression packs the selected elements low.** `vcompress.vm` of
/// `[1, 2, 3, 4]` by `0b1010` is `[2, 4]`, the tail keeping its 9s.
#[test]
fn a_compression_packs_the_selected_elements() {
    assert_eq!(
        kept(
            Kernel::Compress,
            4,
            &[vector(8, &[1, 2, 3, 4]), scalar(0b1010, 8)]
        ),
        Some(vec![2, 4, 9, 9])
    );
}

/// **A reduction folds the selected elements into element zero.** `vredsum`
/// of `[1, 2, 3, 4]` seeded by 10 is 20, the tail keeping its 9s; masked by
/// `0b0101` it is 14; `vwredsumu` of the bytes `[200, 200]` into halfwords
/// is 400.
#[test]
fn a_reduction_folds_into_element_zero() {
    let sum = Kernel::Reduce {
        kind: VectorReduceKind::Add,
        ordered: false,
    };
    let data = [vector(8, &[1, 2, 3, 4]), vector(8, &[10, 0, 0, 0])];
    assert_eq!(kept(sum, 4, &data), Some(vec![20, 9, 9, 9]));
    let masked = controlled(
        sum,
        VectorElement::integer(8, None),
        VectorRounding::None,
        VectorTailPolicy::Undisturbed,
        Some(VectorInactivePolicy::Undisturbed),
    );
    let mut inputs = vec![
        scalar(0, 64),
        scalar(4, 64),
        scalar(0b0101, 8),
        vector(8, &[9, 9, 9, 9]),
    ];
    inputs.extend_from_slice(&data);
    assert_eq!(run(masked, &inputs, 32), Some(vec![14, 9, 9, 9]));
    assert_eq!(
        unmasked(
            Kernel::ReduceWide {
                kind: VectorReduceKind::Add,
                ordered: false,
            },
            VectorElement::integer(16, Some(false)),
            2,
            &[vector(8, &[200, 200]), vector(16, &[0])],
            16
        ),
        Some(vec![400])
    );
}

/// **A mask summary counts or finds the selected set bits.** Of `0b1011`
/// over four elements `vcpop.m` is 3 and `vfirst.m` 0; of no set bit
/// `vfirst.m` is all ones; governed by `0b1100` the count is 1 and the
/// first 3.
#[test]
fn a_mask_summary_counts_and_finds() {
    let summary = |first, governed: Option<u128>, source| {
        let computation = controlled(
            Kernel::MaskSummary { first },
            VectorElement::integer(1, None),
            VectorRounding::None,
            VectorTailPolicy::Agnostic,
            governed.map(|_| VectorInactivePolicy::Undisturbed),
        );
        let mut inputs = vec![scalar(0, 64), scalar(4, 64)];
        inputs.extend(governed.map(|mask| scalar(mask, 8)));
        inputs.extend([scalar(0, 64), scalar(source, 8)]);
        LaneExecution::of(computation, &inputs, None, FloatRules::default())?.summary()
    };
    assert_eq!(summary(false, None, 0b1011), Some(3));
    assert_eq!(summary(true, None, 0b1011), Some(0));
    assert_eq!(summary(true, None, 0), Some(u128::MAX));
    assert_eq!(summary(false, Some(0b1100), 0b1011), Some(1));
    assert_eq!(summary(true, Some(0b1100), 0b1011), Some(3));
}

/// **A carry-less product folds per element.** `vclmul` of `0b11` by
/// itself is `0b101`, `vclmulh` of it 0.
#[test]
fn a_carryless_product_folds_per_element() {
    let product = |part| Kernel::CarrylessMultiply { part };
    let data = [vector(64, &[0b11]), vector(64, &[0b11])];
    let quadwords = VectorElement::integer(64, None);
    assert_eq!(
        unmasked(product(CarrylessPart::Low), quadwords, 1, &data, 64),
        Some(vec![0b101])
    );
    assert_eq!(
        unmasked(product(CarrylessPart::High), quadwords, 1, &data, 64),
        Some(vec![0])
    );
}

/// **A crypto step folds group by group.** `vaesem.vv` over two 128-bit
/// groups is each group's round alone; zeroed by its mask, the second group
/// is zero; `vaesem.vs` keys every group with the one key group.
#[test]
fn a_crypto_step_folds_group_by_group() {
    let kind = VectorCryptoKind::AesEncrypt;
    let group = |values: [u128; 2]| {
        Some(LaneOperand::Vector {
            value: VectorConst::from_lanes(128, values).unwrap(),
            lane_bits: Some(128),
            lane_format: None,
        })
    };
    let alone = |state: u128, key: u128| {
        let one = |value| VectorConst::from_lanes(128, [value]).unwrap();
        kind.evaluate(&[one(state), one(key)], 32).unwrap()[0].lane(0)
    };
    let (states, keys) = ([0x0123_4567, 0x89ab_cdef << 64], [7, 9 << 100]);
    let step = |inactive| {
        controlled(
            Kernel::Crypto(kind),
            VectorElement::integer(128, None),
            VectorRounding::None,
            VectorTailPolicy::Agnostic,
            inactive,
        )
    };
    let inputs = [
        scalar(0, 64),
        scalar(2, 64),
        None,
        group(states),
        group(keys),
    ];
    assert_eq!(
        run(step(None), &inputs, 256).map(|words| words.len()),
        Some(4)
    );
    let words = |groups: [u128; 2]| -> Vec<u128> {
        groups
            .iter()
            .flat_map(|group| [group & u128::from(u64::MAX), group >> 64])
            .collect()
    };
    assert_eq!(
        run(step(None), &inputs, 256),
        Some(words([
            alone(states[0], keys[0]),
            alone(states[1], keys[1])
        ]))
    );
    let masked = [
        scalar(0, 64),
        scalar(2, 64),
        scalar(0b01, 8),
        None,
        group(states),
        group(keys),
    ];
    assert_eq!(
        run(step(Some(VectorInactivePolicy::Zero)), &masked, 256),
        Some(words([alone(states[0], keys[0]), 0]))
    );
    let shared = [
        scalar(0, 64),
        scalar(2, 64),
        None,
        group(states),
        Some(LaneOperand::Vector {
            value: VectorConst::from_lanes(128, [keys[0]]).unwrap(),
            lane_bits: Some(128),
            lane_format: None,
        }),
    ];
    assert_eq!(
        run(step(None), &shared, 256),
        Some(words([
            alone(states[0], keys[0]),
            alone(states[1], keys[0])
        ]))
    );
}

/// A vector of `format` elements whose type declares them so.
fn floats(format: FloatFormat, lanes: &[u128]) -> Option<LaneOperand> {
    Some(LaneOperand::Vector {
        value: VectorConst::from_lanes(format.bits(), lanes.iter().copied()).unwrap(),
        lane_bits: Some(format.bits()),
        lane_format: Some(format),
    })
}

/// The status an execution writes beside a data result `result_bits` wide.
fn status(
    computation: Computation,
    inputs: &[Option<LaneOperand>],
    result_bits: u32,
) -> Option<u128> {
    LaneExecution::of(computation, inputs, None, FloatRules::default())?.status(result_bits)
}

/// Single-precision bits.
fn single(value: f32) -> u128 {
    u128::from(value.to_bits())
}

/// A signalling single-precision NaN.
const SIGNALLING: u128 = 0x7fa0_0000;

/// **The float status unites what the executed elements raise.** `vfadd`
/// of `1 + 2^-30` is inexact, of a signalling NaN invalid, of the largest
/// float twice overflow and inexact; an element the mask leaves inactive
/// raises nothing, and an execution of no element reports zero. The data
/// result, its passthrough not known, folds nothing all the while.
#[test]
fn float_status_unites_the_executed_elements_exceptions() {
    let mut add = controlled(
        Kernel::Binary(VectorBinaryKind::Add),
        VectorElement::float(32),
        VectorRounding::FloatingEnvironment,
        VectorTailPolicy::Agnostic,
        Some(VectorInactivePolicy::Undisturbed),
    );
    add.status = VectorStatus::FloatingExceptions;
    let inputs = |length, mask| {
        [
            scalar(0, 64),
            scalar(length, 64),
            scalar(mask, 8),
            scalar(0, 64),
            None,
            floats(
                FloatFormat::Single,
                &[single(1.0), SIGNALLING, single(f32::MAX), 0],
            ),
            floats(
                FloatFormat::Single,
                &[single(2.0f32.powi(-30)), single(1.0), single(f32::MAX), 0],
            ),
        ]
    };
    assert_eq!(status(add, &inputs(4, 0b0001), 128), Some(0b00001));
    assert_eq!(status(add, &inputs(4, 0b0011), 128), Some(0b10001));
    assert_eq!(status(add, &inputs(4, 0b0111), 128), Some(0b10101));
    assert_eq!(status(add, &inputs(0, 0b1111), 128), Some(0));
    assert_eq!(run(add, &inputs(4, 0b0001), 128), None);
}

/// **Flags no rounding mode is named for fold where every mode agrees.**
/// The largest float twice overflows, inexactly, in every mode.
#[test]
fn flags_every_mode_agrees_on_fold_without_the_mode() {
    let mut add = controlled(
        Kernel::Binary(VectorBinaryKind::Add),
        VectorElement::float(32),
        VectorRounding::None,
        VectorTailPolicy::Agnostic,
        None,
    );
    add.status = VectorStatus::FloatingExceptions;
    let largest = floats(FloatFormat::Single, &[single(f32::MAX)]);
    let inputs = [scalar(0, 64), scalar(1, 64), None, largest.clone(), largest];
    assert_eq!(status(add, &inputs, 32), Some(0b00101));
}

/// **A signalling comparison raises invalid of a quiet NaN, a quiet one
/// only of a signalling NaN.** `vmflt` of a quiet NaN raises invalid,
/// `vmfeq` does not; `vmfeq` of a signalling NaN does.
#[test]
fn a_signalling_comparison_raises_invalid_of_a_quiet_nan() {
    let compare = |relation, signaling| {
        let mut compare = controlled(
            Kernel::Compare {
                relation,
                signaling,
            },
            VectorElement::float(32),
            VectorRounding::None,
            VectorTailPolicy::Undisturbed,
            None,
        );
        compare.status = VectorStatus::FloatingExceptions;
        compare
    };
    let inputs = |left| {
        [
            scalar(0, 64),
            scalar(2, 64),
            scalar(0, 8),
            floats(FloatFormat::Single, &[left, single(1.0)]),
            floats(FloatFormat::Single, &[single(1.0), single(1.0)]),
        ]
    };
    let quiet = 0x7fc0_0000;
    assert_eq!(
        status(compare(VectorCompareKind::Lt, true), &inputs(quiet), 8),
        Some(0b10000)
    );
    assert_eq!(
        status(compare(VectorCompareKind::Eq, false), &inputs(quiet), 8),
        Some(0)
    );
    assert_eq!(
        status(
            compare(VectorCompareKind::Eq, false),
            &inputs(SIGNALLING),
            8
        ),
        Some(0b10000)
    );
}

/// **A widening float operation promotes its sources exactly, in the
/// formats their types declare.** `vfwadd.vv` of the singles 1.5 and 2.25 is
/// the double 3.75, as is `vfwadd.vf` of a single scalar; `vfwmaccbf16` of
/// the bfloat16s 1.5 (`0x3fc0`) and 2.0 (`0x4000`) plus 1.0 is 4.0, while the
/// same bits declared binary16 (1.9375 and 2.0) give 4.875, and declared of
/// no format fold nothing. Promoting a signalling NaN raises invalid.
#[test]
fn a_widening_operation_promotes_its_float_sources_exactly() {
    let mut add = controlled(
        Kernel::Binary(VectorBinaryKind::Add),
        VectorElement::float(64),
        VectorRounding::None,
        VectorTailPolicy::Agnostic,
        None,
    );
    let inputs = |right: Option<LaneOperand>| {
        [
            scalar(0, 64),
            scalar(1, 64),
            None,
            floats(FloatFormat::Single, &[single(1.5)]),
            right,
        ]
    };
    let double = Some(vec![u128::from(3.75f64.to_bits())]);
    assert_eq!(
        run(
            add,
            &inputs(floats(FloatFormat::Single, &[single(2.25)])),
            64
        ),
        double
    );
    let scalar_single = Some(LaneOperand::Scalar {
        value: single(2.25),
        bits: 32,
        format: Some(FloatFormat::Single),
    });
    assert_eq!(run(add, &inputs(scalar_single), 64), double);
    assert_eq!(
        run(add, &inputs(declared(32, &[single(2.25)], Some(32))), 64),
        None
    );
    add.status = VectorStatus::FloatingExceptions;
    let signalling = [
        scalar(0, 64),
        scalar(1, 64),
        None,
        floats(FloatFormat::Single, &[SIGNALLING]),
        floats(FloatFormat::Single, &[single(1.0)]),
    ];
    assert_eq!(status(add, &signalling, 64), Some(0b10000));
    let fused = controlled(
        Kernel::FusedMultiplyAdd {
            negate_product: false,
            negate_addend: false,
        },
        VectorElement::float(32),
        VectorRounding::None,
        VectorTailPolicy::Agnostic,
        None,
    );
    let factors = |format| {
        [
            scalar(0, 64),
            scalar(1, 64),
            None,
            floats(format, &[0x3fc0]),
            floats(format, &[0x4000]),
            floats(FloatFormat::Single, &[single(1.0)]),
        ]
    };
    assert_eq!(
        run(fused, &factors(FloatFormat::BFloat16), 32),
        Some(vec![single(4.0)])
    );
    assert_eq!(
        run(fused, &factors(FloatFormat::Half), 32),
        Some(vec![single(4.875)])
    );
    let undeclared = [
        scalar(0, 64),
        scalar(1, 64),
        None,
        declared(16, &[0x3fc0], Some(16)),
        declared(16, &[0x4000], Some(16)),
        floats(FloatFormat::Single, &[single(1.0)]),
    ];
    assert_eq!(run(fused, &undeclared, 32), None);
}

/// **A saturating add or subtract reports whether an executed element
/// saturated.** `vsadd` of 100 and 100 saturates a signed byte, of 1 and 1
/// not, and masked off it reports nothing; `vsaddu` of 200 and 100
/// saturates an unsigned byte, of 200 and 55 not; `vssub` of -100 and 100
/// saturates.
#[test]
fn a_saturating_operation_reports_saturation() {
    let saturating = |kind, signed| {
        let mut computation = controlled(
            Kernel::Binary(kind),
            VectorElement::integer(8, Some(signed)),
            VectorRounding::None,
            VectorTailPolicy::Agnostic,
            Some(VectorInactivePolicy::Undisturbed),
        );
        computation.status = VectorStatus::Saturation;
        computation
    };
    let inputs = |mask, left: &[u128], right: &[u128]| {
        [
            scalar(0, 64),
            scalar(2, 64),
            scalar(mask, 8),
            None,
            vector(8, left),
            vector(8, right),
        ]
    };
    let add = saturating(VectorBinaryKind::SatAdd, true);
    assert_eq!(
        status(add, &inputs(0b11, &[100, 1], &[100, 1]), 16),
        Some(1)
    );
    assert_eq!(
        status(add, &inputs(0b10, &[100, 1], &[100, 1]), 16),
        Some(0)
    );
    let unsigned = saturating(VectorBinaryKind::SatAdd, false);
    assert_eq!(
        status(unsigned, &inputs(0b11, &[200, 0], &[100, 0]), 16),
        Some(1)
    );
    assert_eq!(
        status(unsigned, &inputs(0b11, &[200, 0], &[55, 0]), 16),
        Some(0)
    );
    let subtract = saturating(VectorBinaryKind::SatSub, true);
    assert_eq!(
        status(subtract, &inputs(0b11, &[156, 0], &[100, 0]), 16),
        Some(1)
    );
}

/// **A clip reports whether it clamped.** `vnclip` of the halfword 256 to a
/// signed byte clamps to 127 and saturates; of 64 it is exact.
#[test]
fn a_clip_reports_saturation() {
    let mut clip = controlled(
        Kernel::Fixed {
            kind: VectorFixedKind::Clip,
        },
        VectorElement::integer(8, Some(true)),
        VectorRounding::FixedEnvironment,
        VectorTailPolicy::Agnostic,
        None,
    );
    clip.status = VectorStatus::Saturation;
    let inputs = |source| {
        [
            scalar(0, 64),
            scalar(1, 64),
            scalar(0, 64),
            None,
            declared(16, &[source], Some(16)),
            vector(8, &[0]),
        ]
    };
    assert_eq!(run(clip, &inputs(256), 8), Some(vec![127]));
    assert_eq!(status(clip, &inputs(256), 8), Some(1));
    assert_eq!(run(clip, &inputs(64), 8), Some(vec![64]));
    assert_eq!(status(clip, &inputs(64), 8), Some(0));
}

/// **An ordered sum raises what each step raises; an unordered one only
/// where every order adds exactly.** `vfredosum` of `1, 2^-30` seeded by 0
/// is inexact; `vfredusum` of the same may round in another order, so its
/// flags fold nothing; of `1, 2` seeded by 4 every order is exact.
#[test]
fn a_sum_reports_the_flags_its_order_decides() {
    let sum = |ordered| {
        let mut sum = controlled(
            Kernel::Reduce {
                kind: VectorReduceKind::Add,
                ordered,
            },
            VectorElement::float(32),
            VectorRounding::None,
            VectorTailPolicy::Undisturbed,
            None,
        );
        sum.status = VectorStatus::FloatingExceptions;
        sum
    };
    let inputs = |values: &[u128], seed| {
        [
            scalar(0, 64),
            scalar(2, 64),
            vector(32, &[0, 0]),
            floats(FloatFormat::Single, values),
            floats(FloatFormat::Single, &[seed, 0]),
        ]
    };
    let rounding = inputs(&[single(1.0), single(2.0f32.powi(-30))], 0);
    assert_eq!(status(sum(true), &rounding, 64), Some(0b00001));
    assert_eq!(status(sum(false), &rounding, 64), None);
    let exact = inputs(&[single(1.0), single(2.0)], single(4.0));
    assert_eq!(status(sum(false), &exact, 64), Some(0));
}
