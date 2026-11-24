//! Shared floating-exception evaluation with independent status-bit oracles.

use std::{cell::Cell, process};

use analyssa::{
    analysis::{
        ConstEvaluator, SsaEvaluator, SsaVerifier, VerifyLevel,
        consts::{Destination, OperandLayout},
        context::AnalysisContext,
        execution::{StaticExecution, Value},
    },
    ir::{
        ConstValue, FloatBits, FloatExceptionLayout, FloatExceptions, FloatFormat, SsaOp, SsaVarId,
        float::FloatRounding,
        function::{SsaDefSpec, SsaFunctionBuilder},
        lanes::VectorConst,
        ops::{
            VectorBinaryKind, VectorCompareKind, VectorElement, VectorElementKind, VectorMaskMode,
        },
    },
    testing::{MACHINE, MockTarget, MockType},
};

/// The constants used by the float-exception fixtures.
type Cv = ConstValue<MockTarget>;

/// Makes an exact double constant.
fn float(value: f64) -> Cv {
    Cv::Float(FloatBits::from(value))
}

/// Names one fixed fixture operand.
fn var(index: usize) -> SsaVarId {
    SsaVarId::from_index(index)
}

/// Builds a pure scalar operation without reading its result as an input.
fn operation(kind: u8, dest: SsaVarId, left: SsaVarId, right: SsaVarId) -> SsaOp<MockTarget> {
    match kind {
        0 => SsaOp::Add {
            dest,
            left,
            right,
            flags: None,
        },
        1 => SsaOp::Sub {
            dest,
            left,
            right,
            flags: None,
        },
        2 => SsaOp::Mul {
            dest,
            left,
            right,
            flags: None,
        },
        _ => SsaOp::FloatCompareFlags {
            flags: dest,
            left,
            right,
            signaling: kind == 4,
        },
    }
}

/// Wraps one scalar operation with the requested status representation.
fn raised(kind: u8, flags: FloatExceptions, layout: FloatExceptionLayout) -> SsaOp<MockTarget> {
    SsaOp::float_raise(
        var(2),
        operation(kind, var(2), var(0), var(1)),
        None,
        flags,
        layout,
    )
}

/// Evaluates numeric float carriers at the operands' declared format.
fn fold(op: &SsaOp<MockTarget>, left: Cv, right: Cv, format: FloatFormat) -> Option<Cv> {
    MACHINE.evaluate_with_operand_layouts(
        op,
        Some(64),
        |id| match id.index() {
            0 => Some(left.clone()),
            1 => Some(right.clone()),
            _ => None,
        },
        |_| OperandLayout::Float(format),
    )
}

/// Exact arithmetic raises nothing, including cancellation to zero and sign changes.
#[test]
fn float_raise_exact_scalar_arithmetic() {
    for kind in 0..3 {
        let op = raised(kind, FloatExceptions::IEEE, FloatExceptionLayout::IEEE);
        assert_eq!(
            fold(&op, float(2.0f64), float(-2.0f64), FloatFormat::Double),
            Some(Cv::U64(0)),
            "kind={kind}"
        );
    }
}

/// IEEE invalid, overflow and inexact bits are independent of the numeric result's NaN policy.
#[test]
fn float_raise_scalar_ieee_status() {
    for (kind, left, right, expected) in [
        (0, f64::MAX, f64::MAX, 5),
        (0, 1.0, 2.0f64.powi(-53), 1),
        (1, f64::INFINITY, f64::INFINITY, 16),
        (2, 0.0, f64::INFINITY, 16),
    ] {
        let op = raised(kind, FloatExceptions::IEEE, FloatExceptionLayout::IEEE);
        assert_eq!(
            fold(&op, float(left), float(right), FloatFormat::Double),
            Some(Cv::U64(expected)),
            "kind={kind}"
        );
    }
}

/// Quiet and signaling compares distinguish signaling NaNs across all supported formats.
#[test]
fn float_raise_compare_nan_status_by_format() {
    for (format, one, quiet, signaling) in [
        (FloatFormat::Half, 0x3c00, 0x7e01, 0x7c01),
        (FloatFormat::BFloat16, 0x3f80, 0x7fc1, 0x7f81),
        (FloatFormat::Single, 0x3f800000, 0x7fc00001, 0x7f800001),
        (
            FloatFormat::Double,
            0x3ff0000000000000,
            0x7ff8000000000001,
            0x7ff0000000000001,
        ),
        (
            FloatFormat::X87Extended,
            0x3fff8000000000000000,
            0x7fffc000000000000001,
            0x7fff8000000000000001,
        ),
        (
            FloatFormat::Quad,
            0x3fff0000000000000000000000000000,
            0x7fff8000000000000000000000000001,
            0x7fff0000000000000000000000000001,
        ),
    ] {
        let cv = |bits| Cv::Float(FloatBits::new(format, bits).unwrap_or_else(|| process::abort()));
        for (kind, input, expected) in [
            (3, quiet, 0),
            (3, signaling, 16),
            (4, quiet, 16),
            (4, signaling, 16),
        ] {
            let op = raised(kind, FloatExceptions::IEEE, FloatExceptionLayout::IEEE);
            assert_eq!(
                fold(&op, cv(input), cv(one), format),
                Some(Cv::U64(expected)),
                "{format:?}, kind={kind}"
            );
        }
    }
}

/// The flag mask and register placement apply to exceptions, without comparing result values.
#[test]
fn float_raise_mask_and_status_layout() {
    let layout = FloatExceptionLayout::new([Some(5), Some(4), Some(3), Some(2), Some(0), None]);
    for (flags, expected) in [
        (FloatExceptions::IEEE, 0x28),
        (FloatExceptions::OVERFLOW, 8),
        (FloatExceptions::INVALID, 0),
    ] {
        let op = raised(0, flags, layout);
        assert_eq!(
            fold(&op, float(f64::MAX), float(f64::MAX), FloatFormat::Double),
            Some(Cv::U64(expected))
        );
    }
}

/// No requested flags and a disabled scalar lane need no numeric operand values.
#[test]
fn float_raise_empty_work_does_not_read_operands() {
    for empty_flags in [false, true] {
        let mut op = raised(
            0,
            if empty_flags {
                FloatExceptions::NONE
            } else {
                FloatExceptions::IEEE
            },
            FloatExceptionLayout::IEEE,
        );
        if let SsaOp::FloatRaise { lanes, .. } = &mut op {
            *lanes = (!empty_flags).then(|| var(3));
        }
        let reads = Cell::new(0);
        let result = MACHINE.evaluate(&op, Some(64), |id| {
            if id == var(3) {
                Some(Cv::U64(0))
            } else {
                reads.set(reads.get() + 1);
                None
            }
        });
        assert_eq!(result, Some(Cv::U64(0)));
        assert_eq!(reads.get(), 0);
    }
}

/// A numeric unknown stays unknown when its flags are demanded.
#[test]
fn float_raise_unknown_operand_has_no_invented_status() {
    let op = raised(0, FloatExceptions::IEEE, FloatExceptionLayout::IEEE);
    assert_eq!(MACHINE.evaluate(&op, Some(64), |_| None), None);
}

/// Recursive folding, concrete symbolic evaluation and execution use one status computation.
#[test]
fn float_raise_shared_consumers() {
    for kind in [0, 3, 4] {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let result = builder
            .in_block(0, |block| {
                let left = block.const_value(SsaDefSpec::tmp(MockType::F64), float(1.0f64))?;
                let right = block.const_value(SsaDefSpec::tmp(MockType::F64), float(2.0f64))?;
                let result = block.emit_def(SsaDefSpec::tmp(MockType::U64), |dest| {
                    SsaOp::float_raise(
                        dest,
                        operation(kind, dest, left, right),
                        None,
                        FloatExceptions::IEEE,
                        FloatExceptionLayout::IEEE,
                    )
                })?;
                block.ret(Some(result))?;
                Ok(result)
            })
            .unwrap_or_else(|_| process::abort());
        let ssa = builder.finish().unwrap_or_else(|_| process::abort());
        assert!(
            SsaVerifier::new(&ssa)
                .verify(VerifyLevel::Standard)
                .is_empty()
        );
        assert_eq!(
            ConstEvaluator::new(&ssa, MACHINE).evaluate_var(result),
            Some(Cv::U64(0)),
            "constant evaluator"
        );
        let mut symbolic = SsaEvaluator::new(&ssa, MACHINE);
        symbolic.evaluate_block(0);
        assert_eq!(
            symbolic.get_concrete(result),
            Some(&Cv::U64(0)),
            "symbolic concrete path"
        );
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context).run();
        assert!(facts.is_complete());
        assert_eq!(
            facts.returned().and_then(Value::constant),
            Some(&Cv::U64(0)),
            "static execution"
        );
    }
}

/// Builds a two-lane binary raise with independent outer and instruction masks.
fn vector_raise(
    predicated: bool,
    scalar: bool,
    outer: Option<u64>,
    inner: Option<u64>,
) -> (SsaOp<MockTarget>, [Cv; 4]) {
    let mut element = VectorElement::float(64);
    element.scalar = scalar;
    let computation = if predicated {
        SsaOp::VectorPredicatedBinary {
            dest: var(4),
            left: var(0),
            right: var(1),
            mask: var(3),
            passthrough: None,
            kind: VectorBinaryKind::Mul,
            element,
            mode: VectorMaskMode::Zero,
            environment: None,
        }
    } else {
        SsaOp::VectorBinary {
            dest: var(4),
            left: var(0),
            right: var(1),
            kind: VectorBinaryKind::Mul,
            element,
            environment: None,
        }
    };
    let op = SsaOp::float_raise(
        var(4),
        computation,
        outer.map(|_| var(2)),
        FloatExceptions::IEEE,
        FloatExceptionLayout::IEEE,
    );
    let vector =
        |lanes| Cv::Vector(VectorConst::from_lanes(64, lanes).unwrap_or_else(|| process::abort()));
    (
        op,
        [
            vector([1.0f64.to_bits() as u128, 0]),
            vector([1.0f64.to_bits() as u128, f64::INFINITY.to_bits() as u128]),
            Cv::U64(outer.unwrap_or(3)),
            Cv::U64(inner.unwrap_or(3)),
        ],
    )
}

/// A higher active lane contributes invalid even when the low mask bit is clear.
#[test]
fn float_raise_vector_masks_do_not_discard_higher_lanes() {
    for (mask, expected) in [(0, 0), (1, 0), (2, 16), (3, 16)] {
        let (op, values) = vector_raise(false, false, Some(mask), None);
        assert_eq!(
            MACHINE.evaluate(&op, Some(64), |id| values.get(id.index()).cloned()),
            Some(Cv::U64(expected)),
            "mask={mask}"
        );
    }
}

/// An instruction mask and outer observation mask intersect, with no passthrough computation.
#[test]
fn float_raise_vector_predication_intersects_outer_mask() {
    for (outer, inner, expected) in [(3, 1, 0), (1, 2, 0), (2, 2, 16), (3, 3, 16)] {
        let (op, values) = vector_raise(true, false, Some(outer), Some(inner));
        assert_eq!(
            MACHINE.evaluate(&op, Some(64), |id| values.get(id.index()).cloned()),
            Some(Cv::U64(expected)),
            "outer={outer} inner={inner}"
        );
    }
}

/// A scalar lane operation never raises from preserved upper lanes.
#[test]
fn float_raise_vector_scalar_form_ignores_upper_lanes() {
    let (op, values) = vector_raise(false, true, None, None);
    assert_eq!(
        MACHINE.evaluate(&op, Some(64), |id| values.get(id.index()).cloned()),
        Some(Cv::U64(0))
    );
}

/// A typed floating input interprets raw carrier bits before probing its exceptions.
#[test]
fn float_raise_typed_inputs_keep_float_interpretation() {
    let op = raised(4, FloatExceptions::IEEE, FloatExceptionLayout::IEEE);
    assert_eq!(
        fold(
            &op,
            Cv::U64(0x7ff8_0000_0000_0001),
            Cv::U64(0x3ff0_0000_0000_0000),
            FloatFormat::Double
        ),
        Some(Cv::U64(16))
    );
}

/// Underflow can depend on rounding while the requested inexact bit remains known.
#[test]
fn float_raise_rounding_agrees_on_requested_status_bits() {
    for (rounding, requested, expected) in [
        (None, FloatExceptions::IEEE, None),
        (None, FloatExceptions::INEXACT, Some(1)),
        (Some(FloatRounding::Up), FloatExceptions::IEEE, Some(1)),
        (Some(FloatRounding::Down), FloatExceptions::IEEE, Some(3)),
    ] {
        let mut element = VectorElement::float(64);
        element.scalar = true;
        element.rounding = rounding;
        let op = SsaOp::float_raise(
            var(2),
            SsaOp::VectorBinary {
                dest: var(2),
                left: var(0),
                right: var(1),
                kind: VectorBinaryKind::Mul,
                element,
                environment: None,
            },
            None,
            requested,
            FloatExceptionLayout::IEEE,
        );
        assert_eq!(
            MACHINE.evaluate(&op, Some(64), |id| match id.index() {
                0 => Some(Cv::float(FloatFormat::Double, 0x0010_0000_0000_0000)?),
                1 => Some(Cv::float(FloatFormat::Double, 0x3fef_ffff_ffff_ffff)?),
                _ => None,
            }),
            expected.map(Cv::U64),
            "{rounding:?}"
        );
    }
}

/// An always-underflowing lane makes the complete status word invariant across modes.
#[test]
fn float_raise_rounding_agreement_follows_lane_union() {
    let op = SsaOp::float_raise(
        var(2),
        SsaOp::VectorBinary {
            dest: var(2),
            left: var(0),
            right: var(1),
            kind: VectorBinaryKind::Mul,
            element: VectorElement::float(64),
            environment: None,
        },
        None,
        FloatExceptions::IEEE,
        FloatExceptionLayout::IEEE,
    );
    let values = [
        VectorConst::from_lanes(64, [0x0010_0000_0000_0000, 1]).map(Cv::Vector),
        VectorConst::from_lanes(64, [0x3fef_ffff_ffff_ffff, 0x3fe0_0000_0000_0000]).map(Cv::Vector),
    ];
    assert_eq!(
        MACHINE.evaluate(&op, Some(64), |id| values
            .get(id.index())
            .cloned()
            .flatten()),
        Some(Cv::U64(3))
    );
}

/// Vector comparisons report signaling behavior independently of the predicate value.
#[test]
fn float_raise_vector_comparison_signaling() {
    for (signaling, expected) in [(false, 0), (true, 16)] {
        let op = SsaOp::float_raise(
            var(2),
            SsaOp::VectorCompare {
                dest: var(2),
                left: var(0),
                right: var(1),
                kind: VectorCompareKind::Eq,
                element: VectorElement::float(64),
                signaling,
            },
            None,
            FloatExceptions::IEEE,
            FloatExceptionLayout::IEEE,
        );
        let values = [
            VectorConst::from_lanes(64, [0x7ff8_0000_0000_0001, 0]).map(Cv::Vector),
            VectorConst::from_lanes(64, [0x3ff0_0000_0000_0000, 0]).map(Cv::Vector),
        ];
        assert_eq!(
            MACHINE.evaluate(&op, Some(64), |id| values
                .get(id.index())
                .cloned()
                .flatten()),
            Some(Cv::U64(expected))
        );
    }
}

/// Input-denormal and inexact rounding-direction flags require their unstated environment.
#[test]
fn float_raise_environment_dependent_status_stays_unknown() {
    let layout = FloatExceptionLayout::new([Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)])
        .rounded_up_at(6);
    let subnormal = Cv::float(FloatFormat::Double, 1).unwrap_or_else(|| process::abort());
    let op = raised(0, FloatExceptions::INPUT_DENORMAL, layout);
    assert_eq!(fold(&op, subnormal, float(1.0), FloatFormat::Double), None);
    let op = raised(0, FloatExceptions::ROUNDED_UP, layout);
    assert_eq!(
        fold(
            &op,
            float(1.0),
            float(2.0f64.powi(-53)),
            FloatFormat::Double
        ),
        None
    );
    assert_eq!(
        fold(&op, float(1.0), float(2.0), FloatFormat::Double),
        Some(Cv::U64(0))
    );
}

/// All typed consumers preserve the packed operation and its instruction mask.
#[test]
fn float_raise_vector_shared_consumers() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let result = builder
        .in_block(0, |block| {
            let vector = |values| {
                Cv::Vector(VectorConst::from_lanes(64, values).unwrap_or_else(|| process::abort()))
            };
            let left = block.const_value(SsaDefSpec::tmp(MockType::V2F64), vector([0, 0]))?;
            let right = block.const_value(
                SsaDefSpec::tmp(MockType::V2F64),
                vector([0, 0x7ff0_0000_0000_0000]),
            )?;
            let mask = block.const_value(SsaDefSpec::tmp(MockType::U64), Cv::U64(2))?;
            let result = block.emit_def(SsaDefSpec::tmp(MockType::U64), |dest| {
                SsaOp::float_raise(
                    dest,
                    SsaOp::VectorPredicatedBinary {
                        dest,
                        left,
                        right,
                        mask,
                        passthrough: None,
                        kind: VectorBinaryKind::Mul,
                        element: VectorElement::float(64),
                        mode: VectorMaskMode::Zero,
                        environment: None,
                    },
                    None,
                    FloatExceptions::IEEE,
                    FloatExceptionLayout::IEEE,
                )
            })?;
            block.ret(Some(result))?;
            Ok(result)
        })
        .unwrap_or_else(|_| process::abort());
    let ssa = builder.finish().unwrap_or_else(|_| process::abort());
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    assert_eq!(
        ConstEvaluator::new(&ssa, MACHINE).evaluate_var(result),
        Some(Cv::U64(16))
    );
    let mut symbolic = SsaEvaluator::new(&ssa, MACHINE);
    symbolic.evaluate_block(0);
    assert_eq!(symbolic.get_concrete(result), Some(&Cv::U64(16)));
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete());
    assert_eq!(
        facts.returned().and_then(Value::constant),
        Some(&Cv::U64(16))
    );
}

/// Each active lane contributes its independently raised status bits.
#[test]
fn float_raise_vector_status_is_a_union() {
    let op = SsaOp::float_raise(
        var(2),
        SsaOp::VectorBinary {
            dest: var(2),
            left: var(0),
            right: var(1),
            kind: VectorBinaryKind::Mul,
            element: VectorElement::float(64),
            environment: None,
        },
        None,
        FloatExceptions::IEEE,
        FloatExceptionLayout::IEEE,
    );
    let values = [
        VectorConst::from_lanes(64, [0x7fef_ffff_ffff_ffff, 0]).map(Cv::Vector),
        VectorConst::from_lanes(64, [0x7fef_ffff_ffff_ffff, 0x7ff0_0000_0000_0000]).map(Cv::Vector),
    ];
    assert_eq!(
        MACHINE.evaluate(&op, Some(64), |id| values
            .get(id.index())
            .cloned()
            .flatten()),
        Some(Cv::U64(21))
    );
}

/// Every supported float format reads packed arithmetic at its explicit lane width.
#[test]
fn float_raise_packed_formats_and_binary_kinds() {
    for (format, one, infinity, signaling) in [
        (FloatFormat::Half, 0x3c00u128, 0x7c00u128, 0x7c01u128),
        (FloatFormat::BFloat16, 0x3f80, 0x7f80, 0x7f81),
        (FloatFormat::Single, 0x3f800000, 0x7f800000, 0x7f800001),
        (
            FloatFormat::Double,
            0x3ff0000000000000,
            0x7ff0000000000000,
            0x7ff0000000000001,
        ),
        (
            FloatFormat::X87Extended,
            0x3fff8000000000000000,
            0x7fff8000000000000000,
            0x7fff8000000000000001,
        ),
        (
            FloatFormat::Quad,
            0x3fff0000000000000000000000000000,
            0x7fff0000000000000000000000000000,
            0x7fff0000000000000000000000000001,
        ),
    ] {
        let mut element = VectorElement::float(format.bits());
        if format == FloatFormat::BFloat16 {
            element.kind = VectorElementKind::BFloat16;
        }
        for (kind, left, right, expected) in [
            (VectorBinaryKind::Add, one, one, 0),
            (VectorBinaryKind::Sub, infinity, infinity, 16),
            (VectorBinaryKind::Mul, 0, infinity, 16),
            (VectorBinaryKind::Div, one, 0, 8),
            (VectorBinaryKind::MinNum, signaling, one, 16),
            (VectorBinaryKind::MaxNum, one, signaling, 16),
        ] {
            let op = SsaOp::float_raise(
                var(2),
                SsaOp::VectorBinary {
                    dest: var(2),
                    left: var(0),
                    right: var(1),
                    kind,
                    element,
                    environment: None,
                },
                None,
                FloatExceptions::IEEE,
                FloatExceptionLayout::IEEE,
            );
            let values = [
                VectorConst::from_lanes(format.bits(), [left, left]).map(Cv::Vector),
                VectorConst::from_lanes(format.bits(), [right, right]).map(Cv::Vector),
            ];
            assert_eq!(
                MACHINE.evaluate(&op, Some(64), |id| values
                    .get(id.index())
                    .cloned()
                    .flatten()),
                Some(Cv::U64(expected)),
                "{format:?} {kind:?}"
            );
        }
    }
}

/// A declared integer mask reads its scalar carrier without becoming a packed predicate.
#[test]
fn float_raise_integer_mask_carrier_keeps_declared_interpretation() {
    let (op, mut values) = vector_raise(true, false, None, Some(2));
    values[3] = Cv::Vector(VectorConst::from_lanes(64, [2]).unwrap_or_else(|| process::abort()));
    let result = MACHINE.evaluate_with_operand_layouts(
        &op,
        Some(64),
        |id| values.get(id.index()).cloned(),
        |id| {
            if id == var(3) {
                OperandLayout::Register(Destination {
                    bits: Some(64),
                    integer: true,
                    ..Destination::default()
                })
            } else {
                OperandLayout::default()
            }
        },
    );
    assert_eq!(result, Some(Cv::U64(16)));
}

/// Status words use their declared width instead of retaining a 64-bit helper carrier.
#[test]
fn float_raise_status_obeys_its_declared_register_width() {
    for width in [1, 5, 7, 8, 16, 24, 32, 64, 65, 127, 128] {
        let op = raised(0, FloatExceptions::IEEE, FloatExceptionLayout::IEEE);
        let value = MACHINE.evaluate(&op, Some(width), |id| match id.index() {
            0 | 1 => Some(float(0.0)),
            _ => None,
        });
        let bits = value.as_ref().and_then(|value| match value {
            Cv::False => Some(0),
            Cv::True => Some(1),
            _ => value
                .scalar_integer_pattern(MACHINE.ptr_size)
                .map(|(bits, _)| bits),
        });
        assert_eq!(bits, Some(0), "width={width}");
    }
}

/// A status bit outside its destination is refused instead of silently truncated.
#[test]
fn float_raise_status_does_not_truncate_raised_bits() {
    let layout = FloatExceptionLayout::new([Some(63), None, None, None, None, None]);
    let op = raised(0, FloatExceptions::INEXACT, layout);
    assert_eq!(
        MACHINE.evaluate(&op, Some(8), |id| match id.index() {
            0 => Some(float(1.0)),
            1 => Some(float(2.0f64.powi(-53))),
            _ => None,
        }),
        None
    );
}

/// Native-style byte status fields remain known through all typed evaluation consumers.
#[test]
fn float_raise_byte_status_shared_consumers() {
    let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
    let result = builder
        .in_block(0, |block| {
            let input = block.const_value(SsaDefSpec::tmp(MockType::F64), float(0.0))?;
            let result = block.emit_def(SsaDefSpec::tmp(MockType::U8), |dest| {
                SsaOp::float_raise(
                    dest,
                    SsaOp::VectorBinary {
                        dest,
                        left: input,
                        right: input,
                        kind: VectorBinaryKind::Add,
                        element: VectorElement {
                            scalar: true,
                            ..VectorElement::float(64)
                        },
                        environment: None,
                    },
                    None,
                    FloatExceptions::IEEE,
                    FloatExceptionLayout::IEEE,
                )
            })?;
            block.ret(Some(result))?;
            Ok(result)
        })
        .unwrap_or_else(|_| process::abort());
    let ssa = builder.finish().unwrap_or_else(|_| process::abort());
    assert!(
        SsaVerifier::new(&ssa)
            .verify(VerifyLevel::Standard)
            .is_empty()
    );
    assert_eq!(
        ConstEvaluator::new(&ssa, MACHINE).evaluate_var(result),
        Some(Cv::U8(0))
    );
    let mut symbolic = SsaEvaluator::new(&ssa, MACHINE);
    symbolic.evaluate_block(0);
    assert_eq!(symbolic.get_concrete(result), Some(&Cv::U8(0)));
    let context = AnalysisContext::new(MACHINE);
    let facts = StaticExecution::new(&ssa, &context).run();
    assert!(facts.is_complete());
    assert_eq!(facts.returned().and_then(Value::constant), Some(&Cv::U8(0)));
}

/// Returns independent IEEE encodings for finite, quiet-NaN and signaling-NaN inputs.
fn select_formats() -> [(FloatFormat, u128, u128, u128); 6] {
    [
        (FloatFormat::Half, 0x3c00, 0x7e01, 0x7c01),
        (FloatFormat::BFloat16, 0x3f80, 0x7fc1, 0x7f81),
        (FloatFormat::Single, 0x3f80_0000, 0x7fc0_0001, 0x7f80_0001),
        (
            FloatFormat::Double,
            0x3ff0_0000_0000_0000,
            0x7ff8_0000_0000_0001,
            0x7ff0_0000_0000_0001,
        ),
        (
            FloatFormat::X87Extended,
            0x3fff_8000_0000_0000_0000,
            0x7fff_c000_0000_0000_0001,
            0x7fff_8000_0000_0000_0001,
        ),
        (
            FloatFormat::Quad,
            0x3fff_0000_0000_0000_0000_0000_0000_0000,
            0x7fff_8000_0000_0000_0000_0000_0000_0001,
            0x7fff_0000_0000_0000_0000_0000_0000_0001,
        ),
    ]
}

/// Constructs a select observation, optionally intersecting two independent masks.
fn select_raise(
    kind: VectorBinaryKind,
    element: VectorElement,
    predicated: bool,
) -> SsaOp<MockTarget> {
    let computation = if predicated {
        SsaOp::VectorPredicatedBinary {
            dest: var(4),
            left: var(0),
            right: var(1),
            mask: var(3),
            passthrough: None,
            kind,
            element,
            mode: VectorMaskMode::Zero,
            environment: None,
        }
    } else {
        SsaOp::VectorBinary {
            dest: var(4),
            left: var(0),
            right: var(1),
            kind,
            element,
            environment: None,
        }
    };
    SsaOp::float_raise(
        var(4),
        computation,
        predicated.then(|| var(2)),
        FloatExceptions::IEEE,
        FloatExceptionLayout::IEEE,
    )
}

/// Both select comparisons signal on either NaN, independently of their selected result.
#[test]
fn float_raise_select_nan_family() {
    for (format, one, quiet, signaling) in select_formats() {
        let mut element = VectorElement::float(format.bits());
        if format == FloatFormat::BFloat16 {
            element.kind = VectorElementKind::BFloat16;
        }
        let sign = 1u128 << (format.bits() - 1);
        let infinity = signaling & !1;
        for kind in [VectorBinaryKind::MinLess, VectorBinaryKind::MaxGreater] {
            for rounding in [None, Some(FloatRounding::Down), Some(FloatRounding::Up)] {
                element.rounding = rounding;
                let op = select_raise(kind, element, false);
                for (left, right, expected) in [
                    (one, one, 0),
                    (0, sign, 0),
                    (sign, 0, 0),
                    (one | sign, one, 0),
                    (infinity, infinity | sign, 0),
                    (quiet, one, 16),
                    (one, quiet, 16),
                    (signaling, one, 16),
                    (one, signaling, 16),
                    (quiet, quiet, 16),
                    (quiet, signaling, 16),
                    (signaling, quiet, 16),
                ] {
                    let values = [left, right].map(|bits| {
                        VectorConst::from_lanes(format.bits(), [bits, bits]).map(Cv::Vector)
                    });
                    assert_eq!(
                        MACHINE.evaluate(&op, Some(64), |id| values
                            .get(id.index())
                            .cloned()
                            .flatten()),
                        Some(Cv::U64(expected)),
                        "{format:?} {kind:?} {rounding:?} {left:x} {right:x}"
                    );
                }
            }
        }
    }
}

/// Predicated selects contribute only from the intersection of active, computed lanes.
#[test]
fn float_raise_select_masks_and_scalar_lanes() {
    for (format, one, quiet, _) in select_formats() {
        let mut element = VectorElement::float(format.bits());
        if format == FloatFormat::BFloat16 {
            element.kind = VectorElementKind::BFloat16;
        }
        for kind in [VectorBinaryKind::MinLess, VectorBinaryKind::MaxGreater] {
            for scalar in [false, true] {
                element.scalar = scalar;
                let op = select_raise(kind, element, true);
                for outer in 0u64..4 {
                    for inner in 0u64..4 {
                        let values = [
                            Cv::Vector(
                                VectorConst::from_lanes(format.bits(), [one, quiet])
                                    .unwrap_or_else(|| process::abort()),
                            ),
                            Cv::Vector(
                                VectorConst::from_lanes(format.bits(), [one, one])
                                    .unwrap_or_else(|| process::abort()),
                            ),
                            Cv::U64(outer),
                            Cv::U64(inner),
                        ];
                        let expected = if !scalar && outer & inner & 2 != 0 {
                            16
                        } else {
                            0
                        };
                        assert_eq!(
                            MACHINE.evaluate(&op, Some(64), |id| values.get(id.index()).cloned()),
                            Some(Cv::U64(expected)),
                            "{format:?} {kind:?} scalar={scalar} masks={outer}/{inner}"
                        );
                    }
                }
            }
        }
    }
}

/// Unknown active inputs stay unknown, while either empty mask avoids numeric reads.
#[test]
fn float_raise_select_unknown_and_inactive_inputs() {
    for kind in [VectorBinaryKind::MinLess, VectorBinaryKind::MaxGreater] {
        let op = select_raise(kind, VectorElement::float(64), true);
        for (outer, inner, expected, maximum_numeric_reads) in [
            (3, 3, None, 1),
            (0, 3, Some(Cv::U64(0)), 0),
            (3, 0, Some(Cv::U64(0)), 0),
        ] {
            let reads = Cell::new(0);
            let result = MACHINE.evaluate(&op, Some(64), |id| match id.index() {
                2 => Some(Cv::U64(outer)),
                3 => Some(Cv::U64(inner)),
                _ => {
                    reads.set(reads.get() + 1);
                    None
                }
            });
            assert_eq!(result, expected, "{kind:?} masks={outer}/{inner}");
            assert!(reads.get() <= maximum_numeric_reads);
        }
    }
}

/// Verified SSA exposes the same select exceptions through all three evaluator consumers.
#[test]
fn float_raise_select_shared_consumers() {
    for kind in [VectorBinaryKind::MinLess, VectorBinaryKind::MaxGreater] {
        for mask in [1, 2] {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
            let result = builder
                .in_block(0, |block| {
                    let vector = |lanes| {
                        Cv::Vector(
                            VectorConst::from_lanes(64, lanes).unwrap_or_else(|| process::abort()),
                        )
                    };
                    let left = block.const_value(
                        SsaDefSpec::tmp(MockType::V2F64),
                        vector([0x3ff0_0000_0000_0000, 0x7ff8_0000_0000_0001]),
                    )?;
                    let right = block.const_value(
                        SsaDefSpec::tmp(MockType::V2F64),
                        vector([0x3ff0_0000_0000_0000; 2]),
                    )?;
                    let lanes = block.const_value(SsaDefSpec::tmp(MockType::U64), Cv::U64(mask))?;
                    let result = block.emit_def(SsaDefSpec::tmp(MockType::U64), |dest| {
                        SsaOp::float_raise(
                            dest,
                            SsaOp::VectorBinary {
                                dest,
                                left,
                                right,
                                kind,
                                element: VectorElement::float(64),
                                environment: None,
                            },
                            Some(lanes),
                            FloatExceptions::IEEE,
                            FloatExceptionLayout::IEEE,
                        )
                    })?;
                    block.ret(Some(result))?;
                    Ok(result)
                })
                .unwrap_or_else(|_| process::abort());
            let ssa = builder.finish().unwrap_or_else(|_| process::abort());
            assert!(
                SsaVerifier::new(&ssa)
                    .verify(VerifyLevel::Standard)
                    .is_empty()
            );
            let expected = Cv::U64(if mask == 2 { 16 } else { 0 });
            assert_eq!(
                ConstEvaluator::new(&ssa, MACHINE).evaluate_var(result),
                Some(expected.clone())
            );
            let mut symbolic = SsaEvaluator::new(&ssa, MACHINE);
            symbolic.evaluate_block(0);
            assert_eq!(symbolic.get_concrete(result), Some(&expected));
            let context = AnalysisContext::new(MACHINE);
            let facts = StaticExecution::new(&ssa, &context).run();
            assert!(facts.is_complete());
            assert_eq!(facts.returned().and_then(Value::constant), Some(&expected));
        }
    }
}
