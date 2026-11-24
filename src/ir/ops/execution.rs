//! Computations: the kernels a machine computes as one operation
//! ([`Kernel`]), over a whole value or over the lanes of a vector under an
//! explicit active interval and lane policies ([`LaneControl`]), each stated
//! with its element and its rounding ([`Computation`]).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod emit;

mod computation;
mod kernel;

pub use self::{
    computation::{Computation, LaneControl},
    kernel::Kernel,
};

/// The treatment of an inactive element inside the active interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorInactivePolicy {
    /// Retains the element from the passthrough value.
    Undisturbed,
    /// Writes zero without evaluating the element's computation.
    Zero,
    /// Retains the old element or writes all ones, independently per element.
    ///
    /// The choice is unspecified and need not be repeatable. A consumer must
    /// not infer either outcome, or correlate choices between executions.
    Agnostic,
}

/// The action a controlled vector operation takes at one result element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorLaneAction {
    /// Evaluates the kernel for this element.
    Execute,
    /// Retains the passthrough element without executing the kernel.
    Preserve,
    /// Writes zero without executing the kernel.
    Zero,
    /// Uses the inactive or tail policy's unspecified result without executing the kernel.
    Agnostic,
    /// Retains the old predicate bit, writes one, or writes the computed comparison bit.
    ///
    /// Computing this optional tail result never raises exceptions or contributes
    /// to status results. The descriptor's explicit tail limit bounds the source.
    MaskTail,
}

impl VectorLaneAction {
    /// Returns whether this action permits a candidate predicate bit.
    ///
    /// `computed` is the kernel's hypothetical result at this index. It need
    /// not be evaluated for actions that do not permit its value.
    #[must_use]
    pub const fn permits_mask_bit(self, old: bool, computed: bool, candidate: bool) -> bool {
        match self {
            Self::Execute => candidate == computed,
            Self::Preserve => candidate == old,
            Self::Zero => !candidate,
            Self::Agnostic => candidate == old || candidate,
            Self::MaskTail => candidate == old || candidate || candidate == computed,
        }
    }

    /// Returns whether evaluating this lane may contribute exceptions or status.
    #[must_use]
    pub const fn raises_effects(self) -> bool {
        matches!(self, Self::Execute)
    }
}

/// The data source of a selected slide result element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorSlideSource {
    /// Retains the old destination below an upward slide's offset.
    Preserve,
    /// Supplies zero beyond the source vector's capacity.
    Zero,
    /// Supplies the scalar fill argument.
    Scalar,
    /// Supplies the specified element from the source vector.
    Element(u64),
}

/// The physical placement of a vector's predicate bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorPredicateLayout {
    /// Bit `i` governs logical element `i`.
    Packed,
    /// Bit `i * element_bytes` governs logical element `i`.
    ///
    /// Predicate results clear positions between those element bits. A predicate
    /// register contains one physical bit per byte of its corresponding vector.
    ByteGranular,
}

impl VectorPredicateLayout {
    /// Returns the physical predicate bit for a logical element, refusing overflow.
    #[must_use]
    pub const fn bit(self, index: u64, element_bits: u32) -> Option<u64> {
        match self {
            Self::Packed => Some(index),
            Self::ByteGranular if element_bits != 0 && element_bits.is_multiple_of(8) => {
                index.checked_mul((element_bits / 8) as u64)
            }
            Self::ByteGranular => None,
        }
    }
}

/// Where a bit field's position comes from ([`Kernel::Bitfield`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BitfieldPosition {
    /// The instruction states it (`extrq xmm0, 8, 4`).
    Immediate {
        /// The field's first bit.
        index: u8,
        /// The field's width, zero for the whole quadword.
        length: u8,
    },
    /// A control quadword after the data states it, its bits 5 to 0 the
    /// width and 13 to 8 the first bit (`extrq xmm0, xmm1`).
    Register,
}

/// The relation of selected result bits to the first selected source bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorMaskPrefix {
    /// Sets selected positions before the first selected true source bit.
    Before,
    /// Sets selected positions through the first selected true source bit.
    Including,
    /// Sets only the first selected true source bit, if one exists.
    Only,
}

/// The rounding input a controlled vector kernel consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorRounding {
    /// The kernel does not consult a rounding environment.
    None,
    /// Rounds toward zero without consulting the environment.
    TowardZero,
    /// Rounds an inexact result to an odd least significant retained bit.
    ToOdd,
    /// Consumes an explicit IEEE rounding mode after the execution mask.
    ///
    /// The mode uses zero for nearest-even, one for toward-zero, two for
    /// toward-negative, three for toward-positive and four for nearest-away.
    /// Other mode values are outside the operation's domain.
    FloatingEnvironment,
    /// Consumes an explicit fixed-point rounding mode after the execution mask.
    ///
    /// The mode uses zero for nearest-up, one for nearest-even, two for down
    /// and three for odd. Other values are outside the operation's domain.
    FixedEnvironment,
}

/// The floating-point operation class queried independently of vector state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum VectorFloatCapability {
    /// IEEE arithmetic at the selected element precision.
    Arithmetic = 0,
    /// Conversion between bfloat16 and binary32; the selected width is sixteen.
    BFloatConversions = 1,
    /// Minimal binary16/binary32 conversions; the selected width is sixteen.
    HalfConversions = 2,
    /// Fused bfloat16 products accumulated in binary32; the width is sixteen.
    BFloatWideFused = 3,
}

impl VectorFloatCapability {
    /// Returns the stable query selector.
    #[must_use]
    pub const fn selector(self) -> u64 {
        self as u64
    }

    /// Returns the operation class named by a query selector.
    #[must_use]
    pub const fn from_selector(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::Arithmetic),
            1 => Some(Self::BFloatConversions),
            2 => Some(Self::HalfConversions),
            3 => Some(Self::BFloatWideFused),
            _ => None,
        }
    }
}

/// An additional result reporting effects of executed elements only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorStatus {
    /// Produces no additional status result.
    None,
    /// Produces IEEE exception bits, inexact through invalid in bits zero to four.
    ///
    /// The result is the union of exceptions from executed elements, and zero
    /// when none execute. The caller combines it with its prior sticky flags.
    FloatingExceptions,
    /// Produces whether any executed element saturated.
    Saturation,
}

/// Integer fixed-point arithmetic with an explicit two-bit rounding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorFixedKind {
    /// Forms a full-precision sum, rounds away one bit, then truncates to SEW.
    AverageAdd,
    /// Forms a SEW+1-bit difference, rounds away one bit, then truncates to SEW.
    ///
    /// Unsigned subtraction wraps at the intermediate width before rounding.
    AverageSub,
    /// Rounds a signed double-width product right by SEW-1, then saturates to SEW.
    FractionalMultiply,
    /// Rounds a right shift whose count is reduced modulo SEW.
    Shift,
    /// Rounds a double-width source right, then saturates to SEW.
    ///
    /// The shift count is reduced modulo 2*SEW. Rounding precedes saturation.
    Clip,
}

impl VectorFixedKind {
    /// Returns whether this kernel reports saturation of an executed element.
    #[must_use]
    pub const fn saturates(self) -> bool {
        matches!(self, Self::FractionalMultiply | Self::Clip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            FloatRules, VectorConst,
            ops::{
                BcdAdjustKind, CarrylessPart, FlagsMask, MemoryEffectLocation, MemoryTagKind,
                PacKind, SsaEffectKind, VectorBinaryKind, VectorCastKind, VectorCompareKind,
                VectorCryptoKind, VectorElement, VectorElementKind, VectorFaultMode,
                VectorMaskBinaryKind, VectorReduceKind, VectorUnaryKind,
            },
        },
        target::VectorTailPolicy,
    };

    fn execution() -> Computation {
        Computation {
            canonical_nan: false,
            kernel: Kernel::Binary(VectorBinaryKind::Add),
            element: VectorElement::integer(32, None),
            predicate: VectorPredicateLayout::Packed,
            rounding: VectorRounding::None,
            status: VectorStatus::None,
            lanes: Some(LaneControl {
                mask_tail: false,
                tail: VectorTailPolicy::Undisturbed,
                inactive: Some(VectorInactivePolicy::Undisturbed),
                no_passthrough: false,
            }),
        }
    }

    /// The lane control of a computation over lanes.
    fn lanes(kind: &mut Computation) -> &mut LaneControl {
        kind.lanes.as_mut().expect("a computation over lanes")
    }

    #[test]
    fn controlled_float_kernels_validate_clipping_sign_classification_and_estimates() {
        let mut kind = execution();
        for bits in [8u32, 16, 32, 64, 128] {
            for unsigned in [false, true] {
                for nan_maximum in [false, true] {
                    kind.kernel = Kernel::FloatToInteger {
                        unsigned,
                        nan_maximum,
                    };
                    kind.element = VectorElement::integer(bits, Some(!unsigned));
                    kind.rounding = VectorRounding::FloatingEnvironment;
                    kind.status = VectorStatus::FloatingExceptions;
                    assert!(kind.is_valid());
                    assert_eq!(kind.input_count(), 6); // restart, length, mask, environment, old, input.
                    assert_eq!(kind.output_count(), 2);
                    kind.rounding = VectorRounding::TowardZero;
                    assert!(kind.is_valid());
                    kind.rounding = VectorRounding::None;
                    assert!(!kind.is_valid());
                    kind.rounding = VectorRounding::FloatingEnvironment;
                    kind.element.signed = Some(unsigned);
                    assert!(!kind.is_valid());
                }
            }
            kind.element = VectorElement::integer(bits, None);
            kind.kernel = Kernel::FloatClassify;
            kind.rounding = VectorRounding::None;
            kind.status = VectorStatus::None;
            assert_eq!(kind.is_valid(), bits >= 16);
            assert_eq!(kind.input_count(), 5);
            kind.element = VectorElement::float(bits);
            for (negate, xor) in [(false, false), (true, false), (false, true), (true, true)] {
                kind.kernel = Kernel::SignInject { negate, xor };
                assert_eq!(kind.is_valid(), bits >= 16 && !(negate && xor));
                assert_eq!(kind.input_count(), 6);
                kind.canonical_nan = true;
                assert!(!kind.is_valid());
                kind.canonical_nan = false;
            }
            for square_root in [false, true] {
                kind.kernel = Kernel::Estimate7 { square_root };
                kind.canonical_nan = true;
                kind.status = VectorStatus::FloatingExceptions;
                kind.rounding = if square_root {
                    VectorRounding::None
                } else {
                    VectorRounding::FloatingEnvironment
                };
                assert_eq!(kind.is_valid(), bits >= 16);
                assert_eq!(kind.input_count(), if square_root { 5 } else { 6 });
                kind.rounding = VectorRounding::ToOdd;
                assert!(!kind.is_valid());
                kind.canonical_nan = false;
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn controlled_float_kernel_metadata_round_trips_every_new_policy() {
        for kernel in [
            Kernel::SignInject {
                negate: false,
                xor: false,
            },
            Kernel::SignInject {
                negate: true,
                xor: false,
            },
            Kernel::SignInject {
                negate: false,
                xor: true,
            },
            Kernel::FloatClassify,
            Kernel::FloatToInteger {
                unsigned: false,
                nan_maximum: false,
            },
            Kernel::FloatToInteger {
                unsigned: false,
                nan_maximum: true,
            },
            Kernel::FloatToInteger {
                unsigned: true,
                nan_maximum: false,
            },
            Kernel::FloatToInteger {
                unsigned: true,
                nan_maximum: true,
            },
            Kernel::Estimate7 { square_root: false },
            Kernel::Estimate7 { square_root: true },
        ] {
            let text = serde_json::to_string(&kernel).unwrap();
            assert_eq!(serde_json::from_str::<Kernel>(&text).unwrap(), kernel);
        }
    }

    #[test]
    fn canonical_nan_policy_applies_only_to_executed_float_arithmetic() {
        let mut kind = execution();
        kind.element = VectorElement::float(32);
        kind.canonical_nan = true;
        assert!(kind.is_valid());
        kind.element.kind = VectorElementKind::BFloat16;
        kind.element.bits = 16;
        kind.kernel = Kernel::Cast {
            from: VectorCastKind::Float,
            to: VectorCastKind::BFloat16,
        };
        assert!(kind.is_valid());
        kind.element = VectorElement::float(32);
        kind.kernel = Kernel::Binary(VectorBinaryKind::Add);
        for action in [
            VectorLaneAction::Execute,
            VectorLaneAction::Preserve,
            VectorLaneAction::Zero,
            VectorLaneAction::Agnostic,
            VectorLaneAction::MaskTail,
        ] {
            assert_eq!(
                kind.canonicalizes_nan(action),
                action == VectorLaneAction::Execute
            );
        }
        for kernel in [
            Kernel::Move,
            Kernel::Select,
            Kernel::Compare {
                relation: VectorCompareKind::Eq,
                signaling: false,
            },
            Kernel::Binary(VectorBinaryKind::And),
            Kernel::Unary(VectorUnaryKind::Identity),
            Kernel::Cast {
                from: VectorCastKind::Float,
                to: VectorCastKind::Signed,
            },
        ] {
            kind.kernel = kernel;
            assert!(!kind.is_valid(), "{kernel:?}");
        }
        kind.kernel = Kernel::Binary(VectorBinaryKind::Add);
        kind.element = VectorElement::integer(32, None);
        assert!(!kind.is_valid());
    }

    /// **Rounding to odd belongs to a narrowing between float formats.** A
    /// float-to-float cast may round to odd, as Arm's `fcvtxn` and RISC-V's
    /// `vfncvt.rod.f.f.w` do; no arithmetic and no conversion to or from an
    /// integer does.
    #[test]
    fn rounding_to_odd_belongs_to_a_narrowing_between_float_formats() {
        let mut kind = execution();
        kind.element = VectorElement::float(32);
        kind.status = VectorStatus::FloatingExceptions;
        kind.rounding = VectorRounding::ToOdd;
        for (from, to) in [
            (VectorCastKind::Float, VectorCastKind::Float),
            (VectorCastKind::Float, VectorCastKind::BFloat16),
        ] {
            kind.kernel = Kernel::Cast { from, to };
            if to == VectorCastKind::BFloat16 {
                kind.element.kind = VectorElementKind::BFloat16;
                kind.element.bits = 16;
            }
            assert!(kind.is_valid(), "{from:?} to {to:?}");
        }
        kind.element = VectorElement::float(32);
        for kernel in [
            Kernel::Binary(VectorBinaryKind::Add),
            Kernel::Binary(VectorBinaryKind::Mul),
            Kernel::Unary(VectorUnaryKind::Sqrt),
            Kernel::FusedMultiplyAdd {
                negate_product: false,
                negate_addend: false,
            },
            Kernel::Cast {
                from: VectorCastKind::Signed,
                to: VectorCastKind::Float,
            },
        ] {
            kind.kernel = kernel;
            assert!(!kind.is_valid(), "{kernel:?}");
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn canonical_nan_metadata_preserves_default_json_and_control_arity() {
        let original = execution();
        let text = serde_json::to_string(&original).unwrap();
        assert!(!text.contains("canonical_nan"));
        assert_eq!(
            serde_json::from_str::<Computation>(&text).unwrap(),
            original
        );
        let mut extended = original;
        extended.element = VectorElement::float(32);
        extended.canonical_nan = true;
        let text = serde_json::to_string(&extended).unwrap();
        assert!(text.contains("\"canonical_nan\":true"));
        assert_eq!(
            serde_json::from_str::<Computation>(&text).unwrap(),
            extended
        );
        assert_eq!(extended.input_count(), original.input_count());
        assert_eq!(
            extended.control_input_count(),
            original.control_input_count()
        );
        assert_eq!(extended.output_count(), original.output_count());
        extended.canonical_nan = false;
        assert!(!extended.canonicalizes_nan(VectorLaneAction::Execute));
    }

    #[test]
    fn float_capability_query_selectors_preserve_operation_classes() {
        for (selector, capability) in [
            (0, VectorFloatCapability::Arithmetic),
            (1, VectorFloatCapability::BFloatConversions),
            (2, VectorFloatCapability::HalfConversions),
            (3, VectorFloatCapability::BFloatWideFused),
        ] {
            assert_eq!(capability.selector(), selector);
            assert_eq!(
                VectorFloatCapability::from_selector(selector),
                Some(capability)
            );
        }
        for invalid in [4, 255, u64::MAX] {
            assert_eq!(VectorFloatCapability::from_selector(invalid), None);
        }
    }

    #[test]
    fn permutation_slides_preserve_low_holes_and_use_source_capacity() {
        for (up, scalar_fill, offset, expected) in [
            (
                true,
                false,
                3u64,
                vec![
                    VectorSlideSource::Preserve,
                    VectorSlideSource::Preserve,
                    VectorSlideSource::Preserve,
                    VectorSlideSource::Element(0),
                    VectorSlideSource::Element(1),
                ],
            ),
            (
                false,
                false,
                3,
                vec![
                    VectorSlideSource::Element(3),
                    VectorSlideSource::Element(4),
                    VectorSlideSource::Element(5),
                    VectorSlideSource::Element(6),
                    VectorSlideSource::Element(7),
                ],
            ),
            (
                false,
                false,
                5,
                vec![
                    VectorSlideSource::Element(5),
                    VectorSlideSource::Element(6),
                    VectorSlideSource::Element(7),
                    VectorSlideSource::Zero,
                    VectorSlideSource::Zero,
                ],
            ),
            (
                true,
                true,
                99,
                vec![
                    VectorSlideSource::Scalar,
                    VectorSlideSource::Element(0),
                    VectorSlideSource::Element(1),
                    VectorSlideSource::Element(2),
                    VectorSlideSource::Element(3),
                ],
            ),
            (
                false,
                true,
                99,
                vec![
                    VectorSlideSource::Element(1),
                    VectorSlideSource::Element(2),
                    VectorSlideSource::Element(3),
                    VectorSlideSource::Element(4),
                    VectorSlideSource::Scalar,
                ],
            ),
            (false, false, u64::MAX, vec![VectorSlideSource::Zero; 5]),
        ] {
            let mut kind = execution();
            kind.kernel = Kernel::Slide { up, scalar_fill };
            lanes(&mut kind).inactive = Some(VectorInactivePolicy::Agnostic);
            lanes(&mut kind).tail = VectorTailPolicy::Agnostic;
            for (index, expected) in expected.into_iter().enumerate() {
                assert_eq!(
                    kind.slide_source(index as u64, 5, 8, offset),
                    Some(expected)
                );
            }
            assert_eq!(kind.slide_source(5, 5, 8, offset), None);
            for index in 0..10 {
                let expected = if up && !scalar_fill && index < offset && index < 5 {
                    VectorLaneAction::Preserve
                } else {
                    VectorLaneAction::Agnostic
                };
                assert_eq!(kind.lane_action(index, 0, 5, false, 0, offset), expected);
            }
            for (start, length) in [(0u64, 0u64), (5, 5), (6, 5)] {
                for index in 0..10 {
                    assert_eq!(
                        kind.lane_action(index, start, length, true, 0, offset),
                        VectorLaneAction::Preserve
                    );
                }
            }
        }
        let mut kind = execution();
        kind.kernel = Kernel::Slide {
            up: true,
            scalar_fill: false,
        };
        lanes(&mut kind).tail = VectorTailPolicy::Agnostic;
        assert_eq!(
            kind.lane_action(5, 0, 5, true, 0, 7),
            VectorLaneAction::Agnostic
        );
        assert_eq!(
            kind.lane_action(4, 0, 5, true, 0, 7),
            VectorLaneAction::Preserve
        );
        kind.kernel = Kernel::Slide {
            up: false,
            scalar_fill: false,
        };
        assert_eq!(kind.slide_source(1, 5, 4, 3), Some(VectorSlideSource::Zero));
        assert_eq!(
            kind.slide_source(1, 5, 8, u64::MAX),
            Some(VectorSlideSource::Zero)
        );
    }

    #[test]
    fn widening_left_shift_matches_every_byte_pair_at_the_doubled_width() {
        for data in 0u16..=255 {
            for count in 0u16..=255 {
                let left = VectorConst::from_lanes(8, [u128::from(data)]).unwrap();
                let right = VectorConst::from_lanes(8, [u128::from(count)]).unwrap();
                let result = left.widen_shift_left(&right).unwrap();
                assert_eq!(result.lane_bits(), 16);
                assert_eq!(
                    result.lane(0),
                    u128::from(data.wrapping_shl(u32::from(count & 15)))
                );
            }
        }
    }

    #[test]
    fn widening_left_shift_matches_independent_bit_position_oracles_and_scalar_counts() {
        for source_bits in [8u32, 16, 32, 64] {
            let result_bits = source_bits * 2;
            let mask = (1u128 << source_bits) - 1;
            let source =
                VectorConst::from_lanes(source_bits, [0, 1, mask, 1u128 << (source_bits - 1)])
                    .unwrap();
            for raw_count in [
                0u128,
                1,
                u128::from(source_bits),
                u128::from(result_bits - 1),
                u128::from(result_bits),
                u128::MAX,
            ] {
                let count = VectorConst::from_lanes(128, [raw_count]).unwrap();
                let result = source.widen_shift_left(&count).unwrap();
                let amount = (raw_count % u128::from(result_bits)) as u32;
                for index in 0..4 {
                    let mut expected = 0u128;
                    for bit in 0..source_bits {
                        if source.lane(index) & (1u128 << bit) != 0 && bit + amount < result_bits {
                            expected |= 1u128 << (bit + amount);
                        }
                    }
                    assert_eq!(result.lane(index), expected);
                }
            }
        }
        assert!(
            VectorConst::splat(128, 1, 1)
                .unwrap()
                .widen_shift_left(&VectorConst::splat(8, 1, 0).unwrap())
                .is_none()
        );
        assert!(
            VectorConst::splat(8, 3, 1)
                .unwrap()
                .widen_shift_left(&VectorConst::splat(8, 2, 0).unwrap())
                .is_none()
        );
        let wide_count = VectorConst::from_lanes(8, (0..32).map(|index| u128::from(index == 25)))
            .unwrap()
            .relaned(256)
            .unwrap();
        assert!(
            VectorConst::splat(8, 1, 1)
                .unwrap()
                .widen_shift_left(&wide_count)
                .is_none()
        );
    }

    #[test]
    fn folds_and_compression_match_independent_selected_byte_sequences() {
        let values = [255u128, 128, 1, 127, 2, 3, 4, 5];
        let source = VectorConst::from_lanes(8, values).unwrap();
        for mask in 0u16..=255 {
            let selector =
                VectorConst::from_lanes(1, (0..8).map(|bit| u128::from((mask >> bit) & 1)))
                    .unwrap();
            for length in [1u64, 5, 8] {
                let selected: Vec<_> = values
                    .iter()
                    .enumerate()
                    .take(length as usize)
                    .filter(|(index, _)| mask & (1 << index) != 0)
                    .map(|(_, value)| *value)
                    .collect();
                let (packed, count) = source
                    .compress_selected(&selector, length, VectorPredicateLayout::Packed)
                    .unwrap();
                assert_eq!(count, selected.len() as u64);
                assert_eq!(
                    packed
                        .lane_values()
                        .take(selected.len())
                        .collect::<Vec<_>>(),
                    selected
                );
                for kind in [
                    VectorReduceKind::Add,
                    VectorReduceKind::And,
                    VectorReduceKind::Or,
                    VectorReduceKind::Xor,
                    VectorReduceKind::Min,
                    VectorReduceKind::Max,
                ] {
                    for signed in [false, true] {
                        let initial = 254u128;
                        let expected = selected.iter().fold(initial, |sum, value| match kind {
                            VectorReduceKind::Add => (sum + value) & 255,
                            VectorReduceKind::And => sum & value,
                            VectorReduceKind::Or => sum | value,
                            VectorReduceKind::Xor => sum ^ value,
                            VectorReduceKind::Min if signed => {
                                if (sum as u8 as i8) < (*value as u8 as i8) {
                                    sum
                                } else {
                                    *value
                                }
                            }
                            VectorReduceKind::Max if signed => {
                                if (sum as u8 as i8) > (*value as u8 as i8) {
                                    sum
                                } else {
                                    *value
                                }
                            }
                            VectorReduceKind::Min => sum.min(*value),
                            VectorReduceKind::Max => sum.max(*value),
                            _ => unreachable!(),
                        });
                        assert_eq!(
                            source.reduce_selected(
                                kind,
                                VectorElement::integer(8, Some(signed)),
                                initial,
                                Some(&selector),
                                length,
                                VectorPredicateLayout::Packed
                            ),
                            Some(expected)
                        );
                    }
                }
            }
        }
        let zero = VectorConst::splat(1, 8, 0).unwrap();
        assert_eq!(
            source
                .compress_selected(&zero, 0, VectorPredicateLayout::Packed)
                .unwrap()
                .1,
            0
        );
        assert_eq!(
            source.reduce_selected(
                VectorReduceKind::Add,
                VectorElement::integer(16, Some(true)),
                42,
                Some(&zero),
                8,
                VectorPredicateLayout::Packed
            ),
            Some(42)
        );
        assert!(
            source
                .reduce_selected(
                    VectorReduceKind::Add,
                    VectorElement::integer(16, Some(true)),
                    42,
                    None,
                    0,
                    VectorPredicateLayout::Packed
                )
                .is_none()
        );
    }

    #[test]
    fn folds_and_compression_preserve_wide_numeric_domains_and_predicate_layout() {
        for bits in [8u32, 16, 32, 64] {
            let high = 1u128 << (bits - 1);
            let source =
                VectorConst::from_lanes(bits, [high, (high << 1) - 1, 1, high - 1]).unwrap();
            assert_eq!(
                source.reduce_selected(
                    VectorReduceKind::Add,
                    VectorElement::integer(bits * 2, Some(true)),
                    0,
                    None,
                    4,
                    VectorPredicateLayout::Packed
                ),
                Some(u128::MAX >> (128 - bits * 2))
            );
            assert_eq!(
                source.reduce_selected(
                    VectorReduceKind::Add,
                    VectorElement::integer(bits * 2, Some(false)),
                    0,
                    None,
                    4,
                    VectorPredicateLayout::Packed
                ),
                Some((1u128 << bits) * 2 - 1)
            );
        }
        let source = VectorConst::from_lanes(128, [u128::MAX, 1u128 << 127, 7]).unwrap();
        for (kind, expected) in [
            (VectorReduceKind::Min, 7),
            (VectorReduceKind::Max, u128::MAX),
        ] {
            assert_eq!(
                source.reduce(
                    kind,
                    VectorElement::integer(128, Some(false)),
                    None,
                    false,
                    None,
                    FloatRules::default()
                ),
                Some(expected)
            );
        }
        let source = VectorConst::from_lanes(32, [11, 22, 33, 44]).unwrap();
        let selector = VectorConst::from_lanes(
            1,
            (0..16).map(|bit| u128::from(bit % 4 != 0 || matches!(bit / 4, 0 | 2))),
        )
        .unwrap();
        let (packed, count) = source
            .compress_selected(&selector, 4, VectorPredicateLayout::ByteGranular)
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(packed.lane_values().take(2).collect::<Vec<_>>(), [11, 33]);
        assert_eq!(
            source.reduce_selected(
                VectorReduceKind::Add,
                VectorElement::integer(64, Some(false)),
                100,
                Some(&selector),
                4,
                VectorPredicateLayout::ByteGranular
            ),
            Some(144)
        );
        assert!(
            source
                .compress_selected(
                    &VectorConst::splat(1, 1, 1).unwrap(),
                    4,
                    VectorPredicateLayout::Packed
                )
                .is_none()
        );
        assert!(
            source
                .compress_selected(&selector, 5, VectorPredicateLayout::ByteGranular)
                .is_none()
        );
        let short = VectorConst::splat(1, 4, 1).unwrap();
        assert!(
            source
                .compress_selected(&short, 4, VectorPredicateLayout::ByteGranular)
                .is_none()
        );
        let wide = VectorConst::from_lanes(8, (0..32).map(|index| u128::from(index == 25)))
            .unwrap()
            .relaned(256)
            .unwrap();
        assert_eq!(wide.bits_at(200, 1), 1);
        assert!(
            wide.compress_selected(
                &VectorConst::splat(1, 1, 1).unwrap(),
                1,
                VectorPredicateLayout::Packed
            )
            .is_none()
        );
    }

    #[test]
    fn folds_and_compression_apply_result_index_rules_and_zero_length_preservation() {
        let mut kind = execution();
        kind.kernel = Kernel::Reduce {
            kind: VectorReduceKind::Add,
            ordered: false,
        };
        lanes(&mut kind).inactive = Some(VectorInactivePolicy::Agnostic);
        for tail in [VectorTailPolicy::Undisturbed, VectorTailPolicy::Agnostic] {
            lanes(&mut kind).tail = tail;
            for kernel in [
                Kernel::Reduce {
                    kind: VectorReduceKind::Add,
                    ordered: false,
                },
                Kernel::ReduceWide {
                    kind: VectorReduceKind::Add,
                    ordered: false,
                },
            ] {
                kind.kernel = kernel;
                for index in 0..8 {
                    assert_eq!(
                        kind.lane_action(index, 0, 0, false, 0, 0),
                        VectorLaneAction::Preserve
                    );
                    assert_eq!(
                        kind.lane_action(index, 0, 5, false, 0, 0),
                        if index == 0 {
                            VectorLaneAction::Execute
                        } else if tail == VectorTailPolicy::Undisturbed {
                            VectorLaneAction::Preserve
                        } else {
                            VectorLaneAction::Agnostic
                        }
                    );
                }
            }
            kind.kernel = Kernel::Compress;
            lanes(&mut kind).inactive = None;
            for count in 0..=5 {
                for index in 0..8 {
                    assert_eq!(
                        kind.lane_action(index, 0, 5, false, 0, count),
                        if index < count {
                            VectorLaneAction::Execute
                        } else if tail == VectorTailPolicy::Undisturbed {
                            VectorLaneAction::Preserve
                        } else {
                            VectorLaneAction::Agnostic
                        }
                    );
                    assert_eq!(
                        kind.lane_action(index, 0, 0, true, 0, count),
                        VectorLaneAction::Preserve
                    );
                }
            }
        }
        kind.kernel = Kernel::ReduceWide {
            kind: VectorReduceKind::Add,
            ordered: false,
        };
        kind.element = VectorElement::integer(128, Some(false));
        assert!(kind.is_valid());
        kind.status = VectorStatus::Saturation;
        assert!(!kind.is_valid());
        kind.status = VectorStatus::None;
        kind.element.bits = 8;
        assert!(!kind.is_valid());
        kind.element = VectorElement::integer(64, Some(false));
        kind.predicate = VectorPredicateLayout::ByteGranular;
        assert_eq!(kind.predicate_bit(1), Some(4));
        assert_eq!(kind.predicate_bit(3), Some(12));
        kind.element.kind = VectorElementKind::Float;
        kind.element.signed = None;
        kind.element.bits = 16;
        assert!(!kind.is_valid());
        kind.element.bits = 32;
        assert!(kind.is_valid());
    }

    #[test]
    fn mask_scans_match_exhaustive_independent_selected_position_oracles() {
        for source in 0u16..=255 {
            for governor in 0u16..=255 {
                let source_bits =
                    VectorConst::from_lanes(1, (0..8).map(|bit| u128::from((source >> bit) & 1)))
                        .unwrap();
                let governing =
                    VectorConst::from_lanes(1, (0..8).map(|bit| u128::from((governor >> bit) & 1)))
                        .unwrap();
                for length in [0u64, 1, 3, 8] {
                    let enabled: Vec<_> = (0..length)
                        .filter(|bit| (source & governor) & (1 << bit) != 0)
                        .collect();
                    assert_eq!(
                        source_bits.mask_summary(
                            Some(&governing),
                            length,
                            VectorPredicateLayout::Packed,
                            1
                        ),
                        Some((
                            enabled.len() as u64,
                            enabled.first().copied().unwrap_or(u64::MAX)
                        ))
                    );
                    for prefix in [
                        VectorMaskPrefix::Before,
                        VectorMaskPrefix::Including,
                        VectorMaskPrefix::Only,
                    ] {
                        let output = source_bits
                            .mask_prefix(
                                prefix,
                                Some(&governing),
                                length,
                                VectorPredicateLayout::Packed,
                                1,
                            )
                            .unwrap();
                        for bit in 0..8 {
                            let expected = bit < length
                                && governor & (1 << bit) != 0
                                && match prefix {
                                    VectorMaskPrefix::Before => {
                                        enabled.first().is_none_or(|first| bit < *first)
                                    }
                                    VectorMaskPrefix::Including => {
                                        enabled.first().is_none_or(|first| bit <= *first)
                                    }
                                    VectorMaskPrefix::Only => enabled.first() == Some(&bit),
                                };
                            assert_eq!(output.lane(bit as u32), u128::from(expected));
                        }
                    }
                }
                for bit in 0u64..=8 {
                    let expected = ((source & governor) & ((1u16 << bit) - 1)).count_ones();
                    assert_eq!(
                        source_bits.mask_iota_value(
                            Some(&governing),
                            bit,
                            VectorPredicateLayout::Packed,
                            8
                        ),
                        Some(u128::from(expected))
                    );
                }
            }
        }
    }

    #[test]
    fn mask_scans_keep_physical_spacing_capacity_and_wrapped_wide_prefixes() {
        for bits in [8u32, 16, 32, 64, 128] {
            let stride = bits / 8;
            let source = VectorConst::from_lanes(
                1,
                (0..8 * stride).map(|bit| u128::from(bit % stride == 0 && bit / stride % 3 != 0)),
            )
            .unwrap();
            let mask = VectorConst::from_lanes(
                1,
                (0..8 * stride).map(|bit| u128::from(bit % stride == 0 && bit / stride % 2 == 0)),
            )
            .unwrap();
            assert_eq!(
                source.mask_summary(Some(&mask), 8, VectorPredicateLayout::ByteGranular, bits),
                Some((2, 2))
            );
            assert_eq!(
                source.mask_iota_value(Some(&mask), 7, VectorPredicateLayout::ByteGranular, bits),
                Some(2)
            );
            let prefix = source
                .mask_prefix(
                    VectorMaskPrefix::Including,
                    Some(&mask),
                    8,
                    VectorPredicateLayout::ByteGranular,
                    bits,
                )
                .unwrap();
            for bit in 0..8 * stride {
                assert_eq!(prefix.lane(bit), u128::from(bit == 0 || bit == 2 * stride));
            }
        }
        let all = VectorConst::splat(1, 512, 1).unwrap();
        assert_eq!(
            all.mask_iota_value(None, 300, VectorPredicateLayout::Packed, 8),
            Some(44)
        );
        assert_eq!(
            all.mask_iota_value(None, 512, VectorPredicateLayout::Packed, 128),
            Some(512)
        );
        assert_eq!(
            all.mask_summary(None, 512, VectorPredicateLayout::Packed, 1),
            Some((512, 0))
        );
        assert!(
            all.mask_summary(None, 513, VectorPredicateLayout::Packed, 1)
                .is_none()
        );
        assert!(
            all.mask_summary(None, u64::MAX, VectorPredicateLayout::ByteGranular, 64)
                .is_none()
        );
        assert!(
            all.mask_summary(
                Some(&VectorConst::splat(1, 1, 1).unwrap()),
                8,
                VectorPredicateLayout::Packed,
                1
            )
            .is_none()
        );
        assert!(
            all.mask_summary(
                Some(&VectorConst::splat(8, 8, 1).unwrap()),
                8,
                VectorPredicateLayout::Packed,
                1
            )
            .is_none()
        );
        assert!(
            all.mask_summary(None, 0, VectorPredicateLayout::ByteGranular, 1)
                .is_none()
        );
        assert!(
            all.mask_iota_value(None, 1, VectorPredicateLayout::Packed, 256)
                .is_none()
        );
        assert_eq!(
            all.mask_summary(None, 0, VectorPredicateLayout::Packed, 1),
            Some((0, u64::MAX))
        );
    }

    #[test]
    fn mask_scans_distinguish_scalar_zero_length_and_vector_execution_policies() {
        let mut kind = execution();
        kind.element = VectorElement::integer(1, None);
        for first in [false, true] {
            kind.kernel = Kernel::MaskSummary { first };
            lanes(&mut kind).inactive = Some(VectorInactivePolicy::Undisturbed);
            assert!(kind.is_valid());
            assert_eq!(kind.input_count(), 5);
            for length in [0, 1, 16] {
                assert_eq!(
                    kind.lane_action(0, 0, length, false, 0, 0),
                    VectorLaneAction::Execute
                );
                assert_eq!(
                    kind.lane_action(1, 0, length, true, 0, 0),
                    VectorLaneAction::Preserve
                );
                assert_eq!(
                    kind.lane_action(0, 1, length, true, 0, 0),
                    VectorLaneAction::Preserve
                );
            }
            lanes(&mut kind).mask_tail = true;
            lanes(&mut kind).tail = VectorTailPolicy::Agnostic;
            assert!(!kind.is_valid());
            lanes(&mut kind).mask_tail = false;
        }
        kind.kernel = Kernel::MaskPrefix(VectorMaskPrefix::Only);
        lanes(&mut kind).mask_tail = true;
        assert!(kind.is_valid());
        assert_eq!(
            kind.lane_action(0, 0, 0, true, 64, 0),
            VectorLaneAction::Preserve
        );
        assert_eq!(
            kind.lane_action(5, 0, 3, true, 64, 0),
            VectorLaneAction::MaskTail
        );
        assert_eq!(
            kind.lane_action(5, 0, 3, false, 64, 0),
            VectorLaneAction::Agnostic
        );
        kind.element = VectorElement::integer(32, None);
        assert!(!kind.is_valid());
        kind.kernel = Kernel::MaskIota;
        lanes(&mut kind).mask_tail = false;
        assert!(kind.is_valid());
        kind.status = VectorStatus::FloatingExceptions;
        assert!(!kind.is_valid());
    }

    #[test]
    fn permutation_helpers_refuse_lanes_wider_than_the_numeric_representation() {
        let source = VectorConst::from_lanes(8, [10, 11]).unwrap();
        let wide = VectorConst::from_lanes(8, (0..32).map(|index| if index == 25 { 1 } else { 0 }))
            .unwrap()
            .relaned(256)
            .unwrap();
        assert_eq!(wide.bits_at(200, 1), 1);
        assert!(source.gather(&wide).is_none());
        assert!(wide.gather(&VectorConst::splat(8, 1, 0).unwrap()).is_none());
        assert!(
            wide.select_mask(
                &wide,
                &VectorConst::splat(1, 1, 1).unwrap(),
                VectorPredicateLayout::Packed
            )
            .is_none()
        );
    }

    #[test]
    fn permutation_gathers_do_not_narrow_unsigned_indices_or_bound_them_by_vl() {
        let source = VectorConst::from_lanes(8, 10u128..18).unwrap();
        let indices = VectorConst::from_lanes(16, [0, 7, 8, 256, 2, 4, 65535, 1]).unwrap();
        let expected = [10u128, 17, 0, 0, 12, 14, 0, 11];
        assert_eq!(
            source
                .gather(&indices)
                .unwrap()
                .lane_values()
                .collect::<Vec<_>>(),
            expected
        );
        for index in [8u128, 256, 1 << 32, 1 << 63, u128::MAX] {
            let indices = VectorConst::splat(128, 1, index).unwrap();
            assert!(
                source
                    .gather(&indices)
                    .unwrap()
                    .lane_values()
                    .all(|value| value == 0)
            );
        }
        let indices = VectorConst::splat(64, 1, 7).unwrap();
        assert!(
            source
                .gather(&indices)
                .unwrap()
                .lane_values()
                .all(|value| value == 17)
        );
        assert!(
            source
                .gather(&VectorConst::splat(8, 2, 0).unwrap())
                .is_none()
        );
    }

    #[test]
    fn permutation_selections_follow_packed_and_byte_granular_physical_bits() {
        for bits in [8u32, 16, 32, 64, 128] {
            let left = VectorConst::from_lanes(bits, [10, 20, 30, 40]).unwrap();
            let right = VectorConst::from_lanes(bits, [1, 2, 3, 4]).unwrap();
            for layout in [
                VectorPredicateLayout::Packed,
                VectorPredicateLayout::ByteGranular,
            ] {
                let stride = if layout == VectorPredicateLayout::Packed {
                    1
                } else {
                    bits / 8
                };
                let mask = VectorConst::from_lanes(
                    1,
                    (0..4 * stride).map(|bit| {
                        if bit % stride == 0 {
                            u128::from((bit / stride) % 2 == 0)
                        } else {
                            1
                        }
                    }),
                )
                .unwrap();
                assert_eq!(
                    left.select_mask(&right, &mask, layout)
                        .unwrap()
                        .lane_values()
                        .collect::<Vec<_>>(),
                    [1, 20, 3, 40]
                );
                let scalar = VectorConst::splat(bits, 1, 99).unwrap();
                assert_eq!(
                    left.select_mask(&scalar, &mask, layout)
                        .unwrap()
                        .lane_values()
                        .collect::<Vec<_>>(),
                    [99, 20, 99, 40]
                );
            }
            for flag in [0u128, 1] {
                let mask = VectorConst::splat(1, 1, flag).unwrap();
                assert_eq!(
                    left.select_mask(&right, &mask, VectorPredicateLayout::Packed)
                        .unwrap(),
                    if flag == 0 {
                        left.clone()
                    } else {
                        right.clone()
                    }
                );
            }
        }
        let left = VectorConst::from_lanes(32, [0x7fc00042, 0x7f800001, 0x80000000, 1]).unwrap();
        let right = VectorConst::from_lanes(32, [0x7f800005, 0xffc00013, 0, 0x80000000]).unwrap();
        let mask = VectorConst::from_lanes(1, [1, 0, 1, 0]).unwrap();
        assert_eq!(
            left.select_mask(&right, &mask, VectorPredicateLayout::Packed)
                .unwrap()
                .lane_values()
                .collect::<Vec<_>>(),
            [0x7f800005, 0x7f800001, 0, 1]
        );
        assert!(
            left.select_mask(
                &right,
                &VectorConst::splat(1, 2, 1).unwrap(),
                VectorPredicateLayout::Packed
            )
            .is_none()
        );
        assert!(
            left.select_mask(
                &right,
                &VectorConst::splat(8, 4, 1).unwrap(),
                VectorPredicateLayout::Packed
            )
            .is_none()
        );
    }

    #[test]
    fn integer_carry_matches_all_byte_pairs_both_inputs_and_outputs() {
        let left = VectorConst::from_lanes(8, 0u128..256).unwrap();
        for b in 0u128..256 {
            let right = VectorConst::splat(8, 256, b).unwrap();
            for c in 0u128..=1 {
                let carry = VectorConst::splat(1, 256, c).unwrap();
                for subtract in [false, true] {
                    let (data, outgoing) =
                        left.integer_with_carry(&right, &carry, subtract).unwrap();
                    for a in 0u128..256 {
                        let exact = if subtract {
                            a as i32 - b as i32 - c as i32
                        } else {
                            a as i32 + b as i32 + c as i32
                        };
                        assert_eq!(data.lane(a as u32), exact.rem_euclid(256) as u128);
                        assert_eq!(
                            outgoing.lane(a as u32),
                            u128::from(if subtract { exact < 0 } else { exact >= 256 })
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn integer_carry_keeps_wide_overflow_broadcasts_and_extra_mask_bits() {
        for bits in [16u32, 32, 64] {
            let maximum = (1u128 << bits) - 1;
            let sign = 1u128 << (bits - 1);
            for a in [0, 1, sign - 1, sign, maximum] {
                for b in [0, 1, sign - 1, sign, maximum] {
                    for c in [0u128, 1] {
                        for subtract in [false, true] {
                            let left = VectorConst::from_lanes(bits, [a]).unwrap();
                            let right = VectorConst::from_lanes(bits, [b]).unwrap();
                            let carry = VectorConst::from_lanes(1, [c]).unwrap();
                            let (data, outgoing) =
                                left.integer_with_carry(&right, &carry, subtract).unwrap();
                            let exact = if subtract {
                                a as i128 - b as i128 - c as i128
                            } else {
                                a as i128 + b as i128 + c as i128
                            };
                            assert_eq!(data.lane(0), exact as u128 & maximum);
                            assert_eq!(
                                outgoing.lane(0),
                                u128::from(if subtract {
                                    exact < 0
                                } else {
                                    exact > maximum as i128
                                })
                            );
                        }
                    }
                }
            }
        }
        for (a, b, c, subtract, expected, bit) in [
            (u128::MAX, 0, 1, false, 0, 1),
            (u128::MAX, u128::MAX, 1, false, u128::MAX, 1),
            (1u128 << 127, 1u128 << 127, 0, false, 0, 1),
            (0, u128::MAX, 1, true, 0, 1),
            (0, 0, 1, true, u128::MAX, 1),
            (u128::MAX, u128::MAX, 1, true, u128::MAX, 1),
            (u128::MAX, 0, 1, true, u128::MAX - 1, 0),
            (u128::from(u64::MAX), 1, 0, false, 1u128 << 64, 0),
        ] {
            let left = VectorConst::from_lanes(128, [a]).unwrap();
            let right = VectorConst::from_lanes(128, [b]).unwrap();
            let carry = VectorConst::from_lanes(1, [c]).unwrap();
            let (data, outgoing) = left.integer_with_carry(&right, &carry, subtract).unwrap();
            assert_eq!(data.lane(0), expected);
            assert_eq!(outgoing.lane(0), bit);
        }
        let left = VectorConst::from_lanes(8, [0, 1, 127, 128, 254, 255, 13, 42]).unwrap();
        let right = VectorConst::splat(8, 1, 1).unwrap();
        for carry in [
            VectorConst::splat(1, 1, 1).unwrap(),
            VectorConst::splat(1, 64, 1).unwrap(),
        ] {
            let (data, outgoing) = left.integer_with_carry(&right, &carry, false).unwrap();
            for index in 0..8 {
                assert_eq!(data.lane(index), (left.lane(index) + 2) & 255);
                assert_eq!(outgoing.lane(index), u128::from(left.lane(index) >= 254));
            }
        }
        assert!(
            left.integer_with_carry(&right, &VectorConst::splat(1, 4, 0).unwrap(), false)
                .is_none()
        );
        assert!(
            left.integer_with_carry(
                &VectorConst::splat(8, 2, 0).unwrap(),
                &VectorConst::splat(1, 1, 0).unwrap(),
                false
            )
            .is_none()
        );
        assert!(
            left.integer_with_carry(&right, &VectorConst::splat(8, 8, 0).unwrap(), false)
                .is_none()
        );
    }

    #[test]
    fn integer_carry_domains_never_treat_incoming_bits_as_governing_predicates() {
        for subtract in [false, true] {
            for mask_result in [false, true] {
                let mut kind = execution();
                kind.kernel = Kernel::IntegerCarry {
                    subtract,
                    mask_result,
                };
                kind.element = VectorElement::integer(32, None);
                lanes(&mut kind).mask_tail = mask_result;
                lanes(&mut kind).tail = VectorTailPolicy::Agnostic;
                lanes(&mut kind).inactive = None;
                assert!(kind.is_valid());
                assert_eq!(kind.input_count(), 6 + usize::from(mask_result));
                assert_eq!(kind.output_count(), 1);
                assert_eq!(
                    kind.lane_action(0, 0, 1, false, 64, 0),
                    VectorLaneAction::Execute
                );
                assert!(
                    !Computation {
                        lanes: kind.lanes.map(|lanes| LaneControl {
                            inactive: Some(VectorInactivePolicy::Undisturbed),
                            ..lanes
                        }),
                        ..kind
                    }
                    .is_valid()
                );
                assert!(
                    !Computation {
                        rounding: VectorRounding::FixedEnvironment,
                        ..kind
                    }
                    .is_valid()
                );
                assert!(
                    !Computation {
                        status: VectorStatus::Saturation,
                        ..kind
                    }
                    .is_valid()
                );
                assert!(
                    !Computation {
                        element: VectorElement::integer(32, Some(false)),
                        ..kind
                    }
                    .is_valid()
                );
                if !mask_result {
                    assert!(
                        !Computation {
                            lanes: kind.lanes.map(|lanes| LaneControl {
                                mask_tail: true,
                                ..lanes
                            }),
                            ..kind
                        }
                        .is_valid()
                    );
                }
            }
        }
    }

    #[test]
    fn mask_logical_relations_match_complete_byte_truth_tables() {
        for (kind, table) in [
            (VectorMaskBinaryKind::And, 0b1000u8),
            (VectorMaskBinaryKind::Or, 0b1110),
            (VectorMaskBinaryKind::Xor, 0b0110),
            (VectorMaskBinaryKind::AndNot, 0b0100),
            (VectorMaskBinaryKind::OrNot, 0b1101),
            (VectorMaskBinaryKind::Nand, 0b0111),
            (VectorMaskBinaryKind::Nor, 0b0001),
            (VectorMaskBinaryKind::Xnor, 0b1001),
        ] {
            for left in 0u16..=255 {
                for right in 0u16..=255 {
                    let mut expected = 0u8;
                    let mut actual = 0u8;
                    for bit in 0..8 {
                        let a = left & (1 << bit) != 0;
                        let b = right & (1 << bit) != 0;
                        let index = u8::from(a) * 2 + u8::from(b);
                        expected |= ((table >> index) & 1) << bit;
                        actual |= u8::from(kind.evaluate(a, b)) << bit;
                    }
                    assert_eq!(actual, expected, "{kind:?} {left:x} {right:x}");
                }
            }
        }
    }

    #[test]
    fn mask_logical_execution_uses_predicate_domains_and_retains_restart() {
        for relation in [
            VectorMaskBinaryKind::And,
            VectorMaskBinaryKind::Or,
            VectorMaskBinaryKind::Xor,
            VectorMaskBinaryKind::AndNot,
            VectorMaskBinaryKind::OrNot,
            VectorMaskBinaryKind::Nand,
            VectorMaskBinaryKind::Nor,
            VectorMaskBinaryKind::Xnor,
        ] {
            let mut kind = execution();
            kind.kernel = Kernel::MaskBinary(relation);
            kind.element = VectorElement::integer(1, None);
            lanes(&mut kind).inactive = None;
            lanes(&mut kind).tail = VectorTailPolicy::Agnostic;
            lanes(&mut kind).mask_tail = true;
            assert!(kind.is_valid());
            assert_eq!(kind.input_count(), 6);
            assert_eq!(kind.output_count(), 1);
            for (index, start, length, expected) in [
                (0, 1, 8, VectorLaneAction::Preserve),
                (1, 1, 8, VectorLaneAction::Execute),
                (8, 1, 8, VectorLaneAction::MaskTail),
                (63, 1, 8, VectorLaneAction::MaskTail),
                (64, 1, 8, VectorLaneAction::Agnostic),
                (0, 0, 0, VectorLaneAction::Preserve),
                (63, 8, 8, VectorLaneAction::Preserve),
            ] {
                assert_eq!(
                    kind.lane_action(index, start, length, true, 64, 0),
                    expected
                );
            }
            for element in [
                VectorElement::integer(8, None),
                VectorElement::integer(1, Some(false)),
            ] {
                assert!(!Computation { element, ..kind }.is_valid());
            }
            let mut invalid = kind;
            invalid.status = VectorStatus::Saturation;
            assert!(!invalid.is_valid());
            invalid = kind;
            invalid.rounding = VectorRounding::FloatingEnvironment;
            assert!(!invalid.is_valid());
            invalid = kind;
            invalid.element.kind = VectorElementKind::Float;
            assert!(!invalid.is_valid());
            lanes(&mut kind).mask_tail = false;
            kind.element.bits = 8;
            kind.predicate = VectorPredicateLayout::ByteGranular;
            lanes(&mut kind).inactive = Some(VectorInactivePolicy::Zero);
            lanes(&mut kind).tail = VectorTailPolicy::Undisturbed;
            assert!(kind.is_valid());
            assert_eq!(kind.predicate_bit(9), Some(9));
        }
    }

    #[test]
    fn integer_mask_comparisons_match_every_byte_pair_and_wide_boundaries() {
        for signed in [false, true] {
            for relation in [
                VectorCompareKind::Eq,
                VectorCompareKind::Ne,
                VectorCompareKind::Lt,
                VectorCompareKind::Le,
                VectorCompareKind::Gt,
            ] {
                let left = VectorConst::from_lanes(8, 0u128..256).unwrap();
                for b in 0u128..256 {
                    let right = VectorConst::splat(8, 256, b).unwrap();
                    let result = left
                        .compare(&right, relation, VectorElement::integer(8, Some(signed)))
                        .unwrap();
                    for a in 0u128..256 {
                        let a_read = if signed {
                            i128::from(a as u8 as i8)
                        } else {
                            a as i128
                        };
                        let b_read = if signed {
                            i128::from(b as u8 as i8)
                        } else {
                            b as i128
                        };
                        let expected = match relation {
                            VectorCompareKind::Eq => a_read == b_read,
                            VectorCompareKind::Ne => a_read != b_read,
                            VectorCompareKind::Lt => a_read < b_read,
                            VectorCompareKind::Le => a_read <= b_read,
                            VectorCompareKind::Gt => a_read > b_read,
                            _ => unreachable!(),
                        };
                        assert_eq!(result.lane(a as u32), if expected { 255 } else { 0 });
                    }
                }
            }
        }
        for bits in [16u32, 32, 64, 128] {
            let sign = 1u128 << (bits - 1);
            let max = if bits == 128 {
                u128::MAX
            } else {
                (1u128 << bits) - 1
            };
            for a in [0, 1, sign - 1, sign, max] {
                for b in [0, 1, sign - 1, sign, max] {
                    for signed in [false, true] {
                        for relation in [
                            VectorCompareKind::Eq,
                            VectorCompareKind::Ne,
                            VectorCompareKind::Lt,
                            VectorCompareKind::Le,
                            VectorCompareKind::Gt,
                        ] {
                            let left = VectorConst::from_lanes(bits, [a]).unwrap();
                            let right = VectorConst::from_lanes(bits, [b]).unwrap();
                            let result = left
                                .compare(
                                    &right,
                                    relation,
                                    VectorElement::integer(bits, Some(signed)),
                                )
                                .unwrap();
                            let read = |value: u128| match bits {
                                16 => i128::from(value as i16),
                                32 => i128::from(value as i32),
                                64 => i128::from(value as i64),
                                128 => value as i128,
                                _ => unreachable!(),
                            };
                            let order = if signed {
                                read(a).cmp(&read(b))
                            } else {
                                a.cmp(&b)
                            };
                            let expected = match relation {
                                VectorCompareKind::Eq => order.is_eq(),
                                VectorCompareKind::Ne => order.is_ne(),
                                VectorCompareKind::Lt => order.is_lt(),
                                VectorCompareKind::Le => order.is_le(),
                                VectorCompareKind::Gt => order.is_gt(),
                                _ => unreachable!(),
                            };
                            assert_eq!(result.lane(0), if expected { max } else { 0 });
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn computed_mask_tails_have_explicit_bounds_and_no_executed_effects() {
        let mut kind = execution();
        kind.kernel = Kernel::Compare {
            relation: VectorCompareKind::Lt,
            signaling: false,
        };
        lanes(&mut kind).mask_tail = true;
        lanes(&mut kind).tail = VectorTailPolicy::Agnostic;
        assert!(kind.is_valid());
        for inactive in [
            None,
            Some(VectorInactivePolicy::Undisturbed),
            Some(VectorInactivePolicy::Agnostic),
        ] {
            lanes(&mut kind).inactive = inactive;
            assert_eq!(
                kind.control_input_count(),
                3 + usize::from(inactive.is_some())
            );
            assert_eq!(kind.input_count(), 6 + usize::from(inactive.is_some()));
            for limit in [0u64, 4, 8, 64] {
                for start in [0u64, 3, 8] {
                    for length in [0u64, 4, 8] {
                        for index in 0u64..70 {
                            for predicate in [false, true] {
                                let expected = if start >= length || index < start {
                                    VectorLaneAction::Preserve
                                } else if index >= length {
                                    if index < limit && (inactive.is_none() || predicate) {
                                        VectorLaneAction::MaskTail
                                    } else {
                                        VectorLaneAction::Agnostic
                                    }
                                } else if inactive.is_some() && !predicate {
                                    if inactive == Some(VectorInactivePolicy::Undisturbed) {
                                        VectorLaneAction::Preserve
                                    } else {
                                        VectorLaneAction::Agnostic
                                    }
                                } else {
                                    VectorLaneAction::Execute
                                };
                                let action =
                                    kind.lane_action(index, start, length, predicate, limit, 0);
                                assert_eq!(action, expected);
                                assert_eq!(
                                    action.raises_effects(),
                                    expected == VectorLaneAction::Execute
                                );
                            }
                        }
                    }
                }
            }
        }
        lanes(&mut kind).mask_tail = false;
        assert_eq!(
            kind.lane_action(5, 0, 4, true, 64, 0),
            VectorLaneAction::Agnostic
        );
        lanes(&mut kind).mask_tail = true;
        lanes(&mut kind).tail = VectorTailPolicy::Undisturbed;
        assert!(!kind.is_valid());
        lanes(&mut kind).tail = VectorTailPolicy::Agnostic;
        kind.kernel = Kernel::Binary(VectorBinaryKind::Add);
        assert!(!kind.is_valid());
    }

    #[test]
    fn mask_bit_choices_distinguish_computed_tails_from_old_or_ones() {
        for action in [
            VectorLaneAction::Execute,
            VectorLaneAction::Preserve,
            VectorLaneAction::Zero,
            VectorLaneAction::Agnostic,
            VectorLaneAction::MaskTail,
        ] {
            for old in [false, true] {
                for computed in [false, true] {
                    let permitted = match action {
                        VectorLaneAction::Execute => vec![computed],
                        VectorLaneAction::Preserve => vec![old],
                        VectorLaneAction::Zero => vec![false],
                        VectorLaneAction::Agnostic => vec![old, true],
                        VectorLaneAction::MaskTail => vec![old, true, computed],
                    };
                    for candidate in [false, true] {
                        assert_eq!(
                            action.permits_mask_bit(old, computed, candidate),
                            permitted.contains(&candidate)
                        );
                    }
                }
            }
        }
        assert!(!VectorLaneAction::Agnostic.permits_mask_bit(true, false, false));
        assert!(VectorLaneAction::MaskTail.permits_mask_bit(true, false, false));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn computed_mask_tail_metadata_preserves_unextended_interfaces() {
        let original = execution();
        let text = serde_json::to_string(&original).unwrap();
        assert!(!text.contains("mask_tail"));
        assert_eq!(
            serde_json::from_str::<Computation>(&text).unwrap(),
            original
        );
        let mut extended = original;
        extended.kernel = Kernel::Compare {
            relation: VectorCompareKind::Eq,
            signaling: false,
        };
        lanes(&mut extended).tail = VectorTailPolicy::Agnostic;
        lanes(&mut extended).mask_tail = true;
        let text = serde_json::to_string(&extended).unwrap();
        assert!(text.contains("\"mask_tail\":true"));
        assert_eq!(
            serde_json::from_str::<Computation>(&text).unwrap(),
            extended
        );
    }

    #[test]
    fn controlled_execution_preserves_prestart_and_distinguishes_inactive_policies() {
        let expected = [
            (
                None,
                [
                    VectorLaneAction::Preserve,
                    VectorLaneAction::Preserve,
                    VectorLaneAction::Execute,
                    VectorLaneAction::Execute,
                    VectorLaneAction::Execute,
                ],
            ),
            (
                Some(VectorInactivePolicy::Undisturbed),
                [
                    VectorLaneAction::Preserve,
                    VectorLaneAction::Preserve,
                    VectorLaneAction::Execute,
                    VectorLaneAction::Preserve,
                    VectorLaneAction::Execute,
                ],
            ),
            (
                Some(VectorInactivePolicy::Zero),
                [
                    VectorLaneAction::Preserve,
                    VectorLaneAction::Preserve,
                    VectorLaneAction::Execute,
                    VectorLaneAction::Zero,
                    VectorLaneAction::Execute,
                ],
            ),
            (
                Some(VectorInactivePolicy::Agnostic),
                [
                    VectorLaneAction::Preserve,
                    VectorLaneAction::Preserve,
                    VectorLaneAction::Execute,
                    VectorLaneAction::Agnostic,
                    VectorLaneAction::Execute,
                ],
            ),
        ];
        for (inactive, actions) in expected {
            for tail in [VectorTailPolicy::Undisturbed, VectorTailPolicy::Agnostic] {
                let mut kind = execution();
                lanes(&mut kind).inactive = inactive;
                lanes(&mut kind).tail = tail;
                for (index, action) in actions.into_iter().enumerate() {
                    assert_eq!(
                        kind.lane_action(index as u64, 2, 5, index % 2 == 0, 0, 0),
                        action
                    );
                }
                let tail_action = if tail == VectorTailPolicy::Agnostic {
                    VectorLaneAction::Agnostic
                } else {
                    VectorLaneAction::Preserve
                };
                for index in [5, 6, 63, 511, 65535, u64::MAX] {
                    assert_eq!(kind.lane_action(index, 2, 5, true, 0, 0), tail_action);
                }
                for (start, length) in [(0, 0), (5, 5), (6, 5), (u64::MAX, 5), (u64::MAX, u64::MAX)]
                {
                    for index in [0, 1, 4, 5, 8, 65535, u64::MAX] {
                        assert_eq!(
                            kind.lane_action(index, start, length, true, 0, 0),
                            VectorLaneAction::Preserve
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn controlled_execution_effects_preserve_nondeterministic_choices_and_faults() {
        let mut kind = execution();
        assert!(kind.effects().is_pure());
        lanes(&mut kind).tail = VectorTailPolicy::Agnostic;
        let effects = kind.effects();
        assert_eq!(effects.kind, SsaEffectKind::RuntimeDefined);
        assert_eq!(effects.memory, MemoryEffectLocation::None);
        assert!(!effects.is_pure());
        assert!(effects.removable_when_unused());
        assert!(!effects.reads_memory() && !effects.writes_memory());
        for load in [false, true] {
            kind.kernel = Kernel::Memory {
                load,
                access_bits: 32,
                sign_extend: false,
                byte_stream: false,
                coalescible: false,
                indexed: true,
                ordered: true,
                fields: 3,
                fault: VectorFaultMode::Normal,
            };
            assert!(kind.is_valid());
            let effects = kind.effects();
            assert!(effects.may_throw);
            assert_eq!(effects.reads_memory(), load);
            assert_eq!(effects.writes_memory(), !load);
            assert_eq!(kind.data_outputs(), if load { 3 } else { 0 });
            assert_eq!(kind.output_count(), if load { 3 } else { 0 });
            assert_eq!(kind.input_count(), 8);
        }
        kind.kernel = Kernel::Memory {
            load: true,
            access_bits: 32,
            sign_extend: false,
            byte_stream: false,
            coalescible: false,
            indexed: false,
            ordered: true,
            fields: 8,
            fault: VectorFaultMode::FaultOnlyFirst,
        };
        assert_eq!((kind.output_count(), kind.input_count()), (9, 13));
        kind.kernel = Kernel::Memory {
            load: false,
            access_bits: 32,
            sign_extend: false,
            byte_stream: false,
            coalescible: false,
            indexed: false,
            ordered: true,
            fields: 8,
            fault: VectorFaultMode::FirstFault,
        };
        assert!(!kind.is_valid());
        kind = execution();
        kind.element.kind = VectorElementKind::Float;
        kind.rounding = VectorRounding::FloatingEnvironment;
        kind.status = VectorStatus::FloatingExceptions;
        assert_eq!((kind.output_count(), kind.input_count()), (2, 7));
    }

    #[test]
    fn controlled_execution_predicate_layout_preserves_physical_bit_spacing() {
        let mut kind = execution();
        assert_eq!(kind.predicate_bit(7), Some(7));
        assert_eq!(kind.predicate_bit(u64::MAX), Some(u64::MAX));
        kind.predicate = VectorPredicateLayout::ByteGranular;
        for (bits, seventh) in [(8, 7), (16, 14), (32, 28), (64, 56), (128, 112)] {
            kind.element.bits = bits;
            assert!(kind.is_valid());
            assert_eq!(kind.predicate_bit(0), Some(0));
            assert_eq!(kind.predicate_bit(7), Some(seventh));
        }
        assert_eq!(kind.predicate_bit(u64::MAX), None);
        kind.element.bits = 7;
        assert!(!kind.is_valid());
        assert_eq!(kind.predicate_bit(0), None);
    }
    #[test]
    fn controlled_execution_integer_domains_are_explicit() {
        for operation in [
            VectorUnaryKind::TrailingZeros,
            VectorUnaryKind::ByteSwap,
            VectorUnaryKind::BitReverseIn8,
        ] {
            let mut kind = execution();
            kind.kernel = Kernel::Unary(operation);
            assert!(kind.is_valid());
            kind.element.kind = VectorElementKind::Float;
            assert!(!kind.is_valid());
            kind.element.kind = VectorElementKind::Integer;
            kind.element.bits = 7;
            assert_eq!(kind.is_valid(), operation == VectorUnaryKind::TrailingZeros);
        }
        for unsigned in [false, true] {
            for remainder in [false, true] {
                let mut kind = execution();
                kind.kernel = Kernel::IntegerDivide {
                    remainder,
                    unsigned,
                };
                kind.element.signed = Some(!unsigned);
                assert!(kind.is_valid());
                kind.element.signed = Some(unsigned);
                assert!(!kind.is_valid());
                kind.element.signed = Some(!unsigned);
                kind.element.kind = VectorElementKind::Float;
                assert!(!kind.is_valid());
            }
        }
        for operation in [
            VectorBinaryKind::Shl,
            VectorBinaryKind::Shr,
            VectorBinaryKind::Rol,
            VectorBinaryKind::Ror,
            VectorBinaryKind::Add,
        ] {
            for width in [7, 8, 16, 32, 64, 128, 129] {
                let mut kind = execution();
                kind.kernel = Kernel::ModuloShift { kind: operation };
                kind.element.bits = width;
                assert_eq!(
                    kind.is_valid(),
                    width.is_power_of_two() && operation != VectorBinaryKind::Add
                );
            }
        }
    }

    #[test]
    fn scalar_insertion_preserves_prestart_and_suppressed_tails() {
        for tail in [VectorTailPolicy::Undisturbed, VectorTailPolicy::Agnostic] {
            let mut kind = execution();
            kind.kernel = Kernel::InsertScalar;
            lanes(&mut kind).inactive = None;
            lanes(&mut kind).tail = tail;
            assert!(kind.is_valid());
            assert_eq!(kind.input_count(), 4);
            assert_eq!(kind.output_count(), 1);
            for length in [0, 1, 2, 8, u64::MAX] {
                for start in [0, 1, 2, 8, u64::MAX] {
                    for index in 0..16 {
                        let expected = if start >= length || index < start {
                            VectorLaneAction::Preserve
                        } else if index == 0 {
                            VectorLaneAction::Execute
                        } else if tail == VectorTailPolicy::Agnostic {
                            VectorLaneAction::Agnostic
                        } else {
                            VectorLaneAction::Preserve
                        };
                        assert_eq!(kind.lane_action(index, start, length, true, 0, 0), expected);
                    }
                }
            }
            lanes(&mut kind).inactive = Some(VectorInactivePolicy::Zero);
            assert!(!kind.is_valid());
        }
    }

    #[test]
    fn widening_multiply_accumulates_preserve_both_factor_domains() {
        for signed_left in [false, true] {
            for signed_right in [false, true] {
                for bits in [8, 16, 24, 32, 64, 128, 256] {
                    let mut kind = execution();
                    kind.kernel = Kernel::WidenMultiplyAdd {
                        signed_left,
                        signed_right,
                    };
                    kind.element = VectorElement::integer(bits, Some(signed_left));
                    let valid = matches!(bits, 16 | 32 | 64 | 128);
                    assert_eq!(kind.is_valid(), valid);
                    assert_eq!(kind.input_count(), 7);
                    if valid {
                        kind.element.signed = Some(!signed_left);
                        assert!(!kind.is_valid());
                        kind.element.signed = Some(signed_left);
                        kind.element.kind = VectorElementKind::Float;
                        assert!(!kind.is_valid());
                        kind.element.kind = VectorElementKind::Integer;
                        kind.rounding = VectorRounding::FixedEnvironment;
                        assert!(!kind.is_valid());
                        kind.rounding = VectorRounding::None;
                        kind.status = VectorStatus::Saturation;
                        assert!(!kind.is_valid());
                    }
                }
            }
        }
    }

    #[test]
    fn integer_multiply_accumulates_reject_floating_rounding_and_status() {
        for subtract_product in [false, true] {
            let mut kind = execution();
            kind.kernel = Kernel::IntegerMultiplyAdd { subtract_product };
            assert!(kind.is_valid());
            assert_eq!(kind.input_count(), 7);
            assert_eq!(kind.output_count(), 1);
            kind.element.kind = VectorElementKind::Float;
            assert!(!kind.is_valid());
            kind.element.kind = VectorElementKind::Integer;
            for rounding in [
                VectorRounding::FixedEnvironment,
                VectorRounding::FloatingEnvironment,
                VectorRounding::TowardZero,
            ] {
                kind.rounding = rounding;
                assert!(!kind.is_valid());
            }
            kind.rounding = VectorRounding::None;
            kind.status = VectorStatus::Saturation;
            assert!(!kind.is_valid());
        }
    }

    #[test]
    fn fixed_point_execution_requires_explicit_rounding_and_matching_status() {
        for operation in [
            VectorFixedKind::AverageAdd,
            VectorFixedKind::AverageSub,
            VectorFixedKind::FractionalMultiply,
            VectorFixedKind::Shift,
            VectorFixedKind::Clip,
        ] {
            for bits in [0, 7, 8, 16, 24, 32, 64, 128] {
                for signed in [None, Some(false), Some(true)] {
                    let mut kind = execution();
                    kind.kernel = Kernel::Fixed { kind: operation };
                    lanes(&mut kind).inactive = None;
                    kind.element = VectorElement::integer(bits, signed);
                    kind.rounding = VectorRounding::FixedEnvironment;
                    kind.status = if operation.saturates() {
                        VectorStatus::Saturation
                    } else {
                        VectorStatus::None
                    };
                    let valid = matches!(bits, 8 | 16 | 32 | 64)
                        && signed.is_some()
                        && (operation != VectorFixedKind::FractionalMultiply
                            || signed == Some(true));
                    assert_eq!(kind.is_valid(), valid, "{operation:?}, {bits}, {signed:?}");
                    assert_eq!(kind.control_input_count(), 3);
                    assert_eq!(kind.input_count(), 6);
                    assert_eq!(kind.output_count(), 1 + usize::from(operation.saturates()));
                    if valid {
                        for rounding in [
                            VectorRounding::None,
                            VectorRounding::FloatingEnvironment,
                            VectorRounding::ToOdd,
                        ] {
                            kind.rounding = rounding;
                            assert!(!kind.is_valid());
                        }
                        kind.rounding = VectorRounding::FixedEnvironment;
                        kind.status = VectorStatus::FloatingExceptions;
                        assert!(!kind.is_valid());
                        kind.status = if operation.saturates() {
                            VectorStatus::Saturation
                        } else {
                            VectorStatus::None
                        };
                        kind.element.kind = VectorElementKind::Float;
                        assert!(!kind.is_valid());
                    }
                }
            }
        }
    }

    #[test]
    fn saturation_status_has_integer_domain_and_only_executed_lane_contributions() {
        for operation in [VectorBinaryKind::SatAdd, VectorBinaryKind::SatSub] {
            for signed in [false, true] {
                let mut kind = execution();
                kind.kernel = Kernel::Binary(operation);
                kind.element = VectorElement::integer(32, Some(signed));
                kind.status = VectorStatus::Saturation;
                assert!(kind.is_valid());
                assert_eq!(kind.output_count(), 2);
                for tail in [VectorTailPolicy::Undisturbed, VectorTailPolicy::Agnostic] {
                    for inactive in [
                        VectorInactivePolicy::Undisturbed,
                        VectorInactivePolicy::Agnostic,
                    ] {
                        lanes(&mut kind).tail = tail;
                        lanes(&mut kind).inactive = Some(inactive);
                        for start in [0, 1, 4, 8] {
                            for length in [0, 1, 4, 8] {
                                for index in 0..16 {
                                    for predicate in [false, true] {
                                        let contributes = kind
                                            .lane_action(index, start, length, predicate, 0, 0)
                                            == VectorLaneAction::Execute;
                                        assert_eq!(
                                            contributes,
                                            start < length
                                                && index >= start
                                                && index < length
                                                && predicate
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                kind.element.kind = VectorElementKind::Float;
                assert!(!kind.is_valid());
                kind.element.kind = VectorElementKind::Integer;
                for rounding in [
                    VectorRounding::FixedEnvironment,
                    VectorRounding::FloatingEnvironment,
                    VectorRounding::ToOdd,
                    VectorRounding::TowardZero,
                ] {
                    kind.rounding = rounding;
                    assert!(!kind.is_valid());
                }
                kind.rounding = VectorRounding::None;
                kind.status = VectorStatus::FloatingExceptions;
                assert!(!kind.is_valid());
            }
        }
    }

    #[test]
    fn integer_extensions_refuse_floating_rounding_and_status_domains() {
        for signed in [false, true] {
            let mut kind = execution();
            kind.kernel = Kernel::Extend { signed };
            assert!(kind.is_valid());
            assert_eq!(kind.input_count(), 5);
            kind.element.kind = VectorElementKind::Float;
            assert!(!kind.is_valid());
            kind.element.kind = VectorElementKind::Integer;
            kind.rounding = VectorRounding::FloatingEnvironment;
            assert!(!kind.is_valid());
            kind.rounding = VectorRounding::None;
            kind.status = VectorStatus::Saturation;
            assert!(!kind.is_valid());
        }
    }

    #[test]
    fn widening_integer_domains_require_supported_widths_operations_and_signs() {
        for bits in [0, 8, 16, 24, 32, 64, 128, 256] {
            for operation in [
                VectorBinaryKind::Add,
                VectorBinaryKind::Sub,
                VectorBinaryKind::Mul,
                VectorBinaryKind::Div,
                VectorBinaryKind::Shl,
            ] {
                for wide_left in [false, true] {
                    for signed_left in [false, true] {
                        for signed_right in [false, true] {
                            let mut kind = execution();
                            kind.element.bits = bits;
                            kind.element.signed = Some(signed_left);
                            kind.kernel = Kernel::WidenBinary {
                                kind: operation,
                                wide_left,
                                signed_left,
                                signed_right,
                            };
                            let width_valid = matches!(bits, 16 | 32 | 64 | 128);
                            let operation_valid = matches!(
                                operation,
                                VectorBinaryKind::Add
                                    | VectorBinaryKind::Sub
                                    | VectorBinaryKind::Mul
                                    | VectorBinaryKind::Shl
                            ) && (!wide_left
                                || !matches!(
                                    operation,
                                    VectorBinaryKind::Mul | VectorBinaryKind::Shl
                                ));
                            let signs_valid = if operation == VectorBinaryKind::Shl {
                                !signed_left && !signed_right
                            } else {
                                signed_left == signed_right
                                    || operation == VectorBinaryKind::Mul
                                        && signed_left
                                        && !signed_right
                            };
                            assert_eq!(
                                kind.is_valid(),
                                width_valid && operation_valid && signs_valid
                            );
                            kind.rounding = VectorRounding::FixedEnvironment;
                            assert!(!kind.is_valid());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn narrowing_integer_shift_domains_keep_result_width_and_source_sign_separate() {
        for bits in [0, 1, 7, 8, 16, 24, 32, 64, 128] {
            for unsigned in [false, true] {
                for signed in [None, Some(false), Some(true)] {
                    let mut kind = execution();
                    kind.element.bits = bits;
                    kind.element.signed = signed;
                    kind.kernel = Kernel::NarrowShift { unsigned };
                    assert_eq!(kind.input_count(), 6);
                    assert_eq!(
                        kind.is_valid(),
                        matches!(bits, 8 | 16 | 32 | 64)
                            && signed.is_none_or(|signed| signed != unsigned)
                    );
                    kind.rounding = VectorRounding::FixedEnvironment;
                    assert!(!kind.is_valid());
                    kind.rounding = VectorRounding::None;
                    kind.status = VectorStatus::Saturation;
                    assert!(!kind.is_valid());
                }
            }
        }
    }
    /// The carry-less product of `left` and `right`, each `bits` wide, whole:
    /// the exclusive-or of `left` shifted by each bit `right` holds.
    fn polynomial_product(left: u128, right: u128, bits: u32) -> u128 {
        (0..bits)
            .filter(|bit| right >> bit & 1 != 0)
            .fold(0, |product, bit| product ^ left << bit)
    }

    /// Lane by lane, every part of the carry-less product is the slice of the
    /// whole one it names: over every pair of bytes, over every pair of
    /// single bits at each wider element width, and in the double-width lanes
    /// of the whole product. Lanes of another count compute nothing.
    #[test]
    fn carryless_lanes_are_their_part_of_the_polynomial_product() {
        let parts = |product: u128, bits: u32| {
            let mask = (1u128 << bits) - 1;
            [
                (CarrylessPart::Low, product & mask),
                (CarrylessPart::High, product >> bits & mask),
                (CarrylessPart::Reversed, product >> (bits - 1) & mask),
                (CarrylessPart::Full, product),
            ]
        };
        let lanes = |bits: u32, values: &[u128]| {
            VectorConst::from_lanes(bits, values.iter().copied()).unwrap()
        };
        let mut pairs: Vec<(u32, u128, u128)> = (0..=255u128)
            .flat_map(|left| (0..=255u128).map(move |right| (8, left, right)))
            .collect();
        for bits in [16u32, 32, 64] {
            for left in 0..bits {
                for right in 0..bits {
                    pairs.push((bits, 1 << left, 1 << right));
                }
            }
        }
        for (bits, left, right) in pairs {
            for (part, expected) in parts(polynomial_product(left, right, bits), bits) {
                let width = if part == CarrylessPart::Full {
                    2 * bits
                } else {
                    bits
                };
                assert_eq!(
                    Kernel::CarrylessMultiply { part }
                        .evaluate_lanes(&[lanes(bits, &[left, 0]), lanes(bits, &[right, 1])], bits),
                    Some(lanes(width, &[expected, 0])),
                    "{part:?} bits={bits} {left:#x} {right:#x}"
                );
            }
        }
        // At 128 bits the product of two single bits is the one bit at the
        // sum of their positions, below 256; the whole product, 256 bits,
        // folds nothing.
        for left in 0..128u32 {
            for right in 0..128u32 {
                let position = left + right;
                let at = |shift: u32| {
                    position
                        .checked_sub(shift)
                        .filter(|bit| *bit < 128)
                        .map_or(0, |bit| 1u128 << bit)
                };
                for (part, expected) in [
                    (CarrylessPart::Low, at(0)),
                    (CarrylessPart::High, at(128)),
                    (CarrylessPart::Reversed, at(127)),
                ] {
                    assert_eq!(
                        Kernel::CarrylessMultiply { part }.evaluate_lanes(
                            &[lanes(128, &[1 << left]), lanes(128, &[1 << right])],
                            128
                        ),
                        Some(lanes(128, &[expected])),
                        "{part:?} 2^{left} * 2^{right}"
                    );
                }
            }
        }
        let full = Kernel::CarrylessMultiply {
            part: CarrylessPart::Full,
        };
        assert_eq!(
            full.evaluate_lanes(&[lanes(128, &[3]), lanes(128, &[3])], 128),
            None
        );
        let low = Kernel::CarrylessMultiply {
            part: CarrylessPart::Low,
        };
        assert_eq!(
            low.evaluate_lanes(&[lanes(16, &[1, 2]), lanes(16, &[3])], 16),
            None
        );
        assert_eq!(
            Kernel::Move.evaluate_lanes(&[lanes(16, &[1]), lanes(16, &[3])], 16),
            None
        );
    }

    /// The bytes the register oracles read: `i * 37 + 11` and `i * 91 + 200`,
    /// wrapped.
    fn oracle_bytes() -> (VectorConst, VectorConst) {
        let bytes = |scale: u8, offset: u8| {
            VectorConst::from_lanes(
                8,
                (0..32u8).map(|i| u128::from(i.wrapping_mul(scale).wrapping_add(offset))),
            )
            .unwrap()
        };
        (bytes(37, 11), bytes(91, 200))
    }

    /// `phminposuw` and `mpsadbw` compute what the hardware does. The
    /// expected words are those `_mm_minpos_epu16`, `_mm_mpsadbw_epu8` (every
    /// control) and `_mm256_mpsadbw_epu8` give on an x86-64 with SSE4.1 and
    /// AVX2 (a Broadwell), of the bytes [`oracle_bytes`] states; a tie of the
    /// least word is resolved to its first index.
    #[test]
    fn the_sse41_register_kernels_compute_what_the_hardware_does() {
        let (a, b) = oracle_bytes();
        let low = |vector: &VectorConst| vector.slice_bits(0, 8, 16).unwrap();
        let words = |values: &[u128]| VectorConst::from_lanes(16, values.iter().copied()).unwrap();
        assert_eq!(
            Kernel::HorizontalMinPos.evaluate_registers(&[low(&a)], 16),
            Some(vec![words(&[0xee9, 3, 0, 0, 0, 0, 0, 0])])
        );
        let tie = words(&[9, 3, 7, 3, 3, 8, 65535, 4]);
        assert_eq!(
            Kernel::HorizontalMinPos.evaluate_registers(&[tie], 16),
            Some(vec![words(&[3, 1, 0, 0, 0, 0, 0, 0])])
        );
        let sse: [[u128; 8]; 8] = [
            [0x152, 0x108, 0x100, 0x120, 0x200, 0x1e0, 0x102, 0x14c],
            [0x152, 0x108, 0x100, 0x120, 0xd8, 0x1d8, 0x200, 0x14c],
            [0x198, 0x14c, 0x128, 0x128, 0x16e, 0xfc, 0x1b2, 0x190],
            [0x102, 0x100, 0x126, 0x174, 0x126, 0x204, 0x200, 0x100],
            [0x200, 0x1e0, 0x102, 0x14c, 0x102, 0x100, 0x126, 0x200],
            [0xd8, 0x1d8, 0x200, 0x14c, 0x102, 0x100, 0x126, 0xdc],
            [0x16e, 0xfc, 0x1b2, 0x190, 0x146, 0x128, 0x128, 0x16c],
            [0x126, 0x204, 0x200, 0x100, 0x100, 0x12c, 0x180, 0x12c],
        ];
        for (offsets, expected) in (0u8..).zip(sse) {
            assert_eq!(
                Kernel::MultiSad { offsets }.evaluate_registers(&[low(&a), low(&b)], 8),
                Some(vec![words(&expected)]),
                "mpsadbw {offsets}"
            );
        }
        let avx2: [(u8, [u128; 16]); 3] = [
            (
                0x00,
                [
                    0x152, 0x108, 0x100, 0x120, 0x200, 0x1e0, 0x102, 0x14c, 0x128, 0x138, 0x16c,
                    0xd8, 0x16c, 0x134, 0x128, 0x128,
                ],
            ),
            (
                0x2d,
                [
                    0xd8, 0x1d8, 0x200, 0x14c, 0x102, 0x100, 0x126, 0xdc, 0x12e, 0x184, 0x12e,
                    0x100, 0x100, 0x200, 0x206, 0x128,
                ],
            ),
            (
                0x3f,
                [
                    0x126, 0x204, 0x200, 0x100, 0x100, 0x12c, 0x180, 0x12c, 0xde, 0x128, 0x100,
                    0x100, 0x14a, 0x200, 0x1d8, 0xd8,
                ],
            ),
        ];
        for (offsets, expected) in avx2 {
            assert_eq!(
                Kernel::MultiSad { offsets }.evaluate_registers(&[a.clone(), b.clone()], 8),
                Some(vec![words(&expected)]),
                "vmpsadbw {offsets:#x}"
            );
        }
        assert_eq!(
            Kernel::MultiSad { offsets: 0 }.evaluate_registers(&[low(&a), b], 8),
            None
        );
    }

    /// **`vdbpsadbw` sums each quadword's sliding differences from the
    /// shuffled second source.** Worked from the SDM's definition: against a
    /// first source of zeros and a second of the bytes `0..32`, the identity
    /// selector `0xe4` gives each quadword `q` the words of bytes `8q..8q+4`
    /// sliding by one, `6, 10, 14, 18` then `38, 42, 46, 50`, and the upper
    /// 128-bit lane shuffles its own doublewords (`70, 74, 78, 82, 102, 106,
    /// 110, 114`); the reversing selector `0x1b` puts the lane's doublewords
    /// in the order `3, 2, 1, 0`, so the first quadword slides over bytes
    /// `12 13 14 15 8 9 10`. Against a first source of the bytes `0..16` and a
    /// second of zeros, words 0 and 1 of a quadword sum its low four bytes
    /// and words 2 and 3 its high four. Sources of unequal width decline.
    #[test]
    fn a_double_block_sad_sums_sliding_quadwords_of_the_shuffled_source() {
        let bytes = |values: &[u128]| VectorConst::from_lanes(8, values.iter().copied()).unwrap();
        let words = |values: &[u128]| VectorConst::from_lanes(16, values.iter().copied()).unwrap();
        let ascending: Vec<u128> = (0..32).collect();
        let zeros = vec![0u128; 32];
        let sad = |selector: u8, first: &[u128], second: &[u128]| {
            Kernel::DoubleBlockSad { selector }
                .evaluate_registers(&[bytes(first), bytes(second)], 8)
        };
        assert_eq!(
            sad(0xe4, &zeros[..16], &ascending[..16]),
            Some(vec![words(&[6, 10, 14, 18, 38, 42, 46, 50])])
        );
        assert_eq!(
            sad(0xe4, &zeros, &ascending),
            Some(vec![words(&[
                6, 10, 14, 18, 38, 42, 46, 50, 70, 74, 78, 82, 102, 106, 110, 114
            ])])
        );
        assert_eq!(
            sad(0x1b, &zeros[..16], &ascending[..16]),
            Some(vec![words(&[54, 50, 46, 42, 22, 18, 14, 10])])
        );
        assert_eq!(
            sad(0xe4, &ascending[..16], &zeros[..16]),
            Some(vec![words(&[6, 6, 22, 22, 38, 38, 54, 54])])
        );
        assert_eq!(sad(0xe4, &zeros[..16], &ascending), None);
    }

    /// Every SSE4.2 string compare computes what the hardware does: for each
    /// of the 128 controls, four pairs of strings (vowels in a sentence,
    /// ranges of letters and digits, bytes with their sign bits set, and a
    /// substring), the implicit-length forms and the explicit-length ones of
    /// three length pairs (one negative, one past the register), the index,
    /// the mask and the four flags. `testdata/pcmpstr_broadwell.txt` holds
    /// what `_mm_cmpistri`, `_mm_cmpistrm`, `_mm_cmpestri`, `_mm_cmpestrm` and
    /// their flag intrinsics give on a Broadwell; a row is the form, the
    /// control, the pair, the two lengths, the index, the mask's two
    /// quadwords and the carry, zero, sign and overflow flags.
    #[test]
    fn every_string_compare_computes_what_the_hardware_does() {
        let pairs: [(&[u8; 16], &[u8; 16]); 4] = [
            (b"aeiou\0zzzzzzzzzz", b"Hello, world!\0xy"),
            (b"AZaz09\0\0\0\0\0\0\0\0\0\0", b"Hi 2u!\0\0\0\0\0\0\0\0\0\0"),
            (
                &[
                    0x80, 0x7f, 0xff, 0x01, 0x41, 0x42, 0xc3, 0x10, 0x90, 0x22, 0x7e, 0x81, 0x05,
                    0xfe, 0x33, 0x44,
                ],
                &[
                    0x41, 0x81, 0x7f, 0x00, 0x90, 0xff, 0x22, 0x10, 0x80, 0x01, 0xc3, 0x42, 0x44,
                    0x33, 0xfe, 0x05,
                ],
            ),
            (b"lo, w\0\0\0\0\0\0\0\0\0\0\0", b"Hello, world!\0\0\0"),
        ];
        let string = |bytes: &[u8; 16]| {
            VectorConst::from_lanes(8, bytes.iter().map(|byte| u128::from(*byte))).unwrap()
        };
        let length =
            |value: i32| VectorConst::from_lanes(32, [u128::from(value.cast_unsigned())]).unwrap();
        let rows = include_str!("testdata/pcmpstr_broadwell.txt");
        let mut checked = 0;
        for row in rows.lines() {
            let fields: Vec<&str> = row.split_whitespace().collect();
            let [form, control, pair, first, second, index, low, high, flags] = fields[..] else {
                panic!("a malformed row: {row}");
            };
            let number = |field: &str| field.parse::<i64>().unwrap();
            let hex =
                |field: &str| u128::from_str_radix(field.trim_start_matches("0x"), 16).unwrap();
            let control = u8::try_from(number(control)).unwrap();
            let (a, b) = pairs[usize::try_from(number(pair)).unwrap()];
            let explicit_length = form == "e";
            let mut inputs = vec![string(a), string(b)];
            if explicit_length {
                for value in [first, second] {
                    inputs.push(length(i32::try_from(number(value)).unwrap()));
                }
            }
            let mut expected_flags = 0u16;
            for (digit, flag) in flags.chars().zip([
                FlagsMask::CARRY,
                FlagsMask::ZERO,
                FlagsMask::SIGN,
                FlagsMask::OVERFLOW,
            ]) {
                if digit == '1' {
                    expected_flags |= flag.bits();
                }
            }
            let flags = VectorConst::from_lanes(32, [u128::from(expected_flags)]).unwrap();
            for (result_index, expected) in [
                (
                    true,
                    VectorConst::from_lanes(32, [u128::try_from(number(index)).unwrap()]).unwrap(),
                ),
                (
                    false,
                    VectorConst::from_lanes(64, [hex(low), hex(high)]).unwrap(),
                ),
            ] {
                assert_eq!(
                    Kernel::StringCompare {
                        control,
                        explicit_length,
                        result_index,
                    }
                    .evaluate_registers(&inputs, 8),
                    Some(vec![expected, flags.clone()]),
                    "{row} (index {result_index})"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 4096);
    }

    /// Every decimal adjust computes what the hardware does, its undefined
    /// flags included. `testdata/bcd_broadwell.txt` holds what a Broadwell's
    /// `daa`, `das`, `aaa`, `aas`, `aam` and `aad` give in 32-bit mode: a row
    /// is the adjust (0 to 5 in that order), the radix, `AX` and `EFLAGS`
    /// before, and `AX` and `EFLAGS` after, in hex; every `AL` with each carry
    /// and adjust flag for the first four (`AH` 0 and `0xff` for the ASCII
    /// ones), every `AL` in radix 10, 16 and 7 for `aam`, and with `AH` 0, 9
    /// and `0xff` for `aad`. The other incoming flags change nothing there.
    #[test]
    fn every_decimal_adjust_computes_what_the_hardware_does() {
        let kinds = [
            BcdAdjustKind::DecimalAddAdjust,
            BcdAdjustKind::DecimalSubAdjust,
            BcdAdjustKind::AsciiAddAdjust,
            BcdAdjustKind::AsciiSubAdjust,
            BcdAdjustKind::AsciiMulAdjust,
            BcdAdjustKind::AsciiDivAdjust,
        ];
        // EFLAGS' bits of the six status flags.
        let flags = |eflags: u16| {
            [
                (0x001, FlagsMask::CARRY),
                (0x004, FlagsMask::PARITY),
                (0x010, FlagsMask::ADJUST),
                (0x040, FlagsMask::ZERO),
                (0x080, FlagsMask::SIGN),
                (0x800, FlagsMask::OVERFLOW),
            ]
            .into_iter()
            .filter(|(bit, _)| eflags & bit != 0)
            .fold(0u16, |mask, (_, flag)| mask | flag.bits())
        };
        let rows = include_str!("testdata/bcd_broadwell.txt");
        let mut checked = 0;
        for row in rows.lines() {
            let fields: Vec<u16> = row
                .split_whitespace()
                .map(|field| u16::from_str_radix(field, 16).unwrap())
                .collect();
            let [kind, base, before, eflags, after, eflags_after] = fields[..] else {
                panic!("a malformed row: {row}");
            };
            let kind = kinds[usize::from(kind)];
            let decimal = matches!(
                kind,
                BcdAdjustKind::DecimalAddAdjust | BcdAdjustKind::DecimalSubAdjust
            );
            let (bits, accumulator, expected) = if decimal {
                (8, before & 0xff, after & 0xff)
            } else {
                (16, before, after)
            };
            let mut inputs =
                vec![VectorConst::from_lanes(bits, [u128::from(accumulator)]).unwrap()];
            if !matches!(
                kind,
                BcdAdjustKind::AsciiMulAdjust | BcdAdjustKind::AsciiDivAdjust
            ) {
                inputs.push(VectorConst::from_lanes(32, [u128::from(flags(eflags))]).unwrap());
            }
            let base = u8::try_from(base).unwrap();
            assert_eq!(
                Kernel::DecimalAdjust { kind, base }.evaluate_registers(&inputs, bits),
                Some(vec![
                    VectorConst::from_lanes(bits, [u128::from(expected)]).unwrap(),
                    VectorConst::from_lanes(32, [u128::from(flags(eflags_after))]).unwrap(),
                ]),
                "{row}"
            );
            checked += 1;
        }
        assert_eq!(checked, 9216);
        // A zero radix divides by zero.
        let al = VectorConst::from_lanes(16, [0x0042]).unwrap();
        assert_eq!(
            Kernel::DecimalAdjust {
                kind: BcdAdjustKind::AsciiMulAdjust,
                base: 0
            }
            .evaluate_registers(&[al], 16),
            None
        );
    }

    /// An SSE4a bit field is AMD's: `extrq` of `0x0123_4567_89ab_cdef` by
    /// length 8 at index 4 is `0xde`, `insertq` of `0x12` there into all
    /// ones `0xffff_ffff_ffff_f12f`, the register forms read the same field
    /// from the control's bits 5 to 0 and 13 to 8, a length of zero is the
    /// whole quadword, and a field past bit 63 folds nothing.
    #[test]
    fn an_sse4a_bit_field_is_the_one_its_length_and_index_name() {
        let value = 0x0123_4567_89ab_cdefu64;
        let at = |insert: bool, position: BitfieldPosition, inputs: &[u64]| {
            let inputs: Vec<(u64, u32)> = inputs.iter().map(|input| (*input, 64)).collect();
            Kernel::Bitfield { insert, position }.evaluate_integers(&inputs, 64)
        };
        let field = |insert: bool, index: u8, length: u8, inputs: &[u64]| {
            at(
                insert,
                BitfieldPosition::Immediate { index, length },
                inputs,
            )
        };
        let register =
            |insert: bool, inputs: &[u64]| at(insert, BitfieldPosition::Register, inputs);
        assert_eq!(field(false, 4, 8, &[value]), Some(0xde));
        assert_eq!(field(false, 0, 0, &[value]), Some(value));
        assert_eq!(field(false, 60, 4, &[value]), Some(0x0));
        assert_eq!(field(false, 61, 4, &[value]), None);
        assert_eq!(
            field(true, 4, 8, &[u64::MAX, 0x12]),
            Some(0xffff_ffff_ffff_f12f)
        );
        assert_eq!(field(true, 0, 0, &[u64::MAX, value]), Some(value));
        // The control states length 8 at index 4.
        let control = 4 << 8 | 8;
        assert_eq!(register(false, &[value, control]), Some(0xde));
        assert_eq!(
            register(true, &[u64::MAX, 0x12, control]),
            Some(0xffff_ffff_ffff_f12f)
        );
        // A position stated twice, or not at all, is no field.
        assert_eq!(field(false, 4, 8, &[value, control]), None);
        assert_eq!(register(false, &[value]), None);
        // Only the low six bits of each count.
        assert_eq!(field(false, 0x44, 0x48, &[value]), Some(0xde));
    }

    /// A ternary logic function is its truth table, indexed by the three
    /// inputs' bits as `a << 2 | b << 1 | c`: `0x96` is their exclusive-or,
    /// `0xe8` their majority, `0xca` the select of `b` where `a` is set and
    /// `c` elsewhere, and `0xf0`, `0xcc` and `0xaa` the inputs themselves.
    #[test]
    fn a_ternary_logic_function_is_its_truth_table() {
        let quad = |words: [u128; 4]| VectorConst::from_lanes(32, words).unwrap();
        let (a, b, c) = (
            quad([0xdead_beef, 0x0123_4567, 0xffff_0000, 0x5555_aaaa]),
            quad([0x8765_4321, 0xfedc_ba98, 0x0f0f_0f0f, 0x3333_cccc]),
            quad([0x1357_9bdf, 0x2468_ace0, 0x00ff_00ff, 0x0f0f_f0f0]),
        );
        let lanes = |f: fn(u128, u128, u128) -> u128| {
            quad(std::array::from_fn(|i| {
                f(a.lane(i as u32), b.lane(i as u32), c.lane(i as u32)) & 0xffff_ffff
            }))
        };
        for (table, expected) in [
            (0x96, lanes(|a, b, c| a ^ b ^ c)),
            (0xe8, lanes(|a, b, c| a & b | a & c | b & c)),
            (0xca, lanes(|a, b, c| a & b | !a & c)),
            (0xf0, a.clone()),
            (0xcc, b.clone()),
            (0xaa, c.clone()),
            (0x00, quad([0; 4])),
            (0xff, quad([0xffff_ffff; 4])),
        ] {
            assert_eq!(
                Kernel::TernaryLogic { table }
                    .evaluate_registers(&[a.clone(), b.clone(), c.clone()], 32),
                Some(vec![expected]),
                "{table:#04x}"
            );
        }
    }

    /// A bit shuffle reads, for each control byte, the source bit its low six
    /// bits name in the byte's own quadword; an intersection marks the
    /// elements of each source the other holds. Both write a whole 64-bit
    /// mask register.
    #[test]
    fn a_bit_shuffle_and_an_intersection_write_their_masks() {
        let source = VectorConst::from_lanes(64, [0x8000_0000_0000_0001, 0x2]).unwrap();
        let control = VectorConst::from_lanes(
            8,
            [0, 63, 1, 0x40, 0x7f, 2, 0, 0, 1, 0x41, 0, 63, 0, 0, 0, 0x81],
        )
        .unwrap();
        let mask = |bits: u128| VectorConst::from_lanes(64, [bits]).unwrap();
        assert_eq!(
            Kernel::ShuffleBits.evaluate_registers(&[source, control], 8),
            Some(vec![mask(0x83db)])
        );
        let words = |values: [u128; 4]| VectorConst::from_lanes(32, values).unwrap();
        assert_eq!(
            Kernel::Intersect.evaluate_registers(&[words([1, 2, 3, 4]), words([4, 4, 9, 1])], 32),
            Some(vec![mask(0b1001), mask(0b1011)])
        );
        assert_eq!(
            Kernel::Intersect.evaluate_registers(&[words([1, 2, 3, 4]), words([4, 4, 9, 1])], 16),
            None
        );
    }

    /// A whole product of two 32-bit integers folds to its 64 bits, and only
    /// at twice the factors' width.
    #[test]
    fn a_whole_carryless_product_of_integers_is_twice_their_width() {
        let full = Kernel::CarrylessMultiply {
            part: CarrylessPart::Full,
        };
        let (left, right) = (0xdead_beef_u64, 0x8765_4321_u64);
        let expected = u64::try_from(polynomial_product(left.into(), right.into(), 32)).unwrap();
        assert_eq!(
            full.evaluate_integers(&[(left, 32), (right, 32)], 64),
            Some(expected)
        );
        assert_eq!(full.evaluate_integers(&[(left, 32), (right, 32)], 32), None);
        assert_eq!(full.evaluate_integers(&[(left, 64), (right, 64)], 64), None);
    }

    #[test]
    fn carryless_execution_requires_unrounded_unsigned_bit_patterns() {
        let mut kind = Computation {
            kernel: Kernel::CarrylessMultiply {
                part: CarrylessPart::Low,
            },
            element: VectorElement::integer(64, None),
            predicate: VectorPredicateLayout::Packed,
            rounding: VectorRounding::None,
            status: VectorStatus::None,
            canonical_nan: false,
            lanes: Some(LaneControl {
                tail: VectorTailPolicy::Undisturbed,
                inactive: None,
                mask_tail: false,
                no_passthrough: false,
            }),
        };
        for bits in [8, 16, 32, 64, 128] {
            kind.element.bits = bits;
            assert!(kind.is_valid());
        }
        for bits in [0, 1, 7, 24, 256] {
            kind.element.bits = bits;
            assert!(!kind.is_valid());
        }
        kind.element = VectorElement::float(64);
        assert!(!kind.is_valid());
        kind.element = VectorElement::integer(64, Some(false));
        assert!(!kind.is_valid());
        kind.element = VectorElement::integer(64, None);
        kind.rounding = VectorRounding::FloatingEnvironment;
        assert!(!kind.is_valid());
        kind.rounding = VectorRounding::None;
        kind.status = VectorStatus::Saturation;
        assert!(!kind.is_valid());
    }
    #[test]
    fn cryptographic_groups_require_complete_width_and_exact_input_roles() {
        for (primitive, bits, count) in [
            (VectorCryptoKind::AesEncrypt, 128, 2),
            (VectorCryptoKind::AesEncryptLast, 128, 2),
            (VectorCryptoKind::AesDecrypt, 128, 2),
            (VectorCryptoKind::AesDecryptKeyMix, 128, 2),
            (VectorCryptoKind::AesDecryptLast, 128, 2),
            (VectorCryptoKind::Sm4Rounds, 128, 2),
            (VectorCryptoKind::AesExpand128, 128, 2),
            (VectorCryptoKind::AesExpand256, 128, 3),
            (VectorCryptoKind::GhashMultiply, 128, 2),
            (VectorCryptoKind::GhashMultiplyAdd, 128, 3),
            (VectorCryptoKind::Sha2Schedule4, 128, 3),
            (VectorCryptoKind::Sha2Schedule4, 256, 3),
            (VectorCryptoKind::Sha2Compress2Low, 128, 3),
            (VectorCryptoKind::Sha2Compress2Low, 256, 3),
            (VectorCryptoKind::Sha2Compress2High, 128, 3),
            (VectorCryptoKind::Sha2Compress2High, 256, 3),
            (VectorCryptoKind::Sm3Expand8, 256, 2),
            (VectorCryptoKind::Sm3Compress2, 256, 3),
            (VectorCryptoKind::Sm4Expand4, 128, 2),
        ] {
            let mut kind = Computation {
                kernel: Kernel::Crypto(primitive),
                element: VectorElement::integer(bits, None),
                predicate: VectorPredicateLayout::Packed,
                rounding: VectorRounding::None,
                status: VectorStatus::None,
                canonical_nan: false,
                lanes: Some(LaneControl {
                    tail: VectorTailPolicy::Undisturbed,
                    inactive: None,
                    mask_tail: false,
                    no_passthrough: false,
                }),
            };
            assert!(kind.is_valid(), "{primitive:?}");
            assert_eq!(kind.input_count(), count + 3);
            assert_eq!(kind.output_count(), 1);
            assert_eq!(primitive.group_input_count(), Some(count));
            // Over whole registers it reads as many values as groups.
            assert_eq!(primitive.whole_input_count(), count, "{primitive:?}");
            for bad in [0, 32, 64, 512] {
                kind.element.bits = bad;
                assert!(!kind.is_valid(), "{primitive:?} {bad}");
            }
            kind.element.bits = bits;
            kind.rounding = VectorRounding::FloatingEnvironment;
            assert!(!kind.is_valid());
            kind.rounding = VectorRounding::None;
            kind.status = VectorStatus::FloatingExceptions;
            assert!(!kind.is_valid());
            kind.status = VectorStatus::None;
            kind.element = VectorElement::float(bits);
            assert!(!kind.is_valid());
        }
        assert_eq!(VectorCryptoKind::AesKeygenAssist.group_input_count(), None);
    }

    /// A whole-value kernel states every operand it computes with: the
    /// pointer, its modifiers and the key's two halves of a pointer
    /// authentication, the double alone of a modular conversion, the
    /// address, mask or offsets and GCR_EL1 of a tag step, and nothing of a
    /// random draw, which writes the flags reporting its success beside its
    /// value, as a random tag writes the generator state it advances. Each
    /// computes on the integer it states: a 64-bit address, a 32-bit
    /// result, a random draw's register.
    #[test]
    fn whole_kernels_state_every_operand() {
        let element = VectorElement::integer(64, None);
        for (kernel, inputs, outputs) in [
            (Kernel::PointerAuth(PacKind::Sign), 4, 1),
            (Kernel::PointerAuth(PacKind::Authenticate), 4, 1),
            (Kernel::PointerAuth(PacKind::SignDiversified), 5, 1),
            (Kernel::PointerAuth(PacKind::AuthenticateDiversified), 5, 1),
            (Kernel::PointerAuth(PacKind::Strip), 1, 1),
            (Kernel::PointerAuth(PacKind::GenericMac), 4, 1),
            (Kernel::FloatToIntModular, 1, 1),
            (Kernel::MemoryTag(MemoryTagKind::Random), 4, 2),
            (Kernel::MemoryTag(MemoryTagKind::Add), 4, 1),
            (
                Kernel::Random {
                    from_entropy: false,
                },
                0,
                2,
            ),
            (Kernel::Random { from_entropy: true }, 0, 2),
        ] {
            let mut kind = Computation::whole(kernel, element, VectorRounding::None);
            if let Some(bits) = kernel.fixed_integer_bits() {
                kind.element = VectorElement::integer(bits, None);
                // Another width, or a float, is no such computation.
                for wrong in [
                    VectorElement::integer(bits * 2, None),
                    VectorElement::float(bits),
                ] {
                    kind.element = wrong;
                    assert!(!kind.is_valid(), "{kernel:?} {wrong:?}");
                }
                kind.element = VectorElement::integer(bits, None);
            } else {
                for bits in [16, 32] {
                    kind.element = VectorElement::integer(bits, None);
                    assert!(kind.is_valid(), "{kernel:?} {bits}");
                }
                kind.element = VectorElement::integer(8, None);
                assert!(!kind.is_valid(), "{kernel:?} 8");
                kind.element = element;
            }
            assert!(kind.is_valid(), "{kernel:?}");
            assert_eq!(
                (kind.input_count(), kind.output_count()),
                (inputs, outputs),
                "{kernel:?}"
            );
        }
    }

    /// **A predicate-as-counter encodes and expands as Arm's pseudocode does.**
    /// At a 256-bit vector (32-bit predicates): `ptrue pn8.b` encodes all 32
    /// bytes as an inverted zero, `0x8001`, which expands to every byte of
    /// every part; three words (`whilelo pn8.s, vlx2`) encode `3 << 3 | 0b100`
    /// and expand to bits 0, 4 and 8 of part 0; the last five halfwords of
    /// four registers (`whilegt pn8.h, vlx4`) encode the inverted 59 and
    /// expand to bits 22 to 30 of part 3. And every count of every element
    /// width over two or four registers expands to that many elements, the
    /// first ones, or the last of the registers where inverted.
    #[test]
    fn a_predicate_as_counter_encodes_and_expands_as_the_pseudocode_does() {
        let encode = |element_bits, elements: u64, count: u64, invert: bool| {
            Kernel::PredicateCounter { element_bits }
                .evaluate_integers(&[(count, 64), (elements, 64), (u64::from(invert), 1)], 32)
        };
        let expand = |counter, part| {
            Kernel::CounterPredicate { part }.evaluate_integers(&[(counter, 32)], 32)
        };
        assert_eq!(encode(8, 32, 32, false), Some(0x8001));
        assert_eq!(encode(32, 8, 8, false), Some(0x8004));
        assert_eq!(encode(32, 16, 3, false), Some(0x1c));
        assert_eq!(encode(16, 64, 5, true), Some(0x80ee));
        assert_eq!(encode(16, 64, 0, true), Some(0));
        assert_eq!(encode(16, 64, 65, false), None);
        assert_eq!(encode(12, 64, 1, false), None);
        for part in 0..4 {
            assert_eq!(expand(0x8001, part), Some(0xffff_ffff));
            assert_eq!(expand(0, part), Some(0));
        }
        assert_eq!(expand(0x1c, 0), Some(0x111));
        assert_eq!(expand(0x1c, 1), Some(0));
        assert_eq!(expand(0x80ee, 3), Some(0x5540_0000));
        assert_eq!(expand(0x80ee, 2), Some(0));
        assert_eq!(expand(0x8001, 4), None);
        for element_bits in [8u8, 16, 32, 64] {
            let bytes = u32::from(element_bits / 8);
            for registers in [2u32, 4] {
                let elements = registers * 32 / bytes;
                for count in 0..=elements {
                    for invert in [false, true] {
                        let counter =
                            encode(element_bits, u64::from(elements), u64::from(count), invert)
                                .unwrap();
                        let active: Vec<u32> = (0..registers)
                            .flat_map(|part| {
                                let bits = expand(counter, u8::try_from(part).unwrap()).unwrap();
                                (0..32)
                                    .filter(move |bit| bits >> bit & 1 != 0)
                                    .map(move |bit| (part * 32 + bit) / bytes)
                            })
                            .collect();
                        let expected: Vec<u32> = if invert {
                            (elements - count..elements).collect()
                        } else {
                            (0..count).collect()
                        };
                        assert_eq!(
                            active, expected,
                            "{element_bits}-bit, {registers} registers, {count}, {invert}"
                        );
                    }
                }
            }
        }
    }
}
