//! Allocation and shape limits for ordinary and layout-typed byte decoding.

use std::{iter, process};

use analyssa::{
    PointerSize, Target,
    ir::{ConstValue, FloatFormat, exception::HandlerKind, lanes::MAX_VECTOR_BITS},
    target::{Endianness, TypeLayout, VectorLaneKind, VectorShape},
};

use crate::measuring::{assert_bounded, assert_within, measured};

/// A flat host declaration whose vector dimensions are supplied by each fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ByteTarget;

impl Target for ByteTarget {
    type TypeRef = ();
    type MethodRef = ();
    type FieldRef = ();
    type SigRef = ();
    type SymbolRef = ();
    type ExceptionKind = ();
    type Type = VectorShape;
    type OriginalInstruction = ();
    type LocalSignature = ();
    type VariableFacts = ();
    type InstructionFacts = ();
    type FunctionFacts = ();
    type Capability = ();

    fn ptr_bytes(&self) -> u32 {
        8
    }
    fn synthetic_instruction() {}
    fn unknown_type() -> VectorShape {
        VectorShape {
            lane_count: 0,
            lane_kind: VectorLaneKind::Integer,
            lane_bits: 0,
            total_bits: 0,
        }
    }
    fn is_integer(_: &VectorShape) -> bool {
        false
    }
    fn is_floating(_: &VectorShape) -> bool {
        false
    }
    fn is_signed(_: &VectorShape) -> bool {
        false
    }
    fn is_pointer(_: &VectorShape) -> bool {
        false
    }
    fn is_reference(_: &VectorShape) -> bool {
        false
    }
    fn is_unknown(ty: &VectorShape) -> bool {
        ty.total_bits == 0
    }
    fn bit_width(ty: &VectorShape) -> Option<u32> {
        Some(ty.total_bits)
    }
    fn vector_shape(ty: &VectorShape) -> Option<VectorShape> {
        Some(*ty)
    }
    fn instruction_mnemonic(_: &()) -> &'static str {
        "test"
    }
    fn instruction_rva(_: &()) -> u64 {
        0
    }
    fn handler_kind(_: &()) -> HandlerKind {
        HandlerKind::Catch
    }
}

/// Enumerates every integer lane width, exact floating format and native lane category.
fn lane_formats() -> impl Iterator<Item = (VectorLaneKind, u32)> {
    (1..=128)
        .map(|bits| (VectorLaneKind::Integer, bits))
        .chain(
            [
                FloatFormat::Half,
                FloatFormat::BFloat16,
                FloatFormat::Single,
                FloatFormat::Double,
                FloatFormat::X87Extended,
                FloatFormat::Quad,
            ]
            .into_iter()
            .map(|format| (VectorLaneKind::Float(format), format.bits())),
        )
        .chain(iter::once((VectorLaneKind::NativeInteger, 64)))
}

/// Oversized representations are refused without first allocating their declared lane array.
#[test]
fn oversized_vector_bytes_refuse_before_allocating() {
    for (kind, lane_bits) in lane_formats() {
        for lanes in [MAX_VECTOR_BITS + MAX_VECTOR_BITS / 4, 4 * MAX_VECTOR_BITS] {
            let shape = VectorShape::new(
                lanes,
                kind,
                lane_bits,
                lanes
                    .checked_mul(lane_bits)
                    .unwrap_or_else(|| process::abort()),
            )
            .unwrap_or_else(|| process::abort());
            assert!(shape.total_bits > MAX_VECTOR_BITS);
            let bytes = vec![
                0x5a;
                usize::try_from(shape.total_bits / 8)
                    .unwrap_or_else(|_| process::abort())
            ];
            let layout = TypeLayout::of::<ByteTarget>(&shape, PointerSize::Bit64);
            for order in [Endianness::Little, Endianness::Big] {
                for projected in [false, true] {
                    let (result, largest) = measured(|| {
                        if projected {
                            ConstValue::<ByteTarget>::from_layout_bytes(&bytes, layout, order)
                        } else {
                            ConstValue::<ByteTarget>::from_typed_bytes(&bytes, &shape, order)
                        }
                    });
                    assert!(
                        result.is_none(),
                        "{shape:?} {order:?} projected={projected}: {result:?}"
                    );
                    assert_bounded("refused vector byte decoding", largest);
                    assert_within("refused oversized vector byte decoding", largest, 0);
                }
            }
        }
    }
}

/// Exact representable byte and packed-lane boundaries retain every input bit.
#[test]
fn maximum_vector_bytes_preserve_lane_values() {
    for (kind, lane_bits) in lane_formats().filter(|(kind, bits)| {
        *kind != VectorLaneKind::NativeInteger && (*bits < 8 || *bits % 8 == 0)
    }) {
        let maximum = MAX_VECTOR_BITS
            .checked_div(lane_bits)
            .unwrap_or_else(|| process::abort());
        let lanes = (1..=maximum)
            .rev()
            .find(|count| {
                count
                    .checked_mul(lane_bits)
                    .is_some_and(|bits| bits % 8 == 0)
            })
            .unwrap_or_else(|| process::abort());
        let total_bits = lanes
            .checked_mul(lane_bits)
            .unwrap_or_else(|| process::abort());
        let shape = VectorShape::new(lanes, kind, lane_bits, total_bits)
            .unwrap_or_else(|| process::abort());
        let bytes: Vec<_> = (0..total_bits / 8)
            .map(|i| i.wrapping_mul(37).wrapping_add(11) as u8)
            .collect();
        let layout = TypeLayout::of::<ByteTarget>(&shape, PointerSize::Bit64);
        for order in [Endianness::Little, Endianness::Big] {
            for projected in [false, true] {
                let result = if projected {
                    ConstValue::<ByteTarget>::from_layout_bytes(&bytes, layout, order)
                } else {
                    ConstValue::<ByteTarget>::from_typed_bytes(&bytes, &shape, order)
                }
                .unwrap_or_else(|| process::abort());
                let vector = result
                    .as_lanes(PointerSize::Bit64)
                    .unwrap_or_else(|| process::abort());
                assert_eq!(vector.lanes(), lanes);
                assert_eq!(vector.lane_bits(), lane_bits);
                for lane in 0..lanes {
                    let expected = if lane_bits < 8 {
                        (0..lane_bits).fold(0u128, |raw, part| {
                            let bit = lane.wrapping_mul(lane_bits).wrapping_add(part);
                            let byte = bytes
                                .get((bit / 8) as usize)
                                .unwrap_or_else(|| process::abort());
                            raw | u128::from(byte.wrapping_shr(bit % 8) & 1).wrapping_shl(part)
                        })
                    } else {
                        let width = (lane_bits / 8) as usize;
                        let start = (lane as usize).wrapping_mul(width);
                        bytes
                            .get(start..start.wrapping_add(width))
                            .unwrap_or_else(|| process::abort())
                            .iter()
                            .enumerate()
                            .fold(0u128, |raw, (index, byte)| {
                                let position = match order {
                                    Endianness::Little => index,
                                    Endianness::Big => width.wrapping_sub(1).wrapping_sub(index),
                                };
                                raw | u128::from(*byte)
                                    .wrapping_shl(position.wrapping_mul(8) as u32)
                            })
                    };
                    assert_eq!(
                        vector.lane(lane),
                        expected,
                        "{shape:?} {order:?} projected={projected} lane={lane}"
                    );
                }
            }
        }
    }
}

/// Inconsistent host descriptors cannot manufacture a different vector shape from the bytes.
#[test]
fn malformed_vector_bytes_are_refused_without_allocation() {
    for (count, bits, total, kind, len) in [
        (2, 8, 8, VectorLaneKind::Integer, 2),
        (0, 8, 8, VectorLaneKind::Integer, 1),
        (1, 0, 8, VectorLaneKind::Integer, 1),
        (1, 16, 8, VectorLaneKind::Integer, 2),
        (8, 1, 16, VectorLaneKind::Integer, 2),
        (2, 16, 32, VectorLaneKind::Float(FloatFormat::Single), 4),
        (u32::MAX, 2, u32::MAX, VectorLaneKind::Integer, 4),
    ] {
        let shape = VectorShape {
            lane_count: count,
            lane_bits: bits,
            total_bits: total,
            lane_kind: kind,
        };
        let bytes = [0x5au8; 4];
        let bytes = bytes.get(..len).unwrap_or_else(|| process::abort());
        let layout = TypeLayout::of::<ByteTarget>(&shape, PointerSize::Bit64);
        for order in [Endianness::Little, Endianness::Big] {
            for projected in [false, true] {
                let (result, largest) = measured(|| {
                    if projected {
                        ConstValue::<ByteTarget>::from_layout_bytes(bytes, layout, order)
                    } else {
                        ConstValue::<ByteTarget>::from_typed_bytes(bytes, &shape, order)
                    }
                });
                assert!(
                    result.is_none(),
                    "{shape:?} {order:?} projected={projected}: {result:?}"
                );
                assert_within("malformed vector byte decoding", largest, 0);
            }
        }
    }
}
