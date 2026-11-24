//! `Target` trait - the abstraction that lets the SSA IR core be generic over
//! the source instruction set.
//!
//! The IR core (`SsaOp`, `SsaInstruction`, `SsaFunction`, `SsaBlock`,
//! `ConstValue`, `SsaExceptionHandler`, `MemoryLocation`) is `<T: Target>`-
//! generic; host-specific data is hidden behind `Target`'s associated types.
//! Each embedding crate supplies its own concrete implementation.
//!
//! # Design notes
//!
//! - **Type queries live on `Target`.** Most queries are pure functions of the
//!   type (`is_integer`, `bit_width`) and do not need a runtime instance. Only
//!   `ptr_bytes` is `&self`. Keeping these queries here keeps pass signatures
//!   to `<T: Target>`.
//!
//! - **`Target` is `Sized + 'static`.** No reason to support unsized targets;
//!   the `'static` bound makes the type usable in trait-object contexts later
//!   if a dynamic-pass-registry shows up.
//!
//! - **`ptr_bytes` is runtime, not const.** CLR is bi-arch (32 vs 64). A
//!   typical instance carries the pointer width chosen at construction.

use std::{fmt::Debug, hash::Hash};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use self::layout::{BooleanStorage, StorageLayout, TypeLayout};
use crate::{
    PointerSize,
    analysis::{
        execution::{Address, Root},
        image::ImageRead,
    },
    encoder::{ContentWriter, Emit},
    facts,
    ir::{
        content::ContentSink,
        exception::HandlerKind,
        float::{FloatFormat, FloatRounding},
        integer::IntegerType,
        value::ConstValue,
    },
    scheduling::DeobfuscationCapability,
};

mod layout;

/// Element category for a target-independent vector lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorLaneKind {
    /// Integer lane with an explicit bit width.
    Integer,
    /// Floating-point lane of the format named, which states its width.
    Float(FloatFormat),
    /// Pointer-sized native integer lane whose concrete width is target-dependent.
    NativeInteger,
}

/// Describes the lane layout of a target-independent vector value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorShape {
    /// Number of lanes in the vector.
    pub lane_count: u32,
    /// Kind of scalar stored in each lane.
    pub lane_kind: VectorLaneKind,
    /// Bit width of each lane.
    pub lane_bits: u32,
    /// Total bit width of the vector register or value.
    pub total_bits: u32,
}

impl VectorShape {
    /// Creates a vector shape after validating its lane and total widths.
    ///
    /// Rejects degenerate descriptors up front so downstream lane math can trust
    /// the invariant `lane_count * lane_bits == total_bits`: any zero field, or a
    /// product that disagrees with `total_bits`, yields `None` rather than a shape
    /// that would mis-slice a vector value.
    ///
    /// # Arguments
    ///
    /// * `lane_count` - Number of lanes in the vector.
    /// * `lane_kind` - Scalar category stored in each lane.
    /// * `lane_bits` - Bit width of a single lane.
    /// * `total_bits` - Total bit width; must equal `lane_count * lane_bits`.
    ///
    /// # Returns
    ///
    /// The validated shape, or `None` if any field is zero or the widths are
    /// inconsistent.
    #[must_use]
    pub const fn new(
        lane_count: u32,
        lane_kind: VectorLaneKind,
        lane_bits: u32,
        total_bits: u32,
    ) -> Option<Self> {
        if lane_count == 0 || lane_bits == 0 || total_bits == 0 {
            return None;
        }
        if lane_count.saturating_mul(lane_bits) != total_bits {
            return None;
        }
        if let VectorLaneKind::Float(format) = lane_kind
            && format.bits() != lane_bits
        {
            return None;
        }
        Some(Self {
            lane_count,
            lane_kind,
            lane_bits,
            total_bits,
        })
    }

    /// Returns `true` when the shape has a valid lane count and width product.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.lane_count != 0
            && self.lane_bits != 0
            && self.total_bits != 0
            && self.lane_count.saturating_mul(self.lane_bits) == self.total_bits
    }

    /// The format of the shape's lanes where they are floats.
    #[must_use]
    pub const fn lane_format(self) -> Option<FloatFormat> {
        match self.lane_kind {
            VectorLaneKind::Float(format) => Some(format),
            VectorLaneKind::Integer | VectorLaneKind::NativeInteger => None,
        }
    }
}

/// Describes the representation used for vector masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorMaskShape {
    /// Number of predicate lanes in the mask.
    pub lane_count: u32,
    /// Number of bits used by each predicate lane.
    pub lane_bits: u32,
}

impl VectorMaskShape {
    /// Creates a vector mask shape when the lane count and lane width are non-zero.
    ///
    /// # Arguments
    ///
    /// * `lane_count` - Number of predicate lanes in the mask.
    /// * `lane_bits` - Bits used by each predicate lane.
    ///
    /// # Returns
    ///
    /// The mask shape, or `None` if either field is zero.
    #[must_use]
    pub const fn new(lane_count: u32, lane_bits: u32) -> Option<Self> {
        if lane_count == 0 || lane_bits == 0 {
            return None;
        }
        Some(Self {
            lane_count,
            lane_bits,
        })
    }

    /// Returns `true` when the mask has non-zero lane count and lane width.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.lane_count != 0 && self.lane_bits != 0
    }
}

/// Scales a vector shape relative to the target's runtime vector length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorLengthMultiplier {
    /// Numerator of the vector-length multiplier.
    pub numerator: u32,
    /// Denominator of the vector-length multiplier.
    pub denominator: u32,
}

impl VectorLengthMultiplier {
    /// Creates a vector-length multiplier when both parts are non-zero.
    ///
    /// The multiplier scales a scalable type's minimum lane count against the
    /// target's runtime vector length; a zero numerator or denominator would
    /// describe an empty or undefined scaling, so it is rejected.
    ///
    /// # Arguments
    ///
    /// * `numerator` - Numerator of the vector-length multiplier.
    /// * `denominator` - Denominator of the vector-length multiplier.
    ///
    /// # Returns
    ///
    /// The multiplier, or `None` if either part is zero.
    #[must_use]
    pub const fn new(numerator: u32, denominator: u32) -> Option<Self> {
        if numerator == 0 || denominator == 0 {
            return None;
        }
        Some(Self {
            numerator,
            denominator,
        })
    }

    /// Returns the neutral vector-length multiplier.
    #[must_use]
    pub const fn one() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }

    /// Returns `true` when the multiplier has non-zero parts.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.numerator != 0 && self.denominator != 0
    }
}

/// Tail-lane behavior for scalable vector operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorTailPolicy {
    /// Inactive tail lanes may take any value.
    Agnostic,
    /// Inactive tail lanes preserve their previous value.
    Undisturbed,
}

/// Inactive-mask-lane behavior for scalable vector operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorMaskPolicy {
    /// Inactive mask lanes may take any value.
    Agnostic,
    /// Inactive mask lanes preserve their previous value.
    Undisturbed,
}

/// Describes a scalable vector value whose concrete lane count is runtime-dependent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScalableVectorShape {
    /// Minimum number of lanes guaranteed by the target type.
    pub min_lane_count: u32,
    /// Kind of scalar stored in each lane.
    pub lane_kind: VectorLaneKind,
    /// Bit width of each lane.
    pub lane_bits: u32,
    /// Runtime vector-length multiplier for this type.
    pub length_multiplier: VectorLengthMultiplier,
    /// Tail-lane behavior associated with operations over this type.
    pub tail_policy: VectorTailPolicy,
    /// Mask-lane behavior associated with predicated operations over this type.
    pub mask_policy: VectorMaskPolicy,
}

impl ScalableVectorShape {
    /// Creates a scalable vector shape when its lane and multiplier fields are valid.
    ///
    /// A scalable type only fixes the *minimum* lane count; the concrete count is
    /// the minimum scaled by `length_multiplier` at runtime. The constructor
    /// guards the two fields that must be non-zero for that scaling to be
    /// meaningful (min lane count and lane width) and defers to the multiplier's
    /// own validity check, returning `None` if any is degenerate.
    ///
    /// # Arguments
    ///
    /// * `min_lane_count` - Minimum number of lanes guaranteed by the type.
    /// * `lane_kind` - Scalar category stored in each lane.
    /// * `lane_bits` - Bit width of a single lane.
    /// * `length_multiplier` - Runtime vector-length multiplier for the type.
    /// * `tail_policy` - Tail-lane behavior for operations over the type.
    /// * `mask_policy` - Inactive-mask-lane behavior for predicated operations.
    ///
    /// # Returns
    ///
    /// The validated shape, or `None` if the minimum lane count or lane width is
    /// zero, or the multiplier is invalid.
    #[must_use]
    pub const fn new(
        min_lane_count: u32,
        lane_kind: VectorLaneKind,
        lane_bits: u32,
        length_multiplier: VectorLengthMultiplier,
        tail_policy: VectorTailPolicy,
        mask_policy: VectorMaskPolicy,
    ) -> Option<Self> {
        if min_lane_count == 0 || lane_bits == 0 || !length_multiplier.is_valid() {
            return None;
        }
        Some(Self {
            min_lane_count,
            lane_kind,
            lane_bits,
            length_multiplier,
            tail_policy,
            mask_policy,
        })
    }

    /// Returns `true` when the scalable vector descriptor is structurally valid.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.min_lane_count != 0 && self.lane_bits != 0 && self.length_multiplier.is_valid()
    }
}

/// Describes a scalable predicate or mask value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScalableVectorMaskShape {
    /// Minimum number of predicate lanes guaranteed by the target type.
    pub min_lane_count: u32,
    /// Number of bits used by each predicate lane.
    pub lane_bits: u32,
    /// Runtime vector-length multiplier for this predicate type.
    pub length_multiplier: VectorLengthMultiplier,
}

impl ScalableVectorMaskShape {
    /// Creates a scalable vector mask shape when all fields are non-zero.
    ///
    /// # Arguments
    ///
    /// * `min_lane_count` - Minimum number of predicate lanes guaranteed by the type.
    /// * `lane_bits` - Bits used by each predicate lane.
    /// * `length_multiplier` - Runtime vector-length multiplier for the type.
    ///
    /// # Returns
    ///
    /// The validated mask shape, or `None` if the minimum lane count or lane
    /// width is zero, or the multiplier is invalid.
    #[must_use]
    pub const fn new(
        min_lane_count: u32,
        lane_bits: u32,
        length_multiplier: VectorLengthMultiplier,
    ) -> Option<Self> {
        if min_lane_count == 0 || lane_bits == 0 || !length_multiplier.is_valid() {
            return None;
        }
        Some(Self {
            min_lane_count,
            lane_bits,
            length_multiplier,
        })
    }

    /// Returns `true` when the scalable mask descriptor is structurally valid.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.min_lane_count != 0 && self.lane_bits != 0 && self.length_multiplier.is_valid()
    }
}

/// Unified descriptor for fixed-width and scalable vector values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorDescriptor {
    /// Fixed-width vector with a statically known lane count and total width.
    Fixed(VectorShape),
    /// Scalable vector with a runtime-dependent lane count.
    Scalable(ScalableVectorShape),
}

impl VectorDescriptor {
    /// Returns the vector lane kind.
    #[must_use]
    pub const fn lane_kind(self) -> VectorLaneKind {
        match self {
            Self::Fixed(shape) => shape.lane_kind,
            Self::Scalable(shape) => shape.lane_kind,
        }
    }

    /// Returns the vector lane bit width.
    #[must_use]
    pub const fn lane_bits(self) -> u32 {
        match self {
            Self::Fixed(shape) => shape.lane_bits,
            Self::Scalable(shape) => shape.lane_bits,
        }
    }

    /// Returns the fixed lane count when statically known.
    #[must_use]
    pub const fn fixed_lane_count(self) -> Option<u32> {
        match self {
            Self::Fixed(shape) => Some(shape.lane_count),
            Self::Scalable(_) => None,
        }
    }

    /// Returns the minimum guaranteed lane count.
    #[must_use]
    pub const fn min_lane_count(self) -> u32 {
        match self {
            Self::Fixed(shape) => shape.lane_count,
            Self::Scalable(shape) => shape.min_lane_count,
        }
    }

    /// Returns the fixed total bit width when statically known.
    #[must_use]
    pub const fn total_bits(self) -> Option<u32> {
        match self {
            Self::Fixed(shape) => Some(shape.total_bits),
            Self::Scalable(_) => None,
        }
    }

    /// Returns the canonical mask descriptor for this vector's lane count.
    ///
    /// Derives the predicate shape that pairs with this vector: one 1-bit lane
    /// per vector lane, preserving the fixed-vs-scalable distinction (and, for
    /// scalable vectors, the same length multiplier) so a predicated operation's
    /// mask tracks the vector length exactly.
    ///
    /// # Returns
    ///
    /// A mask descriptor with matching lane count and single-bit lanes.
    #[must_use]
    pub const fn mask_descriptor(self) -> VectorMaskDescriptor {
        match self {
            Self::Fixed(shape) => VectorMaskDescriptor::Fixed(VectorMaskShape {
                lane_count: shape.lane_count,
                lane_bits: 1,
            }),
            Self::Scalable(shape) => VectorMaskDescriptor::Scalable(ScalableVectorMaskShape {
                min_lane_count: shape.min_lane_count,
                lane_bits: 1,
                length_multiplier: shape.length_multiplier,
            }),
        }
    }
}

/// Unified descriptor for fixed-width and scalable vector masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorMaskDescriptor {
    /// Fixed-width mask with a statically known lane count.
    Fixed(VectorMaskShape),
    /// Scalable predicate mask with a runtime-dependent lane count.
    Scalable(ScalableVectorMaskShape),
}

impl VectorMaskDescriptor {
    /// Returns the fixed lane count when statically known.
    #[must_use]
    pub const fn fixed_lane_count(self) -> Option<u32> {
        match self {
            Self::Fixed(shape) => Some(shape.lane_count),
            Self::Scalable(_) => None,
        }
    }

    /// Returns the minimum guaranteed mask lane count.
    #[must_use]
    pub const fn min_lane_count(self) -> u32 {
        match self {
            Self::Fixed(shape) => shape.lane_count,
            Self::Scalable(shape) => shape.min_lane_count,
        }
    }

    /// Returns the number of bits used by each mask lane.
    #[must_use]
    pub const fn lane_bits(self) -> u32 {
        match self {
            Self::Fixed(shape) => shape.lane_bits,
            Self::Scalable(shape) => shape.lane_bits,
        }
    }
}

/// One lane selector in a vector shuffle mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum VectorShuffleLane {
    /// Produces an undefined lane.
    Undef,
    /// Produces a zero lane.
    Zero,
    /// Selects a lane from the first vector input.
    Left(u32),
    /// Selects a lane from the second vector input.
    Right(u32),
}

/// Describes lane selection for one- or two-input vector shuffles.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorShuffleMask {
    lanes: Vec<VectorShuffleLane>,
}

impl VectorShuffleMask {
    /// Creates a shuffle mask from explicit lane selectors.
    #[must_use]
    pub fn new(lanes: Vec<VectorShuffleLane>) -> Self {
        Self { lanes }
    }

    /// Returns the output lane selectors.
    #[must_use]
    pub fn lanes(&self) -> &[VectorShuffleLane] {
        &self.lanes
    }

    /// Returns the mask selecting what `indices` select from a left input of
    /// `left_lanes` lanes and a right one of `right_lanes`, laid end to end:
    /// an index below `left_lanes` is a left lane, one past it a right lane.
    /// `None` where an index lies past both, which selects no lane, or where
    /// there are no indices.
    #[must_use]
    pub fn of_indices(
        indices: impl IntoIterator<Item = u128>,
        left_lanes: u32,
        right_lanes: Option<u32>,
    ) -> Option<Self> {
        let lanes = indices
            .into_iter()
            .map(|index| {
                let index = u32::try_from(index).ok()?;
                match index.checked_sub(left_lanes) {
                    None => Some(VectorShuffleLane::Left(index)),
                    Some(right) if right < right_lanes? => Some(VectorShuffleLane::Right(right)),
                    Some(_) => None,
                }
            })
            .collect::<Option<Vec<_>>>()?;
        (!lanes.is_empty()).then_some(Self { lanes })
    }

    /// Returns `true` when every selected input lane is in bounds.
    ///
    /// Validates the mask against the widths of the shuffle's actual inputs:
    /// `Undef` and `Zero` selectors are always in range, a `Left(i)` must index
    /// within `left_lanes`, and a `Right(i)` requires a second input to exist and
    /// index within it. An empty mask is rejected, since a shuffle must produce at
    /// least one lane.
    ///
    /// # Arguments
    ///
    /// * `left_lanes` - Lane count of the first (mandatory) input vector.
    /// * `right_lanes` - Lane count of the second input, or `None` for a
    ///   one-input shuffle.
    ///
    /// # Returns
    ///
    /// `true` if the mask is non-empty and every selector is in range for the
    /// given inputs.
    #[must_use]
    pub fn is_valid_for(&self, left_lanes: u32, right_lanes: Option<u32>) -> bool {
        if self.lanes.is_empty() {
            return false;
        }
        self.lanes.iter().all(|lane| match *lane {
            VectorShuffleLane::Undef | VectorShuffleLane::Zero => true,
            VectorShuffleLane::Left(idx) => idx < left_lanes,
            VectorShuffleLane::Right(idx) => right_lanes.is_some_and(|count| idx < count),
        })
    }

    /// Returns `true` when the shuffle keeps the first input's low lanes where
    /// they are and zero-fills every lane above them.
    ///
    /// `Left(0), Left(1), …, Left(k - 1)` followed only by `Zero` lanes, with at
    /// least one lane kept. Such a shuffle moves no bit: it is the input's low
    /// bits, zero-extended when the output is wider and truncated when it is
    /// narrower - how a value is placed into, or read out of, the low end of a
    /// wider register.
    #[must_use]
    pub fn keeps_low_lanes(&self) -> bool {
        let kept = self
            .lanes
            .iter()
            .enumerate()
            .take_while(|(at, lane)| {
                u32::try_from(*at).is_ok_and(|at| **lane == VectorShuffleLane::Left(at))
            })
            .count();
        kept > 0
            && self
                .lanes
                .get(kept..)
                .is_some_and(|rest| rest.iter().all(|lane| *lane == VectorShuffleLane::Zero))
    }
}

/// Endianness of a target architecture.
///
/// Determines the byte ordering for multi-byte integer and pointer values in
/// memory. Used by [`Target::endianness`] to let passes and codegen reason
/// about byte layout.
///
/// # Examples
///
/// ```rust
/// use analyssa::target::Endianness;
///
/// // x86, RISC-V, and Nios II are little-endian
/// assert_eq!(Endianness::Little, Endianness::Little);
///
/// // MIPS (big-endian mode), ARM (big-endian mode), and SPARC are big-endian
/// assert_eq!(Endianness::Big, Endianness::Big);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Endianness {
    /// Least significant byte stored first (x86, RISC-V, Nios II, default).
    Little,
    /// Most significant byte stored first (MIPS BE, ARM BE, SPARC, z/Arch).
    Big,
}

impl Endianness {
    /// Returns `true` if this is little-endian.
    #[must_use]
    pub const fn is_little(self) -> bool {
        matches!(self, Self::Little)
    }

    /// Returns `true` if this is big-endian.
    #[must_use]
    pub const fn is_big(self) -> bool {
        matches!(self, Self::Big)
    }

    /// Reads the unsigned number `bytes` hold in this byte order.
    ///
    /// Any width up to eight bytes; a longer slice keeps the low eight bytes'
    /// worth of the number, as the shifts out of the top discard the rest.
    ///
    /// # Examples
    ///
    /// ```
    /// use analyssa::Endianness;
    ///
    /// assert_eq!(Endianness::Little.read_uint(&[0x34, 0x12]), 0x1234);
    /// assert_eq!(Endianness::Big.read_uint(&[0x12, 0x34, 0x56]), 0x12_3456);
    /// ```
    #[must_use]
    pub fn read_uint(self, bytes: &[u8]) -> u64 {
        let push = |value: u64, byte: &u8| (value << 8) | u64::from(*byte);
        match self {
            Self::Little => bytes.iter().rev().fold(0, push),
            Self::Big => bytes.iter().fold(0, push),
        }
    }

    /// Converts a `u16` value from this endianness to the host's native byte
    /// order.
    ///
    /// On little-endian hosts this is a no-op for [`Little`](Endianness::Little)
    /// and a byte swap for [`Big`](Endianness::Big).
    #[must_use]
    pub fn to_native_u16(self, value: u16) -> u16 {
        match self {
            Self::Little => u16::from_le(value),
            Self::Big => u16::from_be(value),
        }
    }

    /// Converts a `u32` value from this endianness to the host's native byte
    /// order.
    #[must_use]
    pub fn to_native_u32(self, value: u32) -> u32 {
        match self {
            Self::Little => u32::from_le(value),
            Self::Big => u32::from_be(value),
        }
    }

    /// Converts a `u64` value from this endianness to the host's native byte
    /// order.
    #[must_use]
    pub fn to_native_u64(self, value: u64) -> u64 {
        match self {
            Self::Little => u64::from_le(value),
            Self::Big => u64::from_be(value),
        }
    }

    /// Converts a `u128` value from this endianness to the host's native byte
    /// order.
    #[must_use]
    pub fn to_native_u128(self, value: u128) -> u128 {
        match self {
            Self::Little => u128::from_le(value),
            Self::Big => u128::from_be(value),
        }
    }

    /// Converts a `u16` value from the host's native byte order to this
    /// endianness.
    #[must_use]
    pub fn from_native_u16(self, value: u16) -> u16 {
        match self {
            Self::Little => u16::to_le(value),
            Self::Big => u16::to_be(value),
        }
    }

    /// Converts a `u32` value from the host's native byte order to this
    /// endianness.
    #[must_use]
    pub fn from_native_u32(self, value: u32) -> u32 {
        match self {
            Self::Little => u32::to_le(value),
            Self::Big => u32::to_be(value),
        }
    }

    /// Converts a `u64` value from the host's native byte order to this
    /// endianness.
    #[must_use]
    pub fn from_native_u64(self, value: u64) -> u64 {
        match self {
            Self::Little => u64::to_le(value),
            Self::Big => u64::to_be(value),
        }
    }

    /// Converts a `u128` value from the host's native byte order to this
    /// endianness.
    #[must_use]
    pub fn from_native_u128(self, value: u128) -> u128 {
        match self {
            Self::Little => u128::to_le(value),
            Self::Big => u128::to_be(value),
        }
    }

    /// Returns the bytes of a `u16` value in this endianness as a 2-byte array.
    #[must_use]
    pub fn bytes_of_u16(self, value: u16) -> [u8; 2] {
        self.from_native_u16(value).to_ne_bytes()
    }

    /// Returns the bytes of a `u32` value in this endianness as a 4-byte array.
    #[must_use]
    pub fn bytes_of_u32(self, value: u32) -> [u8; 4] {
        self.from_native_u32(value).to_ne_bytes()
    }

    /// Returns the bytes of a `u64` value in this endianness as an 8-byte array.
    #[must_use]
    pub fn bytes_of_u64(self, value: u64) -> [u8; 8] {
        self.from_native_u64(value).to_ne_bytes()
    }

    /// Returns the bytes of a `u128` value in this endianness as a 16-byte array.
    #[must_use]
    pub fn bytes_of_u128(self, value: u128) -> [u8; 16] {
        self.from_native_u128(value).to_ne_bytes()
    }

    /// Reads a `u16` from a byte slice in this endianness.
    ///
    /// The read goes through a fixed-size array conversion, so `bytes` must be
    /// exactly 2 bytes long; any other length returns `0` rather than panicking.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Source bytes to decode; must be exactly 2 bytes.
    ///
    /// # Returns
    ///
    /// The decoded value, or `0` if `bytes` is not exactly 2 bytes long.
    #[must_use]
    pub fn read_u16(self, bytes: &[u8]) -> u16 {
        let arr: [u8; 2] = match bytes.try_into() {
            Ok(a) => a,
            Err(_) => return 0,
        };
        self.to_native_u16(u16::from_ne_bytes(arr))
    }

    /// Reads a `u32` from a byte slice in this endianness.
    ///
    /// The read goes through a fixed-size array conversion, so `bytes` must be
    /// exactly 4 bytes long; any other length returns `0` rather than panicking.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Source bytes to decode; must be exactly 4 bytes.
    ///
    /// # Returns
    ///
    /// The decoded value, or `0` if `bytes` is not exactly 4 bytes long.
    #[must_use]
    pub fn read_u32(self, bytes: &[u8]) -> u32 {
        let arr: [u8; 4] = match bytes.try_into() {
            Ok(a) => a,
            Err(_) => return 0,
        };
        self.to_native_u32(u32::from_ne_bytes(arr))
    }

    /// Reads a `u64` from a byte slice in this endianness.
    ///
    /// The read goes through a fixed-size array conversion, so `bytes` must be
    /// exactly 8 bytes long; any other length returns `0` rather than panicking.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Source bytes to decode; must be exactly 8 bytes.
    ///
    /// # Returns
    ///
    /// The decoded value, or `0` if `bytes` is not exactly 8 bytes long.
    #[must_use]
    pub fn read_u64(self, bytes: &[u8]) -> u64 {
        let arr: [u8; 8] = match bytes.try_into() {
            Ok(a) => a,
            Err(_) => return 0,
        };
        self.to_native_u64(u64::from_ne_bytes(arr))
    }

    /// Reads a `u128` from a byte slice in this endianness.
    ///
    /// The read goes through a fixed-size array conversion, so `bytes` must be
    /// exactly 16 bytes long; any other length returns `0` rather than panicking.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Source bytes to decode; must be exactly 16 bytes.
    ///
    /// # Returns
    ///
    /// The decoded value, or `0` if `bytes` is not exactly 16 bytes long.
    #[must_use]
    pub fn read_u128(self, bytes: &[u8]) -> u128 {
        let arr: [u8; 16] = match bytes.try_into() {
            Ok(a) => a,
            Err(_) => return 0,
        };
        self.to_native_u128(u128::from_ne_bytes(arr))
    }

    /// Reads the unsigned number `bytes` hold in this byte order, as
    /// [`Self::read_uint`] does, at any width up to sixteen bytes: a vector
    /// lane, or an integer wider than a `u64`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Source bytes to decode; at most 16 bytes.
    ///
    /// # Returns
    ///
    /// The decoded value, or `None` if `bytes` is longer than 16 bytes.
    #[must_use]
    pub fn read_wide_uint(self, bytes: &[u8]) -> Option<u128> {
        let mut little = [0u8; 16];
        let used = little.get_mut(..bytes.len())?;
        used.copy_from_slice(bytes);
        if self == Self::Big {
            used.reverse();
        }
        Some(u128::from_le_bytes(little))
    }

    /// Returns the byte representation of a native-width value for this
    /// endianness, given the pointer size.
    ///
    /// The value is first masked to `ptr_size`, then laid out in this byte order,
    /// so the result is always exactly `ptr_size`'s width in bytes. `Bit128`
    /// promotes through the 128-bit mask path (the incoming `u64` is
    /// zero-extended); every other width truncates the mask to its own byte
    /// count.
    ///
    /// # Arguments
    ///
    /// * `value` - Native-width value to encode (masked to `ptr_size`).
    /// * `ptr_size` - Target pointer width that fixes the output byte count.
    ///
    /// # Returns
    ///
    /// The value's bytes in this endianness, `ptr_size` bytes long.
    #[must_use]
    pub fn bytes_of_ptr_sized(self, value: u64, ptr_size: PointerSize) -> Vec<u8> {
        let masked = ptr_size.mask_unsigned(value);
        match ptr_size {
            PointerSize::Bit8 => vec![masked as u8],
            PointerSize::Bit16 => self.bytes_of_u16(masked as u16).to_vec(),
            PointerSize::Bit32 => self.bytes_of_u32(masked as u32).to_vec(),
            PointerSize::Bit64 => self.bytes_of_u64(masked).to_vec(),
            PointerSize::Bit128 => {
                // For 128-bit, promote through the 128-bit mask path
                let v128 = ptr_size.mask_unsigned_128(u128::from(masked));
                self.bytes_of_u128(v128).to_vec()
            }
        }
    }

    /// Reads a pointer-sized unsigned value from bytes, given the target
    /// pointer size.
    ///
    /// Decodes exactly `ptr_size`'s worth of bytes in this endianness and widens
    /// the result to `u64`. `Bit8` reads a single byte; `Bit128` reads a full
    /// `u128` and truncates to `u64`, so the high 64 bits are discarded. As with
    /// the fixed-width readers, a slice whose length does not match `ptr_size`
    /// decodes as `0` rather than panicking.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Source bytes to decode; length should match `ptr_size`.
    /// * `ptr_size` - Target pointer width that fixes how many bytes are read.
    ///
    /// # Returns
    ///
    /// The decoded value widened to `u64`.
    #[must_use]
    pub fn read_ptr_sized(self, bytes: &[u8], ptr_size: PointerSize) -> u64 {
        match ptr_size {
            PointerSize::Bit8 => u64::from(bytes.first().copied().unwrap_or(0)),
            PointerSize::Bit16 => u64::from(self.read_u16(bytes)),
            PointerSize::Bit32 => u64::from(self.read_u32(bytes)),
            PointerSize::Bit64 => self.read_u64(bytes),
            PointerSize::Bit128 => {
                // Read as u128 then truncate to u64 (warning: loses precision)
                self.read_u128(bytes) as u64
            }
        }
    }
}

/// The abstraction that makes the SSA IR core generic over an instruction
/// set. See module docs.
///
/// `Clone + Debug + Eq + Hash` supertraits exist so derive macros on the
/// generic IR types (`ConstValue<T>`, `SsaOp<T>`, `SsaInstruction<T>`,
/// `SsaFunction<T>`, `MemoryLocation<T>`, …) do not need manual impls. The
/// implementation cost is negligible for marker-style impls.
pub trait Target: Clone + Debug + Eq + Hash + Sized + 'static {
    /// Reference to a user-defined or built-in type in the host's metadata.
    type TypeRef: Clone + Eq + Hash + Debug;

    /// Reference to a method in the host's metadata.
    type MethodRef: Clone + Eq + Hash + Debug;

    /// Reference to a field in the host's metadata.
    type FieldRef: Clone + Eq + Hash + Debug;

    /// Reference to a standalone signature in the host's metadata.
    type SigRef: Clone + Eq + Hash + Debug;

    /// Reference to a located or named entity in the host's address space - a
    /// function, a global, an import, a metadata member.
    ///
    /// Distinct from [`Self::MethodRef`], which names a *callable* and carries
    /// call-site identity: a symbol constant is a **value**, and most of the
    /// things it names are not callable at all.
    type SymbolRef: Clone + Eq + Hash + Debug;

    /// Host-defined exception-handler kind (e.g. EXCEPTION/FINALLY/FILTER on CIL).
    type ExceptionKind: Clone + Eq + Debug;

    /// Host's type representation (e.g. `SsaType` for CIL).
    type Type: Clone + Eq + Hash + Debug;

    /// Original-instruction breadcrumb retained on each `SsaInstruction` for
    /// debugging and source mapping. Hosts that don't want this can use `()`.
    type OriginalInstruction: Clone + Debug;

    /// Local-variable signature data preserved through SSA construction. Used
    /// by codegen to recover types that aren't structurally reconstructible
    /// from the SSA op stream.
    type LocalSignature: Clone + Debug;

    /// Host data carried on every SSA value, moved with it through
    /// renumbering, derivation, joins and substitution.
    ///
    /// `()` for a host that carries none. See [`crate::facts`].
    type VariableFacts: facts::VariableFacts;

    /// Host data carried on every instruction.
    ///
    /// `()` for a host that carries none. See [`crate::facts`].
    type InstructionFacts: facts::InstructionFacts;

    /// Host data carried on the function as a whole.
    ///
    /// `()` for a host that carries none. See [`crate::facts`].
    type FunctionFacts: facts::FunctionFacts;

    /// Pass-pipeline capability tag used by the analyssa scheduler for
    /// dependency-aware ordering. Hosts that don't run the pass scheduler
    /// can use `()`. Hosts that do should embed
    /// [`crate::scheduling::DeobfuscationCapability`] in their concrete
    /// enum so generic analyssa passes can declare provides/requires using
    /// the shared deobfuscation vocabulary.
    type Capability: Copy + Eq + Hash + Debug + Send + Sync + 'static;

    /// This host's spelling of a shared capability milestone.
    ///
    /// A pass written here describes its effect in the shared vocabulary of
    /// [`DeobfuscationCapability`], because it cannot name a value of a host's
    /// enum. This is the bridge, and it is what lets such a pass declare what
    /// it provides at all: without it the declaration can only be made by the
    /// host, one registration at a time, and a pass whose documentation says it
    /// provides something can quietly provide nothing.
    ///
    /// `None` is the honest answer for a host that does not model the
    /// milestone: the declaration is then simply absent rather than mapped onto
    /// a tag that means something else. The default is `None` for every
    /// milestone, which suits a host that does not run the scheduler.
    ///
    /// # Arguments
    ///
    /// * `shared` - The milestone, in the shared vocabulary.
    ///
    /// # Returns
    ///
    /// The host's tag for it, or `None` when the host has none.
    ///
    #[must_use]
    fn capability(shared: DeobfuscationCapability) -> Option<Self::Capability> {
        let _ = shared;
        None
    }

    /// The numeric address a symbol denotes, when it has one.
    ///
    /// **Required for numeric analyses not to regress when an address stops
    /// being an integer constant.** Once a host emits
    /// [`ConstValue::Symbol`] for a materialised
    /// address, `ConstValue::resolved_i64` - and therefore address normalisation, alias
    /// keys, and the interval lattice - can only still see the address by
    /// asking here. Hosts whose symbols are not addresses (bytecode metadata
    /// tokens) correctly return `None`.
    ///
    /// Whether a symbol with no address here still has storage is
    /// [`Self::symbol_bound`]'s question.
    fn symbol_address(_symbol: &Self::SymbolRef) -> Option<i64> {
        None
    }

    /// Whether the host proves `symbol` names storage the loader mapped
    /// before the function runs, though it has no address in this image: an
    /// import the loader must bind, without which the program does not run.
    ///
    /// Such a symbol's address is no null pointer, and a read of its first
    /// word cannot fault. A weak import, which the loader leaves null where
    /// nothing defines it, is not bound. `facts` are the function's own
    /// ([`SsaFunction::facts`](crate::ir::SsaFunction::facts)), where a host
    /// carries what its loader states of the imports the function reads.
    /// `false` by default: a symbol the host says nothing of may be null.
    ///
    /// # Arguments
    ///
    /// * `facts` - The function's host facts.
    /// * `symbol` - The symbol asked about.
    fn symbol_bound(facts: &Self::FunctionFacts, symbol: &Self::SymbolRef) -> bool {
        let _ = (facts, symbol);
        false
    }

    /// Resolves a symbol in its actual storage coordinate space for execution.
    ///
    /// The default interprets numeric symbols as image addresses. Targets can
    /// retain an allocation, stack or absolute root without supplying a false
    /// image address to numeric analyses.
    fn execution_address(symbol: &Self::SymbolRef, width: PointerSize) -> Option<Address> {
        Self::symbol_address(symbol).map(|offset| Address::new(Root::Symbol, offset, width))
    }

    /// The symbol denoting the numeric address `address`, when the target
    /// names addresses by symbols: the inverse of [`Self::symbol_address`].
    ///
    /// A pass that writes an address back into the function names it the way
    /// the host's own lift does, rather than as a bare number whose type the
    /// target cannot size. The default names none, and such a pass then falls
    /// back to the number.
    ///
    /// # Arguments
    ///
    /// * `address` - The address.
    fn symbol_at(_address: u64) -> Option<Self::SymbolRef> {
        None
    }

    /// The callable at the numeric address `address`, when the target names
    /// code by address: what a call through a pointer that holds `address`
    /// calls.
    ///
    /// An indirect call whose function pointer folded to a constant is a direct
    /// call, and naming its callee the way the host's own lift names one is
    /// what lets every analysis that follows a call edge follow it. The default
    /// names none, and such a call stays indirect.
    ///
    /// # Arguments
    ///
    /// * `address` - The address the pointer holds.
    fn method_at(_address: u64) -> Option<Self::MethodRef> {
        None
    }

    /// Keeps `read` - a load of image data an analysis folded away - with the
    /// function it was read by.
    ///
    /// The default keeps nothing, which is right for a host whose facts
    /// describe no memory. See [`ImageRead`]
    /// for why a host that recovers data objects wants it.
    ///
    /// # Arguments
    ///
    /// * `facts` - The folding function's facts.
    /// * `read` - What was read, and where.
    fn record_image_read(facts: &mut Self::FunctionFacts, read: ImageRead<Self>) {
        let _ = (facts, read);
    }

    /// Keeps that the function reads `len` bytes at `address`, in the host's
    /// symbol space, where the image maps them: storage the loader mapped,
    /// which a read cannot fault on ([`Self::address_mapped`]).
    ///
    /// Recorded while the image is at hand (the memory pass), so a later step
    /// that has only the function - the lowering's folds - still knows the
    /// read cannot fault. The default keeps nothing.
    ///
    /// # Arguments
    ///
    /// * `facts` - The reading function's facts.
    /// * `address` - Where the read starts, in the host's symbol space.
    /// * `len` - How many bytes it reads.
    fn record_mapped_read(facts: &mut Self::FunctionFacts, address: u64, len: u64) {
        let _ = (facts, address, len);
    }

    /// Whether `facts` hold that `len` bytes at `address` are storage the
    /// image maps ([`Self::record_mapped_read`]), so a read of them cannot
    /// fault. `false` by default.
    ///
    /// # Arguments
    ///
    /// * `facts` - The function's facts.
    /// * `address` - Where the read starts, in the host's symbol space.
    /// * `len` - How many bytes it reads.
    fn address_mapped(facts: &Self::FunctionFacts, address: u64, len: u64) -> bool {
        let _ = (facts, address, len);
        false
    }

    /// The target's pointer type of `bytes` bytes, when it has one.
    ///
    /// An associated function, so it cannot read the instance's own pointer
    /// width: the caller supplies it, from what it holds - a pointer-sized
    /// image cell, the [`AnalysisContext`](crate::analysis::AnalysisContext)'s
    /// pointer size. The default names none, and a pass that would type a
    /// value as an address then leaves its type as it was.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The pointer's width in bytes.
    fn pointer_type(_bytes: u32) -> Option<Self::Type> {
        None
    }

    /// Pointer width in bytes (typically 4 or 8). Runtime so bi-arch hosts
    /// can vary it per-instance.
    fn ptr_bytes(&self) -> u32;

    /// Byte ordering for multi-byte values in memory.
    ///
    /// Returns [`Endianness::Little`] by default, which covers x86, RISC-V,
    /// and the default mode of most modern ISAs. Hosts that target bi-endian
    /// architectures (MIPS, ARM) or big-endian-only architectures (SPARC,
    /// z/Arch) should override this.
    fn endianness(&self) -> Endianness {
        Endianness::Little
    }

    /// Returns a placeholder original-instruction value for synthetic IR
    /// nodes (e.g., phi-node carriers, transform-inserted instructions).
    fn synthetic_instruction() -> Self::OriginalInstruction;

    /// The canonical "unknown / not-yet-inferred" type. Used by builders and
    /// fixtures that haven't run inference.
    fn unknown_type() -> Self::Type;

    /// The semantic layout of a borrowed type at the machine's pointer width,
    /// aliases resolved: what a static execution reads and writes a value of
    /// the type as. `None` where the host states none; the default states none.
    /// The layout is stable for the same type and machine.
    fn type_layout(_ty: &Self::Type, _ptr_size: PointerSize) -> Option<TypeLayout> {
        None
    }

    /// Returns whether `t` stores a Boolean value rather than a signed one-bit integer.
    ///
    /// Hosts with a distinct Boolean type override this predicate.
    fn is_boolean(_t: &Self::Type) -> bool {
        false
    }

    /// Returns an attested canonical Boolean memory representation.
    ///
    /// Logical Boolean classification alone does not declare a storage extent
    /// or encoding. The default declines memory interpretation. Hosts return a
    /// descriptor only when unsigned zero and one are their exact stored forms.
    fn boolean_storage(_t: &Self::Type) -> Option<BooleanStorage> {
        None
    }

    /// `true` if `t` is an integer type (any width, signed or unsigned).
    fn is_integer(t: &Self::Type) -> bool;

    /// `true` if `t` is a floating-point type.
    fn is_floating(t: &Self::Type) -> bool;

    /// `true` if `t` is a signed integer type.
    fn is_signed(t: &Self::Type) -> bool;

    /// `true` if `t` is a pointer or managed reference (byref) to another type.
    fn is_pointer(t: &Self::Type) -> bool;

    /// `true` if `t` is a reference type (object/string/class/array).
    fn is_reference(t: &Self::Type) -> bool;

    /// `true` if `t` is the unknown / not-yet-inferred type.
    fn is_unknown(t: &Self::Type) -> bool;

    /// Bit-width for primitive types where it is statically known. `None` for
    /// pointer-sized integers, references, and aggregates.
    fn bit_width(t: &Self::Type) -> Option<u32>;

    /// The width the verifier compares a value of type `t` at as an operand of
    /// a scalar arithmetic, logical, shift, comparison or select operation,
    /// where it requires the operands of one operation to agree
    /// ([`crate::analysis::VerifierError::InvalidOperandWidths`]). `None` where
    /// the width is not fixed - a pointer-sized integer, a float whose
    /// evaluation width is the runtime's, a truth value - and such an operand is
    /// not compared.
    ///
    /// The default is the type's own width ([`Self::bit_width`]), and a target
    /// should answer no other: every analysis reads an operand at its type's
    /// own width, and a conversion reads its operand at that width and extends
    /// it by the conversion's signedness. A host whose operations widen what
    /// they read - CIL's evaluation stack holds every integer up to 32 bits as
    /// an `int32` (ECMA-335 III.1.1) - states each widening as an
    /// [`SsaOp::IntConv`](crate::ir::ops::SsaOp::IntConv) of the narrow value.
    /// A wider answer here only makes the verifier accept operands of mixed
    /// widths, which the analyses then read at their own widths, not as the
    /// host computes them.
    fn operand_width(t: &Self::Type) -> Option<u32> {
        Self::bit_width(t)
    }

    /// The integer type `bits` wide with signedness `signed`, when the target
    /// has one to name.
    ///
    /// A pass that synthesises an integer operation - static execution writing
    /// a decoded byte back, say - needs a type for its result and cannot build
    /// one from nothing. The default knows none, so such a pass does nothing on
    /// a target that does not answer.
    ///
    /// # Arguments
    ///
    /// * `bits` - The width, in bits.
    /// * `signed` - Whether the integer is signed.
    fn integer_type(_bits: u32, _signed: bool) -> Option<Self::Type> {
        None
    }

    /// The floating-point type of `format`, when the target has one to name.
    ///
    /// What an integer's bits read as a float are typed as: a conversion from
    /// a float whose operand turned out to be held as an integer reads it
    /// through this type. The default knows none, and such a conversion is
    /// left reading the integer.
    ///
    /// # Arguments
    ///
    /// * `format` - The float's format.
    fn float_type(_format: FloatFormat) -> Option<Self::Type> {
        None
    }

    /// The format of the floating-point type `t`.
    ///
    /// The default is the IEEE format of the type's width ([`FloatFormat::of_width`]),
    /// which is right for every target whose floats are all IEEE; a target
    /// with bfloat16 values names their format itself.
    ///
    /// # Arguments
    ///
    /// * `t` - The type.
    fn float_format(t: &Self::Type) -> Option<FloatFormat> {
        if !Self::is_floating(t) {
            return None;
        }
        FloatFormat::of_width(Self::bit_width(t)?)
    }

    /// The rounding a float operation that states none applies, where the
    /// target fixes it: the CLI rounds to nearest (ECMA-335 I.12.1.3).
    ///
    /// The default, `None`, is a mode the program sets at run time
    /// (`fesetround`, RISC-V `frm`, x86 `MXCSR.RC`, AArch64 `FPCR.RMode`),
    /// so a constant fold gives only a result every mode rounds alike
    /// ([`FloatRounding::decide`]).
    fn environment_rounding() -> Option<FloatRounding> {
        None
    }

    /// Returns `true` if `t` is a vector type known to this target.
    fn is_vector(t: &Self::Type) -> bool {
        Self::vector_descriptor(t).is_some()
    }

    /// Returns the target-independent vector shape for `t`, if known.
    fn vector_shape(_t: &Self::Type) -> Option<VectorShape> {
        None
    }

    /// Returns the scalable vector shape for `t`, if known.
    fn scalable_vector_shape(_t: &Self::Type) -> Option<ScalableVectorShape> {
        None
    }

    /// Returns the fixed or scalable vector descriptor for `t`, if known.
    fn vector_descriptor(t: &Self::Type) -> Option<VectorDescriptor> {
        Self::vector_shape(t)
            .map(VectorDescriptor::Fixed)
            .or_else(|| Self::scalable_vector_shape(t).map(VectorDescriptor::Scalable))
    }

    /// Returns the target type for `shape`, if the target supports it.
    fn vector_type(_shape: VectorShape) -> Option<Self::Type> {
        None
    }

    /// Returns the target type for scalable `shape`, if the target supports it.
    fn scalable_vector_type(_shape: ScalableVectorShape) -> Option<Self::Type> {
        None
    }

    /// Returns the scalar lane type for `shape`, if the target supports it.
    fn vector_lane_type(_shape: VectorShape) -> Option<Self::Type> {
        None
    }

    /// Returns the scalar lane type for scalable `shape`, if the target supports it.
    fn scalable_vector_lane_type(_shape: ScalableVectorShape) -> Option<Self::Type> {
        None
    }

    /// Returns the scalar lane type for fixed or scalable `shape`, if supported.
    fn vector_descriptor_lane_type(shape: VectorDescriptor) -> Option<Self::Type> {
        match shape {
            VectorDescriptor::Fixed(shape) => Self::vector_lane_type(shape),
            VectorDescriptor::Scalable(shape) => Self::scalable_vector_lane_type(shape),
        }
    }

    /// Returns the fixed vector mask shape for `t`, if known.
    fn vector_mask_shape(_t: &Self::Type) -> Option<VectorMaskShape> {
        None
    }

    /// Returns the scalable vector mask shape for `t`, if known.
    fn scalable_vector_mask_shape(_t: &Self::Type) -> Option<ScalableVectorMaskShape> {
        None
    }

    /// Returns the fixed or scalable vector mask descriptor for `t`, if known.
    fn vector_mask_descriptor(t: &Self::Type) -> Option<VectorMaskDescriptor> {
        Self::vector_mask_shape(t)
            .map(VectorMaskDescriptor::Fixed)
            .or_else(|| Self::scalable_vector_mask_shape(t).map(VectorMaskDescriptor::Scalable))
    }

    /// Returns the target mask type for `shape`, if the target supports it.
    fn vector_mask_type(_shape: VectorMaskShape) -> Option<Self::Type> {
        None
    }

    /// Returns the target scalable mask type for `shape`, if the target supports it.
    fn scalable_vector_mask_type(_shape: ScalableVectorMaskShape) -> Option<Self::Type> {
        None
    }

    /// Returns the target mask type for fixed or scalable `shape`, if supported.
    fn vector_mask_descriptor_type(shape: VectorMaskDescriptor) -> Option<Self::Type> {
        match shape {
            VectorMaskDescriptor::Fixed(shape) => Self::vector_mask_type(shape),
            VectorMaskDescriptor::Scalable(shape) => Self::scalable_vector_mask_type(shape),
        }
    }

    /// Mnemonic for the original instruction breadcrumb (e.g. `"add"`, `"ret"`).
    /// Hosts that don't carry a real instruction return a placeholder.
    fn instruction_mnemonic(instr: &Self::OriginalInstruction) -> &'static str;

    /// RVA of the original instruction. Hosts without source mapping return 0.
    fn instruction_rva(instr: &Self::OriginalInstruction) -> u64;

    /// Declaration-order member index of `field` within its parent composite,
    /// when the host has resolved one.
    ///
    /// Field-sensitive analyses (points-to, alias partitioning) use this to keep
    /// distinct members of the same object disjoint; `None` means the field is
    /// unresolved and must fall back to the sound whole-object approximation.
    /// Defaults to `None`, which is exactly that conservative fallback.
    fn field_member_index(_field: &Self::FieldRef) -> Option<u32> {
        None
    }

    /// Classifies a host's opaque exception-handler flags.
    ///
    /// Required and total: every clause a host records is one of the four
    /// [`HandlerKind`]s, and a host that cannot tell them apart still has to say
    /// which one it is guessing. A defaulted hook would answer for hosts that
    /// never considered the question, and structured recovery would report that
    /// answer as fact.
    ///
    /// This is the crate's only handler classifier. Nothing else asks what a
    /// clause does, so nothing else can contradict it.
    fn handler_kind(flags: &Self::ExceptionKind) -> HandlerKind;

    /// Whether a handler of these flags is entered holding the exception, so
    /// that the exception object is a value at its entry
    /// ([`RuntimeValueKind::ExceptionObject`](crate::ir::ops::RuntimeValueKind::ExceptionObject)).
    ///
    /// Defaults to the CLR's rule, by the handler's kind: a catch or a filter
    /// is entered with the exception, and a finally or a fault, which run for
    /// an exit, is not. A host whose clauses are entered otherwise says so:
    /// an Itanium or EHABI landing pad for a cleanup is entered by the
    /// unwinder holding the exception, which it passes on when it resumes.
    fn handler_receives_exception(flags: &Self::ExceptionKind) -> bool {
        matches!(
            Self::handler_kind(flags),
            HandlerKind::Catch | HandlerKind::Filter
        )
    }

    /// Whether a handler of these flags is where the function resumes once a
    /// handler that runs elsewhere has dealt with the exception, so that its
    /// entry may lie inside the range it guards.
    ///
    /// No for the CLR, whose handlers are blocks of their own outside every
    /// range they guard. A host whose catch handlers are funclets resumes the
    /// function at a continuation, and that continuation can be code the same
    /// state still protects: MSVC x64 states one IP-to-state range over the
    /// calls before and after the point a `catch` returns to, so an exception
    /// raised after resuming is caught again by the same clause.
    fn handler_resumes_protected(flags: &Self::ExceptionKind) -> bool {
        let _ = flags;
        false
    }

    // Result-type queries used by `SsaOp::infer_result_type` to lift type
    // inference onto generic `Target`. Each is decomposed per opcode group.
    // All default to `None` so hosts only implement the queries they have a
    // useful answer for; test targets can keep every default.

    /// Result type for a `Const` op; mapped from the `ConstValue` variant.
    ///
    /// Each of these queries answers what the operation alone determines,
    /// without its function or its operands. A host whose pointer width is
    /// the function's, not the target's, has no answer for a native-width
    /// value (`NativeInt`, `NativeUInt`, [`Self::native_int_result_type`],
    /// [`Self::function_ptr_result_type`]) and answers `None`: the producer
    /// states the type, and a rewrite keeps it
    /// ([`SsaInstruction::set_op_preserving_type`](crate::ir::SsaInstruction::set_op_preserving_type)).
    fn result_type_for_const(_value: &ConstValue<Self>) -> Option<Self::Type> {
        None
    }

    /// Result type of a comparison op (`Ceq`, `Clt`, `Cgt`).
    fn comparison_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of plain integer arithmetic ops (`Add`, `Sub`, …, `SizeOf`).
    ///
    /// `None` for a host whose arithmetic has its operands' type, which the
    /// operation alone does not say.
    fn arithmetic_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of `LocalAlloc` and `ArrayLength` ops (CIL: native int).
    fn native_int_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of `Ckfinite` (CIL: F64).
    fn ckfinite_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of `LoadFunctionPtr` / `LoadVirtFunctionPtr` (CIL: native int).
    fn function_ptr_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of object-producing ops (`Box`, `NewObj`, `NewArr`,
    /// `CastClass`, `IsInst`).
    fn object_result_type() -> Option<Self::Type> {
        None
    }

    /// Result type of `UnboxAny` / `LoadObj`: a value-typed view of `r`.
    fn value_type_from_ref(_r: &Self::TypeRef) -> Option<Self::Type> {
        None
    }

    /// Result type of `Unbox`: a managed reference (`byref`) to the
    /// value-typed view of `r`.
    fn byref_value_type_from_ref(_r: &Self::TypeRef) -> Option<Self::Type> {
        None
    }

    /// Result type of `LoadElementAddr`: a managed reference (`byref`) to a
    /// class-typed element of `r`.
    fn byref_class_type_from_ref(_r: &Self::TypeRef) -> Option<Self::Type> {
        None
    }

    /// Projects the integer range and representation used by scalar conversions.
    ///
    /// The default reads declared integer and pointer layouts. Hosts with aliases
    /// or enumeration wrappers project their underlying scalar here; arithmetic
    /// and overflow checks remain in the shared integer kernel.
    fn integer_conversion_type(target: &Self::Type, ptr_size: PointerSize) -> Option<IntegerType> {
        IntegerType::of::<Self>(target, ptr_size)
    }

    /// Converts a constant to `target_type` through [`ConstValue::convert_to`].
    /// (CIL `conv.*` semantics). The default uses the shared integer kernel for
    /// attested scalar layouts. Other value and type domains remain unsupported.
    fn convert_const(
        value: &ConstValue<Self>,
        target_type: &Self::Type,
        unsigned_source: bool,
        ptr_bytes: u32,
    ) -> Option<ConstValue<Self>> {
        let ptr_size = PointerSize::try_from_bytes(ptr_bytes)?;
        Self::integer_conversion_type(target_type, ptr_size)?
            .convert(value, unsigned_source, false, ptr_size)
            .into_value()
    }

    /// Converts a scalar integer with overflow checking through the shared kernel.
    ///
    /// Returns no value on overflow or unsupported source and target layouts.
    fn convert_const_checked(
        value: &ConstValue<Self>,
        target_type: &Self::Type,
        unsigned_source: bool,
        ptr_bytes: u32,
    ) -> Option<ConstValue<Self>> {
        let ptr_size = PointerSize::try_from_bytes(ptr_bytes)?;
        Self::integer_conversion_type(target_type, ptr_size)?
            .convert(value, unsigned_source, true, ptr_size)
            .into_value()
    }

    /// Converts a signed 64-bit source through the host's typed conversion hook.
    ///
    /// This convenience entry point supplies an I64 constant to
    /// [`Self::convert_const`]. Evaluation of typed SSA operands uses that hook
    /// with the original constant, preserving its source width and bit pattern.
    fn evaluate_int_conv(
        value: i64,
        target_type: &Self::Type,
        unsigned: bool,
        ptr_bytes: u32,
    ) -> Option<ConstValue<Self>> {
        Self::convert_const(&ConstValue::I64(value), target_type, unsigned, ptr_bytes)
    }
}

impl<T: Target> Emit<T> for VectorShuffleLane {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Undef => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Zero => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Left(field_0) => {
                Emit::emit(&2u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Right(field_0) => {
                Emit::emit(&3u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorShuffleMask {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self { lanes } = self;
        Emit::emit(lanes, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorTailPolicy {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Agnostic => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Undisturbed => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        VectorShuffleLane::{Left, Right, Undef, Zero},
        *,
    };

    /// A mask keeps its input's low lanes only when it is the input's first
    /// lanes, in order, then zeros - a widening or a narrowing, never a
    /// permutation.
    #[test]
    fn keeps_low_lanes_is_a_prefix_then_zeros() {
        let keeps = |lanes: Vec<VectorShuffleLane>| VectorShuffleMask::new(lanes).keeps_low_lanes();
        assert!(keeps(vec![Left(0), Zero, Zero, Zero]), "a widening");
        assert!(keeps(vec![Left(0), Left(1)]), "a narrowing");
        assert!(keeps(vec![Left(0)]), "one lane");
        assert!(!keeps(vec![Left(1), Zero]), "not from lane 0");
        assert!(!keeps(vec![Left(0), Left(2)]), "a gap");
        assert!(
            !keeps(vec![Left(0), Zero, Left(1)]),
            "a lane after the zeros"
        );
        assert!(!keeps(vec![Left(0), Right(1)]), "a second input");
        assert!(!keeps(vec![Left(0), Undef]), "an undefined lane");
        assert!(!keeps(vec![Zero, Zero]), "nothing kept");
    }

    // Endianness query helpers

    /// `Little` reports little-endian and not big-endian.
    #[test]
    fn little_is_little_not_big() {
        assert!(Endianness::Little.is_little());
        assert!(!Endianness::Little.is_big());
    }

    /// `Big` reports big-endian and not little-endian.
    #[test]
    fn big_is_big_not_little() {
        assert!(Endianness::Big.is_big());
        assert!(!Endianness::Big.is_little());
    }

    // to_native - round-trip through native byte order (host is LE on x86)

    /// `Little::to_native_u16` matches `u16::from_le`.
    #[test]
    fn little_to_native_u16_is_le() {
        let value = 0x0102_u16;
        assert_eq!(Endianness::Little.to_native_u16(value), u16::from_le(value));
    }

    /// `Big::to_native_u16` matches `u16::from_be`.
    #[test]
    fn big_to_native_u16_is_be() {
        let value = 0x0102_u16;
        assert_eq!(Endianness::Big.to_native_u16(value), u16::from_be(value));
    }

    /// `Little::to_native_u32` matches `u32::from_le`.
    #[test]
    fn little_to_native_u32_is_le() {
        let value = 0x01020304_u32;
        assert_eq!(Endianness::Little.to_native_u32(value), u32::from_le(value));
    }

    /// `Big::to_native_u32` matches `u32::from_be`.
    #[test]
    fn big_to_native_u32_is_be() {
        let value = 0x01020304_u32;
        assert_eq!(Endianness::Big.to_native_u32(value), u32::from_be(value));
    }

    /// `Little::to_native_u64` matches `u64::from_le`.
    #[test]
    fn little_to_native_u64_is_le() {
        let value = 0x0102030405060708_u64;
        assert_eq!(Endianness::Little.to_native_u64(value), u64::from_le(value));
    }

    /// `Big::to_native_u64` matches `u64::from_be`.
    #[test]
    fn big_to_native_u64_is_be() {
        let value = 0x0102030405060708_u64;
        assert_eq!(Endianness::Big.to_native_u64(value), u64::from_be(value));
    }

    /// `Little::to_native_u128` matches `u128::from_le`.
    #[test]
    fn little_to_native_u128_is_le() {
        let value = 0x0102030405060708090a0b0c0d0e0f10_u128;
        assert_eq!(
            Endianness::Little.to_native_u128(value),
            u128::from_le(value)
        );
    }

    /// `Big::to_native_u128` matches `u128::from_be`.
    #[test]
    fn big_to_native_u128_is_be() {
        let value = 0x0102030405060708090a0b0c0d0e0f10_u128;
        assert_eq!(Endianness::Big.to_native_u128(value), u128::from_be(value));
    }

    // from_native - round-trip through native byte order

    /// `Little::from_native_u16` matches `u16::to_le`.
    #[test]
    fn little_from_native_u16_is_le() {
        let value = 0x0102_u16;
        assert_eq!(Endianness::Little.from_native_u16(value), u16::to_le(value));
    }

    /// `Big::from_native_u16` matches `u16::to_be`.
    #[test]
    fn big_from_native_u16_is_be() {
        let value = 0x0102_u16;
        assert_eq!(Endianness::Big.from_native_u16(value), u16::to_be(value));
    }

    /// `to_native` inverts `from_native` for both byte orders.
    #[test]
    fn from_native_round_trips_through_to_native() {
        let value = 0xdeadbeef_u32;
        for endianness in [Endianness::Little, Endianness::Big] {
            let converted = endianness.from_native_u32(value);
            let restored = endianness.to_native_u32(converted);
            assert_eq!(restored, value, "round-trip failed for {endianness:?}");
        }
    }

    // bytes_of_* - verify byte layout matches endianness

    /// Little-endian `bytes_of_u16` emits the least-significant byte first.
    #[test]
    fn little_bytes_of_u16_match_le_byte_order() {
        assert_eq!(Endianness::Little.bytes_of_u16(0x0102), [0x02, 0x01]);
    }

    /// Big-endian `bytes_of_u16` emits the most-significant byte first.
    #[test]
    fn big_bytes_of_u16_match_be_byte_order() {
        assert_eq!(Endianness::Big.bytes_of_u16(0x0102), [0x01, 0x02]);
    }

    /// Little-endian `bytes_of_u32` orders bytes least-significant first.
    #[test]
    fn little_bytes_of_u32_match_le_byte_order() {
        assert_eq!(
            Endianness::Little.bytes_of_u32(0x01020304),
            [0x04, 0x03, 0x02, 0x01]
        );
    }

    /// Big-endian `bytes_of_u32` orders bytes most-significant first.
    #[test]
    fn big_bytes_of_u32_match_be_byte_order() {
        assert_eq!(
            Endianness::Big.bytes_of_u32(0x01020304),
            [0x01, 0x02, 0x03, 0x04]
        );
    }

    /// Little-endian `bytes_of_u64` orders bytes least-significant first.
    #[test]
    fn little_bytes_of_u64_match_le_byte_order() {
        assert_eq!(
            Endianness::Little.bytes_of_u64(0x0102030405060708),
            [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );
    }

    /// Big-endian `bytes_of_u64` orders bytes most-significant first.
    #[test]
    fn big_bytes_of_u64_match_be_byte_order() {
        assert_eq!(
            Endianness::Big.bytes_of_u64(0x0102030405060708),
            [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
    }

    /// Little-endian `bytes_of_u128` orders bytes least-significant first.
    #[test]
    fn little_bytes_of_u128_match_le_byte_order() {
        let bytes = Endianness::Little.bytes_of_u128(0x0102030405060708090a0b0c0d0e0f10);
        assert_eq!(
            bytes,
            [
                0x10, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03,
                0x02, 0x01,
            ]
        );
    }

    /// Big-endian `bytes_of_u128` orders bytes most-significant first.
    #[test]
    fn big_bytes_of_u128_match_be_byte_order() {
        let bytes = Endianness::Big.bytes_of_u128(0x0102030405060708090a0b0c0d0e0f10);
        assert_eq!(
            bytes,
            [
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
                0x0f, 0x10,
            ]
        );
    }

    // read_u* - reading bytes back matches bytes_of_* output

    /// `read_u16` recovers the value written by `bytes_of_u16`.
    #[test]
    fn read_u16_round_trips_with_bytes_of_u16() {
        let value = 0xabcd_u16;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_u16(value);
            let restored = endianness.read_u16(&bytes);
            assert_eq!(
                restored, value,
                "read_u16 round-trip failed for {endianness:?}"
            );
        }
    }

    /// `read_u32` recovers the value written by `bytes_of_u32`.
    #[test]
    fn read_u32_round_trips_with_bytes_of_u32() {
        let value = 0xdeadbeef_u32;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_u32(value);
            let restored = endianness.read_u32(&bytes);
            assert_eq!(
                restored, value,
                "read_u32 round-trip failed for {endianness:?}"
            );
        }
    }

    /// `read_u64` recovers the value written by `bytes_of_u64`.
    #[test]
    fn read_u64_round_trips_with_bytes_of_u64() {
        let value = 0xdeadbeef_cafebabe_u64;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_u64(value);
            let restored = endianness.read_u64(&bytes);
            assert_eq!(
                restored, value,
                "read_u64 round-trip failed for {endianness:?}"
            );
        }
    }

    /// `read_u128` recovers the value written by `bytes_of_u128`.
    #[test]
    fn read_u128_round_trips_with_bytes_of_u128() {
        let value = 0xdeadbeef_cafebabe_01020304_05060708_u128;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_u128(value);
            let restored = endianness.read_u128(&bytes);
            assert_eq!(
                restored, value,
                "read_u128 round-trip failed for {endianness:?}"
            );
        }
    }

    // bytes_of_ptr_sized - endianness × PointerSize interaction

    /// `Bit8` encodes to a single byte regardless of endianness.
    #[test]
    fn bytes_of_ptr_sized_bit8_is_always_one_byte() {
        let value = 0xAB;
        for endianness in [Endianness::Little, Endianness::Big] {
            let bytes = endianness.bytes_of_ptr_sized(value, PointerSize::Bit8);
            assert_eq!(bytes, vec![0xAB], "Bit8 differs for {endianness:?}");
        }
    }

    /// `Bit16` orders its two bytes by endianness.
    #[test]
    fn bytes_of_ptr_sized_bit16_depends_on_endianness() {
        let value = 0x0102;
        assert_eq!(
            Endianness::Little.bytes_of_ptr_sized(value, PointerSize::Bit16),
            vec![0x02, 0x01],
        );
        assert_eq!(
            Endianness::Big.bytes_of_ptr_sized(value, PointerSize::Bit16),
            vec![0x01, 0x02],
        );
    }

    /// `Bit32` orders its four bytes by endianness.
    #[test]
    fn bytes_of_ptr_sized_bit32_depends_on_endianness() {
        let value = 0x01020304;
        assert_eq!(
            Endianness::Little.bytes_of_ptr_sized(value, PointerSize::Bit32),
            vec![0x04, 0x03, 0x02, 0x01],
        );
        assert_eq!(
            Endianness::Big.bytes_of_ptr_sized(value, PointerSize::Bit32),
            vec![0x01, 0x02, 0x03, 0x04],
        );
    }

    /// `Bit64` orders its eight bytes by endianness.
    #[test]
    fn bytes_of_ptr_sized_bit64_depends_on_endianness() {
        let value = 0x0102030405060708;
        assert_eq!(
            Endianness::Little.bytes_of_ptr_sized(value, PointerSize::Bit64),
            vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
        );
        assert_eq!(
            Endianness::Big.bytes_of_ptr_sized(value, PointerSize::Bit64),
            vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        );
    }

    /// `Bit128` zero-extends a 64-bit value and orders all 16 bytes by endianness.
    #[test]
    fn bytes_of_ptr_sized_bit128_depends_on_endianness() {
        let value = 0x0102030405060708; // only 64-bit of value, zero-extended
        let le = Endianness::Little.bytes_of_ptr_sized(value, PointerSize::Bit128);
        let be = Endianness::Big.bytes_of_ptr_sized(value, PointerSize::Bit128);

        // LE: low bytes first
        assert_eq!(le[0..8], [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
        assert_eq!(&le[8..16], &[0, 0, 0, 0, 0, 0, 0, 0]);

        // BE: high bytes first
        assert_eq!(&be[0..8], &[0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(be[8..16], [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    }

    // read_ptr_sized - round-trip with bytes_of_ptr_sized for all combos

    /// `read_ptr_sized` inverts `bytes_of_ptr_sized` for every pointer size.
    #[test]
    fn read_ptr_sized_round_trips_for_all_pointer_sizes() {
        let test_vectors: Vec<(PointerSize, u64)> = vec![
            (PointerSize::Bit8, 0xAB),
            (PointerSize::Bit16, 0xABCD),
            (PointerSize::Bit32, 0xDEAD_BEEF),
            (PointerSize::Bit64, 0xDEAD_BEEF_CAFE_BABE),
            (PointerSize::Bit128, 0xDEAD_BEEF_CAFE_BABE),
        ];

        for endianness in [Endianness::Little, Endianness::Big] {
            for (ptr_size, value) in &test_vectors {
                let bytes = endianness.bytes_of_ptr_sized(*value, *ptr_size);
                let restored = endianness.read_ptr_sized(&bytes, *ptr_size);
                assert_eq!(
                    restored, *value,
                    "round-trip failed for {endianness:?} × {ptr_size:?}",
                );
            }
        }
    }

    // LE and BE produce different byte sequences for multi-byte widths

    /// Little- and big-endian encodings differ for every multi-byte width.
    #[test]
    fn le_and_be_differ_for_all_multi_byte_sizes() {
        let value = 0x0102030405060708;
        for ptr_size in [
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            let le_bytes = Endianness::Little.bytes_of_ptr_sized(value, ptr_size);
            let be_bytes = Endianness::Big.bytes_of_ptr_sized(value, ptr_size);
            assert_ne!(
                le_bytes, be_bytes,
                "LE and BE should differ for {ptr_size:?}"
            );
        }
    }

    /// Single-byte encodings are identical across endianness.
    #[test]
    fn le_and_be_agree_for_bit8() {
        let le = Endianness::Little.bytes_of_ptr_sized(0xAB, PointerSize::Bit8);
        let be = Endianness::Big.bytes_of_ptr_sized(0xAB, PointerSize::Bit8);
        assert_eq!(le, be, "Bit8 should be identical regardless of endianness");
    }
}
