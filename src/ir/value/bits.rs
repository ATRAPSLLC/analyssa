//! Bit operations of constants: the logical operations, counts, swaps and
//! reversals, and shifts and rotates.

use crate::{
    PointerSize,
    ir::{
        float::FloatRules,
        integer::{BitWidth, shift_distance},
        lanes::VectorConst,
        ops::{VectorBinaryKind, VectorElement, VectorElementKind, VectorUnaryKind},
        value::{ConstValue, RegisterBitCount},
    },
    target::Target,
};

impl<T: Target> ConstValue<T> {
    /// Attempts to negate this constant.
    #[must_use]
    pub fn negate(&self, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) {
            return Self::of_lane(
                ptr_size.bits(),
                self.scalar_integer_pattern(ptr_size)?.0.wrapping_neg(),
            );
        }
        if let Self::Vector(a) = self {
            let element = VectorElement {
                kind: VectorElementKind::Integer,
                ..VectorElement::default()
            };
            return a
                .unary(VectorUnaryKind::Neg, element, None, FloatRules::default())
                .map(Self::Vector);
        }
        match self {
            Self::I8(v) => Some(Self::I8(v.wrapping_neg())),
            Self::I16(v) => Some(Self::I16(v.wrapping_neg())),
            Self::I32(v) => Some(Self::I32(v.wrapping_neg())),
            Self::I64(v) => Some(Self::I64(v.wrapping_neg())),
            Self::NativeInt(v) => Some(Self::NativeInt(v.wrapping_neg())),
            Self::Float(v) => Some(Self::Float(v.negate())),
            // Unsigned negation wraps
            Self::U8(v) => Some(Self::U8(v.wrapping_neg())),
            Self::U16(v) => Some(Self::U16(v.wrapping_neg())),
            Self::U32(v) => Some(Self::U32(v.wrapping_neg())),
            Self::U64(v) => Some(Self::U64(v.wrapping_neg())),
            Self::NativeUInt(v) => Some(Self::NativeUInt(v.wrapping_neg())),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to perform bitwise NOT on this constant.
    #[must_use]
    pub fn bitwise_not(&self, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) {
            return Self::of_lane(ptr_size.bits(), !self.scalar_integer_pattern(ptr_size)?.0);
        }
        if let Self::Vector(a) = self {
            return Some(Self::Vector(a.not()));
        }
        match self {
            Self::True => Some(Self::False),
            Self::False => Some(Self::True),
            Self::I8(v) => Some(Self::I8(!v)),
            Self::I16(v) => Some(Self::I16(!v)),
            Self::I32(v) => Some(Self::I32(!v)),
            Self::I64(v) => Some(Self::I64(!v)),
            Self::U8(v) => Some(Self::U8(!v)),
            Self::U16(v) => Some(Self::U16(!v)),
            Self::U32(v) => Some(Self::U32(!v)),
            Self::U64(v) => Some(Self::U64(!v)),
            Self::NativeInt(v) => Some(Self::NativeInt(!v)),
            Self::NativeUInt(v) => Some(Self::NativeUInt(!v)),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to perform bitwise AND on two constants.
    #[must_use]
    pub fn bitwise_and(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) || other.is_wide_native(ptr_size) {
            return self.fold_wide_integer_bits(other, ptr_size, |l, r| l & r);
        }
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            return a.and(b).map(Self::Vector);
        }
        if let Some((a, b)) = self.boolean_pair(other) {
            return Some(Self::from_bool(a & b));
        }
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a & b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a & b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a & b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a & b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a & b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a & b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a & b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a & b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a & b)),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::NativeUInt(a & b)),
            // Cross-type: promote to i64 for mixed signed operations
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a) & b))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a) & b))
            }
            _ => self.fold_integer_bits(other, ptr_size, |l, r| l & r, |l, r| l & r),
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to perform bitwise OR on two constants.
    #[must_use]
    pub fn bitwise_or(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) || other.is_wide_native(ptr_size) {
            return self.fold_wide_integer_bits(other, ptr_size, |l, r| l | r);
        }
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            return a.or(b).map(Self::Vector);
        }
        if let Some((a, b)) = self.boolean_pair(other) {
            return Some(Self::from_bool(a | b));
        }
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a | b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a | b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a | b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a | b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a | b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a | b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a | b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a | b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a | b)),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::NativeUInt(a | b)),
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a) | b))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a) | b))
            }
            _ => self.fold_integer_bits(other, ptr_size, |l, r| l | r, |l, r| l | r),
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to perform bitwise XOR on two constants.
    #[must_use]
    pub fn bitwise_xor(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) || other.is_wide_native(ptr_size) {
            return self.fold_wide_integer_bits(other, ptr_size, |l, r| l ^ r);
        }
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            return a.xor(b).map(Self::Vector);
        }
        if let Some((a, b)) = self.boolean_pair(other) {
            return Some(Self::from_bool(a ^ b));
        }
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a ^ b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a ^ b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a ^ b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a ^ b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a ^ b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a ^ b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a ^ b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a ^ b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a ^ b)),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::NativeUInt(a ^ b)),
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a) ^ b))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a) ^ b))
            }
            _ => self.fold_integer_bits(other, ptr_size, |l, r| l ^ r, |l, r| l ^ r),
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Returns the truth values of two boolean constants, or `None` unless
    /// both are `True`/`False`. The bitwise operations act on booleans as on
    /// one-bit values; a boolean against an integer has no common width and
    /// does not fold.
    fn boolean_pair(&self, other: &Self) -> Option<(bool, bool)> {
        match (self, other) {
            (Self::True | Self::False, Self::True | Self::False) => {
                Some((matches!(self, Self::True), matches!(other, Self::True)))
            }
            _ => None,
        }
    }

    /// Counts a register's bits using native intrinsics for primitive carriers.
    #[inline]
    pub fn bit_count(&self, kind: RegisterBitCount, ptr_size: PointerSize) -> Option<u32> {
        let pattern = match self {
            Self::I8(value) => u64::from(value.cast_unsigned()),
            Self::U8(value) => u64::from(*value),
            Self::I16(value) => u64::from(value.cast_unsigned()),
            Self::U16(value) => u64::from(*value),
            Self::I32(value) => u64::from(value.cast_unsigned()),
            Self::U32(value) => u64::from(*value),
            Self::I64(value) => value.cast_unsigned(),
            Self::U64(value) => *value,
            Self::NativeInt(value) if ptr_size.bits() <= 64 => {
                ptr_size.mask_unsigned(value.cast_unsigned())
            }
            Self::NativeUInt(value) if ptr_size.bits() <= 64 => ptr_size.mask_unsigned(*value),
            _ => return kind.wide(self.register_pattern(ptr_size)?.0),
        };
        kind.narrow(pattern)
    }

    /// Swaps a scalar register's bytes, retaining primitive carriers on the fast path.
    #[inline(always)]
    pub fn byte_swap(&self, ptr_size: PointerSize) -> Option<Self> {
        Some(match self {
            Self::I8(value) => Self::I8(*value),
            Self::U8(value) => Self::U8(*value),
            Self::I16(value) => Self::I16(value.swap_bytes()),
            Self::U16(value) => Self::U16(value.swap_bytes()),
            Self::I32(value) => Self::I32(value.swap_bytes()),
            Self::U32(value) => Self::U32(value.swap_bytes()),
            Self::I64(value) => Self::I64(value.swap_bytes()),
            Self::U64(value) => Self::U64(value.swap_bytes()),
            _ => return self.permute_register_bits(ptr_size, true),
        })
    }

    /// Reverses a scalar register's bits, retaining primitive carriers on the fast path.
    #[inline(always)]
    pub fn bit_reverse(&self, ptr_size: PointerSize) -> Option<Self> {
        Some(match self {
            Self::I8(value) => Self::I8(value.reverse_bits()),
            Self::U8(value) => Self::U8(value.reverse_bits()),
            Self::I16(value) => Self::I16(value.reverse_bits()),
            Self::U16(value) => Self::U16(value.reverse_bits()),
            Self::I32(value) => Self::I32(value.reverse_bits()),
            Self::U32(value) => Self::U32(value.reverse_bits()),
            Self::I64(value) => Self::I64(value.reverse_bits()),
            Self::U64(value) => Self::U64(value.reverse_bits()),
            _ => return self.permute_register_bits(ptr_size, false),
        })
    }

    /// Permutes packed and native register bits at their complete target width.
    fn permute_register_bits(&self, ptr_size: PointerSize, bytes: bool) -> Option<Self> {
        let (pattern, width) = self.register_pattern(ptr_size)?;
        if bytes && !width.is_multiple_of(8) {
            return None;
        }
        let result = if width <= 64 && !matches!(self, Self::Vector(_)) {
            let pattern = pattern as u64;
            let reversed = if bytes {
                pattern.swap_bytes()
            } else {
                pattern.reverse_bits()
            };
            u128::from(reversed.checked_shr(64u32.checked_sub(width)?)?)
        } else {
            let reversed = if bytes {
                pattern.swap_bytes()
            } else {
                pattern.reverse_bits()
            };
            reversed.checked_shr(128u32.checked_sub(width)?)?
        };
        match self {
            Self::True | Self::False => Some(Self::from_bool(result != 0)),
            Self::Vector(_) => VectorConst::from_lanes(width, [result]).map(Self::Vector),
            _ if width <= 64 => self.with_integer_bits(result as u64, ptr_size),
            _ => Self::of_lane(width, result),
        }
    }

    /// Returns this value read as a shift count, if it is representable and in
    /// range for a value of `bits` width, or `None` to decline the fold.
    ///
    /// `SsaOp::Shl`/`Shr` document only `dest = value << amount` and define no
    /// out-of-range behaviour, and real ISAs disagree about it - x86 masks the
    /// count by 31 (63 with REX.W) *regardless of the destination's sub-register
    /// width*, while AArch64 and RISC-V mask by the register width. Rust's
    /// `wrapping_shl` masks by the value's own type, so an `I8` arm masked by 7:
    /// `shl bl, 8` is 0 on hardware, but folding `U8(0x81) << 8` returned
    /// `U8(0x81)` - a third semantics that matches neither.
    ///
    /// Declining to fold is sound under every convention, so an out-of-range or
    /// negative count is simply not folded.
    fn shift_count(&self, bits: u32, ptr_size: PointerSize) -> Option<u32> {
        let distance = match self {
            Self::NativeInt(value) => u32::try_from(ptr_size.mask_signed(*value)).ok()?,
            Self::NativeUInt(value) => u32::try_from(ptr_size.mask_unsigned(*value)).ok()?,
            Self::Vector(_) => u32::try_from(self.scalar_integer_pattern(ptr_size)?.0).ok()?,
            _ => u32::try_from(self.as_u64()?).ok()?,
        };
        shift_distance(u128::from(distance), bits)
    }

    /// Attempts to shift left.
    ///
    /// Returns `None` when the shift distance is negative or at least the
    /// value's own width - see `shift_count` for why declining is the
    /// only convention-independent answer.
    #[must_use]
    pub fn shl(&self, amount: &Self, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) {
            let shift = amount.shift_count(ptr_size.bits(), ptr_size)?;
            return Self::of_lane(
                ptr_size.bits(),
                self.scalar_integer_pattern(ptr_size)?.0.wrapping_shl(shift),
            );
        }
        if let Some((a, b)) = self.lanes_and_count(amount, ptr_size, false) {
            return Self::shifted_lanes(&a, &b, VectorBinaryKind::Shl, None);
        }
        let shift = amount.shift_count(self.bit_width(ptr_size)?, ptr_size)?;
        match self {
            Self::I8(v) => Some(Self::I8(v.wrapping_shl(shift))),
            Self::I16(v) => Some(Self::I16(v.wrapping_shl(shift))),
            Self::I32(v) => Some(Self::I32(v.wrapping_shl(shift))),
            Self::I64(v) => Some(Self::I64(v.wrapping_shl(shift))),
            Self::U8(v) => Some(Self::U8(v.wrapping_shl(shift))),
            Self::U16(v) => Some(Self::U16(v.wrapping_shl(shift))),
            Self::U32(v) => Some(Self::U32(v.wrapping_shl(shift))),
            Self::U64(v) => Some(Self::U64(v.wrapping_shl(shift))),
            Self::NativeInt(v) => Some(Self::NativeInt(v.wrapping_shl(shift))),
            Self::NativeUInt(v) => Some(Self::NativeUInt(v.wrapping_shl(shift))),
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Rotates the integer left, or right when `right`, within its own width
    /// (a native integer's being the pointer's), by the amount taken modulo
    /// that width: a rotate instruction reads only the low bits of its count,
    /// so rotating a 64-bit value by 65 rotates it by 1, and the count's own
    /// variant does not matter. Vectors rotate lane by lane.
    ///
    /// # Arguments
    ///
    /// * `amount` - How far to rotate.
    /// * `right` - Rotate toward the low bits rather than the high ones.
    /// * `ptr_size` - Target pointer size, the width of the native integers.
    ///
    /// # Returns
    ///
    /// The rotated value, retaining this carrier where it represents the
    /// complete width and using one scalar lane for wider native results.
    /// Returns `None` when either operand is not an integer.
    #[must_use]
    pub fn rotate(&self, amount: &Self, right: bool, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) {
            let (bits, width) = self.scalar_integer_pattern(ptr_size)?;
            let distance = u32::try_from(
                amount
                    .scalar_integer_pattern(ptr_size)?
                    .0
                    .checked_rem(u128::from(width))?,
            )
            .ok()?;
            return Self::of_lane(
                width,
                if right {
                    bits.rotate_right(distance)
                } else {
                    bits.rotate_left(distance)
                },
            );
        }
        if let Some((a, b)) = self.lanes_and_count(amount, ptr_size, true) {
            let kind = if right {
                VectorBinaryKind::Ror
            } else {
                VectorBinaryKind::Rol
            };
            return Self::shifted_lanes(&a, &b, kind, None);
        }
        let (bits, width) = self.integer_extended(ptr_size)?;
        // Primitive register widths are powers of two at most 64 bits, so
        // count bits above 63 cannot change the modulo. Scalar lanes use the
        // full count in lanes_and_count before this path.
        let count = match amount {
            Self::Vector(_) => amount.scalar_integer_pattern(ptr_size)?.0 as u64,
            _ => amount.integer_extended(ptr_size)?.0,
        };
        let distance = u32::try_from(count.checked_rem(u64::from(width))?).ok()?;
        let raw = BitWidth::new(width).low_bits(bits);
        let back = width.checked_sub(distance)?;
        let rotated = match (distance, right) {
            (0, _) => raw,
            (_, false) => raw.wrapping_shl(distance) | raw.wrapping_shr(back),
            (_, true) => raw.wrapping_shr(distance) | raw.wrapping_shl(back),
        };
        self.with_integer_bits(BitWidth::new(width).low_bits(rotated), ptr_size)
    }

    /// Attempts to shift right: logical when `unsigned`, arithmetic otherwise.
    ///
    /// The flag decides, not the constant's type: the operation states how its
    /// operand is read, and an arithmetic shift of a value that happens to be
    /// typed unsigned still copies the top bit down.
    ///
    /// Returns `None` when the shift distance is negative or at least the
    /// value's own width - see `shift_count`.
    #[must_use]
    #[allow(clippy::cast_sign_loss)] // Shift amounts and unsigned shifts use intentional casts
    #[allow(clippy::cast_possible_wrap)] // Wrapping is expected for logical shift operations
    pub fn shr(&self, amount: &Self, unsigned: bool, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) {
            let shift = amount.shift_count(ptr_size.bits(), ptr_size)?;
            let bits = self.scalar_integer_pattern(ptr_size)?.0;
            let result = if unsigned {
                bits >> shift
            } else {
                (bits.cast_signed() >> shift).cast_unsigned()
            };
            return Self::of_lane(ptr_size.bits(), result);
        }
        if let Some((a, b)) = self.lanes_and_count(amount, ptr_size, false) {
            return Self::shifted_lanes(&a, &b, VectorBinaryKind::Shr, Some(!unsigned));
        }
        let shift = amount.shift_count(self.bit_width(ptr_size)?, ptr_size)?;
        match self {
            Self::I8(v) => {
                if unsigned {
                    Some(Self::I8((*v as u8).wrapping_shr(shift) as i8))
                } else {
                    Some(Self::I8(v.wrapping_shr(shift)))
                }
            }
            Self::I16(v) => {
                if unsigned {
                    Some(Self::I16((*v as u16).wrapping_shr(shift) as i16))
                } else {
                    Some(Self::I16(v.wrapping_shr(shift)))
                }
            }
            Self::I32(v) => {
                if unsigned {
                    Some(Self::I32((*v as u32).wrapping_shr(shift) as i32))
                } else {
                    Some(Self::I32(v.wrapping_shr(shift)))
                }
            }
            Self::I64(v) => {
                if unsigned {
                    Some(Self::I64((*v as u64).wrapping_shr(shift) as i64))
                } else {
                    Some(Self::I64(v.wrapping_shr(shift)))
                }
            }
            Self::U8(v) => {
                if unsigned {
                    Some(Self::U8(v.wrapping_shr(shift)))
                } else {
                    Some(Self::U8((*v as i8).wrapping_shr(shift) as u8))
                }
            }
            Self::U16(v) => {
                if unsigned {
                    Some(Self::U16(v.wrapping_shr(shift)))
                } else {
                    Some(Self::U16((*v as i16).wrapping_shr(shift) as u16))
                }
            }
            Self::U32(v) => {
                if unsigned {
                    Some(Self::U32(v.wrapping_shr(shift)))
                } else {
                    Some(Self::U32((*v as i32).wrapping_shr(shift) as u32))
                }
            }
            Self::U64(v) => {
                if unsigned {
                    Some(Self::U64(v.wrapping_shr(shift)))
                } else {
                    Some(Self::U64((*v as i64).wrapping_shr(shift) as u64))
                }
            }
            Self::NativeInt(v) => {
                if unsigned {
                    let raw = ptr_size.mask_unsigned(v.cast_unsigned());
                    Some(Self::NativeInt(raw.wrapping_shr(shift).cast_signed()))
                } else {
                    Some(Self::NativeInt(
                        ptr_size.mask_signed(*v).wrapping_shr(shift),
                    ))
                }
            }
            Self::NativeUInt(v) => {
                if unsigned {
                    Some(Self::NativeUInt(
                        ptr_size.mask_unsigned(*v).wrapping_shr(shift),
                    ))
                } else {
                    let signed = ptr_size.mask_signed(*v as i64);
                    Some(Self::NativeUInt(signed.wrapping_shr(shift) as u64))
                }
            }
            _ => None,
        }
        .map(|v| v.mask_native(ptr_size))
    }
}
