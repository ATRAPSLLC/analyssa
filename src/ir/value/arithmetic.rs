//! Arithmetic of constants: sums, products, quotients and remainders, checked and
//! wrapping, at a width or double it.

use crate::{
    PointerSize,
    ir::{
        division::{DivisionFailure, DivisionRules},
        evaluation::{ArithmeticTrap, Evaluation},
        integer::{BitWidth, IntegerType},
        lanes::VectorConst,
        ops::VectorBinaryKind,
        value::{CheckedArithmetic, ConstValue},
    },
    target::Target,
};

impl<T: Target> ConstValue<T> {
    /// Attempts to add two constants.
    #[must_use]
    pub fn add(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) || other.is_wide_native(ptr_size) {
            return self.fold_wide_integer_bits(other, ptr_size, u128::wrapping_add);
        }
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            return Self::integer_lanes(a, b, VectorBinaryKind::Add, None);
        }
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a.wrapping_add(*b))),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a.wrapping_add(*b))),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a.wrapping_add(*b))),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a.wrapping_add(*b))),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a.wrapping_add(*b))),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a.wrapping_add(*b))),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a.wrapping_add(*b))),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a.wrapping_add(*b))),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a.wrapping_add(*b))),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => {
                Some(Self::NativeUInt(a.wrapping_add(*b)))
            }
            (Self::Float(a), Self::Float(b)) => a
                .checked_add(*b, T::environment_rounding())
                .map(Self::Float),
            // Cross-type promotions
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a).wrapping_add(*b)))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a).wrapping_add(*b)))
            }
            _ => self.fold_integer_bits(other, ptr_size, u64::wrapping_add, u128::wrapping_add),
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to subtract two constants.
    #[must_use]
    pub fn sub(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) || other.is_wide_native(ptr_size) {
            return self.fold_wide_integer_bits(other, ptr_size, u128::wrapping_sub);
        }
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            return Self::integer_lanes(a, b, VectorBinaryKind::Sub, None);
        }
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a.wrapping_sub(*b))),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a.wrapping_sub(*b))),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a.wrapping_sub(*b))),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a.wrapping_sub(*b))),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a.wrapping_sub(*b))),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a.wrapping_sub(*b))),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a.wrapping_sub(*b))),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a.wrapping_sub(*b))),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a.wrapping_sub(*b))),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => {
                Some(Self::NativeUInt(a.wrapping_sub(*b)))
            }
            (Self::Float(a), Self::Float(b)) => a
                .checked_sub(*b, T::environment_rounding())
                .map(Self::Float),
            (Self::I32(a), Self::I64(b)) => Some(Self::I64(i64::from(*a).wrapping_sub(*b))),
            (Self::I64(a), Self::I32(b)) => Some(Self::I64(a.wrapping_sub(i64::from(*b)))),
            (Self::U32(a), Self::U64(b)) => Some(Self::U64(u64::from(*a).wrapping_sub(*b))),
            (Self::U64(a), Self::U32(b)) => Some(Self::U64(a.wrapping_sub(u64::from(*b)))),
            _ => self.fold_integer_bits(other, ptr_size, u64::wrapping_sub, u128::wrapping_sub),
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Attempts to multiply two constants.
    #[must_use]
    pub fn mul(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if self.is_wide_native(ptr_size) || other.is_wide_native(ptr_size) {
            return self.fold_wide_integer_bits(other, ptr_size, u128::wrapping_mul);
        }
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            return Self::integer_lanes(a, b, VectorBinaryKind::Mul, None);
        }
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::I8(a.wrapping_mul(*b))),
            (Self::I16(a), Self::I16(b)) => Some(Self::I16(a.wrapping_mul(*b))),
            (Self::I32(a), Self::I32(b)) => Some(Self::I32(a.wrapping_mul(*b))),
            (Self::I64(a), Self::I64(b)) => Some(Self::I64(a.wrapping_mul(*b))),
            (Self::U8(a), Self::U8(b)) => Some(Self::U8(a.wrapping_mul(*b))),
            (Self::U16(a), Self::U16(b)) => Some(Self::U16(a.wrapping_mul(*b))),
            (Self::U32(a), Self::U32(b)) => Some(Self::U32(a.wrapping_mul(*b))),
            (Self::U64(a), Self::U64(b)) => Some(Self::U64(a.wrapping_mul(*b))),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::NativeInt(a.wrapping_mul(*b))),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => {
                Some(Self::NativeUInt(a.wrapping_mul(*b)))
            }
            (Self::Float(a), Self::Float(b)) => a
                .checked_mul(*b, T::environment_rounding())
                .map(Self::Float),
            (Self::I32(a), Self::I64(b)) | (Self::I64(b), Self::I32(a)) => {
                Some(Self::I64(i64::from(*a).wrapping_mul(*b)))
            }
            (Self::U32(a), Self::U64(b)) | (Self::U64(b), Self::U32(a)) => {
                Some(Self::U64(u64::from(*a).wrapping_mul(*b)))
            }
            _ => self.fold_integer_bits(other, ptr_size, u64::wrapping_mul, u128::wrapping_mul),
        }
        .map(|v| v.mask_native(ptr_size))
    }

    /// Adds integer constants with overflow checking at their common width.
    ///
    /// Signedness is selected by `unsigned`, independently of the constant
    /// variants. Returns `None` on overflow, unequal widths or non-integers.
    #[must_use]
    pub fn add_checked(&self, other: &Self, unsigned: bool, ptr_size: PointerSize) -> Option<Self> {
        self.checked_arithmetic(other, unsigned, ptr_size, CheckedArithmetic::Add)
    }

    /// Subtracts integer constants with overflow checking at their common width.
    ///
    /// Signedness is selected by `unsigned`, independently of the constant
    /// variants. Returns `None` on overflow, unequal widths or non-integers.
    #[must_use]
    pub fn sub_checked(&self, other: &Self, unsigned: bool, ptr_size: PointerSize) -> Option<Self> {
        self.checked_arithmetic(other, unsigned, ptr_size, CheckedArithmetic::Sub)
    }

    /// Multiplies integer constants with overflow checking at their common width.
    ///
    /// Signedness is selected by `unsigned`, independently of the constant
    /// variants. Returns `None` on overflow, unequal widths or non-integers.
    #[must_use]
    pub fn mul_checked(&self, other: &Self, unsigned: bool, ptr_size: PointerSize) -> Option<Self> {
        self.checked_arithmetic(other, unsigned, ptr_size, CheckedArithmetic::Mul)
    }

    /// Computes checked arithmetic before reconstructing the operand's representation.
    pub fn checked_arithmetic(
        &self,
        other: &Self,
        unsigned: bool,
        ptr_size: PointerSize,
        op: CheckedArithmetic,
    ) -> Option<Self> {
        let (left, width) = self.scalar_integer_pattern(ptr_size)?;
        let (right, right_width) = other.scalar_integer_pattern(ptr_size)?;
        if width != right_width {
            return None;
        }
        let shift = 128u32.checked_sub(width)?;
        let mask = BitWidth::new(width).checked_mask_u128()?;
        let result = if unsigned {
            match op {
                CheckedArithmetic::Add => left.checked_add(right),
                CheckedArithmetic::Sub => left.checked_sub(right),
                CheckedArithmetic::Mul => left.checked_mul(right),
            }
            .filter(|value| *value <= mask)?
        } else {
            let left = (left << shift).cast_signed() >> shift;
            let right = (right << shift).cast_signed() >> shift;
            let result = match op {
                CheckedArithmetic::Add => left.checked_add(right),
                CheckedArithmetic::Sub => left.checked_sub(right),
                CheckedArithmetic::Mul => left.checked_mul(right),
            }?;
            let max = (mask >> 1).cast_signed();
            if result < !max || result > max {
                return None;
            }
            result.cast_unsigned() & mask
        };
        if width > 64 || matches!(self, Self::Vector(_)) {
            VectorConst::from_lanes(width, [result]).map(Self::Vector)
        } else {
            self.with_integer_bits(result as u64, ptr_size)
        }
    }

    /// Divides two constants, signed: integers are read as signed values of
    /// their shared width, whatever their variants, as CIL's `div` and a native
    /// signed division read them, so `U32(0xFFFF_FFF8) / U32(3)` is `-8 / 3 =
    /// -2`. Scalar quotient overflow (the width's minimum over -1), a zero
    /// divisor, different operand widths and non-numbers answer `None`;
    /// floats divide by IEEE 754 rules.
    ///
    /// A scalar result retains the left carrier where it can represent the
    /// full register width; wider native results use one scalar lane.
    /// Multiple vector lanes use the explicit per-lane division semantics.
    ///
    /// # Arguments
    ///
    /// * `other` - The divisor.
    /// * `ptr_size` - Target pointer size, giving the native variants' width.
    ///
    /// # Returns
    ///
    /// The quotient, or `None` when it is not defined here.
    #[must_use]
    pub fn div(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            if a.lanes() == 1 {
                return self.wide_scalar_division(other, ptr_size, false, false);
            }
            return Self::integer_lanes(a, b, VectorBinaryKind::Div, Some(true));
        }
        match (self, other) {
            // Float div by zero is inf - IEEE 754 has no panic, no overflow.
            (Self::Float(a), Self::Float(b)) => a
                .checked_div(*b, T::environment_rounding())
                .map(Self::Float),
            _ => self.signed_division(other, ptr_size, false),
        }
    }

    /// Separates exact scalar division values, integer faults and unsupported carriers.
    pub fn division_outcome(
        &self,
        other: &Self,
        unsigned: bool,
        remainder: bool,
        ptr_size: PointerSize,
        rules: DivisionRules,
    ) -> Evaluation<Self> {
        let result = match (unsigned, remainder) {
            (false, false) => self.div(other, ptr_size),
            (false, true) => self.rem(other, ptr_size),
            (true, false) => self.div_un(other, ptr_size),
            (true, true) => self.rem_un(other, ptr_size),
        };
        if let Some(value) = result {
            return Evaluation::Value(value);
        }
        match (
            self.scalar_integer_pattern(ptr_size),
            other.scalar_integer_pattern(ptr_size),
        ) {
            (Some((_, width)), Some((0, other_width))) if width == other_width => self
                .division_failure(
                    if remainder {
                        rules.remainder_zero
                    } else {
                        rules.quotient_zero
                    },
                    ArithmeticTrap::DivideByZero,
                    ptr_size,
                ),
            (Some((left, width)), Some((right, other_width)))
                if !unsigned
                    && width == other_width
                    && left == 1u128.wrapping_shl(width.saturating_sub(1))
                    && right == BitWidth::new(width).mask_u128() =>
            {
                self.division_failure(
                    if remainder {
                        rules.remainder_overflow
                    } else {
                        rules.quotient_overflow
                    },
                    ArithmeticTrap::Overflow,
                    ptr_size,
                )
            }
            _ => Evaluation::Unsupported,
        }
    }

    /// Applies an exceptional result in its attested scalar width and representation.
    fn division_failure(
        &self,
        failure: DivisionFailure,
        trap: ArithmeticTrap,
        ptr_size: PointerSize,
    ) -> Evaluation<Self> {
        let Some((_, width)) = self.scalar_integer_pattern(ptr_size) else {
            return Evaluation::Unsupported;
        };
        let Some(mut layout) = IntegerType::new(width, self.is_signed()) else {
            return Evaluation::Unsupported;
        };
        if matches!(self, Self::NativeInt(_) | Self::NativeUInt(_)) {
            layout = layout.native();
        }
        failure.evaluate(Some(self), layout, trap, ptr_size)
    }

    /// Divides two integer constants as unsigned values of their own width.
    ///
    /// The unsigned counterpart of [`Self::div`], for `div.un` and `udiv`: both
    /// operands' bit patterns are read as unsigned at the operands' width, so
    /// `I32(-8) / I32(2)` is `0xFFFF_FFF8 / 2 = 0x7FFF_FFFC`, not `-4`. The
    /// result keeps the left carrier where it represents the complete width;
    /// wider native results use one scalar lane. Operands of different widths,
    /// non-integers, and a zero divisor answer `None`.
    ///
    /// # Arguments
    ///
    /// * `other` - The divisor.
    /// * `ptr_size` - Target pointer size, giving the native variants' width.
    ///
    /// # Returns
    ///
    /// The quotient, or `None` when it is not defined here.
    #[must_use]
    pub fn div_un(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            if a.lanes() == 1 {
                return self.wide_scalar_division(other, ptr_size, true, false);
            }
            return Self::integer_lanes(a, b, VectorBinaryKind::Div, Some(false));
        }
        self.unsigned_integer_division::<false>(other, ptr_size)
    }

    /// Computes the remainder of two integer constants as unsigned values of
    /// their own width.
    ///
    /// The unsigned counterpart of [`Self::rem`]; see [`Self::div_un`] for how
    /// the operands are read.
    ///
    /// # Arguments
    ///
    /// * `other` - The divisor.
    /// * `ptr_size` - Target pointer size, giving the native variants' width.
    ///
    /// # Returns
    ///
    /// The remainder, or `None` when it is not defined here.
    #[must_use]
    pub fn rem_un(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            if a.lanes() == 1 {
                return self.wide_scalar_division(other, ptr_size, true, true);
            }
            return Self::integer_lanes(a, b, VectorBinaryKind::Rem, Some(false));
        }
        self.unsigned_integer_division::<true>(other, ptr_size)
    }

    /// Divides unsigned primitive patterns, wide carriers on the fallback path.
    fn unsigned_integer_division<const REMAINDER: bool>(
        &self,
        other: &Self,
        ptr_size: PointerSize,
    ) -> Option<Self> {
        let (Some((left, width)), Some((right, right_width))) = (
            self.integer_extended(ptr_size),
            other.integer_extended(ptr_size),
        ) else {
            return self.wide_scalar_division(other, ptr_size, true, REMAINDER);
        };
        if width != right_width || width > 64 {
            return self.wide_scalar_division(other, ptr_size, true, REMAINDER);
        }
        let mask = BitWidth::new(width).mask();
        let (left, right) = (left & mask, right & mask);
        let result = if REMAINDER {
            left.checked_rem(right)?
        } else {
            left.checked_div(right)?
        };
        self.with_integer_bits(result, ptr_size)
    }

    /// Computes the signed remainder of two constants; see [`Self::div`] for
    /// how the integers are read. The remainder of an overflowing division
    /// answers `None` with it; floats take the remainder of truncating
    /// division.
    ///
    /// # Arguments
    ///
    /// * `other` - The divisor.
    /// * `ptr_size` - Target pointer size, giving the native variants' width.
    ///
    /// # Returns
    ///
    /// The remainder, or `None` when it is not defined here.
    #[must_use]
    pub fn rem(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if let (Self::Vector(a), Self::Vector(b)) = (self, other) {
            if a.lanes() == 1 {
                return self.wide_scalar_division(other, ptr_size, false, true);
            }
            return Self::integer_lanes(a, b, VectorBinaryKind::Rem, Some(true));
        }
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => a.checked_rem(*b).map(Self::Float),
            _ => self.signed_division(other, ptr_size, true),
        }
    }

    /// Divides signed primitive patterns, wide carriers on the fallback path.
    ///
    /// The quotient must fit the shared register width, including when only
    /// the remainder is requested. A zero divisor or incompatible operands
    /// provide no constant result.
    fn signed_division(
        &self,
        other: &Self,
        ptr_size: PointerSize,
        remainder: bool,
    ) -> Option<Self> {
        let (Some((left, width)), Some((right, right_width))) = (
            self.integer_extended(ptr_size),
            other.integer_extended(ptr_size),
        ) else {
            return self.wide_scalar_division(other, ptr_size, false, remainder);
        };
        if width != right_width || width > 64 {
            return self.wide_scalar_division(other, ptr_size, false, remainder);
        }
        let dividend = Self::sign_extend(left, width);
        let divisor = Self::sign_extend(right, width);
        let quotient = dividend.checked_div(divisor)?;
        if Self::sign_extend(quotient.cast_unsigned(), width) != quotient {
            return None;
        }
        let result = if remainder {
            dividend.checked_rem(divisor)?
        } else {
            quotient
        };
        self.with_integer_bits(result.cast_unsigned(), ptr_size)
    }

    /// Divides complete scalar bitvectors without a 64-bit intermediate.
    fn wide_scalar_division(
        &self,
        other: &Self,
        ptr_size: PointerSize,
        unsigned: bool,
        remainder: bool,
    ) -> Option<Self> {
        let (left, width) = self.scalar_integer_pattern(ptr_size)?;
        let (right, other_width) = match other {
            Self::True | Self::False => return None,
            _ => other.register_pattern(ptr_size)?,
        };
        if width != other_width {
            return None;
        }
        let shift = 128u32.checked_sub(width)?;
        let result = if unsigned {
            if remainder {
                left.checked_rem(right)?
            } else {
                left.checked_div(right)?
            }
        } else {
            let left = (left << shift).cast_signed() >> shift;
            let right = (right << shift).cast_signed() >> shift;
            let quotient = left.checked_div(right)?;
            if (quotient.cast_unsigned() << shift).cast_signed() >> shift != quotient {
                return None;
            }
            if remainder {
                left.checked_rem(right)?.cast_unsigned()
            } else {
                quotient.cast_unsigned()
            }
        };
        match self {
            Self::Vector(_) => VectorConst::from_lanes(width, [result]).map(Self::Vector),
            _ if width <= 64 => self.with_integer_bits(result as u64, ptr_size),
            _ => Self::of_lane(width, result),
        }
    }

    /// Returns a scaled index and signed displacement at the target pointer width.
    pub fn scaled_offset(
        index: Option<&Self>,
        stride: u64,
        offset: i64,
        ptr_size: PointerSize,
    ) -> Option<u128> {
        let index = match index {
            Some(index) => index.scalar_pattern_at_width(ptr_size.bits(), ptr_size)?,
            None => 0,
        };
        Some(
            ptr_size.mask_unsigned_128(
                index
                    .wrapping_mul(u128::from(stride))
                    .wrapping_add(i128::from(offset).cast_unsigned()),
            ),
        )
    }

    /// Computes a numeric scaled pointer sum at the target width.
    ///
    /// Intermediate products and sums wrap at `ptr_size`. Symbolic addresses
    /// require the execution address domain; this method accepts scalar integers.
    /// Returns `None` for non-integer bases or indices.
    #[must_use]
    pub fn pointer_add(
        &self,
        index: Option<&Self>,
        stride: u64,
        offset: i64,
        ptr_size: PointerSize,
    ) -> Option<Self> {
        let base = self.scalar_pattern_at_width(ptr_size.bits(), ptr_size)?;
        let offset = Self::scaled_offset(index, stride, offset, ptr_size)?;
        Self::of_lane(
            ptr_size.bits(),
            ptr_size.mask_unsigned_128(base.wrapping_add(offset)),
        )
    }

    /// The double-width product of two integers of one width, as its low and
    /// high halves, each a constant of `self`'s type: what [`SsaOp::WideMul`]
    /// computes.
    ///
    /// The operands are read as signed or, when `unsigned`, unsigned integers
    /// of their width, whatever their variants say.
    ///
    /// # Arguments
    ///
    /// * `other` - The other factor, of `self`'s width.
    /// * `unsigned` - Whether the factors are unsigned.
    /// * `ptr_size` - The width of the native variants.
    ///
    /// # Returns
    ///
    /// `(low, high)`, or `None` when either is not an integer or the widths
    /// differ.
    ///
    /// [`SsaOp::WideMul`]: crate::ir::ops::SsaOp::WideMul
    #[must_use]
    pub fn wide_mul(
        &self,
        other: &Self,
        unsigned: bool,
        ptr_size: PointerSize,
    ) -> Option<(Self, Self)> {
        let bits = self.bit_width(ptr_size)?;
        if other.bit_width(ptr_size)? != bits {
            return None;
        }
        let read = |value: &Self| {
            let raw = BitWidth::new(bits).low_bits(value.as_u64_bits()?);
            Some(if unsigned {
                i128::from(raw)
            } else {
                i128::from(Self::sign_extend(raw, bits))
            })
        };
        // Two factors of at most 64 bits: the product fits 128 bits, signed
        // or not, so neither the multiplication nor the shift wraps.
        let product = read(self)?.wrapping_mul(read(other)?);
        let low = product.cast_unsigned() as u64;
        let high = product.wrapping_shr(bits).cast_unsigned() as u64;
        Some((
            self.of_same_type_from_bits(low, bits)?,
            self.of_same_type_from_bits(high, bits)?,
        ))
    }

    /// Separates supported wide division values from zero-divisor and quotient-overflow faults.
    pub fn wide_division_outcome(
        &self,
        high: &Self,
        divisor: &Self,
        unsigned: bool,
        ptr_size: PointerSize,
    ) -> Evaluation<(Self, Self)> {
        let (Some((_, width)), Some((_, high_width)), Some((divisor_bits, divisor_width))) = (
            self.integer_pattern(ptr_size),
            high.integer_pattern(ptr_size),
            divisor.integer_pattern(ptr_size),
        ) else {
            return Evaluation::Unsupported;
        };
        if !(1..=64).contains(&width) || width != high_width || width != divisor_width {
            return Evaluation::Unsupported;
        }
        if divisor_bits == 0 {
            return Evaluation::Trap(ArithmeticTrap::DivideByZero);
        }
        self.wide_div(high, divisor, unsigned, ptr_size).map_or(
            Evaluation::Trap(ArithmeticTrap::Overflow),
            Evaluation::Value,
        )
    }

    /// The quotient and remainder of a double-width dividend `high:low` by a
    /// divisor of one width, each a constant of `self`'s type: what
    /// [`SsaOp::WideDiv`] computes. `self` is the low half.
    ///
    /// Division truncates towards zero and the remainder takes the dividend's
    /// sign, as x86's `div` and `idiv` do. A zero divisor, and a quotient that
    /// does not fit the width (which traps), have no value.
    ///
    /// # Arguments
    ///
    /// * `high` - The high half of the dividend.
    /// * `divisor` - The divisor, of `self`'s width.
    /// * `unsigned` - Whether the dividend and divisor are unsigned.
    /// * `ptr_size` - The width of the native variants.
    ///
    /// # Returns
    ///
    /// `(quotient, remainder)`, or `None` as above or when the operands are
    /// not integers of one width.
    ///
    /// [`SsaOp::WideDiv`]: crate::ir::ops::SsaOp::WideDiv
    #[must_use]
    pub fn wide_div(
        &self,
        high: &Self,
        divisor: &Self,
        unsigned: bool,
        ptr_size: PointerSize,
    ) -> Option<(Self, Self)> {
        let bits = self.bit_width(ptr_size)?;
        if !(1..=64).contains(&bits)
            || high.bit_width(ptr_size)? != bits
            || divisor.bit_width(ptr_size)? != bits
        {
            return None;
        }
        let low = u128::from(BitWidth::new(bits).low_bits(self.as_u64_bits()?));
        let high = u128::from(BitWidth::new(bits).low_bits(high.as_u64_bits()?));
        let divisor = BitWidth::new(bits).low_bits(divisor.as_u64_bits()?);
        let dividend = high.wrapping_shl(bits) | low;
        let (quotient, remainder) = if unsigned {
            let divisor = u128::from(divisor);
            let quotient = dividend.checked_div(divisor)?;
            if quotient >> bits != 0 {
                return None;
            }
            (quotient as u64, dividend.checked_rem(divisor)? as u64)
        } else {
            // The dividend is a signed integer of twice the width: shifted to
            // the top of 128 bits and back, it sign-extends.
            let shift = 128u32.saturating_sub(bits.saturating_mul(2));
            let dividend = dividend
                .wrapping_shl(shift)
                .cast_signed()
                .wrapping_shr(shift);
            let divisor = i128::from(Self::sign_extend(divisor, bits));
            let quotient = dividend.checked_div(divisor)?;
            let limit = 1i128.wrapping_shl(bits.saturating_sub(1));
            if quotient < limit.wrapping_neg() || quotient >= limit {
                return None;
            }
            (
                quotient.cast_unsigned() as u64,
                dividend.checked_rem(divisor)?.cast_unsigned() as u64,
            )
        };
        Some((
            self.of_same_type_from_bits(quotient, bits)?,
            self.of_same_type_from_bits(remainder, bits)?,
        ))
    }

    /// The integer `self` less `amount`, of `self`'s variant and wrapping at
    /// its width: the count a counted block operation leaves
    /// ([`SsaOp::ScanBlk`]). `None` for a constant that is no integer.
    ///
    /// [`SsaOp::ScanBlk`]: crate::ir::ops::SsaOp::ScanBlk
    #[must_use]
    pub fn decremented(&self, amount: u64) -> Option<Self> {
        self.of_same_type_from_bits(self.as_u64_bits()?.wrapping_sub(amount), 64)
    }
}
