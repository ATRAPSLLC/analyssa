//! Conversions of vector constants: casts, saturation, fixed-point arithmetic
//! and lane reinterpretation.

use crate::ir::{
    float::{
        FloatBits, FloatExceptions, FloatFormat, FloatRounding, FloatRules, RaisingArithmetic,
    },
    integer::BitWidth,
    lanes::{MAX_LANE_BITS, VectorConst, lane_range, signed_of},
    ops::{VectorCastKind, VectorElement, VectorElementKind, VectorFixedKind},
};

impl VectorConst {
    /// The exceptions `arithmetic` of the lanes of `sources`, as `element`
    /// reads them, raises in the lanes `active` names (every lane where
    /// `None`; lane 0 alone of a scalar element), of the flags `held`, the
    /// ones the status register keeps of those the operation can raise.
    ///
    /// The flags must be the same in every rounding mode the program may set
    /// where neither the element nor the environment states one, and
    /// [`FloatExceptions::ROUNDED_UP`] (x87's `C1`) is set where the rounded
    /// result's magnitude exceeds the one rounded toward zero, that is the
    /// exact one's. `None` where they are not decided: an operand that is
    /// subnormal (input-denormal, and a flush mode reading it as zero), a
    /// result a flush mode the program may set would write as zero
    /// ([`FloatRules::program_flushes_subnormals`]), and lanes whose specials
    /// the implementation decides. A NaN operand raises by its kind in the
    /// target's encoding ([`FloatRules::nan_encoding`]).
    #[must_use]
    pub fn raised(
        sources: &[&Self],
        arithmetic: RaisingArithmetic,
        element: VectorElement,
        environment: Option<FloatRounding>,
        rules: FloatRules,
        held: FloatExceptions,
        active: Option<&[bool]>,
    ) -> Option<FloatExceptions> {
        let first = sources.first()?;
        if element.kind.opens_specials() || !element.kind.is_float() {
            return None;
        }
        let bits = if element.bits == 0 {
            first.lane_bits()
        } else {
            element.bits
        };
        let format = element.kind.float_format(bits)?;
        let sources = sources
            .iter()
            .map(|source| source.relaned(bits))
            .collect::<Option<Vec<_>>>()?;
        let lanes = if element.scalar {
            1
        } else {
            sources.first()?.lanes()
        };
        if sources.iter().any(|source| source.lanes() < lanes) {
            return None;
        }
        let rounding = element.rounding_in(environment);
        let mut raised = FloatExceptions::NONE;
        for index in 0..lanes {
            let lane = usize::try_from(index).ok()?;
            if active.is_some_and(|active| !active.get(lane).copied().unwrap_or(false)) {
                continue;
            }
            let values = sources
                .iter()
                .map(|source| source.float_lane(index, format))
                .collect::<Option<Vec<_>>>()?;
            let lane_raised = rules.raises(arithmetic, &values, rounding, held)?;
            raised = raised.union(lane_raised);
        }
        Some(raised)
    }

    /// Zero-extends data to twice its width and shifts modulo that result width.
    ///
    /// Data lanes from eight through 64 bits are supported. Counts have their own
    /// unsigned width, with one count broadcast or one count per data element.
    #[must_use]
    pub fn widen_shift_left(&self, counts: &Self) -> Option<Self> {
        self.computable()?;
        counts.computable()?;
        let bits = self.lane_bits();
        if !(8..=64).contains(&bits)
            || !bits.is_power_of_two()
            || counts.lanes() != 1 && counts.lanes() != self.lanes()
        {
            return None;
        }
        let result_bits = bits.checked_mul(2)?;
        let mask = u128::from(result_bits.checked_sub(1)?);
        let mut result = Self::splat(result_bits, self.lanes(), 0)?;
        for index in 0..self.lanes() {
            let count = counts.lane(if counts.lanes() == 1 { 0 } else { index }) & mask;
            let amount = u32::try_from(count).ok()?;
            result.set_lane(
                index,
                BitWidth::new(result_bits).low_bits_u128(self.lane(index).checked_shl(amount)?),
            );
        }
        Some(result)
    }

    /// Computes unsigned addition or subtraction with packed carry or borrow bits.
    ///
    /// Returns wrapped data and one outgoing bit per element. A one-element
    /// right source or carry source broadcasts. The carry source has one-bit
    /// lanes and can hold more bits than there are data elements. Both numeric
    /// sources share the left source's element width.
    /// Integer widths from eight through 128 bits are supported.
    #[must_use]
    pub fn integer_with_carry(
        &self,
        right: &Self,
        carry: &Self,
        subtract: bool,
    ) -> Option<(Self, Self)> {
        let bits = self.lane_bits();
        if !(8..=128).contains(&bits)
            || !bits.is_power_of_two()
            || right.lane_bits() != bits
            || right.lanes() != 1 && right.lanes() != self.lanes()
            || carry.lane_bits() != 1
            || carry.lanes() != 1 && carry.lanes() < self.lanes()
        {
            return None;
        }
        let mut data = Self::zeroed(bits, self.lanes())?;
        let mut outgoing = Self::zeroed(1, self.lanes())?;
        for index in 0..self.lanes() {
            let a = self.lane(index);
            let b = right.lane(if right.lanes() == 1 { 0 } else { index });
            let c = carry.lane(if carry.lanes() == 1 { 0 } else { index });
            let (value, bit) = if subtract {
                let (difference, first) = a.overflowing_sub(b);
                let (difference, second) = difference.overflowing_sub(c);
                (difference, first || second)
            } else {
                let (sum, first) = a.overflowing_add(b);
                let (sum, second) = sum.overflowing_add(c);
                (
                    sum,
                    first || second || BitWidth::new(bits).low_bits_u128(sum) != sum,
                )
            };
            data.set_lane(index, value);
            outgoing.set_lane(index, u128::from(bit));
        }
        Some((data, outgoing))
    }

    /// Computes fixed-point arithmetic under a fixed rounding environment.
    ///
    /// Returns the data lanes and whether any lane saturated. The caller selects
    /// executed lanes before aggregating their status. Mode values zero through
    /// three select nearest-up, nearest-even, down and odd. A one-lane right
    /// operand broadcasts; other right operands have the source's lane count.
    /// Clip reads double-width source lanes; all other sources have the result
    /// width. Returns `None` for unsupported widths, domains, modes or shapes.
    #[must_use]
    pub fn fixed_binary(
        &self,
        other: &Self,
        kind: VectorFixedKind,
        element: VectorElement,
        mode: u8,
    ) -> Option<(Self, bool)> {
        let bits = element.bits;
        let signed = element.signed?;
        if !matches!(element.kind, VectorElementKind::Integer)
            || element.scalar
            || !(8..=64).contains(&bits)
            || !bits.is_power_of_two()
            || mode > 3
            || matches!(kind, VectorFixedKind::FractionalMultiply) && !signed
            || other.lanes() != 1 && other.lanes() != self.lanes()
        {
            return None;
        }
        let source_bits = if matches!(kind, VectorFixedKind::Clip) {
            bits.checked_mul(2)?
        } else {
            bits
        };
        if self.lane_bits() != source_bits
            || !matches!(kind, VectorFixedKind::Shift | VectorFixedKind::Clip)
                && other.lane_bits() != bits
        {
            return None;
        }
        let mut result = Self::zeroed(bits, self.lanes())?;
        let (minimum, maximum) = lane_range(bits, signed)?;
        let mut saturated = false;
        for index in 0..self.lanes() {
            let a = self.lane(index);
            let b = other.lane(if other.lanes() == 1 { 0 } else { index });
            let (value, width) = match kind {
                VectorFixedKind::AverageAdd | VectorFixedKind::AverageSub => {
                    let width = bits.checked_add(1)?;
                    let a = if signed {
                        signed_of(a, bits).cast_unsigned()
                    } else {
                        a
                    };
                    let b = if signed {
                        signed_of(b, bits).cast_unsigned()
                    } else {
                        b
                    };
                    let exact = if matches!(kind, VectorFixedKind::AverageSub) {
                        a.wrapping_sub(b)
                    } else {
                        a.wrapping_add(b)
                    };
                    (fixed_round(exact, width, 1, signed, mode)?, width)
                }
                VectorFixedKind::FractionalMultiply => {
                    let width = bits.checked_mul(2)?;
                    let product = signed_of(a, bits).checked_mul(signed_of(b, bits))?;
                    (
                        fixed_round(
                            product.cast_unsigned(),
                            width,
                            bits.checked_sub(1)?,
                            true,
                            mode,
                        )?,
                        width,
                    )
                }
                VectorFixedKind::Shift | VectorFixedKind::Clip => {
                    let count = u32::try_from(b & u128::from(source_bits.checked_sub(1)?)).ok()?;
                    (
                        fixed_round(a, source_bits, count, signed, mode)?,
                        source_bits,
                    )
                }
            };
            let value = if kind.saturates() {
                if signed {
                    let value = signed_of(value, width);
                    saturated |= value < minimum || value > maximum;
                    value.clamp(minimum, maximum).cast_unsigned()
                } else {
                    let maximum = maximum.cast_unsigned();
                    saturated |= value > maximum;
                    value.min(maximum)
                }
            } else {
                value
            };
            result.set_lane(index, value);
        }
        Some((result, saturated))
    }

    /// Returns the low `lanes` lanes converted from `from` to `to`, each
    /// `lane_bits` wide: an integer extended or cut as `from` reads it, a
    /// float converted between formats or from an integer rounding by
    /// `rounding` (by the program's mode where it states none,
    /// [`FloatRounding::decide`]), a NaN converted between formats as `rules`
    /// state ([`FloatRules::converted`]), and a float converted to an integer
    /// toward zero. A NaN the rules state no value for, a float whose integer
    /// the target lane does not hold, an integer converted to odd, a
    /// subnormal a flush mode the program may set reads or writes as zero
    /// ([`FloatRules::program_flushes_subnormals`]), or a lane past the
    /// vector folds nothing: each architecture writes its own value for the
    /// first two.
    #[must_use]
    pub fn cast(
        &self,
        from: VectorCastKind,
        to: VectorCastKind,
        rounding: Option<FloatRounding>,
        lane_bits: u32,
        lanes: u32,
        rules: FloatRules,
    ) -> Option<Self> {
        if lanes > self.computable()?.lanes() || lane_bits > MAX_LANE_BITS {
            return None;
        }
        let narrow = self.lane_bits();
        let format_of = |kind: VectorCastKind, bits: u32| match kind {
            VectorCastKind::BFloat16 => (bits == 16).then_some(FloatFormat::BFloat16),
            _ => FloatFormat::of_width(bits),
        };
        let float = VectorCastKind::is_float;
        let lanes = (0..lanes)
            .map(|index| {
                let lane = self.lane(index);
                match (float(from), float(to)) {
                    // Extended as `from` reads it, or cut.
                    (false, false) => Some(BitWidth::new(lane_bits).low_bits_u128(
                        if from == VectorCastKind::Signed {
                            self.signed_lane(index).cast_unsigned()
                        } else {
                            lane
                        },
                    )),
                    (true, true) => {
                        let value = FloatBits::new(format_of(from, narrow)?, lane)?;
                        let to = format_of(to, lane_bits)?;
                        let result = FloatRounding::decide(rounding, |rounding| {
                            Some(value.convert_rounded(to, rounding, false)?.value)
                        })?;
                        // A flush mode reads a subnormal lane, and writes a
                        // subnormal result, as zero.
                        if !(rules.unflushed(value.format(), &[value])
                            && rules.unflushed(to, &[result]))
                        {
                            return None;
                        }
                        Some(rules.converted(result)?.bits())
                    }
                    (false, true) => {
                        let to = format_of(to, lane_bits)?;
                        let result = FloatRounding::decide(rounding, |rounding| {
                            Some(
                                FloatBits::from_integer_rounded(
                                    to,
                                    lane,
                                    narrow,
                                    from == VectorCastKind::Signed,
                                    rounding,
                                )?
                                .value,
                            )
                        })?;
                        Some(result.bits())
                    }
                    (true, false) => {
                        let value = FloatBits::new(format_of(from, narrow)?, lane)?.to_integer()?;
                        let (min, max) = lane_range(lane_bits, to == VectorCastKind::Signed)?;
                        (min..=max)
                            .contains(&value)
                            .then(|| BitWidth::new(lane_bits).low_bits_u128(value.cast_unsigned()))
                    }
                }
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(lane_bits, lanes)
    }

    /// Returns each lane clamped to the range of a lane `lane_bits` wide,
    /// signed or unsigned as `unsigned_dst` says, the lanes read signed where
    /// `signed_src`: a saturating integer cast.
    #[must_use]
    pub fn saturated(&self, lane_bits: u32, signed_src: bool, unsigned_dst: bool) -> Option<Self> {
        self.computable()?;
        let (min, max) = lane_range(lane_bits, !unsigned_dst)?;
        let lanes = (0..self.lanes())
            .map(|index| {
                Some(
                    self.read_lane(index, signed_src)?
                        .clamp(min, max)
                        .cast_unsigned(),
                )
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(lane_bits, lanes)
    }

    /// Returns each signed lane's absolute value, or its negation, saturated:
    /// the minimum's is the maximum.
    #[must_use]
    pub fn saturating_sign(&self, negate: bool) -> Option<Self> {
        let (min, max) = lane_range(self.computable()?.lane_bits(), true)?;
        let lanes = (0..self.lanes())
            .map(|index| {
                let value = self.signed_lane(index);
                let result = if negate {
                    value.checked_neg()?
                } else {
                    value.checked_abs()?
                };
                Some(result.clamp(min, max).cast_unsigned())
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(self.lane_bits(), lanes)
    }
}

/// Rounds away `amount` low bits using a fixed-point environment.
pub fn fixed_round(value: u128, bits: u32, amount: u32, signed: bool, mode: u8) -> Option<u128> {
    if bits == 0 || bits > 128 || amount >= bits || mode > 3 {
        return None;
    }
    let value = BitWidth::new(bits).low_bits_u128(value);
    if amount == 0 {
        return Some(value);
    }
    let quotient = if signed {
        (signed_of(value, bits) >> amount).cast_unsigned()
    } else {
        value >> amount
    };
    let discarded = BitWidth::new(amount).low_bits_u128(value);
    let half = 1u128.checked_shl(amount.checked_sub(1)?)?;
    let increment = match mode {
        0 => discarded >= half,
        1 => discarded > half || discarded == half && quotient & 1 != 0,
        2 => false,
        3 => discarded != 0 && quotient & 1 == 0,
        _ => return None,
    };
    Some(BitWidth::new(bits).low_bits_u128(quotient.wrapping_add(u128::from(increment))))
}
