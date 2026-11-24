//! Lane-wise operations of vector constants: unary, binary, ternary and
//! comparison kernels applied lane by lane.

use crate::ir::{
    float::{FloatBinary, FloatBits, FloatRounding, FloatRules},
    integer::BitWidth,
    lanes::{VectorConst, lane_range, signed_of},
    ops::{
        VectorBinaryKind, VectorCompareKind, VectorElement, VectorElementKind, VectorTernaryKind,
        VectorUnaryKind,
    },
};

impl VectorConst {
    /// Returns the lane-wise wrapping integer `kind` of `self` and `other`, for
    /// the arithmetic every integer lane computes the same signed or unsigned:
    /// addition, subtraction, multiplication, and the shifts by a lane's count
    /// (a count past the lane shifts every bit out, and the right shift is
    /// logical).
    #[must_use]
    pub fn wrapping(&self, other: &Self, kind: VectorBinaryKind) -> Option<Self> {
        let element = VectorElement {
            kind: VectorElementKind::Integer,
            bits: self.lane_bits(),
            scalar: false,
            signed: Some(false),
            rounding: None,
        };
        match kind {
            VectorBinaryKind::Add
            | VectorBinaryKind::Sub
            | VectorBinaryKind::Mul
            | VectorBinaryKind::Shl
            | VectorBinaryKind::Shr => {
                self.binary(other, kind, element, None, FloatRules::default())
            }
            _ => None,
        }
    }

    /// Returns `kind` of the lanes of `self` and `other` as `element` reads
    /// them.
    ///
    /// Float lanes ([`VectorElementKind::is_float`]) add, subtract, multiply,
    /// divide, take the minimum and maximum, as [`FloatBits`] computes them,
    /// rounding as the element states, else as the `environment` does
    /// ([`VectorElement::rounding_in`]), else by the program's mode
    /// ([`FloatRounding::decide`]); a NaN lane folds nothing. Integer lanes
    /// compute every kind whose result the lanes decide, at the lane's width:
    /// wrapping arithmetic, the logic, shifts by a lane's count, the saturating
    /// sums, the minimum, maximum, average, absolute difference, quotient and
    /// remainder of lanes read as the element's signedness states, the
    /// extremes by magnitude of lanes read signed, the high halves of
    /// products, and the variable shifts by a count's signed low byte. A kind that depends on a signedness the element does not state, or
    /// on anything but the lanes, folds nothing.
    ///
    /// The lane width is the element's where it states one and the operands'
    /// otherwise; `element.scalar` computes lane 0 and keeps `self`'s other
    /// lanes.
    ///
    /// An invalid operation of float lanes no operand of which is a NaN
    /// (infinity minus infinity, zero times infinity) gives the target's default NaN ([`FloatRules::default_nan`]) in
    /// its lane, where the target fixes one; without one, such a lane folds
    /// nothing.
    #[must_use]
    pub fn binary(
        &self,
        other: &Self,
        kind: VectorBinaryKind,
        element: VectorElement,
        environment: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<Self> {
        // Lanes whose specials the implementation decides fold only of
        // numbers, to a number.
        let open = element.kind.opens_specials();
        if open && !(self.numbers_as(element) && other.numbers_as(element)) {
            return None;
        }
        let result = self.binary_lanes(other, kind, element, environment, rules)?;
        (!open || result.numbers_as(element)).then_some(result)
    }

    /// [`Self::binary`] of lanes whose specials are known.
    fn binary_lanes(
        &self,
        other: &Self,
        kind: VectorBinaryKind,
        element: VectorElement,
        environment: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<Self> {
        let bits = if element.bits == 0 {
            self.lane_bits()
        } else {
            element.bits
        };
        let (left, right) = (self.relaned(bits)?, other.relaned(bits)?);
        left.computable()?;
        if element.scalar {
            let lane = Self::from_lanes(bits, [left.lane(0)])?;
            let computed = lane.binary(
                &Self::from_lanes(bits, [right.lane(0)])?,
                kind,
                VectorElement {
                    scalar: false,
                    ..element
                },
                environment,
                rules,
            )?;
            return left.with_lane(0, computed.lane(0));
        }
        if left.lanes() != right.lanes() {
            return None;
        }
        // A bitwise operation reads the bits of float lanes as it does any
        // others (x86 `andps`, `andnpd`): only arithmetic reads them as floats.
        if element.kind.is_float() && !kind.is_bitwise() {
            let rounding = element.rounding_in(environment);
            let format = element.kind.float_format(bits)?;
            // A scale by an integer exponent reads its second operand's lanes
            // as signed integers. Exact or nothing: a subnormal operand or
            // result is what a flush mode the program may set reads or writes
            // as zero.
            if kind == VectorBinaryKind::ScaleInteger {
                let lanes = (0..left.lanes())
                    .map(|index| {
                        let value = left.float_lane(index, format)?;
                        let power = i32::try_from(right.signed_lane(index)).ok()?;
                        let scaled = value.scaled_by_power(power)?;
                        (!value.is_subnormal() && !scaled.is_subnormal()).then(|| scaled.bits())
                    })
                    .collect::<Option<Vec<_>>>()?;
                return Self::from_lanes(bits, lanes);
            }
            let lanes = (0..left.lanes())
                .map(|index| {
                    let (a, b) = (
                        left.float_lane(index, format)?,
                        right.float_lane(index, format)?,
                    );
                    // A NaN operand gives the NaN the target's rule picks, and
                    // an invalid operation of numbers the target's default
                    // NaN, whatever the rounding.
                    let arithmetic = |operation: FloatBinary, exact: Option<FloatBits>| {
                        if a.is_nan() || b.is_nan() {
                            return rules.propagated(&[a, b]);
                        }
                        exact.or_else(|| {
                            rules
                                .default_nan
                                .filter(|_| a.invalid_with(b, operation))
                                .and_then(|nan| nan.of(format))
                        })
                    };
                    let result = match kind {
                        VectorBinaryKind::Add => {
                            arithmetic(FloatBinary::Add, a.checked_add(b, rounding))?
                        }
                        VectorBinaryKind::Sub => {
                            arithmetic(FloatBinary::Subtract, a.checked_sub(b, rounding))?
                        }
                        VectorBinaryKind::Mul => {
                            arithmetic(FloatBinary::Multiply, a.checked_mul(b, rounding))?
                        }
                        VectorBinaryKind::Div => {
                            arithmetic(FloatBinary::Divide, a.checked_div(b, rounding))?
                        }
                        _ => kind.float_extreme(a, b)?,
                    };
                    // A flush mode the program may set reads a subnormal
                    // operand, and writes a subnormal result, as zero.
                    rules
                        .unflushed(format, &[a, b, result])
                        .then(|| result.bits())
                })
                .collect::<Option<Vec<_>>>()?;
            return Self::from_lanes(bits, lanes);
        }
        let signed = element.signed;
        let read =
            |vector: &Self, index: u32| signed.and_then(|signed| vector.read_lane(index, signed));
        let range = signed.and_then(|signed| lane_range(bits, signed));
        let lanes = (0..left.lanes())
            .map(|index| {
                let (a, b) = (left.lane(index), right.lane(index));
                // A count past the lane shifts every bit out.
                let count = u32::try_from(b).unwrap_or(u32::MAX).min(bits);
                let pair = || Some((read(&left, index)?, read(&right, index)?));
                Some(match kind {
                    VectorBinaryKind::Add => a.wrapping_add(b),
                    VectorBinaryKind::Sub => a.wrapping_sub(b),
                    VectorBinaryKind::Mul => a.wrapping_mul(b),
                    VectorBinaryKind::And => a & b,
                    VectorBinaryKind::Or => a | b,
                    VectorBinaryKind::Xor => a ^ b,
                    VectorBinaryKind::AndNot => a & !b,
                    VectorBinaryKind::OrNot => a | !b,
                    VectorBinaryKind::Nor => !(a | b),
                    VectorBinaryKind::Shl => a.checked_shl(count).unwrap_or(0),
                    VectorBinaryKind::Shr if signed == Some(true) => (left.signed_lane(index)
                        >> count.min(bits.saturating_sub(1)))
                    .cast_unsigned(),
                    VectorBinaryKind::Shr => a.checked_shr(count).unwrap_or(0),
                    VectorBinaryKind::Rol | VectorBinaryKind::Ror => {
                        let count = u32::try_from(b.checked_rem(u128::from(bits))?).ok()?;
                        let count = if kind == VectorBinaryKind::Ror {
                            bits.wrapping_sub(count).checked_rem(bits)?
                        } else {
                            count
                        };
                        a.checked_shl(count).unwrap_or(0)
                            | a.checked_shr(bits.wrapping_sub(count)).unwrap_or(0)
                    }
                    VectorBinaryKind::SatAdd | VectorBinaryKind::SatSub => {
                        let ((x, y), (min, max)) = (pair()?, range?);
                        let exact = if kind == VectorBinaryKind::SatAdd {
                            x.checked_add(y)?
                        } else {
                            x.checked_sub(y)?
                        };
                        exact.clamp(min, max).cast_unsigned()
                    }
                    VectorBinaryKind::Min => {
                        let (x, y) = pair()?;
                        x.min(y).cast_unsigned()
                    }
                    VectorBinaryKind::Max => {
                        let (x, y) = pair()?;
                        x.max(y).cast_unsigned()
                    }
                    // Rounded up: `(a + b + 1) >> 1`.
                    VectorBinaryKind::Avg => {
                        let (x, y) = pair()?;
                        (x.checked_add(y)?.checked_add(1)? >> 1).cast_unsigned()
                    }
                    VectorBinaryKind::AbsDiff => {
                        let (x, y) = pair()?;
                        x.checked_sub(y)?.unsigned_abs()
                    }
                    // The lane whose signed reading has the greater, or the
                    // smaller, absolute value, compared unsigned so the
                    // minimum integer has the greatest; the second source on
                    // a tie, as every integer user keeps it (MSA `max_a`,
                    // Cyrix `pmagw` stated with its sources reversed).
                    VectorBinaryKind::MaxMagnitude | VectorBinaryKind::MinMagnitude => {
                        let (x, y) = (
                            left.signed_lane(index).unsigned_abs(),
                            right.signed_lane(index).unsigned_abs(),
                        );
                        let first = if kind == VectorBinaryKind::MaxMagnitude {
                            x > y
                        } else {
                            x < y
                        };
                        if first { a } else { b }
                    }
                    VectorBinaryKind::Div if signed == Some(false) => a.checked_div(b)?,
                    VectorBinaryKind::Rem if signed == Some(false) => a.checked_rem(b)?,
                    VectorBinaryKind::Div => {
                        let (x, y) = pair()?;
                        x.checked_div(y)?.cast_unsigned()
                    }
                    VectorBinaryKind::Rem => {
                        let (x, y) = pair()?;
                        x.checked_rem(y)?.cast_unsigned()
                    }
                    VectorBinaryKind::MulHigh => {
                        let (x, y) = pair()?;
                        (x.checked_mul(y)? >> bits).cast_unsigned()
                    }
                    // `(a * b + 2^(n - 1)) >> n`.
                    VectorBinaryKind::MulHighRound => {
                        let (x, y) = pair()?;
                        let round = 1i128.checked_shl(bits.checked_sub(1)?)?;
                        (x.checked_mul(y)?.checked_add(round)? >> bits).cast_unsigned()
                    }
                    // `sat((2ab + round) >> n)` of signed lanes: only two
                    // minimums double past the range.
                    VectorBinaryKind::MulHighDoubleSat
                    | VectorBinaryKind::MulHighDoubleSatRound
                    | VectorBinaryKind::MulHighDoubleRound => {
                        let (x, y) = (left.signed_lane(index), right.signed_lane(index));
                        let rounds = matches!(
                            kind,
                            VectorBinaryKind::MulHighDoubleSatRound
                                | VectorBinaryKind::MulHighDoubleRound
                        );
                        let round = if rounds {
                            1i128.checked_shl(bits.checked_sub(1)?)?
                        } else {
                            0
                        };
                        let doubled = x.checked_mul(y)?.checked_mul(2)?.checked_add(round)? >> bits;
                        let saturates = matches!(
                            kind,
                            VectorBinaryKind::MulHighDoubleSat
                                | VectorBinaryKind::MulHighDoubleSatRound
                        );
                        if saturates {
                            let (min, max) = lane_range(bits, true)?;
                            doubled.clamp(min, max).cast_unsigned()
                        } else {
                            doubled.cast_unsigned()
                        }
                    }
                    // The count is the lane's signed low byte; a negative one
                    // shifts right, and one past the lane shifts every bit out.
                    VectorBinaryKind::VariableShiftArithmetic
                    | VectorBinaryKind::VariableShiftLogical => {
                        let count = signed_of(b & 0xff, 8);
                        let arithmetic = kind == VectorBinaryKind::VariableShiftArithmetic;
                        let width = i128::from(bits);
                        let magnitude = u32::try_from(count.unsigned_abs()).unwrap_or(u32::MAX);
                        match count {
                            0.. if count < width => a.checked_shl(magnitude).unwrap_or(0),
                            0.. => 0,
                            _ if arithmetic => (left.signed_lane(index)
                                >> magnitude.min(bits.saturating_sub(1)))
                            .cast_unsigned(),
                            _ => a.checked_shr(magnitude).unwrap_or(0),
                        }
                    }
                    _ => return None,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(bits, lanes)
    }

    /// Returns `kind` of each lane as `element` reads it.
    ///
    /// Integer lanes negate, invert, take the absolute value (of a signed
    /// lane, wrapping at the minimum), count their set, leading zero, leading
    /// sign and leading one bits, reverse their bits, and pass through. Float
    /// lanes negate, take the absolute value and pass through, NaN included,
    /// round to an integral value in the modes [`FloatBits`] names, and take
    /// the square root and round to an integral value, one a bounded round
    /// gives the integer's minimum for out of its range
    /// ([`FloatBits::integral_within`]), as the element states, else as the
    /// `environment` does ([`VectorElement::rounding_in`]), else by the
    /// program's mode ([`FloatRounding::decide`]). Any other kind folds
    /// nothing.
    ///
    /// An invalid operation of a float lane that is no NaN (the square root of
    /// a negative number) gives the target's default NaN ([`FloatRules::default_nan`]) in its lane, where the target
    /// fixes one; without one, such a lane folds nothing.
    #[must_use]
    pub fn unary(
        &self,
        kind: VectorUnaryKind,
        element: VectorElement,
        environment: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<Self> {
        // Lanes whose specials the implementation decides fold only of
        // numbers, to a number.
        let open = element.kind.opens_specials();
        if open && !(self.numbers_as(element)) {
            return None;
        }
        let result = self.unary_lanes(kind, element, environment, rules)?;
        (!open || result.numbers_as(element)).then_some(result)
    }

    /// [`Self::unary`] of lanes whose specials are known.
    fn unary_lanes(
        &self,
        kind: VectorUnaryKind,
        element: VectorElement,
        environment: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<Self> {
        let bits = if element.bits == 0 {
            self.lane_bits()
        } else {
            element.bits
        };
        let vector = self.relaned(bits)?;
        vector.computable()?;
        if element.scalar {
            let lane = Self::from_lanes(bits, [vector.lane(0)])?.unary(
                kind,
                VectorElement {
                    scalar: false,
                    ..element
                },
                environment,
                rules,
            )?;
            return vector.with_lane(0, lane.lane(0));
        }
        // A bitwise operation reads the bits of float lanes as it does any
        // others: only arithmetic reads them as floats.
        if element.kind.is_float() && !kind.is_bitwise() {
            let rounding = element.rounding_in(environment);
            let format = element.kind.float_format(bits)?;
            return vector.map(|lane| {
                let value = FloatBits::new(format, lane)?;
                let result = match kind {
                    VectorUnaryKind::Neg => value.negate(),
                    VectorUnaryKind::Abs => value.abs(),
                    // -1, 0 or 1 in the lane's format: either zero is 0. A
                    // NaN has no sign to state.
                    VectorUnaryKind::Sign if !value.is_nan() => {
                        let sign = if value.abs().bits() == 0 {
                            0
                        } else if value.is_negative() {
                            -1
                        } else {
                            1
                        };
                        FloatBits::new(format, format.from_integer(sign))?
                    }
                    VectorUnaryKind::NegArithmetic if !value.is_nan() => value.negate(),
                    VectorUnaryKind::AbsArithmetic if !value.is_nan() => value.abs(),
                    VectorUnaryKind::Identity => value,
                    VectorUnaryKind::Truncate => value.trunc()?,
                    VectorUnaryKind::Floor => value.floor()?,
                    VectorUnaryKind::Ceil => value.ceil()?,
                    VectorUnaryKind::RoundNearestEven => value.round_ties_even()?,
                    VectorUnaryKind::RoundNearestAway => value.round_ties_away()?,
                    VectorUnaryKind::Round | VectorUnaryKind::RoundExact => {
                        FloatRounding::decide(rounding, |rounding| value.integral(rounding))?
                    }
                    VectorUnaryKind::RoundInt32 | VectorUnaryKind::RoundInt64 => {
                        let width = kind.integer_bound()?;
                        FloatRounding::decide(rounding, |rounding| {
                            value.integral_within(rounding, width)
                        })?
                    }
                    VectorUnaryKind::Sqrt if value.is_nan() => rules.propagated(&[value])?,
                    VectorUnaryKind::Sqrt => FloatRounding::decide(rounding, |rounding| {
                        Some(value.sqrt_rounded(rounding, false)?.value)
                    })
                    .or_else(|| {
                        rules
                            .default_nan
                            .filter(|_| value.invalid_sqrt())
                            .and_then(|nan| nan.of(format))
                    })?,
                    _ => return None,
                };
                // The sign operations are bit operations no flush mode reads;
                // every other one reads a subnormal operand, and writes a
                // subnormal result, as zero under one.
                let bits_only = matches!(
                    kind,
                    VectorUnaryKind::Neg | VectorUnaryKind::Abs | VectorUnaryKind::Identity
                );
                (bits_only || rules.unflushed(format, &[value, result])).then(|| result.bits())
            });
        }
        if matches!(
            kind,
            VectorUnaryKind::ByteSwap | VectorUnaryKind::BitReverseIn8
        ) && !bits.is_multiple_of(8)
        {
            return None;
        }
        let leading = |lane: u128| {
            let unused = 128u32.saturating_sub(bits);
            (lane << unused).leading_zeros().min(bits)
        };
        vector.map(|lane| {
            Some(match kind {
                VectorUnaryKind::Neg => lane.wrapping_neg(),
                VectorUnaryKind::Not => !lane,
                VectorUnaryKind::Identity => lane,
                VectorUnaryKind::Abs => signed_of(lane, bits).unsigned_abs(),
                // -1, 0 or 1 as the lane is read: an unsigned lane is never
                // negative, and one read neither way has no sign to state.
                VectorUnaryKind::Sign => match element.signed? {
                    true => BitWidth::new(bits)
                        .low_bits_u128(signed_of(lane, bits).signum().cast_unsigned()),
                    false => u128::from(lane != 0),
                },
                VectorUnaryKind::Popcount => u128::from(lane.count_ones()),
                VectorUnaryKind::LeadingZeros => u128::from(leading(lane)),
                VectorUnaryKind::TrailingZeros => u128::from(lane.trailing_zeros().min(bits)),
                VectorUnaryKind::ByteSwap => lane.swap_bytes() >> 128u32.saturating_sub(bits),
                VectorUnaryKind::BitReverseIn8 => lane.reverse_bits().swap_bytes(),
                VectorUnaryKind::LeadingOnes => {
                    u128::from(leading(BitWidth::new(bits).low_bits_u128(!lane)))
                }
                // The bits below the sign that equal it.
                VectorUnaryKind::LeadingSignBits => {
                    let sign = (lane >> bits.saturating_sub(1)) & 1;
                    let same = if sign == 1 {
                        BitWidth::new(bits).low_bits_u128(!lane)
                    } else {
                        lane
                    };
                    u128::from(leading(same).saturating_sub(1))
                }
                VectorUnaryKind::BitReverse => lane.reverse_bits() >> 128u32.saturating_sub(bits),
                _ => return None,
            })
        })
    }

    /// Returns each lane where the relation `kind` holds between the lanes of
    /// `self` and `other` all ones, and zero where it does not, at the
    /// operands' lane width.
    ///
    /// Integer lanes order as `element` reads them, signed or unsigned; they
    /// are never unordered. Float lanes order as IEEE does: a NaN is
    /// unordered with everything, `-0.0` equals `0.0`.
    #[must_use]
    pub fn compare(
        &self,
        other: &Self,
        kind: VectorCompareKind,
        element: VectorElement,
    ) -> Option<Self> {
        // Lanes whose specials the implementation decides compare only as
        // numbers.
        if element.kind.opens_specials() && !(self.numbers_as(element) && other.numbers_as(element))
        {
            return None;
        }
        self.compare_lanes(other, kind, element)
    }

    /// [`Self::compare`] of lanes whose specials are known.
    fn compare_lanes(
        &self,
        other: &Self,
        kind: VectorCompareKind,
        element: VectorElement,
    ) -> Option<Self> {
        let bits = if element.bits == 0 {
            self.lane_bits()
        } else {
            element.bits
        };
        let (left, right) = (self.relaned(bits)?, other.relaned(bits)?);
        left.computable()?;
        if left.lanes() != right.lanes() {
            return None;
        }
        let format = if element.kind.is_float() {
            Some(element.kind.float_format(bits)?)
        } else {
            None
        };
        // Equality is the same read either way; an order is not.
        let signed = match kind {
            VectorCompareKind::Eq
            | VectorCompareKind::Ne
            | VectorCompareKind::AlwaysTrue
            | VectorCompareKind::AlwaysFalse => element.signed.unwrap_or(false),
            _ if format.is_some() => false,
            _ => element.signed?,
        };
        let all = BitWidth::new(bits).mask_u128();
        let lanes = (0..left.lanes())
            .map(|index| {
                let order = match format {
                    Some(format) => left
                        .float_lane(index, format)?
                        .compare(right.float_lane(index, format)?),
                    None if signed => Some(left.signed_lane(index).cmp(&right.signed_lane(index))),
                    None => Some(left.lane(index).cmp(&right.lane(index))),
                };
                let holds = match (kind, order) {
                    (VectorCompareKind::AlwaysTrue, _) => true,
                    (VectorCompareKind::AlwaysFalse, _) => false,
                    (VectorCompareKind::Unordered, order) => order.is_none(),
                    (VectorCompareKind::Ordered, order) => order.is_some(),
                    (VectorCompareKind::UnorderedEq, order) => {
                        order.is_none_or(|order| order.is_eq())
                    }
                    (VectorCompareKind::Ne, order) => order.is_none_or(|order| order.is_ne()),
                    (VectorCompareKind::NotLt, order) => order.is_none_or(|order| order.is_ge()),
                    (VectorCompareKind::NotLe, order) => order.is_none_or(|order| order.is_gt()),
                    (VectorCompareKind::NotGe, order) => order.is_none_or(|order| order.is_lt()),
                    (VectorCompareKind::NotGt, order) => order.is_none_or(|order| order.is_le()),
                    (_, None) => false,
                    (VectorCompareKind::Eq, Some(order)) => order.is_eq(),
                    (VectorCompareKind::OrderedNe, Some(order)) => order.is_ne(),
                    (VectorCompareKind::Lt, Some(order)) => order.is_lt(),
                    (VectorCompareKind::Le, Some(order)) => order.is_le(),
                    (VectorCompareKind::Gt, Some(order)) => order.is_gt(),
                    (VectorCompareKind::Ge, Some(order)) => order.is_ge(),
                };
                Some(if holds { all } else { 0 })
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(bits, lanes)
    }

    /// Returns `kind` of the three vectors `self`, `second` and `third`, at
    /// the element's lane width or else `self`'s.
    ///
    /// [`VectorTernaryKind::Fma`] computes `self * second + third` of float
    /// lanes rounded once, as [`FloatBits`] does, as the element states, else
    /// as the `environment` does ([`VectorElement::rounding_in`]), else by the
    /// program's mode ([`FloatRounding::decide`]), and its signed forms
    /// ([`VectorTernaryKind::fused_signs`]) the product or the addend negated;
    /// a NaN operand gives the NaN the target's rule picks of the operands as
    /// read ([`FloatRules::propagated`]); a scalar element computes lane 0 and
    /// keeps `self`'s other lanes. [`VectorTernaryKind::Select`]
    /// takes `second`'s lane where `third`'s has
    /// its most significant bit set and `self`'s where it is clear; the
    /// signed doubling high multiplies that accumulate
    /// ([`VectorTernaryKind::MulAddHighDoubleSatRound`],
    /// [`VectorTernaryKind::MulSubHighDoubleSatRound`]) compute
    /// `sat(((c << n) ± 2ab + 2^(n - 1)) >> n)` of `self` and `second` into
    /// `third`; the funnel shifts shift the double-width `self:second` by
    /// `third`'s lane modulo the width and keep the high or low half; the
    /// fix-up with NaN passthrough ([`VectorTernaryKind::FixupNan`]) fixes up
    /// `second`'s float lane by `third`'s table, keeping `self`'s where the
    /// table leaves it. Any other kind folds nothing.
    ///
    /// An invalid operation of float lanes no operand of which is a NaN (a
    /// fused zero times infinity) gives the target's default NaN ([`FloatRules::default_nan`]) in its lane, where the
    /// target fixes one; without one, such a lane folds nothing.
    #[must_use]
    pub fn ternary(
        &self,
        second: &Self,
        third: &Self,
        kind: VectorTernaryKind,
        element: VectorElement,
        environment: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<Self> {
        // Lanes whose specials the implementation decides fold only of
        // numbers, to a number.
        let open = element.kind.opens_specials();
        if open
            && !(self.numbers_as(element)
                && second.numbers_as(element)
                && third.numbers_as(element))
        {
            return None;
        }
        let result = self.ternary_lanes(second, third, kind, element, environment, rules)?;
        (!open || result.numbers_as(element)).then_some(result)
    }

    /// [`Self::ternary`] of lanes whose specials are known.
    fn ternary_lanes(
        &self,
        second: &Self,
        third: &Self,
        kind: VectorTernaryKind,
        element: VectorElement,
        environment: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<Self> {
        let bits = if element.bits == 0 {
            self.lane_bits()
        } else {
            element.bits
        };
        let first = self.relaned(bits)?;
        first.computable()?;
        let (second, third) = (second.relaned(bits)?, third.relaned(bits)?);
        if let Some((negate_product, negate_addend)) = kind.fused_signs() {
            let rounding = element.rounding_in(environment);
            let format = element
                .kind
                .float_format(bits)
                .filter(|_| element.kind.is_float())?;
            let lanes = if element.scalar { 1 } else { first.lanes() };
            if second.lanes() != first.lanes() || third.lanes() != first.lanes() {
                return None;
            }
            let mut result = first.clone();
            for index in 0..lanes {
                let [a, b, c] =
                    [&first, &second, &third].map(|vector| vector.float_lane(index, format));
                let (a, b, c) = (a?, b?, c?);
                // A NaN operand gives the NaN the target's rule picks of the
                // operands as read, multiplicand, multiplier and addend in
                // that order: the signs are the operation's.
                if [a, b, c].iter().any(|operand| operand.is_nan()) {
                    result.set_lane(index, rules.propagated(&[a, b, c])?.bits());
                    continue;
                }
                // Negating a number is exact, so the signs applied to the
                // operands give the one rounding of the signed result.
                let a = if negate_product { a.negate() } else { a };
                let c = if negate_addend { c.negate() } else { c };
                let fused = a.checked_mul_add(b, c, rounding).or_else(|| {
                    rules
                        .default_nan
                        .filter(|_| a.invalid_mul_add(b, c))
                        .and_then(|nan| nan.of(format))
                })?;
                if !rules.unflushed(format, &[a, b, c, fused]) {
                    return None;
                }
                result.set_lane(index, fused.bits());
            }
            return Some(result);
        }
        if second.lanes() != first.lanes() || third.lanes() != first.lanes() {
            return None;
        }
        if kind == VectorTernaryKind::FixupNan {
            let format = element
                .kind
                .float_format(bits)
                .filter(|_| element.kind.is_float())?;
            let lanes = (0..first.lanes())
                .map(|index| {
                    let source = second.float_lane(index, format)?;
                    Some(match source.fixed_up_passing_nans(third.lane(index))? {
                        Some(value) => value.bits(),
                        None => first.lane(index),
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            return Self::from_lanes(bits, lanes);
        }
        let lanes = (0..first.lanes())
            .map(|index| {
                let (a, b, c) = (first.lane(index), second.lane(index), third.lane(index));
                Some(match kind {
                    VectorTernaryKind::Select => {
                        if (c >> bits.saturating_sub(1)) & 1 == 1 {
                            b
                        } else {
                            a
                        }
                    }
                    VectorTernaryKind::MulAddHighDoubleSatRound
                    | VectorTernaryKind::MulSubHighDoubleSatRound => {
                        let [x, y, z] =
                            [&first, &second, &third].map(|vector| vector.signed_lane(index));
                        let product = x.checked_mul(y)?.checked_mul(2)?;
                        let product = if kind == VectorTernaryKind::MulSubHighDoubleSatRound {
                            product.checked_neg()?
                        } else {
                            product
                        };
                        let round = 1i128.checked_shl(bits.checked_sub(1)?)?;
                        let value = z
                            .checked_mul(1i128.checked_shl(bits)?)?
                            .checked_add(product)?
                            .checked_add(round)?
                            >> bits;
                        let (min, max) = lane_range(bits, true)?;
                        value.clamp(min, max).cast_unsigned()
                    }
                    // The double-width `a:b` shifted by `c` modulo the
                    // width: its high half shifted left, its low half right.
                    VectorTernaryKind::FunnelLeft | VectorTernaryKind::FunnelRight => {
                        let count = u32::try_from(c.checked_rem(u128::from(bits))?).ok()?;
                        let back = bits.wrapping_sub(count);
                        match (kind, count) {
                            (VectorTernaryKind::FunnelLeft, 0) => a,
                            (_, 0) => b,
                            (VectorTernaryKind::FunnelLeft, _) => {
                                a.checked_shl(count).unwrap_or(0) | b.checked_shr(back).unwrap_or(0)
                            }
                            _ => {
                                b.checked_shr(count).unwrap_or(0) | a.checked_shl(back).unwrap_or(0)
                            }
                        }
                    }
                    _ => return None,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(bits, lanes)
    }
}
