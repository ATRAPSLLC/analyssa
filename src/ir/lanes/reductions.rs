//! Reductions of vector constants: whole-vector, horizontal and selected-lane
//! reductions.

use core::cmp::Ordering;

use crate::ir::{
    float::{FloatBinary, FloatBits, FloatRounding, FloatRules},
    integer::BitWidth,
    lanes::{MAX_LANE_BITS, VectorConst, signed_of},
    ops::{VectorElement, VectorElementKind, VectorPredicateLayout, VectorReduceKind},
};

impl VectorConst {
    /// Whether `bits`, one lane of `element`'s floats as this vector's lanes
    /// are where the element states no width, is a number.
    fn number_as(&self, element: VectorElement, bits: u128) -> bool {
        let width = if element.bits == 0 {
            self.lane_bits()
        } else {
            element.bits
        };
        element
            .kind
            .float_format(width)
            .is_some_and(|format| Self::number_as_format(format, bits))
    }

    /// Reduces selected integer source elements with an always-included initial value.
    ///
    /// Sources extend to the result element width before reduction. A nonzero
    /// source length with no selected elements returns the initial value. Zero
    /// source length, noninteger domains and unrepresentable constants are refused.
    #[must_use]
    pub fn reduce_selected(
        &self,
        kind: VectorReduceKind,
        element: VectorElement,
        initial: u128,
        governing: Option<&Self>,
        length: u64,
        layout: VectorPredicateLayout,
    ) -> Option<u128> {
        self.computable()?;
        let source_bits = self.lane_bits();
        if length == 0
            || length > u64::from(self.lanes())
            || element.kind != VectorElementKind::Integer
            || element.bits < source_bits
            || element.bits > MAX_LANE_BITS
            || !element.bits.is_power_of_two()
        {
            return None;
        }
        layout.bit(0, source_bits)?;
        if let Some(mask) = governing {
            mask.mask_summary(None, length, layout, source_bits)?;
        }
        let mut values = Vec::new();
        for index in 0..length {
            let bit = u32::try_from(layout.bit(index, source_bits)?).ok()?;
            if governing.is_none_or(|mask| mask.lane(bit) != 0) {
                let value = self.lane(u32::try_from(index).ok()?);
                values.push(if element.signed == Some(true) {
                    signed_of(value, source_bits).cast_unsigned()
                } else {
                    value
                });
            }
        }
        if values.is_empty() {
            return Some(BitWidth::new(element.bits).low_bits_u128(initial));
        }
        Self::from_lanes(element.bits, values)?.reduce(
            kind,
            element,
            Some(initial),
            true,
            None,
            FloatRules::default(),
        )
    }

    /// Returns `kind` of the lanes, as `element` reads them, starting from
    /// `initial` where one is given (the bits of a scalar of the lane type):
    /// their sum, product, bitwise and, or and exclusive or at the lane's
    /// width; the least or greatest, read as the element's signedness states;
    /// whether any or every lane is non-zero, as one bit, which starts from
    /// nothing.
    ///
    /// Float lanes combine `ordered`, one at a time in lane order after the
    /// initial value, each step rounded as the element states, else as the
    /// `environment` does ([`VectorElement::rounding_in`]), else by the
    /// program's mode ([`FloatRounding::decide`]), or in an order the
    /// reduction does not
    /// state, where the fold gives only what every order gives: a sum no order
    /// rounds ([`FloatBits::sum_in_any_order`]) and a product of at most two
    /// values. The least or greatest folds unless a zero of the other sign
    /// ties with it, which of the two is the architecture's choice.
    ///
    /// A float sum or product no value of which is a NaN, but which an invalid
    /// step makes one (infinity plus minus infinity, zero times infinity), is
    /// the target's default NaN ([`FloatRules::default_nan`]) where it fixes
    /// one: every later step carries that
    /// NaN, whichever architecture, and a sum in no stated order holding both
    /// infinities meets them in every order. Without one it folds nothing.
    #[must_use]
    pub fn reduce(
        &self,
        kind: VectorReduceKind,
        element: VectorElement,
        initial: Option<u128>,
        ordered: bool,
        environment: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<u128> {
        // Lanes whose specials the implementation decides fold only of
        // numbers, to a number.
        let open = element.kind.opens_specials();
        if open && !(self.numbers_as(element)) {
            return None;
        }
        let result = self.reduce_lanes(kind, element, initial, ordered, environment, rules)?;
        (!open || self.number_as(element, result)).then_some(result)
    }

    /// [`Self::reduce`] of lanes whose specials are known.
    fn reduce_lanes(
        &self,
        kind: VectorReduceKind,
        element: VectorElement,
        initial: Option<u128>,
        ordered: bool,
        environment: Option<FloatRounding>,
        rules: FloatRules,
    ) -> Option<u128> {
        let bits = if element.bits == 0 {
            self.lane_bits()
        } else {
            element.bits
        };
        let vector = self.relaned(bits)?;
        vector.computable()?;
        let starting = initial.map(|initial| BitWidth::new(bits).low_bits_u128(initial));
        if element.kind.is_float() {
            let rounding = element.rounding_in(environment);
            let format = element.kind.float_format(bits)?;
            let start = match starting {
                Some(initial) => Some(FloatBits::new(format, initial)?),
                None => None,
            };
            let values = start
                .into_iter()
                .map(Some)
                .chain((0..vector.lanes()).map(|index| vector.float_lane(index, format)))
                .collect::<Option<Vec<_>>>()?;
            let (&first, rest) = values.split_first()?;
            let nan = || rules.default_nan.and_then(|nan| nan.of(format));
            // The steps in order, each rounded alike in every mode the program
            // may set, until one is invalid: the default NaN from there on.
            // Only consulted where the whole fold does not decide, so where
            // the target fixes a default NaN.
            let stepped = |operation: FloatBinary| {
                nan()?;
                let mut accumulated = first;
                for value in rest {
                    if accumulated.invalid_with(*value, operation) {
                        return nan();
                    }
                    accumulated = FloatRounding::decide(rounding, |rounding| match operation {
                        FloatBinary::Multiply => accumulated.checked_mul(*value, Some(rounding)),
                        _ => accumulated.checked_add(*value, Some(rounding)),
                    })?;
                }
                Some(accumulated)
            };
            let infinity = |negative: bool| {
                values.iter().any(|value| {
                    value.invalid_with(*value, FloatBinary::Subtract)
                        && (value.abs() != *value) == negative
                })
            };
            let reduced = match kind {
                VectorReduceKind::Add if ordered => FloatRounding::decide(rounding, |rounding| {
                    rest.iter()
                        .try_fold(first, |sum, value| sum.checked_add(*value, Some(rounding)))
                })
                .or_else(|| stepped(FloatBinary::Add))?,
                VectorReduceKind::Add if infinity(false) && infinity(true) => nan()?,
                VectorReduceKind::Add => FloatBits::sum_in_any_order(&values, rounding)?,
                // Two values multiply the same either way round.
                VectorReduceKind::Mul if ordered || rest.len() <= 1 => {
                    FloatRounding::decide(rounding, |rounding| {
                        rest.iter().try_fold(first, |product, value| {
                            product.checked_mul(*value, Some(rounding))
                        })
                    })
                    .or_else(|| stepped(FloatBinary::Multiply))?
                }
                VectorReduceKind::Min
                | VectorReduceKind::MinNum
                | VectorReduceKind::Max
                | VectorReduceKind::MaxNum => {
                    let least = matches!(kind, VectorReduceKind::Min | VectorReduceKind::MinNum);
                    let extreme = rest.iter().try_fold(first, |extreme, value| {
                        let order = value.compare(extreme)?;
                        Some(if (least && order.is_lt()) || (!least && order.is_gt()) {
                            *value
                        } else {
                            extreme
                        })
                    })?;
                    // `-0.0` and `0.0` compare equal, so which of them is the
                    // extreme is the architecture's choice.
                    if values.iter().any(|value| {
                        *value != extreme && value.compare(extreme).is_some_and(Ordering::is_eq)
                    }) {
                        return None;
                    }
                    extreme
                }
                // The IEEE 754-2019 forms order every pair they state, so
                // lanes reduce to one result whatever the order.
                kind if kind.extreme().is_some() => {
                    let extreme = kind.extreme()?;
                    rest.iter().try_fold(first, |reduced, value| {
                        extreme.float_extreme(reduced, *value)
                    })?
                }
                _ => return None,
            };
            return Some(reduced.bits());
        }
        let values = || starting.into_iter().chain(vector.lane_values());
        Some(match kind {
            VectorReduceKind::Add => {
                BitWidth::new(bits).low_bits_u128(values().fold(0u128, u128::wrapping_add))
            }
            VectorReduceKind::Mul => {
                BitWidth::new(bits).low_bits_u128(values().fold(1u128, u128::wrapping_mul))
            }
            VectorReduceKind::And => {
                values().fold(BitWidth::new(bits).mask_u128(), |all, lane| all & lane)
            }
            VectorReduceKind::Or => values().fold(0, |any, lane| any | lane),
            VectorReduceKind::Xor => values().fold(0, |parity, lane| parity ^ lane),
            VectorReduceKind::Any | VectorReduceKind::All if starting.is_some() => return None,
            VectorReduceKind::Any => u128::from(!vector.is_zero()),
            VectorReduceKind::All => u128::from(vector.lane_values().all(|lane| lane != 0)),
            VectorReduceKind::Min | VectorReduceKind::Max => {
                let signed = element.signed?;
                values().reduce(|left, right| {
                    let order = if signed {
                        signed_of(left, bits).cmp(&signed_of(right, bits))
                    } else {
                        left.cmp(&right)
                    };
                    if kind == VectorReduceKind::Min && order.is_gt()
                        || kind == VectorReduceKind::Max && order.is_lt()
                    {
                        right
                    } else {
                        left
                    }
                })?
            }
            // A difference reduces groups alone ([`Self::horizontal`]).
            VectorReduceKind::MinNum
            | VectorReduceKind::MaxNum
            | VectorReduceKind::MinimumNumber
            | VectorReduceKind::MaximumNumber
            | VectorReduceKind::MinimumMagnitude
            | VectorReduceKind::MaximumMagnitude
            | VectorReduceKind::MinimumMagnitudeNumber
            | VectorReduceKind::MaximumMagnitudeNumber
            | VectorReduceKind::Difference => {
                return None;
            }
        })
    }

    /// Returns `lanes` lanes `wide` bits each, each the sum of the group of
    /// lanes `narrow` bits wide it covers, extended as `unsigned` says, or,
    /// `subtract`, the first of a pair less the second.
    #[must_use]
    pub fn horizontal(
        &self,
        narrow: u32,
        wide: u32,
        lanes: u32,
        unsigned: bool,
        subtract: bool,
    ) -> Option<Self> {
        let source = self.relaned(narrow)?;
        source.computable()?;
        if wide > MAX_LANE_BITS {
            return None;
        }
        let group = wide.checked_div(narrow).filter(|group| *group > 0)?;
        let extended = |index: u32| {
            if unsigned {
                source.lane(index)
            } else {
                source.signed_lane(index).cast_unsigned()
            }
        };
        let computed = (0..lanes)
            .map(|index| {
                let first = index.checked_mul(group)?;
                if first.checked_add(group)? > source.lanes() {
                    return None;
                }
                Some(if subtract {
                    extended(first).wrapping_sub(extended(first.checked_add(1)?))
                } else {
                    (first..first.saturating_add(group))
                        .fold(0u128, |sum, lane| sum.wrapping_add(extended(lane)))
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(wide, computed)
    }
}
