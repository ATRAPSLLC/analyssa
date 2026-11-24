//! Products of vector constants: dot products, exact sums and complex
//! multiplication.

use crate::ir::{
    float::{FloatBits, FloatRounding},
    lanes::{VectorConst, lane_range, signed_of},
    ops::{
        ComplexProduct, ComplexRotation, ComplexShape, DotShape, FloatDotRule, FloatElements,
        VectorElementKind,
    },
};

impl VectorConst {
    /// Multiplies integer lanes and combines them with a wide or equal-width addend.
    ///
    /// Each factor is interpreted according to its corresponding `source_signedness`
    /// Boolean. The result width equals the factor width or twice it, up to 128
    /// bits. Arithmetic wraps at the result width. A one-lane right factor
    /// broadcasts; other inputs have the left factor's lane count. Returns `None`
    /// for incompatible widths or shapes. The caller selects executed lanes and
    /// merges inactive elements with their passthrough value.
    #[must_use]
    pub fn integer_multiply_add(
        &self,
        right: &Self,
        addend: &Self,
        result_bits: u32,
        source_signedness: (bool, bool),
        subtract_product: bool,
    ) -> Option<Self> {
        let bits = self.lane_bits();
        if bits == 0
            || bits > 128
            || result_bits > 128
            || result_bits != bits && Some(result_bits) != bits.checked_mul(2)
            || right.lane_bits() != bits
            || addend.lane_bits() != result_bits
            || right.lanes() != 1 && right.lanes() != self.lanes()
            || addend.lanes() != self.lanes()
        {
            return None;
        }
        let mut result = Self::zeroed(result_bits, self.lanes())?;
        for index in 0..self.lanes() {
            let left = self.lane(index);
            let right = right.lane(if right.lanes() == 1 { 0 } else { index });
            let left = if source_signedness.0 {
                signed_of(left, bits).cast_unsigned()
            } else {
                left
            };
            let right = if source_signedness.1 {
                signed_of(right, bits).cast_unsigned()
            } else {
                right
            };
            let product = left.wrapping_mul(right);
            let value = if subtract_product {
                addend.lane(index).wrapping_sub(product)
            } else {
                addend.lane(index).wrapping_add(product)
            };
            result.set_lane(index, value);
        }
        Some(result)
    }

    /// The float dot product `elements` and `rule` state: each lane of
    /// `self`, the accumulator, of the destination format, plus the products
    /// of the group of source lanes of `left` and of `right` that lane
    /// covers, as the instruction adds them.
    ///
    /// x86's ([`FloatDotRule::Sequential`], `vdpbf16ps`) adds the odd
    /// pair's product, then the even pair's, each sum rounded to nearest even,
    /// a subnormal input or sum read and written as zero whatever MXCSR says;
    /// it folds where each product is a normal number or zero, which the
    /// accumulator's format holds exactly, so whether the instruction rounds
    /// the product apart from the sum decides nothing. Arm's
    /// ([`FloatDotRule::Fpcr`], `bfdot`; [`FloatDotRule::DotAdd`], `fdot`)
    /// sums the two products and adds the sum, rounding by FPCR: it folds
    /// where every product and sum is exact and none is subnormal, which
    /// every rounding gives alike, and both rules then give that one value.
    ///
    /// A NaN or infinity folds nothing, and neither does a source format a
    /// register names at run time ([`FloatSource::Fp8`]), whose values are
    /// unknown. Nothing folds beyond a group of two either: a longer group's
    /// products can sum exactly one way and round another, so the
    /// implementation's order would decide the answer. `None` in each of
    /// those cases and where the lanes do not group.
    ///
    /// [`FloatSource::Fp8`]: crate::ir::ops::FloatSource::Fp8
    #[must_use]
    pub fn dot_float(
        &self,
        left: &Self,
        right: &Self,
        elements: FloatElements,
        rule: FloatDotRule,
    ) -> Option<Self> {
        let source = elements.source.format()?;
        let dest = elements.dest;
        let group = elements.group();
        let accumulator = self.relaned(dest.bits())?;
        let (left, right) = (left.relaned(source.bits())?, right.relaned(source.bits())?);
        if group == 0
            || left.lanes() != accumulator.lanes().checked_mul(group)?
            || right.lanes() != left.lanes()
        {
            return None;
        }
        let lanes = (0..accumulator.lanes())
            .map(|lane| {
                let factors = |vector: &Self| {
                    let first = lane.checked_mul(group)?;
                    (0..group)
                        .map(|index| {
                            let index = first.checked_add(index)?;
                            let narrow = FloatBits::new(source, vector.lane(index))?;
                            let widened = source.convert(narrow.bits(), dest)?;
                            FloatBits::new(dest, widened).filter(|value| value.is_finite())
                        })
                        .collect::<Option<Vec<_>>>()
                };
                let (a, b) = (factors(&left)?, factors(&right)?);
                let accumulator = FloatBits::new(dest, accumulator.lane(lane))
                    .filter(|value| value.is_finite())?;
                let sum = match (rule, a.as_slice(), b.as_slice()) {
                    (FloatDotRule::Sequential, [a_even, a_odd], [b_even, b_odd]) => {
                        Self::dot_in_turn(accumulator, [*a_even, *a_odd], [*b_even, *b_odd])
                    }
                    (
                        FloatDotRule::Fpcr | FloatDotRule::DotAdd,
                        [a_even, a_odd],
                        [b_even, b_odd],
                    ) => Self::dot_exact(accumulator, [*a_even, *a_odd], [*b_even, *b_odd]),
                    _ => None,
                }?;
                Some(sum.bits())
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(dest.bits(), lanes)
    }

    /// x86's float dot product of one lane ([`Self::dot_float`]): the odd
    /// pair's product added to `accumulator`, then the even pair's, each sum
    /// rounded to nearest even, subnormal inputs and sums read and written
    /// as zero.
    fn dot_in_turn(
        accumulator: FloatBits,
        [a_even, a_odd]: [FloatBits; 2],
        [b_even, b_odd]: [FloatBits; 2],
    ) -> Option<FloatBits> {
        let nearest = Some(FloatRounding::NearestEven);
        let flushed = |value: FloatBits| {
            if value.is_subnormal() {
                FloatBits::new(value.format(), value.bits() & 1 << 31)
            } else {
                Some(value)
            }
        };
        let mut sum = flushed(accumulator)?;
        for (a, b) in [(a_odd, b_odd), (a_even, b_even)] {
            let product = Self::exact_product(flushed(a)?, flushed(b)?)?;
            sum = flushed(
                sum.checked_add(product, nearest)
                    .filter(|value| value.is_finite())?,
            )?;
        }
        Some(sum)
    }

    /// Arm's float dot product of one lane ([`Self::dot_float`]): the
    /// pairs' products summed and the sum added to `accumulator`, where each
    /// of those is exact and none subnormal, so every rounding its rule
    /// could apply gives that value.
    fn dot_exact(
        accumulator: FloatBits,
        [a_even, a_odd]: [FloatBits; 2],
        [b_even, b_odd]: [FloatBits; 2],
    ) -> Option<FloatBits> {
        let normal = |value: FloatBits| (!value.is_subnormal()).then_some(value);
        let exact_sum = |left: FloatBits, right: FloatBits| {
            let up = left.checked_add(right, Some(FloatRounding::Up))?;
            let down = left.checked_add(right, Some(FloatRounding::Down))?;
            (up.bits() == down.bits())
                .then_some(up)
                .and_then(normal)
                .filter(|value| value.is_finite())
        };
        let products = [
            Self::exact_product(normal(a_even)?, normal(b_even)?)?,
            Self::exact_product(normal(a_odd)?, normal(b_odd)?)?,
        ];
        exact_sum(normal(accumulator)?, exact_sum(products[0], products[1])?)
    }

    /// The product of two binary32 values where binary32 holds it exactly as
    /// a normal number or zero, `None` otherwise.
    fn exact_product(left: FloatBits, right: FloatBits) -> Option<FloatBits> {
        let up = left.checked_mul(right, Some(FloatRounding::Up))?;
        let down = left.checked_mul(right, Some(FloatRounding::Down))?;
        (up.bits() == down.bits() && !up.is_subnormal())
            .then_some(up)
            .filter(|value| value.is_finite())
    }

    /// The sum of two floats where their format holds it exactly as a
    /// normal number or zero, `None` otherwise: rounded up and down alike, so
    /// every rounding gives it, and a zero of opposite signs, whose sign the
    /// rounding decides, is none.
    fn exact_sum(left: FloatBits, right: FloatBits) -> Option<FloatBits> {
        let up = left.checked_add(right, Some(FloatRounding::Up))?;
        let down = left.checked_add(right, Some(FloatRounding::Down))?;
        (up.bits() == down.bits() && !up.is_subnormal())
            .then_some(up)
            .filter(|value| value.is_finite())
    }

    /// The complex contraction `shape` computes ([`ComplexProduct`]) of the
    /// complex elements of `self` and `other`, each a pair of lanes with the
    /// real part in the even lane, plus those of `accumulator` where the
    /// product accumulates.
    ///
    /// A product of floats (x86's `vfmulcph`, `vfcmulcph`, `vfmaddcph`,
    /// `vfcmaddcph`) folds where each of its products, each sum of two of
    /// its terms and its result is exact and none subnormal: an
    /// implementation may round the products apart or fuse them, and add the
    /// terms in any order, and every one gives that result alike. A rotation
    /// (Arm's `fcadd`, `cadd`, `sqcadd`) adds the second source turned a
    /// quarter or three quarters to the first: floats where each sum is
    /// exact and none subnormal, which every rounding gives alike; integers
    /// wrapping, or saturating to the signed range where `shape.saturates`.
    /// A scalar form computes the lowest element and keeps the first
    /// source's others. A NaN or infinity folds nothing, and neither does an
    /// integer product, which no instruction states. `None` where the lanes
    /// do not pair.
    #[must_use]
    pub fn complex(
        &self,
        other: &Self,
        accumulator: Option<&Self>,
        shape: ComplexShape,
    ) -> Option<Self> {
        let element = shape.element;
        let bits = if element.bits == 0 {
            self.lane_bits()
        } else {
            element.bits
        };
        let (left, right) = (self.relaned(bits)?, other.relaned(bits)?);
        let lanes = left.lanes();
        if lanes % 2 != 0 || right.lanes() != lanes {
            return None;
        }
        let accumulator = match accumulator {
            Some(accumulator) => Some(accumulator.relaned(bits)?),
            None => None,
        };
        if accumulator
            .as_ref()
            .is_some_and(|accumulator| accumulator.lanes() != lanes)
        {
            return None;
        }
        let float = match element.kind {
            VectorElementKind::Float | VectorElementKind::BFloat16 => {
                Some(element.kind.float_format(bits)?)
            }
            VectorElementKind::Integer => None,
            _ => return None,
        };
        let pairs = if element.scalar { 1 } else { lanes / 2 };
        let mut out: Vec<u128> = (0..lanes).map(|lane| left.lane(lane)).collect();
        for pair in 0..pairs {
            let real = pair.checked_mul(2)?;
            let imaginary = real.checked_add(1)?;
            let parts = match float {
                Some(format) => {
                    let part = |vector: &Self, lane: u32| {
                        FloatBits::new(format, vector.lane(lane)).filter(|value| value.is_finite())
                    };
                    let a = [part(&left, real)?, part(&left, imaginary)?];
                    let b = [part(&right, real)?, part(&right, imaginary)?];
                    let c = match &accumulator {
                        Some(accumulator) => {
                            Some([part(accumulator, real)?, part(accumulator, imaginary)?])
                        }
                        None => None,
                    };
                    Self::complex_floats(shape.product, a, b, c)?.map(FloatBits::bits)
                }
                None => {
                    let a = [left.signed_lane(real), left.signed_lane(imaginary)];
                    let b = [right.signed_lane(real), right.signed_lane(imaginary)];
                    Self::complex_integers(shape.product, a, b, bits, shape.saturates)?
                }
            };
            let [real_part, imaginary_part] = parts;
            *out.get_mut(usize::try_from(real).ok()?)? = real_part;
            *out.get_mut(usize::try_from(imaginary).ok()?)? = imaginary_part;
        }
        Self::from_lanes(bits, out)
    }

    /// One complex element of a float contraction ([`Self::complex`]): the
    /// product of `a` and `b`, `b` conjugated as `product` states, plus `c`
    /// where it accumulates, or `a` plus `b` turned; `None` where a step is
    /// inexact or subnormal.
    fn complex_floats(
        product: ComplexProduct,
        [a_real, a_imaginary]: [FloatBits; 2],
        [b_real, b_imaginary]: [FloatBits; 2],
        c: Option<[FloatBits; 2]>,
    ) -> Option<[FloatBits; 2]> {
        let normal = |value: FloatBits| (!value.is_subnormal()).then_some(value);
        let (a_real, a_imaginary) = (normal(a_real)?, normal(a_imaginary)?);
        let (b_real, b_imaginary) = (normal(b_real)?, normal(b_imaginary)?);
        match product {
            ComplexProduct::Multiply { conjugate, .. } => {
                let b_imaginary = if conjugate {
                    b_imaginary.negate()
                } else {
                    b_imaginary
                };
                // Each part is the sum of its two products and the
                // accumulator's part: exact in every order of adding them.
                let part = |terms: [FloatBits; 2], added: Option<FloatBits>| {
                    let [first, second] = terms;
                    let sum = Self::exact_sum(first, second)?;
                    match added {
                        None => Some(sum),
                        Some(added) => {
                            let added = normal(added)?;
                            Self::exact_sum(first, added)?;
                            Self::exact_sum(second, added)?;
                            Self::exact_sum(sum, added)
                        }
                    }
                };
                let real = [
                    Self::exact_product(a_real, b_real)?,
                    Self::exact_product(a_imaginary, b_imaginary)?.negate(),
                ];
                let imaginary = [
                    Self::exact_product(a_real, b_imaginary)?,
                    Self::exact_product(a_imaginary, b_real)?,
                ];
                Some([
                    part(real, c.map(|[real, _]| real))?,
                    part(imaginary, c.map(|[_, imaginary]| imaginary))?,
                ])
            }
            ComplexProduct::Rotate(rotation) => {
                let (real_term, imaginary_term) = match rotation {
                    ComplexRotation::Quarter => (b_imaginary.negate(), b_real),
                    ComplexRotation::ThreeQuarters => (b_imaginary, b_real.negate()),
                };
                Some([
                    Self::exact_sum(a_real, real_term)?,
                    Self::exact_sum(a_imaginary, imaginary_term)?,
                ])
            }
        }
    }

    /// One complex element of an integer rotation ([`Self::complex`]), `bits`
    /// wide: `a` plus `b` turned, wrapping, or saturating to the signed range;
    /// `None` for a product, which no integer instruction states.
    fn complex_integers(
        product: ComplexProduct,
        [a_real, a_imaginary]: [i128; 2],
        [b_real, b_imaginary]: [i128; 2],
        bits: u32,
        saturates: bool,
    ) -> Option<[u128; 2]> {
        let ComplexProduct::Rotate(rotation) = product else {
            return None;
        };
        let (real, imaginary) = match rotation {
            ComplexRotation::Quarter => (
                a_real.checked_sub(b_imaginary)?,
                a_imaginary.checked_add(b_real)?,
            ),
            ComplexRotation::ThreeQuarters => (
                a_real.checked_add(b_imaginary)?,
                a_imaginary.checked_sub(b_real)?,
            ),
        };
        let maximum = 1i128.checked_shl(bits.checked_sub(1)?)?.checked_sub(1)?;
        let minimum = maximum.checked_neg()?.checked_sub(1)?;
        let mask = 1u128
            .checked_shl(bits)
            .map_or(u128::MAX, |bound| bound.wrapping_sub(1));
        let written = |value: i128| {
            let value = if saturates {
                value.clamp(minimum, maximum)
            } else {
                value
            };
            value.cast_unsigned() & mask
        };
        Some([written(real), written(imaginary)])
    }

    /// Returns each 64-bit lane of `self`, the accumulator, plus a 52-bit
    /// part of the product of the low 52 bits of `left`'s and `right`'s
    /// lanes, unsigned and wrapped: the low 52 bits of the 104-bit product,
    /// or, when `high`, bits 103 to 52 (`vpmadd52luq`, `vpmadd52huq`).
    /// `None` where the three have different lanes.
    #[must_use]
    pub fn multiply_add52(&self, left: &Self, right: &Self, high: bool) -> Option<Self> {
        const PART: u128 = (1 << 52) - 1;
        let (accumulator, left, right) = (self.relaned(64)?, left.relaned(64)?, right.relaned(64)?);
        if accumulator.lanes() != left.lanes() || left.lanes() != right.lanes() {
            return None;
        }
        let lanes = (0..accumulator.lanes())
            .map(|lane| {
                let product = (left.lane(lane) & PART).checked_mul(right.lane(lane) & PART)?;
                let part = if high { product >> 52 } else { product } & PART;
                Some(accumulator.lane(lane).wrapping_add(part) & u128::from(u64::MAX))
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(64, lanes)
    }

    /// Returns the dot products of `self` and `other` as `shape` reads them:
    /// each lane `shape.dest_bits` wide is the sum of the products of the
    /// lanes `shape.source_bits` wide it covers, each source read signed or
    /// not, plus `accumulator`'s lane where there is one, wrapped or
    /// saturated as `shape` says. `None` where the sources are no whole
    /// number of groups, the accumulator has other lanes, or the products
    /// are too wide to sum exactly.
    #[must_use]
    pub fn dot(&self, other: &Self, accumulator: Option<&Self>, shape: DotShape) -> Option<Self> {
        if shape.source_bits == 0 || shape.source_bits > 32 || shape.dest_bits > 64 {
            return None;
        }
        let (a, b) = (
            self.relaned(shape.source_bits)?,
            other.relaned(shape.source_bits)?,
        );
        let group = shape.dest_bits.checked_div(shape.source_bits)?;
        let lanes = a.lanes().checked_div(group)?;
        if lanes == 0 || a.lanes() != b.lanes() || !a.lanes().is_multiple_of(group) {
            return None;
        }
        let accumulator = match accumulator {
            Some(accumulator) => {
                let accumulator = accumulator.relaned(shape.dest_bits)?;
                (accumulator.lanes() == lanes).then_some(Some(accumulator))?
            }
            None => None,
        };
        let range = match shape.saturates {
            Some(signed) => Some(lane_range(shape.dest_bits, signed)?),
            None => None,
        };
        let computed = (0..lanes)
            .map(|lane| {
                let first = lane.checked_mul(group)?;
                let mut sum = match &accumulator {
                    Some(accumulator) => {
                        accumulator.read_lane(lane, shape.saturates == Some(true))?
                    }
                    None => 0,
                };
                for index in first..first.checked_add(group)? {
                    let product = a
                        .read_lane(index, shape.signed_a)?
                        .checked_mul(b.read_lane(index, shape.signed_b)?)?;
                    sum = sum.checked_add(product)?;
                }
                Some(match range {
                    Some((min, max)) => sum.clamp(min, max).cast_unsigned(),
                    None => sum.cast_unsigned(),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(shape.dest_bits, computed)
    }
}
