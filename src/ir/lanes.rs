//! Vector constants: the bits of a vector value and the lanes they make.
//!
//! A [`VectorConst`] is a vector value by its bits, lane 0 lowest, together
//! with the lane width it is laid out in. The bits are what a register holds;
//! the lanes are how the operation that produced the value reads them. An
//! operation that reads the value at another lane width reshapes it
//! ([`VectorConst::resized`]): a reinterpret is free.
//!
//! The lane semantics of the vector operations are methods here, one per
//! operation family: the lane-wise integer and float arithmetic
//! ([`VectorConst::binary`], [`VectorConst::unary`]), compares, selects,
//! shuffles and permutes, casts, reductions, saturating narrows and the
//! gathered lane bits. Integer lanes compute exactly at their width, up to
//! 128 bits; float lanes compute as [`FloatBits`] does, in round-to-nearest-even
//! and never on a NaN.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, de, de::SeqAccess, de::Visitor};

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{
        content::ContentSink,
        float::{FloatBits, FloatFormat},
        integer::BitWidth,
        ops::VectorElement,
    },
    target::Target,
};

mod conversions;
mod lanewise;
mod products;
mod reductions;
mod selection;

/// The widest vector a constant holds, in bits: a 16-bit lane index for each
/// byte of an 8192-bit matrix register (an AMX tile, SME's ZA at a 256-bit
/// streaming vector length), which selects lanes of the register.
pub const MAX_VECTOR_BITS: u32 = 16384;

/// The words a constant's bits take at most ([`MAX_VECTOR_BITS`]), the
/// bound a serialized constant's storage is read within.
#[cfg(feature = "serde")]
const MAX_VECTOR_WORDS: usize = MAX_VECTOR_BITS.div_ceil(64) as usize;

/// The widest lane the lane-wise operations compute on, in bits. A vector
/// may be laid out in wider lanes - a whole register typed as one lane - which
/// move, reshape and combine bit by bit, but compute nothing lane by lane.
pub const MAX_LANE_BITS: u32 = 128;

/// A vector constant: `lanes` lanes of `lane_bits` bits each, lane 0 in the
/// lowest bits.
///
/// Equality and hashing are by shape and bits: two constants of one bit
/// pattern laid out in different lanes differ, as the values two different
/// vector types hold do.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VectorConst {
    /// The shape, then the bits: word 0 holds the lane width in its low 32
    /// bits and the lane count in its high 32, and the words after it the
    /// bits, the lowest first, those above the vector clear. One allocation
    /// keeps a vector [`crate::ir::ConstValue`] as small as a scalar one.
    #[cfg_attr(feature = "serde", serde(deserialize_with = "deserialize_words"))]
    words: Box<[u64]>,
}

/// Reads only the shape's exact bounded storage, rejecting noncanonical bits.
#[cfg(feature = "serde")]
fn deserialize_words<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Box<[u64]>, D::Error> {
    deserializer.deserialize_seq(VectorWords)
}

/// A fixed-size decoding buffer; no serialized length can reserve heap storage.
#[cfg(feature = "serde")]
struct VectorWords;

#[cfg(feature = "serde")]
impl<'de> Visitor<'de> for VectorWords {
    type Value = Box<[u64]>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "a shape word and its exact vector bits, at most {MAX_VECTOR_BITS} bits"
        )
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut source: A) -> Result<Self::Value, A::Error> {
        let header: u64 = source
            .next_element()?
            .ok_or_else(|| de::Error::custom("missing vector shape"))?;
        let lane_bits = u32::try_from(header & u64::from(u32::MAX)).unwrap_or(0);
        let lanes = u32::try_from(header >> 32).unwrap_or(0);
        let bits = lane_bits
            .checked_mul(lanes)
            .filter(|bits| *bits != 0 && *bits <= MAX_VECTOR_BITS)
            .ok_or_else(|| de::Error::custom("invalid vector shape"))?;
        let count = usize::try_from(bits.div_ceil(64))
            .ok()
            .and_then(|limbs| limbs.checked_add(1))
            .ok_or_else(|| de::Error::custom("invalid vector storage length"))?;
        // One shape word and at most the bound's data words, independent of
        // a size hint.
        let mut storage = [0u64; MAX_VECTOR_WORDS + 1];
        let words = storage
            .get_mut(..count)
            .ok_or_else(|| de::Error::custom("vector storage exceeds its bound"))?;
        for (index, word) in words.iter_mut().enumerate() {
            *word = if index == 0 {
                header
            } else {
                source
                    .next_element()?
                    .ok_or_else(|| de::Error::custom("incomplete vector storage"))?
            };
        }
        if source.next_element::<u64>()?.is_some() {
            return Err(de::Error::custom("excess vector storage"));
        }
        let used = bits % 64;
        if used != 0 && words.last().is_some_and(|last| last >> used != 0) {
            return Err(de::Error::custom("nonzero unused vector bits"));
        }
        Ok(Box::from(words as &[u64]))
    }
}

/// The two's-complement reading of the low `bits` of `value`, `bits` from 1
/// to 128.
const fn signed_of(value: u128, bits: u32) -> i128 {
    let shift = 128u32.saturating_sub(bits);
    (value << shift).cast_signed() >> shift
}

/// The least and greatest values a lane `bits` wide holds, read signed or
/// unsigned, `bits` from 1 to 127.
fn lane_range(bits: u32, signed: bool) -> Option<(i128, i128)> {
    let half = 1i128.checked_shl(bits.checked_sub(1)?)?;
    Some(if signed {
        (half.wrapping_neg(), half.wrapping_sub(1))
    } else {
        (0, half.checked_mul(2)?.wrapping_sub(1))
    })
}

impl VectorConst {
    /// Whether the lanes an operation of `element` computes on are each a
    /// number, finite and not a NaN, read as its floats: lane 0 alone of a
    /// scalar element, which keeps the others as they are.
    fn numbers_as(&self, element: VectorElement) -> bool {
        let bits = if element.bits == 0 {
            self.lane_bits()
        } else {
            element.bits
        };
        let Some((format, vector)) = element.kind.float_format(bits).zip(self.relaned(bits)) else {
            return false;
        };
        let lanes = if element.scalar { 1 } else { vector.lanes() };
        (0..lanes).all(|index| Self::number_as_format(format, vector.lane(index)))
    }

    /// Whether `bits` of `format` is a number, finite and not a NaN.
    fn number_as_format(format: FloatFormat, bits: u128) -> bool {
        FloatBits::new(format, bits).is_some_and(FloatBits::is_finite)
    }

    /// Returns the vector of `lanes` zero lanes `lane_bits` wide; `None` for no
    /// lanes, a lane narrower than a bit, or a vector wider than
    /// [`MAX_VECTOR_BITS`].
    #[must_use]
    pub fn zeroed(lane_bits: u32, lanes: u32) -> Option<Self> {
        let bits = lane_bits.checked_mul(lanes)?;
        if lane_bits == 0 || lanes == 0 || bits > MAX_VECTOR_BITS {
            return None;
        }
        let limbs = usize::try_from(bits.div_ceil(64)).ok()?;
        let mut words = vec![0; limbs.checked_add(1)?];
        if let Some(header) = words.first_mut() {
            *header = u64::from(lane_bits) | u64::from(lanes) << 32;
        }
        Some(Self {
            words: words.into_boxed_slice(),
        })
    }

    /// Returns the vector of the lanes `values` gives, lane 0 first, each
    /// `lane_bits` wide and cut to that width; `None` where [`Self::zeroed`]
    /// gives none for their count. Reads at most one lane beyond the widest
    /// admitted shape; size hints cannot reserve an unbounded temporary buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::ir::lanes::VectorConst;
    ///
    /// let vector = VectorConst::from_lanes(16, [1, 2, 0x1_0003]).unwrap();
    /// assert_eq!(vector.lane(2), 3);
    /// assert_eq!(vector.bits(), 48);
    /// ```
    pub fn from_lanes(lane_bits: u32, values: impl IntoIterator<Item = u128>) -> Option<Self> {
        let maximum = usize::try_from(MAX_VECTOR_BITS.checked_div(lane_bits)?).ok()?;
        if maximum == 0 {
            return None;
        }
        // One excess lane establishes refusal without consuming an unbounded source.
        let values: Vec<u128> = values.into_iter().take(maximum.saturating_add(1)).collect();
        let mut vector = Self::zeroed(lane_bits, u32::try_from(values.len()).ok()?)?;
        for (index, value) in (0u32..).zip(values) {
            vector.set_lane(index, value);
        }
        Some(vector)
    }

    /// Returns `lanes` lanes `lane_bits` wide, each `value` cut to that width.
    #[must_use]
    pub fn splat(lane_bits: u32, lanes: u32, value: u128) -> Option<Self> {
        let mut vector = Self::zeroed(lane_bits, lanes)?;
        for index in 0..lanes {
            vector.set_lane(index, value);
        }
        Some(vector)
    }

    /// Returns the shape word.
    fn header(&self) -> u64 {
        self.words.first().copied().unwrap_or(0)
    }

    /// Returns the bits, in limbs, the lowest first.
    fn limbs(&self) -> &[u64] {
        self.words.get(1..).unwrap_or_default()
    }

    /// Returns the bits, in limbs, for writing.
    fn limbs_mut(&mut self) -> &mut [u64] {
        self.words.get_mut(1..).unwrap_or_default()
    }

    /// Returns the width of one lane, in bits.
    #[must_use]
    pub fn lane_bits(&self) -> u32 {
        u32::try_from(self.header() & u64::from(u32::MAX)).unwrap_or(0)
    }

    /// Returns the number of lanes.
    #[must_use]
    pub fn lanes(&self) -> u32 {
        u32::try_from(self.header() >> 32).unwrap_or(0)
    }

    /// Returns the width of the vector, in bits.
    #[must_use]
    pub fn bits(&self) -> u32 {
        self.lane_bits().saturating_mul(self.lanes())
    }

    /// Returns the vector where its lanes are narrow enough to compute on
    /// ([`MAX_LANE_BITS`]).
    fn computable(&self) -> Option<&Self> {
        (self.lane_bits() <= MAX_LANE_BITS).then_some(self)
    }

    /// Returns the `width` bits starting at bit `start`, `width` up to 128; the
    /// bits above the vector read as zero.
    #[must_use]
    pub fn bits_at(&self, start: u32, width: u32) -> u128 {
        let limb = |index: u32| -> u128 {
            usize::try_from(index)
                .ok()
                .and_then(|index| self.limbs().get(index))
                .map_or(0, |limb| u128::from(*limb))
        };
        let (first, offset) = (start / 64, start % 64);
        // Three limbs cover any 128 bits that start inside the first.
        let window = limb(first) | limb(first.saturating_add(1)) << 64;
        let high = limb(first.saturating_add(2));
        let value =
            (window >> offset) | high.checked_shl(128u32.saturating_sub(offset)).unwrap_or(0);
        BitWidth::new(width).low_bits_u128(value)
    }

    /// Writes the low `width` bits of `value` at bit `start`, `width` up to
    /// 128; bits past the vector are dropped.
    fn set_bits(&mut self, start: u32, width: u32, value: u128) {
        let value = BitWidth::new(width).low_bits_u128(value);
        for bit in 0..width.min(128) {
            let at = start.saturating_add(bit);
            if at >= self.bits() {
                break;
            }
            let (Ok(limb), offset) = (usize::try_from(at / 64), at % 64) else {
                break;
            };
            if let Some(limb) = self.limbs_mut().get_mut(limb) {
                let mask = 1u64 << offset;
                if (value >> bit) & 1 == 1 {
                    *limb |= mask;
                } else {
                    *limb &= !mask;
                }
            }
        }
    }

    /// Returns lane `index`, unsigned; zero for a lane past the vector, and
    /// the low [`MAX_LANE_BITS`] of a wider lane.
    #[must_use]
    pub fn lane(&self, index: u32) -> u128 {
        if index >= self.lanes() {
            return 0;
        }
        self.bits_at(index.saturating_mul(self.lane_bits()), self.lane_bits())
    }

    /// Returns lane `index` read as a two's-complement integer.
    #[must_use]
    pub fn signed_lane(&self, index: u32) -> i128 {
        signed_of(self.lane(index), self.lane_bits())
    }

    /// Returns lane `index` read signed or unsigned, as `signed` says; `None`
    /// for an unsigned lane 128 bits wide past what an `i128` holds.
    fn read_lane(&self, index: u32, signed: bool) -> Option<i128> {
        if signed {
            Some(self.signed_lane(index))
        } else {
            i128::try_from(self.lane(index)).ok()
        }
    }

    /// Returns lane `index` as a float of `format`, `None` where the lanes are
    /// not that wide.
    fn float_lane(&self, index: u32, format: FloatFormat) -> Option<FloatBits> {
        (format.bits() == self.lane_bits())
            .then(|| FloatBits::new(format, self.lane(index)))
            .flatten()
    }

    /// Sets lane `index` to the low lane bits of `value`.
    fn set_lane(&mut self, index: u32, value: u128) {
        if index < self.lanes() {
            self.set_bits(
                index.saturating_mul(self.lane_bits()),
                self.lane_bits(),
                value,
            );
        }
    }

    /// Returns the vector with lane `index` replaced by `value`; `None` for a
    /// lane past the vector.
    #[must_use]
    pub fn with_lane(&self, index: u32, value: u128) -> Option<Self> {
        if index >= self.computable()?.lanes() {
            return None;
        }
        let mut vector = self.clone();
        vector.set_lane(index, value);
        Some(vector)
    }

    /// Returns the lanes, lane 0 first.
    pub fn lane_values(&self) -> impl Iterator<Item = u128> + '_ {
        (0..self.lanes()).map(|index| self.lane(index))
    }

    /// Returns the same bits laid out as `lanes` lanes `lane_bits` wide: cut
    /// where the new shape is narrower, zero above the old bits where it is
    /// wider.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::ir::lanes::VectorConst;
    ///
    /// let words = VectorConst::from_lanes(32, [0x0403_0201, 0x0807_0605]).unwrap();
    /// let bytes = words.resized(8, 8).unwrap();
    /// assert_eq!(bytes.lane(5), 6);
    /// ```
    #[must_use]
    pub fn resized(&self, lane_bits: u32, lanes: u32) -> Option<Self> {
        let mut vector = Self::zeroed(lane_bits, lanes)?;
        let copied = vector.bits().min(self.bits());
        let mut start = 0u32;
        while start < copied {
            let width = copied.saturating_sub(start).min(64);
            vector.set_bits(start, width, self.bits_at(start, width));
            start = start.saturating_add(64);
        }
        Some(vector)
    }

    /// Returns a bit slice in the requested lane layout, zeroing bits outside
    /// the source. The offset is unsigned and measured in bits.
    ///
    /// Returns `None` for an invalid or oversized destination shape.
    #[must_use]
    pub fn slice_bits(&self, offset: u64, lane_bits: u32, lanes: u32) -> Option<Self> {
        let mut result = Self::zeroed(lane_bits, lanes)?;
        for at in (0..result.bits()).step_by(128) {
            let value = offset
                .checked_add(u64::from(at))
                .and_then(|start| u32::try_from(start).ok())
                .map_or(0, |start| self.bits_at(start, 128));
            result.set_bits(at, 128, value);
        }
        Some(result)
    }

    /// Returns these bits with `value` inserted at the unsigned bit offset.
    ///
    /// Bits outside this vector are discarded, and every other bit and the
    /// original lane layout are preserved.
    #[must_use]
    pub fn insert_bits(&self, offset: u64, value: &Self) -> Self {
        let mut result = self.clone();
        let Ok(offset) = u32::try_from(offset) else {
            return result;
        };
        for at in (0..value.bits()).step_by(128) {
            let Some(start) = offset.checked_add(at).filter(|start| *start < self.bits()) else {
                break;
            };
            let width = value.bits().saturating_sub(at).min(128);
            result.set_bits(start, width, value.bits_at(at, width));
        }
        result
    }

    /// Returns the concatenated parts in the requested lane layout, with the
    /// first part in the low bits.
    ///
    /// Returns `None` for no parts, an invalid destination shape, or a result
    /// width different from the sum of the part widths.
    #[must_use]
    pub fn concatenate(parts: &[Self], lane_bits: u32, lanes: u32) -> Option<Self> {
        if parts.is_empty() {
            return None;
        }
        let bits = parts
            .iter()
            .try_fold(0u32, |sum, part| sum.checked_add(part.bits()))?;
        let mut result = Self::zeroed(lane_bits, lanes)?;
        if result.bits() != bits {
            return None;
        }
        let mut offset = 0u32;
        for part in parts {
            for at in (0..part.bits()).step_by(128) {
                let width = part.bits().saturating_sub(at).min(128);
                result.set_bits(offset.saturating_add(at), width, part.bits_at(at, width));
            }
            offset = offset.checked_add(part.bits())?;
        }
        Some(result)
    }

    /// Returns the same bits laid out in lanes `lane_bits` wide, `None` where
    /// the vector is no whole number of them.
    #[must_use]
    pub fn relaned(&self, lane_bits: u32) -> Option<Self> {
        if lane_bits == 0 || !self.bits().is_multiple_of(lane_bits) {
            return None;
        }
        self.resized(lane_bits, self.bits().checked_div(lane_bits)?)
    }

    /// Returns whether every bit is clear.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.limbs().iter().all(|limb| *limb == 0)
    }

    /// Returns whether every bit is set.
    #[must_use]
    pub fn is_all_ones(&self) -> bool {
        let bits = self.bits();
        self.limbs().iter().enumerate().all(|(at, limb)| {
            let used =
                bits.saturating_sub(u32::try_from(at).unwrap_or(u32::MAX).saturating_mul(64));
            let ones = BitWidth::new(used).mask();
            *limb == ones
        })
    }

    /// Returns the lanes `f` maps each lane to, the lane width unchanged; `None`
    /// where `f` gives none for a lane.
    fn map(&self, f: impl Fn(u128) -> Option<u128>) -> Option<Self> {
        let lanes = self.lane_values().map(f).collect::<Option<Vec<_>>>()?;
        Self::from_lanes(self.lane_bits(), lanes)
    }

    /// Returns the bits `f` combines the bits of `self` and `other` into,
    /// limb by limb; `None` where the two differ in width.
    fn bitwise(&self, other: &Self, f: impl Fn(u64, u64) -> u64) -> Option<Self> {
        if self.bits() != other.bits() {
            return None;
        }
        let mut vector = self.clone();
        for (limb, other) in vector.limbs_mut().iter_mut().zip(other.limbs().iter()) {
            *limb = f(*limb, *other);
        }
        vector.clear_above();
        Some(vector)
    }

    /// Clears the bits of the last limb above the vector.
    fn clear_above(&mut self) {
        let used = self.bits() % 64;
        if used != 0
            && let Some(last) = self.limbs_mut().last_mut()
        {
            *last &= BitWidth::new(used).mask();
        }
    }

    /// Returns the bitwise and of two vectors of one width.
    #[must_use]
    pub fn and(&self, other: &Self) -> Option<Self> {
        self.bitwise(other, |a, b| a & b)
    }

    /// Returns the bitwise or of two vectors of one width.
    #[must_use]
    pub fn or(&self, other: &Self) -> Option<Self> {
        self.bitwise(other, |a, b| a | b)
    }

    /// Returns the bitwise exclusive or of two vectors of one width.
    #[must_use]
    pub fn xor(&self, other: &Self) -> Option<Self> {
        self.bitwise(other, |a, b| a ^ b)
    }

    /// Returns every bit inverted.
    #[must_use]
    pub fn not(&self) -> Self {
        let mut vector = self.clone();
        for limb in vector.limbs_mut().iter_mut() {
            *limb = !*limb;
        }
        vector.clear_above();
        vector
    }
}

/// The lanes in order, lane 0 first: `<1, 2, 3, 4>` for four lanes.
impl fmt::Display for VectorConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<")?;
        for (index, lane) in self.lane_values().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{lane:#x}")?;
        }
        write!(f, ">")
    }
}

/// The shape and the lanes: `VectorConst(32x4 <0x1, 0x2, 0x3, 0x4>)`.
impl fmt::Debug for VectorConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VectorConst({}x{} {self})",
            self.lane_bits(),
            self.lanes()
        )
    }
}

impl<T: Target> Emit<T> for VectorConst {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self { words } = self;
        Emit::emit(words, writer)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    #[cfg(feature = "serde")]
    use serde::de::{self, DeserializeSeed, SeqAccess, value::Error};

    #[cfg(feature = "serde")]
    use crate::ir::lanes::{MAX_VECTOR_WORDS, VectorWords};
    use crate::{
        ir::{
            float::{
                DefaultNan, FloatFormat, FloatRounding, FloatRules, NanEncoding, NanOperands,
                NanPropagation,
            },
            lanes::{MAX_VECTOR_BITS, VectorConst},
            ops::{
                ComplexProduct, ComplexRotation, ComplexShape, DotShape, FloatDotRule,
                FloatElements, FloatSource, VectorBinaryKind, VectorBitmaskKind, VectorCastKind,
                VectorCompareKind, VectorElement, VectorElementKind, VectorFixedKind,
                VectorReduceKind, VectorTernaryKind, VectorUnaryKind,
            },
        },
        target::VectorShuffleLane::{Left, Right, Undef, Zero},
    };

    /// An unending source with a tripwire before reads beyond the admitted prefix.
    struct LaneSource {
        pulls: Rc<Cell<usize>>,
        maximum_pulls: usize,
        hint: usize,
    }

    impl Iterator for LaneSource {
        type Item = u128;

        fn next(&mut self) -> Option<Self::Item> {
            let count = self.pulls.get() + 1;
            self.pulls.set(count);
            assert!(
                count <= self.maximum_pulls,
                "read beyond the vector lane bound"
            );
            Some(u128::MAX)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.hint, Some(self.hint))
        }
    }

    /// Constructors reject bad widths before source reads and cap every lane source.
    #[test]
    fn vector_constructors_bound_source_reads_before_collecting() {
        for width in [
            0,
            1,
            2,
            3,
            7,
            8,
            64,
            128,
            2048,
            MAX_VECTOR_BITS,
            MAX_VECTOR_BITS + 1,
            u32::MAX,
        ] {
            let maximum = MAX_VECTOR_BITS.checked_div(width).unwrap_or(0);
            let expected = if maximum == 0 {
                0
            } else {
                maximum as usize + 1
            };
            for hint in [0, 3, usize::MAX] {
                let pulls = Rc::new(Cell::new(0));
                let source = LaneSource {
                    pulls: Rc::clone(&pulls),
                    maximum_pulls: expected,
                    hint,
                };
                assert!(VectorConst::from_lanes(width, source).is_none());
                assert_eq!(pulls.get(), expected);
            }
        }
    }

    /// A vector is all ones where every bit of every lane is set, across
    /// words and in a last word it fills only partly, and not where one bit
    /// is clear.
    #[test]
    fn a_vector_is_all_ones_where_every_bit_is_set() {
        for (width, count) in [(1, 1), (3, 5), (7, 19), (64, 2), (128, 3), (8, 1024)] {
            let ones = VectorConst::splat(width, count, u128::MAX).expect("ones");
            assert!(ones.is_all_ones(), "{width}x{count}");
            assert!(!ones.is_zero(), "{width}x{count}");
            let lanes = (0..count).map(|lane| {
                let full = u128::MAX >> (128 - width);
                if lane == count - 1 { full - 1 } else { full }
            });
            let one_clear = VectorConst::from_lanes(width, lanes).expect("lanes");
            assert!(!one_clear.is_all_ones(), "{width}x{count}");
        }
        assert!(!VectorConst::zeroed(32, 4).expect("zeros").is_all_ones());
    }

    /// Repeated lanes preserve shapes and bit patterns at the full vector bound.
    #[test]
    fn vector_splats_fill_their_exact_admitted_storage() {
        for width in [1, 3, 7, 8, 64, 127, 128, 256, 2048, MAX_VECTOR_BITS] {
            let count = MAX_VECTOR_BITS / width;
            let vector = VectorConst::splat(width, count, u128::MAX).expect("bounded splat");
            assert_eq!(vector.lanes(), count);
            assert_eq!(vector.lane_bits(), width);
            let expected = if width >= 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            assert_eq!(vector.lane(0), expected);
            assert_eq!(vector.lane(count - 1), expected);
            assert_eq!(
                vector,
                VectorConst::from_lanes(width, (0..count).map(|_| u128::MAX))
                    .expect("bounded lanes")
            );
        }
        for (width, count) in [
            (0, 1),
            (1, 0),
            (1, u32::MAX),
            (MAX_VECTOR_BITS + 1, 1),
            (u32::MAX, u32::MAX),
        ] {
            assert!(VectorConst::splat(width, count, u128::MAX).is_none());
        }
    }

    /// Serialized storage obeys the same shape and padding bounds as constructors.
    #[cfg(feature = "serde")]
    #[test]
    fn serialized_vector_storage_is_bounded_and_canonical() {
        for width in 1..=MAX_VECTOR_BITS {
            for count in [1, MAX_VECTOR_BITS / width] {
                let value = VectorConst::zeroed(width, count).expect("valid shape");
                let encoded = serde_json::to_string(&value).expect("serialize");
                let decoded: VectorConst = serde_json::from_str(&encoded).expect("deserialize");
                assert_eq!(decoded, value);
                assert_eq!(
                    decoded.words.len(),
                    usize::try_from(value.bits().div_ceil(64) + 1).unwrap()
                );
            }
        }
        // The representation stays compatible with the struct's words sequence.
        let old = r#"{"words":[4294967304,90]}"#;
        let value: VectorConst = serde_json::from_str(old).expect("existing encoding");
        assert_eq!(value, VectorConst::from_lanes(8, [90]).unwrap());
        for words in [
            vec![],
            vec![0],
            vec![8, 0],
            vec![1u64 << 32, 0],
            vec![(1u64 << 32) | u64::from(MAX_VECTOR_BITS + 1), 0],
            vec![u64::MAX, 0],
            vec![(1u64 << 32) | 8],
            vec![(1u64 << 32) | 8, 256],
            vec![(1u64 << 32) | 8, 90, 0],
        ] {
            let input = serde_json::json!({"words": words});
            assert!(
                serde_json::from_value::<VectorConst>(input.clone()).is_err(),
                "{input}"
            );
        }
    }

    /// A hostile sequence advertises any length and counts every requested word.
    #[cfg(feature = "serde")]
    struct WordSource {
        header: u64,
        advertised: usize,
        pulls: Rc<Cell<usize>>,
    }

    #[cfg(feature = "serde")]
    impl<'de> SeqAccess<'de> for WordSource {
        type Error = Error;

        fn next_element_seed<T: DeserializeSeed<'de>>(
            &mut self,
            seed: T,
        ) -> Result<Option<T::Value>, Error> {
            let index = self.pulls.get();
            self.pulls.set(index + 1);
            seed.deserialize(de::value::U64Deserializer::new(if index == 0 {
                self.header
            } else {
                0
            }))
            .map(Some)
        }

        fn size_hint(&self) -> Option<usize> {
            Some(self.advertised)
        }
    }

    /// Storage validation stops an infinite input after the shape and one excess word.
    #[cfg(feature = "serde")]
    #[test]
    fn vector_deserialization_does_not_trust_or_collect_sequence_lengths() {
        for advertised in [0, 2, usize::MAX] {
            for (header, expected) in [
                (0, 1),
                (u64::MAX, 1),
                ((1u64 << 32) | 8, 3),
                ((1u64 << 32) | 2048, 34),
                (
                    (1u64 << 32) | u64::from(MAX_VECTOR_BITS),
                    MAX_VECTOR_WORDS + 2,
                ),
            ] {
                let pulls = Rc::new(Cell::new(0));
                let source = WordSource {
                    header,
                    advertised,
                    pulls: Rc::clone(&pulls),
                };
                let result = serde::de::Visitor::visit_seq(VectorWords, source);
                assert!(result.is_err());
                assert_eq!(pulls.get(), expected);
            }
        }
    }

    /// The rounding the host's arithmetic takes, which the oracles here
    /// compute by.
    const NEAREST: Option<FloatRounding> = Some(FloatRounding::NearestEven);

    /// A target that states no float rule.
    const RULES: FloatRules = FloatRules {
        out_of_range: None,
        default_nan: None,
        nan_conversion: None,
        nan_operands: None,
        extended_nan_operands: None,

        program_flushes_subnormals: false,

        nan_encoding: NanEncoding::Ieee,
    };

    fn lanes(bits: u32, values: &[u128]) -> VectorConst {
        VectorConst::from_lanes(bits, values.iter().copied()).expect("a vector")
    }

    fn integer(signed: Option<bool>) -> VectorElement {
        VectorElement {
            kind: VectorElementKind::Integer,
            bits: 0,
            scalar: false,
            signed,
            rounding: None,
        }
    }

    fn float(bits: u32) -> VectorElement {
        VectorElement {
            kind: VectorElementKind::Float,
            bits,
            scalar: false,
            signed: None,
            rounding: None,
        }
    }

    /// **Xeon Phi's fix-up with NaN passthrough gives each class its token.**
    /// The table names a token per class (327364-001, VFIXUPNANPS): minus
    /// infinity takes the largest finite value (6), a negative number plus
    /// infinity (4), minus zero minus infinity (1), plus zero the default NaN
    /// (5), a positive number nothing (0, the destination's lane) and plus
    /// infinity the negated largest (7). A NaN passes through quieted, a
    /// subnormal folds nothing (MXCSR.DAZ decides it), and a signalling NaN
    /// raises. Expected bits from Rust's `f32`.
    #[test]
    fn a_fixup_passing_nans_gives_each_class_its_token() {
        let table: u128 = 6 | 4 << 3 | 1 << 6 | 5 << 9 | 7 << 15;
        let old = 0x1234_5678_u128;
        let fold = |source: f32| {
            lanes(32, &[old])
                .ternary(
                    &lanes(32, &[u128::from(source.to_bits())]),
                    &lanes(32, &[table]),
                    VectorTernaryKind::FixupNan,
                    float(32),
                    None,
                    FloatRules::default(),
                )
                .map(|vector| vector.lane(0))
        };
        let bits = |value: f32| Some(u128::from(value.to_bits()));
        assert_eq!(fold(f32::NEG_INFINITY), bits(f32::MAX));
        assert_eq!(fold(-2.5), bits(f32::INFINITY));
        assert_eq!(fold(-0.0), bits(f32::NEG_INFINITY));
        assert_eq!(fold(0.0), Some(0xffc0_0000));
        assert_eq!(fold(2.5), Some(old));
        assert_eq!(fold(f32::INFINITY), bits(-f32::MAX));
        assert_eq!(fold(f32::from_bits(0x7f80_0001)), Some(0x7fc0_0001));
        assert_eq!(fold(f32::from_bits(0xffc0_1234)), Some(0xffc0_1234));
        assert_eq!(fold(f32::from_bits(0x0000_0001)), None);
        let double = |source: f64| {
            lanes(64, &[old])
                .ternary(
                    &lanes(64, &[u128::from(source.to_bits())]),
                    &lanes(64, &[table]),
                    VectorTernaryKind::FixupNan,
                    float(64),
                    None,
                    FloatRules::default(),
                )
                .map(|vector| vector.lane(0))
        };
        assert_eq!(
            double(f64::NEG_INFINITY),
            Some(u128::from(f64::MAX.to_bits()))
        );
        assert_eq!(double(0.0), Some(0xfff8_0000_0000_0000));
    }

    /// **A vector is its bits.** Lanes cross the limbs, a lane of 128 bits
    /// is whole, and a reshape reads the same bits at another width, cut or
    /// zero-extended.
    #[test]
    fn an_integer_fold_reads_signedness_where_its_kind_says() {
        // Lanes whose signed and unsigned readings differ: the minimum, a
        // negative one, the maximum and a small one, against a negative, a
        // small, a negative and the minimum.
        let left = lanes(16, &[0x8000, 0xfff0, 0x7fff, 0x0003]);
        let right = lanes(16, &[0xfff0, 0x0002, 0x8001, 0x8000]);
        let fold =
            |kind, signed| left.binary(&right, kind, integer(signed), None, FloatRules::default());
        let mut wrong = Vec::new();
        for number in 0..=u16::MAX {
            let Ok(kind) = VectorBinaryKind::try_from(number) else {
                continue;
            };
            let readings = [
                fold(kind, Some(true)),
                fold(kind, Some(false)),
                fold(kind, None),
            ];
            let reads = readings[0] != readings[1] || readings[0] != readings[2];
            if reads != kind.reads_signedness() {
                wrong.push(format!("{kind:?}: {readings:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    #[test]
    fn a_vector_is_its_bits_in_any_lanes() {
        let wide = lanes(128, &[0x0123_4567_89ab_cdef_fedc_ba98_7654_3210, u128::MAX]);
        assert_eq!(wide.bits(), 256);
        assert_eq!(wide.lane(0), 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
        let words = wide.relaned(32).expect("eight words");
        assert_eq!(words.lane(0), 0x7654_3210);
        assert_eq!(words.lane(3), 0x0123_4567);
        assert_eq!(words.lane(4), 0xffff_ffff);
        // A 48-bit lane straddles the first two limbs.
        assert_eq!(wide.bits_at(40, 48), 0xabcd_effe_dcba);
        let cut = wide.resized(16, 3).expect("three halves");
        assert_eq!(
            cut.lane_values().collect::<Vec<_>>(),
            [0x3210, 0x7654, 0xba98]
        );
        let grown = lanes(8, &[0xff]).resized(8, 3).expect("zeros above");
        assert_eq!(grown.lane_values().collect::<Vec<_>>(), [0xff, 0, 0]);
        assert_eq!(
            lanes(8, &[1, 2, 3]).relaned(16),
            None,
            "24 bits are no halves"
        );
        assert_eq!(VectorConst::zeroed(512, 1).map(|v| v.bits()), Some(512));
        assert_eq!(VectorConst::zeroed(0, 1), None);
        assert_eq!(
            VectorConst::zeroed(64, MAX_VECTOR_BITS / 64 + 1),
            None,
            "wider than the widest constant"
        );
        assert!(VectorConst::zeroed(32, 4).expect("zeros").is_zero());
        assert_eq!(
            lanes(4, &[0xf, 0x3])
                .not()
                .lane_values()
                .collect::<Vec<_>>(),
            [0, 0xc]
        );
    }

    /// **Integer lanes compute at their width.** Every value below is the
    /// hardware's (x86 `paddsb`, `pavgb`, `pmulhw`, `psadbw`'s absolute
    /// difference; NEON `sqdmulh`, `sqrdmulh`).
    #[test]
    fn integer_multiply_add_matches_complete_byte_domains_and_wide_accumulators() {
        for widening in [false, true] {
            let result_bits = if widening { 16u32 } else { 8 };
            let mask = (1u128 << result_bits) - 1;
            let batch = 2048 / result_bits;
            for signed_left in [false, true] {
                for signed_right in [false, true] {
                    let decode = |value: u128, signed| {
                        if signed {
                            i128::from(value as u8 as i8)
                        } else {
                            value as i128
                        }
                    };
                    for first in (0..256u128).step_by(batch as usize) {
                        let left =
                            lanes(8, &(first..first + u128::from(batch)).collect::<Vec<_>>());
                        for right in 0..256u128 {
                            for addend in [0, 1, mask / 2, mask / 2 + 1, mask] {
                                let addends = lanes(result_bits, &vec![addend; batch as usize]);
                                for subtract in [false, true] {
                                    let actual = left
                                        .integer_multiply_add(
                                            &lanes(8, &[right]),
                                            &addends,
                                            result_bits,
                                            (signed_left, signed_right),
                                            subtract,
                                        )
                                        .unwrap();
                                    for index in 0..batch {
                                        let a = first + u128::from(index);
                                        let product =
                                            decode(a, signed_left) * decode(right, signed_right);
                                        let exact = if subtract {
                                            addend as i128 - product
                                        } else {
                                            addend as i128 + product
                                        };
                                        assert_eq!(
                                            actual.lane(index),
                                            exact as u128 & mask,
                                            "wide={widening} sign={signed_left}/{signed_right}, a={a}, b={right}, c={addend}, subtract={subtract}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn integer_multiply_add_checks_all_intermediate_widths_by_signed_magnitudes() {
        for bits in [16u32, 32, 64] {
            let source_modulus = 1u128 << bits;
            let sign_bit = source_modulus / 2;
            let source_mask = source_modulus - 1;
            let inputs = [
                0,
                1,
                sign_bit - 1,
                sign_bit,
                sign_bit + 1,
                source_mask - 1,
                source_mask,
            ];
            for widening in [false, true] {
                if widening && bits == 64 {
                    continue;
                }
                let result_bits = bits * if widening { 2 } else { 1 };
                let modulus = 1u128 << result_bits;
                for signed_left in [false, true] {
                    for signed_right in [false, true] {
                        for a in inputs {
                            for b in inputs {
                                let negative_left = signed_left && a >= sign_bit;
                                let negative_right = signed_right && b >= sign_bit;
                                let magnitude_left =
                                    if negative_left { source_modulus - a } else { a };
                                let magnitude_right = if negative_right {
                                    source_modulus - b
                                } else {
                                    b
                                };
                                let magnitude = magnitude_left * magnitude_right % modulus;
                                let product = if negative_left != negative_right && magnitude != 0 {
                                    modulus - magnitude
                                } else {
                                    magnitude
                                };
                                for c in [0, modulus / 2, modulus - 1] {
                                    for subtract in [false, true] {
                                        let expected = if subtract {
                                            (c + modulus - product) % modulus
                                        } else {
                                            (c + product) % modulus
                                        };
                                        let actual = lanes(bits, &[a])
                                            .integer_multiply_add(
                                                &lanes(bits, &[b]),
                                                &lanes(result_bits, &[c]),
                                                result_bits,
                                                (signed_left, signed_right),
                                                subtract,
                                            )
                                            .unwrap();
                                        assert_eq!(
                                            actual.lane(0),
                                            expected,
                                            "bits={bits}, widen={widening}, a={a}, b={b}, c={c}, signs={signed_left}/{signed_right}, subtract={subtract}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn integer_multiply_add_wraps_full_128_bit_products_and_accumulators() {
        let maximum = u128::from(u64::MAX);
        let minimum = 1u128 << 63;
        for (a, b, c, signs, subtract, expected) in [
            (
                maximum,
                maximum,
                0,
                (false, false),
                false,
                0xffff_ffff_ffff_fffe_0000_0000_0000_0001u128,
            ),
            (
                maximum,
                maximum,
                u128::MAX,
                (false, false),
                false,
                0xffff_ffff_ffff_fffe_0000_0000_0000_0000,
            ),
            (
                maximum,
                maximum,
                0,
                (false, false),
                true,
                0x0000_0000_0000_0001_ffff_ffff_ffff_ffff,
            ),
            (
                minimum,
                minimum,
                0,
                (true, true),
                false,
                0x4000_0000_0000_0000_0000_0000_0000_0000,
            ),
            (
                minimum,
                maximum,
                0,
                (true, false),
                false,
                0x8000_0000_0000_0000_8000_0000_0000_0000,
            ),
            (
                maximum,
                minimum,
                0,
                (false, true),
                false,
                0x8000_0000_0000_0000_8000_0000_0000_0000,
            ),
            (
                maximum,
                maximum,
                0,
                (true, false),
                false,
                0xffff_ffff_ffff_ffff_0000_0000_0000_0001,
            ),
            (
                maximum,
                maximum,
                0,
                (false, true),
                true,
                0x0000_0000_0000_0000_ffff_ffff_ffff_ffff,
            ),
        ] {
            let actual = lanes(64, &[a])
                .integer_multiply_add(&lanes(64, &[b]), &lanes(128, &[c]), 128, signs, subtract)
                .unwrap();
            assert_eq!(
                actual.lane(0),
                expected,
                "{a:x} {b:x} {c:x} {signs:?} {subtract}"
            );
        }
        for signs in [(false, false), (true, false), (false, true), (true, true)] {
            assert_eq!(
                lanes(128, &[u128::MAX])
                    .integer_multiply_add(
                        &lanes(128, &[u128::MAX]),
                        &lanes(128, &[u128::MAX]),
                        128,
                        signs,
                        false
                    )
                    .unwrap()
                    .lane(0),
                0
            );
            assert_eq!(
                lanes(128, &[1 << 127])
                    .integer_multiply_add(&lanes(128, &[2]), &lanes(128, &[123]), 128, signs, false)
                    .unwrap()
                    .lane(0),
                123
            );
        }
        assert!(
            lanes(8, &[1])
                .integer_multiply_add(&lanes(16, &[1]), &lanes(16, &[1]), 16, (true, true), false)
                .is_none()
        );
        assert!(
            lanes(8, &[1])
                .integer_multiply_add(&lanes(8, &[1]), &lanes(32, &[1]), 32, (true, true), false)
                .is_none()
        );
        assert!(
            lanes(8, &[1, 2])
                .integer_multiply_add(
                    &lanes(8, &[1, 2, 3]),
                    &lanes(16, &[1, 2]),
                    16,
                    (true, true),
                    false
                )
                .is_none()
        );
        assert!(
            lanes(8, &[1, 2])
                .integer_multiply_add(&lanes(8, &[1]), &lanes(16, &[1]), 16, (true, true), false)
                .is_none()
        );
    }

    /// Uses Euclidean division and distances to the two neighboring integers.
    fn fixed_reference(value: i128, amount: u32, mode: u8) -> i128 {
        let denominator = 1i128 << amount;
        let lower = value.div_euclid(denominator);
        let distance = value.rem_euclid(denominator);
        match mode {
            0 => lower + i128::from(distance * 2 >= denominator),
            1 => {
                lower
                    + i128::from(
                        distance * 2 > denominator || distance * 2 == denominator && lower % 2 != 0,
                    )
            }
            2 => lower,
            3 if distance != 0 => lower | 1,
            _ => lower,
        }
    }

    #[test]
    fn fixed_point_average_and_fractional_product_exhaust_all_byte_pairs() {
        let source = lanes(8, &(0..=255).collect::<Vec<_>>());
        for kind in [
            VectorFixedKind::AverageAdd,
            VectorFixedKind::AverageSub,
            VectorFixedKind::FractionalMultiply,
        ] {
            for signed in [false, true] {
                if kind == VectorFixedKind::FractionalMultiply && !signed {
                    continue;
                }
                let decode = |value: u128| {
                    if signed {
                        i128::from(value as u8 as i8)
                    } else {
                        value as i128
                    }
                };
                for b in 0..=255u128 {
                    for mode in 0..4 {
                        let (actual, saturated) = source
                            .fixed_binary(
                                &lanes(8, &[b]),
                                kind,
                                VectorElement::integer(8, Some(signed)),
                                mode,
                            )
                            .unwrap();
                        let mut expected_saturated = false;
                        for a in 0..=255u128 {
                            let (exact, amount) = match kind {
                                VectorFixedKind::AverageAdd => (decode(a) + decode(b), 1),
                                VectorFixedKind::AverageSub => {
                                    let difference = decode(a) - decode(b);
                                    (
                                        if signed {
                                            difference
                                        } else {
                                            difference.rem_euclid(512)
                                        },
                                        1,
                                    )
                                }
                                _ => (decode(a) * decode(b), 7),
                            };
                            let rounded = fixed_reference(exact, amount, mode);
                            let expected = if kind == VectorFixedKind::FractionalMultiply {
                                expected_saturated |= !(-128..=127).contains(&rounded);
                                rounded.clamp(-128, 127)
                            } else {
                                rounded
                            };
                            assert_eq!(
                                actual.lane(a as u32),
                                expected as u128 & 255,
                                "{kind:?} signed={signed} a={a} b={b} mode={mode}"
                            );
                        }
                        assert_eq!(saturated, expected_saturated);
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_point_shifts_and_clips_match_rational_rounding_before_saturation() {
        for clip in [false, true] {
            let source_bits = if clip { 16 } else { 8 };
            let limit = 1u128 << source_bits;
            let batch = 2048 / source_bits;
            for signed in [false, true] {
                for first in (0..limit).step_by(batch as usize) {
                    let values: Vec<_> = (first..first + u128::from(batch)).collect();
                    let source = lanes(source_bits, &values);
                    for count in [0u128, 1, 2, 7, 8, 15, 16, 31, u128::MAX] {
                        for mode in 0..4 {
                            let kind = if clip {
                                VectorFixedKind::Clip
                            } else {
                                VectorFixedKind::Shift
                            };
                            let (actual, saturated) = source
                                .fixed_binary(
                                    &lanes(128, &[count]),
                                    kind,
                                    VectorElement::integer(8, Some(signed)),
                                    mode,
                                )
                                .unwrap();
                            let mut expected_saturated = false;
                            for (index, value) in values.iter().enumerate() {
                                let exact = if signed && value & (limit / 2) != 0 {
                                    *value as i128 - limit as i128
                                } else {
                                    *value as i128
                                };
                                let rounded = fixed_reference(
                                    exact,
                                    (count % u128::from(source_bits)) as u32,
                                    mode,
                                );
                                let (minimum, maximum) =
                                    if signed { (-128, 127) } else { (0, 255) };
                                let expected = if clip {
                                    expected_saturated |= rounded < minimum || rounded > maximum;
                                    rounded.clamp(minimum, maximum)
                                } else {
                                    rounded
                                };
                                assert_eq!(
                                    actual.lane(index as u32),
                                    expected as u128 & 255,
                                    "clip={clip} signed={signed} value={value} count={count} mode={mode}"
                                );
                            }
                            assert_eq!(saturated, expected_saturated);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_point_wide_averages_products_and_shifts_preserve_width() {
        for bits in [16u32, 32, 64] {
            let sign = 1u128 << (bits - 1);
            let mask = (1u128 << bits) - 1;
            let values = [0, 1, sign - 1, sign, sign + 1, mask - 1, mask];
            for signed in [false, true] {
                let decode = |value: u128| {
                    if signed && value >= sign {
                        value as i128 - (1i128 << bits)
                    } else {
                        value as i128
                    }
                };
                for a in values {
                    for b in values {
                        for kind in [
                            VectorFixedKind::AverageAdd,
                            VectorFixedKind::AverageSub,
                            VectorFixedKind::FractionalMultiply,
                        ] {
                            if kind == VectorFixedKind::FractionalMultiply && !signed {
                                continue;
                            }
                            let (exact, amount) = match kind {
                                VectorFixedKind::AverageAdd => (decode(a) + decode(b), 1),
                                VectorFixedKind::AverageSub => {
                                    let value = decode(a) - decode(b);
                                    (
                                        if signed {
                                            value
                                        } else {
                                            value.rem_euclid(1i128 << (bits + 1))
                                        },
                                        1,
                                    )
                                }
                                _ => (decode(a) * decode(b), bits - 1),
                            };
                            for mode in 0..4 {
                                let rounded = fixed_reference(exact, amount, mode);
                                let minimum = -(sign as i128);
                                let maximum = sign as i128 - 1;
                                let (expected, saturated) =
                                    if kind == VectorFixedKind::FractionalMultiply {
                                        (
                                            rounded.clamp(minimum, maximum),
                                            rounded < minimum || rounded > maximum,
                                        )
                                    } else {
                                        (rounded, false)
                                    };
                                let (value, flag) = lanes(bits, &[a])
                                    .fixed_binary(
                                        &lanes(bits, &[b]),
                                        kind,
                                        VectorElement::integer(bits, Some(signed)),
                                        mode,
                                    )
                                    .unwrap();
                                assert_eq!(
                                    value.lane(0),
                                    expected as u128 & mask,
                                    "{kind:?}, bits={bits}, a={a}, b={b}, mode={mode}"
                                );
                                assert_eq!(flag, saturated);
                            }
                        }
                    }
                    for count in [
                        0u128,
                        1,
                        u128::from(bits - 1),
                        u128::from(bits),
                        u128::from(bits + 1),
                        u128::MAX,
                    ] {
                        for mode in 0..4 {
                            let (value, flag) = lanes(bits, &[a])
                                .fixed_binary(
                                    &lanes(128, &[count]),
                                    VectorFixedKind::Shift,
                                    VectorElement::integer(bits, Some(signed)),
                                    mode,
                                )
                                .unwrap();
                            assert_eq!(
                                value.lane(0),
                                fixed_reference(decode(a), (count % u128::from(bits)) as u32, mode)
                                    as u128
                                    & mask
                            );
                            assert!(!flag);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_point_widest_operands_and_malformed_domains_are_bounded() {
        for bits in [8, 16, 32, 64] {
            let minimum = 1u128 << (bits - 1);
            let (value, saturated) = lanes(bits, &[minimum])
                .fixed_binary(
                    &lanes(bits, &[minimum]),
                    VectorFixedKind::FractionalMultiply,
                    VectorElement::integer(bits, Some(true)),
                    0,
                )
                .unwrap();
            assert_eq!(value.lane(0), minimum - 1);
            assert!(saturated);
        }
        for signed in [false, true] {
            for mode in 0..4 {
                let source = if signed { i128::MAX as u128 } else { u128::MAX };
                let (value, saturated) = lanes(128, &[source])
                    .fixed_binary(
                        &lanes(64, &[64]),
                        VectorFixedKind::Clip,
                        VectorElement::integer(64, Some(signed)),
                        mode,
                    )
                    .unwrap();
                assert_eq!(
                    value.lane(0),
                    if signed {
                        i64::MAX as u128
                    } else {
                        u64::MAX as u128
                    }
                );
                assert_eq!(saturated, mode < 2);
            }
        }
        for bits in [0, 1, 7, 24, 128] {
            assert!(
                lanes(8, &[1])
                    .fixed_binary(
                        &lanes(8, &[2]),
                        VectorFixedKind::AverageAdd,
                        VectorElement::integer(bits, Some(true)),
                        0
                    )
                    .is_none()
            );
        }
        assert!(
            lanes(8, &[1])
                .fixed_binary(
                    &lanes(8, &[2]),
                    VectorFixedKind::AverageAdd,
                    VectorElement::integer(8, Some(true)),
                    4
                )
                .is_none()
        );
        assert!(
            lanes(8, &[1])
                .fixed_binary(
                    &lanes(8, &[2]),
                    VectorFixedKind::FractionalMultiply,
                    VectorElement::integer(8, Some(false)),
                    0
                )
                .is_none()
        );
        assert!(
            lanes(8, &[1])
                .fixed_binary(
                    &lanes(8, &[2]),
                    VectorFixedKind::Clip,
                    VectorElement::integer(8, Some(true)),
                    0
                )
                .is_none()
        );
        assert!(
            lanes(8, &[1, 2, 3])
                .fixed_binary(
                    &lanes(8, &[2, 3]),
                    VectorFixedKind::AverageAdd,
                    VectorElement::integer(8, Some(true)),
                    0
                )
                .is_none()
        );
    }

    #[test]
    fn saturating_integer_lanes_match_independent_wide_arithmetic() {
        for bits in [8u32, 16, 32, 64] {
            let mask = (1u128 << bits) - 1;
            let values: Vec<_> = if bits == 8 {
                (0..=255).collect()
            } else {
                vec![
                    0,
                    1,
                    2,
                    (1u128 << (bits - 1)) - 2,
                    (1u128 << (bits - 1)) - 1,
                    1u128 << (bits - 1),
                    (1u128 << (bits - 1)) + 1,
                    mask - 1,
                    mask,
                ]
            };
            for signed in [false, true] {
                let decode = |value: u128| {
                    if signed && value & (1u128 << (bits - 1)) != 0 {
                        value as i128 - (1i128 << bits)
                    } else {
                        value as i128
                    }
                };
                let minimum = if signed { -(1i128 << (bits - 1)) } else { 0 };
                let maximum = if signed {
                    (1i128 << (bits - 1)) - 1
                } else {
                    mask as i128
                };
                for a in &values {
                    for b in &values {
                        for subtract in [false, true] {
                            let mathematical = if subtract {
                                decode(*a) - decode(*b)
                            } else {
                                decode(*a) + decode(*b)
                            };
                            let expected = if mathematical < minimum {
                                minimum
                            } else if mathematical > maximum {
                                maximum
                            } else {
                                mathematical
                            } as u128
                                & mask;
                            let kind = if subtract {
                                VectorBinaryKind::SatSub
                            } else {
                                VectorBinaryKind::SatAdd
                            };
                            let result = lanes(bits, &[*a])
                                .binary(
                                    &lanes(bits, &[*b]),
                                    kind,
                                    VectorElement::integer(bits, Some(signed)),
                                    None,
                                    FloatRules::default(),
                                )
                                .unwrap();
                            assert_eq!(
                                result.lane(0),
                                expected,
                                "bits={bits}, signed={signed}, subtract={subtract}, a={a}, b={b}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn integer_lanes_compute_at_their_width() {
        let binary = |a: &[u128], b: &[u128], kind, signed| {
            lanes(8, a)
                .binary(
                    &lanes(8, b),
                    kind,
                    integer(signed),
                    None,
                    FloatRules::default(),
                )
                .map(|vector| vector.lane_values().collect::<Vec<_>>())
        };
        assert_eq!(
            binary(&[0xff, 1], &[1, 0xff], VectorBinaryKind::Add, None),
            Some(vec![0, 0])
        );
        assert_eq!(
            binary(
                &[0x7f, 0x80],
                &[1, 0xff],
                VectorBinaryKind::SatAdd,
                Some(true)
            ),
            Some(vec![0x7f, 0x80])
        );
        assert_eq!(
            binary(
                &[0xf0, 0x10],
                &[0x20, 0x20],
                VectorBinaryKind::SatAdd,
                Some(false)
            ),
            Some(vec![0xff, 0x30])
        );
        assert_eq!(
            binary(
                &[0x10, 0x00],
                &[0x20, 0x01],
                VectorBinaryKind::SatSub,
                Some(false)
            ),
            Some(vec![0, 0])
        );
        assert_eq!(
            binary(&[1, 0xff], &[2, 0xfe], VectorBinaryKind::Avg, Some(false)),
            Some(vec![2, 0xff])
        );
        assert_eq!(
            binary(&[0x80, 3], &[0x7f, 5], VectorBinaryKind::Min, Some(true)),
            Some(vec![0x80, 3])
        );
        assert_eq!(
            binary(&[0x80, 3], &[0x7f, 5], VectorBinaryKind::Min, Some(false)),
            Some(vec![0x7f, 3])
        );
        assert_eq!(
            binary(&[0x80], &[0x7f], VectorBinaryKind::Min, None),
            None,
            "no signedness"
        );
        assert_eq!(
            binary(&[3, 0x10], &[10, 1], VectorBinaryKind::AbsDiff, Some(false)),
            Some(vec![7, 0x0f])
        );
        // |-128| is the greatest magnitude; -3 against 3, 2 against -2 and
        // 127 against -127 tie, which keeps the second source; the element's
        // signedness is not read.
        for signed in [None, Some(false), Some(true)] {
            assert_eq!(
                binary(
                    &[0x80, 0xfd, 2, 0x7f],
                    &[0x7f, 3, 0xfe, 0x81],
                    VectorBinaryKind::MaxMagnitude,
                    signed
                ),
                Some(vec![0x80, 3, 0xfe, 0x81]),
                "{signed:?}"
            );
            assert_eq!(
                binary(
                    &[0x80, 0xfd, 2, 0x7f],
                    &[0x7f, 3, 0xfe, 0x81],
                    VectorBinaryKind::MinMagnitude,
                    signed
                ),
                Some(vec![0x7f, 3, 0xfe, 0x81]),
                "{signed:?}"
            );
        }
        assert_eq!(
            binary(&[0x81], &[1], VectorBinaryKind::Shr, Some(true)),
            Some(vec![0xc0])
        );
        assert_eq!(
            binary(&[0x81], &[1], VectorBinaryKind::Shr, Some(false)),
            Some(vec![0x40])
        );
        assert_eq!(
            binary(&[0x81], &[9], VectorBinaryKind::Shl, None),
            Some(vec![0]),
            "shifted out"
        );
        assert_eq!(
            binary(&[0x81], &[1], VectorBinaryKind::Rol, None),
            Some(vec![0x03])
        );
        assert_eq!(
            binary(&[0x81], &[1], VectorBinaryKind::Ror, None),
            Some(vec![0xc0])
        );
        assert_eq!(
            binary(&[7], &[0], VectorBinaryKind::Div, Some(false)),
            None,
            "by zero"
        );
        assert_eq!(
            binary(&[0xf9], &[2], VectorBinaryKind::Div, Some(true)),
            Some(vec![0xfd])
        );
        // pmulhw: the high half of 0x4000 * 0x4000.
        let words = |a: &[u128], b: &[u128], kind, signed| {
            lanes(16, a)
                .binary(
                    &lanes(16, b),
                    kind,
                    integer(signed),
                    None,
                    FloatRules::default(),
                )
                .map(|vector| vector.lane_values().collect::<Vec<_>>())
        };
        assert_eq!(
            words(
                &[0x4000, 0xffff],
                &[0x4000, 2],
                VectorBinaryKind::MulHigh,
                Some(true)
            ),
            Some(vec![0x1000, 0xffff])
        );
        // sqdmulh of the minimum by itself saturates.
        assert_eq!(
            words(
                &[0x8000, 0x4000],
                &[0x8000, 0x4000],
                VectorBinaryKind::MulHighDoubleSat,
                Some(true)
            ),
            Some(vec![0x7fff, 0x2000])
        );
        // sqrdmulh rounds: 2 * 0x0001 * 0x4000 = 0x8000, + 0x8000 >> 16 = 1.
        assert_eq!(
            words(
                &[1],
                &[0x4000],
                VectorBinaryKind::MulHighDoubleSatRound,
                Some(true)
            ),
            Some(vec![1])
        );
        // A variable shift's negative count shifts right.
        assert_eq!(
            words(
                &[0x8000, 1],
                &[0xff, 3],
                VectorBinaryKind::VariableShiftArithmetic,
                Some(true)
            ),
            Some(vec![0xc000, 8])
        );
        assert_eq!(
            words(
                &[0x8000],
                &[0xff],
                VectorBinaryKind::VariableShiftLogical,
                Some(false)
            ),
            Some(vec![0x4000])
        );
        // The element's lane width is the one computed in: two words as four bytes.
        let as_bytes = VectorElement {
            bits: 8,
            ..integer(None)
        };
        assert_eq!(
            lanes(16, &[0x01ff, 0x0101])
                .binary(
                    &lanes(16, &[0x0101, 0x0101]),
                    VectorBinaryKind::Add,
                    as_bytes,
                    None,
                    FloatRules::default()
                )
                .map(|vector| vector.lane_values().collect::<Vec<_>>()),
            Some(vec![0x00, 0x02, 0x02, 0x02])
        );
    }

    /// **A float minimum or maximum folds where every architecture computing
    /// it agrees.** x86's select (`minss`: `a < b ? a : b`) gives the second
    /// operand for a NaN in either and for two equal zeros; the number forms
    /// (`fminnm`, RISC-V `fmin`) give the number beside a quiet NaN and leave
    /// a signalling NaN and the sign of two zeros to the architecture; the
    /// magnitude forms (MSA `fmin_a`) give the operand of smaller or larger
    /// absolute value and leave a tie; the absolute forms (`famax`) give the
    /// absolute value itself; the NaN-propagating forms (`fmin`) and the
    /// absolute forms fold numbers only.
    #[test]
    fn a_float_extreme_folds_where_every_architecture_agrees() {
        const QUIET: u32 = 0x7fc0_0000;
        const SIGNALLING: u32 = 0x7fa0_0000;
        let single = |values: &[u32]| {
            lanes(
                32,
                &values.iter().map(|v| u128::from(*v)).collect::<Vec<_>>(),
            )
        };
        let bits = |value: f32| value.to_bits();
        let fold = |kind, a: &[u32], b: &[u32]| {
            single(a).binary(&single(b), kind, float(32), NEAREST, FloatRules::default())
        };
        let (a, b) = (
            [QUIET, bits(1.0), bits(-0.0), bits(0.5), SIGNALLING],
            [bits(3.0), QUIET, bits(0.0), bits(1.0), bits(4.0)],
        );
        assert_eq!(
            fold(VectorBinaryKind::MinLess, &a, &b),
            Some(single(&[bits(3.0), QUIET, bits(0.0), bits(0.5), bits(4.0)]))
        );
        assert_eq!(
            fold(VectorBinaryKind::MaxGreater, &a, &b),
            Some(single(&[bits(3.0), QUIET, bits(0.0), bits(1.0), bits(4.0)]))
        );
        assert_eq!(
            fold(
                VectorBinaryKind::MinNum,
                &[QUIET, bits(1.0), bits(0.5)],
                &[bits(3.0), QUIET, bits(1.0)]
            ),
            Some(single(&[bits(3.0), bits(1.0), bits(0.5)]))
        );
        assert_eq!(
            fold(VectorBinaryKind::MaxNum, &[QUIET], &[bits(3.0)]),
            Some(single(&[bits(3.0)]))
        );
        for (a, b) in [
            ([SIGNALLING], [bits(4.0)]),
            ([QUIET], [QUIET]),
            ([bits(-0.0)], [bits(0.0)]),
        ] {
            assert_eq!(
                fold(VectorBinaryKind::MinNum, &a, &b),
                None,
                "{a:x?} {b:x?}"
            );
        }
        assert_eq!(
            fold(
                VectorBinaryKind::MinMagnitude,
                &[bits(-3.0), QUIET],
                &[bits(2.0), bits(5.0)]
            ),
            Some(single(&[bits(2.0), bits(5.0)]))
        );
        assert_eq!(
            fold(VectorBinaryKind::MaxMagnitude, &[bits(-3.0)], &[bits(2.0)]),
            Some(single(&[bits(-3.0)]))
        );
        assert_eq!(
            fold(VectorBinaryKind::MinMagnitude, &[bits(-2.0)], &[bits(2.0)]),
            None,
            "a tie"
        );
        assert_eq!(
            fold(VectorBinaryKind::Min, &[bits(1.0)], &[bits(2.0)]),
            Some(single(&[bits(1.0)]))
        );
        assert_eq!(fold(VectorBinaryKind::Min, &[QUIET], &[bits(1.0)]), None);
    }

    /// **The IEEE 754-2019 minimum and maximum forms fold every pair of
    /// numbers they order.** `minimumNumber` gives the number beside a quiet
    /// or a signalling NaN and orders negative zero below positive;
    /// `minimumMagnitude` gives the operand of smaller absolute value, of
    /// `-2` and `2` the lesser, and leaves a NaN result to the architecture;
    /// `minimumMagnitudeNumber` gives the number beside a NaN. Two NaNs fold
    /// to none of them. A reduction of the lanes of one is the pairwise fold,
    /// a NaN lane losing to the numbers in a number form.
    #[test]
    fn the_ieee_2019_extremes_fold_every_pair_they_order() {
        const QUIET: u32 = 0x7fc0_0000;
        const SIGNALLING: u32 = 0x7fa0_0000;
        let single = |values: &[u32]| {
            lanes(
                32,
                &values.iter().map(|v| u128::from(*v)).collect::<Vec<_>>(),
            )
        };
        let bits = |value: f32| value.to_bits();
        let fold = |kind, a: &[u32], b: &[u32]| {
            single(a).binary(&single(b), kind, float(32), NEAREST, FloatRules::default())
        };
        let (a, b) = (
            [QUIET, bits(1.0), bits(-0.0), SIGNALLING, bits(-2.0)],
            [bits(3.0), SIGNALLING, bits(0.0), bits(4.0), bits(2.0)],
        );
        assert_eq!(
            fold(VectorBinaryKind::MinimumNumber, &a, &b),
            Some(single(&[
                bits(3.0),
                bits(1.0),
                bits(-0.0),
                bits(4.0),
                bits(-2.0)
            ]))
        );
        assert_eq!(
            fold(VectorBinaryKind::MaximumNumber, &a, &b),
            Some(single(&[
                bits(3.0),
                bits(1.0),
                bits(0.0),
                bits(4.0),
                bits(2.0)
            ]))
        );
        let (a, b) = (
            [bits(-3.0), bits(-2.0), bits(-0.0), bits(0.5)],
            [bits(2.0), bits(2.0), bits(0.0), bits(-1.0)],
        );
        assert_eq!(
            fold(VectorBinaryKind::MinimumMagnitude, &a, &b),
            Some(single(&[bits(2.0), bits(-2.0), bits(-0.0), bits(0.5)]))
        );
        assert_eq!(
            fold(VectorBinaryKind::MaximumMagnitude, &a, &b),
            Some(single(&[bits(-3.0), bits(2.0), bits(0.0), bits(-1.0)]))
        );
        // `famax` and `famin` give the absolute values themselves, the
        // second's on a tie.
        assert_eq!(
            fold(VectorBinaryKind::MaxAbs, &a, &b),
            Some(single(&[bits(3.0), bits(2.0), bits(0.0), bits(1.0)]))
        );
        assert_eq!(
            fold(VectorBinaryKind::MinAbs, &a, &b),
            Some(single(&[bits(2.0), bits(2.0), bits(0.0), bits(0.5)]))
        );
        assert_eq!(fold(VectorBinaryKind::MaxAbs, &[QUIET], &[bits(1.0)]), None);
        assert_eq!(
            fold(VectorBinaryKind::MinimumMagnitude, &[QUIET], &[bits(1.0)]),
            None,
            "a NaN result is the architecture's"
        );
        assert_eq!(
            fold(
                VectorBinaryKind::MinimumMagnitudeNumber,
                &[QUIET, bits(-5.0)],
                &[bits(1.0), SIGNALLING]
            ),
            Some(single(&[bits(1.0), bits(-5.0)]))
        );
        assert_eq!(
            fold(
                VectorBinaryKind::MaximumMagnitudeNumber,
                &[bits(-5.0)],
                &[bits(5.0)]
            ),
            Some(single(&[bits(5.0)]))
        );
        for kind in [
            VectorBinaryKind::MinimumNumber,
            VectorBinaryKind::MaximumNumber,
            VectorBinaryKind::MinimumMagnitudeNumber,
            VectorBinaryKind::MaximumMagnitudeNumber,
        ] {
            assert_eq!(fold(kind, &[QUIET], &[SIGNALLING]), None, "{kind:?}");
        }
        let float_lanes = single(&[bits(4.0), QUIET, bits(-6.0), bits(1.0)]);
        let reduce =
            |kind| float_lanes.reduce(kind, float(32), None, false, NEAREST, FloatRules::default());
        assert_eq!(
            reduce(VectorReduceKind::MinimumNumber),
            Some(u128::from(bits(-6.0)))
        );
        assert_eq!(
            reduce(VectorReduceKind::MinimumMagnitudeNumber),
            Some(u128::from(bits(1.0)))
        );
        assert_eq!(reduce(VectorReduceKind::MaximumMagnitude), None);
    }

    /// **Float lanes compute as IEEE does**, a NaN folding nothing; a scalar
    /// element computes lane 0 and keeps the rest (x86 `addss`).
    /// **A NaN operand folds to the NaN the target's rule picks.** SSE's
    /// rule gives the first NaN quieted (`addps`): a quiet NaN of payload 1
    /// beside a number, then a number beside a signalling NaN, which comes out
    /// quiet. x87's (`fadd`) prefers a quiet NaN to a signalling one, then the
    /// larger significand, and leaves two that differ only in sign undecided.
    /// RISC-V's gives its canonical NaN whatever the operands. A target with
    /// no rule (Arm's, which `FPCR.DN` decides) folds nothing. A fused
    /// multiply-add and its signed forms pick among the multiplicand, the
    /// multiplier and the addend in that order, each as read.
    #[test]
    fn a_nan_operand_folds_to_the_nan_the_targets_rule_picks() {
        let rules = |rule: NanOperands| FloatRules {
            default_nan: Some(DefaultNan::Positive),
            nan_operands: Some(rule),
            extended_nan_operands: Some(rule),
            ..FloatRules::default()
        };
        let (quiet_one, signalling_two) = (0x7fc0_0001_u128, 0x7f80_0002_u128);
        let one = u128::from(1.0f32.to_bits());
        let add = |a: &[u128], b: &[u128], rules: FloatRules| {
            lanes(32, a)
                .binary(
                    &lanes(32, b),
                    VectorBinaryKind::Add,
                    float(32),
                    NEAREST,
                    rules,
                )
                .map(|vector| vector.lane_values().collect::<Vec<_>>())
        };
        assert_eq!(
            add(
                &[quiet_one, one],
                &[one, signalling_two],
                rules(NanOperands::First)
            ),
            Some(vec![quiet_one, 0x7fc0_0002])
        );
        assert_eq!(
            add(&[signalling_two], &[quiet_one], rules(NanOperands::First)),
            Some(vec![0x7fc0_0002]),
            "the first NaN, a signalling one, quieted"
        );
        assert_eq!(
            add(&[quiet_one], &[signalling_two], rules(NanOperands::Default)),
            Some(vec![0x7fc0_0000]),
            "RISC-V's canonical NaN"
        );
        assert_eq!(
            add(&[quiet_one], &[one], FloatRules::default()),
            None,
            "control: no rule"
        );
        // x87 extended NaNs: the exponent all ones, the integer bit set, the
        // quiet bit 62.
        let extended = |quiet: bool, payload: u128, negative: bool| {
            let sign = if negative { 0x8000 } else { 0 };
            (sign | 0x7fff) << 64 | 1 << 63 | u128::from(quiet) << 62 | payload
        };
        let x87 = |a: u128, b: u128| {
            lanes(80, &[a])
                .binary(
                    &lanes(80, &[b]),
                    VectorBinaryKind::Add,
                    VectorElement {
                        scalar: true,
                        ..float(80)
                    },
                    NEAREST,
                    rules(NanOperands::LargerSignificand),
                )
                .map(|vector| vector.lane(0))
        };
        assert_eq!(
            x87(extended(true, 1, false), extended(false, 5, false)),
            Some(extended(true, 1, false)),
            "a quiet NaN over a signalling one"
        );
        assert_eq!(
            x87(extended(false, 1, false), extended(false, 5, false)),
            Some(extended(true, 5, false)),
            "of two signalling NaNs the larger significand, quieted"
        );
        assert_eq!(
            x87(extended(true, 3, false), extended(true, 3, true)),
            None,
            "two that differ only in sign"
        );
        // A fused multiply-add of `a * b + c` gives the first NaN of the
        // multiplicand, the multiplier and the addend, as read: a signed
        // form's NaN is not negated (x86's, Intel SDM vol. 1, table 14-17).
        let fused = |kind: VectorTernaryKind, operands: [u128; 3], rules: FloatRules| {
            let [a, b, c] = operands.map(|operand| lanes(32, &[operand]));
            a.ternary(&b, &c, kind, float(32), NEAREST, rules)
                .map(|vector| vector.lane(0))
        };
        let signalling_three = 0x7f80_0003_u128;
        for kind in [
            VectorTernaryKind::Fma,
            VectorTernaryKind::Fms,
            VectorTernaryKind::Fnma,
            VectorTernaryKind::Fnms,
        ] {
            assert_eq!(
                fused(
                    kind,
                    [one, quiet_one, signalling_three],
                    rules(NanOperands::First)
                ),
                Some(quiet_one),
                "{kind:?}: the multiplier's NaN before the addend's"
            );
            assert_eq!(
                fused(
                    kind,
                    [one, one, signalling_three],
                    rules(NanOperands::First)
                ),
                Some(0x7fc0_0003),
                "{kind:?}: the addend's NaN, quieted, its sign as read"
            );
            assert_eq!(
                fused(
                    kind,
                    [signalling_two, quiet_one, one],
                    rules(NanOperands::Default)
                ),
                Some(0x7fc0_0000),
                "{kind:?}: RISC-V's canonical NaN"
            );
            assert_eq!(
                fused(kind, [quiet_one, one, one], FloatRules::default()),
                None,
                "{kind:?}: control, no rule"
            );
        }
        // The signs of numbers: 1.5 * 1.5 and 0.25 make 2.5, 2, -2 and -2.5.
        let single = |value: f32| u128::from(value.to_bits());
        for (kind, sum) in [
            (VectorTernaryKind::Fma, 2.5f32),
            (VectorTernaryKind::Fms, 2.0),
            (VectorTernaryKind::Fnma, -2.0),
            (VectorTernaryKind::Fnms, -2.5),
        ] {
            assert_eq!(
                fused(
                    kind,
                    [single(1.5), single(1.5), single(0.25)],
                    FloatRules::default()
                ),
                Some(single(sum)),
                "{kind:?}"
            );
        }
    }

    /// **A complex product or rotation folds where every way of computing
    /// it agrees.** Expected halves from IEEE binary16 packing (Python's
    /// `struct` `e` format), the oracle: `(1 + 2i)(3 + 4i)` is `-5 + 10i`,
    /// `(2 + 0i)(0.5 + 0.25i)` is `1 + 0.5i`, conjugated `(1 + 2i)(3 - 4i)` is
    /// `11 + 2i`, and accumulated onto `0.5 + 0.25i` it is `-4.5 + 10.25i`.
    /// `(1 + 2^-10)^2` is no binary16, a zero summed of opposite signs takes
    /// its sign from the rounding, and a NaN is no number: each folds
    /// nothing. A scalar form keeps the first source's upper element.
    /// Arm's rotations add the second source turned: `(1 + 2i) + i(3 + 4i)`
    /// is `-3 + 5i` and `(1 + 2i) - i(3 + 4i)` is `5 - 1i`; on bytes,
    /// `(-100 + 100i) + i(100 + 100i)` wraps to `56 - 56i` and saturates to
    /// `-128 + 127i`.
    #[test]
    fn a_complex_product_or_rotation_folds_where_every_computation_agrees() {
        let half = VectorElement::float(16);
        let product = |conjugate: bool, accumulates: bool, element: VectorElement| ComplexShape {
            product: ComplexProduct::Multiply {
                conjugate,
                accumulates,
            },
            element,
            saturates: false,
        };
        let folded = |a: &[u128], b: &[u128], c: Option<&[u128]>, shape: ComplexShape| {
            let bits = if shape.element.bits == 0 {
                16
            } else {
                shape.element.bits
            };
            let accumulator = c.map(|c| lanes(bits, c));
            lanes(bits, a)
                .complex(&lanes(bits, b), accumulator.as_ref(), shape)
                .map(|vector| {
                    (0..vector.lanes())
                        .map(|lane| vector.lane(lane))
                        .collect::<Vec<_>>()
                })
        };
        let a = [0x3c00, 0x4000, 0x4000, 0x0000];
        let b = [0x4200, 0x4400, 0x3800, 0x3400];
        assert_eq!(
            folded(&a, &b, None, product(false, false, half)),
            Some(vec![0xc500, 0x4900, 0x3c00, 0x3800])
        );
        assert_eq!(
            folded(&a[..2], &b[..2], None, product(true, false, half)),
            Some(vec![0x4980, 0x4000])
        );
        assert_eq!(
            folded(
                &a[..2],
                &b[..2],
                Some(&[0x3800, 0x3400]),
                product(false, true, half)
            ),
            Some(vec![0xc480, 0x4920])
        );
        assert_eq!(
            folded(
                &[0x3c01, 0x0000],
                &[0x3c01, 0x0000],
                None,
                product(false, false, half)
            ),
            None,
            "(1 + 2^-10)^2 is no binary16"
        );
        assert_eq!(
            folded(
                &[0x0000, 0x0000],
                &[0x0000, 0x0000],
                None,
                product(false, false, half)
            ),
            None,
            "0 - 0 takes its sign from the rounding"
        );
        assert_eq!(
            folded(
                &[0x7e00, 0x0000],
                &b[..2],
                None,
                product(false, false, half)
            ),
            None,
            "a NaN"
        );
        let scalar = VectorElement {
            scalar: true,
            ..half
        };
        assert_eq!(
            folded(&a, &b, None, product(false, false, scalar)),
            Some(vec![0xc500, 0x4900, 0x4000, 0x0000]),
            "the upper element is the first source's"
        );
        let rotate =
            |rotation: ComplexRotation, element: VectorElement, saturates: bool| ComplexShape {
                product: ComplexProduct::Rotate(rotation),
                element,
                saturates,
            };
        assert_eq!(
            folded(
                &a[..2],
                &b[..2],
                None,
                rotate(ComplexRotation::Quarter, half, false)
            ),
            Some(vec![0xc200, 0x4500])
        );
        assert_eq!(
            folded(
                &a[..2],
                &b[..2],
                None,
                rotate(ComplexRotation::ThreeQuarters, half, false)
            ),
            Some(vec![0x4500, 0xbc00])
        );
        let byte = VectorElement::integer(8, Some(true));
        let (minus_hundred, hundred) = (0x9c, 0x64);
        assert_eq!(
            folded(
                &[minus_hundred, hundred],
                &[hundred, hundred],
                None,
                rotate(ComplexRotation::Quarter, byte, false)
            ),
            Some(vec![0x38, 0xc8])
        );
        assert_eq!(
            folded(
                &[minus_hundred, hundred],
                &[hundred, hundred],
                None,
                rotate(ComplexRotation::Quarter, byte, true)
            ),
            Some(vec![0x80, 0x7f])
        );
    }

    /// **A bfloat16 dot product folds as its instruction adds.** x86's
    /// `vdpbf16ps` adds the odd pair's product, then the even pair's, each
    /// sum rounded to nearest even, which Rust's binary32 addition, the
    /// oracle, computes too: `1 + 1.5 * 2 + 2 * 0.5` is 5, and `1 + 2^-24 +
    /// 2^-24` stays 1, each tie rounding to even. A subnormal accumulator
    /// reads as zero. Arm's `bfdot` sums the products first: it folds the
    /// exact 5 and `1 + 2^-23`, and declines `1 + 3 * 2^-25`, whose rounding,
    /// to odd or by FPCR, the program decides, and the subnormal. A NaN folds
    /// neither.
    #[test]
    fn a_bfloat16_dot_product_folds_as_its_instruction_adds() {
        let bf16 = |value: f32| u128::from(value.to_bits() >> 16);
        let bfloat = FloatElements::new(
            FloatSource::Format(FloatFormat::BFloat16),
            FloatFormat::Single,
        );
        let dot = |accumulator: u32, a: [f32; 2], b: [f32; 2], kind: FloatDotRule| {
            lanes(32, &[u128::from(accumulator)])
                .dot_float(
                    &lanes(16, &a.map(bf16)),
                    &lanes(16, &b.map(bf16)),
                    bfloat,
                    kind,
                )
                .map(|vector| vector.lane(0))
        };
        let single = |value: f32| u128::from(value.to_bits());
        let tiny = 2f32.powi(-12);
        let oracle = |accumulator: f32, a: [f32; 2], b: [f32; 2]| {
            single(accumulator + a[1] * b[1] + a[0] * b[0])
        };
        for kind in [FloatDotRule::Sequential, FloatDotRule::Fpcr] {
            assert_eq!(
                dot(1.0f32.to_bits(), [2.0, 1.5], [0.5, 2.0], kind),
                Some(oracle(1.0, [2.0, 1.5], [0.5, 2.0])),
                "{kind:?}: exact"
            );
            assert_eq!(
                dot(1.0f32.to_bits(), [f32::NAN, 1.0], [1.0, 1.0], kind),
                None,
                "{kind:?}: a NaN"
            );
        }
        assert_eq!(
            dot(
                1.0f32.to_bits(),
                [tiny, tiny],
                [tiny, tiny],
                FloatDotRule::Sequential
            ),
            Some(oracle(1.0, [tiny, tiny], [tiny, tiny])),
            "x86: each sum rounded"
        );
        assert_eq!(oracle(1.0, [tiny, tiny], [tiny, tiny]), single(1.0));
        // Arm sums the products first, `2^-23`, which `1` holds exactly.
        assert_eq!(
            dot(
                1.0f32.to_bits(),
                [tiny, tiny],
                [tiny, tiny],
                FloatDotRule::Fpcr
            ),
            Some(single(1.0 + 2f32.powi(-23))),
            "Arm: the products summed first"
        );
        assert_eq!(
            dot(
                1.0f32.to_bits(),
                [tiny, tiny],
                [tiny, tiny / 2.0],
                FloatDotRule::Fpcr
            ),
            None,
            "Arm: a sum its rounding decides"
        );
        // A subnormal accumulator and zero products.
        assert_eq!(
            dot(1, [0.0, 0.0], [0.0, 0.0], FloatDotRule::Sequential),
            Some(0),
            "x86: the subnormal read as zero"
        );
        assert_eq!(
            dot(1, [0.0, 0.0], [0.0, 0.0], FloatDotRule::Fpcr),
            None,
            "Arm: a subnormal its flush decides"
        );
    }

    /// **A half-precision dot product folds where every rounding agrees.**
    /// Arm's `fdot` (SVE2.1) sums the two products with one rounding and
    /// adds that sum to the accumulator with another
    /// ([`FloatDotRule::DotAdd`]), so it folds where each of those is exact
    /// and none subnormal, which every rounding gives alike: `1 + 1.5 * 2 +
    /// 2 * 0.5` is 5, and `1 + 2^-24 + 2^-24` is `1 + 2^-23`, which binary32
    /// holds. `1 + 2^-24 + 2^-25` does not fold: binary32 holds the products'
    /// sum but not the accumulator's, so the rounding mode decides it. A NaN
    /// folds nothing, nor an 8-bit source whose format a register names at
    /// run time, nor a group of more than two, whose sum the order of the
    /// additions would decide.
    #[test]
    fn a_half_precision_dot_product_folds_its_exact_products() {
        let half = |value: f32| {
            FloatFormat::Single
                .convert(u128::from(value.to_bits()), FloatFormat::Half)
                .expect("a number half precision holds")
        };
        let single = |value: f32| u128::from(value.to_bits());
        let halves =
            FloatElements::new(FloatSource::Format(FloatFormat::Half), FloatFormat::Single);
        let dot = |accumulator: f32, a: [f32; 2], b: [f32; 2]| {
            lanes(32, &[single(accumulator)])
                .dot_float(
                    &lanes(16, &a.map(half)),
                    &lanes(16, &b.map(half)),
                    halves,
                    FloatDotRule::DotAdd,
                )
                .map(|vector| vector.lane(0))
        };
        let tiny = 2f32.powi(-12);
        assert_eq!(
            dot(1.0, [2.0, 1.5], [0.5, 2.0]),
            Some(single(5.0)),
            "the products exact in binary32"
        );
        assert_eq!(
            dot(1.0, [tiny, tiny], [tiny, tiny]),
            Some(single(1.0 + 2f32.powi(-23))),
            "the products summed first"
        );
        assert_eq!(
            dot(1.0, [tiny, tiny], [tiny, tiny / 2.0]),
            None,
            "a sum its rounding decides"
        );
        assert_eq!(
            lanes(32, &[single(1.0)]).dot_float(
                &lanes(16, &[FloatFormat::Half.canonical_nan_bits(), half(1.0)]),
                &lanes(16, &[half(1.0), half(1.0)]),
                halves,
                FloatDotRule::DotAdd,
            ),
            None,
            "a NaN"
        );
        assert_eq!(
            lanes(32, &[single(1.0)]).dot_float(
                &lanes(8, &[0, 0, 0, 0]),
                &lanes(8, &[0, 0, 0, 0]),
                FloatElements::new(FloatSource::Fp8, FloatFormat::Single),
                FloatDotRule::Fp8,
            ),
            None,
            "a format a register names at run time"
        );
        assert_eq!(
            lanes(64, &[u128::from(1.0f64.to_bits())]).dot_float(
                &lanes(16, &[half(1.0), half(1.0), half(1.0), half(1.0)]),
                &lanes(16, &[half(1.0), half(1.0), half(1.0), half(1.0)]),
                FloatElements::new(FloatSource::Format(FloatFormat::Half), FloatFormat::Double),
                FloatDotRule::DotAdd,
            ),
            None,
            "a group of four, whose order would decide the sum"
        );
    }

    /// **Lanes whose rounding and specials the implementation decides fold
    /// only exact results of numbers.** 3DNow!'s `pfadd` of `1.5 + 0.25`
    /// folds whatever the environment's mode; `1 + 2^-30`, which each mode
    /// rounds its own way, does not, nor a NaN operand, nor a sum that
    /// overflows, nor a compare of a NaN. The same lanes read as IEEE floats
    /// fold the inexact sum by the environment, the control.
    #[test]
    fn open_float_lanes_fold_only_exact_results_of_numbers() {
        let single = |values: &[f32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|v| u128::from(v.to_bits()))
                    .collect::<Vec<_>>(),
            )
        };
        let open = VectorElement {
            kind: VectorElementKind::FloatOpen,
            ..float(32)
        };
        let add = |a: &[f32], b: &[f32], element| {
            single(a).binary(
                &single(b),
                VectorBinaryKind::Add,
                element,
                NEAREST,
                FloatRules::default(),
            )
        };
        assert_eq!(add(&[1.5], &[0.25], open), Some(single(&[1.75])));
        let tiny = f32::powi(2.0, -30);
        assert_eq!(add(&[1.0], &[tiny], open), None, "inexact");
        assert_eq!(
            add(&[1.0], &[tiny], float(32)),
            Some(single(&[1.0])),
            "control: IEEE lanes round by the environment"
        );
        assert_eq!(add(&[f32::NAN], &[1.0], open), None, "a NaN operand");
        assert_eq!(add(&[f32::MAX], &[f32::MAX], open), None, "overflows");
        assert_eq!(
            single(&[f32::NAN]).compare(&single(&[1.0]), VectorCompareKind::Eq, open),
            None,
            "a compare of a NaN"
        );
    }

    /// **A scale by an integer exponent is exact or folds nothing.** `1.5 *
    /// 2^2`, `-3.0 * 2^-1` and `1.0 * 2^127` fold; a power that overflows,
    /// one that makes a subnormal, and a subnormal operand decline, and the
    /// exponent lane is read signed (`-1` is `0xffff_ffff`).
    #[test]
    fn a_scale_by_an_integer_exponent_is_exact_or_folds_nothing() {
        let single = |values: &[f32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|v| u128::from(v.to_bits()))
                    .collect::<Vec<_>>(),
            )
        };
        let scale = |values: &[f32], powers: &[i32]| {
            let powers: Vec<u128> = powers
                .iter()
                .map(|power| u128::from(power.cast_unsigned()))
                .collect();
            single(values).binary(
                &lanes(32, &powers),
                VectorBinaryKind::ScaleInteger,
                float(32),
                NEAREST,
                FloatRules::default(),
            )
        };
        assert_eq!(
            scale(&[1.5, -3.0, 1.0], &[2, -1, 127]),
            Some(single(&[6.0, -1.5, f32::powi(2.0, 127)]))
        );
        assert_eq!(scale(&[1.0], &[128]), None, "overflows");
        assert_eq!(scale(&[1.0], &[-127]), None, "a subnormal result");
        assert_eq!(
            scale(&[f32::from_bits(1)], &[149]),
            None,
            "a subnormal operand"
        );
    }

    #[test]
    fn float_lanes_compute_as_ieee_does() {
        let single = |values: &[f32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|v| u128::from(v.to_bits()))
                    .collect::<Vec<_>>(),
            )
        };
        let sum = single(&[1.5, -2.0]).binary(
            &single(&[0.25, 0.5]),
            VectorBinaryKind::Add,
            float(32),
            NEAREST,
            FloatRules::default(),
        );
        assert_eq!(sum, Some(single(&[1.75, -1.5])));
        let min = single(&[1.0, -0.0]).binary(
            &single(&[2.0, 0.0]),
            VectorBinaryKind::Min,
            float(32),
            NEAREST,
            FloatRules::default(),
        );
        assert_eq!(
            min, None,
            "the sign of a zero minimum is the architecture's"
        );
        let nan = single(&[f32::NAN]).binary(
            &single(&[1.0]),
            VectorBinaryKind::Add,
            float(32),
            NEAREST,
            FloatRules::default(),
        );
        assert_eq!(nan, None);
        let scalar = VectorElement {
            scalar: true,
            ..float(32)
        };
        let addss = single(&[1.0, 7.0]).binary(
            &single(&[2.0, 9.0]),
            VectorBinaryKind::Mul,
            scalar,
            NEAREST,
            FloatRules::default(),
        );
        assert_eq!(addss, Some(single(&[2.0, 7.0])));
        let negated = single(&[1.0, -0.0]).unary(
            VectorUnaryKind::Neg,
            float(32),
            NEAREST,
            FloatRules::default(),
        );
        assert_eq!(negated, Some(single(&[-1.0, 0.0])));
        let floor = single(&[-1.5, 2.5]).unary(
            VectorUnaryKind::Floor,
            float(32),
            NEAREST,
            FloatRules::default(),
        );
        assert_eq!(floor, Some(single(&[-2.0, 2.0])));
    }

    /// **An arithmetic negation or absolute value of numbers folds as the sign
    /// operation does**, and one of a NaN folds nothing: arithmetic
    /// propagates it as the hardware's NaN rules say, a signalling one quieted
    /// into the default NaN (legacy MIPS `neg.d`, `abs.d`).
    #[test]
    fn an_arithmetic_sign_operation_folds_numbers_alone() {
        let single = |values: &[f32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|v| u128::from(v.to_bits()))
                    .collect::<Vec<_>>(),
            )
        };
        let unary = |values: &[f32], kind: VectorUnaryKind| {
            single(values).unary(kind, float(32), NEAREST, FloatRules::default())
        };
        assert_eq!(
            unary(&[1.0, -0.0, f32::INFINITY], VectorUnaryKind::NegArithmetic),
            Some(single(&[-1.0, 0.0, f32::NEG_INFINITY]))
        );
        assert_eq!(
            unary(
                &[-1.0, -0.0, f32::NEG_INFINITY],
                VectorUnaryKind::AbsArithmetic
            ),
            Some(single(&[1.0, 0.0, f32::INFINITY]))
        );
        let signalling = f32::from_bits(0x7fa0_0000);
        for kind in [
            VectorUnaryKind::NegArithmetic,
            VectorUnaryKind::AbsArithmetic,
        ] {
            assert_eq!(
                unary(&[1.0, f32::NAN], kind),
                None,
                "{kind:?} of a quiet NaN"
            );
            assert_eq!(
                unary(&[signalling], kind),
                None,
                "{kind:?} of a signalling NaN"
            );
        }
        assert_eq!(
            unary(&[-f32::NAN], VectorUnaryKind::Abs),
            Some(single(&[f32::NAN])),
            "the sign operation clears a NaN's sign"
        );
    }

    /// **A bitwise operation on float lanes reads their bits**, a NaN's too:
    /// x86 `andps` masks the sign off, `and.not` clears the bits of its second
    /// source, `orps` sets bits and `not` complements them whatever the float
    /// they spell, where arithmetic on the same lanes reads them as floats.
    #[test]
    fn a_bitwise_operation_on_float_lanes_reads_their_bits() {
        let raw = |values: &[u128]| lanes(32, values);
        let (a, b) = (
            raw(&[
                u128::from(f32::NAN.to_bits()),
                u128::from((-2.0f32).to_bits()),
            ]),
            raw(&[0x7fff_ffff, 0x7fff_ffff]),
        );
        let ieee = float(32);
        assert_eq!(
            a.binary(
                &b,
                VectorBinaryKind::And,
                ieee,
                NEAREST,
                FloatRules::default()
            ),
            Some(raw(&[0x7fc0_0000, u128::from(2.0f32.to_bits())]))
        );
        assert_eq!(
            b.binary(
                &a,
                VectorBinaryKind::AndNot,
                ieee,
                NEAREST,
                FloatRules::default()
            ),
            Some(raw(&[0x003f_ffff, 0x3fff_ffff]))
        );
        assert_eq!(
            a.binary(
                &b,
                VectorBinaryKind::Or,
                ieee,
                NEAREST,
                FloatRules::default()
            ),
            Some(raw(&[0x7fff_ffff, 0xffff_ffff]))
        );
        assert_eq!(
            a.binary(
                &b,
                VectorBinaryKind::Add,
                ieee,
                NEAREST,
                FloatRules::default()
            ),
            None,
            "a NaN sum"
        );
        // The not of float lanes is their bits complemented, a NaN's too,
        // and the identity their bits.
        assert_eq!(
            a.unary(VectorUnaryKind::Not, ieee, NEAREST, FloatRules::default()),
            Some(raw(&[0x803f_ffff, 0x3fff_ffff]))
        );
        assert_eq!(
            a.unary(
                VectorUnaryKind::Identity,
                ieee,
                NEAREST,
                FloatRules::default()
            ),
            Some(a)
        );
    }

    /// **The integer unaries count and reverse at the lane's width.**
    #[test]
    fn integer_unaries_count_at_the_lane_width() {
        let unary = |values: &[u128], kind| {
            lanes(8, values)
                .unary(kind, integer(None), None, FloatRules::default())
                .map(|vector| vector.lane_values().collect::<Vec<_>>())
        };
        assert_eq!(
            unary(&[0x80, 0x01], VectorUnaryKind::Abs),
            Some(vec![0x80, 0x01])
        );
        assert_eq!(
            unary(&[0xff, 0x7f], VectorUnaryKind::Abs),
            Some(vec![0x01, 0x7f])
        );
        assert_eq!(
            unary(&[0x0f, 0], VectorUnaryKind::LeadingZeros),
            Some(vec![4, 8])
        );
        assert_eq!(
            unary(&[0xf0, 0xff], VectorUnaryKind::LeadingOnes),
            Some(vec![4, 8])
        );
        assert_eq!(
            unary(&[0xf0, 0x01], VectorUnaryKind::LeadingSignBits),
            Some(vec![3, 6])
        );
        assert_eq!(unary(&[0x0b], VectorUnaryKind::Popcount), Some(vec![3]));
        assert_eq!(
            unary(&[0x01, 0x0b], VectorUnaryKind::BitReverse),
            Some(vec![0x80, 0xd0])
        );
        assert_eq!(unary(&[1], VectorUnaryKind::Sqrt), None);
    }

    /// **A compare is all ones where it holds**, integers ordered by their
    /// stated signedness and floats as IEEE orders them: a NaN is unordered,
    /// and `-0.0` equals `0.0`.
    #[test]
    fn a_compare_sets_the_lanes_where_it_holds() {
        let compare = |a: &[u128], b: &[u128], kind, element| {
            lanes(8, a)
                .compare(&lanes(8, b), kind, element)
                .map(|vector| vector.lane_values().collect::<Vec<_>>())
        };
        assert_eq!(
            compare(
                &[0x80, 1],
                &[0x7f, 1],
                VectorCompareKind::Gt,
                integer(Some(true))
            ),
            Some(vec![0, 0])
        );
        assert_eq!(
            compare(
                &[0x80, 1],
                &[0x7f, 1],
                VectorCompareKind::Gt,
                integer(Some(false))
            ),
            Some(vec![0xff, 0])
        );
        assert_eq!(
            compare(&[1], &[2], VectorCompareKind::Lt, integer(None)),
            None,
            "no signedness"
        );
        assert_eq!(
            compare(&[1, 3], &[1, 2], VectorCompareKind::Eq, integer(None)),
            Some(vec![0xff, 0])
        );
        let single = |values: &[f32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|v| u128::from(v.to_bits()))
                    .collect::<Vec<_>>(),
            )
        };
        let mask = |a: &[f32], b: &[f32], kind| {
            single(a)
                .compare(&single(b), kind, float(32))
                .map(|vector| vector.lane_values().collect::<Vec<_>>())
        };
        assert_eq!(
            mask(&[-0.0, f32::NAN], &[0.0, f32::NAN], VectorCompareKind::Eq),
            Some(vec![0xffff_ffff, 0])
        );
        assert_eq!(
            mask(&[1.0, f32::NAN], &[2.0, 1.0], VectorCompareKind::NotGe),
            Some(vec![0xffff_ffff, 0xffff_ffff])
        );
        assert_eq!(
            mask(&[1.0, f32::NAN], &[2.0, 1.0], VectorCompareKind::Unordered),
            Some(vec![0, 0xffff_ffff])
        );
    }

    /// **Lanes move as the selectors say**: a shuffle picks from either
    /// operand or zeroes, a permute indexes its tables end to end, an insert
    /// replaces one lane.
    #[test]
    fn lanes_move_as_the_selectors_say() {
        let left = lanes(32, &[10, 11, 12, 13]);
        let right = lanes(32, &[20, 21, 22, 23]);
        let shuffled = left.shuffle(Some(&right), &[Left(3), Right(0), Zero, Left(0)]);
        assert_eq!(shuffled, Some(lanes(32, &[13, 20, 0, 10])));
        assert_eq!(left.shuffle(None, &[Right(0)]), None);
        assert_eq!(left.shuffle(None, &[Undef]), None);
        assert_eq!(left.shuffle(None, &[Left(4)]), None);
        let indices = lanes(8, &[5, 0, 7]);
        let tables = [lanes(8, &[1, 2, 3, 4]), lanes(8, &[5, 6, 7, 8])];
        assert_eq!(indices.permute(&tables), Some(lanes(8, &[6, 1, 8])));
        assert_eq!(lanes(8, &[8]).permute(&tables), None, "past both tables");
        assert_eq!(
            left.with_lane(2, 0x1_0000_0001),
            Some(lanes(32, &[10, 11, 1, 13]))
        );
        assert_eq!(left.with_lane(4, 1), None);
    }

    /// **A cast converts the low lanes it names** (x86 `cvtdq2pd` two of
    /// four, `pmovsxbw` sign-extending, `cvttps2dq` toward zero), and a
    /// float whose integer the lane does not hold folds nothing.
    #[test]
    fn a_cast_converts_the_low_lanes() {
        let doubles = lanes(32, &[0xffff_fffe, 3, 9, 9])
            .cast(
                VectorCastKind::Signed,
                VectorCastKind::Float,
                None,
                64,
                2,
                RULES,
            )
            .expect("two doubles");
        assert_eq!(
            doubles.lane_values().collect::<Vec<_>>(),
            [(-2.0f64).to_bits(), 3.0f64.to_bits()].map(u128::from)
        );
        let words = lanes(8, &[0x80, 0x7f])
            .cast(
                VectorCastKind::Signed,
                VectorCastKind::Signed,
                None,
                16,
                2,
                RULES,
            )
            .expect("two words");
        assert_eq!(words.lane_values().collect::<Vec<_>>(), [0xff80, 0x7f]);
        let single = |values: &[f32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|v| u128::from(v.to_bits()))
                    .collect::<Vec<_>>(),
            )
        };
        let truncated = single(&[-1.75, 2.5])
            .cast(
                VectorCastKind::Float,
                VectorCastKind::Signed,
                None,
                32,
                2,
                RULES,
            )
            .expect("two integers");
        assert_eq!(
            truncated.lane_values().collect::<Vec<_>>(),
            [0xffff_ffff, 2]
        );
        assert_eq!(
            single(&[3e9]).cast(
                VectorCastKind::Float,
                VectorCastKind::Signed,
                None,
                32,
                1,
                RULES
            ),
            None,
            "past the lane"
        );
        assert_eq!(
            single(&[1.0]).cast(
                VectorCastKind::Float,
                VectorCastKind::Signed,
                None,
                32,
                2,
                RULES
            ),
            None
        );
    }

    /// **A NaN cast between formats is what the target's rule makes it.** x86
    /// (`cvtpd2ps`) keeps the NaN, quiet, its sign and the top of its
    /// payload: the double `0x7ff0000020000000` (signaling) is the single
    /// `0x7fc00001`, `0xfff8000000000000` the single `0xffc00000`. RISC-V
    /// (`vfncvt.f.f.w`) gives its canonical NaN `0x7fc00000` for both, and a
    /// target that states no rule (Arm, whose `FPCR.DN` decides) folds
    /// nothing.
    #[test]
    fn a_nan_cast_between_formats_is_the_targets_rule() {
        let nans = lanes(64, &[0x7ff0_0000_2000_0000, 0xfff8_0000_0000_0000]);
        let narrowed = |nan_conversion| {
            nans.cast(
                VectorCastKind::Float,
                VectorCastKind::Float,
                NEAREST,
                32,
                2,
                FloatRules {
                    default_nan: Some(DefaultNan::Positive),
                    nan_conversion,
                    ..RULES
                },
            )
            .map(|vector| vector.lane_values().collect::<Vec<_>>())
        };
        assert_eq!(
            narrowed(Some(NanPropagation::Operand)),
            Some(vec![0x7fc0_0001, 0xffc0_0000])
        );
        assert_eq!(
            narrowed(Some(NanPropagation::Default)),
            Some(vec![0x7fc0_0000, 0x7fc0_0000])
        );
        assert_eq!(narrowed(None), None);
    }

    /// **A float cast honours the flush mode the program may set.** The least
    /// single widens to a double exactly, and the double `2^-130` narrows to a
    /// single's subnormal; where the program may flush subnormals the first
    /// reads as zero and the second writes as zero, so neither folds. A
    /// normal lane converts either way.
    #[test]
    fn a_float_cast_honours_the_flush_mode() {
        let cast = |vector: &VectorConst, bits: u32, flushes: bool| {
            vector
                .cast(
                    VectorCastKind::Float,
                    VectorCastKind::Float,
                    NEAREST,
                    bits,
                    1,
                    FloatRules {
                        program_flushes_subnormals: flushes,
                        ..RULES
                    },
                )
                .map(|vector| vector.lane(0))
        };
        let least = lanes(32, &[1]);
        assert_eq!(
            cast(&least, 64, false),
            Some(u128::from(f64::from(f32::from_bits(1)).to_bits()))
        );
        assert_eq!(cast(&least, 64, true), None);
        let tiny = lanes(64, &[u128::from(2.0f64.powi(-130).to_bits())]);
        // 2^-130 is 2^19 of the single's least subnormal, 2^-149.
        assert_eq!(cast(&tiny, 32, false), Some(1 << 19));
        assert_eq!(cast(&tiny, 32, true), None);
        let two = lanes(64, &[u128::from(2.0f64.to_bits())]);
        assert_eq!(cast(&two, 32, true), Some(u128::from(2.0f32.to_bits())));
    }

    /// **A cast that states its rounding converts by it.** The double `1/3`
    /// narrows to the single `0x3eaaaaaa` down and `0x3eaaaaab` up and to odd;
    /// the integer `2^24 + 1` is the single `2^24` down and `2^24 + 2` up, and
    /// converts no way to odd.
    #[test]
    fn a_cast_stating_its_rounding_converts_by_it() {
        let third = lanes(64, &[u128::from((1.0f64 / 3.0).to_bits())]);
        let narrowed = |rounding| {
            third
                .cast(
                    VectorCastKind::Float,
                    VectorCastKind::Float,
                    Some(rounding),
                    32,
                    1,
                    RULES,
                )
                .map(|lanes| lanes.lane(0))
        };
        assert_eq!(narrowed(FloatRounding::Down), Some(0x3eaa_aaaa));
        assert_eq!(narrowed(FloatRounding::Up), Some(0x3eaa_aaab));
        assert_eq!(narrowed(FloatRounding::ToOdd), Some(0x3eaa_aaab));
        let odd_integer = lanes(32, &[16_777_217]);
        let converted = |rounding| {
            odd_integer
                .cast(
                    VectorCastKind::Signed,
                    VectorCastKind::Float,
                    Some(rounding),
                    32,
                    1,
                    RULES,
                )
                .map(|lanes| lanes.lane(0))
        };
        assert_eq!(converted(FloatRounding::Down), Some(0x4b80_0000));
        assert_eq!(converted(FloatRounding::Up), Some(0x4b80_0001));
        assert_eq!(converted(FloatRounding::ToOdd), None);
    }

    /// **A float reduction folds only to what every order gives.** A
    /// reduction states no order: `[2^24, 1, -2^24]` sums to 0 left to right
    /// and to 1 with the ones first, so it folds nothing, where exact lanes
    /// sum; a product of more than two lanes folds nothing, of two it does;
    /// the least of `0.0` and `-0.0` is the order's choice, the least of
    /// `-1.0` beside them is not.
    #[test]
    fn a_float_reduction_folds_only_what_every_order_gives() {
        let single = |values: &[f32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|value| u128::from(value.to_bits()))
                    .collect::<Vec<_>>(),
            )
        };
        let bits = |value: f32| Some(u128::from(value.to_bits()));
        let element = float(32);
        assert_eq!(
            single(&[16_777_216.0, 1.0, -16_777_216.0, 0.0]).reduce(
                VectorReduceKind::Add,
                element,
                None,
                false,
                NEAREST,
                FloatRules::default()
            ),
            None
        );
        assert_eq!(
            single(&[1.5, -0.25, 4.0, 2.0]).reduce(
                VectorReduceKind::Add,
                element,
                None,
                false,
                NEAREST,
                FloatRules::default()
            ),
            bits(7.25)
        );
        assert_eq!(
            single(&[2.0, 3.0, 0.5, 1.0]).reduce(
                VectorReduceKind::Mul,
                element,
                None,
                false,
                NEAREST,
                FloatRules::default()
            ),
            None
        );
        assert_eq!(
            single(&[2.0, 3.0]).reduce(
                VectorReduceKind::Mul,
                element,
                None,
                false,
                NEAREST,
                FloatRules::default()
            ),
            bits(6.0)
        );
        assert_eq!(
            single(&[0.0, -0.0, 1.0, 2.0]).reduce(
                VectorReduceKind::Min,
                element,
                None,
                false,
                NEAREST,
                FloatRules::default()
            ),
            None
        );
        assert_eq!(
            single(&[0.0, -0.0, -1.0, 2.0]).reduce(
                VectorReduceKind::Min,
                element,
                None,
                false,
                NEAREST,
                FloatRules::default()
            ),
            bits(-1.0)
        );
        assert_eq!(
            single(&[0.0, -0.0, -1.0, 2.0]).reduce(
                VectorReduceKind::MaxNum,
                element,
                None,
                false,
                NEAREST,
                FloatRules::default()
            ),
            bits(2.0)
        );
    }

    /// **A reduction starts from its initial value.** Ordered, the lanes add
    /// one at a time after it: `1 + 2^24 + 1 - 2^24 + 0.5` of singles is
    /// `0.5`, each `1` rounded away, and `-0.0` plus two `-0.0` lanes stays
    /// `-0.0`; unordered, the same lanes may meet in another order, so they
    /// fold nothing, and an ordered product of three values folds. Integers
    /// count the initial value as one more lane, signed or unsigned as the
    /// element says, and a test of any or every lane takes none.
    #[test]
    fn a_reduction_starts_from_its_initial_value() {
        let single = |value: f32| u128::from(value.to_bits());
        let singles = |values: &[f32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|value| single(*value))
                    .collect::<Vec<_>>(),
            )
        };
        let element = float(32);
        let wide = 2.0f32.powi(24);
        let rounding = singles(&[wide, 1.0, -wide, 0.5]);
        assert_eq!(
            rounding.reduce(
                VectorReduceKind::Add,
                element,
                Some(single(1.0)),
                true,
                NEAREST,
                FloatRules::default()
            ),
            Some(single(0.5))
        );
        assert_eq!(
            rounding.reduce(
                VectorReduceKind::Add,
                element,
                Some(single(1.0)),
                false,
                NEAREST,
                FloatRules::default()
            ),
            None
        );
        assert_eq!(
            singles(&[-0.0, -0.0]).reduce(
                VectorReduceKind::Add,
                element,
                Some(single(-0.0)),
                true,
                NEAREST,
                FloatRules::default()
            ),
            Some(single(-0.0))
        );
        assert_eq!(
            singles(&[3.0, 0.5]).reduce(
                VectorReduceKind::Mul,
                element,
                Some(single(2.0)),
                true,
                NEAREST,
                FloatRules::default()
            ),
            Some(single(3.0))
        );
        let bytes = lanes(8, &[1, 2, 0xf0]);
        assert_eq!(
            bytes.reduce(
                VectorReduceKind::Add,
                integer(None),
                Some(5),
                false,
                None,
                FloatRules::default()
            ),
            Some(0xf8)
        );
        assert_eq!(
            bytes.reduce(
                VectorReduceKind::Min,
                integer(Some(true)),
                Some(0xf9),
                false,
                None,
                FloatRules::default()
            ),
            Some(0xf0),
            "-16 is less than the initial -7"
        );
        assert_eq!(
            bytes.reduce(
                VectorReduceKind::Min,
                integer(Some(false)),
                Some(0),
                false,
                None,
                FloatRules::default()
            ),
            Some(0)
        );
        assert_eq!(
            bytes.reduce(
                VectorReduceKind::Any,
                integer(None),
                Some(0),
                false,
                None,
                FloatRules::default()
            ),
            None
        );
    }

    /// **An unsigned lane 128 bits wide is never read as a negative.** A lane
    /// with its top bit set is greater than one, so it is not less than it,
    /// not its minimum and not a negative dividend: where the lane is past
    /// what the signed reading holds, the fold states nothing rather than
    /// the signed answer.
    #[test]
    fn an_unsigned_lane_of_128_bits_is_not_a_negative() {
        let (top, one) = (lanes(128, &[1 << 127]), lanes(128, &[1]));
        let unsigned = integer(Some(false));
        let holds = |result: Option<VectorConst>| result.map(|vector| vector.lane(0) != 0);
        assert_ne!(
            holds(top.compare(&one, VectorCompareKind::Lt, unsigned)),
            Some(true)
        );
        assert_ne!(
            top.binary(
                &one,
                VectorBinaryKind::Min,
                unsigned,
                NEAREST,
                FloatRules::default()
            )
            .map(|vector| vector.lane(0)),
            Some(1 << 127)
        );
        assert_ne!(
            top.binary(
                &lanes(128, &[2]),
                VectorBinaryKind::Div,
                unsigned,
                NEAREST,
                FloatRules::default()
            )
            .map(|vector| vector.lane(0)),
            Some((1 << 127) | (1 << 126))
        );
        assert_ne!(
            lanes(128, &[1 << 127, 1]).reduce(
                VectorReduceKind::Min,
                unsigned,
                None,
                false,
                NEAREST,
                FloatRules::default()
            ),
            Some(1 << 127)
        );
        assert_eq!(
            top.cast(
                VectorCastKind::Unsigned,
                VectorCastKind::Unsigned,
                None,
                64,
                1,
                RULES
            )
            .map(|vector| vector.lane(0)),
            Some(0),
            "a cut keeps the low bits"
        );
        assert_eq!(
            top.compare(&one, VectorCompareKind::Lt, integer(Some(true)))
                .map(|vector| vector.lane(0)),
            Some(u128::MAX),
            "read signed, the top bit is the sign"
        );
    }

    /// **Reductions, gathered bits, selects and narrows** at the lane width.
    #[test]
    fn reductions_and_narrows_compute_at_the_lane_width() {
        let vector = lanes(8, &[0x80, 0x01, 0xff, 0x10]);
        assert_eq!(
            vector.reduce(
                VectorReduceKind::Add,
                integer(None),
                None,
                false,
                None,
                FloatRules::default()
            ),
            Some(0x90)
        );
        assert_eq!(
            vector.reduce(
                VectorReduceKind::Max,
                integer(Some(true)),
                None,
                false,
                None,
                FloatRules::default()
            ),
            Some(0x10)
        );
        assert_eq!(
            vector.reduce(
                VectorReduceKind::Max,
                integer(Some(false)),
                None,
                false,
                None,
                FloatRules::default()
            ),
            Some(0xff)
        );
        assert_eq!(
            vector.reduce(
                VectorReduceKind::Max,
                integer(None),
                None,
                false,
                None,
                FloatRules::default()
            ),
            None
        );
        assert_eq!(
            vector.reduce(
                VectorReduceKind::Xor,
                integer(None),
                None,
                false,
                None,
                FloatRules::default()
            ),
            Some(0x6e)
        );
        assert_eq!(
            vector.reduce(
                VectorReduceKind::All,
                integer(None),
                None,
                false,
                None,
                FloatRules::default()
            ),
            Some(1)
        );
        assert_eq!(
            vector.bitmask(VectorBitmaskKind::LaneMostSignificantBits),
            Some(0b0101)
        );
        assert_eq!(
            vector.bitmask(VectorBitmaskKind::PredicateBits),
            Some(0b0110)
        );
        // Only the lanes a select chooses are read.
        let mask = lanes(8, &[0x80, 0]);
        assert_eq!(mask.select(None, Some(&lanes(8, &[7, 9]))), None);
        assert_eq!(
            lanes(8, &[0x80, 0x80]).select(None, Some(&lanes(8, &[7, 9]))),
            Some(lanes(8, &[7, 9]))
        );
        // A lane as wide as the vector moves, but computes nothing.
        let whole = lanes(128, &[3]).resized(256, 1).expect("one wide lane");
        assert_eq!(whole.relaned(64).map(|v| v.lane(0)), Some(3));
        assert_eq!(
            whole.unary(
                VectorUnaryKind::Not,
                integer(None),
                None,
                FloatRules::default()
            ),
            None
        );
        let select = lanes(8, &[1, 2]).ternary(
            &lanes(8, &[3, 4]),
            &lanes(8, &[0x80, 0x7f]),
            VectorTernaryKind::Select,
            integer(None),
            None,
            FloatRules::default(),
        );
        assert_eq!(select, Some(lanes(8, &[3, 2])));
        // A fused multiply-add rounds once: 1.5 * 1.5 + 0.25 is 2.5 exactly,
        // and the scalar form keeps the first operand's upper lanes.
        let singles = |values: &[f32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|v| u128::from(v.to_bits()))
                    .collect::<Vec<_>>(),
            )
        };
        let fma = singles(&[1.5, 2.0]).ternary(
            &singles(&[1.5, 3.0]),
            &singles(&[0.25, -6.0]),
            VectorTernaryKind::Fma,
            float(32),
            NEAREST,
            FloatRules::default(),
        );
        assert_eq!(fma, Some(singles(&[2.5, 0.0])));
        let scalar = VectorElement {
            scalar: true,
            ..float(32)
        };
        let fmass = singles(&[1.5, 2.0]).ternary(
            &singles(&[1.5, 3.0]),
            &singles(&[0.25, -6.0]),
            VectorTernaryKind::Fma,
            scalar,
            NEAREST,
            FloatRules::default(),
        );
        assert_eq!(fmass, Some(singles(&[2.5, 2.0])));
        let funnel = |kind| {
            lanes(8, &[0x81]).ternary(
                &lanes(8, &[0x42]),
                &lanes(8, &[9]),
                kind,
                integer(None),
                None,
                FloatRules::default(),
            )
        };
        assert_eq!(
            funnel(VectorTernaryKind::FunnelLeft),
            Some(lanes(8, &[0x02]))
        );
        assert_eq!(
            funnel(VectorTernaryKind::FunnelRight),
            Some(lanes(8, &[0xa1]))
        );
        let narrowed = lanes(16, &[0x7fff, 0x8000, 0x0050]).saturated(8, true, false);
        assert_eq!(narrowed, Some(lanes(8, &[0x7f, 0x80, 0x50])));
        // Signed to unsigned: a negative lane is zero, one above the maximum
        // the maximum.
        let unsigned = lanes(16, &[0xffff, 0x0123]).saturated(8, true, true);
        assert_eq!(unsigned, Some(lanes(8, &[0, 0xff])));
        let pairs = lanes(8, &[0xff, 1, 2, 3]).horizontal(8, 16, 2, false, false);
        assert_eq!(pairs, Some(lanes(16, &[0, 5])));
        let differences = lanes(8, &[1, 3, 9, 2]).horizontal(8, 16, 2, true, true);
        assert_eq!(differences, Some(lanes(16, &[0xfffe, 7])));
        assert_eq!(
            lanes(8, &[0x80, 0x05]).saturating_sign(true),
            Some(lanes(8, &[0x7f, 0xfb]))
        );
    }

    /// **A dot product sums each group's products, as its instruction states
    /// them.** `pmaddwd` of the halfwords `[1, -2, 0x7fff, -0x8000]` and
    /// `[3, 4, 0x7fff, -0x8000]` is `[-5, 0x7fff0001]`; `pmaddubsw` of the
    /// bytes `255` by `127` and `255` by `-128` saturates its pair sums to
    /// `32767` and `-32768`; `vpdpbusds` from `0x7ffffff0` saturates signed
    /// and `vpdpbuuds` from `0xfffffff0` unsigned, where the wrapping forms
    /// carry out; `sdot` from 10 of `[-1, 2, -3, 4]` by `[5, -6, 7, -8]` is
    /// `-60`.
    #[test]
    fn a_dot_product_sums_each_groups_products_as_its_instruction_states() {
        let shape = |source_bits: u32,
                     dest_bits: u32,
                     signs: (bool, bool),
                     accumulates: bool,
                     saturates: Option<bool>| DotShape {
            source_bits,
            dest_bits,
            signed_a: signs.0,
            signed_b: signs.1,
            accumulates,
            saturates,
        };
        let halves =
            |values: [i16; 4]| lanes(16, &values.map(|value| u128::from(value.cast_unsigned())));
        let bytes =
            |values: [i8; 4]| lanes(8, &values.map(|value| u128::from(value.cast_unsigned())));
        let words = |values: &[u32]| {
            lanes(
                32,
                &values
                    .iter()
                    .map(|value| u128::from(*value))
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(
            halves([1, -2, 0x7fff, -0x8000]).dot(
                &halves([3, 4, 0x7fff, -0x8000]),
                None,
                shape(16, 32, (true, true), false, None)
            ),
            Some(words(&[(-5i32).cast_unsigned(), 0x7fff_0001]))
        );
        assert_eq!(
            lanes(8, &[255, 255, 255, 255]).dot(
                &bytes([127, 127, -128, -128]),
                None,
                shape(8, 16, (false, true), false, Some(true))
            ),
            Some(lanes(16, &[0x7fff, 0x8000]))
        );
        let all = lanes(8, &[255, 255, 255, 255]);
        let high = lanes(8, &[127, 127, 127, 127]);
        let near_signed_top = words(&[0x7fff_fff0]);
        let near_unsigned_top = words(&[0xffff_fff0]);
        assert_eq!(
            all.dot(
                &high,
                Some(&near_signed_top),
                shape(8, 32, (false, true), true, Some(true))
            ),
            Some(words(&[0x7fff_ffff]))
        );
        assert_eq!(
            all.dot(
                &high,
                Some(&near_signed_top),
                shape(8, 32, (false, true), true, None)
            ),
            Some(words(&[0x7fff_fff0u32.wrapping_add(4 * 255 * 127)]))
        );
        assert_eq!(
            all.dot(
                &all,
                Some(&near_unsigned_top),
                shape(8, 32, (false, false), true, Some(false))
            ),
            Some(words(&[0xffff_ffff]))
        );
        assert_eq!(
            all.dot(
                &all,
                Some(&near_unsigned_top),
                shape(8, 32, (false, false), true, None)
            ),
            Some(words(&[0xffff_fff0u32.wrapping_add(4 * 255 * 255)]))
        );
        let sdot = DotShape {
            source_bits: 8,
            dest_bits: 32,
            signed_a: true,
            signed_b: true,
            accumulates: true,
            saturates: None,
        };
        assert_eq!(
            bytes([-1, 2, -3, 4]).dot(&bytes([5, -6, 7, -8]), Some(&words(&[10])), sdot),
            Some(words(&[(-60i32).cast_unsigned()]))
        );
        // A source that is no whole number of groups folds nothing.
        assert_eq!(lanes(8, &[1, 2]).dot(&lanes(8, &[3, 4]), None, sdot), None);
    }

    /// Bit slices cross words and lanes without changing their bit order, and
    /// an offset outside the source gives zero instead of wrapping.
    #[test]
    fn bit_regions_slice_across_lanes_and_words() {
        let source = lanes(32, &[0x0bad_f00d, 0x89ab_cdef, 0x0123_4567]);
        let bits = 0x0123_4567_89ab_cdef_0bad_f00du128;
        for offset in [0u64, 1, 7, 31, 63, 64, 95, 96, 127, 128, u64::MAX] {
            for lane_bits in [8, 16, 32, 64] {
                let result = source
                    .slice_bits(offset, lane_bits, 64 / lane_bits)
                    .unwrap();
                let expected = if offset < 128 {
                    (bits >> offset) & u128::from(u64::MAX)
                } else {
                    0
                };
                assert_eq!(
                    result.bits_at(0, 64),
                    expected,
                    "offset {offset}, lanes {lane_bits}"
                );
                assert_eq!(result.lane_bits(), lane_bits);
            }
        }
        assert!(source.slice_bits(0, 0, 1).is_none());
        assert!(source.slice_bits(0, MAX_VECTOR_BITS, 2).is_none());
    }

    /// Insertion changes exactly the covered bits, including a clipped last
    /// byte and an insertion that starts outside the destination.
    #[test]
    fn bit_regions_insert_preserves_every_other_bit() {
        let base = lanes(32, &[0x0bad_f00d, 0x89ab_cdef, 0x0123_4567]);
        let value = lanes(8, &[0x12, 0x34, 0x56]);
        let original = 0x0123_4567_89ab_cdef_0bad_f00du128;
        for offset in [0u64, 1, 7, 63, 64, 80, 95, 96, 127, u64::MAX] {
            let result = base.insert_bits(offset, &value);
            let mut expected = 0u128;
            for bit in 0u64..96 {
                let set = if bit >= offset && bit - offset < 24 {
                    (0x56_3412u128 >> (bit - offset)) & 1
                } else {
                    (original >> bit) & 1
                };
                expected |= set << bit;
            }
            assert_eq!(result.bits_at(0, 96), expected, "offset {offset}");
            assert_eq!((result.lane_bits(), result.lanes()), (32, 3));
        }
    }

    /// Concatenation puts the first part low, independent of input or output
    /// lane boundaries, and requires an exact destination width.
    #[test]
    fn bit_regions_concatenate_whole_parts_low_first() {
        let parts = [
            lanes(7, &[1, 2, 3]),
            lanes(8, &[4, 5, 6, 7, 8]),
            lanes(17, &[9, 10]),
        ];
        let result = VectorConst::concatenate(&parts, 5, 19).unwrap();
        let mut expected = 0u128;
        let mut offset = 0u32;
        for part in &parts {
            expected |= part.bits_at(0, part.bits()) << offset;
            assert_eq!(
                result
                    .slice_bits(u64::from(offset), part.lane_bits(), part.lanes())
                    .as_ref(),
                Some(part)
            );
            offset += part.bits();
        }
        assert_eq!(result.bits_at(0, 95), expected);
        assert!(VectorConst::concatenate(&parts, 8, 12).is_none());
        assert!(VectorConst::concatenate(&[], 8, 1).is_none());
    }

    /// Region copies preserve each bit across multiple 128-bit chunks, with
    /// unaligned boundaries independent of either vector's lane layout.
    #[test]
    fn bit_regions_copy_multiple_chunks() {
        let values = [
            0x1234_5678,
            0x9abc_def0,
            0x8765_4321,
            0xfedc_ba98,
            0x1111_2222,
            0x3333_4444,
            0x5555_6666,
            0x7777_8888,
            0xaaaa_bbbb,
            0xcccc_dddd,
            0xeeee_ffff,
            0x0123_4567,
        ];
        let source = lanes(32, &values);
        let bit = |at: u64| -> u128 {
            values
                .get((at / 32) as usize)
                .map_or(0, |word| (word >> (at % 32)) & 1)
        };
        for offset in [0u64, 1, 63, 127, 128, 129, 255, 383, u64::MAX] {
            let sliced = source.slice_bits(offset, 8, 32).unwrap();
            let base = VectorConst::splat(16, 32, 0xa55a).unwrap();
            let inserted = base.insert_bits(offset, &source);
            for at in 0u64..512 {
                if at < 256 {
                    let expected = offset.checked_add(at).map_or(0, bit);
                    assert_eq!(
                        sliced.bits_at(at as u32, 1),
                        expected,
                        "slice {offset}+{at}"
                    );
                }
                let expected = if at >= offset && at - offset < 384 {
                    bit(at - offset)
                } else {
                    (0xa55au128 >> (at % 16)) & 1
                };
                assert_eq!(
                    inserted.bits_at(at as u32, 1),
                    expected,
                    "insert {offset}+{at}"
                );
            }
        }
        let parts = [source.clone(), source];
        let joined = VectorConst::concatenate(&parts, 16, 48).unwrap();
        for at in 0u64..768 {
            assert_eq!(joined.bits_at(at as u32, 1), bit(at % 384));
        }
    }
    /// A sign is -1, 0 or 1 as the lane is read: signed, unsigned (a byte of
    /// 200 is positive), or a float in its format, either zero 0. An integer
    /// read neither way and a NaN have no sign, and fold to nothing.
    #[test]
    fn a_sign_is_the_lanes_reading() {
        let sign = |bits: u32, value: u128, signed: Option<bool>| {
            VectorConst::from_lanes(bits, [value])
                .unwrap()
                .unary(
                    VectorUnaryKind::Sign,
                    VectorElement::integer(bits, signed),
                    None,
                    FloatRules::default(),
                )
                .map(|lanes| lanes.lane(0))
        };
        assert_eq!(sign(8, 200, Some(false)), Some(1));
        assert_eq!(sign(8, 200, Some(true)), Some(0xff));
        assert_eq!(sign(8, 0, Some(true)), Some(0));
        assert_eq!(sign(32, 7, Some(true)), Some(1));
        assert_eq!(sign(32, 0x8000_0000, Some(true)), Some(0xffff_ffff));
        assert_eq!(sign(32, 0, Some(false)), Some(0));
        assert_eq!(sign(32, 7, None), None, "no reading, no sign");

        let float_sign = |value: f64| {
            VectorConst::from_lanes(64, [u128::from(value.to_bits())])
                .unwrap()
                .unary(
                    VectorUnaryKind::Sign,
                    VectorElement {
                        kind: VectorElementKind::Float,
                        bits: 64,
                        scalar: false,
                        signed: None,
                        rounding: None,
                    },
                    None,
                    FloatRules::default(),
                )
                .map(|lanes| f64::from_bits(u64::try_from(lanes.lane(0)).unwrap()))
        };
        assert_eq!(float_sign(-2.5), Some(-1.0));
        assert_eq!(float_sign(0.25), Some(1.0));
        assert_eq!(float_sign(f64::INFINITY), Some(1.0));
        assert_eq!(float_sign(-0.0).map(f64::to_bits), Some(0.0f64.to_bits()));
        assert_eq!(float_sign(f64::NAN), None);
    }

    #[test]
    fn integer_unary_byte_and_trailing_operations_follow_bit_oracles() {
        for bits in [8u32, 16, 32, 64, 128] {
            let mask = if bits == 128 {
                u128::MAX
            } else {
                (1u128 << bits) - 1
            };
            let values: Vec<_> = (0u128..=255)
                .chain([
                    mask,
                    1u128 << (bits - 1),
                    0x0123456789abcdef0011223344556677,
                ])
                .map(|value| value & mask)
                .collect();
            for value in values {
                let vector = VectorConst::from_lanes(bits, [value]).unwrap();
                for kind in [
                    VectorUnaryKind::TrailingZeros,
                    VectorUnaryKind::ByteSwap,
                    VectorUnaryKind::BitReverseIn8,
                ] {
                    let expected = match kind {
                        VectorUnaryKind::TrailingZeros => u128::from(
                            (0..bits)
                                .find(|bit| value & (1u128 << bit) != 0)
                                .unwrap_or(bits),
                        ),
                        VectorUnaryKind::ByteSwap => (0..bits / 8).fold(0u128, |result, byte| {
                            result | ((value >> (byte * 8)) & 255) << (bits - 8 - byte * 8)
                        }),
                        VectorUnaryKind::BitReverseIn8 => (0..bits).fold(0u128, |result, bit| {
                            result | ((value >> bit) & 1) << (bit / 8 * 8 + 7 - bit % 8)
                        }),
                        _ => unreachable!(),
                    };
                    assert_eq!(
                        vector
                            .unary(
                                kind,
                                VectorElement::integer(bits, None),
                                None,
                                FloatRules::default()
                            )
                            .unwrap()
                            .lane(0),
                        expected,
                        "{kind:?} bits={bits} value={value:x}"
                    );
                }
            }
        }
        for kind in [VectorUnaryKind::ByteSwap, VectorUnaryKind::BitReverseIn8] {
            assert!(
                VectorConst::from_lanes(7, [1])
                    .unwrap()
                    .unary(
                        kind,
                        VectorElement::integer(7, None),
                        None,
                        FloatRules::default()
                    )
                    .is_none()
            );
        }
    }
}
