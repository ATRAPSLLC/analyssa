//! Fixed-width integer layouts and exact conversions between them.
//!
//! [`IntegerType`] is a validated 1- to 128-bit integer destination with its
//! signedness and constant representation. Conversions between widths compare
//! the full mathematical value, so a checked conversion traps exactly when the
//! value does not fit, including unsigned values above the signed 128-bit
//! maximum. Known-bit facts about such integers live in [`IntegerBits`].

pub use self::bits::IntegerBits;
use crate::{
    PointerSize,
    encoder::{ContentWriter, Emit},
    ir::{
        content::ContentSink,
        evaluation::{ArithmeticTrap, Evaluation},
        value::ConstValue,
    },
    target::Target,
};

mod bits;

/// A validated integer destination, including its constant representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntegerType {
    /// Width from one through 128 bits.
    bits: u32,
    /// Whether the destination interprets its high bit as a sign.
    signed: bool,
    /// Whether a matching native-width result uses a native constant carrier.
    native: bool,
}

impl IntegerType {
    /// Creates a fixed-width scalar integer layout, refusing unsupported widths.
    #[must_use]
    pub const fn new(bits: u32, signed: bool) -> Option<Self> {
        if bits == 0 || bits > 128 {
            return None;
        }
        Some(Self {
            bits,
            signed,
            native: false,
        })
    }

    /// Selects a native carrier when the layout matches the machine's native width.
    #[must_use]
    pub const fn native(mut self) -> Self {
        self.native = true;
        self
    }

    /// Returns the validated bit width.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// Returns whether the integer range is signed.
    #[must_use]
    pub const fn signed(self) -> bool {
        self.signed
    }

    /// Every bit of the integer: its low `bits` bits set.
    #[must_use]
    pub const fn mask(self) -> u128 {
        BitWidth::new(self.bits).mask_u128()
    }

    /// Projects an attested integer or pointer type without owning host metadata.
    #[must_use]
    pub fn of<T: Target>(ty: &T::Type, ptr_size: PointerSize) -> Option<Self> {
        let pointer = T::is_pointer(ty);
        if (!pointer && !T::is_integer(ty)) || T::vector_shape(ty).is_some() {
            return None;
        }
        let bits = T::bit_width(ty).or_else(|| pointer.then_some(ptr_size.bits()))?;
        let layout = Self::new(bits, !pointer && T::is_signed(ty))?;
        Some(if pointer && bits == ptr_size.bits() {
            layout.native()
        } else {
            layout
        })
    }

    /// Converts an integer pattern, distinguishing checked overflow from unsupported inputs.
    ///
    /// Source signedness follows `unsigned_source`, independently of its carrier.
    /// Booleans represent positive zero or one. Scalar lanes retain all 128 bits;
    /// explicit vectors, floating values and reference handles are unsupported.
    /// Checked conversion compares the complete mathematical sign and pattern,
    /// including unsigned values above the signed 128-bit maximum.
    #[must_use]
    pub fn convert<T: Target>(
        self,
        value: &ConstValue<T>,
        unsigned_source: bool,
        checked: bool,
        ptr_size: PointerSize,
    ) -> Evaluation<ConstValue<T>> {
        let source = value.conversion_integer_pattern(ptr_size);
        let Some((raw, width)) = source else {
            return Evaluation::Unsupported;
        };
        let Some(source_mask) = Self::new(width, false).map(Self::mask) else {
            return Evaluation::Unsupported;
        };
        let target_mask = self.mask();
        let source_negative = !unsigned_source && high_bit(raw, width);
        let extended = if source_negative {
            raw | !source_mask
        } else {
            raw
        };
        let pattern = extended & target_mask;
        let target_negative = self.signed && high_bit(pattern, self.bits);
        let restored = if target_negative {
            pattern | !target_mask
        } else {
            pattern
        };
        if checked && (source_negative != target_negative || extended != restored) {
            return Evaluation::Trap(ArithmeticTrap::Overflow);
        }
        self.constant(pattern, ptr_size)
            .map_or(Evaluation::Unsupported, Evaluation::Value)
    }

    /// Boxes a masked integer pattern using this destination's declared representation.
    pub fn constant<T: Target>(
        self,
        pattern: u128,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        let pattern = pattern & self.mask();
        if self.native && self.bits == ptr_size.bits() && self.bits <= 64 {
            if self.signed {
                let raw = if high_bit(pattern, self.bits) {
                    pattern | !self.mask()
                } else {
                    pattern
                };
                return Some(ConstValue::NativeInt(raw as u64 as i64));
            }
            return Some(ConstValue::NativeUInt(pattern as u64));
        }
        if self.bits == 1 && !self.signed {
            return Some(ConstValue::from_bool(pattern != 0));
        }
        match self.bits {
            8 | 16 | 32 | 64 => ConstValue::integer(self.bits, self.signed, pattern as u64 as i64),
            _ => ConstValue::of_lane(self.bits, pattern),
        }
    }
}

/// A width in bits: of an integer, a lane, a field or a shift's survivors.
///
/// It owns the low-bit masks of its width. A mask is total: a width of zero
/// sets no bit and a width at or above the word's sets every one, where the
/// shift idioms it replaces panic (`u64::MAX >> (64 - width)` at zero,
/// `(1 << width) - 1` at 64). A caller that has no meaning for those widths
/// says so through the checked forms.
///
/// # Examples
///
/// ```
/// use analyssa::ir::integer::BitWidth;
///
/// assert_eq!(BitWidth::new(12).mask(), 0xfff);
/// assert_eq!(BitWidth::new(0).mask(), 0);
/// assert_eq!(BitWidth::new(64).mask(), u64::MAX);
/// assert_eq!(BitWidth::new(0).checked_mask(), None);
/// assert_eq!(BitWidth::new(8).low_bits(0x1234), 0x34);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BitWidth(u32);

impl BitWidth {
    /// Creates the width of `bits` bits.
    #[must_use]
    pub const fn new(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the number of bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns the low bits of a 64-bit word this width covers: none for a
    /// width of zero, every bit for a width of 64 or more.
    #[must_use]
    pub const fn mask(self) -> u64 {
        match u64::MAX.checked_shr(64u32.saturating_sub(self.0)) {
            Some(mask) if self.0 > 0 => mask,
            _ => 0,
        }
    }

    /// Returns the low bits of a 128-bit word this width covers: none for a
    /// width of zero, every bit for a width of 128 or more.
    #[must_use]
    pub const fn mask_u128(self) -> u128 {
        match u128::MAX.checked_shr(128u32.saturating_sub(self.0)) {
            Some(mask) if self.0 > 0 => mask,
            _ => 0,
        }
    }

    /// Returns [`Self::mask`] for a width of 1 to 64 bits, the widths a
    /// 64-bit word holds an integer of; `None` for any other.
    #[must_use]
    pub const fn checked_mask(self) -> Option<u64> {
        if self.0 == 0 || self.0 > 64 {
            None
        } else {
            Some(self.mask())
        }
    }

    /// Returns [`Self::mask_u128`] for a width of 1 to 128 bits, the widths a
    /// 128-bit word holds an integer of; `None` for any other.
    #[must_use]
    pub const fn checked_mask_u128(self) -> Option<u128> {
        if self.0 == 0 || self.0 > 128 {
            None
        } else {
            Some(self.mask_u128())
        }
    }

    /// Returns the low bits of `value` this width covers ([`Self::mask`]).
    #[must_use]
    pub const fn low_bits(self, value: u64) -> u64 {
        value & self.mask()
    }

    /// Returns the low bits of `value` this width covers
    /// ([`Self::mask_u128`]).
    #[must_use]
    pub const fn low_bits_u128(self, value: u128) -> u128 {
        value & self.mask_u128()
    }
}

/// Admits a convention-independent scalar shift distance.
///
/// Negative patterns and counts at least as large as the value width are
/// unsupported until the frontend supplies its instruction-specific masking.
pub fn shift_distance(pattern: u128, width: u32) -> Option<u32> {
    let count = u32::try_from(pattern).ok()?;
    (count < width).then_some(count)
}

/// Returns whether the high bit of a validated scalar pattern is set.
fn high_bit(pattern: u128, width: u32) -> bool {
    width
        .checked_sub(1)
        .and_then(|shift| 1u128.checked_shl(shift))
        .is_some_and(|bit| pattern & bit != 0)
}

impl<T: Target> Emit<T> for IntegerType {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            bits,
            signed,
            native,
        } = self;
        bits.emit(writer)?;
        signed.emit(writer)?;
        native.emit(writer)
    }
}

#[cfg(test)]
mod tests {
    use super::{BitWidth, IntegerType};
    use crate::{
        PointerSize,
        ir::{
            evaluation::{ArithmeticTrap, Evaluation},
            lanes::VectorConst,
            value::ConstValue,
        },
        testing::MockTarget,
    };

    /// Exhaustive byte-domain conversions agree with mathematical integer ranges.
    #[test]
    fn conversions_match_independent_small_integer_oracles() {
        for width in 1..=8 {
            for raw in 0..(1u128 << width) {
                let source = ConstValue::<MockTarget>::of_lane(width, raw).unwrap();
                for unsigned in [false, true] {
                    let exact = if !unsigned && raw & (1 << (width - 1)) != 0 {
                        raw as i128 - (1i128 << width)
                    } else {
                        raw as i128
                    };
                    for output in 1..=10 {
                        for signed in [false, true] {
                            let (minimum, maximum) = if signed {
                                (-(1i128 << (output - 1)), (1i128 << (output - 1)) - 1)
                            } else {
                                (0, (1i128 << output) - 1)
                            };
                            let target = IntegerType::new(output, signed).unwrap();
                            for checked in [false, true] {
                                let result =
                                    target.convert(&source, unsigned, checked, PointerSize::Bit64);
                                if checked && !(minimum..=maximum).contains(&exact) {
                                    assert!(matches!(
                                        result,
                                        Evaluation::Trap(ArithmeticTrap::Overflow)
                                    ));
                                } else {
                                    let value = result.into_value().unwrap();
                                    let (actual, _) = value
                                        .conversion_integer_pattern(PointerSize::Bit64)
                                        .unwrap();
                                    let expected = exact.rem_euclid(1i128 << output) as u128;
                                    assert_eq!(
                                        actual, expected,
                                        "{width}/{raw}/{unsigned}->{output}/{signed}/{checked}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Checked full-width conversions retain unsigned values above the signed maximum.
    #[test]
    fn full_width_ranges_preserve_sign_and_unsigned_maxima() {
        let unsigned = IntegerType::new(128, false).unwrap();
        let signed = IntegerType::new(128, true).unwrap();
        let maximum = ConstValue::<MockTarget>::of_lane(128, u128::MAX).unwrap();
        assert_eq!(
            unsigned
                .convert(&maximum, true, true, PointerSize::Bit64)
                .into_value(),
            Some(maximum.clone())
        );
        assert!(matches!(
            signed.convert(&maximum, true, true, PointerSize::Bit64),
            Evaluation::Trap(ArithmeticTrap::Overflow)
        ));
        assert!(matches!(
            unsigned.convert(&maximum, false, true, PointerSize::Bit64),
            Evaluation::Trap(ArithmeticTrap::Overflow)
        ));
        let minus_one =
            IntegerType::new(8, true)
                .unwrap()
                .convert(&maximum, false, true, PointerSize::Bit64);
        assert_eq!(minus_one.into_value(), Some(ConstValue::I8(-1)));
        let minimum = ConstValue::<MockTarget>::of_lane(128, 1u128 << 127).unwrap();
        assert_eq!(
            signed
                .convert(&minimum, false, true, PointerSize::Bit64)
                .into_value(),
            Some(minimum)
        );
        let sixty_five = IntegerType::new(65, false).unwrap();
        let source = ConstValue::<MockTarget>::of_lane(65, 1u128 << 64).unwrap();
        assert_eq!(
            sixty_five
                .convert(&source, true, true, PointerSize::Bit64)
                .into_value(),
            Some(source)
        );
        assert!(matches!(
            IntegerType::new(64, false)
                .unwrap()
                .convert(&maximum, true, true, PointerSize::Bit64),
            Evaluation::Trap(ArithmeticTrap::Overflow)
        ));
    }

    /// Unsupported value domains and invalid widths supply no conversion proof.
    #[test]
    fn conversion_layouts_refuse_noninteger_domains() {
        assert_eq!(IntegerType::new(0, false), None);
        assert_eq!(IntegerType::new(129, false), None);
        let target = IntegerType::new(64, true).unwrap();
        for value in [
            ConstValue::<MockTarget>::from(1.0f64),
            ConstValue::Type(7),
            ConstValue::Symbol(7),
            ConstValue::Vector(VectorConst::from_lanes(32, [1, 2]).unwrap()),
        ] {
            assert!(matches!(
                target.convert(&value, false, true, PointerSize::Bit64),
                Evaluation::Unsupported
            ));
        }
    }

    #[test]
    fn masks_set_exactly_the_low_bits_at_every_width() {
        // Oracle: the mask built one bit at a time.
        let ones = |width: u32| (0..width.min(128)).fold(0u128, |mask, bit| mask | 1u128 << bit);
        let word = u128::from(u64::MAX);
        for bits in 0..=130 {
            let width = BitWidth::new(bits);
            let expected = ones(bits);
            assert_eq!(width.bits(), bits);
            assert_eq!(width.mask_u128(), expected, "u128 width {bits}");
            assert_eq!(
                u128::from(width.mask()),
                expected & word,
                "u64 width {bits}"
            );
            assert_eq!(
                width.checked_mask().map(u128::from),
                (1..=64).contains(&bits).then_some(expected),
                "checked u64 width {bits}"
            );
            assert_eq!(
                width.checked_mask_u128(),
                (1..=128).contains(&bits).then_some(expected),
                "checked u128 width {bits}"
            );
            // Every bit of the operand above the width is dropped.
            assert_eq!(u128::from(width.low_bits(u64::MAX)), expected & word);
            assert_eq!(width.low_bits_u128(u128::MAX), expected);
            assert_eq!(
                u128::from(width.low_bits(0xa5a5_a5a5_a5a5_a5a5)),
                0xa5a5_a5a5_a5a5_a5a5 & expected
            );
        }
        let at = |bits| BitWidth::new(bits);
        assert_eq!(
            (at(0).mask(), at(64).mask(), at(65).mask()),
            (0, u64::MAX, u64::MAX)
        );
        assert_eq!(
            (at(0).mask_u128(), at(128).mask_u128(), at(129).mask_u128()),
            (0, u128::MAX, u128::MAX)
        );
    }
}
