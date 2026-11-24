//! Conversions of constants between types and float formats.

use crate::{
    PointerSize,
    ir::{
        float::{FloatBits, FloatFormat, FloatRounding, FloatRules},
        value::ConstValue,
    },
    target::Target,
};

impl<T: Target> ConstValue<T> {
    /// The signed integer `value` at the pointer width `width`, truncated to
    /// it.
    #[must_use]
    pub fn native(width: PointerSize, value: i64) -> Self {
        match width.bits() {
            8 => Self::I8(value as i8),
            16 => Self::I16(value as i16),
            32 => Self::I32(value as i32),
            128 => Self::NativeInt(value),
            _ => Self::I64(value),
        }
    }

    /// Converts this constant through the host's scalar type projection.
    ///
    /// Standard integer layouts share the integer conversion kernel through
    /// [`Target::convert_const`]. A host may supply additional conversion domains.
    #[must_use]
    pub fn convert_to(
        &self,
        target: &T::Type,
        unsigned_source: bool,
        ptr_bytes: u32,
    ) -> Option<Self> {
        T::convert_const(self, target, unsigned_source, ptr_bytes)
    }

    /// This constant converted to the float format `target` names, rounding
    /// by `rounding` where the result is inexact: a float to another format,
    /// or an integer, read unsigned when `unsigned_source` says so. A
    /// conversion that states no rounding rounds as
    /// [`Target::environment_rounding`] says, by [`FloatRounding::decide`]
    /// where it names no rule.
    ///
    /// A NaN converts as `rules` state ([`FloatRules::converted`]), the
    /// payload being the architecture's to decide.
    ///
    /// `None` where the target names no float format, the constant is neither
    /// a float nor an integer, the rounding is none the conversion defines
    /// (an integer rounds no way to odd), the result is a NaN the rules
    /// state no value for, or a subnormal float operand or result is one a
    /// flush mode the program may set reads or writes as zero
    /// ([`FloatRules::program_flushes_subnormals`]).
    #[must_use]
    pub fn convert_rounded(
        &self,
        target: &T::Type,
        unsigned_source: bool,
        rounding: Option<FloatRounding>,
        ptr_size: PointerSize,
        rules: FloatRules,
    ) -> Option<Self> {
        self.convert_to_format(
            T::float_format(target)?,
            unsigned_source,
            rounding,
            ptr_size,
            rules,
        )
    }

    /// Converts to an exact format under the supplied rounding, NaN and
    /// flush rules.
    ///
    /// Returns `None` when the source, rounding, or NaN result is
    /// unsupported, and for a subnormal float operand or result a flush mode
    /// the program may set reads or writes as zero. Does not inspect host
    /// type metadata.
    pub fn convert_to_format(
        &self,
        to: FloatFormat,
        unsigned_source: bool,
        rounding: Option<FloatRounding>,
        ptr_size: PointerSize,
        rules: FloatRules,
    ) -> Option<Self> {
        let result = FloatRounding::decide(
            rounding.or_else(T::environment_rounding),
            |rounding| match self {
                Self::Float(value) => Some(value.convert_rounded(to, rounding, false)?.value),
                _ => {
                    let (low, width) = self.conversion_integer_pattern(ptr_size)?;
                    Some(
                        FloatBits::from_integer_rounded(
                            to,
                            low,
                            width,
                            !unsigned_source,
                            rounding,
                        )?
                        .value,
                    )
                }
            },
        )?;
        // A flush mode reads a subnormal operand, and writes a subnormal
        // result, as zero; an integer converts to no subnormal.
        if let Self::Float(value) = self
            && !(rules.unflushed(value.format(), &[*value]) && rules.unflushed(to, &[result]))
        {
            return None;
        }
        rules.converted(result).map(Self::Float)
    }

    /// Converts this constant with overflow checking. Forwards to
    /// `Target::convert_const_checked`.
    #[must_use]
    pub fn convert_to_checked(
        &self,
        target: &T::Type,
        unsigned_source: bool,
        ptr_bytes: u32,
    ) -> Option<Self> {
        T::convert_const_checked(self, target, unsigned_source, ptr_bytes)
    }
}
