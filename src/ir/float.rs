//! Floating-point formats and their constants.
//!
//! A float is a format as well as a width: IEEE binary16 and bfloat16 are both
//! 16 bits wide, x87's extended precision is 80 bits with an explicit integer
//! bit, and binary128 is 128. [`FloatFormat`] names each, and [`FloatBits`]
//! holds a constant of any of them by its bits. Its arithmetic is IEEE's
//! exactly, computed by `rustc_apfloat` without host floating arithmetic. Checked
//! conveniences round as told or, told nothing, give only a result every mode
//! an environment can set rounds alike ([`FloatRounding::decide`]), and refuse
//! NaNs. Explicitly rounded operations report IEEE exceptions and apply a
//! selected NaN policy.

use core::{cmp::Ordering, fmt, marker::PhantomData};

use rustc_apfloat::{
    Float, FloatConvert, Round, Status, StatusAnd,
    ieee::{
        BFloat, BFloatS, Double, DoubleS, Half, HalfS, IeeeFloat, Quad, QuadS, Semantics, Single,
        SingleS, X87DoubleExtended, X87DoubleExtendedS,
    },
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{content::ContentSink, integer::BitWidth, ops::FlagsMask},
    target::Target,
};

/// A specified floating-point rounding rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FloatRounding {
    /// Rounds to nearest, resolving ties toward an even significand.
    NearestEven,
    /// Rounds toward zero.
    TowardZero,
    /// Rounds toward negative infinity.
    Down,
    /// Rounds toward positive infinity.
    Up,
    /// Rounds to nearest, resolving ties away from zero.
    NearestAway,
    /// Truncates a conversion and sets its least retained bit when inexact.
    ToOdd,
}

/// Writes the rule's name: `nearest`, `tozero`, `down`, `up`, `away` and
/// `odd`.
impl fmt::Display for FloatRounding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NearestEven => "nearest",
            Self::TowardZero => "tozero",
            Self::Down => "down",
            Self::Up => "up",
            Self::NearestAway => "away",
            Self::ToOdd => "odd",
        })
    }
}

impl FloatRounding {
    /// The rules a floating-point environment's mode can name: every rule but
    /// odd, which only a conversion states.
    pub const ENVIRONMENT: [Self; 5] = [
        Self::NearestEven,
        Self::TowardZero,
        Self::Down,
        Self::Up,
        Self::NearestAway,
    ];

    /// The environment's rounding mode naming this rule
    /// ([`Self::of_environment`]); `None` for odd, which no environment names.
    #[must_use]
    pub const fn environment_mode(self) -> Option<u8> {
        match self {
            Self::NearestEven => Some(0),
            Self::TowardZero => Some(1),
            Self::Down => Some(2),
            Self::Up => Some(3),
            Self::NearestAway => Some(4),
            Self::ToOdd => None,
        }
    }

    /// The rule an environment's rounding mode `mode` names, in
    /// [`VectorRounding::FloatingEnvironment`]'s encoding: 0 nearest even, 1
    /// toward zero, 2 down, 3 up, 4 nearest away. `None` for any other value,
    /// which no operation rounds by.
    ///
    /// [`VectorRounding::FloatingEnvironment`]: crate::ir::ops::VectorRounding::FloatingEnvironment
    #[must_use]
    pub const fn of_environment(mode: u64) -> Option<Self> {
        match mode {
            0 => Some(Self::NearestEven),
            1 => Some(Self::TowardZero),
            2 => Some(Self::Down),
            3 => Some(Self::Up),
            4 => Some(Self::NearestAway),
            _ => None,
        }
    }

    /// The result `compute` gives rounding by `rounding`, or, where that is
    /// `None` - an operation rounding by whatever mode the program has set
    /// (`fesetround`, RISC-V `frm`, x86 `MXCSR.RC`) - the result it gives in
    /// every mode of [`Self::ENVIRONMENT`] when they all agree.
    ///
    /// They agree on an exact result, with two exceptions IEEE 754 makes the
    /// mode's: an overflow, which rounds to the largest finite value toward
    /// zero, and an exact zero sum of operands of opposite signs, which is
    /// `-0` rounding down and `+0` otherwise. `None` where they disagree or
    /// `compute` gives nothing in one of them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::ir::float::{FloatBits, FloatRounding};
    ///
    /// let (one, three) = (FloatBits::from(1.0f64), FloatBits::from(3.0f64));
    /// let up = one.checked_div(three, Some(FloatRounding::Up)).unwrap();
    /// let nearest = one.checked_div(three, Some(FloatRounding::NearestEven)).unwrap();
    /// assert_ne!(up, nearest);
    /// // A third is inexact: the mode the program set decides it.
    /// assert_eq!(one.checked_div(three, None), None);
    /// // Three divided by three is exact in every mode.
    /// assert_eq!(three.checked_div(three, None), Some(one));
    /// ```
    pub fn decide<R: PartialEq>(
        rounding: Option<Self>,
        mut compute: impl FnMut(Self) -> Option<R>,
    ) -> Option<R> {
        let Some(rounding) = rounding else {
            let (&first, rest) = Self::ENVIRONMENT.split_first()?;
            let result = compute(first)?;
            return rest
                .iter()
                .all(|mode| compute(*mode).as_ref() == Some(&result))
                .then_some(result);
        };
        compute(rounding)
    }

    /// Returns the software IEEE rounding rule, excluding conversion-only odd rounding.
    const fn arithmetic(self) -> Option<Round> {
        match self {
            Self::NearestEven => Some(Round::NearestTiesToEven),
            Self::TowardZero => Some(Round::TowardZero),
            Self::Down => Some(Round::TowardNegative),
            Self::Up => Some(Round::TowardPositive),
            Self::NearestAway => Some(Round::NearestTiesToAway),
            Self::ToOdd => None,
        }
    }
}

/// A set of floating-point exception flags: IEEE 754's five, ordered from
/// inexact in bit zero to invalid in bit four, the input-denormal flag two
/// architectures add in bit five, and in bit six the direction of an inexact
/// result's rounding, which x87 reports with them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FloatExceptions(u8);

impl FloatExceptions {
    /// No exception.
    pub const NONE: Self = Self(0);
    /// The mathematical result differs from the represented result.
    pub const INEXACT: Self = Self(1);
    /// An inexact result is tiny after rounding.
    pub const UNDERFLOW: Self = Self(2);
    /// A finite mathematical result exceeds the destination range.
    pub const OVERFLOW: Self = Self(4);
    /// Division of a nonzero finite value by zero produces infinity.
    pub const DIVIDE_BY_ZERO: Self = Self(8);
    /// An operation is invalid or consumes a signaling NaN.
    pub const INVALID: Self = Self(16);
    /// An operand is subnormal: x86's denormal-operand flag, which an
    /// operation sets unless denormals read as zero, and Arm's input-denormal
    /// flag, which it sets where flush-to-zero flushes the operand. Which the
    /// operation does is the mode the program has set.
    pub const INPUT_DENORMAL: Self = Self(32);
    /// An inexact result was rounded up: its magnitude exceeds the exact
    /// result's. No exception; x87 reports it in `C1` after each instruction
    /// that rounds, afresh rather than accrued.
    pub const ROUNDED_UP: Self = Self(64);
    /// Every flag.
    pub const ALL: Self = Self(127);
    /// IEEE 754's five flags, every flag but input-denormal.
    pub const IEEE: Self = Self(31);

    /// Returns the seven flag bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The set whose flags are the set bits of `bits`, the top bit ignored.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & Self::ALL.0)
    }

    /// Returns whether every flag in `other` is present.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns whether no flag is present.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the union of two exception sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the flags present in both sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns the flags of this set that `other` does not hold.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// The name of each flag, in bit order.
    pub const NAMES: [&'static str; 7] = [
        "inexact",
        "underflow",
        "overflow",
        "divzero",
        "invalid",
        "denormal",
        "roundup",
    ];

    /// The names of the flags present, in bit order ([`Self::NAMES`]).
    fn names(self) -> impl Iterator<Item = &'static str> {
        Self::NAMES
            .into_iter()
            .enumerate()
            .filter(move |(bit, _)| self.0 & (1 << bit) != 0)
            .map(|(_, name)| name)
    }
}

/// Writes the flags' names joined by `|` in bit order (`inexact|invalid`), or
/// `none` for the empty set.
impl fmt::Display for FloatExceptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        for (index, name) in self.names().enumerate() {
            if index > 0 {
                f.write_str("|")?;
            }
            f.write_str(name)?;
        }
        Ok(())
    }
}

/// Where a status register keeps each exception flag: the bit of each flag of
/// [`FloatExceptions`], or none for a flag the register does not hold.
///
/// RISC-V's `fflags` holds IEEE's five in [`FloatExceptions`]' own order
/// ([`Self::IEEE`]); x86's MXCSR and x87 status word, Arm's FPSR and FPSCR
/// and MIPS's FCSR each order them their own way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FloatExceptionLayout {
    /// The bit of each flag, indexed by the flag's bit in [`FloatExceptions`].
    bits: [Option<u8>; 7],
}

impl FloatExceptionLayout {
    /// IEEE's five flags in [`FloatExceptions`]' order from bit zero, without
    /// input-denormal: RISC-V's `fflags`.
    pub const IEEE: Self = Self::new([Some(0), Some(1), Some(2), Some(3), Some(4), None]);

    /// The layout keeping inexact, underflow, overflow, divide-by-zero,
    /// invalid and input-denormal, in that order, at the bits `bits` names;
    /// a bit above 63 holds nothing. It keeps no rounding direction
    /// ([`Self::rounded_up_at`]).
    #[must_use]
    pub const fn new(bits: [Option<u8>; 6]) -> Self {
        let [inexact, underflow, overflow, divide, invalid, denormal] = bits;
        Self {
            bits: [
                Self::held(inexact),
                Self::held(underflow),
                Self::held(overflow),
                Self::held(divide),
                Self::held(invalid),
                Self::held(denormal),
                None,
            ],
        }
    }

    /// This layout keeping [`FloatExceptions::ROUNDED_UP`] at `bit`: x87's
    /// `C1`.
    #[must_use]
    pub const fn rounded_up_at(mut self, bit: u8) -> Self {
        self.bits[6] = Self::held(Some(bit));
        self
    }

    /// `bit`, where a 64-bit register has it.
    const fn held(bit: Option<u8>) -> Option<u8> {
        match bit {
            Some(bit) if bit < 64 => Some(bit),
            _ => None,
        }
    }

    /// The bit each flag is kept at, in [`FloatExceptions`]' order.
    #[must_use]
    pub const fn bits(self) -> [Option<u8>; 7] {
        self.bits
    }

    /// The flags the register holds.
    #[must_use]
    pub fn holds(self) -> FloatExceptions {
        self.flags()
            .filter(|(_, bit)| bit.is_some())
            .fold(FloatExceptions::NONE, |set, (flag, _)| set.union(flag))
    }

    /// The register bits `exceptions` sets: each held flag's bit, the flags
    /// the register does not hold dropped.
    #[must_use]
    pub fn place(self, exceptions: FloatExceptions) -> u64 {
        self.flags()
            .filter(|(flag, _)| exceptions.contains(*flag))
            .filter_map(|(_, bit)| bit)
            .fold(0, |bits, bit| bits | 1u64.wrapping_shl(u32::from(bit)))
    }

    /// The flags `bits` of the register hold.
    #[must_use]
    pub fn read(self, bits: u64) -> FloatExceptions {
        self.flags()
            .filter(|(_, bit)| bit.is_some_and(|bit| bits.wrapping_shr(u32::from(bit)) & 1 != 0))
            .fold(FloatExceptions::NONE, |set, (flag, _)| set.union(flag))
    }

    /// Each flag with the bit it is kept at.
    fn flags(self) -> impl Iterator<Item = (FloatExceptions, Option<u8>)> {
        self.bits
            .into_iter()
            .enumerate()
            .map(|(index, bit)| (FloatExceptions(1 << index), bit))
    }
}

/// The layout as one word an instruction can state: byte `i` is one past the
/// bit flag `i` of [`FloatExceptions`] is kept at, or zero where the register
/// does not hold the flag.
impl From<FloatExceptionLayout> for u64 {
    fn from(layout: FloatExceptionLayout) -> Self {
        layout
            .bits
            .into_iter()
            .enumerate()
            .fold(0, |word, (index, bit)| {
                let byte = bit.map_or(0, |bit| u64::from(bit).saturating_add(1));
                word | byte.wrapping_shl(u32::try_from(index).unwrap_or(0).wrapping_mul(8))
            })
    }
}

/// The layout a word states ([`u64::from`]): a flag whose byte names no bit a
/// 64-bit register has is held nowhere.
impl From<u64> for FloatExceptionLayout {
    fn from(word: u64) -> Self {
        let mut bits = [None; 7];
        for (index, bit) in bits.iter_mut().enumerate() {
            let byte = word.wrapping_shr(u32::try_from(index).unwrap_or(0).wrapping_mul(8)) & 0xff;
            *bit = Self::held(u8::try_from(byte).ok().and_then(|byte| byte.checked_sub(1)));
        }
        Self { bits }
    }
}

impl From<Status> for FloatExceptions {
    fn from(status: Status) -> Self {
        Self(
            u8::from(status.contains(Status::INEXACT))
                | (u8::from(status.contains(Status::UNDERFLOW)) << 1)
                | (u8::from(status.contains(Status::OVERFLOW)) << 2)
                | (u8::from(status.contains(Status::DIV_BY_ZERO)) << 3)
                | (u8::from(status.contains(Status::INVALID_OP)) << 4),
        )
    }
}

/// A rounded value and the exceptions generated by its computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatResult {
    /// The represented result, including its specified NaN encoding.
    pub value: FloatBits,
    /// Exceptions before sticky accumulation.
    pub exceptions: FloatExceptions,
}

/// An integer bit pattern and exceptions generated by a floating-point conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatIntegerResult {
    /// The integer bits, with signed values encoded in two's complement.
    pub bits: u128,
    /// Exceptions before sticky accumulation.
    pub exceptions: FloatExceptions,
}

/// What a conversion of a float to an integer gives for a value its integer
/// does not hold, a NaN or an infinity: the processor's rule, which each
/// architecture fixes for itself and a host states in its
/// [`FloatRules`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutOfRangeConversion {
    /// The integer indefinite (x86): the most negative signed integer of the
    /// width, or every bit set for an unsigned destination.
    Indefinite,
    /// The nearest integer the width holds (Arm, RISC-V), a NaN giving zero
    /// or, when `nan_maximum`, the largest integer (RISC-V).
    Saturate {
        /// Whether a NaN converts to the largest integer, rather than zero.
        nan_maximum: bool,
    },
    /// The largest signed integer of the width for every value it does not
    /// hold, a NaN, an infinity and either overflow alike (MIPS's legacy NaN
    /// mode: `2^31 - 1` from `cvt.w.fmt`, `2^63 - 1` from `cvt.l.fmt`). MIPS
    /// converts only to signed integers; an unsigned destination is given the
    /// same bits.
    Maximum,
}

impl OutOfRangeConversion {
    /// The integer `value` converts to, rounded toward zero, at `width` bits
    /// (one through 128), read signed or not: the exact integer where the
    /// width holds it, this rule's otherwise.
    ///
    /// # Returns
    ///
    /// The integer's bits, two's complement for a signed one, or `None` for
    /// another width.
    #[must_use]
    pub fn convert(self, value: FloatBits, width: u32, signed: bool) -> Option<u128> {
        let nan_maximum = matches!(self, Self::Saturate { nan_maximum: true });
        let converted =
            value.to_integer_clamped(width, signed, nan_maximum, FloatRounding::TowardZero)?;
        if matches!(self, Self::Saturate { .. })
            || !converted.exceptions.contains(FloatExceptions::INVALID)
        {
            return Some(converted.bits);
        }
        let mask = BitWidth::new(width).checked_mask_u128()?;
        Some(match self {
            Self::Indefinite if signed => 1u128.checked_shl(width.checked_sub(1)?)?,
            Self::Indefinite => mask,
            _ => mask >> 1,
        })
    }
}

/// The NaN an invalid operation gives when no operand is a NaN (infinity
/// times zero, the square root of a negative number): the processor's own,
/// which each architecture fixes for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefaultNan {
    /// The positive quiet NaN with an empty payload (Arm, RISC-V, MIPS's 2008
    /// mode): [`FloatFormat::canonical_nan_bits`].
    Positive,
    /// The same NaN negative (x86's "QNaN floating-point indefinite").
    Negative,
    /// MIPS's legacy NaN: positive, the quiet bit clear (legacy MIPS reads a
    /// set one as signaling) and every other fraction bit set, for a single
    /// and a double.
    MipsLegacy,
}

impl DefaultNan {
    /// The bits of this NaN in `format`, `None` for a format the rule does not
    /// name it in.
    #[must_use]
    pub const fn bits(self, format: FloatFormat) -> Option<u128> {
        let positive = format.canonical_nan_bits();
        Some(match self {
            Self::Positive => positive,
            Self::Negative => positive | 1u128.wrapping_shl(format.bits().saturating_sub(1)),
            Self::MipsLegacy => match format {
                FloatFormat::Single => 0x7fbf_ffff,
                FloatFormat::Double => 0x7ff7_ffff_ffff_ffff,
                _ => return None,
            },
        })
    }

    /// This NaN in `format`.
    #[must_use]
    pub const fn of(self, format: FloatFormat) -> Option<FloatBits> {
        match self.bits(format) {
            Some(bits) => FloatBits::new(format, bits),
            None => None,
        }
    }
}

/// What a conversion between float formats gives of a NaN, whose payload IEEE
/// leaves to the processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NanPropagation {
    /// The NaN itself, quiet, its sign and the top of its payload kept (x86's
    /// `cvtsd2ss` and x87's `fst`: the x87 indefinite is the double
    /// `0xfff8000000000000`).
    Operand,
    /// The default NaN ([`FloatRules::default_nan`]), whatever the operand
    /// (RISC-V's canonical NaN).
    Default,
}

/// Which NaN an arithmetic operation gives of operands one or more of which is
/// a NaN, which IEEE 754 leaves to the processor: the sum, difference,
/// product, quotient and square root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NanOperands {
    /// The first NaN in the operands' order, quieted: x86's SSE and AVX
    /// arithmetic, the first source's NaN where it is one and else the
    /// second's (Intel SDM vol. 1, table 4-7).
    First,
    /// A quiet NaN over a signalling one, and of two of one kind the one of
    /// the larger significand, quieted: x87's arithmetic (Intel SDM vol. 1,
    /// table 4-7). Two that differ only in sign are left undecided.
    LargerSignificand,
    /// The target's default NaN ([`FloatRules::default_nan`]), whatever the
    /// operands: RISC-V's canonical NaN.
    Default,
}

/// The float operations whose raised exceptions fold
/// ([`FloatRules::raises`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaisingArithmetic {
    /// A sum, difference, product or quotient of two values.
    Binary(FloatBinary),
    /// The square root of one value.
    Sqrt,
    /// The fused `a * b + c` of three values.
    MulAdd,
    /// The round of one value to an integral one that signals inexact
    /// (`roundsd`'s exact form, the round `cvtsd2si` makes first).
    RoundExact,
    /// The conversion of one value to a signed or unsigned integer of
    /// `width` bits, truncated.
    ToInteger {
        /// The integer's width.
        width: u32,
        /// Whether the integer is signed.
        signed: bool,
    },
    /// The compare of two values, which raises invalid of a signalling NaN,
    /// and of a quiet one where it signals (`comisd` beside `ucomisd`).
    Compare {
        /// Whether a quiet NaN raises invalid too.
        signaling: bool,
    },
    /// The conversion of one value to another float format.
    Convert(FloatFormat),
    /// The seven-bit reciprocal or reciprocal square root estimate of one
    /// value ([`FloatBits::estimate7`]).
    Estimate7 {
        /// Whether the estimate is of the reciprocal square root.
        square_root: bool,
    },
    /// The reduction of the values after the first into the first, each
    /// step rounded ([`FloatBits::reduce_rounded`]): in source order where
    /// `ordered`, and otherwise in an order the operation does not state,
    /// which raises one set of flags only where every order adds exactly
    /// ([`FloatBits::sum_in_any_order`]) or the operation is a minimum or
    /// maximum, whose flags no order changes.
    Reduce {
        /// The step: an add, a minimum or a maximum.
        operation: FloatBinary,
        /// Whether the steps run in source order.
        ordered: bool,
    },
}

/// What a target's float operations give where IEEE leaves the result to the
/// processor: a float-to-integer conversion out of range, the NaN of an
/// invalid operation, the NaN arithmetic gives of NaN operands, and a NaN
/// converted between float formats. The shared evaluator folds a constant
/// operation by them where the host states them, and refuses it where not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FloatRules {
    /// What a conversion to an integer gives out of range.
    pub out_of_range: Option<OutOfRangeConversion>,
    /// The NaN an invalid operation with no NaN operand gives.
    pub default_nan: Option<DefaultNan>,
    /// What a conversion between float formats gives of a NaN; `None` where
    /// it depends on state no rule states (Arm's `FPCR.DN`).
    pub nan_conversion: Option<NanPropagation>,
    /// Which NaN arithmetic gives of NaN operands of the IEEE interchange
    /// formats; `None` where it depends on state no rule states (Arm's
    /// `FPCR.DN`) or the rule is not known.
    pub nan_operands: Option<NanOperands>,
    /// The same of the x87 extended format, which a target computing in it
    /// treats by rules of its own (x86's x87 unit beside SSE).
    pub extended_nan_operands: Option<NanOperands>,
    /// Whether the program may set a mode that flushes subnormal operands or
    /// results to zero, which it sets at run time as it does the rounding:
    /// x86's `MXCSR.FTZ` and `DAZ`, Arm's `FPCR.FZ`, MIPS's `FS`. A fold of
    /// a subnormal operand or result then declines; the x87 extended format,
    /// which no flush mode reaches, folds it.
    pub program_flushes_subnormals: bool,
    /// Which of a NaN's kinds its quiet bit encodes ([`NanEncoding`]): a
    /// raise reads a NaN operand's kind in it ([`Self::raises`]), and a NaN
    /// operand's value folds by an operand rule only in IEEE's
    /// ([`Self::propagated`]).
    pub nan_encoding: NanEncoding,
}

/// How a target encodes a NaN's kind in its quiet bit, the top bit of the
/// fraction ([`FloatFormat::quiet_bit`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NanEncoding {
    /// IEEE 754-2008's: a NaN is quiet where the bit is set
    /// ([`FloatBits::is_signaling`]).
    #[default]
    Ieee,
    /// Legacy MIPS's, the reverse: quiet where the bit is clear
    /// ([`FloatBits::from_legacy_nan`]).
    MipsLegacy,
    /// Not stated: a MIPS function whose image's standard is not known. A
    /// fold that reads a NaN's kind declines.
    Unstated,
}

impl FloatRules {
    /// The rules of a target that states none: every operation IEEE leaves
    /// to the processor folds nothing (a bytecode's, whose runtimes differ).
    pub const NONE: Self = Self {
        out_of_range: None,
        default_nan: None,
        nan_conversion: None,
        nan_operands: None,
        extended_nan_operands: None,

        program_flushes_subnormals: false,

        nan_encoding: NanEncoding::Ieee,
    };

    /// Whether a value of `format` computed of `values` is one the flush
    /// mode the program may set leaves alone: none of them subnormal, or no
    /// flush mode reaching the format ([`Self::program_flushes_subnormals`]).
    #[must_use]
    pub fn unflushed(self, format: FloatFormat, values: &[FloatBits]) -> bool {
        !self.program_flushes_subnormals
            || format == FloatFormat::X87Extended
            || !values.iter().any(|value| value.is_subnormal())
    }

    /// The exceptions `arithmetic` of `values` raises, of the flags `held`:
    /// the same in every rounding mode the program may set where `rounding`
    /// is `None`, with [`FloatExceptions::ROUNDED_UP`] (x87's `C1`) set where
    /// a rounded result's magnitude exceeds the one rounded toward zero, the
    /// exact one's. `None` where they are not decided: a subnormal input when
    /// input-denormal status is requested or a possible flush mode can read it
    /// as zero, a result a flush
    /// mode the program may set would write as zero
    /// ([`Self::program_flushes_subnormals`]), and a NaN whose encoding the
    /// target does not state ([`NanEncoding::Unstated`]). A NaN the target
    /// encodes in reverse raises by its kind in that encoding
    /// ([`FloatBits::from_legacy_nan`]).
    #[must_use]
    pub fn raises(
        self,
        arithmetic: RaisingArithmetic,
        values: &[FloatBits],
        rounding: Option<FloatRounding>,
        held: FloatExceptions,
    ) -> Option<FloatExceptions> {
        let format = values.first()?.format();
        if values.iter().any(|value| value.is_subnormal())
            && (held.contains(FloatExceptions::INPUT_DENORMAL)
                || self.program_flushes_subnormals && format != FloatFormat::X87Extended)
        {
            return None;
        }
        // A NaN's kind in the target's encoding: what it raises reads that
        // alone.
        let legacy: Vec<FloatBits>;
        let values = match self.nan_encoding {
            NanEncoding::Ieee => values,
            NanEncoding::MipsLegacy => {
                legacy = values.iter().map(|value| value.from_legacy_nan()).collect();
                legacy.as_slice()
            }
            NanEncoding::Unstated if values.iter().any(|value| value.is_nan()) => return None,
            NanEncoding::Unstated => values,
        };
        // The exceptions of one mode, and the float result where there is
        // one, which a flush mode may write as zero and `C1` reads.
        let compute = |rounding: FloatRounding| -> Option<(FloatExceptions, Option<FloatBits>)> {
            let result = match (arithmetic, values) {
                (RaisingArithmetic::Binary(operation), [a, b]) => {
                    a.binary_rounded(*b, operation, rounding, true)?
                }
                (RaisingArithmetic::Sqrt, [a]) => a.sqrt_rounded(rounding, true)?,
                (RaisingArithmetic::MulAdd, [a, b, c]) => {
                    a.mul_add_rounded(*b, *c, rounding, true)?
                }
                (RaisingArithmetic::Convert(to), [a]) => a.convert_rounded(to, rounding, true)?,
                (RaisingArithmetic::RoundExact, [a]) => {
                    let exceptions = if a.is_nan() {
                        if a.is_signaling() {
                            FloatExceptions::INVALID
                        } else {
                            FloatExceptions::NONE
                        }
                    } else if a.integral(rounding)?.bits() == a.bits() {
                        FloatExceptions::NONE
                    } else {
                        FloatExceptions::INEXACT
                    };
                    return Some((exceptions, None));
                }
                (RaisingArithmetic::ToInteger { width, signed }, [a]) => {
                    let converted = a.to_integer_clamped(width, signed, false, rounding)?;
                    return Some((converted.exceptions, None));
                }
                (RaisingArithmetic::Compare { signaling }, [a, b]) => {
                    let (_, exceptions) =
                        a.compare_with_exceptions(*b, FloatComparison::Equal, signaling)?;
                    return Some((exceptions, None));
                }
                (RaisingArithmetic::Estimate7 { square_root }, [a]) => {
                    a.estimate7(square_root, rounding, true)?
                }
                (RaisingArithmetic::Reduce { operation, ordered }, [seed, rest @ ..]) => {
                    // Every partial result is a value a flush mode may write
                    // as zero, which the final one alone does not show.
                    if self.program_flushes_subnormals {
                        return None;
                    }
                    if ordered || rest.len() < 2 || operation != FloatBinary::Add {
                        seed.reduce_rounded(operation, rest, rounding, true)?
                    } else {
                        // Every order adds exactly: no flag in any of them.
                        FloatResult {
                            value: FloatBits::sum_in_any_order(values, Some(rounding))?,
                            exceptions: FloatExceptions::NONE,
                        }
                    }
                }
                _ => return None,
            };
            Some((result.exceptions, Some(result.value)))
        };
        FloatRounding::decide(rounding, |rounding| {
            let (mut exceptions, value) = compute(rounding)?;
            if let Some(value) = value {
                let flushed = self.program_flushes_subnormals
                    && format != FloatFormat::X87Extended
                    && exceptions.contains(FloatExceptions::UNDERFLOW);
                if flushed || !self.unflushed(format, &[value]) {
                    return None;
                }
                if exceptions.contains(FloatExceptions::INEXACT) {
                    let (_, toward_zero) = compute(FloatRounding::TowardZero)?;
                    if value.abs().bits() != toward_zero?.abs().bits() {
                        exceptions = exceptions.union(FloatExceptions::ROUNDED_UP);
                    }
                }
            }
            Some(exceptions.intersection(held))
        })
    }

    /// The NaN an arithmetic operation of `operands`, of one format and one
    /// or more of them a NaN, gives by the rule for that format
    /// ([`Self::nan_operands`], [`Self::extended_nan_operands`]); `None`
    /// where the rules state none or leave the choice undecided.
    #[must_use]
    pub fn propagated(self, operands: &[FloatBits]) -> Option<FloatBits> {
        let format = operands.first()?.format();
        if operands.iter().any(|operand| operand.format() != format) {
            return None;
        }
        let rule = if format == FloatFormat::X87Extended {
            self.extended_nan_operands
        } else {
            self.nan_operands
        }?;
        // Which NaN is quiet is the reverse of what the bits read here say.
        if self.nan_encoding != NanEncoding::Ieee && rule != NanOperands::Default {
            return None;
        }
        let mut nans = operands.iter().copied().filter(|operand| operand.is_nan());
        match rule {
            NanOperands::First => nans.next().map(FloatBits::quieted),
            NanOperands::LargerSignificand => {
                let nans: Vec<FloatBits> = nans.collect();
                let quiet: Vec<FloatBits> = nans
                    .iter()
                    .copied()
                    .filter(|nan| !nan.is_signaling())
                    .collect();
                let pool = if quiet.is_empty() { nans } else { quiet };
                let largest = pool.iter().map(|nan| nan.significand()).max()?;
                let mut chosen = pool.iter().filter(|nan| nan.significand() == largest);
                let first = *chosen.next()?;
                chosen
                    .all(|other| other.bits() == first.bits())
                    .then(|| first.quieted())
            }
            NanOperands::Default => self.default_nan?.of(format),
        }
    }

    /// What a conversion that computed `result` gives: `result` itself unless
    /// it is a NaN, whose value [`Self::nan_conversion`] states, and `None`
    /// where the rules state none.
    ///
    /// `result` is the conversion IEEE propagates (quiet, the sign and the top
    /// of the payload kept), which [`NanPropagation::Operand`] keeps.
    #[must_use]
    pub fn converted(self, result: FloatBits) -> Option<FloatBits> {
        if !result.is_nan() {
            return Some(result);
        }
        match self.nan_conversion? {
            NanPropagation::Operand => Some(result),
            NanPropagation::Default => self.default_nan?.of(result.format()),
        }
    }
}

/// A floating-point binary arithmetic relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatBinary {
    /// Addition.
    Add,
    /// Subtraction.
    Subtract,
    /// Multiplication.
    Multiply,
    /// Division.
    Divide,
    /// MinimumNumber, preferring negative zero on a tie.
    MinimumNumber,
    /// MaximumNumber, preferring positive zero on a tie.
    MaximumNumber,
}

/// A floating-point comparison, false on unordered inputs except inequality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatComparison {
    /// Equality.
    Equal,
    /// Inequality, including unordered inputs.
    NotEqual,
    /// Strictly less than.
    Less,
    /// Less than or equal.
    LessEqual,
    /// Strictly greater than.
    Greater,
    /// Greater than or equal.
    GreaterEqual,
}

/// The format of a floating-point value.
///
/// The discriminants are the format's code where a host stores one
/// (`From<FloatFormat> for u8`, `TryFrom<u8>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum FloatFormat {
    /// IEEE 754 binary16: 1 sign, 5 exponent and 10 fraction bits.
    Half = 0,
    /// bfloat16: binary32's sign and 8 exponent bits with 7 fraction bits.
    BFloat16 = 1,
    /// IEEE 754 binary32.
    Single = 2,
    /// IEEE 754 binary64.
    Double = 3,
    /// The x87 80-bit extended format: 1 sign, 15 exponent bits and a 64-bit
    /// significand whose integer bit is explicit.
    X87Extended = 4,
    /// IEEE 754 binary128.
    Quad = 5,
}

/// The format's code.
impl From<FloatFormat> for u8 {
    fn from(format: FloatFormat) -> Self {
        format as u8
    }
}

/// The format a code names; the code itself as the error when it names none.
impl TryFrom<u8> for FloatFormat {
    type Error = u8;

    fn try_from(code: u8) -> Result<Self, Self::Error> {
        Self::ALL
            .into_iter()
            .find(|format| u8::from(*format) == code)
            .ok_or(code)
    }
}

impl FloatFormat {
    /// Every format, narrowest first.
    pub const ALL: [Self; 6] = [
        Self::Half,
        Self::BFloat16,
        Self::Single,
        Self::Double,
        Self::X87Extended,
        Self::Quad,
    ];

    /// The width of the format's exponent field.
    #[must_use]
    pub const fn exponent_bits(self) -> u32 {
        match self {
            Self::Half => 5,
            Self::BFloat16 | Self::Single => 8,
            Self::Double => 11,
            Self::X87Extended | Self::Quad => 15,
        }
    }

    /// The width of the significand bits below the exponent field: the
    /// fraction, and for the x87 extended format its explicit integer bit
    /// too.
    #[must_use]
    pub const fn fraction_bits(self) -> u32 {
        match self {
            Self::Half => 10,
            Self::BFloat16 => 7,
            Self::Single => 23,
            Self::Double => 52,
            Self::X87Extended => 64,
            Self::Quad => 112,
        }
    }

    /// The bits of precision the format's significand carries: its fraction
    /// and the leading bit, explicit in the x87 extended format and implied
    /// in the others.
    #[must_use]
    pub const fn precision(self) -> u32 {
        match self {
            Self::X87Extended => 64,
            _ => self.fraction_bits().saturating_add(1),
        }
    }

    /// Whether every value of `other` is a value of this format: it carries
    /// as much precision and as wide an exponent, so a conversion from
    /// `other` is exact (single to double, half to single, either to x87
    /// extended).
    #[must_use]
    pub const fn includes(self, other: Self) -> bool {
        self.precision() >= other.precision() && self.exponent_bits() >= other.exponent_bits()
    }

    /// Whether every integer `bits` wide, read signed or unsigned, is a value
    /// of this format, so a conversion of one is exact (a 32-bit integer to a
    /// double, not to a single).
    #[must_use]
    pub const fn includes_integer(self, bits: u32, signed: bool) -> bool {
        let magnitude = if signed { bits.saturating_sub(1) } else { bits };
        let largest_power = (1u32 << self.exponent_bits().saturating_sub(1)).saturating_sub(1);
        magnitude <= self.precision() && magnitude <= largest_power.saturating_add(1)
    }

    /// The width of a value of the format, in bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::Half | Self::BFloat16 => 16,
            Self::Single => 32,
            Self::Double => 64,
            Self::X87Extended => 80,
            Self::Quad => 128,
        }
    }

    /// The IEEE (or x87) format of a float `bits` wide: binary16 at 16 bits,
    /// x87 extended at 80. bfloat16 is never inferred from a width.
    #[must_use]
    pub const fn of_width(bits: u32) -> Option<Self> {
        match bits {
            16 => Some(Self::Half),
            32 => Some(Self::Single),
            64 => Some(Self::Double),
            80 => Some(Self::X87Extended),
            128 => Some(Self::Quad),
            _ => None,
        }
    }

    /// The format as the SSA listing spells it: `f16`, `bf16`, `f32`, `f64`,
    /// `f80`, `f128`.
    #[must_use]
    pub const fn printed(self) -> &'static str {
        match self {
            Self::Half => "f16",
            Self::BFloat16 => "bf16",
            Self::Single => "f32",
            Self::Double => "f64",
            Self::X87Extended => "f80",
            Self::Quad => "f128",
        }
    }

    /// The format's name, as its specification spells it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Half => "binary16",
            Self::BFloat16 => "bfloat16",
            Self::Single => "binary32",
            Self::Double => "binary64",
            Self::X87Extended => "x87 extended",
            Self::Quad => "binary128",
        }
    }

    /// The bits of the format's value `bits` converted to the format `to`,
    /// rounding to nearest with ties to even. `None` for a NaN, whose payload
    /// the architecture converting it decides, and for bits wider than the
    /// format.
    #[must_use]
    pub fn convert(self, bits: u128, to: Self) -> Option<u128> {
        if !self.holds(bits) {
            return None;
        }
        macro_rules! from {
            ($from:ty) => {{
                let value = <$from>::from_bits(bits);
                if value.is_nan() {
                    return None;
                }
                match to {
                    Self::Half => convert_to::<$from, Half>(value),
                    Self::BFloat16 => convert_to::<$from, BFloat>(value),
                    Self::Single => convert_to::<$from, Single>(value),
                    Self::Double => convert_to::<$from, Double>(value),
                    Self::X87Extended => convert_to::<$from, X87DoubleExtended>(value),
                    Self::Quad => convert_to::<$from, Quad>(value),
                }
            }};
        }
        match self {
            Self::Half => from!(Half),
            Self::BFloat16 => from!(BFloat),
            Self::Single => from!(Single),
            Self::Double => from!(Double),
            Self::X87Extended => from!(X87DoubleExtended),
            Self::Quad => from!(Quad),
        }
    }

    /// The bits of the integer `value` in the format, rounded to nearest
    /// with ties to even; one too large for the format is its infinity.
    #[must_use]
    pub fn from_integer(self, value: i128) -> u128 {
        macro_rules! of {
            ($float:ty) => {
                <$float>::from_i128_r(value, Round::NearestTiesToEven)
                    .value
                    .to_bits()
            };
        }
        match self {
            Self::Half => of!(Half),
            Self::BFloat16 => of!(BFloat),
            Self::Single => of!(Single),
            Self::Double => of!(Double),
            Self::X87Extended => of!(X87DoubleExtended),
            Self::Quad => of!(Quad),
        }
    }

    /// Returns the positive quiet NaN with an otherwise empty payload.
    #[must_use]
    pub const fn canonical_nan_bits(self) -> u128 {
        match self {
            Self::Half => 0x7e00,
            Self::BFloat16 => 0x7fc0,
            Self::Single => 0x7fc0_0000,
            Self::Double => 0x7ff8_0000_0000_0000,
            Self::X87Extended => (0x7fffu128 << 64) | 0xc000_0000_0000_0000,
            Self::Quad => 0x7fff_8000_0000_0000_0000_0000_0000_0000,
        }
    }

    /// The quiet bit of the format's NaNs: the top bit of the fraction,
    /// below x87's explicit integer bit, the lowest bit of the canonical NaN
    /// in every format.
    #[must_use]
    pub const fn quiet_bit(self) -> u128 {
        1u128 << self.canonical_nan_bits().trailing_zeros()
    }

    /// Whether `bits` fits the format's width.
    const fn holds(self, bits: u128) -> bool {
        self.bits() == 128 || bits >> self.bits() == 0
    }
}

/// `value` converted to `To`, rounding to nearest with ties to even.
fn convert_to<From: FloatConvert<To>, To: Float>(value: From) -> Option<u128> {
    let mut loses_info = false;
    let converted = value
        .convert_r(Round::NearestTiesToEven, &mut loses_info)
        .value;
    Some(converted.to_bits())
}

/// The encoding a float classification states a value's class in, the
/// classifying instruction's own ([`crate::ir::SsaOp::FpClassify`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FloatClassLayout {
    /// RISC-V's `fclass`: one bit of ten, negative infinity (bit 0), a
    /// negative normal, subnormal and zero, a positive zero, subnormal and
    /// normal, positive infinity (bit 7), a signaling NaN and a quiet NaN
    /// (bit 9).
    RiscV,
    /// MIPS's `class.fmt` and MSA's `fclass` (LoongArch's `fclass` alike):
    /// one bit of ten, a signaling NaN (bit 0), a quiet NaN, then negative
    /// infinity, normal, subnormal and zero (bit 5), then positive infinity,
    /// normal, subnormal and zero (bit 9).
    Mips,
    /// x87's `fxam`: the condition codes `C3`, `C2` and `C0` as a flags
    /// word, `C3` the zero flag, `C2` the parity flag and `C0` the carry: a
    /// NaN `C0`, a normal `C2`, an infinity `C2` and `C0`, a zero `C3`, a
    /// denormal (a pseudo-denormal too) `C3` and `C2`, and an encoding the
    /// extended format does not support (an unnormal, a pseudo-infinity, a
    /// pseudo-NaN) none. An empty register is a state of the register, not a
    /// class of its value.
    X87,
}

impl FloatClassLayout {
    /// The layout's name, its architecture's: `riscv`, `mips` or `x87`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RiscV => "riscv",
            Self::Mips => "mips",
            Self::X87 => "x87",
        }
    }

    /// The class `value` has in this layout: the bit of its class, or for
    /// x87 the flags word of its condition codes.
    #[must_use]
    pub fn class(self, value: FloatBits) -> u64 {
        // The class bit in RISC-V's order: negative infinity, negative normal,
        // subnormal and zero, positive zero, subnormal, normal and infinity,
        // signaling and quiet NaN.
        let riscv = value.classify().trailing_zeros();
        match self {
            Self::RiscV => 1 << riscv,
            Self::Mips => {
                let bit = match riscv {
                    8 => 0,
                    9 => 1,
                    0 => 2,
                    1 => 3,
                    2 => 4,
                    3 => 5,
                    7 => 6,
                    6 => 7,
                    5 => 8,
                    _ => 9,
                };
                1 << bit
            }
            Self::X87 => {
                let (c0, c2, c3) = (
                    u64::from(FlagsMask::CARRY.bits()),
                    u64::from(FlagsMask::PARITY.bits()),
                    u64::from(FlagsMask::ZERO.bits()),
                );
                if value.format() == FloatFormat::X87Extended {
                    let bits = value.bits();
                    let exponent = bits >> 64 & 0x7fff;
                    let integer = bits >> 63 & 1;
                    if exponent != 0 && integer == 0 {
                        return 0;
                    }
                    if exponent == 0 && integer == 1 {
                        return c3 | c2;
                    }
                }
                match riscv {
                    8 | 9 => c0,
                    0 | 7 => c2 | c0,
                    3 | 4 => c3,
                    2 | 5 => c3 | c2,
                    _ => c2,
                }
            }
        }
    }
}

/// A float constant of any [`FloatFormat`] by its bits.
///
/// Equality and hashing are bitwise, as [`ConstValue`](crate::ir::ConstValue)'s
/// are: a constant is its bit pattern, so a NaN equals itself and `-0.0`
/// differs from `0.0`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FloatBits {
    /// The format.
    format: FloatFormat,
    /// The bits, low half first.
    bits: [u64; 2],
}

/// Applies `$body` with `$float` bound to the `rustc_apfloat` type of
/// `$format`, one of the formats [`FloatBits`] holds.
macro_rules! with_format {
    ($format:expr, $float:ident => $body:expr) => {
        match $format {
            FloatFormat::Half => {
                type $float = Half;
                $body
            }
            FloatFormat::BFloat16 => {
                type $float = BFloat;
                $body
            }
            FloatFormat::X87Extended => {
                type $float = X87DoubleExtended;
                $body
            }
            FloatFormat::Quad => {
                type $float = Quad;
                $body
            }
            FloatFormat::Single => {
                type $float = Single;
                $body
            }
            FloatFormat::Double => {
                type $float = Double;
                $body
            }
        }
    };
}

impl FloatBits {
    /// The constant of `format` whose bits are `bits`. `None` for bits wider
    /// than the format.
    #[must_use]
    pub const fn new(format: FloatFormat, bits: u128) -> Option<Self> {
        if !format.holds(bits) {
            return None;
        }
        Some(Self {
            format,
            bits: [bits as u64, (bits >> 64) as u64],
        })
    }

    /// The constant's format.
    #[must_use]
    pub const fn format(self) -> FloatFormat {
        self.format
    }

    /// The constant's bits.
    #[must_use]
    pub const fn bits(self) -> u128 {
        (self.bits[0] as u128) | ((self.bits[1] as u128) << 64)
    }

    /// Whether the constant is a NaN.
    #[must_use]
    pub fn is_nan(self) -> bool {
        with_format!(self.format, F => F::from_bits(self.bits()).is_nan())
    }

    /// The constant with its quiet bit set, the top bit of a NaN's fraction
    /// (below x87's explicit integer bit): a signalling NaN made quiet, any
    /// other NaN as it is.
    #[must_use]
    pub fn quieted(self) -> Self {
        Self::new(self.format, self.bits() | self.format.quiet_bit()).unwrap_or(self)
    }

    /// This constant read in legacy MIPS's NaN encoding, where a NaN is
    /// quiet with its quiet bit clear, as the IEEE 754-2008 constant of the
    /// same kind: a legacy quiet NaN with the bit set, a legacy signalling
    /// NaN with it clear (its lowest fraction bit set where nothing else of
    /// the fraction would keep it a NaN). Any other value is itself. What a
    /// NaN operand raises depends on its kind alone, which the IEEE
    /// arithmetic reads from it this way.
    #[must_use]
    pub fn from_legacy_nan(self) -> Self {
        if !self.is_nan() {
            return self;
        }
        let quiet_bit = self.format.quiet_bit();
        if self.bits() & quiet_bit == 0 {
            return self.quieted();
        }
        let cleared = self.bits() & !quiet_bit;
        let signalling = Self::new(self.format, cleared).filter(|value| value.is_nan());
        signalling
            .or_else(|| Self::new(self.format, cleared | 1))
            .unwrap_or(self)
    }

    /// The format's largest finite value, positive.
    #[must_use]
    pub fn largest(format: FloatFormat) -> Option<Self> {
        let fraction = format.fraction_bits();
        let exponent = BitWidth::new(format.exponent_bits()).checked_mask_u128()?;
        let field = BitWidth::new(fraction).checked_mask_u128()?;
        Self::new(
            format,
            exponent.checked_sub(1)?.checked_shl(fraction)? | field,
        )
    }

    /// x86's indefinite NaN of `format`, the quiet NaN with the sign set and
    /// an otherwise empty payload that an invalid operation gives.
    #[must_use]
    pub fn indefinite(format: FloatFormat) -> Option<Self> {
        let sign = 1u128.checked_shl(format.bits().checked_sub(1)?)?;
        Self::new(format, format.canonical_nan_bits() | sign)
    }

    /// The biased exponent field and the fraction field of the bits.
    fn fields(self) -> Option<(u128, u128)> {
        let fraction = self.format.fraction_bits();
        let exponent = BitWidth::new(self.format.exponent_bits()).checked_mask_u128()?;
        let field = BitWidth::new(fraction).checked_mask_u128()?;
        Some(((self.bits() >> fraction) & exponent, self.bits() & field))
    }

    /// The constant rounded to an integral multiple of `2^-scale` by
    /// `rounding`, x86's `vrndscale`: `2^-scale * round(2^scale * x)`, the
    /// product exact whatever the exponent range. A NaN is quieted, an
    /// infinity and a zero are kept; `None` for a subnormal, which a
    /// denormals-are-zero mode decides.
    #[must_use]
    pub fn round_scaled(self, scale: i32, rounding: FloatRounding) -> Option<Self> {
        if self.is_nan() {
            return Some(self.quieted());
        }
        if !self.is_finite() || self.bits() & !self.sign_bit() == 0 {
            return Some(self);
        }
        if self.is_subnormal() {
            return None;
        }
        let (exponent, _) = self.fields()?;
        let bias = (1i64 << self.format.exponent_bits().checked_sub(1)?).checked_sub(1)?;
        // Integral at the scale already: every bit of the fraction counts
        // whole multiples of `2^-scale`.
        let unbiased = i64::try_from(exponent).ok()?.checked_sub(bias)?;
        if unbiased.checked_add(i64::from(scale))? >= i64::from(self.format.fraction_bits()) {
            return Some(self);
        }
        self.scaled_by_power(scale)?
            .integral(rounding)?
            .scaled_by_power(scale.checked_neg()?)
    }

    /// The constant's significand normalized into an interval, with a sign
    /// the controls name, x86's `vgetmant` (Intel SDM `getmant_fp`):
    /// `interval` `0` for `[1, 2)`, `1` for `[1/2, 2)` (by the exponent's
    /// parity), `2` for `[1/2, 1)`, `3` for `[3/4, 3/2)` (by the fraction's
    /// top bit); `sign_control` bit 0 makes the result positive, bit 1 makes a
    /// negative source the indefinite NaN. A NaN is quieted, a zero and an
    /// infinity are one (its sign by bit 0 for a negative one); `None` for a
    /// subnormal, which a denormals-are-zero mode decides.
    #[must_use]
    pub fn mantissa(self, interval: u8, sign_control: u8) -> Option<Self> {
        if self.is_nan() {
            return Some(self.quieted());
        }
        let format = self.format;
        let sign = self.sign_bit();
        let negative = self.bits() & sign != 0;
        let zero = self.bits() & !sign == 0;
        let one = format.from_integer(1);
        let signed_one = if sign_control & 1 != 0 {
            one
        } else {
            one | sign
        };
        if !negative && (zero || !self.is_finite()) {
            return Self::new(format, one);
        }
        if negative && (zero || (!self.is_finite() && sign_control & 2 == 0)) {
            return Self::new(format, signed_one);
        }
        if negative && sign_control & 2 != 0 {
            return Self::indefinite(format);
        }
        if self.is_subnormal() {
            return None;
        }
        let (exponent, fraction) = self.fields()?;
        let bias = BitWidth::new(format.exponent_bits().checked_sub(1)?).mask_u128();
        // The unbiased exponent's parity: its low bit, which a difference
        // shares with an exclusive or.
        let odd = (exponent ^ bias) & 1 == 1;
        let top = fraction >> format.fraction_bits().checked_sub(1)? & 1 == 1;
        let below = bias.checked_sub(1)?;
        let exponent = match interval & 3 {
            0 => bias,
            1 if odd => below,
            1 => bias,
            2 => below,
            _ if top => below,
            _ => bias,
        };
        let sign = if sign_control & 1 != 0 {
            0
        } else {
            self.bits() & sign
        };
        Self::new(
            format,
            sign | exponent.checked_shl(format.fraction_bits())? | fraction,
        )
    }

    /// The fraction field's bits, the quiet bit and the payload below it,
    /// which x87's arithmetic orders two NaNs by.
    #[must_use]
    pub fn significand(self) -> u128 {
        self.bits() & (self.format.quiet_bit() << 1).wrapping_sub(1)
    }

    /// Whether the constant is a number: neither a NaN nor an infinity.
    #[must_use]
    pub fn is_finite(self) -> bool {
        with_format!(self.format, F => F::from_bits(self.bits()).is_finite())
    }

    /// Whether the constant is subnormal: nonzero, with the least exponent
    /// and no implicit integer bit, the value a flush-to-zero or
    /// denormals-are-zero mode reads or writes as zero.
    #[must_use]
    pub fn is_subnormal(self) -> bool {
        with_format!(self.format, F => F::from_bits(self.bits()).is_denormal())
    }

    /// Whether the constant is a signalling NaN, read as IEEE 754-2008 encodes
    /// one: the fraction's top bit clear.
    #[must_use]
    pub fn is_signaling(self) -> bool {
        with_format!(self.format, F => F::from_bits(self.bits()).is_signaling())
    }

    /// The constant with its sign flipped: a bit operation, NaN included.
    #[must_use]
    pub fn negate(self) -> Self {
        self.with_sign(!self.is_negative())
    }

    /// The constant with its sign cleared: a bit operation, NaN included.
    #[must_use]
    pub fn abs(self) -> Self {
        self.with_sign(false)
    }

    /// Whether the sign bit is set: a negative number, negative zero or a
    /// NaN of negative sign.
    #[must_use]
    pub fn is_negative(self) -> bool {
        self.bits() & self.sign_bit() != 0
    }

    /// The format's sign bit, its highest.
    const fn sign_bit(self) -> u128 {
        1u128 << self.format.bits().saturating_sub(1)
    }

    /// The constant with its sign bit set to `negative`.
    fn with_sign(self, negative: bool) -> Self {
        let sign = self.sign_bit();
        let bits = if negative {
            self.bits() | sign
        } else {
            self.bits() & !sign
        };
        Self {
            format: self.format,
            bits: [bits as u64, (bits >> 64) as u64],
        }
    }

    /// The sum of two constants of one format, rounded by `rounding`, or
    /// by the program's mode where that is `None` ([`FloatRounding::decide`]).
    #[must_use]
    pub fn checked_add(self, rhs: Self, rounding: Option<FloatRounding>) -> Option<Self> {
        self.binary(rhs, Binary::Add, rounding)
    }

    /// The difference of two constants of one format, rounded by `rounding`,
    /// or by the program's mode where that is `None` ([`FloatRounding::decide`]).
    #[must_use]
    pub fn checked_sub(self, rhs: Self, rounding: Option<FloatRounding>) -> Option<Self> {
        self.binary(rhs, Binary::Sub, rounding)
    }

    /// The product of two constants of one format, rounded by `rounding`, or
    /// by the program's mode where that is `None` ([`FloatRounding::decide`]).
    #[must_use]
    pub fn checked_mul(self, rhs: Self, rounding: Option<FloatRounding>) -> Option<Self> {
        self.binary(rhs, Binary::Mul, rounding)
    }

    /// The quotient of two constants of one format, rounded by `rounding`, or
    /// by the program's mode where that is `None` ([`FloatRounding::decide`]).
    #[must_use]
    pub fn checked_div(self, rhs: Self, rounding: Option<FloatRounding>) -> Option<Self> {
        self.binary(rhs, Binary::Div, rounding)
    }

    /// `self * rhs + addend` of three constants of one format, rounded once by
    /// `rounding`, or by the program's mode where that is `None`
    /// ([`FloatRounding::decide`]); `None` when their formats differ or an
    /// operand or the result is a NaN.
    #[must_use]
    pub fn checked_mul_add(
        self,
        rhs: Self,
        addend: Self,
        rounding: Option<FloatRounding>,
    ) -> Option<Self> {
        if self.format != rhs.format || self.format != addend.format {
            return None;
        }
        FloatRounding::decide(rounding, |rounding| {
            let round = rounding.arithmetic()?;
            let bits = with_format!(self.format, F => {
                let (a, b, c) = (
                    F::from_bits(self.bits()),
                    F::from_bits(rhs.bits()),
                    F::from_bits(addend.bits()),
                );
                if a.is_nan() || b.is_nan() || c.is_nan() {
                    return None;
                }
                let result = a.mul_add_r(b, c, round).value;
                if result.is_nan() {
                    return None;
                }
                result.to_bits()
            });
            Self::new(self.format, bits)
        })
    }

    /// The truncating remainder of two constants of one format, C's `fmod`:
    /// the sign of the dividend, exact in every mode.
    #[must_use]
    pub fn checked_rem(self, rhs: Self) -> Option<Self> {
        self.binary(rhs, Binary::Rem, Some(FloatRounding::NearestEven))
    }

    /// The constant times two to the power of `by` rounded toward zero, x87's
    /// `fscale`: `None` for a NaN or an infinity of either, a power too large
    /// to name, or a result the scaling rounds (one that overflows, or loses
    /// bits as a subnormal), where the instruction's own result is not this
    /// exact product.
    #[must_use]
    pub fn scaled(self, by: Self) -> Option<Self> {
        if self.format != by.format {
            return None;
        }
        self.scaled_by_power(i32::try_from(by.trunc()?.to_integer()?).ok()?)
    }

    /// The constant times two to the power `power`, C's `ldexp` (SVE
    /// `fscale`): `None` for a NaN or an infinity, and for a result the
    /// scaling rounds (one that overflows, or loses bits as a subnormal),
    /// which the rounding mode and the flush modes decide.
    #[must_use]
    pub fn scaled_by_power(self, power: i32) -> Option<Self> {
        let bits = with_format!(self.format, F => {
            let value = F::from_bits(self.bits());
            if !value.is_finite() {
                return None;
            }
            let scaled = value.scalbn(power);
            if !scaled.is_finite() || scaled.scalbn(power.checked_neg()?).to_bits() != value.to_bits()
            {
                return None;
            }
            scaled.to_bits()
        });
        Self::new(self.format, bits)
    }

    /// The constant's unbiased exponent and its significand scaled into `[1,
    /// 2)` with its sign, x87's `fxtract`: `(exponent, significand)`, each of
    /// the constant's format, so that `significand * 2^exponent` is the
    /// constant. `None` for a NaN, an infinity and a zero, whose extraction
    /// signals.
    #[must_use]
    pub fn extracted(self) -> Option<(Self, Self)> {
        let (exponent, significand) = with_format!(self.format, F => {
            let value = F::from_bits(self.bits());
            if !value.is_finite() || value.is_zero() {
                return None;
            }
            let exponent = value.ilogb();
            let significand = value.scalbn(exponent.checked_neg()?);
            (exponent, significand.to_bits())
        });
        let exponent = Self::new(self.format, self.format.from_integer(i128::from(exponent)))?;
        Some((exponent, Self::new(self.format, significand)?))
    }

    /// The remainder of the constant divided by `divisor`, and the three low
    /// bits of the quotient's magnitude, x87's complete partial remainder:
    /// the quotient rounded toward zero (`fprem`), or to nearest with ties to
    /// even (`fprem1`, the IEEE remainder). Each is exact.
    ///
    /// `None` for a NaN, an infinity of the dividend, a zero divisor, and a
    /// dividend 64 or more binary orders above the divisor, whose reduction
    /// the instruction leaves incomplete.
    #[must_use]
    pub fn partial_remainder(self, divisor: Self, nearest: bool) -> Option<(Self, u8)> {
        if self.format != divisor.format {
            return None;
        }
        let (remainder, quotient) = with_format!(self.format, F => {
            let (x, y) = (F::from_bits(self.bits()), F::from_bits(divisor.bits()));
            if !x.is_finite() || y.is_nan() || y.is_zero() {
                return None;
            }
            let remainder = if nearest { x.ieee_rem(y) } else { x.c_fmod(y) }.value;
            if remainder.is_nan() {
                return None;
            }
            if x.is_zero() || y.is_infinite() {
                (remainder.to_bits(), 0u8)
            } else {
                // The magnitudes as integer significands of the format's
                // precision at their binary orders, `x = mx * 2^ex`.
                let significand = |value: F| -> Option<(u128, i32)> {
                    let order = value.ilogb();
                    let lead = i32::try_from(F::PRECISION.checked_sub(1)?).ok()?;
                    let scaled = value.abs().scalbn(lead.checked_sub(order)?);
                    let mut exact = false;
                    let integer = scaled.to_u128_r(128, Round::TowardZero, &mut exact).value;
                    exact.then_some((integer, order))
                };
                let ((mx, ex), (my, ey)) = (significand(x)?, significand(y)?);
                let difference = ex.checked_sub(ey)?;
                if difference >= 64 {
                    return None;
                }
                let quotient = if difference < -1 {
                    0
                } else {
                    let (numerator, denominator) = if difference >= 0 {
                        (mx.checked_shl(u32::try_from(difference).ok()?)?, my)
                    } else {
                        (mx, my.checked_shl(difference.unsigned_abs())?)
                    };
                    let truncated = numerator.checked_div(denominator)?;
                    let left = numerator.checked_rem(denominator)?;
                    let twice = left.checked_mul(2)?;
                    let up = nearest
                        && (twice > denominator || (twice == denominator && truncated & 1 == 1));
                    truncated.checked_add(u128::from(up))?
                };
                (remainder.to_bits(), u8::try_from(quotient & 7).ok()?)
            }
        });
        Some((Self::new(self.format, remainder)?, quotient))
    }

    /// The constant rounded to an integral value toward zero.
    #[must_use]
    pub fn trunc(self) -> Option<Self> {
        self.round_to_integral(Round::TowardZero)
    }

    /// The constant rounded to an integral value toward negative infinity.
    #[must_use]
    pub fn floor(self) -> Option<Self> {
        self.round_to_integral(Round::TowardNegative)
    }

    /// The constant rounded to an integral value toward positive infinity.
    #[must_use]
    pub fn ceil(self) -> Option<Self> {
        self.round_to_integral(Round::TowardPositive)
    }

    /// The constant rounded to the nearest integral value, ties to even.
    #[must_use]
    pub fn round_ties_even(self) -> Option<Self> {
        self.round_to_integral(Round::NearestTiesToEven)
    }

    /// The constant rounded to the nearest integral value, ties away from
    /// zero.
    #[must_use]
    pub fn round_ties_away(self) -> Option<Self> {
        self.round_to_integral(Round::NearestTiesToAway)
    }

    /// The constant rounded to an integral value by `rounding`: `None` for a
    /// NaN and for odd rounding, which only a conversion states.
    #[must_use]
    pub fn integral(self, rounding: FloatRounding) -> Option<Self> {
        self.round_to_integral(rounding.arithmetic()?)
    }

    /// The constant rounded to an integral value by `rounding` that a signed
    /// integer of `width` bits holds, or that integer's minimum where the
    /// rounded value is outside its range, an infinity or a NaN (AArch64
    /// `frint32x`, `frint64z`). `None` for odd rounding and a width past 128.
    #[must_use]
    pub fn integral_within(self, rounding: FloatRounding, width: u32) -> Option<Self> {
        rounding.arithmetic()?;
        let shift = width.checked_sub(1).filter(|shift| *shift < 128)?;
        let minimum = i128::MIN.checked_shr(127u32.checked_sub(shift)?)?;
        let maximum = !minimum;
        let held = self.integral(rounding).filter(|rounded| {
            rounded
                .to_integer()
                .is_some_and(|value| (minimum..=maximum).contains(&value))
        });
        held.or_else(|| Self::new(self.format, self.format.from_integer(minimum)))
    }

    /// The constant rounded to an integral value of its format in the
    /// direction `round`: `None` for a NaN.
    fn round_to_integral(self, round: Round) -> Option<Self> {
        let bits = with_format!(self.format, F => {
            let value = F::from_bits(self.bits());
            if value.is_nan() {
                return None;
            }
            value.round_to_integral(round).value.to_bits()
        });
        Self::new(self.format, bits)
    }

    /// The constant in `to` when that holds it exactly, converting back to
    /// the same bits (`2.0` as a binary64 is a binary16 too, `0.1` is not);
    /// `None` otherwise, and for a NaN, whose payload a conversion may change.
    #[must_use]
    pub fn exactly(self, to: FloatFormat) -> Option<Self> {
        let converted = self.format.convert(self.bits(), to)?;
        if to.convert(converted, self.format)? != self.bits() {
            return None;
        }
        Self::new(to, converted)
    }

    /// The value in decimal with no suffix, in digits enough to read back as
    /// the same bits in its format: Rust's shortest for a single and a double,
    /// `rustc_apfloat`'s for the rest; `NaN`, `inf` or `-inf` for a value
    /// that is not a number.
    #[must_use]
    pub fn decimal(self) -> String {
        match self.format {
            FloatFormat::Single => f32::from_bits(self.bits[0] as u32).to_string(),
            FloatFormat::Double => f64::from_bits(self.bits[0]).to_string(),
            _ => with_format!(self.format, F => F::from_bits(self.bits()).to_string()),
        }
    }

    /// The constant converted to an integer, rounded toward zero; `None` for
    /// a NaN, an infinity, or a value past what 128 bits hold.
    #[must_use]
    pub fn to_integer(self) -> Option<i128> {
        with_format!(self.format, F => {
            let value = F::from_bits(self.bits());
            if !value.is_finite() {
                return None;
            }
            let mut exact = false;
            let converted = value.to_i128_r(128, Round::TowardZero, &mut exact);
            (!converted.status.contains(Status::INVALID_OP)).then_some(converted.value)
        })
    }

    /// The constant converted to an integer of `width` bits modulo two to that
    /// power, as JavaScript's `ToInt32` converts: rounded toward zero, the low
    /// `width` bits of that unbounded integer kept (two's complement for a
    /// negative one), and zero for a NaN or an infinity.
    ///
    /// Every finite value has one, however large: a value past the width is a
    /// multiple of its lowest significand bit, and the bits below that are
    /// zero.
    ///
    /// # Returns
    ///
    /// The integer's bits, or `None` for a width other than 8, 16, 32 or 64.
    #[must_use]
    pub fn to_integer_modular(self, width: u32) -> Option<u128> {
        if !matches!(width, 8 | 16 | 32 | 64) {
            return None;
        }
        let modulus = i32::try_from(width).ok()?;
        let mask = BitWidth::new(width).checked_mask_u128()?;
        with_format!(self.format, F => {
            let value = F::from_bits(self.bits());
            if !value.is_finite() {
                return Some(0);
            }
            let whole = value.round_to_integral(Round::TowardZero).value;
            // The multiple of two to the width toward zero from `whole`, and
            // `whole` less it: each step exact. Scaling by a power of two is
            // exact for any quotient of at least one, a smaller one truncating
            // to zero however it rounded, and the difference is bits `whole`
            // holds, which its format holds too.
            let multiple = whole
                .scalbn(modulus.checked_neg()?)
                .round_to_integral(Round::TowardZero)
                .value
                .scalbn(modulus);
            let rest = whole.sub_r(multiple, Round::TowardZero).value;
            let mut exact = false;
            let rest = rest.to_i128_r(128, Round::TowardZero, &mut exact).value;
            Some(rest.cast_unsigned() & mask)
        })
    }

    /// Whether `operation` of this and `rhs` is invalid without either being
    /// a NaN: infinity minus infinity, zero times infinity, zero by zero,
    /// infinity by infinity. Its result is then the target's
    /// [`DefaultNan`], whatever the rounding.
    #[must_use]
    pub fn invalid_with(self, rhs: Self, operation: FloatBinary) -> bool {
        !self.is_nan()
            && !rhs.is_nan()
            && self
                .binary_rounded(rhs, operation, FloatRounding::NearestEven, true)
                .is_some_and(|result| result.value.is_nan())
    }

    /// Whether this times `rhs` plus `addend`, fused, is invalid without a
    /// NaN operand: a zero times an infinity, or an infinite product plus the
    /// infinity of the other sign.
    #[must_use]
    pub fn invalid_mul_add(self, rhs: Self, addend: Self) -> bool {
        !self.is_nan()
            && !rhs.is_nan()
            && !addend.is_nan()
            && self
                .mul_add_rounded(rhs, addend, FloatRounding::NearestEven, true)
                .is_some_and(|result| result.value.is_nan())
    }

    /// Whether the square root of this is invalid: a number below zero.
    #[must_use]
    pub fn invalid_sqrt(self) -> bool {
        !self.is_nan()
            && self
                .sqrt_rounded(FloatRounding::NearestEven, true)
                .is_some_and(|result| result.value.is_nan())
    }

    /// How two constants of one format order, `None` when either is a NaN or
    /// their formats differ.
    #[must_use]
    pub fn compare(self, rhs: Self) -> Option<Ordering> {
        if self.format != rhs.format {
            return None;
        }
        with_format!(self.format, F => {
            F::from_bits(self.bits()).partial_cmp(&F::from_bits(rhs.bits()))
        })
    }

    /// The sum of `values` that every order of adding them gives, each step
    /// rounded by `rounding`, or by the program's mode where that is `None`
    /// ([`FloatRounding::decide`]); `None` where two orders could round
    /// differently, for a NaN, and for values of more than one format.
    ///
    /// Two values add the same either way round. More are added a subset at a
    /// time, each partial sum the sum of some of them, so every order gives
    /// one sum when every subset sum is exact. It is when the values, each an
    /// integer times a power of two at the least exponent among them, have
    /// magnitudes that sum to less than the format's precision holds, and to
    /// a finite value: every subset sum is then such an integer times that
    /// power, which the format holds. An infinity among more than two values
    /// is refused, since an order that overflows before reaching it meets it
    /// with the infinity of the other sign.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::ir::float::{FloatBits, FloatRounding};
    ///
    /// let exact = [1.5f32, -0.25, 4.0].map(FloatBits::from);
    /// assert_eq!(FloatBits::sum_in_any_order(&exact, None), Some(FloatBits::from(5.25f32)));
    /// // `(2^24 + 1) - 2^24` rounds the 1 away; `(2^24 - 2^24) + 1` keeps it.
    /// let rounding = [16_777_216.0f32, 1.0, -16_777_216.0].map(FloatBits::from);
    /// let nearest = Some(FloatRounding::NearestEven);
    /// assert_eq!(FloatBits::sum_in_any_order(&rounding, nearest), None);
    /// ```
    #[must_use]
    pub fn sum_in_any_order(values: &[Self], rounding: Option<FloatRounding>) -> Option<Self> {
        let (&first, rest) = values.split_first()?;
        if values.len() > 2 && !Self::subset_sums_are_exact(values)? {
            return None;
        }
        FloatRounding::decide(rounding, |rounding| {
            rest.iter()
                .try_fold(first, |sum, value| sum.checked_add(*value, Some(rounding)))
        })
    }

    /// Whether every sum of a subset of `values` is exact in their format, by
    /// the bound [`Self::sum_in_any_order`] states: `false` for a value that
    /// is not finite, `None` for values of more than one format.
    fn subset_sums_are_exact(values: &[Self]) -> Option<bool> {
        let format = values.first()?.format;
        if values.iter().any(|value| value.format != format) {
            return None;
        }
        with_format!(format, F => {
            let precision = i32::try_from(F::PRECISION).ok()?;
            let mut parts = Vec::with_capacity(values.len());
            for value in values {
                let value = F::from_bits(value.bits());
                if !value.is_finite() {
                    return Some(false);
                }
                if value.is_zero() {
                    continue;
                }
                // `|value| = significand * 2^exponent`, the significand an
                // integer below `2^precision`: scaling by a power of two that
                // leaves it so is exact. Its trailing zeros go to the
                // exponent, so the least exponent is the value's own lowest
                // set bit.
                let exponent = value.ilogb().checked_sub(precision.checked_sub(1)?)?;
                let significand = value.abs().scalbn(exponent.checked_neg()?).to_u128(128).value;
                let zeros = significand.trailing_zeros();
                parts.push((
                    significand >> zeros,
                    exponent.checked_add(i32::try_from(zeros).ok()?)?,
                ));
            }
            let Some(least) = parts.iter().map(|(_, exponent)| *exponent).min() else {
                return Some(true);
            };
            let mut total = 0u128;
            for (significand, exponent) in parts {
                let shift = u32::try_from(exponent.checked_sub(least)?).ok()?;
                if shift >= significand.leading_zeros() {
                    return Some(false);
                }
                let Some(sum) = total.checked_add(significand << shift) else {
                    return Some(false);
                };
                total = sum;
            }
            if total >= 1u128.checked_shl(u32::try_from(precision).ok()?)? {
                return Some(false);
            }
            // No subset sums to more than the magnitudes do, so where their
            // total is finite none overflows.
            Some(F::from_u128(total).value.scalbn(least).is_finite())
        })
    }

    /// Injects a source sign, optionally inverted or XORed with this value's sign.
    ///
    /// Every magnitude and NaN payload bit is retained. Neither input raises
    /// floating-point exceptions. The formats must agree.
    #[must_use]
    pub fn inject_sign(self, rhs: Self, negate: bool, xor: bool) -> Option<Self> {
        if self.format != rhs.format {
            return None;
        }
        Some(self.with_sign(rhs.is_negative() ^ negate ^ (xor && self.is_negative())))
    }

    /// Computes a seven-bit midpoint-table reciprocal or reciprocal-square-root estimate.
    ///
    /// The reciprocal indexes its normalized fraction's seven high bits; the
    /// square-root form indexes exponent parity and six high fraction bits.
    /// The table rounds the bucket midpoint's normalized reciprocal to seven
    /// fraction bits. Other result fraction bits are zero, with subnormal
    /// reciprocal outputs truncated. Only reciprocal overflow depends on the
    /// rounding rule and raises overflow plus inexact. Zeros raise divide by
    /// zero, signaling NaNs and negative square roots raise invalid.
    ///
    /// Supports implicit-significand binary formats. NaN results are refused
    /// unless canonical NaNs are requested. Odd rounding and x87 are refused.
    #[must_use]
    pub fn estimate7(
        self,
        square_root: bool,
        rounding: FloatRounding,
        canonical_nan: bool,
    ) -> Option<FloatResult> {
        rounding.arithmetic()?;
        let (fraction_bits, exponent_bits, bias) = match self.format {
            FloatFormat::Half => (10u32, 5u32, 15i32),
            FloatFormat::BFloat16 => (7, 8, 127),
            FloatFormat::Single => (23, 8, 127),
            FloatFormat::Double => (52, 11, 1023),
            FloatFormat::Quad => (112, 15, 16383),
            FloatFormat::X87Extended => return None,
        };
        let leading = 1u128.checked_shl(fraction_bits)?;
        let exponent_mask = BitWidth::new(exponent_bits).mask_u128();
        let infinity = exponent_mask.checked_shl(fraction_bits)?;
        let fraction = self.bits() & leading.checked_sub(1)?;
        let encoded_exponent = self.bits().checked_shr(fraction_bits)? & exponent_mask;
        let sign = self.bits() & self.sign_bit();
        let invalid = self.classify() & (1 << 8) != 0;
        if self.is_nan() || square_root && sign != 0 && (encoded_exponent != 0 || fraction != 0) {
            return rounded_result(
                self.format,
                self.format.canonical_nan_bits(),
                if invalid || !self.is_nan() {
                    Status::INVALID_OP
                } else {
                    Status::OK
                },
                canonical_nan,
            );
        }
        if encoded_exponent == 0 && fraction == 0 {
            return rounded_result(
                self.format,
                sign | infinity,
                Status::DIV_BY_ZERO,
                canonical_nan,
            );
        }
        if encoded_exponent == exponent_mask {
            return rounded_result(self.format, sign, Status::OK, canonical_nan);
        }
        let mut exponent = i32::try_from(encoded_exponent).ok()?;
        let mut mantissa = leading | fraction;
        if encoded_exponent == 0 {
            exponent = 1;
            mantissa = fraction;
            while mantissa < leading {
                mantissa = mantissa.checked_shl(1)?;
                exponent = exponent.checked_sub(1)?;
            }
        }
        let index_bits = if square_root { 6 } else { 7 };
        let index = (mantissa & leading.checked_sub(1)?)
            .checked_shr(fraction_bits.checked_sub(index_bits)?)?;
        let table = estimate7_significand(index, square_root, exponent & 1 != 0)?;
        let output_exponent = if square_root {
            bias.checked_mul(3)?
                .checked_sub(1)?
                .checked_sub(exponent)?
                .div_euclid(2)
        } else {
            bias.checked_mul(2)?.checked_sub(1)?.checked_sub(exponent)?
        };
        if output_exponent > bias.checked_mul(2)? {
            let finite = matches!(rounding, FloatRounding::TowardZero)
                || sign == 0 && matches!(rounding, FloatRounding::Down)
                || sign != 0 && matches!(rounding, FloatRounding::Up);
            return rounded_result(
                self.format,
                sign | if finite {
                    infinity.checked_sub(1)?
                } else {
                    infinity
                },
                Status::OVERFLOW | Status::INEXACT,
                canonical_nan,
            );
        }
        let fraction = table.checked_shl(fraction_bits.checked_sub(7)?)?;
        let bits = if output_exponent <= 0 {
            (leading | fraction)
                .checked_shr(u32::try_from(1i32.checked_sub(output_exponent)?).ok()?)?
        } else {
            u128::try_from(output_exponent)
                .ok()?
                .checked_shl(fraction_bits)?
                | fraction
        };
        rounded_result(self.format, sign | bits, Status::OK, canonical_nan)
    }

    /// Reduces selected values into this seed in source order with IEEE flags.
    ///
    /// Accepts addition, minimum-number and maximum-number. Narrow inputs are
    /// promoted exactly to the seed's format before each operation. The caller
    /// supplies only selected inputs; an empty selection copies the seed bits
    /// without quieting NaNs, canonicalization or exceptions. This is an ordered
    /// reduction and one permitted deterministic tree for an unordered reduction;
    /// it does not assert that every permitted tree produces the same bits.
    /// Returns `None` for unsupported relations, odd rounding, wider inputs or
    /// a format conversion that is not exact for finite values.
    #[must_use]
    pub fn reduce_rounded(
        self,
        operation: FloatBinary,
        values: &[Self],
        rounding: FloatRounding,
        canonical_nan: bool,
    ) -> Option<FloatResult> {
        rounding.arithmetic()?;
        if !matches!(
            operation,
            FloatBinary::Add | FloatBinary::MinimumNumber | FloatBinary::MaximumNumber
        ) {
            return None;
        }
        let mut result = FloatResult {
            value: self,
            exceptions: FloatExceptions::NONE,
        };
        for value in values {
            let mut value = *value;
            if value.format != self.format {
                if value.format.bits() >= self.format.bits() {
                    return None;
                }
                let converted = value.convert_rounded(self.format, rounding, canonical_nan)?;
                if converted.exceptions.bits() & !FloatExceptions::INVALID.bits() != 0 {
                    return None;
                }
                result.exceptions = result.exceptions.union(converted.exceptions);
                value = converted.value;
            }
            let next = result
                .value
                .binary_rounded(value, operation, rounding, canonical_nan)?;
            result.value = next.value;
            result.exceptions = result.exceptions.union(next.exceptions);
        }
        Some(result)
    }

    /// Evaluates binary arithmetic with explicit rounding and IEEE exceptions.
    ///
    /// NaN results are refused unless positive canonical NaNs are requested.
    /// MinimumNumber and MaximumNumber ignore one NaN, produce a NaN for two,
    /// and raise invalid for a signaling NaN even when returning the other input.
    #[must_use]
    pub fn binary_rounded(
        self,
        rhs: Self,
        operation: FloatBinary,
        rounding: FloatRounding,
        canonical_nan: bool,
    ) -> Option<FloatResult> {
        if self.format != rhs.format {
            return None;
        }
        let round = rounding.arithmetic()?;
        with_format!(self.format, F => {
            let (left, right) = (F::from_bits(self.bits()), F::from_bits(rhs.bits()));
            let result = match operation {
                FloatBinary::Add => left.add_r(right, round),
                FloatBinary::Subtract => left.sub_r(right, round),
                FloatBinary::Multiply => left.mul_r(right, round),
                FloatBinary::Divide => left.div_r(right, round),
                FloatBinary::MinimumNumber | FloatBinary::MaximumNumber => {
                    let status = if left.is_signaling() || right.is_signaling() { Status::INVALID_OP } else { Status::OK };
                    let minimum = operation == FloatBinary::MinimumNumber;
                    let value = if left.is_nan() && right.is_nan() { F::NAN }
                        else if left.is_nan() { right } else if right.is_nan() { left }
                        else if left.is_zero() && right.is_zero() {
                            if minimum && left.is_negative() || !minimum && !left.is_negative() { left } else { right }
                        } else if left.partial_cmp(&right).is_some_and(|order| if minimum { order.is_lt() } else { order.is_gt() }) { left } else { right };
                    status.and(value)
                }
            };
            let result = binary_overflow_status(left, right, operation, round, result);
            rounded_result(self.format, result.value.to_bits(), result.status, canonical_nan)
        })
    }

    /// Evaluates a fused product and addend with one rounding and explicit exceptions.
    ///
    /// Infinite times zero raises invalid even with a quiet NaN addend.
    /// NaN results are refused unless positive canonical NaNs are requested.
    #[must_use]
    pub fn mul_add_rounded(
        self,
        rhs: Self,
        addend: Self,
        rounding: FloatRounding,
        canonical_nan: bool,
    ) -> Option<FloatResult> {
        if self.format != rhs.format || self.format != addend.format {
            return None;
        }
        let round = rounding.arithmetic()?;
        with_format!(self.format, F => {
            let (left, right, addend) = (F::from_bits(self.bits()), F::from_bits(rhs.bits()), F::from_bits(addend.bits()));
            let result = left.mul_add_r(right, addend, round);
            let result = fused_overflow_status(left, right, addend, round, result);
            rounded_result(self.format, result.value.to_bits(), result.status, canonical_nan)
        })
    }

    /// Converts to a bounded integer, clamping invalid results to an endpoint.
    ///
    /// The width is from one through 128 bits. Signed
    /// results use two's-complement bits. Negative overflow clamps to the
    /// minimum; positive overflow clamps to the maximum. `nan_maximum` chooses
    /// the maximum for every NaN, or zero when false. Invalid conversions raise
    /// only invalid, while valid fractional conversions raise inexact.
    #[must_use]
    pub fn to_integer_clamped(
        self,
        width: u32,
        signed: bool,
        nan_maximum: bool,
        rounding: FloatRounding,
    ) -> Option<FloatIntegerResult> {
        if !(1..=128).contains(&width) {
            return None;
        }
        let round = rounding.arithmetic()?;
        let mask = BitWidth::new(width).checked_mask_u128()?;
        with_format!(self.format, F => {
            let value = F::from_bits(self.bits());
            if value.is_nan() {
                return Some(FloatIntegerResult {
                    bits: if nan_maximum { mask >> u32::from(signed) } else { 0 },
                    exceptions: FloatExceptions::INVALID,
                });
            }
            let mut exact = false;
            let result = if signed && width == 1 {
                // A signed bit holds -1 or zero. Convert through two bits so the
                // backend never receives an unsigned width of zero, then clamp
                // its rounded value to the actual one-bit interval.
                let converted = value.to_i128_r(2, round, &mut exact);
                if converted.value < -1 {
                    Status::INVALID_OP.and(1)
                } else if converted.value > 0 {
                    Status::INVALID_OP.and(0)
                } else {
                    converted.map(|value| value as u128 & mask)
                }
            } else if signed {
                value.to_i128_r(width as usize, round, &mut exact).map(|value| value as u128 & mask)
            } else {
                value.to_u128_r(width as usize, round, &mut exact)
            };
            Some(FloatIntegerResult {
                bits: result.value,
                exceptions: FloatExceptions::from(result.status),
            })
        })
    }

    /// Converts an integer bit pattern to the specified floating-point format.
    ///
    /// The width is from one through 128 bits. Bits
    /// outside that width are refused. Signed inputs use two's complement.
    /// The result reports rounding exceptions, including finite-endpoint overflow.
    #[must_use]
    pub fn from_integer_rounded(
        format: FloatFormat,
        bits: u128,
        width: u32,
        signed: bool,
        rounding: FloatRounding,
    ) -> Option<FloatResult> {
        if !(1..=128).contains(&width) {
            return None;
        }
        let mask = BitWidth::new(width).checked_mask_u128()?;
        if bits & !mask != 0 {
            return None;
        }
        let round = rounding.arithmetic()?;
        let shift = 128u32.checked_sub(width)?;
        let signed_value = ((bits << shift) as i128) >> shift;
        with_format!(format, F => {
            let result = if signed { F::from_i128_r(signed_value, round) }
                else { F::from_u128_r(bits, round) };
            let result = integer_overflow_status(bits, signed.then_some(signed_value), round, result);
            rounded_result(format, result.value.to_bits(), result.status, true)
        })
    }

    /// Converts formats with explicit rounding, NaN policy and exception bits.
    ///
    /// Odd rounding truncates first, then sets the least retained bit if inexact.
    /// A NaN converts to the canonical NaN where `canonical_nan` says so, and
    /// otherwise to the NaN the conversion propagates: quiet, its sign and the
    /// top of its payload kept (what [`NanPropagation::Operand`] names).
    #[must_use]
    pub fn convert_rounded(
        self,
        to: FloatFormat,
        rounding: FloatRounding,
        canonical_nan: bool,
    ) -> Option<FloatResult> {
        with_format!(self.format, F => {
            let value = F::from_bits(self.bits());
            match to {
                FloatFormat::Half => convert_rounded_to::<_, HalfS>(value, to, rounding, canonical_nan),
                FloatFormat::BFloat16 => convert_rounded_to::<_, BFloatS>(value, to, rounding, canonical_nan),
                FloatFormat::Single => convert_rounded_to::<_, SingleS>(value, to, rounding, canonical_nan),
                FloatFormat::Double => convert_rounded_to::<_, DoubleS>(value, to, rounding, canonical_nan),
                FloatFormat::X87Extended => convert_rounded_to::<_, X87DoubleExtendedS>(value, to, rounding, canonical_nan),
                FloatFormat::Quad => convert_rounded_to::<_, QuadS>(value, to, rounding, canonical_nan),
            }
        })
    }

    /// Compares values with explicit quiet or signaling unordered behavior.
    ///
    /// Signaling comparisons raise invalid for every NaN; quiet comparisons
    /// raise invalid only for signaling NaNs. No other exception is produced.
    #[must_use]
    pub fn compare_with_exceptions(
        self,
        rhs: Self,
        relation: FloatComparison,
        signaling: bool,
    ) -> Option<(bool, FloatExceptions)> {
        if self.format != rhs.format {
            return None;
        }
        with_format!(self.format, F => {
            let (left, right) = (F::from_bits(self.bits()), F::from_bits(rhs.bits()));
            let unordered = left.is_nan() || right.is_nan();
            let invalid = left.is_signaling() || right.is_signaling() || signaling && unordered;
            let order = left.partial_cmp(&right);
            let value = match relation {
                FloatComparison::Equal => order == Some(Ordering::Equal),
                FloatComparison::NotEqual => order != Some(Ordering::Equal),
                FloatComparison::Less => order == Some(Ordering::Less),
                FloatComparison::LessEqual => order.is_some_and(|order| !order.is_gt()),
                FloatComparison::Greater => order == Some(Ordering::Greater),
                FloatComparison::GreaterEqual => order.is_some_and(|order| !order.is_lt()),
            };
            Some((value, if invalid { FloatExceptions::INVALID } else { FloatExceptions::NONE }))
        })
    }

    /// Returns the IEEE class bit, from negative infinity at zero through quiet NaN at nine.
    #[must_use]
    pub fn classify(self) -> u16 {
        with_format!(self.format, F => {
            let value = F::from_bits(self.bits());
            let bit = if value.is_nan() { if value.is_signaling() { 8 } else { 9 } }
                else if value.is_infinite() { if value.is_negative() { 0 } else { 7 } }
                else if value.is_zero() { if value.is_negative() { 3 } else { 4 } }
                else if value.is_denormal() { if value.is_negative() { 2 } else { 5 } }
                else if value.is_negative() { 1 } else { 6 };
            1u16 << bit
        })
    }

    /// The lane Xeon Phi's fix-up with NaN passthrough (`vfixupnanps`,
    /// `vfixupnanpd`) gives this source and the 21-bit `table`: a NaN quieted,
    /// and any other value the token of its class, three bits of the table at
    /// `3 * class` for minus infinity (0), a negative number (1), minus zero
    /// (2), plus zero (3), a positive number (4) and plus infinity (5). The
    /// tokens give the destination unchanged (0, `Some(None)`), minus
    /// infinity, minus zero, plus zero, plus infinity, the default NaN (the
    /// quiet NaN with the sign set and no payload), the largest finite value
    /// and its negation (1 to 7). `None` for a subnormal source, which the
    /// instruction reads as a zero only under MXCSR.DAZ, and for a format
    /// without an implicit significand bit (x87).
    #[must_use]
    pub fn fixed_up_passing_nans(self, table: u128) -> Option<Option<Self>> {
        if self.format == FloatFormat::X87Extended {
            return None;
        }
        with_format!(self.format, F => {
            let value = F::from_bits(self.bits());
            if value.is_nan() {
                let quiet = 1u128.checked_shl(u32::try_from(F::PRECISION.checked_sub(2)?).ok()?)?;
                return Some(Self::new(self.format, self.bits() | quiet));
            }
            if value.is_denormal() {
                return None;
            }
            let negative = value.is_negative();
            let class: u32 = if value.is_infinite() {
                if negative { 0 } else { 5 }
            } else if value.is_zero() {
                if negative { 2 } else { 3 }
            } else if negative {
                1
            } else {
                4
            };
            let token = (table >> class.checked_mul(3)?) & 0b111;
            let result = match token {
                0 => return Some(None),
                1 => -F::INFINITY,
                2 => -F::ZERO,
                3 => F::ZERO,
                4 => F::INFINITY,
                5 => -F::NAN,
                6 => F::largest(),
                _ => return Some(Self::new(self.format, F::largest().to_bits()).map(Self::negate)),
            };
            Some(Self::new(self.format, result.to_bits()))
        })
    }

    /// Computes a correctly rounded square root for an implicit-significand binary format.
    ///
    /// Supports binary16, bfloat16, binary32, binary64 and binary128. The integer
    /// digit algorithm streams a significand of up to 226 bits through a bounded
    /// remainder. The x87 format and conversion-only odd rounding are refused.
    #[must_use]
    pub fn sqrt_rounded(self, rounding: FloatRounding, canonical_nan: bool) -> Option<FloatResult> {
        rounding.arithmetic()?;
        let (fraction_bits, exponent_bits, bias) = match self.format {
            FloatFormat::Half => (10u32, 5u32, 15i32),
            FloatFormat::BFloat16 => (7, 8, 127),
            FloatFormat::Single => (23, 8, 127),
            FloatFormat::Double => (52, 11, 1023),
            FloatFormat::Quad => (112, 15, 16383),
            FloatFormat::X87Extended => return None,
        };
        let bits = self.bits();
        let negative = bits.checked_shr(self.format.bits().checked_sub(1)?)? != 0;
        let leading = 1u128.checked_shl(fraction_bits)?;
        let fraction = bits & leading.checked_sub(1)?;
        let exponent_mask = BitWidth::new(exponent_bits).mask_u128();
        let encoded_exponent = bits.checked_shr(fraction_bits)? & exponent_mask;
        if encoded_exponent == exponent_mask && fraction != 0 {
            let quiet_bit = leading.checked_shr(1)?;
            return rounded_result(
                self.format,
                bits,
                if fraction & quiet_bit == 0 {
                    Status::INVALID_OP
                } else {
                    Status::OK
                },
                canonical_nan,
            );
        }
        if encoded_exponent == 0 && fraction == 0 {
            return Some(FloatResult {
                value: self,
                exceptions: FloatExceptions::NONE,
            });
        }
        if negative {
            return rounded_result(
                self.format,
                self.format.canonical_nan_bits(),
                Status::INVALID_OP,
                canonical_nan,
            );
        }
        if encoded_exponent == exponent_mask {
            return Some(FloatResult {
                value: self,
                exceptions: FloatExceptions::NONE,
            });
        }
        let mut mantissa = if encoded_exponent == 0 {
            fraction
        } else {
            leading | fraction
        };
        let mut exponent = if encoded_exponent == 0 {
            1i32.checked_sub(bias)?
        } else {
            i32::try_from(encoded_exponent).ok()?.checked_sub(bias)?
        };
        while mantissa < leading {
            mantissa = mantissa.checked_shl(1)?;
            exponent = exponent.checked_sub(1)?;
        }
        if exponent & 1 != 0 {
            mantissa = mantissa.checked_shl(1)?;
            exponent = exponent.checked_sub(1)?;
        }
        let radicand_bit = |position: u32| -> u128 {
            if position < fraction_bits {
                0
            } else {
                mantissa
                    .checked_shr(position.saturating_sub(fraction_bits))
                    .unwrap_or(0)
                    & 1
            }
        };
        let mut root = 0u128;
        let mut remainder = 0u128;
        for pair in (0..=fraction_bits).rev() {
            let position = pair.checked_mul(2)?;
            let next = radicand_bit(position) | (radicand_bit(position.checked_add(1)?) << 1);
            remainder = remainder.checked_shl(2)? | next;
            root = root.checked_shl(1)?;
            let trial = root.checked_shl(1)? | 1;
            if remainder >= trial {
                remainder = remainder.checked_sub(trial)?;
                root = root.checked_add(1)?;
            }
        }
        let increment = match rounding {
            FloatRounding::NearestEven | FloatRounding::NearestAway => remainder > root,
            FloatRounding::Up => remainder != 0,
            FloatRounding::TowardZero | FloatRounding::Down => false,
            FloatRounding::ToOdd => return None,
        };
        if increment {
            root = root.checked_add(1)?;
        }
        let mut result_exponent = exponent.checked_div(2)?;
        if root >= leading.checked_shl(1)? {
            root >>= 1;
            result_exponent = result_exponent.checked_add(1)?;
        }
        let encoded = u128::try_from(result_exponent.checked_add(bias)?)
            .ok()?
            .checked_shl(fraction_bits)?
            | (root & leading.checked_sub(1)?);
        Some(FloatResult {
            value: Self::new(self.format, encoded)?,
            exceptions: if remainder == 0 {
                FloatExceptions::NONE
            } else {
                FloatExceptions::INEXACT
            },
        })
    }

    /// `op` of two constants of one format, rounded as
    /// [`FloatRounding::decide`] says; `None` when their formats differ or an
    /// operand or the result is a NaN.
    fn binary(self, rhs: Self, op: Binary, rounding: Option<FloatRounding>) -> Option<Self> {
        if self.format != rhs.format {
            return None;
        }
        FloatRounding::decide(rounding, |rounding| {
            let round = rounding.arithmetic()?;
            let bits = with_format!(self.format, F => {
                let (a, b) = (F::from_bits(self.bits()), F::from_bits(rhs.bits()));
                if a.is_nan() || b.is_nan() {
                    return None;
                }
                let result = match op {
                    Binary::Add => a.add_r(b, round),
                    Binary::Sub => a.sub_r(b, round),
                    Binary::Mul => a.mul_r(b, round),
                    Binary::Div => a.div_r(b, round),
                    Binary::Rem => a.c_fmod(b),
                }
                .value;
                if result.is_nan() {
                    return None;
                }
                result.to_bits()
            });
            Self::new(self.format, bits)
        })
    }
}

/// An arithmetic-only exponent range, retaining the original significand precision.
///
/// The range covers products and quotients of every supported input format.
/// Values of this type are never encoded as interchange bit patterns.
struct ExpandedExponent<S>(PhantomData<S>);

impl<S: Semantics> Semantics for ExpandedExponent<S> {
    const BITS: usize = S::BITS;
    const EXP_BITS: usize = S::EXP_BITS;
    const PRECISION: usize = S::PRECISION;
    const MIN_EXP: i32 = -65_534;
    const MAX_EXP: i32 = 65_535;
}

/// Extends the exponent range without rounding an input's significand.
fn expanded_exponent<S: Semantics>(value: IeeeFloat<S>) -> IeeeFloat<ExpandedExponent<S>> {
    value.convert_r(Round::NearestTiesToEven, &mut false).value
}

/// Returns whether the software result may have saturated without an overflow flag.
fn needs_overflow_check<S: Semantics>(result: &StatusAnd<IeeeFloat<S>>) -> bool {
    result.status.contains(Status::INEXACT)
        && !result.status.contains(Status::OVERFLOW)
        && result.value.is_finite()
        && result.value.abs().bitwise_eq(IeeeFloat::<S>::largest())
}

/// Restores IEEE overflow when directed arithmetic returned a finite endpoint.
fn binary_overflow_status<S: Semantics>(
    left: IeeeFloat<S>,
    right: IeeeFloat<S>,
    operation: FloatBinary,
    round: Round,
    mut result: StatusAnd<IeeeFloat<S>>,
) -> StatusAnd<IeeeFloat<S>> {
    if !needs_overflow_check(&result) {
        return result;
    }
    let (left, right) = (expanded_exponent(left), expanded_exponent(right));
    let extended = match operation {
        FloatBinary::Add => left.add_r(right, round),
        FloatBinary::Subtract => left.sub_r(right, round),
        FloatBinary::Multiply => left.mul_r(right, round),
        FloatBinary::Divide => left.div_r(right, round),
        FloatBinary::MinimumNumber | FloatBinary::MaximumNumber => return result,
    }
    .value;
    if extended.abs() > expanded_exponent(IeeeFloat::<S>::largest()) {
        result.status |= Status::OVERFLOW;
    }
    result
}

/// Restores IEEE overflow from the unbounded-exponent fused result at the same precision.
fn fused_overflow_status<S: Semantics>(
    left: IeeeFloat<S>,
    right: IeeeFloat<S>,
    addend: IeeeFloat<S>,
    round: Round,
    mut result: StatusAnd<IeeeFloat<S>>,
) -> StatusAnd<IeeeFloat<S>> {
    if needs_overflow_check(&result) {
        let extended = expanded_exponent(left)
            .mul_add_r(expanded_exponent(right), expanded_exponent(addend), round)
            .value;
        if extended.abs() > expanded_exponent(IeeeFloat::<S>::largest()) {
            result.status |= Status::OVERFLOW;
        }
    }
    result
}

/// Restores overflow for an integer rounded to a finite floating-point endpoint.
fn integer_overflow_status<S: Semantics>(
    bits: u128,
    signed_value: Option<i128>,
    round: Round,
    mut result: StatusAnd<IeeeFloat<S>>,
) -> StatusAnd<IeeeFloat<S>> {
    if needs_overflow_check(&result) {
        let extended = if let Some(value) = signed_value {
            IeeeFloat::<ExpandedExponent<S>>::from_i128_r(value, round)
        } else {
            IeeeFloat::<ExpandedExponent<S>>::from_u128_r(bits, round)
        }
        .value;
        if extended.abs() > expanded_exponent(IeeeFloat::<S>::largest()) {
            result.status |= Status::OVERFLOW;
        }
    }
    result
}

/// Applies a specified NaN policy while preserving the IEEE exception set.
fn rounded_result(
    format: FloatFormat,
    bits: u128,
    status: Status,
    canonical_nan: bool,
) -> Option<FloatResult> {
    let mut value = FloatBits::new(format, bits)?;
    if value.is_nan() {
        if !canonical_nan {
            return None;
        }
        value = FloatBits::new(format, format.canonical_nan_bits())?;
    }
    Some(FloatResult {
        value,
        exceptions: FloatExceptions::from(status),
    })
}

/// Converts to one software floating-point destination format.
fn convert_rounded_to<FromS: Semantics, ToS: Semantics>(
    value: IeeeFloat<FromS>,
    format: FloatFormat,
    rounding: FloatRounding,
    canonical_nan: bool,
) -> Option<FloatResult> {
    let round = if rounding == FloatRounding::ToOdd {
        Round::TowardZero
    } else {
        rounding.arithmetic()?
    };
    let mut loses_info = false;
    let mut result: StatusAnd<IeeeFloat<ToS>> = value.convert_r(round, &mut loses_info);
    if needs_overflow_check(&result) {
        let extended: IeeeFloat<ExpandedExponent<ToS>> = value.convert_r(round, &mut false).value;
        if extended.abs() > expanded_exponent(IeeeFloat::<ToS>::largest()) {
            result.status |= Status::OVERFLOW;
        }
    }
    if result.value.is_nan() && !canonical_nan {
        return Some(FloatResult {
            value: FloatBits::new(format, result.value.to_bits())?,
            exceptions: FloatExceptions::from(result.status),
        });
    }
    let mut bits = result.value.to_bits();
    if rounding == FloatRounding::ToOdd
        && result.status.contains(Status::INEXACT)
        && result.value.is_finite()
    {
        bits |= 1;
    }
    rounded_result(format, bits, result.status, canonical_nan)
}

/// Computes the exact midpoint-table entry with bounded integer arithmetic.
fn estimate7_significand(index: u128, square_root: bool, exponent_odd: bool) -> Option<u128> {
    if !square_root {
        if index >= 128 {
            return None;
        }
        let denominator = index.checked_mul(2)?.checked_add(257)?;
        let quotient = 65536u128.checked_div(denominator)?;
        let remainder = 65536u128.checked_rem(denominator)?;
        return quotient
            .checked_add(u128::from(remainder.checked_mul(2)? > denominator))?
            .checked_sub(128);
    }
    if index >= 64 {
        return None;
    }
    let denominator = index.checked_mul(2)?.checked_add(129)?;
    let numerator = if exponent_odd {
        8388608u128
    } else {
        4194304u128
    };
    let mut low = 128u128;
    let mut high = 256u128;
    while high.checked_sub(low)? > 1 {
        let middle = low.checked_add(high)?.checked_div(2)?;
        if middle.checked_mul(middle)?.checked_mul(denominator)? <= numerator {
            low = middle;
        } else {
            high = middle;
        }
    }
    let twice = low.checked_mul(2)?.checked_add(1)?;
    let midpoint = twice.checked_mul(twice)?.checked_mul(denominator)?;
    low.checked_add(u128::from(numerator.checked_mul(4)? > midpoint))?
        .checked_sub(128)
}

/// The binary operations [`FloatBits`] evaluates.
#[derive(Clone, Copy)]
enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// The value in decimal ([`FloatBits::decimal`]), suffixed with its format:
/// `1.5f16`, `1.5bf16`, `1.5f`, `1.5`, `1.5f80`, `1.5f128`.
impl fmt::Display for FloatBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suffix = match self.format {
            FloatFormat::Half => "f16",
            FloatFormat::BFloat16 => "bf16",
            FloatFormat::Single => "f",
            FloatFormat::Double => "",
            FloatFormat::X87Extended => "f80",
            FloatFormat::Quad => "f128",
        };
        write!(f, "{}{suffix}", self.decimal())
    }
}

/// The format and the bits in hexadecimal: `Double(0x3ff8000000000000)`.
impl fmt::Debug for FloatBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({:#x})", self.format, self.bits())
    }
}

/// A binary32 constant of the single's bits.
impl From<f32> for FloatBits {
    fn from(value: f32) -> Self {
        Self {
            format: FloatFormat::Single,
            bits: [u64::from(value.to_bits()), 0],
        }
    }
}

/// A binary64 constant of the double's bits.
impl From<f64> for FloatBits {
    fn from(value: f64) -> Self {
        Self {
            format: FloatFormat::Double,
            bits: [value.to_bits(), 0],
        }
    }
}

/// The single a binary32 constant's bits encode; any other format is returned
/// as the error.
impl TryFrom<FloatBits> for f32 {
    type Error = FloatBits;

    fn try_from(value: FloatBits) -> Result<Self, Self::Error> {
        match value.format {
            FloatFormat::Single => u32::try_from(value.bits())
                .map(Self::from_bits)
                .map_err(|_| value),
            _ => Err(value),
        }
    }
}

/// The double a binary64 constant's bits encode; any other format is returned
/// as the error.
impl TryFrom<FloatBits> for f64 {
    type Error = FloatBits;

    fn try_from(value: FloatBits) -> Result<Self, Self::Error> {
        match value.format {
            FloatFormat::Double => u64::try_from(value.bits())
                .map(Self::from_bits)
                .map_err(|_| value),
            _ => Err(value),
        }
    }
}

impl<T: Target> Emit<T> for FloatBits {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self { format, bits } = self;
        Emit::emit(format, writer)?;
        Emit::emit(bits, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for FloatFormat {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Half => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::BFloat16 => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Single => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Double => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::X87Extended => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Quad => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for FloatRounding {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::NearestEven => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::TowardZero => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Down => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Up => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::NearestAway => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::ToOdd => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for FloatExceptions {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self(bits) = self;
        Emit::emit(bits, writer)
    }
}
impl<T: Target> Emit<T> for FloatExceptionLayout {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self { bits } = self;
        Emit::emit(bits, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for FloatClassLayout {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::RiscV => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Mips => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::X87 => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use super::*;

    /// The rounding the host's arithmetic takes, which the oracles here
    /// compute by.
    const NEAREST: Option<FloatRounding> = Some(FloatRounding::NearestEven);

    /// Ratified V 1.0 reciprocal table, independent of the integer midpoint computation.
    const RECIPROCAL_TABLE: [u8; 128] = [
        127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 99, 97, 96,
        94, 93, 91, 90, 88, 87, 85, 84, 83, 81, 80, 79, 77, 76, 75, 74, 72, 71, 70, 69, 68, 66, 65,
        64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
        41, 40, 40, 39, 38, 37, 36, 35, 35, 34, 33, 32, 31, 31, 30, 29, 28, 28, 27, 26, 25, 25, 24,
        23, 23, 22, 21, 21, 20, 19, 19, 18, 17, 17, 16, 15, 15, 14, 14, 13, 12, 12, 11, 11, 10, 9,
        9, 8, 8, 7, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0,
    ];

    /// Ratified V 1.0 reciprocal-square-root tables for the two exponent parities.
    const RECIPROCAL_ROOT_TABLE: [[u8; 64]; 2] = [
        [
            52, 51, 50, 48, 47, 46, 44, 43, 42, 41, 40, 39, 38, 36, 35, 34, 33, 32, 31, 30, 30, 29,
            28, 27, 26, 25, 24, 23, 23, 22, 21, 20, 19, 19, 18, 17, 16, 16, 15, 14, 14, 13, 12, 12,
            11, 10, 10, 9, 9, 8, 7, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0,
        ],
        [
            127, 125, 123, 121, 119, 118, 116, 114, 113, 111, 109, 108, 106, 105, 103, 102, 100,
            99, 97, 96, 95, 93, 92, 91, 90, 88, 87, 86, 85, 84, 83, 82, 80, 79, 78, 77, 76, 75, 74,
            73, 72, 71, 70, 70, 69, 68, 67, 66, 65, 64, 63, 63, 62, 61, 60, 59, 59, 58, 57, 56, 56,
            55, 54, 53,
        ],
    ];

    /// The constant `value` as an x87 extended float, exactly.
    fn x87(value: f64) -> FloatBits {
        FloatBits::new(FloatFormat::Double, u128::from(value.to_bits()))
            .and_then(|double| double.exactly(FloatFormat::X87Extended))
            .expect("an f64 is an x87 extended float exactly")
    }

    /// The x87 extended constant `value` as a double, exactly.
    fn double(value: FloatBits) -> f64 {
        let bits = value
            .exactly(FloatFormat::Double)
            .expect("the result is a double exactly")
            .bits();
        f64::from_bits(u64::try_from(bits).expect("64 bits"))
    }

    /// **`fscale` multiplies by two to the power rounded toward zero, where
    /// that does not round.** `3 * 2^trunc(2.7)` is 12 and `1 * 2^trunc(-1.5)`
    /// is 0.5; a NaN, an infinity, and a product past the format's range
    /// decline.
    #[test]
    fn a_scale_is_the_product_by_the_truncated_power() {
        let scaled = |value: f64, by: f64| x87(value).scaled(x87(by)).map(double);
        assert_eq!(scaled(3.0, 2.7), Some(12.0));
        assert_eq!(scaled(1.0, -1.5), Some(0.5));
        assert_eq!(scaled(-0.75, 3.0), Some(-6.0));
        // The x87 default quiet NaN: exponent all ones, integer and quiet bits.
        let nan = FloatBits::new(FloatFormat::X87Extended, 0x7fff_c000_0000_0000_0000)
            .expect("an x87 NaN");
        assert_eq!(nan.scaled(x87(1.0)), None);
        assert_eq!(scaled(f64::INFINITY, 1.0), None);
        assert_eq!(scaled(1.0, 40_000.0), None, "past the exponent range");
    }

    /// **`fxtract` gives the exponent and the significand in `[1, 2)`.** 12 is
    /// `1.5 * 2^3` and -0.375 is `-1.5 * 2^-2`; a zero, whose extraction
    /// signals divide-by-zero, and an infinity decline.
    #[test]
    fn an_extraction_is_the_exponent_and_the_significand() {
        let extracted = |value: f64| {
            x87(value)
                .extracted()
                .map(|(exponent, significand)| (double(exponent), double(significand)))
        };
        assert_eq!(extracted(12.0), Some((3.0, 1.5)));
        assert_eq!(extracted(-0.375), Some((-2.0, -1.5)));
        assert_eq!(extracted(0.0), None);
        assert_eq!(extracted(f64::INFINITY), None);
    }

    /// **`fprem` and `fprem1` are the remainders of the truncated and the
    /// nearest quotient, with its three low bits.** `7 / 2` truncates to 3
    /// (remainder 1, bits `011`) and rounds to the even 4 (remainder -1, bits
    /// `100`); -7 keeps its sign in the remainder and the magnitude's bits;
    /// `fmod` is the truncated remainder's oracle. A dividend 64 binary orders
    /// above the divisor, which the instruction reduces only partly, and a
    /// zero divisor decline.
    #[test]
    fn a_partial_remainder_is_exact_with_its_quotient_bits() {
        let remainder = |value: f64, divisor: f64, nearest: bool| {
            x87(value)
                .partial_remainder(x87(divisor), nearest)
                .map(|(remainder, quotient)| (double(remainder), quotient))
        };
        assert_eq!(remainder(7.0, 2.0, false), Some((7.0 % 2.0, 3)));
        assert_eq!(remainder(7.0, 2.0, true), Some((-1.0, 4)));
        assert_eq!(remainder(-7.0, 2.0, false), Some((-7.0 % 2.0, 3)));
        assert_eq!(remainder(5.5, 1.25, false), Some((5.5 % 1.25, 4)));
        assert_eq!(remainder(1.0, 4.0, true), Some((1.0, 0)));
        assert_eq!(remainder(0.0, 3.0, false), Some((0.0, 0)));
        assert_eq!(remainder(2.0_f64.powi(70), 3.0, false), None);
        assert_eq!(remainder(1.0, 0.0, false), None);
    }

    /// A bounded round keeps an integral value its integer holds and gives
    /// the integer's minimum for one past either end, an infinity and a NaN.
    #[test]
    fn a_bounded_round_gives_the_minimum_out_of_range() {
        let round = |value: f64, width| {
            FloatBits::from(value)
                .integral_within(FloatRounding::NearestEven, width)
                .and_then(|rounded| f64::try_from(rounded).ok())
        };
        let minimum = -2_147_483_648.0;
        assert_eq!(round(2.5, 32), Some(2.0));
        assert_eq!(round(-2.5, 32), Some(-2.0));
        assert_eq!(round(minimum, 32), Some(minimum));
        assert_eq!(round(2_147_483_647.0, 32), Some(2_147_483_647.0));
        assert_eq!(round(2_147_483_648.0, 32), Some(minimum));
        assert_eq!(round(-2_147_483_649.0, 32), Some(minimum));
        assert_eq!(round(f64::INFINITY, 32), Some(minimum));
        assert_eq!(round(f64::NAN, 32), Some(minimum));
        assert_eq!(round(2_147_483_648.0, 64), Some(2_147_483_648.0));
        assert_eq!(round(1e19, 64), Some(-9_223_372_036_854_775_808.0));
        assert_eq!(
            FloatBits::from(2.5f64).integral_within(FloatRounding::ToOdd, 32),
            None
        );
    }

    /// A layout puts each flag at its own bit and reads it back, dropping a
    /// flag it keeps nowhere: AArch64's FPSR keeps input-denormal at bit 7
    /// and IEEE's five in reverse order from bit 0.
    #[test]
    fn a_layout_keeps_each_flag_at_its_bit() {
        let fpsr =
            FloatExceptionLayout::new([Some(4), Some(3), Some(2), Some(1), Some(0), Some(7)]);
        let raised = FloatExceptions::INEXACT.union(FloatExceptions::INPUT_DENORMAL);
        assert_eq!(fpsr.place(raised), 0x90);
        assert_eq!(fpsr.read(0x90), raised);
        assert_eq!(
            fpsr.holds(),
            FloatExceptions::ALL.difference(FloatExceptions::ROUNDED_UP)
        );
        assert_eq!(FloatExceptionLayout::IEEE.place(raised), 0x1);
        assert_eq!(FloatExceptionLayout::IEEE.holds(), FloatExceptions::IEEE);
        assert_eq!(FloatExceptionLayout::IEEE.read(0x3f), FloatExceptions::IEEE);
        assert_eq!(raised.to_string(), "inexact|denormal");
        assert_eq!(FloatExceptions::NONE.to_string(), "none");
        assert_eq!(
            FloatExceptions::ALL.difference(FloatExceptions::IEEE),
            FloatExceptions::INPUT_DENORMAL.union(FloatExceptions::ROUNDED_UP)
        );
        assert_eq!(FloatExceptions::from_bits(0xff), FloatExceptions::ALL);
    }

    /// **A legacy MIPS NaN raises by its kind in that encoding.** `0x7fbfffff`,
    /// its quiet bit clear, is legacy MIPS's quiet NaN and reads as an IEEE
    /// quiet one; `0x7fc00000`, the bit set and nothing else of the fraction,
    /// is a signalling one and reads as the IEEE signalling NaN `0x7f800001`,
    /// not as infinity. A quiet compare of the quiet one raises nothing, of
    /// the signalling one invalid; the control, the same compare in IEEE's
    /// encoding, reads them the other way round, and where no encoding is
    /// stated it folds nothing.
    #[test]
    fn a_legacy_nan_raises_by_its_kind_in_that_encoding() {
        let single = |bits| FloatBits::new(FloatFormat::Single, bits).unwrap();
        let (quiet, signalling) = (single(0x7fbf_ffff), single(0x7fc0_0000));
        assert_eq!(quiet.from_legacy_nan().bits(), 0x7fff_ffff);
        assert_eq!(signalling.from_legacy_nan().bits(), 0x7f80_0001);
        assert_eq!(single(0x3f80_0000).from_legacy_nan().bits(), 0x3f80_0000);
        let compare = |value: FloatBits, nan_encoding: NanEncoding| {
            FloatRules {
                nan_encoding,
                ..FloatRules::NONE
            }
            .raises(
                RaisingArithmetic::Compare { signaling: false },
                &[value, value],
                Some(FloatRounding::NearestEven),
                FloatExceptions::ALL,
            )
        };
        assert_eq!(
            compare(quiet, NanEncoding::MipsLegacy),
            Some(FloatExceptions::NONE)
        );
        assert_eq!(
            compare(signalling, NanEncoding::MipsLegacy),
            Some(FloatExceptions::INVALID)
        );
        assert_eq!(
            compare(quiet, NanEncoding::Ieee),
            Some(FloatExceptions::INVALID)
        );
        assert_eq!(
            compare(signalling, NanEncoding::Ieee),
            Some(FloatExceptions::NONE)
        );
        assert_eq!(compare(quiet, NanEncoding::Unstated), None);
    }

    /// A layout stated as a word reads back as the same layout: each flag at
    /// its bit, bit zero apart from a flag held nowhere.
    #[test]
    fn a_layout_survives_its_word() {
        let x87 = FloatExceptionLayout::new([Some(5), Some(4), Some(3), Some(2), Some(0), Some(1)])
            .rounded_up_at(9);
        let fpsr =
            FloatExceptionLayout::new([Some(4), Some(3), Some(2), Some(1), Some(0), Some(7)]);
        for layout in [FloatExceptionLayout::IEEE, x87, fpsr] {
            assert_eq!(FloatExceptionLayout::from(u64::from(layout)), layout);
        }
        assert_eq!(u64::from(FloatExceptionLayout::IEEE), 0x05_0403_0201);
        assert_eq!(FloatExceptionLayout::from(0).holds(), FloatExceptions::NONE);
    }

    /// **A layout keeps the rounding direction only where it names a bit for
    /// it**: x87's `C1`, which no exception layout holds.
    #[test]
    fn a_layout_keeps_the_rounding_direction_where_it_names_one() {
        let c1 = FloatExceptionLayout::new([None; 6]).rounded_up_at(0);
        let raised = FloatExceptions::INEXACT.union(FloatExceptions::ROUNDED_UP);
        assert_eq!(c1.holds(), FloatExceptions::ROUNDED_UP);
        assert_eq!(c1.place(raised), 1);
        assert_eq!(c1.read(1), FloatExceptions::ROUNDED_UP);
        assert_eq!(
            FloatExceptionLayout::IEEE.place(FloatExceptions::ROUNDED_UP),
            0
        );
        assert_eq!(raised.to_string(), "inexact|roundup");
    }

    #[test]
    fn controlled_float_estimates_match_every_ratified_lookup_entry() {
        for (format, fraction_bits, bias) in [
            (FloatFormat::Half, 10u32, 15u128),
            (FloatFormat::BFloat16, 7, 127),
            (FloatFormat::Single, 23, 127),
            (FloatFormat::Double, 52, 1023),
            (FloatFormat::Quad, 112, 16383),
        ] {
            for (index, expected) in RECIPROCAL_TABLE.into_iter().enumerate() {
                let input = FloatBits::new(
                    format,
                    (bias << fraction_bits) | ((index as u128) << (fraction_bits - 7)),
                )
                .unwrap();
                let result = input
                    .estimate7(false, FloatRounding::NearestEven, true)
                    .unwrap();
                assert_eq!(
                    result.value.bits(),
                    ((bias - 1) << fraction_bits) | (u128::from(expected) << (fraction_bits - 7))
                );
                assert_eq!(result.exceptions.bits(), 0);
            }
            for (parity, table) in RECIPROCAL_ROOT_TABLE.into_iter().enumerate() {
                let exponent = bias - 1 + parity as u128;
                for (index, expected) in table.into_iter().enumerate() {
                    let input = FloatBits::new(
                        format,
                        (exponent << fraction_bits) | ((index as u128) << (fraction_bits - 6)),
                    )
                    .unwrap();
                    for mode in [
                        FloatRounding::NearestEven,
                        FloatRounding::TowardZero,
                        FloatRounding::Down,
                        FloatRounding::Up,
                        FloatRounding::NearestAway,
                    ] {
                        let result = input.estimate7(true, mode, true).unwrap();
                        assert_eq!(
                            result.value.bits(),
                            ((bias - parity as u128) << fraction_bits)
                                | (u128::from(expected) << (fraction_bits - 7))
                        );
                        assert_eq!(result.exceptions.bits(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn controlled_float_estimates_keep_subnormals_specials_and_overflow_flags() {
        for (bits, reciprocal, root) in [
            (0x00718abcu128, 0x7e900000u128, 0x5f080000u128),
            (0x7f765432, 0x00214000, 0x1f820000),
        ] {
            let value = FloatBits::new(FloatFormat::Single, bits).unwrap();
            for (square_root, expected) in [(false, reciprocal), (true, root)] {
                let result = value
                    .estimate7(square_root, FloatRounding::NearestEven, true)
                    .unwrap();
                assert_eq!(
                    (result.value.bits(), result.exceptions.bits()),
                    (expected, 0)
                );
            }
        }
        for (bits, square_root, expected, flags) in [
            (0u16, false, 0x7c00u128, 8),
            (0x8000, false, 0xfc00, 8),
            (0, true, 0x7c00, 8),
            (0x8000, true, 0xfc00, 8),
            (0x7c00, false, 0, 0),
            (0xfc00, false, 0x8000, 0),
            (0x7c00, true, 0, 0),
            (0xfc00, true, 0x7e00, 16),
            (0xbc00, true, 0x7e00, 16),
            (0x8001, true, 0x7e00, 16),
            (0x7c01, false, 0x7e00, 16),
            (0xfe55, false, 0x7e00, 0),
            (0x7c01, true, 0x7e00, 16),
            (0xfe55, true, 0x7e00, 0),
        ] {
            let result = half(bits)
                .estimate7(square_root, FloatRounding::NearestEven, true)
                .unwrap();
            assert_eq!(
                (result.value.bits(), result.exceptions.bits()),
                (expected, flags)
            );
        }
        for (mode, positive, negative) in [
            (FloatRounding::NearestEven, 0x7c00, 0xfc00),
            (FloatRounding::NearestAway, 0x7c00, 0xfc00),
            (FloatRounding::TowardZero, 0x7bff, 0xfbff),
            (FloatRounding::Down, 0x7bff, 0xfc00),
            (FloatRounding::Up, 0x7c00, 0xfbff),
        ] {
            for (input, expected) in [(1u16, positive), (0x8001, negative)] {
                let result = half(input).estimate7(false, mode, true).unwrap();
                assert_eq!(
                    (result.value.bits(), result.exceptions.bits()),
                    (expected, 5)
                );
            }
        }
        assert!(
            half(0x7e55)
                .estimate7(false, FloatRounding::NearestEven, false)
                .is_none()
        );
        assert!(
            half(0x3c00)
                .estimate7(false, FloatRounding::ToOdd, true)
                .is_none()
        );
    }

    #[test]
    fn controlled_float_sign_injection_preserves_every_half_payload() {
        for bits in 0u16..=u16::MAX {
            for rhs in [0u16, 0x8000, 0x7c01, 0xfe55] {
                for (negate, xor) in [(false, false), (true, false), (false, true), (true, true)] {
                    let sign = (rhs & 0x8000)
                        ^ if negate { 0x8000 } else { 0 }
                        ^ if xor { bits & 0x8000 } else { 0 };
                    assert_eq!(
                        half(bits)
                            .inject_sign(half(rhs), negate, xor)
                            .unwrap()
                            .bits(),
                        u128::from((bits & 0x7fff) | sign)
                    );
                }
            }
        }
        assert!(
            half(0)
                .inject_sign(FloatBits::from(0.0f32), false, false)
                .is_none()
        );
    }

    #[test]
    fn controlled_float_integer_clamping_matches_every_half_encoding() {
        for bits in 0u16..=u16::MAX {
            let magnitude = bits & 0x7fff;
            let input = if magnitude > 0x7c00 {
                f64::NAN
            } else if magnitude == 0x7c00 {
                f64::INFINITY
            } else {
                reference_positive_half(magnitude)
            };
            let input = if bits & 0x8000 != 0 { -input } else { input };
            for (mode, rounded) in [
                (FloatRounding::NearestEven, input.round_ties_even()),
                (FloatRounding::TowardZero, input.trunc()),
                (FloatRounding::Down, input.floor()),
                (FloatRounding::Up, input.ceil()),
                (FloatRounding::NearestAway, input.round()),
            ] {
                for width in [1u32, 7, 8, 16, 32] {
                    for signed in [false, true] {
                        let mask = (1u128 << width) - 1;
                        let maximum = if signed {
                            (1i128 << (width - 1)) - 1
                        } else {
                            mask as i128
                        };
                        let minimum = if signed { -(1i128 << (width - 1)) } else { 0 };
                        let invalid = !rounded.is_finite()
                            || rounded < minimum as f64
                            || rounded > maximum as f64;
                        let expected = if rounded.is_nan() || rounded > maximum as f64 {
                            maximum
                        } else if rounded < minimum as f64 {
                            minimum
                        } else {
                            rounded as i128
                        };
                        let flags = if invalid {
                            16
                        } else {
                            u8::from(input != rounded)
                        };
                        let result = half(bits)
                            .to_integer_clamped(width, signed, true, mode)
                            .unwrap();
                        assert_eq!(
                            (result.bits, result.exceptions.bits()),
                            (expected as u128 & mask, flags),
                            "input={bits:04x} width={width} signed={signed} mode={mode:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn controlled_float_integer_conversions_keep_wide_signs_ties_and_nan_policy() {
        for format in FloatFormat::ALL {
            for width in [8u32, 16, 32, 64, 128] {
                let mask = u128::MAX >> (128 - width);
                let negative_one = FloatBits::from_integer_rounded(
                    format,
                    mask,
                    width,
                    true,
                    FloatRounding::NearestEven,
                )
                .unwrap();
                assert_eq!(
                    negative_one.value,
                    FloatBits::from(-1.0f64).exactly(format).unwrap()
                );
                assert_eq!(negative_one.exceptions.bits(), 0);
                for signed in [false, true] {
                    for nan_bits in [format.canonical_nan_bits(), format.canonical_nan_bits() | 1] {
                        let nan = FloatBits::new(format, nan_bits).unwrap();
                        for nan_maximum in [false, true] {
                            let result = nan
                                .to_integer_clamped(
                                    width,
                                    signed,
                                    nan_maximum,
                                    FloatRounding::NearestEven,
                                )
                                .unwrap();
                            assert_eq!(
                                result.bits,
                                if nan_maximum {
                                    mask >> u32::from(signed)
                                } else {
                                    0
                                }
                            );
                            assert_eq!(result.exceptions.bits(), 16);
                        }
                    }
                }
            }
        }
        for (bits, signed, expected) in [
            (2049u128, false, 0x6800),
            (0x8001, true, 0xf800),
            (0xffff, true, 0xbc00),
        ] {
            let result = FloatBits::from_integer_rounded(
                FloatFormat::Half,
                bits,
                16,
                signed,
                FloatRounding::NearestEven,
            )
            .unwrap();
            assert_eq!(result.value.bits(), expected);
        }
        for (input, mode, expected, flags) in [
            (65504u128, FloatRounding::TowardZero, 0x7bff, 0),
            (65505, FloatRounding::TowardZero, 0x7bff, 1),
            (131008, FloatRounding::TowardZero, 0x7bff, 5),
            (131008, FloatRounding::Up, 0x7c00, 5),
            (u128::MAX, FloatRounding::TowardZero, 0x7bff, 5),
        ] {
            let result =
                FloatBits::from_integer_rounded(FloatFormat::Half, input, 128, false, mode)
                    .unwrap();
            assert_eq!(
                (result.value.bits(), result.exceptions.bits()),
                (expected, flags)
            );
        }
        let minimum = FloatBits::from_integer_rounded(
            FloatFormat::Quad,
            1u128 << 127,
            128,
            true,
            FloatRounding::NearestEven,
        )
        .unwrap();
        assert_eq!(minimum.value.bits(), 0xc07e0000000000000000000000000000);
        assert_eq!(minimum.exceptions.bits(), 0);
        let result = minimum
            .value
            .to_integer_clamped(128, true, true, FloatRounding::NearestEven)
            .unwrap();
        assert_eq!((result.bits, result.exceptions.bits()), (1u128 << 127, 0));
        let maximum =
            FloatBits::new(FloatFormat::Quad, 0x407e0000000000000000000000000000).unwrap();
        let result = maximum
            .to_integer_clamped(128, true, true, FloatRounding::NearestEven)
            .unwrap();
        assert_eq!(
            (result.bits, result.exceptions.bits()),
            ((1u128 << 127) - 1, 16)
        );
        for width in [0, 129, 256] {
            assert!(
                FloatBits::from_integer_rounded(
                    FloatFormat::Half,
                    0,
                    width,
                    false,
                    FloatRounding::NearestEven
                )
                .is_none()
            );
            assert!(
                half(0)
                    .to_integer_clamped(width, false, true, FloatRounding::NearestEven)
                    .is_none()
            );
        }
        assert!(
            FloatBits::from_integer_rounded(
                FloatFormat::Half,
                256,
                8,
                false,
                FloatRounding::NearestEven
            )
            .is_none()
        );
        assert!(
            half(0)
                .to_integer_clamped(32, false, true, FloatRounding::ToOdd)
                .is_none()
        );
    }

    #[test]
    fn controlled_float_rounding_and_flags_match_literal_half_boundaries() {
        for (mode, positive, negative, overflow, tiny) in [
            (FloatRounding::NearestEven, 0x3c00, 0xbc00, 0x7c00, 0),
            (FloatRounding::TowardZero, 0x3c00, 0xbc00, 0x7bff, 0),
            (FloatRounding::Down, 0x3c00, 0xbc01, 0x7bff, 0),
            (FloatRounding::Up, 0x3c01, 0xbc00, 0x7c00, 1),
            (FloatRounding::NearestAway, 0x3c01, 0xbc01, 0x7c00, 1),
        ] {
            for (left, right, expected) in [(0x3c00, 0x1000, positive), (0xbc00, 0x9000, negative)]
            {
                let result = half(left)
                    .binary_rounded(half(right), FloatBinary::Add, mode, true)
                    .unwrap();
                assert_eq!(result.value.bits(), expected);
                assert_eq!(result.exceptions.bits(), 1);
            }
            let result = half(0x7bff)
                .binary_rounded(half(0x7bff), FloatBinary::Add, mode, true)
                .unwrap();
            assert_eq!(
                (result.value.bits(), result.exceptions.bits()),
                (overflow, 5)
            );
            let result = half(1)
                .binary_rounded(half(0x4000), FloatBinary::Divide, mode, true)
                .unwrap();
            assert_eq!((result.value.bits(), result.exceptions.bits()), (tiny, 3));
        }
        let result = half(0x0400)
            .binary_rounded(
                half(0x4000),
                FloatBinary::Divide,
                FloatRounding::NearestEven,
                true,
            )
            .unwrap();
        assert_eq!((result.value.bits(), result.exceptions.bits()), (0x0200, 0));
        for (operation, left, right, bits, flags) in [
            (FloatBinary::Divide, 0x3c00, 0, 0x7c00, 8),
            (FloatBinary::Divide, 0xbc00, 0, 0xfc00, 8),
            (FloatBinary::Divide, 0, 0, 0x7e00, 16),
            (FloatBinary::Multiply, 0x7c00, 0, 0x7e00, 16),
            (FloatBinary::Add, 0xfe55, 0x3c00, 0x7e00, 0),
            (FloatBinary::Add, 0x7c01, 0x3c00, 0x7e00, 16),
        ] {
            let result = half(left)
                .binary_rounded(half(right), operation, FloatRounding::NearestEven, true)
                .unwrap();
            assert_eq!(
                (result.value.bits(), result.exceptions.bits()),
                (bits, flags)
            );
        }
        assert!(
            half(0x7e55)
                .binary_rounded(
                    half(0x3c00),
                    FloatBinary::Add,
                    FloatRounding::NearestEven,
                    false
                )
                .is_none()
        );
        for (value, mode) in [
            (0, FloatRounding::NearestEven),
            (1, FloatRounding::TowardZero),
            (2, FloatRounding::Down),
            (3, FloatRounding::Up),
            (4, FloatRounding::NearestAway),
        ] {
            assert_eq!(FloatRounding::of_environment(value), Some(mode));
        }
        for value in 0u8..=255 {
            assert_eq!(
                FloatRounding::of_environment(u64::from(value)).is_some(),
                value < 5
            );
        }
    }

    #[test]
    fn controlled_float_minimum_comparison_and_classification_keep_nan_and_zero_rules() {
        for (operation, zero) in [
            (FloatBinary::MinimumNumber, 0x8000),
            (FloatBinary::MaximumNumber, 0),
        ] {
            for (left, right) in [(0, 0x8000), (0x8000, 0)] {
                let result = half(left)
                    .binary_rounded(half(right), operation, FloatRounding::NearestEven, true)
                    .unwrap();
                assert_eq!((result.value.bits(), result.exceptions.bits()), (zero, 0));
            }
            for (nan, flags) in [(0x7e11, 0), (0x7c01, 16)] {
                let result = half(nan)
                    .binary_rounded(half(0x3c00), operation, FloatRounding::NearestEven, true)
                    .unwrap();
                assert_eq!(
                    (result.value.bits(), result.exceptions.bits()),
                    (0x3c00, flags)
                );
            }
        }
        for (index, bits) in [
            0xfc00, 0xbc00, 0x8001, 0x8000, 0, 1, 0x3c00, 0x7c00, 0x7c01, 0x7e01,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(half(bits).classify(), 1 << index);
        }
        for relation in [
            FloatComparison::Equal,
            FloatComparison::NotEqual,
            FloatComparison::Less,
            FloatComparison::LessEqual,
            FloatComparison::Greater,
            FloatComparison::GreaterEqual,
        ] {
            for (nan, flags) in [(0x7e11, 0), (0x7c01, 16)] {
                let (value, exceptions) = half(nan)
                    .compare_with_exceptions(half(0x3c00), relation, false)
                    .unwrap();
                assert_eq!(value, relation == FloatComparison::NotEqual);
                assert_eq!(exceptions.bits(), flags);
                assert_eq!(
                    half(nan)
                        .compare_with_exceptions(half(0x3c00), relation, true)
                        .unwrap()
                        .1,
                    FloatExceptions::INVALID
                );
            }
        }
        assert!(
            half(0x3c00)
                .compare_with_exceptions(FloatBits::from(1.0f32), FloatComparison::Equal, false)
                .is_none()
        );
    }

    #[test]
    fn controlled_float_fma_rounds_once_and_reports_invalid_products_with_nan_addends() {
        let result = half(0x3c01)
            .mul_add_rounded(half(0x3bfe), half(0xbc00), FloatRounding::NearestEven, true)
            .unwrap();
        assert_eq!((result.value.bits(), result.exceptions.bits()), (0x8010, 0));
        let product = half(0x3c01)
            .binary_rounded(
                half(0x3bfe),
                FloatBinary::Multiply,
                FloatRounding::NearestEven,
                true,
            )
            .unwrap();
        assert_eq!(product.value.bits(), 0x3c00);
        for addend in [0x3c00, 0x7e01, 0x7c01] {
            let result = half(0)
                .mul_add_rounded(half(0x7c00), half(addend), FloatRounding::NearestEven, true)
                .unwrap();
            assert_eq!(
                (result.value.bits(), result.exceptions.bits()),
                (0x7e00, 16)
            );
        }
    }

    #[test]
    fn controlled_float_conversion_detects_tininess_after_rounding_and_jams_odd_results() {
        for (source, mode, expected, flags) in [
            (0x3f801000, FloatRounding::NearestEven, 0x3c00, 1),
            (0x3f801000, FloatRounding::NearestAway, 0x3c01, 1),
            (0x3f801000, FloatRounding::ToOdd, 0x3c01, 1),
            (0xbf801000, FloatRounding::ToOdd, 0xbc01, 1),
            (0x387fe000, FloatRounding::NearestEven, 0x0400, 1),
            (0x387fe000, FloatRounding::TowardZero, 0x03ff, 3),
            (0x387fe000, FloatRounding::ToOdd, 0x03ff, 3),
        ] {
            let source = FloatBits::new(FloatFormat::Single, source).unwrap();
            let result = source
                .convert_rounded(FloatFormat::Half, mode, true)
                .unwrap();
            assert_eq!(
                (result.value.bits(), result.exceptions.bits()),
                (expected, flags)
            );
        }
        for (bits, mode, expected, flags) in [
            (0x3f808000, FloatRounding::NearestEven, 0x3f80, 1),
            (0x3f808000, FloatRounding::NearestAway, 0x3f81, 1),
            (0x3f808000, FloatRounding::ToOdd, 0x3f81, 1),
            (1, FloatRounding::NearestEven, 0, 3),
            (1, FloatRounding::ToOdd, 1, 3),
        ] {
            let result = FloatBits::new(FloatFormat::Single, bits)
                .unwrap()
                .convert_rounded(FloatFormat::BFloat16, mode, true)
                .unwrap();
            assert_eq!(
                (result.value.bits(), result.exceptions.bits()),
                (expected, flags)
            );
        }
    }

    #[test]
    fn controlled_float_bfloat_widening_preserves_every_finite_encoding_and_nan_flags() {
        for bits in 0u16..=u16::MAX {
            let source = FloatBits::new(FloatFormat::BFloat16, u128::from(bits)).unwrap();
            let result = source
                .convert_rounded(FloatFormat::Single, FloatRounding::NearestEven, true)
                .unwrap();
            let nan = bits & 0x7f80 == 0x7f80 && bits & 0x7f != 0;
            let expected = if nan {
                0x7fc0_0000
            } else {
                u128::from(bits) << 16
            };
            let flags = if nan && bits & 0x40 == 0 { 16 } else { 0 };
            assert_eq!(
                (result.value.bits(), result.exceptions.bits()),
                (expected, flags)
            );
        }
    }

    /// Evaluates a nonnegative finite binary16 encoding independently of the implementation.
    fn reference_positive_half(bits: u16) -> f64 {
        let exponent = (bits >> 10) & 31;
        let fraction = bits & 1023;
        if exponent == 0 {
            f64::from(fraction) * 2.0f64.powi(-24)
        } else {
            f64::from(1024 + fraction) * 2.0f64.powi(i32::from(exponent) - 25)
        }
    }

    #[test]
    fn controlled_float_sqrt_matches_every_nonnegative_finite_half_and_rounding_mode() {
        for bits in 0u16..0x7c00 {
            let input = reference_positive_half(bits);
            let ideal = input.sqrt();
            let mut low = 0u16;
            let mut high = 0x7c00u16;
            while high - low > 1 {
                let middle = low + (high - low) / 2;
                if reference_positive_half(middle) <= ideal {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            let lower = reference_positive_half(low);
            let exact = lower * lower == input;
            let upper = reference_positive_half(low + 1);
            let nearer_up = ideal - lower > upper - ideal;
            for mode in [
                FloatRounding::NearestEven,
                FloatRounding::TowardZero,
                FloatRounding::Down,
                FloatRounding::Up,
                FloatRounding::NearestAway,
            ] {
                let increment = !exact
                    && match mode {
                        FloatRounding::NearestEven | FloatRounding::NearestAway => nearer_up,
                        FloatRounding::Up => true,
                        _ => false,
                    };
                let expected = low + u16::from(increment);
                let result = half(bits).sqrt_rounded(mode, true).unwrap();
                assert_eq!(
                    (result.value.bits(), result.exceptions.bits()),
                    (u128::from(expected), u8::from(!exact)),
                    "input={bits:04x}, mode={mode:?}"
                );
            }
        }
    }

    #[test]
    fn controlled_float_sqrt_keeps_wide_precision_and_special_value_flags() {
        for (format, input, lower, nearest, upper) in [
            (FloatFormat::Half, 0x4000u128, 0x3da8, 0x3da8, 0x3da9),
            (FloatFormat::BFloat16, 0x4000, 0x3fb5, 0x3fb5, 0x3fb6),
            (
                FloatFormat::Single,
                0x40000000,
                0x3fb504f3,
                0x3fb504f3,
                0x3fb504f4,
            ),
            (
                FloatFormat::Double,
                0x4000000000000000,
                0x3ff6a09e667f3bcc,
                0x3ff6a09e667f3bcd,
                0x3ff6a09e667f3bcd,
            ),
            // Independently rounded with Python Decimal at 200 decimal digits.
            (
                FloatFormat::Quad,
                0x40000000000000000000000000000000,
                0x3fff6a09e667f3bcc908b2fb1366ea95,
                0x3fff6a09e667f3bcc908b2fb1366ea95,
                0x3fff6a09e667f3bcc908b2fb1366ea96,
            ),
        ] {
            let value = FloatBits::new(format, input).unwrap();
            for (mode, expected) in [
                (FloatRounding::NearestEven, nearest),
                (FloatRounding::NearestAway, nearest),
                (FloatRounding::TowardZero, lower),
                (FloatRounding::Down, lower),
                (FloatRounding::Up, upper),
            ] {
                let result = value.sqrt_rounded(mode, true).unwrap();
                assert_eq!(
                    (result.value.bits(), result.exceptions.bits()),
                    (expected, 1)
                );
            }
        }
        for (input, expected, flags) in [
            (0x8000, 0x8000, 0),
            (0, 0, 0),
            (0x4400, 0x4000, 0),
            (1, 0x0c00, 0),
            (0xbc00, 0x7e00, 16),
            (0xfc00, 0x7e00, 16),
            (0x7c00, 0x7c00, 0),
            (0x7c01, 0x7e00, 16),
            (0xfe55, 0x7e00, 0),
        ] {
            let result = half(input)
                .sqrt_rounded(FloatRounding::NearestEven, true)
                .unwrap();
            assert_eq!(
                (result.value.bits(), result.exceptions.bits()),
                (expected, flags)
            );
        }
        assert!(
            half(0x7e11)
                .sqrt_rounded(FloatRounding::NearestEven, false)
                .is_none()
        );
        assert!(
            half(0x4000)
                .sqrt_rounded(FloatRounding::ToOdd, true)
                .is_none()
        );
    }

    #[test]
    fn controlled_float_overflow_flags_cover_every_format_and_do_not_count_nearby_inexact_results()
    {
        for (format, maximum) in [
            (FloatFormat::Half, 0x7bffu128),
            (FloatFormat::BFloat16, 0x7f7f),
            (FloatFormat::Single, 0x7f7fffff),
            (FloatFormat::Double, 0x7fefffffffffffff),
            (FloatFormat::X87Extended, 0x7ffeffffffffffffffff),
            (FloatFormat::Quad, 0x7ffeffffffffffffffffffffffffffff),
        ] {
            let max = FloatBits::new(format, maximum).unwrap();
            let one = FloatBits::from(1.0f64).exactly(format).unwrap();
            let two = FloatBits::from(2.0f64).exactly(format).unwrap();
            let tiny = FloatBits::from(0.5f64).exactly(format).unwrap();
            let zero = FloatBits::from(0.0f64).exactly(format).unwrap();
            for (negative, rounding) in [
                (false, FloatRounding::TowardZero),
                (false, FloatRounding::Down),
                (true, FloatRounding::TowardZero),
                (true, FloatRounding::Up),
            ] {
                let value = if negative { max.negate() } else { max };
                let same_sign_tiny = if negative { tiny.negate() } else { tiny };
                for result in [
                    value
                        .binary_rounded(value, FloatBinary::Add, rounding, true)
                        .unwrap(),
                    value
                        .binary_rounded(two, FloatBinary::Multiply, rounding, true)
                        .unwrap(),
                    value
                        .binary_rounded(tiny, FloatBinary::Divide, rounding, true)
                        .unwrap(),
                    value.mul_add_rounded(two, zero, rounding, true).unwrap(),
                ] {
                    assert_eq!(result.value, value);
                    assert_eq!(
                        result.exceptions.bits(),
                        5,
                        "format={format:?}, negative={negative}, rounding={rounding:?}"
                    );
                }
                for result in [
                    value
                        .binary_rounded(same_sign_tiny, FloatBinary::Add, rounding, true)
                        .unwrap(),
                    value
                        .mul_add_rounded(one, same_sign_tiny, rounding, true)
                        .unwrap(),
                ] {
                    assert_eq!(result.value, value);
                    assert_eq!(
                        result.exceptions.bits(),
                        1,
                        "near boundary format={format:?}, negative={negative}"
                    );
                }
            }
        }
        for (source, expected_flags) in [(131008.0f64, 5), (65504.5, 1)] {
            for rounding in [
                FloatRounding::TowardZero,
                FloatRounding::Down,
                FloatRounding::ToOdd,
            ] {
                let result = FloatBits::from(source)
                    .convert_rounded(FloatFormat::Half, rounding, true)
                    .unwrap();
                assert_eq!(
                    (result.value.bits(), result.exceptions.bits()),
                    (0x7bff, expected_flags)
                );
            }
        }
    }

    /// **A constant's decimal reads back as its bits.** The oracle is the
    /// parse of the printed text in the same format: a third, a tenth and
    /// the largest finite value of each format, which no shorter decimal of
    /// a narrower format would hold.
    #[test]
    fn a_decimal_reads_back_as_the_same_bits() {
        for format in FloatFormat::ALL {
            let integer =
                |value| FloatBits::new(format, format.from_integer(value)).expect("an integer");
            let third = integer(1)
                .checked_div(integer(3), NEAREST)
                .expect("a third");
            let tenth = integer(1)
                .checked_div(integer(10), NEAREST)
                .expect("a tenth");
            let largest = with_format!(format, F => F::largest().to_bits());
            for value in [
                third,
                tenth,
                FloatBits::new(format, largest).expect("the largest"),
            ] {
                let text = value.decimal();
                let parsed =
                    with_format!(format, F => F::from_str(&text).expect("a decimal").to_bits());
                assert_eq!(parsed, value.bits(), "{format:?} {text}");
            }
        }
    }

    /// **A format's code reads back as the format.** Every code from 0 to
    /// 5 names one format, and 6 names none.
    #[test]
    fn a_format_code_round_trips() {
        for (code, format) in FloatFormat::ALL.into_iter().enumerate() {
            assert_eq!(usize::from(u8::from(format)), code);
            assert_eq!(FloatFormat::try_from(u8::from(format)), Ok(format));
        }
        assert_eq!(FloatFormat::try_from(6), Err(6));
    }

    /// **A conversion is exact only when it converts back.** The oracle is
    /// the formats' encodings: `2.0` is `0x4000` in binary16, `0.1`'s binary64
    /// bits have no binary16 or binary32 equal, and binary32 `0.1` widens to a
    /// binary64 exactly.
    #[test]
    fn a_constant_is_restated_exactly_or_not_at_all() {
        let two = FloatBits::from(2.0f64);
        assert_eq!(two.exactly(FloatFormat::Half), Some(half(0x4000)));
        assert_eq!(FloatBits::from(0.1f64).exactly(FloatFormat::Half), None);
        assert_eq!(FloatBits::from(0.1f64).exactly(FloatFormat::Single), None);
        assert_eq!(
            FloatBits::from(0.1f32).exactly(FloatFormat::Double),
            Some(FloatBits::from(f64::from(0.1f32)))
        );
        assert_eq!(FloatBits::from(f64::NAN).exactly(FloatFormat::Single), None);
    }

    fn half(bits: u16) -> FloatBits {
        FloatBits::new(FloatFormat::Half, u128::from(bits)).expect("a half")
    }

    fn bfloat(bits: u16) -> FloatBits {
        FloatBits::new(FloatFormat::BFloat16, u128::from(bits)).expect("a bfloat16")
    }

    fn extended(bits: u128) -> FloatBits {
        FloatBits::new(FloatFormat::X87Extended, bits).expect("an x87 extended")
    }

    fn quad(bits: u128) -> FloatBits {
        FloatBits::new(FloatFormat::Quad, bits).expect("a binary128")
    }

    /// Arithmetic in each format is IEEE's, rounded to nearest even: the
    /// expected bits are the formats' encodings of the exact or rounded
    /// results (`1/3` is `0x3555` in binary16, `0x3fd...aaab` in x87 extended).
    #[test]
    fn arithmetic_is_ieee_in_each_format() {
        // 1.5 + 2.25 = 3.75, exact in binary16.
        assert_eq!(
            half(0x3e00).checked_add(half(0x4080), NEAREST),
            Some(half(0x4380))
        );
        // 1 / 3 rounds to 0x3555.
        assert_eq!(
            half(0x3c00).checked_div(half(0x4200), NEAREST),
            Some(half(0x3555))
        );
        // bfloat16 1.0 + 2^-8 ties to even: 1.0.
        assert_eq!(
            bfloat(0x3f80).checked_add(bfloat(0x3b80), NEAREST),
            Some(bfloat(0x3f80))
        );
        // x87: 1 + 1 = 2, and 1 / 3 with its explicit integer bit.
        let one = 0x3fff_8000_0000_0000_0000;
        assert_eq!(
            extended(one).checked_add(extended(one), NEAREST),
            Some(extended(0x4000_8000_0000_0000_0000))
        );
        let three = 0x4000_c000_0000_0000_0000;
        assert_eq!(
            extended(one).checked_div(extended(three), NEAREST),
            Some(extended(0x3ffd_aaaa_aaaa_aaaa_aaab))
        );
        // binary128 1 / 3.
        let quad_one = 0x3fff_0000_0000_0000_0000_0000_0000_0000;
        let quad_three = 0x4000_8000_0000_0000_0000_0000_0000_0000;
        assert_eq!(
            quad(quad_one).checked_div(quad(quad_three), NEAREST),
            Some(quad(0x3ffd_5555_5555_5555_5555_5555_5555_5555))
        );
        // fmod keeps the dividend's sign: -5.5 rem 2 is -1.5.
        assert_eq!(half(0xc580).checked_rem(half(0x4000)), Some(half(0xbe00)));
        // Formats do not mix.
        assert_eq!(half(0x3c00).checked_add(bfloat(0x3f80), NEAREST), None);
    }

    /// **The program's mode folds only what every mode rounds alike.** An
    /// exact result folds; an inexact one does not; nor does an exact zero
    /// sum of opposite signs, `-0` rounding down and `+0` otherwise, nor an
    /// overflow, infinite to nearest and the largest finite value toward zero.
    /// A sum in any order folds when its every order gives one value in each
    /// mode, a zero partial sum's sign included.
    #[test]
    fn the_programs_mode_folds_only_what_every_mode_rounds_alike() {
        let (one, minus_one, three) = (
            FloatBits::from(1.0f64),
            FloatBits::from(-1.0f64),
            FloatBits::from(3.0f64),
        );
        assert_eq!(three.checked_sub(one, None), Some(FloatBits::from(2.0f64)));
        assert_eq!(one.checked_div(three, None), None);
        assert_eq!(
            one.checked_mul_add(three, minus_one, None),
            Some(FloatBits::from(2.0f64))
        );
        assert_eq!(one.checked_add(minus_one, None), None);
        assert_eq!(
            one.checked_add(minus_one, Some(FloatRounding::Down)),
            Some(FloatBits::from(-0.0f64))
        );
        assert_eq!(
            one.checked_add(minus_one, NEAREST),
            Some(FloatBits::from(0.0f64))
        );
        let largest = FloatBits::from(f64::MAX);
        assert_eq!(largest.checked_add(largest, None), None);
        assert_eq!(
            largest.checked_add(largest, Some(FloatRounding::TowardZero)),
            Some(largest)
        );
        assert_eq!(
            FloatBits::sum_in_any_order(&[one, minus_one, three], None),
            Some(three)
        );
        assert_eq!(FloatRounding::decide(None, |_| Some(1)), Some(1));
        assert_eq!(
            FloatRounding::decide(None, |rounding| (rounding != FloatRounding::Up)
                .then_some(1)),
            None,
            "a mode that gives nothing decides nothing"
        );
    }

    /// A NaN is not evaluated, its payload being the architecture's choice,
    /// but its sign is a bit like any other.
    #[test]
    fn a_nan_is_not_evaluated_but_its_sign_moves() {
        let nan = half(0x7e00);
        assert!(nan.is_nan());
        assert_eq!(nan.checked_add(half(0x3c00), NEAREST), None);
        // inf - inf is a NaN.
        assert_eq!(half(0x7c00).checked_sub(half(0x7c00), NEAREST), None);
        assert_eq!(nan.trunc(), None);
        assert_eq!(nan.negate(), half(0xfe00));
        assert_eq!(half(0xfe00).abs(), nan);
        assert_eq!(nan.compare(nan), None);
    }

    /// Every scalar width preserves exact unit values across all six formats.
    #[test]
    fn integer_conversions_support_every_scalar_width_and_float_format() {
        for (format, one) in [
            (FloatFormat::Half, 0x3c00u128),
            (FloatFormat::BFloat16, 0x3f80),
            (FloatFormat::Single, 0x3f80_0000),
            (FloatFormat::Double, 0x3ff0_0000_0000_0000),
            (FloatFormat::X87Extended, 0x3fff_8000_0000_0000_0000),
            (FloatFormat::Quad, 0x3fff_0000_0000_0000_0000_0000_0000_0000),
        ] {
            for width in 1..=128 {
                let mask = u128::MAX >> (128 - width);
                for signed in [false, true] {
                    for negative in [false, true] {
                        let bits = one
                            | if negative {
                                1u128 << (format.bits() - 1)
                            } else {
                                0
                            };
                        let converted = FloatBits::new(format, bits)
                            .unwrap()
                            .to_integer_clamped(width, signed, false, FloatRounding::TowardZero)
                            .unwrap();
                        let invalid = (negative && !signed) || (!negative && signed && width == 1);
                        let expected = if invalid {
                            0
                        } else if negative {
                            mask
                        } else {
                            1
                        };
                        assert_eq!(
                            converted.bits, expected,
                            "{format:?}/{width}/{signed}/{negative}"
                        );
                        assert_eq!(
                            converted.exceptions.contains(FloatExceptions::INVALID),
                            invalid
                        );
                    }
                    let integer = FloatBits::from_integer_rounded(
                        format,
                        1,
                        width,
                        signed,
                        FloatRounding::NearestEven,
                    )
                    .unwrap();
                    let expected = one
                        | if signed && width == 1 {
                            1u128 << (format.bits() - 1)
                        } else {
                            0
                        };
                    assert_eq!(integer.value.bits(), expected);
                    assert_eq!(integer.exceptions.bits(), 0);
                }
            }
        }
        let fractional = FloatBits::from(-0.75f64)
            .to_integer_clamped(7, false, false, FloatRounding::TowardZero)
            .unwrap();
        assert_eq!(fractional.bits, 0);
        assert!(!fractional.exceptions.contains(FloatExceptions::INVALID));
    }

    /// Rounding to an integral value goes the way each method names.
    #[test]
    fn rounding_to_integral_follows_its_direction() {
        // binary16 2.5 and -2.5.
        let (up, down) = (half(0x4100), half(0xc100));
        assert_eq!(up.trunc(), Some(half(0x4000)));
        assert_eq!(down.floor(), Some(half(0xc200)));
        assert_eq!(up.ceil(), Some(half(0x4200)));
        assert_eq!(up.round_ties_even(), Some(half(0x4000)));
        assert_eq!(up.round_ties_away(), Some(half(0x4200)));
        // -0.5 truncates to -0.0.
        assert_eq!(half(0xb800).trunc(), Some(half(0x8000)));
    }

    /// Conversions between formats round to nearest even and carry every
    /// value the narrower format holds exactly.
    #[test]
    fn conversions_between_formats_round_to_nearest_even() {
        let third = u128::from((1.0f64 / 3.0).to_bits());
        assert_eq!(
            FloatFormat::Double.convert(third, FloatFormat::Half),
            Some(0x3555)
        );
        assert_eq!(
            FloatFormat::Half.convert(0x3e00, FloatFormat::Double),
            Some(u128::from(1.5f64.to_bits()))
        );
        // binary32 to bfloat16 ties to even: 0x3f808000 stays 0x3f80, and
        // 0x3f818000 rounds up to 0x3f82.
        assert_eq!(
            FloatFormat::Single.convert(0x3f80_8000, FloatFormat::BFloat16),
            Some(0x3f80)
        );
        assert_eq!(
            FloatFormat::Single.convert(0x3f81_8000, FloatFormat::BFloat16),
            Some(0x3f82)
        );
        assert_eq!(
            FloatFormat::Single.convert(u128::from(3.0f32.to_bits()), FloatFormat::X87Extended),
            Some(0x4000_c000_0000_0000_0000)
        );
        // 65520, halfway past binary16's largest finite value, rounds to infinity.
        assert_eq!(
            FloatFormat::Single.convert(u128::from(65520.0f32.to_bits()), FloatFormat::Half),
            Some(0x7c00)
        );
        assert_eq!(
            FloatFormat::Half.convert(0x7e00, FloatFormat::Single),
            None,
            "a NaN"
        );
        // An integer rounds to nearest even: 2049 is 2048 in binary16, 70000
        // past its range, and 2^64 exact in x87 extended.
        assert_eq!(FloatFormat::Half.from_integer(2049), 0x6800);
        assert_eq!(FloatFormat::Half.from_integer(70_000), 0x7c00);
        assert_eq!(
            FloatFormat::X87Extended.from_integer(1 << 64),
            0x403f_8000_0000_0000_0000
        );
        assert_eq!(
            FloatFormat::Single.from_integer(-3),
            u128::from((-3.0f32).to_bits())
        );
        assert_eq!(
            FloatFormat::Half.convert(0x1_0000, FloatFormat::Single),
            None,
            "too wide"
        );
    }

    /// Constants order as their values; display names the format.
    #[test]
    fn constants_compare_and_display_by_format() {
        assert_eq!(half(0x3c00).compare(half(0x4000)), Some(Ordering::Less));
        assert_eq!(
            half(0x8000).compare(half(0x0000)),
            Some(Ordering::Equal),
            "-0 == +0"
        );
        assert_ne!(half(0x8000), half(0x0000), "but distinct constants");
        assert_eq!(half(0x3e00).to_string(), "1.5f16");
        assert_eq!(bfloat(0x3fc0).to_string(), "1.5bf16");
        assert_eq!(
            FloatBits::new(FloatFormat::Single, 0x3fc0_0000),
            Some(FloatBits::from(1.5f32)),
            "a binary32 is a constant of its format too"
        );
        assert_eq!(FloatBits::from(1.5f32).to_string(), "1.5f");
        assert_eq!(FloatBits::from(1.5f64).to_string(), "1.5");
        assert_eq!(
            FloatBits::new(FloatFormat::Half, 0x1_0000),
            None,
            "too wide"
        );
    }
    /// Every order of adding `values` in host `f32` arithmetic, which is
    /// IEEE's round-to-nearest-even: the permutations, as bits.
    fn every_order_sums(values: &[f32]) -> Vec<u32> {
        fn orders(values: &[f32], chosen: &mut Vec<f32>, used: &mut Vec<bool>, out: &mut Vec<u32>) {
            if chosen.len() == values.len() {
                let mut sum = chosen[0];
                for value in &chosen[1..] {
                    sum += *value;
                }
                out.push(sum.to_bits());
                return;
            }
            for index in 0..values.len() {
                if !used[index] {
                    used[index] = true;
                    chosen.push(values[index]);
                    orders(values, chosen, used, out);
                    chosen.pop();
                    used[index] = false;
                }
            }
        }
        let mut out = Vec::new();
        orders(
            values,
            &mut Vec::new(),
            &mut vec![false; values.len()],
            &mut out,
        );
        out
    }

    /// **A sum folds only to what every order of adding gives.** The oracle
    /// is the host's own `f32` sums in every order, over every multiset of
    /// three and four values from a pool of signed zeros, small exact values,
    /// values a neighbour rounds away, the largest finite and the least
    /// subnormal magnitudes, and the infinities: where the fold states a sum
    /// every order gives that sum, bit for bit, and where every order agrees
    /// on a sum of values that are all small integers it states one.
    #[test]
    fn a_sum_folds_only_to_what_every_order_gives() {
        let pool = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            1.5,
            -0.25,
            3.0,
            16_777_216.0,
            -16_777_216.0,
            f32::MAX,
            -f32::MAX,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            2.0f32.powi(127),
            -(2.0f32.powi(127)),
        ];
        let mut folded = 0usize;
        for size in [3usize, 4] {
            let mut picks = vec![0usize; size];
            loop {
                let values: Vec<f32> = picks.iter().map(|index| pool[*index]).collect();
                let constants: Vec<FloatBits> =
                    values.iter().copied().map(FloatBits::from).collect();
                let sums = every_order_sums(&values);
                match FloatBits::sum_in_any_order(&constants, NEAREST) {
                    Some(sum) => {
                        folded += 1;
                        assert!(
                            sums.iter().all(|order| u128::from(*order) == sum.bits()),
                            "{values:?} folded to {sum} where the orders give {sums:x?}"
                        );
                    }
                    None => {
                        let small = values
                            .iter()
                            .all(|value| value.fract() == 0.0 && value.abs() <= 3.0);
                        assert!(
                            !small,
                            "{values:?}: small integers add exactly in any order"
                        );
                    }
                }
                // The next multiset, its picks non-decreasing.
                let Some(position) = (0..size)
                    .rev()
                    .find(|position| picks[*position] + 1 < pool.len())
                else {
                    break;
                };
                let next = picks[position] + 1;
                for pick in &mut picks[position..] {
                    *pick = next;
                }
            }
        }
        assert!(folded > 100, "the fold states a sum for {folded} multisets");
        let two = [f32::INFINITY, 1.0].map(FloatBits::from);
        assert_eq!(
            FloatBits::sum_in_any_order(&two, NEAREST),
            Some(FloatBits::from(f32::INFINITY)),
            "two values add the same either way round"
        );
        let overflowing =
            [2.0f32.powi(127), 2.0f32.powi(127), -(2.0f32.powi(127))].map(FloatBits::from);
        assert_eq!(
            FloatBits::sum_in_any_order(&overflowing, NEAREST),
            None,
            "`(a + b) - c` overflows where `a + (b - c)` does not"
        );
        assert_eq!(
            FloatBits::sum_in_any_order(
                &[FloatBits::from(1.0f32), FloatBits::from(1.0f64)],
                NEAREST
            ),
            None,
            "two formats"
        );
    }
    #[test]
    fn rounded_reductions_preserve_every_empty_seed_bit_and_exception_state() {
        for format in [
            FloatFormat::Half,
            FloatFormat::BFloat16,
            FloatFormat::Single,
            FloatFormat::Double,
            FloatFormat::Quad,
        ] {
            for raw in [
                0,
                1,
                format.canonical_nan_bits(),
                format.canonical_nan_bits() ^ 1,
            ] {
                let seed = FloatBits::new(format, raw).unwrap();
                for mode in 0..5 {
                    for operation in [
                        FloatBinary::Add,
                        FloatBinary::MinimumNumber,
                        FloatBinary::MaximumNumber,
                    ] {
                        let result = seed
                            .reduce_rounded(
                                operation,
                                &[],
                                FloatRounding::of_environment(mode).unwrap(),
                                true,
                            )
                            .unwrap();
                        assert_eq!(result.value.bits(), raw);
                        assert_eq!(result.exceptions.bits(), 0);
                    }
                }
            }
        }
        let signaling = FloatBits::new(FloatFormat::Single, 0x7f800001).unwrap();
        assert_eq!(
            signaling
                .reduce_rounded(FloatBinary::Add, &[], FloatRounding::NearestEven, true)
                .unwrap()
                .value,
            signaling
        );
    }

    #[test]
    fn rounded_reductions_promote_before_rounding_and_accrue_only_selected_inputs() {
        let seed = FloatBits::new(FloatFormat::Single, 0x3f800000).unwrap();
        let half = FloatBits::new(FloatFormat::Half, 1).unwrap();
        for mode in 0..5 {
            let result = seed
                .reduce_rounded(
                    FloatBinary::Add,
                    &[half],
                    FloatRounding::of_environment(mode).unwrap(),
                    true,
                )
                .unwrap();
            assert_eq!(
                result.value.bits(),
                if mode == 3 || mode == 4 {
                    0x3f800001
                } else {
                    0x3f800000
                }
            );
            assert_eq!(result.exceptions.bits(), 1);
        }
        let negative = FloatBits::new(FloatFormat::Half, 0x8001).unwrap();
        let serial = seed
            .reduce_rounded(
                FloatBinary::Add,
                &[half, negative],
                FloatRounding::NearestEven,
                true,
            )
            .unwrap();
        assert_eq!(serial.value.bits(), 0x3f7fffff);
        assert_eq!(serial.exceptions.bits(), 1);
        assert!(
            half.reduce_rounded(FloatBinary::Add, &[seed], FloatRounding::NearestEven, true)
                .is_none()
        );
        assert!(
            seed.reduce_rounded(FloatBinary::Add, &[], FloatRounding::ToOdd, true)
                .is_none()
        );
    }

    #[test]
    fn rounded_reductions_quiet_only_computed_nans_and_keep_signed_zero_extrema() {
        let one = FloatBits::new(FloatFormat::Half, 0x3c00).unwrap();
        for (bits, flags) in [(0x7fc00001, 0), (0x7f800001, 16)] {
            let seed = FloatBits::new(FloatFormat::Single, bits).unwrap();
            for operation in [FloatBinary::MinimumNumber, FloatBinary::MaximumNumber] {
                let result = seed
                    .reduce_rounded(operation, &[one], FloatRounding::NearestEven, true)
                    .unwrap();
                assert_eq!(result.value.bits(), 0x3f800000);
                assert_eq!(result.exceptions.bits(), flags);
            }
        }
        for (seed, value) in [(0, 0x80000000), (0x80000000, 0)] {
            let seed = FloatBits::new(FloatFormat::Single, seed).unwrap();
            let value = FloatBits::new(FloatFormat::Single, value).unwrap();
            assert_eq!(
                seed.reduce_rounded(
                    FloatBinary::MinimumNumber,
                    &[value],
                    FloatRounding::NearestEven,
                    true
                )
                .unwrap()
                .value
                .bits(),
                0x80000000
            );
            assert_eq!(
                seed.reduce_rounded(
                    FloatBinary::MaximumNumber,
                    &[value],
                    FloatRounding::NearestEven,
                    true
                )
                .unwrap()
                .value
                .bits(),
                0
            );
        }
    }

    /// **A class is stated in its architecture's encoding.** Each class of a
    /// single (negative infinity, normal, subnormal and zero, positive zero,
    /// subnormal, normal and infinity, signaling and quiet NaN) is its bit in
    /// RISC-V's `fclass` order and in MIPS's `class.fmt` order, and x87's
    /// condition codes: `C0` a NaN, `C2` a normal, both an infinity, `C3` a
    /// zero, `C3` and `C2` a denormal. Of the extended format, an unnormal (an
    /// exponent without the integer bit) is unsupported, no code, and a
    /// pseudo-denormal (the integer bit with no exponent) a denormal.
    #[test]
    fn a_class_is_stated_in_its_architectures_encoding() {
        let single = |bits: u32| FloatBits::new(FloatFormat::Single, u128::from(bits)).unwrap();
        let (c0, c2, c3) = (
            u64::from(FlagsMask::CARRY.bits()),
            u64::from(FlagsMask::PARITY.bits()),
            u64::from(FlagsMask::ZERO.bits()),
        );
        // Each value, its RISC-V bit, its MIPS bit and its x87 codes.
        let classes = [
            (0xff80_0000, 0, 2, c2 | c0),
            (0xbf80_0000, 1, 3, c2),
            (0x8000_0001, 2, 4, c3 | c2),
            (0x8000_0000, 3, 5, c3),
            (0x0000_0000, 4, 9, c3),
            (0x0000_0001, 5, 8, c3 | c2),
            (0x3f80_0000, 6, 7, c2),
            (0x7f80_0000, 7, 6, c2 | c0),
            (0x7f80_0001, 8, 0, c0),
            (0x7fc0_0000, 9, 1, c0),
        ];
        for (bits, riscv, mips, x87) in classes {
            let value = single(bits);
            assert_eq!(
                FloatClassLayout::RiscV.class(value),
                1 << riscv,
                "{bits:#x}"
            );
            assert_eq!(FloatClassLayout::Mips.class(value), 1 << mips, "{bits:#x}");
            assert_eq!(FloatClassLayout::X87.class(value), x87, "{bits:#x}");
        }
        let extended = |bits: u128| FloatBits::new(FloatFormat::X87Extended, bits).unwrap();
        // One, its integer bit set; the same exponent without it; and the
        // integer bit with a zero exponent.
        assert_eq!(
            FloatClassLayout::X87.class(extended(0x3fff_8000_0000_0000_0000)),
            c2
        );
        assert_eq!(
            FloatClassLayout::X87.class(extended(0x3fff_0000_0000_0000_0000)),
            0
        );
        assert_eq!(
            FloatClassLayout::X87.class(extended(0x0000_8000_0000_0000_0001)),
            c3 | c2
        );
    }
}
