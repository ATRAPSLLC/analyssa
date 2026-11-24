//! Native machine-state modeling - block-string streams, wide compare-exchange,
//! and condition-flag semantics - plus the structural binary / unary operation
//! views used by generic passes.
//!
//! State an operation touches is named by the operation: registers it writes
//! are its `outputs`, flags it defines are the flags value it produces, and
//! anything genuinely unmodelled is declared through
//! [`SsaEffectKind::Opaque`], which is what
//! actually constrains motion and elimination.
//!
//! [`SsaEffectKind::Opaque`]: crate::ir::ops::SsaEffectKind::Opaque

use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{
        content::ContentSink,
        ops::kinds::{AtomicRmwOp, CmpKind},
        variable::SsaVarId,
    },
    target::Target,
};

impl fmt::Display for AtomicRmwOp {
    /// Writes the lowercase mnemonic for this atomic read-modify-write kind.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xchg => write!(f, "xchg"),
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Xor => write!(f, "xor"),
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
            Self::AndNot => write!(f, "andnot"),
            Self::MinU => write!(f, "minu"),
            Self::MaxU => write!(f, "maxu"),
            Self::Neg => write!(f, "neg"),
            Self::FAdd => write!(f, "fadd"),
            Self::FMax => write!(f, "fmax"),
            Self::FMin => write!(f, "fmin"),
            Self::FMaxNum => write!(f, "fmaxnm"),
            Self::FMinNum => write!(f, "fminnm"),
        }
    }
}

/// Bitmask for selecting flag bits from a flags-defining operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FlagsMask(u16);

impl FlagsMask {
    /// Carry flag bit.
    pub const CARRY: Self = Self(1 << 0);
    /// Parity flag bit.
    pub const PARITY: Self = Self(1 << 1);
    /// Auxiliary carry / adjust flag bit.
    pub const ADJUST: Self = Self(1 << 2);
    /// Zero flag bit.
    pub const ZERO: Self = Self(1 << 3);
    /// Sign flag bit.
    pub const SIGN: Self = Self(1 << 4);
    /// Overflow flag bit.
    pub const OVERFLOW: Self = Self(1 << 5);
    /// String-direction flag bit (x86 `DF`). A control flag, not a status flag:
    /// no comparison produces it and no condition code reads it, so it is
    /// deliberately outside [`Self::x86_status`].
    pub const DIRECTION: Self = Self(1 << 6);
    /// Interrupt-enable flag bit (x86 `IF`).
    pub const INTERRUPT: Self = Self(1 << 7);
    /// Alignment-check / access-control flag bit (x86 `AC`).
    pub const ALIGN_CHECK: Self = Self(1 << 8);

    /// Every defined flag bit paired with its display name, in bit order.
    ///
    /// The single table [`Self::ALL`], [`Self::is_valid`] and the
    /// [`Display`](fmt::Display) impl are all derived from. A bit added here
    /// becomes valid, printable and round-trippable in one edit; a bit added as
    /// a bare constant and not listed here stays *undefined*, which the
    /// verifier and the block builder both reject - so the table is the
    /// definition of the mask, not a copy of it.
    pub const NAMED: &'static [(Self, &'static str)] = &[
        (Self::CARRY, "CF"),
        (Self::PARITY, "PF"),
        (Self::ADJUST, "AF"),
        (Self::ZERO, "ZF"),
        (Self::SIGN, "SF"),
        (Self::OVERFLOW, "OF"),
        (Self::DIRECTION, "DF"),
        (Self::INTERRUPT, "IF"),
        (Self::ALIGN_CHECK, "AC"),
    ];

    /// The union of every defined flag bit.
    pub const ALL: Self = Self::union_of(Self::NAMED);

    /// Folds a slice of named bits into their union.
    ///
    /// Written as a recursion over slice patterns because a `const` context has
    /// no iterators, and indexing is denied crate-wide.
    const fn union_of(named: &[(Self, &'static str)]) -> Self {
        match named {
            [] => Self(0),
            [(first, _), rest @ ..] => Self(first.0 | Self::union_of(rest).0),
        }
    }

    /// Creates a flag mask from raw bits, defined or not.
    ///
    /// Deliberately unmasked: a mask read back from a persisted blob must equal
    /// the one written, so undefined bits survive to be *reported* rather than
    /// being silently dropped here. [`Self::from_bits_checked`] is the
    /// validating constructor, and [`Self::is_valid`] the predicate.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Creates a flag mask from raw bits, rejecting undefined ones.
    ///
    /// `None` when `bits` names a flag this crate does not model - which is a
    /// diagnosable input error, not a mask with fewer bits.
    #[must_use]
    pub const fn from_bits_checked(bits: u16) -> Option<Self> {
        let mask = Self(bits);
        if mask.is_valid() { Some(mask) } else { None }
    }

    /// Returns `true` when every bit set in this mask is a defined flag.
    ///
    /// The empty mask is valid: selecting no flags is a legal thing to ask for.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 & !Self::ALL.0 == 0
    }

    /// Returns the bits set in this mask that name no defined flag.
    #[must_use]
    pub const fn undefined_bits(self) -> u16 {
        self.0 & !Self::ALL.0
    }
    /// Returns the raw flag bits.
    pub const fn bits(self) -> u16 {
        self.0
    }
    /// Returns `true` when the mask selects no flags.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` when all bits in `other` are selected by this mask.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns `true` when this mask and `other` select at least one bit in
    /// common.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Returns the union of two flag masks.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the bits both masks select.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns the bits this mask selects and `other` does not.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Returns `true` when the mask selects exactly one flag.
    #[must_use]
    pub const fn is_single_bit(self) -> bool {
        self.0.count_ones() == 1
    }

    /// Returns the x86/x64 arithmetic status flags.
    pub const fn x86_status() -> Self {
        Self(
            Self::CARRY.0
                | Self::PARITY.0
                | Self::ADJUST.0
                | Self::ZERO.0
                | Self::SIGN.0
                | Self::OVERFLOW.0,
        )
    }

    /// Returns the mask bit for a target-independent flag bit.
    pub const fn from_flag_bit(bit: NativeFlagBit) -> Self {
        match bit {
            NativeFlagBit::Carry => Self::CARRY,
            NativeFlagBit::Parity => Self::PARITY,
            NativeFlagBit::Adjust => Self::ADJUST,
            NativeFlagBit::Zero => Self::ZERO,
            NativeFlagBit::Sign => Self::SIGN,
            NativeFlagBit::Overflow => Self::OVERFLOW,
            NativeFlagBit::Direction => Self::DIRECTION,
            NativeFlagBit::Interrupt => Self::INTERRUPT,
            NativeFlagBit::AlignCheck => Self::ALIGN_CHECK,
        }
    }
}

impl fmt::Display for FlagsMask {
    /// Renders the selected flags as `CF,ZF`, and `none` when nothing is
    /// selected.
    ///
    /// Bits outside [`FlagsMask::NAMED`] are appended as `?0x…` rather than
    /// dropped, so an invalid mask cannot render as `none` or as a truncated
    /// but plausible-looking list - the two spellings that would make a
    /// malformed mask read as a well-formed one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (bit, name) in Self::NAMED {
            if self.intersects(*bit) {
                if !first {
                    write!(f, ",")?;
                }
                write!(f, "{name}")?;
                first = false;
            }
        }
        let undefined = self.undefined_bits();
        if undefined != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "?0x{undefined:x}")?;
            first = false;
        }
        if first {
            write!(f, "none")?;
        }
        Ok(())
    }
}

/// Target-independent status flag bit used by native flag semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NativeFlagBit {
    /// Carry or borrow flag, such as x86 `CF` or AArch64 `C`.
    Carry,
    /// Parity flag, such as x86 `PF`.
    Parity,
    /// Auxiliary carry or adjust flag, such as x86 `AF`.
    Adjust,
    /// Zero flag, such as x86 `ZF` or AArch64 `Z`.
    Zero,
    /// Sign or negative flag, such as x86 `SF` or AArch64 `N`.
    Sign,
    /// Signed overflow flag, such as x86 `OF` or AArch64 `V`.
    Overflow,
    /// String-direction control flag (x86 `DF`) - selects whether the string
    /// operations advance or retreat. Not a status flag.
    Direction,
    /// Interrupt-enable control flag (x86 `IF`).
    Interrupt,
    /// Alignment-check / access-control flag (x86 `AC`).
    AlignCheck,
}

/// Describes how an instruction writes one native status flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FlagWriteState {
    /// The flag receives a defined value from the instruction semantics.
    Defined,
    /// The flag receives an architecturally undefined value.
    Undefined,
    /// The flag keeps its prior value.
    Preserved,
    /// The flag is architecturally cleared to zero.
    Cleared,
}

/// One native flag write performed by an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FlagWrite {
    /// Flag bit affected by the instruction.
    pub bit: NativeFlagBit,
    /// Write behavior for the flag bit.
    pub state: FlagWriteState,
}

impl FlagWrite {
    /// Creates a native flag write descriptor.
    #[must_use]
    pub const fn new(bit: NativeFlagBit, state: FlagWriteState) -> Self {
        Self { bit, state }
    }

    /// Creates a descriptor for a defined flag write.
    #[must_use]
    pub const fn defined(bit: NativeFlagBit) -> Self {
        Self::new(bit, FlagWriteState::Defined)
    }

    /// Creates a descriptor for an undefined flag write.
    #[must_use]
    pub const fn undefined(bit: NativeFlagBit) -> Self {
        Self::new(bit, FlagWriteState::Undefined)
    }

    /// Creates a descriptor for a preserved flag.
    #[must_use]
    pub const fn preserved(bit: NativeFlagBit) -> Self {
        Self::new(bit, FlagWriteState::Preserved)
    }

    /// Creates a descriptor for a cleared flag.
    #[must_use]
    pub const fn cleared(bit: NativeFlagBit) -> Self {
        Self::new(bit, FlagWriteState::Cleared)
    }
}

const X86_STATUS_DEFINED: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Carry),
    FlagWrite::defined(NativeFlagBit::Parity),
    FlagWrite::defined(NativeFlagBit::Adjust),
    FlagWrite::defined(NativeFlagBit::Zero),
    FlagWrite::defined(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Overflow),
];

const X86_LOGICAL_WRITES: &[FlagWrite] = &[
    FlagWrite::cleared(NativeFlagBit::Carry),
    FlagWrite::defined(NativeFlagBit::Parity),
    FlagWrite::undefined(NativeFlagBit::Adjust),
    FlagWrite::defined(NativeFlagBit::Zero),
    FlagWrite::defined(NativeFlagBit::Sign),
    FlagWrite::cleared(NativeFlagBit::Overflow),
];

const X86_MUL_WRITES: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Carry),
    FlagWrite::undefined(NativeFlagBit::Parity),
    FlagWrite::undefined(NativeFlagBit::Adjust),
    FlagWrite::undefined(NativeFlagBit::Zero),
    FlagWrite::undefined(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Overflow),
];

const X86_ROTATE_WRITES: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Carry),
    FlagWrite::preserved(NativeFlagBit::Parity),
    FlagWrite::preserved(NativeFlagBit::Adjust),
    FlagWrite::preserved(NativeFlagBit::Zero),
    FlagWrite::preserved(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Overflow),
];

const AARCH64_NZCV_DEFINED: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Zero),
    FlagWrite::defined(NativeFlagBit::Carry),
    FlagWrite::defined(NativeFlagBit::Overflow),
];

const AARCH64_LOGICAL_WRITES: &[FlagWrite] = &[
    FlagWrite::defined(NativeFlagBit::Sign),
    FlagWrite::defined(NativeFlagBit::Zero),
    FlagWrite::cleared(NativeFlagBit::Carry),
    FlagWrite::cleared(NativeFlagBit::Overflow),
];

/// Canonical native flag producer semantics for common instruction families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FlagProducerSemantics {
    /// x86/x64 add, adc, sub, sbb, cmp, inc, and dec style arithmetic flags.
    X86Arithmetic,
    /// x86/x64 and, or, xor, and test style logical flags.
    X86Logical,
    /// x86/x64 mul and imul style implicit-width multiply flags.
    X86Multiply,
    /// x86/x64 shift flags.
    X86Shift,
    /// x86/x64 rotate flags.
    X86Rotate,
    /// AArch64 adds, subs, cmp, and cmn style `NZCV` arithmetic flags.
    AArch64Arithmetic,
    /// AArch64 logical instructions that update `NZCV`, such as `ANDS`.
    AArch64Logical,
}

impl FlagProducerSemantics {
    /// Returns the flag writes performed by this native flag producer.
    #[must_use]
    pub const fn writes(self) -> &'static [FlagWrite] {
        match self {
            Self::X86Arithmetic | Self::X86Shift => X86_STATUS_DEFINED,
            Self::X86Logical => X86_LOGICAL_WRITES,
            Self::X86Multiply => X86_MUL_WRITES,
            Self::X86Rotate => X86_ROTATE_WRITES,
            Self::AArch64Arithmetic => AARCH64_NZCV_DEFINED,
            Self::AArch64Logical => AARCH64_LOGICAL_WRITES,
        }
    }

    /// Returns the set of flags whose value is defined after this producer.
    ///
    /// Both `Defined` and `Cleared` writes count: a cleared flag holds a known
    /// value (zero), so a later condition-code read can rely on it exactly as
    /// on a computed one. `Undefined` writes (no reliable value) and
    /// `Preserved` flags (untouched by this producer, so not defined *by it*)
    /// are excluded.
    #[must_use]
    pub fn defined_mask(self) -> FlagsMask {
        let mut mask = FlagsMask::from_bits(0);
        for write in self.writes() {
            if matches!(
                write.state,
                FlagWriteState::Defined | FlagWriteState::Cleared
            ) {
                mask = mask.union(FlagsMask::from_flag_bit(write.bit));
            }
        }
        mask
    }
}

/// Condition code for flag-based branch operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FlagCondition {
    /// Tests whether carry is set.
    Carry,
    /// Tests whether carry is clear.
    NotCarry,
    /// Tests whether zero is set.
    Zero,
    /// Tests whether zero is clear.
    NotZero,
    /// Tests whether overflow is set.
    Overflow,
    /// Tests whether overflow is clear.
    NotOverflow,
    /// Tests whether sign is set.
    Negative,
    /// Tests whether sign is clear.
    Positive,
    /// Tests whether parity is even.
    ParityEven,
    /// Tests whether parity is odd.
    ParityOdd,
}

impl FlagCondition {
    /// Returns the condition that holds exactly when this one does not.
    ///
    /// Every variant is one half of a pair testing a single flag bit set or
    /// clear, so the complement is total and exact - and exact *whatever set the
    /// flags*, because a bit is either set or it is not. That is what separates
    /// this from [`CmpKind::negated`](crate::ir::ops::kinds::CmpKind::negated),
    /// whose ordered relations stop partitioning once a NaN is involved: an
    /// unordered float compare still leaves `ZF` in one definite state, and
    /// `Zero`/`NotZero` still partition on it.
    #[must_use]
    pub const fn negated(self) -> Self {
        match self {
            Self::Carry => Self::NotCarry,
            Self::NotCarry => Self::Carry,
            Self::Zero => Self::NotZero,
            Self::NotZero => Self::Zero,
            Self::Overflow => Self::NotOverflow,
            Self::NotOverflow => Self::Overflow,
            Self::Negative => Self::Positive,
            Self::Positive => Self::Negative,
            Self::ParityEven => Self::ParityOdd,
            Self::ParityOdd => Self::ParityEven,
        }
    }

    /// The relation to zero this condition states about the value the flags
    /// describe, when it states one.
    ///
    /// Flags describe a value - the result of the arithmetic that set them -
    /// and four of the conditions say which side of zero that value is on. The
    /// pair is `(comparison, whether the value is the comparison's left
    /// operand)`, which is what decides the direction of an asymmetric
    /// relation.
    ///
    /// `None` for the conditions that say nothing an interval can hold: carry,
    /// overflow and parity are facts about how the arithmetic *ran*, not about
    /// the magnitude of what it produced.
    ///
    /// # Returns
    ///
    /// `Some((cmp, value_is_left))` for the sign and zero tests, `None`
    /// otherwise.
    #[must_use]
    pub const fn as_relation_to_zero(self) -> Option<(CmpKind, bool)> {
        match self {
            Self::Zero => Some((CmpKind::Eq, true)),
            Self::NotZero => Some((CmpKind::Ne, true)),
            Self::Negative => Some((CmpKind::Lt, true)),
            Self::Positive => Some((CmpKind::Ge, true)),
            Self::Carry
            | Self::NotCarry
            | Self::Overflow
            | Self::NotOverflow
            | Self::ParityEven
            | Self::ParityOdd => None,
        }
    }

    /// Returns the status flags required to evaluate this condition.
    #[must_use]
    pub const fn required_flags(self) -> FlagsMask {
        match self {
            Self::Carry | Self::NotCarry => FlagsMask::CARRY,
            Self::Zero | Self::NotZero => FlagsMask::ZERO,
            Self::Overflow | Self::NotOverflow => FlagsMask::OVERFLOW,
            Self::Negative | Self::Positive => FlagsMask::SIGN,
            Self::ParityEven | Self::ParityOdd => FlagsMask::PARITY,
        }
    }
}

impl fmt::Display for FlagCondition {
    /// Writes the snake_case name of this branch condition.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Carry => write!(f, "carry"),
            Self::NotCarry => write!(f, "not_carry"),
            Self::Zero => write!(f, "zero"),
            Self::NotZero => write!(f, "not_zero"),
            Self::Overflow => write!(f, "overflow"),
            Self::NotOverflow => write!(f, "not_overflow"),
            Self::Negative => write!(f, "negative"),
            Self::Positive => write!(f, "positive"),
            Self::ParityEven => write!(f, "parity_even"),
            Self::ParityOdd => write!(f, "parity_odd"),
        }
    }
}

/// What a conditional operation tests of the comparison `left - right`: a
/// relation of the two values, or one flag of their difference (x86
/// `cmp<cc>xadd` names its condition by the flags of that subtraction, sign,
/// overflow and parity among them).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CompareCondition {
    /// The relation `left cmp right`, with the operands unsigned or signed.
    Relation {
        /// The relation.
        cmp: CmpKind,
        /// Whether the operands are read unsigned.
        unsigned: bool,
    },
    /// One flag of `left - right`.
    Flag(FlagCondition),
}

impl fmt::Display for CompareCondition {
    /// Writes the relation's mnemonic, `u`-prefixed when unsigned, or the
    /// flag's name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relation {
                cmp,
                unsigned: true,
            } => write!(f, "u{}", cmp.as_str()),
            Self::Relation { cmp, .. } => write!(f, "{}", cmp.as_str()),
            Self::Flag(flag) => write!(f, "{flag}"),
        }
    }
}

/// Kind of binary operation for extracted binary op info.
///
/// This enum categorizes all binary operations in `SsaOp` for uniform
/// handling in optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BinaryOpKind {
    /// Addition: `left + right`
    Add,
    /// Addition with overflow check
    AddOvf,
    /// Subtraction: `left - right`
    Sub,
    /// Subtraction with overflow check
    SubOvf,
    /// Multiplication: `left * right`
    Mul,
    /// Multiplication with overflow check
    MulOvf,
    /// Division: `left / right`
    Div,
    /// Remainder: `left % right`
    Rem,
    /// Bitwise AND: `left & right`
    And,
    /// Bitwise OR: `left | right`
    Or,
    /// Bitwise XOR: `left ^ right`
    Xor,
    /// Shift left: `value << amount`
    Shl,
    /// Shift right: `value >> amount`
    Shr,
    /// Compare equal: `left == right`
    Ceq,
    /// Compare less than: `left < right`
    Clt,
    /// Compare greater than: `left > right`
    Cgt,
    /// Rotate left
    Rol,
    /// Rotate right
    Ror,
}

impl fmt::Display for BinaryOpKind {
    /// Writes the short mnemonic for this binary operation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::AddOvf => write!(f, "add.ovf"),
            Self::Sub => write!(f, "sub"),
            Self::SubOvf => write!(f, "sub.ovf"),
            Self::Mul => write!(f, "mul"),
            Self::MulOvf => write!(f, "mul.ovf"),
            Self::Div => write!(f, "div"),
            Self::Rem => write!(f, "rem"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Xor => write!(f, "xor"),
            Self::Shl => write!(f, "shl"),
            Self::Shr => write!(f, "shr"),
            Self::Ceq => write!(f, "ceq"),
            Self::Clt => write!(f, "clt"),
            Self::Cgt => write!(f, "cgt"),
            Self::Rol => write!(f, "rol"),
            Self::Ror => write!(f, "ror"),
        }
    }
}

impl BinaryOpKind {
    /// Returns `true` if this operation is commutative (`a op b == b op a`).
    ///
    /// Commutative operations can have their operands swapped without changing
    /// the result. This is useful for normalization in optimizations like GVN.
    ///
    /// # Commutative Operations
    ///
    /// - Arithmetic: `Add`, `AddOvf`, `Mul`, `MulOvf`
    /// - Bitwise: `And`, `Or`, `Xor`
    /// - Comparison: `Ceq` (equality is symmetric)
    #[must_use]
    pub const fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::AddOvf
                | Self::Mul
                | Self::MulOvf
                | Self::And
                | Self::Or
                | Self::Xor
                | Self::Ceq
        )
    }

    /// Returns `true` if this is a comparison operation.
    ///
    /// Comparison operations produce a boolean result (0 or 1) based on
    /// comparing two operands.
    #[must_use]
    pub const fn is_comparison(self) -> bool {
        matches!(self, Self::Ceq | Self::Clt | Self::Cgt)
    }

    /// Returns the operation with swapped operand semantics, if applicable.
    ///
    /// For comparison operations:
    /// - `Clt` (less than) becomes `Cgt` (greater than) when operands swap
    /// - `Cgt` (greater than) becomes `Clt` (less than) when operands swap
    /// - `Ceq` (equal) stays the same (symmetric)
    ///
    /// For non-comparison operations, returns `self` unchanged.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::ir::BinaryOpKind;
    ///
    /// // a < b is equivalent to b > a
    /// assert_eq!(BinaryOpKind::Clt.swapped(), BinaryOpKind::Cgt);
    /// ```
    #[must_use]
    pub const fn swapped(self) -> Self {
        match self {
            Self::Clt => Self::Cgt,
            Self::Cgt => Self::Clt,
            other => other,
        }
    }

    /// Returns `true` if signedness affects the operation's semantics.
    ///
    /// Operations where the `unsigned` flag changes behavior:
    /// - `Div`, `Rem`: Signed vs unsigned division/remainder
    /// - `Shr`: Arithmetic (signed) vs logical (unsigned) shift
    /// - `Clt`, `Cgt`: Signed vs unsigned comparison
    ///
    /// For other operations, the unsigned flag has no effect.
    #[must_use]
    pub const fn is_signedness_sensitive(self) -> bool {
        matches!(
            self,
            Self::Div | Self::Rem | Self::Shr | Self::Clt | Self::Cgt
        )
    }
}

/// Information about a binary operation extracted from an `SsaOp`.
///
/// This provides a uniform view of binary operations for optimization passes,
/// allowing them to handle all binary ops generically without matching on
/// each variant individually.
///
/// # Example
///
/// ```rust
/// use analyssa::{MockTarget, ir::{SsaOp, SsaVarId}};
///
/// let op = SsaOp::<MockTarget>::Add {
///     dest: SsaVarId::from_index(2),
///     left: SsaVarId::from_index(0),
///     right: SsaVarId::from_index(1),
///     flags: None,
/// };
/// if let Some(info) = op.as_binary_op() {
///     // Handle all binary ops uniformly
///     println!("{} = {} {} {}", info.dest, info.left, info.kind, info.right);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BinaryOpInfo {
    /// The kind of binary operation.
    pub kind: BinaryOpKind,
    /// Destination variable for the result.
    pub dest: SsaVarId,
    /// Left operand.
    pub left: SsaVarId,
    /// Right operand.
    pub right: SsaVarId,
    /// Whether the operation treats operands as unsigned.
    pub unsigned: bool,
    /// Optional flags variable defined by this operation.
    pub flags: Option<SsaVarId>,
}

impl BinaryOpInfo {
    /// Returns a normalized version of this operation for value numbering.
    ///
    /// For commutative operations, this ensures operands are in a canonical
    /// order (smaller variable index first). For non-commutative comparisons
    /// like `Clt` and `Cgt`, swapping operands also swaps the operation kind.
    ///
    /// This is useful for Global Value Numbering (GVN) where `a + b` and `b + a`
    /// should hash to the same value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::ir::{BinaryOpInfo, BinaryOpKind, SsaVarId};
    ///
    /// let v2 = SsaVarId::from_index(2);
    /// let v5 = SsaVarId::from_index(5);
    /// let info = BinaryOpInfo {
    ///     kind: BinaryOpKind::Add,
    ///     dest: SsaVarId::from_index(9),
    ///     left: v5,
    ///     right: v2,
    ///     unsigned: false,
    ///     flags: None,
    /// };
    /// let normalized = info.normalized();
    /// assert_eq!(normalized.left, v2);
    /// assert_eq!(normalized.right, v5);
    /// ```
    #[must_use]
    pub fn normalized(self) -> Self {
        // Only normalize if right operand should come first
        if self.right.index() < self.left.index() {
            if self.kind.is_commutative() {
                // Commutative: just swap operands
                Self {
                    left: self.right,
                    right: self.left,
                    ..self
                }
            } else if self.kind.is_comparison() {
                // Non-commutative comparison: swap operands AND operation
                Self {
                    kind: self.kind.swapped(),
                    left: self.right,
                    right: self.left,
                    ..self
                }
            } else {
                // Non-commutative, non-comparison: don't normalize
                self
            }
        } else {
            self
        }
    }

    /// Returns a tuple suitable for use as a hash key in value numbering.
    ///
    /// The tuple includes all semantically relevant fields:
    /// - Operation kind
    /// - Unsigned flag (only if the operation is signedness-sensitive)
    /// - Left and right operands
    ///
    /// For operations where signedness doesn't matter, the unsigned field
    /// is normalized to `false` to ensure consistent hashing.
    #[must_use]
    pub fn value_key(self) -> (BinaryOpKind, bool, SsaVarId, SsaVarId) {
        let unsigned = if self.kind.is_signedness_sensitive() {
            self.unsigned
        } else {
            false // Normalize for consistent hashing
        };
        (self.kind, unsigned, self.left, self.right)
    }
}

/// Kind of unary operation for extracted unary op info.
///
/// This enum categorizes all unary operations in `SsaOp` for uniform
/// handling in optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UnaryOpKind {
    /// Negation: `-operand`
    Neg,
    /// Bitwise NOT: `~operand`
    Not,
    /// Check finite
    Ckfinite,
    /// Byte swap (endian conversion)
    BSwap,
    /// Bit reverse
    BRev,
    /// Bit scan forward (find first set bit, LSB-based)
    BitScanForward,
    /// Bit scan reverse (find first set bit, MSB-based)
    BitScanReverse,
    /// Population count
    Popcount,
    /// Parity (1 if odd number of set bits)
    Parity,
}

impl fmt::Display for UnaryOpKind {
    /// Writes the short mnemonic for this unary operation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neg => write!(f, "neg"),
            Self::Not => write!(f, "not"),
            Self::Ckfinite => write!(f, "ckfinite"),
            Self::BSwap => write!(f, "bswap"),
            Self::BRev => write!(f, "brev"),
            Self::BitScanForward => write!(f, "bsf"),
            Self::BitScanReverse => write!(f, "bsr"),
            Self::Popcount => write!(f, "popcnt"),
            Self::Parity => write!(f, "parity"),
        }
    }
}

/// Information about a unary operation extracted from an `SsaOp`.
///
/// This provides a uniform view of unary operations for optimization passes,
/// allowing them to handle all unary ops generically without matching on
/// each variant individually.
///
/// # Example
///
/// ```rust
/// use analyssa::{ir::{SsaOp, SsaVarId, UnaryOpKind}, MockTarget};
///
/// let op = SsaOp::<MockTarget>::Neg {
///     dest: SsaVarId::from_index(1),
///     operand: SsaVarId::from_index(0),
///     flags: None,
/// };
///
/// // Handle all unary ops uniformly, without matching each variant.
/// let info = op.as_unary_op().expect("Neg is a unary op");
/// assert_eq!(info.kind, UnaryOpKind::Neg);
/// assert_eq!(info.dest, SsaVarId::from_index(1));
/// assert_eq!(info.operand, SsaVarId::from_index(0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnaryOpInfo {
    /// The kind of unary operation.
    pub kind: UnaryOpKind,
    /// Destination variable for the result.
    pub dest: SsaVarId,
    /// The operand.
    pub operand: SsaVarId,
}

impl<T: Target> Emit<T> for CompareCondition {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Relation { cmp, unsigned } => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(cmp, writer)?;
                Emit::emit(unsigned, writer)?;
                Ok(())
            }
            Self::Flag(field_0) => {
                Emit::emit(&1u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for FlagCondition {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Carry => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::NotCarry => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Zero => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::NotZero => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Overflow => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::NotOverflow => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Negative => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Positive => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::ParityEven => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::ParityOdd => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
        }
    }
}
