//! Operand-level enums and classification vocabulary shared by [`SsaOp`].
//!
//! Comparison and signedness kinds, fence / atomic qualifiers, the operation
//! and similarity taxonomies, and the native intrinsic, system, compute and
//! BCD descriptors. These are the small vocabulary types the op variants are
//! built out of; the variants themselves live in [`crate::ir::ops::def`].
//!
//! [`SsaOp`]: crate::ir::ops::def::SsaOp

use std::fmt;

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::{
    ops::{
        effects::{SsaEffectKind, SsaEffects},
        native::FlagsMask,
    },
    variable::SsaVarId,
};

mod crypto;
mod emit;
mod machine;
mod memory;
mod system;

pub use self::{
    crypto::{CarrylessPart, CrcPolynomial, MemoryTagKind, PacKind, ScalarCipherKind},
    machine::MachineStateOp,
    memory::{
        AccessPermission, AccessQualifiers, AtomicAccessWidth, AtomicOrdering, AtomicRmwOp,
        AtomicStatus, CompareBlkData, FenceKind, Locality, RcwCheck, RcwProtection,
    },
    system::{
        BarrierOp, BreakpointOp, CacheMaintenanceOp, FpuControlKind, HardwareEngineOp, HintOp,
        HypervisorOp, InterruptReturnOp, SysRegOp, SystemOpKind, SystemTransactionKind,
        TlbMaintenanceOp, TrapOp,
    },
};

/// Comparison kind for `BranchCmp` operations.
///
/// Represents the comparison operator used in combined compare-and-branch
/// operations like `blt`, `beq`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CmpKind {
    /// Equal: `left == right`
    Eq,
    /// Not equal: `left != right`
    Ne,
    /// Less than: `left < right`
    Lt,
    /// Less than or equal: `left <= right`
    Le,
    /// Greater than: `left > right`
    Gt,
    /// Greater than or equal: `left >= right`
    Ge,
}

impl CmpKind {
    /// Returns the stable short mnemonic for this relation (`"eq"`, `"ne"`,
    /// `"lt"`, `"le"`, `"gt"`, `"ge"`).
    ///
    /// Distinct from [`Display`](fmt::Display), which renders the operator
    /// symbol (`==`, `<`) for human-readable output. This is the
    /// identifier-safe spelling consumers use to build tokens and labels; it
    /// lives on the type so every consumer shares one spelling instead of
    /// re-deriving its own.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        }
    }

    /// Returns the relation that holds exactly when this one does not, **over a
    /// total order**.
    ///
    /// `Eq`/`Ne`, `Lt`/`Ge` and `Gt`/`Le` are complements on the integers, so
    /// `!(a < b)` is `a >= b` and the pair partitions every operand pair.
    ///
    /// # Floats: the complement also flips `unsigned`
    ///
    /// IEEE comparison is a *partial* order: with a NaN operand every ordered
    /// relation is false, so `!(a < b)` and `a >= b` are both false and the
    /// pair does not partition anything. `Eq`/`Ne` survive - `Ne` is defined as
    /// the unordered-or-not-equal relation - but the ordered four do not on
    /// their own.
    ///
    /// On a float comparison the `unsigned` flag that travels with a `CmpKind`
    /// (as on [`SsaOp::BranchCmp`](crate::ir::ops::SsaOp::BranchCmp)) means
    /// "unordered or", as CIL's `.un` branches do. The exact complement of an
    /// ordered `(cmp, unsigned)` on floats is therefore
    /// `(cmp.negated(), !unsigned)` (`Eq`/`Ne` ignore the flag):
    /// `!(a < b)` is "unordered or `a >= b`", which is `bge.un`, and the
    /// complement of `ble` is `bgt.un`. On integers the complement keeps the
    /// flag. A caller that reasons about *both* sides of this negation must
    /// either establish its operands are integers or flip the flag for floats;
    /// one that only reads the relation symbol need do neither.
    #[must_use]
    pub const fn negated(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Ge => Self::Lt,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
        }
    }

    /// Returns the same relation read with its operands exchanged: `a < b` is
    /// `b > a`. Equality and inequality read the same either way.
    #[must_use]
    pub const fn swapped(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::Lt => Self::Gt,
            Self::Gt => Self::Lt,
            Self::Le => Self::Ge,
            Self::Ge => Self::Le,
        }
    }
}

impl fmt::Display for CmpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eq => write!(f, "=="),
            Self::Ne => write!(f, "!="),
            Self::Lt => write!(f, "<"),
            Self::Le => write!(f, "<="),
            Self::Gt => write!(f, ">"),
            Self::Ge => write!(f, ">="),
        }
    }
}

/// Operand signedness interpretation carried by an operation's payload.
///
/// Returned by [`SsaOp::arith_signedness`] for operations whose semantics
/// depend on whether operands are treated as signed or unsigned (division,
/// remainder, arithmetic-vs-logical shift, ordered comparison, conversion).
///
/// [`SsaOp::arith_signedness`]: crate::ir::ops::def::SsaOp::arith_signedness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Signedness {
    /// Operands are interpreted as signed values.
    Signed,
    /// Operands are interpreted as unsigned values.
    Unsigned,
}

impl Signedness {
    /// Converts the payload `unsigned` flag carried by [`SsaOp`] variants.
    ///
    /// [`SsaOp`]: crate::ir::ops::def::SsaOp
    #[must_use]
    pub const fn from_unsigned(unsigned: bool) -> Self {
        if unsigned {
            Self::Unsigned
        } else {
            Self::Signed
        }
    }

    /// Returns `true` when operands are interpreted as unsigned values.
    #[must_use]
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::Unsigned)
    }
}

impl fmt::Display for Signedness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Signed => write!(f, "signed"),
            Self::Unsigned => write!(f, "unsigned"),
        }
    }
}

/// Which value the execution context supplies at a program point.
///
/// A runtime-defined value has no producing computation: entering the block is
/// what makes it hold. The kind says *which* such value it is, so a consumer
/// that must know - a host lowering a handler prologue, a verifier checking
/// where the definition is legal - does not have to infer it from position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RuntimeValueKind {
    /// The exception the runtime dispatched with, as a catch handler or a
    /// filter sees it on entry.
    ///
    /// Pushed only for the two clause kinds that inspect it. A finally or a
    /// fault runs for an exit rather than to examine what was raised, and is
    /// entered with no object.
    ExceptionObject,
    /// State a handler is entered holding, which no predecessor in this
    /// function supplies.
    ///
    /// A handler whose protected region is not part of the function has
    /// nothing to merge on entry: there is no block here for a throw to leave.
    /// That happens two ways, and both are ordinary - the region was removed as
    /// dead, or the handler is a funclet the platform compiles as a separate
    /// function. The values it reads are then whatever the runtime restored,
    /// which is a fact about entering the block and not a computation.
    HandlerEntry,
    /// A value the specification leaves to the implementation, each
    /// occurrence a choice of its own made where control reaches it: Xeon
    /// Phi's gathers and scatters work on a subset of their mask the
    /// implementation picks. It stands anywhere in a block.
    Unspecified,
}

impl fmt::Display for RuntimeValueKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExceptionObject => write!(f, "exception"),
            Self::HandlerEntry => write!(f, "handler-entry"),
            Self::Unspecified => write!(f, "unspecified"),
        }
    }
}

/// Role of an SSA variable operand within an operation.
///
/// Emitted by [`SsaOp::visit_operands`] / [`SsaOp::visit_operands_mut`] for
/// every variable an operation touches: definitions first (the primary
/// destination, then secondary and flag outputs), then uses in payload order.
///
/// [`SsaOp::visit_operands`]: crate::ir::ops::def::SsaOp::visit_operands
/// [`SsaOp::visit_operands_mut`]: crate::ir::ops::def::SsaOp::visit_operands_mut
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OperandRole {
    /// Value defined by the operation. The first `Def` visited is the
    /// primary destination.
    Def,
    /// Condition-flags bundle defined alongside the primary result.
    FlagsDef,
    /// Value read by the operation.
    Use,
}

/// High-level operation family used by verifiers, lifters, and pass scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SsaOpClass {
    /// No-operation or synthetic placeholder.
    Synthetic,
    /// Scalar constant, arithmetic, bitwise, comparison, conversion, or select operation.
    Scalar,
    /// Boolean operation over scalar truth values.
    Boolean,
    /// Condition-code flag producer or consumer.
    Flags,
    /// Vector or SIMD operation.
    Vector,
    /// Ordinary memory load, store, allocation, or block operation.
    Memory,
    /// Atomic memory operation.
    Atomic,
    /// Call or function-pointer operation.
    Call,
    /// Control-flow terminator or branch operation.
    Control,
    /// Implicit-width native arithmetic operation.
    WideArithmetic,
}

/// Stable operation family for similarity and feature extraction.
///
/// These classes are intentionally target-generic and less granular than
/// individual opcodes. They provide a stable vocabulary for MinHash,
/// tracelet, type-flow, memory-shape, and side-effect feature extraction
/// without requiring host crates to match every [`SsaOp`] variant directly.
///
/// [`SsaOp`]: crate::ir::ops::def::SsaOp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SsaSimilarityClass {
    /// Synthetic operation that primarily exists to maintain SSA form.
    Synthetic,
    /// Literal or target metadata constant.
    Constant,
    /// Scalar arithmetic operation.
    Arithmetic,
    /// Scalar bitwise or bit-manipulation operation.
    Bitwise,
    /// Scalar shift or rotate operation.
    ShiftRotate,
    /// Scalar comparison operation.
    Compare,
    /// Boolean operation over truth values.
    Boolean,
    /// Conditional value selection.
    Select,
    /// Type, representation, or checked floating-point conversion.
    Conversion,
    /// Argument, local, or function metadata access.
    TypeFlow,
    /// Memory read or address-producing operation.
    MemoryRead,
    /// Memory write operation.
    MemoryWrite,
    /// Memory read-write or bulk-memory operation.
    MemoryReadWrite,
    /// Allocation operation.
    Allocation,
    /// Atomic memory operation.
    Atomic,
    /// Memory ordering barrier.
    Fence,
    /// Call or function-pointer operation.
    Call,
    /// Control-flow terminator or branch operation.
    Control,
    /// Vector or SIMD operation.
    Vector,
    /// Condition-code flag producer or consumer.
    Flags,
    /// Implicit-width native arithmetic operation.
    WideArithmetic,
}

/// Canonical target-generic feature token for an SSA operation.
///
/// The token avoids host-specific metadata and variable IDs. It captures the
/// opcode family, side-effect class, arity, and definition count in a stable
/// shape suitable for deterministic similarity features.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SsaFeatureToken {
    /// Stable opcode name.
    pub opcode: &'static str,
    /// Coarse operation class.
    pub op_class: SsaOpClass,
    /// Similarity-oriented operation class.
    pub similarity_class: SsaSimilarityClass,
    /// Memory and side-effect class.
    pub effect_kind: SsaEffectKind,
    /// Number of SSA definitions produced by the operation.
    pub def_count: usize,
    /// Number of SSA variables used by the operation.
    pub use_count: usize,
    /// Whether the operation can trap or throw.
    pub may_throw: bool,
}

impl fmt::Display for SsaFeatureToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "op={};class={:?};sim={:?};effect={:?};defs={};uses={};throw={}",
            self.opcode,
            self.op_class,
            self.similarity_class,
            self.effect_kind,
            self.def_count,
            self.use_count,
            self.may_throw
        )
    }
}

/// Original native instruction metadata retained for opaque operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NativeInstructionMetadata {
    /// Architecture or backend family that produced the instruction.
    pub architecture: Option<String>,
    /// Original instruction address when available.
    pub address: Option<u64>,
    /// Original encoded bytes when available.
    pub raw_bytes: Vec<u8>,
}

impl NativeInstructionMetadata {
    /// Creates native instruction metadata.
    #[must_use]
    pub fn new(architecture: Option<String>, address: Option<u64>, raw_bytes: Vec<u8>) -> Self {
        Self {
            architecture,
            address,
            raw_bytes,
        }
    }
}

/// Boxed payload shared by the kind-tagged vector/native compute ops whose shape
/// is exactly a structured `kind` plus explicit SSA out/in lists. Each `SsaOp`
/// variant keeps its own identity (opcode name, effects, display) through the
/// `K` it instantiates; this struct only unifies the layout those variants share,
/// which would otherwise be a dozen byte-identical `*Data` structs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KindedVecData<K> {
    /// Structured identity of the operation.
    pub kind: K,
    /// Explicit SSA outputs defined by the operation.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload shared by the kind-tagged native ops that also carry a native
/// mnemonic and optional source metadata alongside their SSA operand lists.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NativeKindedData<K> {
    /// Structured identity of the operation.
    pub kind: K,
    /// Human-readable native mnemonic, for display / provenance only.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs defined by the operation.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation.
    pub inputs: Vec<SsaVarId>,
}

/// A floating-point function the machine computes as one operation
/// ([`SsaOp::FloatFunction`]): a transcendental, or one of the exact
/// functions (the remainders, scaling and extraction).
///
/// [`SsaOp::FloatFunction`]: crate::ir::ops::def::SsaOp::FloatFunction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum FloatFunctionKind {
    /// Sine (`fsin`).
    Sin,
    /// Cosine (`fcos`).
    Cos,
    /// Sine and cosine together (`fsincos` - two results).
    SinCos,
    /// Tangent, pushing the `1.0` constant (`fptan` - two results).
    Tan,
    /// Arctangent of `arg1/arg0` (`fpatan` - two operands).
    Atan,
    /// `2^x - 1` (`f2xm1`).
    Exp2m1,
    /// `arg1 * log2(arg0)` (`fyl2x` - two operands).
    Ylog2,
    /// `arg1 * log2(arg0 + 1)` (`fyl2xp1` - two operands).
    Ylog2p1,
    /// Partial remainder (`fprem` - two operands), the quotient truncated.
    /// Its results are the remainder, the condition codes it leaves as the
    /// x87 compare's flags lay them out (`C2` at the parity flag, set where
    /// the reduction is incomplete; the quotient's bit 2 in `C0` at the carry
    /// flag and bit 1 in `C3` at the zero flag), and the quotient's bit 0
    /// (`C1`).
    Rem,
    /// IEEE partial remainder (`fprem1` - two operands), the quotient rounded
    /// to nearest; its results are [`Self::Rem`]'s.
    Rem1,
    /// Scale by a power of two (`fscale` - two operands): the first times two
    /// to the second rounded toward zero.
    Scale,
    /// Extract exponent and significand (`fxtract` - two results): the
    /// unbiased exponent, then the significand in `[1, 2)` with the sign.
    Extract,
}

impl FloatFunctionKind {
    /// Returns the IR name of this float function, such as `sin`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// The operand type is a qualifier of the row that holds the operation, not part of the name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::SinCos => "sincos",
            Self::Tan => "tan",
            Self::Atan => "atan2",
            Self::Exp2m1 => "exp2m1",
            Self::Ylog2 => "ylog2",
            Self::Ylog2p1 => "ylog2p1",
            Self::Rem => "rem.trunc",
            Self::Rem1 => "rem.round",
            Self::Scale => "scale.trunc",
            Self::Extract => "exponent_mantissa",
        }
    }
}

/// The legacy x86 binary-coded-decimal adjust a [`Kernel::DecimalAdjust`]
/// computes: `daa`, `das`, `aaa`, `aas`, `aam` and `aad`. The accumulator
/// and flags flow through as typed SSA values rather than the flag-dependent
/// correction being decomposed into branches.
///
/// [`Kernel::DecimalAdjust`]: crate::ir::ops::Kernel::DecimalAdjust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum BcdAdjustKind {
    /// Decimal adjust `AL` after addition (`daa`).
    DecimalAddAdjust,
    /// Decimal adjust `AL` after subtraction (`das`).
    DecimalSubAdjust,
    /// ASCII adjust `AL` after addition (`aaa`).
    AsciiAddAdjust,
    /// ASCII adjust `AL` after subtraction (`aas`).
    AsciiSubAdjust,
    /// ASCII adjust `AX` after multiply (`aam`), in the kernel's radix (10
    /// unless an explicit `imm8` is given).
    AsciiMulAdjust,
    /// ASCII adjust `AX` before division (`aad`), in the kernel's radix.
    AsciiDivAdjust,
}

impl BcdAdjustKind {
    /// Returns whether the adjust reads the incoming carry and adjust flags:
    /// every one but `aam` and `aad`, which are functions of the accumulator
    /// and the radix alone.
    #[must_use]
    pub const fn reads_flags(self) -> bool {
        !matches!(self, Self::AsciiMulAdjust | Self::AsciiDivAdjust)
    }

    /// Returns the IR name of this decimal adjust, such as `bcd.adjust.add`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DecimalAddAdjust => "bcd.adjust.add",
            Self::DecimalSubAdjust => "bcd.adjust.sub",
            Self::AsciiAddAdjust => "ascii.adjust.add",
            Self::AsciiSubAdjust => "ascii.adjust.sub",
            Self::AsciiMulAdjust => "ascii.adjust.mul",
            Self::AsciiDivAdjust => "ascii.adjust.div",
        }
    }

    /// The adjusted accumulator and flags this adjust computes of the
    /// accumulator (`AL`, or `AX` for the ASCII adjusts) and the incoming
    /// flags (the carry and adjust flags the add and subtract adjusts read),
    /// in radix `base` for `aam` and `aad`.
    ///
    /// The flags the SDM leaves undefined are those an Intel Broadwell
    /// computes, for every input: the overflow flag cleared after `daa`,
    /// `das`, `aaa` and `aas`; the sign, zero and parity flags of the masked
    /// `AL` after `aaa` and `aas`; the carry, adjust and overflow flags
    /// cleared after `aam`; and after `aad` the flags of the byte addition
    /// of `AL` and `AH * base` that forms the result.
    ///
    /// # Returns
    ///
    /// The accumulator and the flags, or `None` for `aam` in radix 0, which
    /// raises a divide error.
    #[must_use]
    pub fn evaluate(
        self,
        base: u8,
        accumulator: u16,
        flags: FlagsMask,
    ) -> Option<(u16, FlagsMask)> {
        let [al, ah] = accumulator.to_le_bytes();
        let incoming = |flag: FlagsMask| flags.bits() & flag.bits() != 0;
        let (carry, adjust) = (incoming(FlagsMask::CARRY), incoming(FlagsMask::ADJUST));
        let bit = |set: bool, flag: FlagsMask| if set { flag.bits() } else { 0 };
        // The sign, zero and parity flags of a byte result.
        let szp = |byte: u8| {
            bit(byte == 0, FlagsMask::ZERO)
                | bit(byte & 0x80 != 0, FlagsMask::SIGN)
                | bit(byte.count_ones().is_multiple_of(2), FlagsMask::PARITY)
        };
        let (value, flags) = match self {
            Self::DecimalAddAdjust | Self::DecimalSubAdjust => {
                let low = al & 0xf > 9 || adjust;
                let high = al > 0x99 || carry;
                let correction = (if low { 0x06 } else { 0 }) | (if high { 0x60 } else { 0 });
                let (result, carry) = if self == Self::DecimalAddAdjust {
                    (al.wrapping_add(correction), high)
                } else {
                    // The borrow of the low correction stands where no high
                    // one replaces it.
                    (al.wrapping_sub(correction), high || low && al < 6)
                };
                let flags =
                    bit(carry, FlagsMask::CARRY) | bit(low, FlagsMask::ADJUST) | szp(result);
                (u16::from(result), flags)
            }
            Self::AsciiAddAdjust | Self::AsciiSubAdjust => {
                let adjusted = al & 0xf > 9 || adjust;
                let value = match (adjusted, self == Self::AsciiAddAdjust) {
                    (false, _) => accumulator,
                    (true, true) => accumulator.wrapping_add(0x106),
                    (true, false) => accumulator.wrapping_sub(0x106),
                } & 0xff0f;
                let [low, _] = value.to_le_bytes();
                let flags =
                    bit(adjusted, FlagsMask::CARRY) | bit(adjusted, FlagsMask::ADJUST) | szp(low);
                (value, flags)
            }
            Self::AsciiMulAdjust => {
                let (quotient, remainder) = (al.checked_div(base)?, al.checked_rem(base)?);
                (u16::from_le_bytes([remainder, quotient]), szp(remainder))
            }
            Self::AsciiDivAdjust => {
                let product = ah.wrapping_mul(base);
                let (sum, carry) = al.overflowing_add(product);
                let half = (al & 0xf).wrapping_add(product & 0xf) > 0xf;
                let overflow = (al ^ sum) & (product ^ sum) & 0x80 != 0;
                let flags = bit(carry, FlagsMask::CARRY)
                    | bit(half, FlagsMask::ADJUST)
                    | bit(overflow, FlagsMask::OVERFLOW)
                    | szp(sum);
                (u16::from(sum), flags)
            }
        };
        Some((value, FlagsMask::from_bits(flags)))
    }

    /// Effect summary - every BCD adjust is a pure function of the accumulator
    /// and the incoming arithmetic flags (no memory, no trap, no nondeterminism).
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        SsaEffects::new(SsaEffectKind::Pure, false)
    }

    /// Stable display / fingerprint key for this BCD adjust.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::DecimalAddAdjust => "bcd.daa",
            Self::DecimalSubAdjust => "bcd.das",
            Self::AsciiAddAdjust => "bcd.aaa",
            Self::AsciiSubAdjust => "bcd.aas",
            Self::AsciiMulAdjust => "bcd.aam",
            Self::AsciiDivAdjust => "bcd.aad",
        }
    }
}

impl_kinded_payload! {
    KindedVecData<K> { kind, outputs, inputs };
    NativeKindedData<K> { kind, mnemonic, metadata, outputs, inputs };
}
