//! Operand-level enums and classification vocabulary shared by [`SsaOp`].
//!
//! Comparison and signedness kinds, fence / atomic qualifiers, the operation
//! and similarity taxonomies, and the native intrinsic, system, compute and
//! BCD descriptors. These are the small vocabulary types the op variants are
//! built out of; the variants themselves live in [`super::def`].
//!
//! [`SsaOp`]: crate::ir::ops::def::SsaOp

use std::fmt;

use crate::ir::{
    ops::{
        effects::{ControlEffect, MemoryEffectLocation, SsaEffectKind, SsaEffects},
        operands::impl_kinded_payload,
        table::OpKindTable,
    },
    variable::SsaVarId,
};

/// Comparison kind for `BranchCmp` operations.
///
/// Represents the comparison operator used in combined compare-and-branch
/// operations like `blt`, `beq`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Memory fence / barrier kind for atomic ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FenceKind {
    /// Full memory barrier
    Full,
    /// Acquire barrier
    Acquire,
    /// Release barrier
    Release,
    /// Acquire+Release barrier
    AcqRel,
    /// Sequentially consistent barrier
    SeqCst,
}

impl fmt::Display for FenceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Acquire => write!(f, "acquire"),
            Self::Release => write!(f, "release"),
            Self::AcqRel => write!(f, "acqrel"),
            Self::SeqCst => write!(f, "seqcst"),
        }
    }
}

impl FenceKind {
    /// Returns the closest atomic ordering represented by this fence.
    #[must_use]
    pub const fn ordering(self) -> AtomicOrdering {
        match self {
            Self::Full | Self::SeqCst => AtomicOrdering::SeqCst,
            Self::Acquire => AtomicOrdering::Acquire,
            Self::Release => AtomicOrdering::Release,
            Self::AcqRel => AtomicOrdering::AcqRel,
        }
    }
}

/// Memory ordering constraint for native atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AtomicOrdering {
    /// No cross-thread ordering beyond atomicity.
    Relaxed,
    /// Acquire ordering for operations that read memory.
    Acquire,
    /// Release ordering for operations that write memory.
    Release,
    /// Acquire and release ordering.
    AcqRel,
    /// Sequentially consistent ordering.
    SeqCst,
}

impl fmt::Display for AtomicOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relaxed => write!(f, "relaxed"),
            Self::Acquire => write!(f, "acquire"),
            Self::Release => write!(f, "release"),
            Self::AcqRel => write!(f, "acqrel"),
            Self::SeqCst => write!(f, "seqcst"),
        }
    }
}

/// Access width for native atomic memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AtomicAccessWidth {
    /// 8-bit atomic access.
    Bits8,
    /// 16-bit atomic access.
    Bits16,
    /// 32-bit atomic access.
    Bits32,
    /// 64-bit atomic access.
    Bits64,
    /// 128-bit atomic access.
    Bits128,
    /// Target pointer-sized atomic access.
    Pointer,
}

impl AtomicAccessWidth {
    /// Returns the concrete bit width when it is target-independent.
    #[must_use]
    pub const fn bits(self) -> Option<u32> {
        match self {
            Self::Bits8 => Some(8),
            Self::Bits16 => Some(16),
            Self::Bits32 => Some(32),
            Self::Bits64 => Some(64),
            Self::Bits128 => Some(128),
            Self::Pointer => None,
        }
    }
}

impl fmt::Display for AtomicAccessWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bits8 => write!(f, "i8"),
            Self::Bits16 => write!(f, "i16"),
            Self::Bits32 => write!(f, "i32"),
            Self::Bits64 => write!(f, "i64"),
            Self::Bits128 => write!(f, "i128"),
            Self::Pointer => write!(f, "ptr"),
        }
    }
}

/// Atomic read-modify-write operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AtomicRmwOp {
    /// Exchange
    Xchg,
    /// Add
    Add,
    /// Sub
    Sub,
    /// And
    And,
    /// Or
    Or,
    /// Xor
    Xor,
    /// Min (signed)
    Min,
    /// Max (signed)
    Max,
    /// Bit-clear: `*p &= !value` (AArch64 `ldclr`/`stclr`).
    AndNot,
    /// Min (unsigned) (AArch64 `ldumin`/`stumin`).
    MinU,
    /// Max (unsigned) (AArch64 `ldumax`/`stumax`).
    MaxU,
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Native opaque operation.
    NativeOpaque,
    /// Implicit-width native arithmetic operation.
    WideArithmetic,
    /// Metadata prefix or target constraint.
    Prefix,
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Native opaque operation.
    NativeOpaque,
    /// Metadata prefix or target constraint.
    Prefix,
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Boxed payload for [`SsaOp::NativeOpaque`].
///
/// `NativeOpaque` carries an entire native instruction's worth of state
/// (mnemonic, original encoding metadata, explicit inputs/outputs, and an
/// effect summary). Inlining that into the enum would make *every*
/// [`SsaOp`] as large as this rare variant, so the payload is held behind a
/// `Box` and the common operations stay compact.
///
/// [`SsaOp::NativeOpaque`]: crate::ir::ops::def::SsaOp::NativeOpaque
/// [`SsaOp`]: crate::ir::ops::def::SsaOp
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NativeOpaqueData {
    /// Human-readable instruction mnemonic or description.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs defined by the instruction.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the instruction.
    pub inputs: Vec<SsaVarId>,
    /// Conservative effect summary for optimization barriers.
    pub effects: SsaEffects,
}

/// PAC sub-operation carried by [`ComputeKind::PointerAuth`].
///
/// ARMv8.3 pointer authentication signs a pointer with a cryptographic MAC in
/// its unused high bits, authenticates (validates) it before use, strips the
/// MAC back to the raw pointer, or computes a generic MAC into a register.
/// Distinguishing these is required to faithfully reconstruct the native
/// instruction role; without it every PAC op collapses to "sign".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum PacKind {
    /// Sign a pointer, inserting the authentication code (`pacia`/`pacib`/
    /// `pacda`/`pacdb`/`paciasp`/`pacibsp`/`paciaz`/…).
    Sign,
    /// Authenticate (validate) a signed pointer, faulting/poisoning on tamper
    /// (`autia`/`autib`/`autda`/`autdb`/`autiasp`/`autibsp`/…).
    Authenticate,
    /// Strip the authentication code, yielding the raw pointer (`xpaci`/
    /// `xpacd`/`xpaclri`).
    Strip,
    /// Compute a generic authentication code into a register (`pacga`).
    GenericMac,
}

impl PacKind {
    /// Returns the neutral display mnemonic for this sub-operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Sign => "pac.sign",
            Self::Authenticate => "pac.auth",
            Self::Strip => "pac.strip",
            Self::GenericMac => "pac.genmac",
        }
    }

    /// Returns the stable display / fingerprint key for this sub-operation.
    ///
    /// [`ComputeKind::PointerAuth`] reads it, so the four PAC operations keep
    /// four fingerprints. One spelling for the whole family would collapse them
    /// onto a single identity — including the sign/authenticate pair, whose
    /// *difference* is the security property.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Sign => "compute.pac.sign",
            Self::Authenticate => "compute.pac.auth",
            Self::Strip => "compute.pac.strip",
            Self::GenericMac => "compute.pac.genmac",
        }
    }
}

impl fmt::Display for PacKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Sign => "sign",
            Self::Authenticate => "auth",
            Self::Strip => "strip",
            Self::GenericMac => "genmac",
        };
        f.write_str(name)
    }
}

/// Boxed payload shared by the kind-tagged vector/native compute ops whose shape
/// is exactly a structured `kind` plus explicit SSA out/in lists. Each `SsaOp`
/// variant keeps its own identity (opcode name, effects, display) through the
/// `K` it instantiates; this struct only unifies the layout those variants share,
/// which would otherwise be a dozen byte-identical `*Data` structs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KindedVecData<K> {
    /// Structured identity of the operation.
    pub kind: K,
    /// Explicit SSA outputs defined by the operation.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload shared by vector ops parameterized by a single 8-bit immediate
/// (truth table / block-offset / category selector) plus SSA out/in lists.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VecImm8Data {
    /// The 8-bit immediate selecting the operation's mode.
    pub imm8: u8,
    /// Explicit SSA outputs defined by the operation.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation.
    pub inputs: Vec<SsaVarId>,
}

/// Boxed payload shared by the kind-tagged native ops that also carry a native
/// mnemonic and optional source metadata alongside their SSA operand lists.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Hardware floating-point transcendental / residue function
/// ([`SsaOp::FpTranscendental`]).
///
/// [`SsaOp::FpTranscendental`]: crate::ir::ops::def::SsaOp::FpTranscendental
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum TranscendentalKind {
    /// Sine (`fsin`).
    Sin,
    /// Cosine (`fcos`).
    Cos,
    /// Sine and cosine together (`fsincos` — two results).
    SinCos,
    /// Tangent, pushing the `1.0` constant (`fptan` — two results).
    Tan,
    /// Arctangent of `arg1/arg0` (`fpatan` — two operands).
    Atan,
    /// `2^x - 1` (`f2xm1`).
    Exp2m1,
    /// `arg1 * log2(arg0)` (`fyl2x` — two operands).
    Ylog2,
    /// `arg1 * log2(arg0 + 1)` (`fyl2xp1` — two operands).
    Ylog2p1,
    /// Partial remainder (`fprem` — two operands).
    Rem,
    /// IEEE partial remainder (`fprem1` — two operands).
    Rem1,
    /// Scale by power of two (`fscale` — two operands).
    Scale,
    /// Extract exponent and mantissa (`fxtract` — two results).
    Extract,
}

/// Floating-point unit control / state operation ([`SsaOp::FpuControl`]).
///
/// [`SsaOp::FpuControl`]: crate::ir::ops::def::SsaOp::FpuControl
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum FpuControlKind {
    /// Load the FPU control word (`fldcw`).
    LoadControlWord,
    /// Store the FPU control word (`fnstcw`/`fstcw`).
    StoreControlWord,
    /// Store the FPU status word (`fnstsw`/`fstsw`).
    StoreStatusWord,
    /// Load the FPU environment (`fldenv`).
    LoadEnvironment,
    /// Store the FPU environment (`fnstenv`/`fstenv`).
    StoreEnvironment,
    /// Save full FPU state (`fnsave`/`fsave`).
    Save,
    /// Restore full FPU state (`frstor`).
    Restore,
    /// Save extended (SSE+FPU) state (`fxsave`).
    SaveExtended,
    /// Restore extended (SSE+FPU) state (`fxrstor`).
    RestoreExtended,
    /// Clear floating-point exceptions (`fnclex`/`fclex`).
    ClearExceptions,
    /// Decrement the FPU stack-top pointer (`fdecstp`).
    DecrementStackTop,
    /// Increment the FPU stack-top pointer (`fincstp`).
    IncrementStackTop,
    /// Mark an FPU register free (`ffree`).
    FreeRegister,
    /// FPU no-op (`fnop`).
    NoOp,
    /// Wait for pending FPU exceptions (`wait`/`fwait`).
    Wait,
    /// Initialize / reset the FPU unit (`fninit`/`finit`) — clears the control
    /// word, status word, tag word and resets the stack top.
    Initialize,
    /// Empty the MMX technology state (`emms`/`femms`) — marks every x87 tag
    /// word entry empty, transitioning the register file out of MMX mode.
    EmptyMmxState,
    /// Load the SSE control/status register from memory (`ldmxcsr`/`vldmxcsr`).
    LoadMxcsr,
    /// Store the SSE control/status register to memory (`stmxcsr`/`vstmxcsr`).
    StoreMxcsr,
    /// Save the processor extended state selected by the `EDX:EAX` component
    /// mask, in the standard layout (`xsave`).
    SaveExtendedMasked,
    /// Save the mask-selected extended state in the compacted layout
    /// (`xsavec`).
    SaveExtendedCompact,
    /// Save the mask-selected extended state, skipping components the
    /// init/modified optimization proves unchanged (`xsaveopt`).
    SaveExtendedOptimized,
    /// Save the mask-selected extended state including supervisor components
    /// (`xsaves`).
    SaveExtendedSupervisor,
    /// Restore the extended state selected by the `EDX:EAX` component mask
    /// (`xrstor`).
    RestoreExtendedMasked,
    /// Restore the mask-selected extended state including supervisor
    /// components (`xrstors`).
    RestoreExtendedSupervisor,
}

/// Cache-maintenance operation named by [`SystemOpKind::CacheMaintenance`].
///
/// One variant per operation, so the family reconstructs the instruction's
/// own mnemonic rather than a representative one standing for the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum CacheMaintenanceOp {
    /// Invalidate every cache without writing back (`invd`).
    InvalidateAll,
    /// Write back and invalidate every cache (`wbinvd`).
    WriteBackInvalidateAll,
    /// Write back every cache without invalidating (`wbnoinvd`).
    WriteBackAll,
    /// Flush one cache line — `clflush` and its unordered form.
    FlushLine,
    /// Write back one line, keeping it valid (`clwb`).
    WriteBackLine,
    /// Move one line to a more distant level (`cldemote`).
    DemoteLine,
    /// Zero one cache line (`clzero`).
    ZeroLine,
    /// Evict one line from a named level (`clevict0`, `clevict1`).
    EvictLine,
    /// Invalidate a memory block (`cl1invmb`).
    InvalidateBlock,
    /// Data-cache maintenance, the operation selected by an operand — AArch64 `dc`, MIPS `cache`.
    DataCacheOperation,
    /// Instruction-cache maintenance — AArch64 `ic`, MIPS `synci`.
    InstructionCacheOperation,
}

impl CacheMaintenanceOp {
    /// Returns the neutral display mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::InvalidateAll => "cache.invalidate.all",
            Self::WriteBackInvalidateAll => "cache.writeback.invalidate.all",
            Self::WriteBackAll => "cache.writeback.all",
            Self::FlushLine => "cache.flush.line",
            Self::WriteBackLine => "cache.writeback.line",
            Self::DemoteLine => "cache.demote.line",
            Self::ZeroLine => "cache.zero.line",
            Self::EvictLine => "cache.evict.line",
            Self::InvalidateBlock => "cache.invalidate.block",
            Self::DataCacheOperation => "cache.data.op",
            Self::InstructionCacheOperation => "cache.code.op",
        }
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::InvalidateAll => "system.cache.invalidate.all",
            Self::WriteBackInvalidateAll => "system.cache.writeback.invalidate.all",
            Self::WriteBackAll => "system.cache.writeback.all",
            Self::FlushLine => "system.cache.flush.line",
            Self::WriteBackLine => "system.cache.writeback.line",
            Self::DemoteLine => "system.cache.demote.line",
            Self::ZeroLine => "system.cache.zero.line",
            Self::EvictLine => "system.cache.evict.line",
            Self::InvalidateBlock => "system.cache.invalidate.block",
            Self::DataCacheOperation => "system.cache.data.op",
            Self::InstructionCacheOperation => "system.cache.code.op",
        }
    }
}

/// TLB-maintenance operation named by [`SystemOpKind::TlbMaintenance`].
///
/// One variant per operation, so the family reconstructs the instruction's
/// own mnemonic rather than a representative one standing for the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum TlbMaintenanceOp {
    /// Invalidate the entries for an address — x86 `invlpg`, AArch64 `tlbi`. One operation.
    InvalidatePage,
    /// Invalidate a guest entry by address space (`invlpga`).
    InvalidatePageGuest,
    /// Broadcast an invalidation to other processors (`invlpgb`).
    InvalidatePageBroadcast,
    /// Invalidate entries by process or virtual-processor identifier (`invpcid`, `invvpid`).
    InvalidateByContext,
    /// Invalidate cached nested-paging mappings (`invept`).
    InvalidateNestedMapping,
    /// Order translation writes against later translations (`tlbsync`, `sfence.vma`).
    Synchronize,
    /// Write the entry the index register selects (`tlbwi`).
    WriteEntryIndexed,
    /// Write the entry the random register selects (`tlbwr`).
    WriteEntryRandom,
    /// Read the entry the index register selects (`tlbr`).
    ReadEntry,
    /// Probe for a matching entry (`tlbp`).
    Probe,
}

impl TlbMaintenanceOp {
    /// Returns the neutral display mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::InvalidatePage => "tlb.invalidate.page",
            Self::InvalidatePageGuest => "tlb.invalidate.guest",
            Self::InvalidatePageBroadcast => "tlb.invalidate.broadcast",
            Self::InvalidateByContext => "tlb.invalidate.context",
            Self::InvalidateNestedMapping => "tlb.invalidate.nested",
            Self::Synchronize => "tlb.sync",
            Self::WriteEntryIndexed => "tlb.write.indexed",
            Self::WriteEntryRandom => "tlb.write.random",
            Self::ReadEntry => "tlb.read",
            Self::Probe => "tlb.probe",
        }
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::InvalidatePage => "system.tlb.invalidate.page",
            Self::InvalidatePageGuest => "system.tlb.invalidate.guest",
            Self::InvalidatePageBroadcast => "system.tlb.invalidate.broadcast",
            Self::InvalidateByContext => "system.tlb.invalidate.context",
            Self::InvalidateNestedMapping => "system.tlb.invalidate.nested",
            Self::Synchronize => "system.tlb.sync",
            Self::WriteEntryIndexed => "system.tlb.write.indexed",
            Self::WriteEntryRandom => "system.tlb.write.random",
            Self::ReadEntry => "system.tlb.read",
            Self::Probe => "system.tlb.probe",
        }
    }
}

/// Serializing operation named by [`SystemOpKind::Barrier`].
///
/// One variant per operation, so the family reconstructs the instruction's
/// own mnemonic rather than a representative one standing for the class.
///
/// Transactional memory is **not** here. `xend`, `xtest`, `xabort` and the
/// load-tracking pair name transactional state rather than memory ordering, and
/// their one home is [`SystemTransactionKind`]. Two homes for one instruction
/// means two effect summaries for it, and whether a store may move across
/// `xend` would then depend on which spelling a front-end picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum BarrierOp {
    /// Serialize instruction execution (`serialize`).
    Serialize,
    /// Commit outstanding stores and report faults (`mcommit`).
    CommitStores,
    /// Commit stores to persistent memory (`pcommit`).
    CommitPersistent,
}

impl BarrierOp {
    /// Returns the canonical assembler mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Serialize => "serialize",
            Self::CommitStores => "mcommit",
            Self::CommitPersistent => "pcommit",
        }
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Serialize => "system.barrier.serialize",
            Self::CommitStores => "system.barrier.mcommit",
            Self::CommitPersistent => "system.barrier.pcommit",
        }
    }
}

/// Virtualization or secure-enclave operation named by [`SystemOpKind::Hypervisor`].
///
/// One variant per operation, so the family reconstructs the instruction's
/// own mnemonic rather than a representative one standing for the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum HypervisorOp {
    /// Call the hypervisor — x86 `vmcall` and `vmmcall`, ARM `hvc`. One operation.
    HypervisorCall,
    /// Call the secure monitor (ARM `smc`).
    SecureMonitorCall,
    /// Transfer to a guest context (`vmlaunch`, `vmresume`, `vmrun`).
    GuestEnter,
    /// Exit to the hypervisor from a guest (`vmgexit`).
    GuestExit,
    /// Enter virtualization operation (`vmxon`).
    VirtualizationEnter,
    /// Leave virtualization operation (`vmxoff`).
    VirtualizationLeave,
    /// Load the current guest control block (`vmptrld`, `vmload`).
    ControlBlockLoad,
    /// Store the current guest control block (`vmptrst`, `vmsave`).
    ControlBlockStore,
    /// Clear a guest control block and flush it (`vmclear`).
    ControlBlockClear,
    /// Read a field of the guest control block (`vmread`).
    ControlBlockRead,
    /// Write a field of the guest control block (`vmwrite`).
    ControlBlockWrite,
    /// Invoke a virtualization function without exiting (`vmfunc`).
    VirtualizationFunction,
    /// Set the global interrupt flag (`stgi`).
    GlobalInterruptEnable,
    /// Clear the global interrupt flag (`clgi`).
    GlobalInterruptDisable,
    /// Initialise a measured secure environment (`skinit`, `getsec`).
    SecureInitialize,
    /// Execute an enclave leaf the operand selects (`encls`, `enclu`, `enclv`).
    EnclaveOperation,
    /// Return into an enclave (`erets`, `eretu`).
    EnclaveReturn,
    /// Call a trusted module (`seamcall`, `tdcall`).
    TrustedModuleCall,
    /// Return from a trusted module (`seamret`).
    TrustedModuleReturn,
    /// Trusted-module management operation (`seamops`).
    TrustedModuleOperation,
}

impl HypervisorOp {
    /// Returns the neutral display mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::HypervisorCall => "hvcall",
            Self::SecureMonitorCall => "smcall",
            Self::GuestEnter => "guest.enter",
            Self::GuestExit => "guest.exit",
            Self::VirtualizationEnter => "virt.enter",
            Self::VirtualizationLeave => "virt.leave",
            Self::ControlBlockLoad => "vmcb.load",
            Self::ControlBlockStore => "vmcb.store",
            Self::ControlBlockClear => "vmcb.clear",
            Self::ControlBlockRead => "vmcb.read",
            Self::ControlBlockWrite => "vmcb.write",
            Self::VirtualizationFunction => "virt.function",
            Self::GlobalInterruptEnable => "virt.gif.set",
            Self::GlobalInterruptDisable => "virt.gif.clear",
            Self::SecureInitialize => "secure.init",
            Self::EnclaveOperation => "enclave.op",
            Self::EnclaveReturn => "enclave.return",
            Self::TrustedModuleCall => "tdx.call",
            Self::TrustedModuleReturn => "tdx.return",
            Self::TrustedModuleOperation => "tdx.op",
        }
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::HypervisorCall => "system.hvcall",
            Self::SecureMonitorCall => "system.smcall",
            Self::GuestEnter => "system.guest.enter",
            Self::GuestExit => "system.guest.exit",
            Self::VirtualizationEnter => "system.virt.enter",
            Self::VirtualizationLeave => "system.virt.leave",
            Self::ControlBlockLoad => "system.vmcb.load",
            Self::ControlBlockStore => "system.vmcb.store",
            Self::ControlBlockClear => "system.vmcb.clear",
            Self::ControlBlockRead => "system.vmcb.read",
            Self::ControlBlockWrite => "system.vmcb.write",
            Self::VirtualizationFunction => "system.virt.function",
            Self::GlobalInterruptEnable => "system.virt.gif.set",
            Self::GlobalInterruptDisable => "system.virt.gif.clear",
            Self::SecureInitialize => "system.secure.init",
            Self::EnclaveOperation => "system.enclave.op",
            Self::EnclaveReturn => "system.enclave.return",
            Self::TrustedModuleCall => "system.tdx.call",
            Self::TrustedModuleReturn => "system.tdx.return",
            Self::TrustedModuleOperation => "system.tdx.op",
        }
    }
}

/// On-chip engine operation named by [`SystemOpKind::HardwareEngine`].
///
/// One variant per operation, so the family reconstructs the instruction's
/// own mnemonic rather than a representative one standing for the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum HardwareEngineOp {
    /// Store bytes from the on-chip random-number engine (`xstore`).
    RandomStore,
    /// Encrypt or decrypt a buffer in ECB mode (`xcryptecb`).
    CipherEcb,
    /// Encrypt or decrypt a buffer in CBC mode (`xcryptcbc`).
    CipherCbc,
    /// Encrypt or decrypt a buffer in counter mode (`xcryptctr`).
    CipherCtr,
    /// Encrypt or decrypt a buffer in CFB mode (`xcryptcfb`).
    CipherCfb,
    /// Encrypt or decrypt a buffer in OFB mode (`xcryptofb`).
    CipherOfb,
    /// Hash a buffer with SHA-1 (`xsha1`).
    HashSha1,
    /// Hash a buffer with SHA-256 (`xsha256`).
    HashSha256,
    /// Hash a buffer with SHA-512 (`xsha512`).
    HashSha512,
    /// Montgomery-multiply a buffer (`montmul`).
    MontgomeryMultiply,
    /// Encrypt a buffer with the counter-mode engine (`ccs_encrypt`).
    CounterModeEncrypt,
    /// Hash a buffer with the counter-mode engine (`ccs_hash`).
    CounterModeHash,
}

impl HardwareEngineOp {
    /// Returns the canonical assembler mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::RandomStore => "xstore",
            Self::CipherEcb => "xcryptecb",
            Self::CipherCbc => "xcryptcbc",
            Self::CipherCtr => "xcryptctr",
            Self::CipherCfb => "xcryptcfb",
            Self::CipherOfb => "xcryptofb",
            Self::HashSha1 => "xsha1",
            Self::HashSha256 => "xsha256",
            Self::HashSha512 => "xsha512",
            Self::MontgomeryMultiply => "montmul",
            Self::CounterModeEncrypt => "ccs_encrypt",
            Self::CounterModeHash => "ccs_hash",
        }
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::RandomStore => "system.hwengine.xstore",
            Self::CipherEcb => "system.hwengine.xcryptecb",
            Self::CipherCbc => "system.hwengine.xcryptcbc",
            Self::CipherCtr => "system.hwengine.xcryptctr",
            Self::CipherCfb => "system.hwengine.xcryptcfb",
            Self::CipherOfb => "system.hwengine.xcryptofb",
            Self::HashSha1 => "system.hwengine.xsha1",
            Self::HashSha256 => "system.hwengine.xsha256",
            Self::HashSha512 => "system.hwengine.xsha512",
            Self::MontgomeryMultiply => "system.hwengine.montmul",
            Self::CounterModeEncrypt => "system.hwengine.ccs_encrypt",
            Self::CounterModeHash => "system.hwengine.ccs_hash",
        }
    }
}

/// Interrupt or exception return named by [`SystemOpKind::InterruptReturn`].
///
/// One variant per operation, so the family reconstructs the instruction's
/// own mnemonic rather than a representative one standing for the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum InterruptReturnOp {
    /// Return to the interrupted context — x86 `iret`/`iretd`/`iretq`, ARM
    /// `rfe*`, AArch64 `eret`. The stack-frame width and the addressing mode
    /// are operand properties, not different operations.
    ExceptionReturn,
    /// Return to the interrupted context, authenticating the restored address
    /// first, so a corrupted frame faults instead of transferring (AArch64
    /// `eretaa` / `eretab`).
    AuthenticatedExceptionReturn,
    /// Return from a machine-level trap, restoring machine-level state
    /// (RISC-V `mret`).
    MachineTrapReturn,
    /// Return from a supervisor-level trap (RISC-V `sret`).
    SupervisorTrapReturn,
    /// Return from a user-level trap (RISC-V `uret`).
    UserTrapReturn,
}

impl InterruptReturnOp {
    /// Returns the neutral display mnemonic for this return.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::ExceptionReturn => "eret",
            Self::AuthenticatedExceptionReturn => "eret.auth",
            Self::MachineTrapReturn => "eret.machine",
            Self::SupervisorTrapReturn => "eret.supervisor",
            Self::UserTrapReturn => "eret.user",
        }
    }

    /// Returns the stable display / fingerprint key for this return.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::ExceptionReturn => "system.iret.exception",
            Self::AuthenticatedExceptionReturn => "system.iret.authenticated",
            Self::MachineTrapReturn => "system.iret.machine",
            Self::SupervisorTrapReturn => "system.iret.supervisor",
            Self::UserTrapReturn => "system.iret.user",
        }
    }
}

/// Breakpoint or undefined-instruction trap named by [`SsaOp::Break`].
///
/// One variant per operation. These share a lowering, not an identity: a
/// renderer reconstructs the instruction from the payload, and two of them
/// never collapse onto one another in a similarity class.
///
/// [`SsaOp::Break`]: crate::ir::ops::def::SsaOp::Break
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum BreakpointOp {
    /// Trap to a debugger — x86 `int3` and `int1`, ARM `bkpt`, AArch64 `brk`,
    /// MIPS `break` and `sdbbp`, RISC-V `ebreak`, CIL `break`. One operation:
    /// raise a breakpoint exception at this address.
    Breakpoint,
    /// Raise an undefined-instruction fault — x86 `ud0`/`ud1`/`ud2`, ARM and
    /// AArch64 `udf`, RISC-V `unimp`.
    UndefinedInstruction,
    /// Halt into an external debugger, which is a distinct exception from a
    /// breakpoint — ARM and AArch64 `hlt`.
    DebugHalt,
}

impl BreakpointOp {
    /// Returns the neutral display mnemonic for this trap.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Breakpoint => "breakpoint",
            Self::UndefinedInstruction => "undefined",
            Self::DebugHalt => "debughalt",
        }
    }

    /// Returns the stable display / fingerprint key for this trap.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Breakpoint => "break.breakpoint",
            Self::UndefinedInstruction => "break.undefined",
            Self::DebugHalt => "break.debughalt",
        }
    }
}

/// Software trap named by [`SystemOpKind::Trap`].
///
/// One variant per operation. These share a lowering, not an identity: a
/// renderer reconstructs the instruction from the payload, and two of them
/// never collapse onto one another in a similarity class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum TrapOp {
    /// Trap to a handler the operand names — x86 `int`, ARM `trap`.
    SoftwareInterrupt,
    /// Trap when the overflow flag is set (`into`).
    OverflowTrap,
    /// Trap when an index is outside its bounds (`bound`).
    BoundsCheck,
    /// Trap when the two compared values are equal. Whether the second is a register or an immediate is an operand property.
    TrapIfEqual,
    /// Trap when the two compared values differ.
    TrapIfNotEqual,
    /// Trap on a signed less-than comparison.
    TrapIfLess,
    /// Trap on an unsigned less-than comparison.
    TrapIfLessUnsigned,
    /// Trap on a signed greater-or-equal comparison.
    TrapIfGreaterOrEqual,
    /// Trap on an unsigned greater-or-equal comparison.
    TrapIfGreaterOrEqualUnsigned,
}

impl TrapOp {
    /// Returns the neutral display mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::SoftwareInterrupt => "trap",
            Self::OverflowTrap => "trap.overflow",
            Self::BoundsCheck => "trap.bounds",
            Self::TrapIfEqual => "trap.eq",
            Self::TrapIfNotEqual => "trap.ne",
            Self::TrapIfLess => "trap.lt",
            Self::TrapIfLessUnsigned => "trap.ltu",
            Self::TrapIfGreaterOrEqual => "trap.ge",
            Self::TrapIfGreaterOrEqualUnsigned => "trap.geu",
        }
    }

    /// Returns the comparison that decides whether this trap fires, and whether
    /// it is unsigned, or `None` when the trap is unconditional.
    ///
    /// The predicate is the whole content of a conditional trap: `teq` transfers
    /// only when its two inputs are equal. Reading it off the variant *name* is
    /// not something a pass can do, so it is exposed as data — the two compared
    /// values are the operation's inputs, and this says what is done with them.
    #[must_use]
    pub const fn condition(self) -> Option<(CmpKind, bool)> {
        match self {
            Self::TrapIfEqual => Some((CmpKind::Eq, false)),
            Self::TrapIfNotEqual => Some((CmpKind::Ne, false)),
            Self::TrapIfLess => Some((CmpKind::Lt, false)),
            Self::TrapIfLessUnsigned => Some((CmpKind::Lt, true)),
            Self::TrapIfGreaterOrEqual => Some((CmpKind::Ge, false)),
            Self::TrapIfGreaterOrEqualUnsigned => Some((CmpKind::Ge, true)),
            Self::SoftwareInterrupt | Self::OverflowTrap | Self::BoundsCheck => None,
        }
    }

    /// Returns `true` when the trap fires only if its comparison holds, so
    /// control continues past it otherwise.
    #[must_use]
    pub const fn is_conditional(self) -> bool {
        self.condition().is_some()
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::SoftwareInterrupt => "system.trap",
            Self::OverflowTrap => "system.trap.overflow",
            Self::BoundsCheck => "system.trap.bounds",
            Self::TrapIfEqual => "system.trap.eq",
            Self::TrapIfNotEqual => "system.trap.ne",
            Self::TrapIfLess => "system.trap.lt",
            Self::TrapIfLessUnsigned => "system.trap.ltu",
            Self::TrapIfGreaterOrEqual => "system.trap.ge",
            Self::TrapIfGreaterOrEqualUnsigned => "system.trap.geu",
        }
    }
}

/// Control- or status-register access named by [`SystemOpKind::ControlRegister`].
///
/// The register file an instruction reaches is a property of the instruction,
/// not of a namespace: `mrc`, `mfc0` and `csrrw` all move a special register to
/// or from a general one, and each is its own operation. There is deliberately
/// **no namespace field**: a namespace beside an operation gives an instruction
/// like `rdmsr` two representations that collide on one `kind_str`, and makes
/// illegal pairings representable. The architecture a variant like
/// [`Self::ReadSystemRegister`] comes from (ARM `mrs`, ARM `mrc`, MIPS `mfc0`)
/// rides [`NativeKindedData::metadata`], which is where target identity
/// belongs.
///
/// **Partition rule against [`MachineStateOp`].** A variant belongs here when
/// the instruction's whole job is moving a value between a general register and
/// a control, system or coprocessor register. An instruction that *interprets*
/// such a register, or that drives a machine-state mechanism through one, is a
/// [`MachineStateOp`]: `lmsw` and `smsw` load and store the machine status word
/// as a mode change, and `rdmsrlist`/`wrmsrns` drive a list engine and a
/// non-serializing write path, so all four stay there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum SysRegOp {
    /// Move a system, coprocessor or control register into a general register — ARM `mrs`/`mrc`, MIPS `mfc0`/`mfc2`, and their doubleword forms.
    ReadSystemRegister,
    /// Read a model-specific register into `EDX:EAX` (`rdmsr`).
    ReadModelSpecific,
    /// Write `EDX:EAX` into a model-specific register (`wrmsr`).
    WriteModelSpecific,
    /// Read an extended control register into `EDX:EAX` (`xgetbv`).
    ReadExtendedControl,
    /// Write `EDX:EAX` into an extended control register (`xsetbv`).
    WriteExtendedControl,
    /// Move a control register into a general register (x86 `mov r, CRn`).
    ReadControlRegister,
    /// Move a general register into a control register (x86 `mov CRn, r`).
    WriteControlRegister,
    /// Move a debug register into a general register (x86 `mov r, DRn`).
    ReadDebugRegister,
    /// Move a general register into a debug register (x86 `mov DRn, r`).
    WriteDebugRegister,
    /// Move a general register into a system, coprocessor or control register — ARM `msr`/`mcr`, MIPS `mtc0`/`mtc2`.
    WriteSystemRegister,
    /// Move a system register into a register pair (ARM `mrrc`).
    ReadSystemRegisterPair,
    /// Move a register pair into a system register (ARM `mcrr`).
    WriteSystemRegisterPair,
    /// Coprocessor data operation with no general-register transfer (ARM `cdp`).
    CoprocessorOperation,
    /// Load a coprocessor register from memory (ARM `ldc`).
    CoprocessorLoad,
    /// Store a coprocessor register to memory (ARM `stc`).
    CoprocessorStore,
    /// Atomically read a system register and write a new value into it (RISC-V `csrrw`). The read and the write are one operation, which a plain read or write is not.
    ExchangeSystemRegister,
    /// Atomically read a system register and set the bits a mask names (RISC-V `csrrs`).
    SetSystemRegisterBits,
    /// Atomically read a system register and clear the bits a mask names (RISC-V `csrrc`).
    ClearSystemRegisterBits,
    /// Extract from an accumulator with a right shift (MIPS DSP `extr`).
    AccumulatorExtract,
    /// Extract from an accumulator, rounding the result.
    AccumulatorExtractRounded,
    /// Extract from an accumulator, rounding then saturating.
    AccumulatorExtractRoundedSaturated,
    /// Extract from an accumulator, saturating the result.
    AccumulatorExtractSaturated,
    /// Extract from an accumulator at the DSP position pointer.
    AccumulatorExtractPosition,
    /// Extract at the position pointer, then decrement it.
    AccumulatorExtractPositionDecrement,
    /// Shift an accumulator (MIPS DSP `shilo`).
    AccumulatorShift,
    /// Copy a register into an accumulator and advance the position pointer.
    AccumulatorCopyAdvance,
    /// Write a multiplier accumulator (Octeon `mtm`). The register the opcode selects rides the operand list.
    WriteMultiplier,
    /// Write a multiplier product register (Octeon `mtp`).
    WriteMultiplierProduct,
}

impl SysRegOp {
    /// Returns the neutral display mnemonic for this access.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::ReadSystemRegister => "sysreg.read",
            Self::ReadModelSpecific => "sysreg.msr.read",
            Self::WriteModelSpecific => "sysreg.msr.write",
            Self::ReadExtendedControl => "sysreg.xcr.read",
            Self::WriteExtendedControl => "sysreg.xcr.write",
            Self::ReadControlRegister => "sysreg.cr.read",
            Self::WriteControlRegister => "sysreg.cr.write",
            Self::ReadDebugRegister => "sysreg.dr.read",
            Self::WriteDebugRegister => "sysreg.dr.write",
            Self::WriteSystemRegister => "sysreg.write",
            Self::ReadSystemRegisterPair => "sysreg.read.pair",
            Self::WriteSystemRegisterPair => "sysreg.write.pair",
            Self::CoprocessorOperation => "sysreg.operation",
            Self::CoprocessorLoad => "sysreg.load",
            Self::CoprocessorStore => "sysreg.store",
            Self::ExchangeSystemRegister => "sysreg.exchange",
            Self::SetSystemRegisterBits => "sysreg.set",
            Self::ClearSystemRegisterBits => "sysreg.clear",
            Self::AccumulatorExtract => "acc.extract",
            Self::AccumulatorExtractRounded => "acc.extract.round",
            Self::AccumulatorExtractRoundedSaturated => "acc.extract.round.sat",
            Self::AccumulatorExtractSaturated => "acc.extract.sat",
            Self::AccumulatorExtractPosition => "acc.extract.pos",
            Self::AccumulatorExtractPositionDecrement => "acc.extract.pos.dec",
            Self::AccumulatorShift => "acc.shift",
            Self::AccumulatorCopyAdvance => "acc.copy.advance",
            Self::WriteMultiplier => "mult.write",
            Self::WriteMultiplierProduct => "mult.write.product",
        }
    }

    /// Returns the stable display / fingerprint key for this access.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::ReadSystemRegister => "system.sysreg.read",
            Self::ReadModelSpecific => "system.sysreg.msr.read",
            Self::WriteModelSpecific => "system.sysreg.msr.write",
            Self::ReadExtendedControl => "system.sysreg.xcr.read",
            Self::WriteExtendedControl => "system.sysreg.xcr.write",
            Self::ReadControlRegister => "system.sysreg.cr.read",
            Self::WriteControlRegister => "system.sysreg.cr.write",
            Self::ReadDebugRegister => "system.sysreg.dr.read",
            Self::WriteDebugRegister => "system.sysreg.dr.write",
            Self::WriteSystemRegister => "system.sysreg.write",
            Self::ReadSystemRegisterPair => "system.sysreg.read.pair",
            Self::WriteSystemRegisterPair => "system.sysreg.write.pair",
            Self::CoprocessorOperation => "system.sysreg.operation",
            Self::CoprocessorLoad => "system.sysreg.load",
            Self::CoprocessorStore => "system.sysreg.store",
            Self::ExchangeSystemRegister => "system.sysreg.exchange",
            Self::SetSystemRegisterBits => "system.sysreg.set",
            Self::ClearSystemRegisterBits => "system.sysreg.clear",
            Self::AccumulatorExtract => "system.acc.extract",
            Self::AccumulatorExtractRounded => "system.acc.extract.round",
            Self::AccumulatorExtractRoundedSaturated => "system.acc.extract.round.sat",
            Self::AccumulatorExtractSaturated => "system.acc.extract.sat",
            Self::AccumulatorExtractPosition => "system.acc.extract.pos",
            Self::AccumulatorExtractPositionDecrement => "system.acc.extract.pos.dec",
            Self::AccumulatorShift => "system.acc.shift",
            Self::AccumulatorCopyAdvance => "system.acc.copy.advance",
            Self::WriteMultiplier => "system.mult.write",
            Self::WriteMultiplierProduct => "system.mult.write.product",
        }
    }

    /// Returns `true` when the operation's first explicit operand is the
    /// destination it defines.
    ///
    /// A lift-shape question — which operand slot the lifter must turn into a
    /// definition — not an effect question. [`Self::effects`] is what says what
    /// the operation does to machine state.
    #[must_use]
    pub const fn writes_destination_operand(self) -> bool {
        matches!(
            self,
            Self::ReadSystemRegister
                | Self::ReadSystemRegisterPair
                | Self::ReadModelSpecific
                | Self::ReadExtendedControl
                | Self::ReadControlRegister
                | Self::ReadDebugRegister
                | Self::ExchangeSystemRegister
                | Self::SetSystemRegisterBits
                | Self::ClearSystemRegisterBits
                | Self::AccumulatorExtract
                | Self::AccumulatorExtractRounded
                | Self::AccumulatorExtractRoundedSaturated
                | Self::AccumulatorExtractSaturated
                | Self::AccumulatorExtractPosition
                | Self::AccumulatorExtractPositionDecrement
        )
    }

    /// Precise effect summary for this access, derived from the operation.
    ///
    /// Not derivable from [`Self::writes_destination_operand`], which answers a
    /// different question: RISC-V `csrrw`, `csrrs` and `csrrc` define a
    /// destination *and* write the CSR, so they are `ReadWrite` rather than
    /// `Read`, and ARM `ldc` defines no destination operand yet only reads.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            // Moves a special register into a general one, or loads one.
            Self::ReadSystemRegister
            | Self::ReadSystemRegisterPair
            | Self::ReadModelSpecific
            | Self::ReadExtendedControl
            | Self::ReadControlRegister
            | Self::ReadDebugRegister
            | Self::CoprocessorLoad
            | Self::AccumulatorExtract
            | Self::AccumulatorExtractRounded
            | Self::AccumulatorExtractRoundedSaturated
            | Self::AccumulatorExtractSaturated
            | Self::AccumulatorExtractPosition
            | Self::AccumulatorExtractPositionDecrement => {
                SsaEffects::new(SsaEffectKind::Read, false)
            }
            // One instruction that both reads the register and writes it back.
            Self::ExchangeSystemRegister
            | Self::SetSystemRegisterBits
            | Self::ClearSystemRegisterBits => SsaEffects::new(SsaEffectKind::ReadWrite, false),
            // Moves a general register into a special one, stores one, or runs a
            // coprocessor operation whose result lands outside the SSA world.
            Self::WriteSystemRegister
            | Self::WriteSystemRegisterPair
            | Self::WriteModelSpecific
            | Self::WriteExtendedControl
            | Self::WriteControlRegister
            | Self::WriteDebugRegister
            | Self::CoprocessorOperation
            | Self::CoprocessorStore
            | Self::AccumulatorShift
            | Self::AccumulatorCopyAdvance
            | Self::WriteMultiplier
            | Self::WriteMultiplierProduct => SsaEffects::new(SsaEffectKind::Write, false),
        }
    }
}

/// The hint named by [`SystemOpKind::Hint`].
///
/// The hint a target's `InstructionFamily::Hint` names — plain prose rather
/// than a link, because `InstructionFamily` belongs to the layer above this
/// crate, which cannot be named from inside it.
///
/// A hint has no architectural data effect, which is exactly why it needs an
/// identity: nothing downstream can recover `endbr64` from `pause` once both
/// are "the instruction that does nothing". [`SystemOpKind::Hint`] is the
/// carrier that puts that identity into an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum HintOp {
    /// Do nothing.
    NoOperation,
    /// Hint that this thread is spinning and may yield the pipeline — x86 `pause`, ARM `yield`. One operation.
    SpinWait,
    /// Mark a valid indirect-branch target — x86 CET `endbr32`/`endbr64`, AArch64 `bti`. One operation, and the reason a forward-edge integrity scheme is visible at all.
    IndirectBranchLandingPad,
    /// Wait until an event is signalled.
    WaitForEvent,
    /// Wait for an event or a deadline.
    WaitForEventTimed,
    /// Signal an event to every processor.
    SendEvent,
    /// Signal an event to this processor only.
    SendEventLocal,
    /// Prefetch data into the cache. The level a hint requests is advisory and rides the operand list.
    Prefetch,
    /// Prefetch anticipating a write.
    PrefetchForWrite,
    /// Prefetch instructions rather than data.
    PrefetchInstruction,
    /// Prefetch with non-temporal locality.
    PrefetchNonTemporal,
    /// Vector prefetch.
    PrefetchVector,
    /// Hint that gathering loads and stores is unprofitable.
    DataGatheringHint,
    /// Synchronize the statistical profiling buffer.
    ProfilingSynchronizationBarrier,
    /// Clear execution hazards.
    ExecutionHazardBarrier,
    /// Hint to the debug logic.
    DebugHint,
    /// Set up conditional execution for the following block (Thumb `it`).
    ConditionalExecutionBlock,
    /// An architected hint selected by its immediate, with no further identity in the encoding.
    GenericHint,
    /// An MPX bounds operation. Removed from the architecture and executed as a no-op, so all seven encodings do the same thing: nothing.
    RemovedBoundsOperation,
}

impl HintOp {
    /// Returns the neutral display mnemonic for this hint.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::NoOperation => "nop",
            Self::SpinWait => "spinwait",
            Self::IndirectBranchLandingPad => "landingpad",
            Self::WaitForEvent => "wfe",
            Self::WaitForEventTimed => "wfe.timed",
            Self::SendEvent => "sev",
            Self::SendEventLocal => "sev.local",
            Self::Prefetch => "prefetch",
            Self::PrefetchForWrite => "prefetch.write",
            Self::PrefetchInstruction => "prefetch.code",
            Self::PrefetchNonTemporal => "prefetch.nontemporal",
            Self::PrefetchVector => "prefetch.vector",
            Self::DataGatheringHint => "dgh",
            Self::ProfilingSynchronizationBarrier => "psb",
            Self::ExecutionHazardBarrier => "ehb",
            Self::DebugHint => "dbg",
            Self::ConditionalExecutionBlock => "itblock",
            Self::GenericHint => "hint",
            Self::RemovedBoundsOperation => "bounds.removed",
        }
    }

    /// Returns the stable display / fingerprint key for this hint.
    ///
    /// Prefixed `system.` like every other [`SystemOpKind`] payload, so the one
    /// flat namespace [`SsaOp::opcode_name`] hands its keys to is grouped by
    /// the carrier a key comes from.
    ///
    /// [`SsaOp::opcode_name`]: crate::ir::ops::def::SsaOp::opcode_name
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::NoOperation => "system.hint.nop",
            Self::SpinWait => "system.hint.spinwait",
            Self::IndirectBranchLandingPad => "system.hint.landingpad",
            Self::WaitForEvent => "system.hint.wfe",
            Self::WaitForEventTimed => "system.hint.wfe.timed",
            Self::SendEvent => "system.hint.sev",
            Self::SendEventLocal => "system.hint.sev.local",
            Self::Prefetch => "system.hint.prefetch",
            Self::PrefetchForWrite => "system.hint.prefetch.write",
            Self::PrefetchInstruction => "system.hint.prefetch.code",
            Self::PrefetchNonTemporal => "system.hint.prefetch.nontemporal",
            Self::PrefetchVector => "system.hint.prefetch.vector",
            Self::DataGatheringHint => "system.hint.dgh",
            Self::ProfilingSynchronizationBarrier => "system.hint.psb",
            Self::ExecutionHazardBarrier => "system.hint.ehb",
            Self::DebugHint => "system.hint.dbg",
            Self::ConditionalExecutionBlock => "system.hint.itblock",
            Self::GenericHint => "system.hint.hint",
            Self::RemovedBoundsOperation => "system.hint.bounds.removed",
        }
    }

    /// Precise effect summary for this hint.
    ///
    /// Never [`SsaEffectKind::Pure`]. A prefetch reads memory it names, so it
    /// is a `Read`. Every other hint is [`SsaEffectKind::Opaque`], because the
    /// two that matter most are exactly the two a pure classification would
    /// delete: `endbr64` is a control-flow-integrity landing pad, and
    /// `pause`/`wfe` are synchronization primitives whose absence is a spin
    /// loop that never yields.
    ///
    /// The location is [`MemoryEffectLocation::None`] rather than the `Opaque`
    /// default of [`MemoryEffectLocation::Unknown`]: a hint that touches no
    /// memory must be non-removable *without* being a memory barrier.
    /// `MemorySsa::classify_memory_operation` turns an `Opaque` op with an
    /// unknown location into a `MemoryOp::Barrier`, which would make every
    /// lifted `nop` — instruction padding, an alignment run between two basic
    /// blocks — block store forwarding across it.
    ///
    /// [`MemoryEffectLocation::None`]: super::MemoryEffectLocation::None
    /// [`MemoryEffectLocation::Unknown`]: super::MemoryEffectLocation::Unknown
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::Prefetch
            | Self::PrefetchForWrite
            | Self::PrefetchInstruction
            | Self::PrefetchNonTemporal
            | Self::PrefetchVector => SsaEffects::new(SsaEffectKind::Read, false),
            Self::NoOperation
            | Self::SpinWait
            | Self::IndirectBranchLandingPad
            | Self::WaitForEvent
            | Self::WaitForEventTimed
            | Self::SendEvent
            | Self::SendEventLocal
            | Self::DataGatheringHint
            | Self::ProfilingSynchronizationBarrier
            | Self::ExecutionHazardBarrier
            | Self::DebugHint
            | Self::ConditionalExecutionBlock
            | Self::GenericHint
            | Self::RemovedBoundsOperation => SsaEffects::new(SsaEffectKind::Opaque, false)
                .with_memory(MemoryEffectLocation::None),
        }
    }
}

/// The machine-state operation named by [`SystemOpKind::MachineState`].
///
/// One variant per operation, so a renderer can reconstruct the native mnemonic
/// from the kind alone and two different instructions never share a similarity
/// class. The names describe the role rather than any one assembler's spelling;
/// the spelling is [`Self::mnemonic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum MachineStateOp {
    // Descriptor tables and the machine status word.
    /// Load the global descriptor table register (`lgdt`).
    LoadGlobalDescriptorTable,
    /// Store the global descriptor table register (`sgdt`).
    StoreGlobalDescriptorTable,
    /// Load the interrupt descriptor table register (`lidt`).
    LoadInterruptDescriptorTable,
    /// Store the interrupt descriptor table register (`sidt`).
    StoreInterruptDescriptorTable,
    /// Load the local descriptor table register (`lldt`).
    LoadLocalDescriptorTable,
    /// Store the local descriptor table register (`sldt`).
    StoreLocalDescriptorTable,
    /// Load the task register (`ltr`).
    LoadTaskRegister,
    /// Store the task register (`str`).
    StoreTaskRegister,
    /// Load the machine status word — `CR0`'s low half (`lmsw`).
    LoadMachineStatusWord,
    /// Store the machine status word (`smsw`).
    StoreMachineStatusWord,
    /// Clear the task-switched flag in `CR0` (`clts`).
    ClearTaskSwitched,
    // Segment-descriptor queries. Each samples a descriptor and reports through its
    // destination or the zero flag.
    /// Load a segment descriptor's access rights (`lar`).
    ReadAccessRights,
    /// Load a segment descriptor's limit (`lsl`).
    ReadSegmentLimit,
    /// Verify a segment is readable at the current privilege (`verr`).
    VerifySegmentReadable,
    /// Verify a segment is writable at the current privilege (`verw`).
    VerifySegmentWritable,
    /// Adjust a selector's requested privilege level (`arpl`).
    AdjustRequestedPrivilege,
    // Far-pointer loads: a segment selector and an offset read together.
    /// Load a far pointer into `DS` and a general register (`lds`).
    LoadFarPointerDs,
    /// Load a far pointer into `ES` and a general register (`les`).
    LoadFarPointerEs,
    /// Load a far pointer into `FS` and a general register (`lfs`).
    LoadFarPointerFs,
    /// Load a far pointer into `GS` and a general register (`lgs`).
    LoadFarPointerGs,
    /// Load a far pointer into `SS` and a general register (`lss`).
    LoadFarPointerSs,
    // Thread-pointer and segment-base registers.
    /// Read the `FS` segment base (`rdfsbase`).
    ReadFsBase,
    /// Write the `FS` segment base (`wrfsbase`).
    WriteFsBase,
    /// Read the `GS` segment base (`rdgsbase`).
    ReadGsBase,
    /// Write the `GS` segment base (`wrgsbase`).
    WriteGsBase,
    /// Exchange the `GS` base with `IA32_KERNEL_GS_BASE` (`swapgs`).
    SwapGsBase,
    /// Load `IA32_KERNEL_GS_BASE` from a selector (`lkgs`).
    LoadKernelGsBase,
    // Port I/O. Every form both samples a device and may change its state.
    /// Read from an I/O port into the accumulator (`in`).
    PortIn,
    /// Write the accumulator to an I/O port (`out`).
    PortOut,
    /// Read a byte from an I/O port to `ES:[rDI]` (`insb`).
    PortInStringByte,
    /// Read a word from an I/O port to `ES:[rDI]` (`insw`).
    PortInStringWord,
    /// Read a doubleword from an I/O port to `ES:[rDI]` (`insd`).
    PortInStringDword,
    /// Write a byte from `DS:[rSI]` to an I/O port (`outsb`).
    PortOutStringByte,
    /// Write a word from `DS:[rSI]` to an I/O port (`outsw`).
    PortOutStringWord,
    /// Write a doubleword from `DS:[rSI]` to an I/O port (`outsd`).
    PortOutStringDword,
    // Control-flow-enforcement shadow stack.
    /// Advance the shadow stack pointer by a doubleword count (`incsspd`).
    ShadowStackIncrementDword,
    /// Advance the shadow stack pointer by a quadword count (`incsspq`).
    ShadowStackIncrementQword,
    /// Read the low half of the shadow stack pointer (`rdsspd`).
    ShadowStackReadDword,
    /// Read the shadow stack pointer (`rdsspq`).
    ShadowStackReadQword,
    /// Write a doubleword to the shadow stack (`wrssd`).
    ShadowStackWriteDword,
    /// Write a quadword to the shadow stack (`wrssq`).
    ShadowStackWriteQword,
    /// Write a doubleword to the user shadow stack (`wrussd`).
    ShadowStackWriteUserDword,
    /// Write a quadword to the user shadow stack (`wrussq`).
    ShadowStackWriteUserQword,
    /// Mark the current shadow stack busy (`setssbsy`).
    ShadowStackMarkBusy,
    /// Clear a shadow stack's busy flag (`clrssbsy`).
    ShadowStackClearBusy,
    /// Restore a saved shadow stack pointer (`rstorssp`).
    ShadowStackRestore,
    /// Save the previous shadow stack pointer (`saveprevssp`).
    ShadowStackSavePrevious,
    // User interrupts.
    /// Clear the user-interrupt flag (`clui`).
    UserInterruptDisable,
    /// Set the user-interrupt flag (`stui`).
    UserInterruptEnable,
    /// Report the user-interrupt flag in `CF` (`testui`).
    UserInterruptTest,
    /// Return from a user-interrupt handler (`uiret`) — a control transfer.
    UserInterruptReturn,
    /// Send a user interprocessor interrupt (`senduipi`).
    UserInterruptSend,
    // Halt, monitor and wait.
    /// Halt the processor until an interrupt arrives (`hlt`).
    Halt,
    /// Arm address monitoring (`monitor`).
    MonitorAddress,
    /// Arm address monitoring for the timed wait (`monitorx`).
    MonitorAddressExtended,
    /// Arm address monitoring at user privilege (`umonitor`).
    MonitorAddressUser,
    /// Wait for a monitored write (`mwait`).
    MonitorWait,
    /// Wait for a monitored write or a timeout (`mwaitx`).
    MonitorWaitTimed,
    /// Wait at user privilege for a monitored write or a timeout (`umwait`).
    MonitorWaitUser,
    /// Pause until a deadline, with no armed monitor (`tpause`).
    TimedPause,
    // Protection keys and processor identity.
    /// Read the user protection-key rights register (`rdpkru`).
    ReadProtectionKeyRights,
    /// Write the user protection-key rights register (`wrpkru`).
    WriteProtectionKeyRights,
    /// Read the processor identifier (`rdpid`).
    ReadProcessorId,
    /// Read a processor power-management register (`rdpru`).
    ReadProcessorPower,
    // Model-specific register lists.
    /// Read a list of model-specific registers (`rdmsrlist`).
    ReadMsrList,
    /// Write a list of model-specific registers (`wrmsrlist`).
    WriteMsrList,
    /// Write a model-specific register without serializing (`wrmsrns`).
    WriteMsrNonSerializing,
    // Key locker.
    /// Load the internal wrapping key (`loadiwkey`).
    KeyLockerLoadInternalKey,
    /// Wrap a 128-bit key handle (`encodekey128`).
    KeyLockerEncodeKey128,
    /// Wrap a 256-bit key handle (`encodekey256`).
    KeyLockerEncodeKey256,
    // Platform and memory-encryption management.
    /// Configure a platform security feature (`pconfig`).
    PlatformConfigure,
    /// Bind a platform key (`pbndkb`).
    PlatformBindKey,
    /// Split a large private page in the reverse map (`psmash`).
    PlatformSmashPage,
    /// Validate a guest private page (`pvalidate`).
    PlatformValidatePage,
    /// Adjust a reverse-map entry's permissions (`rmpadjust`).
    ReverseMapAdjust,
    /// Query a reverse-map entry (`rmpquery`).
    ReverseMapQuery,
    /// Update a reverse-map entry (`rmpupdate`).
    ReverseMapUpdate,
    /// Reset selected prediction history (`hreset`).
    HistoryReset,
    // Direct and enqueue stores — memory writes that bypass the cache hierarchy.
    /// Store a doubleword or quadword directly (`movdiri`).
    DirectStoreWord,
    /// Store a 64-byte block directly (`movdir64b`).
    DirectStoreBlock,
    /// Enqueue a command to a device, reporting status in `ZF` (`enqcmd`).
    EnqueueCommand,
    /// Enqueue a supervisor command to a device (`enqcmds`).
    EnqueueCommandSupervisor,
    // Trace and system-management mode.
    /// Write a value into the processor trace packet stream (`ptwrite`).
    TraceWrite,
    /// Resume from system-management mode (`rsm`) — a control transfer.
    ResumeFromSystemManagement,
    // AMD lightweight profiling.
    /// Load the lightweight-profiling control block (`llwpcb`).
    ProfilingLoadControlBlock,
    /// Store the lightweight-profiling control block address (`slwpcb`).
    ProfilingStoreControlBlock,
    /// Insert a value into the profiling ring buffer (`lwpval`).
    ProfilingInsertValue,
    /// Insert an event record into the profiling ring buffer (`lwpins`).
    ProfilingInsertRecord,
    // Undocumented and obsolete vendor instructions.
    ///
    // None is emitted by any current toolchain, and in practice they appear
    // where a linear sweep walked into data. Naming each one is still what lets
    // a listing say which byte pattern was decoded, rather than collapsing the
    // whole set into one label.
    /// VIA alternate instruction set entry (`altinst`).
    VendorAlternateInstruction,
    /// Cyrix branch-buffer 0 reset (`bb0_reset`).
    VendorBranchBuffer0Reset,
    /// Cyrix branch-buffer 1 reset (`bb1_reset`).
    VendorBranchBuffer1Reset,
    /// Cyrix configuration-register read (`cpu_read`).
    VendorCpuRead,
    /// Cyrix configuration-register write (`cpu_write`).
    VendorCpuWrite,
    /// Cyrix debug-mode interrupt (`dmint`).
    VendorDebugInterrupt,
    /// Cyrix software system-management interrupt (`smint`).
    VendorSoftwareSmi,
    /// Cyrix round-to-nearest-integer (`frinear`).
    VendorRoundToNearestInteger,
    /// 80386 B-step insert bit string (`ibts`).
    VendorInsertBitString,
    /// 80386 B-step extract bit string (`xbts`).
    VendorExtractBitString,
    /// 80286/80386 load-all-registers (`loadall`).
    VendorLoadAllRegisters,
    /// Undocumented store-all-registers (`storeall`).
    VendorStoreAllRegisters,
    /// Cyrix read model-specific state (`rdm`).
    VendorReadModelRegister,
    /// Cyrix read SMM header register (`rdshr`).
    VendorReadShadowRegister,
    /// Cyrix write SMM header register (`wrshr`).
    VendorWriteShadowRegister,
    /// Undocumented debug-register read (`rdudbg`).
    VendorReadDebugRegister,
    /// Undocumented debug-register write (`wrudbg`).
    VendorWriteDebugRegister,
    /// Cyrix restore segment register and descriptor (`rsdc`).
    VendorRestoreDataSegment,
    /// Cyrix save segment register and descriptor (`svdc`).
    VendorSaveDataSegment,
    /// Cyrix restore local descriptor table (`rsldt`).
    VendorRestoreLocalDescriptor,
    /// Cyrix save local descriptor table (`svldt`).
    VendorSaveLocalDescriptor,
    /// Cyrix restore task-state register (`rsts`).
    VendorRestoreTaskState,
    /// Cyrix save task-state register (`svts`).
    VendorSaveTaskState,
    /// Cyrix set protected-mode flat addressing (`spflt`).
    VendorSetProtectedFlat,
    /// 80386 unprivileged move to or from a debug agent (`umov`).
    VendorUnprivilegedMove,
    /// An undocumented opcode the decoder recognises but does not name (`undoc`).
    VendorUndocumented,
    // ARM A32/T32 processor state.
    /// Change processor state — interrupt masks and mode (`cps`).
    ProcessorStateChange,
    /// Set the data endianness bit (`setend`).
    SetEndianness,
    /// Set the privileged-access-never state bit (`setpan`).
    SetPrivilegedAccessNever,
    /// Mark a valid secure-state entry point (`sg`).
    SecureGateway,
    /// Store return state, decrement after (`srsda`).
    StoreReturnStateDecrementAfter,
    /// Store return state, decrement before (`srsdb`).
    StoreReturnStateDecrementBefore,
    /// Store return state, increment after (`srsia`).
    StoreReturnStateIncrementAfter,
    /// Store return state, increment before (`srsib`).
    StoreReturnStateIncrementBefore,
    /// Test an address's security attributes (`tt`).
    TestTarget,
    /// Test an address in the alternate security domain (`tta`).
    TestTargetAlternate,
    /// Test an address, alternate domain, unprivileged (`ttat`).
    TestTargetAlternateUnprivileged,
    /// Test an address at unprivileged access (`ttt`).
    TestTargetUnprivileged,
    // AArch64 system state.
    /// Translate an address and report through `PAR_EL1` (`at`).
    AddressTranslate,
    /// Inject or read the branch-record buffer (`brb`).
    BranchRecordBuffer,
    /// Debug change to EL1 (`dcps1`).
    DebugChangeStateEl1,
    /// Debug change to EL2 (`dcps2`).
    DebugChangeStateEl2,
    /// Debug change to EL3 (`dcps3`).
    DebugChangeStateEl3,
    /// Debug restore of the saved processor state (`drps`) — a control transfer.
    DebugRestoreState,
    /// Enter SME streaming or `ZA` state (`smstart`).
    StreamingModeStart,
    /// Leave SME streaming or `ZA` state (`smstop`).
    StreamingModeStop,
    /// Generic system instruction with no result (`sys`).
    SystemInstruction,
    /// Generic system instruction returning a result (`sysl`).
    SystemInstructionRead,
    // MIPS interrupt masking. Not x86 `cli`/`sti`: these optionally *return*
    // the prior status register, so their effect is not confined to a flag bit
    // and they stay here rather than joining `FlagAdjustKind`.
    /// Disable interrupts, optionally returning the prior status (`di`).
    InterruptDisable,
    /// Enable interrupts, optionally returning the prior status (`ei`).
    InterruptEnable,
    /// Stall until an interrupt is pending — ARM and AArch64 `wfi`, RISC-V
    /// `wfi`, MIPS `wait`. One operation.
    WaitForInterrupt,
    /// Stall until an interrupt is pending or a deadline passes (AArch64
    /// `wfit`).
    WaitForInterruptTimed,
}

impl MachineStateOp {
    /// Returns the canonical assembler mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::LoadGlobalDescriptorTable => "lgdt",
            Self::StoreGlobalDescriptorTable => "sgdt",
            Self::LoadInterruptDescriptorTable => "lidt",
            Self::StoreInterruptDescriptorTable => "sidt",
            Self::LoadLocalDescriptorTable => "lldt",
            Self::StoreLocalDescriptorTable => "sldt",
            Self::LoadTaskRegister => "ltr",
            Self::StoreTaskRegister => "str",
            Self::LoadMachineStatusWord => "lmsw",
            Self::StoreMachineStatusWord => "smsw",
            Self::ClearTaskSwitched => "clts",
            Self::ReadAccessRights => "lar",
            Self::ReadSegmentLimit => "lsl",
            Self::VerifySegmentReadable => "verr",
            Self::VerifySegmentWritable => "verw",
            Self::AdjustRequestedPrivilege => "arpl",
            Self::LoadFarPointerDs => "lds",
            Self::LoadFarPointerEs => "les",
            Self::LoadFarPointerFs => "lfs",
            Self::LoadFarPointerGs => "lgs",
            Self::LoadFarPointerSs => "lss",
            Self::ReadFsBase => "rdfsbase",
            Self::WriteFsBase => "wrfsbase",
            Self::ReadGsBase => "rdgsbase",
            Self::WriteGsBase => "wrgsbase",
            Self::SwapGsBase => "swapgs",
            Self::LoadKernelGsBase => "lkgs",
            Self::PortIn => "in",
            Self::PortOut => "out",
            Self::PortInStringByte => "insb",
            Self::PortInStringWord => "insw",
            Self::PortInStringDword => "insd",
            Self::PortOutStringByte => "outsb",
            Self::PortOutStringWord => "outsw",
            Self::PortOutStringDword => "outsd",
            Self::ShadowStackIncrementDword => "incsspd",
            Self::ShadowStackIncrementQword => "incsspq",
            Self::ShadowStackReadDword => "rdsspd",
            Self::ShadowStackReadQword => "rdsspq",
            Self::ShadowStackWriteDword => "wrssd",
            Self::ShadowStackWriteQword => "wrssq",
            Self::ShadowStackWriteUserDword => "wrussd",
            Self::ShadowStackWriteUserQword => "wrussq",
            Self::ShadowStackMarkBusy => "setssbsy",
            Self::ShadowStackClearBusy => "clrssbsy",
            Self::ShadowStackRestore => "rstorssp",
            Self::ShadowStackSavePrevious => "saveprevssp",
            Self::UserInterruptDisable => "clui",
            Self::UserInterruptEnable => "stui",
            Self::UserInterruptTest => "testui",
            Self::UserInterruptReturn => "uiret",
            Self::UserInterruptSend => "senduipi",
            Self::Halt => "hlt",
            Self::MonitorAddress => "monitor",
            Self::MonitorAddressExtended => "monitorx",
            Self::MonitorAddressUser => "umonitor",
            Self::MonitorWait => "mwait",
            Self::MonitorWaitTimed => "mwaitx",
            Self::MonitorWaitUser => "umwait",
            Self::TimedPause => "tpause",
            Self::ReadProtectionKeyRights => "rdpkru",
            Self::WriteProtectionKeyRights => "wrpkru",
            Self::ReadProcessorId => "rdpid",
            Self::ReadProcessorPower => "rdpru",
            Self::ReadMsrList => "rdmsrlist",
            Self::WriteMsrList => "wrmsrlist",
            Self::WriteMsrNonSerializing => "wrmsrns",
            Self::KeyLockerLoadInternalKey => "loadiwkey",
            Self::KeyLockerEncodeKey128 => "encodekey128",
            Self::KeyLockerEncodeKey256 => "encodekey256",
            Self::PlatformConfigure => "pconfig",
            Self::PlatformBindKey => "pbndkb",
            Self::PlatformSmashPage => "psmash",
            Self::PlatformValidatePage => "pvalidate",
            Self::ReverseMapAdjust => "rmpadjust",
            Self::ReverseMapQuery => "rmpquery",
            Self::ReverseMapUpdate => "rmpupdate",
            Self::HistoryReset => "hreset",
            Self::DirectStoreWord => "movdiri",
            Self::DirectStoreBlock => "movdir64b",
            Self::EnqueueCommand => "enqcmd",
            Self::EnqueueCommandSupervisor => "enqcmds",
            Self::TraceWrite => "ptwrite",
            Self::ResumeFromSystemManagement => "rsm",
            Self::ProfilingLoadControlBlock => "llwpcb",
            Self::ProfilingStoreControlBlock => "slwpcb",
            Self::ProfilingInsertValue => "lwpval",
            Self::ProfilingInsertRecord => "lwpins",
            Self::VendorAlternateInstruction => "altinst",
            Self::VendorBranchBuffer0Reset => "bb0_reset",
            Self::VendorBranchBuffer1Reset => "bb1_reset",
            Self::VendorCpuRead => "cpu_read",
            Self::VendorCpuWrite => "cpu_write",
            Self::VendorDebugInterrupt => "dmint",
            Self::VendorSoftwareSmi => "smint",
            Self::VendorRoundToNearestInteger => "frinear",
            Self::VendorInsertBitString => "ibts",
            Self::VendorExtractBitString => "xbts",
            Self::VendorLoadAllRegisters => "loadall",
            Self::VendorStoreAllRegisters => "storeall",
            Self::VendorReadModelRegister => "rdm",
            Self::VendorReadShadowRegister => "rdshr",
            Self::VendorWriteShadowRegister => "wrshr",
            Self::VendorReadDebugRegister => "rdudbg",
            Self::VendorWriteDebugRegister => "wrudbg",
            Self::VendorRestoreDataSegment => "rsdc",
            Self::VendorSaveDataSegment => "svdc",
            Self::VendorRestoreLocalDescriptor => "rsldt",
            Self::VendorSaveLocalDescriptor => "svldt",
            Self::VendorRestoreTaskState => "rsts",
            Self::VendorSaveTaskState => "svts",
            Self::VendorSetProtectedFlat => "spflt",
            Self::VendorUnprivilegedMove => "umov",
            Self::VendorUndocumented => "undoc",
            Self::ProcessorStateChange => "cps",
            Self::SetEndianness => "setend",
            Self::SetPrivilegedAccessNever => "setpan",
            Self::SecureGateway => "sg",
            Self::StoreReturnStateDecrementAfter => "srsda",
            Self::StoreReturnStateDecrementBefore => "srsdb",
            Self::StoreReturnStateIncrementAfter => "srsia",
            Self::StoreReturnStateIncrementBefore => "srsib",
            Self::TestTarget => "tt",
            Self::TestTargetAlternate => "tta",
            Self::TestTargetAlternateUnprivileged => "ttat",
            Self::TestTargetUnprivileged => "ttt",
            Self::AddressTranslate => "at",
            Self::BranchRecordBuffer => "brb",
            Self::DebugChangeStateEl1 => "dcps1",
            Self::DebugChangeStateEl2 => "dcps2",
            Self::DebugChangeStateEl3 => "dcps3",
            Self::DebugRestoreState => "drps",
            Self::StreamingModeStart => "smstart",
            Self::StreamingModeStop => "smstop",
            Self::SystemInstruction => "sys",
            Self::SystemInstructionRead => "sysl",
            Self::InterruptDisable => "di",
            Self::InterruptEnable => "ei",
            Self::WaitForInterrupt => "wait.interrupt",
            Self::WaitForInterruptTimed => "wait.interrupt.timed",
        }
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::LoadGlobalDescriptorTable => "system.state.lgdt",
            Self::StoreGlobalDescriptorTable => "system.state.sgdt",
            Self::LoadInterruptDescriptorTable => "system.state.lidt",
            Self::StoreInterruptDescriptorTable => "system.state.sidt",
            Self::LoadLocalDescriptorTable => "system.state.lldt",
            Self::StoreLocalDescriptorTable => "system.state.sldt",
            Self::LoadTaskRegister => "system.state.ltr",
            Self::StoreTaskRegister => "system.state.str",
            Self::LoadMachineStatusWord => "system.state.lmsw",
            Self::StoreMachineStatusWord => "system.state.smsw",
            Self::ClearTaskSwitched => "system.state.clts",
            Self::ReadAccessRights => "system.state.lar",
            Self::ReadSegmentLimit => "system.state.lsl",
            Self::VerifySegmentReadable => "system.state.verr",
            Self::VerifySegmentWritable => "system.state.verw",
            Self::AdjustRequestedPrivilege => "system.state.arpl",
            Self::LoadFarPointerDs => "system.state.lds",
            Self::LoadFarPointerEs => "system.state.les",
            Self::LoadFarPointerFs => "system.state.lfs",
            Self::LoadFarPointerGs => "system.state.lgs",
            Self::LoadFarPointerSs => "system.state.lss",
            Self::ReadFsBase => "system.state.rdfsbase",
            Self::WriteFsBase => "system.state.wrfsbase",
            Self::ReadGsBase => "system.state.rdgsbase",
            Self::WriteGsBase => "system.state.wrgsbase",
            Self::SwapGsBase => "system.state.swapgs",
            Self::LoadKernelGsBase => "system.state.lkgs",
            Self::PortIn => "system.state.in",
            Self::PortOut => "system.state.out",
            Self::PortInStringByte => "system.state.insb",
            Self::PortInStringWord => "system.state.insw",
            Self::PortInStringDword => "system.state.insd",
            Self::PortOutStringByte => "system.state.outsb",
            Self::PortOutStringWord => "system.state.outsw",
            Self::PortOutStringDword => "system.state.outsd",
            Self::ShadowStackIncrementDword => "system.state.incsspd",
            Self::ShadowStackIncrementQword => "system.state.incsspq",
            Self::ShadowStackReadDword => "system.state.rdsspd",
            Self::ShadowStackReadQword => "system.state.rdsspq",
            Self::ShadowStackWriteDword => "system.state.wrssd",
            Self::ShadowStackWriteQword => "system.state.wrssq",
            Self::ShadowStackWriteUserDword => "system.state.wrussd",
            Self::ShadowStackWriteUserQword => "system.state.wrussq",
            Self::ShadowStackMarkBusy => "system.state.setssbsy",
            Self::ShadowStackClearBusy => "system.state.clrssbsy",
            Self::ShadowStackRestore => "system.state.rstorssp",
            Self::ShadowStackSavePrevious => "system.state.saveprevssp",
            Self::UserInterruptDisable => "system.state.clui",
            Self::UserInterruptEnable => "system.state.stui",
            Self::UserInterruptTest => "system.state.testui",
            Self::UserInterruptReturn => "system.state.uiret",
            Self::UserInterruptSend => "system.state.senduipi",
            Self::Halt => "system.state.hlt",
            Self::MonitorAddress => "system.state.monitor",
            Self::MonitorAddressExtended => "system.state.monitorx",
            Self::MonitorAddressUser => "system.state.umonitor",
            Self::MonitorWait => "system.state.mwait",
            Self::MonitorWaitTimed => "system.state.mwaitx",
            Self::MonitorWaitUser => "system.state.umwait",
            Self::TimedPause => "system.state.tpause",
            Self::ReadProtectionKeyRights => "system.state.rdpkru",
            Self::WriteProtectionKeyRights => "system.state.wrpkru",
            Self::ReadProcessorId => "system.state.rdpid",
            Self::ReadProcessorPower => "system.state.rdpru",
            Self::ReadMsrList => "system.state.rdmsrlist",
            Self::WriteMsrList => "system.state.wrmsrlist",
            Self::WriteMsrNonSerializing => "system.state.wrmsrns",
            Self::KeyLockerLoadInternalKey => "system.state.loadiwkey",
            Self::KeyLockerEncodeKey128 => "system.state.encodekey128",
            Self::KeyLockerEncodeKey256 => "system.state.encodekey256",
            Self::PlatformConfigure => "system.state.pconfig",
            Self::PlatformBindKey => "system.state.pbndkb",
            Self::PlatformSmashPage => "system.state.psmash",
            Self::PlatformValidatePage => "system.state.pvalidate",
            Self::ReverseMapAdjust => "system.state.rmpadjust",
            Self::ReverseMapQuery => "system.state.rmpquery",
            Self::ReverseMapUpdate => "system.state.rmpupdate",
            Self::HistoryReset => "system.state.hreset",
            Self::DirectStoreWord => "system.state.movdiri",
            Self::DirectStoreBlock => "system.state.movdir64b",
            Self::EnqueueCommand => "system.state.enqcmd",
            Self::EnqueueCommandSupervisor => "system.state.enqcmds",
            Self::TraceWrite => "system.state.ptwrite",
            Self::ResumeFromSystemManagement => "system.state.rsm",
            Self::ProfilingLoadControlBlock => "system.state.llwpcb",
            Self::ProfilingStoreControlBlock => "system.state.slwpcb",
            Self::ProfilingInsertValue => "system.state.lwpval",
            Self::ProfilingInsertRecord => "system.state.lwpins",
            Self::VendorAlternateInstruction => "system.state.altinst",
            Self::VendorBranchBuffer0Reset => "system.state.bb0_reset",
            Self::VendorBranchBuffer1Reset => "system.state.bb1_reset",
            Self::VendorCpuRead => "system.state.cpu_read",
            Self::VendorCpuWrite => "system.state.cpu_write",
            Self::VendorDebugInterrupt => "system.state.dmint",
            Self::VendorSoftwareSmi => "system.state.smint",
            Self::VendorRoundToNearestInteger => "system.state.frinear",
            Self::VendorInsertBitString => "system.state.ibts",
            Self::VendorExtractBitString => "system.state.xbts",
            Self::VendorLoadAllRegisters => "system.state.loadall",
            Self::VendorStoreAllRegisters => "system.state.storeall",
            Self::VendorReadModelRegister => "system.state.rdm",
            Self::VendorReadShadowRegister => "system.state.rdshr",
            Self::VendorWriteShadowRegister => "system.state.wrshr",
            Self::VendorReadDebugRegister => "system.state.rdudbg",
            Self::VendorWriteDebugRegister => "system.state.wrudbg",
            Self::VendorRestoreDataSegment => "system.state.rsdc",
            Self::VendorSaveDataSegment => "system.state.svdc",
            Self::VendorRestoreLocalDescriptor => "system.state.rsldt",
            Self::VendorSaveLocalDescriptor => "system.state.svldt",
            Self::VendorRestoreTaskState => "system.state.rsts",
            Self::VendorSaveTaskState => "system.state.svts",
            Self::VendorSetProtectedFlat => "system.state.spflt",
            Self::VendorUnprivilegedMove => "system.state.umov",
            Self::VendorUndocumented => "system.state.undoc",
            Self::ProcessorStateChange => "system.state.cps",
            Self::SetEndianness => "system.state.setend",
            Self::SetPrivilegedAccessNever => "system.state.setpan",
            Self::SecureGateway => "system.state.sg",
            Self::StoreReturnStateDecrementAfter => "system.state.srsda",
            Self::StoreReturnStateDecrementBefore => "system.state.srsdb",
            Self::StoreReturnStateIncrementAfter => "system.state.srsia",
            Self::StoreReturnStateIncrementBefore => "system.state.srsib",
            Self::TestTarget => "system.state.tt",
            Self::TestTargetAlternate => "system.state.tta",
            Self::TestTargetAlternateUnprivileged => "system.state.ttat",
            Self::TestTargetUnprivileged => "system.state.ttt",
            Self::AddressTranslate => "system.state.at",
            Self::BranchRecordBuffer => "system.state.brb",
            Self::DebugChangeStateEl1 => "system.state.dcps1",
            Self::DebugChangeStateEl2 => "system.state.dcps2",
            Self::DebugChangeStateEl3 => "system.state.dcps3",
            Self::DebugRestoreState => "system.state.drps",
            Self::StreamingModeStart => "system.state.smstart",
            Self::StreamingModeStop => "system.state.smstop",
            Self::SystemInstruction => "system.state.sys",
            Self::SystemInstructionRead => "system.state.sysl",
            Self::InterruptDisable => "system.state.di",
            Self::InterruptEnable => "system.state.ei",
            Self::WaitForInterrupt => "system.state.wfi",
            Self::WaitForInterruptTimed => "system.state.wfi.timed",
        }
    }

    /// Returns the precise effect summary for this operation.
    ///
    /// `Read` is reserved for the operations that only sample machine state
    /// into a destination the lift binds; port I/O and the enqueue stores report
    /// `ReadWrite` because they both observe a device and may change it; the two
    /// state restorations that resume a suspended context report `Call` with a
    /// call control effect, exactly as [`SystemOpKind::InterruptReturn`] does.
    /// Everything else writes machine state.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::ReadAccessRights
            | Self::ReadSegmentLimit
            | Self::VerifySegmentReadable
            | Self::VerifySegmentWritable
            | Self::LoadFarPointerDs
            | Self::LoadFarPointerEs
            | Self::LoadFarPointerFs
            | Self::LoadFarPointerGs
            | Self::LoadFarPointerSs
            | Self::ReadFsBase
            | Self::ReadGsBase
            | Self::ShadowStackReadDword
            | Self::ShadowStackReadQword
            | Self::UserInterruptTest
            | Self::ReadProtectionKeyRights
            | Self::ReadProcessorId
            | Self::ReadProcessorPower
            | Self::ReverseMapQuery
            | Self::ProfilingStoreControlBlock
            | Self::VendorCpuRead
            | Self::VendorReadModelRegister
            | Self::VendorReadShadowRegister
            | Self::VendorReadDebugRegister
            | Self::TestTarget
            | Self::TestTargetAlternate
            | Self::TestTargetAlternateUnprivileged
            | Self::TestTargetUnprivileged
            | Self::SystemInstructionRead => SsaEffects::new(SsaEffectKind::Read, false),
            Self::PortIn
            | Self::PortOut
            | Self::PortInStringByte
            | Self::PortInStringWord
            | Self::PortInStringDword
            | Self::PortOutStringByte
            | Self::PortOutStringWord
            | Self::PortOutStringDword
            | Self::EnqueueCommand
            | Self::EnqueueCommandSupervisor => SsaEffects::new(SsaEffectKind::ReadWrite, false),
            Self::UserInterruptReturn
            | Self::ResumeFromSystemManagement
            | Self::DebugRestoreState => {
                SsaEffects::new(SsaEffectKind::Call, false).with_control(ControlEffect::Call)
            }
            _ => SsaEffects::new(SsaEffectKind::Write, false),
        }
    }

    /// Returns `true` when the operation's first explicit operand is the
    /// destination it defines.
    ///
    /// The registers an encoding writes implicitly reach the lift through its
    /// effect descriptor; this covers the ones named by an operand instead, so
    /// `lar`, `rdpid`, `sysl` and their peers define what they read into rather
    /// than leaving every later use of it unbound.
    #[must_use]
    pub const fn writes_destination_operand(self) -> bool {
        matches!(
            self,
            Self::ReadAccessRights
                | Self::ReadSegmentLimit
                | Self::AdjustRequestedPrivilege
                | Self::LoadFarPointerDs
                | Self::LoadFarPointerEs
                | Self::LoadFarPointerFs
                | Self::LoadFarPointerGs
                | Self::LoadFarPointerSs
                | Self::ReadFsBase
                | Self::ReadGsBase
                | Self::ShadowStackReadDword
                | Self::ShadowStackReadQword
                | Self::ReadProcessorId
                | Self::ProfilingStoreControlBlock
                | Self::TestTarget
                | Self::TestTargetAlternate
                | Self::TestTargetAlternateUnprivileged
                | Self::TestTargetUnprivileged
                | Self::SystemInstructionRead
                | Self::InterruptDisable
                | Self::InterruptEnable
        )
    }
}

/// Structured identity of a typed native **system / privileged** operation
/// ([`SsaOp::SystemOp`]) — the first-class replacement for the system,
/// control-register, syscall, trap and cache/TLB cases formerly carried by
/// `NativeOpaque`. The kind drives a precise effect summary
/// and a distinct similarity class; there is no catch-all.
///
/// [`SsaOp::SystemOp`]: crate::ir::ops::def::SsaOp::SystemOp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SystemOpKind {
    /// Processor identification (`cpuid`).
    CpuId,
    /// Time-stamp / cycle counter read (`rdtsc`/`rdtscp`/`rdcycle`); `aux` is
    /// true when an auxiliary value (e.g. TSC_AUX) is also produced (`rdtscp`).
    Timestamp {
        /// True when an auxiliary value (e.g. `TSC_AUX` for `rdtscp`) is also produced.
        aux: bool,
    },
    /// Read a performance-monitoring counter (`rdpmc`).
    ReadPerfCounter,
    /// Control-, debug-, model-specific- or status-register access, named by
    /// its [`SysRegOp`].
    ///
    /// The single home for every special-register move, model-specific,
    /// extended-control, control and debug registers included: each is a
    /// [`SysRegOp`] variant, because the register file an instruction reaches is
    /// a property of the instruction.
    ControlRegister(SysRegOp),
    /// System-call entry. Every architecture has one — x86 `syscall`, ARM
    /// `svc`, RISC-V `ecall` — and they are the same operation, so they are the
    /// same kind and the same similarity token.
    SystemCall,
    /// System-call or fast-return to the caller's privilege level.
    SystemReturn,
    /// Software trap, named by its [`TrapOp`]; `vector` carries the trap number
    /// when statically known.
    Trap {
        /// Which trap the instruction raises.
        op: TrapOp,
        /// Trap / interrupt vector when statically known (e.g. `int 0x80`), else
        /// `None`. Only the operand-carrying forms have one.
        vector: Option<u8>,
    },
    /// Interrupt or exception return, named by its [`InterruptReturnOp`].
    InterruptReturn(InterruptReturnOp),
    /// Cache maintenance, named by its [`CacheMaintenanceOp`].
    CacheMaintenance(CacheMaintenanceOp),
    /// TLB maintenance, named by its [`TlbMaintenanceOp`].
    TlbMaintenance(TlbMaintenanceOp),
    /// A serializing operation not expressible as a plain `Fence`, named by its
    /// [`BarrierOp`].
    Barrier(BarrierOp),
    /// A named machine-state operation — descriptor tables, segment bases, port
    /// I/O, shadow stack, power control and their peers. The payload names which
    /// one, so no two instructions share a kind.
    MachineState(MachineStateOp),
    /// Virtualization or secure-enclave operation, named by its
    /// [`HypervisorOp`].
    Hypervisor(HypervisorOp),
    /// On-chip hardware acceleration engine operating on memory buffers (VIA
    /// PadLock `xstore`/`xcrypt*`/`xsha*`): a crypto / hash / random-number
    /// engine driven by implicit pointer/count registers, writing its output
    /// buffer to memory. Named by its [`HardwareEngineOp`].
    HardwareEngine(HardwareEngineOp),
    /// Hardware transactional-memory control (AArch64 TME `tstart`/`tcommit`/
    /// `tcancel`/`ttest`, x86 TSX `xbegin`/`xend`/`xabort`/`xtest`), named by
    /// its [`SystemTransactionKind`], which is their only home.
    Transaction(SystemTransactionKind),
    /// An architectural hint, named by its [`HintOp`] — `nop`, `pause`,
    /// `endbr64`, `wfe`, the prefetch family.
    ///
    /// A hint changes no architectural data, which is why carrying its identity
    /// matters: an operation that only records "a hint happened" cannot tell
    /// `endbr64` from instruction padding.
    Hint(HintOp),
}

/// The hardware transactional-memory operation carried by
/// [`SystemOpKind::Transaction`].
///
/// The single home for transactional memory, load-address tracking included.
/// [`BarrierOp`] names serializing operations and nothing transactional, so
/// every operation whose meaning is a property of the transaction — beginning
/// one, ending one, asking whether one is running, changing what it tracks —
/// has exactly one identity and therefore exactly one effect summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum SystemTransactionKind {
    /// Begin a transaction (`tstart`/`xbegin`).
    Start,
    /// Commit the current transaction (`tcommit`/`xend`).
    Commit,
    /// Abort the current transaction (`tcancel`/`xabort`).
    Cancel,
    /// Test whether executing transactionally (`ttest`/`xtest`).
    Test,
    /// Resume load-address tracking inside a transaction (`xresldtrk`).
    ResumeLoadTracking,
    /// Suspend load-address tracking inside a transaction (`xsusldtrk`).
    SuspendLoadTracking,
}

impl SystemTransactionKind {
    /// Returns the neutral display mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Start => "txn.start",
            Self::Commit => "txn.commit",
            Self::Cancel => "txn.cancel",
            Self::Test => "txn.test",
            Self::ResumeLoadTracking => "txn.resume.load.tracking",
            Self::SuspendLoadTracking => "txn.suspend.load.tracking",
        }
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Start => "system.txn.start",
            Self::Commit => "system.txn.commit",
            Self::Cancel => "system.txn.cancel",
            Self::Test => "system.txn.test",
            Self::ResumeLoadTracking => "system.txn.resume.load.tracking",
            Self::SuspendLoadTracking => "system.txn.suspend.load.tracking",
        }
    }

    /// Precise effect summary for this transactional operation.
    ///
    /// A commit publishes every write the transaction buffered and an abort
    /// discards them, so both are ordering points other threads observe: they
    /// are sequentially consistent fences, exactly like [`SsaOp::Fence`].
    /// [`Self::Test`] only reads transactional status and touches no memory, so
    /// it is a `Read` — `MemoryOp::Load` over an unknown location is still
    /// ordered against every may-aliasing store. Beginning a transaction and
    /// changing what it tracks write processor state, so they are `Write`.
    ///
    /// [`SsaOp::Fence`]: crate::ir::ops::def::SsaOp::Fence
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::Commit | Self::Cancel => {
                SsaEffects::new(SsaEffectKind::Fence, false).fence_ordering(AtomicOrdering::SeqCst)
            }
            Self::Test => SsaEffects::new(SsaEffectKind::Read, false),
            Self::Start | Self::ResumeLoadTracking | Self::SuspendLoadTracking => {
                SsaEffects::new(SsaEffectKind::Write, false)
            }
        }
    }
}

impl SystemOpKind {
    /// Precise effect summary for this system op — derived from the kind, never
    /// an opaque echo. Reads of machine state report `Read`; writes (incl. cache
    /// / TLB / privileged state mutation) report `Write`; control-transferring
    /// ops (syscalls, traps, hypervisor calls, interrupt returns) report `Call`
    /// with the matching control effect.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::CpuId | Self::Timestamp { .. } | Self::ReadPerfCounter => {
                SsaEffects::new(SsaEffectKind::Read, false)
            }
            Self::ControlRegister(op) => op.effects(),
            Self::MachineState(op) => op.effects(),
            Self::Transaction(op) => op.effects(),
            Self::Hint(op) => op.effects(),
            Self::CacheMaintenance(_) | Self::TlbMaintenance(_) | Self::HardwareEngine(_) => {
                SsaEffects::new(SsaEffectKind::Write, false)
            }
            // `serialize`, `mcommit` and `pcommit` order execution and stores:
            // they must classify as a fence so Memory SSA emits a
            // `MemoryOp::Barrier` and the verifier's fence invariant applies,
            // exactly like `SsaOp::Fence`. Classifying them as `Write` is
            // conservatively safe for movement but models an ordering construct
            // as a clobber. The kind does not say how far the ordering reaches,
            // so assume the strongest.
            Self::Barrier(_) => {
                SsaEffects::new(SsaEffectKind::Fence, false).fence_ordering(AtomicOrdering::SeqCst)
            }
            Self::SystemCall | Self::SystemReturn | Self::Hypervisor(_) | Self::Trap { .. } => {
                SsaEffects::new(SsaEffectKind::Call, true).with_control(ControlEffect::Call)
            }
            // `iret`/`eret` transfer control externally to the interrupted
            // context — a `Call`-class control transfer, exactly like
            // `SystemReturn` (`sysret`/`sysexit`), NOT a `Return`. Front-ends
            // classify it as a non-terminating typed system op (its family is
            // not a block terminator), so the block structurally continues past
            // it; declaring `ControlEffect::Return` here would make a
            // non-terminator op claim a block-ending effect and fail the
            // `check_native_effects` verifier invariant.
            Self::InterruptReturn(_) => {
                SsaEffects::new(SsaEffectKind::Call, false).with_control(ControlEffect::Call)
            }
        }
    }

    /// Stable display / fingerprint key for this system op (used by
    /// [`SsaOp::opcode_name`]).
    ///
    /// [`SsaOp::opcode_name`]: crate::ir::ops::def::SsaOp::opcode_name
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::CpuId => "system.cpuid",
            // `aux` is an identity field, not data: `rdtscp` reads TSC_AUX and
            // `rdtsc` does not, so one spelling for both merges two
            // instructions onto one fingerprint.
            Self::Timestamp { aux: false } => "system.timestamp",
            Self::Timestamp { aux: true } => "system.timestamp.aux",
            Self::ReadPerfCounter => "system.perfcounter",
            Self::ControlRegister(op) => op.kind_str(),
            Self::SystemCall => "system.syscall",
            Self::SystemReturn => "system.sysreturn",
            Self::Trap { op, .. } => op.kind_str(),
            Self::InterruptReturn(op) => op.kind_str(),
            Self::CacheMaintenance(op) => op.kind_str(),
            Self::TlbMaintenance(op) => op.kind_str(),
            Self::Barrier(op) => op.kind_str(),
            Self::MachineState(op) => op.kind_str(),
            Self::Hypervisor(op) => op.kind_str(),
            Self::HardwareEngine(op) => op.kind_str(),
            Self::Transaction(op) => op.kind_str(),
            Self::Hint(op) => op.kind_str(),
        }
    }

    /// Returns the family this kind belongs to.
    ///
    /// An exhaustive match on purpose, and the first link of the chain that
    /// keeps [`Self::identities`] total: a new variant does not compile until
    /// its author names a family, a new family does not compile until it is
    /// added to [`SystemOpFamily::ALL`], and the test over `identities()` then
    /// fails until the family is expanded there too. Every link but the last is
    /// a compile error.
    pub(crate) const fn family(self) -> SystemOpFamily {
        match self {
            Self::CpuId => SystemOpFamily::CpuId,
            Self::Timestamp { .. } => SystemOpFamily::Timestamp,
            Self::ReadPerfCounter => SystemOpFamily::ReadPerfCounter,
            Self::ControlRegister(_) => SystemOpFamily::ControlRegister,
            Self::SystemCall => SystemOpFamily::SystemCall,
            Self::SystemReturn => SystemOpFamily::SystemReturn,
            Self::Trap { .. } => SystemOpFamily::Trap,
            Self::InterruptReturn(_) => SystemOpFamily::InterruptReturn,
            Self::CacheMaintenance(_) => SystemOpFamily::CacheMaintenance,
            Self::TlbMaintenance(_) => SystemOpFamily::TlbMaintenance,
            Self::Barrier(_) => SystemOpFamily::Barrier,
            Self::MachineState(_) => SystemOpFamily::MachineState,
            Self::Hypervisor(_) => SystemOpFamily::Hypervisor,
            Self::HardwareEngine(_) => SystemOpFamily::HardwareEngine,
            Self::Transaction(_) => SystemOpFamily::Transaction,
            Self::Hint(_) => SystemOpFamily::Hint,
        }
    }

    /// Returns every system-op identity the IR can name.
    ///
    /// Rust cannot count the variants of an enum with payloads, so the domain
    /// is built family by family from the payload enums' own
    /// [`OpKindTable::all`] iterators. Data fields — the ones
    /// [`Self::kind_str`] does not read — are fixed at a canonical placeholder
    /// (`Trap { vector: None }`), which is the identity/data split made
    /// executable: two values differing only in a data field are one identity.
    ///
    /// [`OpKindTable::all`]: super::OpKindTable::all
    #[must_use]
    pub fn identities() -> Vec<Self> {
        let mut all = Vec::new();
        for family in SystemOpFamily::ALL {
            let mut expanded: Vec<Self> = Vec::new();
            match family {
                SystemOpFamily::CpuId => expanded.push(Self::CpuId),
                SystemOpFamily::Timestamp => {
                    expanded.push(Self::Timestamp { aux: false });
                    expanded.push(Self::Timestamp { aux: true });
                }
                SystemOpFamily::ReadPerfCounter => expanded.push(Self::ReadPerfCounter),
                SystemOpFamily::ControlRegister => {
                    expanded.extend(SysRegOp::all().map(Self::ControlRegister));
                }
                SystemOpFamily::SystemCall => expanded.push(Self::SystemCall),
                SystemOpFamily::SystemReturn => expanded.push(Self::SystemReturn),
                SystemOpFamily::Trap => {
                    expanded.extend(TrapOp::all().map(|op| Self::Trap { op, vector: None }));
                }
                SystemOpFamily::InterruptReturn => {
                    expanded.extend(InterruptReturnOp::all().map(Self::InterruptReturn));
                }
                SystemOpFamily::CacheMaintenance => {
                    expanded.extend(CacheMaintenanceOp::all().map(Self::CacheMaintenance));
                }
                SystemOpFamily::TlbMaintenance => {
                    expanded.extend(TlbMaintenanceOp::all().map(Self::TlbMaintenance));
                }
                SystemOpFamily::Barrier => expanded.extend(BarrierOp::all().map(Self::Barrier)),
                SystemOpFamily::MachineState => {
                    expanded.extend(MachineStateOp::all().map(Self::MachineState));
                }
                SystemOpFamily::Hypervisor => {
                    expanded.extend(HypervisorOp::all().map(Self::Hypervisor));
                }
                SystemOpFamily::HardwareEngine => {
                    expanded.extend(HardwareEngineOp::all().map(Self::HardwareEngine));
                }
                SystemOpFamily::Transaction => {
                    expanded.extend(SystemTransactionKind::all().map(Self::Transaction));
                }
                SystemOpFamily::Hint => expanded.extend(HintOp::all().map(Self::Hint)),
            }
            // An arm that expands into the wrong family contributes nothing
            // rather than making one family appear twice and another not at
            // all — which the domain-coverage test then reports as the missing
            // family, naming the arm that has to be fixed.
            expanded.retain(|kind| kind.family() == family);
            all.append(&mut expanded);
        }
        all
    }
}

/// One [`SystemOpKind`] variant, with its payload erased.
///
/// The countable shadow of a payload-carrying enum: `SystemOpKind` itself has
/// no finite variant count Rust can name, but its families do, and
/// [`Self::ALL`] pins that count as an array length so adding a family without
/// listing it fails to compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum SystemOpFamily {
    /// [`SystemOpKind::CpuId`].
    CpuId,
    /// [`SystemOpKind::Timestamp`].
    Timestamp,
    /// [`SystemOpKind::ReadPerfCounter`].
    ReadPerfCounter,
    /// [`SystemOpKind::ControlRegister`].
    ControlRegister,
    /// [`SystemOpKind::SystemCall`].
    SystemCall,
    /// [`SystemOpKind::SystemReturn`].
    SystemReturn,
    /// [`SystemOpKind::Trap`].
    Trap,
    /// [`SystemOpKind::InterruptReturn`].
    InterruptReturn,
    /// [`SystemOpKind::CacheMaintenance`].
    CacheMaintenance,
    /// [`SystemOpKind::TlbMaintenance`].
    TlbMaintenance,
    /// [`SystemOpKind::Barrier`].
    Barrier,
    /// [`SystemOpKind::MachineState`].
    MachineState,
    /// [`SystemOpKind::Hypervisor`].
    Hypervisor,
    /// [`SystemOpKind::HardwareEngine`].
    HardwareEngine,
    /// [`SystemOpKind::Transaction`].
    Transaction,
    /// [`SystemOpKind::Hint`].
    Hint,
}

impl SystemOpFamily {
    /// Every family, in declaration order.
    ///
    /// The array length is the pin: a family added to the enum and not to this
    /// list is a type error, not a silently unexpanded branch of
    /// [`SystemOpKind::identities`].
    pub(crate) const ALL: [Self; 16] = [
        Self::CpuId,
        Self::Timestamp,
        Self::ReadPerfCounter,
        Self::ControlRegister,
        Self::SystemCall,
        Self::SystemReturn,
        Self::Trap,
        Self::InterruptReturn,
        Self::CacheMaintenance,
        Self::TlbMaintenance,
        Self::Barrier,
        Self::MachineState,
        Self::Hypervisor,
        Self::HardwareEngine,
        Self::Transaction,
        Self::Hint,
    ];
}

/// Structured identity of a typed native **compute** operation
/// ([`SsaOp::ComputeOp`]) — the first-class replacement for the hardware
/// compute intrinsics (`pdep`/`pext`, `crc32`, `rdrand`/`rdseed`, pointer
/// authentication). The kind
/// drives a precise effect summary and a distinct similarity class; there is no
/// catch-all.
///
/// [`SsaOp::ComputeOp`]: crate::ir::ops::def::SsaOp::ComputeOp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComputeKind {
    /// BMI2 `pdep` — parallel bit deposit of the low bits of the source into the
    /// positions set in the mask. Pure and deterministic.
    BitDeposit,
    /// BMI2 `pext` — parallel bit extract of the masked bits of the source into
    /// the low bits of the result. Pure and deterministic.
    BitExtract,
    /// Hardware CRC32 accumulation (`crc32`) — pure checksum step.
    Checksum,
    /// Hardware random number / seed (`rdrand`/`rdseed`). Nondeterministic: it
    /// reads a hardware entropy source, so it is never pure / foldable;
    /// `from_entropy` distinguishes `rdseed` (true) from `rdrand` (false).
    Random {
        /// `true` for `rdseed` (conditioned entropy), `false` for `rdrand`.
        from_entropy: bool,
    },
    /// AArch64 pointer authentication carrying the PAC sub-operation
    /// (`pac*`/`aut*`/`xpac*`/`pacga`). Pure and deterministic given the key.
    PointerAuth(PacKind),
    /// MIPS DSP-ASE accumulator operation — extract / shift / shift-load of the
    /// 64-bit DSP accumulator pair, and the modular-subtract address step
    /// (`extr*`/`extp*`/`shilo*`/`mthlip`/`modsub`). The specific operation
    /// (including the rounding / saturating extract variants) is preserved in
    /// [`NativeKindedData::mnemonic`]; all are pure given their register inputs.
    MipsDspAccumulate,
}

impl ComputeKind {
    /// Precise effect summary for this compute op — derived from the kind, never
    /// an opaque echo. Bit-permute / checksum / pointer-auth are pure;
    /// random-source reads a nondeterministic hardware entropy source (`Read`)
    /// so it is never folded or eliminated.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::BitDeposit
            | Self::BitExtract
            | Self::Checksum
            | Self::PointerAuth(_)
            | Self::MipsDspAccumulate => SsaEffects::new(SsaEffectKind::Pure, false),
            Self::Random { .. } => SsaEffects::new(SsaEffectKind::Read, false),
        }
    }

    /// Stable display / fingerprint key for this compute op (used by
    /// [`SsaOp::opcode_name`]).
    ///
    /// [`SsaOp::opcode_name`]: crate::ir::ops::def::SsaOp::opcode_name
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::BitDeposit => "compute.pdep",
            Self::BitExtract => "compute.pext",
            Self::Checksum => "compute.crc32",
            Self::Random {
                from_entropy: false,
            } => "compute.rdrand",
            Self::Random { from_entropy: true } => "compute.rdseed",
            // The sub-operation is identity, not data: signing, authenticating
            // and stripping are three instructions, and the difference between
            // the first two is the whole security property.
            Self::PointerAuth(op) => op.kind_str(),
            Self::MipsDspAccumulate => "compute.mips_dsp_acc",
        }
    }

    /// Returns the family this kind belongs to.
    ///
    /// Exhaustive, for the same reason [`SystemOpKind::family`] is: a new
    /// variant cannot be added without deciding how [`Self::identities`]
    /// enumerates it.
    pub(crate) const fn family(self) -> ComputeFamily {
        match self {
            Self::BitDeposit => ComputeFamily::BitDeposit,
            Self::BitExtract => ComputeFamily::BitExtract,
            Self::Checksum => ComputeFamily::Checksum,
            Self::Random { .. } => ComputeFamily::Random,
            Self::PointerAuth(_) => ComputeFamily::PointerAuth,
            Self::MipsDspAccumulate => ComputeFamily::MipsDspAccumulate,
        }
    }

    /// Returns every compute-op identity the IR can name.
    ///
    /// Built family by family, so the payload-carrying variants expand through
    /// their payload enum rather than through a hand-written sample.
    #[must_use]
    pub fn identities() -> Vec<Self> {
        let mut all = Vec::new();
        for family in ComputeFamily::ALL {
            let mut expanded: Vec<Self> = Vec::new();
            match family {
                ComputeFamily::BitDeposit => expanded.push(Self::BitDeposit),
                ComputeFamily::BitExtract => expanded.push(Self::BitExtract),
                ComputeFamily::Checksum => expanded.push(Self::Checksum),
                ComputeFamily::Random => {
                    expanded.push(Self::Random {
                        from_entropy: false,
                    });
                    expanded.push(Self::Random { from_entropy: true });
                }
                ComputeFamily::PointerAuth => {
                    expanded.extend(PacKind::all().map(Self::PointerAuth));
                }
                ComputeFamily::MipsDspAccumulate => expanded.push(Self::MipsDspAccumulate),
            }
            // See `SystemOpKind::identities`: an arm expanding into the wrong
            // family drops out here and shows up as a missing family.
            expanded.retain(|kind| kind.family() == family);
            all.append(&mut expanded);
        }
        all
    }
}

/// One [`ComputeKind`] variant, with its payload erased.
///
/// The countable shadow of [`ComputeKind`], exactly as [`SystemOpFamily`] is of
/// [`SystemOpKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum ComputeFamily {
    /// [`ComputeKind::BitDeposit`].
    BitDeposit,
    /// [`ComputeKind::BitExtract`].
    BitExtract,
    /// [`ComputeKind::Checksum`].
    Checksum,
    /// [`ComputeKind::Random`].
    Random,
    /// [`ComputeKind::PointerAuth`].
    PointerAuth,
    /// [`ComputeKind::MipsDspAccumulate`].
    MipsDspAccumulate,
}

impl ComputeFamily {
    /// Every family, in declaration order. The array length is the pin.
    pub(crate) const ALL: [Self; 6] = [
        Self::BitDeposit,
        Self::BitExtract,
        Self::Checksum,
        Self::Random,
        Self::PointerAuth,
        Self::MipsDspAccumulate,
    ];
}

/// Structured identity of a legacy x86 **binary-coded-decimal adjust**
/// operation ([`SsaOp::BcdAdjust`]) — the first-class, named model for the
/// `daa`/`das`/`aaa`/`aas`/`aam`/`aad` instructions. Each variant names the
/// exact hardware operation (the LLVM-intrinsic model): the lifter wires the
/// accumulator and flags through as typed SSA values rather than decomposing
/// the flag-dependent correction into branches, and never carries it opaquely.
///
/// [`SsaOp::BcdAdjust`]: crate::ir::ops::def::SsaOp::BcdAdjust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u16)]
#[derive(num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
pub enum BcdAdjustKind {
    /// Decimal adjust `AL` after addition (`daa`).
    DecimalAddAdjust,
    /// Decimal adjust `AL` after subtraction (`das`).
    DecimalSubAdjust,
    /// ASCII adjust `AL` after addition (`aaa`).
    AsciiAddAdjust,
    /// ASCII adjust `AL` after subtraction (`aas`).
    AsciiSubAdjust,
    /// ASCII adjust `AX` after multiply (`aam`); the radix rides
    /// [`BcdAdjustData::base`] (10 unless an explicit `imm8` is given).
    AsciiMulAdjust,
    /// ASCII adjust `AX` before division (`aad`); the radix rides
    /// [`BcdAdjustData::base`].
    AsciiDivAdjust,
}

impl BcdAdjustKind {
    /// Effect summary — every BCD adjust is a pure function of the accumulator
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

/// Boxed payload for [`SsaOp::BcdAdjust`]. Mirrors the typed-compute operand
/// shape — explicit SSA inputs/outputs (the accumulator and the flag values)
/// plus optional source provenance — with the effect summary and similarity
/// class derived from the [`BcdAdjustKind`].
///
/// [`SsaOp::BcdAdjust`]: crate::ir::ops::def::SsaOp::BcdAdjust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BcdAdjustData {
    /// Structured identity of the operation.
    pub kind: BcdAdjustKind,
    /// Radix for the ASCII multiply/divide adjusts (`aam`/`aad`); 10 (or unused)
    /// for the decimal/ASCII add/subtract adjusts.
    pub base: u8,
    /// Human-readable native mnemonic, for display / provenance only.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs defined by the operation (adjusted accumulator,
    /// result flags).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation (source accumulator, and the
    /// incoming flags for the add/subtract adjusts).
    pub inputs: Vec<SsaVarId>,
}

impl_kinded_payload! {
    NativeOpaqueData { mnemonic, metadata, outputs, inputs, effects };
    KindedVecData<K> { kind, outputs, inputs };
    VecImm8Data { imm8, outputs, inputs };
    NativeKindedData<K> { kind, mnemonic, metadata, outputs, inputs };
    BcdAdjustData { kind, base, mnemonic, metadata, outputs, inputs };
}
