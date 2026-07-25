//! Operand-level enums and classification vocabulary shared by [`SsaOp`].
//!
//! Comparison and signedness kinds, fence / atomic qualifiers, the operation
//! and similarity taxonomies, and the native intrinsic, system, compute and
//! BCD descriptors. These are the small vocabulary types the op variants are
//! built out of; the variants themselves live in [`super::def`].

use std::fmt;

use super::*;
use crate::ir::variable::SsaVarId;

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
    /// Recognized native intrinsic with a structured identity.
    NativeIntrinsic,
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
/// (mnemonic, original encoding metadata, explicit inputs/outputs, clobbers,
/// and an effect summary). Inlining that into the enum would make *every*
/// [`SsaOp`] as large as this rare variant, so the payload is held behind a
/// `Box` and the common operations stay compact.
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
    /// Abstract target state clobbered by the instruction.
    pub clobbers: Vec<NativeClobber>,
    /// Conservative effect summary for optimization barriers.
    pub effects: SsaEffects,
}

/// Identity of a recognized native intrinsic — a native operation with no
/// primitive closed form but faithful, fully-modellable dataflow (its inputs,
/// outputs and machine-state effects are known even though the computation
/// itself is hardware-defined).
///
/// Unlike [`SsaOp::NativeOpaque`] (the genuine "unmodelled" escape hatch), a
/// [`SsaOp::NativeIntrinsic`] carries a stable identity so passes, similarity
/// and rendering can reason about *which* native op it is. New native ops are
/// added as a single [`NativeIntrinsicId`] arm rather than a new [`SsaOp`]
/// variant; the [`NativeIntrinsicData::mnemonic`] field carries the raw mnemonic for ops not
/// yet promoted to a dedicated id.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NativeIntrinsicId {
    /// x86 `cpuid` — processor identification (reads EAX/ECX, writes EAX/EBX/ECX/EDX).
    Cpuid,
    /// x86 `rdtsc` — read time-stamp counter into EDX:EAX.
    Rdtsc,
    /// x86 `rdtscp` — `rdtsc` plus TSC_AUX into ECX.
    Rdtscp,
    /// x86 `rdmsr` — read model-specific register.
    Rdmsr,
    /// x86 `wrmsr` — write model-specific register.
    Wrmsr,
    /// x86 `rdpmc` — read performance-monitoring counter.
    Rdpmc,
    /// x86 `xgetbv` — read extended control register.
    Xgetbv,
    /// x86 `xsetbv` — write extended control register.
    Xsetbv,
    /// System-call entry (`syscall`/`sysenter`/`svc`/`ecall`).
    SystemCall,
    /// System-call return (`sysret`/`sysexit`).
    SystemReturn,
    /// BMI2 `pdep` — parallel bit deposit under a mask.
    BitDeposit,
    /// BMI2 `pext` — parallel bit extract under a mask.
    BitExtract,
    /// Hardware CRC32 accumulation (`crc32`).
    Crc32,
    /// Hardware random number (`rdrand`).
    RandomNumber,
    /// Hardware random seed (`rdseed`).
    RandomSeed,
    /// AArch64 pointer authentication, carrying which PAC sub-operation ran
    /// (`pac*` sign, `aut*` authenticate, `xpac*` strip, `pacga` generic MAC).
    PointerAuth(PacKind),
    /// Virtualization / hypervisor op (x86 `vmcall`/`vmlaunch`/`vmread`/`vmwrite`,
    /// ARM `hvc`).
    Hypervisor,
    /// Privileged machine-state op with no value result (`hlt`/`cli`/`sti`,
    /// `lgdt`/`lidt`/`swapgs`/`invlpg`/`wbinvd`, ARM `cps`).
    Privileged,
    /// Control / status / system-register access (`mrs`/`msr`, RISC-V `csrr*`,
    /// MIPS `mfc0`/`rdhwr`).
    ControlRegister,
}

/// PAC sub-operation carried by [`NativeIntrinsicId::PointerAuth`].
///
/// ARMv8.3 pointer authentication signs a pointer with a cryptographic MAC in
/// its unused high bits, authenticates (validates) it before use, strips the
/// MAC back to the raw pointer, or computes a generic MAC into a register.
/// Distinguishing these is required to faithfully reconstruct the native
/// instruction role; without it every PAC op collapses to "sign".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
/// mnemonic, optional source metadata, and architectural clobbers.
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
    /// Architectural state the operation clobbers.
    pub clobbers: Vec<NativeClobber>,
}

/// Hardware floating-point transcendental / residue function
/// ([`SsaOp::FpTranscendental`]).
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
}

/// Boxed payload for [`SsaOp::NativeIntrinsic`].
///
/// Mirrors [`NativeOpaqueData`] (explicit inputs/outputs, clobbers, effect
/// summary) but adds a structured [`NativeIntrinsicId`]. Boxed for the same
/// size reason as [`NativeOpaqueData`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NativeIntrinsicData {
    /// Structured identity of the native operation.
    pub id: NativeIntrinsicId,
    /// Human-readable instruction mnemonic or description.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs defined by the instruction.
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the instruction.
    pub inputs: Vec<SsaVarId>,
    /// Abstract target state clobbered by the instruction.
    pub clobbers: Vec<NativeClobber>,
    /// Conservative effect summary for optimization barriers.
    pub effects: SsaEffects,
}

/// Namespace of a system / control / status register accessed by a
/// [`SystemOpKind::ReadSysReg`] / [`SystemOpKind::WriteSysReg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SysRegNamespace {
    /// x86 model-specific register (`rdmsr`/`wrmsr`).
    X86Msr,
    /// x86 extended control register (`xgetbv`/`xsetbv`).
    X86Xcr,
    /// x86 control register (`mov cr`).
    X86ControlReg,
    /// x86 debug register (`mov dr`).
    X86DebugReg,
    /// AArch64 system register (`mrs`/`msr`).
    Arm64System,
    /// RISC-V control/status register (`csrr*`/`csrw*`).
    RiscvCsr,
    /// MIPS coprocessor-0 register (`mfc0`/`mtc0`).
    MipsCop0,
}

/// Structured identity of a typed native **system / privileged** operation
/// ([`SsaOp::SystemOp`]) — the first-class replacement for the system,
/// control-register, syscall, trap and cache/TLB cases formerly carried by
/// `NativeIntrinsic` / `NativeOpaque`. The kind drives a precise effect summary
/// and a distinct similarity class; there is no catch-all.
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
    /// Read a system/control/model-specific register into a GPR.
    ReadSysReg {
        /// Register file the operand names (MSR / CR / DR / AArch64 sysreg / RISC-V CSR).
        namespace: SysRegNamespace,
    },
    /// Write a GPR into a system/control/model-specific register.
    WriteSysReg {
        /// Register file the operand names (MSR / CR / DR / AArch64 sysreg / RISC-V CSR).
        namespace: SysRegNamespace,
    },
    /// Read a performance-monitoring counter (`rdpmc`).
    ReadPerfCounter,
    /// System-call entry (`syscall`/`sysenter`/`svc`/`ecall`).
    SystemCall,
    /// System-call / fast-return (`sysret`/`sysexit`).
    SystemReturn,
    /// Software trap / interrupt (`int N`/`int3`/`brk`/`bkpt`); `vector` carries
    /// the trap number when statically known.
    Trap {
        /// Trap / interrupt vector when statically known (e.g. `int 0x80`), else `None`.
        vector: Option<u8>,
    },
    /// Interrupt / exception return (`iret`/`iretq`/`eret`).
    InterruptReturn,
    /// Cache maintenance (`invd`/`wbinvd`/`clflush`/`dc`/`ic`).
    CacheMaintenance,
    /// TLB maintenance (`invlpg`/`tlbi`).
    TlbMaintenance,
    /// Privileged barrier / serialization not expressible as a plain `Fence`.
    Barrier,
    /// Privileged machine-state op with no value result (`hlt`/`cli`/`sti`,
    /// `lgdt`/`lidt`/`swapgs`, ARM `cps`).
    Privileged,
    /// Virtualization / hypervisor call (`vmcall`/`vmlaunch`, ARM `hvc`).
    Hypervisor,
    /// On-chip hardware acceleration engine operating on memory buffers (VIA
    /// PadLock `xstore`/`xcrypt*`/`xsha*`): a crypto / hash / random-number
    /// engine driven by implicit pointer/count registers, writing its output
    /// buffer to memory.
    HardwareEngine,
    /// Hardware transactional-memory control (AArch64 TME `tstart`/`tcommit`/
    /// `tcancel`/`ttest`, x86 TSX `xbegin`/`xend`/`xabort`/`xtest`), named by
    /// its [`SystemTransactionKind`].
    Transaction(SystemTransactionKind),
}

/// The hardware transactional-memory operation carried by
/// [`SystemOpKind::Transaction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SystemTransactionKind {
    /// Begin a transaction (`tstart`/`xbegin`).
    Start,
    /// Commit the current transaction (`tcommit`/`xend`).
    Commit,
    /// Abort the current transaction (`tcancel`/`xabort`).
    Cancel,
    /// Test whether executing transactionally (`ttest`/`xtest`).
    Test,
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
            Self::CpuId
            | Self::Timestamp { .. }
            | Self::ReadSysReg { .. }
            | Self::ReadPerfCounter => SsaEffects::new(SsaEffectKind::Read, false),
            Self::WriteSysReg { .. }
            | Self::Privileged
            | Self::CacheMaintenance
            | Self::TlbMaintenance
            | Self::HardwareEngine
            | Self::Transaction(_) => SsaEffects::new(SsaEffectKind::Write, false),
            // `dsb`/`dmb`/`isb`/`mfence` order memory: they must classify as a
            // fence so Memory SSA emits a `MemoryOp::Barrier` and the verifier's
            // fence invariant applies, exactly like `SsaOp::Fence`. Classifying
            // them as `Write` is conservatively safe for movement but models an
            // ordering construct as a clobber. The kind does not distinguish
            // `dmb` from `isb`, so assume the strongest ordering.
            Self::Barrier => {
                SsaEffects::new(SsaEffectKind::Fence, false).fence_ordering(AtomicOrdering::SeqCst)
            }
            Self::SystemCall | Self::SystemReturn | Self::Hypervisor | Self::Trap { .. } => {
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
            Self::InterruptReturn => {
                SsaEffects::new(SsaEffectKind::Call, false).with_control(ControlEffect::Call)
            }
        }
    }

    /// Stable display / fingerprint key for this system op (used by
    /// [`SsaOp::opcode_name`]).
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::CpuId => "system.cpuid",
            Self::Timestamp { .. } => "system.timestamp",
            Self::ReadSysReg { .. } => "system.sysreg.read",
            Self::WriteSysReg { .. } => "system.sysreg.write",
            Self::ReadPerfCounter => "system.perfcounter",
            Self::SystemCall => "system.syscall",
            Self::SystemReturn => "system.sysreturn",
            Self::Trap { .. } => "system.trap",
            Self::InterruptReturn => "system.iret",
            Self::CacheMaintenance => "system.cache",
            Self::TlbMaintenance => "system.tlb",
            Self::Barrier => "system.barrier",
            Self::Privileged => "system.privileged",
            Self::Hypervisor => "system.hypervisor",
            Self::HardwareEngine => "system.hwengine",
            Self::Transaction(SystemTransactionKind::Start) => "system.txn.start",
            Self::Transaction(SystemTransactionKind::Commit) => "system.txn.commit",
            Self::Transaction(SystemTransactionKind::Cancel) => "system.txn.cancel",
            Self::Transaction(SystemTransactionKind::Test) => "system.txn.test",
        }
    }
}

/// Structured identity of a typed native **compute** operation
/// ([`SsaOp::ComputeOp`]) — the first-class replacement for the hardware
/// compute intrinsics (`pdep`/`pext`, `crc32`, `rdrand`/`rdseed`, pointer
/// authentication) formerly carried opaquely by `NativeIntrinsic`. The kind
/// drives a precise effect summary and a distinct similarity class; there is no
/// catch-all.
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
            Self::PointerAuth(_) => "compute.pac",
            Self::MipsDspAccumulate => "compute.mips_dsp_acc",
        }
    }
}

/// Structured identity of a legacy x86 **binary-coded-decimal adjust**
/// operation ([`SsaOp::BcdAdjust`]) — the first-class, named model for the
/// `daa`/`das`/`aaa`/`aas`/`aam`/`aad` instructions. Each variant names the
/// exact hardware operation (the LLVM-intrinsic model): the lifter wires the
/// accumulator and flags through as typed SSA values rather than decomposing
/// the flag-dependent correction into branches, and never carries it opaquely.
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
/// shape — explicit SSA inputs/outputs (the accumulator and the flag values),
/// clobbered architectural state, optional source provenance — with the effect
/// summary and similarity class derived from the [`BcdAdjustKind`].
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
    /// Architectural state the operation clobbers.
    pub clobbers: Vec<NativeClobber>,
}
