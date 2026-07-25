//! Native machine-state modeling — registers, state accesses, clobbers, and
//! condition-flag semantics — plus the structural binary / unary operation
//! views used by generic passes.

use std::fmt;

use super::*;
use crate::ir::variable::SsaVarId;

/// Structured identity of a typed native **block-string** operation
/// ([`SsaOp::BlockString`]) — the first-class replacement for the
/// `rep`-prefixed compare / scan / load string streams formerly carried
/// opaquely by `NativeOpaque`. (`rep movs`/`rep stos` already lower to the
/// structured `CopyBlk`/`InitBlk` ops and are not represented here.) The kind
/// drives a precise effect summary and a distinct similarity class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlockStringKind {
    /// `rep`/`repe`/`repne cmps*` — streamed element compare (reads two
    /// buffers, sets flags). Reads and writes machine state (counter/pointers).
    Compare,
    /// `rep`/`repe`/`repne scas*` — streamed scan of one buffer against the
    /// accumulator (reads a buffer, sets flags).
    Scan,
    /// `rep lods*` — streamed load of successive elements into the accumulator
    /// (reads a buffer, no flag effect).
    Load,
}

impl BlockStringKind {
    /// Precise effect summary — `Read` for the load stream (`lods`), `ReadWrite`
    /// for the compare / scan streams (they advance counter/pointers and set
    /// flags). Never opaque.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::Load => SsaEffects::new(SsaEffectKind::Read, false),
            Self::Compare | Self::Scan => SsaEffects::new(SsaEffectKind::ReadWrite, false),
        }
    }

    /// Stable display / fingerprint key for this block-string op (used by
    /// [`SsaOp::opcode_name`]).
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Compare => "blockstring.cmps",
            Self::Scan => "blockstring.scas",
            Self::Load => "blockstring.lods",
        }
    }
}

/// Repeat-prefix variant carried by a [`BlockStringOpData`] — preserves the
/// exact `rep` / `repe` / `repne` semantics (the loop-termination condition) so
/// the host can faithfully reconstruct the native mnemonic without re-decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlockStringPrefix {
    /// `rep` — repeat `rcx` times unconditionally (used by `lods`).
    Repeat,
    /// `repe` / `repz` — repeat while equal (ZF=1) and `rcx != 0`.
    RepeatEqual,
    /// `repne` / `repnz` — repeat while not equal (ZF=0) and `rcx != 0`.
    RepeatNotEqual,
}

/// Boxed payload for [`SsaOp::BlockString`]. Mirrors the native-op operand
/// shape — explicit SSA inputs/outputs (advanced counter/pointers, loaded
/// accumulator), clobbered architectural state (memory, flags), and optional
/// source provenance — but the effect summary and similarity class derive from
/// the [`BlockStringKind`], not an echoed opaque blob. Carries the repeat
/// prefix and element width so the native mnemonic round-trips losslessly.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlockStringOpData {
    /// Structured identity of the operation.
    pub kind: BlockStringKind,
    /// Repeat-prefix variant (`rep` / `repe` / `repne`).
    pub prefix: BlockStringPrefix,
    /// Element width streamed by the operation, in bits (8/16/32/64).
    pub element_bits: u16,
    /// Human-readable native mnemonic, for display / provenance only.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs defined by the operation (advanced counter /
    /// pointers / accumulator).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs used by the operation (buffer addresses, count).
    pub inputs: Vec<SsaVarId>,
    /// Architectural state the operation clobbers (block memory, flags).
    pub clobbers: Vec<NativeClobber>,
    /// `true` when the operation proceeds from high to low addresses (x86 `rep
    /// cmps`/`scas`/`lods` with the direction flag set); `false` is forward.
    pub reverse: bool,
}

/// Boxed payload for [`SsaOp::WideCompareExchange`] — the double-width
/// compare-and-swap (`cmpxchg8b` / `cmpxchg16b`) that cannot be expressed as a
/// single-width [`SsaOp::CmpXchg`] (which is fixed-width and pointer-typed).
/// The first-class typed replacement for the wide-CAS case of `NativeOpaque`:
/// it carries explicit `EDX:EAX`-vs-memory expected / `ECX:EBX` desired inputs
/// and `EDX:EAX` readback outputs, with a precise sequentially-consistent
/// atomic effect (never opaque).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WideCmpXchgData {
    /// `true` for the 128-bit `cmpxchg16b`; `false` for the 64-bit `cmpxchg8b`.
    pub wide: bool,
    /// Human-readable native mnemonic, for display / provenance only.
    pub mnemonic: String,
    /// Original native instruction metadata when known.
    pub metadata: Option<NativeInstructionMetadata>,
    /// Explicit SSA outputs (the `EDX:EAX` / `RDX:RAX` readback halves).
    pub outputs: Vec<SsaVarId>,
    /// Explicit SSA inputs (memory address, expected low/high, desired low/high).
    pub inputs: Vec<SsaVarId>,
    /// Architectural state the operation clobbers (ZF / flags).
    pub clobbers: Vec<NativeClobber>,
}

/// Target register or subregister identity used by native machine-state effects.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NativeRegister {
    /// Architecture or target family that owns the register.
    pub architecture: String,
    /// Register bank or class, such as `gpr`, `xmm`, `zmm`, `p`, or `csr`.
    pub bank: String,
    /// Canonical full-register name used for alias comparisons.
    pub base: String,
    /// Specific architectural spelling, such as `al`, `eax`, `rax`, or `x0`.
    pub name: String,
    /// Bit offset within the canonical full register.
    pub bit_offset: u32,
    /// Bit width of this register view.
    pub bit_width: u32,
}

impl NativeRegister {
    /// Creates a native register descriptor.
    ///
    /// Returns `None` when any identity field is empty or `bit_width` is zero.
    #[must_use]
    pub fn new(
        architecture: impl Into<String>,
        bank: impl Into<String>,
        base: impl Into<String>,
        name: impl Into<String>,
        bit_offset: u32,
        bit_width: u32,
    ) -> Option<Self> {
        let architecture = architecture.into();
        let bank = bank.into();
        let base = base.into();
        let name = name.into();
        if architecture.is_empty() || bank.is_empty() || base.is_empty() || name.is_empty() {
            return None;
        }
        if bit_width == 0 {
            return None;
        }
        Some(Self {
            architecture,
            bank,
            base,
            name,
            bit_offset,
            bit_width,
        })
    }

    /// Returns `true` when this register view overlaps `other`.
    #[must_use]
    pub fn aliases(&self, other: &Self) -> bool {
        if self.architecture != other.architecture
            || self.bank != other.bank
            || self.base != other.base
        {
            return false;
        }
        let self_end = self.bit_offset.saturating_add(self.bit_width);
        let other_end = other.bit_offset.saturating_add(other.bit_width);
        self.bit_offset < other_end && other.bit_offset < self_end
    }

    /// Returns `true` when this register descriptor has valid identity and width fields.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.architecture.is_empty()
            && !self.bank.is_empty()
            && !self.base.is_empty()
            && !self.name.is_empty()
            && self.bit_width != 0
    }
}

/// Abstract native machine-state location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NativeStateLocation {
    /// Concrete architectural register or subregister.
    Register(NativeRegister),
    /// Register class whose concrete member is unknown or intentionally grouped.
    RegisterClass(String),
    /// Target flag register or named flag set.
    Flags(String),
    /// Architectural stack pointer state.
    StackPointer,
    /// Architectural program counter or instruction pointer state.
    ProgramCounter,
    /// Runtime vector-length configuration, such as AArch64 SVE `VL` or RISC-V `vl`.
    VectorLength,
    /// Runtime vector type/configuration state, such as RISC-V `vtype`.
    VectorConfig,
    /// Predicate or mask architectural state not represented as an SSA value.
    PredicateState(String),
    /// Control or status register.
    ControlRegister(String),
    /// Abstract memory location or memory class.
    Memory(String),
    /// Target-specific state not otherwise categorized.
    Other(String),
}

/// Access mode for a native machine-state location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NativeStateAccessKind {
    /// Reads the prior state value.
    Read,
    /// Writes the state without reading the prior value.
    Write,
    /// Reads and writes the state.
    ReadWrite,
    /// Clobbers the state with an unknown value.
    Clobber,
}

impl NativeStateAccessKind {
    /// Returns `true` when the access reads prior state.
    #[must_use]
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    /// Returns `true` when the access writes or clobbers state.
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite | Self::Clobber)
    }
}

/// Explicit native machine-state access for opaque and native operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NativeStateAccess {
    /// Machine-state location being accessed.
    pub location: NativeStateLocation,
    /// Access mode for the location.
    pub kind: NativeStateAccessKind,
    /// Optional access width when the state has a meaningful bit width.
    pub width_bits: Option<u32>,
    /// Whether the access is implicit in the native instruction encoding.
    pub implicit: bool,
}

impl NativeStateAccess {
    /// Creates a native machine-state access descriptor.
    ///
    /// Returns `None` when `width_bits` is present but zero.
    #[must_use]
    pub fn new(
        location: NativeStateLocation,
        kind: NativeStateAccessKind,
        width_bits: Option<u32>,
        implicit: bool,
    ) -> Option<Self> {
        if let Some(0) = width_bits {
            return None;
        }
        Some(Self {
            location,
            kind,
            width_bits,
            implicit,
        })
    }

    /// Creates an implicit read of a machine-state location.
    #[must_use]
    pub fn implicit_read(location: NativeStateLocation, width_bits: Option<u32>) -> Option<Self> {
        Self::new(location, NativeStateAccessKind::Read, width_bits, true)
    }

    /// Creates an implicit write of a machine-state location.
    #[must_use]
    pub fn implicit_write(location: NativeStateLocation, width_bits: Option<u32>) -> Option<Self> {
        Self::new(location, NativeStateAccessKind::Write, width_bits, true)
    }

    /// Creates an implicit read-write access to a machine-state location.
    #[must_use]
    pub fn implicit_read_write(
        location: NativeStateLocation,
        width_bits: Option<u32>,
    ) -> Option<Self> {
        Self::new(location, NativeStateAccessKind::ReadWrite, width_bits, true)
    }

    /// Returns `true` when this access reads prior machine state.
    #[must_use]
    pub const fn reads(&self) -> bool {
        self.kind.reads()
    }

    /// Returns `true` when this access writes or clobbers machine state.
    #[must_use]
    pub const fn writes(&self) -> bool {
        self.kind.writes()
    }

    /// Returns `true` when this machine-state access is structurally valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if let Some(0) = self.width_bits {
            return false;
        }
        match &self.location {
            NativeStateLocation::Register(register) => register.is_valid(),
            NativeStateLocation::RegisterClass(name)
            | NativeStateLocation::Flags(name)
            | NativeStateLocation::PredicateState(name)
            | NativeStateLocation::ControlRegister(name)
            | NativeStateLocation::Memory(name)
            | NativeStateLocation::Other(name) => !name.is_empty(),
            NativeStateLocation::StackPointer
            | NativeStateLocation::ProgramCounter
            | NativeStateLocation::VectorLength
            | NativeStateLocation::VectorConfig => true,
        }
    }
}

/// Abstract location clobbered by an opaque native operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NativeClobber {
    /// Structured machine-state access.
    MachineState(NativeStateAccess),
    /// Concrete target register or subregister.
    Register(NativeRegister),
    /// A target register or register alias class.
    RegisterClass(String),
    /// A target flag set such as x86 `eflags`.
    Flags(String),
    /// An abstract memory location or memory class.
    Memory(String),
    /// Target-specific state not represented by data or memory operands.
    Other(String),
}

impl NativeClobber {
    /// Returns `true` when this clobber touches register state.
    #[must_use]
    pub fn touches_registers(&self) -> bool {
        match self {
            Self::MachineState(access) => matches!(
                access.location,
                NativeStateLocation::Register(_) | NativeStateLocation::RegisterClass(_)
            ),
            Self::Register(_) | Self::RegisterClass(_) => true,
            Self::Flags(_) | Self::Memory(_) | Self::Other(_) => false,
        }
    }

    /// Returns `true` when this clobber touches flags or condition-code state.
    #[must_use]
    pub fn touches_flags(&self) -> bool {
        match self {
            Self::MachineState(access) => matches!(access.location, NativeStateLocation::Flags(_)),
            Self::Flags(_) => true,
            Self::Register(_) | Self::RegisterClass(_) | Self::Memory(_) | Self::Other(_) => false,
        }
    }

    /// Returns `true` when this clobber touches memory state.
    #[must_use]
    pub fn touches_memory(&self) -> bool {
        match self {
            Self::MachineState(access) => matches!(access.location, NativeStateLocation::Memory(_)),
            Self::Memory(_) => true,
            Self::Register(_) | Self::RegisterClass(_) | Self::Flags(_) | Self::Other(_) => false,
        }
    }
}

impl fmt::Display for AtomicRmwOp {
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
        }
    }
}

/// Bitmask for selecting flag bits from a flags-defining operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Creates a flag mask from raw bits.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
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

    /// Returns the union of two flag masks.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
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
        }
    }
}

impl fmt::Display for FlagsMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        if self.0 & Self::CARRY.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "CF")?;
            first = false;
        }
        if self.0 & Self::PARITY.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "PF")?;
            first = false;
        }
        if self.0 & Self::ADJUST.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "AF")?;
            first = false;
        }
        if self.0 & Self::ZERO.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "ZF")?;
            first = false;
        }
        if self.0 & Self::SIGN.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "SF")?;
            first = false;
        }
        if self.0 & Self::OVERFLOW.0 != 0 {
            if !first {
                write!(f, ",")?;
            }
            write!(f, "OF")?;
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
}

/// Describes how an instruction writes one native status flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl fmt::Display for FlagCondition {
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

impl FlagCondition {
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

/// Kind of binary operation for extracted binary op info.
///
/// This enum categorizes all binary operations in `SsaOp` for uniform
/// handling in optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Rotate through carry left
    Rcl,
    /// Rotate through carry right.
    Rcr,
}

impl fmt::Display for BinaryOpKind {
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
            Self::Rcl => write!(f, "rcl"),
            Self::Rcr => write!(f, "rcr"),
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnaryOpInfo {
    /// The kind of unary operation.
    pub kind: UnaryOpKind,
    /// Destination variable for the result.
    pub dest: SsaVarId,
    /// The operand.
    pub operand: SsaVarId,
}
