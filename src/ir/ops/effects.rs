//! Memory- and side-effect classification for [`SsaOp`].
//!
//! Defines the effect vocabulary ([`SsaEffects`] and friends) and implements
//! the queries the generic passes rely on to decide whether an operation may
//! be reordered, duplicated, or deleted: [`SsaOp::effects`],
//! [`SsaOp::may_throw`], [`SsaOp::is_pure`], and [`SsaOp::memory_effect`].
//!
//! DCE, GVN and LICM consult these, so a mis-classified variant is a
//! miscompile rather than a missed optimization. The test battery
//! cross-checks the predicates against each other for every variant.

use crate::{
    ir::{
        ops::{
            def::SsaOp,
            kinds::AtomicOrdering,
            vector::{PredicateGenKind, PredicateOpKind},
        },
        variable::SsaVarId,
    },
    target::Target,
};

/// Direct single-address memory access extracted from an operation's payload.
///
/// Returned by [`SsaOp::memory_effect`] for operations that read or write
/// memory through exactly one address variable (indirect loads/stores, atomic
/// accesses, vector loads/stores, block/object initialization). Operations
/// with two address operands (`CopyBlk`, `CopyObj`) or structured addressing
/// payloads (field, element, gather/scatter, segment) are not representable
/// here and return `None`; hosts read those payloads directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryEffect<'a, T: Target> {
    /// SSA variable holding the accessed address.
    pub addr: SsaVarId,
    /// Whether the operation reads memory at `addr`.
    pub reads: bool,
    /// Whether the operation writes memory at `addr`.
    pub writes: bool,
    /// Accessed value type when the payload carries one.
    pub value_type: Option<&'a T::Type>,
}

/// Coarse memory/effect class for an SSA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SsaEffectKind {
    /// No side effects and no memory dependency.
    Pure,
    /// Reads memory without writing it.
    Read,
    /// Writes memory without requiring the previous value.
    Write,
    /// Reads and writes memory.
    ReadWrite,
    /// Acts as a memory ordering barrier.
    Fence,
    /// Performs an atomic memory operation.
    Atomic,
    /// Calls unknown host code.
    Call,
    /// Has target-specific effects not otherwise modeled.
    Opaque,
}

/// Abstract memory location class used by effect summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryEffectLocation {
    /// No memory location is involved.
    None,
    /// Precise location is unknown or target-specific.
    Unknown,
    /// Stack memory.
    Stack,
    /// Managed or native heap memory.
    Heap,
    /// Global or static storage.
    Global,
    /// Code memory.
    Code,
    /// Memory-mapped or port-backed I/O.
    Io,
}

/// Detailed memory access semantics for an SSA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryAccessSemantics {
    /// No memory access semantics apply.
    None,
    /// Ordinary memory access.
    Normal,
    /// Volatile memory access.
    Volatile,
    /// Atomic memory access.
    Atomic,
    /// Memory fence or barrier.
    Fence,
    /// Target-specific access whose semantics are opaque.
    Opaque,
}

/// Trap or fault class associated with an SSA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrapClass {
    /// Operation cannot trap.
    None,
    /// Operation may fault for an unknown or target-specific reason.
    Unknown,
    /// Operation may fault on invalid memory access.
    MemoryFault,
    /// Operation may fault on null reference or pointer access.
    NullAccess,
    /// Operation may fault on array bounds checks.
    Bounds,
    /// Operation may fault on integer division by zero.
    DivideByZero,
    /// Operation may fault on arithmetic overflow.
    Overflow,
    /// Operation may fault on invalid type conversion or cast.
    InvalidCast,
    /// Operation transfers a language-level exception.
    UserThrow,
    /// Operation may fault as an illegal or privileged instruction.
    IllegalInstruction,
}

/// Control-flow constraint imposed by an SSA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ControlEffect {
    /// Does not constrain control flow.
    None,
    /// Terminates the current basic block.
    Terminator,
    /// Calls target code and may transfer control externally.
    Call,
    /// Returns from the current function or handler.
    Return,
    /// Throws or resumes exception propagation.
    Throw,
    /// Target-specific control-flow constraint.
    Opaque,
}

/// Summary of an SSA operation's effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SsaEffects {
    /// Coarse effect class.
    pub kind: SsaEffectKind,
    /// Whether the operation may throw or trap.
    pub may_throw: bool,
    /// Abstract memory location class touched by the operation.
    pub memory: MemoryEffectLocation,
    /// Detailed memory access semantics.
    pub memory_semantics: MemoryAccessSemantics,
    /// Whether the memory access is volatile.
    pub volatile: bool,
    /// Atomic ordering when the operation has atomic or fence semantics.
    pub ordering: Option<AtomicOrdering>,
    /// Trap or fault class when known.
    pub trap: TrapClass,
    /// Control-flow constraint imposed by the operation.
    pub control: ControlEffect,
}

impl SsaEffects {
    /// Returns a pure, non-throwing effect summary.
    #[must_use]
    pub const fn pure() -> Self {
        Self {
            kind: SsaEffectKind::Pure,
            may_throw: false,
            memory: MemoryEffectLocation::None,
            memory_semantics: MemoryAccessSemantics::None,
            volatile: false,
            ordering: None,
            trap: TrapClass::None,
            control: ControlEffect::None,
        }
    }

    /// Creates an effect summary from an effect class and trap flag.
    #[must_use]
    pub const fn new(kind: SsaEffectKind, may_throw: bool) -> Self {
        let memory_semantics = match kind {
            SsaEffectKind::Pure => MemoryAccessSemantics::None,
            SsaEffectKind::Fence => MemoryAccessSemantics::Fence,
            SsaEffectKind::Atomic => MemoryAccessSemantics::Atomic,
            SsaEffectKind::Opaque | SsaEffectKind::Call => MemoryAccessSemantics::Opaque,
            SsaEffectKind::Read | SsaEffectKind::Write | SsaEffectKind::ReadWrite => {
                MemoryAccessSemantics::Normal
            }
        };
        let memory = match kind {
            SsaEffectKind::Pure | SsaEffectKind::Fence => MemoryEffectLocation::None,
            SsaEffectKind::Read
            | SsaEffectKind::Write
            | SsaEffectKind::ReadWrite
            | SsaEffectKind::Atomic
            | SsaEffectKind::Call
            | SsaEffectKind::Opaque => MemoryEffectLocation::Unknown,
        };
        let trap = if may_throw {
            TrapClass::Unknown
        } else {
            TrapClass::None
        };
        let control = match kind {
            SsaEffectKind::Call => ControlEffect::Call,
            SsaEffectKind::Opaque => ControlEffect::Opaque,
            SsaEffectKind::Pure
            | SsaEffectKind::Read
            | SsaEffectKind::Write
            | SsaEffectKind::ReadWrite
            | SsaEffectKind::Fence
            | SsaEffectKind::Atomic => ControlEffect::None,
        };
        Self {
            kind,
            may_throw,
            memory,
            memory_semantics,
            volatile: false,
            ordering: None,
            trap,
            control,
        }
    }

    /// Returns this summary with a refined memory location class.
    #[must_use]
    pub const fn with_memory(mut self, memory: MemoryEffectLocation) -> Self {
        self.memory = memory;
        self
    }

    /// Returns this summary with volatile memory semantics.
    #[must_use]
    pub const fn volatile(mut self) -> Self {
        self.volatile = true;
        self
    }

    /// Returns this summary with atomic memory semantics and ordering.
    #[must_use]
    pub const fn atomic_ordering(mut self, ordering: AtomicOrdering) -> Self {
        self.memory_semantics = MemoryAccessSemantics::Atomic;
        self.ordering = Some(ordering);
        self
    }

    /// Returns this summary with fence semantics and ordering.
    #[must_use]
    pub const fn fence_ordering(mut self, ordering: AtomicOrdering) -> Self {
        self.memory_semantics = MemoryAccessSemantics::Fence;
        self.ordering = Some(ordering);
        self
    }

    /// Returns this summary with a known trap class.
    #[must_use]
    pub const fn with_trap(mut self, trap: TrapClass) -> Self {
        self.trap = trap;
        self.may_throw = !matches!(trap, TrapClass::None);
        self
    }

    /// Returns this summary with a known control-flow constraint.
    #[must_use]
    pub const fn with_control(mut self, control: ControlEffect) -> Self {
        self.control = control;
        if matches!(self.kind, SsaEffectKind::Opaque)
            && !matches!(control, ControlEffect::None | ControlEffect::Opaque)
        {
            self.memory = MemoryEffectLocation::None;
            self.memory_semantics = MemoryAccessSemantics::None;
        }
        self
    }

    /// Returns `true` if the operation has no side effects and cannot trap.
    #[must_use]
    pub const fn is_pure(self) -> bool {
        matches!(self.kind, SsaEffectKind::Pure) && !self.may_throw
    }

    /// Returns `true` if the operation may read memory.
    #[must_use]
    pub const fn reads_memory(self) -> bool {
        matches!(
            self.kind,
            SsaEffectKind::Read
                | SsaEffectKind::ReadWrite
                | SsaEffectKind::Atomic
                | SsaEffectKind::Call
                | SsaEffectKind::Opaque
        )
    }

    /// Returns `true` if the operation may write memory.
    #[must_use]
    pub const fn writes_memory(self) -> bool {
        matches!(
            self.kind,
            SsaEffectKind::Write
                | SsaEffectKind::ReadWrite
                | SsaEffectKind::Atomic
                | SsaEffectKind::Call
                | SsaEffectKind::Opaque
        )
    }

    /// Returns `true` if the operation can be removed when all definitions are unused.
    #[must_use]
    pub const fn removable_when_unused(self) -> bool {
        self.is_pure()
    }
}

impl<T: Target> SsaOp<T> {
    /// Returns `true` if this operation may throw an exception.
    #[must_use]
    pub const fn may_throw(&self) -> bool {
        matches!(
            self,
            Self::Div { .. }
                | Self::Rem { .. }
                | Self::FloatCompareFlags {
                    signaling: true,
                    ..
                }
                | Self::WideDiv { .. }
                | Self::AddOvf { .. }
                | Self::SubOvf { .. }
                | Self::MulOvf { .. }
                | Self::IntConv {
                    overflow_check: true,
                    ..
                }
                | Self::FloatToInt {
                    overflow_check: true,
                    ..
                }
                | Self::LoadField { .. }
                | Self::StoreField { .. }
                | Self::LoadStaticField { .. }
                | Self::StoreStaticField { .. }
                | Self::LoadElement { .. }
                | Self::StoreElement { .. }
                | Self::LoadElementAddr { .. }
                | Self::LoadIndirect { .. }
                | Self::StoreIndirect { .. }
                | Self::LoadObj { .. }
                | Self::StoreObj { .. }
                | Self::InitObj { .. }
                | Self::CopyObj { .. }
                | Self::InitBlk { .. }
                | Self::CopyBlk { .. }
                | Self::NewObj { .. }
                | Self::NewArr { .. }
                | Self::CastClass { .. }
                | Self::Unbox { .. }
                | Self::UnboxAny { .. }
                | Self::Call { .. }
                | Self::CallVirt { .. }
                | Self::CallIndirect { .. }
                | Self::Throw { .. }
                | Self::Rethrow
                | Self::Break(_)
                | Self::Ckfinite { .. }
                | Self::CmpXchg { .. }
                | Self::WideCompareExchange { .. }
                | Self::AtomicRmw { .. }
                | Self::AtomicLoad { .. }
                | Self::AtomicStore { .. }
                | Self::AtomicPairLoad { .. }
                | Self::AtomicPairStoreConditional { .. }
                | Self::AtomicExchange { .. }
                | Self::AtomicLockRmw { .. }
                | Self::AtomicStoreConditional { .. }
                | Self::AtomicCmpXchg { .. }
                | Self::AtomicPairCmpXchg { .. }
                | Self::VectorLoad { .. }
                | Self::VectorMaskedLoad { .. }
                | Self::VectorBroadcastLoad { .. }
                | Self::VectorGather { .. }
                | Self::VectorFaultingLoad { .. }
                | Self::VectorSegmentLoad { .. }
                | Self::VectorPackLoad { .. }
                | Self::VectorStructLoadReplicate(_)
                | Self::VectorStore { .. }
                | Self::VectorMaskedStore { .. }
                | Self::VectorScatter { .. }
                | Self::VectorSegmentStore { .. }
                | Self::VectorPackStore { .. }
        ) || matches!(self, Self::NativeOpaque(data) if data.effects.may_throw)
            || matches!(self, Self::SystemOp(data) if data.kind.effects().may_throw)
            || matches!(self, Self::ComputeOp(data) if data.kind.effects().may_throw)
            || matches!(self, Self::BcdAdjust(data) if data.kind.effects().may_throw)
            || matches!(self, Self::VectorCrypto(data) if data.kind.effects().may_throw)
            || matches!(self, Self::TileOp(data) if data.kind.effects().may_throw)
            || matches!(self, Self::BlockString(data) if data.kind.effects().may_throw)
    }

    /// Returns the memory and trapping effects of this operation.
    #[must_use]
    pub const fn effects(&self) -> SsaEffects {
        match self {
            Self::LoadField { .. }
            | Self::LoadStaticField { .. }
            | Self::LoadElement { .. }
            | Self::LoadIndirect { .. }
            | Self::LoadObj { .. }
            | Self::VectorLoad { .. }
            | Self::VectorMaskedLoad { .. }
            | Self::VectorBroadcastLoad { .. }
            | Self::VectorGather { .. }
            | Self::VectorFaultingLoad { .. }
            | Self::VectorSegmentLoad { .. }
            | Self::VectorPackLoad { .. }
            | Self::VectorStructLoadReplicate(_) => {
                SsaEffects::new(SsaEffectKind::Read, self.may_throw())
                    .with_trap(TrapClass::MemoryFault)
            }

            Self::StoreField { .. }
            | Self::StoreStaticField { .. }
            | Self::StoreElement { .. }
            | Self::StoreIndirect { .. }
            | Self::StoreObj { .. }
            | Self::InitObj { .. }
            | Self::VectorStore { .. }
            | Self::VectorMaskedStore { .. }
            | Self::VectorScatter { .. }
            | Self::VectorSegmentStore { .. }
            | Self::VectorPackStore { .. } => {
                SsaEffects::new(SsaEffectKind::Write, self.may_throw())
                    .with_trap(TrapClass::MemoryFault)
            }

            Self::CopyBlk { .. } | Self::InitBlk { .. } | Self::CopyObj { .. } => {
                SsaEffects::new(SsaEffectKind::ReadWrite, self.may_throw())
                    .with_trap(TrapClass::MemoryFault)
            }

            Self::CmpXchg { .. } | Self::AtomicRmw { .. } => {
                SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                    .atomic_ordering(AtomicOrdering::SeqCst)
                    .with_trap(TrapClass::MemoryFault)
            }

            Self::AtomicLoad {
                ordering, volatile, ..
            }
            | Self::AtomicStore {
                ordering, volatile, ..
            }
            | Self::AtomicPairLoad {
                ordering, volatile, ..
            }
            | Self::AtomicExchange {
                ordering, volatile, ..
            }
            | Self::AtomicLockRmw {
                ordering, volatile, ..
            } => {
                let effects = SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                    .atomic_ordering(*ordering)
                    .with_trap(TrapClass::MemoryFault);
                if *volatile {
                    effects.volatile()
                } else {
                    effects
                }
            }

            Self::AtomicCmpXchg {
                success_ordering,
                volatile,
                ..
            }
            | Self::AtomicStoreConditional {
                success_ordering,
                volatile,
                ..
            }
            | Self::AtomicPairStoreConditional {
                success_ordering,
                volatile,
                ..
            }
            | Self::AtomicPairCmpXchg {
                success_ordering,
                volatile,
                ..
            } => {
                let effects = SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                    .atomic_ordering(*success_ordering)
                    .with_trap(TrapClass::MemoryFault);
                if *volatile {
                    effects.volatile()
                } else {
                    effects
                }
            }

            Self::Fence { kind } => {
                SsaEffects::new(SsaEffectKind::Fence, false).fence_ordering(kind.ordering())
            }
            Self::NativeOpaque(data) => data.effects,
            Self::SystemOp(data) => data.kind.effects(),
            Self::ComputeOp(data) => data.kind.effects(),
            Self::BcdAdjust(data) => data.kind.effects(),
            Self::VectorCrypto(data) => data.kind.effects(),
            Self::TileOp(data) => data.kind.effects(),
            // Pure value-producing native/vector compute: no memory,
            // no traps, no control effects — results depend only on operands.
            Self::VectorPermute(_)
            | Self::VectorMultiplyAdd(_)
            | Self::VectorPackNarrow(_)
            | Self::VectorNarrowSaturate(_)
            | Self::VectorPredicateWhile(_)
            | Self::VectorPredicateBreak(_)
            | Self::VectorComplexAdd(_)
            | Self::VectorCountAdjust(_)
            | Self::VectorExtendInLane(_)
            | Self::VectorElementCount(_)
            | Self::VectorSveAddressGen(_)
            | Self::VectorSmeMisc(_)
            | Self::VectorSveCompute(_)
            | Self::VectorReverseChunks(_)
            | Self::VectorMatrixMulAcc(_)
            | Self::VectorSmeOuterProduct(_)
            | Self::VectorFpHelper(_)
            | Self::VectorSvePermute(_)
            | Self::VectorTernaryLogic(_)
            | Self::VectorDotProduct(_)
            | Self::VectorMultiSad(_)
            | Self::VectorIntDotProduct(_)
            | Self::VectorStringCompare(_)
            | Self::VectorBitfield(_)
            | Self::VectorIntersect(_)
            | Self::VectorShuffleBits(_)
            | Self::VectorConditionalMove(_)
            | Self::VectorHorizontalMinPos(_)
            | Self::VectorComplexMul(_)
            | Self::VectorClassify(_)
            | Self::VectorHorizontalReduce(_) => SsaEffects::new(SsaEffectKind::Pure, false),
            // `rdffr` *reads* the SVE first-fault register — the same hidden
            // machine state `setffr`/`wrffr` write below. Since the FFR is not an
            // SSA operand, `rdffr` is not a function of its SSA operands either,
            // and its `inputs` list is typically empty. Classifying it Pure lets
            // GVN collapse every `rdffr` in a function onto the first (they all
            // normalize to one key) and lets LICM hoist it out of the loop
            // containing the first-faulting load that sets the state it reports.
            Self::VectorPredicateGen(data) => match data.kind {
                PredicateGenKind::ReadFfr => SsaEffects::new(SsaEffectKind::Opaque, false),
                // The rest are genuine functions of their operands.
                PredicateGenKind::True
                | PredicateGenKind::False
                | PredicateGenKind::Next
                | PredicateGenKind::First
                | PredicateGenKind::UnpackHi
                | PredicateGenKind::UnpackLo
                | PredicateGenKind::Select
                | PredicateGenKind::HazardRw
                | PredicateGenKind::HazardWr => SsaEffects::new(SsaEffectKind::Pure, false),
            },
            // `setffr`/`wrffr` write the SVE first-fault register, which is not
            // modeled as an SSA operand and whose `outputs` may be empty. Pure +
            // zero defs means DCE deletes them outright, silently dropping the
            // FFR initialization a following first-faulting load depends on.
            Self::VectorPredicateOp(data) => match data.op {
                PredicateOpKind::SetFirstFault | PredicateOpKind::WriteFirstFault => {
                    SsaEffects::new(SsaEffectKind::Opaque, false)
                }
                _ => SsaEffects::new(SsaEffectKind::Pure, false),
            },
            // Derived from the kind, never from the variant: an adjustment
            // confined to the status flags the IR models as SSA values is a
            // pure function of its operands, while one writing DF or AC is
            // opaque -- the `FpuControl` treatment two arms above, for the same
            // reason. A future variant inherits the right class rather than the
            // convenient one, and a kind that declares a result but carries no
            // output degrades to opaque instead of becoming deletable.
            Self::FlagAdjust(data) => data.kind.effects_for_outputs(data.outputs.len()),
            Self::BlockString(data) => data.kind.effects(),
            Self::WideCompareExchange { .. } => {
                SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                    .atomic_ordering(AtomicOrdering::SeqCst)
                    .with_trap(TrapClass::MemoryFault)
            }

            // Transcendentals compute a value (pure, modulo FP exception flags);
            // FPU control ops mutate FPU control/status/tag state — a barrier.
            Self::FpTranscendental { .. } => SsaEffects::new(SsaEffectKind::Pure, false),
            Self::FpuControl { .. } => SsaEffects::new(SsaEffectKind::Opaque, false),

            Self::FloatCompareFlags { signaling, .. } => {
                if *signaling {
                    SsaEffects::new(SsaEffectKind::Pure, true).with_trap(TrapClass::Unknown)
                } else {
                    SsaEffects::new(SsaEffectKind::Pure, false)
                }
            }

            Self::Call { .. } | Self::CallVirt { .. } | Self::CallIndirect { .. } => {
                SsaEffects::new(SsaEffectKind::Call, true).with_control(ControlEffect::Call)
            }

            Self::Jump { .. }
            | Self::Branch { .. }
            | Self::BranchCmp { .. }
            | Self::BranchFlags { .. }
            | Self::IndirectBranch { .. }
            | Self::Switch { .. } => SsaEffects::new(SsaEffectKind::Opaque, false)
                .with_control(ControlEffect::Terminator),

            Self::Return { .. } | Self::Leave { .. } => {
                SsaEffects::new(SsaEffectKind::Opaque, false).with_control(ControlEffect::Return)
            }

            Self::NewObj { .. }
            | Self::NewArr { .. }
            | Self::CastClass { .. }
            | Self::Unbox { .. }
            | Self::UnboxAny { .. }
            | Self::Box { .. }
            | Self::LocalAlloc { .. } => SsaEffects::new(SsaEffectKind::Opaque, self.may_throw()),

            Self::Throw { .. } | Self::Rethrow => SsaEffects::new(SsaEffectKind::Opaque, true)
                .with_trap(TrapClass::UserThrow)
                .with_control(ControlEffect::Throw),

            Self::EndFinally
            | Self::EndFilter { .. }
            | Self::InterruptReturn
            | Self::Unreachable => SsaEffects::new(SsaEffectKind::Opaque, self.may_throw())
                .with_control(ControlEffect::Terminator),

            Self::Break(_) => SsaEffects::new(SsaEffectKind::Opaque, self.may_throw())
                .with_trap(TrapClass::IllegalInstruction),

            Self::Constrained { .. }
            | Self::Volatile
            | Self::Unaligned { .. }
            | Self::TailPrefix
            | Self::Readonly => SsaEffects::new(SsaEffectKind::Opaque, self.may_throw()),

            Self::VectorZeroUpper { .. } => SsaEffects::new(SsaEffectKind::Opaque, false),

            // Pure value-producing ops: scalar/vector arithmetic, bitwise,
            // comparison, boolean, bit-manipulation, conversions, address
            // computation, and SSA bookkeeping. The `may_throw` bit is threaded
            // from `may_throw()` so the trapping members of this group
            // (`Div`/`Rem`/`WideDiv`, the overflow-checked arithmetic,
            // `Ckfinite`, checked `Conv`, `LoadElementAddr`) report a trap while
            // the rest stay non-throwing. `Rol`/`Ror` belong here; the
            // carry-coupled `Rcl`/`Rcr` do NOT (see the opaque arm below).
            Self::Add { .. }
            | Self::Sub { .. }
            | Self::Mul { .. }
            | Self::WideMul { .. }
            | Self::Div { .. }
            | Self::Rem { .. }
            | Self::WideDiv { .. }
            | Self::AddOvf { .. }
            | Self::SubOvf { .. }
            | Self::MulOvf { .. }
            | Self::And { .. }
            | Self::Or { .. }
            | Self::Xor { .. }
            | Self::Neg { .. }
            | Self::Not { .. }
            | Self::Shl { .. }
            | Self::Shr { .. }
            | Self::Rol { .. }
            | Self::Ror { .. }
            | Self::Ceq { .. }
            | Self::Cgt { .. }
            | Self::Clt { .. }
            | Self::BoolAnd { .. }
            | Self::BoolOr { .. }
            | Self::BoolXor { .. }
            | Self::BoolNot { .. }
            | Self::BSwap { .. }
            | Self::BRev { .. }
            | Self::BitScanForward { .. }
            | Self::BitScanReverse { .. }
            | Self::Popcount { .. }
            | Self::Parity { .. }
            | Self::Bitcast { .. }
            | Self::IntConv { .. }
            | Self::IntToPtr { .. }
            | Self::PtrToInt { .. }
            | Self::IntToFloat { .. }
            | Self::FloatToInt { .. }
            | Self::FloatConv { .. }
            | Self::Ckfinite { .. }
            | Self::FpClassify { .. }
            | Self::Const { .. }
            | Self::Copy { .. }
            | Self::Select { .. }
            | Self::SizeOf { .. }
            | Self::ReadFlags { .. }
            | Self::ComputeFlags { .. }
            | Self::CallClobber { .. }
            | Self::Phi { .. }
            | Self::Nop
            | Self::Pop { .. }
            | Self::LoadArg { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadElementAddr { .. }
            | Self::PtrAdd { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::LoadToken { .. }
            | Self::VectorUnary { .. }
            | Self::VectorBinary { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorExtract { .. }
            | Self::VectorInsert { .. }
            | Self::VectorSplat { .. }
            | Self::VectorShuffle { .. }
            | Self::VectorPack { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. }
            | Self::VectorMaskUnary { .. }
            | Self::VectorMaskBinary { .. } => {
                SsaEffects::new(SsaEffectKind::Pure, self.may_throw())
            }

            // Memory-reading value ops: read architectural/object state that is
            // not modelled as an SSA operand, so they are NOT pure (must not be
            // hoisted or value-numbered as pure). Throw bit threaded from
            // `may_throw()` for the reference-faulting members.
            Self::LoadLocal { .. }
            | Self::IsInst { .. }
            | Self::ArrayLength { .. }
            | Self::LoadVirtFunctionPtr { .. } => {
                SsaEffects::new(SsaEffectKind::Read, self.may_throw())
            }

            // Carry-coupled rotates read and write the carry flag, which is not
            // an explicit SSA operand, so they have a hidden input/output and
            // must never be value-numbered, reordered, or eliminated as pure.
            Self::Rcl { .. } | Self::Rcr { .. } => {
                SsaEffects::new(SsaEffectKind::Opaque, self.may_throw())
            } // NOTE: this match is intentionally exhaustive — there is NO `_`
              // catch-all. A newly added `SsaOp` variant must be classified here
              // explicitly or the crate will not compile, which is what keeps a
              // forgotten op from being silently treated as a pure value.
        }
    }

    /// Returns the direct single-address memory access for this operation.
    ///
    /// `Some` for operations that read or write memory through exactly one
    /// address variable: indirect loads/stores, atomic accesses, vector
    /// loads/stores, and single-destination block/object initialization. The
    /// access direction is semantic (a store-conditional writes; an exchange
    /// or read-modify-write both reads and writes). `None` for operations
    /// with two address operands (`CopyBlk`, `CopyObj`) and for structured
    /// addressing payloads (field, element, gather/scatter, segment), which
    /// hosts read directly from the payload.
    #[must_use]
    pub const fn memory_effect(&self) -> Option<MemoryEffect<'_, T>> {
        match self {
            Self::LoadIndirect {
                addr, value_type, ..
            }
            | Self::AtomicLoad {
                addr, value_type, ..
            } => Some(MemoryEffect {
                addr: *addr,
                reads: true,
                writes: false,
                value_type: Some(value_type),
            }),
            Self::StoreIndirect {
                addr, value_type, ..
            }
            | Self::AtomicStore {
                addr, value_type, ..
            }
            | Self::AtomicStoreConditional {
                addr, value_type, ..
            } => Some(MemoryEffect {
                addr: *addr,
                reads: false,
                writes: true,
                value_type: Some(value_type),
            }),
            Self::CmpXchg { addr, .. }
            | Self::AtomicRmw { addr, .. }
            | Self::AtomicExchange { addr, .. }
            | Self::AtomicLockRmw { addr, .. }
            | Self::AtomicCmpXchg { addr, .. }
            | Self::AtomicPairCmpXchg { addr, .. } => Some(MemoryEffect {
                addr: *addr,
                reads: true,
                writes: true,
                value_type: None,
            }),
            Self::AtomicPairLoad { addr, .. }
            | Self::VectorLoad { addr, .. }
            | Self::VectorMaskedLoad { addr, .. }
            | Self::VectorBroadcastLoad { addr, .. }
            | Self::VectorFaultingLoad { addr, .. }
            | Self::VectorPackLoad { addr, .. } => Some(MemoryEffect {
                addr: *addr,
                reads: true,
                writes: false,
                value_type: None,
            }),
            Self::AtomicPairStoreConditional { addr, .. }
            | Self::VectorStore { addr, .. }
            | Self::VectorMaskedStore { addr, .. }
            | Self::VectorPackStore { addr, .. } => Some(MemoryEffect {
                addr: *addr,
                reads: false,
                writes: true,
                value_type: None,
            }),
            Self::InitBlk { dest_addr, .. }
            | Self::InitObj { dest_addr, .. }
            | Self::StoreObj { dest_addr, .. } => Some(MemoryEffect {
                addr: *dest_addr,
                reads: false,
                writes: true,
                value_type: None,
            }),
            Self::LoadObj { src_addr, .. } => Some(MemoryEffect {
                addr: *src_addr,
                reads: true,
                writes: false,
                value_type: None,
            }),
            _ => None,
        }
    }

    /// Returns `true` if this operation is pure (has no side effects).
    ///
    /// Pure operations can be eliminated if their result is unused.
    #[must_use]
    pub const fn is_pure(&self) -> bool {
        self.effects().is_pure()
    }
}
