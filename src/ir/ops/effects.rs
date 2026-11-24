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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{
        content::ContentSink,
        ops::{calls::CallEffects, def::SsaOp, kinds::AtomicOrdering},
        variable::SsaVarId,
    },
    target::Target,
};

/// Direct single-address memory access extracted from an operation's payload.
///
/// Returned by [`SsaOp::memory_effect`] for operations that read or write
/// memory through exactly one address variable (indirect loads/stores, atomic
/// accesses, vector loads/stores, block/object initialization). Operations
/// with two address operands (`CopyBlk`, `MoveBlk`, `CopyObj`) or structured addressing
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    /// Produces a value the execution context supplies at this point.
    ///
    /// Nothing is read, written or trapped, so the operation costs nothing to
    /// delete when its result is dead. What it is *not* is movable: the value
    /// holds because control arrived here, so hoisting it, duplicating it into
    /// another block or numbering two of them as the same computation would all
    /// claim something the runtime never promised. Keeping it out of
    /// [`SsaEffects::is_pure`] is what buys that, since the rewriting passes ask
    /// exactly that question before they move or share an instruction.
    RuntimeDefined,
    /// Has target-specific effects not otherwise modeled.
    Opaque,
}

/// Abstract memory location class used by effect summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    ///
    /// The coarse `kind` drives the finer fields, so a caller sets one value
    /// rather than a group that must agree: it selects the memory-access
    /// semantics and location class (pure and fence touch no memory; reads,
    /// writes, atomics, calls and opaque ops touch an as-yet-`Unknown`
    /// location) and the control constraint (calls and opaque ops constrain
    /// control, the rest do not), while `may_throw` coarsens into an `Unknown`
    /// trap class. The builder methods ([`Self::with_memory`],
    /// [`Self::atomic_ordering`], [`Self::with_trap`] and friends) then refine
    /// any of these in place.
    #[must_use]
    pub const fn new(kind: SsaEffectKind, may_throw: bool) -> Self {
        let memory_semantics = match kind {
            SsaEffectKind::Pure | SsaEffectKind::RuntimeDefined => MemoryAccessSemantics::None,
            SsaEffectKind::Fence => MemoryAccessSemantics::Fence,
            SsaEffectKind::Atomic => MemoryAccessSemantics::Atomic,
            SsaEffectKind::Opaque | SsaEffectKind::Call => MemoryAccessSemantics::Opaque,
            SsaEffectKind::Read | SsaEffectKind::Write | SsaEffectKind::ReadWrite => {
                MemoryAccessSemantics::Normal
            }
        };
        let memory = match kind {
            SsaEffectKind::Pure | SsaEffectKind::RuntimeDefined | SsaEffectKind::Fence => {
                MemoryEffectLocation::None
            }
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
            | SsaEffectKind::RuntimeDefined
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
    ///
    /// Giving an `Opaque` operation a concrete control constraint (a
    /// terminator, call, return, or throw - anything but `None` or `Opaque`)
    /// also clears its memory location and semantics: the `Unknown` memory that
    /// [`Self::new`] attaches to every opaque op is a fallback for unmodelled
    /// state, and a plain control transfer does not touch memory, so it would
    /// otherwise over-report a dependency and block motion needlessly.
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

    /// Returns `true` if the operation may read data memory.
    ///
    /// An explicit no-memory location refines the coarse effect class without
    /// removing architectural state or control effects.
    #[must_use]
    pub const fn reads_memory(self) -> bool {
        !matches!(self.memory, MemoryEffectLocation::None)
            && matches!(
                self.kind,
                SsaEffectKind::Read
                    | SsaEffectKind::ReadWrite
                    | SsaEffectKind::Atomic
                    | SsaEffectKind::Call
                    | SsaEffectKind::Opaque
            )
    }

    /// Returns `true` if the operation may write data memory.
    ///
    /// An explicit no-memory location refines the coarse effect class without
    /// removing architectural state or control effects.
    #[must_use]
    pub const fn writes_memory(self) -> bool {
        !matches!(self.memory, MemoryEffectLocation::None)
            && matches!(
                self.kind,
                SsaEffectKind::Write
                    | SsaEffectKind::ReadWrite
                    | SsaEffectKind::Atomic
                    | SsaEffectKind::Call
                    | SsaEffectKind::Opaque
            )
    }

    /// Returns `true` if the operation can be removed when all definitions are unused.
    ///
    /// Wider than [`Self::is_pure`], and deliberately: an operation is
    /// removable when its only effect is its result, which a
    /// [`SsaEffectKind::RuntimeDefined`] operation satisfies while still
    /// refusing to be moved or shared.
    #[must_use]
    pub const fn removable_when_unused(self) -> bool {
        matches!(
            self.kind,
            SsaEffectKind::Pure | SsaEffectKind::RuntimeDefined
        ) && !self.may_throw
    }
}

impl<T: Target> SsaOp<T> {
    /// Returns what a call states it does, for a call whose host stated it;
    /// `None` for an operation that is not a call and for a call nobody
    /// described, which may do anything.
    #[must_use]
    pub fn stated_call_effects(&self) -> Option<&CallEffects<T>> {
        match self {
            Self::Call(call) => call.effects.as_ref(),
            _ => None,
        }
    }

    /// Returns `true` if this operation may raise an exception that leaves the
    /// function for the caller it unwinds to, when nothing in the function
    /// catches it: a call or a constructor whose callee may throw, or a throw.
    ///
    /// A fault - a memory access, a division, an overflow check - is not
    /// counted, though it [may throw](Self::may_throw) to the function's own
    /// handlers: whether a fault reaches the caller at all is the runtime's
    /// to say, a signal on native code and an exception on a managed one.
    #[must_use]
    pub fn may_unwind(&self) -> bool {
        matches!(
            self,
            Self::Call(_) | Self::NewObj { .. } | Self::Throw { .. } | Self::Rethrow
        ) && self.may_throw()
    }

    /// Returns `true` if this operation may throw an exception.
    #[must_use]
    pub fn may_throw(&self) -> bool {
        if let Some(effects) = self.stated_call_effects() {
            return effects.may_throw;
        }
        matches!(
            self,
            Self::Div { .. }
                | Self::Rem { .. }
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
                | Self::ArgStep { .. }
                | Self::StoreIndirect { .. }
                | Self::LoadObj { .. }
                | Self::StoreObj { .. }
                | Self::InitObj { .. }
                | Self::CopyObj { .. }
                | Self::InitBlk { .. }
                | Self::CopyBlk { .. }
                | Self::ArgNext { .. }
                | Self::MoveBlk { .. }
                | Self::CompareBlk(_)
                | Self::ScanBlk { .. }
                | Self::LoadBlk { .. }
                | Self::NewObj { .. }
                | Self::NewArr { .. }
                | Self::CastClass { .. }
                | Self::TypedRefValue { .. }
                | Self::Unbox { .. }
                | Self::UnboxAny { .. }
                | Self::Call(_)
                | Self::Throw { .. }
                | Self::Rethrow
                | Self::Break(_)
                | Self::Ckfinite { .. }
                | Self::AtomicCondRmw { .. }
                | Self::AtomicLoad { .. }
                | Self::AtomicStore { .. }
                | Self::AtomicPairLoad { .. }
                | Self::AtomicPairStoreConditional { .. }
                | Self::AtomicExchange { .. }
                | Self::AtomicLockRmw { .. }
                | Self::AtomicStoreConditional { .. }
                | Self::AtomicCmpXchg { .. }
                | Self::AtomicPairCmpXchg { .. }
                | Self::AtomicPairRmw { .. }
                | Self::AtomicCheckedRmw { .. }
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
        ) || matches!(self, Self::SystemOp(data) if data.kind.effects().may_throw)
            || matches!(self, Self::Compute(data) if data.kind.effects().may_throw)
    }

    /// Returns the memory and trapping effects of this operation.
    ///
    /// A call reports what its [`CallEffects`] summarize to
    /// ([`CallEffects::summary`]), or, when nobody stated them, what any call
    /// may do.
    #[must_use]
    pub fn effects(&self) -> SsaEffects {
        if let Some(effects) = self.stated_call_effects() {
            return effects.summary();
        }
        let effects = self.unqualified_effects();
        if self.is_volatile() {
            effects.volatile()
        } else {
            effects
        }
    }

    /// Whether a memory access this operation makes is volatile
    /// ([`AccessQualifiers::volatile`](crate::ir::ops::AccessQualifiers::volatile)).
    #[must_use]
    pub fn is_volatile(&self) -> bool {
        let mut volatile = false;
        self.visit_qualifiers(|qualifiers| volatile |= qualifiers.volatile);
        volatile
    }

    /// What this operation does before its accesses' qualifiers are read.
    fn unqualified_effects(&self) -> SsaEffects {
        match self {
            Self::LoadField { .. }
            | Self::LoadStaticField { .. }
            | Self::LoadElement { .. }
            | Self::LoadIndirect { .. }
            | Self::ArgStep { .. }
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

            Self::CopyBlk { .. }
            | Self::MoveBlk { .. }
            | Self::InitBlk { .. }
            | Self::ArgNext { .. }
            | Self::CopyObj { .. } => SsaEffects::new(SsaEffectKind::ReadWrite, self.may_throw())
                .with_trap(TrapClass::MemoryFault),

            // The block compares, scans and loads read memory and write none.
            Self::CompareBlk(_) | Self::ScanBlk { .. } | Self::LoadBlk { .. } => {
                SsaEffects::new(SsaEffectKind::Read, self.may_throw())
                    .with_trap(TrapClass::MemoryFault)
            }

            Self::AtomicLoad { ordering, .. }
            | Self::AtomicStore { ordering, .. }
            | Self::AtomicPairLoad { ordering, .. }
            | Self::AtomicExchange { ordering, .. }
            | Self::AtomicLockRmw { ordering, .. } => {
                SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                    .atomic_ordering(*ordering)
                    .with_trap(TrapClass::MemoryFault)
            }

            Self::AtomicCmpXchg {
                success_ordering, ..
            }
            | Self::AtomicStoreConditional {
                success_ordering, ..
            }
            | Self::AtomicPairStoreConditional {
                success_ordering, ..
            }
            | Self::AtomicPairCmpXchg {
                success_ordering, ..
            }
            | Self::AtomicPairRmw {
                ordering: success_ordering,
                ..
            }
            | Self::AtomicCheckedRmw {
                ordering: success_ordering,
                ..
            }
            | Self::AtomicCondRmw {
                ordering: success_ordering,
                ..
            } => SsaEffects::new(SsaEffectKind::Atomic, self.may_throw())
                .atomic_ordering(*success_ordering)
                .with_trap(TrapClass::MemoryFault),

            Self::Fence { kind } => {
                SsaEffects::new(SsaEffectKind::Fence, false).fence_ordering(kind.ordering())
            }
            Self::SystemOp(data) => data.kind.effects(),
            Self::Compute(data) => data.kind.effects(),
            // Pure value-producing native/vector compute: no memory,
            // no traps, no control effects - results depend only on operands.
            Self::Contract(_)
            | Self::VectorPredicateWhile(_)
            | Self::VectorExtendInLane(_)
            | Self::VectorElementCount(_)
            | Self::VectorSveCompute(_) => SsaEffects::new(SsaEffectKind::Pure, false),

            // A float function computes its results, condition codes
            // included, from its operands (the exceptions it raises are a
            // `FloatRaise` of its own); FPU control ops mutate FPU
            // control/status/tag state, and touch memory only as the kind says.
            Self::FloatFunction { .. } => SsaEffects::new(SsaEffectKind::Pure, false),
            Self::FpuControl(data) => data.kind.effects(),

            // A float operation does not trap: the exceptions it raises are
            // flags it sets ([`Self::FloatRaise`]), as they are where the
            // program leaves them masked. An unmasked exception's trap is not
            // modelled, a signalling compare's no more than an addition's.
            Self::FloatCompareFlags { .. } | Self::FloatRaise { .. } => {
                SsaEffects::new(SsaEffectKind::Pure, false)
            }

            Self::Call(_) => {
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
            | Self::TypedRefValue { .. }
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

            // Pure value-producing ops: scalar/vector arithmetic, bitwise,
            // comparison, boolean, bit-manipulation, conversions, address
            // computation, and SSA bookkeeping. The `may_throw` bit is threaded
            // from `may_throw()` so the trapping members of this group
            // (`Div`/`Rem`/`WideDiv`, the overflow-checked arithmetic,
            // `Ckfinite`, checked `Conv`, `LoadElementAddr`) report a trap while
            // the rest stay non-throwing.
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
            | Self::MergeFlags { .. }
            | Self::InsertFlag { .. }
            | Self::Phi { .. }
            | Self::Nop
            | Self::Pop { .. }
            | Self::LoadArgAddr { .. }
            | Self::LoadLocalAddr { .. }
            | Self::LoadFieldAddr { .. }
            | Self::LoadStaticFieldAddr { .. }
            | Self::LoadElementAddr { .. }
            | Self::PtrAdd { .. }
            | Self::LoadFunctionPtr { .. }
            | Self::LoadToken { .. }
            | Self::ArgList { .. }
            | Self::MakeTypedRef { .. }
            | Self::TypedRefType { .. }
            | Self::VectorUnary { .. }
            | Self::VectorBinary { .. }
            | Self::VectorTernary { .. }
            | Self::VectorPredicatedUnary { .. }
            | Self::VectorPredicatedBinary { .. }
            | Self::VectorPredicatedTernary { .. }
            | Self::VectorCompare { .. }
            | Self::VectorCast { .. }
            | Self::VectorReinterpret { .. }
            | Self::VectorSlice { .. }
            | Self::VectorInsertSlice { .. }
            | Self::VectorConcat { .. }
            | Self::VectorExtract { .. }
            | Self::VectorInsert { .. }
            | Self::VectorSplat { .. }
            | Self::VectorShuffle { .. }
            | Self::VectorPack { .. }
            | Self::VectorReduce { .. }
            | Self::VectorBitmask { .. } => SsaEffects::new(SsaEffectKind::Pure, self.may_throw()),

            // A call's outputs are what that call left in each register: they
            // hold right after it and nowhere else.
            Self::RuntimeValue { .. } | Self::CallClobber { .. } => {
                SsaEffects::new(SsaEffectKind::RuntimeDefined, false)
            }

            // Memory-reading value ops: read architectural/object state that is
            // not modelled as an SSA operand, so they are NOT pure (must not be
            // hoisted or value-numbered as pure). Throw bit threaded from
            // `may_throw()` for the reference-faulting members. A `LoadLocal` or
            // `LoadArg` that survives SSA construction reads an address-taken
            // slot, which a `StoreIndirect` through its address can change.
            Self::LoadLocal { .. }
            | Self::LoadArg { .. }
            | Self::IsInst { .. }
            | Self::ArrayLength { .. }
            | Self::LoadVirtFunctionPtr { .. } => {
                SsaEffects::new(SsaEffectKind::Read, self.may_throw())
            } // catch-all. A newly added `SsaOp` variant must be classified here
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
    /// with two address operands (`CopyBlk`, `MoveBlk`, `CopyObj`) and for structured
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
            Self::AtomicExchange { addr, .. }
            | Self::AtomicCondRmw { addr, .. }
            | Self::AtomicLockRmw { addr, .. }
            | Self::AtomicCmpXchg { addr, .. }
            | Self::AtomicPairCmpXchg { addr, .. }
            | Self::AtomicPairRmw { addr, .. }
            | Self::AtomicCheckedRmw { addr, .. } => Some(MemoryEffect {
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
    pub fn is_pure(&self) -> bool {
        self.effects().is_pure()
    }
}

impl<T: Target> Emit<T> for ControlEffect {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::None => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Terminator => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Call => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Return => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Throw => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Opaque => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for MemoryAccessSemantics {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::None => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Normal => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Volatile => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Atomic => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Fence => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Opaque => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for MemoryEffectLocation {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::None => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Unknown => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Stack => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Heap => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Global => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Code => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Io => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for SsaEffectKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Pure => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Read => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Write => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::ReadWrite => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Fence => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Atomic => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Call => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::RuntimeDefined => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::Opaque => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for SsaEffects {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            kind,
            may_throw,
            memory,
            memory_semantics,
            volatile,
            ordering,
            trap,
            control,
        } = self;
        Emit::emit(kind, writer)?;
        Emit::emit(may_throw, writer)?;
        Emit::emit(memory, writer)?;
        Emit::emit(memory_semantics, writer)?;
        Emit::emit(volatile, writer)?;
        Emit::emit(ordering, writer)?;
        Emit::emit(trap, writer)?;
        Emit::emit(control, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for TrapClass {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::None => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Unknown => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::MemoryFault => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::NullAccess => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Bounds => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::DivideByZero => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Overflow => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::InvalidCast => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::UserThrow => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::IllegalInstruction => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ir::{ops::SsaOp, variable::SsaVarId},
        testing::MockTarget,
    };

    /// A `LoadArg` that survives SSA construction reads an address-taken
    /// argument, so like `LoadLocal` it reads state a store through the address
    /// can change: two of them around such a store are not one value.
    #[test]
    fn a_surviving_argument_load_reads_memory_like_a_local_load() {
        let dest = SsaVarId::from_index(0);
        let arg = SsaOp::<MockTarget>::LoadArg { dest, arg_index: 0 };
        let local = SsaOp::<MockTarget>::LoadLocal {
            dest,
            local_index: 0,
        };
        assert!(!arg.is_pure());
        assert!(arg.effects().reads_memory());
        assert_eq!(arg.effects().kind, local.effects().kind);
    }
}
