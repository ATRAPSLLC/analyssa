//! System operations: floating-point control, cache and TLB maintenance,
//! barriers, hypervisor and hardware-engine calls, interrupt returns,
//! breakpoints, traps, system registers, hints and transactions.

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::ops::{
    effects::{ControlEffect, MemoryEffectLocation, SsaEffectKind, SsaEffects},
    kinds::{CmpKind, machine::MachineStateOp, memory::AtomicOrdering},
    table::OpKindTable,
};

/// Floating-point unit control / state operation ([`SsaOp::FpuControl`]).
///
/// [`SsaOp::FpuControl`]: crate::ir::ops::def::SsaOp::FpuControl
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
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
    /// Initialize / reset the FPU unit (`fninit`/`finit`) - clears the control
    /// word, status word, tag word and resets the stack top.
    Initialize,
    /// Empty the MMX technology state (`emms`/`femms`) - marks every x87 tag
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
    /// Mark an FPU register free and pop the stack (`ffreep`).
    FreeRegisterPop,
    /// Store the FPU environment in its 14-byte layout, the one a 16-bit
    /// operand size selects (`fnstenv` with `0x66`).
    StoreEnvironmentShort,
    /// Load the FPU environment from its 14-byte layout (`fldenv` with
    /// `0x66`).
    LoadEnvironmentShort,
    /// Save full FPU state in its 94-byte layout (`fnsave` with `0x66`).
    SaveShort,
    /// Restore full FPU state from its 94-byte layout (`frstor` with `0x66`).
    RestoreShort,
}

impl FpuControlKind {
    /// Returns the IR name of this FPU control operation, such as `fpu.control.load`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::LoadControlWord => "fpu.control.load",
            Self::StoreControlWord => "fpu.control.store",
            Self::StoreStatusWord => "fpu.status.store",
            Self::LoadEnvironment => "fpu.env.load",
            Self::StoreEnvironment => "fpu.env.store",
            Self::Save => "fpu.save",
            Self::Restore => "fpu.restore",
            Self::SaveExtended => "fpu.save.ext",
            Self::RestoreExtended => "fpu.restore.ext",
            Self::ClearExceptions => "fpu.exceptions.clear",
            Self::DecrementStackTop => "fpu.top.dec",
            Self::IncrementStackTop => "fpu.top.inc",
            Self::FreeRegister => "fpu.free",
            Self::NoOp => "fpu.nop",
            Self::Wait => "fpu.wait",
            Self::Initialize => "fpu.init",
            Self::EmptyMmxState => "fpu.mmx_empty",
            Self::LoadMxcsr => "fpu.vcontrol.load",
            Self::StoreMxcsr => "fpu.vcontrol.store",
            Self::SaveExtendedMasked => "fpu.save.masked",
            Self::SaveExtendedCompact => "fpu.save.compact",
            Self::SaveExtendedOptimized => "fpu.save.modified",
            Self::SaveExtendedSupervisor => "fpu.save.supervisor",
            Self::RestoreExtendedMasked => "fpu.restore.masked",
            Self::RestoreExtendedSupervisor => "fpu.restore.supervisor",
            Self::FreeRegisterPop => "fpu.free.pop",
            Self::StoreEnvironmentShort => "fpu.env.store.short",
            Self::LoadEnvironmentShort => "fpu.env.load.short",
            Self::SaveShort => "fpu.save.short",
            Self::RestoreShort => "fpu.restore.short",
        }
    }

    /// Whether the operation moves a state image through memory: an
    /// environment (`fnstenv`, `fldenv`), the whole x87 state (`fnsave`,
    /// `frstor`) or the extended state (`fxsave`, `xsave` and their restores).
    #[must_use]
    pub const fn moves_image(self) -> bool {
        self.saves_image() || self.restores_image()
    }

    /// Whether the operation writes a state image to memory.
    #[must_use]
    pub const fn saves_image(self) -> bool {
        matches!(
            self,
            Self::StoreEnvironment
                | Self::StoreEnvironmentShort
                | Self::Save
                | Self::SaveShort
                | Self::SaveExtended
                | Self::SaveExtendedMasked
                | Self::SaveExtendedCompact
                | Self::SaveExtendedOptimized
                | Self::SaveExtendedSupervisor
        )
    }

    /// Whether the operation reads a state image from memory.
    #[must_use]
    pub const fn restores_image(self) -> bool {
        matches!(
            self,
            Self::LoadEnvironment
                | Self::LoadEnvironmentShort
                | Self::Restore
                | Self::RestoreShort
                | Self::RestoreExtended
                | Self::RestoreExtendedMasked
                | Self::RestoreExtendedSupervisor
        )
    }

    /// Whether the image the operation moves holds the x87 stack's eight
    /// registers: a whole x87 state or an extended state, not an environment.
    #[must_use]
    pub const fn image_holds_stack(self) -> bool {
        self.moves_image()
            && !matches!(
                self,
                Self::StoreEnvironment
                    | Self::StoreEnvironmentShort
                    | Self::LoadEnvironment
                    | Self::LoadEnvironmentShort
            )
    }

    /// Whether the image is the extended state: `fxsave`'s, or the
    /// component-selected one of `xsave` and `xrstor`, which hold `MXCSR` and
    /// the vector registers beside the x87 state.
    #[must_use]
    pub const fn image_is_extended(self) -> bool {
        matches!(
            self,
            Self::SaveExtended
                | Self::RestoreExtended
                | Self::SaveExtendedMasked
                | Self::SaveExtendedCompact
                | Self::SaveExtendedOptimized
                | Self::SaveExtendedSupervisor
                | Self::RestoreExtendedMasked
                | Self::RestoreExtendedSupervisor
        )
    }

    /// Whether the components the image holds are those an `EDX:EAX` mask
    /// selects (`xsave`, `xrstor` and their forms), which the operation reads.
    #[must_use]
    pub const fn image_is_masked(self) -> bool {
        matches!(
            self,
            Self::SaveExtendedMasked
                | Self::SaveExtendedCompact
                | Self::SaveExtendedOptimized
                | Self::SaveExtendedSupervisor
                | Self::RestoreExtendedMasked
                | Self::RestoreExtendedSupervisor
        )
    }

    /// The bytes of memory the image occupies: an environment 28 (14 in the
    /// 16-bit layout), a whole x87 state 108 (94), `fxsave`'s 512. `None` for
    /// one whose extent the enabled components decide (`xsave`), and for an
    /// operation that moves no image.
    #[must_use]
    pub const fn image_bytes(self) -> Option<u32> {
        match self {
            Self::StoreEnvironment | Self::LoadEnvironment => Some(28),
            Self::StoreEnvironmentShort | Self::LoadEnvironmentShort => Some(14),
            Self::Save | Self::Restore => Some(108),
            Self::SaveShort | Self::RestoreShort => Some(94),
            Self::SaveExtended | Self::RestoreExtended => Some(512),
            _ => None,
        }
    }

    /// Whether the operation reads the status word's condition codes (`C0`-`C3`)
    /// a compare or an examine set: storing the status word or an environment
    /// or state image that holds it.
    #[must_use]
    pub const fn reads_status_word(self) -> bool {
        matches!(
            self,
            Self::StoreStatusWord
                | Self::StoreEnvironment
                | Self::StoreEnvironmentShort
                | Self::Save
                | Self::SaveShort
                | Self::SaveExtended
                | Self::SaveExtendedMasked
                | Self::SaveExtendedCompact
                | Self::SaveExtendedOptimized
                | Self::SaveExtendedSupervisor
        )
    }

    /// Whether the operation replaces the status word's condition codes:
    /// loading an environment or state image that holds it, resetting the unit
    /// (`fninit`, and `fnsave`, which resets it after saving), or clearing the
    /// exceptions, which leaves the codes undefined.
    #[must_use]
    pub const fn writes_status_word(self) -> bool {
        matches!(
            self,
            Self::LoadEnvironment
                | Self::LoadEnvironmentShort
                | Self::Save
                | Self::SaveShort
                | Self::Restore
                | Self::RestoreShort
                | Self::RestoreExtended
                | Self::RestoreExtendedMasked
                | Self::RestoreExtendedSupervisor
                | Self::ClearExceptions
                | Self::Initialize
        )
    }

    /// Whether the operation reads the control word: storing it, or an
    /// environment or state image that holds it.
    #[must_use]
    pub const fn reads_control_word(self) -> bool {
        matches!(
            self,
            Self::StoreControlWord
                | Self::StoreEnvironment
                | Self::StoreEnvironmentShort
                | Self::Save
                | Self::SaveShort
                | Self::SaveExtended
                | Self::SaveExtendedMasked
                | Self::SaveExtendedCompact
                | Self::SaveExtendedOptimized
                | Self::SaveExtendedSupervisor
        )
    }

    /// Whether the operation replaces the control word: loading it or an
    /// environment or state image that holds it, resetting the unit
    /// (`fninit`, and `fnsave`, which resets it after saving, both to
    /// `0x037f`), or storing the environment, which masks every exception
    /// after it stores (`fnstenv`).
    #[must_use]
    pub const fn writes_control_word(self) -> bool {
        matches!(
            self,
            Self::LoadControlWord
                | Self::LoadEnvironment
                | Self::LoadEnvironmentShort
                | Self::StoreEnvironment
                | Self::StoreEnvironmentShort
                | Self::Save
                | Self::SaveShort
                | Self::Restore
                | Self::RestoreShort
                | Self::RestoreExtended
                | Self::RestoreExtendedMasked
                | Self::RestoreExtendedSupervisor
                | Self::Initialize
        )
    }

    /// Whether the operation reads `MXCSR`: storing it, or saving an extended
    /// state image that holds it.
    #[must_use]
    pub const fn reads_mxcsr(self) -> bool {
        matches!(
            self,
            Self::StoreMxcsr
                | Self::SaveExtended
                | Self::SaveExtendedMasked
                | Self::SaveExtendedCompact
                | Self::SaveExtendedOptimized
                | Self::SaveExtendedSupervisor
        )
    }

    /// Whether the operation replaces `MXCSR`: loading it, or restoring an
    /// extended state image that holds it.
    #[must_use]
    pub const fn writes_mxcsr(self) -> bool {
        matches!(
            self,
            Self::LoadMxcsr
                | Self::RestoreExtended
                | Self::RestoreExtendedMasked
                | Self::RestoreExtendedSupervisor
        )
    }

    /// The width in bits of the one state word the operation moves through
    /// an operand - the control and status words' 16, `MXCSR`'s 32 - or `None`
    /// for one that moves no word, or a whole state image.
    #[must_use]
    pub const fn state_word_bits(self) -> Option<u32> {
        match self {
            Self::LoadControlWord | Self::StoreControlWord | Self::StoreStatusWord => Some(16),
            Self::LoadMxcsr | Self::StoreMxcsr => Some(32),
            _ => None,
        }
    }

    /// Whether the operation takes its state word in (`fldcw`, `ldmxcsr`)
    /// rather than giving it out.
    #[must_use]
    pub const fn takes_state_word(self) -> bool {
        matches!(self, Self::LoadControlWord | Self::LoadMxcsr)
    }

    /// Precise effect summary for this operation.
    ///
    /// An image save writes the memory at the address it is given, and an
    /// image restore reads it: the registers the image holds are the
    /// operation's inputs and outputs, so nothing else it does is beyond its
    /// operands. Every other kind is a native operation a host lifts as the
    /// registers it moves - a word move as a move of the state register, a
    /// reset as the values it sets, the stack's bookkeeping as the stack's -
    /// and is no operation of the SSA ([`Self::moves_image`]); as one it would
    /// change state no operand states.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        if self.saves_image() {
            return SsaEffects::new(SsaEffectKind::Write, false);
        }
        if self.restores_image() {
            return SsaEffects::new(SsaEffectKind::Read, false);
        }
        SsaEffects::new(SsaEffectKind::Opaque, false).with_memory(MemoryEffectLocation::None)
    }
}

/// Cache-maintenance operation named by [`SystemOpKind::CacheMaintenance`].
///
/// One variant per operation, so the family reconstructs the instruction's
/// own mnemonic rather than a representative one standing for the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum CacheMaintenanceOp {
    /// Invalidate every cache without writing back (`invd`).
    InvalidateAll,
    /// Write back and invalidate every cache (`wbinvd`).
    WriteBackInvalidateAll,
    /// Write back every cache without invalidating (`wbnoinvd`).
    WriteBackAll,
    /// Write back and invalidate one line - `clflush` and its unordered form,
    /// RISC-V `cbo.flush`, AArch64 `dc civac`.
    FlushLine,
    /// Write back one line, keeping it valid (`clwb`, RISC-V `cbo.clean`,
    /// AArch64 `dc cvac`).
    WriteBackLine,
    /// Move one line to a more distant level (`cldemote`).
    DemoteLine,
    /// Zero one cache line (`clzero`).
    ZeroLine,
    /// Evict one line from a named level (`clevict0`, `clevict1`).
    EvictLine,
    /// Invalidate a line or block without writing it back (`cl1invmb`, RISC-V
    /// `cbo.inval`, AArch64 `dc ivac`), which may discard data.
    InvalidateBlock,
    /// Data-cache maintenance whose operation an operand selects and the lift
    /// names no closer: AArch64 `dc` by set or way, MIPS `cache` tag operations.
    DataCacheOperation,
    /// Instruction-cache maintenance - AArch64 `ic`, MIPS `synci`.
    InstructionCacheOperation,
}

impl CacheMaintenanceOp {
    /// Returns the IR name of this cache-maintenance operation, such as `cache.invalidate.all`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidateAll => "cache.invalidate.all",
            Self::WriteBackInvalidateAll => "cache.flush.all",
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

    /// Returns what the operation does to memory's contents.
    ///
    /// A write-back, a flush, a demotion and an eviction move bytes between
    /// the caches and memory and keep every one as it was: each reads memory,
    /// so a store before it is what it moves on and stays, and writes none, so
    /// a value known before it holds after it. An instruction-cache operation
    /// reads too: the code stores before it are what it makes visible to
    /// fetch. An invalidation, which may discard data not yet written back, a
    /// zeroing and a data-cache operation an operand selects write memory.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::WriteBackInvalidateAll
            | Self::WriteBackAll
            | Self::FlushLine
            | Self::WriteBackLine
            | Self::DemoteLine
            | Self::EvictLine
            | Self::InstructionCacheOperation => SsaEffects::new(SsaEffectKind::Read, false),
            Self::InvalidateAll
            | Self::ZeroLine
            | Self::InvalidateBlock
            | Self::DataCacheOperation => SsaEffects::new(SsaEffectKind::Write, false),
        }
    }

    /// Whether the operation acts on the one line, or block, an address names.
    ///
    /// Such an operation's first input is that address: `clflush [rdi+32]`,
    /// RISC-V `cbo.clean (a0)`, AArch64 `dc cvac, x0`. What it reads or writes
    /// is the line's bytes, at and around the address in a size the
    /// implementation fixes. The whole-cache operations name no address, and
    /// neither do the ones an operand selects (`dc` by set or way, an `ic`).
    #[must_use]
    pub const fn names_a_line(self) -> bool {
        matches!(
            self,
            Self::FlushLine
                | Self::WriteBackLine
                | Self::DemoteLine
                | Self::ZeroLine
                | Self::EvictLine
                | Self::InvalidateBlock
        )
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum TlbMaintenanceOp {
    /// Invalidate the entries for an address - x86 `invlpg`, AArch64 `tlbi`. One operation.
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
    /// Order translation writes against later guest-virtual translations, the
    /// first stage of a guest's (RISC-V `hfence.vvma`).
    SynchronizeGuest,
    /// Order translation writes against later guest-physical translations, the
    /// second stage a hypervisor maps (RISC-V `hfence.gvma`).
    SynchronizeNested,
    /// Invalidate translations without ordering the invalidation against the
    /// accesses around it, which [`Self::OrderBeforeInvalidation`] and
    /// [`Self::OrderAfterInvalidation`] do instead (RISC-V `sinval.vma`).
    InvalidateUnordered,
    /// Invalidate guest-virtual translations without ordering them (RISC-V
    /// `hinval.vvma`).
    InvalidateGuestUnordered,
    /// Invalidate guest-physical translations without ordering them (RISC-V
    /// `hinval.gvma`).
    InvalidateNestedUnordered,
    /// Order earlier stores before later unordered invalidations (RISC-V
    /// `sfence.w.inval`).
    OrderBeforeInvalidation,
    /// Order earlier unordered invalidations before later references to the
    /// translation structures (RISC-V `sfence.inval.ir`).
    OrderAfterInvalidation,
}

impl TlbMaintenanceOp {
    /// Returns the IR name of this TLB-maintenance operation, such as `tlb.invalidate.page`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
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
            Self::SynchronizeGuest => "tlb.sync.guest",
            Self::SynchronizeNested => "tlb.sync.nested",
            Self::InvalidateUnordered => "tlb.invalidate.unordered",
            Self::InvalidateGuestUnordered => "tlb.invalidate.guest.unordered",
            Self::InvalidateNestedUnordered => "tlb.invalidate.nested.unordered",
            Self::OrderBeforeInvalidation => "tlb.order.writes",
            Self::OrderAfterInvalidation => "tlb.order.invalidations",
        }
    }

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
            Self::SynchronizeGuest => "tlb.sync.guest",
            Self::SynchronizeNested => "tlb.sync.nested",
            Self::InvalidateUnordered => "tlb.invalidate.unordered",
            Self::InvalidateGuestUnordered => "tlb.invalidate.guest.unordered",
            Self::InvalidateNestedUnordered => "tlb.invalidate.nested.unordered",
            Self::OrderBeforeInvalidation => "tlb.order.writes",
            Self::OrderAfterInvalidation => "tlb.order.invalidations",
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
            Self::SynchronizeGuest => "system.tlb.sync.guest",
            Self::SynchronizeNested => "system.tlb.sync.nested",
            Self::InvalidateUnordered => "system.tlb.invalidate.unordered",
            Self::InvalidateGuestUnordered => "system.tlb.invalidate.guest.unordered",
            Self::InvalidateNestedUnordered => "system.tlb.invalidate.nested.unordered",
            Self::OrderBeforeInvalidation => "system.tlb.order.writes",
            Self::OrderAfterInvalidation => "system.tlb.order.invalidations",
        }
    }
}

/// Serializing or synchronizing operation named by [`SystemOpKind::Barrier`]:
/// one that orders execution, instruction fetch, speculation or the
/// exclusive monitor rather than memory accesses alone, which
/// [`SsaOp::Fence`](crate::ir::ops::SsaOp::Fence) states.
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum BarrierOp {
    /// Serialize instruction execution (`serialize`).
    Serialize,
    /// Commit outstanding stores and report faults (`mcommit`).
    CommitStores,
    /// Commit stores to persistent memory (`pcommit`).
    CommitPersistent,
    /// Synchronize instruction fetch with earlier stores and context changes
    /// (AArch64 and ARM `isb`, RISC-V `fence.i`).
    InstructionSync,
    /// Stop speculation past the barrier (AArch64 `sb`, `ssbb`, `pssbb`, ARM
    /// `csdb`).
    SpeculationBarrier,
    /// Clear the local exclusive monitor, so a later store-exclusive fails
    /// (AArch64 and ARM `clrex`).
    ClearExclusive,
    /// Synchronize pending errors (AArch64 `esb`).
    ErrorSync,
    /// Clear the branch history speculation reads (ARM `clrbhb`).
    ClearBranchHistory,
}

impl BarrierOp {
    /// Returns the IR name of this serializing operation, such as `barrier.serialize`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Serialize => "barrier.serialize",
            Self::CommitStores => "barrier.commit_stores",
            Self::CommitPersistent => "barrier.commit_persistent",
            Self::InstructionSync => "barrier.instruction_sync",
            Self::SpeculationBarrier => "barrier.speculation",
            Self::ClearExclusive => "barrier.clear_exclusive",
            Self::ErrorSync => "barrier.error_sync",
            Self::ClearBranchHistory => "barrier.clear_branch_history",
        }
    }

    /// Returns the canonical assembler mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Serialize => "serialize",
            Self::CommitStores => "mcommit",
            Self::CommitPersistent => "pcommit",
            Self::InstructionSync => "isb",
            Self::SpeculationBarrier => "sb",
            Self::ClearExclusive => "clrex",
            Self::ErrorSync => "esb",
            Self::ClearBranchHistory => "clrbhb",
        }
    }

    /// Returns the stable display / fingerprint key for this operation.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Serialize => "system.barrier.serialize",
            Self::CommitStores => "system.barrier.mcommit",
            Self::CommitPersistent => "system.barrier.pcommit",
            Self::InstructionSync => "system.barrier.isb",
            Self::SpeculationBarrier => "system.barrier.sb",
            Self::ClearExclusive => "system.barrier.clrex",
            Self::ErrorSync => "system.barrier.esb",
            Self::ClearBranchHistory => "system.barrier.clrbhb",
        }
    }
}

/// Virtualization or secure-enclave operation named by [`SystemOpKind::Hypervisor`].
///
/// One variant per operation, so the family reconstructs the instruction's
/// own mnemonic rather than a representative one standing for the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum HypervisorOp {
    /// Call the hypervisor - x86 `vmcall` and `vmmcall`, ARM `hvc`. One operation.
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
    /// Ask the attached debugger for a host service (semihosting), which it
    /// performs and resumes after: Arm `hlt #0xf000`, T32 `hlt #0x3c`,
    /// M-profile `bkpt #0xab`, A32 `svc #0x123456` and T32 `svc #0xab`, the
    /// RISC-V `ebreak` that `slli zero, zero, 0x1f` and `srai zero, zero, 7`
    /// bracket, and the MIPS UHI `sdbbp 1`. The breakpoint its encoding
    /// shares would restart at itself; the debugger advances past it instead.
    SemihostingCall,
}

impl HypervisorOp {
    /// Returns the IR name of this virtualization operation, such as `hv.call`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::HypervisorCall => "hv.call",
            Self::SecureMonitorCall => "secure_monitor.call",
            Self::GuestEnter => "guest.enter",
            Self::GuestExit => "guest.exit",
            Self::VirtualizationEnter => "virt.on",
            Self::VirtualizationLeave => "virt.off",
            Self::ControlBlockLoad => "guest_control.load",
            Self::ControlBlockStore => "guest_control.store",
            Self::ControlBlockClear => "guest_control.clear",
            Self::ControlBlockRead => "guest_control.read",
            Self::ControlBlockWrite => "guest_control.write",
            Self::VirtualizationFunction => "virt.function",
            Self::GlobalInterruptEnable => "intr.global.enable",
            Self::GlobalInterruptDisable => "intr.global.disable",
            Self::SecureInitialize => "secure.init",
            Self::EnclaveOperation => "enclave.op",
            Self::EnclaveReturn => "enclave.return",
            Self::TrustedModuleCall => "trusted_module.call",
            Self::TrustedModuleReturn => "trusted_module.return",
            Self::TrustedModuleOperation => "trusted_module.op",
            Self::SemihostingCall => "semihosting.call",
        }
    }

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
            Self::SemihostingCall => "semihosting.call",
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
            Self::SemihostingCall => "system.semihosting.call",
        }
    }
}

/// On-chip engine operation named by [`SystemOpKind::HardwareEngine`].
///
/// One variant per operation, so the family reconstructs the instruction's
/// own mnemonic rather than a representative one standing for the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
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
    /// Returns the IR name of this hardware-engine operation, such as `engine.random_store`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RandomStore => "engine.random_store",
            Self::CipherEcb => "engine.cipher.ecb",
            Self::CipherCbc => "engine.cipher.cbc",
            Self::CipherCtr => "engine.cipher.ctr",
            Self::CipherCfb => "engine.cipher.cfb",
            Self::CipherOfb => "engine.cipher.ofb",
            Self::HashSha1 => "engine.hash.sha1",
            Self::HashSha256 => "engine.hash.sha256",
            Self::HashSha512 => "engine.hash.sha512",
            Self::MontgomeryMultiply => "engine.montgomery_mul",
            Self::CounterModeEncrypt => "engine.ctr.encrypt",
            Self::CounterModeHash => "engine.ctr.hash",
        }
    }

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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum InterruptReturnOp {
    /// Return to the interrupted context - x86 `iret`/`iretd`/`iretq`, ARM
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
    /// Return from a resumable non-maskable interrupt, restoring the state the
    /// interrupt saved (RISC-V `mnret`).
    NmiTrapReturn,
    /// Leave debug mode and resume where it was entered (RISC-V `dret`).
    DebugReturn,
}

impl InterruptReturnOp {
    /// Returns the IR name of this interrupt-return operation, such as `exception.return`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ExceptionReturn => "exception.return",
            Self::AuthenticatedExceptionReturn => "exception.return.auth",
            Self::MachineTrapReturn => "trap.return.machine",
            Self::SupervisorTrapReturn => "trap.return.supervisor",
            Self::UserTrapReturn => "trap.return.user",
            Self::NmiTrapReturn => "trap.return.nmi",
            Self::DebugReturn => "debug.return",
        }
    }

    /// Returns the neutral display mnemonic for this return.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::ExceptionReturn => "eret",
            Self::AuthenticatedExceptionReturn => "eret.auth",
            Self::MachineTrapReturn => "eret.machine",
            Self::SupervisorTrapReturn => "eret.supervisor",
            Self::UserTrapReturn => "eret.user",
            Self::NmiTrapReturn => "eret.nmi",
            Self::DebugReturn => "eret.debug",
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
            Self::NmiTrapReturn => "system.iret.nmi",
            Self::DebugReturn => "system.iret.debug",
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum BreakpointOp {
    /// Trap to a debugger, which resumes after the breakpoint - x86 `int3`
    /// and `int1`, whose trap saves the address that follows, and CIL `break`.
    Breakpoint,
    /// Raise an undefined-instruction fault - x86 `ud0`/`ud1`/`ud2`, ARM and
    /// AArch64 `udf`, RISC-V `unimp`.
    UndefinedInstruction,
    /// Halt into an external debugger, which is a distinct exception from a
    /// breakpoint - ARM and AArch64 `hlt`. Its restart address is the `hlt`
    /// itself.
    DebugHalt,
    /// Trap to a debugger at the breakpoint, whose exception saves the
    /// breakpoint's own address, so that resuming executes it again - ARM
    /// `bkpt`, AArch64 `brk`, MIPS `break` and `sdbbp`, RISC-V `ebreak`.
    /// Nothing falls through it, as nothing falls through an
    /// [`Self::UndefinedInstruction`].
    BreakpointInPlace,
}

impl BreakpointOp {
    /// Returns the IR name of this breakpoint operation, such as `breakpoint`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Breakpoint => "breakpoint",
            Self::UndefinedInstruction => "trap.undefined",
            Self::DebugHalt => "debug.halt",
            Self::BreakpointInPlace => "breakpoint.in.place",
        }
    }

    /// Returns the neutral display mnemonic for this trap.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::Breakpoint => "breakpoint",
            Self::UndefinedInstruction => "undefined",
            Self::DebugHalt => "debughalt",
            Self::BreakpointInPlace => "breakpoint.inplace",
        }
    }

    /// Returns the stable display / fingerprint key for this trap.
    #[must_use]
    pub const fn kind_str(self) -> &'static str {
        match self {
            Self::Breakpoint => "break.breakpoint",
            Self::UndefinedInstruction => "break.undefined",
            Self::DebugHalt => "break.debughalt",
            Self::BreakpointInPlace => "break.inplace",
        }
    }
}

/// Software trap named by [`SystemOpKind::Trap`].
///
/// One variant per operation. These share a lowering, not an identity: a
/// renderer reconstructs the instruction from the payload, and two of them
/// never collapse onto one another in a similarity class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum TrapOp {
    /// Trap to a handler the operand names - x86 `int`, ARM `trap`.
    SoftwareInterrupt,
    /// Trap when the overflow flag is set (`into`).
    OverflowTrap,
    /// Trap when an index is outside its bounds (`bound`).
    BoundsCheck,
    /// Trap with a bound-range fault when a value lies below a lower bound:
    /// MPX `bndcl`. The inputs are the bound and then the value, in the
    /// instruction's operand order, so it fires when the first exceeds the
    /// second, unsigned.
    BoundsLower,
    /// Trap with a bound-range fault when a value lies above an upper bound:
    /// MPX `bndcu` and `bndcn`, which differ only in whether the bound is
    /// complemented first. The inputs are the bound and then the value, so it
    /// fires when the first is below the second, unsigned.
    BoundsUpper,
    /// Trap when the two compared values are equal. Whether the second is a register or an
    /// immediate is an operand property.
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
    /// Trap on a signed greater-than comparison (A64 `cfltgt`).
    TrapIfGreater,
    /// Trap on an unsigned greater-than comparison (A64 `cflthi`).
    TrapIfGreaterUnsigned,
    /// Trap on a signed less-or-equal comparison.
    TrapIfLessOrEqual,
    /// Trap on an unsigned less-or-equal comparison.
    TrapIfLessOrEqualUnsigned,
}

impl TrapOp {
    /// Returns the IR name of this trap, such as `trap`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SoftwareInterrupt => "trap",
            Self::OverflowTrap => "trap.overflow",
            Self::BoundsCheck => "trap.bounds",
            Self::BoundsLower => "trap.bounds.lower",
            Self::BoundsUpper => "trap.bounds.upper",
            Self::TrapIfEqual => "trap.cmp.eq",
            Self::TrapIfNotEqual => "trap.cmp.ne",
            Self::TrapIfLess => "trap.cmp.slt",
            Self::TrapIfLessUnsigned => "trap.cmp.ult",
            Self::TrapIfGreaterOrEqual => "trap.cmp.sge",
            Self::TrapIfGreaterOrEqualUnsigned => "trap.cmp.uge",
            Self::TrapIfGreater => "trap.cmp.sgt",
            Self::TrapIfGreaterUnsigned => "trap.cmp.ugt",
            Self::TrapIfLessOrEqual => "trap.cmp.sle",
            Self::TrapIfLessOrEqualUnsigned => "trap.cmp.ule",
        }
    }

    /// Returns the neutral display mnemonic for this operation.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::SoftwareInterrupt => "trap",
            Self::OverflowTrap => "trap.overflow",
            Self::BoundsCheck => "trap.bounds",
            Self::BoundsLower => "trap.bounds.lower",
            Self::BoundsUpper => "trap.bounds.upper",
            Self::TrapIfEqual => "trap.eq",
            Self::TrapIfNotEqual => "trap.ne",
            Self::TrapIfLess => "trap.lt",
            Self::TrapIfLessUnsigned => "trap.ltu",
            Self::TrapIfGreaterOrEqual => "trap.ge",
            Self::TrapIfGreaterOrEqualUnsigned => "trap.geu",
            Self::TrapIfGreater => "trap.gt",
            Self::TrapIfGreaterUnsigned => "trap.gtu",
            Self::TrapIfLessOrEqual => "trap.le",
            Self::TrapIfLessOrEqualUnsigned => "trap.leu",
        }
    }

    /// Returns the comparison that decides whether this trap fires, and whether
    /// it is unsigned, or `None` when the trap is unconditional.
    ///
    /// The predicate is the whole content of a conditional trap: `teq` transfers
    /// only when its two inputs are equal. Reading it off the variant *name* is
    /// not something a pass can do, so it is exposed as data - the two compared
    /// values are the operation's inputs, and this says what is done with them.
    #[must_use]
    pub const fn condition(self) -> Option<(CmpKind, bool)> {
        match self {
            Self::TrapIfEqual => Some((CmpKind::Eq, false)),
            Self::TrapIfNotEqual => Some((CmpKind::Ne, false)),
            Self::TrapIfLess => Some((CmpKind::Lt, false)),
            Self::TrapIfLessUnsigned | Self::BoundsUpper => Some((CmpKind::Lt, true)),
            Self::TrapIfGreaterOrEqual => Some((CmpKind::Ge, false)),
            Self::TrapIfGreaterOrEqualUnsigned => Some((CmpKind::Ge, true)),
            Self::TrapIfGreater => Some((CmpKind::Gt, false)),
            Self::TrapIfGreaterUnsigned | Self::BoundsLower => Some((CmpKind::Gt, true)),
            Self::TrapIfLessOrEqual => Some((CmpKind::Le, false)),
            Self::TrapIfLessOrEqualUnsigned => Some((CmpKind::Le, true)),
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
            Self::BoundsLower => "system.trap.bounds.lower",
            Self::BoundsUpper => "system.trap.bounds.upper",
            Self::TrapIfEqual => "system.trap.eq",
            Self::TrapIfNotEqual => "system.trap.ne",
            Self::TrapIfLess => "system.trap.lt",
            Self::TrapIfLessUnsigned => "system.trap.ltu",
            Self::TrapIfGreaterOrEqual => "system.trap.ge",
            Self::TrapIfGreaterOrEqualUnsigned => "system.trap.geu",
            Self::TrapIfGreater => "system.trap.gt",
            Self::TrapIfGreaterUnsigned => "system.trap.gtu",
            Self::TrapIfLessOrEqual => "system.trap.le",
            Self::TrapIfLessOrEqualUnsigned => "system.trap.leu",
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
/// rides [`NativeKindedData::metadata`](crate::ir::ops::kinds::NativeKindedData::metadata), which is where target identity
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum SysRegOp {
    /// Move a system, coprocessor or control register into a general register - ARM `mrs`/`mrc`,
    /// MIPS `mfc0`/`mfc2`, and their doubleword forms.
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
    /// Move a general register into a system, coprocessor or control register - ARM `msr`/`mcr`,
    /// MIPS `mtc0`/`mtc2`.
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
    /// Atomically read a system register and write a new value into it (RISC-V `csrrw`). The read
    /// and the write are one operation, which a plain read or write is not.
    ExchangeSystemRegister,
    /// Atomically read a system register and set the bits a mask names (RISC-V `csrrs`).
    SetSystemRegisterBits,
    /// Atomically read a system register and clear the bits a mask names (RISC-V `csrrc`).
    ClearSystemRegisterBits,
    /// Write a multiplier accumulator (Octeon `mtm`). The register the opcode selects rides the
    /// operand list.
    WriteMultiplier,
    /// Write a multiplier product register (Octeon `mtp`).
    WriteMultiplierProduct,
}

impl SysRegOp {
    /// Returns the IR name of this system-register operation, such as `sysreg.read`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ReadSystemRegister => "sysreg.read",
            Self::ReadModelSpecific => "msr.read",
            Self::WriteModelSpecific => "msr.write",
            Self::ReadExtendedControl => "xcr.read",
            Self::WriteExtendedControl => "xcr.write",
            Self::ReadControlRegister => "ctlreg.read",
            Self::WriteControlRegister => "ctlreg.write",
            Self::ReadDebugRegister => "debugreg.read",
            Self::WriteDebugRegister => "debugreg.write",
            Self::WriteSystemRegister => "sysreg.write",
            Self::ReadSystemRegisterPair => "sysreg.read.pair",
            Self::WriteSystemRegisterPair => "sysreg.write.pair",
            Self::CoprocessorOperation => "coproc.op",
            Self::CoprocessorLoad => "coproc.load",
            Self::CoprocessorStore => "coproc.store",
            Self::ExchangeSystemRegister => "sysreg.exchange",
            Self::SetSystemRegisterBits => "sysreg.set_bits",
            Self::ClearSystemRegisterBits => "sysreg.clear_bits",
            Self::WriteMultiplier => "mulunit.write",
            Self::WriteMultiplierProduct => "mulunit.write_product",
        }
    }

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
            Self::WriteMultiplier => "system.mult.write",
            Self::WriteMultiplierProduct => "system.mult.write.product",
        }
    }

    /// Returns `true` when the operation's first explicit operand is the
    /// destination it defines.
    ///
    /// A lift-shape question - which operand slot the lifter must turn into a
    /// definition - not an effect question. [`Self::effects`] is what says what
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
        )
    }

    /// Whether this access moves memory: a coprocessor register loaded from
    /// or stored to the address its operand names (ARM `ldc`, `stc`). Every
    /// other access moves a register the machine holds, and reads no memory;
    /// the [`Self::effects`] of one that writes the register still say it
    /// writes, since a register can decide what memory holds at an address
    /// (a translation base, an MPU region).
    #[must_use]
    pub const fn moves_memory(self) -> bool {
        matches!(self, Self::CoprocessorLoad | Self::CoprocessorStore)
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
            // Moves a special register into a general one: the value the
            // machine holds where control reaches it, read from no memory.
            Self::ReadSystemRegister
            | Self::ReadSystemRegisterPair
            | Self::ReadModelSpecific
            | Self::ReadExtendedControl
            | Self::ReadControlRegister
            | Self::ReadDebugRegister => SsaEffects::new(SsaEffectKind::RuntimeDefined, false),
            // Loads a coprocessor register from memory.
            Self::CoprocessorLoad => SsaEffects::new(SsaEffectKind::Read, false),
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
            | Self::WriteMultiplier
            | Self::WriteMultiplierProduct => SsaEffects::new(SsaEffectKind::Write, false),
        }
    }
}

/// The hint named by [`SystemOpKind::Hint`].
///
/// The hint a target's `InstructionFamily::Hint` names - plain prose rather
/// than a link, because `InstructionFamily` belongs to the layer above this
/// crate, which cannot be named from inside it.
///
/// A hint has no architectural data effect, which is exactly why it needs an
/// identity: nothing downstream can recover `endbr64` from `pause` once both
/// are "the instruction that does nothing". [`SystemOpKind::Hint`] is the
/// carrier that puts that identity into an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
pub enum HintOp {
    /// Do nothing.
    NoOperation,
    /// Hint that this thread is spinning and may yield the pipeline - x86 `pause`, ARM `yield`. One
    /// operation.
    SpinWait,
    /// Mark a valid indirect-branch target - x86 CET `endbr32`/`endbr64`, AArch64 `bti`. One
    /// operation, and the reason a forward-edge integrity scheme is visible at all.
    IndirectBranchLandingPad,
    /// Mark an indirect-branch target that accepts only a branch naming its label, the
    /// operand (RISC-V `lpad`, checked against the label `t2` holds; label 0 accepts any).
    LabelledLandingPad,
    /// Wait until an event is signalled.
    WaitForEvent,
    /// Wait for an event or the deadline an operand states (AArch64 `wfet`).
    WaitForEventTimed,
    /// Wait for an event or a bound the implementation chooses, which no
    /// operand states (RISC-V `wrs.sto`, a short wait on a reservation).
    WaitForEventBounded,
    /// Signal an event to every processor.
    SendEvent,
    /// Signal an event to this processor only.
    SendEventLocal,
    /// Prefetch data into the cache. The level a hint requests is advisory and rides the operand
    /// list.
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
    /// Hint that the next memory access has no temporal locality at the
    /// levels of the memory hierarchy an operand names (RISC-V `ntl.p1`,
    /// `ntl.pall`, `ntl.s1`, `ntl.all`).
    NonTemporalLocality,
}

impl HintOp {
    /// Returns the IR name of this hint, such as `nop`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NoOperation => "nop",
            Self::SpinWait => "hint.spin",
            Self::IndirectBranchLandingPad => "landing_pad",
            Self::LabelledLandingPad => "landing_pad.labelled",
            Self::WaitForEvent => "wait.event",
            Self::WaitForEventTimed => "wait.event.timed",
            Self::WaitForEventBounded => "wait.event.bounded",
            Self::SendEvent => "event.send",
            Self::SendEventLocal => "event.send.local",
            Self::Prefetch => "prefetch",
            Self::PrefetchForWrite => "prefetch.write",
            Self::PrefetchInstruction => "prefetch.code",
            Self::PrefetchNonTemporal => "prefetch.nontemporal",
            Self::PrefetchVector => "prefetch.vec",
            Self::DataGatheringHint => "hint.no_gather",
            Self::ProfilingSynchronizationBarrier => "barrier.profiling",
            Self::ExecutionHazardBarrier => "barrier.hazard",
            Self::DebugHint => "hint.debug",
            Self::ConditionalExecutionBlock => "cond_block",
            Self::GenericHint => "hint",
            Self::NonTemporalLocality => "hint.nontemporal",
        }
    }

    /// Returns the neutral display mnemonic for this hint.
    #[must_use]
    pub const fn mnemonic(self) -> &'static str {
        match self {
            Self::NoOperation => "nop",
            Self::SpinWait => "spinwait",
            Self::IndirectBranchLandingPad => "landingpad",
            Self::LabelledLandingPad => "landingpad.labelled",
            Self::WaitForEvent => "wfe",
            Self::WaitForEventTimed => "wfe.timed",
            Self::WaitForEventBounded => "wfe.bounded",
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
            Self::NonTemporalLocality => "ntl",
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
            Self::LabelledLandingPad => "system.hint.landingpad.labelled",
            Self::WaitForEvent => "system.hint.wfe",
            Self::WaitForEventTimed => "system.hint.wfe.timed",
            Self::WaitForEventBounded => "system.hint.wfe.bounded",
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
            Self::NonTemporalLocality => "system.hint.ntl",
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
    /// `MemoryLocator::classify` turns an `Opaque` op with an
    /// unknown location into a `MemoryOp::Barrier`, which would make every
    /// lifted `nop` - instruction padding, an alignment run between two basic
    /// blocks - block store forwarding across it.
    ///
    /// [`MemoryEffectLocation::None`]: crate::ir::ops::MemoryEffectLocation::None
    /// [`MemoryEffectLocation::Unknown`]: crate::ir::ops::MemoryEffectLocation::Unknown
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
            | Self::LabelledLandingPad
            | Self::WaitForEvent
            | Self::WaitForEventTimed
            | Self::WaitForEventBounded
            | Self::SendEvent
            | Self::SendEventLocal
            | Self::DataGatheringHint
            | Self::ProfilingSynchronizationBarrier
            | Self::ExecutionHazardBarrier
            | Self::DebugHint
            | Self::ConditionalExecutionBlock
            | Self::GenericHint
            | Self::NonTemporalLocality => SsaEffects::new(SsaEffectKind::Opaque, false)
                .with_memory(MemoryEffectLocation::None),
        }
    }
}

/// Structured identity of a typed native **system / privileged** operation
/// ([`SsaOp::SystemOp`]): system, control-register, syscall, trap and
/// cache/TLB operations. The kind drives a precise effect summary and a
/// distinct similarity class; there is no catch-all.
///
/// [`SsaOp::SystemOp`]: crate::ir::ops::def::SsaOp::SystemOp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    /// System-call entry. Every architecture has one - x86 `syscall`, ARM
    /// `svc`, RISC-V `ecall` - and they are the same operation, so they are the
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
    /// A named machine-state operation - descriptor tables, segment bases, port
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
    /// An architectural hint, named by its [`HintOp`] - `nop`, `pause`,
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
/// every operation whose meaning is a property of the transaction - beginning
/// one, ending one, asking whether one is running, changing what it tracks -
/// has exactly one identity and therefore exactly one effect summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
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
    /// Returns the IR name of this transactional-memory operation, such as `txn.start`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Start => "txn.start",
            Self::Commit => "txn.commit",
            Self::Cancel => "txn.abort",
            Self::Test => "txn.test",
            Self::ResumeLoadTracking => "txn.tracking.resume",
            Self::SuspendLoadTracking => "txn.tracking.suspend",
        }
    }

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
    /// it is a `Read` - `MemoryOp::Load` over an unknown location is still
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
    /// Precise effect summary for this system op - derived from the kind, never
    /// an opaque echo. A query of the machine (`cpuid`, the timestamp and
    /// performance counters, a system register read) reports
    /// [`SsaEffectKind::RuntimeDefined`]: the value the processor supplies where
    /// control reaches it, reading no memory, so a store before one is not kept
    /// alive by it, and the query is neither moved nor shared. An operation
    /// that samples memory reports `Read`; writes (incl. cache / TLB /
    /// privileged state mutation) report `Write`; control-transferring ops
    /// (syscalls, traps, hypervisor calls, interrupt returns) report `Call` with
    /// the matching control effect.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            Self::CpuId | Self::Timestamp { .. } | Self::ReadPerfCounter => {
                SsaEffects::new(SsaEffectKind::RuntimeDefined, false)
            }
            Self::ControlRegister(op) => op.effects(),
            Self::MachineState(op) => op.effects(),
            Self::Transaction(op) => op.effects(),
            Self::Hint(op) => op.effects(),
            Self::CacheMaintenance(op) => op.effects(),
            Self::TlbMaintenance(_) | Self::HardwareEngine(_) => {
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
            // context - a `Call`-class control transfer, exactly like
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
    const fn family(self) -> SystemOpFamily {
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
    /// [`OpKindTable::all`] iterators. Data fields - the ones
    /// [`Self::kind_str`] does not read - are fixed at a canonical placeholder
    /// (`Trap { vector: None }`), which is the identity/data split made
    /// executable: two values differing only in a data field are one identity.
    ///
    /// [`OpKindTable::all`]: crate::ir::ops::OpKindTable::all
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
            // all - which the domain-coverage test then reports as the missing
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
enum SystemOpFamily {
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
    const ALL: [Self; 16] = [
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// Every family is expanded by `identities()`, so the domain the injectivity
    /// and effect checks quantify over really is the whole domain.
    ///
    /// The last link of the chain `family()` starts: a new variant fails to compile
    /// against the exhaustive `family` match, a new family fails to compile against
    /// the pinned `ALL` array, and a family nothing expands fails here.
    #[test]
    fn identity_domains_cover_every_family() {
        let system: BTreeSet<_> = SystemOpKind::identities()
            .into_iter()
            .map(SystemOpKind::family)
            .collect();
        assert_eq!(
            system.len(),
            SystemOpFamily::ALL.len(),
            "SystemOpKind::identities() expands {} of {} families",
            system.len(),
            SystemOpFamily::ALL.len(),
        );
    }
}
