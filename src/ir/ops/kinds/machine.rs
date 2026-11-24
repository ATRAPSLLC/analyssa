//! Operations on machine state no other kind states ([`MachineStateOp`]).

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::ops::effects::{ControlEffect, MemoryEffectLocation, SsaEffectKind, SsaEffects};

/// The machine-state operation named by [`SystemOpKind::MachineState`](crate::ir::ops::kinds::SystemOpKind::MachineState).
///
/// One variant per operation, so a renderer can reconstruct the native mnemonic
/// from the kind alone and two different instructions never share a similarity
/// class. The names describe the role rather than any one assembler's spelling;
/// the spelling is [`Self::mnemonic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u16)]
#[derive(IntoPrimitive, TryFromPrimitive)]
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
    /// Load the machine status word - `CR0`'s low half (`lmsw`).
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
    /// Read 32 bits of the shadow stack pointer: the low half of a 64-bit one (`rdsspd`), all of
    /// a 32-bit one (RV32 `ssrdp`).
    ShadowStackReadDword,
    /// Read the shadow stack pointer (`rdsspq`, RV64 `ssrdp`).
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
    /// Push a register onto the shadow stack (RISC-V `sspush`).
    ShadowStackPush,
    /// Pop the shadow stack and compare the value with a register, raising a
    /// software-check exception when they differ (RISC-V `sspopchk`).
    ShadowStackPopCheck,
    /// Swap a register with a doubleword of shadow-stack memory atomically,
    /// returning the old one (RISC-V `ssamoswap.w`).
    ShadowStackSwapDword,
    /// Swap a register with a quadword of shadow-stack memory atomically,
    /// returning the old one (RISC-V `ssamoswap.d`).
    ShadowStackSwapQword,
    // Bound tables.
    /// Load a bound register from the bound-table entry of a pointer (MPX
    /// `bndldx`). The inputs are the pointer's address, which selects the
    /// entry through the bound directory `BNDCFGx` names, and the pointer
    /// value the entry must hold; an entry holding another pointer loads the
    /// bounds that admit every address. An invalid directory entry raises a
    /// bound-range fault.
    BoundTableLoad,
    /// Store a bound register and the pointer value it belongs to into the
    /// bound-table entry of the pointer's address (MPX `bndstx`). The entry is
    /// found through the bound directory as for [`Self::BoundTableLoad`].
    BoundTableStore,
    // User interrupts.
    /// Clear the user-interrupt flag (`clui`).
    UserInterruptDisable,
    /// Set the user-interrupt flag (`stui`).
    UserInterruptEnable,
    /// Report the user-interrupt flag in `CF` (`testui`).
    UserInterruptTest,
    /// Return from a user-interrupt handler (`uiret`) - a control transfer.
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
    // Direct and enqueue stores - memory writes that bypass the cache hierarchy.
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
    /// Resume from system-management mode (`rsm`) - a control transfer.
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
    /// Change processor state - interrupt masks and mode (`cps`).
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
    /// Debug restore of the saved processor state (`drps`) - a control transfer.
    DebugRestoreState,
    /// Enter SME streaming or `ZA` state (`smstart`).
    StreamingModeStart,
    /// Leave SME streaming or `ZA` state (`smstop`).
    StreamingModeStop,
    /// Generic system instruction with no result (`sys`).
    SystemInstruction,
    /// Generic system instruction returning a result (`sysl`), which may read
    /// and write memory (a guarded control stack pop, a stack switch's second
    /// half).
    SystemInstructionRead,
    // Interrupt masking: MIPS `di`/`ei`, which optionally return the prior
    // status register, and the machine half of x86 `cli`/`sti`, whose flag
    // half is the interrupt flag a lift inserts beside it.
    /// Disable interrupts, optionally returning the prior status (`di`,
    /// `cli`).
    InterruptDisable,
    /// Enable interrupts, optionally returning the prior status (`ei`,
    /// `sti`).
    InterruptEnable,
    /// Stall until an interrupt is pending - ARM and AArch64 `wfi`, RISC-V
    /// `wfi`, MIPS `wait`. One operation.
    WaitForInterrupt,
    /// Stall until an interrupt is pending or a deadline passes (AArch64
    /// `wfit`).
    WaitForInterruptTimed,
    /// Clear every control-transfer record (RISC-V `sctrclr`).
    ControlTransferRecordsClear,
    /// Commits a selected vector configuration and active element count, and
    /// resets the restart position to zero. Inputs are the selected native
    /// configuration value and the active count; validation and length choice
    /// precede this atomic machine-state write. A machine that traps on an
    /// unsupported configuration may raise an exception before committing it.
    WriteVectorConfiguration,
    /// Commits active vector length and restart position without revalidating
    /// or changing the selected element configuration. Inputs are length and
    /// restart position. This internal state write cannot raise an exception.
    WriteVectorProgress,
    /// Prevents the control-flow predictions an execution context taught from steering another (`cfp rctx`).
    ControlFlowPredictionRestrict,
    /// Prevents the cache prefetch predictions an execution context taught from steering another (`cpp rctx`).
    CachePrefetchPredictionRestrict,
    /// Prevents the data value predictions an execution context taught from steering another (`dvp rctx`).
    DataValuePredictionRestrict,
    /// Clears the other speculative predictions an execution context taught (`cosp rctx`).
    OtherSpeculationClear,
    /// Changes the physical address space a granule of memory belongs to (`apas`).
    PhysicalAddressSpaceChange,
    /// A GICv5 CPU interface operation with no result (`gic`).
    InterruptControllerOperation,
    /// A GICv5 CPU interface operation returning a result into its register (`gicr`), acknowledging what it reads.
    InterruptControllerRead,
    /// A GICv5 synchronization barrier, completing the interrupt controller effects before it (`gsb`).
    InterruptControllerBarrier,
    /// Emits the trace instrumentation packet its register holds (`trcit`).
    TraceInstrumentation,
}

impl MachineStateOp {
    /// Returns the IR name of this machine-state operation, such as `gdt.load`.
    ///
    /// The name is the operation's words in the IR naming grammar: lowercase words joined
    /// by `.`, the same on every architecture, and different for every operation. It
    /// carries no operand width.
    /// Distinct from [`Self::mnemonic`], the SSA display spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::LoadGlobalDescriptorTable => "gdt.load",
            Self::StoreGlobalDescriptorTable => "gdt.store",
            Self::LoadInterruptDescriptorTable => "idt.load",
            Self::StoreInterruptDescriptorTable => "idt.store",
            Self::LoadLocalDescriptorTable => "ldt.load",
            Self::StoreLocalDescriptorTable => "ldt.store",
            Self::LoadTaskRegister => "task_register.load",
            Self::StoreTaskRegister => "task_register.store",
            Self::LoadMachineStatusWord => "status_word.load",
            Self::StoreMachineStatusWord => "status_word.store",
            Self::ClearTaskSwitched => "task_switched.clear",
            Self::ReadAccessRights => "segment.access_rights",
            Self::ReadSegmentLimit => "segment.limit",
            Self::VerifySegmentReadable => "segment.verify_read",
            Self::VerifySegmentWritable => "segment.verify_write",
            Self::AdjustRequestedPrivilege => "selector.adjust_rpl",
            Self::LoadFarPointerDs => "farptr.load.ds",
            Self::LoadFarPointerEs => "farptr.load.es",
            Self::LoadFarPointerFs => "farptr.load.fs",
            Self::LoadFarPointerGs => "farptr.load.gs",
            Self::LoadFarPointerSs => "farptr.load.ss",
            Self::ReadFsBase => "segbase.read.fs",
            Self::WriteFsBase => "segbase.write.fs",
            Self::ReadGsBase => "segbase.read.gs",
            Self::WriteGsBase => "segbase.write.gs",
            Self::SwapGsBase => "segbase.swap.gs",
            Self::LoadKernelGsBase => "segbase.load_kernel.gs",
            Self::PortIn => "port.in",
            Self::PortOut => "port.out",
            Self::PortInStringByte => "port.in.string.8",
            Self::PortInStringWord => "port.in.string.16",
            Self::PortInStringDword => "port.in.string.32",
            Self::PortOutStringByte => "port.out.string.8",
            Self::PortOutStringWord => "port.out.string.16",
            Self::PortOutStringDword => "port.out.string.32",
            Self::ShadowStackIncrementDword => "shadow_stack.advance.32",
            Self::ShadowStackIncrementQword => "shadow_stack.advance.64",
            Self::ShadowStackReadDword => "shadow_stack.read.32",
            Self::ShadowStackReadQword => "shadow_stack.read.64",
            Self::ShadowStackWriteDword => "shadow_stack.write.32",
            Self::ShadowStackWriteQword => "shadow_stack.write.64",
            Self::ShadowStackWriteUserDword => "shadow_stack.write_user.32",
            Self::ShadowStackWriteUserQword => "shadow_stack.write_user.64",
            Self::ShadowStackMarkBusy => "shadow_stack.mark_busy",
            Self::ShadowStackClearBusy => "shadow_stack.clear_busy",
            Self::ShadowStackRestore => "shadow_stack.restore",
            Self::ShadowStackSavePrevious => "shadow_stack.save_previous",
            Self::ShadowStackPush => "shadow_stack.push",
            Self::ShadowStackPopCheck => "shadow_stack.pop_check",
            Self::ShadowStackSwapDword => "shadow_stack.swap.32",
            Self::ShadowStackSwapQword => "shadow_stack.swap.64",
            Self::BoundTableLoad => "bounds.table.load",
            Self::BoundTableStore => "bounds.table.store",
            Self::UserInterruptDisable => "user_intr.disable",
            Self::UserInterruptEnable => "user_intr.enable",
            Self::UserInterruptTest => "user_intr.test",
            Self::UserInterruptReturn => "user_intr.return",
            Self::UserInterruptSend => "user_intr.send",
            Self::Halt => "halt",
            Self::MonitorAddress => "monitor.arm",
            Self::MonitorAddressExtended => "monitor.arm.timed",
            Self::MonitorAddressUser => "monitor.arm.user",
            Self::MonitorWait => "monitor.wait",
            Self::MonitorWaitTimed => "monitor.wait.timed",
            Self::MonitorWaitUser => "monitor.wait.user",
            Self::TimedPause => "wait.timed",
            Self::ReadProtectionKeyRights => "pkey_rights.read",
            Self::WriteProtectionKeyRights => "pkey_rights.write",
            Self::ReadProcessorId => "cpu.id.read",
            Self::ReadProcessorPower => "cpu.power.read",
            Self::ReadMsrList => "msr.read.list",
            Self::WriteMsrList => "msr.write.list",
            Self::WriteMsrNonSerializing => "msr.write.nonserializing",
            Self::KeyLockerLoadInternalKey => "keylocker.load_key",
            Self::KeyLockerEncodeKey128 => "keylocker.wrap.128",
            Self::KeyLockerEncodeKey256 => "keylocker.wrap.256",
            Self::PlatformConfigure => "platform.configure",
            Self::PlatformBindKey => "platform.bind_key",
            Self::PlatformSmashPage => "platform.split_page",
            Self::PlatformValidatePage => "platform.validate_page",
            Self::ReverseMapAdjust => "rmp.adjust",
            Self::ReverseMapQuery => "rmp.query",
            Self::ReverseMapUpdate => "rmp.update",
            Self::HistoryReset => "predictor.reset",
            Self::DirectStoreWord => "store.direct",
            Self::DirectStoreBlock => "store.direct.block",
            Self::EnqueueCommand => "device.enqueue",
            Self::EnqueueCommandSupervisor => "device.enqueue.supervisor",
            Self::TraceWrite => "trace.write",
            Self::ResumeFromSystemManagement => "smm.resume",
            Self::ProfilingLoadControlBlock => "lwprof.control.load",
            Self::ProfilingStoreControlBlock => "lwprof.control.store",
            Self::ProfilingInsertValue => "lwprof.insert.value",
            Self::ProfilingInsertRecord => "lwprof.insert.record",
            Self::VendorAlternateInstruction => "vendor.alternate_instruction",
            Self::VendorBranchBuffer0Reset => "vendor.branch_buffer0_reset",
            Self::VendorBranchBuffer1Reset => "vendor.branch_buffer1_reset",
            Self::VendorCpuRead => "vendor.cpu_read",
            Self::VendorCpuWrite => "vendor.cpu_write",
            Self::VendorDebugInterrupt => "vendor.debug_interrupt",
            Self::VendorSoftwareSmi => "vendor.software_smi",
            Self::VendorRoundToNearestInteger => "vendor.rint",
            Self::VendorInsertBitString => "vendor.insert_bit_string",
            Self::VendorExtractBitString => "vendor.extract_bit_string",
            Self::VendorLoadAllRegisters => "vendor.load_all_regs",
            Self::VendorStoreAllRegisters => "vendor.store_all_regs",
            Self::VendorReadModelRegister => "vendor.read_model_reg",
            Self::VendorReadShadowRegister => "vendor.read_shadow_reg",
            Self::VendorWriteShadowRegister => "vendor.write_shadow_reg",
            Self::VendorReadDebugRegister => "vendor.read_debug_reg",
            Self::VendorWriteDebugRegister => "vendor.write_debug_reg",
            Self::VendorRestoreDataSegment => "vendor.restore_data_segment",
            Self::VendorSaveDataSegment => "vendor.save_data_segment",
            Self::VendorRestoreLocalDescriptor => "vendor.restore_ldt",
            Self::VendorSaveLocalDescriptor => "vendor.save_ldt",
            Self::VendorRestoreTaskState => "vendor.restore_tss",
            Self::VendorSaveTaskState => "vendor.save_tss",
            Self::VendorSetProtectedFlat => "vendor.set_protected_flat",
            Self::VendorUnprivilegedMove => "vendor.unprivileged_move",
            Self::VendorUndocumented => "vendor.undocumented",
            Self::ProcessorStateChange => "cpu.state.change",
            Self::SetEndianness => "cpu.endian.set",
            Self::SetPrivilegedAccessNever => "cpu.pan.set",
            Self::SecureGateway => "secure.gateway",
            Self::StoreReturnStateDecrementAfter => "return_state.store.dec_after",
            Self::StoreReturnStateDecrementBefore => "return_state.store.dec_before",
            Self::StoreReturnStateIncrementAfter => "return_state.store.inc_after",
            Self::StoreReturnStateIncrementBefore => "return_state.store.inc_before",
            Self::TestTarget => "secure.test_target",
            Self::TestTargetAlternate => "secure.test_target.alt",
            Self::TestTargetAlternateUnprivileged => "secure.test_target.alt.unpriv",
            Self::TestTargetUnprivileged => "secure.test_target.unpriv",
            Self::AddressTranslate => "address.translate",
            Self::BranchRecordBuffer => "branch_record.op",
            Self::DebugChangeStateEl1 => "debug.change_level.1",
            Self::DebugChangeStateEl2 => "debug.change_level.2",
            Self::DebugChangeStateEl3 => "debug.change_level.3",
            Self::DebugRestoreState => "debug.restore",
            Self::StreamingModeStart => "streaming.start",
            Self::StreamingModeStop => "streaming.stop",
            Self::SystemInstruction => "sys.op",
            Self::SystemInstructionRead => "sys.op.read",
            Self::InterruptDisable => "intr.disable",
            Self::InterruptEnable => "intr.enable",
            Self::WaitForInterrupt => "wait.interrupt",
            Self::WaitForInterruptTimed => "wait.interrupt.timed",
            Self::ControlTransferRecordsClear => "ctr.clear",
            Self::WriteVectorConfiguration => "vector.configuration.write",
            Self::WriteVectorProgress => "vector.progress.write",
            Self::ControlFlowPredictionRestrict => "prediction.restrict.flow",
            Self::CachePrefetchPredictionRestrict => "prediction.restrict.prefetch",
            Self::DataValuePredictionRestrict => "prediction.restrict.value",
            Self::OtherSpeculationClear => "prediction.clear.other",
            Self::PhysicalAddressSpaceChange => "address_space.change",
            Self::InterruptControllerOperation => "gic.op",
            Self::InterruptControllerRead => "gic.op.read",
            Self::InterruptControllerBarrier => "gic.barrier",
            Self::TraceInstrumentation => "trace.instrument",
        }
    }

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
            Self::ShadowStackPush => "sspush",
            Self::ShadowStackPopCheck => "sspopchk",
            Self::ShadowStackSwapDword => "ssamoswap.w",
            Self::ShadowStackSwapQword => "ssamoswap.d",
            Self::BoundTableLoad => "bndldx",
            Self::BoundTableStore => "bndstx",
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
            Self::ControlTransferRecordsClear => "ctr.clear",
            Self::WriteVectorConfiguration => "vector.configuration.write",
            Self::WriteVectorProgress => "vector.progress.write",
            Self::ControlFlowPredictionRestrict => "cfp",
            Self::CachePrefetchPredictionRestrict => "cpp",
            Self::DataValuePredictionRestrict => "dvp",
            Self::OtherSpeculationClear => "cosp",
            Self::PhysicalAddressSpaceChange => "apas",
            Self::InterruptControllerOperation => "gic",
            Self::InterruptControllerRead => "gicr",
            Self::InterruptControllerBarrier => "gsb",
            Self::TraceInstrumentation => "trcit",
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
            Self::ShadowStackPush => "system.state.sspush",
            Self::ShadowStackPopCheck => "system.state.sspopchk",
            Self::ShadowStackSwapDword => "system.state.ssamoswap.w",
            Self::ShadowStackSwapQword => "system.state.ssamoswap.d",
            Self::BoundTableLoad => "system.state.bndldx",
            Self::BoundTableStore => "system.state.bndstx",
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
            Self::ControlTransferRecordsClear => "system.state.ctr.clear",
            Self::WriteVectorConfiguration => "system.state.vector.configuration.write",
            Self::WriteVectorProgress => "system.state.vector.progress.write",
            Self::ControlFlowPredictionRestrict => "system.state.cfp",
            Self::CachePrefetchPredictionRestrict => "system.state.cpp",
            Self::DataValuePredictionRestrict => "system.state.dvp",
            Self::OtherSpeculationClear => "system.state.cosp",
            Self::PhysicalAddressSpaceChange => "system.state.apas",
            Self::InterruptControllerOperation => "system.state.gic",
            Self::InterruptControllerRead => "system.state.gicr",
            Self::InterruptControllerBarrier => "system.state.gsb",
            Self::TraceInstrumentation => "system.state.trcit",
        }
    }

    /// Returns the precise effect summary for this operation.
    ///
    /// A query of a processor register (`rdfsbase`, `rdpkru`, `rdpid`, `tt`)
    /// reads no memory: it reports [`SsaEffectKind::RuntimeDefined`], the value
    /// the machine supplies where control reaches it, so it neither keeps a
    /// store alive nor moves. `Read` is reserved for the operations that sample
    /// memory into a destination the lift binds: the descriptor-table queries
    /// (`lar`, `lsl`, `verr`, `verw`), the far-pointer loads and the
    /// reverse-map query. A system instruction returning a result (`sysl`, a
    /// guarded control stack pop) reports `ReadWrite`, since one may read and
    /// write memory, as the stack pop does, and `slwpcb` reports `Write`: it
    /// flushes the profiling state into the control block before it returns
    /// its address. Port I/O and the enqueue stores report
    /// `ReadWrite` because they both observe a device and may change it, the
    /// shadow-stack swaps because they read and write shadow-stack memory, and
    /// the bound-table store because it reads the bound directory before it
    /// writes the table; the bound-table load reads the directory and the table
    /// into the bound register it binds. Both bound-table operations may raise a
    /// bound-range fault on an invalid directory entry. The two
    /// state restorations that resume a suspended context report `Call` with a
    /// call control effect, exactly as [`SystemOpKind::InterruptReturn`](crate::ir::ops::kinds::SystemOpKind::InterruptReturn) does.
    /// Everything else writes machine state.
    #[must_use]
    pub const fn effects(self) -> SsaEffects {
        match self {
            // The prediction restrictions, the interrupt controller's
            // operations and barrier and a trace packet change processor
            // state alone, through no memory.
            Self::WriteVectorProgress
            | Self::ControlFlowPredictionRestrict
            | Self::CachePrefetchPredictionRestrict
            | Self::DataValuePredictionRestrict
            | Self::OtherSpeculationClear
            | Self::InterruptControllerOperation
            | Self::InterruptControllerBarrier
            | Self::TraceInstrumentation => {
                SsaEffects::new(SsaEffectKind::Write, false).with_memory(MemoryEffectLocation::None)
            }
            // An acknowledgement reads the controller and changes it.
            Self::InterruptControllerRead => SsaEffects::new(SsaEffectKind::ReadWrite, false)
                .with_memory(MemoryEffectLocation::None),
            Self::WriteVectorConfiguration => {
                SsaEffects::new(SsaEffectKind::Write, true).with_memory(MemoryEffectLocation::None)
            }
            Self::ReadAccessRights
            | Self::ReadSegmentLimit
            | Self::VerifySegmentReadable
            | Self::VerifySegmentWritable
            | Self::LoadFarPointerDs
            | Self::LoadFarPointerEs
            | Self::LoadFarPointerFs
            | Self::LoadFarPointerGs
            | Self::LoadFarPointerSs
            | Self::ReverseMapQuery => SsaEffects::new(SsaEffectKind::Read, false),
            Self::ReadFsBase
            | Self::ReadGsBase
            | Self::ShadowStackReadDword
            | Self::ShadowStackReadQword
            | Self::UserInterruptTest
            | Self::ReadProtectionKeyRights
            | Self::ReadProcessorId
            | Self::ReadProcessorPower
            | Self::VendorCpuRead
            | Self::VendorReadModelRegister
            | Self::VendorReadShadowRegister
            | Self::VendorReadDebugRegister
            | Self::TestTarget
            | Self::TestTargetAlternate
            | Self::TestTargetAlternateUnprivileged
            | Self::TestTargetUnprivileged => SsaEffects::new(SsaEffectKind::RuntimeDefined, false),
            Self::SystemInstructionRead => SsaEffects::new(SsaEffectKind::ReadWrite, false),
            Self::PortIn
            | Self::PortOut
            | Self::PortInStringByte
            | Self::PortInStringWord
            | Self::PortInStringDword
            | Self::PortOutStringByte
            | Self::PortOutStringWord
            | Self::PortOutStringDword
            | Self::EnqueueCommand
            | Self::EnqueueCommandSupervisor
            | Self::ShadowStackSwapDword
            | Self::ShadowStackSwapQword => SsaEffects::new(SsaEffectKind::ReadWrite, false),
            Self::BoundTableLoad => SsaEffects::new(SsaEffectKind::Read, true),
            Self::BoundTableStore => SsaEffects::new(SsaEffectKind::ReadWrite, true),
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
    /// than leaving every later use of it unbound. The stores of the machine
    /// status word, the descriptor-table registers and the task register
    /// (`smsw`, `sgdt`, `sidt`, `sldt`, `str`) write their operand, a register
    /// or the memory it addresses.
    #[must_use]
    pub const fn writes_destination_operand(self) -> bool {
        matches!(
            self,
            Self::StoreMachineStatusWord
                | Self::StoreGlobalDescriptorTable
                | Self::StoreInterruptDescriptorTable
                | Self::StoreLocalDescriptorTable
                | Self::StoreTaskRegister
                | Self::ReadAccessRights
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
                | Self::ShadowStackSwapDword
                | Self::ShadowStackSwapQword
                | Self::BoundTableLoad
                | Self::ReadProcessorId
                | Self::ProfilingStoreControlBlock
                | Self::TestTarget
                | Self::TestTargetAlternate
                | Self::TestTargetAlternateUnprivileged
                | Self::TestTargetUnprivileged
                | Self::SystemInstructionRead
                | Self::InterruptControllerRead
                | Self::InterruptDisable
                | Self::InterruptEnable
        )
    }
}
