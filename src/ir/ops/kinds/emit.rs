//! The content encoding of each kinds type ([`Emit`]): a tag for its
//! variant, then its fields in declaration order.

use std::num::NonZeroU16;

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{
        content::ContentSink,
        ops::kinds::{
            AccessPermission, AccessQualifiers, AtomicAccessWidth, AtomicOrdering, AtomicRmwOp,
            AtomicStatus, BarrierOp, BcdAdjustKind, BreakpointOp, CacheMaintenanceOp,
            CarrylessPart, CmpKind, CompareBlkData, CrcPolynomial, FenceKind, FloatFunctionKind,
            FpuControlKind, HardwareEngineOp, HintOp, HypervisorOp, InterruptReturnOp,
            KindedVecData, Locality, MachineStateOp, MemoryTagKind, NativeInstructionMetadata,
            NativeKindedData, PacKind, RcwProtection, RuntimeValueKind, ScalarCipherKind, SysRegOp,
            SystemOpKind, SystemTransactionKind, TlbMaintenanceOp, TrapOp,
        },
    },
    target::Target,
};

impl<T: Target> Emit<T> for CompareBlkData {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            remaining,
            left_next,
            right_next,
            flags,
            left,
            right,
            count,
            flags_in,
            backward,
            element_bytes,
            while_equal,
            left_access,
            right_access,
        } = self;
        Emit::emit(remaining, writer)?;
        Emit::emit(left_next, writer)?;
        Emit::emit(right_next, writer)?;
        Emit::emit(flags, writer)?;
        Emit::emit(left, writer)?;
        Emit::emit(right, writer)?;
        Emit::emit(count, writer)?;
        Emit::emit(flags_in, writer)?;
        Emit::emit(backward, writer)?;
        Emit::emit(element_bytes, writer)?;
        Emit::emit(while_equal, writer)?;
        Emit::emit(left_access, writer)?;
        Emit::emit(right_access, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for AtomicStatus {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let tag = match self {
            Self::Unspecified => 0u16,
            Self::ZeroNonzero => 1u16,
            Self::ZeroOne => 2u16,
            Self::OneZero => 3u16,
        };
        Emit::emit(&tag, writer)
    }
}

impl<T: Target> Emit<T> for AccessPermission {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Current => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Unprivileged => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Execute => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::ShadowStack => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::UnprivilegedShadowStack => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for AtomicAccessWidth {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Bits8 => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Bits16 => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Bits32 => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Bits64 => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Bits128 => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Pointer => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Bits512 => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for RcwProtection {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Enabled => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Regime => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for AtomicOrdering {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Relaxed => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Acquire => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Release => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::AcqRel => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::SeqCst => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for AtomicRmwOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Xchg => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Add => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Sub => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::And => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Or => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Xor => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Min => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Max => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::AndNot => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::MinU => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::MaxU => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::Neg => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::FAdd => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::FMax => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::FMin => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::FMaxNum => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::FMinNum => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for BarrierOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Serialize => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::CommitStores => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::CommitPersistent => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::InstructionSync => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::SpeculationBarrier => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::ClearExclusive => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::ErrorSync => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::ClearBranchHistory => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for BcdAdjustKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::DecimalAddAdjust => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::DecimalSubAdjust => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::AsciiAddAdjust => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::AsciiSubAdjust => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::AsciiMulAdjust => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::AsciiDivAdjust => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for BreakpointOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Breakpoint => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::UndefinedInstruction => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::DebugHalt => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::BreakpointInPlace => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for CacheMaintenanceOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::InvalidateAll => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::WriteBackInvalidateAll => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::WriteBackAll => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::FlushLine => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::WriteBackLine => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::DemoteLine => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::ZeroLine => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::EvictLine => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::InvalidateBlock => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::DataCacheOperation => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::InstructionCacheOperation => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for CarrylessPart {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Low => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::High => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Reversed => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Full => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for CmpKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Eq => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Ne => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Lt => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Le => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Gt => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Ge => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for CrcPolynomial {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Ieee => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Castagnoli => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for FenceKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Full => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Acquire => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Release => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::AcqRel => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::SeqCst => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Loads => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Stores => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for FpuControlKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::LoadControlWord => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::StoreControlWord => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::StoreStatusWord => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::LoadEnvironment => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::StoreEnvironment => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Save => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Restore => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::SaveExtended => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::RestoreExtended => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::ClearExceptions => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::DecrementStackTop => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::IncrementStackTop => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::FreeRegister => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::NoOp => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::Wait => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::Initialize => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::EmptyMmxState => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::LoadMxcsr => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::StoreMxcsr => {
                Emit::emit(&18u16, writer)?;
                Ok(())
            }
            Self::SaveExtendedMasked => {
                Emit::emit(&19u16, writer)?;
                Ok(())
            }
            Self::SaveExtendedCompact => {
                Emit::emit(&20u16, writer)?;
                Ok(())
            }
            Self::SaveExtendedOptimized => {
                Emit::emit(&21u16, writer)?;
                Ok(())
            }
            Self::SaveExtendedSupervisor => {
                Emit::emit(&22u16, writer)?;
                Ok(())
            }
            Self::RestoreExtendedMasked => {
                Emit::emit(&23u16, writer)?;
                Ok(())
            }
            Self::RestoreExtendedSupervisor => {
                Emit::emit(&24u16, writer)?;
                Ok(())
            }
            Self::FreeRegisterPop => {
                Emit::emit(&25u16, writer)?;
                Ok(())
            }
            Self::StoreEnvironmentShort => {
                Emit::emit(&26u16, writer)?;
                Ok(())
            }
            Self::LoadEnvironmentShort => {
                Emit::emit(&27u16, writer)?;
                Ok(())
            }
            Self::SaveShort => {
                Emit::emit(&28u16, writer)?;
                Ok(())
            }
            Self::RestoreShort => {
                Emit::emit(&29u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for HardwareEngineOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::RandomStore => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::CipherEcb => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::CipherCbc => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::CipherCtr => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::CipherCfb => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::CipherOfb => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::HashSha1 => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::HashSha256 => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::HashSha512 => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::MontgomeryMultiply => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::CounterModeEncrypt => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::CounterModeHash => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for HintOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::NoOperation => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::SpinWait => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::IndirectBranchLandingPad => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::LabelledLandingPad => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::WaitForEvent => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::WaitForEventTimed => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::WaitForEventBounded => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::SendEvent => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::SendEventLocal => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::Prefetch => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::PrefetchForWrite => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::PrefetchInstruction => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::PrefetchNonTemporal => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::PrefetchVector => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::DataGatheringHint => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::ProfilingSynchronizationBarrier => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::ExecutionHazardBarrier => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::DebugHint => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::ConditionalExecutionBlock => {
                Emit::emit(&18u16, writer)?;
                Ok(())
            }
            Self::GenericHint => {
                Emit::emit(&19u16, writer)?;
                Ok(())
            }
            Self::NonTemporalLocality => {
                Emit::emit(&20u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for HypervisorOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::HypervisorCall => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::SecureMonitorCall => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::GuestEnter => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::GuestExit => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::VirtualizationEnter => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::VirtualizationLeave => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::ControlBlockLoad => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::ControlBlockStore => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::ControlBlockClear => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::ControlBlockRead => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::ControlBlockWrite => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::VirtualizationFunction => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::GlobalInterruptEnable => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::GlobalInterruptDisable => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::SecureInitialize => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::EnclaveOperation => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::EnclaveReturn => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::TrustedModuleCall => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::TrustedModuleReturn => {
                Emit::emit(&18u16, writer)?;
                Ok(())
            }
            Self::TrustedModuleOperation => {
                Emit::emit(&19u16, writer)?;
                Ok(())
            }
            Self::SemihostingCall => {
                Emit::emit(&20u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for InterruptReturnOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::ExceptionReturn => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::AuthenticatedExceptionReturn => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::MachineTrapReturn => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::SupervisorTrapReturn => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::UserTrapReturn => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::NmiTrapReturn => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::DebugReturn => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for MachineStateOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::LoadGlobalDescriptorTable => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::StoreGlobalDescriptorTable => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::LoadInterruptDescriptorTable => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::StoreInterruptDescriptorTable => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::LoadLocalDescriptorTable => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::StoreLocalDescriptorTable => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::LoadTaskRegister => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::StoreTaskRegister => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::LoadMachineStatusWord => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::StoreMachineStatusWord => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::ClearTaskSwitched => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::ReadAccessRights => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::ReadSegmentLimit => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::VerifySegmentReadable => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::VerifySegmentWritable => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::AdjustRequestedPrivilege => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::LoadFarPointerDs => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::LoadFarPointerEs => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::LoadFarPointerFs => {
                Emit::emit(&18u16, writer)?;
                Ok(())
            }
            Self::LoadFarPointerGs => {
                Emit::emit(&19u16, writer)?;
                Ok(())
            }
            Self::LoadFarPointerSs => {
                Emit::emit(&20u16, writer)?;
                Ok(())
            }
            Self::ReadFsBase => {
                Emit::emit(&21u16, writer)?;
                Ok(())
            }
            Self::WriteFsBase => {
                Emit::emit(&22u16, writer)?;
                Ok(())
            }
            Self::ReadGsBase => {
                Emit::emit(&23u16, writer)?;
                Ok(())
            }
            Self::WriteGsBase => {
                Emit::emit(&24u16, writer)?;
                Ok(())
            }
            Self::SwapGsBase => {
                Emit::emit(&25u16, writer)?;
                Ok(())
            }
            Self::LoadKernelGsBase => {
                Emit::emit(&26u16, writer)?;
                Ok(())
            }
            Self::PortIn => {
                Emit::emit(&27u16, writer)?;
                Ok(())
            }
            Self::PortOut => {
                Emit::emit(&28u16, writer)?;
                Ok(())
            }
            Self::PortInStringByte => {
                Emit::emit(&29u16, writer)?;
                Ok(())
            }
            Self::PortInStringWord => {
                Emit::emit(&30u16, writer)?;
                Ok(())
            }
            Self::PortInStringDword => {
                Emit::emit(&31u16, writer)?;
                Ok(())
            }
            Self::PortOutStringByte => {
                Emit::emit(&32u16, writer)?;
                Ok(())
            }
            Self::PortOutStringWord => {
                Emit::emit(&33u16, writer)?;
                Ok(())
            }
            Self::PortOutStringDword => {
                Emit::emit(&34u16, writer)?;
                Ok(())
            }
            Self::ShadowStackIncrementDword => {
                Emit::emit(&35u16, writer)?;
                Ok(())
            }
            Self::ShadowStackIncrementQword => {
                Emit::emit(&36u16, writer)?;
                Ok(())
            }
            Self::ShadowStackReadDword => {
                Emit::emit(&37u16, writer)?;
                Ok(())
            }
            Self::ShadowStackReadQword => {
                Emit::emit(&38u16, writer)?;
                Ok(())
            }
            Self::ShadowStackWriteDword => {
                Emit::emit(&39u16, writer)?;
                Ok(())
            }
            Self::ShadowStackWriteQword => {
                Emit::emit(&40u16, writer)?;
                Ok(())
            }
            Self::ShadowStackWriteUserDword => {
                Emit::emit(&41u16, writer)?;
                Ok(())
            }
            Self::ShadowStackWriteUserQword => {
                Emit::emit(&42u16, writer)?;
                Ok(())
            }
            Self::ShadowStackMarkBusy => {
                Emit::emit(&43u16, writer)?;
                Ok(())
            }
            Self::ShadowStackClearBusy => {
                Emit::emit(&44u16, writer)?;
                Ok(())
            }
            Self::ShadowStackRestore => {
                Emit::emit(&45u16, writer)?;
                Ok(())
            }
            Self::ShadowStackSavePrevious => {
                Emit::emit(&46u16, writer)?;
                Ok(())
            }
            Self::ShadowStackPush => {
                Emit::emit(&47u16, writer)?;
                Ok(())
            }
            Self::ShadowStackPopCheck => {
                Emit::emit(&48u16, writer)?;
                Ok(())
            }
            Self::ShadowStackSwapDword => {
                Emit::emit(&49u16, writer)?;
                Ok(())
            }
            Self::ShadowStackSwapQword => {
                Emit::emit(&50u16, writer)?;
                Ok(())
            }
            Self::BoundTableLoad => {
                Emit::emit(&51u16, writer)?;
                Ok(())
            }
            Self::BoundTableStore => {
                Emit::emit(&52u16, writer)?;
                Ok(())
            }
            Self::UserInterruptDisable => {
                Emit::emit(&53u16, writer)?;
                Ok(())
            }
            Self::UserInterruptEnable => {
                Emit::emit(&54u16, writer)?;
                Ok(())
            }
            Self::UserInterruptTest => {
                Emit::emit(&55u16, writer)?;
                Ok(())
            }
            Self::UserInterruptReturn => {
                Emit::emit(&56u16, writer)?;
                Ok(())
            }
            Self::UserInterruptSend => {
                Emit::emit(&57u16, writer)?;
                Ok(())
            }
            Self::Halt => {
                Emit::emit(&58u16, writer)?;
                Ok(())
            }
            Self::MonitorAddress => {
                Emit::emit(&59u16, writer)?;
                Ok(())
            }
            Self::MonitorAddressExtended => {
                Emit::emit(&60u16, writer)?;
                Ok(())
            }
            Self::MonitorAddressUser => {
                Emit::emit(&61u16, writer)?;
                Ok(())
            }
            Self::MonitorWait => {
                Emit::emit(&62u16, writer)?;
                Ok(())
            }
            Self::MonitorWaitTimed => {
                Emit::emit(&63u16, writer)?;
                Ok(())
            }
            Self::MonitorWaitUser => {
                Emit::emit(&64u16, writer)?;
                Ok(())
            }
            Self::TimedPause => {
                Emit::emit(&65u16, writer)?;
                Ok(())
            }
            Self::ReadProtectionKeyRights => {
                Emit::emit(&66u16, writer)?;
                Ok(())
            }
            Self::WriteProtectionKeyRights => {
                Emit::emit(&67u16, writer)?;
                Ok(())
            }
            Self::ReadProcessorId => {
                Emit::emit(&68u16, writer)?;
                Ok(())
            }
            Self::ReadProcessorPower => {
                Emit::emit(&69u16, writer)?;
                Ok(())
            }
            Self::ReadMsrList => {
                Emit::emit(&70u16, writer)?;
                Ok(())
            }
            Self::WriteMsrList => {
                Emit::emit(&71u16, writer)?;
                Ok(())
            }
            Self::WriteMsrNonSerializing => {
                Emit::emit(&72u16, writer)?;
                Ok(())
            }
            Self::KeyLockerLoadInternalKey => {
                Emit::emit(&73u16, writer)?;
                Ok(())
            }
            Self::KeyLockerEncodeKey128 => {
                Emit::emit(&74u16, writer)?;
                Ok(())
            }
            Self::KeyLockerEncodeKey256 => {
                Emit::emit(&75u16, writer)?;
                Ok(())
            }
            Self::PlatformConfigure => {
                Emit::emit(&76u16, writer)?;
                Ok(())
            }
            Self::PlatformBindKey => {
                Emit::emit(&77u16, writer)?;
                Ok(())
            }
            Self::PlatformSmashPage => {
                Emit::emit(&78u16, writer)?;
                Ok(())
            }
            Self::PlatformValidatePage => {
                Emit::emit(&79u16, writer)?;
                Ok(())
            }
            Self::ReverseMapAdjust => {
                Emit::emit(&80u16, writer)?;
                Ok(())
            }
            Self::ReverseMapQuery => {
                Emit::emit(&81u16, writer)?;
                Ok(())
            }
            Self::ReverseMapUpdate => {
                Emit::emit(&82u16, writer)?;
                Ok(())
            }
            Self::HistoryReset => {
                Emit::emit(&83u16, writer)?;
                Ok(())
            }
            Self::DirectStoreWord => {
                Emit::emit(&84u16, writer)?;
                Ok(())
            }
            Self::DirectStoreBlock => {
                Emit::emit(&85u16, writer)?;
                Ok(())
            }
            Self::EnqueueCommand => {
                Emit::emit(&86u16, writer)?;
                Ok(())
            }
            Self::EnqueueCommandSupervisor => {
                Emit::emit(&87u16, writer)?;
                Ok(())
            }
            Self::TraceWrite => {
                Emit::emit(&88u16, writer)?;
                Ok(())
            }
            Self::ResumeFromSystemManagement => {
                Emit::emit(&89u16, writer)?;
                Ok(())
            }
            Self::ProfilingLoadControlBlock => {
                Emit::emit(&90u16, writer)?;
                Ok(())
            }
            Self::ProfilingStoreControlBlock => {
                Emit::emit(&91u16, writer)?;
                Ok(())
            }
            Self::ProfilingInsertValue => {
                Emit::emit(&92u16, writer)?;
                Ok(())
            }
            Self::ProfilingInsertRecord => {
                Emit::emit(&93u16, writer)?;
                Ok(())
            }
            Self::VendorAlternateInstruction => {
                Emit::emit(&94u16, writer)?;
                Ok(())
            }
            Self::VendorBranchBuffer0Reset => {
                Emit::emit(&95u16, writer)?;
                Ok(())
            }
            Self::VendorBranchBuffer1Reset => {
                Emit::emit(&96u16, writer)?;
                Ok(())
            }
            Self::VendorCpuRead => {
                Emit::emit(&97u16, writer)?;
                Ok(())
            }
            Self::VendorCpuWrite => {
                Emit::emit(&98u16, writer)?;
                Ok(())
            }
            Self::VendorDebugInterrupt => {
                Emit::emit(&99u16, writer)?;
                Ok(())
            }
            Self::VendorSoftwareSmi => {
                Emit::emit(&100u16, writer)?;
                Ok(())
            }
            Self::VendorRoundToNearestInteger => {
                Emit::emit(&101u16, writer)?;
                Ok(())
            }
            Self::VendorInsertBitString => {
                Emit::emit(&102u16, writer)?;
                Ok(())
            }
            Self::VendorExtractBitString => {
                Emit::emit(&103u16, writer)?;
                Ok(())
            }
            Self::VendorLoadAllRegisters => {
                Emit::emit(&104u16, writer)?;
                Ok(())
            }
            Self::VendorStoreAllRegisters => {
                Emit::emit(&105u16, writer)?;
                Ok(())
            }
            Self::VendorReadModelRegister => {
                Emit::emit(&106u16, writer)?;
                Ok(())
            }
            Self::VendorReadShadowRegister => {
                Emit::emit(&107u16, writer)?;
                Ok(())
            }
            Self::VendorWriteShadowRegister => {
                Emit::emit(&108u16, writer)?;
                Ok(())
            }
            Self::VendorReadDebugRegister => {
                Emit::emit(&109u16, writer)?;
                Ok(())
            }
            Self::VendorWriteDebugRegister => {
                Emit::emit(&110u16, writer)?;
                Ok(())
            }
            Self::VendorRestoreDataSegment => {
                Emit::emit(&111u16, writer)?;
                Ok(())
            }
            Self::VendorSaveDataSegment => {
                Emit::emit(&112u16, writer)?;
                Ok(())
            }
            Self::VendorRestoreLocalDescriptor => {
                Emit::emit(&113u16, writer)?;
                Ok(())
            }
            Self::VendorSaveLocalDescriptor => {
                Emit::emit(&114u16, writer)?;
                Ok(())
            }
            Self::VendorRestoreTaskState => {
                Emit::emit(&115u16, writer)?;
                Ok(())
            }
            Self::VendorSaveTaskState => {
                Emit::emit(&116u16, writer)?;
                Ok(())
            }
            Self::VendorSetProtectedFlat => {
                Emit::emit(&117u16, writer)?;
                Ok(())
            }
            Self::VendorUnprivilegedMove => {
                Emit::emit(&118u16, writer)?;
                Ok(())
            }
            Self::VendorUndocumented => {
                Emit::emit(&119u16, writer)?;
                Ok(())
            }
            Self::ProcessorStateChange => {
                Emit::emit(&120u16, writer)?;
                Ok(())
            }
            Self::SetEndianness => {
                Emit::emit(&121u16, writer)?;
                Ok(())
            }
            Self::SetPrivilegedAccessNever => {
                Emit::emit(&122u16, writer)?;
                Ok(())
            }
            Self::SecureGateway => {
                Emit::emit(&123u16, writer)?;
                Ok(())
            }
            Self::StoreReturnStateDecrementAfter => {
                Emit::emit(&124u16, writer)?;
                Ok(())
            }
            Self::StoreReturnStateDecrementBefore => {
                Emit::emit(&125u16, writer)?;
                Ok(())
            }
            Self::StoreReturnStateIncrementAfter => {
                Emit::emit(&126u16, writer)?;
                Ok(())
            }
            Self::StoreReturnStateIncrementBefore => {
                Emit::emit(&127u16, writer)?;
                Ok(())
            }
            Self::TestTarget => {
                Emit::emit(&128u16, writer)?;
                Ok(())
            }
            Self::TestTargetAlternate => {
                Emit::emit(&129u16, writer)?;
                Ok(())
            }
            Self::TestTargetAlternateUnprivileged => {
                Emit::emit(&130u16, writer)?;
                Ok(())
            }
            Self::TestTargetUnprivileged => {
                Emit::emit(&131u16, writer)?;
                Ok(())
            }
            Self::AddressTranslate => {
                Emit::emit(&132u16, writer)?;
                Ok(())
            }
            Self::BranchRecordBuffer => {
                Emit::emit(&133u16, writer)?;
                Ok(())
            }
            Self::DebugChangeStateEl1 => {
                Emit::emit(&134u16, writer)?;
                Ok(())
            }
            Self::DebugChangeStateEl2 => {
                Emit::emit(&135u16, writer)?;
                Ok(())
            }
            Self::DebugChangeStateEl3 => {
                Emit::emit(&136u16, writer)?;
                Ok(())
            }
            Self::DebugRestoreState => {
                Emit::emit(&137u16, writer)?;
                Ok(())
            }
            Self::StreamingModeStart => {
                Emit::emit(&138u16, writer)?;
                Ok(())
            }
            Self::StreamingModeStop => {
                Emit::emit(&139u16, writer)?;
                Ok(())
            }
            Self::SystemInstruction => {
                Emit::emit(&140u16, writer)?;
                Ok(())
            }
            Self::SystemInstructionRead => {
                Emit::emit(&141u16, writer)?;
                Ok(())
            }
            Self::InterruptDisable => {
                Emit::emit(&142u16, writer)?;
                Ok(())
            }
            Self::InterruptEnable => {
                Emit::emit(&143u16, writer)?;
                Ok(())
            }
            Self::WaitForInterrupt => {
                Emit::emit(&144u16, writer)?;
                Ok(())
            }
            Self::WaitForInterruptTimed => {
                Emit::emit(&145u16, writer)?;
                Ok(())
            }
            Self::ControlTransferRecordsClear => {
                Emit::emit(&146u16, writer)?;
                Ok(())
            }
            Self::WriteVectorConfiguration => {
                Emit::emit(&147u16, writer)?;
                Ok(())
            }
            Self::WriteVectorProgress => {
                Emit::emit(&148u16, writer)?;
                Ok(())
            }
            Self::ControlFlowPredictionRestrict => {
                Emit::emit(&149u16, writer)?;
                Ok(())
            }
            Self::CachePrefetchPredictionRestrict => {
                Emit::emit(&150u16, writer)?;
                Ok(())
            }
            Self::DataValuePredictionRestrict => {
                Emit::emit(&151u16, writer)?;
                Ok(())
            }
            Self::OtherSpeculationClear => {
                Emit::emit(&152u16, writer)?;
                Ok(())
            }
            Self::PhysicalAddressSpaceChange => {
                Emit::emit(&153u16, writer)?;
                Ok(())
            }
            Self::InterruptControllerOperation => {
                Emit::emit(&154u16, writer)?;
                Ok(())
            }
            Self::InterruptControllerRead => {
                Emit::emit(&155u16, writer)?;
                Ok(())
            }
            Self::InterruptControllerBarrier => {
                Emit::emit(&156u16, writer)?;
                Ok(())
            }
            Self::TraceInstrumentation => {
                Emit::emit(&157u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for AccessQualifiers {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            address_space,
            permission,
            locality,
            volatile,
            alignment,
            required,
        } = self;
        Emit::emit(&address_space.map(NonZeroU16::get), writer)?;
        Emit::emit(permission, writer)?;
        Emit::emit(locality, writer)?;
        Emit::emit(volatile, writer)?;
        Emit::emit(&alignment.map(|bytes| bytes.get()), writer)?;
        Emit::emit(&required.map(|bytes| bytes.get()), writer)
    }
}

impl<T: Target> Emit<T> for MemoryTagKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Random => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Add => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for NativeInstructionMetadata {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            architecture,
            address,
            raw_bytes,
        } = self;
        Emit::emit(architecture, writer)?;
        Emit::emit(address, writer)?;
        Emit::emit(raw_bytes, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for PacKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Sign => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Authenticate => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Strip => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::GenericMac => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::SignDiversified => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::AuthenticateDiversified => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for RuntimeValueKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Unspecified => Emit::emit(&2u16, writer),
            Self::ExceptionObject => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::HandlerEntry => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for ScalarCipherKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Aes32EncryptByte => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Aes32EncryptByteMix => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Aes32DecryptByte => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Aes32DecryptByteMix => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Aes64EncryptHalf => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Aes64EncryptHalfMix => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Aes64DecryptHalf => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Aes64DecryptHalfMix => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::Aes64InverseMix => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::Aes64KeyStep1 => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::Aes64KeyStep2 => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::Sm4EncryptByte => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::Sm4KeyByte => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for SysRegOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::ReadSystemRegister => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::ReadModelSpecific => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::WriteModelSpecific => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::ReadExtendedControl => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::WriteExtendedControl => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::ReadControlRegister => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::WriteControlRegister => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::ReadDebugRegister => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::WriteDebugRegister => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::WriteSystemRegister => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::ReadSystemRegisterPair => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::WriteSystemRegisterPair => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::CoprocessorOperation => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::CoprocessorLoad => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::CoprocessorStore => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::ExchangeSystemRegister => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::SetSystemRegisterBits => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::ClearSystemRegisterBits => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::WriteMultiplier => {
                Emit::emit(&18u16, writer)?;
                Ok(())
            }
            Self::WriteMultiplierProduct => {
                Emit::emit(&19u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for SystemOpKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::CpuId => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Timestamp { aux } => {
                Emit::emit(&1u16, writer)?;
                Emit::emit(aux, writer)?;
                Ok(())
            }
            Self::ReadPerfCounter => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::ControlRegister(field_0) => {
                Emit::emit(&3u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::SystemCall => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::SystemReturn => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Trap { op, vector } => {
                Emit::emit(&6u16, writer)?;
                Emit::emit(op, writer)?;
                Emit::emit(vector, writer)?;
                Ok(())
            }
            Self::InterruptReturn(field_0) => {
                Emit::emit(&7u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::CacheMaintenance(field_0) => {
                Emit::emit(&8u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::TlbMaintenance(field_0) => {
                Emit::emit(&9u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Barrier(field_0) => {
                Emit::emit(&10u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::MachineState(field_0) => {
                Emit::emit(&11u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Hypervisor(field_0) => {
                Emit::emit(&12u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::HardwareEngine(field_0) => {
                Emit::emit(&13u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Transaction(field_0) => {
                Emit::emit(&14u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Hint(field_0) => {
                Emit::emit(&15u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for SystemTransactionKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Start => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Commit => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Cancel => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Test => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::ResumeLoadTracking => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::SuspendLoadTracking => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for TlbMaintenanceOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::InvalidatePage => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::InvalidatePageGuest => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::InvalidatePageBroadcast => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::InvalidateByContext => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::InvalidateNestedMapping => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Synchronize => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::WriteEntryIndexed => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::WriteEntryRandom => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::ReadEntry => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::Probe => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::SynchronizeGuest => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::SynchronizeNested => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::InvalidateUnordered => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::InvalidateGuestUnordered => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::InvalidateNestedUnordered => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::OrderBeforeInvalidation => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::OrderAfterInvalidation => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for TrapOp {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::SoftwareInterrupt => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::OverflowTrap => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::BoundsCheck => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::BoundsLower => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::BoundsUpper => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::TrapIfEqual => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::TrapIfNotEqual => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::TrapIfLess => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::TrapIfLessUnsigned => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::TrapIfGreaterOrEqual => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::TrapIfGreaterOrEqualUnsigned => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::TrapIfGreater => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::TrapIfGreaterUnsigned => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::TrapIfLessOrEqual => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::TrapIfLessOrEqualUnsigned => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for Locality {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Temporal => Emit::emit(&0u16, writer),
            Self::NonTemporal => Emit::emit(&1u16, writer),
        }
    }
}

impl<T: Target> Emit<T> for FloatFunctionKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Sin => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Cos => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::SinCos => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Tan => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Atan => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Exp2m1 => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Ylog2 => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Ylog2p1 => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::Rem => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::Rem1 => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::Scale => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::Extract => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target, K: Emit<T>> Emit<T> for KindedVecData<K> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            kind,
            outputs,
            inputs,
        } = self;
        Emit::emit(kind, writer)?;
        Emit::emit(outputs, writer)?;
        Emit::emit(inputs, writer)?;
        Ok(())
    }
}

impl<T: Target, K: Emit<T>> Emit<T> for NativeKindedData<K> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            kind,
            mnemonic,
            metadata,
            outputs,
            inputs,
        } = self;
        Emit::emit(kind, writer)?;
        Emit::emit(mnemonic, writer)?;
        Emit::emit(metadata, writer)?;
        Emit::emit(outputs, writer)?;
        Emit::emit(inputs, writer)?;
        Ok(())
    }
}
