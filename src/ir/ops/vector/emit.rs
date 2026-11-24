//! The content encoding of each vector type ([`Emit`]): a tag for its
//! variant, then its fields in declaration order.

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{
        content::ContentSink,
        ops::vector::{
            ComplexPart, ComplexProduct, ComplexRotation, ComplexShape, ContractElements,
            Contraction, ControlledKind, DotShape, ElementPattern, FlagAdjustKind, FloatDotRule,
            FloatElements, FloatSource, FpHelperKind, MatrixLayout, MatrixShape, OuterProduct,
            OuterSelection, OuterShape, ReduceGroups, SveComputeKind, VectorBinaryKind,
            VectorBitmaskKind, VectorCastKind, VectorCompareKind, VectorCryptoKind, VectorElement,
            VectorElementCountData, VectorElementKind, VectorExtendInLaneData, VectorFaultMode,
            VectorMaskBinaryKind, VectorMaskMode, VectorPackKind, VectorPredicateWhileData,
            VectorReduceKind, VectorSegmentLayout, VectorShuffleSelector,
            VectorStructLoadReplicateData, VectorSveComputeData, VectorTernaryKind,
            VectorUnaryKind,
        },
    },
    target::Target,
};

impl<T: Target> Emit<T> for ElementPattern {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Pow2 => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Fixed(field_0) => {
                Emit::emit(&1u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Mul4 => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Mul3 => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::All => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Reserved(field_0) => {
                Emit::emit(&5u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for FlagAdjustKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::InvertCarry => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::RotateMaskInsert => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::SetNzFrom8 => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::SetNzFrom16 => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::ConvertToFpFlags => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::ConvertFromFpFlags => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::ClearCarry => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::SetCarry => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::ClearDirection => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::SetDirection => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::ClearAlignCheck => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::SetAlignCheck => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::ClearInterrupt => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::SetInterrupt => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::LoadStatusFromReg => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::StoreStatusToReg => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::SetRegFromCarry => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for FpHelperKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::ReciprocalExponent => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::ExtractExponent => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::ExpAccelerate => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::TrigMulAdd => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::TrigSelectMul => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::TrigSelect => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for SveComputeKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::AddCarryBottom => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::AddCarryTop => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::SubCarryBottom => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::SubCarryTop => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Histogram => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::HistogramSegment => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::MatchElements => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::NoMatchElements => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::ClampSigned => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::ClampUnsigned => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::DivideReversedSigned => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::DivideReversedUnsigned => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::InterleaveXorBottomTop => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::InterleaveXorTopBottom => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::SaturatingAbs => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::SaturatingNeg => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::LogicalNot => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::ComplexDotProduct => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::ComplexMulAddRounding => {
                Emit::emit(&18u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorBinaryKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Add => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Sub => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Mul => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Div => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Min => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Max => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::And => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Or => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::Xor => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::Shl => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::Shr => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::AndNot => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::MulHigh => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::MulHighRound => {
                Emit::emit(&21u16, writer)?;
                Ok(())
            }
            Self::Avg => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::AbsDiff => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::SatAdd => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::SatSub => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::Scale => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::Rol => {
                Emit::emit(&19u16, writer)?;
                Ok(())
            }
            Self::Ror => {
                Emit::emit(&20u16, writer)?;
                Ok(())
            }
            Self::VariableShiftLogical => {
                Emit::emit(&22u16, writer)?;
                Ok(())
            }
            Self::VariableShiftArithmetic => {
                Emit::emit(&23u16, writer)?;
                Ok(())
            }
            Self::ApplySign => {
                Emit::emit(&24u16, writer)?;
                Ok(())
            }
            Self::ReciprocalIter1 => {
                Emit::emit(&25u16, writer)?;
                Ok(())
            }
            Self::ReciprocalIter2 => {
                Emit::emit(&26u16, writer)?;
                Ok(())
            }
            Self::ReciprocalSqrtIter1 => {
                Emit::emit(&27u16, writer)?;
                Ok(())
            }
            Self::MaxMagnitude => {
                Emit::emit(&28u16, writer)?;
                Ok(())
            }
            Self::MulNegate => {
                Emit::emit(&29u16, writer)?;
                Ok(())
            }
            Self::OrNot => {
                Emit::emit(&30u16, writer)?;
                Ok(())
            }
            Self::AddAbs => {
                Emit::emit(&31u16, writer)?;
                Ok(())
            }
            Self::Rem => {
                Emit::emit(&32u16, writer)?;
                Ok(())
            }
            Self::Nor => {
                Emit::emit(&33u16, writer)?;
                Ok(())
            }
            Self::MulHighDoubleRound => {
                Emit::emit(&34u16, writer)?;
                Ok(())
            }
            Self::MulHighDoubleSat => {
                Emit::emit(&35u16, writer)?;
                Ok(())
            }
            Self::MulHighDoubleSatRound => {
                Emit::emit(&36u16, writer)?;
                Ok(())
            }
            Self::MinNum => {
                Emit::emit(&37u16, writer)?;
                Ok(())
            }
            Self::MaxNum => {
                Emit::emit(&38u16, writer)?;
                Ok(())
            }
            Self::MinMagnitude => {
                Emit::emit(&39u16, writer)?;
                Ok(())
            }
            Self::MinLess => {
                Emit::emit(&40u16, writer)?;
                Ok(())
            }
            Self::MaxGreater => {
                Emit::emit(&41u16, writer)?;
                Ok(())
            }
            Self::ScaleInteger => {
                Emit::emit(&42u16, writer)?;
                Ok(())
            }
            Self::MinimumNumber => {
                Emit::emit(&43u16, writer)?;
                Ok(())
            }
            Self::MaximumNumber => {
                Emit::emit(&44u16, writer)?;
                Ok(())
            }
            Self::MinimumMagnitude => {
                Emit::emit(&45u16, writer)?;
                Ok(())
            }
            Self::MaximumMagnitude => {
                Emit::emit(&46u16, writer)?;
                Ok(())
            }
            Self::MinimumMagnitudeNumber => {
                Emit::emit(&47u16, writer)?;
                Ok(())
            }
            Self::MaximumMagnitudeNumber => {
                Emit::emit(&48u16, writer)?;
                Ok(())
            }
            Self::MinAbs => {
                Emit::emit(&49u16, writer)?;
                Ok(())
            }
            Self::MaxAbs => {
                Emit::emit(&50u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorBitmaskKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::LaneMostSignificantBits => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::PredicateBits => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorCastKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Signed => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Unsigned => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Float => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::BFloat16 => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::FloatOpen => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorCompareKind {
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
            Self::Unordered => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Ordered => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::NotLt => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::NotLe => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::NotGe => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::NotGt => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::AlwaysTrue => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::AlwaysFalse => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::UnorderedEq => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::OrderedNe => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorCryptoKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::AesEncrypt => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::AesEncryptLast => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::AesDecrypt => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::AesDecryptLast => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::AesInvMixColumns => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::AesKeygenAssist => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::AesEncryptRound => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::AesDecryptRound => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::AesMixColumns => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::Sha1Rounds4 => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::Sha1NextE => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::Sha1Msg1 => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::Sha1Msg2 => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::Sha256Rounds2 => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::Sha256Msg1 => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::Sha256Msg2 => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::Sha512Rounds2 => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::Sha512Msg1 => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::Sha512Msg2 => {
                Emit::emit(&18u16, writer)?;
                Ok(())
            }
            Self::Sm3Msg1 => {
                Emit::emit(&19u16, writer)?;
                Ok(())
            }
            Self::Sm3Msg2 => {
                Emit::emit(&20u16, writer)?;
                Ok(())
            }
            Self::Sm3Rounds2 => {
                Emit::emit(&21u16, writer)?;
                Ok(())
            }
            Self::Sm4Key => {
                Emit::emit(&22u16, writer)?;
                Ok(())
            }
            Self::Sm4Rounds => {
                Emit::emit(&23u16, writer)?;
                Ok(())
            }
            Self::Gf2p8Affine => {
                Emit::emit(&24u16, writer)?;
                Ok(())
            }
            Self::Gf2p8AffineInv => {
                Emit::emit(&25u16, writer)?;
                Ok(())
            }
            Self::Gf2p8Mul => {
                Emit::emit(&26u16, writer)?;
                Ok(())
            }
            Self::AesEncryptKeyLocker => {
                Emit::emit(&27u16, writer)?;
                Ok(())
            }
            Self::AesDecryptKeyLocker => {
                Emit::emit(&28u16, writer)?;
                Ok(())
            }
            Self::Sha1HashChoose => {
                Emit::emit(&29u16, writer)?;
                Ok(())
            }
            Self::Sha1HashMajority => {
                Emit::emit(&30u16, writer)?;
                Ok(())
            }
            Self::Sha1HashParity => {
                Emit::emit(&31u16, writer)?;
                Ok(())
            }
            Self::Sha1FixedRotate => {
                Emit::emit(&32u16, writer)?;
                Ok(())
            }
            Self::Sha1ScheduleUpdate0 => {
                Emit::emit(&33u16, writer)?;
                Ok(())
            }
            Self::Sha1ScheduleUpdate1 => {
                Emit::emit(&34u16, writer)?;
                Ok(())
            }
            Self::Sha256Hash => {
                Emit::emit(&35u16, writer)?;
                Ok(())
            }
            Self::Sha256Hash2 => {
                Emit::emit(&36u16, writer)?;
                Ok(())
            }
            Self::Sha256ScheduleUpdate0 => {
                Emit::emit(&37u16, writer)?;
                Ok(())
            }
            Self::Sha256ScheduleUpdate1 => {
                Emit::emit(&38u16, writer)?;
                Ok(())
            }
            Self::Sm3PartW1 => {
                Emit::emit(&39u16, writer)?;
                Ok(())
            }
            Self::Sm3PartW2 => {
                Emit::emit(&40u16, writer)?;
                Ok(())
            }
            Self::Sm3SS1 => {
                Emit::emit(&41u16, writer)?;
                Ok(())
            }
            Self::Sm3TT1A => {
                Emit::emit(&42u16, writer)?;
                Ok(())
            }
            Self::Sm3TT1B => {
                Emit::emit(&43u16, writer)?;
                Ok(())
            }
            Self::Sm3TT2A => {
                Emit::emit(&44u16, writer)?;
                Ok(())
            }
            Self::Sm3TT2B => {
                Emit::emit(&45u16, writer)?;
                Ok(())
            }
            Self::AesExpand128 => {
                Emit::emit(&46u16, writer)?;
                Ok(())
            }
            Self::AesExpand256 => {
                Emit::emit(&47u16, writer)?;
                Ok(())
            }
            Self::GhashMultiply => {
                Emit::emit(&48u16, writer)?;
                Ok(())
            }
            Self::GhashMultiplyAdd => {
                Emit::emit(&49u16, writer)?;
                Ok(())
            }
            Self::Sha2Schedule4 => {
                Emit::emit(&50u16, writer)?;
                Ok(())
            }
            Self::Sha2Compress2Low => {
                Emit::emit(&51u16, writer)?;
                Ok(())
            }
            Self::Sha2Compress2High => {
                Emit::emit(&52u16, writer)?;
                Ok(())
            }
            Self::Sm3Expand8 => {
                Emit::emit(&53u16, writer)?;
                Ok(())
            }
            Self::Sm3Compress2 => {
                Emit::emit(&54u16, writer)?;
                Ok(())
            }
            Self::Sm4Expand4 => {
                Emit::emit(&55u16, writer)?;
                Ok(())
            }
            Self::AesDecryptKeyMix => {
                Emit::emit(&56u16, writer)?;
                Ok(())
            }
            Self::AesEncryptRoundMix => {
                Emit::emit(&57u16, writer)?;
                Ok(())
            }
            Self::AesDecryptRoundInvMix => {
                Emit::emit(&58u16, writer)?;
                Ok(())
            }
            Self::Sha512Hash => {
                Emit::emit(&59u16, writer)?;
                Ok(())
            }
            Self::Sha512Hash2 => {
                Emit::emit(&60u16, writer)?;
                Ok(())
            }
            Self::Sha512ScheduleUpdate0 => {
                Emit::emit(&61u16, writer)?;
                Ok(())
            }
            Self::Sha512ScheduleUpdate1 => {
                Emit::emit(&62u16, writer)?;
                Ok(())
            }
            Self::AesEncryptKeyLockerWide => {
                Emit::emit(&63u16, writer)?;
                Ok(())
            }
            Self::AesDecryptKeyLockerWide => {
                Emit::emit(&64u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorElement {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            kind,
            bits,
            scalar,
            signed,
            rounding,
        } = self;
        Emit::emit(kind, writer)?;
        Emit::emit(bits, writer)?;
        Emit::emit(scalar, writer)?;
        Emit::emit(signed, writer)?;
        Emit::emit(rounding, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorElementCountData {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            element_bits,
            multiplier,
            pattern,
            streaming,
            outputs,
        } = self;
        Emit::emit(element_bits, writer)?;
        Emit::emit(multiplier, writer)?;
        Emit::emit(pattern, writer)?;
        Emit::emit(streaming, writer)?;
        Emit::emit(outputs, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorElementKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Integer => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Float => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Unknown => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::BFloat16 => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::FloatOpen => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorExtendInLaneData {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            signed,
            source_bits,
            element_bits,
            outputs,
            inputs,
        } = self;
        Emit::emit(signed, writer)?;
        Emit::emit(source_bits, writer)?;
        Emit::emit(element_bits, writer)?;
        Emit::emit(outputs, writer)?;
        Emit::emit(inputs, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorFaultMode {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Normal => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::FirstFault => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::FaultOnlyFirst => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::NoFault => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorMaskBinaryKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::And => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Or => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Xor => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::AndNot => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::OrNot => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Nand => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Nor => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Xnor => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorMaskMode {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Merge => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Zero => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorPackKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Compress => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Expand => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorShuffleSelector {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Mask(mask) => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(mask, writer)
            }
            Self::Index(index) => {
                Emit::emit(&1u16, writer)?;
                Emit::emit(index, writer)
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorPredicateWhileData {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            kind,
            unsigned,
            elements,
            outputs,
            inputs,
        } = self;
        Emit::emit(kind, writer)?;
        Emit::emit(unsigned, writer)?;
        Emit::emit(elements, writer)?;
        Emit::emit(outputs, writer)?;
        Emit::emit(inputs, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorReduceKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Add => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Mul => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::And => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Or => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Xor => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Min => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Max => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Any => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::All => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::MinNum => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::MaxNum => {
                Emit::emit(&10u16, writer)?;
                Ok(())
            }
            Self::Difference => {
                Emit::emit(&11u16, writer)?;
                Ok(())
            }
            Self::MinimumNumber => {
                Emit::emit(&12u16, writer)?;
                Ok(())
            }
            Self::MaximumNumber => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::MinimumMagnitude => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::MaximumMagnitude => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::MinimumMagnitudeNumber => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::MaximumMagnitudeNumber => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for ReduceGroups {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self { lanes, bits } = self;
        Emit::emit(lanes, writer)?;
        Emit::emit(bits, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorSegmentLayout {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Interleaved => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Consecutive => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorStructLoadReplicateData {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            count,
            element_bits,
            outputs,
            inputs,
        } = self;
        Emit::emit(count, writer)?;
        Emit::emit(element_bits, writer)?;
        Emit::emit(outputs, writer)?;
        Emit::emit(inputs, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorSveComputeData {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            op,
            element_bits,
            rotation,
            outputs,
            inputs,
        } = self;
        Emit::emit(op, writer)?;
        Emit::emit(element_bits, writer)?;
        Emit::emit(rotation, writer)?;
        Emit::emit(outputs, writer)?;
        Emit::emit(inputs, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for VectorTernaryKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Fma => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Select => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::FunnelLeft => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::FunnelRight => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::MulAddHighDoubleSatRound => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::MulSubHighDoubleSatRound => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::FixupNan => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Fms => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::Fnma => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::Fnms => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorUnaryKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Neg => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Not => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Popcount => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Abs => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::HorizontalAdd => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
            Self::Sqrt => {
                Emit::emit(&5u16, writer)?;
                Ok(())
            }
            Self::Round => {
                Emit::emit(&6u16, writer)?;
                Ok(())
            }
            Self::Reciprocal => {
                Emit::emit(&7u16, writer)?;
                Ok(())
            }
            Self::ReciprocalSqrt => {
                Emit::emit(&8u16, writer)?;
                Ok(())
            }
            Self::GetExponent => {
                Emit::emit(&9u16, writer)?;
                Ok(())
            }
            Self::Identity => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::LeadingZeros => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::Fraction => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::Conflict => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::Exp2 => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::Log2 => {
                Emit::emit(&18u16, writer)?;
                Ok(())
            }
            Self::LeadingSignBits => {
                Emit::emit(&19u16, writer)?;
                Ok(())
            }
            Self::LeadingOnes => {
                Emit::emit(&20u16, writer)?;
                Ok(())
            }
            Self::Truncate => {
                Emit::emit(&21u16, writer)?;
                Ok(())
            }
            Self::Floor => {
                Emit::emit(&22u16, writer)?;
                Ok(())
            }
            Self::Ceil => {
                Emit::emit(&23u16, writer)?;
                Ok(())
            }
            Self::RoundNearestEven => {
                Emit::emit(&24u16, writer)?;
                Ok(())
            }
            Self::RoundNearestAway => {
                Emit::emit(&25u16, writer)?;
                Ok(())
            }
            Self::Sign => {
                Emit::emit(&26u16, writer)?;
                Ok(())
            }
            Self::BitReverse => {
                Emit::emit(&27u16, writer)?;
                Ok(())
            }
            Self::TrailingZeros => {
                Emit::emit(&28u16, writer)?;
                Ok(())
            }
            Self::ByteSwap => {
                Emit::emit(&29u16, writer)?;
                Ok(())
            }
            Self::BitReverseIn8 => {
                Emit::emit(&30u16, writer)?;
                Ok(())
            }
            Self::RoundExact => Emit::emit(&31u16, writer),
            Self::RoundInt32 => Emit::emit(&32u16, writer),
            Self::RoundInt64 => Emit::emit(&33u16, writer),
            Self::NegArithmetic => Emit::emit(&34u16, writer),
            Self::AbsArithmetic => Emit::emit(&35u16, writer),
        }
    }
}

impl<T: Target> Emit<T> for ControlledKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::RoundScale => Emit::emit(&0u16, writer),
            Self::Reduce => Emit::emit(&1u16, writer),
            Self::GetMantissa => Emit::emit(&2u16, writer),
            Self::Range => Emit::emit(&3u16, writer),
            Self::Fixup => Emit::emit(&4u16, writer),
            Self::Classify => Emit::emit(&5u16, writer),
            Self::RoundFixedPoint => Emit::emit(&6u16, writer),
        }
    }
}

impl<T: Target> Emit<T> for FloatDotRule {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Sequential => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Fpcr => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::DotAdd => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Fp8 => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for DotShape {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            source_bits,
            dest_bits,
            signed_a,
            signed_b,
            accumulates,
            saturates,
        } = self;
        Emit::emit(source_bits, writer)?;
        Emit::emit(dest_bits, writer)?;
        Emit::emit(signed_a, writer)?;
        Emit::emit(signed_b, writer)?;
        Emit::emit(accumulates, writer)?;
        Emit::emit(saturates, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for FloatSource {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Format(format) => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(format, writer)?;
                Ok(())
            }
            Self::Fp8 => Emit::emit(&1u16, writer),
        }
    }
}

impl<T: Target> Emit<T> for FloatElements {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        // The group only where it is not the widths', which a container's
        // tag says, so the elements of every dot product the widths group
        // keep their fingerprint.
        let Self {
            source,
            dest,
            group,
        } = self;
        Emit::emit(source, writer)?;
        Emit::emit(dest, writer)?;
        if !self.group_is_the_widths() {
            Emit::emit(group, writer)?;
        }
        Ok(())
    }
}

impl<T: Target> Emit<T> for ContractElements {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Integer {
                source_bits,
                dest_bits,
                signed_a,
                signed_b,
            } => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(source_bits, writer)?;
                Emit::emit(dest_bits, writer)?;
                Emit::emit(signed_a, writer)?;
                Emit::emit(signed_b, writer)?;
                Ok(())
            }
            // Tag 3, not the 1 the stated-format case had nor the 2 of the
            // FP8 case it subsumes: a tag states one shape for good, so a
            // stream holding either of the old ones fails to decode rather
            // than decoding as this.
            // Tag 4 where the elements state their group.
            Self::Float(elements) => {
                let tag: u16 = if elements.group_is_the_widths() { 3 } else { 4 };
                Emit::emit(&tag, writer)?;
                Emit::emit(elements, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for MatrixLayout {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Segments { segment_bits } => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(segment_bits, writer)?;
                Ok(())
            }
            Self::Tiles => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for ComplexPart {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Real => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Imaginary => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for MatrixShape {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            elements,
            layout,
            complex,
        } = self;
        Emit::emit(elements, writer)?;
        Emit::emit(layout, writer)?;
        Emit::emit(complex, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for OuterShape {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            elements,
            subtract,
            product,
            governed,
            selection,
        } = self;
        Emit::emit(elements, writer)?;
        Emit::emit(subtract, writer)?;
        Emit::emit(product, writer)?;
        // A governed dense product emits what it always did, so nothing the
        // sparse forms add moves an existing content hash.
        if !governed {
            Emit::emit(&2u16, writer)?;
        }
        if let Some(OuterSelection {
            values,
            candidates,
            per_value,
        }) = selection
        {
            Emit::emit(&1u16, writer)?;
            Emit::emit(values, writer)?;
            Emit::emit(candidates, writer)?;
            Emit::emit(per_value, writer)?;
        }
        Ok(())
    }
}

impl<T: Target> Emit<T> for OuterProduct {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Multiply => Emit::emit(&0u16, writer)?,
            Self::MatchingBits => Emit::emit(&1u16, writer)?,
        }
        Ok(())
    }
}

impl<T: Target> Emit<T> for ComplexRotation {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Quarter => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::ThreeQuarters => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for ComplexProduct {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Multiply {
                conjugate,
                accumulates,
            } => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(conjugate, writer)?;
                Emit::emit(accumulates, writer)?;
                Ok(())
            }
            Self::Rotate(field_0) => {
                Emit::emit(&1u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for ComplexShape {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            product,
            element,
            saturates,
        } = self;
        Emit::emit(product, writer)?;
        Emit::emit(element, writer)?;
        Emit::emit(saturates, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for Contraction {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Dot(field_0) => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }

            Self::MultiplyAdd52 { high } => {
                Emit::emit(&2u16, writer)?;
                Emit::emit(high, writer)?;
                Ok(())
            }
            Self::MaskedDot {
                control,
                element_bits,
            } => {
                Emit::emit(&3u16, writer)?;
                Emit::emit(control, writer)?;
                Emit::emit(element_bits, writer)?;
                Ok(())
            }
            Self::Matrix(field_0) => {
                Emit::emit(&4u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Outer(field_0) => {
                Emit::emit(&5u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Complex(field_0) => {
                Emit::emit(&6u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            // Tag 7, not the 1 the bfloat16 dot it generalises had: a tag
            // states one shape for good, so a stream holding the old one
            // fails to decode rather than decoding as this.
            // Tag 8 where the elements state their group.
            Self::FloatDot { elements, rule } => {
                let tag: u16 = if elements.group_is_the_widths() { 7 } else { 8 };
                Emit::emit(&tag, writer)?;
                Emit::emit(elements, writer)?;
                Emit::emit(rule, writer)?;
                Ok(())
            }
        }
    }
}
