//! The content encoding of each execution type ([`Emit`]): a tag for its
//! variant, then its fields in declaration order.

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{
        content::ContentSink,
        ops::execution::{
            BitfieldPosition, Computation, Kernel, LaneControl, VectorFixedKind,
            VectorInactivePolicy, VectorMaskPrefix, VectorPredicateLayout, VectorRounding,
            VectorStatus,
        },
    },
    target::Target,
};

impl<T: Target> Emit<T> for VectorFixedKind {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::AverageAdd => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::AverageSub => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::FractionalMultiply => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::Shift => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::Clip => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorInactivePolicy {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Undisturbed => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Zero => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Agnostic => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorMaskPrefix {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Before => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Including => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Only => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorPredicateLayout {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Packed => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::ByteGranular => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorRounding {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::None => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::TowardZero => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::ToOdd => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
            Self::FloatingEnvironment => {
                Emit::emit(&3u16, writer)?;
                Ok(())
            }
            Self::FixedEnvironment => {
                Emit::emit(&4u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for VectorStatus {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::None => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::FloatingExceptions => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
            Self::Saturation => {
                Emit::emit(&2u16, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for Kernel {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Move => {
                Emit::emit(&0u16, writer)?;
                Ok(())
            }
            Self::Unary(field0) => {
                Emit::emit(&1u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::Binary(field0) => {
                Emit::emit(&2u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::IntegerDivide {
                remainder,
                unsigned,
            } => {
                Emit::emit(&3u16, writer)?;
                Emit::emit(remainder, writer)?;
                Emit::emit(unsigned, writer)?;
                Ok(())
            }
            Self::ModuloShift { kind } => {
                Emit::emit(&4u16, writer)?;
                Emit::emit(kind, writer)?;
                Ok(())
            }
            Self::MultiplyHigh {
                signed_left,
                signed_right,
            } => {
                Emit::emit(&5u16, writer)?;
                Emit::emit(signed_left, writer)?;
                Emit::emit(signed_right, writer)?;
                Ok(())
            }
            Self::Compare {
                relation,
                signaling,
            } => {
                Emit::emit(&6u16, writer)?;
                Emit::emit(relation, writer)?;
                Emit::emit(signaling, writer)?;
                Ok(())
            }
            Self::Cast { from, to } => {
                Emit::emit(&7u16, writer)?;
                Emit::emit(from, writer)?;
                Emit::emit(to, writer)?;
                Ok(())
            }
            Self::Extend { signed } => {
                Emit::emit(&8u16, writer)?;
                Emit::emit(signed, writer)?;
                Ok(())
            }
            Self::WidenBinary {
                wide_left,
                kind,
                signed_left,
                signed_right,
            } => {
                Emit::emit(&9u16, writer)?;
                Emit::emit(wide_left, writer)?;
                Emit::emit(kind, writer)?;
                Emit::emit(signed_left, writer)?;
                Emit::emit(signed_right, writer)?;
                Ok(())
            }
            Self::FusedMultiplyAdd {
                negate_product,
                negate_addend,
            } => {
                Emit::emit(&10u16, writer)?;
                Emit::emit(negate_product, writer)?;
                Emit::emit(negate_addend, writer)?;
                Ok(())
            }
            Self::IntegerMultiplyAdd { subtract_product } => {
                Emit::emit(&11u16, writer)?;
                Emit::emit(subtract_product, writer)?;
                Ok(())
            }
            Self::Reduce { kind, ordered } => {
                Emit::emit(&12u16, writer)?;
                Emit::emit(kind, writer)?;
                Emit::emit(ordered, writer)?;
                Ok(())
            }
            Self::Gather => {
                Emit::emit(&13u16, writer)?;
                Ok(())
            }
            Self::Slide { up, scalar_fill } => {
                Emit::emit(&14u16, writer)?;
                Emit::emit(up, writer)?;
                Emit::emit(scalar_fill, writer)?;
                Ok(())
            }
            Self::Index => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::Memory {
                load,
                access_bits,
                sign_extend,
                byte_stream,
                coalescible,
                indexed,
                ordered,
                fields,
                fault,
            } => {
                Emit::emit(&16u16, writer)?;
                Emit::emit(load, writer)?;
                Emit::emit(access_bits, writer)?;
                Emit::emit(sign_extend, writer)?;
                Emit::emit(byte_stream, writer)?;
                Emit::emit(coalescible, writer)?;
                Emit::emit(indexed, writer)?;
                Emit::emit(ordered, writer)?;
                Emit::emit(fields, writer)?;
                Emit::emit(fault, writer)?;
                Ok(())
            }
            Self::InsertScalar => {
                Emit::emit(&17u16, writer)?;
                Ok(())
            }
            Self::NarrowShift { unsigned } => {
                Emit::emit(&18u16, writer)?;
                Emit::emit(unsigned, writer)?;
                Ok(())
            }
            Self::Fixed { kind } => {
                Emit::emit(&19u16, writer)?;
                Emit::emit(kind, writer)?;
                Ok(())
            }
            Self::WidenMultiplyAdd {
                signed_left,
                signed_right,
            } => {
                Emit::emit(&20u16, writer)?;
                Emit::emit(signed_left, writer)?;
                Emit::emit(signed_right, writer)?;
                Ok(())
            }
            Self::MaskBinary(field0) => {
                Emit::emit(&21u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::IntegerCarry {
                subtract,
                mask_result,
            } => {
                Emit::emit(&22u16, writer)?;
                Emit::emit(subtract, writer)?;
                Emit::emit(mask_result, writer)?;
                Ok(())
            }
            Self::Select => {
                Emit::emit(&23u16, writer)?;
                Ok(())
            }
            Self::MaskPrefix(field0) => {
                Emit::emit(&24u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::MaskIota => {
                Emit::emit(&25u16, writer)?;
                Ok(())
            }
            Self::MaskSummary { first } => {
                Emit::emit(&26u16, writer)?;
                Emit::emit(first, writer)?;
                Ok(())
            }
            Self::ReduceWide { kind, ordered } => {
                Emit::emit(&27u16, writer)?;
                Emit::emit(kind, writer)?;
                Emit::emit(ordered, writer)?;
                Ok(())
            }
            Self::Compress => {
                Emit::emit(&28u16, writer)?;
                Ok(())
            }
            Self::SignInject { negate, xor } => {
                Emit::emit(&29u16, writer)?;
                Emit::emit(negate, writer)?;
                Emit::emit(xor, writer)?;
                Ok(())
            }
            Self::FloatClassify => {
                Emit::emit(&30u16, writer)?;
                Ok(())
            }
            Self::FloatToInteger {
                unsigned,
                nan_maximum,
            } => {
                Emit::emit(&31u16, writer)?;
                Emit::emit(unsigned, writer)?;
                Emit::emit(nan_maximum, writer)?;
                Ok(())
            }
            Self::Estimate7 { square_root } => {
                Emit::emit(&32u16, writer)?;
                Emit::emit(square_root, writer)?;
                Ok(())
            }
            Self::CarrylessMultiply { part } => {
                Emit::emit(&33u16, writer)?;
                Emit::emit(part, writer)?;
                Ok(())
            }
            Self::Crypto(field0) => {
                Emit::emit(&34u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::BitDeposit => {
                Emit::emit(&35u16, writer)?;
                Ok(())
            }
            Self::BitExtract => {
                Emit::emit(&36u16, writer)?;
                Ok(())
            }
            Self::BitGroup => {
                Emit::emit(&37u16, writer)?;
                Ok(())
            }
            Self::Checksum(field0) => {
                Emit::emit(&38u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::Random { from_entropy } => {
                Emit::emit(&39u16, writer)?;
                Emit::emit(from_entropy, writer)?;
                Ok(())
            }
            Self::PointerAuth(field0) => {
                Emit::emit(&40u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::ScalarCipher(field0) => {
                Emit::emit(&41u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::FloatToIntModular => {
                Emit::emit(&42u16, writer)?;
                Ok(())
            }
            Self::MemoryTag(field0) => {
                Emit::emit(&43u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::VectorShapeSupported => {
                Emit::emit(&44u16, writer)?;
                Ok(())
            }
            Self::VectorLengthChoice => {
                Emit::emit(&45u16, writer)?;
                Ok(())
            }
            Self::VectorFloatSupported => {
                Emit::emit(&46u16, writer)?;
                Ok(())
            }
            Self::VectorCryptoSupported => {
                Emit::emit(&47u16, writer)?;
                Ok(())
            }
            Self::DecimalAdjust { kind, base } => {
                Emit::emit(&48u16, writer)?;
                Emit::emit(kind, writer)?;
                Emit::emit(base, writer)?;
                Ok(())
            }
            Self::FloatHelper(field0) => {
                Emit::emit(&49u16, writer)?;
                Emit::emit(field0, writer)?;
                Ok(())
            }
            Self::TernaryLogic { table } => {
                Emit::emit(&50u16, writer)?;
                Emit::emit(table, writer)?;
                Ok(())
            }
            Self::MultiSad { offsets } => {
                Emit::emit(&51u16, writer)?;
                Emit::emit(offsets, writer)?;
                Ok(())
            }
            Self::DoubleBlockSad { selector } => {
                Emit::emit(&52u16, writer)?;
                Emit::emit(selector, writer)?;
                Ok(())
            }
            Self::StringCompare {
                control,
                explicit_length,
                result_index,
            } => {
                Emit::emit(&53u16, writer)?;
                Emit::emit(control, writer)?;
                Emit::emit(explicit_length, writer)?;
                Emit::emit(result_index, writer)?;
                Ok(())
            }
            Self::Bitfield { insert, position } => {
                Emit::emit(&54u16, writer)?;
                Emit::emit(insert, writer)?;
                Emit::emit(position, writer)?;
                Ok(())
            }
            Self::Intersect => {
                Emit::emit(&55u16, writer)?;
                Ok(())
            }
            Self::ShuffleBits => {
                Emit::emit(&56u16, writer)?;
                Ok(())
            }
            Self::HorizontalMinPos => {
                Emit::emit(&57u16, writer)?;
                Ok(())
            }
            Self::FloatControlled { kind, control } => {
                Emit::emit(&58u16, writer)?;
                Emit::emit(kind, writer)?;
                Emit::emit(control, writer)?;
                Ok(())
            }
            Self::CounterPredicate { part } => {
                Emit::emit(&59u16, writer)?;
                Emit::emit(part, writer)?;
                Ok(())
            }
            Self::PredicateCounter { element_bits } => {
                Emit::emit(&60u16, writer)?;
                Emit::emit(element_bits, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for Computation {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            kernel,
            element,
            predicate,
            rounding,
            status,
            canonical_nan,
            lanes,
        } = self;
        Emit::emit(kernel, writer)?;
        Emit::emit(element, writer)?;
        Emit::emit(predicate, writer)?;
        Emit::emit(rounding, writer)?;
        Emit::emit(status, writer)?;
        Emit::emit(canonical_nan, writer)?;
        Emit::emit(lanes, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for LaneControl {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            mask_tail,
            tail,
            inactive,
            no_passthrough,
        } = self;
        Emit::emit(mask_tail, writer)?;
        Emit::emit(tail, writer)?;
        Emit::emit(inactive, writer)?;
        Emit::emit(no_passthrough, writer)?;
        Ok(())
    }
}

impl<T: Target> Emit<T> for BitfieldPosition {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::Immediate { index, length } => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(index, writer)?;
                Emit::emit(length, writer)?;
                Ok(())
            }
            Self::Register => {
                Emit::emit(&1u16, writer)?;
                Ok(())
            }
        }
    }
}
