//! The content encoding of an operation ([`Emit`]): a tag for its variant,
//! then its fields in declaration order.

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{content::ContentSink, ops::def::SsaOp},
    metadata::MetadataRef,
    target::Target,
};

impl<T: Target> Emit<T> for SsaOp<T> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let mut current = self;
        loop {
            return match current {
                Self::Const { dest, value } => {
                    Emit::emit(&0u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::Add {
                    dest,
                    left,
                    right,
                    flags,
                } => {
                    Emit::emit(&1u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::AddOvf {
                    dest,
                    left,
                    right,
                    unsigned,
                    flags,
                } => {
                    Emit::emit(&2u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::Sub {
                    dest,
                    left,
                    right,
                    flags,
                } => {
                    Emit::emit(&3u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::SubOvf {
                    dest,
                    left,
                    right,
                    unsigned,
                    flags,
                } => {
                    Emit::emit(&4u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::Mul {
                    dest,
                    left,
                    right,
                    flags,
                } => {
                    Emit::emit(&5u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::MulOvf {
                    dest,
                    left,
                    right,
                    unsigned,
                    flags,
                } => {
                    Emit::emit(&6u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::WideMul {
                    low,
                    high,
                    left,
                    right,
                    unsigned,
                } => {
                    Emit::emit(&7u16, writer)?;
                    Emit::emit(low, writer)?;
                    Emit::emit(high, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Ok(())
                }
                Self::Div {
                    dest,
                    left,
                    right,
                    unsigned,
                    flags,
                } => {
                    Emit::emit(&8u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::Rem {
                    dest,
                    left,
                    right,
                    unsigned,
                    flags,
                } => {
                    Emit::emit(&9u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::FloatCompareFlags {
                    flags,
                    left,
                    right,
                    signaling,
                } => {
                    Emit::emit(&10u16, writer)?;
                    Emit::emit(flags, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(signaling, writer)?;
                    Ok(())
                }
                Self::FloatRaise {
                    dest,
                    op,
                    lanes,
                    raises,
                    layout,
                } => {
                    Emit::emit(&208u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(lanes, writer)?;
                    Emit::emit(raises, writer)?;
                    Emit::emit(layout, writer)?;
                    current = op;
                    continue;
                }
                Self::WideDiv {
                    quotient,
                    remainder,
                    high,
                    low,
                    divisor,
                    unsigned,
                } => {
                    Emit::emit(&11u16, writer)?;
                    Emit::emit(quotient, writer)?;
                    Emit::emit(remainder, writer)?;
                    Emit::emit(high, writer)?;
                    Emit::emit(low, writer)?;
                    Emit::emit(divisor, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Ok(())
                }
                Self::Neg {
                    dest,
                    operand,
                    flags,
                } => {
                    Emit::emit(&12u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::And {
                    dest,
                    left,
                    right,
                    flags,
                } => {
                    Emit::emit(&13u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::Or {
                    dest,
                    left,
                    right,
                    flags,
                } => {
                    Emit::emit(&14u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::Xor {
                    dest,
                    left,
                    right,
                    flags,
                } => {
                    Emit::emit(&15u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::Not {
                    dest,
                    operand,
                    flags,
                } => {
                    Emit::emit(&16u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::Shl {
                    dest,
                    value,
                    amount,
                    flags,
                } => {
                    Emit::emit(&17u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(amount, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::Shr {
                    dest,
                    value,
                    amount,
                    unsigned,
                    flags,
                } => {
                    Emit::emit(&18u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(amount, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(flags, writer)?;
                    Ok(())
                }
                Self::Rol {
                    dest,
                    value,
                    amount,
                } => {
                    Emit::emit(&19u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(amount, writer)?;
                    Ok(())
                }
                Self::Ror {
                    dest,
                    value,
                    amount,
                } => {
                    Emit::emit(&20u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(amount, writer)?;
                    Ok(())
                }
                Self::BSwap { dest, src } => {
                    Emit::emit(&21u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(src, writer)?;
                    Ok(())
                }
                Self::BRev { dest, src } => {
                    Emit::emit(&22u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(src, writer)?;
                    Ok(())
                }
                Self::BitScanForward { dest, src } => {
                    Emit::emit(&23u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(src, writer)?;
                    Ok(())
                }
                Self::BitScanReverse { dest, src } => {
                    Emit::emit(&24u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(src, writer)?;
                    Ok(())
                }
                Self::Popcount { dest, src } => {
                    Emit::emit(&25u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(src, writer)?;
                    Ok(())
                }
                Self::Parity { dest, src } => {
                    Emit::emit(&26u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(src, writer)?;
                    Ok(())
                }
                Self::Ceq {
                    dest,
                    left,
                    right,
                    signaling,
                } => {
                    Emit::emit(&27u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(signaling, writer)?;
                    Ok(())
                }
                Self::Clt {
                    dest,
                    left,
                    right,
                    unsigned,
                    signaling,
                } => {
                    Emit::emit(&28u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(signaling, writer)?;
                    Ok(())
                }
                Self::Cgt {
                    dest,
                    left,
                    right,
                    unsigned,
                    signaling,
                } => {
                    Emit::emit(&29u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(signaling, writer)?;
                    Ok(())
                }
                Self::BoolAnd { dest, left, right } => {
                    Emit::emit(&30u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Ok(())
                }
                Self::BoolOr { dest, left, right } => {
                    Emit::emit(&31u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Ok(())
                }
                Self::BoolXor { dest, left, right } => {
                    Emit::emit(&32u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Ok(())
                }
                Self::BoolNot { dest, value } => {
                    Emit::emit(&33u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::IntConv {
                    dest,
                    operand,
                    target,
                    overflow_check,
                    unsigned,
                } => {
                    Emit::emit(&34u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    writer.metadata(MetadataRef::Type(target))?;
                    Emit::emit(overflow_check, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Ok(())
                }
                Self::IntToPtr {
                    dest,
                    operand,
                    target,
                } => {
                    Emit::emit(&35u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    writer.metadata(MetadataRef::Type(target))?;
                    Ok(())
                }
                Self::PtrToInt {
                    dest,
                    operand,
                    target,
                } => {
                    Emit::emit(&36u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    writer.metadata(MetadataRef::Type(target))?;
                    Ok(())
                }
                Self::IntToFloat {
                    dest,
                    operand,
                    target,
                    unsigned,
                    rounding,
                    environment,
                } => {
                    Emit::emit(&37u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    writer.metadata(MetadataRef::Type(target))?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(rounding, writer)?;
                    Emit::emit(environment, writer)?;
                    Ok(())
                }
                Self::FloatToInt {
                    dest,
                    operand,
                    target,
                    overflow_check,
                    unsigned,
                    source,
                } => {
                    Emit::emit(&38u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    writer.metadata(MetadataRef::Type(target))?;
                    Emit::emit(overflow_check, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(source, writer)?;
                    Ok(())
                }
                Self::FloatConv {
                    dest,
                    operand,
                    target,
                    rounding,
                    source,
                    environment,
                } => {
                    Emit::emit(&39u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    writer.metadata(MetadataRef::Type(target))?;
                    Emit::emit(rounding, writer)?;
                    Emit::emit(source, writer)?;
                    Emit::emit(environment, writer)?;
                    Ok(())
                }
                Self::Bitcast {
                    dest,
                    operand,
                    target,
                } => {
                    Emit::emit(&40u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    writer.metadata(MetadataRef::Type(target))?;
                    Ok(())
                }
                Self::PtrAdd {
                    dest,
                    base,
                    index,
                    stride,
                    offset,
                    result_type,
                } => {
                    Emit::emit(&41u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(base, writer)?;
                    Emit::emit(index, writer)?;
                    Emit::emit(stride, writer)?;
                    Emit::emit(offset, writer)?;
                    writer.metadata(MetadataRef::Type(result_type))?;
                    Ok(())
                }
                Self::Select {
                    dest,
                    condition,
                    true_val,
                    false_val,
                } => {
                    Emit::emit(&42u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(condition, writer)?;
                    Emit::emit(true_val, writer)?;
                    Emit::emit(false_val, writer)?;
                    Ok(())
                }
                Self::ReadFlags { dest, flags, mask } => {
                    Emit::emit(&43u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(flags, writer)?;
                    Emit::emit(mask, writer)?;
                    Ok(())
                }
                Self::MergeFlags {
                    dest,
                    base,
                    update,
                    mask,
                } => {
                    Emit::emit(&44u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(base, writer)?;
                    Emit::emit(update, writer)?;
                    Emit::emit(mask, writer)?;
                    Ok(())
                }
                Self::InsertFlag {
                    dest,
                    base,
                    bit,
                    mask,
                } => {
                    Emit::emit(&45u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(base, writer)?;
                    Emit::emit(bit, writer)?;
                    Emit::emit(mask, writer)?;
                    Ok(())
                }
                Self::VectorUnary {
                    dest,
                    value,
                    kind,
                    element,
                    environment,
                } => {
                    Emit::emit(&46u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(element, writer)?;
                    Emit::emit(environment, writer)?;
                    Ok(())
                }
                Self::VectorBinary {
                    dest,
                    left,
                    right,
                    kind,
                    element,
                    environment,
                } => {
                    Emit::emit(&47u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(element, writer)?;
                    Emit::emit(environment, writer)?;
                    Ok(())
                }
                Self::VectorTernary {
                    dest,
                    first,
                    second,
                    third,
                    kind,
                    element,
                    environment,
                } => {
                    Emit::emit(&48u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(first, writer)?;
                    Emit::emit(second, writer)?;
                    Emit::emit(third, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(element, writer)?;
                    Emit::emit(environment, writer)?;
                    Ok(())
                }
                Self::VectorPredicatedUnary {
                    dest,
                    value,
                    mask,
                    passthrough,
                    kind,
                    element,
                    mode,
                    environment,
                } => {
                    Emit::emit(&49u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(mask, writer)?;
                    Emit::emit(passthrough, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(element, writer)?;
                    Emit::emit(mode, writer)?;
                    Emit::emit(environment, writer)?;
                    Ok(())
                }
                Self::VectorPredicatedBinary {
                    dest,
                    left,
                    right,
                    mask,
                    passthrough,
                    kind,
                    element,
                    mode,
                    environment,
                } => {
                    Emit::emit(&50u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(mask, writer)?;
                    Emit::emit(passthrough, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(element, writer)?;
                    Emit::emit(mode, writer)?;
                    Emit::emit(environment, writer)?;
                    Ok(())
                }
                Self::VectorPredicatedTernary {
                    dest,
                    first,
                    second,
                    third,
                    mask,
                    passthrough,
                    kind,
                    element,
                    mode,
                    environment,
                } => {
                    Emit::emit(&51u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(first, writer)?;
                    Emit::emit(second, writer)?;
                    Emit::emit(third, writer)?;
                    Emit::emit(mask, writer)?;
                    Emit::emit(passthrough, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(element, writer)?;
                    Emit::emit(mode, writer)?;
                    Emit::emit(environment, writer)?;
                    Ok(())
                }
                Self::VectorCompare {
                    dest,
                    left,
                    right,
                    kind,
                    element,
                    signaling,
                } => {
                    Emit::emit(&52u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(element, writer)?;
                    Emit::emit(signaling, writer)?;
                    Ok(())
                }
                Self::VectorLoad {
                    dest,
                    addr,
                    vector_type,
                    access,
                } => {
                    Emit::emit(&53u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(addr, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorStore {
                    addr,
                    value,
                    vector_type,
                    access,
                } => {
                    Emit::emit(&54u16, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorMaskedLoad {
                    dest,
                    addr,
                    mask,
                    passthrough,
                    vector_type,
                    mode,
                    access,
                } => {
                    Emit::emit(&55u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(mask, writer)?;
                    Emit::emit(passthrough, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(mode, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorMaskedStore {
                    addr,
                    value,
                    mask,
                    vector_type,
                    access,
                } => {
                    Emit::emit(&56u16, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(mask, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorBroadcastLoad {
                    dest,
                    addr,
                    vector_type,
                    access,
                } => {
                    Emit::emit(&57u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(addr, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorGather {
                    dest,
                    base,
                    indices,
                    mask,
                    passthrough,
                    vector_type,
                    mode,
                    access,
                } => {
                    Emit::emit(&58u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(base, writer)?;
                    Emit::emit(indices, writer)?;
                    Emit::emit(mask, writer)?;
                    Emit::emit(passthrough, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(mode, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorFaultingLoad {
                    dest,
                    fault,
                    addr,
                    mask,
                    passthrough,
                    vector_type,
                    fault_mode,
                    mask_mode,
                    access,
                } => {
                    Emit::emit(&59u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(fault, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(mask, writer)?;
                    Emit::emit(passthrough, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(fault_mode, writer)?;
                    Emit::emit(mask_mode, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorSegmentLoad {
                    dests,
                    base,
                    mask,
                    vector_type,
                    segments,
                    layout,
                    access,
                } => {
                    Emit::emit(&60u16, writer)?;
                    Emit::emit(dests, writer)?;
                    Emit::emit(base, writer)?;
                    Emit::emit(mask, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(segments, writer)?;
                    Emit::emit(layout, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorScatter {
                    base,
                    indices,
                    value,
                    mask,
                    vector_type,
                    access,
                } => {
                    Emit::emit(&61u16, writer)?;
                    Emit::emit(base, writer)?;
                    Emit::emit(indices, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(mask, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorSegmentStore {
                    base,
                    values,
                    mask,
                    vector_type,
                    segments,
                    layout,
                    access,
                } => {
                    Emit::emit(&62u16, writer)?;
                    Emit::emit(base, writer)?;
                    Emit::emit(values, writer)?;
                    Emit::emit(mask, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(segments, writer)?;
                    Emit::emit(layout, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorExtract { dest, vector, lane } => {
                    Emit::emit(&63u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(vector, writer)?;
                    Emit::emit(lane, writer)?;
                    Ok(())
                }
                Self::VectorInsert {
                    dest,
                    vector,
                    lane,
                    value,
                } => {
                    Emit::emit(&64u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(vector, writer)?;
                    Emit::emit(lane, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::VectorSplat {
                    dest,
                    value,
                    vector_type,
                } => {
                    Emit::emit(&65u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Ok(())
                }
                Self::VectorShuffle {
                    dest,
                    left,
                    right,
                    selector,
                } => {
                    Emit::emit(&66u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(selector, writer)?;
                    Ok(())
                }
                Self::VectorCast {
                    dest,
                    value,
                    target_type,
                    from,
                    to,
                    rounding,
                    saturate,
                    environment,
                    format,
                } => {
                    Emit::emit(&67u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::Type(target_type))?;
                    Emit::emit(from, writer)?;
                    Emit::emit(to, writer)?;
                    Emit::emit(rounding, writer)?;
                    Emit::emit(saturate, writer)?;
                    Emit::emit(environment, writer)?;
                    // Only where one is stated, so a cast naming every format
                    // hashes as it always has.
                    if let Some(format) = format {
                        Emit::emit(format, writer)?;
                    }
                    Ok(())
                }
                Self::VectorReinterpret {
                    dest,
                    value,
                    target_type,
                } => {
                    Emit::emit(&68u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::Type(target_type))?;
                    Ok(())
                }
                Self::VectorPack {
                    dest,
                    value,
                    mask,
                    passthrough,
                    vector_type,
                    element_bits,
                    kind,
                    mode,
                } => {
                    Emit::emit(&69u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(mask, writer)?;
                    Emit::emit(passthrough, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(element_bits, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(mode, writer)?;
                    Ok(())
                }
                Self::VectorPackLoad {
                    dest,
                    addr,
                    mask,
                    passthrough,
                    vector_type,
                    element_bits,
                    kind,
                    mode,
                    access,
                } => {
                    Emit::emit(&70u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(mask, writer)?;
                    Emit::emit(passthrough, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(element_bits, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(mode, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::VectorPackStore {
                    addr,
                    value,
                    mask,
                    vector_type,
                    element_bits,
                    kind,
                    access,
                } => {
                    Emit::emit(&71u16, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(mask, writer)?;
                    writer.metadata(MetadataRef::Type(vector_type))?;
                    Emit::emit(element_bits, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }

                Self::VectorReduce {
                    dest,
                    value,
                    kind,
                    element,
                    initial,
                    ordered,
                    groups,
                    environment,
                } => {
                    Emit::emit(&75u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(kind, writer)?;
                    Emit::emit(element, writer)?;
                    Emit::emit(initial, writer)?;
                    Emit::emit(ordered, writer)?;
                    Emit::emit(groups, writer)?;
                    Emit::emit(environment, writer)?;
                    Ok(())
                }
                Self::VectorBitmask { dest, value, kind } => {
                    Emit::emit(&76u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(kind, writer)?;
                    Ok(())
                }
                Self::Jump { target } => {
                    Emit::emit(&77u16, writer)?;
                    Emit::emit(target, writer)?;
                    Ok(())
                }
                Self::Branch {
                    condition,
                    true_target,
                    false_target,
                } => {
                    Emit::emit(&78u16, writer)?;
                    Emit::emit(condition, writer)?;
                    Emit::emit(true_target, writer)?;
                    Emit::emit(false_target, writer)?;
                    Ok(())
                }
                Self::BranchCmp {
                    left,
                    right,
                    cmp,
                    unsigned,
                    true_target,
                    false_target,
                } => {
                    Emit::emit(&79u16, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    Emit::emit(cmp, writer)?;
                    Emit::emit(unsigned, writer)?;
                    Emit::emit(true_target, writer)?;
                    Emit::emit(false_target, writer)?;
                    Ok(())
                }
                Self::BranchFlags {
                    flags,
                    condition,
                    true_target,
                    false_target,
                } => {
                    Emit::emit(&80u16, writer)?;
                    Emit::emit(flags, writer)?;
                    Emit::emit(condition, writer)?;
                    Emit::emit(true_target, writer)?;
                    Emit::emit(false_target, writer)?;
                    Ok(())
                }
                Self::Switch {
                    value,
                    targets,
                    default,
                } => {
                    Emit::emit(&81u16, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(targets, writer)?;
                    Emit::emit(default, writer)?;
                    Ok(())
                }
                Self::IndirectBranch {
                    target,
                    resolved_targets,
                    state,
                } => {
                    // A branch handing over state has its own tag, so a reader
                    // cannot take it for one handing over none.
                    let tag = if state.is_empty() { 82u16 } else { 215u16 };
                    Emit::emit(&tag, writer)?;
                    Emit::emit(target, writer)?;
                    Emit::emit(resolved_targets, writer)?;
                    if !state.is_empty() {
                        Emit::emit(state, writer)?;
                    }
                    Ok(())
                }
                Self::Return {
                    value,
                    state,
                    results,
                    transfer,
                } => {
                    // A return handing back values beside its own, after the
                    // rest of it under that tag.
                    if !results.is_empty() {
                        Emit::emit(&219u16, writer)?;
                        Emit::emit(results, writer)?;
                    }
                    // A return handing back registers has its own tag, so a
                    // reader cannot take it for one handing back none.
                    let target = transfer.as_ref().and_then(|transfer| transfer.target);
                    let handover = transfer
                        .as_ref()
                        .filter(|transfer| !transfer.handover.is_empty());
                    let tag = match (target.is_some(), handover) {
                        (false, None) => 83u16,
                        (true, None) => 214u16,
                        (_, Some(_)) => 216u16,
                    };
                    Emit::emit(&tag, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(state, writer)?;
                    if let Some(transfer) = handover {
                        Emit::emit(&target, writer)?;
                        Emit::emit(&transfer.handover, writer)?;
                    } else if let Some(target) = &target {
                        Emit::emit(target, writer)?;
                    }
                    Ok(())
                }
                Self::LoadField {
                    dest,
                    object,
                    field,
                    access,
                } => {
                    Emit::emit(&84u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(object, writer)?;
                    writer.metadata(MetadataRef::Field(field))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::StoreField {
                    object,
                    field,
                    value,
                    access,
                } => {
                    Emit::emit(&85u16, writer)?;
                    Emit::emit(object, writer)?;
                    writer.metadata(MetadataRef::Field(field))?;
                    Emit::emit(value, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::LoadStaticField {
                    dest,
                    field,
                    access,
                } => {
                    Emit::emit(&86u16, writer)?;
                    Emit::emit(dest, writer)?;
                    writer.metadata(MetadataRef::Field(field))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::StoreStaticField {
                    field,
                    value,
                    access,
                } => {
                    Emit::emit(&87u16, writer)?;
                    writer.metadata(MetadataRef::Field(field))?;
                    Emit::emit(value, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::LoadFieldAddr {
                    dest,
                    object,
                    field,
                } => {
                    Emit::emit(&88u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(object, writer)?;
                    writer.metadata(MetadataRef::Field(field))?;
                    Ok(())
                }
                Self::LoadStaticFieldAddr { dest, field } => {
                    Emit::emit(&89u16, writer)?;
                    Emit::emit(dest, writer)?;
                    writer.metadata(MetadataRef::Field(field))?;
                    Ok(())
                }
                Self::LoadElement {
                    dest,
                    array,
                    index,
                    elem_type,
                } => {
                    Emit::emit(&90u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(array, writer)?;
                    Emit::emit(index, writer)?;
                    writer.metadata(MetadataRef::Type(elem_type))?;
                    Ok(())
                }
                Self::StoreElement {
                    array,
                    index,
                    value,
                    elem_type,
                } => {
                    Emit::emit(&91u16, writer)?;
                    Emit::emit(array, writer)?;
                    Emit::emit(index, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::Type(elem_type))?;
                    Ok(())
                }
                Self::LoadElementAddr {
                    dest,
                    array,
                    index,
                    elem_type,
                    readonly,
                } => {
                    Emit::emit(&92u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(array, writer)?;
                    Emit::emit(index, writer)?;
                    writer.metadata(MetadataRef::TypeRef(elem_type))?;
                    Emit::emit(readonly, writer)?;
                    Ok(())
                }
                Self::ArrayLength { dest, array } => {
                    Emit::emit(&93u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(array, writer)?;
                    Ok(())
                }
                Self::LoadIndirect {
                    dest,
                    addr,
                    value_type,
                    access,
                } => {
                    Emit::emit(&94u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(addr, writer)?;
                    writer.metadata(MetadataRef::Type(value_type))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::StoreIndirect {
                    addr,
                    value,
                    value_type,
                    access,
                } => {
                    Emit::emit(&95u16, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::Type(value_type))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::NewObj { dest, ctor, args } => {
                    Emit::emit(&96u16, writer)?;
                    Emit::emit(dest, writer)?;
                    writer.metadata(MetadataRef::Method(ctor))?;
                    Emit::emit(args, writer)?;
                    Ok(())
                }
                Self::NewArr {
                    dest,
                    elem_type,
                    length,
                } => {
                    Emit::emit(&97u16, writer)?;
                    Emit::emit(dest, writer)?;
                    writer.metadata(MetadataRef::TypeRef(elem_type))?;
                    Emit::emit(length, writer)?;
                    Ok(())
                }
                Self::CastClass {
                    dest,
                    object,
                    target_type,
                } => {
                    Emit::emit(&98u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(object, writer)?;
                    writer.metadata(MetadataRef::TypeRef(target_type))?;
                    Ok(())
                }
                Self::IsInst {
                    dest,
                    object,
                    target_type,
                } => {
                    Emit::emit(&99u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(object, writer)?;
                    writer.metadata(MetadataRef::TypeRef(target_type))?;
                    Ok(())
                }
                Self::Box {
                    dest,
                    value,
                    value_type,
                } => {
                    Emit::emit(&100u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Ok(())
                }
                Self::Unbox {
                    dest,
                    object,
                    value_type,
                } => {
                    Emit::emit(&101u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(object, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Ok(())
                }
                Self::UnboxAny {
                    dest,
                    object,
                    value_type,
                } => {
                    Emit::emit(&102u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(object, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Ok(())
                }
                Self::SizeOf { dest, value_type } => {
                    Emit::emit(&103u16, writer)?;
                    Emit::emit(dest, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Ok(())
                }
                Self::LoadToken { dest, token } => {
                    Emit::emit(&104u16, writer)?;
                    Emit::emit(dest, writer)?;
                    writer.metadata(MetadataRef::TypeRef(token))?;
                    Ok(())
                }
                Self::ArgList { dest } => {
                    Emit::emit(&105u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Ok(())
                }
                Self::ArgNext { dest, list, ty } => {
                    Emit::emit(&212u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(list, writer)?;
                    writer.metadata(MetadataRef::Type(ty))?;
                    Ok(())
                }
                Self::ArgStep {
                    dest,
                    next,
                    list,
                    ty,
                } => {
                    Emit::emit(&213u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(next, writer)?;
                    Emit::emit(list, writer)?;
                    writer.metadata(MetadataRef::Type(ty))?;
                    Ok(())
                }
                Self::MakeTypedRef {
                    dest,
                    address,
                    value_type,
                } => {
                    Emit::emit(&106u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(address, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Ok(())
                }
                Self::TypedRefValue {
                    dest,
                    typed_ref,
                    value_type,
                } => {
                    Emit::emit(&107u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(typed_ref, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Ok(())
                }
                Self::TypedRefType { dest, typed_ref } => {
                    Emit::emit(&108u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(typed_ref, writer)?;
                    Ok(())
                }
                Self::Call(value) => {
                    // A call returning values beside its result has its own
                    // tag, so a reader cannot take it for one returning one.
                    if value.results.is_empty() {
                        Emit::emit(&109u16, writer)?;
                        Emit::emit(value, writer)?;
                    } else {
                        Emit::emit(&218u16, writer)?;
                        Emit::emit(value, writer)?;
                        Emit::emit(value.results.as_slice(), writer)?;
                    }
                    Ok(())
                }
                Self::LoadFunctionPtr {
                    dest,
                    method,
                    constraint,
                } => {
                    Emit::emit(&112u16, writer)?;
                    Emit::emit(dest, writer)?;
                    writer.metadata(MetadataRef::Method(method))?;
                    writer.optional_metadata(constraint.as_ref().map(MetadataRef::TypeRef))?;
                    Ok(())
                }
                Self::LoadVirtFunctionPtr {
                    dest,
                    object,
                    method,
                } => {
                    Emit::emit(&113u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(object, writer)?;
                    writer.metadata(MetadataRef::Method(method))?;
                    Ok(())
                }
                Self::LoadArg { dest, arg_index } => {
                    Emit::emit(&114u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(arg_index, writer)?;
                    Ok(())
                }
                Self::LoadLocal { dest, local_index } => {
                    Emit::emit(&115u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(local_index, writer)?;
                    Ok(())
                }
                Self::LoadArgAddr { dest, arg_index } => {
                    Emit::emit(&116u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(arg_index, writer)?;
                    Ok(())
                }
                Self::LoadLocalAddr { dest, local_index } => {
                    Emit::emit(&117u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(local_index, writer)?;
                    Ok(())
                }
                Self::Copy { dest, src } => {
                    Emit::emit(&118u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(src, writer)?;
                    Ok(())
                }
                Self::Pop { value } => {
                    Emit::emit(&119u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::Throw { exception } => {
                    Emit::emit(&120u16, writer)?;
                    Emit::emit(exception, writer)?;
                    Ok(())
                }
                Self::Rethrow => {
                    Emit::emit(&121u16, writer)?;
                    Ok(())
                }
                Self::EndFinally => {
                    Emit::emit(&122u16, writer)?;
                    Ok(())
                }
                Self::EndFilter { result } => {
                    Emit::emit(&123u16, writer)?;
                    Emit::emit(result, writer)?;
                    Ok(())
                }
                Self::RuntimeValue { dest, kind } => {
                    Emit::emit(&124u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(kind, writer)?;
                    Ok(())
                }
                Self::InterruptReturn => {
                    Emit::emit(&125u16, writer)?;
                    Ok(())
                }
                Self::Unreachable => {
                    Emit::emit(&126u16, writer)?;
                    Ok(())
                }
                Self::Leave { target } => {
                    Emit::emit(&127u16, writer)?;
                    Emit::emit(target, writer)?;
                    Ok(())
                }
                Self::InitBlk {
                    dest_addr,
                    value,
                    size,
                    backward,
                    access,
                } => {
                    Emit::emit(&128u16, writer)?;
                    Emit::emit(dest_addr, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(size, writer)?;
                    Emit::emit(backward, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::MoveBlk {
                    dest_addr,
                    src_addr,
                    size,
                    src_access,
                    dest_access,
                } => {
                    Emit::emit(&129u16, writer)?;
                    Emit::emit(dest_addr, writer)?;
                    Emit::emit(src_addr, writer)?;
                    Emit::emit(size, writer)?;
                    Emit::emit(src_access, writer)?;
                    Emit::emit(dest_access, writer)?;
                    Ok(())
                }
                Self::CopyBlk {
                    dest_addr,
                    src_addr,
                    size,
                    element_bytes,
                    backward,
                    src_access,
                    dest_access,
                } => {
                    Emit::emit(&130u16, writer)?;
                    Emit::emit(dest_addr, writer)?;
                    Emit::emit(src_addr, writer)?;
                    Emit::emit(size, writer)?;
                    Emit::emit(element_bytes, writer)?;
                    Emit::emit(backward, writer)?;
                    Emit::emit(src_access, writer)?;
                    Emit::emit(dest_access, writer)?;
                    Ok(())
                }
                Self::Fence { kind } => {
                    Emit::emit(&131u16, writer)?;
                    Emit::emit(kind, writer)?;
                    Ok(())
                }
                Self::SystemOp(value) => {
                    Emit::emit(&133u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }

                Self::Contract(value) => {
                    Emit::emit(&139u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::VectorPredicateWhile(value) => {
                    Emit::emit(&142u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::VectorExtendInLane(value) => {
                    Emit::emit(&146u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::VectorElementCount(value) => {
                    Emit::emit(&147u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }

                Self::VectorStructLoadReplicate(value) => {
                    Emit::emit(&150u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::VectorSveCompute(value) => {
                    Emit::emit(&153u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }

                Self::CompareBlk(value) => {
                    Emit::emit(&173u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::ScanBlk {
                    remaining,
                    next,
                    flags,
                    address,
                    value,
                    count,
                    flags_in,
                    backward,
                    element_bytes,
                    while_equal,
                    access,
                } => {
                    Emit::emit(&210u16, writer)?;
                    Emit::emit(remaining, writer)?;
                    Emit::emit(next, writer)?;
                    Emit::emit(flags, writer)?;
                    Emit::emit(address, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(count, writer)?;
                    Emit::emit(flags_in, writer)?;
                    Emit::emit(backward, writer)?;
                    Emit::emit(element_bytes, writer)?;
                    Emit::emit(while_equal, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::LoadBlk {
                    value,
                    next,
                    address,
                    count,
                    value_in,
                    backward,
                    element_bytes,
                    access,
                } => {
                    Emit::emit(&211u16, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(next, writer)?;
                    Emit::emit(address, writer)?;
                    Emit::emit(count, writer)?;
                    Emit::emit(value_in, writer)?;
                    Emit::emit(backward, writer)?;
                    Emit::emit(element_bytes, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::CallClobber { outputs, machine } => {
                    let keyed = machine.as_deref().is_some_and(|values| !values.is_empty());
                    Emit::emit(&(if keyed { 213u16 } else { 175u16 }), writer)?;
                    Emit::emit(outputs, writer)?;
                    if let Some(machine) = machine.as_deref().filter(|_| keyed) {
                        Emit::emit(machine, writer)?;
                    }
                    Ok(())
                }
                Self::AtomicLoad {
                    dest,
                    addr,
                    value_type,
                    ordering,
                    width,
                    reserves,
                    access,
                } => {
                    Emit::emit(&176u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(addr, writer)?;
                    writer.metadata(MetadataRef::Type(value_type))?;
                    Emit::emit(ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(reserves, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicStore {
                    addr,
                    value,
                    value_type,
                    ordering,
                    width,
                    access,
                } => {
                    Emit::emit(&177u16, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::Type(value_type))?;
                    Emit::emit(ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicStoreConditional {
                    status,
                    status_encoding,
                    consumes_reservation,
                    addr,
                    value,
                    value_type,
                    success_ordering,
                    failure_ordering,
                    width,
                    access,
                } => {
                    Emit::emit(&178u16, writer)?;
                    Emit::emit(status, writer)?;
                    Emit::emit(status_encoding, writer)?;
                    Emit::emit(consumes_reservation, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::Type(value_type))?;
                    Emit::emit(success_ordering, writer)?;
                    Emit::emit(failure_ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicPairLoad {
                    first,
                    second,
                    addr,
                    first_type,
                    second_type,
                    ordering,
                    width,
                    access,
                } => {
                    Emit::emit(&179u16, writer)?;
                    Emit::emit(first, writer)?;
                    Emit::emit(second, writer)?;
                    Emit::emit(addr, writer)?;
                    writer.metadata(MetadataRef::Type(first_type))?;
                    writer.metadata(MetadataRef::Type(second_type))?;
                    Emit::emit(ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicPairStoreConditional {
                    status,
                    status_encoding,
                    consumes_reservation,
                    addr,
                    first_value,
                    second_value,
                    first_type,
                    second_type,
                    success_ordering,
                    failure_ordering,
                    width,
                    access,
                } => {
                    Emit::emit(&180u16, writer)?;
                    Emit::emit(status, writer)?;
                    Emit::emit(status_encoding, writer)?;
                    Emit::emit(consumes_reservation, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(first_value, writer)?;
                    Emit::emit(second_value, writer)?;
                    writer.metadata(MetadataRef::Type(first_type))?;
                    writer.metadata(MetadataRef::Type(second_type))?;
                    Emit::emit(success_ordering, writer)?;
                    Emit::emit(failure_ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicExchange {
                    dest,
                    addr,
                    value,
                    ordering,
                    width,
                    access,
                } => {
                    Emit::emit(&181u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicLockRmw {
                    dest,
                    addr,
                    value,
                    op,
                    ordering,
                    width,
                    access,
                } => {
                    Emit::emit(&182u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(op, writer)?;
                    Emit::emit(ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicCondRmw {
                    old,
                    addr,
                    compare,
                    value,
                    op,
                    condition,
                    ordering,
                    width,
                    access,
                } => {
                    Emit::emit(&183u16, writer)?;
                    Emit::emit(old, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(compare, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(op, writer)?;
                    Emit::emit(condition, writer)?;
                    Emit::emit(ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicCmpXchg {
                    old,
                    success,
                    addr,
                    expected,
                    desired,
                    success_ordering,
                    failure_ordering,
                    width,
                    weak,
                    access,
                } => {
                    Emit::emit(&184u16, writer)?;
                    Emit::emit(old, writer)?;
                    Emit::emit(success, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(expected, writer)?;
                    Emit::emit(desired, writer)?;
                    Emit::emit(success_ordering, writer)?;
                    Emit::emit(failure_ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(weak, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicPairCmpXchg {
                    old_first,
                    old_second,
                    addr,
                    expected_first,
                    expected_second,
                    desired_first,
                    desired_second,
                    success_ordering,
                    failure_ordering,
                    width,
                    weak,
                    access,
                } => {
                    Emit::emit(&185u16, writer)?;
                    Emit::emit(old_first, writer)?;
                    Emit::emit(old_second, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(expected_first, writer)?;
                    Emit::emit(expected_second, writer)?;
                    Emit::emit(desired_first, writer)?;
                    Emit::emit(desired_second, writer)?;
                    Emit::emit(success_ordering, writer)?;
                    Emit::emit(failure_ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(weak, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicPairRmw {
                    old_first,
                    old_second,
                    addr,
                    value_first,
                    value_second,
                    op,
                    ordering,
                    width,
                    access,
                } => {
                    Emit::emit(&186u16, writer)?;
                    Emit::emit(old_first, writer)?;
                    Emit::emit(old_second, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(value_first, writer)?;
                    Emit::emit(value_second, writer)?;
                    Emit::emit(op, writer)?;
                    Emit::emit(ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::AtomicCheckedRmw {
                    old,
                    status,
                    addr,
                    compare,
                    value,
                    op,
                    rcw_mask,
                    rcws_mask,
                    protection,
                    ordering,
                    width,
                    access,
                } => {
                    Emit::emit(&217u16, writer)?;
                    Emit::emit(old, writer)?;
                    Emit::emit(status, writer)?;
                    Emit::emit(addr, writer)?;
                    Emit::emit(compare, writer)?;
                    Emit::emit(value, writer)?;
                    Emit::emit(op, writer)?;
                    Emit::emit(rcw_mask, writer)?;
                    Emit::emit(rcws_mask, writer)?;
                    Emit::emit(protection, writer)?;
                    Emit::emit(ordering, writer)?;
                    Emit::emit(width, writer)?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::InitObj {
                    dest_addr,
                    value_type,
                } => {
                    Emit::emit(&187u16, writer)?;
                    Emit::emit(dest_addr, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Ok(())
                }
                Self::CopyObj {
                    dest_addr,
                    src_addr,
                    value_type,
                } => {
                    Emit::emit(&188u16, writer)?;
                    Emit::emit(dest_addr, writer)?;
                    Emit::emit(src_addr, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Ok(())
                }
                Self::LoadObj {
                    dest,
                    src_addr,
                    value_type,
                    access,
                } => {
                    Emit::emit(&189u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(src_addr, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::StoreObj {
                    dest_addr,
                    value,
                    value_type,
                    access,
                } => {
                    Emit::emit(&190u16, writer)?;
                    Emit::emit(dest_addr, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::TypeRef(value_type))?;
                    Emit::emit(access, writer)?;
                    Ok(())
                }
                Self::Nop => {
                    Emit::emit(&191u16, writer)?;
                    Ok(())
                }
                Self::Break(value) => {
                    Emit::emit(&192u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::Ckfinite { dest, operand } => {
                    Emit::emit(&193u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    Ok(())
                }
                Self::FpClassify {
                    dest,
                    operand,
                    layout,
                } => {
                    Emit::emit(&194u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operand, writer)?;
                    Emit::emit(layout, writer)?;
                    Ok(())
                }
                Self::Compute(value) => {
                    Emit::emit(&209u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::FloatFunction(value) => {
                    Emit::emit(&195u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::FpuControl(value) => {
                    Emit::emit(&196u16, writer)?;
                    Emit::emit(value, writer)?;
                    Ok(())
                }
                Self::LocalAlloc { dest, size, zeroed } => {
                    Emit::emit(&197u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(size, writer)?;
                    Emit::emit(zeroed, writer)?;
                    Ok(())
                }
                Self::Phi { dest, operands } => {
                    Emit::emit(&203u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(operands, writer)?;
                    Ok(())
                }
                Self::VectorSlice {
                    dest,
                    vector,
                    offset,
                    length,
                    target_type,
                } => {
                    Emit::emit(&204u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(vector, writer)?;
                    Emit::emit(offset, writer)?;
                    Emit::emit(length, writer)?;
                    writer.metadata(MetadataRef::Type(target_type))?;
                    Ok(())
                }
                Self::VectorInsertSlice {
                    dest,
                    vector,
                    offset,
                    value,
                    target_type,
                } => {
                    Emit::emit(&205u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(vector, writer)?;
                    Emit::emit(offset, writer)?;
                    Emit::emit(value, writer)?;
                    writer.metadata(MetadataRef::Type(target_type))?;
                    Ok(())
                }
                Self::VectorConcat {
                    dest,
                    left,
                    right,
                    target_type,
                } => {
                    Emit::emit(&206u16, writer)?;
                    Emit::emit(dest, writer)?;
                    Emit::emit(left, writer)?;
                    Emit::emit(right, writer)?;
                    writer.metadata(MetadataRef::Type(target_type))?;
                    Ok(())
                }
            };
        }
    }
}
