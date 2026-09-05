//! Human-readable rendering of [`SsaOp`].
//!
//! The format is diagnostic output — verifier messages, pass logs, test
//! assertions — not a parseable serialization; use the `serde` feature for
//! round-tripping.

use std::fmt;

use crate::{
    ir::{
        ops::{
            def::SsaOp,
            kinds::{
                BcdAdjustData, BcdAdjustKind, KindedVecData, NativeInstructionMetadata,
                NativeKindedData, NativeOpaqueData, VecImm8Data,
            },
            native::{BlockStringOpData, WideCmpXchgData},
            vector::{
                VectorBitfieldData, VectorComplexAddData, VectorConditionalMoveData,
                VectorCountAdjustData, VectorDotProductData, VectorElementCountData,
                VectorExtendInLaneData, VectorHorizontalMinPosData, VectorHorizontalReduceData,
                VectorIntDotProductData, VectorIntersectData, VectorMatrixMulAccData,
                VectorNarrowSaturateData, VectorPackNarrowData, VectorPermuteData,
                VectorPredicateBreakData, VectorPredicateOpData, VectorPredicateWhileData,
                VectorReverseChunksData, VectorShuffleBitsData, VectorSmeMiscData,
                VectorSmeOuterProductData, VectorStringCompareData, VectorStructLoadReplicateData,
                VectorSveAddressGenData, VectorSveComputeData,
            },
        },
        variable::SsaVarId,
    },
    target::Target,
};

/// Writes an operation's definitions as its rendered prefix -- `v1, v2 = ` --
/// or nothing when it defines nothing.
///
/// Every rendered operation that defines a variable opens this way, so the
/// prefix is written once. An arm that spelled it itself could disagree about
/// the separator, or emit a bare `" = "` for an empty definition list.
fn write_defs(f: &mut fmt::Formatter<'_>, defs: &[SsaVarId]) -> fmt::Result {
    if defs.is_empty() {
        return Ok(());
    }
    for (i, def) in defs.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{def}")?;
    }
    write!(f, " = ")?;
    Ok(())
}

/// Writes an operation's operand list after its mnemonic -- ` v1, v2` -- or
/// nothing when it reads nothing.
///
/// Takes no separator argument on purpose. A separator parameter is the 45-way
/// choice this helper exists to close: every call site would have to decide
/// again, and the answer would drift apart across the variants exactly as it
/// did when each arm wrote its own loop.
fn write_inputs(f: &mut fmt::Formatter<'_>, inputs: &[SsaVarId]) -> fmt::Result {
    if inputs.is_empty() {
        return Ok(());
    }
    write!(f, " ")?;
    for (i, input) in inputs.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{input}")?;
    }
    Ok(())
}

/// Writes the provenance suffix a lifted native instruction carries --
/// architecture, original address, and encoded length -- omitting each part
/// that is absent.
fn write_native_metadata(
    f: &mut fmt::Formatter<'_>,
    metadata: Option<&NativeInstructionMetadata>,
) -> fmt::Result {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    if let Some(architecture) = &metadata.architecture {
        write!(f, " arch={architecture}")?;
    }
    if let Some(address) = metadata.address {
        write!(f, " addr=0x{address:x}")?;
    }
    if !metadata.raw_bytes.is_empty() {
        write!(f, " bytes={}", metadata.raw_bytes.len())?;
    }
    Ok(())
}

/// Writes an `addrspace(N)` qualifier prefix, or nothing for the default
/// (flat) space.
fn write_address_space(f: &mut fmt::Formatter<'_>, address_space: Option<u16>) -> fmt::Result {
    match address_space {
        Some(space) => write!(f, "addrspace({space}) "),
        None => Ok(()),
    }
}

impl<T: Target> fmt::Display for SsaOp<T>
where
    T::TypeRef: fmt::Display,
    T::MethodRef: fmt::Display,
    T::FieldRef: fmt::Display,
    T::SymbolRef: fmt::Display,
    T::SigRef: fmt::Display,
    T::Type: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const { dest, value } => write!(f, "{dest} = {value}"),
            Self::Add {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = add {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = add {left}, {right}")
                }
            }
            Self::AddOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = add.ovf{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = add.ovf{suffix} {left}, {right}")
                }
            }
            Self::Sub {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = sub {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = sub {left}, {right}")
                }
            }
            Self::SubOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = sub.ovf{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = sub.ovf{suffix} {left}, {right}")
                }
            }
            Self::Mul {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = mul {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = mul {left}, {right}")
                }
            }
            Self::MulOvf {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = mul.ovf{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = mul.ovf{suffix} {left}, {right}")
                }
            }
            Self::WideMul {
                low,
                high,
                left,
                right,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(f, "{low}, {high} = widemul{suffix} {left}, {right}")
            }
            Self::Div {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = div{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = div{suffix} {left}, {right}")
                }
            }
            Self::Rem {
                dest,
                left,
                right,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = rem{suffix} {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = rem{suffix} {left}, {right}")
                }
            }
            Self::FloatCompareFlags {
                flags,
                left,
                right,
                signaling,
            } => {
                let suffix = if *signaling { ".signaling" } else { "" };
                write!(f, "{flags} = fcmp.flags{suffix} {left}, {right}")
            }
            Self::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(
                    f,
                    "{quotient}, {remainder} = widediv{suffix} {high}:{low}, {divisor}"
                )
            }
            Self::Neg {
                dest,
                operand,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = neg {operand} flags={flags}")
                } else {
                    write!(f, "{dest} = neg {operand}")
                }
            }
            Self::And {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = and {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = and {left}, {right}")
                }
            }
            Self::Or {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = or {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = or {left}, {right}")
                }
            }
            Self::Xor {
                dest,
                left,
                right,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = xor {left}, {right} flags={flags}")
                } else {
                    write!(f, "{dest} = xor {left}, {right}")
                }
            }
            Self::Not {
                dest,
                operand,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = not {operand} flags={flags}")
                } else {
                    write!(f, "{dest} = not {operand}")
                }
            }
            Self::Shl {
                dest,
                value,
                amount,
                flags,
            } => {
                if let Some(flags) = flags {
                    write!(f, "{dest} = shl {value}, {amount} flags={flags}")
                } else {
                    write!(f, "{dest} = shl {value}, {amount}")
                }
            }
            Self::Shr {
                dest,
                value,
                amount,
                unsigned,
                flags,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                if let Some(flags) = flags {
                    write!(f, "{dest} = shr{suffix} {value}, {amount} flags={flags}")
                } else {
                    write!(f, "{dest} = shr{suffix} {value}, {amount}")
                }
            }
            Self::Rol {
                dest,
                value,
                amount,
            } => write!(f, "{dest} = rol {value}, {amount}"),
            Self::Ror {
                dest,
                value,
                amount,
            } => write!(f, "{dest} = ror {value}, {amount}"),
            Self::Rcl {
                dest,
                value,
                amount,
            } => write!(f, "{dest} = rcl {value}, {amount}"),
            Self::Rcr {
                dest,
                value,
                amount,
            } => write!(f, "{dest} = rcr {value}, {amount}"),
            Self::BSwap { dest, src } => write!(f, "{dest} = bswap {src}"),
            Self::BRev { dest, src } => write!(f, "{dest} = brev {src}"),
            Self::BitScanForward { dest, src } => write!(f, "{dest} = bsf {src}"),
            Self::BitScanReverse { dest, src } => write!(f, "{dest} = bsr {src}"),
            Self::Popcount { dest, src } => write!(f, "{dest} = popcnt {src}"),
            Self::Parity { dest, src } => write!(f, "{dest} = parity {src}"),
            Self::Ceq { dest, left, right } => write!(f, "{dest} = ceq {left}, {right}"),
            Self::Clt {
                dest,
                left,
                right,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(f, "{dest} = clt{suffix} {left}, {right}")
            }
            Self::Cgt {
                dest,
                left,
                right,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(f, "{dest} = cgt{suffix} {left}, {right}")
            }
            Self::BoolAnd { dest, left, right } => {
                write!(f, "{dest} = bool.and {left}, {right}")
            }
            Self::BoolOr { dest, left, right } => {
                write!(f, "{dest} = bool.or {left}, {right}")
            }
            Self::BoolXor { dest, left, right } => {
                write!(f, "{dest} = bool.xor {left}, {right}")
            }
            Self::BoolNot { dest, value } => write!(f, "{dest} = bool.not {value}"),
            Self::IntConv {
                dest,
                operand,
                target,
                ..
            } => write!(f, "{dest} = conv.{target} {operand}"),
            Self::IntToPtr {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = inttoptr.{target} {operand}"),
            Self::PtrToInt {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = ptrtoint.{target} {operand}"),
            Self::IntToFloat {
                dest,
                operand,
                target,
                ..
            } => write!(f, "{dest} = inttofloat.{target} {operand}"),
            Self::FloatToInt {
                dest,
                operand,
                target,
                ..
            } => write!(f, "{dest} = floattoint.{target} {operand}"),
            Self::FloatConv {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = fconv.{target} {operand}"),
            Self::Bitcast {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = bitcast.{target} {operand}"),
            Self::ReadFlags { dest, flags, mask } => {
                write!(f, "{dest} = readflags {flags}, {mask}")
            }
            Self::ComputeFlags { dest, inputs } => {
                write_defs(f, std::slice::from_ref(dest))?;
                write!(f, "flags.compute")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::CallClobber { outputs } => {
                write_defs(f, outputs)?;
                write!(f, "call.clobber")
            }
            Self::VectorUnary {
                dest, value, kind, ..
            } => {
                write!(f, "{dest} = vunary.{kind:?} {value}")
            }
            Self::VectorBinary {
                dest,
                left,
                right,
                kind,
                ..
            } => write!(f, "{dest} = vbinary.{kind:?} {left}, {right}"),
            Self::VectorTernary {
                dest,
                first,
                second,
                third,
                kind,
            } => write!(f, "{dest} = vternary.{kind:?} {first}, {second}, {third}"),
            Self::VectorPredicatedUnary {
                dest,
                value,
                mask,
                passthrough,
                kind,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vunary.pred.{kind:?}.{mode:?} {value}, {mask}, {passthrough}"
                    )
                } else {
                    write!(f, "{dest} = vunary.pred.{kind:?}.{mode:?} {value}, {mask}")
                }
            }
            Self::VectorPredicatedBinary {
                dest,
                left,
                right,
                mask,
                passthrough,
                kind,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vbinary.pred.{kind:?}.{mode:?} {left}, {right}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vbinary.pred.{kind:?}.{mode:?} {left}, {right}, {mask}"
                    )
                }
            }
            Self::VectorPredicatedTernary {
                dest,
                first,
                second,
                third,
                mask,
                passthrough,
                kind,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vternary.pred.{kind:?}.{mode:?} {first}, {second}, {third}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vternary.pred.{kind:?}.{mode:?} {first}, {second}, {third}, {mask}"
                    )
                }
            }
            Self::VectorCompare {
                dest,
                left,
                right,
                kind,
                unsigned,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(f, "{dest} = vcmp.{kind:?}{suffix} {left}, {right}")
            }
            Self::VectorLoad {
                dest,
                addr,
                vector_type,
            } => write!(f, "{dest} = vload.{vector_type} {addr}"),
            Self::VectorStore {
                addr,
                value,
                vector_type,
            } => write!(f, "vstore.{vector_type} {addr}, {value}"),
            Self::VectorMaskedLoad {
                dest,
                addr,
                mask,
                passthrough,
                vector_type,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vload.masked.{mode:?}.{vector_type} {addr}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vload.masked.{mode:?}.{vector_type} {addr}, {mask}"
                    )
                }
            }
            Self::VectorMaskedStore {
                addr,
                value,
                mask,
                vector_type,
            } => write!(f, "vstore.masked.{vector_type} {addr}, {value}, {mask}"),
            Self::VectorBroadcastLoad {
                dest,
                addr,
                vector_type,
            } => write!(f, "{dest} = vbroadcast.load.{vector_type} {addr}"),
            Self::VectorGather {
                dest,
                base,
                indices,
                mask,
                passthrough,
                vector_type,
                mode,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vgather.{mode:?}.{vector_type} {base}, {indices}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vgather.{mode:?}.{vector_type} {base}, {indices}, {mask}"
                    )
                }
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
            } => match (fault, mask, passthrough) {
                (Some(fault), Some(mask), Some(passthrough)) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {mask}, {passthrough}"
                ),
                (Some(fault), Some(mask), None) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {mask}"
                ),
                (Some(fault), None, Some(passthrough)) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {passthrough}"
                ),
                (Some(fault), None, None) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}"
                ),
                (None, Some(mask), Some(passthrough)) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {mask}, {passthrough}"
                ),
                (None, Some(mask), None) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {mask}"
                ),
                (None, None, Some(passthrough)) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}, {passthrough}"
                ),
                (None, None, None) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {addr}"
                ),
            },
            Self::VectorSegmentLoad {
                dests,
                base,
                mask,
                vector_type,
                segments,
                layout,
            } => {
                write_defs(f, dests)?;
                if let Some(mask) = mask {
                    write!(
                        f,
                        "vload.segment.{layout:?}.{segments}.{vector_type} {base}, {mask}"
                    )
                } else {
                    write!(
                        f,
                        "vload.segment.{layout:?}.{segments}.{vector_type} {base}"
                    )
                }
            }
            Self::VectorScatter {
                base,
                indices,
                value,
                mask,
                vector_type,
            } => write!(
                f,
                "vscatter.{vector_type} {base}, {indices}, {value}, {mask}"
            ),
            Self::VectorSegmentStore {
                base,
                values,
                mask,
                vector_type,
                segments,
                layout,
            } => {
                if let Some(mask) = mask {
                    write!(
                        f,
                        "vstore.segment.{layout:?}.{segments}.{vector_type} {base}, {values:?}, {mask}"
                    )
                } else {
                    write!(
                        f,
                        "vstore.segment.{layout:?}.{segments}.{vector_type} {base}, {values:?}"
                    )
                }
            }
            Self::VectorExtract { dest, vector, lane } => {
                write!(f, "{dest} = vextract {vector}, {lane}")
            }
            Self::VectorInsert {
                dest,
                vector,
                lane,
                value,
            } => write!(f, "{dest} = vinsert {vector}, {lane}, {value}"),
            Self::VectorSplat {
                dest,
                value,
                vector_type,
            } => write!(f, "{dest} = vsplat.{vector_type} {value}"),
            Self::VectorShuffle {
                dest, left, right, ..
            } => {
                if let Some(right) = right {
                    write!(f, "{dest} = vshuffle {left}, {right}")
                } else {
                    write!(f, "{dest} = vshuffle {left}")
                }
            }
            Self::VectorCast {
                dest,
                value,
                target_type,
                kind,
            } => write!(f, "{dest} = vcast.{kind:?}.{target_type} {value}"),
            Self::VectorReinterpret {
                dest,
                value,
                target_type,
            } => write!(f, "{dest} = vreinterpret.{target_type} {value}"),
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
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vpack.{kind:?}.{mode:?}.e{element_bits}.{vector_type} {value}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vpack.{kind:?}.{mode:?}.e{element_bits}.{vector_type} {value}, {mask}"
                    )
                }
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
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vpack.load.{kind:?}.{mode:?}.e{element_bits}.{vector_type} [{addr}], {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vpack.load.{kind:?}.{mode:?}.e{element_bits}.{vector_type} [{addr}], {mask}"
                    )
                }
            }
            Self::VectorPackStore {
                addr,
                value,
                mask,
                vector_type,
                element_bits,
                kind,
            } => write!(
                f,
                "vpack.store.{kind:?}.e{element_bits}.{vector_type} [{addr}], {value}, {mask}"
            ),
            Self::VectorZeroUpper { all } => {
                let suffix = if *all { "all" } else { "upper" };
                write!(f, "vzero.{suffix}")
            }
            Self::VectorMaskUnary { dest, mask, kind } => {
                write!(f, "{dest} = vmask.unary.{kind:?} {mask}")
            }
            Self::VectorMaskBinary {
                dest,
                left,
                right,
                kind,
            } => write!(f, "{dest} = vmask.binary.{kind:?} {left}, {right}"),
            Self::VectorReduce { dest, value, kind } => {
                write!(f, "{dest} = vreduce.{kind:?} {value}")
            }
            Self::VectorBitmask { dest, value, kind } => {
                write!(f, "{dest} = vbitmask.{kind:?} {value}")
            }
            Self::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => write!(f, "{dest} = select {condition}, {true_val}, {false_val}"),
            Self::Jump { target } => write!(f, "jump B{target}"),
            Self::Branch {
                condition,
                true_target,
                false_target,
            } => write!(f, "branch {condition}, B{true_target}, B{false_target}"),
            Self::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                write!(
                    f,
                    "branchcmp{suffix} {left} {cmp} {right}, B{true_target}, B{false_target}"
                )
            }
            Self::BranchFlags {
                flags,
                condition,
                true_target,
                false_target,
            } => {
                write!(
                    f,
                    "branchflags {flags} {condition} B{true_target}, B{false_target}"
                )
            }
            Self::Switch {
                value,
                targets,
                default,
            } => {
                write!(f, "switch {value}, [")?;
                for (i, t) in targets.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "B{t}")?;
                }
                write!(f, "], B{default}")
            }
            Self::IndirectBranch {
                target,
                resolved_targets,
            } => {
                write!(f, "branch.indirect {target}")?;
                if !resolved_targets.is_empty() {
                    write!(f, " [")?;
                    for (i, t) in resolved_targets.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "B{t}")?;
                    }
                    write!(f, "]")?;
                }
                Ok(())
            }
            Self::Return { value: Some(v) } => write!(f, "ret {v}"),
            Self::Return { value: None } => write!(f, "ret"),
            Self::LoadField {
                dest,
                object,
                field,
            } => {
                write!(f, "{dest} = ldfld {field}, {object}")
            }
            Self::StoreField {
                object,
                field,
                value,
            } => write!(f, "stfld {field}, {object}, {value}"),
            Self::LoadStaticField { dest, field } => write!(f, "{dest} = ldsfld {field}"),
            Self::StoreStaticField { field, value } => write!(f, "stsfld {field}, {value}"),
            Self::LoadFieldAddr {
                dest,
                object,
                field,
            } => {
                write!(f, "{dest} = ldflda {field}, {object}")
            }
            Self::LoadStaticFieldAddr { dest, field } => write!(f, "{dest} = ldsflda {field}"),
            Self::LoadElement {
                dest,
                array,
                index,
                elem_type,
            } => write!(f, "{dest} = ldelem.{elem_type} {array}[{index}]"),
            Self::StoreElement {
                array,
                index,
                value,
                elem_type,
            } => write!(f, "stelem.{elem_type} {array}[{index}], {value}"),
            Self::LoadElementAddr {
                dest, array, index, ..
            } => write!(f, "{dest} = ldelema {array}[{index}]"),
            Self::PtrAdd {
                dest,
                base,
                index,
                stride,
                offset,
                ..
            } => {
                write!(f, "{dest} = ptradd {base}")?;
                if let Some(index) = index {
                    write!(f, " + {index}*{stride}")?;
                }
                if *offset != 0 {
                    write!(f, " + {offset}")?;
                }
                Ok(())
            }
            Self::ArrayLength { dest, array } => write!(f, "{dest} = ldlen {array}"),
            Self::LoadIndirect {
                dest,
                addr,
                value_type,
                address_space,
            } => {
                write!(f, "{dest} = ldind.{value_type} ")?;
                write_address_space(f, *address_space)?;
                write!(f, "{addr}")
            }
            Self::StoreIndirect {
                addr,
                value,
                value_type,
                address_space,
            } => {
                write!(f, "stind.{value_type} ")?;
                write_address_space(f, *address_space)?;
                write!(f, "{addr}, {value}")
            }
            Self::NewObj { dest, ctor, args } => {
                write!(f, "{dest} = newobj {ctor}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::NewArr {
                dest,
                elem_type,
                length,
            } => write!(f, "{dest} = newarr {elem_type}[{length}]"),
            Self::CastClass {
                dest,
                object,
                target_type,
            } => write!(f, "{dest} = castclass {target_type}, {object}"),
            Self::IsInst {
                dest,
                object,
                target_type,
            } => write!(f, "{dest} = isinst {target_type}, {object}"),
            Self::Box {
                dest,
                value,
                value_type,
            } => write!(f, "{dest} = box {value_type}, {value}"),
            Self::Unbox {
                dest,
                object,
                value_type,
            } => write!(f, "{dest} = unbox {value_type}, {object}"),
            Self::UnboxAny {
                dest,
                object,
                value_type,
            } => write!(f, "{dest} = unbox.any {value_type}, {object}"),
            Self::SizeOf { dest, value_type } => write!(f, "{dest} = sizeof {value_type}"),
            Self::LoadToken { dest, token } => write!(f, "{dest} = ldtoken {token}"),
            Self::Call { dest, method, args } => {
                if let Some(d) = dest {
                    write!(f, "{d} = ")?;
                }
                write!(f, "call {method}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::CallVirt { dest, method, args } => {
                if let Some(d) = dest {
                    write!(f, "{d} = ")?;
                }
                write!(f, "callvirt {method}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::CallIndirect {
                dest, fptr, args, ..
            } => {
                if let Some(d) = dest {
                    write!(f, "{d} = ")?;
                }
                write!(f, "calli {fptr}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::LoadFunctionPtr { dest, method } => write!(f, "{dest} = ldftn {method}"),
            Self::LoadVirtFunctionPtr {
                dest,
                object,
                method,
            } => write!(f, "{dest} = ldvirtftn {method}, {object}"),
            Self::LoadArg { dest, arg_index } => write!(f, "{dest} = ldarg {arg_index}"),
            Self::LoadLocal { dest, local_index } => write!(f, "{dest} = ldloc {local_index}"),
            Self::LoadArgAddr { dest, arg_index } => write!(f, "{dest} = ldarga {arg_index}"),
            Self::LoadLocalAddr { dest, local_index } => {
                write!(f, "{dest} = ldloca {local_index}")
            }
            Self::Copy { dest, src } => write!(f, "{dest} = {src}"),
            Self::Pop { value } => write!(f, "pop {value}"),
            Self::Throw { exception } => write!(f, "throw {exception}"),
            Self::Rethrow => write!(f, "rethrow"),
            Self::EndFinally => write!(f, "endfinally"),
            Self::InterruptReturn => write!(f, "iret"),
            Self::Unreachable => write!(f, "unreachable"),
            Self::EndFilter { result } => write!(f, "endfilter {result}"),
            Self::Leave { target } => write!(f, "leave B{target}"),
            Self::InitBlk {
                dest_addr,
                value,
                size,
                reverse,
            } => write!(
                f,
                "initblk{} {dest_addr}, {value}, {size}",
                if *reverse { " rev" } else { "" }
            ),
            Self::CopyBlk {
                dest_addr,
                src_addr,
                size,
                reverse,
            } => write!(
                f,
                "cpblk{} {dest_addr}, {src_addr}, {size}",
                if *reverse { " rev" } else { "" }
            ),
            Self::Fence { kind } => write!(f, "fence {kind}"),
            Self::NativeOpaque(data) => {
                let NativeOpaqueData {
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                    effects,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "native.opaque {mnemonic}")?;
                write_inputs(f, inputs)?;
                write_native_metadata(f, metadata.as_ref())?;
                write!(f, " effects={:?}", effects.kind)
            }
            Self::SystemOp(data) => {
                let NativeKindedData {
                    kind,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "{} {mnemonic}", kind.kind_str())?;
                write_inputs(f, inputs)?;
                write_native_metadata(f, metadata.as_ref())?;
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::ComputeOp(data) => {
                let NativeKindedData {
                    kind,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "{} {mnemonic}", kind.kind_str())?;
                write_inputs(f, inputs)?;
                write_native_metadata(f, metadata.as_ref())?;
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::BcdAdjust(data) => {
                let BcdAdjustData {
                    kind,
                    base,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "{} {mnemonic}", kind.kind_str())?;
                if matches!(
                    kind,
                    BcdAdjustKind::AsciiMulAdjust | BcdAdjustKind::AsciiDivAdjust
                ) {
                    write!(f, " base={base}")?;
                }
                write_inputs(f, inputs)?;
                write_native_metadata(f, metadata.as_ref())?;
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::VectorCrypto(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "{}", kind.kind_str())?;
                write_inputs(f, inputs)?;
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::TileOp(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "{}", kind.kind_str())?;
                write_inputs(f, inputs)?;
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::VectorPermute(data) => {
                let VectorPermuteData { outputs, inputs } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.permute")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorMultiplyAdd(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "{}", kind.kind_str())?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorSvePermute(data) => {
                let KindedVecData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.sve.perm")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorFpHelper(data) => {
                let KindedVecData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.fphelper")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorPredicateGen(data) => {
                let KindedVecData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.pgen")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorSmeOuterProduct(data) => {
                let VectorSmeOuterProductData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.sme.mopa")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorMatrixMulAcc(data) => {
                let VectorMatrixMulAccData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.mmla")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorReverseChunks(data) => {
                let VectorReverseChunksData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.revchunks")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorCountAdjust(data) => {
                let VectorCountAdjustData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.countadj")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorExtendInLane(data) => {
                let VectorExtendInLaneData {
                    signed,
                    source_bits,
                    element_bits,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                let kind = if *signed { "sxt" } else { "uxt" };
                write!(f, "vector.{kind} i{source_bits}->i{element_bits}")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorElementCount(data) => {
                let VectorElementCountData {
                    element_bits,
                    multiplier,
                    outputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.cnt e{element_bits} x{multiplier}")?;
                Ok(())
            }
            Self::VectorSveAddressGen(data) => {
                let VectorSveAddressGenData {
                    signed_extend,
                    shift,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                let ext = match signed_extend {
                    Some(true) => "sxtw",
                    Some(false) => "uxtw",
                    None => "lsl",
                };
                write!(f, "vector.adr {ext} #{shift}")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::FlagAdjust(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "{}", kind.kind_str())?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorStructLoadReplicate(data) => {
                let VectorStructLoadReplicateData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.ldNr")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorSmeMisc(data) => {
                let VectorSmeMiscData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.sme.misc")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorPredicateOp(data) => {
                let VectorPredicateOpData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.predop")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorSveCompute(data) => {
                let VectorSveComputeData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.sve.compute")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorComplexAdd(data) => {
                let VectorComplexAddData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.cadd")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorPredicateBreak(data) => {
                let VectorPredicateBreakData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.break")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorPredicateWhile(data) => {
                let VectorPredicateWhileData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.while")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorNarrowSaturate(data) => {
                let VectorNarrowSaturateData {
                    unsigned_dst,
                    outputs,
                    inputs,
                    ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(
                    f,
                    "{}",
                    if *unsigned_dst {
                        "vector.narrow.saturate.u"
                    } else {
                        "vector.narrow.saturate.s"
                    }
                )?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorPackNarrow(data) => {
                let VectorPackNarrowData {
                    unsigned,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(
                    f,
                    "{}",
                    if *unsigned {
                        "vector.pack.narrow.u"
                    } else {
                        "vector.pack.narrow.s"
                    }
                )?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorTernaryLogic(data) => {
                let VecImm8Data {
                    imm8,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.ternlog")?;
                write_inputs(f, inputs)?;
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorDotProduct(data) => {
                let VectorDotProductData {
                    imm8,
                    element_bits,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.dotproduct.{element_bits}")?;
                write_inputs(f, inputs)?;
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorMultiSad(data) => {
                let VecImm8Data {
                    imm8,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.mpsadbw")?;
                write_inputs(f, inputs)?;
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorIntDotProduct(data) => {
                let VectorIntDotProductData {
                    signed_a,
                    signed_b,
                    source_bits,
                    dest_bits,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.intdot")?;
                write_inputs(f, inputs)?;
                write!(
                    f,
                    " s_a={signed_a} s_b={signed_b} src={source_bits} dst={dest_bits}"
                )
            }
            Self::VectorStringCompare(data) => {
                let VectorStringCompareData {
                    imm8,
                    explicit_length,
                    result_index,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(
                    f,
                    "vector.pcmp{}str{}",
                    if *explicit_length { "e" } else { "i" },
                    if *result_index { "i" } else { "m" }
                )?;
                write_inputs(f, inputs)?;
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorBitfield(data) => {
                let VectorBitfieldData {
                    insert,
                    index,
                    length,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.{}", if *insert { "insertq" } else { "extrq" })?;
                write_inputs(f, inputs)?;
                write!(f, " index={index} length={length}")
            }
            Self::VectorIntersect(data) => {
                let VectorIntersectData { outputs, inputs } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.p2intersect")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorShuffleBits(data) => {
                let VectorShuffleBitsData { outputs, inputs } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.shufbitqmb")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorConditionalMove(data) => {
                let VectorConditionalMoveData {
                    condition,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.condmove.{}", condition.kind_str())?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorHorizontalMinPos(data) => {
                let VectorHorizontalMinPosData { outputs, inputs } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.phminposuw")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorComplexMul(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "{}", kind.kind_str())?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::VectorClassify(data) => {
                let VecImm8Data {
                    imm8,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.fpclass")?;
                write_inputs(f, inputs)?;
                write!(f, " imm={imm8:#04x}")
            }
            Self::VectorHorizontalReduce(data) => {
                let VectorHorizontalReduceData {
                    subtract,
                    source_bits,
                    dest_bits,
                    outputs,
                    inputs,
                    ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(
                    f,
                    "vector.hreduce.{} {}->{}",
                    if *subtract { "sub" } else { "add" },
                    source_bits,
                    dest_bits
                )?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::BlockString(data) => {
                let BlockStringOpData {
                    kind,
                    prefix: _,
                    element_bits: _,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                    reverse: _,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "{} {mnemonic}", kind.kind_str())?;
                write_inputs(f, inputs)?;
                write_native_metadata(f, metadata.as_ref())?;
                write!(f, " effects={:?}", kind.effects().kind)
            }
            Self::WideCompareExchange(data) => {
                let WideCmpXchgData {
                    wide,
                    mnemonic,
                    metadata,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(
                    f,
                    "{} {mnemonic}",
                    if *wide {
                        "atomic.cmpxchg16b"
                    } else {
                        "atomic.cmpxchg8b"
                    }
                )?;
                write_inputs(f, inputs)?;
                write_native_metadata(f, metadata.as_ref())?;
                write!(f, " effects=Atomic")
            }
            Self::FpTranscendental(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "fp.transcendental.{kind:?}")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::FpuControl(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "fpu.control.{kind:?}")?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::CmpXchg {
                dest,
                addr,
                expected,
                desired,
            } => write!(f, "{dest} = cmpxchg {addr}, {expected}, {desired}"),
            Self::AtomicRmw {
                dest,
                addr,
                value,
                op,
            } => write!(f, "{dest} = atomicrmw.{op} {addr}, {value}"),
            Self::AtomicLoad {
                dest,
                addr,
                value_type,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{dest} = atomicload{volatile}.{ordering}.{width} {value_type}, {addr}"
                )
            }
            Self::AtomicStore {
                addr,
                value,
                value_type,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "atomicstore{volatile}.{ordering}.{width} {value_type}, {addr}, {value}"
                )
            }
            Self::AtomicStoreConditional {
                status,
                addr,
                value,
                value_type,
                success_ordering,
                failure_ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{status} = atomicstore.conditional{volatile}.{success_ordering}/{failure_ordering}.{width} {value_type}, {addr}, {value}"
                )
            }
            Self::AtomicPairLoad {
                first,
                second,
                addr,
                first_type,
                second_type,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{first}, {second} = atomicload.pair{volatile}.{ordering}.{width} {first_type}/{second_type}, {addr}"
                )
            }
            Self::AtomicPairStoreConditional {
                status,
                addr,
                first_value,
                second_value,
                first_type,
                second_type,
                success_ordering,
                failure_ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{status} = atomicstore.conditional.pair{volatile}.{success_ordering}/{failure_ordering}.{width} {first_type}/{second_type}, {addr}, {first_value}, {second_value}"
                )
            }
            Self::AtomicExchange {
                dest,
                addr,
                value,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{dest} = atomicxchg{volatile}.{ordering}.{width} {addr}, {value}"
                )
            }
            Self::AtomicLockRmw {
                dest,
                addr,
                value,
                op,
                ordering,
                width,
                volatile,
            } => {
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{dest} = lock.atomicrmw{volatile}.{op}.{ordering}.{width} {addr}, {value}"
                )
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
                volatile,
            } => {
                write!(f, "{old}")?;
                if let Some(success) = success {
                    write!(f, ", {success}")?;
                }
                let weak = if *weak { ".weak" } else { "" };
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    " = cmpxchg{weak}{volatile}.{success_ordering}/{failure_ordering}.{width} {addr}, {expected}, {desired}"
                )
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
                volatile,
            } => {
                let weak = if *weak { ".weak" } else { "" };
                let volatile = if *volatile { ".volatile" } else { "" };
                write!(
                    f,
                    "{old_first}, {old_second} = cmpxchg.pair{weak}{volatile}.{success_ordering}/{failure_ordering}.{width} {addr}, {expected_first}, {expected_second}, {desired_first}, {desired_second}"
                )
            }
            Self::InitObj {
                dest_addr,
                value_type,
            } => write!(f, "initobj {value_type}, {dest_addr}"),
            Self::CopyObj {
                dest_addr,
                src_addr,
                value_type,
            } => write!(f, "cpobj {value_type}, {dest_addr}, {src_addr}"),
            Self::LoadObj {
                dest,
                src_addr,
                value_type,
            } => write!(f, "{dest} = ldobj {value_type}, {src_addr}"),
            Self::StoreObj {
                dest_addr,
                value,
                value_type,
            } => write!(f, "stobj {value_type}, {dest_addr}, {value}"),
            Self::LocalAlloc { dest, size } => write!(f, "{dest} = localloc {size}"),
            Self::Constrained { constraint_type } => {
                write!(f, "constrained. {constraint_type}")
            }
            Self::Volatile => write!(f, "volatile."),
            Self::Unaligned { alignment } => write!(f, "unaligned. {alignment}"),
            Self::TailPrefix => write!(f, "tail."),
            Self::Readonly => write!(f, "readonly."),
            Self::Ckfinite { dest, operand } => write!(f, "{dest} = ckfinite {operand}"),
            Self::FpClassify { dest, operand } => write!(f, "{dest} = fpclassify {operand}"),
            Self::Nop => write!(f, "nop"),
            Self::Break(op) => write!(f, "{}", op.mnemonic()),
            Self::Phi { dest, operands } => {
                write!(f, "{dest} = phi(")?;
                for (i, (block, var)) in operands.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "B{block}: {var}")?;
                }
                write!(f, ")")
            }
        }
    }
}
