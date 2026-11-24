//! Human-readable rendering of [`SsaOp`].
//!
//! The format is diagnostic output - verifier messages, pass logs, test
//! assertions - not a parseable serialization; use the `serde` feature for
//! round-tripping.

use std::fmt::{self, Write};

use crate::{
    ir::{
        float::{FloatExceptionLayout, FloatExceptions},
        ops::{
            calls::{AccessMode, CallData, CallLocation, CallTermination, Callee, FrameOrigin},
            def::SsaOp,
            kinds::{CompareBlkData, KindedVecData, NativeInstructionMetadata, NativeKindedData},
            vector::{
                Contraction, MatrixLayout, ReduceGroups, VectorElementCountData,
                VectorExtendInLaneData, VectorPredicateWhileData, VectorStructLoadReplicateData,
                VectorSveComputeData,
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

/// Streams text after discarding one exact ASCII definition prefix.
///
/// A mismatch flushes the partial prefix before forwarding the remaining text.
/// Only the bounded SSA identifier prefix is retained, never the operation text.
struct PrefixWriter<'a, 'b> {
    /// The caller's diagnostic formatter.
    inner: &'a mut fmt::Formatter<'b>,
    /// The definition prefix that may be elided.
    prefix: &'a str,
    /// Prefix bytes seen so far, all ASCII and therefore character boundaries.
    matched: usize,
    /// Whether subsequent text can be forwarded immediately.
    passthrough: bool,
}

impl PrefixWriter<'_, '_> {
    /// Flushes an incomplete prefix when the operation ended before matching it.
    fn finish(&mut self) -> fmt::Result {
        if !self.passthrough && self.matched != 0 {
            self.inner.write_str(&self.prefix[..self.matched])?;
        }
        Ok(())
    }
}

impl fmt::Write for PrefixWriter<'_, '_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.passthrough {
            return self.inner.write_str(text);
        }
        let matched = self.prefix[self.matched..]
            .bytes()
            .zip(text.bytes())
            .take_while(|(expected, actual)| expected == actual)
            .count();
        self.matched = self.matched.checked_add(matched).ok_or(fmt::Error)?;
        if matched == text.len() && self.matched < self.prefix.len() {
            return Ok(());
        }
        self.passthrough = true;
        if self.matched < self.prefix.len() && self.matched != 0 {
            self.inner.write_str(&self.prefix[..self.matched])?;
        }
        if matched == text.len() {
            Ok(())
        } else {
            self.inner.write_str(&text[matched..])
        }
    }
}

/// Writes one raise's flags and optional definition prefix without visiting its child.
fn write_raise_head(
    f: &mut fmt::Formatter<'_>,
    dest: Option<SsaVarId>,
    raises: FloatExceptions,
    layout: FloatExceptionLayout,
) -> fmt::Result {
    if let Some(dest) = dest {
        write!(f, "{dest} = ")?;
    }
    write!(f, "float.raise")?;
    for (index, (name, bit)) in FloatExceptions::NAMES.iter().zip(layout.bits()).enumerate() {
        let flag = FloatExceptions::from_bits(1 << index);
        if let Some(bit) = bit
            && raises.contains(flag)
        {
            write!(f, " {name}@{bit}")?;
        }
    }
    Ok(())
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
    /// Renders one operation in the `dest = mnemonic operands` diagnostic form.
    ///
    /// One match arm per [`SsaOp`] variant, since the rendered mnemonic and the
    /// order and spelling of operands are per-variant facts with no structural
    /// shortcut. The compact arithmetic and control variants inline their whole
    /// line; the boxed-payload variants (native, vector, and the wider mixed
    /// forms) route their shared pieces through `write_defs`, `write_inputs`,
    /// and `write_native_metadata`, and every memory access its
    /// [`AccessQualifiers`](crate::ir::ops::kinds::AccessQualifiers),
    /// so the definition prefix, operand list, provenance suffix, and space qualifier stay
    /// spelled the same across every arm that has them. Optional modifiers
    /// (`flags=`, the `.ovf`/`.un`/`.signaling` suffixes) are appended only when
    /// present, so an operation with none renders no trailing noise.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_operation(f)?;
        // The environment's rounding mode an operation rounds by, which it
        // reads beside its operands.
        match self.environment() {
            Some(mode) => write!(f, " mode={mode}"),
            None => Ok(()),
        }
    }
}

impl<T: Target> SsaOp<T>
where
    T::TypeRef: fmt::Display,
    T::MethodRef: fmt::Display,
    T::FieldRef: fmt::Display,
    T::SymbolRef: fmt::Display,
    T::SigRef: fmt::Display,
    T::Type: fmt::Display,
{
    /// The operation's line without the environment's rounding mode it
    /// reads ([`fmt::Display::fmt`]).
    fn fmt_operation(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VectorSlice {
                dest,
                vector,
                offset,
                length,
                target_type,
            } => write!(
                f,
                "{dest} = vslice.{target_type} {vector}, {offset}, {length}"
            ),
            Self::VectorInsertSlice {
                dest,
                vector,
                offset,
                value,
                target_type,
            } => write!(
                f,
                "{dest} = vinsert.slice.{target_type} {vector}, {offset}, {value}"
            ),
            Self::VectorConcat {
                dest,
                left,
                right,
                target_type,
            } => write!(f, "{dest} = vconcat.{target_type} {left}, {right}"),
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
            Self::BSwap { dest, src } => write!(f, "{dest} = bswap {src}"),
            Self::BRev { dest, src } => write!(f, "{dest} = brev {src}"),
            Self::BitScanForward { dest, src } => write!(f, "{dest} = bsf {src}"),
            Self::BitScanReverse { dest, src } => write!(f, "{dest} = bsr {src}"),
            Self::Popcount { dest, src } => write!(f, "{dest} = popcnt {src}"),
            Self::Parity { dest, src } => write!(f, "{dest} = parity {src}"),
            Self::Ceq {
                dest,
                left,
                right,
                signaling,
            } => {
                let signal = if *signaling { ".signaling" } else { "" };
                write!(f, "{dest} = ceq{signal} {left}, {right}")
            }
            Self::Clt {
                dest,
                left,
                right,
                unsigned,
                signaling,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                let signal = if *signaling { ".signaling" } else { "" };
                write!(f, "{dest} = clt{suffix}{signal} {left}, {right}")
            }
            Self::Cgt {
                dest,
                left,
                right,
                unsigned,
                signaling,
            } => {
                let suffix = if *unsigned { ".un" } else { "" };
                let signal = if *signaling { ".signaling" } else { "" };
                write!(f, "{dest} = cgt{suffix}{signal} {left}, {right}")
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
            // CIL's spelling: `.ovf` checks the result fits, `.un` reads the
            // integer operand unsigned. An `IntConv` from a narrower source
            // zero-extends where it is `.un` and sign-extends where not.
            Self::IntConv {
                dest,
                operand,
                target,
                overflow_check,
                unsigned,
            } => write!(
                f,
                "{dest} = conv{}.{target}{} {operand}",
                if *overflow_check { ".ovf" } else { "" },
                if *unsigned { ".un" } else { "" }
            ),
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
                unsigned,
                rounding,
                ..
            } => {
                write!(f, "{dest} = inttofloat.{target}")?;
                if *unsigned {
                    write!(f, ".un")?;
                }
                if let Some(rounding) = rounding {
                    write!(f, ".{rounding}")?;
                }
                write!(f, " {operand}")
            }
            Self::FloatToInt {
                dest,
                operand,
                target,
                overflow_check,
                unsigned,
                ..
            } => write!(
                f,
                "{dest} = floattoint{}.{target}{} {operand}",
                if *overflow_check { ".ovf" } else { "" },
                if *unsigned { ".un" } else { "" }
            ),
            Self::FloatConv {
                dest,
                operand,
                target,
                rounding,
                ..
            } => {
                write!(f, "{dest} = fconv.{target}")?;
                if let Some(rounding) = rounding {
                    write!(f, ".{rounding}")?;
                }
                write!(f, " {operand}")
            }
            Self::Bitcast {
                dest,
                operand,
                target,
            } => write!(f, "{dest} = bitcast.{target} {operand}"),
            Self::ReadFlags { dest, flags, mask } => {
                write!(f, "{dest} = readflags {flags}, {mask}")
            }
            Self::MergeFlags {
                dest,
                base,
                update,
                mask,
            } => write!(f, "{dest} = mergeflags {base}, {update}, {mask}"),
            Self::InsertFlag {
                dest,
                base,
                bit,
                mask,
            } => write!(f, "{dest} = insertflag {base}, {bit}, {mask}"),
            Self::FloatRaise {
                dest,
                op,
                lanes,
                raises,
                layout,
            } => {
                write_raise_head(f, Some(*dest), *raises, *layout)?;
                write!(f, " (")?;
                let mut parent = *dest;
                let mut operation = op.as_ref();
                let mut masks = Vec::new();
                while let Self::FloatRaise {
                    dest,
                    op,
                    lanes,
                    raises,
                    layout,
                } = operation
                {
                    write_raise_head(f, (*dest != parent).then_some(*dest), *raises, *layout)?;
                    write!(f, " (")?;
                    masks.push(*lanes);
                    parent = *dest;
                    operation = op;
                }
                // Stream the leaf once, stripping only its exact parent prefix.
                // Neither a full subtree string nor recursive formatting is needed.
                let prefix = format!("{parent} = ");
                let mut writer = PrefixWriter {
                    inner: f,
                    prefix: &prefix,
                    matched: 0,
                    passthrough: false,
                };
                write!(&mut writer, "{operation}")?;
                writer.finish()?;
                for mask in masks.into_iter().rev() {
                    write!(f, ")")?;
                    if let Some(mask) = mask {
                        write!(f, " lanes={mask}")?;
                    }
                }
                write!(f, ")")?;
                if let Some(lanes) = lanes {
                    write!(f, " lanes={lanes}")?;
                }
                Ok(())
            }
            Self::CallClobber { outputs, machine } => {
                write_defs(f, outputs)?;
                write!(f, "call.clobber")?;
                if let Some(machine) = machine.as_deref().filter(|values| !values.is_empty()) {
                    write!(f, " [machine ")?;
                    for (index, output) in machine.iter().enumerate() {
                        if index > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}:#{}", output.key.get(), output.index)?;
                    }
                    write!(f, "]")?;
                }
                Ok(())
            }
            Self::VectorUnary {
                dest,
                value,
                kind,
                element,
                ..
            } => {
                let rounding = element.rounding_suffix();
                write!(f, "{dest} = vunary.{kind:?}{rounding} {value}")
            }
            Self::VectorBinary {
                dest,
                left,
                right,
                kind,
                element,
                ..
            } => {
                let signedness = element.signedness_suffix();
                let rounding = element.rounding_suffix();
                write!(
                    f,
                    "{dest} = vbinary.{kind:?}{signedness}{rounding} {left}, {right}"
                )
            }
            Self::VectorTernary {
                dest,
                first,
                second,
                third,
                kind,
                element,
                ..
            } => {
                let signedness = element.signedness_suffix();
                let rounding = element.rounding_suffix();
                write!(
                    f,
                    "{dest} = vternary.{kind:?}{signedness}{rounding} {first}, {second}, {third}"
                )
            }
            Self::VectorPredicatedUnary {
                dest,
                value,
                mask,
                passthrough,
                kind,
                mode,
                ..
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
                element,
                mode,
                ..
            } => {
                let signedness = element.signedness_suffix();
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vbinary.pred.{kind:?}{signedness}.{mode:?} {left}, {right}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vbinary.pred.{kind:?}{signedness}.{mode:?} {left}, {right}, {mask}"
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
                element,
                mode,
                ..
            } => {
                let signedness = element.signedness_suffix();
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vternary.pred.{kind:?}{signedness}.{mode:?} {first}, {second}, {third}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vternary.pred.{kind:?}{signedness}.{mode:?} {first}, {second}, {third}, {mask}"
                    )
                }
            }
            Self::VectorCompare {
                dest,
                left,
                right,
                kind,
                element,
                signaling,
            } => {
                let float = if element.kind.is_float() { ".f" } else { "" };
                let signedness = element.signedness_suffix();
                let signaling = if *signaling { ".signaling" } else { "" };
                write!(
                    f,
                    "{dest} = vcmp.{kind:?}{float}{signedness}{signaling} {left}, {right}"
                )
            }
            Self::VectorLoad {
                dest,
                addr,
                vector_type,
                access: space,
            } => write!(f, "{dest} = vload.{vector_type} {space}{addr}"),
            Self::VectorStore {
                addr,
                value,
                vector_type,
                access: space,
            } => write!(f, "vstore.{vector_type} {space}{addr}, {value}"),
            Self::VectorMaskedLoad {
                dest,
                addr,
                mask,
                passthrough,
                vector_type,
                mode,
                access: space,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vload.masked.{mode:?}.{vector_type} {space}{addr}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vload.masked.{mode:?}.{vector_type} {space}{addr}, {mask}"
                    )
                }
            }
            Self::VectorMaskedStore {
                addr,
                value,
                mask,
                vector_type,
                access: space,
            } => write!(
                f,
                "vstore.masked.{vector_type} {space}{addr}, {value}, {mask}"
            ),
            Self::VectorBroadcastLoad {
                dest,
                addr,
                vector_type,
                access: space,
            } => write!(f, "{dest} = vbroadcast.load.{vector_type} {space}{addr}"),
            Self::VectorGather {
                dest,
                base,
                indices,
                mask,
                passthrough,
                vector_type,
                mode,
                access: space,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vgather.{mode:?}.{vector_type} {space}{base}, {indices}, {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vgather.{mode:?}.{vector_type} {space}{base}, {indices}, {mask}"
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
                access: space,
            } => match (fault, mask, passthrough) {
                (Some(fault), Some(mask), Some(passthrough)) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {space}{addr}, {mask}, {passthrough}"
                ),
                (Some(fault), Some(mask), None) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {space}{addr}, {mask}"
                ),
                (Some(fault), None, Some(passthrough)) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {space}{addr}, {passthrough}"
                ),
                (Some(fault), None, None) => write!(
                    f,
                    "{dest}, {fault} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {space}{addr}"
                ),
                (None, Some(mask), Some(passthrough)) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {space}{addr}, {mask}, {passthrough}"
                ),
                (None, Some(mask), None) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {space}{addr}, {mask}"
                ),
                (None, None, Some(passthrough)) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {space}{addr}, {passthrough}"
                ),
                (None, None, None) => write!(
                    f,
                    "{dest} = vload.faulting.{fault_mode:?}.{mask_mode:?}.{vector_type} {space}{addr}"
                ),
            },
            Self::VectorSegmentLoad {
                dests,
                base,
                mask,
                vector_type,
                segments,
                layout,
                access: space,
            } => {
                write_defs(f, dests)?;
                if let Some(mask) = mask {
                    write!(
                        f,
                        "vload.segment.{layout:?}.{segments}.{vector_type} {space}{base}, {mask}"
                    )
                } else {
                    write!(
                        f,
                        "vload.segment.{layout:?}.{segments}.{vector_type} {space}{base}"
                    )
                }
            }
            Self::VectorScatter {
                base,
                indices,
                value,
                mask,
                vector_type,
                access: space,
            } => write!(
                f,
                "vscatter.{vector_type} {space}{base}, {indices}, {value}, {mask}"
            ),
            Self::VectorSegmentStore {
                base,
                values,
                mask,
                vector_type,
                segments,
                layout,
                access: space,
            } => {
                if let Some(mask) = mask {
                    write!(
                        f,
                        "vstore.segment.{layout:?}.{segments}.{vector_type} {space}{base}, {values:?}, {mask}"
                    )
                } else {
                    write!(
                        f,
                        "vstore.segment.{layout:?}.{segments}.{vector_type} {space}{base}, {values:?}"
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
                dest,
                left,
                right,
                selector,
            } => {
                write!(f, "{dest} = vshuffle {left}")?;
                if let Some(right) = right {
                    write!(f, ", {right}")?;
                }
                match selector.index() {
                    Some(index) => write!(f, " by {index}"),
                    None => Ok(()),
                }
            }
            Self::VectorCast {
                dest,
                value,
                target_type,
                from,
                to,
                rounding,
                saturate,
                ..
            } => {
                write!(f, "{dest} = vcast.{from:?}.{to:?}.{target_type}")?;
                if let Some(rounding) = rounding {
                    write!(f, ".{rounding}")?;
                }
                if *saturate {
                    write!(f, ".sat")?;
                }
                write!(f, " {value}")
            }
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
                access: space,
            } => {
                if let Some(passthrough) = passthrough {
                    write!(
                        f,
                        "{dest} = vpack.load.{kind:?}.{mode:?}.e{element_bits}.{vector_type} {space}[{addr}], {mask}, {passthrough}"
                    )
                } else {
                    write!(
                        f,
                        "{dest} = vpack.load.{kind:?}.{mode:?}.e{element_bits}.{vector_type} {space}[{addr}], {mask}"
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
                access: space,
            } => write!(
                f,
                "vpack.store.{kind:?}.e{element_bits}.{vector_type} {space}[{addr}], {value}, {mask}"
            ),
            Self::VectorReduce {
                dest,
                value,
                kind,
                element,
                initial,
                ordered,
                groups,
                ..
            } => {
                let signedness = element.signedness_suffix();
                let rounding = element.rounding_suffix();
                let order = if *ordered { ".ordered" } else { "" };
                write!(f, "{dest} = vreduce.{kind:?}{signedness}{rounding}{order}")?;
                if let Some(ReduceGroups { lanes, bits }) = groups {
                    write!(f, ".groups{lanes}.i{bits}")?;
                }
                write!(f, " {value}")?;
                match initial {
                    Some(initial) => write!(f, " from {initial}"),
                    None => Ok(()),
                }
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
                state,
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
                if !state.is_empty() {
                    write!(f, " [state ")?;
                    for (index, input) in state.iter().enumerate() {
                        if index > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}:{}", input.key.get(), input.value)?;
                    }
                    write!(f, "]")?;
                }
                Ok(())
            }
            Self::Return {
                value,
                state,
                results,
                transfer,
            } => {
                let target = transfer.as_ref().and_then(|transfer| transfer.target);
                let handover = transfer
                    .as_ref()
                    .map_or(&[][..], |transfer| &transfer.handover[..]);
                f.write_str("ret")?;
                if let Some(v) = value {
                    write!(f, " {v}")?;
                }
                if let Some((first, rest)) = state.split_first() {
                    write!(f, " [state {first}")?;
                    for held in rest {
                        write!(f, ", {held}")?;
                    }
                    f.write_str("]")?;
                }
                if !results.is_empty() {
                    f.write_str(" [results ")?;
                    for (index, result) in results.iter().enumerate() {
                        if index > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{}:{}", result.key.get(), result.value)?;
                    }
                    f.write_str("]")?;
                }
                if let Some(target) = target {
                    write!(f, " [through {target}]")?;
                }
                if !handover.is_empty() {
                    f.write_str(" [handover ")?;
                    for (index, input) in handover.iter().enumerate() {
                        if index > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{}:{}", input.key.get(), input.value)?;
                    }
                    f.write_str("]")?;
                }
                Ok(())
            }
            Self::LoadField {
                dest,
                object,
                field,
                access,
            } => {
                write!(f, "{dest} = ldfld {field}, {access}{object}")
            }
            Self::StoreField {
                object,
                field,
                value,
                access,
            } => write!(f, "stfld {field}, {access}{object}, {value}"),
            Self::LoadStaticField {
                dest,
                field,
                access,
            } => write!(f, "{dest} = ldsfld {access}{field}"),
            Self::StoreStaticField {
                field,
                value,
                access,
            } => write!(f, "stsfld {access}{field}, {value}"),
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
                dest,
                array,
                index,
                readonly,
                ..
            } => {
                let readonly = if *readonly { ".readonly" } else { "" };
                write!(f, "{dest} = ldelema{readonly} {array}[{index}]")
            }
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
                access: space,
            } => write!(f, "{dest} = ldind.{value_type} {space}{addr}"),
            Self::StoreIndirect {
                addr,
                value,
                value_type,
                access: space,
            } => write!(f, "stind.{value_type} {space}{addr}, {value}"),
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
            Self::ArgList { dest } => write!(f, "{dest} = arglist"),
            Self::ArgNext { dest, list, ty } => write!(f, "{dest} = arg.next.{ty} {list}"),
            Self::ArgStep {
                dest,
                next,
                list,
                ty,
            } => write!(f, "{dest}, {next} = arg.step.{ty} {list}"),
            Self::MakeTypedRef {
                dest,
                address,
                value_type,
            } => write!(f, "{dest} = mkrefany {value_type}, {address}"),
            Self::TypedRefValue {
                dest,
                typed_ref,
                value_type,
            } => write!(f, "{dest} = refanyval {value_type}, {typed_ref}"),
            Self::TypedRefType { dest, typed_ref } => write!(f, "{dest} = refanytype {typed_ref}"),
            Self::Call(call) => write!(f, "{call}"),
            Self::LoadFunctionPtr {
                dest,
                method,
                constraint: None,
            } => write!(f, "{dest} = ldftn {method}"),
            Self::LoadFunctionPtr {
                dest,
                method,
                constraint: Some(constraint),
            } => write!(f, "{dest} = constrained({constraint}) ldftn {method}"),
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
            Self::RuntimeValue { dest, kind } => write!(f, "{dest} = runtime {kind}"),
            Self::Leave { target } => write!(f, "leave B{target}"),
            Self::InitBlk {
                dest_addr,
                value,
                size,
                backward,
                access: space,
            } => {
                write!(f, "initblk {space}{dest_addr}, {value}, {size}")?;
                match backward {
                    Some(flag) => write!(f, " backward={flag}"),
                    None => Ok(()),
                }
            }
            Self::CopyBlk {
                dest_addr,
                src_addr,
                size,
                element_bytes,
                backward,
                src_access: src_space,
                dest_access: dest_space,
            } => {
                write!(f, "cpblk")?;
                if *element_bytes != 1 {
                    write!(f, ".{element_bytes}")?;
                }
                write!(f, " {dest_space}{dest_addr}, {src_space}{src_addr}, {size}")?;
                match backward {
                    Some(flag) => write!(f, " backward={flag}"),
                    None => Ok(()),
                }
            }
            Self::MoveBlk {
                dest_addr,
                src_addr,
                size,
                src_access: src_space,
                dest_access: dest_space,
            } => write!(
                f,
                "mvblk {dest_space}{dest_addr}, {src_space}{src_addr}, {size}"
            ),
            Self::Fence { kind } => write!(f, "fence {kind}"),
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
            Self::Compute(data) => {
                write_defs(f, &data.outputs)?;
                write!(f, "{}", data.kind)?;
                write_inputs(f, &data.inputs)
            }
            Self::Contract(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "contract.{}", kind.name())?;
                write_inputs(f, inputs)?;
                match kind {
                    Contraction::Dot(shape) => {
                        let sign = |signed: bool| if signed { 'i' } else { 'u' };
                        write!(
                            f,
                            " {}{}x{}{} -> {}",
                            sign(shape.signed_a),
                            shape.source_bits,
                            sign(shape.signed_b),
                            shape.source_bits,
                            shape.dest_bits
                        )?;
                        if shape.accumulates {
                            write!(f, " acc")?;
                        }
                        match shape.saturates {
                            Some(true) => write!(f, " sat.i"),
                            Some(false) => write!(f, " sat.u"),
                            None => Ok(()),
                        }
                    }
                    Contraction::MaskedDot {
                        control,
                        element_bits,
                    } => write!(f, " e{element_bits} control={control:#04x}"),
                    Contraction::Matrix(shape) => {
                        write!(f, " {}", shape.elements)?;
                        match shape.layout {
                            MatrixLayout::Segments { segment_bits } => {
                                write!(f, " seg{segment_bits}")
                            }
                            MatrixLayout::Tiles => write!(f, " tiles"),
                        }
                    }
                    Contraction::Outer(shape) => {
                        write!(f, " {}", shape.elements)?;
                        match shape.selection {
                            Some(selection) => write!(
                                f,
                                " sparse{}of{}{}",
                                shape.elements.group(),
                                u32::from(selection.values)
                                    .saturating_mul(u32::from(selection.candidates)),
                                if selection.per_value { ".each" } else { "" }
                            ),
                            None => Ok(()),
                        }
                    }
                    Contraction::Complex(shape) => {
                        write!(
                            f,
                            " e{}{}{}",
                            shape.element.bits,
                            shape.element.signedness_suffix(),
                            shape.element.rounding_suffix()
                        )?;
                        if shape.saturates {
                            write!(f, " sat")?;
                        }
                        Ok(())
                    }
                    Contraction::FloatDot { elements, .. } => write!(f, " {elements}"),
                    Contraction::MultiplyAdd52 { .. } => Ok(()),
                }
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
                    pattern,
                    streaming,
                    outputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.cnt e{element_bits} {pattern} x{multiplier}")?;
                if let Some(streaming) = streaming {
                    write!(f, " streaming={streaming}")?;
                }
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
            Self::VectorSveCompute(data) => {
                let VectorSveComputeData {
                    outputs, inputs, ..
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "vector.sve.compute")?;
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
            Self::CompareBlk(data) => {
                let CompareBlkData {
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
                } = data.as_ref();
                write!(
                    f,
                    "{remaining}, {left_next}, {right_next}, {flags} = compare.blk.{element_bytes}.{} {left_access}{left}, {right_access}{right}, {count}",
                    if *while_equal { "eq" } else { "ne" }
                )?;
                if let Some(flags_in) = flags_in {
                    write!(f, ", {flags_in}")?;
                }
                match backward {
                    Some(flag) => write!(f, " backward={flag}"),
                    None => Ok(()),
                }
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
                write!(
                    f,
                    "{remaining}, {next}, {flags} = scan.blk.{element_bytes}.{} {access}{address}, {value}, {count}",
                    if *while_equal { "eq" } else { "ne" }
                )?;
                if let Some(flags_in) = flags_in {
                    write!(f, ", {flags_in}")?;
                }
                match backward {
                    Some(flag) => write!(f, " backward={flag}"),
                    None => Ok(()),
                }
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
                write!(
                    f,
                    "{value}, {next} = load.blk.{element_bytes} {access}{address}, {count}"
                )?;
                if let Some(value_in) = value_in {
                    write!(f, ", {value_in}")?;
                }
                match backward {
                    Some(flag) => write!(f, " backward={flag}"),
                    None => Ok(()),
                }
            }
            Self::FloatFunction(data) => {
                let KindedVecData {
                    kind,
                    outputs,
                    inputs,
                } = data.as_ref();
                write_defs(f, outputs)?;
                write!(f, "float.{}", kind.name())?;
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
                write!(f, "{}", kind.name())?;
                write_inputs(f, inputs)?;
                Ok(())
            }
            Self::AtomicLoad {
                dest,
                addr,
                value_type,
                ordering,
                width,
                reserves,
                access: space,
            } => {
                let reserves = if *reserves { ".reserve" } else { "" };
                write!(
                    f,
                    "{dest} = atomicload{reserves}.{ordering}.{width} {value_type}, {space}{addr}"
                )
            }
            Self::AtomicStore {
                addr,
                value,
                value_type,
                ordering,
                width,
                access: space,
            } => {
                write!(
                    f,
                    "atomicstore.{ordering}.{width} {value_type}, {space}{addr}, {value}"
                )
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
                access: space,
            } => {
                let consume = if *consumes_reservation {
                    ".consume"
                } else {
                    ""
                };
                write!(
                    f,
                    "{status} = atomicstore.conditional.{success_ordering}/{failure_ordering}.{width}.status({status_encoding:?}){consume} {value_type}, {space}{addr}, {value}"
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
                access: space,
            } => {
                write!(
                    f,
                    "{first}, {second} = atomicload.pair.{ordering}.{width} {first_type}/{second_type}, {space}{addr}"
                )
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
                access: space,
            } => {
                let consume = if *consumes_reservation {
                    ".consume"
                } else {
                    ""
                };
                write!(
                    f,
                    "{status} = atomicstore.conditional.pair.{success_ordering}/{failure_ordering}.{width}.status({status_encoding:?}){consume} {first_type}/{second_type}, {space}{addr}, {first_value}, {second_value}"
                )
            }
            Self::AtomicExchange {
                dest,
                addr,
                value,
                ordering,
                width,
                access: space,
            } => {
                write!(
                    f,
                    "{dest} = atomicxchg.{ordering}.{width} {space}{addr}, {value}"
                )
            }
            Self::AtomicLockRmw {
                dest,
                addr,
                value,
                op,
                ordering,
                width,
                access: space,
            } => {
                write!(
                    f,
                    "{dest} = lock.atomicrmw.{op}.{ordering}.{width} {space}{addr}, {value}"
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
                access: space,
            } => {
                write!(f, "{old}")?;
                if let Some(success) = success {
                    write!(f, ", {success}")?;
                }
                let weak = if *weak { ".weak" } else { "" };
                write!(
                    f,
                    " = cmpxchg{weak}.{success_ordering}/{failure_ordering}.{width} {space}{addr}, {expected}, {desired}"
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
                access: space,
            } => {
                let weak = if *weak { ".weak" } else { "" };
                write!(
                    f,
                    "{old_first}, {old_second} = cmpxchg.pair{weak}.{success_ordering}/{failure_ordering}.{width} {space}{addr}, {expected_first}, {expected_second}, {desired_first}, {desired_second}"
                )
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
                access: space,
            } => {
                write!(
                    f,
                    "{old} = atomicrmw.cond.{op}.{condition}.{ordering}.{width} {space}{addr}, {compare}, {value}"
                )
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
                access: space,
            } => {
                write!(
                    f,
                    "{old_first}, {old_second} = atomicrmw.pair.{op}.{ordering}.{width} {space}{addr}, {value_first}, {value_second}"
                )
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
                access: space,
            } => {
                let soft = if rcws_mask.is_some() { ".soft" } else { "" };
                write!(
                    f,
                    "{old}, {status} = atomicrmw.checked.{op}{soft}.{protection}.{ordering}.{width} {space}{addr}, "
                )?;
                if let Some(compare) = compare {
                    write!(f, "{compare}, ")?;
                }
                write!(f, "{value}, {rcw_mask}")?;
                if let Some(rcws_mask) = rcws_mask {
                    write!(f, ", {rcws_mask}")?;
                }
                Ok(())
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
                access,
            } => write!(f, "{dest} = ldobj {value_type}, {access}{src_addr}"),
            Self::StoreObj {
                dest_addr,
                value,
                value_type,
                access,
            } => write!(f, "stobj {value_type}, {access}{dest_addr}, {value}"),
            Self::LocalAlloc { dest, size, zeroed } => {
                let zeroed = if *zeroed { " zeroed" } else { "" };
                write!(f, "{dest} = localloc {size}{zeroed}")
            }
            Self::Ckfinite { dest, operand } => write!(f, "{dest} = ckfinite {operand}"),
            Self::FpClassify {
                dest,
                operand,
                layout,
            } => write!(f, "{dest} = fpclassify.{} {operand}", layout.name()),
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

impl<T: Target> fmt::Display for CallData<T>
where
    T::TypeRef: fmt::Display,
    T::MethodRef: fmt::Display,
    T::FieldRef: fmt::Display,
    T::SigRef: fmt::Display,
{
    /// Renders `dest = [tail ]call m(args)`, `callvirt`, `constrained(T)
    /// callvirt` or `calli fptr(args)`, with the values returned beside the
    /// result after it under their keys (`v1, 2:v4 = call m()`, `_` where only
    /// those are read), the values staged beside the arguments
    /// (`staged[sp+8: v3]`), and the effects the host stated: `[pure]` or the
    /// memory, world and control it names.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.dest, self.results.is_empty()) {
            (Some(dest), true) => write!(f, "{dest} = ")?,
            (None, true) => {}
            (dest, false) => {
                match dest {
                    Some(dest) => write!(f, "{dest}")?,
                    None => f.write_str("_")?,
                }
                for result in &self.results {
                    write!(f, ", {}:{}", result.key.get(), result.value)?;
                }
                f.write_str(" = ")?;
            }
        }
        if self.tail {
            write!(f, "tail ")?;
        }
        match &self.callee {
            Callee::Direct(method) => write!(f, "call {method}(")?,
            Callee::Virtual(method) => write!(f, "callvirt {method}(")?,
            Callee::Constrained {
                method,
                constraint_type,
                instance,
            } => {
                let call = if *instance { "callvirt" } else { "call" };
                write!(f, "constrained({constraint_type}) {call} {method}(")?;
            }
            Callee::Indirect { fptr, .. } => write!(f, "calli {fptr}(")?,
        }
        let (args, effects) = (&self.args, self.effects.as_ref());
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{arg}")?;
        }
        write!(f, ")")?;
        // Staged values are no arguments, and printed apart from them.
        if !self.staged.is_empty() {
            write!(f, " staged[")?;
            for (i, staged) in self.staged.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "sp+{}: {}", staged.offset, staged.value)?;
            }
            write!(f, "]")?;
        }
        if !self.implicit_inputs.is_empty() {
            write!(f, " with {{")?;
            for (index, input) in self.implicit_inputs.iter().enumerate() {
                if index > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}={}", input.key.get(), input.value)?;
            }
            write!(f, "}}")?;
        }
        let Some(effects) = effects else {
            return Ok(());
        };
        if effects.is_pure() {
            return write!(f, " [pure]");
        }
        let mut parts: Vec<String> = Vec::new();
        match effects.accesses() {
            None => parts.push("memory".to_owned()),
            Some(accesses) => {
                for access in accesses {
                    let location = match &access.location {
                        CallLocation::Pointee { arg, .. } => format!("*arg{arg}"),
                        CallLocation::Field(field) => field.to_string(),
                        CallLocation::Escaped => "escaped".to_owned(),
                        CallLocation::Frame {
                            origin: FrameOrigin::Entry,
                            offset,
                            bytes,
                        } => format!("frame[{offset}; {bytes}]"),
                        CallLocation::Frame {
                            origin: FrameOrigin::Alignment,
                            offset,
                            bytes,
                        } => format!("aligned[{offset}; {bytes}]"),
                    };
                    let mode = match access.access {
                        AccessMode::Read => "reads",
                        AccessMode::Write => "writes",
                        AccessMode::MayWrite => "may write",
                        AccessMode::Update => "updates",
                    };
                    parts.push(format!("{mode} {location}"));
                }
            }
        }
        if effects.world {
            parts.push("world".to_owned());
        }
        match effects.termination {
            CallTermination::Returns => {}
            CallTermination::MayNotReturn => parts.push("may not return".to_owned()),
            CallTermination::NeverReturns => parts.push("never returns".to_owned()),
        }
        if effects.returns_twice {
            parts.push("returns twice".to_owned());
        }
        if effects.may_throw {
            parts.push("may throw".to_owned());
        }
        write!(f, " [{}]", parts.join("; "))
    }
}
