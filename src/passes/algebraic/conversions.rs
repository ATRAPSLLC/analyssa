//! Rules for conversions and reshapes: a chain read past, a reshape collapsed,
//! a flag read routed to its source, and a narrowing that keeps only the bits it
//! needs.

use crate::{
    analysis::{
        convert::{collapse_conversion_chain, drop_discarded_disjunct, narrow_truncated_operation},
        flags::{self, FlagSource},
        reshape,
    },
    ir::ops::SsaOp,
    passes::algebraic::{Simplification, SimplificationCandidate, site::Site},
    target::Target,
};

/// Wraps [`collapse_conversion_chain`] as a simplification candidate.
///
/// The rule itself - which conversions may be read past, and how far - lives in
/// [`crate::analysis::convert`]. This only decides how the pass spells the
/// answer: a conversion to the operand's own type becomes a `Copy`, and a
/// collapsed chain becomes a `Rewrite` of the conversion in place.
pub fn convert_chain<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { ssa, op, .. } = *site;
    let dest = op.dest()?;
    let (simplification, description) = match collapse_conversion_chain(ssa, op)? {
        SsaOp::Copy { src, .. } => (
            Simplification::Copy(src),
            "conversion to the operand's own type",
        ),
        collapsed => (
            Simplification::Rewrite(Box::new(collapsed)),
            "convert chain → single convert",
        ),
    };
    Some(site.candidate(dest, simplification, description))
}

/// A flag read of a composed flags value reads the input its bits came
/// from - a flags value, or the boolean one bit was set from. See
/// [`flags::route`].
pub fn flag_read<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { ssa, op, .. } = *site;
    let dest = site.dest?;
    if let SsaOp::ReadFlags { flags, mask, .. } = op
        && let Some(source) = flags::route(ssa, *flags, *mask)
    {
        let simplification = match source {
            FlagSource::Flags(flags) => Simplification::Rewrite(Box::new(SsaOp::ReadFlags {
                dest,
                flags,
                mask: *mask,
            })),
            FlagSource::Bit(bit) => Simplification::Copy(bit),
            // A chosen read is three operations, the reads and their select,
            // which `flags::fold_result_flag_reads` emits.
            FlagSource::Chosen { .. } => return None,
        };
        return Some(site.candidate(dest, simplification, "flag read through its composition"));
    }
    None
}

/// A reshape reads straight past every reshape feeding it: a value that
/// crossed a register arrives wrapped in the reinterprets and shuffles of
/// the register's write and read, and none of them moves a bit. See
/// [`reshape::collapse_reshape`].
pub fn reshape_chain<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { ssa, op, .. } = *site;
    let dest = site.dest?;
    if let Some(collapsed) = reshape::collapse_reshape(ssa, op) {
        let (simplification, description) = match collapsed {
            SsaOp::Copy { src, .. } => (Simplification::Copy(src), "reshape back to its own bits"),
            collapsed => (
                Simplification::Rewrite(Box::new(collapsed)),
                "reshape chain → single reshape",
            ),
        };
        return Some(site.candidate(dest, simplification, description));
    }
    None
}

/// A narrowing conversion reads only the disjuncts whose bits it keeps. See
/// [`drop_discarded_disjunct`]: the byte-insert idiom a sub-register write
/// lowers to otherwise leaves the narrow value looking as wide as the
/// register it was inserted into.
pub fn discarded_disjunct<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { ssa, op, .. } = *site;
    let dest = site.dest?;
    if let Some(rewritten) = drop_discarded_disjunct(ssa, op) {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(rewritten)),
            "narrowing convert drops a discarded disjunct",
        ));
    }
    None
}

/// A narrowing conversion of an operation on values extended from the
/// narrow width is the operation at that width. See
/// [`narrow_truncated_operation`]: a 64-bit machine computing a 32-bit
/// expression partly at its register's width otherwise leaves the one
/// expression at two widths, which the idiom rules below cannot match.
pub fn narrow_truncated<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, machine, ..
    } = *site;
    let dest = site.dest?;
    let ptr_size = machine.ptr_size;
    if let Some(narrowed) = narrow_truncated_operation(ssa, op, ptr_size) {
        return Some(site.candidate(
            dest,
            if narrowed.inserted.is_empty() {
                Simplification::Rewrite(Box::new(narrowed.op))
            } else {
                Simplification::Narrowed(Box::new(narrowed.op), narrowed.inserted, narrowed.ty)
            },
            "narrowing convert computes its operation at the narrow width",
        ));
    }
    None
}
