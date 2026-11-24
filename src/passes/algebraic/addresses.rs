//! Rules for address arithmetic: a difference or comparison of two addresses of
//! one base, a round trip through an integer, a constant or scaled index, and
//! an address that adds nothing.

use crate::{
    analysis::{address, address::AddressWalk},
    ir::{
        ops::{SsaOp, kinds::CmpKind},
        value::ConstValue,
        variable::VariableRole,
    },
    passes::algebraic::{Simplification, SimplificationCandidate, site::Site},
    target::Target,
};

/// The difference of two addresses measured from one base, when `op` takes
/// one: `&a[4] - &a[0]`, or two symbols' distance apart.
///
/// Address arithmetic is the address model's, not the constants': a symbol
/// folds to no number of its own, so two symbols subtracted fold to nothing
/// under constant evaluation. The address model states both as a base and a
/// displacement, and two addresses from one base - one value, or one absolute
/// space - differ by their displacements, whatever the base turns out to be.
/// That is exact where the loader moves the image, since both move with it.
/// Found because `__libc_csu_init`'s bound, `__init_array_end -
/// __init_array_start`, stopped folding to zero once each loaded address
/// folded to its symbol: the loop over the empty array came back, with an
/// indirect call in it.
///
/// An indexed address is not a displacement from its base, and a difference
/// that is not a whole number of bytes is not an integer; neither folds.
pub fn difference<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, machine, ..
    } = *site;
    let ptr_size = machine.ptr_size;
    let dest = site.dest?;
    let SsaOp::Sub { left, right, .. } = op else {
        return None;
    };
    let walk = AddressWalk::new(ssa, ptr_size);
    let left = walk.normalize(*left);
    let right = walk.normalize(*right);
    if left.base != right.base || left.index.is_some() || right.index.is_some() {
        return None;
    }
    // Two plain integers normalize to their own bases at no displacement, and
    // constant folding answers for them; only a base both sides share, with
    // somewhere for it to have moved, is the address model's to answer.
    if left.base.value().is_some() && left.offset_bits == 0 && right.offset_bits == 0 {
        return None;
    }
    let bits = left.offset_bits.checked_sub(right.offset_bits)?;
    if bits % 8 != 0 {
        return None;
    }
    let ty = ssa.variable(dest)?.var_type();
    let value = ConstValue::integer(T::bit_width(ty)?, T::is_signed(ty), bits / 8)?;
    Some(site.candidate(dest, Simplification::Constant(value), "address difference"))
}

/// The value `op` moves back to where it was computed from, when it does: a
/// stack pointer moved down for a buffer and up again past it, `(sp - 32) +
/// 32` through its integer conversions, is the stack pointer it started as.
/// Left apart, the two are different values of one place, and a loop's phi
/// joining them never collapses.
///
/// The address model reads through the moves ([`AddressWalk::normalize`](crate::analysis::address::AddressWalk::normalize)):
/// what lands on a value of the destination's own type at no displacement
/// and no index is that value, and on a pointer held as an integer of its
/// width, that pointer converted: RISC-V moves a `va_list` 56 bytes on with a
/// pointer add and back with an integer one, `(ap + 56) - 56`, which is `ap`.
/// Not the entry stack pointer: an address measured from it is a place in the
/// frame, `&local_20[8]` one past an array there, and converted it would name
/// the stack pointer instead of the place.
pub fn round_trip<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, machine, ..
    } = *site;
    let ptr_size = machine.ptr_size;
    let dest = site.dest?;
    if !matches!(
        op,
        SsaOp::Add { .. } | SsaOp::Sub { .. } | SsaOp::PtrAdd { index: None, .. }
    ) {
        return None;
    }
    let address = AddressWalk::new(ssa, ptr_size).normalize(dest);
    let base = address.base.value()?;
    if base == dest || address.offset_bits != 0 || address.index.is_some() {
        return None;
    }
    let base_variable = ssa.variable(base)?;
    let base_type = base_variable.var_type();
    let dest_type = ssa.variable(dest)?.var_type();
    if base_type == dest_type {
        return Some(site.candidate(
            dest,
            Simplification::Copy(base),
            "an address moved back to its base",
        ));
    }
    if base_variable.role() == Some(VariableRole::EntryStackPointer)
        || !(T::is_pointer(base_type)
            && T::is_integer(dest_type)
            && T::bit_width(dest_type) == Some(ptr_size.bits()))
    {
        return None;
    }
    Some(site.candidate(
        dest,
        Simplification::Rewrite(Box::new(SsaOp::PtrToInt {
            dest,
            operand: base,
            target: dest_type.clone(),
        })),
        "an address moved back to its base, as an integer",
    ))
}

/// The relation of two addresses measured from one base, when `op` compares
/// them: `__TMC_END__ == __TMC_LIST__`, two `lea`s of one symbol, decided
/// by their displacements as [`AddressWalk::compare`](crate::analysis::address::AddressWalk::compare) decides it.
///
/// A symbol evaluates to no number, so constant folding decides no comparison
/// of two; the address model, which subtracts them ([`addresses::difference`](crate::passes::algebraic::addresses::difference)),
/// orders them too. The result is the `I32` boolean every folded comparison
/// is, and a value compared with itself is the identity rules', which know
/// whether it may be a float.
pub fn comparison<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, machine, ..
    } = *site;
    let ptr_size = machine.ptr_size;
    let dest = site.dest?;
    let (left, right, cmp, unsigned) = match op {
        SsaOp::Ceq { left, right, .. } => (*left, *right, CmpKind::Eq, false),
        SsaOp::Clt {
            left,
            right,
            unsigned,
            ..
        } => (*left, *right, CmpKind::Lt, *unsigned),
        SsaOp::Cgt {
            left,
            right,
            unsigned,
            ..
        } => (*left, *right, CmpKind::Gt, *unsigned),
        _ => return None,
    };
    if left == right {
        return None;
    }
    let holds = AddressWalk::new(ssa, ptr_size).compare(left, right, cmp, unsigned)?;
    Some(site.candidate(
        dest,
        Simplification::Constant(ConstValue::I32(i32::from(holds))),
        "address comparison",
    ))
}

/// An address whose scaled index is a known constant is a base plus a
/// displacement. Canonicalised here, at the source, so no consumer is handed
/// a constant in the position meant for a run-time index.
pub fn constant_index<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        op,
        constants,
        machine,
        ..
    } = *site;
    let dest = site.dest?;
    let ptr_size = machine.ptr_size;
    if let Some(folded) = address::fold_constant_index(op, constants, ptr_size) {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(folded)),
            "constant index → displacement",
        ));
    }
    None
}

/// An index scaled before it is added is the scaled index the address
/// states: `ptradd b, (i << 2), 1` is `ptradd b, i, 4`, the form an add
/// chain normalizes to. Only an index computed at the pointer's width,
/// where the shift or product wraps as the address arithmetic does; a
/// narrower one wraps first, and is a different address.
pub fn scaled_index<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, machine, ..
    } = *site;
    let dest = site.dest?;
    let ptr_size = machine.ptr_size;
    if let SsaOp::PtrAdd {
        dest: address_dest,
        base,
        index: Some(index),
        stride,
        offset,
        result_type,
    } = op
        && ssa.bit_width(*index) == Some(ptr_size.bits())
        && let Some((unscaled, scale)) = AddressWalk::new(ssa, ptr_size).scaled_index(*index)
        && let Some(stride) = stride.checked_mul(scale)
    {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(SsaOp::PtrAdd {
                dest: *address_dest,
                base: *base,
                index: Some(unscaled),
                stride,
                offset: *offset,
                result_type: result_type.clone(),
            })),
            "scaled index -> stride",
        ));
    }
    None
}

/// An address that adds nothing to a base already of its type is the base:
/// `ptradd p` is `p`. Over a base of another type it converts the base,
/// and stays.
pub fn zero_offset<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { ssa, op, .. } = *site;
    let dest = site.dest?;
    if let SsaOp::PtrAdd {
        base,
        index: None,
        offset: 0,
        result_type,
        ..
    } = op
        && ssa
            .variable(*base)
            .is_some_and(|variable| variable.var_type() == result_type)
    {
        return Some(site.candidate(dest, Simplification::Copy(*base), "ptradd p → p"));
    }
    None
}
