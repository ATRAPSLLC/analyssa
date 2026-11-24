//! Rules for the idioms a compiler spells in several operations: a byte swap, a
//! rotate, double-width arithmetic in halves, a value and its complement or
//! absorption, shifted parts, and mixed boolean arithmetic.

use std::collections::BTreeMap;

use crate::{
    analysis::{
        byteswap, convert::NarrowValue, distribute, distribute::Absorption, doubleword,
        doubleword::Doubleword, mba, rotate,
    },
    ir::{
        function::SsaFunction, integer::BitWidth, ops::SsaOp, value::ConstValue, variable::SsaVarId,
    },
    passes::algebraic::{ShiftedPart, Simplification, SimplificationCandidate, site::Site},
    pointer::PointerSize,
    target::Target,
};

/// A byte reversal is a whole subtree, not a single op: one masked shift
/// per byte, or-ed together, which is how C spells an operation it has no
/// operator for. It is matched against the operands' defining instructions
/// rather than through `simplify_op`.
pub fn byte_reversal<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        ..
    } = *site;
    let dest = site.dest?;
    if let Some(fold) = byteswap::fold(ssa, op, constants, uses) {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(fold.op)),
            fold.description,
        ));
    }
    None
}

/// A rotate is a whole subtree too: two shifts of one value by
/// complementary amounts, or-ed together.
pub fn rotation<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        ..
    } = *site;
    let dest = site.dest?;
    if let Some(fold) = rotate::fold(ssa, op, constants, uses) {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(fold.op)),
            fold.description,
        ));
    }
    None
}

/// A value composed of two halves summed a half at a time is the sum of the
/// values their halves compose. The sum carries no flags output, so the
/// composition's must be dead.
pub fn doubleword_sum<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        machine,
        ..
    } = *site;
    let dest = site.dest?;
    if op
        .flags_dest()
        .is_none_or(|flags| uses.get(&flags).copied().unwrap_or(0) == 0)
        && let Some(fold) = Doubleword::new(ssa, constants, machine).fold(op)
    {
        let read_apart = fold
            .halves
            .iter()
            .any(|half| uses.get(half).copied().unwrap_or(0) > 1);
        return Some(site.candidate(
            dest,
            Simplification::Doubleword { fold, read_apart },
            "a double-width sum computed in halves",
        ));
    }
    None
}

/// A double-width sum adding the two halves of a wider value apart, its low
/// half as a term and its high half among the high half's, adds the value
/// ([`Doubleword::regroup`]). The sum carries no flags output, so the
/// operation's must be dead.
pub fn doubleword_regroup<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        machine,
        ..
    } = *site;
    let dest = site.dest?;
    if op
        .flags_dest()
        .is_none_or(|flags| uses.get(&flags).copied().unwrap_or(0) == 0)
        && let Some(regrouping) = Doubleword::new(ssa, constants, machine).regroup(op)
    {
        return Some(site.candidate(
            dest,
            Simplification::Regrouped(Box::new(regrouping)),
            "a double-width sum adding a value's halves apart",
        ));
    }
    None
}

/// A comparison of double-width values computed in halves, the high
/// halves compared with the low halves' borrow, is the comparison of the
/// values, where each side is one value already or constants.
pub fn doubleword_comparison<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    let dest = site.dest?;
    if let Some(wide) = doubleword::comparison(ssa, op, constants) {
        return Some(site.candidate(
            dest,
            match wide.constant {
                Some((operand, value, ty)) => Simplification::Narrowed(
                    Box::new(wide.op),
                    vec![(operand, NarrowValue::Constant(value))],
                    ty,
                ),
                None => Simplification::Rewrite(Box::new(wide.op)),
            },
            "a double-width comparison computed in halves",
        ));
    }
    None
}

/// Two loads of adjacent words composed are the load of both, where the
/// byte order is known.
pub fn wide_load<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        machine,
        ..
    } = *site;
    let dest = site.dest?;
    if op
        .flags_dest()
        .is_none_or(|flags| uses.get(&flags).copied().unwrap_or(0) == 0)
        && let Some(load) = Doubleword::new(ssa, constants, machine).wide_load(op, uses)
    {
        return Some(site.candidate(
            dest,
            Simplification::WideLoad(Box::new(load)),
            "two adjacent words loaded as one",
        ));
    }
    None
}

/// A value joined with its complement is every bit set or none.
pub fn complement_join<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        ..
    } = *site;
    let dest = site.dest?;
    if op
        .flags_dest()
        .is_none_or(|flags| uses.get(&flags).copied().unwrap_or(0) == 0)
        && let Some(value) = distribute::complement(ssa, op, constants)
    {
        return Some(site.candidate(
            dest,
            Simplification::Constant(value),
            "a value joined with its complement",
        ));
    }
    None
}

/// A value absorbs a join it is part of: `x & (x | y)` is `x`, `x & ~(x |
/// y)` none, `x & (~x | y)` is `x & y`, and so for `|` over `&`.
pub fn absorption<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        ..
    } = *site;
    let dest = site.dest?;
    if op
        .flags_dest()
        .is_none_or(|flags| uses.get(&flags).copied().unwrap_or(0) == 0)
        && let Some(absorption) = distribute::absorbed(ssa, op, constants)
    {
        return Some(site.candidate(
            dest,
            match absorption {
                Absorption::Operand(operand) => Simplification::Copy(operand),
                Absorption::Constant(value) => Simplification::Constant(value),
                Absorption::Joined(kind, left, right) => {
                    Simplification::Rewrite(Box::new(kind.op(dest, left, right)))
                }
            },
            "a value absorbs a join it is part of",
        ));
    }
    None
}

/// A value shifted up and back down is the part of it the shifts keep,
/// extended and moved where they leave it.
pub fn shifted_part<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        ..
    } = *site;
    let dest = site.dest?;
    if op
        .flags_dest()
        .is_none_or(|flags| uses.get(&flags).copied().unwrap_or(0) == 0)
        && let Some(part) = part_kept(ssa, op, constants, uses)
    {
        return Some(site.candidate(
            dest,
            Simplification::ShiftedPart(Box::new(part)),
            "a value shifted up and back down is the part it keeps",
        ));
    }
    None
}

/// A value with its low bit cleared, plus its half, is three halves.
pub fn tripled_half<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        ..
    } = *site;
    let dest = site.dest?;
    if op
        .flags_dest()
        .is_none_or(|flags| uses.get(&flags).copied().unwrap_or(0) == 0)
        && let Some((tripled, inserted, ty)) = tripled(ssa, op, constants)
    {
        return Some(site.candidate(
            dest,
            Simplification::Narrowed(Box::new(tripled), inserted, ty),
            "a value with its low bit cleared and its half are three halves",
        ));
    }
    None
}

/// Two parts of one value joined, `(s & p) | (s & q)`, are the value under
/// the join of the masks.
pub fn factored<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { ssa, op, uses, .. } = *site;
    let dest = site.dest?;
    if op
        .flags_dest()
        .is_none_or(|flags| uses.get(&flags).copied().unwrap_or(0) == 0)
        && let Some(factoring) = distribute::factored(ssa, op, uses)
    {
        return Some(site.candidate(
            dest,
            Simplification::Factored(factoring),
            "a shared operand taken out of a join",
        ));
    }
    None
}

/// A value composed with its own sign above it is its sign extension.
pub fn sign_extension<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        uses,
        ..
    } = *site;
    let dest = site.dest?;
    if op
        .flags_dest()
        .is_none_or(|flags| uses.get(&flags).copied().unwrap_or(0) == 0)
        && let Some(extension) = doubleword::extension(ssa, op, constants)
    {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(extension)),
            "a value composed with its sign",
        ));
    }
    None
}

/// A mixed boolean-arithmetic identity is a whole subtree as well: a
/// tangle of boolean and arithmetic operations that computes one simple
/// operation at every input of its width.
pub fn mixed_boolean<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        block_idx,
        instr_idx,
        constants,
        uses,
        machine,
        ..
    } = *site;
    let dest = site.dest?;
    let ptr_size = machine.ptr_size;
    if let Some(fold) = mba::fold(ssa, op, constants, uses, ptr_size) {
        let op = Box::new(fold.op);
        return Some(SimplificationCandidate {
            block_idx,
            instr_idx,
            dest,
            simplification: match fold.dest_type {
                Some(dest_type) => Simplification::Retyped(op, dest_type),
                None => Simplification::Rewrite(op),
            },
            also: fold.also,
            description: fold.description,
        });
    }
    None
}

/// The part of a value `op` selects, where it shifts right a value shifted
/// left: `(x << k) >> j` keeps bits of `x` and extends them.
///
/// At width `W`, the left shift drops `x`'s top `k` bits, and the right one
/// brings what is left down by `j`, extending by sign where it is arithmetic
/// and by zero where it is logical. Where `j >= k` that is the bits
/// `j - k..W - k` of `x`, extended from `W - j` bits; where `j < k`, the bits
/// `0..W - k`, extended, and moved up `k - j`. A lift spells an extension or a
/// scaled extension so where the instruction set has no other: RV64's
/// `slli a0, a0, 32; srli a0, a0, 29` is a 32-bit index zero-extended and
/// scaled by eight, which a reader recognises only as
/// `(uint64_t)(uint32_t)i << 3`.
///
/// A signed part of a width that is no integer type's, a bit or a field,
/// stays: no mask states its sign. Shifts by a variable amount, by the width or more, and a
/// left shift whose flags something reads stay too.
fn part_kept<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Option<ShiftedPart<T>> {
    let SsaOp::Shr {
        dest,
        value,
        amount,
        unsigned,
        ..
    } = op
    else {
        return None;
    };
    let constant = |value: &SsaVarId| constants.get(value).and_then(ConstValue::as_u64);
    let Some(SsaOp::Shl {
        value: source,
        amount: left,
        flags,
        ..
    }) = ssa
        .recorded_definition(*value)
        .map(|found| found.instruction().op())
    else {
        return None;
    };
    if flags.is_some_and(|flags| uses.get(&flags).copied().unwrap_or(0) > 0) {
        return None;
    }
    let width = ssa.bit_width(*dest)?;
    if ssa.bit_width(*value)? != width || ssa.bit_width(*source)? != width {
        return None;
    }
    let left = u32::try_from(constant(left)?).ok()?;
    let right = u32::try_from(constant(amount)?).ok()?;
    if left == 0 || left >= width || right >= width {
        return None;
    }
    let (low, bits, up) = if right >= left {
        (right.checked_sub(left)?, width.checked_sub(right)?, 0)
    } else {
        (0, width.checked_sub(left)?, left.checked_sub(right)?)
    };
    let signed = !*unsigned;
    // A part of an integer type's width, which C spells; a bit or a field of
    // another width is no such type, whatever the target can name.
    let narrow = (bits >= 8 && bits.is_power_of_two())
        .then(|| T::integer_type(bits, signed))
        .flatten();
    if signed && narrow.is_none() {
        return None;
    }
    Some(ShiftedPart {
        source: *source,
        low,
        bits,
        signed,
        up,
        narrow,
        wide: ssa.variable(*dest)?.var_type().clone(),
        width,
        amount_bits: ssa.bit_width(*amount)?,
        amount: ssa.variable(*amount)?.var_type().clone(),
    })
}

/// `(x & ~1) + (x >> 1)` as `(x >> 1) * 3`: clearing the low bit of `x` is
/// doubling its half. A compiler computes `mid * 3` this way from the sum a
/// binary search halves, so an array of records is indexed by the record.
///
/// # Returns
///
/// The multiplication, the operand that stands for its constant until the
/// constant is defined, and the operation's type.
fn tripled<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<TripledHalf<T>> {
    let SsaOp::Add {
        dest, left, right, ..
    } = op
    else {
        return None;
    };
    let definition = |value: SsaVarId| {
        ssa.recorded_definition(value)
            .map(|found| found.instruction().op())
    };
    let halved = |value: SsaVarId| match definition(value) {
        Some(SsaOp::Shr {
            value: operand,
            amount,
            unsigned: true,
            ..
        }) if constants.get(amount).and_then(ConstValue::as_u64) == Some(1) => Some(*operand),
        _ => None,
    };
    let bits = ssa.bit_width(*dest)?;
    let cleared = |value: SsaVarId| match definition(value) {
        Some(SsaOp::And {
            left: operand,
            right: mask,
            ..
        }) => {
            let (pattern, _) = constants.get(mask)?.integer_pattern(PointerSize::Bit64)?;
            let full = BitWidth::new(bits).checked_mask()?;
            (pattern & full == full & !1).then_some(*operand)
        }
        _ => None,
    };
    let (masked, half) = match (
        cleared(*left),
        halved(*right),
        cleared(*right),
        halved(*left),
    ) {
        (Some(source), Some(halved_from), _, _) if source == halved_from => (*left, *right),
        (_, _, Some(source), Some(halved_from)) if source == halved_from => (*right, *left),
        _ => return None,
    };
    let ty = ssa.variable(*dest)?.var_type().clone();
    let three = ConstValue::from_bits_of_type(3, &ty)?;
    Some((
        SsaOp::Mul {
            dest: *dest,
            left: half,
            right: masked,
            flags: None,
        },
        vec![(masked, NarrowValue::Constant(three))],
        ty,
    ))
}

/// A multiplication, the operand standing for its constant, and its type.
type TripledHalf<T> = (
    SsaOp<T>,
    Vec<(SsaVarId, NarrowValue<T>)>,
    <T as Target>::Type,
);
