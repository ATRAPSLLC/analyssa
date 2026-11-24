//! Rules for the identities of one operation: a negation or complement and its
//! own spellings, a mask that keeps every bit, a field read back, a value
//! moved back, and a choice already made.

use std::collections::BTreeMap;

use crate::{
    analysis::{bits::Reader, select},
    ir::{
        function::SsaFunction,
        ops::{Kernel, PacKind, SsaOp},
        value::ConstValue,
        variable::SsaVarId,
    },
    passes::algebraic::{Simplification, SimplificationCandidate, site::Site},
    target::Target,
};

/// A leading-zero count shifted right by its width's bits is the zero test
/// it spells: ARM's `clz r0, r1; lsr r0, r0, #5` is `r1 == 0`, applied to
/// pointers as often as to integers. See [`zero_test`].
pub fn zero_test<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    let dest = site.dest?;
    if let Some(test) = zero_test_of(ssa, op, constants) {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(test)),
            "leading-zero count shifted by its width's bits is a zero test",
        ));
    }
    None
}

/// An exclusive or with all ones at the operation's own width is the
/// complement, the one spelling the idiom rules below read: RISC-V's `not`
/// is `xori rd, rs, -1`.
pub fn complement_xor<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        machine,
        ..
    } = *site;
    let dest = site.dest?;
    let ptr_size = machine.ptr_size;
    if let SsaOp::Xor {
        dest: xor_dest,
        left,
        right,
        flags: None,
    } = op
        && let Some(width) = op.dest().and_then(|dest| ssa.bit_width(dest))
    {
        let all_ones = |var: &SsaVarId| {
            constants
                .get(var)
                .is_some_and(|c| c.is_all_ones() && c.bit_width(ptr_size) == Some(width))
        };
        let operand = if all_ones(right) {
            Some(*left)
        } else if all_ones(left) {
            Some(*right)
        } else {
            None
        };
        if let Some(operand) = operand {
            return Some(site.candidate(
                dest,
                Simplification::Rewrite(Box::new(SsaOp::Not {
                    dest: *xor_dest,
                    operand,
                    flags: None,
                })),
                "an exclusive or with all ones is the complement",
            ));
        }
    }
    None
}

/// A subtraction from zero is the negation, the one spelling the idiom
/// rules below read: RISC-V's `neg` is `sub rd, zero, rs`.
pub fn negation_sub<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        machine,
        ..
    } = *site;
    let dest = site.dest?;
    let ptr_size = machine.ptr_size;
    if let SsaOp::Sub {
        dest: sub_dest,
        left,
        right,
        flags: None,
    } = op
        && let Some(width) = op.dest().and_then(|dest| ssa.bit_width(dest))
        && constants
            .get(left)
            .is_some_and(|c| c.is_zero() && c.bit_width(ptr_size) == Some(width))
    {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(SsaOp::Neg {
                dest: *sub_dest,
                operand: *right,
                flags: None,
            })),
            "a subtraction from zero is the negation",
        ));
    }
    None
}

/// Adding a negation subtracts what it negates, subtracting one adds it,
/// and negating one is its operand: the arithmetic a negation spelt as a
/// subtraction from zero leaves once it reads as one (RISC-V's `neg` then
/// `add`, ARM's `rsb`). The negation's own flags, when it states any, are
/// another of its results and do not change the value read here; the
/// rewritten operation states none, because a subtraction's carry is not
/// an addition's.
pub fn negation_fold<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { op, .. } = *site;
    let dest = site.dest?;
    let negation_fold = match op {
        SsaOp::Add {
            dest: add_dest,
            left,
            right,
            flags: None,
        } => site
            .negated(right)
            .map(|negated| SsaOp::Sub {
                dest: *add_dest,
                left: *left,
                right: negated,
                flags: None,
            })
            .or_else(|| {
                site.negated(left).map(|negated| SsaOp::Sub {
                    dest: *add_dest,
                    left: *right,
                    right: negated,
                    flags: None,
                })
            })
            .map(|op| (op, "adding a negation subtracts what it negates")),
        SsaOp::Sub {
            dest: sub_dest,
            left,
            right,
            flags: None,
        } => site.negated(right).map(|negated| {
            (
                SsaOp::Add {
                    dest: *sub_dest,
                    left: *left,
                    right: negated,
                    flags: None,
                },
                "subtracting a negation adds what it negates",
            )
        }),
        _ => None,
    };
    if let Some((rewritten, description)) = negation_fold {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(rewritten)),
            description,
        ));
    }
    None
}

/// The negation of a negation is its operand.
pub fn double_negation<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { op, .. } = *site;
    let dest = site.dest?;
    if let SsaOp::Neg { operand, .. } = op
        && let Some(negated) = site.negated(operand)
    {
        return Some(site.candidate(
            dest,
            Simplification::Copy(negated),
            "the negation of a negation is its operand",
        ));
    }
    None
}

/// Authenticating a pointer signed with the same modifier and the same key
/// gives back the pointer: AArch64's `paciasp` at entry and `autiasp` before
/// `ret` sign and authenticate the link register with the stack pointer and
/// the A key, and the return goes through the address the function was
/// entered with. Only the very values make the pair: another key, modifier or
/// second modifier fails the authentication, so nothing is folded for one.
pub fn authenticated_signature<T: Target>(
    site: &Site<'_, T>,
) -> Option<SimplificationCandidate<T>> {
    let Site { ssa, op, .. } = *site;
    let dest = site.dest?;
    let SsaOp::Compute(authentication) = op else {
        return None;
    };
    let signing = match authentication.kind.kernel {
        Kernel::PointerAuth(PacKind::Authenticate) => PacKind::Sign,
        Kernel::PointerAuth(PacKind::AuthenticateDiversified) => PacKind::SignDiversified,
        _ => return None,
    };
    let (pointer, under) = authentication.inputs.split_first()?;
    let Some(SsaOp::Compute(signature)) = ssa.get_definition(*pointer) else {
        return None;
    };
    let (signed, signed_under) = signature.inputs.split_first()?;
    (signature.kind.kernel == Kernel::PointerAuth(signing)
        && signature.outputs.as_slice() == [*pointer]
        && authentication.outputs.as_slice() == [dest]
        && under == signed_under)
        .then(|| {
            site.candidate(
                dest,
                Simplification::Copy(*signed),
                "a pointer signed and authenticated under one modifier and key is the pointer",
            )
        })
}

/// So is the complement of a complement: MPX's `bndcu` complements back
/// the upper bound `bndmk` holds complemented, and the check then compares
/// with the address it was made from.
pub fn double_complement<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { ssa, op, .. } = *site;
    let dest = site.dest?;
    if let SsaOp::Not { operand, .. } = op
        && let Some(SsaOp::Not {
            operand: complemented,
            ..
        }) = ssa.get_definition(*operand)
    {
        return Some(site.candidate(
            dest,
            Simplification::Copy(*complemented),
            "the complement of a complement is its operand",
        ));
    }
    None
}

/// A mask that keeps every bit its operand can have is its operand - a
/// shift count masked by the program and again by the ISA's count rule.
/// Tried before the idioms, so a rotate's amounts reach them in the shape
/// they know. See [`Reader::redundant_mask`](crate::analysis::bits::Reader::redundant_mask).
pub fn kept_mask<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    let dest = site.dest?;
    if let Some(operand) = Reader::new(ssa, constants).redundant_mask(op) {
        return Some(site.candidate(
            dest,
            Simplification::Copy(operand),
            "mask keeps every bit of its operand",
        ));
    }
    None
}

/// A field read back out of a container is the value written into it,
/// whatever else the container held. See [`Reader::demanded_field`](crate::analysis::bits::Reader::demanded_field).
pub fn field_read_back<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    let dest = site.dest?;
    if let Some(rewrite) = Reader::new(ssa, constants).demanded_field(op) {
        return Some(site.candidate(
            dest,
            Simplification::Rewrite(Box::new(rewrite)),
            "a field read back is the value written",
        ));
    }
    None
}

/// A value that is another value's bits, moved back where they were, is
/// that value: a flag written into a status register and read back out of
/// it. See [`Reader::copied_value`](crate::analysis::bits::Reader::copied_value).
pub fn bits_moved_back<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    let dest = site.dest?;
    if let Some(rewrite) = Reader::new(ssa, constants).copied_value(op) {
        return Some(site.candidate(
            dest,
            match rewrite {
                SsaOp::Copy { src, .. } => Simplification::Copy(src),
                other => Simplification::Rewrite(Box::new(other)),
            },
            "a value's bits moved back are the value",
        ));
    }
    None
}

/// A select whose arm is a select on the same condition, or its negation,
/// reads the arm that condition already picked: a pair of complementary
/// guarded writes never keeps the old value. See [`select::fold`].
pub fn select_choice<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { ssa, op, .. } = *site;
    let dest = site.dest?;
    if let Some(fold) = select::fold(ssa, op) {
        return Some(site.candidate(
            dest,
            match fold {
                SsaOp::Copy { src, .. } => Simplification::Copy(src),
                other => Simplification::Rewrite(Box::new(other)),
            },
            "a choice the condition already made",
        ));
    }
    None
}

/// The zero test `op` spells, where it shifts a leading-zero count right by
/// its width's bits: `(v == 0 ? W : (W - 1) - bsr(v)) >> log2(W)`, a lifted
/// `clz` shifted by `lsr #5` at 32 bits, is `zext(v == 0)`.
///
/// Exact: the count is `W` only for zero and at most `W - 1` otherwise, so the
/// shift leaves one for zero and nothing for any other value. The shape is the
/// lift's own, a [`SsaOp::Select`] on [`SsaOp::Ceq`] against zero between `W`
/// and `W - 1` less [`SsaOp::BitScanReverse`] of the same value.
fn zero_test_of<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<SsaOp<T>> {
    let SsaOp::Shr {
        dest,
        value,
        amount,
        unsigned: true,
        ..
    } = op
    else {
        return None;
    };
    let constant = |value: &SsaVarId| constants.get(value).and_then(ConstValue::as_u64);
    let definition = |value: SsaVarId| {
        ssa.recorded_definition(value)
            .map(|found| found.instruction().op())
    };
    let shift = constant(amount)?;
    let Some(SsaOp::Select {
        condition,
        true_val,
        false_val,
        ..
    }) = definition(*value)
    else {
        return None;
    };
    let width = constant(true_val)?;
    if !width.is_power_of_two() || u64::from(width.trailing_zeros()) != shift {
        return None;
    }
    let Some(SsaOp::Sub {
        left: top,
        right: index,
        ..
    }) = definition(*false_val)
    else {
        return None;
    };
    if constant(top)? != width.checked_sub(1)? {
        return None;
    }
    let Some(SsaOp::BitScanReverse { src, .. }) = definition(*index) else {
        return None;
    };
    let Some(SsaOp::Ceq { left, right, .. }) = definition(*condition) else {
        return None;
    };
    let tested = match (constant(left), constant(right)) {
        (_, Some(0)) => left,
        (Some(0), _) => right,
        _ => return None,
    };
    if tested != src {
        return None;
    }
    Some(SsaOp::IntConv {
        dest: *dest,
        operand: *condition,
        target: ssa.variable(*dest)?.var_type().clone(),
        overflow_check: false,
        unsigned: true,
    })
}
