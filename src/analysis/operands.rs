//! The operands an idiom matcher reads: a value seen through what changes no
//! bit of it, and a constant amount.
//!
//! The rotate, byte-swap, double-width and distribution folds match over one
//! substrate, so each reads its operands by one rule.

use std::collections::BTreeMap;

use crate::{
    ir::{SsaFunction, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    target::Target,
};

/// Sees through copies and width-preserving integer conversions to the value
/// underneath.
///
/// gcc inserts a cast between a boolean op and the arithmetic that reads it
/// (`(uint32_t)(a ^ b)` over `int` operands), so the operand of an `Add` is a
/// conversion, not the `Xor` itself. A conversion between two types of the same
/// width changes no bit, so the value underneath is the value the outer
/// operation reads. A widening or narrowing conversion does change bits - a
/// zero extension clears the high ones, a truncation drops them - so it is a
/// different value and the walk stops there, as it does when either width is
/// unknown.
pub fn peel<T: Target>(ssa: &SsaFunction<T>, mut var: SsaVarId) -> SsaVarId {
    for _ in 0..8 {
        match ssa.get_definition(var) {
            Some(SsaOp::Copy { src, .. }) => var = *src,
            Some(SsaOp::IntConv { operand, .. })
                if ssa.bit_width(var).is_some()
                    && ssa.bit_width(var) == ssa.bit_width(*operand) =>
            {
                var = *operand;
            }
            _ => break,
        }
    }
    var
}

/// `var`'s constant value as a width or shift amount, if it has one.
pub fn const_amount<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    var: SsaVarId,
) -> Option<u32> {
    // `as_i64` rather than `as_i32`: x86 materialises the mask as a 32-bit
    // constant, while AArch64 and MIPS emit `31` as an `I64`, which `as_i32`
    // answers `None` for.
    let value = constants.get(&peel(ssa, var))?;
    u32::try_from(value.as_i64()?).ok()
}

/// The width `value` is shifted, rotated or swapped in: its type's, where
/// that is a machine word of 8, 16, 32 or 64 bits.
///
/// Taken unpeeled: the shift reads the value at a definite width, and peeling
/// to the wider thing a conversion narrowed would name the wrong rotation.
pub fn word_width<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId) -> Option<u32> {
    ssa.bit_width(value)
        .filter(|width| matches!(width, 8 | 16 | 32 | 64))
}
