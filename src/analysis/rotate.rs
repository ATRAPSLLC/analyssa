//! Folding a shift pair back to the rotate it spells.
//!
//! C has no rotate operator, so a rotate is written as two shifts or-ed
//! together: `(x << r) | (x >> (32 - r))`. Compilers recognise that and emit
//! the ISA's single instruction at `-O2`, but at `-O0` they emit the source's
//! own shift chain. Recovered code that
//! reproduces the chain has reproduced the spelling rather than the operation,
//! and the operation is what a reader is after.
//!
//! # What a fold requires
//!
//! Both shifts must read the **same SSA value**, because the identity is on
//! values: two independent reads of a `volatile` are two different values and
//! rotating one by the other is not what the program does. The right shift must
//! be logical, since an arithmetic shift feeds in sign bits rather than the
//! bits the left shift displaced. Both shifts, the `or` and the value must
//! share one known width, the width the rotate is taken at: a conversion
//! between them changes which bits are shifted in, so the operands are only
//! looked through `operands::peel`, which crosses no width change. And the two amounts
//! must be complementary modulo the width, in one of the two spellings a
//! compiler emits:
//!
//! - **masked negation**, which is what `-O0` produces: `r & 31` against
//!   `(-r) & 31`, the mask making both amounts well defined for `r == 0`;
//! - **constant**, where the two amounts are literals summing to the width.
//!
//! # Why the flags must be dead
//!
//! [`SsaOp::Rol`] and [`SsaOp::Ror`] carry no flags output, while the `or` that
//! terminates the idiom often sets them. Rewriting one to the other therefore
//! drops a definition, which is sound only when nothing reads it - and it is
//! *not* merely a bookkeeping concern: a real `rol` sets flags differently from
//! an `or`, so a fold performed while something observed them would change the
//! program's meaning. The caller supplies use counts taken from the blocks
//! (see [`SsaFunction::count_uses`]) rather than a def-use index: this runs
//! inside a pass whose entry does not guarantee the index is current, and a
//! stale list would answer "dead" for a value still read, which is the unsound
//! direction. Counting once per pass and passing the counts in keeps the check
//! a lookup.

use std::collections::BTreeMap;

use crate::{
    analysis::operands::{const_amount, peel, word_width},
    ir::{SsaFunction, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    target::Target,
};

/// The rewrite a rotate idiom yields: the rotate the shift pair computes.
pub struct RotateFold<T: Target> {
    /// The operation to replace the matched `or` with, keeping its `dest`.
    pub op: SsaOp<T>,
    /// The idiom that matched, for the pass's event log.
    pub description: &'static str,
}

/// The `(source, mask)` of an `and` against a constant mask, either way round.
fn masked<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    var: SsaVarId,
) -> Option<(SsaVarId, u32)> {
    let SsaOp::And { left, right, .. } = ssa.get_definition(amount_source(ssa, var))? else {
        return None;
    };
    let (source, mask) = if let Some(mask) = const_amount(ssa, constants, *right) {
        (*left, mask)
    } else {
        (*right, const_amount(ssa, constants, *left)?)
    };
    // The crossing `amount_source` made is only sound because the amount that
    // reaches the shift still holds the masked value. It does whenever the
    // amount's own type is wide enough for the mask, which is the check, not an
    // assumption: a type narrower than the mask would have dropped bits the
    // rotate depends on.
    let needed = u32::BITS.saturating_sub(mask.leading_zeros());
    if ssa.bit_width(var).is_some_and(|bits| bits >= needed) {
        Some((source, mask))
    } else {
        None
    }
}

/// A shift amount's definition, seen through the conversions an ISA imposes on
/// a shift count.
///
/// x86 takes a variable count in `cl`, so a 32-bit rotate's amount reaches the
/// shift as an 8-bit value and the masking that produced it sits on the far
/// side of a narrowing conversion. [`peel`] refuses that crossing, and is right
/// to in general - a narrowing drops bits. For an amount it does not, and
/// [`masked`] checks the one condition that makes it so before returning: the
/// value is masked to `width - 1`, which needs six bits at the widest rotate
/// any ISA has, and the narrowest count operand is a byte. Without this the
/// masked-negation spelling - the one every compiler emits at `-O0`, and the
/// reason this recogniser exists - never matched on x86 at all.
fn amount_source<T: Target>(ssa: &SsaFunction<T>, var: SsaVarId) -> SsaVarId {
    let mut var = peel(ssa, var);
    for _ in 0..8 {
        match ssa.get_definition(var) {
            Some(SsaOp::Copy { src, .. } | SsaOp::IntConv { operand: src, .. }) => var = *src,
            _ => break,
        }
    }
    var
}

/// Whether `negated` is defined as the negation of `source`.
fn is_negation<T: Target>(ssa: &SsaFunction<T>, source: SsaVarId, negated: SsaVarId) -> bool {
    matches!(
        ssa.get_definition(peel(ssa, negated)),
        Some(SsaOp::Neg { operand, .. }) if peel(ssa, *operand) == peel(ssa, source)
    )
}

/// Folds `(x << a) | (x >> b)` to the rotate it spells, when it spells one.
///
/// # Arguments
///
/// * `ssa` - The function, for resolving each operand's defining instruction.
/// * `op` - The candidate operation, folded only when it is an `or`.
/// * `constants` - Known constant values, for the mask and amount literals.
/// * `uses` - Use counts for the whole function, as
///   [`SsaFunction::count_uses`] reports them, to tell whether the `or`'s flags
///   output is read.
///
/// # Returns
///
/// The rotate to install in the `or`'s place, or `None` when the operands do
/// not spell one or the `or`'s flags are still read.
pub fn fold<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Option<RotateFold<T>> {
    let SsaOp::Or {
        dest,
        left,
        right,
        flags,
    } = op
    else {
        return None;
    };
    if flags.is_some_and(|flags| uses.get(&flags).copied().unwrap_or(0) != 0) {
        return None;
    }

    // Either operand can carry the left shift; `or` is commutative and which
    // side a compiler emits first is not a fact about the program.
    for (shifted_left, shifted_right) in [(*left, *right), (*right, *left)] {
        let Some(SsaOp::Shl {
            value: left_value,
            amount: left_amount,
            ..
        }) = ssa.get_definition(peel(ssa, shifted_left))
        else {
            continue;
        };
        // Logical only: an arithmetic shift feeds in sign bits, not the bits
        // the left shift displaced, so the pair is not a rotation.
        let Some(SsaOp::Shr {
            value: right_value,
            amount: right_amount,
            unsigned: true,
            ..
        }) = ssa.get_definition(peel(ssa, shifted_right))
        else {
            continue;
        };
        if peel(ssa, *left_value) != peel(ssa, *right_value) {
            continue;
        }
        // One width for the `or`, both shifts and both shifted values: the
        // width the `or` computes at is then the width of every bit the pair
        // moves. A shifted value of no stated type is the bits the shift
        // reads, as an `-O0` argument a move copies is.
        let Some(width) = word_width(ssa, *dest) else {
            continue;
        };
        if [*left, *right]
            .iter()
            .any(|var| ssa.bit_width(*var) != Some(width))
            || [*left_value, *right_value]
                .iter()
                .any(|var| !ssa.is_read_whole_at(*var, width))
        {
            continue;
        }

        let rotate = rotate_for(
            ssa,
            constants,
            width,
            (*dest, *left_value),
            *left_amount,
            *right_amount,
        );
        if rotate.is_some() {
            return rotate;
        }
    }
    None
}

/// The rotate of `value` into `dest`, `width` bits wide, that `left_amount`
/// and `right_amount` describe, if they are complementary modulo the width.
fn rotate_for<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    width: u32,
    (dest, value): (SsaVarId, SsaVarId),
    left_amount: SsaVarId,
    right_amount: SsaVarId,
) -> Option<RotateFold<T>> {
    // The `-O0` spelling: `r & (w - 1)` against `(-r) & (w - 1)`. The rotate
    // takes whichever amount is the un-negated one, since that is the `r` the
    // source rotated by, and the masking is what `Rol`/`Ror` do anyway.
    if let (Some((left_source, left_mask)), Some((right_source, right_mask))) = (
        masked(ssa, constants, left_amount),
        masked(ssa, constants, right_amount),
    ) && left_mask == right_mask
        && left_mask == width.saturating_sub(1)
    {
        if is_negation(ssa, left_source, right_source) {
            return Some(RotateFold {
                op: SsaOp::Rol {
                    dest,
                    value,
                    amount: left_amount,
                },
                description: "shift pair → rol",
            });
        }
        if is_negation(ssa, right_source, left_source) {
            return Some(RotateFold {
                op: SsaOp::Ror {
                    dest,
                    value,
                    amount: right_amount,
                },
                description: "shift pair → ror",
            });
        }
    }

    // The constant spelling. The two literals name the width between them, so
    // no operand type has to be consulted: `(x << 7) | (x >> 25)` is a 32-bit
    // rotate because 7 + 25 is 32. A zero amount is excluded because `x | x`
    // is not a rotation of anything.
    let (left_constant, right_constant) = (
        const_amount(ssa, constants, left_amount)?,
        const_amount(ssa, constants, right_amount)?,
    );
    // Against the rotate's own width, not against any plausible width: over a
    // 32-bit value `(x << 7) | (x >> 9)` sums to 16 and is not a rotation of
    // anything, because the bits the left shift pushed out are gone.
    if left_constant > 0
        && right_constant > 0
        && left_constant.checked_add(right_constant)? == width
    {
        return Some(RotateFold {
            op: SsaOp::Rol {
                dest,
                value,
                amount: left_amount,
            },
            description: "shift pair → rol",
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::SsaFunctionBuilder,
        testing::{MockTarget, MockType, mock_i32},
    };

    /// How the operands of the terminating `or` were built.
    struct Shape {
        /// `true` for the masked-negation spelling, `false` for constants.
        masked: bool,
        /// `true` when the right shift is logical, as a rotate requires.
        unsigned: bool,
        /// `true` when both shifts read the same value.
        same_value: bool,
        /// Amounts that are complementary modulo the width.
        complementary: bool,
        /// A flags output on the `or`, and whether anything reads it.
        flags: Option<bool>,
    }

    impl Shape {
        /// The shape `-O0` emits: masked negation, logical shift, one value.
        const fn rotate() -> Self {
            Self {
                masked: true,
                unsigned: true,
                same_value: true,
                complementary: true,
                flags: None,
            }
        }
    }

    /// Builds `(x << a) | (x >> b)` under `shape`, returning the `or` to fold.
    fn fixture(
        shape: &Shape,
    ) -> (
        SsaFunction<MockTarget>,
        SsaOp<MockTarget>,
        BTreeMap<SsaVarId, ConstValue<MockTarget>>,
    ) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 2);
        let mut or_op = None;
        builder
            .in_block(0, |block| {
                let x = block.const_i32(mock_i32(), 0x1234)?;
                let other = block.const_i32(mock_i32(), 0x5678)?;
                let r = block.const_i32(mock_i32(), 7)?;
                let (left_amount, right_amount) = if shape.masked {
                    let mask = block.const_i32(mock_i32(), 31)?;
                    let left = block.and(mock_i32(), r, mask)?;
                    // A non-complementary pair masks the *same* source rather
                    // than its negation, which is `x << r | x >> r`.
                    let source = if shape.complementary {
                        block.neg(mock_i32(), r)?
                    } else {
                        r
                    };
                    let right = block.and(mock_i32(), source, mask)?;
                    (left, right)
                } else {
                    let left = block.const_i32(mock_i32(), 7)?;
                    let right =
                        block.const_i32(mock_i32(), if shape.complementary { 25 } else { 9 })?;
                    (left, right)
                };
                let shifted_left = block.shl(mock_i32(), x, left_amount)?;
                let shift_value = if shape.same_value { x } else { other };
                let shifted_right =
                    block.shr(mock_i32(), shift_value, right_amount, shape.unsigned)?;
                let result = block.or(mock_i32(), shifted_left, shifted_right)?;
                let flags = match shape.flags {
                    Some(_) => Some(block.const_i32(mock_i32(), 0)?),
                    None => None,
                };
                or_op = Some(SsaOp::Or {
                    dest: result,
                    left: shifted_left,
                    right: shifted_right,
                    flags,
                });
                // A read of the flags value, when the shape asks for a live one.
                if shape.flags == Some(true)
                    && let Some(flags) = flags
                {
                    let observed = block.and(mock_i32(), flags, flags)?;
                    block.ret(Some(observed))?;
                } else {
                    block.ret(Some(result))?;
                }
                Ok(())
            })
            .expect("build block");
        let ssa = builder.finish().expect("finish");
        let mut constants = BTreeMap::new();
        for (_, _, instr) in ssa.iter_instructions() {
            if let SsaOp::Const { dest, value } = instr.op() {
                constants.insert(*dest, value.clone());
            }
        }
        (ssa, or_op.expect("or op"), constants)
    }

    /// The `-O0` spelling folds to a rotate.
    ///
    /// `r & 31` against `(-r) & 31` is what every target emits at `-O0` for a
    /// 32-bit rotate by a run-time amount.
    #[test]
    fn a_masked_negation_pair_folds_to_a_rotate() {
        let (ssa, op, constants) = fixture(&Shape::rotate());
        let fold = fold(&ssa, &op, &constants, &ssa.count_uses())
            .expect("the -O0 rotate spelling must fold");
        assert!(
            matches!(fold.op, SsaOp::Rol { .. }),
            "expected a rol, got {:?}",
            fold.op
        );
    }

    /// **A rotated value of no stated type is the bits its shifts read.** An
    /// `-O0` argument a move copies states no type, and the shifts and the
    /// `or` state the width the pair moves; the control types the value at
    /// another width, whose bits the 32-bit pair does not rotate.
    #[test]
    fn an_untyped_value_rotates_at_its_shifts_width() {
        let rotated = |ty: MockType| {
            let (mut ssa, op, constants) = fixture(&Shape::rotate());
            let SsaOp::Or { left, .. } = op else {
                unreachable!("the fixture folds an `or`")
            };
            let Some(SsaOp::Shl { value, .. }) = ssa.get_definition(left).cloned() else {
                unreachable!("the left operand is the left shift")
            };
            if let Some(variable) = ssa.variable_mut(value) {
                variable.set_type(ty);
            }
            fold(&ssa, &op, &constants, &ssa.count_uses()).map(|fold| fold.op)
        };
        assert!(
            matches!(rotated(MockType::Unknown), Some(SsaOp::Rol { .. })),
            "{:?}",
            rotated(MockType::Unknown)
        );
        assert_eq!(rotated(MockType::I64), None, "the control");
    }

    /// Two literals naming the width between them fold to a rotate.
    #[test]
    fn a_constant_pair_summing_to_the_width_folds() {
        let (ssa, op, constants) = fixture(&Shape {
            masked: false,
            ..Shape::rotate()
        });
        let fold =
            fold(&ssa, &op, &constants, &ssa.count_uses()).expect("7 and 25 name a 32-bit rotate");
        assert!(matches!(fold.op, SsaOp::Rol { .. }));
    }

    /// An arithmetic right shift is not a rotation.
    ///
    /// The control that matters most: an arithmetic shift feeds in sign bits
    /// rather than the bits the left shift displaced, so the pair computes
    /// something else entirely and folding it would be a wrong answer, not a
    /// missed one.
    #[test]
    fn an_arithmetic_shift_does_not_fold() {
        let (ssa, op, constants) = fixture(&Shape {
            unsigned: false,
            ..Shape::rotate()
        });
        assert!(
            fold(&ssa, &op, &constants, &ssa.count_uses()).is_none(),
            "an arithmetic shift is not a rotate"
        );
    }

    /// Two different values are not one value rotated.
    #[test]
    fn shifts_of_different_values_do_not_fold() {
        let (ssa, op, constants) = fixture(&Shape {
            same_value: false,
            ..Shape::rotate()
        });
        assert!(fold(&ssa, &op, &constants, &ssa.count_uses()).is_none());
    }

    /// Amounts that do not complement each other are not a rotation.
    #[test]
    fn non_complementary_amounts_do_not_fold() {
        for masked in [true, false] {
            let (ssa, op, constants) = fixture(&Shape {
                masked,
                complementary: false,
                ..Shape::rotate()
            });
            assert!(
                fold(&ssa, &op, &constants, &ssa.count_uses()).is_none(),
                "masked={masked}: the amounts do not span the width"
            );
        }
    }

    /// A rotate carries no flags, so one that is read keeps the `or`.
    ///
    /// `Rol` and `Ror` have no flags output. Folding an `or` whose flags a
    /// later instruction reads would leave that read with no definition, and a
    /// real `rol` sets flags differently from an `or` regardless - so the fold
    /// is refused rather than made and patched up.
    #[test]
    fn a_read_flags_output_refuses_the_fold() {
        let (ssa, op, constants) = fixture(&Shape {
            flags: Some(true),
            ..Shape::rotate()
        });
        assert!(
            fold(&ssa, &op, &constants, &ssa.count_uses()).is_none(),
            "the flags are read, so the or has to stay"
        );
    }

    /// A flags output nothing reads does not block the fold.
    #[test]
    fn a_dead_flags_output_still_folds() {
        let (ssa, op, constants) = fixture(&Shape {
            flags: Some(false),
            ..Shape::rotate()
        });
        assert!(
            fold(&ssa, &op, &constants, &ssa.count_uses()).is_some(),
            "nothing reads the flags, so the rotate is sound"
        );
    }
}
