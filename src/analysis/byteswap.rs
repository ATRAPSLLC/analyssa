//! Folding a masked shift chain back to the byte reversal it spells.
//!
//! C has no byte-swap operator, so a reversal is written as one shift per byte,
//! each masked to the lane it lands in, or-ed together:
//!
//! ```c
//! (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24)
//! ```
//!
//! Compilers emit the ISA's single instruction for that at `-O2` where one
//! exists, and the source's own chain at `-O0`.
//! Recovered code that reproduces the chain has reproduced the spelling rather
//! than the operation.
//!
//! # What a fold requires
//!
//! Every term must read the **same SSA value**, and the terms together must
//! move each byte to the position a reversal puts it in. That last condition is
//! what separates this from any other chain of four masked shifts: a chain that
//! assembles the bytes in some *other* order is a permutation, a field
//! extraction, or a checksum step, and folding it to `bswap` would be a wrong
//! answer rather than a missed one. Each term is therefore resolved to the byte
//! it reads and the byte it writes, and the fold is refused unless every lane
//! `to` satisfies `from == n - 1 - to`.
//!
//! A term's mask may sit after the shift, naming the byte written
//! (`(x << 8) & 0x00ff0000`), or before it, naming the byte read
//! (`(x & 0x0000ff00) << 8`). A mask is read by its bit pattern at the term's
//! width, so the top lane's `0xff000000` is the same mask whether the constant
//! is carried signed (a negative `I32`) or unsigned. An unmasked term is
//! accepted only when its shift amount is `width - 8`, because only then does
//! the shift alone leave a single byte standing: `x << 24` on a 32-bit value
//! keeps the low byte and discards the rest, which is why the compiler omits
//! the mask there and nowhere else.
//!
//! Every term, every `or` joining them and the value reversed must share one
//! known width. A conversion between them changes which bits a term reads
//! (`zext32(trunc16(x)) >> 8` has lost byte 2 of `x`), so operands are only
//! looked through `operands::peel`, which crosses no width change.
//!
//! # Why the flags must be dead
//!
//! As in [`crate::analysis::rotate`]: [`SsaOp::BSwap`] carries no flags output
//! while the `or` terminating the chain often sets them, so the rewrite drops a
//! definition and is sound only when nothing reads it.

use std::collections::BTreeMap;

use crate::{
    analysis::operands::{const_amount, peel, word_width},
    ir::{SsaFunction, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    target::Target,
};

/// How deep an `or` tree is walked looking for the chain's terms.
///
/// A reversal of a 64-bit value is eight terms, so seven `or` nodes and a
/// depth of seven. The bound keeps a malformed or cyclic chain from walking
/// forever without excluding any shape a compiler emits.
const MAX_OR_DEPTH: usize = 8;

/// The rewrite a byte-reversal chain yields.
pub struct ByteSwapFold<T: Target> {
    /// The operation to replace the matched `or` with, keeping its `dest`.
    pub op: SsaOp<T>,
    /// The idiom that matched, for the pass's event log.
    pub description: &'static str,
}

/// One term of the chain: the value it reads, and which byte goes where.
struct Lane {
    /// The value being reversed, before peeling.
    source: SsaVarId,
    /// The byte of the source this term reads.
    from: u32,
    /// The byte of the result this term writes.
    to: u32,
}

/// Collects the non-`or` leaves of an `or` tree.
fn collect_terms<T: Target>(
    ssa: &SsaFunction<T>,
    var: SsaVarId,
    depth: usize,
    terms: &mut Vec<SsaVarId>,
) {
    if depth >= MAX_OR_DEPTH || terms.len() > 8 {
        return;
    }
    if let Some(SsaOp::Or { left, right, .. }) = ssa.get_definition(peel(ssa, var)) {
        collect_terms(ssa, *left, depth.saturating_add(1), terms);
        collect_terms(ssa, *right, depth.saturating_add(1), terms);
    } else {
        terms.push(var);
    }
}

/// A mask constant's bit pattern at `width` bits.
///
/// Separate from [`const_amount`], which answers in `u32` because shift amounts
/// are small: a 64-bit reversal's top mask is `0xff00000000000000`, which does
/// not fit there at all. Read as a bit pattern rather than a number, because
/// the top lane's mask has the sign bit set: a signed 32-bit `0xff000000` is
/// the negative `I32(-16777216)`, and it masks exactly the bits its unsigned
/// spelling does.
fn const_mask<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    var: SsaVarId,
    width: u32,
) -> Option<u64> {
    let value = constants.get(&peel(ssa, var))?;
    let bits = match value.as_u64() {
        Some(bits) => bits,
        None => value.as_i64()?.cast_unsigned(),
    };
    let keep = 1u64
        .checked_shl(width)
        .map_or(u64::MAX, |bound| bound.wrapping_sub(1));
    Some(bits & keep)
}

/// The byte a single-byte mask selects, or `None` if it selects anything else.
///
/// `0xff << (8 * lane)` names that lane and nothing else. A mask covering two
/// bytes, or one offset from a byte boundary, is some other extraction.
fn mask_lane(mask: u64) -> Option<u32> {
    let lane = mask.trailing_zeros() / 8;
    (mask == 0xffu64.checked_shl(lane.checked_mul(8)?)?).then_some(lane)
}

/// The `(operand, mask)` of an `and` of `var` against a constant, either way
/// round.
fn masked_by<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    var: SsaVarId,
    width: u32,
) -> Option<(SsaVarId, u64)> {
    let SsaOp::And { left, right, .. } = ssa.get_definition(peel(ssa, var))? else {
        return None;
    };
    match const_mask(ssa, constants, *right, width) {
        Some(mask) => Some((*left, mask)),
        None => Some((*right, const_mask(ssa, constants, *left, width)?)),
    }
}

/// The value a term shifts, without needing to know the width first.
///
/// The width is read from this value's own type, so resolving a whole lane to
/// find it would be circular.
fn source_of<T: Target>(ssa: &SsaFunction<T>, term: SsaVarId) -> Option<SsaVarId> {
    let shifted = match ssa.get_definition(peel(ssa, term))? {
        SsaOp::And { left, right, .. } => {
            if matches!(
                ssa.get_definition(peel(ssa, *left)),
                Some(SsaOp::Shl { .. } | SsaOp::Shr { .. })
            ) {
                *left
            } else {
                *right
            }
        }
        _ => term,
    };
    match ssa.get_definition(peel(ssa, shifted))? {
        SsaOp::Shl { value, .. } | SsaOp::Shr { value, .. } => Some(*value),
        _ => None,
    }
}

/// Resolves one term to the byte it moves, or `None` if it moves no single one.
fn lane_of<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    term: SsaVarId,
    width: u32,
) -> Option<Lane> {
    if !ssa.is_read_whole_at(term, width) {
        return None;
    }
    // A mask after the shift, where the compiler emitted one. It names the
    // destination byte.
    let (shifted, mask_after) = match masked_by(ssa, constants, term, width) {
        Some((shifted, mask)) => (shifted, Some(mask)),
        None => (term, None),
    };

    let (value, amount, shifts_left) = match ssa.get_definition(peel(ssa, shifted))? {
        SsaOp::Shl { value, amount, .. } => (*value, const_amount(ssa, constants, *amount)?, true),
        SsaOp::Shr {
            value,
            amount,
            unsigned: true,
            ..
        } => (*value, const_amount(ssa, constants, *amount)?, false),
        _ => return None,
    };
    if amount % 8 != 0 || !ssa.is_read_whole_at(value, width) {
        return None;
    }
    let lanes = width / 8;
    let moved = amount / 8;
    // Where the byte read at `from` lands, and where the byte written at `to`
    // was read.
    let landing = |from: u32| {
        if shifts_left {
            from.checked_add(moved)
        } else {
            from.checked_sub(moved)
        }
    };
    let origin = |to: u32| {
        if shifts_left {
            to.checked_sub(moved)
        } else {
            to.checked_add(moved)
        }
    };

    let (source, from, to) = match mask_after {
        Some(mask) => {
            let to = mask_lane(mask)?;
            (value, origin(to)?, to)
        }
        None => match masked_by(ssa, constants, value, width) {
            // A mask before the shift names the source byte. The shift is
            // logical, so the masked byte is all that moves: nothing else is
            // left set to be shifted in beside it.
            Some((source, mask)) => {
                let from = mask_lane(mask)?;
                (source, from, landing(from)?)
            }
            // Unmasked is only a single byte when the shift discards every
            // other one, which happens at exactly one amount.
            None if amount == width.checked_sub(8)? => {
                let to = if shifts_left {
                    lanes.checked_sub(1)?
                } else {
                    0
                };
                (value, origin(to)?, to)
            }
            None => return None,
        },
    };
    if from >= lanes || to >= lanes || !ssa.is_read_whole_at(source, width) {
        return None;
    }
    Some(Lane { source, from, to })
}

/// Folds a masked shift chain to the byte reversal it spells, when it spells
/// one.
///
/// # Arguments
///
/// * `ssa` - The function, for resolving each operand's defining instruction.
/// * `op` - The candidate operation, folded only when it is an `or`.
/// * `constants` - Known constant values, for the shift amounts and masks.
/// * `uses` - Use counts for the whole function, as
///   [`SsaFunction::count_uses`] reports them, to tell whether the `or`'s flags
///   output is read.
///
/// # Returns
///
/// The byte swap to install in the `or`'s place, or `None` when the terms do
/// not spell a reversal of one value or the `or`'s flags are still read.
pub fn fold<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Option<ByteSwapFold<T>> {
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

    let mut terms = Vec::new();
    collect_terms(ssa, *left, 0, &mut terms);
    collect_terms(ssa, *right, 0, &mut terms);

    // The width is the one the `or` computes at, which the value reversed
    // is read whole at, so the term count and the lane arithmetic are both
    // checked against the same fact.
    let width = word_width(ssa, *dest)?;
    let lanes = width / 8;
    if terms.len() != lanes as usize
        || !ssa.is_read_whole_at(source_of(ssa, *terms.first()?)?, width)
    {
        return None;
    }

    let mut seen = vec![false; lanes as usize];
    let mut source = None;
    for term in &terms {
        let lane = lane_of(ssa, constants, *term, width)?;
        // One value, or it is not a reversal of anything in particular.
        match source {
            None => source = Some(lane.source),
            Some(known) if peel(ssa, known) == peel(ssa, lane.source) => {}
            Some(_) => return None,
        }
        // The reversal condition: byte `from` lands at `n - 1 - from`.
        if lane.from != lanes.checked_sub(1)?.checked_sub(lane.to)? {
            return None;
        }
        let slot = seen.get_mut(lane.to as usize)?;
        if *slot {
            return None;
        }
        *slot = true;
    }
    if !seen.iter().all(|filled| *filled) {
        return None;
    }
    Some(ByteSwapFold {
        op: SsaOp::BSwap {
            dest: *dest,
            src: source?,
        },
        description: "shift chain → bswap",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::SsaFunctionBuilder,
        testing::{MockTarget, MockType, mock_i32},
    };

    /// How the four terms of the chain were built.
    struct Shape {
        /// `true` for the real reversal, `false` for some other permutation.
        reversal: bool,
        /// `true` when every term reads the same value.
        same_value: bool,
        /// `true` to mask a term across two bytes instead of one.
        wide_mask: bool,
        /// A flags output on the outermost `or`, and whether anything reads it.
        flags: Option<bool>,
    }

    impl Shape {
        /// The chain gcc emits at `-O0` for a 32-bit reversal.
        const fn reversal() -> Self {
            Self {
                reversal: true,
                same_value: true,
                wide_mask: false,
                flags: None,
            }
        }
    }

    /// Builds the four-term chain under `shape`, returning the `or` to fold.
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
                let x = block.const_i32(mock_i32(), 0x1234_5678)?;
                let other = block.const_i32(mock_i32(), 0x0bad_f00d)?;
                let c8 = block.const_i32(mock_i32(), 8)?;
                let c24 = block.const_i32(mock_i32(), 24)?;

                // Each term as (shifts_left, amount, mask). The reversal is
                // what gcc emits; the alternative is a byte *rotation*, every
                // byte moving one lane down with the low byte wrapping to the
                // top. That matters: a rotation fills four *distinct* lanes, so
                // the duplicate-lane check cannot reject it and only the
                // reversal condition can. A control that collided two terms on
                // one lane would pass on the duplicate check instead, proving
                // nothing about the rule it names.
                let plan: [(bool, i32, Option<i32>); 4] = if shape.reversal {
                    [
                        (true, 24, None),
                        (
                            true,
                            8,
                            Some(if shape.wide_mask {
                                0x00ff_ff00
                            } else {
                                0x00ff_0000
                            }),
                        ),
                        (false, 8, Some(0x0000_ff00)),
                        (false, 24, None),
                    ]
                } else {
                    [
                        (false, 8, Some(0x0000_00ff)),
                        (false, 8, Some(0x0000_ff00)),
                        (false, 8, Some(0x00ff_0000)),
                        (true, 24, None),
                    ]
                };

                let mut built: Vec<SsaVarId> = Vec::new();
                for (index, (shifts_left, amount, mask)) in plan.into_iter().enumerate() {
                    // One term reads a different value when the shape asks for
                    // it, so "every term reads the same value" is what fails.
                    let read = if !shape.same_value && index == 2 {
                        other
                    } else {
                        x
                    };
                    let by = if amount == 8 { c8 } else { c24 };
                    let shifted = if shifts_left {
                        block.shl(mock_i32(), read, by)?
                    } else {
                        block.shr(mock_i32(), read, by, true)?
                    };
                    let term = match mask {
                        Some(mask) => {
                            let mask = block.const_i32(mock_i32(), mask)?;
                            block.and(mock_i32(), shifted, mask)?
                        }
                        None => shifted,
                    };
                    built.push(term);
                }

                let first = block.or(mock_i32(), built[0], built[1])?;
                let second = block.or(mock_i32(), first, built[2])?;
                let low = built[3];
                let result = block.or(mock_i32(), second, low)?;
                let flags = match shape.flags {
                    Some(_) => Some(block.const_i32(mock_i32(), 0)?),
                    None => None,
                };
                or_op = Some(SsaOp::Or {
                    dest: result,
                    left: second,
                    right: low,
                    flags,
                });
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

    /// The `-O0` chain folds to a byte swap.
    #[test]
    fn the_four_term_chain_folds_to_a_byte_swap() {
        let (ssa, op, constants) = fixture(&Shape::reversal());
        let folded =
            fold(&ssa, &op, &constants, &ssa.count_uses()).expect("the -O0 reversal must fold");
        assert!(
            matches!(folded.op, SsaOp::BSwap { .. }),
            "expected a bswap, got {:?}",
            folded.op
        );
    }

    /// **A reversed value of no stated type is the bits its shifts read.**
    /// The chain's shifts, masks and `or`s state the width it reverses; the
    /// control types the value at another width, whose bytes the 32-bit
    /// chain does not reverse.
    #[test]
    fn an_untyped_value_reverses_at_its_chains_width() {
        let swapped = |ty: MockType| {
            let (mut ssa, op, constants) = fixture(&Shape::reversal());
            let reversed = ssa
                .iter_instructions()
                .find_map(|(_, _, instr)| match instr.op() {
                    SsaOp::Const {
                        dest,
                        value: ConstValue::I32(0x1234_5678),
                    } => Some(*dest),
                    _ => None,
                })
                .expect("the reversed value");
            if let Some(variable) = ssa.variable_mut(reversed) {
                variable.set_type(ty);
            }
            fold(&ssa, &op, &constants, &ssa.count_uses()).map(|fold| fold.op)
        };
        assert!(
            matches!(swapped(MockType::Unknown), Some(SsaOp::BSwap { .. })),
            "{:?}",
            swapped(MockType::Unknown)
        );
        assert_eq!(swapped(MockType::I64), None, "the control");
    }

    /// Bytes assembled in some other order are not a reversal.
    ///
    /// The control that matters most here. Four masked shifts of one value is
    /// the shape of a reversal, a permutation, a field extraction and a
    /// checksum step alike; only where each byte lands separates them, so a
    /// matcher that counted terms would fold all four to `bswap` and be wrong
    /// about three.
    #[test]
    fn another_permutation_does_not_fold() {
        let (ssa, op, constants) = fixture(&Shape {
            reversal: false,
            ..Shape::reversal()
        });
        assert!(
            fold(&ssa, &op, &constants, &ssa.count_uses()).is_none(),
            "a permutation that is not a reversal must not fold to bswap"
        );
    }

    /// A mask spanning two bytes names no single lane.
    #[test]
    fn a_two_byte_mask_does_not_fold() {
        let (ssa, op, constants) = fixture(&Shape {
            wide_mask: true,
            ..Shape::reversal()
        });
        assert!(fold(&ssa, &op, &constants, &ssa.count_uses()).is_none());
    }

    /// Terms reading different values are not one value reversed.
    #[test]
    fn terms_over_different_values_do_not_fold() {
        let (ssa, op, constants) = fixture(&Shape {
            same_value: false,
            ..Shape::reversal()
        });
        assert!(fold(&ssa, &op, &constants, &ssa.count_uses()).is_none());
    }

    /// A byte swap carries no flags, so one that is read keeps the `or`.
    #[test]
    fn a_read_flags_output_refuses_the_fold() {
        let (ssa, op, constants) = fixture(&Shape {
            flags: Some(true),
            ..Shape::reversal()
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
            ..Shape::reversal()
        });
        assert!(fold(&ssa, &op, &constants, &ssa.count_uses()).is_some());
    }
}
