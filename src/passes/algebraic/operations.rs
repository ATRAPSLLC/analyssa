//! Rules for an operation that states what it reads: a flag, a rounding mode, a
//! shuffle index, a count, a lane control. Each rewrites the operation to read
//! what is known of the input in its place.

use std::collections::BTreeMap;

use crate::{
    analysis::{
        algebraic::executing_nothing, bits::Reader, consts::OperandLayout,
        convert::narrow_extended_comparison, doubleword::Doubleword,
    },
    ir::{
        float::FloatRounding,
        function::SsaFunction,
        ops::{SsaOp, VectorShuffleSelector},
        value::ConstValue,
        variable::SsaVarId,
    },
    passes::algebraic::{Simplification, SimplificationCandidate, site::Site},
    target::{Target, VectorShuffleMask},
};

/// The double-width arithmetic, the exact float functions of several
/// results and the computations of several define each over constant
/// operands: one `Const` each, where every result folds, since the
/// replacement defines no other.
pub fn results<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        machine,
        ..
    } = *site;
    if let Some(results) = machine.evaluate_results_with_operand_layouts(
        op,
        |var| constants.get(&var).cloned(),
        |var| OperandLayout::of(ssa, var),
    ) && results.len() == op.defs().count()
        && let Some(&(dest, _)) = results.first()
    {
        return Some(
            site.candidate(
                dest,
                Simplification::Results(
                    results
                        .into_iter()
                        .map(|(dest, value)| SsaOp::Const { dest, value })
                        .collect(),
                ),
                "const fold of every result",
            ),
        );
    }
    None
}

/// A computation over lanes that executes no element changes nothing and
/// touches nothing: each result is what it leaves.
pub fn idle_lanes<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    if let Some(results) = executing_nothing(op, constants, |var| {
        ssa.variable(var)
            .map(|variable| variable.var_type().clone())
    }) && let Some(dest) = op.defs().next().or_else(|| op.first_use())
    {
        return Some(site.candidate(
            dest,
            Simplification::Results(results),
            "lane computation executing no element",
        ));
    }
    None
}

/// A block operation whose direction flag is known clear walks upward, as
/// one that reads no flag does: the flag, and its reader, go.
pub fn clear_direction<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { op, constants, .. } = *site;
    if let Some(flag) = op.direction_flag()
        && constants.get(&flag).and_then(ConstValue::as_bool) == Some(false)
    {
        let mut upward = op.clone();
        if let Some(backward) = upward.direction_flag_mut() {
            *backward = None;
        }
        return Some(site.candidate(
            flag,
            Simplification::Rewrite(Box::new(upward)),
            "block operation with a clear direction flag walks upward",
        ));
    }
    None
}

/// An operation rounding by the environment's mode, the mode known, rounds
/// by that mode as one stating it does: the mode operand, and its reader,
/// go. So does the operation a raise is of.
pub fn known_mode<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site { op, constants, .. } = *site;
    let mut computed = op;
    while let SsaOp::FloatRaise { op: inner, .. } = computed {
        computed = inner;
    }
    if let Some(mode) = computed.environment()
        && let Some(rounding) = constants
            .get(&mode)
            .and_then(ConstValue::as_u64)
            .and_then(FloatRounding::of_environment)
    {
        let mut rounded = op.clone();
        let mut target = &mut rounded;
        while let SsaOp::FloatRaise { op: inner, .. } = target {
            target = inner.as_mut();
        }
        *target = target.clone().rounded_by(rounding);
        return Some(site.candidate(
            mode,
            Simplification::Rewrite(Box::new(rounded)),
            "an operation rounding by a known mode states it",
        ));
    }
    None
}

/// A shuffle by a known index vector selects what a fixed mask does: the
/// index, and its reader, go (`pshufb` by a constant control).
pub fn known_shuffle<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    if let SsaOp::VectorShuffle {
        dest,
        left,
        right,
        selector: VectorShuffleSelector::Index(index),
    } = op
        && let Some(ConstValue::Vector(indices)) = constants.get(index)
        && let Some(index_shape) = ssa.vector_shape(*index)
        && let Some(left_shape) = ssa.vector_shape(*left)
        && let Some(right_lanes) = match right {
            Some(right) => ssa.vector_shape(*right).map(|right| Some(right.lane_count)),
            None => Some(None),
        }
        && let Some(indices) = indices.relaned(index_shape.lane_bits)
        && let Some(mask) =
            VectorShuffleMask::of_indices(indices.lane_values(), left_shape.lane_count, right_lanes)
    {
        return Some(site.candidate(
            *index,
            Simplification::Rewrite(Box::new(SsaOp::VectorShuffle {
                dest: *dest,
                left: *left,
                right: *right,
                selector: VectorShuffleSelector::Mask(mask),
            })),
            "a shuffle by a known index selects by mask",
        ));
    }
    None
}

/// Two stores of a double-width value's halves to one cell's two words are
/// the store of the value, where the byte order says which word is which.
pub fn store_pair<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        block_idx,
        instr_idx,
        constants,
        machine,
        ..
    } = *site;
    if let SsaOp::StoreIndirect { value, .. } = op
        && let Some(pair) =
            Doubleword::new(ssa, constants, machine).store_pair(block_idx, instr_idx)
    {
        return Some(site.candidate(
            *value,
            Simplification::StorePair(pair),
            "a double-width value stored as its halves",
        ));
    }
    None
}

/// A comparison of values extended from one narrower width is the
/// comparison of the values extended. See [`narrow_extended_comparison`]:
/// a machine without narrow comparisons compares its narrow values' whole
/// registers, and a reader would see the extensions where the values are.
/// A branch defines nothing, and its first operand stands for what it
/// rewrites.
pub fn narrowed_comparison<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, machine, ..
    } = *site;
    let ptr_size = machine.ptr_size;
    if let Some(narrowed) = narrow_extended_comparison(ssa, op, ptr_size)
        && let Some(dest) = op.defs().next().or_else(|| op.first_use())
    {
        return Some(site.candidate(
            dest,
            if narrowed.inserted.is_empty() {
                Simplification::Rewrite(Box::new(narrowed.op))
            } else {
                Simplification::Narrowed(Box::new(narrowed.op), narrowed.inserted, narrowed.ty)
            },
            "comparison of extended values compares them at the narrow width",
        ));
    }
    None
}

/// A repeated compare, scan or load whose count is never zero is never
/// empty, so what an empty run would leave is read by nothing.
pub fn never_empty<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    if let Some(nonempty) = nonempty_run(ssa, op, constants) {
        return Some(site.candidate(
            op.dest()?,
            Simplification::Rewrite(Box::new(nonempty)),
            "a repeated block operation whose count is never zero",
        ));
    }
    None
}

/// `op`, a repeated compare, scan or load (`rep cmps`, `repne scas`, `rep
/// lods`), without the input an empty run leaves, where its count is never
/// zero: a constant other than zero, or a value with a bit known set
/// ([`KnownBits::never_zero`](crate::analysis::bits::KnownBits::never_zero)).
/// x86's `strlen` idiom scans with a count of `-1` (`or ecx, -1; repne
/// scasb`), and the flags it would leave when empty were whatever an earlier
/// compare left, a computation no reader needed. `None` for any other op, and
/// for one that states no such input already.
fn nonempty_run<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<SsaOp<T>> {
    let never_zero = |count: SsaVarId| {
        constants
            .get(&count)
            .and_then(ConstValue::as_u64_bits)
            .is_some_and(|bits| bits != 0)
            || Reader::new(ssa, constants)
                .known_bits(count)
                .is_some_and(|known| known.never_zero())
    };
    match op {
        SsaOp::ScanBlk {
            count,
            flags_in: Some(_),
            ..
        } if never_zero(*count) => {
            let mut nonempty = op.clone();
            if let SsaOp::ScanBlk { flags_in, .. } = &mut nonempty {
                *flags_in = None;
            }
            Some(nonempty)
        }
        SsaOp::CompareBlk(data) if data.flags_in.is_some() && never_zero(data.count) => {
            let mut data = data.clone();
            data.flags_in = None;
            Some(SsaOp::CompareBlk(data))
        }
        SsaOp::LoadBlk {
            count,
            value_in: Some(_),
            ..
        } if never_zero(*count) => {
            let mut nonempty = op.clone();
            if let SsaOp::LoadBlk { value_in, .. } = &mut nonempty {
                *value_in = None;
            }
            Some(nonempty)
        }
        _ => None,
    }
}
