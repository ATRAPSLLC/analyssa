//! Rules that fold a value to a constant: of constant operands, of known bits,
//! of every bit known, and by the algebraic identities.

use crate::{
    analysis::{
        algebraic::{SimplifyResult, simplify_op},
        bits::Reader,
        consts::Destination,
    },
    ir::{ops::SsaOp, variable::SsaVariable},
    passes::algebraic::{Simplification, SimplificationCandidate, site::Site},
    target::Target,
};

/// General constant folding: an op whose every value operand is already a
/// known constant evaluates to a constant, whatever the op is. `simplify_op`
/// below only knows algebraic *identities* (`x & -1`, `x ^ x`), not the
/// arithmetic of two literals, and nothing else in the pipeline folds a
/// conversion of a constant - so a split constant built by `movz`/`movk`,
/// `lui`/`ori` or `movw`/`movt` (a narrow half masked and or-ed with a
/// shifted half, each half a literal) reaches the page as that bit-arithmetic
/// instead of the number it computes. Evaluated here, the chain collapses one
/// level per fixpoint iteration until only the literal remains.
///
/// A `Const` is skipped: it already *is* the literal, and folding it to
/// itself is a no-op the pass would report as a change and never settle on.
/// So is a copy between two values of one type: copy propagation forwards
/// its source to every reader, which keeps one constant where a fold would
/// make two. A copy into another type is one it refuses, and folded here.
pub fn const_fold<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        machine,
        ..
    } = *site;
    let dest = site.dest?;
    let forwarded = match op {
        SsaOp::Copy { dest, src } => {
            ssa.variable(*dest).map(SsaVariable::var_type)
                == ssa.variable(*src).map(SsaVariable::var_type)
        }
        _ => false,
    };
    if !matches!(op, SsaOp::Const { .. })
        && !forwarded
        && let Some(value) = machine.evaluate(op, Destination::of_operation(ssa, op), |var| {
            constants.get(&var).cloned()
        })
    {
        return Some(site.candidate(dest, Simplification::Constant(value), "const fold"));
    }
    None
}

/// Bits read out of a value known in part (a lane of a lane-wise
/// operation, a register merged over an unknown one) are known however the
/// rest is not.
pub fn partly_known<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa,
        op,
        constants,
        machine,
        ..
    } = *site;
    let dest = site.dest?;
    if let Some(value) = machine.fold_known_bits(ssa, op, constants) {
        return Some(site.candidate(dest, Simplification::Constant(value), "lane fold"));
    }
    None
}

/// A bit operation whose every result bit is known is a constant, though
/// an operand is not: a field read back after a write of it into a
/// container nothing else says. See [`Reader::known_constant`](crate::analysis::bits::Reader::known_constant).
pub fn every_bit_known<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    let dest = site.dest?;
    if let Some(value) = Reader::new(ssa, constants).known_constant(op) {
        return Some(site.candidate(dest, Simplification::Constant(value), "every bit known"));
    }
    None
}

/// The *operand* type decides whether a self-cancelling identity holds
/// (`x - x` is NaN for a NaN float) and at what width its constant should be
/// materialised. A comparison's destination is a boolean, so the destination
/// type would answer neither question.
pub fn algebraic<T: Target>(site: &Site<'_, T>) -> Option<SimplificationCandidate<T>> {
    let Site {
        ssa, op, constants, ..
    } = *site;
    let dest = site.dest?;
    let operand_type = op
        .first_use()
        .and_then(|operand| ssa.variable(operand))
        .map(|var| var.var_type().clone());
    match simplify_op(op, constants, operand_type.as_ref()) {
        SimplifyResult::Constant(value) => {
            Some(site.candidate(dest, Simplification::Constant(value), "algebraic → const"))
        }
        SimplifyResult::Copy(src) => {
            Some(site.candidate(dest, Simplification::Copy(src), "algebraic → copy"))
        }
        SimplifyResult::None => None,
    }
}
