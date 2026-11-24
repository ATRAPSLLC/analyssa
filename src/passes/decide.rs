//! Deciding a terminator from what is proved about the values it reads.
//!
//! The range analysis decides a branch while it propagates, to prune the edge
//! not taken, and again when it folds the terminator; the jump threader asks
//! the same of the values a path carries. One decision, read by all three, so
//! they cannot disagree about which edge a value takes.

use crate::{
    analysis::{congruence::Congruence, range::ValueRange},
    ir::{
        CmpKind,
        function::SsaFunction,
        ops::{SsaOp, native::FlagCondition},
        variable::SsaVarId,
    },
    target::Target,
};

/// What the analysis proved about one operand of a comparison.
///
/// The two domains travel together because the decision reads both and they
/// must describe the same variable at the same point of the same fixpoint.
/// Fetching one and then the other is how a caller comes to pair a refined
/// interval with a residue set from an earlier iteration.
#[derive(Debug, Clone, Copy)]
pub struct Known<'a> {
    /// What the interval domain proved.
    pub range: &'a ValueRange,
    /// What the congruence domain proved.
    pub congruence: Congruence,
}

/// The target `op` transfers to, given what is *proved* about its operands
/// rather than what they are.
///
/// **One decider for every conditional terminator, and one authority for each
/// kind of question.** A comparison is settled by [`decide_comparison`] and a
/// flag test by [`ValueRange::satisfies_flag`]; this routes each terminator to
/// the right one rather than re-deriving either. It is the terminator family in
/// one place: `Branch` names a condition variable, `BranchCmp` states a relation
/// inline, `BranchFlags` tests a bit of a flags value, and `Switch` indexes. A
/// caller that reasons about "where does this go when I know the values" wants
/// all four or none, and enumerating them by hand is how `passes::ranges` came
/// to have no `BranchCmp` arm, `passes::controlflow` no `BranchFlags` arm, and
/// `passes::threading` nothing but `Branch`.
///
/// It asks what is *proved* about an operand rather than what it is, because a
/// single value is the special case: `x` in `[3, 3]` decides `x > 2` exactly as
/// the constant 3 does, and `x` in `[2, 3]` decides `x != 0`, which no single
/// value could stand for.
///
/// # Arguments
///
/// * `op` - The terminator to decide.
/// * `ssa` - The function, to find what set a flags value.
/// * `known` - What is proved about a variable on the path in question, or
///   `None` where nothing is. For a flags variable the lookup is made on the
///   *producing instruction's result* (see [`SsaFunction::flag_source`]).
pub fn taken_target_known<T: Target>(
    op: &SsaOp<T>,
    ssa: &SsaFunction<T>,
    known: &mut dyn FnMut(SsaVarId) -> Option<(ValueRange, Congruence)>,
) -> Option<usize> {
    // One value, where one is all a rule needs: `Branch` and `Switch` read an
    // index, not an interval. Every other rule reads `known` directly.
    let value_of = |known: &mut dyn FnMut(SsaVarId) -> Option<(ValueRange, Congruence)>,
                    var: SsaVarId|
     -> Option<i64> {
        let (range, _) = known(var)?;
        range.min().filter(|min| Some(*min) == range.max())
    };
    match op {
        SsaOp::Branch {
            condition,
            true_target,
            false_target,
        } => Some(if value_of(known, *condition)? == 0 {
            *false_target
        } else {
            *true_target
        }),
        SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned,
            true_target,
            false_target,
        } => {
            let (left_known, right_known) = (known(*left)?, known(*right)?);
            let taken = decide_operands(
                ssa,
                *cmp,
                *unsigned,
                (
                    *left,
                    Known {
                        range: &left_known.0,
                        congruence: left_known.1,
                    },
                ),
                (
                    *right,
                    Known {
                        range: &right_known.0,
                        congruence: right_known.1,
                    },
                ),
            )?;
            Some(if taken { *true_target } else { *false_target })
        }
        SsaOp::BranchFlags {
            flags,
            condition,
            true_target,
            false_target,
        } => {
            let source = ssa.flag_source(*flags)?;
            let value = value_of(known, source)?;
            let taken = flag_holds(ssa, source, &ValueRange::constant(value), *condition)?;
            Some(if taken { *true_target } else { *false_target })
        }
        SsaOp::Switch {
            value,
            targets,
            default,
        } => {
            let index = usize::try_from(value_of(known, *value)?).ok()?;
            Some(targets.get(index).copied().unwrap_or(*default))
        }
        _ => None,
    }
}

/// How an integer variable's range is held: the machine width and the window
/// its values are read in.
///
/// Every range this pass records for a variable of known integer width lies in
/// that variable's window - `[-2^(w-1), 2^(w-1)-1]` for a signed type,
/// `[0, 2^w - 1]` for an unsigned one - so one machine value has exactly one
/// reading and two variables of the same type compare directly. A 64-bit value
/// is always held signed: an `i64` bound cannot name the upper half of the
/// unsigned window (see [`ValueRange::wrap`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntShape {
    /// Width in bits, `1..=64`.
    pub width: u32,
    /// Whether the value is held in the signed window.
    pub signed: bool,
}

impl IntShape {
    /// The shape of a value of type `ty`, or `None` when the target gives it
    /// no integer width this domain can hold.
    pub fn of_type<T: Target>(ty: &T::Type) -> Option<Self> {
        let width = T::bit_width(ty).filter(|width| (1..=64).contains(width))?;
        Some(Self {
            width,
            signed: width == 64 || T::is_signed(ty),
        })
    }

    /// The shape of `var`'s declared type.
    pub fn of_var<T: Target>(ssa: &SsaFunction<T>, var: SsaVarId) -> Option<Self> {
        ssa.variable(var)
            .and_then(|variable| Self::of_type::<T>(variable.var_type()))
    }

    /// `range` reduced to machine values of this shape.
    pub fn hold(self, range: &ValueRange) -> ValueRange {
        range.wrap(self.width, self.signed)
    }

    /// The shape an operation's result is held in.
    ///
    /// The destination's recovered type first, and the operands' when it has
    /// none: an operation whose result has its operands' width computes in
    /// that width whatever type recovery made of the destination (see
    /// [`SsaOp::width_operand`]). `None` when neither says.
    pub fn of_result<T: Target>(ssa: &SsaFunction<T>, op: &SsaOp<T>) -> Option<Self> {
        if let Some(dest) = op.dest()
            && let Some(shape) = Self::of_var(ssa, dest)
        {
            return Some(shape);
        }
        op.width_operand(|operand| ssa.bit_width(operand))
            .and_then(|operand| Self::of_var(ssa, operand))
    }
}

/// `range`, held in `shape`, read as a `width`-bit value in the `signed`
/// window.
///
/// A narrower operand compared against a wider one is extended first, and
/// which extension applies is the producer's business rather than the
/// comparison's, so the answer covers both: the sign-extended reading and the
/// zero-extended one.
fn view_at(range: &ValueRange, shape: IntShape, width: u32, signed: bool) -> ValueRange {
    if shape.width >= width {
        return range.wrap(width, signed);
    }
    let sign_extended = range.wrap(shape.width, true).wrap(width, signed);
    let zero_extended = range.wrap(shape.width, false).wrap(width, signed);
    sign_extended.join(&zero_extended)
}

/// Both operands of a comparison, read the way the comparison reads them.
///
/// A signed ordering reads both in the signed window, an unsigned one in the
/// unsigned window, and an equality in either (the signed one here) - the same
/// window for both sides, so equal bits are equal numbers. An operand of
/// unknown width is left as recorded, and so is its partner.
fn comparison_view<T: Target>(
    ssa: &SsaFunction<T>,
    cmp: CmpKind,
    unsigned: bool,
    (left, left_range): (SsaVarId, &ValueRange),
    (right, right_range): (SsaVarId, &ValueRange),
) -> (ValueRange, ValueRange) {
    let (Some(left_shape), Some(right_shape)) =
        (IntShape::of_var(ssa, left), IntShape::of_var(ssa, right))
    else {
        return (left_range.clone(), right_range.clone());
    };
    let width = left_shape.width.max(right_shape.width);
    let signed = !unsigned || matches!(cmp, CmpKind::Eq | CmpKind::Ne);
    (
        view_at(left_range, left_shape, width, signed),
        view_at(right_range, right_shape, width, signed),
    )
}

/// Decides `left cmp right` from what is known of each operand, reading both
/// as the comparison does (see [`comparison_view`]).
///
/// The one entry point every comparison goes through - the fixpoint's pruning,
/// its evaluation of a materialised comparison, the post-pass fold and
/// [`taken_target_known`] - so none of them can read an operand in a window the
/// others do not.
pub fn decide_operands<T: Target>(
    ssa: &SsaFunction<T>,
    cmp: CmpKind,
    unsigned: bool,
    (left, left_known): (SsaVarId, Known<'_>),
    (right, right_known): (SsaVarId, Known<'_>),
) -> Option<bool> {
    // Two addresses the range domain holds no number for are decided as
    // the constants they are
    // ([`ConstValue::symbol_relation`](crate::ir::value::ConstValue::symbol_relation)):
    // `deregister_tm_clones` compares `__TMC_END__` loaded twice.
    if let (Some(left), Some(right)) = (ssa.symbol_of(left), ssa.symbol_of(right))
        && let Some(holds) = left.symbol_relation(right, cmp, unsigned)
    {
        return Some(holds);
    }
    let (left_range, right_range) = comparison_view(
        ssa,
        cmp,
        unsigned,
        (left, left_known.range),
        (right, right_known.range),
    );
    decide_comparison(
        cmp,
        unsigned,
        Known {
            range: &left_range,
            congruence: left_known.congruence,
        },
        Known {
            range: &right_range,
            congruence: right_known.congruence,
        },
    )
}

/// Whether `condition` holds of the value `source` computed, given its range.
///
/// The sign flag is the top bit at the operation's width, so the range is read
/// in the signed window of `source`'s width first: an unsigned byte holding
/// `200` has its sign bit set.
pub fn flag_holds<T: Target>(
    ssa: &SsaFunction<T>,
    source: SsaVarId,
    range: &ValueRange,
    condition: FlagCondition,
) -> Option<bool> {
    match IntShape::of_var(ssa, source) {
        Some(shape) => range.wrap(shape.width, true).satisfies_flag(condition),
        None => range.satisfies_flag(condition),
    }
}

pub fn decide_comparison(
    cmp: CmpKind,
    unsigned: bool,
    left: Known<'_>,
    right: Known<'_>,
) -> Option<bool> {
    let (left_range, right_range) = (left.range, right.range);
    let ordering = !matches!(cmp, CmpKind::Eq | CmpKind::Ne);
    if unsigned
        && ordering
        && !(left_range.is_always_non_negative() && right_range.is_always_non_negative())
    {
        return None;
    }
    match cmp {
        CmpKind::Eq | CmpKind::Ne => {
            // **The congruence decides what the intervals cannot.** Equal values
            // are congruent modulo every modulus, so operands whose residues are
            // disjoint cannot be equal - and that settles pairs no interval
            // separates. `a*a` and `7*b*b - 1` overlap completely as intervals
            // and share no residue at all, which is the whole reason the domain
            // exists.
            //
            // Safe to read *during* the fixpoint, alongside the intervals, for
            // one reason: the domain is monotone in the executable-edge set. A
            // value's residues are joined, never replaced, and a value nothing
            // has computed reads `Top` rather than the empty set - so a fact
            // computed from a partial set of predecessors is always an
            // over-approximation of the one computed from the whole set, and an
            // edge pruned on it stays prunable when the set grows.
            //
            // Getting that wrong is not theoretical. Seeding uncomputed values
            // to the empty set would make an entry value - a stack pointer no
            // transfer function defines - claim it cannot exist; the emptiness
            // spreads through every arithmetic operation that reads it, and a
            // phi joining it reports only its *other* operand's residues. A
            // flattened dispatcher then reads its state as `{0, 1}` when the
            // program also writes 2 and 3, and the `state == 2` arm is deleted
            // as unreachable.
            if left.congruence.disjoint(right.congruence) {
                return Some(matches!(cmp, CmpKind::Ne));
            }
            let equal =
                if let (Some(l), Some(r)) = (left_range.as_constant(), right_range.as_constant()) {
                    l == r
                } else if ranges_overlap(left_range, right_range) {
                    return None;
                } else {
                    false
                };
            Some(if matches!(cmp, CmpKind::Ne) {
                !equal
            } else {
                equal
            })
        }
        CmpKind::Lt => {
            if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min())
                && l_max < r_min
            {
                return Some(true);
            }
            if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max())
                && l_min >= r_max
            {
                return Some(false);
            }
            None
        }
        CmpKind::Le => {
            if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min())
                && l_max <= r_min
            {
                return Some(true);
            }
            if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max())
                && l_min > r_max
            {
                return Some(false);
            }
            None
        }
        CmpKind::Gt => {
            if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max())
                && l_min > r_max
            {
                return Some(true);
            }
            if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min())
                && l_max <= r_min
            {
                return Some(false);
            }
            None
        }
        CmpKind::Ge => {
            if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max())
                && l_min >= r_max
            {
                return Some(true);
            }
            if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min())
                && l_max < r_min
            {
                return Some(false);
            }
            None
        }
    }
}

/// Returns `true` when two ranges could share at least one value.
///
/// Used by the range pass's comparison folding to decide an equality: operands whose
/// ranges are disjoint can never be equal. `Top` (unknown) is treated as
/// overlapping everything and `Bottom` (unreachable) as overlapping nothing;
/// a half-open range conservatively counts as overlapping.
///
/// # Arguments
///
/// * `a` - First range.
/// * `b` - Second range.
///
/// # Returns
///
/// `true` if the ranges are not provably disjoint.
pub fn ranges_overlap(a: &ValueRange, b: &ValueRange) -> bool {
    if a.is_top() || b.is_top() {
        return true;
    }
    if a.is_bottom() || b.is_bottom() {
        return false;
    }
    match (a.max(), a.min(), b.max(), b.min()) {
        (Some(a_max), Some(a_min), Some(b_max), Some(b_min)) => a_max >= b_min && a_min <= b_max,
        _ => true,
    }
}
