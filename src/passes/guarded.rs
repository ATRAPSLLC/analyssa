//! What a guard states and the ranges it proves on the arm it admits.
//!
//! A comparison or a conditional branch states a relation between two values;
//! taking one arm of the branch proves that relation, or its negation, about
//! them. The range analysis reads it while it checks an assumed loop invariant,
//! [`GuardedRanges::range_at`] reads it over the guards dominating a block, and
//! the jump threader reads it on the path it carries. The refinement lives here,
//! off the public [`GuardedRanges`] trait, so a consumer asks for a range at a
//! block and never for the steps that build it.

use crate::{
    analysis::{
        congruence::Congruence,
        range::{IntervalRange, ValueRange},
    },
    graph::{NodeId, algorithms::DominatorTree},
    ir::{CmpKind, function::SsaFunction, ops::SsaOp, variable::SsaVarId},
    passes::{decide::IntShape, ranges::GuardedRanges},
    target::Target,
};

/// The relation a value-producing comparison states.
#[derive(Debug, Clone, Copy)]
pub struct ComparisonOperands {
    /// Where the boolean result is written.
    pub dest: SsaVarId,
    /// Left operand.
    pub left: SsaVarId,
    /// Right operand.
    pub right: SsaVarId,
    /// Which relation is tested.
    pub cmp: CmpKind,
    /// Whether the operands are ordered as unsigned.
    pub unsigned: bool,
}

/// The relation an op states, whether it computes one or branches on one.
///
/// A comparison instruction names its result; a `BranchCmp` states the same
/// relation inline and names no value. Both say the same thing about their two
/// operands, and a caller asking "what does taking this arm prove" needs either
/// - so this answers for both, with no destination to give.
pub const fn stated_relation<T: Target>(
    op: &SsaOp<T>,
) -> Option<(SsaVarId, SsaVarId, CmpKind, bool)> {
    match op {
        SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned,
            ..
        } => Some((*left, *right, *cmp, *unsigned)),
        _ => match comparison_operands(op) {
            Some(operands) => Some((
                operands.left,
                operands.right,
                operands.cmp,
                operands.unsigned,
            )),
            None => None,
        },
    }
}

/// Reads the relation out of a `Clt`, `Cgt` or `Ceq`, or `None` for anything
/// else.
///
/// Shared because two readers need it and they must agree: the fixpoint's own
/// evaluation of what a comparison *is worth*, and the post-pass fold that
/// replaces it with a constant. Destructured in each of them separately, the
/// two could come to read one comparison differently - one deciding the
/// relation from the operands' ranges while the other answers `0..=1`.
pub const fn comparison_operands<T: Target>(op: &SsaOp<T>) -> Option<ComparisonOperands> {
    let (dest, left, right, cmp, unsigned) = match op {
        SsaOp::Clt {
            dest,
            left,
            right,
            unsigned,
            ..
        } => (*dest, *left, *right, CmpKind::Lt, *unsigned),
        SsaOp::Cgt {
            dest,
            left,
            right,
            unsigned,
            ..
        } => (*dest, *left, *right, CmpKind::Gt, *unsigned),
        SsaOp::Ceq {
            dest, left, right, ..
        } => (*dest, *left, *right, CmpKind::Eq, false),
        _ => return None,
    };
    Some(ComparisonOperands {
        dest,
        left,
        right,
        cmp,
        unsigned,
    })
}

/// The range of `value` after an integer conversion from `source` to
/// `target`.
///
/// The rule is the one [`crate::analysis::convert`] states for the chain
/// collapse, applied to an interval instead of to an operation: a
/// conversion to width `w` yields the low `w` bits of its operand, extended
/// above the operand's width when the operand is narrower - by sign or by
/// zero, according to the conversion's own `unsigned` flag.
///
/// - **Narrowing, or the same width.** The low bits, read in the target
///   type's window: truncating `200` to an unsigned byte is `200`, to a
///   signed one `-56`.
/// - **Widening.** The operand is read in its own width - signed for a
///   sign-extension, unsigned for a zero-extension - and that number is
///   the result: `0x8000_0000` sign-extended from 32 bits is `-2^31`, and
///   `-1` zero-extended from 32 bits is `4294967295`.
///
/// An unknown target proves nothing. An unknown source width leaves one
/// case: a zero-extension or a truncation of a non-negative value is that
/// value in the target's window, whichever of the two it is, while a
/// sign-extension depends on where the source's sign bit sits and proves
/// nothing (zero excepted, which every extension keeps). A conversion that
/// checks overflow traps instead of wrapping, so its non-trapping result is
/// the exact value; treating it like the unchecked form is sound, only less
/// precise.
pub fn convert(
    value: &ValueRange,
    source: Option<IntShape>,
    target: Option<IntShape>,
    unsigned: bool,
) -> ValueRange {
    let Some(target) = target else {
        return ValueRange::top();
    };
    let Some(source) = source else {
        let keeps_its_value =
            (unsigned && value.is_always_non_negative()) || value.as_constant() == Some(0);
        return if keeps_its_value {
            target.hold(value)
        } else {
            ValueRange::top()
        };
    };
    if target.width <= source.width {
        return value.read_bounded(target.width, target.signed);
    }
    let extended = value.read_bounded(source.width, !unsigned);
    extended.read_bounded(target.width, target.signed)
}

/// An exact step from one value to the next: a copy, an integer conversion,
/// or adding or subtracting a constant, each computing its result from one
/// varying operand.
///
/// A guard on either end of a step bounds the other end. The result is a
/// function of the operand, so the operand's range on an arm, stepped
/// forward, holds of the result. Stepped back, the result's range holds of
/// the operand wherever the step loses nothing over what the operand can be:
/// an offset wraps in one window both ends share, an extension's operand is
/// the low bits of its result, and a narrowing keeps a value that already
/// fits the narrower window.
#[derive(Debug, Clone, Copy)]
pub struct Step {
    /// The varying operand the step reads.
    pub from: SsaVarId,
    /// The value the step defines.
    pub to: SsaVarId,
    /// The operand's window, when it has one.
    source: Option<IntShape>,
    /// The result's window, when it has one.
    target: Option<IntShape>,
    /// What the step computes.
    kind: StepKind,
}

/// What a [`Step`] computes from its operand.
#[derive(Debug, Clone, Copy)]
enum StepKind {
    /// The operand, unchanged.
    Copy,
    /// The operand converted to the result's width, extended by zero when
    /// `unsigned` and by sign otherwise.
    Convert {
        /// Whether a widening extends by zero.
        unsigned: bool,
    },
    /// The operand plus a constant, wrapping in the window both share.
    Offset(i64),
}

impl Step {
    /// The step defining `var`, if its definition is one, with `constant`
    /// answering which values are a known constant.
    pub fn defining<T: Target>(
        ssa: &SsaFunction<T>,
        var: SsaVarId,
        constant: impl Fn(SsaVarId) -> Option<i64>,
    ) -> Option<Self> {
        let (from, kind) = match ssa.get_definition(var)? {
            SsaOp::Copy { src, .. } => (*src, StepKind::Copy),
            SsaOp::IntConv {
                operand, unsigned, ..
            } => (
                *operand,
                StepKind::Convert {
                    unsigned: *unsigned,
                },
            ),
            SsaOp::Add { left, right, .. } => match (constant(*left), constant(*right)) {
                (_, Some(offset)) => (*left, StepKind::Offset(offset)),
                (Some(offset), None) => (*right, StepKind::Offset(offset)),
                (None, None) => return None,
            },
            SsaOp::Sub { left, right, .. } => {
                (*left, StepKind::Offset(constant(*right)?.checked_neg()?))
            }
            _ => return None,
        };
        let (source, target) = (IntShape::of_var(ssa, from), IntShape::of_var(ssa, var));
        // An offset wraps in one window, which both ends must share.
        if matches!(kind, StepKind::Offset(_)) && (source.is_none() || source != target) {
            return None;
        }
        Some(Self {
            from,
            to: var,
            source,
            target,
            kind,
        })
    }

    /// The steps `var` is computed by, the first reading the value they all
    /// start from and the last defining `var`; none when `var` is no step's.
    ///
    /// Walked by its definitions, as many as the function has values, so a
    /// chain the binary makes long costs no stack.
    pub fn chain<T: Target>(
        ssa: &SsaFunction<T>,
        var: SsaVarId,
        constant: impl Fn(SsaVarId) -> Option<i64>,
    ) -> Vec<Self> {
        let mut steps = Vec::new();
        let mut at = var;
        for _ in 0..ssa.variable_count() {
            let Some(step) = Self::defining(ssa, at, &constant) else {
                break;
            };
            at = step.from;
            steps.push(step);
        }
        steps.reverse();
        steps
    }

    /// The result's range when the operand lies in `range`.
    pub fn forward(&self, range: &ValueRange) -> ValueRange {
        match self.kind {
            StepKind::Copy => self
                .target
                .map_or_else(|| range.clone(), |shape| shape.hold(range)),
            StepKind::Convert { unsigned } => convert(range, self.source, self.target, unsigned),
            StepKind::Offset(offset) => self.target.map_or_else(ValueRange::top, |shape| {
                shape.hold(&range.add(&ValueRange::constant(offset)))
            }),
        }
    }

    /// The operand's range when the result lies in `range`, if the step loses
    /// nothing over what the operand can be.
    ///
    /// With `known`, what the operand is otherwise known to be, the answer is
    /// sound: a narrowing steps back only where `known` fits the narrower
    /// window. Without it, the answer is the candidate a guess takes, which
    /// assumes a narrowing loses nothing.
    pub fn backward(&self, range: &ValueRange, known: Option<&ValueRange>) -> Option<ValueRange> {
        match self.kind {
            StepKind::Copy => Some(
                self.source
                    .map_or_else(|| range.clone(), |shape| shape.hold(range)),
            ),
            StepKind::Offset(offset) => {
                let shape = self.source?;
                Some(shape.hold(&range.sub(&ValueRange::constant(offset))))
            }
            StepKind::Convert { .. } => {
                let (source, target) = (self.source?, self.target?);
                // An extension's operand is the low bits of its result.
                if target.width > source.width {
                    return Some(range.read_bounded(source.width, source.signed));
                }
                // A narrowing keeps the number of an operand that fits.
                let Some(known) = known else {
                    return Some(source.hold(range));
                };
                let window = ValueRange::window(target.width, target.signed);
                (known.join(&window) == window).then(|| range.meet(known))
            }
        }
    }
}

/// The refinement steps behind [`GuardedRanges::range_at`], answered for every
/// guarded range source.
pub trait GuardRefinement: GuardedRanges {
    /// `var`'s range at `block`, refined by the guards that dominate it
    /// ([`GuardedRanges::range_at`]).
    ///
    /// A guard bounds `var` itself and every value `var` is computed from by
    /// [`Step`]s: the value they start from is refined first, and each step's
    /// result is what the step makes of its operand, met with what the guards
    /// prove of the result itself.
    fn guarded_at<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        dominators: &DominatorTree,
        var: SsaVarId,
        block: usize,
    ) -> ValueRange {
        self.along_steps(ssa, var, |link| {
            self.guarded_on(ssa, dominators, link, block)
        })
    }

    /// `var`'s range refined at each value of the [`Step`]s it is computed
    /// by: `refined` answers for one value, and each step carries what the
    /// value before it holds forward into the next.
    fn along_steps<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        var: SsaVarId,
        refined: impl Fn(SsaVarId) -> ValueRange,
    ) -> ValueRange {
        let steps = Step::chain(ssa, var, |value| {
            self.proved(value).and_then(ValueRange::as_constant)
        });
        let Some(first) = steps.first() else {
            return refined(var);
        };
        let mut range = refined(first.from);
        for step in &steps {
            range = refined(step.to).meet(&step.forward(&range));
        }
        range
    }

    /// `var`'s range at `block`, refined by the guards that dominate it on
    /// `var` itself.
    fn guarded_on<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        dominators: &DominatorTree,
        var: SsaVarId,
        block: usize,
    ) -> ValueRange {
        let mut range = self.proved(var).cloned().unwrap_or_default();
        let mut guarded = false;
        let predecessors = self.entering();
        let mut current = NodeId::new(block);
        // Bound the climb by the block count: a dominator chain cannot be
        // longer, and this keeps a malformed tree from looping.
        for _ in 0..ssa.block_count() {
            let Some(parent) = dominators.immediate_dominator(current) else {
                break;
            };
            if let Some(refined) =
                self.refine_through_edge(ssa, dominators, predecessors, var, parent, current)
            {
                range = range.meet(&refined);
                guarded = true;
            }
            current = parent;
        }
        // A guard cuts one side of the window the value still lies in: an
        // unknown byte below `127` is one of `[0, 126]`, not of `(-inf, 126]`.
        // A value no guard reaches stays as known.
        match IntShape::of_var(ssa, var) {
            Some(shape) if guarded => range.meet(&ValueRange::window(shape.width, shape.signed)),
            _ => range,
        }
    }

    /// Returns what `parent`'s terminator proves about `var` on the path that
    /// reaches `child`, if anything.
    fn refine_through_edge<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        dominators: &DominatorTree,
        predecessors: &[usize],
        var: SsaVarId,
        parent: NodeId,
        child: NodeId,
    ) -> Option<ValueRange> {
        let block = ssa.block(parent.index())?;
        let terminator = block.control_terminator()?;
        // Every conditional terminator, not only `Branch`. A comparison is as
        // often fused into the transfer as computed before it - it always is on
        // a native target - and reading only the unfused spelling is how this
        // refinement learned nothing at all about a lifted binary.
        let (true_target, false_target) = terminator.two_way_targets()?;
        if true_target == false_target {
            return None;
        }
        // An arm only guards what every path to `child` enters through it: its
        // target must be reached from the branch alone and dominate `child`.
        let through = |target: usize| {
            predecessors.get(target) == Some(&1)
                && (child.index() == target || dominators.dominates(NodeId::new(target), child))
        };
        let taken = match (through(true_target), through(false_target)) {
            (true, false) => true,
            (false, true) => false,
            _ => return None,
        };
        self.refine_leaving(ssa, var, parent.index(), taken)
    }

    /// What `block`'s two-way terminator proves about `var` on the arm it
    /// takes when its condition is `taken`, if anything: the relation a
    /// `BranchCmp` states, the one a `Branch` names, or a flag test's against
    /// zero.
    fn refine_leaving<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        var: SsaVarId,
        block: usize,
        taken: bool,
    ) -> Option<ValueRange> {
        let terminator = ssa.block(block)?.control_terminator()?;
        match terminator {
            // The relation is stated inline.
            SsaOp::BranchCmp { .. } => {
                let relation = stated_relation(terminator)?;
                self.refine_from_relation(ssa, relation, var, taken)
            }
            // The relation is what produced the flags, read against zero:
            // `flags` describe a value, and the condition says which side of
            // zero the arm proves it is on.
            SsaOp::BranchFlags {
                flags, condition, ..
            } => {
                if ssa.flag_source(*flags)? != var {
                    return None;
                }
                let (cmp, var_is_left) = condition.as_relation_to_zero()?;
                let shape = IntShape::of_var(ssa, var);
                self.refine_against(
                    var,
                    shape,
                    &ValueRange::Interval(IntervalRange::new(Some(0), Some(0))),
                    shape,
                    cmp,
                    false,
                    taken,
                    var_is_left,
                )
            }
            // A separate comparison the condition names.
            SsaOp::Branch { condition, .. } => {
                let comparison = ssa.get_definition(*condition)?;
                self.refine_from_comparison(ssa, comparison, var, taken)
            }
            _ => None,
        }
    }

    /// Intersects what a comparison proves about `var` when it evaluates to
    /// `taken`.
    ///
    /// Only the constant-bounded forms are used: `x < k`, `x <= k`, `x > k`,
    /// `x >= k`, their negations, and `x == k` on its taken arm. `k` is read from the other
    /// operand's proved range, so a comparison against a computed-but-bounded
    /// value refines too.
    ///
    /// The bound is only meaningful in the window both ranges are held in (a
    /// value's integer width and sign), so it applies only where the comparison reads both
    /// operands that way: a signed ordering over signed-held values of one
    /// width, or an unsigned ordering over values proved non-negative - where
    /// the unsigned order and the held one agree.
    fn refine_from_comparison<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        comparison: &SsaOp<T>,
        var: SsaVarId,
        taken: bool,
    ) -> Option<ValueRange> {
        let (left, right, cmp, unsigned) = stated_relation(comparison)?;
        self.refine_from_relation(ssa, (left, right, cmp, unsigned), var, taken)
    }

    /// Intersects what a stated relation proves about `var` when it holds
    /// (`taken`) or does not.
    ///
    /// The relation is `(left, right, comparison, unsigned)`, as a comparison
    /// or a `BranchCmp` terminator states it, so a caller holding one - an
    /// edge it is about to reroute - can ask without a comparison instruction
    /// to point at.
    fn refine_from_relation<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        relation: (SsaVarId, SsaVarId, CmpKind, bool),
        var: SsaVarId,
        taken: bool,
    ) -> Option<ValueRange> {
        let (left, right, cmp, unsigned) = relation;
        let other = if left == var {
            right
        } else if right == var {
            left
        } else {
            return self.refine_stepped_into(ssa, relation, var, taken);
        };
        let (var_shape, other_shape) = (IntShape::of_var(ssa, var), IntShape::of_var(ssa, other));
        if var_shape.map(|shape| shape.width) != other_shape.map(|shape| shape.width) {
            return None;
        }
        let limit = self.proved(other)?;
        self.refine_against(
            var,
            var_shape,
            limit,
            other_shape,
            cmp,
            unsigned,
            taken,
            left == var,
        )
    }

    /// What a stated relation proves about `var` where it compares a value
    /// `var` steps into: the operand's refinement, stepped back through each
    /// [`Step`] to `var`. `x + 1 != 512` bounds `x` where it bounds `x + 1`,
    /// and `(int32_t)i < 10` bounds a 64-bit `i` known to fit 32 bits.
    fn refine_stepped_into<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        relation: (SsaVarId, SsaVarId, CmpKind, bool),
        var: SsaVarId,
        taken: bool,
    ) -> Option<ValueRange> {
        let (left, right, ..) = relation;
        [left, right].into_iter().find_map(|operand| {
            let steps = Step::chain(ssa, operand, |value| {
                self.proved(value).and_then(ValueRange::as_constant)
            });
            let from = steps.iter().rposition(|step| step.from == var)?;
            let mut range = self.refine_from_relation(ssa, relation, operand, taken)?;
            for step in steps.get(from..)?.iter().rev() {
                range = step.backward(&range, Some(self.proved(step.from)?))?;
            }
            Some(range)
        })
    }

    /// The half of [`Self::refine_from_relation`] that reads a limit rather
    /// than looks one up.
    ///
    /// A relation's limit is usually another variable's proved range, but not
    /// always: a flag test states a relation to zero, which no variable names.
    /// Both ask the same question of the same interval arithmetic, so both ask
    /// it here.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable being refined.
    /// * `var_shape` - `var`'s integer window, when it has one.
    /// * `limit` - What the relation compares `var` against.
    /// * `other_shape` - The limit's integer window, when it has one.
    /// * `cmp` - The comparison.
    /// * `unsigned` - Whether it is an unsigned comparison.
    /// * `taken` - Whether the relation holds or is negated.
    /// * `var_is_left` - Whether `var` is the comparison's left operand, which
    ///   is what decides the direction of an asymmetric relation.
    #[allow(clippy::too_many_arguments)] // Each is a distinct fact about one relation.
    fn refine_against(
        &self,
        var: SsaVarId,
        var_shape: Option<IntShape>,
        limit: &ValueRange,
        other_shape: Option<IntShape>,
        cmp: CmpKind,
        unsigned: bool,
        taken: bool,
        var_is_left: bool,
    ) -> Option<ValueRange> {
        // A bound the guard makes is moved in to the nearest value `var`'s
        // residues admit: below `512`, a multiple of four is at most `508`.
        // A residue is of the value's bits, which read as the same number
        // modulo the modulus only at a width the modulus divides.
        let residues = var_shape
            .filter(|shape| shape.width >= Congruence::MODULUS_BITS)
            .map(|_| self.congruence(var));
        let lower = |bound: i64| {
            residues
                .and_then(|residues| residues.greatest_at_most(bound))
                .unwrap_or(bound)
        };
        let raise = |bound: i64| {
            residues
                .and_then(|residues| residues.least_at_least(bound))
                .unwrap_or(bound)
        };
        if matches!(cmp, CmpKind::Eq | CmpKind::Ne) {
            // `x == k` pins `x` to `k`, read in `x`'s window.
            if taken == matches!(cmp, CmpKind::Eq) {
                return Some(var_shape.map_or_else(|| limit.clone(), |shape| shape.hold(limit)));
            }
            // `x != k` is a hole, which an interval cannot express - *unless*
            // the hole is at an end of what `x` already is, and then it is a
            // shorter interval. `x` in `[2, 3]` with `x != 2` is `x == 3`, and
            // that is how a compiler spells a two-case dispatch: a state
            // written as one of two literals and tested against one of them.
            let excluded = limit.min().filter(|min| Some(*min) == limit.max())?;
            let known = self.proved(var)?;
            let (low, high) = (known.min()?, known.max()?);
            return match (low == excluded, high == excluded) {
                (true, false) => low
                    .checked_add(1)
                    .map(raise)
                    .map(|low| ValueRange::Interval(IntervalRange::new(Some(low), Some(high)))),
                (false, true) => high
                    .checked_sub(1)
                    .map(lower)
                    .map(|high| ValueRange::Interval(IntervalRange::new(Some(low), Some(high)))),
                _ => None,
            };
        }
        // A value held in an unsigned window narrower than 64 bits is
        // non-negative by that window, whatever else is known of it: an
        // unknown byte a switch compares unsigned is one of `[0, 255]`.
        let held_as_read = |shape: Option<IntShape>, range: Option<&ValueRange>| {
            if unsigned {
                range.is_some_and(ValueRange::is_always_non_negative)
                    || shape.is_some_and(|shape| !shape.signed && shape.width < 64)
            } else {
                shape.is_none_or(|shape| shape.signed)
            }
        };
        if !held_as_read(var_shape, self.proved(var)) || !held_as_read(other_shape, Some(limit)) {
            return None;
        }
        // The relation as it holds of `var`: `var` on the left, and the arm
        // taken. All four orderings, each exactly - a fused `BranchCmp` states
        // `<=` and `>=` as often as `<` and `>`, and reading every ordering
        // but `<` as `>` put `x` below 37 on the arm that proves `x > 36`.
        let relation = if var_is_left { cmp } else { cmp.swapped() };
        let relation = if taken { relation } else { relation.negated() };
        let at_least = |bound: i64| ValueRange::Interval(IntervalRange::at_least(raise(bound)));
        // An unsigned relation reads `var` as a number no less than zero,
        // which is what made it one to read here: below a limit it is in
        // `[0, limit)`. `(n - 1 < 8191) ? n - 1 : 8191` bounds an unknown
        // count to the 8191 it is clamped to.
        let at_most = |bound: i64| {
            let high = lower(bound);
            let low = raise(0);
            ValueRange::Interval(if unsigned && low <= high {
                IntervalRange::new(Some(low), Some(high))
            } else {
                IntervalRange::at_most(high)
            })
        };
        match relation {
            // var < limit: var <= limit_max - 1
            CmpKind::Lt => limit.max().and_then(|m| m.checked_sub(1)).map(at_most),
            // var <= limit: var <= limit_max
            CmpKind::Le => limit.max().map(at_most),
            // var > limit: var >= limit_min + 1
            CmpKind::Gt => limit.min().and_then(|m| m.checked_add(1)).map(at_least),
            // var >= limit: var >= limit_min
            CmpKind::Ge => limit.min().map(at_least),
            // Handled above: an ordering negates to an ordering.
            CmpKind::Eq | CmpKind::Ne => None,
        }
    }
}

impl<G: GuardedRanges + ?Sized> GuardRefinement for G {}
