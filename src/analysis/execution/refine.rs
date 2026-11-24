//! What one side of a fork knows from the condition it took.
//!
//! A branch the explorer cannot decide sends a path down each side. Each side
//! is refined by the condition it took, read in the one form [`Question`]
//! reads a branch: a value the side proves equal to a number holds that
//! number, and so does each value the number undoes to
//! ([`Equality::implied`]); a value the side proves ordered against a number
//! keeps the bits that order fixes ([`IntegerBits::bounded`]); a branch's own
//! truth value is the one the side took. A later branch on a refined value,
//! or on one computed from it, then decides, and a side no input satisfies is
//! no path.
//!
//! A refinement is read in place of the value until the value is defined
//! again ([`Frame::refine`]); what the value was defined as is kept, so two
//! paths that join keep the identity both gave it.
//!
//! [`Frame::refine`]: crate::analysis::execution::frame::Frame::refine

use std::cell::OnceCell;

use crate::{
    analysis::{
        SsaCfg,
        consts::{Machine, OperandLayout},
        dataflow::{AnalysisResults, DataFlowSolver, LiveVariables, LivenessResult},
        execution::{
            state::State,
            types::TypeLayouts,
            value::{Root, Value},
        },
        guards::{Definitions, Equality, Question},
    },
    ir::{
        function::SsaFunction,
        integer::IntegerBits,
        ops::{CmpKind, FlagCondition, SsaOp},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// What the conditions a path took establish of one integer: the bits they
/// fix, and the numbers they rule out that the bits still admit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Facts {
    /// What is known of its bits.
    bits: IntegerBits,
    /// Numbers it is not, in order: each one the bits admit.
    excluded: Vec<u128>,
}

impl Facts {
    /// How many excluded numbers one value keeps; a further inequality is
    /// dropped, which only forgets.
    const EXCLUDED: usize = 4;

    /// What `bits` alone say.
    pub const fn of(bits: IntegerBits) -> Self {
        Self {
            bits,
            excluded: Vec::new(),
        }
    }

    /// What is known of the bits.
    pub const fn bits(&self) -> IntegerBits {
        self.bits
    }

    /// These facts with `value cmp bound` holding as well, compared unsigned
    /// or two's complement; `None` when no value they admit satisfies it.
    ///
    /// An order or equality adds the bits it fixes
    /// ([`IntegerBits::bounded`]) and keeps the excluded numbers the new bits
    /// still admit; an inequality excludes its number, where the bits admit
    /// it. A number excluded is never the value.
    #[must_use]
    pub fn bounded(&self, cmp: CmpKind, unsigned: bool, bound: u128) -> Option<Self> {
        let bound = bound & self.bits.mask();
        if cmp == CmpKind::Ne {
            if self.bits.value() == Some(bound) {
                return None;
            }
            let mut excluded = self.excluded.clone();
            if self.admits(bound)
                && excluded.len() < Self::EXCLUDED
                && let Err(at) = excluded.binary_search(&bound)
            {
                excluded.insert(at, bound);
            }
            return Some(Self {
                bits: self.bits,
                excluded,
            });
        }
        let bits = self.bits.bounded(cmp, unsigned, bound)?;
        let narrowed = Self::of(bits);
        let excluded: Vec<u128> = self
            .excluded
            .iter()
            .copied()
            .filter(|number| narrowed.admits(*number))
            .collect();
        if bits.value().is_some_and(|value| excluded.contains(&value)) {
            return None;
        }
        Some(Self { bits, excluded })
    }

    /// Whether the bits admit `number`.
    fn admits(&self, number: u128) -> bool {
        IntegerBits::constant(self.bits.width(), number)
            .is_some_and(|number| self.bits.compare(CmpKind::Eq, true, number) != Some(false))
    }
}

/// What the conditions a path took establish of one value: what the value
/// is read as, and the facts that say so, which identify the refinement.
#[derive(Debug, Clone, PartialEq)]
pub struct Refined<T: Target> {
    /// What the value is read as.
    value: Value<T>,
    /// What is known of it.
    facts: Facts,
}

impl<T: Target> Refined<T> {
    /// What the value is read as.
    pub const fn value(&self) -> &Value<T> {
        &self.value
    }

    /// What is known of it.
    pub const fn facts(&self) -> &Facts {
        &self.facts
    }
}

/// What one function's forks are refined through, and which refinements
/// still matter where: derived from the function once a fork in it first
/// needs them, and shared by every path through it.
pub struct Conditions<T: Target> {
    /// The function's definitions, which a question and an equality are read
    /// through.
    definitions: OnceCell<Definitions<T>>,
    /// The values live on entry to each of its blocks.
    live: OnceCell<AnalysisResults<LivenessResult>>,
}

impl<T: Target> Default for Conditions<T> {
    fn default() -> Self {
        Self {
            definitions: OnceCell::new(),
            live: OnceCell::new(),
        }
    }
}

impl<T: Target> Conditions<T> {
    /// The definitions of `ssa`, the function these conditions are of.
    pub fn definitions(&self, ssa: &SsaFunction<T>) -> &Definitions<T> {
        self.definitions.get_or_init(|| Definitions::of(ssa))
    }

    /// Whether `value` may be read at or after the entry to `block` of `ssa`,
    /// the function these conditions are of; every value is, where the
    /// block is outside it.
    pub fn live_in(&self, ssa: &SsaFunction<T>, block: usize, value: SsaVarId) -> bool {
        self.live
            .get_or_init(|| {
                DataFlowSolver::new(LiveVariables::new(ssa)).solve(ssa, &SsaCfg::from_ssa(ssa))
            })
            .in_state(block)
            .is_none_or(|live| live.is_live(value))
    }
}

/// The sides of one function's forks, each refined by the condition it
/// took.
pub struct Taken<'a, T: Target> {
    /// The function whose terminators fork.
    ssa: &'a SsaFunction<T>,
    /// Its definitions, which a question and an equality are read through.
    definitions: &'a Definitions<T>,
    /// The declared layout of each of its values.
    layouts: &'a TypeLayouts<'a, T>,
    /// The machine whose integers the values are.
    machine: Machine,
}

impl<'a, T: Target> Taken<'a, T> {
    /// How many definitions a branch's truth value is read back through.
    const DEPTH: usize = 16;

    /// The sides of `ssa`'s forks, read through `definitions` at `layouts`.
    pub const fn new(
        ssa: &'a SsaFunction<T>,
        definitions: &'a Definitions<T>,
        layouts: &'a TypeLayouts<'a, T>,
        machine: Machine,
    ) -> Self {
        Self {
            ssa,
            definitions,
            layouts,
            machine,
        }
    }

    /// Refines `state` by the condition `terminator` took to send it to
    /// `target`; `false` when no input satisfies that condition together
    /// with what `state` already holds, so the side is no path.
    ///
    /// A terminator that asks no question, and a switch case reached by
    /// several indices, refine nothing.
    pub fn refine(&self, terminator: &SsaOp<T>, target: usize, state: &mut State<T>) -> bool {
        match terminator {
            SsaOp::Branch {
                condition,
                true_target,
                ..
            } => {
                let taken = target == *true_target;
                if !self.truth(*condition, taken, state) {
                    return false;
                }
                let (question, holds) = Question::of_value(*condition, self.definitions);
                self.answered(question, taken == holds, state)
            }
            SsaOp::BranchCmp { true_target, .. } | SsaOp::BranchFlags { true_target, .. } => {
                Question::of_terminator(terminator, self.definitions).is_none_or(
                    |(question, holds)| {
                        self.answered(question, (target == *true_target) == holds, state)
                    },
                )
            }
            SsaOp::Switch {
                value,
                targets,
                default,
            } => {
                let mut cases = targets
                    .iter()
                    .enumerate()
                    .filter(|(_, case)| **case == target)
                    .map(|(index, _)| index);
                match (cases.next(), cases.next()) {
                    (Some(index), None) if target != *default => u128::try_from(index)
                        .is_ok_and(|index| self.settle(*value, CmpKind::Eq, true, index, state)),
                    (None, _) if target == *default => u128::try_from(targets.len())
                        .is_ok_and(|count| self.settle(*value, CmpKind::Ge, true, count, state)),
                    _ => true,
                }
            }
            _ => true,
        }
    }

    /// Fixes the truth value `condition` and each truth value it was copied
    /// or negated from to the one the side `taken` holds; `false` when one
    /// already holds the other.
    fn truth(&self, condition: SsaVarId, taken: bool, state: &mut State<T>) -> bool {
        let (mut current, mut holds) = (condition, taken);
        for _ in 0..Self::DEPTH {
            if !matches!(self.layouts.operand(current), OperandLayout::Boolean) {
                return true;
            }
            match state.value(current) {
                Value::Const(constant) => {
                    if constant.as_bool().is_some_and(|held| held != holds) {
                        return false;
                    }
                }
                _ => state.refine(
                    current,
                    IntegerBits::constant(1, u128::from(holds)).map(|bits| Refined {
                        value: Value::Const(ConstValue::from_bool(holds)),
                        facts: Facts::of(bits),
                    }),
                ),
            }
            match self.definitions.get_definition(current) {
                Some(SsaOp::Copy { src, .. }) => current = *src,
                Some(SsaOp::BoolNot { value, .. }) => {
                    current = *value;
                    holds = !holds;
                }
                _ => return true,
            }
        }
        true
    }

    /// Refines `state` by `question` answered `held`; `false` when nothing
    /// it holds answers so.
    fn answered(&self, question: Question, held: bool, state: &mut State<T>) -> bool {
        match question {
            Question::Compare {
                left,
                right,
                cmp,
                unsigned,
            } => {
                if [left, right]
                    .iter()
                    .any(|value| matches!(self.layouts.operand(*value), OperandLayout::Float(_)))
                {
                    return true;
                }
                let cmp = if held { cmp } else { cmp.negated() };
                let (left_bound, right_bound) = (self.whole(left, state), self.whole(right, state));
                right_bound.is_none_or(|bound| self.settle(left, cmp, unsigned, bound, state))
                    && left_bound.is_none_or(|bound| {
                        self.settle(right, cmp.swapped(), unsigned, bound, state)
                    })
            }
            Question::Flags { flags, condition } => {
                // The flags describe the result of the operation that set
                // them, where it has one besides them.
                let Some(result) = self
                    .ssa
                    .recorded_definition(flags)
                    .and_then(|definition| definition.op().dest())
                    .filter(|result| *result != flags)
                else {
                    return true;
                };
                let condition = if held { condition } else { condition.negated() };
                let (cmp, unsigned) = match condition {
                    FlagCondition::Zero => (CmpKind::Eq, true),
                    FlagCondition::NotZero => (CmpKind::Ne, true),
                    FlagCondition::Negative => (CmpKind::Lt, false),
                    FlagCondition::Positive => (CmpKind::Ge, false),
                    FlagCondition::Carry
                    | FlagCondition::NotCarry
                    | FlagCondition::Overflow
                    | FlagCondition::NotOverflow
                    | FlagCondition::ParityEven
                    | FlagCondition::ParityOdd => return true,
                };
                self.settle(result, cmp, unsigned, 0, state)
            }
            Question::Value(value) => {
                let cmp = if held { CmpKind::Ne } else { CmpKind::Eq };
                self.settle(value, cmp, true, 0, state)
            }
        }
    }

    /// The number `value` holds on this path, where it holds one whole: its
    /// bits at its declared integer layout.
    fn whole(&self, value: SsaVarId, state: &State<T>) -> Option<u128> {
        let layout = self.layouts.operand(value);
        self.known(&state.value(value), layout)?.value()
    }

    /// What is known of the bits of `held`, read at `layout`: a constant
    /// whole, a partial integer as far as it is known, a numeric address as
    /// its number. An address of any other root is no number known here.
    fn known(&self, held: &Value<T>, layout: OperandLayout) -> Option<IntegerBits> {
        let width = layout.scalar_integer()?.bits();
        match held {
            Value::Const(constant) => self.machine.constant_integer_bits(constant, layout),
            Value::Integer(_) => held.integer_bits().filter(|bits| bits.width() == width),
            Value::Addr(address) if address.root() == Root::Number => {
                IntegerBits::constant(width, u128::from(address.offset().cast_unsigned()))
            }
            Value::Addr(_) | Value::StackDerived | Value::Unknown => None,
        }
    }

    /// Refines `value` by `value cmp bound` holding, and, for an equality,
    /// each value the number undoes to; `false` when what this path holds
    /// of one of them satisfies no such number.
    fn settle(
        &self,
        value: SsaVarId,
        cmp: CmpKind,
        unsigned: bool,
        bound: u128,
        state: &mut State<T>,
    ) -> bool {
        if !self.bounded(value, cmp, unsigned, bound, state) {
            return false;
        }
        if cmp != CmpKind::Eq {
            return true;
        }
        let Some(width) = self
            .layouts
            .operand(value)
            .scalar_integer()
            .map(|ty| ty.bits())
        else {
            return true;
        };
        let Ok(bits) = u64::try_from(bound) else {
            return true;
        };
        Equality::new(value, bits, width)
            .implied(self.definitions, self.machine.ptr_size)
            .into_iter()
            .skip(1)
            .all(|implied| {
                self.bounded(
                    implied.value(),
                    CmpKind::Eq,
                    true,
                    u128::from(implied.bits()),
                    state,
                )
            })
    }

    /// Refines `value` alone by `value cmp bound` holding; `false` when what
    /// this path holds of it satisfies no such number.
    ///
    /// A constant and a numeric address are judged and kept. A partial
    /// integer or a named input keeps what the bound adds: read as the number
    /// it is proved equal to, or as the partial integer the bits leave it,
    /// or, for a pointer (whose bits say nothing worth the object it names),
    /// as itself with the numbers it is not. An address of a known object is
    /// judged only where it can be no null, and kept.
    fn bounded(
        &self,
        value: SsaVarId,
        cmp: CmpKind,
        unsigned: bool,
        bound: u128,
        state: &mut State<T>,
    ) -> bool {
        let layout = self.layouts.operand(value);
        let Some(width) = layout.scalar_integer().map(|ty| ty.bits()) else {
            return true;
        };
        let refinement = state
            .frame()
            .and_then(|frame| frame.refinement(value))
            .cloned();
        let held = state.value(value);
        // What is known of it, what it was defined as where it takes a bound,
        // and nothing more where it is known whole.
        let (facts, defined) = match (&refinement, &held) {
            (Some(refined), _) => (Some(refined.facts.clone()), Some(held.clone())),
            (None, Value::Addr(address)) if address.is_named() => (
                IntegerBits::unknown(width).map(Facts::of),
                Some(held.clone()),
            ),
            (None, Value::Addr(address)) if address.root() != Root::Number => {
                let never_null = matches!(address.root(), Root::Heap(_)) && address.offset() == 0;
                return !(never_null && cmp == CmpKind::Eq && bound == 0);
            }
            (None, Value::Integer(_)) => {
                (self.known(&held, layout).map(Facts::of), Some(held.clone()))
            }
            (None, _) => (self.known(&held, layout).map(Facts::of), None),
        };
        let Some(facts) = facts else {
            return true;
        };
        let Some(refined) = facts.bounded(cmp, unsigned, bound) else {
            return false;
        };
        let Some(defined) = defined.filter(|_| refined != facts) else {
            return true;
        };
        let stack_derived = defined.may_reference_stack();
        let pointer = self
            .ssa
            .variable(value)
            .is_some_and(|variable| T::is_pointer(variable.var_type()));
        let bits = refined.bits();
        // A value that may reference the stack keeps that provenance as a
        // partial integer, however much of it is known.
        let whole = bits
            .value()
            .filter(|_| !stack_derived)
            .and_then(|pattern| match layout {
                OperandLayout::Boolean => Some(ConstValue::from_bool(pattern != 0)),
                OperandLayout::Register(dest) => self.machine.integer_bits_constant(bits, dest),
                OperandLayout::Float(_) => None,
            });
        let read = match whole {
            Some(constant) => Value::Const(constant),
            None if !pointer && !bits.is_unknown() => Value::partial_integer(bits, stack_derived),
            None => defined,
        };
        state.refine(
            value,
            Some(Refined {
                value: read,
                facts: refined,
            }),
        );
        true
    }
}
