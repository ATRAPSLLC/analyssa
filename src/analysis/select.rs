//! A choice another choice has already made.
//!
//! A guarded write lifts as a select of the new value and the old: ARM's
//! `moveq r1, r2` is `r1 = eq ? r2 : r1`, x86's `cmovne rax, rcx` is
//! `rax = ne ? rcx : rax`. Two of them under complementary conditions write
//! the register whichever way the condition falls - Thumb's `ite eq; moveq r1,
//! r2; movne r1, r3` - and lift as `!eq ? r3 : (eq ? r2 : r1)`. The inner
//! select is reached only where `!eq` fails, so where `eq` holds, and there it
//! is `r2`: the old `r1` is never chosen. Left in place it is a read of `r1`
//! the program does not make, and a function opening with such a pair takes
//! `r1` as a parameter it does not have.
//!
//! # The rule
//!
//! An arm of `select(c, t, f)` that is itself `select(d, x, y)` is decided by
//! `c` when `d` is `c` or its negation: the true arm is reached where `c`
//! holds, the false arm where it fails, and each picks the inner arm `d` then
//! picks. Conditions are compared as one value under [`SsaOp::BoolNot`]s,
//! counted for their parity: `d` is `c` when both reach the same value through
//! as many negations, modulo two, and its negation otherwise. "The same value"
//! is one variable, or two copies of one computation over the same values:
//! never what a comparison computes, so the rule holds for a float condition
//! as for an integer one.
//!
//! A select whose condition is whether its own two arms are equal is the
//! other kind of choice made: `a != b ? a : b` takes `a` where the two
//! differ and `b`, which is `a`, where they do not. A compiler leaves one
//! where a test it duplicated has been merged back, the copy's value and the
//! original's now one. Floats are left alone: `+0` and `-0` compare equal and
//! are two values.
//!
//! A choice between two booleans one of which is its condition, or the
//! condition negated, is their conjunction or disjunction: `c ? c : f` is `c
//! || f`, `c ? t : c` is `t && c`, `c ? !c : f` is `!c && f` and `c ? t : !c`
//! is `t || !c`. ARM's `cmp r0, #61; cmpne r0, #0` leaves a branch on `!(r0
//! == 61) ? r0 == 0 : r0 == 61`, the one test `r0 == 0 || r0 == 61`.

use crate::{
    ir::{
        function::SsaFunction,
        ops::{OperandRole, SsaOp},
        variable::SsaVarId,
    },
    target::Target,
};

/// How many negations, and how many nested selects, are looked through.
///
/// A lift writes one negation per complemented guard and one select per
/// guarded write; a block of guarded writes to one register is a handful.
const MAX_DEPTH: usize = 8;

/// The value `condition` tests and whether it tests it negated: the value
/// under every [`SsaOp::BoolNot`] it is defined by, and the parity of their
/// number.
fn tested<T: Target>(ssa: &SsaFunction<T>, condition: SsaVarId) -> (SsaVarId, bool) {
    let mut value = condition;
    let mut negated = false;
    for _ in 0..MAX_DEPTH {
        let Some(SsaOp::BoolNot { value: inner, .. }) = ssa.get_definition(value) else {
            break;
        };
        value = *inner;
        negated = !negated;
    }
    (value, negated)
}

/// Whether `a` and `b` are one value: the same variable, two equal constants,
/// or two definitions of one computation ([`SsaOp::computes_from_operands`])
/// over operands that are one value in turn, looked through `depth` levels.
///
/// A lift writes a flag read or a comparison once for each instruction that
/// tests it (`moveq` and `movne` read the zero flag each), and nothing numbers
/// the copies before this fold reads them: compared by variable alone, the
/// two reads of one condition are two conditions.
fn same_value<T: Target>(ssa: &SsaFunction<T>, a: SsaVarId, b: SsaVarId, depth: usize) -> bool {
    if a == b {
        return true;
    }
    let Some(depth) = depth.checked_sub(1) else {
        return false;
    };
    let (Some(first), Some(second)) = (ssa.get_definition(a), ssa.get_definition(b)) else {
        return false;
    };
    if let (SsaOp::Const { value: x, .. }, SsaOp::Const { value: y, .. }) = (first, second) {
        return x == y;
    }
    if !first.computes_from_operands() || !second.computes_from_operands() {
        return false;
    }
    let (first_uses, second_uses) = (first.uses(), second.uses());
    if first_uses.len() != second_uses.len()
        || !first_uses
            .iter()
            .zip(&second_uses)
            .all(|(x, y)| same_value(ssa, *x, *y, depth))
    {
        return false;
    }
    // One computation: the same operation, defining nothing in particular,
    // over the first's operands.
    let restated = |op: &SsaOp<T>| {
        let mut op = op.clone();
        let mut uses = first_uses.iter();
        op.visit_operands_mut(|role, var| match role {
            OperandRole::Def | OperandRole::FlagsDef => *var = SsaVarId::PLACEHOLDER,
            OperandRole::Use => {
                if let Some(&used) = uses.next() {
                    *var = used;
                }
            }
        });
        op
    };
    restated(first) == restated(second)
}

/// Whether `value` holds a boolean, by its type.
fn boolean<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId) -> bool {
    ssa.variable(value)
        .is_some_and(|variable| T::is_boolean(variable.var_type()))
}

/// Whether `value` is known not to be a float: its type is neither a
/// floating one nor unknown.
fn integral<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId) -> bool {
    ssa.variable(value).is_some_and(|variable| {
        let ty = variable.var_type();
        !T::is_floating(ty) && !T::is_unknown(ty)
    })
}

/// The value `arm` is where `condition` is known to be `holds`: through every
/// select on that condition or its negation, the arm the known outcome picks.
fn decided<T: Target>(
    ssa: &SsaFunction<T>,
    condition: (SsaVarId, bool),
    holds: bool,
    arm: SsaVarId,
) -> SsaVarId {
    let mut value = arm;
    for _ in 0..MAX_DEPTH {
        let Some(SsaOp::Select {
            condition: inner,
            true_val,
            false_val,
            ..
        }) = ssa.get_definition(value)
        else {
            break;
        };
        let (tests, negated) = tested(ssa, *inner);
        if !same_value(ssa, tests, condition.0, MAX_DEPTH) {
            break;
        }
        // The inner condition holds exactly when the outer one is `holds`,
        // flipped once for each side that tests it negated.
        let inner_holds = holds ^ negated ^ condition.1;
        value = if inner_holds { *true_val } else { *false_val };
    }
    value
}

/// Folds a select whose condition is a constant, or compares its two arms
/// for equality, or whose arms are selects the same condition decides, or a
/// select of booleans one arm of which is its condition or its negation.
///
/// A constant condition is a choice made: the select is a copy of the arm it
/// picks, whatever that arm holds (a flags value, a register's old value),
/// where constant propagation folds it only when the arm is a constant too.
/// A condition that is whether the two integer arms are equal, or not, picks
/// one value either way: `a != b ? a : b` is `a`, and `a == b ? a : b` is
/// `b`.
///
/// # Arguments
///
/// * `ssa` - The function `op` is in, for the arms' definitions.
/// * `op` - The candidate operation.
///
/// # Returns
///
/// A copy of the arm a constant condition picks, the select reading the
/// decided arms, a copy when both arms are then one value, the `And` or `Or`
/// of two boolean arms, or `None` when nothing is decided.
#[must_use]
pub fn fold<T: Target>(ssa: &SsaFunction<T>, op: &SsaOp<T>) -> Option<SsaOp<T>> {
    let SsaOp::Select {
        dest,
        condition,
        true_val,
        false_val,
    } = op
    else {
        return None;
    };
    let tests = tested(ssa, *condition);
    if let Some(SsaOp::Const { value, .. }) = ssa.get_definition(tests.0)
        && let Some(truth) = value.as_bool()
    {
        let holds = truth ^ tests.1;
        let arm = if holds { *true_val } else { *false_val };
        return Some(SsaOp::Copy {
            dest: *dest,
            src: decided(ssa, tests, holds, arm),
        });
    }
    // A choice between the two values its condition compares for equality
    // is the arm it takes where they differ: where they are equal, either arm
    // is that value. Integers only: two floats compare equal as `+0` and `-0`
    // while their bits differ.
    if let Some(SsaOp::Ceq { left, right, .. }) = ssa.get_definition(tests.0)
        && [*left, *right].iter().all(|value| integral(ssa, *value))
        && ((same_value(ssa, *left, *true_val, MAX_DEPTH)
            && same_value(ssa, *right, *false_val, MAX_DEPTH))
            || (same_value(ssa, *left, *false_val, MAX_DEPTH)
                && same_value(ssa, *right, *true_val, MAX_DEPTH)))
    {
        return Some(SsaOp::Copy {
            dest: *dest,
            src: if tests.1 { *true_val } else { *false_val },
        });
    }
    // A boolean arm that is the condition, or its negation, is the value
    // where it is taken: `c ? c : f` is `c || f`, `c ? !c : f` is `!c && f`,
    // and the false arm alike, `c ? t : c` being `t && c` and `c ? t : !c`
    // `t || !c`. Whether an arm is the condition (`false`) or its negation
    // (`true`):
    if [*dest, *true_val, *false_val]
        .iter()
        .all(|value| boolean(ssa, *value))
    {
        let condition_or_negation = |arm: SsaVarId| {
            let (value, negated) = tested(ssa, arm);
            same_value(ssa, value, tests.0, MAX_DEPTH).then_some(negated ^ tests.1)
        };
        let disjunction = match (
            condition_or_negation(*true_val),
            condition_or_negation(*false_val),
        ) {
            (Some(false), _) | (_, Some(true)) => Some(true),
            (Some(true), _) | (_, Some(false)) => Some(false),
            (None, None) => None,
        };
        if let Some(disjunction) = disjunction {
            let (dest, left, right) = (*dest, *true_val, *false_val);
            return Some(if disjunction {
                SsaOp::Or {
                    dest,
                    left,
                    right,
                    flags: None,
                }
            } else {
                SsaOp::And {
                    dest,
                    left,
                    right,
                    flags: None,
                }
            });
        }
    }
    let chosen = decided(ssa, tests, true, *true_val);
    let otherwise = decided(ssa, tests, false, *false_val);
    if chosen == *true_val && otherwise == *false_val {
        return None;
    }
    if chosen == otherwise {
        return Some(SsaOp::Copy {
            dest: *dest,
            src: chosen,
        });
    }
    Some(SsaOp::Select {
        dest: *dest,
        condition: *condition,
        true_val: chosen,
        false_val: otherwise,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{SsaDefSpec, SsaFunctionBuilder},
        testing::{MockTarget, MockType},
    };

    /// The selects of one block, in order, after building `build`.
    fn function(
        build: impl FnOnce(&mut crate::ir::SsaBlockBuilder<'_, MockTarget>) -> crate::Result<SsaVarId>,
    ) -> (SsaFunction<MockTarget>, SsaOp<MockTarget>) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(4, 0);
        let last = builder
            .in_block(0, |block| {
                let last = build(block)?;
                block.ret(Some(last))?;
                Ok(last)
            })
            .expect("the block builds");
        let ssa = builder.finish().expect("the function builds");
        let op = ssa.get_definition(last).expect("defined").clone();
        (ssa, op)
    }

    /// The value a fold copies, or `None` when it folds to anything else.
    fn copied(op: Option<SsaOp<MockTarget>>) -> Option<SsaVarId> {
        match op {
            Some(SsaOp::Copy { src, .. }) => Some(src),
            _ => None,
        }
    }

    fn arg(block: &mut crate::ir::SsaBlockBuilder<'_, MockTarget>, index: u16) -> SsaVarId {
        block
            .load_arg(SsaDefSpec::argument(index, MockType::I32), index)
            .expect("an argument")
    }

    /// **A guarded pair under complementary conditions never keeps the old
    /// value.** `!c ? b : (c ? a : old)` is `!c ? b : a`.
    #[test]
    fn a_complementary_pair_reads_no_old_value() {
        let mut seen = None;
        let (ssa, op) = function(|block| {
            let condition = arg(block, 0);
            let (old, a, b) = (arg(block, 1), arg(block, 2), arg(block, 3));
            let first = block.select(SsaDefSpec::tmp(MockType::I32), condition, a, old)?;
            let negated = block.bool_not(SsaDefSpec::tmp(MockType::I1), condition)?;
            seen = Some((negated, a, b));
            block.select(SsaDefSpec::tmp(MockType::I32), negated, b, first)
        });
        let (negated, a, b) = seen.expect("built");
        let Some(SsaOp::Select {
            condition,
            true_val,
            false_val,
            ..
        }) = fold(&ssa, &op)
        else {
            panic!("folds: {:?}", fold(&ssa, &op));
        };
        assert_eq!((condition, true_val, false_val), (negated, b, a));
    }

    /// **The same condition twice keeps the inner arm the outer one
    /// reaches,** and a choice of one value either way is that value:
    /// `c ? a : (c ? b : a)` is `a`.
    #[test]
    fn the_same_condition_twice_is_one_choice() {
        let mut a = None;
        let (ssa, op) = function(|block| {
            let condition = arg(block, 0);
            let (first, second) = (arg(block, 1), arg(block, 2));
            a = Some(first);
            let inner = block.select(SsaDefSpec::tmp(MockType::I32), condition, second, first)?;
            block.select(SsaDefSpec::tmp(MockType::I32), condition, first, inner)
        });
        assert_eq!(
            fold(&ssa, &op).and_then(|op| match op {
                SsaOp::Copy { src, .. } => Some(src),
                _ => None,
            }),
            a
        );
    }

    /// **Two copies of one comparison are one condition.** A lift reads a
    /// flag once per guarded instruction, so `moveq`/`movne` test two
    /// `x == y` of their own: `!(x == y) ? b : ((x == y) ? a : old)` is still
    /// `!(x == y) ? b : a`. A comparison of other values is another
    /// condition, and decides nothing.
    #[test]
    fn two_copies_of_one_comparison_are_one_condition() {
        for same in [true, false] {
            let mut arms = None;
            let (ssa, op) = function(|block| {
                let (x, y, other) = (arg(block, 0), arg(block, 1), arg(block, 2));
                let (old, a) = (arg(block, 3), arg(block, 2));
                let first = block.ceq(SsaDefSpec::tmp(MockType::I1), x, y)?;
                let second = block.ceq(
                    SsaDefSpec::tmp(MockType::I1),
                    x,
                    if same { y } else { other },
                )?;
                let guarded = block.select(SsaDefSpec::tmp(MockType::I32), second, a, old)?;
                let negated = block.bool_not(SsaDefSpec::tmp(MockType::I1), first)?;
                arms = Some((a, old));
                block.select(SsaDefSpec::tmp(MockType::I32), negated, old, guarded)
            });
            let (a, old) = arms.expect("built");
            let folded = fold(&ssa, &op);
            if same {
                assert!(
                    matches!(folded, Some(SsaOp::Select { true_val, false_val, .. })
                        if (true_val, false_val) == (old, a)),
                    "{folded:?}"
                );
            } else {
                assert!(folded.is_none(), "{folded:?}");
            }
        }
    }

    /// **A constant condition is a choice made.** `true ? a : b` is `a`,
    /// `!true ? a : b` is `b`, and the arm it picks is decided through the
    /// selects on that condition: `false ? a : (false ? b : c)` is `c`, where
    /// the arms are no constants for constant propagation to fold.
    #[test]
    fn a_constant_condition_is_a_choice_made() {
        let mut arms = None;
        let (ssa, op) = function(|block| {
            let (a, b) = (arg(block, 0), arg(block, 1));
            arms = Some((a, b));
            let holds = block.const_bool(SsaDefSpec::tmp(MockType::I1), true)?;
            block.select(SsaDefSpec::tmp(MockType::I32), holds, a, b)
        });
        assert_eq!(copied(fold(&ssa, &op)), arms.map(|(a, _)| a));
        let (ssa, op) = function(|block| {
            let (a, b) = (arg(block, 0), arg(block, 1));
            arms = Some((a, b));
            let holds = block.const_bool(SsaDefSpec::tmp(MockType::I1), true)?;
            let fails = block.bool_not(SsaDefSpec::tmp(MockType::I1), holds)?;
            block.select(SsaDefSpec::tmp(MockType::I32), fails, a, b)
        });
        assert_eq!(copied(fold(&ssa, &op)), arms.map(|(_, b)| b));
        let mut last = None;
        let (ssa, op) = function(|block| {
            let (a, b, c) = (arg(block, 0), arg(block, 1), arg(block, 2));
            last = Some(c);
            let fails = block.const_bool(SsaDefSpec::tmp(MockType::I1), false)?;
            let inner = block.select(SsaDefSpec::tmp(MockType::I32), fails, b, c)?;
            block.select(SsaDefSpec::tmp(MockType::I32), fails, a, inner)
        });
        assert_eq!(copied(fold(&ssa, &op)), last);
    }

    /// **An unrelated condition decides nothing.** `c ? a : (d ? b : old)`
    /// stays as it is.
    #[test]
    fn an_unrelated_condition_decides_nothing() {
        let (ssa, op) = function(|block| {
            let (condition, other) = (arg(block, 0), arg(block, 1));
            let (old, a) = (arg(block, 2), arg(block, 3));
            let inner = block.select(SsaDefSpec::tmp(MockType::I32), other, a, old)?;
            block.select(SsaDefSpec::tmp(MockType::I32), condition, a, inner)
        });
        assert_eq!(fold(&ssa, &op), None);
    }

    /// The select `a == b ? x : y` over two arguments of type `ty`, with the
    /// condition negated as `negated` says and `(x, y)` = `(a, b)`, or
    /// `(b, a)` when `swapped`. Returns the function, the select, `a` and
    /// `b`.
    fn over_equality(
        ty: MockType,
        negated: bool,
        swapped: bool,
    ) -> (
        SsaFunction<MockTarget>,
        SsaOp<MockTarget>,
        SsaVarId,
        SsaVarId,
    ) {
        let mut seen = None;
        let (ssa, op) = function(|block| {
            let a = block.load_arg(SsaDefSpec::argument(0, ty), 0)?;
            let b = block.load_arg(SsaDefSpec::argument(1, ty), 1)?;
            seen = Some((a, b));
            let mut condition = block.ceq(SsaDefSpec::tmp(MockType::I1), a, b)?;
            if negated {
                condition = block.bool_not(SsaDefSpec::tmp(MockType::I1), condition)?;
            }
            let (x, y) = if swapped { (b, a) } else { (a, b) };
            block.select(SsaDefSpec::tmp(ty), condition, x, y)
        });
        let (a, b) = seen.expect("built");
        (ssa, op, a, b)
    }

    /// **A choice over whether its own arms are equal is one value.**
    /// `a != b ? a : b` is `a`, `a == b ? a : b` is `b`, and either with the
    /// arms swapped is the arm taken where they differ.
    #[test]
    fn a_choice_between_the_values_it_compares_is_one_value() {
        for (negated, swapped) in [(true, false), (false, false), (true, true), (false, true)] {
            let (ssa, op, a, b) = over_equality(MockType::I32, negated, swapped);
            let (x, y) = if swapped { (b, a) } else { (a, b) };
            let expected = if negated { x } else { y };
            assert_eq!(
                copied(fold(&ssa, &op)),
                Some(expected),
                "negated {negated}, swapped {swapped}"
            );
        }
    }

    /// Where a boolean select's arm is its condition: the true arm, the false
    /// arm, or neither.
    #[derive(Clone, Copy)]
    enum Arm {
        True,
        False,
        Neither,
    }

    /// The boolean select of an argument `x == y` (the condition, negated as
    /// `negated` says), another comparison `x == z`, and the condition at
    /// `arm`, itself negated as `arm_negated` says; its operation with the
    /// select's two arms.
    fn over_condition(
        negated: bool,
        arm: Arm,
        arm_negated: bool,
        ty: MockType,
    ) -> (SsaOp<MockTarget>, Option<SsaOp<MockTarget>>) {
        let (ssa, op) = function(|block| {
            let (x, y, z) = (arg(block, 0), arg(block, 1), arg(block, 2));
            let tested = block.ceq(SsaDefSpec::tmp(ty), x, y)?;
            let other = block.ceq(SsaDefSpec::tmp(ty), x, z)?;
            let negation = |block: &mut crate::ir::SsaBlockBuilder<'_, MockTarget>, on: bool| {
                if on {
                    block.bool_not(SsaDefSpec::tmp(ty), tested)
                } else {
                    Ok(tested)
                }
            };
            let condition = negation(block, negated)?;
            let at = negation(block, arm_negated)?;
            let (true_val, false_val) = match arm {
                Arm::True => (at, other),
                Arm::False => (other, at),
                Arm::Neither => (other, other),
            };
            block.select(SsaDefSpec::tmp(ty), condition, true_val, false_val)
        });
        let folded = fold(&ssa, &op);
        (op, folded)
    }

    /// **A boolean choice one arm of which is its condition is an `&&` or an
    /// `||`.** `c ? c : f` is `c || f`, `c ? t : c` is `t && c`, `c ? !c : f`
    /// is `!c && f` and `c ? t : !c` is `t || !c`, read the same through a
    /// negated condition: ARM's `!(r0 == 61) ? r0 == 0 : r0 == 61` is `r0 == 0
    /// || r0 == 61`. The controls: an arm that is no form of the condition
    /// decides nothing, and integers are no booleans.
    #[test]
    fn a_boolean_choice_of_its_condition_is_a_conjunction_or_a_disjunction() {
        for negated in [false, true] {
            for (arm, arm_negated, disjunction) in [
                (Arm::True, false, true),
                (Arm::False, false, false),
                (Arm::True, true, false),
                (Arm::False, true, true),
            ] {
                let (op, folded) =
                    over_condition(negated, arm, arm_negated ^ negated, MockType::Bool);
                let SsaOp::Select {
                    dest,
                    true_val,
                    false_val,
                    ..
                } = op
                else {
                    panic!("a select: {op:?}");
                };
                let expected = if disjunction {
                    SsaOp::Or {
                        dest,
                        left: true_val,
                        right: false_val,
                        flags: None,
                    }
                } else {
                    SsaOp::And {
                        dest,
                        left: true_val,
                        right: false_val,
                        flags: None,
                    }
                };
                assert_eq!(folded, Some(expected), "negated {negated}");
            }
        }
        let (_, unrelated) = over_condition(false, Arm::Neither, false, MockType::Bool);
        assert!(
            !matches!(unrelated, Some(SsaOp::And { .. } | SsaOp::Or { .. })),
            "{unrelated:?}"
        );
        let (_, integers) = over_condition(false, Arm::True, false, MockType::I32);
        assert_eq!(integers, None);
    }

    /// **Floats equal are not one value.** The control of
    /// [`a_choice_between_the_values_it_compares_is_one_value`]: `+0` and `-0`
    /// compare equal, so `a != b ? a : b` over floats picks `b` where `a` is
    /// `-0` and `b` is `+0`, and stays as it is.
    #[test]
    fn a_choice_between_floats_it_compares_stays() {
        let (ssa, op, _, _) = over_equality(MockType::F64, true, false);
        assert_eq!(fold(&ssa, &op), None);
    }
}
