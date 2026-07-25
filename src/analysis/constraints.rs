//! Constraint types for SSA path analysis.
//!
//! This module provides constraint types derived from branch conditions during
//! path-aware SSA evaluation. When traversing a specific CFG path, branch
//! conditions impose constraints on variable values that can be used to detect
//! dead code, prune infeasible paths, and improve constant propagation.
//!
//! # Constraint Types
//!
//! - [`Constraint`]: A constraint on a single value (e.g., `x == 5`, `x > 10`)
//! - [`PathConstraint`]: Associates a constraint with a specific SSA variable
//!
//! # Constraint Derivation
//!
//! Constraints are derived from comparison operations at branch points:
//!
//! | Branch Condition | True Path Constraint | False Path Constraint |
//! |------------------|---------------------|-----------------------|
//! | `ceq(x, c)`      | `x == c`            | `x != c`              |
//! | `cgt(x, c)`      | `x > c`             | `x <= c`              |
//! | `clt(x, c)`      | `x < c`             | `x >= c`              |
//!
//! # Conflict Detection
//!
//! Constraints support `conflicts_with()` to detect when two constraints
//! cannot both be true (e.g., `x == 5` conflicts with `x > 10`). This enables
//! detection of infeasible paths for opaque predicate identification.
//!
//! # Usage with SsaEvaluator
//!
//! The [`SsaEvaluator`](super::SsaEvaluator) tracks path constraints during evaluation.
//! When taking a branch, the evaluator records constraints that must hold on that path.
//!
//! For example, after taking the true branch of `if (x == 5)`:
//! - We know `x == 5` on this path
//! - This is recorded as `PathConstraint { variable: x, constraint: Constraint::Equal(ConstValue::I32(5)) }`
//!
//! # Use Cases
//!
//! - Constraint solving with Z3
//! - Value range analysis
//! - Dead code detection
//! - Path-sensitive constant propagation

use crate::{
    PointerSize,
    ir::{value::ConstValue, variable::SsaVarId},
    target::Target,
};

/// A constraint on a variable's value derived from branch conditions.
///
/// When following a specific branch path, we can derive facts about variable values.
/// For example, after taking the true branch of `if (x == 5)`, we know `x == 5`.
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint<T: Target> {
    /// Variable equals a concrete value: `x == value`
    Equal(ConstValue<T>),
    /// Variable does not equal a concrete value: `x != value`
    NotEqual(ConstValue<T>),
    /// Variable is greater than a value (signed): `x > value`
    GreaterThan(ConstValue<T>),
    /// Variable is less than a value (signed): `x < value`
    LessThan(ConstValue<T>),
    /// Variable is greater than or equal (signed): `x >= value`
    GreaterOrEqual(ConstValue<T>),
    /// Variable is less than or equal (signed): `x <= value`
    LessOrEqual(ConstValue<T>),
    /// Variable is greater than (unsigned): `(uint)x > value`
    GreaterThanUnsigned(ConstValue<T>),
    /// Variable is less than (unsigned): `(uint)x < value`
    LessThanUnsigned(ConstValue<T>),
    /// Variable is greater than or equal (unsigned): `(uint)x >= value`
    ///
    /// The negation of [`Self::LessThanUnsigned`]. Recording the *signed*
    /// [`Self::GreaterOrEqual`] on the false edge of an unsigned comparison is
    /// wrong: `!(x <u k)` says nothing about `x` under signed ordering, since a
    /// value above `i64::MAX` unsigned is negative signed.
    GreaterOrEqualUnsigned(ConstValue<T>),
    /// Variable is less than or equal (unsigned): `(uint)x <= value`
    ///
    /// The negation of [`Self::GreaterThanUnsigned`]; see
    /// [`Self::GreaterOrEqualUnsigned`].
    LessOrEqualUnsigned(ConstValue<T>),
}

impl<T: Target> Constraint<T> {
    /// Checks if a concrete value satisfies this constraint.
    ///
    /// Uses typed comparison methods from `ConstValue`.
    #[must_use]
    pub fn is_satisfied_by(&self, value: &ConstValue<T>) -> bool {
        match self {
            Self::Equal(v) => value.ceq(v).is_some_and(|r| !r.is_zero()),
            Self::NotEqual(v) => value.ceq(v).is_some_and(|r| r.is_zero()),
            Self::GreaterThan(v) => value.cgt(v).is_some_and(|r| !r.is_zero()),
            Self::LessThan(v) => value.clt(v).is_some_and(|r| !r.is_zero()),
            Self::GreaterOrEqual(v) => value.clt(v).is_some_and(|r| r.is_zero()),
            Self::LessOrEqual(v) => value.cgt(v).is_some_and(|r| r.is_zero()),
            Self::GreaterThanUnsigned(v) => value.cgt_un(v).is_some_and(|r| !r.is_zero()),
            Self::LessThanUnsigned(v) => value.clt_un(v).is_some_and(|r| !r.is_zero()),
            // `x >=u k` is exactly `!(x <u k)`, so satisfaction is the negation
            // of the strict form — and still requires the comparison to have
            // been performed at all.
            Self::GreaterOrEqualUnsigned(v) => value.clt_un(v).is_some_and(|r| r.is_zero()),
            Self::LessOrEqualUnsigned(v) => value.cgt_un(v).is_some_and(|r| r.is_zero()),
        }
    }

    /// Returns the concrete value if this is an equality constraint.
    #[must_use]
    pub fn as_equal(&self) -> Option<&ConstValue<T>> {
        match self {
            Self::Equal(v) => Some(v),
            _ => None,
        }
    }

    /// Checks if this constraint conflicts with another (both can't be true).
    ///
    /// Returns `true` only when the conflict is *proved*. Constant pairs that
    /// [`ConstValue`] cannot compare — mixed signedness, mixed non-promotable
    /// widths, booleans against integers — yield `false` (no conflict known),
    /// because "cannot compare" is not evidence of infeasibility. Reporting a
    /// conflict there would mark a live path unreachable.
    ///
    /// # Arguments
    ///
    /// * `other` - The other constraint to check against.
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    #[must_use]
    pub fn conflicts_with(&self, other: &Constraint<T>, ptr_size: PointerSize) -> bool {
        match (self, other) {
            // x == a conflicts with x == b when a and b are provably different.
            // Compared numerically, not structurally: `I32(5)` and `I64(5)` are
            // the same value, and `!=` on the enum would call them conflicting.
            // A pair `ceq` cannot compare proves nothing, so it is not a conflict.
            (Self::Equal(a), Self::Equal(b)) => a.ceq(b).is_some_and(|r| r.is_zero()),
            // x == a conflicts with x != a
            (Self::Equal(a), Self::NotEqual(b)) | (Self::NotEqual(b), Self::Equal(a)) => a == b,
            // x == a conflicts with x > b when a <= b (i.e., a is not greater than b)
            (Self::Equal(a), Self::GreaterThan(b)) | (Self::GreaterThan(b), Self::Equal(a)) => {
                // a <= b means a is not strictly greater than b
                a.cgt(b).is_some_and(|r| r.is_zero())
            }
            // x == a conflicts with x < b when a >= b (i.e., a is not less than b)
            (Self::Equal(a), Self::LessThan(b)) | (Self::LessThan(b), Self::Equal(a)) => {
                // a >= b means a is not strictly less than b
                a.clt(b).is_some_and(|r| r.is_zero())
            }
            // x > a conflicts with x < b if ranges don't overlap
            (Self::GreaterThan(a), Self::LessThan(b))
            | (Self::LessThan(b), Self::GreaterThan(a)) => {
                // x > a AND x < b requires b > a + 1 (there must be room for at least one integer)
                // Conflicts when b <= a + 1
                let one: ConstValue<T> = ConstValue::I32(1);
                a.add(&one, ptr_size)
                    .and_then(|a_plus_1| b.cgt(&a_plus_1))
                    .is_some_and(|r| r.is_zero())
            }
            _ => false,
        }
    }
}

/// A constraint on a path derived from branch conditions.
///
/// Associates a [`Constraint`] with a specific SSA variable. These constraints
/// are accumulated during path evaluation and can be used for constraint
/// solving with Z3.
#[derive(Debug, Clone, PartialEq)]
pub struct PathConstraint<T: Target> {
    /// The variable this constraint applies to.
    pub variable: SsaVarId,
    /// The constraint on the variable's value.
    pub constraint: Constraint<T>,
}

impl<T: Target> PathConstraint<T> {
    /// Creates a new equality constraint.
    #[must_use]
    pub fn equal(variable: SsaVarId, value: ConstValue<T>) -> Self {
        Self {
            variable,
            constraint: Constraint::Equal(value),
        }
    }

    /// Creates a new inequality constraint.
    #[must_use]
    pub fn not_equal(variable: SsaVarId, value: ConstValue<T>) -> Self {
        Self {
            variable,
            constraint: Constraint::NotEqual(value),
        }
    }

    /// Creates a new less-than constraint.
    #[must_use]
    pub fn less_than(variable: SsaVarId, value: ConstValue<T>) -> Self {
        Self {
            variable,
            constraint: Constraint::LessThan(value),
        }
    }

    /// Creates a new greater-than constraint.
    #[must_use]
    pub fn greater_than(variable: SsaVarId, value: ConstValue<T>) -> Self {
        Self {
            variable,
            constraint: Constraint::GreaterThan(value),
        }
    }

    /// Checks if a concrete value satisfies this constraint.
    #[must_use]
    pub fn is_satisfied_by(&self, value: &ConstValue<T>) -> bool {
        self.constraint.is_satisfied_by(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockTarget;

    const PTR: PointerSize = PointerSize::Bit64;

    fn eq(v: ConstValue<MockTarget>) -> Constraint<MockTarget> {
        Constraint::Equal(v)
    }

    /// `ConstValue` comparison returns `None` for pairs it has no arm for —
    /// mixed signedness, mixed non-promotable widths, booleans against integers.
    /// "Cannot compare" is not evidence that both constraints are unsatisfiable,
    /// so it must not be reported as a conflict: a conflict marks the path
    /// infeasible and everything on it gets deleted.
    #[test]
    fn uncomparable_constants_are_not_a_conflict() {
        // I32 vs U32: no comparison arm exists.
        assert!(
            !eq(ConstValue::I32(5))
                .conflicts_with(&Constraint::GreaterThan(ConstValue::U32(3)), PTR),
            "mixed signedness cannot be compared, so no conflict is proved"
        );
        assert!(
            !eq(ConstValue::I32(5)).conflicts_with(&Constraint::LessThan(ConstValue::U32(9)), PTR),
        );
        // I8 vs I32: not promotable by `clt`/`cgt`.
        assert!(
            !eq(ConstValue::I8(5))
                .conflicts_with(&Constraint::GreaterThan(ConstValue::I32(3)), PTR),
        );
        // A boolean against an integer.
        assert!(
            !eq(ConstValue::True).conflicts_with(&Constraint::LessThan(ConstValue::I32(1)), PTR)
        );
    }

    /// Equality is a question about values, not about which enum arm carries
    /// them: `I32(5)` and `I64(5)` are the same number.
    #[test]
    fn equal_constraints_compare_numerically_not_structurally() {
        assert!(
            !eq(ConstValue::I32(5)).conflicts_with(&eq(ConstValue::I64(5)), PTR),
            "I32(5) and I64(5) are the same value and must not conflict"
        );
        assert!(
            eq(ConstValue::I32(5)).conflicts_with(&eq(ConstValue::I32(6)), PTR),
            "two different values of the same type do conflict"
        );
        assert!(
            !eq(ConstValue::I32(5)).conflicts_with(&eq(ConstValue::U32(5)), PTR),
            "uncomparable pair proves nothing either way"
        );
    }

    /// `!(x <u k)` is `x >=u k`, which says nothing under *signed* ordering: a
    /// value above `i64::MAX` unsigned is negative signed. Recording the signed
    /// form on the false edge of an unsigned comparison claims a fact the branch
    /// does not prove.
    #[test]
    fn unsigned_negations_use_unsigned_ordering() {
        // 0xFFFF_FFFF is huge unsigned, and -1 signed.
        let big = ConstValue::<MockTarget>::U32(0xFFFF_FFFF);
        let ten = ConstValue::<MockTarget>::U32(10);

        // Taking the false edge of `x <u 10` proves `x >=u 10`, which 0xFFFF_FFFF
        // satisfies...
        assert!(Constraint::GreaterOrEqualUnsigned(ten.clone()).is_satisfied_by(&big));
        // ...while it does *not* satisfy `x >= 10` in unsigned terms being
        // conflated with signed: the signed reading of the same bits is -1.
        assert!(!Constraint::<MockTarget>::LessThanUnsigned(ten.clone()).is_satisfied_by(&big));

        // And the mirror: the false edge of `x >u 10` proves `x <=u 10`.
        assert!(Constraint::LessOrEqualUnsigned(ten.clone()).is_satisfied_by(&ConstValue::U32(10)));
        assert!(!Constraint::LessOrEqualUnsigned(ten).is_satisfied_by(&big));
    }

    /// The conflicts the analysis exists to find must still be found.
    #[test]
    fn provable_conflicts_are_still_reported() {
        assert!(
            eq(ConstValue::I32(5)).conflicts_with(&Constraint::NotEqual(ConstValue::I32(5)), PTR)
        );
        // x == 5 and x > 10 cannot both hold.
        assert!(
            eq(ConstValue::I32(5))
                .conflicts_with(&Constraint::GreaterThan(ConstValue::I32(10)), PTR)
        );
        // x == 20 and x < 10 cannot both hold.
        assert!(
            eq(ConstValue::I32(20)).conflicts_with(&Constraint::LessThan(ConstValue::I32(10)), PTR)
        );
        // x > 5 and x < 6 leaves no integer.
        assert!(
            Constraint::<MockTarget>::GreaterThan(ConstValue::I32(5))
                .conflicts_with(&Constraint::LessThan(ConstValue::I32(6)), PTR)
        );
        // x > 5 and x < 100 is satisfiable.
        assert!(
            !Constraint::<MockTarget>::GreaterThan(ConstValue::I32(5))
                .conflicts_with(&Constraint::LessThan(ConstValue::I32(100)), PTR)
        );
    }
}
