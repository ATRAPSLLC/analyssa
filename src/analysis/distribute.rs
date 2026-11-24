//! A value split by bit operations and joined again.
//!
//! Lifted code splits a value by masks and recombines the parts: a sign
//! injection keeps one register's sign and another's magnitude (x86
//! `andps`/`andnps`/`orps`), a field write keeps a container's other bits,
//! and once the values the parts came from are proven one (by value
//! numbering, by copy propagation), what is left is a join of two parts of
//! the same value. Two rules take those apart:
//!
//! - **A shared operand comes out** ([`factored`]): `&` distributes over `|`
//!   and `^`, and `|` over `&`, so `(s & p) | (s & q)` is `s & (p | q)`,
//!   `(s & p) ^ (s & q)` is `s & (p ^ q)` and `(s | p) & (s | q)` is `s | (p &
//!   q)`, whichever side of each operation `s` is on. Over constant masks the
//!   inner operation is a constant, and the whole a single mask of `s`.
//! - **A value joined with its complement is all ones or none**
//!   ([`complement`]): `p | ~p` and `p ^ ~p` are every bit set, `p & ~p` none,
//!   so `(x & m) | (x & ~m)` factors to `x & (m | ~m)` and that to `x & -1`,
//!   which is `x`.
//! - **A value absorbs a join it is part of** ([`absorbed`]): `x & (x | y)` is
//!   `x` and `x | (x & y)` is `x`; against the complement of such a join, `x &
//!   ~(x | y)` is none and `x | ~(x & y)` every bit set; and against a join
//!   with its own complement, `x & (~x | y)` is `x & y` and `x | (~x & y)` is
//!   `x | y`. A Xeon Phi gather whose completed subset is the whole mask
//!   leaves its mask register as `k & ~(k | p)`, which is none.
//!
//! Every identity holds bit for bit at any width, so no range or sign is
//! consulted. A factoring is offered only where both inner operations are
//! read by nothing else: it then replaces three operations by two, where
//! otherwise it would add one.

use std::collections::BTreeMap;

use crate::{
    analysis::operands::peel,
    ir::{SsaFunction, integer::BitWidth, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    target::Target,
};

/// A bit operation of two operands, as [`Factoring`] names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseKind {
    /// `&`.
    And,
    /// `|`.
    Or,
    /// `^`.
    Xor,
}

impl BitwiseKind {
    /// The kind and the operands of `op`, when it is a bit operation of two.
    fn of<T: Target>(op: &SsaOp<T>) -> Option<(Self, SsaVarId, SsaVarId)> {
        match op {
            SsaOp::And { left, right, .. } => Some((Self::And, *left, *right)),
            SsaOp::Or { left, right, .. } => Some((Self::Or, *left, *right)),
            SsaOp::Xor { left, right, .. } => Some((Self::Xor, *left, *right)),
            _ => None,
        }
    }

    /// The operation of this kind defining `dest` from `left` and `right`.
    #[must_use]
    pub fn op<T: Target>(self, dest: SsaVarId, left: SsaVarId, right: SsaVarId) -> SsaOp<T> {
        match self {
            Self::And => SsaOp::And {
                dest,
                left,
                right,
                flags: None,
            },
            Self::Or => SsaOp::Or {
                dest,
                left,
                right,
                flags: None,
            },
            Self::Xor => SsaOp::Xor {
                dest,
                left,
                right,
                flags: None,
            },
        }
    }
}

/// An operation `(s a p) o (s a q)` rewritten as `s a (p o q)`: the shared
/// operand `s` taken out of two inner operations of kind `a` the outer one of
/// kind `o` joins ([`factored`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factoring {
    /// The operand both inner operations share.
    pub shared: SsaVarId,
    /// The kind of the inner operations, which the rewrite applies last.
    pub inner: BitwiseKind,
    /// The kind of the outer operation, which the rewrite applies to the two
    /// other operands first.
    pub outer: BitwiseKind,
    /// The first inner operation's other operand.
    pub left: SsaVarId,
    /// The second inner operation's other operand.
    pub right: SsaVarId,
}

/// The factoring of `op`, a join of two operations that share an operand,
/// where the shared operand distributes over the join: `&` under `|` or `^`,
/// and `|` under `&`.
///
/// # Arguments
///
/// * `ssa` - The function, for the operands' definitions.
/// * `op` - The candidate join.
/// * `uses` - Use counts for the whole function, as
///   [`SsaFunction::count_uses`] reports them: both inner operations must be
///   read by `op` alone.
///
/// # Returns
///
/// The factoring, or `None` when `op` joins no two such operations.
#[must_use]
pub fn factored<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    uses: &BTreeMap<SsaVarId, usize>,
) -> Option<Factoring> {
    let (outer, first, second) = BitwiseKind::of(op)?;
    let inner = match outer {
        BitwiseKind::Or | BitwiseKind::Xor => BitwiseKind::And,
        BitwiseKind::And => BitwiseKind::Or,
    };
    let once = |var: SsaVarId| uses.get(&var).copied().unwrap_or(0) == 1;
    if first == second || !once(first) || !once(second) {
        return None;
    }
    let operands = |var: SsaVarId| {
        let (kind, left, right) = BitwiseKind::of(ssa.get_definition(var)?)?;
        (kind == inner).then_some((peel(ssa, left), peel(ssa, right)))
    };
    let ((a1, a2), (b1, b2)) = (operands(first)?, operands(second)?);
    [(a1, a2), (a2, a1)].into_iter().find_map(|(shared, left)| {
        let right = if b1 == shared {
            b2
        } else if b2 == shared {
            b1
        } else {
            return None;
        };
        Some(Factoring {
            shared,
            inner,
            outer,
            left,
            right,
        })
    })
}

/// The constant `op` is when it joins a value with its complement: every bit
/// set for `p | ~p` and `p ^ ~p`, none for `p & ~p`.
///
/// A complement is a `not` of the other operand, an `xor` of it with every
/// bit set, or, of two constants, the constant with every bit the other has
/// clear.
///
/// # Arguments
///
/// * `ssa` - The function, for the operands' definitions and the result's
///   type.
/// * `op` - The candidate join.
/// * `constants` - Known constant values.
///
/// # Returns
///
/// The constant, of the result's type, or `None` when `op` joins no value
/// with its complement.
#[must_use]
pub fn complement<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<ConstValue<T>> {
    let (kind, left, right) = BitwiseKind::of(op)?;
    let ty = ssa.variable(op.dest()?)?.var_type();
    let bits = T::bit_width(ty)?;
    let complements = |value: SsaVarId, other: SsaVarId| {
        complemented(ssa, constants, bits, other) == Some(peel(ssa, value))
    };
    let constant_complements = || {
        let pattern = |var: SsaVarId| {
            constants
                .get(&peel(ssa, var))?
                .integer_pattern(crate::pointer::PointerSize::Bit64)
                .filter(|(_, width)| *width == bits)
                .map(|(pattern, _)| pattern)
        };
        let mask = BitWidth::new(bits).mask();
        matches!((pattern(left), pattern(right)), (Some(a), Some(b)) if a ^ b == mask)
    };
    if !(complements(left, right) || complements(right, left) || constant_complements()) {
        return None;
    }
    let value = match kind {
        BitwiseKind::Or | BitwiseKind::Xor => -1,
        BitwiseKind::And => 0,
    };
    ConstValue::integer(bits, T::is_signed(ty), value)
}

/// What `op` is when one operand absorbs the other ([`absorbed`]).
#[derive(Debug, Clone, PartialEq)]
pub enum Absorption<T: Target> {
    /// The operand that absorbs the join: `x & (x | y)` is `x`.
    Operand(SsaVarId),
    /// Every bit set or none, of the result's type: `x | ~(x & y)`, `x & ~(x |
    /// y)`.
    Constant(ConstValue<T>),
    /// The operation of the same kind on the operand and the join's other
    /// operand: `x & (~x | y)` is `x & y`.
    Joined(BitwiseKind, SsaVarId, SsaVarId),
}

/// What `op`, an `&` or a `|`, is when one operand absorbs the other: the
/// other is the dual join (`|` under `&`, `&` under `|`) of the operand with
/// something, the complement of such a join, or such a join of the operand's
/// complement.
///
/// A complement is what [`complement`] reads as one: a `not`, or an `xor` with
/// every bit set.
///
/// # Arguments
///
/// * `ssa` - The function, for the operands' definitions and the result's
///   type.
/// * `op` - The candidate.
/// * `constants` - Known constant values.
///
/// # Returns
///
/// The absorption, or `None` when neither operand absorbs the other.
#[must_use]
pub fn absorbed<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
) -> Option<Absorption<T>> {
    let (kind, left, right) = BitwiseKind::of(op)?;
    let dual = match kind {
        BitwiseKind::And => BitwiseKind::Or,
        BitwiseKind::Or => BitwiseKind::And,
        BitwiseKind::Xor => return None,
    };
    let ty = ssa.variable(op.dest()?)?.var_type();
    let bits = T::bit_width(ty)?;
    // The operands of the dual join defining `var`, peeled.
    let joined = |var: SsaVarId| {
        let (inner, first, second) = BitwiseKind::of(ssa.get_definition(peel(ssa, var))?)?;
        (inner == dual).then(|| (peel(ssa, first), peel(ssa, second)))
    };
    [(left, right), (right, left)]
        .into_iter()
        .find_map(|(operand, other)| {
            let operand = peel(ssa, operand);
            if let Some((first, second)) = joined(other) {
                if first == operand || second == operand {
                    return Some(Absorption::Operand(operand));
                }
                let complement = |var: SsaVarId| complemented(ssa, constants, bits, var);
                let kept = if complement(first) == Some(operand) {
                    second
                } else if complement(second) == Some(operand) {
                    first
                } else {
                    return None;
                };
                return Some(Absorption::Joined(kind, operand, kept));
            }
            let (first, second) = joined(complemented(ssa, constants, bits, other)?)?;
            (first == operand || second == operand)
                .then(|| {
                    let value = match kind {
                        BitwiseKind::And => 0,
                        BitwiseKind::Or | BitwiseKind::Xor => -1,
                    };
                    ConstValue::integer(bits, T::is_signed(ty), value)
                })
                .flatten()
                .map(Absorption::Constant)
        })
}

/// The value `var` is the complement of, peeled: the operand of a `not`, or
/// the other operand of an `xor` with every bit set at `bits`.
fn complemented<T: Target>(
    ssa: &SsaFunction<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    bits: u32,
    var: SsaVarId,
) -> Option<SsaVarId> {
    let ones = |var: SsaVarId| {
        constants
            .get(&peel(ssa, var))
            .and_then(|value| value.integer_pattern(crate::pointer::PointerSize::Bit64))
            .is_some_and(|(pattern, width)| width == bits && pattern == BitWidth::new(bits).mask())
    };
    match ssa.get_definition(peel(ssa, var))? {
        SsaOp::Not { operand, .. } => Some(peel(ssa, *operand)),
        SsaOp::Xor { left, right, .. } if ones(*right) => Some(peel(ssa, *left)),
        SsaOp::Xor { left, right, .. } if ones(*left) => Some(peel(ssa, *right)),
        _ => None,
    }
}
