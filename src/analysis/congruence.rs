//! What a value can be **modulo a power of two**.
//!
//! The interval domain in [`range`](crate::analysis::range) answers how large a
//! value is. This answers what it is *congruent to*, and the two are
//! independent: an interval knows nothing about parity, and a congruence knows
//! nothing about magnitude. Together they decide comparisons neither settles
//! alone.
//!
//! # Why a power of two, and why this one
//!
//! Machine arithmetic **is** arithmetic modulo `2^width`, so reducing further to
//! `2^k` is a ring homomorphism: for `+`, `-`, `*` and `<<` the residue of the
//! result depends only on the residues of the operands, *exactly*, with no
//! approximation and no dependence on the operand width. The bitwise operations
//! are exact for the same reason from the other direction - a residue mod `2^k`
//! **is** the low `k` bits, and `&`, `|`, `^` act on bits independently.
//!
//! That is the whole of the soundness argument, and it is why the domain is not
//! "modulo some number": at any other modulus a truncating machine multiply
//! would not commute with the reduction and the abstraction would be wrong.
//!
//! [`MODULUS`] is 64, so a set of residues is one `u64` bitmask and the domain
//! costs one machine word per value. It subsumes reasoning at 2, 4, 8, 16 and 32
//! at once: a set closed under the coarser modulus is representable here, so
//! nothing is lost by not tracking several.
//!
//! # What it is for
//!
//! Opaque predicates built from quadratic identities, which no interval and no
//! pattern decides. `7*b*b - 1 == a*a` has no solution in any `int32_t` because
//! `a*a` is congruent to one of `{0, 1, 4}` modulo 8 while `7*b*b - 1` is
//! congruent to one of `{3, 6, 7}`, and the two sets are disjoint. The proof is
//! finite, it needs no solver, and it is exactly what this domain computes.

use std::iter;

use crate::{ir::value::ConstValue, target::Target};

/// The modulus every residue is taken at.
///
/// A power of two, so the reduction is a ring homomorphism of machine
/// arithmetic (see the module docs), and exactly 64 so that a set of residues
/// is one `u64`.
pub const MODULUS: u64 = 64;

/// The set of residues modulo [`MODULUS`] a value may take.
///
/// A bitmask: bit `r` is set when the value may be congruent to `r`. The empty
/// set is [`Self::BOTTOM`] - an unreachable value - and the full set is
/// [`Self::TOP`], which states nothing.
///
/// # Ordering
///
/// The lattice order is set inclusion: a smaller set is more precise. Join
/// (union) is what a phi does, meet (intersection) is what two facts about one
/// value do. That is the opposite convention from
/// [`ValueRange`](crate::analysis::range::ValueRange), whose `meet` widens; the
/// names here follow the set operations so neither can be read as the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Congruence {
    /// Bit `r` set when the value may be `r` modulo [`MODULUS`].
    mask: u64,
}

impl Default for Congruence {
    fn default() -> Self {
        Self::TOP
    }
}

impl Congruence {
    /// How many low bits of a value [`MODULUS`] covers.
    ///
    /// The width a conversion must keep for a residue to survive it: truncation
    /// below this many bits discards part of the residue, and everything at or
    /// above it leaves the tracked bits alone.
    pub const MODULUS_BITS: u32 = MODULUS.trailing_zeros();

    /// Every residue: the value is unconstrained.
    pub const TOP: Self = Self { mask: u64::MAX };

    /// No residue: the value is unreachable.
    pub const BOTTOM: Self = Self { mask: 0 };

    /// The single residue `value` reduces to.
    #[must_use]
    pub const fn exact(value: i64) -> Self {
        // `as u64` is the two's-complement bit pattern, and the low bits of that
        // pattern *are* the residue: a negative number is congruent to its
        // pattern modulo any power of two, so no sign correction is needed or
        // wanted.
        Self::residue(value as u64)
    }

    /// The single residue `bits` reduces to, read as an unsigned pattern.
    #[must_use]
    pub const fn residue(bits: u64) -> Self {
        Self {
            mask: 1u64 << (bits % MODULUS),
        }
    }

    /// The residues of a constant, or [`Self::TOP`] for one that is not an
    /// integer.
    ///
    /// A float, a string or a reference has no residue: its bits are not the
    /// value, and reducing them would state a congruence the program never
    /// computes.
    #[must_use]
    pub fn of_constant<T: Target>(value: &ConstValue<T>) -> Self {
        value.as_i64().map_or(Self::TOP, Self::exact)
    }

    /// Whether this states nothing.
    #[must_use]
    pub const fn is_top(self) -> bool {
        self.mask == u64::MAX
    }

    /// Whether no value satisfies this.
    #[must_use]
    pub const fn is_bottom(self) -> bool {
        self.mask == 0
    }

    /// Whether `residue` is one of the possibilities.
    #[must_use]
    pub const fn contains(self, residue: u64) -> bool {
        self.mask & (1u64 << (residue % MODULUS)) != 0
    }

    /// The union - what a phi's result may be, given its operands.
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        Self {
            mask: self.mask | other.mask,
        }
    }

    /// The intersection - two facts about one value, held at once.
    #[must_use]
    pub const fn meet(self, other: Self) -> Self {
        Self {
            mask: self.mask & other.mask,
        }
    }

    /// The greatest value at most `bound` whose residue is in the set, or
    /// `None` for the empty set or one the `i64` range ends before.
    ///
    /// A guard's endpoint reads it: `i != 512` cuts `[4, 512]` to `[4, 511]`,
    /// and an `i` that is a multiple of four is at most `508`.
    #[must_use]
    pub fn greatest_at_most(self, bound: i64) -> Option<i64> {
        // One modulus below `bound` holds every residue once.
        (0..MODULUS).find_map(|below| {
            bound
                .checked_sub_unsigned(below)
                .filter(|value| !self.meet(Self::exact(*value)).is_bottom())
        })
    }

    /// The least value at least `bound` whose residue is in the set, or
    /// `None` for the empty set or one the `i64` range ends before.
    #[must_use]
    pub fn least_at_least(self, bound: i64) -> Option<i64> {
        (0..MODULUS).find_map(|above| {
            bound
                .checked_add_unsigned(above)
                .filter(|value| !self.meet(Self::exact(*value)).is_bottom())
        })
    }

    /// The step every value lies apart from every other by a multiple of:
    /// the largest power of two, up to [`MODULUS`], that every residue in the
    /// set is congruent modulo. `{0, 8, ..., 56}` steps by 8, one residue by
    /// the whole modulus, and an unconstrained value by 1. `None` for the
    /// empty set.
    ///
    /// A counter stepped by 8 from 0 holds no value between two of its
    /// steps, so accesses at `counter`, `counter + 1`, ..., `counter + 7`
    /// interleave: each lies between two of another's, and one object holds
    /// them all.
    #[must_use]
    pub fn step(self) -> Option<u64> {
        let first = self.residues().next()?;
        let mut step = MODULUS;
        while step > 1
            && !self
                .residues()
                .all(|residue| residue.wrapping_sub(first).is_multiple_of(step))
        {
            step /= 2;
        }
        Some(step)
    }

    /// Whether the two sets share no residue.
    ///
    /// **The disproof.** Two values whose residues are disjoint cannot be equal,
    /// whatever their widths and whatever their intervals, because equal values
    /// are congruent modulo every modulus. Neither set may be empty: `Bottom` is
    /// an unreachable value, and "unreachable is never equal to anything" is a
    /// statement about reachability that this domain must not make on its own.
    #[must_use]
    pub const fn disjoint(self, other: Self) -> bool {
        !self.is_bottom() && !other.is_bottom() && self.mask & other.mask == 0
    }

    /// The residues of `self + other`.
    ///
    /// Adding a residue `l` modulo [`MODULUS`] moves every residue of the other
    /// set `l` places round, which on the mask is a rotation: the sum is one
    /// rotation of either set per residue of the other, the smaller set's
    /// residues taken, and not a pass over every pair.
    #[must_use]
    pub fn sum(self, other: Self) -> Self {
        let (few, many) = if self.mask.count_ones() <= other.mask.count_ones() {
            (self, other)
        } else {
            (other, self)
        };
        let mut out = Self::BOTTOM;
        for residue in few.residues() {
            out.mask |= many
                .mask
                .rotate_left(u32::try_from(residue).unwrap_or_default());
            if out.is_top() {
                break;
            }
        }
        out
    }

    /// The residues of `self - other`: the sum with `-other`.
    #[must_use]
    pub fn difference(self, other: Self) -> Self {
        self.sum(other.negate())
    }

    /// The residues of `self * other`, for two **independent** values.
    ///
    /// `x * x` is not this: see [`Self::square`]. Passing one value's residues
    /// as both arguments is sound and almost always useless, because the
    /// pairwise image of a set with itself is far larger than the image of its
    /// squares - over the whole set it is everything, which is what makes the
    /// distinction the difference between deciding a quadratic predicate and
    /// not.
    #[must_use]
    pub fn product(self, other: Self) -> Self {
        self.combine(other, u64::wrapping_mul)
    }

    /// The residues of `self * self`.
    ///
    /// **The relational half of the domain.** A congruence set says what one
    /// value may be and nothing about how two values relate, so multiplying a
    /// set by itself pairwise answers for `x * y` where `y` happens to have the
    /// same possibilities - which over an unconstrained value is every residue.
    /// Squaring is the image of the *diagonal*, and it is small: twelve residues
    /// modulo 64 out of sixty-four. Every quadratic opaque predicate turns on
    /// that gap, so the transfer function that feeds this domain has to notice
    /// when a multiply's two operands are the same SSA value.
    #[must_use]
    pub fn square(self) -> Self {
        self.map(|value| value.wrapping_mul(value))
    }

    /// The residues of `self & other`, and likewise for `|` and `^`.
    ///
    /// Exact, because a residue modulo `2^k` **is** the low `k` bits and a
    /// bitwise operation acts on each bit independently.
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        self.combine(other, |a, b| a & b)
    }

    /// The residues of `self | other`.
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        self.combine(other, |a, b| a | b)
    }

    /// The residues of `self ^ other`.
    #[must_use]
    pub fn xor(self, other: Self) -> Self {
        self.combine(other, |a, b| a ^ b)
    }

    /// The residues of `self << shift`, for a shift known exactly.
    ///
    /// A shift of [`MODULUS`]'s width or more leaves only zero: every bit the
    /// residue tracks has been shifted past it. A shift that is not a known
    /// constant is the caller's to reject - this takes the count, not a set of
    /// them, because a set of shifts is rare and the loop over it belongs to
    /// whoever has one.
    #[must_use]
    pub fn shift_left(self, shift: u32) -> Self {
        if u64::from(shift) >= MODULUS.trailing_zeros().into() {
            return Self::exact(0);
        }
        self.map(|value| value.wrapping_shl(shift))
    }

    /// The residues of `-self`.
    ///
    /// `-r` modulo [`MODULUS`] is `64 - r`, the residue `63 - r` one place
    /// further round, so the mask reversed, which takes bit `r` to bit
    /// `63 - r`, and rotated by one is the negation of every residue at once.
    #[must_use]
    pub const fn negate(self) -> Self {
        Self {
            mask: self.mask.reverse_bits().rotate_left(1),
        }
    }

    /// The residues of `!self`.
    #[must_use]
    pub fn complement(self) -> Self {
        self.map(|value| !value)
    }

    /// This set, for a value computed at `width` bits.
    ///
    /// The ring-homomorphism argument in the module docs needs the machine
    /// modulus `2^width` to be a multiple of [`MODULUS`]. Below
    /// [`Self::MODULUS_BITS`] bits it is not: a 1-bit `!true` is `false`, while
    /// the complement of residue 1 modulo 64 is 62. A narrower operation's
    /// residues therefore state nothing and this answers [`Self::TOP`]; an
    /// unknown width is the caller's to decide, and is passed through.
    ///
    /// # Arguments
    ///
    /// * `width` - The operation's width in bits, when known.
    #[must_use]
    pub fn at_width(self, width: Option<u32>) -> Self {
        match width {
            Some(width) if width < Self::MODULUS_BITS => Self::TOP,
            _ => self,
        }
    }

    /// The image of this set under `op`.
    ///
    /// Enumerated even from [`Self::TOP`], deliberately: the image of every
    /// residue under a *non-surjective* operation is smaller than every residue,
    /// and squaring is exactly that case. At this modulus the loop is sixty-four
    /// iterations of one instruction, so there is nothing to save by guessing.
    fn map(self, op: impl Fn(u64) -> u64) -> Self {
        let mut out = Self::BOTTOM;
        for residue in self.residues() {
            out.mask |= 1u64 << (op(residue) % MODULUS);
        }
        out
    }

    /// The image of the two sets under `op`, pairwise.
    ///
    /// **Both** operands at [`Self::TOP`] give `TOP`, and for every operation
    /// this domain uses that is *exact* rather than a shortcut: fix the left
    /// operand at the identity (`0` for `+`, `-` and `^`, `1` for `*`, all-ones
    /// for `&`, zero for `|`) and the right one already ranges over everything.
    /// One operand at `TOP` is enumerated like any other set, because there the
    /// image is often smaller - `x * 4` is a multiple of four whatever `x` is,
    /// and short-circuiting would throw that away.
    ///
    /// The walk visits only the residues each set holds and stops as soon as
    /// the image is every residue, since nothing further can change it.
    fn combine(self, other: Self, op: impl Fn(u64, u64) -> u64) -> Self {
        if self.is_top() && other.is_top() {
            return Self::TOP;
        }
        let mut out = Self::BOTTOM;
        for left in self.residues() {
            for right in other.residues() {
                out.mask |= 1u64 << (op(left, right) % MODULUS);
            }
            if out.is_top() {
                break;
            }
        }
        out
    }

    /// The residues in this set, ascending.
    fn residues(self) -> impl Iterator<Item = u64> {
        let mut remaining = self.mask;
        iter::from_fn(move || {
            if remaining == 0 {
                return None;
            }
            let residue = u64::from(remaining.trailing_zeros());
            remaining &= remaining.wrapping_sub(1);
            Some(residue)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sets the pairwise oracle below is checked over: the empty and the
    /// full set, every single residue, and pseudo-random masks of every
    /// density.
    fn sample_sets() -> Vec<Congruence> {
        let mut sets = vec![Congruence::BOTTOM, Congruence::TOP];
        sets.extend((0..MODULUS).map(Congruence::residue));
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for density in 0..32u32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mut mask = state;
            for _ in 0..density % 4 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                mask &= state;
            }
            sets.push(Congruence { mask });
        }
        sets
    }

    /// Every residue `op` takes a pair of the two sets' residues to.
    fn pairwise(left: Congruence, right: Congruence, op: fn(u64, u64) -> u64) -> Congruence {
        let mut out = Congruence::BOTTOM;
        for l in 0..MODULUS {
            for r in 0..MODULUS {
                if left.contains(l) && right.contains(r) {
                    out.mask |= 1 << (op(l, r) % MODULUS);
                }
            }
        }
        out
    }

    /// A sum and a difference by rotation are every pairwise sum and
    /// difference, and a negation by reversal every residue negated.
    #[test]
    fn rotation_is_the_pairwise_image() {
        let sets = sample_sets();
        for &left in &sets {
            let mut negated = Congruence::BOTTOM;
            for residue in 0..MODULUS {
                if left.contains(residue) {
                    negated.mask |= 1 << (residue.wrapping_neg() % MODULUS);
                }
            }
            assert_eq!(left.negate(), negated, "-{left:?}");
            for &right in &sets {
                assert_eq!(
                    left.sum(right),
                    pairwise(left, right, u64::wrapping_add),
                    "{left:?} + {right:?}"
                );
                assert_eq!(
                    left.difference(right),
                    pairwise(left, right, u64::wrapping_sub),
                    "{left:?} - {right:?}"
                );
            }
        }
    }

    /// The residues of `x * x` for every `x`, as the domain computes them.
    fn squares() -> Congruence {
        Congruence::TOP.square()
    }

    /// Every residue of a set, as a sorted vector.
    fn listed(set: Congruence) -> Vec<u64> {
        (0..MODULUS).filter(|r| set.contains(*r)).collect()
    }

    /// A negative value is congruent to its two's-complement pattern.
    ///
    /// The one place a sign could be got wrong: `-1` is `63` modulo 64, not
    /// `-1`, and a sign correction here would put it somewhere else.
    #[test]
    fn a_negative_value_reduces_by_its_bit_pattern() {
        assert_eq!(listed(Congruence::exact(-1)), vec![MODULUS - 1]);
        assert_eq!(listed(Congruence::exact(-64)), vec![0]);
        assert_eq!(listed(Congruence::exact(-65)), vec![MODULUS - 1]);
    }

    /// **Two unconstrained operands give an unconstrained result, and one does
    /// not.**
    ///
    /// The second half is the precision that matters: `x * 4` is a multiple of
    /// four however little is known about `x`, and a short-circuit on "either
    /// side is `TOP`" would lose it - along with the squares, which is the whole
    /// point of the domain.
    #[test]
    fn one_unconstrained_operand_still_constrains() {
        assert!(Congruence::TOP.sum(Congruence::TOP).is_top());
        assert!(Congruence::TOP.product(Congruence::TOP).is_top());

        // Every multiple of four, and nothing else.
        let scaled = Congruence::TOP.product(Congruence::exact(4));
        assert_eq!(listed(scaled), (0..MODULUS).step_by(4).collect::<Vec<_>>());
        // An odd multiplier is a bijection, so nothing is learned.
        assert!(Congruence::TOP.product(Congruence::exact(3)).is_top());
        // And an unconstrained addend learns nothing either.
        assert!(Congruence::TOP.sum(Congruence::exact(3)).is_top());
    }

    /// **A square is not a product of two unknowns.**
    ///
    /// The relational fact the pass has to supply: over an unconstrained value
    /// the pairwise product is every residue, while the squares are twelve of
    /// sixty-four. Without the distinction the quadratic disproof below cannot
    /// be made at all.
    #[test]
    fn squaring_is_not_multiplying_by_a_twin() {
        assert!(Congruence::TOP.product(Congruence::TOP).is_top());
        assert!(!Congruence::TOP.square().is_top());
        assert_eq!(listed(Congruence::TOP.square()).len(), 12);
        // And it is the diagonal, not a coincidence of the modulus.
        for residue in 0..MODULUS {
            assert!(
                Congruence::TOP
                    .square()
                    .contains(residue.wrapping_mul(residue)),
                "{residue}^2 is a square"
            );
        }
    }

    /// The arithmetic is the machine's: wrapping, and exact modulo 64.
    #[test]
    fn arithmetic_is_exact_modulo_the_modulus() {
        let three = Congruence::exact(3);
        let five = Congruence::exact(5);
        assert_eq!(listed(three.sum(five)), vec![8]);
        assert_eq!(listed(three.difference(five)), vec![MODULUS - 2]);
        assert_eq!(listed(three.product(five)), vec![15]);
        // 3 * 22 = 66, which is 2 modulo 64.
        assert_eq!(listed(three.product(Congruence::exact(22))), vec![2]);
        assert_eq!(listed(three.shift_left(4)), vec![48]);
        // Shifted past everything the modulus tracks.
        assert_eq!(listed(three.shift_left(6)), vec![0]);
        assert_eq!(listed(three.and(five)), vec![1]);
        assert_eq!(listed(three.or(five)), vec![7]);
        assert_eq!(listed(three.xor(five)), vec![6]);
        assert_eq!(listed(three.negate()), vec![MODULUS - 3]);
    }

    /// **The disproof this domain exists for.**
    ///
    /// `7*b*b - 1 == a*a` has no solution in any fixed-width two's-complement
    /// integer, and this is the finite proof: the squares and the left-hand side
    /// share no residue modulo 64. Built the way the lift builds it -
    /// `(b*b << 3) - b*b - 1` - so the test exercises the operations the pass
    /// will actually apply, in the order it will apply them.
    ///
    /// The abstract left-hand side is **wider than the true one**, and
    /// deliberately so: the two occurrences of `b*b` are one value, and this
    /// domain relates no two values, so `sub` pairs the shifted set against the
    /// unshifted one and admits `8x - y` for every `x` and `y` in it. Eighteen
    /// residues where the relational answer has twelve. The disproof survives
    /// that loss, which is the point worth pinning: it does not depend on
    /// precision this domain cannot deliver.
    #[test]
    fn a_quadratic_opaque_predicate_is_disproved() {
        let squares = squares();
        let b_squared = squares;
        let left = b_squared
            .shift_left(3)
            .difference(b_squared)
            .difference(Congruence::exact(1));
        assert_eq!(
            listed(squares),
            vec![0, 1, 4, 9, 16, 17, 25, 33, 36, 41, 49, 57]
        );
        assert!(
            left.disjoint(squares),
            "a square and 7b^2-1 share no residue, so the two can never be equal:\n  \
             squares {:?}\n  left    {:?}",
            listed(squares),
            listed(left)
        );

        // The relational answer, for the record: what the left-hand side really
        // ranges over when the two `b*b` are recognised as one value. The
        // abstract set above contains it, which is what soundness requires.
        let exact = squares.map(|s| (s.wrapping_shl(3)).wrapping_sub(s).wrapping_sub(1));
        assert_eq!(
            listed(exact),
            vec![6, 14, 22, 27, 30, 38, 46, 47, 54, 59, 62, 63]
        );
        for residue in listed(exact) {
            assert!(left.contains(residue), "the abstraction contains {residue}");
        }
        assert!(listed(left).len() > listed(exact).len());
    }

    /// **Bottom is disjoint from nothing.**
    ///
    /// The control for the rule above. An empty set is an unreachable value, and
    /// answering "these can never be equal" for one would be a claim about
    /// reachability made by a domain that does not track it - which is how an
    /// optimistic analysis deletes live code.
    #[test]
    fn an_empty_set_proves_no_inequality() {
        assert!(!Congruence::BOTTOM.disjoint(Congruence::exact(1)));
        assert!(!Congruence::exact(1).disjoint(Congruence::BOTTOM));
        assert!(!Congruence::BOTTOM.disjoint(Congruence::BOTTOM));
        // Two different constants are the ordinary positive case.
        assert!(Congruence::exact(1).disjoint(Congruence::exact(2)));
        assert!(!Congruence::exact(1).disjoint(Congruence::exact(1)));
        // And a value 64 apart is *not* disproved: they are congruent.
        assert!(!Congruence::exact(1).disjoint(Congruence::exact(65)));
    }

    /// **The empty set is the identity of a join, which is why it must never
    /// stand for "not computed yet".**
    ///
    /// The distinction a fixpoint over this domain has to keep. A pass that
    /// accumulates by joining wants to start
    /// from the empty set, so it is tempting to seed every value with one - but
    /// a value no transfer function will ever define, such as a function's entry
    /// registers, then keeps it forever. The emptiness is not inert: it spreads
    /// through every operation that reads such a value, and a join that meets it
    /// reports only its *other* operand, which is an under-approximation
    /// presented as a fact.
    ///
    /// The reader of this domain has to keep the two apart: "nothing computed
    /// this" is [`Congruence::TOP`], and only a value something *did* compute
    /// may be empty.
    #[test]
    fn the_empty_set_absorbs_nothing_and_is_absorbed_by_everything() {
        let two = Congruence::exact(2);
        // A join with the empty set is the identity - so an operand wrongly left
        // empty simply vanishes from the answer.
        assert_eq!(Congruence::BOTTOM.join(two), two);
        // And an operation over it is empty, so the emptiness spreads.
        assert!(Congruence::BOTTOM.sum(two).is_bottom());
        assert!(Congruence::BOTTOM.product(two).is_bottom());
        assert!(Congruence::BOTTOM.square().is_bottom());
        // `Top` behaves the other way round in both, which is what makes it the
        // right stand-in for a value nothing has computed.
        assert!(Congruence::TOP.join(two).is_top());
        assert!(Congruence::TOP.sum(two).is_top());
    }

    /// Join is union and meet is intersection, with `TOP` and `BOTTOM` their
    /// identities.
    #[test]
    fn the_lattice_operations_are_set_operations() {
        let one = Congruence::exact(1);
        let two = Congruence::exact(2);
        assert_eq!(listed(one.join(two)), vec![1, 2]);
        assert!(one.meet(two).is_bottom());
        assert_eq!(one.join(Congruence::BOTTOM), one);
        assert_eq!(one.meet(Congruence::TOP), one);
        assert!(one.join(Congruence::TOP).is_top());
    }

    /// Every operation agrees with the machine, checked exhaustively over the
    /// residues rather than on examples.
    ///
    /// The abstraction is only sound if the abstract result contains the
    /// concrete one for *every* pair, which is a finite claim at this modulus
    /// and therefore checkable rather than argued.
    #[test]
    fn the_abstraction_contains_every_concrete_result() {
        for left in 0..MODULUS {
            for right in 0..MODULUS {
                let (a, b) = (Congruence::residue(left), Congruence::residue(right));
                assert!(a.sum(b).contains(left.wrapping_add(right)));
                assert!(a.difference(b).contains(left.wrapping_sub(right)));
                assert!(a.product(b).contains(left.wrapping_mul(right)));
                assert!(a.and(b).contains(left & right));
                assert!(a.or(b).contains(left | right));
                assert!(a.xor(b).contains(left ^ right));
            }
            let a = Congruence::residue(left);
            assert!(a.negate().contains(left.wrapping_neg()));
            assert!(a.complement().contains(!left));
            assert!(a.square().contains(left.wrapping_mul(left)));
            for shift in 0..6 {
                assert!(a.shift_left(shift).contains(left.wrapping_shl(shift)));
            }
        }
    }

    /// A residue modulo 64 says nothing about a value computed in fewer than
    /// six bits: a 1-bit `!true` is `false`, while the complement of residue 1
    /// is 62.
    #[test]
    fn a_narrow_operation_states_no_residue() {
        let one = Congruence::exact(1);
        assert!(one.complement().at_width(Some(1)).is_top());
        assert!(one.at_width(Some(Congruence::MODULUS_BITS - 1)).is_top());
        assert_eq!(one.at_width(Some(Congruence::MODULUS_BITS)), one);
        assert_eq!(one.at_width(Some(32)), one);
        assert_eq!(one.at_width(None), one);
    }

    /// **The step is the largest power of two the residues share.** A counter
    /// stepped by 8 steps by 8, one residue by the whole modulus, anything
    /// by 1, and the empty set by none.
    #[test]
    fn the_step_is_the_largest_power_of_two_the_residues_share() {
        assert_eq!(
            (0..8u64)
                .map(|step| Congruence::residue(step * 8))
                .fold(Congruence::BOTTOM, Congruence::join)
                .step(),
            Some(8)
        );
        assert_eq!(Congruence::exact(5).step(), Some(MODULUS));
        assert_eq!(Congruence::TOP.step(), Some(1));
        assert_eq!(Congruence::BOTTOM.step(), None);
        // The oracle: every residue apart from the first by a multiple of the
        // step, and by no multiple of any larger power of two.
        let shared = |set: Congruence, modulus: u64| {
            let residues: Vec<u64> = (0..MODULUS).filter(|r| set.contains(*r)).collect();
            residues.first().is_some_and(|first| {
                residues
                    .iter()
                    .all(|r| (r + MODULUS - first).is_multiple_of(modulus))
            })
        };
        for set in sample_sets() {
            let Some(step) = set.step() else {
                assert!(set.is_bottom());
                continue;
            };
            assert!(shared(set, step), "{set:?} steps by {step}");
            let mut larger = step * 2;
            while larger <= MODULUS {
                assert!(!shared(set, larger), "{set:?} steps by {larger} too");
                larger *= 2;
            }
        }
    }

    /// The residues are walked in order, and a combination that reaches every
    /// residue early gives the same answer as the full walk.
    #[test]
    fn residues_are_walked_in_order_and_saturation_stops_the_walk() {
        let set = Congruence::exact(3)
            .join(Congruence::exact(0))
            .join(Congruence::exact(63));
        assert_eq!(set.residues().collect::<Vec<_>>(), vec![0, 3, 63]);
        assert_eq!(Congruence::BOTTOM.residues().count(), 0);
        assert_eq!(Congruence::TOP.residues().count(), 64);
        // `x + y` over an unconstrained `x` saturates on the first `y`.
        assert!(Congruence::TOP.sum(set).is_top());
        assert!(set.sum(Congruence::TOP).is_top());
    }
}
