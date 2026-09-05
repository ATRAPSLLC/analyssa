//! Lattice traits for data flow analysis.
//!
//! A lattice is a mathematical structure defining how abstract values combine
//! at control flow join/split points. This module provides the fundamental
//! traits that analysis domains must implement, along with lattice instances
//! for common types.
//!
//! # Lattice Theory
//!
//! For data flow analysis, lattices model the abstract values of program state:
//!
//! - **Partial Order**: Elements can be compared (≤)
//! - **Meet (∧)**: Greatest lower bound — combines information from multiple
//!   paths (intersection for must-analysis, union for may-analysis)
//! - **Join (∨)**: Least upper bound — merges information conservatively
//! - **Top (⊤)**: Greatest element ("no information" / "unknown")
//! - **Bottom (⊥)**: Least element ("conflict" / "all information")
//!
//! # Trait Hierarchy
//!
//! ```text
//! MeetSemiLattice (meet + is_bottom)
//!     |
//! JoinSemiLattice (join + is_top)
//!     |
//! Lattice (top + bottom + both operations)
//! ```
//!
//! # Forward vs Backward Analysis
//!
//! - **Forward analyses** (reaching definitions, constant propagation):
//!   Use `MeetSemiLattice::meet()` at join points (multiple predecessors)
//! - **Backward analyses** (liveness): Use `JoinSemiLattice::join()` at split
//!   points (multiple successors)
//!
//! The solver automatically selects the appropriate operation based on the
//! `Direction` enum. For `BitSet`-based analyses:
//! - Meet = union (for may-analysis like reaching definitions)
//! - Join = intersection (for must-analysis)
//!
//! # Requirements
//!
//! Meet operation must satisfy:
//! - Idempotent: `x.meet(x) = x`
//! - Commutative: `x.meet(y) = y.meet(x)`
//! - Associative: `x.meet(y.meet(z)) = (x.meet(y)).meet(z)`

use std::fmt::Debug;

use crate::bitset::BitSet;

/// A meet semi-lattice with a meet (greatest lower bound) operation.
///
/// The meet operation combines information from multiple control flow paths.
/// It must satisfy:
///
/// - **Idempotent**: `x.meet(x) = x`
/// - **Commutative**: `x.meet(y) = y.meet(x)`
/// - **Associative**: `x.meet(y.meet(z)) = (x.meet(y)).meet(z)`
///
/// # Examples
///
/// ```rust
/// use analyssa::analysis::dataflow::MeetSemiLattice;
///
/// #[derive(Clone, Debug, PartialEq)]
/// enum ConstantLattice {
///     /// No information yet.
///     Top,
///     /// Known to hold exactly this constant.
///     Const(i32),
///     /// Conflicting values on different paths.
///     Bottom,
/// }
///
/// impl MeetSemiLattice for ConstantLattice {
///     fn meet(&self, other: &Self) -> Self {
///         match (self, other) {
///             (Self::Top, x) | (x, Self::Top) => x.clone(),
///             (Self::Const(a), Self::Const(b)) if a == b => Self::Const(*a),
///             _ => Self::Bottom,
///         }
///     }
///
///     fn is_bottom(&self) -> bool {
///         matches!(self, Self::Bottom)
///     }
/// }
///
/// let top = ConstantLattice::Top;
/// let one = ConstantLattice::Const(1);
/// let two = ConstantLattice::Const(2);
///
/// // Top is the identity: meeting with it preserves information.
/// assert_eq!(top.meet(&one), one);
///
/// // Idempotent and commutative.
/// assert_eq!(one.meet(&one), one);
/// assert_eq!(one.meet(&two), two.meet(&one));
///
/// // Conflicting constants collapse to Bottom, which absorbs further meets.
/// let conflict = one.meet(&two);
/// assert_eq!(conflict, ConstantLattice::Bottom);
/// assert!(conflict.is_bottom());
/// assert_eq!(conflict.meet(&one), ConstantLattice::Bottom);
/// ```
pub trait MeetSemiLattice: Clone + Debug + PartialEq {
    /// Meets `other` into `self` in place.
    ///
    /// Defaults to `*self = self.meet(other)`, so implementing it is optional.
    /// Override it where the meet can be done without allocating — a `BitSet`
    /// union, for example. The solver folds the meet over every predecessor of
    /// every block on every visit, so the default's per-predecessor temporary is
    /// the dominant allocation in a dense CFG.
    fn meet_into(&mut self, other: &Self) {
        *self = self.meet(other);
    }

    /// Computes the meet (greatest lower bound) of two lattice elements.
    ///
    /// The meet represents combining information from two paths that merge.
    #[must_use]
    fn meet(&self, other: &Self) -> Self;

    /// Returns `true` if this is the bottom element.
    ///
    /// The bottom element represents "all information" or "conflict".
    /// Once bottom is reached, further meets cannot change the value.
    fn is_bottom(&self) -> bool;
}

/// A join semi-lattice with a join (least upper bound) operation.
///
/// The join operation combines information when paths split (for backward analysis)
/// or when we want to widen the approximation.
///
/// It must satisfy:
///
/// - **Idempotent**: `x.join(x) = x`
/// - **Commutative**: `x.join(y) = y.join(x)`
/// - **Associative**: `x.join(y.join(z)) = (x.join(y)).join(z)`
pub trait JoinSemiLattice: Clone + Debug + PartialEq {
    /// Computes the join (least upper bound) of two lattice elements.
    ///
    /// The join represents the least specific value that covers both inputs.
    #[must_use]
    fn join(&self, other: &Self) -> Self;

    /// Returns `true` if this is the top element.
    ///
    /// The top element represents "no information" or "unknown".
    /// It is the identity for meet: `x.meet(top) = x`.
    fn is_top(&self) -> bool;
}

/// A complete lattice with both meet and join operations.
///
/// Most data flow analyses operate over complete lattices, which have
/// both a greatest and least element, plus meet and join operations.
///
/// # Required Properties
///
/// - All properties of `MeetSemiLattice` and `JoinSemiLattice`
/// - **Absorption**: `x.meet(x.join(y)) = x` and `x.join(x.meet(y)) = x`
pub trait Lattice: MeetSemiLattice + JoinSemiLattice {
    /// Returns the top (⊤) element of the lattice.
    ///
    /// Top represents "no information" and is the identity for meet.
    fn top() -> Self;

    /// Returns the bottom (⊥) element of the lattice.
    ///
    /// Bottom represents "all information" or "conflict".
    fn bottom() -> Self;
}

// Lattice trait implementations for BitSet (defined in crate::utils::bitset)

impl MeetSemiLattice for BitSet {
    /// Meet is union for reaching definitions (may analysis).
    fn meet(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.union_with(other);
        result
    }

    /// Union in place — no temporary set per predecessor.
    fn meet_into(&mut self, other: &Self) {
        self.union_with(other);
    }

    fn is_bottom(&self) -> bool {
        // For may analysis, bottom is the full set.
        self.is_full()
    }
}

impl JoinSemiLattice for BitSet {
    /// Join is intersection for reaching definitions.
    fn join(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.intersect_with(other);
        result
    }

    fn is_top(&self) -> bool {
        self.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::dataflow::{JoinSemiLattice, Lattice, MeetSemiLattice};

    /// The lattice traits must stay reachable from `analysis::dataflow` directly,
    /// like every other trait the module's header advertises — not only via the
    /// longer `dataflow::lattice::` path, which would make the documented import
    /// fail to compile. Importing through the short path here pins the
    /// re-export.
    /// A two-point lattice, the smallest thing that can implement [`Lattice`].
    ///
    /// No type in the crate implements the full [`Lattice`] trait — `BitSet`
    /// provides only the two halves — so this also serves as the proof that the
    /// trait is implementable as declared.
    #[derive(Clone, Debug, PartialEq)]
    enum TwoPoint {
        Top,
        Bottom,
    }

    impl MeetSemiLattice for TwoPoint {
        fn meet(&self, other: &Self) -> Self {
            if *self == Self::Bottom || *other == Self::Bottom {
                Self::Bottom
            } else {
                Self::Top
            }
        }

        fn is_bottom(&self) -> bool {
            *self == Self::Bottom
        }
    }

    impl JoinSemiLattice for TwoPoint {
        fn join(&self, other: &Self) -> Self {
            if *self == Self::Top || *other == Self::Top {
                Self::Top
            } else {
                Self::Bottom
            }
        }

        fn is_top(&self) -> bool {
            *self == Self::Top
        }
    }

    impl Lattice for TwoPoint {
        fn top() -> Self {
            Self::Top
        }

        fn bottom() -> Self {
            Self::Bottom
        }
    }

    /// The lattice traits must stay reachable from `analysis::dataflow` directly,
    /// like every other trait the module header advertises — not only via the
    /// longer `dataflow::lattice::` path, which would make the documented import
    /// fail to compile. Naming them through the short path here pins the
    /// re-export.
    /// `meet_into` is an optimisation of `meet`, so an override that disagrees
    /// with it would silently corrupt every analysis built on the lattice. The
    /// solver calls it once per predecessor per visit, so a divergence would be
    /// both pervasive and hard to attribute.
    #[test]
    fn meet_into_agrees_with_meet_for_every_lattice() {
        fn check<L: MeetSemiLattice>(a: &L, b: &L) {
            let expected = a.meet(b);
            let mut actual = a.clone();
            actual.meet_into(b);
            assert_eq!(actual, expected, "meet_into must equal meet");
        }

        let mut a = BitSet::new(128);
        a.insert(1);
        a.insert(70);
        let mut b = BitSet::new(128);
        b.insert(70);
        b.insert(100);
        check(&a, &b);
        check(&b, &a);
        check(&a, &a.clone());
        check(&BitSet::new(128), &a);
        check(&a, &BitSet::new(128));
        // A lazily-allocated operand must behave identically.
        check(&a, &BitSet::lazy(128));
        check(&BitSet::lazy(128), &a);

        // And the default implementation, via a lattice that does not override.
        check(&TwoPoint::Top, &TwoPoint::Bottom);
        check(&TwoPoint::Bottom, &TwoPoint::Top);
        check(&TwoPoint::Top, &TwoPoint::Top);
    }

    #[test]
    fn lattice_traits_are_re_exported_from_dataflow() {
        fn assert_meet<T: MeetSemiLattice>() {}
        fn assert_join<T: JoinSemiLattice>() {}
        fn assert_lattice<T: Lattice>() {}

        assert_meet::<BitSet>();
        assert_join::<BitSet>();
        assert_lattice::<TwoPoint>();

        // Absorption, the property `Lattice` documents but cannot enforce.
        let (top, bottom) = (TwoPoint::top(), TwoPoint::bottom());
        assert_eq!(top.meet(&top.join(&bottom)), top);
        assert_eq!(top.join(&top.meet(&bottom)), top);
    }

    #[test]
    fn test_bitset_meet_union() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(3);

        let mut b = BitSet::new(10);
        b.insert(2);
        b.insert(3);

        let result = a.meet(&b);

        // Meet is union: {1, 3} ∪ {2, 3} = {1, 2, 3}
        assert!(result.contains(1));
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(!result.contains(0));
        assert!(!result.contains(4));
    }

    #[test]
    fn test_bitset_meet_idempotent() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(5);

        let result = a.meet(&a);

        // Idempotent: x.meet(x) = x
        assert_eq!(a, result);
    }

    #[test]
    fn test_bitset_meet_commutative() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(3);

        let mut b = BitSet::new(10);
        b.insert(2);
        b.insert(4);

        // Commutative: x.meet(y) = y.meet(x)
        assert_eq!(a.meet(&b), b.meet(&a));
    }

    #[test]
    fn test_bitset_meet_associative() {
        let mut a = BitSet::new(10);
        a.insert(1);

        let mut b = BitSet::new(10);
        b.insert(2);

        let mut c = BitSet::new(10);
        c.insert(3);

        // Associative: x.meet(y.meet(z)) = (x.meet(y)).meet(z)
        let left = a.meet(&b.meet(&c));
        let right = a.meet(&b).meet(&c);
        assert_eq!(left, right);
    }

    #[test]
    fn test_bitset_join_intersection() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(2);
        a.insert(3);

        let mut b = BitSet::new(10);
        b.insert(2);
        b.insert(3);
        b.insert(4);

        let result = a.join(&b);

        // Join is intersection: {1, 2, 3} ∩ {2, 3, 4} = {2, 3}
        assert!(!result.contains(1));
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(!result.contains(4));
    }

    #[test]
    fn test_bitset_join_idempotent() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(5);

        let result = a.join(&a);

        // Idempotent: x.join(x) = x
        assert_eq!(a, result);
    }

    #[test]
    fn test_bitset_join_commutative() {
        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(3);

        let mut b = BitSet::new(10);
        b.insert(2);
        b.insert(3);

        // Commutative: x.join(y) = y.join(x)
        assert_eq!(a.join(&b), b.join(&a));
    }

    #[test]
    fn test_bitset_is_top_empty() {
        let empty = BitSet::new(10);
        assert!(empty.is_top());

        let mut non_empty = BitSet::new(10);
        non_empty.insert(0);
        assert!(!non_empty.is_top());
    }

    #[test]
    fn test_bitset_is_bottom_full() {
        let full = BitSet::full(10);
        assert!(full.is_bottom());

        let mut partial = BitSet::new(10);
        partial.insert(0);
        assert!(!partial.is_bottom());
    }

    #[test]
    fn test_bitset_meet_with_empty() {
        let empty = BitSet::new(10);

        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(2);

        // Meet with empty (top) should give the other set
        let result = a.meet(&empty);
        assert!(result.contains(1));
        assert!(result.contains(2));
        assert_eq!(result.count(), 2);
    }

    #[test]
    fn test_bitset_join_with_empty() {
        let empty = BitSet::new(10);

        let mut a = BitSet::new(10);
        a.insert(1);
        a.insert(2);

        // Join with empty (top) should give empty
        let result = a.join(&empty);
        assert!(result.is_empty());
    }
}
