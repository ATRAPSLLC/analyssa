//! Value range analysis for SSA variables.
//!
//! This module provides interval-based range analysis for tracking the possible
//! values of integer variables. It is used for opaque predicate detection, bounds
//! check elimination, and general range-based optimization.
//!
//! # Range Types
//!
//! | Type | Representation | Example |
//! |------|---------------|---------|
//! | Constant | `[5, 5]` | Exact value known |
//! | Bounded | `[0, 255]` | Value within bounds |
//! | Half-open | `[0, +∞)` | Non-negative |
//! | Half-open | `(-∞, 10]` | Upper bound only |
//! | Union | `[0, 5] ∪ [10, 15]` | Disjoint intervals |
//!
//! # Lattice Structure
//!
//! The `ValueRange` forms a lattice for dataflow analysis:
//!
//! ```text
//!           Top (all values - no information)
//!              |
//!     [MIN, MAX] full range
//!        /    \
//!   [a, b]    [c, d]  bounded ranges
//!        \    /
//!         [x, x]  constant (singleton)
//!           |
//!        Bottom (no values - unreachable)
//! ```
//!
//! # Lattice Operations
//!
//! - **Meet (intersection)**: `x.meet(y)` returns the range of values in BOTH
//!   x and y. Used at control flow merge points for forward analyses where
//!   information must be true on all incoming paths.
//!
//! - **Join (union hull)**: `x.join(y)` returns a range containing all values
//!   from EITHER x or y (may include values not in either due to hull widening).
//!   Used at control flow merge points for backward analyses.
//!
//! - **Widen**: `x.widen(y)` extends bounds toward infinity when loop iteration
//!   causes bounds to grow, ensuring termination of fixed-point iteration.
//!   If min grows downward or max grows upward, the bound is widened to infinity.
//!
//! # Range Arithmetic
//!
//! Ranges support arithmetic operations using interval arithmetic:
//!
//! - **Add**: `[a, b] + [c, d] = [a + c, b + d]`
//! - **Sub**: `[a, b] - [c, d] = [a - d, b - c]`
//! - **Mul**: `[a, b] * [c, d] = [min(a*c, a*d, b*c, b*d), max(...)]`
//!   (all four corner products considered, overflow yields full range)
//! - **And constant**: `x & mask` bounds the result to `[0, mask]` for
//!   non-negative masks
//!
//! # Complexity
//!
//! All operations are O(1) for Interval ranges and O(k) for Union ranges where
//! k is the number of intervals. Normalization after union operations is O(k log k).
//!
//! ```rust
//! use analyssa::analysis::ValueRange;
//!
//! // Create ranges
//! let constant = ValueRange::constant(42);
//! let non_negative = ValueRange::non_negative();
//! let byte_range = ValueRange::bounded(0, 255);
//!
//! // Query ranges
//! assert!(constant.is_constant());
//! assert!(non_negative.is_always_non_negative());
//! assert!(byte_range.always_less_than(256) == Some(true));
//!
//! // Range arithmetic
//! let sum = byte_range.add(&ValueRange::constant(1));
//! assert_eq!(sum.min(), Some(1));
//! assert_eq!(sum.max(), Some(256));
//! ```

use std::{
    cmp::{Ordering, max, min},
    fmt,
};

use crate::{
    analysis::dataflow::lattice::{JoinSemiLattice, MeetSemiLattice},
    ir::ops::native::FlagCondition,
};

/// A range of possible integer values for analysis.
///
/// Represents the set of values a variable might hold at runtime.
/// Used for opaque predicate detection, bounds check elimination,
/// and general range-based optimization.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub enum ValueRange {
    /// No possible values (unreachable code).
    Bottom,

    /// A single contiguous interval `[min, max]`.
    Interval(IntervalRange),

    /// A union of disjoint intervals (for precision).
    /// Intervals are sorted by min value and non-overlapping.
    Union(Vec<IntervalRange>),

    /// All values possible (no information).
    #[default]
    Top,
}

/// Lets the generic dataflow solver drive a range analysis: the inherent
/// `meet`/`join` already have the right algebra, this exposes them under the
/// lattice traits the solver is written against.
impl MeetSemiLattice for ValueRange {
    fn meet(&self, other: &Self) -> Self {
        Self::meet(self, other)
    }

    fn is_bottom(&self) -> bool {
        Self::is_bottom(self)
    }
}

impl JoinSemiLattice for ValueRange {
    fn join(&self, other: &Self) -> Self {
        Self::join(self, other)
    }

    fn is_top(&self) -> bool {
        Self::is_top(self)
    }
}

/// A single contiguous interval `[min, max]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntervalRange {
    /// Minimum value (inclusive). `None` means negative infinity.
    pub min: Option<i64>,
    /// Maximum value (inclusive). `None` means positive infinity.
    pub max: Option<i64>,
}

impl IntervalRange {
    /// Creates a new interval range.
    #[must_use]
    pub const fn new(min: Option<i64>, max: Option<i64>) -> Self {
        Self { min, max }
    }

    /// Creates a constant (singleton) interval.
    #[must_use]
    pub const fn constant(value: i64) -> Self {
        Self {
            min: Some(value),
            max: Some(value),
        }
    }

    /// Creates a bounded interval `[min, max]`.
    #[must_use]
    pub const fn bounded(min: i64, max: i64) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
        }
    }

    /// Creates a non-negative interval `[0, +∞)`.
    #[must_use]
    pub const fn non_negative() -> Self {
        Self {
            min: Some(0),
            max: None,
        }
    }

    /// Creates an interval from min to infinity `[min, +∞)`.
    #[must_use]
    pub const fn at_least(min: i64) -> Self {
        Self {
            min: Some(min),
            max: None,
        }
    }

    /// Creates an interval from negative infinity to max `(-∞, max]`.
    #[must_use]
    pub const fn at_most(max: i64) -> Self {
        Self {
            min: None,
            max: Some(max),
        }
    }

    /// Creates a full interval `(-∞, +∞)`.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            min: None,
            max: None,
        }
    }

    /// Returns `true` if this is a constant (singleton) range.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        matches!((self.min, self.max), (Some(a), Some(b)) if a == b)
    }

    /// Returns the constant value if this is a singleton.
    #[must_use]
    pub fn as_constant(&self) -> Option<i64> {
        match (self.min, self.max) {
            (Some(a), Some(b)) if a == b => Some(a),
            _ => None,
        }
    }

    /// Returns `true` if all values in this range are non-negative.
    #[must_use]
    pub fn is_always_non_negative(&self) -> bool {
        self.min.is_some_and(|m| m >= 0)
    }

    /// Returns `true` if all values in this range are positive.
    #[must_use]
    pub fn is_always_positive(&self) -> bool {
        self.min.is_some_and(|m| m > 0)
    }

    /// Returns `true` if all values in this range are negative.
    #[must_use]
    pub fn is_always_negative(&self) -> bool {
        self.max.is_some_and(|m| m < 0)
    }

    /// Returns `true` if all values in this range are non-positive.
    #[must_use]
    pub fn is_always_non_positive(&self) -> bool {
        self.max.is_some_and(|m| m <= 0)
    }

    /// Determines whether every value in the range is strictly less than `value`.
    ///
    /// The query is decided by whichever bound is relevant: a known `max`
    /// settles it outright (all values are `< value` exactly when `max < value`,
    /// otherwise the maximum itself is a counterexample). With `max` unbounded a
    /// known `min` at or above `value` still proves the answer `false`. When
    /// neither bound constrains the side that matters the answer is undecidable.
    ///
    /// # Arguments
    ///
    /// * `value` - The exclusive upper threshold each range value is tested against.
    ///
    /// # Returns
    ///
    /// `Some(true)` if all values are `< value`, `Some(false)` if at least one
    /// is not, or `None` if the bounds are insufficient to decide.
    #[must_use]
    pub fn always_less_than(&self, value: i64) -> Option<bool> {
        if let Some(max_val) = self.max {
            if max_val < value {
                return Some(true);
            }
            // If max >= value, then at least one value (max) is NOT < value
            return Some(false);
        }
        if let Some(min_val) = self.min
            && min_val >= value
        {
            return Some(false);
        }
        None
    }

    /// Determines whether every value in the range is strictly greater than `value`.
    ///
    /// Mirror of [`always_less_than`](Self::always_less_than): a known `min`
    /// decides it (all values are `> value` exactly when `min > value`, otherwise
    /// the minimum is a counterexample), and with `min` unbounded a known `max`
    /// at or below `value` still proves the answer `false`. Otherwise undecidable.
    ///
    /// # Arguments
    ///
    /// * `value` - The exclusive lower threshold each range value is tested against.
    ///
    /// # Returns
    ///
    /// `Some(true)` if all values are `> value`, `Some(false)` if at least one
    /// is not, or `None` if the bounds are insufficient to decide.
    #[must_use]
    pub fn always_greater_than(&self, value: i64) -> Option<bool> {
        if let Some(min_val) = self.min {
            if min_val > value {
                return Some(true);
            }
            // If min <= value, then at least one value (min) is NOT > value
            return Some(false);
        }
        if let Some(max_val) = self.max
            && max_val <= value
        {
            return Some(false);
        }
        None
    }

    /// Determines whether every value in the range is less than or equal to `value`.
    ///
    /// A known `max` at or below `value` proves `true`; a known `min` strictly
    /// above `value` proves `false`. If the deciding bound is unbounded the
    /// answer is `None`.
    ///
    /// # Arguments
    ///
    /// * `value` - The inclusive upper threshold each range value is tested against.
    ///
    /// # Returns
    ///
    /// `Some(true)` if all values are `<= value`, `Some(false)` if at least one
    /// is not, or `None` if the bounds are insufficient to decide.
    #[must_use]
    pub fn always_less_equal(&self, value: i64) -> Option<bool> {
        if let Some(max_val) = self.max
            && max_val <= value
        {
            return Some(true);
        }
        if let Some(min_val) = self.min
            && min_val > value
        {
            return Some(false);
        }
        None
    }

    /// Determines whether every value in the range is greater than or equal to `value`.
    ///
    /// A known `min` at or above `value` proves `true`; a known `max` strictly
    /// below `value` proves `false`. If the deciding bound is unbounded the
    /// answer is `None`.
    ///
    /// # Arguments
    ///
    /// * `value` - The inclusive lower threshold each range value is tested against.
    ///
    /// # Returns
    ///
    /// `Some(true)` if all values are `>= value`, `Some(false)` if at least one
    /// is not, or `None` if the bounds are insufficient to decide.
    #[must_use]
    pub fn always_greater_equal(&self, value: i64) -> Option<bool> {
        if let Some(min_val) = self.min
            && min_val >= value
        {
            return Some(true);
        }
        if let Some(max_val) = self.max
            && max_val < value
        {
            return Some(false);
        }
        None
    }

    /// Determines whether the range is exactly the singleton `value`.
    ///
    /// Only a fully bounded interval can answer definitively: it equals `value`
    /// for all elements exactly when it is the singleton `[value, value]`, and
    /// it is provably never `value` when `value` falls outside `[min, max]`. A
    /// bounded interval that contains `value` alongside other elements returns
    /// `None` (it may but need not equal `value`), as does any half-open or
    /// unbounded interval.
    ///
    /// # Arguments
    ///
    /// * `value` - The value tested for exact equality against the whole range.
    ///
    /// # Returns
    ///
    /// `Some(true)` if the range is the singleton `value`, `Some(false)` if
    /// `value` is outside the range, or `None` if it cannot be decided.
    #[must_use]
    pub fn always_equal_to(&self, value: i64) -> Option<bool> {
        match (self.min, self.max) {
            (Some(min_val), Some(max_val)) => {
                if min_val == max_val && min_val == value {
                    Some(true)
                } else if value < min_val || value > max_val {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Returns `true` if `value` lies within this interval.
    ///
    /// An unbounded side (`None`) never rules `value` out, so a half-open or
    /// full interval is treated as extending to the corresponding infinity.
    ///
    /// # Arguments
    ///
    /// * `value` - The candidate value to test for membership.
    ///
    /// # Returns
    ///
    /// `true` if `value` is on or inside both bounds.
    #[must_use]
    pub fn may_contain(&self, value: i64) -> bool {
        let above_min = self.min.is_none_or(|m| value >= m);
        let below_max = self.max.is_none_or(|m| value <= m);
        above_min && below_max
    }

    /// Returns `true` if this interval shares at least one value with `other`.
    ///
    /// Rather than test for a common point directly, it checks the two ways the
    /// intervals could be disjoint - this one entirely below `other`, or `other`
    /// entirely below this - and reports overlap when neither holds. A missing
    /// bound cannot separate the intervals, so it never counts as disjoint.
    ///
    /// # Arguments
    ///
    /// * `other` - The interval tested for a shared value.
    ///
    /// # Returns
    ///
    /// `true` if the intervals are not disjoint.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        // Check if intervals are disjoint
        let self_below_other = match (self.max, other.min) {
            (Some(self_max), Some(other_min)) => self_max < other_min,
            _ => false,
        };
        let other_below_self = match (other.max, self.min) {
            (Some(other_max), Some(self_min)) => other_max < self_min,
            _ => false,
        };
        !self_below_other && !other_below_self
    }

    /// Returns `true` if this interval abuts `other` with no gap between them.
    ///
    /// Adjacency means one interval's max is exactly one below the other's min
    /// (`self_max + 1 == other_min` or the symmetric case), so their union is
    /// still a single contiguous interval. The `checked_add` guards the edge
    /// where incrementing `i64::MAX` would overflow. Union normalization uses
    /// this to fuse touching intervals rather than keep them separate.
    ///
    /// # Arguments
    ///
    /// * `other` - The interval tested for adjacency.
    ///
    /// # Returns
    ///
    /// `true` if the two intervals can be merged into one without a gap.
    #[must_use]
    pub fn adjacent(&self, other: &Self) -> bool {
        // Check if max + 1 == other.min or other.max + 1 == self.min
        if let (Some(self_max), Some(other_min)) = (self.max, other.min)
            && self_max.checked_add(1) == Some(other_min)
        {
            return true;
        }
        if let (Some(other_max), Some(self_min)) = (other.max, self.min)
            && other_max.checked_add(1) == Some(self_min)
        {
            return true;
        }
        false
    }

    /// Intersects two intervals, yielding the values common to both.
    ///
    /// The intersection takes the larger of the two lower bounds and the smaller
    /// of the two upper bounds, treating a `None` bound as infinity (so the other
    /// operand's finite bound wins). If the resulting min exceeds the resulting
    /// max the intervals are disjoint and there is no intersection.
    ///
    /// # Arguments
    ///
    /// * `other` - The interval to intersect with.
    ///
    /// # Returns
    ///
    /// `Some` intersection interval, or `None` if the intervals do not overlap.
    #[must_use]
    pub fn meet(&self, other: &Self) -> Option<Self> {
        let new_min = match (self.min, other.min) {
            (Some(a), Some(b)) => Some(max(a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        let new_max = match (self.max, other.max) {
            (Some(a), Some(b)) => Some(min(a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        // Check if result is empty
        if let (Some(min_val), Some(max_val)) = (new_min, new_max)
            && min_val > max_val
        {
            return None; // Empty intersection
        }

        Some(Self {
            min: new_min,
            max: new_max,
        })
    }

    /// Forms the convex hull of two intervals - the smallest interval containing both.
    ///
    /// The hull takes the smaller of the two lower bounds and the larger of the
    /// two upper bounds; if either operand is unbounded on a side the hull is
    /// unbounded there too. This can admit values in neither input (the gap
    /// between disjoint intervals is filled), which is the precision loss that
    /// keeps the result a single interval rather than a union.
    ///
    /// # Arguments
    ///
    /// * `other` - The interval to hull with.
    ///
    /// # Returns
    ///
    /// The tightest single interval enclosing both operands.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        let new_min = match (self.min, other.min) {
            (Some(a), Some(b)) => Some(min(a, b)),
            _ => None, // Either unbounded -> result unbounded
        };

        let new_max = match (self.max, other.max) {
            (Some(a), Some(b)) => Some(max(a, b)),
            _ => None, // Either unbounded -> result unbounded
        };

        Self {
            min: new_min,
            max: new_max,
        }
    }

    /// Widens this interval toward `other`, extending any growing bound to infinity.
    ///
    /// Widening is the classic termination device for fixed-point iteration over
    /// loops: comparing the current interval (`self`) against the next iterate
    /// (`other`), a lower bound that moved down or an upper bound that moved up
    /// is jumped straight to infinity rather than tracked step by step. Bounds
    /// that are stable or tightening are kept, so the sequence can only lose
    /// bounds a finite number of times and must converge.
    ///
    /// # Arguments
    ///
    /// * `other` - The next iterate whose bound movement drives widening.
    ///
    /// # Returns
    ///
    /// A widened interval with each growing bound relaxed to infinity.
    #[must_use]
    pub fn widen(&self, other: &Self) -> Self {
        let new_min = match (self.min, other.min) {
            // `None` is -infinity, so an unbounded incoming bound is *lower*
            // than any finite one: it has to widen the result to -infinity too.
            // Matching only `(Some, Some)` would keep the finite bound and
            // produce a result narrower than the join - the unsound direction.
            (Some(a), Some(b)) if b < a => None, // Growing down -> -∞
            (Some(_), None) => None,             // Already unbounded below
            (min_val, _) => min_val,             // Keep current
        };

        let new_max = match (self.max, other.max) {
            // `None` is +infinity; same argument as `new_min` above.
            (Some(a), Some(b)) if b > a => None, // Growing up -> +∞
            (Some(_), None) => None,             // Already unbounded above
            (max_val, _) => max_val,             // Keep current
        };

        Self {
            min: new_min,
            max: new_max,
        }
    }

    /// Adds two intervals using interval arithmetic: `[a, b] + [c, d] = [a + c, b + d]`.
    ///
    /// Each bound is the sum of the corresponding bounds; an unbounded operand
    /// keeps the result unbounded on that side. A `checked_add` that overflows
    /// also drops the bound to infinity rather than wrapping, so the result stays
    /// a sound over-approximation.
    ///
    /// # Arguments
    ///
    /// * `other` - The interval to add.
    ///
    /// # Returns
    ///
    /// The interval of all possible sums.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let new_min = match (self.min, other.min) {
            (Some(a), Some(b)) => a.checked_add(b),
            _ => None,
        };

        let new_max = match (self.max, other.max) {
            (Some(a), Some(b)) => a.checked_add(b),
            _ => None,
        };

        Self {
            min: new_min,
            max: new_max,
        }
    }

    /// Subtracts two intervals using interval arithmetic: `[a, b] - [c, d] = [a - d, b - c]`.
    ///
    /// Subtraction flips the operand's bounds: the smallest difference pairs this
    /// interval's min with `other`'s max, and the largest pairs this max with
    /// `other`'s min. An unbounded operand bound or a `checked_sub` overflow
    /// leaves the corresponding result bound at infinity.
    ///
    /// # Arguments
    ///
    /// * `other` - The interval to subtract.
    ///
    /// # Returns
    ///
    /// The interval of all possible differences.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        // [a, b] - [c, d] = [a - d, b - c]
        let new_min = match (self.min, other.max) {
            (Some(a), Some(d)) => a.checked_sub(d),
            _ => None,
        };

        let new_max = match (self.max, other.min) {
            (Some(b), Some(c)) => b.checked_sub(c),
            _ => None,
        };

        Self {
            min: new_min,
            max: new_max,
        }
    }

    /// Multiplies two intervals, considering all four corner products.
    ///
    /// Because sign changes reorder the extremes, the product interval is not
    /// simply `[a*c, b*d]`: it is the min and max over the four corner products
    /// (`a*c, a*d, b*c, b*d`). Multiplication is only attempted when both
    /// intervals are fully bounded; any unbounded operand, or any corner product
    /// that overflows, collapses the result to the full range so it stays sound.
    ///
    /// # Arguments
    ///
    /// * `other` - The interval to multiply by.
    ///
    /// # Returns
    ///
    /// The interval spanning all corner products, or the full range on overflow
    /// or an unbounded operand.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        // For multiplication, we need to consider all corner combinations
        // because signs matter
        match (self.min, self.max, other.min, other.max) {
            (Some(a), Some(b), Some(c), Some(d)) => {
                // Compute all four products
                let products = [
                    a.checked_mul(c),
                    a.checked_mul(d),
                    b.checked_mul(c),
                    b.checked_mul(d),
                ];

                // If any overflowed, return unbounded
                if products.iter().any(Option::is_none) {
                    return Self::full();
                }

                let products: Vec<i64> = products.iter().filter_map(|&p| p).collect();
                let new_min = products.iter().copied().min();
                let new_max = products.iter().copied().max();

                Self {
                    min: new_min,
                    max: new_max,
                }
            }
            _ => Self::full(),
        }
    }

    /// Bounds the result of a bitwise AND against a constant `mask`.
    ///
    /// For a non-negative mask, `x & mask` cannot exceed `mask` nor go below
    /// zero regardless of `x`, so the result is `[0, mask]` - a cheap, sound
    /// bound that ignores `self` entirely. A negative mask has its sign bit set,
    /// so no such bound holds and the result is the full range.
    ///
    /// # Arguments
    ///
    /// * `mask` - The constant AND mask.
    ///
    /// # Returns
    ///
    /// `[0, mask]` for a non-negative mask, otherwise the full range.
    #[must_use]
    pub fn and_constant(&self, mask: i64) -> Self {
        if mask >= 0 {
            // AND with non-negative mask always produces [0, mask]
            Self::bounded(0, mask)
        } else {
            Self::full()
        }
    }

    /// Bounds the result of a bitwise OR against a constant `value`.
    ///
    /// OR is hard to bound precisely, so this handles only the tractable case:
    /// when both `self` and `value` are non-negative, `x | value` is at least
    /// `max(self.min, value)` (OR-ing in set bits can only raise the value) and
    /// has no useful upper bound. Any other case falls back to the full range.
    ///
    /// # Arguments
    ///
    /// * `value` - The constant OR operand.
    ///
    /// # Returns
    ///
    /// A lower-bounded range when both operands are non-negative, otherwise the
    /// full range.
    #[must_use]
    pub fn or_constant(&self, value: i64) -> Self {
        // Hard to compute precisely, be conservative
        if self.is_always_non_negative() && value >= 0 {
            // Both non-negative: result >= max(self.min, value)
            let new_min = max(self.min.unwrap_or(0), value);
            Self {
                min: Some(new_min),
                max: None,
            }
        } else {
            Self::full()
        }
    }
}

impl Default for IntervalRange {
    fn default() -> Self {
        Self::full()
    }
}

impl fmt::Display for IntervalRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.min, self.max) {
            (Some(a), Some(b)) if a == b => write!(f, "{a}"),
            (Some(a), Some(b)) => write!(f, "[{a}, {b}]"),
            (Some(a), None) => write!(f, "[{a}, +∞)"),
            (None, Some(b)) => write!(f, "(-∞, {b}]"),
            (None, None) => write!(f, "(-∞, +∞)"),
        }
    }
}

impl ValueRange {
    /// Creates the bottom element (empty set, unreachable).
    #[must_use]
    pub const fn bottom() -> Self {
        Self::Bottom
    }

    /// Creates the top element (all values, no information).
    #[must_use]
    pub const fn top() -> Self {
        Self::Top
    }

    /// Creates a constant (singleton) range.
    #[must_use]
    pub fn constant(value: i64) -> Self {
        Self::Interval(IntervalRange::constant(value))
    }

    /// Creates a bounded interval `[min, max]`.
    #[must_use]
    pub fn bounded(min_val: i64, max_val: i64) -> Self {
        if min_val > max_val {
            Self::Bottom
        } else {
            Self::Interval(IntervalRange::bounded(min_val, max_val))
        }
    }

    /// Creates a non-negative interval `[0, +∞)`.
    #[must_use]
    pub fn non_negative() -> Self {
        Self::Interval(IntervalRange::non_negative())
    }

    /// Creates an interval from min to infinity `[min, +∞)`.
    #[must_use]
    pub fn at_least(min_val: i64) -> Self {
        Self::Interval(IntervalRange::at_least(min_val))
    }

    /// Creates an interval from negative infinity to max `(-∞, max]`.
    #[must_use]
    pub fn at_most(max_val: i64) -> Self {
        Self::Interval(IntervalRange::at_most(max_val))
    }

    /// Creates a range for non-null references.
    ///
    /// This is semantically different from numeric ranges - it indicates
    /// that a reference is known to be non-null.
    #[must_use]
    pub fn non_null() -> Self {
        // For references, we use a special marker value
        // In practice, non-null is tracked separately, but we can use Top
        // to indicate "some reference value exists"
        Self::Top
    }

    /// Combines two ranges into the range of values in either.
    ///
    /// `Bottom` drops out and `Top` absorbs. Two intervals that overlap or abut
    /// collapse to their hull; otherwise they are kept as a sorted two-element
    /// `Union` so the gap between them is preserved. Anything involving an
    /// existing `Union` is flattened and handed to `normalize_union`, which
    /// re-sorts and re-merges the whole set.
    ///
    /// # Arguments
    ///
    /// * `a` - The first range.
    /// * `b` - The second range.
    ///
    /// # Returns
    ///
    /// The union, simplified to a single interval where the pieces are contiguous.
    #[must_use]
    pub fn union(a: Self, b: Self) -> Self {
        match (a, b) {
            (Self::Bottom, other) | (other, Self::Bottom) => other,
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Interval(ia), Self::Interval(ib)) => {
                if ia.overlaps(&ib) || ia.adjacent(&ib) {
                    Self::Interval(ia.join(&ib))
                } else {
                    // Sort intervals
                    let (first, second) = if ia.min <= ib.min { (ia, ib) } else { (ib, ia) };
                    Self::Union(vec![first, second])
                }
            }
            (Self::Union(mut intervals), Self::Interval(i))
            | (Self::Interval(i), Self::Union(mut intervals)) => {
                intervals.push(i);
                Self::normalize_union(intervals)
            }
            (Self::Union(mut a_intervals), Self::Union(b_intervals)) => {
                a_intervals.extend(b_intervals);
                Self::normalize_union(a_intervals)
            }
        }
    }

    /// Canonicalizes a bag of intervals into a sorted, gap-preserving range.
    ///
    /// Sorting by lower bound (unbounded-below first) puts touching intervals
    /// next to each other, so a single left-to-right sweep can fuse each
    /// interval into the previous one whenever they overlap or abut. The result
    /// is demoted to a single `Interval` when only one survives (and `Bottom`
    /// when none do), keeping the `Union` variant reserved for genuinely
    /// disjoint sets - the invariant the rest of the module relies on.
    ///
    /// # Arguments
    ///
    /// * `intervals` - The intervals to sort and merge; consumed by the call.
    ///
    /// # Returns
    ///
    /// The normalized range: `Bottom`, a single `Interval`, or a disjoint `Union`.
    fn normalize_union(mut intervals: Vec<IntervalRange>) -> Self {
        if intervals.is_empty() {
            return Self::Bottom;
        }
        if intervals.len() == 1 {
            return Self::Interval(intervals.remove(0));
        }

        // Sort by min value
        intervals.sort_by(|a, b| match (a.min, b.min) {
            (Some(a_min), Some(b_min)) => a_min.cmp(&b_min),
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        });

        // Merge overlapping/adjacent intervals
        let mut merged: Vec<IntervalRange> = Vec::new();
        for interval in intervals {
            if let Some(last) = merged.last_mut()
                && (last.overlaps(&interval) || last.adjacent(&interval))
            {
                *last = last.join(&interval);
                continue;
            }
            merged.push(interval);
        }

        if merged.len() == 1 {
            Self::Interval(merged.remove(0))
        } else {
            Self::Union(merged)
        }
    }

    /// Returns `true` if this is the bottom element (empty set).
    #[must_use]
    pub const fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }

    /// Returns `true` if this is the top element (all values).
    #[must_use]
    pub const fn is_top(&self) -> bool {
        matches!(self, Self::Top)
    }

    /// Returns `true` if this is a constant (singleton) range.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        match self {
            Self::Interval(i) => i.is_constant(),
            _ => false,
        }
    }

    /// Returns the constant value if this is a singleton.
    #[must_use]
    pub fn as_constant(&self) -> Option<i64> {
        match self {
            Self::Interval(i) => i.as_constant(),
            _ => None,
        }
    }

    /// Returns the minimum value if bounded below.
    #[must_use]
    pub fn min(&self) -> Option<i64> {
        match self {
            Self::Bottom | Self::Top => None,
            Self::Interval(i) => i.min,
            Self::Union(intervals) => intervals.first().and_then(|i| i.min),
        }
    }

    /// Returns the maximum value if bounded above.
    #[must_use]
    pub fn max(&self) -> Option<i64> {
        match self {
            Self::Bottom | Self::Top => None,
            Self::Interval(i) => i.max,
            Self::Union(intervals) => intervals.last().and_then(|i| i.max),
        }
    }

    /// Returns `true` if all values are non-negative.
    #[must_use]
    pub fn is_always_non_negative(&self) -> bool {
        match self {
            Self::Bottom => true, // Vacuously true
            Self::Top => false,
            Self::Interval(i) => i.is_always_non_negative(),
            Self::Union(intervals) => intervals.iter().all(IntervalRange::is_always_non_negative),
        }
    }

    /// Returns `true` if all values are negative.
    ///
    /// The partner of [`Self::is_always_non_negative`], and the other half of
    /// the pair a sign test needs: neither holding is what "the range straddles
    /// zero, so the sign bit is unknown" looks like.
    #[must_use]
    pub fn is_always_negative(&self) -> bool {
        match self {
            Self::Bottom => true, // Vacuously true
            Self::Top => false,
            Self::Interval(i) => i.is_always_negative(),
            Self::Union(intervals) => intervals.iter().all(IntervalRange::is_always_negative),
        }
    }

    /// Returns `true` if all values are positive.
    #[must_use]
    pub fn is_always_positive(&self) -> bool {
        match self {
            Self::Bottom => true,
            Self::Top => false,
            Self::Interval(i) => i.is_always_positive(),
            Self::Union(intervals) => intervals.iter().all(IntervalRange::is_always_positive),
        }
    }

    /// Determines whether every value in the range is strictly less than `value`.
    ///
    /// `Bottom` is vacuously `true` (no counterexample exists) and `Top` is
    /// always undecidable. An interval defers to
    /// [`IntervalRange::always_less_than`]. A `Union` combines its members'
    /// verdicts: `true` only if every member is definitely `true`, `false` if
    /// any member is definitely `false`, and `None` otherwise (some member is
    /// undecidable but none is a definite counterexample).
    ///
    /// # Arguments
    ///
    /// * `value` - The exclusive upper threshold each range value is tested against.
    ///
    /// # Returns
    ///
    /// `Some(true)` if all values are `< value`, `Some(false)` if at least one
    /// is not, or `None` if it cannot be decided.
    #[must_use]
    pub fn always_less_than(&self, value: i64) -> Option<bool> {
        match self {
            Self::Bottom => Some(true), // Vacuously true
            Self::Top => None,
            Self::Interval(i) => i.always_less_than(value),
            Self::Union(intervals) => {
                // All intervals must satisfy
                let results: Vec<_> = intervals
                    .iter()
                    .map(|i| i.always_less_than(value))
                    .collect();
                if results.iter().all(|r| *r == Some(true)) {
                    Some(true)
                } else if results.contains(&Some(false)) {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }

    /// Determines whether every value in the range is strictly greater than `value`.
    ///
    /// Mirror of [`always_less_than`](Self::always_less_than): `Bottom` is
    /// vacuously `true`, `Top` undecidable, an interval defers to
    /// [`IntervalRange::always_greater_than`], and a `Union` is `true` only when
    /// every member is, `false` if any member definitely is not, else `None`.
    ///
    /// # Arguments
    ///
    /// * `value` - The exclusive lower threshold each range value is tested against.
    ///
    /// # Returns
    ///
    /// `Some(true)` if all values are `> value`, `Some(false)` if at least one
    /// is not, or `None` if it cannot be decided.
    #[must_use]
    pub fn always_greater_than(&self, value: i64) -> Option<bool> {
        match self {
            Self::Bottom => Some(true),
            Self::Top => None,
            Self::Interval(i) => i.always_greater_than(value),
            Self::Union(intervals) => {
                let results: Vec<_> = intervals
                    .iter()
                    .map(|i| i.always_greater_than(value))
                    .collect();
                if results.iter().all(|r| *r == Some(true)) {
                    Some(true)
                } else if results.contains(&Some(false)) {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }

    /// Determines whether the range is exactly the singleton `value`.
    ///
    /// `Bottom` is vacuously `true`, and a single interval defers to
    /// [`IntervalRange::always_equal_to`]. Both `Top` and any `Union` return
    /// `None`: `Top` is unconstrained, and a `Union` holds several disjoint
    /// intervals so it can never be a single value, yet reporting `false` would
    /// be wrong when `value` is one of the values it contains.
    ///
    /// # Arguments
    ///
    /// * `value` - The value tested for exact equality against the whole range.
    ///
    /// # Returns
    ///
    /// `Some(true)` if the range is the singleton `value`, `Some(false)` if
    /// `value` is excluded, or `None` if it cannot be decided.
    #[must_use]
    pub fn always_equal_to(&self, value: i64) -> Option<bool> {
        match self {
            Self::Bottom => Some(true), // Vacuously true
            Self::Interval(i) => i.always_equal_to(value),
            // Top represents all possible values, Union has multiple disjoint intervals
            Self::Top | Self::Union(_) => None,
        }
    }

    /// Whether `condition` holds for flags set by an operation whose result
    /// lies in this range.
    ///
    /// **A flags value has to be able to say something about the value that set
    /// it.** A machine's conditional branch does not name the value it is really
    /// testing: it names a flags token, and the fact the analysis has - that the
    /// result is `[0, 0x3FFF0001]`, say - sits on the *other* variable, the
    /// operation's primary destination. This is the bridge, and it belongs on
    /// the range because that is the side that carries the fact; a caller finds
    /// the producing operation and hands its result range here.
    ///
    /// # Only the two flags the result's magnitude decides
    ///
    /// `Zero` and `Negative` are read off the result and nothing else, so an
    /// interval decides them:
    ///
    /// - `Zero` / `NotZero` - whether the range is the singleton zero.
    /// - `Negative` / `Positive` - which side of zero the whole range is on.
    ///   `Positive` is the sign bit *clear*, which includes zero; it is not the
    ///   strict `> 0` its name suggests, and reading it that way would make
    ///   `0 >= 0` decide the wrong way.
    ///
    /// The rest answer `None`, and not for want of trying. `Carry` and
    /// `Overflow` are facts about *how* the result was computed - whether the
    /// addition carried out of the width, whether it crossed the signed
    /// boundary - and two operations with identical result ranges differ in
    /// them. `ParityEven`/`ParityOdd` count the set bits of the low byte, which
    /// an interval pins only when it pins that byte; a range wide enough to be
    /// interesting never does, and answering from a singleton alone would be a
    /// rule that fires only where constant folding already has.
    ///
    /// # Arguments
    ///
    /// * `condition` - The flag test to decide.
    ///
    /// # Returns
    ///
    /// `Some(true)` or `Some(false)` when the range decides it, `None` when it
    /// does not - which includes every condition this cannot derive at all.
    #[must_use]
    pub fn satisfies_flag(&self, condition: FlagCondition) -> Option<bool> {
        match condition {
            FlagCondition::Zero => self.always_equal_to(0),
            FlagCondition::NotZero => self.always_equal_to(0).map(|zero| !zero),
            // `is_always_negative` and `is_always_non_negative` are each a
            // one-sided proof: neither holding means the range straddles zero
            // and the sign bit is genuinely unknown.
            FlagCondition::Negative => {
                if self.is_always_negative() {
                    Some(true)
                } else if self.is_always_non_negative() {
                    Some(false)
                } else {
                    None
                }
            }
            FlagCondition::Positive => {
                if self.is_always_non_negative() {
                    Some(true)
                } else if self.is_always_negative() {
                    Some(false)
                } else {
                    None
                }
            }
            FlagCondition::Carry
            | FlagCondition::NotCarry
            | FlagCondition::Overflow
            | FlagCondition::NotOverflow
            | FlagCondition::ParityEven
            | FlagCondition::ParityOdd => None,
        }
    }

    /// Returns `true` if `value` could be one of the range's values.
    ///
    /// `Bottom` contains nothing and `Top` contains everything. An interval
    /// tests its bounds, and a `Union` reports membership if any member interval
    /// contains `value` - so a value in a gap between members is correctly
    /// excluded.
    ///
    /// # Arguments
    ///
    /// * `value` - The candidate value to test for membership.
    ///
    /// # Returns
    ///
    /// `true` if some part of the range includes `value`.
    #[must_use]
    pub fn may_contain(&self, value: i64) -> bool {
        match self {
            Self::Bottom => false,
            Self::Top => true,
            Self::Interval(i) => i.may_contain(value),
            Self::Union(intervals) => intervals.iter().any(|i| i.may_contain(value)),
        }
    }

    /// Meet operation (intersection) - the greatest lower bound in the lattice.
    ///
    /// Returns the values in both `self` and `other`. `Bottom` absorbs and `Top`
    /// is the identity. Two intervals delegate to `IntervalRange::meet`,
    /// becoming `Bottom` when disjoint. Meeting against a `Union` intersects
    /// every member interval, and two unions take the pairwise intersection of
    /// the cross product; either way the surviving pieces are re-normalized so
    /// the result stays canonical.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to intersect with.
    ///
    /// # Returns
    ///
    /// The intersection range, or `Bottom` if the ranges are disjoint.
    #[must_use]
    pub fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            // Bottom absorbs
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,

            // Top is identity
            (Self::Top, x) | (x, Self::Top) => x.clone(),

            // Interval meet
            (Self::Interval(a), Self::Interval(b)) => match a.meet(b) {
                Some(result) => Self::Interval(result),
                None => Self::Bottom,
            },

            // Union meet: intersect each pair
            (Self::Union(intervals), Self::Interval(i))
            | (Self::Interval(i), Self::Union(intervals)) => {
                let results: Vec<_> = intervals.iter().filter_map(|ui| ui.meet(i)).collect();
                Self::from_intervals(results)
            }

            (Self::Union(a), Self::Union(b)) => {
                let mut results = Vec::new();
                for ai in a {
                    for bi in b {
                        if let Some(r) = ai.meet(bi) {
                            results.push(r);
                        }
                    }
                }
                Self::from_intervals(results)
            }
        }
    }

    /// Join operation (union hull) - the least upper bound in the lattice.
    ///
    /// Returns a range containing all values from both `self` and `other`.
    /// `Bottom` is the identity and `Top` absorbs. Two plain intervals take
    /// their hull, which may admit values in neither input; this differs from
    /// [`union`](Self::union), which would keep disjoint intervals apart. Cases
    /// involving a `Union` are flattened through `normalize_union`, merging only
    /// contiguous pieces.
    ///
    /// # Arguments
    ///
    /// * `other` - The range to join with.
    ///
    /// # Returns
    ///
    /// The smallest range in the lattice above both operands.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        match (self, other) {
            // Bottom is identity
            (Self::Bottom, x) | (x, Self::Bottom) => x.clone(),

            // Top absorbs
            (Self::Top, _) | (_, Self::Top) => Self::Top,

            // Interval join: take hull
            (Self::Interval(a), Self::Interval(b)) => Self::Interval(a.join(b)),

            // Union join
            (Self::Union(intervals), Self::Interval(i))
            | (Self::Interval(i), Self::Union(intervals)) => {
                let mut all = intervals.clone();
                all.push(*i);
                Self::normalize_union(all)
            }

            (Self::Union(a), Self::Union(b)) => {
                let mut all = a.clone();
                all.extend(b.iter().copied());
                Self::normalize_union(all)
            }
        }
    }

    /// Widen operation for loop fixpoint computation.
    ///
    /// Ensures the analysis terminates by extending any growing bound to
    /// infinity instead of tracking it iteration by iteration. `Bottom` is the
    /// identity and `Top` absorbs. Two intervals delegate to
    /// `IntervalRange::widen`; unions are first collapsed to their hulls (widening
    /// a growing union bound-by-bound need not converge) and then widened, giving
    /// up to `Top` if either hull is not a plain interval.
    ///
    /// # Arguments
    ///
    /// * `other` - The next iterate whose bound movement drives widening.
    ///
    /// # Returns
    ///
    /// A widened range guaranteed to stabilize the fixed-point sequence.
    #[must_use]
    pub fn widen(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, x) | (x, Self::Bottom) => x.clone(),
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Interval(a), Self::Interval(b)) => Self::Interval(a.widen(b)),
            // For unions, widen the hull
            _ => {
                let self_hull = self.hull();
                let other_hull = other.hull();
                match (self_hull, other_hull) {
                    (Self::Interval(a), Self::Interval(b)) => Self::Interval(a.widen(&b)),
                    _ => Self::Top,
                }
            }
        }
    }

    /// Collapses this range to the single interval spanning all its values.
    ///
    /// `Bottom` and `Top` are already single points of the lattice and pass
    /// through unchanged, as does a lone `Interval`. A `Union` is flattened to
    /// `[first.min, last.max]` - valid because normalization keeps its members
    /// sorted, so the first holds the lowest bound and the last the highest. The
    /// gaps between members are discarded, which is why callers use this only
    /// where a single interval is required (widening, binary ops).
    ///
    /// # Returns
    ///
    /// A range with at most one interval enclosing every value of `self`.
    #[must_use]
    pub fn hull(&self) -> Self {
        match self {
            Self::Bottom => Self::Bottom,
            Self::Top => Self::Top,
            Self::Interval(_) => self.clone(),
            Self::Union(intervals) => {
                if intervals.is_empty() {
                    Self::Bottom
                } else {
                    let min_val = intervals.first().and_then(|i| i.min);
                    let max_val = intervals.last().and_then(|i| i.max);
                    Self::Interval(IntervalRange::new(min_val, max_val))
                }
            }
        }
    }

    /// Addition of two ranges.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.binary_op(other, IntervalRange::add)
    }

    /// Subtraction of two ranges.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.binary_op(other, IntervalRange::sub)
    }

    /// Multiplication of two ranges.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        self.binary_op(other, IntervalRange::mul)
    }

    /// Bounds the result of a bitwise AND against a constant `mask`.
    ///
    /// A non-negative mask forces the result into `[0, mask]` no matter what
    /// `self` holds, so any non-`Bottom` range maps to that fixed bound; a
    /// negative mask (sign bit set) admits no such bound and yields `Top`.
    /// `Bottom` stays `Bottom` because there is no value to mask.
    ///
    /// # Arguments
    ///
    /// * `mask` - The constant AND mask.
    ///
    /// # Returns
    ///
    /// `[0, mask]` for a non-negative mask, `Top` for a negative one, or
    /// `Bottom` if `self` is `Bottom`.
    #[must_use]
    pub fn and_constant(&self, mask: i64) -> Self {
        match self {
            Self::Bottom => Self::Bottom,
            Self::Top | Self::Interval(_) | Self::Union(_) => {
                if mask >= 0 {
                    Self::bounded(0, mask)
                } else {
                    Self::Top
                }
            }
        }
    }

    /// Joins every range in `ranges` at once.
    ///
    /// The result contains every value of every input, exactly as a chain of
    /// [`Self::join`] calls would, but the pieces are gathered and normalised
    /// once instead of re-sorting a growing union at each step: a phi with
    /// thousands of operands costs one sort, not one per operand. Plain
    /// intervals take their hull, as a pairwise join does; once a `Union` is
    /// involved the gaps are kept, up to `MAX_JOINED_PIECES` pieces, and
    /// beyond that the hull is taken so the representation stays bounded.
    ///
    /// # Arguments
    ///
    /// * `ranges` - The ranges to join; an empty input yields `Bottom`.
    ///
    /// # Returns
    ///
    /// A range containing every value of every input.
    #[must_use]
    pub fn join_all<'a>(ranges: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut pieces: Vec<IntervalRange> = Vec::new();
        let mut saw_union = false;
        for range in ranges {
            match range {
                Self::Bottom => {}
                Self::Top => return Self::Top,
                Self::Interval(interval) => pieces.push(*interval),
                Self::Union(intervals) => {
                    saw_union = true;
                    pieces.extend(intervals.iter().copied());
                }
            }
        }
        let Some(first) = pieces.first().copied() else {
            return Self::Bottom;
        };
        if !saw_union || pieces.len() > Self::MAX_JOINED_PIECES {
            let hull = pieces
                .iter()
                .skip(1)
                .fold(first, |acc, piece| acc.join(piece));
            return Self::Interval(hull);
        }
        Self::normalize_union(pieces)
    }

    /// The most intervals [`Self::join_all`] keeps apart before it settles for
    /// their hull.
    const MAX_JOINED_PIECES: usize = 64;

    /// Reduces every value of this range to a `width`-bit machine value.
    ///
    /// Interval arithmetic here runs on mathematical integers (an `i64` bound,
    /// or infinity where a bound overflowed), while the machine keeps only the
    /// low `width` bits. This maps each value to the one its low bits denote,
    /// read in the window a `width`-bit value lives in: `[-2^(w-1), 2^(w-1)-1]`
    /// when `signed`, `[0, 2^w - 1]` otherwise. Two values that differ by a
    /// multiple of `2^w` are the same machine value, so this is also how one
    /// variable's range is read under another signedness.
    ///
    /// A 64-bit value is always held in the signed window, whatever `signed`
    /// says: an `i64` bound cannot name the upper half of the unsigned one. The
    /// caller reads such a range as its bits, and an unsigned question about it
    /// is only answered where the range is non-negative (where both readings
    /// agree). A 64-bit window that the range fills, or that an infinite bound
    /// escapes, is [`Self::Top`]: "every 64-bit value" and "nothing is known"
    /// are the same statement.
    ///
    /// An interval that crosses one window boundary splits into two pieces; one
    /// that spans more than a window, or has an infinite bound, is the whole
    /// window.
    ///
    /// # Arguments
    ///
    /// * `width` - The machine width in bits, `1..=64`; anything else is `Top`.
    /// * `signed` - Whether the result is read in the signed window.
    ///
    /// # Returns
    ///
    /// The range of machine values, inside the chosen window.
    #[must_use]
    pub fn wrap(&self, width: u32, signed: bool) -> Self {
        if width == 0 || width > 64 {
            return Self::Top;
        }
        match self {
            Self::Bottom => Self::Bottom,
            Self::Top => Self::Top,
            Self::Interval(interval) => {
                let wrapped = Self::wrap_interval(*interval, width, signed);
                if wrapped.is_empty() {
                    Self::Top
                } else {
                    Self::from_intervals(wrapped)
                }
            }
            Self::Union(intervals) => {
                let mut pieces = Vec::new();
                for interval in intervals {
                    let wrapped = Self::wrap_interval(*interval, width, signed);
                    if wrapped.is_empty() {
                        return Self::Top;
                    }
                    pieces.extend(wrapped);
                }
                Self::from_intervals(pieces)
            }
        }
    }

    /// The pieces [`Self::wrap`] maps one interval to; empty for a 64-bit
    /// value that is unconstrained.
    fn wrap_interval(interval: IntervalRange, width: u32, signed: bool) -> Vec<IntervalRange> {
        Self::try_wrap_interval(interval, width, signed).unwrap_or_else(|| {
            Self::window_bounds(width, signed)
                .map(|(low, high)| vec![IntervalRange::bounded(low, high)])
                .unwrap_or_default()
        })
    }

    /// Every value of a `width`-bit integer read in the `signed` window:
    /// [`Self::Top`] for a width of 64, for which every `i64` is one, and for
    /// a width of zero or above 64, which has no window.
    #[must_use]
    pub fn window(width: u32, signed: bool) -> Self {
        Self::window_bounds(width, signed).map_or(Self::Top, |(low, high)| Self::bounded(low, high))
    }

    /// `self` read as a `width`-bit value in the `signed` window, where an
    /// unknown value still lies in that window.
    ///
    /// [`Self::wrap`] keeps `Top` as `Top`; a conversion wants the window
    /// instead, because extending an unknown byte still gives a value the
    /// byte's width bounds.
    #[must_use]
    pub fn read_bounded(&self, width: u32, signed: bool) -> Self {
        if self.is_top() {
            Self::window(width, signed)
        } else {
            self.wrap(width, signed)
        }
    }

    /// The bounds of the `width`-bit window [`Self::wrap`] reads values in, or
    /// `None` for the 64-bit one (which is [`Self::Top`]) and for an invalid
    /// width.
    fn window_bounds(width: u32, signed: bool) -> Option<(i64, i64)> {
        if width == 0 || width >= 64 {
            return None;
        }
        // In 128 bits: `1i64 << 63` is `i64::MIN`, not the modulus of a
        // 63-bit integer.
        let modulus = 1i128.checked_shl(width)?;
        let (low, high) = if signed {
            let half = modulus.checked_div(2)?;
            (half.checked_neg()?, half.checked_sub(1)?)
        } else {
            (0, modulus.checked_sub(1)?)
        };
        Some((i64::try_from(low).ok()?, i64::try_from(high).ok()?))
    }

    /// [`Self::wrap_interval`] for an interval with two finite bounds that
    /// spans at most one window boundary; `None` when the answer is the whole
    /// window.
    fn try_wrap_interval(
        interval: IntervalRange,
        width: u32,
        signed: bool,
    ) -> Option<Vec<IntervalRange>> {
        let (min, max) = (interval.min?, interval.max?);
        if width == 64 {
            // Every finite `i64` already is a 64-bit value in the signed window.
            return Some(vec![interval]);
        }
        let (low, high) = Self::window_bounds(width, signed)?;
        if min >= low && max <= high {
            return Some(vec![interval]);
        }
        let modulus = 1i128.checked_shl(width)?;
        let (min, max, low_wide) = (i128::from(min), i128::from(max), i128::from(low));
        let period = |value: i128| value.checked_sub(low_wide)?.checked_div_euclid(modulus);
        let (min_period, max_period) = (period(min)?, period(max)?);
        let shifted = |value: i128, period: i128| {
            i64::try_from(value.checked_sub(period.checked_mul(modulus)?)?).ok()
        };
        let (shifted_min, shifted_max) = (shifted(min, min_period)?, shifted(max, max_period)?);
        if min_period == max_period {
            Some(vec![IntervalRange::bounded(shifted_min, shifted_max)])
        } else if max_period == min_period.checked_add(1)? {
            Some(vec![
                IntervalRange::bounded(shifted_min, high),
                IntervalRange::bounded(low, shifted_max),
            ])
        } else {
            None
        }
    }

    /// Lifts a per-interval arithmetic operation to whole ranges.
    ///
    /// Shared backbone for [`add`](Self::add), [`sub`](Self::sub), and
    /// [`mul`](Self::mul): `Bottom` and `Top` short-circuit (an operation on an
    /// unknown operand is unknown), two intervals apply `op` directly, and any
    /// `Union` operand is reduced to its hull before `op` runs. Working on hulls
    /// is a deliberate precision-for-simplicity trade: still sound, since the
    /// hull contains every member, just coarser than operating per interval.
    ///
    /// # Arguments
    ///
    /// * `other` - The right-hand range.
    /// * `op` - The interval-level operation to apply once both sides are single intervals.
    ///
    /// # Returns
    ///
    /// The range produced by applying `op`, or `Top` when an operand cannot be
    /// reduced to a single interval.
    fn binary_op<F>(&self, other: &Self, op: F) -> Self
    where
        F: Fn(&IntervalRange, &IntervalRange) -> IntervalRange,
    {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Interval(a), Self::Interval(b)) => Self::Interval(op(a, b)),
            // For unions, operate on hulls (loses precision but correct)
            _ => {
                let a_hull = self.hull();
                let b_hull = other.hull();
                match (a_hull, b_hull) {
                    (Self::Interval(a), Self::Interval(b)) => Self::Interval(op(&a, &b)),
                    _ => Self::Top,
                }
            }
        }
    }

    /// Builds a range from intersection results, mapping the empty set to `Bottom`.
    ///
    /// The meet paths collect surviving intersections into a `Vec` that may be
    /// empty when every pair was disjoint; this treats that empty case as
    /// `Bottom` and otherwise hands the intervals to `normalize_union` for
    /// sorting and merging.
    ///
    /// # Arguments
    ///
    /// * `intervals` - The intervals to assemble; empty means an empty intersection.
    ///
    /// # Returns
    ///
    /// `Bottom` if `intervals` is empty, otherwise the normalized range.
    fn from_intervals(intervals: Vec<IntervalRange>) -> Self {
        if intervals.is_empty() {
            Self::Bottom
        } else {
            Self::normalize_union(intervals)
        }
    }
}

impl fmt::Debug for ValueRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bottom => write!(f, "⊥"),
            Self::Top => write!(f, "⊤"),
            Self::Interval(i) => write!(f, "{i}"),
            Self::Union(intervals) => {
                write!(f, "(")?;
                for (i, interval) in intervals.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ∪ ")?;
                    }
                    write!(f, "{interval}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for ValueRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two flags a result's magnitude decides, and the six it does not.
    ///
    /// `Positive` is the sign bit *clear*, so it includes zero. Reading it as
    /// the strict `> 0` its name suggests would decide `[0, 0]` the wrong way,
    /// and `[0, n]` is the shape every masked value has.
    #[test]
    fn a_range_decides_the_zero_and_sign_flags_and_no_others() {
        let non_negative = ValueRange::bounded(0, 1_073_676_289);
        assert_eq!(
            non_negative.satisfies_flag(FlagCondition::Positive),
            Some(true)
        );
        assert_eq!(
            non_negative.satisfies_flag(FlagCondition::Negative),
            Some(false)
        );
        assert_eq!(non_negative.satisfies_flag(FlagCondition::Zero), None);

        let zero = ValueRange::constant(0);
        assert_eq!(zero.satisfies_flag(FlagCondition::Zero), Some(true));
        assert_eq!(zero.satisfies_flag(FlagCondition::NotZero), Some(false));
        assert_eq!(
            zero.satisfies_flag(FlagCondition::Positive),
            Some(true),
            "zero has its sign bit clear"
        );

        let negative = ValueRange::bounded(-10, -1);
        assert_eq!(negative.satisfies_flag(FlagCondition::Negative), Some(true));
        assert_eq!(
            negative.satisfies_flag(FlagCondition::Positive),
            Some(false)
        );
        assert_eq!(negative.satisfies_flag(FlagCondition::NotZero), Some(true));

        let straddling = ValueRange::bounded(-1, 1);
        assert_eq!(straddling.satisfies_flag(FlagCondition::Negative), None);
        assert_eq!(straddling.satisfies_flag(FlagCondition::Positive), None);
        assert_eq!(straddling.satisfies_flag(FlagCondition::Zero), None);

        // Carry, overflow and parity are facts about *how* the result was
        // computed, not about where it lies: two operations with identical
        // result ranges differ in them.
        for condition in [
            FlagCondition::Carry,
            FlagCondition::NotCarry,
            FlagCondition::Overflow,
            FlagCondition::NotOverflow,
            FlagCondition::ParityEven,
            FlagCondition::ParityOdd,
        ] {
            assert_eq!(
                non_negative.satisfies_flag(condition),
                None,
                "{condition} is not a function of the result's range"
            );
        }
    }

    /// `is_always_negative` is the partner of `is_always_non_negative`, and
    /// neither holding is what "straddles zero" looks like.
    #[test]
    fn a_range_is_negative_non_negative_or_neither() {
        assert!(ValueRange::bounded(-10, -1).is_always_negative());
        assert!(!ValueRange::bounded(-10, 0).is_always_negative());
        assert!(!ValueRange::bounded(0, 10).is_always_negative());
        assert!(!ValueRange::top().is_always_negative());
        let straddling = ValueRange::bounded(-1, 1);
        assert!(!straddling.is_always_negative() && !straddling.is_always_non_negative());
    }

    /// A constant interval reports itself as a singleton with matching bounds.
    #[test]
    fn test_interval_constant() {
        let r = IntervalRange::constant(42);
        assert!(r.is_constant());
        assert_eq!(r.as_constant(), Some(42));
        assert_eq!(r.min, Some(42));
        assert_eq!(r.max, Some(42));
    }

    /// A bounded interval is not constant and reports membership at and beyond its edges.
    #[test]
    fn test_interval_bounded() {
        let r = IntervalRange::bounded(0, 255);
        assert!(!r.is_constant());
        assert_eq!(r.as_constant(), None);
        assert!(r.is_always_non_negative());
        assert!(!r.is_always_positive());
        assert!(r.may_contain(0));
        assert!(r.may_contain(255));
        assert!(!r.may_contain(256));
        assert!(!r.may_contain(-1));
    }

    /// A non-negative half-open interval is non-negative but not strictly positive.
    #[test]
    fn test_interval_non_negative() {
        let r = IntervalRange::non_negative();
        assert!(r.is_always_non_negative());
        assert!(!r.is_always_positive());
        assert_eq!(r.always_less_than(0), Some(false));
        assert_eq!(r.always_greater_equal(0), Some(true));
    }

    /// The tri-state comparison queries agree with the "all values" semantics, including undecidable unbounded cases.
    #[test]
    fn test_interval_comparisons() {
        let r = IntervalRange::bounded(5, 10);

        // always_less_than: "are ALL values in range < X?"
        assert_eq!(r.always_less_than(11), Some(true)); // All of [5,10] < 11
        assert_eq!(r.always_less_than(10), Some(false)); // 10 is not < 10
        assert_eq!(r.always_less_than(5), Some(false)); // 5,6,7,8,9,10 are not all < 5
        assert_eq!(r.always_less_than(8), Some(false)); // 8,9,10 are not < 8

        // always_greater_than: "are ALL values in range > X?"
        assert_eq!(r.always_greater_than(4), Some(true)); // All of [5,10] > 4
        assert_eq!(r.always_greater_than(5), Some(false)); // 5 is not > 5
        assert_eq!(r.always_greater_than(10), Some(false)); // 5,6,7,8,9 are not > 10
        assert_eq!(r.always_greater_than(7), Some(false)); // 5,6,7 are not > 7

        // always_equal_to
        let c = IntervalRange::constant(5);
        assert_eq!(c.always_equal_to(5), Some(true));
        assert_eq!(c.always_equal_to(6), Some(false));
        assert_eq!(r.always_equal_to(5), None); // Range [5,10] might equal 5 (but not always)

        // Unbounded ranges
        let unbounded = IntervalRange::full();
        assert_eq!(unbounded.always_less_than(0), None); // Cannot determine
        assert_eq!(unbounded.always_greater_than(0), None);
    }

    /// Meet intersects overlapping intervals and yields `None` for disjoint ones.
    #[test]
    fn test_interval_meet() {
        let a = IntervalRange::bounded(0, 10);
        let b = IntervalRange::bounded(5, 15);
        let meet = a.meet(&b).unwrap();
        assert_eq!(meet.min, Some(5));
        assert_eq!(meet.max, Some(10));

        // Disjoint intervals
        let c = IntervalRange::bounded(0, 5);
        let d = IntervalRange::bounded(10, 15);
        assert!(c.meet(&d).is_none());
    }

    /// Join takes the hull spanning both intervals' extremes.
    #[test]
    fn test_interval_join() {
        let a = IntervalRange::bounded(0, 10);
        let b = IntervalRange::bounded(5, 15);
        let join = a.join(&b);
        assert_eq!(join.min, Some(0));
        assert_eq!(join.max, Some(15));
    }

    /// Widening relaxes a bound that grows outward to the corresponding infinity.
    /// Widening toward an unbounded bound must reach infinity, not keep the
    /// finite one.
    ///
    /// `None` encodes infinity, so `other` being unbounded means the value grew
    /// past every finite bound. Matching only `(Some, Some)` let this case fall
    /// through to "keep current", returning a range *narrower* than the join -
    /// the unsound direction, and the one a consumer folds a live branch on.
    #[test]
    fn widen_toward_an_unbounded_bound_reaches_infinity() {
        let bounded = IntervalRange::new(Some(0), Some(0));
        let unbounded_above = IntervalRange::new(Some(0), None);
        let widened = bounded.widen(&unbounded_above);
        assert_eq!(widened.min, Some(0));
        assert_eq!(
            widened.max, None,
            "[0,0] widened toward [0,+inf) must lose its upper bound"
        );

        let unbounded_below = IntervalRange::new(None, Some(0));
        let widened = bounded.widen(&unbounded_below);
        assert_eq!(
            widened.min, None,
            "[0,0] widened toward (-inf,0] must lose its lower bound"
        );
        assert_eq!(widened.max, Some(0));
    }

    /// The invariant that makes widening sound: it must over-approximate the
    /// join for every combination of bounded and unbounded endpoints.
    ///
    /// Stated as a law over a matrix rather than as one example, because the
    /// defect this guards lived only in the combinations the single-example
    /// tests never reached.
    #[test]
    fn widen_over_approximates_join() {
        let samples = [
            ValueRange::Interval(IntervalRange::new(Some(0), Some(0))),
            ValueRange::Interval(IntervalRange::new(Some(0), Some(1))),
            ValueRange::Interval(IntervalRange::new(Some(0), None)),
            ValueRange::Interval(IntervalRange::new(None, Some(0))),
            ValueRange::Interval(IntervalRange::new(Some(-5), Some(5))),
            ValueRange::Bottom,
            ValueRange::Top,
        ];
        for a in &samples {
            for b in &samples {
                let widened = a.widen(b);
                let joined = a.join(b);
                assert_eq!(
                    widened.join(&joined),
                    widened,
                    "widen({a:?}, {b:?}) = {widened:?} must contain join = {joined:?}"
                );
            }
        }
    }

    #[test]
    fn test_interval_widen() {
        let a = IntervalRange::bounded(0, 10);
        let b = IntervalRange::bounded(0, 20); // Growing up
        let widen = a.widen(&b);
        assert_eq!(widen.min, Some(0));
        assert_eq!(widen.max, None); // Widened to +∞

        let c = IntervalRange::bounded(5, 10);
        let d = IntervalRange::bounded(0, 10); // Growing down
        let widen2 = c.widen(&d);
        assert_eq!(widen2.min, None); // Widened to -∞
        assert_eq!(widen2.max, Some(10));
    }

    /// Add, sub, and mul on positive intervals produce the expected bounds.
    #[test]
    fn test_interval_arithmetic() {
        let a = IntervalRange::bounded(1, 5);
        let b = IntervalRange::bounded(2, 3);

        // Add: [1,5] + [2,3] = [3,8]
        let sum = a.add(&b);
        assert_eq!(sum.min, Some(3));
        assert_eq!(sum.max, Some(8));

        // Sub: [1,5] - [2,3] = [1-3, 5-2] = [-2, 3]
        let diff = a.sub(&b);
        assert_eq!(diff.min, Some(-2));
        assert_eq!(diff.max, Some(3));

        // Mul: [1,5] * [2,3] = [2, 15]
        let prod = a.mul(&b);
        assert_eq!(prod.min, Some(2));
        assert_eq!(prod.max, Some(15));
    }

    /// Multiplication across a sign change picks the extreme corner products.
    #[test]
    fn test_interval_mul_with_negatives() {
        let a = IntervalRange::bounded(-2, 3);
        let b = IntervalRange::bounded(-1, 4);

        // Products: (-2)*(-1)=2, (-2)*4=-8, 3*(-1)=-3, 3*4=12
        // Range: [-8, 12]
        let prod = a.mul(&b);
        assert_eq!(prod.min, Some(-8));
        assert_eq!(prod.max, Some(12));
    }

    /// AND with a byte mask bounds the interval to `[0, 255]`.
    #[test]
    fn test_interval_and_constant() {
        let r = IntervalRange::bounded(0, 1000);
        let masked = r.and_constant(0xFF);
        assert_eq!(masked.min, Some(0));
        assert_eq!(masked.max, Some(255));
    }

    /// A constant `ValueRange` is a singleton that is neither bottom nor top.
    #[test]
    fn test_range_constant() {
        let r = ValueRange::constant(42);
        assert!(r.is_constant());
        assert_eq!(r.as_constant(), Some(42));
        assert!(!r.is_bottom());
        assert!(!r.is_top());
    }

    /// A bounded `ValueRange` exposes its non-negativity and its min/max.
    #[test]
    fn test_range_bounded() {
        let r = ValueRange::bounded(0, 255);
        assert!(!r.is_constant());
        assert!(r.is_always_non_negative());
        assert_eq!(r.min(), Some(0));
        assert_eq!(r.max(), Some(255));
    }

    /// Constructing a bounded range with `min > max` collapses to `Bottom`.
    #[test]
    fn test_range_invalid_bounded() {
        let r = ValueRange::bounded(10, 5); // min > max
        assert!(r.is_bottom());
    }

    /// Meet honors the lattice laws: `Top` identity, `Bottom` absorbing, disjoint to `Bottom`.
    #[test]
    fn test_range_lattice_meet() {
        // Top is identity
        let a = ValueRange::bounded(0, 10);
        assert_eq!(a.meet(&ValueRange::top()), a);
        assert_eq!(ValueRange::top().meet(&a), a);

        // Bottom absorbs
        assert!(a.meet(&ValueRange::bottom()).is_bottom());
        assert!(ValueRange::bottom().meet(&a).is_bottom());

        // Interval intersection
        let b = ValueRange::bounded(5, 15);
        let meet = a.meet(&b);
        assert_eq!(meet.min(), Some(5));
        assert_eq!(meet.max(), Some(10));

        // Disjoint -> bottom
        let c = ValueRange::bounded(20, 30);
        assert!(a.meet(&c).is_bottom());
    }

    /// Join honors the lattice laws: `Bottom` identity, `Top` absorbing, hull otherwise.
    #[test]
    fn test_range_lattice_join() {
        // Bottom is identity
        let a = ValueRange::bounded(0, 10);
        assert_eq!(a.join(&ValueRange::bottom()), a);
        assert_eq!(ValueRange::bottom().join(&a), a);

        // Top absorbs
        assert!(a.join(&ValueRange::top()).is_top());
        assert!(ValueRange::top().join(&a).is_top());

        // Interval hull
        let b = ValueRange::bounded(5, 15);
        let join = a.join(&b);
        assert_eq!(join.min(), Some(0));
        assert_eq!(join.max(), Some(15));
    }

    /// Union keeps disjoint intervals apart but merges adjacent and overlapping ones.
    #[test]
    fn test_range_union() {
        let a = ValueRange::bounded(0, 5);
        let b = ValueRange::bounded(10, 15);
        let union = ValueRange::union(a, b);

        // Should be a union, not merged
        assert!(matches!(union, ValueRange::Union(_)));
        assert!(union.may_contain(3));
        assert!(union.may_contain(12));
        assert!(!union.may_contain(7)); // Gap

        // Adjacent ranges merge
        let c = ValueRange::bounded(0, 5);
        let d = ValueRange::bounded(6, 10);
        let merged = ValueRange::union(c, d);
        assert!(matches!(merged, ValueRange::Interval(_)));
        assert_eq!(merged.min(), Some(0));
        assert_eq!(merged.max(), Some(10));

        // Overlapping ranges merge
        let e = ValueRange::bounded(0, 7);
        let f = ValueRange::bounded(5, 10);
        let merged2 = ValueRange::union(e, f);
        assert!(matches!(merged2, ValueRange::Interval(_)));
        assert_eq!(merged2.min(), Some(0));
        assert_eq!(merged2.max(), Some(10));
    }

    /// Widening a `ValueRange` relaxes the growing bound to infinity.
    #[test]
    fn test_range_widen() {
        let a = ValueRange::bounded(0, 10);
        let b = ValueRange::bounded(0, 20); // Growing up
        let widen = a.widen(&b);
        assert_eq!(widen.min(), Some(0));
        assert_eq!(widen.max(), None); // Widened to +∞

        let c = ValueRange::bounded(5, 10);
        let d = ValueRange::bounded(0, 10); // Growing down
        let widen2 = c.widen(&d);
        assert_eq!(widen2.min(), None); // Widened to -∞
        assert_eq!(widen2.max(), Some(10));
    }

    /// Arithmetic on `ValueRange` intervals matches the interval-level results.
    #[test]
    fn test_range_arithmetic() {
        let a = ValueRange::bounded(1, 5);
        let b = ValueRange::bounded(2, 3);

        let sum = a.add(&b);
        assert_eq!(sum.min(), Some(3));
        assert_eq!(sum.max(), Some(8));

        let diff = a.sub(&b);
        assert_eq!(diff.min(), Some(-2));
        assert_eq!(diff.max(), Some(3));

        let prod = a.mul(&b);
        assert_eq!(prod.min(), Some(2));
        assert_eq!(prod.max(), Some(15));
    }

    /// AND with a mask bounds any range, including `Top`, to `[0, mask]`.
    #[test]
    fn test_range_and_constant_mask() {
        let r = ValueRange::bounded(0, 1000);
        let masked = r.and_constant(0xFF);
        assert_eq!(masked.min(), Some(0));
        assert_eq!(masked.max(), Some(255));

        // Top with mask
        let top_masked = ValueRange::top().and_constant(0x0F);
        assert_eq!(top_masked.min(), Some(0));
        assert_eq!(top_masked.max(), Some(15));
    }

    /// The `ValueRange` comparison queries reproduce the interval tri-state answers.
    #[test]
    fn test_range_comparison_queries() {
        let r = ValueRange::bounded(5, 10);

        // always_less_than: "are ALL values in range < X?"
        assert_eq!(r.always_less_than(11), Some(true)); // All of [5,10] < 11
        assert_eq!(r.always_less_than(5), Some(false)); // 5..10 are not all < 5
        assert_eq!(r.always_less_than(8), Some(false)); // 8,9,10 are not < 8

        // always_greater_than: "are ALL values in range > X?"
        assert_eq!(r.always_greater_than(4), Some(true)); // All of [5,10] > 4
        assert_eq!(r.always_greater_than(10), Some(false)); // 5..10 are not all > 10
        assert_eq!(r.always_greater_than(7), Some(false)); // 5,6,7 are not > 7
    }

    /// Non-negativity holds for non-negative constructions and is vacuously true for `Bottom`.
    #[test]
    fn test_range_always_non_negative() {
        assert!(ValueRange::non_negative().is_always_non_negative());
        assert!(ValueRange::bounded(0, 100).is_always_non_negative());
        assert!(ValueRange::at_least(5).is_always_non_negative());
        assert!(!ValueRange::bounded(-5, 100).is_always_non_negative());
        assert!(!ValueRange::top().is_always_non_negative());
        assert!(ValueRange::bottom().is_always_non_negative()); // Vacuously true
    }

    /// Display renders each range shape with its expected symbolic notation.
    #[test]
    fn test_range_display() {
        assert_eq!(format!("{}", ValueRange::bottom()), "⊥");
        assert_eq!(format!("{}", ValueRange::top()), "⊤");
        assert_eq!(format!("{}", ValueRange::constant(42)), "42");
        assert_eq!(format!("{}", ValueRange::bounded(0, 10)), "[0, 10]");
        assert_eq!(format!("{}", ValueRange::non_negative()), "[0, +∞)");
        assert_eq!(format!("{}", ValueRange::at_most(10)), "(-∞, 10]");
    }

    /// A disjoint union renders with the union symbol between its member intervals.
    #[test]
    fn test_range_union_display() {
        let union = ValueRange::union(ValueRange::bounded(0, 5), ValueRange::bounded(10, 15));
        let s = format!("{union}");
        assert!(s.contains("∪"));
        assert!(s.contains("[0, 5]"));
        assert!(s.contains("[10, 15]"));
    }

    /// The window of a width holds exactly the integers of that width, and a
    /// width with no window (none, 64 or more bits) is every value.
    #[test]
    fn a_window_holds_exactly_the_values_of_its_width() {
        for width in 1..64 {
            // Oracle: the extremes of a `width`-bit two's-complement integer.
            let unsigned_max = i64::try_from((1u128 << width) - 1).unwrap();
            let signed_min = -i64::try_from(1u128 << (width - 1)).unwrap();
            let signed_max = i64::try_from((1u128 << (width - 1)) - 1).unwrap();
            assert_eq!(
                ValueRange::window(width, false),
                ValueRange::bounded(0, unsigned_max),
                "unsigned {width}"
            );
            assert_eq!(
                ValueRange::window(width, true),
                ValueRange::bounded(signed_min, signed_max),
                "signed {width}"
            );
        }
        for width in [0, 64, 65, 128] {
            for signed in [false, true] {
                assert_eq!(
                    ValueRange::window(width, signed),
                    ValueRange::Top,
                    "{width} {signed}"
                );
            }
        }
        // The widest window that is not every `i64`: a 63-bit value wraps
        // into it like any narrower one.
        assert_eq!(
            ValueRange::constant(5).wrap(63, true),
            ValueRange::constant(5)
        );
        assert_eq!(
            ValueRange::constant(1 << 62).wrap(63, true),
            ValueRange::constant(-(1 << 62))
        );
        // An unknown value read at a width lies in its window; a known one is
        // wrapped into it.
        assert_eq!(
            ValueRange::Top.read_bounded(8, true),
            ValueRange::bounded(-128, 127)
        );
        assert_eq!(
            ValueRange::constant(200).read_bounded(8, true),
            ValueRange::constant(-56)
        );
    }

    /// A value is reduced to the machine value its low bits denote, in the
    /// window of the width and signedness asked for.
    #[test]
    fn wrap_reduces_to_the_window() {
        let byte = |range: ValueRange, signed| range.wrap(8, signed);
        assert_eq!(
            byte(ValueRange::constant(200), true),
            ValueRange::constant(-56)
        );
        assert_eq!(
            byte(ValueRange::constant(-56), false),
            ValueRange::constant(200)
        );
        assert_eq!(
            byte(ValueRange::constant(256), false),
            ValueRange::constant(0)
        );
        // Inside the window: unchanged.
        let inside = ValueRange::bounded(-3, 90);
        assert_eq!(byte(inside.clone(), true), inside);
        // One boundary crossed: the two pieces either side, and not the gap.
        let crossing = byte(ValueRange::bounded(250, 260), false);
        assert!(crossing.may_contain(255) && crossing.may_contain(0) && crossing.may_contain(4));
        assert!(!crossing.may_contain(100), "{crossing:?}");
        // More than a window, or an infinite bound: the whole window.
        assert_eq!(
            byte(ValueRange::bounded(0, 1_000), true),
            ValueRange::bounded(-128, 127)
        );
        assert_eq!(
            byte(ValueRange::non_negative(), false),
            ValueRange::bounded(0, 255)
        );
        // At 64 bits a finite bound is already a machine value and an infinite
        // one wrapped somewhere unknown.
        let finite = ValueRange::bounded(i64::MIN, 0);
        assert_eq!(finite.wrap(64, false), finite);
        assert!(ValueRange::at_least(1).wrap(64, true).is_top());
        assert!(ValueRange::top().wrap(8, true).is_top());
        assert!(ValueRange::bottom().wrap(8, true).is_bottom());
    }

    /// Joining many ranges at once contains every input, as a chain of joins
    /// does, and keeps the gaps between union members.
    #[test]
    fn join_all_contains_every_input() {
        let ranges = [
            ValueRange::constant(1),
            ValueRange::union(ValueRange::constant(10), ValueRange::constant(20)),
            ValueRange::constant(5),
            ValueRange::bottom(),
        ];
        let joined = ValueRange::join_all(&ranges);
        for value in [1, 5, 10, 20] {
            assert!(joined.may_contain(value), "{value} in {joined:?}");
        }
        assert!(
            !joined.may_contain(15),
            "the union's gap is kept: {joined:?}"
        );
        // Plain intervals take their hull, as a pairwise join does.
        let plain = [ValueRange::constant(1), ValueRange::constant(9)];
        assert_eq!(ValueRange::join_all(&plain), ValueRange::bounded(1, 9));
        assert!(ValueRange::join_all(&[ValueRange::constant(1), ValueRange::top()]).is_top());
        assert!(ValueRange::join_all(&[]).is_bottom());
        // Past the cap, the pieces collapse to their hull.
        let many: Vec<ValueRange> = (0..=i64::try_from(ValueRange::MAX_JOINED_PIECES)
            .unwrap_or(64))
            .map(|value| ValueRange::union(ValueRange::constant(value * 3), ValueRange::bottom()))
            .chain([ValueRange::Union(vec![
                IntervalRange::constant(-10),
                IntervalRange::constant(-5),
            ])])
            .collect();
        let capped = ValueRange::join_all(&many);
        assert!(matches!(capped, ValueRange::Interval(_)), "{capped:?}");
        assert!(capped.may_contain(-10) && capped.may_contain(192));
    }
}
