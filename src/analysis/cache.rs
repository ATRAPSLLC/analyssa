//! [`FunctionAnalyses`] — the derived analyses of one SSA function, each
//! computed at most once.
//!
//! # Why this exists
//!
//! Several analyses in [`crate::analysis`] are pure functions of a function's
//! SSA: run them twice on the same IR and they return the same thing. The
//! pipeline does exactly that — a driver that re-recovers a function's types
//! because its *seeds* moved re-derives the points-to relation and the alias
//! keys as well, even though the IR it derived them from cannot have changed.
//!
//! # Validity
//!
//! A derived analysis is valid exactly as long as the IR it was derived from is
//! unchanged, and nothing about a `(key, analysis)` pair records that. So the
//! memo **borrows** the function it describes and hands that same borrow back
//! through [`FunctionAnalyses::ir`]: while a handle is alive no `&mut
//! SsaFunction` can exist, so a cached analysis cannot outlive the IR it
//! describes and a consumer cannot pair one function's relation with another's
//! IR. Misuse does not compile rather than being caught by review.
//!
//! The consequence is that a handle cannot span a mutation. Code that rewrites
//! the IR between two analyses builds a second handle afterwards, and that is
//! the honest outcome: those two analyses genuinely describe different
//! programs.
//!
//! # Laziness
//!
//! Nothing is computed until asked for. This is load-bearing, not an
//! optimization: callers exist that return early without ever needing an
//! analysis, and an eager handle would do work the current code skips.

use std::{collections::BTreeMap, sync::OnceLock};

use crate::{
    analysis::{
        address::{AliasKey, alias_keys_for_function},
        pointsto::{PointsTo, analyze_function},
    },
    ir::function::SsaFunction,
    pointer::PointerSize,
    target::Target,
};

/// The derived analyses of one SSA function, computed on demand and reused.
///
/// Each analysis is a separate slot, so a consumer that needs only one pays
/// only for that one. Slots are [`OnceLock`]s rather than a lock around a map:
/// accessors take `&self`, the type stays `Sync`, and once a slot is filled
/// reads are uncontended.
///
/// # Examples
///
/// ```ignore
/// let analyses = FunctionAnalyses::new(&function.ir, function.ptr_size());
/// let relation = analyses.points_to();   // solves here
/// let again = analyses.points_to();      // same relation, no second solve
/// ```
#[derive(Debug)]
pub struct FunctionAnalyses<'ir, T: Target> {
    /// The function every slot below describes.
    ir: &'ir SsaFunction<T>,
    /// Pointer width, for the analyses that need the host's address size.
    ptr_size: PointerSize,
    /// Inclusion-based points-to relation ([`analyze_function`]).
    points_to: OnceLock<PointsTo>,
    /// Cell-identity projection of the function's addresses
    /// ([`alias_keys_for_function`]).
    alias_keys: OnceLock<BTreeMap<u32, AliasKey>>,
}

impl<'ir, T: Target> FunctionAnalyses<'ir, T> {
    /// Builds an empty set of analyses over `ir`.
    ///
    /// Nothing is computed here; each accessor computes its own slot on first
    /// use.
    #[must_use]
    pub const fn new(ir: &'ir SsaFunction<T>, ptr_size: PointerSize) -> Self {
        Self {
            ir,
            ptr_size,
            points_to: OnceLock::new(),
            alias_keys: OnceLock::new(),
        }
    }

    /// Returns the function these analyses describe.
    ///
    /// Consumers should read the IR through this rather than carrying their own
    /// reference alongside the handle — that is what makes a mismatched pair
    /// unrepresentable.
    #[must_use]
    pub const fn ir(&self) -> &'ir SsaFunction<T> {
        self.ir
    }

    /// Returns the pointer width the analyses were parameterized with.
    #[must_use]
    pub const fn ptr_size(&self) -> PointerSize {
        self.ptr_size
    }

    /// Returns the function's points-to relation, solving on first use.
    ///
    /// See [`analyze_function`] for what the relation models.
    #[must_use]
    pub fn points_to(&self) -> &PointsTo {
        self.points_to.get_or_init(|| analyze_function(self.ir))
    }

    /// Returns the function's alias keys, deriving them on first use.
    ///
    /// See [`alias_keys_for_function`] for the cell-identity projection.
    #[must_use]
    pub fn alias_keys(&self) -> &BTreeMap<u32, AliasKey> {
        self.alias_keys
            .get_or_init(|| alias_keys_for_function(self.ir, self.ptr_size))
    }

    /// Returns whether an analysis has been computed, for tests and for
    /// cache accounting.
    #[must_use]
    pub fn is_computed(&self, analysis: Analysis) -> bool {
        match analysis {
            Analysis::PointsTo => self.points_to.get().is_some(),
            Analysis::AliasKeys => self.alias_keys.get().is_some(),
        }
    }
}

/// Names one of the analyses [`FunctionAnalyses`] holds.
///
/// The "analysis id" half of the `(function, analysis)` key: the cache key
/// selects the function, this selects which of its analyses is meant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Analysis {
    /// The inclusion-based points-to relation.
    PointsTo,
    /// The cell-identity alias keys.
    AliasKeys,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{MockTarget, memory_effect_fixture};

    #[test]
    fn an_analysis_is_computed_once_and_reused() {
        let function = memory_effect_fixture();
        let analyses = FunctionAnalyses::<MockTarget>::new(&function, PointerSize::Bit64);

        assert!(!analyses.is_computed(Analysis::PointsTo));
        let first = analyses.points_to();
        assert!(analyses.is_computed(Analysis::PointsTo));
        let second = analyses.points_to();

        // Same allocation, not merely an equal value: the second call did not
        // re-solve.
        assert!(std::ptr::eq(first, second));
    }

    #[test]
    fn each_analysis_is_independent() {
        let function = memory_effect_fixture();
        let analyses = FunctionAnalyses::<MockTarget>::new(&function, PointerSize::Bit64);

        let _ = analyses.points_to();

        assert!(analyses.is_computed(Analysis::PointsTo));
        assert!(
            !analyses.is_computed(Analysis::AliasKeys),
            "asking for one analysis must not compute the others"
        );
    }

    #[test]
    fn the_handle_yields_the_function_it_describes() {
        let function = memory_effect_fixture();
        let analyses = FunctionAnalyses::<MockTarget>::new(&function, PointerSize::Bit64);

        assert!(std::ptr::eq(analyses.ir(), &function));
    }
}
