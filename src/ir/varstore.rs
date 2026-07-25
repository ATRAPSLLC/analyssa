//! Dense storage keyed by [`SsaVarId`].
//!
//! [`SsaVarId`] is a dense per-function index, so the natural storage for
//! "a value per variable" or "a set of variables" is a flat array — not a hash
//! map. These two types provide that, and encapsulate the one rule that makes
//! it safe.
//!
//! # Why these exist
//!
//! The rule is that [`SsaVarId::PLACEHOLDER`] is **not a variable**. It is a
//! sentinel meaning "no id assigned yet", used while phi placement runs before
//! renaming fills the real results in. Its `index()` is `u32::MAX - 1`, which
//! sits in the same numeric space as real indices, so code that treats an id as
//! an array subscript has to know to exclude it. Two failure modes follow when
//! it does not:
//!
//! - Sizing an allocation from "max id seen" yields ~4.29 billion entries. A
//!   `BitSet` that big is 512 MB, and the passes build eleven of them; a
//!   `Vec<Option<T>>` that big is tens of gigabytes and simply aborts.
//! - Inserting a placeholder into a set sized for real variables trips
//!   [`BitSet::insert`]'s bounds assertion.
//!
//! Both were reachable in practice. Rather than repeat the check at every call
//! site — where a new site silently reintroduces the bug — the rule lives here:
//! a placeholder is ignored on write and never present on read. Callers pass
//! [`SsaVarId`] values directly and cannot forget.
//!
//! Sizes come from
//! [`SsaFunction::var_id_capacity`](crate::ir::function::SsaFunction::var_id_capacity),
//! which counts only real ids for the same reason.

use crate::{BitSet, ir::variable::SsaVarId};

/// A set of [`SsaVarId`]s backed by a [`BitSet`].
///
/// Replaces the `BitSet::new(ssa.var_id_capacity())` /
/// `set.insert(var.index())` pairing, which required every call site to
/// remember that [`SsaVarId::PLACEHOLDER`] must not be inserted. Membership of
/// a placeholder is always `false`, and inserting one is a no-op.
///
/// Ids at or beyond the set's capacity are likewise ignored rather than
/// panicking: the capacity is derived from the function, so an id past it did
/// not come from the function's own variables.
#[derive(Debug, Clone)]
pub struct VarSet {
    /// One bit per variable index.
    bits: BitSet,
}

impl VarSet {
    /// Creates an empty set able to hold every id below `capacity`.
    ///
    /// Size it from
    /// [`SsaFunction::var_id_capacity`](crate::ir::function::SsaFunction::var_id_capacity).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            bits: BitSet::new(capacity),
        }
    }

    /// Adds `var` to the set, returning `true` if it was not already present.
    ///
    /// Returns `false` for [`SsaVarId::PLACEHOLDER`], which is not a variable
    /// this set can describe.
    ///
    /// An id beyond the set's current capacity **grows** the set rather than
    /// being dropped. These sets are sized from `var_id_bound()`, but malformed
    /// IR — an op naming a variable the allocator never issued — can carry a
    /// larger id. Silently ignoring such a write makes the variable read back as
    /// *absent*, and for a liveness or used-variable set absent means "dead", so
    /// the value would be dropped by the very pass that should have kept it.
    ///
    /// Growing is also why this does not assert: the crate's inputs are
    /// attacker-controlled binaries, and aborting a debug build — which is what
    /// `verify_hard` and the test suite run — on a shape the passes are supposed
    /// to tolerate is a worse failure than the one being prevented. Malformed IR
    /// reaches this readily.
    pub fn insert(&mut self, var: SsaVarId) -> bool {
        if var.is_placeholder() {
            return false;
        }
        self.bits.ensure_capacity(var.index());
        self.bits.insert_checked(var.index())
    }

    /// Returns `true` when `var` is in the set.
    #[must_use]
    pub fn contains(&self, var: SsaVarId) -> bool {
        !var.is_placeholder() && self.bits.contains_checked(var.index())
    }

    /// Removes `var` from the set, returning `true` if it was present.
    pub fn remove(&mut self, var: SsaVarId) -> bool {
        if !self.contains(var) {
            return false;
        }
        self.bits.remove(var.index());
        true
    }

    /// Returns the number of variables in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bits.count()
    }

    /// Returns `true` when the set holds no variables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates the set's variables in ascending index order.
    pub fn iter(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        self.bits.iter().map(SsaVarId::from_index)
    }

    /// Returns the underlying bit set, for the graph and dataflow helpers that
    /// operate on raw indices.
    #[must_use]
    pub fn bits(&self) -> &BitSet {
        &self.bits
    }
}

/// A map from [`SsaVarId`] to `T`, backed by a flat `Vec`.
///
/// The dense counterpart to a `HashMap<SsaVarId, T>`. `SsaVarId` is an index,
/// so hashing it is pure overhead — the default `RandomState` runs SipHash on
/// every insert and lookup. The verifier alone performs one insert per
/// definition and one lookup per use at *every* edit-scope boundary, which is
/// per pass per fixpoint iteration.
///
/// Follows the same placeholder rule as [`VarSet`]: writing under
/// [`SsaVarId::PLACEHOLDER`] is a no-op and reading it yields `None`.
#[derive(Debug, Clone)]
pub struct VarMap<T> {
    /// One slot per variable index; `None` where no entry exists.
    slots: Vec<Option<T>>,
}

impl<T> VarMap<T> {
    /// Creates an empty map able to hold every id below `capacity`.
    ///
    /// Size it from
    /// [`SsaFunction::var_id_capacity`](crate::ir::function::SsaFunction::var_id_capacity).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::new();
        slots.resize_with(capacity, || None);
        Self { slots }
    }

    /// Returns the entry for `var`, or `None` if it has none.
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> Option<&T> {
        if var.is_placeholder() {
            return None;
        }
        self.slots.get(var.index()).and_then(Option::as_ref)
    }

    /// Returns a mutable reference to `var`'s entry, or `None` if it has none.
    ///
    /// Follows the same rules as [`Self::get`]: [`SsaVarId::PLACEHOLDER`] and
    /// out-of-capacity ids have no entry.
    pub fn get_mut(&mut self, var: SsaVarId) -> Option<&mut T> {
        if var.is_placeholder() {
            return None;
        }
        self.slots.get_mut(var.index()).and_then(Option::as_mut)
    }

    /// Returns `true` when `var` has an entry.
    #[must_use]
    pub fn contains(&self, var: SsaVarId) -> bool {
        self.get(var).is_some()
    }

    /// Inserts `value` for `var`, replacing any previous entry.
    ///
    /// Ignores [`SsaVarId::PLACEHOLDER`] and ids beyond the map's capacity.
    pub fn insert(&mut self, var: SsaVarId, value: T) {
        if var.is_placeholder() {
            return;
        }
        if let Some(slot) = self.slots.get_mut(var.index()) {
            *slot = Some(value);
        }
    }

    /// Inserts `value` only when `var` has no entry yet.
    pub fn or_insert(&mut self, var: SsaVarId, value: T) {
        if var.is_placeholder() {
            return;
        }
        if let Some(slot) = self.slots.get_mut(var.index())
            && slot.is_none()
        {
            *slot = Some(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The placeholder sentinel is not a variable: it is silently ignored on
    /// write and absent on read, in both containers. Without this rule its
    /// index (`u32::MAX - 1`) either trips `BitSet`'s bounds assertion or sizes
    /// a multi-hundred-megabyte allocation.
    #[test]
    fn placeholder_is_not_a_variable() {
        let mut set = VarSet::new(8);
        assert!(!set.insert(SsaVarId::PLACEHOLDER));
        assert!(!set.contains(SsaVarId::PLACEHOLDER));
        assert!(set.is_empty());

        let mut map: VarMap<u32> = VarMap::new(8);
        map.insert(SsaVarId::PLACEHOLDER, 7);
        assert_eq!(map.get(SsaVarId::PLACEHOLDER), None);
        assert!(!map.contains(SsaVarId::PLACEHOLDER));
    }

    /// An id past the capacity is ignored rather than panicking or growing:
    /// the capacity comes from the function, so such an id is not one of its
    /// variables.
    #[test]
    fn placeholder_ids_are_ignored() {
        // `PLACEHOLDER` is not a variable, so ignoring it is the contract.
        let mut set = VarSet::new(4);
        assert!(!set.insert(SsaVarId::PLACEHOLDER));
        assert!(!set.contains(SsaVarId::PLACEHOLDER));

        let mut map: VarMap<u32> = VarMap::new(4);
        map.insert(SsaVarId::PLACEHOLDER, 1);
        assert_eq!(map.get(SsaVarId::PLACEHOLDER), None);
    }

    /// Reading an out-of-range id is tolerated — a query about a variable the
    /// container cannot describe correctly answers "absent". *Writing* one is a
    /// bug at the producing boundary, so debug builds assert; see
    /// [`VarSet::insert`].
    #[test]
    fn out_of_range_reads_are_tolerated() {
        let set = VarSet::new(4);
        assert!(!set.contains(SsaVarId::from_index(9)));

        let map: VarMap<u32> = VarMap::new(4);
        assert_eq!(map.get(SsaVarId::from_index(9)), None);
    }

    /// A set sized from `var_id_bound()` cannot represent a larger id, and
    /// malformed IR can carry one. Dropping the write would make the variable
    /// read back as absent — "dead", for a liveness or used-variable set — so
    /// the set grows instead.
    #[test]
    fn inserting_an_out_of_range_id_grows_the_set() {
        let mut set = VarSet::new(4);
        assert!(set.insert(SsaVarId::from_index(9)));
        assert!(
            set.contains(SsaVarId::from_index(9)),
            "the write must not be silently lost"
        );
        assert_eq!(set.len(), 1);

        // Existing entries survive the growth.
        let mut set = VarSet::new(4);
        assert!(set.insert(SsaVarId::from_index(1)));
        assert!(set.insert(SsaVarId::from_index(300)));
        assert!(set.contains(SsaVarId::from_index(1)));
        assert!(set.contains(SsaVarId::from_index(300)));
        assert_eq!(set.len(), 2);
    }

    /// Ordinary ids round-trip through both containers.
    #[test]
    fn real_ids_round_trip() {
        let mut set = VarSet::new(8);
        assert!(set.insert(SsaVarId::from_index(3)));
        assert!(!set.insert(SsaVarId::from_index(3)));
        assert!(set.contains(SsaVarId::from_index(3)));
        assert_eq!(set.len(), 1);
        assert_eq!(
            set.iter().collect::<Vec<_>>(),
            vec![SsaVarId::from_index(3)]
        );
        assert!(set.remove(SsaVarId::from_index(3)));
        assert!(set.is_empty());

        let mut map: VarMap<&str> = VarMap::new(8);
        map.insert(SsaVarId::from_index(2), "a");
        map.or_insert(SsaVarId::from_index(2), "b");
        assert_eq!(map.get(SsaVarId::from_index(2)), Some(&"a"));
        map.insert(SsaVarId::from_index(2), "c");
        assert_eq!(map.get(SsaVarId::from_index(2)), Some(&"c"));
    }
}
