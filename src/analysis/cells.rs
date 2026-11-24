//! Alias-aware cells the memory analyses keep a payload in for each current
//! store: [`MemoryCells`].

use rustc_hash::FxHashMap;

use crate::{analysis::memory::MemoryLocation, target::Target};

/// Alias-aware cells with a caller-owned payload for each current store.
///
/// The same invalidation and lookup rules serve static SSA identities and
/// store-time evaluator snapshots. Payload ordering is supplied by the caller,
/// so deterministic lookup does not constrain the stored value representation.
#[derive(Debug, Clone)]
pub struct MemoryCells<T: Target, V> {
    /// Current, mutually nonoverlapping store records.
    values: FxHashMap<MemoryLocation<T>, V>,
}

impl<T: Target, V> MemoryCells<T, V> {
    /// Creates an empty table without allocating cell storage.
    pub fn new() -> Self {
        Self {
            values: FxHashMap::default(),
        }
    }

    /// Forgets every possible overlap before retaining a store's owned payload.
    pub fn store(&mut self, location: MemoryLocation<T>, value: V) {
        self.forget(&location);
        self.values.insert(location, value);
    }

    /// Forgets all cells that a write to `location` may change.
    pub fn forget(&mut self, location: &MemoryLocation<T>) {
        self.values.retain(|known, _| !known.may_alias(location));
    }

    /// Returns a direct store, or the highest-ranked must-aliasing payload.
    ///
    /// `rank` supplies a deterministic total preference for fallback matches.
    pub fn load<K: Ord>(&self, location: &MemoryLocation<T>, rank: impl Fn(&V) -> K) -> Option<&V> {
        if let Some(value) = self.values.get(location) {
            return Some(value);
        }
        self.values
            .iter()
            .filter(|(known, _)| location.must_alias(known))
            .max_by_key(|(_, value)| rank(value))
            .map(|(_, value)| value)
    }

    /// Returns only the payload stored at the exact key.
    pub fn direct(&self, location: &MemoryLocation<T>) -> Option<&V> {
        self.values.get(location)
    }

    /// Returns whether any tracked cell may alias `location`.
    pub fn has_may_alias(&self, location: &MemoryLocation<T>) -> bool {
        self.values.keys().any(|known| known.may_alias(location))
    }

    /// Updates or removes payloads while retaining only cells accepted by `keep`.
    pub fn retain(&mut self, keep: impl FnMut(&MemoryLocation<T>, &mut V) -> bool) {
        self.values.retain(keep);
    }

    /// Discards every tracked store.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Returns the number of current cells.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the table contains no cells.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
