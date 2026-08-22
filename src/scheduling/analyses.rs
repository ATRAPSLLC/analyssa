//! [`AnalysisCache`] — per-method [`FunctionAnalyses`], keyed by the host's
//! method identity.
//!
//! [`FunctionAnalyses`] memoizes the analyses of *one* function. A driver that
//! walks a whole program needs one per method and needs them to outlive the
//! visit that created them, which is what this holds.
//!
//! Together the two halves are the `(method, analysis)` key: the map selects
//! the function, [`Analysis`](crate::analysis::cache::Analysis) selects which of
//! its analyses is meant.
//!
//! # Invalidation
//!
//! [`AnalysisCache::invalidate`] drops a method's analyses. It is the seam a
//! [`DirtySet`](crate::host::DirtySet) attaches to: a pass that changes a
//! method's SSA marks it dirty, and the same identity invalidates here. Within a
//! single run over immutable IR nothing needs to call it — the borrow already
//! guarantees freshness — so it exists for the iterative case, where the program
//! changes underneath the analyses.

use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex},
};

/// How many lookups a cache served and how many it had to compute.
///
/// The instrumented count: `misses` is the number of times an analysis set was
/// actually built, so a driver that visits a method repeatedly should show one
/// miss and many hits for it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups served from an existing entry.
    pub hits: usize,
    /// Lookups that had to build a new entry.
    pub misses: usize,
}

/// Per-method analysis sets, built on first use and reused thereafter.
///
/// Generic over the host's method identity `K` and over the entry type `V`, so
/// a host can cache its own richer per-function handle rather than
/// [`FunctionAnalyses`] directly — `analysir` pairs the analyses with its own
/// function wrapper, for instance.
///
/// # Concurrency
///
/// Entries are handed out as [`Arc`]s and the map lock is released before the
/// caller touches one, so an analysis runs with no lock held: two threads
/// asking for *different* methods never contend, and two asking for the same
/// method share one entry whose own slots settle the race.
///
/// The builder may run more than once for a key under contention; the first
/// entry to be inserted wins and is the one every caller sees. Analyses are
/// pure functions of the IR, so a discarded duplicate costs work, never
/// correctness.
///
/// [`FunctionAnalyses`]: crate::analysis::cache::FunctionAnalyses
#[derive(Debug)]
pub struct AnalysisCache<K, V> {
    /// Method identity to its analysis set.
    entries: Mutex<HashMap<K, Arc<V>>>,
    /// Hit/miss accounting.
    stats: Mutex<CacheStats>,
}

impl<K, V> Default for AnalysisCache<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            stats: Mutex::new(CacheStats::default()),
        }
    }
}

impl<K: Eq + Hash + Clone, V> AnalysisCache<K, V> {
    /// Builds an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `key`'s analysis set, building it with `make` on first use.
    ///
    /// The lock is not held while `make` runs.
    pub fn get_or_insert_with<F>(&self, key: &K, make: F) -> Arc<V>
    where
        F: FnOnce() -> V,
    {
        if let Some(entry) = self.lookup(key) {
            self.record(true);
            return entry;
        }
        let built = Arc::new(make());
        self.record(false);
        match self.entries.lock() {
            Ok(mut entries) => Arc::clone(entries.entry(key.clone()).or_insert(built)),
            // A poisoned lock means another thread panicked mid-update. The
            // cache holds only derived facts, so the caller still gets a
            // correct — merely uncached — answer.
            Err(_) => built,
        }
    }

    /// Drops `key`'s analyses, so the next lookup rebuilds them.
    ///
    /// Call when the method's SSA has changed.
    pub fn invalidate(&self, key: &K) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.remove(key);
        }
    }

    /// Returns the hit/miss counts accumulated so far.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().map(|stats| *stats).unwrap_or_default()
    }

    /// Returns `key`'s entry when one is present.
    fn lookup(&self, key: &K) -> Option<Arc<V>> {
        self.entries.lock().ok()?.get(key).map(Arc::clone)
    }

    /// Records one lookup outcome.
    fn record(&self, hit: bool) {
        if let Ok(mut stats) = self.stats.lock() {
            if hit {
                stats.hits = stats.hits.saturating_add(1);
            } else {
                stats.misses = stats.misses.saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn an_entry_is_built_once_and_then_served() {
        let cache: AnalysisCache<u32, u32> = AnalysisCache::new();
        let builds = Cell::new(0);
        let mut build = || {
            builds.set(builds.get() + 1);
            builds.get()
        };

        assert_eq!(*cache.get_or_insert_with(&7, &mut build), 1);
        assert_eq!(*cache.get_or_insert_with(&7, &mut build), 1);
        assert_eq!(*cache.get_or_insert_with(&7, &mut build), 1);

        assert_eq!(builds.get(), 1);
        assert_eq!(cache.stats(), CacheStats { hits: 2, misses: 1 });
    }

    #[test]
    fn distinct_keys_get_distinct_entries() {
        let cache: AnalysisCache<u32, u32> = AnalysisCache::new();

        assert_eq!(*cache.get_or_insert_with(&1, || 10), 10);
        assert_eq!(*cache.get_or_insert_with(&2, || 20), 20);
        assert_eq!(*cache.get_or_insert_with(&1, || 99), 10);

        assert_eq!(cache.stats().misses, 2);
    }

    #[test]
    fn invalidating_forces_a_rebuild() {
        let cache: AnalysisCache<u32, u32> = AnalysisCache::new();

        assert_eq!(*cache.get_or_insert_with(&1, || 10), 10);
        cache.invalidate(&1);
        assert_eq!(*cache.get_or_insert_with(&1, || 20), 20);

        assert_eq!(cache.stats().misses, 2, "the rebuild is a miss");
    }

    #[test]
    fn invalidating_leaves_other_methods_alone() {
        let cache: AnalysisCache<u32, u32> = AnalysisCache::new();

        let _ = cache.get_or_insert_with(&1, || 10);
        let _ = cache.get_or_insert_with(&2, || 20);
        cache.invalidate(&1);

        assert_eq!(*cache.get_or_insert_with(&2, || 99), 20);
    }
}
