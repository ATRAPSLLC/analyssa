//! [`AnalysisCache`] - per-method [`FunctionAnalyses`], keyed by the host's
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
//! [`AnalysisCache::invalidate`] drops a method's analyses. A host wires it to
//! [`SsaPassHost::invalidate_analyses`](crate::scheduling::SsaPassHost::invalidate_analyses),
//! which the scheduler calls after every pass run that changed a method, so the
//! next pass in the same iteration sees analyses of the current IR. Within a
//! single run over immutable IR nothing needs to call it - the borrow already
//! guarantees freshness - so it exists for the iterative case, where the program
//! changes underneath the analyses.
//!
//! [`FunctionAnalyses`]: crate::analysis::cache::FunctionAnalyses

use std::{
    collections::VecDeque,
    hash::Hash,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use rustc_hash::FxHashMap;

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
/// [`FunctionAnalyses`] directly - `analysir` pairs the analyses with its own
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
/// # Capacity
///
/// [`AnalysisCache::new`] keeps every entry for the life of the cache. A driver
/// that visits a large program holds the analyses of every method it has
/// touched, however long ago, and most of those are never asked for again:
/// [`AnalysisCache::bounded`] keeps at most a fixed number, and when a new entry
/// would exceed it the entry inserted earliest is dropped. A dropped method is
/// rebuilt on its next lookup, and an [`Arc`] already handed out stays valid
/// for as long as its holder keeps it.
///
/// [`FunctionAnalyses`]: crate::analysis::cache::FunctionAnalyses
#[derive(Debug)]
pub struct AnalysisCache<K, V> {
    /// The entries held.
    entries: Mutex<Entries<K, V>>,
    /// Hit/miss accounting.
    stats: Mutex<CacheStats>,
}

/// The entries of an [`AnalysisCache`], and the order they were inserted in.
#[derive(Debug)]
struct Entries<K, V> {
    /// Method identity to its analysis set.
    map: FxHashMap<K, Arc<V>>,
    /// The keys in `map`, earliest insertion first. Kept only when the cache is
    /// bounded.
    order: VecDeque<K>,
    /// The most entries held at once, or `None` for no limit.
    capacity: Option<NonZeroUsize>,
}

impl<K, V> Default for AnalysisCache<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(Entries {
                map: FxHashMap::default(),
                order: VecDeque::new(),
                capacity: None,
            }),
            stats: Mutex::new(CacheStats::default()),
        }
    }
}

impl<K: Eq + Hash + Clone, V> AnalysisCache<K, V> {
    /// Builds an empty cache.
    ///
    /// Nothing is computed up front; entries are built lazily on first lookup
    /// through [`get_or_insert_with`](Self::get_or_insert_with), and the
    /// hit/miss counters start at zero.
    ///
    /// # Returns
    ///
    /// A cache holding no analysis sets.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an empty cache that holds at most `capacity` entries.
    ///
    /// Inserting a new entry into a full cache drops the one inserted earliest;
    /// a hit does not renew an entry. See the type's *Capacity* section.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The most entries held at once.
    ///
    /// # Returns
    ///
    /// A cache holding no analysis sets.
    #[must_use]
    pub fn bounded(capacity: NonZeroUsize) -> Self {
        let cache = Self::default();
        if let Ok(mut entries) = cache.entries.lock() {
            entries.capacity = Some(capacity);
        }
        cache
    }

    /// Returns `key`'s analysis set, building it with `make` on first use.
    ///
    /// A hit clones the stored [`Arc`] and returns immediately. On a miss
    /// `make` runs **without the map lock held** - analyses can be expensive
    /// and must not serialize unrelated methods - then the result is committed
    /// under the lock. The first insert of a key wins: if another thread
    /// inserted the same key meanwhile, its entry stands and this thread's
    /// build is discarded. A bounded cache that is full drops its earliest
    /// entry to make room.
    /// A poisoned lock (another thread panicked mid-update) falls back to
    /// returning the freshly built, uncached value rather than propagating the
    /// panic: the cache holds only derived facts, so a correct answer is still
    /// available.
    ///
    /// # Arguments
    ///
    /// * `key` - Method identity to look up or populate.
    /// * `make` - Builder invoked once on a miss to produce the analysis set.
    ///
    /// # Returns
    ///
    /// A shared [`Arc`] to `key`'s analysis set - the existing entry on a hit,
    /// or the winning entry on a miss.
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
            Ok(mut entries) => entries.insert(key, built),
            // A poisoned lock means another thread panicked mid-update. The
            // cache holds only derived facts, so the caller still gets a
            // correct - merely uncached - answer.
            Err(_) => built,
        }
    }

    /// Drops `key`'s analyses, so the next lookup rebuilds them.
    ///
    /// A host calls this from
    /// [`SsaPassHost::invalidate_analyses`](crate::scheduling::SsaPassHost::invalidate_analyses):
    /// when a pass rewrites a method's SSA the cached facts no longer describe
    /// it, and removing the entry forces the next [`get_or_insert_with`](Self::get_or_insert_with)
    /// to recompute from the current IR. A missing key is a no-op, and a
    /// poisoned lock is swallowed - a failed invalidation only risks a stale
    /// entry, never a crash.
    ///
    /// # Arguments
    ///
    /// * `key` - Method identity whose analyses should be discarded.
    pub fn invalidate(&self, key: &K) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.remove(key);
        }
    }

    /// Returns the hit/miss counts accumulated so far.
    ///
    /// A snapshot copy of the counters, not a live handle. A poisoned lock
    /// yields the default (all-zero) [`CacheStats`] rather than panicking, so
    /// instrumentation can never take down a caller.
    ///
    /// # Returns
    ///
    /// The [`CacheStats`] recorded up to this call.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().map(|stats| *stats).unwrap_or_default()
    }

    /// Returns `key`'s entry when one is present.
    ///
    /// Acquires the map lock, clones the stored [`Arc`] on a hit, and releases
    /// the lock before returning - the caller touches the analysis set with no
    /// lock held. A poisoned lock is treated as a miss.
    ///
    /// # Arguments
    ///
    /// * `key` - Method identity to look up.
    ///
    /// # Returns
    ///
    /// The shared entry when present, otherwise `None`.
    fn lookup(&self, key: &K) -> Option<Arc<V>> {
        self.entries.lock().ok()?.map.get(key).map(Arc::clone)
    }

    /// Records one lookup outcome.
    ///
    /// Bumps the hit or miss counter with a saturating add so the tallies never
    /// wrap on a long-running driver, and silently drops the update if the
    /// stats lock is poisoned - accounting is diagnostic and must not perturb
    /// the caller.
    ///
    /// # Arguments
    ///
    /// * `hit` - `true` for a served entry, `false` for a freshly built one.
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

impl<K: Eq + Hash + Clone, V> Entries<K, V> {
    /// Commits `built` as `key`'s entry unless one is already there, and
    /// returns the entry that stands.
    ///
    /// A bounded cache drops its earliest entries until the new one fits.
    fn insert(&mut self, key: &K, built: Arc<V>) -> Arc<V> {
        if let Some(existing) = self.map.get(key) {
            return Arc::clone(existing);
        }
        self.map.insert(key.clone(), Arc::clone(&built));
        if let Some(capacity) = self.capacity {
            self.order.push_back(key.clone());
            while self.map.len() > capacity.get() {
                let Some(oldest) = self.order.pop_front() else {
                    break;
                };
                self.map.remove(&oldest);
            }
        }
        built
    }

    /// Drops `key`'s entry, if there is one.
    fn remove(&mut self, key: &K) {
        if self.map.remove(key).is_some() && self.capacity.is_some() {
            self.order.retain(|held| held != key);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    /// A repeated lookup of the same key builds the entry exactly once and
    /// serves the cached [`Arc`] thereafter, leaving one miss and the rest hits.
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

    /// Distinct keys are cached independently: a second key gets its own entry,
    /// and re-requesting the first returns its original value, not the new
    /// builder's output.
    #[test]
    fn distinct_keys_get_distinct_entries() {
        let cache: AnalysisCache<u32, u32> = AnalysisCache::new();

        assert_eq!(*cache.get_or_insert_with(&1, || 10), 10);
        assert_eq!(*cache.get_or_insert_with(&2, || 20), 20);
        assert_eq!(*cache.get_or_insert_with(&1, || 99), 10);

        assert_eq!(cache.stats().misses, 2);
    }

    /// Invalidating a key drops its entry so the next lookup rebuilds from the
    /// current builder, and the rebuild is counted as a fresh miss.
    #[test]
    fn invalidating_forces_a_rebuild() {
        let cache: AnalysisCache<u32, u32> = AnalysisCache::new();

        assert_eq!(*cache.get_or_insert_with(&1, || 10), 10);
        cache.invalidate(&1);
        assert_eq!(*cache.get_or_insert_with(&1, || 20), 20);

        assert_eq!(cache.stats().misses, 2, "the rebuild is a miss");
    }

    /// Invalidation is scoped to its key: dropping one method's analyses leaves
    /// every other method's cached entry intact.
    #[test]
    fn invalidating_leaves_other_methods_alone() {
        let cache: AnalysisCache<u32, u32> = AnalysisCache::new();

        let _ = cache.get_or_insert_with(&1, || 10);
        let _ = cache.get_or_insert_with(&2, || 20);
        cache.invalidate(&1);

        assert_eq!(*cache.get_or_insert_with(&2, || 99), 20);
    }

    /// A bounded cache of two.
    fn two() -> AnalysisCache<u32, u32> {
        AnalysisCache::bounded(NonZeroUsize::MIN.saturating_add(1))
    }

    /// A bounded cache drops the entry inserted earliest when a new one would
    /// exceed its capacity, rebuilds the dropped method on its next lookup, and
    /// leaves the entries that stayed alone.
    #[test]
    fn a_full_cache_drops_the_earliest_entry() {
        let cache = two();

        let _ = cache.get_or_insert_with(&1, || 10);
        let _ = cache.get_or_insert_with(&2, || 20);
        let _ = cache.get_or_insert_with(&3, || 30);

        assert_eq!(*cache.get_or_insert_with(&2, || 99), 20, "2 stayed");
        assert_eq!(*cache.get_or_insert_with(&3, || 99), 30, "3 stayed");
        assert_eq!(*cache.get_or_insert_with(&1, || 11), 11, "1 was rebuilt");
        assert_eq!(cache.stats(), CacheStats { hits: 2, misses: 4 });
    }

    /// A hit does not renew an entry: the one inserted earliest goes first even
    /// when it was the last one asked for.
    #[test]
    fn a_hit_does_not_renew_an_entry() {
        let cache = two();

        let _ = cache.get_or_insert_with(&1, || 10);
        let _ = cache.get_or_insert_with(&2, || 20);
        let _ = cache.get_or_insert_with(&1, || 99);
        let _ = cache.get_or_insert_with(&3, || 30);

        assert_eq!(*cache.get_or_insert_with(&1, || 11), 11, "1 went first");
    }

    /// An [`Arc`] handed out before its entry was dropped stays usable, and an
    /// invalidated key frees its slot so a later insert drops nothing.
    #[test]
    fn a_dropped_entry_stays_valid_for_its_holder() {
        let cache = two();

        let held = cache.get_or_insert_with(&1, || 10);
        let _ = cache.get_or_insert_with(&2, || 20);
        let _ = cache.get_or_insert_with(&3, || 30);
        assert_eq!(*held, 10);
        assert_eq!(*cache.get_or_insert_with(&1, || 11), 11, "1 was dropped");

        cache.invalidate(&3);
        let _ = cache.get_or_insert_with(&4, || 40);
        assert_eq!(*cache.get_or_insert_with(&1, || 99), 11, "1 stayed");
    }

    /// An unbounded cache never drops an entry.
    #[test]
    fn an_unbounded_cache_keeps_everything() {
        let cache: AnalysisCache<u32, u32> = AnalysisCache::new();

        for key in 0..64 {
            let _ = cache.get_or_insert_with(&key, || key);
        }

        for key in 0..64 {
            assert_eq!(*cache.get_or_insert_with(&key, || 999), key);
        }
        assert_eq!(
            cache.stats(),
            CacheStats {
                hits: 64,
                misses: 64
            }
        );
    }
}
