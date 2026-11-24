//! Per-variable tables sized by the variable count rather than by the ids the
//! IR names.
//!
//! A function's variable table holds the ids `0..count`. Malformed or
//! deserialized IR can still name an id far past it (three billion and more),
//! and a scan that sizes its storage by the largest id it meets turns one bad
//! operand into an allocation of tens of gigabytes. [`VarTable`] and
//! [`VarBuckets`] size their dense part by the variable count and keep the ids
//! past it in a map, so such an id costs one entry.
//!
//! Both follow the rule of [`VarSet`](crate::ir::varstore::VarSet):
//! [`SsaVarId::PLACEHOLDER`] is not a variable, so it has no entry, reads back
//! as absent and takes no writes.

use rustc_hash::FxHashMap;

use crate::ir::variable::SsaVarId;

/// A table with an entry per [`SsaVarId`].
///
/// The ids below the variable count index a dense vector. An id at or above
/// its length is malformed or deserialized IR that a scan has to report or
/// tolerate, not size an allocation for: such ids go to a map keyed by id, so
/// the table's size follows the function's variable count.
#[derive(Debug, Clone)]
pub struct VarTable<T> {
    /// One entry per id below the variable count.
    dense: Vec<T>,
    /// The entries of ids at or above [`Self::dense`]'s length, once written.
    beyond: FxHashMap<usize, T>,
}

impl<T> VarTable<T> {
    /// Creates a table with a default entry for each of `count` ids.
    ///
    /// Size it from the function's variable count.
    #[must_use]
    pub fn new(count: usize) -> Self
    where
        T: Clone + Default,
    {
        Self {
            dense: vec![T::default(); count],
            beyond: FxHashMap::default(),
        }
    }

    /// Returns the number of ids the dense part has an entry for.
    #[must_use]
    pub fn dense_len(&self) -> usize {
        self.dense.len()
    }

    /// Returns the entry for `var`.
    ///
    /// Every id below [`Self::dense_len`] has one. The result is `None` for the
    /// placeholder and for an id past the dense part that [`Self::entry`] has
    /// not created.
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> Option<&T> {
        if var.is_placeholder() {
            return None;
        }
        let index = var.index();
        self.dense.get(index).or_else(|| self.beyond.get(&index))
    }

    /// Returns the entry for `var`, creating a default one for an id past the
    /// dense part; `None` for the placeholder.
    ///
    /// The entry exists afterwards even when the caller writes nothing to it,
    /// so a later [`Self::get`] of the id finds the default.
    pub fn entry(&mut self, var: SsaVarId) -> Option<&mut T>
    where
        T: Default,
    {
        if var.is_placeholder() {
            return None;
        }
        let index = var.index();
        if index < self.dense.len() {
            return self.dense.get_mut(index);
        }
        Some(self.beyond.entry(index).or_default())
    }
}

/// Occurrences grouped by [`SsaVarId`].
///
/// The dense part is a counting sort over the ids below the variable count: a
/// handful of allocations however many variables the function has. An id past
/// it is malformed or deserialized IR whose occurrences are rare, and they are
/// grouped by id in a map instead of sizing the sort by the largest id.
///
/// Occurrences of [`SsaVarId::PLACEHOLDER`] are dropped and it reads back as
/// having none.
#[derive(Debug, Clone)]
pub struct VarBuckets<S> {
    /// Where each dense id's occurrences start in `sites`; the occurrences of
    /// index `v` are `sites[starts[v]..starts[v + 1]]`.
    starts: Vec<usize>,
    /// The occurrences of the dense ids, grouped by id.
    sites: Vec<S>,
    /// The occurrences of each id at or above the dense part.
    beyond: FxHashMap<usize, Vec<S>>,
}

impl<S> VarBuckets<S> {
    /// Groups the occurrences `occurrences` emits.
    ///
    /// # Arguments
    ///
    /// * `dense_len` - The number of ids with a dense bucket, the function's
    ///   variable count.
    /// * `pad` - Any value of `S`, used to size the grouped storage before it is
    ///   filled.
    /// * `occurrences` - Emits every `(variable, occurrence)` pair; called twice,
    ///   once to count and once to place, and must emit the same pairs both
    ///   times. A pair the second run emits beyond the count the first run
    ///   found for its id is dropped, so a mismatch never spills into the next
    ///   id's bucket.
    #[must_use]
    pub fn build(
        dense_len: usize,
        pad: S,
        occurrences: impl Fn(&mut dyn FnMut(SsaVarId, S)),
    ) -> Self
    where
        S: Copy,
    {
        let mut starts: Vec<usize> = vec![0; dense_len.saturating_add(1)];
        occurrences(&mut |var, _| {
            if !var.is_placeholder()
                && let Some(count) = starts.get_mut(var.index().saturating_add(1))
            {
                *count = count.saturating_add(1);
            }
        });
        let mut running = 0usize;
        for start in &mut starts {
            running = running.saturating_add(*start);
            *start = running;
        }
        // Filled from each id's start; `next` walks forward as occurrences land
        // and stops at the id's end.
        let mut next = starts.clone();
        let mut sites = vec![pad; running];
        let mut beyond: FxHashMap<usize, Vec<S>> = FxHashMap::default();
        occurrences(&mut |var, site| {
            if var.is_placeholder() {
                return;
            }
            let index = var.index();
            if index >= dense_len {
                beyond.entry(index).or_default().push(site);
            } else if let Some(cursor) = next.get_mut(index)
                && let Some(&end) = starts.get(index.saturating_add(1))
                && *cursor < end
                && let Some(slot) = sites.get_mut(*cursor)
            {
                *slot = site;
                *cursor = cursor.saturating_add(1);
            }
        });
        Self {
            starts,
            sites,
            beyond,
        }
    }

    /// Returns the occurrences of `var`.
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> &[S] {
        if var.is_placeholder() {
            return &[];
        }
        let index = var.index();
        if index >= self.starts.len().saturating_sub(1) {
            return self.beyond.get(&index).map_or(&[], Vec::as_slice);
        }
        match (
            self.starts.get(index),
            self.starts.get(index.saturating_add(1)),
        ) {
            (Some(&from), Some(&to)) => self.sites.get(from..to).unwrap_or_default(),
            _ => &[],
        }
    }

    /// Returns the occurrences of `var`, mutable.
    pub fn get_mut(&mut self, var: SsaVarId) -> &mut [S] {
        if var.is_placeholder() {
            return &mut [];
        }
        let index = var.index();
        if index >= self.starts.len().saturating_sub(1) {
            return self
                .beyond
                .get_mut(&index)
                .map_or(&mut [], Vec::as_mut_slice);
        }
        match (
            self.starts.get(index),
            self.starts.get(index.saturating_add(1)),
        ) {
            (Some(&from), Some(&to)) => self.sites.get_mut(from..to).unwrap_or_default(),
            _ => &mut [],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    /// An id past the dense part takes one map entry and does not size the
    /// table; the placeholder has no entry and takes no write.
    #[test]
    fn a_table_keeps_a_far_id_in_the_map() {
        let far = SsaVarId::from_index(3_000_000_000);
        let mut table: VarTable<u32> = VarTable::new(4);
        assert_eq!(table.dense_len(), 4);
        assert_eq!(table.get(far), None);

        let Some(entry) = table.entry(far) else {
            unreachable!("a far id has an entry once asked for");
        };
        *entry = 7;
        assert_eq!(table.get(far), Some(&7));
        assert_eq!(table.dense_len(), 4);
        assert_eq!(table.dense.len(), 4);
        assert_eq!(table.beyond.len(), 1);

        let Some(inside) = table.entry(SsaVarId::from_index(2)) else {
            unreachable!("an id below the count has an entry");
        };
        *inside = 3;
        assert_eq!(table.get(SsaVarId::from_index(2)), Some(&3));
        assert_eq!(
            table.beyond.len(),
            1,
            "an id inside the dense part is not mapped"
        );

        assert!(table.entry(SsaVarId::PLACEHOLDER).is_none());
        assert_eq!(table.get(SsaVarId::PLACEHOLDER), None);
        assert_eq!(table.beyond.len(), 1);
    }

    /// Asking for a far id's entry creates it even when nothing is written.
    #[test]
    fn asking_for_a_far_entry_creates_it() {
        let far = SsaVarId::from_index(3_000_000_000);
        let mut table: VarTable<u32> = VarTable::new(2);
        assert_eq!(table.get(far), None);
        assert!(table.entry(far).is_some());
        assert_eq!(table.get(far), Some(&0));
    }

    /// Occurrences group by id; an id past the dense part is bucketed in the
    /// map beside them, and the placeholder has none.
    #[test]
    fn buckets_group_dense_and_far_ids_and_drop_the_placeholder() {
        let far = SsaVarId::from_index(3_000_000_000);
        let (a, b) = (SsaVarId::from_index(0), SsaVarId::from_index(2));
        let pairs = [
            (b, 1u32),
            (far, 2),
            (a, 3),
            (SsaVarId::PLACEHOLDER, 4),
            (b, 5),
            (far, 6),
        ];
        let mut buckets = VarBuckets::build(3, 0, |emit| {
            for (var, site) in pairs {
                emit(var, site);
            }
        });
        assert_eq!(buckets.get(a), [3]);
        assert_eq!(buckets.get(SsaVarId::from_index(1)), [] as [u32; 0]);
        assert_eq!(buckets.get(b), [1, 5]);
        assert_eq!(buckets.get(far), [2, 6]);
        assert_eq!(buckets.get(SsaVarId::PLACEHOLDER), [] as [u32; 0]);
        assert_eq!(buckets.sites.len(), 3, "only the dense ids take the sort");
        assert_eq!(buckets.beyond.len(), 1);

        buckets.get_mut(b).reverse();
        assert_eq!(buckets.get(b), [5, 1]);
        assert!(buckets.get_mut(SsaVarId::PLACEHOLDER).is_empty());
    }

    /// A second emit run that names an id more often than the first stops at
    /// that id's bucket: the surplus pair is dropped and the next bucket keeps
    /// its own occurrences. The control emits the same pairs both times and
    /// every pair lands in its bucket.
    #[test]
    fn a_mismatched_second_run_does_not_spill_into_the_next_bucket() {
        let (a, b) = (SsaVarId::from_index(0), SsaVarId::from_index(1));
        let runs = Cell::new(0u32);
        let mismatched = VarBuckets::build(2, 0u32, |emit| {
            let second = runs.get() == 1;
            runs.set(runs.get().saturating_add(1));
            emit(b, 20);
            emit(a, 10);
            if second {
                emit(a, 11);
            }
        });
        assert_eq!(mismatched.get(a), [10]);
        assert_eq!(mismatched.get(b), [20]);

        let matched = VarBuckets::build(2, 0u32, |emit| {
            emit(b, 20);
            emit(a, 10);
            emit(a, 11);
        });
        assert_eq!(matched.get(a), [10, 11]);
        assert_eq!(matched.get(b), [20]);
    }
}
