//! Ordered execution cells with shared snapshots and copy-on-write pages.
//!
//! Roots share a page index until mutation. A write copies only its page's
//! cells, at most 128. Snapshots share the index and its immutable pages until
//! mutation makes the index and the affected page unique.

use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
    ops::{Bound, RangeBounds},
    sync::Arc,
};

use crate::analysis::execution::{memory::Cell, value::Root};

/// An ordered memory coordinate.
type Key = (Root, i64);
/// A root and its signed page index.
type PageKey = (Root, i64);
/// A fixed-range page's populated byte positions.
type Page = BTreeMap<u8, Cell>;
/// Shared pages in complete signed-coordinate order.
type Pages = BTreeMap<PageKey, Arc<Page>>;

/// Low coordinate bits held inside one page.
const PAGE_BITS: u32 = 7;

/// Snapshot storage whose clones share cells and whose mutations preserve isolation.
#[derive(Debug, Clone, Default)]
pub struct Cells {
    /// No allocation is needed for an empty memory.
    pages: Option<Arc<Pages>>,
    /// Populated cells, independently of page occupancy.
    len: usize,
}

impl Cells {
    /// Returns whether equal-sized memories share their exact cell storage.
    pub fn shares(&self, other: &Self) -> bool {
        self.len == other.len
            && match (&self.pages, &other.pages) {
                (None, None) => true,
                (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                _ => false,
            }
    }

    /// Shares only the pages belonging to one root, excluding unrelated storage.
    pub fn for_root(&self, root: Root) -> Self {
        let pages: Pages = self
            .pages
            .iter()
            .flat_map(|pages| pages.range((root, i64::MIN)..=(root, i64::MAX)))
            .map(|(key, page)| (*key, Arc::clone(page)))
            .collect();
        let len = pages
            .values()
            .fold(0usize, |count, page| count.saturating_add(page.len()));
        Self {
            pages: (!pages.is_empty()).then(|| Arc::new(pages)),
            len,
        }
    }

    /// Enumerates page identities without hiding scans of shared pages.
    pub fn page_keys(&self) -> impl Iterator<Item = PageKey> {
        self.pages.iter().flat_map(|pages| pages.keys().copied())
    }

    /// Returns whether a page is present.
    pub fn contains_page(&self, key: PageKey) -> bool {
        self.pages
            .as_ref()
            .is_some_and(|pages| pages.contains_key(&key))
    }

    /// Returns whether both memories retain the same immutable leaf.
    pub fn shares_page(&self, other: &Self, key: PageKey) -> bool {
        match (
            self.pages.as_ref().and_then(|pages| pages.get(&key)),
            other.pages.as_ref().and_then(|pages| pages.get(&key)),
        ) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    /// Returns the smallest and largest explicit coordinates in one bounded leaf.
    pub fn page_limits(&self, key: PageKey) -> Option<(i64, i64)> {
        let page = self.pages.as_ref()?.get(&key)?;
        let base = key.1.wrapping_shl(PAGE_BITS);
        Some((
            base.saturating_add(i64::from(*page.first_key_value()?.0)),
            base.saturating_add(i64::from(*page.last_key_value()?.0)),
        ))
    }

    /// Enumerates explicit coordinates in one bounded leaf.
    pub fn keys_on_page(&self, key: PageKey) -> impl Iterator<Item = Key> {
        let base = key.1.wrapping_shl(PAGE_BITS);
        self.pages
            .iter()
            .filter_map(move |pages| pages.get(&key))
            .flat_map(move |page| {
                page.keys()
                    .map(move |slot| (key.0, base.saturating_add(i64::from(*slot))))
            })
    }

    /// Divides a signed coordinate into its page and byte slot.
    fn coordinate(key: Key) -> (PageKey, u8) {
        (
            (key.0, key.1 >> PAGE_BITS),
            (key.1.cast_unsigned() & 127) as u8,
        )
    }

    /// Reads one explicit cell.
    pub fn get(&self, key: &Key) -> Option<&Cell> {
        let (page, slot) = Self::coordinate(*key);
        self.pages.as_ref()?.get(&page)?.get(&slot)
    }

    /// Returns whether a coordinate is populated.
    pub fn contains_key(&self, key: &Key) -> bool {
        self.get(key).is_some()
    }

    /// Makes the page index unique while retaining shared immutable leaves.
    fn writable_pages(&mut self) -> &mut Pages {
        Arc::make_mut(self.pages.get_or_insert_with(|| Arc::new(Pages::new())))
    }

    /// Stores a cell without changing any prior snapshot.
    pub fn insert(&mut self, key: Key, cell: Cell) {
        if self
            .pages
            .as_ref()
            .is_some_and(|pages| Arc::strong_count(pages) > 1)
            && self.get(&key) == Some(&cell)
        {
            return;
        }
        let (key, slot) = Self::coordinate(key);
        let page = self.writable_pages().entry(key).or_default();
        if Arc::make_mut(page).insert(slot, cell).is_none() {
            self.len = self.len.saturating_add(1);
        }
    }

    /// Removes a coordinate without changing any prior snapshot.
    pub fn remove(&mut self, key: &Key) {
        if self.get(key).is_none() {
            return;
        }
        let (key, slot) = Self::coordinate(*key);
        let pages = self.writable_pages();
        let Some(page) = pages.get_mut(&key) else {
            return;
        };
        if page.len() == 1 {
            pages.remove(&key);
        } else {
            Arc::make_mut(page).remove(&slot);
        }
        let empty = pages.is_empty();
        self.len = self.len.saturating_sub(1);
        if empty {
            self.pages = None;
        }
    }

    /// Enumerates a page's full coordinates, borrowing its cells.
    fn page_entries<'a>(
        (key, page): (&'a PageKey, &'a Arc<Page>),
    ) -> impl DoubleEndedIterator<Item = (Key, &'a Cell)> + 'a {
        let root = key.0;
        let base = key.1.wrapping_shl(PAGE_BITS);
        page.iter()
            .map(move |(slot, cell)| ((root, base.saturating_add(i64::from(*slot))), cell))
    }

    /// Iterates populated cells in the exact order of a flat ordered map.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Key, &Cell)> {
        self.pages
            .iter()
            .flat_map(|pages| pages.iter())
            .flat_map(Self::page_entries)
    }

    /// Returns the first populated coordinate and its cell.
    pub fn first_key_value(&self) -> Option<(Key, &Cell)> {
        self.iter().next()
    }

    /// Visits a bounded coordinate range without scanning unrelated pages.
    pub fn range<R: RangeBounds<Key>>(
        &self,
        bounds: R,
    ) -> impl DoubleEndedIterator<Item = (Key, &Cell)> {
        let lower = match bounds.start_bound() {
            Bound::Included(key) | Bound::Excluded(key) => {
                Bound::Included(Self::coordinate(*key).0)
            }
            Bound::Unbounded => Bound::Unbounded,
        };
        let upper = match bounds.end_bound() {
            Bound::Included(key) | Bound::Excluded(key) => {
                Bound::Included(Self::coordinate(*key).0)
            }
            Bound::Unbounded => Bound::Unbounded,
        };
        self.pages
            .iter()
            .flat_map(move |pages| pages.range((lower, upper)))
            .flat_map(Self::page_entries)
            .filter(move |(key, _)| bounds.contains(key))
    }
}

impl PartialEq for Cells {
    fn eq(&self, other: &Self) -> bool {
        if self.shares(other) {
            return true;
        }
        if self.len != other.len {
            return false;
        }
        if self.len == 0 {
            return true;
        }
        let (Some(left), Some(right)) = (&self.pages, &other.pages) else {
            return false;
        };
        if left.len() != right.len() {
            return false;
        }
        for ((key, page), (other_key, other_page)) in left.iter().zip(right.iter()) {
            if key != other_key {
                return false;
            }
            if Arc::ptr_eq(page, other_page) {
                continue;
            }
            if page.len() != other_page.len() {
                return false;
            }
            for (left, right) in page.iter().zip(other_page.iter()) {
                if left != right {
                    return false;
                }
            }
        }
        true
    }
}

impl Eq for Cells {}

impl Hash for Cells {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        for (key, cell) in self.iter() {
            key.hash(state);
            cell.hash(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hash::DefaultHasher;

    use super::*;

    /// Builds a coordinate oracle independently of the page representation.
    fn fixture() -> BTreeMap<Key, Cell> {
        let mut cells = BTreeMap::new();
        for root in [Root::Symbol, Root::Number, Root::Stack, Root::StackFrame(3)] {
            for offset in [
                i64::MIN,
                i64::MIN + 1,
                -129,
                -128,
                -127,
                -1,
                0,
                1,
                127,
                128,
                i64::MAX - 1,
                i64::MAX,
            ] {
                cells.insert((root, offset), Cell::Byte(offset as u8));
            }
        }
        cells
    }

    /// Signed boundaries, root transitions and excluded bounds match a flat map.
    #[test]
    fn coordinate_order_and_ranges_match_independent_flat_map() {
        let expected = fixture();
        let mut cells = Cells::default();
        for (key, cell) in &expected {
            cells.insert(*key, *cell);
        }
        let pairs = |(key, cell): (&Key, &Cell)| (*key, *cell);
        assert_eq!(
            cells
                .iter()
                .map(|(key, cell)| (key, *cell))
                .collect::<Vec<_>>(),
            expected.iter().map(pairs).collect::<Vec<_>>()
        );
        let keys: Vec<_> = expected.keys().copied().collect();
        for (index, start) in keys.iter().enumerate() {
            for end in &keys[index..] {
                for lower in [
                    Bound::Unbounded,
                    Bound::Included(*start),
                    Bound::Excluded(*start),
                ] {
                    for upper in [
                        Bound::Unbounded,
                        Bound::Included(*end),
                        Bound::Excluded(*end),
                    ] {
                        if lower == Bound::Excluded(*end) && upper == Bound::Excluded(*end) {
                            continue;
                        }
                        let bounds = (lower, upper);
                        let expected: Vec<_> = expected.range(bounds).map(pairs).collect();
                        assert_eq!(
                            cells
                                .range(bounds)
                                .map(|(key, cell)| (key, *cell))
                                .collect::<Vec<_>>(),
                            expected
                        );
                        assert_eq!(
                            cells
                                .range(bounds)
                                .rev()
                                .map(|(key, cell)| (key, *cell))
                                .collect::<Vec<_>>(),
                            expected.into_iter().rev().collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }

    /// Mutations through several snapshots preserve all previous logical contents.
    #[test]
    fn writes_and_removals_preserve_snapshot_isolation_and_logical_hash() {
        let expected = fixture();
        let mut cells = Cells::default();
        for (key, cell) in &expected {
            cells.insert(*key, *cell);
        }
        let snapshot = cells.clone();
        let mut second = cells.clone();
        let mut independent = Cells::default();
        for (key, cell) in &expected {
            independent.insert(*key, *cell);
        }
        assert!(!cells.shares(&independent));
        let hash = |cells: &Cells| {
            let mut hasher = DefaultHasher::new();
            cells.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash(&cells), hash(&independent));
        for (index, key) in expected.keys().enumerate() {
            if index % 2 == 0 {
                cells.remove(key);
            } else {
                cells.insert(*key, Cell::StackBits);
            }
            second.insert(*key, Cell::Byte(0x77));
        }
        assert_eq!(snapshot, independent);
        assert!(snapshot.shares(&snapshot.clone()));
        assert_eq!(cells.len, expected.len() / 2);
        assert_eq!(second.len, expected.len());
        for (index, key) in expected.keys().enumerate() {
            assert_eq!(
                cells.get(key),
                if index % 2 == 0 {
                    None
                } else {
                    Some(&Cell::StackBits)
                }
            );
            assert_eq!(second.get(key), Some(&Cell::Byte(0x77)));
        }
        for key in expected.keys() {
            cells.remove(key);
            second.remove(key);
        }

        assert_eq!(second, Cells::default());
        assert_eq!(snapshot, independent);
    }

    /// A mutation copies its page while retaining the neighboring shared page.
    #[test]
    fn leaf_mutation_preserves_neighboring_shared_storage() {
        let mut original = Cells::default();
        for slot in 0..256i64 {
            original.insert((Root::Stack, slot), Cell::Byte(slot as u8));
        }
        let mut changed = original.clone();
        changed.insert((Root::Stack, 42), Cell::StackBits);
        assert!(!changed.shares_page(&original, (Root::Stack, 0)));
        assert!(changed.shares_page(&original, (Root::Stack, 1)));
        assert_eq!(changed.get(&(Root::Stack, 42)), Some(&Cell::StackBits));
        assert_eq!(original.get(&(Root::Stack, 42)), Some(&Cell::Byte(42)));
    }

    /// Restoring a changed byte restores agreement without changing snapshot storage.
    #[test]
    fn agreement_reuses_shared_pages_and_checks_the_changed_leaf() {
        let mut original = Cells::default();
        for slot in 0..8192i64 {
            original.insert((Root::Stack, slot), Cell::Byte(slot as u8));
        }
        let mut changed = original.clone();
        changed.insert((Root::Stack, 0), Cell::StackBits);
        changed.insert((Root::Stack, 0), Cell::Byte(0));
        assert!(!changed.shares(&original));
        assert_eq!(original, changed);
        changed.insert((Root::Stack, 8191), Cell::StackBits);
        assert_ne!(original, changed);
        assert_eq!(original.get(&(Root::Stack, 8191)), Some(&Cell::Byte(255)));
    }
}
