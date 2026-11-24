//! Dense storage keyed by [`SsaVarId`].
//!
//! [`SsaVarId`] is a dense per-function index, so the natural storage for
//! "a value per variable" or "a set of variables" is a flat array - not a hash
//! map. [`VarSet`] and [`VarMap`] provide that, and encapsulate the two rules
//! that make it safe.
//!
//! # Why these exist
//!
//! The first rule is that [`SsaVarId::PLACEHOLDER`] is **not a variable**. It is
//! a sentinel meaning "no id assigned yet", used while phi placement runs before
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
//! The second is that **a far id is not a reason to size anything**. Malformed
//! or deserialized IR can name any 32-bit id, and a container that grows to hold
//! it asks for gigabytes the way the placeholder does. Both containers size
//! their dense part by the count they are made for and hold an id at or above
//! it in an ordered part, one entry each.
//!
//! Both were reachable in practice. Rather than repeat the checks at every call
//! site - where a new site silently reintroduces the bug - the rules live here:
//! a placeholder is ignored on write and never present on read, and a far id
//! costs one entry. Callers pass [`SsaVarId`] values directly and cannot forget.
//!
//! Sizes come from
//! [`SsaFunction::var_id_bound`](crate::ir::function::SsaFunction::var_id_bound),
//! the allocator's high-water mark, which a malformed id cannot inflate.

use std::{
    array,
    collections::{BTreeMap, BTreeSet, btree_map},
    iter::Enumerate,
    slice,
    sync::Arc,
    vec,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{BitSet, ir::variable::SsaVarId};

/// A set of [`SsaVarId`]s: a [`BitSet`] over the ids below the count it was made
/// for, and an ordered set for the ids at or above it.
///
/// Replaces the `BitSet::new(ssa.var_id_capacity())` /
/// `set.insert(var.index())` pairing, which required every call site to
/// remember that [`SsaVarId::PLACEHOLDER`] must not be inserted and sized the
/// bits by the largest id the function names. Membership of a placeholder is
/// always `false`, and inserting one is a no-op.
///
/// An id at or above the count is malformed or deserialized IR that a scan has
/// to tolerate, not size an allocation for: it is held in the ordered set and
/// costs one entry. Either way it is held, so a write is never lost and the
/// variable never reads back as absent. Iteration is in ascending id order
/// across both parts, and two sets holding the same ids compare equal however
/// each was sized; a [`Default`] set holds every id in the ordered set.
#[derive(Debug, Clone)]
pub struct VarSet {
    /// One bit per variable index below the count the set was made for.
    bits: BitSet,
    /// The ids at or above [`Self::bits`]'s capacity.
    beyond: BTreeSet<SsaVarId>,
}

impl VarSet {
    /// Creates an empty set with a bit for each id below `count`.
    ///
    /// Size it from
    /// [`SsaFunction::var_id_bound`](crate::ir::function::SsaFunction::var_id_bound).
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            bits: BitSet::new(count),
            beyond: BTreeSet::new(),
        }
    }

    /// Adds `var` to the set, returning `true` if it was not already present.
    ///
    /// Returns `false` for [`SsaVarId::PLACEHOLDER`], which is not a variable
    /// this set can describe.
    ///
    /// An id at or above the set's count is held beyond the bits rather than
    /// dropped. These sets are sized from `var_id_bound()`, but malformed IR -
    /// an op naming a variable the allocator never issued - can carry a larger
    /// id. Silently ignoring such a write makes the variable read back as
    /// *absent*, and for a liveness or used-variable set absent means "dead", so
    /// the value would be dropped by the very pass that should have kept it.
    ///
    /// Holding it beyond the bits is also why this does not grow them: the
    /// crate's inputs are attacker-controlled binaries, and an id of three
    /// billion would otherwise size the bits by hundreds of megabytes.
    pub fn insert(&mut self, var: SsaVarId) -> bool {
        if var.is_placeholder() {
            return false;
        }
        let index = var.index();
        if index < self.bits.len() {
            self.bits.insert(index)
        } else {
            self.beyond.insert(var)
        }
    }

    /// Returns `true` when `var` is in the set.
    #[must_use]
    pub fn contains(&self, var: SsaVarId) -> bool {
        if var.is_placeholder() {
            return false;
        }
        let index = var.index();
        if index < self.bits.len() {
            self.bits.contains(index)
        } else {
            self.beyond.contains(&var)
        }
    }

    /// Removes `var` from the set, returning `true` if it was present.
    pub fn remove(&mut self, var: SsaVarId) -> bool {
        if !self.contains(var) {
            return false;
        }
        let index = var.index();
        if index < self.bits.len() {
            self.bits.remove(index);
            true
        } else {
            self.beyond.remove(&var)
        }
    }

    /// Returns the number of variables in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bits.count().saturating_add(self.beyond.len())
    }

    /// Returns `true` when the set holds no variables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.beyond.is_empty() && self.bits.is_empty()
    }

    /// Iterates the set's variables in ascending index order.
    pub fn iter(&self) -> impl Iterator<Item = SsaVarId> + '_ {
        self.bits
            .iter()
            .map(SsaVarId::from_index)
            .chain(self.beyond.iter().copied())
    }
}

impl Default for VarSet {
    fn default() -> Self {
        Self::new(0)
    }
}

impl PartialEq for VarSet {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for VarSet {}

/// A map from [`SsaVarId`] to `T`, held in a slot per variable.
///
/// The dense counterpart to a `HashMap<SsaVarId, T>` or a `BTreeMap`.
/// `SsaVarId` is an index, so hashing it is pure overhead - the default
/// `RandomState` runs SipHash on every insert and lookup - and a tree holds its
/// entries in nodes that are partly empty and carry their own bookkeeping,
/// which for small entries is most of what they cost. A slot per id costs the
/// entry and the slots of the ids between, and finds it without a search.
///
/// # Dense and beyond
///
/// The ids below the count the map was made for index a vector of slots. The
/// vector ends at the highest of them held and grows by exactly what an id
/// needs, so no slack outlives a growth. An id at or above the count is
/// malformed or deserialized IR that a scan has to tolerate, not size an
/// allocation for: it is held in an ordered map and costs one entry. Either way
/// it is held, and a write is never lost.
///
/// Iteration is in ascending id order across both parts, as an ordered map
/// iterates, and two maps holding the same entries compare equal however each
/// was made.
///
/// A map made without a count has none to size by. One collected from entries
/// or deserialized keeps the ids below twice its entry count in the vector,
/// which holds dense ids and sends a far one to the ordered map; a [`Default`]
/// map holds everything in the ordered map.
///
/// Follows the same placeholder rule as [`VarSet`]: writing under
/// [`SsaVarId::PLACEHOLDER`] is a no-op and reading it yields `None`.
#[derive(Debug, Clone)]
pub struct VarMap<T> {
    /// The entry of variable `i` at slot `i`, for the ids below [`Self::dense`];
    /// ends at the highest id held.
    slots: Vec<Option<T>>,
    /// The ids below this index the slots; the ones at or above it are in
    /// [`Self::beyond`].
    dense: usize,
    /// The entries of the ids at or above [`Self::dense`].
    beyond: BTreeMap<SsaVarId, T>,
    /// How many entries are held.
    len: usize,
}

impl<T> VarMap<T> {
    /// Creates an empty map for the ids below `count`.
    ///
    /// Size it from
    /// [`SsaFunction::var_id_bound`](crate::ir::function::SsaFunction::var_id_bound).
    /// Room for `count` slots is reserved exactly, so a map that fills up does
    /// not grow.
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            slots: Vec::with_capacity(count),
            dense: count,
            beyond: BTreeMap::new(),
            len: 0,
        }
    }

    /// Reserves an empty union's dense span without promoting far ids to slots.
    ///
    /// The smaller dense bound keeps every overflow id in overflow storage.
    /// Dense capacity covers the existing spans under that bound, so ordered
    /// insertion initializes holes without repeated reallocation.
    pub fn empty_union(&self, other: &Self) -> Self {
        let dense = self.dense.min(other.dense);
        let capacity = self.slots.len().max(other.slots.len()).min(dense);

        Self {
            slots: Vec::with_capacity(capacity),
            dense,
            beyond: BTreeMap::new(),
            len: 0,
        }
    }

    /// Returns the number of entries held.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no entry is held.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the entry for `var`, or `None` if it has none.
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> Option<&T> {
        if var.is_placeholder() {
            return None;
        }
        let index = var.index();
        if index >= self.dense {
            return self.beyond.get(&var);
        }
        self.slots.get(index).and_then(Option::as_ref)
    }

    /// Returns a mutable reference to `var`'s entry, or `None` if it has none.
    ///
    /// Follows the same rules as [`Self::get`]: [`SsaVarId::PLACEHOLDER`] has no
    /// entry.
    pub fn get_mut(&mut self, var: SsaVarId) -> Option<&mut T> {
        if var.is_placeholder() {
            return None;
        }
        let index = var.index();
        if index >= self.dense {
            return self.beyond.get_mut(&var);
        }
        self.slots.get_mut(index).and_then(Option::as_mut)
    }

    /// Returns `true` when `var` has an entry.
    #[must_use]
    pub fn contains(&self, var: SsaVarId) -> bool {
        self.get(var).is_some()
    }

    /// Inserts `value` for `var`, returning the entry it replaces.
    ///
    /// Ignores [`SsaVarId::PLACEHOLDER`], which is not a variable: the value is
    /// dropped and nothing is replaced. An id at or above the map's count is
    /// held beyond the slots, not dropped.
    pub fn insert(&mut self, var: SsaVarId, value: T) -> Option<T> {
        if var.is_placeholder() {
            return None;
        }
        let index = var.index();
        let replaced = if index >= self.dense {
            self.beyond.insert(var, value)
        } else {
            let wanted = index.saturating_add(1);
            if wanted > self.slots.len() {
                self.slots
                    .reserve_exact(wanted.saturating_sub(self.slots.len()));
                self.slots.resize_with(wanted, || None);
            }
            self.slots.get_mut(index)?.replace(value)
        };
        if replaced.is_none() {
            self.len = self.len.saturating_add(1);
        }
        replaced
    }

    /// Inserts `value` only when `var` has no entry yet.
    pub fn or_insert(&mut self, var: SsaVarId, value: T) {
        if !self.contains(var) {
            self.insert(var, value);
        }
    }

    /// Removes the entry for `var`, returning it.
    pub fn remove(&mut self, var: SsaVarId) -> Option<T> {
        if var.is_placeholder() {
            return None;
        }
        let index = var.index();
        let removed = if index >= self.dense {
            self.beyond.remove(&var)
        } else {
            self.slots.get_mut(index)?.take()
        };
        if removed.is_some() {
            self.len = self.len.saturating_sub(1);
        }
        removed
    }

    /// Keeps the entries `keep` answers `true` for, visiting them in ascending
    /// id order.
    pub fn retain(&mut self, mut keep: impl FnMut(SsaVarId, &mut T) -> bool) {
        let mut dropped = 0usize;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if let Some(entry) = slot
                && !keep(SsaVarId::from_index(index), entry)
            {
                *slot = None;
                dropped = dropped.saturating_add(1);
            }
        }
        let held = self.beyond.len();
        self.beyond.retain(|var, entry| keep(*var, entry));
        dropped = dropped.saturating_add(held.saturating_sub(self.beyond.len()));
        self.len = self.len.saturating_sub(dropped);
    }

    /// Returns the entries with their ids, in ascending id order.
    pub fn iter(&self) -> VarMapIter<'_, T> {
        VarMapIter {
            slots: self.slots.iter().enumerate(),
            beyond: self.beyond.iter(),
        }
    }

    /// Returns the ids of the entries held, in ascending order.
    pub fn keys(&self) -> impl DoubleEndedIterator<Item = SsaVarId> + '_ {
        self.iter().map(|(var, _)| var)
    }

    /// Returns these entries transformed by `convert`, each under the id it was
    /// held under.
    ///
    /// The slots carry over as they are, and the result holds no more than they
    /// need.
    #[must_use]
    pub fn map<U>(self, mut convert: impl FnMut(SsaVarId, T) -> U) -> VarMap<U> {
        let mut slots: Vec<Option<U>> = self
            .slots
            .into_iter()
            .enumerate()
            .map(|(index, slot)| slot.map(|entry| convert(SsaVarId::from_index(index), entry)))
            .collect();
        slots.shrink_to_fit();
        VarMap {
            slots,
            dense: self.dense,
            beyond: self
                .beyond
                .into_iter()
                .map(|(var, entry)| (var, convert(var, entry)))
                .collect(),
            len: self.len,
        }
    }
}

impl<T> Default for VarMap<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T: PartialEq> PartialEq for VarMap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<T: Eq> Eq for VarMap<T> {}

/// The entries of a [`VarMap`] with their ids, borrowed.
pub struct VarMapIter<'a, T> {
    /// The slots not yet visited, numbered by id.
    slots: Enumerate<slice::Iter<'a, Option<T>>>,
    /// The entries beyond the slots not yet visited.
    beyond: btree_map::Iter<'a, SsaVarId, T>,
}

impl<'a, T> Iterator for VarMapIter<'a, T> {
    type Item = (SsaVarId, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.slots
            .find_map(|(index, slot)| Some((SsaVarId::from_index(index), slot.as_ref()?)))
            .or_else(|| self.beyond.next().map(|(var, entry)| (*var, entry)))
    }
}

impl<T> DoubleEndedIterator for VarMapIter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some((var, entry)) = self.beyond.next_back() {
            return Some((*var, entry));
        }
        loop {
            let (index, slot) = self.slots.next_back()?;
            if let Some(entry) = slot {
                return Some((SsaVarId::from_index(index), entry));
            }
        }
    }
}

impl<'a, T> IntoIterator for &'a VarMap<T> {
    type Item = (SsaVarId, &'a T);
    type IntoIter = VarMapIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// The entries of a [`VarMap`] with their ids, owned.
pub struct VarMapIntoIter<T> {
    /// The slots not yet visited, numbered by id.
    slots: Enumerate<vec::IntoIter<Option<T>>>,
    /// The entries beyond the slots not yet visited.
    beyond: btree_map::IntoIter<SsaVarId, T>,
}

impl<T> Iterator for VarMapIntoIter<T> {
    type Item = (SsaVarId, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.slots
            .find_map(|(index, slot)| Some((SsaVarId::from_index(index), slot?)))
            .or_else(|| self.beyond.next())
    }
}

impl<T> IntoIterator for VarMap<T> {
    type Item = (SsaVarId, T);
    type IntoIter = VarMapIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        VarMapIntoIter {
            slots: self.slots.into_iter().enumerate(),
            beyond: self.beyond.into_iter(),
        }
    }
}

impl<T> FromIterator<(SsaVarId, T)> for VarMap<T> {
    /// Collects the entries, the last of two under one id winning.
    ///
    /// The ids below twice the number of entries take slots, sized to the
    /// highest of them; a far id, which has no count to be measured against, is
    /// held beyond them.
    fn from_iter<I: IntoIterator<Item = (SsaVarId, T)>>(entries: I) -> Self {
        let entries: Vec<(SsaVarId, T)> = entries
            .into_iter()
            .filter(|(var, _)| !var.is_placeholder())
            .collect();
        let dense = entries.len().saturating_mul(2);
        let top = entries
            .iter()
            .map(|(var, _)| var.index().saturating_add(1))
            .filter(|&wanted| wanted <= dense)
            .max()
            .unwrap_or(0);
        let mut map = Self::new(top);
        map.dense = dense;
        map.extend(entries);
        map
    }
}

impl<T> Extend<(SsaVarId, T)> for VarMap<T> {
    fn extend<I: IntoIterator<Item = (SsaVarId, T)>>(&mut self, entries: I) {
        for (var, entry) in entries {
            self.insert(var, entry);
        }
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for VarMap<T> {
    /// Serializes as the map from id to entry, as an ordered map does.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.iter())
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for VarMap<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(BTreeMap::<SsaVarId, T>::deserialize(deserializer)?
            .into_iter()
            .collect())
    }
}

/// Low id bits that select a slot inside one [`SharedVarMap`] page.
const SHARED_PAGE_BITS: u32 = 5;

/// Slots one [`SharedVarMap`] page holds.
const SHARED_PAGE_SLOTS: usize = 1_usize.wrapping_shl(SHARED_PAGE_BITS);

/// The low id bits that select a slot inside one page.
const SHARED_PAGE_MASK: usize = SHARED_PAGE_SLOTS.wrapping_sub(1);

/// The slots of one [`SharedVarMap`] page: consecutive ids from a multiple of
/// [`SHARED_PAGE_SLOTS`].
type SharedPage<T> = [Option<T>; SHARED_PAGE_SLOTS];

/// A map from [`SsaVarId`] to `T` whose copies share their storage until one
/// of them writes.
///
/// A [`VarMap`] holds a slot per id, so copying one costs every slot up to the
/// highest id held. A map that is copied far more often than it is written -
/// one activation's values in an execution state that is snapshot at every
/// block a path reaches - pays that on each copy. This map keeps its slots in
/// pages of 32 ids behind a shared page table: a clone
/// shares the table and every page, and the first write after it copies the
/// table and the one page it lands in. A read finds its entry by two indexes.
///
/// The rules are [`VarMap`]'s. The ids below the count the map was made for
/// are paged, and a page is made only when an id in it is written; an id at or
/// above the count is held in an ordered part, one entry each. Writing under
/// [`SsaVarId::PLACEHOLDER`] is a no-op and reading it yields `None`.
/// Iteration is in ascending id order, and two maps holding the same entries
/// compare equal however each was made.
#[derive(Debug, Clone)]
pub struct SharedVarMap<T> {
    /// The pages by number, `id >> SHARED_PAGE_BITS`, for the ids below
    /// [`Self::dense`]; ends at the highest page written.
    pages: Arc<Vec<Option<Arc<SharedPage<T>>>>>,
    /// The ids below this index are paged; the ones at or above it are in
    /// [`Self::beyond`].
    dense: usize,
    /// The entries of the ids at or above [`Self::dense`].
    beyond: Arc<BTreeMap<SsaVarId, T>>,
    /// How many entries are held.
    len: usize,
}

impl<T: Clone> SharedVarMap<T> {
    /// Creates an empty map for the ids below `count`.
    ///
    /// Size it from
    /// [`SsaFunction::var_id_bound`](crate::ir::function::SsaFunction::var_id_bound).
    /// Nothing is allocated until an entry is written.
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            pages: Arc::new(Vec::new()),
            dense: count,
            beyond: Arc::new(BTreeMap::new()),
            len: 0,
        }
    }

    /// Returns an empty map for the entries of this map and `other` together.
    ///
    /// The smaller count bounds the paged ids, so an id either map holds
    /// beyond its count stays in the ordered part.
    #[must_use]
    pub fn empty_union(&self, other: &Self) -> Self {
        Self::new(self.dense.min(other.dense))
    }

    /// Returns the number of entries held.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no entry is held.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the entry for `var`, or `None` if it has none.
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> Option<&T> {
        if var.is_placeholder() {
            return None;
        }
        let index = var.index();
        if index >= self.dense {
            return self.beyond.get(&var);
        }
        self.pages
            .get(index >> SHARED_PAGE_BITS)?
            .as_ref()?
            .get(index & SHARED_PAGE_MASK)?
            .as_ref()
    }

    /// Inserts `value` for `var`, returning the entry it replaces.
    ///
    /// Copies the page table and the page `var` lands in when another map
    /// shares them. Ignores [`SsaVarId::PLACEHOLDER`], which is not a variable:
    /// the value is dropped and nothing is replaced. An id at or above the
    /// map's count is held in the ordered part, not dropped.
    pub fn insert(&mut self, var: SsaVarId, value: T) -> Option<T> {
        if var.is_placeholder() {
            return None;
        }
        let index = var.index();
        let replaced = if index >= self.dense {
            Arc::make_mut(&mut self.beyond).insert(var, value)
        } else {
            let number = index >> SHARED_PAGE_BITS;
            let pages = Arc::make_mut(&mut self.pages);
            if number >= pages.len() {
                pages.resize_with(number.saturating_add(1), || None);
            }
            let page = pages
                .get_mut(number)?
                .get_or_insert_with(|| Arc::new(array::from_fn(|_| None)));
            Arc::make_mut(page)
                .get_mut(index & SHARED_PAGE_MASK)?
                .replace(value)
        };
        if replaced.is_none() {
            self.len = self.len.saturating_add(1);
        }
        replaced
    }

    /// Iterates the entries in ascending id order.
    pub fn iter(&self) -> impl Iterator<Item = (SsaVarId, &T)> + '_ {
        self.pages
            .iter()
            .enumerate()
            .filter_map(|(number, page)| {
                Some((number.wrapping_shl(SHARED_PAGE_BITS), page.as_ref()?))
            })
            .flat_map(|(first, page)| {
                page.iter().enumerate().filter_map(move |(slot, entry)| {
                    Some((SsaVarId::from_index(first | slot), entry.as_ref()?))
                })
            })
            .chain(self.beyond.iter().map(|(var, entry)| (*var, entry)))
    }
}

impl<T: Clone> Default for SharedVarMap<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T: Clone + PartialEq> PartialEq for SharedVarMap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<T: Clone + Eq> Eq for SharedVarMap<T> {}

#[cfg(test)]
mod tests {
    use std::process;

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

    /// The placeholder is ignored by a small set and map as well.
    #[test]
    fn placeholder_ids_are_ignored() {
        let mut set = VarSet::new(4);
        assert!(!set.insert(SsaVarId::PLACEHOLDER));
        assert!(!set.contains(SsaVarId::PLACEHOLDER));

        let mut map: VarMap<u32> = VarMap::new(4);
        map.insert(SsaVarId::PLACEHOLDER, 1);
        assert_eq!(map.get(SsaVarId::PLACEHOLDER), None);
    }

    /// Reading an out-of-range id is tolerated - a query about a variable the
    /// container does not hold answers "absent". Writing one to a [`VarSet`]
    /// holds it; see [`VarSet::insert`].
    #[test]
    fn out_of_range_reads_are_tolerated() {
        let set = VarSet::new(4);
        assert!(!set.contains(SsaVarId::from_index(9)));

        let map: VarMap<u32> = VarMap::new(4);
        assert_eq!(map.get(SsaVarId::from_index(9)), None);
    }

    /// A set sized from `var_id_bound()` cannot give a larger id a bit, and
    /// malformed IR can carry one. Dropping the write would make the variable
    /// read back as absent - "dead", for a liveness or used-variable set - so
    /// the id is held beyond the bits instead.
    #[test]
    fn inserting_an_out_of_range_id_holds_it() {
        let mut set = VarSet::new(4);
        assert!(set.insert(SsaVarId::from_index(9)));
        assert!(
            set.contains(SsaVarId::from_index(9)),
            "the write must not be silently lost"
        );
        assert_eq!(set.len(), 1);

        // Existing entries survive, and iteration is ascending across both parts.
        let mut set = VarSet::new(4);
        assert!(set.insert(SsaVarId::from_index(300)));
        assert!(set.insert(SsaVarId::from_index(1)));
        assert!(!set.insert(SsaVarId::from_index(300)));
        assert!(set.contains(SsaVarId::from_index(1)));
        assert!(set.contains(SsaVarId::from_index(300)));
        assert_eq!(set.len(), 2);
        assert_eq!(
            set.iter().collect::<Vec<_>>(),
            vec![SsaVarId::from_index(1), SsaVarId::from_index(300)]
        );
        assert!(set.remove(SsaVarId::from_index(300)));
        assert!(!set.remove(SsaVarId::from_index(300)));
        assert!(!set.contains(SsaVarId::from_index(300)));
        assert_eq!(set.len(), 1);
    }

    /// Two sets holding the same ids are equal whatever count each was made
    /// for, the default's ordered part included; a set holding another id is
    /// not.
    #[test]
    fn sets_compare_by_their_ids() {
        let mut small = VarSet::new(2);
        let mut large = VarSet::new(64);
        let mut default = VarSet::default();
        for index in [1, 5, 40] {
            small.insert(SsaVarId::from_index(index));
            large.insert(SsaVarId::from_index(index));
            default.insert(SsaVarId::from_index(index));
        }
        assert_eq!(small, large);
        assert_eq!(large, default);
        assert_eq!(VarSet::new(8), VarSet::default());
        large.insert(SsaVarId::from_index(6));
        assert_ne!(small, large);
        small.insert(SsaVarId::from_index(7));
        assert_ne!(small, large, "as many ids, not the same ones");
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

    /// An id far above any variable count a test builds.
    const FAR: usize = 3_000_000_000;

    /// The map and the ordered map it stands in for, held side by side.
    fn same(map: &VarMap<u32>, model: &BTreeMap<SsaVarId, u32>) -> bool {
        map.len() == model.len()
            && map.is_empty() == model.is_empty()
            && map
                .iter()
                .map(|(var, entry)| (var, *entry))
                .eq(model.iter().map(|(var, entry)| (*var, *entry)))
            && map.keys().rev().eq(model.keys().rev().copied())
    }

    /// The placeholder has no entry in any operation: writes drop it, reads
    /// miss it, and a real id beside it is held as usual.
    #[test]
    fn a_map_ignores_the_placeholder_everywhere() {
        let placeholder = SsaVarId::PLACEHOLDER;
        let mut map: VarMap<u32> = VarMap::new(4);
        assert_eq!(map.insert(placeholder, 1), None);
        map.or_insert(placeholder, 2);
        map.extend([(placeholder, 3), (SsaVarId::from_index(1), 4)]);
        assert_eq!(map.len(), 1, "only the real id was written");
        assert_eq!(map.get(placeholder), None);
        assert!(map.get_mut(placeholder).is_none());
        assert!(!map.contains(placeholder));
        assert_eq!(map.remove(placeholder), None);
        assert_eq!(
            map.keys().collect::<Vec<_>>(),
            vec![SsaVarId::from_index(1)]
        );

        let collected: VarMap<u32> = [(placeholder, 5), (SsaVarId::from_index(0), 6)]
            .into_iter()
            .collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected.get(placeholder), None);
        assert_eq!(collected.get(SsaVarId::from_index(0)), Some(&6));
    }

    /// Any run of inserts, removals and retains over dense, far and placeholder
    /// ids leaves the map holding what an ordered map holds, in the order it
    /// iterates.
    #[test]
    fn a_map_holds_what_an_ordered_map_holds() {
        let mut map = VarMap::new(16);
        let mut model = BTreeMap::new();
        let mut state = 0x2545_f491_u32;
        for step in 0..6000_u32 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let var = match state % 11 {
                0 => SsaVarId::PLACEHOLDER,
                1 | 2 => SsaVarId::from_index(FAR + (state >> 8) as usize % 5),
                _ => SsaVarId::from_index((state >> 4) as usize % 40),
            };
            match state % 7 {
                0..=2 => {
                    let replaced = map.insert(var, step);
                    let expected = if var.is_placeholder() {
                        None
                    } else {
                        model.insert(var, step)
                    };
                    assert_eq!(replaced, expected);
                }
                3 | 4 => assert_eq!(
                    map.remove(var),
                    if var.is_placeholder() {
                        None
                    } else {
                        model.remove(&var)
                    }
                ),
                5 => {
                    let modulus = state % 5 + 2;
                    map.retain(|_, entry| *entry % modulus != 0);
                    model.retain(|_, entry| *entry % modulus != 0);
                }
                _ => assert_eq!(map.get(var), model.get(&var)),
            }
            assert!(same(&map, &model), "diverged at step {step}");
        }
        assert!(!model.is_empty(), "the run ends holding entries");
    }

    /// The comparison above notices a map that differs from the model, so a
    /// passing run means the two agreed.
    #[test]
    fn the_comparison_notices_a_difference() {
        let mut map: VarMap<u32> = [(SsaVarId::from_index(3), 1), (SsaVarId::from_index(FAR), 2)]
            .into_iter()
            .collect();
        let model: BTreeMap<SsaVarId, u32> =
            [(SsaVarId::from_index(3), 1), (SsaVarId::from_index(FAR), 2)]
                .into_iter()
                .collect();
        assert!(same(&map, &model));
        map.insert(SsaVarId::from_index(FAR), 3);
        assert!(!same(&map, &model));
        map.insert(SsaVarId::from_index(FAR), 2);
        map.insert(SsaVarId::from_index(4), 0);
        assert!(!same(&map, &model));
    }

    /// An id at or above the count is held beyond the slots, one entry each,
    /// and the slots are not sized by it. The control is an id inside the
    /// count, which takes a slot.
    #[test]
    fn a_far_id_is_held_beyond_the_slots() {
        let mut map: VarMap<u32> = VarMap::new(4);
        map.insert(SsaVarId::from_index(FAR), 7);
        map.insert(SsaVarId::from_index(4), 8);
        assert_eq!(map.get(SsaVarId::from_index(FAR)), Some(&7));
        assert_eq!(map.get(SsaVarId::from_index(4)), Some(&8));
        assert_eq!(map.slots.len(), 0, "no slot was made for either");
        assert_eq!(map.beyond.len(), 2);

        map.insert(SsaVarId::from_index(2), 9);
        assert_eq!(map.slots.len(), 3, "an id inside the count takes a slot");
        assert_eq!(map.beyond.len(), 2);
        assert_eq!(map.remove(SsaVarId::from_index(FAR)), Some(7));
        assert_eq!(map.get(SsaVarId::from_index(FAR)), None);
        assert_eq!(map.len(), 2);
    }

    /// Slots grow by what an id needs and no more. The control is a map made
    /// for a count, which holds room for all of them from the start.
    #[test]
    fn slots_grow_exactly() {
        let mut map: VarMap<u32> = VarMap::new(0);
        map.dense = 100;
        for index in 0..30 {
            map.insert(SsaVarId::from_index(index), 1);
            assert_eq!(map.slots.capacity(), map.slots.len());
        }
        let mut mapped = map.map(|_, entry| u64::from(entry));
        assert_eq!(mapped.slots.capacity(), mapped.slots.len());
        mapped.insert(SsaVarId::from_index(50), 2);
        assert_eq!(mapped.slots.len(), 51);
        assert_eq!(mapped.slots.capacity(), 51);

        let mut sized: VarMap<u32> = VarMap::new(100);
        sized.insert(SsaVarId::from_index(3), 1);
        assert_eq!(sized.slots.len(), 4);
        assert!(sized.slots.capacity() >= 100, "room was reserved up front");
    }

    /// A map collected from entries keeps the dense ids in slots and a far one
    /// beyond, and equals a map holding the same entries made another way.
    #[test]
    fn a_collected_map_sizes_by_its_entries() {
        let collected: VarMap<u32> = [
            (SsaVarId::from_index(2), 7),
            (SsaVarId::from_index(5), 8),
            (SsaVarId::from_index(FAR), 9),
        ]
        .into_iter()
        .collect();
        assert_eq!(collected.slots.len(), 6);
        assert_eq!(collected.beyond.len(), 1);
        assert_eq!(collected.len(), 3);

        let mut built: VarMap<u32> = VarMap::new(40);
        built.insert(SsaVarId::from_index(FAR), 9);
        built.insert(SsaVarId::from_index(5), 8);
        built.insert(SsaVarId::from_index(2), 7);
        built.insert(SsaVarId::from_index(30), 1);
        built.remove(SsaVarId::from_index(30));
        assert_eq!(built, collected);
        built.insert(SsaVarId::from_index(30), 1);
        assert_ne!(built, collected);

        let repeated: VarMap<u32> = [(SsaVarId::from_index(1), 1), (SsaVarId::from_index(1), 2)]
            .into_iter()
            .collect();
        assert_eq!(
            repeated.get(SsaVarId::from_index(1)),
            Some(&2),
            "the last entry wins"
        );
        assert_eq!(repeated.len(), 1);
    }

    /// Entries come out in ascending id order across the slots and the far
    /// ids, forwards and backwards, borrowed and owned.
    #[test]
    fn iteration_is_ascending_across_both_parts() {
        let mut map: VarMap<usize> = VarMap::new(8);
        for index in [FAR + 1, 6, FAR, 1, 9, 3] {
            map.insert(SsaVarId::from_index(index), index);
        }
        let ascending = [1, 3, 6, 9, FAR, FAR + 1].map(SsaVarId::from_index);
        assert_eq!(map.keys().collect::<Vec<_>>(), ascending);
        assert_eq!(
            map.keys().rev().collect::<Vec<_>>(),
            ascending.into_iter().rev().collect::<Vec<_>>()
        );
        assert_eq!(
            map.clone()
                .into_iter()
                .map(|(var, _)| var)
                .collect::<Vec<_>>(),
            ascending
        );
        let converted = map.map(|var, entry| (var.index(), entry));
        assert_eq!(converted.get(SsaVarId::from_index(FAR)), Some(&(FAR, FAR)));
        assert_eq!(converted.get(SsaVarId::from_index(6)), Some(&(6, 6)));
        assert_eq!(converted.len(), 6);
    }

    /// Entry comparisons preserve dense holes and far keys and propagate predicate failures.
    #[test]
    fn entry_comparison_preserves_dense_and_sparse_storage() {
        let mut source = VarMap::new(4096);
        let high = SsaVarId::from_index(4095);
        source.insert(high, 17u32);
        source.remove(high);
        source.insert(SsaVarId::from_index(0), 9);
        source.insert(SsaVarId::from_index(FAR), 23);
        let copy = source.clone();
        assert_eq!(copy, source);
        assert_eq!(copy.slots.len(), source.slots.len());
        assert_eq!(copy.beyond.len(), source.beyond.len());
        assert_eq!(source, copy);
        let mut different = copy;
        different.insert(SsaVarId::from_index(4096), 5);
        assert_ne!(source, different);
    }

    /// A copy of a shared map shares its page table and pages; a write to the
    /// copy copies the table and the one page it lands in, and the original
    /// keeps what it held. The control is a page the write does not touch,
    /// which both still share.
    #[test]
    fn a_shared_map_copies_only_the_page_a_write_lands_in() {
        let mut map: SharedVarMap<u32> = SharedVarMap::new(1024);
        for index in [0, 31, 32, 700] {
            map.insert(SsaVarId::from_index(index), 1);
        }
        let mut copy = map.clone();
        assert!(
            Arc::ptr_eq(&map.pages, &copy.pages),
            "a copy shares the table"
        );

        assert_eq!(copy.insert(SsaVarId::from_index(33), 2), None);
        assert!(
            !Arc::ptr_eq(&map.pages, &copy.pages),
            "a write copies the table"
        );
        let page = |of: &SharedVarMap<u32>, number: usize| {
            of.pages
                .get(number)
                .and_then(Option::clone)
                .unwrap_or_else(|| process::abort())
        };
        assert!(Arc::ptr_eq(&page(&map, 0), &page(&copy, 0)));
        assert!(Arc::ptr_eq(&page(&map, 21), &page(&copy, 21)));
        assert!(!Arc::ptr_eq(&page(&map, 1), &page(&copy, 1)));
        assert_eq!(map.get(SsaVarId::from_index(33)), None);
        assert_eq!(copy.get(SsaVarId::from_index(33)), Some(&2));
        assert_eq!(copy.get(SsaVarId::from_index(32)), Some(&1));
        assert_eq!((map.len(), copy.len()), (4, 5));
    }

    /// A shared map keeps [`VarMap`]'s rules: the placeholder is never held,
    /// an id at or above the count is held beyond the pages without sizing
    /// them, iteration is in ascending id order, and maps holding the same
    /// entries compare equal however each was made.
    #[test]
    fn a_shared_map_keeps_the_var_map_rules() {
        let mut map: SharedVarMap<u32> = SharedVarMap::new(64);
        assert_eq!(map.insert(SsaVarId::PLACEHOLDER, 9), None);
        assert_eq!(map.get(SsaVarId::PLACEHOLDER), None);
        assert!(map.is_empty());

        map.insert(SsaVarId::from_index(FAR), 7);
        assert!(map.pages.is_empty(), "a far id makes no page");
        for index in [40, 3, 63, 0] {
            map.insert(
                SsaVarId::from_index(index),
                u32::try_from(index).unwrap_or(0),
            );
        }
        assert_eq!(map.insert(SsaVarId::from_index(3), 4), Some(3));
        let ids: Vec<usize> = map.iter().map(|(var, _)| var.index()).collect();
        assert_eq!(ids, [0, 3, 40, 63, FAR]);
        assert_eq!(map.len(), 5);

        let mut other: SharedVarMap<u32> = SharedVarMap::new(64);
        for index in [63, 40, 0] {
            other.insert(
                SsaVarId::from_index(index),
                u32::try_from(index).unwrap_or(0),
            );
        }
        other.insert(SsaVarId::from_index(FAR), 7);
        assert_ne!(map, other);
        other.insert(SsaVarId::from_index(3), 4);
        assert_eq!(map, other);
    }
}
