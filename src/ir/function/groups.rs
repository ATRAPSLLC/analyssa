//! The rename group of each variable of a function.
//!
//! Rename groups say which variables are versions of one storage location. They
//! are indexed by variable id like every per-variable table, so they follow the
//! same rule as [`VarMap`](crate::ir::varstore::VarMap): a vector with a slot
//! for each id below the function's variable bound, and an ordered map for an id
//! at or above it. Malformed or deserialized IR can name an id of three billion,
//! and a vector grown to hold its group is gigabytes.

use std::{collections::BTreeMap, fmt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::variable::SsaVarId;

/// The rename group of each variable, [`Self::NONE`] where it has none.
///
/// The ids below the bound a group is set under index a vector, which ends at
/// the highest of them set and grows by the usual doubling as variables are
/// created. A group set for an id at or above the bound is held in an ordered
/// map and costs one entry; the entry moves into the vector if the vector later
/// grows to reach it, so the two parts never both describe one id.
///
/// [`SsaVarId::PLACEHOLDER`] is not a variable: a group set for it is ignored
/// and it reads back as ungrouped.
///
/// The table serializes and prints as the vector alone. The map holds groups of
/// ids the function never allocated, which only a rebuild in progress sets, so
/// they are not part of a function's persisted state.
#[derive(Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
pub struct RenameGroups {
    /// The group of variable `i` at slot `i`, for the ids the vector reaches.
    dense: Vec<u32>,
    /// The groups of the ids past [`Self::dense`], each set under a bound at or
    /// below the id.
    #[cfg_attr(feature = "serde", serde(skip))]
    beyond: BTreeMap<SsaVarId, u32>,
}

impl fmt::Debug for RenameGroups {
    /// Lists the vector, then each id the map holds as `id: group`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        list.entries(&self.dense);
        for (var, group) in &self.beyond {
            list.entry(&format_args!("{var}: {group}"));
        }
        list.finish()
    }
}

impl RenameGroups {
    /// The group of a variable that has none.
    pub const NONE: u32 = u32::MAX;

    /// Creates a table with no groups, with room for `capacity` ids.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            dense: Vec::with_capacity(capacity),
            beyond: BTreeMap::new(),
        }
    }

    /// Creates a table in which each of the `count` ids is ungrouped.
    #[must_use]
    pub fn ungrouped(count: usize) -> Self {
        Self {
            dense: vec![Self::NONE; count],
            beyond: BTreeMap::new(),
        }
    }

    /// Returns the group of `var`, or [`Self::NONE`] when it has none.
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> u32 {
        if var.is_placeholder() {
            return Self::NONE;
        }
        self.dense
            .get(var.index())
            .or_else(|| self.beyond.get(&var))
            .copied()
            .unwrap_or(Self::NONE)
    }

    /// Whether `var` has a slot of its own, set to [`Self::NONE`] or to a group.
    ///
    /// A variable without one has never had its group decided, which is not the
    /// same as having been decided to have none.
    #[must_use]
    pub fn records(&self, var: SsaVarId) -> bool {
        !var.is_placeholder() && (var.index() < self.dense.len() || self.beyond.contains_key(&var))
    }

    /// Sets the group of `var`.
    ///
    /// `bound` is the function's
    /// [`var_id_bound`](crate::ir::function::SsaFunction::var_id_bound): an id below it takes
    /// a slot of the vector, extending it as far as the id needs; any other id
    /// is held in the map. Setting [`Self::NONE`] under an id the map holds
    /// removes the entry.
    pub fn set(&mut self, var: SsaVarId, group: u32, bound: usize) {
        if var.is_placeholder() {
            return;
        }
        let index = var.index();
        if let Some(slot) = self.dense.get_mut(index) {
            *slot = group;
        } else if index < bound {
            self.dense.resize(index.saturating_add(1), Self::NONE);
            // The entries the vector now reaches move in first, so the group
            // being set replaces any the id held beyond it.
            self.absorb_beyond();
            if let Some(slot) = self.dense.get_mut(index) {
                *slot = group;
            }
        } else if group == Self::NONE {
            self.beyond.remove(&var);
        } else {
            self.beyond.insert(var, group);
        }
    }

    /// Returns the highest group any variable carries, or `None` when no
    /// variable is grouped.
    #[must_use]
    pub fn max_group(&self) -> Option<u32> {
        self.dense
            .iter()
            .chain(self.beyond.values())
            .copied()
            .filter(|group| *group != Self::NONE)
            .max()
    }

    /// Whether the table holds a slot for exactly the ids below `count` and
    /// nothing beyond them.
    #[must_use]
    pub fn describes(&self, count: usize) -> bool {
        self.dense.len() == count && self.beyond.is_empty()
    }

    /// Releases the room the vector grew past its contents.
    pub fn shrink_to_fit(&mut self) {
        self.dense.shrink_to_fit();
    }

    /// Moves the entries of the map that the vector now reaches into it.
    fn absorb_beyond(&mut self) {
        while let Some(entry) = self.beyond.first_entry()
            && entry.key().index() < self.dense.len()
        {
            let (var, group) = entry.remove_entry();
            if let Some(slot) = self.dense.get_mut(var.index()) {
                *slot = group;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An id far above any variable bound a test uses.
    const FAR: usize = 3_000_000_000;

    #[test]
    fn a_group_reads_back_under_the_id_it_was_set_for() {
        let mut groups = RenameGroups::default();
        groups.set(SsaVarId::from_index(3), 7, 8);
        groups.set(SsaVarId::from_index(FAR), 9, 8);
        assert_eq!(groups.get(SsaVarId::from_index(3)), 7);
        assert_eq!(groups.get(SsaVarId::from_index(FAR)), 9);
        assert_eq!(groups.get(SsaVarId::from_index(2)), RenameGroups::NONE);
        assert_eq!(groups.max_group(), Some(9));
    }

    #[test]
    fn a_far_id_takes_no_slot_of_the_vector() {
        let mut groups = RenameGroups::default();
        groups.set(SsaVarId::from_index(FAR), 4, 8);
        assert_eq!(groups.dense.len(), 0);
        assert!(groups.records(SsaVarId::from_index(FAR)));
        assert!(!groups.describes(8));

        groups.set(SsaVarId::from_index(FAR), RenameGroups::NONE, 8);
        assert!(!groups.records(SsaVarId::from_index(FAR)));
        assert!(groups.describes(0));
    }

    #[test]
    fn the_placeholder_is_not_a_variable() {
        let mut groups = RenameGroups::default();
        groups.set(SsaVarId::PLACEHOLDER, 5, usize::MAX);
        assert_eq!(groups.get(SsaVarId::PLACEHOLDER), RenameGroups::NONE);
        assert!(!groups.records(SsaVarId::PLACEHOLDER));
        assert_eq!(groups.max_group(), None);
        assert!(groups.describes(0));
    }

    /// A group held beyond the vector moves into it when the vector grows to
    /// reach the id, so the id keeps its group and one slot describes it.
    #[test]
    fn the_vector_absorbs_a_group_it_grows_to_reach() {
        let mut groups = RenameGroups::default();
        groups.set(SsaVarId::from_index(6), 3, 4);
        assert_eq!(groups.dense.len(), 0);
        groups.set(SsaVarId::from_index(8), 1, 20);
        assert_eq!(groups.dense.len(), 9);
        assert!(groups.beyond.is_empty());
        assert_eq!(groups.get(SsaVarId::from_index(6)), 3);
        assert_eq!(groups.get(SsaVarId::from_index(8)), 1);
        assert_eq!(groups.get(SsaVarId::from_index(7)), RenameGroups::NONE);
    }

    /// The table prints as the vector it holds, so a function prints as it did
    /// when its groups were a vector; an id past the vector follows the list.
    #[test]
    fn the_table_prints_as_its_vector() {
        let mut groups = RenameGroups::default();
        groups.set(SsaVarId::from_index(0), 3, 4);
        groups.set(SsaVarId::from_index(2), 5, 4);
        assert_eq!(
            format!("{groups:?}"),
            format!("{:?}", vec![3, RenameGroups::NONE, 5])
        );
        groups.set(SsaVarId::from_index(FAR), 7, 4);
        assert_eq!(format!("{groups:?}"), "[3, 4294967295, 5, v3000000000: 7]");
    }

    /// The table serializes as the vector, so a function serializes as it did
    /// when its groups were a vector, and the ids past it are not persisted.
    #[cfg(feature = "serde")]
    #[test]
    fn the_table_serializes_as_its_vector() {
        let mut groups = RenameGroups::default();
        groups.set(SsaVarId::from_index(0), 3, 4);
        groups.set(SsaVarId::from_index(2), 5, 4);
        groups.set(SsaVarId::from_index(FAR), 7, 4);
        let text = serde_json::to_string(&groups).unwrap_or_default();
        assert_eq!(text, "[3,4294967295,5]");
        let read: RenameGroups = serde_json::from_str(&text).unwrap_or_default();
        assert_eq!(read.get(SsaVarId::from_index(2)), 5);
        assert_eq!(read.get(SsaVarId::from_index(FAR)), RenameGroups::NONE);
    }

    /// Any run of sets, under bounds that rise and fall, reads back as an
    /// ordered map of the groups set last would.
    #[test]
    fn groups_read_back_what_a_map_holds() {
        let mut groups = RenameGroups::default();
        let mut model: BTreeMap<SsaVarId, u32> = BTreeMap::new();
        let mut state = 0x9e37_79b9_u32;
        let mut bound = 0usize;
        for step in 0..6000_u32 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if state.is_multiple_of(5) {
                bound = bound.saturating_add(1 + (state >> 9) as usize % 3);
            }
            let var = match state % 9 {
                0 => SsaVarId::PLACEHOLDER,
                1 | 2 => SsaVarId::from_index(FAR + (state >> 8) as usize % 5),
                _ => SsaVarId::from_index((state >> 4) as usize % 60),
            };
            let group = if state.is_multiple_of(4) {
                RenameGroups::NONE
            } else {
                step
            };
            groups.set(var, group, bound);
            if !var.is_placeholder() {
                if group == RenameGroups::NONE {
                    model.remove(&var);
                } else {
                    model.insert(var, group);
                }
            }
            let probes = (0..60)
                .map(SsaVarId::from_index)
                .chain((0..5).map(|k| SsaVarId::from_index(FAR + k)));
            for probe in probes {
                assert_eq!(
                    groups.get(probe),
                    model.get(&probe).copied().unwrap_or(RenameGroups::NONE),
                    "diverged at step {step}"
                );
            }
            assert_eq!(groups.max_group(), model.values().copied().max());
        }
    }
}
