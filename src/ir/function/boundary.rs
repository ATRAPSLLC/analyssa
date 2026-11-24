//! SSA construction from block boundaries.
//!
//! A front end that lifts a block at a time can answer every read inside a
//! block itself: it keeps what each storage location holds as it goes, and a
//! read after a write in the same block names the written value directly. What
//! it cannot answer is a read no write in its own block precedes - the value
//! came in from a predecessor, or from the caller. Such a front end also binds
//! values to locations without an operation: after `mov ebx, eax` one value is
//! held in two registers, and no instruction defines `ebx`.
//!
//! [`SsaFunction::construct_at_boundaries`] is construction for that front end.
//! It is told, per block, the reads that reach the block from outside and the
//! writes the block makes, each against a rename group standing for one
//! location; it places the phis the joins need, answers each outside read with
//! the definition that reaches it, and gives every location one value on
//! entry to the function. No copy is made for a value bound to a second
//! location, and none has to be folded away afterwards: the binding is what
//! the front end reports, and construction reads it as the definition it is.
//!
//! [`SsaFunction::construct_ssa`] is the other entry point, for a front end
//! whose every definition is an operation: it reads the versions of a location
//! from the variables of its rename group, and rebuilds the reads from them.

use std::{
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    iter,
};

use crate::{
    BitSet,
    analysis::{
        cfg::SsaCfg,
        phis::{PhiPlacementConfig, place_pruned_phis},
    },
    graph::{
        NodeId,
        algorithms::{DominatorTree, compute_dominance_frontiers, compute_dominators},
    },
    ir::{
        function::SsaFunction,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    passes::utils::resolve_chains,
    target::Target,
};

/// What one block tells construction: the reads it makes of locations it has
/// not written, and the writes it makes, each against a rename group.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockBoundary {
    /// Reads no earlier write in the block answers, as `(group, placeholder)`
    /// in the order they happened. The placeholder is the variable the front
    /// end used for the value; construction replaces it with the definition
    /// that reaches the block.
    pub reads: Vec<(u32, SsaVarId)>,
    /// Writes, as `(group, value)` in the order they happened; a group's last
    /// write is what it holds when the block exits.
    pub writes: Vec<(u32, SsaVarId)>,
}

/// What [`SsaFunction::construct_at_boundaries`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundaryConstruction {
    /// Each placeholder's answer, resolved through any chain: the definition
    /// that reaches its block, or its location's value on entry. A placeholder
    /// that became its location's entry value maps to nothing.
    pub replacements: BTreeMap<SsaVarId, SsaVarId>,
    /// Every phi placed, as `(block, group, result)`.
    pub phis: Vec<(usize, u32, SsaVarId)>,
    /// The entry values construction minted for groups no read named, as
    /// `(group, value)`.
    pub minted_entry_values: Vec<(u32, SsaVarId)>,
    /// Whether the entry block had predecessors and was split off behind a
    /// fresh one first ([`SsaFunction::ensure_entry_has_no_predecessors`]).
    ///
    /// Every block index moved up by one when it was, and every block and
    /// phi this report names is in the new numbering. The boundaries the
    /// caller passed were read in the old one.
    pub entry_split: bool,
}

/// A write a block-at-a-time front end made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedWrite<K> {
    /// The block the write happened in.
    pub block_idx: usize,
    /// The location written.
    pub key: K,
    /// The value it holds from here.
    pub var: SsaVarId,
    /// The write's position in lift order.
    ///
    /// Writes are logged in the order they are recorded, which is not always
    /// the order they happened: a front end that defers a definition (the
    /// registers a call destroys, minted only when a later read needs one)
    /// records it after writes that followed it. Within a block only the last
    /// write of a location is what the block exits with, so construction reads
    /// the log in this order ([`BoundaryRecorder::write_at`]).
    pub seq: u64,
}

/// A read no earlier write in its block answered, standing in for the value
/// construction will name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedRead<K> {
    /// The block the read happened in.
    pub block_idx: usize,
    /// The location read.
    pub key: K,
    /// The placeholder the front end used for the value.
    pub var: SsaVarId,
    /// Whether an operation consumed the value, rather than only carrying it
    /// through a write of part of its location.
    ///
    /// A write of part of a location reads the location to keep the part it
    /// does not write, and that read is not the program using the value.
    pub consumed: bool,
}

/// The record a front end that lifts a block at a time keeps while it runs,
/// for [`SsaFunction::construct_at_boundaries`] to answer once it has.
///
/// Per block, what each location holds; and across the function, every write
/// and every read no earlier write in its block answered. Locations are keys
/// of the front end's choosing: registers, frame slots, stack positions.
#[derive(Debug, Clone)]
pub struct BoundaryRecorder<K> {
    /// What each location holds, per block, as the lift has it now.
    bindings: Vec<BTreeMap<K, SsaVarId>>,
    /// Every write, in the order recorded.
    writes: Vec<RecordedWrite<K>>,
    /// Every read no earlier write in its block answered, in the order
    /// recorded.
    reads: Vec<RecordedRead<K>>,
    /// Per block, the locations an operation destroyed without defining, with
    /// the site that destroyed each, until something binds them again.
    unbound: Vec<BTreeMap<K, usize>>,
    /// Every location any read asked for, in any block.
    read_keys: BTreeSet<K>,
    /// The next write's position in lift order.
    seq: u64,
}

impl<K> Default for BoundaryRecorder<K> {
    /// A record of no blocks.
    fn default() -> Self {
        Self {
            bindings: Vec::new(),
            writes: Vec::new(),
            reads: Vec::new(),
            unbound: Vec::new(),
            read_keys: BTreeSet::new(),
            seq: 0,
        }
    }
}

impl<K: Ord + Copy> BoundaryRecorder<K> {
    /// Starts an empty record for a function of `block_count` blocks.
    #[must_use]
    pub fn new(block_count: usize) -> Self {
        Self {
            bindings: vec![BTreeMap::new(); block_count],
            writes: Vec::new(),
            reads: Vec::new(),
            unbound: vec![BTreeMap::new(); block_count],
            read_keys: BTreeSet::new(),
            seq: 0,
        }
    }

    /// What `key` holds in `block_idx` as the lift has it now, or `None` when
    /// nothing in the block has bound it.
    #[must_use]
    pub fn bound(&self, block_idx: usize, key: K) -> Option<SsaVarId> {
        self.bindings
            .get(block_idx)
            .and_then(|bindings| bindings.get(&key).copied())
    }

    /// Records that `block_idx` wrote `var` to `key`, at the next position in
    /// lift order.
    pub fn write(&mut self, block_idx: usize, key: K, var: SsaVarId) {
        let seq = self.next_seq();
        self.write_at(block_idx, key, var, seq);
    }

    /// Records a write that belongs at `seq` in lift order rather than at the
    /// end of it: a definition the lift deferred, made where its cause was.
    pub fn write_at(&mut self, block_idx: usize, key: K, var: SsaVarId, seq: u64) {
        if let Some(bindings) = self.bindings.get_mut(block_idx) {
            bindings.insert(key, var);
        }
        if let Some(unbound) = self.unbound.get_mut(block_idx) {
            unbound.remove(&key);
        }
        self.writes.push(RecordedWrite {
            block_idx,
            key,
            var,
            seq,
        });
    }

    /// Takes the next position in lift order.
    pub fn next_seq(&mut self) -> u64 {
        let seq = self.seq;
        self.seq = self.seq.saturating_add(1);
        seq
    }

    /// Records that the operation at `site` destroyed `key` in `block_idx`
    /// without defining it, and returns what it held.
    ///
    /// The binding is removed rather than overwritten, so a later read in the
    /// block does not see the old value; [`Self::unbound_by`] tells such a
    /// read apart from one of a location the block never touched.
    pub fn unbind(&mut self, block_idx: usize, key: K, site: usize) -> Option<SsaVarId> {
        let prior = self
            .bindings
            .get_mut(block_idx)
            .and_then(|bindings| bindings.remove(&key));
        if let Some(unbound) = self.unbound.get_mut(block_idx) {
            unbound.insert(key, site);
        }
        prior
    }

    /// The site that destroyed `key` in `block_idx`, when one did and nothing
    /// has bound it since.
    #[must_use]
    pub fn unbound_by(&self, block_idx: usize, key: K) -> Option<usize> {
        self.unbound
            .get(block_idx)
            .and_then(|unbound| unbound.get(&key).copied())
    }

    /// Records that something asked for `key`'s value.
    pub fn note_read(&mut self, key: K) {
        self.read_keys.insert(key);
    }

    /// Every location any read asked for.
    #[must_use]
    pub fn read_keys(&self) -> &BTreeSet<K> {
        &self.read_keys
    }

    /// Records a read of `key` in `block_idx` that no earlier write in the
    /// block answered, standing in as `var`, and binds `var` there.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block reading.
    /// * `key` - The location read.
    /// * `var` - The placeholder the lift uses for the value.
    /// * `consumed` - Whether an operation uses the value
    ///   ([`RecordedRead::consumed`]).
    pub fn record_read(&mut self, block_idx: usize, key: K, var: SsaVarId, consumed: bool) {
        if let Some(bindings) = self.bindings.get_mut(block_idx) {
            bindings.insert(key, var);
        }
        if let Some(unbound) = self.unbound.get_mut(block_idx) {
            unbound.remove(&key);
        }
        self.read_keys.insert(key);
        self.reads.push(RecordedRead {
            block_idx,
            key,
            var,
            consumed,
        });
    }

    /// Marks the read `block_idx` recorded for `key` as `var` consumed: a use
    /// of a value an earlier read in the block only carried through.
    pub fn note_consumed(&mut self, block_idx: usize, key: K, var: SsaVarId) {
        if let Some(read) = self
            .reads
            .iter_mut()
            .rev()
            .find(|read| read.block_idx == block_idx && read.key == key && read.var == var)
        {
            read.consumed = true;
        }
    }

    /// Every write, in the order recorded until [`Self::sort_writes`], and in
    /// lift order after it.
    #[must_use]
    pub fn writes(&self) -> &[RecordedWrite<K>] {
        &self.writes
    }

    /// Every read no earlier write in its block answered, in the order
    /// recorded.
    #[must_use]
    pub fn reads(&self) -> &[RecordedRead<K>] {
        &self.reads
    }

    /// Puts the write log in lift order.
    pub fn sort_writes(&mut self) {
        self.writes.sort_by_key(|write| write.seq);
    }

    /// Moves every block-indexed record up by one block, after construction
    /// split a fresh entry block off ahead of the old one
    /// ([`BoundaryConstruction::entry_split`]). The new entry holds nothing.
    pub fn shift_for_entry_split(&mut self) {
        self.bindings.insert(0, BTreeMap::new());
        self.unbound.insert(0, BTreeMap::new());
        for write in &mut self.writes {
            write.block_idx = write.block_idx.saturating_add(1);
        }
        for read in &mut self.reads {
            read.block_idx = read.block_idx.saturating_add(1);
        }
    }

    /// Answers every recorded read and places the phis the joins need, through
    /// [`SsaFunction::construct_at_boundaries`].
    ///
    /// Each location is a rename group, numbered in the order the locations
    /// were first written and then first read; that is the order their phis
    /// are placed in. Writes are taken in lift order whether or not
    /// [`Self::sort_writes`] has run.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The lifted function, whose blocks the record indexes.
    /// * `entry_type` - The type of a minted entry value of a location.
    ///
    /// # Returns
    ///
    /// What construction did, with the location of each group.
    pub fn construct<T: Target>(
        &self,
        ssa: &mut SsaFunction<T>,
        entry_type: impl Fn(K) -> T::Type,
    ) -> RecordedConstruction<K> {
        let mut keys: Vec<K> = Vec::new();
        let mut groups: BTreeMap<K, u32> = BTreeMap::new();
        for key in self
            .writes
            .iter()
            .map(|write| write.key)
            .chain(self.reads.iter().map(|read| read.key))
        {
            if let Entry::Vacant(slot) = groups.entry(key) {
                slot.insert(u32::try_from(keys.len()).unwrap_or(u32::MAX));
                keys.push(key);
            }
        }
        let mut writes: Vec<&RecordedWrite<K>> = self.writes.iter().collect();
        writes.sort_by_key(|write| write.seq);
        let mut boundaries = vec![BlockBoundary::default(); ssa.block_count()];
        for read in &self.reads {
            if let (Some(boundary), Some(group)) =
                (boundaries.get_mut(read.block_idx), groups.get(&read.key))
            {
                boundary.reads.push((*group, read.var));
            }
        }
        for write in writes {
            if let (Some(boundary), Some(group)) =
                (boundaries.get_mut(write.block_idx), groups.get(&write.key))
            {
                boundary.writes.push((*group, write.var));
            }
        }
        let construction = ssa.construct_at_boundaries(&boundaries, |group| {
            keys.get(usize::try_from(group).unwrap_or(usize::MAX))
                .map_or_else(T::unknown_type, |key| entry_type(*key))
        });
        RecordedConstruction { construction, keys }
    }
}

/// What [`BoundaryRecorder::construct`] did, with the location each rename
/// group stands for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedConstruction<K> {
    /// What construction did, in rename groups.
    pub construction: BoundaryConstruction,
    /// The location of each group, indexed by group.
    keys: Vec<K>,
}

impl<K: Copy> RecordedConstruction<K> {
    /// The location `group` stands for.
    #[must_use]
    pub fn key(&self, group: u32) -> Option<K> {
        self.keys.get(usize::try_from(group).ok()?).copied()
    }
}

impl<T: Target> SsaFunction<T> {
    /// Places the phis, answers every read that crosses a block boundary, and
    /// gives each location one value on entry.
    ///
    /// The algorithm is Cytron's over the reported writes: phis at the iterated
    /// dominance frontier of each group's writing blocks, then one walk of the
    /// dominator tree carrying what each group holds, answering each block's
    /// outside reads from what reaches it and filling each phi from what its
    /// predecessors hold on exit.
    ///
    /// A location is entered with one value however many paths read it before
    /// writing it. The first read no definition reaches becomes that value, and
    /// every later one - and every phi operand along a path that never wrote the
    /// location - is answered by it; where no read names one, a variable is
    /// minted, typed by `entry_type`.
    ///
    /// Phi results are created untyped: their operands are not known when they
    /// are placed. [`Self::settle_phi_types`] types them once they are.
    ///
    /// An entry block a terminator names - a function whose first instruction
    /// heads a loop - is split off behind a fresh block first. Placement needs
    /// an edge the function is entered by: without one, a location the loop
    /// redefines has no way to carry the value it was entered with into the
    /// entry, and a read of it there was answered by the definition below it.
    /// The report says when this happened ([`BoundaryConstruction::entry_split`]).
    ///
    /// # Arguments
    ///
    /// * `blocks` - Each block's boundary, indexed by block.
    /// * `entry_type` - The type of a minted entry value of a group.
    ///
    /// # Returns
    ///
    /// The answers given, the phis placed, and the entry values minted.
    pub fn construct_at_boundaries(
        &mut self,
        blocks: &[BlockBoundary],
        entry_type: impl Fn(u32) -> T::Type,
    ) -> BoundaryConstruction {
        if self.block_count() == 0 {
            return BoundaryConstruction::default();
        }
        // Only an entry without phis: a front end that placed its own has
        // nothing for this construction to answer at the entry, and splitting
        // it would rebuild what it placed.
        let entry_split = self
            .block(0)
            .is_some_and(|entry| entry.phi_nodes().is_empty())
            && self.ensure_entry_has_no_predecessors();
        let shifted: Vec<BlockBoundary>;
        let blocks = if entry_split {
            shifted = iter::once(BlockBoundary::default())
                .chain(blocks.iter().cloned())
                .collect();
            shifted.as_slice()
        } else {
            blocks
        };
        let block_count = self.block_count();
        let mut def_blocks: BTreeMap<u32, BitSet> = BTreeMap::new();
        for (block_idx, boundary) in blocks.iter().enumerate() {
            for (group, _) in &boundary.writes {
                def_blocks
                    .entry(*group)
                    .or_insert_with(|| BitSet::new(block_count))
                    .insert(block_idx);
            }
        }
        if def_blocks.is_empty() && blocks.iter().all(|boundary| boundary.reads.is_empty()) {
            return BoundaryConstruction {
                entry_split,
                ..BoundaryConstruction::default()
            };
        }

        let BoundaryCfg {
            dom_tree,
            frontiers,
            predecessors,
            throwing,
        } = self.boundary_cfg();
        let live_in: BTreeMap<u32, BitSet> = BTreeMap::new();
        let group_filter = |_| true;
        let group_to_origin = |_| VariableOrigin::Phi;
        let placements = place_pruned_phis(
            self.blocks_mut(),
            PhiPlacementConfig {
                defs: &def_blocks,
                live_in: &live_in,
                dominance_frontiers: &frontiers,
                reachable: None,
                group_filter: &group_filter,
                group_to_origin: &group_to_origin,
                leave_target_fn: None,
            },
        );
        let phi_results = self.allocate_boundary_phis(&placements);

        let mut entry = EntryValues::default();
        let walk = boundary_walk(blocks, &dom_tree, block_count, &phi_results, &mut entry);
        self.fill_boundary_phis(
            &phi_results,
            &predecessors,
            &walk.exit_values,
            &mut entry,
            &entry_type,
        );
        self.fill_exceptional_boundary_operands(
            &phi_results,
            &throwing,
            blocks,
            &walk,
            &mut entry,
            &entry_type,
        );
        let replacements = walk.replacements;
        let replacements = resolve_chains(&replacements);
        self.apply_boundary_replacements(&replacements);

        BoundaryConstruction {
            replacements: replacements.into_iter().collect(),
            phis: phi_results
                .into_iter()
                .map(|((block, group), result)| (block, group, result))
                .collect(),
            minted_entry_values: entry.minted.into_iter().collect(),
            entry_split,
        }
    }

    /// The dominator tree, dominance frontiers, per-block predecessors and
    /// throwing blocks construction reads.
    ///
    /// Dominance runs over every edge, a throw's to its handler included: a
    /// handler is reached from every block that may throw to it, and its phis
    /// are placed for what those blocks define. A phi's operands are another
    /// matter. A predecessor whose terminator transfers here supplies what it
    /// holds on exit; a block that throws here supplies what it may hold at
    /// any point it runs ([`Self::fill_exceptional_boundary_operands`]). One
    /// block can be both - a finally a runtime enters by a jump as well as by
    /// a throw - and supplies both.
    fn boundary_cfg(&self) -> BoundaryCfg {
        let cfg = SsaCfg::from_ssa(self);
        let dom_tree = compute_dominators(&cfg, NodeId::new(0));
        let frontiers = compute_dominance_frontiers(&cfg, &dom_tree);
        let predecessors = (0..self.block_count())
            .map(|block_idx| {
                let mut preds = cfg.terminator_predecessors(block_idx).to_vec();
                preds.dedup();
                preds
            })
            .collect();
        let throwing = (0..self.block_count())
            .map(|block_idx| cfg.exceptional_predecessors(block_idx).to_vec())
            .collect();
        BoundaryCfg {
            dom_tree,
            frontiers,
            predecessors,
            throwing,
        }
    }

    /// Gives each handler phi an exceptional operand for every value its
    /// group may hold in each block that throws to the handler: what the
    /// block was entered with, and each value it writes. A throw can happen
    /// anywhere in the block, so the handler merges over its program points
    /// rather than over an edge's exit.
    fn fill_exceptional_boundary_operands(
        &mut self,
        phi_results: &BTreeMap<(usize, u32), SsaVarId>,
        throwing: &[Vec<usize>],
        blocks: &[BlockBoundary],
        walk: &BoundaryWalk,
        entry: &mut EntryValues,
        entry_type: &impl Fn(u32) -> T::Type,
    ) {
        for ((block_idx, group), result) in phi_results {
            let Some(sources) = throwing
                .get(*block_idx)
                .filter(|sources| !sources.is_empty())
            else {
                continue;
            };
            let mut operands: Vec<(usize, SsaVarId)> = Vec::new();
            for source in sources {
                let on_entry = match walk
                    .entry_values
                    .get(*source)
                    .and_then(|state| state.get(group))
                {
                    Some(value) => *value,
                    None => entry.arriving(self, *group, entry_type),
                };
                operands.push((*source, on_entry));
                let written = blocks
                    .get(*source)
                    .into_iter()
                    .flat_map(|boundary| boundary.writes.iter())
                    .filter(|(written, _)| written == group)
                    .map(|(_, value)| (*source, *value));
                operands.extend(written);
            }
            if let Some(block) = self.block_mut(*block_idx)
                && let Some(phi) = block
                    .phi_nodes_mut()
                    .iter_mut()
                    .find(|phi| phi.result() == *result)
            {
                // Every throwing block owes one, the phi's own value
                // included: a block a resumption entry dominates throws back
                // to it holding what the entry merged.
                for (source, value) in operands {
                    phi.add_exceptional_operand(source, value);
                }
            }
        }
    }

    /// Gives every placed phi a fresh result, returning them by
    /// `(block, group)`.
    fn allocate_boundary_phis(
        &mut self,
        placements: &[(usize, u32)],
    ) -> BTreeMap<(usize, u32), SsaVarId> {
        let mut results = BTreeMap::new();
        for (block_idx, group) in placements {
            let result = self.create_variable(
                VariableOrigin::Phi,
                0,
                DefSite::phi(*block_idx),
                T::unknown_type(),
            );
            if let Some(block) = self.block_mut(*block_idx)
                && let Some(phi) = block
                    .phi_nodes_mut()
                    .iter_mut()
                    .find(|phi| phi.result() == SsaVarId::PLACEHOLDER)
            {
                phi.set_result(result);
            }
            results.insert((*block_idx, *group), result);
        }
        results
    }

    /// Fills each phi from what its predecessors hold on exit, and from the
    /// group's entry value along a path that never wrote it.
    fn fill_boundary_phis(
        &mut self,
        phi_results: &BTreeMap<(usize, u32), SsaVarId>,
        predecessors: &[Vec<usize>],
        exit_values: &[BTreeMap<u32, SsaVarId>],
        entry: &mut EntryValues,
        entry_type: &impl Fn(u32) -> T::Type,
    ) {
        for ((block_idx, group), result) in phi_results {
            let Some(preds) = predecessors.get(*block_idx) else {
                continue;
            };
            let mut operands: Vec<(usize, SsaVarId)> = Vec::with_capacity(preds.len());
            let mut unreached: Vec<usize> = Vec::new();
            for pred in preds {
                match exit_values.get(*pred).and_then(|state| state.get(group)) {
                    Some(value) => operands.push((*pred, *value)),
                    None => unreached.push(*pred),
                }
            }
            // A path that never wrote the group carries the value the function
            // was entered with, which is defined at the entry and so reaches
            // every predecessor. The phi's own result would be a
            // self-reference no predecessor is dominated by.
            if !unreached.is_empty() {
                let arriving = entry.arriving(self, *group, entry_type);
                operands.extend(unreached.into_iter().map(|pred| (pred, arriving)));
            }
            if let Some(block) = self.block_mut(*block_idx)
                && let Some(phi) = block
                    .phi_nodes_mut()
                    .iter_mut()
                    .find(|phi| phi.result() == *result)
            {
                for (pred, value) in operands {
                    phi.set_operand(pred, value);
                }
            }
        }
    }

    /// Rewrites every use of a placeholder to its answer, in instructions and
    /// phi operands alike.
    fn apply_boundary_replacements(&mut self, replacements: &BTreeMap<SsaVarId, SsaVarId>) {
        if replacements.is_empty() {
            return;
        }
        for block in self.blocks_mut() {
            for phi in block.phi_nodes_mut() {
                let updates: Vec<(usize, SsaVarId)> = phi
                    .operands()
                    .iter()
                    .filter(|operand| !operand.is_exceptional())
                    .filter_map(|operand| {
                        replacements
                            .get(&operand.value())
                            .map(|value| (operand.predecessor(), *value))
                    })
                    .collect();
                for (pred, value) in updates {
                    phi.set_operand(pred, value);
                }
                // An exceptional operand is keyed by its value as well as its
                // block, so it is re-added under its answer rather than set.
                let exceptional: Vec<(usize, SsaVarId)> = phi
                    .operands()
                    .iter()
                    .filter(|operand| operand.is_exceptional())
                    .filter_map(|operand| {
                        replacements
                            .get(&operand.value())
                            .map(|value| (operand.predecessor(), *value))
                    })
                    .collect();
                if !exceptional.is_empty() {
                    phi.retain_operands(|operand| {
                        !operand.is_exceptional() || !replacements.contains_key(&operand.value())
                    });
                    for (source, value) in exceptional {
                        phi.add_exceptional_operand(source, value);
                    }
                }
            }
            for instruction in block.instructions_mut() {
                instruction
                    .op_mut()
                    .replace_uses_with(|var| replacements.get(&var).copied());
            }
        }
    }
}

/// What each group holds when each block exits, by block.
type ExitValues = Vec<BTreeMap<u32, SsaVarId>>;

/// The graph facts construction reads ([`SsaFunction::boundary_cfg`]).
struct BoundaryCfg {
    /// Dominance over every edge, a throw's included.
    dom_tree: DominatorTree,
    /// Each block's dominance frontier.
    frontiers: Vec<BitSet>,
    /// Each block's predecessors whose terminator transfers to it, once each.
    predecessors: Vec<Vec<usize>>,
    /// Each handler's blocks that may throw to it.
    throwing: Vec<Vec<usize>>,
}

/// What the dominator-tree walk established ([`boundary_walk`]).
struct BoundaryWalk {
    /// What each group holds when each block is entered: its phis, and the
    /// values that reach it from its dominator or from the function's entry.
    entry_values: ExitValues,
    /// What each group holds when each block exits.
    exit_values: ExitValues,
    /// The answer each outside read was given.
    replacements: Vec<(SsaVarId, SsaVarId)>,
}

/// The one value each group holds on entry to the function.
///
/// The first read no definition reaches becomes it, and answers every later
/// such read. Sound because each candidate is defined at [`DefSite::entry`],
/// which dominates every block, so any one of them can stand wherever another
/// is read.
#[derive(Debug, Default)]
struct EntryValues {
    /// The value each group arrives as.
    values: BTreeMap<u32, SsaVarId>,
    /// The values minted for groups no unreached read named.
    minted: BTreeMap<u32, SsaVarId>,
}

impl EntryValues {
    /// Answers a read of `group` no definition reaches: an earlier such read's
    /// value, or `placeholder`, which becomes it.
    fn answer(&mut self, group: u32, placeholder: SsaVarId) -> SsaVarId {
        *self.values.entry(group).or_insert(placeholder)
    }

    /// The value `group` arrives as, minting one when no unreached read named
    /// one.
    fn arriving<T: Target>(
        &mut self,
        ssa: &mut SsaFunction<T>,
        group: u32,
        entry_type: &impl Fn(u32) -> T::Type,
    ) -> SsaVarId {
        if let Some(value) = self.values.get(&group) {
            return *value;
        }
        let value = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            entry_type(group),
        );
        self.values.insert(group, value);
        self.minted.insert(group, value);
        value
    }
}

/// Walks the dominator tree carrying what each group holds, returning each
/// block's exit values and the answer each outside read was given.
///
/// Per block: what its immediate dominator held on exit, then its own phis,
/// then its outside reads answered from that - or from the group's entry value
/// where nothing reaches - then its writes in order.
fn boundary_walk(
    blocks: &[BlockBoundary],
    dom_tree: &DominatorTree,
    block_count: usize,
    phi_results: &BTreeMap<(usize, u32), SsaVarId>,
    entry: &mut EntryValues,
) -> BoundaryWalk {
    let mut phis_by_block: Vec<Vec<(u32, SsaVarId)>> = vec![Vec::new(); block_count];
    for ((block_idx, group), result) in phi_results {
        if let Some(bucket) = phis_by_block.get_mut(*block_idx) {
            bucket.push((*group, *result));
        }
    }
    let mut entry_values: ExitValues = vec![BTreeMap::new(); block_count];
    let mut exit_values: ExitValues = vec![BTreeMap::new(); block_count];
    let mut replacements: Vec<(SsaVarId, SsaVarId)> = Vec::new();
    let mut stack: Vec<NodeId> = vec![NodeId::new(0)];
    while let Some(node) = stack.pop() {
        let block_idx = node.index();
        let mut current = if block_idx == 0 {
            BTreeMap::new()
        } else {
            dom_tree
                .immediate_dominator(node)
                .and_then(|idom| exit_values.get(idom.index()).cloned())
                .unwrap_or_default()
        };
        if let Some(bucket) = phis_by_block.get(block_idx) {
            for (group, result) in bucket {
                current.insert(*group, *result);
            }
        }
        if let Some(boundary) = blocks.get(block_idx) {
            for (group, placeholder) in &boundary.reads {
                match current.get(group).copied() {
                    Some(reaching) if reaching != *placeholder => {
                        replacements.push((*placeholder, reaching));
                    }
                    Some(_) => {}
                    // Nothing reaches the read: it sees the value the function
                    // was entered with, which is also what this block holds
                    // for the group from here, or a successor's phi finds no
                    // operand for the edge.
                    None => {
                        let arriving = entry.answer(*group, *placeholder);
                        if arriving != *placeholder {
                            replacements.push((*placeholder, arriving));
                        }
                        current.insert(*group, arriving);
                    }
                }
            }
            // Every outside read was answered by what the block was entered
            // with, so this is the block's state before its first write.
            if let Some(slot) = entry_values.get_mut(block_idx) {
                *slot = current.clone();
            }
            for (group, value) in &boundary.writes {
                current.insert(*group, *value);
            }
        } else if let Some(slot) = entry_values.get_mut(block_idx) {
            *slot = current.clone();
        }
        if let Some(slot) = exit_values.get_mut(block_idx) {
            *slot = current;
        }
        for child in dom_tree.children(node) {
            stack.push(*child);
        }
    }
    BoundaryWalk {
        entry_values,
        exit_values,
        replacements,
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockBoundary, BoundaryRecorder};
    use crate::{
        ir::{
            exception::{BlockRange, SsaExceptionHandler},
            function::{SsaDefSpec, SsaFunction, SsaFunctionBuilder},
            instruction::SsaInstruction,
            ops::SsaOp,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// A diamond: B0 branches to B1 and B2, which join at B3.
    fn diamond() -> SsaFunction<MockTarget> {
        let spec = || SsaDefSpec::<MockTarget>::tmp(MockType::I32);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.ensure_block(3);
        builder
            .in_block(0, |block| {
                let condition = block.const_i32(spec(), 1)?;
                block.branch(condition, 1, 2)
            })
            .expect("B0 builds");
        for arm in [1, 2] {
            builder
                .in_block(arm, |block| block.jump(3))
                .expect("an arm builds");
        }
        builder
            .in_block(3, |block| block.ret(None))
            .expect("B3 builds");
        builder.finish().expect("the diamond is well-formed")
    }

    /// A value the front end used for a read it could not answer.
    fn read(ssa: &mut SsaFunction<MockTarget>) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::I32,
        )
    }

    /// Two reads of one group on sibling paths, neither reached by a write, see
    /// the one value the group is entered with.
    #[test]
    fn unreached_reads_of_one_group_share_its_entry_value() {
        let mut ssa = diamond();
        let left = read(&mut ssa);
        let right = read(&mut ssa);
        let mut blocks = vec![BlockBoundary::default(); 4];
        blocks[1].reads.push((0, left));
        blocks[2].reads.push((0, right));

        let constructed = ssa.construct_at_boundaries(&blocks, |_| MockType::I32);

        assert_eq!(
            constructed.replacements.len(),
            1,
            "one read answers the other"
        );
        let answer = constructed.replacements.values().next().copied();
        assert!(answer == Some(left) || answer == Some(right));
        assert!(constructed.minted_entry_values.is_empty());
        assert!(constructed.phis.is_empty(), "nothing writes the group");
    }

    /// A join whose operand on one path no write reaches takes the entry value a
    /// read on another path named, and nothing is minted.
    #[test]
    fn a_phi_operand_no_write_reaches_takes_the_named_entry_value() {
        let mut ssa = diamond();
        let entered = read(&mut ssa);
        let written = read(&mut ssa);
        let mut blocks = vec![BlockBoundary::default(); 4];
        blocks[1].reads.push((0, entered));
        blocks[1].writes.push((0, written));

        let constructed = ssa.construct_at_boundaries(&blocks, |_| MockType::I32);

        let phi = ssa
            .block(3)
            .and_then(|block| block.phi_nodes().first())
            .expect("the join merges the group");
        let operand = |pred: usize| {
            phi.operands()
                .iter()
                .find(|operand| operand.predecessor() == pred)
                .map(|operand| operand.value())
        };
        assert_eq!(operand(1), Some(written));
        assert_eq!(operand(2), Some(entered), "the path that never wrote it");
        assert_eq!(constructed.phis.len(), 1);
        assert!(constructed.minted_entry_values.is_empty());
    }

    /// **A resumption entry merges its own value from the blocks it
    /// dominates.** `B0 -> B1 -> B2 -> B3`, with `B0..B3` protected and `B1`
    /// the clause's continuation, entered once a handler elsewhere has
    /// caught: `B1` reads the group and `B2` writes it, so `B2`'s throw brings
    /// the written value back to `B1`, which merges it. `B1` throws holding
    /// that merge, and owes the merge an operand of it: the phi's own value.
    /// Without it the verifier found the throwing block contributing nothing.
    #[test]
    fn a_resumption_entry_merges_its_own_value() {
        let spec = || SsaDefSpec::<MockTarget>::tmp(MockType::I32);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.ensure_block(3);
        for (block_idx, next) in [(0, 1), (1, 2), (2, 3)] {
            builder
                .in_block(block_idx, |block| {
                    let _ = block.const_i32(spec(), 1)?;
                    block.jump(next)
                })
                .expect("the block builds");
        }
        builder
            .in_block(3, |block| block.ret(None))
            .expect("B3 builds");
        let mut ssa = builder.finish().expect("the chain is well-formed");
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 4,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(0, 3),
            handler_range: BlockRange::new(1, 2),
            filter_range: None,
            selector: None,
        }]);
        let resumed = read(&mut ssa);
        let written = read(&mut ssa);
        let mut blocks = vec![BlockBoundary::default(); 4];
        blocks[1].reads.push((0, resumed));
        blocks[2].writes.push((0, written));

        ssa.construct_at_boundaries(&blocks, |_| MockType::I32);

        let phi = ssa
            .block(1)
            .and_then(|block| block.phi_nodes().first())
            .expect("the continuation merges the group");
        let from = |source: usize| {
            phi.exceptional_operands()
                .filter(|operand| operand.predecessor() == source)
                .map(|operand| operand.value())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            from(2),
            vec![phi.result(), written],
            "the later block before its write and after"
        );
        assert_eq!(from(1), vec![phi.result()], "the block's own merge");
    }

    /// `B0: v = read(g); v' = f(v); branch B0 | B1`, `B1: return`: the entry
    /// heads a loop that redefines the group it reads.
    fn entry_loop() -> SsaFunction<MockTarget> {
        let spec = || SsaDefSpec::<MockTarget>::tmp(MockType::I32);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder.ensure_block(1);
        builder
            .in_block(0, |block| {
                let condition = block.const_i32(spec(), 1)?;
                block.branch(condition, 0, 1)
            })
            .expect("B0 builds");
        builder
            .in_block(1, |block| block.ret(None))
            .expect("B1 builds");
        builder.finish().expect("the loop is well-formed")
    }

    /// **An entry that heads a loop is entered by an edge of its own.** The
    /// entry reads a group and then writes it, and branches back to itself.
    /// Placed with the entry as is, the loop's phi had no edge to carry the
    /// entered value in on, and the read was answered by the write below it:
    /// a use above its definition, which the later entry split could not
    /// repair, so the function was refused.
    #[test]
    fn an_entry_heading_a_loop_is_split_before_placement() {
        let mut ssa = entry_loop();
        let entered = read(&mut ssa);
        let redefined = read(&mut ssa);
        let mut blocks = vec![BlockBoundary::default(); 2];
        blocks[0].reads.push((0, entered));
        blocks[0].writes.push((0, redefined));

        let constructed = ssa.construct_at_boundaries(&blocks, |_| MockType::I32);

        assert!(constructed.entry_split);
        assert_eq!(ssa.block_count(), 3);
        let phi = ssa
            .block(1)
            .and_then(|block| block.phi_nodes().first())
            .expect("the old entry merges the group");
        let operand = |pred: usize| {
            phi.operands()
                .iter()
                .find(|operand| operand.predecessor() == pred)
                .map(|operand| operand.value())
        };
        let minted = constructed
            .minted_entry_values
            .first()
            .map(|(_, value)| *value);
        assert!(
            minted.is_some(),
            "the group is entered with a value of its own"
        );
        assert_eq!(operand(0), minted, "carried in on the new entry edge");
        assert_eq!(operand(1), Some(redefined), "and the loop's around");
        assert_eq!(
            constructed.replacements.get(&entered),
            Some(&phi.result()),
            "the read is the merge, not the write below it"
        );
    }

    /// Control for [`an_entry_heading_a_loop_is_split_before_placement`]: an
    /// entry nothing branches to is left where it is.
    #[test]
    fn an_entry_nothing_branches_to_is_not_split() {
        let mut ssa = diamond();
        let entered = read(&mut ssa);
        let mut blocks = vec![BlockBoundary::default(); 4];
        blocks[1].reads.push((0, entered));
        let constructed = ssa.construct_at_boundaries(&blocks, |_| MockType::I32);
        assert!(!constructed.entry_split);
        assert_eq!(ssa.block_count(), 4);
    }

    /// A read answered by another read, itself answered by a definition, is
    /// answered by the definition, and every use of it says so.
    #[test]
    fn a_read_of_a_read_is_answered_by_the_definition() {
        let mut ssa = diamond();
        let defined = read(&mut ssa);
        let first = read(&mut ssa);
        let second = read(&mut ssa);
        // B0 writes group 0; B1 reads it and moves it into group 1; B3 reads
        // group 1, which only B1 writes - so a phi merges it with the entry
        // value, and B1's move carries B1's read, which B0's write answers.
        let mut blocks = vec![BlockBoundary::default(); 4];
        blocks[0].writes.push((0, defined));
        blocks[1].reads.push((0, first));
        blocks[1].writes.push((1, first));
        blocks[3].reads.push((1, second));
        if let Some(block) = ssa.block_mut(3) {
            block
                .instructions_mut()
                .insert(0, SsaInstruction::synthetic(SsaOp::returns(Some(first))));
        }

        let constructed = ssa.construct_at_boundaries(&blocks, |_| MockType::I32);

        assert_eq!(constructed.replacements.get(&first), Some(&defined));
        let returned = ssa
            .block(3)
            .and_then(|block| block.instructions().first())
            .map(|i| i.op().clone());
        assert_eq!(returned, Some(SsaOp::returns(Some(defined))));
    }

    /// A location written before a diamond and read after it, by neither arm
    /// written, is answered by the write before it, with no phi for it; a
    /// location both arms write gets one.
    #[test]
    fn a_recorded_read_after_a_diamond_takes_the_write_before_it() {
        let mut ssa = diamond();
        let (before, left, right) = (read(&mut ssa), read(&mut ssa), read(&mut ssa));
        let (kept, merged) = (read(&mut ssa), read(&mut ssa));
        let mut record = BoundaryRecorder::<char>::new(4);
        record.write(0, 'c', before);
        record.write(1, 'a', left);
        record.write(2, 'a', right);
        record.record_read(3, 'c', kept, true);
        record.record_read(3, 'a', merged, true);

        let constructed = record.construct(&mut ssa, |_| MockType::I32);

        let replacements = &constructed.construction.replacements;
        assert_eq!(replacements.get(&kept), Some(&before));
        let phis = &constructed.construction.phis;
        assert_eq!(phis.len(), 1);
        let (block, group, result) = phis[0];
        assert_eq!((block, constructed.key(group)), (3, Some('a')));
        assert_eq!(replacements.get(&merged), Some(&result));
    }

    /// A deferred write recorded at an earlier position is not what its block
    /// exits with when a later write of the same location followed it.
    #[test]
    fn a_deferred_write_takes_its_place_in_lift_order() {
        let mut ssa = diamond();
        let (early, late, after) = (read(&mut ssa), read(&mut ssa), read(&mut ssa));
        let mut record = BoundaryRecorder::<char>::new(4);
        let deferred = record.next_seq();
        record.write(0, 'r', late);
        record.write_at(0, 'r', early, deferred);
        record.record_read(1, 'r', after, true);

        let constructed = record.construct(&mut ssa, |_| MockType::I32);
        assert_eq!(
            constructed.construction.replacements.get(&after),
            Some(&late)
        );
    }

    /// Unbinding forgets the value in the block and remembers why; a write
    /// binds it again.
    #[test]
    fn an_unbound_location_reads_as_destroyed_until_written() {
        let mut ssa = diamond();
        let (value, again) = (read(&mut ssa), read(&mut ssa));
        let mut record = BoundaryRecorder::<char>::new(4);
        record.write(0, 'r', value);
        assert_eq!(record.unbind(0, 'r', 7), Some(value));
        assert_eq!(
            (record.bound(0, 'r'), record.unbound_by(0, 'r')),
            (None, Some(7))
        );
        record.write(0, 'r', again);
        assert_eq!(
            (record.bound(0, 'r'), record.unbound_by(0, 'r')),
            (Some(again), None)
        );
    }

    /// Locations are numbered in the order they were first written, then
    /// first read, each once.
    #[test]
    fn locations_are_numbered_by_first_write_then_first_read() {
        let mut ssa = diamond();
        let values: Vec<SsaVarId> = (0..4).map(|_| read(&mut ssa)).collect();
        let mut record = BoundaryRecorder::<char>::new(4);
        record.record_read(2, 'r', values[3], true);
        record.write(0, 's', values[0]);
        record.write(1, 'f', values[1]);
        record.write(0, 's', values[2]);

        let constructed = record.construct(&mut ssa, |_| MockType::I32);
        let keys: Vec<Option<char>> = (0..4).map(|group| constructed.key(group)).collect();
        assert_eq!(keys, [Some('s'), Some('f'), Some('r'), None]);
    }
}
