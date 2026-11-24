//! Webs: the SSA values that can be one variable, and the interference test
//! that decides them.
//!
//! A host that turns SSA back into variables - a lowering that coalesces
//! values into locals, an identity that names a register's successive values
//! as one - asks one question of every pair it would merge: may these two be
//! one variable? [`Gathering`] answers it for a whole function at once: it
//! joins each phi operand the caller accepts into the phi result's web, then
//! offers the values of each storage [`Group`] to each other first-fit, under
//! one interference test. The caller only says what to join, what it folds
//! into its uses, and which values it knows cannot be one variable whatever
//! their lives (two whose types admit no one type, say).
//!
//! # Why the obvious shortcut is wrong
//!
//! "Coalesce when the operand's only use is the phi" is unsound, and it is
//! unsound in exactly the case that matters - a loop:
//!
//! ```text
//! P:  v = 1
//!     ... use r ...      <- r still live after v's definition
//!     goto J
//! J:  r = phi(..., v@P)
//! ```
//!
//! `v` is single-use, but `r` is live past `v`'s definition inside `P`, so the
//! two overlap. Merging them puts two simultaneously-live values in one local:
//! wrong code, silently, and the SSA stays structurally valid so the verifier
//! sees nothing. The interference test is not optional.
//!
//! # The test
//!
//! Chaitin's condition, which is exact in SSA because each value has exactly
//! one definition: **two values interfere when one is live at the other's
//! definition point.** Liveness is [`LiveVariables`]', which places a phi
//! operand's use at the *predecessor's outgoing edge* rather than at the join -
//! the convention this depends on, and the one a def-use index does not use.
//!
//! A group kept in memory ([`Group::in_memory`]) adds one more definition: an
//! instruction that may write memory may write it, so a value still read after
//! one is not the group's there ([`Gathered::detached`]). An abstract local
//! shared with values outside the physical group cannot use its storage name:
//! those other definitions would introduce writes to the slot.

use std::{iter::once, mem};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::{
    analysis::{
        DefUseIndex, SsaCfg,
        dataflow::{AnalysisResults, DataFlowSolver, LiveVariables, LivenessResult},
    },
    bitset::BitSet,
    classes::Classes,
    ir::{
        SsaFunction,
        phi::PhiOperand,
        variable::{SsaVarId, UseSite},
    },
    target::Target,
};

/// Liveness, solved once per function.
type Liveness = AnalysisResults<LivenessResult>;

/// The webs under construction: the classes, and the members of each.
struct Webs {
    /// Which class each value is in.
    classes: Classes,
    /// Each class's members, stored at its root.
    ///
    /// Inline capacity one: a class is a singleton until it merges, and it is
    /// still a singleton for most values when the pass finishes - one heap
    /// allocation per SSA value, in every function, bought nothing.
    members: Vec<SmallVec<[SsaVarId; 1]>>,
    /// Each class's kinds ([`Conflicts::kinds`]), stored at its root: the web
    /// conflicts with another wherever two of their kinds do.
    kinds: Vec<SmallVec<[u32; 1]>>,
    /// Whether a class holds a value that interferes with every other
    /// ([`Interference::universal`]), stored at its root.
    universal: Vec<bool>,
    /// Whether a class holds a value that is a variable of its own
    /// ([`Gathering::anchored`]), stored at its root.
    anchored: Vec<bool>,
}

impl Webs {
    /// Creates `count` singleton webs, each of its value's kind and
    /// universality as `interference` decides them, and anchored where
    /// `anchored` names it.
    fn new<T: Target>(
        count: usize,
        interference: &Interference<'_, T>,
        anchored: &[usize],
    ) -> Self {
        let mut held = vec![false; count];
        for &value in anchored {
            if let Some(flag) = held.get_mut(value) {
                *flag = true;
            }
        }
        Self {
            classes: Classes::new(count),
            members: (0..count)
                .map(|index| SmallVec::from_buf([SsaVarId::from_index(index)]))
                .collect(),
            kinds: (0..count)
                .map(|index| interference.kind(index).into_iter().collect())
                .collect(),
            universal: (0..count)
                .map(|index| interference.universal.get(index).copied().unwrap_or(false))
                .collect(),
            anchored: held,
        }
    }

    /// Whether a storage group may offer `b`'s web to `a`'s: always, unless
    /// either holds a variable of its own ([`Gathering::anchored`]), which
    /// takes in only a web one of whose members is computed from one of its
    /// own, or one of whose own is computed from a member of the other.
    fn derives<T: Target>(&mut self, a: usize, b: usize, defuse: &DefUseIndex<T>) -> bool {
        let (left, right) = (self.classes.find(a), self.classes.find(b));
        if left == right {
            return true;
        }
        let anchored = |root: usize| self.anchored.get(root).copied().unwrap_or(false);
        if !anchored(left) && !anchored(right) {
            return true;
        }
        // A value is in the other web where its class's root is that web's:
        // the members are kept at the root, and nothing else is.
        let Self {
            classes, members, ..
        } = self;
        let mut reads = |readers: usize, read: usize| {
            members.get(readers).into_iter().flatten().any(|member| {
                defuse.def_op(*member).is_some_and(|op| {
                    let mut found = false;
                    op.for_each_use(|used| found |= classes.find(used.index()) == read);
                    found
                })
            })
        };
        reads(left, right) || reads(right, left)
    }

    /// Puts `a` and `b` in one web unless the two webs interfere.
    ///
    /// Webs are compared whole: a value that would come to share a local with
    /// three others is checked against all three, not only the one it was
    /// offered. Two webs interfere where a kind of one conflicts with a kind
    /// of the other, where either holds a value that interferes with every
    /// other, or where a member of the smaller has a neighbour in the larger
    /// ([`Interference::neighbours`]): each value is on the smaller side of a
    /// merge at most logarithmically often, so the merges cost the
    /// interference's size, not the square of the webs'.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of a value in the first web.
    /// * `b` - Index of a value in the second web.
    /// * `interference` - The function's interference probe.
    ///
    /// # Returns
    ///
    /// Whether the two values now share a web, including when they already did.
    fn join<T: Target>(
        &mut self,
        a: usize,
        b: usize,
        interference: &mut Interference<'_, T>,
    ) -> bool {
        let (left, right) = (self.classes.find(a), self.classes.find(b));
        if left == right {
            return true;
        }
        if self.universal.get(left).copied().unwrap_or(false)
            || self.universal.get(right).copied().unwrap_or(false)
        {
            return false;
        }
        let kinds = |root: usize| self.kinds.get(root).cloned().unwrap_or_default();
        let (left_kinds, right_kinds) = (kinds(left), kinds(right));
        if left_kinds.iter().any(|a| {
            right_kinds
                .iter()
                .any(|b| interference.kinds_conflict(Some(*a), Some(*b)))
        }) {
            return false;
        }
        let size = |root: usize| self.members.get(root).map_or(0, SmallVec::len);
        let (smaller, larger) = if size(left) <= size(right) {
            (left, right)
        } else {
            (right, left)
        };
        let probed = self.members.get(smaller).cloned().unwrap_or_default();
        for member in probed {
            let neighbours = interference
                .neighbours
                .get(member.index())
                .map_or(&[][..], Vec::as_slice);
            for &neighbour in neighbours {
                if self.classes.find(neighbour as usize) == larger {
                    return false;
                }
            }
        }
        self.force_join(a, b);
        true
    }

    /// Returns the representative of `value`'s web.
    fn find(&mut self, value: usize) -> usize {
        self.classes.find(value)
    }

    /// Puts `a` and `b` in one web without asking whether they interfere.
    ///
    /// `a`'s root stays the web's representative; the members, the kinds and
    /// the universality of the other move to it, the smaller list into the
    /// larger.
    fn force_join(&mut self, a: usize, b: usize) {
        let (left, right) = (self.classes.find(a), self.classes.find(b));
        if left == right {
            return;
        }
        self.classes.union(left, right);
        let mut moved = self
            .members
            .get_mut(right)
            .map(mem::take)
            .unwrap_or_default();
        if let Some(into) = self.members.get_mut(left) {
            if into.len() < moved.len() {
                mem::swap(into, &mut moved);
            }
            into.extend(moved);
        }
        let moved = self.kinds.get_mut(right).map(mem::take).unwrap_or_default();
        if let Some(into) = self.kinds.get_mut(left) {
            for kind in moved {
                if !into.contains(&kind) {
                    into.push(kind);
                }
            }
        }
        if self.universal.get(right).copied().unwrap_or(false)
            && let Some(flag) = self.universal.get_mut(left)
        {
            *flag = true;
        }
        if self.anchored.get(right).copied().unwrap_or(false)
            && let Some(flag) = self.anchored.get_mut(left)
        {
            *flag = true;
        }
    }
}

/// What the lowering makes of the SSA, as far as sharing a local depends on it.
///
/// Two values may share a local only if neither is still to be read where the
/// other's local is written - a question about the listing, which the SSA
/// answers only for a value that becomes a row and is read where it is used.
/// Two decisions of the lowering move the answer, and both are recorded here.
/// A folded computation is rendered where its result is used, so its operands
/// are read there - possibly blocks later than the SSA reads them. And a phi
/// result is written again by the copy placed before each predecessor's
/// terminator, a copy that runs whichever successor is then taken.
struct Listing {
    /// For each value, the folded computations that read it, directly or
    /// through another fold. Each stands in for the value wherever it is live.
    readers: Vec<SmallVec<[SsaVarId; 2]>>,
    /// For each phi result, the predecessors its copies are placed in, each
    /// with the operand it copies.
    copies: FxHashMap<SsaVarId, SmallVec<[(usize, SsaVarId); 2]>>,
    /// For each block, the index of the terminator its copies are placed
    /// before; `None` for a block that falls through, whose copies close it.
    terminators: Vec<Option<usize>>,
    /// Every instruction that may write memory, as `(block, instruction)`:
    /// each may change what a slot kept in memory holds.
    memory_writes: Vec<(usize, usize)>,
}

impl Listing {
    /// Records what the lowering makes of `ssa`.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function being lowered.
    /// * `defuse` - Its definitions and uses.
    /// * `rowless` - For each value index, whether the lowering folds it into its
    ///   use; a missing entry is a value with a row of its own.
    fn of<T: Target>(ssa: &SsaFunction<T>, defuse: &DefUseIndex<T>, rowless: &[bool]) -> Self {
        let count = ssa.variable_count();
        // The folded computation a use sits in, when it sits in one. A phi
        // operand is read by a copy, which is a row.
        let folded_user = |site: &UseSite| -> Option<SsaVarId> {
            if site.is_phi_operand() {
                return None;
            }
            let user = ssa
                .blocks()
                .get(site.block())?
                .instructions()
                .get(site.index())?
                .op()
                .defs()
                .next()?;
            rowless
                .get(user.index())
                .copied()
                .unwrap_or(false)
                .then_some(user)
        };
        let readers = (0..count)
            .map(|index| {
                let mut found: SmallVec<[SsaVarId; 2]> = SmallVec::new();
                // A folded value is rendered at every use it has - one, or
                // each memory access that shares one address - and a use
                // folded in turn is rendered at its own uses, so the
                // computations rendered around this one are every fold
                // reachable through the uses, not one path.
                let mut pending: SmallVec<[SsaVarId; 2]> =
                    SmallVec::from_elem(SsaVarId::from_index(index), 1);
                while let Some(value) = pending.pop() {
                    for site in defuse.uses_of(value).unwrap_or_default() {
                        if let Some(reader) = folded_user(site)
                            && !found.contains(&reader)
                        {
                            found.push(reader);
                            pending.push(reader);
                        }
                    }
                }
                found
            })
            .collect();
        let copies = ssa
            .blocks()
            .iter()
            .flat_map(|block| block.phi_nodes())
            .map(|phi| {
                let copied = phi
                    .operands()
                    .iter()
                    .map(|operand| (operand.predecessor(), operand.value()))
                    .collect();
                (phi.result(), copied)
            })
            .collect();
        let terminators = ssa
            .blocks()
            .iter()
            .map(|block| {
                let instructions = block.instructions();
                instructions
                    .last()
                    .filter(|last| last.op().is_terminator())
                    .map(|_| instructions.len().saturating_sub(1))
            })
            .collect();
        let memory_writes = ssa
            .blocks()
            .iter()
            .enumerate()
            .flat_map(|(block, instructions)| {
                instructions
                    .instructions()
                    .iter()
                    .enumerate()
                    .filter(|(_, instruction)| instruction.op().effects().writes_memory())
                    .map(move |(index, _)| (block, index))
            })
            .collect();
        Self {
            readers,
            copies,
            terminators,
            memory_writes,
        }
    }

    /// Returns the folded computations that read `value`.
    fn readers(&self, value: SsaVarId) -> &[SsaVarId] {
        self.readers
            .get(value.index())
            .map_or(&[], |readers| readers.as_slice())
    }
}

/// The interference test, decided once for every pair a gathering can ask.
///
/// Chaitin's condition, asked of the listing: two values interfere when either
/// is still to be read where the other's local is written. Merging two webs
/// asks it of every pair of their members, and the webs keep growing - over a
/// function of tens of thousands of values a register's web holds thousands,
/// and asking pair by pair is quadratic in them. So the answer is decided
/// once, value by value, as the values still to be read at each place a
/// value's local is written ([`Self::neighbours`]), and a merge asks only the
/// smaller web's neighbours whether they lie in the other.
///
/// Only pairs that could come to share a web are recorded: two values a
/// storage group, a phi operand offered to its result or a set that must
/// share links, directly or through others ([`Gathering::run`]'s
/// components). No other pair is ever asked.
struct Interference<'a, T: Target> {
    /// The analysis whose phi-out sets the probe consults.
    analysis: &'a LiveVariables,

    /// The solved liveness.
    live: &'a Liveness,

    /// Definitions and uses.
    defuse: &'a DefUseIndex<T>,

    /// Where the lowering reads and writes what the SSA does not say.
    listing: &'a Listing,

    /// What keeps two values apart whatever their lives.
    conflicts: Option<Conflicts<'a>>,

    /// Each value's neighbours: the values that interfere with it on its
    /// liveness, by index, sorted.
    neighbours: Vec<Vec<u32>>,

    /// Values with no definition the probe can place, which interfere with
    /// every other: merging blind is the failure this type exists to prevent.
    universal: Vec<bool>,

    /// Answers to whether two kinds conflict, keyed by the pair in order.
    kind_conflicts: FxHashMap<(u32, u32), bool>,
}

impl<'a, T: Target> Interference<'a, T> {
    /// Decides the interference among the values each component holds.
    ///
    /// # Arguments
    ///
    /// * `analysis` - The liveness analysis, for its phi-out sets.
    /// * `live` - The solved liveness.
    /// * `defuse` - Definitions and uses.
    /// * `listing` - What the lowering makes of the SSA.
    /// * `conflicts` - What keeps two values apart whatever their lives.
    /// * `component` - Each value index's component: only two values of one
    ///   component are ever asked about.
    fn new(
        analysis: &'a LiveVariables,
        live: &'a Liveness,
        defuse: &'a DefUseIndex<T>,
        listing: &'a Listing,
        conflicts: Option<Conflicts<'a>>,
        component: &[usize],
    ) -> Self {
        let mut interference = Self {
            analysis,
            live,
            defuse,
            listing,
            conflicts,
            neighbours: vec![Vec::new(); component.len()],
            universal: vec![false; component.len()],
            kind_conflicts: FxHashMap::default(),
        };
        interference.decide(component);
        interference
    }

    /// Records every interfering pair within a component
    /// ([`Self::neighbours`]).
    ///
    /// `who` is still to be read where `at`'s local is written when `who`, or a
    /// folded computation that reads it and stands in for it, is live there
    /// ([`Self::live_at_definition`], and after its copies in a predecessor,
    /// [`Self::read_after_copies`]). Only a value
    /// live out of the writing block, read by a successor's phi on its
    /// outgoing edge, live into it or read in it can be: those are the
    /// candidates each write is asked about.
    fn decide(&mut self, component: &[usize]) {
        let count = component.len();
        // How many values each component holds: one alone is never asked.
        let mut sizes: FxHashMap<usize, usize> = FxHashMap::default();
        for &root in component {
            let size = sizes.entry(root).or_insert(0usize);
            *size = size.saturating_add(1);
        }
        let shared = |value: usize| {
            component
                .get(value)
                .is_some_and(|root| sizes.get(root).copied().unwrap_or(0) > 1)
        };
        // The values each fold stands in for: `who` is read wherever a fold
        // that reads it is.
        let mut stands_for: Vec<SmallVec<[u32; 2]>> = vec![SmallVec::new(); count];
        for who in 0..count {
            for reader in self.listing.readers(SsaVarId::from_index(who)) {
                if let (Some(slot), Ok(who)) =
                    (stands_for.get_mut(reader.index()), u32::try_from(who))
                {
                    slot.push(who);
                }
            }
        }
        // Each block's readers: every value an instruction there reads, a phi
        // operand aside, read on its incoming edge.
        let blocks = self.listing.terminators.len();
        let mut read_in: Vec<Vec<u32>> = vec![Vec::new(); blocks];
        for value in 0..count {
            let Ok(index) = u32::try_from(value) else {
                continue;
            };
            for site in self
                .defuse
                .uses_of(SsaVarId::from_index(value))
                .unwrap_or_default()
            {
                if !site.is_phi_operand()
                    && let Some(readers) = read_in.get_mut(site.block())
                    && readers.last() != Some(&index)
                {
                    readers.push(index);
                }
            }
        }
        // The copies each block holds, of every successor's phis: they run
        // together before its terminator, whichever edge it takes.
        let mut copied_in: Vec<Vec<(SsaVarId, SsaVarId)>> = vec![Vec::new(); blocks];
        for (&result, copies) in &self.listing.copies {
            for &(predecessor, operand) in copies {
                if let Some(held) = copied_in.get_mut(predecessor) {
                    held.push((result, operand));
                }
            }
        }
        let mut seen = vec![usize::MAX; count];
        let mut stamp = 0usize;
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for value in 0..count {
            if !shared(value) {
                continue;
            }
            let at = SsaVarId::from_index(value);
            let Some(def) = self.defuse.def_site(at) else {
                if let Some(flag) = self.universal.get_mut(value) {
                    *flag = true;
                }
                continue;
            };
            let Ok(at_index) = u32::try_from(value) else {
                continue;
            };
            // The writes of `at`'s local: its definition, and for a phi
            // result the copy in each predecessor.
            let copies = self.listing.copies.get(&at).cloned().unwrap_or_default();
            let record = |reader: SsaVarId, edges: &mut Vec<(u32, u32)>| {
                let Ok(reader_index) = u32::try_from(reader.index()) else {
                    return;
                };
                let stood = stands_for
                    .get(reader.index())
                    .map_or(&[][..], |values| values.as_slice());
                for &who in once(&reader_index).chain(stood) {
                    if who != at_index && component.get(who as usize) == component.get(value) {
                        edges.push((who.min(at_index), who.max(at_index)));
                    }
                }
            };
            stamp = stamp.wrapping_add(1);
            for reader in self.candidates(def.block(), def.index().is_none(), &read_in) {
                if let Some(mark) = seen.get_mut(reader.index()) {
                    if *mark == stamp {
                        continue;
                    }
                    *mark = stamp;
                }
                if self.live_at_definition(reader, at) {
                    record(reader, &mut edges);
                }
            }
            for &(predecessor, operand) in &copies {
                stamp = stamp.wrapping_add(1);
                for reader in self.candidates(predecessor, false, &read_in) {
                    if let Some(mark) = seen.get_mut(reader.index()) {
                        if *mark == stamp {
                            continue;
                        }
                        *mark = stamp;
                    }
                    if reader != operand && self.read_after_copies(reader, predecessor) {
                        record(reader, &mut edges);
                    }
                }
                // Another phi result copied in the same block is written at
                // the same point, and its local must hold its own operand on
                // its edge while this one holds this operand on this one: one
                // local for both holds whichever copy ran last on every edge,
                // unless both copy one value.
                for &(other, other_operand) in copied_in.get(predecessor).into_iter().flatten() {
                    if other != at && other_operand != operand {
                        record(other, &mut edges);
                    }
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        for (a, b) in edges {
            if let Some(list) = self.neighbours.get_mut(a as usize) {
                list.push(b);
            }
            if let Some(list) = self.neighbours.get_mut(b as usize) {
                list.push(a);
            }
        }
        for list in &mut self.neighbours {
            list.sort_unstable();
        }
    }

    /// The values that can be still to be read where a local is written in
    /// `block`: those live out of it, read by a successor's phi on its
    /// outgoing edge or read in it, and for a write at its top those live
    /// into it.
    fn candidates<'r>(
        &'r self,
        block: usize,
        at_top: bool,
        read_in: &'r [Vec<u32>],
    ) -> impl Iterator<Item = SsaVarId> + 'r {
        let out = self
            .live
            .out_state(block)
            .into_iter()
            .flat_map(LivenessResult::variables);
        let phi_out = self
            .analysis
            .phi_out_set(block)
            .into_iter()
            .flat_map(BitSet::iter)
            .map(SsaVarId::from_index);
        let into = self
            .live
            .in_state(block)
            .filter(|_| at_top)
            .into_iter()
            .flat_map(LivenessResult::variables);
        let read = read_in
            .get(block)
            .into_iter()
            .flatten()
            .map(|value| SsaVarId::from_index(*value as usize));
        out.chain(phi_out).chain(into).chain(read)
    }

    /// Returns whether `a` and `b` may not be one variable.
    ///
    /// Two reasons, both a C variable's: one of them is still to be read where
    /// the other's local is written, or the caller knows they are not one
    /// ([`Gathering::conflicts`]). A register that holds a table's address and
    /// then the value read through it holds the two at different times, and
    /// still holds two variables.
    #[cfg(test)]
    fn interferes(&mut self, a: SsaVarId, b: SsaVarId) -> bool {
        a != b
            && (self.kinds_conflict(self.kind(a.index()), self.kind(b.index()))
                || self.universal.get(a.index()).copied().unwrap_or(false)
                || self.universal.get(b.index()).copied().unwrap_or(false)
                || self.neighbours.get(a.index()).is_some_and(|list| {
                    u32::try_from(b.index()).is_ok_and(|b| list.binary_search(&b).is_ok())
                }))
    }

    /// The kind the caller gives `value` ([`Conflicts::kinds`]), `None` where
    /// it gives none.
    fn kind(&self, value: usize) -> Option<u32> {
        self.conflicts?.kinds.get(value).copied()
    }

    /// Whether a value of kind `a` and one of kind `b` may not share a web.
    fn kinds_conflict(&mut self, a: Option<u32>, b: Option<u32>) -> bool {
        let (Some(conflicts), Some(a), Some(b)) = (self.conflicts, a, b) else {
            return false;
        };
        *self
            .kind_conflicts
            .entry((a.min(b), a.max(b)))
            .or_insert_with(|| (conflicts.conflict)(a, b))
    }

    /// Returns whether `who` is live just after `at` is defined.
    ///
    /// Precise to the instruction inside the defining block. A value read by
    /// the very instruction that defines `at` dies there - `x = x + 1` reads the
    /// old value and writes the new - and a value defined later in the block
    /// does not exist yet; counting either as live keeps apart values that are
    /// one variable on the page. Past the block the solved liveness answers,
    /// including the values a successor's phi reads on the outgoing edge, which
    /// the copies before the terminator read.
    ///
    /// "Later in the block" includes every instruction when `at` is defined at
    /// the block's top - an entry value, or a phi result. The value a stack slot
    /// arrives with and the value the body then stores back into it
    /// (`arg_0 = arg_0 + 1`) are one variable, but asking whether the stored
    /// value is live at the top of the entry block used to answer from its reads
    /// further down that block, before it was ever defined: the two were kept
    /// apart, and the arrival lost the slot's name to a temporary nothing
    /// assigns (measured on `dash-riscv64`).
    fn live_at_definition(&self, who: SsaVarId, at: SsaVarId) -> bool {
        let Some(def) = self.defuse.def_site(at) else {
            return true;
        };
        let block = def.block();
        let born_later = self.defuse.def_site(who).is_some_and(|site| {
            site.block() == block
                && site
                    .index()
                    .is_some_and(|index| def.index().is_none_or(|position| index > position))
        });
        if born_later {
            return false;
        }
        let live_past_block = self
            .live
            .out_state(block)
            .is_some_and(|state| state.is_live(who))
            || self
                .analysis
                .phi_out_set(block)
                .is_some_and(|phi_uses| phi_uses.contains(who.index()));
        if live_past_block {
            return true;
        }
        // A phi operand is recorded at the *join*, because that is where the
        // operand syntactically sits, but it is read on the incoming edge - so
        // only instruction uses count as reads inside the block.
        let read_after = |after: Option<usize>| {
            self.defuse.uses_of(who).is_some_and(|uses| {
                uses.iter().any(|site| {
                    site.block() == block
                        && !site.is_phi_operand()
                        && match after {
                            Some(position) => site.index() > position,
                            None => true,
                        }
                })
            })
        };
        match def.index() {
            // Defined at the top of the block: live there when it enters live
            // or anything in the block reads it.
            None => {
                self.live
                    .in_state(block)
                    .is_some_and(|state| state.is_live(who))
                    || read_after(None)
            }
            Some(position) => read_after(Some(position)),
        }
    }

    /// Returns whether `who` is still to be read after an instruction that
    /// may write memory runs: whether, kept in a slot a pointer can write, it
    /// could be read after the slot no longer holds it.
    ///
    /// A folded computation that reads `who` stands in for it, as it does for
    /// the pairwise test. An instruction that defines `who` is not one it
    /// outlives: a call's result is what the call leaves.
    fn outlives_a_memory_write(&self, who: SsaVarId) -> bool {
        once(who)
            .chain(self.listing.readers(who).iter().copied())
            .any(|reader| {
                self.listing
                    .memory_writes
                    .iter()
                    .any(|&(block, index)| self.live_after(reader, block, index))
            })
    }

    /// Returns whether `who` is live just after instruction `index` of
    /// `block`: defined before it, and read after it in the block or live out
    /// of the block.
    fn live_after(&self, who: SsaVarId, block: usize, index: usize) -> bool {
        let born_there_or_later = self.defuse.def_site(who).is_some_and(|site| {
            site.block() == block && site.index().is_some_and(|defined| defined >= index)
        });
        if born_there_or_later {
            return false;
        }
        self.live
            .out_state(block)
            .is_some_and(|state| state.is_live(who))
            || self
                .analysis
                .phi_out_set(block)
                .is_some_and(|phi_uses| phi_uses.contains(who.index()))
            || self.defuse.uses_of(who).is_some_and(|uses| {
                uses.iter().any(|site| {
                    site.block() == block && !site.is_phi_operand() && site.index() > index
                })
            })
    }

    /// Returns whether `who` is still read after the copies placed in
    /// `predecessor`: its terminator reads it, or it leaves the block live
    /// along any edge, since the copies run on all of them.
    fn read_after_copies(&self, who: SsaVarId, predecessor: usize) -> bool {
        self.live
            .out_state(predecessor)
            .is_some_and(|state| state.is_live(who))
            || self
                .listing
                .terminators
                .get(predecessor)
                .copied()
                .flatten()
                .is_some_and(|terminator| {
                    self.defuse.uses_of(who).is_some_and(|uses| {
                        uses.iter().any(|site| {
                            site.block() == predecessor
                                && !site.is_phi_operand()
                                && site.index() >= terminator
                        })
                    })
                })
    }

    /// Decides which phi results the lowering writes through a temporary: a
    /// result still read after its own copy in some predecessor.
    ///
    /// A copy that shares the result's local is never emitted, so it cannot
    /// overwrite anything and needs no temporary.
    ///
    /// # Arguments
    ///
    /// * `representative` - The finished coalescing, by value index.
    ///
    /// # Returns
    ///
    /// Whether each value index is a staged phi result.
    fn staged(&self, representative: &[usize]) -> Vec<bool> {
        let class = |value: SsaVarId| {
            representative
                .get(value.index())
                .copied()
                .unwrap_or(value.index())
        };
        let mut staged = vec![false; representative.len()];
        for (&result, copies) in &self.listing.copies {
            let needed = copies.iter().any(|&(predecessor, operand)| {
                class(operand) != class(result)
                    && once(result)
                        .chain(self.listing.readers(result).iter().copied())
                        .any(|reader| {
                            reader != operand && self.read_after_copies(reader, predecessor)
                        })
            });
            if needed && let Some(slot) = staged.get_mut(result.index()) {
                *slot = true;
            }
        }
        staged
    }
}

/// One gathering of values into webs: what is offered, and where it is read.
///
/// The engine both consumers run. The lowering's coalescing and the register
/// identity differ in the values the listing folds, in the storage groups they
/// offer, and in which phi operands may join their result - never in how a join
/// is decided.
pub struct Gathering<'g> {
    /// For each value index, whether the lowering renders it without a row of
    /// its own - folded into its use - so that its operands are read where it is.
    /// Empty when nothing is lowered.
    pub rowless: &'g [bool],
    /// The values kept in one place, each group offered to itself first-fit in
    /// the order given, after the phi joins.
    pub groups: Vec<Group>,
    /// What keeps two values from sharing a web however their lives fall -
    /// two whose recovered types admit no one type, say. `None` constrains
    /// nothing.
    pub conflicts: Option<Conflicts<'g>>,
    /// Sets of value indices that must be one variable, each one web before
    /// anything else is offered.
    ///
    /// They are joined without asking the interference test - the host knows
    /// the values are never live together in a way liveness cannot show, or
    /// that sharing one place is the only lowering there is. The values a
    /// throw delivers to a handler's phi are the case: no copy runs on an
    /// exceptional edge, so the handler reads the place the protected code
    /// wrote, and liveness, which puts an exceptional operand's use at the end
    /// of the throwing block, reports versions of one variable as overlapping
    /// when they hold the place one after the other. Everything offered
    /// afterwards is tested against the whole web, these members included.
    pub must_share: &'g [Vec<usize>],
    /// Value indices each of which is a variable of its own: its web takes in
    /// what a storage group alone offers only where one of the two is
    /// computed from the other, a member's definition reading a member of
    /// the other web.
    ///
    /// A parameter is the case. A register a parameter arrives in holds,
    /// after the parameter's last read, whatever the function computes next,
    /// and the parameter's update (`x = x + 1`, `p = p->next`) is the
    /// parameter again where an unrelated value is not: joined, the
    /// unrelated value's type declared the parameter. A phi still joins its
    /// operands as it would.
    pub anchored: &'g [usize],
}

/// What a caller knows keeps two values apart whatever their lives: each
/// value's kind, and which two kinds admit no one variable.
///
/// A conflict is a question of kinds, never of the values that carry them,
/// so a web is compared with another by the kinds each holds, however many
/// values carry them: a register's thousands of values over a large function
/// are a few types.
#[derive(Clone, Copy)]
pub struct Conflicts<'g> {
    /// Each value's kind, by value index; a value past the end has none and
    /// conflicts with nothing.
    pub kinds: &'g [u32],
    /// Whether a value of one kind and a value of another may not share a
    /// web; asked of a kind and itself too, for two values of one kind.
    pub conflict: &'g dyn Fn(u32, u32) -> bool,
}

/// Values kept in one place, offered to one another.
pub struct Group {
    /// The values, by index, in the order they are offered.
    pub values: Vec<usize>,
    /// Whether the place is a frame slot kept in memory, which a pointer may
    /// write where no access names it.
    ///
    /// A value read out of such a slot is its content only until something
    /// that may write memory runs; one still read after that is not the slot
    /// there, and is not offered to it ([`Gathered::detached`]). A shared
    /// abstract local also withholds this physical name if any member belongs
    /// outside this group or any physical member is stale. This preserves the
    /// abstract coalescing without introducing physical writes.
    pub in_memory: bool,
}

/// The webs a [`Gathering`] found.
///
/// The empty value is the gathering that shares nothing: every value its own
/// web, nothing staged, nothing detached.
#[derive(Debug, Clone, Default)]
pub struct Gathered {
    /// Value index -> the index of the representative of its web.
    pub representative: Vec<usize>,
    /// Value index -> whether the value is a phi result the lowering writes
    /// through a temporary: one still read after its own copy in some
    /// predecessor, which no choice of webs can keep the copy from overwriting.
    pub staged: Vec<bool>,
    /// Value index -> whether the value's physical storage cannot name its
    /// shared abstract local. A stale content, a web member outside the physical
    /// group, or conflicting physical groups withhold the name. Sharing an
    /// abstract local does not authorize writes to addressable storage.
    pub detached: Vec<bool>,
}

impl Gathered {
    /// Returns the representative of `value`'s web: the value whose variable
    /// it shares.
    #[must_use]
    pub fn representative(&self, value: usize) -> usize {
        self.representative.get(value).copied().unwrap_or(value)
    }

    /// Returns whether the phi result `value` is written through a temporary.
    ///
    /// A phi's copies stand before each predecessor's terminator and run on
    /// every edge out of it. A result that is itself still read there - by the
    /// terminator, or along another edge - would be overwritten while the
    /// program still holds it, and no choice of webs can prevent that, because
    /// the conflict is between the result and itself. Such a result is copied
    /// into a temporary on every incoming edge and moved into its own variable
    /// where the phi stands.
    #[must_use]
    pub fn is_staged(&self, value: usize) -> bool {
        self.staged.get(value).copied().unwrap_or(false)
    }

    /// Returns whether `value` must withhold its physical storage name.
    ///
    /// Its shared abstract local contains a stale physical member or values
    /// outside that physical group, so the local cannot stand for its storage.
    #[must_use]
    pub fn is_detached(&self, value: usize) -> bool {
        self.detached.get(value).copied().unwrap_or(false)
    }

    /// The copies taking `ssa` out of SSA form places, as
    /// `(predecessor block, phi result, operand)` in phi order: the order the
    /// blocks, their phis and the phis' operands come in. Grouped by
    /// predecessor, each group is one
    /// [`ParallelCopy`](crate::analysis::copies::ParallelCopy).
    ///
    /// An operand that shares its result's web needs no copy: it would read
    /// and write one variable. A staged result is filled on every incoming
    /// edge, since it is moved out of its temporary where the phi stands
    /// whichever edge was taken - an operand sharing its web included.
    #[must_use]
    pub fn phi_copies<T: Target>(&self, ssa: &SsaFunction<T>) -> Vec<(usize, SsaVarId, SsaVarId)> {
        let mut copies = Vec::new();
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                let result = phi.result();
                for operand in phi.operands() {
                    let value = operand.value();
                    if !self.is_staged(result.index())
                        && self.representative(result.index()) == self.representative(value.index())
                    {
                        continue;
                    }
                    copies.push((operand.predecessor(), result, value));
                }
            }
        }
        copies
    }
}

impl Gathering<'_> {
    /// Gathers `ssa`'s values into webs.
    ///
    /// Each set [`Self::must_share`] names is one web first. Then each phi
    /// operand `joins_phi` accepts joins its result's web, and each group's
    /// values join the first web of the group they interfere with nothing in.
    /// Webs are compared whole: a value that would come to share a web with
    /// three others is checked against all three.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function whose values are gathered.
    /// * `defuse` - Its definitions and uses.
    /// * `joins_phi` - Given a phi result's index, an operand and the operand's
    ///   index, whether the operand is offered to the result's web.
    #[must_use]
    pub fn run<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        defuse: &DefUseIndex<T>,
        joins_phi: impl Fn(usize, SsaVarId, usize) -> bool,
    ) -> Gathered {
        let count = ssa.variable_count();
        // A function with no phi and no group of two has nothing to gather, and
        // everything below - the exception-aware CFG, the liveness solve, one web
        // per value - is work whose answer is already known to be the identity.
        // This is a guard, not an optimisation: it belongs before the analysis,
        // not inside it.
        let has_phi = ssa
            .blocks()
            .iter()
            .any(|block| !block.phi_nodes().is_empty());
        if count == 0
            || (!has_phi
                && self.must_share.iter().all(|set| set.len() < 2)
                && self
                    .groups
                    .iter()
                    .all(|group| group.values.len() < 2 && !group.in_memory))
        {
            return Gathered {
                representative: (0..count).collect(),
                staged: vec![false; count],
                detached: vec![false; count],
            };
        }

        // The exception-aware view, not the bare terminator relation: a handler
        // entry is reached by the runtime rather than by a terminator, so solving
        // over `SsaCfg` alone would treat handler code as unreachable and report
        // nothing live across it.
        let graph = SsaCfg::from_ssa(ssa);
        // Two instances: the solver consumes the analysis, and the OUT states it
        // returns are the *terminator* liveness - they do not include the values a
        // successor's phi reads on this block's outgoing edges. Those are exactly
        // the values a join must see, so `phi_out_set` is consulted alongside
        // them; see `Interference::live_at_definition`.
        let analysis = LiveVariables::new(ssa);
        let live = DataFlowSolver::new(LiveVariables::new(ssa)).solve(ssa, &graph);
        let listing = Listing::of(ssa, defuse, self.rowless);
        // What could come to share a web: values a must-share set, an offered
        // phi operand or a storage group links, directly or through others.
        // Interference is decided within each, and asked nowhere else.
        // Phi webs chain: a result coalesced with an operand may itself be
        // another phi's operand, so the classes merge rather than being
        // decided pairwise.
        let mut components = Classes::new(count);
        let mut link = |a: usize, b: usize| {
            components.union(a, b);
        };
        for set in self.must_share {
            for pair in set.windows(2) {
                if let [a, b] = pair {
                    link(*a, *b);
                }
            }
        }
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                let Some(result) = ssa.var_index(phi.result()) else {
                    continue;
                };
                for operand in phi.operands() {
                    if let Some(index) = ssa.var_index(operand.value())
                        && joins_phi(result, operand.value(), index)
                    {
                        link(result, index);
                    }
                }
            }
        }
        for group in &self.groups {
            for pair in group.values.windows(2) {
                if let [a, b] = pair {
                    link(*a, *b);
                }
            }
        }
        let component: Vec<usize> = (0..count).map(|value| components.find(value)).collect();
        let mut interference = Interference::new(
            &analysis,
            &live,
            defuse,
            &listing,
            self.conflicts,
            &component,
        );
        let mut webs = Webs::new(count, &interference, self.anchored);

        // The sets the host requires first, unconditionally.
        for set in self.must_share {
            if let Some((&first, rest)) = set.split_first() {
                for &value in rest {
                    webs.force_join(first, value);
                }
            }
        }

        // Phi webs next: a phi copy is a row, and a storage group only decides a
        // name.
        //
        // A phi's operands are offered in the order of what refusing one costs,
        // and one costs more than the rest: an operand arriving from a block
        // with no instruction of its own.
        //
        // A refused operand leaves a copy in the block it arrives from. Where
        // that block holds nothing but its terminator - an edge the lift split,
        // between two blocks that have instructions of their own - the copy is
        // the only reason the block is written at all, and it reaches the
        // listing as a block of no size holding a copy and a jump. Refusing any
        // other operand costs its copy alone, written among rows that are there
        // anyway.
        //
        // Offered last, such an operand meets the class an earlier operand has
        // joined and is refused by what that class has brought in - a value it
        // interferes with, or one whose recovered type cannot be its own -
        // rather than by anything it is related to. That is how a loop counter
        // and its own decrement came to sit in two webs, with the decrement's
        // copy in a block of no size on the back edge: the entering operand had
        // joined first, and its own class held a pointer the decrement's
        // integer could not share a local with.
        //
        // Preferring every loop-carried operand, or every operand on a critical
        // edge, moves copies elsewhere instead of removing them - measured on
        // the fixtures; this is the condition that pays.
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                let Some(result) = ssa.var_index(phi.result()) else {
                    continue;
                };
                // False sorts first: the operand whose refusal writes a block.
                let cheaper = |operand: &&PhiOperand| {
                    ssa.blocks().get(operand.predecessor()).is_none_or(|from| {
                        from.instructions().len() > usize::from(from.control_terminator().is_some())
                    })
                };
                let mut operands: Vec<&PhiOperand> = phi.operands().iter().collect();
                // Stable, so operands whose edges cost alike keep the order the
                // phi states them in and the webs stay a function of the IR.
                operands.sort_by_key(cheaper);
                for operand in operands {
                    let Some(index) = ssa.var_index(operand.value()) else {
                        continue;
                    };
                    if joins_phi(result, operand.value(), index) {
                        webs.join(result, index, &mut interference);
                    }
                }
            }
        }
        // A value no operation defines, no phi merges and nothing reads is not in
        // the program: an entry placeholder SSA construction rewrote to its key's
        // one entry value, or an arrival every read of which the optimizer
        // removed. It keeps its storage origin all the same. Offered to its
        // storage's group it founds a web - it comes first in value order - and
        // the value that really arrives in that storage, defined at the same
        // point and live there, is refused by a value no row writes: measured
        // on `dash-riscv64`, the live arrival then lost the slot's name to the
        // web the dead one founded and printed as a temporary nothing assigns.
        let mut detached = vec![false; count];
        for group in &self.groups {
            let mut firsts: Vec<usize> = Vec::new();
            for &value in group
                .values
                .iter()
                .filter(|value| !defuse.is_absent(SsaVarId::from_index(**value)))
            {
                if group.in_memory
                    && interference.outlives_a_memory_write(SsaVarId::from_index(value))
                {
                    if let Some(flag) = detached.get_mut(value) {
                        *flag = true;
                    }
                    continue;
                }
                if !firsts.iter().any(|&first| {
                    webs.derives(first, value, defuse) && webs.join(first, value, &mut interference)
                }) {
                    firsts.push(value);
                }
            }
        }

        let representative: Vec<usize> = (0..count).map(|value| webs.find(value)).collect();
        self.detach_memory_webs(&representative, &mut detached);
        let staged = interference.staged(&representative);
        Gathered {
            representative,
            staged,
            detached,
        }
    }

    /// Detaches physical names unless every member of the shared local can use them.
    ///
    /// Liveness permits sharing an abstract local across unrelated definitions.
    /// A physical slot also retains bytes between its explicit accesses, so an
    /// unrelated definition in that local would introduce a write to the slot.
    /// A stale member likewise cannot regain its physical name through a phi.
    /// Keep the abstract coalescing and withhold the physical name instead.
    fn detach_memory_webs(&self, representative: &[usize], detached: &mut [bool]) {
        if !self.groups.iter().any(|group| group.in_memory) {
            return;
        }
        // Zero means no physical group; MAX means overlapping physical groups.
        // A group id is one plus its index in an allocated vector, so it can
        // reach neither sentinel. Only functions with physical groups pay for
        // these two dense arrays and the linear membership pass.
        let mut owners = vec![0usize; representative.len()];
        for (index, group) in self
            .groups
            .iter()
            .enumerate()
            .filter(|(_, group)| group.in_memory)
        {
            let id = index.saturating_add(1);
            for &value in &group.values {
                if let Some(owner) = owners.get_mut(value) {
                    *owner = if *owner == 0 || *owner == id {
                        id
                    } else {
                        usize::MAX
                    };
                }
            }
        }
        let mut invalid = vec![false; representative.len()];
        for ((&owner, &root), &stale) in owners.iter().zip(representative).zip(detached.iter()) {
            if (stale || owners.get(root) != Some(&owner))
                && let Some(flag) = invalid.get_mut(root)
            {
                *flag = true;
            }
        }
        for (index, group) in self
            .groups
            .iter()
            .enumerate()
            .filter(|(_, group)| group.in_memory)
        {
            let id = index.saturating_add(1);
            for &value in &group.values {
                let Some(&root) = representative.get(value) else {
                    continue;
                };
                if (owners.get(root) != Some(&id) || invalid.get(root).copied().unwrap_or(true))
                    && let Some(flag) = detached.get_mut(value)
                {
                    *flag = true;
                }
            }
        }
    }
}

/// Partitions `items` greedily: each joins the first group none of whose
/// members it conflicts with, or opens a new one.
///
/// For a partition with no liveness to consult - storage a host knows only by
/// which pairs cannot share it, say two values of different types in one
/// slot. The result depends on the order of `items`, which is the caller's to
/// make canonical.
///
/// A conflict is a question of the items, so a group holds each distinct item
/// once: partitioning values by their kinds ([`Conflicts::kinds`]) asks each
/// group of the few kinds it holds, not of every value that carries them.
///
/// # Arguments
///
/// * `items` - What to partition, in the order groups are filled.
/// * `conflict` - Whether two items may not share a group.
///
/// # Returns
///
/// Each item's group, by position in `items`; groups are numbered from 0 in
/// the order they open.
pub fn first_fit_partition<K: Copy + PartialEq>(
    items: &[K],
    conflict: impl Fn(K, K) -> bool,
) -> Vec<usize> {
    let mut groups: Vec<Vec<K>> = Vec::new();
    let mut assignment = Vec::with_capacity(items.len());
    for &item in items {
        let group = groups
            .iter()
            .position(|members| members.iter().all(|&member| !conflict(member, item)))
            .unwrap_or(groups.len());
        if group == groups.len() {
            groups.push(Vec::new());
        }
        if let Some(members) = groups.get_mut(group)
            && !members.contains(&item)
        {
            members.push(item);
        }
        assignment.push(group);
    }
    assignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{SsaDefSpec, SsaFunctionBuilder, SsaOp},
        testing::{self, MockTarget, MockType},
    };

    /// A folded value is rendered at every use it has, and a use folded in
    /// turn at its own: what such a fold reads is read by each of them.
    ///
    /// ```text
    /// base = arg0
    /// one  = 1
    /// addr = base + one     folded: the address both accesses share
    /// store [addr], one
    /// read = load [addr]    folded into its use
    /// sum  = read + one
    /// return sum
    /// ```
    ///
    /// `base` is read where `sum` is rendered, through `read`, though `addr`
    /// has two uses.
    #[test]
    fn a_shared_fold_is_read_wherever_each_use_is_rendered() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let (base, addr, read) = builder
            .in_block(0, |block| {
                let base = block.load_arg(SsaDefSpec::argument(0, MockType::I32), 0)?;
                let one = block.const_i32(SsaDefSpec::tmp(MockType::I32), 1)?;
                let addr = block.add(SsaDefSpec::tmp(MockType::I32), base, one)?;
                block.store_indirect(addr, one, MockType::I32)?;
                let read =
                    block.load_indirect(SsaDefSpec::tmp(MockType::I32), addr, MockType::I32)?;
                let sum = block.add(SsaDefSpec::tmp(MockType::I32), read, one)?;
                block.ret(Some(sum))?;
                Ok((base, addr, read))
            })
            .expect("the block builds");
        let ssa = builder.finish().expect("the function builds");
        let defuse = DefUseIndex::build_with_ops(&ssa);
        let mut rowless = vec![false; ssa.variable_count()];
        rowless[addr.index()] = true;
        rowless[read.index()] = true;

        let listing = Listing::of(&ssa, &defuse, &rowless);
        let readers = listing.readers(base);
        assert!(
            readers.contains(&addr) && readers.contains(&read),
            "{readers:?}"
        );
    }

    /// Each item takes the first group it fits; one conflicting with every
    /// group opens a new one.
    #[test]
    fn first_fit_fills_the_earliest_group_that_admits_the_item() {
        // Even and odd items conflict; items of one parity may share.
        let groups = first_fit_partition(&[0u8, 1, 2, 3, 4], |a, b| a % 2 != b % 2);
        assert_eq!(groups, [0, 1, 0, 1, 0]);
        assert!(first_fit_partition::<u8>(&[], |_, _| true).is_empty());
    }

    /// A phi result and its back-edge operand interfere when both are live at
    /// the block's exit - which is the case the unsound shortcut gets wrong.
    ///
    /// `loop_counter_fixture` is the shape:
    ///
    /// ```text
    /// B1: counter = phi(zero@B0, next@B1)
    ///     next    = counter + 1
    ///     branch keep_going -> B1, B2
    /// B2: return counter
    /// ```
    ///
    /// `next`'s only use is the phi, so "coalesce when the operand is
    /// single-use" would merge it with `counter`. They overlap: `counter` is
    /// live out of B1 because B2 returns it, and `next` is live out of B1
    /// because the back edge carries it into the phi. One local for both is
    /// wrong code, and the SSA stays well-formed - nothing downstream would
    /// report it.
    #[test]
    fn a_loop_counter_and_its_increment_cannot_share_a_local() {
        let ssa = testing::loop_counter_fixture();
        let graph = SsaCfg::from_ssa(&ssa);
        let analysis = LiveVariables::new(&ssa);
        let live = DataFlowSolver::new(LiveVariables::new(&ssa)).solve(&ssa, &graph);
        let defuse = DefUseIndex::build_with_ops(&ssa);

        let (counter, back_edge) = ssa
            .blocks()
            .iter()
            .find_map(|block| {
                let phi = block.phi_nodes().first()?;
                let operand = phi
                    .operands()
                    .iter()
                    .find(|operand| operand.predecessor() == block.id())?;
                Some((phi.result(), operand.value()))
            })
            .expect("the fixture carries a self-referencing phi");

        assert!(
            Interference::new(
                &analysis,
                &live,
                &defuse,
                &Listing::of(&ssa, &defuse, &[]),
                None,
                &vec![0; ssa.variable_count()]
            )
            .interferes(counter, back_edge),
            "both are live at the loop block's exit, so they need separate locals"
        );
    }

    /// **Two phi results copied in one block on different edges interfere.**
    /// A block that ends in a branch holds the copies of every successor's
    /// phis, and runs them all whichever edge it takes:
    ///
    /// ```text
    /// B0: start = arg0
    ///     branch start < 3 -> B1, B2
    /// B1: p = phi(start@B0, n@B1)
    ///     n = p + 1
    ///     t = p + 3
    ///     branch n < t -> B1, B2      copies: p = n (to B1), r = t (to B2)
    /// B2: r = phi(start@B0, t@B1)
    ///     return r
    /// ```
    ///
    /// One local for `p` and `r` is written `n` and then `t` before the
    /// branch, and the back edge carries `t` into the loop where `n` belongs
    /// (`dash-arm` `sub_d9b8`: `reg_0 = var_5; reg_0 = reg_14;` in its loop).
    /// In SSA they never meet: each is defined in its own block. Where both
    /// copies write one value, `r = n` here, one local holds it either way.
    #[test]
    fn phi_results_copied_in_one_block_on_different_edges_interfere() {
        let fixture = |exit_takes_next: bool| {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 2);
            builder.ensure_block(2);
            let start = builder
                .in_block(0, |block| {
                    let start = block.load_arg(SsaDefSpec::argument(0, MockType::I32), 0)?;
                    let three = block.const_i32(SsaDefSpec::tmp(MockType::I32), 3)?;
                    let small = block.clt(SsaDefSpec::tmp(MockType::I32), start, three, false)?;
                    block.branch(small, 1, 2)?;
                    Ok(start)
                })
                .expect("B0 builds");
            let (p, n, t) = builder
                .in_block(1, |block| {
                    let p = block.empty_phi(SsaDefSpec::local(0, MockType::I32))?;
                    let one = block.const_i32(SsaDefSpec::tmp(MockType::I32), 1)?;
                    let n = block.add(SsaDefSpec::tmp(MockType::I32), p, one)?;
                    let three = block.const_i32(SsaDefSpec::tmp(MockType::I32), 3)?;
                    let t = block.add(SsaDefSpec::tmp(MockType::I32), p, three)?;
                    let more = block.clt(SsaDefSpec::tmp(MockType::I32), n, t, false)?;
                    block.branch(more, 1, 2)?;
                    Ok((p, n, t))
                })
                .expect("B1 builds");
            let r = builder
                .in_block(2, |block| {
                    let r = block.empty_phi(SsaDefSpec::local(1, MockType::I32))?;
                    block.ret(Some(r))?;
                    Ok(r)
                })
                .expect("B2 builds");
            builder.add_phi_operand(1, p, 0, start).expect("operand");
            builder.add_phi_operand(1, p, 1, n).expect("operand");
            builder.add_phi_operand(2, r, 0, start).expect("operand");
            builder
                .add_phi_operand(2, r, 1, if exit_takes_next { n } else { t })
                .expect("operand");
            (builder.finish().expect("the function builds"), p, r)
        };
        for (exit_takes_next, expected) in [(false, true), (true, false)] {
            let (ssa, p, r) = fixture(exit_takes_next);
            let graph = SsaCfg::from_ssa(&ssa);
            let analysis = LiveVariables::new(&ssa);
            let live = DataFlowSolver::new(LiveVariables::new(&ssa)).solve(&ssa, &graph);
            let defuse = DefUseIndex::build_with_ops(&ssa);
            assert_eq!(
                Interference::new(
                    &analysis,
                    &live,
                    &defuse,
                    &Listing::of(&ssa, &defuse, &[]),
                    None,
                    &vec![0; ssa.variable_count()]
                )
                .interferes(p, r),
                expected,
                "the exit copies {}",
                if exit_takes_next { "n" } else { "t" }
            );
        }
    }

    /// **A phi operand whose copy would be a block of its own is offered
    /// first.** A loop counter entered from a value of another kind: the
    /// decrement arrives on the back edge, from a block that holds nothing but
    /// its jump, and the entering operand carries a recovered type the
    /// decrement's cannot be joined with - a register that held a pointer
    /// before the loop counted with it. Offered in the order the phi states
    /// them, the entering operand joins and the decrement is then refused by
    /// the type it brought, which leaves the decrement's copy on the back edge,
    /// in a block with nothing else in it. Offered first, the decrement joins
    /// and the entering operand is refused instead, whose copy goes in the
    /// entering block among the rows already there.
    ///
    /// The control is the same function with the entering operand's kind made
    /// to agree: then nothing is refused and both operands share the web
    /// whatever the order, so the test cannot pass by the gathering joining
    /// everything.
    #[test]
    fn a_phi_operand_whose_copy_would_be_a_block_of_its_own_is_offered_first() {
        let spec = || SsaDefSpec::<MockTarget>::tmp(MockType::I32);
        let fixture = || {
            let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 4);
            for block in 1..4 {
                builder.ensure_block(block);
            }
            let entered = builder
                .in_block(0, |block| {
                    let held = block.load_arg(SsaDefSpec::argument(0, MockType::I32), 0)?;
                    let one = block.const_i32(spec(), 1)?;
                    // A row of its own, so a copy left here has rows to sit
                    // among - and so that it is no constant, which is never
                    // coalesced.
                    let entered = block.add(spec(), held, one)?;
                    block.jump(1)?;
                    Ok(entered)
                })
                .expect("B0 builds");
            let (counter, decrement) = builder
                .in_block(1, |block| {
                    let counter = block.empty_phi(SsaDefSpec::local(0, MockType::I32))?;
                    let step = block.const_i32(spec(), -1)?;
                    let decrement = block.add(spec(), counter, step)?;
                    let zero = block.const_i32(spec(), 0)?;
                    let going = block.clt(spec(), zero, decrement, false)?;
                    block.branch(going, 2, 3)?;
                    Ok((counter, decrement))
                })
                .expect("B1 builds");
            // The back edge's block holds nothing but its jump: a copy left
            // here is the only reason it would be written at all.
            builder
                .in_block(2, |block| block.jump(1))
                .expect("B2 builds");
            builder
                .in_block(3, |block| block.ret(Some(decrement)))
                .expect("B3 builds");
            builder
                .add_phi_operand(1, counter, 0, entered)
                .expect("the loop is entered");
            builder
                .add_phi_operand(1, counter, 2, decrement)
                .expect("the loop carries its counter");
            (
                builder.finish().expect("the function builds"),
                entered,
                counter,
                decrement,
            )
        };
        // Kind 1 is the entering operand's, kind 2 the decrement's, and the two
        // cannot be one variable; in the control every value is of one kind.
        for conflicting in [true, false] {
            let (ssa, entered, counter, decrement) = fixture();
            let mut kinds = vec![0u32; ssa.variable_count()];
            if conflicting {
                for (value, kind) in [(entered, 1), (decrement, 2)] {
                    if let Some(slot) = ssa.var_index(value).and_then(|at| kinds.get_mut(at)) {
                        *slot = kind;
                    }
                }
            }
            let defuse = DefUseIndex::build_with_ops(&ssa);
            let conflict = |a: u32, b: u32| a != 0 && b != 0 && a != b;
            let gathered = Gathering {
                rowless: &[],
                groups: Vec::new(),
                conflicts: Some(Conflicts {
                    kinds: &kinds,
                    conflict: &conflict,
                }),
                must_share: &[],
                anchored: &[],
            }
            .run(&ssa, &defuse, |_, _, _| true);
            let web = |value: SsaVarId| {
                ssa.var_index(value)
                    .and_then(|index| gathered.representative.get(index).copied())
                    .expect("every value has a web")
            };
            assert_eq!(
                web(counter),
                web(decrement),
                "the counter and its decrement share a web, conflicting {conflicting}"
            );
            assert_eq!(
                web(counter) == web(entered),
                !conflicting,
                "the entering operand is the one refused, conflicting {conflicting}"
            );
        }
    }

    /// **A variable of its own takes in only what is computed from it.** A
    /// register holds a parameter, then its update `p + 1`, and after the
    /// update's last read a value computed from neither. The parameter is
    /// anchored ([`Gathering::anchored`]): its update, which reads it, joins
    /// its web, and the unrelated value, which only the register relates to
    /// either, keeps a web of its own. The control: with nothing anchored
    /// the unrelated value joins the web, so the test cannot pass by the
    /// gathering keeping everything apart.
    #[test]
    fn a_variable_of_its_own_takes_in_only_what_is_computed_from_it() {
        let spec = || SsaDefSpec::<MockTarget>::tmp(MockType::I32);
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 4);
        let (parameter, update, unrelated) = builder
            .in_block(0, |block| {
                let parameter = block.load_arg(SsaDefSpec::argument(0, MockType::I32), 0)?;
                let one = block.const_i32(spec(), 1)?;
                let update = block.add(spec(), parameter, one)?;
                let doubled = block.add(spec(), update, update)?;
                let seven = block.const_i32(spec(), 7)?;
                let unrelated = block.add(spec(), seven, one)?;
                let sum = block.add(spec(), doubled, unrelated)?;
                block.ret(Some(sum))?;
                Ok((parameter, update, unrelated))
            })
            .expect("B0 builds");
        let ssa = builder.finish().expect("the function builds");
        let index = |value: SsaVarId| ssa.var_index(value).expect("a value of the function");
        let defuse = DefUseIndex::build_with_ops(&ssa);
        for anchoring in [true, false] {
            let anchored = [index(parameter)];
            let gathered = Gathering {
                rowless: &[],
                groups: vec![Group {
                    values: [parameter, update, unrelated]
                        .into_iter()
                        .map(index)
                        .collect(),
                    in_memory: false,
                }],
                conflicts: None,
                must_share: &[],
                anchored: if anchoring { &anchored } else { &[] },
            }
            .run(&ssa, &defuse, |_, _, _| true);
            let web = |value: SsaVarId| {
                gathered
                    .representative
                    .get(index(value))
                    .copied()
                    .expect("every value has a web")
            };
            assert_eq!(
                web(parameter),
                web(update),
                "the update reads the parameter, anchored {anchoring}"
            );
            assert_eq!(
                web(unrelated) == web(parameter),
                !anchoring,
                "only the register relates the unrelated value, anchored {anchoring}"
            );
        }
    }

    /// **A register's thousands of values gather in time linear in what
    /// interferes.** Every value of a chain of 3,000 diamonds offered as one
    /// storage group, the shape of a register reused across a large function:
    /// each value joins the first web it interferes with nothing in, which
    /// holds nearly every value before it, so asking pair by pair is quadratic
    /// in the group (tens of millions of probes here, minutes), and asking the
    /// smaller web's neighbours is not. The webs are the ones liveness says:
    /// each diamond's arms join its phi, and no two values live at once share
    /// one.
    #[test]
    fn a_large_storage_group_gathers_in_time_linear_in_its_interference() {
        let ssa = testing::diamond_chain_fixture(3_000);
        let defuse = DefUseIndex::build_with_ops(&ssa);
        let values: Vec<usize> = (0..ssa.variable_count()).collect();
        let started = std::time::Instant::now();
        let gathered = Gathering {
            rowless: &[],
            groups: vec![Group {
                values,
                in_memory: false,
            }],
            conflicts: None,
            must_share: &[],
            anchored: &[],
        }
        .run(&ssa, &defuse, |_, _, _| true);
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "{elapsed:?} for {} values",
            ssa.variable_count()
        );
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                for operand in phi.operands() {
                    assert_eq!(
                        gathered.representative(phi.result().index()),
                        gathered.representative(operand.value().index()),
                        "an arm joins its phi"
                    );
                }
            }
        }
        let webs: std::collections::BTreeSet<usize> = (0..ssa.variable_count())
            .map(|value| gathered.representative(value))
            .collect();
        assert!(
            webs.len() < ssa.variable_count() / 4,
            "{} webs for {} values",
            webs.len(),
            ssa.variable_count()
        );
    }

    /// A must-share set is one web whatever the interference test says, and the
    /// plain run still keeps the pair apart.
    #[test]
    fn a_must_share_set_is_one_web() {
        let ssa = testing::loop_counter_fixture();
        let defuse = DefUseIndex::build_with_ops(&ssa);
        let (counter, back_edge) = ssa
            .blocks()
            .iter()
            .find_map(|block| {
                let phi = block.phi_nodes().first()?;
                let operand = phi
                    .operands()
                    .iter()
                    .find(|operand| operand.predecessor() == block.id())?;
                Some((phi.result(), operand.value()))
            })
            .expect("the fixture carries a self-referencing phi");
        let gathering = Gathering {
            rowless: &[],
            groups: Vec::new(),
            conflicts: None,
            must_share: &[],
            anchored: &[],
        };

        let apart = gathering.run(&ssa, &defuse, |_, _, _| true);
        assert_ne!(
            apart.representative(counter.index()),
            apart.representative(back_edge.index()),
            "the two interfere, so the phi join is refused"
        );

        let shared = [vec![counter.index(), back_edge.index()]];
        let forced = Gathering {
            must_share: &shared,
            anchored: &[],
            ..gathering
        }
        .run(&ssa, &defuse, |_, _, _| true);
        assert_eq!(
            forced.representative(counter.index()),
            forced.representative(back_edge.index())
        );
    }

    /// Values that never overlap do coalesce.
    ///
    /// The guard above only proves the check says "no" somewhere; one that
    /// never says "yes" would pass with the interference test hardwired true,
    /// and the copies this exists to remove would all survive.
    ///
    /// The pair has to be genuinely unrelated, and this is where the original
    /// version of this test went wrong: it asserted `!interferes(counter,
    /// counter)`, which `interferes` answers from its own `a != b` guard
    /// before consulting liveness at all. The assertion held no matter what
    /// [`live_at_definition_of`] did, so the regression it names would have
    /// sailed through. `zero` is defined in B0 and reaches B1 only along the
    /// phi edge; `limit` is defined in B1 and dies there. Neither is live where
    /// the other is defined, and answering that requires the liveness logic.
    #[test]
    fn an_unrelated_pair_does_not_interfere() {
        let ssa = testing::loop_counter_fixture();
        let graph = SsaCfg::from_ssa(&ssa);
        let analysis = LiveVariables::new(&ssa);
        let live = DataFlowSolver::new(LiveVariables::new(&ssa)).solve(&ssa, &graph);
        let defuse = DefUseIndex::build_with_ops(&ssa);

        // B0's only definition: the phi's entry operand.
        let zero = ssa
            .blocks()
            .first()
            .and_then(|block| block.instructions().iter().find_map(|i| i.op().dest()))
            .expect("B0 defines the loop's initial value");
        // B1's comparison bound - the last constant defined in the loop body.
        let limit = ssa
            .blocks()
            .get(1)
            .and_then(|block| {
                block
                    .instructions()
                    .iter()
                    .rev()
                    .find(|i| matches!(i.op(), SsaOp::Const { .. }))
                    .and_then(|i| i.op().dest())
            })
            .expect("the loop body defines its comparison bound");

        assert_ne!(zero, limit, "the two are distinct values");
        assert!(
            !Interference::new(
                &analysis,
                &live,
                &defuse,
                &Listing::of(&ssa, &defuse, &[]),
                None,
                &vec![0; ssa.variable_count()]
            )
            .interferes(zero, limit),
            "neither is live where the other is defined, so they may share a local"
        );
        assert!(
            !Interference::new(
                &analysis,
                &live,
                &defuse,
                &Listing::of(&ssa, &defuse, &[]),
                None,
                &vec![0; ssa.variable_count()]
            )
            .interferes(zero, zero),
            "a value never interferes with itself"
        );
    }

    /// A shared abstract local does not grant its other definitions physical storage.
    #[test]
    fn memory_web_mixed_storage_detaches_physical_members() {
        let ssa = testing::diamond_chain_fixture(1);
        let phi = ssa.blocks()[3].phi_nodes().first().unwrap();
        let result = phi.result().index();
        let left = phi.operands()[0].value().index();
        let right = phi.operands()[1].value().index();
        let defuse = DefUseIndex::build_with_ops(&ssa);
        let gathered = Gathering {
            rowless: &[],
            groups: vec![Group {
                values: vec![left, result],
                in_memory: true,
            }],
            conflicts: None,
            must_share: &[],
            anchored: &[],
        }
        .run(&ssa, &defuse, |_, _, _| true);
        assert_eq!(
            gathered.representative(left),
            gathered.representative(right)
        );
        assert_eq!(
            gathered.representative(left),
            gathered.representative(result)
        );
        assert!(gathered.is_detached(left));
        assert!(gathered.is_detached(result));
        assert!(!gathered.is_detached(right));
    }

    /// Values of one physical group keep its name when their entire web belongs to it.
    #[test]
    fn memory_web_one_storage_keeps_its_name() {
        let ssa = testing::diamond_chain_fixture(1);
        let phi = ssa.blocks()[3].phi_nodes().first().unwrap();
        let values: Vec<_> = phi
            .operands()
            .iter()
            .map(|operand| operand.value().index())
            .chain([phi.result().index()])
            .collect();
        let defuse = DefUseIndex::build_with_ops(&ssa);
        let gathered = Gathering {
            rowless: &[],
            groups: vec![Group {
                values: values.clone(),
                in_memory: true,
            }],
            conflicts: None,
            must_share: &[],
            anchored: &[],
        }
        .run(&ssa, &defuse, |_, _, _| true);
        for value in values {
            assert_eq!(
                gathered.representative(value),
                gathered.representative(phi.result().index())
            );
            assert!(!gathered.is_detached(value));
        }
    }

    /// Distinct or overlapping physical groups cannot name a shared abstract local.
    #[test]
    fn memory_web_distinct_or_overlapping_storage_detaches_each_name() {
        let ssa = testing::diamond_chain_fixture(1);
        let phi = ssa.blocks()[3].phi_nodes().first().unwrap();
        let result = phi.result().index();
        let left = phi.operands()[0].value().index();
        let right = phi.operands()[1].value().index();
        let defuse = DefUseIndex::build_with_ops(&ssa);
        for overlap in [false, true] {
            for reverse in [false, true] {
                let mut groups = vec![
                    Group {
                        values: if overlap {
                            vec![left, right, result]
                        } else {
                            vec![left, result]
                        },
                        in_memory: true,
                    },
                    Group {
                        values: vec![right],
                        in_memory: true,
                    },
                ];
                if reverse {
                    groups.reverse();
                }
                let gathered = Gathering {
                    rowless: &[],
                    groups,
                    conflicts: None,
                    must_share: &[],
                    anchored: &[],
                }
                .run(&ssa, &defuse, |_, _, _| true);
                for value in [left, right, result] {
                    assert!(
                        gathered.is_detached(value),
                        "overlap={overlap} reverse={reverse} value={value}"
                    );
                }
            }
        }
    }

    /// One stale physical member cannot acquire its name from another member of its web.
    #[test]
    fn memory_web_stale_member_detaches_the_whole_storage_name() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        let value = builder
            .in_block(0, |block| {
                let one = block.const_i32(SsaDefSpec::tmp(MockType::I32), 1)?;
                let value = block.add(SsaDefSpec::tmp(MockType::I32), one, one)?;
                block.call_void(crate::ir::Callee::Direct(1), vec![])?;
                block.jump(1)?;
                Ok(value)
            })
            .unwrap();
        let result = builder
            .in_block(1, |block| {
                let result = block.phi(SsaDefSpec::tmp(MockType::I32), [(0, value)])?;
                block.ret(Some(result))?;
                Ok(result)
            })
            .unwrap();
        let ssa = builder.finish().unwrap();
        let defuse = DefUseIndex::build_with_ops(&ssa);
        let gathered = Gathering {
            rowless: &[],
            groups: vec![Group {
                values: vec![value.index(), result.index()],
                in_memory: true,
            }],
            conflicts: None,
            must_share: &[],
            anchored: &[],
        }
        .run(&ssa, &defuse, |_, _, _| true);
        assert_eq!(
            gathered.representative(value.index()),
            gathered.representative(result.index())
        );
        assert!(gathered.is_detached(value.index()));
        assert!(gathered.is_detached(result.index()));
    }
}
