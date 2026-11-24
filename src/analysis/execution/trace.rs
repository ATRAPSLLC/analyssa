//! What a path remembers about how it got where it is.
//!
//! The [`State`](crate::analysis::execution::state::State) is what a path knows; its trace is how it
//! came to know it. Two things are kept:
//!
//! - **The lineage**: which undecided branches the path took. Two arrivals at
//!   one place with one lineage are one path coming round again - the next
//!   iteration of a loop the program decides - while two lineages are
//!   alternatives, which the explorer joins. The lineage is a hash chain; two
//!   alternatives whose lineages collide are explored apart without joining,
//!   which costs work and never soundness.
//! - **The loops the path is in**, each visit with the bytes written during it
//!   and whether it stayed *clean*: entered once, left once, with no undecided
//!   branch, no join and no call inside. A clean visit is one whole run of the
//!   loop the program alone decided.

use std::{
    cell::RefCell,
    collections::BTreeSet,
    hash::{DefaultHasher, Hash, Hasher},
    rc::Rc,
};

use crate::{
    analysis::execution::{
        LoopEffect,
        findings::{Decision, Decisions},
        memory::Memory,
        uses::{UseTrace, Uses},
        value::{Address, Value},
    },
    ir::variable::SsaVarId,
    pointer::PointerSize,
    target::Target,
};

/// Which undecided branches a path took.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Lineage(u64);

impl Lineage {
    /// The lineage of the path that entered the function.
    pub const ENTRY: Self = Self(0);

    /// The lineage of a path taking `successor` at an undecided branch ending
    /// `block`.
    #[must_use]
    pub fn fork(self, block: usize, successor: usize) -> Self {
        Self::chain(&(self.0, 0u8, block, successor))
    }

    /// Extends the lineage with one modelled call alternative.
    pub fn call(self, site: (usize, usize), alternative: usize) -> Self {
        Self::chain(&(self.0, 2u8, site, alternative))
    }

    /// The lineage of the state joined at `block`, arrived at from
    /// `previous`, for the `generation`th time.
    #[must_use]
    pub fn merged(block: usize, previous: Option<usize>, generation: u32) -> Self {
        Self::chain(&(1u8, block, previous, generation))
    }

    /// Hashes `link` into a lineage.
    fn chain(link: &impl Hash) -> Self {
        let mut hasher = DefaultHasher::new();
        link.hash(&mut hasher);
        Self(hasher.finish())
    }
}

/// One visit of a path to one loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopVisit {
    /// The loop's header block.
    pub header: usize,
    /// The activation the loop runs in.
    pub activation: u32,
    /// Every byte written since the visit began.
    pub written: BTreeSet<Address>,
    /// Whether the visit is still one run the program alone decided.
    pub clean: bool,
}

impl LoopVisit {
    /// The effect of this visit, which left by `exit` with `memory`
    /// and the loop's values at `exit_values`; `None` when the visit was not
    /// one run the program alone decided, or left a written byte unknown.
    pub fn effect<T: Target>(
        &self,
        exit: (usize, usize),
        memory: &Memory,
        exit_values: Vec<(SsaVarId, Value<T>)>,
    ) -> Option<LoopEffect<T>> {
        if !self.clean {
            return None;
        }
        let mut writes: Vec<(Address, Vec<u8>)> = Vec::new();
        for address in &self.written {
            let byte = memory.known_byte(*address)?;

            let follows = writes
                .last()
                .is_some_and(|(start, run)| address.follows(*start, run.len()));
            match writes.last_mut() {
                Some((_, run)) if follows => run.push(byte),
                _ => {
                    writes.push((*address, vec![byte]));
                }
            }
        }
        Some(LoopEffect {
            header: self.header,
            exit,
            writes,
            exit_values,
        })
    }
}

/// How one path got where it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trace {
    /// The undecided branches taken.
    lineage: Lineage,
    /// The loops the path is in, outermost first.
    loops: Vec<LoopVisit>,
    /// Invocation-local access coverage, separate from machine state and loop proofs.
    pub uses: Option<UseTrace>,
    /// The decisions taken, as a finding's witness reads them.
    decisions: Decisions,
}

impl Trace {
    /// The trace of the path entering the function.
    pub const fn new() -> Self {
        Self {
            lineage: Lineage::ENTRY,
            loops: Vec::new(),
            uses: None,
            decisions: Decisions::new(),
        }
    }

    /// Associates access coverage without changing the path's machine state.
    pub fn with_uses(mut self, uses: Rc<RefCell<Uses>>) -> Self {
        self.uses = Some(UseTrace::new(uses));
        self
    }

    /// The lineage.
    pub const fn lineage(&self) -> Lineage {
        self.lineage
    }

    /// The decisions taken.
    pub const fn decisions(&self) -> &Decisions {
        &self.decisions
    }

    /// Takes `successor` at the undecided branch ending `block` of `body`:
    /// forks the lineage, records the decision and marks every open loop
    /// visit unclean.
    pub fn fork(&mut self, body: usize, block: usize, successor: usize) {
        self.taint();
        self.lineage = self.lineage.fork(block, successor);
        self.decisions = self.decisions.then(Decision::Branch {
            body,
            block,
            successor,
        });
    }

    /// Takes `alternative` of the modelled call at `site` of `body`, and
    /// marks every open loop visit unclean.
    pub fn fork_call(&mut self, body: usize, site: (usize, usize), alternative: usize) {
        self.taint();
        self.lineage = self.lineage.call(site, alternative);
        self.decisions = self.decisions.then(Decision::Alternative {
            body,
            site,
            alternative,
        });
    }

    /// Enters `handler` for a raise at `site` of `body`: forks the lineage,
    /// records the decision and marks every open loop visit unclean, since a
    /// raise leaves the loops it is in by no edge of theirs.
    pub fn raise(&mut self, body: usize, site: (usize, usize), handler: usize) {
        self.taint();
        self.lineage = self.lineage.fork(site.0, handler);
        self.decisions = self.decisions.then(Decision::Raise {
            body,
            site,
            handler,
        });
    }

    /// Marks every open loop visit unclean.
    pub fn taint(&mut self) {
        for visit in &mut self.loops {
            visit.clean = false;
        }
    }

    /// Begins a visit to `header` in `activation`.
    pub fn enter_loop(&mut self, header: usize, activation: u32) {
        self.loops.push(LoopVisit {
            header,
            activation,
            written: BTreeSet::new(),
            clean: true,
        });
    }

    /// Ends the innermost visit if it leaves, without allocating a drain buffer.
    ///
    /// A visit stays open while one nested inside it does.
    pub fn pop_loop(&mut self, leaves: impl FnOnce(&LoopVisit) -> bool) -> Option<LoopVisit> {
        let visit = self.loops.last()?;

        if leaves(visit) {
            self.loops.pop()
        } else {
            None
        }
    }

    /// Collects closed visits for isolated semantic fixtures.
    #[cfg(test)]
    pub fn leave_loops(&mut self, leaves: impl Fn(&LoopVisit) -> bool) -> Vec<LoopVisit> {
        let mut visits = Vec::new();
        while let Some(visit) = self.pop_loop(&leaves) {
            visits.push(visit);
        }
        visits
    }

    /// Returns the clean loop visits that may retain a read witness.
    pub fn clean_headers(&self) -> impl Iterator<Item = usize> + '_ {
        self.loops
            .iter()
            .filter(|visit| visit.clean)
            .map(|visit| visit.header)
    }

    /// The loops the path is in, outermost first.
    #[cfg(test)]
    pub fn loops(&self) -> &[LoopVisit] {
        &self.loops
    }

    /// Records that `len` bytes were written at `address`, in every clean open visit.
    pub fn record_write(&mut self, address: Address, len: usize, ptr_size: PointerSize) {
        if let Some(uses) = &mut self.uses {
            uses.write(address, len);
        }
        if len == 0 {
            return;
        }
        for visit in self.loops.iter_mut().filter(|visit| visit.clean) {
            for index in 0..len {
                if let Ok(delta) = i64::try_from(index)
                    && let Some(at) = address.moved(delta, ptr_size)
                {
                    visit.written.insert(at);
                } else {
                    visit.clean = false;
                }
            }
        }
    }

    /// The trace of the state joined at `block` from `previous`: a new
    /// lineage, and the loops both paths are in, none of them clean.
    ///
    /// # Arguments
    ///
    /// * `other` - The trace of the other path.
    /// * `block` - Where the paths meet.
    /// * `previous` - The block both arrived from.
    /// * `generation` - How many joins have been made there before.
    pub fn join(
        &self,
        other: &Self,
        block: usize,
        previous: Option<usize>,
        generation: u32,
    ) -> Self {
        let mut loops = Vec::new();
        for (mine, theirs) in self.loops.iter().zip(&other.loops) {
            if mine.header != theirs.header || mine.activation != theirs.activation {
                break;
            }

            let written = mine.written.union(&theirs.written).copied().collect();
            loops.push(LoopVisit {
                header: mine.header,
                activation: mine.activation,
                written,
                clean: false,
            });
        }
        Self {
            lineage: Lineage::merged(block, previous, generation),
            decisions: self.decisions.joined(),
            loops,
            uses: match (&self.uses, &other.uses) {
                (Some(left), Some(right)) => left.join(right),
                (Some(uses), None) | (None, Some(uses)) => {
                    uses.unresolved();
                    None
                }
                (None, None) => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::analysis::execution::value::Root;

    /// Lineages part at every different choice and are equal for equal ones.
    #[test]
    fn lineages_part_at_each_choice() {
        let taken = Lineage::ENTRY.fork(3, 4);
        assert_eq!(taken, Lineage::ENTRY.fork(3, 4));
        assert_ne!(taken, Lineage::ENTRY.fork(3, 5));
        assert_ne!(taken.fork(3, 4), taken);
        assert_ne!(
            Lineage::merged(3, Some(2), 0),
            Lineage::merged(3, Some(2), 1)
        );
    }

    /// A fork inside a loop spoils the visit; writes land in every open visit.
    #[test]
    fn a_fork_spoils_the_visits_it_is_in() {
        let mut trace = Trace::new();
        trace.enter_loop(1, 0);
        trace.enter_loop(2, 0);
        let at = Address::new(Root::Stack, -4, PointerSize::Bit64);
        trace.record_write(at, 2, PointerSize::Bit64);
        assert!(
            trace
                .loops()
                .iter()
                .all(|visit| visit.clean && visit.written.len() == 2)
        );
        trace.fork(0, 2, 3);
        assert!(trace.loops().iter().all(|visit| !visit.clean));
    }

    /// Leaving takes the innermost visits first and stops at one still open.
    #[test]
    fn leaving_is_innermost_first() {
        let mut trace = Trace::new();
        trace.enter_loop(1, 0);
        trace.enter_loop(2, 0);
        let left = trace.leave_loops(|visit| visit.header == 2);
        assert_eq!(left.len(), 1);
        assert_eq!(trace.loops().len(), 1);
        assert!(trace.leave_loops(|visit| visit.header == 9).is_empty());
    }

    /// A joined trace is a new lineage, keeps the loops both are in, is
    /// clean nowhere, and keeps its own decisions as a joined witness.
    #[test]
    fn a_join_is_new_and_unclean() {
        let mut left = Trace::new();
        left.enter_loop(1, 0);
        left.fork(0, 3, 4);
        let mut right = left.clone();
        right.enter_loop(2, 0);
        let joined = left.join(&right, 5, Some(4), 0);
        assert_ne!(joined.lineage(), left.lineage());
        assert_eq!(joined.loops().len(), 1);
        assert!(!joined.loops().iter().any(|visit| visit.clean));
        let witness = joined.decisions().witness();
        assert!(witness.joined);
        assert_eq!(witness.decisions, left.decisions().witness().decisions);
        assert!(!left.decisions().witness().joined);
    }

    /// Builds a trace containing the requested number of nested loop visits.
    fn nested(count: usize) -> Trace {
        let mut trace = Trace::new();
        for header in 0..count {
            trace.enter_loop(header, 0);
        }
        trace
    }

    /// Joins preserve common visits and sorted unique writes without changing either source.
    #[test]
    fn trace_join_and_copy_preserve_sources_and_exact_visit_contents() {
        let mut left = nested(3);
        let mut right = nested(3);
        right.loops.last_mut().unwrap().header = 9;
        for offset in [-4, -2] {
            left.record_write(
                Address::new(Root::Stack, offset, PointerSize::Bit64),
                1,
                PointerSize::Bit64,
            );
        }
        for offset in [-3, -2] {
            right.record_write(
                Address::new(Root::Stack, offset, PointerSize::Bit64),
                1,
                PointerSize::Bit64,
            );
        }
        let before = left.clone();
        let other_before = right.clone();

        assert_eq!(left, before);
        assert_eq!(right, other_before);
        let joined = left.join(&right, 4, Some(3), 1);
        assert_eq!(joined.loops.len(), 2);
        for visit in joined.loops {
            assert!(!visit.clean);
            assert_eq!(
                visit
                    .written
                    .into_iter()
                    .map(Address::offset)
                    .collect::<Vec<_>>(),
                [-4, -3, -2]
            );
        }
        assert_eq!(left, before);
    }
}
