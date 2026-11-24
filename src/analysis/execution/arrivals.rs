//! Paths, and what happens when one arrives at a block.
//!
//! Every arrival is recorded by where it stands: each activation's function,
//! number and position, the edge it came in by included. A later arrival at
//! the same place either continues the recorded path - it has the same
//! lineage, so it is that path coming round again - or is an alternative to
//! it, and is joined into it. Two paths that disagree on what the
//! conditions they took establish of a value still read there stand apart,
//! and are not joined.

use std::{
    collections::{BTreeMap, btree_map::Entry},
    hash::{Hash, Hasher},
    mem,
};

use rustc_hash::FxHasher;

use crate::{
    analysis::execution::{
        frame::{CallScope, Position},
        refine::Facts,
        returning::ReturnBinding,
        state::State,
        trace::{Lineage, Trace},
    },
    ir::variable::SsaVarId,
    target::Target,
};

/// Whether a value of a function, numbered as the explorer numbers bodies,
/// may be read at or after the entry to one of its blocks.
pub type Live<'a> = &'a dyn Fn(usize, usize, SsaVarId) -> bool;

/// Where one activation stands.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Standing {
    /// Its function, as the explorer numbers bodies.
    body: usize,
    /// Its number.
    activation: u32,
    /// Where it is in its function.
    position: Position,
    /// The call observation it continues, if any.
    observation: Option<CallScope>,
    /// How its result maps to the continuation it returns to.
    return_binding: ReturnBinding,
    /// What the conditions its path took establish of its values.
    refinements: Vec<(SsaVarId, Facts)>,
}

/// An exact position key with a fixed-width lookup fingerprint.
#[derive(Debug, Clone)]
struct Stand {
    /// The exact semantic position, never replaced by its fingerprint.
    frames: Vec<Standing>,
    /// A lookup accelerator; collisions require checked frame comparisons.
    fingerprint: u64,
}

impl Stand {
    /// Compares exact frame positions after fingerprint lookup.
    fn matches(&self, other: &Self) -> bool {
        self.frames == other.frames
    }

    /// Returns the innermost position entry.
    fn last(&self) -> Option<&Standing> {
        self.frames.last()
    }
}

/// A retained position and its complete execution state.
type StandEntry<T> = Box<(Stand, Arrival<T>)>;

/// Collision storage with a cheap singleton and bounded tree-node insertion.
///
/// Growing a bucket moves fixed-size handles inside tree nodes; it never
/// reallocates a vector containing all previously retained execution records.
#[derive(Debug)]
enum Bucket<T: Target> {
    /// A fingerprint naming exactly one position.
    One(StandEntry<T>),
    /// Distinct positions sharing a fingerprint, ordered by insertion ordinal.
    Many(BTreeMap<usize, StandEntry<T>>),
}

impl<T: Target> Bucket<T> {
    /// Inserts one admitted record without moving any prior execution storage.
    fn insert(&mut self, entry: StandEntry<T>) {
        let prior = mem::replace(self, Self::Many(BTreeMap::new()));
        let mut entries = match prior {
            Self::One(first) => BTreeMap::from([(0, first)]),
            Self::Many(entries) => entries,
        };
        entries.insert(entries.len(), entry);
        *self = Self::Many(entries);
    }

    /// Visits exact keys without hiding scans behind a filtered iterator.
    fn entries_mut(&mut self) -> impl Iterator<Item = &mut (Stand, Arrival<T>)> {
        let (one, many) = match self {
            Self::One(entry) => (Some(entry.as_mut()), None),
            Self::Many(entries) => (None, Some(entries)),
        };
        one.into_iter().chain(
            many.into_iter()
                .flat_map(|entries| entries.values_mut().map(|entry| entry.as_mut())),
        )
    }

    /// Visits fixture keys through the same stable record storage.
    #[cfg(test)]
    fn entries(&self) -> impl Iterator<Item = &(Stand, Arrival<T>)> {
        let (one, many) = match self {
            Self::One(entry) => (Some(entry.as_ref()), None),
            Self::Many(entries) => (None, Some(entries)),
        };
        one.into_iter().chain(
            many.into_iter()
                .flat_map(|entries| entries.values().map(|entry| entry.as_ref())),
        )
    }

    /// Returns the fixture record count.
    #[cfg(test)]
    fn len(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Many(entries) => entries.len(),
        }
    }
}

/// Retained arrivals behind fixed-width lookup keys and checked collision buckets.
///
/// Ordered fingerprint lookup avoids an input-controlled vector hash or equality
/// scan inside the container. Fixed-width keys bound each tree comparison;
/// exact comparisons resolve every fingerprint collision.
#[derive(Debug)]
struct StandMap<T: Target> {
    /// Each bucket retains exact position keys, including colliding fingerprints.
    buckets: BTreeMap<u64, Bucket<T>>,
}

impl<T: Target> Default for StandMap<T> {
    fn default() -> Self {
        Self {
            buckets: BTreeMap::new(),
        }
    }
}

impl<T: Target> StandMap<T> {
    /// Clears all retained records.
    fn clear(&mut self) {
        self.buckets.clear();
    }

    /// Finds the exact position, resolving fingerprint collisions by frame identity.
    fn get_mut(&mut self, key: &Stand) -> Option<&mut Arrival<T>> {
        let bucket = self.buckets.get_mut(&key.fingerprint)?;
        for (stand, record) in bucket.entries_mut() {
            if stand.matches(key) {
                return Some(record);
            }
        }
        None
    }

    /// Installs a record without moving prior execution states.
    fn insert(&mut self, key: Stand, record: Arrival<T>) {
        let fingerprint = key.fingerprint;
        let entry = Box::new((key, record));
        match self.buckets.entry(fingerprint) {
            Entry::Vacant(bucket) => {
                bucket.insert(Bucket::One(entry));
            }
            Entry::Occupied(mut bucket) => bucket.get_mut().insert(entry),
        }
    }

    /// Finds a fixture record through the same exact position comparison.
    #[cfg(test)]
    fn get(&self, key: &Stand) -> Option<&Arrival<T>> {
        self.buckets
            .get(&key.fingerprint)?
            .entries()
            .find_map(|(stand, record)| (stand.matches(key)).then_some(record))
    }

    /// Counts retained fixture records, including fingerprint collisions.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.buckets.values().map(Bucket::len).sum()
    }

    /// Returns whether no fixture records remain.
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }
}

/// One path: what it knows and how it got there.
#[derive(Debug, Clone)]
pub struct Path<T: Target> {
    /// What it knows.
    pub state: State<T>,
    /// How it got there.
    pub trace: Trace,
}

impl<T: Target> Path<T> {
    /// Where the path stands: every activation's function, number and
    /// position, outermost first, with what the conditions it took establish
    /// of each value `live` says is read at or after that position.
    fn stand(&self, live: Live<'_>) -> Stand {
        let mut stand = Vec::with_capacity(self.state.depth());
        let mut hash = FxHasher::default();
        for frame in self.state.frames() {
            let entry = Standing {
                body: frame.body(),
                activation: frame.activation(),
                position: frame.position(),
                observation: frame.observation(),
                return_binding: frame.return_binding(),
                refinements: frame
                    .refinements()
                    .into_iter()
                    .filter(|(value, _)| live(frame.body(), frame.position().block, *value))
                    .collect(),
            };
            entry.hash(&mut hash);
            stand.push(entry);
        }
        Stand {
            frames: stand,
            fingerprint: hash.finish(),
        }
    }

    /// Identifies the original entry activation, even when a tail callee occupies its depth.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.state
            .frame()
            .is_some_and(|frame| frame.activation() == 0 && frame.body() == 0)
    }
}

/// The path last recorded at one place.
#[derive(Debug)]
struct Arrival<T: Target> {
    /// Its lineage.
    lineage: Lineage,
    /// What it knew.
    state: State<T>,
    /// How it got there.
    trace: Trace,
    /// How many joins have been made here.
    generation: u32,
}

/// Every place a path has arrived at, with the path last recorded there.
#[derive(Debug)]
pub struct Arrivals<T: Target> {
    /// The records, by where they stand.
    records: StandMap<T>,
    /// State snapshots created over this execution.
    snapshots: usize,
}

impl<T: Target> Default for Arrivals<T> {
    fn default() -> Self {
        Self {
            records: StandMap::default(),
            snapshots: 0,
        }
    }
}

impl<T: Target> Arrivals<T> {
    /// Drops historical arrivals when no independent execution path remains.
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Returns the total number of state snapshots created for arrivals.
    pub const fn snapshots(&self) -> usize {
        self.snapshots
    }

    /// Admits `path`, which has just arrived where it stands: returns the path
    /// to run on from here, or `None` when what it could do has already been
    /// explored.
    pub fn admit(&mut self, path: Path<T>, live: Live<'_>) -> Option<Path<T>> {
        let stand = path.stand(live);
        let Some(record) = self.records.get_mut(&stand) else {
            let copied = path.clone();
            self.records.insert(
                stand,
                Arrival {
                    lineage: copied.trace.lineage(),
                    state: copied.state,
                    trace: copied.trace,
                    generation: 0,
                },
            );
            self.snapshots = self.snapshots.saturating_add(1);
            return Some(path);
        };
        if record.lineage == path.trace.lineage() {
            let copied = path.clone();
            record.state = copied.state;
            record.trace = copied.trace;
            self.snapshots = self.snapshots.saturating_add(1);
            return Some(path);
        }

        let Some(joined) = record.state.join(&path.state) else {
            return Some(path);
        };
        if joined == record.state {
            return None;
        }
        let Some(position) = stand.last().map(|standing| standing.position) else {
            return Some(path);
        };
        let generation = record.generation.saturating_add(1);
        let trace = record
            .trace
            .join(&path.trace, position.block, position.previous, generation);

        let copied_state = joined.clone();
        let copied_trace = trace.clone();
        record.state = copied_state;
        record.lineage = copied_trace.lineage();
        record.trace = copied_trace;
        record.generation = generation;
        self.snapshots = self.snapshots.saturating_add(1);
        Some(Path {
            state: joined,
            trace,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;
    use crate::{
        analysis::execution::{frame::Frame, value::Value},
        ir::{function::SsaFunction, value::ConstValue, variable::SsaVarId},
        pointer::PointerSize,
        testing::MockTarget,
    };

    /// Every value is read everywhere.
    fn every(_: usize, _: usize, _: SsaVarId) -> bool {
        true
    }

    /// A path at block 1, from block 0, holding `held` in value 0.
    fn path(held: i32, trace: Trace) -> Path<MockTarget> {
        let ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut frame = Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64);
        frame.move_to(Position {
            block: 1,
            previous: Some(0),
            next: 0,
        });
        let mut state = State::new(frame);
        state.define(SsaVarId::from_index(0), Value::Const(ConstValue::I32(held)));
        Path { state, trace }
    }

    fn forked(successor: usize) -> Trace {
        let mut trace = Trace::new();
        trace.fork(0, 0, successor);
        trace
    }

    /// The same lineage coming round again runs on, whatever it now holds.
    #[test]
    fn an_iteration_runs_on() {
        let mut arrivals = Arrivals::default();
        assert!(arrivals.admit(path(0, Trace::new()), &every).is_some());
        assert!(arrivals.admit(path(1, Trace::new()), &every).is_some());
        assert!(arrivals.admit(path(2, Trace::new()), &every).is_some());
    }

    /// An alternative is joined in; one the record already covers stops.
    #[test]
    fn alternatives_join_and_stop_once_covered() {
        let mut arrivals = Arrivals::default();
        arrivals.admit(path(0, forked(1)), &every);
        let joined = arrivals
            .admit(path(1, forked(2)), &every)
            .unwrap_or_else(|| process::abort());
        assert_eq!(joined.state.value(SsaVarId::from_index(0)), Value::Unknown);
        assert!(arrivals.admit(path(5, forked(3)), &every).is_none());
    }

    /// An alternative that agrees with the record adds nothing.
    #[test]
    fn an_agreeing_alternative_stops() {
        let mut arrivals = Arrivals::default();
        arrivals.admit(path(4, forked(1)), &every);
        assert!(arrivals.admit(path(4, forked(2)), &every).is_none());
    }

    /// Distinct selected call continuations keep separate arrivals after a tail transfer.
    #[test]
    fn selected_call_continuations_are_distinct_arrival_positions() {
        let scopes = [
            None,
            Some(CallScope {
                body: Some(1),
                returns: None,
                returns_binding: ReturnBinding::Identity,
                returns_since: 0,
            }),
            Some(CallScope {
                body: Some(1),
                returns: Some(2),
                returns_binding: ReturnBinding::Identity,
                returns_since: 0,
            }),
            Some(CallScope {
                body: Some(1),
                returns: Some(3),
                returns_binding: ReturnBinding::Identity,
                returns_since: 0,
            }),
            Some(CallScope {
                body: None,
                returns: Some(2),
                returns_binding: ReturnBinding::Identity,
                returns_since: 0,
            }),
        ];
        let mut arrivals = Arrivals::default();
        let baseline = path(4, forked(1));
        let mut states = Vec::new();
        for (index, scope) in scopes.into_iter().enumerate() {
            let mut candidate = path(4, forked(index.saturating_add(1)));
            let frame = baseline
                .state
                .frame()
                .unwrap()
                .clone()
                .with_observation(scope);
            candidate.state = State::new(frame);
            for state in &states {
                assert_ne!(&candidate.state, state);
                assert!(candidate.state.join(state).is_none());
            }
            states.push(candidate.state.clone());
            assert!(arrivals.admit(candidate, &every).is_some());
        }
        assert_eq!(arrivals.records.len(), scopes.len());
    }

    /// Fingerprint collisions retain distinct positions and lookup remains interruptible.
    #[test]
    fn arrival_fingerprints_never_replace_exact_position_identity() {
        let first_path = path(0, Trace::new());
        let mut second_path = path(1, Trace::new());
        second_path.state.frame_mut().unwrap().move_to(Position {
            block: 2,
            previous: Some(0),
            next: 0,
        });
        let first = first_path.stand(&every);
        let mut second = second_path.stand(&every);
        second.fingerprint = first.fingerprint;
        let mut records = StandMap::default();
        records.insert(
            first.clone(),
            Arrival {
                lineage: Lineage::ENTRY,
                state: first_path.state,
                trace: first_path.trace,
                generation: 0,
            },
        );
        records.insert(
            second.clone(),
            Arrival {
                lineage: Lineage::ENTRY,
                state: second_path.state,
                trace: second_path.trace,
                generation: 0,
            },
        );
        assert_eq!(records.len(), 2);

        assert_eq!(
            records
                .get(&first)
                .unwrap()
                .state
                .value(SsaVarId::from_index(0))
                .integer(),
            Some(0)
        );
        assert_eq!(
            records
                .get(&second)
                .unwrap()
                .state
                .value(SsaVarId::from_index(0))
                .integer(),
            Some(1)
        );

        let record = records.get_mut(&second).unwrap();
        assert_eq!(
            record.state.value(SsaVarId::from_index(0)).integer(),
            Some(1)
        );
    }

    /// Exact-key comparison checks long frame prefixes instead of a vector equality.
    #[test]
    fn stand_comparisons_check_each_frame_before_accepting_an_arrival() {
        let ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut path = path(0, Trace::new());
        for activation in 1..1024 {
            path.state.call(Frame::enter(
                &ssa,
                0,
                activation,
                0,
                vec![],
                PointerSize::Bit64,
            ));
        }
        let key = path.stand(&every);
        let other = path.stand(&every);

        assert!(key.matches(&other));
    }

    /// Promoting and growing one colliding bucket preserves exact keys and checked lookup.
    #[test]
    fn crowded_arrival_buckets_preserve_all_records_and_bound_lookup() {
        let mut records = StandMap::default();
        let mut keys = Vec::new();
        for index in 0..1024 {
            let mut path = path(index as i32, Trace::new());
            path.state.frame_mut().unwrap().move_to(Position {
                block: index + 1,
                previous: Some(0),
                next: 0,
            });
            let mut stand = path.stand(&every);
            stand.fingerprint = 0;
            keys.push(stand.clone());
            records.insert(
                stand,
                Arrival {
                    lineage: Lineage::ENTRY,
                    state: path.state,
                    trace: path.trace,
                    generation: 0,
                },
            );
        }
        assert_eq!(records.len(), 1024);
        for index in [0, 1, 511, 1023] {
            assert_eq!(
                records
                    .get(&keys[index])
                    .unwrap()
                    .state
                    .value(SsaVarId::from_index(0))
                    .integer(),
                Some(index as i64)
            );
        }

        assert_eq!(records.len(), 1024);

        let last = records.get_mut(&keys[1023]).unwrap();
        assert_eq!(
            last.state.value(SsaVarId::from_index(0)).integer(),
            Some(1023)
        );

        records.clear();
        assert!(records.is_empty());
    }

    /// Live and retired-segment bindings participate in exact arrival identity.
    #[test]
    fn return_binding_contracts_keep_separate_arrival_records() {
        let scope = CallScope {
            body: None,
            returns: Some(2),
            returns_binding: ReturnBinding::Word(32),
            returns_since: 0,
        };
        let contracts = [
            (ReturnBinding::Word(32), None),
            (ReturnBinding::Unknown, None),
            (ReturnBinding::Unknown, Some(scope)),
            (
                ReturnBinding::Unknown,
                Some(CallScope {
                    returns_binding: ReturnBinding::Unknown,
                    ..scope
                }),
            ),
        ];
        let baseline = path(4, forked(1));
        let mut arrivals = Arrivals::default();
        let mut retained: Vec<Path<MockTarget>> = Vec::new();
        for (index, (binding, observation)) in contracts.into_iter().enumerate() {
            let mut candidate = path(4, forked(index.saturating_add(1)));
            let frame = baseline
                .state
                .frame()
                .unwrap_or_else(|| process::abort())
                .clone()
                .with_return_binding(binding)
                .with_observation(observation);
            candidate.state = State::new(frame);
            for previous in &retained {
                assert_ne!(
                    candidate.stand(&every).frames,
                    previous.stand(&every).frames
                );
            }
            retained.push(candidate.clone());
            assert!(arrivals.admit(candidate, &every).is_some());
        }
        assert_eq!(arrivals.records.len(), contracts.len());
        // Identical bindings still use ordinary covered-state admission. This
        // rejects a broken fix that simply disables joins for every new field.
        for (index, mut candidate) in retained.into_iter().enumerate() {
            candidate.trace = forked(index.saturating_add(10));
            assert!(arrivals.admit(candidate, &every).is_none());
        }
        assert_eq!(arrivals.records.len(), contracts.len());
    }
}
