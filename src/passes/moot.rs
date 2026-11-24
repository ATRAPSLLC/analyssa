//! Dropping a test whose outcome changes nothing.
//!
//! A compiler that knows where a loop's work goes may copy the loop for each
//! way a test of its state could go: GCC's `-O2` reads an AAPCS64 `va_list`
//! in a register loop, a stack loop and a read between them, picked by tests
//! of the list's count. Once each read is the one `va_arg` it is, every path
//! does the same per trip, and the tests choose between copies of one loop.
//!
//! [`MootTestPass`] finds a test whose two arms do the same thing from where they
//! start: run side by side, symbolically, they make the same effects in the
//! same order on the same values and leave by the same returns, whichever way
//! the test went. Each side is run as values, not rows:
//!
//! - **A value is what it computes**: a pure operation is its operation of
//!   the values it reads, so two sides computing alike in another order are
//!   alike; a load is its address read in the memory the effects so far have
//!   left.
//! - **Effects go in lockstep**: anything neither pure nor a load (a call, a
//!   store, a `va_arg`, an operation that may trap) is matched with the other
//!   side's next one, the same operation on the same values, and what each
//!   defines is one value from there on.
//! - **A branch pairs with a branch on the same question**, its targets with
//!   theirs, or with its negation, its targets crossed. A branch that pairs
//!   with nothing decides nothing where each of its arms goes on as the other
//!   side does.
//! - **Both sides leave alike**: by the same return of the same values.
//! - **Two sides entering a pair of joins** that take alike values on both
//!   are taken to go on alike from there: the proof is coinductive, a loop's
//!   next trip is what this one proves. A join keeps only what its phis take;
//!   anything else read past it must be read again. A side that has only
//!   computed values since its block began still stands at its start: where
//!   the other side enters a join, or branches into one, the two may be a
//!   pair, as when one copy of a loop tests the walk's state again before
//!   the trip the other begins with its read.
//! - **A value is read alike on both sides** where it is computed alike on
//!   both, or defined where the test's way in passed and no path from the
//!   test reaches again.
//!
//! The proof is incomplete, never unsound: two sides that go round together
//! through blocks with no phi meet at no join, and are not taken for alike.
//!
//! The test then jumps to an arm that goes on out of the innermost loop the
//! test is in, and of two that do, or two that do not, to the one with more
//! blocks ahead of it, the other being the copy, so the most of the
//! program's own code is kept, and what only the other reached goes. An arm
//! whose blocks, short of coming back through the test, all lie in that
//! loop goes round it: kept, with the test a jump into it, the loop would
//! never end, whether the function returns or, as a firmware's main loop,
//! never does. An arm that leaves the loop another way (a copy of the loop
//! with an exit of its own, chosen between) still goes on. A FIFO drain,
//! `for (i = 0; i < n; i += 4) (void)*fifo;`, reads a word a trip and never
//! uses it, so a trip does nothing the way out does not and its test is
//! moot: the test jumps out, never round. Each effect, branch and return of
//! the copy that goes is one the kept arm makes in its place, and the kept
//! one takes in its facts
//! ([`absorb_twin`](crate::facts::InstructionFacts::absorb_twin)):
//! a host's account of where the work came from, an `int 2Eh` and a `syscall`
//! entering one service each a call site, loses neither.
//!
//! A test one of whose arms only computes values on its way into the other is
//! left to
//! [control-flow simplification](crate::passes::controlflow), which keeps the
//! code the arm ran over. A function with exception handlers is left as it
//! is: a throw's way out is no branch the sides are run through.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    hash::{Hash, Hasher},
    mem,
    rc::Rc,
};

use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;

use crate::{
    Result,
    analysis::{budget::Budget, cfg::SsaCfg, loops::detect_loops},
    bitset::BitSet,
    events::EventKind,
    graph::{
        NodeId, Predecessors, Successors,
        algorithms::{DominatorTree, compute_dominators},
    },
    interrupt::{Interrupt, NeverInterrupted},
    ir::{
        function::{
            Again, Fingerprint, Refutation, SsaEditOptions, SsaFunction, SsaRollbackPolicy,
            Unsettled,
        },
        ops::{OperandRole, SsaOp, kinds::CmpKind},
        variable::SsaVarId,
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// A value as both sides compute it, by its index among the terms met.
type Term = usize;

/// The rows two arms made alike, `(block, row)` on the arm kept beside the
/// one on the arm that goes.
type Pairs = Vec<((usize, usize), (usize, usize))>;

/// Where a value a search reads is defined, as its refutation records it
/// ([`content`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Site {
    /// In a block the search read: the block, the row or `None` for a phi,
    /// and which of the row's definitions or which of the block's phis.
    Read(usize, Option<usize>, usize),
    /// In another block, or at the function's entry: the block, and the
    /// order the search's blocks first name the value in.
    Outside(usize, usize),
    /// A value the function does not have.
    Unknown,
}

/// What a value is, in terms both sides share.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Value<T: Target> {
    /// A value defined where the test's way in passed, and no path from the
    /// test reaches again: one value on both sides.
    Before(SsaVarId),
    /// A value the test's block defines as the test is reached, before its
    /// branch: one value on both sides.
    Entered(SsaVarId),
    /// A value a pair of joins takes alike on both sides, by its slot.
    Joined(usize),
    /// What an effect defined: the effect's number in order, and which of
    /// its definitions.
    Effect(usize, usize),
    /// A pure operation: the operation with its operands as terms and its
    /// definitions blank, and which of its definitions.
    Pure(SsaOp<T>, usize),
    /// A load, as a pure operation, in the memory a numbered state holds.
    Load(SsaOp<T>, usize),
}

/// The question a branch asks.
#[derive(Debug, Clone, PartialEq)]
enum Asked {
    /// A comparison of two values.
    Compare {
        /// The left value.
        left: Term,
        /// The right value.
        right: Term,
        /// The relation.
        cmp: CmpKind,
        /// Whether the values compare unsigned.
        unsigned: bool,
        /// Whether a value may be a float, whose ordered relations do not
        /// negate.
        float: bool,
    },
    /// A boolean value.
    Holds(Term),
}

impl Asked {
    /// Whether `other` holds exactly where this does not.
    fn negates(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Compare {
                    left,
                    right,
                    cmp,
                    unsigned,
                    float,
                },
                Self::Compare {
                    left: other_left,
                    right: other_right,
                    cmp: other_cmp,
                    unsigned: other_unsigned,
                    float: other_float,
                },
            ) => {
                let ordered = !matches!(cmp, CmpKind::Eq | CmpKind::Ne);
                left == other_left
                    && right == other_right
                    && unsigned == other_unsigned
                    && cmp.negated() == *other_cmp
                    && !(ordered && (*float || *other_float))
            }
            _ => false,
        }
    }
}

/// What one side does next.
#[derive(Debug, Clone)]
enum Step<T: Target> {
    /// An effect: the operation with its operands as terms, and what it
    /// defines.
    Effect(SsaOp<T>, Vec<SsaVarId>),
    /// A jump.
    Jump(usize),
    /// A branch: its question, and its targets where it holds and not.
    Branch(Asked, usize, usize),
    /// A way out of the function, with its operands as terms.
    Leave(SsaOp<T>),
}

/// One side of the run: where it is, the term of each value it has met, and
/// what each change to those replaced.
///
/// Where only the other side goes on from a step both took, the side is put
/// back where it stood: it is marked before the step ([`Self::mark`]) and
/// rolled back ([`Self::rollback`]), at the cost of what the step changed
/// rather than of a copy of everything the side holds.
#[derive(Debug, Clone, Default)]
struct Side {
    /// The block it is in.
    block: usize,
    /// The position of the next row to run.
    at: usize,
    /// The term of each value met.
    values: BTreeMap<SsaVarId, Term>,
    /// The values a join let go of: defined again on the way to it, they are
    /// no longer what they were as the test was reached.
    forgotten: BTreeSet<SsaVarId>,
    /// What each change since the side was made replaced, latest last.
    trail: Vec<Change>,
}

/// What one change to a [`Side`] replaced.
#[derive(Debug, Clone)]
enum Change {
    /// A value met, with the term it held before, if any.
    Met(SsaVarId, Option<Term>),
    /// The values and the let-go values a join replaced.
    Joined(BTreeMap<SsaVarId, Term>, BTreeSet<SsaVarId>),
}

/// Where a [`Side`] stood: its block, its row, and how many changes it had
/// made ([`Side::rollback`]).
#[derive(Debug, Clone, Copy)]
struct Mark {
    /// The block it was in.
    block: usize,
    /// The row it stood at.
    at: usize,
    /// How long its trail was.
    trail: usize,
}

impl Side {
    /// A side in `block` before its first row, holding nothing.
    fn new(block: usize) -> Self {
        Self {
            block,
            at: 0,
            values: BTreeMap::new(),
            forgotten: BTreeSet::new(),
            trail: Vec::new(),
        }
    }

    /// Where the side stands now.
    fn mark(&self) -> Mark {
        Mark {
            block: self.block,
            at: self.at,
            trail: self.trail.len(),
        }
    }

    /// Takes `term` as what `value` is.
    fn meet(&mut self, value: SsaVarId, term: Term) {
        let before = self.values.insert(value, term);
        self.trail.push(Change::Met(value, before));
    }

    /// Takes `values` and `forgotten` as everything the side holds and has let
    /// go of, as a join leaves it.
    fn generalize(&mut self, values: BTreeMap<SsaVarId, Term>, forgotten: BTreeSet<SsaVarId>) {
        let values = mem::replace(&mut self.values, values);
        let forgotten = mem::replace(&mut self.forgotten, forgotten);
        self.trail.push(Change::Joined(values, forgotten));
    }

    /// Puts back every change made since `mark`, so the side stands where
    /// and as it stood then.
    fn rollback(&mut self, mark: Mark) {
        while self.trail.len() > mark.trail {
            match self.trail.pop() {
                Some(Change::Met(value, Some(term))) => {
                    self.values.insert(value, term);
                }
                Some(Change::Met(value, None)) => {
                    self.values.remove(&value);
                }
                Some(Change::Joined(values, forgotten)) => {
                    self.values = values;
                    self.forgotten = forgotten;
                }
                None => break,
            }
        }
        self.block = mark.block;
        self.at = mark.at;
    }
}

/// What a phi of a join takes, as the join is assumed alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Slot {
    /// A value both sides share, by its place among them.
    Shared(usize),
    /// A value fixed however many trips are made: one the test's way in
    /// defined.
    Fixed(Term),
}

/// A state of the game a test's arms are played in ([`Moot::play`]): the
/// sides as the state was first reached, from which it is run, and what it
/// goes on to.
struct Node {
    /// The first side, until the state is run ([`Moot::expand`]), which
    /// takes it: nothing reads it after.
    first: Side,
    /// The second side, likewise.
    second: Side,
    /// The memory state.
    memory: usize,
    /// Whether both sides have just entered blocks.
    entered: bool,
    /// Whether the sides have just met at a pair of joins: the states a
    /// proof that goes round must pass through again and again.
    joined: bool,
    /// The blocks a side has entered alone since the sides last went on
    /// together ([`Waits`]), as the first way to the state left them: they
    /// are no part of what the state is ([`Configuration`]).
    waits: Waits,
    /// How to come to the state, where it has not been come to yet: a state
    /// is made only once it is wanted ([`Moot::play`]).
    recipe: Option<Recipe>,
    /// What the state goes on to.
    moves: Moves,
    /// The rows the run from the state matched, `(block, row)` on each side.
    matched: Pairs,
}

impl Node {
    /// A state that is its own end: `moves` is [`Moves::Won`] or
    /// [`Moves::Lost`].
    fn end(moves: Moves) -> Self {
        Self {
            first: Side::new(0),
            second: Side::new(0),
            memory: 0,
            entered: false,
            joined: false,
            waits: Waits::new(),
            recipe: None,
            moves,
            matched: Vec::new(),
        }
    }
}

/// How to come to a [`Node`] not made yet: the sides it is come to from,
/// the block each enters, if any, the memory state, whether both have just
/// entered blocks, and the waits.
struct Recipe {
    /// The first side before it enters its block.
    first: Rc<Side>,
    /// The second side before it enters its block.
    second: Rc<Side>,
    /// The block each side enters.
    into: [Option<usize>; 2],
    /// The memory state.
    memory: usize,
    /// Whether both sides will have just entered blocks.
    entered: bool,
    /// The waits, the new one counted.
    waits: Waits,
}

/// What a [`Node`] goes on to.
enum Moves {
    /// Not run yet.
    Open,
    /// The sides leave alike.
    Won,
    /// The sides differ.
    Lost,
    /// One state.
    Next(usize),
    /// Every one of the states: the ways on that no choice decides.
    All(Vec<usize>),
    /// Every state of either list, as the proof chooses: which side branches
    /// alone while the other waits.
    Either([Vec<usize>; 2]),
}

impl Moves {
    /// The states this goes on to.
    fn states(&self) -> Vec<usize> {
        match self {
            Self::Open | Self::Won | Self::Lost => Vec::new(),
            Self::Next(state) => vec![*state],
            Self::All(states) => states.clone(),
            Self::Either([first, second]) => first.iter().chain(second).copied().collect(),
        }
    }
}

/// The state every way the sides differ goes to ([`Moot::play`]).
const LOST: usize = 0;

/// The blocks a side has entered alone while the other stood where it
/// stands, since the two last went on together - a matched effect, a pair of
/// branches asking one question, a pair of joins: whether it is the first
/// side, the block, and where the other stood.
///
/// A side that enters such a block again while the other still stands there
/// goes round a loop that shows nothing, and what it would do on leaving is
/// no proof of the same ([`Moot::next`]).
type Waits = BTreeSet<(bool, usize, usize, usize)>;

/// What [`Moot::solve`] says of a state.
#[derive(Debug, Clone, Copy)]
struct Won {
    /// Whether the proof wins it.
    won: bool,
    /// At a [`Moves::Either`] the proof wins, the list it chooses.
    choice: Option<usize>,
}

/// The game of [`Moot::solve`] over the states: each [`Moves::Either`] is a
/// choice of the proof's between two states of the refutation's, one per
/// list, which every state of the list follows.
struct Arena {
    /// Whether the proof chooses at each state.
    proof: Vec<bool>,
    /// Whether each state is one the proof must come back to.
    accepting: Vec<bool>,
    /// What each state goes on to.
    next: Vec<Vec<usize>>,
    /// What goes on to each state.
    previous: Vec<Vec<usize>>,
}

impl Arena {
    /// The game of `nodes`; the lists of an [`Moves::Either`] at the end.
    fn of(nodes: &[Node]) -> Self {
        let mut proof = Vec::new();
        let mut accepting = Vec::new();
        let mut next = Vec::new();
        let mut lists = Vec::new();
        for (state, node) in nodes.iter().enumerate() {
            let (chooses, goes) = match &node.moves {
                Moves::Won => (true, vec![state]),
                Moves::Open | Moves::Lost => (false, vec![state]),
                Moves::Next(other) => (false, vec![*other]),
                Moves::All(states) => (false, states.clone()),
                Moves::Either([first, second]) => {
                    let at = nodes.len().saturating_add(lists.len());
                    lists.push(first.clone());
                    lists.push(second.clone());
                    (true, vec![at, at.saturating_add(1)])
                }
            };
            proof.push(chooses);
            accepting.push(node.joined || matches!(node.moves, Moves::Won));
            next.push(goes);
        }
        for list in lists {
            proof.push(false);
            accepting.push(false);
            next.push(list);
        }
        let mut previous = vec![Vec::new(); next.len()];
        for (state, goes) in next.iter().enumerate() {
            for other in goes {
                if let Some(into) = previous.get_mut(*other) {
                    into.push(state);
                }
            }
        }
        Self {
            proof,
            accepting,
            next,
            previous,
        }
    }

    /// How many states the game has.
    fn size(&self) -> usize {
        self.next.len()
    }

    /// Whether `state` is one the proof must come back to.
    fn accepting(&self, state: usize) -> bool {
        self.accepting.get(state).copied().unwrap_or(false)
    }

    /// The live states from which the proof (`proof`) or the refutation can
    /// force the play into `target`, and each one's distance from it.
    fn attractor(&self, alive: &[bool], target: &[bool], proof: bool) -> (Vec<bool>, Vec<usize>) {
        let live = |state: usize| alive.get(state) == Some(&true);
        let mut reached = vec![false; self.size()];
        let mut ranks = vec![usize::MAX; self.size()];
        let mut left: Vec<usize> = self
            .next
            .iter()
            .map(|goes| goes.iter().filter(|other| live(**other)).count())
            .collect();
        let mut queue = VecDeque::new();
        for state in 0..self.size() {
            if live(state) && target.get(state) == Some(&true) {
                if let (Some(seen), Some(rank)) = (reached.get_mut(state), ranks.get_mut(state)) {
                    *seen = true;
                    *rank = 0;
                }
                queue.push_back(state);
            }
        }
        while let Some(state) = queue.pop_front() {
            let rank = ranks.get(state).copied().unwrap_or(0).saturating_add(1);
            for before in self.previous.get(state).into_iter().flatten().copied() {
                if !live(before) || reached.get(before) == Some(&true) {
                    continue;
                }
                let mine = self.proof.get(before).copied().unwrap_or(false) == proof;
                let forced = mine || {
                    let count = left.get_mut(before);
                    count.is_some_and(|count| {
                        *count = count.saturating_sub(1);
                        *count == 0
                    })
                };
                if forced {
                    if let (Some(seen), Some(at)) = (reached.get_mut(before), ranks.get_mut(before))
                    {
                        *seen = true;
                        *at = rank;
                    }
                    queue.push_back(before);
                }
            }
        }
        (reached, ranks)
    }

    /// At a state the proof chooses at, the live list nearest a meeting.
    fn choice(&self, state: usize, alive: &[bool], ranks: &[usize]) -> Option<usize> {
        if !self.proof.get(state).copied().unwrap_or(false) {
            return None;
        }
        let goes = self.next.get(state)?;
        if goes.len() != 2 {
            return None;
        }
        goes.iter()
            .enumerate()
            .filter(|(_, other)| alive.get(**other) == Some(&true))
            .min_by_key(|(_, other)| ranks.get(**other).copied().unwrap_or(usize::MAX))
            .map(|(at, _)| at)
    }
}

/// How two sides pair a pair of joins they have entered ([`Moot::pairing`]):
/// for each side its phis, the slot each takes, what it holds as the test was
/// reached, and what it lets go of.
struct Pairing {
    /// The first side's phis, slots, held values and let-go values.
    first: Joined,
    /// The second side's.
    second: Joined,
}

/// One side of a [`Pairing`]: its join's phis, the slot each takes, what it
/// holds as the test was reached, and what it lets go of.
type Joined = (
    Vec<SsaVarId>,
    Vec<Slot>,
    Vec<(SsaVarId, Term)>,
    BTreeSet<SsaVarId>,
);

/// The search for a test whose outcome changes nothing.
struct Moot<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// Its dominators.
    dominators: &'a DominatorTree,
    /// The blocks more than one edge enters: where two ways through can
    /// come to one state ([`Self::node`]).
    merges: &'a BitSet,
    /// The test's block.
    test: usize,
    /// The blocks a path from the test reaches.
    ahead: BitSet,
    /// Every term met, by what it is.
    terms: FxHashMap<Value<T>, Term>,
    /// What each term is, by its number.
    meanings: Vec<Value<T>>,
    /// Whether each term is one value however many trips the sides make
    /// ([`Self::invariant`]), by its number: decided once, as the term is
    /// first met, from its operands', which are terms met before it.
    invariants: Vec<bool>,
    /// Whether each term is built from something the run itself made up -
    /// an effect's definition, a load's memory state - by its number
    /// ([`Namer`]): decided once, as the term is first met.
    fresh: Vec<bool>,
    /// Effects matched so far.
    effects: usize,
    /// Memory states numbered so far.
    states: usize,
    /// Events left to run.
    steps: usize,
    /// The game's states ([`Self::play`]), the ends first.
    nodes: Vec<Node>,
    /// The states by where they stand ([`Configuration`]), each under its
    /// sketch ([`Moot::sketch`]), in places of [`Self::kept`].
    ///
    /// Most states a search comes to it never comes to again, and naming one
    /// costs every made-up term its values are built from. A state is named
    /// only once another one comes that is sketched alike: the same state
    /// always is, whatever the run made up on the way to it.
    index: FxHashMap<u64, SmallVec<[usize; 1]>>,
    /// The states the index holds, by their place in it.
    kept: Vec<Kept>,
    /// What each term is with every number the run made up left out, by
    /// term ([`Moot::blind`]).
    blind: Vec<u64>,
    /// Every made-up term a configuration names, each held once, by its
    /// number ([`Configuration::made`]), and the forms states name them by.
    names: Names<T>,
    /// The blocks a path from each block reaches, as asked.
    reach: RefCell<FxHashMap<usize, BitSet>>,
    /// What the search asks, before each state it runs, whether it must stop.
    interrupt: &'a dyn Interrupt,
    /// The blocks the search has read: the test's, and each a side entered or
    /// ran in. What it concludes is a function of these, of the shape of the
    /// control flow, and of where each value it reads is defined
    /// ([`content`]).
    read: BitSet,
}

/// Where one side stands in a [`Configuration`]: its block and row, and
/// what it holds and has let go of that is defined there, each value by its
/// name ([`Namer`]).
#[derive(Debug, PartialEq, Eq, Hash)]
struct Standing {
    /// The block it is in.
    block: usize,
    /// The row it stands at.
    at: usize,
    /// The values it holds that are defined there, with their names.
    values: Vec<(SsaVarId, usize)>,
    /// The values it has let go of that are defined there.
    forgotten: Vec<SsaVarId>,
}

/// Where a run of the two sides stands, as far as what it goes on to do
/// depends on it ([`Snapshot::named`]): each side's block and row with
/// the values it holds and has let go of that are defined where it stands,
/// the terms the run made up that those values are built from, and whether
/// both sides have just entered blocks.
///
/// The waits are no part of it ([`Waits`]). Two ways to one configuration
/// that took one side's lone branches in different orders leave different
/// waits, and the sides go on alike from both: the waits only end, as
/// lost, a way that goes round with one side alone. Kept apart, every order
/// of the lone branches of two sides that differ would be a state of its own,
/// as many as the orders. A state keeps the waits of the first way that came
/// to it, and they still end, as lost, a side that goes on from it alone and
/// comes back, so the states stay finite. What the waits end is only ever
/// lost, so a proof is still a way through states the sides really go to,
/// whichever way's waits a state kept.
///
/// What the run made up - the memory states it numbers and the effects it
/// matches - is named in the order it is met here, the memory the run stands
/// in first ([`Namer`]): two configurations that differ only in those
/// numbers go on alike, as a loop's next trip goes on as its last did.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Configuration {
    /// Where each side stands.
    sides: [Standing; 2],
    /// The made-up terms the sides' names refer to, in naming order, each by
    /// its number among the search's made-up terms ([`Names::numbers`]).
    made: Box<[u32]>,
    /// Whether both sides have just entered blocks.
    entered: bool,
    /// Whether the sides have just met at a pair of joins ([`Moot::meet`]),
    /// which is what a proof that goes round must pass through.
    joined: bool,
}

/// Where a side stands, as a [`Configuration`] is taken from it: the values
/// it holds by term, not yet by name.
struct Held {
    /// The block it is in.
    block: usize,
    /// The row it stands at.
    at: usize,
    /// The values it holds, with their terms.
    values: Vec<(SsaVarId, Term)>,
    /// The values it has let go of.
    forgotten: Vec<SsaVarId>,
}

impl Held {
    /// Where `side` stands.
    fn of(side: &Side) -> Self {
        Self {
            block: side.block,
            at: side.at,
            values: side
                .values
                .iter()
                .map(|(value, term)| (*value, *term))
                .collect(),
            forgotten: side.forgotten.iter().copied().collect(),
        }
    }
}

/// What a [`Configuration`] is taken from: where the sides stand, the memory
/// the run stands in, and whether the sides have just entered blocks or met
/// at a pair of joins.
struct Snapshot {
    /// Where each side stands.
    sides: [Held; 2],
    /// The memory state.
    memory: usize,
    /// Whether both sides have just entered blocks.
    entered: bool,
    /// Whether the sides have just met at a pair of joins.
    joined: bool,
}

impl Snapshot {
    /// Where the run stands ([`Configuration`]), its terms named
    /// ([`Namer`]).
    fn named<T: Target>(
        &self,
        meanings: &[Value<T>],
        fresh: &[bool],
        names: &mut Names<T>,
    ) -> Configuration {
        let mut namer = Namer::new(meanings, fresh, names, self.memory);
        let mut standing = |side: &Held| {
            let values = side
                .values
                .iter()
                .map(|(value, term)| (*value, namer.name(*term)))
                .collect();
            Standing {
                block: side.block,
                at: side.at,
                values,
                forgotten: side.forgotten.clone(),
            }
        };
        let [first, second] = &self.sides;
        let sides = [standing(first), standing(second)];
        Configuration {
            sides,
            made: namer.made.into_boxed_slice(),
            entered: self.entered,
            joined: self.joined,
        }
    }
}

/// A state the index holds ([`Moot::index`]).
enum Kept {
    /// Where it stands, not named yet: no state sketched alike has come.
    Taken(usize, Snapshot),
    /// Where it stands, named.
    Named(usize, Configuration),
}

/// A term the run made up, as a [`Configuration`] names it: each made-up
/// operand by its name, and each made-up number by the order it was met in.
#[derive(Debug, PartialEq, Eq, Hash)]
enum Made {
    /// The `index`th definition of the `effect`th effect met.
    Effect(usize, usize),
    /// The `index`th definition of a pure operation of shape `shape`
    /// ([`Form`]), reading the made-up terms named `operands`.
    Pure(u32, SmallVec<[u32; 4]>, usize),
    /// A load of shape `shape`, reading the made-up terms named `operands`,
    /// in the `memory`th memory state met.
    Load(u32, SmallVec<[u32; 4]>, usize),
}

/// A made-up operation or load as every state names it: what it is with its
/// made-up operands left out, and those operands in the order it reads them.
///
/// What a term is never changes as the search goes on; only the names its
/// made-up operands take in each state do. Held once per term, a state names
/// it by its operands' names alone, rather than by a copy of the operation
/// renamed for that state.
struct Form {
    /// The number of the operation with every made-up operand left out
    /// ([`Names::shapes`]).
    shape: u32,
    /// The made-up terms the operation reads, in the order read.
    operands: SmallVec<[Term; 4]>,
}

/// What a search's states name their made-up terms by, held across them all.
struct Names<T: Target> {
    /// Each made-up term by its number.
    ///
    /// A search that keeps going round makes up deeper terms on every trip,
    /// and each state it stands in names every one its values are built from:
    /// thousands of terms, nearly all of them the last state's. Held once each
    /// and named by number, a state costs four bytes per term rather than a
    /// copy of the term's operation.
    numbers: FxHashMap<Made, u32>,
    /// Each operation, its made-up operands left out, by its number.
    shapes: FxHashMap<SsaOp<T>, u32>,
    /// Each made-up operation's or load's form, by term, once one is named.
    forms: Vec<Option<Form>>,
    /// Each term's place among the made-up terms the state being named has
    /// named, by term: valid where its stamp is the state's ([`Self::stamp`]).
    ///
    /// A search names states by the thousand, each of them many made-up
    /// terms; held across them and stamped, the places are found by index,
    /// and no state builds or grows a map of its own.
    places: Vec<(u32, u32)>,
    /// The stamp of the state being named.
    stamp: u32,
}

impl<T: Target> Default for Names<T> {
    fn default() -> Self {
        Self {
            numbers: FxHashMap::default(),
            shapes: FxHashMap::default(),
            forms: Vec::new(),
            places: Vec::new(),
            stamp: 0,
        }
    }
}

impl<T: Target> Names<T> {
    /// Starts naming a state: every place set before is let go.
    fn start(&mut self) {
        if self.stamp == u32::MAX {
            self.places.clear();
            self.stamp = 0;
        }
        self.stamp = self.stamp.saturating_add(1);
    }

    /// The place of `term` in the state being named, if it is named.
    fn place(&self, term: Term) -> Option<u32> {
        self.places
            .get(term)
            .and_then(|(stamp, place)| (*stamp == self.stamp).then_some(*place))
    }

    /// Gives `term` the place `place` in the state being named.
    fn set(&mut self, term: Term, place: u32) {
        if self.places.len() <= term {
            self.places.resize(term.saturating_add(1), (0, 0));
        }
        if let Some(slot) = self.places.get_mut(term) {
            *slot = (self.stamp, place);
        }
    }
}

/// Names the terms of a [`Configuration`]: a term the run did not make up by
/// itself, twice over, and a made-up one by its place in [`Configuration::made`],
/// twice over and one more, numbering what it was made up from in the order
/// it is met.
struct Namer<'m, T: Target> {
    /// What each term is.
    meanings: &'m [Value<T>],
    /// Whether each term is made up.
    fresh: &'m [bool],
    /// The search's made-up terms and their forms ([`Moot::names`]).
    held: &'m mut Names<T>,
    /// The made-up terms named so far, by their numbers.
    made: Vec<u32>,
    /// Each memory state's number, in the order met.
    memories: FxHashMap<usize, usize>,
    /// Each effect's number, in the order met.
    effects: FxHashMap<usize, usize>,
    /// The terms [`Self::name`] is walking, kept from one name to the next.
    walk: Vec<Term>,
}

impl<'m, T: Target> Namer<'m, T> {
    /// A namer that meets `memory` first.
    fn new(
        meanings: &'m [Value<T>],
        fresh: &'m [bool],
        held: &'m mut Names<T>,
        memory: usize,
    ) -> Self {
        let mut memories = FxHashMap::default();
        memories.insert(memory, 0);
        held.start();
        Self {
            meanings,
            fresh,
            held,
            made: Vec::new(),
            memories,
            effects: FxHashMap::default(),
            walk: Vec::new(),
        }
    }

    /// The name of `term`, naming first every made-up term it is built from,
    /// operands before what reads them and in the order read.
    ///
    /// The terms are walked on a stack of their own, not by recursion: a
    /// chain of values computed one from the other, as long as the code that
    /// computes them, names in a stack of one frame.
    fn name(&mut self, term: Term) -> usize {
        if self.unnamed(term) {
            let mut walk = mem::take(&mut self.walk);
            walk.push(term);
            while let Some(&top) = walk.last() {
                if let Some(operand) = self.unnamed_operand(top) {
                    walk.push(operand);
                    continue;
                }
                walk.pop();
                if self.unnamed(top) {
                    self.make(top);
                }
            }
            self.walk = walk;
        }
        self.named(term)
    }

    /// Whether `term` is one the run made up by itself, which a state names
    /// by its place among those it names.
    fn made_up(meanings: &[Value<T>], fresh: &[bool], term: Term) -> bool {
        fresh.get(term).copied().unwrap_or(false)
            && matches!(
                meanings.get(term),
                Some(Value::Effect(..) | Value::Pure(..) | Value::Load(..))
            )
    }

    /// Whether `term` is a made-up term not named yet.
    fn unnamed(&self, term: Term) -> bool {
        self.held.place(term).is_none() && Self::made_up(self.meanings, self.fresh, term)
    }

    /// The form of the made-up operation or load `term`, found the first time
    /// a state names it; `None` for an effect.
    fn form(&mut self, term: Term) -> Option<&Form> {
        let (Value::Pure(op, _) | Value::Load(op, _)) = self.meanings.get(term)? else {
            return None;
        };
        if self.held.forms.len() <= term {
            self.held.forms.resize_with(term.saturating_add(1), || None);
        }
        if self.held.forms.get(term).is_some_and(Option::is_none) {
            let (meanings, fresh) = (self.meanings, self.fresh);
            let mut shape = op.clone();
            let mut operands = SmallVec::new();
            // A made-up operand is left out by one odd name, which no other
            // operand, named twice over, takes.
            shape.visit_operands_mut(|role, value| {
                if role == OperandRole::Use {
                    let operand = value.index();
                    *value = if Self::made_up(meanings, fresh, operand) {
                        operands.push(operand);
                        SsaVarId::from_index(1)
                    } else {
                        SsaVarId::from_index(operand.saturating_mul(2))
                    };
                }
            });
            let next = u32::try_from(self.held.shapes.len()).unwrap_or(u32::MAX);
            let shape = *self.held.shapes.entry(shape).or_insert(next);
            if let Some(slot) = self.held.forms.get_mut(term) {
                *slot = Some(Form { shape, operands });
            }
        }
        self.held.forms.get(term)?.as_ref()
    }

    /// The first operand of the unnamed term `term` that is unnamed too, if
    /// any.
    fn unnamed_operand(&mut self, term: Term) -> Option<Term> {
        if !self.unnamed(term) {
            return None;
        }
        let count = self.form(term)?.operands.len();
        (0..count)
            .filter_map(|at| {
                self.held
                    .forms
                    .get(term)?
                    .as_ref()?
                    .operands
                    .get(at)
                    .copied()
            })
            .find(|operand| self.unnamed(*operand))
    }

    /// The names of the made-up terms `term`'s form reads, each named already.
    fn operand_names(&mut self, term: Term) -> (u32, SmallVec<[u32; 4]>) {
        let Some(form) = self.form(term) else {
            return (u32::MAX, SmallVec::new());
        };
        let shape = form.shape;
        let held = &*self.held;
        let names = held
            .forms
            .get(term)
            .and_then(Option::as_ref)
            .map(|form| {
                form.operands
                    .iter()
                    .map(|operand| held.place(*operand).unwrap_or(u32::MAX))
                    .collect()
            })
            .unwrap_or_default();
        (shape, names)
    }

    /// Names `term`, every made-up term it is built from named already.
    fn make(&mut self, term: Term) {
        let made = match self.meanings.get(term) {
            Some(Value::Effect(effect, index)) => {
                let count = self.effects.len();
                Made::Effect(*self.effects.entry(*effect).or_insert(count), *index)
            }
            Some(Value::Pure(_, index)) => {
                let index = *index;
                let (shape, operands) = self.operand_names(term);
                Made::Pure(shape, operands, index)
            }
            Some(Value::Load(_, memory)) => {
                let memory = *memory;
                let (shape, operands) = self.operand_names(term);
                let count = self.memories.len();
                Made::Load(
                    shape,
                    operands,
                    *self.memories.entry(memory).or_insert(count),
                )
            }
            _ => return,
        };
        let next = u32::try_from(self.held.numbers.len()).unwrap_or(u32::MAX);
        let number = *self.held.numbers.entry(made).or_insert(next);
        let name = u32::try_from(self.made.len()).unwrap_or(u32::MAX);
        self.made.push(number);
        self.held.set(term, name);
    }

    /// The name of `term`, named already or not made up: a made-up term by
    /// its place in [`Self::made`], any other by itself.
    fn named(&self, term: Term) -> usize {
        match self.held.place(term) {
            Some(name) => usize::try_from(name)
                .unwrap_or(usize::MAX)
                .saturating_mul(2)
                .saturating_add(1),
            None => term.saturating_mul(2),
        }
    }
}

impl<'a, T: Target> Moot<'a, T> {
    /// The search of `test`, with `ahead` the blocks a path from it reaches,
    /// within `steps` events.
    fn new(
        ssa: &'a SsaFunction<T>,
        dominators: &'a DominatorTree,
        merges: &'a BitSet,
        test: usize,
        ahead: BitSet,
        steps: usize,
    ) -> Self {
        Self {
            ssa,
            dominators,
            merges,
            test,
            ahead,
            terms: FxHashMap::default(),
            meanings: Vec::new(),
            invariants: Vec::new(),
            fresh: Vec::new(),
            effects: 0,
            states: 0,
            steps,
            nodes: Vec::new(),
            index: FxHashMap::default(),
            kept: Vec::new(),
            blind: Vec::new(),
            names: Names::default(),
            reach: RefCell::new(FxHashMap::default()),
            interrupt: &NeverInterrupted,
            read: BitSet::new(ssa.block_count()),
        }
    }

    /// The first test of `ssa` whose outcome changes nothing, and the arm it
    /// goes on to, within `steps` events, which it spends.
    ///
    /// A test `ssa` holds a refutation for ([`SsaFunction::refutations`]) is
    /// not searched while the blocks that search read read as they did: its
    /// arms still differ there. One whose search ran out of steps, read the
    /// same with no more steps left than it had, runs out again, and the
    /// search ends there as it would have. Each search that settles nothing
    /// and was not stopped is added to `learned`, for the caller to record.
    fn find(
        ssa: &'a SsaFunction<T>,
        steps: &mut usize,
        interrupt: &'a dyn Interrupt,
        learned: &mut Vec<(u128, Refutation)>,
    ) -> Option<(usize, usize, Pairs)> {
        let shape = shape(ssa);
        let cfg = SsaCfg::from_ssa(ssa);
        let merges = merges(&cfg);
        let dominators = compute_dominators(&cfg, NodeId::new(0));
        let reached = |from: usize| -> BitSet {
            let mut seen = BitSet::new(ssa.block_count());
            let mut stack = vec![from];
            while let Some(block) = stack.pop() {
                if seen.insert_checked(block) {
                    stack.extend(cfg.successors(NodeId::new(block)).map(NodeId::index));
                }
            }
            seen
        };
        let loops = detect_loops(&cfg, &dominators);
        // Whether the arm into `to`, with `test` a jump into it, goes round
        // the innermost loop `test` is in for ever: every block it reaches
        // short of coming back through `test` lies in that loop.
        let stuck = |to: usize, test: usize| -> bool {
            let Some(inner) = loops.innermost_loop(NodeId::new(test)) else {
                return false;
            };
            let mut seen = BitSet::new(ssa.block_count());
            let mut stack = vec![to];
            while let Some(block) = stack.pop() {
                if block == test || !seen.insert_checked(block) {
                    continue;
                }
                if !inner.body.contains(block) {
                    return false;
                }
                stack.extend(cfg.successors(NodeId::new(block)).map(NodeId::index));
            }
            true
        };
        for test in 0..ssa.block_count() {
            if interrupt.interrupted() {
                return None;
            }
            if *steps == 0 {
                return None;
            }
            let Some((on_true, on_false)) = ssa
                .block(test)
                .and_then(|block| block.control_terminator())
                .and_then(|terminator| match terminator {
                    SsaOp::Branch { .. } | SsaOp::BranchCmp { .. } => terminator.two_way_targets(),
                    _ => None,
                })
            else {
                continue;
            };
            // An arm that only computes values on its way into the other is
            // control-flow simplification's, which keeps what it ran over.
            if on_true == on_false
                || runs_into(ssa, on_true, on_false)
                || runs_into(ssa, on_false, on_true)
            {
                continue;
            }
            let question = question(shape, test, on_true, on_false);
            if let Some(known) = ssa.refutations().get(question)
                && known.content() == content(ssa, known.read())
            {
                // Asked again of what it read, the search ends as it did,
                // spending what it spent.
                match known.ended().again(*steps) {
                    Again::Refuted(left) => {
                        *steps = left;
                        continue;
                    }
                    Again::RanOut => {
                        *steps = 0;
                        return None;
                    }
                    Again::Search => {}
                }
            }
            let (first, second) = (reached(on_true), reached(on_false));
            let mut ahead = first.clone();
            ahead.union_with(&second);
            let mut search = Moot::new(ssa, &dominators, &merges, test, ahead, *steps);
            search.interrupt = interrupt;
            let had = *steps;
            let played = search.play(on_true, on_false);
            *steps = search.steps;
            // A stopped search settled nothing it could be asked again for.
            if played.is_none() && !interrupt.interrupted() {
                let read: Vec<usize> = search.read.iter().collect();
                let content = content(ssa, &read);
                let ended = if search.steps == 0 {
                    Unsettled::RanOut(had)
                } else {
                    Unsettled::Refuted(had.saturating_sub(search.steps))
                };
                learned.push((question, Refutation::new(read, content, ended)));
            }
            if let Some(pairs) = played {
                // With the test a jump, an arm round a loop the test was the
                // only way out of goes round for ever.
                let keeps_false = match (stuck(on_true, test), stuck(on_false, test)) {
                    (true, false) => true,
                    (false, true) => false,
                    _ => second.count() > first.count(),
                };
                return Some(if keeps_false {
                    let swapped = pairs.into_iter().map(|(first, second)| (second, first));
                    (test, on_false, swapped.collect())
                } else {
                    (test, on_true, pairs)
                });
            }
        }
        None
    }

    /// The rows the arms `on_true` and `on_false` pair where they do the
    /// same, from what the test's block defines as its branch is reached;
    /// `None` where they differ or the steps run out first.
    ///
    /// The arms are played as a game over the states of the two sides
    /// ([`Configuration`]): each state is run once to where the sides do
    /// something that decides ([`Self::expand`]) - a pair of joins, which
    /// they meet at as a loop goes round, ways on that every one of must go
    /// on alike, or a choice of which side branches alone - and the proof
    /// wins where it can keep the sides alike for ever, meeting at joins
    /// again and again ([`Self::solve`]). A side that goes round with no join
    /// to meet the other at shows nothing, and is no proof. Each state is
    /// run once however many ways lead to it, so the play grows with the
    /// code and not with the ways through it.
    fn play(&mut self, on_true: usize, on_false: usize) -> Option<Pairs> {
        let (entered, memory) = self.entered()?;
        let (mut first, mut second) = (entered.clone(), entered);
        self.enter(&mut first, on_true)?;
        self.enter(&mut second, on_false)?;
        self.nodes = vec![Node::end(Moves::Lost), Node::end(Moves::Won)];
        self.index.clear();
        self.kept.clear();
        let root = self.node(first, second, memory, true, false, Waits::new());
        // What leads to each state, and which states are lost for certain:
        // a state every way on from which must hold is lost once one is, a
        // choice once both its lists are, and a state only lost states lead
        // to is never run. Run first way on first, a test that differs is
        // lost as soon as one way through shows it, and one that does not is
        // proved once.
        let mut previous: Vec<Vec<(usize, Option<usize>)>> = Vec::new();
        let mut lost = vec![true, false];
        let mut open = vec![root];
        while let Some(state) = open.pop() {
            if self.interrupt.interrupted() {
                return None;
            }
            if lost.get(root) == Some(&true) {
                return None;
            }
            // Wanted where something not yet lost leads to it, through a list
            // of a choice that is not lost either.
            let wanted = state == root
                || previous.get(state).is_some_and(|before| {
                    before.iter().any(|(at, list)| {
                        lost.get(*at) != Some(&true)
                            && list.is_none_or(|list| !self.list_lost(*at, list, &lost))
                    })
                });
            if !wanted || !matches!(self.nodes.get(state)?.moves, Moves::Open) {
                continue;
            }
            let moves = match self.come(state) {
                Some(moves) => moves,
                None => self.expand(state)?,
            };
            let goes: Vec<(usize, Option<usize>)> = match &moves {
                Moves::Either(lists) => lists
                    .iter()
                    .enumerate()
                    .flat_map(|(list, states)| states.iter().map(move |next| (*next, Some(list))))
                    .collect(),
                _ => moves
                    .states()
                    .into_iter()
                    .map(|next| (next, None))
                    .collect(),
            };
            previous.resize_with(self.nodes.len(), Vec::new);
            lost.resize(self.nodes.len(), false);
            for (next, list) in goes.iter().rev() {
                if let Some(before) = previous.get_mut(*next) {
                    before.push((state, *list));
                }
                if self
                    .nodes
                    .get(*next)
                    .is_some_and(|node| matches!(node.moves, Moves::Open))
                {
                    open.push(*next);
                }
            }
            self.nodes.get_mut(state)?.moves = moves;
            let mut settled = vec![state];
            while let Some(at) = settled.pop() {
                if lost.get(at) == Some(&true) || !self.lost(at, &lost) {
                    continue;
                }
                if let Some(entry) = lost.get_mut(at) {
                    *entry = true;
                }
                settled.extend(
                    previous
                        .get(at)
                        .into_iter()
                        .flatten()
                        .map(|(before, _)| *before),
                );
            }
        }
        let strategy = self.solve();
        let mut chosen = BitSet::new(self.nodes.len());
        let mut pending = vec![root];
        let mut pairs = Vec::new();
        while let Some(state) = pending.pop() {
            if !chosen.insert_checked(state) {
                continue;
            }
            let node = self.nodes.get(state)?;
            match &node.moves {
                Moves::Won => {}
                Moves::Next(next) => pending.push(*next),
                Moves::All(states) => pending.extend(states),
                Moves::Either(lists) => pending.extend(lists.get(strategy.get(state)?.choice?)?),
                Moves::Open | Moves::Lost => return None,
            }
            if !strategy.get(state)?.won {
                return None;
            }
            pairs.extend(node.matched.iter().copied());
        }
        Some(pairs)
    }

    /// Whether `state` is lost for certain, given the states `lost` says are:
    /// it ends differing, or every way on from it must hold and one is lost,
    /// or it chooses and every choice has a lost way on.
    fn lost(&self, state: usize, lost: &[bool]) -> bool {
        let is_lost = |other: &usize| lost.get(*other) == Some(&true);
        match self.nodes.get(state).map(|node| &node.moves) {
            Some(Moves::Lost) => true,
            Some(Moves::Next(other)) => is_lost(other),
            Some(Moves::All(states)) => states.iter().any(is_lost),
            Some(Moves::Either(lists)) => lists.iter().all(|list| list.iter().any(is_lost)),
            Some(Moves::Open | Moves::Won) | None => false,
        }
    }

    /// Whether the `list`th list of the choice `state` makes has a lost way on.
    fn list_lost(&self, state: usize, list: usize, lost: &[bool]) -> bool {
        match self.nodes.get(state).map(|node| &node.moves) {
            Some(Moves::Either(lists)) => lists
                .get(list)
                .is_some_and(|states| states.iter().any(|other| lost.get(*other) == Some(&true))),
            _ => false,
        }
    }

    /// The state the sides stand in, run or not; a new one is run later.
    fn node(
        &mut self,
        first: Side,
        second: Side,
        memory: usize,
        entered: bool,
        joined: bool,
        waits: Waits,
    ) -> usize {
        let (mut first, mut second, mut waits) = (first, second, waits);
        let state = self.nodes.len();
        if self.merging(&first, &second, joined) {
            waits.retain(|(is_first, target, _, _)| {
                let side = if *is_first { first.block } else { second.block };
                self.reaches(side, *target)
            });
            self.prune(&mut first);
            self.prune(&mut second);
            if let Some(same) = self.settle(&first, &second, memory, entered, joined, state) {
                return same;
            }
        } else {
            first.trail.clear();
            second.trail.clear();
        }
        self.nodes.push(Node {
            first,
            second,
            memory,
            entered,
            joined,
            waits,
            recipe: None,
            moves: Moves::Open,
            matched: Vec::new(),
        });
        state
    }

    /// Ends a run where a side has jumped into a block more than one edge
    /// enters: what goes on from there is the state it stands in, which
    /// another way may come to too ([`Self::merging`]).
    fn cut(&mut self, first: Side, second: Side, memory: usize, waits: Waits) -> Moves {
        let entered = first.at == 0 && second.at == 0;
        Moves::Next(self.node(first, second, memory, entered, false, waits))
    }

    /// Whether two ways through can come to the state `first` and `second`
    /// stand in: where they have just met at a pair of joins, or where a side
    /// has just entered a block more than one edge enters. Anywhere else the
    /// state is come to one way, and is not looked up.
    fn merging(&self, first: &Side, second: &Side, joined: bool) -> bool {
        joined
            || [first, second]
                .iter()
                .any(|side| side.at == 0 && self.merges.contains_checked(side.block))
    }

    /// Runs `state` to what it goes on to, recording the rows it matches;
    /// `None` where the steps run out.
    fn expand(&mut self, state: usize) -> Option<Moves> {
        // A state is run once, and nothing reads its sides after: they are
        // taken, not copied. A reader of them added later finds them empty.
        let node = self.nodes.get_mut(state)?;
        let (mut first, mut second) = (mem::take(&mut node.first), mem::take(&mut node.second));
        let (mut memory, entered) = (node.memory, node.entered);
        let mut waits = node.waits.clone();
        let mut matched = Vec::new();
        if entered && let Some(moves) = self.meet(&mut first, &mut second, &mut memory) {
            return Some(moves);
        }
        let moves = loop {
            self.steps = self.steps.checked_sub(1)?;
            // Where each side stood before its step: a side that only
            // computes values on its way to its next step is where it began,
            // as the other reaches a join or a branch.
            let (first_was, second_was) = (first.mark(), second.mark());
            let (Some(ours), Some(theirs)) = (
                self.advance(&mut first, memory),
                self.advance(&mut second, memory),
            ) else {
                break Moves::Lost;
            };
            match (ours, theirs) {
                (Step::Jump(to), Step::Jump(other)) => {
                    if self.enter(&mut first, to).is_none()
                        || self.enter(&mut second, other).is_none()
                    {
                        break Moves::Lost;
                    }
                    if let Some(moves) = self.meet(&mut first, &mut second, &mut memory) {
                        break moves;
                    }
                    if self.merging(&first, &second, false) {
                        break self.cut(first, second, memory, waits);
                    }
                }
                (Step::Jump(to), _) => {
                    if self.enter(&mut first, to).is_none() {
                        break Moves::Lost;
                    }
                    if second_was.at == 0 {
                        second.rollback(second_was);
                        if let Some(moves) = self.meet(&mut first, &mut second, &mut memory) {
                            break moves;
                        }
                    }
                    if self.merging(&first, &second, false) {
                        break self.cut(first, second, memory, waits);
                    }
                }
                (_, Step::Jump(to)) => {
                    if self.enter(&mut second, to).is_none() {
                        break Moves::Lost;
                    }
                    if first_was.at == 0 {
                        first.rollback(first_was);
                        if let Some(moves) = self.meet(&mut first, &mut second, &mut memory) {
                            break moves;
                        }
                    }
                    if self.merging(&first, &second, false) {
                        break self.cut(first, second, memory, waits);
                    }
                }
                (Step::Effect(op, defs), Step::Effect(other, other_defs)) => {
                    if op != other || defs.len() != other_defs.len() {
                        break Moves::Lost;
                    }
                    let effect = self.effects;
                    self.effects = effect.saturating_add(1);
                    matched.push(((first.block, first.at), (second.block, second.at)));
                    for (index, (mine, theirs)) in defs.into_iter().zip(other_defs).enumerate() {
                        let term = self.term(Value::Effect(effect, index));
                        first.meet(mine, term);
                        second.meet(theirs, term);
                    }
                    first.at = first.at.saturating_add(1);
                    second.at = second.at.saturating_add(1);
                    memory = self.state();
                    waits.clear();
                }
                (Step::Leave(op), Step::Leave(other)) => {
                    matched.push(((first.block, first.at), (second.block, second.at)));
                    break if op == other { Moves::Won } else { Moves::Lost };
                }
                (Step::Branch(asked, on, off), Step::Branch(other, other_on, other_off)) => {
                    let paired = if asked == other {
                        Some([(on, other_on), (off, other_off)])
                    } else if asked.negates(&other) {
                        Some([(on, other_off), (off, other_on)])
                    } else {
                        None
                    };
                    // Two branches asking one question pair, or differ:
                    // neither decides nothing where the other asks it too.
                    if let Some(pairs) = paired {
                        matched.push(((first.block, first.at), (second.block, second.at)));
                        let (shared_first, shared_second) =
                            (Rc::new(first.clone()), Rc::new(second.clone()));
                        let states = pairs.map(|(mine, theirs)| {
                            self.next(
                                &shared_first,
                                &shared_second,
                                [Some(mine), Some(theirs)],
                                memory,
                                true,
                                &Waits::new(),
                            )
                        });
                        break Moves::All(states.to_vec());
                    }
                    // Either side branches alone, the other standing where it
                    // stood before its step.
                    let mut first_stood = first.clone();
                    first_stood.rollback(first_was);
                    let mut second_stood = second.clone();
                    second_stood.rollback(second_was);
                    let entered = second_stood.at == 0;
                    let (shared_first, shared_second_stood) =
                        (Rc::new(first.clone()), Rc::new(second_stood));
                    let alone = [on, off].map(|target| {
                        self.next(
                            &shared_first,
                            &shared_second_stood,
                            [Some(target), None],
                            memory,
                            entered,
                            &waits,
                        )
                    });
                    let entered = first_stood.at == 0;
                    let (shared_first_stood, shared_second) =
                        (Rc::new(first_stood), Rc::new(second.clone()));
                    let other_alone = [other_on, other_off].map(|target| {
                        self.next(
                            &shared_first_stood,
                            &shared_second,
                            [None, Some(target)],
                            memory,
                            entered,
                            &waits,
                        )
                    });
                    break Moves::Either([alone.to_vec(), other_alone.to_vec()]);
                }
                (Step::Branch(_, on, off), _) => {
                    second.rollback(second_was);
                    let entered = second.at == 0;
                    let (shared_first, shared_second) =
                        (Rc::new(first.clone()), Rc::new(second.clone()));
                    let states = [on, off].map(|target| {
                        self.next(
                            &shared_first,
                            &shared_second,
                            [Some(target), None],
                            memory,
                            entered,
                            &waits,
                        )
                    });
                    break Moves::All(states.to_vec());
                }
                (_, Step::Branch(_, on, off)) => {
                    first.rollback(first_was);
                    let entered = first.at == 0;
                    let (shared_first, shared_second) =
                        (Rc::new(first.clone()), Rc::new(second.clone()));
                    let states = [on, off].map(|target| {
                        self.next(
                            &shared_first,
                            &shared_second,
                            [None, Some(target)],
                            memory,
                            entered,
                            &waits,
                        )
                    });
                    break Moves::All(states.to_vec());
                }
                _ => break Moves::Lost,
            }
        };
        self.nodes.get_mut(state)?.matched = matched;
        Some(moves)
    }

    /// The state `first` and `second` stand in once each enters the block
    /// `into` names for it, if any, with `waits` those since the two last
    /// went on together; [`LOST`] where one cannot, and where one side alone
    /// enters a block it entered alone while the other stood where it stands
    /// ([`Waits`]).
    fn next(
        &mut self,
        first: &Rc<Side>,
        second: &Rc<Side>,
        into: [Option<usize>; 2],
        memory: usize,
        entered: bool,
        waits: &Waits,
    ) -> usize {
        let mut waits = waits.clone();
        match into {
            [Some(target), None] => {
                if !waits.insert((true, target, second.block, second.at)) {
                    return LOST;
                }
            }
            [None, Some(target)] => {
                if !waits.insert((false, target, first.block, first.at)) {
                    return LOST;
                }
            }
            _ => waits.clear(),
        }
        let state = self.nodes.len();
        let mut node = Node::end(Moves::Open);
        node.recipe = Some(Recipe {
            first: Rc::clone(first),
            second: Rc::clone(second),
            into,
            memory,
            entered,
            waits,
        });
        self.nodes.push(node);
        state
    }

    /// Comes to the state `state`'s recipe describes ([`Recipe`]): the state
    /// itself, to be run, where no state stands where it does; otherwise
    /// what it goes on to, [`Moves::Next`] the state that does, or
    /// [`Moves::Lost`] where a side cannot enter its block.
    fn come(&mut self, state: usize) -> Option<Moves> {
        let Recipe {
            first,
            second,
            into,
            memory,
            entered,
            mut waits,
        } = self.nodes.get_mut(state)?.recipe.take()?;
        let (mut first, mut second) = (Rc::unwrap_or_clone(first), Rc::unwrap_or_clone(second));
        for (side, target) in [(&mut first, into[0]), (&mut second, into[1])] {
            if let Some(target) = target
                && self.enter(side, target).is_none()
            {
                return Some(Moves::Lost);
            }
        }
        if self.merging(&first, &second, false) {
            // A block its side can no longer come to is no wait it can
            // repeat, and two ways that differ only there are one state.
            waits.retain(|(is_first, target, _, _)| {
                let side = if *is_first { first.block } else { second.block };
                self.reaches(side, *target)
            });
            self.prune(&mut first);
            self.prune(&mut second);
            if let Some(same) = self.settle(&first, &second, memory, entered, false, state) {
                return Some(Moves::Next(same));
            }
        } else {
            first.trail.clear();
            second.trail.clear();
        }
        let node = self.nodes.get_mut(state)?;
        node.first = first;
        node.second = second;
        node.memory = memory;
        node.entered = entered;
        node.waits = waits;
        None
    }

    /// Whether a path from `from` reaches `to`, `from` itself included.
    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut reach = self.reach.borrow_mut();
        let ssa = self.ssa;
        reach
            .entry(from)
            .or_insert_with(|| {
                let mut seen = BitSet::new(ssa.block_count());
                let mut stack = vec![from];
                while let Some(block) = stack.pop() {
                    if seen.insert_checked(block) {
                        stack.extend(
                            ssa.block(block)
                                .into_iter()
                                .flat_map(|data| data.successors()),
                        );
                    }
                }
                seen
            })
            .contains_checked(to)
    }

    /// Where `first` and `second` have just entered a pair of joins: the
    /// state they meet in, or [`Moves::Lost`] where a phi took nothing;
    /// `None` where either block joins nothing ([`Self::pairing`]).
    fn meet(&mut self, first: &mut Side, second: &mut Side, memory: &mut usize) -> Option<Moves> {
        let pairing = match self.pairing(first, second)? {
            Some(pairing) => pairing,
            None => return Some(Moves::Lost),
        };
        self.met(first, second, pairing, memory);
        let state = self.node(
            first.clone(),
            second.clone(),
            *memory,
            false,
            true,
            Waits::new(),
        );
        Some(Moves::Next(state))
    }

    /// Which states the proof wins, and at each it wins by choosing, the list
    /// it chooses.
    ///
    /// A Buchi game: the proof chooses at [`Moves::Either`], the refutation
    /// at every other state, and the proof must meet at joins again and
    /// again or leave alike ([`Moves::Won`]). Solved by the classical
    /// attractor iteration: the states from which the refutation can keep
    /// the sides from ever meeting again are taken away, with all from which
    /// it can force the play into them, until none are left; the proof wins
    /// the rest, choosing at each state the list nearest a meeting.
    fn solve(&self) -> Vec<Won> {
        let game = Arena::of(&self.nodes);
        let mut alive = vec![true; game.size()];
        let ranks = loop {
            let meets: Vec<bool> = (0..game.size())
                .map(|state| alive.get(state) == Some(&true) && game.accepting(state))
                .collect();
            let (reached, ranks) = game.attractor(&alive, &meets, true);
            let trapped: Vec<bool> = (0..game.size())
                .map(|state| alive.get(state) == Some(&true) && reached.get(state) != Some(&true))
                .collect();
            if !trapped.contains(&true) {
                break ranks;
            }
            let (lost, _) = game.attractor(&alive, &trapped, false);
            for (live, lost) in alive.iter_mut().zip(lost) {
                *live &= !lost;
            }
        };
        (0..self.nodes.len())
            .map(|state| Won {
                won: alive.get(state) == Some(&true),
                choice: game.choice(state, &alive, &ranks),
            })
            .collect()
    }

    /// The test's block run to its branch, from what it defines as it is
    /// reached, and the memory state it leaves; `None` where the block does
    /// not run to a branch.
    fn entered(&mut self) -> Option<(Side, usize)> {
        let ssa = self.ssa;
        self.read.insert(self.test);
        let block = ssa.block(self.test)?;
        let mut entered = Side::new(self.test);
        for phi in block.phi_nodes() {
            let term = self.term(Value::Entered(phi.result()));
            entered.meet(phi.result(), term);
        }
        let mut memory = self.state();
        loop {
            match self.advance(&mut entered, memory) {
                Some(Step::Effect(_, defs)) => {
                    for def in defs {
                        let term = self.term(Value::Entered(def));
                        entered.meet(def, term);
                    }
                    entered.at = entered.at.saturating_add(1);
                    memory = self.state();
                }
                Some(Step::Branch(..)) => break,
                _ => return None,
            }
        }
        // Each is what it was as the test was reached, however many trips
        // the sides make; one computed from constants and values fixed before
        // the test is that on every trip, and stays what it computes.
        let defined: Vec<(SsaVarId, Term)> = entered
            .values
            .iter()
            .map(|(value, term)| (*value, *term))
            .collect();
        for (value, term) in defined {
            if !self.invariant(term) {
                let term = self.term(Value::Entered(value));
                entered.meet(value, term);
            }
        }
        // Nothing is rolled back past where the arms begin.
        entered.trail.clear();
        Some((entered, memory))
    }

    /// Whether `term` is one value however many trips the sides make: fixed
    /// before the test, or a pure operation of such values, a constant among
    /// them.
    fn invariant(&self, term: Term) -> bool {
        self.invariants.get(term).copied().unwrap_or(false)
    }

    /// The term of `value`.
    ///
    /// Terms are shared, so the meanings form a graph in which one term is
    /// reached by many ways: whether a new term is invariant is read from its
    /// operands' answers, never by walking down to what they are.
    fn term(&mut self, value: Value<T>) -> Term {
        if let Some(term) = self.terms.get(&value) {
            return *term;
        }
        let invariant = match &value {
            Value::Before(_) => true,
            Value::Pure(op, _) => {
                let mut fixed = true;
                op.for_each_use(|operand| fixed &= self.invariant(operand.index()));
                fixed
            }
            _ => false,
        };
        let fresh = match &value {
            Value::Effect(..) | Value::Load(..) => true,
            Value::Pure(op, _) => {
                let mut fresh = false;
                op.for_each_use(|operand| {
                    fresh |= self.fresh.get(operand.index()).copied().unwrap_or(false);
                });
                fresh
            }
            Value::Before(_) | Value::Entered(_) | Value::Joined(_) => false,
        };
        let next = self.meanings.len();
        let blind = self.blinded(&value, next, fresh);
        self.meanings.push(value.clone());
        self.invariants.push(invariant);
        self.fresh.push(fresh);
        self.blind.push(blind);
        self.terms.insert(value, next);
        next
    }

    /// What the term `term`, which is `value`, is with every number the run
    /// made up left out: an effect's number, a load's memory state, and the
    /// places a state names made-up terms by ([`Namer`]). A made-up term is
    /// what it is built from, so two that a state names alike, in states of
    /// their own, are blinded alike; any other term is itself.
    ///
    /// Taken as a term is made, from its operands', made before it.
    fn blinded(&self, value: &Value<T>, term: Term, fresh: bool) -> u64 {
        let mut hasher = FxHasher::default();
        match value {
            Value::Effect(_, index) if fresh => {
                0_u8.hash(&mut hasher);
                index.hash(&mut hasher);
            }
            Value::Pure(op, _) | Value::Load(op, _) if fresh => {
                let (meanings, fresh, blind) = (&self.meanings, &self.fresh, &self.blind);
                let mut shape = op.clone();
                let mut operands: SmallVec<[u64; 4]> = SmallVec::new();
                shape.visit_operands_mut(|role, value| {
                    if role == OperandRole::Use {
                        let operand = value.index();
                        *value = if Namer::made_up(meanings, fresh, operand) {
                            operands.push(blind.get(operand).copied().unwrap_or(0));
                            SsaVarId::from_index(1)
                        } else {
                            SsaVarId::from_index(operand.saturating_mul(2))
                        };
                    }
                });
                match value {
                    Value::Pure(_, index) => {
                        1_u8.hash(&mut hasher);
                        index.hash(&mut hasher);
                    }
                    _ => 2_u8.hash(&mut hasher),
                }
                shape.hash(&mut hasher);
                operands.hash(&mut hasher);
            }
            _ => {
                3_u8.hash(&mut hasher);
                term.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// What `snapshot` is with every number the run made up left out
    /// ([`Self::blinded`]): two snapshots of one state are sketched alike.
    fn sketch(&self, snapshot: &Snapshot) -> u64 {
        let mut hasher = FxHasher::default();
        snapshot.entered.hash(&mut hasher);
        snapshot.joined.hash(&mut hasher);
        for side in &snapshot.sides {
            side.block.hash(&mut hasher);
            side.at.hash(&mut hasher);
            side.values.len().hash(&mut hasher);
            for (value, term) in &side.values {
                value.hash(&mut hasher);
                self.blind
                    .get(*term)
                    .copied()
                    .unwrap_or(0)
                    .hash(&mut hasher);
            }
            side.forgotten.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// The state the index holds where `first` and `second` stand, the
    /// sides pruned ([`Self::prune`]); where it holds none, it holds `state`
    /// there from now on.
    fn settle(
        &mut self,
        first: &Side,
        second: &Side,
        memory: usize,
        entered: bool,
        joined: bool,
        state: usize,
    ) -> Option<usize> {
        let snapshot = Snapshot {
            sides: [Held::of(first), Held::of(second)],
            memory,
            entered,
            joined,
        };
        let sketch = self.sketch(&snapshot);
        let alike = self.index.get(&sketch).cloned().unwrap_or_default();
        let kept = if alike.is_empty() {
            Kept::Taken(state, snapshot)
        } else {
            let Self {
                meanings,
                fresh,
                names,
                kept,
                ..
            } = self;
            let key = snapshot.named(meanings, fresh, names);
            for place in alike {
                let Some(slot) = kept.get_mut(place) else {
                    continue;
                };
                if let Kept::Taken(held, taken) = slot {
                    *slot = Kept::Named(*held, taken.named(meanings, fresh, names));
                }
                if let Kept::Named(same, named) = slot
                    && *named == key
                {
                    return Some(*same);
                }
            }
            Kept::Named(state, key)
        };
        let place = self.kept.len();
        self.kept.push(kept);
        self.index.entry(sketch).or_default().push(place);
        None
    }

    /// A memory state no other is.
    fn state(&mut self) -> usize {
        self.states = self.states.saturating_add(1);
        self.states
    }

    /// The term `side` reads `value` as, where both sides read it alike.
    fn read(&mut self, side: &Side, value: SsaVarId) -> Option<Term> {
        if let Some(term) = side.values.get(&value) {
            return Some(*term);
        }
        let defined = self.ssa.variable(value)?.def_site().block();
        if side.forgotten.contains(&value)
            || !self
                .dominators
                .dominates(NodeId::new(defined), NodeId::new(self.test))
        {
            return None;
        }
        // Defined on the test's way in, a value is what it was as the test
        // was reached until a side defines it again; where no path from the
        // test reaches its definition, it always is.
        Some(if self.ahead.contains_checked(defined) {
            self.term(Value::Entered(value))
        } else {
            self.term(Value::Before(value))
        })
    }

    /// `op` as `side` reads it: its operands the terms of what it reads, its
    /// definitions blank.
    fn blank(&mut self, side: &Side, op: &SsaOp<T>) -> Option<SsaOp<T>> {
        // One pass of the visitor that writes them: each use is read as it is
        // met, in the order the copy's operands are written.
        let mut blank = op.clone();
        let mut readable = true;
        blank.visit_operands_mut(|role, value| match role {
            OperandRole::Use => match self.read(side, *value) {
                Some(term) => *value = SsaVarId::from_index(term),
                None => readable = false,
            },
            OperandRole::Def | OperandRole::FlagsDef => *value = SsaVarId::from_index(0),
        });
        readable.then_some(blank)
    }

    /// Moves `side` into `to` from the block it is in, its phis taking what
    /// the edge brings.
    fn enter(&mut self, side: &mut Side, to: usize) -> Option<()> {
        let ssa = self.ssa;
        self.read.insert(to);
        let from = side.block;
        let mut taken = Vec::new();
        for phi in ssa.block(to)?.phi_nodes() {
            let value = phi.operand_from(from)?.value();
            taken.push((phi.result(), self.read(side, value)?));
        }
        for (phi, term) in taken {
            side.meet(phi, term);
        }
        side.block = to;
        side.at = 0;
        Some(())
    }

    /// Runs `side` to what it does next, in the memory `memory` numbers:
    /// every pure row and load on the way is its term.
    ///
    /// What it reads and computes is fixed by the side and the memory alone,
    /// so a side rolled back to where it stood ([`Side::rollback`]) steps
    /// again to the same place, holding the same terms.
    fn advance(&mut self, side: &mut Side, memory: usize) -> Option<Step<T>> {
        let ssa = self.ssa;
        self.read.insert(side.block);
        loop {
            let row = ssa.block(side.block)?.instructions().get(side.at)?;
            let op = row.op();
            if row.is_terminator() {
                return match op {
                    SsaOp::Jump { target } => Some(Step::Jump(*target)),
                    SsaOp::Branch {
                        condition,
                        true_target,
                        false_target,
                    } => Some(Step::Branch(
                        Asked::Holds(self.read(side, *condition)?),
                        *true_target,
                        *false_target,
                    )),
                    SsaOp::BranchCmp {
                        left,
                        right,
                        cmp,
                        unsigned,
                        true_target,
                        false_target,
                    } => {
                        let float = [left, right].iter().any(|value| {
                            ssa.variable(**value)
                                .is_some_and(|variable| T::is_floating(variable.var_type()))
                        });
                        Some(Step::Branch(
                            Asked::Compare {
                                left: self.read(side, *left)?,
                                right: self.read(side, *right)?,
                                cmp: *cmp,
                                unsigned: *unsigned,
                                float,
                            },
                            *true_target,
                            *false_target,
                        ))
                    }
                    SsaOp::Return { .. } => Some(Step::Leave(self.blank(side, op)?)),
                    _ => None,
                };
            }
            let load = matches!(op, SsaOp::LoadIndirect { .. }) && !op.is_volatile();
            if !load && !op.effects().is_pure() {
                return Some(Step::Effect(self.blank(side, op)?, op.defs().collect()));
            }
            let blank = self.blank(side, op)?;
            for (index, def) in op.defs().enumerate() {
                let value = if load {
                    Value::Load(blank.clone(), memory)
                } else {
                    Value::Pure(blank.clone(), index)
                };
                let term = self.term(value);
                side.meet(def, term);
            }
            side.at = side.at.saturating_add(1);
        }
    }

    /// Lets `side` keep only what it can still read: what it holds and has
    /// let go of that is defined where it stands ([`Self::defined_at`]),
    /// since anything an arm defined that the way to here need not pass is
    /// one nothing on from here reads, which SSA rules out; and nothing to
    /// roll back.
    fn prune(&self, side: &mut Side) {
        let (block, at) = (side.block, side.at);
        side.values
            .retain(|value, _| self.defined_at(*value, block, at));
        side.forgotten
            .retain(|value| self.defined_at(*value, block, at));
        side.trail.clear();
    }

    /// Whether `value` is defined at row `at` of `block`: before it in the
    /// block, at the block's entry, or in a block that dominates it.
    fn defined_at(&self, value: SsaVarId, block: usize, at: usize) -> bool {
        let Some(variable) = self.ssa.variable(value) else {
            return false;
        };
        let site = variable.def_site();
        if site.block() == block {
            return site.index().is_none_or(|row| row < at);
        }
        self.dominators
            .dominates(NodeId::new(site.block()), NodeId::new(block))
    }

    /// The slot each phi of a pair of joins takes, the sides in this order:
    /// a value fixed however many trips are made stays itself, and any other
    /// is a slot both sides share where they take it alike, numbered as met.
    /// `None` where a phi took nothing.
    fn slots(
        meanings: &[Value<T>],
        first: (&Side, &[SsaVarId]),
        second: (&Side, &[SsaVarId]),
    ) -> Option<(Vec<Slot>, Vec<Slot>)> {
        let mut shared: Vec<Term> = Vec::new();
        let mut slots = |side: &Side, phis: &[SsaVarId]| -> Option<Vec<Slot>> {
            phis.iter()
                .map(|phi| {
                    let term = *side.values.get(phi)?;
                    if matches!(
                        meanings.get(term),
                        Some(Value::Entered(_) | Value::Before(_))
                    ) {
                        return Some(Slot::Fixed(term));
                    }
                    let at = shared
                        .iter()
                        .position(|known| *known == term)
                        .unwrap_or_else(|| {
                            shared.push(term);
                            shared.len().saturating_sub(1)
                        });
                    Some(Slot::Shared(at))
                })
                .collect()
        };
        let mine = slots(first.0, first.1)?;
        let other = slots(second.0, second.1)?;
        Some((mine, other))
    }

    /// Generalizes `first` and `second` as `pairing` pairs their joins: each
    /// phi to its slot, what each held as the test was reached kept, the rest
    /// let go of, in a memory state of their own.
    fn met(&mut self, first: &mut Side, second: &mut Side, pairing: Pairing, memory: &mut usize) {
        let Pairing {
            first: (ours, mine, held, forgotten),
            second: (theirs, other, other_held, other_forgotten),
            ..
        } = pairing;
        let mut generalized =
            |phis: Vec<SsaVarId>, slots: Vec<Slot>, held: Vec<(SsaVarId, Term)>| {
                phis.into_iter()
                    .zip(slots)
                    .map(|(phi, slot)| match slot {
                        Slot::Fixed(term) => (phi, term),
                        Slot::Shared(at) => (phi, self.term(Value::Joined(at))),
                    })
                    .chain(held)
                    .collect::<BTreeMap<SsaVarId, Term>>()
            };
        let first_values = generalized(ours, mine, held);
        let second_values = generalized(theirs, other, other_held);
        first.generalize(first_values, forgotten);
        second.generalize(second_values, other_forgotten);
        *memory = self.state();
    }

    /// How `first` and `second`, just entered a pair of joins, pair them:
    /// `None` where either block has no phi and joins nothing, `Some(None)`
    /// where a phi took nothing, which no pairing reads.
    fn pairing(&self, first: &Side, second: &Side) -> Option<Option<Pairing>> {
        let ssa = self.ssa;
        let phis = |block: usize| -> Vec<SsaVarId> {
            ssa.block(block)
                .map(|data| data.phi_nodes().iter().map(|phi| phi.result()).collect())
                .unwrap_or_default()
        };
        // A block with no phi joins nothing, and keeps what its way in
        // computed: a loop's header has phis, and a way out may read a value
        // the block before it defined.
        let (ours, theirs) = (phis(first.block), phis(second.block));
        if ours.is_empty() || theirs.is_empty() {
            return None;
        }
        let Some((mine, other)) = Self::slots(&self.meanings, (first, &ours), (second, &theirs))
        else {
            return Some(None);
        };
        // The pair is the pair whichever side is read first.
        if Self::slots(&self.meanings, (second, &theirs), (first, &ours)).is_none() {
            return Some(None);
        }
        // What each side holds as the test was reached is that value however
        // many trips are made, where no phi rebinds it, and so is what is one
        // value on every trip. Whatever else a side met is let go of: what it
        // read past here, it reads again. One pass over what the side holds
        // parts the two, a phi found by search among the join's sorted phis.
        let parted = |side: &Side, phis: &[SsaVarId]| {
            let mut sorted = phis.to_vec();
            sorted.sort_unstable();
            let mut held: Vec<(SsaVarId, Term)> = Vec::new();
            let mut forgotten = side.forgotten.clone();
            for (value, term) in &side.values {
                if sorted.binary_search(value).is_ok() {
                    continue;
                }
                if matches!(self.meanings.get(*term), Some(Value::Entered(_)))
                    || self.invariant(*term)
                {
                    held.push((*value, *term));
                } else {
                    forgotten.insert(*value);
                }
            }
            (held, forgotten)
        };
        let ((held, forgotten), (other_held, other_forgotten)) =
            (parted(first, &ours), parted(second, &theirs));
        Some(Some(Pairing {
            first: (ours, mine, held, forgotten),
            second: (theirs, other, other_held, other_forgotten),
        }))
    }
}

/// Pass that drops a test whose outcome changes nothing.
///
/// Within the context's [`Budget::MootSteps`]: a test whose arms, run side by side,
/// make the same effects on the same values and leave alike, loops and all,
/// jumps to the arm with more blocks ahead.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - turns the test into a jump and
/// removes what only the other arm reached.
#[derive(Debug, Default, Clone, Copy)]
pub struct MootTestPass;

impl MootTestPass {
    /// Creates a new moot-test pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for MootTestPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "moot-test"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Drop a test whose outcome changes nothing"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Drops every test whose outcome changes nothing, each jumping to an arm
    /// that goes on out of the innermost loop the test is in, and of two such,
    /// or of none, to the one with more blocks ahead.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// Whether any test was dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if an edit session fails.
    ///
    /// [`Budget::MootSteps`]: crate::analysis::Budget::MootSteps
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let budget = env.context().budget().limit(Budget::MootSteps);
        let interrupt = env.context().interrupt();
        if ssa.has_exception_handlers() {
            return Ok(false);
        }
        let mut steps = budget;
        let mut dropped = false;
        let mut learned = Vec::new();
        // Each test dropped leaves one branch fewer.
        for _ in 0..ssa.block_count() {
            let found = Moot::find(ssa, &mut steps, interrupt, &mut learned);
            for (question, refutation) in learned.drain(..) {
                ssa.refutations_mut().record(question, refutation);
            }
            let Some((test, kept, stands_for)) = found else {
                break;
            };
            ssa.edit(
                SsaEditOptions::new()
                    .with_verify(true)
                    .with_rollback(SsaRollbackPolicy::OnFailure),
                |editor| {
                    editor.fold_terminator_pruning_phis(test, SsaOp::Jump { target: kept })?;
                    // What only the other arm reached goes.
                    let reachable = {
                        let ssa = editor.function();
                        let cfg = SsaCfg::from_ssa(ssa);
                        let mut seen = BitSet::new(ssa.block_count());
                        let mut stack = vec![0];
                        while let Some(block) = stack.pop() {
                            if seen.insert_checked(block) {
                                stack.extend(cfg.successors(NodeId::new(block)).map(NodeId::index));
                            }
                        }
                        seen
                    };
                    // The kept arm's work stands for the copy's that goes.
                    for ((block, at), (gone_block, gone_at)) in &stands_for {
                        if !reachable.contains(*block) || reachable.contains(*gone_block) {
                            continue;
                        }
                        let Some(facts) = editor
                            .function()
                            .block(*gone_block)
                            .and_then(|gone| gone.instructions().get(*gone_at))
                            .map(|gone| gone.facts().clone())
                        else {
                            continue;
                        };
                        editor.absorb_twin_facts(*block, *at, &facts)?;
                    }
                    editor.clear_unreached_blocks(&reachable);
                    Ok(())
                },
            )?;
            env.record(EventKind::ControlFlowRestructured)
                .location(test)
                .message(format!(
                    "moot test: B{test}'s arms do the same, it goes on to B{kept}"
                ));
            dropped = true;
        }
        Ok(dropped)
    }
}

/// The shape of `ssa`'s control flow: every block's successors. The
/// blocks ahead of a test, the joins, the dominators and the reach a
/// search reads are all made from it.
fn shape<T: Target>(ssa: &SsaFunction<T>) -> u128 {
    let mut fingerprint = Fingerprint::new();
    ssa.block_count().hash(&mut fingerprint);
    for block in 0..ssa.block_count() {
        ssa.block(block)
            .map(|data| data.successors())
            .hash(&mut fingerprint);
    }
    fingerprint.finish128()
}

/// The question a search of `test` answers beyond the blocks it reads:
/// which test, which arm is which, in a function of control flow `shape`.
fn question(shape: u128, test: usize, on_true: usize, on_false: usize) -> u128 {
    let mut fingerprint = Fingerprint::new();
    "moot".hash(&mut fingerprint);
    (shape, test, on_true, on_false).hash(&mut fingerprint);
    fingerprint.finish128()
}

/// What a search reads of `blocks`: each block's phis and rows, every
/// value by where it is defined ([`Site`]), and whether each value a
/// comparison branches on may be a float.
///
/// A value is no more to a search than where it is defined: a value
/// defined in a block it reads is the row or phi there, read with the
/// rest of the block; one defined elsewhere is an opaque value of the
/// block that defines it, which only the control flow's shape relates to
/// the test ([`shape`]), and is told from another such value by the
/// order the search's blocks first name it in. So a value renamed, or the
/// rows around a definition the search never runs rewritten, leave what
/// the search reads as it was.
fn content<T: Target>(ssa: &SsaFunction<T>, blocks: &[usize]) -> u128 {
    let mut fingerprint = Fingerprint::new();
    let mut outside: FxHashMap<SsaVarId, usize> = FxHashMap::default();
    let mut site = |value: SsaVarId| -> Site {
        let Some(def) = ssa.variable(value).map(|variable| variable.def_site()) else {
            return Site::Unknown;
        };
        if blocks.binary_search(&def.block()).is_ok() {
            let block = ssa.block(def.block());
            let place = match def.index() {
                Some(row) => block
                    .and_then(|data| data.instructions().get(row))
                    .and_then(|data| data.op().defs().position(|defined| defined == value)),
                None => block.and_then(|data| {
                    data.phi_nodes()
                        .iter()
                        .position(|phi| phi.result() == value)
                }),
            };
            if let Some(place) = place {
                return Site::Read(def.block(), def.index(), place);
            }
        }
        let next = outside.len();
        Site::Outside(def.block(), *outside.entry(value).or_insert(next))
    };
    for &index in blocks {
        index.hash(&mut fingerprint);
        let Some(block) = ssa.block(index) else {
            usize::MAX.hash(&mut fingerprint);
            continue;
        };
        block.phi_nodes().len().hash(&mut fingerprint);
        for phi in block.phi_nodes() {
            site(phi.result()).hash(&mut fingerprint);
            phi.operands().len().hash(&mut fingerprint);
            for operand in phi.operands() {
                operand.edge().hash(&mut fingerprint);
                site(operand.value()).hash(&mut fingerprint);
            }
        }
        block.instructions().len().hash(&mut fingerprint);
        for row in block.instructions() {
            let op = row.op();
            // The operation with its operands blank, then each operand by
            // its role and where it is defined.
            let mut blank = op.clone();
            blank.visit_operands_mut(|_, value| *value = SsaVarId::from_index(0));
            blank.hash(&mut fingerprint);
            op.visit_operands(|role, value| {
                role.hash(&mut fingerprint);
                site(value).hash(&mut fingerprint);
            });
            if let SsaOp::BranchCmp { left, right, .. } = op {
                for value in [left, right] {
                    ssa.variable(*value)
                        .is_some_and(|variable| T::is_floating(variable.var_type()))
                        .hash(&mut fingerprint);
                }
            }
        }
    }
    fingerprint.finish128()
}

/// Whether `from` runs on into `into` through blocks that compute values
/// alone and jump.
fn runs_into<T: Target>(ssa: &SsaFunction<T>, from: usize, into: usize) -> bool {
    let mut at = from;
    for _ in 0..ssa.block_count() {
        if at == into {
            return true;
        }
        let Some(block) = ssa.block(at) else {
            return false;
        };
        let Some((last, rows)) = block.instructions().split_last() else {
            return false;
        };
        let SsaOp::Jump { target } = last.op() else {
            return false;
        };
        if !block.phi_nodes().is_empty() || !rows.iter().all(|row| row.op().effects().is_pure()) {
            return false;
        }
        at = *target;
    }
    false
}

/// The blocks of `cfg` more than one way enters, the function's own entry
/// to block 0 counted as one: every cycle of blocks passes through one,
/// so a run that goes round comes to a state it is looked up by
/// ([`Moot::merging`]).
fn merges<T: Target>(cfg: &SsaCfg<'_, T>) -> BitSet {
    let count = cfg.block_count();
    let mut merges = BitSet::new(count);
    for block in 0..count {
        let entry = usize::from(block == 0);
        if cfg
            .predecessors(NodeId::new(block))
            .count()
            .saturating_add(entry)
            > 1
        {
            merges.insert_checked(block);
        }
    }
    merges
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::{
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MockTarget, MockType},
    };

    /// A test whose one arm passes through `diamonds` tests of its own, each
    /// an if and an else that only jump on to the next, and whose other arm
    /// goes straight on; both return the same value, so the test decides
    /// nothing.
    fn diamonds(diamonds: usize) -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 2);
        let mut value = |index: u16| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                1,
                DefSite::instruction(0, usize::from(index)),
                MockType::I32,
            )
        };
        let (five, zero) = (value(0), value(1));
        let compare = |true_target: usize, false_target: usize| {
            SsaInstruction::synthetic(SsaOp::BranchCmp {
                left: five,
                right: zero,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target,
                false_target,
            })
        };
        // Block 0 tests, block 1 is the straight arm, then each diamond is a
        // head and its two sides, and the last head returns.
        let head = |index: usize| 2 + index * 3;
        let mut test = SsaBlock::new(0);
        for (dest, number) in [(five, 5), (zero, 0)] {
            test.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(number),
            }));
        }
        test.add_instruction(compare(head(0), 1));
        ssa.add_block(test);
        let mut straight = SsaBlock::new(1);
        straight.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(five))));
        ssa.add_block(straight);
        for index in 0..diamonds {
            let mut block = SsaBlock::new(head(index));
            block.add_instruction(compare(head(index) + 1, head(index) + 2));
            ssa.add_block(block);
            for side in 1..=2 {
                let mut block = SsaBlock::new(head(index) + side);
                block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump {
                    target: head(index + 1),
                }));
                ssa.add_block(block);
            }
        }
        let mut last = SsaBlock::new(head(diamonds));
        last.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(five))));
        ssa.add_block(last);
        ssa.recompute_uses();
        ssa
    }

    /// A test whose arms each pass through `loops` loops of their own before
    /// both return the same value, one arm's asking whether five is below one
    /// and the other's whether zero is above one, each loop's head going round
    /// through a block that jumps back or on to the next. No test of one arm
    /// asks what one of the other's asks, so each side branches alone, in
    /// whichever order. Blocks: 0 tests, the first arm's loops from 1, three
    /// blocks each, then its return, and the second arm's likewise after it.
    fn lone_loops(loops: usize) -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 3);
        let mut value = |index: u16| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                1,
                DefSite::instruction(0, usize::from(index)),
                MockType::I32,
            )
        };
        let (five, zero, one) = (value(0), value(1), value(2));
        let compare = |left: SsaVarId, cmp: CmpKind, true_target: usize, false_target: usize| {
            SsaInstruction::synthetic(SsaOp::BranchCmp {
                left,
                right: one,
                cmp,
                unsigned: false,
                true_target,
                false_target,
            })
        };
        let first = |index: usize| 1 + index * 3;
        let second = |index: usize| first(loops) + 1 + index * 3;
        let mut test = SsaBlock::new(0);
        for (dest, number) in [(five, 5), (zero, 0), (one, 1)] {
            test.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(number),
            }));
        }
        test.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
            left: five,
            right: zero,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: first(0),
            false_target: second(0),
        }));
        ssa.add_block(test);
        for (head, left, cmp) in [
            (&first as &dyn Fn(usize) -> usize, five, CmpKind::Lt),
            (&second, zero, CmpKind::Gt),
        ] {
            for index in 0..loops {
                let mut block = SsaBlock::new(head(index));
                block.add_instruction(compare(left, cmp, head(index) + 1, head(index) + 2));
                ssa.add_block(block);
                for (side, target) in [(1, head(index)), (2, head(index + 1))] {
                    let mut block = SsaBlock::new(head(index) + side);
                    block.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target }));
                    ssa.add_block(block);
                }
            }
            let mut last = SsaBlock::new(head(loops));
            last.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(five))));
            ssa.add_block(last);
        }
        ssa.recompute_uses();
        ssa
    }

    /// A test whose one arm goes round a loop that only tests, as often as
    /// it likes, before it returns, and whose other arm returns at once with
    /// the same value: one side may go round for ever while the other has
    /// returned, so the test decides something.
    fn a_loop_against_a_return() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 2);
        let mut value = |index: u16| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                1,
                DefSite::instruction(0, usize::from(index)),
                MockType::I32,
            )
        };
        let (five, zero) = (value(0), value(1));
        let compare = |true_target: usize, false_target: usize| {
            SsaInstruction::synthetic(SsaOp::BranchCmp {
                left: five,
                right: zero,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target,
                false_target,
            })
        };
        let returns = || SsaInstruction::synthetic(SsaOp::returns(Some(five)));
        // 0 tests; 1 returns; 2 is the loop's head, 3 its body going back,
        // 4 its way out.
        let mut test = SsaBlock::new(0);
        for (dest, number) in [(five, 5), (zero, 0)] {
            test.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
                dest,
                value: ConstValue::I32(number),
            }));
        }
        test.add_instruction(compare(2, 1));
        ssa.add_block(test);
        let mut straight = SsaBlock::new(1);
        straight.add_instruction(returns());
        ssa.add_block(straight);
        let mut head = SsaBlock::new(2);
        head.add_instruction(compare(3, 4));
        ssa.add_block(head);
        let mut body = SsaBlock::new(3);
        body.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 2 }));
        ssa.add_block(body);
        let mut out = SsaBlock::new(4);
        out.add_instruction(returns());
        ssa.add_block(out);
        ssa.recompute_uses();
        ssa
    }

    /// **Every cycle passes a block more than one way enters**: the entry
    /// block heading a loop has one edge in, the loop's, and the function's
    /// own entry is the other, so it is one; a block entered from one other
    /// is not.
    #[test]
    fn the_entry_heading_a_loop_is_entered_two_ways() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let value = ssa.create_variable(
            VariableOrigin::Local(0),
            1,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        // 0 tests and is the loop's head; 1 goes back to it; 2 returns.
        let mut head = SsaBlock::new(0);
        head.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: value,
            value: ConstValue::I32(0),
        }));
        head.add_instruction(SsaInstruction::synthetic(SsaOp::BranchCmp {
            left: value,
            right: value,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(head);
        let mut back = SsaBlock::new(1);
        back.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 0 }));
        ssa.add_block(back);
        let mut out = SsaBlock::new(2);
        out.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(value))));
        ssa.add_block(out);
        ssa.recompute_uses();
        let merges = merges(&SsaCfg::from_ssa(&ssa));
        assert!(merges.contains_checked(0), "the entry heads the loop");
        assert!(!merges.contains_checked(1), "entered from the head alone");
        assert!(!merges.contains_checked(2), "entered from the head alone");
    }

    /// **A side that may go round for ever is no proof**, explored
    /// configurations or not: the wait the loop comes back to stays part of
    /// where the run stands, so going round is never taken for alike.
    #[test]
    fn a_side_that_may_go_round_for_ever_is_no_proof() {
        let (_, found) = search_steps(&a_loop_against_a_return());
        assert!(!found, "the loop may never return");
    }

    /// **A state is found again whatever the run made up on the way to it,
    /// and only then.** A side holding the sum of an effect's result with
    /// itself stands where it stood with another effect's; a side holding the
    /// sum of two effects is sketched alike, and told apart by its name; and
    /// that sum with its effects met the other way round is found again.
    #[test]
    fn a_state_is_found_again_whatever_was_made_up_on_the_way() {
        let ssa = lone_loops(2);
        let cfg = SsaCfg::from_ssa(&ssa);
        let merges = merges(&cfg);
        let dominators = compute_dominators(&cfg, NodeId::new(0));
        let ahead = BitSet::new(ssa.block_count());
        let mut search = Moot::new(&ssa, &dominators, &merges, 0, ahead, usize::MAX);
        let add = |left: Term, right: Term| {
            Value::<MockTarget>::Pure(
                SsaOp::Add {
                    dest: SsaVarId::from_index(0),
                    left: SsaVarId::from_index(left),
                    right: SsaVarId::from_index(right),
                    flags: None,
                },
                0,
            )
        };
        let (one, other) = (
            search.term(Value::Effect(0, 0)),
            search.term(Value::Effect(1, 0)),
        );
        let doubled = search.term(add(one, one));
        let doubled_other = search.term(add(other, other));
        let both = search.term(add(one, other));
        let both_swapped = search.term(add(other, one));
        let holding = |term: Term| {
            let mut side = Side::new(1);
            side.values.insert(SsaVarId::from_index(5), term);
            side
        };
        let mut settle = |term: Term, memory: usize, state: usize| {
            search.settle(&holding(term), &Side::new(8), memory, true, true, state)
        };
        assert_eq!(settle(doubled, 0, 10), None, "the first state");
        assert_eq!(
            settle(doubled_other, 3, 11),
            Some(10),
            "the same sum of another effect, in another memory"
        );
        assert_eq!(settle(both, 0, 12), None, "a sum of two effects");
        assert_eq!(
            settle(both_swapped, 0, 13),
            Some(12),
            "that sum, its effects met the other way round"
        );
        assert_eq!(settle(both, 0, 14), Some(12), "that sum again");
    }

    /// **Ways to a state that differ only in their waits come to one state**:
    /// one side's lone branch taken before the other's, or after it, leaves
    /// different waits and the same sides, and the sides go on alike from
    /// both, so the second way comes to the state the first made.
    #[test]
    fn ways_that_differ_only_in_their_waits_come_to_one_state() {
        let ssa = lone_loops(2);
        let cfg = SsaCfg::from_ssa(&ssa);
        let merges = merges(&cfg);
        let dominators = compute_dominators(&cfg, NodeId::new(0));
        let ahead = BitSet::new(ssa.block_count());
        let mut search = Moot::new(&ssa, &dominators, &merges, 0, ahead, usize::MAX);
        search.nodes = vec![Node::end(Moves::Lost), Node::end(Moves::Won)];
        // The sides at the heads of their first loops, block 1 and block 8,
        // each having entered its own head alone while the other stood at its.
        let (first_head, second_head) = (1, 8);
        let one_order = Waits::from([(true, first_head, second_head, 0)]);
        let other_order = Waits::from([(false, second_head, first_head, 0)]);
        let one = search.node(
            Side::new(first_head),
            Side::new(second_head),
            0,
            true,
            true,
            one_order,
        );
        let other = search.node(
            Side::new(first_head),
            Side::new(second_head),
            0,
            true,
            true,
            other_order,
        );
        assert_eq!(one, other);
        assert_eq!(search.nodes.len(), 3);
    }

    /// The steps a search of `ssa` takes, with as many as it wants, whether
    /// it found a test to drop, and the refutations it learned.
    fn search_learning(ssa: &SsaFunction<MockTarget>) -> (usize, bool, Vec<(u128, Refutation)>) {
        let mut steps = usize::MAX;
        let mut learned = Vec::new();
        let found = Moot::find(ssa, &mut steps, &NeverInterrupted, &mut learned).is_some();
        (usize::MAX - steps, found, learned)
    }

    /// **A refuted test is searched again only once what its search read
    /// changes**: the function remembers the refutation, a second search of the
    /// same code runs none and spends the steps they took, and a block a search
    /// read reading differently has that search, and only that one, run again.
    #[test]
    fn a_refuted_test_is_searched_again_only_once_what_it_read_changes() {
        let mut ssa = a_loop_against_a_return();
        // Both tests, the function's and the loop's, are refuted.
        let (spent, found, learned) = search_learning(&ssa);
        assert!(!found);
        assert!(spent > 0);
        assert_eq!(learned.len(), 2);
        // The loop's way out, block 4, is about to change.
        let out = 4;
        let mut reading_out: Vec<u128> = learned
            .iter()
            .filter(|(_, refutation)| refutation.read().contains(&out))
            .map(|(question, _)| *question)
            .collect();
        assert!(!reading_out.is_empty());
        for (question, refutation) in learned {
            ssa.refutations_mut().record(question, refutation);
        }

        // Reused, the refutations spend what their searches spent, so a
        // search after them has the steps it would have had.
        let (again, found, learned) = search_learning(&ssa);
        assert!(!found);
        assert!(
            learned.is_empty(),
            "an unchanged refutation is searched again"
        );
        assert_eq!(again, spent);

        // The way out returns the other value now.
        let zero = SsaVarId::from_index(1);
        let row = ssa
            .block_mut(out)
            .and_then(|block| block.instructions_mut().first_mut())
            .expect("the loop's way out");
        row.set_op(SsaOp::returns(Some(zero)));
        ssa.recompute_uses();
        let (changed, found, learned) = search_learning(&ssa);
        assert!(!found);
        assert!(
            changed > 0,
            "a refutation outlived the block its search read"
        );
        let mut searched: Vec<u128> = learned.iter().map(|(question, _)| *question).collect();
        searched.sort_unstable();
        reading_out.sort_unstable();
        assert_eq!(
            searched, reading_out,
            "only the searches that read the way out run again"
        );
    }

    /// The search of `ssa` with `steps` to take: the steps left, whether it
    /// found a test to drop, and what it learned.
    fn search_with(
        ssa: &SsaFunction<MockTarget>,
        steps: usize,
    ) -> (usize, bool, Vec<(u128, Refutation)>) {
        let mut left = steps;
        let mut learned = Vec::new();
        let found = Moot::find(ssa, &mut left, &NeverInterrupted, &mut learned).is_some();
        (left, found, learned)
    }

    /// **A search that ran out runs out again with no more steps**: the
    /// function remembers the steps it had, a search of the same code with as
    /// many or fewer ends as running out ends it without searching, and one
    /// with more steps searches again.
    #[test]
    fn a_search_that_ran_out_is_searched_again_only_with_more_steps() {
        let mut ssa = diamonds(8);
        let (whole, found, _) = search_with(&ssa, usize::MAX);
        assert!(found, "with every step it wants the test is moot");
        let needed = usize::MAX - whole;
        let short = needed / 2;
        let (left, found, learned) = search_with(&ssa, short);
        assert_eq!((left, found), (0, false));
        assert_eq!(learned.len(), 1);
        assert_eq!(learned[0].1.ended(), Unsettled::RanOut(short));
        for (question, refutation) in learned {
            ssa.refutations_mut().record(question, refutation);
        }

        for fewer in [short, short - 1] {
            let (left, found, learned) = search_with(&ssa, fewer);
            assert_eq!((left, found), (0, false));
            assert!(
                learned.is_empty(),
                "a search that ran out was run again with {fewer} steps"
            );
        }
        let (_, found, _) = search_with(&ssa, needed);
        assert!(
            found,
            "with the steps it needs the test is searched again, and moot"
        );
    }

    /// The steps a search of `ssa` takes, with as many as it wants.
    fn search_steps(ssa: &SsaFunction<MockTarget>) -> (usize, bool) {
        let mut steps = usize::MAX;
        let found = Moot::find(ssa, &mut steps, &NeverInterrupted, &mut Vec::new()).is_some();
        (usize::MAX - steps, found)
    }

    /// **A search grows with the code, not with its paths**: each test one
    /// side makes alone doubles the ways through, and the search proving the
    /// arms alike walks code, not ways.
    #[test]
    fn a_search_grows_with_the_code_not_its_paths() {
        let measured: Vec<(usize, usize, bool)> = (2..=14)
            .map(|count| {
                let (steps, found) = search_steps(&diamonds(count));
                (count, steps, found)
            })
            .collect();
        eprintln!("{measured:?}");
        for (count, steps, found) in &measured {
            assert!(found, "{count} diamonds: the arms do the same");
            assert!(*steps <= 16 * count, "{count} diamonds took {steps} steps");
        }
    }

    /// A side holding `values`, at row `at` of block 0.
    fn side(at: usize, values: &[(usize, Term)]) -> Side {
        let mut side = Side::new(0);
        for (value, term) in values {
            side.meet(SsaVarId::from_index(*value), *term);
        }
        side.trail.clear();
        side.at = at;
        side
    }

    /// **A side rolled back stands as it stood**: a value it met anew is
    /// gone, one it met again holds its old term, a join's values and the
    /// values it let go of are the old ones, the block and row are the old
    /// ones, and a mark taken later and rolled back first leaves the earlier
    /// changes in place.
    #[test]
    fn a_side_rolled_back_stands_as_it_stood() {
        let stood = side(2, &[(1, 10), (2, 20)]);
        let mut moved = stood.clone();
        let was = moved.mark();
        moved.meet(SsaVarId::from_index(3), 30);
        moved.meet(SsaVarId::from_index(2), 21);
        let later = moved.mark();
        moved.meet(SsaVarId::from_index(3), 31);
        moved.generalize(
            [(SsaVarId::from_index(7), 70)].into_iter().collect(),
            [SsaVarId::from_index(1)].into_iter().collect(),
        );
        moved.block = 4;
        moved.at = 5;
        moved.rollback(later);
        assert_eq!(
            moved.values.get(&SsaVarId::from_index(3)),
            Some(&30),
            "what came before the later mark stays"
        );
        assert_eq!(moved.values.get(&SsaVarId::from_index(2)), Some(&21));
        assert!(moved.forgotten.is_empty(), "the join's let-go values go");
        assert_eq!((moved.block, moved.at), (0, 2));
        moved.rollback(was);
        assert_eq!(moved.values, stood.values);
        assert_eq!(moved.forgotten, stood.forgotten);
        assert_eq!((moved.block, moved.at), (stood.block, stood.at));
        assert!(moved.trail.is_empty(), "nothing is left to roll back");
    }

    /// **States name made-up terms by what they are, not by which they are.**
    /// Two sums of an effect's result with itself, of different effects, name
    /// alike in states of their own; a sum of the two effects names apart;
    /// and sums that differ in an operand the run did not make up name apart,
    /// however often a search's states have named their forms before.
    #[test]
    fn states_name_made_up_terms_by_what_they_are() {
        let add = |left: usize, right: usize| {
            Value::<MockTarget>::Pure(
                SsaOp::Add {
                    dest: SsaVarId::from_index(0),
                    left: SsaVarId::from_index(left),
                    right: SsaVarId::from_index(right),
                    flags: None,
                },
                0,
            )
        };
        let meanings = vec![
            Value::Effect(0, 0),
            Value::Effect(1, 0),
            add(0, 0),
            add(1, 1),
            add(0, 1),
            Value::Before(SsaVarId::from_index(40)),
            Value::Before(SsaVarId::from_index(41)),
            add(0, 5),
            add(0, 6),
        ];
        let fresh = vec![true, true, true, true, true, false, false, true, true];
        let mut names = Names::default();
        let mut made = |term: usize| {
            let mut namer = Namer::new(&meanings, &fresh, &mut names, 0);
            let name = namer.name(term);
            (name, namer.made)
        };
        for _ in 0..2 {
            let own = made(2);
            assert_eq!(own, made(3), "a sum of either effect with itself");
            assert_eq!(own.1.len(), 2, "the effect, then the sum");
            assert_ne!(own.1, made(4).1, "a sum of the two effects");
            assert_ne!(made(7).1, made(8).1, "sums of different given values");
            assert_eq!(made(7), made(7), "one sum, named again");
        }
    }

    /// **A state's places are its own, however many states came before.** A
    /// place set while one state is named is gone once the next is started,
    /// also when the stamps run out and start again.
    #[test]
    fn a_states_places_are_its_own() {
        let mut names = Names::<MockTarget>::default();
        names.start();
        names.set(3, 7);
        assert_eq!(names.place(3), Some(7));
        assert_eq!(names.place(2), None, "a term never placed");
        names.start();
        assert_eq!(names.place(3), None, "placed in the state before");
        names.stamp = u32::MAX.saturating_sub(1);
        names.start();
        names.set(3, 9);
        assert_eq!(names.place(3), Some(9));
        names.start();
        assert_eq!(names.stamp, 1, "the stamps start again");
        assert_eq!(names.place(3), None, "placed before the stamps ran out");
        names.set(4, 2);
        assert_eq!(names.place(4), Some(2));
    }

    /// **A long chain of made-up values is named without recursion**: each a
    /// pure operation over the one before, down to an effect's definition,
    /// as straight-line code computing from a call's result makes them. The
    /// naming walks the chain on a stack of its own, so a thread with little
    /// stack names it, operands first.
    #[test]
    fn a_long_chain_of_made_up_values_is_named_without_recursion() {
        const CHAIN: usize = 100_000;
        let mut meanings: Vec<Value<MockTarget>> = vec![Value::Effect(0, 0)];
        for before in 0..CHAIN.saturating_sub(1) {
            let operand = SsaVarId::from_index(before);
            meanings.push(Value::Pure(
                SsaOp::Add {
                    dest: SsaVarId::from_index(0),
                    left: operand,
                    right: operand,
                    flags: None,
                },
                0,
            ));
        }
        let fresh = vec![true; CHAIN];
        let named = thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let mut names = Names::default();
                let mut namer = Namer::new(&meanings, &fresh, &mut names, 0);
                let name = namer.name(CHAIN.saturating_sub(1));
                (name, namer.made.len())
            })
            .map(thread::JoinHandle::join);
        assert!(
            matches!(named, Ok(Ok((name, made))) if made == CHAIN
                && name == CHAIN.saturating_sub(1).saturating_mul(2).saturating_add(1)),
            "every value named, the last one last"
        );
    }
}
