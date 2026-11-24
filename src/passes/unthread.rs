//! Merging back a block a compiler duplicated into the block it copies.
//!
//! A compiler that threads a jump past a test, or rotates a loop so it tests
//! at its foot, copies the test - and what the test reads - into the block
//! ahead of it. `rmaliases` in dash reads `*app` and tests the flag of what it
//! points at, once ahead of the loop and once in it: two blocks that compute
//! the same thing and branch to the same two blocks. Recovered as written,
//! the loop is entered at two blocks and reads with a copied body and a
//! goto; the source has one loop.
//!
//! A block entered from outside a loop, whose instructions end in a copy of
//! those of a block on the loop, with the same branch to the same blocks, is
//! that block entered by one more edge. Blocks alike in straight-line code,
//! or entered from the loop itself, are not: the source writes those twice
//! itself, as `switch` arms in a loop that each open a file do. The copy
//! goes: the block keeps what it computes ahead of the copy and jumps to the
//! original, whose phis take, along the new edge, the values the copy
//! computed with. The source's one loop is entered at one block again.
//!
//! A compiler that threads a jump copies a test and sends the copy past the
//! next one where the path has answered it. `setsignal` in dash dispatches on
//! `action` with `== 2` and then `!= 3`; the path that has just found
//! `action != 3` copies the `== 2` and goes straight to where `!= 3` leads.
//! The copy branches as the original does but for that block, which computes
//! nothing but values and decides, by what the copy is entered knowing
//! ([`GuardFacts::answer`]), to go on where the copy goes: asked of the
//! copy's values, each of its phis read as what it takes from the original,
//! and a phi of a block above the copy read as the operand the copy's way
//! takes ([`GuardFacts::phi_on_way`]). `cdcmd` in dash copies `p =
//! strtok(cdcomppath, "/")` from the join ahead of its loop into each arm
//! reaching it, and sends each copy's `p == 0` past the loop's trailing `if
//! (new > lim)`, which a compare above the arm decided. Such a copy is a
//! compiler's wherever it is: the source writes no jump past a test it has
//! just passed. The copy jumps to the original, and the dispatch is entered
//! at one block. The original never dominates the copy, so nothing the
//! copy's way in proves is of a value the original computes: such a value is
//! read only where the original dominates. A copy asking the original's
//! question the other way round is one where it copies a computation too, as
//! the call: a test alone asking it the other way is another test.
//!
//! A compiler also threads an edge without copying anything: `trapcmd` in
//! dash writes `if (action[0] == '-' && action[1] == '\0') ... else if
//! (*action) ...`, and the path that found `action[0] == '-'` and
//! `action[1] != 0` goes straight into the `if (*action)` arm: `*action` is
//! `'-'` there. Such an edge goes back through the test it skipped: the test
//! computes nothing but values, the edge's way in decides it through a value
//! proved a number without asking it ([`GuardFacts::asks`]), the block both
//! go to takes the same values from each, and the test neither dominates the
//! edge nor lies on a loop the edge is not on. The arm is entered from its
//! test alone, as the source's `else if` enters it: an arm entered by the two
//! alone, for a block more ways reach - the end of a `switch` each case
//! breaks to, past a compare tree whose tests every case has decided - is a
//! join, and its edges are `break`s.
//!
//! Past a loop, the ways out that skip the test after it go through it
//! together: `expmeta` in dash breaks out of its scan with `if (metaflag)
//! break;` and leaves at its latch past `metaflag = 1`, and the compiler sends
//! both straight into the arm of `out:`'s `if (metaflag == 0)`. Each way out
//! decides the test, by its own branch or by the version of the value the
//! test reads that its way on to the latch or header keeps; the test joins
//! those versions, and its arm is entered from it alone
//! ([`SsaEditor::route_through`](crate::ir::function::SsaEditor::route_through)).
//! Jump threading leaves such an edge going through the test
//! ([`crate::passes::threading`]).
//!
//! A compiler that rotates a loop to test at its foot copies the test ahead
//! of it, asked of the values the loop starts with: `expmeta` in dash asks
//! `if (*name == 0)` of the name's first byte, and its loop goes round on
//! `c != 0` of the character widened. Entered from outside only through a
//! block that jumps to its header, from one block branching there and past
//! it, the loop is entered at its test instead, whose phis take the values
//! the loop starts with
//! ([`SsaEditor::unrotate`](crate::ir::function::SsaEditor::unrotate)): the
//! source's `for` tests first. Its way out with nothing done goes through
//! what the loop's way out goes through, which those values decide, as they
//! decide `expmeta`'s `if (metaflag == 0)` with `metaflag` `0`.
//!
//! A compiler that rotates a loop writes its test at the end of each way
//! round: `expmeta` in dash goes on through its character dispatch's arms,
//! each ending `c = p[k]; if (c) continue;`, where the source tests `*p` once.
//! Two or more blocks inside a loop that branch on to its header, or to a
//! block that only jumps there, and out to one block, asking one question
//! ([`Question::of_terminator`]: the relation and the constants it compares
//! with) of different values, become one test of their own
//! ([`SsaEditor::merge_tests`](crate::ir::function::SsaEditor::merge_tests)),
//! reading a phi of what each read. A branch on a value whose definition asks
//! nothing is never one of them: two such are a `break` on one condition and
//! on another.
//!
//! A compiler also copies a test into each way into a join: `binop2` in
//! dash computes `a = noeval ? b : do_binop(op, a, b)` and then tests the
//! next operator, and each of the three ways to the join tests `noeval`,
//! two of them sending `noeval != 0` straight past the operator test their
//! way has answered. Two or more blocks entered into a join that ask one
//! question of one value, the join on the same answer, become one test of
//! their own ([`SsaEditor::merge_tests`](crate::ir::function::SsaEditor::merge_tests))
//! where each copy's other answer goes on: to the test the join goes on to,
//! through a block computing alike what its way into the join computes
//! ([`SsaEditor::merge_forwarders`](crate::ir::function::SsaEditor::merge_forwarders)),
//! or to where that test, entered from the copy, would go: decided by what
//! the copy's edge knows, through blocks computing values and loads that
//! carry into the arm what the copy carried
//! ([`SsaEditor::send_onward`](crate::ir::function::SsaEditor::send_onward)).
//! A load on that way reads a cell a load above the copy read, in the state
//! the copy leaves memory in, so it faults nowhere the path did not.
//!
//! # What makes two copies one
//!
//! Each instruction of the copy is the original's under one mapping of the
//! original's values to the copy's: what the original defines to what the copy
//! defines, each of the original's phis to the value the copy reads in its
//! place, and every other value to itself - or to a constant alike, or to a
//! load the copy reads of the same cell in the same state as the original's
//! ([`MemorySsa::version_before`]), the same value read twice. The blocks
//! both lead to take, from the copy, what they take from the original - or
//! from the block it goes through - under the same mapping, and nothing past
//! the copy reads what it defines, though
//! the copy may dominate the loop. The values the original's phis take along
//! the new edge are defined ahead of the copy, or before the block.
//!
//! The function's values all keep their meaning, and what the copy defined
//! goes with it: nothing else read it.

use std::{
    cell::{Cell, OnceCell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use rustc_hash::FxHashMap;

use crate::{
    Result,
    analysis::{
        cfg::SsaCfg,
        context::AnalysisContext,
        guards::{Definitions, GuardFacts, Question},
        memory::{MemoryOp, MemorySsa},
    },
    events::EventKind,
    graph::{
        GraphBase, NodeId, Predecessors, Successors,
        algorithms::{DominatorTree, compute_dominators, strongly_connected_components},
    },
    ir::{
        block::SsaBlock,
        function::{SsaEditOptions, SsaFunction, SsaRollbackPolicy},
        ops::{FlagCondition, OperandRole, SsaOp, kinds::CmpKind},
        phi::PhiOperand,
        value::ConstValue,
        variable::SsaVarId,
    },
    passes::utils::resolve_chain,
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// A test a compiler copied ahead of a loop it rotated to test at its foot
/// ([`Copies::rotated_guard`]).
struct Unrotated {
    /// The block ending in the copy.
    guard: usize,
    /// The block the guard enters the loop through.
    forwarder: usize,
    /// The loop's test.
    latch: usize,
    /// The block the test goes on to, which the forwarder entered.
    header: usize,
    /// The value each phi of `latch` takes from `forwarder`, in order.
    entering: Vec<SsaVarId>,
}

/// An edge a compiler threaded past a test: `from` branched straight to
/// `to`, where `through` would have sent it.
struct Reroute {
    /// The block whose edge moves.
    from: usize,
    /// The block it went to.
    to: usize,
    /// The test it skipped.
    through: usize,
}

/// The edges into one block that go through the test they skipped, and
/// what the test joins for them ([`Copies::reroutes`]).
struct Rerouted {
    /// The blocks whose edges move.
    froms: Vec<usize>,
    /// The phis of the block the edges went to that the test joins.
    joined: Vec<usize>,
    /// Each value the test reads or passes on that it joins, with each
    /// edge's version of it, in the order of `froms`.
    rejoined: Vec<(SsaVarId, Vec<SsaVarId>)>,
    /// The block that only jumps on, holding at most constants, by which the
    /// test reaches the block the edges went to, where it does not go there
    /// itself ([`Copies::arm_into`]).
    via: Option<usize>,
}

/// How one edge skips a test ([`Copies::skips`]).
struct Skipped {
    /// The phis of the block the edge went to that the test joins for it.
    joined: Vec<usize>,
    /// The versions the edge holds of the values the test reads
    /// ([`Copies::versions`]).
    versions: BTreeMap<SsaVarId, SsaVarId>,
}

/// One edit the search found, in the view it was found in.
enum Step<T: Target> {
    /// A block that ends in a copy of another ([`Copies::find`]).
    Merge(Found),
    /// Tests a loop goes on through alike ([`Copies::continuing_tests`]).
    Tests(Vec<usize>),
    /// The copies of one test into a join ([`Copies::joined_copies`]).
    Joined(JoinedCopies<T>),
    /// A test copied ahead of a loop rotated to test at its foot
    /// ([`Copies::rotated_guard`]).
    Unrotate(Unrotated),
    /// Edges threaded past a test, with what the test joins for them
    /// ([`Copies::reroutes`]).
    Reroute {
        /// The block they went to.
        to: usize,
        /// The test they skipped.
        through: usize,
        /// The blocks, phis and values that move and join.
        rerouted: Rerouted,
    },
}

/// The kinds of edit, searched for in this order: each until it finds no
/// more, or until its bound. An un-rotation leaves one branch where there
/// were two, so there are at most as many as blocks.
#[derive(Clone, Copy)]
enum Phase {
    /// Blocks that end in a copy of another.
    Copies,
    /// Tests a loop goes on through alike.
    ContinuingTests,
    /// Tests alike that pick between two blocks a loop is entered at.
    EnteringTests,
    /// Copies of one test into a join.
    JoinedCopies,
    /// Edges threaded past a test.
    Reroutes,
    /// Tests copied ahead of a rotated loop.
    Guards,
    /// Nothing more to search for.
    Done,
}

/// Where one run of the pass stands: the kind of edit it searches for, how
/// many more of that kind it may make, and the threaded edges it goes back
/// through.
///
/// Each kind is bounded by what every edit of it removes. A merge replaces a
/// copy of at least two instructions by one jump, so there are at most as
/// many merges as instructions. A merge of tests leaves one branch where
/// there were two or more, so there are at most as many as blocks. Each
/// threaded edge is checked once, as the function stands when its turn
/// comes; an edge the moves make threaded is the next run's, which the
/// pipeline repeats to its fixpoint, within its rounds.
struct Search {
    /// The kind of edit searched for.
    phase: Phase,
    /// How many more edits of that kind may be made.
    left: usize,
    /// The threaded edges, listed once, as the function stood when the
    /// search for them began.
    pending: Option<Vec<Reroute>>,
    /// The next of [`Self::pending`] to check.
    next: usize,
}

impl Search {
    /// A search over `ssa` that begins with its copies.
    fn of<T: Target>(ssa: &SsaFunction<T>) -> Self {
        Self {
            phase: Phase::Copies,
            left: ssa.instruction_count(),
            pending: None,
            next: 0,
        }
    }

    /// The next edit `copies` holds, moving on to the next kind of edit
    /// wherever the current one finds no more in the same view; `None` once
    /// every kind is done.
    fn next<T: Target>(&mut self, copies: &Copies<'_, '_, T>) -> Option<Step<T>> {
        loop {
            let found = match self.phase {
                Phase::Copies => self.bounded(|| copies.find().map(Step::Merge)),
                Phase::ContinuingTests => {
                    self.bounded(|| copies.continuing_tests().map(Step::Tests))
                }
                Phase::EnteringTests => self.bounded(|| copies.entering_tests().map(Step::Tests)),
                Phase::JoinedCopies => self.bounded(|| copies.joined_copies().map(Step::Joined)),
                Phase::Reroutes => {
                    let edges = self.pending.get_or_insert_with(|| copies.threaded_edges());
                    let mut found = None;
                    while let Some(&Reroute { from, to, through }) = edges.get(self.next) {
                        self.next = self.next.saturating_add(1);
                        if let Some(rerouted) = copies.reroutes(from, to, through) {
                            found = Some(Step::Reroute {
                                to,
                                through,
                                rerouted,
                            });
                            break;
                        }
                    }
                    found
                }
                Phase::Guards => self.bounded(|| copies.rotated_guard().map(Step::Unrotate)),
                Phase::Done => return None,
            };
            if found.is_some() {
                return found;
            }
            (self.phase, self.left) = match self.phase {
                Phase::Copies => (Phase::ContinuingTests, copies.ssa.block_count()),
                Phase::ContinuingTests => (Phase::EnteringTests, copies.ssa.block_count()),
                Phase::EnteringTests => (Phase::JoinedCopies, copies.ssa.block_count()),
                Phase::JoinedCopies => (Phase::Reroutes, 0),
                Phase::Reroutes => (Phase::Guards, copies.ssa.block_count()),
                Phase::Guards | Phase::Done => (Phase::Done, 0),
            };
        }
    }

    /// What `find` finds, spending one of the edits left, or nothing once
    /// none is left.
    fn bounded<T: Target>(&mut self, find: impl FnOnce() -> Option<Step<T>>) -> Option<Step<T>> {
        let left = self.left.checked_sub(1)?;
        let found = find()?;
        self.left = left;
        Some(found)
    }
}

/// A test copied into each way into a join ([`Copies::joined_copies`]):
/// blocks asking one question of one value, each sending one answer to the
/// join, and the other on to where the join goes, or to where that test
/// would have sent it, its answer known on the copy's way.
struct JoinedCopies<T: Target> {
    /// The copies.
    tests: Vec<usize>,
    /// The join.
    join: usize,
    /// The test the join goes on to.
    onward: usize,
    /// Copies that reach the join and the onward test each through a block
    /// computing values alike.
    forwarded: Vec<Forwarded>,
    /// Copies whose other answer goes where the onward test would send them.
    sent: Vec<Sent>,
    /// Each copy whose branch reads alike constants of another variable, with
    /// its branch reading the one every copy reads once merged
    /// ([`Copies::read_alike`]).
    rewritten: Vec<(usize, SsaOp<T>)>,
}

/// A copy reaching the join and the onward test through two blocks that
/// compute values alike and jump on.
struct Forwarded {
    /// The copy.
    test: usize,
    /// The block into the join.
    to_join: usize,
    /// The block into the onward test.
    to_onward: usize,
    /// Each value `to_onward` defines, with `to_join`'s in its place.
    renamed: Vec<(SsaVarId, SsaVarId)>,
}

/// A copy whose other answer jumped past the onward test.
struct Sent {
    /// The copy.
    test: usize,
    /// Where its other answer goes.
    arm: usize,
    /// What each phi of the onward test takes from the copy, in order.
    values: Vec<SsaVarId>,
}

/// What a way from the onward test to where a copy's other answer went
/// holds, as [`Copies::carries`] reads it.
#[derive(Clone)]
struct Way {
    /// The blocks of the way, in order.
    blocks: Vec<usize>,
    /// The copy.
    test: usize,
    /// What each phi met on the way holds there.
    held: BTreeMap<SsaVarId, SsaVarId>,
    /// The onward test's phis whose value on the way is not known yet.
    pending: BTreeSet<SsaVarId>,
    /// The loads on the way proven to read what a load above the copy read.
    proven: BTreeSet<SsaVarId>,
}

/// How many blocks the way from the onward test to where a copy's other
/// answer went may run through.
const ONWARD_BLOCKS: usize = 8;

/// How many definitions deep a value along that way is read
/// ([`Copies::carries`]).
const CARRIED_DEPTH: usize = 8;

/// How many ways [`Copies::matches_from`] tries of giving the original's
/// phis that only the tests a copy jumped past read a value on the copy's
/// way.
const OPEN_PHI_CHOICES: usize = 16;

/// Tests alike that pick between the same two blocks
/// ([`Copies::entering_tests`]).
struct Picking<T: Target> {
    /// The two blocks the tests branch to, the lower first.
    blocks: (usize, usize),
    /// The question every test asks.
    shape: TestShape<T>,
    /// The tests.
    tests: Vec<usize>,
}

/// What a two-way branch asks, as [`Copies::shape`] reads it: two branches
/// with one shape ask one question of different values.
#[derive(PartialEq)]
struct TestShape<T: Target> {
    /// The branch with every value it reads set aside.
    blank: SsaOp<T>,
    /// The constant each value it reads is, in order, or `None` for a value
    /// that is none.
    constants: Vec<Option<ConstValue<T>>>,
    /// The question it asks.
    asked: Asked<T>,
    /// The answer that takes its true target.
    holds: bool,
}

/// A question as a branch's condition asks it, with the values it asks of set
/// aside but for the constants.
#[derive(PartialEq)]
enum Asked<T: Target> {
    /// A comparison of two values.
    Compare {
        /// The relation.
        cmp: CmpKind,
        /// Whether the values compare unsigned.
        unsigned: bool,
        /// The left value, where it is a constant.
        left: Option<ConstValue<T>>,
        /// The right value, where it is a constant.
        right: Option<ConstValue<T>>,
    },
    /// A flags value's condition.
    Flags(FlagCondition),
}

/// One block found to end in a copy of another.
struct Found {
    /// The block that ends in the copy.
    copy: usize,
    /// The block it copies.
    original: usize,
    /// Where the copy begins in `copy`.
    at: usize,
    /// Where the rows `copy` ends in a copy of begin in `original`: 0 for
    /// all of them, else the original is split there first
    /// ([`SsaEditor::split_off`](crate::ir::function::SsaEditor::split_off)).
    split: usize,
    /// The values defined in `original` ahead of `split` that are read past
    /// it, each of which the split gives a phi, in that order; empty where
    /// `split` is 0.
    carried: Vec<SsaVarId>,
    /// The value each phi of the block the copy jumps to takes from `copy`,
    /// in the order of the phis: the original's own, or the split's.
    entering: Vec<SsaVarId>,
}

/// The cycles of a function's graph with one block left out
/// ([`Copies::enters_loop`]): each block's strongly connected component in
/// it, and which components hold a cycle.
///
/// A block reaches another and is reached from it, both without passing the
/// left-out block, exactly where the two share a component; and a path of at
/// least one edge leads from a block back to itself so exactly where its
/// component holds more than one block or the block loops on itself.
#[derive(Debug)]
struct LoopsWithout {
    /// Each block's component; the left-out block's is its own.
    component: Vec<usize>,
    /// Whether each component holds a cycle.
    cyclic: Vec<bool>,
}

impl LoopsWithout {
    /// The cycles of `whole` without node `left_out`.
    fn of<G: Successors>(whole: &G, left_out: usize) -> Self {
        let graph = Without {
            graph: whole,
            left_out: NodeId::new(left_out),
        };
        let mut component = vec![0; whole.node_count()];
        let mut cyclic = Vec::new();
        for (index, members) in strongly_connected_components(&graph)
            .into_iter()
            .enumerate()
        {
            let looped = members.len() > 1
                || members
                    .first()
                    .is_some_and(|node| graph.successors(*node).any(|next| next == *node));
            cyclic.push(looped);
            for node in members {
                if let Some(slot) = component.get_mut(node.index()) {
                    *slot = index;
                }
            }
        }
        Self { component, cyclic }
    }

    /// Whether a path of at least one edge leads from `block` back to it.
    fn looped(&self, block: usize) -> bool {
        self.component
            .get(block)
            .and_then(|component| self.cyclic.get(*component))
            .copied()
            .unwrap_or(false)
    }

    /// Whether `one` and `other` each reach the other.
    fn together(&self, one: usize, other: usize) -> bool {
        matches!(
            (self.component.get(one), self.component.get(other)),
            (Some(one), Some(other)) if one == other
        )
    }
}

/// A graph with one node left out: its edges to and from that node gone
/// ([`LoopsWithout`]).
struct Without<'g, G> {
    /// The whole graph.
    graph: &'g G,
    /// The node left out.
    left_out: NodeId,
}

impl<G: GraphBase> GraphBase for Without<'_, G> {
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        self.graph.node_ids()
    }
}

impl<G: Successors> Successors for Without<'_, G> {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        let left_out = self.left_out;
        self.graph
            .successors(node)
            .filter(move |next| node != left_out && *next != left_out)
    }
}

/// What the search for copies reads of one function.
///
/// Dominance, memory and guard facts are built on first use: most functions
/// hold no pair of blocks that branch alike, and are searched without them.
struct Copies<'a, 'c, T: Target> {
    /// The function searched.
    ssa: &'a SsaFunction<T>,
    /// The machine the function runs on.
    context: &'a AnalysisContext<'c>,
    /// Its control-flow graph.
    cfg: SsaCfg<'a, T>,
    /// Dominance from its entry.
    dominators: OnceCell<DominatorTree>,
    /// The function's memory, which says when two loads read one cell in one
    /// state.
    memory: OnceCell<MemorySsa<T>>,
    /// What each block is entered knowing, which says where a test the
    /// compiler jumped past is answered.
    guards: OnceCell<(Definitions<T>, GuardFacts)>,
    /// The cycle each block lies on, by the index of its strongly connected
    /// component; `None` for a block on none.
    cycles: OnceCell<Vec<Option<usize>>>,
    /// The cycles of the graph without each block a copy is sought of
    /// ([`Self::loops_without`]).
    without: RefCell<FxHashMap<usize, Rc<LoopsWithout>>>,
    /// The blocks of each cycle, ascending ([`Self::cycle_members`]).
    members: OnceCell<FxHashMap<usize, Vec<usize>>>,
}

impl<'a, 'c, T: Target> Copies<'a, 'c, T> {
    /// Builds what the search reads of `ssa`.
    fn of(ssa: &'a SsaFunction<T>, context: &'a AnalysisContext<'c>) -> Self {
        Self {
            ssa,
            context,
            cfg: SsaCfg::from_ssa(ssa),
            dominators: OnceCell::new(),
            memory: OnceCell::new(),
            guards: OnceCell::new(),
            cycles: OnceCell::new(),
            without: RefCell::new(FxHashMap::default()),
            members: OnceCell::new(),
        }
    }

    /// The first block that ends in a copy of another, with its original.
    fn find(&self) -> Option<Found> {
        // The blocks that branch to each block.
        let mut entering: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for (index, block) in self.ssa.blocks().iter().enumerate() {
            for successor in block.successors() {
                entering.entry(successor).or_default().insert(index);
            }
        }
        for (copy, block) in self.ssa.blocks().iter().enumerate() {
            // The block entered from one other, or the function's entry,
            // entered from outside every loop.
            let mut sources = self.cfg.predecessors(NodeId::new(copy));
            let source = match (sources.next(), sources.next()) {
                (Some(source), None) => Some(source),
                (None, _) if copy == 0 => None,
                _ => continue,
            };
            let successors = block.successors();
            // An original branches to at least one block the copy does.
            let originals: BTreeSet<usize> = successors
                .iter()
                .filter_map(|successor| entering.get(successor))
                .flatten()
                .copied()
                .collect();
            for original in originals {
                // An original that dominates the copy heads a loop the copy
                // goes round in: what the copy's way in proves of the
                // original's values is of this time round, and merged, the
                // original would compute them afresh from what the copy
                // carries in.
                if original == copy
                    || successors.contains(&original)
                    || self
                        .dominators()
                        .dominates(NodeId::new(original), NodeId::new(copy))
                {
                    continue;
                }
                let Some(route) = self.route(copy, original) else {
                    continue;
                };
                // A copy that skips a test the original's path would make
                // is a compiler's: the source writes no jump past a test it
                // has just passed. One that branches as the original does is
                // the copy a compiler places ahead of a loop, where the
                // original tests on the loop and the copy is entered from
                // outside it. Blocks alike in straight-line code, or entered
                // from the loop itself - the arms of a `switch` in a loop,
                // each opening a file - are the source's own, and stay.
                let threaded = route.iter().any(Option::is_some);
                let entered = match source {
                    Some(source) => self.enters_loop(source, copy, original),
                    None => self
                        .cycles()
                        .get(original)
                        .copied()
                        .flatten()
                        .is_some_and(|cycle| {
                            self.cycles().get(copy).copied().flatten() != Some(cycle)
                        }),
                };
                if !threaded && !entered {
                    continue;
                }
                if let Some(found) = self.matches(copy, original, &route) {
                    return Some(found);
                }
            }
        }
        None
    }

    /// How each block `original` branches to leads to the one `copy`
    /// branches to in its place: straight there (`None`), or through a block
    /// that decides, on the copy's way in, to go on there (`Some`); `None`
    /// when a pair of them does neither.
    fn route(&self, copy: usize, original: usize) -> Option<Vec<Option<usize>>> {
        let ours = self.ssa.block(copy)?.successors();
        let theirs = self.ssa.block(original)?.successors();
        let distinct =
            |targets: &[usize]| targets.iter().collect::<BTreeSet<_>>().len() == targets.len();
        if ours.is_empty() || ours.len() != theirs.len() || !distinct(&ours) || !distinct(&theirs) {
            return None;
        }
        let paired = |theirs: &[usize]| -> Option<Vec<Option<usize>>> {
            ours.iter()
                .zip(theirs)
                .map(|(&mine, &other)| {
                    if mine == other {
                        Some(None)
                    } else {
                        self.forwards(copy, original, other, mine)
                            .then_some(Some(other))
                    }
                })
                .collect()
        };
        if let Some(route) = paired(&theirs) {
            return Some(route);
        }
        // A branch asking the question the other way round names its targets
        // the other way round ([`canonical_terminator`]): where one of the two
        // does, the original's targets pair with the copy's reversed, the
        // two going to the same blocks.
        let (definitions, _) = self.guards();
        let flipped = |block: usize| {
            self.ssa
                .block(block)
                .and_then(SsaBlock::control_terminator)
                .is_some_and(|op| {
                    canonical_terminator(op, definitions).successors() != op.successors()
                })
        };
        if flipped(copy) == flipped(original) {
            return None;
        }
        let mut reversed = theirs.clone();
        reversed.reverse();
        paired(&reversed)
    }

    /// Whether `through`, which `original` branches to where `copy` branches
    /// to `to`, may be a test the compiler's copy jumped past: it computes
    /// nothing but values and branches two ways, one of them to `to`. Whether
    /// the copy's way decides it to go on there is asked once the copy's
    /// values for the original's are known ([`Self::decides_reading`]).
    fn forwards(&self, copy: usize, original: usize, through: usize, to: usize) -> bool {
        if through == copy || through == original || through == to {
            return false;
        }
        let Some((last, rows)) = self
            .ssa
            .block(through)
            .and_then(|block| block.instructions().split_last())
        else {
            return false;
        };
        rows.iter().all(|row| row.op().is_pure())
            && last
                .op()
                .two_way_targets()
                .is_some_and(|(left, right)| left != right && (left == to || right == to))
    }

    /// Whether `through` computes nothing but values, holds no phis, and its
    /// two-way branch is decided to go on to `to` by what `from` is entered
    /// knowing ([`GuardFacts::answer`]): a test the path had answered before
    /// `from`'s own. What `from`'s own branch says is no test skipped: a
    /// block asking it again is a copy of `from`'s test.
    fn decides_onward(&self, from: usize, through: usize, to: usize) -> bool {
        self.ssa
            .block(through)
            .is_some_and(|block| block.phi_nodes().is_empty())
            && self.decides_reading(from, through, to, |value| value)
    }

    /// Whether `through` computes nothing but values, and its two-way
    /// branch, asked of the value `reading` gives in place of each it reads,
    /// is decided to go on to `to` by what `from` is entered knowing
    /// ([`GuardFacts::answer`]).
    fn decides_reading(
        &self,
        from: usize,
        through: usize,
        to: usize,
        reading: impl Fn(SsaVarId) -> SsaVarId,
    ) -> bool {
        let Some(block) = self.ssa.block(through) else {
            return false;
        };
        let Some((last, rows)) = block.instructions().split_last() else {
            return false;
        };
        if !rows.iter().all(|row| row.op().is_pure()) {
            return false;
        }
        let terminator = last.op();
        let Some((true_target, false_target)) = terminator.two_way_targets() else {
            return false;
        };
        let (definitions, guards) = self.guards();
        let Some((question, takes_true)) = Question::of_terminator(terminator, definitions) else {
            return false;
        };
        guards
            .answer(from, question.reading(reading), definitions)
            .is_some_and(|holds| {
                let taken = if holds == takes_true {
                    true_target
                } else {
                    false_target
                };
                taken == to
            })
    }

    /// What `through`, a test past a loop the edge leaves, comes to on
    /// `from`'s edge to `to`, when it sends it on to `to`
    /// ([`GuardFacts::edge_answer`]: the edge's own branch, what `from` is
    /// entered knowing, and a literal): `fill_arglist`'s `if (*lastp) break;`
    /// and its caller's `if (osp)` on the same cell, once the compiler sends
    /// the break straight into the `if`, or `expmeta`'s latch past
    /// `metaflag = 1` and the `if (metaflag)` after its loop. `through` holds
    /// no phis and computes nothing but values and loads, and each load reads,
    /// on the edge's way, what a load at or above `from` read: the cell in
    /// the state memory leaves `from` in. Run on that way it reads what was
    /// read and faults nowhere the path did not. The question is asked of
    /// those values, and of the version the edge holds of each value
    /// `versions` names ([`Question::reading`]).
    ///
    /// # Returns
    ///
    /// Each load of `through` with the value it reads on the edge's way, or
    /// `None` where `through` is no such test.
    fn retests_exit(
        &self,
        from: usize,
        through: usize,
        to: usize,
        onward: usize,
        versions: &BTreeMap<SsaVarId, SsaVarId>,
    ) -> Option<BTreeMap<SsaVarId, SsaVarId>> {
        let block = self.ssa.block(through)?;
        let (last, rows) = block.instructions().split_last()?;
        if !block.phi_nodes().is_empty() {
            return None;
        }
        let terminator = last.op();
        let (true_target, false_target) = terminator.two_way_targets()?;
        let (definitions, guards) = self.guards();
        let memory = self.memory();
        let leaving = self.ssa.block(from)?.instructions().len();
        let mut read_as: BTreeMap<SsaVarId, SsaVarId> = BTreeMap::new();
        for (index, row) in rows.iter().enumerate() {
            if row.op().is_pure() {
                continue;
            }
            let (location, loaded) = memory.operations().iter().find_map(|op| match op {
                MemoryOp::Load {
                    location,
                    dest,
                    block,
                    instr,
                } if *block == through && *instr == index => Some((location, *dest)),
                _ => None,
            })?;
            let cell = memory.location_id(location)?;
            let version = memory.version_before_id(cell, from, leaving)?;
            let read = definitions.loads_reading(cell, version).find(|load| {
                self.ssa.variable(*load).is_some_and(|var| {
                    self.dominators()
                        .dominates(NodeId::new(var.def_site().block()), NodeId::new(from))
                })
            })?;
            read_as.insert(loaded, read);
        }
        let (question, takes_true) = Question::of_terminator(terminator, definitions)?;
        let asked = question.reading(|value| {
            read_as
                .get(&value)
                .or_else(|| versions.get(&value))
                .copied()
                .unwrap_or(value)
        });
        let holds = guards.edge_answer(self.ssa, from, to, asked, definitions)?;
        let taken = if holds == takes_true {
            true_target
        } else {
            false_target
        };
        (taken == onward).then_some(read_as)
    }

    /// Whether `through`'s test is one `from`'s way in has asked already
    /// ([`GuardFacts::asks`]): a block asking it again is a copy of that
    /// test, not one the compiler jumped past, and the edge stays.
    fn asks_again(&self, from: usize, through: usize) -> bool {
        let Some(terminator) = self
            .ssa
            .block(through)
            .and_then(|block| block.instructions().last())
        else {
            return true;
        };
        let (definitions, guards) = self.guards();
        Question::of_terminator(terminator.op(), definitions)
            .is_none_or(|(question, _)| guards.asks(from, question, definitions))
    }

    /// Two or more tests a loop goes on through alike: blocks inside the loop
    /// that branch on to the loop's header, or to a block that only jumps
    /// there, and out to one block, with one branch alike but for the values
    /// it reads ([`Self::shape`]). A compiler that rotates a loop writes its
    /// test at the end of each way round, and the source wrote it once.
    fn continuing_tests(&self) -> Option<Vec<usize>> {
        let blocks = self.ssa.block_count();
        let headers: BTreeSet<usize> = (0..blocks)
            .flat_map(|block| {
                self.cfg
                    .successors(NodeId::new(block))
                    .filter(move |next| self.dominators().dominates(*next, NodeId::new(block)))
                    .map(NodeId::index)
                    .collect::<Vec<_>>()
            })
            .collect();
        for header in headers {
            let within = |block: usize| {
                self.dominators()
                    .dominates(NodeId::new(header), NodeId::new(block))
            };
            // The blocks the loop goes on through: its header, and those that
            // only jump to it.
            let mut onward = vec![header];
            onward.extend(
                self.cfg
                    .predecessors(NodeId::new(header))
                    .map(NodeId::index)
                    .filter(|block| {
                        *block != header
                            && within(*block)
                            && self.ssa.block(*block).is_some_and(|block| {
                                let rows = block.instructions();
                                rows.len() == 1
                                    && rows.first().is_some_and(|row| {
                                        row.op().successors() == [header]
                                            && matches!(row.op(), SsaOp::Jump { .. })
                                    })
                            })
                    }),
            );
            for &through in &onward {
                // Only two or more tests become one, and only a two-way branch
                // to two blocks has a shape ([`Self::shape`]): where fewer
                // tests than two branch so, no shape is read.
                let tests: Vec<usize> = self
                    .cfg
                    .predecessors(NodeId::new(through))
                    .map(NodeId::index)
                    .filter(|test| {
                        !onward.contains(test)
                            && within(*test)
                            && self
                                .ssa
                                .block(*test)
                                .and_then(SsaBlock::control_terminator)
                                .and_then(SsaOp::two_way_targets)
                                .is_some_and(|(on, off)| on != off)
                    })
                    .collect();
                if tests.len() < 2 {
                    continue;
                }
                let mut alike: Vec<(TestShape<T>, Vec<usize>)> = Vec::new();
                for test in tests {
                    let Some(shape) = self.shape(test) else {
                        continue;
                    };
                    match alike.iter_mut().find(|(known, _)| *known == shape) {
                        Some((_, members)) => members.push(test),
                        None => alike.push((shape, vec![test])),
                    }
                }
                let merged = blocks;
                if let Some((_, tests)) = alike.into_iter().find(|(_, tests)| {
                    tests.len() >= 2
                        && tests.iter().all(|test| {
                            self.ssa.block(*test).is_some_and(|block| {
                                block.successors().iter().all(|next| {
                                    self.ssa.retarget_respects_regions(*test, *next, merged)
                                        && self.ssa.retarget_respects_regions(merged, *next, *next)
                                })
                            })
                        })
                }) {
                    return Some(tests);
                }
            }
        }
        None
    }

    /// Two or more tests alike that pick between two blocks a loop is
    /// entered at: blocks that branch to the same two blocks, with one branch
    /// alike but for the values it reads ([`Self::shape`]), where each of the
    /// two reaches the other without passing the nearest block dominating
    /// both. Such a loop is entered at two blocks, which the source writes no
    /// way: a compiler that wrote the loop as two versions, `expmeta` in dash
    /// copying a name with one version for the character past a backslash,
    /// copied the test that picks one into each way in, and the source wrote
    /// it once. Two arms of an `if` in a loop reach each other only round
    /// the loop, through the test that dominates both, and stay.
    fn entering_tests(&self) -> Option<Vec<usize>> {
        let blocks = self.ssa.block_count();
        let cycles = self.cycles();
        let mut alike: Vec<Picking<T>> = Vec::new();
        for test in 0..blocks {
            let Some((on, off)) = self
                .ssa
                .block(test)
                .and_then(SsaBlock::control_terminator)
                .and_then(SsaOp::two_way_targets)
            else {
                continue;
            };
            let looped = cycles
                .get(on)
                .copied()
                .flatten()
                .is_some_and(|cycle| cycles.get(off).copied().flatten() == Some(cycle));
            if on == off
                || !looped
                || self
                    .dominators()
                    .dominates(NodeId::new(on), NodeId::new(off))
                || self
                    .dominators()
                    .dominates(NodeId::new(off), NodeId::new(on))
            {
                continue;
            }
            let Some(shape) = self.shape(test) else {
                continue;
            };
            let pair = (on.min(off), on.max(off));
            match alike
                .iter_mut()
                .find(|known| known.blocks == pair && known.shape == shape)
            {
                Some(known) => known.tests.push(test),
                None => alike.push(Picking {
                    blocks: pair,
                    shape,
                    tests: vec![test],
                }),
            }
        }
        let merged = blocks;
        alike.into_iter().find_map(
            |Picking {
                 blocks: (on, off),
                 tests,
                 ..
             }| {
                let meet = self
                    .dominators()
                    .nearest_common_dominator(NodeId::new(on), NodeId::new(off))?
                    .index();
                (tests.len() >= 2
                    && self.reaches_avoiding(on, off, meet)
                    && self.reaches_avoiding(off, on, meet)
                    && tests.iter().all(|test| {
                        self.ssa.block(*test).is_some_and(|block| {
                            block.successors().iter().all(|next| {
                                self.ssa.retarget_respects_regions(*test, *next, merged)
                                    && self.ssa.retarget_respects_regions(merged, *next, *next)
                            })
                        })
                    }))
                .then_some(tests)
            },
        )
    }

    /// Whether control reaches `to` from `from` without entering `avoiding`.
    fn reaches_avoiding(&self, from: usize, to: usize, avoiding: usize) -> bool {
        let mut seen = vec![false; self.ssa.block_count()];
        let mut pending = vec![from];
        while let Some(block) = pending.pop() {
            for next in self.cfg.successors(NodeId::new(block)).map(NodeId::index) {
                if next == to {
                    return true;
                }
                if next != avoiding
                    && let Some(slot) = seen.get_mut(next)
                    && !*slot
                {
                    *slot = true;
                    pending.push(next);
                }
            }
        }
        false
    }

    /// The shape of `block`'s two-way branch to two distinct blocks: the
    /// branch with each value it reads set aside, save the constants, and the
    /// question it asks read through what defines its condition
    /// ([`Question::of_terminator`]): the relation, and the constants it
    /// compares with. `None` for a block ending otherwise, or in a branch on
    /// a value whose definition asks nothing: two such branches ask two
    /// questions no reading tells apart, a `break` on one condition and on
    /// another.
    fn shape(&self, block: usize) -> Option<TestShape<T>> {
        // Read through its definitions, a branch at most exchanges its
        // targets: one that is not two-way to two blocks stays so.
        let branch = self.ssa.block(block)?.control_terminator()?;
        let (true_target, false_target) = branch.two_way_targets()?;
        if true_target == false_target {
            return None;
        }
        let (definitions, _) = self.guards();
        let terminator = &canonical_terminator(branch, definitions);
        let constant = |var: SsaVarId| match self.ssa.get_definition(var) {
            Some(SsaOp::Const { value, .. }) => Some(value.clone()),
            _ => None,
        };
        let (question, holds) = Question::of_terminator(terminator, definitions)?;
        let asked = match question {
            Question::Compare {
                left,
                right,
                cmp,
                unsigned,
            } => Asked::Compare {
                cmp,
                unsigned,
                left: constant(left),
                right: constant(right),
            },
            Question::Flags { condition, .. } => Asked::Flags(condition),
            Question::Value(_) => return None,
        };
        let mut constants = Vec::new();
        let mut blank = terminator.clone();
        blank.visit_operands_mut(|role, var| {
            if role == OperandRole::Use {
                constants.push(constant(*var));
                *var = SsaVarId::PLACEHOLDER;
            }
        });
        Some(TestShape {
            blank,
            constants,
            asked,
            holds,
        })
    }

    /// Every edge a compiler threaded past a test, as `from`'s edge to `to`
    /// skipping `through` ([`Self::reroutes`]), at most one per edge.
    fn threaded_edges(&self) -> Vec<Reroute> {
        let mut found = Vec::new();
        for (from, block) in self.ssa.blocks().iter().enumerate() {
            for to in block.successors() {
                // Each test into `to`: its own way in, or the test a block
                // that only jumps there is entered from.
                let tests: Vec<usize> = self
                    .cfg
                    .predecessors(NodeId::new(to))
                    .map(NodeId::index)
                    .flat_map(|entering| {
                        let mut tests = vec![entering];
                        tests.extend(self.forwarded_from(entering, to));
                        tests
                    })
                    .collect();
                if let Some(through) = tests
                    .into_iter()
                    .find(|through| self.reroutes(from, to, *through).is_some())
                {
                    found.push(Reroute { from, to, through });
                }
            }
        }
        found
    }

    /// Whether `from`'s edge into `to` skips `through`, another way into `to`
    /// that would send `from` there, and how every way into `to` goes through
    /// it with the edge ([`Self::skips`]). The block the edge goes into is
    /// the test's arm once they do: entered from the test alone. A block that
    /// more ways reach, within one region - the end of a `switch` every case
    /// breaks to - is a join, and an edge into it is a `break`, not a test
    /// skipped. Past a loop, the breaks out of it that skip the test after it
    /// go through it together: `expmeta`'s `if (metaflag) goto out` and its
    /// latch past `metaflag = 1` both reach the arm `out:`'s `if (metaflag ==
    /// 0)` sends them to. A test that reaches `to` through a block that only
    /// jumps there, or defines the constants `to` takes on its edge
    /// ([`Self::arm_into`]), is one into `to` as well: the block stands for
    /// the split edge.
    ///
    /// Each value `through` reads, or passes on to its other arm, that it
    /// does not compute is one each edge holds: the value itself, defined
    /// above the edge, or the version the edge holds of it
    /// ([`Self::versions`]), which `through` joins
    /// ([`SsaEditor::route_through`](crate::ir::function::SsaEditor::route_through)).
    ///
    /// # Returns
    ///
    /// The edges into `to` and what `through` joins for them, or `None`
    /// where the edge skips no test.
    fn reroutes(&self, from: usize, to: usize, through: usize) -> Option<Rerouted> {
        let arm = self.arm_into(through, to)?;
        // The block the test goes to on its way into `to`.
        let onward = if arm == through { to } else { arm };
        let mut froms = vec![from];
        froms.extend(
            self.cfg
                .predecessors(NodeId::new(to))
                .map(NodeId::index)
                .filter(|entering| *entering != from && *entering != arm),
        );
        // Read from the graph alone, before any fact is.
        if froms.len() > 1
            && !froms
                .iter()
                .all(|entering| self.leaves(*entering, to, through))
        {
            return None;
        }
        let mut joined = BTreeSet::new();
        let mut held = Vec::with_capacity(froms.len());
        for &entering in &froms {
            let skipped = self.skips(entering, to, through, arm, onward)?;
            joined.extend(skipped.joined);
            held.push(skipped.versions);
        }
        let mut rejoined = Vec::new();
        for value in self.carried_past(through, onward) {
            let versions = froms
                .iter()
                .zip(&held)
                .map(|(&entering, versions)| {
                    versions.get(&value).copied().or_else(|| {
                        self.ssa
                            .variable(value)
                            .is_some_and(|var| {
                                self.dominators().dominates(
                                    NodeId::new(var.def_site().block()),
                                    NodeId::new(entering),
                                )
                            })
                            .then_some(value)
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            if versions.iter().any(|version| *version != value) {
                rejoined.push((value, versions));
            }
        }
        // A phi of `to` taking a value `through` joins takes what each edge
        // carried, not the edge's version: the test reads the version, and
        // `to` what it read before.
        for (index, phi) in self.ssa.block(to)?.phi_nodes().iter().enumerate() {
            let theirs = phi.operand_from(arm)?.value();
            if rejoined.iter().any(|(value, _)| *value == theirs) {
                joined.insert(index);
            }
        }
        Some(Rerouted {
            froms,
            joined: joined.into_iter().collect(),
            rejoined,
            via: (arm != through).then_some(arm),
        })
    }

    /// The block `through` goes into `to` by: `through` itself where it
    /// branches there, or a block it alone enters that only jumps there -
    /// the edge from a test to a join split, as dash's `expmeta` on RISC-V
    /// reaches what follows `out:` from its `if (metaflag == 0)`. `None`
    /// where it reaches `to` neither way.
    fn arm_into(&self, through: usize, to: usize) -> Option<usize> {
        let successors = self.ssa.block(through)?.successors();
        if successors.contains(&to) {
            return Some(through);
        }
        successors
            .into_iter()
            .find(|arm| self.forwarded_from(*arm, to) == Some(through))
    }

    /// The block that alone enters `block`, where `block` holds no phis and
    /// nothing but constants and a jump to `to`: the constants a phi of `to`
    /// takes on that edge, as `expmeta`'s `reg_22 = 0` on RISC-V.
    fn forwarded_from(&self, block: usize, to: usize) -> Option<usize> {
        let forwarding = self.ssa.block(block)?;
        let (jump, rows) = forwarding.instructions().split_last()?;
        if !forwarding.phi_nodes().is_empty()
            || !matches!(jump.op(), SsaOp::Jump { target } if *target == to)
            || !rows
                .iter()
                .all(|row| matches!(row.op(), SsaOp::Const { .. }))
        {
            return None;
        }
        let mut entering = self.cfg.predecessors(NodeId::new(block));
        match (entering.next(), entering.next()) {
            (Some(single), None) => Some(single.index()),
            _ => None,
        }
    }

    /// Whether `from`'s edge into `to` skips `through`, judged alone
    /// ([`Self::reroutes`]). The test is one of two kinds. One decides on
    /// `from`'s way in to go on to `to` ([`Self::decides_onward`]) through a
    /// value the way in proved a number, never by asking it
    /// ([`Self::asks_again`]), and the edge sent through it enters no loop it
    /// did not ([`Self::cycles`]). The other lies past a loop the edge
    /// leaves, and the edge decides it ([`Self::retests_exit`]) - its own
    /// branch, as where the source broke out of the loop and tested once
    /// more and the compiler jumped from the break past the test, or the
    /// versions the edge holds of the values the test reads
    /// ([`Self::versions`]), as where the latch that set `metaflag = 1`
    /// leaves past the `if (metaflag)` after the loop, which reads the
    /// latch's `metaflag`. Either way `through` does not dominate `from`, so nothing
    /// `from`'s way in proves is of a value `through` computes.
    ///
    /// `to` takes from `through` what it takes from `from` on the edge's way,
    /// or, past a loop, a value defined above `through` that `through` joins
    /// with `from`'s: the header's `argp` past a loop, where the break
    /// carries the next one.
    ///
    /// The compiler's jump past a test runs no test of its own. An edge that
    /// passes host code ending in a test ([`BlockProvenance::ends_in_a_test`])
    /// is no such jump: the host decides there, and a fold decided it, `if
    /// (syntax == ARISYNTAX)` in dash's `readtoken1` once the path proved
    /// `syntax` another table; the host runs that test on the edge, not the
    /// one the edge would be sent through. Code that only moves values and
    /// goes on, a register copy the lowering coalesced, is passed by a jump
    /// the compiler threaded as much as by any other.
    ///
    /// [`BlockProvenance::ends_in_a_test`]: crate::ir::block::BlockProvenance::ends_in_a_test
    fn skips(
        &self,
        from: usize,
        to: usize,
        through: usize,
        arm: usize,
        onward: usize,
    ) -> Option<Skipped> {
        let block = self.ssa.block(from)?;
        let provenance = block.provenance();
        if provenance
            .passed_on(to)
            .into_iter()
            .any(|host| provenance.ends_in_a_test(host))
        {
            return None;
        }
        let successors = block.successors();
        if through == from
            || through == to
            || successors.contains(&through)
            || !successors.contains(&to)
            || self
                .dominators()
                .dominates(NodeId::new(through), NodeId::new(from))
        {
            return None;
        }
        let cycles = self.cycles();
        let cycle = |block: usize| cycles.get(block).copied().flatten();
        let (looped, own) = (cycle(through), cycle(from));
        let leaves = self.leaves(from, to, through);
        let skipped = (looped.is_none() || looped == own)
            && !leaves
            && self.decides_onward(from, through, onward)
            && !self.asks_again(from, through);
        let mut versions = BTreeMap::new();
        // What `through` reads on the edge's way, where it is a test past the
        // loop the edge leaves.
        let read_as = if skipped {
            BTreeMap::new()
        } else if leaves {
            versions = self.versions(from, to, through);
            self.retests_exit(from, through, to, onward, &versions)?
        } else {
            return None;
        };
        let mut joined = Vec::new();
        for (index, phi) in self.ssa.block(to)?.phi_nodes().iter().enumerate() {
            let ours = phi.operand_from(from)?.value();
            let theirs = phi.operand_from(arm)?.value();
            if self.same_value(ours, read_as.get(&theirs).copied().unwrap_or(theirs)) {
                continue;
            }
            // A skipped test the edge would carry another value through is
            // not one the compiler threaded past; a loop's exit carries the
            // step the break took. A value `through` computes is no value it
            // can join.
            if skipped
                || self
                    .ssa
                    .variable(theirs)
                    .is_none_or(|var| var.def_site().block() == through)
            {
                return None;
            }
            joined.push(index);
        }
        self.ssa
            .retarget_respects_regions(from, to, through)
            .then_some(Skipped { joined, versions })
    }

    /// The first test a compiler copied ahead of a loop it rotated: a loop
    /// whose test `latch` lies at its foot, going on to `header` or out,
    /// entered from outside only through a block that jumps to `header`,
    /// which only a `guard` outside the loop enters ([`Self::unrotates`]).
    fn rotated_guard(&self) -> Option<Unrotated> {
        let cycles = self.cycles();
        let cycle = |block: usize| cycles.get(block).copied().flatten();
        for (latch, block) in self.ssa.blocks().iter().enumerate() {
            let Some(own) = cycle(latch) else {
                continue;
            };
            let Some((true_target, false_target)) =
                block.control_terminator().and_then(SsaOp::two_way_targets)
            else {
                continue;
            };
            let (header, exit) = match (
                cycle(true_target) == Some(own),
                cycle(false_target) == Some(own),
            ) {
                (true, false) => (true_target, false_target),
                (false, true) => (false_target, true_target),
                _ => continue,
            };
            if header == latch {
                continue;
            }
            let mut outside = self
                .cfg
                .predecessors(NodeId::new(header))
                .map(NodeId::index)
                .filter(|entering| cycle(*entering) != Some(own));
            let (Some(forwarder), None) = (outside.next(), outside.next()) else {
                continue;
            };
            if let Some(found) = self.unrotates(latch, header, exit, forwarder) {
                return Some(found);
            }
        }
        None
    }

    /// Whether the block entering `forwarder` is a copy of `latch`'s test a
    /// compiler placed ahead of the loop when it rotated it, and how the
    /// loop is entered at its test instead.
    ///
    /// `forwarder` computes only values and jumps to `header`, and one
    /// block, `guard`, enters it, branching to it and past the loop. Each
    /// phi of `latch` goes round to a phi of `header`, and holds on the
    /// guard's way what that phi takes from `forwarder`: the value it starts
    /// the loop with, `metaflag`'s `0` in dash's `expmeta`. Every phi of
    /// `header` takes from `latch` one of those, or what it takes from
    /// `forwarder` too. Run with those, `latch` asks what `guard` asked
    /// ([`Question::relates`]), and sends the guard's way into the loop on
    /// to `header`: `expmeta`'s `if (*name == 0)` on the byte read and its
    /// latch's `c != 0` on the character widened. The guard's way past the
    /// loop leaves it at `latch`'s exit and reaches the block `guard`
    /// branched to past it, holding what `guard` gave it: the exit is that
    /// block, or computes only values and decides, on the guard's way past
    /// the loop, to go on there ([`GuardFacts::edge_answer`]), as
    /// `expmeta`'s `if (metaflag == 0)` after the loop does with `metaflag`
    /// `0`. `latch` reads nothing else the loop computes, and nothing the
    /// loop computes is read past it but by a phi of a way out: entered
    /// from `forwarder`, `latch` runs where the loop has not.
    fn unrotates(
        &self,
        latch: usize,
        header: usize,
        exit: usize,
        forwarder: usize,
    ) -> Option<Unrotated> {
        let cycles = self.cycles();
        let cycle = |block: usize| cycles.get(block).copied().flatten();
        let own = cycle(latch)?;
        let forwarding = self.ssa.block(forwarder)?;
        let (jump, rows) = forwarding.instructions().split_last()?;
        if !matches!(jump.op(), SsaOp::Jump { target } if *target == header)
            || !forwarding.phi_nodes().is_empty()
            || !rows.iter().all(|row| row.op().is_pure())
        {
            return None;
        }
        let mut entered = self.cfg.predecessors(NodeId::new(forwarder));
        let guard = match (entered.next(), entered.next()) {
            (Some(guard), None) => guard.index(),
            _ => return None,
        };
        if cycle(guard).is_some_and(|cycle| cycle == own) {
            return None;
        }
        let guarding = self.ssa.block(guard)?;
        let asked = guarding.control_terminator()?;
        let (guard_true, guard_false) = asked.two_way_targets()?;
        let skipped = match (guard_true == forwarder, guard_false == forwarder) {
            (true, false) => guard_false,
            (false, true) => guard_true,
            _ => return None,
        };
        // What each of `latch`'s phis holds on the guard's way: what the
        // phi of `header` it goes round to starts the loop with.
        let tested = self.ssa.block(latch)?;
        let heading = self.ssa.block(header)?;
        let mut held: BTreeMap<SsaVarId, SsaVarId> = BTreeMap::new();
        let mut entering = Vec::with_capacity(tested.phi_nodes().len());
        for phi in tested.phi_nodes() {
            let fed = heading
                .phi_nodes()
                .iter()
                .find(|fed| fed.operand_from(latch).map(PhiOperand::value) == Some(phi.result()))?;
            let start = fed.operand_from(forwarder)?.value();
            if !self.available(start, forwarder, rows.len()) {
                return None;
            }
            held.insert(phi.result(), start);
            entering.push(start);
        }
        for fed in heading.phi_nodes() {
            let round = fed.operand_from(latch)?.value();
            let start = fed.operand_from(forwarder)?.value();
            if held.get(&round) != Some(&start) && round != start {
                return None;
            }
        }
        // What `latch` reads, and passes on along its ways out, is its own,
        // or defined where the guard's way into it has it.
        let own_value = |value: SsaVarId| {
            self.ssa.variable(value).is_some_and(|var| {
                let block = var.def_site().block();
                block == latch
                    || self
                        .dominators()
                        .dominates(NodeId::new(block), NodeId::new(forwarder))
            })
        };
        if !tested
            .instructions()
            .iter()
            .all(|row| row.op().is_pure() || row.op().two_way_targets().is_some())
            || !tested
                .instructions()
                .iter()
                .flat_map(|row| row.op().uses())
                .all(own_value)
        {
            return None;
        }
        for target in [header, exit] {
            for phi in self.ssa.block(target)?.phi_nodes() {
                if !phi
                    .operand_from(latch)
                    .is_some_and(|operand| own_value(operand.value()))
                {
                    return None;
                }
            }
        }
        // Nothing the loop computes is read past it but by a phi of a way
        // out of it.
        for &block in self.cycle_members().get(&own).into_iter().flatten() {
            let Some(body) = self.ssa.block(block).filter(|_| block != latch) else {
                continue;
            };
            let defined = body
                .phi_nodes()
                .iter()
                .map(|phi| phi.result())
                .chain(body.instructions().iter().flat_map(|row| row.op().defs()));
            for value in defined {
                let leaves = self.ssa.variable(value)?.uses().iter().any(|site| {
                    if cycle(site.block()) == Some(own) {
                        return false;
                    }
                    !site.is_phi_operand()
                        || self
                            .ssa
                            .block(site.block())
                            .and_then(|reader| reader.phi_nodes().get(site.index()))
                            .is_none_or(|phi| {
                                phi.operands().iter().any(|operand| {
                                    operand.value() == value
                                        && cycle(operand.predecessor()) != Some(own)
                                })
                            })
                });
                if leaves {
                    return None;
                }
            }
        }
        // Run with what it holds on the guard's way, `latch` asks what
        // `guard` asks, and goes on into the loop where `guard` did.
        let (definitions, guards) = self.guards();
        let ptr_size = self.context.machine().ptr_size;
        let (question, takes_true) =
            Question::of_terminator(tested.control_terminator()?, definitions)?;
        let read = question.reading(|value| held.get(&value).copied().unwrap_or(value));
        let (guarded, guard_takes_true) = Question::of_terminator(asked, definitions)?;
        let same = guarded.relates(read, definitions, ptr_size)?;
        let into = if guard_true == forwarder {
            guard_takes_true
        } else {
            !guard_takes_true
        };
        let (latch_true, _) = tested.control_terminator()?.two_way_targets()?;
        let onward = if latch_true == header {
            takes_true
        } else {
            !takes_true
        };
        if (into == same) != onward {
            return None;
        }
        // Past the loop, the guard's way reaches where `guard` sent it.
        let way_out = |value: SsaVarId| -> SsaVarId {
            let through_exit = self
                .ssa
                .block(exit)
                .and_then(|block| block.phi_nodes().iter().find(|phi| phi.result() == value))
                .and_then(|phi| phi.operand_from(latch))
                .map_or(value, PhiOperand::value);
            held.get(&through_exit).copied().unwrap_or(through_exit)
        };
        if exit != skipped {
            let leaving = self.ssa.block(exit)?;
            let (last, rows) = leaving.instructions().split_last()?;
            if !rows.iter().all(|row| row.op().is_pure()) {
                return None;
            }
            let (exit_true, exit_false) = last.op().two_way_targets()?;
            let (left, exit_takes_true) = Question::of_terminator(last.op(), definitions)?;
            let holds =
                guards.edge_answer(self.ssa, guard, skipped, left.reading(way_out), definitions)?;
            let taken = if holds == exit_takes_true {
                exit_true
            } else {
                exit_false
            };
            if taken != skipped {
                return None;
            }
        }
        let from = if exit == skipped { latch } else { exit };
        for phi in self.ssa.block(skipped)?.phi_nodes() {
            let given = phi.operand_from(guard)?.value();
            let reached = phi.operand_from(from)?.value();
            let reached = if exit == skipped {
                held.get(&reached).copied().unwrap_or(reached)
            } else {
                way_out(reached)
            };
            if !self.same_value(given, reached) {
                return None;
            }
        }
        (self.ssa.retarget_respects_regions(forwarder, header, latch)
            && self
                .ssa
                .retarget_respects_regions(guard, skipped, forwarder))
        .then_some(Unrotated {
            guard,
            forwarder,
            latch,
            header,
            entering,
        })
    }

    /// Whether `from`'s edge into `to` leaves a loop that neither `through`
    /// nor `to` lies on ([`Self::cycles`]).
    fn leaves(&self, from: usize, to: usize, through: usize) -> bool {
        let cycles = self.cycles();
        let cycle = |block: usize| cycles.get(block).copied().flatten();
        let own = cycle(from);
        own.is_some() && cycle(through) != own && cycle(to) != own
    }

    /// Each value `through` reads that is a version of a variable `from`
    /// holds another version of, with that version: a phi of a block `from`
    /// goes on to besides `to` - the loop's header or latch, entered again on
    /// `from`'s other way, directly or through blocks that jump on - above
    /// `through`, and the operand it takes on that way, read back through
    /// the joins it passed, where that operand is defined above `from`:
    /// `expmeta`'s `if (metaflag)` past its loop reads the latch's
    /// `metaflag`, which the way on from the `if (metaflag) break` keeps.
    fn versions(&self, from: usize, to: usize, through: usize) -> BTreeMap<SsaVarId, SsaVarId> {
        let read: BTreeSet<SsaVarId> = self
            .ssa
            .block(through)
            .map(|block| {
                block
                    .instructions()
                    .iter()
                    .flat_map(|row| row.op().uses())
                    .collect()
            })
            .unwrap_or_default();
        let mut versions = BTreeMap::new();
        for first in self
            .ssa
            .block(from)
            .map(SsaBlock::successors)
            .unwrap_or_default()
        {
            if first == to || first == through {
                continue;
            }
            // The way on from `from`: each block it passes that only joins
            // values and jumps, and the block it reaches.
            let mut way = vec![from];
            let mut next = first;
            for _ in 0..self.ssa.block_count() {
                let Some(block) = self.ssa.block(next) else {
                    break;
                };
                let forwards = matches!(block.control_terminator(), Some(SsaOp::Jump { .. }));
                match block.successors()[..] {
                    [onward]
                        if forwards
                            && !way.contains(&onward)
                            && onward != to
                            && onward != through =>
                    {
                        way.push(next);
                        next = onward;
                    }
                    _ => break,
                }
            }
            if !self
                .dominators()
                .dominates(NodeId::new(next), NodeId::new(through))
            {
                continue;
            }
            for phi in self
                .ssa
                .block(next)
                .map(SsaBlock::phi_nodes)
                .unwrap_or_default()
            {
                if !read.contains(&phi.result()) {
                    continue;
                }
                // Back along the way, through each join it passed: a block
                // that joins nothing passes the value on as it is.
                let mut value = phi.result();
                let mut at = next;
                for &previous in way.iter().rev() {
                    if let Some(joined) = self.ssa.block(at).and_then(|block| {
                        block
                            .phi_nodes()
                            .iter()
                            .find(|joined| joined.result() == value)
                    }) {
                        let Some(operand) = joined.operand_from(previous) else {
                            break;
                        };
                        value = operand.value();
                    }
                    at = previous;
                }
                let held = self.ssa.variable(value).is_some_and(|var| {
                    self.dominators()
                        .dominates(NodeId::new(var.def_site().block()), NodeId::new(from))
                });
                if at == from && value != phi.result() && held {
                    versions.insert(phi.result(), value);
                }
            }
        }
        versions
    }

    /// The values `through` reads, or passes on to the phis of a block it
    /// goes to other than `to`, that it does not compute: what each way into
    /// it must hold.
    fn carried_past(&self, through: usize, to: usize) -> BTreeSet<SsaVarId> {
        let Some(block) = self.ssa.block(through) else {
            return BTreeSet::new();
        };
        let mut carried: BTreeSet<SsaVarId> = block
            .instructions()
            .iter()
            .flat_map(|row| row.op().uses())
            .collect();
        for next in block.successors() {
            if next == to {
                continue;
            }
            for phi in self
                .ssa
                .block(next)
                .map(SsaBlock::phi_nodes)
                .unwrap_or_default()
            {
                carried.extend(phi.operand_from(through).map(PhiOperand::value));
            }
        }
        carried.retain(|value| {
            self.ssa
                .variable(*value)
                .is_some_and(|var| var.def_site().block() != through)
        });
        carried
    }

    /// Whether `copy` ends in a copy of `original`, and how, where `route`
    /// says how each block `original` branches to leads to the copy's
    /// ([`Self::route`]): of all its rows, or else of the rows it ends in,
    /// the longest run of them that matches. A compiler that copies a loop's
    /// test ahead of the loop leaves out what the test's block does first
    /// for the way round: `pmatch`'s latch steps `q` and then reads the next
    /// pattern character, and the copy at the entry only reads it.
    fn matches(&self, copy: usize, original: usize, route: &[Option<usize>]) -> Option<Found> {
        let rows = self.ssa.block(original)?.instructions().len();
        (0..rows.saturating_sub(1))
            .find_map(|split| self.matches_from(copy, original, split, route))
    }

    /// Whether `copy` ends in a copy of `original`'s rows from `split` on
    /// ([`Self::matches`]). The values `original` defines ahead of `split` -
    /// its phis, and its rows before it - are what a split gives a phi of the
    /// split's own where something past it reads them, so they bind as the
    /// original's phis do.
    fn matches_from(
        &self,
        copy: usize,
        original: usize,
        split: usize,
        route: &[Option<usize>],
    ) -> Option<Found> {
        let copied = self.ssa.block(copy)?;
        let kept = self.ssa.block(original)?;
        let rows = copied.instructions();
        let model = kept.instructions().get(split..)?;
        // A block that only branches copies nothing: a jump that runs on to
        // another jump is a trampoline, not a copy.
        if model.len() < 2 {
            return None;
        }
        let at = rows.len().checked_sub(model.len())?;
        let suffix = rows.get(at..)?;
        let mut phis: Vec<SsaVarId> = kept.phi_nodes().iter().map(|phi| phi.result()).collect();
        if split > 0 {
            // Only a test is copied so: rows that run on into a jump are a
            // shared tail, which the compiler did not copy and the source may
            // well have written twice.
            model.last().and_then(|row| row.op().two_way_targets())?;
            // The rows the copy leaves out compute only what the original
            // carries on past its test - `q`'s step - with no effect, and
            // none of what the copied rows read: a test that reads a value
            // computed ahead of it asks something the copy computed
            // otherwise, or not at all.
            let ahead: BTreeSet<SsaVarId> = kept
                .instructions()
                .iter()
                .take(split)
                .flat_map(|row| row.op().defs())
                .collect();
            if kept
                .instructions()
                .iter()
                .take(split)
                .any(|row| !row.op().is_pure())
                || model
                    .iter()
                    .any(|row| row.op().uses().iter().any(|value| ahead.contains(value)))
            {
                return None;
            }
            phis.extend(ahead);
        }
        let mut mapping: BTreeMap<SsaVarId, SsaVarId> = BTreeMap::new();
        let mut reversed = false;
        for (theirs, ours) in model.iter().zip(suffix) {
            // Every field of the two operations agrees: the original's, with
            // the copy's operands in its operands' places, is the copy's,
            // where the copy branches as the original does.
            let mut expected = ours.op().clone();
            for (mine, through) in copied.successors().iter().zip(route) {
                if let Some(through) = through {
                    expected.redirect_target(*mine, *through);
                }
            }
            let bound = mapping.clone();
            let compare =
                |mapping: &mut BTreeMap<SsaVarId, SsaVarId>, theirs: &SsaOp<T>, ours: &SsaOp<T>| {
                    theirs.same_under(ours, |role, value, own| match role {
                        OperandRole::Def | OperandRole::FlagsDef => {
                            mapping.insert(value, own);
                            true
                        }
                        OperandRole::Use => self.agrees(mapping, &phis, value, own),
                    })
                };
            if compare(&mut mapping, theirs.op(), &expected) {
                continue;
            }
            // Integer constants of one width and bit pattern are one word,
            // whichever way each is written ([`Self::same_value`]).
            if let (
                SsaOp::Const {
                    dest: their_dest,
                    value: their_value,
                },
                SsaOp::Const {
                    dest: our_dest,
                    value: our_value,
                },
            ) = (theirs.op(), &expected)
                && their_value.same_integer(our_value, self.context.machine().ptr_size)
            {
                mapping.insert(*their_dest, *our_dest);
                continue;
            }
            // A branch asking the question the other way round, its targets
            // exchanged, is the same branch ([`canonical_terminator`]).
            mapping = bound;
            let (definitions, _) = self.guards();
            let alike = compare(
                &mut mapping,
                &canonical_terminator(theirs.op(), definitions),
                &canonical_terminator(&expected, definitions),
            );
            if !alike {
                return None;
            }
            reversed = true;
        }
        // A copy asking the original's question the other way round, sent
        // past a test, is a copy where it copies a computation too, as a
        // call: a test alone asking it the other way is another test, the
        // dispatch's own.
        let threaded = route.iter().any(Option::is_some);
        let computes = model.iter().any(|row| {
            row.op().two_way_targets().is_none() && !matches!(row.op(), SsaOp::Const { .. })
        });
        if reversed && threaded && !computes {
            return None;
        }
        // A phi of the block the way goes through is, on the original's way,
        // what it takes from the original.
        let entering_through = |through: usize, value: SsaVarId| -> SsaVarId {
            self.ssa
                .block(through)
                .and_then(|block| block.phi_nodes().iter().find(|phi| phi.result() == value))
                .and_then(|phi| phi.operand_from(original))
                .map_or(value, PhiOperand::value)
        };
        // Where both lead, each block takes from the copy what it takes from
        // the original, or from the block the original's way there goes
        // through.
        for (target, through) in copied.successors().into_iter().zip(route) {
            let from = through.unwrap_or(original);
            for phi in self.ssa.block(target)?.phi_nodes() {
                let mut theirs = phi.operand_from(from).map(PhiOperand::value)?;
                if let Some(through) = through {
                    theirs = entering_through(*through, theirs);
                }
                let ours = phi.operand_from(copy).map(PhiOperand::value)?;
                if !self.agrees(&mut mapping, &phis, theirs, ours) {
                    return None;
                }
            }
        }
        // Each test the original's way goes on through is decided, on the
        // copy's way in, to go on where the copy goes: asked of the copy's
        // values for the original's.
        let decided = |mapping: &BTreeMap<SsaVarId, SsaVarId>| {
            copied
                .successors()
                .into_iter()
                .zip(route)
                .all(|(target, through)| {
                    through.is_none_or(|through| {
                        let read = |value: SsaVarId| {
                            let theirs = entering_through(through, value);
                            let ours = mapping.get(&theirs).copied().unwrap_or(theirs);
                            self.on_way(copy, ours)
                        };
                        self.decides_reading(copy, through, target, read)
                    })
                })
        };
        // A phi of the original that no copied row reads and no block the
        // copy branches to takes from it is read, on the copy's way, by
        // nothing but the tests the copy jumped past: every other reader is
        // beneath the original, off that way. On the copy's way it may hold
        // any value defined where the copy ends that those tests are decided
        // by: one of its own values, as `cdcmd`'s `lim` in dash, read only by
        // the `if (new > lim)` its copied `strtok` jumps past.
        let open: Vec<(SsaVarId, Vec<SsaVarId>)> = kept
            .phi_nodes()
            .iter()
            .filter(|phi| !mapping.contains_key(&phi.result()))
            .map(|phi| {
                let defined: BTreeSet<SsaVarId> = phi
                    .operands()
                    .iter()
                    .map(PhiOperand::value)
                    .filter(|value| self.available(*value, copy, at))
                    .collect();
                (phi.result(), defined.into_iter().collect())
            })
            .filter(|(_, defined): &(SsaVarId, Vec<SsaVarId>)| !defined.is_empty())
            .collect();
        // Past so many ways, the phis stay unbound, as no test reads them.
        let choices = open
            .iter()
            .try_fold(1usize, |count, (_, defined)| {
                count.checked_mul(defined.len())
            })
            .filter(|count| *count <= OPEN_PHI_CHOICES);
        let open = if choices.is_some() { open } else { Vec::new() };
        let choices = choices.unwrap_or(1);
        mapping = (0..choices).find_map(|choice| {
            let mut trial = mapping.clone();
            let mut left = choice;
            for (phi, defined) in &open {
                let value = defined.get(left.checked_rem(defined.len())?)?;
                trial.insert(*phi, *value);
                left = left.checked_div(defined.len())?;
            }
            decided(&trial).then_some(trial)
        })?;
        // What the copy defines is read only by its own later rows and by the
        // operands the blocks it branches to take from it: a block the copy
        // dominates - the loop's body, the original itself - may read it too,
        // and would read nothing once it is gone.
        let successors = copied.successors();
        for row in suffix {
            for value in row.op().defs() {
                let inside = self
                    .ssa
                    .variable(value)?
                    .uses()
                    .iter()
                    .filter(|site| !site.is_phi_operand() && site.block() == copy)
                    .count();
                let carried: usize = successors
                    .iter()
                    .filter_map(|target| self.ssa.block(*target))
                    .flat_map(|block| block.phi_nodes())
                    .filter(|phi| phi.operand_from(copy).map(PhiOperand::value) == Some(value))
                    .count();
                if self.ssa.variable(value)?.uses().len() != inside.saturating_add(carried) {
                    return None;
                }
            }
        }
        // What a split carries: the values ahead of it something past it
        // reads.
        let carried: Vec<SsaVarId> = if split == 0 {
            Vec::new()
        } else {
            phis.iter()
                .copied()
                .filter(|value| {
                    self.ssa.variable(*value).is_some_and(|var| {
                        var.uses().iter().any(|site| {
                            site.block() != original
                                || site.is_phi_operand()
                                || site.index() >= split
                        })
                    })
                })
                .collect()
        };
        let entered = if split == 0 { &phis } else { &carried };
        // Each phi of the block the copy jumps to has a value along the new
        // edge, defined where the copy's remaining code ends.
        let mut entering = Vec::with_capacity(entered.len());
        for phi in entered {
            let value = *mapping.get(phi)?;
            if !self.available(value, copy, at) {
                return None;
            }
            entering.push(value);
        }
        if !self.ssa.retarget_respects_regions(
            copy,
            copied.successors().first().copied()?,
            original,
        ) {
            return None;
        }
        Some(Found {
            copy,
            original,
            at,
            split,
            carried,
            entering,
        })
    }

    /// Whether the original's `value` and the copy's `own` are one under
    /// `mapping`, binding an original phi met for the first time.
    fn agrees(
        &self,
        mapping: &mut BTreeMap<SsaVarId, SsaVarId>,
        phis: &[SsaVarId],
        value: SsaVarId,
        own: SsaVarId,
    ) -> bool {
        if let Some(mapped) = mapping.get(&value) {
            return self.same_value(*mapped, own);
        }
        if phis.contains(&value) {
            mapping.insert(value, own);
            return true;
        }
        self.same_value(value, own)
    }

    /// The value `value` is on the way into `block`: where it is a phi of a
    /// block dominating `block` on no cycle, the operand that way takes
    /// ([`GuardFacts::phi_on_way`]), read the same way in turn; else
    /// `value`.
    fn on_way(&self, block: usize, value: SsaVarId) -> SsaVarId {
        let (definitions, guards) = self.guards();
        let mut value = value;
        for _ in 0..self.ssa.block_count() {
            let Some(at) = self
                .ssa
                .variable(value)
                .filter(|var| var.def_site().is_phi())
                .map(|var| var.def_site().block())
            else {
                break;
            };
            if self.cycles().get(at).copied().flatten().is_some()
                || !self
                    .dominators()
                    .dominates(NodeId::new(at), NodeId::new(block))
            {
                break;
            }
            let Some(phi) = self
                .ssa
                .block(at)
                .and_then(|joined| joined.phi_nodes().iter().find(|phi| phi.result() == value))
            else {
                break;
            };
            match guards.phi_on_way(self.ssa, block, at, phi, definitions) {
                Some(taken) if taken != value => value = taken,
                _ => break,
            }
        }
        value
    }

    /// Whether `left` and `right` hold one value: one variable, constants
    /// alike of one type, integer constants of one width and one bit
    /// pattern (an operation's signedness is its own, as `BranchCmp`'s
    /// `unsigned` says: a `0` written unsigned and a `0` written signed are
    /// one word), or loads of one cell in one state.
    fn same_value(&self, left: SsaVarId, right: SsaVarId) -> bool {
        if left == right {
            return true;
        }
        let constant = |value: SsaVarId| match self.ssa.get_definition(value) {
            Some(SsaOp::Const { value: held, .. }) => {
                Some((held.clone(), self.ssa.variable(value)?.var_type().clone()))
            }
            _ => None,
        };
        if let (Some(first), Some(second)) = (constant(left), constant(right)) {
            return first == second
                || first
                    .0
                    .same_integer(&second.0, self.context.machine().ptr_size);
        }
        self.guards().0.same_read(left, right)
    }

    /// The first join whose ways in are copies of one test
    /// ([`Self::copies_into`]).
    fn joined_copies(&self) -> Option<JoinedCopies<T>> {
        (0..self.ssa.block_count()).find_map(|join| self.copies_into(join))
    }

    /// The copies of one test `join` is entered from, where they become one.
    ///
    /// A compiler that copies a test into each block reaching a join, and
    /// sends each copy's other answer past the test the join goes on to where
    /// that copy's way has answered it, leaves the join entered from two or
    /// more blocks asking one question of one value: `binop2` in dash computes
    /// `a = noeval ? b : do_binop(op, a, b)` and then tests the next operator,
    /// and the three ways to the join each test `noeval`. The source tests
    /// once. They are one test, entered from each, where:
    ///
    /// - **Each way into `join` is a copy, or a block computing values only
    ///   that one copy alone enters.** The join jumps on to `onward`, which
    ///   computes values only and branches, and `onward`'s nearest dominator
    ///   dominates every copy, so whatever a block `onward` dominates reads is
    ///   defined on each copy's way too.
    /// - **The copies ask one question of one value**, the join on the same
    ///   answer.
    /// - **Each copy's other answer goes on**: to `onward` itself; through a
    ///   block computing alike what the copy's way into the join computes, to
    ///   `onward` ([`Self::alike_forwarders`]); or to where `onward`, entered
    ///   from the copy, goes on its way ([`Self::sent_onward`]).
    fn copies_into(&self, join: usize) -> Option<JoinedCopies<T>> {
        let Some(SsaOp::Jump { target: onward }) = self.ssa.block(join)?.control_terminator()
        else {
            return None;
        };
        let onward = *onward;
        if onward == join {
            return None;
        }
        let ways: BTreeSet<usize> = self
            .cfg
            .predecessors(NodeId::new(join))
            .map(NodeId::index)
            .collect();
        if ways.len() < 2 {
            return None;
        }
        let mut copies: Vec<(usize, Option<usize>)> = Vec::with_capacity(ways.len());
        for &way in &ways {
            let branches_in = self.ssa.block(way).is_some_and(|block| {
                block
                    .control_terminator()
                    .and_then(SsaOp::two_way_targets)
                    .is_some_and(|(left, right)| left != right && (left == join || right == join))
            });
            if branches_in {
                copies.push((way, None));
            } else {
                copies.push((self.forwarded_by(way, join)?, Some(way)));
            }
        }
        let tests: Vec<usize> = copies.iter().map(|(test, _)| *test).collect();
        if tests.iter().collect::<BTreeSet<_>>().len() != tests.len() {
            return None;
        }
        // One question of one value, the join on the same answer.
        let (definitions, _) = self.guards();
        let asked = |test: usize, into: usize| -> Option<(Question, bool)> {
            let terminator = self.ssa.block(test)?.control_terminator()?;
            let (true_target, false_target) = terminator.two_way_targets()?;
            let (question, takes_true) = Question::of_terminator(terminator, definitions)?;
            let joins_on = if true_target == into {
                takes_true
            } else if false_target == into {
                !takes_true
            } else {
                return None;
            };
            Some((question, joins_on))
        };
        let (&(first, first_via), rest) = copies.split_first()?;
        let (question, joins_on) = asked(first, first_via.unwrap_or(join))?;
        for &(test, via) in rest {
            let (other, other_joins_on) = asked(test, via.unwrap_or(join))?;
            if other_joins_on != joins_on || !self.asks_alike(question, other) {
                return None;
            }
        }
        // `onward` computes values only, branches, and its nearest
        // dominator dominates every copy.
        let onward_block = self.ssa.block(onward)?;
        let (last, rows) = onward_block.instructions().split_last()?;
        last.op().two_way_targets()?;
        if !rows.iter().all(|row| row.op().is_pure()) {
            return None;
        }
        let above = self.dominators().immediate_dominator(NodeId::new(onward))?;
        let reached_through_join = above.index() == join && self.read_only_through(join, onward);
        if !reached_through_join
            && !tests
                .iter()
                .all(|test| self.dominators().dominates(above, NodeId::new(*test)))
        {
            return None;
        }
        let mut forwarded = Vec::new();
        let mut sent = Vec::new();
        for (test, via) in copies {
            let (left, right) = self
                .ssa
                .block(test)?
                .control_terminator()?
                .two_way_targets()?;
            let into = via.unwrap_or(join);
            let arm = if left == into { right } else { left };
            if let Some(to_join) = via {
                let renamed = self.alike_forwarders(test, to_join, arm, onward)?;
                forwarded.push(Forwarded {
                    test,
                    to_join,
                    to_onward: arm,
                    renamed,
                });
            } else if arm != onward {
                if !self.ssa.retarget_respects_regions(test, arm, onward) {
                    return None;
                }
                let values = self.sent_onward(test, join, onward, arm)?;
                sent.push(Sent { test, arm, values });
            }
        }
        // The merged test is a block of its own: each edge into it and out
        // of it must be one the exception structure lets through.
        let merged = self.ssa.block_count();
        let respects = tests.iter().all(|test| {
            [join, onward].iter().all(|next| {
                self.ssa.retarget_respects_regions(*test, *next, merged)
                    && self.ssa.retarget_respects_regions(merged, *next, *next)
            })
        });
        let reads = self.read_alike(&tests)?;
        let mut rewritten = Vec::new();
        for &test in &tests {
            let mut terminator = self.ssa.block(test)?.control_terminator()?.clone();
            let mut chosen = reads.iter();
            let mut changed = false;
            terminator.visit_operands_mut(|role, value| {
                if role == OperandRole::Use
                    && let Some(Some(read)) = chosen.next()
                    && *value != *read
                {
                    *value = *read;
                    changed = true;
                }
            });
            if changed {
                rewritten.push((test, terminator));
            }
        }
        respects.then_some(JoinedCopies {
            tests,
            join,
            onward,
            forwarded,
            sent,
            rewritten,
        })
    }

    /// What the branches of `tests` read alike, position by position: where
    /// they read alike constants of different variables
    /// ([`Self::same_operand`]), the one whose definition every test is
    /// reached through, which the merged test then reads once rather than a
    /// phi of the constants; `None` at a position where each reads its own,
    /// which the merged test reads a phi of. `None` altogether where the
    /// branches read different numbers of values, or alike constants none of
    /// which is defined above every test.
    fn read_alike(&self, tests: &[usize]) -> Option<Vec<Option<SsaVarId>>> {
        let reads: Vec<Vec<SsaVarId>> = tests
            .iter()
            .map(|test| {
                let mut uses = Vec::new();
                self.ssa
                    .block(*test)?
                    .control_terminator()?
                    .visit_operands(|role, value| {
                        if role == OperandRole::Use {
                            uses.push(value);
                        }
                    });
                Some(uses)
            })
            .collect::<Option<_>>()?;
        let count = reads.first()?.len();
        if reads.iter().any(|uses| uses.len() != count) {
            return None;
        }
        (0..count)
            .map(|position| {
                let values: Vec<SsaVarId> = reads
                    .iter()
                    .filter_map(|uses| uses.get(position).copied())
                    .collect();
                let &first = values.first()?;
                let constants = values.iter().all(|value| {
                    matches!(self.ssa.get_definition(*value), Some(SsaOp::Const { .. }))
                        && self.same_operand(first, *value)
                });
                if values.iter().all(|value| *value == first) || !constants {
                    return Some(None);
                }
                values
                    .iter()
                    .copied()
                    .find(|value| {
                        self.ssa.variable(*value).is_some_and(|var| {
                            tests.iter().all(|test| {
                                self.dominators().dominates(
                                    NodeId::new(var.def_site().block()),
                                    NodeId::new(*test),
                                ) && var.def_site().block() != *test
                            })
                        })
                    })
                    .map(Some)
            })
            .collect()
    }

    /// Whether every value `join` defines is read only in `join` itself or
    /// carried by `onward`'s phis: once the copies are one, `join` dominates
    /// nothing, and what it computes reaches past it through those phis
    /// alone.
    fn read_only_through(&self, join: usize, onward: usize) -> bool {
        let Some(block) = self.ssa.block(join) else {
            return false;
        };
        let defined = block
            .phi_nodes()
            .iter()
            .map(|phi| phi.result())
            .chain(block.instructions().iter().flat_map(|row| row.op().defs()));
        defined.into_iter().all(|value| {
            self.ssa.variable(value).is_some_and(|var| {
                var.uses().iter().all(|site| {
                    site.block() == join && !site.is_phi_operand()
                        || site.is_phi_operand() && site.block() == onward
                })
            })
        })
    }

    /// The copy `way` is the way into `join` of: `way` computes values
    /// only, holds no phi, jumps to `join`, and is entered from one block
    /// alone, which branches to it.
    fn forwarded_by(&self, way: usize, join: usize) -> Option<usize> {
        let block = self.ssa.block(way)?;
        let (last, rows) = block.instructions().split_last()?;
        if !block.phi_nodes().is_empty()
            || !matches!(last.op(), SsaOp::Jump { target } if *target == join)
            || !rows.iter().all(|row| row.op().is_pure())
        {
            return None;
        }
        let mut entering = self.cfg.predecessors(NodeId::new(way)).map(NodeId::index);
        let (Some(test), None) = (entering.next(), entering.next()) else {
            return None;
        };
        self.ssa
            .block(test)?
            .control_terminator()?
            .two_way_targets()
            .filter(|(left, right)| left != right && (*left == way || *right == way))
            .map(|_| test)
    }

    /// Whether `left` and `right` are one operand of a comparison: one value
    /// ([`Self::same_value`]), or integer constants of the same bits at the
    /// same width ([`ConstValue::same_integer`]), which a comparison, its
    /// signedness its own, reads alike (`noeval == 0` against an unsigned and
    /// a signed zero).
    fn same_operand(&self, left: SsaVarId, right: SsaVarId) -> bool {
        if self.same_value(left, right) {
            return true;
        }
        matches!(
            (self.ssa.get_definition(left), self.ssa.get_definition(right)),
            (Some(SsaOp::Const { value: one, .. }), Some(SsaOp::Const { value: other, .. }))
                if one.same_integer(other, self.context.machine().ptr_size)
        )
    }

    /// Whether `one` and `other` ask one question of one value: the same
    /// relation, of operands [`Self::same_operand`] finds one.
    fn asks_alike(&self, one: Question, other: Question) -> bool {
        match (one, other) {
            (
                Question::Compare {
                    left,
                    right,
                    cmp,
                    unsigned,
                },
                Question::Compare {
                    left: other_left,
                    right: other_right,
                    cmp: other_cmp,
                    unsigned: other_unsigned,
                },
            ) => {
                cmp == other_cmp
                    && unsigned == other_unsigned
                    && self.same_operand(left, other_left)
                    && self.same_operand(right, other_right)
            }
            (
                Question::Flags { flags, condition },
                Question::Flags {
                    flags: other_flags,
                    condition: other_condition,
                },
            ) => condition == other_condition && self.same_value(flags, other_flags),
            (Question::Value(held), Question::Value(other_held)) => {
                self.same_value(held, other_held)
            }
            _ => false,
        }
    }

    /// The values `to_onward` defines, each with the one `to_join` defines in
    /// its place, where the two compute alike: each is entered from `test`
    /// alone, holds no phi, computes values only and jumps on, `to_join` to
    /// the join and `to_onward` to `onward`, one operation for one.
    fn alike_forwarders(
        &self,
        test: usize,
        to_join: usize,
        to_onward: usize,
        onward: usize,
    ) -> Option<Vec<(SsaVarId, SsaVarId)>> {
        let entered_alone = |block: usize| {
            let mut entering = self.cfg.predecessors(NodeId::new(block)).map(NodeId::index);
            matches!((entering.next(), entering.next()), (Some(only), None) if only == test)
        };
        let (into_join, into_onward) = (self.ssa.block(to_join)?, self.ssa.block(to_onward)?);
        let (theirs_last, theirs) = into_onward.instructions().split_last()?;
        let (_, mine) = into_join.instructions().split_last()?;
        if !entered_alone(to_onward)
            || !into_onward.phi_nodes().is_empty()
            || !matches!(theirs_last.op(), SsaOp::Jump { target } if *target == onward)
            || theirs.len() != mine.len()
            || !theirs.iter().all(|row| row.op().is_pure())
        {
            return None;
        }
        let mut renamed: Vec<(SsaVarId, SsaVarId)> = Vec::with_capacity(theirs.len());
        for (their_row, my_row) in theirs.iter().zip(mine) {
            let mut uses: Vec<(SsaVarId, SsaVarId)> = Vec::new();
            let mut defs: Vec<(SsaVarId, SsaVarId)> = Vec::new();
            let alike = their_row.op().same_under(my_row.op(), |role, their, my| {
                match role {
                    OperandRole::Use => uses.push((their, my)),
                    OperandRole::Def | OperandRole::FlagsDef => defs.push((their, my)),
                }
                true
            });
            let bound = |value: SsaVarId| {
                renamed
                    .iter()
                    .find(|(their, _)| *their == value)
                    .map(|(_, my)| *my)
            };
            if !alike
                || !uses.iter().all(|(their, my)| {
                    bound(*their).map_or_else(|| self.same_value(*their, *my), |own| own == *my)
                })
            {
                return None;
            }
            renamed.extend(defs);
        }
        Some(renamed)
    }

    /// What each phi of `onward` takes from `test`, where `test`'s other
    /// answer, which goes to `arm`, goes where `onward` entered from `test`
    /// would go on its way: `onward`'s branch, asked of what its phis take
    /// from `test`, is answered by what `test`'s edge to `arm` knows
    /// ([`GuardFacts::edge_answer`]), and the blocks from there to `arm`
    /// compute values and loads only, jump on, and carry into `arm`'s phis
    /// what `test` carries ([`Self::carries`]).
    ///
    /// A phi of `onward` takes from `test` what it takes from `join` where
    /// that is a phi of `join` (its value from `test`) or defined above
    /// `test`; otherwise, what `arm`'s phis read of it on the way, a value
    /// `join` computes standing for one `test` carries past it.
    fn sent_onward(
        &self,
        test: usize,
        join: usize,
        onward: usize,
        arm: usize,
    ) -> Option<Vec<SsaVarId>> {
        let (definitions, guards) = self.guards();
        let onward_block = self.ssa.block(onward)?;
        let join_phis = self.ssa.block(join)?.phi_nodes();
        // What `onward`'s phis hold on `test`'s way, where known.
        let mut way = Way {
            blocks: Vec::new(),
            test,
            held: BTreeMap::new(),
            pending: BTreeSet::new(),
            proven: BTreeSet::new(),
        };
        for phi in onward_block.phi_nodes() {
            let from_join = phi.operand_from(join)?.value();
            let joined = join_phis
                .iter()
                .find(|joined| joined.result() == from_join)
                .and_then(|joined| joined.operand_from(test))
                .map(PhiOperand::value);
            let above = self.ssa.variable(from_join).is_some_and(|var| {
                self.dominators()
                    .dominates(NodeId::new(var.def_site().block()), NodeId::new(test))
                    && var.def_site().block() != join
            });
            match joined {
                Some(value) => {
                    way.held.insert(phi.result(), value);
                }
                None if above => {
                    way.held.insert(phi.result(), from_join);
                }
                None => {
                    way.pending.insert(phi.result());
                }
            }
        }
        // Where `onward` goes on `test`'s way.
        let terminator = onward_block.control_terminator()?;
        let (true_target, false_target) = terminator.two_way_targets()?;
        let (question, takes_true) = Question::of_terminator(terminator, definitions)?;
        let reads_pending = Cell::new(false);
        let asked = question.reading(|value| {
            if way.pending.contains(&value) {
                reads_pending.set(true);
            }
            way.held.get(&value).copied().unwrap_or(value)
        });
        if reads_pending.get() {
            return None;
        }
        let holds = guards.edge_answer(self.ssa, test, arm, asked, definitions)?;
        let mut at = if holds == takes_true {
            true_target
        } else {
            false_target
        };
        // The way on to `arm`: blocks of values and loads, each jumping on,
        // each phi taking what the block before carried.
        let mut before = onward;
        while at != arm {
            if way.blocks.len() >= ONWARD_BLOCKS
                || at == onward
                || at == join
                || at == test
                || way.blocks.contains(&at)
            {
                return None;
            }
            let block = self.ssa.block(at)?;
            for phi in block.phi_nodes() {
                let carried = resolve_chain(&way.held, phi.operand_from(before)?.value());
                way.held.insert(phi.result(), carried);
            }
            let (last, rows) = block.instructions().split_last()?;
            let SsaOp::Jump { target } = last.op() else {
                return None;
            };
            if !rows
                .iter()
                .all(|row| row.op().is_pure() || matches!(row.op(), SsaOp::LoadIndirect { .. }))
            {
                return None;
            }
            way.blocks.push(at);
            before = at;
            at = *target;
        }
        // `arm` takes what `test` carried into it.
        for phi in self.ssa.block(arm)?.phi_nodes() {
            let expected = phi.operand_from(test)?.value();
            let carried = phi.operand_from(before)?.value();
            if !self.carries(&mut way, carried, expected, CARRIED_DEPTH) {
                return None;
            }
        }
        // Every load on the way reads what a load above `test` read: it
        // faults nowhere the path did not.
        let loads_proven = way.blocks.iter().all(|&block| {
            self.ssa.block(block).is_some_and(|block| {
                block.instructions().iter().all(|row| match row.op() {
                    SsaOp::LoadIndirect { dest, .. } => way.proven.contains(dest),
                    _ => true,
                })
            })
        });
        if !loads_proven || !way.pending.is_empty() {
            return None;
        }
        onward_block
            .phi_nodes()
            .iter()
            .map(|phi| way.held.get(&phi.result()).copied())
            .collect()
    }

    /// Whether `carried`, as the way from the onward test computes it, is
    /// `expected`, which `test` carries: one value ([`Self::same_value`]), a
    /// phi of the onward test not yet known, which takes `expected` then,
    /// one operation for one over operands carried in turn (either order
    /// for an operation whose operands commute), or a load of a cell carried
    /// in turn that the load `expected` names read, in the state memory
    /// leaves the copy in ([`Self::unchanged_to`]). Only values the way
    /// itself computes are read through; every load proven so is recorded on
    /// the way.
    fn carries(&self, way: &mut Way, carried: SsaVarId, expected: SsaVarId, depth: usize) -> bool {
        let carried = resolve_chain(&way.held, carried);
        if way.pending.remove(&carried) {
            way.held.insert(carried, expected);
            return true;
        }
        let on_way = self
            .ssa
            .variable(carried)
            .is_some_and(|var| way.blocks.contains(&var.def_site().block()));
        if self.same_value(carried, expected) {
            // A load of one cell in one state with `expected`'s reads what was
            // read: it is proven by being one value with it.
            if on_way
                && matches!(
                    self.ssa.get_definition(carried),
                    Some(SsaOp::LoadIndirect { .. })
                )
            {
                way.proven.insert(carried);
            }
            return true;
        }
        if depth == 0 || !on_way {
            return false;
        }
        let (Some(mine), Some(theirs)) = (
            self.ssa.get_definition(carried),
            self.ssa.get_definition(expected),
        ) else {
            return false;
        };
        let load = matches!(mine, SsaOp::LoadIndirect { .. });
        if load && !self.unchanged_to(expected, way.test) {
            return false;
        }
        if !load && !mine.is_pure() {
            return false;
        }
        let pairs = |op: &SsaOp<T>| {
            let mut uses: Vec<(SsaVarId, SsaVarId)> = Vec::new();
            op.same_under(theirs, |role, own, their| {
                if role == OperandRole::Use {
                    uses.push((own, their));
                }
                true
            })
            .then_some(uses)
        };
        let readings: Vec<Vec<(SsaVarId, SsaVarId)>> = [Some(mine.clone()), mine.commuted()]
            .into_iter()
            .flatten()
            .filter_map(|op| pairs(&op))
            .collect();
        for uses in readings {
            let mut tried = way.clone();
            let all = uses.iter().all(|(own, their)| {
                self.carries(&mut tried, *own, *their, depth.saturating_sub(1))
            });
            if all {
                if load {
                    tried.proven.insert(carried);
                }
                *way = tried;
                return true;
            }
        }
        false
    }

    /// Whether the load defining `value` lies above `test` and reads its
    /// cell in the state memory leaves `test` in: nothing writes the cell
    /// between them on any way.
    fn unchanged_to(&self, value: SsaVarId, test: usize) -> bool {
        let Some(site) = self.ssa.variable(value).map(|var| var.def_site()) else {
            return false;
        };
        let Some(instruction) = site.index() else {
            return false;
        };
        if !self
            .dominators()
            .dominates(NodeId::new(site.block()), NodeId::new(test))
        {
            return false;
        }
        let memory = self.memory();
        let Some(cell) = memory.operations().iter().find_map(|op| match op {
            MemoryOp::Load { location, dest, .. } if *dest == value => memory.location_id(location),
            _ => None,
        }) else {
            return false;
        };
        let leaving = self
            .ssa
            .block(test)
            .map_or(0, |block| block.instructions().len());
        let read = memory.version_before_id(cell, site.block(), instruction);
        read.is_some() && read == memory.version_before_id(cell, test, leaving)
    }

    /// Whether `original` lies on a loop that `copy`, entered from `source`,
    /// enters from outside: the blocks on a cycle through `original` that
    /// does not pass `copy` hold no `source`.
    fn enters_loop(&self, source: NodeId, copy: usize, original: usize) -> bool {
        let loops = self.loops_without(copy);
        let looped = loops.looped(original);
        looped && !(source.index() != copy && loops.together(source.index(), original))
    }

    /// The cycles of the function's graph without block `copy`
    /// ([`LoopsWithout`]), worked out the first time a copy is asked of and
    /// kept for the view: each of the copy's originals asks of the same graph.
    fn loops_without(&self, copy: usize) -> Rc<LoopsWithout> {
        if let Some(loops) = self.without.borrow().get(&copy) {
            return Rc::clone(loops);
        }
        let loops = Rc::new(LoopsWithout::of(&self.cfg, copy));
        self.without.borrow_mut().insert(copy, Rc::clone(&loops));
        loops
    }

    /// Whether `value` is defined where the code of `block` before `at`
    /// ends: ahead of that position in the block, or in a block that
    /// strictly dominates it, or as one of the function's inputs.
    fn available(&self, value: SsaVarId, block: usize, at: usize) -> bool {
        let Some(variable) = self.ssa.variable(value) else {
            return false;
        };
        let site = variable.def_site();
        if site.block() == block {
            return site.index().is_none_or(|instruction| instruction < at);
        }
        self.dominators()
            .dominates(NodeId::new(site.block()), NodeId::new(block))
    }

    /// What each block is entered knowing, over the function's definitions.
    fn guards(&self) -> &(Definitions<T>, GuardFacts) {
        self.guards.get_or_init(|| {
            let definitions = Definitions::of(self.ssa).reading(self.memory());
            let facts = GuardFacts::of(
                self.ssa,
                &self.cfg,
                &self.ssa.exception_blocks(),
                &definitions,
                self.context.machine().ptr_size,
                Some(self.memory()),
            );
            (definitions, facts)
        })
    }

    /// The cycle each block lies on, by the index of its strongly connected
    /// component, a block that branches to itself on one of its own; `None`
    /// for a block on none.
    fn cycles(&self) -> &[Option<usize>] {
        self.cycles.get_or_init(|| {
            let mut cycles = vec![None; self.ssa.block_count()];
            for (index, component) in strongly_connected_components(&self.cfg)
                .into_iter()
                .enumerate()
            {
                let looped = component.len() > 1
                    || component
                        .first()
                        .is_some_and(|node| self.cfg.successors(*node).any(|next| next == *node));
                if looped {
                    for node in component {
                        if let Some(slot) = cycles.get_mut(node.index()) {
                            *slot = Some(index);
                        }
                    }
                }
            }
            cycles
        })
    }

    /// The blocks of each cycle ([`Self::cycles`]), ascending: what a loop
    /// holds, without a walk of every block per loop asked of.
    fn cycle_members(&self) -> &FxHashMap<usize, Vec<usize>> {
        self.members.get_or_init(|| {
            let mut members: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
            for (block, cycle) in self.cycles().iter().enumerate() {
                if let Some(cycle) = cycle {
                    members.entry(*cycle).or_default().push(block);
                }
            }
            members
        })
    }

    /// Dominance from the function's entry.
    fn dominators(&self) -> &DominatorTree {
        self.dominators
            .get_or_init(|| compute_dominators(&self.cfg, NodeId::new(0)))
    }

    /// The function's memory.
    fn memory(&self) -> &MemorySsa<T> {
        self.memory
            .get_or_init(|| MemorySsa::build(self.ssa, self.context))
    }
}

/// Pass that merges a block a compiler duplicated back into the block it
/// copies.
///
/// A block entered from
/// outside a loop whose instructions end in a copy of a block on it, with the
/// same branch to the same blocks, jumps to that block instead, which takes
/// the values the copy computed with through its phis.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - retargets the copy and changes the
/// predecessors of the blocks both branch to.
#[derive(Debug, Default, Clone, Copy)]
pub struct DuplicateMergingPass;

impl DuplicateMergingPass {
    /// Creates a new duplicate-merging pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for DuplicateMergingPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "duplicate-merging"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Merge blocks a compiler duplicated back into the block they copy"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Merges every block a compiler duplicated back into the block it copies.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// Whether any block was merged.
    ///
    /// # Errors
    ///
    /// Returns an error if an edit session fails.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let context = env.context();
        let mut changed = false;
        let mut search = Search::of(ssa);
        // One view of the function - its memory and guard facts among what it
        // reads - serves every phase until an edit is made, and is read afresh
        // only after one.
        loop {
            let Some(step) = search.next(&Copies::of(ssa, context)) else {
                break;
            };
            // Each edit sets every phi operand the new and the removed edges
            // carry, so the session is repaired once it is made; verified, so an
            // edit that left anything out is refused rather than kept.
            let options = SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(SsaRollbackPolicy::OnFailure);
            let (location, message) = match step {
                Step::Merge(found) => {
                    ssa.edit(options, |editor| {
                        let original = if found.split == 0 {
                            found.original
                        } else {
                            editor.split_off(found.original, found.split, &found.carried)?
                        };
                        editor.merge_duplicate(found.copy, original, found.at, &found.entering)?;
                        editor.declare_repaired();
                        Ok(())
                    })?;
                    (
                        found.copy,
                        format!(
                            "unthreaded: B{} copied B{} and now jumps to it",
                            found.copy, found.original
                        ),
                    )
                }
                Step::Tests(tests) => {
                    let block = Cell::new(0usize);
                    ssa.edit(options, |editor| {
                        block.set(editor.merge_tests(&tests)?);
                        editor.declare_repaired();
                        Ok(())
                    })?;
                    (
                        block.get(),
                        format!(
                            "unthreaded: the loop tests of {tests:?} are B{}",
                            block.get()
                        ),
                    )
                }
                Step::Joined(found) => {
                    let block = Cell::new(0usize);
                    ssa.edit(options, |editor| {
                        for (test, terminator) in &found.rewritten {
                            editor.replace_terminator_op(*test, terminator.clone())?;
                        }
                        for forwarded in &found.forwarded {
                            editor.merge_forwarders(
                                forwarded.test,
                                forwarded.to_join,
                                forwarded.to_onward,
                                &forwarded.renamed,
                            )?;
                        }
                        for sent in &found.sent {
                            editor.send_onward(sent.test, sent.arm, found.onward, &sent.values)?;
                        }
                        block.set(editor.merge_tests(&found.tests)?);
                        editor.declare_repaired();
                        Ok(())
                    })?;
                    (
                        block.get(),
                        format!(
                            "unthreaded: the copies {:?} of one test into B{}'s ways in are B{}",
                            found.tests,
                            found.join,
                            block.get()
                        ),
                    )
                }
                Step::Unrotate(found) => {
                    ssa.edit(options, |editor| {
                        editor.unrotate(
                            found.guard,
                            found.forwarder,
                            found.latch,
                            found.header,
                            &found.entering,
                        )?;
                        editor.declare_repaired();
                        Ok(())
                    })?;
                    (
                        found.guard,
                        format!(
                            "unthreaded: B{} copied the test B{} of its rotated loop, which it now enters",
                            found.guard, found.latch
                        ),
                    )
                }
                Step::Reroute {
                    to,
                    through,
                    rerouted,
                } => {
                    let Rerouted {
                        froms,
                        joined,
                        rejoined,
                        via,
                    } = rerouted;
                    ssa.edit(options, |editor| {
                        editor.route_through(&froms, to, through, via, &joined, &rejoined)?;
                        editor.declare_repaired();
                        Ok(())
                    })?;
                    (
                        froms.first().copied().unwrap_or(through),
                        format!(
                            "unthreaded: the edges of {froms:?} to B{to} go through B{through}"
                        ),
                    )
                }
            };
            env.record(EventKind::ControlFlowRestructured)
                .location(location)
                .message(message);
            changed = true;
        }
        Ok(changed)
    }
}

/// `terminator` stated one way of each pair of branches that ask one
/// question: an integer comparison by `==`, `<` or `<=`, a flags branch by
/// a condition that holds when its flag is set, the other relation's
/// branch read negated with its targets exchanged. A loop's test at its
/// top asks `hi <= lo` to leave, and the copy at the end of a way round
/// asks `hi > lo` to stay (`bsearch` inlined in dash's `findstring`): one
/// test, which [`Copies::shape`] then reads alike. A float comparison stays
/// as it is, since its relations do not partition a NaN.
fn canonical_terminator<T: Target>(
    terminator: &SsaOp<T>,
    definitions: &Definitions<T>,
) -> SsaOp<T> {
    match *terminator {
        SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned,
            true_target,
            false_target,
        } if matches!(cmp, CmpKind::Ne | CmpKind::Ge | CmpKind::Gt)
            && !definitions.maybe_float(left)
            && !definitions.maybe_float(right) =>
        {
            SsaOp::BranchCmp {
                left,
                right,
                cmp: cmp.negated(),
                unsigned,
                true_target: false_target,
                false_target: true_target,
            }
        }
        SsaOp::BranchFlags {
            flags,
            condition,
            true_target,
            false_target,
        } if matches!(
            condition,
            FlagCondition::NotCarry
                | FlagCondition::NotZero
                | FlagCondition::NotOverflow
                | FlagCondition::Positive
                | FlagCondition::ParityOdd
        ) =>
        {
            SsaOp::BranchFlags {
                flags,
                condition: condition.negated(),
                true_target: false_target,
                false_target: true_target,
            }
        }
        _ => terminator.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::DirectedGraph,
        ir::{
            instruction::SsaInstruction,
            phi::PhiNode,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    /// The first version of local `idx`, defined at `(block, instr)`.
    fn var(ssa: &mut SsaFunction<MockTarget>, idx: u16, block: usize, instr: usize) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            1,
            DefSite::instruction(block, instr),
            MockType::I32,
        )
    }

    /// The block `index` holding `ops` in order.
    fn block(index: usize, ops: Vec<SsaOp<MockTarget>>) -> SsaBlock<MockTarget> {
        let mut block = SsaBlock::new(index);
        for op in ops {
            block.add_instruction(SsaInstruction::synthetic(op));
        }
        block
    }

    /// `dest = value` and `test = n < dest`, branching to `on` or `off`.
    fn test(
        dest: SsaVarId,
        value: i32,
        test: SsaVarId,
        n: SsaVarId,
        [on, off]: [usize; 2],
    ) -> Vec<SsaOp<MockTarget>> {
        vec![
            SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            },
            SsaOp::Clt {
                dest: test,
                left: n,
                right: dest,
                unsigned: false,
                signaling: false,
            },
            SsaOp::Branch {
                condition: test,
                true_target: on,
                false_target: off,
            },
        ]
    }

    /// `ssa` with its blocks, checked well formed.
    fn finished(
        mut ssa: SsaFunction<MockTarget>,
        blocks: Vec<SsaBlock<MockTarget>>,
    ) -> SsaFunction<MockTarget> {
        for block in blocks {
            ssa.add_block(block);
        }
        ssa.recompute_uses();
        assert_eq!(ssa.validate(), Ok(()), "the fixture is valid: {ssa:?}");
        ssa
    }

    /// A join `3` that three tests reach, `0` straight and `2` past `1`:
    ///
    /// ```text
    /// 0: if (n < 1) goto 1 else goto 3
    /// 1: if (n < 2) goto 2 else goto 3
    /// 2: if (n < 3) goto 3 else goto 4
    /// 3: return n
    /// 4: return n
    /// ```
    ///
    /// `0`'s edge to `3` and `2`'s test fit everything a threaded edge is
    /// asked but the ways into `3`.
    fn three_ways_in() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(1, 8);
        let n = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let tests = [(0, 1, [1, 3]), (1, 2, [2, 3]), (2, 3, [3, 4])].map(|(at, value, to)| {
            let local = u16::try_from(at).unwrap_or_default().saturating_mul(2);
            let bound = var(&mut ssa, local.saturating_add(1), at, 0);
            let asked = var(&mut ssa, local.saturating_add(2), at, 1);
            block(at, test(bound, value, asked, n, to))
        });
        let returns = [3, 4].map(|at| block(at, vec![SsaOp::returns(Some(n))]));
        finished(ssa, tests.into_iter().chain(returns).collect())
    }

    /// A loop `1` that `0` may skip, whose way out joins the skip at `4`
    /// through `3`, a block that only jumps:
    ///
    /// ```text
    /// 0: if (n < 1) goto 1 else goto 3
    /// 1: i = phi(0: n, 2: j); if (n < 10) goto 2 else goto 4
    /// 2: j = i + n; goto 1
    /// 3: goto 4
    /// 4: return n
    /// ```
    ///
    /// `1`'s edge to `4` leaves the loop, and the way it would have gone
    /// through `3` asks nothing; nor does `2`, the loop's way round.
    fn loop_past_a_jump() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(1, 8);
        let n = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let one = var(&mut ssa, 1, 0, 0);
        let skip = var(&mut ssa, 2, 0, 1);
        let i = ssa.create_variable(VariableOrigin::Local(3), 1, DefSite::phi(1), MockType::I32);
        let ten = var(&mut ssa, 4, 1, 0);
        let again = var(&mut ssa, 5, 1, 1);
        let j = var(&mut ssa, 6, 2, 0);
        let mut header = block(1, test(ten, 10, again, n, [2, 4]));
        let mut phi = PhiNode::new(i, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(n, 0));
        phi.add_operand(PhiOperand::new(j, 2));
        header.add_phi(phi);
        let round = block(
            2,
            vec![
                SsaOp::Add {
                    dest: j,
                    left: i,
                    right: n,
                    flags: None,
                },
                SsaOp::Jump { target: 1 },
            ],
        );
        let blocks = vec![
            block(0, test(one, 1, skip, n, [1, 3])),
            header,
            round,
            block(3, vec![SsaOp::Jump { target: 4 }]),
            block(4, vec![SsaOp::returns(Some(n))]),
        ];
        finished(ssa, blocks)
    }

    /// A loop `1` whose way round `2` tests whether to go round again, its
    /// one test into the header:
    ///
    /// ```text
    /// 0: goto 1
    /// 1: i = phi(0: n, 2: j); if (n < 10) goto 2 else goto 3
    /// 2: j = i + n; if (n < 5) goto 1 else goto 3
    /// 3: return n
    /// ```
    fn one_test_round() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(1, 8);
        let n = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let i = ssa.create_variable(VariableOrigin::Local(1), 1, DefSite::phi(1), MockType::I32);
        let ten = var(&mut ssa, 2, 1, 0);
        let again = var(&mut ssa, 3, 1, 1);
        let j = var(&mut ssa, 4, 2, 0);
        let five = var(&mut ssa, 5, 2, 1);
        let round_again = var(&mut ssa, 6, 2, 2);
        let mut header = block(1, test(ten, 10, again, n, [2, 3]));
        let mut phi = PhiNode::new(i, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(n, 0));
        phi.add_operand(PhiOperand::new(j, 2));
        header.add_phi(phi);
        let mut round = vec![SsaOp::Add {
            dest: j,
            left: i,
            right: n,
            flags: None,
        }];
        round.extend(test(five, 5, round_again, n, [1, 3]));
        let blocks = vec![
            block(0, vec![SsaOp::Jump { target: 1 }]),
            header,
            block(2, round),
            block(3, vec![SsaOp::returns(Some(n))]),
        ];
        finished(ssa, blocks)
    }

    /// Whether the search over `copies` has built its memory or guard facts.
    fn read_facts(copies: &Copies<'_, '_, MockTarget>) -> bool {
        copies.memory.get().is_some() || copies.guards.get().is_some()
    }

    #[test]
    fn an_edge_into_a_join_entered_three_ways_is_judged_without_facts() {
        let ssa = three_ways_in();
        let context = AnalysisContext::new(MACHINE);
        let copies = Copies::of(&ssa, &context);
        assert!(copies.threaded_edges().is_empty());
        assert!(!read_facts(&copies), "the ways into 3 decide it");
    }

    #[test]
    fn a_way_on_that_asks_nothing_is_judged_without_facts() {
        let ssa = loop_past_a_jump();
        let context = AnalysisContext::new(MACHINE);
        let copies = Copies::of(&ssa, &context);
        assert!(copies.threaded_edges().is_empty());
        assert!(!read_facts(&copies), "3 branches on nothing");
        assert_eq!(copies.continuing_tests(), None);
        assert!(!read_facts(&copies), "2 branches on nothing");
    }

    #[test]
    fn a_loops_one_test_round_is_judged_without_facts() {
        let ssa = one_test_round();
        let context = AnalysisContext::new(MACHINE);
        let copies = Copies::of(&ssa, &context);
        assert_eq!(copies.continuing_tests(), None);
        assert!(!read_facts(&copies), "one test is no two alike");
    }

    /// **A loop the copy enters is the one the walks find.** Over graphs with
    /// nested loops, a self loop, a loop reached only through one block and
    /// a block nothing reaches, every source, copy and original agree with
    /// the two walks `enters_loop` stands in for: from the original avoiding
    /// the copy, forward and back, the original looped when a successor
    /// reaches back to it, and the source on that loop when both walks find
    /// it.
    #[test]
    fn a_loop_the_copy_enters_is_the_one_the_walks_find() {
        let graphs: [&[(usize, usize)]; 3] = [
            // 0 -> 1 -> 2 -> 1 (inner), 2 -> 3 -> 0 (outer), 3 -> 4, 4 -> 4,
            // 5 -> 2 (nothing reaches 5).
            &[
                (0, 1),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 0),
                (3, 4),
                (4, 4),
                (5, 2),
            ],
            // A diamond whose join loops back to its test through one block.
            &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 0), (4, 5)],
            // Two loops sharing a block.
            &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3)],
        ];
        let mut asked = BTreeMap::<bool, usize>::new();
        for edges in graphs {
            let nodes = edges
                .iter()
                .flat_map(|&(from, to)| [from, to])
                .max()
                .map_or(0, |last| last + 1);
            let mut graph: DirectedGraph<'_, usize, ()> = DirectedGraph::new();
            let ids: Vec<NodeId> = (0..nodes).map(|node| graph.add_node(node)).collect();
            for &(from, to) in edges {
                graph.add_edge(ids[from], ids[to], ()).expect("an edge");
            }
            let successors = |node: usize| -> Vec<usize> {
                edges
                    .iter()
                    .filter(|(from, _)| *from == node)
                    .map(|(_, to)| *to)
                    .collect()
            };
            let predecessors = |node: usize| -> Vec<usize> {
                edges
                    .iter()
                    .filter(|(_, to)| *to == node)
                    .map(|(from, _)| *from)
                    .collect()
            };
            for copy in 0..nodes {
                let loops = LoopsWithout::of(&graph, copy);
                for original in (0..nodes).filter(|original| *original != copy) {
                    let walk = |forward: bool| {
                        let mut seen = BTreeSet::new();
                        let mut stack = vec![original];
                        while let Some(node) = stack.pop() {
                            if node == copy || !seen.insert(node) {
                                continue;
                            }
                            stack.extend(if forward {
                                successors(node)
                            } else {
                                predecessors(node)
                            });
                        }
                        seen
                    };
                    let (reached, reaching) = (walk(true), walk(false));
                    let looped = successors(original)
                        .iter()
                        .any(|next| reaching.contains(next));
                    for source in 0..nodes {
                        let on_loop = reached.contains(&source) && reaching.contains(&source);
                        let walked = looped && !on_loop;
                        let found = loops.looped(original)
                            && !(source != copy && loops.together(source, original));
                        assert_eq!(
                            found, walked,
                            "{edges:?}: copy {copy}, original {original}, source {source}"
                        );
                        *asked.entry(walked).or_default() += 1;
                    }
                }
            }
        }
        assert!(
            asked.get(&true).is_some_and(|count| *count > 0)
                && asked.get(&false).is_some_and(|count| *count > 0),
            "both answers are asked: {asked:?}"
        );
    }
}
