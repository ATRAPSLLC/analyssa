//! Value Range Propagation pass - simplifies branches and comparisons
//! using sparse conditional propagation over [`ValueRange`] intervals and
//! [`Congruence`] residues.
//!
//! # Algorithm
//!
//! A worklist-based dataflow analysis shaped like Wegman-Zadeck SCCP, tracking
//! two facts per variable instead of a constant: an interval and the set of
//! residues modulo 64.
//!
//! 1. **Initialize**: every variable starts *uncomputed*, which an operand
//!    reads as `Top` and which lets a value accumulate by joining. A variable
//!    that no instruction or phi defines - an argument, a register live on
//!    entry, a local defined at the entry - is recorded as `Top` outright,
//!    since nothing will ever compute it. The entry block is marked
//!    executable, its definitions evaluated, and its CFG successors queued.
//! 2. **Propagate**: process CFG edges and SSA variables from worklists.
//!    - When a new edge is executable, the target's phis are joined over their
//!      executable predecessors, then its definitions are evaluated.
//!    - When a variable's facts change, every instruction and phi that uses it
//!      is re-evaluated; a terminator re-decides its outgoing edges.
//!    - `Branch`, `BranchCmp`, `BranchFlags` and `Switch` queue only the
//!      targets their operands admit; a terminator whose operand is not yet
//!      computed waits for it.
//!    - Facts only grow (joined, then widened once a value looks
//!      loop-carried), so the analysis terminates and a partial fact is never
//!      narrower than the final one.
//! 3. **Evaluate** each operation's result:
//!    - `Const`, `Copy`, `IntConv` (truncation, sign- and zero-extension).
//!    - `Add`, `Sub`, `Mul`, `Neg` by interval arithmetic; `Shl` and `Shr` by a
//!      constant count below the value's width.
//!    - `And`, `Or`, `Xor` bounded through non-negative operands; `Rem` by a
//!      positive constant over a non-negative dividend.
//!    - `Ceq`, `Clt`, `Cgt` decided from their operands where possible,
//!      otherwise `[0, 1]`; `ArrayLength` non-negative.
//!    - The congruence domain mirrors the arithmetic, bitwise and conversion
//!      operations modulo 64, and decides equalities no interval separates.
//!
//!    Every result is **held in its variable's window**: the machine keeps the
//!    low `w` bits, so a range is reduced to `[-2^(w-1), 2^(w-1)-1]` for a
//!    signed `w`-bit type and `[0, 2^w - 1]` for an unsigned one. Comparisons
//!    and flag tests read their operands in the window the comparison uses,
//!    so one machine value never has two readings.
//! 4. **Hold loop invariants**: where widening left a loop's phis unbounded,
//!    guess each group of them an interval (what enters the loop and what its
//!    branches compare them with), run again with the group held to it and
//!    every read of it refined by the guards it lies under, and keep the run
//!    when every trip keeps the group within its interval: a binary search's
//!    midpoint lies in `[0, 38]` of a 39-entry table, and `i < 10` from `0`
//!    holds `i` in `[0, 10]`. A branch comparing a value stepped from a phi
//!    (a copy, a conversion, a constant added) bounds the phi by the limit
//!    stepped back: `i + 1 != 512` from `0` holds `i` in `[0, 511]`.
//! 5. **Apply results**: simplify conditional terminators whose operands decide
//!    them, replace comparisons whose operands guarantee the result, and delete
//!    a conditional trap whose comparison they decide false, which never
//!    fires - only in blocks the converged analysis proved executable.
//!
//! # Limitations
//!
//! - Does not track non-numeric types (pointers, objects).
//! - A loop-carried bound that keeps growing is widened to infinity, and an
//!   infinite bound held at a known width is the whole window, so loop
//!   counters keep their width but not their trip bound, unless an interval
//!   guessed from what enters the loop and what its branches compare it with
//!   holds on every trip (step 4).
//! - A 64-bit value is held in the signed window only: an unsigned 64-bit
//!   comparison is decided only where both operands are proved non-negative.

use std::{
    collections::{BTreeMap, VecDeque},
    marker::PhantomData,
    mem,
    sync::{Arc, Mutex, PoisonError},
};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    Result,
    analysis::{
        budget::{Budget, Exhaustion, WorkBudget},
        cfg::SsaCfg,
        congruence::Congruence,
        range::ValueRange,
    },
    bitset::BitSet,
    events::EventKind,
    graph::{
        NodeId, RootedGraph, Successors,
        algorithms::{DominatorTree, compute_dominators},
    },
    ir::{
        block::SsaBlock,
        function::{SsaEditOptions, SsaFunction},
        instruction::SsaInstruction,
        integer::BitWidth,
        ops::{SsaOp, SystemOpKind},
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::SsaVarId,
    },
    passes::{
        decide::{IntShape, Known, decide_operands, flag_holds},
        guarded::{
            ComparisonOperands, GuardRefinement, Step, comparison_operands, convert,
            stated_relation,
        },
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// Converged per-variable value ranges for one function.
///
/// Only produced by [`analyze`], and only when the analysis reached its
/// fixpoint - the analysis is optimistic, so a partial run's ranges are too
/// narrow and prove things that are not true. There is deliberately no way to
/// obtain this type from an unconverged run.
#[derive(Debug, Clone, Default)]
pub struct ValueRanges {
    ranges: FxHashMap<SsaVarId, ValueRange>,
    /// Per-variable congruences, computed by the same fixpoint.
    congruences: FxHashMap<SsaVarId, Congruence>,
    /// The blocks the fixpoint proved reachable.
    ///
    /// A range is a fact about a variable *where the variable is live*, and the
    /// analysis only evaluates a block's definitions once an executable edge
    /// reaches it. A block no edge ever reached therefore has no ranges of its
    /// own - what [`Self::get`] answers for a variable used there is whatever a
    /// dominating block left in the map, which says nothing about a point
    /// control never arrives at. Carrying the set lets [`ValueRangePropagationPass`] rewrite only the
    /// subgraph the analysis actually reasoned over.
    executable: BitSet,
    /// Per block, how many exception-aware edges enter it, in the function
    /// the ranges were computed for. [`Self::range_at`] reads it to tell a
    /// guard's own arm from a critical edge; recorded once here rather than
    /// rebuilt per query, since a caller asks once per block of interest.
    predecessors: Vec<usize>,
}

impl ValueRanges {
    /// Returns the range proved for `var`, if any.
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> Option<&ValueRange> {
        self.ranges.get(&var)
    }

    /// Returns the congruence proved for `var`, if any.
    ///
    /// The independent half of [`Self::get`]: what the value is modulo a power
    /// of two, which an interval says nothing about. Computed by the same
    /// fixpoint over the same executable-edge set, so the two facts describe
    /// the same program points and may be used together (see
    /// [`Congruence`]).
    #[must_use]
    pub fn congruence(&self, var: SsaVarId) -> Option<&Congruence> {
        self.congruences.get(&var)
    }

    /// Returns `true` when the fixpoint proved `block` reachable.
    ///
    /// The ranges are facts only about the blocks this admits (see
    /// [`ValueRanges`]): a rewrite in a block it does not is a rewrite on what
    /// the analysis never evaluated.
    ///
    /// # Arguments
    ///
    /// * `block` - Index of the block to test.
    #[must_use]
    pub fn is_executable(&self, block: usize) -> bool {
        self.executable.contains_checked(block)
    }

    /// What the analysis proved about `var`, both domains at once.
    ///
    /// `None` when no range was recorded for `var`. A variable with a range
    /// but no recorded congruence is unconstrained modulo the modulus, which is
    /// what [`Congruence::TOP`] says.
    fn known(&self, var: SsaVarId) -> Option<Known<'_>> {
        Some(Known {
            range: self.ranges.get(&var)?,
            congruence: self.congruences.get(&var).copied().unwrap_or_default(),
        })
    }

    /// Returns the number of variables with a recorded range.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Returns `true` when no variable has a recorded range.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Iterates the variables with a recorded range.
    pub fn iter(&self) -> impl Iterator<Item = (SsaVarId, &ValueRange)> {
        self.ranges.iter().map(|(var, range)| (*var, range))
    }
}

/// Ranges refined by the guards a program point lies under: what the branches
/// that dominate a block, or the edge a value arrives on, prove of a value
/// there, over ranges proved for the whole function.
///
/// The finished [`ValueRanges`] answer a consumer's question at a block; the
/// analysis itself reads its own facts the same way while it checks an
/// assumed loop invariant, so the one reasoning serves both.
pub trait GuardedRanges {
    /// The range proved for `var` over the whole function, if any.
    fn proved(&self, var: SsaVarId) -> Option<&ValueRange>;

    /// The residues proved for `var` over the whole function:
    /// [`Congruence::TOP`] where none are.
    fn congruence(&self, var: SsaVarId) -> Congruence;

    /// Per block, how many edges enter it, which tells a guard's own arm from
    /// a critical edge.
    fn entering(&self) -> &[usize];

    /// Returns `var`'s range as known *at* `block`, refined by the branch
    /// guards that dominate it.
    ///
    /// The whole-function range is a single fact per variable, so it cannot say
    /// that `idx` is bounded only on the path a `cmp idx, N; ja default` guard
    /// admits - which is exactly the fact a switch dispatch depends on. This
    /// walks up the dominator tree from `block`; whenever the path arrives
    /// through one arm of a conditional branch, the comparison driving that
    /// branch is intersected into the result.
    ///
    /// Refinement only applies where the guard is *unambiguous*: every path to
    /// `block` must leave the branch through the same arm. That holds when the
    /// arm's target has the branching block as its only predecessor and
    /// dominates `block`; a target also reached some other way (a critical
    /// edge) or reached through both arms learns nothing from the branch.
    ///
    /// The bound is stated in the window `var`'s range is held in, so a guard
    /// that reads `var` differently - a signed comparison of an unsigned
    /// value, or an unsigned comparison of a possibly negative one - refines
    /// nothing.
    ///
    /// A guard bounds every value an exact step joins to what it compares: a
    /// copy, an integer conversion, or adding or subtracting a constant. A
    /// switch compares the byte it dispatches on, `126 <u c`, and indexes its
    /// table by `c` zero-extended to a word, computed before the comparison;
    /// a loop compares `i + 1` and reads `i`. The compared value's range is
    /// stepped forward into the values computed from it, and back into the
    /// values it is computed from where the step loses nothing.
    #[must_use]
    fn range_at<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        dominators: &DominatorTree,
        var: SsaVarId,
        block: usize,
    ) -> ValueRange {
        self.guarded_at(ssa, dominators, var, block)
    }
}

impl<T: Target> GuardedRanges for RangeAnalysis<T> {
    fn proved(&self, var: SsaVarId) -> Option<&ValueRange> {
        self.ranges.get(&var)
    }

    fn congruence(&self, var: SsaVarId) -> Congruence {
        self.get_congruence(var)
    }

    fn entering(&self) -> &[usize] {
        self.assumed
            .as_ref()
            .map_or(&[], |assumption| assumption.predecessors.as_slice())
    }
}

impl GuardedRanges for ValueRanges {
    fn proved(&self, var: SsaVarId) -> Option<&ValueRange> {
        self.ranges.get(&var)
    }

    fn congruence(&self, var: SsaVarId) -> Congruence {
        self.congruences.get(&var).copied().unwrap_or_default()
    }

    fn entering(&self) -> &[usize] {
        &self.predecessors
    }
}

/// How many edges of `cfg` enter each of its `blocks` blocks.
fn predecessor_counts<G: Successors>(cfg: &G, blocks: usize) -> Vec<usize> {
    let mut counts = vec![0usize; blocks];
    for block in 0..blocks {
        for successor in cfg.successors(NodeId::new(block)) {
            if let Some(count) = counts.get_mut(successor.index()) {
                *count = count.saturating_add(1);
            }
        }
    }
    counts
}

/// Computes value ranges for `ssa` without modifying it.
///
/// This is the query entry point for consumers that want the numeric facts
/// rather than the rewrites - indirect-branch recovery, bounds reasoning, and
/// tests that need to assert on a range directly. A pass reads them through
/// its environment's [`RangeFacts`] instead, which computes this once for
/// every pass that asks.
///
/// # Arguments
///
/// * `ssa` - The SSA function to analyze.
/// * `budget` - The work budget; [`Budget::RangeRounds`] bounds the fixpoint.
/// * `results` - The range a host states for the result of a call, by the
///   call's destination: what the called function returns, whoever calls
///   it. A call it states nothing of returns anything.
///
/// # Errors
///
/// The [`Exhaustion`] of [`Budget::RangeRounds`] when the fixpoint was not
/// reached within it. Nothing was proved then, which is not to say that
/// nothing is provable: a larger budget may converge.
pub fn analyze<T: Target>(
    ssa: &SsaFunction<T>,
    budget: &WorkBudget,
    results: &BTreeMap<SsaVarId, ValueRange>,
) -> Result<ValueRanges, Exhaustion> {
    let mut analysis = RangeAnalysis::new(budget.limit(Budget::RangeRounds));
    analysis.results = results.clone();
    analysis.analyze(ssa)
}

/// The proved value ranges of one function, computed on first use and shared
/// by every reader until the function changes.
///
/// The range fixpoint is the heaviest analysis the passes share - range
/// propagation, jump threading, strength reduction and switch pruning all read
/// it - and it is mostly asked of a function no pass changed since it was last
/// computed. One store per function answers all of them from one computation.
///
/// The store holds no reference to the function, because the passes that read
/// it hold the function mutably. Whoever changes the function therefore says
/// so with [`Self::invalidate`]:
/// [`run_pass_with_repair`](crate::scheduling::run_pass_with_repair) does after
/// every pass that changed it, and the
/// [`FunctionPipeline`](crate::scheduling::FunctionPipeline) after every group
/// it rolls back. A pass that changes the function and then asks again within
/// its own run gets the facts of the function as it received it: still true of
/// every value it did not redefine, and silent about the values it introduced.
///
/// An unconverged fixpoint is stored as its [`Exhaustion`], so a store asked
/// again answers the same without running it again.
#[derive(Debug)]
pub struct RangeFacts {
    /// The budget every computation of this store runs within.
    budget: WorkBudget,
    /// The current answer, or `None` until asked since the last change.
    slot: Mutex<Option<Result<Arc<ValueRanges>, Exhaustion>>>,
}

impl RangeFacts {
    /// Builds an empty store whose computations run within `budget`.
    ///
    /// # Arguments
    ///
    /// * `budget` - The work budget; [`Budget::RangeRounds`] bounds the
    ///   fixpoint.
    #[must_use]
    pub const fn new(budget: WorkBudget) -> Self {
        Self {
            budget,
            slot: Mutex::new(None),
        }
    }

    /// Returns the ranges of `ssa`, computing them unless the store holds them.
    ///
    /// `ssa` must be the function the store describes, as it stands since the
    /// last [`Self::invalidate`].
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function the store describes.
    ///
    /// # Errors
    ///
    /// The [`Exhaustion`] [`analyze`] reports for a fixpoint the budget did not
    /// let converge.
    pub fn get<T: Target>(&self, ssa: &SsaFunction<T>) -> Result<Arc<ValueRanges>, Exhaustion> {
        // A panic inside the computation leaves the slot unwritten, so a
        // poisoned lock still guards a consistent value.
        let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
        slot.get_or_insert_with(|| analyze(ssa, &self.budget, &BTreeMap::new()).map(Arc::new))
            .clone()
    }

    /// Forgets the stored answer: the function it described has changed.
    pub fn invalidate(&self) {
        *self.slot.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }

    /// Returns `true` when the store holds an answer, converged or not.
    #[must_use]
    pub fn is_computed(&self) -> bool {
        self.slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }
}

/// Folds a comparison whose operands' proved ranges force its result.
///
/// Returns the constant a `Clt`, `Cgt`, or `Ceq` must produce when the two
/// operand ranges cannot overlap the way the comparison tests - a `Clt` whose
/// left maximum is below its right minimum is always true, a `Ceq` on disjoint
/// ranges is always false, and so on. Only these three comparison ops are
/// recognized; anything else, or operands whose ranges are too wide to decide,
/// yields `None`.
///
/// # Arguments
///
/// * `op` - The instruction operation to test.
/// * `result` - Converged ranges queried for both operands.
/// * `ssa` - The function, for the operands' widths.
///
/// # Returns
///
/// `Some((dest, value))` with the boolean the comparison is proved to yield, or
/// `None` when it cannot be decided.
fn try_simplify_comparison<T: Target>(
    op: &SsaOp<T>,
    result: &ValueRanges,
    ssa: &SsaFunction<T>,
) -> Option<(SsaVarId, bool)> {
    let ComparisonOperands {
        dest,
        left,
        right,
        cmp,
        unsigned,
    } = comparison_operands(op)?;
    let value = decide_operands(
        ssa,
        cmp,
        unsigned,
        (left, result.known(left)?),
        (right, result.known(right)?),
    )?;
    Some((dest, value))
}

/// Sparse range propagation analysis engine.
///
/// Worklist algorithm similar to Wegman-Zadeck SCCP but tracking
/// [`ValueRange`] intervals instead of just constants.
struct RangeAnalysis<T: Target> {
    /// Per-variable computed ranges.
    ranges: FxHashMap<SsaVarId, ValueRange>,
    /// Per-variable computed congruences - the same fixpoint, the other domain.
    ///
    /// Kept beside the intervals rather than folded into them: they answer
    /// independent questions (magnitude against residue), they widen
    /// differently, and a value may be `Top` in one while precise in the other.
    /// One revision step writes both, so the two can never describe different
    /// iterations of the analysis.
    congruences: FxHashMap<SsaVarId, Congruence>,
    /// Set of CFG edges `(from, to)` determined to be executable.
    executable_edges: FxHashSet<(usize, usize)>,
    /// Blocks that have at least one executable incoming edge.
    executable_blocks: BitSet,
    /// Worklist of SSA variables whose range changed and need re-evaluation.
    ssa_worklist: VecDeque<SsaVarId>,
    /// Worklist of CFG edges to process for first-time execution.
    cfg_worklist: VecDeque<(usize, usize)>,
    /// Number of times each variable's range has grown because an operand
    /// moved, driving the switch to widening once a value looks loop-carried.
    update_counts: FxHashMap<SsaVarId, u32>,
    /// Rounds over the function the caller allows before giving up
    /// ([`Budget::RangeRounds`]).
    rounds: usize,
    /// The loop invariant a run is checking: phis held to assumed intervals,
    /// and what their reads are refined by ([`Self::analyze`]).
    assumed: Option<Assumption>,
    /// The guarded ranges of the assumed phis the instruction being evaluated
    /// reads, which [`Self::get_range`] answers for them.
    guarded: FxHashMap<SsaVarId, ValueRange>,
    /// The range the host states for each call's result ([`analyze`]).
    results: BTreeMap<SsaVarId, ValueRange>,
    _phantom: PhantomData<T>,
}

/// Intervals a run holds some loop phis to, with what refines their reads.
///
/// A loop-carried interval is widened to infinity: plain intervals would need
/// a round per trip. Its real bound is often an invariant intervals can only
/// check, not find: a binary search's `lo` and `hi` stay within `[0, 39]`
/// because the body runs only while `lo < hi`, so the midpoint stays below
/// `hi`. A run holding them to `[0, 39]`, each read refined by the guards it
/// lies under, shows whether the body keeps them there.
#[derive(Debug)]
struct Assumption {
    /// Each assumed phi's interval.
    held: FxHashMap<SsaVarId, ValueRange>,
    /// Every phi's result: the only values such a run widens. Every cycle
    /// passes through a phi, so that alone ends the ascent, and a value
    /// computed from a held phi keeps the bound the phi gives it.
    phis: FxHashSet<SsaVarId>,
    /// The dominator tree a read's guards are found by.
    dominators: DominatorTree,
    /// Per block, how many edges enter it.
    predecessors: Vec<usize>,
}

/// Loop phis that may share one invariant interval, and the interval guessed
/// for them.
///
/// Two phis are one group where one carries the other in, or a branch
/// compares them; the guess is the hull of what enters the loop's headers
/// from outside and of the bounded values a branch compares a member with.
#[derive(Debug, Clone)]
struct Invariant {
    /// Each member phi, by its block and result.
    members: Vec<(usize, SsaVarId)>,
    /// The interval every member is assumed to stay within.
    range: ValueRange,
}

/// One side of a relation a loop's branch states, read as the phi it is
/// stepped from ([`Step`]).
#[derive(Debug)]
struct Compared {
    /// The compared value itself.
    side: SsaVarId,
    /// The phi the steps start from, or `side` where none leads to one.
    root: SsaVarId,
    /// The steps from `root` to `side`, the first reading `root`.
    steps: Vec<Step>,
}

/// Why a value is being revised, which decides whether the revision can be
/// the ascent widening exists to cut short.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cause {
    /// A block or an edge became executable for the first time. Each does so
    /// once, so revisions for this reason are finite without any widening -
    /// however many there are. A merge with hundreds of predecessors is
    /// revised once per predecessor as they are reached, and that is breadth,
    /// not a loop.
    Reached,
    /// An operand's value moved. Around a cycle this repeats once per trip,
    /// and it is the only way a value can keep growing.
    Moved,
}

/// Growths of one variable's range, each caused by an operand moving, before
/// the analysis starts widening instead of joining.
///
/// A value that keeps growing as its operands move is loop-carried: each trip
/// round the loop grows it by one step, so taking the exact range would need as
/// many rounds as the loop has trips. Widening drops the growing bound to
/// infinity, which reaches the fixpoint in a bounded number of steps at the cost
/// of precision on that bound.
///
/// Only growth counts (see [`Cause`]). Counting every revision widened any
/// merge with more than a few predecessors, loop or not - a flattened
/// dispatcher's state, a finite set of constants, read as unbounded - and
/// counted revisions that changed nothing at all.
const WIDEN_AFTER_REVISIONS: u32 = 3;

impl<T: Target> RangeAnalysis<T> {
    /// Creates an analysis with empty state and the given step budget.
    ///
    /// The worklists, range map, and executable sets all start empty; they are
    /// populated by [`Self::initialize`] at the start of each run, so one
    /// instance can be reused across functions.
    ///
    /// # Arguments
    ///
    /// * `rounds` - Rounds over the function the caller allows;
    ///   [`Self::propagate`] scales this into a concrete step budget.
    fn new(rounds: usize) -> Self {
        Self {
            ranges: FxHashMap::default(),
            congruences: FxHashMap::default(),
            executable_edges: FxHashSet::default(),
            executable_blocks: BitSet::new(0),
            ssa_worklist: VecDeque::new(),
            cfg_worklist: VecDeque::new(),
            update_counts: FxHashMap::default(),
            rounds,
            assumed: None,
            guarded: FxHashMap::default(),
            results: BTreeMap::new(),
            _phantom: PhantomData,
        }
    }

    /// Runs the analysis over `ssa` and returns the computed ranges.
    ///
    /// Builds the exception-aware CFG once (so `catch`/`finally` edges are
    /// honored like any other successor), seeds the initial state via
    /// [`Self::initialize`], then drives both worklists to the fixpoint with
    /// [`Self::propagate`], then tightens the loop phis widening left
    /// unbounded wherever an invariant interval holds
    /// ([`Self::hold_loop_invariants`]). The ranges are moved out of `self`, so
    /// the instance is reusable only after another `initialize`.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to analyze; not modified.
    ///
    /// # Errors
    ///
    /// The [`Exhaustion`] of [`Budget::RangeRounds`] when the fixpoint was not
    /// reached: the analysis is optimistic, so what it holds then proves things
    /// that are not true, and none of it is returned. It needed at least one
    /// round more than it had.
    fn analyze(&mut self, ssa: &SsaFunction<T>) -> Result<ValueRanges, Exhaustion> {
        let eh = SsaCfg::from_ssa(ssa);
        self.initialize(ssa, &eh);
        if !self.propagate(ssa, &eh) {
            return Err(Exhaustion {
                budget: Budget::RangeRounds,
                limit: self.rounds,
                demand: self.rounds.saturating_add(1),
            });
        }
        self.hold_loop_invariants(ssa, &eh);
        Ok(ValueRanges {
            ranges: mem::take(&mut self.ranges),
            congruences: mem::take(&mut self.congruences),
            executable: mem::replace(&mut self.executable_blocks, BitSet::new(0)),
            predecessors: predecessor_counts(&eh, ssa.block_count()),
        })
    }

    /// Tightens the loop phis widening left unbounded wherever an invariant
    /// interval holds.
    ///
    /// Each group of loop phis a widened function's loops move is guessed an
    /// interval
    /// ([`Self::loop_invariants`]), and a run of the whole analysis holds every
    /// member to its group's interval, reading an assumed phi as the guards
    /// it lies under refine it. A group whose members' operands then all stay
    /// within its interval keeps it: the interval holds on entering the loop
    /// by construction and every trip keeps it, so it always holds, and every
    /// range the run drew from it is sound. A group an operand leaves is
    /// dropped and the rest run again; with none left, or a run that does not
    /// settle, the widened ranges stand.
    fn hold_loop_invariants<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        let dominators = compute_dominators(cfg, cfg.entry());
        let mut invariants = self.loop_invariants(ssa, &dominators);
        if invariants.is_empty() {
            return;
        }
        let widened = (
            mem::take(&mut self.ranges),
            mem::take(&mut self.congruences),
            mem::replace(&mut self.executable_blocks, BitSet::new(0)),
        );
        let mut assumption = Assumption {
            held: FxHashMap::default(),
            phis: ssa
                .iter_blocks()
                .flat_map(|(_, block)| block.phi_nodes().iter().map(PhiNode::result))
                .collect(),
            dominators,
            predecessors: predecessor_counts(cfg, ssa.block_count()),
        };
        loop {
            assumption.held = invariants
                .iter()
                .flat_map(|invariant| {
                    invariant
                        .members
                        .iter()
                        .map(|(_, var)| (*var, invariant.range.clone()))
                })
                .collect();
            self.assumed = Some(assumption);
            self.initialize(ssa, cfg);
            let mut settled = self.propagate(ssa, cfg);
            let mut sweeps = 0usize;
            while settled {
                self.sweep_assumed(ssa, cfg);
                if self.ssa_worklist.is_empty() && self.cfg_worklist.is_empty() {
                    break;
                }
                sweeps = sweeps.saturating_add(1);
                settled = sweeps <= self.rounds && self.propagate(ssa, cfg);
            }
            let guessed = invariants.len();
            if settled {
                self.narrow(ssa);
            }
            if settled {
                invariants.retain(|invariant| self.holds(ssa, invariant));
            }
            let Some(checked) = self.assumed.take() else {
                break;
            };
            assumption = checked;
            if settled && invariants.len() == guessed {
                return;
            }
            if !settled || invariants.is_empty() {
                break;
            }
        }
        (self.ranges, self.congruences, self.executable_blocks) = widened;
    }

    /// The groups of loop phis an invariant interval may hold, each with the
    /// interval guessed for it.
    ///
    /// Only where widening touched the function, which an exact join never
    /// needs. A group is the phis of executable blocks linked to a loop
    /// header's phi its loop moved past what enters it, two linked where one
    /// is the other's operand or a branch compares them. Its
    /// guess is the hull of the values entering the loop's headers from
    /// outside, read in the members' one window, and of the bounded values a
    /// branch compares a member with, stepped back to the member where the
    /// branch compares a value stepped from it: a binary search's `lo` and
    /// `hi` entering as `0` and `39` are guessed `[0, 39]`, `i < 10` guesses
    /// `i` within `[0, 10]`, and `i + 1 != 512` within `[0, 511]`. A group of members held in different windows, or whose
    /// guess is unbounded or one value, is none: a value a loop never changes
    /// is not widened.
    fn loop_invariants(&self, ssa: &SsaFunction<T>, dominators: &DominatorTree) -> Vec<Invariant> {
        let mut phis: FxHashMap<SsaVarId, (usize, &PhiNode)> = FxHashMap::default();
        for (block_id, block) in ssa.iter_blocks() {
            if !self.is_block_executable(block_id) {
                continue;
            }
            for phi in block.phi_nodes() {
                phis.insert(phi.result(), (block_id, phi));
            }
        }
        // Without widening every range is the exact join, and no bound is
        // missing.
        if !self
            .update_counts
            .values()
            .any(|growths| *growths > WIDEN_AFTER_REVISIONS)
        {
            return Vec::new();
        }
        // The values entering a loop header from outside it, by the edges
        // the run took.
        let entering = |block_id: usize, phi: &PhiNode| {
            phi.operands()
                .iter()
                .filter(|operand| {
                    !dominators.dominates(NodeId::new(block_id), NodeId::new(operand.predecessor()))
                        && self
                            .executable_edges
                            .contains(&(operand.predecessor(), block_id))
                })
                .map(|operand| operand.value())
                .collect::<Vec<_>>()
        };
        let header = |block_id: usize, phi: &PhiNode| {
            phi.operands().iter().any(|operand| {
                dominators.dominates(NodeId::new(block_id), NodeId::new(operand.predecessor()))
            })
        };
        // A header phi its loop moved past what enters it: what the trips
        // carry back made it wider, and widening may have made it unbounded.
        let grown = |var: &SsaVarId| {
            phis.get(var).is_some_and(|(block_id, phi)| {
                header(*block_id, phi) && {
                    let entered: Vec<ValueRange> = entering(*block_id, phi)
                        .into_iter()
                        .map(|value| self.get_range(value))
                        .collect();
                    !entered.is_empty()
                        && ValueRange::join_all(&entered).join(&self.get_range(*var))
                            != ValueRange::join_all(&entered)
                }
            })
        };
        // What links the phis, and what each branch compares.
        let mut links: FxHashMap<SsaVarId, Vec<SsaVarId>> = FxHashMap::default();
        let link =
            |links: &mut FxHashMap<SsaVarId, Vec<SsaVarId>>, one: SsaVarId, other: SsaVarId| {
                links.entry(one).or_default().push(other);
                links.entry(other).or_default().push(one);
            };
        for (result, (_, phi)) in &phis {
            for operand in phi.operands() {
                if phis.contains_key(&operand.value()) {
                    link(&mut links, *result, operand.value());
                }
            }
        }
        // The phis a loop carries: those its headers' phis reach through
        // operands, or that reach them. A phi merging values outside every
        // loop is no member of a loop's invariant, only a bound on one.
        let mut looping: FxHashSet<SsaVarId> = phis
            .iter()
            .filter(|(_, (block_id, phi))| header(*block_id, phi))
            .map(|(result, _)| *result)
            .collect();
        let mut reached: Vec<SsaVarId> = looping.iter().copied().collect();
        while let Some(phi) = reached.pop() {
            for linked in links.get(&phi).into_iter().flatten() {
                if looping.insert(*linked) {
                    reached.push(*linked);
                }
            }
        }
        // A compared value is read as the phi it is stepped from, with the
        // steps that lead from it: a latch compares `i + 1`, which bounds `i`
        // by the limit stepped back.
        let rooted = |side: SsaVarId| {
            let steps = Step::chain(ssa, side, |value| self.get_range(value).as_constant());
            let from = steps
                .iter()
                .rposition(|step| phis.contains_key(&step.from))
                .unwrap_or(steps.len());
            Compared {
                side,
                root: steps.get(from).map_or(side, |step| step.from),
                steps: steps.get(from..).map(<[Step]>::to_vec).unwrap_or_default(),
            }
        };
        let mut relations: Vec<(Compared, Compared)> = Vec::new();
        for (block_id, block) in ssa.iter_blocks() {
            if !self.is_block_executable(block_id) {
                continue;
            }
            let relation = match block.control_terminator() {
                Some(SsaOp::Branch { condition, .. }) => {
                    ssa.get_definition(*condition).and_then(stated_relation)
                }
                Some(terminator) => stated_relation(terminator),
                None => None,
            };
            if let Some((left, right, _, _)) = relation {
                let (left, right) = (rooted(left), rooted(right));
                if looping.contains(&left.root) && looping.contains(&right.root) {
                    link(&mut links, left.root, right.root);
                }
                relations.push((left, right));
            }
        }
        // Each group reached from a widened phi, once.
        let mut grouped: FxHashSet<SsaVarId> = FxHashSet::default();
        let mut invariants = Vec::new();
        let mut seeds: Vec<SsaVarId> = phis.keys().copied().filter(grown).collect();
        seeds.sort_unstable();
        for seed in seeds {
            if !grouped.insert(seed) {
                continue;
            }
            let mut members = vec![seed];
            let mut next = 0;
            while let Some(member) = members.get(next).copied() {
                next = next.saturating_add(1);
                for linked in links.get(&member).into_iter().flatten() {
                    if grouped.insert(*linked) {
                        members.push(*linked);
                    }
                }
            }
            let mut shapes = members.iter().map(|member| IntShape::of_var(ssa, *member));
            let Some(Some(shape)) = shapes.next() else {
                continue;
            };
            if shapes.any(|other| other != Some(shape)) {
                continue;
            }
            let mut parts: Vec<ValueRange> = Vec::new();
            for member in &members {
                let Some((block_id, phi)) = phis.get(member) else {
                    continue;
                };
                if !header(*block_id, phi) {
                    continue;
                }
                for value in entering(*block_id, phi) {
                    if !members.contains(&value) {
                        parts.push(shape.hold(&self.get_range(value)));
                    }
                }
            }
            for (left, right) in &relations {
                let (member, limit) =
                    match (members.contains(&left.root), members.contains(&right.root)) {
                        (true, false) => (left, right),
                        (false, true) => (right, left),
                        _ => continue,
                    };
                // The limit is the compared side as it is, not the phi it
                // may be stepped from.
                let mut range = Some(self.get_range(limit.side));
                for step in member.steps.iter().rev() {
                    range = range.and_then(|range| step.backward(&range, None));
                }
                let Some(range) = range.map(|range| shape.hold(&range)) else {
                    continue;
                };
                if range.min().is_some() && range.max().is_some() {
                    parts.push(range);
                }
            }
            let range = shape.hold(&ValueRange::join_all(&parts));
            let bounded = range
                .min()
                .zip(range.max())
                .is_some_and(|(min, max)| min < max);
            if parts.is_empty() || !bounded {
                continue;
            }
            invariants.push(Invariant {
                members: members
                    .iter()
                    .filter_map(|member| phis.get(member).map(|(block, _)| (*block, *member)))
                    .collect(),
                range,
            });
        }
        invariants
    }

    /// `value` as it arrives over the edge from `from` to `to`: as the guards
    /// dominating `from` refine it, and the branch `from` takes to `to`,
    /// each also refining the values `value` is stepped from
    /// ([`GuardRefinement::guarded_at`]).
    fn arriving(
        &self,
        ssa: &SsaFunction<T>,
        value: SsaVarId,
        from: usize,
        to: usize,
    ) -> ValueRange {
        let Some(assumption) = &self.assumed else {
            return self.get_range(value);
        };
        let taken = ssa
            .block(from)
            .and_then(SsaBlock::control_terminator)
            .and_then(SsaOp::two_way_targets)
            .filter(|(true_target, false_target)| true_target != false_target)
            .and_then(|(true_target, false_target)| {
                if true_target == to {
                    Some(true)
                } else if false_target == to {
                    Some(false)
                } else {
                    None
                }
            });
        // Each value `value` is stepped from is refined on the edge too: a
        // latch compares `i + 1` and carries it back converted.
        self.along_steps(ssa, value, |link| {
            let range = self.guarded_on(ssa, &assumption.dominators, link, from);
            match taken.and_then(|taken| self.refine_leaving(ssa, link, from, taken)) {
                Some(refined) => range.meet(&refined),
                None => range,
            }
        })
    }

    /// Revises every assumed phi and every instruction reading one, in the
    /// executable blocks, once more.
    ///
    /// A guard refines a read by another value's range, and nothing queues a
    /// read again when only that other value moves: a run settled with such a
    /// read stale is no fixpoint. Sweeping until a sweep changes nothing
    /// makes it one.
    fn sweep_assumed<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        let Some(held) = self
            .assumed
            .as_ref()
            .map(|assumption| assumption.held.keys().copied().collect::<FxHashSet<_>>())
        else {
            return;
        };
        for (block_id, block) in ssa.iter_blocks() {
            if !self.is_block_executable(block_id) {
                continue;
            }
            for phi in block.phi_nodes() {
                if held.contains(&phi.result()) {
                    self.join_phi_operands(phi, block_id, ssa, Cause::Moved, |_| true);
                }
            }
            for instr in block.instructions() {
                if instr.uses().iter().any(|var| held.contains(var)) {
                    self.update_instruction_defs(instr, block_id, ssa, Cause::Moved);
                    if instr.is_terminator() {
                        self.propagate_outgoing_edges(block_id, block, ssa, cfg);
                    }
                }
            }
        }
    }

    /// What `phi`'s executable operands bring it, each as it arrives
    /// ([`Self::arriving`]), joined and read in the phi's window. `None` while
    /// no edge into its block is executable.
    fn arriving_join(
        &self,
        ssa: &SsaFunction<T>,
        phi: &PhiNode,
        block_id: usize,
    ) -> Option<ValueRange> {
        let shape = IntShape::of_var(ssa, phi.result());
        let incoming: Vec<ValueRange> = phi
            .operands()
            .iter()
            .filter(|operand| {
                self.executable_edges
                    .contains(&(operand.predecessor(), block_id))
            })
            .map(|operand| {
                let range = self.arriving(ssa, operand.value(), operand.predecessor(), block_id);
                shape.map_or_else(|| range.clone(), |shape| shape.hold(&range))
            })
            .collect();
        (!incoming.is_empty()).then(|| ValueRange::join_all(&incoming))
    }

    /// Narrows a settled run: every value of the executable blocks computed
    /// once more from the settled values and met into what it holds, until a
    /// pass narrows nothing or the rounds run out.
    ///
    /// Values only grow while a run settles, so a phi joined while a guard's
    /// comparand was still unknown keeps what it was read as then: a binary
    /// search's `lo` read `[0, 39]` before `hi` was known keeps `[0, 39]`
    /// though the back edge it arrives on bounds it below `hi`. One more
    /// application of the transfer to a settled state is settled too, and no
    /// wider, so each pass keeps every fact sound and drops what the early
    /// reads added. Edges stay as the run found them.
    fn narrow(&mut self, ssa: &SsaFunction<T>) {
        for _ in 0..self.rounds {
            let mut narrowed = false;
            for (block_id, block) in ssa.iter_blocks() {
                if !self.is_block_executable(block_id) {
                    continue;
                }
                for phi in block.phi_nodes() {
                    let held = self
                        .assumed
                        .as_ref()
                        .and_then(|assumption| assumption.held.get(&phi.result()))
                        .cloned();
                    let Some(range) = self.arriving_join(ssa, phi, block_id) else {
                        continue;
                    };
                    let range = held.map_or_else(|| range.clone(), |held| range.meet(&held));
                    narrowed |= self.narrow_to(phi.result(), &range);
                }
                for instr in block.instructions() {
                    let Some(dest) = instr.op().dest() else {
                        continue;
                    };
                    if let Some(assumption) = &self.assumed {
                        let guarded: FxHashMap<SsaVarId, ValueRange> = instr
                            .uses()
                            .into_iter()
                            .filter(|var| assumption.held.contains_key(var))
                            .map(|var| {
                                (
                                    var,
                                    self.guarded_at(ssa, &assumption.dominators, var, block_id),
                                )
                            })
                            .collect();
                        self.guarded = guarded;
                    }
                    let range = self.evaluate_instruction(instr.op(), ssa);
                    self.guarded.clear();
                    narrowed |= self.narrow_to(dest, &range);
                }
            }
            if !narrowed {
                break;
            }
        }
    }

    /// Meets `range` into `var`'s recorded range, as [`Self::narrow`] does;
    /// whether it narrowed.
    fn narrow_to(&mut self, var: SsaVarId, range: &ValueRange) -> bool {
        let Some(old) = self.ranges.get(&var) else {
            return false;
        };
        let narrowed = old.meet(range);
        if narrowed == *old {
            return false;
        }
        self.ranges.insert(var, narrowed);
        true
    }

    /// Whether every operand the run brings `invariant`'s members stays
    /// within its interval: each read where it leaves its block, as the
    /// guards over that block and the branch taken to the member's block
    /// refine it.
    fn holds(&self, ssa: &SsaFunction<T>, invariant: &Invariant) -> bool {
        if self.assumed.is_none() {
            return false;
        }
        invariant.members.iter().all(|(block_id, member)| {
            ssa.block(*block_id)
                .and_then(|block| block.phi_nodes().iter().find(|phi| phi.result() == *member))
                .is_some_and(|phi| {
                    self.arriving_join(ssa, phi, *block_id)
                        .is_none_or(|range| invariant.range.join(&range) == invariant.range)
                })
        })
    }

    /// Resets all state and seeds the worklists from the entry block.
    ///
    /// Every variable starts uncomputed except those nothing in the body
    /// defines, which are recorded as [`ValueRange::top`]; the entry block is
    /// marked executable and its definitions evaluated, and its outgoing CFG
    /// edges are queued so propagation has somewhere to begin.
    /// Clearing first is what lets one [`RangeAnalysis`] be reused across
    /// functions.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function whose variables and entry block seed the state.
    /// * `cfg` - Exception-aware CFG providing the entry node and its
    ///   successors.
    fn initialize<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        self.ranges.clear();
        self.congruences.clear();
        self.executable_edges.clear();
        self.executable_blocks = BitSet::new(ssa.block_count());
        self.ssa_worklist.clear();
        self.cfg_worklist.clear();
        self.update_counts.clear();

        // Neither map is seeded, save for the one exception below. An absent
        // entry means "nothing has computed this yet", which
        // [`Self::get_range`] and [`Self::get_congruence`] both read as the
        // *least informative* value for an operand, and [`Self::update_value`]
        // reads as "the first computation stands on its own" - which is what
        // lets a value accumulate by joining.
        //
        // Seeding every range to `Top` would conflate "not computed yet" with
        // "computed, and unconstrained": joining against a seeded `Top` is
        // `Top` forever, so nothing would accumulate. Seeding the congruences
        // to `Bottom` - the empty set an accumulating join starts from - is
        // unsound for a value no transfer function produces: it would stay
        // empty, the emptiness spreads through every operation that reads it,
        // and a phi joining it reports only its *other* operands' residues. A
        // flattened dispatcher's state would read as `{0, 1}` modulo 64 while
        // the program also writes 2 and 3, and the `state == 2` arm would be
        // deleted as unreachable.
        //
        // **The exception: a value nothing in the body defines.** An argument,
        // a register live on entry, a version-0 local the rebuild placed at
        // the entry, a zero-initialised local - none has a defining
        // instruction or phi, so no transfer function will ever produce it and
        // it would stay absent for the whole run. Absent is read as the
        // least-informative value, which is the right *answer* for it; what is
        // wrong is the *reason*. [`Self::propagate_outgoing_edges`] defers a
        // terminator whose operand has not been computed, on the promise that
        // the computation is coming and will re-trigger it. For these values it
        // never comes, so deferring would stall the analysis at the first
        // branch on one, drain the worklist, and report convergence with the
        // arms behind that branch never executable.
        //
        // Recording them as `Top` states the fact directly: computed, and
        // unconstrained forever. Nothing ever calls [`Self::update_value`] on a
        // variable with no definer, so seeding exactly these changes no join.
        let mut defined = vec![false; ssa.variable_count()];
        for (_, block) in ssa.iter_blocks() {
            for phi in block.phi_nodes() {
                if let Some(slot) = defined.get_mut(phi.result().index()) {
                    *slot = true;
                }
            }
            for instr in block.instructions() {
                for def in instr.defs() {
                    if let Some(slot) = defined.get_mut(def.index()) {
                        *slot = true;
                    }
                }
            }
        }
        for (index, is_defined) in defined.into_iter().enumerate() {
            if !is_defined {
                let var = SsaVarId::from_index(index);
                self.ranges.insert(var, ValueRange::top());
                self.congruences.insert(var, Congruence::TOP);
            }
        }

        let entry = cfg.entry().index();
        self.mark_block_executable(entry);
        for succ in cfg.successors(cfg.entry()) {
            self.cfg_worklist.push_back((entry, succ.index()));
        }
        if let Some(block) = ssa.block(entry) {
            self.process_block_definitions(block, ssa);
        }
    }

    /// Drains both worklists to the fixpoint.
    ///
    /// Returns `true` when the fixpoint was reached. This is **not** advisory:
    /// the analysis is optimistic - it starts from an empty executable-edge set
    /// and grows ranges as edges are discovered - so its conclusions are only
    /// sound *at* the fixpoint. A partial run leaves ranges that are too narrow,
    /// which proves comparisons that are not actually provable. The caller must
    /// discard the result when this returns `false`.
    ///
    /// Termination comes from widening (see [`WIDEN_AFTER_REVISIONS`]); the step
    /// budget is a backstop for pathological constraint counts, sized from the
    /// function so ordinary code always converges well inside it.
    fn propagate<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G) -> bool
    where
        G: RootedGraph + Successors,
    {
        let work = ssa
            .block_count()
            .saturating_add(ssa.variable_count())
            .saturating_add(16);
        let budget = self.rounds.saturating_mul(work).saturating_add(64);
        let mut steps: usize = 0;
        loop {
            while let Some((from, to)) = self.cfg_worklist.pop_front() {
                steps = steps.saturating_add(1);
                if steps > budget {
                    return false;
                }
                if self.executable_edges.insert((from, to)) {
                    self.process_edge(from, to, ssa, cfg);
                }
            }
            let Some(var) = self.ssa_worklist.pop_front() else {
                return true;
            };
            steps = steps.saturating_add(1);
            if steps > budget {
                return false;
            }
            self.process_variable_uses(var, ssa, cfg);
        }
    }

    /// Processes a CFG edge that has just been proved executable.
    ///
    /// On the first visit to the target it is marked executable and its
    /// definitions evaluated; regardless of visit count, every phi at the target
    /// with an operand from `from` is re-joined, since a newly executable
    /// predecessor can only widen a phi's range. Outgoing edges are propagated
    /// only on the first visit - re-processing the same edge later changes
    /// nothing about reachability.
    ///
    /// # Arguments
    ///
    /// * `from` - Source block of the edge.
    /// * `to` - Target block of the edge.
    /// * `ssa` - The function being analyzed.
    /// * `cfg` - Exception-aware CFG used to enumerate `to`'s successors.
    fn process_edge<G>(&mut self, from: usize, to: usize, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        let first_visit = !self.is_block_executable(to);
        if first_visit {
            self.mark_block_executable(to);
        }
        // **Phis first, then the definitions that read them.**
        //
        // A block's instructions take their operands from its phis, so
        // evaluating the instructions first computes them against a phi that has
        // no value yet - which an absent entry reads as *any* value. On a first
        // visit nothing recomputes them before `propagate_outgoing_edges` runs
        // below, so the terminator is decided from those stale definitions and
        // queues **both** arms; and because the executable set only ever grows,
        // that is permanent.
        //
        // It is the difference between an architecture that compares and
        // branches in one instruction and one that does not. x86 spells `i < 0`
        // as a `BranchCmp` over the induction phi itself, so the terminator
        // reads the fresh phi and a zero-trip latch folds. MIPS and RISC-V
        // materialise the comparison into a register first (`slt`, then `bnez`),
        // and AArch64 reaches it through two conversions, so on those the
        // terminator reads a *definition of this block*, which evaluating the
        // phis second would compute from a phi that has no value.
        if let Some(block) = ssa.block(to) {
            for phi in block.phi_nodes() {
                // On the first visit every phi is evaluated, so one with no
                // operand for this edge still records a value rather than
                // staying uncomputed forever. Afterwards only this edge's
                // operands are new; see `Self::join_phi_operands`.
                if first_visit {
                    self.update_phi(phi, to, ssa);
                } else {
                    self.join_phi_operands(phi, to, ssa, Cause::Reached, |operand| {
                        operand.predecessor() == from
                    });
                }
            }
            if first_visit {
                self.process_block_definitions(block, ssa);
                self.propagate_outgoing_edges(to, block, ssa, cfg);
            }
        }
    }

    /// Evaluates every instruction in `block`, recording its result ranges.
    ///
    /// Called when a block first becomes executable so its definitions enter the
    /// range map immediately, rather than waiting for a use of each to surface
    /// on the worklist.
    ///
    /// # Arguments
    ///
    /// * `block` - The newly executable block.
    /// * `ssa` - The function, needed to look up operand ranges and types.
    fn process_block_definitions(&mut self, block: &SsaBlock<T>, ssa: &SsaFunction<T>) {
        for instr in block.instructions() {
            self.update_instruction_defs(instr, block.id(), ssa, Cause::Reached);
        }
    }

    /// Re-evaluates everything that uses `var` after its range changed.
    ///
    /// Walks `var`'s recorded use sites and, for each one in an executable
    /// block, recomputes the affected phi or instruction; a terminator whose
    /// operand moved additionally re-propagates its outgoing edges, since the
    /// new range may have just made a branch arm reachable or unreachable. Uses
    /// in blocks not yet executable are skipped - their ranges are not yet
    /// meaningful.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable whose range just changed.
    /// * `ssa` - The function being analyzed.
    /// * `cfg` - Exception-aware CFG used when a terminator re-propagates edges.
    fn process_variable_uses<G>(&mut self, var: SsaVarId, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        if let Some(ssa_var) = ssa.variable(var) {
            // Every operand is a use site of its own, so a phi carrying `var`
            // in from many predecessors is listed once per operand. One
            // revision joins all of them; the rest would redo it.
            let mut revised_phis: FxHashSet<(usize, usize)> = FxHashSet::default();
            for use_site in ssa_var.uses() {
                let block_id = use_site.block();
                if !self.is_block_executable(block_id) {
                    continue;
                }
                if use_site.is_phi_operand() {
                    if !revised_phis.insert((block_id, use_site.index())) {
                        continue;
                    }
                    if let Some(block) = ssa.block(block_id)
                        && let Some(phi) = block.phi(use_site.index())
                    {
                        self.join_phi_operands(phi, block_id, ssa, Cause::Moved, |operand| {
                            operand.value() == var
                        });
                    }
                } else if let Some(block) = ssa.block(block_id)
                    && let Some(instr) = block.instruction(use_site.index())
                {
                    self.update_instruction_defs(instr, block_id, ssa, Cause::Moved);
                    if instr.is_terminator() {
                        self.propagate_outgoing_edges(block_id, block, ssa, cfg);
                    }
                }
            }
        }
    }

    /// Recomputes and stores the ranges an instruction defines.
    ///
    /// The primary destination gets the range from
    /// [`Self::evaluate_instruction`]; any secondary definitions - side outputs
    /// the evaluator does not model - are set to [`ValueRange::top`] so nothing
    /// unsound is assumed about them.
    ///
    /// # Arguments
    ///
    /// While a loop invariant is checked, an assumed phi the instruction reads
    /// is read as the guards dominating `block` refine it.
    ///
    /// # Arguments
    ///
    /// * `instr` - The instruction whose definitions to update.
    /// * `block` - The block it lies in.
    /// * `ssa` - The function, needed for operand ranges and destination width.
    /// * `cause` - Whether the block was just reached or an operand moved.
    fn update_instruction_defs(
        &mut self,
        instr: &SsaInstruction<T>,
        block: usize,
        ssa: &SsaFunction<T>,
        cause: Cause,
    ) {
        let primary = instr.op().dest();
        if let Some(assumption) = &self.assumed {
            let guarded: FxHashMap<SsaVarId, ValueRange> = instr
                .uses()
                .into_iter()
                .filter(|var| assumption.held.contains_key(var))
                .map(|var| {
                    (
                        var,
                        self.guarded_at(ssa, &assumption.dominators, var, block),
                    )
                })
                .collect();
            self.guarded = guarded;
        }
        let range = self.evaluate_instruction(instr.op(), ssa);
        self.guarded.clear();
        let congruence = self
            .evaluate_instruction_congruence(instr.op(), ssa)
            .at_width(operation_width(instr.op(), ssa));
        let mut moved = false;
        for def in instr.defs() {
            if Some(def) == primary {
                moved |= self.update_value(def, &range, congruence, cause);
            } else {
                self.update_value(def, &ValueRange::top(), Congruence::TOP, cause);
            }
        }
        // **A flags variable's value moves when its producer's result moves.**
        //
        // Every secondary definition is recorded as `Top` and stays there, so
        // `update_value` never reports it changed and nothing re-runs its users.
        // For a flags variable that is wrong: `BranchFlags` names the *flags*,
        // and what decides it is the producing instruction's result (see
        // `Self::flag_source_range`). The dependency is real and the IR gives it
        // no operand, so the worklist has to carry it by hand.
        //
        // Without this the fixpoint decides a flag-tested branch once, on
        // whatever the producer was worth on the first visit, and never again.
        // A flattened dispatcher is that shape: `test state, state` with the
        // state phi still holding only its entry value, so `state == 0` would
        // read as certain, the `state == 1` arm would never be queued, and the
        // phi would never learn the states the loop writes back. Every later
        // arm of the dispatch chain would then fold against a state that cannot
        // be right.
        if moved {
            for def in instr.defs() {
                if Some(def) != primary {
                    self.ssa_worklist.push_back(def);
                }
            }
        }
    }

    /// Queues the CFG edges reachable from `block` given its terminator's
    /// operand ranges.
    ///
    /// This is where range facts prune control flow during the analysis itself:
    /// a `Branch` whose condition is a known constant (or provably zero, or
    /// provably positive) enqueues only the arm that can execute, and a `Switch`
    /// on a known index enqueues only the selected case. Non-CFG terminators
    /// (returns, throws) add nothing, and any unrecognized terminator falls back
    /// to the CFG's own successor list.
    ///
    /// # Undecided is not the same as uncomputed
    ///
    /// Every deciding arm asks [`Self::is_computed`] first and adds *nothing*
    /// when an operand has no value yet, then adds *every* arm when it has one
    /// that does not decide the branch. Those are the only two conservative
    /// answers, and which one is right turns entirely on whether the operand can
    /// still move.
    ///
    /// An operand nothing has computed yet is about to be computed, and that
    /// computation is a change, so [`Self::process_variable_uses`] re-runs this
    /// function for the terminator. Waiting costs a round and keeps the
    /// precision. An operand already recorded as `Top` cannot move again: the
    /// lattice is ascending and `Top` is its greatest element, so no later join
    /// changes it, nothing is ever re-queued, and waiting for it to sharpen
    /// waits forever. The result would not be a slower fixpoint but a wrong
    /// one: propagation stops at the first unconstrained condition, the
    /// worklist drains, and the run reports convergence with most of the
    /// function never marked executable. That is also why every variable no
    /// instruction or phi defines is seeded as `Top` (see [`Self::initialize`]):
    /// nothing will ever compute it, so waiting for it is the same stall.
    ///
    /// # Arguments
    ///
    /// * `block_id` - Index of the block whose terminator is examined.
    /// * `block` - The block itself, for its terminator.
    /// * `cfg` - Exception-aware CFG used for the fallback successor enumeration.
    fn propagate_outgoing_edges<G>(
        &mut self,
        block_id: usize,
        block: &SsaBlock<T>,
        ssa: &SsaFunction<T>,
        cfg: &G,
    ) where
        G: RootedGraph + Successors,
    {
        match block.control_terminator() {
            Some(SsaOp::Branch {
                condition,
                true_target,
                false_target,
            }) => {
                if !self.is_computed(*condition) {
                    return;
                }
                let range = self.get_range(*condition);
                if let Some(val) = range.as_constant() {
                    if val != 0 {
                        self.add_cfg_edge(block_id, *true_target);
                    } else {
                        self.add_cfg_edge(block_id, *false_target);
                    }
                } else if range.always_equal_to(0) == Some(true) {
                    self.add_cfg_edge(block_id, *false_target);
                } else if range.is_always_positive() {
                    self.add_cfg_edge(block_id, *true_target);
                } else {
                    self.add_cfg_edge(block_id, *true_target);
                    self.add_cfg_edge(block_id, *false_target);
                }
            }
            Some(SsaOp::Switch {
                value,
                targets,
                default,
            }) => {
                if !self.is_computed(*value) {
                    return;
                }
                let range = self.get_range(*value);
                if let Some(idx) = range.as_constant().and_then(|i| usize::try_from(i).ok()) {
                    if let Some(&target) = targets.get(idx) {
                        self.add_cfg_edge(block_id, target);
                    } else {
                        self.add_cfg_edge(block_id, *default);
                    }
                } else {
                    for &target in targets {
                        self.add_cfg_edge(block_id, target);
                    }
                    self.add_cfg_edge(block_id, *default);
                }
            }
            // A `BranchCmp` states its comparison inline and names no
            // condition variable, so the `Branch` arm above cannot see it, and
            // the `_` case below would enqueue every successor unconditionally.
            // That would defeat this function's purpose for the shape it
            // matters most on: a zero-trip loop's latch is exactly a
            // compare-and-branch, and marking its back edge executable lets the
            // phi join the incremented counter, widening lift it to
            // `[0, +inf)`, and the width wrap lose the `0` floor.
            //
            // It asks [`decide_operands`] rather than re-deciding the
            // comparison, so the pruning done here and the folding done after
            // the fixpoint cannot disagree.
            Some(SsaOp::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            }) => {
                if !self.is_computed(*left) || !self.is_computed(*right) {
                    return;
                }
                let (left_range, right_range) = (self.get_range(*left), self.get_range(*right));
                let left_known = Known {
                    range: &left_range,
                    congruence: self.get_congruence(*left),
                };
                let right_known = Known {
                    range: &right_range,
                    congruence: self.get_congruence(*right),
                };
                match decide_operands(
                    ssa,
                    *cmp,
                    *unsigned,
                    (*left, left_known),
                    (*right, right_known),
                ) {
                    Some(true) => self.add_cfg_edge(block_id, *true_target),
                    Some(false) => self.add_cfg_edge(block_id, *false_target),
                    // Undecided, so both arms may run.
                    None => {
                        self.add_cfg_edge(block_id, *true_target);
                        self.add_cfg_edge(block_id, *false_target);
                    }
                }
            }
            // **The third conditional terminator.** A flag-tested branch is
            // the shape every x86 comparison takes at `-O0` and the shape of
            // every `test`/`cmp` idiom at `-O2`, so it is decided here rather
            // than left to the catch-all below, which enqueues every successor.
            //
            // A flags value names an operation rather than a value, so the
            // question goes back to the operation's result - see
            // [`Self::flag_source_range`] - and [`flag_holds`] answers it.
            // Undecidable conditions take both arms.
            Some(SsaOp::BranchFlags {
                flags,
                condition,
                true_target,
                false_target,
            }) => {
                // A flags value with no recorded producer has nothing that will
                // ever be computed for it, so waiting would stall: both arms.
                let Some(source) = ssa.flag_source(*flags) else {
                    self.add_cfg_edge(block_id, *true_target);
                    self.add_cfg_edge(block_id, *false_target);
                    return;
                };
                // A producer not computed yet is the same deferral the other
                // arms make: its computation re-propagates this terminator.
                if let Some(range) = self.flag_source_range(*flags, ssa) {
                    match flag_holds(ssa, source, &range, *condition) {
                        Some(true) => self.add_cfg_edge(block_id, *true_target),
                        Some(false) => self.add_cfg_edge(block_id, *false_target),
                        None => {
                            self.add_cfg_edge(block_id, *true_target);
                            self.add_cfg_edge(block_id, *false_target);
                        }
                    }
                }
            }
            Some(SsaOp::Jump { target }) => {
                self.add_cfg_edge(block_id, *target);
            }
            Some(
                SsaOp::Return { .. }
                | SsaOp::Throw { .. }
                | SsaOp::Rethrow
                | SsaOp::EndFilter { .. }
                | SsaOp::InterruptReturn,
            ) => {}
            // `endfinally` resumes at the targets of the `leave`s that run the
            // finally, which the graph states: code after it runs.
            Some(SsaOp::EndFinally) => {
                for succ in cfg.successors(NodeId::new(block_id)) {
                    self.add_cfg_edge(block_id, succ.index());
                }
            }
            _ => {
                let node = NodeId::new(block_id);
                for succ in cfg.successors(node) {
                    self.add_cfg_edge(block_id, succ.index());
                }
            }
        }
    }

    /// The range of the value whose computation set `flags`.
    ///
    /// A flags variable is a *secondary* definition, and
    /// [`Self::update_instruction_defs`] records every secondary definition as
    /// `Top` - deliberately, because nothing in an interval domain models a
    /// flags word. So the flags variable's own range says nothing, and asking
    /// it is how a flag-tested branch stays undecidable however well the
    /// analysis knows the value.
    ///
    /// The fact is one variable over: on the *primary* destination of the same
    /// instruction, which is the value the flags describe. This walks the
    /// flags variable's definition site back to that instruction and reads its
    /// result.
    ///
    /// # Arguments
    ///
    /// * `flags` - The flags variable a terminator tests.
    /// * `ssa` - The function, for the definition site.
    ///
    /// # Returns
    ///
    /// The producer's result range, or `None` when no instruction defines
    /// `flags`, when it defines no primary value, or when nothing has computed
    /// that value yet.
    fn flag_source_range(&self, flags: SsaVarId, ssa: &SsaFunction<T>) -> Option<ValueRange> {
        let dest = ssa.flag_source(flags)?;
        // Not `get_range`: absent means "not computed", and answering the
        // least-informative range for it would decide the branch from a fact
        // nothing established. See `is_computed`.
        self.is_computed(dest).then(|| self.get_range(dest))
    }

    /// Enqueues a CFG edge for processing unless it is already executable.
    ///
    /// The executable-edge set is the dedup guard: an edge already proved
    /// executable needs no re-processing, so re-queuing it would only spin the
    /// worklist.
    ///
    /// # Arguments
    ///
    /// * `from` - Source block of the edge.
    /// * `to` - Target block of the edge.
    fn add_cfg_edge(&mut self, from: usize, to: usize) {
        if !self.executable_edges.contains(&(from, to)) {
            self.cfg_worklist.push_back((from, to));
        }
    }

    /// Returns `true` if `block` is currently marked executable.
    ///
    /// Block indices flow in from terminator targets (`true_target`,
    /// `false_target`, switch targets, `Jump.target`), which the IR permits to
    /// be out of range - a terminator may reference a block that was never
    /// recovered (common in stripped/obfuscated binaries), and the
    /// [`crate::analysis::verifier`] explicitly tolerates such dangling
    /// successors. The `executable_blocks` bitset is sized to exactly
    /// `block_count`, so an out-of-range target is by definition unreachable:
    /// report it `false` instead of indexing past the bitset and panicking.
    fn is_block_executable(&self, block: usize) -> bool {
        self.executable_blocks.contains_checked(block)
    }

    /// Marks `block` executable, ignoring out-of-range indices (see
    /// [`Self::is_block_executable`]).
    fn mark_block_executable(&mut self, block: usize) {
        self.executable_blocks.insert_checked(block);
    }

    /// Re-evaluates `phi` in `block_id` and records its result.
    ///
    /// Both domains are held to the result variable's shape: the operands may
    /// be held in other windows (a phi can merge a signed and an unsigned
    /// byte), and a narrow result states no residue.
    fn update_phi(&mut self, phi: &PhiNode, block_id: usize, ssa: &SsaFunction<T>) {
        let shape = IntShape::of_var(ssa, phi.result());
        let joined = self.evaluate_phi(phi, block_id);
        let range = shape.map_or(joined.clone(), |shape| shape.hold(&joined));
        let congruence = self
            .evaluate_phi_congruence(phi, block_id)
            .at_width(ssa.bit_width(phi.result()));
        self.update_value(phi.result(), &range, congruence, Cause::Reached);
    }

    /// Joins into `phi`'s result the executable operands `select` picks.
    ///
    /// **Only what moved.** A phi is revised when one of its operands changes
    /// or one of its incoming edges becomes executable, and in both cases the
    /// other operands contribute nothing new: [`Self::update_value`] joins the
    /// input into what the result already holds, and that already holds every
    /// other executable operand as it currently stands - each was joined in
    /// when its edge became executable and again every time it moved, and
    /// operand values only ever grow. Re-joining the whole phi reaches the same
    /// fixpoint, but costs its operand count on every revision, and the phi
    /// that is revised most - a flattened dispatcher's, whose operands number
    /// in the hundreds and each move several times - made the analysis
    /// quadratic in them: one 600-block method spent 87 million operand visits
    /// and eight seconds there.
    ///
    /// Each selected operand is held to the result's shape on its own rather
    /// than after the join, which is at least as precise: holding a hull wraps
    /// whatever the hull spans, holding each operand only what it spans.
    ///
    /// A phi with no selected executable operand is left alone; nothing
    /// reached it, so nothing revised it.
    ///
    /// # Arguments
    ///
    /// * `phi` - The phi node to revise.
    /// * `block_id` - Block containing the phi, used to test each incoming edge.
    /// * `ssa` - The function, for the result's shape and width.
    /// * `cause` - Whether an edge was reached or an operand moved.
    /// * `select` - Picks the operands whose value or edge is new.
    fn join_phi_operands(
        &mut self,
        phi: &PhiNode,
        block_id: usize,
        ssa: &SsaFunction<T>,
        cause: Cause,
        select: impl Fn(&PhiOperand) -> bool,
    ) {
        let shape = IntShape::of_var(ssa, phi.result());
        // An assumed phi reads each operand as the edge it arrives on refines
        // it: a rotated loop's exit test is what bounds what its back edge
        // carries in.
        if self
            .assumed
            .as_ref()
            .is_some_and(|assumption| assumption.held.contains_key(&phi.result()))
        {
            let Some(range) = self.arriving_join(ssa, phi, block_id) else {
                return;
            };
            let congruence = phi
                .operands()
                .iter()
                .filter(|operand| {
                    self.executable_edges
                        .contains(&(operand.predecessor(), block_id))
                })
                .fold(Congruence::BOTTOM, |congruence, operand| {
                    congruence.join(self.get_congruence(operand.value()))
                })
                .at_width(ssa.bit_width(phi.result()));
            self.update_value(phi.result(), &range, congruence, cause);
            return;
        }
        // Each distinct value once: a dispatcher's phi carries one unchanged
        // value in from most of its hundreds of predecessors, and reading it
        // once per operand is the same quadratic cost again.
        let mut values: Vec<SsaVarId> = Vec::new();
        for operand in phi.operands() {
            if !select(operand)
                || values.contains(&operand.value())
                || !self
                    .executable_edges
                    .contains(&(operand.predecessor(), block_id))
            {
                continue;
            }
            values.push(operand.value());
        }
        if values.is_empty() {
            return;
        }
        let mut incoming: Vec<ValueRange> = Vec::with_capacity(values.len());
        let mut congruence = Congruence::BOTTOM;
        for value in values {
            let range = self.get_range(value);
            incoming.push(shape.map_or_else(|| range.clone(), |shape| shape.hold(&range)));
            congruence = congruence.join(self.get_congruence(value));
        }
        let range = ValueRange::join_all(&incoming);
        let congruence = congruence.at_width(ssa.bit_width(phi.result()));
        self.update_value(phi.result(), &range, congruence, cause);
    }

    /// Joins the ranges flowing into a phi from its executable predecessors.
    ///
    /// Only operands whose incoming edge is executable contribute, so operands
    /// arriving on not-yet-reachable edges add nothing - the optimism that lets
    /// the analysis prove tight ranges. A phi with no executable operand yet
    /// evaluates to `Top` rather than `Bottom`. The operands are joined in one
    /// step ([`ValueRange::join_all`]), so a phi with thousands of operands -
    /// a flattened dispatcher's state - costs one normalisation per evaluation
    /// rather than one per operand.
    ///
    /// # Arguments
    ///
    /// * `phi` - The phi node to evaluate.
    /// * `block_id` - Block containing the phi, used to test each incoming edge.
    ///
    /// # Returns
    ///
    /// The joined range over the reachable operands.
    fn evaluate_phi(&self, phi: &PhiNode, block_id: usize) -> ValueRange {
        let mut incoming: Vec<ValueRange> = Vec::new();
        for operand in phi.operands() {
            if !self
                .executable_edges
                .contains(&(operand.predecessor(), block_id))
            {
                continue;
            }
            let range = self.get_range(operand.value());
            if range.is_top() {
                return ValueRange::top();
            }
            incoming.push(range);
        }
        if incoming.is_empty() {
            return ValueRange::top();
        }
        ValueRange::join_all(&incoming)
    }

    /// The congruence a phi's result may take, joined over its executable
    /// operands.
    ///
    /// The same walk [`Self::evaluate_phi`] makes, over the same edge set, so
    /// the two facts about the result are drawn from the same set of reachable
    /// predecessors. A phi with no executable operand is unreached and states
    /// nothing.
    fn evaluate_phi_congruence(&self, phi: &PhiNode, block_id: usize) -> Congruence {
        let mut result = Congruence::BOTTOM;
        let mut has_executable_operand = false;
        for operand in phi.operands() {
            let pred = operand.predecessor();
            if !self.executable_edges.contains(&(pred, block_id)) {
                continue;
            }
            has_executable_operand = true;
            result = result.join(self.get_congruence(operand.value()));
            if result.is_top() {
                break;
            }
        }
        if has_executable_operand {
            result
        } else {
            Congruence::TOP
        }
    }

    /// The congruence an operation's result may take.
    ///
    /// Mirrors [`Self::evaluate_instruction`] and is deliberately partial:
    /// every operation this does not model answers [`Congruence::TOP`], which
    /// states nothing. Sound by construction - machine arithmetic is already
    /// modular, so the operations below commute with the reduction exactly (see
    /// [`Congruence`]).
    ///
    /// # The one relational case
    ///
    /// `Mul` whose operands are the **same SSA value** is a square, and a
    /// square is not the pairwise product of two values that happen to have the
    /// same possibilities. Over an unconstrained value the pairwise product is
    /// every residue and the squares are twelve of sixty-four, so this test is
    /// the difference between deciding a quadratic opaque predicate and
    /// answering `Top`. It is the only place this domain relates two operands,
    /// and it does so by identity rather than by reasoning.
    fn evaluate_instruction_congruence(&self, op: &SsaOp<T>, ssa: &SsaFunction<T>) -> Congruence {
        match op {
            SsaOp::Const { value, .. } => Congruence::of_constant(value),
            SsaOp::Copy { src, .. } => self.get_congruence(*src),
            // As for the range: one arm or the other, so what both agree on.
            SsaOp::Select {
                true_val,
                false_val,
                ..
            } => self
                .get_congruence(*true_val)
                .join(self.get_congruence(*false_val)),
            SsaOp::Add { left, right, .. } => {
                self.get_congruence(*left).sum(self.get_congruence(*right))
            }
            SsaOp::Sub { left, right, .. } => self
                .get_congruence(*left)
                .difference(self.get_congruence(*right)),
            SsaOp::Mul { left, right, .. } if left == right => self.get_congruence(*left).square(),
            SsaOp::Mul { left, right, .. } => self
                .get_congruence(*left)
                .product(self.get_congruence(*right)),
            SsaOp::And { left, right, .. } => {
                self.get_congruence(*left).and(self.get_congruence(*right))
            }
            SsaOp::Or { left, right, .. } => {
                self.get_congruence(*left).or(self.get_congruence(*right))
            }
            SsaOp::Xor { left, right, .. } => {
                self.get_congruence(*left).xor(self.get_congruence(*right))
            }
            SsaOp::Neg { operand, .. } => self.get_congruence(*operand).negate(),
            SsaOp::Not { operand, .. } => self.get_congruence(*operand).complement(),
            SsaOp::Shl { value, amount, .. } => match self.shift_count(ssa, *value, *amount) {
                Some(shift) => self.get_congruence(*value).shift_left(shift),
                // A shift by an unknown amount is every shift, and their union
                // is everything the value's residues reach - which is not worth
                // enumerating for the precision it buys. A count of the width or
                // more is undefined and proves nothing.
                None => Congruence::TOP,
            },
            // A conversion keeps the low bits: truncation takes them and both
            // extensions leave them alone. So the residue passes through
            // whenever the *narrower* of the two widths still covers the bits
            // the modulus tracks, and otherwise there is nothing to carry.
            SsaOp::IntConv {
                operand,
                target,
                unsigned,
                ..
            } => Self::through_reinterpretation(
                self.get_congruence(*operand),
                ssa.bit_width(*operand),
                T::bit_width(target),
                *unsigned,
            ),
            // An address is a number and these two only change how it is
            // spelled, so the residue survives both under the same width rule a
            // conversion follows. Not a detail: the lift expresses `x - 1` on a
            // value it has decided is address-like as `PtrAdd offset -1`, so
            // declining here loses the residue of every such subtraction - and
            // an opaque predicate that ends in `- 1` is exactly that shape.
            SsaOp::PtrToInt {
                operand, target, ..
            }
            | SsaOp::IntToPtr {
                operand, target, ..
            } => Self::through_reinterpretation(
                self.get_congruence(*operand),
                ssa.bit_width(*operand),
                T::bit_width(target),
                // An address has no sign to extend: the conversion reinterprets
                // the pattern and never widens it by replicating a bit.
                true,
            ),
            // `base + index * stride + offset`, which is arithmetic this domain
            // computes exactly. An index the analysis knows nothing about
            // contributes `Top` and the whole address follows, which is the same
            // answer declining would give.
            SsaOp::PtrAdd {
                base,
                index,
                stride,
                offset,
                ..
            } => {
                let scaled = match index {
                    Some(index) => self
                        .get_congruence(*index)
                        .product(Congruence::residue(*stride)),
                    None => Congruence::exact(0),
                };
                self.get_congruence(*base)
                    .sum(scaled)
                    .sum(Congruence::exact(*offset))
            }
            _ => Congruence::TOP,
        }
    }

    /// The congruence carried through an operation that only reinterprets bits.
    ///
    /// Two conditions, and the second is narrower than it first looks.
    ///
    /// The **destination** must be at least as wide as the bits the modulus
    /// tracks. Below that the result has been truncated to fewer bits than the
    /// residue describes, and what remains is not determined by the residue that
    /// went in.
    ///
    /// The **source** only matters for a sign-extension. A truncation keeps the
    /// low bits and a zero-extension leaves them alone, so either preserves the
    /// residue whatever the source width was - which is why `unsigned` alone is
    /// enough and the source width need not be known at all. A *sign*-extension
    /// writes ones above the source width, and if the source is narrower than
    /// the modulus those ones land inside the residue; so that case wants a
    /// source width, and declines without one.
    ///
    /// The distinction is load-bearing rather than pedantic. The lift types many
    /// temporaries with nothing the target can give a width for, and requiring a
    /// source width unconditionally left every conversion in an opaque
    /// predicate's chain at `Top` - which is to say, it lost the whole chain.
    fn through_reinterpretation(
        operand: Congruence,
        from: Option<u32>,
        to: Option<u32>,
        unsigned: bool,
    ) -> Congruence {
        let wide_enough = |width: Option<u32>| width.is_some_and(|w| w >= Congruence::MODULUS_BITS);
        if !wide_enough(to) {
            return Congruence::TOP;
        }
        if unsigned || wide_enough(from) {
            return operand;
        }
        Congruence::TOP
    }

    /// Computes the output range of a single operation from its operand ranges.
    ///
    /// This is the transfer function of the analysis. Interval arithmetic runs
    /// on mathematical integers, and the machine keeps only the low bits, so
    /// every result is reduced to the window of the shape it is held in (see
    /// [`IntShape`]); an operation that can carry out of its width answers
    /// [`ValueRange::top`] when that shape is unknown. Shifts, `Rem` and
    /// `ArrayLength` are modeled only where a sound bound is available (a
    /// shift, for instance, only by a constant below the value's width), and
    /// comparisons are decided through [`decide_operands`]. Anything not
    /// modeled returns [`ValueRange::top`].
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to evaluate.
    /// * `ssa` - The function, used to look up the operands' and the
    ///   destination's shapes.
    ///
    /// # Returns
    ///
    /// The range proved for the operation's result, in its destination's
    /// window.
    fn evaluate_instruction(&self, op: &SsaOp<T>, ssa: &SsaFunction<T>) -> ValueRange {
        let shape = IntShape::of_result(ssa, op);
        // A result that may have carried out of its width means nothing until
        // it is reduced to that width, and an unknown width cannot reduce it.
        let wrapped =
            |raw: ValueRange| shape.map_or_else(ValueRange::top, |shape| shape.hold(&raw));
        // A result that cannot leave its operands' bits is still held in its
        // destination's window, which the operands' own need not be.
        let held = |raw: ValueRange| shape.map_or(raw.clone(), |shape| shape.hold(&raw));
        match op {
            SsaOp::Const { value, .. } => {
                // The constant's bits, read as its destination reads them: a
                // `U8(200)` written to a signed byte is `-56`.
                let bits = value.as_i64().or_else(|| {
                    value
                        .as_u64()
                        .map(|bits| i64::from_ne_bytes(bits.to_ne_bytes()))
                });
                bits.map_or_else(ValueRange::top, |bits| held(ValueRange::constant(bits)))
            }
            SsaOp::Copy { src, .. } => held(self.get_range(*src)),
            // A select is one arm or the other, so its range is the join of
            // the two - never wider, and never the top the absence of this rule
            // leaves behind. A compiler that has no sub-register write spells
            // `state = (i < n) ? 2 : 3` as a select of two byte constants
            // inserted into a register, and without this the state of a
            // flattened dispatcher reads as "any byte" instead of the two
            // values it takes.
            //
            // Each arm is read where it is chosen: a condition comparing it
            // bounds it there, as it would on a branch's edge. `(n - 1 < 8191)
            // ? n - 1 : 8191` is how a compiler spells `min(n - 1, 8191)`, and
            // its range is `[0, 8191]` whatever `n` is.
            SsaOp::Select {
                condition,
                true_val,
                false_val,
                ..
            } => {
                let comparison = ssa.get_definition(*condition);
                let arm = |value: SsaVarId, taken: bool| {
                    let range = self.get_range(value);
                    comparison
                        .and_then(|comparison| {
                            self.refine_from_comparison(ssa, comparison, value, taken)
                        })
                        .map_or_else(|| range.clone(), |refined| range.meet(&refined))
                };
                held(arm(*true_val, true).join(&arm(*false_val, false)))
            }
            SsaOp::IntConv {
                operand,
                target,
                unsigned,
                ..
            } => held(convert(
                &self.get_range(*operand),
                IntShape::of_var(ssa, *operand),
                IntShape::of_type::<T>(target),
                *unsigned,
            )),
            SsaOp::Add { left, right, .. } => {
                wrapped(self.get_range(*left).add(&self.get_range(*right)))
            }
            SsaOp::Sub { left, right, .. } => {
                // Two addresses of one image move together wherever it lands,
                // so their distance is a constant though neither is: `cmp r3,
                // r0` over `__TMC_END__` loaded twice is zero.
                let symbol = |var: SsaVarId| ssa.symbol_of(var).and_then(ConstValue::resolved_i64);
                match (symbol(*left), symbol(*right)) {
                    (Some(left), Some(right)) => left
                        .checked_sub(right)
                        .map_or_else(ValueRange::top, |distance| {
                            wrapped(ValueRange::constant(distance))
                        }),
                    _ => wrapped(self.get_range(*left).sub(&self.get_range(*right))),
                }
            }
            SsaOp::Mul { left, right, .. } => {
                wrapped(self.get_range(*left).mul(&self.get_range(*right)))
            }
            SsaOp::Neg { operand, .. } => {
                let value = self.get_range(*operand);
                let negated = match (value.min(), value.max()) {
                    (Some(min), Some(max)) => match (min.checked_neg(), max.checked_neg()) {
                        (Some(neg_min), Some(neg_max)) => {
                            ValueRange::bounded(neg_max.min(neg_min), neg_max.max(neg_min))
                        }
                        _ => ValueRange::top(),
                    },
                    _ => ValueRange::top(),
                };
                wrapped(negated)
            }
            SsaOp::Shl { value, amount, .. } => {
                let val_range = self.get_range(*value);
                if let Some(shift) = self.shift_count(ssa, *value, *amount)
                    && let (Some(min), Some(max)) = (val_range.min(), val_range.max())
                    && let (Some(new_min), Some(new_max)) =
                        (min.checked_shl(shift), max.checked_shl(shift))
                    // A left shift is only a multiply while nothing leaves the
                    // top of the `i64`; verify by shifting back. What leaves the
                    // top of the machine width is the wrap below.
                    && new_min.checked_shr(shift) == Some(min)
                    && new_max.checked_shr(shift) == Some(max)
                {
                    return wrapped(ValueRange::bounded(
                        new_min.min(new_max),
                        new_min.max(new_max),
                    ));
                }
                ValueRange::top()
            }
            SsaOp::Or { left, right, .. } | SsaOp::Xor { left, right, .. } => {
                // For non-negative operands the result cannot exceed the next
                // power of two above either bound, and cannot go negative.
                let l = self.get_range(*left);
                let r = self.get_range(*right);
                let raw = match (l.max(), r.max()) {
                    (Some(l_max), Some(r_max))
                        if l.is_always_non_negative()
                            && r.is_always_non_negative()
                            && l_max >= 0
                            && r_max >= 0 =>
                    {
                        let bound = l_max.max(r_max);
                        // Saturate to all-ones at the width of the larger bound.
                        let bits = i64::BITS.saturating_sub(bound.leading_zeros());
                        match i64::try_from(BitWidth::new(bits).mask()) {
                            Ok(all_ones) => ValueRange::bounded(0, all_ones),
                            Err(_) => ValueRange::non_negative(),
                        }
                    }
                    _ => ValueRange::top(),
                };
                held(raw)
            }
            SsaOp::And { left, right, .. } => {
                // **`x & x` is `x`**, and on x86 it is not a curiosity: `test
                // reg, reg` is how every comparison against zero sets the flags,
                // so this shape carries the range that every flag-tested branch
                // then asks about.
                if left == right {
                    return held(self.get_range(*left));
                }
                let (l, r) = (self.get_range(*left), self.get_range(*right));
                if let (Some(left_value), Some(right_value)) = (l.as_constant(), r.as_constant()) {
                    return held(ValueRange::constant(left_value & right_value));
                }
                // **One rule, and a constant mask is a case of it.** AND only
                // ever clears bits, so an operand that is non-negative and at
                // most `m` bounds the *result* to `[0, m]` whatever the other
                // side holds: its sign bit is clear, so the result's is too, and
                // no bit above `m`'s highest survives. Both sides bounded takes
                // the tighter.
                //
                // Bounding by the mask alone is sound but answers `[0, 1]` for
                // `0 & 1` instead of `0`. AArch64 spells the sign test as
                // `(square >> 31) & 1`, so that imprecision would stand between
                // a bounded square and a folded branch.
                let bound = |range: &ValueRange| {
                    range
                        .is_always_non_negative()
                        .then(|| range.max())
                        .flatten()
                };
                let raw = match (bound(&l), bound(&r)) {
                    (Some(left_bound), Some(right_bound)) => {
                        ValueRange::bounded(0, left_bound.min(right_bound))
                    }
                    (Some(only), None) | (None, Some(only)) => ValueRange::bounded(0, only),
                    // A non-negative operand with no upper bound still clears
                    // the sign bit of the result.
                    (None, None) => {
                        if l.is_always_non_negative() || r.is_always_non_negative() {
                            ValueRange::non_negative()
                        } else {
                            ValueRange::top()
                        }
                    }
                };
                held(raw)
            }
            SsaOp::Shr {
                value,
                amount,
                unsigned,
                ..
            } => {
                // The value as the shift reads it: unsigned for a logical
                // shift, signed for an arithmetic one. Over a non-negative
                // reading the two agree, and both divide by a power of two.
                let read = self.operand_view(ssa, *value, !*unsigned);
                let shift = self.shift_count(ssa, *value, *amount);
                if let Some(shift) = shift
                    && read.is_always_non_negative()
                    && let (Some(min), Some(max)) = (read.min(), read.max())
                {
                    return held(ValueRange::bounded(min >> shift, max >> shift));
                }
                // A logical shift by a constant clears the top bits it shifts
                // in, whatever it reads: a 64-bit `x >> 49` lies in
                // `[0, 2^15 - 1]`, the masking `(x << 49) >> 49` spells.
                if *unsigned
                    && let Some(shift) = shift.filter(|shift| *shift > 0)
                    && let Some(width) = ssa.bit_width(*value)
                {
                    let kept = width.saturating_sub(shift);
                    let max = 1i64
                        .checked_shl(kept)
                        .and_then(|bound| bound.checked_sub(1))
                        .filter(|_| kept < 63)
                        .unwrap_or(i64::MAX);
                    return held(ValueRange::bounded(0, max));
                }
                ValueRange::top()
            }
            SsaOp::Rem {
                left,
                right,
                unsigned,
                ..
            } => {
                // Both operands as the remainder reads them; over a
                // non-negative dividend and a positive divisor both readings
                // agree and the result lies below the divisor.
                let divisor = self.operand_view(ssa, *right, !*unsigned);
                if let Some(n) = divisor.as_constant()
                    && n > 0
                    && self
                        .operand_view(ssa, *left, !*unsigned)
                        .is_always_non_negative()
                {
                    return held(ValueRange::bounded(0, n.saturating_sub(1)));
                }
                ValueRange::top()
            }
            // A length is a non-negative count that fits the type it is held
            // in, so it is bounded by the window's upper half rather than
            // wrapped (which would lose the floor to the unbounded top).
            SsaOp::ArrayLength { .. } => match shape {
                Some(shape) if shape.width < 64 => {
                    ValueRange::non_negative().meet(&ValueRange::window(shape.width, true))
                }
                _ => ValueRange::non_negative(),
            },
            SsaOp::NewArr { .. }
            | SsaOp::NewObj { .. }
            | SsaOp::Box { .. }
            | SsaOp::LoadToken { .. } => ValueRange::top(),
            // **A comparison is worth what it proves, not what its type admits.**
            //
            // `0..=1` is the bound every comparison satisfies, but answering it
            // unconditionally would leave the *fixpoint* unable to see what
            // [`try_simplify_comparison`] can see perfectly well after it: a
            // comparison whose operands settle it is a constant. The two would
            // then answer one question differently, which is precisely what
            // [`decide_comparison`]'s own contract says must not happen - so
            // this asks it.
            //
            // The consequence is not cosmetic. A materialised comparison is how
            // MIPS and RISC-V spell every conditional branch - `slt` into a
            // register, then `bnez` on that register - so without this a branch
            // on one could never be decided however completely its operands
            // were known, and a zero-trip latch that folds on an architecture
            // that compares and branches in one instruction would survive on
            // these two.
            SsaOp::Ceq { .. } | SsaOp::Clt { .. } | SsaOp::Cgt { .. } => {
                let decided = match comparison_operands(op) {
                    Some(ComparisonOperands {
                        left,
                        right,
                        cmp,
                        unsigned,
                        ..
                    }) => {
                        let (left_range, right_range) =
                            (self.get_range(left), self.get_range(right));
                        let left_known = Known {
                            range: &left_range,
                            congruence: self.get_congruence(left),
                        };
                        let right_known = Known {
                            range: &right_range,
                            congruence: self.get_congruence(right),
                        };
                        match decide_operands(
                            ssa,
                            cmp,
                            unsigned,
                            (left, left_known),
                            (right, right_known),
                        ) {
                            Some(true) => ValueRange::constant(1),
                            Some(false) => ValueRange::constant(0),
                            None => ValueRange::bounded(0, 1),
                        }
                    }
                    None => ValueRange::bounded(0, 1),
                };
                held(decided)
            }
            // A call returns what the host states its callee returns.
            SsaOp::Call(call) => call
                .dest
                .and_then(|dest| self.results.get(&dest))
                .map_or_else(ValueRange::top, |stated| held(stated.clone())),
            _ => ValueRange::top(),
        }
    }

    /// `var`'s range read in the `signed` or unsigned window of its own width,
    /// or as recorded when its width is unknown.
    ///
    /// A 64-bit value is held signed whatever is asked (see [`IntShape`]), so
    /// an unsigned question about one is only answered where it is
    /// non-negative - which every caller checks.
    fn operand_view(&self, ssa: &SsaFunction<T>, var: SsaVarId, signed: bool) -> ValueRange {
        let range = self.get_range(var);
        match IntShape::of_var(ssa, var) {
            Some(shape) => range.wrap(shape.width, signed),
            None => range,
        }
    }

    /// The count a shift of `value` by `amount` shifts by, when it is a known
    /// constant below `value`'s width.
    ///
    /// A count of the width or more is undefined in the IR - x86 masks it,
    /// other machines shift everything out - so it proves nothing, and neither
    /// does a count for a value of unknown width.
    fn shift_count(&self, ssa: &SsaFunction<T>, value: SsaVarId, amount: SsaVarId) -> Option<u32> {
        let width = ssa.bit_width(value)?;
        let count = self.get_range(amount).as_constant()?;
        u32::try_from(count).ok().filter(|count| *count < width)
    }

    /// Returns `var`'s current range, or the default when none is recorded.
    ///
    /// A missing entry means the variable has not been reached yet in this run;
    /// the default range is the sound stand-in during the optimistic analysis.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to look up.
    ///
    /// # Returns
    ///
    /// The recorded range, or the default range when absent.
    fn get_range(&self, var: SsaVarId) -> ValueRange {
        if let Some(guarded) = self.guarded.get(&var) {
            return guarded.clone();
        }
        self.ranges.get(&var).cloned().unwrap_or_default()
    }

    /// Returns `true` once some transfer function has recorded a range for
    /// `var`.
    ///
    /// [`Self::get_range`] answers the least-informative range for a variable
    /// that is absent *and* for one recorded as `Top`, which is the right
    /// reading for an operand - both mean "assume anything" - but the wrong one
    /// for deciding whether to wait. This distinguishes them; see
    /// [`Self::propagate_outgoing_edges`] for why the difference decides
    /// reachability.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable to look up.
    fn is_computed(&self, var: SsaVarId) -> bool {
        self.ranges.contains_key(&var)
    }

    /// Returns `var`'s current congruence, or [`Congruence::TOP`] when none is
    /// recorded.
    fn get_congruence(&self, var: SsaVarId) -> Congruence {
        self.congruences.get(&var).copied().unwrap_or_default()
    }

    /// Revises both of `var`'s abstract values, widening once the value has
    /// grown often enough to look loop-carried.
    ///
    /// **One step, two domains.** The interval and the congruence describe the
    /// same variable at the same point of the same fixpoint, and revising them
    /// separately would give them separate revision counts and separate
    /// worklist entries - so one could widen while the other did not, and a
    /// consumer reading both would be reading two different iterations.
    ///
    /// Widening differs by domain because their heights do. Without widening a
    /// counted loop needs one round per trip, so a bounded run stops mid-ascent
    /// holding a range that is too narrow - the unsound direction. An interval
    /// widens by dropping the unstable bound to infinity. A congruence is a
    /// subset of sixty-four residues, so *joining* is already a bounded ascent:
    /// the set only grows, at most sixty-four times, and reaches `Top` on its
    /// own. Joining is therefore both the widening and the more precise choice.
    fn update_value(
        &mut self,
        var: SsaVarId,
        new_range: &ValueRange,
        new_congruence: Congruence,
        cause: Cause,
    ) -> bool {
        // The raw entries, not the `get_*` readers: absent here means "never
        // computed", and the first computation stands on its own. Reading the
        // least-informative value that absence means to an *operand* would make
        // every join saturate immediately and both domains would state nothing.
        let computed_range = self.ranges.get(&var).cloned();
        let computed_congruence = self.congruences.get(&var).copied();
        // **Monotone, in both domains: joined or widened, never replaced.**
        //
        // This analysis decides a branch from an *interval*, not from a
        // constant, and that is only sound if an interval can never be narrower
        // than the truth. Replacing the previous value does not give that.
        // A value computed while only some of a block's predecessors are known
        // executable is computed from a partial phi, so it can come out
        // narrower than it will finally be; a branch decided on it prunes an
        // edge, and the pruned edge is the one that would have widened it. The
        // prophecy is self-fulfilling, and the fixpoint settles on an executable
        // set that excludes a reachable arm.
        //
        // Joining removes the question: the value only ever grows, so an early
        // evaluation is an under-approximation the next one repairs, and no
        // decision is ever made on a value that later turns out to have been too
        // tight. Nothing is lost at the fixpoint either - the inputs are
        // monotone too, so the last evaluation is the widest and the join equals
        // it.
        //
        // Widening still applies on top, for the reason it always did: a
        // loop-carried interval grows by one step per trip, so joining alone
        // would need as many rounds as the loop has trips. It is reserved for
        // growth an operand's move caused; see `Cause`.
        let next_range = match &computed_range {
            None => new_range.clone(),
            Some(old) => {
                let joined = old.join(new_range);
                // A run checking an invariant widens phis alone.
                let widens = self
                    .assumed
                    .as_ref()
                    .is_none_or(|assumption| assumption.phis.contains(&var));
                if cause == Cause::Moved && joined != *old {
                    let growths = self.update_counts.entry(var).or_insert(0);
                    *growths = growths.saturating_add(1);
                    if widens && *growths > WIDEN_AFTER_REVISIONS {
                        old.widen(new_range)
                    } else {
                        joined
                    }
                } else {
                    joined
                }
            }
        };
        // An assumed phi stays within its assumed interval, which widening
        // then reaches rather than passing: whether its operands stay within
        // it too is checked once the run settles.
        let next_range = match self
            .assumed
            .as_ref()
            .and_then(|assumption| assumption.held.get(&var))
        {
            Some(held) => next_range.meet(held),
            None => next_range,
        };
        // The congruence needs no widening of its own: it is a subset of
        // sixty-four residues, so joining is already a bounded ascent that
        // reaches `Top` in at most that many steps.
        let next_congruence = match computed_congruence {
            Some(old) => old.join(new_congruence),
            None => new_congruence,
        };
        let mut changed = false;
        if computed_range.as_ref() != Some(&next_range) {
            self.ranges.insert(var, next_range);
            changed = true;
        }
        if computed_congruence != Some(next_congruence) {
            self.congruences.insert(var, next_congruence);
            changed = true;
        }
        if changed {
            self.ssa_worklist.push_back(var);
        }
        changed
    }
}

/// Pass that propagates value-range information across the SSA graph.
///
/// Uses sparse conditional
/// range propagation to simplify branches and replace comparisons whose
/// result is provable from operand ranges. The ranges are the environment's
/// ([`PassEnv::ranges`]), computed within the host's
/// [`Budget::RangeRounds`] and
/// shared with every other pass that reads them.
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - can replace conditional branches
/// with unconditional jumps, changing CFG edges.
#[derive(Debug, Default, Clone, Copy)]
pub struct ValueRangePropagationPass;

impl ValueRangePropagationPass {
    /// Creates a new value-range propagation pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for ValueRangePropagationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "value-range-propagation"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Propagate value-range information across the SSA graph"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Run value-range propagation on `ssa`.
    ///
    /// Reads the function's ranges from `facts`, then simplifies branches and
    /// comparisons based on them. A fixpoint that did not converge proves nothing:
    /// its exhaustion is reported as [`EventKind::BudgetExhausted`] and nothing is
    /// rewritten.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any branch or comparison was simplified.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let facts = env.ranges();
        let result = match facts.get(ssa) {
            Ok(result) => result,
            Err(exhaustion) => {
                env.exhausted(exhaustion);
                return Ok(false);
            }
        };

        let mut branch_simplifications: Vec<(usize, usize, bool)> = Vec::new();
        let mut comparison_replacements: Vec<(usize, usize, SsaVarId, bool)> = Vec::new();
        let mut traps_never_taken: Vec<(usize, usize)> = Vec::new();

        for (block_idx, block) in ssa.iter_blocks() {
            // **Rewrite only what the analysis reached.**
            //
            // Every decision below reads ranges, and a range is only a fact about a
            // block the fixpoint proved executable. In a block it never reached,
            // `get_range` answers with whatever a dominating block last wrote -
            // often an entry constant, because the definitions that would have
            // overwritten it were never evaluated. Folding on that turns "the
            // analysis did not get here" into "the analysis proved this branch",
            // and the two have nothing to do with each other: it is how a stalled
            // propagation became a deleted loop body rather than a missed
            // optimisation.
            //
            // Unreachable blocks are left exactly as they are. Deleting them is
            // real, but it belongs to whichever pass owns reachability, not to a
            // range analysis that happens to have noticed.
            if !result.is_executable(block_idx) {
                continue;
            }

            // A `BranchCmp` states its comparison inline and names no condition
            // variable, so the `Branch` arm below cannot see it - which left every
            // compare-and-branch terminator unexamined however precisely its
            // operands were bounded. A zero-trip loop latch is exactly that shape.
            if let Some(SsaOp::BranchCmp {
                left,
                right,
                cmp,
                unsigned,
                true_target,
                false_target,
            }) = block.control_terminator()
                && let (Some(left_known), Some(right_known)) =
                    (result.known(*left), result.known(*right))
                && let Some(taken) = decide_operands(
                    ssa,
                    *cmp,
                    *unsigned,
                    (*left, left_known),
                    (*right, right_known),
                )
            {
                let target = if taken { *true_target } else { *false_target };
                branch_simplifications.push((block_idx, target, taken));
            }

            // The same question the fixpoint's own `BranchFlags` arm asks, asked
            // again against the settled ranges - the pairing `BranchCmp` above
            // already has, so pruning during the fixpoint and folding after it
            // cannot disagree.
            if let Some(SsaOp::BranchFlags {
                flags,
                condition,
                true_target,
                false_target,
            }) = block.control_terminator()
                && let Some(source) = ssa.flag_source(*flags)
                && let Some(range) = result.get(source)
                && let Some(taken) = flag_holds(ssa, source, range, *condition)
            {
                let target = if taken { *true_target } else { *false_target };
                branch_simplifications.push((block_idx, target, taken));
            }

            if let Some(SsaOp::Branch {
                condition,
                true_target,
                false_target,
            }) = block.control_terminator()
                && let Some(range) = result.get(*condition)
            {
                if let Some(is_true) = range.always_equal_to(0)
                    && is_true
                {
                    branch_simplifications.push((block_idx, *false_target, false));
                }
                if let Some(val) = range.as_constant()
                    && val != 0
                {
                    branch_simplifications.push((block_idx, *true_target, true));
                }
            }

            for (instr_idx, instr) in block.instructions().iter().enumerate() {
                if let Some((dest, value)) = try_simplify_comparison(instr.op(), &result, ssa) {
                    comparison_replacements.push((block_idx, instr_idx, dest, value));
                }
                // A conditional trap fires where its two inputs compare as it
                // states (`teq`, `fwait`'s pending exceptions against zero); one
                // whose operands decide the comparison false never fires.
                if let SsaOp::SystemOp(data) = instr.op()
                    && let SystemOpKind::Trap { op, .. } = data.kind
                    && let Some((cmp, unsigned)) = op.condition()
                    && let [left, right] = data.inputs.as_slice()
                    && let (Some(left_known), Some(right_known)) =
                        (result.known(*left), result.known(*right))
                    && decide_operands(
                        ssa,
                        cmp,
                        unsigned,
                        (*left, left_known),
                        (*right, right_known),
                    ) == Some(false)
                {
                    traps_never_taken.push((block_idx, instr_idx));
                }
            }
        }

        let mut changed = false;

        let result = ssa.edit(SsaEditOptions::new(), |editor| {
            for (block_idx, target, is_true) in branch_simplifications {
                let Some(_) = editor.function().block(block_idx) else {
                    continue;
                };

                // Folding a branch removes an edge, which can only *increase*
                // dominance - no new phi can be required, so this needs a phi-operand
                // prune plus `repair_ssa`, and no value reconstruction.
                editor.fold_terminator_pruning_phis(block_idx, SsaOp::Jump { target })?;
                env.record(EventKind::OpaquePredicateRemoved)
                    .location(block_idx)
                    .message(format!(
                        "range analysis: condition always {}",
                        if is_true { "true" } else { "false" }
                    ));
                env.record(EventKind::BranchSimplified)
                    .location(block_idx)
                    .message(format!("simplified to unconditional jump to {target}"));
                changed = true;
            }

            for (block_idx, instr_idx) in traps_never_taken {
                if editor
                    .function()
                    .block(block_idx)
                    .and_then(|block| block.instruction(instr_idx))
                    .is_none()
                {
                    continue;
                }
                editor.nop_instruction(block_idx, instr_idx)?;
                env.record(EventKind::OpaquePredicateRemoved)
                    .location(instr_idx)
                    .message("range analysis: trap condition always false".to_string());
                changed = true;
            }

            for (block_idx, instr_idx, dest, value) in comparison_replacements {
                if editor
                    .function()
                    .block(block_idx)
                    .and_then(|block| block.instruction(instr_idx))
                    .is_none()
                {
                    continue;
                }

                let const_value = ConstValue::from_bool(value);
                editor.replace_instruction_op(
                    block_idx,
                    instr_idx,
                    SsaOp::Const {
                        dest,
                        value: const_value,
                    },
                )?;
                env.record(EventKind::ConstantFolded)
                    .location(instr_idx)
                    .message(format!("range analysis: comparison → {value}"));
                changed = true;
            }
            Ok(())
        });

        if result.is_err() {
            // The session runs under `SsaRollbackPolicy::Never`, so a failed edit or
            // boundary repair leaves the edits applied - the function is mutated and
            // possibly mid-repair. Reporting "unchanged" would make the pass-group
            // transaction skip **both** verification and rollback, keeping damaged
            // IR and keeping it unchecked. Report the change so the transaction
            // verifies this function and rolls it back.
            return Ok(true);
        }
        Ok(changed)
    }
}

/// The width an operation computes in, for the congruence domain: the
/// destination's, or the operands' when the destination has none.
fn operation_width<T: Target>(op: &SsaOp<T>, ssa: &SsaFunction<T>) -> Option<u32> {
    op.dest().and_then(|dest| ssa.bit_width(dest)).or_else(|| {
        op.width_operand(|operand| ssa.bit_width(operand))
            .and_then(|operand| ssa.bit_width(operand))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{context::AnalysisContext, range::IntervalRange},
        events::EventLog,
        ir::{
            CmpKind,
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            instruction::SsaInstruction,
            ops::{NativeKindedData, TrapOp, native::FlagCondition},
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        passes::decide::{decide_comparison, ranges_overlap, taken_target_known},
        testing::{
            MACHINE, MockTarget, MockType, mock_terminator_at, run_mock_malformed_boundary,
            run_mock_pass_boundary,
        },
    };

    /// `B0: jump B1` / `B1 (try): leave B3` / `B2 (finally): endfinally` /
    /// `B3: y = 5; ret y`: `endfinally` resumes at the leave's target, so the
    /// range pass reaches B3 and knows `y`.
    #[test]
    fn code_after_a_finally_is_reached() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let y = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(3, 0),
            MockType::I32,
        );
        for (index, op) in [
            SsaOp::Jump { target: 1 },
            SsaOp::Leave { target: 3 },
            SsaOp::EndFinally,
        ]
        .into_iter()
        .enumerate()
        {
            let mut block = SsaBlock::new(index);
            block.add_instruction(SsaInstruction::synthetic(op));
            ssa.add_block(block);
        }
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: y,
            value: ConstValue::I32(5),
        }));
        b3.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(y))));
        ssa.add_block(b3);
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(1, 2),
            handler_range: BlockRange::new(2, 3),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let ranges = analyze(
            &ssa,
            &WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 10),
            &BTreeMap::new(),
        )
        .expect("the analysis converges");
        assert_eq!(ranges.get(y).and_then(ValueRange::as_constant), Some(5));
    }

    /// Overlap detection across disjoint, overlapping, identical, `top`, and
    /// `bottom` range pairs.
    #[test]
    fn ranges_overlap_basics() {
        // Non-overlapping ranges
        let a = ValueRange::bounded(0, 5);
        let b = ValueRange::bounded(10, 15);
        assert!(!ranges_overlap(&a, &b));

        // Overlapping ranges
        let c = ValueRange::bounded(0, 10);
        let d = ValueRange::bounded(5, 15);
        assert!(ranges_overlap(&c, &d));

        // Same range
        let e = ValueRange::bounded(5, 10);
        assert!(ranges_overlap(&e, &e));

        // Top overlaps with everything
        let top = ValueRange::top();
        assert!(ranges_overlap(&top, &a));

        // Bottom doesn't overlap
        let bottom = ValueRange::bottom();
        assert!(!ranges_overlap(&bottom, &a));
    }

    /// The shape of a signed and an unsigned 8-, 32- and 64-bit value.
    fn shape(width: u32, signed: bool) -> Option<IntShape> {
        Some(IntShape { width, signed })
    }

    /// A result that escapes its width is reduced to the width, not erased to
    /// `Top`.
    ///
    /// The destination is a 32-bit machine value, which is never outside the
    /// 32-bit window, and a bound that escaped to infinity says nothing more
    /// precise than that. Erasing it instead is what lets one wrapped increment
    /// poison an induction variable through its phi.
    #[test]
    fn a_result_escaping_its_width_is_held_in_the_window() {
        let unbounded_above = ValueRange::Interval(IntervalRange::new(Some(1), None));
        let held = IntShape {
            width: 32,
            signed: true,
        }
        .hold(&unbounded_above);
        assert_eq!(held.min(), Some(-(1i64 << 31)));
        assert_eq!(held.max(), Some((1i64 << 31) - 1));
        assert!(!held.is_top(), "a held range still carries both bounds");
    }

    /// A result that wraps by a known amount lands where the machine puts it.
    ///
    /// `i32::MAX + 1` is `i32::MIN`, not "somewhere in the window", and the
    /// same byte reads `200` unsigned and `-56` signed.
    #[test]
    fn a_wrapped_result_lands_where_the_machine_puts_it() {
        let signed = IntShape {
            width: 32,
            signed: true,
        };
        assert_eq!(
            signed.hold(&ValueRange::constant(1i64 << 31)),
            ValueRange::constant(-(1i64 << 31))
        );
        let byte = ValueRange::constant(200);
        assert_eq!(
            IntShape {
                width: 8,
                signed: true
            }
            .hold(&byte),
            ValueRange::constant(-56)
        );
        assert_eq!(
            IntShape {
                width: 8,
                signed: false
            }
            .hold(&ValueRange::constant(-56)),
            byte
        );
        // Crossing one window boundary splits into the two pieces either side.
        let crossing = IntShape {
            width: 8,
            signed: true,
        }
        .hold(&ValueRange::bounded(120, 130));
        assert!(crossing.may_contain(127) && crossing.may_contain(-128));
        assert!(!crossing.may_contain(0), "{crossing:?}");
    }

    /// A result that fits its window is returned unchanged, so holding it costs
    /// no precision where nothing wrapped.
    #[test]
    fn a_result_inside_its_width_is_kept() {
        let inside = ValueRange::bounded(0, 10);
        let kept = IntShape {
            width: 32,
            signed: true,
        }
        .hold(&inside);
        assert_eq!(kept, inside);
    }

    /// A 64-bit value keeps finite bounds, and one whose bound escaped `i64` is
    /// unknown.
    ///
    /// Every finite `i64` bound already is a 64-bit machine value, so there is
    /// nothing to reduce - a value known to be `0` stays `0` across every
    /// widening to 64 bits. An infinite bound is different: the interval
    /// arithmetic produces one when a bound overflowed, and the machine value
    /// then wrapped. `[0, i64::MAX] + 1` is `[1, +inf)` as integers, but at
    /// `i64::MAX` the machine sum is `i64::MIN`, so the only sound reading is
    /// the whole window - which at 64 bits is `Top`.
    #[test]
    fn a_full_width_keeps_finite_bounds_and_forgets_escaped_ones() {
        let full = IntShape {
            width: 64,
            signed: true,
        };
        let escaping = ValueRange::Interval(IntervalRange::new(Some(1), None));
        assert!(
            full.hold(&escaping).is_top(),
            "an overflowed bound wrapped, so the sign is unknown"
        );
        let zero = ValueRange::constant(0);
        assert_eq!(full.hold(&zero), zero);
        let bounded = ValueRange::bounded(i64::MIN, 5);
        assert_eq!(full.hold(&bounded), bounded);
    }

    /// Widening with sign extension reads the source signed.
    #[test]
    fn a_sign_extending_widening_preserves_a_signed_range() {
        let value = ValueRange::bounded(-5, 5);
        let widened = convert(&value, shape(32, true), shape(64, true), false);
        assert_eq!(widened, value);
    }

    /// Sign-extending a value whose top bit is set makes it negative, whatever
    /// window the source was held in.
    #[test]
    fn a_sign_extending_widening_of_a_high_bit_is_negative() {
        let high = ValueRange::bounded(0, 1i64 << 31);
        let widened = convert(&high, shape(32, false), shape(64, true), false);
        assert!(widened.may_contain(-(1i64 << 31)), "{widened:?}");
        assert!(!widened.is_always_non_negative());
    }

    /// Widening with zero extension turns a negative source into a large
    /// positive number, so the source interval must **not** be carried over.
    ///
    /// `[-1, -1]` fits in 64 bits, so a width check alone would keep it, but
    /// the zero-extended value is `4294967295`.
    #[test]
    fn a_zero_extending_widening_of_a_negative_source_is_not_the_source_range() {
        let negative = ValueRange::bounded(-1, -1);
        let widened = convert(&negative, shape(32, true), shape(64, true), true);
        assert_eq!(widened, ValueRange::constant((1i64 << 32) - 1));
        let unknown = convert(&ValueRange::top(), shape(32, true), shape(64, true), true);
        assert_eq!(unknown, ValueRange::bounded(0, (1i64 << 32) - 1));
    }

    /// A non-negative source is unchanged by zero extension, so precision is
    /// kept where the two interpretations agree.
    #[test]
    fn a_zero_extending_widening_of_a_non_negative_source_preserves_it() {
        let value = ValueRange::bounded(0, 10);
        let widened = convert(&value, shape(32, true), shape(64, true), true);
        assert_eq!(widened, value);
    }

    /// Narrowing keeps a value that still fits and reduces one that does not
    /// to the target's window - signed or unsigned, as the target is.
    #[test]
    fn a_narrowing_conversion_reads_the_low_bits_in_the_targets_window() {
        let fits = ValueRange::bounded(0, 100);
        assert_eq!(
            convert(&fits, shape(32, true), shape(16, true), false),
            fits
        );
        let escapes = ValueRange::bounded(0, 1 << 20);
        let narrowed = convert(&escapes, shape(32, true), shape(16, true), false);
        assert_eq!(narrowed.min(), Some(-(1i64 << 15)));
        assert_eq!(narrowed.max(), Some((1i64 << 15) - 1));
        // An unsigned byte target reads the low bits unsigned.
        let byte = convert(&ValueRange::top(), shape(32, true), shape(8, false), true);
        assert_eq!(byte, ValueRange::bounded(0, 255));
        assert_eq!(
            convert(
                &ValueRange::constant(200),
                shape(32, true),
                shape(8, true),
                false
            ),
            ValueRange::constant(-56)
        );
    }

    /// **An unknown *destination* width proves nothing; an unknown source one
    /// leaves little.**
    ///
    /// What the source width is needed for is the extension: a sign-extension
    /// depends on where the source's sign bit sits, so it proves nothing
    /// without one (zero excepted, which every extension keeps). A
    /// zero-extension or truncation of a non-negative value is that value
    /// whichever it is, so the destination alone settles it.
    #[test]
    fn a_conversion_keeps_what_it_can_without_a_source_width() {
        let value = ValueRange::bounded(0, 10);
        assert!(convert(&value, shape(32, true), None, false).is_top());
        assert_eq!(convert(&value, None, shape(32, true), true), value);
        assert!(
            convert(&value, None, shape(32, true), false).is_top(),
            "a byte holding 200 sign-extends negative, and nothing here says it is not a byte"
        );
        assert_eq!(
            convert(&ValueRange::constant(0), None, shape(64, true), false),
            ValueRange::constant(0)
        );
        let signed = ValueRange::bounded(-1, 10);
        assert!(convert(&signed, None, shape(32, true), true).is_top());
    }

    /// Builds converged [`ValueRanges`] directly from range entries.
    ///
    /// Every block is executable: these tests exercise the deciding helpers,
    /// not the reachability guard [`ValueRanges::is_executable`] applies in
    /// [`ValueRangePropagationPass`].
    fn make_result(entries: Vec<(SsaVarId, ValueRange)>) -> ValueRanges {
        let mut executable = BitSet::new(BLOCKS_IN_TEST_RESULT);
        for block in 0..BLOCKS_IN_TEST_RESULT {
            executable.insert_checked(block);
        }
        ValueRanges {
            ranges: entries.into_iter().collect(),
            congruences: FxHashMap::default(),
            executable,
            predecessors: Vec::new(),
        }
    }

    /// An empty store whose fixpoint is given `limit` rounds.
    fn rounds(limit: usize) -> RangeFacts {
        RangeFacts::new(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, limit))
    }

    /// A function declaring no variables, so every operand's width is unknown
    /// and a comparison reads the ranges exactly as recorded.
    fn no_variables() -> SsaFunction<MockTarget> {
        SsaFunction::new(0, 0)
    }

    /// How many blocks [`make_result`] marks executable.
    const BLOCKS_IN_TEST_RESULT: usize = 64;

    /// A fixpoint the budget stops short proves nothing: no ranges are
    /// returned, the exhaustion says which budget and by how much, a store
    /// keeps that answer, and the pass rewrites nothing and reports it.
    #[test]
    fn an_unconverged_fixpoint_is_an_exhaustion() {
        let (mut ssa, _) = increment_chain();
        let exhaustion = Exhaustion {
            budget: Budget::RangeRounds,
            limit: 0,
            demand: 1,
        };
        assert_eq!(
            analyze(
                &ssa,
                &WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 0),
                &BTreeMap::new()
            )
            .err(),
            Some(exhaustion)
        );

        let facts = rounds(0);
        assert_eq!(facts.get(&ssa).err(), Some(exhaustion));
        assert!(facts.is_computed(), "the exhaustion is the stored answer");

        let log: EventLog<MockTarget> = EventLog::new();
        let env = PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 0));
        assert_eq!(ValueRangePropagationPass.run(&mut ssa, &env), Ok(false));
        assert_eq!(log.count_kind(EventKind::BudgetExhausted), 1);
        assert!(
            analyze(&ssa, &WorkBudget::DEFAULT, &BTreeMap::new()).is_ok(),
            "the default budget converges on the same function"
        );
    }

    /// A store computes once and answers every later reader from that
    /// computation, until it is told the function changed.
    #[test]
    fn a_store_answers_from_one_computation_until_invalidated() {
        let (ssa, counter) = increment_chain();
        let facts = rounds(32);
        assert!(!facts.is_computed());
        let first = facts.get(&ssa).expect("the chain converges");
        let second = facts.get(&ssa).expect("the chain converges");
        assert!(
            Arc::ptr_eq(&first, &second),
            "the second read is the first's"
        );
        assert!(first.get(counter).is_some());

        facts.invalidate();
        assert!(!facts.is_computed());
        let third = facts.get(&ssa).expect("the chain converges");
        assert!(
            !Arc::ptr_eq(&first, &third),
            "an invalidated store recomputes"
        );
    }

    /// A `Clt` whose left range is entirely below its right folds to `true`.
    #[test]
    fn try_simplify_clt_always_true() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(0, 5)),
            (v1, ValueRange::bounded(10, 20)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: v0,
            right: v1,
            unsigned: false,
            signaling: false,
        };
        assert_eq!(
            try_simplify_comparison(&op, &result, &no_variables()),
            Some((dest, true))
        );
    }

    /// An unsigned ordering is refused where an operand may be negative.
    ///
    /// The regression this guards: `ValueRange` is a signed interval, so the
    /// constant `-1` is *below* `0` here - while the same bits read unsigned
    /// (`0xFFFF_FFFF`) are far above it. Deciding the unsigned comparison from
    /// the signed bound inverts the answer, and this predicate is what a
    /// terminator fold consults, so the wrong answer deletes a live edge.
    #[test]
    fn an_unsigned_ordering_over_a_possibly_negative_range_is_refused() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::constant(-1)),
            (v1, ValueRange::constant(0)),
        ]);
        let signed: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: v0,
            right: v1,
            unsigned: false,
            signaling: false,
        };
        assert_eq!(
            try_simplify_comparison(&signed, &result, &no_variables()),
            Some((dest, true)),
            "signed, -1 is below 0"
        );
        let unsigned: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: v0,
            right: v1,
            unsigned: true,
            signaling: false,
        };
        assert_eq!(
            try_simplify_comparison(&unsigned, &result, &no_variables()),
            None,
            "unsigned, the same bits are above 0 - the signed bound cannot say"
        );
    }

    /// An unsigned ordering *is* decided where both operands are non-negative,
    /// which is exactly where the signed and unsigned orders coincide.
    #[test]
    fn an_unsigned_ordering_over_non_negative_ranges_is_decided() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(0, 5)),
            (v1, ValueRange::bounded(10, 20)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: v0,
            right: v1,
            unsigned: true,
            signaling: false,
        };
        assert_eq!(
            try_simplify_comparison(&op, &result, &no_variables()),
            Some((dest, true))
        );
    }

    /// Every ordering refines the arm it is on, from either operand's side.
    ///
    /// `<=` and `>=` reach the refinement through a fused `BranchCmp`, which is
    /// how every native comparison branch is lifted; reading them as `>`
    /// refined `x <= 36`'s false arm to `x <= 36`, and jump threading then
    /// took a switch's `'*'` case for `'$'`'s.
    #[test]
    fn every_ordering_refines_the_arm_it_is_on() {
        let x = SsaVarId::from_index(0);
        let result = ValueRanges {
            ranges: [(x, ValueRange::bounded(0, 100))].into_iter().collect(),
            congruences: FxHashMap::default(),
            executable: BitSet::new(0),
            predecessors: Vec::new(),
        };
        let limit = ValueRange::constant(36);
        let at_least = |low| Some(ValueRange::Interval(IntervalRange::at_least(low)));
        let at_most = |high| Some(ValueRange::Interval(IntervalRange::at_most(high)));
        // (relation, `x` on the left, arm taken, what `x` is on that arm)
        let cases = [
            (CmpKind::Lt, true, true, at_most(35)),
            (CmpKind::Lt, true, false, at_least(36)),
            (CmpKind::Le, true, true, at_most(36)),
            (CmpKind::Le, true, false, at_least(37)),
            (CmpKind::Gt, true, true, at_least(37)),
            (CmpKind::Gt, true, false, at_most(36)),
            (CmpKind::Ge, true, true, at_least(36)),
            (CmpKind::Ge, true, false, at_most(35)),
            // `36 <= x` is `x >= 36`, and so on with `x` on the right.
            (CmpKind::Le, false, true, at_least(36)),
            (CmpKind::Le, false, false, at_most(35)),
            (CmpKind::Lt, false, true, at_least(37)),
            (CmpKind::Ge, false, true, at_most(36)),
            (CmpKind::Gt, false, false, at_least(36)),
        ];
        for (cmp, x_is_left, taken, expected) in cases {
            assert_eq!(
                result.refine_against(x, None, &limit, None, cmp, false, taken, x_is_left),
                expected,
                "{cmp} with x {} and the arm {}",
                if x_is_left { "left" } else { "right" },
                if taken { "taken" } else { "not taken" },
            );
        }
    }

    /// Equality carries no signedness guard: the two interpretations disagree
    /// about which value is larger, never about whether they are the same.
    #[test]
    fn equality_is_decided_under_either_interpretation() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::constant(-1)),
            (v1, ValueRange::constant(-1)),
        ]);
        assert_eq!(
            decide_comparison(
                CmpKind::Eq,
                true,
                result.known(v0).unwrap(),
                result.known(v1).unwrap()
            ),
            Some(true)
        );
        assert_eq!(
            decide_comparison(
                CmpKind::Ne,
                true,
                result.known(v0).unwrap(),
                result.known(v1).unwrap()
            ),
            Some(false)
        );
        let _ = dest;
    }

    /// A `Cgt` whose left range is entirely above its right folds to `true`.
    #[test]
    fn try_simplify_cgt_always_true() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(100, 200)),
            (v1, ValueRange::bounded(0, 50)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Cgt {
            dest,
            left: v0,
            right: v1,
            unsigned: false,
            signaling: false,
        };
        assert_eq!(
            try_simplify_comparison(&op, &result, &no_variables()),
            Some((dest, true))
        );
    }

    /// A `Ceq` on disjoint ranges folds to `false`.
    #[test]
    fn try_simplify_ceq_never() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(0, 5)),
            (v1, ValueRange::bounded(10, 20)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Ceq {
            dest,
            left: v0,
            right: v1,
            signaling: false,
        };
        assert_eq!(
            try_simplify_comparison(&op, &result, &no_variables()),
            Some((dest, false))
        );
    }

    /// A `Ceq` on two equal constant ranges folds to `true`.
    #[test]
    fn try_simplify_ceq_constants_equal() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::constant(42)),
            (v1, ValueRange::constant(42)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Ceq {
            dest,
            left: v0,
            right: v1,
            signaling: false,
        };
        assert_eq!(
            try_simplify_comparison(&op, &result, &no_variables()),
            Some((dest, true))
        );
    }

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
    }

    /// A single block counting from zero in steps of one, and the last count.
    ///
    /// Long enough that the fixpoint's fixed allowance of steps does not cover
    /// it: a budget of no rounds stops it short.
    fn increment_chain() -> (SsaFunction<MockTarget>, SsaVarId) {
        const STEPS: u16 = 100;
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, usize::from(STEPS) + 2);
        let one = local_at(&mut ssa, 0, 0, 0);
        let mut count = local_at(&mut ssa, 1, 0, 1);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: count,
            value: ConstValue::I32(0),
        }));
        for step in 0..STEPS {
            let next = local_at(&mut ssa, step + 2, 0, usize::from(step) + 2);
            b0.add_instruction(instr(SsaOp::Add {
                dest: next,
                left: count,
                right: one,
                flags: None,
            }));
            count = next;
        }
        b0.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b0);
        ssa.recompute_uses();
        (ssa, count)
    }

    fn local_at(
        ssa: &mut SsaFunction<MockTarget>,
        idx: u16,
        block: usize,
        instr: usize,
    ) -> SsaVarId {
        ssa.create_variable(
            VariableOrigin::Local(idx),
            0,
            DefSite::instruction(block, instr),
            MockType::I32,
        )
    }

    /// A `Copy`-only function yields ranges but triggers no rewrite.
    #[test]
    fn range_propagation_through_copy() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(10),
        }));
        block.add_instruction(instr(SsaOp::Copy { dest: v1, src: v0 }));
        block.add_instruction(instr(SsaOp::returns(Some(v1))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "copy range propagation", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        assert!(
            !changed,
            "copy-only range propagation should not rewrite SSA"
        );
    }

    /// Range propagation through `Add` produces no rewrite on its own.
    #[test]
    fn range_on_add_propagates() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let v2 = local_at(&mut ssa, 2, 0, 2);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(5),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(3),
        }));
        block.add_instruction(instr(SsaOp::Add {
            dest: v2,
            left: v0,
            right: v1,
            flags: None,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(v2))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "add range propagation", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        assert!(
            !changed,
            "range propagation through add should not rewrite SSA"
        );
    }

    /// A branch on a constant condition is folded to an unconditional jump.
    #[test]
    fn range_simplifies_branch_with_constant_condition() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: v0,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "constant branch range folding", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        // Branch on constant 1 should be simplified to Jump 1
        assert!(changed, "constant branch should be simplified");
        assert!(matches!(
            mock_terminator_at(&ssa, 0),
            SsaOp::Jump { target: 1 }
        ));
    }

    /// `x & -16` clears the low four bits and leaves everything else alone, so
    /// its value is unconstrained. Deriving `[0, 0]` from the mask - which
    /// `mask.max(0)` does for any negative mask - claims the result is the
    /// constant zero, and the pass then folds a live branch to its false arm and
    /// deletes the true arm.
    #[test]
    fn and_with_a_negative_mask_does_not_prove_a_constant() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let x = local_at(&mut ssa, 0, 0, 0);
        let mask = local_at(&mut ssa, 1, 0, 1);
        let masked = local_at(&mut ssa, 2, 0, 2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: x,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(-16),
        }));
        b0.add_instruction(instr(SsaOp::And {
            dest: masked,
            left: x,
            right: mask,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: masked,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        assert!(
            ValueRangePropagationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20))
                )
                .is_ok()
        );

        assert!(
            matches!(mock_terminator_at(&ssa, 0), SsaOp::Branch { .. }),
            "the branch on `x & -16` must survive; got {:?}",
            mock_terminator_at(&ssa, 0)
        );
    }

    /// The non-negative case must keep folding: `x & 15` really is in `[0, 15]`.
    #[test]
    fn and_with_a_non_negative_mask_still_bounds_the_result() {
        let range = ValueRange::top().and_constant(15);
        assert_eq!(range.min(), Some(0));
        assert_eq!(range.max(), Some(15));

        let unconstrained = ValueRange::top().and_constant(-16);
        assert!(
            unconstrained.as_constant().is_none(),
            "a negative mask must not yield a constant, got {unconstrained:?}"
        );
    }

    /// A straight-line block with no branch is left unchanged.
    #[test]
    fn single_block_no_branch_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let v0 = SsaVarId::from_index(0);
        ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(42),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(Some(v0))));
        ssa.add_block(block);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "single-block range pass", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        assert!(!changed);
    }

    /// A comparison of two known constants folds to a `Const`.
    #[test]
    fn comparison_folding_with_ranges() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 0, 1);
        let v2 = local_at(&mut ssa, 2, 0, 2);
        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(1),
        }));
        block.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(100),
        }));
        // v0 < v1 is always true since both are constants
        block.add_instruction(instr(SsaOp::Clt {
            dest: v2,
            left: v0,
            right: v1,
            unsigned: false,
            signaling: false,
        }));
        block.add_instruction(instr(SsaOp::returns(Some(v2))));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "comparison range folding", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        // Should fold Clt to Const(true)
        assert!(changed, "range-known comparison should fold");
        assert!(log.has(EventKind::ConstantFolded));
    }

    /// A wide phi holds every operand that reaches it.
    ///
    /// The shape a flattened dispatcher leaves: a switch fanning out to many
    /// arms that all rejoin, one phi taking a different constant from each arm
    /// and another taking the same value from all of them but one. A phi is
    /// revised from only the operands that moved, so this checks that nothing
    /// an arm contributes is lost along the way, whichever order the arms and
    /// their values are discovered in.
    #[test]
    fn a_wide_phi_holds_every_operand_that_reaches_it() {
        const ARMS: usize = 40;
        let default_block = ARMS + 1;
        let merge = ARMS + 2;
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, ARMS + 4);
        let selector = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let seven = local_at(&mut ssa, 0, 0, 0);
        let zero = local_at(&mut ssa, 1, 0, 1);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: seven,
            value: ConstValue::I32(7),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Switch {
            value: selector,
            targets: (1..=ARMS).collect(),
            default: default_block,
        }));
        ssa.add_block(b0);

        let mut distinct = PhiNode::new(
            ssa.create_variable(
                VariableOrigin::Local(2),
                0,
                DefSite::phi(merge),
                MockType::I32,
            ),
            VariableOrigin::Local(2),
        );
        let mut shared = PhiNode::new(
            ssa.create_variable(
                VariableOrigin::Local(3),
                0,
                DefSite::phi(merge),
                MockType::I32,
            ),
            VariableOrigin::Local(3),
        );
        for arm in 1..=ARMS {
            let index = u16::try_from(arm + 3).expect("few arms");
            let value = local_at(&mut ssa, index, arm, 0);
            let mut block = SsaBlock::new(arm);
            block.add_instruction(instr(SsaOp::Const {
                dest: value,
                value: ConstValue::I32(i32::try_from(arm * 3).expect("small")),
            }));
            block.add_instruction(instr(SsaOp::Jump { target: merge }));
            ssa.add_block(block);
            distinct.add_operand(PhiOperand::new(value, arm));
            shared.add_operand(PhiOperand::new(seven, arm));
        }
        let mut default = SsaBlock::new(default_block);
        default.add_instruction(instr(SsaOp::Jump { target: merge }));
        ssa.add_block(default);
        distinct.add_operand(PhiOperand::new(seven, default_block));
        shared.add_operand(PhiOperand::new(zero, default_block));

        let (distinct_var, shared_var) = (distinct.result(), shared.result());
        let mut exit = SsaBlock::new(merge);
        exit.add_phi(distinct);
        exit.add_phi(shared);
        exit.add_instruction(instr(SsaOp::returns(Some(distinct_var))));
        ssa.add_block(exit);
        ssa.recompute_uses();

        let ranges =
            analyze(&ssa, &WorkBudget::DEFAULT, &BTreeMap::new()).expect("the analysis converges");
        let distinct = ranges.get(distinct_var).expect("the phi has a range");
        for arm in 1..=ARMS {
            let value = i64::try_from(arm * 3).expect("small");
            assert!(
                distinct.may_contain(value),
                "arm {arm}'s {value} is lost: {distinct:?}"
            );
        }
        assert!(
            distinct.may_contain(7),
            "the default's 7 is lost: {distinct:?}"
        );
        assert_eq!(distinct.min(), Some(3), "{distinct:?}");
        assert_eq!(
            distinct.max(),
            Some(i64::try_from(ARMS * 3).expect("small"))
        );

        let shared = ranges.get(shared_var).expect("the phi has a range");
        assert!(shared.may_contain(0) && shared.may_contain(7), "{shared:?}");
        assert_eq!(
            (shared.min(), shared.max()),
            (Some(0), Some(7)),
            "{shared:?}"
        );
    }

    /// A zero-trip loop's latch prunes its own back edge.
    ///
    /// `for (i = 0; i < 0; i++)` reaches the analysis as a phi joining `0` and
    /// `i + 1` under a `branchcmp i < 0` latch. Only the entry edge is
    /// executable when the terminator is first examined, so the phi is `[0, 0]`
    /// and the comparison is decidably false: the back edge must never be
    /// queued, which leaves the phi at `[0, 0]` and the loop provably dead.
    ///
    /// Falling to the catch-all that enqueues every successor would make the
    /// back edge live, join the increment into the phi, and let widening lift
    /// it to `[0, +inf)` - a counter that reads as `Top` for want of a pruned
    /// edge, not for any lattice defect.
    #[test]
    fn a_zero_trip_latch_prunes_its_back_edge() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let zero = local_at(&mut ssa, 0, 0, 0);
        let bound = local_at(&mut ssa, 1, 0, 1);
        let one = local_at(&mut ssa, 2, 1, 0);
        let next = local_at(&mut ssa, 3, 1, 1);
        let counter =
            ssa.create_variable(VariableOrigin::Local(4), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: bound,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        // The body: `i + 1`, reached only if the latch lets it be.
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b1.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: counter,
            right: one,
            flags: None,
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(counter, VariableOrigin::Local(4));
        phi.add_operand(PhiOperand::new(zero, 0));
        phi.add_operand(PhiOperand::new(next, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::BranchCmp {
            left: counter,
            right: bound,
            cmp: CmpKind::Lt,
            unsigned: false,
            true_target: 1,
            false_target: 3,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(Some(counter))));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "zero-trip latch", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        assert!(changed, "the latch is decidably false and must fold");
        let terminator = ssa
            .block(2)
            .and_then(SsaBlock::control_terminator)
            .cloned()
            .expect("block 2 keeps a terminator");
        assert!(
            matches!(terminator, SsaOp::Jump { target: 3 }),
            "the decided latch becomes a jump past the body, got {terminator:?}"
        );
    }

    /// A phi joining constant operands is handled, and the constant branch
    /// before it still folds.
    #[test]
    fn range_propagation_does_not_crash_with_phi() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let v0 = local_at(&mut ssa, 0, 0, 0);
        let v1 = local_at(&mut ssa, 1, 1, 0);
        let phi_var =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);
        let cond = local_at(&mut ssa, 3, 0, 1);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: v0,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: v1,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(phi_var, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(v0, 0));
        phi.add_operand(PhiOperand::new(v1, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::returns(Some(phi_var))));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "phi range propagation", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        assert!(changed, "constant branch before phi should simplify");
    }

    /// An out-of-range branch target is treated as unreachable instead of
    /// panicking.
    #[test]
    fn out_of_range_branch_targets_do_not_panic() {
        // A terminator naming a block the function does not hold is malformed
        // IR - `VerifierError::DanglingBranchTarget` reports it - but a lift
        // that failed to recover a target produces it, and this pass runs
        // before anyone has checked. The `executable_blocks` bitset is sized to
        // `block_count`, so the out-of-range target must be treated as
        // unreachable rather than reach the asserting `BitSet::contains`. Here
        // the condition is an unconstrained argument (range stays `top`), so
        // both the in-range and the out-of-range edge are explored.
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let cond = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 1,
            // Block 99 does not exist - only blocks 0 and 1 are present.
            false_target: 99,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_malformed_boundary(&mut ssa, |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        // Unconstrained condition → nothing folds, and crucially no panic.
        assert!(!changed, "unconstrained branch must not be simplified");
    }

    /// A masked value squared and compared against zero folds, through the
    /// conversions AArch64 and MIPS reach the comparison by.
    ///
    /// Neither architecture has a flags register, so a masked-square opaque
    /// predicate arrives as `BranchCmp v, 0, Ge` over the square rather than as
    /// the flag test x86 needs a bridge for. **This shape folds**, which is the point
    /// of writing it down: the chain `and` a mask, widen, narrow, multiply,
    /// narrow, compare folds on its own, so a real lift that does not fold
    /// differs in something else - operand types, the mask's own width, or
    /// what defines the value being masked.
    #[test]
    fn a_masked_square_compared_against_zero_folds() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let arg = local_at(&mut ssa, 0, 0, 0);
        let mask = local_at(&mut ssa, 1, 0, 1);
        let masked = local_at(&mut ssa, 2, 0, 2);
        let widened = local_at(&mut ssa, 3, 0, 3);
        let narrowed = local_at(&mut ssa, 4, 0, 4);
        let square = local_at(&mut ssa, 5, 0, 5);
        let result = local_at(&mut ssa, 6, 0, 6);
        let zero = local_at(&mut ssa, 7, 0, 7);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: arg,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(0x7fff),
        }));
        b0.add_instruction(instr(SsaOp::And {
            dest: masked,
            left: arg,
            right: mask,
            flags: None,
        }));
        // The two conversions AArch64 reaches its comparison through, and the
        // one MIPS does. Each must keep the masked bound at 64 bits.
        b0.add_instruction(instr(SsaOp::IntConv {
            dest: widened,
            operand: masked,
            target: MockType::I64,
            overflow_check: false,
            unsigned: true,
        }));
        b0.add_instruction(instr(SsaOp::IntConv {
            dest: narrowed,
            operand: widened,
            target: MockType::I32,
            overflow_check: false,
            unsigned: true,
        }));
        b0.add_instruction(instr(SsaOp::Mul {
            dest: square,
            left: narrowed,
            right: narrowed,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::IntConv {
            dest: result,
            operand: square,
            target: MockType::I32,
            overflow_check: false,
            unsigned: true,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::BranchCmp {
            left: result,
            right: zero,
            cmp: CmpKind::Ge,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        for index in 1..=2 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "masked square against zero", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        let folded = ssa.block(0).and_then(SsaBlock::control_terminator);
        assert!(
            matches!(folded, Some(SsaOp::Jump { target: 1 })),
            "a masked square is non-negative, so `>= 0` holds, got {folded:?}"
        );
    }

    /// **A branch over two addresses is decided as the addresses are.**
    /// `deregister_tm_clones` loads `__TMC_END__` twice from its pool and
    /// compares the two: the range domain holds no number for a symbol, and
    /// the comparison of the constants decides it, equal. Signed, two
    /// addresses are ordered by nothing a load base keeps, and the branch
    /// stays.
    #[test]
    fn a_branch_over_two_addresses_is_decided_as_the_addresses_are() {
        let decided = |cmp: CmpKind, unsigned: bool, symbols: (u32, u32)| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
            let first = local_at(&mut ssa, 0, 0, 0);
            let second = local_at(&mut ssa, 1, 0, 1);
            let mut b0 = SsaBlock::new(0);
            b0.add_instruction(instr(SsaOp::Const {
                dest: first,
                value: ConstValue::Symbol(symbols.0),
            }));
            b0.add_instruction(instr(SsaOp::Const {
                dest: second,
                value: ConstValue::Symbol(symbols.1),
            }));
            b0.add_instruction(instr(SsaOp::BranchCmp {
                left: second,
                right: first,
                cmp,
                unsigned,
                true_target: 1,
                false_target: 2,
            }));
            ssa.add_block(b0);
            for index in 1..=2 {
                let mut block = SsaBlock::new(index);
                block.add_instruction(instr(SsaOp::returns(None)));
                ssa.add_block(block);
            }
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            let method = 0u32;
            ValueRangePropagationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
                )
                .ok();
            ssa.block(0).and_then(SsaBlock::control_terminator).cloned()
        };
        assert!(
            matches!(
                decided(CmpKind::Eq, false, (0x250ec, 0x250ec)),
                Some(SsaOp::Jump { target: 1 })
            ),
            "one address twice is equal"
        );
        assert!(
            matches!(
                decided(CmpKind::Lt, true, (0x250ec, 0x25100)),
                Some(SsaOp::Jump { target: 2 })
            ),
            "ordered unsigned as the addresses"
        );
        assert!(
            matches!(
                decided(CmpKind::Lt, false, (0x250ec, 0x25100)),
                Some(SsaOp::BranchCmp { .. })
            ),
            "control: no signed order"
        );
    }

    /// A masked square spelled with a shift pair folds: `(x << 49) >> 49`
    /// keeps the low 15 bits of a 64-bit register, which is how RISC-V's
    /// `slli`/`srli` mask a value, so the logical shift alone bounds it to
    /// `[0, 2^15 - 1]` whatever `x` is, and the square stays below `2^31`.
    #[test]
    fn a_logical_shift_by_a_constant_bounds_what_it_reads() {
        let wide = |ssa: &mut SsaFunction<MockTarget>, idx: u16, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(idx),
                0,
                DefSite::instruction(0, at),
                MockType::I64,
            )
        };
        let build = |count: i32| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
            let arg = wide(&mut ssa, 0, 0);
            let amount = local_at(&mut ssa, 1, 0, 1);
            let up = wide(&mut ssa, 2, 2);
            let down = wide(&mut ssa, 3, 3);
            let narrowed = local_at(&mut ssa, 4, 0, 4);
            let square = local_at(&mut ssa, 5, 0, 5);
            let zero = local_at(&mut ssa, 6, 0, 6);
            let mut b0 = SsaBlock::new(0);
            b0.add_instruction(instr(SsaOp::LoadArg {
                dest: arg,
                arg_index: 0,
            }));
            b0.add_instruction(instr(SsaOp::Const {
                dest: amount,
                value: ConstValue::I32(count),
            }));
            b0.add_instruction(instr(SsaOp::Shl {
                dest: up,
                value: arg,
                amount,
                flags: None,
            }));
            b0.add_instruction(instr(SsaOp::Shr {
                dest: down,
                value: up,
                amount,
                unsigned: true,
                flags: None,
            }));
            b0.add_instruction(instr(SsaOp::IntConv {
                dest: narrowed,
                operand: down,
                target: MockType::I32,
                overflow_check: false,
                unsigned: true,
            }));
            b0.add_instruction(instr(SsaOp::Mul {
                dest: square,
                left: narrowed,
                right: narrowed,
                flags: None,
            }));
            b0.add_instruction(instr(SsaOp::Const {
                dest: zero,
                value: ConstValue::I32(0),
            }));
            b0.add_instruction(instr(SsaOp::BranchCmp {
                left: square,
                right: zero,
                cmp: CmpKind::Lt,
                unsigned: false,
                true_target: 1,
                false_target: 2,
            }));
            ssa.add_block(b0);
            for index in 1..=2 {
                let mut block = SsaBlock::new(index);
                block.add_instruction(instr(SsaOp::returns(None)));
                ssa.add_block(block);
            }
            ssa.recompute_uses();
            ssa
        };
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let mut masked = build(49);
        run_mock_pass_boundary(&mut masked, "shift-masked square", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        assert!(
            matches!(
                masked.block(0).and_then(SsaBlock::control_terminator),
                Some(SsaOp::Jump { target: 2 })
            ),
            "a 15-bit square is never negative at 32 bits"
        );
        // The control: a shift of zero clears nothing, so the square is free.
        let mut unmasked = build(0);
        run_mock_pass_boundary(&mut unmasked, "unmasked square", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        assert!(matches!(
            unmasked.block(0).and_then(SsaBlock::control_terminator),
            Some(SsaOp::BranchCmp { .. })
        ));
    }

    /// A binary search over a table of `entries`, dash's `find_builtin`:
    /// `lo = 0`, `hi = entries`, and while `lo < hi` (unsigned) the midpoint
    /// `mid = (lo + hi) >> 1` halves the interval, `hi = mid` or `lo = mid +
    /// 1` as an unknown comparison decides. Returns the function and `mid`.
    fn binary_search(entries: u32) -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 6);
        let compared = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let typed = |ssa: &mut SsaFunction<MockTarget>, index: u16, block: usize, at: usize| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(block, at),
                MockType::U32,
            )
        };
        let start = typed(&mut ssa, 0, 0, 0);
        let end = typed(&mut ssa, 1, 0, 1);
        let lo = ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(1), MockType::U32);
        let hi = ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(1), MockType::U32);
        let sum = typed(&mut ssa, 4, 2, 0);
        let one_bit = ssa.create_variable(
            VariableOrigin::Local(5),
            0,
            DefSite::instruction(2, 1),
            MockType::U8,
        );
        let mid = typed(&mut ssa, 6, 2, 2);
        let zero = local_at(&mut ssa, 7, 2, 3);
        let one = typed(&mut ssa, 8, 4, 0);
        let above = typed(&mut ssa, 9, 4, 1);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: start,
            value: ConstValue::U32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: end,
            value: ConstValue::U32(entries),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut low = PhiNode::new(lo, VariableOrigin::Local(2));
        low.add_operand(PhiOperand::new(start, 0));
        low.add_operand(PhiOperand::new(lo, 3));
        low.add_operand(PhiOperand::new(above, 4));
        b1.add_phi(low);
        let mut high = PhiNode::new(hi, VariableOrigin::Local(3));
        high.add_operand(PhiOperand::new(end, 0));
        high.add_operand(PhiOperand::new(mid, 3));
        high.add_operand(PhiOperand::new(hi, 4));
        b1.add_phi(high);
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: lo,
            right: hi,
            cmp: CmpKind::Ge,
            unsigned: true,
            true_target: 5,
            false_target: 2,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Add {
            dest: sum,
            left: lo,
            right: hi,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: one_bit,
            value: ConstValue::U8(1),
        }));
        b2.add_instruction(instr(SsaOp::Shr {
            dest: mid,
            value: sum,
            amount: one_bit,
            unsigned: true,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b2.add_instruction(instr(SsaOp::BranchCmp {
            left: compared,
            right: zero,
            cmp: CmpKind::Lt,
            unsigned: false,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::U32(1),
        }));
        b4.add_instruction(instr(SsaOp::Add {
            dest: above,
            left: mid,
            right: one,
            flags: None,
        }));
        b4.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b4);

        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::returns(Some(lo))));
        ssa.add_block(b5);
        ssa.recompute_uses();
        (ssa, mid)
    }

    /// **A binary search's midpoint lies within the table it halves.** `lo`
    /// and `hi` start at `0` and `39` and only close in, and the body runs
    /// only while `lo < hi`, so `mid = (lo + hi) >> 1` is one of `[0, 38]`:
    /// the index the search reads a 39-entry table at. Widening the bounds
    /// as loop-carried left the midpoint unbounded, and the table the search
    /// reads no object.
    #[test]
    fn a_binary_searchs_midpoint_lies_within_the_table_it_halves() {
        let (ssa, mid) = binary_search(39);
        let ranges = rounds(64).get(&ssa).expect("converges");
        let range = ranges.get(mid).cloned();
        assert_eq!(
            range
                .as_ref()
                .and_then(ValueRange::min)
                .zip(range.as_ref().and_then(ValueRange::max)),
            Some((0, 38)),
            "{range:?}"
        );
    }

    /// The binary search as a compiler rotates it, dash-arm's: the body runs
    /// first, the arms meet in a latch whose phis the exit test compares
    /// (`lo >= hi` unsigned leaves), and `hi` takes `mid + reach` on one arm.
    /// Returns the function and `mid`.
    fn rotated_search(entries: u32, reach: u32) -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 6);
        let compared = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let typed = |ssa: &mut SsaFunction<MockTarget>, index: u16, site: DefSite| {
            ssa.create_variable(VariableOrigin::Local(index), 0, site, MockType::U32)
        };
        let start = typed(&mut ssa, 0, DefSite::instruction(0, 0));
        let end = typed(&mut ssa, 1, DefSite::instruction(0, 1));
        let lo = typed(&mut ssa, 2, DefSite::phi(1));
        let hi = typed(&mut ssa, 3, DefSite::phi(1));
        let sum = typed(&mut ssa, 4, DefSite::instruction(1, 0));
        let one_bit = ssa.create_variable(
            VariableOrigin::Local(5),
            0,
            DefSite::instruction(1, 1),
            MockType::U8,
        );
        let mid = typed(&mut ssa, 6, DefSite::instruction(1, 2));
        let zero = local_at(&mut ssa, 7, 1, 3);
        let step = typed(&mut ssa, 8, DefSite::instruction(2, 0));
        let below = typed(&mut ssa, 9, DefSite::instruction(2, 1));
        let one = typed(&mut ssa, 10, DefSite::instruction(3, 0));
        let above = typed(&mut ssa, 11, DefSite::instruction(3, 1));
        let next_lo = typed(&mut ssa, 12, DefSite::phi(4));
        let next_hi = typed(&mut ssa, 13, DefSite::phi(4));

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: start,
            value: ConstValue::U32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: end,
            value: ConstValue::U32(entries),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut low = PhiNode::new(lo, VariableOrigin::Local(2));
        low.add_operand(PhiOperand::new(start, 0));
        low.add_operand(PhiOperand::new(next_lo, 4));
        b1.add_phi(low);
        let mut high = PhiNode::new(hi, VariableOrigin::Local(3));
        high.add_operand(PhiOperand::new(end, 0));
        high.add_operand(PhiOperand::new(next_hi, 4));
        b1.add_phi(high);
        b1.add_instruction(instr(SsaOp::Add {
            dest: sum,
            left: lo,
            right: hi,
            flags: None,
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: one_bit,
            value: ConstValue::U8(1),
        }));
        b1.add_instruction(instr(SsaOp::Shr {
            dest: mid,
            value: sum,
            amount: one_bit,
            unsigned: true,
            flags: None,
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: compared,
            right: zero,
            cmp: CmpKind::Lt,
            unsigned: false,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: step,
            value: ConstValue::U32(reach),
        }));
        b2.add_instruction(instr(SsaOp::Add {
            dest: below,
            left: mid,
            right: step,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 4 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::U32(1),
        }));
        b3.add_instruction(instr(SsaOp::Add {
            dest: above,
            left: mid,
            right: one,
            flags: None,
        }));
        b3.add_instruction(instr(SsaOp::Jump { target: 4 }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        let mut latch_low = PhiNode::new(next_lo, VariableOrigin::Local(12));
        latch_low.add_operand(PhiOperand::new(lo, 2));
        latch_low.add_operand(PhiOperand::new(above, 3));
        b4.add_phi(latch_low);
        let mut latch_high = PhiNode::new(next_hi, VariableOrigin::Local(13));
        latch_high.add_operand(PhiOperand::new(below, 2));
        latch_high.add_operand(PhiOperand::new(hi, 3));
        b4.add_phi(latch_high);
        b4.add_instruction(instr(SsaOp::BranchCmp {
            left: next_lo,
            right: next_hi,
            cmp: CmpKind::Ge,
            unsigned: true,
            true_target: 5,
            false_target: 1,
        }));
        ssa.add_block(b4);

        let mut b5 = SsaBlock::new(5);
        b5.add_instruction(instr(SsaOp::returns(Some(next_lo))));
        ssa.add_block(b5);
        ssa.recompute_uses();
        (ssa, mid)
    }

    /// The bounds of `var` the analysis proves for `ssa`.
    fn bounds_of(ssa: &SsaFunction<MockTarget>, var: SsaVarId) -> Option<(i64, i64)> {
        let ranges = rounds(64).get(ssa).expect("converges");
        let range = ranges.get(var)?;
        range.min().zip(range.max())
    }

    /// **The rotated search's midpoint lies within its table too, and a
    /// search that can step past its bounds proves nothing.** The exit test
    /// compares the latch's phis, so the bounds are read on the back edge it
    /// leaves by, `lo < hi`: `mid` is one of `[0, 38]`. Where one arm sets
    /// `hi = mid + 2`, `hi` can reach `40`, past what entered, and the guess
    /// fails: `mid` stays as widening left it.
    #[test]
    fn a_rotated_searchs_midpoint_lies_within_its_table_and_an_escaping_one_proves_nothing() {
        let (ssa, mid) = rotated_search(39, 0);
        assert_eq!(bounds_of(&ssa, mid), Some((0, 38)));
        let (ssa, mid) = rotated_search(39, 2);
        assert!(
            bounds_of(&ssa, mid).is_none_or(|(_, max)| max > 39),
            "{:?}",
            bounds_of(&ssa, mid)
        );
    }

    /// **A search whose one arm leaves straight from its head narrows to its
    /// table.** dash-arm's `find_builtin`: the head computes `mid` and goes
    /// either to the latch at once (`hi = mid`) or through a block setting
    /// `lo = mid + 1`; the latch's exit test, `lo >= hi` unsigned, returns to
    /// the head. The head's `lo` is first joined before the test's `hi` is
    /// known and reads the whole guess then; the narrowed run holds it below
    /// `hi`, and `mid` is one of `[0, 38]`.
    #[test]
    fn a_search_leaving_its_head_straight_for_the_latch_narrows_to_its_table() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 5);
        let compared = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let typed = |ssa: &mut SsaFunction<MockTarget>, index: u16, site: DefSite| {
            ssa.create_variable(VariableOrigin::Local(index), 0, site, MockType::U32)
        };
        let start = typed(&mut ssa, 0, DefSite::instruction(0, 0));
        let end = typed(&mut ssa, 1, DefSite::instruction(0, 1));
        let lo = typed(&mut ssa, 2, DefSite::phi(1));
        let hi = typed(&mut ssa, 3, DefSite::phi(1));
        let sum = typed(&mut ssa, 4, DefSite::instruction(1, 0));
        let one_bit = ssa.create_variable(
            VariableOrigin::Local(5),
            0,
            DefSite::instruction(1, 1),
            MockType::U8,
        );
        let mid = typed(&mut ssa, 6, DefSite::instruction(1, 2));
        let zero = local_at(&mut ssa, 7, 1, 3);
        let one = typed(&mut ssa, 8, DefSite::instruction(2, 0));
        let above = typed(&mut ssa, 9, DefSite::instruction(2, 1));
        let next_lo = typed(&mut ssa, 10, DefSite::phi(3));
        let next_hi = typed(&mut ssa, 11, DefSite::phi(3));

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: start,
            value: ConstValue::U32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: end,
            value: ConstValue::U32(39),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut low = PhiNode::new(lo, VariableOrigin::Local(2));
        low.add_operand(PhiOperand::new(start, 0));
        low.add_operand(PhiOperand::new(next_lo, 3));
        b1.add_phi(low);
        let mut high = PhiNode::new(hi, VariableOrigin::Local(3));
        high.add_operand(PhiOperand::new(end, 0));
        high.add_operand(PhiOperand::new(next_hi, 3));
        b1.add_phi(high);
        b1.add_instruction(instr(SsaOp::Add {
            dest: sum,
            left: lo,
            right: hi,
            flags: None,
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: one_bit,
            value: ConstValue::U8(1),
        }));
        b1.add_instruction(instr(SsaOp::Shr {
            dest: mid,
            value: sum,
            amount: one_bit,
            unsigned: true,
            flags: None,
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: compared,
            right: zero,
            cmp: CmpKind::Lt,
            unsigned: false,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::U32(1),
        }));
        b2.add_instruction(instr(SsaOp::Add {
            dest: above,
            left: mid,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        let mut latch_low = PhiNode::new(next_lo, VariableOrigin::Local(10));
        latch_low.add_operand(PhiOperand::new(above, 2));
        latch_low.add_operand(PhiOperand::new(lo, 1));
        b3.add_phi(latch_low);
        let mut latch_high = PhiNode::new(next_hi, VariableOrigin::Local(11));
        latch_high.add_operand(PhiOperand::new(hi, 2));
        latch_high.add_operand(PhiOperand::new(mid, 1));
        b3.add_phi(latch_high);
        b3.add_instruction(instr(SsaOp::BranchCmp {
            left: next_lo,
            right: next_hi,
            cmp: CmpKind::Ge,
            unsigned: true,
            true_target: 4,
            false_target: 1,
        }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::returns(Some(next_lo))));
        ssa.add_block(b4);
        ssa.recompute_uses();

        assert_eq!(bounds_of(&ssa, mid), Some((0, 38)));
        assert_eq!(bounds_of(&ssa, lo), Some((0, 38)));
        assert_eq!(bounds_of(&ssa, hi), Some((1, 39)));
    }

    /// A counted loop, `for (i = start; i < limit; i++)`: returns the function,
    /// `i` and the body's `i + 1`.
    fn counted_loop(limit: Option<i32>) -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 4);
        let unknown = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let start = local_at(&mut ssa, 0, 0, 0);
        let bound = local_at(&mut ssa, 1, 0, 1);
        let counter =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(1), MockType::I32);
        let one = local_at(&mut ssa, 3, 2, 0);
        let next = local_at(&mut ssa, 4, 2, 1);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: start,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(match limit {
            Some(limit) => SsaOp::Const {
                dest: bound,
                value: ConstValue::I32(limit),
            },
            None => SsaOp::Copy {
                dest: bound,
                src: unknown,
            },
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(counter, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(start, 0));
        phi.add_operand(PhiOperand::new(next, 2));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: counter,
            right: bound,
            cmp: CmpKind::Lt,
            unsigned: false,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: counter,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(Some(counter))));
        ssa.add_block(b3);
        ssa.recompute_uses();
        (ssa, counter, next)
    }

    /// **A counted loop's counter stays within its bound.** `i < 10` from `0`
    /// holds `i` within `[0, 10]` and its step within `[1, 10]`, where
    /// widening left them unbounded above. Against a bound nothing proves, the
    /// counter stays as widening left it.
    #[test]
    fn a_counted_loops_counter_stays_within_its_bound() {
        let (ssa, counter, next) = counted_loop(Some(10));
        assert_eq!(bounds_of(&ssa, counter), Some((0, 10)));
        assert_eq!(bounds_of(&ssa, next), Some((1, 10)));
        let (ssa, counter, _) = counted_loop(None);
        assert!(
            bounds_of(&ssa, counter).is_none_or(|(_, max)| max > 10),
            "{:?}",
            bounds_of(&ssa, counter)
        );
    }

    /// A bottom-tested loop leaving by equality: `i` counts from `0` by
    /// `step` and the loop runs again while `i + step != limit`. With
    /// `widened`, `i` is a 64-bit phi the body reads truncated to 32 bits and
    /// writes back zero-extended, as an x86-64 lift spells an `int` counter.
    /// Returns the function, `i` and the body's `i + step`.
    fn equality_exit_loop(
        step: i32,
        limit: i32,
        widened: bool,
    ) -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let wide = if widened {
            MockType::I64
        } else {
            MockType::I32
        };
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 7);
        let start = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            wide,
        );
        let counter = ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(1), wide);
        let low = local_at(&mut ssa, 2, 1, 0);
        let stride = local_at(&mut ssa, 3, 1, 1);
        let next = local_at(&mut ssa, 4, 1, 2);
        let back = ssa.create_variable(
            VariableOrigin::Local(5),
            0,
            DefSite::instruction(1, 3),
            wide,
        );
        let bound = local_at(&mut ssa, 6, 1, 4);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: start,
            value: if widened {
                ConstValue::I64(0)
            } else {
                ConstValue::I32(0)
            },
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(counter, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(start, 0));
        phi.add_operand(PhiOperand::new(back, 1));
        b1.add_phi(phi);
        b1.add_instruction(instr(if widened {
            SsaOp::IntConv {
                dest: low,
                operand: counter,
                target: MockType::I32,
                overflow_check: false,
                unsigned: false,
            }
        } else {
            SsaOp::Copy {
                dest: low,
                src: counter,
            }
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: stride,
            value: ConstValue::I32(step),
        }));
        b1.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: low,
            right: stride,
            flags: None,
        }));
        b1.add_instruction(instr(if widened {
            SsaOp::IntConv {
                dest: back,
                operand: next,
                target: MockType::I64,
                overflow_check: false,
                unsigned: true,
            }
        } else {
            SsaOp::Copy {
                dest: back,
                src: next,
            }
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: bound,
            value: ConstValue::I32(limit),
        }));
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: next,
            right: bound,
            cmp: CmpKind::Ne,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(Some(low))));
        ssa.add_block(b2);
        ssa.recompute_uses();
        (ssa, counter, next)
    }

    /// **A counter leaving by equality stays below the value it leaves at.**
    /// `i + 1 != 512` from `0` holds `i` within `[0, 511]`, whether the
    /// branch compares the counter's own width or a truncated copy of a
    /// 64-bit phi written back zero-extended. The branch compares `i + 1`,
    /// not the phi, so the bound is guessed through the step, and the back
    /// edge carries the guard through the conversion it writes back by.
    #[test]
    fn a_counter_leaving_by_equality_stays_below_where_it_leaves() {
        for widened in [false, true] {
            let (ssa, counter, next) = equality_exit_loop(1, 512, widened);
            assert_eq!(
                bounds_of(&ssa, counter),
                Some((0, 511)),
                "widened {widened}"
            );
            assert_eq!(bounds_of(&ssa, next), Some((1, 512)), "widened {widened}");
        }
    }

    /// **A guard on a narrowed copy bounds the counter it narrows.** A 64-bit
    /// `i` from `0` runs its body while `(int32_t)i < 10`, as a lift reads a
    /// 32-bit comparison of a register the loop carries whole. The body's `i`
    /// fits 32 bits, so the guard on its low half bounds all of it: `i`
    /// within `[0, 10]`, the body's `i + 1` within `[1, 10]`.
    #[test]
    fn a_guard_on_a_narrowed_copy_bounds_the_counter_it_narrows() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 6);
        let start = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I64,
        );
        let counter =
            ssa.create_variable(VariableOrigin::Local(1), 0, DefSite::phi(1), MockType::I64);
        let low = local_at(&mut ssa, 2, 1, 0);
        let bound = local_at(&mut ssa, 3, 1, 1);
        let one = ssa.create_variable(
            VariableOrigin::Local(4),
            0,
            DefSite::instruction(2, 0),
            MockType::I64,
        );
        let next = ssa.create_variable(
            VariableOrigin::Local(5),
            0,
            DefSite::instruction(2, 1),
            MockType::I64,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: start,
            value: ConstValue::I64(0),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(counter, VariableOrigin::Local(1));
        phi.add_operand(PhiOperand::new(start, 0));
        phi.add_operand(PhiOperand::new(next, 2));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::IntConv {
            dest: low,
            operand: counter,
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
        }));
        b1.add_instruction(instr(SsaOp::Const {
            dest: bound,
            value: ConstValue::I32(10),
        }));
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: low,
            right: bound,
            cmp: CmpKind::Lt,
            unsigned: false,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I64(1),
        }));
        b2.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: counter,
            right: one,
            flags: None,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(Some(low))));
        ssa.add_block(b3);
        ssa.recompute_uses();

        assert_eq!(bounds_of(&ssa, counter), Some((0, 10)));
        assert_eq!(bounds_of(&ssa, next), Some((1, 10)));
    }

    /// **A strided counter leaving by equality stays below where it leaves.**
    /// From `0` by `4`, `i + 4 != 512` holds `i` within `[0, 508]`: every
    /// value `i + 4` takes is a multiple of four, so below `512` it is at
    /// most `508`.
    #[test]
    fn a_strided_counter_leaving_by_equality_stays_below_where_it_leaves() {
        for widened in [false, true] {
            let (ssa, counter, next) = equality_exit_loop(4, 512, widened);
            assert_eq!(
                bounds_of(&ssa, counter),
                Some((0, 508)),
                "widened {widened}"
            );
            assert_eq!(bounds_of(&ssa, next), Some((4, 512)), "widened {widened}");
        }
    }

    /// **A counter that can step over the value it leaves at is unbounded.**
    /// From `0` by `3`, `i + 3` is never `512`: the counter wraps through
    /// its width, and no guessed interval holds.
    #[test]
    fn a_counter_stepping_over_its_exit_value_stays_unbounded() {
        for widened in [false, true] {
            let (ssa, counter, _) = equality_exit_loop(3, 512, widened);
            assert!(
                bounds_of(&ssa, counter).is_none_or(|(_, max)| max > 512),
                "widened {widened}: {:?}",
                bounds_of(&ssa, counter)
            );
        }
    }

    /// **A counter bounded by a value reached late is bounded by all of it.**
    /// `n` is `3` on the short path into the loop and `10` on a longer one,
    /// so the run may read the body's `i < n` while `n` is still `3`; the
    /// counter still reaches `10`, which the run's last reads of the body
    /// under the grown `n` have to say.
    #[test]
    fn a_counter_bounded_by_a_value_reached_late_is_bounded_by_all_of_it() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 8);
        let unknown = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let sign = local_at(&mut ssa, 0, 0, 0);
        let three = local_at(&mut ssa, 1, 1, 0);
        let ten = local_at(&mut ssa, 2, 3, 0);
        let bound =
            ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(4), MockType::I32);
        let start = local_at(&mut ssa, 4, 4, 0);
        let counter =
            ssa.create_variable(VariableOrigin::Local(5), 0, DefSite::phi(5), MockType::I32);
        let one = local_at(&mut ssa, 6, 6, 0);
        let next = local_at(&mut ssa, 7, 6, 1);
        let constant = |dest, value| {
            instr(SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            })
        };

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(constant(sign, 0));
        b0.add_instruction(instr(SsaOp::BranchCmp {
            left: unknown,
            right: sign,
            cmp: CmpKind::Lt,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(constant(three, 3));
        b1.add_instruction(instr(SsaOp::Jump { target: 4 }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(constant(ten, 10));
        b3.add_instruction(instr(SsaOp::Jump { target: 4 }));
        ssa.add_block(b3);
        let mut b4 = SsaBlock::new(4);
        let mut merged = PhiNode::new(bound, VariableOrigin::Local(3));
        merged.add_operand(PhiOperand::new(three, 1));
        merged.add_operand(PhiOperand::new(ten, 3));
        b4.add_phi(merged);
        b4.add_instruction(constant(start, 0));
        b4.add_instruction(instr(SsaOp::Jump { target: 5 }));
        ssa.add_block(b4);
        let mut b5 = SsaBlock::new(5);
        let mut phi = PhiNode::new(counter, VariableOrigin::Local(5));
        phi.add_operand(PhiOperand::new(start, 4));
        phi.add_operand(PhiOperand::new(next, 6));
        b5.add_phi(phi);
        b5.add_instruction(instr(SsaOp::BranchCmp {
            left: counter,
            right: bound,
            cmp: CmpKind::Lt,
            unsigned: false,
            true_target: 6,
            false_target: 7,
        }));
        ssa.add_block(b5);
        let mut b6 = SsaBlock::new(6);
        b6.add_instruction(constant(one, 1));
        b6.add_instruction(instr(SsaOp::Add {
            dest: next,
            left: counter,
            right: one,
            flags: None,
        }));
        b6.add_instruction(instr(SsaOp::Jump { target: 5 }));
        ssa.add_block(b6);
        let mut b7 = SsaBlock::new(7);
        b7.add_instruction(instr(SsaOp::returns(Some(counter))));
        ssa.add_block(b7);
        ssa.recompute_uses();

        let bounds = bounds_of(&ssa, counter);
        assert!(
            bounds.is_none_or(|(min, max)| min <= 0 && max >= 10),
            "the counter reaches 10: {bounds:?}"
        );
        assert_eq!(bounds, Some((0, 10)));
    }

    /// **A sweep revises a read a guard left stale.** A guard refines a read by
    /// another value's range, and nothing queues the read again when only
    /// that value moves. A run checking `i < 10`'s invariant with the body's
    /// `i + 1` left at `[1, 3]`, as if read while the bound was `3`, revises
    /// it to `[1, 10]` in one sweep, and the sweep after that is quiet.
    #[test]
    fn a_sweep_revises_a_read_a_guard_left_stale() {
        let (ssa, counter, next) = counted_loop(Some(10));
        let cfg = SsaCfg::from_ssa(&ssa);
        let mut analysis: RangeAnalysis<MockTarget> = RangeAnalysis::new(64);
        analysis.assumed = Some(Assumption {
            held: [(counter, ValueRange::bounded(0, 10))]
                .into_iter()
                .collect(),
            phis: [counter].into_iter().collect(),
            dominators: compute_dominators(&cfg, cfg.entry()),
            predecessors: predecessor_counts(&cfg, ssa.block_count()),
        });
        analysis.initialize(&ssa, &cfg);
        assert!(analysis.propagate(&ssa, &cfg));
        assert_eq!(analysis.get_range(next), ValueRange::bounded(1, 10));
        analysis.ranges.insert(next, ValueRange::bounded(1, 3));
        analysis.sweep_assumed(&ssa, &cfg);
        assert!(analysis.propagate(&ssa, &cfg));
        assert_eq!(analysis.get_range(next), ValueRange::bounded(1, 10));
        analysis.sweep_assumed(&ssa, &cfg);
        assert!(analysis.ssa_worklist.is_empty() && analysis.cfg_worklist.is_empty());
    }

    /// An operation's width is the operation's, not its destination variable's
    /// recovered type.
    ///
    /// `Mul { dest, left, right }` is one width throughout - `WideMul` is the
    /// separate widening op - so when type recovery leaves `dest` `Unknown` the
    /// operands still say what the machine multiplied in. Reading `dest` alone
    /// would make a *missing type* erase a *proved range*: a product of unknown
    /// width is `Top`, so `[0, 32767]` squared would come out unbounded and
    /// every test downstream of it would become undecidable.
    #[test]
    fn an_untyped_destination_takes_its_width_from_its_operands() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let arg = local_at(&mut ssa, 0, 0, 0);
        let mask = local_at(&mut ssa, 1, 0, 1);
        let masked = local_at(&mut ssa, 2, 0, 2);
        // The destination the lift left untyped, which is the whole point.
        let square = ssa.create_variable(
            VariableOrigin::Local(3),
            0,
            DefSite::instruction(0, 3),
            MockType::Unknown,
        );
        let zero = local_at(&mut ssa, 4, 0, 4);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: arg,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(0x7fff),
        }));
        b0.add_instruction(instr(SsaOp::And {
            dest: masked,
            left: arg,
            right: mask,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Mul {
            dest: square,
            left: masked,
            right: masked,
            flags: None,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::BranchCmp {
            left: square,
            right: zero,
            cmp: CmpKind::Ge,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        for index in 1..=2 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "untyped multiply", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        let folded = ssa.block(0).and_then(SsaBlock::control_terminator);
        assert!(
            matches!(folded, Some(SsaOp::Jump { target: 1 })),
            "the operands are 32-bit, so the product is, and it fits: {folded:?}"
        );
    }

    /// AND is bounded by whichever side is non-negative, constant or not.
    ///
    /// One rule, and a constant mask is a case of it: an operand that is
    /// non-negative and at most `m` bounds the result to `[0, m]` whatever the
    /// other side holds, because its clear sign bit clears the result's and no
    /// bit above `m`'s highest survives. Bounding by a constant mask alone
    /// would answer `[0, 1]` for `0 & 1` - the imprecision AArch64's
    /// `(square >> 31) & 1` sign test turns into an undecidable branch.
    #[test]
    fn and_is_bounded_by_whichever_side_is_non_negative() {
        let evaluate = |left: ValueRange, right: ValueRange| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
            let a = local_at(&mut ssa, 0, 0, 0);
            let b = local_at(&mut ssa, 1, 0, 1);
            let dest = local_at(&mut ssa, 2, 0, 2);
            let mut analysis: RangeAnalysis<MockTarget> = RangeAnalysis::new(20);
            analysis.ranges.insert(a, left);
            analysis.ranges.insert(b, right);
            analysis.evaluate_instruction(
                &SsaOp::And {
                    dest,
                    left: a,
                    right: b,
                    flags: None,
                },
                &ssa,
            )
        };

        // `5 & 3` is `1`. The bound below would answer `[0, 3]`, which is sound
        // and weaker, so this pins the exact fold rather than something the
        // general rule happens to agree with.
        assert_eq!(
            evaluate(ValueRange::constant(5), ValueRange::constant(3)),
            ValueRange::constant(1),
            "two constants give the constant AND, not a bound around it"
        );
        assert_eq!(
            evaluate(ValueRange::bounded(0, 7), ValueRange::bounded(0, 1_000)),
            ValueRange::bounded(0, 7),
            "the tighter non-negative side bounds the result"
        );
        assert_eq!(
            evaluate(ValueRange::top(), ValueRange::bounded(0, 15)),
            ValueRange::bounded(0, 15),
            "a non-negative mask bounds the result whatever the other side is"
        );
        assert!(
            evaluate(ValueRange::top(), ValueRange::bounded(-16, -1)).is_top(),
            "a negative operand clears nothing reliably"
        );
    }

    /// When a value moves, the flags it set move with it.
    ///
    /// A flags variable is a *secondary* definition, recorded as `Top` once and
    /// left there, so `update_value` never reports it changed and nothing
    /// re-runs its users. For a flags variable that is wrong: `BranchFlags`
    /// names the flags, and what decides it is the producing instruction's
    /// result (see [`RangeAnalysis::flag_source_range`]). The IR gives that
    /// dependency no operand, so the worklist has to carry it by hand - and
    /// without it the fixpoint decides a flag-tested branch once, on whatever
    /// the producer was worth on the first visit, and never again.
    ///
    /// A flattened dispatcher is the shape that needs it: `test state, state`
    /// with the state phi still holding only its entry value, so without the
    /// requeue the second arm of the dispatcher is never explored and the
    /// accumulation it guards is deleted. This asserts the queueing directly,
    /// because the end-to-end consequence needs a dispatcher to show and a
    /// smaller graph re-propagates through other paths anyway.
    #[test]
    fn a_moved_result_queues_the_flags_it_set() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let operand = local_at(&mut ssa, 0, 0, 0);
        let result = local_at(&mut ssa, 1, 0, 1);
        let flags = local_at(&mut ssa, 2, 0, 1);

        let mut block = SsaBlock::new(0);
        block.add_instruction(instr(SsaOp::LoadArg {
            dest: operand,
            arg_index: 0,
        }));
        block.add_instruction(instr(SsaOp::And {
            dest: result,
            left: operand,
            right: operand,
            flags: Some(flags),
        }));
        block.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(block);
        ssa.recompute_uses();

        let instruction = ssa
            .block(0)
            .and_then(|block| block.instruction(1))
            .expect("the `and`")
            .clone();

        let mut analysis: RangeAnalysis<MockTarget> = RangeAnalysis::new(20);
        analysis.ranges.insert(operand, ValueRange::constant(0));
        analysis.update_instruction_defs(&instruction, 0, &ssa, Cause::Moved);
        analysis.ssa_worklist.clear();

        // The operand widens, as a loop-carried phi does when its back edge is
        // first taken.
        analysis.ranges.insert(operand, ValueRange::bounded(0, 1));
        analysis.update_instruction_defs(&instruction, 0, &ssa, Cause::Moved);
        assert!(
            analysis.ssa_worklist.contains(&flags),
            "the flags have to be re-queued so the branch testing them is asked \
             again; queued {:?}",
            analysis.ssa_worklist
        );
    }

    /// A flag-tested branch is decided from the range of the value that set the
    /// flags.
    ///
    /// The shape is the masked-square opaque predicate and, at `-O0`, every
    /// x86 comparison: mask a value, square it, `test` the square against
    /// itself, branch on the sign bit. The interval domain bounds the square at
    /// `[0, 0x3FFF0001]` and the branch tests a *flags* variable, whose own
    /// range is `Top` by construction - so without the bridge the fold is one
    /// instruction out of reach.
    #[test]
    fn a_sign_flag_branch_is_decided_by_its_producers_range() {
        let folded = flag_branch_over_a_masked_square(FlagCondition::Positive);
        assert!(
            matches!(folded, Some(SsaOp::Jump { target: 1 })),
            "a non-negative square clears the sign bit, got {folded:?}"
        );
        let folded = flag_branch_over_a_masked_square(FlagCondition::Negative);
        assert!(
            matches!(folded, Some(SsaOp::Jump { target: 2 })),
            "and never sets it, got {folded:?}"
        );
        // Carry is not a function of the result's range, so it stays a branch
        // however well the square is bounded.
        let kept = flag_branch_over_a_masked_square(FlagCondition::Carry);
        assert!(
            matches!(kept, Some(SsaOp::BranchFlags { .. })),
            "carry is not decidable from the result, got {kept:?}"
        );
    }

    /// Builds `((arg & 0x7fff) * (arg & 0x7fff))` tested against itself, and
    /// returns the entry block's terminator after the pass.
    fn flag_branch_over_a_masked_square(condition: FlagCondition) -> Option<SsaOp<MockTarget>> {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let arg = local_at(&mut ssa, 0, 0, 0);
        let mask = local_at(&mut ssa, 1, 0, 1);
        let masked = local_at(&mut ssa, 2, 0, 2);
        let mask_flags = local_at(&mut ssa, 3, 0, 2);
        let square = local_at(&mut ssa, 4, 0, 3);
        let square_flags = local_at(&mut ssa, 5, 0, 3);
        let tested = local_at(&mut ssa, 6, 0, 4);
        let test_flags = local_at(&mut ssa, 7, 0, 4);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: arg,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: mask,
            value: ConstValue::I32(0x7fff),
        }));
        b0.add_instruction(instr(SsaOp::And {
            dest: masked,
            left: arg,
            right: mask,
            flags: Some(mask_flags),
        }));
        b0.add_instruction(instr(SsaOp::Mul {
            dest: square,
            left: masked,
            right: masked,
            flags: Some(square_flags),
        }));
        // `test reg, reg`: the x86 idiom whose range the flag test reads.
        b0.add_instruction(instr(SsaOp::And {
            dest: tested,
            left: square,
            right: square,
            flags: Some(test_flags),
        }));
        b0.add_instruction(instr(SsaOp::BranchFlags {
            flags: test_flags,
            condition,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);

        for index in 1..=2 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "sign flag over a masked square", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        ssa.block(0).and_then(SsaBlock::control_terminator).cloned()
    }

    /// **The distance of two addresses of one image is a constant.** Arm's
    /// `deregister_tm_clones` loads `__TMC_END__` twice and branches on the
    /// zero flag of `cmp r3, r0`: neither load is a number the range domain
    /// holds, but the two move together wherever the image lands, so their
    /// difference is zero and the branch is taken. The lift converts each to
    /// an integer of its width first (`ptrtoint.u32 sym(0x250ec)`). The
    /// control: two different addresses differ, and the branch falls through.
    #[test]
    fn the_distance_of_two_addresses_of_one_image_is_a_constant() {
        let decided = |symbols: (u32, u32), converted: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
            let first = local_at(&mut ssa, 0, 0, 0);
            let second = local_at(&mut ssa, 1, 0, 1);
            let first_int = local_at(&mut ssa, 4, 0, 2);
            let second_int = local_at(&mut ssa, 5, 0, 3);
            let difference = local_at(&mut ssa, 2, 0, 4);
            let flags = local_at(&mut ssa, 3, 0, 4);
            let mut b0 = SsaBlock::new(0);
            b0.add_instruction(instr(SsaOp::Const {
                dest: first,
                value: ConstValue::Symbol(symbols.0),
            }));
            b0.add_instruction(instr(SsaOp::Const {
                dest: second,
                value: ConstValue::Symbol(symbols.1),
            }));
            b0.add_instruction(instr(SsaOp::PtrToInt {
                dest: first_int,
                operand: first,
                target: MockType::I32,
            }));
            b0.add_instruction(instr(SsaOp::PtrToInt {
                dest: second_int,
                operand: second,
                target: MockType::I32,
            }));
            let (left, right) = if converted {
                (second_int, first_int)
            } else {
                (second, first)
            };
            b0.add_instruction(instr(SsaOp::Sub {
                dest: difference,
                left,
                right,
                flags: Some(flags),
            }));
            b0.add_instruction(instr(SsaOp::BranchFlags {
                flags,
                condition: FlagCondition::Zero,
                true_target: 1,
                false_target: 2,
            }));
            ssa.add_block(b0);
            for index in 1..=2 {
                let mut block = SsaBlock::new(index);
                block.add_instruction(instr(SsaOp::returns(None)));
                ssa.add_block(block);
            }
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            let method = 0u32;
            ValueRangePropagationPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
                )
                .ok();
            ssa.block(0).and_then(SsaBlock::control_terminator).cloned()
        };
        for converted in [false, true] {
            assert!(
                matches!(
                    decided((0x250ec, 0x250ec), converted),
                    Some(SsaOp::Jump { target: 1 })
                ),
                "one address twice is no distance (converted: {converted})"
            );
            assert!(
                matches!(
                    decided((0x250ec, 0x25100), converted),
                    Some(SsaOp::Jump { target: 2 })
                ),
                "control: two addresses differ (converted: {converted})"
            );
        }
    }

    /// `x & x` is `x`, and `x & y` over non-negative operands cannot exceed
    /// either of them.
    #[test]
    fn and_carries_its_operands_bounds_when_neither_is_constant() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let value = local_at(&mut ssa, 0, 0, 0);
        drop(ssa);
        let analysis: RangeAnalysis<MockTarget> = RangeAnalysis::new(20);
        // Nothing is recorded, so both operands read as the least-informative
        // range and only the structural rule can answer.
        let self_and = analysis.evaluate_instruction(
            &SsaOp::And {
                dest: value,
                left: value,
                right: value,
                flags: None,
            },
            &SsaFunction::<MockTarget>::new(0, 1),
        );
        assert!(
            self_and.is_top(),
            "an unknown value ANDed with itself is still that unknown value"
        );
    }

    /// An unconstrained condition explores **both** arms, so the analysis
    /// reaches what lies past it.
    ///
    /// Waiting on a `Top` condition to sharpen and be re-decided would wait
    /// forever: the lattice ascends and `Top` is its greatest element, so no
    /// later join moves a value already there and nothing re-queues the
    /// terminator. Propagation would stop dead at the first unconstrained
    /// condition, the worklist drain, and the run report convergence with most
    /// of the function never marked executable.
    ///
    /// The probe is a decidable fold *behind* such a branch: `1 == 1` in `b1`
    /// only simplifies if `b1` was reached at all.
    #[test]
    fn an_unconstrained_condition_explores_both_arms() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        // A remainder by a non-constant divisor is *computed* and answers `Top`
        // - unconstrained, rather than merely absent from the map, which is the
        // distinction this test turns on. It is the shape lifted code presents
        // everywhere: a flags read, a load, anything the evaluator cannot bound.
        let num = local_at(&mut ssa, 0, 1, 0);
        let den = local_at(&mut ssa, 1, 1, 1);
        let cond = local_at(&mut ssa, 2, 1, 2);
        let one = local_at(&mut ssa, 3, 2, 0);

        // The branch is deliberately **not** in the entry block: `initialize`
        // seeds the worklist with the entry's CFG successors outright, so an
        // entry-block terminator is never the one that decides them.
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::LoadArg {
            dest: num,
            arg_index: 0,
        }));
        b1.add_instruction(instr(SsaOp::LoadArg {
            dest: den,
            arg_index: 1,
        }));
        b1.add_instruction(instr(SsaOp::Rem {
            dest: cond,
            left: num,
            right: den,
            unsigned: false,
            flags: None,
        }));
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::BranchCmp {
            left: one,
            right: one,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "top condition explores both", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        // `1 == 1` behind the unconstrained branch. It folds only if `b2` was
        // reached at all, so the terminator is the probe - not `changed`, which
        // any other rewrite would also set.
        assert!(
            matches!(
                ssa.block(2).and_then(SsaBlock::control_terminator),
                Some(SsaOp::Jump { target: 3 })
            ),
            "a decidable comparison behind an unconstrained branch must fold; \
             leaving it alone means propagation stalled at the branch, got {:?}",
            ssa.block(2).and_then(SsaBlock::control_terminator)
        );
    }

    /// A branch on a function argument explores both arms instead of stalling.
    ///
    /// The deferral in [`RangeAnalysis::propagate_outgoing_edges`] rests on an
    /// uncomputed operand being *about to be* computed. A caller-supplied value
    /// never is - nothing in the body defines it - so without the seeding in
    /// [`RangeAnalysis::initialize`] the promise is never kept and propagation
    /// stops dead at the first branch on a parameter. `bnez $a0` is one, and it
    /// is how MIPS and RISC-V spell every conditional branch on a parameter, so
    /// this is not a corner.
    #[test]
    fn a_branch_on_an_argument_does_not_stall() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let arg = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::I32,
        );
        let zero = local_at(&mut ssa, 0, 1, 0);
        let one = local_at(&mut ssa, 1, 2, 0);

        // Again not the entry block: `initialize` seeds the entry's successors
        // outright, so a stall there would not show.
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: arg,
            right: zero,
            cmp: CmpKind::Ne,
            unsigned: false,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b2.add_instruction(instr(SsaOp::BranchCmp {
            left: one,
            right: one,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "argument branch does not stall", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        assert!(
            matches!(
                ssa.block(2).and_then(SsaBlock::control_terminator),
                Some(SsaOp::Jump { target: 3 })
            ),
            "the arm behind a branch on an argument must be analyzed; leaving \
             `1 == 1` unfolded means the run stalled on an operand nothing will \
             ever compute, got {:?}",
            ssa.block(2).and_then(SsaBlock::control_terminator)
        );
    }

    /// A block the fixpoint never reached is left alone, however decidable its
    /// operands look.
    ///
    /// `run`'s rewrite loop walks every block, and a range is only a fact about
    /// a block the analysis proved executable. In one it never reached,
    /// a range answers with whatever a dominating block last wrote - here
    /// the two constants of `b0` - because the definitions that would have
    /// overwritten it were never evaluated. Folding on that turns "the analysis
    /// did not get here" into "the analysis proved this branch", which is how a
    /// stalled propagation deleted a live loop body rather than missing an
    /// optimisation.
    #[test]
    fn an_unreached_block_is_not_rewritten() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let zero = local_at(&mut ssa, 0, 0, 0);
        let one = local_at(&mut ssa, 1, 0, 1);

        // `b0` returns, so no edge ever leaves it.
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: one,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b0);

        // No predecessor, and a comparison that `0 != 1` decides outright.
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: zero,
            right: one,
            cmp: CmpKind::Ne,
            unsigned: false,
            true_target: 2,
            false_target: 2,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "unreached block untouched", |ssa| {
            ValueRangePropagationPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                    .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
            ) == Ok(true)
        });
        assert!(
            !changed,
            "a block no executable edge reaches must not be rewritten from \
             ranges the analysis never established there"
        );
    }

    /// **A conditional trap whose comparison never holds never fires.** A
    /// trap if not equal of 0 against 0 (an `fwait` whose pending exceptions
    /// fold to none) is deleted; one of 0 against 1 always fires and stays,
    /// as does one whose operand is unknown.
    #[test]
    fn a_trap_whose_condition_never_holds_is_deleted() {
        let traps_left = |left: Option<i32>, right: i32| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
            let first = match left {
                Some(_) => local_at(&mut ssa, 0, 0, 0),
                None => ssa.create_variable(
                    VariableOrigin::EntryLiveIn,
                    0,
                    DefSite::entry(),
                    MockType::I32,
                ),
            };
            let second = local_at(&mut ssa, 1, 0, usize::from(left.is_some()));
            let mut b0 = SsaBlock::new(0);
            if let Some(left) = left {
                b0.add_instruction(instr(SsaOp::Const {
                    dest: first,
                    value: ConstValue::I32(left),
                }));
            }
            b0.add_instruction(instr(SsaOp::Const {
                dest: second,
                value: ConstValue::I32(right),
            }));
            b0.add_instruction(instr(SsaOp::SystemOp(Box::new(NativeKindedData {
                kind: SystemOpKind::Trap {
                    op: TrapOp::TrapIfNotEqual,
                    vector: Some(16),
                },
                mnemonic: String::new(),
                metadata: None,
                outputs: Vec::new(),
                inputs: vec![first, second],
            }))));
            b0.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(b0);
            ssa.recompute_uses();
            let log: EventLog<MockTarget> = EventLog::new();
            let method = 0u32;
            run_mock_pass_boundary(&mut ssa, "traps", |ssa| {
                ValueRangePropagationPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE))
                        .with_budget(WorkBudget::DEFAULT.with_limit(Budget::RangeRounds, 20)),
                ) == Ok(true)
            });
            ssa.blocks()
                .iter()
                .flat_map(|block| block.instructions())
                .filter(|instruction| matches!(instruction.op(), SsaOp::SystemOp(_)))
                .count()
        };
        assert_eq!(traps_left(Some(0), 0), 0);
        assert_eq!(traps_left(Some(0), 1), 1);
        assert_eq!(traps_left(None, 0), 1);
    }

    /// Overlap detection at the boundaries: adjacent, single-point, and
    /// negative ranges.
    #[test]
    fn ranges_overlap_edge_cases() {
        // Exactly adjacent
        let a = ValueRange::bounded(0, 5);
        let b = ValueRange::bounded(6, 10);
        assert!(
            !ranges_overlap(&a, &b),
            "adjacent ranges should not overlap"
        );

        // Single point overlapping
        let c = ValueRange::bounded(5, 5);
        let d = ValueRange::bounded(5, 10);
        assert!(
            ranges_overlap(&c, &d),
            "single point should overlap if same value"
        );

        // Negative ranges
        let e = ValueRange::bounded(-10, -1);
        let f = ValueRange::bounded(-5, 5);
        assert!(
            ranges_overlap(&e, &f),
            "negative ranges should overlap correctly"
        );

        // Non-overlapping negatives
        let g = ValueRange::bounded(-10, -5);
        let h = ValueRange::bounded(-4, 5);
        assert!(
            !ranges_overlap(&g, &h),
            "non-overlapping negatives should not overlap"
        );
    }

    /// A `Clt` whose left range is entirely above its right folds to `false`.
    #[test]
    fn try_simplify_clt_always_false() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(10, 20)),
            (v1, ValueRange::bounded(0, 5)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: v0,
            right: v1,
            unsigned: false,
            signaling: false,
        };
        assert_eq!(
            try_simplify_comparison(&op, &result, &no_variables()),
            Some((dest, false))
        );
    }

    /// A `Cgt` whose left range is entirely below its right folds to `false`.
    #[test]
    fn try_simplify_cgt_always_false() {
        let v0 = SsaVarId::from_index(0);
        let v1 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let result = make_result(vec![
            (v0, ValueRange::bounded(0, 5)),
            (v1, ValueRange::bounded(10, 20)),
        ]);
        let op: SsaOp<MockTarget> = SsaOp::Cgt {
            dest,
            left: v0,
            right: v1,
            unsigned: false,
            signaling: false,
        };
        assert_eq!(
            try_simplify_comparison(&op, &result, &no_variables()),
            Some((dest, false))
        );
    }

    /// A value resolved on a path is decided at its variable's width and
    /// signedness, not at the width of the `i64` it arrives in.
    ///
    /// `0x8000_0000` in a 32-bit register has its sign bit set, and `-1` in a
    /// signed register is the same 32 bits as `0xFFFF_FFFF` in an unsigned
    /// one. Read as plain `i64`s, the first looks non-negative and the second
    /// pair unequal.
    #[test]
    fn a_resolved_value_is_decided_at_its_operands_width() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 4);
        let input = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::U32,
        );
        let high = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::U32,
        );
        let flags = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let signed =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::entry(), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::And {
            dest: high,
            left: input,
            right: input,
            flags: Some(flags),
        }));
        b0.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let values: FxHashMap<SsaVarId, i64> =
            [(high, 0x8000_0000), (input, 0xFFFF_FFFF), (signed, -1)]
                .into_iter()
                .collect();
        let mut known = |var: SsaVarId| {
            values
                .get(&var)
                .map(|value| (ValueRange::constant(*value), Congruence::exact(*value)))
        };

        let sign_test: SsaOp<MockTarget> = SsaOp::BranchFlags {
            flags,
            condition: FlagCondition::Negative,
            true_target: 1,
            false_target: 2,
        };
        assert_eq!(
            taken_target_known(&sign_test, &ssa, &mut known),
            Some(1),
            "bit 31 of a 32-bit result is its sign"
        );
        let equal: SsaOp<MockTarget> = SsaOp::BranchCmp {
            left: signed,
            right: input,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        };
        assert_eq!(
            taken_target_known(&equal, &ssa, &mut known),
            Some(1),
            "-1 and 0xFFFF_FFFF are the same 32 bits"
        );
    }

    /// **A select that clamps a value to a limit is bounded by the limit.**
    /// `(n <u 8191) ? n : 8191` is how a compiler spells `min(n, 8191)` over
    /// an unknown unsigned `n`: the arm that takes `n` takes it below 8191,
    /// so the select is `[0, 8191]`, one more `[1, 8192]`, and a mask of it
    /// with `0x3ff8` `[0, 8192]`. The control selects on a condition that
    /// compares something else, and the select stays unbounded.
    #[test]
    fn a_select_clamping_a_value_to_a_limit_is_bounded_by_it() {
        let clamped = |compares_the_arm: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(2, 8);
            let n = ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::U32,
            );
            let other = ssa.create_variable(
                VariableOrigin::Argument(1),
                0,
                DefSite::entry(),
                MockType::U32,
            );
            let variable = |ssa: &mut SsaFunction<MockTarget>, idx: u16, at: usize, ty| {
                ssa.create_variable(
                    VariableOrigin::Local(idx),
                    0,
                    DefSite::instruction(0, at),
                    ty,
                )
            };
            let limit = variable(&mut ssa, 0, 0, MockType::U32);
            let below = variable(&mut ssa, 1, 1, MockType::I32);
            let select = variable(&mut ssa, 2, 2, MockType::U32);
            let one = variable(&mut ssa, 3, 3, MockType::U32);
            let more = variable(&mut ssa, 4, 4, MockType::U32);
            let mask = variable(&mut ssa, 5, 5, MockType::U32);
            let masked = variable(&mut ssa, 6, 6, MockType::U32);
            let mut b0 = SsaBlock::new(0);
            for op in [
                SsaOp::Const {
                    dest: limit,
                    value: ConstValue::U32(8191),
                },
                SsaOp::Clt {
                    dest: below,
                    left: if compares_the_arm { n } else { other },
                    right: limit,
                    unsigned: true,
                    signaling: false,
                },
                SsaOp::Select {
                    dest: select,
                    condition: below,
                    true_val: n,
                    false_val: limit,
                },
                SsaOp::Const {
                    dest: one,
                    value: ConstValue::U32(1),
                },
                SsaOp::Add {
                    dest: more,
                    left: select,
                    right: one,
                    flags: None,
                },
                SsaOp::Const {
                    dest: mask,
                    value: ConstValue::U32(0x3ff8),
                },
                SsaOp::And {
                    dest: masked,
                    left: more,
                    right: mask,
                    flags: None,
                },
                SsaOp::returns(Some(masked)),
            ] {
                b0.add_instruction(instr(op));
            }
            ssa.add_block(b0);
            ssa.recompute_uses();
            (
                bounds_of(&ssa, select),
                bounds_of(&ssa, more),
                bounds_of(&ssa, masked),
            )
        };
        assert_eq!(
            clamped(true),
            (Some((0, 8191)), Some((1, 8192)), Some((0, 8192)))
        );
        let (select, _, masked) = clamped(false);
        assert_ne!(select, Some((0, 8191)), "control");
        assert_eq!(masked, Some((0, 0x3ff8)), "control: the mask alone");
    }
}
