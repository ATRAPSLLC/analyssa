//! Jump threading pass - redirects branches through predecessors when the
//! branch condition is known from the incoming path.
//!
//! # Algorithm
//!
//! For each block ending in a conditional transfer:
//!
//! 1. **Evaluate predecessor**: use [`SsaEvaluator`] to compute concrete
//!    values for all variables at the end of the predecessor block.
//! 2. **Resolve phis**: set the evaluator's predecessor context so phi
//!    nodes in the branch block are resolved to the value coming from that
//!    specific predecessor.
//! 3. **Evaluate condition**: if the condition variable resolves to a
//!    concrete constant (via [`SsaEvaluator::get_concrete`] or
//!    [`SsaEvaluator::resolve_with_trace`]), the branch target is known.
//! 4. **Redirect**: retarget the predecessor's arm (or arms) naming the
//!    branch block to the proven target, bypassing the branch. Only those
//!    arms change: a predecessor ending in a `Branch`, `BranchCmp`,
//!    `BranchFlags` or `Switch` keeps its own condition and every other arm,
//!    because the proof holds only for the edge into the branch block.
//!
//! # Scope
//!
//! Every conditional transfer is a candidate: `Branch`, `BranchCmp`,
//! `BranchFlags` and `Switch`, decided through
//! `passes::ranges`'s `taken_target_known` so that this pass and
//! the range fixpoint cannot disagree about where a terminator goes. `Jump` is
//! not threaded - a trampoline block is the block merger's job, not this one.
//!
//! Every kind matters because of control-flow-flattened dispatchers: a
//! dispatcher is a `Switch`, or a chain of `BranchCmp` and `BranchFlags` once a
//! compiler has lowered one, and threading is precisely what undoes it. Each
//! back edge into the dispatcher carries a constant state, so each resolves to
//! one case, and the dispatcher falls away with the loop it invented.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::{
    Result,
    analysis::{
        cfg::SsaCfg,
        congruence::Congruence,
        consts::Machine,
        defuse::DefUseIndex,
        evaluator::SsaEvaluator,
        loops::{LoopForest, detect_loops},
        range::ValueRange,
    },
    events::EventKind,
    graph::{NodeId, RootedGraph, algorithms::compute_dominators},
    ir::{
        function::{
            SsaEditOptions, SsaEditor, SsaFunction, SsaRollbackPolicy, StorageSlots, ValueUpdate,
        },
        ops::SsaOp,
        phi::{PhiNode, PhiOperand},
        value::ConstValue,
        variable::SsaVarId,
    },
    passes::{decide::taken_target_known, guarded::GuardRefinement, ranges::ValueRanges},
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::Target,
};

/// Whether an edge from `source` into `branch` leaves a loop `branch` lies
/// past: one `source` is in and `branch` is not.
///
/// A test there is the one the source writes after its loop, which a `break`
/// out of the loop reaches: `expmeta`'s `if (metaflag == 0)` after the loop
/// that sets `metaflag = 1` on two of its ways out. Threading the break past
/// the test makes the test's arm a join of the break and the test, with a
/// `goto` into it, and [`crate::passes::unthread`] sends the break back
/// through the test, every round. [`JumpThreadingPass`] keeps such a reroute only where the
/// path dispatches, testing its state against several cases on the way, or
/// where every way into the test is rerouted past it too: then the test runs
/// no more, as a flattener's dispatcher whose last state leaves its loop for
/// the one case past it.
fn leaves_past(loops: &LoopForest, source: usize, branch: usize) -> bool {
    let (source, branch) = (NodeId::new(source), NodeId::new(branch));
    loops
        .loops()
        .iter()
        .any(|looped| looped.contains(source) && !looped.contains(branch))
}

/// The heads of the loops an edge from `source` to `target` would enter
/// elsewhere than at their heads: each one `target` is in, `source` is not,
/// and whose head `target` is not - every loop of a nest the edge enters.
///
/// The head is then no longer the loop's one entry unless no edge into it
/// stays, and the loop is irreducible: a loop whose head tests a value its
/// entry paths set would be entered past its head (`which`'s `path_clean_up`
/// tests a flag its two entry paths set to one constant each, and its back
/// edge to a computed one), its body printed twice with a `goto` into it.
/// [`JumpThreadingPass`] keeps such a reroute only where no edge into the head stays, or
/// where the head dispatches: a dispatcher a control-flow flattener built is
/// a loop too, entered at the case its first state names, which threading it
/// is for, and it tests its state against several cases on the way where a
/// loop's own head tests one condition.
fn loops_entered_aside(loops: &LoopForest, source: usize, target: usize) -> Vec<usize> {
    let (source, target) = (NodeId::new(source), NodeId::new(target));
    loops
        .loops()
        .iter()
        .filter(|looped| {
            looped.contains(target) && !looped.contains(source) && looped.header != target
        })
        .map(|looped| looped.header.index())
        .collect()
}

/// Whether one of `tests`, the conditional transfers a reroute bypasses, tests
/// a dispatcher's state: a value of the loop `head` heads that every back edge
/// sets to a constant - a phi at the head whose operands from inside the loop
/// are each a constant, or a phi of constants.
///
/// A flattened dispatcher's state is the next case's number, a constant on
/// each edge back to it (`ob_flattened` sets 1 on both, merged before its back
/// edge); a loop's own test reads what the body computed (`path_clean_up`'s
/// flag is whether the byte just read was a slash).
fn tests_a_state<T: Target>(
    ssa: &SsaFunction<T>,
    loops: &LoopForest,
    head: usize,
    tests: &[&SsaOp<T>],
) -> bool {
    let Some(looped) = loops.loop_for_header(NodeId::new(head)) else {
        return false;
    };
    let constant = |var: SsaVarId| {
        let mut pending = vec![var];
        let mut seen = BTreeSet::new();
        while let Some(var) = pending.pop() {
            if !seen.insert(var) {
                continue;
            }
            match ssa.get_definition(var) {
                Some(SsaOp::Const { .. }) => {}
                _ => match ssa.find_phi_defining(var) {
                    Some((_, phi)) => pending.extend(phi.operands().iter().map(PhiOperand::value)),
                    None => return false,
                },
            }
        }
        true
    };
    let state = |var: SsaVarId| {
        ssa.block(head)
            .and_then(|block| block.find_phi_defining(var))
            .is_some_and(|phi| {
                phi.operands()
                    .iter()
                    .filter(|operand| looped.contains(NodeId::new(operand.predecessor())))
                    .all(|operand| constant(operand.value()))
            })
    };
    tests.iter().any(|op| match op {
        SsaOp::BranchCmp { left, right, .. } => state(*left) || state(*right),
        SsaOp::Switch { value, .. } => state(*value),
        SsaOp::Branch { condition, .. } => state(*condition),
        _ => false,
    })
}

/// One predecessor's reroute past one path.
struct Reroute {
    /// The last block on the bypassed path, whose edge into the target is the
    /// one the new edge is modelled on.
    last: usize,
    /// The block the predecessor is sent to instead.
    target: usize,
    /// For every value the path carried out of it, the blocks that define the
    /// variable it names - what the reconstruction places its phis from.
    repairs: Vec<(SsaVarId, Definitions)>,
    /// The blocks the reroute bypasses, in walk order.
    path: Vec<usize>,
}

/// Test whether a path can be bypassed without losing a value something reads.
///
/// Threading reroutes a predecessor straight to a proven target, so on that
/// edge none of the path's blocks runs. The skip is sound when nothing the
/// rerouted execution goes on to reach needs what they would have produced.
///
/// "Goes on to reach" is [`reached_without_path`]: the blocks the new edge can
/// reach without entering the path again. A use *inside* that region is a read
/// the path no longer answers; a use anywhere else is reached only by going
/// back through the path, which still computes it, so it is not at risk.
///
/// Within that region the two kinds of definition are judged differently:
///
/// * A value the path **computes** may not be read there at all: on the
///   rerouted edge nothing defines it, and a join where one arm computed it
///   and the other did not has no value to give the other arm.
/// * A **phi result** may, on the terms [`escaping_phis`] states: a phi is not
///   work the path does but the name a variable carries on this edge, and the
///   reroute answers for it by reconstructing that value at the target.
///
/// The path's blocks must also be pure: a dispatcher's blocks compute the
/// comparisons that pick a case and nothing else, so skipping them is
/// invisible, while a block that stores, calls or reads volatile memory is
/// doing work the program can tell was not done. So is a block that writes a
/// slot the function reads as storage ([`StorageSlots`]): the write is pure as
/// an operation, but a read through the slot sees it, and no SSA reader says so.
///
/// # Arguments
///
/// * `analysis` - The function and what the pass computed once for it.
/// * `source` - The predecessor being rerouted.
/// * `path` - The blocks the reroute would bypass, in walk order.
/// * `target` - The block the reroute sends that predecessor to.
///
/// # Returns
///
/// The values the reroute must reconstruct at the target, or `None` when the
/// predecessor may not be rerouted past this path at all.
fn is_safe_to_bypass_path<T: Target>(
    analysis: &Analysis<'_, T>,
    source: usize,
    path: &[usize],
    target: usize,
) -> Option<Vec<(SsaVarId, Definitions)>> {
    let (ssa, cfg, defuse) = (analysis.ssa, analysis.cfg, analysis.defuse);
    let on_path: BTreeSet<usize> = path.iter().copied().collect();
    if on_path.contains(&target) {
        return None;
    }
    let past = reached_without_path(cfg, target, &on_path);
    let mut computed = BTreeSet::new();

    for &block_idx in path {
        let block = ssa.block(block_idx)?;
        let terminator_idx = block.instructions().len().checked_sub(1)?;
        if !block.control_terminator().is_some_and(is_path_link) {
            return None;
        }
        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            if instr_idx == terminator_idx {
                continue;
            }
            if !instr.op().is_pure() || analysis.slots.written_by(ssa, instr.op()) {
                return None;
            }
            computed.extend(instr.defs());
        }
    }

    let repairs = escaping_phis(ssa, source, path, defuse)?;
    if computed.is_empty() {
        return Some(repairs);
    }

    // Querying the prebuilt def-use index touches only each def's own use sites
    // rather than rescanning the function per candidate path.
    for local in &computed {
        if let Some(uses) = defuse.uses_of(*local) {
            for use_site in uses {
                if past.contains(&use_site.block()) {
                    return None;
                }
            }
        }
    }

    Some(repairs)
}

/// The blocks a rerouted edge reaches without entering the bypassed path.
///
/// This is the region the reroute changes: control arrives at `target` without
/// the path having run, and every block it can go on to reach from there - as
/// long as it does not re-enter the path - sees that absence. A block outside
/// the region is reachable only back through the path, which still computes
/// what it computes, so a read there is answered as it always was.
///
/// Walking the region is what lets the safety test speak about *reads the
/// reroute breaks* rather than *reads outside the path*, which are not the same
/// set: a lowered dispatcher's chain defines a value in its head and reads it in
/// every link, and every one of those links sits behind the head.
///
/// With an empty `on_path` this is plain reachability from `target`, which is
/// how the pass asks which predecessors are edges the program can take.
///
/// # Arguments
///
/// * `cfg` - The function's CFG.
/// * `target` - The block the reroute sends the predecessor to.
/// * `on_path` - The bypassed blocks, which the walk stops at.
fn reached_without_path<T: Target>(
    cfg: &SsaCfg<T>,
    target: usize,
    on_path: &BTreeSet<usize>,
) -> BTreeSet<usize> {
    let mut reached = BTreeSet::new();
    let mut pending = vec![target];
    while let Some(block) = pending.pop() {
        if on_path.contains(&block) || !reached.insert(block) {
            continue;
        }
        pending.extend(cfg.block_successors(block).iter().copied());
    }
    reached
}

/// The function and everything computed once for a run of this pass.
///
/// The walk consults all of it per candidate path, and computing any of it per
/// candidate would make the pass quadratic in the dispatcher it exists to undo.
struct Analysis<'a, T: Target> {
    /// The function being threaded.
    ssa: &'a SsaFunction<T>,
    /// Its control-flow graph.
    cfg: &'a SsaCfg<'a, T>,
    /// Def-use over the unmutated function, for the safety test.
    defuse: &'a DefUseIndex<T>,
    /// Whole-function ranges, when the analysis converged.
    ranges: Option<Arc<ValueRanges>>,
    /// The machine the function computes on, for the evaluator.
    machine: Machine,
    /// The slots the function reads as storage, whose writes a bypassed path
    /// may not skip.
    slots: StorageSlots,
}

/// What the walk knows about a value's *range* on the path it is following.
///
/// The evaluator answers with values, and a dispatcher whose cases write the
/// state as a literal needs nothing more. A compiler that computes the state
/// instead - `state = (i < n) ? 2 : 3`, which is what `-O1` and above emit -
/// leaves no edge carrying a value, and the decision is then about what the
/// edge *proves*: a state in `[2, 3]` settles `state == 0` and `state != 1`
/// without settling what the state is.
///
/// Three sources, in order of precision:
///
/// 1. the edge being threaded, whose predecessor branched on a relation, so
///    taking it proves that relation (`state != 2` over `[2, 3]` is `state ==
///    3`);
/// 2. a phi the walk resolved, which on this edge *is* its operand, so the
///    operand's range is the phi's;
/// 3. the whole-function range analysis, which is sound everywhere.
struct PathFacts<'a> {
    /// The whole-function ranges, when the analysis converged.
    ranges: Option<&'a ValueRanges>,
    /// A value the walk proved is another value on this path.
    aliases: BTreeMap<SsaVarId, SsaVarId>,
    /// A value the entry edge narrowed by the relation its predecessor took.
    narrowed: BTreeMap<SsaVarId, ValueRange>,
}

impl PathFacts<'_> {
    /// What is proved about `var` on this path, or `None` where nothing is.
    fn known(&self, var: SsaVarId) -> Option<(ValueRange, Congruence)> {
        let mut var = var;
        // The alias chain is a phi resolved to its operand, and a phi's
        // operand can be another phi; bounded by the map so a cycle stops.
        for _ in 0..=self.aliases.len() {
            match self.aliases.get(&var) {
                Some(next) => var = *next,
                None => break,
            }
        }
        let range = self
            .narrowed
            .get(&var)
            .cloned()
            .or_else(|| self.ranges.and_then(|ranges| ranges.get(var).cloned()))?;
        let congruence = self
            .ranges
            .and_then(|ranges| ranges.congruence(var).copied())
            .unwrap_or_default();
        Some((range, congruence))
    }

    /// Records that taking the edge out of `pred_block` toward `first` proves
    /// its terminator's relation.
    fn narrow_on_entry<T: Target>(
        &mut self,
        ssa: &SsaFunction<T>,
        pred_block: usize,
        first: usize,
    ) {
        let Some(ranges) = self.ranges else {
            return;
        };
        let Some(terminator) = ssa
            .block(pred_block)
            .and_then(|block| block.control_terminator())
        else {
            return;
        };
        let SsaOp::BranchCmp {
            left,
            right,
            cmp,
            unsigned,
            true_target,
            false_target,
        } = terminator
        else {
            return;
        };
        // Only when the two arms differ: a branch whose targets agree proves
        // nothing about which one was taken.
        let taken = match (*true_target == first, *false_target == first) {
            (true, false) => true,
            (false, true) => false,
            _ => return,
        };
        for var in [*left, *right] {
            if let Some(refined) =
                ranges.refine_from_relation(ssa, (*left, *right, *cmp, *unsigned), var, taken)
            {
                let held = match ranges.get(var) {
                    Some(known) => known.meet(&refined),
                    None => refined,
                };
                self.narrowed.insert(var, held);
            }
        }
    }
}

/// For one value the path carries out of it, the blocks that define the
/// variable it names.
type Definitions = Vec<(usize, SsaVarId)>;

/// What a walk of one threadable path found: the blocks it bypasses, the block
/// it ends at, and the values the reroute has to reconstruct there.
type ThreadedPath = (Vec<usize>, usize, Vec<(SsaVarId, Definitions)>);

/// The values a bypassed path carries out of it, and where each is really
/// defined.
///
/// A phi on the path is not work the path does; it is the name a variable
/// carries on this edge. Rerouting past it costs nothing when nothing beyond
/// the path reads the result - and when something does, the reroute has to say
/// what the variable is, without the path's phis to say it.
///
/// **The phis on the path are names, not definitions.** Expanding one through
/// its operands, and through any operand that is itself a path phi, lands on
/// the blocks that actually assign the variable: a dispatcher's cases, and the
/// entry. That set is what a [`ValueUpdate`] states, and it is what the
/// reconstruction places the new phis from. An operand that names the phi being
/// expanded is the loop carrying its own value and contributes no definition -
/// which is exactly why an edge-by-edge answer cannot state this: the value on
/// the back edge *is* the phi that has not been placed yet.
///
/// A path phi whose expansion reaches a value the path **computes** is refused:
/// that value is not produced on the rerouted edge, so no definition can be
/// stated for it.
///
/// # Arguments
///
/// * `ssa` - The SSA function being threaded.
/// * `source` - The predecessor being rerouted.
/// * `path` - The blocks the reroute bypasses, in walk order.
/// * `defuse` - Prebuilt def-use index over the unmutated function.
///
/// # Returns
///
/// One entry per phi result read off the path, with the blocks that define the
/// variable it names, or `None` when one of them cannot be stated.
fn escaping_phis<T: Target>(
    ssa: &SsaFunction<T>,
    source: usize,
    path: &[usize],
    defuse: &DefUseIndex<T>,
) -> Option<Vec<(SsaVarId, Definitions)>> {
    let on_path: BTreeSet<usize> = path.iter().copied().collect();
    // Every phi in the function, not only the ones on the path. A phi is a
    // name wherever it stands, and expanding one that sits *off* the path is
    // what keeps the answer true after the reroute: a dispatcher's merge block
    // is off the path of the reroute that enters it, and it stops existing once
    // the last case stops routing through it. Naming its result as a definition
    // would pin the variable to a value that is about to go.
    let mut phis: BTreeMap<SsaVarId, Vec<(usize, SsaVarId)>> = BTreeMap::new();
    for (_, block) in ssa.iter_blocks() {
        for phi in block.phi_nodes() {
            phis.insert(
                phi.result(),
                phi.operands()
                    .iter()
                    .map(|operand| (operand.predecessor(), operand.value()))
                    .collect(),
            );
        }
    }
    let mut path_phis: BTreeSet<SsaVarId> = BTreeSet::new();
    let mut computed: BTreeSet<SsaVarId> = BTreeSet::new();
    for &block_idx in path {
        let block = ssa.block(block_idx)?;
        path_phis.extend(block.phi_nodes().iter().map(PhiNode::result));
        for instr in block.instructions() {
            computed.extend(instr.defs());
        }
    }

    let mut updates = Vec::new();
    for result in &path_phis {
        // Read anywhere off the path, not only in the region *this* reroute
        // reaches. A session threads several edges, and the blocks one reroute
        // still arrives at through the path may be cut off by the next: a
        // dispatcher's cases keep its phis alive one at a time until the last
        // edge goes. Naming the value for every reroute that could orphan it
        // costs a phi the trivial-phi cleanup takes out again; missing one
        // costs the value.
        let escapes = defuse.uses_of(*result).is_some_and(|uses| {
            uses.iter()
                .any(|use_site| !on_path.contains(&use_site.block()))
        });
        if !escapes {
            continue;
        }
        let mut definitions: BTreeMap<usize, SsaVarId> = BTreeMap::new();
        let mut pending = vec![*result];
        let mut expanded: BTreeSet<SsaVarId> = BTreeSet::new();
        while let Some(phi) = pending.pop() {
            if !expanded.insert(phi) {
                continue;
            }
            for (predecessor, operand) in phis.get(&phi)? {
                if phis.contains_key(operand) {
                    pending.push(*operand);
                    continue;
                }
                if computed.contains(operand) {
                    return None;
                }
                // **Where the value stands, not where it was named.** An
                // operand arriving from a block on the bypassed path is what
                // the rerouted edge carries, and that block will not run on
                // it - nor, once every edge into it is rerouted, at all. The
                // value is the reroute's own: nothing on the path computes it,
                // so it is already available where the edge now leaves. Kept
                // at the bypassed block, the definition vanished with it, and
                // the reconstruction reached past it to an earlier one - a
                // dispatcher's stream local read as the `null` it held before
                // the case that assigned it.
                let at = if on_path.contains(predecessor) {
                    source
                } else {
                    *predecessor
                };
                match definitions.get(&at) {
                    Some(existing) if existing != operand => return None,
                    Some(_) => {}
                    None => {
                        definitions.insert(at, *operand);
                    }
                }
            }
        }
        if definitions.is_empty() {
            return None;
        }
        updates.push((*result, definitions.into_iter().collect()));
    }
    Some(updates)
}

/// How far a threaded path is followed before giving up.
///
/// A lowered dispatcher is a chain of comparisons, one per case, so the bound
/// is the number of cases a flattener plausibly emits rather than a limit on
/// anything the walk costs - each hop is one evaluation.
const MAX_THREAD_PATH: usize = 32;

/// Walks the chain of conditional transfers a constant predecessor decides, and
/// returns the blocks it passes through with the block it ends at.
///
/// Threading one block at a time cannot unflatten a dispatcher, because a
/// lowered one is not one block: on `micro/elf_x86_64_O0` it is six - the head
/// holds the conversions the chain reads, and each link holds one comparison.
/// Every link resolves under the same constant state, so the whole chain is one
/// decision, and following it is what turns "jump to the dispatcher with
/// `state = 2`" back into "jump to the code `case 2` names".
///
/// Stops at the first block the walk cannot continue through - the case body,
/// which is where the path is going. A block that carries phi nodes is *not* a
/// stop: the walk knows which edge it arrives on, because it is the block the
/// walk just left, so the phis resolve to the values this path carries. That is
/// what lets a path cross the merge block a flattener's back edges come
/// together in, which is the only place the state is a constant.
///
/// An unconditional `Jump` is followed like any other link, but the prefix that
/// is returned always ends at a *conditional* transfer: a walk that only ever
/// jumps proves nothing about control flow and its blocks are the merger's to
/// fold, not this pass's to bypass.
///
/// # Arguments
///
/// * `analysis` - The function and everything computed once for the pass.
/// * `pred_block` - The predecessor whose constants decide the chain.
/// * `first` - The first block of the chain, a successor of `pred_block`.
/// * `terminator` - `first`'s conditional transfer.
///
/// # Returns
///
/// The blocks walked through, the block the chain arrives at, and the values
/// the reroute must reconstruct there - or `None` when the first link does not
/// resolve, or no prefix of the walk is safe to bypass.
fn thread_path<T: Target>(
    analysis: &Analysis<'_, T>,
    pred_block: usize,
    first: usize,
    terminator: &SsaOp<T>,
) -> Option<ThreadedPath> {
    let Analysis {
        ssa,
        ranges,
        machine,
        ..
    } = analysis;
    let (ssa, machine) = (*ssa, *machine);
    let ranges = ranges.as_deref();
    // One evaluator for the whole walk. Each link reads what the links before
    // it computed - the dispatcher's head converts the state and every
    // comparison reads the conversion - so deciding a link needs the earlier
    // blocks' values in scope. Building a fresh evaluator per hop, as deciding
    // a single block does, leaves them out and the walk stops at the first
    // link.
    let mut eval = SsaEvaluator::new(ssa, machine);
    eval.evaluate_block(pred_block);

    let mut facts = PathFacts {
        ranges,
        aliases: BTreeMap::new(),
        narrowed: BTreeMap::new(),
    };
    facts.narrow_on_entry(ssa, pred_block, first);
    // A value the edge narrows to one value is that value on the path, and so
    // is everything computed from it: the dispatcher's state `((i < n) ^ 1) +
    // 2`, which the predecessor's `state != 2` leaves 3, reaches the head's
    // comparisons through the conversion RV64 reads it with. Before the
    // phis, which carry it into the first block.
    for (var, range) in &facts.narrowed {
        if let Some(value) = range.as_constant()
            && let Some(constant) = ssa.variable(*var).and_then(|variable| {
                ConstValue::from_bits_of_type(value.cast_unsigned(), variable.var_type())
            })
        {
            eval.set_concrete(*var, constant);
        }
    }
    eval.set_predecessor(Some(pred_block));
    eval.evaluate_phis(first);
    alias_phis(ssa, first, pred_block, &mut facts);

    let mut path = vec![first];
    let mut targets = Vec::new();
    // Whether each block on the path ends in a conditional transfer, so the
    // prefix search below can refuse to stop on a block that only jumps.
    let mut decides = vec![is_conditional_transfer(terminator)];
    let mut block_idx = first;
    let mut link = terminator.clone();
    for _ in 0..MAX_THREAD_PATH {
        eval.evaluate_block(block_idx);
        let Some(target) = next_on_path(&link, ssa, &mut eval, &facts) else {
            break;
        };
        targets.push(target);
        if path.contains(&target) {
            break;
        }
        let Some(block) = ssa.block(target) else {
            break;
        };
        let Some(next) = block.control_terminator().cloned() else {
            break;
        };
        if !is_path_link(&next) {
            break;
        }
        // The arriving edge is the block the walk just left, so the target's
        // phis read the values this path carries rather than a merge of every
        // edge's.
        eval.set_predecessor(Some(block_idx));
        eval.evaluate_phis(target);
        alias_phis(ssa, target, block_idx, &mut facts);
        path.push(target);
        decides.push(is_conditional_transfer(&next));
        block_idx = target;
        link = next;
    }
    if targets.is_empty() {
        return None;
    }

    // The **longest safe prefix**, which is not the same as extending while
    // safe. The dispatcher's state is read by every link in the chain, so the
    // head is never bypassable on its own - it becomes bypassable exactly when
    // the path grows to include the blocks that read what it defines. Judging
    // each extension in isolation would stop at the first block and thread
    // nothing.
    for length in (1..=targets.len().min(path.len())).rev() {
        let Some(prefix) = path.get(..length) else {
            continue;
        };
        let Some(&target) = targets.get(length.saturating_sub(1)) else {
            continue;
        };
        // The prefix has to end on a block that *decided* something. Ending on
        // a jump would reroute a predecessor around a block for no reason this
        // pass proved, which is the block merger's judgement to make.
        if !decides
            .get(length.saturating_sub(1))
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        if let Some(repairs) = is_safe_to_bypass_path(analysis, pred_block, prefix, target) {
            return Some((prefix.to_vec(), target, repairs));
        }
    }
    None
}

/// Records each phi of `block` as the value its operand on the arriving edge
/// carries.
///
/// On this path the phi *is* that operand, so whatever is proved about the
/// operand is proved about the phi - which is the fact a join of two
/// dispatcher states otherwise hides behind their union.
///
/// # Arguments
///
/// * `ssa` - The SSA function being threaded.
/// * `block` - The block whose phis are being resolved.
/// * `arriving` - The block the walk entered `block` from.
/// * `facts` - The path's facts, extended in place.
fn alias_phis<T: Target>(
    ssa: &SsaFunction<T>,
    block: usize,
    arriving: usize,
    facts: &mut PathFacts<'_>,
) {
    let Some(block) = ssa.block(block) else {
        return;
    };
    for phi in block.phi_nodes() {
        if let Some(operand) = phi.operand_from(arriving) {
            facts.aliases.insert(phi.result(), operand.value());
        }
    }
}

/// The block one link of a threaded path leads to, under this walk's values.
///
/// An unconditional `Jump` names its successor outright; a conditional transfer
/// is asked through `passes::ranges`'s `taken_target_known`, so the threader and the
/// range fixpoint cannot disagree about where a terminator goes.
///
/// # Arguments
///
/// * `link` - The terminator to follow.
/// * `ssa` - The SSA function being threaded.
/// * `eval` - The walk's evaluator, carrying every value the path proved.
/// * `facts` - What the path proves about a value the evaluator cannot name.
fn next_on_path<T: Target>(
    link: &SsaOp<T>,
    ssa: &SsaFunction<T>,
    eval: &mut SsaEvaluator<'_, T>,
    facts: &PathFacts<'_>,
) -> Option<usize> {
    if let SsaOp::Jump { target } = link {
        return Some(*target);
    }
    taken_target_known(link, ssa, &mut |var| {
        // The evaluator first: a value it holds is exact, and exact beats
        // proved. What it cannot answer, the path's facts may - see
        // [`PathFacts`].
        eval.get_concrete(var)
            .and_then(ConstValue::as_i64)
            .or_else(|| eval.resolve_with_trace(var, 10).and_then(|e| e.as_i64()))
            .map(|value| (ValueRange::constant(value), Congruence::exact(value)))
            .or_else(|| facts.known(var))
    })
}

/// Whether `op` is a link a threaded path may follow: a conditional transfer,
/// or the unconditional `Jump` of a block the path passes through.
///
/// A dispatcher's back edges meet in a merge block before they reach its head,
/// so a path that may not cross a plain jump can never start where the state is
/// still a constant. Crossing one is not block merging, because the prefix
/// [`thread_path`] returns always ends at a conditional transfer.
///
/// # Arguments
///
/// * `op` - The terminator to classify.
fn is_path_link<T: Target>(op: &SsaOp<T>) -> bool {
    matches!(op, SsaOp::Jump { .. }) || is_conditional_transfer(op)
}

/// Whether `op` is a conditional transfer: a two-way branch of any kind, or
/// a `Switch`.
///
/// The gate on both the collection loop and the bypass check: a terminator
/// with no set of alternatives is not a threading candidate.
///
/// # Arguments
///
/// * `op` - The terminator to classify.
fn is_conditional_transfer<T: Target>(op: &SsaOp<T>) -> bool {
    matches!(op, SsaOp::Switch { .. }) || op.two_way_targets().is_some()
}

/// Rewrite the predecessor's terminator so each arm naming a bypassed branch
/// block names that block's proven target instead.
///
/// Applies the threading opportunities [`thread_path`] found for one
/// predecessor. `reroutes` maps the last block on each bypassed path to its proven
/// target, and every arm of the *original* terminator is mapped through it at
/// once, whatever the terminator kind (`Jump`, `Leave`, `Branch`,
/// `BranchCmp`, `BranchFlags`, `Switch`). The predecessor keeps its own
/// condition and every arm that does not name a bypassed block: the proof
/// covers only the edge into the branch block, so the other arms lead where
/// they always did.
///
/// Mapping the original arms in one step, rather than redirecting one block
/// at a time, keeps one rewrite from feeding the next: when one arm's proven
/// target is another bypassed block, that arm must stop there, not be
/// carried on by the second redirect, whose proof was made for the other
/// edge's values.
///
/// Each new target's phis gain an operand for the predecessor, carrying the
/// value they read on the bypassed block's edge. That value is the right one:
/// no block on the bypassed path defines anything live past the path (see
/// [`is_safe_to_bypass_path`]), so what reached the target through the path
/// is what reaches the target from the predecessor directly. The operand is
/// written here rather than left to the boundary rebuild because the rebuild
/// groups variables by phi connectivity: once the bypassed block is
/// unreachable its operand is pruned, and a value connected to the phi only
/// through that operand would lose its group and be replaced by the slot's
/// entry value.
///
/// A predecessor can already have an edge into a new target, directly or
/// through a second reroute; one edge carries one value per phi, so a reroute
/// whose value disagrees with the one already on that edge is skipped.
///
/// # Arguments
///
/// * `editor` - Active edit session on the function.
/// * `pred_block` - Predecessor whose terminator is rewritten.
/// * `reroutes` - Bypassed branch block to proven target, for this predecessor.
/// * `carried_edges` - Collects, per value, the edges this rewrite adds and
///   what each carries, for the one reconstruction the session ends with.
/// * `env` - The pass environment: where events are recorded, naming the method.
///
/// # Returns
///
/// `true` if the terminator was rewritten.
fn apply_threading<T>(
    editor: &mut SsaEditor<T>,
    pred_block: usize,
    reroutes: &BTreeMap<usize, Reroute>,
    carried_edges: &mut BTreeMap<SsaVarId, BTreeMap<usize, SsaVarId>>,
    env: &PassEnv<'_, T>,
) -> bool
where
    T: Target,
{
    let Some(original) = editor
        .function()
        .block(pred_block)
        .and_then(|block| block.control_terminator())
        .cloned()
    else {
        return false;
    };

    let mut phi_operands =
        accepted_phi_operands(editor.function(), pred_block, &original, reroutes);
    phi_operands.retain(|(branch_block, _, new_target, _)| {
        editor
            .function()
            .retarget_respects_regions(pred_block, *branch_block, *new_target)
    });
    let accepted: BTreeMap<usize, usize> = phi_operands
        .iter()
        .map(|(branch_block, _last, new_target, _)| (*branch_block, *new_target))
        .collect();
    if accepted.is_empty() {
        return false;
    }

    let reroute = |target: &mut usize| {
        if let Some(new_target) = accepted.get(target) {
            *target = *new_target;
        }
    };
    let mut rewritten = original.clone();
    match &mut rewritten {
        SsaOp::Jump { target } | SsaOp::Leave { target } => reroute(target),
        SsaOp::Branch {
            true_target,
            false_target,
            ..
        }
        | SsaOp::BranchCmp {
            true_target,
            false_target,
            ..
        }
        | SsaOp::BranchFlags {
            true_target,
            false_target,
            ..
        } => {
            reroute(true_target);
            reroute(false_target);
        }
        SsaOp::Switch {
            targets, default, ..
        } => {
            targets.iter_mut().for_each(reroute);
            reroute(default);
        }
        _ => return false,
    }
    if rewritten == original {
        return false;
    }
    if editor.replace_terminator_op(pred_block, rewritten).is_err() {
        return false;
    }
    // A rerouted edge keeps what it passed, which runs on the way out of the
    // predecessor.
    editor.retarget_passed(pred_block, |edge| {
        Some(accepted.get(&edge).copied().unwrap_or(edge))
    });
    // The edges this rewrite adds, and what each carries for the values the
    // bypassed path was producing. Reported rather than acted on: the session's
    // single reconstruction at the end sees every new edge at once.
    for (branch_block, _, new_target, _) in &phi_operands {
        for (value, definitions) in reroutes
            .get(branch_block)
            .into_iter()
            .flat_map(|reroute| reroute.repairs.iter())
        {
            carried_edges
                .entry(*value)
                .or_default()
                .extend(definitions.iter().copied());
        }
        // The bypassed blocks still run on the rerouted edge; none of them
        // computes anything it needs.
        let mut bypassed = vec![*branch_block];
        bypassed.extend(
            reroutes
                .get(branch_block)
                .into_iter()
                .flat_map(|reroute| reroute.path.iter()),
        );
        editor.record_passed_through(pred_block, *new_target, &bypassed);
    }

    for (branch_block, last_on_path, new_target, needs_operand) in phi_operands {
        // The phi's existing operand is the one from the path's last block, so
        // that is the edge the new predecessor is modelled on.
        if needs_operand
            && editor
                .expand_phi_predecessor(new_target, last_on_path, &[last_on_path, pred_block])
                .is_err()
        {
            return false;
        }
        env.record(EventKind::ControlFlowRestructured)
            .location(pred_block)
            .message(format!(
                "threaded: B{pred_block}'s edge into B{branch_block} now goes to B{new_target}"
            ));
    }
    true
}

/// The reroutes of `pred_block` whose phi operands can be stated, as
/// `(branch_block, new_target, needs_operand)`.
///
/// A reroute is kept when every phi in its new target either has no operand
/// for `pred_block` yet (`needs_operand`: the value on the bypassed block's
/// edge is added for it) or already carries that same value for it. One that
/// would give an edge two values is dropped, and so is one whose arm the
/// original terminator does not have. Two reroutes into one target are
/// checked against each other the same way, since they become one edge.
///
/// # Arguments
///
/// * `ssa` - The function, before this predecessor's rewrite.
/// * `pred_block` - The predecessor being rewritten.
/// * `original` - Its terminator before the rewrite.
/// * `reroutes` - Bypassed branch block to proven target.
fn accepted_phi_operands<T: Target>(
    ssa: &SsaFunction<T>,
    pred_block: usize,
    original: &SsaOp<T>,
    reroutes: &BTreeMap<usize, Reroute>,
) -> Vec<(usize, usize, usize, bool)> {
    // Per (target, phi index), the value this predecessor's edge will carry.
    let mut planned: BTreeMap<(usize, usize), SsaVarId> = BTreeMap::new();
    let mut accepted = Vec::new();
    for (&branch_block, reroute) in reroutes {
        let (last_on_path, new_target) = (reroute.last, reroute.target);
        if !original.has_successor(branch_block) {
            continue;
        }
        let Some(target) = ssa.block(new_target) else {
            continue;
        };
        let mut needs_operand = false;
        let mut consistent = true;
        let mut values = Vec::new();
        for (phi_idx, phi) in target.phi_nodes().iter().enumerate() {
            // The value arrives from the *last* block of the bypassed path,
            // which is the edge the target's phi actually records - the first
            // block is only where the path starts.
            let Some(through) = phi.operand_from(last_on_path).map(PhiOperand::value) else {
                consistent = false;
                break;
            };
            let existing = planned
                .get(&(new_target, phi_idx))
                .copied()
                .or_else(|| phi.operand_from(pred_block).map(PhiOperand::value));
            match existing {
                Some(value) if value != through => {
                    consistent = false;
                    break;
                }
                Some(_) => {}
                None => needs_operand = true,
            }
            values.push((phi_idx, through));
        }
        if !consistent {
            continue;
        }
        for (phi_idx, value) in values {
            planned.insert((new_target, phi_idx), value);
        }
        accepted.push((branch_block, last_on_path, new_target, needs_operand));
    }
    accepted
}

/// Pass that threads jumps through predictable branch conditions.
///
/// Evaluates branch
/// conditions from the incoming path and redirects predecessors directly
/// to the proven target. The evaluator computes as the context's machine
/// does ([`AnalysisContext::machine`](crate::analysis::AnalysisContext::machine)).
///
/// # Modification scope
///
/// [`ModificationScope::CfgModifying`] - changes terminator operations
/// and predecessor relationships.
#[derive(Debug, Default, Clone, Copy)]
pub struct JumpThreadingPass;

impl JumpThreadingPass {
    /// Creates a new jump-threading pass.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<T> FunctionPass<T> for JumpThreadingPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "jump-threading"
    }

    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::CfgModifying
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Thread jumps through empty or predictable blocks"
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Run jump threading on `ssa`.
    ///
    /// For each predecessor of each branch block, evaluates the path from the
    /// predecessor through the branch using [`SsaEvaluator`] and redirects
    /// the predecessor's terminator when the condition is provably constant.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any predecessor was rerouted.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let machine = env.context().machine();
        let facts = env.ranges();
        if ssa.is_empty() {
            return Ok(false);
        }

        // The whole-function ranges, read once for the pass. A dispatcher whose
        // state is *computed* rather than written as a literal leaves no edge
        // carrying a value, and what the edge proves is the only thing that decides
        // it (see [`PathFacts`]). An unconverged analysis yields none, and the pass
        // falls back to deciding on values alone.
        let ranges = facts.get(ssa).ok();

        let cfg = SsaCfg::from_ssa(ssa);
        // One def-use index for the whole collection phase (ssa is not mutated until
        // the edit below), so the per-branch safety check is O(local defs) instead
        // of an O(N) full-function rescan.
        let defuse = DefUseIndex::build(ssa);
        let analysis = Analysis {
            ssa,
            cfg: &cfg,
            defuse: &defuse,
            ranges,
            machine,
            slots: ssa.storage_slots(),
        };

        // Collect threading opportunities first to avoid borrow conflicts: per
        // predecessor, which branch block each of its arms is rerouted past and
        // where to. Keyed by predecessor so that one predecessor with arms into two
        // threadable blocks gets one rewrite covering both.
        let mut threadings: BTreeMap<usize, BTreeMap<usize, Reroute>> = BTreeMap::new();
        // The reroutes past a test that asks one thing, each with the block whose
        // every way in must be rerouted too for it to stay: `(predecessor, branch
        // block, head)`. One that enters a loop past its head names the head; one
        // that leaves a loop past the test after it names that test.
        let mut entering: Vec<(usize, usize, usize)> = Vec::new();

        // An edge the program cannot take is not an edge worth rerouting. Walking
        // one costs an evaluation and, worse, a reroute of it reports the function
        // changed - so the pass-group transaction rebuilds, verifies and re-runs
        // the group for a rewrite of code that never executes.
        let reachable = reached_without_path(&cfg, 0, &BTreeSet::new());
        // The loops as they stand, which no reroute may enter anywhere but at
        // their heads.
        let loops = detect_loops(&cfg, &compute_dominators(&cfg, cfg.entry()));

        for (block_idx, block) in ssa.iter_blocks() {
            // Every conditional transfer, not only `Branch`. A dispatcher that a
            // control-flow flattener emits is a `Switch`, or a chain of `BranchCmp`
            // and `BranchFlags` once a compiler has lowered one, and threading is
            // exactly what undoes it: each back edge carries a constant state, so
            // each resolves to one case.
            //
            // A path may also *start* at an unconditional `Jump`, because a lowered
            // dispatcher's back edges come together in a merge block before they
            // reach the dispatcher head. Starting there is not block merging: the
            // accepted prefix must still end at a conditional transfer, so a chain
            // of plain jumps threads nothing and stays the block merger's work.
            let Some(terminator) = block.control_terminator().cloned() else {
                continue;
            };
            if !is_path_link(&terminator) {
                continue;
            }

            for pred_idx in cfg.block_predecessors(block_idx) {
                if !reachable.contains(pred_idx) {
                    continue;
                }
                // `block_predecessors` lists a predecessor once per edge; one
                // decision covers all of its arms into this block.
                let reroutes = threadings.entry(*pred_idx).or_default();
                if reroutes.contains_key(&block_idx) {
                    continue;
                }
                // The path is walked before it is judged: how far this
                // predecessor's constants carry decides which blocks are being
                // skipped, and the safety question is about all of them together.
                let Some((path, target, repairs)) =
                    thread_path(&analysis, *pred_idx, block_idx, &terminator)
                else {
                    continue;
                };
                if target == block_idx || path.contains(&target) {
                    continue;
                }
                // A dispatcher selects among states: the path the reroute
                // bypasses tests the state more than once, or switches on it. A
                // loop's own test decides one condition.
                let tests = path
                    .iter()
                    .filter_map(|block| ssa.block(*block)?.control_terminator())
                    .filter(|op| is_conditional_transfer(op))
                    .collect::<Vec<_>>();
                let chained =
                    tests.len() > 1 || tests.iter().any(|op| matches!(op, SsaOp::Switch { .. }));
                if !chained && leaves_past(&loops, *pred_idx, block_idx) {
                    entering.push((*pred_idx, block_idx, block_idx));
                }
                if !chained {
                    for head in loops_entered_aside(&loops, *pred_idx, target) {
                        if !tests_a_state(ssa, &loops, head, &tests) {
                            entering.push((*pred_idx, block_idx, head));
                        }
                    }
                }
                let last = path.last().copied().unwrap_or(block_idx);
                reroutes.insert(
                    block_idx,
                    Reroute {
                        last,
                        target,
                        repairs,
                        path,
                    },
                );
            }
        }

        // A reroute into a loop past a head that tests one condition leaves the
        // head entered from elsewhere unless every edge into the head goes
        // elsewhere too: then the head runs no more and its loop is gone. So with
        // the test after a loop that a way out skips ([`leaves_past`]).
        loop {
            let refused: Vec<(usize, usize)> = entering
                .iter()
                .filter(|(predecessor, branch, _)| {
                    threadings
                        .get(predecessor)
                        .is_some_and(|reroutes| reroutes.contains_key(branch))
                })
                .filter(|(_, _, head)| {
                    cfg.block_predecessors(*head).iter().any(|into| {
                        reachable.contains(into)
                            && !threadings
                                .get(into)
                                .is_some_and(|reroutes| reroutes.contains_key(head))
                    })
                })
                .map(|(predecessor, branch, _)| (*predecessor, *branch))
                .collect();
            if refused.is_empty() {
                break;
            }
            for (predecessor, branch) in refused {
                if let Some(reroutes) = threadings.get_mut(&predecessor) {
                    reroutes.remove(&branch);
                }
            }
        }

        if threadings.values().all(BTreeMap::is_empty) {
            return Ok(false);
        }

        let mut changed = false;
        // Verified, and rolled back when the verification fails. Threading a path
        // through a merge block leaves values the path defined to be read after it,
        // and the reconstruction below places the phis that carry them - a repair
        // this pass should not assume succeeded. The clone this policy costs is
        // paid only when there is something to thread, which the check above
        // establishes.
        let result = ssa.edit(
            SsaEditOptions::new()
                .with_verify(true)
                .with_rollback(SsaRollbackPolicy::OnFailure),
            |editor| {
                let mut carried_edges: BTreeMap<SsaVarId, BTreeMap<usize, SsaVarId>> =
                    BTreeMap::new();
                for (pred_block, reroutes) in &threadings {
                    if !reroutes.is_empty()
                        && apply_threading(editor, *pred_block, reroutes, &mut carried_edges, env)
                    {
                        changed = true;
                    }
                }
                // One reconstruction, over the graph every reroute has finished
                // rewriting. A value several rerouted edges carry is named once,
                // with all of them: reconstructing per predecessor would have the
                // second reroute re-deriving a value the first had already
                // replaced.
                let updates: Vec<ValueUpdate> = carried_edges
                    .into_iter()
                    .map(|(value, definitions)| ValueUpdate {
                        value,
                        definitions: definitions.into_iter().collect(),
                    })
                    .collect();
                if !updates.is_empty() {
                    editor.reconstruct_values(&updates)?;
                }
                // Every value the reroutes carried is reconstructed above, and
                // their phi operands were set as each reroute was applied. The
                // full rebuild would recompute every use from its rename group,
                // which on optimised SSA reads a group's own versions where a phi
                // carried another's - replacing the reconstruction with a wrong
                // value. The verification this session runs rejects a repair that
                // left anything out.
                editor.declare_repaired();
                Ok(())
            },
        );
        if result.is_err() {
            // `SsaRollbackPolicy::OnFailure` restored the pre-edit function, so
            // nothing of this rewrite survives and "unchanged" is the truth. The
            // pass group needs no further rollback of its own.
            return Ok(false);
        }
        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use super::*;
    use crate::{
        analysis::context::AnalysisContext,
        events::EventLog,
        ir::{
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            instruction::SsaInstruction,
            ops::kinds::CmpKind,
            phi::{PhiNode, PhiOperand},
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType, run_mock_pass_boundary},
    };

    fn instr(op: SsaOp<MockTarget>) -> SsaInstruction<MockTarget> {
        SsaInstruction::synthetic(op)
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

    /// **A branch the edge into it has already decided is threaded past.** A
    /// shared tail each case of a switch jumps to tests a flag the arm
    /// reaching it fell out of: `if (x == 0) { L: if (x == 0) { ... } }`, where
    /// `L` is also a jump target, so the test holds for the jump and is
    /// already decided for the fall-through. Nothing here is constant - what
    /// the edge proves is the relation - and the taken arm belongs past the
    /// second test. The control: a tail testing another value is not decided
    /// by the arm and keeps its edge. (A tail testing the same value against
    /// another constant *is* decided - zero is not one - and threading it to
    /// the arm that follows is right.)
    #[test]
    fn a_branch_the_edge_already_decided_is_threaded_past() {
        let threaded = |same_value: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
            let flag = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::I32,
            );
            let other = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::I32,
            );
            let zero = local_at(&mut ssa, 0, 0, 0);
            let mut entry = SsaBlock::new(0);
            entry.add_instruction(instr(SsaOp::Const {
                dest: zero,
                value: ConstValue::I32(0),
            }));
            entry.add_instruction(instr(SsaOp::BranchCmp {
                left: flag,
                right: zero,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target: 1,
                false_target: 4,
            }));
            ssa.add_block(entry);

            let mut tail = SsaBlock::new(1);
            let second = local_at(&mut ssa, 1, 1, 0);
            tail.add_instruction(instr(SsaOp::Const {
                dest: second,
                value: ConstValue::I32(0),
            }));
            tail.add_instruction(instr(SsaOp::BranchCmp {
                left: if same_value { flag } else { other },
                right: second,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target: 2,
                false_target: 3,
            }));
            ssa.add_block(tail);

            for index in 2..=3 {
                let mut block = SsaBlock::new(index);
                block.add_instruction(instr(SsaOp::returns(None)));
                ssa.add_block(block);
            }
            // The jump every case of the switch makes, which is why the tail
            // keeps its test.
            let mut jumped = SsaBlock::new(4);
            jumped.add_instruction(instr(SsaOp::Jump { target: 1 }));
            ssa.add_block(jumped);
            ssa.recompute_uses();

            let log: EventLog<MockTarget> = EventLog::new();
            let method = 0u32;
            run_mock_pass_boundary(&mut ssa, "decided tail", |ssa| {
                JumpThreadingPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            ssa.block(0).and_then(SsaBlock::control_terminator).cloned()
        };
        assert!(
            matches!(
                threaded(true),
                Some(SsaOp::BranchCmp {
                    true_target: 2,
                    false_target: 4,
                    ..
                })
            ),
            "the arm that proves the tail's test goes past it, got {:?}",
            threaded(true)
        );
        assert!(
            matches!(
                threaded(false),
                Some(SsaOp::BranchCmp {
                    true_target: 1,
                    false_target: 4,
                    ..
                })
            ),
            "a test of another value is not decided by the arm, got {:?}",
            threaded(false)
        );
    }

    /// A `Switch` on a state that is constant on the incoming edge is threaded
    /// to its case.
    ///
    /// The dispatcher shape a control-flow flattener emits, in the smallest
    /// form that still is one. A `Switch` is a candidate like every other
    /// conditional transfer - see the module docs - and `taken_target_known` answers
    /// for all four kinds.
    #[test]
    fn a_switch_on_a_constant_state_is_threaded_to_its_case() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let two = local_at(&mut ssa, 0, 0, 0);
        let state = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::I32);

        // b0 enters the dispatcher with the state fixed at 2.
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: two,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        // b1 is the dispatcher: nothing but the switch on the state.
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(state, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(two, 0));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Switch {
            value: state,
            targets: vec![2, 3, 4],
            default: 5,
        }));
        ssa.add_block(b1);

        for index in 2..=5 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "switch dispatcher", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let entry = ssa.block(0).and_then(SsaBlock::control_terminator);
        assert!(
            matches!(entry, Some(SsaOp::Jump { target: 4 })),
            "state 2 selects the third case, so the edge goes straight there, got {entry:?}"
        );
    }

    /// **A dispatcher that writes a slot read as storage is not threaded past.**
    ///
    /// The dispatcher also writes `local1 = 0`, a pure operation nothing reads
    /// as a value. The case state 2 selects reads the slot with `ldloc`, which
    /// sees the write: rerouting the entry past the dispatcher would skip it.
    /// With no read through the slot the write is only a value, and the entry
    /// is threaded as before - the control.
    #[test]
    fn a_dispatcher_writing_a_slot_read_as_storage_is_not_bypassed() {
        for read_through_slot in [true, false] {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
            let two = local_at(&mut ssa, 0, 0, 0);
            let state = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::I32);
            let flag = local_at(&mut ssa, 1, 1, 0);
            let read = read_through_slot.then(|| {
                ssa.create_variable(
                    VariableOrigin::Phi,
                    0,
                    DefSite::instruction(4, 0),
                    MockType::I32,
                )
            });

            let mut b0 = SsaBlock::new(0);
            b0.add_instruction(instr(SsaOp::Const {
                dest: two,
                value: ConstValue::I32(2),
            }));
            b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
            ssa.add_block(b0);

            let mut b1 = SsaBlock::new(1);
            let mut phi = PhiNode::new(state, VariableOrigin::Phi);
            phi.add_operand(PhiOperand::new(two, 0));
            b1.add_phi(phi);
            b1.add_instruction(instr(SsaOp::Const {
                dest: flag,
                value: ConstValue::I32(0),
            }));
            b1.add_instruction(instr(SsaOp::Switch {
                value: state,
                targets: vec![2, 3, 4],
                default: 5,
            }));
            ssa.add_block(b1);

            for index in 2..=5 {
                let mut block = SsaBlock::new(index);
                if let (4, Some(read)) = (index, read) {
                    block.add_instruction(instr(SsaOp::LoadLocal {
                        dest: read,
                        local_index: 1,
                    }));
                    block.add_instruction(instr(SsaOp::returns(Some(read))));
                } else {
                    block.add_instruction(instr(SsaOp::returns(None)));
                }
                ssa.add_block(block);
            }
            ssa.recompute_uses();

            let log: EventLog<MockTarget> = EventLog::new();
            run_mock_pass_boundary(&mut ssa, "slot-writing dispatcher", |ssa| {
                JumpThreadingPass.run(
                    ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            let entry = ssa.block(0).and_then(SsaBlock::control_terminator);
            if read_through_slot {
                assert!(
                    matches!(entry, Some(SsaOp::Jump { target: 1 })),
                    "the entry still runs the write, got {entry:?}"
                );
            } else {
                assert!(
                    matches!(entry, Some(SsaOp::Jump { target: 4 })),
                    "nothing reads the slot, so the write is only a value, got {entry:?}"
                );
            }
        }
    }

    /// A predecessor threaded past a dispatcher records the dispatcher's host
    /// block as passed through: it still runs on that edge, and decides nothing
    /// there.
    #[test]
    fn a_threaded_predecessor_passes_through_what_it_skips() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let two = local_at(&mut ssa, 0, 0, 0);
        let state = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::I32);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: two,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        b0.provenance_mut().carry(0x10);
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(state, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(two, 0));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Switch {
            value: state,
            targets: vec![2, 3, 4],
            default: 5,
        }));
        b1.provenance_mut().carry(0x20);
        ssa.add_block(b1);
        for index in 2..=5 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "switch dispatcher", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let entry = ssa.block(0).unwrap_or_else(|| panic!("the entry survives"));
        assert!(matches!(
            entry.control_terminator(),
            Some(SsaOp::Jump { target: 4 })
        ));
        assert_eq!(entry.provenance().carried(), &[0x10]);
        assert_eq!(entry.provenance().passed(), &[0x20]);
        assert_eq!(
            entry.provenance().passed_edges().collect::<Vec<_>>(),
            [Some(4)],
            "on the rerouted edge"
        );
    }

    /// A dispatcher heading a `try` is its entry, and the only block a branch
    /// from outside may enter it at: threading past it would land the edge in
    /// the protected region's middle, which no exception table can express.
    #[test]
    fn a_dispatcher_entering_a_try_is_not_threaded_past() {
        const PROTECTED_END: usize = 4;
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let two = local_at(&mut ssa, 0, 0, 0);
        let state = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: two,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut phi = PhiNode::new(state, VariableOrigin::Phi);
        phi.add_operand(PhiOperand::new(two, 0));
        b1.add_phi(phi);
        b1.add_instruction(instr(SsaOp::Switch {
            value: state,
            targets: vec![2, 3],
            default: 3,
        }));
        ssa.add_block(b1);
        for index in 2..PROTECTED_END {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::Leave {
                target: PROTECTED_END + 1,
            }));
            ssa.add_block(block);
        }
        let mut handler = SsaBlock::new(PROTECTED_END);
        handler.add_instruction(instr(SsaOp::EndFinally));
        ssa.add_block(handler);
        let mut exit = SsaBlock::new(PROTECTED_END + 1);
        exit.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(exit);
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 2,
            try_offset: 0,
            try_length: 0,
            handler_offset: 0,
            handler_length: 0,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(1, PROTECTED_END),
            handler_range: BlockRange::new(PROTECTED_END, PROTECTED_END + 1),
            filter_range: None,
            selector: None,
        }]);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        run_mock_pass_boundary(&mut ssa, "dispatcher entering a try", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let entry = ssa.block(0).and_then(SsaBlock::control_terminator);
        assert!(
            matches!(entry, Some(SsaOp::Jump { target: 1 })),
            "the edge still enters the try at its start, got {entry:?}"
        );
    }

    /// A lowered dispatcher is a chain, and the whole chain is one decision.
    ///
    /// A compiler lowers a `switch` on a state into a run of comparisons, one
    /// per case, so "the dispatcher" is several blocks rather than one. Every
    /// link tests the same value, so a predecessor that fixes it decides all of
    /// them at once, and threading has to follow the chain to the case rather
    /// than stop at its first link.
    ///
    /// The links here hold a comparison each, which the old "inert block" rule
    /// also refused: they are pure and nothing outside reads what they compute,
    /// which is the condition that replaced it.
    #[test]
    fn a_chain_of_comparisons_is_threaded_the_whole_way_to_its_case() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let two = local_at(&mut ssa, 0, 0, 0);
        let state = ssa.create_variable(VariableOrigin::Phi, 0, DefSite::phi(1), MockType::I32);

        // b0 enters the chain with the state fixed at 2.
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: two,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        // b1..b3 are the chain: `state == 0`, `state == 1`, `state == 2`, each
        // with its own constant to compare against, falling through to the next.
        for (index, case) in [(1usize, 0i32), (2, 1), (3, 2)] {
            let mut block = SsaBlock::new(index);
            if index == 1 {
                let mut phi = PhiNode::new(state, VariableOrigin::Phi);
                phi.add_operand(PhiOperand::new(two, 0));
                block.add_phi(phi);
            }
            let want = local_at(&mut ssa, u16::try_from(index).unwrap_or(0), index, 0);
            block.add_instruction(instr(SsaOp::Const {
                dest: want,
                value: ConstValue::I32(case),
            }));
            block.add_instruction(instr(SsaOp::BranchCmp {
                left: state,
                right: want,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target: index.saturating_add(4),
                false_target: index.saturating_add(1),
            }));
            ssa.add_block(block);
        }
        // b4 is the default, b5..b7 the cases.
        for index in 4..=7 {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "dispatcher chain", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        let entry = ssa.block(0).and_then(SsaBlock::control_terminator);
        assert!(
            matches!(entry, Some(SsaOp::Jump { target: 7 })),
            "state 2 is decided by the third link, so the edge goes straight to \
             its case rather than into the chain, got {entry:?}"
        );
    }

    /// **A state the edge pins to one value decides what is computed from
    /// it.** A flattened dispatcher's case computes its next state, `((i < n)
    /// ^ 1) + 2`, and branches back to the head unless it is 2, so on that
    /// edge the state is 3; RV64's head compares the state through its sign
    /// extension. The conversion is computed from the pinned state, and the
    /// edge goes straight to the case the head picks for 3. The control: a
    /// state the edge leaves two values decides nothing.
    #[test]
    fn a_value_the_edge_pins_decides_what_is_computed_from_it() {
        let rerouted = |mask: i32| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 6);
            let x = ssa.create_variable(
                VariableOrigin::Argument(0),
                0,
                DefSite::entry(),
                MockType::I32,
            );
            let low = local_at(&mut ssa, 0, 0, 0);
            let picked = local_at(&mut ssa, 1, 0, 1);
            let two = local_at(&mut ssa, 2, 0, 2);
            let state = local_at(&mut ssa, 3, 0, 3);
            let wide = ssa.create_variable(
                VariableOrigin::Local(4),
                0,
                DefSite::instruction(1, 0),
                MockType::I64,
            );
            let three = ssa.create_variable(
                VariableOrigin::Local(5),
                0,
                DefSite::instruction(1, 1),
                MockType::I64,
            );
            let mut b0 = SsaBlock::new(0);
            for op in [
                SsaOp::Const {
                    dest: low,
                    value: ConstValue::I32(mask),
                },
                SsaOp::And {
                    dest: picked,
                    left: x,
                    right: low,
                    flags: None,
                },
                SsaOp::Const {
                    dest: two,
                    value: ConstValue::I32(2),
                },
                SsaOp::Add {
                    dest: state,
                    left: picked,
                    right: two,
                    flags: None,
                },
                SsaOp::BranchCmp {
                    left: state,
                    right: two,
                    cmp: CmpKind::Ne,
                    unsigned: false,
                    true_target: 1,
                    false_target: 3,
                },
            ] {
                b0.add_instruction(instr(op));
            }
            ssa.add_block(b0);
            let mut b1 = SsaBlock::new(1);
            for op in [
                SsaOp::IntConv {
                    dest: wide,
                    operand: state,
                    target: MockType::I64,
                    overflow_check: false,
                    unsigned: false,
                },
                SsaOp::Const {
                    dest: three,
                    value: ConstValue::I64(3),
                },
                SsaOp::BranchCmp {
                    left: wide,
                    right: three,
                    cmp: CmpKind::Eq,
                    unsigned: false,
                    true_target: 2,
                    false_target: 3,
                },
            ] {
                b1.add_instruction(instr(op));
            }
            ssa.add_block(b1);
            add_returns(&mut ssa, 2..=3);
            ssa.recompute_uses();
            match threaded(ssa).first().cloned().flatten() {
                Some(SsaOp::BranchCmp { true_target, .. }) => true_target,
                other => panic!("the entry keeps its branch, got {other:?}"),
            }
        };
        assert_eq!(
            rerouted(1),
            2,
            "the state is 3 on the edge, so the head picks b2"
        );
        assert_eq!(
            rerouted(3),
            1,
            "control: 3 or 4 on the edge decides nothing"
        );
    }

    /// Runs the pass over `ssa` and returns its terminators afterwards.
    fn threaded(mut ssa: SsaFunction<MockTarget>) -> Vec<Option<SsaOp<MockTarget>>> {
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run_mock_pass_boundary(&mut ssa, "threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        (0..ssa.block_count())
            .map(|index| {
                ssa.block(index)
                    .and_then(SsaBlock::control_terminator)
                    .cloned()
            })
            .collect()
    }

    /// Appends one `return` block per index in `indices`.
    fn add_returns(ssa: &mut SsaFunction<MockTarget>, indices: RangeInclusive<usize>) {
        for index in indices {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(block);
        }
    }

    /// A conditional predecessor keeps its condition and its other arm.
    ///
    /// ```text
    /// b0: branch arg -> b1, b2
    /// b1: branch k (= 1) -> b3, b4
    /// ```
    ///
    /// The edge `b0 -> b1` proves `b1` goes to `b3`, which says nothing about
    /// `arg`: `b0` must become `branch arg -> b3, b2`, not `jump b3`, or the
    /// `arg == 0` path to `b2` is lost.
    #[test]
    fn a_conditional_predecessor_keeps_its_other_arm() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 2);
        let arg = local_at(&mut ssa, 0, 0, 0);
        let k = local_at(&mut ssa, 1, 0, 1);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: arg,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: k,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: arg,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: k,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b1);
        add_returns(&mut ssa, 2..=4);
        ssa.recompute_uses();

        // The rebuild at the edit boundary renames variables, so the kind and
        // the arms are what is compared.
        let entry = threaded(ssa).first().cloned().flatten();
        assert!(
            matches!(entry, Some(SsaOp::Branch { .. }))
                && entry.as_ref().and_then(SsaOp::two_way_targets) == Some((3, 2)),
            "b0 must stay a branch on `arg`, to b3 and b2, got {entry:?}"
        );
    }

    /// One predecessor whose two arms lead into two different threadable
    /// blocks gets both arms rerouted, each to its own block's target.
    ///
    /// ```text
    /// b0: branch_cmp arg == 0 -> b1, b2
    /// b1: branch k1 (= 1) -> b3, b4
    /// b2: branch k0 (= 0) -> b4, b5
    /// ```
    #[test]
    fn two_arms_into_two_threadable_blocks_are_both_rerouted() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 4);
        let arg = local_at(&mut ssa, 0, 0, 0);
        let zero = local_at(&mut ssa, 1, 0, 1);
        let k1 = local_at(&mut ssa, 2, 0, 2);
        let k0 = local_at(&mut ssa, 3, 0, 3);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: arg,
            arg_index: 0,
        }));
        for (dest, value) in [(zero, 0), (k1, 1), (k0, 0)] {
            b0.add_instruction(instr(SsaOp::Const {
                dest,
                value: ConstValue::I32(value),
            }));
        }
        b0.add_instruction(instr(SsaOp::BranchCmp {
            left: arg,
            right: zero,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 1,
            false_target: 2,
        }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: k1,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b1);
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Branch {
            condition: k0,
            true_target: 4,
            false_target: 5,
        }));
        ssa.add_block(b2);
        add_returns(&mut ssa, 3..=5);
        ssa.recompute_uses();

        let entry = threaded(ssa).first().cloned().flatten();
        assert!(
            matches!(
                entry,
                Some(SsaOp::BranchCmp {
                    cmp: CmpKind::Eq,
                    ..
                })
            ) && entry.as_ref().and_then(SsaOp::two_way_targets) == Some((3, 5)),
            "b0 must stay a compare on `arg`, to b3 and b5, got {entry:?}"
        );
    }

    /// Runs `ssa` from its entry and returns the value it returns.
    fn returned(ssa: &SsaFunction<MockTarget>) -> Option<ConstValue<MockTarget>> {
        let mut evaluator = SsaEvaluator::new(ssa, MACHINE);
        let trace = evaluator.execute_from_entry(1_000);
        let last = trace.last_block()?;
        match ssa.block(last)?.control_terminator()? {
            SsaOp::Return {
                value: Some(value), ..
            } => evaluator.get_concrete(*value).cloned(),
            _ => None,
        }
    }

    /// Threading an arm into a join keeps the value the join's phi receives.
    ///
    /// ```text
    /// b0: x = 10; branch c -> b1, b2
    /// b1: branch k (= 1) -> b3, b4
    /// b2: x' = 20; jump b3
    /// b3: r = phi(x @ b1, x' @ b2); return r
    /// ```
    ///
    /// Threading gives `b3` a new predecessor, `b0`, whose phi operand must be
    /// the value that reached `b3` through `b1`. The function returns 10 when
    /// `c` holds and 20 when it does not, before threading and after.
    #[test]
    fn threading_into_a_join_keeps_what_the_function_returns() {
        for condition in [1, 0] {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
            let x = local_at(&mut ssa, 0, 0, 0);
            let c = local_at(&mut ssa, 1, 0, 1);
            let k = local_at(&mut ssa, 2, 0, 2);
            let x2 = ssa.create_variable(
                VariableOrigin::Local(0),
                1,
                DefSite::instruction(2, 0),
                MockType::I32,
            );
            let r =
                ssa.create_variable(VariableOrigin::Local(0), 2, DefSite::phi(3), MockType::I32);
            let mut b0 = SsaBlock::new(0);
            for (dest, value) in [(x, 10), (c, condition), (k, 1)] {
                b0.add_instruction(instr(SsaOp::Const {
                    dest,
                    value: ConstValue::I32(value),
                }));
            }
            b0.add_instruction(instr(SsaOp::Branch {
                condition: c,
                true_target: 1,
                false_target: 2,
            }));
            ssa.add_block(b0);
            let mut b1 = SsaBlock::new(1);
            b1.add_instruction(instr(SsaOp::Branch {
                condition: k,
                true_target: 3,
                false_target: 4,
            }));
            ssa.add_block(b1);
            let mut b2 = SsaBlock::new(2);
            b2.add_instruction(instr(SsaOp::Const {
                dest: x2,
                value: ConstValue::I32(20),
            }));
            b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
            ssa.add_block(b2);
            let mut b3 = SsaBlock::new(3);
            let mut phi = PhiNode::new(r, VariableOrigin::Local(0));
            phi.add_operand(PhiOperand::new(x, 1));
            phi.add_operand(PhiOperand::new(x2, 2));
            b3.add_phi(phi);
            b3.add_instruction(instr(SsaOp::returns(Some(r))));
            ssa.add_block(b3);
            add_returns(&mut ssa, 4..=4);
            ssa.recompute_uses();

            let expected = if condition == 1 { 10 } else { 20 };
            assert_eq!(
                returned(&ssa),
                Some(ConstValue::I32(expected)),
                "the fixture"
            );
            let log: EventLog<MockTarget> = EventLog::new();
            let method = 0u32;
            let changed = run_mock_pass_boundary(&mut ssa, "threading into a join", |ssa| {
                JumpThreadingPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            assert!(changed, "the edge b0 -> b1 is threadable");
            assert_eq!(
                returned(&ssa),
                Some(ConstValue::I32(expected)),
                "c = {condition}, after threading:\n{ssa}"
            );
        }
    }

    /// A reroute that would give one edge two phi values is refused.
    ///
    /// ```text
    /// b0: a = 10; b = 20; branch c -> b1, b3
    /// b1: branch k (= 1) -> b3, b4
    /// b3: r = phi(a @ b0, b @ b1); return r
    /// ```
    ///
    /// Rerouting `b0`'s arm into `b1` straight to `b3` would give `b0` two
    /// edges into `b3` - one carrying `a`, one carrying `b` - which a phi
    /// cannot express. The function returns 20 when `c` holds and 10 when it
    /// does not, and must go on doing so.
    #[test]
    fn a_reroute_that_would_merge_two_phi_values_is_refused() {
        for condition in [1, 0] {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 5);
            let a = local_at(&mut ssa, 0, 0, 0);
            let b = local_at(&mut ssa, 1, 0, 1);
            let c = local_at(&mut ssa, 2, 0, 2);
            let k = local_at(&mut ssa, 3, 0, 3);
            let r =
                ssa.create_variable(VariableOrigin::Local(4), 0, DefSite::phi(3), MockType::I32);
            let mut b0 = SsaBlock::new(0);
            for (dest, value) in [(a, 10), (b, 20), (c, condition), (k, 1)] {
                b0.add_instruction(instr(SsaOp::Const {
                    dest,
                    value: ConstValue::I32(value),
                }));
            }
            b0.add_instruction(instr(SsaOp::Branch {
                condition: c,
                true_target: 1,
                false_target: 3,
            }));
            ssa.add_block(b0);
            let mut b1 = SsaBlock::new(1);
            b1.add_instruction(instr(SsaOp::Branch {
                condition: k,
                true_target: 3,
                false_target: 4,
            }));
            ssa.add_block(b1);
            let mut b2 = SsaBlock::new(2);
            b2.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(b2);
            let mut b3 = SsaBlock::new(3);
            let mut phi = PhiNode::new(r, VariableOrigin::Local(4));
            phi.add_operand(PhiOperand::new(a, 0));
            phi.add_operand(PhiOperand::new(b, 1));
            b3.add_phi(phi);
            b3.add_instruction(instr(SsaOp::returns(Some(r))));
            ssa.add_block(b3);
            add_returns(&mut ssa, 4..=4);
            ssa.recompute_uses();

            let expected = if condition == 1 { 20 } else { 10 };
            assert_eq!(
                returned(&ssa),
                Some(ConstValue::I32(expected)),
                "the fixture"
            );
            let log: EventLog<MockTarget> = EventLog::new();
            let method = 0u32;
            run_mock_pass_boundary(&mut ssa, "conflicting phi values", |ssa| {
                JumpThreadingPass.run(
                    ssa,
                    &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
                ) == Ok(true)
            });
            assert_eq!(
                returned(&ssa),
                Some(ConstValue::I32(expected)),
                "c = {condition}, after threading:\n{ssa}"
            );
        }
    }

    /// A predecessor supplying a constant-true condition is threaded to the true arm.
    #[test]
    fn thread_constant_condition_to_true_branch() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 2);
        let true_val = local_at(&mut ssa, 0, 0, 0);
        let cond = local_at(&mut ssa, 1, 0, 1);

        // B0 is a predecessor that jumps to B1 (where the branch lives)
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: true_val,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        // B1 has the branch - it has a predecessor (B0) so threading can work
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        // B2 true target
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(Some(true_val))));
        ssa.add_block(b2);

        // B3 false target
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "constant branch threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "constant true condition should be threaded");
        assert!(
            log.has(EventKind::BranchSimplified) || log.has(EventKind::ControlFlowRestructured)
        );
    }

    /// A phi resolved per-predecessor lets a merged condition be threaded.
    #[test]
    fn thread_with_phi_value_resolution() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let true_cond = local_at(&mut ssa, 0, 0, 0);
        let false_cond = local_at(&mut ssa, 1, 1, 0);
        let merged_cond =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: true_cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: false_cond,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(merged_cond, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(true_cond, 0));
        phi.add_operand(PhiOperand::new(false_cond, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Branch {
            condition: merged_cond,
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
        let changed = run_mock_pass_boundary(&mut ssa, "phi value branch threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(changed, "phi-based threading should work");
    }

    /// A bypassed phi value read after the branch is reconstructed, not refused.
    ///
    /// The phi is the name a variable carries on the bypassed edge, and the
    /// reroute states what that variable is instead of declining to move.
    #[test]
    fn a_bypassed_phi_value_read_after_the_branch_is_reconstructed() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let true_cond = local_at(&mut ssa, 0, 0, 0);
        let false_cond = local_at(&mut ssa, 1, 1, 0);
        let merged_cond =
            ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(2), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: true_cond,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Const {
            dest: false_cond,
            value: ConstValue::I32(0),
        }));
        b1.add_instruction(instr(SsaOp::Jump { target: 2 }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(merged_cond, VariableOrigin::Local(2));
        phi.add_operand(PhiOperand::new(true_cond, 0));
        phi.add_operand(PhiOperand::new(false_cond, 1));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::Branch {
            condition: merged_cond,
            true_target: 3,
            false_target: 4,
        }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(Some(merged_cond))));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "live phi branch threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });

        assert!(changed, "the edge whose value is proven should be threaded");
        // `b0` sets the condition to 1, so its edge is rerouted past `b2` to
        // `b3` - which reads the phi `b2` defined. The read is answered by
        // reconstruction, not by refusing the thread: on that edge the phi's
        // value is `b0`'s own, and that is what `b3` must now return.
        let returned = ssa
            .block(3)
            .and_then(|block| block.instructions().first())
            .map(|instr| instr.op().clone());
        assert_eq!(
            returned,
            Some(SsaOp::returns(Some(true_cond))),
            "the bypassed phi's value on this edge is the operand it carried"
        );
    }

    /// **A reroute does not enter a loop past its head.** `B1` heads a loop
    /// (`B1 -> B2 -> B3 -> B1`) and tests a flag the entry `B0` sets to 0, so
    /// on the entry's edge it goes to `B2`; the flag `B3` sets on the back edge
    /// is an argument, which decides nothing. Rerouted, the entry would jump
    /// into the loop's body past its head, which would no longer be the loop's
    /// one entry: the body prints twice, with a `goto` into it (`which`'s
    /// `path_clean_up`). The entry keeps its edge to the head. Where `B3`
    /// sets the flag to a constant instead, `B1` is a dispatcher over a state
    /// every back edge names (`ob_flattened`), and the entry is threaded to
    /// the case its state selects.
    #[test]
    fn a_reroute_does_not_enter_a_loop_past_its_head() {
        let entry = |latch_sets_constant: bool| {
            let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 5);
            let zero = local_at(&mut ssa, 0, 0, 0);
            let again = local_at(&mut ssa, 1, 3, 0);
            let flag =
                ssa.create_variable(VariableOrigin::Local(2), 0, DefSite::phi(1), MockType::I32);

            let mut b0 = SsaBlock::new(0);
            b0.add_instruction(instr(SsaOp::Const {
                dest: zero,
                value: ConstValue::I32(0),
            }));
            b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
            ssa.add_block(b0);

            let mut b1 = SsaBlock::new(1);
            let mut flag_phi = PhiNode::new(flag, VariableOrigin::Local(2));
            flag_phi.add_operand(PhiOperand::new(zero, 0));
            flag_phi.add_operand(PhiOperand::new(again, 3));
            b1.add_phi(flag_phi);
            b1.add_instruction(instr(SsaOp::BranchCmp {
                left: flag,
                right: zero,
                cmp: CmpKind::Eq,
                unsigned: false,
                true_target: 2,
                false_target: 4,
            }));
            ssa.add_block(b1);

            let mut b2 = SsaBlock::new(2);
            b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
            ssa.add_block(b2);

            let mut b3 = SsaBlock::new(3);
            b3.add_instruction(instr(if latch_sets_constant {
                SsaOp::Const {
                    dest: again,
                    value: ConstValue::I32(1),
                }
            } else {
                SsaOp::LoadArg {
                    dest: again,
                    arg_index: 0,
                }
            }));
            b3.add_instruction(instr(SsaOp::Jump { target: 1 }));
            ssa.add_block(b3);

            let mut b4 = SsaBlock::new(4);
            b4.add_instruction(instr(SsaOp::returns(None)));
            ssa.add_block(b4);
            ssa.recompute_uses();

            let log: EventLog<MockTarget> = EventLog::new();
            assert!(
                JumpThreadingPass
                    .run(
                        &mut ssa,
                        &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                    )
                    .is_ok()
            );
            ssa.block(0)
                .and_then(|block| block.control_terminator())
                .cloned()
        };
        assert_eq!(
            entry(false),
            Some(SsaOp::Jump { target: 1 }),
            "a loop's own test"
        );
        assert_eq!(
            entry(true),
            Some(SsaOp::Jump { target: 2 }),
            "a dispatcher's state"
        );
    }

    /// **A dispatcher's entry is threaded to the case its state selects.**
    /// MIPS `-O2` `ob_flattened`: the head `B1` tests a state the entry sets to
    /// 0, and its one back edge carries the state two cases set to 1 each,
    /// merged in `B4` before it: a constant on every way back. The head also
    /// computes the value a case branches on, so the back edge, whose path
    /// would skip that computation, is not rerouted. The head is a dispatcher all the
    /// same, and the entry goes to case 0, `B2`; read as a loop's own test it
    /// stayed, and the dispatch printed as an `if` inside the loop.
    #[test]
    fn a_dispatchers_entry_is_threaded_to_its_case() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 6);
        let zero = local_at(&mut ssa, 0, 0, 0);
        let first = local_at(&mut ssa, 1, 5, 0);
        let second = local_at(&mut ssa, 2, 6, 0);
        let state =
            ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(1), MockType::I32);
        let next = ssa.create_variable(VariableOrigin::Local(4), 0, DefSite::phi(4), MockType::I32);
        let seed = local_at(&mut ssa, 5, 0, 1);
        let argument = ssa.create_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(1, 0),
            MockType::I32,
        );

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: zero,
            value: ConstValue::I32(0),
        }));
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: seed,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut state_phi = PhiNode::new(state, VariableOrigin::Local(3));
        state_phi.add_operand(PhiOperand::new(zero, 0));
        state_phi.add_operand(PhiOperand::new(next, 4));
        b1.add_phi(state_phi);
        b1.add_instruction(instr(SsaOp::Add {
            dest: argument,
            left: seed,
            right: seed,
            flags: None,
        }));
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: state,
            right: zero,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        // Case 0, and the case that picks its way on by the argument.
        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Jump { target: 5 }));
        ssa.add_block(b2);
        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Branch {
            condition: argument,
            true_target: 6,
            false_target: 7,
        }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        let mut next_phi = PhiNode::new(next, VariableOrigin::Local(4));
        next_phi.add_operand(PhiOperand::new(first, 5));
        next_phi.add_operand(PhiOperand::new(second, 6));
        b4.add_phi(next_phi);
        b4.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b4);

        for (index, dest) in [(5, first), (6, second)] {
            let mut block = SsaBlock::new(index);
            block.add_instruction(instr(SsaOp::Const {
                dest,
                value: ConstValue::I32(1),
            }));
            block.add_instruction(instr(SsaOp::Jump { target: 4 }));
            ssa.add_block(block);
        }
        let mut b7 = SsaBlock::new(7);
        b7.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b7);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            JumpThreadingPass.run(
                &mut ssa,
                &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
            ) == Ok(true),
            "{ssa:?}"
        );
        assert_eq!(
            ssa.block(0)
                .and_then(|block| block.control_terminator())
                .cloned(),
            Some(SsaOp::Jump { target: 2 }),
            "{ssa:?}"
        );
    }

    /// A case rerouted past the block that sets the next state still carries
    /// the value it computed.
    ///
    /// A flattened dispatcher: `B2` computes a value, `B3` only sets the next
    /// state, and the dispatcher `B1` merges the value (`null` on entry) and
    /// sends state 20 to `B4`, which returns it. Threading reroutes `B2` past
    /// `B3` and `B1` straight to `B4`. The value the dispatcher's phi receives
    /// from `B3` was recorded as defined *at* `B3`, the block being bypassed;
    /// the reconstruction then found the entry's `null` as the latest
    /// definition on `B2`'s new edge, and `B4` returned `null`.
    #[test]
    fn a_case_rerouted_past_its_state_setter_keeps_its_value() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(1, 7);
        let first_state = local_at(&mut ssa, 0, 0, 0);
        let twenty = local_at(&mut ssa, 1, 0, 1);
        let null = local_at(&mut ssa, 2, 0, 2);
        let computed = local_at(&mut ssa, 3, 2, 0);
        let next_state = local_at(&mut ssa, 4, 3, 0);
        let state =
            ssa.create_variable(VariableOrigin::Local(5), 0, DefSite::phi(1), MockType::I32);
        let value =
            ssa.create_variable(VariableOrigin::Local(6), 0, DefSite::phi(1), MockType::I32);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: first_state,
            value: ConstValue::I32(10),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: twenty,
            value: ConstValue::I32(20),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: null,
            value: ConstValue::Null,
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        let mut state_phi = PhiNode::new(state, VariableOrigin::Local(5));
        state_phi.add_operand(PhiOperand::new(first_state, 0));
        state_phi.add_operand(PhiOperand::new(next_state, 3));
        b1.add_phi(state_phi);
        let mut value_phi = PhiNode::new(value, VariableOrigin::Local(6));
        value_phi.add_operand(PhiOperand::new(null, 0));
        value_phi.add_operand(PhiOperand::new(computed, 3));
        b1.add_phi(value_phi);
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: state,
            right: twenty,
            cmp: CmpKind::Eq,
            unsigned: false,
            true_target: 4,
            false_target: 2,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::LoadArg {
            dest: computed,
            arg_index: 0,
        }));
        b2.add_instruction(instr(SsaOp::Jump { target: 3 }));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::Const {
            dest: next_state,
            value: ConstValue::I32(20),
        }));
        b3.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b3);

        let mut b4 = SsaBlock::new(4);
        b4.add_instruction(instr(SsaOp::returns(Some(value))));
        ssa.add_block(b4);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        assert!(
            JumpThreadingPass
                .run(
                    &mut ssa,
                    &PassEnv::new(&0u32, &log, AnalysisContext::new(MACHINE))
                )
                .is_ok()
        );
        assert!(ssa.take_edit_failure().is_none());
        let returned = ssa
            .block(4)
            .and_then(|block| block.instructions().last())
            .and_then(|instr| match instr.op() {
                SsaOp::Return { value, .. } => *value,
                _ => None,
            })
            .expect("B4 returns a value");
        let arrives = |value: SsaVarId| -> bool {
            match ssa.get_definition(value) {
                Some(SsaOp::LoadArg { .. }) => true,
                Some(SsaOp::Const { .. }) => false,
                _ => ssa.find_phi_defining(value).is_some_and(|(_, phi)| {
                    phi.operands().iter().all(|operand| {
                        matches!(
                            ssa.get_definition(operand.value()),
                            Some(SsaOp::LoadArg { .. })
                        )
                    })
                }),
            }
        };
        assert!(
            arrives(returned),
            "B4 is only reached after B2 computed the value, and returns it:\n{ssa:?}"
        );
    }

    /// A condition sourced from a runtime argument cannot be threaded.
    #[test]
    fn no_threading_when_all_unknown() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 1);
        let cond = local_at(&mut ssa, 0, 0, 0);

        let mut b0 = SsaBlock::new(0);
        // cond comes from LoadArg, not a constant - can't thread
        b0.add_instruction(instr(SsaOp::LoadArg {
            dest: cond,
            arg_index: 0,
        }));
        b0.add_instruction(instr(SsaOp::Branch {
            condition: cond,
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
        let changed = run_mock_pass_boundary(&mut ssa, "unknown condition threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(
            !changed,
            "no threading should occur when condition cannot be resolved"
        );
    }

    /// An empty function produces no threading changes.
    #[test]
    fn empty_function_no_changes() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "empty threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed);
    }

    /// A condition reached through a copy chain is traced to a constant and threaded.
    #[test]
    fn threading_with_copy_chain() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 3);
        let src = local_at(&mut ssa, 0, 0, 0);
        let mid = local_at(&mut ssa, 1, 0, 1);
        let cond = local_at(&mut ssa, 2, 0, 2);

        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: src,
            value: ConstValue::I32(1),
        }));
        b0.add_instruction(instr(SsaOp::Copy { dest: mid, src }));
        b0.add_instruction(instr(SsaOp::Copy {
            dest: cond,
            src: mid,
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Branch {
            condition: cond,
            true_target: 2,
            false_target: 3,
        }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b2);

        let mut b3 = SsaBlock::new(3);
        b3.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b3);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "copy-chain branch threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        // The evaluator should trace through Copy chain to find constant
        assert!(changed, "copy-chain condition should be threaded");
    }

    /// A block with no branch terminator offers nothing to thread.
    #[test]
    fn no_branch_terminator_no_threading() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b0);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "non-branch threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "no branch means no threading");
    }

    /// A plain `Jump` terminator is left alone by this pass.
    #[test]
    fn jump_not_branch_not_threaded() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 0);
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::returns(None)));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "jump terminator threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });
        assert!(!changed, "Jump should not be threaded by this pass");
    }
    /// A back edge that proves a range decides a link no value could.
    ///
    /// `-O1` and above compute a flattened dispatcher's state rather than
    /// writing it as a literal - `state = (i < n) ? 2 : 3` - so no edge into
    /// the dispatcher carries a value. The edge still *proves* one: its
    /// predecessor branched on `state != 2`, and taking that arm over a state
    /// in `[2, 3]` leaves `state == 3`.
    #[test]
    fn a_range_the_edge_proves_decides_a_link_no_value_could() {
        let mut ssa: SsaFunction<MockTarget> = SsaFunction::new(0, 4);
        let two = local_at(&mut ssa, 0, 0, 0);
        let three = local_at(&mut ssa, 1, 0, 1);
        let state = local_at(&mut ssa, 2, 1, 0);
        let merged =
            ssa.create_variable(VariableOrigin::Local(3), 0, DefSite::phi(2), MockType::I32);

        // b0: entry, jumps into the loop body.
        let mut b0 = SsaBlock::new(0);
        b0.add_instruction(instr(SsaOp::Const {
            dest: two,
            value: ConstValue::I32(2),
        }));
        b0.add_instruction(instr(SsaOp::Const {
            dest: three,
            value: ConstValue::I32(3),
        }));
        b0.add_instruction(instr(SsaOp::Jump { target: 1 }));
        ssa.add_block(b0);

        // b1: the state is one of the two literals, and the branch to the
        // dispatcher is taken only when it is not 2 - which leaves 3.
        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Select {
            dest: state,
            condition: two,
            true_val: three,
            false_val: two,
        }));
        b1.add_instruction(instr(SsaOp::BranchCmp {
            left: state,
            right: two,
            cmp: CmpKind::Ne,
            unsigned: false,
            true_target: 2,
            false_target: 4,
        }));
        ssa.add_block(b1);

        // b2: the dispatcher head, entered from b1 and from b3.
        let mut b2 = SsaBlock::new(2);
        let mut phi = PhiNode::new(merged, VariableOrigin::Local(3));
        phi.add_operand(PhiOperand::new(state, 1));
        phi.add_operand(PhiOperand::new(two, 3));
        b2.add_phi(phi);
        b2.add_instruction(instr(SsaOp::BranchCmp {
            left: merged,
            right: two,
            cmp: CmpKind::Gt,
            unsigned: false,
            true_target: 5,
            false_target: 3,
        }));
        ssa.add_block(b2);

        for (idx, target) in [(3usize, 2usize), (4, 4), (5, 5)] {
            let mut block = SsaBlock::new(idx);
            block.add_instruction(instr(SsaOp::Jump { target }));
            ssa.add_block(block);
        }
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "range threading", |ssa| {
            JumpThreadingPass.run(
                ssa,
                &PassEnv::new(&method, &log, AnalysisContext::new(MACHINE)),
            ) == Ok(true)
        });

        assert!(changed, "the edge proves which arm the dispatcher takes");
        // Every block between the entry and the case the state selects is
        // gone: the edge proved which arm the dispatcher takes, so the
        // dispatcher itself had nothing left to decide.
        assert_eq!(
            ssa.block(0).and_then(|block| block.control_terminator()),
            Some(&SsaOp::Jump { target: 5 }),
            "the entry reaches the selected case with no dispatch in between"
        );
        assert!(
            ssa.block(2)
                .is_none_or(|block| block.control_terminator().is_none()),
            "the dispatcher head is unreachable and cleared"
        );
    }
}
