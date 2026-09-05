//! Value Range Propagation pass — simplifies branches and comparisons
//! using sparse conditional constant propagation with [`ValueRange`].
//!
//! # Algorithm
//!
//! Uses a worklist-based dataflow analysis similar to Wegman-Zadeck SCCP,
//! but tracks [`ValueRange`] intervals instead of just constants:
//!
//! 1. **Initialize**: every variable starts at `ValueRange::top()` (unknown).
//!    The entry block is marked executable, and its CFG successors are
//!    added to the worklist.
//! 2. **Propagate**: process CFG edges and SSA variables from worklists.
//!    - When a new edge is executable, process phi nodes at the target
//!      (join ranges from executable predecessors).
//!    - When a variable's range changes, re-evaluate all instructions
//!      and phi nodes that use it.
//!    - For `Branch` and `Switch` terminators, only propagate to targets
//!      that are reachable given the condition's current range.
//! 3. **Evaluate**: compute output ranges for each operation type:
//!    - `Const` → exact constant range.
//!    - `Copy` → same range as source.
//!    - `Add`, `Sub`, `Mul` → arithmetic on operand ranges.
//!    - `Shr` (unsigned, non-negative) → shifted range.
//!    - `Rem` (positive divisor, non-negative dividend) → bounded range.
//!    - `And` → bounded by mask.
//!    - `Ceq`, `Clt`, `Cgt` → bounded to 0 or 1.
//!    - `ArrayLength` → non-negative.
//! 4. **Apply results**: simplify branches whose condition range is
//!    provably constant, and replace comparisons whose operands' ranges
//!    guarantee the result.
//!
//! # Limitations
//!
//! - Does not track non-numeric types (pointers, objects).
//! - Does not handle loop-carried ranges (widening/ narrowing not
//!   implemented — loops iterate up to `max_iterations` then stop).

use std::{
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
};

use crate::{
    analysis::{
        exceptions::EhCfg,
        range::{IntervalRange, ValueRange},
    },
    bitset::BitSet,
    events::{EventKind, EventListener},
    graph::{NodeId, RootedGraph, Successors, algorithms::DominatorTree},
    ir::{
        block::SsaBlock,
        function::{SsaEditOptions, SsaFunction},
        instruction::SsaInstruction,
        ops::SsaOp,
        phi::PhiNode,
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// Converged per-variable value ranges for one function.
///
/// Only produced by [`analyze`], and only when the analysis reached its
/// fixpoint — the analysis is optimistic, so a partial run's ranges are too
/// narrow and prove things that are not true. There is deliberately no way to
/// obtain this type from an unconverged run.
#[derive(Debug, Clone, Default)]
pub struct ValueRanges {
    ranges: HashMap<SsaVarId, ValueRange>,
}

impl ValueRanges {
    /// Returns the range proved for `var`, if any.
    #[must_use]
    pub fn get(&self, var: SsaVarId) -> Option<&ValueRange> {
        self.ranges.get(&var)
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

    /// Returns `var`'s range as known *at* `block`, refined by the branch
    /// guards that dominate it.
    ///
    /// The whole-function range is a single fact per variable, so it cannot say
    /// that `idx` is bounded only on the path a `cmp idx, N; ja default` guard
    /// admits — which is exactly the fact a switch dispatch depends on. This
    /// walks up the dominator tree from `block`; whenever the path arrives
    /// through one arm of a conditional branch, the comparison driving that
    /// branch is intersected into the result.
    ///
    /// Refinement only applies where the guard is *unambiguous*: the path must
    /// enter through exactly one arm, so a block reachable through both arms of
    /// the same branch learns nothing from it.
    #[must_use]
    pub fn range_at<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        dominators: &DominatorTree,
        var: SsaVarId,
        block: usize,
    ) -> ValueRange {
        let mut range = self.get(var).cloned().unwrap_or_default();
        let mut current = NodeId::new(block);
        // Bound the climb by the block count: a dominator chain cannot be
        // longer, and this keeps a malformed tree from looping.
        for _ in 0..ssa.block_count() {
            let Some(parent) = dominators.immediate_dominator(current) else {
                break;
            };
            if let Some(refined) = self.refine_through_edge(ssa, dominators, var, parent, current) {
                range = range.meet(&refined);
            }
            current = parent;
        }
        range
    }

    /// Returns what `parent`'s terminator proves about `var` on the path that
    /// reaches `child`, if anything.
    fn refine_through_edge<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        dominators: &DominatorTree,
        var: SsaVarId,
        parent: NodeId,
        child: NodeId,
    ) -> Option<ValueRange> {
        let block = ssa.block(parent.index())?;
        let SsaOp::Branch {
            condition,
            true_target,
            false_target,
        } = block.control_terminator()?
        else {
            return None;
        };
        // The guard only holds if control necessarily arrived through one arm.
        // A block that both arms reach learns nothing.
        let via_true =
            dominators.dominates(NodeId::new(*true_target), child) || child.index() == *true_target;
        let via_false = dominators.dominates(NodeId::new(*false_target), child)
            || child.index() == *false_target;
        let taken = match (via_true, via_false) {
            (true, false) => true,
            (false, true) => false,
            _ => return None,
        };
        let comparison = ssa.get_definition(*condition)?;
        self.refine_from_comparison(comparison, var, taken)
    }

    /// Intersects what a comparison proves about `var` when it evaluates to
    /// `taken`.
    ///
    /// Only the constant-bounded forms are used: `x < k`, `x > k`, and their
    /// negations. `k` is read from the other operand's proved range, so a
    /// comparison against a computed-but-bounded value refines too.
    fn refine_from_comparison<T: Target>(
        &self,
        comparison: &SsaOp<T>,
        var: SsaVarId,
        taken: bool,
    ) -> Option<ValueRange> {
        match comparison {
            SsaOp::Clt { left, right, .. } => {
                if *left == var {
                    // taken: x < limit -> x <= limit_max - 1
                    // else:  x >= limit -> x >= limit_min
                    let limit = self.get(*right)?;
                    if taken {
                        limit
                            .max()
                            .and_then(|m| m.checked_sub(1))
                            .map(|upper| ValueRange::Interval(IntervalRange::at_most(upper)))
                    } else {
                        limit
                            .min()
                            .map(|lower| ValueRange::Interval(IntervalRange::at_least(lower)))
                    }
                } else if *right == var {
                    // taken: limit < x -> x >= limit_min + 1
                    let limit = self.get(*left)?;
                    if taken {
                        limit
                            .min()
                            .and_then(|m| m.checked_add(1))
                            .map(|lower| ValueRange::Interval(IntervalRange::at_least(lower)))
                    } else {
                        limit
                            .max()
                            .map(|upper| ValueRange::Interval(IntervalRange::at_most(upper)))
                    }
                } else {
                    None
                }
            }
            SsaOp::Cgt { left, right, .. } => {
                if *left == var {
                    let limit = self.get(*right)?;
                    if taken {
                        limit
                            .min()
                            .and_then(|m| m.checked_add(1))
                            .map(|lower| ValueRange::Interval(IntervalRange::at_least(lower)))
                    } else {
                        limit
                            .max()
                            .map(|upper| ValueRange::Interval(IntervalRange::at_most(upper)))
                    }
                } else if *right == var {
                    let limit = self.get(*left)?;
                    if taken {
                        limit
                            .max()
                            .and_then(|m| m.checked_sub(1))
                            .map(|upper| ValueRange::Interval(IntervalRange::at_most(upper)))
                    } else {
                        limit
                            .min()
                            .map(|lower| ValueRange::Interval(IntervalRange::at_least(lower)))
                    }
                } else {
                    None
                }
            }
            // `x == k` on the taken arm pins `x` to `k`; the untaken arm proves
            // only that it differs, which an interval cannot express.
            SsaOp::Ceq { left, right, .. } if taken => {
                let other = if *left == var {
                    *right
                } else if *right == var {
                    *left
                } else {
                    return None;
                };
                self.get(other).cloned()
            }
            _ => None,
        }
    }
}

/// Computes value ranges for `ssa` without modifying it.
///
/// This is the query entry point for consumers that want the numeric facts
/// rather than the rewrites — indirect-branch recovery, bounds reasoning, and
/// tests that need to assert on a range directly.
///
/// # Arguments
///
/// * `ssa` — The SSA function to analyze.
/// * `max_iterations` — Rounds over the function before the analysis gives up.
///
/// # Returns
///
/// `Some(ranges)` when the analysis reached its fixpoint, `None` otherwise. A
/// `None` means nothing was proved, not that nothing is provable — retrying
/// with a larger budget may succeed.
#[must_use]
pub fn analyze<T: Target>(ssa: &SsaFunction<T>, max_iterations: usize) -> Option<ValueRanges> {
    let mut analysis: RangeAnalysis<T> = RangeAnalysis::new(max_iterations);
    let result = analysis.analyze(ssa);
    result.converged.then_some(ValueRanges {
        ranges: result.ranges,
    })
}

/// Run value-range propagation on `ssa`.
///
/// Applies sparse conditional range propagation, then simplifies branches
/// and comparisons based on the computed ranges.
///
/// # Arguments
///
/// * `ssa` — The SSA function to analyze and simplify in place.
/// * `method` — Opaque method reference recorded in emitted events.
/// * `events` — Event sink for [`EventKind::OpaquePredicateRemoved`],
///   [`EventKind::BranchSimplified`], and [`EventKind::ConstantFolded`].
/// * `max_iterations` — Cap on the inner range-propagation worklist loop.
///
/// # Returns
///
/// `true` if any branch or comparison was simplified.
pub fn run<T, L>(
    ssa: &mut SsaFunction<T>,
    method: &T::MethodRef,
    events: &L,
    max_iterations: usize,
) -> bool
where
    T: Target,
    L: EventListener<T> + ?Sized,
{
    let mut analysis: RangeAnalysis<T> = RangeAnalysis::new(max_iterations);
    let result = analysis.analyze(ssa);

    let mut branch_simplifications: Vec<(usize, usize, bool)> = Vec::new();
    let mut comparison_replacements: Vec<(usize, usize, SsaVarId, bool)> = Vec::new();

    for (block_idx, block) in ssa.iter_blocks() {
        if let Some(SsaOp::Branch {
            condition,
            true_target,
            false_target,
        }) = block.control_terminator()
            && let Some(range) = result.get_range(*condition)
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
            if let Some((dest, value)) = try_simplify_comparison(instr.op(), &result) {
                comparison_replacements.push((block_idx, instr_idx, dest, value));
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
            // dominance — no new phi can be required, so this needs a phi-operand
            // prune plus `repair_ssa`, not a full `rebuild_ssa`.
            editor.fold_terminator_pruning_phis(block_idx, SsaOp::Jump { target })?;
            let event = crate::events::Event {
                kind: EventKind::OpaquePredicateRemoved,
                method: Some(method.clone()),
                location: Some(block_idx),
                message: format!(
                    "range analysis: condition always {}",
                    if is_true { "true" } else { "false" }
                ),
                pass: None,
            };
            events.push(event);
            let event = crate::events::Event {
                kind: EventKind::BranchSimplified,
                method: Some(method.clone()),
                location: Some(block_idx),
                message: format!("simplified to unconditional jump to {target}"),
                pass: None,
            };
            events.push(event);
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

            let const_value = if value {
                ConstValue::True
            } else {
                ConstValue::False
            };
            editor.replace_instruction_op(
                block_idx,
                instr_idx,
                SsaOp::Const {
                    dest,
                    value: const_value,
                },
            )?;
            let event = crate::events::Event {
                kind: EventKind::ConstantFolded,
                method: Some(method.clone()),
                location: Some(instr_idx),
                message: format!("range analysis: comparison → {value}"),
                pass: None,
            };
            events.push(event);
            changed = true;
        }
        Ok(())
    });

    if result.is_err() {
        // The session runs under `SsaRollbackPolicy::Never`, so a failed edit or
        // boundary repair leaves the edits applied — the function is mutated and
        // possibly mid-repair. Reporting "unchanged" would make the pass-group
        // transaction skip **both** verification and rollback, keeping damaged
        // IR and keeping it unchecked. Report the change so the transaction
        // verifies this function and rolls it back.
        return true;
    }

    changed
}

fn try_simplify_comparison<T: Target>(
    op: &SsaOp<T>,
    result: &RangeResult,
) -> Option<(SsaVarId, bool)> {
    match op {
        SsaOp::Clt {
            dest, left, right, ..
        } => {
            let left_range = result.get_range(*left)?;
            let right_range = result.get_range(*right)?;
            if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min())
                && l_max < r_min
            {
                return Some((*dest, true));
            }
            if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max())
                && l_min >= r_max
            {
                return Some((*dest, false));
            }
            None
        }
        SsaOp::Cgt {
            dest, left, right, ..
        } => {
            let left_range = result.get_range(*left)?;
            let right_range = result.get_range(*right)?;
            if let (Some(l_min), Some(r_max)) = (left_range.min(), right_range.max())
                && l_min > r_max
            {
                return Some((*dest, true));
            }
            if let (Some(l_max), Some(r_min)) = (left_range.max(), right_range.min())
                && l_max <= r_min
            {
                return Some((*dest, false));
            }
            None
        }
        SsaOp::Ceq { dest, left, right } => {
            let left_range = result.get_range(*left)?;
            let right_range = result.get_range(*right)?;
            if let (Some(l), Some(r)) = (left_range.as_constant(), right_range.as_constant()) {
                return Some((*dest, l == r));
            }
            if !ranges_overlap(left_range, right_range) {
                return Some((*dest, false));
            }
            None
        }
        _ => None,
    }
}

fn ranges_overlap(a: &ValueRange, b: &ValueRange) -> bool {
    if a.is_top() || b.is_top() {
        return true;
    }
    if a.is_bottom() || b.is_bottom() {
        return false;
    }
    match (a.max(), a.min(), b.max(), b.min()) {
        (Some(a_max), Some(a_min), Some(b_max), Some(b_min)) => a_max >= b_min && a_min <= b_max,
        _ => true,
    }
}

/// Sparse range propagation analysis engine.
///
/// Worklist algorithm similar to Wegman-Zadeck SCCP but tracking
/// [`ValueRange`] intervals instead of just constants.
struct RangeAnalysis<T: Target> {
    /// Per-variable computed ranges.
    ranges: HashMap<SsaVarId, ValueRange>,
    /// Set of CFG edges `(from, to)` determined to be executable.
    executable_edges: HashSet<(usize, usize)>,
    /// Blocks that have at least one executable incoming edge.
    executable_blocks: BitSet,
    /// Worklist of SSA variables whose range changed and need re-evaluation.
    ssa_worklist: VecDeque<SsaVarId>,
    /// Worklist of CFG edges to process for first-time execution.
    cfg_worklist: VecDeque<(usize, usize)>,
    /// Number of times each variable's range has been revised, driving the
    /// switch to widening once a value looks loop-carried.
    update_counts: HashMap<SsaVarId, u32>,
    /// Rounds over the function the caller allows before giving up.
    max_iterations: usize,
    _phantom: PhantomData<T>,
}

/// Revisions of one variable's range before the analysis starts widening
/// instead of taking the newly computed range verbatim.
///
/// A value revised more than this is loop-carried: each trip round the loop
/// grows it by one step, so taking the exact range would need as many rounds as
/// the loop has trips. Widening drops the growing bound to infinity, which
/// reaches the fixpoint in a bounded number of steps at the cost of precision on
/// that bound.
const WIDEN_AFTER_REVISIONS: u32 = 3;

impl<T: Target> RangeAnalysis<T> {
    fn new(max_iterations: usize) -> Self {
        Self {
            ranges: HashMap::new(),
            executable_edges: HashSet::new(),
            executable_blocks: BitSet::new(0),
            ssa_worklist: VecDeque::new(),
            cfg_worklist: VecDeque::new(),
            update_counts: HashMap::new(),
            max_iterations,
            _phantom: PhantomData,
        }
    }

    fn analyze(&mut self, ssa: &SsaFunction<T>) -> RangeResult {
        let eh = EhCfg::from_ssa(ssa);
        self.initialize(ssa, &eh);
        let converged = self.propagate(ssa, &eh);
        RangeResult {
            ranges: std::mem::take(&mut self.ranges),
            converged,
        }
    }

    fn initialize<G>(&mut self, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        self.ranges.clear();
        self.executable_edges.clear();
        self.executable_blocks = BitSet::new(ssa.block_count());
        self.ssa_worklist.clear();
        self.cfg_worklist.clear();
        self.update_counts.clear();

        for var in ssa.variables() {
            self.ranges.insert(var.id(), ValueRange::top());
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
    /// the analysis is optimistic — it starts from an empty executable-edge set
    /// and grows ranges as edges are discovered — so its conclusions are only
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
        let budget = self.max_iterations.saturating_mul(work).saturating_add(64);
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

    fn process_edge<G>(&mut self, from: usize, to: usize, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        let first_visit = !self.is_block_executable(to);
        if first_visit {
            self.mark_block_executable(to);
            if let Some(block) = ssa.block(to) {
                self.process_block_definitions(block, ssa);
            }
        }
        if let Some(block) = ssa.block(to) {
            for phi in block.phi_nodes() {
                if phi.operand_from(from).is_some() {
                    let new_range = self.evaluate_phi(phi, to);
                    self.update_range(phi.result(), &new_range);
                }
            }
        }
        if first_visit && let Some(block) = ssa.block(to) {
            self.propagate_outgoing_edges(to, block, cfg);
        }
    }

    fn process_block_definitions(&mut self, block: &SsaBlock<T>, ssa: &SsaFunction<T>) {
        for instr in block.instructions() {
            self.update_instruction_defs(instr, ssa);
        }
    }

    fn process_variable_uses<G>(&mut self, var: SsaVarId, ssa: &SsaFunction<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        if let Some(ssa_var) = ssa.variable(var) {
            for use_site in ssa_var.uses() {
                let block_id = use_site.block;
                if !self.is_block_executable(block_id) {
                    continue;
                }
                if use_site.is_phi_operand {
                    if let Some(block) = ssa.block(block_id)
                        && let Some(phi) = block.phi(use_site.instruction)
                    {
                        let new_range = self.evaluate_phi(phi, block_id);
                        self.update_range(phi.result(), &new_range);
                    }
                } else if let Some(block) = ssa.block(block_id)
                    && let Some(instr) = block.instruction(use_site.instruction)
                {
                    self.update_instruction_defs(instr, ssa);
                    if instr.is_terminator() {
                        self.propagate_outgoing_edges(block_id, block, cfg);
                    }
                }
            }
        }
    }

    fn update_instruction_defs(&mut self, instr: &SsaInstruction<T>, ssa: &SsaFunction<T>) {
        let primary = instr.op().dest();
        let range = self.evaluate_instruction(instr.op(), ssa);
        for def in instr.defs() {
            if Some(def) == primary {
                self.update_range(def, &range);
            } else {
                self.update_range(def, &ValueRange::top());
            }
        }
    }

    fn propagate_outgoing_edges<G>(&mut self, block_id: usize, block: &SsaBlock<T>, cfg: &G)
    where
        G: RootedGraph + Successors,
    {
        match block.control_terminator() {
            Some(SsaOp::Branch {
                condition,
                true_target,
                false_target,
            }) => {
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
                } else if range.is_top() {
                    // unknown
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
            Some(SsaOp::Jump { target }) => {
                self.add_cfg_edge(block_id, *target);
            }
            Some(
                SsaOp::Return { .. }
                | SsaOp::Throw { .. }
                | SsaOp::Rethrow
                | SsaOp::EndFinally
                | SsaOp::EndFilter { .. }
                | SsaOp::InterruptReturn,
            ) => {}
            _ => {
                let node = NodeId::new(block_id);
                for succ in cfg.successors(node) {
                    self.add_cfg_edge(block_id, succ.index());
                }
            }
        }
    }

    fn add_cfg_edge(&mut self, from: usize, to: usize) {
        if !self.executable_edges.contains(&(from, to)) {
            self.cfg_worklist.push_back((from, to));
        }
    }

    /// Returns `true` if `block` is currently marked executable.
    ///
    /// Block indices flow in from terminator targets (`true_target`,
    /// `false_target`, switch targets, `Jump.target`), which the IR permits to
    /// be out of range — a terminator may reference a block that was never
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

    fn evaluate_phi(&self, phi: &PhiNode, block_id: usize) -> ValueRange {
        let mut result = ValueRange::bottom();
        let mut has_executable_operand = false;
        for operand in phi.operands() {
            let pred = operand.predecessor();
            if !self.executable_edges.contains(&(pred, block_id)) {
                continue;
            }
            has_executable_operand = true;
            let op_range = self.get_range(operand.value());
            result = result.join(&op_range);
            if result.is_top() {
                break;
            }
        }
        if !has_executable_operand {
            return ValueRange::top();
        }
        result
    }

    /// Returns the bit width of `var`'s declared type, when the target knows it.
    fn width_of(ssa: &SsaFunction<T>, var: SsaVarId) -> Option<u32> {
        ssa.variable(var).and_then(|v| T::bit_width(v.var_type()))
    }

    /// Clamps `range` to what `width_bits` can represent, falling back to `Top`
    /// when it cannot.
    ///
    /// Interval arithmetic here is done in `i64`, but the values are not: a
    /// 32-bit `add` wraps, so `0x7fff_ffff + 1` is `-0x8000_0000` and not
    /// `0x8000_0000`. A result that escapes the destination's width has
    /// therefore wrapped to somewhere this domain cannot name, and the only
    /// sound answer is no information.
    fn wrap_to_width(range: ValueRange, width_bits: Option<u32>) -> ValueRange {
        let Some(width) = width_bits else {
            // An unknown width cannot be checked, so nothing may be assumed
            // about whether the operation stayed in range.
            return ValueRange::top();
        };
        if width == 0 || width > 64 {
            return ValueRange::top();
        }
        // Signed bounds for `width`: [-2^(w-1), 2^(w-1) - 1].
        let Some(shift) = width.checked_sub(1) else {
            return ValueRange::top();
        };
        let Some(magnitude) = 1i64.checked_shl(shift) else {
            return ValueRange::top();
        };
        let Some(upper) = magnitude.checked_sub(1) else {
            return ValueRange::top();
        };
        let lower = magnitude.saturating_neg();
        match (range.min(), range.max()) {
            (Some(min), Some(max)) if min >= lower && max <= upper => range,
            _ => ValueRange::top(),
        }
    }

    fn evaluate_instruction(&self, op: &SsaOp<T>, ssa: &SsaFunction<T>) -> ValueRange {
        let dest_width = op.dest().and_then(|dest| Self::width_of(ssa, dest));
        match op {
            SsaOp::Const { value, .. } => {
                if let Some(v) = value.as_i64() {
                    ValueRange::constant(v)
                } else {
                    ValueRange::top()
                }
            }
            SsaOp::Copy { src, .. } => self.get_range(*src),
            SsaOp::Add { left, right, .. } => {
                let l = self.get_range(*left);
                let r = self.get_range(*right);
                Self::wrap_to_width(l.add(&r), dest_width)
            }
            SsaOp::Sub { left, right, .. } => {
                let l = self.get_range(*left);
                let r = self.get_range(*right);
                Self::wrap_to_width(l.sub(&r), dest_width)
            }
            SsaOp::Mul { left, right, .. } => {
                let l = self.get_range(*left);
                let r = self.get_range(*right);
                Self::wrap_to_width(l.mul(&r), dest_width)
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
                Self::wrap_to_width(negated, dest_width)
            }
            SsaOp::Shl { value, amount, .. } => {
                let val_range = self.get_range(*value);
                let amt_range = self.get_range(*amount);
                if let Some(amt) = amt_range.as_constant()
                    && (0..64).contains(&amt)
                    && let Ok(shift) = u32::try_from(amt)
                    && let (Some(min), Some(max)) = (val_range.min(), val_range.max())
                    && let (Some(new_min), Some(new_max)) =
                        (min.checked_shl(shift), max.checked_shl(shift))
                    // A left shift is only a multiply while nothing leaves the
                    // top of the value; verify by shifting back.
                    && new_min.checked_shr(shift) == Some(min)
                    && new_max.checked_shr(shift) == Some(max)
                {
                    return Self::wrap_to_width(
                        ValueRange::bounded(new_min.min(new_max), new_min.max(new_max)),
                        dest_width,
                    );
                }
                ValueRange::top()
            }
            SsaOp::Or { left, right, .. } | SsaOp::Xor { left, right, .. } => {
                // For non-negative operands the result cannot exceed the next
                // power of two above either bound, and cannot go negative.
                let l = self.get_range(*left);
                let r = self.get_range(*right);
                match (l.max(), r.max()) {
                    (Some(l_max), Some(r_max))
                        if l.is_always_non_negative()
                            && r.is_always_non_negative()
                            && l_max >= 0
                            && r_max >= 0 =>
                    {
                        let bound = l_max.max(r_max);
                        // Saturate to all-ones at the width of the larger bound.
                        let bits = i64::BITS.saturating_sub(bound.leading_zeros());
                        match 1i64.checked_shl(bits).map(|v| v.saturating_sub(1)) {
                            Some(all_ones) if all_ones >= 0 => ValueRange::bounded(0, all_ones),
                            _ => ValueRange::non_negative(),
                        }
                    }
                    _ => ValueRange::top(),
                }
            }
            SsaOp::And { left, right, .. } => {
                // Delegate to the lattice operation rather than re-deriving the
                // bound here. `mask.max(0)` looks like a clamp but is not: for a
                // negative mask it yields `bounded(0, 0)`, which claims the
                // result is the *constant zero*. `x & -16` clears only the low
                // four bits and can be any value, so the correct answer for a
                // negative mask is Top — which is what `and_constant` returns.
                let r = self.get_range(*right);
                if let Some(mask) = r.as_constant() {
                    r.and_constant(mask)
                } else {
                    let l = self.get_range(*left);
                    if let Some(mask) = l.as_constant() {
                        l.and_constant(mask)
                    } else {
                        ValueRange::top()
                    }
                }
            }
            SsaOp::Shr {
                value,
                amount,
                unsigned,
                ..
            } => {
                let val_range = self.get_range(*value);
                let amt_range = self.get_range(*amount);
                if let Some(amt) = amt_range.as_constant()
                    && (0..64).contains(&amt)
                    && *unsigned
                    && val_range.is_always_non_negative()
                    && let (Some(min), Some(max)) = (val_range.min(), val_range.max())
                {
                    let new_min = min >> amt;
                    let new_max = max >> amt;
                    return ValueRange::bounded(new_min, new_max);
                }
                ValueRange::top()
            }
            SsaOp::Rem { left, right, .. } => {
                let r = self.get_range(*right);
                if let Some(n) = r.as_constant()
                    && n > 0
                {
                    let l = self.get_range(*left);
                    if l.is_always_non_negative() {
                        return ValueRange::bounded(0, n.saturating_sub(1));
                    }
                }
                ValueRange::top()
            }
            SsaOp::ArrayLength { .. } => ValueRange::non_negative(),
            SsaOp::NewArr { .. }
            | SsaOp::NewObj { .. }
            | SsaOp::Box { .. }
            | SsaOp::LoadToken { .. } => ValueRange::top(),
            SsaOp::Ceq { .. } | SsaOp::Clt { .. } | SsaOp::Cgt { .. } => ValueRange::bounded(0, 1),
            _ => ValueRange::top(),
        }
    }

    fn get_range(&self, var: SsaVarId) -> ValueRange {
        self.ranges.get(&var).cloned().unwrap_or_default()
    }

    /// Revises `var`'s range, widening once the value has been revised often
    /// enough to look loop-carried.
    ///
    /// Without widening a counted loop needs one round per trip, so a bounded
    /// run stops mid-ascent holding a range that is too narrow — the unsound
    /// direction. Widening drops the unstable bound to infinity instead, which
    /// converges in a bounded number of steps.
    fn update_range(&mut self, var: SsaVarId, new_range: &ValueRange) {
        let old_range = self.ranges.get(&var).cloned().unwrap_or_default();
        let revisions = self.update_counts.entry(var).or_insert(0);
        *revisions = revisions.saturating_add(1);
        let next_range = if *revisions > WIDEN_AFTER_REVISIONS {
            old_range.widen(new_range)
        } else {
            new_range.clone()
        };
        if next_range != old_range {
            self.ranges.insert(var, next_range);
            self.ssa_worklist.push_back(var);
        }
    }
}

#[derive(Debug)]
struct RangeResult {
    ranges: HashMap<SsaVarId, ValueRange>,
    /// `false` when the analysis stopped before reaching its fixpoint, in which
    /// case no range in `ranges` may be trusted.
    converged: bool,
}

impl RangeResult {
    /// Returns `var`'s range, or `None` when the analysis did not converge and
    /// therefore proved nothing.
    fn get_range(&self, var: SsaVarId) -> Option<&ValueRange> {
        if !self.converged {
            return None;
        }
        self.ranges.get(&var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        events::EventLog,
        ir::{
            block::SsaBlock,
            instruction::SsaInstruction,
            phi::{PhiNode, PhiOperand},
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin},
        },
        testing::{MockTarget, MockType, mock_terminator_at, run_mock_pass_boundary},
    };

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

    fn make_result(entries: Vec<(SsaVarId, ValueRange)>) -> RangeResult {
        RangeResult {
            ranges: entries.into_iter().collect(),
            converged: true,
        }
    }

    /// A result from a run that never reached its fixpoint proves nothing, so
    /// every query answers `None` regardless of what was recorded.
    #[test]
    fn unconverged_result_yields_no_ranges() {
        let var = SsaVarId::from_index(0);
        let result = RangeResult {
            ranges: [(var, ValueRange::constant(5))].into_iter().collect(),
            converged: false,
        };
        assert_eq!(result.get_range(var), None);
    }

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
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, true)));
    }

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
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, true)));
    }

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
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, false)));
    }

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
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, true)));
    }

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
        block.add_instruction(instr(SsaOp::Return { value: Some(v1) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "copy range propagation", |ssa| {
            run(ssa, &method, &log, 20)
        });
        assert!(
            !changed,
            "copy-only range propagation should not rewrite SSA"
        );
    }

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
        block.add_instruction(instr(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "add range propagation", |ssa| {
            run(ssa, &method, &log, 20)
        });
        assert!(
            !changed,
            "range propagation through add should not rewrite SSA"
        );
    }

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
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "constant branch range folding", |ssa| {
            run(ssa, &method, &log, 20)
        });
        // Branch on constant 1 should be simplified to Jump 1
        assert!(changed, "constant branch should be simplified");
        assert!(matches!(
            mock_terminator_at(&ssa, 0),
            SsaOp::Jump { target: 1 }
        ));
    }

    /// `x & -16` clears the low four bits and leaves everything else alone, so
    /// its value is unconstrained. Deriving `[0, 0]` from the mask — which
    /// `mask.max(0)` does for any negative mask — claims the result is the
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
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);

        let mut b2 = SsaBlock::new(2);
        b2.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        run(&mut ssa, &method, &log, 20);

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
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: Some(v0) }));
        ssa.add_block(block);
        ssa.recompute_uses();
        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "single-block range pass", |ssa| {
            run(ssa, &method, &log, 20)
        });
        assert!(!changed);
    }

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
        }));
        block.add_instruction(instr(SsaOp::Return { value: Some(v2) }));
        ssa.add_block(block);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "comparison range folding", |ssa| {
            run(ssa, &method, &log, 20)
        });
        // Should fold Clt to Const(true)
        assert!(changed, "range-known comparison should fold");
        assert!(log.has(EventKind::ConstantFolded));
    }

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
        b2.add_instruction(instr(SsaOp::Return {
            value: Some(phi_var),
        }));
        ssa.add_block(b2);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "phi range propagation", |ssa| {
            run(ssa, &method, &log, 20)
        });
        assert!(changed, "constant branch before phi should simplify");
    }

    #[test]
    fn out_of_range_branch_targets_do_not_panic() {
        // A terminator may reference a block that was never recovered (the IR
        // permits dangling successors and the verifier tolerates them). The
        // `executable_blocks` bitset is sized to `block_count`, so an
        // out-of-range target index must be treated as unreachable rather than
        // reaching the asserting `BitSet::contains`. Here the condition is an unconstrained argument (range stays
        // `top`), so both the in-range and the out-of-range edge are explored.
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
            // Block 99 does not exist — only blocks 0 and 1 are present.
            false_target: 99,
        }));
        ssa.add_block(b0);

        let mut b1 = SsaBlock::new(1);
        b1.add_instruction(instr(SsaOp::Return { value: None }));
        ssa.add_block(b1);
        ssa.recompute_uses();

        let log: EventLog<MockTarget> = EventLog::new();
        let method = 0u32;
        let changed = run_mock_pass_boundary(&mut ssa, "out-of-range branch target", |ssa| {
            run(ssa, &method, &log, 20)
        });
        // Unconstrained condition → nothing folds, and crucially no panic.
        assert!(!changed, "unconstrained branch must not be simplified");
    }

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
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, false)));
    }

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
        };
        assert_eq!(try_simplify_comparison(&op, &result), Some((dest, false)));
    }
}
