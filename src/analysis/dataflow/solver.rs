//! Worklist-based data flow solver.
//!
//! This module provides the iterative solver that computes fixpoints for
//! data flow analyses. It uses a worklist algorithm with traversal order
//! optimized for the analysis direction.
//!
//! # Algorithm
//!
//! The solver iterates until a fixpoint is reached:
//!
//! **Initialization**:
//! 1. All blocks start with the analysis's `initial()` value
//! 2. Set `boundary()` value at entry (forward) or exits (backward)
//! 3. Add all blocks to the worklist in traversal order (reverse postorder
//!    for forward, postorder for backward)
//!
//! **Iteration**:
//! 1. Pop a block from the worklist (deduplicated via `in_worklist` flag)
//! 2. Compute the input/output by meeting lattice values from all
//!    predecessors (forward) or successors (backward); a backward analysis
//!    also meets the handlers a throw from inside a protected block reaches
//! 3. Meet the boundary value into the entry (forward) or every exit
//!    (backward)
//! 4. Apply the transfer function to compute the output/input
//! 5. If the result changed, add affected adjacent blocks
//!    (successors for forward, predecessors for backward) to the worklist
//!
//! **Finalization**:
//! 1. Call the analysis's `finalize()` hook for any post-processing
//! 2. Return `AnalysisResults` with per-block in/out states
//!
//! # Complexity
//!
//! On reducible CFGs, converges in O(d) iterations where d is the loop
//! nesting depth (typically small). Total work: O(n * d * h) where n is
//! the block count and h is the lattice height (maximum chain length from
//! top to bottom). Each iteration processes each block at most once via
//! the worklist deduplication flag.

use std::{cmp::Reverse, collections::BinaryHeap, marker::PhantomData};

use crate::{
    analysis::{
        dataflow::{
            framework::{AnalysisResults, DataFlowAnalysis, DataFlowCfg, Direction},
            lattice::MeetSemiLattice,
        },
        exceptions::FunctionRoots,
    },
    bitset::BitSet,
    graph::NodeId,
    ir::function::SsaFunction,
    target::Target,
};

/// Worklist-based data flow solver.
///
/// This solver computes fixpoints for data flow analyses using an iterative
/// worklist algorithm. It supports both forward and backward analyses.
///
/// # Usage
///
/// ```rust
/// use analyssa::{
///     analysis::{
///         SsaCfg,
///         dataflow::{DataFlowSolver, ReachingDefinitions},
///     },
///     ir::SsaVarId,
///     testing,
/// };
///
/// let ssa = testing::diamond_phi_fixture();
/// let graph = SsaCfg::from_ssa(&ssa);
///
/// let analysis = ReachingDefinitions::new(&ssa);
/// let solver = DataFlowSolver::new(analysis);
/// let results = solver.solve(&ssa, &graph);
///
/// // Access results: the branch condition defined in block 0 reaches block 1.
/// let condition = SsaVarId::from_index(0);
/// let in_state = results.in_state(1).unwrap();
/// assert!(in_state.definitions().any(|var| var == condition));
/// ```
pub struct DataFlowSolver<T: Target, A: DataFlowAnalysis<T>> {
    /// The analysis being solved.
    analysis: A,
    /// Input state for each block.
    in_states: Vec<A::Lattice>,
    /// Output state for each block.
    out_states: Vec<A::Lattice>,
    /// Worklist of blocks to process, prioritized by traversal order so blocks
    /// are visited in (reverse) postorder. `Reverse((priority, block))` turns
    /// the max-heap into a min-heap on the priority, which reduces fixpoint
    /// iterations on loops compared with a plain FIFO.
    worklist: BinaryHeap<Reverse<(usize, usize)>>,
    /// Whether each block is currently in the worklist (for deduplication).
    in_worklist: Vec<bool>,
    /// Times each block's transfer function has run, handed to
    /// [`DataFlowAnalysis::widen`] so an unbounded-height lattice can stay
    /// exact for the first few passes and widen after.
    visits: Vec<usize>,
    /// Per-block traversal priority (position in RPO for forward analyses,
    /// postorder for backward); smaller is processed first.
    order_priority: Vec<usize>,
    /// Exit blocks, precomputed once per solve for O(1) membership in the
    /// backward solver hot loop (avoids re-allocating `cfg.exits()` per visit).
    exit_blocks: BitSet,
    /// The backward boundary value, which every exit block's out-state starts
    /// from. `None` for a forward analysis.
    boundary: Option<A::Lattice>,
    /// The function's exception entry points, for a backward analysis of a
    /// function that has any.
    ///
    /// A block inside a protected region can leave through a throw from any of
    /// its instructions, so its out-state also meets the in-states of the
    /// handlers that throw reaches ([`FunctionRoots::exceptional_successors`]).
    /// The graph's own exceptional edge runs from the region's guard only,
    /// which is what dominance needs but not what a backward fact needs: a
    /// value the handler reads is live at every block that can throw into it.
    roots: Option<FunctionRoots>,
    /// Per handler root, the blocks whose out-state meets its in-state, so a
    /// change to the handler re-enqueues them.
    exceptional_predecessors: Vec<Vec<usize>>,
    /// Number of iterations performed.
    iterations: usize,
    _phantom: PhantomData<T>,
}

impl<T: Target, A: DataFlowAnalysis<T>> DataFlowSolver<T, A> {
    /// Creates a new solver for the given analysis.
    #[must_use]
    pub fn new(analysis: A) -> Self {
        Self {
            analysis,
            in_states: Vec::new(),
            out_states: Vec::new(),
            worklist: BinaryHeap::new(),
            in_worklist: Vec::new(),
            visits: Vec::new(),
            order_priority: Vec::new(),
            exit_blocks: BitSet::new(0),
            boundary: None,
            roots: None,
            exceptional_predecessors: Vec::new(),
            iterations: 0,
            _phantom: PhantomData,
        }
    }

    /// Solves the data flow analysis to a fixpoint.
    ///
    /// Seeds every block with the analysis's initial value, stamps the boundary
    /// value onto the entry (forward) or exit blocks (backward), then drains a
    /// priority worklist until no block's state changes. There is no iteration
    /// cap: termination rests on the lattice having finite height, or on
    /// [`DataFlowAnalysis::widen`] forcing convergence for an unbounded one. A
    /// function with no blocks short-circuits to empty results, since the state
    /// vectors would otherwise be indexed by a block that does not exist.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function whose blocks and variables the analysis reads.
    /// * `cfg` - The control flow graph supplying entry, exits, traversal order,
    ///   and the predecessor/successor edges the fixpoint propagates along.
    ///
    /// # Returns
    ///
    /// Per-block input and output states at the fixpoint.
    pub fn solve<C: DataFlowCfg>(
        mut self,
        ssa: &SsaFunction<T>,
        cfg: &C,
    ) -> AnalysisResults<A::Lattice>
    where
        A::Lattice: Clone,
    {
        let num_blocks = ssa.block_count();
        if num_blocks == 0 {
            return AnalysisResults::new(Vec::new(), Vec::new());
        }

        // Initialize states
        self.initialize(ssa, cfg);

        // Main iteration loop
        self.iterate(ssa, cfg);

        // Finalize
        self.analysis
            .finalize(&self.in_states, &self.out_states, ssa);

        AnalysisResults::new(self.in_states, self.out_states)
    }

    /// Returns the number of iterations performed.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Initializes the solver state before the fixpoint loop.
    ///
    /// Fills the in/out state vectors with the analysis's `initial()` value, then
    /// overwrites the boundary: forward analyses stamp `boundary()` onto the entry
    /// block's in-state, backward analyses onto every exit block's out-state. The
    /// exit set is cached in a [`BitSet`] here so the backward hot loop can test
    /// membership without rebuilding `cfg.exits()` on each visit, and a backward
    /// analysis of a function with exception handlers records which blocks each
    /// handler is an exceptional successor of. Finally it
    /// assigns each block a traversal priority - its position in reverse postorder
    /// (forward) or postorder (backward) - and seeds the priority worklist with
    /// every block, so the first sweep visits blocks in the order that minimizes
    /// re-visits on reducible CFGs.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function; supplies the block count and initial/boundary values.
    /// * `cfg` - The control flow graph; supplies entry, exits, and traversal order.
    fn initialize<C: DataFlowCfg>(&mut self, ssa: &SsaFunction<T>, cfg: &C)
    where
        A::Lattice: Clone,
    {
        let num_blocks = ssa.block_count();
        let initial = self.analysis.initial(ssa);
        let boundary = self.analysis.boundary(ssa);

        // Initialize all blocks with the initial value
        self.in_states = vec![initial.clone(); num_blocks];
        self.out_states = vec![initial; num_blocks];
        self.in_worklist = vec![false; num_blocks];
        self.visits = vec![0; num_blocks];

        // Set boundary conditions based on direction
        match A::DIRECTION {
            Direction::Forward => {
                // Entry block gets boundary value
                let entry = cfg.entry().index();
                if let Some(slot) = self.in_states.get_mut(entry) {
                    *slot = boundary;
                }
            }
            Direction::Backward => {
                // Exit blocks get boundary value. Cache them in a BitSet so the
                // backward fixpoint loop can test membership without rebuilding
                // (and reallocating) `cfg.exits()` on every block visit.
                let mut exit_blocks = BitSet::new(num_blocks);
                for exit in cfg.exits() {
                    let idx = exit.index();
                    exit_blocks.insert(idx);
                    if let Some(slot) = self.out_states.get_mut(idx) {
                        *slot = boundary.clone();
                    }
                }
                self.exit_blocks = exit_blocks;
                self.boundary = Some(boundary);

                let roots = FunctionRoots::of(ssa);
                self.exceptional_predecessors = vec![Vec::new(); num_blocks];
                if roots.has_flow_roots() {
                    for block in 0..num_blocks {
                        for &handler in roots.exceptional_successors(block) {
                            if let Some(preds) = self.exceptional_predecessors.get_mut(handler) {
                                preds.push(block);
                            }
                        }
                    }
                    self.roots = Some(roots);
                }
            }
        }

        // Assign each block a traversal priority (position in RPO for forward,
        // postorder for backward) and seed the priority worklist with it.
        let order = match A::DIRECTION {
            Direction::Forward => cfg.reverse_postorder(),
            Direction::Backward => cfg.postorder(),
        };

        self.order_priority = vec![usize::MAX; num_blocks];
        for (pos, node) in order.iter().enumerate() {
            if let Some(slot) = self.order_priority.get_mut(node.index()) {
                *slot = pos;
            }
        }

        for node in &order {
            let idx = node.index();
            if let Some(slot) = self.in_worklist.get_mut(idx) {
                let prio = self.order_priority.get(idx).copied().unwrap_or(usize::MAX);
                self.worklist.push(Reverse((prio, idx)));
                *slot = true;
            }
        }
    }

    /// Drains the worklist, processing one block per pop until the fixpoint.
    ///
    /// Popping the min-priority entry clears the block's `in_worklist` flag before
    /// processing, so the block can legitimately be re-enqueued if a later change
    /// affects it. Each pop dispatches on direction and, whenever the block's
    /// state changed, re-enqueues the blocks that depend on it (successors for
    /// forward, predecessors for backward). The loop ends when the worklist empties,
    /// which - because a block is only ever re-added after an actual state change -
    /// coincides with the fixpoint.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function passed through to the transfer function.
    /// * `cfg` - The control flow graph supplying the edges to propagate along.
    fn iterate<C: DataFlowCfg>(&mut self, ssa: &SsaFunction<T>, cfg: &C)
    where
        A::Lattice: Clone,
    {
        while let Some(Reverse((_, block_idx))) = self.worklist.pop() {
            if let Some(slot) = self.in_worklist.get_mut(block_idx) {
                *slot = false;
            }
            self.iterations = self.iterations.saturating_add(1);

            let changed = match A::DIRECTION {
                Direction::Forward => self.process_forward(block_idx, ssa, cfg),
                Direction::Backward => self.process_backward(block_idx, ssa, cfg),
            };

            if changed {
                // Add affected blocks to worklist
                self.add_affected_to_worklist(block_idx, cfg);
            }
        }
    }

    /// Processes a block in the forward direction.
    ///
    /// Recomputes the block's in-state as the meet of every predecessor's
    /// out-state, folding predecessors in place so the whole join costs one clone
    /// rather than one per edge, then applies the transfer function and widens the
    /// result against the block's previous out-state. A block with no predecessors
    /// keeps its seeded in-state, and the entry block meets its predecessors (if
    /// any) into the boundary value set in [`Self::initialize`], so the boundary
    /// survives without hiding a loop through the entry. The change flag reports whether
    /// the out-state actually moved, which is what tells [`Self::iterate`] whether
    /// to re-enqueue successors.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Index of the block to process.
    /// * `ssa` - The SSA function, source of the block and its instructions.
    /// * `cfg` - The control flow graph, source of predecessor edges.
    ///
    /// # Returns
    ///
    /// `true` if the block's output state changed.
    fn process_forward<C: DataFlowCfg>(
        &mut self,
        block_idx: usize,
        ssa: &SsaFunction<T>,
        cfg: &C,
    ) -> bool
    where
        A::Lattice: Clone,
    {
        // Compute input by meeting all predecessor outputs
        let node = NodeId::new(block_idx);
        let Some(current_in) = self.in_states.get(block_idx).cloned() else {
            return false;
        };
        // Meet all predecessor outputs. The accumulator is seeded once and the
        // rest are folded in place, so the whole fold costs one clone rather
        // than one per predecessor.
        //
        // The entry block starts from its boundary value and meets its
        // predecessors into it. A well-formed function gives the entry none
        // (the verifier's `EntryHasPredecessors`), so this is the boundary
        // alone; one that does not still gets its loop's state instead of
        // having it silently replaced by the boundary. Meeting into the
        // previous in-state is exact because the boundary is where it started
        // and meet is idempotent.
        let is_entry = node == cfg.entry();
        let mut result: Option<A::Lattice> = is_entry.then(|| current_in.clone());
        for pred in cfg.predecessors(node) {
            let Some(pred_out) = self.out_states.get(pred.index()) else {
                continue;
            };
            match result.as_mut() {
                None => result = Some(pred_out.clone()),
                Some(acc) => acc.meet_into(pred_out),
            }
        }
        // A block with no predecessors (the entry, or an unreachable block)
        // keeps its current in-state.
        let input = result.unwrap_or_else(|| current_in.clone());

        if let Some(slot) = self.in_states.get_mut(block_idx) {
            *slot = input.clone();
        }

        // Apply transfer function
        let Some(block) = ssa.block(block_idx) else {
            return false;
        };
        let output = self.analysis.transfer(block_idx, block, &input, ssa);

        // Check if output changed
        let visit = self.bump_visit(block_idx);
        let Some(out_slot) = self.out_states.get_mut(block_idx) else {
            return false;
        };
        let output = self.analysis.widen(block_idx, out_slot, output, visit);
        let changed = output != *out_slot;
        *out_slot = output;

        changed
    }

    /// Records another transfer of `block_idx` and returns the running count.
    fn bump_visit(&mut self, block_idx: usize) -> usize {
        let Some(slot) = self.visits.get_mut(block_idx) else {
            return 0;
        };
        *slot = slot.saturating_add(1);
        *slot
    }

    /// Processes a block in the backward direction.
    ///
    /// Mirror image of [`Self::process_forward`]: recomputes the block's out-state
    /// as the meet of every successor's in-state and every exceptional
    /// successor's (the handlers a throw from inside the block reaches), applies
    /// the transfer function to obtain the new in-state, and widens it against
    /// the previous one. An exit block (looked up in the cached `exit_blocks`
    /// set) meets those into the boundary value, so a `Return` inside a
    /// protected region still sees what its handler needs; a block with no
    /// successor of either kind keeps its current out-state. The change flag
    /// reports whether the in-state moved, which tells [`Self::iterate`] whether
    /// to re-enqueue predecessors.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Index of the block to process.
    /// * `ssa` - The SSA function, source of the block and its instructions.
    /// * `cfg` - The control flow graph, source of successor edges.
    ///
    /// # Returns
    ///
    /// `true` if the block's input state changed.
    fn process_backward<C: DataFlowCfg>(
        &mut self,
        block_idx: usize,
        ssa: &SsaFunction<T>,
        cfg: &C,
    ) -> bool
    where
        A::Lattice: Clone,
    {
        let node = NodeId::new(block_idx);
        // An exit starts from the boundary; any other block from its first
        // successor. The rest are folded in place, so the whole meet costs one
        // clone rather than one per edge.
        let mut result: Option<A::Lattice> = if self.exit_blocks.contains_checked(block_idx) {
            self.boundary.clone()
        } else {
            None
        };
        let handlers = self
            .roots
            .as_ref()
            .map_or(&[][..], |roots| roots.exceptional_successors(block_idx));
        let successors = cfg.successors(node).map(NodeId::index);
        for succ in successors.chain(handlers.iter().copied()) {
            let Some(succ_in) = self.in_states.get(succ) else {
                continue;
            };
            match result.as_mut() {
                None => result = Some(succ_in.clone()),
                Some(acc) => acc.meet_into(succ_in),
            }
        }
        // A dead end (no successor of either kind, not an exit) keeps its
        // current out-state.
        let Some(output) = result.or_else(|| self.out_states.get(block_idx).cloned()) else {
            return false;
        };

        if let Some(slot) = self.out_states.get_mut(block_idx) {
            *slot = output.clone();
        }

        // Apply transfer function (backward: input = transfer(output))
        let Some(block) = ssa.block(block_idx) else {
            return false;
        };
        let input = self.analysis.transfer(block_idx, block, &output, ssa);

        // Check if input changed
        let visit = self.bump_visit(block_idx);
        let Some(in_slot) = self.in_states.get_mut(block_idx) else {
            return false;
        };
        let input = self.analysis.widen(block_idx, in_slot, input, visit);
        let changed = input != *in_slot;
        *in_slot = input;

        changed
    }

    /// Re-enqueues the blocks whose states depend on a block that just changed.
    ///
    /// When `block_idx`'s state moves, only its dataflow neighbours can be
    /// affected: successors for a forward analysis, predecessors (including the
    /// blocks that can throw into it) for a backward one. Each is pushed at its precomputed traversal priority, and the
    /// `in_worklist` flag guards against enqueuing a block that is already pending,
    /// keeping the heap free of duplicates.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block whose state just changed.
    /// * `cfg` - The control flow graph supplying the neighbour edges.
    fn add_affected_to_worklist<C: DataFlowCfg>(&mut self, block_idx: usize, cfg: &C) {
        let node = NodeId::new(block_idx);

        let priority = &self.order_priority;
        let enqueue =
            |idx: usize, list: &mut Vec<bool>, work: &mut BinaryHeap<Reverse<(usize, usize)>>| {
                if let Some(slot) = list.get_mut(idx)
                    && !*slot
                {
                    let prio = priority.get(idx).copied().unwrap_or(usize::MAX);
                    work.push(Reverse((prio, idx)));
                    *slot = true;
                }
            };

        match A::DIRECTION {
            Direction::Forward => {
                // Forward: successors are affected
                for succ in cfg.successors(node) {
                    enqueue(succ.index(), &mut self.in_worklist, &mut self.worklist);
                }
            }
            Direction::Backward => {
                // Backward: predecessors are affected, and so is every block
                // that can throw into this one.
                for pred in cfg.predecessors(node) {
                    enqueue(pred.index(), &mut self.in_worklist, &mut self.worklist);
                }
                let thrower_lists = self.exceptional_predecessors.get(block_idx);
                for &pred in thrower_lists.map_or(&[][..], Vec::as_slice) {
                    enqueue(pred, &mut self.in_worklist, &mut self.worklist);
                }
            }
        }
    }
}
