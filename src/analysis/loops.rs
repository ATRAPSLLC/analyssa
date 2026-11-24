//! Extended loop analysis infrastructure.
//!
//! This module provides comprehensive loop analysis beyond basic natural loop detection,
//! including preheader identification, latch detection, exit analysis, loop
//! classification, nesting relationship computation, and induction variable detection.
//!
//! # Loop Structure
//!
//! A well-formed loop has the following structure:
//!
//! ```text
//!     [preheader]     <- Single entry predecessor (optional, may need insertion)
//!          |
//!          v
//!     [header] <------+  <- Single entry point, dominates all loop nodes
//!          |          |
//!          v          |
//!     [body ...]      |  <- Loop body nodes
//!          |          |
//!          v          |
//!     [latch] --------+  <- Back edge source(s)
//!          |
//!          v
//!     [exit ...]         <- Exit blocks (outside loop, have predecessor in loop)
//! ```
//!
//! # Algorithm: `detect_loops`
//!
//! The loop detection algorithm works on any graph implementing the required
//! traits (`GraphBase`, `Successors`, `Predecessors`), enabling loop detection
//! on CIL CFGs, SSA CFGs, x86 CFGs, and other graph structures.
//!
//! **Phase 1: Back Edge Detection** (O(E * D) where D is dominator query cost):
//! For each edge `(n -> s)`, if `s` dominates `n`, then `(n, s)` is a back edge
//! and `s` is a loop header. This identifies all natural loops.
//!
//! **Phase 2: Loop Body Expansion** (O(L * B)):
//! For each back edge `(latch, header)`, expand the loop body starting from
//! the latch, adding predecessors until the header is reached. The body includes
//! all nodes that can reach the latch without going through the header.
//!
//! **Phase 3: Preheader Identification** (O(N)):
//! Among the header's predecessors, find the unique predecessor that is NOT
//! in the loop body. If exactly one such predecessor exists, it's the preheader.
//!
//! **Phase 4: Exit Analysis** (O(B * S)):
//! For each block in the loop body, check its successors. Any successor outside
//! the body is an exit edge.
//!
//! **Phase 5: Loop Classification** (O(E)):
//! Classify the loop type based on exit edge locations:
//! - All exits from header = PreTested (while loop)
//! - All exits from latch = PostTested (do-while loop)
//! - No exits = Infinite
//! - Mixed exits or multiple latches = Complex
//!
//! **Phase 6: Nesting Computation** (O(L^2)):
//! For each pair of loops, determine containment by checking if one loop's
//! header is in another loop's body. Depths are computed by walking parent chains.
//!
//! # Induction Variable Detection
//!
//! Induction variables are found by analyzing phi nodes at loop headers:
//! - One operand comes from outside the loop (initial value)
//! - One operand comes from inside the loop (updated value)
//!
//! The update instruction is analyzed to determine the update kind (Add, Sub, Mul)
//! and stride (constant increment/decrement per iteration).
//!
//! # Loop Types
//!
//! | Type | Description | Example |
//! |------|-------------|---------|
//! | PreTested | Condition at header | `while(cond) { body }` |
//! | PostTested | Condition at latch | `do { body } while(cond)` |
//! | Infinite | No exit edges | `while(true) { body }` |
//! | Complex | Multiple latches or irregular | Irreducible control flow |
//!
//! # Canonical Form
//!
//! Canonical loops have:
//! - Single preheader (unique non-loop predecessor to header)
//! - Single latch (unique back edge to header)
//!
//! # Loop Forest
//!
//! [`LoopForest`] provides:
//! - O(1) innermost loop lookup for any block
//! - Iteration by nesting depth (ascending or descending)
//! - Loop containment queries via `loop_depth` and `is_in_loop`
//!
//! # Complexity
//!
//! Full detection: O(B^2 + E + L^2) worst case where B is block count,
//! E is edge count, and L is loop count. O(B log B) typical for reducible CFGs.
//!
//! ```rust
//! use analyssa::{
//!     analysis::{SsaCfg, detect_loops},
//!     graph::{algorithms::compute_dominators, NodeId, RootedGraph},
//!     testing,
//! };
//!
//! // Works with any graph implementing the traits
//! let ssa = testing::loop_counter_fixture();
//! let graph = SsaCfg::from_ssa(&ssa);
//!
//! let dominators = compute_dominators(&graph, graph.entry());
//! let forest = detect_loops(&graph, &dominators);
//!
//! // The fixture has a single self-loop headed by block 1.
//! assert_eq!(forest.loops().len(), 1);
//! assert_eq!(forest.loops()[0].header, NodeId::new(1));
//!
//! // Containment queries: block 1 is in the loop, the entry block is not.
//! assert!(forest.is_in_loop(NodeId::new(1)));
//! assert_eq!(forest.loop_depth(NodeId::new(1)), 1);
//! assert!(!forest.is_in_loop(NodeId::new(0)));
//! ```

use std::{
    cmp::Reverse,
    collections::{BTreeMap, btree_map::Entry},
};

use rustc_hash::FxHashMap;

use crate::{
    bitset::BitSet,
    graph::{GraphBase, NodeId, Predecessors, Successors, algorithms::DominatorTree},
    ir::{block::SsaBlock, function::SsaFunction, ops::SsaOp, variable::SsaVarId},
    target::Target,
};

/// Classification of loop types based on structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopType {
    /// Pre-tested loop (while): exit condition at header.
    /// ```text
    /// while (cond) { body }
    /// ```
    PreTested,

    /// Post-tested loop (do-while): exit condition at latch.
    /// ```text
    /// do { body } while (cond)
    /// ```
    PostTested,

    /// Infinite loop: no exit edges from loop body.
    /// ```text
    /// while (true) { body }
    /// ```
    Infinite,

    /// Complex loop: multiple latches, irregular exits, or irreducible.
    Complex,
}

/// Exit edge information for a loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopExit {
    /// The block inside the loop that branches out.
    pub exiting_block: NodeId,
    /// The block outside the loop that is the exit target.
    pub exit_block: NodeId,
}

/// Classification of induction variable update operations.
///
/// Describes how an induction variable's value changes each loop iteration.
/// The kind is determined by analyzing the instruction that computes the
/// updated value (typically in the loop latch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InductionUpdateKind {
    /// Induction variable is incremented: `i = i + stride`.
    Add,
    /// Induction variable is decremented: `i = i - stride`.
    Sub,
    /// Induction variable is scaled: `i = i * stride`.
    Mul,
    /// Unknown or complex update pattern that cannot be classified.
    /// Examples: bitwise operations, division, function calls, or phi-based updates.
    Unknown,
}

/// Represents an induction variable in a loop.
///
/// An induction variable is a variable whose value changes by a fixed amount
/// on each iteration of a loop. Classic examples include loop counters (`i++`).
///
/// # Structure
///
/// An induction variable has:
/// - A phi node at the loop header that merges the initial and updated values
/// - An initial value from outside the loop (preheader)
/// - An updated value computed inside the loop (typically in the latch)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InductionVar {
    /// The phi node result variable at the loop header.
    pub phi_result: SsaVarId,
    /// The initial value (from preheader or outside loop).
    pub init_value: SsaVarId,
    /// The block providing the initial value.
    pub init_block: NodeId,
    /// The updated value (from inside loop, typically latch).
    pub update_value: SsaVarId,
    /// The block providing the updated value.
    pub update_block: NodeId,
    /// The type of update operation.
    pub update_kind: InductionUpdateKind,
    /// The stride (constant value added/subtracted per iteration), if known.
    pub stride: Option<i64>,
}

/// Comprehensive loop information.
///
/// This extends `NaturalLoop` with additional structural information needed
/// for loop canonicalization and optimization.
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// The header block (single entry point, dominates all loop nodes).
    pub header: NodeId,

    /// All blocks in the loop body (including header).
    pub body: BitSet,

    /// Back edge sources (blocks that jump to the header from within the loop).
    pub latches: Vec<NodeId>,

    /// Preheader block if one exists (single non-loop predecessor of header).
    /// `None` if header has multiple non-loop predecessors or none.
    pub preheader: Option<NodeId>,

    /// Exit edges from the loop.
    pub exits: Vec<LoopExit>,

    /// Loop nesting depth (0 = outermost).
    pub depth: usize,

    /// Classification of the loop type.
    pub loop_type: LoopType,

    /// Parent loop header, if this loop is nested.
    pub parent: Option<NodeId>,

    /// Immediate child loop headers.
    pub children: Vec<NodeId>,
}

impl LoopInfo {
    /// Creates a new `LoopInfo` with the given header.
    #[must_use]
    pub fn new(header: NodeId, node_count: usize) -> Self {
        let mut body = BitSet::new(node_count);
        body.insert(header.index());
        Self {
            header,
            body,
            latches: Vec::new(),
            preheader: None,
            exits: Vec::new(),
            depth: 0,
            loop_type: LoopType::Complex,
            parent: None,
            children: Vec::new(),
        }
    }

    /// Returns true if this loop contains the given block.
    #[must_use]
    pub fn contains(&self, node: NodeId) -> bool {
        self.body.contains_checked(node.index())
    }

    /// Returns the number of blocks in the loop.
    #[must_use]
    pub fn size(&self) -> usize {
        self.body.count()
    }

    /// Returns true if the loop has a single latch (canonical form).
    #[must_use]
    pub fn has_single_latch(&self) -> bool {
        self.latches.len() == 1
    }

    /// Returns the single latch if there is exactly one.
    #[must_use]
    pub fn single_latch(&self) -> Option<NodeId> {
        if self.latches.len() == 1 {
            self.latches.first().copied()
        } else {
            None
        }
    }

    /// Returns true if the loop has a preheader (canonical form).
    #[must_use]
    pub fn has_preheader(&self) -> bool {
        self.preheader.is_some()
    }

    /// Returns true if the loop is in canonical form.
    ///
    /// A canonical loop has:
    /// - A single preheader
    /// - A single latch
    #[must_use]
    pub fn is_canonical(&self) -> bool {
        self.has_preheader() && self.has_single_latch()
    }

    /// Returns true if this is an innermost loop (no children).
    #[must_use]
    pub fn is_innermost(&self) -> bool {
        self.children.is_empty()
    }

    /// Returns true if this is an outermost loop (no parent).
    #[must_use]
    pub fn is_outermost(&self) -> bool {
        self.parent.is_none()
    }

    /// Returns all exit blocks (blocks outside loop reachable from inside).
    pub fn exit_blocks(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.exits.iter().map(|e| e.exit_block)
    }

    /// Returns all exiting blocks (blocks inside loop that branch out).
    pub fn exiting_blocks(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.exits.iter().map(|e| e.exiting_block)
    }

    /// Returns the number of exits from this loop.
    #[must_use]
    pub fn exit_count(&self) -> usize {
        self.exits.len()
    }

    /// Returns true if the header is also an exiting block.
    ///
    /// This indicates a pre-tested loop (condition at entry).
    #[must_use]
    pub fn header_is_exiting(&self) -> bool {
        self.exits.iter().any(|e| e.exiting_block == self.header)
    }

    /// Returns true if a latch is also an exiting block.
    ///
    /// This indicates a post-tested loop (condition at end).
    #[must_use]
    pub fn latch_is_exiting(&self) -> bool {
        self.exits
            .iter()
            .any(|e| self.latches.contains(&e.exiting_block))
    }

    /// Finds the condition block inside the loop body.
    ///
    /// For control-flow flattened code (e.g., ConfuserEx), the actual loop
    /// condition is often inside a case block rather than at the dispatcher
    /// header. This method searches the loop body for a block ending in a
    /// two-way conditional terminator - `Branch`, `BranchCmp` or `BranchFlags`
    /// (see [`SsaOp::two_way_targets`]).
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function whose block terminators are inspected.
    ///
    /// # Returns
    ///
    /// - `Some(NodeId)` - The first block found with a conditional branch
    /// - `None` - No conditional branch found in the loop body
    #[must_use]
    pub fn find_condition_in_body<T: Target>(&self, ssa: &SsaFunction<T>) -> Option<NodeId> {
        self.body
            .iter()
            .find(|&block_idx| ends_in_condition(ssa, block_idx))
            .map(NodeId::new)
    }

    /// Finds all conditional blocks within the loop body.
    ///
    /// Unlike `find_condition_in_body`, this returns all blocks with
    /// conditional branches, useful for complex loops with multiple exit points.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function whose block terminators are inspected.
    ///
    /// # Returns
    ///
    /// Every body block terminated by a two-way conditional terminator
    /// (`Branch`, `BranchCmp` or `BranchFlags`), in body-iteration order.
    #[must_use]
    pub fn find_all_conditions_in_body<T: Target>(&self, ssa: &SsaFunction<T>) -> Vec<NodeId> {
        self.body
            .iter()
            .filter(|&block_idx| ends_in_condition(ssa, block_idx))
            .map(NodeId::new)
            .collect()
    }

    /// Identifies induction variables in this loop.
    ///
    /// An induction variable is identified by finding phi nodes at the loop
    /// header where:
    /// - One operand comes from outside the loop (initial value)
    /// - One operand comes from inside the loop (updated value)
    ///
    /// The method attempts to classify the update kind (add, sub, etc.) by
    /// analyzing the instruction that produces the update value.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function providing the header's phi nodes and the
    ///   defining instructions of their operands.
    ///
    /// # Returns
    ///
    /// A vector of [`InductionVar`] structures describing each induction variable.
    #[must_use]
    pub fn find_induction_vars<T: Target>(&self, ssa: &SsaFunction<T>) -> Vec<InductionVar> {
        let mut induction_vars = Vec::new();

        // Get phi nodes at the header
        let Some(header_block) = ssa.block(self.header.index()) else {
            return induction_vars;
        };

        for phi in header_block.phi_nodes() {
            let operands = phi.operands();

            // Need at least 2 operands (init + update)
            if operands.len() < 2 {
                continue;
            }

            // Find operands from inside vs outside the loop
            let (inside_ops, outside_ops): (Vec<&_>, Vec<&_>) = operands
                .iter()
                .partition(|op| self.contains(NodeId::new(op.predecessor())));

            // Classic induction variable: 1 init from outside, 1+ updates from inside
            if outside_ops.len() == 1 && !inside_ops.is_empty() {
                let (Some(init_op), Some(update_op)) = (outside_ops.first(), inside_ops.first())
                else {
                    continue;
                };

                // Try to determine update kind by analyzing the defining instruction
                let (update_kind, stride) =
                    analyze_update_instruction(ssa, update_op.value(), phi.result());

                induction_vars.push(InductionVar {
                    phi_result: phi.result(),
                    init_value: init_op.value(),
                    init_block: NodeId::new(init_op.predecessor()),
                    update_value: update_op.value(),
                    update_block: NodeId::new(update_op.predecessor()),
                    update_kind,
                    stride,
                });
            }
        }

        induction_vars
    }
}

/// Loop forest containing all loops in a function.
///
/// Provides efficient queries for loop membership, nesting, and iteration.
#[derive(Debug, Clone)]
pub struct LoopForest {
    /// All loops indexed by their header block.
    loops: Vec<LoopInfo>,
    /// Map from block to the innermost loop containing it.
    block_to_loop: Vec<Option<usize>>,
}

impl LoopForest {
    /// Creates an empty loop forest.
    #[must_use]
    pub fn new(block_count: usize) -> Self {
        Self {
            loops: Vec::new(),
            block_to_loop: vec![None; block_count],
        }
    }

    /// Adds a loop to the forest, updating the block-to-loop map.
    ///
    /// Every block in the loop body is pointed at this loop, but only when this
    /// loop is more deeply nested than whatever already claims the block. That
    /// keeps `block_to_loop` resolving to the *innermost* loop for each block,
    /// which is the invariant [`innermost_loop`](Self::innermost_loop) and the
    /// depth queries depend on. Correctness therefore relies on outer loops being
    /// added before inner ones, or at least on the depth comparison ordering them.
    ///
    /// # Arguments
    ///
    /// * `loop_info` - The loop to append; its body drives the map update.
    pub fn add_loop(&mut self, loop_info: LoopInfo) {
        let loop_index = self.loops.len();
        for block in loop_info.body.iter() {
            if let Some(slot) = self.block_to_loop.get_mut(block)
                && slot.is_none_or(|existing| {
                    self.loops
                        .get(existing)
                        .is_some_and(|info| info.depth < loop_info.depth)
                })
            {
                *slot = Some(loop_index);
            }
        }
        self.loops.push(loop_info);
    }

    /// Returns all loops in the forest.
    #[must_use]
    pub fn loops(&self) -> &[LoopInfo] {
        &self.loops
    }

    /// Returns the number of loops.
    #[must_use]
    pub fn len(&self) -> usize {
        self.loops.len()
    }

    /// Returns true if there are no loops.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.loops.is_empty()
    }

    /// Returns the innermost loop containing the given block.
    #[must_use]
    pub fn innermost_loop(&self, block: NodeId) -> Option<&LoopInfo> {
        let block_idx = block.index();
        let loop_idx = (*self.block_to_loop.get(block_idx)?)?;
        self.loops.get(loop_idx)
    }

    /// Returns the loop with the given header.
    #[must_use]
    pub fn loop_for_header(&self, header: NodeId) -> Option<&LoopInfo> {
        self.loops.iter().find(|l| l.header == header)
    }

    /// Returns the loop depth for a block (0 if not in any loop).
    ///
    /// This is a one-based nesting count, not the zero-based `depth` field stored
    /// on [`LoopInfo`]: it looks up the innermost containing loop and adds one, so
    /// an outermost loop's blocks report `1` while a block in no loop reports `0`.
    /// The offset lets a single return value distinguish "not in a loop" from
    /// "in the outermost loop".
    ///
    /// # Arguments
    ///
    /// * `block` - The block whose nesting depth is queried.
    ///
    /// # Returns
    ///
    /// The one-based nesting depth, or `0` when the block is in no loop.
    #[must_use]
    pub fn loop_depth(&self, block: NodeId) -> usize {
        self.innermost_loop(block)
            .map_or(0, |l| l.depth.saturating_add(1))
    }

    /// Returns true if a block is in any loop.
    #[must_use]
    pub fn is_in_loop(&self, block: NodeId) -> bool {
        self.innermost_loop(block).is_some()
    }

    /// Iterates over all loops in the forest.
    pub fn iter(&self) -> impl Iterator<Item = &LoopInfo> {
        self.loops.iter()
    }

    /// Returns loops sorted by depth (outermost first).
    #[must_use]
    pub fn by_depth_ascending(&self) -> Vec<&LoopInfo> {
        let mut sorted: Vec<_> = self.loops.iter().collect();
        sorted.sort_by_key(|l| l.depth);
        sorted
    }

    /// Returns loops sorted by depth (innermost first).
    #[must_use]
    pub fn by_depth_descending(&self) -> Vec<&LoopInfo> {
        let mut sorted: Vec<_> = self.loops.iter().collect();
        sorted.sort_by_key(|l| Reverse(l.depth));
        sorted
    }
}

/// Detects all natural loops in a graph using dominance-based back edge detection.
///
/// This is the primary entry point for loop detection. It works with any graph
/// implementing the required traits, enabling loop analysis on various graph types
/// (CIL CFGs, SSA CFGs, x86 CFGs, etc.).
///
/// # Algorithm
///
/// The detection algorithm:
/// 1. Finds back edges using dominance (n -> h where h dominates n)
/// 2. For each back edge, computes the natural loop body
/// 3. Computes preheaders, exits, and loop types
/// 4. Establishes nesting relationships
///
/// # Arguments
///
/// * `graph` - Any graph implementing `GraphBase + Successors + Predecessors`
/// * `dominators` - Pre-computed dominator tree for the graph
///
/// # Returns
///
/// A [`LoopForest`] containing all detected loops with their full analysis.
///
/// # Examples
///
/// ```rust
/// use analyssa::{
///     analysis::{SsaCfg, detect_loops},
///     graph::{algorithms::compute_dominators, NodeId, RootedGraph},
///     testing,
/// };
///
/// let ssa = testing::loop_counter_fixture();
/// let graph = SsaCfg::from_ssa(&ssa);
///
/// let dominators = compute_dominators(&graph, graph.entry());
/// let forest = detect_loops(&graph, &dominators);
///
/// let headers: Vec<_> = forest
///     .loops()
///     .iter()
///     .map(|loop_info| (loop_info.header, loop_info.size()))
///     .collect();
///
/// // One loop, headed by block 1, whose body is just that block.
/// assert_eq!(headers, vec![(NodeId::new(1), 1)]);
/// ```
#[must_use]
pub fn detect_loops<G>(graph: &G, dominators: &DominatorTree) -> LoopForest
where
    G: GraphBase + Successors + Predecessors,
{
    let block_count = graph.node_count();
    let mut forest = LoopForest::new(block_count);
    // Header ordering determines stable loop indices.
    let mut loops_by_header: BTreeMap<NodeId, LoopInfo> = BTreeMap::new();
    for node in graph.node_ids() {
        for succ in graph.successors(node) {
            if dominators.dominates(succ, node) {
                let loop_info = match loops_by_header.entry(succ) {
                    Entry::Vacant(entry) => entry.insert(LoopInfo::new(succ, block_count)),
                    Entry::Occupied(entry) => entry.into_mut(),
                };
                loop_info.latches.push(node);
                expand_loop_body(graph, dominators, loop_info, node);
            }
        }
    }
    for info in loops_by_header.values_mut() {
        compute_preheader(graph, info);
        compute_exits(graph, info);
        info.loop_type = classify_loop(info);
    }

    let mut loops = Vec::with_capacity(loops_by_header.len());
    for info in loops_by_header.into_values() {
        loops.push(info);
    }
    compute_nesting(&mut loops);
    for info in loops {
        forest.add_loop(info);
    }
    forest
}

/// Checks if a graph has any back edges (loops).
///
/// This is a fast check that returns as soon as the first back edge is found,
/// without building the full loop forest. Use this when you only need to know
/// whether loops exist, not their detailed structure.
///
/// # Arguments
///
/// * `graph` - Any graph implementing `GraphBase + Successors`
/// * `dominators` - Pre-computed dominator tree for the graph
///
/// # Returns
///
/// `true` if at least one back edge exists, `false` otherwise.
///
/// # Examples
///
/// ```rust
/// use analyssa::{
///     analysis::{SsaCfg, loops::has_back_edges},
///     graph::{algorithms::compute_dominators, RootedGraph},
///     testing,
/// };
///
/// // A loop fixture has a back edge...
/// let looping = testing::loop_counter_fixture();
/// let graph = SsaCfg::from_ssa(&looping);
/// let dominators = compute_dominators(&graph, graph.entry());
/// assert!(has_back_edges(&graph, &dominators));
///
/// // ...while an acyclic diamond CFG does not.
/// let diamond = testing::diamond_phi_fixture();
/// let graph = SsaCfg::from_ssa(&diamond);
/// let dominators = compute_dominators(&graph, graph.entry());
/// assert!(!has_back_edges(&graph, &dominators));
/// ```
#[must_use]
pub fn has_back_edges<G>(graph: &G, dominators: &DominatorTree) -> bool
where
    G: GraphBase + Successors,
{
    for node in graph.node_ids() {
        for succ in graph.successors(node) {
            if dominators.dominates(succ, node) {
                return true;
            }
        }
    }
    false
}

/// Expands the loop body to include all nodes that can reach the latch.
///
/// Uses a worklist algorithm: starting from the latch, we add
/// predecessors that aren't the header until we've found all loop body nodes.
/// Only a predecessor the header dominates belongs to the loop. Every block
/// the entry reaches that reaches the latch without passing the header is one,
/// so this keeps out exactly the blocks no path from the entry reaches: one
/// that branches into the body would otherwise join it, and its other
/// successors would read as exits the loop never takes.
fn expand_loop_body<G: Predecessors>(
    graph: &G,
    dominators: &DominatorTree,
    loop_info: &mut LoopInfo,
    latch: NodeId,
) {
    if loop_info.contains(latch) {
        return;
    }
    let mut worklist = Vec::new();
    worklist.push(latch);
    while let Some(node) = worklist.pop() {
        if loop_info.body.insert(node.index()) {
            for pred in graph.predecessors(node) {
                if pred != loop_info.header
                    && !loop_info.body.contains_checked(pred.index())
                    && dominators.dominates(loop_info.header, pred)
                {
                    worklist.push(pred);
                }
            }
        }
    }
}

/// Finds the unique external predecessor, stopping once two distinct blocks disprove it.
fn compute_preheader<G: Predecessors>(graph: &G, info: &mut LoopInfo) {
    let mut first = None;
    for pred in graph.predecessors(info.header) {
        if !info.body.contains_checked(pred.index()) {
            match first {
                None => first = Some(pred),
                Some(known) if pred != known => {
                    info.preheader = None;
                    return;
                }
                Some(_) => {}
            }
        }
    }
    info.preheader = first;
}

/// Collects every exit edge from the loop's blocks, preserving graph iteration order.
fn compute_exits<G: Successors>(graph: &G, info: &mut LoopInfo) {
    info.exits.clear();
    for index in info.body.iter() {
        let block = NodeId::new(index);
        for successor in graph.successors(block) {
            if !info.body.contains_checked(successor.index()) {
                info.exits.push(LoopExit {
                    exiting_block: block,
                    exit_block: successor,
                });
            }
        }
    }
}

/// Classifies the loop from its exit locations.
fn classify_loop(info: &LoopInfo) -> LoopType {
    if info.exits.is_empty() {
        return LoopType::Infinite;
    }
    if info.latches.len() > 1 {
        return LoopType::Complex;
    }
    let latch = info.single_latch();
    let mut latch_exits = 0usize;
    let mut header_exits = 0usize;
    for exit in &info.exits {
        if Some(exit.exiting_block) == latch {
            latch_exits = latch_exits.saturating_add(1);
        }
        if exit.exiting_block == info.header {
            header_exits = header_exits.saturating_add(1);
        }
    }
    if latch.is_some() && latch_exits == info.exits.len() {
        LoopType::PostTested
    } else if header_exits == info.exits.len() {
        LoopType::PreTested
    } else {
        LoopType::Complex
    }
}

/// Computes deterministic parent, child and depth relations.
fn compute_nesting(loops: &mut [LoopInfo]) {
    let n = loops.len();

    let mut header_to_idx = FxHashMap::with_capacity_and_hasher(n, Default::default());

    let mut sizes = Vec::with_capacity(n);
    for (index, info) in loops.iter().enumerate() {
        header_to_idx.insert(info.header, index);
        sizes.push(info.body.count());
    }
    for i in 0..n {
        let Some(header) = loops.get(i).map(|info| info.header) else {
            continue;
        };
        let mut parent = None;
        for j in 0..n {
            if j != i && loops.get(j).is_some_and(|info| info.contains(header)) {
                let key = (sizes.get(j).copied().unwrap_or(usize::MAX), j);
                if parent.is_none_or(|known| key < known) {
                    parent = Some(key);
                }
            }
        }
        if let Some((_, index)) = parent {
            let header = loops.get(index).map(|info| info.header);
            if let Some(info) = loops.get_mut(i) {
                info.parent = header;
            }
        }
    }
    for i in 0..n {
        let Some((parent, header)) = loops
            .get(i)
            .and_then(|info| Some((info.parent?, info.header)))
        else {
            continue;
        };
        if let Some(index) = header_to_idx.get(&parent)
            && let Some(info) = loops.get_mut(*index)
        {
            info.children.push(header);
        }
    }
    for i in 0..n {
        let mut depth = 0usize;
        let mut current = loops.get(i).and_then(|info| info.parent);
        while let Some(parent) = current {
            depth = depth.saturating_add(1);
            current = header_to_idx
                .get(&parent)
                .and_then(|index| loops.get(*index))
                .and_then(|info| info.parent);
        }
        if let Some(info) = loops.get_mut(i) {
            info.depth = depth;
        }
    }
}

/// Whether `block_idx` ends in a two-way conditional terminator.
fn ends_in_condition<T: Target>(ssa: &SsaFunction<T>, block_idx: usize) -> bool {
    ssa.block(block_idx)
        .and_then(SsaBlock::control_terminator)
        .and_then(SsaOp::two_way_targets)
        .is_some()
}

/// Analyzes an instruction to determine if it's an induction update.
///
/// Looks for patterns like `v = phi_result + const` or `v = phi_result - const`,
/// recovering the stride from the other operand when it is a constant. `Add`
/// and `Mul` accept the phi result on either side (both are commutative),
/// but `Sub` only recognizes `phi_result - const`, since a subtraction with
/// the phi on the right is not a simple stride. Anything else, including a
/// non-constant stride, yields `Unknown` with no stride.
///
/// # Arguments
///
/// * `ssa` - The SSA function used to look up the defining instruction and
///   resolve constant operands.
/// * `update_var` - The variable holding the per-iteration updated value.
/// * `phi_result` - The header phi result the update is expected to build on.
///
/// # Returns
///
/// The classified update kind paired with the stride, if it could be determined.
fn analyze_update_instruction<T: Target>(
    ssa: &SsaFunction<T>,
    update_var: SsaVarId,
    phi_result: SsaVarId,
) -> (InductionUpdateKind, Option<i64>) {
    // Find the instruction that defines update_var.
    //
    // Recovering lookup: this runs once per induction candidate, so the
    // fallback scan is bounded by the number of candidates, and what a
    // stale def site would otherwise cost is a *stride* - the loop's trip
    // count and every consumer of it. The failure policy is
    // `InductionUpdateKind::Unknown`, which is what this already answers
    // for an update it cannot read, so recovery only adds answers; it never
    // turns an unknown stride into a wrong one.
    let Some(instr) = ssa.get_definition_instruction(update_var) else {
        return (InductionUpdateKind::Unknown, None);
    };

    // Check for Add/Sub patterns
    match instr.op() {
        // Check if one operand is the phi result
        SsaOp::Add { left, right, .. } if *left == phi_result || *right == phi_result => {
            let other = if *left == phi_result { *right } else { *left };
            let stride = ssa.try_constant_value(other).and_then(|v| v.as_i64());
            return (InductionUpdateKind::Add, stride);
        }
        // For subtraction, left should be phi_result
        SsaOp::Sub { left, right, .. } if *left == phi_result => {
            let stride = ssa.try_constant_value(*right).and_then(|v| v.as_i64());
            return (InductionUpdateKind::Sub, stride);
        }
        SsaOp::Mul { left, right, .. } if *left == phi_result || *right == phi_result => {
            let other = if *left == phi_result { *right } else { *left };
            let stride = ssa.try_constant_value(other).and_then(|v| v.as_i64());
            return (InductionUpdateKind::Mul, stride);
        }
        _ => {}
    }

    (InductionUpdateKind::Unknown, None)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::{
        analysis::cfg::SsaCfg,
        graph::{DirectedGraph, algorithms::compute_dominators},
        ir::{
            SsaFunction, SsaVarId,
            instruction::SsaInstruction,
            ops::{CmpKind, SsaOp, native::FlagCondition},
        },
        testing::MockTarget,
    };

    /// **A block no path from the entry reaches is in no loop.** The loop
    /// `0 -> 1 -> 2 -> 0` never leaves, and `3`, which nothing reaches,
    /// branches into it at `1` and out of it to `4`. Walking back from the
    /// latch reached `3` and took it into the body, so `4` read as an exit and
    /// the structurer resumed the loop that never ends at a block that never
    /// runs.
    #[test]
    fn an_unreachable_block_is_in_no_loop() {
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        for _ in 0..5 {
            graph.add_node(());
        }
        for (from, to) in [(0, 1), (1, 2), (2, 0), (3, 1), (3, 4)] {
            graph
                .add_edge(NodeId::new(from), NodeId::new(to), ())
                .expect("edge endpoints exist");
        }
        let dominators = compute_dominators(&graph, NodeId::new(0));
        let forest = detect_loops(&graph, &dominators);
        let [looped] = forest.loops() else {
            panic!("one loop: {:?}", forest.loops());
        };
        assert!(!looped.contains(NodeId::new(3)));
        assert_eq!(looped.exit_count(), 0);
    }

    /// **A loop lists its children in header order.** Nesting numbers the
    /// loops by position, so a position that follows the detection map's hash
    /// order lists the siblings of one parent in that order, and the forest
    /// differs between two builds of the same graph.
    #[test]
    fn a_loops_children_ascend_by_header() {
        // 0 -> 1, the outer loop's header, which holds self-loops at 2 to 6
        // in a chain and returns from 7; 8 is its exit.
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        for _ in 0..9 {
            graph.add_node(());
        }
        let edges = [
            (0, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 4),
            (4, 4),
            (4, 5),
            (5, 5),
            (5, 6),
            (6, 6),
            (6, 7),
            (7, 1),
            (1, 8),
        ];
        for (from, to) in edges {
            graph
                .add_edge(NodeId::new(from), NodeId::new(to), ())
                .expect("edge endpoints exist");
        }
        let dominators = compute_dominators(&graph, NodeId::new(0));
        let forest = detect_loops(&graph, &dominators);

        let outer = forest
            .loops()
            .iter()
            .find(|looped| looped.header == NodeId::new(1))
            .expect("the outer loop is detected");
        let children: Vec<usize> = outer.children.iter().map(|child| child.index()).collect();
        assert_eq!(children, vec![2, 3, 4, 5, 6]);
    }

    /// A block index the set cannot represent is not in the loop - the answer,
    /// not a programming error. Callers reach this with indices from variable
    /// def sites and CFG metadata, which describe the function as it was and can
    /// name a block it no longer has; asking the `BitSet` directly asserts, and
    /// a bounds assert is a panic.
    #[test]
    fn a_block_index_past_the_end_is_not_in_the_loop() {
        let mut info = LoopInfo::new(NodeId::new(0), 4);
        info.body.insert(2);

        assert!(
            info.contains(NodeId::new(0)),
            "the header is in its own loop"
        );
        assert!(info.contains(NodeId::new(2)));
        assert!(!info.contains(NodeId::new(3)));
        assert!(!info.contains(NodeId::new(4)), "one past the end");
        assert!(!info.contains(NodeId::new(9_999)), "far past the end");
    }

    /// A single-instruction block, for edge-shape fixtures.
    fn block(id: usize, op: SsaOp<MockTarget>) -> SsaBlock<MockTarget> {
        let mut block = SsaBlock::new(id);
        block.add_instruction(SsaInstruction::synthetic(op));
        block
    }

    /// A header entered twice from one outside block still has a preheader.
    ///
    /// `Branch c, header, header` is two edges from one block. There is one
    /// place to hoist loop-invariant code to and one terminator to redirect, so
    /// counting edges rather than blocks would refuse a canonical loop.
    #[test]
    fn a_doubled_branch_arm_into_a_header_is_one_preheader() {
        // B0 branches both ways into the header B1; B1 loops to itself and exits.
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(
            0,
            SsaOp::Branch {
                condition: SsaVarId::from_index(0),
                true_target: 1,
                false_target: 1,
            },
        ));
        ssa.add_block(block(
            1,
            SsaOp::Branch {
                condition: SsaVarId::from_index(0),
                true_target: 1,
                false_target: 2,
            },
        ));
        ssa.add_block(block(2, SsaOp::returns(None)));

        // The terminator relation, deliberately: the question is how many edges
        // a branch takes into the header. `compute_dominators` takes its entry
        // explicitly, so this graph does not need to be rooted.
        let cfg = SsaCfg::from_ssa(&ssa);
        assert_eq!(
            cfg.block_predecessors(1),
            &[0, 0, 1],
            "two edges from B0 and the latch"
        );

        let dominators = compute_dominators(&cfg, NodeId::new(0));
        let forest = detect_loops(&cfg, &dominators);
        let loop_info = forest
            .loops()
            .iter()
            .find(|info| info.header == NodeId::new(1));
        assert!(loop_info.is_some(), "the self-loop is a loop");
        assert_eq!(
            loop_info.and_then(|info| info.preheader),
            Some(NodeId::new(0)),
            "one distinct block enters from outside, so it is the preheader"
        );
    }

    /// A loop whose condition is a compare-and-branch or a flag test reports
    /// that block as its condition, as it does a plain `Branch`.
    #[test]
    fn every_two_way_terminator_is_a_loop_condition() {
        let var = SsaVarId::from_index(0);
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(0, SsaOp::Jump { target: 1 }));
        ssa.add_block(block(
            1,
            SsaOp::BranchCmp {
                left: var,
                right: var,
                cmp: CmpKind::Lt,
                unsigned: false,
                true_target: 2,
                false_target: 3,
            },
        ));
        ssa.add_block(block(
            2,
            SsaOp::BranchFlags {
                flags: var,
                condition: FlagCondition::Zero,
                true_target: 1,
                false_target: 3,
            },
        ));
        ssa.add_block(block(3, SsaOp::returns(None)));

        let mut info = LoopInfo::new(NodeId::new(1), 4);
        info.body.insert(2);
        assert_eq!(info.find_condition_in_body(&ssa), Some(NodeId::new(1)));
        assert_eq!(
            info.find_all_conditions_in_body(&ssa),
            [NodeId::new(1), NodeId::new(2)]
        );
    }

    /// A freshly created `LoopInfo` contains only its header and is not yet canonical.
    #[test]
    fn test_loop_info_creation() {
        let header = NodeId::new(0);
        let loop_info = LoopInfo::new(header, 10);

        assert_eq!(loop_info.header, header);
        assert!(loop_info.contains(header));
        assert_eq!(loop_info.size(), 1);
        assert!(!loop_info.has_single_latch());
        assert!(!loop_info.has_preheader());
        assert!(!loop_info.is_canonical());
    }

    /// A loop with one preheader and one latch reports as canonical.
    #[test]
    fn test_loop_info_canonical() {
        let header = NodeId::new(1);
        let mut loop_info = LoopInfo::new(header, 10);

        loop_info.preheader = Some(NodeId::new(0));
        loop_info.latches.push(NodeId::new(2));

        assert!(loop_info.has_preheader());
        assert!(loop_info.has_single_latch());
        assert!(loop_info.is_canonical());
    }

    /// The forest reports each block's depth from the innermost loop containing it.
    #[test]
    fn test_loop_forest() {
        let mut forest = LoopForest::new(10);

        let mut outer_loop = LoopInfo::new(NodeId::new(1), 10);
        outer_loop.body.insert(2);
        outer_loop.body.insert(3);
        outer_loop.depth = 0;

        let mut inner_loop = LoopInfo::new(NodeId::new(2), 10);
        inner_loop.body.insert(3);
        inner_loop.depth = 1;

        forest.add_loop(outer_loop);
        forest.add_loop(inner_loop);

        assert_eq!(forest.len(), 2);

        // Block 3 should be in inner loop (depth 1)
        assert_eq!(forest.loop_depth(NodeId::new(3)), 2);

        // Block 1 should be in outer loop only
        assert_eq!(forest.loop_depth(NodeId::new(1)), 1);

        // Block 0 should not be in any loop
        assert_eq!(forest.loop_depth(NodeId::new(0)), 0);
    }
    /// Builds a nested loop with two inner exits and one outer exit.
    fn admission_fixture() -> DirectedGraph<'static, (), ()> {
        let mut graph = DirectedGraph::new();
        for _ in 0..8 {
            graph.add_node(());
        }
        for (from, to) in [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 1),
            (1, 7),
            (3, 6),
            (6, 4),
        ] {
            graph
                .add_edge(NodeId::new(from), NodeId::new(to), ())
                .unwrap();
        }
        graph
    }

    /// Natural loop membership, ordering and depth match the explicit graph structure.
    #[test]
    fn loop_detection_preserves_the_nested_loop_structure() {
        let graph = admission_fixture();
        let dominators = compute_dominators(&graph, NodeId::new(0));
        let forest = detect_loops(&graph, &dominators);
        let outer = forest.loop_for_header(NodeId::new(1)).unwrap();
        let inner = forest.loop_for_header(NodeId::new(2)).unwrap();
        assert_eq!(outer.body.iter().collect::<Vec<_>>(), [1, 2, 3, 4, 6]);
        assert_eq!(inner.body.iter().collect::<Vec<_>>(), [2, 3]);
        assert_eq!(outer.children, [NodeId::new(2)]);
        assert_eq!(outer.depth, 0);
        assert_eq!(inner.depth, 1);
        assert_eq!(inner.parent, Some(NodeId::new(1)));
        assert_eq!(
            forest
                .innermost_loop(NodeId::new(3))
                .map(|info| info.header),
            Some(NodeId::new(2))
        );
        assert_eq!(outer.loop_type, LoopType::PreTested);
        assert_eq!(inner.loop_type, LoopType::PostTested);
    }

    /// Counts predecessor edges the loop expander actually consumes.
    struct CountedLoop {
        nodes: usize,
        inspected: Cell<usize>,
    }
    impl GraphBase for CountedLoop {
        fn node_count(&self) -> usize {
            self.nodes
        }
        fn node_ids(&self) -> impl Iterator<Item = NodeId> {
            (0..self.nodes).map(NodeId::new)
        }
    }
    impl Successors for CountedLoop {
        fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
            let next = node.index().saturating_add(1);
            [NodeId::new(if next < self.nodes { next } else { 1 })].into_iter()
        }
    }
    impl Predecessors for CountedLoop {
        fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
            let index = node.index();
            let first = (index > 0).then(|| NodeId::new(index.saturating_sub(1)));
            let back = (index == 1).then(|| NodeId::new(self.nodes.saturating_sub(1)));
            [first, back]
                .into_iter()
                .flatten()
                .inspect(|_| self.inspected.set(self.inspected.get().saturating_add(1)))
        }
    }

    /// Iterative expansion covers a large loop's full body.
    #[test]
    fn loop_body_expansion_covers_a_deep_cycle() {
        let graph = CountedLoop {
            nodes: 4096,
            inspected: Cell::new(0),
        };
        let dominators = compute_dominators(&graph, NodeId::new(0));
        let forest = detect_loops(&graph, &dominators);
        assert_eq!(
            forest.loop_for_header(NodeId::new(1)).map(LoopInfo::size),
            Some(4095)
        );
        assert!(graph.inspected.get() >= 4095);
    }

    /// Nested loop parent and depth selection agree with the explicit containment order.
    #[test]
    fn nesting_preserves_all_parent_relations() {
        let mut loops: Vec<_> = (0..128)
            .map(|header| {
                let mut info = LoopInfo::new(NodeId::new(header), 128);
                for member in header..128 {
                    info.body.insert(member);
                }
                info
            })
            .collect();
        compute_nesting(&mut loops);
        assert_eq!(
            loops.iter().filter(|info| info.parent.is_some()).count(),
            127
        );
        assert_eq!(
            loops.last().map(|info| (info.parent, info.depth)),
            Some((Some(NodeId::new(126)), 127))
        );
    }
}
