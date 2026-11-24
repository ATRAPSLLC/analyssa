//! Control flow graph view of SSA functions.
//!
//! This module provides [`SsaCfg`], a lightweight CFG view constructed directly
//! from an [`SsaFunction`]: one graph carrying both relations control actually
//! has, so a consumer picks a question rather than a graph type.
//!
//! # The two relations
//!
//! Control leaves a block two ways. Its terminator names a transfer, and a
//! throw anywhere inside it transfers to whichever handler or filter entry
//! covers it - dispatched by the runtime, named by no terminator.
//!
//! The **complete** relation holds both, and is what the plain accessors
//! ([`block_successors`](SsaCfg::block_successors),
//! [`block_predecessors`](SsaCfg::block_predecessors)) return. It is the
//! default because it is the one that is true: a handler is reachable in it,
//! so a dominator tree, a liveness solve or a reachability prune built on it
//! accounts for the values a throw carries into a handler instead of treating
//! the handler as dead code.
//!
//! The **terminator** relation holds only the transfers terminators name, and
//! is reached through [`terminator_successors`](SsaCfg::terminator_successors),
//! [`terminator_predecessors`](SsaCfg::terminator_predecessors) or, for an
//! algorithm that wants it as a graph, [`terminator_view`](SsaCfg::terminator_view).
//! A consumer wants it when an edge must correspond to something a block
//! *chooses*: a block's successor count is its terminator's arity there, which
//! is what structure recovery reads, and a phi's normal operands stand one per
//! such edge.
//!
//! Which edges are exceptional is never guesswork:
//! [`exceptional_successors`](SsaCfg::exceptional_successors),
//! [`exceptional_predecessors`](SsaCfg::exceptional_predecessors) and
//! [`is_exception_edge`](SsaCfg::is_exception_edge) name them, and
//! [`roots`](SsaCfg::roots) says where control can enter at all.
//!
//! # Design
//!
//! `SsaCfg` holds a reference to the SSA function (zero-copy) and caches both
//! relations' successor and predecessor lists. It implements the standard graph
//! traits over the complete relation:
//! - [`GraphBase`] - Node count and iteration
//! - [`Successors`] - Forward edge traversal
//! - [`Predecessors`] - Backward edge traversal
//! - [`RootedGraph`] - Entry plus every handler and filter entry
//!
//! This bridges the gap between passes (which receive `SsaFunction`, not the
//! original CIL CFG) and dataflow analyses that require a CFG.
//!
//! # Complexity
//!
//! Construction: O(V + E) time and memory, where E counts both relations'
//! edges. All queries are O(1) or O(k), where k is the number of adjacent
//! nodes.
//!
//! # Construction
//!
//! The CFG is constructed on-demand from the SSA function:
//!
//! ```rust
//! use analyssa::{analysis::cfg::SsaCfg, MockTarget, ir::{SsaBlock, SsaFunction, SsaInstruction, SsaOp}};
//!
//! let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
//! let mut block = SsaBlock::new(0);
//! block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
//! ssa.add_block(block);
//! let cfg = SsaCfg::from_ssa(&ssa);
//! assert_eq!(cfg.block_count(), 1);
//! ```

use crate::{
    BitSet,
    analysis::exceptions::{FinallyFlow, FunctionRoots},
    graph::{Adjacency, GraphBase, NodeId, Predecessors, RootedGraph, Successors},
    ir::{function::SsaFunction, ops::SsaOp},
    target::Target,
};

/// A lightweight control flow graph view of an SSA function.
///
/// This struct provides a CFG interface over an existing [`SsaFunction`],
/// extracting control flow edges from block terminators. It's designed to
/// enable dataflow analyses that require a CFG without duplicating the
/// underlying SSA structure.
///
/// # Performance
///
/// The CFG computes and caches predecessor lists on construction. This is
/// an O(E) operation where E is the number of edges (typically similar to
/// the number of blocks). Once constructed, all queries are O(1) or O(k)
/// where k is the number of adjacent nodes.
///
/// # Lifetime
///
/// The CFG holds a reference to the SSA function it was created from.
/// The CFG must not outlive the SSA function.
#[derive(Debug)]
pub struct SsaCfg<'a, T: Target> {
    /// Reference to the SSA function.
    ssa: &'a SsaFunction<T>,
    /// The blocks each block's terminator transfers to.
    ///
    /// Compressed rather than `Vec<Vec<usize>>` because a CFG view is rebuilt
    /// repeatedly (dominators, loop analysis, and most passes construct one),
    /// and a vector per block cost two allocations per block every time.
    successors: Adjacency,
    /// The exact inverse of [`Self::successors`].
    predecessors: Adjacency,
    /// Where control can enter the function, and which blocks a throw in can
    /// reach which handler.
    roots: FunctionRoots,
    /// The successors including the transfers a throw makes.
    flow_successors: Adjacency,
    /// The exact inverse of [`Self::flow_successors`].
    flow_predecessors: Adjacency,
}

impl<'a, T: Target> SsaCfg<'a, T> {
    /// Creates a CFG view from an SSA function.
    ///
    /// This extracts control flow edges by examining the terminator of each
    /// SSA block. Predecessors are computed and cached for efficient backward
    /// traversal.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to create a CFG view of.
    ///
    /// # Returns
    ///
    /// A new `SsaCfg` view of the given function.
    ///
    /// # Example
    ///
    /// ```rust
    /// use analyssa::{analysis::SsaCfg, graph::GraphBase, MockTarget, ir::{SsaBlock, SsaFunction, SsaInstruction, SsaOp}};
    ///
    /// let mut ssa_function = SsaFunction::<MockTarget>::new(0, 0);
    /// let mut block = SsaBlock::new(0);
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::returns(None)));
    /// ssa_function.add_block(block);
    ///
    /// let cfg = SsaCfg::from_ssa(&ssa_function);
    /// assert_eq!(cfg.node_count(), ssa_function.block_count());
    /// ```
    #[must_use]
    pub fn from_ssa(ssa: &'a SsaFunction<T>) -> Self {
        let n = ssa.block_count();
        let finally_flow = FinallyFlow::of(ssa);
        let successors = Adjacency::from_rows(n, |index, row| {
            let Some(block) = ssa.block(index) else {
                return;
            };
            match block.control_terminator() {
                Some(SsaOp::Leave { .. }) if finally_flow.diverted_target(index).is_some() => {
                    if let Some(entry) = finally_flow.diverted_target(index)
                        && entry < n
                    {
                        row.push(entry);
                    }
                }
                Some(SsaOp::EndFinally) => {
                    for resume in finally_flow.resume_targets(index) {
                        if *resume < n {
                            row.push(*resume);
                        }
                    }
                }
                Some(op) => op.for_each_successor(|successor| {
                    if successor < n {
                        row.push(successor);
                    }
                }),
                None => {}
            }
        });
        let roots = FunctionRoots::of(ssa);
        let flow_successors = Adjacency::from_rows(n, |index, row| {
            let terminator = successors.row(index);
            row.extend_from_slice(terminator);
            for root in roots.exceptional_successors(index) {
                if *root < n && !terminator.contains(root) {
                    row.push(*root);
                }
            }
        });
        let predecessors = successors.inverse();
        let flow_predecessors = flow_successors.inverse();
        Self {
            ssa,
            successors,
            predecessors,
            roots,
            flow_successors,
            flow_predecessors,
        }
    }

    /// Returns the underlying SSA function.
    ///
    /// This can be used to access block and instruction data while
    /// traversing the CFG.
    #[must_use]
    pub const fn ssa(&self) -> &'a SsaFunction<T> {
        self.ssa
    }

    /// Returns the number of blocks in the CFG.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.ssa.block_count()
    }

    /// Returns true if the CFG has no blocks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ssa.is_empty()
    }

    /// Returns the blocks this block's terminator transfers to.
    ///
    /// One entry per transfer the terminator names, in its own operand order -
    /// so `Branch c, 1, 1` yields `[1, 1]`, and the slice length is the
    /// terminator's arity. A throw is not a transfer the terminator names, so
    /// it is not here; [`Self::block_successors`] is the complete relation.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block index to query.
    ///
    /// # Returns
    ///
    /// A slice of successor block indices. Empty if the block has no
    /// successors (e.g., return, throw) or doesn't exist.
    #[must_use]
    pub fn terminator_successors(&self, block_idx: usize) -> &[usize] {
        self.successors.row(block_idx)
    }

    /// Returns every block control can reach from this one.
    ///
    /// The terminator's transfers, plus the handler or filter a throw in this
    /// block dispatches to. This is the relation a question about reachability,
    /// dominance, liveness or where values merge is about, and it is the one
    /// the graph traits expose.
    ///
    /// [`Self::terminator_successors`] is the narrower question - *what does
    /// this block's terminator name* - which arity, trampoline and
    /// edge-rewriting decisions ask instead.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block index to query.
    #[must_use]
    pub fn block_successors(&self, block_idx: usize) -> &[usize] {
        self.flow_successors.row(block_idx)
    }

    /// Returns the blocks whose terminator transfers to this one.
    ///
    /// The exact inverse of [`Self::terminator_successors`], **multiplicity
    /// included**: a block reached twice from one predecessor lists that
    /// predecessor twice, and the entries ascend by source block.
    ///
    /// The multiset is load-bearing, not an oversight. `topological_sort`
    /// counts in-degree through this relation and decrements it through the
    /// successor one, so a one-sided deduplication reports a cycle on an
    /// acyclic CFG containing `Branch c, 1, 1`; `compute_preheader` and the
    /// structurer make count decisions on the same trait; and phi placement
    /// tests `predecessors.len() < 2` to find joins, which has to agree with
    /// what phi *validation* sees. Use [`Self::predecessor_blocks`] for the
    /// set.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block index to query.
    ///
    /// # Returns
    ///
    /// A slice of predecessor block indices.
    #[must_use]
    pub fn terminator_predecessors(&self, block_idx: usize) -> &[usize] {
        self.predecessors.row(block_idx)
    }

    /// Returns every block control can reach this one from.
    ///
    /// The inverse of [`Self::block_successors`]: the blocks whose terminator
    /// transfers here, plus - for a handler or filter entry - every block a
    /// throw in which dispatches here.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block index to query.
    #[must_use]
    pub fn block_predecessors(&self, block_idx: usize) -> &[usize] {
        self.flow_predecessors.row(block_idx)
    }

    /// Returns where control can enter the function, and which blocks can
    /// throw to which handler.
    #[must_use]
    pub const fn roots(&self) -> &FunctionRoots {
        &self.roots
    }

    /// Returns the handlers and filters a throw in this block can reach.
    ///
    /// The exceptional half of [`Self::block_successors`].
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block a throw would leave.
    #[must_use]
    pub fn exceptional_successors(&self, block_idx: usize) -> &[usize] {
        self.roots.exceptional_successors(block_idx)
    }

    /// Returns the blocks a throw in which dispatches to this one.
    ///
    /// The exceptional half of [`Self::block_predecessors`].
    ///
    /// # Arguments
    ///
    /// * `block_idx` - A handler or filter entry.
    #[must_use]
    pub fn exceptional_predecessors(&self, block_idx: usize) -> &[usize] {
        self.roots.exceptional_predecessors(block_idx)
    }

    /// Returns the blocks an exceptional edge leaves.
    ///
    /// A definition in one of these may not have run when the throw happened,
    /// which is what [`EhDominance::definition_reaches`] refines.
    ///
    /// [`EhDominance::definition_reaches`]: crate::analysis::exceptions::EhDominance::definition_reaches
    #[must_use]
    pub fn edge_sources(&self) -> BitSet {
        let mut sources = BitSet::new(self.ssa.block_count().max(1));
        for block in 0..self.ssa.block_count() {
            if !self.exceptional_successors(block).is_empty() {
                sources.insert_checked(block);
            }
        }
        sources
    }

    /// Returns a view of this graph restricted to terminator edges.
    ///
    /// The same object read the narrow way, for the questions that are about
    /// what terminators name rather than about where control can go: branch
    /// arity, trampolines, and the dominance a value has along ordinary
    /// control flow.
    #[must_use]
    pub const fn terminator_view(&self) -> TerminatorView<'_, 'a, T> {
        TerminatorView { cfg: self }
    }

    /// Returns whether a throw in `from` can take control to `to`.
    ///
    /// Not the opposite of a terminator edge: one edge can be both. A protected
    /// block that leaves to its own `finally` gets there two ways - by running
    /// to the `leave`, and by throwing part-way through - and a phi at that
    /// handler has to account for each, with a normal operand for the first and
    /// exceptional operands for the second. Ask
    /// [`terminator_successors`](Self::terminator_successors) for the other
    /// half rather than inferring it from this answer.
    ///
    /// # Arguments
    ///
    /// * `from` - The block control leaves.
    /// * `to` - The block it arrives at.
    #[must_use]
    pub fn is_exception_edge(&self, from: usize, to: usize) -> bool {
        self.exceptional_successors(from).contains(&to)
    }

    /// Returns the successors of every block, as an owned snapshot.
    ///
    /// For the callers that mutate the function afterwards and so cannot hold
    /// this view's borrow. Row `b` is [`Self::block_successors`] for `b`.
    #[must_use]
    pub fn to_successor_lists(&self) -> Vec<Vec<usize>> {
        (0..self.ssa.block_count())
            .map(|block| self.block_successors(block).to_vec())
            .collect()
    }

    /// Returns each distinct block that transfers to `block_idx`, ascending.
    ///
    /// The set view of [`Self::block_predecessors`], for callers asking "which
    /// blocks flow into this one" rather than "how many edges arrive". No
    /// allocation: the CSR row already ascends, so duplicates are adjacent.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The block index to query.
    pub fn predecessor_blocks(&self, block_idx: usize) -> impl Iterator<Item = usize> + '_ {
        let row = self.block_predecessors(block_idx);
        row.iter().enumerate().filter_map(move |(position, block)| {
            match position
                .checked_sub(1)
                .and_then(|previous| row.get(previous))
            {
                Some(earlier) if earlier == block => None,
                _ => Some(*block),
            }
        })
    }

    /// Returns the distinct predecessors of every block, as an owned snapshot.
    ///
    /// For the callers that mutate the function afterwards and so cannot hold
    /// this view's borrow. Row `b` is [`Self::predecessor_blocks`] for `b`.
    #[must_use]
    pub fn to_predecessor_sets(&self) -> Vec<Vec<usize>> {
        (0..self.ssa.block_count())
            .map(|block| self.predecessor_blocks(block).collect())
            .collect()
    }

    /// Returns the exit nodes of the CFG.
    ///
    /// A block whose terminator transfers nowhere: control leaves the function
    /// there. Read from the terminator relation deliberately - a `Return`
    /// inside a protected region still ends the function, even though a throw
    /// earlier in the same block would have gone to the handler.
    ///
    /// # Returns
    ///
    /// A vector of exit node IDs.
    #[must_use]
    pub fn exits(&self) -> Vec<NodeId> {
        let mut exits = Vec::new();
        for idx in 0..self.ssa.block_count() {
            if self.terminator_successors(idx).is_empty() {
                exits.push(NodeId::new(idx));
            }
        }
        exits
    }
}

impl<T: Target> GraphBase for SsaCfg<'_, T> {
    fn node_count(&self) -> usize {
        self.ssa.block_count()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.ssa.block_count()).map(NodeId::new)
    }
}

impl<T: Target> Successors for SsaCfg<'_, T> {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.block_successors(node.index())
            .iter()
            .copied()
            .map(NodeId::new)
    }
}

/// A [`SsaCfg`] read as its terminator edges alone.
///
/// Produced by [`SsaCfg::terminator_view`]. It borrows the graph rather than
/// rebuilding one, so asking both questions of one function costs one
/// construction.
#[derive(Debug)]
pub struct TerminatorView<'c, 'a, T: Target> {
    /// The graph being read narrowly.
    cfg: &'c SsaCfg<'a, T>,
}

impl<T: Target> GraphBase for TerminatorView<'_, '_, T> {
    fn node_count(&self) -> usize {
        self.cfg.node_count()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        self.cfg.node_ids()
    }
}

impl<T: Target> Successors for TerminatorView<'_, '_, T> {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.cfg
            .terminator_successors(node.index())
            .iter()
            .copied()
            .map(NodeId::new)
    }
}

impl<T: Target> Predecessors for TerminatorView<'_, '_, T> {
    fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.cfg
            .terminator_predecessors(node.index())
            .iter()
            .copied()
            .map(NodeId::new)
    }
}

impl<T: Target> RootedGraph for TerminatorView<'_, '_, T> {
    fn entry(&self) -> NodeId {
        NodeId::new(0)
    }
}

impl<T: Target> RootedGraph for SsaCfg<'_, T> {
    fn entry(&self) -> NodeId {
        NodeId::new(0)
    }
}

impl<T: Target> Predecessors for SsaCfg<'_, T> {
    fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.block_predecessors(node.index())
            .iter()
            .copied()
            .map(NodeId::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::verifier::{SsaVerifier, VerifierError, VerifyLevel},
        graph::{GraphBase, Predecessors, Successors},
        ir::{
            SsaVarId,
            block::SsaBlock,
            exception::{BlockRange, SsaExceptionHandler},
            instruction::SsaInstruction,
            ops::SsaOp,
        },
        testing::MockTarget,
    };

    fn block(id: usize, op: SsaOp<MockTarget>) -> SsaBlock<MockTarget> {
        let mut block = SsaBlock::new(id);
        block.add_instruction(SsaInstruction::synthetic(op));
        block
    }

    /// The predecessor relation is the exact inverse of the successor one.
    ///
    /// This invariant lets canonical predecessor inversion
    /// remain a separate derivation: it inverts the same `Successors` impl in
    /// the same node order, so it cannot disagree with this CSR.
    #[test]
    fn predecessor_relation_is_the_exact_inverse_of_the_successor_relation() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(
            0,
            SsaOp::Branch {
                condition: SsaVarId::from_index(0),
                true_target: 1,
                false_target: 2,
            },
        ));
        ssa.add_block(block(1, SsaOp::Jump { target: 3 }));
        ssa.add_block(block(2, SsaOp::Jump { target: 3 }));
        ssa.add_block(block(3, SsaOp::returns(None)));

        let cfg = SsaCfg::from_ssa(&ssa);

        let mut inverted: Vec<Vec<usize>> = vec![Vec::new(); cfg.block_count()];
        for from in 0..cfg.block_count() {
            for to in cfg.block_successors(from) {
                if let Some(row) = inverted.get_mut(*to) {
                    row.push(from);
                }
            }
        }
        for block_idx in 0..cfg.block_count() {
            assert_eq!(
                cfg.block_predecessors(block_idx),
                inverted.get(block_idx).map_or(&[][..], Vec::as_slice),
                "predecessors of B{block_idx} are not the inverted successors"
            );
        }
    }

    /// A doubled branch arm is two edges but one predecessor block.
    ///
    /// Both readings are needed and neither may be dropped: in-degree counting
    /// wants the edges, and "which blocks flow in" wants the block.
    #[test]
    fn a_doubled_branch_arm_is_two_edges_and_one_predecessor_block() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(
            0,
            SsaOp::Branch {
                condition: SsaVarId::from_index(0),
                true_target: 1,
                false_target: 1,
            },
        ));
        ssa.add_block(block(1, SsaOp::returns(None)));

        let cfg = SsaCfg::from_ssa(&ssa);

        assert_eq!(cfg.block_successors(0), &[1, 1], "arity is two");
        assert_eq!(cfg.block_predecessors(1), &[0, 0], "and so is in-degree");
        assert_eq!(
            cfg.predecessor_blocks(1).collect::<Vec<_>>(),
            vec![0],
            "but only one block flows in"
        );
        assert_eq!(cfg.to_predecessor_sets().get(1), Some(&vec![0]));
    }

    /// A self-loop is recorded, because a terminator really does name it.
    #[test]
    fn a_self_loop_is_an_edge_like_any_other() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(0, SsaOp::Jump { target: 1 }));
        ssa.add_block(block(
            1,
            SsaOp::Branch {
                condition: SsaVarId::from_index(0),
                true_target: 1,
                false_target: 2,
            },
        ));
        ssa.add_block(block(2, SsaOp::returns(None)));

        let cfg = SsaCfg::from_ssa(&ssa);
        assert_eq!(cfg.block_predecessors(1), &[0, 1]);
        assert_eq!(cfg.predecessor_blocks(1).collect::<Vec<_>>(), vec![0, 1]);
    }

    /// A stray terminator draws no edges from where control cannot reach it.
    ///
    /// The block is `[Jump 1, Const]`: it *contains* a terminator, but its last
    /// instruction is not one, so control falls off the end rather than taking
    /// the jump. The CFG must not report block 1 as a successor, and the
    /// verifier must name the defect.
    #[test]
    fn a_block_whose_last_instruction_is_not_a_terminator_has_no_edges() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut stray = SsaBlock::new(0);
        stray.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        stray.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        ssa.add_block(stray);
        ssa.add_block(block(1, SsaOp::returns(None)));

        let cfg = SsaCfg::from_ssa(&ssa);
        assert!(
            cfg.block_successors(0).is_empty(),
            "a jump control cannot reach must contribute no edge"
        );
        assert!(cfg.block_predecessors(1).is_empty());

        let errors = SsaVerifier::new(&ssa).verify(VerifyLevel::Standard);
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, VerifierError::TerminatorNotLast { block: 0, .. })),
            "the shape must be diagnosable, not merely edgeless: {errors:?}"
        );
    }

    /// End-to-end check that a small diamond CFG exposes the successors,
    /// predecessors, exits, and graph-trait iterators derived from its
    /// terminators, and that out-of-range block queries return empty.
    #[test]
    fn cfg_extracts_successors_and_predecessors_from_terminators() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(
            0,
            SsaOp::Branch {
                condition: SsaVarId::from_index(0),
                true_target: 1,
                false_target: 2,
            },
        ));
        ssa.add_block(block(1, SsaOp::Jump { target: 3 }));
        ssa.add_block(block(2, SsaOp::returns(None)));
        ssa.add_block(block(3, SsaOp::returns(None)));

        let cfg = SsaCfg::from_ssa(&ssa);

        assert_eq!(cfg.ssa().block_count(), 4);
        assert_eq!(cfg.block_count(), 4);
        assert!(!cfg.is_empty());
        assert_eq!(cfg.block_successors(0), &[1, 2]);
        assert_eq!(cfg.block_predecessors(3), &[1]);
        assert_eq!(cfg.exits(), vec![NodeId::new(2), NodeId::new(3)]);
        assert_eq!(cfg.node_count(), 4);
        assert_eq!(
            cfg.node_ids().collect::<Vec<_>>(),
            vec![
                NodeId::new(0),
                NodeId::new(1),
                NodeId::new(2),
                NodeId::new(3)
            ]
        );
        assert_eq!(
            cfg.successors(NodeId::new(0)).collect::<Vec<_>>(),
            vec![NodeId::new(1), NodeId::new(2)]
        );
        assert_eq!(
            cfg.predecessors(NodeId::new(3)).collect::<Vec<_>>(),
            vec![NodeId::new(1)]
        );
        let none: &[usize] = &[];
        assert_eq!(cfg.block_successors(99), none);
        assert_eq!(cfg.block_predecessors(99), none);
    }

    /// A protected region contributes an edge to the complete relation and
    /// none to the terminator one, which is what keeps arity readable.
    ///
    /// Block 0 begins a protected region and ends in an unconditional `Jump`.
    /// A consumer reading arity asks the terminator relation and sees the one
    /// transfer that is really named; a consumer asking where control can go
    /// sees the handler too, because a throw really does go there.
    #[test]
    fn a_protected_region_adds_an_edge_only_to_the_complete_relation() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(0, SsaOp::Jump { target: 1 }));
        ssa.add_block(block(1, SsaOp::returns(None)));
        ssa.add_block(block(2, SsaOp::returns(None)));
        ssa.set_exception_handlers(vec![SsaExceptionHandler {
            flags: 0,
            try_offset: 0,
            try_length: 1,
            handler_offset: 2,
            handler_length: 1,
            class_token_or_filter: 0,
            protected_range: BlockRange::new(0, 1),
            handler_range: BlockRange::new(2, 3),
            filter_range: None,
            selector: None,
        }]);

        let cfg = SsaCfg::from_ssa(&ssa);

        assert_eq!(
            cfg.terminator_successors(0),
            &[1],
            "an unconditional jump names one transfer, inside a try region or not"
        );
        assert_eq!(
            cfg.block_successors(0),
            &[1, 2],
            "and a throw in it reaches the handler"
        );
        assert!(
            cfg.terminator_predecessors(2).is_empty(),
            "no terminator transfers to a handler"
        );
        assert_eq!(
            cfg.block_predecessors(2),
            &[0],
            "but a throw in the protected block gets there"
        );
        assert!(cfg.is_exception_edge(0, 2));
        assert!(!cfg.is_exception_edge(0, 1));
        // The region itself is untouched, so a consumer that needs it still
        // has it.
        assert_eq!(ssa.exception_handlers().len(), 1);
    }

    /// A CFG over a function with no blocks answers every query without
    /// panicking: it is empty, has no exits, and iterates no nodes.
    #[test]
    fn an_empty_cfg_answers_without_panicking() {
        let ssa = SsaFunction::<MockTarget>::new(0, 0);
        let cfg = SsaCfg::from_ssa(&ssa);

        assert!(cfg.is_empty());
        assert!(cfg.exits().is_empty());
        assert_eq!(cfg.node_ids().count(), 0);
    }
    /// A nested finally chain with an unrelated preservation-only block.
    fn admission_fixture() -> SsaFunction<MockTarget> {
        let mut ssa = SsaFunction::new(0, 0);
        for (id, op) in [
            SsaOp::Jump { target: 1 },
            SsaOp::Leave { target: 8 },
            SsaOp::Jump { target: 3 },
            SsaOp::EndFinally,
            SsaOp::Leave { target: 8 },
            SsaOp::Jump { target: 6 },
            SsaOp::EndFinally,
            SsaOp::Rethrow,
            SsaOp::returns(None),
        ]
        .into_iter()
        .enumerate()
        {
            ssa.add_block(block(id, op));
        }
        ssa.set_exception_handlers(
            [(1, 2, 2, 4), (0, 5, 5, 7)]
                .into_iter()
                .map(
                    |(start, end, handler_start, handler_end)| SsaExceptionHandler {
                        flags: 2,
                        try_offset: 0,
                        try_length: 0,
                        handler_offset: 0,
                        handler_length: 0,
                        class_token_or_filter: 0,
                        protected_range: BlockRange::new(start, end),
                        handler_range: BlockRange::new(handler_start, handler_end),
                        filter_range: None,
                        selector: None,
                    },
                )
                .collect(),
        );
        ssa
    }

    /// Construction preserves exact normal, finally, exceptional and inverse relations.
    #[test]
    fn cfg_preserves_nested_finally_and_exception_edges() {
        let ssa = admission_fixture();
        let cfg = SsaCfg::from_ssa(&ssa);
        let expected: [&[usize]; 9] = [&[1], &[2], &[3], &[5], &[5], &[6], &[8], &[], &[]];
        let complete: [&[usize]; 9] = [&[1, 5], &[2, 5], &[3, 5], &[5], &[5], &[6], &[8], &[], &[]];
        for node in 0..9 {
            assert_eq!(cfg.terminator_successors(node), expected[node]);
            assert_eq!(cfg.block_successors(node), complete[node]);
        }
        assert_eq!(cfg.terminator_predecessors(5), [3, 4]);
        assert_eq!(cfg.block_predecessors(5), [0, 1, 2, 3, 4]);
        assert_eq!(
            cfg.roots().preservation_roots().collect::<Vec<_>>(),
            [3, 6, 7]
        );
    }
}
