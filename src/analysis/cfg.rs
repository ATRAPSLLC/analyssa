//! Control flow graph view of SSA functions.
//!
//! This module provides [`SsaCfg`], a lightweight CFG view constructed directly
//! from an [`SsaFunction`] by extracting control flow edges from block terminators.
//!
//! # What the edges are, and are not
//!
//! An edge is a transfer some block's terminator names, and nothing else. A
//! block's successor count is therefore exactly its terminator's arity, which
//! is what lets a consumer read one from the other.
//!
//! Exceptional control flow is **not** here. Control reaches a handler because
//! the runtime dispatched to it, from wherever the throw happened — not from
//! any block's terminator — so an edge for it would be an edge no terminator
//! takes, and every consumer reading arity from successor count would see a
//! conditional branch that does not exist. A consumer that needs handlers
//! connected wants [`EhCfg`](crate::analysis::exceptions::EhCfg), which adds
//! those edges deliberately and says where they came from.
//!
//! # Design
//!
//! `SsaCfg` holds a reference to the SSA function (zero-copy) and caches
//! predecessor/successor lists. It implements the standard graph traits:
//! - [`GraphBase`] - Node count and iteration
//! - [`Successors`] - Forward edge traversal (from terminators)
//! - [`Predecessors`] - Backward edge traversal (computed from successors)
//!
//! This bridges the gap between passes (which receive `SsaFunction`, not the
//! original CIL CFG) and dataflow analyses that require a CFG.
//!
//! # Complexity
//!
//! Construction: O(V + E) time and memory, where E is the number of
//! terminator-derived edges. All queries are O(1) or O(k), where k is the
//! number of adjacent nodes.
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
//! block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
//! ssa.add_block(block);
//! let cfg = SsaCfg::from_ssa(&ssa);
//! assert_eq!(cfg.block_count(), 1);
//! ```

use crate::{
    graph::{GraphBase, NodeId, Predecessors, Successors},
    ir::function::SsaFunction,
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
    /// Precomputed successor lists for each block, flattened: block `b` owns
    /// `successor_values[successor_offsets[b]..successor_offsets[b + 1]]`.
    ///
    /// Flattened rather than `Vec<Vec<usize>>` because a CFG view is rebuilt
    /// repeatedly (dominators, loop analysis, and most passes construct one),
    /// and a vector per block cost two allocations per block every time. The
    /// flattened form is a fixed handful of buffers regardless of block count.
    successor_offsets: Vec<u32>,
    /// Concatenated per-block successor lists; see [`Self::successor_offsets`].
    successor_values: Vec<usize>,
    /// Precomputed predecessor lists, flattened; see [`Self::successor_offsets`].
    predecessor_offsets: Vec<u32>,
    /// Concatenated per-block predecessor lists; see [`Self::predecessor_offsets`].
    predecessor_values: Vec<usize>,
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
    /// block.add_instruction(SsaInstruction::synthetic(SsaOp::Return { value: None }));
    /// ssa_function.add_block(block);
    ///
    /// let cfg = SsaCfg::from_ssa(&ssa_function);
    /// assert_eq!(cfg.node_count(), ssa_function.block_count());
    /// ```
    #[must_use]
    pub fn from_ssa(ssa: &'a SsaFunction<T>) -> Self {
        let block_count = ssa.block_count();

        // Successors go straight into CSR. The terminator loop visits blocks in
        // ascending order and a block's successors are contiguous, so the rows
        // are already grouped by source and no intermediate edge list or
        // scatter cursor is needed on this side.
        let mut successor_offsets: Vec<u32> = vec![0; block_count.saturating_add(1)];
        let mut successor_values: Vec<usize> = Vec::with_capacity(block_count.saturating_mul(2));

        for block_idx in 0..block_count {
            if let Some(block) = ssa.block(block_idx) {
                // One definition of "the terminator", shared with `SsaBlock` and
                // the two target mutators: the last instruction, when it
                // transfers control.
                if let Some(op) = block.control_terminator() {
                    op.for_each_successor(|succ| {
                        // An out-of-range successor contributes no edge at all.
                        if succ < block_count {
                            successor_values.push(succ);
                        }
                    });
                }
            }
            if let Some(slot) = block_idx
                .checked_add(1)
                .and_then(|next| successor_offsets.get_mut(next))
            {
                *slot = u32::try_from(successor_values.len()).unwrap_or(u32::MAX);
            }
        }

        // Predecessors are the inverse, counted and scattered from the finished
        // successor rows. Walking those rows in ascending block order gives each
        // predecessor row its sources in ascending order too, so the relation is
        // the exact inverse of the successor relation, multiplicity included.
        let mut predecessor_offsets: Vec<u32> = vec![0; block_count.saturating_add(1)];
        for to in &successor_values {
            if let Some(slot) = to
                .checked_add(1)
                .and_then(|next| predecessor_offsets.get_mut(next))
            {
                *slot = slot.saturating_add(1);
            }
        }
        let mut running_pred: u32 = 0;
        for slot in &mut predecessor_offsets {
            running_pred = running_pred.saturating_add(*slot);
            *slot = running_pred;
        }

        let mut predecessor_values: Vec<usize> = vec![0; running_pred as usize];
        let mut pred_cursors: Vec<u32> = predecessor_offsets.clone();
        for block_idx in 0..block_count {
            let row = Self::csr_row(&successor_offsets, &successor_values, block_idx);
            for to in row {
                if let Some(cursor) = pred_cursors.get_mut(*to) {
                    if let Some(slot) = predecessor_values.get_mut(*cursor as usize) {
                        *slot = block_idx;
                    }
                    *cursor = cursor.saturating_add(1);
                }
            }
        }

        Self {
            ssa,
            successor_offsets,
            successor_values,
            predecessor_offsets,
            predecessor_values,
        }
    }

    /// Returns one block's slice out of a CSR pair, or an empty slice when the
    /// block does not exist.
    fn csr_row<'s>(offsets: &[u32], values: &'s [usize], block: usize) -> &'s [usize] {
        let Some(start) = offsets.get(block).map(|offset| *offset as usize) else {
            return &[];
        };
        let Some(end) = block
            .checked_add(1)
            .and_then(|next| offsets.get(next))
            .map(|offset| *offset as usize)
        else {
            return &[];
        };
        values.get(start..end).unwrap_or(&[])
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

    /// Returns the successor block indices for a given block.
    ///
    /// One entry per transfer the block's terminator names, in the terminator's
    /// own operand order — so `Branch c, 1, 1` yields `[1, 1]`, and the slice
    /// length is the terminator's arity.
    ///
    /// Reach for [`Self::predecessor_blocks`] when the question is *which
    /// blocks*, rather than how many edges.
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
    pub fn block_successors(&self, block_idx: usize) -> &[usize] {
        Self::csr_row(&self.successor_offsets, &self.successor_values, block_idx)
    }

    /// Returns the predecessor block indices for a given block.
    ///
    /// The exact inverse of [`Self::block_successors`], **multiplicity
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
    pub fn block_predecessors(&self, block_idx: usize) -> &[usize] {
        Self::csr_row(
            &self.predecessor_offsets,
            &self.predecessor_values,
            block_idx,
        )
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
    /// Exit nodes are blocks with no successors (blocks that end in return,
    /// throw, or other terminating instructions).
    ///
    /// # Returns
    ///
    /// A vector of exit node IDs.
    #[must_use]
    pub fn exits(&self) -> Vec<NodeId> {
        let mut exits = Vec::new();
        for idx in 0..self.ssa.block_count() {
            if self.block_successors(idx).is_empty() {
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
    /// This is the invariant that lets `dominators::precompute_predecessors`
    /// remain a separate derivation: it inverts the same `Successors` impl in
    /// the same node order, so it cannot disagree with this CSR.
    #[test]
    fn predecessor_relation_is_the_exact_inverse_of_the_successor_relation() {
        let mut ssa = crate::ir::SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(
            0,
            SsaOp::Branch {
                condition: crate::ir::SsaVarId::from_index(0),
                true_target: 1,
                false_target: 2,
            },
        ));
        ssa.add_block(block(1, SsaOp::Jump { target: 3 }));
        ssa.add_block(block(2, SsaOp::Jump { target: 3 }));
        ssa.add_block(block(3, SsaOp::Return { value: None }));

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
        let mut ssa = crate::ir::SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(
            0,
            SsaOp::Branch {
                condition: crate::ir::SsaVarId::from_index(0),
                true_target: 1,
                false_target: 1,
            },
        ));
        ssa.add_block(block(1, SsaOp::Return { value: None }));

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
        let mut ssa = crate::ir::SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(0, SsaOp::Jump { target: 1 }));
        ssa.add_block(block(
            1,
            SsaOp::Branch {
                condition: crate::ir::SsaVarId::from_index(0),
                true_target: 1,
                false_target: 2,
            },
        ));
        ssa.add_block(block(2, SsaOp::Return { value: None }));

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
        let mut ssa = crate::ir::SsaFunction::<MockTarget>::new(0, 0);
        let mut stray = SsaBlock::new(0);
        stray.add_instruction(SsaInstruction::synthetic(SsaOp::Jump { target: 1 }));
        stray.add_instruction(SsaInstruction::synthetic(SsaOp::Nop));
        ssa.add_block(stray);
        ssa.add_block(block(1, SsaOp::Return { value: None }));

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

    #[test]
    fn cfg_extracts_successors_and_predecessors_from_terminators() {
        let mut ssa = crate::ir::SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(
            0,
            SsaOp::Branch {
                condition: crate::ir::SsaVarId::from_index(0),
                true_target: 1,
                false_target: 2,
            },
        ));
        ssa.add_block(block(1, SsaOp::Jump { target: 3 }));
        ssa.add_block(block(2, SsaOp::Return { value: None }));
        ssa.add_block(block(3, SsaOp::Return { value: None }));

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
        assert_eq!(cfg.block_successors(99), &[]);
        assert_eq!(cfg.block_predecessors(99), &[]);
    }

    /// A protected region contributes no edge, so a terminator's arity is what
    /// its successor count says.
    ///
    /// Block 0 begins a protected region and ends in an unconditional `Jump`.
    /// It has exactly one successor, so a consumer reading successor count as
    /// branch arity sees the unconditional transfer that is really there.
    #[test]
    fn a_protected_region_adds_no_edge() {
        let mut ssa = crate::ir::SsaFunction::<MockTarget>::new(0, 0);
        ssa.add_block(block(0, SsaOp::Jump { target: 1 }));
        ssa.add_block(block(1, SsaOp::Return { value: None }));
        ssa.add_block(block(2, SsaOp::Return { value: None }));
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
        }]);

        let cfg = SsaCfg::from_ssa(&ssa);

        assert_eq!(
            cfg.block_successors(0),
            &[1],
            "an unconditional jump has one successor, inside a try region or not"
        );
        assert!(
            cfg.block_predecessors(2).is_empty(),
            "the runtime enters a handler; no terminator does"
        );
        // The region itself is untouched, so a consumer that needs it still
        // has it.
        assert_eq!(ssa.exception_handlers().len(), 1);
    }

    #[test]
    fn an_empty_cfg_answers_without_panicking() {
        let ssa = crate::ir::SsaFunction::<MockTarget>::new(0, 0);
        let cfg = SsaCfg::from_ssa(&ssa);

        assert!(cfg.is_empty());
        assert!(cfg.exits().is_empty());
        // Traversal order is a rooted question, and this graph has no root of
        // its own -- `EhCfg` is what carries one.
        assert_eq!(cfg.node_ids().count(), 0);
    }
}
