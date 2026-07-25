//! Control flow graph view of SSA functions.
//!
//! This module provides [`SsaCfg`], a lightweight CFG view constructed directly
//! from an [`SsaFunction`] by extracting control flow edges from block terminators.
//!
//! # Algorithm
//!
//! Construction proceeds in two phases:
//!
//! 1. **Terminator edge extraction** (O(E)): For each block, the terminator
//!    instruction is found by scanning instructions in reverse. The
//!    `SsaOp::successors()` method provides the set of target blocks. Both
//!    successor and predecessor lists are computed simultaneously.
//!
//! 2. **Exception handler edges** (O(H)): Handler blocks that are only reachable
//!    via runtime exceptions (not explicit branches) get synthetic edges from
//!    their try region's entry block. This ensures analyses like dominator
//!    computation and reachability treat them as connected.
//!
//! # Design
//!
//! `SsaCfg` holds a reference to the SSA function (zero-copy) and caches
//! predecessor/successor lists. It implements the standard graph traits:
//! - [`GraphBase`] - Node count and iteration
//! - [`Successors`] - Forward edge traversal (from terminators)
//! - [`Predecessors`] - Backward edge traversal (computed from successors)
//! - [`RootedGraph`] - Entry node (block 0)
//!
//! This bridges the gap between passes (which receive `SsaFunction`, not the
//! original CIL CFG) and dataflow analyses that require a CFG.
//!
//! # Complexity
//!
//! Construction: O(E + H log H) time, O(E + H) memory, where E is the number of
//! terminator-derived edges and H is the number of exception handler entries.
//! The `log H` is the set used to deduplicate synthetic handler edges against the
//! terminator edges in one pass; checking each handler against the whole edge
//! list instead would be O(H·E), and the handler table is attacker-controlled.
//! All queries are O(1) or O(k) where k is the number of adjacent nodes.
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

use std::collections::BTreeSet;

use crate::{
    graph::{
        GraphBase, NodeId, Predecessors, RootedGraph, Successors,
        algorithms::{postorder, reverse_postorder},
    },
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
    /// Precomputed successor lists for each block (includes exception handler
    /// edges), flattened: block `b` owns
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

        // Collect edges flat, in exactly the order the previous per-block
        // vectors pushed them: terminator edges in block order, then the
        // synthetic handler edges below. Grouping them into CSR afterwards
        // reproduces both lists' contents *and* their ordering.
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(block_count.saturating_mul(2));

        // Build successor/predecessor lists from block terminators
        for block_idx in 0..block_count {
            let Some(block) = ssa.block(block_idx) else {
                continue;
            };
            let terminator = block.instructions().iter().rev().find_map(|instr| {
                let op = instr.op();
                if op.is_terminator() { Some(op) } else { None }
            });
            if let Some(op) = terminator {
                op.for_each_successor(|succ| {
                    // An out-of-range successor contributes no edge at all.
                    if succ < block_count {
                        edges.push((block_idx, succ));
                    }
                });
            }
        }

        // Add synthetic edges for exception handlers. Handler blocks are only
        // reachable via runtime exceptions, not explicit branches, so they
        // appear disconnected in the terminator-based CFG. We add an edge from
        // the try region's entry block to the handler entry block so that
        // analyses (dominator computation, reachability, etc.) treat them as
        // connected.
        //
        // Duplicate suppression is done against a set of the handler pairs rather
        // than by scanning `edges`. A linear `edges.contains(..)` per handler is
        // O(H·E), and the handler table comes from the input binary — while
        // `SsaCfg::from_ssa` is rebuilt by dominators, loop analysis, and most
        // passes, so the cost is paid many times per pipeline run.
        //
        // Scanning only the handler pairs is sufficient: a terminator edge
        // `try_start -> handler_start` can only exist if the try block branches
        // to the handler explicitly, in which case the synthetic edge is
        // genuinely redundant — which the per-destination check below still
        // catches, because that edge is already in `edges` for that destination.
        let mut handler_pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut explicit_targets: BTreeSet<(usize, usize)> = BTreeSet::new();
        for handler in ssa.exception_handlers() {
            if let (Some(try_start), Some(handler_start)) =
                (handler.try_start_block, handler.handler_start_block)
                && handler_start < block_count
                && try_start < block_count
            {
                handler_pairs.insert((try_start, handler_start));
            }
        }
        if !handler_pairs.is_empty() {
            // One pass over the terminator edges to find any that already carry a
            // handler pair, instead of one pass per handler.
            for &edge in &edges {
                if handler_pairs.contains(&edge) {
                    explicit_targets.insert(edge);
                }
            }
            for pair in handler_pairs {
                if !explicit_targets.contains(&pair) {
                    edges.push(pair);
                }
            }
        }

        // Group the flat edge list into CSR: count per endpoint (staged one slot
        // high), prefix-sum in place, then fill through per-block cursors.
        let mut successor_offsets: Vec<u32> = vec![0; block_count.saturating_add(1)];
        let mut predecessor_offsets: Vec<u32> = vec![0; block_count.saturating_add(1)];
        for (from, to) in &edges {
            if let Some(slot) = from
                .checked_add(1)
                .and_then(|next| successor_offsets.get_mut(next))
            {
                *slot = slot.saturating_add(1);
            }
            if let Some(slot) = to
                .checked_add(1)
                .and_then(|next| predecessor_offsets.get_mut(next))
            {
                *slot = slot.saturating_add(1);
            }
        }
        let mut running_succ: u32 = 0;
        for slot in &mut successor_offsets {
            running_succ = running_succ.saturating_add(*slot);
            *slot = running_succ;
        }
        let mut running_pred: u32 = 0;
        for slot in &mut predecessor_offsets {
            running_pred = running_pred.saturating_add(*slot);
            *slot = running_pred;
        }

        let mut successor_values: Vec<usize> = vec![0; running_succ as usize];
        let mut predecessor_values: Vec<usize> = vec![0; running_pred as usize];
        let mut succ_cursors: Vec<u32> = successor_offsets.clone();
        let mut pred_cursors: Vec<u32> = predecessor_offsets.clone();
        for (from, to) in &edges {
            if let Some(cursor) = succ_cursors.get_mut(*from) {
                if let Some(slot) = successor_values.get_mut(*cursor as usize) {
                    *slot = *to;
                }
                *cursor = cursor.saturating_add(1);
            }
            if let Some(cursor) = pred_cursors.get_mut(*to) {
                if let Some(slot) = predecessor_values.get_mut(*cursor as usize) {
                    *slot = *from;
                }
                *cursor = cursor.saturating_add(1);
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
    /// Includes both terminator-derived edges and synthetic exception handler
    /// edges (try entry → handler entry).
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

    /// Returns blocks in postorder traversal.
    ///
    /// Postorder is useful for backward data flow analysis.
    ///
    /// # Returns
    ///
    /// A vector of node IDs in postorder.
    #[must_use]
    pub fn postorder(&self) -> Vec<NodeId> {
        postorder(self, self.entry())
    }

    /// Returns blocks in reverse postorder traversal.
    ///
    /// Reverse postorder is useful for forward data flow analysis.
    ///
    /// # Returns
    ///
    /// A vector of node IDs in reverse postorder.
    #[must_use]
    pub fn reverse_postorder(&self) -> Vec<NodeId> {
        reverse_postorder(self, self.entry())
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

impl<T: Target> RootedGraph for SsaCfg<'_, T> {
    fn entry(&self) -> NodeId {
        NodeId::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::{GraphBase, Predecessors, RootedGraph, Successors},
        ir::{
            block::SsaBlock, exception::SsaExceptionHandler, instruction::SsaInstruction,
            ops::SsaOp,
        },
        testing::MockTarget,
    };

    fn block(id: usize, op: SsaOp<MockTarget>) -> SsaBlock<MockTarget> {
        let mut block = SsaBlock::new(id);
        block.add_instruction(SsaInstruction::synthetic(op));
        block
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
        assert_eq!(cfg.entry(), NodeId::new(0));
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

    #[test]
    fn cfg_adds_exception_handler_edges_without_duplicates() {
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
            try_start_block: Some(0),
            try_end_block: Some(1),
            handler_start_block: Some(2),
            handler_end_block: None,
            filter_start_block: None,
        }]);

        let cfg = SsaCfg::from_ssa(&ssa);

        assert_eq!(cfg.block_successors(0), &[1, 2]);
        assert_eq!(cfg.block_predecessors(2), &[0]);
    }

    #[test]
    fn traversal_helpers_handle_empty_cfg() {
        let ssa = crate::ir::SsaFunction::<MockTarget>::new(0, 0);
        let cfg = SsaCfg::from_ssa(&ssa);

        assert!(cfg.is_empty());
        assert!(cfg.exits().is_empty());
        assert!(cfg.postorder().is_empty());
        assert!(cfg.reverse_postorder().is_empty());
    }
}
