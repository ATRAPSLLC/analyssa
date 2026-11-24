//! What the walk reads of the function and never changes: the graph, its
//! dominance and loops, and what the caller said of its blocks.

use std::collections::BTreeMap;

use crate::{
    analysis::{
        loops::LoopForest,
        structure::{
            BlockSet, CaseRelation, CaseTest, ProtectedRegion, Region, convergence::Convergence,
            endings::Endings, inputs::Statements,
        },
    },
    graph::{
        GraphBase, NodeId, Predecessors, Successors,
        algorithms::{DominatorTree, PostDominatorTree},
    },
};

/// What a walk reads of the function it structures, decided before the walk
/// begins and never changed by it.
pub struct Facts<'a, G> {
    /// The graph being structured.
    pub graph: &'a G,

    /// Node count of [`Self::graph`], and so the index domain every
    /// [`BlockSet`] here shares.
    pub node_count: usize,

    /// The blocks that begin code ending the function, which a walk places
    /// again wherever a path runs into it.
    pub endings: Endings,

    /// Dominance from the entry.
    pub dominators: DominatorTree,

    /// Post-dominance over every path, used to find conditional joins where
    /// [`Self::returning`] has no answer.
    pub post_dominators: PostDominatorTree,

    /// Post-dominance over the paths that return, when some block ends the
    /// function without returning ([`StructureOptions::no_return`](crate::analysis::structure::StructureOptions::no_return)).
    pub returning: Option<Convergence>,

    /// Natural loops of the graph.
    pub loops: &'a LoopForest,

    /// Loop header to its index in [`LoopForest::loops`].
    pub headers: BTreeMap<NodeId, usize>,

    /// Protected regions to recover, normalised to [`Self::node_count`].
    pub regions: &'a [ProtectedRegion],

    /// Every block a protected region or one of its handlers holds, which a
    /// tail is never repeated across; see [`Structurer::tail`](crate::analysis::structure::walk::Structurer::tail).
    pub guarded: BlockSet,

    /// Blocks that compute only their own branch condition.
    pub condition_only: BlockSet,

    /// Blocks whose statements the caller renders inside a condition; see
    /// [`StructureOptions::condition_expressible`](crate::analysis::structure::StructureOptions::condition_expressible).
    pub condition_expressible: BlockSet,

    /// Blocks whose one statement is the operand of their own test; see
    /// [`StructureOptions::condition_operand`](crate::analysis::structure::StructureOptions::condition_operand).
    pub condition_operand: BlockSet,

    /// See [`StructureOptions::edge_blocks`](crate::analysis::structure::StructureOptions::edge_blocks).
    pub edge_blocks: BlockSet,

    /// See [`Inputs::fold_statements`](crate::analysis::structure::inputs::Inputs::fold_statements).
    pub fold_statements: bool,

    /// See [`Inputs::fall_out`](crate::analysis::structure::inputs::Inputs::fall_out).
    pub fall_out: bool,

    /// See [`Inputs::switch_tails`](crate::analysis::structure::inputs::Inputs::switch_tails).
    pub switch_tails: bool,

    /// The value test each block's branch makes; see
    /// [`StructureOptions::case_tests`](crate::analysis::structure::StructureOptions::case_tests).
    pub case_tests: BTreeMap<NodeId, CaseTest>,

    /// Tails to take as joins where this walk can; see
    /// [`Inputs::preferred_joins`](crate::analysis::structure::inputs::Inputs::preferred_joins).
    pub preferred_joins: BlockSet,

    /// The nesting bound, clamped to at least 1.
    pub max_depth: usize,

    /// See [`StructureOptions::statements`](crate::analysis::structure::StructureOptions::statements).
    pub statements: Statements,
}

impl<G> Facts<'_, G>
where
    G: GraphBase + Successors + Predecessors,
{
    /// Returns the successors of `node` that the graph can actually hold.
    ///
    /// A successor id at or past the node count names no block, so it
    /// contributes no edge - the same rule
    /// [`SsaCfg::from_ssa`](crate::analysis::SsaCfg::from_ssa) states for
    /// itself. Routing every read through here is what keeps an out-of-range id
    /// from reaching a set index or a follow choice.
    pub fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        let limit = self.node_count;
        self.graph
            .successors(node)
            .filter(move |successor| successor.index() < limit)
    }

    /// Returns the nearest block every path from `node` that returns passes
    /// through.
    ///
    /// A block whose every path ends without returning answers by every path
    /// instead, since among its own paths that is where they meet.
    pub fn post_dominator(&self, node: NodeId) -> Option<NodeId> {
        match &self.returning {
            Some(returning) if returning.reaches(node) => returning.meet(node),
            _ => self.post_dominators.immediate_post_dominator(node),
        }
    }

    /// Whether every path from `node` to the function's end passes `join`:
    /// `join` post-dominates `node` ([`Self::post_dominator`]), walked up the
    /// post-dominators at most once per node.
    pub fn runs_on_to(&self, node: NodeId, join: NodeId) -> bool {
        let mut at = node;
        for _ in 0..self.node_count {
            match self.post_dominator(at) {
                Some(next) if next == join => return true,
                Some(next) => at = next,
                None => return false,
            }
        }
        false
    }

    /// Returns how many predecessors of `node` the graph can actually hold.
    pub fn predecessor_count(&self, node: NodeId) -> usize {
        let limit = self.node_count;
        self.graph
            .predecessors(node)
            .filter(|predecessor| predecessor.index() < limit)
            .count()
    }

    /// Returns the blocks reachable from `entry`.
    pub fn reachable_from(&self, entry: NodeId) -> BlockSet {
        let mut seen = BlockSet::new(self.node_count);
        let mut stack = vec![entry];
        while let Some(node) = stack.pop() {
            if seen.insert(node) {
                stack.extend(self.successors(node));
            }
        }
        seen
    }

    /// Returns the distinct successors of `node` with the selector indices that
    /// reach each, in first-selector order.
    ///
    /// Collapsing duplicates is what keeps a branch whose arms both target one
    /// block from reading as a conditional with two identical arms.
    pub fn arms(&self, node: NodeId) -> Vec<(NodeId, Vec<usize>)> {
        let mut arms: Vec<(NodeId, Vec<usize>)> = Vec::new();
        for (index, successor) in self.successors(node).enumerate() {
            match arms.iter_mut().find(|(target, _)| *target == successor) {
                Some((_, selectors)) => selectors.push(index),
                None => arms.push((successor, vec![index])),
            }
        }
        arms
    }

    /// Whether control never runs past the end of `region`: it ends in a
    /// `break`, `continue` or goto, in a block that ends the function, or in a
    /// conditional both of whose arms leave.
    pub fn leaves(&self, region: &Region) -> bool {
        match region {
            Region::Break(_) | Region::Continue(_) | Region::Goto(_) | Region::SwitchBreak(_) => {
                true
            }
            Region::Block(node) => self.successors(*node).next().is_none(),
            Region::Seq(parts) => parts.last().is_some_and(|last| self.leaves(last)),
            Region::If(inner) => {
                self.leaves(&inner.then_branch)
                    && inner
                        .else_branch
                        .as_ref()
                        .is_some_and(|other| self.leaves(other))
            }
            _ => false,
        }
    }

    /// Returns whether control can reach the statement after `region`.
    ///
    /// A `continue` is deliberately not an exit here. It is implicit at a loop
    /// body's tail, so treating it as one would hoist the other arm out of the
    /// `else` and leave behind a `continue` that is no longer in tail position
    /// and can no longer be dropped - longer output, for nothing.
    pub fn leaves_the_sequence(&self, region: &Region) -> bool {
        match region {
            Region::Break(_) | Region::Goto(_) | Region::SwitchBreak(_) => true,
            Region::Block(node) => self.successors(*node).next().is_none(),
            Region::Empty
            | Region::Continue(_)
            | Region::Loop(_)
            | Region::Switch(_)
            | Region::Try(_) => false,
            Region::Seq(parts) => parts
                .last()
                .is_some_and(|last| self.leaves_the_sequence(last)),
            Region::If(inner) => match &inner.else_branch {
                Some(other) => {
                    self.leaves_the_sequence(&inner.then_branch) && self.leaves_the_sequence(other)
                }
                None => false,
            },
        }
    }

    /// Returns where control goes on from `node`: the block a block standing
    /// for one edge ([`StructureOptions::edge_blocks`](crate::analysis::structure::StructureOptions::edge_blocks)) goes on to, and
    /// `node` itself for any other block.
    pub fn through_edge_block(&self, node: NodeId) -> NodeId {
        if !self.edge_blocks.contains(node) {
            return node;
        }
        let mut successors = self.successors(node);
        match (successors.next(), successors.next()) {
            (Some(next), None) => next,
            _ => node,
        }
    }

    /// Returns the block a header leaves the region for when its own test
    /// fails, for a header the recovery may render as the region's condition.
    ///
    /// `None` for a header that cannot be a condition, that is not a two-way
    /// branch, or whose arms do not split into one inside the region and one
    /// outside it - in each case the header names no exit and has no claim on
    /// the choice.
    pub fn tested_exit(&self, header: NodeId, inside: &impl Fn(NodeId) -> bool) -> Option<NodeId> {
        if !self.condition_only.contains(header) && !self.condition_expressible.contains(header) {
            return None;
        }
        let arms = self.arms(header);
        if arms.len() != 2 {
            return None;
        }
        arms.iter()
            .find(|(target, _)| !inside(*target))
            .map(|(target, _)| *target)
            .filter(|_| arms.iter().any(|(target, _)| inside(*target)))
    }

    /// Whether the test at `root` sends its matched value straight to `join`
    /// while the values it lets through meet at a block of their own first.
    pub fn guards_case_tree(&self, root: NodeId, test: &CaseTest, join: Option<NodeId>) -> bool {
        let successors: Vec<NodeId> = self.successors(root).collect();
        let [first_arm, second_arm] = successors.as_slice() else {
            return false;
        };
        let (holds, fails) = match test.holds {
            0 => (*first_arm, *second_arm),
            1 => (*second_arm, *first_arm),
            _ => return false,
        };
        let (matched, unmatched) = match test.relation {
            CaseRelation::Equal => (holds, fails),
            CaseRelation::NotEqual => (fails, holds),
            _ => return false,
        };
        join.is_some_and(|join| {
            matched == join
                && self
                    .post_dominator(unmatched)
                    .is_some_and(|tail| tail != join)
        })
    }
}
