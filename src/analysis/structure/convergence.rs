//! Post-dominance over the paths that reach a goal, and only those.

use crate::{
    analysis::structure::BlockSet,
    bitset::BitSet,
    graph::{
        GraphBase, NodeId, Successors,
        algorithms::{PostDominatorTree, compute_post_dominators},
    },
};

/// Post-dominance over the paths that reach a goal, and only those: the paths
/// of a loop body that come round again, or the paths of a function that
/// return.
///
/// A path that ends anywhere else - leaves the loop, calls a function that does
/// not return - is a `break` or a statement that ends its arm on its own, so it
/// has no say in where the paths that continue meet. Post-dominance over the
/// whole graph lets it veto every join it does not pass through.
///
/// Only the blocks a goal is reachable from take part, so building one costs
/// the size of the paths it covers and not the size of the graph: a function
/// of many loops builds one per loop.
pub struct Convergence {
    /// Post-dominance over [`ConvergingGraph`], whose node `i` is
    /// `members[i]`.
    tree: PostDominatorTree,

    /// The blocks a goal is reachable from, goals included, ascending.
    members: Vec<NodeId>,

    /// The synthetic node one past the graph's, a goal when the caller made it
    /// one: every edge back to a loop's header reaches it.
    repeat: NodeId,
}

impl Convergence {
    /// Builds the convergence of the paths over `nodes` toward `goals`.
    ///
    /// `nodes` ascend without repeating. `step` maps each successor of a node
    /// to the node the path continues at: the successor itself, the repeat
    /// node, or `None` where the path ends without reaching a goal. A goal
    /// that is neither among `nodes` nor the repeat node takes no part.
    fn toward<G: Successors>(
        graph: &G,
        node_count: usize,
        nodes: impl Iterator<Item = NodeId>,
        step: impl Fn(NodeId) -> Option<NodeId>,
        goals: impl Iterator<Item = NodeId>,
    ) -> Self {
        let repeat = NodeId::new(node_count);
        let walked: Vec<NodeId> = nodes.filter(|node| node.index() < node_count).collect();
        debug_assert!(
            walked.is_sorted_by(|a, b| a < b),
            "the nodes of a convergence ascend without repeating"
        );
        let mut slots = walked.clone();
        slots.push(repeat);
        let slot_of = |node: NodeId| slots.binary_search(&node).ok();

        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); slots.len()];
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); slots.len()];
        for (from, node) in walked.iter().enumerate() {
            for to in graph
                .successors(*node)
                .filter_map(&step)
                .filter_map(slot_of)
            {
                if let Some(row) = edges.get_mut(from) {
                    row.push(to);
                }
                if let Some(row) = predecessors.get_mut(to) {
                    row.push(from);
                }
            }
        }

        let mut reaches = BitSet::new(slots.len());
        let mut walk: Vec<usize> = goals.filter_map(slot_of).collect();
        while let Some(slot) = walk.pop() {
            if reaches.insert_checked(slot) {
                walk.extend(predecessors.get(slot).into_iter().flatten());
            }
        }

        // Renumber the blocks a goal is reachable from densely, keeping their
        // order, and keep only the edges between them: a block no goal is
        // reachable from has no edge in the converging graph, and the others
        // have none to it.
        let mut position = vec![0usize; slots.len()];
        let mut members = Vec::new();
        for (slot, node) in slots.iter().enumerate() {
            if reaches.contains_checked(slot) {
                if let Some(entry) = position.get_mut(slot) {
                    *entry = members.len();
                }
                members.push(*node);
            }
        }
        let converging: Vec<Vec<NodeId>> = edges
            .iter()
            .enumerate()
            .filter(|(slot, _)| reaches.contains_checked(*slot))
            .map(|(_, row)| {
                row.iter()
                    .filter(|to| reaches.contains_checked(**to))
                    .filter_map(|to| position.get(*to).copied().map(NodeId::new))
                    .collect()
            })
            .collect();
        Self {
            tree: compute_post_dominators(&ConvergingGraph { edges: converging }),
            members,
            repeat,
        }
    }

    /// Builds the convergence of the loop headed at `header` over `body`: the
    /// paths that come round to the header again.
    pub fn of_loop<G: Successors>(
        graph: &G,
        node_count: usize,
        header: NodeId,
        body: &BitSet,
    ) -> Self {
        let repeat = NodeId::new(node_count);
        Self::toward(
            graph,
            node_count,
            body.iter().map(NodeId::new),
            |successor| {
                if successor == header {
                    Some(repeat)
                } else {
                    body.contains_checked(successor.index())
                        .then_some(successor)
                }
            },
            [repeat].into_iter(),
        )
    }

    /// Builds the convergence of a whole function: the paths that end at a
    /// block with no successor outside `no_return`.
    pub fn of_function<G: Successors>(graph: &G, node_count: usize, no_return: &BlockSet) -> Self {
        let nodes = (0..node_count).map(NodeId::new);
        let goals = (0..node_count)
            .map(NodeId::new)
            .filter(|node| graph.successors(*node).next().is_none() && !no_return.contains(*node));
        Self::toward(graph, node_count, nodes, Some, goals)
    }

    /// Returns whether a path from `node` reaches a goal.
    pub fn reaches(&self, node: NodeId) -> bool {
        self.members.binary_search(&node).is_ok()
    }

    /// Returns the nearest block every path from `node` to a goal passes
    /// through, if one does before the goal.
    pub fn meet(&self, node: NodeId) -> Option<NodeId> {
        let position = self.members.binary_search(&node).ok()?;
        let meet = self.members.get(
            self.tree
                .immediate_post_dominator(NodeId::new(position))?
                .index(),
        )?;
        (*meet != self.repeat).then_some(*meet)
    }
}

/// The graph a [`Convergence`] is computed over: only the edges between blocks
/// a goal is reachable from.
pub struct ConvergingGraph {
    /// Successors per block a goal is reachable from, in
    /// [`Convergence::members`] order.
    pub edges: Vec<Vec<NodeId>>,
}

impl GraphBase for ConvergingGraph {
    fn node_count(&self) -> usize {
        self.edges.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.edges.len()).map(NodeId::new)
    }
}

impl Successors for ConvergingGraph {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.edges
            .get(node.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }
}
