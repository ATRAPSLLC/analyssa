//! Liveness over the nodes of any graph, from per-node use and definition sets.
//!
//! [`DataFlowSolver`](crate::analysis::dataflow::DataFlowSolver) runs over an
//! [`SsaFunction`](crate::ir::function::SsaFunction). A host whose question is
//! about something other than SSA variables - the named storage a printed
//! program mentions, the registers a listing reads - still asks the classic
//! backward question over its own graph: which items may be read, along some
//! path from here, before anything writes them. [`BlockLiveness`] answers it
//! for any graph that says where its edges go, with the items numbered by the
//! host.
//!
//! # The equations
//!
//! For every node `n`, with `use(n)` the items `n` reads before writing them
//! and `def(n)` the items it writes:
//!
//! ```text
//! out(n) = union of in(s) for every successor s of n
//! in(n)  = use(n) | (out(n) - def(n))
//! ```
//!
//! The least solution is computed by a worklist seeded in postorder (every
//! node, reachable or not), which visits a node's successors before the node
//! on an acyclic graph; a node whose `in` grows re-queues its predecessors.
//! Each set only grows and is bounded by the universe, so the iteration ends.
//!
//! # Complexity
//!
//! O((V + E) * U / 64 * d), `U` the universe and `d` the loop-nesting depth
//! the postorder seed leaves to iterate; one pass on an acyclic graph.

use std::collections::VecDeque;

use crate::{
    bitset::BitSet,
    graph::{
        NodeId,
        algorithms::traversal::reverse_postorder_all,
        traits::{Predecessors, Successors},
    },
};

/// The items live on entry to and on exit from every node of a graph.
///
/// Built by [`BlockLiveness::compute`]; indexed by the graph's node ids.
#[derive(Debug, Clone)]
pub struct BlockLiveness {
    /// `live_in[n]`: the items some path from `n`'s start reads before any
    /// write.
    live_in: Vec<BitSet>,
    /// `live_out[n]`: the union of `live_in` over `n`'s successors.
    live_out: Vec<BitSet>,
    /// The number of items every set ranges over.
    universe: usize,
}

impl BlockLiveness {
    /// Computes liveness over `graph` from each node's uses and definitions.
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph; its edges are the paths liveness flows back
    ///   along.
    /// * `universe` - The number of items; every set ranges over `0..universe`.
    /// * `uses` - Per node, by index, the items it reads before writing them.
    ///   A node past the slice's end reads nothing.
    /// * `defs` - Per node, by index, the items it writes. A node past the
    ///   slice's end writes nothing.
    ///
    /// # Returns
    ///
    /// The least solution of the liveness equations.
    ///
    /// # Panics
    ///
    /// When a set in `uses` or `defs` does not range over exactly `universe`
    /// items.
    #[must_use]
    pub fn compute<G: Successors + Predecessors>(
        graph: &G,
        universe: usize,
        uses: &[BitSet],
        defs: &[BitSet],
    ) -> Self {
        let count = graph.node_count();
        // Lazy: a node nothing is live at, the usual case, holds no words.
        let mut live_in = vec![BitSet::lazy(universe); count];
        let mut live_out = vec![BitSet::lazy(universe); count];
        for (entry, used) in live_in.iter_mut().zip(uses) {
            entry.union_with(used);
        }

        // Postorder: the reverse of a reverse postorder over every node.
        let mut worklist: VecDeque<NodeId> =
            reverse_postorder_all(graph).into_iter().rev().collect();
        let mut queued = BitSet::full(count);
        // What flows through a node from its end to its start, reused for
        // every visit.
        let mut through = BitSet::lazy(universe);
        while let Some(node) = worklist.pop_front() {
            let index = node.index();
            queued.remove(index);
            let Some(out) = live_out.get_mut(index) else {
                continue;
            };
            out.clear();
            for successor in graph.successors(node) {
                if let Some(entry) = live_in.get(successor.index()) {
                    out.union_with(entry);
                }
            }
            through.clear();
            through.union_with(out);
            if let Some(defined) = defs.get(index) {
                through.difference_with(defined);
            }
            let grew = live_in
                .get_mut(index)
                .is_some_and(|entry| entry.union_with(&through));
            if grew {
                for predecessor in graph.predecessors(node) {
                    let at = predecessor.index();
                    if at < count && queued.insert(at) {
                        worklist.push_back(predecessor);
                    }
                }
            }
        }
        Self {
            live_in,
            live_out,
            universe,
        }
    }

    /// Returns the items live where `node` starts.
    ///
    /// # Arguments
    ///
    /// * `node` - A node of the graph the liveness was computed over.
    ///
    /// # Returns
    ///
    /// The set, or `None` for a node the graph did not have.
    #[must_use]
    pub fn live_in(&self, node: NodeId) -> Option<&BitSet> {
        self.live_in.get(node.index())
    }

    /// Returns the items live where `node` ends: those some successor needs.
    ///
    /// # Arguments
    ///
    /// * `node` - A node of the graph the liveness was computed over.
    ///
    /// # Returns
    ///
    /// The set, or `None` for a node the graph did not have.
    #[must_use]
    pub fn live_out(&self, node: NodeId) -> Option<&BitSet> {
        self.live_out.get(node.index())
    }

    /// Returns the number of items the sets range over.
    #[must_use]
    pub const fn universe(&self) -> usize {
        self.universe
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        bitset::BitSet,
        graph::{DirectedGraph, NodeId, algorithms::liveness::BlockLiveness},
    };

    /// A set over `universe` items holding `items`.
    fn set(universe: usize, items: &[usize]) -> BitSet {
        let mut out = BitSet::new(universe);
        for item in items {
            out.insert(*item);
        }
        out
    }

    /// The items of `set`, ascending.
    fn items(set: Option<&BitSet>) -> Vec<usize> {
        set.map(|set| set.iter().collect()).unwrap_or_default()
    }

    /// A graph of `count` nodes and the given edges.
    fn graph(
        count: usize,
        edges: &[(usize, usize)],
    ) -> (DirectedGraph<'static, (), ()>, Vec<NodeId>) {
        let mut graph = DirectedGraph::new();
        let nodes: Vec<NodeId> = (0..count).map(|_| graph.add_node(())).collect();
        for (from, to) in edges {
            graph
                .add_edge(NodeId::new(*from), NodeId::new(*to), ())
                .unwrap();
        }
        (graph, nodes)
    }

    /// A read flows back along a chain to the write that kills it.
    #[test]
    fn a_read_is_live_back_to_its_write() {
        // 0 defines x, 1 passes, 2 reads x.
        let (graph, nodes) = graph(3, &[(0, 1), (1, 2)]);
        let uses = [set(1, &[]), set(1, &[]), set(1, &[0])];
        let defs = [set(1, &[0]), set(1, &[]), set(1, &[])];
        let live = BlockLiveness::compute(&graph, 1, &uses, &defs);
        assert_eq!(items(live.live_in(nodes[2])), vec![0]);
        assert_eq!(items(live.live_in(nodes[1])), vec![0]);
        assert_eq!(items(live.live_out(nodes[0])), vec![0]);
        assert!(items(live.live_in(nodes[0])).is_empty());
    }

    /// A write on one arm of a diamond leaves the read live through the other.
    #[test]
    fn a_write_on_one_arm_does_not_kill() {
        // 0 -> 1 (writes x) -> 3 (reads x); 0 -> 2 -> 3.
        let (graph, nodes) = graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let uses = [set(1, &[]), set(1, &[]), set(1, &[]), set(1, &[0])];
        let defs = [set(1, &[]), set(1, &[0]), set(1, &[]), set(1, &[])];
        let live = BlockLiveness::compute(&graph, 1, &uses, &defs);
        assert!(items(live.live_in(nodes[1])).is_empty());
        assert_eq!(items(live.live_in(nodes[2])), vec![0]);
        assert_eq!(items(live.live_in(nodes[0])), vec![0]);
    }

    /// A value read at a loop's head and written later in its body is live
    /// around the back edge, and so on entry to the body.
    #[test]
    fn a_loop_carries_what_its_head_reads() {
        // 0 -> 1 (reads x) -> 2 (writes x) -> 1; 1 -> 3.
        let (graph, nodes) = graph(4, &[(0, 1), (1, 2), (2, 1), (1, 3)]);
        let uses = [set(1, &[]), set(1, &[0]), set(1, &[]), set(1, &[])];
        let defs = [set(1, &[0]), set(1, &[]), set(1, &[0]), set(1, &[])];
        let live = BlockLiveness::compute(&graph, 1, &uses, &defs);
        assert_eq!(items(live.live_out(nodes[2])), vec![0]);
        assert_eq!(items(live.live_in(nodes[1])), vec![0]);
        assert!(items(live.live_in(nodes[2])).is_empty());
        assert!(items(live.live_in(nodes[0])).is_empty());
    }

    /// A self loop that reads before writing keeps the item live on entry.
    #[test]
    fn a_self_loop_reads_its_own_write() {
        let (graph, nodes) = graph(2, &[(0, 0), (0, 1)]);
        let uses = [set(2, &[0]), set(2, &[1])];
        let defs = [set(2, &[0]), set(2, &[])];
        let live = BlockLiveness::compute(&graph, 2, &uses, &defs);
        assert_eq!(items(live.live_in(nodes[0])), vec![0, 1]);
        assert_eq!(items(live.live_out(nodes[0])), vec![0, 1]);
    }

    /// An irreducible cycle, entered at either node, converges to the same
    /// answer for both.
    #[test]
    fn an_irreducible_cycle_converges() {
        // 0 -> 1, 0 -> 2, 1 <-> 2, 2 -> 3 (reads x); 1 writes y, 3 reads y.
        let (graph, nodes) = graph(4, &[(0, 1), (0, 2), (1, 2), (2, 1), (2, 3)]);
        let uses = [set(2, &[]), set(2, &[]), set(2, &[]), set(2, &[0, 1])];
        let defs = [set(2, &[]), set(2, &[1]), set(2, &[]), set(2, &[])];
        let live = BlockLiveness::compute(&graph, 2, &uses, &defs);
        assert_eq!(items(live.live_in(nodes[2])), vec![0, 1]);
        assert_eq!(items(live.live_in(nodes[1])), vec![0]);
        assert_eq!(items(live.live_in(nodes[0])), vec![0, 1]);
    }

    /// A node no path reaches still has its own uses live on entry, and a
    /// node past the sets reads and writes nothing.
    #[test]
    fn unreachable_nodes_and_short_sets() {
        let (graph, nodes) = graph(3, &[(0, 1), (2, 1)]);
        let uses = [set(1, &[]), set(1, &[0])];
        let live = BlockLiveness::compute(&graph, 1, &uses, &[]);
        assert_eq!(items(live.live_in(nodes[2])), vec![0]);
        assert_eq!(items(live.live_in(nodes[0])), vec![0]);
        assert_eq!(live.universe(), 1);
        assert!(live.live_in(NodeId::new(7)).is_none());
    }
}
