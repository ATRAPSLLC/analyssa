//! Post-dominance and control dependence.
//!
//! Both are the forward algorithms run backwards: a node `p` post-dominates `n`
//! when every path from `n` to the exit passes through `p`, which is dominance
//! on the reversed graph, and control dependence is the dominance frontier of
//! that same reversed graph. This module therefore contributes the *graph* —
//! [`ReverseGraph`] — and leaves [`compute_dominators`] and
//! [`compute_dominance_frontiers`] to do the work they already do.
//!
//! # Why a virtual exit
//!
//! Dominance needs a single root. A real control-flow graph rarely has a single
//! sink: a function returns from several places, and one that does not return at
//! all has none. Both cases are ordinary, not pathological — an infinite server
//! loop and a `noreturn` call are as real as a single `ret`.
//!
//! So the reversed graph is rooted at a **virtual exit**, numbered one past the
//! last real node, and every *terminal* strongly connected component gets one
//! edge to it.
//!
//! An SCC is terminal when no member has a successor that is in range and
//! outside the component. A terminal singleton with no successors is exactly a
//! real sink, so that case covers multiple returns; every other terminal SCC is
//! a region that reaches no sink at all — an infinite loop. Since the
//! condensation is a finite DAG every node reaches some terminal SCC, and
//! inside an SCC every member reaches every other, so one edge per terminal SCC
//! makes every real node backwards-reachable from the exit. One rule, both
//! cases.
//!
//! # What is a convention rather than a theorem
//!
//! Post-dominance *inside* a region that reaches no sink is genuinely
//! undefined: with `1 <-> 2` and no way out, seeding at 1 or at 2 swaps their
//! immediate post-dominators, and neither answer is more correct. The seed
//! *set* is a graph invariant — independent of edge insertion order and of
//! Tarjan's traversal — but the representative is a convention: the lowest id
//! in the component. [`ReverseGraph::exit_seeds`] exposes what was chosen. For
//! a graph where every node reaches a sink the seeds are exactly the sinks.
//!
//! # Complexity
//!
//! Reversal O(V + E), Tarjan O(V + E), the terminality scan O(V + E), sorting
//! the k seeds O(k log k), then O(V α(V)) for Lengauer-Tarjan.
//!
//! # Reading the results
//!
//! The virtual exit is a real node of [`ReverseGraph`], but it is *not* an
//! answer: [`PostDominatorTree`] and [`ControlDependences`] are indexed by the
//! original nodes only, carry exactly `node_count` rows, and never hand back
//! the exit. Reach for [`PostDominatorTree::as_dominator_tree`] on the rare
//! occasion the exit itself is the question.

use crate::{
    bitset::BitSet,
    graph::{
        NodeId,
        algorithms::{
            dominators::{DominatorTree, compute_dominance_frontiers, compute_dominators},
            scc::strongly_connected_components,
        },
        traits::{GraphBase, Successors},
    },
};

/// A control-flow graph reversed and rooted at a virtual exit.
///
/// Node ids are the original graph's, plus one more — [`Self::virtual_exit`] —
/// standing for "control left the function".
#[derive(Debug, Clone)]
pub struct ReverseGraph {
    /// Reverse adjacency: `edges[n]` are the nodes `n` reaches *backwards*.
    edges: Vec<Vec<NodeId>>,
    /// The synthetic root, `graph.node_count()`.
    virtual_exit: NodeId,
    /// One node per terminal SCC, ascending; see [`ReverseGraph::exit_seeds`].
    exit_seeds: Vec<NodeId>,
}

impl ReverseGraph {
    /// Builds the reversed, exit-rooted view of `graph`.
    ///
    /// # Arguments
    ///
    /// * `graph` - The forward control-flow graph.
    #[must_use]
    pub fn new<G: Successors>(graph: &G) -> Self {
        let count = graph.node_count();
        let virtual_exit = NodeId::new(count);
        let mut edges: Vec<Vec<NodeId>> = vec![Vec::new(); count.saturating_add(1)];

        // Reverse every real edge.
        for node in graph.node_ids() {
            for successor in graph.successors(node) {
                if let Some(row) = edges.get_mut(successor.index()) {
                    row.push(node);
                }
            }
        }

        // Seed one edge per *terminal* strongly connected component.
        //
        // Call an SCC terminal when no member has a successor that is in range
        // and outside the SCC. Then a terminal singleton with no successors is
        // exactly a real sink; every other terminal SCC is a region that can
        // reach no sink at all (every member has a successor, and all of them
        // stay inside). Every node reaches some terminal SCC because the
        // condensation is a finite DAG, and inside an SCC every member reaches
        // every other -- so one edge per terminal SCC makes every real node
        // backwards-reachable from the virtual exit, which is the totality the
        // module docs claim.
        //
        // One rule covers both cases, which is why there is no separate sink
        // pass. Note the direction: "reaches no sink" is a property of forward
        // reachability, and `edges` here holds forward *predecessors*, so it
        // cannot be established by walking `edges` -- the condensation is what
        // answers it.
        let mut scc_of: Vec<usize> = vec![usize::MAX; count];
        let components = strongly_connected_components(graph);
        for (component_index, component) in components.iter().enumerate() {
            for member in component {
                if let Some(slot) = scc_of.get_mut(member.index()) {
                    *slot = component_index;
                }
            }
        }

        let mut seeds: Vec<NodeId> = Vec::new();
        for (component_index, component) in components.iter().enumerate() {
            let leaves = component.iter().any(|member| {
                graph.successors(*member).any(|successor| {
                    // Out-of-range successor ids are ignored here exactly as the
                    // reversal above ignores them.
                    successor.index() < count
                        && scc_of.get(successor.index()).copied() != Some(component_index)
                })
            });
            if leaves {
                continue;
            }
            if let Some(seed) = component
                .iter()
                .filter(|member| member.index() < count)
                .min()
            {
                seeds.push(*seed);
            }
        }

        // Ascending, so the seed list is a graph invariant rather than an
        // artefact of Tarjan's traversal order.
        seeds.sort_unstable();
        if let Some(row) = edges.get_mut(virtual_exit.index()) {
            row.extend_from_slice(&seeds);
        }

        Self {
            edges,
            virtual_exit,
            exit_seeds: seeds,
        }
    }

    /// The nodes the virtual exit was given an edge to: one per terminal
    /// strongly connected component.
    ///
    /// Post-dominance *inside* a region that reaches no sink is genuinely
    /// undefined, so which member represents the region is a convention, not a
    /// theorem — choosing member 2 instead of 1 of a `1 <-> 2` cycle swaps their
    /// immediate post-dominators. The convention is "the lowest id in the
    /// component", which depends on the caller's numbering and nothing else.
    /// This exposes it so a caller can see what was chosen.
    #[must_use]
    pub fn exit_seeds(&self) -> &[NodeId] {
        &self.exit_seeds
    }

    /// The number of real nodes, excluding the virtual exit.
    #[must_use]
    pub const fn real_node_count(&self) -> usize {
        self.virtual_exit.index()
    }

    /// The synthetic node standing for "control left the function".
    #[must_use]
    pub const fn virtual_exit(&self) -> NodeId {
        self.virtual_exit
    }
}

impl GraphBase for ReverseGraph {
    fn node_count(&self) -> usize {
        self.edges.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.edges.len()).map(NodeId::new)
    }
}

impl Successors for ReverseGraph {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.edges
            .get(node.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }
}

/// Post-dominance over a graph's real nodes.
///
/// Wraps the dominator tree of the reversed graph and keeps the virtual exit
/// out of every answer, so a caller never has to check for a node id one past
/// the end. When control can leave the function from a block without passing
/// through any other, that block simply has no immediate post-dominator —
/// see [`Self::leaves_function`].
#[derive(Debug, Clone)]
pub struct PostDominatorTree {
    /// Dominance over the reversed, exit-rooted graph.
    tree: DominatorTree,
    /// How many of its nodes are real.
    node_count: usize,
}

impl PostDominatorTree {
    /// The synthetic node standing for "control left the function".
    #[must_use]
    pub const fn virtual_exit(&self) -> NodeId {
        NodeId::new(self.node_count)
    }

    /// The number of real nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// The nearest node that post-dominates `node`, if it is a real one.
    ///
    /// `None` when `node` is out of range or is the virtual exit, and when the
    /// nearest post-dominator is the virtual exit — that is, when control can
    /// leave the function from `node` without passing through any other block.
    /// This is the single guard that keeps the exit out of caller answers.
    #[must_use]
    pub fn immediate_post_dominator(&self, node: NodeId) -> Option<NodeId> {
        if node.index() >= self.node_count {
            return None;
        }
        let idom = self.tree.immediate_dominator(node)?;
        (idom.index() < self.node_count).then_some(idom)
    }

    /// Whether every path from `b` to the exit passes through `a`.
    #[must_use]
    pub fn post_dominates(&self, a: NodeId, b: NodeId) -> bool {
        a.index() < self.node_count && b.index() < self.node_count && self.tree.dominates(a, b)
    }

    /// [`Self::post_dominates`], excluding the reflexive case.
    #[must_use]
    pub fn strictly_post_dominates(&self, a: NodeId, b: NodeId) -> bool {
        a != b && self.post_dominates(a, b)
    }

    /// Whether control can leave the function from `node` without passing
    /// through any other block.
    #[must_use]
    pub fn leaves_function(&self, node: NodeId) -> bool {
        node.index() < self.node_count && self.immediate_post_dominator(node).is_none()
    }

    /// Every real node, for exhaustive assertions.
    #[cfg(test)]
    fn node_ids_for_test(&self) -> impl Iterator<Item = NodeId> {
        (0..self.node_count).map(NodeId::new)
    }

    /// The underlying tree over the reversed graph, virtual exit included.
    ///
    /// For callers that genuinely need the exit — a totality check, say.
    #[must_use]
    pub const fn as_dominator_tree(&self) -> &DominatorTree {
        &self.tree
    }
}

/// Control dependences over a graph's real nodes.
///
/// Carries exactly `node_count` rows, each `node_count` bits wide.
#[derive(Debug, Clone)]
pub struct ControlDependences {
    /// One row per real node.
    rows: Vec<BitSet>,
}

impl ControlDependences {
    /// The number of real nodes, which is also the number of rows.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.rows.len()
    }

    /// The nodes whose branch decides whether `node` executes.
    #[must_use]
    pub fn controllers(&self, node: NodeId) -> Option<&BitSet> {
        self.rows.get(node.index())
    }

    /// Whether `node`'s execution is decided by a branch in `on`.
    #[must_use]
    pub fn is_control_dependent(&self, node: NodeId, on: NodeId) -> bool {
        self.controllers(node)
            .is_some_and(|row| row.contains_checked(on.index()))
    }

    /// Whether `node` is control dependent on itself.
    ///
    /// # Self-dependence
    ///
    /// This really happens, and it is not a defect. Under
    /// Ferrante-Ottenstein-Warren a loop header with a conditional exit lies in
    /// its own post-dominance frontier: the header's own branch decides whether
    /// the header runs again. A consumer that walks [`Self::controllers`] as a
    /// parent pointer into a chain of enclosing conditions will not terminate —
    /// use [`Self::strict_controllers`], which drops the self-edge.
    #[must_use]
    pub fn is_self_dependent(&self, node: NodeId) -> bool {
        self.is_control_dependent(node, node)
    }

    /// [`Self::controllers`] without the self-edge, so a walk terminates.
    pub fn strict_controllers(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.controllers(node)
            .into_iter()
            .flat_map(BitSet::iter)
            .map(NodeId::new)
            .filter(move |controller| *controller != node)
    }

    /// Every row, indexed by node.
    #[must_use]
    pub fn rows(&self) -> &[BitSet] {
        &self.rows
    }
}

impl ReverseGraph {
    /// Post-dominance over the graph this was reversed from.
    ///
    /// Prefer the free [`compute_post_dominators`] unless you already hold a
    /// `ReverseGraph` — to read [`Self::exit_seeds`], say, or because you want
    /// control dependences from the same reversal. This method and
    /// [`Self::control_dependences`] exist so that wanting both costs one
    /// reversal rather than two; the free functions are the ordinary entry
    /// points, matching every other algorithm in this module.
    #[must_use]
    pub fn post_dominators(&self) -> PostDominatorTree {
        PostDominatorTree {
            tree: compute_dominators(self, self.virtual_exit()),
            node_count: self.real_node_count(),
        }
    }

    /// Post-dominance and control dependence together, from one reversal.
    ///
    /// Returned as a pair rather than offering a
    /// `control_dependences(&self, tree)` that takes a caller-supplied tree:
    /// that signature reintroduces an unchecked precondition of exactly the
    /// kind this type removes.
    #[must_use]
    pub fn control_dependences(&self) -> (PostDominatorTree, ControlDependences) {
        let post_dominators = self.post_dominators();
        let frontiers = compute_dominance_frontiers(self, post_dominators.as_dominator_tree());
        let real = self.real_node_count();

        // Rebuild at real-node width rather than truncating the row count and
        // leaving each row `node_count + 1` bits wide: `BitSet` combinators
        // assert equal width, so a wide row handed to a caller aborts against
        // any real-node-width operand -- the same hazard one level down. The
        // copy is exact because the source rows provably hold no bit for the
        // virtual exit: a node enters a frontier row only when it has two or
        // more predecessors, and the exit has none in the reversed graph.
        let rows = frontiers
            .iter()
            .take(real)
            .map(|row| {
                let mut narrowed = BitSet::new(real);
                for bit in row.iter().filter(|bit| *bit < real) {
                    narrowed.insert_checked(bit);
                }
                narrowed
            })
            .collect();

        (post_dominators, ControlDependences { rows })
    }
}

/// Computes the post-dominator tree of `graph`.
///
/// Reverses `graph` and computes dominance on the result. To get control
/// dependences as well, build a [`ReverseGraph`] once and call
/// [`ReverseGraph::control_dependences`], which returns both from a single
/// reversal.
///
/// # Arguments
///
/// * `graph` - The forward control-flow graph.
#[must_use]
pub fn compute_post_dominators<G: Successors>(graph: &G) -> PostDominatorTree {
    ReverseGraph::new(graph).post_dominators()
}

/// Computes the control dependences of `graph`, one row per node.
///
/// `result.controllers(n)` holds the nodes whose branch decides whether `n`
/// executes — the post-dominance frontier of `n`. A node with an empty row runs
/// whenever the function does. See
/// [`ControlDependences::is_self_dependent`] for why a node can appear in its
/// own row.
///
/// This discards the post-dominator tree it had to compute on the way. Call
/// [`ReverseGraph::control_dependences`] instead when you want both.
///
/// # Arguments
///
/// * `graph` - The forward control-flow graph.
#[must_use]
pub fn control_dependences<G: Successors>(graph: &G) -> ControlDependences {
    ReverseGraph::new(graph).control_dependences().1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DirectedGraph;

    /// Builds a graph from an edge list over `n` nodes.
    fn graph(n: usize, edges: &[(usize, usize)]) -> DirectedGraph<'static, (), ()> {
        let mut g = DirectedGraph::new();
        let ids: Vec<NodeId> = (0..n).map(|_| g.add_node(())).collect();
        for (from, to) in edges {
            g.add_edge(ids[*from], ids[*to], ()).expect("edge");
        }
        g
    }

    /// A diamond: both arms are post-dominated by the join, and the branch is
    /// what decides whether either arm runs.
    #[test]
    fn diamond_post_dominates_at_the_join_and_makes_both_arms_control_dependent() {
        // 0 -> {1, 2} -> 3
        let g = graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let pd = compute_post_dominators(&g);
        // 3 post-dominates 0, 1 and 2; the arms post-dominate nothing but
        // themselves.
        assert!(pd.post_dominates(NodeId::new(3), NodeId::new(0)));
        assert!(pd.post_dominates(NodeId::new(3), NodeId::new(1)));
        assert!(!pd.post_dominates(NodeId::new(1), NodeId::new(0)));

        let cd = control_dependences(&g);
        assert!(
            cd.is_control_dependent(NodeId::new(1), NodeId::new(0)),
            "arm 1 is control dependent on the branch"
        );
        assert!(
            cd.is_control_dependent(NodeId::new(2), NodeId::new(0)),
            "arm 2 is control dependent on the branch"
        );
        assert!(
            cd.controllers(NodeId::new(3)).is_some_and(BitSet::is_empty),
            "the join runs whenever the function does: {:?}",
            cd.controllers(NodeId::new(3))
        );
    }

    /// Two returns. Post-dominance is undefined without a single sink, so the
    /// virtual exit is what makes this answerable at all.
    #[test]
    fn several_returns_share_the_virtual_exit() {
        // 0 -> {1, 2}, both terminal.
        let g = graph(3, &[(0, 1), (0, 2)]);
        let pd = compute_post_dominators(&g);
        // Neither return post-dominates the entry — the other path avoids it.
        assert!(!pd.post_dominates(NodeId::new(1), NodeId::new(0)));
        assert!(!pd.post_dominates(NodeId::new(2), NodeId::new(0)));
        // But the virtual exit does, which is the whole point of adding it --
        // and it is reachable only through the underlying tree, never as an
        // ordinary post-dominance answer.
        assert!(
            pd.as_dominator_tree()
                .dominates(pd.virtual_exit(), NodeId::new(0))
        );
        assert!(pd.leaves_function(NodeId::new(1)));
        assert!(pd.leaves_function(NodeId::new(2)));
    }

    /// A function that never returns still has to be analysable: an infinite
    /// loop reaches no sink, so nothing would post-dominate it.
    #[test]
    fn an_exit_less_loop_still_reaches_the_virtual_exit() {
        // 0 -> 1 -> 2 -> 1, with no way out.
        let g = graph(3, &[(0, 1), (1, 2), (2, 1)]);
        let pd = compute_post_dominators(&g);
        // Deliberately asked of the underlying tree: totality is a statement
        // about the virtual exit, which is exactly what `PostDominatorTree`
        // keeps out of ordinary answers.
        let exit = pd.virtual_exit();
        for node in 0..3 {
            assert!(
                pd.as_dominator_tree().dominates(exit, NodeId::new(node)),
                "node {node} is unreachable from the exit, so the tree is partial"
            );
        }
    }

    /// An exit-less region gets **one** edge from the virtual exit, not one per
    /// node inside it.
    ///
    /// `0 -> 1 -> 2 -> 1` has no sink at all, so the whole graph is stranded.
    /// Seeding per node makes nodes 0 and 1 both children of the virtual exit,
    /// which flattens the tree: node 1 stops post-dominating node 0 even though
    /// it is node 0's only successor, and node 1 is reported control dependent
    /// on a block with one unconditional successor and no branch — impossible
    /// under Ferrante-Ottenstein-Warren.
    #[test]
    fn an_exit_less_loop_gets_one_edge_for_the_whole_region() {
        let g = graph(3, &[(0, 1), (1, 2), (2, 1)]);
        let pd = compute_post_dominators(&g);

        assert!(
            pd.post_dominates(NodeId::new(1), NodeId::new(0)),
            "node 1 is node 0's only successor, so it post-dominates it"
        );
        assert!(
            !control_dependences(&g).is_control_dependent(NodeId::new(1), NodeId::new(0)),
            "node 0 has one unconditional successor; it decides nothing"
        );
    }

    /// The shape the structurer degrades on: a diamond whose join is followed
    /// by a loop that never exits. Per-node seeding gave all four nodes their
    /// own exit edge, so nothing post-dominated anything and the diamond lost
    /// its join.
    #[test]
    fn a_diamond_before_an_infinite_loop_still_finds_its_join() {
        // 0 -> {1, 2}; 1 -> 3; 2 -> 3; 3 -> 3.
        let g = graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 3)]);
        let reversed = ReverseGraph::new(&g);
        assert_eq!(
            reversed.exit_seeds(),
            &[NodeId::new(3)],
            "the self-loop is the one terminal component"
        );

        let pd = compute_post_dominators(&g);
        assert!(
            pd.post_dominates(NodeId::new(3), NodeId::new(0)),
            "block 3 is the join both arms reach"
        );
    }

    #[test]
    fn two_disjoint_exit_less_regions_get_one_edge_each() {
        // 0 -> {1, 3}; 1 <-> 2; 3 <-> 4. Two cycles, neither reaching a sink.
        let g = graph(5, &[(0, 1), (0, 3), (1, 2), (2, 1), (3, 4), (4, 3)]);
        let reversed = ReverseGraph::new(&g);
        assert_eq!(reversed.exit_seeds(), &[NodeId::new(1), NodeId::new(3)]);
    }

    /// The common case must be untouched, ordering included.
    #[test]
    fn a_returning_function_seeds_exactly_its_sinks() {
        let g = graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let reversed = ReverseGraph::new(&g);
        assert_eq!(
            reversed.exit_seeds(),
            &[NodeId::new(3)],
            "one sink, and it is the only seed"
        );

        let two_returns = graph(3, &[(0, 1), (0, 2)]);
        assert_eq!(
            ReverseGraph::new(&two_returns).exit_seeds(),
            &[NodeId::new(1), NodeId::new(2)]
        );
    }

    /// The seed set is a property of the graph, not of the order its edges
    /// happened to be added or of Tarjan's traversal.
    #[test]
    fn seeding_is_independent_of_edge_order() {
        let forward = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 2)]);
        let shuffled = graph(5, &[(4, 2), (2, 3), (0, 1), (3, 4), (1, 2)]);
        assert_eq!(
            ReverseGraph::new(&forward).exit_seeds(),
            ReverseGraph::new(&shuffled).exit_seeds()
        );
    }

    /// One seed for a 20 000-node chain ending in a self-loop.
    ///
    /// The seed count is what the cost scales on: an edge per seed leaves
    /// Lengauer-Tarjan with a root of that out-degree, so pinning it at 1 pins
    /// the shape of the work as well as the answer.
    #[test]
    fn a_long_chain_into_a_self_loop_is_seeded_once() {
        const NODES: usize = 20_000;
        let mut edges: Vec<(usize, usize)> = (0..NODES - 1).map(|i| (i, i + 1)).collect();
        edges.push((NODES - 1, NODES - 1));
        let g = graph(NODES, &edges);

        let reversed = ReverseGraph::new(&g);
        assert_eq!(reversed.exit_seeds().len(), 1);
        assert_eq!(reversed.real_node_count(), NODES);
    }

    /// An irreducible graph — two entries into one cycle — is not a shape any
    /// compiler in the corpus emits, and is exactly what hand-written or
    /// obfuscated code does. It must answer, not panic or diverge.
    #[test]
    fn an_irreducible_cycle_is_answered_rather_than_refused() {
        // 0 -> {1, 2}; 1 <-> 2; 2 -> 3.
        let g = graph(4, &[(0, 1), (0, 2), (1, 2), (2, 1), (2, 3)]);
        let pd = compute_post_dominators(&g);
        assert!(pd.post_dominates(NodeId::new(3), NodeId::new(0)));
        let cd = control_dependences(&g);
        assert_eq!(cd.node_count(), 4, "one row per node, and no more");
    }

    /// A single block is the degenerate case every caller hits on a leaf.
    #[test]
    fn a_single_block_post_dominates_only_itself() {
        let g = graph(1, &[]);
        let pd = compute_post_dominators(&g);
        assert!(
            pd.post_dominates(NodeId::new(0), NodeId::new(0)),
            "reflexive"
        );
        assert!(
            pd.leaves_function(NodeId::new(0)),
            "control leaves from the only block, so nothing else post-dominates it"
        );
        assert_eq!(pd.immediate_post_dominator(NodeId::new(0)), None);
        assert!(
            control_dependences(&g)
                .controllers(NodeId::new(0))
                .is_some_and(BitSet::is_empty)
        );
    }

    /// The virtual exit must not be reachable as an ordinary answer.
    ///
    /// Before the wrapper, two of the three in-crate call sites carried an
    /// `index() >= node_count()` check by hand and the third got away without
    /// one only by accident.
    #[test]
    fn the_virtual_exit_is_not_a_post_dominator_answer() {
        for g in [
            graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]),
            graph(3, &[(0, 1), (1, 2), (2, 1)]),
            graph(3, &[(0, 1), (0, 2)]),
        ] {
            let pd = compute_post_dominators(&g);
            let exit = pd.virtual_exit();
            for node in pd.node_ids_for_test() {
                assert_ne!(
                    pd.immediate_post_dominator(node),
                    Some(exit),
                    "the exit leaked out as an immediate post-dominator"
                );
                assert!(!pd.post_dominates(exit, node), "nor as a post-dominator");
                assert!(!pd.post_dominates(node, exit));
            }
            assert_eq!(pd.immediate_post_dominator(exit), None);
        }
    }

    /// Every row is one per node and at real-node width.
    ///
    /// Truncating the row *count* while leaving each row `node_count + 1` bits
    /// wide moves the hazard one level down instead of removing it: `BitSet`
    /// combinators assert equal width, so a wide row aborts against any
    /// real-node-width operand a caller brings.
    #[test]
    fn control_dependences_rows_are_one_per_node_at_real_width() {
        let g = graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let cd = control_dependences(&g);

        assert_eq!(cd.node_count(), 4);
        assert_eq!(cd.rows().len(), 4);
        for row in cd.rows() {
            // `union_with` asserts equal width, so this aborts outright if a
            // row is still `node_count + 1` bits wide. That abort is the whole
            // point of the assertion: it is what a caller would hit.
            let mut real_width = BitSet::new(4);
            real_width.union_with(row);
        }
    }

    /// Following immediate post-dominators from any real node reaches the exit
    /// in at most `node_count` steps — the totality the seeding rule buys.
    #[test]
    fn every_node_reaches_the_exit_through_its_post_dominators() {
        let g = graph(6, &[(0, 1), (1, 2), (2, 3), (3, 1), (1, 4), (4, 5), (5, 5)]);
        let pd = compute_post_dominators(&g);

        for start in 0..6 {
            let mut node = NodeId::new(start);
            let mut steps = 0;
            while let Some(next) = pd.immediate_post_dominator(node) {
                node = next;
                steps += 1;
                assert!(steps <= 6, "cycle in the post-dominator chain from {start}");
            }
        }
    }

    /// A loop whose body is conditional: the body is control dependent on the
    /// header, which is what a structurer reads to place the condition.
    #[test]
    fn a_loop_body_is_control_dependent_on_its_header() {
        // 0 -> 1; 1 -> {2, 3}; 2 -> 1 (latch); 3 terminal.
        let g = graph(4, &[(0, 1), (1, 2), (1, 3), (2, 1)]);
        let cd = control_dependences(&g);
        assert!(
            cd.is_control_dependent(NodeId::new(2), NodeId::new(1)),
            "the body depends on the header's test"
        );
        assert!(
            !cd.is_control_dependent(NodeId::new(0), NodeId::new(1)),
            "the entry does not depend on a later branch"
        );
        // Under Ferrante-Ottenstein-Warren the header lies in its own frontier:
        // its branch decides whether it runs again. A consumer walking the row
        // as a parent chain would not terminate, which is what
        // `strict_controllers` exists for.
        assert!(
            cd.is_self_dependent(NodeId::new(1)),
            "the header's own branch decides whether the header repeats"
        );
        assert!(
            !cd.strict_controllers(NodeId::new(1))
                .any(|controller| controller == NodeId::new(1)),
            "the strict walk drops the self-edge, so it terminates"
        );
    }
}
