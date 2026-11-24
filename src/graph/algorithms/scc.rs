//! Strongly Connected Components (SCC) using Tarjan's algorithm.
//!
//! This module provides Tarjan's algorithm for finding strongly connected
//! components in a directed graph. A strongly connected component is a maximal
//! set of vertices such that there is a path from every vertex to every other
//! vertex in the set.
//!
//! # Algorithm
//!
//! Tarjan's algorithm performs a single DFS pass over the graph while maintaining:
//!
//! - **index\[v\]**: The discovery time (DFS number) of node `v`, assigned
//!   sequentially as nodes are first visited.
//! - **lowlink\[v\]**: The smallest index reachable from `v` via tree edges and at
//!   most one additional edge (a back edge or cross edge to an unprocessed SCC).
//!   Only nodes on the stack are considered reachable; nodes in already-completed
//!   SCCs are excluded from lowlink computation.
//! - **on_stack**: A boolean tracking which nodes are on the recursion stack,
//!   representing the current SCC being built.
//!
//! When the DFS returns from a node and `lowlink[v] == index[v]`, node `v` is the
//! root of a strongly connected component. All nodes on the stack above `v`
//! (including `v` itself) form one SCC. These nodes are popped from the stack
//! and recorded as a component.
//!
//! The algorithm produces SCCs in **reverse topological order**: if there is an
//! edge from SCC A to SCC B, then A appears after B in the output. This property
//! is useful for processing SCCs in dependency order.
//!
//! # Use Cases
//!
//! - **Recursion detection**: Methods that can call each other form an SCC
//! - **Call graph analysis**: Finding mutually recursive function groups
//! - **Dependency analysis**: Detecting circular dependencies
//! - **Dead code elimination**: Unreachable code forms trivial SCCs

use rustc_hash::FxHashSet;

use crate::graph::{NodeId, Successors};

/// Computes the strongly connected components of a directed graph.
///
/// Uses Tarjan's algorithm with a single DFS pass. The algorithm maintains
/// a stack of vertices and assigns each vertex an index and "lowlink" value.
/// When a vertex's lowlink equals its index, it's the root of an SCC.
///
/// # Arguments
///
/// * `graph` - The directed graph to analyze.
///
/// # Returns
///
/// A vector of SCCs, where each SCC is a vector of `NodeId`s. The SCCs are
/// returned in **reverse topological order** (i.e., if there's an edge from
/// SCC A to SCC B, then A appears after B in the result).
///
/// # Complexity
///
/// - Time: O(V + E)
/// - Space: O(V)
///
/// # Algorithm
///
/// 1. Perform DFS, assigning each node an index in discovery order
/// 2. Compute lowlink values (minimum index reachable via DFS subtree + back edges)
/// 3. When `lowlink[v] == index[v]`, v is root of an SCC; pop stack until v
///
/// # Examples
///
/// ```rust
/// use analyssa::graph::{DirectedGraph, NodeId, algorithms::strongly_connected_components};
///
/// // Simple cycle: A -> B -> C -> A
/// let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
/// let a = graph.add_node('A');
/// let b = graph.add_node('B');
/// let c = graph.add_node('C');
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
/// graph.add_edge(c, a, ());
///
/// let sccs = strongly_connected_components(&graph);
/// // All three nodes form a single SCC
/// assert_eq!(sccs.len(), 1);
/// assert_eq!(sccs[0].len(), 3);
/// ```
///
/// # Acyclic Graph Example
///
/// ```rust
/// use analyssa::graph::{DirectedGraph, NodeId, algorithms::strongly_connected_components};
///
/// // DAG: A -> B -> C
/// let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
/// let a = graph.add_node('A');
/// let b = graph.add_node('B');
/// let c = graph.add_node('C');
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
///
/// let sccs = strongly_connected_components(&graph);
/// // Each node is its own SCC (no cycles)
/// assert_eq!(sccs.len(), 3);
/// for scc in &sccs {
///     assert_eq!(scc.len(), 1);
/// }
/// ```
pub fn strongly_connected_components<G>(graph: &G) -> Vec<Vec<NodeId>>
where
    G: Successors,
{
    let mut state = TarjanState::new(graph.node_count());
    for index in 0..graph.node_count() {
        if state.index.get(index).copied().flatten().is_none() {
            state.strongconnect(graph, NodeId::new(index));
        }
    }
    state.sccs
}

/// Internal state for Tarjan's algorithm.
struct TarjanState {
    /// Discovery index for each node (None if not yet visited)
    index: Vec<Option<usize>>,
    /// Lowlink value for each node
    lowlink: Vec<usize>,
    /// Whether a node is currently on the stack
    on_stack: Vec<bool>,
    /// The DFS stack
    stack: Vec<NodeId>,
    /// Current index counter
    current_index: usize,
    /// Collected SCCs
    sccs: Vec<Vec<NodeId>>,
}

/// One suspended DFS frame borrowing its graph's successor iterator.
///
/// An explicit frame keeps native stack depth bounded and uses constant
/// temporary storage per visited node, independently of its outgoing arity.
struct TarjanFrame<I> {
    /// The node whose successors are being visited.
    node: NodeId,
    /// Its borrowed, resumable successor stream.
    successors: I,
}

impl TarjanState {
    /// Initializes discovery, lowlink and stack membership arrays.
    fn new(n: usize) -> Self {
        Self {
            index: vec![None; n],
            lowlink: vec![0; n],
            on_stack: vec![false; n],
            stack: Vec::new(),
            current_index: 0,
            sccs: Vec::new(),
        }
    }

    /// Records first discovery of `v`: stamps its index and lowlink with the
    /// next discovery number, pushes it onto the SCC stack, and marks it
    /// on-stack.
    ///
    /// Called exactly once per node, the moment it turns gray. Seeding lowlink
    /// with the node's own index establishes the baseline that later back edges
    /// and returning children can only lower; a node whose lowlink never drops
    /// below its index is an SCC root.
    ///
    /// # Arguments
    ///
    /// * `v` - The node being entered for the first time.
    fn visit_node(&mut self, v: NodeId) {
        self.stack.push(v);
        let v_idx = v.index();
        if let Some(slot) = self.index.get_mut(v_idx) {
            *slot = Some(self.current_index);
        }
        if let Some(slot) = self.lowlink.get_mut(v_idx) {
            *slot = self.current_index;
        }
        self.current_index = self.current_index.saturating_add(1);
        if let Some(slot) = self.on_stack.get_mut(v_idx) {
            *slot = true;
        }
    }

    /// Lowers `v`'s lowlink toward `candidate`, keeping the minimum.
    ///
    /// This is the single operation that propagates reachability information: it
    /// is called both when `v` sees a back or cross edge to an on-stack node
    /// (folding in that node's discovery index) and when a child frame returns
    /// (folding in the child's finished lowlink). Taking the min in both cases
    /// is what lets an entire cycle collapse to the discovery index of its
    /// earliest member.
    ///
    /// # Arguments
    ///
    /// * `v` - The node whose lowlink is being relaxed.
    /// * `candidate` - A reachable index or child lowlink to fold in.
    fn update_lowlink(&mut self, v: NodeId, candidate: usize) {
        if let Some(slot) = self.lowlink.get_mut(v.index()) {
            *slot = (*slot).min(candidate);
        }
    }

    /// If `v` is an SCC root, pops its component off the stack and records it.
    ///
    /// A node is the root of its strongly connected component exactly when its
    /// lowlink never fell below its own discovery index, meaning no edge out of
    /// its DFS subtree reached anything discovered earlier and still on the
    /// stack. When that holds, every node sitting above `v` on the stack (down
    /// to and including `v`) belongs to one SCC: they are popped, cleared from
    /// `on_stack`, and emitted as a component. Because roots finish in
    /// postorder, this yields SCCs in reverse topological order.
    ///
    /// # Arguments
    ///
    /// * `v` - The node being finished (all successors exhausted).
    fn finish_component_if_root(&mut self, v: NodeId) {
        let v_idx = v.index();
        let v_index = self.index.get(v_idx).copied().flatten();
        if let Some(idx) = v_index
            && self.lowlink.get(v_idx).copied().unwrap_or(usize::MAX) == idx
        {
            let mut scc = Vec::new();
            while let Some(w) = self.stack.pop() {
                if let Some(slot) = self.on_stack.get_mut(w.index()) {
                    *slot = false;
                }
                scc.push(w);
                if w == v {
                    break;
                }
            }
            self.sccs.push(scc);
        }
    }

    /// Explores the DFS tree rooted at `v`, emitting every SCC found within it.
    ///
    /// This is the iterative heart of Tarjan's algorithm. Rather than recursing,
    /// it drives an explicit `work` stack of [`TarjanFrame`]s: the top frame's
    /// successor cursor is advanced one step per loop turn. An unvisited
    /// successor is entered (visited and pushed as a new frame, simulating a
    /// recursive call); an already-visited successor still on the stack relaxes
    /// the current node's lowlink with that successor's discovery index. When a
    /// frame's successors are exhausted it is finished: `finish_component_if_root`
    /// closes any SCC rooted there, the frame is popped, and its lowlink is
    /// folded back into its parent - the exact bookkeeping a return from
    /// recursion would perform.
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph being analyzed, queried for successors on demand.
    /// * `v` - The root of the DFS tree to explore (an as-yet-unvisited node).
    fn strongconnect<G: Successors>(&mut self, graph: &G, v: NodeId) {
        self.visit_node(v);
        let mut work = Vec::new();
        work.push(TarjanFrame {
            node: v,
            successors: graph.successors(v),
        });
        while let Some(frame) = work.last_mut() {
            if let Some(w) = frame.successors.next() {
                let w_idx = w.index();
                if w_idx >= self.index.len() {
                    continue;
                }
                let w_index_visited = self.index.get(w_idx).copied().flatten();
                if w_index_visited.is_none() {
                    self.visit_node(w);
                    work.push(TarjanFrame {
                        node: w,
                        successors: graph.successors(w),
                    });
                } else if self.on_stack.get(w_idx).copied().unwrap_or(false)
                    && let Some(index) = w_index_visited
                {
                    self.update_lowlink(frame.node, index);
                }
                continue;
            }
            let node = frame.node;
            let lowlink = self
                .lowlink
                .get(node.index())
                .copied()
                .unwrap_or(usize::MAX);
            self.finish_component_if_root(node);
            work.pop();
            if let Some(parent) = work.last() {
                self.update_lowlink(parent.node, lowlink);
            }
        }
    }
}

/// Returns the condensation graph: a DAG where each SCC is collapsed to a single node.
///
/// The condensation graph has one node per SCC, with edges representing
/// connections between different SCCs. This is always a DAG (directed acyclic
/// graph) since edges within SCCs are collapsed.
///
/// # Arguments
///
/// * `graph` - The original graph.
/// * `sccs` - The SCCs as returned by `strongly_connected_components`.
///
/// # Returns
///
/// A tuple containing:
/// - A vector mapping each original node to its SCC index
/// - A vector of edges `(from_scc, to_scc)` in the condensation graph
///
/// # Examples
///
/// ```rust
/// use analyssa::graph::{DirectedGraph, NodeId, algorithms::{strongly_connected_components, condensation}};
///
/// let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
/// let a = graph.add_node('A');
/// let b = graph.add_node('B');
/// let c = graph.add_node('C');
/// let d = graph.add_node('D');
///
/// // Cycle A -> B -> A, plus C -> D (separate)
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, a, ());
/// graph.add_edge(a, c, ());
/// graph.add_edge(c, d, ());
///
/// let sccs = strongly_connected_components(&graph);
/// let (node_to_scc, edges) = condensation(&graph, &sccs);
///
/// // A and B are in the same SCC
/// assert_eq!(node_to_scc[a.index()], node_to_scc[b.index()]);
/// ```
pub fn condensation<G>(graph: &G, sccs: &[Vec<NodeId>]) -> (Vec<usize>, Vec<(usize, usize)>)
where
    G: Successors,
{
    let node_count = graph.node_count();

    // Build mapping from node to SCC index
    let mut node_to_scc = vec![0usize; node_count];
    for (scc_idx, scc) in sccs.iter().enumerate() {
        for &node in scc {
            if let Some(slot) = node_to_scc.get_mut(node.index()) {
                *slot = scc_idx;
            }
        }
    }

    // Find edges between different SCCs
    let mut edges = Vec::new();
    let mut seen_edges = FxHashSet::default();

    for i in 0..node_count {
        let from_node = NodeId::new(i);
        let from_scc = node_to_scc.get(i).copied().unwrap_or(0);

        for to_node in graph.successors(from_node) {
            let to_scc = node_to_scc.get(to_node.index()).copied().unwrap_or(0);

            if from_scc != to_scc && seen_edges.insert((from_scc, to_scc)) {
                edges.push((from_scc, to_scc));
            }
        }
    }

    (node_to_scc, edges)
}

#[cfg(test)]
mod tests {
    use rustc_hash::{FxHashMap, FxHashSet};

    use crate::graph::{
        DirectedGraph, NodeId,
        algorithms::scc::{condensation, strongly_connected_components},
    };

    /// An empty graph has no strongly connected components.
    #[test]
    fn test_scc_empty_graph() {
        let graph: DirectedGraph<(), ()> = DirectedGraph::new();
        let sccs = strongly_connected_components(&graph);
        assert!(sccs.is_empty());
    }

    /// A lone node forms one singleton SCC.
    #[test]
    fn test_scc_single_node() {
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        let a = graph.add_node(());

        let sccs = strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], vec![a]);
    }

    /// A self-loop still yields a single singleton SCC, not a repeat.
    #[test]
    fn test_scc_single_node_self_loop() {
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        let a = graph.add_node(());
        graph.add_edge(a, a, ()).unwrap();

        let sccs = strongly_connected_components(&graph);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], vec![a]);
    }

    /// An acyclic chain gives one singleton SCC per node, in reverse topological order.
    #[test]
    fn test_scc_linear_chain() {
        // A -> B -> C (no cycles)
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();

        let sccs = strongly_connected_components(&graph);

        // Each node is its own SCC
        assert_eq!(sccs.len(), 3);
        for scc in &sccs {
            assert_eq!(scc.len(), 1);
        }

        // SCCs are in reverse topological order: C, B, A
        let scc_nodes: Vec<NodeId> = sccs.iter().map(|scc| scc[0]).collect();
        assert_eq!(scc_nodes, vec![c, b, a]);
    }

    /// A three-node cycle collapses into a single SCC containing all three.
    #[test]
    fn test_scc_simple_cycle() {
        // A -> B -> C -> A
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, a, ()).unwrap();

        let sccs = strongly_connected_components(&graph);

        // All three nodes form one SCC
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);

        let scc_set: FxHashSet<NodeId> = sccs[0].iter().copied().collect();
        assert!(scc_set.contains(&a));
        assert!(scc_set.contains(&b));
        assert!(scc_set.contains(&c));
    }

    /// A mutual pair of nodes forms one SCC of size two.
    #[test]
    fn test_scc_two_nodes_cycle() {
        // A <-> B
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();

        let sccs = strongly_connected_components(&graph);

        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 2);
    }

    /// Two disjoint cycles are reported as two separate SCCs.
    #[test]
    fn test_scc_multiple_components() {
        // Two separate cycles: A <-> B and C <-> D
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, c, ()).unwrap();

        let sccs = strongly_connected_components(&graph);

        assert_eq!(sccs.len(), 2);

        // Each SCC has 2 nodes
        for scc in &sccs {
            assert_eq!(scc.len(), 2);
        }
    }

    /// A one-way edge between two cycles keeps them as distinct SCCs.
    #[test]
    fn test_scc_connected_cycles() {
        // Two cycles connected by an edge:
        // A <-> B -> C <-> D
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, c, ()).unwrap();

        let sccs = strongly_connected_components(&graph);

        assert_eq!(sccs.len(), 2);

        // One SCC contains {A, B}, another contains {C, D}
        let mut found_ab = false;
        let mut found_cd = false;

        for scc in &sccs {
            let scc_set: FxHashSet<NodeId> = scc.iter().copied().collect();
            if scc_set.contains(&a) && scc_set.contains(&b) && scc.len() == 2 {
                found_ab = true;
            }
            if scc_set.contains(&c) && scc_set.contains(&d) && scc.len() == 2 {
                found_cd = true;
            }
        }

        assert!(found_ab);
        assert!(found_cd);
    }

    /// A diamond has no back edges, so every node is its own SCC.
    #[test]
    fn test_scc_diamond_no_cycle() {
        // Diamond: A -> B -> D, A -> C -> D (no cycles)
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(a, c, ()).unwrap();
        graph.add_edge(b, d, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();

        let sccs = strongly_connected_components(&graph);

        // Each node is its own SCC
        assert_eq!(sccs.len(), 4);
        for scc in &sccs {
            assert_eq!(scc.len(), 1);
        }
    }

    /// A figure-eight sharing a one-way link yields two SCCs, not one.
    #[test]
    fn test_scc_figure_eight() {
        // Figure-8 pattern: A <-> B, B -> C, C <-> D
        // This creates two SCCs connected through B and C
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, c, ()).unwrap();

        let sccs = strongly_connected_components(&graph);

        // Two SCCs: {A, B} and {C, D}
        assert_eq!(sccs.len(), 2);
    }

    /// SCCs are emitted in reverse topological order of the condensation.
    #[test]
    fn test_scc_reverse_topological_order() {
        // Chain with cycles: (A <-> B) -> (C <-> D) -> E
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');
        let e = graph.add_node('E');

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, c, ()).unwrap();
        graph.add_edge(d, e, ()).unwrap();

        let sccs = strongly_connected_components(&graph);

        assert_eq!(sccs.len(), 3);

        // Find which SCC each node belongs to
        let find_scc =
            |node: NodeId| -> usize { sccs.iter().position(|scc| scc.contains(&node)).unwrap() };

        let scc_ab = find_scc(a);
        let scc_cd = find_scc(c);
        let scc_e = find_scc(e);

        // In reverse topological order: E comes first, then CD, then AB
        assert!(scc_e < scc_cd);
        assert!(scc_cd < scc_ab);
    }

    /// A 100-node ring collapses into a single SCC of all 100 nodes.
    #[test]
    fn test_scc_large_cycle() {
        // Large cycle: 0 -> 1 -> 2 -> ... -> 99 -> 0
        let mut graph: DirectedGraph<usize, ()> = DirectedGraph::new();
        let nodes: Vec<NodeId> = (0..100).map(|i| graph.add_node(i)).collect();

        for i in 0..100 {
            graph.add_edge(nodes[i], nodes[(i + 1) % 100], ()).unwrap();
        }

        let sccs = strongly_connected_components(&graph);

        // All 100 nodes form one SCC
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 100);
    }

    /// A 10,000-node chain completes without stack overflow, proving the driver is iterative.
    #[test]
    fn test_scc_deep_linear_chain_is_iterative() {
        let mut graph: DirectedGraph<usize, ()> = DirectedGraph::new();
        let nodes: Vec<NodeId> = (0..10_000).map(|i| graph.add_node(i)).collect();

        for pair in nodes.windows(2) {
            graph.add_edge(pair[0], pair[1], ()).unwrap();
        }

        let sccs = strongly_connected_components(&graph);

        assert_eq!(sccs.len(), nodes.len());
        assert_eq!(sccs.first().and_then(|scc| scc.first()), nodes.last());
        assert_eq!(sccs.last().and_then(|scc| scc.first()), nodes.first());
    }

    /// Condensation merges a cycle into one super-node and keeps its outgoing edge.
    #[test]
    fn test_condensation_basic() {
        // A <-> B -> C (single edge to C)
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();

        let sccs = strongly_connected_components(&graph);
        let (node_to_scc, edges) = condensation(&graph, &sccs);

        // A and B are in the same SCC
        assert_eq!(node_to_scc[a.index()], node_to_scc[b.index()]);

        // C is in a different SCC
        assert_ne!(node_to_scc[a.index()], node_to_scc[c.index()]);

        // There's one edge in the condensation graph (from {A,B} SCC to {C} SCC)
        assert_eq!(edges.len(), 1);
    }

    /// Disconnected cycles produce a condensation with no inter-SCC edges.
    #[test]
    fn test_condensation_no_edges() {
        // Two disconnected cycles
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, c, ()).unwrap();

        let sccs = strongly_connected_components(&graph);
        let (_, edges) = condensation(&graph, &sccs);

        // No edges between SCCs
        assert!(edges.is_empty());
    }

    /// A chain of SCCs condenses to a linear DAG with one edge per link.
    #[test]
    fn test_condensation_chain() {
        // Chain of SCCs: (A<->B) -> (C<->D) -> E
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');
        let e = graph.add_node('E');

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, c, ()).unwrap();
        graph.add_edge(d, e, ()).unwrap();

        let sccs = strongly_connected_components(&graph);
        let (node_to_scc, edges) = condensation(&graph, &sccs);

        // Verify SCC assignments
        assert_eq!(node_to_scc[a.index()], node_to_scc[b.index()]);
        assert_eq!(node_to_scc[c.index()], node_to_scc[d.index()]);
        assert_ne!(node_to_scc[a.index()], node_to_scc[c.index()]);
        assert_ne!(node_to_scc[c.index()], node_to_scc[e.index()]);

        // Two edges in condensation: AB->CD, CD->E
        assert_eq!(edges.len(), 2);
    }

    /// Edgeless nodes each form their own singleton SCC.
    #[test]
    fn test_scc_disconnected_graph() {
        // Completely disconnected nodes
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let _a = graph.add_node('A');
        let _b = graph.add_node('B');
        let _c = graph.add_node('C');

        let sccs = strongly_connected_components(&graph);

        // Each node is its own SCC
        assert_eq!(sccs.len(), 3);
        for scc in &sccs {
            assert_eq!(scc.len(), 1);
        }
    }

    /// A mixed graph with nested cycles and tails is partitioned into the expected SCCs.
    #[test]
    fn test_scc_complex_structure() {
        // Complex graph with multiple SCCs
        //
        //     +---+
        //     v   |
        // A-->B-->C
        // |   |
        // v   v
        // D<->E-->F
        //         |
        //         v
        //         G
        let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');
        let e = graph.add_node('E');
        let f = graph.add_node('F');
        let g = graph.add_node('G');

        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, b, ()).unwrap(); // B <-> C cycle
        graph.add_edge(a, d, ()).unwrap();
        graph.add_edge(b, e, ()).unwrap();
        graph.add_edge(d, e, ()).unwrap();
        graph.add_edge(e, d, ()).unwrap(); // D <-> E cycle
        graph.add_edge(e, f, ()).unwrap();
        graph.add_edge(f, g, ()).unwrap();

        let sccs = strongly_connected_components(&graph);

        // SCCs: {B, C}, {D, E}, {A}, {F}, {G}
        assert_eq!(sccs.len(), 5);

        // Count SCC sizes
        let mut size_counts = FxHashMap::default();
        for scc in &sccs {
            *size_counts.entry(scc.len()).or_insert(0) += 1;
        }

        // Two SCCs of size 2, three of size 1
        assert_eq!(size_counts.get(&2), Some(&2));
        assert_eq!(size_counts.get(&1), Some(&3));
    }
}
