//! Dominator tree computation using the Lengauer-Tarjan algorithm.
//!
//! This module provides efficient dominator tree computation for rooted directed
//! graphs. The dominator tree is a fundamental data structure for:
//!
//! - SSA (Static Single Assignment) construction
//! - Loop detection and analysis
//! - Compiler optimizations
//! - Control flow analysis
//!
//! # Theory
//!
//! A node `d` **dominates** a node `n` if every path from the entry node to `n`
//! must pass through `d`. The **immediate dominator** of `n` (idom(n)) is the
//! unique node that strictly dominates `n` but does not strictly dominate any
//! other dominator of `n`.
//!
//! The dominator tree is formed by making each node's immediate dominator its
//! parent. The entry node is the root (it has no dominator).
//!
//! # Algorithm
//!
//! This implementation uses the Lengauer-Tarjan algorithm with path compression,
//! achieving O(V α(V)) time complexity where α is the inverse Ackermann function
//! (effectively constant for all practical inputs).
//!
//! The algorithm proceeds in four phases:
//!
//! 1. **DFS numbering**: Performs a depth-first search from the entry node,
//!    assigning each reachable node a DFS number (dfnum) and recording the
//!    parent in the DFS tree for each node.
//!
//! 2. **Semidominator computation**: For each node `w` in reverse DFS order,
//!    computes the semidominator `semi(w)` using the semidominator theorem.
//!    The semidominator is the node with minimum DFS number reachable from an
//!    ancestor of `w` via a single edge that bypasses tree edges.
//!
//! 3. **Implicit dominators**: Uses the link-eval data structure (a disjoint-set
//!    forest with path compression) to efficiently evaluate semidominator queries.
//!    For each node `v` in the bucket of its semidominator, determines whether
//!    `idom(v) = semi(v)` (immediate) or must be deferred.
//!
//! 4. **Explicit dominators**: Processes nodes in DFS order, resolving any deferred
//!    immediate dominators from phase 3 into their true values.
//!
//! The link-eval data structure provides the O(alpha(V)) factor through path
//! compression. The `compress` method uses an iterative approach to avoid stack
//! overflow on large graphs, instead of the traditional recursive path compression.

use crate::{
    BitSet,
    graph::{NodeId, RootedGraph, Successors},
};

/// Result of dominator tree computation.
///
/// The dominator tree represents the dominance relationships in a control flow
/// graph. Each node (except the entry) has exactly one immediate dominator.
///
/// # Examples
///
/// ```rust
/// use analyssa::graph::{DirectedGraph, NodeId, algorithms::compute_dominators};
///
/// // Simple CFG: entry -> a -> b -> exit
/// let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
/// let entry = graph.add_node("entry");
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let exit = graph.add_node("exit");
///
/// graph.add_edge(entry, a, ());
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, exit, ());
///
/// let dom_tree = compute_dominators(&graph, entry);
///
/// // entry dominates everything
/// assert!(dom_tree.dominates(entry, exit));
/// // a is the immediate dominator of b
/// assert_eq!(dom_tree.immediate_dominator(b), Some(a));
/// ```
#[derive(Debug, Clone)]
pub struct DominatorTree {
    /// The entry (root) node of the dominator tree
    entry: NodeId,
    /// Immediate dominator for each node (indexed by node ID)
    /// Entry node maps to itself (or we could use Option, but this simplifies queries)
    idom: Vec<NodeId>,
    /// Pre-computed children for each node (indexed by node ID)
    /// `children[i]` is the list of nodes whose immediate dominator is node `i`.
    children: Vec<Vec<NodeId>>,
    /// Number of nodes in the graph
    node_count: usize,
    /// Predecessor lists for the graph this tree was computed from, retained so
    /// [`compute_dominance_frontiers`] can reuse them instead of recomputing the
    /// predecessor relation in a second O(V+E) pass.
    predecessors: Vec<Vec<NodeId>>,
    /// DFS entry time of each node in the dominator tree, or [`Self::UNVISITED`]
    /// for nodes unreachable from the entry. Together with `tout` this turns
    /// [`DominatorTree::dominates`] into two integer comparisons; see
    /// [`compute_dfs_intervals`].
    tin: Vec<usize>,
    /// DFS exit time of each node in the dominator tree, or [`Self::UNVISITED`]
    /// for nodes unreachable from the entry.
    tout: Vec<usize>,
}

impl DominatorTree {
    /// Returns the entry (root) node of the dominator tree.
    #[inline]
    pub fn entry(&self) -> NodeId {
        self.entry
    }

    /// Returns the immediate dominator of a node, or `None` for the entry node
    /// or for nodes whose index is out of bounds.
    ///
    /// The immediate dominator is the closest strict dominator of the node.
    #[inline]
    pub fn immediate_dominator(&self, node: NodeId) -> Option<NodeId> {
        if node == self.entry {
            None
        } else {
            self.idom.get(node.index()).copied()
        }
    }

    /// Marks a node that the dominator-tree DFS never reached, i.e. one
    /// unreachable from the entry.
    const UNVISITED: usize = usize::MAX;

    /// Checks if node `a` dominates node `b`.
    ///
    /// A node dominates itself. The entry node dominates all reachable nodes.
    /// Returns `false` if `b` is unreachable from the entry node.
    ///
    /// # Complexity
    ///
    /// O(1) — two integer comparisons against the dominator tree's DFS
    /// intervals, which [`compute_dominators`] computes once at construction.
    ///
    /// Dominance is ancestry in the dominator tree, and in a DFS numbering `a`
    /// is an ancestor of `b` exactly when `a`'s interval encloses `b`'s. This
    /// used to walk the immediate-dominator chain from `b` to the entry, which
    /// is O(depth); loop detection calls it once per CFG edge, making back-edge
    /// discovery O(E × depth) — near-quadratic on the deep dominator trees that
    /// lifted native code produces, and measurably so (loop-forest construction
    /// grew ~O(n^1.7) in `benches/ssa_repair.rs`).
    pub fn dominates(&self, a: NodeId, b: NodeId) -> bool {
        // Reflexive, and deliberately checked before any bounds test so the
        // out-of-bounds behaviour of the previous chain walk is preserved.
        if a == b {
            return true;
        }

        let (Some(&a_in), Some(&b_in)) = (self.tin.get(a.index()), self.tin.get(b.index())) else {
            return false;
        };
        // An unreachable node neither dominates nor is dominated.
        if a_in == Self::UNVISITED || b_in == Self::UNVISITED {
            return false;
        }
        let (Some(&a_out), Some(&b_out)) = (self.tout.get(a.index()), self.tout.get(b.index()))
        else {
            return false;
        };

        a_in <= b_in && b_out <= a_out
    }

    /// Checks if node `a` strictly dominates node `b`.
    ///
    /// Strict dominance excludes self-dominance: a strictly dominates b iff
    /// a dominates b and a ≠ b.
    #[inline]
    pub fn strictly_dominates(&self, a: NodeId, b: NodeId) -> bool {
        a != b && self.dominates(a, b)
    }

    /// Returns an iterator over all dominators of a node, from the node itself
    /// up to (and including) the entry node.
    ///
    /// A node unreachable from the entry has no dominators, so the iterator
    /// yields only that node itself. It never yields the tree's internal
    /// sentinel, which would be a `NodeId` indexing nothing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::graph::{DirectedGraph, NodeId, algorithms::compute_dominators};
    ///
    /// let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
    /// let entry = graph.add_node(());
    /// let a = graph.add_node(());
    /// let b = graph.add_node(());
    /// graph.add_edge(entry, a, ());
    /// graph.add_edge(a, b, ());
    ///
    /// let dom_tree = compute_dominators(&graph, entry);
    /// let dominators: Vec<NodeId> = dom_tree.dominators(b).collect();
    /// // b is dominated by b, a, and entry
    /// assert_eq!(dominators, vec![b, a, entry]);
    /// ```
    pub fn dominators(&self, node: NodeId) -> DominatorIterator<'_> {
        DominatorIterator {
            tree: self,
            current: Some(node),
        }
    }

    /// Returns the depth of a node in the dominator tree.
    ///
    /// The entry node has depth 0. Returns 0 for nodes whose index is out of
    /// bounds or that are unreachable from the entry.
    pub fn depth(&self, node: NodeId) -> usize {
        let mut depth: usize = 0;
        let mut current = node;
        while current != self.entry {
            let Some(&idom) = self.idom.get(current.index()) else {
                return depth;
            };
            // Sentinel idom (== current) means unreachable: stop walking.
            if idom == current {
                return depth;
            }
            current = idom;
            depth = depth.saturating_add(1);
        }
        depth
    }

    /// Returns all children of a node in the dominator tree.
    ///
    /// Children are nodes whose immediate dominator is the given node.
    ///
    /// # Complexity
    ///
    /// O(1) — children are pre-computed during dominator tree construction.
    pub fn children(&self, node: NodeId) -> &[NodeId] {
        self.children.get(node.index()).map_or(&[], Vec::as_slice)
    }

    /// Returns the number of nodes in the dominator tree.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_count
    }
}

/// Iterator over dominators of a node, from the node up to the entry.
pub struct DominatorIterator<'a> {
    /// Reference to the dominator tree being iterated
    tree: &'a DominatorTree,
    /// Current node in the iteration; `None` when all dominators have been yielded
    current: Option<NodeId>,
}

impl Iterator for DominatorIterator<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current?;

        if current == self.tree.entry {
            self.current = None;
            return Some(current);
        }

        // An unreachable node has no immediate dominator, and `idom` stores the
        // internal sentinel there. Yielding it would hand callers a `NodeId` that
        // indexes nothing — the same sentinel `dominates` already tests for
        // rather than trusting.
        self.current = self
            .tree
            .idom
            .get(current.index())
            .copied()
            .filter(|next| next.index() < self.tree.node_count());
        Some(current)
    }
}

/// Computes the dominator tree for a rooted graph using the Lengauer-Tarjan algorithm.
///
/// This algorithm efficiently computes the immediate dominator for every node
/// reachable from the entry node.
///
/// # Arguments
///
/// * `graph` - The graph to analyze (must implement `RootedGraph`)
///
/// # Returns
///
/// A `DominatorTree` containing the dominator relationships.
///
/// # Complexity
///
/// - Time: O(V α(V)) where α is the inverse Ackermann function
/// - Space: O(V)
///
/// # Algorithm Overview
///
/// The Lengauer-Tarjan algorithm works in several phases:
///
/// 1. **DFS numbering**: Assign DFS numbers to nodes and compute the DFS tree
/// 2. **Semidominators**: Compute semidominators using the Semidominator Theorem
/// 3. **Implicit idom**: Compute implicit immediate dominators
/// 4. **Explicit idom**: Convert implicit to explicit immediate dominators
///
/// # Examples
///
/// ```rust
/// use analyssa::graph::{DirectedGraph, NodeId, algorithms::compute_dominators};
///
/// // Diamond CFG:
/// //      entry
/// //      /   \
/// //     a     b
/// //      \   /
/// //       exit
/// let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
/// let entry = graph.add_node("entry");
/// let a = graph.add_node("a");
/// let b = graph.add_node("b");
/// let exit = graph.add_node("exit");
///
/// graph.add_edge(entry, a, ());
/// graph.add_edge(entry, b, ());
/// graph.add_edge(a, exit, ());
/// graph.add_edge(b, exit, ());
///
/// let dom_tree = compute_dominators(&graph, entry);
///
/// // Entry dominates all nodes
/// assert!(dom_tree.dominates(entry, a));
/// assert!(dom_tree.dominates(entry, b));
/// assert!(dom_tree.dominates(entry, exit));
///
/// // a and b don't dominate exit (there are alternative paths)
/// assert!(!dom_tree.strictly_dominates(a, exit));
/// assert!(!dom_tree.strictly_dominates(b, exit));
///
/// // exit's immediate dominator is entry (not a or b)
/// assert_eq!(dom_tree.immediate_dominator(exit), Some(entry));
/// ```
pub fn compute_dominators<G>(graph: &G, entry: NodeId) -> DominatorTree
where
    G: Successors,
{
    let node_count = graph.node_count();

    if node_count == 0 {
        return DominatorTree {
            entry,
            idom: Vec::new(),
            children: Vec::new(),
            node_count: 0,
            predecessors: Vec::new(),
            tin: Vec::new(),
            tout: Vec::new(),
        };
    }

    // Pre-compute predecessors in a single O(V+E) pass for the Lengauer-Tarjan algorithm
    let predecessors = precompute_predecessors(graph);

    // Lengauer-Tarjan algorithm implementation
    let mut lt = LengauerTarjan::new(node_count, entry);
    lt.compute(graph, &predecessors);

    // Build children list in a single O(V) pass from the idom array
    let mut children: Vec<Vec<NodeId>> = vec![Vec::new(); node_count];
    for i in 0..node_count {
        let node = NodeId::new(i);
        if node == entry {
            continue;
        }
        let Some(parent) = lt.idom.get(i).copied() else {
            continue;
        };
        if let Some(slot) = children.get_mut(parent.index()) {
            slot.push(node);
        }
    }

    let (tin, tout) = compute_dfs_intervals(entry, &children, node_count);

    DominatorTree {
        entry,
        idom: lt.idom,
        children,
        node_count,
        predecessors,
        tin,
        tout,
    }
}

/// Convenience function to compute dominators for a [`RootedGraph`].
///
/// This is equivalent to calling `compute_dominators(graph, graph.entry())`.
pub fn compute_dominators_rooted<G>(graph: &G) -> DominatorTree
where
    G: RootedGraph,
{
    compute_dominators(graph, graph.entry())
}

/// Assigns each node its DFS entry and exit time in the dominator tree.
///
/// The pair `(tin, tout)` is what makes [`DominatorTree::dominates`] O(1): `a`
/// dominates `b` exactly when `a` is an ancestor of `b` in this tree, and in a
/// DFS numbering that is `tin[a] <= tin[b] && tout[b] <= tout[a]`.
///
/// Nodes unreachable from `entry` keep [`DominatorTree::UNVISITED`] in both
/// arrays, which `dominates` rejects.
///
/// The traversal uses an explicit stack rather than recursion: a lifted
/// function can have tens of thousands of blocks in one dominator chain, which
/// would overflow the native stack. Nodes are marked on entry, so a malformed
/// `children` array containing a self-edge or cycle terminates rather than
/// looping forever.
fn compute_dfs_intervals(
    entry: NodeId,
    children: &[Vec<NodeId>],
    node_count: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut tin = vec![DominatorTree::UNVISITED; node_count];
    let mut tout = vec![DominatorTree::UNVISITED; node_count];
    if entry.index() >= node_count {
        return (tin, tout);
    }

    let mut timer: usize = 0;
    if let Some(slot) = tin.get_mut(entry.index()) {
        *slot = timer;
    }
    timer = timer.saturating_add(1);

    // (node, index of the next child to descend into)
    let mut stack: Vec<(NodeId, usize)> = vec![(entry, 0)];
    while let Some(&(node, next_child)) = stack.last() {
        let kids = children
            .get(node.index())
            .map_or([].as_slice(), Vec::as_slice);
        if let Some(&child) = kids.get(next_child) {
            if let Some(top) = stack.last_mut() {
                top.1 = next_child.saturating_add(1);
            }
            if tin.get(child.index()).copied() == Some(DominatorTree::UNVISITED) {
                if let Some(slot) = tin.get_mut(child.index()) {
                    *slot = timer;
                }
                timer = timer.saturating_add(1);
                stack.push((child, 0));
            }
        } else {
            if let Some(slot) = tout.get_mut(node.index()) {
                *slot = timer;
            }
            timer = timer.saturating_add(1);
            stack.pop();
        }
    }

    (tin, tout)
}

/// Pre-computes the predecessor list for all nodes in a single O(V+E) pass.
///
/// Returns a vector where `result[i]` contains all predecessors of node `i`.
fn precompute_predecessors<G: Successors>(graph: &G) -> Vec<Vec<NodeId>> {
    let n = graph.node_count();
    let mut preds: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for i in 0..n {
        let v = NodeId::new(i);
        for succ in graph.successors(v) {
            if let Some(slot) = preds.get_mut(succ.index()) {
                slot.push(v);
            }
        }
    }
    preds
}

/// Internal state for the Lengauer-Tarjan algorithm.
struct LengauerTarjan {
    /// Entry node
    entry: NodeId,
    /// DFS number for each node (0 = not visited)
    dfnum: Vec<usize>,
    /// Node with each DFS number (inverse of dfnum)
    vertex: Vec<NodeId>,
    /// Parent in DFS tree
    parent: Vec<NodeId>,
    /// Semidominator (by DFS number, stored as node ID)
    semi: Vec<NodeId>,
    /// Immediate dominator (final result)
    idom: Vec<NodeId>,
    /// Ancestor in the forest for link-eval
    ancestor: Vec<NodeId>,
    /// Best node on path to ancestor (for path compression)
    best: Vec<NodeId>,
    /// Bucket for each node (nodes whose semidominator is this node)
    bucket: Vec<Vec<NodeId>>,
    /// Current DFS counter
    dfs_counter: usize,
}

impl LengauerTarjan {
    /// Sentinel NodeId representing "uninitialized" or "out-of-graph" values.
    #[inline]
    fn sentinel() -> NodeId {
        NodeId::new(usize::MAX)
    }

    #[inline]
    fn get_node(slice: &[NodeId], i: usize) -> NodeId {
        slice.get(i).copied().unwrap_or(Self::sentinel())
    }

    #[inline]
    fn get_dfnum(&self, n: NodeId) -> usize {
        self.dfnum.get(n.index()).copied().unwrap_or(0)
    }

    #[inline]
    fn set_node(slice: &mut [NodeId], i: usize, value: NodeId) {
        if let Some(slot) = slice.get_mut(i) {
            *slot = value;
        }
    }

    fn new(n: usize, entry: NodeId) -> Self {
        let sentinel = NodeId::new(usize::MAX);
        Self {
            entry,
            dfnum: vec![0; n],
            vertex: vec![sentinel; n],
            parent: vec![sentinel; n],
            semi: (0..n).map(NodeId::new).collect(),
            idom: vec![sentinel; n],
            ancestor: vec![sentinel; n],
            best: (0..n).map(NodeId::new).collect(),
            bucket: vec![Vec::new(); n],
            dfs_counter: 0,
        }
    }

    fn compute<G: Successors>(&mut self, graph: &G, predecessors: &[Vec<NodeId>]) {
        // Phase 1: DFS numbering
        self.dfs(graph, self.entry);

        // Process nodes in reverse DFS order (excluding entry)
        for i in (1..self.dfs_counter).rev() {
            let w = Self::get_node(&self.vertex, i);
            let parent_w = Self::get_node(&self.parent, w.index());

            // Phase 2: Compute semidominators
            // semi(w) = min { v : v -> w is a CFG edge and dfnum(v) < dfnum(w) } ∪
            //           { semi(u) : u -> w via tree edges where dfnum(u) > dfnum(w) }
            let preds_w: &[NodeId] = predecessors.get(w.index()).map_or(&[], Vec::as_slice);
            for v in preds_w {
                let v = *v;
                if self.get_dfnum(v) == 0 {
                    // v is unreachable from entry, skip
                    continue;
                }
                let u = self.eval(v);
                let semi_u = Self::get_node(&self.semi, u.index());
                let semi_w = Self::get_node(&self.semi, w.index());
                if self.get_dfnum(semi_u) < self.get_dfnum(semi_w) {
                    Self::set_node(&mut self.semi, w.index(), semi_u);
                }
            }

            // Add w to bucket of its semidominator
            let semi_w = Self::get_node(&self.semi, w.index());
            if let Some(bucket) = self.bucket.get_mut(semi_w.index()) {
                bucket.push(w);
            }

            // Link w into the forest
            self.link(parent_w, w);

            // Phase 3: Implicitly compute immediate dominators
            // Process bucket of parent(w)
            let bucket = self
                .bucket
                .get_mut(parent_w.index())
                .map_or_else(Vec::new, std::mem::take);
            for v in bucket {
                let u = self.eval(v);
                let semi_u = Self::get_node(&self.semi, u.index());
                let semi_v = Self::get_node(&self.semi, v.index());
                if semi_u == semi_v {
                    // idom(v) = semi(v) = parent(w)
                    Self::set_node(&mut self.idom, v.index(), parent_w);
                } else {
                    // idom(v) = idom(u) (will be computed later)
                    Self::set_node(&mut self.idom, v.index(), u);
                }
            }
        }

        // Phase 4: Explicitly compute immediate dominators
        for i in 1..self.dfs_counter {
            let w = Self::get_node(&self.vertex, i);
            let idom_w = Self::get_node(&self.idom, w.index());
            let semi_w = Self::get_node(&self.semi, w.index());
            if idom_w != semi_w {
                let idom_idom = Self::get_node(&self.idom, idom_w.index());
                Self::set_node(&mut self.idom, w.index(), idom_idom);
            }
        }

        // Entry node dominates itself
        Self::set_node(&mut self.idom, self.entry.index(), self.entry);
    }

    /// DFS traversal to assign DFS numbers and build DFS tree.
    fn dfs<G: Successors>(&mut self, graph: &G, start: NodeId) {
        let mut stack = vec![(start, false)];

        while let Some((node, processed)) = stack.pop() {
            let idx = node.index();

            if processed {
                continue;
            }

            if self.dfnum.get(idx).copied().unwrap_or(0) != 0 {
                continue;
            }

            self.dfs_counter = self.dfs_counter.saturating_add(1);
            if let Some(slot) = self.dfnum.get_mut(idx) {
                *slot = self.dfs_counter;
            }
            // dfs_counter is at least 1 here, so subtracting 1 is safe.
            let vertex_idx = self.dfs_counter.saturating_sub(1);
            Self::set_node(&mut self.vertex, vertex_idx, node);

            for succ in graph.successors(node) {
                if self.get_dfnum(succ) == 0 {
                    Self::set_node(&mut self.parent, succ.index(), node);
                    stack.push((succ, false));
                }
            }
        }
    }

    /// Link v as a child of w in the spanning forest.
    fn link(&mut self, w: NodeId, v: NodeId) {
        Self::set_node(&mut self.ancestor, v.index(), w);
    }

    /// Evaluate: find the node with minimum semidominator on the path to the root.
    fn eval(&mut self, v: NodeId) -> NodeId {
        let sentinel = Self::sentinel();
        if Self::get_node(&self.ancestor, v.index()) == sentinel {
            return v;
        }

        self.compress(v);
        Self::get_node(&self.best, v.index())
    }

    /// Path compression for the forest (iterative).
    ///
    /// Collects the ancestor chain from `v` up to the forest root, then walks
    /// back down to update `best` and `ancestor` for every node on the path.
    /// This avoids O(V) recursion depth that can overflow the stack on large
    /// CFF-obfuscated CFGs (500+ blocks).
    fn compress(&mut self, v: NodeId) {
        let sentinel = Self::sentinel();

        // Phase 1: collect the path from v upward until we reach a node
        // whose ancestor is the forest root (ancestor == sentinel).
        let mut path = Vec::new();
        let mut u = v;
        loop {
            let anc_u = Self::get_node(&self.ancestor, u.index());
            let anc_anc_u = Self::get_node(&self.ancestor, anc_u.index());
            if anc_anc_u == sentinel {
                break;
            }
            path.push(u);
            u = anc_u;
        }

        // Phase 2: walk the path in reverse (top-down) to propagate best
        // values and flatten ancestor pointers — same semantics as the
        // recursive version's post-order updates.
        for &node in path.iter().rev() {
            let ancestor_node = Self::get_node(&self.ancestor, node.index());
            let best_ancestor = Self::get_node(&self.best, ancestor_node.index());
            let best_node = Self::get_node(&self.best, node.index());

            let semi_ba = Self::get_node(&self.semi, best_ancestor.index());
            let semi_bn = Self::get_node(&self.semi, best_node.index());
            if self.get_dfnum(semi_ba) < self.get_dfnum(semi_bn) {
                Self::set_node(&mut self.best, node.index(), best_ancestor);
            }

            let new_anc = Self::get_node(&self.ancestor, ancestor_node.index());
            Self::set_node(&mut self.ancestor, node.index(), new_anc);
        }
    }
}

/// Computes dominance frontiers for all nodes.
///
/// The dominance frontier of a node `n` is the set of all nodes `m` such that:
/// - `n` dominates a predecessor of `m`, but
/// - `n` does not strictly dominate `m`
///
/// Dominance frontiers are essential for placing φ-functions in SSA construction.
///
/// # Arguments
///
/// * `graph` - The control flow graph
/// * `dom_tree` - The precomputed dominator tree
///
/// # Returns
///
/// A vector where `result[i]` contains the dominance frontier of node `i`.
///
/// # Complexity
///
/// - Time: Θ(V + E + Σ|DF(v)|). The runner walk visits one node per frontier
///   membership it records, so the total is the summed frontier size, not
///   `O(V + E)` — a spine `b1→b2→…→bn` where every `bi` also branches to one
///   common join is Θ(V²).
/// - Space: Θ(V + Σ|DF(v)|). Rows are allocated on first insert
///   ([`BitSet::lazy`]), so nodes with an empty frontier cost only a `Vec`
///   header; the dense worst case is still `O(V²)` bits.
///
/// # Examples
///
/// ```rust
/// use analyssa::graph::{DirectedGraph, NodeId, algorithms::{compute_dominators, compute_dominance_frontiers}};
///
/// // Diamond CFG with join point
/// let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
/// let entry = graph.add_node("entry");
/// let left = graph.add_node("left");
/// let right = graph.add_node("right");
/// let join = graph.add_node("join");
///
/// graph.add_edge(entry, left, ());
/// graph.add_edge(entry, right, ());
/// graph.add_edge(left, join, ());
/// graph.add_edge(right, join, ());
///
/// let dom_tree = compute_dominators(&graph, entry);
/// let frontiers = compute_dominance_frontiers(&graph, &dom_tree);
///
/// // The dominance frontier of 'left' includes 'join' (where paths merge)
/// assert!(frontiers[left.index()].contains(join.index()));
/// // The dominance frontier of 'right' includes 'join'
/// assert!(frontiers[right.index()].contains(join.index()));
/// ```
pub fn compute_dominance_frontiers<G>(graph: &G, dom_tree: &DominatorTree) -> Vec<BitSet>
where
    G: Successors,
{
    let n = graph.node_count();
    // Per-node frontier bitsets. Consumers (notably the SSA rebuilder) merge
    // handler dominance frontiers via `BitSet` union, so the set representation
    // has to stay — but the rows are allocated lazily. Real dominance frontiers
    // are sparse: most nodes are not on any runner walk and keep an empty
    // frontier forever, so eagerly allocating `n` rows of `n` bits costs
    // `O(V²)` bytes of memset to store almost nothing.
    //
    // This matters because `SsaRebuilder::merge_handler_dom_trees` calls this
    // once per exception-handler root, and handler tables come from the input
    // binary. A 20k-block function with 500 handler roots would otherwise
    // memset ~50 MB per call, ~25 GB for the function.
    let mut frontiers: Vec<BitSet> = (0..n).map(|_| BitSet::lazy(n)).collect();

    // Reuse the predecessor lists cached on the dominator tree when they match
    // this graph (the normal case — the tree was computed from it), avoiding a
    // second O(V+E) predecessor pass. Fall back to recomputing otherwise.
    let fallback;
    let all_preds: &[Vec<NodeId>] = if dom_tree.predecessors.len() == n {
        &dom_tree.predecessors
    } else {
        fallback = precompute_predecessors(graph);
        &fallback
    };

    // For each node, check if it's a join point (has multiple predecessors)
    // For each join point, walk up the dominator tree from each predecessor
    for (node_idx, preds) in all_preds.iter().enumerate() {
        let node = NodeId::new(node_idx);

        if preds.len() < 2 {
            continue; // Not a join point
        }

        // For each predecessor, walk up its dominators until we reach idom(node)
        let idom_node = dom_tree.immediate_dominator(node);

        for &pred in preds {
            let mut runner = pred;
            // Guard against unreachable nodes (their index may be invalid/sentinel)
            while Some(runner) != idom_node && runner != dom_tree.entry() && runner.index() < n {
                if let Some(slot) = frontiers.get_mut(runner.index()) {
                    slot.insert(node.index());
                }
                if let Some(idom) = dom_tree.immediate_dominator(runner) {
                    // Check for sentinel value (unreachable node)
                    if idom.index() >= n {
                        break;
                    }
                    runner = idom;
                } else {
                    break;
                }
            }
            // Also check entry if needed (guard against invalid index)
            if Some(runner) != idom_node
                && runner == dom_tree.entry()
                && runner.index() < n
                && let Some(slot) = frontiers.get_mut(runner.index())
            {
                slot.insert(node.index());
            }
        }
    }

    frontiers
}

#[cfg(test)]
mod tests {
    use crate::graph::{
        DirectedGraph, NodeId,
        algorithms::dominators::{DominatorTree, compute_dominance_frontiers, compute_dominators},
    };

    /// The dominance test exactly as it was implemented before DFS intervals:
    /// walk the immediate-dominator chain from `b` up to the entry.
    ///
    /// Kept verbatim as the oracle for
    /// `dfs_interval_dominance_matches_chain_walk`. The replacement is an
    /// asymptotic change to a predicate that five analyses and the verifier
    /// depend on, so "the existing tests still pass" is too weak a claim —
    /// equivalence is checked over every node pair of every graph below,
    /// including the out-of-bounds and unreachable arguments the original had
    /// specific behaviour for.
    fn dominates_by_chain_walk(tree: &DominatorTree, a: NodeId, b: NodeId) -> bool {
        if a == b {
            return true;
        }
        if b.index() >= tree.node_count {
            return false;
        }
        let mut current = b;
        while current != tree.entry {
            let Some(&idom) = tree.idom.get(current.index()) else {
                return false;
            };
            if idom == a {
                return true;
            }
            if idom == current {
                return false;
            }
            current = idom;
        }
        a == tree.entry
    }

    /// One dominance test case: a label, a node count, and an edge list.
    type DominanceCase = (&'static str, usize, Vec<(usize, usize)>);

    /// Builds a graph from an edge list over `node_count` nodes.
    fn graph_from(node_count: usize, edges: &[(usize, usize)]) -> DirectedGraph<'static, (), ()> {
        let mut graph: DirectedGraph<'static, (), ()> = DirectedGraph::new();
        for _ in 0..node_count {
            graph.add_node(());
        }
        for &(from, to) in edges {
            let _ = graph.add_edge(NodeId::new(from), NodeId::new(to), ());
        }
        graph
    }

    /// A deterministic LCG, so the generated cases are reproducible across runs.
    fn next_rand(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    /// The O(1) interval test must agree with the O(depth) chain walk on every
    /// node pair — including self-pairs, unreachable nodes, and indices past the
    /// end of the graph.
    /// An unreachable node has no immediate dominator, and the tree stores an
    /// internal sentinel there. Walking the chain must stop rather than hand the
    /// caller a `NodeId` that indexes nothing — `dominates` already tests for
    /// the sentinel instead of trusting it.
    #[test]
    fn dominators_of_an_unreachable_node_yield_only_itself() {
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let reachable = graph.add_node("reachable");
        let orphan = graph.add_node("orphan");
        let _ = graph.add_edge(entry, reachable, ());

        let tree = compute_dominators(&graph, entry);

        let chain: Vec<NodeId> = tree.dominators(orphan).collect();
        assert_eq!(
            chain,
            vec![orphan],
            "an unreachable node dominates only itself and must not yield a sentinel"
        );
        for node in &chain {
            assert!(
                node.index() < graph.node_count(),
                "every yielded id must index a real node, got {node:?}"
            );
        }

        // The reachable chain is unaffected.
        assert_eq!(
            tree.dominators(reachable).collect::<Vec<_>>(),
            vec![reachable, entry]
        );
    }

    #[test]
    fn dfs_interval_dominance_matches_chain_walk() {
        let mut cases: Vec<DominanceCase> = vec![
            ("empty", 0, vec![]),
            ("single", 1, vec![]),
            ("self_loop", 1, vec![(0, 0)]),
            ("chain", 4, vec![(0, 1), (1, 2), (2, 3)]),
            ("diamond", 4, vec![(0, 1), (0, 2), (1, 3), (2, 3)]),
            ("loop", 3, vec![(0, 1), (1, 2), (2, 1)]),
            (
                "nested_loops",
                5,
                vec![(0, 1), (1, 2), (2, 3), (3, 2), (3, 4), (4, 1)],
            ),
            // Irreducible: two distinct entries into the same cycle, the case
            // that makes naive dominance reasoning wrong.
            (
                "irreducible",
                5,
                vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 3)],
            ),
            // Blocks 3 and 4 are unreachable from the entry.
            ("unreachable", 5, vec![(0, 1), (1, 2), (3, 4)]),
            // An unreachable self-loop, which drove the original walk's
            // `idom == current` guard.
            ("unreachable_self_loop", 3, vec![(0, 1), (2, 2)]),
        ];

        // Generated graphs: dense enough to produce deep dominator trees and
        // plenty of back edges.
        let mut state = 0x5DEE_CE66_D1BB_1A3Fu64;
        for case in 0..40 {
            let node_count = 2 + (next_rand(&mut state) as usize % 12);
            let edge_count = 1 + (next_rand(&mut state) as usize % 24);
            let mut edges = Vec::new();
            for _ in 0..edge_count {
                let from = next_rand(&mut state) as usize % node_count;
                let to = next_rand(&mut state) as usize % node_count;
                edges.push((from, to));
            }
            let name: &'static str = Box::leak(format!("generated_{case}").into_boxed_str());
            cases.push((name, node_count, edges));
        }

        for (name, node_count, edges) in cases {
            let graph = graph_from(node_count, &edges);
            let tree = compute_dominators(&graph, NodeId::new(0));

            // Probe two indices past the end to cover out-of-bounds arguments.
            let probe = node_count.saturating_add(2);
            for a in 0..probe {
                for b in 0..probe {
                    let (a, b) = (NodeId::new(a), NodeId::new(b));
                    assert_eq!(
                        tree.dominates(a, b),
                        dominates_by_chain_walk(&tree, a, b),
                        "dominates({}, {}) disagrees with the chain walk in case {name}",
                        a.index(),
                        b.index()
                    );
                }
            }
        }
    }

    /// Dominance must remain a partial order: reflexive, antisymmetric, and
    /// transitive. The interval encoding makes these structural, but a bug in
    /// the DFS numbering would break them, and the oracle above cannot catch a
    /// fault the old implementation shared.
    #[test]
    fn dfs_interval_dominance_is_a_partial_order() {
        let graph = graph_from(
            8,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 4),
                (5, 6),
                (6, 7),
                (7, 3),
            ],
        );
        let tree = compute_dominators(&graph, NodeId::new(0));
        let nodes: Vec<NodeId> = (0..8).map(NodeId::new).collect();

        for &a in &nodes {
            assert!(tree.dominates(a, a), "not reflexive at {}", a.index());
            for &b in &nodes {
                if a != b && tree.dominates(a, b) {
                    assert!(
                        !tree.dominates(b, a),
                        "not antisymmetric: {} and {}",
                        a.index(),
                        b.index()
                    );
                }
                for &c in &nodes {
                    if tree.dominates(a, b) && tree.dominates(b, c) {
                        assert!(
                            tree.dominates(a, c),
                            "not transitive: {} -> {} -> {}",
                            a.index(),
                            b.index(),
                            c.index()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_dominator_empty_graph() {
        let graph: DirectedGraph<(), ()> = DirectedGraph::new();
        // With empty graph, we need a valid entry - this is a degenerate case
        let entry = NodeId::new(0);
        let dom_tree = compute_dominators(&graph, entry);
        assert_eq!(dom_tree.node_count(), 0);
    }

    #[test]
    fn test_dominator_single_node() {
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        let entry = graph.add_node(());

        let dom_tree = compute_dominators(&graph, entry);

        assert_eq!(dom_tree.entry(), entry);
        assert_eq!(dom_tree.immediate_dominator(entry), None);
        assert!(dom_tree.dominates(entry, entry));
        assert_eq!(dom_tree.depth(entry), 0);
    }

    #[test]
    fn test_dominator_linear_chain() {
        // entry -> a -> b -> c
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");

        graph.add_edge(entry, a, ()).unwrap();
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);

        // Check immediate dominators
        assert_eq!(dom_tree.immediate_dominator(entry), None);
        assert_eq!(dom_tree.immediate_dominator(a), Some(entry));
        assert_eq!(dom_tree.immediate_dominator(b), Some(a));
        assert_eq!(dom_tree.immediate_dominator(c), Some(b));

        // Check dominance relationships
        assert!(dom_tree.dominates(entry, a));
        assert!(dom_tree.dominates(entry, b));
        assert!(dom_tree.dominates(entry, c));
        assert!(dom_tree.dominates(a, b));
        assert!(dom_tree.dominates(a, c));
        assert!(dom_tree.dominates(b, c));

        // Check non-dominance
        assert!(!dom_tree.dominates(c, b));
        assert!(!dom_tree.dominates(b, a));

        // Check depths
        assert_eq!(dom_tree.depth(entry), 0);
        assert_eq!(dom_tree.depth(a), 1);
        assert_eq!(dom_tree.depth(b), 2);
        assert_eq!(dom_tree.depth(c), 3);
    }

    #[test]
    fn test_dominator_diamond() {
        // Diamond CFG:
        //      entry
        //      /   \
        //     a     b
        //      \   /
        //       exit
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let exit = graph.add_node("exit");

        graph.add_edge(entry, a, ()).unwrap();
        graph.add_edge(entry, b, ()).unwrap();
        graph.add_edge(a, exit, ()).unwrap();
        graph.add_edge(b, exit, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);

        // entry is immediate dominator of a, b, and exit
        assert_eq!(dom_tree.immediate_dominator(a), Some(entry));
        assert_eq!(dom_tree.immediate_dominator(b), Some(entry));
        assert_eq!(dom_tree.immediate_dominator(exit), Some(entry));

        // a and b don't dominate exit (alternative paths exist)
        assert!(!dom_tree.strictly_dominates(a, exit));
        assert!(!dom_tree.strictly_dominates(b, exit));

        // entry dominates all
        assert!(dom_tree.dominates(entry, a));
        assert!(dom_tree.dominates(entry, b));
        assert!(dom_tree.dominates(entry, exit));
    }

    #[test]
    fn test_dominator_if_then_else() {
        // if-then-else:
        //      entry
        //        |
        //       cond
        //      /    \
        //   then    else
        //      \    /
        //       merge
        //        |
        //       exit
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let cond = graph.add_node("cond");
        let then_b = graph.add_node("then");
        let else_b = graph.add_node("else");
        let merge = graph.add_node("merge");
        let exit = graph.add_node("exit");

        graph.add_edge(entry, cond, ()).unwrap();
        graph.add_edge(cond, then_b, ()).unwrap();
        graph.add_edge(cond, else_b, ()).unwrap();
        graph.add_edge(then_b, merge, ()).unwrap();
        graph.add_edge(else_b, merge, ()).unwrap();
        graph.add_edge(merge, exit, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);

        // Check dominator chain
        assert_eq!(dom_tree.immediate_dominator(cond), Some(entry));
        assert_eq!(dom_tree.immediate_dominator(then_b), Some(cond));
        assert_eq!(dom_tree.immediate_dominator(else_b), Some(cond));
        assert_eq!(dom_tree.immediate_dominator(merge), Some(cond));
        assert_eq!(dom_tree.immediate_dominator(exit), Some(merge));

        // cond dominates merge and exit
        assert!(dom_tree.dominates(cond, merge));
        assert!(dom_tree.dominates(cond, exit));

        // then/else don't dominate merge
        assert!(!dom_tree.strictly_dominates(then_b, merge));
        assert!(!dom_tree.strictly_dominates(else_b, merge));
    }

    #[test]
    fn test_dominator_loop() {
        // Simple loop:
        //      entry
        //        |
        //        v
        //   +-> header
        //   |    |
        //   |    v
        //   +-- body
        //        |
        //        v
        //       exit
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let header = graph.add_node("header");
        let body = graph.add_node("body");
        let exit = graph.add_node("exit");

        graph.add_edge(entry, header, ()).unwrap();
        graph.add_edge(header, body, ()).unwrap();
        graph.add_edge(body, header, ()).unwrap(); // back edge
        graph.add_edge(body, exit, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);

        // header dominates body and exit
        assert!(dom_tree.dominates(header, body));
        // body does not dominate header (despite the back edge)
        assert!(!dom_tree.strictly_dominates(body, header));
    }

    #[test]
    fn test_dominator_iterator() {
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");

        graph.add_edge(entry, a, ()).unwrap();
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);

        // Iterate dominators of c
        let dominators: Vec<NodeId> = dom_tree.dominators(c).collect();
        assert_eq!(dominators, vec![c, b, a, entry]);

        // Iterate dominators of entry
        let dominators: Vec<NodeId> = dom_tree.dominators(entry).collect();
        assert_eq!(dominators, vec![entry]);
    }

    #[test]
    fn test_dominator_children() {
        // Diamond CFG
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let exit = graph.add_node("exit");

        graph.add_edge(entry, a, ()).unwrap();
        graph.add_edge(entry, b, ()).unwrap();
        graph.add_edge(a, exit, ()).unwrap();
        graph.add_edge(b, exit, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);

        // entry has children: a, b, exit
        let mut children = dom_tree.children(entry).to_vec();
        children.sort_by_key(|n| n.index());
        assert_eq!(children, vec![a, b, exit]);

        // a, b, exit have no children
        assert!(dom_tree.children(a).is_empty());
        assert!(dom_tree.children(b).is_empty());
        assert!(dom_tree.children(exit).is_empty());
    }

    #[test]
    fn test_dominance_frontier_diamond() {
        // Diamond CFG - classic case for dominance frontiers
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let left = graph.add_node("left");
        let right = graph.add_node("right");
        let join = graph.add_node("join");

        graph.add_edge(entry, left, ()).unwrap();
        graph.add_edge(entry, right, ()).unwrap();
        graph.add_edge(left, join, ()).unwrap();
        graph.add_edge(right, join, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);
        let frontiers = compute_dominance_frontiers(&graph, &dom_tree);

        // entry has no dominance frontier
        assert!(frontiers[entry.index()].is_empty());

        // left's dominance frontier is {join}
        assert!(frontiers[left.index()].contains(join.index()));
        assert_eq!(frontiers[left.index()].count(), 1);

        // right's dominance frontier is {join}
        assert!(frontiers[right.index()].contains(join.index()));
        assert_eq!(frontiers[right.index()].count(), 1);

        // join has no dominance frontier (no successors)
        assert!(frontiers[join.index()].is_empty());
    }

    #[test]
    fn test_dominance_frontier_loop() {
        // Loop with header
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let header = graph.add_node("header");
        let body = graph.add_node("body");
        let exit = graph.add_node("exit");

        graph.add_edge(entry, header, ()).unwrap();
        graph.add_edge(header, body, ()).unwrap();
        graph.add_edge(body, header, ()).unwrap(); // back edge
        graph.add_edge(header, exit, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);
        let frontiers = compute_dominance_frontiers(&graph, &dom_tree);

        // body's dominance frontier includes header (the loop header)
        assert!(frontiers[body.index()].contains(header.index()));
    }

    #[test]
    fn test_dominance_frontier_nested_if() {
        // Nested if structure:
        //       entry
        //         |
        //        if1
        //       /   \
        //      a     b
        //     / \     \
        //    c   d     e
        //     \ /     /
        //     join1  /
        //       \   /
        //       join2
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let if1 = graph.add_node("if1");
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        let e = graph.add_node("e");
        let join1 = graph.add_node("join1");
        let join2 = graph.add_node("join2");

        graph.add_edge(entry, if1, ()).unwrap();
        graph.add_edge(if1, a, ()).unwrap();
        graph.add_edge(if1, b, ()).unwrap();
        graph.add_edge(a, c, ()).unwrap();
        graph.add_edge(a, d, ()).unwrap();
        graph.add_edge(b, e, ()).unwrap();
        graph.add_edge(c, join1, ()).unwrap();
        graph.add_edge(d, join1, ()).unwrap();
        graph.add_edge(e, join2, ()).unwrap();
        graph.add_edge(join1, join2, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);
        let frontiers = compute_dominance_frontiers(&graph, &dom_tree);

        // c and d have join1 in their dominance frontier
        assert!(frontiers[c.index()].contains(join1.index()));
        assert!(frontiers[d.index()].contains(join1.index()));

        // join1 and e have join2 in their dominance frontier
        assert!(frontiers[join1.index()].contains(join2.index()));
        assert!(frontiers[e.index()].contains(join2.index()));
    }

    #[test]
    fn test_strictly_dominates() {
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let a = graph.add_node("a");

        graph.add_edge(entry, a, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);

        // entry dominates itself but doesn't strictly dominate itself
        assert!(dom_tree.dominates(entry, entry));
        assert!(!dom_tree.strictly_dominates(entry, entry));

        // entry strictly dominates a
        assert!(dom_tree.strictly_dominates(entry, a));
    }

    #[test]
    fn test_dominator_complex_cfg() {
        // More complex CFG with multiple paths and joins
        //
        //        entry
        //          |
        //          a
        //         / \
        //        b   c
        //        |   |
        //        d   e
        //         \ / \
        //          f   g
        //          |
        //          h
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("entry");
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        let e = graph.add_node("e");
        let f = graph.add_node("f");
        let g = graph.add_node("g");
        let h = graph.add_node("h");

        graph.add_edge(entry, a, ()).unwrap();
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(a, c, ()).unwrap();
        graph.add_edge(b, d, ()).unwrap();
        graph.add_edge(c, e, ()).unwrap();
        graph.add_edge(d, f, ()).unwrap();
        graph.add_edge(e, f, ()).unwrap();
        graph.add_edge(e, g, ()).unwrap();
        graph.add_edge(f, h, ()).unwrap();

        let dom_tree = compute_dominators(&graph, entry);

        // a dominates everything below it
        assert!(dom_tree.dominates(a, b));
        assert!(dom_tree.dominates(a, c));
        assert!(dom_tree.dominates(a, d));
        assert!(dom_tree.dominates(a, e));
        assert!(dom_tree.dominates(a, f));
        assert!(dom_tree.dominates(a, g));
        assert!(dom_tree.dominates(a, h));

        // f's immediate dominator is a (not d or e, since there are multiple paths)
        assert_eq!(dom_tree.immediate_dominator(f), Some(a));

        // g's immediate dominator is e (only one path to g)
        assert_eq!(dom_tree.immediate_dominator(g), Some(e));
    }
}
