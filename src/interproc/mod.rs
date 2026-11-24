//! Interprocedural summary framework - the whole-program layer.
//!
//! Analyses that only look at one function at a time cannot recover a callee's
//! return type, a caller's argument types, or a calling convention. Those need
//! facts to flow along the call graph, in an order that visits a callee before
//! its callers, with a bounded fixpoint wherever recursion makes that order
//! impossible.
//!
//! This module owns that machinery generically, so every frontend - native,
//! CIL, and any future one - drives the same driver instead of writing its own.
//!
//! # Shape
//!
//! - [`CallGraph`] is the call graph over an opaque method key, built either
//!   from a [`World`] or from an explicit edge list.
//! - [`CallGraph::components_callee_first`] returns the strongly-connected
//!   components in reverse topological order. **The grouping is the point**: a
//!   singleton non-recursive component converges in one visit, while a
//!   recursive component needs iteration, and a flat method list cannot express
//!   that difference.
//! - [`CallGraph::weakly_connected_components`] answers the same question for a
//!   transfer that also reads its **callers**. The dependency relation is then
//!   the call graph and its reverse, so the unit that reaches a joint fixpoint
//!   is the weakly connected component.
//! - [`solve`] drives a caller-supplied [`SummaryTransfer`] over those
//!   components to a fixpoint, choosing the grouping from the transfer's
//!   [`SummaryFlow`].
//!
//! # Parallelism
//!
//! [`solve`] is **sequential and does not spawn**. Component order is a data
//! dependency, so the wavefront cannot be parallelised at this level, and
//! hosts commonly run the whole driver inside their own admission-controlled
//! CPU pool - a nested pool here would oversubscribe it. Hosts that want
//! parallelism within a component drive [`CallGraph::components_callee_first`]
//! themselves.

use std::hash::Hash;

use rustc_hash::{FxHashMap, FxHashSet};

pub use self::summary::{SolveOutcome, SummaryFlow, SummaryStore, SummaryTransfer, solve};
use crate::{
    graph::{IndexedGraph, NodeId, algorithms::strongly_connected_components},
    target::Target,
    world::World,
};

mod summary;

/// Call graph over an opaque method key.
///
/// A thin, purpose-named wrapper over [`IndexedGraph`], which already provides
/// key densification, duplicate-edge suppression, and SCC extraction. The
/// wrapper exists so the interprocedural layer has one vocabulary rather than
/// each host re-deriving the same graph.
#[derive(Debug)]
pub struct CallGraph<K>
where
    K: Hash + Eq + Clone,
{
    graph: IndexedGraph<K, ()>,
    /// Keys with an edge to themselves, which `components_callee_first`'s
    /// singleton components cannot otherwise distinguish from a leaf.
    self_recursive: FxHashSet<K>,
}

impl<K> Default for CallGraph<K>
where
    K: Hash + Eq + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K> CallGraph<K>
where
    K: Hash + Eq + Clone,
{
    /// Creates an empty call graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: IndexedGraph::new(),
            self_recursive: FxHashSet::default(),
        }
    }

    /// Builds a call graph from `methods` and the `callees` of each.
    ///
    /// Edges to keys absent from `methods` are dropped: an interprocedural
    /// analysis can only reason about bodies it can see, and an edge to an
    /// import or an unresolved indirect target carries no summary.
    #[must_use]
    pub fn from_edges<I, C>(methods: I, callees: C) -> Self
    where
        I: IntoIterator<Item = K>,
        C: Fn(&K) -> Vec<K>,
    {
        let mut graph = Self::new();
        let keys: Vec<K> = methods.into_iter().collect();
        let known: FxHashSet<K> = keys.iter().cloned().collect();
        for key in &keys {
            graph.add_method(key.clone());
        }
        for caller in &keys {
            for callee in callees(caller) {
                if known.contains(&callee) {
                    graph.add_call(caller.clone(), callee);
                }
            }
        }
        graph
    }

    /// Builds a call graph from a [`World`].
    #[must_use]
    pub fn from_world<T, W>(world: &W) -> Self
    where
        T: Target<MethodRef = K>,
        W: World<T> + ?Sized,
    {
        Self::from_edges(world.all_methods(), |method| world.callees(method))
    }

    /// Registers a method, so a leaf with no edges still appears in the
    /// component order.
    pub fn add_method(&mut self, method: K) {
        self.graph.add_node(method);
    }

    /// Records that `caller` calls `callee`, registering either key if new.
    pub fn add_call(&mut self, caller: K, callee: K) {
        if caller == callee {
            self.self_recursive.insert(caller.clone());
        }
        // `add_edge` only fails on an internal graph invariant; a duplicate
        // edge is reported as `Ok(false)` and is already suppressed.
        let _ = self.graph.add_edge(caller, callee, ());
    }

    /// Returns the number of methods in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns `true` when the graph holds no methods.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Returns `true` when `method` calls itself directly.
    ///
    /// A singleton component is otherwise indistinguishable from a leaf, and a
    /// self-recursive method needs the same bounded fixpoint a larger cycle
    /// does.
    #[must_use]
    pub fn is_self_recursive(&self, method: &K) -> bool {
        self.self_recursive.contains(method)
    }

    /// Returns the strongly-connected components in reverse topological order
    /// - every callee's component before its callers'.
    #[must_use]
    pub fn components_callee_first(&self) -> Vec<Vec<K>> {
        self.graph
            .map_sccs_to_keys(&strongly_connected_components(self.graph.inner()))
    }

    /// Returns the weakly connected components, each in callee-first order.
    ///
    /// Two methods share a component when a chain of calls joins them in
    /// *either* direction. That is the right grouping for a transfer whose
    /// summaries depend on its callers as well as its callees
    /// ([`SummaryFlow::Bidirectional`]): no call-graph order can visit both
    /// ends of an edge first, so every method a fact can reach has to be
    /// iterated together.
    ///
    /// Each component is built by unioning the strongly-connected components an
    /// edge joins, so the methods inside it keep the callee-first order
    /// [`Self::components_callee_first`] put them in. That ordering is what
    /// makes the fixpoint settle in few passes: the bottom-up half of a
    /// bidirectional flow still completes in one.
    #[must_use]
    pub fn weakly_connected_components(&self) -> Vec<Vec<K>> {
        let sccs = self.components_callee_first();
        let mut scc_of: FxHashMap<&K, usize> = FxHashMap::default();
        for (index, component) in sccs.iter().enumerate() {
            for method in component {
                scc_of.insert(method, index);
            }
        }

        // Union the components an edge joins, whichever way it points.
        let mut parent: Vec<usize> = (0..sccs.len()).collect();
        let inner = self.graph.inner();
        for node in inner.node_ids() {
            let Some(from) = self.graph.get_key(node).and_then(|key| scc_of.get(key)) else {
                continue;
            };
            let from = *from;
            for successor in inner.successors(node) {
                let Some(to) = self
                    .graph
                    .get_key(successor)
                    .and_then(|key| scc_of.get(key))
                else {
                    continue;
                };
                union_components(&mut parent, from, *to);
            }
        }

        // Regroup, keeping both the component order and the order within one.
        let mut order: Vec<usize> = Vec::new();
        let mut grouped: FxHashMap<usize, Vec<K>> = FxHashMap::default();
        for (index, component) in sccs.into_iter().enumerate() {
            let root = component_root(&mut parent, index);
            grouped
                .entry(root)
                .or_insert_with(|| {
                    order.push(root);
                    Vec::new()
                })
                .extend(component);
        }
        order
            .into_iter()
            .filter_map(|root| grouped.remove(&root))
            .collect()
    }

    /// The methods `method` calls and the methods that call it, each once, in
    /// the order the graph holds them: the ones a bidirectional visit of
    /// `method` reads, and the ones a change it makes reaches.
    #[must_use]
    pub fn neighbours(&self, method: &K) -> Vec<K> {
        let Some(node) = self.graph.get_node_id(method) else {
            return Vec::new();
        };
        let inner = self.graph.inner();
        let mut seen: FxHashSet<NodeId> = FxHashSet::default();
        inner
            .successors(node)
            .chain(inner.predecessors(node))
            .filter(|neighbour| *neighbour != node && seen.insert(*neighbour))
            .filter_map(|neighbour| self.graph.get_key(neighbour).cloned())
            .collect()
    }

    /// Returns `true` when `component` needs a fixpoint rather than one visit.
    #[must_use]
    pub fn is_recursive(&self, component: &[K]) -> bool {
        component.len() > 1
            || component
                .first()
                .is_some_and(|method| self.is_self_recursive(method))
    }
}

/// Merges the components holding `left` and `right` into one.
fn union_components(parent: &mut [usize], left: usize, right: usize) {
    let left = component_root(parent, left);
    let right = component_root(parent, right);
    if left != right
        && let Some(slot) = parent.get_mut(right)
    {
        *slot = left;
    }
}

/// Returns the representative of `node`'s component, halving the path to it.
///
/// Every step rewrites a link to point at its grandparent, so repeated lookups
/// over one component flatten it without a second pass.
fn component_root(parent: &mut [usize], node: usize) -> usize {
    let mut node = node;
    while let Some(up) = parent.get(node).copied() {
        if up == node {
            break;
        }
        let grandparent = parent.get(up).copied().unwrap_or(up);
        if let Some(slot) = parent.get_mut(node) {
            *slot = grandparent;
        }
        node = grandparent;
    }
    node
}
