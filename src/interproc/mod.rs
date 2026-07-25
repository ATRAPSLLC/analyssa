//! Interprocedural summary framework — the whole-program layer.
//!
//! Analyses that only look at one function at a time cannot recover a callee's
//! return type, a caller's argument types, or a calling convention. Those need
//! facts to flow along the call graph, in an order that visits a callee before
//! its callers, with a bounded fixpoint wherever recursion makes that order
//! impossible.
//!
//! This module owns that machinery generically, so every frontend — native,
//! CIL, and any future one — drives the same driver instead of writing its own.
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
//! - [`solve`] drives a caller-supplied [`SummaryTransfer`] over those
//!   components to a fixpoint.
//!
//! # Parallelism
//!
//! [`solve`] is **sequential and does not spawn**. Component order is a data
//! dependency, so the wavefront cannot be parallelised at this level, and hosts
//! commonly run the whole driver inside their own admission-controlled CPU pool
//! — a nested pool here would oversubscribe it. Hosts that want parallelism
//! within a component drive [`CallGraph::components_callee_first`] themselves.

use std::{collections::HashSet, hash::Hash};

pub use self::summary::{DEFAULT_MAX_ITERATIONS, SummaryStore, SummaryTransfer, solve};
use crate::{graph::IndexedGraph, target::Target, world::World};

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
    self_recursive: HashSet<K>,
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
            self_recursive: HashSet::new(),
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
        let known: HashSet<K> = keys.iter().cloned().collect();
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
    /// — every callee's component before its callers'.
    #[must_use]
    pub fn components_callee_first(&self) -> Vec<Vec<K>> {
        self.graph
            .map_sccs_to_keys(&crate::graph::algorithms::strongly_connected_components(
                self.graph.inner(),
            ))
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
