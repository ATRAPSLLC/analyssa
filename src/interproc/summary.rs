//! The summary fixpoint: drives a caller-supplied transfer over the call
//! graph's components, callees first.

use std::{collections::HashMap, hash::Hash};

use crate::interproc::CallGraph;

/// Iterations a recursive component runs before its partial summaries are
/// accepted.
///
/// This bounds the *expensive* outer loop — each iteration re-derives every
/// method in the component — so it is deliberately small. It is a backstop, not
/// the termination argument: a monotone transfer over a finite lattice
/// converges well inside it, and a transfer that does not converge here was not
/// monotone to begin with.
pub const DEFAULT_MAX_ITERATIONS: usize = 8;

/// Per-method summaries accumulated by a [`solve`] run.
///
/// Summaries are what flows along the call graph: a callee's recovered return
/// type reaching its callers, a caller's argument types reaching its callee's
/// parameters.
#[derive(Debug, Clone)]
pub struct SummaryStore<K, S>
where
    K: Hash + Eq + Clone,
{
    summaries: HashMap<K, S>,
}

impl<K, S> Default for SummaryStore<K, S>
where
    K: Hash + Eq + Clone,
{
    fn default() -> Self {
        Self {
            summaries: HashMap::new(),
        }
    }
}

impl<K, S> SummaryStore<K, S>
where
    K: Hash + Eq + Clone,
{
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `method`'s summary, if one has been recorded.
    #[must_use]
    pub fn get(&self, method: &K) -> Option<&S> {
        self.summaries.get(method)
    }

    /// Returns a mutable handle to `method`'s summary, inserting the default
    /// when absent.
    pub fn entry(&mut self, method: K) -> &mut S
    where
        S: Default,
    {
        self.summaries.entry(method).or_default()
    }

    /// Replaces `method`'s summary.
    pub fn insert(&mut self, method: K, summary: S) {
        self.summaries.insert(method, summary);
    }

    /// Removes `method`'s summary, returning it.
    pub fn remove(&mut self, method: &K) -> Option<S> {
        self.summaries.remove(method)
    }

    /// Returns the number of recorded summaries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.summaries.len()
    }

    /// Returns `true` when nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.summaries.is_empty()
    }

    /// Iterates the recorded summaries.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &S)> {
        self.summaries.iter()
    }
}

/// The per-method step of an interprocedural analysis.
///
/// [`solve`] owns the traversal order and the fixpoint; an implementation owns
/// what a summary *is* and how one method's facts update its neighbours'.
pub trait SummaryTransfer<K, S>
where
    K: Hash + Eq + Clone,
{
    /// Updates summaries from `method`, returning `true` when anything changed.
    ///
    /// Called with callees already processed, so a callee's summary is
    /// available when its caller is visited — except within a recursive
    /// component, where [`solve`] iterates until this stops reporting change.
    ///
    /// **Monotonicity is the caller's obligation.** The returned flag drives the
    /// fixpoint, so a transfer that reports change on every call will run to the
    /// iteration cap and stop with partial summaries. Combining with a monotone
    /// meet, and recomputing a method's incoming facts from scratch rather than
    /// mutating them incrementally, is what makes conflicting inputs collapse
    /// deterministically instead of oscillating.
    fn visit(&mut self, method: &K, summaries: &mut SummaryStore<K, S>) -> bool;
}

/// Drives `transfer` over `graph` to a fixpoint, callees before callers.
///
/// Non-recursive components are visited once — their callees are already final,
/// so a second visit could not change anything. A recursive component iterates
/// until no summary changes or `max_iterations` is reached.
///
/// Sequential by design: component order is a data dependency, and hosts
/// commonly run this inside their own CPU pool where a nested one would
/// oversubscribe. See the module docs.
///
/// # Arguments
///
/// * `graph` — Call graph whose component order drives the traversal.
/// * `transfer` — Per-method step.
/// * `max_iterations` — Cap on a recursive component's fixpoint; see
///   [`DEFAULT_MAX_ITERATIONS`].
///
/// # Returns
///
/// The accumulated summaries.
pub fn solve<K, S, F>(
    graph: &CallGraph<K>,
    transfer: &mut F,
    max_iterations: usize,
) -> SummaryStore<K, S>
where
    K: Hash + Eq + Clone,
    F: SummaryTransfer<K, S> + ?Sized,
{
    let mut summaries = SummaryStore::new();
    for component in graph.components_callee_first() {
        let iterations = if graph.is_recursive(&component) {
            max_iterations
        } else {
            1
        };
        for _ in 0..iterations {
            let mut changed = false;
            for method in &component {
                changed |= transfer.visit(method, &mut summaries);
            }
            if !changed {
                break;
            }
        }
    }
    summaries
}
