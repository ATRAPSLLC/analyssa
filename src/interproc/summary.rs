//! The summary fixpoint: drives a caller-supplied transfer over the call
//! graph's components, callees first.

use std::{collections::VecDeque, hash::Hash, iter, mem, num::NonZeroUsize};

use rustc_hash::FxHashMap;

use crate::interproc::CallGraph;

/// Which neighbours a method's summary is allowed to depend on.
///
/// [`solve`] visits callees before callers, and that order is a fixpoint by
/// itself only while facts move in that one direction. A transfer that also
/// reads its **callers** - parameter types recovered from the arguments its
/// call sites pass in - cannot be driven that way: at the moment a callee is
/// visited none of its callers has a summary, and a callee-first order never
/// comes back to it. Declaring the flow is what lets [`solve`] choose a
/// grouping that settles both halves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SummaryFlow {
    /// A summary depends only on the summaries of the methods it calls.
    ///
    /// Strongly-connected components, callees first; a non-recursive one is
    /// visited once.
    #[default]
    BottomUp,
    /// A summary depends on its callers' summaries as well as its callees'.
    ///
    /// Weakly connected components, each iterated to a fixpoint: the dependency
    /// relation is the call graph and its reverse, so every method a fact can
    /// reach has to move together. A visit reads its method's summary and
    /// those of the method's callers and callees and no others, and writes
    /// only among them, marking each it changes when it changes another's
    /// ([`SummaryStore::mark_changed`]): [`solve`] visits again only the
    /// methods that read a changed summary.
    Bidirectional,
}

/// Per-method summaries accumulated by a [`solve`] run.
///
/// Summaries are what flows along the call graph: a callee's recovered return
/// type reaching its callers, a caller's argument types reaching its callee's
/// parameters. The second of those is only carried when the transfer declares
/// [`SummaryFlow::Bidirectional`].
#[derive(Debug, Clone)]
pub struct SummaryStore<K, S>
where
    K: Hash + Eq + Clone,
{
    summaries: FxHashMap<K, S>,
    /// The methods whose summaries the current visit marked changed
    /// ([`Self::mark_changed`]), which [`solve`] takes after it.
    changed: Vec<K>,
}

impl<K, S> Default for SummaryStore<K, S>
where
    K: Hash + Eq + Clone,
{
    fn default() -> Self {
        Self {
            summaries: FxHashMap::default(),
            changed: Vec::new(),
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

    /// Records that the visit under way changed `method`'s summary.
    ///
    /// A [`SummaryFlow::Bidirectional`] solve visits again the methods that
    /// read a changed summary: the method and its callers and callees. A
    /// visit that changes a neighbour's summary marks it here, or the
    /// neighbour's other neighbours, which read it, are not visited again for
    /// it; one that marks anything marks every summary it changed, its own
    /// included. A visit that reports a change and marks nothing changed its
    /// own summary alone.
    pub fn mark_changed(&mut self, method: K) {
        self.changed.push(method);
    }

    /// Takes the methods [`Self::mark_changed`] recorded since the last take.
    fn take_changed(&mut self) -> Vec<K> {
        mem::take(&mut self.changed)
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
    /// A visit that changes the summary of a method other than `method`
    /// marks each summary it changed with [`SummaryStore::mark_changed`] as
    /// well.
    ///
    /// Called with callees already processed, so a callee's summary is
    /// available when its caller is visited - except within a recursive
    /// component, where [`solve`] iterates until this stops reporting change.
    ///
    /// **Monotonicity is the caller's obligation.** The returned flag drives the
    /// fixpoint, so a transfer that reports change on every call will run to the
    /// iteration cap and stop with partial summaries. Combining with a monotone
    /// meet, and recomputing a method's incoming facts from scratch rather than
    /// mutating them incrementally, is what makes conflicting inputs collapse
    /// deterministically instead of oscillating.
    fn visit(&mut self, method: &K, summaries: &mut SummaryStore<K, S>) -> bool;

    /// Which neighbours [`Self::visit`] reads, which fixes the traversal.
    ///
    /// The default is [`SummaryFlow::BottomUp`], the direction a callee-first
    /// order settles in one visit. A transfer that reads a method's callers -
    /// typing a parameter from the arguments its call sites pass - must say so,
    /// or that half of its analysis is computed against empty summaries and
    /// silently dropped.
    fn flow(&self) -> SummaryFlow {
        SummaryFlow::BottomUp
    }

    /// Whether the host wants the run to stop, asked before every visit.
    ///
    /// A run that stops leaves summaries for part of the call graph, which
    /// [`SolveOutcome::interrupted`] says; the host discards them rather than
    /// let the answer depend on when it asked. The default never stops.
    fn interrupted(&self) -> bool {
        false
    }
}

/// What a [`solve`] run produced, and whether to believe all of it.
///
/// The summaries alone cannot be read safely: a recursive component that hit
/// the iteration cap leaves partial facts that look exactly like settled ones.
/// A caller that must not act on a half-computed answer reads
/// [`converged`](Self::converged) first, and one that wants to say *which*
/// methods are unsettled reads [`unconverged`](Self::unconverged).
#[derive(Debug, Clone)]
pub struct SolveOutcome<K: Hash + Eq + Clone, S> {
    /// The accumulated summaries.
    pub summaries: SummaryStore<K, S>,
    /// Whether every component reached a fixpoint within its cap.
    pub converged: bool,
    /// The most iterations any one component took.
    ///
    /// A run that converged in one iteration everywhere had nothing recursive
    /// to settle.
    pub iterations: usize,
    /// The methods of every component that hit the cap, in traversal order.
    ///
    /// Their summaries are whatever the last iteration left, which is a lower
    /// bound on the truth for a monotone transfer and nothing at all for one
    /// that is not.
    pub unconverged: Vec<K>,
    /// Whether the run stopped because [`SummaryTransfer::interrupted`] said
    /// so. The summaries then cover part of the call graph, and the run is not
    /// [`converged`](Self::converged).
    pub interrupted: bool,
}

impl<K: Hash + Eq + Clone, S> SolveOutcome<K, S> {}

/// Drives `transfer` over `graph` to a fixpoint, callees before callers.
///
/// The grouping comes from [`SummaryTransfer::flow`]:
///
/// - [`SummaryFlow::BottomUp`] walks the strongly-connected components. A
///   non-recursive one is visited once - its callees are already final, so a
///   second visit could not change anything - and a recursive one iterates
///   until no summary changes or `max_iterations` is reached.
/// - [`SummaryFlow::Bidirectional`] walks the weakly connected components,
///   each over a worklist: every method in component order first, then each
///   method that reads a changed summary - the method whose summary changed,
///   its callers and its callees ([`CallGraph::neighbours`]) - until none is
///   left or the component has had `max_iterations` visits per method. A
///   change reaches only the methods that read it, so a component of
///   thousands of methods revisits the few a fact moves through rather than
///   all of them. No acyclic shortcut applies: a method's callers are only
///   summarised inside this same traversal.
///
/// Sequential by design: component order is a data dependency, and hosts
/// commonly run this inside their own CPU pool where a nested one would
/// oversubscribe. See the module docs.
///
/// # Arguments
///
/// * `graph` - Call graph whose component order drives the traversal.
/// * `transfer` - Per-method step.
/// * `max_iterations` - Cap on a recursive component's fixpoint, a host's
///   [`Budget::SummaryRounds`](crate::analysis::budget::Budget::SummaryRounds).
///   It bounds the expensive outer loop - each iteration re-derives every
///   method in the component - and is a backstop, not the termination
///   argument: a monotone transfer over a finite lattice converges well
///   inside it. A `NonZeroUsize` because a cap of zero would visit nothing
///   and report summaries that were never computed.
///
/// # Returns
///
/// The summaries, whether every component settled within its cap, and whether
/// the host stopped the run.
pub fn solve<K, S, F>(
    graph: &CallGraph<K>,
    transfer: &mut F,
    max_iterations: NonZeroUsize,
) -> SolveOutcome<K, S>
where
    K: Hash + Eq + Clone,
    F: SummaryTransfer<K, S> + ?Sized,
{
    let mut summaries = SummaryStore::new();
    let mut converged = true;
    let mut iterations = 0usize;
    let mut unconverged: Vec<K> = Vec::new();
    let flow = transfer.flow();
    let components = match flow {
        SummaryFlow::BottomUp => graph.components_callee_first(),
        SummaryFlow::Bidirectional => graph.weakly_connected_components(),
    };
    for component in components {
        if matches!(flow, SummaryFlow::Bidirectional) {
            let visits = max_iterations.get().saturating_mul(component.len());
            match drain(graph, &component, transfer, &mut summaries, visits) {
                Drained::Settled(passes) => iterations = iterations.max(passes),
                Drained::Unsettled(passes) => {
                    iterations = iterations.max(passes);
                    converged = false;
                    unconverged.extend(component.iter().cloned());
                }
                Drained::Interrupted(passes) => {
                    return SolveOutcome {
                        summaries,
                        converged: false,
                        iterations: iterations.max(passes),
                        unconverged,
                        interrupted: true,
                    };
                }
            }
            continue;
        }
        let recursive = graph.is_recursive(&component);
        let cap = if recursive { max_iterations.get() } else { 1 };
        let mut settled = false;
        let mut taken = 0usize;
        for _ in 0..cap {
            taken = taken.saturating_add(1);
            let mut changed = false;
            for method in &component {
                if transfer.interrupted() {
                    return SolveOutcome {
                        summaries,
                        converged: false,
                        iterations: iterations.max(taken),
                        unconverged,
                        interrupted: true,
                    };
                }
                changed |= transfer.visit(method, &mut summaries);
                // A callee-first order revisits nothing for a mark: a
                // recursive component's next pass visits every method.
                summaries.changed.clear();
            }
            if !changed {
                settled = true;
                break;
            }
        }
        iterations = iterations.max(taken);
        // A non-recursive bottom-up component is settled by construction: its
        // callees are final, so the single visit is the fixpoint whether or not
        // it reported a change.
        if recursive && !settled {
            converged = false;
            unconverged.extend(component.iter().cloned());
        }
    }
    SolveOutcome {
        summaries,
        converged,
        iterations,
        unconverged,
        interrupted: false,
    }
}

/// How a component's worklist ended, with the visits it took in passes of
/// the component (rounded up).
enum Drained {
    /// No method was left to visit.
    Settled(usize),
    /// The budget ran out with methods still to visit.
    Unsettled(usize),
    /// The host asked the run to stop.
    Interrupted(usize),
}

/// Visits `component`'s methods over a worklist: each in component order,
/// then every method that reads a changed summary - the changed method itself
/// and its [`CallGraph::neighbours`] - in the order the changes were made,
/// each queued once at a time, until none is left or `budget` visits have
/// run. A visit's changed summaries are the ones it marks
/// ([`SummaryStore::mark_changed`]), or its method's alone when it reports a
/// change and marks none.
///
/// A visit reads the summaries of its method and the method's neighbours, so
/// a method none of whose neighbours' summaries nor its own changed computes
/// what it computed before; only the readers of a change run again. The
/// queue is first-in first-out, so the visits are a function of the
/// component and the transfer alone.
fn drain<K, S, F>(
    graph: &CallGraph<K>,
    component: &[K],
    transfer: &mut F,
    summaries: &mut SummaryStore<K, S>,
    budget: usize,
) -> Drained
where
    K: Hash + Eq + Clone,
    F: SummaryTransfer<K, S> + ?Sized,
{
    let position: FxHashMap<&K, usize> = component
        .iter()
        .enumerate()
        .map(|(index, method)| (method, index))
        .collect();
    let mut queued = vec![true; component.len()];
    let mut queue: VecDeque<usize> = (0..component.len()).collect();
    let mut visits = 0usize;
    let passes = |visits: usize| visits.div_ceil(component.len().max(1));
    while let Some(index) = queue.pop_front() {
        if visits >= budget {
            return Drained::Unsettled(passes(visits));
        }
        if transfer.interrupted() {
            return Drained::Interrupted(passes(visits));
        }
        let Some(method) = component.get(index) else {
            continue;
        };
        if let Some(flag) = queued.get_mut(index) {
            *flag = false;
        }
        visits = visits.saturating_add(1);
        let reported = transfer.visit(method, summaries);
        let mut changed = summaries.take_changed();
        if reported && changed.is_empty() {
            changed.push(method.clone());
        }
        for changed in &changed {
            let readers = graph.neighbours(changed);
            let reached = iter::once(changed)
                .chain(readers.iter())
                .filter_map(|reader| position.get(reader).copied());
            for reached in reached {
                if let Some(flag) = queued.get_mut(reached)
                    && !*flag
                {
                    *flag = true;
                    queue.push_back(reached);
                }
            }
        }
    }
    Drained::Settled(passes(visits))
}
