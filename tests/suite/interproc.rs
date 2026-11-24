//! Generic interprocedural summary driver: component order and the recursive
//! fixpoint.

use std::num::NonZeroUsize;

use analyssa::interproc::{CallGraph, SummaryFlow, SummaryStore, SummaryTransfer, solve};
use rustc_hash::FxHashMap;

/// A cap, for the tests that do not care which.
fn cap(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap_or(NonZeroUsize::MIN)
}

/// Records the order methods were visited in, and how many times each was.
#[derive(Default)]
struct VisitLog {
    order: Vec<u32>,
    /// Reports change for this many visits of the given method, forcing the
    /// fixpoint to iterate.
    keep_changing: Option<(u32, usize)>,
}

impl SummaryTransfer<u32, u32> for VisitLog {
    fn visit(&mut self, method: &u32, summaries: &mut SummaryStore<u32, u32>) -> bool {
        self.order.push(*method);
        let visits = self.order.iter().filter(|m| *m == method).count();
        summaries.insert(*method, visits as u32);
        match self.keep_changing {
            Some((target, times)) if target == *method => visits <= times,
            _ => false,
        }
    }
}

/// A chain `1 -> 2 -> 3` is visited callee-first, each method exactly once.
///
/// Non-recursive components need no fixpoint: their callees are already final
/// when they are reached, so a second visit could not change anything.
#[test]
fn a_chain_is_visited_callee_first_and_once_each() {
    let graph = CallGraph::from_edges([1u32, 2, 3], |method| match method {
        1 => vec![2],
        2 => vec![3],
        _ => Vec::new(),
    });

    let mut transfer = VisitLog::default();
    let outcome = solve(&graph, &mut transfer, cap(8));
    let summaries = outcome.summaries;

    assert_eq!(
        transfer.order,
        vec![3, 2, 1],
        "callees are visited before their callers"
    );
    assert_eq!(summaries.get(&3), Some(&1), "each visited exactly once");
    assert_eq!(summaries.get(&1), Some(&1));
}

/// A mutually recursive component is iterated until its summaries settle.
///
/// This is what the SCC *grouping* buys: `1 <-> 2` cannot be ordered
/// callee-first, so the driver must iterate the pair rather than visit each
/// once. A flat method list could not express the difference.
#[test]
fn a_recursive_component_iterates_to_a_fixpoint() {
    let graph = CallGraph::from_edges([1u32, 2], |method| match method {
        1 => vec![2],
        2 => vec![1],
        _ => Vec::new(),
    });
    assert!(
        graph.is_recursive(&[1, 2]),
        "a two-method cycle is recursive"
    );

    // Method 1 reports change for its first three visits.
    let mut transfer = VisitLog {
        keep_changing: Some((1, 3)),
        ..VisitLog::default()
    };
    let outcome = solve(&graph, &mut transfer, cap(8));

    let visits_of_one = transfer.order.iter().filter(|m| **m == 1).count();
    assert_eq!(
        visits_of_one, 4,
        "iterates while the transfer reports change, then once more to confirm"
    );
    assert!(outcome.converged, "and it did settle, inside the cap");
    assert!(outcome.unconverged.is_empty());
}

/// A self-recursive singleton is recursive too - it is otherwise
/// indistinguishable from a leaf, and needs the same bounded fixpoint.
#[test]
fn a_self_recursive_method_is_treated_as_recursive() {
    let graph = CallGraph::from_edges([1u32], |_| vec![1]);

    assert!(graph.is_self_recursive(&1));
    assert!(graph.is_recursive(&[1]), "a self-loop needs a fixpoint");

    let mut transfer = VisitLog {
        keep_changing: Some((1, 2)),
        ..VisitLog::default()
    };
    let outcome = solve(&graph, &mut transfer, cap(8));
    assert!(
        transfer.order.len() > 1,
        "a self-recursive method is revisited"
    );
    assert!(outcome.converged);
}

/// A non-converging transfer stops at the cap rather than running forever.
#[test]
fn a_non_converging_component_stops_at_the_cap() {
    let graph = CallGraph::from_edges([1u32, 2], |method| match method {
        1 => vec![2],
        2 => vec![1],
        _ => Vec::new(),
    });

    // Always reports change.
    let mut transfer = VisitLog {
        keep_changing: Some((1, usize::MAX)),
        ..VisitLog::default()
    };
    let outcome = solve(&graph, &mut transfer, cap(4));

    let visits_of_one = transfer.order.iter().filter(|m| **m == 1).count();
    assert_eq!(visits_of_one, 4, "bounded by the iteration cap");

    // And it says so. Summaries from a component that hit the cap are partial,
    // and indistinguishable from settled ones without this.
    assert!(!outcome.converged, "the cap was reached, not a fixpoint");
    assert_eq!(outcome.iterations, 4);
    let mut unconverged = outcome.unconverged.clone();
    unconverged.sort_unstable();
    assert_eq!(
        unconverged,
        vec![1, 2],
        "every method of the component is unsettled, not only the one that changed"
    );
}

/// Edges to methods outside the analysed set are dropped: an import or an
/// unresolved indirect target carries no summary to flow.
#[test]
fn edges_outside_the_method_set_are_dropped() {
    let graph = CallGraph::from_edges([1u32, 2], |method| match method {
        1 => vec![2, 99],
        _ => Vec::new(),
    });

    assert_eq!(graph.len(), 2, "the unknown callee is not a node");
    let components = graph.components_callee_first();
    let flat: Vec<u32> = components.into_iter().flatten().collect();
    assert_eq!(flat, vec![2, 1]);
}

/// An isolated method still appears, so a leaf with no edges is analysed.
#[test]
fn isolated_methods_still_appear() {
    let graph = CallGraph::from_edges([1u32, 2, 3], |_| Vec::new());
    let flat: Vec<u32> = graph
        .components_callee_first()
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(flat.len(), 3);
}

/// A transfer that only ever learns from its **callers**.
///
/// Each method's summary is its caller's summary plus one, so a fact starting
/// at the root of a chain can only arrive by a traversal that revisits a callee
/// after its callers have been summarised.
struct DownwardFlow {
    /// Who calls whom, so a method can read its callers' summaries.
    callers_of: FxHashMap<u32, Vec<u32>>,
    /// The method whose summary is given, and its value.
    root: (u32, u32),
    /// How the driver swept, for the ordering assertions.
    visits: usize,
}

impl SummaryTransfer<u32, u32> for DownwardFlow {
    fn flow(&self) -> SummaryFlow {
        SummaryFlow::Bidirectional
    }

    fn visit(&mut self, method: &u32, summaries: &mut SummaryStore<u32, u32>) -> bool {
        self.visits = self.visits.saturating_add(1);
        let next = if *method == self.root.0 {
            Some(self.root.1)
        } else {
            self.callers_of
                .get(method)
                .into_iter()
                .flatten()
                .filter_map(|caller| summaries.get(caller).copied())
                .max()
                .map(|from_caller| from_caller.saturating_add(1))
        };
        let Some(next) = next else {
            return false;
        };
        if summaries.get(method).copied() == Some(next) {
            return false;
        }
        summaries.insert(*method, next);
        true
    }
}

/// A fact held by the root of a chain reaches the far end of it.
///
/// The callee-first order alone cannot carry this: every method here is its own
/// non-recursive component, so a single visit each would compute `2` and `3`
/// against summaries that do not exist yet. Declaring the flow bidirectional is
/// what makes the driver iterate the three of them together.
#[test]
fn a_bidirectional_flow_carries_facts_from_callers_to_callees() {
    let graph = CallGraph::from_edges([1u32, 2, 3], |method| match method {
        1 => vec![2],
        2 => vec![3],
        _ => Vec::new(),
    });
    let mut transfer = DownwardFlow {
        callers_of: FxHashMap::from_iter([(2, vec![1]), (3, vec![2])]),
        root: (1, 10),
        visits: 0,
    };

    let outcome = solve(&graph, &mut transfer, cap(8));

    assert!(outcome.converged, "an acyclic chain settles inside the cap");
    assert_eq!(outcome.summaries.get(&1).copied(), Some(10));
    assert_eq!(outcome.summaries.get(&2).copied(), Some(11));
    assert_eq!(
        outcome.summaries.get(&3).copied(),
        Some(12),
        "the fact reached the far end of the chain"
    );
}

/// The whole chain is one component, so one non-converging method leaves every
/// method of it unconverged.
#[test]
fn a_bidirectional_component_reports_all_its_methods_unconverged() {
    let graph = CallGraph::from_edges([1u32, 2], |method| match method {
        1 => vec![2],
        _ => Vec::new(),
    });
    // A root whose value grows every visit never reaches a fixpoint.
    struct Unsettled;
    impl SummaryTransfer<u32, u32> for Unsettled {
        fn flow(&self) -> SummaryFlow {
            SummaryFlow::Bidirectional
        }

        fn visit(&mut self, method: &u32, summaries: &mut SummaryStore<u32, u32>) -> bool {
            let next = summaries
                .get(method)
                .copied()
                .unwrap_or(0)
                .saturating_add(1);
            summaries.insert(*method, next);
            true
        }
    }

    let outcome = solve(&graph, &mut Unsettled, cap(3));

    assert!(!outcome.converged);
    assert_eq!(outcome.iterations, 3, "it ran to the cap");
    let mut unconverged = outcome.unconverged.clone();
    unconverged.sort_unstable();
    assert_eq!(unconverged, vec![1, 2]);
}

/// Two chains that never call each other are separate components, so neither
/// pays for the other's fixpoint and one settling does not mask the other.
#[test]
fn disjoint_call_chains_are_separate_weak_components() {
    let graph = CallGraph::from_edges([1u32, 2, 3, 4], |method| match method {
        1 => vec![2],
        3 => vec![4],
        _ => Vec::new(),
    });

    let components = graph.weakly_connected_components();

    assert_eq!(components.len(), 2, "two disjoint chains, two components");
    for component in &components {
        let mut methods = component.clone();
        methods.sort_unstable();
        assert!(
            methods == vec![1, 2] || methods == vec![3, 4],
            "{methods:?}"
        );
    }
}

/// A diamond is one weak component even though no call joins its two sides:
/// both are reachable from the same caller, so a fact one side recovers can
/// reach the other through it.
#[test]
fn a_diamond_is_one_weak_component() {
    let graph = CallGraph::from_edges([1u32, 2, 3, 4], |method| match method {
        1 => vec![2, 3],
        2 | 3 => vec![4],
        _ => Vec::new(),
    });

    let components = graph.weakly_connected_components();

    assert_eq!(components.len(), 1);
    // One component, so flattening keeps exactly its order.
    let order: Vec<u32> = components.into_iter().flatten().collect();
    assert_eq!(order.len(), 4);
    let position = |method: u32| {
        order
            .iter()
            .position(|found| *found == method)
            .unwrap_or_default()
    };
    assert!(
        position(4) < position(2) && position(2) < position(1),
        "callee-first order survives the regrouping: {order:?}"
    );
}

/// Stops after a fixed number of visits, as a host's interrupt would.
struct StopAfter {
    visits: usize,
    limit: usize,
}

impl SummaryTransfer<u32, u32> for StopAfter {
    fn visit(&mut self, method: &u32, summaries: &mut SummaryStore<u32, u32>) -> bool {
        self.visits = self.visits.saturating_add(1);
        summaries.insert(*method, 1);
        false
    }

    fn interrupted(&self) -> bool {
        self.visits >= self.limit
    }
}

/// A host that asks to stop gets the run stopped before the next visit,
/// reported as interrupted and not converged, with summaries for only the
/// part visited.
#[test]
fn an_interrupted_run_stops_before_the_next_visit() {
    let graph = CallGraph::from_edges([1u32, 2, 3], |method| match method {
        1 => vec![2],
        2 => vec![3],
        _ => Vec::new(),
    });
    let mut transfer = StopAfter {
        visits: 0,
        limit: 2,
    };
    let outcome = solve(&graph, &mut transfer, cap(8));
    assert!(outcome.interrupted);
    assert!(!outcome.converged);
    assert_eq!(transfer.visits, 2);
    assert_eq!(
        outcome.summaries.get(&1),
        None,
        "the caller was never reached"
    );

    let mut transfer = StopAfter {
        visits: 0,
        limit: usize::MAX,
    };
    let outcome = solve(&graph, &mut transfer, cap(8));
    assert!(!outcome.interrupted && outcome.converged);
}

/// **A change runs again only the methods it reaches.** A star: method 1
/// calls methods 2 to 9, all one weak component, and method 9 reports a change
/// on its first two visits. Methods 2 to 8 neither changed nor neighbour a
/// method that did, and are visited once; 9 runs again until it settles, and
/// its caller 1 after it.
#[test]
fn a_change_runs_again_only_the_methods_it_reaches() {
    struct Star {
        order: Vec<u32>,
    }
    impl SummaryTransfer<u32, u32> for Star {
        fn flow(&self) -> SummaryFlow {
            SummaryFlow::Bidirectional
        }

        fn visit(&mut self, method: &u32, summaries: &mut SummaryStore<u32, u32>) -> bool {
            self.order.push(*method);
            let visits = self.order.iter().filter(|seen| *seen == method).count();
            summaries.insert(*method, u32::try_from(visits).unwrap_or(u32::MAX));
            *method == 9 && visits <= 2
        }
    }
    let graph = CallGraph::from_edges(1u32..=9, |method| match method {
        1 => (2..=9).collect(),
        _ => Vec::new(),
    });
    let mut transfer = Star { order: Vec::new() };
    let outcome = solve(&graph, &mut transfer, cap(8));
    assert!(outcome.converged);
    let visits = |method: u32| {
        transfer
            .order
            .iter()
            .filter(|seen| **seen == method)
            .count()
    };
    for quiet in 2..=8 {
        assert_eq!(visits(quiet), 1, "method {quiet}: {:?}", transfer.order);
    }
    assert_eq!(visits(9), 3, "{:?}", transfer.order);
    // Queued from the first round when 9 first changes, once more after.
    assert_eq!(visits(1), 2, "{:?}", transfer.order);
}

/// **A change to a neighbour's summary runs again the methods that read it.**
/// Method 1 calls 2 and 3. The callee visited first reads its caller's
/// summary, as a parameter reads what its callers pass; the other refines
/// it, as a callee's return type reaches the call in its caller. The reader
/// is two calls from the writer, so what the writer's own change reaches
/// does not include it; the mark on 1 does. The control leaves the mark out,
/// and the reader keeps what it read before the writer ran.
#[test]
fn a_marked_change_runs_again_the_readers_of_the_changed_summary() {
    struct Siblings {
        marks: bool,
        reader: Option<u32>,
    }
    impl SummaryTransfer<u32, u32> for Siblings {
        fn flow(&self) -> SummaryFlow {
            SummaryFlow::Bidirectional
        }

        fn visit(&mut self, method: &u32, summaries: &mut SummaryStore<u32, u32>) -> bool {
            if *method == 1 {
                return false;
            }
            let reader = *self.reader.get_or_insert(*method);
            if reader == *method {
                let read = summaries.get(&1).copied().unwrap_or(0);
                let before = summaries.get(method).copied().unwrap_or(0);
                summaries.insert(*method, read);
                return before != read;
            }
            if summaries.get(&1).is_some() {
                return false;
            }
            summaries.insert(1, 7);
            if self.marks {
                summaries.mark_changed(1);
            }
            true
        }
    }
    let graph = CallGraph::from_edges(1u32..=3, |method| match method {
        1 => vec![2, 3],
        _ => Vec::new(),
    });
    let read = |marks: bool| {
        let mut transfer = Siblings {
            marks,
            reader: None,
        };
        let outcome = solve(&graph, &mut transfer, cap(8));
        assert!(outcome.converged);
        transfer
            .reader
            .and_then(|reader| outcome.summaries.get(&reader).copied())
    };
    assert_eq!(read(true), Some(7), "the reader reads what the writer gave");
    assert_eq!(read(false), Some(0), "the control: nothing visits it again");
}
