//! Generic interprocedural summary driver: component order and the recursive
//! fixpoint.

use analyssa::interproc::{CallGraph, SummaryStore, SummaryTransfer, solve};

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
    let summaries = solve(&graph, &mut transfer, 8);

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
    solve(&graph, &mut transfer, 8);

    let visits_of_one = transfer.order.iter().filter(|m| **m == 1).count();
    assert_eq!(
        visits_of_one, 4,
        "iterates while the transfer reports change, then once more to confirm"
    );
}

/// A self-recursive singleton is recursive too — it is otherwise
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
    solve(&graph, &mut transfer, 8);
    assert!(
        transfer.order.len() > 1,
        "a self-recursive method is revisited"
    );
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
    solve(&graph, &mut transfer, 4);

    let visits_of_one = transfer.order.iter().filter(|m| **m == 1).count();
    assert_eq!(visits_of_one, 4, "bounded by the iteration cap");
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
