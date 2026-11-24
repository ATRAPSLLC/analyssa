//! The code that ends the function from each block: the straight-line path a
//! walk can place again wherever a path runs into it.

use crate::{
    analysis::structure::inputs::STATEMENTS_PER_GOTO,
    graph::{NodeId, Successors},
};

/// For each block, whether a straight-line path from it ends the function,
/// and what the path prints. Every block on the path has one successor but
/// the last, which has none, and none is `blocked`. A walk places such a path
/// again, as a tail
/// ([`Structurer::tail`](crate::analysis::structure::walk::Structurer::tail)),
/// where it prints no more than [`STATEMENTS_PER_GOTO`] statements
/// ([`Endings::repeats`]).
///
/// Every block on such a path begins one of its own, so the blocks that do
/// form a forest whose edges are their successors and whose roots are the
/// blocks with none. Whether a block lies on the path another begins is
/// whether it is that block's ancestor there, which the order a depth-first
/// walk enters and leaves the forest in answers at once, and what a path
/// prints is what its first block prints added to what its successor's path
/// does. A walk asks both at every branch it places, so answering them by
/// walking the path would cost the path's length each time.
pub struct Endings {
    /// For each block that begins a path, the positions the forest's walk
    /// entered and left it at; `None` for every other block.
    spans: Vec<Option<(usize, usize)>>,

    /// For each block that begins a path, the statements the path prints.
    printed: Vec<usize>,
}

/// How far [`Endings::of`] has decided a block.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Decided {
    Unknown,
    Walking,
    Ends,
    Not,
}

impl Endings {
    /// Finds the blocks of `graph` below `node_count` that begin a path
    /// ending the function, none of them `blocked`, each block printing
    /// `statements`. A successor at or past `node_count` names no block and
    /// is no edge.
    pub fn of<G: Successors>(
        graph: &G,
        node_count: usize,
        blocked: impl Fn(NodeId) -> bool,
        statements: impl Fn(NodeId) -> usize,
    ) -> Self {
        // The one successor of each block that can lie on a path, or `None`
        // for a block with none; blocks that cannot lie on one are decided.
        let (next, mut decided): (Vec<Option<NodeId>>, Vec<Decided>) = (0..node_count)
            .map(|index| {
                let node = NodeId::new(index);
                let mut successors = graph
                    .successors(node)
                    .filter(|successor| successor.index() < node_count);
                match (successors.next(), successors.next()) {
                    _ if blocked(node) => (None, Decided::Not),
                    (None, _) => (None, Decided::Ends),
                    (Some(successor), None) => (Some(successor), Decided::Unknown),
                    _ => (None, Decided::Not),
                }
            })
            .unzip();
        // Each undecided block follows its successors until one is decided;
        // a block met again on the same walk lies on a cycle, which ends
        // nothing. Every block on the walk takes the answer it reached.
        for start in 0..node_count {
            let mut walked = Vec::new();
            let mut index = start;
            let answer = loop {
                let Some(state) = decided.get_mut(index) else {
                    break Decided::Not;
                };
                match *state {
                    Decided::Unknown => {
                        *state = Decided::Walking;
                        walked.push(index);
                        match next.get(index).copied().flatten() {
                            Some(successor) => index = successor.index(),
                            None => break Decided::Not,
                        }
                    }
                    Decided::Walking => break Decided::Not,
                    known => break known,
                }
            };
            for index in walked {
                if let Some(state) = decided.get_mut(index) {
                    *state = answer;
                }
            }
        }
        // The forest's children of each block, then the order a depth-first
        // walk from each root enters and leaves them in.
        let mut children: Vec<Vec<NodeId>> = vec![Vec::new(); node_count];
        let ends = |index: usize| decided.get(index) == Some(&Decided::Ends);
        for (index, successor) in next.iter().enumerate() {
            if ends(index)
                && let Some(successor) = successor
                && let Some(siblings) = children.get_mut(successor.index())
            {
                siblings.push(NodeId::new(index));
            }
        }
        let mut spans: Vec<Option<(usize, usize)>> = vec![None; node_count];
        let mut printed = vec![0usize; node_count];
        let mut entered = vec![0usize; node_count];
        let mut clock = 0usize;
        let roots = (0..node_count).filter(|index| ends(*index) && next.get(*index) == Some(&None));
        for root in roots {
            let mut stack = vec![(NodeId::new(root), false)];
            while let Some((node, left)) = stack.pop() {
                let index = node.index();
                let onward = next
                    .get(index)
                    .copied()
                    .flatten()
                    .and_then(|successor| printed.get(successor.index()).copied())
                    .unwrap_or(0);
                if let Some(total) = printed.get_mut(index) {
                    *total = statements(node).saturating_add(onward);
                }
                if left {
                    if let (Some(span), Some(entry)) = (spans.get_mut(index), entered.get(index)) {
                        *span = Some((*entry, clock));
                    }
                } else {
                    if let Some(entry) = entered.get_mut(index) {
                        *entry = clock;
                    }
                    stack.push((node, true));
                    stack.extend(
                        children
                            .get(index)
                            .into_iter()
                            .flatten()
                            .map(|child| (*child, false)),
                    );
                }
                clock = clock.saturating_add(1);
            }
        }
        Self { spans, printed }
    }

    /// Whether a straight-line path from `node` ends the function.
    pub fn begins(&self, node: NodeId) -> bool {
        self.spans.get(node.index()).is_some_and(Option::is_some)
    }

    /// Whether a walk places the path that ends the function from `node`
    /// again where a path runs into it: one begins there, and prints no more
    /// than [`STATEMENTS_PER_GOTO`] statements, what the goto it saves is
    /// worth.
    pub fn repeats(&self, node: NodeId) -> bool {
        self.begins(node)
            && self
                .printed
                .get(node.index())
                .is_some_and(|printed| *printed <= STATEMENTS_PER_GOTO)
    }

    /// Whether `node` lies on the path that ends the function from `start`,
    /// `start` itself included; false where `start` begins none.
    pub fn runs_through(&self, start: NodeId, node: NodeId) -> bool {
        let span = |block: NodeId| self.spans.get(block.index()).copied().flatten();
        match (span(start), span(node)) {
            (Some((start_in, start_out)), Some((node_in, node_out))) => {
                node_in <= start_in && start_out <= node_out
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::graph::DirectedGraph;

    fn graph_of(count: usize, edges: &[(usize, usize)]) -> DirectedGraph<'static, (), ()> {
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        for _ in 0..count {
            graph.add_node(());
        }
        for (from, to) in edges {
            graph
                .add_edge(NodeId::new(*from), NodeId::new(*to), ())
                .expect("both ends exist");
        }
        graph
    }

    fn node(index: usize) -> NodeId {
        NodeId::new(index)
    }

    /// The path a walk takes from `start`, followed block by block: the
    /// oracle [`Endings`] answers without walking.
    fn walked(graph: &DirectedGraph<'static, (), ()>, start: usize) -> Option<Vec<usize>> {
        let mut path = vec![start];
        let mut at = start;
        loop {
            let successors: Vec<NodeId> = graph.successors(node(at)).collect();
            match successors.as_slice() {
                [] => return Some(path),
                [next] if !path.contains(&next.index()) => {
                    at = next.index();
                    path.push(at);
                }
                _ => return None,
            }
        }
    }

    /// **Two chains meeting before the end each run through their own blocks
    /// only.** `0 -> 1 -> 3 -> 4` and `2 -> 3`: 3 and 4 lie on the path from
    /// 0 and from 2, 1 only on the path from 0, 2 only on its own - though 1
    /// and 2 lie as far from the end. A branch at 5 and the cycle 6 <-> 7
    /// begin none, nor does 8, which runs into the branch; 9, blocked, begins
    /// none and stops 10 from beginning one.
    #[test]
    fn each_path_runs_through_its_own_blocks() {
        let graph = graph_of(
            11,
            &[
                (0, 1),
                (1, 3),
                (2, 3),
                (3, 4),
                (5, 4),
                (5, 3),
                (6, 7),
                (7, 6),
                (8, 5),
                (10, 9),
            ],
        );
        let endings = Endings::of(&graph, 11, |block| block == node(9), |_| 1);
        for start in 0..11 {
            let path = if start == 9 || start == 10 {
                None
            } else {
                walked(&graph, start)
            };
            assert_eq!(endings.begins(node(start)), path.is_some(), "{start}");
            for on in 0..11 {
                assert_eq!(
                    endings.runs_through(node(start), node(on)),
                    path.as_ref().is_some_and(|path| path.contains(&on)),
                    "{start} through {on}"
                );
            }
        }
    }

    /// **A successor past the node count is no edge.** A block whose only
    /// successor names no block ends the function.
    #[test]
    fn a_successor_past_the_count_is_no_edge() {
        let graph = graph_of(3, &[(0, 1), (1, 2)]);
        let endings = Endings::of(&graph, 2, |_| false, |_| 1);
        assert!(endings.begins(node(0)));
        assert!(endings.runs_through(node(0), node(1)));
        assert!(!endings.begins(node(2)));
    }

    /// **A chain as long as the binary likes is decided within the stack.**
    /// A hundred thousand blocks in a line, and the same line closed into a
    /// cycle, decided on a thread of 256 KiB.
    #[test]
    fn a_chain_as_long_as_the_binary_likes_is_decided_within_the_stack() {
        let decided = thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let count: usize = 100_000;
                let line: Vec<(usize, usize)> =
                    (1..count).map(|to| (to.saturating_sub(1), to)).collect();
                let open = Endings::of(&graph_of(count, &line), count, |_| false, |_| 1);
                let mut closed_edges = line;
                let last = count.saturating_sub(1);
                closed_edges.push((last, 0));
                let closed = Endings::of(&graph_of(count, &closed_edges), count, |_| false, |_| 1);
                (
                    open.begins(node(0)) && open.runs_through(node(0), node(last)),
                    !open.runs_through(node(last), node(0)),
                    (0..count).any(|index| closed.begins(node(index))),
                )
            })
            .expect("the thread starts")
            .join()
            .expect("the chain is decided within the stack");
        assert_eq!(decided, (true, true, false));
    }
}
