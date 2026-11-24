//! Cost of structured control-flow recovery as a function grows.
//!
//! Recovery builds, per loop, the convergence of the loop's paths, and splits
//! each loop entered at more than one block on trial, recovering the whole
//! function again for each split. A cost that is proportional to the graph for
//! each loop, or for each trial, makes a function of many loops superlinear in
//! its size; a cost proportional to the loop does not. These sweeps grow the
//! number of loops of two shapes, and print the time per block, which stays
//! flat when recovery is linear in the size it works on. Run with:
//!
//! ```text
//! cargo bench -p analyssa --bench structure
//! ```
//!
//! # Families
//!
//! | Family | Shape | What grows |
//! |--------|-------|------------|
//! | `natural` | a chain of loops, each with a diamond in its body | the loops the walk builds a convergence for |
//! | `entered_twice` | a chain of cycles each entered at two blocks | the loops, and the trials that split them |

use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use analyssa::{
    analysis::structure::{StructureOptions, structure},
    graph::{GraphBase, NodeId, Predecessors, Successors},
};

/// How many sweeps of each size are timed; the fastest is reported.
const SWEEPS: usize = 3;

/// A graph whose adjacency is written out directly.
struct Cfg {
    /// Successors per node.
    successors: Vec<Vec<NodeId>>,

    /// Predecessors per node.
    predecessors: Vec<Vec<NodeId>>,
}

impl Cfg {
    /// Returns the graph of `count` nodes with the given edges.
    fn of(count: usize, edges: &[(usize, usize)]) -> Self {
        let mut successors = vec![Vec::new(); count];
        let mut predecessors = vec![Vec::new(); count];
        for (from, to) in edges {
            if let Some(row) = successors.get_mut(*from) {
                row.push(NodeId::new(*to));
            }
            if let Some(row) = predecessors.get_mut(*to) {
                row.push(NodeId::new(*from));
            }
        }
        Self {
            successors,
            predecessors,
        }
    }

    /// Returns a chain of `loops` loops, each with a diamond for a body.
    ///
    /// Loop `i` has its header at `5i`, the diamond at `5i + 1 ..= 5i + 3`, its
    /// latch at `5i + 4`, and leaves into the next loop's header.
    fn natural(loops: usize) -> Self {
        let mut edges = Vec::new();
        for index in 0..loops {
            let base = index.saturating_mul(5);
            edges.extend([
                (base, base.saturating_add(1)),
                (base, base.saturating_add(2)),
                (base.saturating_add(1), base.saturating_add(3)),
                (base.saturating_add(2), base.saturating_add(3)),
                (base.saturating_add(3), base.saturating_add(4)),
                (base.saturating_add(4), base),
                (base, base.saturating_add(5)),
            ]);
        }
        Self::of(loops.saturating_mul(5).saturating_add(1), &edges)
    }

    /// Returns a chain of `loops` cycles, each entered at two blocks.
    ///
    /// Cycle `i` is `3i + 1 <-> 3i + 2`, entered from `3i` at either block and
    /// left from `3i + 1` into the next cycle's entry block.
    fn entered_twice(loops: usize) -> Self {
        let mut edges = Vec::new();
        for index in 0..loops {
            let base = index.saturating_mul(3);
            edges.extend([
                (base, base.saturating_add(1)),
                (base, base.saturating_add(2)),
                (base.saturating_add(1), base.saturating_add(2)),
                (base.saturating_add(2), base.saturating_add(1)),
                (base.saturating_add(1), base.saturating_add(3)),
            ]);
        }
        Self::of(loops.saturating_mul(3).saturating_add(1), &edges)
    }
}

impl GraphBase for Cfg {
    fn node_count(&self) -> usize {
        self.successors.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.successors.len()).map(NodeId::new)
    }
}

impl Successors for Cfg {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.successors
            .get(node.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }
}

impl Predecessors for Cfg {
    fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.predecessors
            .get(node.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }
}

/// Times the fastest of [`SWEEPS`] recoveries of `graph`.
fn fastest(graph: &Cfg) -> Duration {
    (0..SWEEPS)
        .map(|_| {
            let start = Instant::now();
            black_box(structure(
                black_box(graph),
                NodeId::new(0),
                &StructureOptions::bare(graph.node_count(), &[]),
            ));
            start.elapsed()
        })
        .min()
        .unwrap_or_default()
}

/// Prints one family's sweep: blocks, time and time per block.
fn sweep(family: &str, sizes: &[usize], build: fn(usize) -> Cfg) {
    println!("{family}");
    for loops in sizes {
        let graph = build(*loops);
        let time = fastest(&graph);
        let blocks = graph.node_count();
        println!(
            "  {blocks:>6} blocks {:>10.2} ms {:>8.2} us/block",
            time.as_secs_f64() * 1e3,
            time.as_secs_f64() * 1e6 / blocks.max(1) as f64
        );
    }
}

fn main() {
    // `cargo test` runs this binary without `--bench`, unoptimised and under
    // coverage: one small size per family keeps that run a smoke test.
    let full = env::args().any(|argument| argument == "--bench");
    let (natural, entered_twice): (&[usize], &[usize]) = if full {
        (&[100, 200, 400, 800, 1600], &[50, 100, 200, 400, 800])
    } else {
        (&[10], &[5])
    };
    sweep("natural", natural, Cfg::natural);
    sweep("entered_twice", entered_twice, Cfg::entered_twice);
}
