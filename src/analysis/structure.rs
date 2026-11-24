//! Structured control-flow recovery: an arbitrary CFG to a region tree.
//!
//! A decompiler's back end cannot print a graph. It needs a tree of statements -
//! sequences, `if`/`else`, loops, `switch` - and this module recovers one from
//! any control-flow graph the front end produces.
//!
//! # Totality
//!
//! The recovery is **total**: every graph yields a region tree that reproduces
//! its control flow exactly. There is no input for which structuring fails, and
//! no input for which it silently drops an edge. That is not a claim about
//! compiler output being well behaved - hand-written assembly, obfuscators,
//! `setjmp`, computed gotos and self-modifying dispatch all produce graphs no
//! `while` loop describes. Totality comes instead from [`Region::Goto`]: any
//! edge that no structured form can express becomes an explicit jump to a
//! labelled block, which is always available and always correct.
//!
//! Totality is quantified over three things, and each is established rather
//! than assumed:
//!
//! - **The index domain.** Every caller-supplied block set is a [`BlockSet`],
//!   which carries the node count it was built for and answers `contains` for
//!   any node at all. [`structure`] renormalises a set built for a
//!   different width once, at the boundary, and counts the renormalisation in
//!   [`StructureMetrics::resized_inputs`]; it also rejects an entry outside the
//!   graph, and drops successor ids the graph reports past its own node count.
//! - **The depth of the output.** No tree this module returns nests deeper than
//!   [`StructureOptions::max_depth`]; see [`Structured`].
//! - **The scope a decision belongs to.** Every walk that enters a region -
//!   an arm, a loop body, a protected body, a handler, a filter - enters it
//!   through one scope operation, so the confinement, the enclosing follow and
//!   the loop-frame barrier are decided together and released together.
//!
//! Quality is therefore a *measurement*, not a precondition. The output is
//! better when it uses fewer gotos, and [`StructureMetrics::gotos`] counts them,
//! but a high count is ugly rather than wrong.
//!
//! # How it works
//!
//! The recovery walks the graph forward from the entry, keeping three facts:
//!
//! - **First arrival**, to know which block owns which region. A block is
//!   placed where the walk first reaches it, and every later arrival is a
//!   transfer to that placement.
//! - **Post-dominance**, to find the *join* of a conditional - the first block
//!   both arms reach. That join bounds the arms and resumes the parent
//!   sequence, which is what turns a diamond into `if`/`else` instead of two
//!   gotos. See [`compute_post_dominators`](crate::graph::algorithms::compute_post_dominators).
//! - **Natural loops**, to know which back edges are `continue` and which exit
//!   edges are `break`.
//!
//! Any block reached a second time has already been emitted, so it becomes a
//! goto. This single rule is what makes irreducible control flow terminate:
//! a cycle with two entries has no back edge dominating it, so no loop is
//! detected, the second entry is simply a revisit, and the cycle prints as a
//! labelled block with a goto.
//!
//! # Loops
//!
//! A natural loop is classified by *where it tests*, which is a structural
//! question and not the [`LoopType`](crate::analysis::loops::LoopType) the loop analysis
//! records for canonicalisation:
//!
//! - the header branches out of the body - [`StructuredLoop::While`]
//! - a single latch branches out - [`StructuredLoop::DoWhile`]
//! - neither - [`StructuredLoop::Endless`], exited by `break` (or not at all)
//!
//! A loop with several latches is still one loop: every latch becomes a
//! `continue`. A loop leaving to several different blocks keeps one of them as
//! the follow - the block control resumes at - and reaches the others by goto.
//!
//! # Labelled transfers
//!
//! [`Region::Break`] and [`Region::Continue`] both name the loop header they
//! act on, and [`Region::SwitchBreak`] the head of the switch it leaves,
//! rather than implying the innermost construct. A break out of two nested
//! loops is expressible in the tree; whether the printer renders it as a
//! labelled break or falls back to a goto is the printer's decision, made with
//! its target language in view, and is not lost here.

use std::collections::BTreeMap;

use crate::graph::{GraphBase, NodeId, Predecessors, Successors};
/// The crate's handler taxonomy, defined beside the exception table it comes
/// from and re-exported here so a caller reading a recovered [`HandlerRegion`]
/// finds it alongside.
///
/// One definition, not a recovery-side copy: two enums with the same four
/// variants can disagree about a clause, and nothing would catch it.
pub use crate::ir::exception::HandlerKind;

mod cases;
mod conditions;
mod convergence;
mod endings;
mod facts;
mod follow;
mod inputs;
mod loops;
mod options;
mod regions;
mod split;
mod walk;

use self::{
    inputs::{Inputs, Statements},
    options::{normalised, normalised_region},
    regions::Attempt,
};
pub use self::{
    options::{DEFAULT_MAX_DEPTH, StructureOptions},
    regions::{
        BlockSet, CaseRelation, CaseTest, HandlerFilter, HandlerRegion, IfRegion, LoopRegion,
        Predicate, ProtectedHandler, ProtectedHandlerKind, ProtectedRegion, Region,
        StructureMetrics, Structured, StructuredLoop, SwitchCase, SwitchRegion, TryRegion,
    },
};

/// Recovers structured control flow for `graph` entered at `entry`, as
/// `options` allows.
///
/// Always succeeds, for every graph and every entry. See the module
/// documentation for what totality costs and how to measure it. This is the
/// one boundary at which the caller's index domain is closed: block sets are
/// renormalised to the graph's node count, an `entry` outside the graph
/// recovers as [`Region::Empty`], and successor ids the graph reports past its
/// own node count contribute no edge.
///
/// `graph` must contain only the edges its terminators take, which is what
/// [`SsaCfg::from_ssa`](crate::analysis::SsaCfg::from_ssa) builds. A graph that
/// also carries exception edges makes a block ending in an unconditional jump
/// look like a conditional branch, and it would be recovered as one - so a
/// caller building the graph from stored rows must drop them there. Handler
/// blocks are therefore unreachable from the entry, since the edge into a
/// handler is taken by the runtime. They are placed inside their
/// [`Region::Try`], so the tree still accounts for them - including when the
/// region's own entry is unreachable too.
///
/// A caller holding a bare graph states [`StructureOptions::bare`]; a caller
/// holding an `SsaFunction` has
/// [`structure_ssa`](crate::analysis::recovery::structure_ssa), which derives
/// the options from the instructions each block holds and the regions from
/// the exception table.
///
/// # Examples
///
/// ```rust
/// use analyssa::{
///     analysis::structure::{StructureOptions, structure},
///     graph::{DirectedGraph, GraphBase},
/// };
///
/// let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
/// let head = graph.add_node(());
/// let then_arm = graph.add_node(());
/// let join = graph.add_node(());
/// graph.add_edge(head, then_arm, ()).unwrap();
/// graph.add_edge(head, join, ()).unwrap();
/// graph.add_edge(then_arm, join, ()).unwrap();
///
/// let options = StructureOptions::bare(graph.node_count(), &[]);
/// let recovered = structure(&graph, head, &options);
/// assert_eq!(recovered.metrics.ifs, 1);
/// assert_eq!(recovered.metrics.gotos, 0);
/// ```
#[must_use]
pub fn structure<G>(graph: &G, entry: NodeId, options: &StructureOptions<'_>) -> Structured
where
    G: GraphBase + Successors + Predecessors,
{
    let node_count = graph.node_count();
    let mut resized = 0usize;
    let inputs = Inputs {
        condition_only: normalised(node_count, &options.condition_only, &mut resized),
        condition_reachability: normalised(
            node_count,
            &options.condition_reachability,
            &mut resized,
        ),
        condition_expressible: normalised(node_count, &options.condition_expressible, &mut resized),
        condition_operand: normalised(node_count, &options.condition_operand, &mut resized),
        no_return: normalised(node_count, &options.no_return, &mut resized),
        edge_blocks: normalised(node_count, &options.edge_blocks, &mut resized),
        regions: options
            .regions
            .iter()
            .map(|region| normalised_region(node_count, region, &mut resized))
            .collect(),
        case_tests: options.case_tests.clone(),
        repeated_tests: options.repeated_tests.clone(),
        threaded_tests: options.threaded_tests.clone(),
        statements: Statements(options.statements.clone()),
        max_depth: options.max_depth.max(1),
        preferred_joins: BlockSet::new(node_count),
        fold_statements: false,
        fall_out: false,
        switch_tails: false,
        route: false,
    };

    let mut structured = if node_count == 0 || entry.index() >= node_count {
        Attempt::unreached(node_count).structured
    } else {
        // Statements written inside a condition are the cost of a goto, or
        // of code placed twice, that they save, and only that: the recovery
        // that folds every block whose statements can be said there is kept
        // where it needs fewer gotos than the one that nests them, or as many
        // with less code placed twice. Each is recovered to the end before
        // the two are compared, since what every later trial saves depends
        // on it. Where no trial kept a block out for the setting alone, the
        // folding recovery would make the same choices, and is not made.
        //
        // Each order is recovered again with branches whose arms meet only
        // where the function ends rejoining where their region does
        // ([`Inputs::fall_out`]), where the first recovery still jumps or
        // places code twice: one that needs no goto only for the copies its
        // splits made can need none of them with the arms rejoined. Every
        // later trial reads the recovery it makes, so it is finished like the
        // first and compared with it as the orders are.
        let polar = normalised(node_count, &options.polar, &mut resized);
        let mut kept: Option<(Attempt, Variant)> = None;
        for flipped in orientations(graph, &polar) {
            for fall_out in [false, true] {
                if fall_out && kept.as_ref().is_some_and(|(best, _)| best.cost() == (0, 0)) {
                    break;
                }
                // The tails a recovery jumped to, offered as joins
                // ([`Inputs::recovered`]), and a switch resuming at one
                // ([`Inputs::switch_tails`]), are judged as every other choice
                // is, on the recovery finished with them: a join that pays
                // where it is taken can cost more once the splits after it
                // have run, and one that costs there can let them remove more.
                for (offer_joins, switch_tails) in [(false, false), (true, false), (true, true)] {
                    let variant = Variant {
                        flipped: flipped.clone(),
                        fall_out,
                        offer_joins,
                        switch_tails,
                    };
                    let (finished, offers) = variant.finish(&inputs, graph, entry);
                    // Switch tails are tried where the offered joins left a
                    // goto and a switch that could resume at one.
                    let switch_trial =
                        finished.gotos() > 0 && finished.structured.metrics.switches > 0;
                    if kept.as_ref().is_none_or(|(best, _)| {
                        finished.rank(&inputs.condition_only) < best.rank(&inputs.condition_only)
                    }) {
                        kept = Some((finished, variant));
                    }
                    if !offers || (offer_joins && !switch_trial) {
                        break;
                    }
                }
            }
        }
        // The routes the caller proved ([`Inputs::route`]) are taken one unit
        // at a time - the routes that share a block, as a threaded edge and
        // the repeat it leaves - on the variant kept, each kept only where the
        // recovery finished with it costs less: a route reads as the same
        // program, so it is never kept for a tie-break, and one route can pay
        // where another beside it costs.
        if let Some((best, variant)) = &mut kept {
            let mut taken = Inputs {
                route: true,
                threaded_tests: BTreeMap::new(),
                repeated_tests: BTreeMap::new(),
                ..inputs.clone()
            };
            for unit in inputs.route_units() {
                if best.gotos() == 0 {
                    break;
                }
                let mut trial = taken.clone();
                trial.threaded_tests.extend(unit.threaded);
                trial.repeated_tests.extend(unit.repeated);
                let (finished, _) = variant.finish(&trial, graph, entry);
                if finished.cost() < best.cost() {
                    *best = finished;
                    taken = trial;
                }
            }
        }
        kept.map_or_else(|| Attempt::unreached(node_count), |(attempt, _)| attempt)
            .structured
    };
    // Tidied once kept: the recoveries are weighed as the walk wrote them.
    structured.drop_onward_gotos();
    structured.metrics.resized_inputs = resized;
    structured
}

/// One way [`structure`] recovers a graph: the order it reads the polar
/// blocks in, whether branches meeting only where the function ends rejoin
/// where their region does ([`Inputs::fall_out`]), and whether tails are
/// offered as joins ([`Inputs::recovered`]) and switches resume at one
/// ([`Inputs::switch_tails`]).
struct Variant {
    /// The polar blocks read the other way round.
    flipped: BlockSet,

    /// See [`Inputs::fall_out`].
    fall_out: bool,

    /// Whether the tails a recovery jumped to are offered as joins.
    offer_joins: bool,

    /// See [`Inputs::switch_tails`].
    switch_tails: bool,
}

impl Variant {
    /// Recovers `graph` from `entry` this way with `inputs`, the joins a goto
    /// still names split last ([`Inputs::split_joins`]), and returns the
    /// recovery with whether it offered any join: one that jumped to nothing
    /// and placed no tail twice offers none, and one that folds statements
    /// into its conditions does so only where the recovery nesting them cost
    /// more ([`Inputs::best`]), whose tails the joins are offered from.
    fn finish<G>(&self, inputs: &Inputs, graph: &G, entry: NodeId) -> (Attempt, bool)
    where
        G: GraphBase + Successors + Predecessors,
    {
        let view = Oriented {
            graph,
            flipped: &self.flipped,
        };
        let reoriented = Inputs {
            fall_out: self.fall_out,
            switch_tails: self.switch_tails,
            ..inputs.reoriented(&self.flipped)
        };
        let (recovered, current, oriented) = reoriented.best(&view, entry, self.offer_joins);
        let offers = oriented.fold_statements
            || !recovered.structured.labels.is_empty()
            || !recovered.rejoining_tails.is_empty();
        // The routes the graph recovered took, which splits keep.
        let repeats = current.repeats().clone();
        let mut finished = oriented.split_joins(recovered, current, entry);
        finished.structured.repeats = repeats;
        finished.reorient(&self.flipped);
        (finished, offers)
    }
}

/// The orders a recovery reads the polar blocks of `graph` in
/// ([`StructureOptions::polar`]), each as the set of blocks whose two
/// successors it reads the other way round from the order given: every polar
/// block's in ascending order of its successors' ids, then in descending
/// order. One order, the one given, where no polar block has two successors.
fn orientations<G>(graph: &G, polar: &BlockSet) -> Vec<BlockSet>
where
    G: GraphBase + Successors,
{
    let node_count = graph.node_count();
    let mut ascending = BlockSet::new(node_count);
    let mut descending = BlockSet::new(node_count);
    let mut two_way = false;
    for node in graph.node_ids() {
        if !polar.contains(node) {
            continue;
        }
        let mut successors = graph.successors(node);
        let (Some(first), Some(second), None) =
            (successors.next(), successors.next(), successors.next())
        else {
            continue;
        };
        if first == second {
            continue;
        }
        two_way = true;
        if first > second {
            ascending.insert(node);
        } else {
            descending.insert(node);
        }
    }
    if two_way {
        vec![ascending, descending]
    } else {
        vec![BlockSet::new(node_count)]
    }
}

/// A graph whose blocks of `flipped` list their two successors the other way
/// round from `graph`.
struct Oriented<'g, G> {
    /// The graph read.
    graph: &'g G,
    /// The blocks read the other way round.
    flipped: &'g BlockSet,
}

impl<G: GraphBase> GraphBase for Oriented<'_, G> {
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        self.graph.node_ids()
    }
}

impl<G: Successors> Successors for Oriented<'_, G> {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        let mut successors: Vec<NodeId> = self.graph.successors(node).collect();
        if self.flipped.contains(node) {
            successors.reverse();
        }
        successors.into_iter()
    }
}

impl<G: Predecessors> Predecessors for Oriented<'_, G> {
    fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.graph.predecessors(node)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::{
        analysis::{
            loops::detect_loops,
            structure::{
                convergence::{Convergence, ConvergingGraph},
                loops::LoopEnd,
                split::SplitGraph,
                walk::recover,
            },
        },
        bitset::BitSet,
        graph::{
            DirectedGraph,
            algorithms::{PostDominatorTree, compute_dominators, compute_post_dominators},
        },
    };

    /// A graph whose adjacency the test writes out directly.
    ///
    /// [`DirectedGraph`] cannot hold an edge to a node it does not have, so it
    /// cannot produce the one input the successor guard exists for: a host
    /// building a CFG from stale rows, whose successor ids outlive the blocks
    /// they named.
    struct RawGraph {
        /// Successor ids per node, unchecked.
        edges: Vec<Vec<NodeId>>,
    }

    impl GraphBase for RawGraph {
        fn node_count(&self) -> usize {
            self.edges.len()
        }

        fn node_ids(&self) -> impl Iterator<Item = NodeId> {
            (0..self.edges.len()).map(NodeId::new)
        }
    }

    impl Successors for RawGraph {
        fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
            self.edges
                .get(node.index())
                .cloned()
                .unwrap_or_default()
                .into_iter()
        }
    }

    impl Predecessors for RawGraph {
        fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
            let mut found = Vec::new();
            for (index, targets) in self.edges.iter().enumerate() {
                if targets.contains(&node) {
                    found.push(NodeId::new(index));
                }
            }
            found.into_iter()
        }
    }

    /// Builds a graph of `count` nodes with the given edges, in edge order.
    /// **A recovery reads a graph the same way whichever way its conditions
    /// are stated.** A loop `0 -> 1 -> 2 -> 3 -> 0` whose blocks 1, 2 and 3
    /// each leave it for 4 recovers with no goto whichever way round each
    /// test lists its two successors: the two trees are one, the second
    /// stating each test negated. The control: with no block declared polar
    /// ([`StructureOptions::polar`]) the order given decides, and the order
    /// that lists each way round the loop first needs a goto the other does
    /// not.
    #[test]
    fn a_recovery_reads_a_graph_the_same_whichever_way_its_conditions_are_stated() {
        let build = |exit_first: bool| {
            let mut edges = vec![(0, 1)];
            for (node, onward) in [(1, 2), (2, 3), (3, 0)] {
                if exit_first {
                    edges.extend([(node, 4), (node, onward)]);
                } else {
                    edges.extend([(node, onward), (node, 4)]);
                }
            }
            graph_of(5, &edges)
        };
        let tests = BlockSet::from_nodes(5, [1, 2, 3].map(NodeId::new));
        let recover = |graph: &DirectedGraph<'static, (), ()>, polar: &BlockSet| {
            let options = StructureOptions {
                polar: polar.clone(),
                ..StructureOptions::bare(5, &[])
            };
            structure(graph, NodeId::new(0), &options)
        };
        let (stated, exit_first) = (build(false), build(true));
        let onward = recover(&stated, &tests);
        let exiting = recover(&exit_first, &tests);
        assert_total(&stated, NodeId::new(0), &onward);
        assert_total(&exit_first, NodeId::new(0), &exiting);
        assert_eq!(onward.root.gotos(), 0, "{:#?}", onward.root);
        let mut exiting = Attempt {
            structured: exiting,
            ..Attempt::unreached(5)
        };
        exiting.reorient(&tests);
        assert_eq!(exiting.structured.root, onward.root);

        let none = BlockSet::new(5);
        assert_ne!(
            recover(&stated, &none).root.gotos(),
            recover(&exit_first, &none).root.gotos(),
            "control: the order given decides"
        );
    }

    /// **A branch whose arms meet only where the function ends falls out to
    /// where its region ends.** dash's `killcmd` (ARM): `0` is `argc <= 1`,
    /// `1` tests for an option, and the loop sending the signals begins at
    /// `3`, entered from `2` where there is none and from `19`, `37` and `40`
    /// deep in the option parsing. The parsing's other arms call what never
    /// returns (`16`, `35`, `41`) or print the signal list and return through
    /// `13`, which the sending loop returns through too, so the branches there
    /// meet nowhere but where the function ends, and each walk placed the loop
    /// in the first arm that reached it and jumped to it from the rest. Taken
    /// as the end of the option branch's region, the loop follows it, and each
    /// arm that reaches it falls out of the branches around it: one jump is
    /// left, into the list loop at `32` that two arms share. The control
    /// recovers with the loop offered as the join and branches keeping no join
    /// of their own: the jumps to `3` stay.
    #[test]
    fn a_branch_whose_arms_meet_where_the_function_ends_falls_out_to_its_regions_end() {
        let graph = graph_of(
            42,
            &[
                (0, 16),
                (0, 1),
                (1, 17),
                (1, 2),
                (2, 3),
                (3, 7),
                (4, 13),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 15),
                (7, 8),
                (8, 14),
                (8, 9),
                (9, 10),
                (10, 4),
                (10, 11),
                (11, 12),
                (11, 13),
                (12, 6),
                (14, 10),
                (15, 10),
                (17, 20),
                (17, 18),
                (18, 16),
                (18, 19),
                (19, 3),
                (20, 23),
                (21, 34),
                (21, 22),
                (22, 23),
                (23, 21),
                (23, 24),
                (24, 25),
                (24, 36),
                (25, 38),
                (25, 26),
                (26, 16),
                (26, 27),
                (27, 32),
                (27, 28),
                (28, 41),
                (28, 29),
                (30, 16),
                (30, 31),
                (31, 32),
                (32, 33),
                (33, 33),
                (33, 13),
                (34, 23),
                (34, 35),
                (36, 16),
                (36, 37),
                (37, 3),
                (38, 30),
                (38, 39),
                (39, 16),
                (39, 40),
                (40, 3),
            ],
        );
        let no_return = set_of(42, &[16, 35, 41]);
        let options = StructureOptions {
            no_return: no_return.clone(),
            ..StructureOptions::bare(42, &[])
        };
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(
            goto_targets(&recovered.root),
            BTreeSet::from([NodeId::new(32)]),
            "{:#?}",
            recovered.root
        );
        assert_eq!(recovered.metrics.gotos, 1, "{:#?}", recovered.root);

        let offered = |fall_out: bool| {
            let inputs = Inputs {
                statements: Statements::default(),
                condition_only: options.condition_only.clone(),
                condition_reachability: options.condition_reachability.clone(),
                condition_expressible: options.condition_expressible.clone(),
                condition_operand: options.condition_operand.clone(),
                no_return: no_return.clone(),
                edge_blocks: options.edge_blocks.clone(),
                regions: Vec::new(),
                case_tests: BTreeMap::new(),
                max_depth: DEFAULT_MAX_DEPTH,
                preferred_joins: set_of(42, &[3]),
                fold_statements: false,
                fall_out,
                switch_tails: false,
                route: false,
                repeated_tests: BTreeMap::new(),
                threaded_tests: BTreeMap::new(),
            };
            goto_targets(
                &recover(&SplitGraph::of(&graph), NodeId::new(0), &inputs)
                    .structured
                    .root,
            )
        };
        assert!(offered(false).contains(&NodeId::new(3)));
        assert!(!offered(true).contains(&NodeId::new(3)));
    }

    fn graph_of(count: usize, edges: &[(usize, usize)]) -> DirectedGraph<'static, (), ()> {
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        for _ in 0..count {
            graph.add_node(());
        }
        for (from, to) in edges {
            graph
                .add_edge(NodeId::new(*from), NodeId::new(*to), ())
                .expect("edge endpoints exist");
        }
        graph
    }

    /// Nodes reachable from `entry`.
    fn reachable(graph: &DirectedGraph<'_, (), ()>, entry: NodeId) -> BTreeSet<NodeId> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![entry];
        while let Some(node) = stack.pop() {
            if seen.insert(node) {
                stack.extend(graph.successors(node));
            }
        }
        seen
    }

    /// Every block a [`Region::Goto`] in the tree targets.
    fn goto_targets(root: &Region) -> BTreeSet<NodeId> {
        goto_jumps(root).into_iter().collect()
    }

    /// The target of each [`Region::Goto`] in the tree, one per goto.
    ///
    /// Written with an explicit stack: a caller must be able to run it on a
    /// tree at the nesting bound.
    fn goto_jumps(root: &Region) -> Vec<NodeId> {
        let mut targets = Vec::new();
        let mut stack = vec![root];
        while let Some(region) = stack.pop() {
            match region {
                Region::Goto(target) => targets.push(*target),
                Region::Seq(parts) => stack.extend(parts.iter()),
                Region::If(inner) => {
                    stack.push(&inner.then_branch);
                    stack.extend(inner.else_branch.iter());
                }
                Region::Loop(inner) => stack.push(&inner.body),
                Region::Switch(inner) => stack.extend(inner.cases.iter().map(|case| &case.body)),
                Region::Try(inner) => {
                    stack.push(&inner.body);
                    for handler in &inner.handlers {
                        stack.extend(handler.filter.iter());
                        stack.push(&handler.body);
                    }
                }
                Region::Empty
                | Region::Block(_)
                | Region::Break(_)
                | Region::Continue(_)
                | Region::SwitchBreak(_) => {}
            }
        }
        targets
    }

    /// How deeply the tree nests, counted with an explicit stack so the
    /// measurement itself survives a tree at the bound.
    fn region_depth(root: &Region) -> usize {
        let mut deepest = 0usize;
        let mut stack = vec![(root, 1usize)];
        while let Some((region, depth)) = stack.pop() {
            deepest = deepest.max(depth);
            let next = depth.saturating_add(1);
            match region {
                Region::Seq(parts) => stack.extend(parts.iter().map(|part| (part, next))),
                Region::If(inner) => {
                    stack.push((&inner.then_branch, next));
                    stack.extend(inner.else_branch.iter().map(|arm| (arm, next)));
                }
                Region::Loop(inner) => stack.push((&inner.body, next)),
                Region::Switch(inner) => {
                    stack.extend(inner.cases.iter().map(|case| (&case.body, next)));
                }
                Region::Try(inner) => {
                    stack.push((&inner.body, next));
                    for handler in &inner.handlers {
                        stack.extend(handler.filter.iter().map(|filter| (filter, next)));
                        stack.push((&handler.body, next));
                    }
                }
                Region::Empty
                | Region::Block(_)
                | Region::Break(_)
                | Region::Continue(_)
                | Region::SwitchBreak(_)
                | Region::Goto(_) => {}
            }
        }
        deepest
    }

    /// Asserts the invariants that hold of every recovery, whatever the graph.
    ///
    /// `resized_inputs` is checked here rather than by an assertion inside the
    /// module: these tests run with `debug_assertions` on, so a `debug_assert!`
    /// at the normalisation boundary would abort the very fixtures written to
    /// prove a short set no longer panics. The counter is the loud signal, and
    /// this is where it is read.
    fn assert_recovery_invariants(recovered: &Structured, max_depth: usize) {
        assert_eq!(
            recovered.metrics.resized_inputs, 0,
            "a fixture handed the recovery a set built for another node count"
        );
        // Every goto C writes names a label, and only those do: a goto's
        // target, and a loop's or switch's follow or header past the one a
        // bare `break` or `continue` binds to.
        let jumps = recovered.root.jumps();
        assert_eq!(
            recovered.labels,
            jumps.iter().flatten().copied().collect::<BTreeSet<_>>(),
            "labels must be exactly the blocks the gotos name"
        );
        assert!(
            goto_targets(&recovered.root).is_subset(&recovered.labels),
            "a goto's target is labelled"
        );
        assert_eq!(
            recovered.metrics.gotos,
            jumps.len(),
            "the goto count is the gotos C writes"
        );
        // The bound is on the walk's descent, and one descent buys at most two
        // tree levels: the construct plus the sequence under it.
        let depth = region_depth(&recovered.root);
        let bound = max_depth.saturating_mul(2).saturating_add(4);
        assert!(
            depth <= bound,
            "tree nests {depth} deep, past the bound of {bound} for max_depth {max_depth}"
        );
        if let Some(after) = code_after_a_transfer(&recovered.root, &recovered.labels) {
            panic!(
                "{after:?} follows a transfer, and no goto names it: {:#?}",
                recovered.root
            );
        }
    }

    /// Whether control never runs past the end of `region` by the tree's own
    /// transfers: it ends in a `break`, `continue` or goto, or in a
    /// conditional both of whose arms do.
    fn transfers_away(region: &Region) -> bool {
        match region {
            Region::Break(_) | Region::Continue(_) | Region::Goto(_) | Region::SwitchBreak(_) => {
                true
            }
            Region::Seq(parts) => parts.last().is_some_and(transfers_away),
            Region::If(inner) => {
                transfers_away(&inner.then_branch)
                    && inner.else_branch.as_ref().is_some_and(transfers_away)
            }
            _ => false,
        }
    }

    /// The first region of `root` placed after a part of its sequence that
    /// transfers away ([`transfers_away`]) and that begins at no goto target:
    /// code no path runs.
    fn code_after_a_transfer<'r>(
        root: &'r Region,
        labels: &BTreeSet<NodeId>,
    ) -> Option<&'r Region> {
        let begins_labelled = |region: &Region| {
            let mut first = None;
            region.for_each_block(&mut |block| {
                first.get_or_insert(block);
            });
            first.is_some_and(|block| labels.contains(&block))
        };
        match root {
            Region::Seq(parts) => parts
                .windows(2)
                .find_map(|pair| {
                    (transfers_away(&pair[0]) && !begins_labelled(&pair[1])).then_some(&pair[1])
                })
                .or_else(|| {
                    parts
                        .iter()
                        .find_map(|part| code_after_a_transfer(part, labels))
                }),
            Region::If(inner) => code_after_a_transfer(&inner.then_branch, labels).or_else(|| {
                inner
                    .else_branch
                    .as_ref()
                    .and_then(|other| code_after_a_transfer(other, labels))
            }),
            Region::Loop(inner) => code_after_a_transfer(&inner.body, labels),
            Region::Switch(inner) => inner
                .cases
                .iter()
                .find_map(|case| code_after_a_transfer(&case.body, labels)),
            Region::Try(inner) => code_after_a_transfer(&inner.body, labels).or_else(|| {
                inner
                    .handlers
                    .iter()
                    .find_map(|handler| code_after_a_transfer(&handler.body, labels))
            }),
            _ => None,
        }
    }

    /// How many `continue`s in `region` name a `do`/`while` other than the
    /// innermost loop around them: written as a jump to its header, which
    /// the loop does not test at.
    fn outer_do_while_continues(region: &Region) -> usize {
        fn count(region: &Region, open: &mut Vec<(NodeId, bool)>) -> usize {
            match region {
                Region::Continue(header) => usize::from(
                    open.last()
                        .is_some_and(|(innermost, _)| innermost != header)
                        && open
                            .iter()
                            .any(|(open, posttested)| open == header && *posttested),
                ),
                Region::Seq(parts) => parts.iter().map(|part| count(part, open)).sum(),
                Region::If(branch) => {
                    count(&branch.then_branch, open)
                        + branch
                            .else_branch
                            .as_ref()
                            .map_or(0, |other| count(other, open))
                }
                Region::Loop(inner) => {
                    open.push((inner.header, inner.kind == StructuredLoop::DoWhile));
                    let inside = count(&inner.body, open);
                    open.pop();
                    inside
                }
                Region::Switch(switch) => switch
                    .cases
                    .iter()
                    .map(|case| count(&case.body, open))
                    .sum(),
                Region::Try(protected) => {
                    count(&protected.body, open)
                        + protected
                            .handlers
                            .iter()
                            .map(|handler| count(&handler.body, open))
                            .sum::<usize>()
                }
                Region::Empty
                | Region::Block(_)
                | Region::Break(_)
                | Region::Goto(_)
                | Region::SwitchBreak(_) => 0,
            }
        }
        count(region, &mut Vec::new())
    }

    /// The latches whose tests end the `do`/`while` loops of `region`.
    fn do_while_latches(region: &Region) -> BTreeSet<NodeId> {
        let mut latches = BTreeSet::new();
        let mut stack = vec![region];
        while let Some(region) = stack.pop() {
            match region {
                Region::Seq(parts) => stack.extend(parts),
                Region::If(branch) => {
                    stack.push(&branch.then_branch);
                    stack.extend(&branch.else_branch);
                }
                Region::Loop(inner) => {
                    if let (StructuredLoop::DoWhile, Some(Predicate::Test { block, .. })) =
                        (&inner.kind, &inner.predicate)
                    {
                        latches.insert(*block);
                    }
                    stack.push(&inner.body);
                }
                Region::Switch(switch) => {
                    stack.extend(switch.cases.iter().map(|case| &case.body));
                }
                Region::Try(protected) => {
                    stack.push(&protected.body);
                    stack.extend(protected.handlers.iter().map(|handler| &handler.body));
                }
                Region::Empty
                | Region::Block(_)
                | Region::Break(_)
                | Region::Continue(_)
                | Region::SwitchBreak(_)
                | Region::Goto(_) => {}
            }
        }
        latches
    }

    /// Asserts the recovery placed every reachable block exactly once.
    ///
    /// This is the totality property. It must hold for every graph, including
    /// the irreducible and multi-latch ones a compiler would not emit, because
    /// a tree that drops or duplicates a block does not describe the function.
    fn assert_total(graph: &DirectedGraph<'_, (), ()>, entry: NodeId, recovered: &Structured) {
        assert_bounded_total(graph, entry, recovered, DEFAULT_MAX_DEPTH);
    }

    /// [`assert_total`] for a recovery run with a non-default depth bound.
    fn assert_bounded_total(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        recovered: &Structured,
        max_depth: usize,
    ) {
        assert_recovery_invariants(recovered, max_depth);

        let placed = recovered.blocks();
        let unique: BTreeSet<NodeId> = placed.iter().copied().collect();
        assert_eq!(
            unique,
            reachable(graph, entry),
            "placement != reachable set"
        );

        // Only straight-line code that ends the function or its region, a
        // `do`/`while`'s latch whose statements a `continue` runs before its
        // test, or a block copied to give a loop one entry, may be placed
        // more than once, and every repeat must be accounted for. A copy of
        // straight-line code never repeats a branch, and a latch's copy
        // repeats only its statements: the branch is the loop's test.
        let latches = do_while_latches(&recovered.root);
        assert_eq!(
            placed.len().saturating_sub(unique.len()),
            recovered
                .metrics
                .replicated_tails
                .saturating_add(recovered.metrics.split_blocks),
            "unaccounted duplicate placement: {placed:?}"
        );
        for node in &unique {
            let count = placed.iter().filter(|entry| *entry == node).count();
            assert!(
                count == 1
                    || graph.successors(*node).nth(1).is_none()
                    || latches.contains(node)
                    || recovered.metrics.split_blocks > 0,
                "block {node:?} placed {count} times but branches"
            );
        }
    }

    /// Builds a block set over `count` nodes.
    fn set_of(count: usize, members: &[usize]) -> BlockSet {
        BlockSet::from_nodes(count, members.iter().map(|member| NodeId::new(*member)))
    }

    /// How many times `recovered` places block `block`.
    fn placements(recovered: &Structured, block: usize) -> usize {
        recovered
            .blocks()
            .iter()
            .filter(|&&placed| placed == NodeId::new(block))
            .count()
    }

    /// Every block placed anywhere in `region`.
    fn placed_in(region: &Region) -> BTreeSet<NodeId> {
        let mut placed = BTreeSet::new();
        region.for_each_block(&mut |node| {
            placed.insert(node);
        });
        placed
    }

    /// Asserts placement totality where a protected region contributes blocks
    /// the entry cannot reach, because the runtime takes the edge into them.
    fn assert_total_protected(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        regions: &[ProtectedRegion],
        recovered: &Structured,
    ) {
        assert_recovery_invariants(recovered, DEFAULT_MAX_DEPTH);

        let placed = recovered.blocks();
        let unique: BTreeSet<NodeId> = placed.iter().copied().collect();
        assert_eq!(
            placed.len().saturating_sub(unique.len()),
            recovered
                .metrics
                .replicated_tails
                .saturating_add(recovered.metrics.split_blocks),
            "unaccounted duplicate placement: {placed:?}"
        );

        let mut expected = reachable(graph, entry);
        for region in regions {
            expected.insert(region.entry);
            expected.extend(region.blocks.iter());
            for handler in &region.handlers {
                expected.insert(handler.entry);
                expected.extend(handler.blocks.iter());
                if let Some(filter) = handler.kind.filter() {
                    expected.insert(filter.entry);
                    expected.extend(filter.blocks.iter());
                }
            }
        }
        assert_eq!(unique, expected, "placement != reachable set");
    }

    /// Structures with every block declared free of side effects, which is the
    /// condition under which conditions may be merged.
    fn structure_merging(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        condition_only: &[usize],
    ) -> Structured {
        let options = StructureOptions {
            statements: Vec::new(),
            regions: &[],
            condition_only: set_of(graph.node_count(), condition_only),
            condition_reachability: BlockSet::new(graph.node_count()),
            condition_expressible: BlockSet::new(graph.node_count()),
            condition_operand: BlockSet::new(graph.node_count()),
            case_tests: BTreeMap::new(),
            no_return: BlockSet::new(graph.node_count()),
            edge_blocks: BlockSet::new(graph.node_count()),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(graph.node_count()),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        structure(graph, entry, &options)
    }

    /// Builds a recovery in which the merged blocks compute as well as test,
    /// which is what [`StructureOptions::condition_expressible`] admits.
    fn structure_expressing(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        condition_expressible: &[usize],
    ) -> Structured {
        let options = StructureOptions {
            statements: Vec::new(),
            regions: &[],
            condition_only: BlockSet::new(graph.node_count()),
            condition_reachability: BlockSet::new(graph.node_count()),
            condition_expressible: set_of(graph.node_count(), condition_expressible),
            condition_operand: BlockSet::new(graph.node_count()),
            case_tests: BTreeMap::new(),
            no_return: BlockSet::new(graph.node_count()),
            edge_blocks: BlockSet::new(graph.node_count()),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(graph.node_count()),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        structure(graph, entry, &options)
    }

    /// Structures every block as condition-only under an explicit depth bound.
    fn structure_bounded(
        graph: &DirectedGraph<'_, (), ()>,
        entry: NodeId,
        regions: &[ProtectedRegion],
        max_depth: usize,
    ) -> Structured {
        let options = StructureOptions {
            statements: Vec::new(),
            regions,
            condition_only: BlockSet::full(graph.node_count()),
            condition_reachability: BlockSet::new(graph.node_count()),
            condition_expressible: BlockSet::new(graph.node_count()),
            condition_operand: BlockSet::new(graph.node_count()),
            case_tests: BTreeMap::new(),
            no_return: BlockSet::new(graph.node_count()),
            edge_blocks: BlockSet::new(graph.node_count()),
            max_depth,
            polar: BlockSet::new(graph.node_count()),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        structure(graph, entry, &options)
    }

    /// Collects the loop header named by every break in the tree.
    fn collect_breaks(region: &Region, into: &mut Vec<NodeId>) {
        match region {
            Region::Break(header) => into.push(*header),
            Region::Seq(regions) => {
                for inner in regions {
                    collect_breaks(inner, into);
                }
            }
            Region::If(inner) => {
                collect_breaks(&inner.then_branch, into);
                if let Some(other) = &inner.else_branch {
                    collect_breaks(other, into);
                }
            }
            Region::Loop(inner) => collect_breaks(&inner.body, into),
            Region::Switch(inner) => {
                for case in &inner.cases {
                    collect_breaks(&case.body, into);
                }
            }
            Region::Try(inner) => {
                collect_breaks(&inner.body, into);
                for handler in &inner.handlers {
                    collect_breaks(&handler.body, into);
                }
            }
            _ => {}
        }
    }

    /// The latches every `do`/`while` loop tests.
    fn collect_latches(region: &Region, into: &mut Vec<NodeId>) {
        match region {
            Region::Seq(parts) => {
                for part in parts {
                    collect_latches(part, into);
                }
            }
            Region::If(branch) => {
                collect_latches(&branch.then_branch, into);
                if let Some(other) = &branch.else_branch {
                    collect_latches(other, into);
                }
            }
            Region::Loop(inner) => {
                if inner.kind == StructuredLoop::DoWhile
                    && let Some(predicate) = &inner.predicate
                {
                    predicate.for_each_block(&mut |block| into.push(block));
                }
                collect_latches(&inner.body, into);
            }
            Region::Switch(switch) => {
                for case in &switch.cases {
                    collect_latches(&case.body, into);
                }
            }
            Region::Try(region) => {
                collect_latches(&region.body, into);
                for handler in &region.handlers {
                    collect_latches(&handler.body, into);
                }
            }
            Region::Block(_)
            | Region::Goto(_)
            | Region::Break(_)
            | Region::Continue(_)
            | Region::SwitchBreak(_)
            | Region::Empty => {}
        }
    }

    /// The headers of every `do`/`while` loop whose body ends in a `break`, so
    /// that its predicate is never reached and the loop runs once.
    fn loops_run_once(region: &Region, into: &mut Vec<NodeId>) {
        match region {
            Region::Seq(parts) => {
                for part in parts {
                    loops_run_once(part, into);
                }
            }
            Region::If(branch) => {
                loops_run_once(&branch.then_branch, into);
                if let Some(other) = &branch.else_branch {
                    loops_run_once(other, into);
                }
            }
            Region::Loop(inner) => {
                let last = match &inner.body {
                    Region::Seq(parts) => parts.last(),
                    other => Some(other),
                };
                if inner.kind == StructuredLoop::DoWhile && matches!(last, Some(Region::Break(_))) {
                    into.push(inner.header);
                }
                loops_run_once(&inner.body, into);
            }
            Region::Switch(switch) => {
                for case in &switch.cases {
                    loops_run_once(&case.body, into);
                }
            }
            Region::Try(region) => {
                loops_run_once(&region.body, into);
                for handler in &region.handlers {
                    loops_run_once(&handler.body, into);
                }
            }
            Region::Block(_)
            | Region::Goto(_)
            | Region::Break(_)
            | Region::Continue(_)
            | Region::SwitchBreak(_)
            | Region::Empty => {}
        }
    }

    /// Counts protected regions in a subtree.
    fn count_tries(region: &Region, into: &mut usize) {
        match region {
            Region::Try(inner) => {
                *into = into.saturating_add(1);
                count_tries(&inner.body, into);
                for handler in &inner.handlers {
                    count_tries(&handler.body, into);
                }
            }
            Region::Seq(regions) => {
                for inner in regions {
                    count_tries(inner, into);
                }
            }
            Region::If(inner) => {
                count_tries(&inner.then_branch, into);
                if let Some(other) = &inner.else_branch {
                    count_tries(other, into);
                }
            }
            Region::Loop(inner) => count_tries(&inner.body, into),
            Region::Switch(inner) => {
                for case in &inner.cases {
                    count_tries(&case.body, into);
                }
            }
            _ => {}
        }
    }

    /// Returns the first [`Region::Try`] found anywhere in `region`.
    fn first_try(region: &Region) -> Option<&TryRegion> {
        match region {
            Region::Try(inner) => Some(inner),
            Region::Seq(parts) => parts.iter().find_map(first_try),
            Region::If(inner) => first_try(&inner.then_branch)
                .or_else(|| inner.else_branch.as_ref().and_then(first_try)),
            Region::Loop(inner) => first_try(&inner.body),
            Region::Switch(inner) => inner.cases.iter().find_map(|case| first_try(&case.body)),
            _ => None,
        }
    }

    // The index domain

    /// A condition-only set sized for the wrong graph is renormalised rather
    /// than indexed past, so the recovery still places every block.
    /// **A `do`/`while` whose latch an `if` arm reaches still ends at its
    /// test.** dash's `jobscmd` rotates `while ((m = nextopt("lp")))` into an
    /// inner loop (header 2, latch 3) inside an outer one (header 1), both
    /// leaving to 4. The header's branch post-dominates at the shared exit, so
    /// the `if` joined there: its arm placed the latch, the walk resumed at the
    /// exit, and the body ended in a `break` before the predicate - a loop that
    /// runs once, so `jobs -l -p` read only its first option.
    #[test]
    fn a_do_while_whose_latch_an_arm_reaches_ends_at_its_test() {
        let graph = graph_of(
            12,
            &[
                (0, 1),
                (1, 4),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 2),
                (3, 4),
                (4, 7),
                (4, 5),
                (5, 5),
                (5, 6),
                (7, 9),
                (7, 8),
                (8, 8),
                (8, 9),
                (9, 6),
                (9, 10),
                (10, 10),
                (10, 11),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        let mut once = Vec::new();
        loops_run_once(&recovered.root, &mut once);
        assert!(
            once.is_empty(),
            "loops that run once: {once:?}\n{:#?}",
            recovered.root
        );
    }

    /// **A path into a `do`/`while`'s latch from inside its body reaches the
    /// test.** dash's `0xc4b0` (`waitproc`'s caller) has an outer loop
    /// (header 1, latch 21) whose body nests two inner loops; a branch deep in
    /// it reaches the latch past a join inside the loop. Every arrival must
    /// either fall into the test or name a block the tree places: a `goto` to
    /// the latch, which prints as the loop's condition rather than as a block,
    /// names no label.
    #[test]
    fn every_path_into_a_do_while_latch_reaches_the_test() {
        let graph = graph_of(
            28,
            &[
                (0, 1),
                (1, 9),
                (1, 2),
                (2, 26),
                (2, 3),
                (3, 3),
                (3, 4),
                (4, 5),
                (5, 7),
                (5, 6),
                (6, 8),
                (6, 7),
                (7, 6),
                (7, 8),
                (8, 9),
                (9, 11),
                (9, 10),
                (10, 10),
                (10, 11),
                (11, 17),
                (11, 12),
                (12, 24),
                (12, 13),
                (13, 14),
                (14, 16),
                (14, 15),
                (15, 16),
                (16, 14),
                (16, 17),
                (17, 23),
                (17, 18),
                (18, 20),
                (18, 19),
                (19, 27),
                (19, 20),
                (20, 22),
                (20, 21),
                (21, 1),
                (21, 22),
                (23, 18),
                (24, 13),
                (24, 25),
                (26, 5),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        let mut once = Vec::new();
        loops_run_once(&recovered.root, &mut once);
        assert!(once.is_empty(), "loops that run once: {once:?}");
        let mut latches = Vec::new();
        collect_latches(&recovered.root, &mut latches);
        let jumped: Vec<_> = goto_targets(&recovered.root)
            .into_iter()
            .filter(|target| latches.contains(target))
            .collect();
        assert!(
            jumped.is_empty(),
            "gotos to a latch printed as a condition: {jumped:?}\n{:#?}",
            recovered.root
        );
    }

    /// The binary search gcc compiles `main`'s option switch to in `which`:
    /// `'v'`, `> 'v'`, `'i'`, `> 'i'`, `'a'`, `> 'a'`, `'V'`, with every
    /// value no test matches reaching the default `9` and `'V'` sharing `'v'`'s
    /// arm. `ranged` sends the values above `'v'` somewhere other than the
    /// default instead.
    fn option_switch(ranged: bool) -> (DirectedGraph<'static, (), ()>, StructureOptions<'static>) {
        let above_v = if ranged { 14 } else { 9 };
        let graph = graph_of(
            15,
            &[
                (0, 10),
                (0, 1),
                (1, above_v),
                (1, 2),
                (2, 11),
                (2, 3),
                (3, 9),
                (3, 4),
                (4, 12),
                (4, 5),
                (5, 9),
                (5, 6),
                (6, 10),
                (6, 9),
                (9, 13),
                (10, 13),
                (11, 13),
                (12, 13),
                (14, 13),
            ],
        );
        let test = |relation, constant| CaseTest {
            value: 7,
            relation,
            constant,
            holds: 0,
            entered: true,
        };
        let greater = CaseRelation::Greater { signed: true };
        let case_tests = BTreeMap::from([
            (NodeId::new(0), test(CaseRelation::Equal, 0x76)),
            (NodeId::new(1), test(greater, 0x76)),
            (NodeId::new(2), test(CaseRelation::Equal, 0x69)),
            (NodeId::new(3), test(greater, 0x69)),
            (NodeId::new(4), test(CaseRelation::Equal, 0x61)),
            (NodeId::new(5), test(greater, 0x61)),
            (NodeId::new(6), test(CaseRelation::Equal, 0x56)),
        ]);
        let options = StructureOptions {
            statements: Vec::new(),
            regions: &[],
            condition_only: set_of(15, &[1, 2, 3, 4, 5, 6]),
            condition_reachability: BlockSet::new(15),
            condition_expressible: BlockSet::new(15),
            condition_operand: BlockSet::new(15),
            case_tests,
            no_return: BlockSet::new(15),
            edge_blocks: BlockSet::new(15),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(15),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        (graph, options)
    }

    /// The first switch in `region`, depth first.
    fn first_switch(region: &Region) -> Option<&SwitchRegion> {
        match region {
            Region::Switch(switch) => Some(switch),
            Region::Seq(parts) => parts.iter().find_map(first_switch),
            Region::If(branch) => first_switch(&branch.then_branch)
                .or_else(|| branch.else_branch.as_ref().and_then(first_switch)),
            Region::Loop(inner) => first_switch(&inner.body),
            _ => None,
        }
    }

    /// **A binary search over one value is the `switch` it was compiled
    /// from.** gcc lowers a sparse `switch` to compares; recovered as `if`s it
    /// read as a tree of `if (x > 'i') {} else if ..` with empty arms on every
    /// ISA. Each value keeps its arm, `'V'` and `'v'` share one, and the
    /// default is the block every unmatched value reaches. The control sends
    /// the values above `'v'` elsewhere: no `switch` says that, so the root's
    /// tests stay `if`s and only the tree below them is one.
    #[test]
    fn a_binary_search_over_one_value_is_a_switch() {
        let (graph, options) = option_switch(false);
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        let switch = first_switch(&recovered.root).expect("a switch");
        assert_eq!(switch.head, NodeId::new(0));
        let arms: Vec<(Vec<i64>, NodeId, bool)> = switch
            .cases
            .iter()
            .map(|case| (case.values.clone(), case.target, case.default))
            .collect();
        assert_eq!(
            arms,
            vec![
                (vec![0x56, 0x76], NodeId::new(10), false),
                (vec![0x61], NodeId::new(12), false),
                (vec![0x69], NodeId::new(11), false),
                (Vec::new(), NodeId::new(9), true),
            ]
        );

        let (graph, options) = option_switch(true);
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        // The root's tree is no `switch` now, but the one below `> 'v'`'s
        // failing arm still is: every value it does not match reaches `9`.
        let switch = first_switch(&recovered.root).expect("the inner switch");
        assert_eq!(switch.head, NodeId::new(2), "{:#?}", recovered.root);
    }

    /// Equality tests of value 7 at 0..=3 (`0x88`, `0x5c`, `0x81`, `0x40`),
    /// the arms `4`, `5`, `6` and the default `9` meeting at `7`, then `8`.
    /// `guard` sends `0x88` straight to `8`, past the tail the others share:
    /// dash's `rmescapes` (`0x939c`) skipping its copy for `CTLQUOTEMARK`.
    fn guarded_switch(guard: bool) -> (DirectedGraph<'static, (), ()>, StructureOptions<'static>) {
        let skipped = if guard { 8 } else { 7 };
        let graph = graph_of(
            10,
            &[
                (0, skipped),
                (0, 1),
                (1, 4),
                (1, 2),
                (2, 5),
                (2, 3),
                (3, 6),
                (3, 9),
                (4, 7),
                (5, 7),
                (6, 7),
                (9, 7),
                (7, 8),
            ],
        );
        let test = |constant| CaseTest {
            value: 7,
            relation: CaseRelation::Equal,
            constant,
            holds: 0,
            entered: true,
        };
        let case_tests = BTreeMap::from([
            (NodeId::new(0), test(0x88)),
            (NodeId::new(1), test(0x5c)),
            (NodeId::new(2), test(0x81)),
            (NodeId::new(3), test(0x40)),
        ]);
        let options = StructureOptions {
            statements: Vec::new(),
            regions: &[],
            condition_only: set_of(10, &[1, 2, 3]),
            condition_reachability: BlockSet::new(10),
            condition_expressible: BlockSet::new(10),
            condition_operand: BlockSet::new(10),
            case_tests,
            no_return: BlockSet::new(10),
            edge_blocks: BlockSet::new(10),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(10),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        (graph, options)
    }

    /// **A tree of tests over one value is one switch whichever way its tests
    /// are stated.** The four tests of [`guarded_switch`], each listing its
    /// two successors the other way round and its `holds` index with them,
    /// recover as the same `switch` once its tests are read in the order given
    /// ([`StructureOptions::polar`], [`CaseTest::holds`]).
    #[test]
    fn a_tree_of_tests_is_one_switch_whichever_way_they_are_stated() {
        let polar = set_of(10, &[0, 1, 2, 3]);
        let (graph, mut options) = guarded_switch(false);
        options.polar = polar.clone();
        let stated = structure(&graph, NodeId::new(0), &options);
        assert!(stated.metrics.switches > 0, "{:#?}", stated.root);

        let reversed = graph_of(
            10,
            &[
                (0, 1),
                (0, 7),
                (1, 2),
                (1, 4),
                (2, 3),
                (2, 5),
                (3, 9),
                (3, 6),
                (4, 7),
                (5, 7),
                (6, 7),
                (9, 7),
                (7, 8),
            ],
        );
        let (_, mut other_options) = guarded_switch(false);
        other_options.polar = polar.clone();
        for test in other_options.case_tests.values_mut() {
            test.holds = 1;
        }
        let other = structure(&reversed, NodeId::new(0), &other_options);
        assert_total(&reversed, NodeId::new(0), &other);
        let mut other = Attempt {
            structured: other,
            ..Attempt::unreached(0)
        };
        other.reorient(&polar);
        assert_eq!(other.structured.root, stated.root);
    }

    /// **A value that skips the tail every other value shares guards the
    /// `switch` rather than joining it.** As a case its arm is empty and the
    /// tail lands inside one arm, which the others reach by goto; as the `if`
    /// around the `switch`, the tail follows it. The control sends the guarded
    /// value through the tail too, and the whole tree is one `switch`.
    #[test]
    fn a_value_that_skips_the_shared_tail_guards_the_switch() {
        let (graph, options) = guarded_switch(true);
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        let switch = first_switch(&recovered.root).expect("the guarded switch");
        assert_eq!(switch.head, NodeId::new(1), "{:#?}", recovered.root);
        assert_eq!(switch.follow, Some(NodeId::new(7)));
        assert!(
            goto_targets(&recovered.root).is_empty(),
            "{:#?}",
            recovered.root
        );

        let (graph, options) = guarded_switch(false);
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        let switch = first_switch(&recovered.root).expect("one switch");
        assert_eq!(switch.head, NodeId::new(0), "{:#?}", recovered.root);
    }

    /// **A test that computes the value it tests joins no tree.** The
    /// `guarded_switch` tests below the root write the value they test before
    /// testing it, as a variable that holds another value in each block does:
    /// they compare four values, not one, and no `switch` states them.
    #[test]
    fn a_test_of_a_value_it_computes_joins_no_switch() {
        let (graph, mut options) = guarded_switch(false);
        for node in 1..=3 {
            if let Some(test) = options.case_tests.get_mut(&NodeId::new(node)) {
                test.entered = false;
            }
        }
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        assert!(
            first_switch(&recovered.root).is_none(),
            "{:#?}",
            recovered.root
        );
    }

    /// `which`'s `path_search`: `0` tests `a`; `1` tests `b != 0` and sends a
    /// held `b` straight to `y` at `5`, past `4`, which tests `b == 0` again
    /// and sends a held `b` to `y` too; `2` tests `c` and runs `x` at `3`; `y`
    /// branches on, through `7` or not, and all meet at `6`. The compiler
    /// threaded `1`'s jump past `4`, so `y` has two entries. `skipped` is
    /// `4`'s test, and the tests of `b` read value 7. `looped` runs it all in
    /// a loop, as `path_search` does: entered from `8`, `6` goes back to `0`
    /// or leaves to `9`, so `4` reaches `1` through the back edge.
    fn threaded(
        skipped: CaseTest,
        looped: bool,
    ) -> (DirectedGraph<'static, (), ()>, StructureOptions<'static>) {
        let mut edges = vec![
            (0, 1),
            (0, 2),
            (1, 5),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 6),
            (4, 6),
            (4, 5),
            (5, 7),
            (5, 6),
            (7, 6),
        ];
        if looped {
            edges.extend([(8, 0), (6, 0), (6, 9)]);
        }
        let graph = graph_of(10, &edges);
        let case_tests = BTreeMap::from([
            (
                NodeId::new(1),
                CaseTest {
                    value: 7,
                    relation: CaseRelation::NotEqual,
                    constant: 0,
                    holds: 0,
                    entered: true,
                },
            ),
            (NodeId::new(4), skipped),
        ]);
        let options = StructureOptions {
            statements: Vec::new(),
            regions: &[],
            condition_only: set_of(10, &[1, 2, 4]),
            condition_reachability: BlockSet::new(10),
            condition_expressible: BlockSet::new(10),
            condition_operand: BlockSet::new(10),
            case_tests,
            no_return: BlockSet::new(10),
            edge_blocks: BlockSet::new(10),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(10),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        (graph, options)
    }

    /// **A jump threaded past a test its own condition decides is routed back
    /// through that test.** Through `4`, `y` has one entry and the function
    /// reads `if (!(a && b) && c) { x } else if (b) { y }` with no goto, in a
    /// loop or not: a back edge is a lap of a loop the graph has. The
    /// controls keep the jump: a `4` that computes the value it tests, and a
    /// `4` testing `b == 5`, which a held `b` does not decide. They say the
    /// rethreading declines them rather than counting the whole recovery's
    /// gotos, which another recovery of the same graph can lower by taking
    /// the tail as a join ([`Structurer::preferred`]).
    #[test]
    fn a_jump_threaded_past_a_test_it_decides_is_routed_through_it() {
        let skipped = |constant, entered| CaseTest {
            value: 7,
            relation: CaseRelation::Equal,
            constant,
            holds: 0,
            entered,
        };
        for looped in [false, true] {
            let entry = NodeId::new(if looped { 8 } else { 0 });
            let gotos = |test: CaseTest| {
                let (graph, options) = threaded(test, looped);
                let recovered = structure(&graph, entry, &options);
                assert_total(&graph, entry, &recovered);
                goto_targets(&recovered.root)
            };
            let declined = |test: CaseTest| {
                let (graph, options) = threaded(test, looped);
                let count = graph.node_count();
                let inputs = Inputs {
                    statements: Statements::default(),
                    condition_only: options.condition_only.clone(),
                    condition_reachability: options.condition_reachability.clone(),
                    condition_expressible: options.condition_expressible.clone(),
                    condition_operand: options.condition_operand.clone(),
                    no_return: options.no_return.clone(),
                    edge_blocks: options.edge_blocks.clone(),
                    regions: options.regions.to_vec(),
                    case_tests: options.case_tests.clone(),
                    max_depth: options.max_depth,
                    preferred_joins: BlockSet::new(count),
                    fold_statements: true,
                    fall_out: false,
                    switch_tails: false,
                    route: false,
                    repeated_tests: BTreeMap::new(),
                    threaded_tests: BTreeMap::new(),
                };
                inputs
                    .rethreaded(&SplitGraph::of(&graph), entry, &BlockSet::new(count))
                    .is_none()
            };
            assert!(
                gotos(skipped(0, true)).is_empty(),
                "routed through 4, looped: {looped}"
            );
            assert!(
                declined(skipped(0, false)),
                "4 computes its value, looped: {looped}"
            );
            assert!(
                declined(skipped(5, true)),
                "b == 5 is not decided, looped: {looped}"
            );
        }
    }

    /// **A test spelling the jump's own comparison is not routed through.**
    /// Three arms of a switch each test one flag and reach one tail, so each
    /// tail-bound edge could be routed through the next test, which decides
    /// it. Printed, that reads as the same question asked three times, once
    /// answered - `if (x == 0) { if (x == 0) { if (x == 0) {` - so a test
    /// that spells the jump's comparison is refused, and the arms keep the
    /// gotos that say they are arms. The control: the same graph with the
    /// skipped test taken the other way is the chain it reads as, and is
    /// routed through.
    #[test]
    fn a_test_spelling_the_jumps_own_comparison_is_not_routed_through() {
        let graph = graph_of(6, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 5)]);
        let jump = CaseTest {
            value: 7,
            relation: CaseRelation::Equal,
            constant: 0,
            holds: 0,
            entered: true,
        };
        let routed = |skipped: CaseTest| {
            let inputs = Inputs {
                statements: Statements::default(),
                condition_only: set_of(6, &[1, 2]),
                condition_reachability: BlockSet::new(6),
                condition_expressible: BlockSet::new(6),
                condition_operand: BlockSet::new(6),
                no_return: BlockSet::new(6),
                edge_blocks: BlockSet::new(6),
                regions: Vec::new(),
                case_tests: BTreeMap::from([(NodeId::new(1), jump), (NodeId::new(2), skipped)]),
                max_depth: DEFAULT_MAX_DEPTH,
                preferred_joins: BlockSet::new(6),
                fold_statements: true,
                fall_out: false,
                switch_tails: false,
                route: false,
                repeated_tests: BTreeMap::new(),
                threaded_tests: BTreeMap::new(),
            };
            inputs
                .rethreaded(&SplitGraph::of(&graph), NodeId::new(0), &BlockSet::new(6))
                .is_some()
        };
        assert!(!routed(jump), "the same comparison reads as a repeat");
        assert!(
            routed(CaseTest {
                relation: CaseRelation::NotEqual,
                holds: 1,
                ..jump
            }),
            "the other way round is the chain it reads as"
        );
    }

    /// **A test that reaches the jump stays skipped.** `1` tests `b == 0` and
    /// leads to `2`, which tests `b != 0` and sends a held `b` straight to
    /// `4`, a successor of `1` too: through `1`, the jump would close the
    /// loop `1`, `2`, `1` the program does not have.
    #[test]
    fn a_test_that_reaches_the_jump_is_not_routed_through() {
        let graph = graph_of(6, &[(0, 1), (0, 2), (1, 2), (1, 4), (2, 4), (2, 5), (4, 5)]);
        let test = |relation| CaseTest {
            value: 7,
            relation,
            constant: 0,
            holds: 0,
            entered: true,
        };
        let inputs = Inputs {
            statements: Statements::default(),
            condition_only: set_of(6, &[1, 2]),
            condition_reachability: BlockSet::new(6),
            condition_expressible: BlockSet::new(6),
            condition_operand: BlockSet::new(6),
            no_return: BlockSet::new(6),
            edge_blocks: BlockSet::new(6),
            regions: Vec::new(),
            case_tests: BTreeMap::from([
                (NodeId::new(1), test(CaseRelation::Equal)),
                (NodeId::new(2), test(CaseRelation::NotEqual)),
            ]),
            max_depth: DEFAULT_MAX_DEPTH,
            preferred_joins: BlockSet::new(6),
            fold_statements: true,
            fall_out: false,
            switch_tails: false,
            route: false,
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        let split = SplitGraph::of(&graph);
        assert!(
            inputs
                .rethreaded(&split, NodeId::new(0), &BlockSet::new(6))
                .is_none()
        );
    }

    /// **A test the jump does not already reach is skipped only where an
    /// answer depended on what a block reaches.** `1` tests `b != 0` and
    /// sends a held `b` to `4`; `3`, under the other arm of `0`, tests
    /// `b != 0` and sends a held `b` to `4` too, but nothing from `1` reaches
    /// `3`. Routed through it, `1` comes to reach `3`'s rows and everything
    /// they reach.
    ///
    /// That is a hazard exactly for a block whose foldability the host
    /// decided from what it reaches - whether a place a condition-only block
    /// defines reaches a reader - and only for one whose own walk crosses the
    /// new edge, which is the block the edge leaves and anything reaching it.
    /// This graph admits a reroute either way (`1` through `3` and `3`
    /// through `1`), so each is refused by a different block: with both
    /// folded blocks decided, neither reroute stands. Where the host says no
    /// membership depended on reachability (`condition_reachability` empty,
    /// which is what an SSA caller reports: a value is assigned once and the
    /// use index names every block reading it) both are admitted and the
    /// `&&` is recovered.
    #[test]
    fn a_test_the_jump_does_not_reach_is_not_routed_through() {
        let graph = graph_of(
            8,
            &[
                (0, 1),
                (0, 2),
                (2, 3),
                (1, 4),
                (1, 5),
                (3, 4),
                (3, 6),
                (4, 7),
                (5, 7),
                (6, 7),
            ],
        );
        // The two spell one question between them - `b != 0` and `b == 0`
        // taken the other way - rather than the same comparison twice, which
        // reads as a repeat and is refused for that alone.
        let jump = CaseTest {
            value: 7,
            relation: CaseRelation::NotEqual,
            constant: 0,
            holds: 0,
            entered: true,
        };
        let skipped = CaseTest {
            relation: CaseRelation::Equal,
            holds: 1,
            ..jump
        };
        for decided in [&[1usize, 3][..], &[]] {
            let inputs = Inputs {
                statements: Statements::default(),
                condition_only: set_of(8, &[1, 3]),
                condition_reachability: set_of(8, decided),
                condition_expressible: BlockSet::new(8),
                condition_operand: BlockSet::new(8),
                no_return: BlockSet::new(8),
                edge_blocks: BlockSet::new(8),
                regions: Vec::new(),
                case_tests: BTreeMap::from([(NodeId::new(1), jump), (NodeId::new(3), skipped)]),
                max_depth: DEFAULT_MAX_DEPTH,
                preferred_joins: BlockSet::new(8),
                fold_statements: true,
                fall_out: false,
                switch_tails: false,
                route: false,
                repeated_tests: BTreeMap::new(),
                threaded_tests: BTreeMap::new(),
            };
            let split = SplitGraph::of(&graph);
            assert_eq!(
                inputs
                    .rethreaded(&split, NodeId::new(0), &BlockSet::new(8))
                    .is_none(),
                !decided.is_empty(),
                "decided: {decided:?}"
            );
        }
    }

    /// A loop at `1`, left to `10`, around equality tests of value 7 at `2`,
    /// `3` and `6` selecting `4`, `5` and `7`, default `8`, every arm reaching
    /// the latch `9`. `leaves` sends `5` out of the loop instead: dash's
    /// `0xd1e4` option parsing.
    fn looped_switch(leaves: bool) -> (DirectedGraph<'static, (), ()>, StructureOptions<'static>) {
        let after_five = if leaves { 10 } else { 9 };
        let graph = graph_of(
            11,
            &[
                (0, 1),
                (1, 2),
                (1, 10),
                (2, 4),
                (2, 3),
                (3, 5),
                (3, 6),
                (6, 7),
                (6, 8),
                (4, 9),
                (5, after_five),
                (7, 9),
                (8, 9),
                (9, 1),
            ],
        );
        let test = |constant| CaseTest {
            value: 7,
            relation: CaseRelation::Equal,
            constant,
            holds: 0,
            entered: true,
        };
        let case_tests = BTreeMap::from([
            (NodeId::new(2), test(1)),
            (NodeId::new(3), test(2)),
            (NodeId::new(6), test(3)),
        ]);
        let options = StructureOptions {
            statements: Vec::new(),
            regions: &[],
            condition_only: set_of(11, &[2, 3, 6]),
            condition_reachability: BlockSet::new(11),
            condition_expressible: BlockSet::new(11),
            condition_operand: BlockSet::new(11),
            case_tests,
            no_return: BlockSet::new(11),
            edge_blocks: BlockSet::new(11),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(11),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        (graph, options)
    }

    /// **A tree whose arm leaves the enclosing loop stays the `if`s it was.**
    /// `break` inside a `switch` leaves only the `switch`, so the arm would
    /// reach the loop's exit by goto, where an `if` breaks. The control keeps
    /// every arm inside the loop and recovers the `switch`.
    #[test]
    fn a_tree_whose_arm_leaves_the_loop_stays_ifs() {
        let (graph, options) = looped_switch(true);
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        assert!(
            first_switch(&recovered.root).is_none(),
            "{:#?}",
            recovered.root
        );
        assert!(
            goto_targets(&recovered.root).is_empty(),
            "{:#?}",
            recovered.root
        );

        let (graph, options) = looped_switch(false);
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        let switch = first_switch(&recovered.root).expect("the switch");
        assert_eq!(switch.head, NodeId::new(2), "{:#?}", recovered.root);
    }

    /// **A path that ends without returning has no say in a join.** `0`
    /// branches to `1` or `2`; `1` either calls a function that does not return
    /// (`3`) or continues to `2`, which goes on to the branch at `4`: dash's `testcmd`
    /// (`0x12be0`) calling `error()` on a missing `]`. Over every path nothing
    /// post-dominates `0`, so the walk placed `2` inside `1`'s arm and reached
    /// it from the other by goto; over the paths that return, `2` is the join.
    /// The control declares no block, and the goto is back.
    ///
    /// Of the join rule alone ([`recover`]), not of a whole recovery: told
    /// the tail it jumped to, a second recovery takes `2` as the join of the
    /// branch at `0`, which dominates it ([`Structurer::preferred`]), so the
    /// control would read with no goto either way and say nothing about what
    /// the declaration decides.
    #[test]
    fn a_call_that_does_not_return_leaves_the_join_to_the_paths_that_do() {
        let graph = graph_of(7, &[(0, 1), (0, 2), (1, 3), (1, 2), (2, 4), (4, 5), (4, 6)]);
        let inputs = |no_return: &[usize]| Inputs {
            statements: Statements::default(),
            regions: Vec::new(),
            condition_only: BlockSet::new(7),
            condition_reachability: BlockSet::new(7),
            condition_expressible: BlockSet::new(7),
            condition_operand: BlockSet::new(7),
            case_tests: BTreeMap::new(),
            no_return: set_of(7, no_return),
            edge_blocks: BlockSet::new(7),
            max_depth: DEFAULT_MAX_DEPTH,
            preferred_joins: BlockSet::new(7),
            fold_statements: true,
            fall_out: false,
            switch_tails: false,
            route: false,
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        let recovered = recover(&graph, NodeId::new(0), &inputs(&[3])).structured;
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);

        let control = recover(&graph, NodeId::new(0), &inputs(&[])).structured;
        assert_total(&graph, NodeId::new(0), &control);
        assert_eq!(control.metrics.gotos, 1, "{:#?}", control.root);
    }

    /// **Code past a loop exit that only that exit reaches keeps its joins.**
    /// A `while` at `1` leaves to `9` or tests at `2`, which repeats through
    /// `8` or leaves to `3`. Past `3` a diamond (`4`, `5`) meets at `6`, which
    /// branches again before the return: `which`'s `func_search` printing a
    /// match. `6` lies outside the loop, but only the exit through `3` reaches
    /// it, so it is that exit's join rather than a block the loop must jump to.
    #[test]
    fn code_past_a_loop_exit_keeps_its_joins() {
        let graph = graph_of(
            13,
            &[
                (0, 1),
                (1, 2),
                (1, 9),
                (2, 3),
                (2, 8),
                (3, 4),
                (3, 5),
                (4, 6),
                (5, 6),
                (6, 7),
                (6, 11),
                (7, 12),
                (11, 12),
                (8, 1),
                (9, 10),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert_eq!(recovered.metrics.loops, 1);
    }

    /// **A conjunction inside a disjunction folds into it.** `0` and `1` each
    /// select `4` when they hold; otherwise `2` and `3` must both hold to reach
    /// it, and every other path reaches `5`: `if (a || b || (c && d)) x;`, the
    /// test `which`'s `path_clean_up` copies a character under. Neither of
    /// `2`'s arms is `4`, so it joins no chain of single tests; `c && d` as a
    /// whole does. The control declares `3` not condition-only, so the inner
    /// conjunction cannot form, and the shared `4`, which runs straight on to
    /// the join `5`, is placed twice.
    #[test]
    fn a_conjunction_inside_a_disjunction_folds_into_it() {
        let graph = graph_of(
            8,
            &[
                (0, 4),
                (0, 1),
                (1, 4),
                (1, 2),
                (2, 3),
                (2, 5),
                (3, 4),
                (3, 5),
                (4, 5),
                (5, 6),
                (5, 7),
            ],
        );
        let entry = NodeId::new(0);
        let recovered = structure_merging(&graph, entry, &[1, 2, 3]);
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        let test = |block| Predicate::Test {
            block: NodeId::new(block),
            negated: false,
        };
        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Any(vec![
                test(0),
                test(1),
                Predicate::All(vec![test(2), test(3)])
            ])
        );

        assert_eq!(
            recovered.metrics.replicated_tails, 0,
            "{:#?}",
            recovered.root
        );

        let control = structure_merging(&graph, entry, &[1, 2]);
        assert_total(&graph, entry, &control);
        assert_eq!(control.metrics.replicated_tails, 1, "{:#?}", control.root);
    }

    /// **A tail longer than a goto is worth is jumped to.** The control of
    /// [`a_conjunction_inside_a_disjunction_folds_into_it`] places the shared
    /// `4` twice; where `4` prints more statements than
    /// [`STATEMENTS_PER_GOTO`](inputs::STATEMENTS_PER_GOTO), the second
    /// arrival jumps to it instead, and it is placed once.
    #[test]
    fn a_tail_longer_than_a_goto_is_worth_is_jumped_to() {
        let graph = graph_of(
            8,
            &[
                (0, 4),
                (0, 1),
                (1, 4),
                (1, 2),
                (2, 3),
                (2, 5),
                (3, 4),
                (3, 5),
                (4, 5),
                (5, 6),
                (5, 7),
            ],
        );
        let entry = NodeId::new(0);
        let weighed = |tail: usize| {
            let mut statements = vec![1; 8];
            if let Some(slot) = statements.get_mut(4) {
                *slot = tail;
            }
            structure(
                &graph,
                entry,
                &StructureOptions {
                    statements,
                    condition_only: set_of(8, &[1, 2]),
                    ..StructureOptions::bare(8, &[])
                },
            )
        };
        let short = weighed(inputs::STATEMENTS_PER_GOTO);
        assert_total(&graph, entry, &short);
        assert_eq!(short.metrics.replicated_tails, 1, "{:#?}", short.root);
        let long = weighed(inputs::STATEMENTS_PER_GOTO.saturating_add(1));
        assert_total(&graph, entry, &long);
        assert_eq!(long.metrics.replicated_tails, 0, "{:#?}", long.root);
        assert_eq!(
            long.metrics.gotos,
            short.metrics.gotos.saturating_add(1),
            "{:#?}",
            long.root
        );
    }

    /// The loop headed at `header` in `region`, depth first.
    fn loop_at(region: &Region, header: NodeId) -> Option<&LoopRegion> {
        match region {
            Region::Loop(inner) if inner.header == header => Some(inner),
            Region::Loop(inner) => loop_at(&inner.body, header),
            Region::Seq(parts) => parts.iter().find_map(|part| loop_at(part, header)),
            Region::If(branch) => loop_at(&branch.then_branch, header).or_else(|| {
                branch
                    .else_branch
                    .as_ref()
                    .and_then(|other| loop_at(other, header))
            }),
            Region::Switch(switch) => switch
                .cases
                .iter()
                .find_map(|case| loop_at(&case.body, header)),
            _ => None,
        }
    }

    /// Collects every `break` that names a loop other than the innermost one
    /// open around it, which C can only say with a goto.
    fn outer_breaks(region: &Region, open: &mut Vec<NodeId>, into: &mut Vec<NodeId>) {
        match region {
            Region::Break(header) if open.last() != Some(header) => into.push(*header),
            Region::Loop(inner) => {
                open.push(inner.header);
                outer_breaks(&inner.body, open, into);
                open.pop();
            }
            Region::Seq(parts) => parts.iter().for_each(|part| outer_breaks(part, open, into)),
            Region::If(branch) => {
                outer_breaks(&branch.then_branch, open, into);
                if let Some(other) = &branch.else_branch {
                    outer_breaks(other, open, into);
                }
            }
            _ => {}
        }
    }

    /// **A join every way into which arrives as a copy is not placed after
    /// the conditional.** Where both arms of a conditional leave, its join is
    /// placed after it only for a path that still arrives there by jumping: a
    /// goto naming it, or a way in the walk has yet to place. A loop latch
    /// that runs straight on to its header is copied, with its `continue`,
    /// into each arm that reaches it, so once every way in is placed nothing
    /// arrives, and placed after the arms it is code no path runs: `argstr`'s
    /// loop (dash, `0x62dc`) printed its latch again after a `continue`. The
    /// graph is the smallest one the recovery's random inputs found, every
    /// edge needed.
    #[test]
    fn a_join_every_way_into_which_arrives_as_a_copy_is_not_placed() {
        let graph = graph_of(
            15,
            &[
                (0, 10),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 14),
                (3, 12),
                (6, 11),
                (8, 1),
                (8, 13),
                (10, 10),
                (10, 2),
                (10, 0),
                (11, 0),
                (12, 2),
                (12, 8),
                (13, 8),
                (13, 0),
                (14, 6),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
    }

    /// **A loop resumes where leaving it would otherwise cost a goto.** An
    /// outer loop at `1` runs an inner one at `3` on one path; the inner loop
    /// leaves to `8` when its test fails, to the return at `7` from `4`, or
    /// through `5` into `6`, which the outer body also reaches from `1`:
    /// `which`'s `path_clean_up` backing up over `..`. `8` and `7` end the
    /// function, so they are repeated where the loop leaves to them, and `5`
    /// is the one exit whose code runs on, so it is the inner follow. `8` is
    /// also the outer loop's follow, which a `break` inside the inner loop
    /// could not reach: the tail is repeated there instead. `4` chooses among
    /// three, a switch, and leaving the inner loop for `5` from its case is
    /// the one goto: a bare `break` there leaves the switch.
    #[test]
    fn a_loop_resumes_where_leaving_would_cost_a_goto() {
        let graph = graph_of(
            10,
            &[
                (0, 1),
                (1, 2),
                (1, 6),
                (2, 3),
                (3, 4),
                (3, 8),
                (4, 3),
                (4, 5),
                (4, 7),
                (5, 6),
                (6, 1),
                (6, 8),
                (8, 9),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 1, "{:#?}", recovered.root);
        let inner = loop_at(&recovered.root, NodeId::new(3)).expect("the inner loop");
        assert_eq!(inner.follow, Some(NodeId::new(5)), "{:#?}", recovered.root);
        let mut breaks = Vec::new();
        outer_breaks(&recovered.root, &mut Vec::new(), &mut breaks);
        assert!(breaks.is_empty(), "breaks out of an outer loop: {breaks:?}");
    }

    /// **A loop exit whose code ends on its own costs nothing where it is.**
    /// A `while` at `1` leaves to the return at `6` when its test fails; its
    /// body returns through `3`, whose arms meet at the return `8`, on a
    /// match: `which`'s `func_search`. Neither exit's code runs on, so the
    /// follow is the `while`'s own, and the match is placed inside the loop.
    #[test]
    fn an_exit_that_ends_on_its_own_leaves_the_loop_its_test() {
        let graph = graph_of(
            9,
            &[
                (0, 1),
                (1, 2),
                (1, 6),
                (2, 3),
                (2, 7),
                (3, 4),
                (3, 5),
                (4, 8),
                (5, 8),
                (7, 1),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        let looped = loop_at(&recovered.root, NodeId::new(1)).expect("the loop");
        assert_eq!(looped.kind, StructuredLoop::While, "{:#?}", recovered.root);
        assert_eq!(looped.follow, Some(NodeId::new(6)));
    }

    /// **An arm that leaves is a guard, and the other arm follows it.** `0`
    /// branches to `1` or `2`; one arm returns at once and the other does more
    /// work through `3` and `4` before its own return. Whichever side the
    /// return is on, it is the guard: the `if` has no `else`, and the work
    /// follows at the level of the `if`. When both arms return, the shorter
    /// is the guard.
    #[test]
    fn an_arm_that_leaves_is_a_guard() {
        let shape = |region: &Region| match region {
            Region::Seq(parts) => match parts.first() {
                Some(Region::If(branch)) => Some((
                    branch.predicate.clone(),
                    branch.else_branch.is_none(),
                    parts.len(),
                )),
                _ => None,
            },
            _ => None,
        };
        let test = |negated| Predicate::Test {
            block: NodeId::new(0),
            negated,
        };

        // The `then` arm (1) returns; the `else` arm (2) runs on.
        let graph = graph_of(5, &[(0, 1), (0, 2), (2, 3), (3, 4)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(
            shape(&recovered.root),
            Some((test(false), true, 4)),
            "{:#?}",
            recovered.root
        );
        assert_eq!(recovered.metrics.if_elses, 0);

        // The `else` arm (1) returns; the `then` arm (2) runs on.
        let graph = graph_of(5, &[(0, 2), (0, 1), (2, 3), (3, 4)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(
            shape(&recovered.root),
            Some((test(true), true, 4)),
            "{:#?}",
            recovered.root
        );

        // Both arms return, the `then` arm (2, 3) after more work than the
        // `else` arm (1): the shorter is the guard.
        let graph = graph_of(4, &[(0, 2), (0, 1), (2, 3)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(
            shape(&recovered.root),
            Some((test(true), true, 3)),
            "{:#?}",
            recovered.root
        );
    }

    #[test]
    fn a_short_condition_only_set_does_not_panic() {
        // Three sets are sized for no graph at all while the graph has four
        // nodes, and the recovery is documented to always succeed, so no read
        // of a set may index past its end.
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let options = StructureOptions {
            statements: Vec::new(),
            regions: &[],
            condition_only: BlockSet::new(0),
            condition_reachability: BlockSet::new(0),
            condition_expressible: BlockSet::new(0),
            condition_operand: BlockSet::new(4),
            case_tests: BTreeMap::new(),
            no_return: BlockSet::new(4),
            edge_blocks: BlockSet::new(4),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(4),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        let recovered = structure(&graph, NodeId::new(0), &options);

        assert_eq!(
            recovered.metrics.resized_inputs, 3,
            "every short set was resized"
        );
        assert_eq!(recovered.blocks().len(), 4, "and every block still placed");
    }

    /// A protected region whose block sets are sized to the region rather than
    /// the function is renormalised, so its handler blocks are still placed.
    #[test]
    fn a_region_sized_to_itself_does_not_panic() {
        // The region's set is sized to the region rather than to the function,
        // so `pick_follow` must not index it by function block.
        let graph = graph_of(7, &[(0, 1), (1, 2), (2, 5), (3, 4), (4, 5)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(3, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(5, &[3, 4]),
            }],
        }];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_eq!(recovered.metrics.resized_inputs, 2);
        assert_eq!(recovered.metrics.tries, 1);
        let placed = placed_in(&recovered.root);
        assert!(
            placed.contains(&NodeId::new(3)) && placed.contains(&NodeId::new(4)),
            "the handler's blocks are still placed: {placed:?}"
        );
    }

    /// An entry id the graph does not hold recovers as `Region::Empty`, with
    /// every block counted unreached and no labels.
    #[test]
    fn an_entry_outside_the_graph_is_an_empty_recovery() {
        let graph = graph_of(3, &[(0, 1), (1, 2)]);
        let recovered = structure(
            &graph,
            NodeId::new(9),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_eq!(recovered.root, Region::Empty);
        assert_eq!(recovered.metrics.blocks, 0);
        assert_eq!(recovered.metrics.unreached, 3);
        assert!(recovered.labels.is_empty());
    }

    /// A successor id past the node count names no block, so the branch reads
    /// as a one-armed block rather than a conditional.
    #[test]
    fn an_out_of_range_successor_contributes_no_edge() {
        // Block 0 branches to block 1 and to an id the graph does not hold, so
        // it is a one-armed block and not a conditional.
        let graph = RawGraph {
            edges: vec![vec![NodeId::new(1), NodeId::new(7)], Vec::new()],
        };
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );

        assert_eq!(recovered.metrics.ifs, 0, "an absent id is not a second arm");
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(
            recovered.blocks(),
            vec![NodeId::new(0), NodeId::new(1)],
            "and both real blocks are still placed"
        );
    }

    /// A plain diamond gives nothing up, so every degradation count reads zero.
    #[test]
    fn a_recovery_within_its_bounds_reports_no_degradation() {
        // The counts that say the recovery gave something up, read out by
        // name. A plain diamond gives nothing up, and naming them here is what
        // keeps a count from being added and then never looked at.
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        let StructureMetrics {
            blocks,
            unreached,
            resized_inputs,
            depth_limited,
            replicated_tails,
            ..
        } = recovered.metrics;
        assert_eq!(blocks, 4);
        assert_eq!(unreached, 0);
        assert_eq!(resized_inputs, 0);
        assert_eq!(depth_limited, 0);
        assert_eq!(replicated_tails, 0);
    }

    /// `BlockSet` answers membership for any id, dropping out-of-domain nodes
    /// and reporting the stated width.
    #[test]
    fn a_block_set_is_total_over_every_node_id() {
        let set = BlockSet::from_nodes(4, [NodeId::new(1), NodeId::new(9)]);
        assert_eq!(set.node_count(), 4);
        assert!(set.contains(NodeId::new(1)));
        assert!(
            !set.contains(NodeId::new(9)),
            "out of domain is not a member"
        );
        assert!(!set.contains(NodeId::new(usize::MAX)));
        assert_eq!(set.count(), 1);

        let widened = BlockSet::from_bits(6, BitSet::new(2));
        assert_eq!(widened.node_count(), 6);
        assert_eq!(BlockSet::full(3).count(), 3);
    }

    /// The walk stops at excluded nodes, so what lies only beyond them is not
    /// reached, and an excluded start reaches nothing.
    #[test]
    fn reachable_avoiding_does_not_walk_through_excluded_nodes() {
        // 0 -> 1 -> 2 -> 3, and 0 -> 4 -> 3.
        let graph = RawGraph {
            edges: vec![
                vec![NodeId::new(1), NodeId::new(4)],
                vec![NodeId::new(2)],
                vec![NodeId::new(3)],
                vec![],
                vec![NodeId::new(3)],
            ],
        };
        let excluded = BlockSet::from_nodes(5, [NodeId::new(1), NodeId::new(4)]);
        let reached = BlockSet::reachable_avoiding(&graph, 5, NodeId::new(0), &excluded);
        assert_eq!(reached.iter().collect::<Vec<_>>(), [NodeId::new(0)]);

        let only_one = BlockSet::from_nodes(5, [NodeId::new(1)]);
        let reached = BlockSet::reachable_avoiding(&graph, 5, NodeId::new(0), &only_one);
        assert_eq!(
            reached.iter().collect::<Vec<_>>(),
            [NodeId::new(0), NodeId::new(3), NodeId::new(4)]
        );

        let start_excluded = BlockSet::from_nodes(5, [NodeId::new(0)]);
        let reached = BlockSet::reachable_avoiding(&graph, 5, NodeId::new(0), &start_excluded);
        assert_eq!(reached.count(), 0);
    }

    /// An inferred handler body stops at the tail it shares with the ordinary
    /// path and at its protected code; the control, with nothing reached from
    /// the entry, runs on into the tail.
    #[test]
    fn an_inferred_handler_body_stops_at_the_shared_tail() {
        // 0 (protected) -> 2 (tail) -> 3; handler 1 -> 4 -> 2, and 4 -> 0.
        let graph = RawGraph {
            edges: vec![
                vec![NodeId::new(2)],
                vec![NodeId::new(4)],
                vec![NodeId::new(3)],
                vec![],
                vec![NodeId::new(2), NodeId::new(0)],
            ],
        };
        let protected = BlockSet::from_nodes(5, [NodeId::new(0)]);
        let entry_reach =
            BlockSet::reachable_avoiding(&graph, 5, NodeId::new(0), &BlockSet::new(5));
        let body =
            ProtectedHandler::inferred_body(&graph, 5, NodeId::new(1), &protected, &entry_reach);
        assert_eq!(
            body.iter().collect::<Vec<_>>(),
            [NodeId::new(1), NodeId::new(4)]
        );

        let unbounded = ProtectedHandler::inferred_body(
            &graph,
            5,
            NodeId::new(1),
            &protected,
            &BlockSet::new(5),
        );
        assert_eq!(
            unbounded.iter().collect::<Vec<_>>(),
            [
                NodeId::new(1),
                NodeId::new(2),
                NodeId::new(3),
                NodeId::new(4)
            ]
        );
    }

    // The depth of the output

    /// Diamonds nested past `max_depth` reach the bound and degrade rather than
    /// nest, and the bounded tree survives a derived clone.
    #[test]
    fn nested_diamonds_are_bounded_by_max_depth() {
        // `if (c0) { if (c1) { .. } }` nested 2000 deep: test i branches into
        // test i+1 or out to exit i, and exit i+1 falls into exit i, so exit i
        // post-dominates test i and every level is a real nesting level.
        let depth: usize = 2000;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for index in 0..depth {
            let exit = depth.saturating_add(index);
            if index.saturating_add(1) < depth {
                edges.push((index, index.saturating_add(1)));
                edges.push((index, exit));
                edges.push((exit.saturating_add(1), exit));
            } else {
                edges.push((index, depth.saturating_mul(2)));
                edges.push((index, exit));
                edges.push((depth.saturating_mul(2), exit));
            }
        }
        let graph = graph_of(depth.saturating_mul(2).saturating_add(1), &edges);
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions::bare(graph.node_count(), &[]),
        );

        assert_total(&graph, entry, &recovered);
        assert!(
            recovered.metrics.depth_limited > 0,
            "2000 levels must reach the bound of {DEFAULT_MAX_DEPTH}"
        );

        // The derived traversals recurse to the tree's depth in the caller, so
        // the bound has to hold for them too.
        let copy = recovered.clone();
        assert_eq!(copy, recovered);
        drop(copy);
        drop(recovered);
    }

    /// An early-return spine iterates rather than nests, in either edge order,
    /// so it never reaches the depth bound.
    #[test]
    fn a_chain_of_conditionals_does_not_recurse_per_conditional() {
        // The early-return spine: `if (x) return; if (y) return; ..`, in both
        // edge orders, because which arm the front end puts first is not the
        // recovery's choice. Neither order may nest.
        let tests: usize = 10_000;
        let last = tests.saturating_mul(2);
        for exit_first in [false, true] {
            let mut edges: Vec<(usize, usize)> = Vec::new();
            for index in 0..tests {
                let sink = tests.saturating_add(index);
                let next = if index.saturating_add(1) < tests {
                    index.saturating_add(1)
                } else {
                    last
                };
                if exit_first {
                    edges.push((index, sink));
                    edges.push((index, next));
                } else {
                    edges.push((index, next));
                    edges.push((index, sink));
                }
            }
            let graph = graph_of(last.saturating_add(1), &edges);
            let entry = NodeId::new(0);
            let recovered = structure(
                &graph,
                entry,
                &StructureOptions::bare(graph.node_count(), &[]),
            );

            assert_total(&graph, entry, &recovered);
            assert_eq!(
                recovered.metrics.depth_limited, 0,
                "an early-return spine must not nest (exit arm first: {exit_first})"
            );
            assert_eq!(recovered.metrics.ifs, tests);
        }
    }

    /// Condition folding stops at the fold bound, capping the tests one merged
    /// predicate may carry.
    #[test]
    fn a_long_merged_condition_is_bounded() {
        // `if (a0 && a1 && ..)` a thousand terms long. Every fold nests the
        // predicate one level when it alternates, so the fold count is bounded
        // by the same number as the tree's depth.
        let tests: usize = 1000;
        let success = tests;
        let failure = tests.saturating_add(1);
        let end = tests.saturating_add(2);
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for index in 0..tests {
            let next = if index.saturating_add(1) < tests {
                index.saturating_add(1)
            } else {
                success
            };
            edges.push((index, next));
            edges.push((index, failure));
        }
        edges.push((success, end));
        edges.push((failure, end));

        let graph = graph_of(end.saturating_add(1), &edges);
        let entry = NodeId::new(0);
        let recovered = structure_bounded(&graph, entry, &[], 8);

        assert_bounded_total(&graph, entry, &recovered, 8);
        assert!(
            recovered.metrics.depth_limited > 0,
            "folding must stop at the bound"
        );
        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert!(
            branch.predicate.tests() <= 9,
            "at most eight folds beyond the head test, got {}",
            branch.predicate.tests()
        );
    }

    /// A `max_depth` of zero is clamped to one, so the outermost walk is never
    /// the limited one and the orphan pass still makes progress.
    #[test]
    fn max_depth_of_zero_still_terminates() {
        // Clamped to one, so the outermost walk is never the limited one and
        // the orphan pass always makes progress.
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let entry = NodeId::new(0);
        let recovered = structure_bounded(&graph, entry, &[], 0);

        assert_bounded_total(&graph, entry, &recovered, 1);
        assert!(recovered.metrics.depth_limited > 0);
    }

    /// A confined walk abandoned at the depth bound flattens its region's
    /// blocks in place rather than letting a goto place them outside the
    /// `Region::Try`.
    #[test]
    fn a_depth_limited_protected_body_keeps_its_blocks_inside_the_try() {
        // A goto out of a confined walk would place a block the region owns at
        // top level, outside the `Region::Try` - protected code printed as
        // unprotected. The flat emission keeps it in.
        let graph = graph_of(7, &[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(7, &[1, 2, 3, 4]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Finally,
                entry: NodeId::new(6),
                blocks: set_of(7, &[6]),
            }],
        }];
        let entry = NodeId::new(0);
        let recovered = structure_bounded(&graph, entry, &regions, 2);

        assert!(recovered.metrics.depth_limited > 0);
        let protected = first_try(&recovered.root).expect("a try region");
        let inside = placed_in(&protected.body);
        for block in [1, 2, 3, 4] {
            assert!(
                inside.contains(&NodeId::new(block)),
                "block {block} belongs to the region and must stay inside it: {inside:?}"
            );
        }
    }

    // Protected regions

    /// Two clauses sharing a try entry both open, with the region covering more
    /// blocks containing the smaller one.
    #[test]
    fn two_regions_sharing_an_entry_both_open() {
        // `try { try { .. } catch { .. } } finally { .. }` is the ordinary
        // encoding: two clauses, one try start. Declared innermost first, as
        // an exception table records them.
        let graph = graph_of(6, &[(0, 1), (1, 2), (2, 5), (3, 5), (4, 5)]);
        let regions = vec![
            ProtectedRegion {
                entry: NodeId::new(1),
                blocks: set_of(6, &[1]),
                handlers: vec![ProtectedHandler {
                    kind: ProtectedHandlerKind::Catch,
                    entry: NodeId::new(3),
                    blocks: set_of(6, &[3]),
                }],
            },
            ProtectedRegion {
                entry: NodeId::new(1),
                blocks: set_of(6, &[1, 2]),
                handlers: vec![ProtectedHandler {
                    kind: ProtectedHandlerKind::Finally,
                    entry: NodeId::new(4),
                    blocks: set_of(6, &[4]),
                }],
            },
        ];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 2, "both clauses open");

        let outer = first_try(&recovered.root).expect("the outer try");
        assert_eq!(
            outer.handlers.first().map(|handler| handler.kind),
            Some(HandlerKind::Finally),
            "the region covering more blocks is the containing one"
        );
        let inner = first_try(&outer.body).expect("the inner try inside the outer body");
        assert_eq!(
            inner.handlers.first().map(|handler| handler.kind),
            Some(HandlerKind::Catch),
            "and the smaller region opens inside it"
        );
    }

    /// A protected region whose entry is never reached still emits its handler
    /// inside a `Region::Try` rather than as unprotected code.
    #[test]
    fn an_unreached_protected_region_still_opens_its_try() {
        // Nothing reaches block 1, so the walk never enters the region. Its
        // handler must still be emitted inside a `Region::Try`: a bare
        // labelled block at top level prints protected code as unprotected.
        let graph = graph_of(5, &[(0, 4), (1, 2)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(5, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(5, &[3]),
            }],
        }];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 1);

        let protected = first_try(&recovered.root).expect("a try region");
        assert_eq!(protected.entry, NodeId::new(1));
        assert_eq!(
            protected.handlers.first().map(|handler| &handler.body),
            Some(&Region::Block(NodeId::new(3))),
            "the handler is inside the try, not beside it"
        );
    }

    // One scope rule

    /// A block folded into a condition has no statement position, so it is
    /// never recorded as a goto target.
    #[test]
    fn a_folded_join_is_not_a_goto_target() {
        // `for (;;) { A; if (c) continue; if (d) continue; break; }`. Block 3
        // is the join of the branch at 2 and is also folded into its
        // condition, so it has no statement position left to resume at.
        let graph = graph_of(5, &[(0, 1), (1, 2), (2, 1), (2, 3), (3, 1), (3, 4)]);
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions::bare(graph.node_count(), &[]),
        );

        assert_total(&graph, entry, &recovered);
        assert!(
            !recovered.labels.contains(&NodeId::new(3)),
            "a folded block has no label position: {:?}",
            recovered.labels
        );
        assert_eq!(recovered.metrics.merged_conditions, 1);
    }

    /// A header that is the whole loop recovers as a `while` with an empty body
    /// rather than placing the header twice.
    #[test]
    fn a_self_looping_while_has_an_empty_body() {
        // The header is the whole loop: it is both the condition and the work,
        // and a `while` already runs its condition every iteration. Emitting
        // the header as the body too would place it twice.
        let graph = graph_of(3, &[(0, 1), (1, 1), (1, 2)]);
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions::bare(graph.node_count(), &[]),
        );

        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::While);
        assert_eq!(looped.body, Region::Empty);
    }

    /// The loop-frame barrier keeps an enclosing loop from bounding a handler's
    /// joins, so a diamond inside the handler recovers as an `if`/`else` rather
    /// than degrading to an over-wide arm plus a goto.
    #[test]
    fn a_handler_conditional_inside_a_loop_recovers_as_an_if() {
        // The handler holds a diamond rejoining at 7, which lies outside the
        // loop the protected region sits in. The runtime enters the handler,
        // not the loop's back edge, so the loop's body does not bound the
        // handler's joins - and without that barrier the diamond degrades to
        // an over-wide arm plus a goto.
        let graph = graph_of(
            8,
            &[
                (0, 1),
                (1, 2),
                (1, 6),
                (2, 1),
                (3, 4),
                (3, 5),
                (4, 7),
                (5, 7),
            ],
        );
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(2),
            blocks: set_of(8, &[2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(8, &[3, 4, 5, 7]),
            }],
        }];
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions::bare(graph.node_count(), &regions),
        );

        assert_total_protected(&graph, entry, &regions, &recovered);
        assert_eq!(recovered.metrics.loops, 1);
        assert_eq!(
            recovered.metrics.if_elses, 1,
            "the handler's diamond is an if/else"
        );
        assert!(
            !recovered.labels.contains(&NodeId::new(7)),
            "the diamond rejoins in tree order: {:?}",
            recovered.labels
        );
    }

    /// The barrier bounds a handler's joins without stopping it from naming the
    /// enclosing loop, so a jump from the handler back to the header is a
    /// `continue` and not a goto.
    #[test]
    fn a_handler_that_continues_the_enclosing_loop_still_continues() {
        // The barrier stops the enclosing loop from *bounding* the handler's
        // walk. It must not stop the handler from naming that loop: the try is
        // emitted lexically inside the loop body, so a jump from the handler
        // back to the header is a `continue` and not a goto.
        let graph = graph_of(6, &[(0, 1), (1, 2), (1, 5), (2, 4), (4, 1), (3, 1)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(2),
            blocks: set_of(6, &[2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(6, &[3]),
            }],
        }];
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions::bare(graph.node_count(), &regions),
        );

        assert_total_protected(&graph, entry, &regions, &recovered);
        let protected = first_try(&recovered.root).expect("a try region");
        assert_eq!(
            protected.handlers.first().map(|handler| &handler.body),
            Some(&Region::Seq(vec![
                Region::Block(NodeId::new(3)),
                Region::Continue(NodeId::new(1)),
            ])),
            "a handler still continues the loop it sits in"
        );
    }

    // No construction site drops a block

    /// A condition's statements give every block it spans a statement
    /// position, in evaluation order.
    #[test]
    fn a_conditions_statements_place_every_test_block() {
        let predicate = Predicate::All(vec![
            Predicate::Test {
                block: NodeId::new(0),
                negated: false,
            },
            Predicate::Any(vec![
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: true,
                },
                Predicate::Test {
                    block: NodeId::new(2),
                    negated: false,
                },
            ]),
        ]);

        let region = predicate.statements();
        let mut placed = Vec::new();
        region.for_each_block(&mut |node| placed.push(node));
        assert_eq!(
            placed,
            vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
            "every block the condition spans keeps a statement position"
        );
    }

    /// Trimming a trailing `if (c) continue;` drops the transfer but keeps the
    /// block the condition tests in.
    #[test]
    fn trim_tail_keeps_the_predicate_blocks() {
        // `if (c) continue;` at a loop body's tail says nothing about control
        // flow, but the block the condition tests in is placed by this `if`
        // and nowhere else.
        let header = NodeId::new(0);
        let region = Region::If(Box::new(IfRegion {
            predicate: Predicate::Test {
                block: NodeId::new(1),
                negated: false,
            },
            then_branch: Region::Continue(header),
            else_branch: None,
        }));

        let mut end = LoopEnd::new(header);
        let trimmed = end.trimmed(region);
        assert_eq!(end.removed, 1);
        assert_eq!(trimmed, Region::Block(NodeId::new(1)));
    }

    // The pre-existing recovery

    /// A straight-line graph recovers as a plain sequence with no gotos or
    /// conditionals.
    #[test]
    fn straight_line_is_a_sequence() {
        let graph = graph_of(3, &[(0, 1), (1, 2)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(recovered.metrics.ifs, 0);
        assert_eq!(
            recovered.root,
            Region::Seq(vec![
                Region::Block(NodeId::new(0)),
                Region::Block(NodeId::new(1)),
                Region::Block(NodeId::new(2)),
            ])
        );
    }

    /// A diamond recovers as an `if`/`else` that owns its branch block, with
    /// the join placed after it.
    #[test]
    fn diamond_is_an_if_else() {
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(recovered.metrics.if_elses, 1);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        // The conditional owns the block it branches from, so it is the first
        // element and the join follows it.
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if first, got {top:?}");
        };
        assert_eq!(branch.then_branch, Region::Block(NodeId::new(1)));
        assert_eq!(branch.else_branch, Some(Region::Block(NodeId::new(2))));
        assert_eq!(top.get(1), Some(&Region::Block(NodeId::new(3))));
    }

    /// A branch whose first edge goes straight to the join inverts its
    /// condition rather than emitting an empty `then` arm.
    #[test]
    fn empty_then_arm_inverts_rather_than_printing_an_empty_block() {
        // 0 branches to 2 directly on its first edge; only the second edge has
        // a body. The `then` arm must become that body, with then_index naming
        // the second successor so the printer negates.
        let graph = graph_of(3, &[(0, 2), (0, 1), (1, 2)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Test {
                block: NodeId::new(0),
                negated: true
            },
            "condition must read as negated"
        );
        assert_eq!(branch.then_branch, Region::Block(NodeId::new(1)));
        assert_eq!(branch.else_branch, None);
    }

    /// A loop tested at its header recovers as a `while` with the header's test
    /// as its condition and the exit as its follow.
    #[test]
    fn pre_tested_loop_is_a_while() {
        // 0 -> 1 header; 1 -> 2 body -> 1; 1 -> 3 exit.
        let graph = graph_of(4, &[(0, 1), (1, 2), (1, 3), (2, 1)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(recovered.metrics.loops, 1);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::While);
        assert_eq!(
            looped.predicate,
            Some(Predicate::Test {
                block: NodeId::new(1),
                negated: false
            })
        );
        assert_eq!(looped.follow, Some(NodeId::new(3)));
        assert_eq!(looped.body, Region::Block(NodeId::new(2)));
    }

    /// A header that does work the condition depends on is not a `while`
    /// header, because `while` runs the header once and the condition always.
    #[test]
    fn a_header_that_computes_more_than_its_condition_is_not_a_while() {
        // 0 -> 1 header; 1 -> 2 body -> 1; 1 -> 3 exit. The header loads the
        // value it tests, so hoisting it above the loop would load once and
        // then spin.
        let graph = graph_of(4, &[(0, 1), (1, 2), (1, 3), (2, 1)]);
        let entry = NodeId::new(0);

        // Cleared: the header is exactly a test, so `while` is exact.
        let cleared = structure_merging(&graph, entry, &[0, 1, 2, 3]);
        let Region::Seq(top) = &cleared.root else {
            panic!("expected a sequence, got {:?}", cleared.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::While);

        // Not cleared: the loop keeps its header inside the body and leaves by
        // `break`, which runs the header every iteration.
        let uncleared = structure_merging(&graph, entry, &[]);
        assert_total(&graph, entry, &uncleared);
        let Region::Seq(top) = &uncleared.root else {
            panic!("expected a sequence, got {:?}", uncleared.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(
            looped.kind,
            StructuredLoop::Endless,
            "an unclear header must stay inside the loop"
        );
        assert_eq!(uncleared.metrics.breaks, 1, "and leave by break");

        let mut placed = Vec::new();
        looped.body.for_each_block(&mut |node| placed.push(node));
        assert!(
            placed.contains(&NodeId::new(1)),
            "the header runs each iteration, so it is part of the body: {placed:?}"
        );
    }

    /// A block that is its own latch is a `do`/`while` over that block, not a
    /// `while` with an empty body.
    ///
    /// The work and the test are the same block, so a `while` has nowhere to put
    /// the work but inside its own condition, and an endless loop spells the
    /// same test as `if (!c) break;` at its end. The body has to be the body,
    /// and the test belongs underneath it.
    #[test]
    fn a_self_loop_is_a_do_while_over_its_block() {
        // 0 -> 1; 1 branches back to itself or out to 2. Block 1 computes *and*
        // tests, which is what `condition_expressible` admits and what every
        // real self-loop looks like - a header that only tests is a `while` and
        // is covered by `pre_tested_loop_is_a_while`.
        let graph = graph_of(3, &[(0, 1), (1, 1), (1, 2)]);
        let recovered = structure_expressing(&graph, NodeId::new(0), &[1]);
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(recovered.metrics.loops, 1);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(
            looped.kind,
            StructuredLoop::DoWhile,
            "a self-loop tests after its work"
        );
        assert_eq!(
            looped.predicate,
            Some(Predicate::Test {
                block: NodeId::new(1),
                negated: false
            }),
            "the block's own branch is the loop's test"
        );
        assert_eq!(
            looped.body,
            Region::Block(NodeId::new(1)),
            "the block does the work every iteration, so it is the body"
        );
        assert_eq!(recovered.metrics.breaks, 0, "the test needs no `break`");
        assert_eq!(recovered.metrics.ifs, 0, "nor a conditional");
    }

    /// A loop tested at its latch recovers as a `do`/`while`, and the latch
    /// test does not also appear as a conditional inside the body.
    #[test]
    fn post_tested_loop_is_a_do_while() {
        // 0 -> 1 header -> 2 latch; latch branches back to 1 or out to 3. The
        // latch terminator is the condition, so it must not also appear as a
        // conditional inside the body.
        let graph = graph_of(4, &[(0, 1), (1, 2), (2, 1), (2, 3)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        assert_eq!(recovered.metrics.ifs, 0, "the latch test is the loop test");

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::DoWhile);
        assert_eq!(
            looped.predicate,
            Some(Predicate::Test {
                block: NodeId::new(2),
                negated: false
            })
        );
        assert_eq!(
            looped.body,
            Region::Seq(vec![
                Region::Block(NodeId::new(1)),
                Region::Block(NodeId::new(2)),
            ])
        );
    }

    /// **A loop tested at both ends takes the form its header can say.** The
    /// header 1 computes as well as tests, and the latch 3 tests too. When the
    /// header's statements are only the operand of its test, the `while`
    /// condition says them as C writes them, `while ((c = f()) != 0)`; any
    /// other statements a condition could carry only as comma operands, so
    /// the loop is the `do`/`while` that runs them at the top of its body,
    /// left by one `break` there. A header that computes nothing but its
    /// test stays a `while` too.
    #[test]
    fn a_loop_tested_at_both_ends_takes_the_form_its_header_can_say() {
        // 0 -> 1 header; 1 -> 2 body or 4 out; 2 -> 3 latch; 3 -> 1 or 4.
        let graph = graph_of(5, &[(0, 1), (1, 2), (1, 4), (2, 3), (3, 1), (3, 4)]);
        let shape = |condition_only: &[usize], condition_operand: &[usize]| {
            let options = StructureOptions {
                statements: Vec::new(),
                regions: &[],
                condition_only: set_of(5, condition_only),
                condition_reachability: BlockSet::new(5),
                condition_expressible: set_of(5, &[1]),
                condition_operand: set_of(5, condition_operand),
                case_tests: BTreeMap::new(),
                no_return: BlockSet::new(5),
                edge_blocks: BlockSet::new(5),
                max_depth: DEFAULT_MAX_DEPTH,
                polar: BlockSet::new(5),
                repeated_tests: BTreeMap::new(),
                threaded_tests: BTreeMap::new(),
            };
            let recovered = structure(&graph, NodeId::new(0), &options);
            assert_total(&graph, NodeId::new(0), &recovered);
            assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
            assert_eq!(recovered.metrics.breaks, 1, "{:#?}", recovered.root);
            let Region::Seq(top) = &recovered.root else {
                panic!("expected a sequence, got {:?}", recovered.root);
            };
            let Some(Region::Loop(looped)) = top.get(1) else {
                panic!("expected a loop, got {top:?}");
            };
            looped.kind
        };
        assert_eq!(shape(&[], &[]), StructuredLoop::DoWhile);
        assert_eq!(shape(&[], &[1]), StructuredLoop::While);
        assert_eq!(shape(&[1], &[]), StructuredLoop::While);
    }

    /// A loop tested at neither header nor latch recovers as endless with an
    /// explicit `break`.
    #[test]
    fn loop_tested_in_the_middle_is_endless_with_a_break() {
        // 0 -> 1 -> 2; 2 branches to 3 (exit) or 4; 4 -> 1. Neither the header
        // nor the latch tests, so this is `for (;;) { ..; if (c) break; .. }`.
        let graph = graph_of(5, &[(0, 1), (1, 2), (2, 3), (2, 4), (4, 1)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 1);
        assert_eq!(recovered.metrics.endless_loops, 1);
        assert_eq!(recovered.metrics.breaks, 1);
        assert_eq!(recovered.metrics.gotos, 0);
    }

    /// A loop with no exit recovers with no follow and emits neither `break`
    /// nor `continue`.
    #[test]
    fn infinite_loop_has_no_follow() {
        let graph = graph_of(3, &[(0, 1), (1, 2), (2, 1)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.endless_loops, 1);
        assert_eq!(recovered.metrics.breaks, 0);
        assert_eq!(
            recovered.metrics.continues, 0,
            "a loop that only falls off its own end needs no continue"
        );

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.follow, None);
    }

    /// BinKit contains no multi-latch loop, so this shape exists only here.
    #[test]
    fn multi_latch_loop_continues_from_every_latch() {
        // Header 1 with two latches, 3 and 4, and an exit at 5.
        let graph = graph_of(6, &[(0, 1), (1, 2), (1, 5), (2, 3), (2, 4), (3, 1), (4, 1)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 1, "two latches are still one loop");
        assert_eq!(recovered.metrics.gotos, 0, "two latches still structure");

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.kind, StructuredLoop::While);
        assert_eq!(looped.follow, Some(NodeId::new(5)));
        // Both latches reach the header by falling off the end of the body, so
        // neither needs to say `continue`.
        assert_eq!(
            looped.body,
            Region::If(Box::new(IfRegion {
                predicate: Predicate::Test {
                    block: NodeId::new(2),
                    negated: false,
                },
                then_branch: Region::Block(NodeId::new(3)),
                else_branch: Some(Region::Block(NodeId::new(4))),
            }))
        );
    }

    /// **A cycle entered at two blocks becomes a loop entered at one.** `0`
    /// branches into either side of the `1 <-> 2` cycle, which no block
    /// dominates, so it is no natural loop and its second entry took a goto.
    /// Copying `2` for the path that enters there gives that path a block of
    /// its own that joins the loop at `1`. The control puts the cycle inside a
    /// protected region, where a copy outside the region would run
    /// unprotected, and the goto stays.
    #[test]
    fn a_cycle_entered_twice_is_split_into_a_loop() {
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 2), (2, 1), (1, 3)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert_eq!(recovered.metrics.split_blocks, 1);
        assert_eq!(recovered.metrics.loops, 1);
        assert!(recovered.labels.is_empty(), "{:?}", recovered.labels);

        let graph = graph_of(5, &[(0, 1), (0, 2), (1, 2), (2, 1), (1, 3)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(0),
            blocks: set_of(5, &[0, 1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(4),
                blocks: set_of(5, &[4]),
            }],
        }];
        let control = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &control);
        assert_eq!(control.metrics.split_blocks, 0);
        assert!(control.metrics.gotos >= 1, "{:#?}", control.root);
    }

    /// **A split copying more than its goto is worth is not kept.** The
    /// cycle of [`a_cycle_entered_twice_is_split_into_a_loop`], its second
    /// entry `2` copied to give the loop one: where `2` prints more statements
    /// than [`STATEMENTS_PER_GOTO`](inputs::STATEMENTS_PER_GOTO), the copy
    /// costs more than the goto it saves, and the entry stays a goto.
    #[test]
    fn a_split_copying_more_than_its_goto_is_worth_is_not_kept() {
        let graph = graph_of(4, &[(0, 1), (0, 2), (1, 2), (2, 1), (1, 3)]);
        let weighed = |copied: usize| {
            structure(
                &graph,
                NodeId::new(0),
                &StructureOptions {
                    statements: vec![1, 1, copied, 1],
                    ..StructureOptions::bare(graph.node_count(), &[])
                },
            )
        };
        let short = weighed(inputs::STATEMENTS_PER_GOTO);
        assert_total(&graph, NodeId::new(0), &short);
        assert_eq!(short.metrics.split_blocks, 1, "{:#?}", short.root);
        assert_eq!(short.metrics.gotos, 0, "{:#?}", short.root);
        let long = weighed(inputs::STATEMENTS_PER_GOTO.saturating_add(1));
        assert_total(&graph, NodeId::new(0), &long);
        assert_eq!(long.metrics.split_blocks, 0, "{:#?}", long.root);
        assert_eq!(long.metrics.gotos, 1, "{:#?}", long.root);
    }

    /// **Splitting never costs a goto, on any graph.** Several thousand small
    /// graphs drawn from a fixed sequence, each recovered with its irreducible
    /// loops split on trial and without: every block is placed, every repeat
    /// is accounted for, and the split recovery never needs more gotos, as C
    /// writes them, than the plain one.
    #[test]
    fn splitting_never_costs_a_goto() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let mut split = 0usize;
        for _ in 0..4000 {
            let count = usize::try_from(next(7)).unwrap_or(0).saturating_add(3);
            let mut edges = Vec::new();
            for from in 0..count {
                let degree = next(3);
                for _ in 0..degree {
                    let to = usize::try_from(next(u64::try_from(count).unwrap_or(1))).unwrap_or(0);
                    edges.push((from, to));
                }
            }
            let graph = graph_of(count, &edges);
            let entry = NodeId::new(0);
            let recovered = structure(
                &graph,
                entry,
                &StructureOptions::bare(graph.node_count(), &[]),
            );
            assert_total(&graph, entry, &recovered);
            let inputs = Inputs {
                statements: Statements::default(),
                condition_only: BlockSet::full(count),
                condition_reachability: BlockSet::new(count),
                condition_expressible: BlockSet::new(count),
                condition_operand: BlockSet::new(count),
                no_return: BlockSet::new(count),
                edge_blocks: BlockSet::new(count),
                regions: Vec::new(),
                case_tests: BTreeMap::new(),
                max_depth: DEFAULT_MAX_DEPTH,
                preferred_joins: BlockSet::new(count),
                fold_statements: true,
                fall_out: false,
                switch_tails: false,
                route: false,
                repeated_tests: BTreeMap::new(),
                threaded_tests: BTreeMap::new(),
            };
            let plain = recover(&graph, entry, &inputs).structured;
            assert!(
                recovered.root.gotos() <= plain.root.gotos(),
                "{edges:?}: split {} > plain {}",
                recovered.root.gotos(),
                plain.root.gotos()
            );
            split = split.saturating_add(usize::from(recovered.metrics.split_blocks > 0));
        }
        assert!(
            split > 0,
            "no graph was split, so nothing above was measured"
        );
    }

    /// The convergence of the paths over `nodes` toward `goals`, built over
    /// tables as wide as the whole graph: the definition [`Convergence`] is
    /// built over only the blocks it covers to agree with.
    fn whole_graph_convergence<G: Successors>(
        graph: &G,
        node_count: usize,
        nodes: impl Iterator<Item = NodeId>,
        step: impl Fn(NodeId) -> Option<NodeId>,
        goals: impl Iterator<Item = NodeId>,
    ) -> (BitSet, PostDominatorTree) {
        let size = node_count + 1;
        let mut edges: Vec<Vec<NodeId>> = vec![Vec::new(); size];
        let mut predecessors: Vec<Vec<NodeId>> = vec![Vec::new(); size];
        for node in nodes.filter(|node| node.index() < node_count) {
            for target in graph.successors(node).filter_map(&step) {
                if target.index() < size {
                    edges[node.index()].push(target);
                    predecessors[target.index()].push(node);
                }
            }
        }
        let mut reaches = BitSet::new(size);
        let mut walk: Vec<NodeId> = goals.collect();
        while let Some(node) = walk.pop() {
            if reaches.insert_checked(node.index()) {
                walk.extend(predecessors[node.index()].iter());
            }
        }
        for row in &mut edges {
            row.retain(|target| reaches.contains_checked(target.index()));
        }
        (reaches, compute_post_dominators(&ConvergingGraph { edges }))
    }

    /// **A convergence over the blocks it covers answers as one over the whole
    /// graph.** Several thousand small graphs drawn from a fixed sequence, each
    /// with a function convergence under a random set of blocks that do not
    /// return and a loop convergence for every natural loop: for every block
    /// of the graph and the nodes past it, `reaches` and `meet` equal what the
    /// post-dominator tree of the whole-graph tables says.
    #[test]
    fn a_convergence_over_the_blocks_it_covers_matches_the_whole_graph_one() {
        let mut state = 0x1357_9bdf_2468_ace0u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let (mut functions, mut loops, mut joins) = (0usize, 0usize, 0usize);
        for _ in 0..4000 {
            let count = usize::try_from(next(9)).unwrap_or(0).saturating_add(3);
            let bound = u64::try_from(count).unwrap_or(1);
            let mut edges = Vec::new();
            for from in 0..count {
                for _ in 0..next(4) {
                    edges.push((from, usize::try_from(next(bound)).unwrap_or(0)));
                }
            }
            let graph = graph_of(count, &edges);
            let repeat = NodeId::new(count);
            let agrees = |covered: &Convergence, reference: &(BitSet, PostDominatorTree)| {
                let (reaches, tree) = reference;
                (0..=count.saturating_add(2)).all(|index| {
                    let node = NodeId::new(index);
                    covered.reaches(node) == reaches.contains_checked(index)
                        && covered.meet(node)
                            == tree
                                .immediate_post_dominator(node)
                                .filter(|meet| *meet != repeat)
                })
            };

            let ends: Vec<usize> = (0..count)
                .filter(|node| {
                    graph.successors(NodeId::new(*node)).next().is_none() && next(2) == 0
                })
                .collect();
            let no_return = set_of(count, &ends);
            let covered = Convergence::of_function(&graph, count, &no_return);
            let reference = whole_graph_convergence(
                &graph,
                count,
                (0..count).map(NodeId::new),
                Some,
                (0..count).map(NodeId::new).filter(|node| {
                    graph.successors(*node).next().is_none() && !no_return.contains(*node)
                }),
            );
            assert!(agrees(&covered, &reference), "function over {edges:?}");
            functions = functions.saturating_add(1);

            let dominators = compute_dominators(&graph, NodeId::new(0));
            for info in detect_loops(&graph, &dominators).loops() {
                let header = info.header;
                let covered = Convergence::of_loop(&graph, count, header, &info.body);
                let reference = whole_graph_convergence(
                    &graph,
                    count,
                    info.body.iter().map(NodeId::new),
                    |successor| {
                        if successor == header {
                            Some(repeat)
                        } else {
                            info.body
                                .contains_checked(successor.index())
                                .then_some(successor)
                        }
                    },
                    [repeat].into_iter(),
                );
                assert!(
                    agrees(&covered, &reference),
                    "loop {header:?} over {edges:?}"
                );
                loops = loops.saturating_add(1);
                joins = joins.saturating_add(
                    (0..count)
                        .filter(|index| covered.meet(NodeId::new(*index)).is_some())
                        .count(),
                );
            }
        }
        assert!(
            functions > 0 && loops > 0 && joins > 0,
            "no loop convergence had a join, so nothing above was measured"
        );
    }

    /// **Recovery is total on any inputs.** Several thousand small graphs
    /// drawn from a fixed sequence, each with every input a caller can give
    /// drawn too: blocks that only test, blocks that can be said in a
    /// condition, blocks that do not return, and value tests over a few values
    /// and constants with every relation, on every block with two successors.
    /// Each recovery ends, places every reachable block, and accounts for every
    /// repeat - whatever shape the tests make, a tree, a cycle or neither.
    /// Recovered without the value tests, which decide branches a run cannot
    /// follow, each one without a `goto`, a `switch` over a value or a
    /// protected region runs the blocks its graph does ([`assert_runs_as_the_graph`]), and one
    /// that kept no block out of a condition for
    /// [`Inputs::fold_statements`] alone is the one with it on.
    #[test]
    fn recovery_is_total_on_any_inputs() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let relations = [
            CaseRelation::Equal,
            CaseRelation::NotEqual,
            CaseRelation::Less { signed: true },
            CaseRelation::LessOrEqual { signed: true },
            CaseRelation::Greater { signed: true },
            CaseRelation::GreaterOrEqual { signed: true },
            CaseRelation::Less { signed: false },
            CaseRelation::Greater { signed: false },
        ];
        let mut switches = 0usize;
        let mut run = 0usize;
        let mut copies = 0usize;
        let mut declined = 0usize;
        let mut idle = 0usize;
        let mut fall_throughs = 0usize;
        for _ in 0..4000 {
            let count = usize::try_from(next(10)).unwrap_or(0).saturating_add(6);
            let bound = u64::try_from(count).unwrap_or(1);
            // Half the graphs open with a chain of equality tests over one
            // value, `1..=chain`, each matching into an arm drawn from every
            // block and failing into the next; arms and the default may lead
            // back into the chain.
            let chain = if next(2) == 0 {
                usize::try_from(next(3)).unwrap_or(0).saturating_add(3)
            } else {
                0
            };
            let mut edges = Vec::new();
            let mut case_tests = BTreeMap::new();
            for from in 0..count {
                if (1..=chain).contains(&from) {
                    let arm = usize::try_from(next(bound)).unwrap_or(0);
                    let fails = if from == chain {
                        usize::try_from(next(bound)).unwrap_or(0)
                    } else {
                        from.saturating_add(1)
                    };
                    edges.push((from, arm));
                    edges.push((from, fails));
                    case_tests.insert(
                        NodeId::new(from),
                        CaseTest {
                            value: 0,
                            relation: CaseRelation::Equal,
                            constant: i64::try_from(from).unwrap_or(0),
                            holds: 0,
                            entered: true,
                        },
                    );
                    continue;
                }
                for _ in 0..next(4) {
                    edges.push((from, usize::try_from(next(bound)).unwrap_or(0)));
                }
            }
            let graph = graph_of(count, &edges);
            let members = |next: &mut dyn FnMut(u64) -> u64| {
                (0..count)
                    .filter(|node| (2..=chain).contains(node) || next(3) != 0)
                    .collect::<Vec<usize>>()
            };
            let condition_only = members(&mut next);
            let condition_expressible = members(&mut next);
            for node in chain.saturating_add(1)..count {
                let degree = edges.iter().filter(|(from, _)| *from == node).count();
                if degree == 2 && next(4) != 0 {
                    let relation = relations[usize::try_from(next(8)).unwrap_or(0)];
                    let constant = i64::try_from(next(6)).unwrap_or(0).saturating_sub(2);
                    case_tests.insert(
                        NodeId::new(node),
                        CaseTest {
                            value: next(2),
                            relation,
                            constant,
                            holds: usize::try_from(next(2)).unwrap_or(0),
                            entered: true,
                        },
                    );
                }
            }
            let ends: Vec<usize> = (0..count)
                .filter(|node| edges.iter().all(|(from, _)| from != node) && next(2) == 0)
                .collect();
            let options = StructureOptions {
                statements: Vec::new(),
                regions: &[],
                condition_only: set_of(count, &condition_only),
                condition_reachability: BlockSet::new(count),
                condition_expressible: set_of(count, &condition_expressible),
                condition_operand: BlockSet::new(count),
                case_tests,
                no_return: set_of(count, &ends),
                edge_blocks: BlockSet::new(count),
                max_depth: DEFAULT_MAX_DEPTH,
                polar: BlockSet::new(count),
                repeated_tests: BTreeMap::new(),
                threaded_tests: BTreeMap::new(),
            };
            let recovered = structure(&graph, NodeId::new(0), &options);
            assert_total(&graph, NodeId::new(0), &recovered);
            assert_eq!(
                outer_do_while_continues(&recovered.root),
                0,
                "{edges:?}, folding {condition_only:?}, saying {condition_expressible:?}: {:#?}",
                recovered.root
            );
            switches = switches.saturating_add(recovered.metrics.value_switches);
            // Where the recovery kept no block out of a condition for
            // `fold_statements` alone, the one with it on is the same.
            let inputs = Inputs {
                statements: Statements::default(),
                regions: Vec::new(),
                condition_only: options.condition_only.clone(),
                condition_reachability: options.condition_reachability.clone(),
                condition_expressible: options.condition_expressible.clone(),
                condition_operand: options.condition_operand.clone(),
                case_tests: options.case_tests.clone(),
                no_return: options.no_return.clone(),
                edge_blocks: options.edge_blocks.clone(),
                max_depth: DEFAULT_MAX_DEPTH,
                preferred_joins: BlockSet::new(count),
                fold_statements: false,
                fall_out: false,
                switch_tails: false,
                route: false,
                repeated_tests: BTreeMap::new(),
                threaded_tests: BTreeMap::new(),
            };
            let (plain, _, _) = inputs.clone().recovered(&graph, NodeId::new(0), true);
            if plain.declined_fold {
                declined = declined.saturating_add(1);
            } else {
                let (folding, _, _) = Inputs {
                    fold_statements: true,
                    ..inputs
                }
                .recovered(&graph, NodeId::new(0), true);
                assert_eq!(folding.structured.root, plain.structured.root, "{edges:?}");
                idle = idle.saturating_add(1);
            }
            // A value test decides others over the same value, which a run
            // choosing each branch freely cannot follow: the run is of the
            // recovery that knows of none.
            let free = structure(
                &graph,
                NodeId::new(0),
                &StructureOptions {
                    case_tests: BTreeMap::new(),
                    ..options.clone()
                },
            );
            if modelled(&free.root) {
                assert_runs_as_the_graph(
                    &graph,
                    NodeId::new(0),
                    &options.condition_only,
                    &BlockSet::new(graph.node_count()),
                    &free,
                );
                run = run.saturating_add(1);
                copies = copies.saturating_add(latch_copies(&free.root));
                fall_throughs = fall_throughs.saturating_add(free.metrics.fall_throughs);
            }
        }
        assert!(
            switches > 0,
            "no value switch was recovered, so the case tests were measured by nothing"
        );
        assert!(run > 1000, "only {run} trees were run against their graphs");
        assert!(
            copies > 0,
            "no path into a latch placed its statements, so none was run"
        );
        assert!(
            fall_throughs > 0,
            "no switch arm fell through, so none was run"
        );
        assert!(
            declined > 0 && idle > 0,
            "{declined} recoveries kept a block out for the fold setting, {idle} none"
        );
    }

    /// When no exit post-dominates the header, taking the header's own exit as
    /// the follow keeps the loop a `while` rather than an endless body opened
    /// by `if (!c) break`.
    #[test]
    fn a_header_that_tests_names_the_follow_when_no_exit_catches_every_path() {
        // Header 1 leaves to 4, body block 2 leaves to 5, and the two never
        // meet - so no exit post-dominates the header and the counting rule
        // would pick 5 for having the lower id. Taking 1's own exit instead is
        // what makes the loop a `while` on 1's test rather than an endless body
        // that opens with `if (!c) break`.
        let graph = graph_of(6, &[(0, 1), (1, 2), (1, 4), (2, 3), (2, 5), (3, 1)]);
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.loops, 1);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        assert_eq!(looped.follow, Some(NodeId::new(4)));
        assert_eq!(looped.kind, StructuredLoop::While);
        assert_eq!(recovered.metrics.endless_loops, 0);
    }

    /// A loop with several exits picks one follow by post-dominance,
    /// independent of the order the edges were added.
    #[test]
    fn loop_leaving_to_several_blocks_keeps_one_follow() {
        // Header 1 leaves to 4; body block 2 leaves to 5. Only one of them can
        // be the block control resumes at, and the choice must not depend on
        // the order the edges were added.
        let forward = graph_of(6, &[(0, 1), (1, 2), (1, 4), (2, 3), (2, 5), (3, 1), (4, 5)]);
        let reversed = graph_of(6, &[(4, 5), (3, 1), (2, 5), (2, 3), (1, 4), (1, 2), (0, 1)]);

        let recovered = structure(
            &forward,
            NodeId::new(0),
            &StructureOptions::bare(forward.node_count(), &[]),
        );
        assert_total(&forward, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 1);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Loop(looped)) = top.get(1) else {
            panic!("expected a loop, got {top:?}");
        };
        // 5 post-dominates the header: every way out of the loop reaches it,
        // so it is the follow and the exit through 4 stays inside the region.
        // Each exit breaks where it leaves - 4 after its own block, 2 at once -
        // since the paths that stay meet at 2 and then at 3.
        assert_eq!(looped.follow, Some(NodeId::new(5)));
        assert_eq!(recovered.metrics.breaks, 2, "{:#?}", recovered.root);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);

        let other = structure(
            &reversed,
            NodeId::new(0),
            &StructureOptions::bare(reversed.node_count(), &[]),
        );
        assert_eq!(
            other.metrics, recovered.metrics,
            "follow selection must not depend on edge order"
        );
    }

    /// **A loop leaving through a block that stands for its exit edge leaves
    /// to where that block goes.** The loop of 1 and 5 leaves from 1 to 2,
    /// which goes on to 3, and from 5 to 6, which holds only what its edge
    /// writes and goes on to 3 too; 3 post-dominates the header. Read as the
    /// edge it stands for, 6 leaves to 3, which is then the block every way
    /// out reaches and the follow: both exits break, and no goto is left
    /// (reduced from dash's `evalcommand`, where the exit edge's copies made
    /// the follow a block nested in one arm and cost five gotos). Read as a
    /// block of its own, neither exit is 3, the follow is one of them, and the
    /// other jumps to 3.
    #[test]
    fn a_loop_leaving_through_an_edge_block_follows_where_it_goes() {
        let graph = graph_of(
            7,
            &[
                (0, 1),
                (1, 2),
                (1, 5),
                (2, 3),
                (3, 4),
                (4, 4),
                (5, 1),
                (5, 6),
                (6, 3),
            ],
        );
        let recover = |edge_blocks: &[usize]| {
            let options = StructureOptions {
                condition_only: BlockSet::new(7),
                condition_reachability: BlockSet::new(7),
                edge_blocks: set_of(7, edge_blocks),
                ..StructureOptions::bare(7, &[])
            };
            let recovered = structure(&graph, NodeId::new(0), &options);
            assert_total(&graph, NodeId::new(0), &recovered);
            recovered
        };
        let through = recover(&[6]);
        assert_eq!(through.metrics.gotos, 0, "{:#?}", through.root);
        let follow = |region: &Region| match region {
            Region::Seq(top) => top.iter().find_map(|region| match region {
                Region::Loop(looped) => Some(looped.follow),
                _ => None,
            }),
            _ => None,
        };
        assert_eq!(
            follow(&through.root),
            Some(Some(NodeId::new(3))),
            "{:#?}",
            through.root
        );
        assert_eq!(
            recover(&[]).metrics.gotos,
            1,
            "control: read as a block of its own, one exit jumps to 3"
        );
    }

    /// **A loop whose header does not leave it follows where every way out
    /// meets.** Reduced from dash's `expmeta`, its scan of a `[` set inside
    /// the scan of the name: the inner loop of 3, 4 and 6 is entered at 3,
    /// which tests nothing, and leaves from 4 through 5, a block standing for
    /// that edge, and from 6 through 10, the `]` arm that sets `metaflag`;
    /// both go on to 7, where the outer loop goes round. The header's first
    /// post-dominator, 4, is the loop's own; past it every way out meets at
    /// 7, which is the follow: both exits break, and no goto is left. Read as
    /// a block of its own, 5 is still entered from the inner loop alone and
    /// goes on to 7 alone, so it runs into 7 as the edge block does.
    #[test]
    fn a_loop_testing_past_its_header_follows_where_its_ways_out_meet() {
        let graph = graph_of(
            11,
            &[
                (0, 1),
                (1, 2),
                (1, 9),
                (2, 3),
                (2, 8),
                (3, 4),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 10),
                (6, 3),
                (10, 7),
                (7, 1),
                (8, 7),
            ],
        );
        let recover = |edge_blocks: &[usize]| {
            let options = StructureOptions {
                condition_only: BlockSet::new(11),
                condition_reachability: BlockSet::new(11),
                edge_blocks: set_of(11, edge_blocks),
                ..StructureOptions::bare(11, &[])
            };
            let recovered = structure(&graph, NodeId::new(0), &options);
            assert_total(&graph, NodeId::new(0), &recovered);
            recovered
        };
        let through = recover(&[5]);
        assert_eq!(through.metrics.gotos, 0, "{:#?}", through.root);
        // The inner loop's follow, wherever the outer loop's body holds it.
        let mut follows = Vec::new();
        let mut pending = vec![&through.root];
        while let Some(region) = pending.pop() {
            match region {
                Region::Seq(regions) => pending.extend(regions.iter()),
                Region::Loop(looped) => {
                    if looped.header == NodeId::new(3) {
                        follows.push(looped.follow);
                    }
                    pending.push(&looped.body);
                }
                Region::If(inner) => {
                    pending.push(&inner.then_branch);
                    pending.extend(inner.else_branch.iter());
                }
                _ => {}
            }
        }
        assert_eq!(follows, [Some(NodeId::new(7))], "{:#?}", through.root);
        assert_eq!(recover(&[]).metrics.gotos, 0);
    }

    /// **A loop whose ways out meet only past the loop around it follows its
    /// own exit.** Reduced from `which`'s `tilde_find_suffix`: the inner scan
    /// of 3 and 8 goes round the outer loop through 4 when it runs out, or
    /// returns through 6, a block standing for its edge to 7, which the outer
    /// loop leaves to as well. Every way out meets at 7, past the outer loop:
    /// resumed there, the way round would jump to the outer header. The inner
    /// loop follows 4 instead, and no goto is left. The control: with 6 read
    /// as a block of its own, the `return` arm stays where it is.
    #[test]
    fn a_loop_whose_ways_out_meet_past_the_loop_around_it_follows_its_own_exit() {
        let graph = graph_of(
            9,
            &[
                (0, 1),
                (1, 2),
                (1, 7),
                (2, 3),
                (3, 8),
                (8, 4),
                (8, 5),
                (4, 1),
                (5, 6),
                (5, 3),
                (6, 7),
            ],
        );
        for edge_blocks in [&[6][..], &[][..]] {
            let options = StructureOptions {
                condition_only: BlockSet::new(9),
                condition_reachability: BlockSet::new(9),
                edge_blocks: set_of(9, edge_blocks),
                ..StructureOptions::bare(9, &[])
            };
            let recovered = structure(&graph, NodeId::new(0), &options);
            assert_total(&graph, NodeId::new(0), &recovered);
            assert_eq!(
                recovered.metrics.gotos, 0,
                "{edge_blocks:?}: {:#?}",
                recovered.root
            );
        }
    }

    /// **A loop's ways out that each run on into where they meet follow
    /// there.** Reduced from dash's `expmeta` on RISC-V: the inner scan of a
    /// `[` set, 3 to 6, leaves from its `/` and end-of-string tests, 4 and 5,
    /// to 7, which makes the moves both share and goes on to the outer test
    /// 1, and from its `]` test 6 to 8, which sets `metaflag` and goes on to
    /// 1 too. Neither is an edge block, but each is entered only from the
    /// loop and goes on to 1 alone: 1 is the follow, each exit breaks with
    /// its own code, and no goto is left. The control: with 7 going on to 9
    /// as well, it no longer runs into 1, and the `]` arm jumps to the outer
    /// test from inside the inner loop.
    #[test]
    fn a_loops_ways_out_running_on_into_where_they_meet_follow_there() {
        let recover = |branching: bool| {
            let mut edges = vec![
                (0, 1),
                (1, 2),
                (1, 9),
                (2, 3),
                (3, 4),
                (3, 6),
                (4, 7),
                (4, 5),
                (5, 7),
                (5, 6),
                (6, 3),
                (6, 8),
                (7, 1),
                (8, 1),
            ];
            if branching {
                edges.push((7, 9));
            }
            let graph = graph_of(10, &edges);
            let options = StructureOptions {
                condition_only: BlockSet::new(10),
                condition_reachability: BlockSet::new(10),
                ..StructureOptions::bare(10, &[])
            };
            let recovered = structure(&graph, NodeId::new(0), &options);
            assert_total(&graph, NodeId::new(0), &recovered);
            recovered
        };
        let joined = recover(false);
        assert_eq!(joined.metrics.gotos, 0, "{:#?}", joined.root);
        assert!(
            recover(true).metrics.gotos > 0,
            "control: 7 branching elsewhere does not run into 1"
        );
    }

    /// **A loop one of whose ways out goes elsewhere follows the exit it
    /// tests for.** Reduced from dash's `pmatch` on ARM, its class scan
    /// `while ((p = match(class, q)) == 0)`: the inner loop of 3 and 4 leaves
    /// from 3, when the class is found, to 5, whose code goes on to 7 and
    /// then 8, and from 4, past the last class, to 6, which goes straight to
    /// 8. Every way out meets at 8, but 6 does not run into it the way 5's
    /// code does: whichever block follows, 6 costs its goto. 3's test stays
    /// the loop's own, `while`, with 5 the follow and its code after the
    /// loop.
    #[test]
    fn a_loop_one_of_whose_ways_out_goes_elsewhere_follows_the_exit_it_tests_for() {
        let graph = graph_of(
            10,
            &[
                (0, 1),
                (1, 2),
                (1, 9),
                (2, 3),
                (3, 5),
                (3, 4),
                (4, 6),
                (4, 3),
                (5, 7),
                (7, 8),
                (6, 8),
                (8, 1),
            ],
        );
        let options = StructureOptions {
            condition_only: BlockSet::new(10),
            condition_reachability: BlockSet::new(10),
            ..StructureOptions::bare(10, &[])
        };
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        let mut follows = Vec::new();
        let mut pending = vec![&recovered.root];
        while let Some(region) = pending.pop() {
            match region {
                Region::Seq(regions) => pending.extend(regions.iter()),
                Region::Loop(looped) => {
                    if looped.header == NodeId::new(3) {
                        follows.push(looped.follow);
                    }
                    pending.push(&looped.body);
                }
                Region::If(inner) => {
                    pending.push(&inner.then_branch);
                    pending.extend(inner.else_branch.iter());
                }
                _ => {}
            }
        }
        assert_eq!(follows, [Some(NodeId::new(5))], "{:#?}", recovered.root);
    }

    /// **A loop two of whose ways out run into where they meet follows
    /// there.** The inner loop of 3, 4 and 11 leaves from 3 to 5 and from 4
    /// to 10, each of which goes on to 8 alone, and from 11 to 6, whose code
    /// runs on through 7 to 8. Following 8 makes both `break`s and leaves 6
    /// and 7 to be placed with the third, where following one exit would
    /// leave the other two to jump: no goto is left. The control: with 10
    /// going on to 7 instead, only 5 runs into 8, and the loop follows an
    /// exit of its own.
    #[test]
    fn a_loop_two_of_whose_ways_out_run_into_where_they_meet_follows_there() {
        let recover = |second: usize| {
            let graph = graph_of(
                12,
                &[
                    (0, 1),
                    (1, 2),
                    (1, 9),
                    (2, 3),
                    (3, 5),
                    (3, 4),
                    (4, 10),
                    (4, 11),
                    (11, 6),
                    (11, 3),
                    (5, 8),
                    (10, second),
                    (6, 7),
                    (7, 8),
                    (8, 1),
                ],
            );
            let options = StructureOptions {
                condition_only: BlockSet::new(12),
                condition_reachability: BlockSet::new(12),
                ..StructureOptions::bare(12, &[])
            };
            let recovered = structure(&graph, NodeId::new(0), &options);
            assert_total(&graph, NodeId::new(0), &recovered);
            let mut follows = Vec::new();
            let mut pending = vec![&recovered.root];
            while let Some(region) = pending.pop() {
                match region {
                    Region::Seq(regions) => pending.extend(regions.iter()),
                    Region::Loop(looped) => {
                        if looped.header == NodeId::new(3) {
                            follows.push(looped.follow);
                        }
                        pending.push(&looped.body);
                    }
                    Region::If(inner) => {
                        pending.push(&inner.then_branch);
                        pending.extend(inner.else_branch.iter());
                    }
                    _ => {}
                }
            }
            (follows, recovered.metrics.gotos)
        };
        assert_eq!(recover(8), (vec![Some(NodeId::new(8))], 0));
        let (follows, _) = recover(7);
        assert_ne!(
            follows,
            [Some(NodeId::new(8))],
            "control: one exit runs into 8"
        );
    }

    /// **A test reaching its condition's join through an edge block folds
    /// with the edge's statements ahead of it.** Reduced from dash's `argstr`:
    /// block 5 goes to 6, a block standing for that edge alone, which goes on
    /// to 9, or to 7, a test of statements that goes to 9 itself or to 8,
    /// another edge's block. With what 6 writes said ahead of 5's test
    /// ([`Predicate::Hoisted`]), 5 and 7 are one condition leading to 9, and
    /// no goto is left; the run's blocks are the graph's, the edges'
    /// statements being the caller's to say wherever they are said. With no
    /// edge declared, 9 is reached from 6 and from 7, and each way takes its
    /// own copy of it.
    #[test]
    fn a_test_reaching_its_join_through_an_edge_block_folds_its_statements_ahead() {
        let graph = graph_of(
            13,
            &[
                (2, 3),
                (4, 1),
                (4, 5),
                (5, 6),
                (5, 7),
                (6, 9),
                (7, 9),
                (7, 8),
                (9, 10),
                (10, 12),
                (11, 1),
                (12, 1),
                (12, 11),
                (0, 4),
                (11, 1),
                (9, 1),
                (2, 3),
                (10, 11),
                (1, 2),
                (0, 1),
            ],
        );
        let none = BlockSet::new(13);
        let edges = set_of(13, &[6, 8]);
        let recover = |edge_blocks: &BlockSet| {
            let options = StructureOptions {
                condition_only: none.clone(),
                condition_reachability: none.clone(),
                condition_expressible: set_of(13, &[7]),
                edge_blocks: edge_blocks.clone(),
                ..StructureOptions::bare(13, &[])
            };
            structure(&graph, NodeId::new(0), &options)
        };
        // Every hoisting test the tree's conditions hold, as (test, edge).
        let hoisted = |root: &Region| {
            let mut found = Vec::new();
            let mut stack = vec![root];
            while let Some(region) = stack.pop() {
                match region {
                    Region::Seq(parts) => stack.extend(parts.iter()),
                    Region::If(inner) => {
                        let mut pending = vec![&inner.predicate];
                        while let Some(predicate) = pending.pop() {
                            match predicate {
                                Predicate::Hoisted { block, edge, .. } => {
                                    found.push((*block, *edge))
                                }
                                Predicate::All(parts) | Predicate::Any(parts) => {
                                    pending.extend(parts.iter());
                                }
                                Predicate::Test { .. } => {}
                            }
                        }
                        stack.push(&inner.then_branch);
                        stack.extend(inner.else_branch.iter());
                    }
                    Region::Loop(inner) => stack.push(&inner.body),
                    _ => {}
                }
            }
            found
        };
        let folded = recover(&edges);
        assert_total(&graph, NodeId::new(0), &folded);
        assert_eq!(folded.metrics.gotos, 0, "{:#?}", folded.root);
        assert!(
            hoisted(&folded.root)
                .iter()
                .any(|&(_, edge)| edge == NodeId::new(6)),
            "{:#?}",
            folded.root
        );
        assert_runs_as_the_graph(&graph, NodeId::new(0), &none, &edges, &folded);
        assert_eq!(placements(&folded, 9), 1, "{:#?}", folded.root);
        let plain = recover(&none);
        assert!(hoisted(&plain.root).is_empty(), "{:#?}", plain.root);
        assert_eq!(
            placements(&plain, 9),
            2,
            "control: read as blocks of their own, each way copies 9: {:#?}",
            plain.root
        );
    }

    /// A four-way branch recovers as a switch with one arm per target and a
    /// shared follow.
    #[test]
    fn multiway_branch_is_a_switch() {
        let graph = graph_of(
            6,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 5),
                (2, 5),
                (3, 5),
                (4, 5),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.switches, 1);
        assert_eq!(recovered.metrics.gotos, 0);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Switch(switch)) = top.first() else {
            panic!("expected a switch, got {top:?}");
        };
        assert_eq!(switch.cases.len(), 4);
        assert_eq!(switch.follow, Some(NodeId::new(5)));
    }

    /// **A switch whose arms mostly meet at a tail running on to its join
    /// resumes at the tail.** dash's `command` (`0xc1cc` on ARM, its
    /// structuring input whole): every case but `TCASE` breaks to the check
    /// of the closing token (11), whose failure ends the function, and
    /// `TCASE` jumps past it to the redirections (12). Resumed at 11, only
    /// `TCASE` jumps; resumed at 12, 11 goes in the first arm reaching it and
    /// every other arm jumps to it.
    #[test]
    fn a_switch_whose_arms_mostly_meet_at_a_tail_resumes_at_it() {
        let graph = graph_of(
            83,
            &[
                (0, 70),
                (0, 1),
                (1, 38),
                (1, 70),
                (1, 70),
                (1, 70),
                (1, 2),
                (1, 2),
                (1, 70),
                (1, 39),
                (1, 70),
                (1, 70),
                (1, 70),
                (1, 70),
                (1, 70),
                (1, 70),
                (1, 19),
                (1, 29),
                (1, 70),
                (1, 70),
                (1, 9),
                (1, 9),
                (1, 37),
                (2, 3),
                (3, 65),
                (3, 4),
                (4, 63),
                (4, 5),
                (5, 54),
                (5, 6),
                (6, 7),
                (7, 8),
                (7, 82),
                (9, 81),
                (9, 10),
                (10, 11),
                (11, 80),
                (11, 12),
                (12, 14),
                (13, 14),
                (14, 13),
                (14, 15),
                (15, 7),
                (15, 16),
                (16, 18),
                (16, 17),
                (17, 18),
                (18, 7),
                (19, 79),
                (19, 20),
                (20, 79),
                (20, 21),
                (21, 79),
                (21, 22),
                (22, 66),
                (22, 23),
                (23, 25),
                (24, 25),
                (25, 24),
                (25, 26),
                (26, 70),
                (26, 27),
                (27, 81),
                (27, 28),
                (28, 11),
                (29, 75),
                (29, 30),
                (30, 33),
                (31, 75),
                (31, 32),
                (32, 33),
                (33, 31),
                (33, 34),
                (34, 74),
                (34, 35),
                (35, 36),
                (36, 11),
                (37, 11),
                (38, 11),
                (39, 40),
                (39, 41),
                (40, 80),
                (41, 78),
                (41, 42),
                (42, 43),
                (43, 53),
                (43, 44),
                (44, 52),
                (44, 45),
                (45, 47),
                (46, 47),
                (47, 46),
                (47, 48),
                (48, 77),
                (48, 49),
                (49, 53),
                (49, 50),
                (50, 43),
                (50, 51),
                (52, 45),
                (53, 12),
                (54, 6),
                (54, 55),
                (55, 6),
                (55, 56),
                (56, 6),
                (56, 57),
                (57, 6),
                (57, 58),
                (58, 77),
                (58, 59),
                (59, 76),
                (59, 60),
                (60, 62),
                (60, 61),
                (61, 76),
                (61, 62),
                (62, 7),
                (63, 71),
                (63, 64),
                (64, 3),
                (65, 3),
                (66, 67),
                (66, 68),
                (67, 68),
                (68, 69),
                (68, 27),
                (69, 27),
                (71, 64),
                (71, 72),
                (72, 64),
                (72, 73),
                (73, 3),
                (74, 36),
            ],
        );
        let options = StructureOptions {
            condition_only: set_of(83, &[4, 5, 7, 16, 20, 34, 44, 50, 55, 56, 57, 61, 68, 72]),
            condition_reachability: set_of(83, &[7, 20, 34, 55, 61]),
            condition_expressible: set_of(
                83,
                &[
                    0, 3, 4, 5, 7, 9, 11, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 29, 31, 33, 34,
                    39, 41, 43, 44, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59, 60, 61, 63, 66, 68, 71,
                    72,
                ],
            ),
            condition_operand: set_of(
                83,
                &[
                    4, 5, 11, 14, 16, 19, 21, 25, 33, 44, 50, 54, 55, 56, 57, 58, 60, 61, 68, 71,
                    72,
                ],
            ),
            no_return: set_of(83, &[51, 70, 75, 76, 77, 78, 79, 80, 81, 82]),
            polar: set_of(
                83,
                &[
                    0, 3, 4, 5, 7, 9, 11, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 29, 31, 33, 34,
                    39, 41, 43, 44, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59, 60, 61, 63, 66, 68, 71,
                    72,
                ],
            ),
            edge_blocks: set_of(83, &[40]),
            ..StructureOptions::bare(83, &[])
        };
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 1, "{:#?}", recovered.root);
    }

    /// **A recovery that needs no goto either way nests its statements
    /// rather than saying them in a condition.** `which`'s
    /// `find_command_in_path` (`0x1c5f` on x86-64): the recovery that nests
    /// them jumps until a tail is offered as a join, and the one folding
    /// statements needs no goto before; finished, both need none, and the
    /// one with no statement in a condition is kept (`if (*p == '~') {
    /// q = tilde_expand(p); ... } else { q = p; }`, not `if ((q = p, *p ==
    /// '~') && (q = tilde_expand(p), ...))`).
    #[test]
    fn a_recovery_needing_no_goto_either_way_nests_its_statements() {
        let graph = graph_of(
            29,
            &[
                (0, 1),
                (0, 2),
                (1, 25),
                (2, 4),
                (2, 3),
                (3, 4),
                (4, 8),
                (4, 5),
                (5, 8),
                (5, 6),
                (6, 8),
                (6, 7),
                (7, 9),
                (8, 9),
                (9, 25),
                (10, 12),
                (10, 11),
                (11, 13),
                (12, 13),
                (13, 14),
                (13, 15),
                (14, 27),
                (15, 16),
                (15, 17),
                (16, 19),
                (17, 19),
                (17, 18),
                (18, 25),
                (19, 22),
                (19, 20),
                (20, 22),
                (20, 21),
                (21, 25),
                (22, 24),
                (22, 23),
                (23, 24),
                (23, 27),
                (24, 25),
                (25, 28),
                (25, 26),
                (26, 28),
                (26, 10),
                (28, 27),
            ],
        );
        let options = StructureOptions {
            condition_only: set_of(29, &[4, 5, 6, 10, 13, 15, 19, 20, 23, 25, 26]),
            condition_reachability: set_of(29, &[26]),
            condition_expressible: set_of(
                29,
                &[0, 2, 4, 5, 6, 10, 13, 15, 17, 19, 20, 22, 23, 25, 26],
            ),
            condition_operand: set_of(29, &[4, 5, 6, 10, 13, 15, 19, 20, 23, 25, 26]),
            polar: set_of(29, &[0, 2, 4, 5, 6, 10, 13, 15, 17, 19, 20, 22, 23, 25, 26]),
            edge_blocks: set_of(29, &[14, 16]),
            ..StructureOptions::bare(29, &[])
        };
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert_eq!(
            recovered
                .root
                .statements_in_conditions(&options.condition_only),
            0,
            "{:#?}",
            recovered.root
        );
        assert_runs_as_the_graph(
            &graph,
            NodeId::new(0),
            &options.condition_only,
            &options.edge_blocks,
            &recovered,
        );
    }

    /// **A tail offered as a join is judged on the recovery finished with
    /// it.** dash's `evalcase` (`0x39dc` on ARM, its structuring input
    /// whole): judged before the loop and join splits that follow it, the
    /// offer is kept or refused on a recovery those splits still change, and
    /// two gotos are left; judged on the finished recovery, one.
    #[test]
    fn an_offered_join_is_judged_on_the_finished_recovery() {
        let graph = graph_of(
            16,
            &[
                (0, 2),
                (0, 1),
                (1, 2),
                (2, 9),
                (2, 3),
                (3, 4),
                (4, 9),
                (4, 5),
                (5, 7),
                (5, 12),
                (6, 12),
                (6, 7),
                (7, 6),
                (7, 8),
                (8, 13),
                (8, 9),
                (9, 10),
                (10, 11),
                (10, 15),
                (12, 4),
                (12, 9),
                (13, 9),
                (13, 14),
                (14, 10),
            ],
        );
        let options = StructureOptions {
            condition_only: set_of(16, &[4, 8, 10]),
            condition_reachability: set_of(16, &[8, 10]),
            condition_expressible: set_of(16, &[0, 2, 4, 5, 6, 7, 8, 10, 12, 13]),
            condition_operand: set_of(16, &[4, 5, 6, 7, 12, 13]),
            no_return: set_of(16, &[15]),
            polar: set_of(16, &[0, 2, 4, 5, 6, 7, 8, 10, 12, 13]),
            ..StructureOptions::bare(16, &[])
        };
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 1, "{:#?}", recovered.root);
    }

    /// dash's `expmeta` (`0x5b40` on ARM), its structuring input whole: the
    /// graph and what the host states of its blocks.
    fn expmeta() -> (DirectedGraph<'static, (), ()>, StructureOptions<'static>) {
        let graph = graph_of(
            66,
            &[
                (0, 9),
                (0, 1),
                (1, 2),
                (2, 49),
                (2, 3),
                (3, 40),
                (3, 4),
                (4, 6),
                (4, 5),
                (5, 45),
                (5, 6),
                (6, 7),
                (7, 2),
                (7, 18),
                (8, 18),
                (8, 9),
                (9, 15),
                (9, 10),
                (10, 11),
                (10, 65),
                (12, 62),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 12),
                (15, 16),
                (16, 58),
                (16, 17),
                (17, 14),
                (18, 25),
                (18, 19),
                (19, 22),
                (20, 25),
                (20, 21),
                (21, 22),
                (22, 20),
                (22, 23),
                (23, 20),
                (23, 24),
                (24, 21),
                (24, 25),
                (25, 26),
                (25, 27),
                (26, 27),
                (27, 10),
                (27, 28),
                (28, 61),
                (28, 29),
                (29, 30),
                (30, 31),
                (30, 32),
                (31, 32),
                (32, 38),
                (32, 33),
                (33, 38),
                (33, 34),
                (34, 32),
                (34, 35),
                (35, 32),
                (35, 36),
                (36, 59),
                (36, 37),
                (37, 33),
                (37, 38),
                (38, 10),
                (38, 39),
                (39, 10),
                (40, 45),
                (40, 41),
                (41, 46),
                (41, 42),
                (42, 44),
                (43, 2),
                (44, 43),
                (44, 8),
                (45, 42),
                (45, 46),
                (46, 47),
                (46, 48),
                (47, 18),
                (48, 44),
                (49, 50),
                (49, 51),
                (50, 51),
                (51, 53),
                (52, 57),
                (52, 53),
                (53, 54),
                (53, 55),
                (54, 55),
                (55, 56),
                (55, 52),
                (56, 44),
                (57, 7),
                (58, 14),
                (59, 64),
                (59, 60),
                (60, 32),
                (61, 30),
                (62, 10),
                (62, 63),
                (63, 10),
                (64, 60),
            ],
        );
        let options = StructureOptions {
            condition_only: set_of(
                66,
                &[
                    2, 3, 4, 5, 7, 8, 9, 10, 15, 18, 22, 28, 32, 34, 40, 44, 46, 55,
                ],
            ),
            condition_reachability: set_of(66, &[10]),
            condition_expressible: set_of(
                66,
                &[
                    0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 15, 16, 18, 20, 22, 23, 24, 25, 27, 28, 30, 32,
                    33, 34, 35, 36, 37, 38, 40, 41, 44, 45, 46, 49, 52, 53, 55, 59, 62,
                ],
            ),
            condition_operand: set_of(
                66,
                &[
                    2, 3, 4, 5, 7, 8, 9, 15, 16, 18, 22, 23, 27, 28, 32, 33, 34, 40, 44, 46, 55, 62,
                ],
            ),
            no_return: set_of(66, &[65]),
            polar: set_of(
                66,
                &[
                    0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 15, 16, 18, 20, 22, 23, 24, 25, 27, 28, 30, 32,
                    33, 34, 35, 36, 37, 38, 40, 41, 44, 45, 46, 49, 52, 53, 55, 59, 62,
                ],
            ),
            edge_blocks: set_of(66, &[13, 43, 47, 56]),
            ..StructureOptions::bare(66, &[])
        };
        (graph, options)
    }

    /// **The recovery's goto count is what C writes as a goto.** dash's
    /// `expmeta` (`0x5b40` on ARM, its structuring input whole): its loop
    /// over a directory's entries goes back to the loop around it from inside
    /// it, a `continue` C cannot say there - a bare one names the inner loop -
    /// so it is a goto, and [`StructureMetrics::gotos`] counts it as
    /// [`Region::gotos`] does, not only the tree's [`Region::Goto`]s, and
    /// the header it goes to is labelled ([`Structured::labels`]).
    /// **An arm that runs on to a `goto` to its own target writes no `goto`
    /// of its own.** `if (c) { b1; goto L; } goto L;` is `if (c) b1; goto
    /// L;`, as RISC-V `readtoken1` in dash has its `synstack` tails, and an
    /// `if` nested at an arm's end loses its own the same way. A loop's body
    /// ending in `goto L` goes round the loop, not on to the `goto` after it,
    /// and keeps it.
    #[test]
    fn an_arm_running_on_to_a_goto_to_its_target_writes_none() {
        let node = NodeId::new;
        let test = |block| Predicate::Test {
            block: node(block),
            negated: false,
        };
        let iffed = |block, then_branch, else_branch| {
            Region::If(Box::new(IfRegion {
                predicate: test(block),
                then_branch,
                else_branch,
            }))
        };
        let tidy = |root| {
            let mut structured = Structured::new(root, StructureMetrics::default());
            structured.drop_onward_gotos();
            structured
        };
        let label = node(9);
        let nested = iffed(
            3,
            Region::Seq(vec![Region::Block(node(4)), Region::Goto(label)]),
            None,
        );
        let root = Region::Seq(vec![
            Region::Block(node(0)),
            iffed(
                1,
                Region::Seq(vec![Region::Block(node(2)), Region::Goto(label)]),
                Some(Region::Seq(vec![Region::Block(node(5)), nested])),
            ),
            Region::Goto(label),
        ]);
        let tidied = tidy(root);
        let expected = Region::Seq(vec![
            Region::Block(node(0)),
            iffed(
                1,
                Region::Block(node(2)),
                Some(Region::Seq(vec![
                    Region::Block(node(5)),
                    iffed(3, Region::Block(node(4)), None),
                ])),
            ),
            Region::Goto(label),
        ]);
        assert_eq!(tidied.root, expected);
        assert_eq!(tidied.metrics.gotos, 1);
        assert_eq!(tidied.labels, BTreeSet::from([label]));

        // Only a `goto` in each arm: the test stays, with one of them.
        let bare = tidy(Region::Seq(vec![
            iffed(1, Region::Goto(label), None),
            Region::Goto(label),
        ]));
        assert_eq!(bare.metrics.gotos, 2, "{:#?}", bare.root);

        // An arm left with nothing asks the other way round.
        let flipped = tidy(Region::Seq(vec![
            iffed(1, Region::Goto(label), Some(Region::Block(node(2)))),
            Region::Goto(label),
        ]));
        assert_eq!(
            flipped.root,
            Region::Seq(vec![
                Region::If(Box::new(IfRegion {
                    predicate: test(1).negate(),
                    then_branch: Region::Block(node(2)),
                    else_branch: None,
                })),
                Region::Goto(label),
            ])
        );

        let looped = Region::Seq(vec![
            Region::Loop(Box::new(LoopRegion {
                header: node(1),
                kind: StructuredLoop::Endless,
                predicate: None,
                body: Region::Seq(vec![Region::Block(node(1)), Region::Goto(label)]),
                follow: None,
            })),
            Region::Goto(label),
        ]);
        assert_eq!(tidy(looped.clone()).root, looped);
    }

    #[test]
    fn a_continue_of_an_outer_loop_counts_as_the_goto_it_is() {
        let (graph, options) = expmeta();
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        let jumps = goto_jumps(&recovered.root).len();
        assert!(
            recovered.root.gotos() > jumps,
            "a transfer only C's goto writes: {:#?}",
            recovered.root
        );
        assert_eq!(
            recovered.metrics.gotos,
            recovered.root.gotos(),
            "{:#?}",
            recovered.root
        );
        // Each goto names a label: the `continue`'s too, past the targets of
        // the tree's gotos.
        let named: BTreeSet<NodeId> = recovered.root.jumps().into_iter().flatten().collect();
        assert_eq!(recovered.labels, named, "{:#?}", recovered.root);
        assert!(
            recovered.labels.len() > goto_targets(&recovered.root).len(),
            "{:#?}",
            recovered.root
        );
    }

    /// **A lap that tests its loop's condition again goes to the test.**
    /// dash's `expmeta` reads directory entries in one `while
    /// (!int_pending() && (dp = readdir64(dirp)) != NULL)`; compiled, the
    /// `atend` arm (37) tests `int_pending()` again and goes on to `readdir`
    /// (33) past the head's test (32), so the body is entered at two blocks
    /// and recovers as two loops. Stated as repeating 32's test, 37 goes to
    /// 32, and the outer loop's `continue`s are the inner loop's own. The
    /// recovery says it routed 37 ([`Structured::repeats`]), so a printer
    /// states 37's copy of the test once, in 32: a recovery that took no
    /// route says none.
    #[test]
    fn a_lap_that_tests_its_loops_condition_again_goes_to_the_test() {
        let (graph, options) = expmeta();
        let plain = structure(&graph, NodeId::new(0), &options);
        let routed = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions {
                repeated_tests: BTreeMap::from([(NodeId::new(37), NodeId::new(32))]),
                ..options.clone()
            },
        );
        assert_total(&graph, NodeId::new(0), &routed);
        assert!(
            routed.metrics.gotos < plain.metrics.gotos,
            "routed {} against {}: {:#?}",
            routed.metrics.gotos,
            plain.metrics.gotos,
            routed.root
        );
        assert_eq!(
            routed.repeats,
            BTreeMap::from([(NodeId::new(37), NodeId::new(32))]),
            "the route taken"
        );
        assert!(plain.repeats.is_empty(), "no route stated");
    }

    /// dash's `argstr` (`0x62dc` on ARM), its structuring input whole: the
    /// graph and what the host states of its blocks.
    fn argstr() -> (DirectedGraph<'static, (), ()>, StructureOptions<'static>) {
        let graph = graph_of(
            102,
            &[
                (0, 1),
                (0, 76),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 16),
                (4, 5),
                (5, 6),
                (5, 7),
                (6, 9),
                (7, 8),
                (7, 38),
                (8, 9),
                (9, 42),
                (9, 10),
                (10, 11),
                (11, 33),
                (11, 12),
                (12, 13),
                (13, 19),
                (13, 14),
                (14, 19),
                (14, 15),
                (15, 30),
                (15, 26),
                (15, 19),
                (15, 27),
                (15, 19),
                (15, 24),
                (15, 19),
                (15, 21),
                (16, 5),
                (16, 17),
                (17, 44),
                (17, 18),
                (18, 13),
                (18, 19),
                (19, 20),
                (20, 4),
                (21, 23),
                (21, 22),
                (22, 95),
                (22, 23),
                (23, 31),
                (24, 72),
                (24, 25),
                (25, 3),
                (26, 3),
                (27, 28),
                (27, 64),
                (28, 29),
                (29, 3),
                (30, 31),
                (31, 20),
                (31, 32),
                (32, 4),
                (33, 36),
                (33, 34),
                (34, 20),
                (34, 35),
                (35, 36),
                (36, 45),
                (36, 37),
                (37, 4),
                (38, 77),
                (38, 39),
                (39, 40),
                (40, 41),
                (40, 42),
                (41, 11),
                (42, 43),
                (42, 101),
                (44, 7),
                (45, 2),
                (45, 46),
                (46, 47),
                (46, 48),
                (47, 58),
                (48, 58),
                (48, 49),
                (49, 50),
                (50, 90),
                (50, 51),
                (51, 52),
                (51, 53),
                (52, 58),
                (53, 2),
                (53, 54),
                (54, 2),
                (54, 55),
                (55, 56),
                (55, 57),
                (56, 58),
                (57, 50),
                (57, 58),
                (58, 60),
                (58, 59),
                (59, 2),
                (60, 97),
                (60, 61),
                (61, 62),
                (62, 2),
                (62, 63),
                (63, 2),
                (64, 71),
                (64, 65),
                (65, 78),
                (65, 66),
                (66, 69),
                (67, 93),
                (67, 68),
                (68, 69),
                (69, 67),
                (69, 70),
                (70, 94),
                (70, 71),
                (71, 65),
                (72, 74),
                (72, 73),
                (73, 74),
                (74, 75),
                (74, 25),
                (75, 25),
                (76, 45),
                (77, 40),
                (78, 82),
                (78, 79),
                (79, 80),
                (80, 82),
                (80, 81),
                (81, 82),
                (82, 84),
                (82, 83),
                (83, 100),
                (83, 84),
                (84, 86),
                (84, 88),
                (85, 88),
                (85, 86),
                (86, 85),
                (86, 87),
                (87, 88),
                (88, 96),
                (88, 89),
                (89, 29),
                (90, 55),
                (90, 91),
                (91, 92),
                (91, 55),
                (92, 58),
                (93, 94),
                (94, 79),
                (94, 80),
                (95, 3),
                (96, 29),
                (97, 99),
                (97, 98),
                (98, 62),
                (99, 2),
            ],
        );
        let options = StructureOptions {
            condition_only: set_of(
                102,
                &[
                    9, 11, 13, 16, 17, 21, 31, 33, 34, 42, 45, 50, 51, 53, 54, 58, 67, 83, 90, 91,
                ],
            ),
            condition_reachability: set_of(102, &[42, 54, 58, 67, 83]),
            condition_expressible: set_of(
                102,
                &[
                    0, 4, 5, 7, 9, 11, 13, 14, 16, 17, 18, 21, 22, 24, 27, 31, 33, 34, 36, 38, 40,
                    42, 45, 46, 48, 50, 51, 53, 54, 55, 57, 58, 60, 62, 64, 65, 67, 69, 70, 72, 74,
                    78, 80, 82, 83, 84, 85, 86, 88, 90, 91, 94, 97,
                ],
            ),
            condition_operand: set_of(
                102,
                &[
                    9, 11, 13, 14, 16, 17, 21, 31, 33, 34, 45, 50, 51, 53, 54, 58, 65, 69, 78, 80,
                    90, 91, 94, 97,
                ],
            ),
            no_return: set_of(102, &[100, 101]),
            polar: set_of(
                102,
                &[
                    0, 4, 5, 7, 9, 11, 13, 14, 16, 17, 18, 21, 22, 24, 27, 31, 33, 34, 36, 38, 40,
                    42, 45, 46, 48, 50, 51, 53, 54, 55, 57, 58, 60, 62, 64, 65, 67, 69, 70, 72, 74,
                    78, 80, 82, 83, 84, 85, 86, 88, 90, 91, 94, 97,
                ],
            ),
            edge_blocks: set_of(102, &[6, 8, 10, 12, 39, 41, 47, 49, 52, 56, 59, 87, 92]),
            ..StructureOptions::bare(102, &[])
        };
        (graph, options)
    }

    /// **A cleanup a compiler threaded past a test the path decides recovers
    /// once.** dash's `argstr` inlines `expbackq`, whose cleanup tests
    /// `in.buf` (94) and then `in.fd >= 0` (80). Its read loop leaves at 65
    /// when `in.fd < 0`, and the compiler gives that exit its own `in.buf`
    /// test (78), which goes past 80 straight to 82. Stated as threaded past
    /// 80, 78's edge goes through 80, which leaves 78 a repeat of 94; stated
    /// so, 78 goes to 94, and the cleanup is written once with 12 gotos, the
    /// source's 12 (its 6 with `exptilde`'s 4 and `expbackq`'s 2). The split
    /// that would save three more copies 55 blocks, each counted as one
    /// statement here, more than three gotos are worth
    /// ([`STATEMENTS_PER_GOTO`](inputs::STATEMENTS_PER_GOTO)).
    #[test]
    fn a_cleanup_threaded_past_a_test_the_path_decides_recovers_once() {
        let (graph, options) = argstr();
        let routed = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions {
                threaded_tests: BTreeMap::from([((NodeId::new(78), 0), NodeId::new(80))]),
                repeated_tests: BTreeMap::from([(NodeId::new(78), NodeId::new(94))]),
                ..options.clone()
            },
        );
        assert_total(&graph, NodeId::new(0), &routed);
        assert_eq!(routed.metrics.gotos, 12, "{:#?}", routed.root);
        for cleanup in [94, 80] {
            assert_eq!(placements(&routed, cleanup), 1, "{:#?}", routed.root);
        }
        // Control: 78 is no repeat of 94 until its edge goes through 80. One
        // arm there runs on to a goto to its own target and writes none.
        let repeated_alone = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions {
                repeated_tests: BTreeMap::from([(NodeId::new(78), NodeId::new(94))]),
                ..options
            },
        );
        assert_eq!(
            repeated_alone.metrics.gotos, 16,
            "{:#?}",
            repeated_alone.root
        );
    }

    /// Case selectors that share a target collapse into a single arm, so a
    /// dispatch left with two distinct targets recovers as an `if`.
    #[test]
    fn shared_case_targets_collapse_to_one_arm() {
        // Selectors 0 and 2 reach block 1; selector 1 reaches block 2. That is
        // `case A: case C:` sharing a body, not two arms with equal contents.
        let graph = graph_of(4, &[(0, 1), (0, 2), (0, 1), (1, 3), (2, 3)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        // Two distinct targets: this reads as a conditional, not a switch.
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Test {
                block: NodeId::new(0),
                negated: false
            }
        );
    }

    /// A switch arm that runs on into the next instead of rejoining falls
    /// through to it, labelled by nothing and jumped to by nothing.
    #[test]
    fn case_fall_through_runs_on_into_the_next_arm() {
        // Arm 1 runs on into arm 2 instead of rejoining; arm 2 branches before
        // it reaches the join 4, so it is no straight-line tail to repeat.
        let graph = graph_of(
            8,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (2, 4),
                (2, 7),
                (7, 4),
                (3, 4),
                (4, 5),
                (4, 6),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_runs_as_the_graph(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]).condition_only,
            &BlockSet::new(graph.node_count()),
            &recovered,
        );
        assert_eq!(recovered.metrics.switches, 1);
        assert_eq!(recovered.metrics.fall_throughs, 1, "{:#?}", recovered.root);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert!(recovered.labels.is_empty(), "{:?}", recovered.labels);
    }

    /// **Tests over one value that lead back to their root are a loop, not a
    /// `switch`.** `do { .. } while (c != 10 && c != -129)` compiles to a
    /// header testing `c` whose test block tests `c` again and returns to the
    /// header. Walked as a case tree, its paths cycle without narrowing the
    /// value, so the walk never ends and grows without bound. The header
    /// computes, so it is no `while` condition, and every shape the loop can
    /// take emits it as a block, which consults the case tests.
    #[test]
    fn case_tests_around_a_loop_are_no_switch() {
        // 1 tests `c != 10` into 2; 2 tests `c == -129` out to 3, else back
        // to 1. Only 2 does nothing but test.
        let graph = graph_of(4, &[(0, 1), (1, 2), (1, 3), (2, 3), (2, 1)]);
        let test = |relation, constant| CaseTest {
            value: 0,
            relation,
            constant,
            holds: 0,
            entered: true,
        };
        let options = StructureOptions {
            statements: Vec::new(),
            regions: &[],
            condition_only: set_of(4, &[2]),
            condition_reachability: BlockSet::new(4),
            condition_expressible: BlockSet::new(4),
            condition_operand: BlockSet::new(4),
            case_tests: BTreeMap::from([
                (NodeId::new(1), test(CaseRelation::NotEqual, 10)),
                (NodeId::new(2), test(CaseRelation::Equal, -129)),
            ]),
            no_return: BlockSet::new(4),
            edge_blocks: BlockSet::new(4),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(4),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 1);
        assert_eq!(recovered.metrics.value_switches, 0);
    }

    /// An inner-body edge to the outer follow recovers as a `break` naming the
    /// outer header, not the inner one.
    #[test]
    fn nested_loops_break_the_loop_they_name() {
        // Outer header 1, inner header 2. The inner body at 3 leaves straight
        // to the outer follow at 6, which is a break of the *outer* loop.
        let graph = graph_of(
            7,
            &[
                (0, 1),
                (1, 2),
                (1, 6),
                (2, 3),
                (2, 5),
                (3, 6),
                (3, 4),
                (4, 2),
                (5, 1),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 2);

        let mut targets = Vec::new();
        collect_breaks(&recovered.root, &mut targets);
        assert!(
            targets.contains(&NodeId::new(1)),
            "expected a break naming the outer header, got {targets:?}"
        );
    }

    /// **A tail past an outer loop is repeated only where a goto is worth
    /// it.** In [`a_loop_resumes_where_leaving_would_cost_a_goto`]'s graph,
    /// `8`, which ends the function through `9`, is the outer loop's follow,
    /// which a `break` inside the inner loop cannot reach: the tail is
    /// repeated there. Where it prints more statements than
    /// [`STATEMENTS_PER_GOTO`](inputs::STATEMENTS_PER_GOTO), it is placed
    /// once, past the outer loop, and the recovery needs no goto more.
    #[test]
    fn a_tail_past_an_outer_loop_is_repeated_only_where_a_goto_is_worth_it() {
        let graph = graph_of(
            10,
            &[
                (0, 1),
                (1, 2),
                (1, 6),
                (2, 3),
                (3, 4),
                (3, 8),
                (4, 3),
                (4, 5),
                (4, 7),
                (5, 6),
                (6, 1),
                (6, 8),
                (8, 9),
            ],
        );
        let entry = NodeId::new(0);
        let weighed = |tail: usize| {
            let mut statements = vec![1; 10];
            if let Some(slot) = statements.get_mut(8) {
                *slot = tail;
            }
            structure(
                &graph,
                entry,
                &StructureOptions {
                    statements,
                    ..StructureOptions::bare(10, &[])
                },
            )
        };
        let placed = |recovered: &Structured| {
            let mut placed = 0_usize;
            recovered.root.for_each_block(&mut |node| {
                placed = placed.saturating_add(usize::from(node == NodeId::new(8)));
            });
            placed
        };
        // `9` prints one statement, so `8` printing one fewer than the bound
        // leaves the tail at it.
        let short = weighed(inputs::STATEMENTS_PER_GOTO.saturating_sub(1));
        assert_total(&graph, entry, &short);
        assert_eq!(short.metrics.gotos, 1, "{:#?}", short.root);
        assert!(placed(&short) > 1, "{:#?}", short.root);
        let long = weighed(inputs::STATEMENTS_PER_GOTO);
        assert_total(&graph, entry, &long);
        assert_eq!(placed(&long), 1, "{:#?}", long.root);
        assert_eq!(long.metrics.gotos, 1, "{:#?}", long.root);
    }

    /// Blocks the walk never reaches are counted as unreached and left out of
    /// the tree.
    #[test]
    fn unreachable_blocks_are_counted_not_placed() {
        let graph = graph_of(4, &[(0, 1), (2, 3)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_eq!(recovered.metrics.blocks, 2);
        assert_eq!(recovered.metrics.unreached, 2);
        assert_eq!(recovered.blocks().len(), 2);
    }

    /// A single-block graph recovers as one block with no gotos.
    #[test]
    fn single_block_function_structures() {
        let graph = graph_of(1, &[]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.root, Region::Block(NodeId::new(0)));
        assert_eq!(recovered.metrics.gotos, 0);
    }

    /// A block branching to itself recovers as a loop.
    #[test]
    fn self_loop_is_a_loop() {
        let graph = graph_of(3, &[(0, 1), (1, 1), (1, 2)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.loops, 1);
    }

    /// A block whose two edges reach the same target is not a conditional.
    #[test]
    fn both_edges_to_one_block_is_not_a_branch() {
        let graph = graph_of(2, &[(0, 1), (0, 1)]);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.ifs, 0);
        assert_eq!(recovered.metrics.gotos, 0);
    }

    /// A long straight line takes the iterative path, so the walk never
    /// descends and the stack stays bounded.
    #[test]
    fn deep_chain_does_not_overflow_the_stack() {
        // A long straight line takes the iterative path: no region is entered,
        // so the walk never descends and the depth bound is never near. The
        // recursive dimension is covered by the two tests above it.
        let count: usize = 20_000;
        let edges: Vec<(usize, usize)> = (0..count.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        let graph = graph_of(count, &edges);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_eq!(recovered.metrics.blocks, count);
        assert_eq!(recovered.metrics.gotos, 0);
    }

    /// A nested arm returning where the outer join also leads is reached by
    /// goto rather than duplicated, and the recovery still accounts for every
    /// block.
    #[test]
    fn early_return_from_a_nested_arm_leaves_by_goto_not_by_duplication() {
        // Inner arm 3 returns at 6, which is also where the outer join leads.
        let graph = graph_of(
            7,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 4),
                (3, 6),
                (4, 5),
                (2, 5),
                (5, 6),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
    }

    /// A protected region recovers as a `Region::Try` whose body and handler
    /// are placed inside it and whose follow is placed after it.
    #[test]
    fn protected_region_becomes_a_try_with_its_handler() {
        // Body 1 -> 2 -> 5 (follow). Handler 3 -> 4, reachable only by raising.
        let graph = graph_of(6, &[(0, 1), (1, 2), (2, 5), (3, 4)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(6, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(6, &[3, 4]),
            }],
        }];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 1);
        assert_eq!(recovered.metrics.handlers, 1);
        assert_eq!(recovered.metrics.gotos, 0);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(protected)) = top.get(1) else {
            panic!("expected a try, got {top:?}");
        };
        assert_eq!(protected.follow, Some(NodeId::new(5)));
        assert_eq!(
            protected.body,
            Region::Seq(vec![
                Region::Block(NodeId::new(1)),
                Region::Block(NodeId::new(2)),
            ])
        );
        assert_eq!(protected.handlers.len(), 1);
        assert_eq!(
            protected.handlers.first().map(|h| &h.body),
            Some(&Region::Seq(vec![
                Region::Block(NodeId::new(3)),
                Region::Block(NodeId::new(4)),
            ]))
        );
        // The follow is placed after the try, not inside it.
        assert_eq!(top.get(2), Some(&Region::Block(NodeId::new(5))));
    }

    /// A finally the body falls into when it completes - as a script runtime
    /// and Delphi's try/finally run one - is the handler running, not the code
    /// after the `try`: the region resumes where the finally leaves, and the
    /// finally is not also placed after it.
    #[test]
    fn a_finally_the_body_falls_into_resumes_where_it_leaves() {
        // Body 1 falls into the finally 2, which leaves to 3.
        let graph = graph_of(4, &[(0, 1), (1, 2), (2, 3)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(4, &[1]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Finally,
                entry: NodeId::new(2),
                blocks: set_of(4, &[2]),
            }],
        }];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.gotos, 0);
        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(protected)) = top.get(1) else {
            panic!("expected a try, got {top:?}");
        };
        assert_eq!(protected.follow, Some(NodeId::new(3)));
        assert_eq!(protected.body, Region::Block(NodeId::new(1)));
        assert_eq!(
            protected.handlers.first().map(|h| &h.body),
            Some(&Region::Block(NodeId::new(2)))
        );
        assert_eq!(top.get(2), Some(&Region::Block(NodeId::new(3))));
        assert_eq!(top.len(), 3, "{top:?}");
    }

    /// Several clauses on one region recover as several handlers in their
    /// declared order.
    #[test]
    fn several_clauses_on_one_region_are_several_handlers() {
        let graph = graph_of(6, &[(0, 1), (1, 5), (2, 5), (3, 5), (4, 5)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(6, &[1]),
            handlers: vec![
                ProtectedHandler {
                    kind: ProtectedHandlerKind::Catch,
                    entry: NodeId::new(2),
                    blocks: set_of(6, &[2]),
                },
                ProtectedHandler {
                    kind: ProtectedHandlerKind::Catch,
                    entry: NodeId::new(3),
                    blocks: set_of(6, &[3]),
                },
                ProtectedHandler {
                    kind: ProtectedHandlerKind::Finally,
                    entry: NodeId::new(4),
                    blocks: set_of(6, &[4]),
                },
            ],
        }];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.handlers, 3);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(protected)) = top.get(1) else {
            panic!("expected a try, got {top:?}");
        };
        let kinds: Vec<HandlerKind> = protected.handlers.iter().map(|h| h.kind).collect();
        assert_eq!(
            kinds,
            vec![HandlerKind::Catch, HandlerKind::Catch, HandlerKind::Finally],
            "clause order is the declared order"
        );
    }

    /// A filter handler recovers its filter expression as a region distinct
    /// from the handler body.
    #[test]
    fn filter_expression_is_recovered_separately_from_the_handler_body() {
        // Filter at 2 decides; handler body at 3.
        let graph = graph_of(5, &[(0, 1), (1, 4), (3, 4)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(5, &[1]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Filter(HandlerFilter {
                    entry: NodeId::new(2),
                    blocks: set_of(5, &[2]),
                }),
                entry: NodeId::new(3),
                blocks: set_of(5, &[3]),
            }],
        }];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(protected)) = top.get(1) else {
            panic!("expected a try, got {top:?}");
        };
        let handler = protected.handlers.first().expect("one handler");
        assert_eq!(handler.filter, Some(Region::Block(NodeId::new(2))));
        assert_eq!(handler.body, Region::Block(NodeId::new(3)));
    }

    /// A region covering another recovers with the inner `Region::Try` inside
    /// the outer body.
    #[test]
    fn nested_protected_regions_nest_in_the_tree() {
        // Outer try covers 1..3, inner try covers 2 alone.
        let graph = graph_of(7, &[(0, 1), (1, 2), (2, 3), (3, 6), (4, 6), (5, 6)]);
        let regions = vec![
            ProtectedRegion {
                entry: NodeId::new(1),
                blocks: set_of(7, &[1, 2, 3]),
                handlers: vec![ProtectedHandler {
                    kind: ProtectedHandlerKind::Catch,
                    entry: NodeId::new(4),
                    blocks: set_of(7, &[4]),
                }],
            },
            ProtectedRegion {
                entry: NodeId::new(2),
                blocks: set_of(7, &[2]),
                handlers: vec![ProtectedHandler {
                    kind: ProtectedHandlerKind::Finally,
                    entry: NodeId::new(5),
                    blocks: set_of(7, &[5]),
                }],
            },
        ];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 2);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(outer)) = top.get(1) else {
            panic!("expected the outer try, got {top:?}");
        };
        assert_eq!(outer.entry, NodeId::new(1));
        let mut nested = 0usize;
        count_tries(&outer.body, &mut nested);
        assert_eq!(nested, 1, "the inner try is inside the outer body");
    }

    /// A block reached by escaping a protected region is labelled and placed
    /// outside the try, not dragged into it.
    #[test]
    fn leaving_a_protected_region_early_does_not_drag_code_inside_it() {
        // Block 2, inside the try, jumps to 5 while the region's follow is 4.
        // 5 belongs to neither, so it must not be placed within the try.
        let graph = graph_of(7, &[(0, 1), (1, 2), (1, 4), (2, 5), (4, 6), (5, 6)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(7, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Finally,
                entry: NodeId::new(3),
                blocks: set_of(7, &[3]),
            }],
        }];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);

        let Region::Seq(top) = &recovered.root else {
            panic!("expected a sequence, got {:?}", recovered.root);
        };
        let Some(Region::Try(protected)) = top.get(1) else {
            panic!("expected a try, got {top:?}");
        };
        let mut inside = Vec::new();
        protected.body.for_each_block(&mut |node| inside.push(node));
        assert!(
            !inside.contains(&NodeId::new(5)),
            "block 5 escaped the region and must not be inside it: {inside:?}"
        );
        assert_eq!(protected.follow, Some(NodeId::new(4)));
        assert!(
            recovered.labels.contains(&NodeId::new(5)),
            "the escape target needs a label"
        );
    }

    /// A protected region wrapping a loop recovers both the try and the loop
    /// without gotos.
    #[test]
    fn a_protected_region_around_a_loop_keeps_both() {
        let graph = graph_of(6, &[(0, 1), (1, 2), (1, 5), (2, 1), (3, 4), (4, 5)]);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(1),
            blocks: set_of(6, &[1, 2]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(3),
                blocks: set_of(6, &[3, 4]),
            }],
        }];

        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &regions),
        );
        assert_total_protected(&graph, NodeId::new(0), &regions, &recovered);
        assert_eq!(recovered.metrics.tries, 1);
        assert_eq!(recovered.metrics.loops, 1);
        assert_eq!(recovered.metrics.gotos, 0);
    }

    /// A short-circuit `&&` recovers as one merged `All` condition when the
    /// middle block may move, placing once the shared arm that would otherwise
    /// be placed twice.
    #[test]
    fn conjunction_becomes_one_condition() {
        // `if (a && b) X; else Y;` - 0 tests a, 1 tests b, and both false arms
        // reach 2. Without merging, 2 has two predecessors and is placed in
        // each arm: it runs straight on to the join 4, so the second arrival
        // repeats it.
        let graph = graph_of(
            7,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 2),
                (2, 4),
                (3, 4),
                (4, 5),
                (4, 6),
            ],
        );
        let entry = NodeId::new(0);

        let unmerged = structure_merging(&graph, entry, &[]);
        assert_total(&graph, entry, &unmerged);
        assert_eq!(
            unmerged.metrics.replicated_tails, 1,
            "the shared false arm is placed twice when the condition cannot merge"
        );

        let merged = structure_merging(&graph, entry, &[1]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0);
        assert_eq!(merged.metrics.replicated_tails, 0, "merging places it once");
        assert_eq!(merged.metrics.merged_conditions, 1);

        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::All(vec![
                Predicate::Test {
                    block: NodeId::new(0),
                    negated: false,
                },
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: false,
                },
            ])
        );
        assert_eq!(branch.then_branch, Region::Block(NodeId::new(3)));
        assert_eq!(branch.else_branch, Some(Region::Block(NodeId::new(2))));
    }

    /// A block that does work before its test still merges into the condition
    /// when it is condition-expressible, since short-circuit evaluation runs
    /// the work on the same paths.
    #[test]
    fn a_block_that_computes_still_joins_the_condition_it_tests_in() {
        // The same disjunction, but 1 does work before its test. Short-circuit
        // evaluation runs that work exactly when the edge 0 -> 1 did, so the
        // merge stays sound and the alternative - placing 1 inside an arm - is
        // what leaves 3 reached twice and costs the goto.
        let graph = graph_of(5, &[(0, 3), (0, 1), (1, 3), (1, 2), (2, 4), (3, 4)]);
        let entry = NodeId::new(0);

        let merged = structure_expressing(&graph, entry, &[1]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0);
        assert_eq!(merged.metrics.merged_conditions, 1);

        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Any(vec![
                Predicate::Test {
                    block: NodeId::new(0),
                    negated: false,
                },
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: false,
                },
            ])
        );
    }

    /// A test every way into which is the condition so far deciding one way
    /// joins it, however many of the condition's blocks lead there:
    /// `if ((a && b) || c) X; else Y;` reaches `c` from both `a` and `b`
    /// failing, as which's `path_search` computes `(show_tilde || skip_tilde)
    /// && strncmp(...) == 0`. Taken for a block of one predecessor alone, the
    /// condition stopped at `a && b` and `c` reached `X` by goto.
    #[test]
    fn a_test_the_condition_reaches_from_several_of_its_blocks_joins_it() {
        // 0 tests a, 1 tests b, 2 tests c; 3 is X, 4 is Y, 5 joins.
        let graph = graph_of(
            6,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 2),
                (2, 3),
                (2, 4),
                (3, 5),
                (4, 5),
            ],
        );
        let entry = NodeId::new(0);
        let merged = structure_expressing(&graph, entry, &[1, 2]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0, "{:?}", merged.root);
        assert_eq!(merged.metrics.merged_conditions, 1);
        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        let test = |block: usize, negated: bool| Predicate::Test {
            block: NodeId::new(block),
            negated,
        };
        assert_eq!(
            branch.predicate,
            Predicate::Any(vec![
                Predicate::All(vec![test(0, false), test(1, false)]),
                test(2, false)
            ])
        );
        assert_eq!(branch.then_branch, Region::Block(NodeId::new(3)));
        assert_eq!(branch.else_branch, Some(Region::Block(NodeId::new(4))));
    }

    /// A block that runs statements, reached from several of the condition's
    /// blocks, stays a nested `if` where its shared arm is where the
    /// conditional rejoins: `if (a || b) { x = f(); if (x) X; }` costs no goto
    /// nested, and folded it would only put `x = f()` in the condition. A
    /// block that only tests joins the condition there all the same.
    #[test]
    fn a_block_that_computes_nests_where_its_shared_arm_is_the_join() {
        // 0 tests a, 1 tests b, both reaching 2 when they hold; 2 tests c,
        // 3 is X, 4 the join.
        let graph = graph_of(5, &[(0, 2), (0, 1), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]);
        let entry = NodeId::new(0);
        let nested = structure_expressing(&graph, entry, &[1, 2]);
        assert_total(&graph, entry, &nested);
        assert_eq!(nested.metrics.gotos, 0);
        let Region::Seq(top) = &nested.root else {
            panic!("expected a sequence, got {:?}", nested.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Any(vec![
                Predicate::Test {
                    block: NodeId::new(0),
                    negated: false,
                },
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: false,
                },
            ]),
            "2 nests: {:?}",
            nested.root
        );
        let merged = structure_merging(&graph, entry, &[1, 2]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0);
        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert!(
            matches!(&branch.predicate, Predicate::All(parts) if parts.len() == 2),
            "a block that only tests joins: {:?}",
            branch.predicate
        );
    }

    /// A block in neither set is not folded, keeping its statements where they
    /// ran.
    #[test]
    fn a_block_neither_moveable_nor_expressible_keeps_its_own_place() {
        // With 1 in neither set the merge must not happen: 1 is a block with
        // effects the recovery cannot restate, so it stays where it ran.
        let graph = graph_of(5, &[(0, 3), (0, 1), (1, 3), (1, 2), (2, 4), (3, 4)]);
        let entry = NodeId::new(0);

        let plain = structure_merging(&graph, entry, &[]);
        assert_total(&graph, entry, &plain);
        assert_eq!(plain.metrics.merged_conditions, 0);
    }

    /// A short-circuit `||` recovers as one merged `Any` condition.
    #[test]
    fn disjunction_becomes_one_condition() {
        // `if (a || b) X; else Y;` - 0's taken arm and 1's taken arm both
        // reach 3.
        let graph = graph_of(5, &[(0, 3), (0, 1), (1, 3), (1, 2), (2, 4), (3, 4)]);
        let entry = NodeId::new(0);

        let merged = structure_merging(&graph, entry, &[1]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0);

        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate,
            Predicate::Any(vec![
                Predicate::Test {
                    block: NodeId::new(0),
                    negated: false,
                },
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: false,
                },
            ])
        );
    }

    /// A chain of `&&` tests flattens into a single `All` condition rather than
    /// nesting one fold per test.
    #[test]
    fn a_chain_of_tests_collapses_into_one_condition() {
        // `if (a && b && c)`: 1 and 2 both test and both fall to 4.
        let graph = graph_of(
            6,
            &[
                (0, 1),
                (0, 4),
                (1, 2),
                (1, 4),
                (2, 3),
                (2, 4),
                (3, 5),
                (4, 5),
            ],
        );
        let entry = NodeId::new(0);

        let merged = structure_merging(&graph, entry, &[1, 2]);
        assert_total(&graph, entry, &merged);
        assert_eq!(merged.metrics.gotos, 0);

        let Region::Seq(top) = &merged.root else {
            panic!("expected a sequence, got {:?}", merged.root);
        };
        let Some(Region::If(branch)) = top.first() else {
            panic!("expected an if, got {top:?}");
        };
        assert_eq!(
            branch.predicate.tests(),
            3,
            "three tests, flattened into one condition: {:?}",
            branch.predicate
        );
        assert!(
            matches!(branch.predicate, Predicate::All(ref parts) if parts.len() == 3),
            "a chain flattens rather than nesting: {:?}",
            branch.predicate
        );
    }

    /// A block not declared condition-only is never folded, at the cost of the
    /// second copy of its shared arm.
    #[test]
    fn a_block_with_side_effects_is_never_folded_into_a_condition() {
        // The same graph as the conjunction, but block 1 is not declared
        // condition-only, so folding it would move whatever it does inside a
        // condition that may not evaluate it.
        let graph = graph_of(
            7,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 2),
                (2, 4),
                (3, 4),
                (4, 5),
                (4, 6),
            ],
        );
        let entry = NodeId::new(0);

        let recovered = structure_merging(&graph, entry, &[]);
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.merged_conditions, 0);
        assert_eq!(
            recovered.metrics.replicated_tails, 1,
            "the copy is the price of not moving a block with effects"
        );
    }

    /// A condition-only test block with a second predecessor is not folded,
    /// since folding it would delete the other path into it.
    #[test]
    fn a_shared_test_block_is_never_folded_into_a_condition() {
        // Block 2 tests and is condition-only, but block 3 also reaches it.
        // Folding it into 0's condition would delete the path through 3.
        let graph = graph_of(
            6,
            &[
                (0, 2),
                (0, 1),
                (3, 2),
                (2, 4),
                (2, 1),
                (1, 5),
                (4, 5),
                (0, 3),
            ],
        );
        let entry = NodeId::new(0);
        let recovered = structure_merging(&graph, entry, &[2]);
        assert_total(&graph, entry, &recovered);
        assert_eq!(
            recovered.metrics.merged_conditions, 0,
            "a block with two predecessors cannot move into a condition"
        );
    }

    /// A loop header is never folded into a condition, so the loop survives.
    #[test]
    fn a_loop_header_is_never_folded_into_a_condition() {
        // Block 1 is a loop header that also tests and is condition-only.
        // Folding it would lose the loop.
        let graph = graph_of(5, &[(0, 1), (0, 3), (1, 2), (1, 3), (2, 1), (3, 4)]);
        let entry = NodeId::new(0);
        let recovered = structure_merging(&graph, entry, &[1]);
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.loops, 1, "the loop survives");
        assert_eq!(recovered.metrics.merged_conditions, 0);
    }

    /// Negating a merged condition applies De Morgan over its parts and
    /// round-trips under a second negation.
    #[test]
    fn negating_a_merged_condition_distributes_over_its_parts() {
        let predicate = Predicate::All(vec![
            Predicate::Test {
                block: NodeId::new(0),
                negated: false,
            },
            Predicate::Test {
                block: NodeId::new(1),
                negated: true,
            },
        ]);
        assert_eq!(
            predicate.negate(),
            Predicate::Any(vec![
                Predicate::Test {
                    block: NodeId::new(0),
                    negated: true,
                },
                Predicate::Test {
                    block: NodeId::new(1),
                    negated: false,
                },
            ])
        );
        assert_eq!(predicate.negate().negate(), predicate);
    }

    /// A shared block with no successors is duplicated rather than reached by a
    /// goto to a bare return, and needs no label.
    #[test]
    fn a_shared_return_is_repeated_rather_than_jumped_to() {
        // Two paths end at the same `return` at 3, and the function also ends
        // at 4, so no single block post-dominates the branch at 0 and 3 cannot
        // become its join. A goto to a bare return reads worse than the return.
        let graph = graph_of(5, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 3)]);
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "no goto to a return");
        assert_eq!(recovered.metrics.replicated_tails, 1);
        assert!(
            recovered.labels.is_empty(),
            "a replicated tail needs no label"
        );
    }

    /// A shared block that runs straight into a return is that return's tail,
    /// and is duplicated with it rather than jumped to.
    #[test]
    fn a_shared_straight_tail_is_repeated_rather_than_jumped_to() {
        // The same shape, except 3 flows on to the return at 5: both repeat.
        let graph = graph_of(6, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (3, 5)]);
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.replicated_tails, 2);
        assert_eq!(recovered.metrics.gotos, 0);
        assert!(recovered.labels.is_empty(), "{:?}", recovered.labels);
    }

    /// **A tail several arms run on to is the conditional's join.** A
    /// `do`/`while` whose body starts at `1` and tests at `6`: `1` branches
    /// to `2` or `3`, `3` branches to `4`, and both `2` and `3` run on to the
    /// tail at `5`, which falls into the test. The fourth path, through `4`,
    /// comes round to the test without the tail, so nothing in the loop
    /// post-dominates `1` but the test itself: the tail is placed inside the
    /// first arm that reaches it and the other arm jumps to it. `which`'s
    /// `path_search` prints the name it found and frees it that way, with
    /// three gotos. Told the tail it jumped to, the recovery takes it as the
    /// join of `1`, the arms fall into it, and the path that skips it is the
    /// `continue` it already is.
    ///
    /// The control lets the entry reach the tail as well: `1` no longer
    /// dominates it, and the loop that holds it does not hold the entry, so
    /// placing it after either branch would carry a block of the loop out of
    /// it - the jump stays.
    #[test]
    fn a_tail_several_arms_run_on_to_is_the_conditionals_join() {
        let edges = [
            (1, 2),
            (1, 3),
            (2, 5),
            (3, 5),
            (3, 4),
            (4, 6),
            (5, 6),
            (6, 1),
            (6, 7),
        ];
        let joined: Vec<(usize, usize)> = [(0, 1)].into_iter().chain(edges).collect();
        let graph = graph_of(8, &joined);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert!(
            goto_targets(&recovered.root).is_empty(),
            "{:#?}",
            recovered.root
        );

        assert_eq!(recovered.metrics.split_blocks, 0, "{:#?}", recovered.root);

        // Without the rule the join is reached by a goto, or by a split that
        // copies what it dominates.
        let reached: Vec<(usize, usize)> = [(0, 1), (0, 5)].into_iter().chain(edges).collect();
        let control = graph_of(8, &reached);
        let recovered = structure(
            &control,
            NodeId::new(0),
            &StructureOptions::bare(control.node_count(), &[]),
        );
        assert_total(&control, NodeId::new(0), &recovered);
        assert!(
            !goto_targets(&recovered.root).is_empty() || recovered.metrics.split_blocks > 0,
            "{:#?}",
            recovered.root
        );
    }

    /// A shared block that flows on into a branch is reached by goto rather
    /// than duplicated, since repeating it would repeat the branch and
    /// everything after it too.
    ///
    /// Of the replication rule alone ([`recover`]): a whole recovery, told
    /// the tail it jumped to, takes `3` as the join of the branch at `0` and
    /// jumps nowhere, which leaves the rule nothing to decide.
    #[test]
    fn a_shared_block_that_continues_is_never_repeated() {
        // 3 flows on to 5, which branches to 6 and 7: no tail.
        let graph = graph_of(
            8,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 4),
                (2, 3),
                (3, 5),
                (5, 6),
                (5, 7),
            ],
        );
        let entry = NodeId::new(0);
        let count = graph.node_count();
        let recovered = recover(
            &graph,
            entry,
            &Inputs {
                statements: Statements::default(),
                regions: Vec::new(),
                condition_only: BlockSet::new(count),
                condition_reachability: BlockSet::new(count),
                condition_expressible: BlockSet::new(count),
                condition_operand: BlockSet::new(count),
                case_tests: BTreeMap::new(),
                no_return: BlockSet::new(count),
                edge_blocks: BlockSet::new(count),
                max_depth: DEFAULT_MAX_DEPTH,
                preferred_joins: BlockSet::new(count),
                fold_statements: true,
                fall_out: false,
                switch_tails: false,
                route: false,
                repeated_tests: BTreeMap::new(),
                threaded_tests: BTreeMap::new(),
            },
        )
        .structured;
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.replicated_tails, 0);
        assert_eq!(recovered.metrics.gotos, 1);
    }

    /// **A tail the arms of a branch in a loop run on to is their join, even
    /// where they meet again only past the loop.** The `while` at `1` holds a
    /// branch at `2` whose arms run on to the tail `6`, except one path
    /// through `3` that continues the loop from `5`; `6` tests whether to go
    /// round again. Over every path the arms meet only at the loop's exit
    /// `7`, so the walk placed `6` in one arm and reached it from the other by
    /// goto: `which`'s `path_search` (`0x22d9` on x86-64).
    #[test]
    fn a_tail_is_the_join_of_arms_that_meet_again_only_past_the_loop() {
        let graph = graph_of(
            8,
            &[
                (0, 1),
                (1, 2),
                (1, 7),
                (2, 3),
                (2, 4),
                (3, 6),
                (3, 5),
                (5, 1),
                (4, 6),
                (6, 1),
                (6, 7),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
    }

    /// **A branch inside a region of a loop body joins at the region's end
    /// when its arms meet only past the loop.** As above, with the arm at
    /// `3` a nest of its own: `8` continues the loop from `5` or tests again
    /// at `9`, whose arms reach the tail `6` directly and through `10`. Every
    /// branch of the nest is walked inside the arm the tail `6` bounds, and
    /// over every path its arms meet only at the loop's exit `7`, past that
    /// bound: taken as the join, the exit left the tail reached by goto from
    /// each.
    #[test]
    fn a_branch_in_a_loop_region_joins_at_the_regions_end() {
        let graph = graph_of(
            11,
            &[
                (0, 1),
                (1, 2),
                (1, 7),
                (2, 3),
                (2, 4),
                (3, 6),
                (3, 8),
                (8, 5),
                (8, 9),
                (5, 1),
                (9, 6),
                (9, 10),
                (10, 6),
                (4, 6),
                (6, 1),
                (6, 7),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
    }

    /// **A tail two arms of a chain share, while a third runs past it to the
    /// chain's end, is placed in each arm that reaches it.** `0` branches to
    /// `1` or `2`, `2` to `4` or `5`; `1` and `4` run on to the tail `3`,
    /// which runs straight on to the join `6`, and `5` reaches `6` without
    /// it: dash's `sub_aea8` (on ARM), `if (x == 1) .. else if (x == 2) ..`
    /// with a call both arms make. No branch's join bounds the tail, so the
    /// walk placed it in `1`'s arm and reached it from `4` by goto. Let the
    /// tail branch, and the tail rule, which repeats straight-line code,
    /// repeats nothing: the split on trial takes the tail along with the
    /// code past it, which ends the function, and no jump stays either
    /// ([`SplitGraph::split_join`]). Let that code loop (`8` repeats itself),
    /// and no copy takes the loop: each arm that reaches the tail takes it
    /// and the test `6` past it, and falls out of the branch to the loop,
    /// placed once.
    #[test]
    fn a_tail_two_arms_share_past_a_third_is_placed_in_each() {
        let shared = |tail_branches: bool, looping: bool| {
            let mut edges = vec![
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 4),
                (2, 5),
                (4, 3),
                (3, 6),
                (5, 6),
                (6, 7),
                (6, 8),
            ];
            if tail_branches {
                edges.push((3, 8));
            }
            if looping {
                edges.push((8, 8));
            }
            graph_of(9, &edges)
        };
        let graph = shared(false, false);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert_eq!(
            recovered.metrics.replicated_tails, 1,
            "{:#?}",
            recovered.root
        );

        let branching = shared(true, false);
        let recovered = structure(
            &branching,
            NodeId::new(0),
            &StructureOptions::bare(branching.node_count(), &[]),
        );
        assert_total(&branching, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert!(recovered.metrics.split_blocks > 0, "{:#?}", recovered.root);

        let control = shared(true, true);
        let recovered = structure(
            &control,
            NodeId::new(0),
            &StructureOptions::bare(control.node_count(), &[]),
        );
        assert_total(&control, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert_eq!(placements(&recovered, 8), 1, "{:#?}", recovered.root);
    }

    /// **A tail placed in each arm is offered as the join, and placed once
    /// where it can be.** A `do`/`while` whose body branches at `1` to `2` or
    /// `3`; `2` branches to the tail `4` or to `5`, and `3` runs on to `4`.
    /// The tail and `5` both run on to the test `6`, which is where the arms
    /// rejoin, so the first recovery places `4` in `2`'s arm and again in
    /// `3`'s. Taken as the join of `1`, the tail is placed once, and the path
    /// through `5` that skips it is the `continue` it already is.
    #[test]
    fn a_tail_placed_in_each_arm_is_offered_as_the_join() {
        let graph = graph_of(
            8,
            &[
                (0, 1),
                (1, 2),
                (1, 3),
                (2, 4),
                (2, 5),
                (3, 4),
                (4, 6),
                (5, 6),
                (6, 1),
                (6, 7),
            ],
        );
        let entry = NodeId::new(0);
        let options = StructureOptions::bare(graph.node_count(), &[]);
        let first = recover(
            &graph,
            entry,
            &Inputs {
                statements: Statements::default(),
                regions: Vec::new(),
                condition_only: options.condition_only.clone(),
                condition_reachability: options.condition_reachability.clone(),
                condition_expressible: options.condition_expressible.clone(),
                condition_operand: options.condition_operand.clone(),
                case_tests: BTreeMap::new(),
                no_return: options.no_return.clone(),
                edge_blocks: options.edge_blocks.clone(),
                max_depth: options.max_depth,
                preferred_joins: BlockSet::new(graph.node_count()),
                fold_statements: true,
                fall_out: false,
                switch_tails: false,
                route: false,
                repeated_tests: BTreeMap::new(),
                threaded_tests: BTreeMap::new(),
            },
        );
        assert_eq!(
            first.structured.metrics.replicated_tails, 1,
            "{:#?}",
            first.structured.root
        );
        assert_eq!(
            first.rejoining_tails,
            BTreeSet::from([NodeId::new(4)]),
            "{:#?}",
            first.structured.root
        );

        let recovered = structure(&graph, entry, &options);
        assert_total(&graph, entry, &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert_eq!(
            recovered.metrics.replicated_tails, 0,
            "{:#?}",
            recovered.root
        );
    }

    /// Whether some conditional in `region` tests exactly `blocks`, folded
    /// into one condition.
    fn folds(region: &Region, blocks: &[usize]) -> bool {
        match region {
            Region::If(branch) => {
                let mut tested = Vec::new();
                branch
                    .predicate
                    .for_each_block(&mut |block| tested.push(block.index()));
                tested == blocks
                    || folds(&branch.then_branch, blocks)
                    || branch
                        .else_branch
                        .as_ref()
                        .is_some_and(|other| folds(other, blocks))
            }
            Region::Seq(parts) => parts.iter().any(|part| folds(part, blocks)),
            Region::Loop(inner) => folds(&inner.body, blocks),
            Region::Switch(switch) => switch.cases.iter().any(|case| folds(&case.body, blocks)),
            Region::Try(inner) => {
                folds(&inner.body, blocks)
                    || inner
                        .handlers
                        .iter()
                        .any(|handler| folds(&handler.body, blocks))
            }
            _ => false,
        }
    }

    /// **Tests past a loop's exit that the walk places in its body fold
    /// among themselves.** The `while` at `1` leaves to the return `9` when
    /// its test fails and, from its body `2`, to `3`, whose code ends the
    /// function on its own, so the walk places it in the body. There `3` and
    /// `4` are `a || b` around `5`, and both lie outside the loop, as the
    /// head of the condition does: `which`'s `process_alias`, testing for a
    /// double or a single quote. The fold was refused for a block outside
    /// the loop whatever side its head was on.
    #[test]
    fn tests_past_a_loop_exit_fold_among_themselves() {
        let graph = graph_of(
            10,
            &[
                (0, 1),
                (1, 2),
                (1, 9),
                (2, 1),
                (2, 3),
                (3, 5),
                (3, 4),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7),
                (7, 8),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert!(folds(&recovered.root, &[3, 4]), "{:#?}", recovered.root);
        assert_eq!(
            recovered.metrics.replicated_tails, 0,
            "{:#?}",
            recovered.root
        );
    }

    /// **A loop both arms of a branch run into follows the branch.** `1`
    /// branches to `2`, which runs into the loop at `4`, or to `3`, which
    /// returns through `5` or runs into the loop too; the loop leaves to the
    /// same return `5`. The arms meet only at the return, which ends the
    /// function, so the walk placed the loop in `2`'s arm and jumped into it
    /// from `3`'s: a variadic `v_fsum` (`pe/x86_64`) whose sum loop starts
    /// after an odd element or none. Taken as the join, the loop follows the
    /// branch, and `3`'s arm carries its own copy of the return.
    #[test]
    fn a_loop_both_arms_of_a_branch_run_into_follows_the_branch() {
        let graph = graph_of(
            7,
            &[
                (0, 6),
                (0, 1),
                (1, 2),
                (1, 3),
                (2, 4),
                (3, 5),
                (3, 4),
                (4, 4),
                (4, 5),
            ],
        );
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        let Region::Seq(top) = &recovered.root else {
            panic!("a sequence: {:#?}", recovered.root);
        };
        assert!(
            top.iter().any(|part| matches!(part, Region::Loop(_))),
            "the loop follows the branch at the top level: {:#?}",
            recovered.root
        );
    }

    /// **A tail that runs on to a loop's exit is placed in each arm, and
    /// breaks.** The `while` at `1` leaves to `7`; its body branches at `2`,
    /// and both arms test at `3` and `4` whether to leave early through `5`,
    /// which runs straight on to `7`, or go round through `6`: dash's
    /// `evalstring` (`0x6ecc` on RISC-V), whose early exit saves its status
    /// on the way out. `5` was placed in one arm and the other jumped to it;
    /// each now carries it, followed by the `break` it ends in. The control
    /// lets `5` branch, and the jump stays.
    #[test]
    fn a_tail_that_runs_on_to_a_loop_exit_is_placed_in_each_arm() {
        let looped = |tail_branches: bool| {
            let mut edges = vec![
                (0, 1),
                (1, 2),
                (1, 7),
                (2, 3),
                (2, 4),
                (3, 5),
                (3, 6),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 1),
            ];
            if tail_branches {
                edges.push((5, 6));
            }
            graph_of(8, &edges)
        };
        let graph = looped(false);
        let recovered = structure(
            &graph,
            NodeId::new(0),
            &StructureOptions::bare(graph.node_count(), &[]),
        );
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert_eq!(
            recovered.metrics.replicated_tails, 1,
            "{:#?}",
            recovered.root
        );

        assert_eq!(recovered.metrics.split_blocks, 0, "{:#?}", recovered.root);

        // Without the rule `5` is reached by a goto, or by a split that copies
        // what it dominates.
        let control = looped(true);
        let recovered = structure(
            &control,
            NodeId::new(0),
            &StructureOptions::bare(control.node_count(), &[]),
        );
        assert_total(&control, NodeId::new(0), &recovered);
        assert!(
            goto_targets(&recovered.root).contains(&NodeId::new(5))
                || recovered.metrics.split_blocks > 0,
            "{:#?}",
            recovered.root
        );
    }

    /// Every `break` or `continue` in `region` that follows a part which
    /// already leaves the sequence it is in, by a transfer of its own or a
    /// conditional both of whose arms end in one.
    fn dead_transfers(region: &Region) -> usize {
        let ends = |part: &Region| -> bool {
            fn ends(part: &Region) -> bool {
                match part {
                    Region::Break(_)
                    | Region::Continue(_)
                    | Region::Goto(_)
                    | Region::SwitchBreak(_) => true,
                    Region::Seq(parts) => parts.last().is_some_and(ends),
                    Region::If(branch) => {
                        ends(&branch.then_branch) && branch.else_branch.as_ref().is_some_and(ends)
                    }
                    _ => false,
                }
            }
            ends(part)
        };
        match region {
            Region::Seq(parts) => {
                let after = parts
                    .windows(2)
                    .filter(|pair| {
                        ends(&pair[0])
                            && matches!(
                                pair[1],
                                Region::Break(_) | Region::Continue(_) | Region::SwitchBreak(_)
                            )
                    })
                    .count();
                after + parts.iter().map(dead_transfers).sum::<usize>()
            }
            Region::If(branch) => {
                dead_transfers(&branch.then_branch)
                    + branch.else_branch.as_ref().map_or(0, dead_transfers)
            }
            Region::Loop(inner) => dead_transfers(&inner.body),
            _ => 0,
        }
    }

    /// **A conditional both of whose arms leave is followed by nothing.**
    /// `which`'s `find_command_in_path` (`0x1c5f` on x86-64), as its lowering
    /// states it: the endless loop at `24` ends in a conditional at `18`
    /// whose arms each go round (`23`, `20`), one of them after leaving to
    /// the loop's exit `27` at `21`. The arms meet only at that exit, and the
    /// walk resumed there after the conditional, printing a `break` no path
    /// reaches: the body ended `continue; break;`.
    #[test]
    fn a_conditional_both_of_whose_arms_leave_is_followed_by_nothing() {
        let count = 28;
        let graph = graph_of(
            count,
            &[
                (0, 1),
                (0, 2),
                (1, 24),
                (2, 4),
                (2, 3),
                (3, 4),
                (4, 8),
                (4, 5),
                (5, 8),
                (5, 6),
                (6, 8),
                (6, 7),
                (7, 9),
                (8, 9),
                (9, 24),
                (10, 12),
                (10, 11),
                (11, 13),
                (12, 13),
                (13, 27),
                (13, 14),
                (14, 18),
                (14, 15),
                (15, 16),
                (15, 17),
                (16, 18),
                (17, 24),
                (18, 21),
                (18, 19),
                (19, 21),
                (19, 20),
                (20, 24),
                (21, 23),
                (21, 22),
                (22, 23),
                (22, 27),
                (23, 24),
                (24, 25),
                (24, 26),
                (25, 27),
                (26, 27),
                (26, 10),
            ],
        );
        let test = |value, entered| CaseTest {
            value,
            relation: CaseRelation::Equal,
            constant: 0,
            holds: 0,
            entered,
        };
        let options = StructureOptions {
            statements: Vec::new(),
            regions: &[],
            condition_only: set_of(count, &[4, 5, 6, 10, 14, 18, 19, 22, 24]),
            condition_reachability: BlockSet::new(count),
            condition_expressible: set_of(
                count,
                &[0, 2, 4, 5, 6, 10, 13, 14, 15, 18, 19, 21, 22, 24, 26],
            ),
            condition_operand: set_of(count, &[4, 5, 6, 10, 14, 18, 19, 22, 24]),
            case_tests: [
                (0, test(3, false)),
                (13, test(8, true)),
                (21, test(13, false)),
                (22, test(14, false)),
                (24, test(6, true)),
            ]
            .into_iter()
            .map(|(node, test)| (NodeId::new(node), test))
            .collect(),
            no_return: BlockSet::new(count),
            edge_blocks: BlockSet::new(count),
            max_depth: DEFAULT_MAX_DEPTH,
            polar: BlockSet::new(count),
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        let recovered = structure(&graph, NodeId::new(0), &options);
        assert_total(&graph, NodeId::new(0), &recovered);
        assert_eq!(dead_transfers(&recovered.root), 0, "{:#?}", recovered.root);
    }

    /// How a transfer leaves the part of a tree [`Run`] was running.
    enum Flow {
        /// On to whatever follows the part.
        Next,
        /// Out of the loop with this header.
        Break(NodeId),
        /// Out of the switch with this head.
        Leave(NodeId),
        /// To the next test of the loop with this header.
        Continue(NodeId),
        /// The function ended, or the run reached its length.
        Stop,
    }

    /// A recovered tree run as a printer writes it, recording each block
    /// whose statements run, in the order they run, with every branch taking
    /// the successor `choose` names for that block's `n`th test.
    ///
    /// A table `switch` takes the arm whose selectors name the successor its
    /// head takes, and runs on through each arm that falls through until one
    /// leaves.
    ///
    /// A block's statements run where the tree places it, and where its test
    /// is evaluated: a conditional prints its head's statements before it
    /// and a merged condition each later block's inside its operand. A
    /// `do`/`while`'s latch is the one exception, placed at the end of the
    /// body while the loop's test only reads its branch. A block folded into
    /// a condition has no statements of its own and is recorded nowhere; its
    /// tests still decide which blocks run after it. The independent half is
    /// [`walk_graph`]: two recoveries that agree with the graph on every
    /// choice run the same program. Every branch is free, so a recovery that
    /// reads one test as decided by another
    /// ([`StructureOptions::case_tests`]) is outside what a run checks.
    struct Run<'g> {
        /// The graph the tree was recovered from.
        graph: &'g DirectedGraph<'static, (), ()>,
        /// The successor, counted modulo the block's successors, a block's
        /// `n`th test takes.
        choose: &'g dyn Fn(NodeId, usize) -> usize,
        /// The blocks folded into a condition: those of the condition-only
        /// set that branch, the only ones a printer folds.
        folded: BTreeSet<NodeId>,
        /// Tests evaluated so far, per block.
        tests: BTreeMap<NodeId, usize>,
        /// The blocks run, in order.
        trace: Vec<NodeId>,
        /// The longest trace recorded.
        limit: usize,
        /// Tests evaluated and loop iterations begun so far.
        evaluated: usize,
        /// Whether the run reached a block that ends the function.
        ended: bool,
    }

    impl Run<'_> {
        /// Records that `node`'s statements run; `false` once the run is as
        /// long as it may get.
        fn record(&mut self, node: NodeId) -> bool {
            if self.trace.len() >= self.limit {
                return false;
            }
            self.trace.push(node);
            true
        }

        /// Runs `region`, or returns `None` for a part this run does not
        /// model: a `goto`, a `switch` or a protected region.
        fn region(&mut self, region: &Region) -> Option<Flow> {
            match region {
                Region::Empty => Some(Flow::Next),
                Region::Block(node) => {
                    if !self.folded.contains(node) && !self.record(*node) {
                        return Some(Flow::Stop);
                    }
                    // A block that ends the function ends the run there.
                    if self.graph.successors(*node).next().is_none() {
                        self.ended = true;
                        return Some(Flow::Stop);
                    }
                    Some(Flow::Next)
                }
                Region::Seq(parts) => {
                    for part in parts {
                        match self.region(part)? {
                            Flow::Next => {}
                            other => return Some(other),
                        }
                    }
                    Some(Flow::Next)
                }
                Region::If(branch) => match self.holds(&branch.predicate, true) {
                    None => Some(Flow::Stop),
                    Some(true) => self.region(&branch.then_branch),
                    Some(false) => match &branch.else_branch {
                        Some(other) => self.region(other),
                        None => Some(Flow::Next),
                    },
                },
                Region::Loop(inner) => loop {
                    // An iteration that runs and tests nothing never reaches
                    // the run's length; it is stopped as long.
                    self.evaluated = self.evaluated.saturating_add(1);
                    if self.evaluated > self.limit.saturating_mul(4) {
                        return Some(Flow::Stop);
                    }
                    if inner.kind == StructuredLoop::While {
                        match inner.predicate.as_ref().map(|test| self.holds(test, true)) {
                            Some(None) => return Some(Flow::Stop),
                            Some(Some(false)) => return Some(Flow::Next),
                            _ => {}
                        }
                    }
                    match self.region(&inner.body)? {
                        Flow::Break(header) if header == inner.header => return Some(Flow::Next),
                        Flow::Next => {}
                        Flow::Continue(header) if header == inner.header => {}
                        other => return Some(other),
                    }
                    if inner.kind == StructuredLoop::DoWhile {
                        match inner.predicate.as_ref().map(|test| self.holds(test, false)) {
                            Some(None) => return Some(Flow::Stop),
                            Some(Some(false)) => return Some(Flow::Next),
                            _ => {}
                        }
                    }
                },
                Region::Switch(switch) => {
                    // A table's dispatch takes the arm whose selectors name
                    // the successor taken, and runs on through each arm that
                    // falls through; none taken runs on past the switch.
                    let head = switch.head;
                    if !self.folded.contains(&head) && !self.record(head) {
                        return Some(Flow::Stop);
                    }
                    let count = self.tests.entry(head).or_default();
                    let choice = (self.choose)(head, *count);
                    *count = count.saturating_add(1);
                    let arms = self.graph.successors(head).count().max(1);
                    let taken = choice % arms;
                    let Some(first) = switch
                        .cases
                        .iter()
                        .position(|case| case.selectors.contains(&taken))
                    else {
                        return Some(Flow::Next);
                    };
                    for case in switch.cases.iter().skip(first) {
                        match self.region(&case.body)? {
                            Flow::Next if case.falls_through => {}
                            Flow::Next => return Some(Flow::Next),
                            Flow::Leave(target) if target == head => return Some(Flow::Next),
                            other => return Some(other),
                        }
                    }
                    Some(Flow::Next)
                }
                Region::Break(header) => Some(Flow::Break(*header)),
                Region::Continue(header) => Some(Flow::Continue(*header)),
                Region::SwitchBreak(head) => Some(Flow::Leave(*head)),
                Region::Goto(_) | Region::Try(_) => None,
            }
        }

        /// Whether `predicate` holds, evaluating its tests in short-circuit
        /// order, each running its block's statements where `runs` says so;
        /// `None` once the run is as long as it may get.
        fn holds(&mut self, predicate: &Predicate, runs: bool) -> Option<bool> {
            match predicate {
                Predicate::Test { block, negated } | Predicate::Hoisted { block, negated, .. } => {
                    if runs && !self.folded.contains(block) && !self.record(*block) {
                        return None;
                    }
                    // A tree that tests without running anything never
                    // reaches the run's length; it is stopped as long.
                    self.evaluated = self.evaluated.saturating_add(1);
                    if self.evaluated > self.limit.saturating_mul(4) {
                        return None;
                    }
                    // The test holds where the branch reaches the block its
                    // first successor names, by whichever edge.
                    let count = self.tests.entry(*block).or_default();
                    let choice = (self.choose)(*block, *count);
                    *count = count.saturating_add(1);
                    let successors: Vec<NodeId> = self.graph.successors(*block).collect();
                    let taken = successors.get(choice % successors.len().max(1));
                    Some((taken == successors.first()) != *negated)
                }
                Predicate::All(parts) => {
                    for part in parts {
                        if !self.holds(part, runs)? {
                            return Some(false);
                        }
                    }
                    Some(true)
                }
                Predicate::Any(parts) => {
                    for part in parts {
                        if self.holds(part, runs)? {
                            return Some(true);
                        }
                    }
                    Some(false)
                }
            }
        }
    }

    /// The blocks a run of `graph` from `entry` passes, at most `limit` of
    /// them and none of the `folded` ones, with every branch taking the
    /// successor `choose` names for that block's `n`th test; and whether the
    /// run reached a block that ends the function.
    fn walk_graph(
        graph: &DirectedGraph<'static, (), ()>,
        entry: NodeId,
        choose: &dyn Fn(NodeId, usize) -> usize,
        folded: &BTreeSet<NodeId>,
        limit: usize,
    ) -> (Vec<NodeId>, bool) {
        let mut trace = Vec::new();
        let mut tests: BTreeMap<NodeId, usize> = BTreeMap::new();
        let mut node = entry;
        // A cycle of folded blocks records nothing; it is stopped as long.
        for _ in 0..limit.saturating_mul(4) {
            if trace.len() >= limit {
                break;
            }
            if !folded.contains(&node) {
                trace.push(node);
            }
            let successors: Vec<NodeId> = graph.successors(node).collect();
            node = match successors.as_slice() {
                [] => return (trace, true),
                [only] => *only,
                _ => {
                    let count = tests.entry(node).or_default();
                    let taken = choose(node, *count);
                    *count = count.saturating_add(1);
                    successors[taken % successors.len()]
                }
            };
        }
        (trace, false)
    }

    /// Whether every part of `region` is one [`Run`] models: no `goto`, no
    /// `switch` over a value rather than a table, and no protected region.
    fn modelled(region: &Region) -> bool {
        match region {
            Region::Switch(switch) => switch
                .cases
                .iter()
                .all(|case| !case.default && case.values.is_empty() && modelled(&case.body)),
            Region::Goto(_) | Region::Try(_) => false,
            Region::Seq(parts) => parts.iter().all(modelled),
            Region::If(branch) => {
                modelled(&branch.then_branch) && branch.else_branch.as_ref().is_none_or(modelled)
            }
            Region::Loop(inner) => modelled(&inner.body),
            Region::Empty
            | Region::Block(_)
            | Region::Break(_)
            | Region::Continue(_)
            | Region::SwitchBreak(_) => true,
        }
    }

    /// How many times `region` places a block straight before a `continue`
    /// to the `do`/`while` that block is the latch of.
    fn latch_copies(region: &Region) -> usize {
        fn count(region: &Region, latches: &mut Vec<(NodeId, NodeId)>) -> usize {
            match region {
                Region::Seq(parts) => {
                    let here = parts
                        .windows(2)
                        .filter(|pair| match pair {
                            [Region::Block(block), Region::Continue(header)] => {
                                latches.contains(&(*header, *block))
                            }
                            _ => false,
                        })
                        .count();
                    here + parts.iter().map(|part| count(part, latches)).sum::<usize>()
                }
                Region::If(branch) => {
                    count(&branch.then_branch, latches)
                        + branch
                            .else_branch
                            .as_ref()
                            .map_or(0, |other| count(other, latches))
                }
                Region::Loop(inner) => {
                    let latch = match (&inner.kind, &inner.predicate) {
                        (StructuredLoop::DoWhile, Some(Predicate::Test { block, .. })) => {
                            Some((inner.header, *block))
                        }
                        _ => None,
                    };
                    latches.extend(latch);
                    let inside = count(&inner.body, latches);
                    if latch.is_some() {
                        latches.pop();
                    }
                    inside
                }
                _ => 0,
            }
        }
        count(region, &mut Vec::new())
    }

    /// Asserts that `recovered` runs the blocks `graph` does under every one
    /// of a spread of branch choices.
    fn assert_runs_as_the_graph(
        graph: &DirectedGraph<'static, (), ()>,
        entry: NodeId,
        condition_only: &BlockSet,
        edge_blocks: &BlockSet,
        recovered: &Structured,
    ) {
        assert!(
            modelled(&recovered.root),
            "a run models the tree: {:#?}",
            recovered.root
        );
        // Neither run records a test folded into a condition, nor a block
        // standing for an edge: what it writes a hoisting test says ahead of
        // itself, on every way out, which the caller states is the edge's
        // statements wherever they are said.
        let folded: BTreeSet<NodeId> = condition_only
            .iter()
            .filter(|node| graph.successors(*node).nth(1).is_some())
            .chain(edge_blocks.iter())
            .collect();
        let edges: Vec<(usize, usize)> = (0..graph.node_count())
            .flat_map(|from| {
                graph
                    .successors(NodeId::new(from))
                    .map(move |to| (from, to.index()))
            })
            .collect();
        let limit = 64;
        for seed in 0..64_usize {
            // Each test's choices spread over its successors as the seed and
            // the visit vary, so loops are both left and gone round.
            let choose = |node: NodeId, count: usize| {
                node.index().wrapping_mul(7)
                    ^ seed.wrapping_mul(13)
                    ^ count.wrapping_mul(5)
                    ^ (seed >> count.min(6))
            };
            let (expected, ended) = walk_graph(graph, entry, &choose, &folded, limit);
            let mut run = Run {
                graph,
                choose: &choose,
                folded: folded.clone(),
                tests: BTreeMap::new(),
                trace: Vec::new(),
                limit,
                evaluated: 0,
                ended: false,
            };
            run.region(&recovered.root);
            // Each run stops at its own bound, so the shorter is a prefix of
            // the longer, and one that ends the function runs no shorter than
            // the other: both that end it run the same blocks.
            let shared = run.trace.len().min(expected.len());
            let ended_early = (run.ended && expected.len() > run.trace.len())
                || (ended && run.trace.len() > expected.len());
            assert!(
                !ended_early,
                "seed {seed}: one run ended the function before the other {edges:?}, \
                 folding {folded:?}: {:?} {:?}\n{:#?}",
                run.trace, expected, recovered.root
            );
            let (ran, walked) = (&run.trace[..shared], &expected[..shared]);
            assert_eq!(
                ran, walked,
                "seed {seed}: the tree runs other blocks than the graph {edges:?}, \
                 folding {folded:?}:\n{:#?}",
                recovered.root
            );
        }
    }

    /// **A path into a `do`/`while`'s latch runs the latch's statements.**
    /// dash's `0xb4b50` on ARM (`sub_4b50`), as its lowering states it: the
    /// inner loop at `2` tests at its latch `4`, whose statements step the
    /// list (`reg_13 = *reg_29; reg_10 = reg_29;`) before its test. Its
    /// tests at `3`, `9` and `10` leave for the latch directly, and a
    /// `continue` there runs the test alone: each arrival places the latch's
    /// statements before its `continue`. Run as a printer writes it, the
    /// tree runs the blocks the graph does. A latch folded into its test
    /// prints nothing an arrival could skip, and one in a protected region
    /// is jumped to rather than copied out of it.
    #[test]
    fn a_path_into_a_latch_that_computes_runs_its_statements() {
        let count = 14;
        let edges = [
            (0, 1),
            (1, 5),
            (1, 2),
            (2, 11),
            (2, 3),
            (3, 9),
            (3, 4),
            (4, 2),
            (4, 5),
            (5, 1),
            (5, 6),
            (6, 8),
            (6, 7),
            (7, 12),
            (7, 8),
            (9, 4),
            (9, 10),
            (10, 4),
            (10, 11),
            (11, 4),
        ];
        let graph = graph_of(count, &edges);
        let entry = NodeId::new(0);
        let latch = NodeId::new(4);
        let inputs = |condition_only: &BlockSet, regions: Vec<ProtectedRegion>| Inputs {
            statements: Statements::default(),
            regions,
            condition_only: condition_only.clone(),
            condition_reachability: set_of(count, &[7]),
            condition_expressible: set_of(count, &[1, 2, 3, 4, 5, 6, 7, 9, 10]),
            condition_operand: set_of(count, &[2]),
            case_tests: BTreeMap::new(),
            no_return: set_of(count, &[12]),
            edge_blocks: BlockSet::new(count),
            max_depth: DEFAULT_MAX_DEPTH,
            preferred_joins: BlockSet::new(count),
            fold_statements: false,
            fall_out: false,
            switch_tails: false,
            route: false,
            repeated_tests: BTreeMap::new(),
            threaded_tests: BTreeMap::new(),
        };
        let placed = |recovered: &Structured| {
            let mut placed = 0_usize;
            recovered.root.for_each_block(&mut |node| {
                placed = placed.saturating_add(usize::from(node == latch));
            });
            placed
        };
        for folded in [set_of(count, &[7]), set_of(count, &[4, 7])] {
            let first = recover(&graph, entry, &inputs(&folded, Vec::new())).structured;
            assert_total(&graph, entry, &first);
            assert_runs_as_the_graph(
                &graph,
                entry,
                &folded,
                &BlockSet::new(graph.node_count()),
                &first,
            );
        }

        // Where nothing else joins the arrivals, the copy is how they reach
        // the test: drawn by `recovery_is_total_on_any_inputs`, the
        // `do`/`while` at `7` tests at `6`, which `3` leaves for from a test
        // folded into the header's condition and `5` after `9`'s statements.
        let small = graph_of(
            10,
            &[
                (0, 8),
                (0, 8),
                (0, 7),
                (1, 2),
                (1, 2),
                (2, 2),
                (2, 3),
                (3, 9),
                (3, 6),
                (4, 8),
                (5, 6),
                (6, 7),
                (6, 8),
                (7, 9),
                (7, 2),
                (7, 2),
                (9, 5),
            ],
        );
        let folded = set_of(10, &[1, 2, 3, 7]);
        let recovered = structure(
            &small,
            entry,
            &StructureOptions {
                condition_only: folded.clone(),
                condition_expressible: folded.clone(),
                ..StructureOptions::bare(10, &[])
            },
        );
        assert_total(&small, entry, &recovered);
        assert_runs_as_the_graph(
            &small,
            entry,
            &folded,
            &BlockSet::new(small.node_count()),
            &recovered,
        );
        assert!(latch_copies(&recovered.root) > 0, "{:#?}", recovered.root);

        // A latch printing more statements than a goto is worth
        // ([`STATEMENTS_PER_GOTO`](inputs::STATEMENTS_PER_GOTO)) is jumped to
        // rather than copied.
        let mut statements = vec![1; 10];
        if let Some(slot) = statements.get_mut(6) {
            *slot = inputs::STATEMENTS_PER_GOTO.saturating_add(1);
        }
        let long = structure(
            &small,
            entry,
            &StructureOptions {
                statements,
                condition_only: folded.clone(),
                condition_expressible: folded.clone(),
                ..StructureOptions::bare(10, &[])
            },
        );
        assert_total(&small, entry, &long);
        assert_runs_as_the_graph(
            &small,
            entry,
            &folded,
            &BlockSet::new(small.node_count()),
            &long,
        );
        assert_eq!(latch_copies(&long.root), 0, "{:#?}", long.root);

        // The loops run inside a `try` whose handler `13` resumes at `5`.
        let mut edges = edges.to_vec();
        edges.push((13, 5));
        let graph = graph_of(count, &edges);
        let regions = vec![ProtectedRegion {
            entry: NodeId::new(2),
            blocks: set_of(count, &[2, 3, 4, 9, 10, 11]),
            handlers: vec![ProtectedHandler {
                kind: ProtectedHandlerKind::Catch,
                entry: NodeId::new(13),
                blocks: set_of(count, &[13]),
            }],
        }];
        let guarded = recover(
            &graph,
            entry,
            &inputs(&set_of(count, &[7]), regions.clone()),
        )
        .structured;
        assert_total_protected(&graph, entry, &regions, &guarded);
        assert_eq!(latch_copies(&guarded.root), 0, "{:#?}", guarded.root);
        assert!(placed(&guarded) <= 2, "{:#?}", guarded.root);
    }

    /// **A path to a `do`/`while`'s header is no `continue`.** Drawn by
    /// `recovery_is_total_on_any_inputs`: the `do`/`while` at `2` tests at
    /// its latch `1`, and `6`, outside it, runs on to the header. Placed
    /// again inside the body, a copy of `6` reaches the header, and C's
    /// `continue` there would run `1`'s test first, which may leave the loop
    /// the graph goes round.
    #[test]
    fn a_path_to_a_do_while_header_is_no_continue() {
        let count = 8;
        let graph = graph_of(
            count,
            &[
                (0, 5),
                (1, 5),
                (1, 2),
                (2, 6),
                (2, 3),
                (3, 0),
                (3, 4),
                (4, 1),
                (4, 5),
                (5, 6),
                (5, 2),
                (6, 2),
                (7, 0),
            ],
        );
        let entry = NodeId::new(0);
        let folded = set_of(count, &[1, 2, 3, 4, 5]);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions {
                condition_only: folded.clone(),
                condition_expressible: folded.clone(),
                ..StructureOptions::bare(count, &[])
            },
        );
        assert_total(&graph, entry, &recovered);
        assert_runs_as_the_graph(
            &graph,
            entry,
            &folded,
            &BlockSet::new(graph.node_count()),
            &recovered,
        );
    }

    /// **A path from a nested loop to a `do`/`while`'s latch jumps to it.**
    /// Drawn by `recovery_is_total_on_any_inputs`: a path from inside a loop
    /// nested in a `do`/`while` reaches the outer loop's latch. A `continue`
    /// of the outer loop from inside the inner one is written as a jump to
    /// the outer header, which skips the test, so the path jumps to the latch
    /// the outer body places, statements and test.
    #[test]
    fn a_path_from_a_nested_loop_to_a_latch_jumps_to_it() {
        let count = 13;
        let graph = graph_of(
            count,
            &[
                (0, 5),
                (0, 1),
                (1, 1),
                (1, 2),
                (2, 11),
                (2, 3),
                (3, 6),
                (3, 4),
                (4, 3),
                (4, 5),
                (5, 11),
                (5, 9),
                (6, 6),
                (6, 8),
                (6, 9),
                (7, 4),
                (7, 0),
                (8, 5),
                (8, 4),
                (9, 4),
                (9, 2),
                (10, 4),
                (10, 1),
                (10, 10),
                (11, 5),
                (11, 1),
                (12, 6),
                (12, 2),
            ],
        );
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions {
                condition_only: set_of(count, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]),
                condition_expressible: set_of(count, &[2, 3, 4, 5, 6, 7, 8, 9, 11]),
                ..StructureOptions::bare(count, &[])
            },
        );
        assert_total(&graph, entry, &recovered);
        assert_eq!(
            outer_do_while_continues(&recovered.root),
            0,
            "{:#?}",
            recovered.root
        );
    }

    /// **A switch arm that runs on into another falls through.** dash's
    /// `openredirect`: `case NTO:` tests `Cflag` and, where it is clear, runs
    /// on into `case NCLOBBER:`'s open (`1` -> `3`), which nothing else but
    /// the dispatch at `0` reaches. The arm at `1` is placed straight before
    /// the one at `3`, though its selector comes after, and falls through
    /// rather than carrying a copy of it; its other path leaves the switch
    /// by `break`.
    #[test]
    fn a_switch_arm_that_runs_on_into_another_falls_through() {
        let count = 6;
        let graph = graph_of(
            count,
            &[
                (0, 3),
                (0, 1),
                (0, 4),
                (1, 2),
                (1, 3),
                (2, 5),
                (3, 5),
                (4, 5),
            ],
        );
        let entry = NodeId::new(0);
        let folded = BlockSet::new(count);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions {
                condition_only: folded.clone(),
                ..StructureOptions::bare(count, &[])
            },
        );
        assert_total(&graph, entry, &recovered);
        assert_runs_as_the_graph(
            &graph,
            entry,
            &folded,
            &BlockSet::new(graph.node_count()),
            &recovered,
        );
        assert_eq!(recovered.metrics.fall_throughs, 1, "{:#?}", recovered.root);
        assert_eq!(
            recovered.metrics.replicated_tails, 0,
            "{:#?}",
            recovered.root
        );
        let switch = first_switch(&recovered.root).expect("the dispatch is a switch");
        let order: Vec<(usize, bool)> = switch
            .cases
            .iter()
            .map(|case| (case.target.index(), case.falls_through))
            .collect();
        assert_eq!(
            order,
            vec![(1, true), (3, false), (4, false)],
            "{:#?}",
            recovered.root
        );
    }

    /// **An arm that falls through keeps the `continue` it ends with.**
    /// Drawn by `recovery_is_total_on_any_inputs`: the switch at `7` ends the
    /// endless loop at `0`, and its arm at `11` runs a loop that leaves for
    /// `0` before the arm runs on into the arm at `10`. A `continue` at the
    /// end of an arm that leaves the switch says nothing at the end of the
    /// loop's body, but one that falls through does not end it there.
    #[test]
    fn an_arm_that_falls_through_keeps_its_continue() {
        let count = 15;
        let graph = graph_of(
            count,
            &[
                (0, 7),
                (1, 14),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 5),
                (3, 4),
                (4, 10),
                (4, 11),
                (6, 12),
                (6, 14),
                (7, 10),
                (7, 6),
                (7, 11),
                (9, 0),
                (9, 2),
                (9, 13),
                (10, 5),
                (11, 1),
                (12, 0),
                (13, 3),
            ],
        );
        let entry = NodeId::new(0);
        let folded = set_of(count, &[1, 2, 3, 4, 7, 9]);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions {
                condition_only: folded.clone(),
                condition_expressible: folded.clone(),
                ..StructureOptions::bare(count, &[])
            },
        );
        assert_total(&graph, entry, &recovered);
        assert_runs_as_the_graph(
            &graph,
            entry,
            &folded,
            &BlockSet::new(graph.node_count()),
            &recovered,
        );
        assert!(recovered.metrics.fall_throughs > 0, "{:#?}", recovered.root);
    }

    /// **A branch in an arm that runs on into the next joins at that arm.**
    /// dash's `openredirect` as its lowering states it (`0xca90` on ARM):
    /// the arm at `1` tests and either runs code that reaches where the
    /// switch resumes (`2` -> `5`) or runs on into the arm at `3`. The
    /// branch's post-dominator is `5`, which branches again, so no copy of
    /// it ends the function; the path to it leaves by the switch's `break`,
    /// and the branch joins at `3`, which the arm falls into.
    #[test]
    fn a_branch_in_a_falling_arm_joins_at_the_next_arm() {
        let count = 8;
        let graph = graph_of(
            count,
            &[
                (0, 3),
                (0, 1),
                (0, 4),
                (1, 2),
                (1, 3),
                (2, 5),
                (3, 5),
                (4, 5),
                (5, 6),
                (5, 7),
                (6, 7),
            ],
        );
        let entry = NodeId::new(0);
        let folded = BlockSet::new(count);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions {
                condition_only: folded.clone(),
                ..StructureOptions::bare(count, &[])
            },
        );
        assert_total(&graph, entry, &recovered);
        assert_runs_as_the_graph(
            &graph,
            entry,
            &folded,
            &BlockSet::new(graph.node_count()),
            &recovered,
        );
        assert_eq!(recovered.metrics.fall_throughs, 1, "{:#?}", recovered.root);
        assert_eq!(
            recovered.metrics.replicated_tails, 0,
            "{:#?}",
            recovered.root
        );
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
    }

    /// **A join a goto names is split.** Both arms of the branch at `0` reach
    /// `4`, which branches itself, so it is no straight-line tail to repeat,
    /// and the arms meet at `7`, which branches too: placed in one arm, the
    /// other jumped to `4`. Split, each arm runs its own copy of what `4`
    /// dominates, `4`, `5` and `6`, and both meet at `7` with no goto.
    #[test]
    fn a_join_a_goto_names_is_split() {
        let count = 11;
        let graph = graph_of(
            count,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 4),
                (2, 4),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7),
                (3, 7),
                (7, 8),
                (7, 9),
                (8, 10),
                (9, 10),
            ],
        );
        let entry = NodeId::new(0);
        let folded = BlockSet::new(count);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions {
                condition_only: folded.clone(),
                ..StructureOptions::bare(count, &[])
            },
        );
        assert_total(&graph, entry, &recovered);
        assert_runs_as_the_graph(
            &graph,
            entry,
            &folded,
            &BlockSet::new(graph.node_count()),
            &recovered,
        );
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert!(recovered.metrics.split_blocks > 0, "{:#?}", recovered.root);
    }

    /// **A split that pays only once another is kept is tried again.** The
    /// exits of the loop at `11`, and `5` ahead of the loop at `6`, reach the
    /// test at `1` that ends the function, to `19` or to `2`, through `14` or
    /// `17`. Split first, `1` needs as many gotos as before and is declined;
    /// split next, `14` pays with its copies taking the code past it that
    /// ends the function; and once they have, `1` pays as well. Tried again
    /// after the split kept, it leaves only the goto the loop at `6` needs,
    /// whose two exits meet at `9`.
    #[test]
    fn a_split_that_pays_only_once_another_is_kept_is_tried_again() {
        let count = 20;
        let graph = graph_of(
            count,
            &[
                (0, 3),
                (1, 19),
                (1, 2),
                (3, 5),
                (3, 4),
                (4, 16),
                (4, 5),
                (5, 14),
                (5, 6),
                (6, 15),
                (6, 7),
                (7, 6),
                (7, 8),
                (8, 9),
                (9, 11),
                (9, 10),
                (11, 17),
                (11, 12),
                (12, 18),
                (12, 13),
                (13, 11),
                (13, 14),
                (14, 1),
                (15, 9),
                (17, 1),
                (18, 14),
            ],
        );
        let entry = NodeId::new(0);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions {
                condition_only: BlockSet::new(count),
                condition_reachability: BlockSet::new(count),
                ..StructureOptions::bare(count, &[])
            },
        );
        assert_total(&graph, entry, &recovered);
        assert_eq!(
            recovered.labels,
            BTreeSet::from([NodeId::new(9)]),
            "{:#?}",
            recovered.root
        );
        assert_eq!(recovered.metrics.gotos, 1, "{:#?}", recovered.root);
    }

    /// **A recovery with no goto that places code twice is tried with its
    /// arms rejoined.** `1` and then `2` test whether to go on to `4`, which
    /// ends the function either way, at `5` or `7`; the other way out runs `3`
    /// and returns at `6`. With the branches' arms nested, both tests reach
    /// `4`, and its split, a copy of `4`, `5` and `7` in each arm, needs no
    /// goto. With the arm that runs on falling out of the branch to where
    /// `4` begins, every block is placed once.
    #[test]
    fn a_recovery_with_no_goto_that_places_code_twice_is_tried_with_its_arms_rejoined() {
        let count = 8;
        let graph = graph_of(
            count,
            &[
                (0, 1),
                (1, 4),
                (1, 2),
                (2, 4),
                (2, 3),
                (3, 6),
                (4, 7),
                (4, 5),
            ],
        );
        let entry = NodeId::new(0);
        let folded = BlockSet::new(count);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions {
                condition_only: folded.clone(),
                condition_reachability: BlockSet::new(count),
                ..StructureOptions::bare(count, &[])
            },
        );
        assert_total(&graph, entry, &recovered);
        assert_runs_as_the_graph(
            &graph,
            entry,
            &folded,
            &BlockSet::new(graph.node_count()),
            &recovered,
        );
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert_eq!(recovered.metrics.split_blocks, 0, "{:#?}", recovered.root);
        assert_eq!(
            recovered.metrics.replicated_tails, 0,
            "{:#?}",
            recovered.root
        );
        assert_eq!(recovered.blocks().len(), count, "{:#?}", recovered.root);
    }

    /// **A loop's exit entered from outside it too is the loop's follow.** `1`
    /// goes on to `4`, which ends the function either way, at `5` or `7`, or
    /// enters the loop at `2`; the loop leaves at `2`'s test to `6`, which
    /// returns, and at `3`'s to `4`. Placed inside the loop, where `3` leaves
    /// to it, `4` would be entered from `1` by a copy of what it begins; as
    /// the follow, it is placed once after the loop, and `6` where the loop
    /// leaves to it.
    #[test]
    fn a_loops_exit_entered_from_outside_it_too_is_the_loops_follow() {
        let count = 8;
        let graph = graph_of(
            count,
            &[
                (0, 1),
                (1, 4),
                (1, 2),
                (2, 6),
                (2, 3),
                (3, 4),
                (3, 2),
                (4, 5),
                (4, 7),
            ],
        );
        let entry = NodeId::new(0);
        let folded = BlockSet::new(count);
        let recovered = structure(
            &graph,
            entry,
            &StructureOptions {
                condition_only: folded.clone(),
                condition_reachability: BlockSet::new(count),
                condition_expressible: set_of(count, &[2]),
                ..StructureOptions::bare(count, &[])
            },
        );
        assert_total(&graph, entry, &recovered);
        assert_runs_as_the_graph(
            &graph,
            entry,
            &folded,
            &BlockSet::new(graph.node_count()),
            &recovered,
        );
        assert_eq!(recovered.metrics.gotos, 0, "{:#?}", recovered.root);
        assert_eq!(recovered.blocks().len(), count, "{:#?}", recovered.root);
    }
}
