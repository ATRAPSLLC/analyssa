//! What a recovery reads beyond the graph, sized to the graph it reads.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    analysis::{
        loops::{LoopForest, detect_loops},
        structure::{
            BlockSet, CaseTest, ProtectedRegion, options::normalised_region, regions::Attempt,
            split::SplitGraph, walk::recover,
        },
    },
    graph::{
        GraphBase, NodeId, Predecessors, Successors,
        algorithms::{DominatorTree, compute_dominators},
    },
};

/// What a recovery reads beyond the graph, sized to the graph it reads it
/// for.
#[derive(Clone)]
pub struct Inputs {
    /// See [`StructureOptions::condition_only`](crate::analysis::structure::StructureOptions::condition_only).
    pub condition_only: BlockSet,

    /// See [`StructureOptions::condition_reachability`](crate::analysis::structure::StructureOptions::condition_reachability).
    pub condition_reachability: BlockSet,

    /// See [`StructureOptions::condition_expressible`](crate::analysis::structure::StructureOptions::condition_expressible).
    pub condition_expressible: BlockSet,

    /// See [`StructureOptions::condition_operand`](crate::analysis::structure::StructureOptions::condition_operand).
    pub condition_operand: BlockSet,

    /// See [`StructureOptions::no_return`](crate::analysis::structure::StructureOptions::no_return).
    pub no_return: BlockSet,

    /// See [`StructureOptions::edge_blocks`](crate::analysis::structure::StructureOptions::edge_blocks).
    pub edge_blocks: BlockSet,

    /// See [`StructureOptions::regions`](crate::analysis::structure::StructureOptions::regions).
    pub regions: Vec<ProtectedRegion>,

    /// See [`StructureOptions::case_tests`](crate::analysis::structure::StructureOptions::case_tests).
    pub case_tests: BTreeMap<NodeId, CaseTest>,

    /// See [`StructureOptions::repeated_tests`](crate::analysis::structure::StructureOptions::repeated_tests).
    pub repeated_tests: BTreeMap<NodeId, NodeId>,

    /// See [`StructureOptions::threaded_tests`](crate::analysis::structure::StructureOptions::threaded_tests).
    pub threaded_tests: BTreeMap<(NodeId, usize), NodeId>,

    /// See [`StructureOptions::statements`](crate::analysis::structure::StructureOptions::statements).
    pub statements: Statements,

    /// See [`StructureOptions::max_depth`](crate::analysis::structure::StructureOptions::max_depth), at least 1.
    pub max_depth: usize,

    /// Blocks to take as the join of the branch that dominates one, where
    /// the walk can ([`Structurer::preferred`](crate::analysis::structure::walk::Structurer::preferred)).
    ///
    /// A tail several arms run on to is the join of the conditional they
    /// belong to, but the block the arms meet at is the branch's immediate
    /// post-dominator, and an arm that leaves the loop instead of running on
    /// to the tail makes that the latch rather than the tail. The tail is
    /// then placed inside the first arm that reaches it and every other arm
    /// jumps to it. A recovery is therefore tried again with the tails the
    /// first one jumped to offered as joins, and kept when it reads with
    /// fewer gotos.
    pub preferred_joins: BlockSet,

    /// Whether a block that runs statements besides its test may join a
    /// short-circuit condition ([`MergedCondition::step`](crate::analysis::structure::conditions::MergedCondition::step)), its statements then
    /// written as the comma operands of the test. Off, only a block that
    /// computes nothing but its condition joins one, or one whose one
    /// statement is the assignment its test reads (`(c = *p) != 0`); a
    /// recovery is tried with it on, and kept where it costs less
    /// ([`Attempt::cost`](crate::analysis::structure::regions::Attempt::cost)).
    pub fold_statements: bool,

    /// Whether a branch whose arms meet only where the function ends rejoins
    /// at the end of the region it is in, when an arm runs on to it
    /// ([`Structurer::rejoin`](crate::analysis::structure::walk::Structurer::rejoin)): the arms that return keep their returns, and
    /// the arm that does not falls out of the branch to where the region
    /// ends, rather than jumping there from inside it. Each order of the
    /// polar blocks is recovered with it on as well, finished, and kept where
    /// it costs less ([`structure`](crate::analysis::structure::structure)): where the arm that runs on reaches the
    /// region's end through a loop it enters elsewhere too, the loop placed
    /// in that arm is the better recovery (`pmatch`'s run of `?`s).
    pub fall_out: bool,

    /// Whether a switch resumes at an offered tail most of its arms meet at
    /// that runs on to its join
    /// ([`Structurer::switch_join`](crate::analysis::structure::walk::Structurer::switch_join)).
    /// A trial of its own, kept where the finished recovery costs less for
    /// it ([`structure`](crate::analysis::structure::structure)): where the
    /// arms that skip the tail can fall through into the arm holding it
    /// instead, resuming at the tail makes each of them jump (dash's
    /// `openredirect`, whose `open` the compiler shares across its cases).
    pub switch_tails: bool,

    /// Whether the routes held here ([`Self::routed`]) are taken before the
    /// other trials read the recovery. [`structure`](crate::analysis::structure::structure)
    /// takes the routes the caller proved one unit at a time
    /// ([`Self::route_units`]), each kept only where the finished recovery
    /// costs less for it: a route that pays where it is taken can cost more
    /// once the splits after it have run, and one beside it can cost where
    /// it pays.
    pub route: bool,
}

/// Routes the caller proved that share a block, taken or left together
/// ([`Inputs::route_units`]).
#[derive(Debug, Default)]
pub struct RouteUnit {
    /// The blocks the routes name.
    pub blocks: BTreeSet<NodeId>,

    /// The unit's threaded edges, as [`Inputs::threaded_tests`] holds them.
    pub threaded: BTreeMap<(NodeId, usize), NodeId>,

    /// The unit's repeats, as [`Inputs::repeated_tests`] holds them.
    pub repeated: BTreeMap<NodeId, NodeId>,
}

impl Inputs {
    /// Recovers the graph with these inputs, then tries the refinements that
    /// need a first recovery to read: jumps threaded past a test routed back
    /// through it, the tails the recovery jumped to or repeated offered as
    /// joins, and loops entered at more than one block split. Returns the
    /// best recovery, which says whether any of them kept a block out of a
    /// condition for [`Self::fold_statements`] alone, with the graph it was
    /// made of and the inputs that made it.
    /// These inputs for the graph read with the blocks of `flipped` the other
    /// way round: a case test's successor index exchanged, and a threaded
    /// edge's.
    pub fn reoriented(&self, flipped: &BlockSet) -> Self {
        let mut inputs = self.clone();
        for (node, test) in &mut inputs.case_tests {
            if flipped.contains(*node) && test.holds < 2 {
                test.holds = usize::from(test.holds == 0);
            }
        }
        inputs.threaded_tests = self
            .threaded_tests
            .iter()
            .map(|(&(from, index), &test)| {
                let index = if flipped.contains(from) && index < 2 {
                    usize::from(index == 0)
                } else {
                    index
                };
                ((from, index), test)
            })
            .collect();
        inputs
    }

    /// The recovery kept of `graph`: nesting each block's statements, or,
    /// where that declined a fold and costs anything, folding every block
    /// whose statements a condition can say, whichever needs fewer gotos, or
    /// as many with less code placed twice.
    ///
    /// Statements written inside a condition are the cost of a goto, or of
    /// code placed twice, that they save, and only that. Each recovery runs to
    /// the end before the two are compared, since what every later trial
    /// saves depends on it. Where no trial kept a block out for the setting
    /// alone, the folding recovery would make the same choices, and is not
    /// made.
    pub fn best<G>(self, graph: &G, entry: NodeId, offer_joins: bool) -> (Attempt, SplitGraph, Self)
    where
        G: GraphBase + Successors + Predecessors,
    {
        let folds = self.would_fold_statements();
        let mut best = self.clone().recovered(graph, entry, offer_joins);
        if folds && best.0.declined_fold && best.0.cost() > (0, 0) {
            let folding = Self {
                fold_statements: true,
                ..self
            };
            let attempt = folding.recovered(graph, entry, offer_joins);
            if attempt.0.cost() < best.0.cost() {
                best = attempt;
            }
        }
        best
    }

    pub fn recovered<G>(
        self,
        graph: &G,
        entry: NodeId,
        offer_joins: bool,
    ) -> (Attempt, SplitGraph, Self)
    where
        G: GraphBase + Successors + Predecessors,
    {
        let node_count = graph.node_count();
        let guarded = self.guarded(node_count);
        let mut inputs = self;
        let mut best = recover(graph, entry, &inputs);
        let mut declined_fold = best.declined_fold;
        let mut current = SplitGraph::of(graph);
        // A jump threaded past a test its condition decides is routed back
        // through the test on the same terms: it is the same program either
        // way, and kept only when it reads with fewer gotos.
        if let Some(rethreaded) = inputs.rethreaded(&current, entry, &guarded) {
            let attempt = recover(&rethreaded, entry, &inputs);
            declined_fold |= attempt.declined_fold;
            if attempt.gotos() < best.gotos() {
                best = attempt;
                current = rethreaded;
            }
        }
        // Where they are tried ([`Self::route`]), the routes the caller
        // proved are taken; whether that pays is decided on the recovery
        // finished with them.
        if inputs.route
            && let Some(routed) = inputs.routed(&current, entry, &guarded)
        {
            best = recover(&routed, entry, &inputs);
            declined_fold |= best.declined_fold;
            current = routed;
        }
        // Where joins are offered (`offer_joins`), each tail the recovery
        // jumped to, or placed again where its arms run on to their region's
        // end, is offered as the join of the conditional that dominates it:
        // taken there, its arms fall into it and an arm that leaves instead
        // prints the transfer it already is. Whether that pays is decided on
        // the recovery finished with it
        // ([`structure`](crate::analysis::structure::structure)). Tried on
        // the graph as it stands, before any split, so the blocks the labels
        // name are the blocks the recovery placed.
        let tails: BTreeSet<NodeId> = best
            .structured
            .labels
            .union(&best.rejoining_tails)
            .copied()
            .collect();
        if offer_joins && !tails.is_empty() {
            let joining = inputs.joining(&tails, node_count);
            best = recover(&current, entry, &joining);
            declined_fold |= best.declined_fold;
            inputs = joining;
        }
        // Each loop entered at more than one block is split on trial, and the
        // split kept only when the whole recovery needs fewer gotos for it -
        // counted as C writes them ([`Region::gotos`]): the copies carry every
        // goto their own structure needs, so splitting a large region can cost
        // more than the entry it removes - and the statements its copies print
        // are worth the gotos it saves ([`Inputs::split_pays`]).
        let mut declined = BTreeSet::new();
        while let Some(split) = current.next(entry, &guarded, &declined) {
            let Some(candidate) = split.graph else {
                declined.insert(split.key);
                continue;
            };
            let mut attempt = recover(&candidate, entry, &inputs.lifted(&candidate));
            declined_fold |= attempt.declined_fold;
            if inputs.split_pays(&best, &attempt, &current, &candidate) {
                attempt.renumber(&|node| candidate.original(node));
                attempt.structured.metrics.split_blocks = candidate.copies();
                best = attempt;
                current = candidate;
            } else {
                declined.insert(split.key);
            }
        }
        best.declined_fold = declined_fold;
        (best, current, inputs)
    }

    /// Splits each block `best` still reaches by a goto, where several paths
    /// join, on trial: every path but one into it takes a copy of what it
    /// dominates ([`SplitGraph::split_join`]), and the split is kept only
    /// when the whole recovery of `current` with these inputs needs fewer
    /// gotos for it, worth the statements its copies print
    /// ([`Inputs::split_pays`]).
    ///
    /// A second walk also offers, where that split does not pay, copies that
    /// take along the code past it that ends the function, so they rejoin
    /// nothing; the cheaper of the two walks' recoveries is kept. Each walk
    /// keeps a split the moment it pays, and a copy carrying more takes a
    /// share of the graph's bound and a shape that later splits of the plain
    /// walk would have bettered, so the second walk alone can end with more
    /// gotos than the first. Returns the best recovery.
    pub fn split_joins(&self, best: Attempt, current: SplitGraph, entry: NodeId) -> Attempt {
        let plain = self.split_walk(best.clone(), current.clone(), entry, false);
        let ending = self.split_walk(best, current, entry, true);
        if ending.cost() < plain.cost() {
            ending
        } else {
            plain
        }
    }

    /// One walk of [`Self::split_joins`]: each label `best` names is tried
    /// as the dominated region it begins, and, where `ending` and that does
    /// not pay, with the code past it that ends the function; the first
    /// split that needs fewer gotos, for copies worth them
    /// ([`Inputs::split_pays`]), is kept.
    ///
    /// A sweep tries each label once. A split kept makes another graph,
    /// whose recovery can need a split the graph before declined: an exit
    /// every early return jumps to does not pay while the arms around those
    /// returns still need gotos of their own, and does once their splits are
    /// kept. So a sweep that kept a split is followed by another over every
    /// label, until one keeps none. Each split kept needs fewer gotos than
    /// the graph before it, so the walk ends.
    fn split_walk(
        &self,
        mut best: Attempt,
        mut current: SplitGraph,
        entry: NodeId,
        ending: bool,
    ) -> Attempt {
        let guarded = self.guarded(current.node_count());
        loop {
            let mut tried = BTreeSet::new();
            let mut kept = false;
            while let Some(target) = best
                .structured
                .labels
                .iter()
                .copied()
                .find(|label| !tried.contains(label))
            {
                tried.insert(target);
                for carried in [false, ending] {
                    let Some(candidate) = current.split_join(target, entry, &guarded, carried)
                    else {
                        continue;
                    };
                    let mut attempt = recover(&candidate, entry, &self.lifted(&candidate));
                    if self.split_pays(&best, &attempt, &current, &candidate) {
                        attempt.renumber(&|node| candidate.original(node));
                        attempt.structured.metrics.split_blocks = candidate.copies();
                        best = attempt;
                        current = candidate;
                        kept = true;
                        break;
                    }
                    if !carried && !ending {
                        break;
                    }
                }
            }
            if !kept {
                return best;
            }
        }
    }

    /// Whether folding statements into conditions
    /// ([`Self::fold_statements`]) would admit a block it does not admit
    /// already: one whose statements can be said in a condition, that does
    /// more than test and is not its test's one operand.
    fn would_fold_statements(&self) -> bool {
        !self.fold_statements
            && self.condition_expressible.iter().any(|node| {
                !self.condition_only.contains(node) && !self.condition_operand.contains(node)
            })
    }

    /// Returns every block a protected region or one of its handlers holds.
    pub fn guarded(&self, node_count: usize) -> BlockSet {
        let mut guarded = BlockSet::new(node_count);
        for region in &self.regions {
            let handlers = region.handlers.iter().flat_map(|handler| {
                let filter = handler.kind.filter().map(|filter| filter.blocks.iter());
                handler.blocks.iter().chain(filter.into_iter().flatten())
            });
            for node in region.blocks.iter().chain(handlers) {
                guarded.insert(node);
            }
        }
        guarded
    }

    /// Returns `graph` with every jump a compiler threaded past a test its
    /// own condition decides routed back through that test, or `None` when
    /// it threads none.
    ///
    /// `if (!(a && b) && c) { x } else if (b) { y }`: a compiler that knows
    /// `b` holds wherever `a && b` does sends that edge straight to `y`,
    /// past the test of `b`, so `y` has two entries and the first reaches it
    /// only by goto. An edge from a test of a value to `target` threads past
    /// `skipped`, a predecessor of `target`, when `skipped` does nothing but
    /// test the same value as it is entered with ([`CaseTest::entered`]) and
    /// every value the edge carries is one `skipped` sends to `target`
    /// ([`CaseTest::decides`]): run on the edge's path, `skipped` computes
    /// nothing and goes where the edge went. The edge keeps the graph's
    /// loops ([`Shape::admits`]), and a test that is the block's other
    /// successor stays skipped, which would leave the block two edges to one
    /// place, as does a block of a protected region or handler (`guarded`),
    /// whose edges are the region's.
    pub fn rethreaded(
        &self,
        graph: &SplitGraph,
        entry: NodeId,
        guarded: &BlockSet,
    ) -> Option<SplitGraph> {
        let mut rethreaded = graph.clone();
        let mut shape = Shape::of(&rethreaded, entry);
        let mut changed = false;
        for (&from, jump) in &self.case_tests {
            if guarded.contains(from) {
                continue;
            }
            for index in 0..2 {
                let targets: Vec<NodeId> = rethreaded.successors(from).collect();
                let (Some(&target), [first, second]) = (targets.get(index), targets.as_slice())
                else {
                    break;
                };
                if first == second {
                    break;
                }
                let through = rethreaded
                    .predecessors(target)
                    .filter(|skipped| {
                        *skipped != from
                            && !targets.contains(skipped)
                            && self.condition_only.contains(*skipped)
                            && !guarded.contains(*skipped)
                    })
                    .filter(|skipped| {
                        let Some(test) = self.case_tests.get(skipped) else {
                            return false;
                        };
                        let sent: Vec<NodeId> = rethreaded.successors(*skipped).collect();
                        test.entered
                            && test.value == jump.value
                            // A test that spells the jump's own comparison is
                            // read as a repeat of it, not as the next arm of a
                            // chain: routing the edge through `x == 0` where
                            // the jump is `x == 0` prints the same question
                            // twice, once answered. Another constant or
                            // another relation reads as the chain it is.
                            && (test.relation != jump.relation
                                || test.constant != jump.constant)
                            && matches!(sent.as_slice(), [one, other] if one != other)
                            && sent
                                .iter()
                                .position(|node| *node == target)
                                .is_some_and(|toward| jump.decides(index, test, toward))
                    })
                    .filter(|skipped| {
                        shape.admits(&rethreaded, &self.condition_reachability, from, *skipped)
                    })
                    .min();
                if let Some(skipped) = through {
                    rethreaded.reroute(from, index, skipped);
                    shape = Shape::of(&rethreaded, entry);
                    changed = true;
                }
            }
        }
        changed.then_some(rethreaded)
    }

    /// Returns `graph` with the routes the caller proved taken, or `None`
    /// when it states none that applies.
    ///
    /// An edge the caller states a compiler threaded past a test the path
    /// decides ([`StructureOptions::threaded_tests`](crate::analysis::structure::StructureOptions::threaded_tests))
    /// goes through that test, where the test goes on to the edge's target,
    /// is not the block's other successor, and the edge keeps the graph's
    /// loops ([`Shape::admits`]). Then a block that ends with the whole of
    /// another block's test
    /// ([`StructureOptions::repeated_tests`](crate::analysis::structure::StructureOptions::repeated_tests))
    /// goes to that block instead, where the two go to the same blocks and
    /// the edge keeps the graph's loops: a threaded edge routed first can
    /// leave its block a repeat of the test it was threaded past. What the
    /// test's rows define the caller has proved harmless on the new path. No
    /// block of a protected region or handler (`guarded`) is routed.
    pub fn routed(
        &self,
        graph: &SplitGraph,
        entry: NodeId,
        guarded: &BlockSet,
    ) -> Option<SplitGraph> {
        let mut routed = graph.clone();
        let mut shape = Shape::of(&routed, entry);
        let mut changed = false;
        for (&(from, index), &test) in &self.threaded_tests {
            if from == test || guarded.contains(from) || guarded.contains(test) {
                continue;
            }
            let sends: Vec<NodeId> = routed.successors(from).collect();
            let Some(&target) = sends.get(index) else {
                continue;
            };
            if sends.contains(&test)
                || !routed.successors(test).any(|next| next == target)
                || !shape.admits(&routed, &self.condition_reachability, from, test)
            {
                continue;
            }
            routed.reroute(from, index, test);
            shape = Shape::of(&routed, entry);
            changed = true;
        }
        for (&from, &test) in &self.repeated_tests {
            if from == test || guarded.contains(from) || guarded.contains(test) {
                continue;
            }
            let mut sends: Vec<NodeId> = routed.successors(from).collect();
            let mut tests: Vec<NodeId> = routed.successors(test).collect();
            sends.sort_unstable();
            tests.sort_unstable();
            if sends.len() != 2
                || sends != tests
                || sends.first() == sends.last()
                || !shape.admits(&routed, &self.condition_reachability, from, test)
            {
                continue;
            }
            routed.repeat(from, test);
            shape = Shape::of(&routed, entry);
            changed = true;
        }
        changed.then_some(routed)
    }

    /// The routes the caller proved, in units that share a block: a threaded
    /// edge with the repeat it can leave its block, and a repeat with the
    /// repeat the other way, in the order of the first block each unit names.
    pub fn route_units(&self) -> Vec<RouteUnit> {
        let mut units: Vec<RouteUnit> = Vec::new();
        let routes = self
            .threaded_tests
            .iter()
            .map(|(&(from, index), &test)| (from, test, Some(index)))
            .chain(
                self.repeated_tests
                    .iter()
                    .map(|(&from, &test)| (from, test, None)),
            );
        for (from, test, index) in routes {
            let (sharing, mut apart): (Vec<_>, Vec<_>) = units
                .into_iter()
                .partition(|unit| unit.blocks.contains(&from) || unit.blocks.contains(&test));
            let mut unit = RouteUnit {
                blocks: BTreeSet::from([from, test]),
                ..RouteUnit::default()
            };
            for shared in sharing {
                unit.blocks.extend(shared.blocks);
                unit.threaded.extend(shared.threaded);
                unit.repeated.extend(shared.repeated);
            }
            match index {
                Some(index) => unit.threaded.insert((from, index), test),
                None => unit.repeated.insert(from, test),
            };
            apart.push(unit);
            units = apart;
        }
        units.sort_by_key(|unit| unit.blocks.first().copied());
        units
    }

    /// Returns the inputs with `joins` offered as conditional joins.
    fn joining(&self, joins: &BTreeSet<NodeId>, node_count: usize) -> Self {
        Self {
            preferred_joins: BlockSet::from_nodes(node_count, joins.iter().copied()),
            ..self.clone()
        }
    }

    /// The statements the copies `split` holds print, each as its original
    /// does.
    fn copied(&self, split: &SplitGraph) -> usize {
        split
            .node_ids()
            .filter(|node| split.original(*node) != *node)
            .map(|node| self.statements.of(split.original(node)))
            .fold(0usize, usize::saturating_add)
    }

    /// Whether the split `candidate`, recovered as `attempt`, is worth more
    /// than the recovery `best` of `current`: it writes fewer gotos, and the
    /// statements its copies add are no more than [`STATEMENTS_PER_GOTO`] for
    /// each goto it saves.
    fn split_pays(
        &self,
        best: &Attempt,
        attempt: &Attempt,
        current: &SplitGraph,
        candidate: &SplitGraph,
    ) -> bool {
        let Some(saved) = best
            .gotos()
            .checked_sub(attempt.gotos())
            .filter(|saved| *saved > 0)
        else {
            return false;
        };
        let added = self.copied(candidate).saturating_sub(self.copied(current));
        added <= saved.saturating_mul(STATEMENTS_PER_GOTO)
    }

    /// Returns the inputs for `split`, each copy taking its original's part.
    ///
    /// A copy is the same code as its original, so it tests, computes and
    /// ends the function exactly as the original does. No copy lies in a
    /// protected region ([`SplitGraph::of`]), so the regions keep their
    /// blocks.
    fn lifted(&self, split: &SplitGraph) -> Self {
        let count = split.node_count();
        let lift = |set: &BlockSet| {
            BlockSet::from_nodes(
                count,
                split
                    .node_ids()
                    .filter(|node| set.contains(split.original(*node))),
            )
        };
        let mut unused = 0usize;
        Self {
            condition_only: lift(&self.condition_only),
            condition_reachability: lift(&self.condition_reachability),
            condition_expressible: lift(&self.condition_expressible),
            condition_operand: lift(&self.condition_operand),
            no_return: lift(&self.no_return),
            edge_blocks: lift(&self.edge_blocks),
            regions: self
                .regions
                .iter()
                .map(|region| normalised_region(count, region, &mut unused))
                .collect(),
            case_tests: split
                .node_ids()
                .filter_map(|node| {
                    self.case_tests
                        .get(&split.original(node))
                        .map(|test| (node, *test))
                })
                .collect(),
            // A copy of a block repeats the test its original repeats, and
            // goes to that test's original block.
            repeated_tests: split
                .node_ids()
                .filter_map(|node| {
                    self.repeated_tests
                        .get(&split.original(node))
                        .map(|test| (node, *test))
                })
                .collect(),
            // A copy's edges are its original's, threaded past the same test.
            threaded_tests: self
                .threaded_tests
                .iter()
                .flat_map(|(&(from, index), &test)| {
                    split
                        .node_ids()
                        .filter(move |node| split.original(*node) == from)
                        .map(move |node| ((node, index), test))
                })
                .collect(),
            // A copy prints what its original prints.
            statements: Statements(
                split
                    .node_ids()
                    .map(|node| self.statements.of(split.original(node)))
                    .collect(),
            ),
            max_depth: self.max_depth,
            preferred_joins: lift(&self.preferred_joins),
            fold_statements: self.fold_statements,
            fall_out: self.fall_out,
            switch_tails: self.switch_tails,
            route: self.route,
        }
    }
}

/// The statements each block prints
/// ([`StructureOptions::statements`](crate::analysis::structure::StructureOptions::statements)),
/// by node.
#[derive(Clone, Default)]
pub struct Statements(pub Vec<usize>);

impl Statements {
    /// The statements `node` prints: one for a block past the end of the
    /// list.
    pub fn of(&self, node: NodeId) -> usize {
        self.0.get(node.index()).copied().unwrap_or(1)
    }
}

/// How many statements a recovery may place again to save one goto: a split
/// whose copies print more than this for each goto it removes is not kept.
pub const STATEMENTS_PER_GOTO: usize = 16;

/// The dominators and natural loops of a graph, which an edge a recovery
/// adds must keep.
struct Shape {
    /// The graph's dominators from its entry.
    dominators: DominatorTree,

    /// The graph's natural loops.
    loops: LoopForest,
}

impl Shape {
    /// Returns the shape of `graph` entered at `entry`.
    fn of(graph: &SplitGraph, entry: NodeId) -> Self {
        let dominators = compute_dominators(graph, entry);
        let loops = detect_loops(graph, &dominators);
        Self { dominators, loops }
    }

    /// Whether an edge of `graph` from `from` to `to` keeps its loops, and
    /// leaves what every block reaches as it was wherever an answer the
    /// recovery was given depended on it (`decided`).
    ///
    /// `to` runs, as some path from the entry reaches it, so the edge makes
    /// no unreached block code that runs; every loop that holds `to` holds
    /// `from` too, unless `to` is its header, so the edge enters no loop
    /// anywhere but its header; and `to` reaches `from` by no path that
    /// takes no loop's back edge, so the only cycles through the edge are
    /// laps of loops the graph has. Inside a loop, every block reaches every
    /// other through its back edge. An edge to the header of a loop holding
    /// `from`, which dominates `from`, is a back edge of that loop: every
    /// cycle through it is one of the loop's laps.
    ///
    /// An edge `from` does not already reach grows what every block reaching
    /// `from` reaches, since the walk from such a block crosses it. A host
    /// that decided anything from what a block reaches - whether a place a
    /// block defines reaches a reader, which decides whether the recovery may
    /// fold that block into a condition - decided it for the graph it was
    /// given, and growing the set would leave the recovery reading an answer
    /// taken before the edge. So such an edge is admitted only where no block
    /// of `decided` reaches `from`, `from` itself included: elsewhere every
    /// path over the new edge need not stand for one the graph had, and
    /// nothing read from the old one does.
    ///
    /// Copying a block is not this hazard: a copy carries the original's rows
    /// and successors, and a block reaches the copy only through a
    /// predecessor that reached the original, so no block comes to reach
    /// anything it did not.
    fn admits(&self, graph: &SplitGraph, decided: &BlockSet, from: NodeId, to: NodeId) -> bool {
        if !self.dominators.is_reachable(to)
            || (!graph.reaches(from, to)
                && decided
                    .iter()
                    .any(|block| block == from || graph.reaches(block, from)))
            || self
                .loops
                .loops()
                .iter()
                .any(|info| info.contains(to) && info.header != to && !info.contains(from))
        {
            return false;
        }
        if self.dominators.dominates(to, from)
            && self
                .loops
                .loops()
                .iter()
                .any(|info| info.header == to && info.contains(from))
        {
            return true;
        }
        let mut seen = vec![false; graph.node_count()];
        let mut pending = vec![to];
        while let Some(node) = pending.pop() {
            for next in graph.successors(node) {
                if self.dominators.dominates(next, node) {
                    continue;
                }
                if next == from {
                    return false;
                }
                if let Some(flag) = seen.get_mut(next.index())
                    && !*flag
                {
                    *flag = true;
                    pending.push(next);
                }
            }
        }
        true
    }
}
