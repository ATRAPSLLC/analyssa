//! The walk that places each block of the graph in the region tree.

use std::{
    cell::Cell,
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
    mem,
};

use crate::{
    analysis::{
        loops::{LoopForest, detect_loops},
        structure::{
            BlockSet, HandlerRegion, IfRegion, LoopRegion, Predicate, ProtectedHandlerKind,
            ProtectedRegion, Region, StructureMetrics, Structured, StructuredLoop, SwitchCase,
            SwitchRegion, TryRegion,
            cases::CaseTree,
            conditions::MergedCondition,
            convergence::Convergence,
            endings::Endings,
            facts::Facts,
            follow::Exits,
            inputs::{Inputs, STATEMENTS_PER_GOTO},
            loops::{LoopEnd, LoopShape},
            regions::Attempt,
        },
    },
    bitset::BitSet,
    graph::{
        GraphBase, NodeId, Predecessors, Successors,
        algorithms::{compute_dominators, compute_post_dominators},
    },
};

/// Recovers the structure of `graph` from `entry`, which lies inside it.
pub fn recover<G>(graph: &G, entry: NodeId, inputs: &Inputs) -> Attempt
where
    G: GraphBase + Successors + Predecessors,
{
    let node_count = graph.node_count();
    let regions = &inputs.regions;
    let dominators = compute_dominators(graph, entry);
    let post_dominators = compute_post_dominators(graph);
    let returning = (inputs.no_return.count() > 0)
        .then(|| Convergence::of_function(graph, node_count, &inputs.no_return));
    let loops = detect_loops(graph, &dominators);

    let mut headers = BTreeMap::new();
    for (index, info) in loops.loops().iter().enumerate() {
        headers.insert(info.header, index);
    }

    // Ascending by size, then by declaration index, and pushed in that
    // order: `Vec::pop` therefore yields the *largest* region first, which
    // is the one that must open first so the smaller one opens inside its
    // body. See `Structurer::protected`.
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by_cached_key(|index| {
        (
            regions
                .get(*index)
                .map_or(0, |region| region.blocks.count()),
            *index,
        )
    });
    let mut protected: BTreeMap<NodeId, Vec<usize>> = BTreeMap::new();
    for index in order {
        if let Some(region) = regions.get(index) {
            protected.entry(region.entry).or_default().push(index);
        }
    }
    let guarded = inputs.guarded(node_count);
    // A tail may not begin a loop or a protected region, nor lie in one
    // ([`Structurer::tail`]).
    let endings = Endings::of(
        graph,
        node_count,
        |node| {
            headers.contains_key(&node) || protected.contains_key(&node) || guarded.contains(node)
        },
        |node| inputs.statements.of(node),
    );

    let mut structurer = Structurer {
        facts: Facts {
            graph,
            node_count,
            endings,
            dominators,
            post_dominators,
            returning,
            loops: &loops,
            headers,
            regions,
            guarded,
            condition_only: inputs.condition_only.clone(),
            condition_expressible: inputs.condition_expressible.clone(),
            condition_operand: inputs.condition_operand.clone(),
            edge_blocks: inputs.edge_blocks.clone(),
            fold_statements: inputs.fold_statements,
            fall_out: inputs.fall_out,
            switch_tails: inputs.switch_tails,
            case_tests: inputs.case_tests.clone(),
            preferred_joins: inputs.preferred_joins.clone(),
            max_depth: inputs.max_depth,
            statements: inputs.statements.clone(),
        },
        protected,
        declined_fold: Cell::new(false),
        visited: BlockSet::new(node_count),
        labels: BTreeSet::new(),
        rejoining_tails: BTreeSet::new(),
        frames: Vec::new(),
        switches: Vec::new(),
        frame_barrier: 0,
        enclosing: Vec::new(),
        confined: Vec::new(),
        depth: 0,
        metrics: StructureMetrics::default(),
    };

    let root = structurer.emit_function(entry);
    let root = structurer.early_exits(root);
    let mut metrics = structurer.metrics;
    metrics.unreached = node_count.saturating_sub(metrics.blocks);
    Attempt {
        structured: Structured::new(root, metrics),
        rejoining_tails: structurer.rejoining_tails,
        declined_fold: structurer.declined_fold.get(),
    }
}

/// One enclosing loop during the walk.
pub struct LoopFrame<'a> {
    /// The loop's header block.
    pub header: NodeId,

    /// Blocks belonging to the loop, as [`crate::analysis::loops::detect_loops`]
    /// recorded them - one bit per node of the graph.
    pub body: &'a BitSet,

    /// Block control resumes at on `break`, if the loop is left.
    pub follow: Option<NodeId>,

    /// Latch whose terminator became the loop condition, and whose outgoing
    /// edges must therefore not be walked as ordinary control flow.
    pub tail: Option<NodeId>,

    /// Post-dominance over the paths that come round the loop again.
    pub continuation: Convergence,
}

/// An arm of a switch before it is placed: where it starts, and what selects
/// it.
struct PendingArm {
    /// The arm's first block.
    target: NodeId,
    /// See [`SwitchCase::selectors`].
    selectors: Vec<usize>,
    /// See [`SwitchCase::values`].
    values: Vec<i64>,
    /// See [`SwitchCase::default`].
    default: bool,
}

/// A `switch` the walk is inside, for the `break` that leaves it.
struct OpenSwitch {
    /// The block whose dispatch the switch is.
    head: NodeId,
    /// Where control resumes after the switch, when its arms meet there.
    join: Option<NodeId>,
    /// How many loops were open when the switch was: a loop entered since
    /// captures a `break`, so the switch's is no longer written so.
    pub frames: usize,
}

impl OpenSwitch {
    /// Whether a path from inside the switch that reaches `node` leaves the
    /// switch by its `break`, with `frames` loops open.
    fn breaks_at(&self, node: NodeId, frames: usize) -> bool {
        Some(node) == self.join && self.frames == frames
    }
}

impl LoopFrame<'_> {
    /// Returns whether `node` belongs to the loop.
    ///
    /// Total over every [`NodeId`]: a loop body is sized to the graph's node
    /// count, so an index past it is simply not a member.
    pub fn holds(&self, node: NodeId) -> bool {
        self.body.contains_checked(node.index())
    }

    /// Whether a path from inside the loop that reaches `node` goes on as
    /// `continue` does: to the next test of the loop, at the header of a
    /// loop tested there or nowhere. A `do`/`while` tests at its latch, so
    /// its `continue` runs that test, and a path straight to its header
    /// skips it: no `continue` says that path.
    fn continued_at(&self, node: NodeId) -> bool {
        node == self.header && self.tail.is_none()
    }
}

/// Where a copied tail may end besides the block that ends the function
/// ([`Structurer::tail`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailEnd {
    /// Only at the block that ends the function.
    Function,

    /// Also before `follow`, where the region the copy is placed in resumes,
    /// and before an enclosing loop's header or follow, which the copy then
    /// leaves to by the `continue` or `break` naming it.
    Region {
        /// Where the region resumes.
        follow: Option<NodeId>,
    },
}

/// Whether a walk may see the loop frames its caller was inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Barrier {
    /// The walk continues inside the same loops.
    Inherit,

    /// The walk is entered by the runtime rather than by an edge of the loop,
    /// so no enclosing loop bounds it.
    Raise,
}

/// One region the walk enters, and everything entering it decides.
///
/// Confinement, enclosing follow and loop-frame barrier are three answers to
/// one question - which scope's state governs the walk about to happen - and
/// keeping them in three hand-managed stacks is what let them disagree. A
/// `Scope` has no default and no builder, so entering a region without
/// deciding all three does not compile.
#[derive(Debug, Clone, Copy)]
struct Scope<'a> {
    /// Blocks the walk may place, or `None` to keep the enclosing confinement.
    confinement: Option<&'a BlockSet>,

    /// Block the enclosing region resumes at, which the walk must leave for
    /// rather than place, or `None` when the region has no follow.
    follow: Option<NodeId>,

    /// Whether the enclosing loops still apply.
    barrier: Barrier,
}

/// Walk state for one function.
pub struct Structurer<'a, G> {
    /// What the walk reads of the function and never changes.
    pub facts: Facts<'a, G>,

    /// Try entry block to the regions starting there that the walk has not
    /// entered yet, **outermost last**.
    ///
    /// `Vec::pop` therefore returns the region that must open first, and
    /// `emit_try` re-enters the walk at the same block, so the next region
    /// down opens inside the body of the one above it. Reversing the order
    /// would nest the clauses inside out.
    ///
    /// An index is removed when the walk enters its region, so the map means
    /// "regions not yet entered" - which is what `chain` reads while it is
    /// still live, and what lets `emit_function` drain the regions whose entry
    /// the walk never reached.
    pub protected: BTreeMap<NodeId, Vec<usize>>,

    /// See [`Attempt::declined_fold`](crate::analysis::structure::regions::Attempt::declined_fold). Set where a condition is grown,
    /// which reads the walk without changing it.
    pub declined_fold: Cell<bool>,

    /// Blocks already placed in the tree.
    pub visited: BlockSet,

    /// Blocks a goto targets.
    labels: BTreeSet<NodeId>,

    /// See [`Attempt::rejoining_tails`](crate::analysis::structure::regions::Attempt::rejoining_tails).
    rejoining_tails: BTreeSet<NodeId>,

    /// Enclosing loops, innermost last.
    pub frames: Vec<LoopFrame<'a>>,

    /// Enclosing switches, innermost last.
    switches: Vec<OpenSwitch>,

    /// Index into [`Structurer::frames`] below which the walk sees no loop.
    ///
    /// A handler or filter is entered by the runtime, not by an edge of the
    /// loop the protected region sits in, so the loop's body does not bound
    /// the handler's joins. `transfer` deliberately still scans every frame:
    /// the `Region::Try` is emitted lexically inside the loop body, so
    /// `break`/`continue` out of a handler is legal and must not become a goto.
    frame_barrier: usize,

    /// Follow blocks of enclosing regions, innermost last.
    enclosing: Vec<NodeId>,

    /// Block sets the walk is confined to, innermost last.
    ///
    /// A protected body and a handler each own a fixed set of blocks, and the
    /// walk must not place a block belonging to neither inside them. Escapes
    /// leave by goto and are placed at the end of the function instead.
    pub confined: Vec<&'a BlockSet>,

    /// Regions entered and not yet left; see [`StructureOptions::max_depth`](crate::analysis::structure::StructureOptions::max_depth).
    depth: usize,

    /// Counts collected during the walk.
    metrics: StructureMetrics,
}

impl<'a, G> Structurer<'a, G>
where
    G: GraphBase + Successors + Predecessors,
{
    /// Runs `walk` inside `scope`, releasing everything it entered.
    ///
    /// The one place a region is entered, so confinement, enclosing follow and
    /// loop-frame barrier are pushed together and popped together whatever the
    /// walk does.
    fn in_scope<R>(&mut self, scope: Scope<'a>, walk: impl FnOnce(&mut Self) -> R) -> R {
        let saved_barrier = self.frame_barrier;
        if scope.barrier == Barrier::Raise {
            self.frame_barrier = self.frames.len();
        }
        if let Some(blocks) = scope.confinement {
            self.confined.push(blocks);
        }
        if let Some(node) = scope.follow {
            self.enclosing.push(node);
        }

        let result = walk(self);

        if scope.follow.is_some() {
            self.enclosing.pop();
        }
        if scope.confinement.is_some() {
            self.confined.pop();
        }
        self.frame_barrier = saved_barrier;
        result
    }

    /// Returns the innermost loop frame the current walk is inside.
    ///
    /// `None` inside a handler or filter, whose walk the enclosing loop does
    /// not bound; see [`Structurer::frame_barrier`].
    pub fn enclosing_frame(&self) -> Option<&LoopFrame<'a>> {
        self.frames
            .get(self.frame_barrier..)
            .and_then(<[LoopFrame<'a>]>::last)
    }

    /// Emits the whole function, including any block the tree walk did not
    /// place.
    ///
    /// The main walk follows control flow from the entry and places everything
    /// it reaches, so ordinarily it leaves nothing behind. Two things can
    /// remain. A protected region whose entry is unreachable - the runtime
    /// takes the edge into a handler, and nothing says the try itself is
    /// reached - is drained first, through `emit_try`, so its handlers are
    /// still inside a [`Region::Try`] rather than printed as unprotected code
    /// at top level. Then any block a bounded walk had to leave behind is
    /// placed, reached by the goto that already names it.
    ///
    /// Running both unconditionally is what makes the placement invariant hold
    /// by construction rather than by argument: after them, every block
    /// reachable from the entry, and every block a region owns, appears in the
    /// tree.
    fn emit_function(&mut self, entry: NodeId) -> Region {
        let mut parts = vec![self.emit(entry, None, false)];

        while let Some(start) = self.protected.keys().next().copied() {
            let Some(index) = self.take_protected(start) else {
                break;
            };
            let (region, _) = self.emit_try(index, None);
            parts.push(region);
        }

        let mut reachable = self.facts.reachable_from(entry);
        for region in self.facts.regions {
            reachable.insert(region.entry);
            for block in region.blocks.iter() {
                reachable.insert(block);
            }
            for handler in &region.handlers {
                reachable.insert(handler.entry);
                for block in handler.blocks.iter() {
                    reachable.insert(block);
                }
                if let Some(filter) = handler.kind.filter() {
                    reachable.insert(filter.entry);
                    for block in filter.blocks.iter() {
                        reachable.insert(block);
                    }
                }
            }
        }

        // One ascending pass rather than a rescan per orphan: emitting one
        // orphan can place later ones, and the `visited` test skips those.
        let orphans: Vec<NodeId> = reachable.iter().collect();
        for orphan in orphans {
            if self.visited.contains(orphan) {
                continue;
            }
            parts.push(self.emit(orphan, None, false));
        }

        Region::sequence(parts)
    }

    /// Takes the region that must open next at `entry`, if one is left.
    fn take_protected(&mut self, entry: NodeId) -> Option<usize> {
        let remaining = self.protected.get_mut(&entry)?;
        let index = remaining.pop();
        if remaining.is_empty() {
            self.protected.remove(&entry);
        }
        index
    }

    /// Emits the region beginning at `start` and ending before `follow`, one
    /// nesting level deeper.
    ///
    /// Every descent goes through here, so this is the single place nesting is
    /// counted. At the bound the walk is abandoned rather than deepened; see
    /// [`StructureOptions::max_depth`](crate::analysis::structure::StructureOptions::max_depth).
    fn emit(&mut self, start: NodeId, follow: Option<NodeId>, skip_transfer: bool) -> Region {
        let depth = self.depth.saturating_add(1);
        if depth > self.facts.max_depth {
            return self.abandon(start);
        }
        self.depth = depth;
        let region = self.emit_sequence(start, follow, skip_transfer);
        self.depth = self.depth.saturating_sub(1);
        region
    }

    /// Gives up on a walk that would exceed the nesting bound.
    ///
    /// Unconfined, the block becomes a goto and the orphan pass places it at
    /// top level - the module's existing escape hatch, at depth 0.
    ///
    /// Confined, a goto would defer a block the enclosing region owns to top
    /// level *outside* its [`Region::Try`], printing protected code as
    /// unprotected. The rest of the confinement is flattened in place instead:
    /// coarse protection rather than wrong protection.
    fn abandon(&mut self, start: NodeId) -> Region {
        self.metrics.depth_limited = self.metrics.depth_limited.saturating_add(1);
        let Some(owned) = self.confined.last().copied() else {
            return self.goto(start);
        };
        if !owned.contains(start) {
            return self.goto(start);
        }
        self.emit_flat(owned)
    }

    /// Emits every unplaced block of `owned` with each of its edges explicit.
    ///
    /// Two levels deep at most, whatever the shape of the code: one statement
    /// per block, and every outgoing edge a goto. It says the same thing the
    /// structured form would have said, in the longest way there is.
    fn emit_flat(&mut self, owned: &BlockSet) -> Region {
        let mut parts = Vec::new();
        for node in owned.iter() {
            if self.visited.contains(node) {
                continue;
            }
            self.visited.insert(node);
            self.metrics.blocks = self.metrics.blocks.saturating_add(1);

            let arms = self.facts.arms(node);
            let region = match (arms.first(), arms.get(1)) {
                (None, _) => Region::Block(node),
                (Some((only, _)), None) => {
                    let jump = self.goto(*only);
                    Region::Seq(vec![Region::Block(node), jump])
                }
                (Some((taken, _)), Some((not_taken, _))) if arms.len() == 2 => {
                    let then_branch = self.goto(*taken);
                    let else_branch = self.goto(*not_taken);
                    self.metrics.ifs = self.metrics.ifs.saturating_add(1);
                    self.metrics.if_elses = self.metrics.if_elses.saturating_add(1);
                    Region::If(Box::new(IfRegion {
                        predicate: Predicate::Test {
                            block: node,
                            negated: false,
                        },
                        then_branch,
                        else_branch: Some(else_branch),
                    }))
                }
                _ => {
                    let mut cases = Vec::with_capacity(arms.len());
                    for (target, selectors) in &arms {
                        let body = self.goto(*target);
                        cases.push(SwitchCase {
                            selectors: selectors.clone(),
                            target: *target,
                            body,
                            values: Vec::new(),
                            default: false,
                            falls_through: false,
                        });
                    }
                    self.metrics.switches = self.metrics.switches.saturating_add(1);
                    Region::Switch(Box::new(SwitchRegion {
                        head: node,
                        cases,
                        follow: None,
                        tests: Vec::new(),
                    }))
                }
            };
            parts.push(region);
        }
        Region::sequence(parts)
    }

    /// Emits the region beginning at `start` and ending before `follow`.
    ///
    /// `skip_transfer` suppresses the `break`/`continue` test for the first
    /// block only, which is how a loop body may begin at its own header without
    /// immediately reading as `continue`.
    fn emit_sequence(
        &mut self,
        start: NodeId,
        follow: Option<NodeId>,
        skip_transfer: bool,
    ) -> Region {
        let mut sequence: Vec<Region> = Vec::new();
        let mut cursor = start;
        let mut first = true;

        loop {
            if Some(cursor) == follow {
                break;
            }

            if !(first && skip_transfer) {
                if let Some(transfer) = self.transfer(cursor) {
                    sequence.push(transfer);
                    break;
                }
                // A `do`/`while` tests at its latch, so a path that reaches
                // the latch runs the test again, which is what `continue`
                // writes. Only inside a region of the body: the body's own
                // walk, which is bounded by nothing, is what places the latch
                // and so the test.
                if follow.is_some()
                    && let Some(region) = self.latch_continue(cursor)
                {
                    sequence.push(region);
                    break;
                }
                if self.enclosing.contains(&cursor) {
                    sequence.push(self.revisit(cursor, follow));
                    break;
                }
            }
            first = false;

            if self
                .confined
                .last()
                .is_some_and(|owned| !owned.contains(cursor))
            {
                sequence.push(self.goto(cursor));
                break;
            }

            if self.visited.contains(cursor) {
                sequence.push(self.revisit(cursor, follow));
                break;
            }

            if let Some(index) = self.take_protected(cursor) {
                let (region, resume) = self.emit_try(index, follow);
                sequence.push(region);
                match resume {
                    Some(next) => {
                        cursor = next;
                        continue;
                    }
                    None => break,
                }
            }

            // A header already on the frame stack is the loop we are inside,
            // reached as the first block of its own body. Re-entering it here
            // would not terminate.
            let entering_loop = self.facts.headers.contains_key(&cursor)
                && !self.frames.iter().any(|frame| frame.header == cursor);
            if entering_loop {
                let (region, resume) = self.emit_loop(cursor, follow);
                sequence.push(region);
                match resume {
                    Some(next) => {
                        cursor = next;
                        continue;
                    }
                    None => break,
                }
            }

            let arms = self.facts.arms(cursor);
            // A latch consumed as a loop condition keeps its statements as a
            // block; its branch is the loop's test, not a conditional.
            let is_tail = self
                .enclosing_frame()
                .is_some_and(|frame| frame.tail == Some(cursor));
            // A conditional or a dispatch owns the block it branches from, so
            // that a condition spanning several blocks has one home in the
            // tree. Its statements still print before the test.
            let owns_head = !is_tail && arms.len() >= 2;

            self.visited.insert(cursor);
            self.metrics.blocks = self.metrics.blocks.saturating_add(1);
            if !owns_head {
                sequence.push(Region::Block(cursor));
            }

            if is_tail {
                break;
            }

            match arms.len() {
                0 => break,
                1 => match arms.first() {
                    Some((target, _)) => {
                        cursor = *target;
                        continue;
                    }
                    None => break,
                },
                2 => {
                    let (region, resume) = match CaseTree::of(self, cursor, follow) {
                        Some(tree) => self.emit_case_switch(cursor, tree, follow),
                        None => self.emit_if(cursor, &arms, follow),
                    };
                    sequence.push(region);
                    match resume {
                        Some(next) => {
                            cursor = next;
                            continue;
                        }
                        None => break,
                    }
                }
                _ => {
                    let (region, resume) = self.emit_switch(cursor, &arms, follow);
                    sequence.push(region);
                    match resume {
                        Some(next) => {
                            cursor = next;
                            continue;
                        }
                        None => break,
                    }
                }
            }
        }

        Region::sequence(sequence)
    }

    /// Emits an arrival at a block whose home in the tree is elsewhere: one
    /// already placed, or the join of an enclosing region, which the region
    /// places once its arms are done.
    ///
    /// Ordinarily this is a goto: the block has a home in the tree and this
    /// path jumps to it. A tail is the exception ([`Self::tail`]): straight-line
    /// code that ends the function, or runs on to `follow`, where the region
    /// this arrival is in resumes, or to an enclosing loop's header or follow,
    /// which the copy leaves to by its `continue` or `break` ([`TailEnd`]), so
    /// placing it twice repeats it and changes
    /// nothing about what runs - and a `goto` to a bare `return`, or to the few
    /// statements before one, is strictly worse to read than the statements
    /// themselves, which is the single most common goto in compiled code. The
    /// second kind is the tail two arms of a chain of conditionals share while
    /// a third arm runs past it to the chain's end: no join of one branch
    /// bounds it, and every arm that reaches it carries its own copy. A copy
    /// of that kind is recorded ([`Attempt::rejoining_tails`](crate::analysis::structure::regions::Attempt::rejoining_tails)) so the next
    /// trial can offer the tail as a join instead, which places it once.
    ///
    /// The trade is size: a tail is placed again once per path that reaches
    /// it, so one that prints more than [`STATEMENTS_PER_GOTO`] statements is
    /// jumped to instead, as a split's copies are weighed.
    fn revisit(&mut self, target: NodeId, follow: Option<NodeId>) -> Region {
        // A `do`/`while` tests at its latch, so a path that comes back to a
        // latch already placed runs the test again, which is what `continue`
        // writes. The first arrival is how the latch is placed at all, and it
        // is not this: the walk reached a block it has emitted.
        if let Some(arrival) = self.latch_continue(target) {
            return arrival;
        }
        match self.tail(target, TailEnd::Region { follow }) {
            Some(tail) => {
                let onward = tail
                    .last()
                    .and_then(|last| self.facts.successors(*last).next());
                if onward.is_some() {
                    self.rejoining_tails.insert(target);
                }
                let copy = self.repeat(tail);
                // A copy that runs on to a loop's header or follow leaves by
                // the `continue` or `break` that names it.
                match onward
                    .filter(|next| Some(*next) != follow)
                    .and_then(|next| self.transfer(next))
                {
                    Some(transfer) => Region::sequence(vec![copy, transfer]),
                    None => copy,
                }
            }
            None => self.goto(target),
        }
    }

    /// Places another copy of `tail`, counting each block it repeats.
    fn repeat(&mut self, tail: Vec<NodeId>) -> Region {
        self.metrics.replicated_tails = self.metrics.replicated_tails.saturating_add(tail.len());
        Region::sequence(tail.into_iter().map(Region::Block).collect())
    }

    /// Returns the straight-line path from `target` to the block that ends the
    /// function, or to where `end` lets it stop, when `target` begins one:
    /// every block on it has one successor but the last, which has none or
    /// runs on to a block `end` names.
    ///
    /// No block on it may begin a loop or a protected region, which have a
    /// structure of their own to keep, or lie in a protected region or handler
    /// at all: a copy placed outside the region would run unprotected. Nor
    /// may it print more than [`STATEMENTS_PER_GOTO`] statements: placed
    /// again, it would cost more than the goto it saves.
    pub fn tail(&self, target: NodeId, end: TailEnd) -> Option<Vec<NodeId>> {
        if matches!(end, TailEnd::Function) && !self.facts.endings.repeats(target) {
            return None;
        }
        let mut path = vec![target];
        let mut printed = self.facts.statements.of(target);
        let mut node = target;
        loop {
            if printed > STATEMENTS_PER_GOTO
                || self.facts.headers.contains_key(&node)
                || self.protected.contains_key(&node)
                || self.facts.guarded.contains(node)
            {
                return None;
            }
            let mut successors = self.facts.successors(node);
            match (successors.next(), successors.next()) {
                (None, _) => return Some(path),
                (Some(next), None) if self.ends_tail(next, end) => return Some(path),
                // A path longer than the graph has blocks runs round a cycle.
                (Some(next), None) if path.len() < self.facts.node_count => {
                    path.push(next);
                    printed = printed.saturating_add(self.facts.statements.of(next));
                    node = next;
                }
                _ => return None,
            }
        }
    }

    /// Whether a copied tail running on to `next` ends there ([`TailEnd`]).
    fn ends_tail(&self, next: NodeId, end: TailEnd) -> bool {
        match end {
            TailEnd::Function => false,
            TailEnd::Region { follow } => Some(next) == follow || self.is_transfer(next),
        }
    }

    /// Records `target` as needing a label.
    fn goto(&mut self, target: NodeId) -> Region {
        self.labels.insert(target);
        Region::Goto(target)
    }

    /// Returns the loop transfer `node` represents, if it is one.
    ///
    /// Enclosing loops are tested innermost first, so a block that is both the
    /// inner header and an outer follow continues the inner loop. Every frame
    /// is scanned, barrier or not: a handler is emitted lexically inside the
    /// loop body its protected region sits in, so a `break` or `continue` from
    /// it is legal.
    fn transfer(&mut self, node: NodeId) -> Option<Region> {
        // A switch no loop has been entered inside of captures a `break`:
        // a path to where it resumes leaves it so.
        if let Some(open) = self.switches.last()
            && open.breaks_at(node, self.frames.len())
        {
            self.metrics.breaks = self.metrics.breaks.saturating_add(1);
            return Some(Region::SwitchBreak(open.head));
        }
        let (index, matched) =
            self.frames
                .iter()
                .enumerate()
                .rev()
                .find_map(|(index, frame)| {
                    if frame.continued_at(node) {
                        Some((index, Region::Continue(frame.header)))
                    } else if Some(node) == frame.follow {
                        Some((index, Region::Break(frame.header)))
                    } else {
                        None
                    }
                })?;

        // C's `break` leaves the innermost loop only, so leaving an enclosing
        // one is a goto; a tail the exit reaches is repeated instead.
        if matches!(matched, Region::Break(_))
            && index.saturating_add(1) < self.frames.len()
            && let Some(tail) = self.tail(node, TailEnd::Function)
        {
            return Some(self.repeat(tail));
        }

        match matched {
            Region::Continue(_) => {
                self.metrics.continues = self.metrics.continues.saturating_add(1)
            }
            _ => self.metrics.breaks = self.metrics.breaks.saturating_add(1),
        }
        Some(matched)
    }

    /// Returns what a path to an enclosing `do`/`while`'s latch is, or `None`
    /// where `node` is no enclosing loop's latch: the `continue` that runs
    /// the loop's test again.
    ///
    /// `continue` runs the test alone, and a latch that computes more than
    /// its test prints those statements at the end of the body, where this
    /// path does not pass. So the path places a copy of them before its
    /// `continue`, unless the latch lies in a protected region or handler,
    /// where a copy placed outside it would run unprotected, or prints more
    /// than [`STATEMENTS_PER_GOTO`] statements, more than the goto it saves is
    /// worth: the path jumps to the latch the body places instead. So does a
    /// path from inside a nested loop: a `continue` of an enclosing loop is
    /// written as a jump to its header, which a `do`/`while` does not test at.
    fn latch_continue(&mut self, node: NodeId) -> Option<Region> {
        let (index, header) = self
            .frames
            .iter()
            .enumerate()
            .rev()
            .find(|(_, frame)| frame.tail == Some(node))
            .map(|(index, frame)| (index, frame.header))?;
        if index.saturating_add(1) < self.frames.len() {
            return Some(self.goto(node));
        }
        if self.facts.condition_only.contains(node) {
            self.metrics.continues = self.metrics.continues.saturating_add(1);
            return Some(Region::Continue(header));
        }
        if self.protected.contains_key(&node)
            || self.facts.guarded.contains(node)
            || self.facts.statements.of(node) > STATEMENTS_PER_GOTO
        {
            return Some(self.goto(node));
        }
        self.metrics.continues = self.metrics.continues.saturating_add(1);
        let statements = self.repeat(vec![node]);
        Some(Region::sequence(vec![statements, Region::Continue(header)]))
    }

    /// Returns whether a path to `node` leaves by a loop's `continue` or
    /// `break` on the frame stack ([`LoopFrame::continued_at`]), or by the
    /// `break` of the switch it is in ([`OpenSwitch::breaks_at`]).
    fn is_transfer(&self, node: NodeId) -> bool {
        self.switches
            .last()
            .is_some_and(|open| open.breaks_at(node, self.frames.len()))
            || self
                .frames
                .iter()
                .any(|frame| frame.continued_at(node) || Some(node) == frame.follow)
    }

    /// Returns the tail offered as the join of the branch at `node`, if one
    /// is and this walk can place it there.
    ///
    /// The branch must dominate the tail: every path to it passes through
    /// this branch, so placing it after the branch runs it on no path that
    /// did not reach it, and an arm that does not reach it ends in the
    /// transfer or the return it already prints. That alone decides which
    /// branch of a nest owns the tail - an inner branch does not dominate a
    /// tail an outer arm also reaches. The walk must not have placed the
    /// tail already, the confinement and the enclosing loop must hold it,
    /// and an enclosing loop's header or follow is no tail: the transfers
    /// that name it are what print `break` and `continue`. No loop holding
    /// the tail may leave this branch out either, or placing the tail after
    /// the branch would put a block of the loop outside it - unless the tail
    /// is that loop's header, where the whole loop follows the branch.
    ///
    /// Of several, the tail the most paths run on to, and the first of those
    /// the graph holds.
    fn preferred(&self, node: NodeId) -> Option<NodeId> {
        if self.facts.preferred_joins.count() == 0 {
            return None;
        }
        self.facts
            .preferred_joins
            .iter()
            .filter(|tail| {
                *tail != node
                    && self.facts.dominators.dominates(node, *tail)
                    && !self.visited.contains(*tail)
                    && !self.is_transfer(*tail)
                    && self
                        .confined
                        .last()
                        .is_none_or(|owned| owned.contains(*tail))
                    && self
                        .enclosing_frame()
                        .is_none_or(|frame| frame.holds(*tail))
                    && self.facts.loops.loops().iter().all(|info| {
                        !info.contains(*tail) || info.contains(node) || info.header == *tail
                    })
            })
            .max_by_key(|tail| (self.facts.predecessor_count(*tail), Reverse(tail.index())))
    }

    /// Returns the block the arms of a branch at `node` run on to: where they
    /// rejoin ([`Self::rejoin`]), or a tail they all run on to
    /// ([`Self::preferred`]) where they would otherwise rejoin at the
    /// enclosing loop's own test, only past the loop, or nowhere at all.
    pub fn join(&self, node: NodeId, follow: Option<NodeId>) -> Option<NodeId> {
        let joined = self.rejoin(node, follow);
        // A tail several arms run on to is the join only where the branch
        // would otherwise rejoin at the loop's own test, past the loop it is
        // in, at code that ends the function, or nowhere: those are the arms
        // that meet at the tail with one path leaving instead, and the path
        // that leaves runs the test again ([`Self::latch_continue`]), leaves
        // the loop with its own `continue`, `break` or return, or carries its
        // own copy of the code that ends the function ([`Self::revisit`]). A branch whose arms meet at a block
        // of their own inside the loop keeps that join, which is nearer than
        // any tail - taken over it, the tail would leave that nearer block
        // reached by a jump.
        let comes_round = match (joined, self.enclosing_frame()) {
            (None, _) => true,
            (Some(join), Some(frame)) => {
                Some(join) == frame.tail || (frame.holds(node) && !frame.holds(join))
            }
            (Some(join), None) => self.facts.endings.repeats(join),
        };
        if comes_round && let Some(tail) = self.preferred(node) {
            return Some(tail);
        }
        joined
    }

    /// The block a switch at `node` resumes at ([`Self::join`]), or a tail
    /// most of its arms meet at that runs on to that block.
    ///
    /// C writes such a switch with the tail after it and a goto past the
    /// tail for the arms that skip it: dash's `command` checks the closing
    /// token after its `switch`, and its `case` arm jumps past the check to
    /// the redirections. Resumed at the tail, the arms that reach it break
    /// out of the switch into it and only those reaching the join some other
    /// way jump; resumed at the join, the tail goes in the first arm reaching
    /// it and every other arm jumps to it. Only with
    /// [`Inputs::switch_tails`](crate::analysis::structure::inputs::Inputs::switch_tails),
    /// and of the tails offered on trial ([`Self::preferred`]).
    pub fn switch_join(&self, node: NodeId, follow: Option<NodeId>) -> Option<NodeId> {
        let joined = self.join(node, follow);
        if !self.facts.switch_tails {
            return joined;
        }
        match (joined, self.preferred(node)) {
            (Some(join), Some(tail)) if tail != join && self.facts.runs_on_to(tail, join) => {
                Some(tail)
            }
            _ => joined,
        }
    }

    /// Returns the block both arms of a branch at `node` rejoin at, if that
    /// block can bound the arms.
    ///
    /// The join is the immediate post-dominator, rejected when it is the
    /// virtual exit (the arms never rejoin), when it has already been placed
    /// (the arms will reach it by goto), or when it lies outside the enclosing
    /// loop other than as that loop's own follow (using it would drag the code
    /// after the loop inside it).
    ///
    /// Inside a `do`/`while` body the arms rejoin at the latch instead, when
    /// the post-dominator is the loop's exit or nothing in the loop: every
    /// path that stays in a single-latch loop runs into its test, and one that
    /// leaves some other way prints its own `break`, `continue` or return.
    /// Joined at the exit, an arm placed the latch, the walk resumed at the
    /// exit, and the body ended in a `break` the test after it never ran past.
    /// Likewise a branch in a region that ends inside its loop rejoins at that
    /// end when the post-dominator lies past the loop, and one in a region
    /// that ends inside its switch when the post-dominator is where the
    /// switch resumes.
    fn rejoin(&self, node: NodeId, follow: Option<NodeId>) -> Option<NodeId> {
        let post_dominator = self.facts.post_dominator(node);
        // Inside a loop whose arms meet only once they have left it, the arms
        // that stay meet where the loop comes round again; the others break.
        if let Some(frame) = self.enclosing_frame()
            && frame.holds(node)
            && post_dominator.is_none_or(|join| Some(join) == frame.follow || !frame.holds(join))
        {
            if let Some(rejoin) = frame.continuation.meet(node)
                && rejoin != node
                && !self.visited.contains(rejoin)
                && self
                    .confined
                    .last()
                    .is_none_or(|owned| owned.contains(rejoin))
            {
                return Some(rejoin);
            }
            if let Some(tail) = frame.tail
                && tail != node
                && !self.visited.contains(tail)
                && follow.is_none()
            {
                return Some(tail);
            }
        }
        // Arms that meet only where the function ends - one returns, or calls
        // what never returns, on every path - meet at the region's end when an
        // arm runs on to it, in the recovery that tries it
        // ([`Inputs::fall_out`]): the others leave by their own return, and
        // that arm falls out of the branch to where the region ends, as
        // `kill`'s signal loop is reached from the arms of its option parsing
        // that do not print and return.
        if self.facts.fall_out
            && post_dominator.is_none()
            && let Some(bound) = follow
            && bound != node
            && !self.visited.contains(bound)
            && self.facts.reachable_from(node).contains(bound)
        {
            return Some(bound);
        }
        // `immediate_post_dominator` answers `None` for the virtual exit, so a
        // real block index is all that reaches the guards below.
        let join = post_dominator?;
        if join == node {
            return None;
        }
        if Some(join) == follow {
            return Some(join);
        }
        // Inside a region of a `do`/`while` body, the latch is the loop's own
        // business and the region ends where it was told to: the arms rejoin
        // at that follow, and an arm that runs to the latch instead writes
        // the `continue` it is ([`Self::latch_continue`]). Taken as the join
        // here, the latch would carry the rest of the body into this region
        // and leave the block the region does end at reached by a jump. The
        // same holds of a join past the loop, for a branch in a region that
        // ends inside it: every arm that stays in the loop stops at the
        // region's end, and the others leave the loop by their own transfer,
        // so the arms meet at that end and nowhere nearer.
        if let Some(bound) = follow
            && self.enclosing_frame().is_some_and(|frame| {
                frame.tail == Some(join)
                    || (frame.holds(node) && frame.holds(bound) && !frame.holds(join))
            })
        {
            return Some(bound);
        }
        // Nor is where the switch the branch is in resumes, for a branch in a
        // region that ends before it - an arm that runs on into the next arm:
        // the paths that reach it leave by the switch's `break`, and the
        // others meet at the region's end.
        if let Some(bound) = follow
            && bound != join
            && self
                .switches
                .last()
                .is_some_and(|open| open.breaks_at(join, self.frames.len()))
        {
            return Some(bound);
        }
        // Nor is code that ends the function a join past the region's end: an
        // arm that runs into it before the region ends carries its own copy
        // ([`Self::revisit`]), so the arms meet where the region does. Code
        // that runs on into the region's end is no such join: it lies inside
        // the region, ahead of where it ends.
        if let Some(bound) = follow
            && self.facts.endings.repeats(join)
            && !self.facts.endings.runs_through(join, bound)
        {
            return Some(bound);
        }
        if self.visited.contains(join) {
            return None;
        }
        if self
            .confined
            .last()
            .is_some_and(|owned| !owned.contains(join))
        {
            return None;
        }
        // A join outside the loop is the loop's own business, except where the
        // branch has already left it: code past an exit that only that exit
        // reaches is placed in the body like any other, and keeps its joins.
        if let Some(frame) = self.enclosing_frame()
            && !frame.holds(join)
            && Some(join) != frame.follow
            && (frame.holds(node) || !self.facts.dominators.dominates(node, join))
        {
            return None;
        }
        Some(join)
    }

    /// Returns whether walking `node` next would place exactly one
    /// function-ending block.
    ///
    /// Every reason `emit_sequence` would do something else is ruled out here,
    /// so the answer is what the walk would produce and not a guess at it.
    fn ends_the_function(&self, node: NodeId, follow: Option<NodeId>) -> bool {
        Some(node) != follow
            && !self.visited.contains(node)
            && !self.is_transfer(node)
            && !self.enclosing.contains(&node)
            && !self.facts.headers.contains_key(&node)
            && !self.protected.contains_key(&node)
            && self
                .confined
                .last()
                .is_none_or(|owned| owned.contains(node))
            && self.facts.successors(node).next().is_none()
    }

    /// Emits a two-way branch at `head` and returns the block to resume at.
    ///
    /// An arm that cannot fall through ends the conditional: the other arm is
    /// returned as the resume point rather than walked as an `else`, so
    /// `if (c) return; ..` stays one statement deep however many times it
    /// repeats. That is what keeps the early-return spine - a few hundred
    /// sequential `if (x) return;` tests, ordinary in decompiled code - off the
    /// nesting bound, which is then a backstop rather than a limit real input
    /// reaches.
    fn emit_if(
        &mut self,
        head: NodeId,
        arms: &[(NodeId, Vec<usize>)],
        follow: Option<NodeId>,
    ) -> (Region, Option<NodeId>) {
        let join = self.join(head, follow);
        let Some(merged) = MergedCondition::merge(
            self,
            head,
            arms,
            self.facts.max_depth.saturating_sub(self.depth),
        ) else {
            return (Region::Empty, None);
        };

        for block in &merged.consumed {
            self.visited.insert(*block);
            self.metrics.blocks = self.metrics.blocks.saturating_add(1);
        }
        if !merged.consumed.is_empty() {
            self.metrics.merged_conditions = self.metrics.merged_conditions.saturating_add(1);
        }
        if merged.truncated {
            self.metrics.depth_limited = self.metrics.depth_limited.saturating_add(1);
        }

        // The join `merge_condition` folded into the condition has no statement
        // position left to resume at; the arms still stop there, which is what
        // makes an arm looping back to it end cleanly instead of jumping.
        let resume = join.filter(|node| !merged.consumed.contains(node));
        let scope = Scope {
            confinement: None,
            follow,
            barrier: Barrier::Inherit,
        };

        // The failing arm ends the function. Emitting it first inverts the
        // condition and leaves the taken arm as the parent's continuation,
        // which is both the shorter output and the one that does not nest.
        if join.is_none() && self.ends_the_function(merged.not_taken, follow) {
            let otherwise = self.in_scope(scope, |walk| walk.emit(merged.not_taken, join, false));
            self.metrics.ifs = self.metrics.ifs.saturating_add(1);
            let region = Region::If(Box::new(IfRegion {
                predicate: merged.predicate.negate(),
                then_branch: otherwise,
                else_branch: None,
            }));
            return (region, Some(merged.taken));
        }

        let taken = self.in_scope(scope, |walk| walk.emit(merged.taken, join, false));
        if self.facts.leaves_the_sequence(&taken) {
            self.metrics.ifs = self.metrics.ifs.saturating_add(1);
            let region = Region::If(Box::new(IfRegion {
                predicate: merged.predicate,
                then_branch: taken,
                else_branch: None,
            }));
            return (region, Some(merged.not_taken));
        }
        let not_taken = self.in_scope(scope, |walk| walk.emit(merged.not_taken, join, false));

        let region = match (taken, not_taken) {
            // Unreachable for any input the walk can produce, and lossy if it
            // were not: the conditional owns the blocks its condition spans.
            (Region::Empty, Region::Empty) => merged.predicate.statements(),
            // An empty `then` reads as a negated condition with one arm, which
            // is always better than `if (c) {} else { .. }`.
            (Region::Empty, other) => {
                self.metrics.ifs = self.metrics.ifs.saturating_add(1);
                Region::If(Box::new(IfRegion {
                    predicate: merged.predicate.negate(),
                    then_branch: other,
                    else_branch: None,
                }))
            }
            (then_branch, otherwise) => {
                self.metrics.ifs = self.metrics.ifs.saturating_add(1);
                let else_branch = (otherwise != Region::Empty).then_some(otherwise);
                if else_branch.is_some() {
                    self.metrics.if_elses = self.metrics.if_elses.saturating_add(1);
                }
                Region::If(Box::new(IfRegion {
                    predicate: merged.predicate,
                    then_branch,
                    else_branch,
                }))
            }
        };

        // Where both arms leave, nothing runs on past the conditional: a join
        // that is an enclosing loop's header or follow would only be the
        // `continue` or `break` no path reaches. A join that is a block of
        // its own is still placed here, for the paths that jump to it: a goto
        // that names it, or a way in the walk has yet to place. One whose
        // every way in is placed and arrives some other way - a copy of it
        // ending in the transfer it runs on to, as `argstr`'s loop latch is
        // copied into each arm that reaches it - is code no path runs.
        let leaves = self.facts.leaves(&region);
        let resume = resume.filter(|join| {
            !leaves
                || (!self.is_transfer(*join)
                    && (self.labels.contains(join)
                        || self
                            .facts
                            .graph
                            .predecessors(*join)
                            .any(|way| !self.visited.contains(way))))
        });
        (region, resume)
    }

    /// Places the arms of the switch at `head`, which resumes at `join`, and
    /// returns them in the order they print.
    ///
    /// An arm runs on into another where the other's first block is reached
    /// only by the dispatch and by that arm ([`Self::fall_into`]): the arm is
    /// placed straight before it, walked up to its first block, and falls
    /// through. Every other arm keeps its place, and a path from inside an
    /// arm to where the switch resumes is its `break` ([`OpenSwitch`]).
    /// `internal` names the tests a switch over a value consumed, which reach
    /// its arms as the dispatch does.
    fn emit_arms(
        &mut self,
        head: NodeId,
        arms: Vec<PendingArm>,
        join: Option<NodeId>,
        internal: &[NodeId],
    ) -> Vec<SwitchCase> {
        let targets: Vec<NodeId> = arms.iter().map(|arm| arm.target).collect();
        let into = self.fall_into(head, &targets, join, internal);
        // Each arm another falls into is placed after it; the rest keep the
        // order they came in, each followed by the chain it begins.
        let mut order = Vec::with_capacity(arms.len());
        let mut placed = vec![false; arms.len()];
        let entered: Vec<bool> = (0..arms.len())
            .map(|arm| into.contains(&Some(arm)))
            .collect();
        let starts = (0..arms.len()).filter(|arm| !entered.get(*arm).copied().unwrap_or(false));
        for start in starts.chain(0..arms.len()) {
            let mut at = start;
            while placed.get(at).is_some_and(|done| !done) {
                if let Some(done) = placed.get_mut(at) {
                    *done = true;
                }
                order.push(at);
                match into.get(at).copied().flatten() {
                    Some(next) if placed.get(next).is_some_and(|done| !done) => at = next,
                    _ => break,
                }
            }
        }
        self.switches.push(OpenSwitch {
            head,
            join,
            frames: self.frames.len(),
        });
        let mut cases = Vec::with_capacity(arms.len());
        for (position, &index) in order.iter().enumerate() {
            let Some(arm) = arms.get(index) else {
                continue;
            };
            let next = order
                .get(position.saturating_add(1))
                .copied()
                .filter(|next| into.get(index).copied().flatten() == Some(*next))
                .and_then(|next| arms.get(next))
                .map(|next| next.target);
            let body = self.emit(arm.target, next.or(join), false);
            if next.is_some() {
                self.metrics.fall_throughs = self.metrics.fall_throughs.saturating_add(1);
            }
            cases.push(SwitchCase {
                selectors: arm.selectors.clone(),
                target: arm.target,
                body,
                values: arm.values.clone(),
                default: arm.default,
                falls_through: next.is_some(),
            });
        }
        self.switches.pop();
        cases
    }

    /// For each of a switch's arm `targets`, the arm it can run on into: one
    /// whose first block every path reaches from the dispatch at `head`,
    /// from a test of `internal`, or from blocks only this arm reaches,
    /// stopping at the arms' first blocks, at `join` and at every transfer.
    /// That block is no loop header and no protected block, and not `join`
    /// itself.
    fn fall_into(
        &self,
        head: NodeId,
        targets: &[NodeId],
        join: Option<NodeId>,
        internal: &[NodeId],
    ) -> Vec<Option<usize>> {
        let outer = self.enclosing.last().copied();
        let reach: Vec<BlockSet> = targets
            .iter()
            .map(|own| {
                let mut seen = BlockSet::new(self.facts.node_count);
                let mut stack = vec![*own];
                while let Some(node) = stack.pop() {
                    let stops = (node != *own && targets.contains(&node))
                        || Some(node) == join
                        || Some(node) == outer
                        || node == head
                        || self.visited.contains(node)
                        || self.is_transfer(node);
                    if stops || !seen.insert(node) {
                        continue;
                    }
                    stack.extend(self.facts.successors(node));
                }
                seen
            })
            .collect();
        let limit = self.facts.node_count;
        (0..targets.len())
            .map(|arm| {
                let own = reach.get(arm)?;
                targets.iter().enumerate().find_map(|(other, into)| {
                    if other == arm
                        || Some(*into) == join
                        || self.facts.headers.contains_key(into)
                        || self.protected.contains_key(into)
                        || self.facts.guarded.contains(*into)
                    {
                        return None;
                    }
                    let mut predecessors = self
                        .facts
                        .graph
                        .predecessors(*into)
                        .filter(|predecessor| predecessor.index() < limit)
                        .peekable();
                    predecessors.peek()?;
                    let mut reached = false;
                    for predecessor in predecessors {
                        if predecessor == head || internal.contains(&predecessor) {
                            continue;
                        }
                        let alone = reach
                            .iter()
                            .enumerate()
                            .all(|(index, set)| index == arm || !set.contains(predecessor));
                        if !own.contains(predecessor) || !alone {
                            return None;
                        }
                        reached = true;
                    }
                    reached.then_some(other)
                })
            })
            .collect()
    }

    /// Emits the `switch` a case tree recovered at `head`, and returns the
    /// block to resume at.
    ///
    /// The tests below the root are placed by the switch, which consumes
    /// them. Each target takes one arm with every value that selects it, in
    /// ascending order of its lowest value, and the default takes the last arm
    /// unless it is where the switch resumes anyway.
    fn emit_case_switch(
        &mut self,
        head: NodeId,
        tree: CaseTree,
        follow: Option<NodeId>,
    ) -> (Region, Option<NodeId>) {
        for node in &tree.internal {
            self.visited.insert(*node);
            self.metrics.blocks = self.metrics.blocks.saturating_add(1);
        }
        let join = self.join(head, follow);
        let mut arms: Vec<(NodeId, Vec<i64>)> = Vec::new();
        for (value, target) in tree.cases {
            match arms.iter_mut().find(|(arm, _)| *arm == target) {
                Some((_, values)) => values.push(value),
                None => arms.push((target, vec![value])),
            }
        }
        let scope = Scope {
            confinement: None,
            follow,
            barrier: Barrier::Inherit,
        };
        let default = tree.default;
        let mut pending: Vec<PendingArm> = arms
            .into_iter()
            .map(|(target, values)| PendingArm {
                target,
                selectors: Vec::new(),
                values,
                default: false,
            })
            .collect();
        if Some(default) != join {
            pending.push(PendingArm {
                target: default,
                selectors: Vec::new(),
                values: Vec::new(),
                default: true,
            });
        }
        let internal = tree.internal.clone();
        let cases = self.in_scope(scope, |walk| {
            let mut cases = walk.emit_arms(head, pending, join, &internal);
            // An empty `default:` says nothing a missing one does not, unless
            // the arm before it runs on into it.
            if let Some(index) = cases
                .iter()
                .position(|case| case.default && case.body == Region::Empty)
                && !index
                    .checked_sub(1)
                    .and_then(|before| cases.get(before))
                    .is_some_and(|before| before.falls_through)
            {
                cases.remove(index);
            }
            cases
        });
        self.metrics.switches = self.metrics.switches.saturating_add(1);
        self.metrics.value_switches = self.metrics.value_switches.saturating_add(1);
        let region = Region::Switch(Box::new(SwitchRegion {
            head,
            cases,
            follow: join,
            tests: tree.internal,
        }));
        (region, join)
    }

    /// Emits a multi-way branch at `head` and returns the block to resume at.
    fn emit_switch(
        &mut self,
        head: NodeId,
        arms: &[(NodeId, Vec<usize>)],
        follow: Option<NodeId>,
    ) -> (Region, Option<NodeId>) {
        let join = self.switch_join(head, follow);
        let scope = Scope {
            confinement: None,
            follow,
            barrier: Barrier::Inherit,
        };

        let pending = arms
            .iter()
            .map(|(target, selectors)| PendingArm {
                target: *target,
                selectors: selectors.clone(),
                values: Vec::new(),
                default: false,
            })
            .collect();
        let cases = self.in_scope(scope, |walk| walk.emit_arms(head, pending, join, &[]));

        self.metrics.switches = self.metrics.switches.saturating_add(1);

        let region = Region::Switch(Box::new(SwitchRegion {
            head,
            cases,
            follow: join,
            tests: Vec::new(),
        }));
        (region, join)
    }

    /// Emits the protected region at `index` and returns the block to resume at.
    ///
    /// The body and each handler are walked confined to their own blocks, so a
    /// `leave` out of the region does not drag the code after it inside. A
    /// handler and a filter are entered by the runtime rather than by an edge,
    /// so they also raise the loop-frame barrier: the loop the region sits in
    /// does not bound their joins, although a `break` or `continue` out of them
    /// still names it.
    fn emit_try(&mut self, index: usize, follow: Option<NodeId>) -> (Region, Option<NodeId>) {
        let regions: &'a [ProtectedRegion] = self.facts.regions;
        let Some(region) = regions.get(index) else {
            return (Region::Empty, None);
        };
        let entry = region.entry;
        let blocks = &region.blocks;

        let inside = |node| blocks.contains(node);
        let body_follow = Exits::of(self, blocks.iter(), &inside).follow(self, entry, &inside);
        // A runtime may run a finally by jumping into it when the body
        // completes - PascalScript's does, and Delphi's try/finally falls into
        // its finally code - so the body's normal exit is the handler's entry.
        // That exit is the finally running, not the code after the `try`: the
        // body stops there, the handler runs to its own exit, and the region
        // resumes where the handler leaves.
        let finally_on_exit = region.handlers.iter().find(|handler| {
            matches!(handler.kind, ProtectedHandlerKind::Finally)
                && body_follow == Some(handler.entry)
        });
        let try_follow = match finally_on_exit {
            Some(finally) => {
                let inside = |node| finally.blocks.contains(node);
                Exits::of(self, finally.blocks.iter(), &inside).follow(self, finally.entry, &inside)
            }
            None => body_follow,
        };
        self.metrics.tries = self.metrics.tries.saturating_add(1);

        let outer = Scope {
            confinement: None,
            follow,
            barrier: Barrier::Inherit,
        };
        let recovered = self.in_scope(outer, |walk| {
            let body = walk.in_scope(
                Scope {
                    confinement: Some(blocks),
                    follow: None,
                    barrier: Barrier::Inherit,
                },
                |walk| walk.emit(entry, body_follow, false),
            );

            let mut handlers = Vec::with_capacity(region.handlers.len());
            for handler in &region.handlers {
                walk.metrics.handlers = walk.metrics.handlers.saturating_add(1);

                let filter = handler.kind.filter().map(|filter| {
                    walk.in_scope(
                        Scope {
                            confinement: Some(&filter.blocks),
                            follow: None,
                            barrier: Barrier::Raise,
                        },
                        |walk| walk.emit(filter.entry, None, false),
                    )
                });

                // A handler that completes normally resumes after the region,
                // exactly as the protected body does, so it ends there rather
                // than jumping there.
                let body = walk.in_scope(
                    Scope {
                        confinement: Some(&handler.blocks),
                        follow: None,
                        barrier: Barrier::Raise,
                    },
                    |walk| walk.emit(handler.entry, try_follow, false),
                );

                handlers.push(HandlerRegion {
                    kind: handler.kind.kind(),
                    entry: handler.entry,
                    filter,
                    body,
                });
            }

            TryRegion {
                entry,
                body,
                handlers,
                follow: try_follow,
            }
        });

        (Region::Try(Box::new(recovered)), try_follow)
    }

    /// Emits the loop headed at `header` and returns the block to resume at.
    fn emit_loop(&mut self, header: NodeId, follow: Option<NodeId>) -> (Region, Option<NodeId>) {
        let loops: &'a LoopForest = self.facts.loops;
        let Some(info) = self
            .facts
            .headers
            .get(&header)
            .copied()
            .and_then(|index| loops.loops().get(index))
        else {
            return (Region::Empty, None);
        };
        let body_set: &'a BitSet = &info.body;

        let inside = |node: NodeId| body_set.contains_checked(node.index());
        let loop_follow = Exits::of(self, body_set.iter().map(NodeId::new), &inside)
            .follow(self, header, &inside);
        let LoopShape {
            kind,
            predicate,
            tail,
            body_start,
        } = LoopShape::classify(&self.facts, header, body_set, &info.latches, loop_follow);

        self.metrics.loops = self.metrics.loops.saturating_add(1);
        if kind == StructuredLoop::Endless {
            self.metrics.endless_loops = self.metrics.endless_loops.saturating_add(1);
        }

        // A `while` consumes its header as the condition, so the header is
        // placed here rather than walked as an ordinary block of the body.
        if kind == StructuredLoop::While {
            self.visited.insert(header);
            self.metrics.blocks = self.metrics.blocks.saturating_add(1);
        }

        let scope = Scope {
            confinement: None,
            follow,
            barrier: Barrier::Inherit,
        };
        let body = self.in_scope(scope, |walk| {
            let continuation =
                Convergence::of_loop(walk.facts.graph, walk.facts.node_count, header, body_set);
            walk.frames.push(LoopFrame {
                header,
                body: body_set,
                follow: loop_follow,
                tail,
                continuation,
            });
            let body = match body_start {
                Some(start) => walk.emit(start, None, start == header),
                None => Region::Empty,
            };
            let body = LoopEnd::new(header).hoisted(body);
            let body = walk.drop_trailing_continue(body, header);
            walk.frames.pop();
            body
        });

        let region = Region::Loop(Box::new(LoopRegion {
            header,
            kind,
            predicate,
            body,
            follow: loop_follow,
        }));
        (region, loop_follow)
    }

    /// Rewrites every conditional with an arm that always leaves so that the
    /// other arm follows it instead of nesting in an `else`.
    ///
    /// Code after a conditional whose arm leaves runs only on the other arm's
    /// path, so `if (c) { ..; return; } else { X }` and
    /// `if (c) { ..; return; } X` run identically, and the second is the
    /// guard clause a reader expects: the exit stated and done with, the work
    /// at the level it belongs to. An `else` arm that leaves while the `then`
    /// arm does not is the guard, under the negated condition; when both
    /// leave, the shorter arm is the guard. A chain of such conditionals
    /// flattens into a sequence of guards.
    fn early_exits(&mut self, region: Region) -> Region {
        match region {
            Region::Seq(parts) => {
                let parts = parts
                    .into_iter()
                    .map(|part| self.early_exits(part))
                    .collect();
                Region::sequence(parts)
            }
            Region::If(inner) => {
                let IfRegion {
                    predicate,
                    then_branch,
                    else_branch,
                } = *inner;
                let then_branch = self.early_exits(then_branch);
                let Some(other) = else_branch.map(|other| self.early_exits(other)) else {
                    return Region::If(Box::new(IfRegion {
                        predicate,
                        then_branch,
                        else_branch: None,
                    }));
                };
                let (then_leaves, other_leaves) =
                    (self.facts.leaves(&then_branch), self.facts.leaves(&other));
                if !then_leaves && !other_leaves {
                    return Region::If(Box::new(IfRegion {
                        predicate,
                        then_branch,
                        else_branch: Some(other),
                    }));
                }
                let size = |region: &Region| {
                    let mut count = 0usize;
                    region.for_each_block(&mut |_| count = count.saturating_add(1));
                    count
                };
                let guard_is_then =
                    then_leaves && (!other_leaves || size(&then_branch) <= size(&other));
                let (predicate, guard, rest) = if guard_is_then {
                    (predicate, then_branch, other)
                } else {
                    (predicate.negate(), other, then_branch)
                };
                self.metrics.if_elses = self.metrics.if_elses.saturating_sub(1);
                Region::sequence(vec![
                    Region::If(Box::new(IfRegion {
                        predicate,
                        then_branch: guard,
                        else_branch: None,
                    })),
                    rest,
                ])
            }
            Region::Loop(mut inner) => {
                inner.body = self.early_exits(mem::replace(&mut inner.body, Region::Empty));
                Region::Loop(inner)
            }
            Region::Switch(mut inner) => {
                for case in &mut inner.cases {
                    case.body = self.early_exits(mem::replace(&mut case.body, Region::Empty));
                }
                Region::Switch(inner)
            }
            Region::Try(mut inner) => {
                inner.body = self.early_exits(mem::replace(&mut inner.body, Region::Empty));
                for handler in &mut inner.handlers {
                    handler.body = self.early_exits(mem::replace(&mut handler.body, Region::Empty));
                    handler.filter = handler.filter.take().map(|filter| self.early_exits(filter));
                }
                Region::Try(inner)
            }
            other => other,
        }
    }

    /// Removes every `continue` that falls in the loop body's tail position,
    /// where the loop repeats anyway and the statement says nothing.
    ///
    /// Tail position is not just the body's last statement. Both arms of a
    /// trailing `if` are in tail position, as is the single arm of one without
    /// an `else`, since the missing arm also falls off the end - so
    /// `if (c) { ..; continue; }` at the end of a body loses its `continue`
    /// too. A nested loop's tail belongs to that loop, so the walk stops there.
    fn drop_trailing_continue(&mut self, body: Region, header: NodeId) -> Region {
        let mut end = LoopEnd::new(header);
        let trimmed = end.trimmed(body);
        self.metrics.continues = self.metrics.continues.saturating_sub(end.removed);
        trimmed
    }
}
