//! The blocks a region leaves to, and the one it resumes at.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
};

use crate::{
    analysis::structure::walk::Structurer,
    graph::{GraphBase, NodeId, Predecessors, Successors},
};

/// The blocks a region leaves to, each with how many of its edges leave to
/// it: the candidates for the block control resumes at after the region.
pub struct Exits {
    /// Each block outside the region an edge of it reaches, with how many do.
    counts: BTreeMap<NodeId, usize>,
}

impl Exits {
    /// The exits of the region whose blocks are `members`, `inside` saying
    /// which blocks belong to it.
    pub fn of<G: GraphBase + Successors + Predecessors>(
        walker: &Structurer<'_, G>,
        members: impl Iterator<Item = NodeId>,
        inside: &impl Fn(NodeId) -> bool,
    ) -> Self {
        let mut counts: BTreeMap<NodeId, usize> = BTreeMap::new();
        for node in members {
            for successor in walker.facts.successors(node) {
                if !inside(successor) {
                    let slot = counts.entry(successor).or_insert(0);
                    *slot = slot.saturating_add(1);
                }
            }
        }
        Self { counts }
    }

    /// The block control resumes at after the region entered at `header`.
    ///
    /// A loop or a protected body may leave to several different blocks. One of
    /// them becomes the follow and the rest are reached by goto, so the choice
    /// decides how much of the function stays structured: an enclosing
    /// `do`/`while`'s latch comes first, then the block every exit path
    /// reaches: the header's nearest post-dominator outside the region. Failing
    /// that the most frequently targeted exit wins,
    /// breaking ties on the lowest id so the recovery does not depend on edge
    /// order.
    pub fn follow<G: GraphBase + Successors + Predecessors>(
        self,
        walker: &Structurer<'_, G>,
        header: NodeId,
        inside: &impl Fn(NodeId) -> bool,
    ) -> Option<NodeId> {
        let mut counts = self.counts;
        if counts.is_empty() {
            return None;
        }

        // An exit into an enclosing `do`/`while`'s latch is where that loop's
        // body goes on: resumed there, the latch is placed at the end of its
        // body and its test runs. Any other choice places the latch inside this
        // region, and the enclosing body ends in a `break` its test never runs
        // past - dash's `xwrite` retried `write` but never wrote the rest.
        if let Some(tail) = walker
            .frames
            .iter()
            .rev()
            .find_map(|frame| frame.tail.filter(|tail| counts.contains_key(tail)))
        {
            return Some(tail);
        }

        // The block every way out reaches, whether the loop leaves to it or
        // through a block standing for its exit edge
        // ([`StructureOptions::edge_blocks`]), which holds what that edge
        // writes and is placed with the `break` that takes it. Every other
        // choice weighs an edge block as the block it is, one more to place.
        // A header that does not itself leave the loop - its test is further
        // in, as `expmeta`'s scan of a `[` set tests the character it read -
        // is post-dominated first by blocks of the loop: every way out
        // reaches the first post-dominator past them, where that is a block
        // of every loop around this one. One past an enclosing loop is where
        // the function goes on after that loop too - `tilde_find_suffix` in
        // `which` returns from its inner scan or goes round the outer one,
        // and both end at its `return` - and resumed there, the way round
        // the enclosing loop would jump to its header.
        let first = walker.facts.post_dominator(header);
        let mut reached = first;
        for _ in 0..walker.facts.node_count {
            match reached {
                Some(node) if inside(node) => reached = walker.facts.post_dominator(node),
                _ => break,
            }
        }
        if let Some(candidate) = reached.filter(|node| {
            !inside(*node)
                && (reached == first
                    || walker
                        .frames
                        .iter()
                        .all(|frame| frame.body.contains(node.index())))
        }) && (counts
            .keys()
            .any(|&exit| exit == candidate || walker.facts.through_edge_block(exit) == candidate)
            || Self::runs_into(walker, header, &counts, candidate))
        {
            return Some(candidate);
        }

        // No exit catches every path out. An exit that is not the follow is
        // placed where the loop first leaves to it, and costs a goto for every
        // other edge that leaves to it and one more when its code runs on into
        // a block placed elsewhere ([`Structurer::exit_closed`]); a tail costs nothing,
        // since it is repeated. An exit the loop's header does not dominate is
        // entered from outside the loop too, so it is placed outside it, and
        // every edge the loop leaves to it by is a goto. The follow costs none
        // of that, so it is best spent on the exit that would cost the most:
        // only those stay candidates.
        let cost = |exit: NodeId, edges: usize| {
            if walker.facts.endings.repeats(exit) {
                0
            } else if !walker.facts.dominators.dominates(header, exit) {
                edges
            } else {
                edges
                    .saturating_sub(1)
                    .saturating_add(usize::from(!walker.exit_closed(exit)))
            }
        };
        let costs: BTreeMap<NodeId, usize> = counts
            .iter()
            .map(|(exit, edges)| (*exit, cost(*exit, *edges)))
            .collect();
        let costliest = costs.values().copied().max().unwrap_or(0);
        counts.retain(|exit, _| costs.get(exit) == Some(&costliest));

        // Among them, a header that can be the region's condition names one:
        // the block it leaves to when its test fails, which is what `while`
        // means. Preferring it over the exit with the most edges is what lets
        // the condition be written as the loop's own, instead of as an
        // `if ... break` inside an endless body - the same control flow
        // spelled longer.
        if let Some(tested) = walker.facts.tested_exit(header, &inside)
            && counts.contains_key(&tested)
        {
            return Some(tested);
        }

        counts
            .iter()
            .max_by_key(|(node, count)| (**count, Reverse(**node)))
            .map(|(node, _)| *node)
    }

    /// Whether the ways the region entered at `header` leaves by, `exits`,
    /// reach `candidate`, a block of every loop around the region, as more
    /// `break`s than any one exit would take: every one of them, or two or
    /// more. Such an exit is `candidate`, or a block standing for the edge to it
    /// ([`StructureOptions::edge_blocks`]), or a block of the region's own
    /// way out - entered only past `header` - that goes on to `candidate`
    /// alone, as the moves `expmeta`'s scan of a `[` set makes on its way out
    /// to the scan of the name it is nested in, entered from two of its
    /// tests, and the `]` arm that sets `metaflag`. Each is then placed with
    /// the `break` that takes it, and the region resumes where the loop
    /// around it goes on, each such way out a `break`. Where only one does
    /// and another goes elsewhere, that one costs its goto whichever block
    /// follows, and the exit the region tests for stays the follow, its code
    /// written after the loop. Past every loop, the block the ways out meet at is where the
    /// function goes on, and an exit's code that runs on there - an option
    /// loop's `return` arms - is no break of the region's.
    ///
    /// [`StructureOptions::edge_blocks`]: crate::analysis::structure::StructureOptions::edge_blocks
    fn runs_into<G: GraphBase + Successors + Predecessors>(
        walker: &Structurer<'_, G>,
        header: NodeId,
        exits: &BTreeMap<NodeId, usize>,
        candidate: NodeId,
    ) -> bool {
        if walker.frames.is_empty()
            || !walker
                .frames
                .iter()
                .all(|frame| frame.body.contains(candidate.index()))
        {
            return false;
        }
        let reaches = |exit: NodeId| {
            if exit == candidate || walker.facts.through_edge_block(exit) == candidate {
                return true;
            }
            let mut onward = walker.facts.successors(exit);
            matches!((onward.next(), onward.next()), (Some(next), None) if next == candidate)
                && walker.facts.dominators.dominates(header, exit)
        };
        let running = exits.keys().filter(|exit| reaches(**exit)).count();
        running == exits.len() || running >= 2
    }
}

impl<G: GraphBase + Successors + Predecessors> Structurer<'_, G> {
    /// Whether the code `exit` begins ends on its own: every block it reaches
    /// is one it dominates, or a tail ([`Structurer::tail`](crate::analysis::structure::walk::Structurer::tail)), so placed at the exit
    /// it needs no goto to anything placed elsewhere.
    fn exit_closed(&self, exit: NodeId) -> bool {
        let mut seen = BTreeSet::new();
        let mut walk = vec![exit];
        while let Some(node) = walk.pop() {
            if !seen.insert(node) || self.facts.endings.repeats(node) {
                continue;
            }
            if self.visited.contains(node) || !self.facts.dominators.dominates(exit, node) {
                return false;
            }
            walk.extend(self.facts.successors(node));
        }
        true
    }
}
