//! A condition the walk grows across the blocks that test it, and what it
//! selects between.

use crate::{
    analysis::structure::{Predicate, walk::Structurer},
    graph::{GraphBase, NodeId, Predecessors, Successors},
};

/// A condition grown across several blocks, and what it selects between.
pub struct MergedCondition {
    /// The recovered condition.
    pub predicate: Predicate,

    /// Block reached when the condition holds.
    pub taken: NodeId,

    /// Block reached when it does not.
    pub not_taken: NodeId,

    /// Blocks folded into the condition beyond the one it started at.
    pub consumed: Vec<NodeId>,

    /// Whether folding stopped at the fold bound rather than at a block that
    /// could not join the condition.
    pub truncated: bool,
}

/// One test folded into a growing condition.
struct ConditionStep {
    /// The test, holding exactly when control reaches [`ConditionStep::kept`].
    test: Predicate,

    /// Block the grown condition now selects on this side.
    kept: NodeId,

    /// The blocks the test folds: the candidate, then those of a compound
    /// test grown below it.
    blocks: Vec<NodeId>,
}

/// Conjoins two conditions, flattening rather than nesting.
fn all(left: Predicate, right: Predicate) -> Predicate {
    match left {
        Predicate::All(mut parts) => {
            parts.push(right);
            Predicate::All(parts)
        }
        other => Predicate::All(vec![other, right]),
    }
}

/// Disjoins two conditions, flattening rather than nesting.
fn any(left: Predicate, right: Predicate) -> Predicate {
    match left {
        Predicate::Any(mut parts) => {
            parts.push(right);
            Predicate::Any(parts)
        }
        other => Predicate::Any(vec![other, right]),
    }
}

impl MergedCondition {
    /// Grows the condition at `head` across every block that only tests, and
    /// returns the two blocks the whole condition selects between.
    ///
    /// A block qualifies as part of the condition when reaching it means the
    /// condition so far has already decided one way, it does nothing but test,
    /// and one of its own arms rejoins the block the condition already leads
    /// to. That is exactly the shape `&&` and `||` compile to, and the walk
    /// repeats so a chain of any length collapses into one condition - up to
    /// [`StructureOptions::max_depth`](crate::analysis::structure::StructureOptions::max_depth) folds, past which the rest of the chain
    /// recovers as further conditionals rather than as one predicate too deep
    /// to traverse.
    ///
    /// A step may be a condition of its own: `a || (b && c)` reaches `b`
    /// when `a` fails, and neither of `b`'s arms is the block `a` leads to, but
    /// `b && c` as a whole selects between that block and the one `a` fails
    /// to. `nesting` bounds how deep such conditions inside conditions go.
    pub fn merge<G: GraphBase + Successors + Predecessors>(
        walker: &Structurer<'_, G>,
        head: NodeId,
        arms: &[(NodeId, Vec<usize>)],
        nesting: usize,
    ) -> Option<Self> {
        let (taken, _) = arms.first()?;
        let (not_taken, _) = arms.get(1)?;

        let merged = (Self {
            predicate: Predicate::Test {
                block: head,
                negated: false,
            },
            taken: *taken,
            not_taken: *not_taken,
            consumed: Vec::new(),
            truncated: false,
        })
        .grown(walker, head, nesting);
        if !merged.consumed.is_empty() {
            return Some(merged);
        }
        // Where nothing folds, an arm that stands for its edge alone may be
        // the block it goes on to, its statements run ahead of the test
        // ([`Predicate::Hoisted`]): kept where a test then folds.
        for (edge, other, edge_taken) in [(*taken, *not_taken, true), (*not_taken, *taken, false)] {
            let Some(next) = walker.hoisting(head, edge) else {
                continue;
            };
            let (taken, not_taken) = if edge_taken {
                (next, other)
            } else {
                (other, next)
            };
            let hoisted = (Self {
                predicate: Predicate::Hoisted {
                    block: head,
                    negated: false,
                    edge,
                },
                taken,
                not_taken,
                consumed: vec![edge],
                truncated: false,
            })
            .grown(walker, head, nesting);
            if hoisted.consumed.len() > 1 {
                return Some(hoisted);
            }
        }
        Some(merged)
    }

    /// Grows `merged`, the condition at `head`, by every test that folds into
    /// it ([`Self::step`]), up to the fold bound.
    fn grown<G: GraphBase + Successors + Predecessors>(
        self,
        walker: &Structurer<'_, G>,
        head: NodeId,
        nesting: usize,
    ) -> Self {
        let mut merged = self;
        loop {
            if merged.consumed.len() >= walker.facts.max_depth {
                merged.truncated = true;
                break;
            }
            // Reaching `taken` means the condition so far held; a test there
            // whose other arm rejoins `not_taken` conjoins with it.
            if let Some(step) = merged.step(walker, head, merged.taken, merged.not_taken, nesting) {
                merged.consumed.extend(step.blocks);
                merged.predicate = all(merged.predicate, step.test);
                merged.taken = step.kept;
                continue;
            }
            // Symmetrically, a test on the failing side that rejoins `taken`
            // disjoins with it - but with the opposite polarity. `chain` reports
            // the test that holds when control reaches the block it keeps, and
            // here that block is the one the *whole* condition rejects, so what
            // joins the disjunction is its negation.
            if let Some(step) = merged.step(walker, head, merged.not_taken, merged.taken, nesting) {
                merged.consumed.extend(step.blocks);
                merged.predicate = any(merged.predicate, step.test.negate());
                merged.not_taken = step.kept;
                continue;
            }
            break;
        }
        merged
    }

    /// Returns the test `candidate` contributes to the condition at `head`
    /// grown so far (`merged`), which already leads to `shared`, when it
    /// contributes one.
    ///
    /// The returned test holds exactly when control at `candidate` reaches
    /// [`ConditionStep::kept`] rather than `shared`. When neither of
    /// `candidate`'s arms is `shared`, the condition grown from `candidate`
    /// may still select `shared` on one side, and is the test, while
    /// `nesting` allows a condition inside one.
    fn step<G: GraphBase + Successors + Predecessors>(
        &self,
        walker: &Structurer<'_, G>,
        head: NodeId,
        candidate: NodeId,
        shared: NodeId,
        nesting: usize,
    ) -> Option<ConditionStep> {
        let merged = self;
        if candidate == shared {
            return None;
        }
        if !walker.facts.condition_only.contains(candidate) {
            if !walker.facts.condition_expressible.contains(candidate) {
                return None;
            }
            // Statements besides the test join a condition only with
            // `fold_statements` on, or where they are the test's operand.
            if !walker.facts.fold_statements && !walker.facts.condition_operand.contains(candidate)
            {
                walker.declined_fold.set(true);
                return None;
            }
        }
        // Folding a block into a condition moves its text, not the paths it
        // runs on: short-circuit evaluation reaches the operand exactly when
        // control reached the block. That leaves the structural conditions -
        // nothing but the condition so far reaches it, it is not itself a
        // region a walk must enter (a loop header or a protected region), and
        // it belongs to whatever region the walk is currently confined to.
        // Every exit of the condition's blocks leads to its taken or its
        // not-taken block, so a candidate reached from those blocks alone is
        // reached exactly when the condition decides the way it stands for,
        // from one of them (`a && b`) or several (`(a && b) || c` reaches `c`
        // when either fails).
        let within = |node: NodeId| node == head || merged.consumed.contains(&node);
        let limit = walker.facts.node_count;
        let predecessors: Vec<NodeId> = walker
            .facts
            .graph
            .predecessors(candidate)
            .filter(|predecessor| predecessor.index() < limit)
            .collect();
        // A block reached from several of the condition's blocks that runs
        // statements of its own joins only where leaving it out costs a
        // goto: where its shared arm is where the conditional rejoins, it
        // nests as an `if` of its own (`if (a || b) { x = f(); if (x) ... }`)
        // and folded it would only put the statements in a condition.
        let nests = predecessors.len() > 1
            && !walker.facts.condition_only.contains(candidate)
            && walker.facts.post_dominator(head) == Some(shared);
        if predecessors.is_empty()
            || !predecessors.iter().copied().all(within)
            || nests
            || walker.facts.headers.contains_key(&candidate)
            || walker.protected.contains_key(&candidate)
            || walker.visited.contains(candidate)
        {
            return None;
        }
        if walker
            .confined
            .last()
            .is_some_and(|owned| !owned.contains(candidate))
        {
            return None;
        }
        // Nor is any loop's latch whose branch became that loop's test: it is
        // tested where the loop is, and folded into a condition inside the body
        // as well it would be tested twice, and every other path into it would
        // have no block to reach. Nor is a block on the other side of the
        // enclosing loop's boundary from the head: a test past the loop's exit
        // folded into one inside it would take the exit out of the loop's
        // structure. Code past an exit that the walk places in the body
        // ([`Self::rejoin`]) is on the same side as the tests it holds, and
        // folds among them like any other.
        if walker
            .frames
            .iter()
            .any(|frame| frame.holds(candidate) && frame.tail == Some(candidate))
            || walker
                .enclosing_frame()
                .is_some_and(|frame| frame.holds(head) != frame.holds(candidate))
        {
            return None;
        }

        let arms = walker.facts.arms(candidate);
        let (first, _) = arms.first()?;
        let (second, _) = arms.get(1)?;
        if arms.len() != 2 {
            return None;
        }

        // The arm that rejoins is the one the condition already accounts for;
        // the other is what the grown condition now selects.
        // An arm that stands for its edge alone and goes on to `shared` is
        // `shared` reached through what the edge writes, which the test then
        // runs ahead of itself ([`Predicate::Hoisted`]).
        let (kept, negated, edge) = if *second == shared {
            (*first, false, None)
        } else if *first == shared {
            (*second, true, None)
        } else if walker.hoisting(candidate, *second) == Some(shared) {
            (*first, false, Some(*second))
        } else if walker.hoisting(candidate, *first) == Some(shared) {
            (*second, true, Some(*first))
        } else {
            let inner = Self::merge(walker, candidate, &arms, nesting.checked_sub(1)?)?;
            if inner.consumed.is_empty() || inner.truncated {
                return None;
            }
            let (test, kept) = if inner.not_taken == shared {
                (inner.predicate, inner.taken)
            } else if inner.taken == shared {
                (inner.predicate.negate(), inner.not_taken)
            } else {
                return None;
            };
            let mut blocks = vec![candidate];
            blocks.extend(inner.consumed);
            return Some(ConditionStep { test, kept, blocks });
        };

        Some(match edge {
            None => ConditionStep {
                test: Predicate::Test {
                    block: candidate,
                    negated,
                },
                kept,
                blocks: vec![candidate],
            },
            Some(edge) => ConditionStep {
                test: Predicate::Hoisted {
                    block: candidate,
                    negated,
                    edge,
                },
                kept,
                blocks: vec![candidate, edge],
            },
        })
    }
}

impl<G: GraphBase + Successors + Predecessors> Structurer<'_, G> {
    /// Returns the block `arm`, an arm of the branch at `test`, goes on to,
    /// when `arm` stands for that edge alone and the condition at `test` may
    /// say its statements ahead of the test instead ([`Predicate::Hoisted`]):
    /// `arm` is one of [`StructureOptions::edge_blocks`](crate::analysis::structure::StructureOptions::edge_blocks), entered from `test`
    /// alone, no region a walk must enter, and the recovery says statements
    /// inside conditions ([`Inputs::fold_statements`](crate::analysis::structure::inputs::Inputs::fold_statements)).
    fn hoisting(&self, test: NodeId, arm: NodeId) -> Option<NodeId> {
        if !self.facts.edge_blocks.contains(arm)
            || self.visited.contains(arm)
            || self.facts.headers.contains_key(&arm)
            || self.protected.contains_key(&arm)
        {
            return None;
        }
        let mut predecessors = self.facts.graph.predecessors(arm);
        if predecessors.next() != Some(test) || predecessors.next().is_some() {
            return None;
        }
        let mut successors = self.facts.successors(arm);
        let (Some(next), None) = (successors.next(), successors.next()) else {
            return None;
        };
        // The edge's statements in a condition are statements there.
        if !self.facts.fold_statements {
            self.declined_fold.set(true);
            return None;
        }
        Some(next)
    }
}
