//! A loop body's end: the exit it states and the `continue`s it does not
//! need.

use std::mem;

use crate::{
    analysis::structure::{IfRegion, Predicate, Region, StructuredLoop, facts::Facts},
    bitset::BitSet,
    graph::{GraphBase, NodeId, Predecessors, Successors},
};

/// The end of the body of the loop at one header, which the walk rewrites
/// once the body is placed: an exit it can state first, and the `continue`s
/// in tail position, which say nothing.
pub struct LoopEnd {
    /// The loop's header, which its `continue` and `break` name.
    header: NodeId,

    /// How many `continue`s [`Self::trimmed`] removed.
    pub removed: usize,
}

impl LoopEnd {
    /// The end of the body of the loop at `header`, nothing removed yet.
    pub const fn new(header: NodeId) -> Self {
        Self { header, removed: 0 }
    }

    /// `body` restated, where it ends `if (c) { .. continue; } break;`, to
    /// end `if (!c) break; ..`.
    ///
    /// The two run identically, and the second is the shape a reader expects: a
    /// loop states its exit condition and then does its work, rather than
    /// wrapping the work in the negation of it. The form arises whenever a
    /// pre-tested loop's header does more than test - the header stays inside
    /// the body, its branch becomes a conditional, and the exit lands after it.
    ///
    /// Only an `if` with no `else` qualifies. With one, the `break` after it is
    /// reached from neither arm and the rewrite would not be the same loop.
    pub fn hoisted(&self, body: Region) -> Region {
        let Region::Seq(mut parts) = body else {
            return body;
        };
        if !matches!(parts.last(), Some(Region::Break(target)) if *target == self.header) {
            return Region::Seq(parts);
        }
        let Some(index) = parts.len().checked_sub(2) else {
            return Region::Seq(parts);
        };
        let Some(Region::If(branch)) = parts.get(index) else {
            return Region::Seq(parts);
        };
        if branch.else_branch.is_some() || !self.ends_with_continue(&branch.then_branch) {
            return Region::Seq(parts);
        }

        parts.pop();
        let Some(Region::If(branch)) = parts.pop() else {
            return Region::Seq(parts);
        };
        let IfRegion {
            predicate,
            then_branch,
            ..
        } = *branch;

        parts.push(Region::If(Box::new(IfRegion {
            predicate: predicate.negate(),
            then_branch: Region::Break(self.header),
            else_branch: None,
        })));
        parts.push(then_branch);
        Region::sequence(parts)
    }

    /// Whether `region` ends by continuing the loop.
    fn ends_with_continue(&self, region: &Region) -> bool {
        match region {
            Region::Continue(target) => *target == self.header,
            Region::Seq(parts) => parts
                .last()
                .is_some_and(|last| self.ends_with_continue(last)),
            _ => false,
        }
    }

    /// `region` with every `continue` of the loop in tail position removed,
    /// each counted in [`Self::removed`].
    pub fn trimmed(&mut self, region: Region) -> Region {
        match region {
            Region::Continue(target) if target == self.header => {
                self.removed = self.removed.saturating_add(1);
                Region::Empty
            }
            Region::Seq(mut regions) => match regions.pop() {
                Some(last) => {
                    regions.push(self.trimmed(last));
                    Region::sequence(regions)
                }
                None => Region::Empty,
            },
            Region::If(mut inner) => {
                inner.then_branch = self.trimmed(inner.then_branch);
                inner.else_branch = inner
                    .else_branch
                    .map(|arm| self.trimmed(arm))
                    .filter(|arm| *arm != Region::Empty);
                if inner.then_branch == Region::Empty && inner.else_branch.is_none() {
                    // The whole conditional was one redundant `continue`, so
                    // the branch itself says nothing about control flow - but
                    // the blocks its condition spans are placed by this `If`
                    // and nowhere else, so they stay.
                    return inner.predicate.statements();
                }
                Region::If(inner)
            }
            // An arm that falls through runs on into the next, so only the end
            // of one that leaves the switch is in tail position.
            Region::Switch(mut inner) => {
                for case in inner.cases.iter_mut().filter(|case| !case.falls_through) {
                    let body = mem::replace(&mut case.body, Region::Empty);
                    case.body = self.trimmed(body);
                }
                Region::Switch(inner)
            }
            // A `continue` at the end of a protected body still runs the
            // region's `finally` on the way out, so it is not implicit.
            Region::Try(inner) => Region::Try(inner),
            other => other,
        }
    }
}

/// Where a loop tests its continuation condition, and what that implies for
/// how its body is walked.
pub struct LoopShape {
    /// The recovered form.
    pub kind: StructuredLoop,

    /// The condition under which the loop continues, if it has one.
    pub predicate: Option<Predicate>,

    /// Latch consumed as the test, whose successors must not be walked.
    pub tail: Option<NodeId>,

    /// Block the body begins at.
    pub body_start: Option<NodeId>,
}

impl LoopShape {
    /// Classifies a loop by where it tests its continuation condition.
    ///
    /// A loop that tests nowhere structured is still a loop; it becomes
    /// [`StructuredLoop::Endless`] and its exits become `break`.
    pub fn classify<G: GraphBase + Successors + Predecessors>(
        facts: &Facts<'_, G>,
        header: NodeId,
        body: &BitSet,
        latches: &[NodeId],
        follow: Option<NodeId>,
    ) -> Self {
        // A header whose statements are more than the operand of its own test
        // can carry them into a `while` only as comma operands. A loop also
        // tested at its latch is the `do`/`while` with those statements at
        // the top of its body, and its header's exit one `break` there.
        let pre = Self::pre_tested(facts, header, body, follow);
        if pre.is_some()
            && !facts.condition_only.contains(header)
            && !facts.condition_operand.contains(header)
            && let Some(post) = Self::post_tested(facts, header, body, latches, follow)
        {
            return post;
        }
        pre.or_else(|| Self::post_tested(facts, header, body, latches, follow))
            .unwrap_or(LoopShape {
                kind: StructuredLoop::Endless,
                predicate: None,
                tail: None,
                body_start: Some(header),
            })
    }

    /// Recognises a loop whose header branches out of the body.
    ///
    /// Two conditions, and the second is the one that is easy to miss.
    ///
    /// The exit must be the block the loop resumes at. A header that branches
    /// somewhere else leaves by a path the `while` form cannot express, so the
    /// loop is not pre-tested even though its header does branch.
    ///
    /// And the header must compute **nothing but its own condition**. A `while`
    /// evaluates its condition before every iteration, but there is nowhere in
    /// the form to put a header's other statements except before the loop,
    /// where they run once. So a header that loads the value it tests, or
    /// computes one the body also reads, is not a `while` header: rendering it
    /// as one hoists that work out of the loop and produces code that does not
    /// loop. Such a loop is [`StructuredLoop::Endless`] instead, which puts the
    /// whole header inside the body and leaves by `break` - longer to read, and
    /// right.
    fn pre_tested<G: GraphBase + Successors + Predecessors>(
        facts: &Facts<'_, G>,
        header: NodeId,
        body: &BitSet,
        follow: Option<NodeId>,
    ) -> Option<Self> {
        // Either the header carries nothing that must be shown, or the caller
        // can show it inside the condition. A `while` puts its header nowhere
        // else, so those are the two ways one can exist.
        if !facts.condition_only.contains(header) && !facts.condition_expressible.contains(header) {
            return None;
        }
        let arms = facts.arms(header);
        if arms.len() != 2 {
            return None;
        }
        let (entry, selectors) = arms
            .iter()
            .find(|(target, _)| body.contains_checked(target.index()))?;
        let (exit, _) = arms
            .iter()
            .find(|(target, _)| !body.contains_checked(target.index()))?;

        // A header that re-enters itself is both the test and the work, and a
        // `while` has nowhere to put the work but inside its own condition.
        // That reads only when there is none: `condition_only` says the block
        // computes nothing the condition does not already carry, while
        // `condition_expressible` admits any block whose other rows happen to
        // print as statements - however many. Admitting the second here would
        // print an empty-bodied `while` with the whole body comma-folded into
        // its header. Anything more than the test belongs to `post_tested`,
        // which recognises the block as its own latch: a `do`/`while` that
        // prints the block as the body and puts the test underneath it.
        if *entry == header && !facts.condition_only.contains(header) {
            return None;
        }

        (Some(*exit) == follow).then(|| LoopShape {
            kind: StructuredLoop::While,
            predicate: Some(Predicate::Test {
                block: header,
                negated: selectors.first().copied() != Some(0),
            }),
            tail: None,
            // A header that re-enters itself is the whole loop: it is both the
            // condition and the work, and the condition already carries it.
            // Emitting the header as the body too would re-place a block that
            // has already been placed, which is a jump back into it - the
            // `for (;;) { goto header; }` shape.
            body_start: (*entry != header).then_some(*entry),
        })
    }

    /// Recognises a loop whose single latch branches out of the body.
    ///
    /// Several latches cannot be post-tested: there is no one place the
    /// condition lives, so the loop reads as endless with a `continue` per
    /// latch instead.
    ///
    /// The latch may be the header itself - a block that loops to itself. It
    /// is then the whole body and the test at its end, which is exactly a
    /// `do`/`while` over one block.
    fn post_tested<G: GraphBase + Successors + Predecessors>(
        facts: &Facts<'_, G>,
        header: NodeId,
        body: &BitSet,
        latches: &[NodeId],
        follow: Option<NodeId>,
    ) -> Option<Self> {
        let [latch] = latches else { return None };
        let arms = facts.arms(*latch);
        if arms.len() != 2 {
            return None;
        }
        let (_, selectors) = arms.iter().find(|(target, _)| *target == header)?;
        let (exit, _) = arms
            .iter()
            .find(|(target, _)| !body.contains_checked(target.index()))?;

        (Some(*exit) == follow).then(|| LoopShape {
            kind: StructuredLoop::DoWhile,
            predicate: Some(Predicate::Test {
                block: *latch,
                negated: selectors.first().copied() != Some(0),
            }),
            tail: Some(*latch),
            body_start: Some(header),
        })
    }
}
