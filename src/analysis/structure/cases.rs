//! A tree of value tests the walk recovers as one `switch`.

use std::{
    collections::{BTreeMap, BTreeSet},
    iter,
};

use crate::{
    analysis::structure::{CaseRelation, regions::Reading, walk::Structurer},
    graph::{GraphBase, NodeId, Predecessors, Successors},
};

/// A tree of value tests recovered as one `switch`.
pub struct CaseTree {
    /// Each case value and the block it selects, in ascending value order.
    pub cases: Vec<(i64, NodeId)>,

    /// The block every value no case names reaches.
    pub default: NodeId,

    /// The tests below the root, which the switch consumes.
    pub internal: Vec<NodeId>,
}

impl CaseTree {
    /// Recovers the `switch` a tree of value tests rooted at `root` was
    /// compiled from, when it is one.
    ///
    /// The tree is every test below `root` that reads the same value as it
    /// is entered with ([`CaseTest::entered`](crate::analysis::structure::CaseTest::entered)), does nothing else and is
    /// reached from nowhere else ([`Structurer::joins_case_tree`]).
    /// Tests that lead back to `root` are a loop rather than a tree, and no
    /// `switch` states them: a lap through them narrows nothing an `==` or a
    /// `!=` can cut, so walking their paths would never end. Each path through it narrows the value to an interval, so an equality
    /// test's case is kept only where its constant can reach it, and a path
    /// whose interval is empty is dead. The tree is a `switch` exactly when
    /// every path that matches no case ends in one block - the `default` - and
    /// it names at least three values; with fewer, the tests read best as the
    /// `if`s they are.
    ///
    /// A tree the `if`s state without a goto the `switch` would need stays the
    /// `if`s: an arm that leaves the enclosing loop, since `break` inside a
    /// `switch` leaves only the `switch`, and a root that sends its value
    /// straight to the join while every other value meets in a tail of its own,
    /// since that root is a guard around the rest rather than one of its cases.
    pub fn of<G: GraphBase + Successors + Predecessors>(
        walker: &Structurer<'_, G>,
        root: NodeId,
        follow: Option<NodeId>,
    ) -> Option<Self> {
        let first = *walker.facts.case_tests.get(&root)?;
        // The blocks of the tree, found first so the signedness every
        // ordering in it compares under is known before any interval is cut.
        let mut members = BTreeSet::from([root]);
        let mut walk = vec![root];
        while let Some(node) = walk.pop() {
            for successor in walker.facts.successors(node) {
                if successor == root {
                    return None;
                }
                let joins = walker
                    .facts
                    .case_tests
                    .get(&successor)
                    .is_some_and(|test| test.entered && test.value == first.value)
                    && walker.joins_case_tree(successor);
                if joins && members.insert(successor) {
                    walk.push(successor);
                }
            }
        }
        let mut signedness = members
            .iter()
            .filter_map(|node| walker.facts.case_tests.get(node))
            .filter_map(|test| test.relation.signed());
        let signed = match signedness.next() {
            Some(signed) => {
                if signedness.any(|other| other != signed) {
                    return None;
                }
                signed
            }
            None => true,
        };
        let reading = Reading { signed };
        let (floor, ceiling) = reading.bounds();

        let mut cases: BTreeMap<i64, NodeId> = BTreeMap::new();
        let mut defaults: BTreeSet<NodeId> = BTreeSet::new();
        let mut internal = Vec::new();
        let mut pending = vec![(root, floor, ceiling)];
        while let Some((node, low, high)) = pending.pop() {
            if low > high {
                continue;
            }
            let test = match walker.facts.case_tests.get(&node) {
                Some(test) if members.contains(&node) => *test,
                _ => {
                    defaults.insert(node);
                    continue;
                }
            };
            if node != root {
                internal.push(node);
            }
            let successors: Vec<NodeId> = walker.facts.successors(node).collect();
            let [first_arm, second_arm] = successors.as_slice() else {
                return None;
            };
            let (holds, fails) = match test.holds {
                0 => (*first_arm, *second_arm),
                1 => (*second_arm, *first_arm),
                _ => return None,
            };
            let constant = reading.value(test.constant);
            match test.relation {
                CaseRelation::Equal | CaseRelation::NotEqual => {
                    let (matched, unmatched) = if test.relation == CaseRelation::Equal {
                        (holds, fails)
                    } else {
                        (fails, holds)
                    };
                    if (low..=high).contains(&constant)
                        && cases
                            .insert(test.constant, matched)
                            .is_some_and(|earlier| earlier != matched)
                    {
                        return None;
                    }
                    pending.push((unmatched, low, high));
                }
                CaseRelation::Less { .. } => {
                    pending.push((holds, low, high.min(constant.saturating_sub(1))));
                    pending.push((fails, low.max(constant), high));
                }
                CaseRelation::LessOrEqual { .. } => {
                    pending.push((holds, low, high.min(constant)));
                    pending.push((fails, low.max(constant.saturating_add(1)), high));
                }
                CaseRelation::Greater { .. } => {
                    pending.push((holds, low.max(constant.saturating_add(1)), high));
                    pending.push((fails, low, high.min(constant)));
                }
                CaseRelation::GreaterOrEqual { .. } => {
                    pending.push((holds, low.max(constant), high));
                    pending.push((fails, low, high.min(constant.saturating_sub(1))));
                }
            }
        }
        let [default] = *defaults.iter().copied().collect::<Vec<_>>().as_slice() else {
            return None;
        };
        let cases: Vec<(i64, NodeId)> = cases
            .into_iter()
            .filter(|(_, target)| *target != default)
            .collect();
        if cases.len() < 3 {
            return None;
        }
        let tree = Self {
            cases,
            default,
            internal,
        };
        let join = walker.join(root, follow);
        (!walker.facts.guards_case_tree(root, &first, join) && !tree.arms_leave_loop(walker, join))
            .then_some(tree)
    }

    /// Whether some of the tree's arms leave the enclosing loop while others
    /// stay in it, which only a goto can say from inside a `switch`: `break`
    /// there leaves the `switch`, so the arms on one side reach their
    /// destination by goto whichever block the `switch` resumes at.
    fn arms_leave_loop<G: GraphBase + Successors + Predecessors>(
        &self,
        walker: &Structurer<'_, G>,
        join: Option<NodeId>,
    ) -> bool {
        let Some(frame) = walker.enclosing_frame() else {
            return false;
        };
        let Some(exit) = frame.follow else {
            return false;
        };
        let (mut leaves, mut stays) = (false, false);
        let mut seen: BTreeSet<NodeId> = self.internal.iter().copied().collect();
        let mut walk: Vec<NodeId> = self
            .cases
            .iter()
            .map(|(_, target)| *target)
            .chain(iter::once(self.default))
            .collect();
        while let Some(node) = walk.pop() {
            if node == exit {
                leaves = true;
            } else if Some(node) == join || node == frame.header || frame.tail == Some(node) {
                stays = true;
            } else if !walker.visited.contains(node) && seen.insert(node) {
                walk.extend(walker.facts.successors(node));
            }
        }
        leaves && stays
    }
}

impl<G: GraphBase + Successors + Predecessors> Structurer<'_, G> {
    /// Whether `node` can be a test inside a case tree: it does nothing but
    /// test, nothing but the tree reaches it, and it is not a block a walk must
    /// enter on its own terms.
    fn joins_case_tree(&self, node: NodeId) -> bool {
        self.facts.condition_only.contains(node)
            && self.facts.predecessor_count(node) == 1
            && !self.visited.contains(node)
            && !self.facts.headers.contains_key(&node)
            && !self.protected.contains_key(&node)
            && self
                .confined
                .last()
                .is_none_or(|owned| owned.contains(node))
            && self.enclosing_frame().is_none_or(|frame| frame.holds(node))
    }
}
