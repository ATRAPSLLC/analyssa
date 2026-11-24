//! Node splitting: copies the blocks a loop's second entry reaches, until
//! every loop is entered at one block, and the blocks a join dominates, for
//! each path but one into it.
//!
//! A loop entered at two blocks - compiled from a structured loop by rotation
//! or jump threading, or written with a goto into its body - is irreducible:
//! no block dominates the rest of it, so it is no natural loop, and structuring
//! by natural loops reaches the second entry only by goto. Copying what that
//! entry runs before it reaches the first gives the outside path its own blocks
//! that rejoin the loop at the first entry, which is a loop and an `if` ahead
//! of it.
//!
//! A copy is the same code as its original, so the copies stand for the
//! original wherever the recovered tree is read; [`SplitGraph::original`] maps
//! one to the other. A copied region carries every goto its own structure
//! needs, so a split pays only where the second entry cost more; the caller
//! measures that and keeps or declines each [`Split`].

use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
};

use crate::{
    analysis::structure::BlockSet,
    graph::{
        GraphBase, NodeId, Predecessors, Successors,
        algorithms::{compute_dominators, strongly_connected_components},
    },
};

/// A control-flow graph some of whose loops entered at more than one block
/// were split, by copying blocks of the graph it was made from, so that each is
/// entered at one.
#[derive(Clone)]
pub struct SplitGraph {
    /// Successors per node, in the order of the node they copy.
    successors: Vec<Vec<NodeId>>,

    /// Predecessors per node, one per edge.
    predecessors: Vec<Vec<NodeId>>,

    /// The node of the original graph each node is, or copies.
    original: Vec<NodeId>,

    /// How many nodes the original graph has; every node past them is a copy.
    originals: usize,

    /// The blocks routed to another block's test they end with
    /// ([`Self::repeat`]), by that test, as nodes of the original graph.
    repeats: BTreeMap<NodeId, NodeId>,
}

/// The split of one loop, for the caller to keep or decline.
pub struct Split {
    /// The original blocks the loop is entered at, which name the loop
    /// however many copies other splits have made of it.
    pub key: Vec<NodeId>,

    /// The graph with the loop split, or `None` when its copies would grow
    /// the graph past twice the original's size, which bounds what a chain of
    /// nested irreducible loops could otherwise compound.
    pub graph: Option<SplitGraph>,
}

impl SplitGraph {
    /// Whether the graph has a path from `from` to `to`, over its back edges
    /// like any other.
    pub fn reaches(&self, from: NodeId, to: NodeId) -> bool {
        let mut seen = vec![false; self.node_count()];
        let mut pending = vec![from];
        while let Some(node) = pending.pop() {
            for next in self.successors(node) {
                if next == to {
                    return true;
                }
                if let Some(flag) = seen.get_mut(next.index())
                    && !*flag
                {
                    *flag = true;
                    pending.push(next);
                }
            }
        }
        false
    }

    /// Returns `graph` with no loop split.
    pub fn of<G: Successors>(graph: &G) -> Self {
        let originals = graph.node_count();
        let successors = (0..originals)
            .map(|index| {
                graph
                    .successors(NodeId::new(index))
                    .filter(|successor| successor.index() < originals)
                    .collect()
            })
            .collect();
        Self::from_parts(
            successors,
            (0..originals).map(NodeId::new).collect(),
            originals,
            BTreeMap::new(),
        )
    }

    /// Returns the split of the next loop reachable from `entry` that is
    /// entered at more than one block, past those `declined` names.
    ///
    /// The loop keeps the entry whose split copies fewest blocks - the
    /// function's entry when it is one - and every other entry's outside edges
    /// move to a copy of what it reaches before that one. A loop holding a
    /// block of a protected region or handler (`guarded`) is never split: a
    /// copy outside its region would run unprotected.
    pub fn next(
        &self,
        entry: NodeId,
        guarded: &BlockSet,
        declined: &BTreeSet<Vec<NodeId>>,
    ) -> Option<Split> {
        let found = irreducible(&self.successors, entry, &self.original, guarded, declined)?;
        let (header, copies) = cheapest(&self.successors, &found.members, &found.entries, entry)?;
        if self.successors.len().saturating_add(copies) > self.originals.saturating_mul(2) {
            return Some(Split {
                key: found.key,
                graph: None,
            });
        }
        let mut successors = self.successors.clone();
        let mut original = self.original.clone();
        for other in found.entries.iter().filter(|other| **other != header) {
            copy_entry(
                &mut successors,
                &mut original,
                &found.members,
                header,
                *other,
            );
        }
        Some(Split {
            key: found.key,
            graph: Some(Self::from_parts(
                successors,
                original,
                self.originals,
                self.repeats.clone(),
            )),
        })
    }

    /// Returns the graph with the join at `target` split, or `None` where it
    /// cannot be: each of `target`'s predecessors but the lowest reaches a
    /// copy of the blocks `target` dominates, and, where `ending`, of the
    /// code they run into that ends the function, which rejoins the graph
    /// where it leaves that region; a predecessor the entry does not reach
    /// takes no part.
    ///
    /// A path reaches a dominated block only through `target`, so each copy
    /// runs what the original runs on its paths and nothing else. Code from
    /// which no cycle is reachable ends the function on every path, as a
    /// tail does, and runs the same whichever way it was entered: a copy
    /// taking it along rejoins nothing, where stopping at it would leave the
    /// copies to join there again (`outstr(p, out)` running into `outc`'s
    /// buffer test, which every arm's copy then enters). A region
    /// holding a cycle, the function's `entry` or a block of a protected
    /// region or handler (`guarded`) is not split: a cycle is a loop's
    /// business, and a copy outside its region would run unprotected. Nor is
    /// one whose copies would grow the graph past twice the original's size.
    pub fn split_join(
        &self,
        target: NodeId,
        entry: NodeId,
        guarded: &BlockSet,
        ending: bool,
    ) -> Option<Self> {
        if target == entry {
            return None;
        }
        let dominators = compute_dominators(self, entry);
        // Only paths from the entry: a block nothing reaches keeps nothing.
        let mut sources: Vec<NodeId> = self
            .predecessors
            .get(target.index())?
            .iter()
            .copied()
            .filter(|source| dominators.dominates(entry, *source))
            .collect();
        sources.sort_unstable();
        sources.dedup();
        let [_, others @ ..] = sources.as_slice() else {
            return None;
        };
        if others.is_empty() {
            return None;
        }
        // A block that loops to itself is a cycle the components of two or
        // more blocks leave out.
        let loops_to_itself = |node: &NodeId| {
            self.successors
                .get(node.index())
                .is_some_and(|targets| targets.contains(node))
        };
        let ends = if ending {
            self.ending(&loops_to_itself)
        } else {
            BTreeSet::new()
        };
        let mut region = BTreeSet::new();
        let mut walk = vec![target];
        while let Some(node) = walk.pop() {
            if (dominators.dominates(target, node) || ends.contains(&node)) && region.insert(node) {
                walk.extend(self.successors.get(node.index()).into_iter().flatten());
            }
        }
        // Carrying the ending code is another split only where it carries some.
        if ending
            && region
                .iter()
                .all(|node| dominators.dominates(target, *node))
        {
            return None;
        }
        let original_of = |node: &NodeId| self.original.get(node.index()).copied().unwrap_or(*node);
        if region.contains(&entry)
            || region
                .iter()
                .any(|node| guarded.contains(original_of(node)))
            || region.iter().any(loops_to_itself)
            || !Induced::of(&self.successors, region.iter().copied())
                .cycles()
                .is_empty()
        {
            return None;
        }
        let grown = self
            .successors
            .len()
            .saturating_add(region.len().saturating_mul(others.len()));
        if grown > self.originals.saturating_mul(2) {
            return None;
        }
        let mut successors = self.successors.clone();
        let mut original = self.original.clone();
        for source in others {
            let mut copy_of: BTreeMap<NodeId, NodeId> = BTreeMap::new();
            for node in &region {
                copy_of.insert(*node, NodeId::new(successors.len()));
                successors.push(Vec::new());
                original.push(original_of(node));
            }
            for (node, copy) in &copy_of {
                let targets: Vec<NodeId> = self
                    .successors
                    .get(node.index())
                    .into_iter()
                    .flatten()
                    .map(|next| copy_of.get(next).copied().unwrap_or(*next))
                    .collect();
                if let Some(row) = successors.get_mut(copy.index()) {
                    *row = targets;
                }
            }
            let entered = copy_of.get(&target).copied()?;
            if let Some(row) = successors.get_mut(source.index()) {
                for next in row.iter_mut().filter(|next| **next == target) {
                    *next = entered;
                }
            }
        }
        Some(Self::from_parts(
            successors,
            original,
            self.originals,
            self.repeats.clone(),
        ))
    }

    /// The blocks from which no cycle is reachable: every path from one ends
    /// the function. `loops_to_itself` says which blocks are a cycle alone.
    fn ending(&self, loops_to_itself: &impl Fn(&NodeId) -> bool) -> BTreeSet<NodeId> {
        let nodes = (0..self.successors.len()).map(NodeId::new);
        let mut cycling: Vec<NodeId> = Induced::of(&self.successors, nodes.clone())
            .cycles()
            .into_iter()
            .flatten()
            .chain(nodes.clone().filter(|node| loops_to_itself(node)))
            .collect();
        // Every block that reaches a cycle, walked back from the cycles.
        let mut reaching: BTreeSet<NodeId> = cycling.iter().copied().collect();
        while let Some(node) = cycling.pop() {
            for previous in self.predecessors.get(node.index()).into_iter().flatten() {
                if reaching.insert(*previous) {
                    cycling.push(*previous);
                }
            }
        }
        nodes.filter(|node| !reaching.contains(node)).collect()
    }

    /// Returns the graph of `successors`, whose nodes copy `original`.
    fn from_parts(
        successors: Vec<Vec<NodeId>>,
        original: Vec<NodeId>,
        originals: usize,
        repeats: BTreeMap<NodeId, NodeId>,
    ) -> Self {
        let mut predecessors: Vec<Vec<NodeId>> = vec![Vec::new(); successors.len()];
        for (index, targets) in successors.iter().enumerate() {
            for target in targets {
                if let Some(row) = predecessors.get_mut(target.index()) {
                    row.push(NodeId::new(index));
                }
            }
        }
        Self {
            successors,
            predecessors,
            original,
            originals,
            repeats,
        }
    }

    /// Returns the node of the original graph `node` is, or copies.
    pub fn original(&self, node: NodeId) -> NodeId {
        self.original.get(node.index()).copied().unwrap_or(node)
    }

    /// Returns how many nodes are copies.
    pub fn copies(&self) -> usize {
        self.successors.len().saturating_sub(self.originals)
    }

    /// Moves the edge in position `index` of `from`'s successors to `to`,
    /// keeping its position, which is what the branch's condition reads.
    pub fn reroute(&mut self, from: NodeId, index: usize, to: NodeId) {
        let Some(slot) = self
            .successors
            .get_mut(from.index())
            .and_then(|targets| targets.get_mut(index))
        else {
            return;
        };
        let old = mem::replace(slot, to);
        if let Some(row) = self.predecessors.get_mut(old.index())
            && let Some(position) = row.iter().position(|node| *node == from)
        {
            row.remove(position);
        }
        if let Some(row) = self.predecessors.get_mut(to.index()) {
            row.push(from);
        }
    }

    /// Makes `from`, which ends with the whole of `test`'s test, go to the
    /// test instead of branching itself, and records the route
    /// ([`Self::repeats`]): the rows `from` ends with are then the test's,
    /// which runs them, and a printer states them once, in the test.
    pub fn repeat(&mut self, from: NodeId, test: NodeId) {
        self.jump(from, test);
        self.repeats
            .insert(self.original(from), self.original(test));
    }

    /// The blocks routed to another block's test they end with
    /// ([`Self::repeat`]), by that test, as nodes of the original graph: a
    /// copy a split makes of one is routed as the block it copies is.
    pub fn repeats(&self) -> &BTreeMap<NodeId, NodeId> {
        &self.repeats
    }

    /// Makes `from` go to `to` alone, in place of every successor it had.
    fn jump(&mut self, from: NodeId, to: NodeId) {
        let Some(targets) = self.successors.get_mut(from.index()) else {
            return;
        };
        let old = mem::replace(targets, vec![to]);
        for target in old {
            if let Some(row) = self.predecessors.get_mut(target.index())
                && let Some(position) = row.iter().position(|node| *node == from)
            {
                row.remove(position);
            }
        }
        if let Some(row) = self.predecessors.get_mut(to.index()) {
            row.push(from);
        }
    }
}

impl GraphBase for SplitGraph {
    fn node_count(&self) -> usize {
        self.successors.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.successors.len()).map(NodeId::new)
    }
}

impl Successors for SplitGraph {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.successors
            .get(node.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }
}

impl Predecessors for SplitGraph {
    fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.predecessors
            .get(node.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }
}

/// The subgraph of a graph over some of its nodes: a member's edges to
/// members, and nothing else.
///
/// The members are renumbered densely, so a search over the view costs the size
/// of the subgraph and not the size of the graph.
struct Induced {
    /// The members, ascending; node `i` of the view is `members[i]`.
    members: Vec<NodeId>,

    /// Successors per member, as positions in [`Induced::members`].
    edges: Vec<Vec<NodeId>>,
}

impl Induced {
    /// Returns the subgraph of `successors` over `members`, which ascend.
    fn of(successors: &[Vec<NodeId>], members: impl Iterator<Item = NodeId>) -> Self {
        let members: Vec<NodeId> = members.collect();
        let edges = members
            .iter()
            .map(|member| {
                successors
                    .get(member.index())
                    .into_iter()
                    .flatten()
                    .filter_map(|successor| members.binary_search(successor).ok())
                    .map(NodeId::new)
                    .collect()
            })
            .collect();
        Self { members, edges }
    }

    /// Returns the cycles of the subgraph with more than one node.
    fn cycles(&self) -> Vec<BTreeSet<NodeId>> {
        strongly_connected_components(self)
            .into_iter()
            .filter(|component| component.len() > 1)
            .map(|component| {
                component
                    .into_iter()
                    .filter_map(|position| self.members.get(position.index()).copied())
                    .collect()
            })
            .collect()
    }
}

impl GraphBase for Induced {
    fn node_count(&self) -> usize {
        self.edges.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..self.edges.len()).map(NodeId::new)
    }
}

impl Successors for Induced {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.edges
            .get(node.index())
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }
}

/// A loop entered at more than one block.
struct Irreducible {
    /// The loop's blocks.
    members: BTreeSet<NodeId>,

    /// The blocks it is entered at.
    entries: Vec<NodeId>,

    /// The original blocks it is entered at, sorted; see [`Split::key`].
    key: Vec<NodeId>,
}

/// Finds a loop reachable from `entry` that is entered at more than one block,
/// past those `declined` names.
///
/// A loop entered at one block has that block for its header; the loops
/// nested inside it are the cycles that avoid the header, and are searched in
/// turn, with the header and every block outside them as their outside. A
/// declined loop is searched the same way, with the entry it would keep for
/// its header.
fn irreducible(
    successors: &[Vec<NodeId>],
    entry: NodeId,
    original: &[NodeId],
    guarded: &BlockSet,
    declined: &BTreeSet<Vec<NodeId>>,
) -> Option<Irreducible> {
    let original_of = |node: &NodeId| original.get(node.index()).copied().unwrap_or(*node);
    let mut reachable = BlockSet::new(successors.len());
    let mut walk = vec![entry];
    while let Some(node) = walk.pop() {
        if reachable.insert(node) {
            walk.extend(successors.get(node.index()).into_iter().flatten());
        }
    }
    let mut predecessors: Vec<Vec<NodeId>> = vec![Vec::new(); successors.len()];
    for node in reachable.iter() {
        for successor in successors.get(node.index()).into_iter().flatten() {
            if let Some(row) = predecessors.get_mut(successor.index()) {
                row.push(node);
            }
        }
    }

    let mut work = Induced::of(successors, reachable.iter()).cycles();
    while let Some(members) = work.pop() {
        if members
            .iter()
            .any(|member| guarded.contains(original_of(member)))
        {
            continue;
        }
        let entries: Vec<NodeId> = members
            .iter()
            .copied()
            .filter(|member| {
                *member == entry
                    || predecessors
                        .get(member.index())
                        .is_some_and(|from| from.iter().any(|source| !members.contains(source)))
            })
            .collect();
        let header = match entries.as_slice() {
            [] => continue,
            [header] => *header,
            _ => {
                let mut key: Vec<NodeId> = entries.iter().map(original_of).collect();
                key.sort_unstable();
                if !declined.contains(&key) {
                    return Some(Irreducible {
                        members,
                        entries,
                        key,
                    });
                }
                match cheapest(successors, &members, &entries, entry) {
                    Some((header, _)) => header,
                    None => continue,
                }
            }
        };
        let inner = members.iter().copied().filter(|member| *member != header);
        work.extend(Induced::of(successors, inner).cycles());
    }
    None
}

/// Returns the blocks of `members` reachable from `from` without passing
/// `header`, `from` included.
fn reach(
    successors: &[Vec<NodeId>],
    members: &BTreeSet<NodeId>,
    from: NodeId,
    header: NodeId,
) -> BTreeSet<NodeId> {
    let mut seen = BTreeSet::new();
    let mut walk = vec![from];
    while let Some(node) = walk.pop() {
        if node == header || !members.contains(&node) || !seen.insert(node) {
            continue;
        }
        walk.extend(successors.get(node.index()).into_iter().flatten());
    }
    seen
}

/// Returns the entry to keep as the loop's header and how many blocks
/// splitting the others copies: the function's entry when it is one, since
/// nothing outside can be moved off it, and otherwise the entry that copies
/// fewest, the lowest on a tie.
fn cheapest(
    successors: &[Vec<NodeId>],
    members: &BTreeSet<NodeId>,
    entries: &[NodeId],
    entry: NodeId,
) -> Option<(NodeId, usize)> {
    let copies = copies_per_header(successors, members, entries);
    if let Some(at) = entries.iter().position(|candidate| *candidate == entry) {
        return Some((entry, copies.get(at).copied()?));
    }
    entries
        .iter()
        .zip(copies)
        .map(|(header, count)| (count, *header))
        .min()
        .map(|(count, header)| (header, count))
}

/// Returns, for each of `entries` kept as the header, how many blocks
/// splitting the others copies: the blocks of `members` each other entry
/// reaches without passing the header ([`reach`]), summed.
///
/// What an entry reaches without passing a header is what it reaches at all,
/// less the blocks the header dominates in the loop's graph rooted at that
/// entry, the header among them. So one dominator tree per entry counts it for
/// every candidate at once, where a walk per entry per candidate cost the
/// square of the entries: a loop entered at two hundred blocks took minutes.
fn copies_per_header(
    successors: &[Vec<NodeId>],
    members: &BTreeSet<NodeId>,
    entries: &[NodeId],
) -> Vec<usize> {
    let induced = Induced::of(successors, members.iter().copied());
    let position = |node: &NodeId| induced.members.binary_search(node).ok().map(NodeId::new);
    let mut copies = vec![0usize; entries.len()];
    for other in entries {
        let Some(root) = position(other) else {
            continue;
        };
        let tree = compute_dominators(&induced, root);
        // Each reached block's dominator subtree, the block included, counted
        // from the leaves up; a block the walk never meets is not reached.
        let mut dominated = vec![0usize; induced.members.len()];
        let mut order = vec![root];
        let mut at = 0;
        while let Some(node) = order.get(at).copied() {
            order.extend_from_slice(tree.children(node));
            at = at.saturating_add(1);
        }
        for node in order.iter().rev() {
            let below: usize = tree
                .children(*node)
                .iter()
                .filter_map(|child| dominated.get(child.index()))
                .sum();
            if let Some(count) = dominated.get_mut(node.index()) {
                *count = below.saturating_add(1);
            }
        }
        let reached = order.len();
        for (header, count) in entries.iter().zip(copies.iter_mut()) {
            if header == other {
                continue;
            }
            let lost = position(header)
                .and_then(|header| dominated.get(header.index()).copied())
                .unwrap_or(0);
            *count = count.saturating_add(reached.saturating_sub(lost));
        }
    }
    copies
}

/// Gives `other`'s outside edges a copy of what it reaches inside `members`
/// before `header`.
///
/// Each copy keeps its original's successors in their order, a successor that
/// was copied too replaced by its copy, so the copied path rejoins the loop at
/// `header` or leaves it where the original would.
fn copy_entry(
    successors: &mut Vec<Vec<NodeId>>,
    original: &mut Vec<NodeId>,
    members: &BTreeSet<NodeId>,
    header: NodeId,
    other: NodeId,
) {
    let region = reach(successors, members, other, header);
    let before = successors.len();
    let mut copy_of: BTreeMap<NodeId, NodeId> = BTreeMap::new();
    for node in &region {
        copy_of.insert(*node, NodeId::new(successors.len()));
        successors.push(Vec::new());
        original.push(original.get(node.index()).copied().unwrap_or(*node));
    }
    for (node, copy) in &copy_of {
        let targets: Vec<NodeId> = successors
            .get(node.index())
            .into_iter()
            .flatten()
            .map(|target| copy_of.get(target).copied().unwrap_or(*target))
            .collect();
        if let Some(row) = successors.get_mut(copy.index()) {
            *row = targets;
        }
    }
    let Some(entered) = copy_of.get(&other).copied() else {
        return;
    };
    for (index, row) in successors.iter_mut().enumerate().take(before) {
        if members.contains(&NodeId::new(index)) {
            continue;
        }
        for target in row.iter_mut() {
            if *target == other {
                *target = entered;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The subgraph over `members` as a view as wide as the whole graph: the
    /// definition [`Induced`] is the dense renumbering of.
    struct WholeGraph<'a> {
        /// Successors per node of the whole graph.
        successors: &'a [Vec<NodeId>],

        /// The nodes the view keeps.
        members: &'a BTreeSet<NodeId>,
    }

    impl GraphBase for WholeGraph<'_> {
        fn node_count(&self) -> usize {
            self.successors.len()
        }

        fn node_ids(&self) -> impl Iterator<Item = NodeId> {
            (0..self.successors.len()).map(NodeId::new)
        }
    }

    impl Successors for WholeGraph<'_> {
        fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
            let kept = self.members.contains(&node);
            self.successors
                .get(node.index())
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .copied()
                .filter(move |successor| kept && self.members.contains(successor))
        }
    }

    /// **The cycles of a dense subgraph are those of the view as wide as the
    /// graph, in the same order.** Several thousand small graphs drawn from a
    /// fixed sequence, each with a random subset of its blocks as members: the
    /// order matters, because the search takes the loops it finds in that order.
    #[test]
    fn the_cycles_of_a_dense_subgraph_match_those_of_the_whole_graph_view() {
        let mut state = 0x7f4a_7c15_9e37_79b9u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let mut found = 0usize;
        for _ in 0..4000 {
            let count = usize::try_from(next(12)).unwrap_or(0).saturating_add(3);
            let bound = u64::try_from(count).unwrap_or(1);
            let successors: Vec<Vec<NodeId>> = (0..count)
                .map(|_| {
                    (0..next(4))
                        .map(|_| NodeId::new(usize::try_from(next(bound)).unwrap_or(0)))
                        .collect()
                })
                .collect();
            let members: BTreeSet<NodeId> = (0..count)
                .filter(|_| next(4) != 0)
                .map(NodeId::new)
                .collect();
            let reference: Vec<BTreeSet<NodeId>> = strongly_connected_components(&WholeGraph {
                successors: &successors,
                members: &members,
            })
            .into_iter()
            .filter(|component| component.len() > 1)
            .map(|component| component.into_iter().collect())
            .collect();
            let cycles = Induced::of(&successors, members.iter().copied()).cycles();
            assert_eq!(cycles, reference, "{successors:?} over {members:?}");
            found = found.saturating_add(cycles.len());
        }
        assert!(
            found > 0,
            "no graph had a cycle, so nothing above was measured"
        );
    }

    /// **Each header's copy count is what the other entries reach without
    /// passing it**, summed: on random graphs and random entries, the count
    /// one dominator tree per entry gives is the sum of the walks
    /// ([`reach`]) it replaces, for every candidate header.
    #[test]
    fn each_headers_copy_count_is_what_the_other_entries_reach_without_it() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let mut compared = 0usize;
        for _ in 0..3000 {
            let count = usize::try_from(next(14)).unwrap_or(0).saturating_add(2);
            let bound = u64::try_from(count).unwrap_or(1);
            let successors: Vec<Vec<NodeId>> = (0..count)
                .map(|_| {
                    (0..next(4))
                        .map(|_| NodeId::new(usize::try_from(next(bound)).unwrap_or(0)))
                        .collect()
                })
                .collect();
            let members: BTreeSet<NodeId> = (0..count)
                .filter(|_| next(5) != 0)
                .map(NodeId::new)
                .collect();
            let entries: Vec<NodeId> = members.iter().copied().filter(|_| next(3) == 0).collect();
            let definition: Vec<usize> = entries
                .iter()
                .map(|header| {
                    entries
                        .iter()
                        .filter(|other| *other != header)
                        .map(|other| reach(&successors, &members, *other, *header).len())
                        .sum()
                })
                .collect();
            assert_eq!(
                copies_per_header(&successors, &members, &entries),
                definition,
                "{successors:?} over {members:?} entered at {entries:?}"
            );
            compared = compared.saturating_add(entries.len());
        }
        assert!(compared > 1000, "{compared}");
    }

    /// **A loop entered at hundreds of blocks picks its header in one dominator
    /// tree per entry.** A ring of a thousand blocks entered at every other one:
    /// from each entry the ring is walked up to the header, so every header
    /// copies the same `m * (m - 1)` blocks for `m` entries, and the lowest is
    /// kept. Walked pair by pair it took minutes.
    #[test]
    fn a_loop_entered_at_hundreds_of_blocks_picks_its_header_at_once() {
        let blocks = 1000usize;
        let successors: Vec<Vec<NodeId>> = (0..blocks)
            .map(|block| vec![NodeId::new((block + 1) % blocks)])
            .collect();
        let members: BTreeSet<NodeId> = (0..blocks).map(NodeId::new).collect();
        let entries: Vec<NodeId> = (0..blocks).step_by(2).map(NodeId::new).collect();
        let outside = NodeId::new(blocks);
        let m = entries.len();
        assert_eq!(
            cheapest(&successors, &members, &entries, outside),
            Some((NodeId::new(0), m * (m - 1)))
        );
    }
}
