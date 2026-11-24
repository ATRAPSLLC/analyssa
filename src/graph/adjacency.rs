//! Adjacency lists in compressed-row form.
//!
//! A graph view that is rebuilt often (a CFG over a function the passes keep
//! editing) pays two allocations per node for a `Vec<Vec<usize>>`. The
//! compressed form keeps every row in one buffer and the row boundaries in
//! another, so building it costs a fixed handful of allocations whatever the
//! node count.

use crate::graph::{GraphBase, NodeId, Successors};

/// Rows of values, one per node, stored contiguously: node `n` owns
/// `values[offsets[n]..offsets[n + 1]]`.
///
/// The values are node indices for a graph's edges, and anything else a host
/// lists per node otherwise: the use sites of each variable, the operands of
/// each instruction.
///
/// Total: a node outside the domain has an empty row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adjacency<V = usize> {
    /// Row boundaries, one more than the node count, non-decreasing.
    ///
    /// Native indices: a boundary is a position in `values`, so it never
    /// exceeds what the buffer itself can address.
    offsets: Vec<usize>,
    /// Every row's values, concatenated in node order.
    values: Vec<V>,
}

impl<V> Default for Adjacency<V> {
    fn default() -> Self {
        Self {
            offsets: vec![0],
            values: Vec::new(),
        }
    }
}

impl<V: Copy> Adjacency<V> {
    /// Builds the rows of `0..node_count` in order: `fill` is called once per
    /// node and pushes that node's values.
    ///
    /// # Arguments
    ///
    /// * `node_count` - The size of the domain.
    /// * `fill` - Given a node and the buffer, appends the node's row.
    pub fn from_rows(node_count: usize, mut fill: impl FnMut(usize, &mut Vec<V>)) -> Self {
        let mut offsets = Vec::with_capacity(node_count.saturating_add(1));
        let mut values = Vec::with_capacity(node_count.saturating_mul(2));
        offsets.push(0);
        for node in 0..node_count {
            fill(node, &mut values);
            offsets.push(values.len());
        }
        Self { offsets, values }
    }

    /// Builds the rows from `(node, value)` pairs in any order, by counting
    /// sort: each row keeps its values in the order `walk` reports them.
    ///
    /// `walk` is called twice, once to count and once to place, and must
    /// report the same pairs both times. A pair whose node is outside
    /// `0..node_count` is dropped.
    ///
    /// # Arguments
    ///
    /// * `node_count` - The size of the domain.
    /// * `filler` - Any value; it occupies the slots until they are placed and
    ///   none is left once the walk is done.
    /// * `walk` - Reports every pair to the visitor it is given.
    pub fn from_walk(
        node_count: usize,
        filler: V,
        walk: impl Fn(&mut dyn FnMut(usize, V)),
    ) -> Self {
        let mut offsets = vec![0usize; node_count.saturating_add(1)];
        walk(&mut |node, _| {
            // Counted one slot high, so the prefix sum below turns the counts
            // into the offsets in place. A count is bounded by the pairs
            // walked, so it cannot saturate before memory runs out.
            if node < node_count
                && let Some(slot) = offsets.get_mut(node.saturating_add(1))
            {
                *slot = slot.saturating_add(1);
            }
        });
        let mut running = 0usize;
        for slot in &mut offsets {
            running = running.saturating_add(*slot);
            *slot = running;
        }
        let mut values = vec![filler; running];
        let mut cursors = offsets.clone();
        walk(&mut |node, value| {
            if node < node_count
                && let Some(cursor) = cursors.get_mut(node)
            {
                if let Some(slot) = values.get_mut(*cursor) {
                    *slot = value;
                }
                *cursor = cursor.saturating_add(1);
            }
        });
        Self { offsets, values }
    }

    /// Builds the rows from `(node, value)` pairs in any order; see
    /// [`Self::from_walk`].
    ///
    /// # Arguments
    ///
    /// * `node_count` - The size of the domain.
    /// * `pairs` - The entries; walked twice, once to count and once to place.
    pub fn from_pairs<I>(node_count: usize, pairs: I) -> Self
    where
        V: Default,
        I: IntoIterator<Item = (usize, V)>,
        I::IntoIter: Clone,
    {
        let pairs = pairs.into_iter();
        Self::from_walk(node_count, V::default(), |visit| {
            for (node, value) in pairs.clone() {
                visit(node, value);
            }
        })
    }

    /// The row of `node`, empty for a node outside the domain.
    #[must_use]
    pub fn row(&self, node: usize) -> &[V] {
        let (Some(&start), Some(&end)) = (
            self.offsets.get(node),
            node.checked_add(1).and_then(|next| self.offsets.get(next)),
        ) else {
            return &[];
        };
        self.values.get(start..end).unwrap_or(&[])
    }

    /// The size of the domain.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The number of entries over every row.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.values.len()
    }
}

impl Adjacency {
    /// The inverse relation over the same domain: `m` lists `n` once for each
    /// time `n` lists `m`.
    ///
    /// Each inverse row holds its nodes in ascending order, so inverting twice
    /// gives back the rows sorted. A value outside the domain has no row to go
    /// in and is dropped.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let node_count = self.node_count();
        Self::from_walk(node_count, 0, |visit| {
            for node in 0..node_count {
                for &to in self.row(node) {
                    visit(to, node);
                }
            }
        })
    }
}

/// A graph whose node `n`'s successors are its row: the graph algorithms
/// (strongly connected components, traversals) run over the compressed rows
/// directly.
impl GraphBase for Adjacency {
    fn node_count(&self) -> usize {
        Adjacency::node_count(self)
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> {
        (0..Adjacency::node_count(self)).map(NodeId::new)
    }
}

impl Successors for Adjacency {
    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> {
        self.row(node.index()).iter().map(|&to| NodeId::new(to))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::graph::algorithms::strongly_connected_components;

    /// Pairs land in their node's row in the order given, and a node past the
    /// domain has an empty row.
    #[test]
    fn pairs_are_grouped_by_node_in_arrival_order() {
        let rows = Adjacency::from_pairs(3, [(2, 7), (0, 5), (2, 1), (9, 4), (0, 6)]);
        assert_eq!(rows.node_count(), 3);
        assert_eq!(rows.row(0), [5, 6]);
        assert!(rows.row(1).is_empty());
        assert_eq!(rows.row(2), [7, 1]);
        assert!(rows.row(3).is_empty());
        assert!(rows.row(usize::MAX).is_empty());
        assert_eq!(rows.entry_count(), 4);
    }

    /// Rows filled in order match the same rows given as pairs.
    #[test]
    fn rows_and_pairs_build_the_same_relation() {
        let lists: [&[usize]; 3] = [&[1, 2], &[], &[0, 0]];
        let filled = Adjacency::<usize>::from_rows(3, |node, values| {
            values.extend_from_slice(lists.get(node).copied().unwrap_or_default());
        });
        let paired = Adjacency::from_pairs(
            3,
            lists
                .iter()
                .enumerate()
                .flat_map(|(node, row)| row.iter().map(move |&to| (node, to))),
        );
        assert_eq!(filled, paired);
        assert_eq!(Adjacency::<usize>::default().node_count(), 0);
    }

    /// The inverse keeps multiplicity, lists sources in ascending order and
    /// drops what lies outside the domain.
    #[test]
    fn the_inverse_is_exact() {
        let rows = Adjacency::from_pairs(3, [(2, 0), (0, 1), (2, 1), (1, 1), (0, 9), (2, 1)]);
        let inverse = rows.inverse();
        assert_eq!(inverse.row(0), [2]);
        assert_eq!(inverse.row(1), [0, 1, 2, 2]);
        assert!(inverse.row(2).is_empty());
        assert_eq!(inverse.inverse().row(2), [0, 1, 1]);
    }
    /// Stable pair streams retain duplicates and ignore rows outside the domain.
    #[test]
    fn pair_walk_preserves_literal_rows() {
        let pairs = [(2, 7), (0, 3), (2, 8), (5, 9), (0, 4)];
        let passes = Cell::new(0);
        let rows = Adjacency::from_walk(3, 0, |visit| {
            passes.set(passes.get() + 1);
            for (node, value) in pairs {
                visit(node, value);
            }
        });
        assert_eq!(passes.get(), 2);
        assert_eq!(rows.row(0), [3, 4]);
        assert!(rows.row(1).is_empty());
        assert_eq!(rows.row(2), [7, 8]);
        assert_eq!(rows.entry_count(), 4);
    }

    /// **The rows are a graph.** Its components come out sinks first: the
    /// cycle `0 <-> 1` reaches `2`, which reaches nothing, and a value outside
    /// the domain is no edge.
    #[test]
    fn rows_are_a_graph_the_algorithms_walk() {
        let rows = Adjacency::from_pairs(3, [(0, 1), (1, 0), (1, 2), (2, 9)]);
        assert_eq!(GraphBase::node_count(&rows), 3);
        assert_eq!(
            rows.successors(NodeId::new(1)).collect::<Vec<_>>(),
            [NodeId::new(0), NodeId::new(2)]
        );
        let mut sccs = strongly_connected_components(&rows);
        for scc in &mut sccs {
            scc.sort_by_key(|node| node.index());
        }
        assert_eq!(
            sccs,
            [vec![NodeId::new(2)], vec![NodeId::new(0), NodeId::new(1)]]
        );
    }
}
