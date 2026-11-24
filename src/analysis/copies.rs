//! Sequencing a parallel copy into ordered assignments.
//!
//! A join's phis read every operand on entry, before any of their results
//! exists, so the copies one predecessor needs when SSA is taken apart take
//! effect together. Code that writes one variable at a time has no parallel
//! assignment. Written in phi order, a copy whose source reads another copy's
//! destination reads the value that copy has just stored - the swap
//! `r1 = phi(.., r2)`, `r2 = phi(.., r1)` becomes `r1 = r2; r2 = r1` and both
//! end up holding `r2`.
//!
//! The order that is correct writes a destination only once no copy still to
//! come reads it. When every remaining destination is still read, the copies
//! form a cycle; reading one source into a temporary first breaks it, and that
//! copy's destination is then written from the temporary once nothing else
//! needs its old value.
//!
//! The destinations and sources are the caller's own: a destination `D` is
//! whatever it writes (a local's id), a source `S` whatever it reads from
//! (a local, a constant, a folded expression), and each copy states which
//! destinations its source reads.

/// One predecessor's copies, which take effect together.
#[derive(Debug, Clone)]
pub struct ParallelCopy<D, S> {
    /// The copies not yet ordered, in the order they were given.
    pending: Vec<Assignment<D, S>>,
}

/// One copy: its destination, its source, and the destinations its source
/// reads.
#[derive(Debug, Clone)]
struct Assignment<D, S> {
    /// What is written.
    dest: D,
    /// What is written to it.
    src: S,
    /// Every destination `src` reads - through a memory operand's base or
    /// index, inside a folded computation, or as itself.
    reads: Vec<D>,
}

impl<D: Copy + PartialEq, S> ParallelCopy<D, S> {
    /// Gathers copies that take effect together.
    ///
    /// # Arguments
    ///
    /// * `copies` - `(destination, source, destinations the source reads)`,
    ///   one per phi. Every destination is distinct.
    pub fn new(copies: impl IntoIterator<Item = (D, S, Vec<D>)>) -> Self {
        Self {
            pending: copies
                .into_iter()
                .map(|(dest, src, reads)| Assignment { dest, src, reads })
                .collect(),
        }
    }

    /// Orders the copies so that no source reads a destination an earlier copy
    /// wrote.
    ///
    /// Among the copies that may go next, the first given goes first, so copies
    /// that do not depend on one another keep their order.
    ///
    /// # Arguments
    ///
    /// * `fresh` - Allocates a destination no copy names, for breaking a cycle.
    /// * `read` - The source that reads a destination back, for writing a
    ///   cycle's copy out of its temporary.
    ///
    /// # Returns
    ///
    /// `(destination, source)` assignments in the order to write them.
    pub fn sequence(self, mut fresh: impl FnMut() -> D, read: impl Fn(D) -> S) -> Vec<(D, S)> {
        let mut pending = self.pending;
        let mut ordered: Vec<Assignment<D, S>> = Vec::with_capacity(pending.len());
        while !pending.is_empty() {
            let ready = (0..pending.len()).find(|&candidate| {
                let dest = pending.get(candidate).map(|copy| copy.dest);
                !pending.iter().enumerate().any(|(other, copy)| {
                    other != candidate && dest.is_some_and(|dest| copy.reads.contains(&dest))
                })
            });
            match ready {
                Some(index) => ordered.push(pending.remove(index)),
                None => {
                    // A cycle: every destination is still to be read. The first
                    // copy's source is read into a temporary now, while all of
                    // its inputs are intact, and its destination is written
                    // from the temporary after the copies that still need the
                    // old value.
                    let first = pending.remove(0);
                    let temporary = fresh();
                    pending.push(Assignment {
                        dest: first.dest,
                        src: read(temporary),
                        reads: vec![temporary],
                    });
                    ordered.push(Assignment {
                        dest: temporary,
                        src: first.src,
                        reads: first.reads,
                    });
                }
            }
        }
        ordered
            .into_iter()
            .map(|copy| (copy.dest, copy.src))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// Runs `assignments` one after another over `initial` and returns the
    /// final value of every variable, where a value is the variable whose
    /// initial content it holds.
    fn run(initial: &[u32], assignments: &[(u32, u32)]) -> BTreeMap<u32, u32> {
        let mut state: BTreeMap<u32, u32> = initial.iter().map(|id| (*id, *id)).collect();
        for (dest, src) in assignments {
            let value = state.get(src).copied().unwrap_or(*src);
            state.insert(*dest, value);
        }
        state
    }

    /// What the copies mean: every destination receives its source's value
    /// from before any of them ran.
    fn simultaneous(initial: &[u32], copies: &[(u32, u32)]) -> BTreeMap<u32, u32> {
        let mut state: BTreeMap<u32, u32> = initial.iter().map(|id| (*id, *id)).collect();
        for (dest, src) in copies {
            state.insert(*dest, *src);
        }
        state
    }

    /// Sequences `copies` over variables `initial`, allocating temporaries
    /// from 100 up, and checks the result against the simultaneous meaning.
    fn assert_sequenced_correctly(initial: &[u32], copies: &[(u32, u32)]) -> Vec<(u32, u32)> {
        let mut next = 100;
        let sequenced =
            ParallelCopy::new(copies.iter().map(|(dest, src)| (*dest, *src, vec![*src]))).sequence(
                || {
                    next += 1;
                    next
                },
                |temporary| temporary,
            );
        let mut after = run(initial, &sequenced);
        after.retain(|id, _| *id < 100);
        assert_eq!(after, simultaneous(initial, copies), "{sequenced:?}");
        sequenced
    }

    /// Copies that read nothing another writes keep the order they were given.
    #[test]
    fn independent_copies_keep_their_order() {
        let sequenced = assert_sequenced_correctly(&[1, 2, 3, 4], &[(1, 3), (2, 4)]);
        assert_eq!(sequenced, vec![(1, 3), (2, 4)]);
    }

    /// A copy whose destination another reads waits for that copy.
    #[test]
    fn a_read_destination_is_written_after_its_reader() {
        let sequenced = assert_sequenced_correctly(&[1, 2, 3], &[(2, 3), (1, 2)]);
        assert_eq!(sequenced, vec![(1, 2), (2, 3)]);
    }

    /// A swap is a cycle, and one temporary breaks it.
    #[test]
    fn a_swap_goes_through_one_temporary() {
        let sequenced = assert_sequenced_correctly(&[1, 2], &[(1, 2), (2, 1)]);
        assert_eq!(sequenced.len(), 3, "{sequenced:?}");
    }

    /// A rotation of three is one cycle too, and a copy hanging off it is
    /// ordered around it.
    #[test]
    fn a_rotation_with_a_tail_is_sequenced() {
        assert_sequenced_correctly(&[1, 2, 3, 4], &[(1, 2), (2, 3), (3, 1), (4, 1)]);
    }

    /// A copy may read its own destination: `x = x` reads before it writes, and
    /// is not a cycle.
    #[test]
    fn a_copy_reading_its_own_destination_needs_no_temporary() {
        let sequenced = assert_sequenced_correctly(&[1, 2], &[(1, 1), (2, 1)]);
        assert_eq!(sequenced.len(), 2, "{sequenced:?}");
    }
}
