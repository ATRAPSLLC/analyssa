//! Which bits of memory no write has reached since their object began.
//!
//! An object begins never written: an allocation that does not zero its
//! bytes, an activation's frame below the stack pointer it was entered with.
//! A write of any value, known or not, ends that for the bytes it reaches,
//! and so does an effect the execution cannot follow that may have reached
//! them. A stored value carries the bits of it that were never written
//! themselves (bits loaded from such memory and passed on unchanged), so a
//! byte can hold some never-written bits after a write.
//!
//! Only what is certain is kept: a bit counts as never written where every
//! path joined agrees it is. Anything this module is not told reads as
//! written, so a mistake loses a finding rather than inventing one.

use std::{collections::BTreeMap, sync::Arc};

use crate::analysis::execution::value::Root;

/// Half-open ranges of offsets, disjoint and sorted, by their start.
type Ranges = BTreeMap<i64, i64>;

/// The bits of memory never written, per root.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Unwritten {
    /// Per root, the offsets whose every bit is never written.
    ranges: Option<Arc<BTreeMap<Root, Ranges>>>,
    /// Bytes outside those ranges holding some never-written bits a stored
    /// value carried, by position: the never-written bits of each.
    marks: Option<Arc<BTreeMap<(Root, i64), u8>>>,
}

impl Unwritten {
    /// The never-written bits of the byte at `offset` of `root`: all of them
    /// in a range never written, a stored value's own otherwise, else none.
    #[must_use]
    pub fn bits(&self, root: Root, offset: i64) -> u8 {
        let unwritten = self
            .ranges
            .as_ref()
            .and_then(|roots| roots.get(&root))
            .and_then(|ranges| ranges.range(..=offset).next_back())
            .is_some_and(|(_, end)| offset < *end);
        if unwritten {
            return u8::MAX;
        }
        self.marks
            .as_ref()
            .and_then(|marks| marks.get(&(root, offset)))
            .copied()
            .unwrap_or(0)
    }

    /// Whether nothing is known never written: every byte reads written.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_none() && self.marks.is_none()
    }

    /// The roots holding some never-written bit, each once, in order.
    #[must_use]
    pub fn roots(&self) -> Vec<Root> {
        let mut roots: Vec<Root> = self
            .ranges
            .iter()
            .flat_map(|roots| roots.keys().copied())
            .chain(
                self.marks
                    .iter()
                    .flat_map(|marks| marks.keys().map(|(root, _)| *root)),
            )
            .collect();
        roots.sort_unstable();
        roots.dedup();
        roots
    }

    /// States that `start..end` of `root` begins never written.
    pub fn begin(&mut self, root: Root, start: i64, end: i64) {
        if start >= end {
            return;
        }
        self.write(root, start, end);
        let roots = Arc::make_mut(self.ranges.get_or_insert_with(Arc::default));
        let ranges = roots.entry(root).or_default();
        let (mut start, mut end) = (start, end);
        // Merge with a range ending at or after `start` that begins before it,
        // and with every range beginning inside or right after the new one.
        if let Some((&prior, &prior_end)) = ranges.range(..=start).next_back()
            && prior_end >= start
        {
            start = prior;
            end = end.max(prior_end);
            ranges.remove(&prior);
        }
        while let Some((&next, &next_end)) = ranges.range(start..=end).next() {
            end = end.max(next_end);
            ranges.remove(&next);
        }
        ranges.insert(start, end);
    }

    /// States that a write, or an effect that may have written, reached
    /// `start..end` of `root`: none of its bits is never written now.
    pub fn write(&mut self, root: Root, start: i64, end: i64) {
        if start >= end {
            return;
        }
        if let Some(roots) = &mut self.ranges
            && roots.get(&root).is_some_and(|ranges| {
                ranges
                    .range(..end)
                    .next_back()
                    .is_some_and(|(_, last)| *last > start)
            })
        {
            let roots_mut = Arc::make_mut(roots);
            if let Some(ranges) = roots_mut.get_mut(&root) {
                let overlapping: Vec<(i64, i64)> = ranges
                    .range(..end)
                    .rev()
                    .take_while(|(_, last)| **last > start)
                    .map(|(first, last)| (*first, *last))
                    .collect();
                for (first, last) in overlapping {
                    ranges.remove(&first);
                    if first < start {
                        ranges.insert(first, start);
                    }
                    if last > end {
                        ranges.insert(end, last);
                    }
                }
                if ranges.is_empty() {
                    roots_mut.remove(&root);
                }
            }
            if roots_mut.is_empty() {
                self.ranges = None;
            }
        }
        if let Some(marks) = &mut self.marks
            && marks.range((root, start)..(root, end)).next().is_some()
        {
            let marks_mut = Arc::make_mut(marks);
            let keys: Vec<(Root, i64)> = marks_mut
                .range((root, start)..(root, end))
                .map(|(key, _)| *key)
                .collect();
            for key in keys {
                marks_mut.remove(&key);
            }
            if marks_mut.is_empty() {
                self.marks = None;
            }
        }
    }

    /// States that every byte of `root` may have been written.
    pub fn write_root(&mut self, root: Root) {
        self.write(root, i64::MIN, i64::MAX);
    }

    /// Records the never-written bits a stored value carried into the bytes
    /// it was written to, from `start` up: `bits[i]` for byte `start + i`.
    /// The bytes were just written, so only these bits are never written.
    pub fn carry(&mut self, root: Root, start: i64, bits: &[u8]) {
        for (index, bits) in bits.iter().enumerate() {
            let Some(offset) = i64::try_from(index)
                .ok()
                .and_then(|index| start.checked_add(index))
            else {
                return;
            };
            if *bits != 0 {
                Arc::make_mut(self.marks.get_or_insert_with(Arc::default))
                    .insert((root, offset), *bits);
            }
        }
    }

    /// The bits never written on both paths.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        if self == other {
            return self.clone();
        }
        let mut joined = Self::default();
        if let (Some(left), Some(right)) = (&self.ranges, &other.ranges) {
            for (root, ranges) in left.iter() {
                let Some(theirs) = right.get(root) else {
                    continue;
                };
                for (start, end) in ranges {
                    for (their_start, their_end) in theirs.range(..*end) {
                        let (from, to) = ((*start).max(*their_start), (*end).min(*their_end));
                        if from < to {
                            joined.begin(*root, from, to);
                        }
                    }
                }
            }
        }
        // A byte some stored value marked on either path keeps the bits
        // never written on both, whichever way each path holds them.
        let marked = self
            .marks
            .iter()
            .chain(other.marks.iter())
            .flat_map(|marks| marks.keys().copied());
        for (root, offset) in marked {
            let bits = self.bits(root, offset) & other.bits(root, offset);
            if bits != 0 && joined.bits(root, offset) != u8::MAX {
                Arc::make_mut(joined.marks.get_or_insert_with(Arc::default))
                    .insert((root, offset), bits);
            }
        }
        joined
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A write ends never-written storage exactly where it reaches.** A
    /// frame begins never written below its entry; a write splits it, a
    /// stored value's marks stay on the bytes it wrote until another write
    /// reaches them, and a join keeps only what both paths agree on.
    #[test]
    fn writes_end_never_written_storage_where_they_reach() {
        let root = Root::Stack;
        let mut memory = Unwritten::default();
        memory.begin(root, i64::MIN, 0);
        assert_eq!(memory.bits(root, -9), u8::MAX);
        memory.write(root, -8, -4);
        assert_eq!(memory.bits(root, -9), u8::MAX);
        assert_eq!(memory.bits(root, -8), 0);
        assert_eq!(memory.bits(root, -5), 0);
        assert_eq!(memory.bits(root, -4), u8::MAX);
        assert_eq!(memory.bits(root, 0), 0, "above entry is the caller's");

        memory.carry(root, -8, &[0, 0xf0]);
        assert_eq!(memory.bits(root, -7), 0xf0);
        let mut overwritten = memory.clone();
        overwritten.write(root, -7, -6);
        assert_eq!(overwritten.bits(root, -7), 0);

        let mut other = Unwritten::default();
        other.begin(root, -16, 0);
        let joined = memory.join(&other);
        assert_eq!(joined.bits(root, -9), u8::MAX);
        assert_eq!(joined.bits(root, -17), 0, "written on the other path");
        assert_eq!(joined.bits(root, -7), 0xf0, "marked here, unwritten there");
        assert_eq!(joined.bits(root, -5), 0);
        assert_eq!(joined, other.join(&memory));

        memory.write_root(root);
        assert!(memory.is_empty());
    }
}
