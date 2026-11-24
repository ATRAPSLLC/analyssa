//! Which write last stored each image byte, so an observation can keep only the
//! bytes its own execution wrote and name the instruction that wrote them.
//!
//! A static execution's memory at a function's return holds every image byte
//! the run has written so far, including bytes the caller wrote before the call
//! and bytes several writes left side by side. [`ImageWriters`] records, per
//! image byte, the activation and the instruction of the write that produced
//! it. An observation scoped to one activation keeps the bytes whose writer is
//! that activation or one it entered ([`ImageWriters::owned_since`]), and splits
//! them where the writing instruction changes.

use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

use crate::analysis::execution::{Address, Root};

/// The instruction that performed a write, in the enclosing execution's body
/// table.
///
/// A library model's write is attributed to the call instruction that ran the
/// model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WriteSite {
    /// Body index in the execution's body-method table.
    pub body: usize,
    /// Block of the writing instruction in that body.
    pub block: usize,
    /// Index of the writing instruction within its block.
    pub instruction: usize,
}

/// The write that produced a byte: the activation that ran it, and the
/// instruction, unless paths that agree on the byte wrote it at different
/// instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Writer {
    /// The writing activation; after a join, the lowest of the joined ones.
    pub activation: u32,
    /// The writing instruction; `None` when joined paths wrote the byte at
    /// different instructions.
    pub site: Option<WriteSite>,
}

impl Writer {
    /// The writer of a byte both of two paths hold: the same writer when they
    /// agree, otherwise the lower activation and no single site.
    fn join(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            Self {
                activation: self.activation.min(other.activation),
                site: None,
            }
        }
    }
}

/// The write the running instruction performs, which every store records.
///
/// It describes the instruction executing now, not what memory holds, so two
/// memories compare equal and hash alike whatever their current writer is.
#[derive(Debug, Clone, Copy, Default)]
pub struct CurrentWriter(pub Option<Writer>);

impl PartialEq for CurrentWriter {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for CurrentWriter {}

impl Hash for CurrentWriter {
    fn hash<H: Hasher>(&self, _: &mut H) {}
}

/// The writer of every image byte a path has written, as disjoint inclusive
/// ranges of the signed offsets image cells are keyed by.
///
/// Adjacent ranges with one writer are coalesced, so two paths that wrote the
/// same bytes with the same writers compare equal however their writes were
/// split.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ImageWriters {
    /// Range start to (inclusive end, writer).
    ranges: BTreeMap<i64, (i64, Writer)>,
}

impl ImageWriters {
    /// Records `writer` for the `len` image bytes at `address`, or forgets their
    /// writer when there is none. Writes under other roots are ignored.
    pub fn write(&mut self, address: Address, len: usize, writer: Option<Writer>) {
        if len == 0 || address.root() != Root::Symbol {
            return;
        }
        let start = address.offset();
        let end = address.offset_at(len.saturating_sub(1));
        match end {
            Some(end) if !address.covers_space(len) && start <= end => {
                self.assign(start, end, writer);
            }
            Some(end) if !address.covers_space(len) => {
                self.assign(start, i64::MAX, writer);
                self.assign(i64::MIN, end, writer);
            }
            _ => self.assign(i64::MIN, i64::MAX, writer),
        }
    }

    /// The writer of the image byte at `offset`.
    pub fn writer(&self, offset: i64) -> Option<Writer> {
        self.ranges
            .range(..=offset)
            .next_back()
            .filter(|(_, (end, _))| offset <= *end)
            .map(|(_, (_, writer))| *writer)
    }

    /// The bytes whose writer is activation `since` or a later one: on one path
    /// the activations entered after `since` are the ones it ran, so these are
    /// the bytes that activation's own execution wrote.
    #[must_use]
    pub fn owned_since(&self, since: u32) -> Self {
        Self {
            ranges: self
                .ranges
                .iter()
                .filter(|(_, (_, writer))| writer.activation >= since)
                .map(|(start, range)| (*start, *range))
                .collect(),
        }
    }

    /// The writers of the bytes both paths wrote. A byte only one path wrote has
    /// no writer here: on the other path it holds what was there before.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        let mut joined = Self::default();
        for (&start, &(end, writer)) in &self.ranges {
            let reaching = other
                .ranges
                .range(..start)
                .next_back()
                .filter(|(_, (other_end, _))| *other_end >= start);
            for (&other_start, &(other_end, other_writer)) in
                reaching.into_iter().chain(other.ranges.range(start..=end))
            {
                joined.assign(
                    start.max(other_start),
                    end.min(other_end),
                    Some(writer.join(other_writer)),
                );
            }
        }
        joined
    }

    /// Sets the writer of the inclusive range `start..=end`, splitting the
    /// ranges it overlaps and coalescing equal neighbours.
    fn assign(&mut self, start: i64, end: i64, writer: Option<Writer>) {
        // Split a range that begins before `start` and reaches into it.
        if let Some((&before, &(before_end, before_writer))) =
            self.ranges.range(..start).next_back()
            && before_end >= start
        {
            self.ranges
                .insert(before, (start.saturating_sub(1), before_writer));
            if before_end > end {
                self.ranges
                    .insert(end.saturating_add(1), (before_end, before_writer));
            }
        }
        // Remove or trim every range that begins inside `start..=end`.
        let inside: Vec<i64> = self.ranges.range(start..=end).map(|(at, _)| *at).collect();
        for at in inside {
            if let Some((range_end, range_writer)) = self.ranges.remove(&at)
                && range_end > end
            {
                self.ranges
                    .insert(end.saturating_add(1), (range_end, range_writer));
            }
        }
        let Some(writer) = writer else {
            return;
        };
        let (mut start, mut end) = (start, end);
        if let Some((&before, &(before_end, before_writer))) =
            self.ranges.range(..start).next_back()
            && before_writer == writer
            && before_end.checked_add(1) == Some(start)
        {
            self.ranges.remove(&before);
            start = before;
        }
        if let Some(after) = end.checked_add(1)
            && let Some(&(after_end, after_writer)) = self.ranges.get(&after)
            && after_writer == writer
        {
            self.ranges.remove(&after);
            end = after_end;
        }
        self.ranges.insert(start, (end, writer));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PointerSize;

    fn at(offset: i64) -> Address {
        Address::new(Root::Symbol, offset, PointerSize::Bit64)
    }

    fn writer(activation: u32, instruction: usize) -> Writer {
        Writer {
            activation,
            site: Some(WriteSite {
                body: 0,
                block: 0,
                instruction,
            }),
        }
    }

    /// A later write takes over the bytes it covers and leaves the rest to the
    /// earlier one: 33 bytes, then 31 over their start, leave two of the first.
    #[test]
    fn an_overlapping_write_keeps_the_tail_of_the_earlier_one() {
        let mut writers = ImageWriters::default();
        writers.write(at(0x100), 33, Some(writer(1, 4)));
        writers.write(at(0x100), 31, Some(writer(2, 9)));
        assert_eq!(writers.writer(0x100), Some(writer(2, 9)));
        assert_eq!(writers.writer(0x11e), Some(writer(2, 9)));
        assert_eq!(writers.writer(0x11f), Some(writer(1, 4)));
        assert_eq!(writers.writer(0x120), Some(writer(1, 4)));
        assert_eq!(writers.writer(0x121), None);
    }

    /// Writes split differently but with the same writers compare equal, so
    /// joined states that agree are recognised as the same state.
    #[test]
    fn equal_writers_coalesce_however_the_writes_were_split() {
        let mut bytewise = ImageWriters::default();
        for offset in 0..8 {
            bytewise.write(at(0x200 + offset), 1, Some(writer(3, 7)));
        }
        let mut whole = ImageWriters::default();
        whole.write(at(0x200), 8, Some(writer(3, 7)));
        assert_eq!(bytewise, whole);
    }

    /// Only the bytes an activation or one it entered wrote are its own.
    #[test]
    fn owned_bytes_are_written_by_the_activation_or_a_later_one() {
        let mut writers = ImageWriters::default();
        writers.write(at(0x300), 4, Some(writer(1, 0)));
        writers.write(at(0x304), 4, Some(writer(5, 1)));
        writers.write(at(0x308), 4, Some(writer(6, 2)));
        let own = writers.owned_since(5);
        assert_eq!(own.writer(0x300), None);
        assert_eq!(own.writer(0x304), Some(writer(5, 1)));
        assert_eq!(own.writer(0x308), Some(writer(6, 2)));
    }

    /// A join keeps a byte's writer only where both paths wrote it: a byte
    /// one path did not write holds what was there before the activation.
    /// Different sites keep the lower activation and no site.
    #[test]
    fn a_join_keeps_bytes_both_paths_wrote() {
        let mut left = ImageWriters::default();
        left.write(at(0x400), 8, Some(writer(4, 1)));
        let mut right = ImageWriters::default();
        right.write(at(0x404), 8, Some(writer(6, 2)));
        let joined = left.join(&right);
        assert_eq!(joined.writer(0x403), None);
        assert_eq!(
            joined.writer(0x404),
            Some(Writer {
                activation: 4,
                site: None
            })
        );
        assert_eq!(joined.writer(0x408), None);
        assert_eq!(left.join(&left), left);
    }

    /// Forgetting a writer splits the range around the forgotten bytes.
    #[test]
    fn forgetting_a_writer_splits_its_range() {
        let mut writers = ImageWriters::default();
        writers.write(at(0x500), 8, Some(writer(1, 1)));
        writers.write(at(0x502), 2, None);
        assert_eq!(writers.writer(0x501), Some(writer(1, 1)));
        assert_eq!(writers.writer(0x502), None);
        assert_eq!(writers.writer(0x503), None);
        assert_eq!(writers.writer(0x504), Some(writer(1, 1)));
    }

    /// Non-image roots carry no writer.
    #[test]
    fn writes_outside_the_image_are_ignored() {
        let mut writers = ImageWriters::default();
        writers.write(
            Address::new(Root::Stack, -16, PointerSize::Bit64),
            8,
            Some(writer(1, 1)),
        );
        assert_eq!(writers, ImageWriters::default());
    }
}
