//! Path-local validity of initial image bindings, independent of known payload bytes.

use std::collections::BTreeMap;

use crate::analysis::execution::{Address, Root};

/// Canonical inclusive image ranges touched by writes, or an unresolved image alias.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ImageOrigins {
    /// Disjoint, nonadjacent ranges in unsigned image coordinates.
    changed: BTreeMap<u64, u64>,
    /// Whether a write may have affected any image cell.
    unknown: bool,
}

impl ImageOrigins {
    /// Records a complete write footprint without retaining events or payloads.
    pub fn write(&mut self, address: Address, len: usize) {
        if len == 0 || self.unknown {
            return;
        }
        match address.root() {
            Root::Symbol => {
                if address.covers_space(len) {
                    self.clobber();
                    return;
                }
                let Some(last) = address.offset_at(len.saturating_sub(1)) else {
                    self.clobber();
                    return;
                };
                let start = address.offset().cast_unsigned();
                let end = last.cast_unsigned();
                if start <= end {
                    self.insert(start, end);
                } else {
                    let maximum = address.pointer_size().mask_unsigned(u64::MAX);
                    self.insert(start, maximum);
                    self.insert(0, end);
                }
            }
            Root::Number | Root::Opaque { .. } | Root::StackDerived { .. } => self.clobber(),
            Root::Stack | Root::StackFrame(_) | Root::StackAligned { .. } | Root::Heap(_) => {}
        }
    }

    /// Invalidates every initial binding after a possible unplaced image write.
    pub fn clobber(&mut self) {
        self.unknown = true;
        self.changed.clear();
    }

    /// Returns whether the entire linear image cell retains its initial binding.
    pub fn unchanged(&self, address: Address, len: usize) -> bool {
        if self.unknown
            || address.root() != Root::Symbol
            || len == 0
            || address.linear_prefix_len(len) != len
        {
            return false;
        }
        let Some(last) = address.offset_at(len.saturating_sub(1)) else {
            return false;
        };
        let start = address.offset().cast_unsigned();
        self.changed
            .range(..=last.cast_unsigned())
            .next_back()
            .is_none_or(|(_, &end)| end < start)
    }

    /// Combines possibly changed cells from both predecessor paths.
    pub fn union(&self, other: &Self) -> Self {
        let mut joined = self.clone();
        if other.unknown {
            joined.clobber();
        } else if !joined.unknown {
            for (&start, &end) in &other.changed {
                joined.insert(start, end);
            }
        }
        joined
    }

    /// Coalesces a range with every overlapping or adjacent existing range.
    fn insert(&mut self, mut start: u64, mut end: u64) {
        if let Some((&before, &last)) = self.changed.range(..=start).next_back()
            && last.saturating_add(1) >= start
        {
            start = before;
            end = end.max(last);
            self.changed.remove(&before);
        }
        while let Some((&next, &last)) = self.changed.range(start..).next() {
            if next > end.saturating_add(1) {
                break;
            }
            end = end.max(last);
            self.changed.remove(&next);
        }
        self.changed.insert(start, end);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PointerSize;

    fn at(offset: u64) -> Address {
        Address::new(Root::Symbol, offset.cast_signed(), PointerSize::Bit64)
    }

    #[test]
    fn adjacent_ranges_canonicalize_independently_of_order() {
        let mut first = ImageOrigins::default();
        first.write(at(100), 4);
        first.write(at(108), 4);
        first.write(at(104), 4);
        let mut second = ImageOrigins::default();
        second.write(at(100), 12);
        assert_eq!(first, second);
        assert!(first.unchanged(at(92), 8));
        assert!(first.unchanged(at(112), 8));
        assert!(!first.unchanged(at(99), 2));
        assert!(!first.unchanged(at(111), 2));
    }

    #[test]
    fn wrapped_and_high_bit_coordinates_do_not_hide_writes() {
        let mut origins = ImageOrigins::default();
        origins.write(Address::new(Root::Symbol, 252, PointerSize::Bit8), 8);
        assert!(!origins.unchanged(at(0), 4));
        assert!(origins.unchanged(at(4), 248));
        assert!(!origins.unchanged(at(252), 4));
        origins.write(at(i64::MAX as u64), 2);
        assert!(!origins.unchanged(at(1u64 << 63), 1));
        origins.write(at(u64::MAX), 2);
        assert!(!origins.unchanged(at(u64::MAX), 1));
        assert!(!origins.unchanged(at(0), 1));
    }

    #[test]
    fn joins_and_unknown_aliases_retain_invalidations() {
        let mut left = ImageOrigins::default();
        let mut right = ImageOrigins::default();
        left.write(at(100), 8);
        right.write(at(200), 8);
        assert_eq!(left.union(&right), right.union(&left));
        let merged = left.union(&right);
        assert!(!merged.unchanged(at(100), 8));
        assert!(!merged.unchanged(at(200), 8));
        assert!(merged.unchanged(at(300), 8));
        left.write(Address::of_root(Root::Number), 0);
        assert!(left.unchanged(at(300), 8));
        left.write(Address::of_root(Root::Number), 1);
        assert!(!left.union(&right).unchanged(at(300), 8));
    }
}
