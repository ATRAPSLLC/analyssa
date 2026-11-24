//! The objects the pointers memory holds were formed into.
//!
//! A pointer formed into an object keeps it when the program stores it and
//! loads it back: a spill, a structure field, an argument passed on the
//! stack. What memory keeps is the object of each pointer-wide store, by
//! where it starts and how wide it is; a load of exactly that span hands it
//! back. Any later write that reaches a byte of the span forgets it, so a
//! load hands back an object only where the stored pointer is what the span
//! still holds.

use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

use crate::analysis::execution::value::{Address, Root};

/// Which way an access moves bytes: out of memory, or into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transfer {
    /// The access reads the bytes it spans.
    Read,
    /// The access writes them.
    Write,
}

/// The objects the pointers of the running access were formed into: the
/// one its reads go through and the one its writes go through, `None` where
/// no pointer formed into an object serves that way. Every access is judged
/// against the object it goes through as well as where it lands.
///
/// It describes the access running now, not what memory holds, so two
/// memories compare equal and hash alike whatever it is.
#[derive(Debug, Clone, Copy, Default)]
pub struct Through {
    /// The start of the object the bytes the access reads are reached
    /// through.
    pub reads: Option<Address>,
    /// The start of the object the bytes it writes are reached through.
    pub writes: Option<Address>,
}

impl Through {
    /// The start of the object an access moving bytes as `transfer` says
    /// goes through.
    #[must_use]
    pub const fn object(self, transfer: Transfer) -> Option<Address> {
        match transfer {
            Transfer::Read => self.reads,
            Transfer::Write => self.writes,
        }
    }
}

impl PartialEq for Through {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for Through {}

impl Hash for Through {
    fn hash<H: Hasher>(&self, _: &mut H) {}
}

/// The object each stored pointer was formed into, by the `(root, offset)`
/// its store starts at, with the store's width.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Formed(BTreeMap<(Root, i64), (usize, Address)>);

impl Formed {
    /// The widest store a pointer takes, which bounds how far before a
    /// written byte a store reaching it can start.
    const WIDEST: i64 = 16;

    /// Keeps `object` as the object of the pointer stored in `width` bytes at
    /// `at`.
    pub fn store(&mut self, at: Address, width: usize, object: Address) {
        self.0.insert((at.root(), at.offset()), (width, object));
    }

    /// The object of the pointer stored in exactly `width` bytes at `at`,
    /// where the span still holds it.
    pub fn load(&self, at: Address, width: usize) -> Option<Address> {
        self.0
            .get(&(at.root(), at.offset()))
            .filter(|(stored, _)| *stored == width)
            .map(|(_, object)| *object)
    }

    /// Forgets every stored pointer a write of `[start, end)` in `root`
    /// reaches.
    pub fn write(&mut self, root: Root, start: i64, end: i64) {
        if self.0.is_empty() || start >= end {
            return;
        }
        let first = (root, start.saturating_sub(Self::WIDEST));
        let reached: Vec<(Root, i64)> = self
            .0
            .range(first..(root, end))
            .filter(|((_, offset), (width, _))| {
                i64::try_from(*width)
                    .ok()
                    .and_then(|width| offset.checked_add(width))
                    .is_none_or(|stored_end| stored_end > start)
            })
            .map(|(key, _)| *key)
            .collect();
        for key in reached {
            self.0.remove(&key);
        }
    }

    /// Forgets every stored pointer in `root`.
    pub fn write_root(&mut self, root: Root) {
        self.0.retain(|(stored, _), _| *stored != root);
    }

    /// The roots holding a stored pointer's object.
    pub fn roots(&self) -> Vec<Root> {
        let mut roots: Vec<Root> = self.0.keys().map(|(root, _)| *root).collect();
        roots.dedup();
        roots
    }

    /// What both of two memories keep: the stored pointers they agree on.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        Self(
            self.0
                .iter()
                .filter(|(key, kept)| other.0.get(key) == Some(*kept))
                .map(|(key, kept)| (*key, *kept))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointer::PointerSize;

    fn at(offset: i64) -> Address {
        Address::new(Root::Stack, offset, PointerSize::Bit64)
    }

    /// A stored pointer's object comes back from a load of its span, and is
    /// forgotten by any write reaching a byte of it.
    #[test]
    fn a_stored_object_lasts_until_its_span_is_written() {
        let object = Address::new(Root::Symbol, 0x3000, PointerSize::Bit64);
        let mut formed = Formed::default();
        formed.store(at(-16), 8, object);
        assert_eq!(formed.load(at(-16), 8), Some(object));
        assert_eq!(formed.load(at(-16), 4), None);
        assert_eq!(formed.load(at(-12), 8), None);
        formed.write(Root::Stack, -32, -16);
        assert_eq!(formed.load(at(-16), 8), Some(object));
        formed.write(Root::Stack, -9, -8);
        assert_eq!(formed.load(at(-16), 8), None);
        formed.store(at(-16), 8, object);
        formed.write_root(Root::Stack);
        assert!(formed.roots().is_empty());
    }
}
