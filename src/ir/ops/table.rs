//! The machine-checkable contract every operation-kind enum shares.
//!
//! The taxonomy in [`kinds`](super::kinds), [`native`](super::native) and
//! [`vector`](super::vector) is a family of small enums that each name one
//! native operation per variant. Three things must hold of every one of them,
//! and each is checked mechanically rather than asserted in prose:
//!
//! - **Enumerability.** The identity domain is finite and can be walked, so a
//!   test can quantify over it instead of over a hand-written sample that goes
//!   stale the moment a variant is added.
//! - **Injective spelling.** [`OpKindTable::kind_str`] is injective over that
//!   domain, and [`OpKindTable::mnemonic`] is injective over the *union* of
//!   every table. [`SsaOp::opcode_name`](super::SsaOp::opcode_name) is one flat
//!   namespace, so two tables spelling one instruction two ways is the same
//!   defect as one table spelling two instructions the same way.
//! - **A pinned size.** [`OpKindTable::COUNT`] is asserted against
//!   [`OpKindTable::from_index`], so adding a variant fails the test suite
//!   until the count is bumped deliberately.
//!
//! # Mechanism
//!
//! [`OpKindTable::from_index`] goes through the [`TryFrom<u16>`] impl that
//! `num_enum::TryFromPrimitive` derives on each `#[repr(u16)]` kind. That
//! conversion is deliberately *not* a supertrait bound: `num_enum` stays a
//! private dependency, so a major bump of it is not a breaking change to this
//! crate, and the primitive stays out of the public signature.
//!
//! `op_kind_table!` writes each impl in one line, so the only per-enum
//! handwritten data is the count. Forgetting an arm of the underlying
//! `kind_str` match is a compile error; forgetting to bump the count is a test
//! failure.
//!
//! # Open hole
//!
//! Nothing at the type level forces a *newly added* kind enum to implement
//! this trait at all — Rust cannot express that without a proc macro or
//! link-time registration. Registration in `all_tables` is by hand, and the
//! closest available forcing function is a renderer bounded on `OpKindTable`.
//! Recorded here rather than claimed closed.

use std::{fmt, marker::PhantomData};

use crate::ir::ops::{
    kinds::{
        BarrierOp, BcdAdjustKind, BreakpointOp, CacheMaintenanceOp, HardwareEngineOp, HintOp,
        HypervisorOp, InterruptReturnOp, MachineStateOp, PacKind, SysRegOp, SystemTransactionKind,
        TlbMaintenanceOp, TrapOp,
    },
    native::BlockStringKind,
    vector::FlagAdjustKind,
};

/// The contract shared by every operation-kind enum in the taxonomy.
///
/// One variant per native operation, a finite index range that enumerates
/// them, and a stable spelling per variant. Implemented through
/// `op_kind_table!` rather than by hand.
pub trait OpKindTable: Copy + Eq + fmt::Debug + 'static {
    /// The enum's own name, for test diagnostics that name the offending table.
    const NAME: &'static str;

    /// How many variants the enum declares.
    ///
    /// The pin: `from_index(COUNT - 1)` must resolve and `from_index(COUNT)`
    /// must not, so a variant added without bumping this fails the test suite
    /// instead of silently shrinking the domain every registry-wide check
    /// quantifies over.
    const COUNT: u16;

    /// Returns the variant with the given discriminant, or `None` past the end.
    fn from_index(index: u16) -> Option<Self>;

    /// Returns the stable display / fingerprint key for this variant.
    ///
    /// Injective over the enum, and — with every other table's keys — over the
    /// union, because [`SsaOp::opcode_name`](super::SsaOp::opcode_name) hands
    /// them all to one namespace.
    fn kind_str(self) -> &'static str;

    /// Returns the neutral assembler-facing mnemonic, when the enum has one.
    ///
    /// `None` for the tables whose only spelling is [`Self::kind_str`].
    fn mnemonic(self) -> Option<&'static str> {
        None
    }

    /// Returns an iterator over every variant, in discriminant order.
    fn all() -> OpKindIter<Self> {
        OpKindIter::new()
    }
}

/// Iterator over every variant of an [`OpKindTable`], in discriminant order.
///
/// Walks `0..COUNT` through [`OpKindTable::from_index`] and skips any index
/// that does not resolve, so a table with a gap in its discriminants still
/// yields exactly its real variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpKindIter<K> {
    /// Next discriminant to try.
    next: u16,
    /// Ties the iterator to its table without storing one.
    marker: PhantomData<K>,
}

impl<K> OpKindIter<K> {
    /// Creates an iterator positioned before the first variant.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next: 0,
            marker: PhantomData,
        }
    }
}

impl<K> Default for OpKindIter<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: OpKindTable> Iterator for OpKindIter<K> {
    type Item = K;

    fn next(&mut self) -> Option<K> {
        while self.next < K::COUNT {
            let index = self.next;
            self.next = self.next.saturating_add(1);
            if let Some(kind) = K::from_index(index) {
                return Some(kind);
            }
        }
        None
    }
}

/// Implements [`OpKindTable`] for an enum that already carries `#[repr(u16)]`,
/// a `num_enum::TryFromPrimitive` derive and an inherent `kind_str`.
///
/// The second form additionally forwards an inherent `mnemonic`.
macro_rules! op_kind_table {
    ($kind:ty, $count:literal) => {
        impl OpKindTable for $kind {
            const NAME: &'static str = stringify!($kind);
            const COUNT: u16 = $count;

            fn from_index(index: u16) -> Option<Self> {
                <Self as TryFrom<u16>>::try_from(index).ok()
            }

            fn kind_str(self) -> &'static str {
                Self::kind_str(self)
            }
        }
    };
    ($kind:ty, $count:literal, with_mnemonic) => {
        impl OpKindTable for $kind {
            const NAME: &'static str = stringify!($kind);
            const COUNT: u16 = $count;

            fn from_index(index: u16) -> Option<Self> {
                <Self as TryFrom<u16>>::try_from(index).ok()
            }

            fn kind_str(self) -> &'static str {
                Self::kind_str(self)
            }

            fn mnemonic(self) -> Option<&'static str> {
                Some(Self::mnemonic(self))
            }
        }
    };
}

op_kind_table!(CacheMaintenanceOp, 11, with_mnemonic);
op_kind_table!(TlbMaintenanceOp, 10, with_mnemonic);
op_kind_table!(BarrierOp, 3, with_mnemonic);
op_kind_table!(HypervisorOp, 20, with_mnemonic);
op_kind_table!(HardwareEngineOp, 12, with_mnemonic);
op_kind_table!(InterruptReturnOp, 5, with_mnemonic);
op_kind_table!(BreakpointOp, 3, with_mnemonic);
op_kind_table!(TrapOp, 9, with_mnemonic);
op_kind_table!(SysRegOp, 28, with_mnemonic);
op_kind_table!(HintOp, 19, with_mnemonic);
op_kind_table!(MachineStateOp, 140, with_mnemonic);
op_kind_table!(SystemTransactionKind, 6, with_mnemonic);
op_kind_table!(PacKind, 4, with_mnemonic);
op_kind_table!(FlagAdjustKind, 17, with_mnemonic);
op_kind_table!(BcdAdjustKind, 6);
op_kind_table!(BlockStringKind, 3);

/// One registered table, flattened so the cross-table checks can hold every
/// table at once without being generic.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct TableInfo {
    /// The enum's name, for diagnostics.
    pub(crate) name: &'static str,
    /// The declared variant count.
    pub(crate) count: u16,
    /// Whether `from_index(COUNT - 1)` resolves — half of the count pin.
    pub(crate) last_index_resolves: bool,
    /// Whether `from_index(COUNT)` resolves — it must not.
    pub(crate) count_index_resolves: bool,
    /// Every variant's `kind_str` and optional `mnemonic`, in order.
    pub(crate) entries: Vec<(&'static str, Option<&'static str>)>,
}

/// Flattens one table into a [`TableInfo`].
#[cfg(test)]
fn table_info<K: OpKindTable>() -> TableInfo {
    TableInfo {
        name: K::NAME,
        count: K::COUNT,
        last_index_resolves: K::COUNT
            .checked_sub(1)
            .is_some_and(|last| K::from_index(last).is_some()),
        count_index_resolves: K::from_index(K::COUNT).is_some(),
        entries: K::all()
            .map(|kind| (kind.kind_str(), kind.mnemonic()))
            .collect(),
    }
}

/// Returns every registered operation-kind table.
///
/// The one place the registry lives, so a cross-table check is written once
/// rather than once per enum. Adding an enum here is by hand — see the module
/// doc's open hole.
#[cfg(test)]
pub(crate) fn all_tables() -> Vec<TableInfo> {
    vec![
        table_info::<CacheMaintenanceOp>(),
        table_info::<TlbMaintenanceOp>(),
        table_info::<BarrierOp>(),
        table_info::<HypervisorOp>(),
        table_info::<HardwareEngineOp>(),
        table_info::<InterruptReturnOp>(),
        table_info::<BreakpointOp>(),
        table_info::<TrapOp>(),
        table_info::<SysRegOp>(),
        table_info::<HintOp>(),
        table_info::<MachineStateOp>(),
        table_info::<SystemTransactionKind>(),
        table_info::<PacKind>(),
        table_info::<FlagAdjustKind>(),
        table_info::<BcdAdjustKind>(),
        table_info::<BlockStringKind>(),
    ]
}
