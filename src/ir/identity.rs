//! An instruction's identity, as the function's own machinery assigns it.
//!
//! Identity comes from the function's allocator, so that ids stay unique
//! within the function: the construction, the edit sessions and the repairs
//! stamp and retire it, and a caller only reads it ([`SsaInstruction::id`](crate::ir::instruction::SsaInstruction::id)).

use std::{fmt, num::NonZeroU32};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Stamping and retiring an instruction's identity.
pub trait Identity {
    /// Assigns this instruction's identity.
    fn set_id(&mut self, id: InstrId);

    /// Forgets this instruction's identity, where an instruction is destroyed
    /// rather than moved: what remains in the slot is a placeholder, and the
    /// id it held is retired rather than inherited.
    fn clear_identity(&mut self);
}

/// Identity of one instruction within its function.
///
/// An instruction's position - the `(block, index)` pair every API addresses it
/// by - changes whenever anything is inserted, removed or moved before it, so a
/// caller's table keyed by position describes a different instruction after the
/// first edit. This is the identity that does not move: it is minted once, it
/// travels with the instruction when a pass relocates it, and it is never
/// reused after the instruction is removed.
///
/// A duplicated instruction is a *different* instruction and gets its own id;
/// the function records where a copy came from, through
/// [`SsaFunction::instruction_provenance`](crate::ir::function::SsaFunction::instruction_provenance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InstrId(NonZeroU32);

impl InstrId {
    /// The id as a raw number, for a host that persists one.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// Rebuilds an id from [`get`](Self::get).
    ///
    /// Answers `None` for zero, which names no instruction.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        match NonZeroU32::new(raw) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

impl fmt::Display for InstrId {
    /// Formats the id as `i<number>`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i{}", self.0.get())
    }
}

/// Per-function allocator producing instruction identities.
///
/// Counts up and never reissues: an id belongs to at most one instruction over
/// the whole life of the function, so a stale entry in a caller's table is
/// absent rather than pointing at whatever was created next.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FunctionInstrAllocator {
    /// The next id to hand out. Starts at 1, since `InstrId` is non-zero.
    next: u32,
}

impl FunctionInstrAllocator {
    /// Creates an allocator that has issued nothing.
    #[must_use]
    pub const fn new() -> Self {
        Self { next: 1 }
    }

    /// Issues the next identity.
    ///
    /// Saturates at [`u32::MAX`], which a function reaches only by minting four
    /// billion instructions; the ids then stop being unique, which the verifier
    /// reports as `DuplicateInstructionId` rather than silently accepting.
    pub fn alloc(&mut self) -> InstrId {
        let id = InstrId(NonZeroU32::new(self.next.max(1)).unwrap_or(NonZeroU32::MIN));
        self.next = self.next.saturating_add(1).max(1);
        id
    }

    /// Returns how many identities have been issued.
    #[must_use]
    pub const fn count(&self) -> usize {
        (self.next as usize).saturating_sub(1)
    }
}

impl Default for FunctionInstrAllocator {
    fn default() -> Self {
        Self::new()
    }
}
