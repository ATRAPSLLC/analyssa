//! Exception handler preservation through SSA transformations.
//!
//! This module provides [`SsaExceptionHandler`] which preserves exception handler metadata
//! from the original method body through SSA construction, optimization, and code generation.
//! Without this preservation, exception handler regions would become invalid when SSA
//! transformations change instruction sizes or reorder blocks.
//!
//! # Preservation Pipeline
//!
//! 1. **Original offsets**: The raw IL byte offsets for try/handler regions are captured
//!    during initial SSA construction and stored verbatim in each `SsaExceptionHandler`.
//! 2. **Block mapping**: During SSA construction, the offset-based regions are translated
//!    to block-index-based regions ([`BlockRange`]s, one per clause part).
//! 3. **Block remapping**: During canonicalization (`remap_block_indices`), block indices
//!    are updated to reflect block removal and renumbering.
//! 4. **Offset regeneration**: During code generation, block offsets are used to compute
//!    new IL offsets for the output method body.
//!
//! # Layout contract
//!
//! A clause part is a **half-open range of block indices**, and the blocks it
//! covers are exactly the members of that range. Nothing narrower is
//! representable and nothing wider is: a protected region is `[start, end)` and
//! never a scattered set.
//!
//! That places one obligation on the frontend: **number blocks in layout
//! order**, so the blocks of one clause part are contiguous. Codegen already
//! depends on it — regenerating an IL exception table means writing a byte
//! range per part, and a byte range is contiguous by definition, so a part whose
//! blocks are scattered has no representation on the way out either. The
//! contract is stated here rather than assumed; [`BlockRange::to_bitset`] is the
//! crate's only expansion of a range into a set, so it is the one place the
//! assumption is spent.
//!
//! # Two views of one clause
//!
//! - [`SsaExceptionHandler::parts`] and [`SsaExceptionHandler::entry_blocks`]
//!   are **total**. They never validate and never reject, because their
//!   consumers are the transformations that must fence a clause *whatever* it
//!   says: a protected range a host wrote wrongly still bounds the code a merge
//!   may not cross.
//! - [`SsaExceptionHandler::layout`] is the **checked** view, used by the
//!   verifier and by structured recovery. It answers `Ok(None)` for a clause
//!   whose blocks lie outside this function — the funclet case — and
//!   [`ExceptionTableError`] for one that cannot be laid out at all.
//!
//! # Edge Cases
//!
//! - **Empty block removal**: `remap_block_indices` maps each part by its
//!   *members*, so a part whose blocks are all removed disappears whole and one
//!   that keeps some shrinks to them. A part can never survive as half of
//!   itself.
//! - **Multiple handlers**: Each handler is preserved independently; there is no limit
//!   on the number of handlers per method.
//! - **Filter handlers**: The `class_token_or_filter` field serves double duty as either
//!   a caught exception type token or the offset of a filter expression, depending on
//!   [`SsaExceptionHandler::kind`].

use crate::{BitSet, target::Target};

/// What a handler does when the protected region raises.
///
/// The crate's single handler taxonomy: [`Target::handler_kind`] classifies a
/// host's opaque `ExceptionKind` into it, and structured recovery reports it
/// back. There is deliberately no second copy for native ISA hosts — two enums
/// with the same four variants can disagree, and nothing would catch it.
///
/// A host maps its own encoding onto these four:
///
/// | Variant | CIL | SEH (x86 Windows) | DWARF (Linux) |
/// |---------|-----|-------------------|---------------|
/// | `Catch` | `EXCEPTION` | `__except` block | landing pad |
/// | `Filter` | `FILTER` | `__except(filter)` expression | — |
/// | `Finally` | `FINALLY` | `__finally` block | — |
/// | `Fault` | `FAULT` | vectored exception handler | — |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandlerKind {
    /// Runs for a matching exception type.
    Catch,

    /// Runs a filter expression first, and the body only if it accepts.
    Filter,

    /// Runs on every exit from the region, raising or not.
    Finally,

    /// Runs only when the region raises, without catching.
    Fault,
}

/// Which part of an exception clause a [`BlockRange`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ClausePart {
    /// The guarded code: the `try` region.
    Protected,

    /// The handler body.
    Handler,

    /// The filter expression, for a [`HandlerKind::Filter`] clause.
    Filter,
}

impl ClausePart {
    /// Returns the part's name, for the [`Display`](std::fmt::Display) form.
    const fn name(self) -> &'static str {
        match self {
            ClausePart::Protected => "protected",
            ClausePart::Handler => "handler",
            ClausePart::Filter => "filter",
        }
    }
}

impl std::fmt::Display for ClausePart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A non-empty half-open range of block indices, `[start, end)`.
///
/// One clause part is one of these. The pairing is the whole point: a start
/// without its end is not a value this type can hold, so the half-mapped clause
/// — a `try` that begins somewhere and ends nowhere — is deleted by the type
/// rather than reported by a checker. Partiality that *is* legitimate lives one
/// level up, as `Option<BlockRange>` per part: a funclet handler maps no block
/// of this function, and canonicalization can remove every block of a part.
///
/// Empty and reversed ranges are equally unrepresentable, so
/// [`is_empty`](Self::is_empty) is always `false` and exists only because a
/// range type without it invites the caller to write `len() == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "(usize, usize)", into = "(usize, usize)")
)]
pub struct BlockRange {
    /// First block of the range.
    start: usize,

    /// One past the last block of the range; always greater than `start`.
    end: usize,
}

impl BlockRange {
    /// Returns the range `[start, end)`, or `None` when it is empty.
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Option<Self> {
        if start >= end {
            return None;
        }
        Some(Self { start, end })
    }

    /// Returns the first block of the range.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// Returns one past the last block of the range.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end
    }

    /// Returns how many blocks the range covers; always at least one.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns whether the range covers no block.
    ///
    /// Always `false`: an empty range is not constructible. Present so a caller
    /// reaching for the usual collection vocabulary gets the answer rather than
    /// computing it.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns whether `block` is a member.
    ///
    /// Total: any index at all is answered, and one outside the range simply is
    /// not a member.
    #[must_use]
    pub const fn contains(&self, block: usize) -> bool {
        self.start <= block && block < self.end
    }

    /// Returns the members in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + use<> {
        self.start..self.end
    }

    /// Returns whether the two ranges share a block.
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Returns whether every block of `other` is a member of `self`.
    #[must_use]
    pub const fn contains_range(&self, other: &Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Expands the range into a set of `width` bits, or `None` when it does not
    /// fit.
    ///
    /// The crate's only interval-to-set expansion, and the one place the
    /// [layout contract](self#layout-contract) is spent. `None` when
    /// [`end`](Self::end) exceeds `width`, so a caller cannot hand a consumer a
    /// set narrower than the graph it will be read against.
    #[must_use]
    pub fn to_bitset(&self, width: usize) -> Option<BitSet> {
        if self.end > width {
            return None;
        }
        let mut bits = BitSet::new(width);
        for block in self.iter() {
            bits.insert_checked(block);
        }
        Some(bits)
    }

    /// Remaps the range through `block_remap`, by its members.
    ///
    /// Collects the new indices of the blocks in `[start, end)` that survive and
    /// rebuilds the range as `[min, max + 1)`; `None` when none survives, which
    /// is a part canonicalization removed entirely.
    ///
    /// Exact when `block_remap` is order-preserving, which compaction is: the
    /// survivors are then consecutive in the new numbering and `[min, max + 1)`
    /// is exactly the set of them. Under a non-monotone remap the result is a
    /// superset of the survivors, which is the conservative direction for both
    /// of the things a range is used as — a transformation barrier and a region
    /// membership set.
    ///
    /// # Arguments
    ///
    /// * `block_remap` - A slice indexed by old block id, where each entry is
    ///   `Some(new_id)` for a block that was kept and renumbered, and `None` for
    ///   one that was removed. An index past the slice is a removed block.
    #[must_use]
    pub fn remap(&self, block_remap: &[Option<usize>]) -> Option<Self> {
        let mut min: Option<usize> = None;
        let mut max: usize = 0;
        for old in self.iter() {
            let Some(new) = block_remap.get(old).copied().flatten() else {
                continue;
            };
            min = Some(min.map_or(new, |current: usize| current.min(new)));
            max = max.max(new);
        }
        Self::new(min?, max.checked_add(1)?)
    }
}

impl From<BlockRange> for (usize, usize) {
    fn from(range: BlockRange) -> Self {
        (range.start, range.end)
    }
}

impl TryFrom<(usize, usize)> for BlockRange {
    type Error = EmptyBlockRange;

    fn try_from((start, end): (usize, usize)) -> Result<Self, Self::Error> {
        Self::new(start, end).ok_or(EmptyBlockRange { start, end })
    }
}

/// A `(start, end)` pair that is not a [`BlockRange`] because it covers no
/// block.
///
/// Reachable only through deserialization, which is the one path that can
/// present a pair the constructor never approved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error("block range [{start}, {end}) covers no block")]
pub struct EmptyBlockRange {
    /// The proposed first block.
    pub start: usize,

    /// The proposed exclusive end.
    pub end: usize,
}

/// Why an exception clause cannot be laid out over a function's blocks.
///
/// Every variant describes a clause that could not regenerate an IL exception
/// table, which is what the block ranges exist for. Notably absent are a
/// half-mapped part and an empty part: [`BlockRange`] makes both
/// unrepresentable, so neither is an error anything can report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum ExceptionTableError {
    /// A part names a block the function does not have.
    #[error("{part} range ends at {end}, past the function's {block_count} block(s)")]
    OutOfRange {
        /// The part whose range runs past the function.
        part: ClausePart,
        /// The part's exclusive end.
        end: usize,
        /// How many blocks the function has.
        block_count: usize,
    },

    /// Two parts of one clause claim the same block.
    #[error("the {first} and {second} ranges of one clause overlap")]
    OverlappingParts {
        /// The earlier part, in [`ClausePart`] declaration order.
        first: ClausePart,
        /// The later part.
        second: ClausePart,
    },

    /// A filter handler has no filter range.
    #[error("a filter handler has no filter range")]
    FilterWithoutRange,

    /// A clause has a filter range but is not a filter handler.
    #[error("a {kind:?} handler has a filter range")]
    RangeWithoutFilterKind {
        /// What the clause says it is.
        kind: HandlerKind,
    },
}

/// What a laid-out handler does, carrying its filter's blocks where it has one.
///
/// Split from [`HandlerKind`], which only classifies, because a laid-out clause
/// has one more thing to say and one fewer way to say it wrongly: the filter's
/// blocks belong to exactly [`LaidOutHandler::Filter`], so a filter with no
/// expression and an expression on a `finally` are unrepresentable rather than
/// checked for a second time by every reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaidOutHandler {
    /// Runs for a matching exception type.
    Catch,

    /// Runs the expression occupying the given blocks first, and the body only
    /// if it accepts.
    Filter(BlockRange),

    /// Runs on every exit from the region, raising or not.
    Finally,

    /// Runs only when the region raises, without catching.
    Fault,
}

impl LaidOutHandler {
    /// Returns what the handler does.
    #[must_use]
    pub const fn kind(&self) -> HandlerKind {
        match self {
            LaidOutHandler::Catch => HandlerKind::Catch,
            LaidOutHandler::Filter(_) => HandlerKind::Filter,
            LaidOutHandler::Finally => HandlerKind::Finally,
            LaidOutHandler::Fault => HandlerKind::Fault,
        }
    }

    /// Returns the filter expression's blocks, for a filter handler.
    #[must_use]
    pub const fn filter(&self) -> Option<BlockRange> {
        match self {
            LaidOutHandler::Filter(blocks) => Some(*blocks),
            LaidOutHandler::Catch | LaidOutHandler::Finally | LaidOutHandler::Fault => None,
        }
    }
}

/// One exception clause laid out over a function's blocks.
///
/// The checked reading of an [`SsaExceptionHandler`]: every part it names exists
/// in the function, the parts do not claim each other's blocks, and the filter
/// range sits inside the kind that has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClauseLayout {
    /// What the handler does, and where its filter expression is.
    pub kind: LaidOutHandler,

    /// The guarded blocks.
    pub protected: BlockRange,

    /// The handler body's blocks.
    pub handler: BlockRange,
}

/// Exception handler information preserved in SSA form.
///
/// Stores both the original IL byte offsets and the SSA block mapping for
/// a single exception handler clause (try/catch/finally/fault/filter). The original
/// offsets are preserved verbatim from the method body; block ranges are set during
/// SSA construction and remapped during canonicalization and code generation.
///
/// # Fields
///
/// | Field | Purpose |
/// |-------|---------|
/// | `flags` | Host-defined exception handler kind, classified by [`Target::handler_kind`] |
/// | `try_offset`/`try_length` | Original IL byte range of the protected try block |
/// | `handler_offset`/`handler_length` | Original IL byte range of the handler code |
/// | `class_token_or_filter` | Caught exception type token or filter offset |
/// | `*_range` fields | SSA block ranges (set during SSA construction) |
///
/// Each block range is `Option<BlockRange>`, and that is the only partiality a
/// clause has: a part is mapped or it is not. See [`BlockRange`] for why a part
/// cannot be half of itself.
///
/// Generic over the host `Target` so `flags` carries a host-defined exception-kind type.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T::ExceptionKind: serde::Serialize, T::TypeRef: serde::Serialize",
        deserialize = "T::ExceptionKind: serde::Deserialize<'de>, T::TypeRef: serde::Deserialize<'de>"
    ))
)]
pub struct SsaExceptionHandler<T: Target> {
    /// Host-defined flags identifying the handler kind:
    /// EXCEPTION, FILTER, FINALLY, or FAULT. For CIL targets, this is
    /// `ExceptionHandlerFlags` from the original method metadata.
    pub flags: T::ExceptionKind,

    /// Original IL byte offset of the protected try region start.
    pub try_offset: u32,

    /// Length of the protected try region in IL bytes.
    pub try_length: u32,

    /// Original IL byte offset of the handler code start.
    pub handler_offset: u32,

    /// Length of the handler code in IL bytes.
    pub handler_length: u32,

    /// Dual-purpose field: for EXCEPTION handlers this is the metadata token
    /// of the caught exception type; for FILTER handlers this is the IL offset
    /// of the filter expression.
    pub class_token_or_filter: u32,

    /// SSA blocks of the protected try region.
    ///
    /// `None` when no block of this function is protected by the clause, which
    /// is the ordinary state for a clause whose code lives in another function.
    pub protected_range: Option<BlockRange>,

    /// SSA blocks of the handler body.
    ///
    /// `None` when the handler is a funclet — a separate function, so no block
    /// of this one belongs to it.
    pub handler_range: Option<BlockRange>,

    /// SSA blocks of the filter expression.
    ///
    /// Meaningful only for a [`HandlerKind::Filter`] clause. Stored rather than
    /// derived: a filter can be laid out either side of its handler, so there is
    /// no rule that recovers its extent from the other two parts.
    pub filter_range: Option<BlockRange>,
}

impl<T: Target> SsaExceptionHandler<T> {
    /// Returns what this handler does.
    #[must_use]
    pub fn kind(&self) -> HandlerKind {
        T::handler_kind(&self.flags)
    }

    /// Returns the filter offset, for a filter handler.
    ///
    /// Gated on [`kind`](Self::kind), so `class_token_or_filter` is read as an
    /// IL offset only where it is one.
    #[must_use]
    pub fn filter_offset(&self) -> Option<u32> {
        if self.kind() == HandlerKind::Filter {
            Some(self.class_token_or_filter)
        } else {
            None
        }
    }

    /// Checks if block ranges have been set for offset remapping.
    #[must_use]
    pub fn has_block_mapping(&self) -> bool {
        self.protected_range.is_some() && self.handler_range.is_some()
    }

    /// Returns every mapped part of the clause, in [`ClausePart`] order.
    ///
    /// **Total**: it validates nothing and rejects nothing. Every barrier the
    /// crate raises around exception structure reads this, and a barrier must
    /// fence even for a clause that cannot be laid out — a malformed clause is
    /// exactly the one whose code must not be moved before somebody looks at it.
    pub fn parts(&self) -> impl Iterator<Item = (ClausePart, BlockRange)> + use<T> {
        [
            (ClausePart::Protected, self.protected_range),
            (ClausePart::Handler, self.handler_range),
            (ClausePart::Filter, self.filter_range),
        ]
        .into_iter()
        .filter_map(|(part, range)| range.map(|range| (part, range)))
    }

    /// Returns the blocks control can enter without a terminator naming them:
    /// the handler entry, then the filter entry.
    ///
    /// **Total**, on the same terms as [`parts`](Self::parts).
    pub fn entry_blocks(&self) -> impl Iterator<Item = usize> + use<T> {
        [self.handler_range, self.filter_range]
            .into_iter()
            .flatten()
            .map(|range| range.start())
    }

    /// Lays the clause out over a function of `block_count` blocks.
    ///
    /// The checked counterpart to [`parts`](Self::parts), and the only reading
    /// of a clause that can refuse one. Three answers:
    ///
    /// - `Ok(Some(layout))` — the clause covers blocks of this function and is
    ///   consistent.
    /// - `Ok(None)` — the protected or handler part maps no block of this
    ///   function. Legal and common: on Windows x64 a handler is a funclet, a
    ///   separate function, so a per-function view cannot contain its blocks.
    /// - `Err` — the clause names blocks the function does not have, claims one
    ///   block for two parts, or disagrees with itself about being a filter.
    ///
    /// # Errors
    ///
    /// Returns [`ExceptionTableError`] for a clause that could not regenerate an
    /// IL exception table.
    pub fn layout(&self, block_count: usize) -> Result<Option<ClauseLayout>, ExceptionTableError> {
        for (part, range) in self.parts() {
            if range.end() > block_count {
                return Err(ExceptionTableError::OutOfRange {
                    part,
                    end: range.end(),
                    block_count,
                });
            }
        }

        let (Some(protected), Some(handler)) = (self.protected_range, self.handler_range) else {
            return Ok(None);
        };

        let kind = match (self.kind(), self.filter_range) {
            (HandlerKind::Filter, Some(filter)) => LaidOutHandler::Filter(filter),
            (HandlerKind::Filter, None) => return Err(ExceptionTableError::FilterWithoutRange),
            (kind, Some(_)) => return Err(ExceptionTableError::RangeWithoutFilterKind { kind }),
            (HandlerKind::Catch, None) => LaidOutHandler::Catch,
            (HandlerKind::Finally, None) => LaidOutHandler::Finally,
            (HandlerKind::Fault, None) => LaidOutHandler::Fault,
        };

        let mapped: Vec<(ClausePart, BlockRange)> = self.parts().collect();
        for (index, (first, first_range)) in mapped.iter().enumerate() {
            for (second, second_range) in mapped.iter().skip(index.saturating_add(1)) {
                if first_range.overlaps(second_range) {
                    return Err(ExceptionTableError::OverlappingParts {
                        first: *first,
                        second: *second,
                    });
                }
            }
        }

        Ok(Some(ClauseLayout {
            kind,
            protected,
            handler,
        }))
    }

    /// Remaps every block range using the provided canonicalization remapping.
    ///
    /// Called during [`crate::ir::function::SsaFunction::canonicalize`] to update exception handler block
    /// references after empty blocks are removed and remaining blocks are renumbered.
    ///
    /// Each part is remapped by its *members* — see [`BlockRange::remap`] — so a
    /// part shrinks to the blocks of it that survive and disappears when none
    /// does. An exclusive end therefore never advances past the part's own
    /// blocks and cannot swallow a neighbouring part, and no part can come back
    /// as half of itself.
    ///
    /// # Arguments
    ///
    /// * `block_remap` - A slice indexed by old block ID, where each entry is:
    ///   - `Some(new_id)` if the block was kept and renumbered to `new_id`
    ///   - `None` if the block was removed
    ///
    /// # Example
    ///
    /// ```text
    /// Before canonicalization: blocks [0, 1, 2, 3, 4]  (block 1 removed)
    /// After canonicalization:  blocks [0, 2, 3, 4] → renumbered [0, 1, 2, 3]
    /// block_remap = [Some(0), None, Some(1), Some(2), Some(3)]
    /// ```
    pub fn remap_block_indices(&mut self, block_remap: &[Option<usize>]) {
        self.protected_range = self
            .protected_range
            .and_then(|range| range.remap(block_remap));
        self.handler_range = self
            .handler_range
            .and_then(|range| range.remap(block_remap));
        self.filter_range = self.filter_range.and_then(|range| range.remap(block_remap));
    }
}

/// A read-only index of the block roles an exception table assigns.
///
/// An exception table says three different things about blocks, and this is
/// where each is read:
///
/// - a **runtime entry** is a block control can enter without any terminator
///   naming it — a handler entry or a filter entry. It is why entry-rooted
///   reachability under-approximates a lifted function;
/// - a **region start** and a **region end** bound a protected range, which is
///   a barrier to transformations that would move code across it.
///
/// # Borrowing
///
/// This borrows the handler slice rather than copying it. An owned snapshot
/// threaded through a mutating fixpoint — `passes::blockmerge` is the live
/// example — is stale after the first iteration that edits the table, and
/// nothing would catch it. Holding the borrow makes that a compile error.
///
/// # Totality
///
/// Built from [`SsaExceptionHandler::parts`], so a clause that cannot be laid
/// out still contributes its roles. Every accessor answers for any `usize`. An
/// index outside the function's block count is not a member of anything, and a
/// clause naming a block the function does not have contributes nothing — the
/// same rule
/// [`SsaCfg::from_ssa`](crate::analysis::cfg::SsaCfg::from_ssa) already applies
/// to an out-of-range successor.
#[derive(Debug, Clone)]
pub struct ExceptionBlocks<'a, T: Target> {
    /// The table these roles were read from; held to pin the borrow.
    handlers: &'a [SsaExceptionHandler<T>],
    /// Handler entries then filter entries, sorted and deduplicated.
    runtime_entries: Vec<usize>,
    /// Handler entries alone, sorted and deduplicated.
    handler_starts: Vec<usize>,
    /// Filter entries alone, sorted and deduplicated.
    filter_starts: Vec<usize>,
    /// Protected-region first blocks, sorted and deduplicated.
    region_starts: Vec<usize>,
    /// Protected-region exclusive end blocks, sorted and deduplicated.
    region_ends: Vec<usize>,
}

impl<'a, T: Target> ExceptionBlocks<'a, T> {
    /// Indexes the block roles in `handlers`, ignoring blocks `>= block_count`.
    #[must_use]
    pub fn from_handlers(handlers: &'a [SsaExceptionHandler<T>], block_count: usize) -> Self {
        let mut handler_starts = Vec::new();
        let mut filter_starts = Vec::new();
        let mut region_starts = Vec::new();
        let mut region_ends = Vec::new();

        for handler in handlers {
            for (part, range) in handler.parts() {
                let target = match part {
                    ClausePart::Protected => {
                        if range.end() < block_count {
                            region_ends.push(range.end());
                        }
                        &mut region_starts
                    }
                    ClausePart::Handler => &mut handler_starts,
                    ClausePart::Filter => &mut filter_starts,
                };
                if range.start() < block_count {
                    target.push(range.start());
                }
            }
        }

        let tidy = |blocks: &mut Vec<usize>| {
            blocks.sort_unstable();
            blocks.dedup();
        };
        tidy(&mut handler_starts);
        tidy(&mut filter_starts);
        tidy(&mut region_starts);
        tidy(&mut region_ends);

        let mut runtime_entries = handler_starts.clone();
        runtime_entries.extend_from_slice(&filter_starts);
        tidy(&mut runtime_entries);

        Self {
            handlers,
            runtime_entries,
            handler_starts,
            filter_starts,
            region_starts,
            region_ends,
        }
    }

    /// Whether the table assigns no in-function block role at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.runtime_entries.is_empty() && self.region_starts.is_empty()
    }

    /// Whether control can enter `block` without a terminator naming it.
    #[must_use]
    pub fn is_runtime_entry(&self, block: usize) -> bool {
        self.runtime_entries.binary_search(&block).is_ok()
    }

    /// Whether `block` begins a protected region.
    #[must_use]
    pub fn is_region_start(&self, block: usize) -> bool {
        self.region_starts.binary_search(&block).is_ok()
    }

    /// Whether `block` is a protected region's exclusive end.
    #[must_use]
    pub fn is_region_end(&self, block: usize) -> bool {
        self.region_ends.binary_search(&block).is_ok()
    }

    /// Every block control can enter without a terminator naming it, ascending.
    #[must_use]
    pub fn runtime_entries(&self) -> &[usize] {
        &self.runtime_entries
    }

    /// Every handler entry block, ascending.
    #[must_use]
    pub fn handler_starts(&self) -> &[usize] {
        &self.handler_starts
    }

    /// Every filter entry block, ascending.
    #[must_use]
    pub fn filter_starts(&self) -> &[usize] {
        &self.filter_starts
    }

    /// The runtime entries of every clause protecting a region that starts at
    /// `entry_block`.
    ///
    /// Several clauses can share one protected range — `try/catch(A)/catch(B)`
    /// is two rows with the same try start — so this can yield more than one.
    pub fn try_entries_of(&self, entry_block: usize) -> impl Iterator<Item = usize> + '_ {
        self.handlers
            .iter()
            .filter(move |handler| {
                handler.protected_range.map(|range| range.start()) == Some(entry_block)
            })
            .flat_map(SsaExceptionHandler::entry_blocks)
            .filter(|block| self.is_runtime_entry(*block))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockTarget;

    /// A clause whose three parts are disjoint: protected `[0, 2)`, handler
    /// `[3, 5)`, filter `[5, 6)`.
    ///
    /// The filter sits *after* the handler on purpose. It is where a CIL filter
    /// often lands, and it is the layout no rule deriving the filter's extent
    /// from its neighbours can express.
    fn handler(flags: u32) -> SsaExceptionHandler<MockTarget> {
        SsaExceptionHandler {
            flags,
            try_offset: 10,
            try_length: 20,
            handler_offset: 30,
            handler_length: 40,
            class_token_or_filter: 50,
            protected_range: BlockRange::new(0, 2),
            handler_range: BlockRange::new(3, 5),
            filter_range: BlockRange::new(5, 6),
        }
    }

    #[test]
    fn block_range_refuses_empty_reversed_and_over_wide_expansions() {
        assert_eq!(BlockRange::new(2, 2), None, "an empty range is not a range");
        assert_eq!(BlockRange::new(3, 1), None, "nor is a reversed one");

        let range = match BlockRange::new(1, 3) {
            Some(range) => range,
            None => unreachable!("[1, 3) is a range"),
        };
        assert_eq!(range.len(), 2);
        assert!(!range.is_empty());
        assert!(range.contains(1) && range.contains(2));
        assert!(!range.contains(0) && !range.contains(3));
        assert_eq!(range.iter().collect::<Vec<_>>(), vec![1, 2]);

        assert!(
            range.to_bitset(2).is_none(),
            "[1, 3) does not fit in 2 bits"
        );
        let bits = match range.to_bitset(4) {
            Some(bits) => bits,
            None => unreachable!("[1, 3) fits in 4 bits"),
        };
        assert_eq!(bits.len(), 4, "the set has the width it was asked for");
        assert_eq!(bits.iter().collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn block_ranges_report_overlap_and_containment() {
        let outer = match BlockRange::new(0, 4) {
            Some(range) => range,
            None => unreachable!("[0, 4) is a range"),
        };
        let inner = match BlockRange::new(1, 3) {
            Some(range) => range,
            None => unreachable!("[1, 3) is a range"),
        };
        let after = match BlockRange::new(4, 6) {
            Some(range) => range,
            None => unreachable!("[4, 6) is a range"),
        };

        assert!(outer.overlaps(&inner) && inner.overlaps(&outer));
        assert!(
            !outer.overlaps(&after),
            "half-open ranges abut, not overlap"
        );
        assert!(outer.contains_range(&inner));
        assert!(!inner.contains_range(&outer));
    }

    /// Every accessor answers for any index, including ones no block has.
    ///
    /// The table is attacker-shaped input: a clause can name any `usize`, and a
    /// membership query is made per block by several passes. An asserting
    /// lookup here would be a reachable abort.
    #[test]
    fn exception_blocks_answers_out_of_range_indices_without_panicking() {
        let handlers = [handler(0)];
        let blocks = ExceptionBlocks::from_handlers(&handlers, 6);

        for index in [usize::MAX, 6, 1_000_000] {
            assert!(!blocks.is_runtime_entry(index));
            assert!(!blocks.is_region_start(index));
            assert!(!blocks.is_region_end(index));
            assert_eq!(blocks.try_entries_of(index).count(), 0);
        }
    }

    #[test]
    fn a_handler_and_a_filter_entry_are_both_runtime_entries() {
        let handlers = [handler(0)];
        let blocks = ExceptionBlocks::from_handlers(&handlers, 6);

        assert_eq!(blocks.handler_starts(), &[3]);
        assert_eq!(blocks.filter_starts(), &[5]);
        assert_eq!(blocks.runtime_entries(), &[3, 5]);
        assert!(blocks.is_region_start(0));
        assert!(blocks.is_region_end(2));
        assert!(!blocks.is_empty());
    }

    /// A clause may name a block the function does not have -- a stale index
    /// that survived canonicalization. It contributes no role, matching the
    /// rule `SsaCfg::from_ssa` already applies to an out-of-range successor.
    #[test]
    fn a_clause_naming_a_block_the_function_lacks_contributes_nothing() {
        let handlers = [handler(0)];
        let blocks = ExceptionBlocks::from_handlers(&handlers, 4);

        assert_eq!(blocks.handler_starts(), &[3], "block 3 exists");
        assert!(
            blocks.filter_starts().is_empty(),
            "block 5 does not exist in a 4-block function"
        );
        assert_eq!(blocks.runtime_entries(), &[3]);
    }

    /// Sibling catches share one protected range, so one region start can have
    /// several runtime entries.
    #[test]
    fn try_entries_of_finds_every_clause_sharing_a_protected_range() {
        let mut second = handler(0);
        second.handler_range = BlockRange::new(5, 6);
        second.filter_range = None;
        let handlers = [handler(0), second];
        let blocks = ExceptionBlocks::from_handlers(&handlers, 6);

        let mut entries: Vec<usize> = blocks.try_entries_of(0).collect();
        entries.sort_unstable();
        entries.dedup();
        assert_eq!(entries, vec![3, 5]);
    }

    #[test]
    fn an_empty_table_assigns_no_roles() {
        let blocks = ExceptionBlocks::<MockTarget>::from_handlers(&[], 4);
        assert!(blocks.is_empty());
        assert!(blocks.runtime_entries().is_empty());
    }

    /// `MockTarget` reads its `u32` flags as `0 = Catch, 1 = Filter,
    /// 2 = Finally, 3 = Fault`, so the filter offset is readable for exactly
    /// one of them.
    #[test]
    fn a_filter_offset_is_read_only_for_a_filter_handler() {
        assert_eq!(handler(0).kind(), HandlerKind::Catch);
        assert_eq!(handler(0).filter_offset(), None);
        assert_eq!(handler(1).kind(), HandlerKind::Filter);
        assert_eq!(handler(1).filter_offset(), Some(50));
        assert_eq!(handler(2).filter_offset(), None);
        assert_eq!(handler(3).filter_offset(), None);
    }

    #[test]
    fn block_mapping_requires_protected_and_handler_ranges() {
        let mut mapped = handler(0);
        assert!(mapped.has_block_mapping());

        mapped.handler_range = None;
        assert!(!mapped.has_block_mapping());
    }

    /// `parts()` and `entry_blocks()` are the total view, and every barrier in
    /// the crate is computed from them. A clause that `layout()` rejects outright
    /// must still fence, so neither may drop anything.
    #[test]
    fn parts_and_entry_blocks_never_reject_a_malformed_clause() {
        let mut malformed = handler(0);
        // Every defect at once: the parts overlap, the ranges run past any
        // plausible function, and a catch carries a filter range.
        malformed.protected_range = BlockRange::new(90, 100);
        malformed.handler_range = BlockRange::new(95, 105);
        malformed.filter_range = BlockRange::new(99, 110);

        assert!(malformed.layout(4).is_err(), "the clause is malformed");

        let parts: Vec<(ClausePart, BlockRange)> = malformed.parts().collect();
        assert_eq!(parts.len(), 3, "every mapped part is still reported");
        assert_eq!(
            parts.iter().map(|(part, _)| *part).collect::<Vec<_>>(),
            vec![
                ClausePart::Protected,
                ClausePart::Handler,
                ClausePart::Filter
            ]
        );
        assert_eq!(
            malformed.entry_blocks().collect::<Vec<_>>(),
            vec![95, 99],
            "the handler entry, then the filter entry"
        );
    }

    /// The five outcomes of the checked view, in one table.
    #[test]
    fn layout_classifies_unmapped_and_malformed_clauses() {
        let well_formed = handler(1).layout(6);
        assert_eq!(
            well_formed,
            Ok(Some(ClauseLayout {
                kind: LaidOutHandler::Filter(BlockRange { start: 5, end: 6 }),
                protected: BlockRange { start: 0, end: 2 },
                handler: BlockRange { start: 3, end: 5 },
            }))
        );
        let laid_out = match well_formed {
            Ok(Some(layout)) => layout.kind,
            _ => unreachable!("the clause lays out"),
        };
        assert_eq!(laid_out.kind(), HandlerKind::Filter);
        assert_eq!(laid_out.filter(), BlockRange::new(5, 6));
        assert_eq!(LaidOutHandler::Finally.kind(), HandlerKind::Finally);
        assert_eq!(LaidOutHandler::Fault.filter(), None);

        let mut funclet = handler(0);
        funclet.handler_range = None;
        funclet.filter_range = None;
        assert_eq!(
            funclet.layout(6),
            Ok(None),
            "a handler in another function is legal and simply not laid out"
        );

        assert_eq!(
            handler(1).layout(5),
            Err(ExceptionTableError::OutOfRange {
                part: ClausePart::Filter,
                end: 6,
                block_count: 5,
            })
        );

        let mut catch_with_filter = handler(0);
        assert_eq!(
            catch_with_filter.layout(6),
            Err(ExceptionTableError::RangeWithoutFilterKind {
                kind: HandlerKind::Catch,
            })
        );
        catch_with_filter.flags = 1;
        catch_with_filter.filter_range = None;
        assert_eq!(
            catch_with_filter.layout(6),
            Err(ExceptionTableError::FilterWithoutRange)
        );

        let mut overlapping = handler(0);
        overlapping.protected_range = BlockRange::new(0, 4);
        overlapping.filter_range = None;
        assert_eq!(
            overlapping.layout(6),
            Err(ExceptionTableError::OverlappingParts {
                first: ClausePart::Protected,
                second: ClausePart::Handler,
            })
        );
    }

    /// A part keeps exactly the blocks of it that survived, and its exclusive
    /// end is one past the last of them.
    ///
    /// Remapping the end *index* instead — and, when its block is gone, scanning
    /// forward for any surviving block at all — reaches past the part into
    /// whatever comes next, and finds nothing when nothing comes next, leaving a
    /// start with no end. Mapping by members has neither failure: the answer is
    /// derived from the part's own blocks and from nothing else.
    #[test]
    fn a_part_shrinks_to_its_surviving_blocks() {
        let mut mapped = handler(0);

        // Blocks 1, 2, 4 and 5 are removed; 0 -> 0 and 3 -> 1.
        mapped.remap_block_indices(&[Some(0), None, None, Some(1), None, None]);

        assert_eq!(
            mapped.protected_range,
            BlockRange::new(0, 1),
            "protected [0, 2) keeps only block 0"
        );
        assert_eq!(
            mapped.handler_range,
            BlockRange::new(1, 2),
            "handler [3, 5) keeps only block 3, and ends one past it"
        );
        assert_eq!(
            mapped.filter_range, None,
            "filter [5, 6) lost its only block"
        );
        assert!(
            mapped.has_block_mapping(),
            "the two parts that kept a block are both whole"
        );
    }

    /// A part whose blocks all vanish disappears; it never survives as a start
    /// with no end, or an end with no start.
    #[test]
    fn an_emptied_part_is_dropped_whole() {
        let mut mapped = handler(0);

        // A remap table shorter than the function: blocks 3, 4 and 5 are past
        // it, which is how a stale index looks from here.
        mapped.remap_block_indices(&[None, Some(0), Some(1)]);

        assert_eq!(
            mapped.protected_range,
            BlockRange::new(0, 1),
            "protected [0, 2) lost block 0 and kept block 1, renumbered to 0"
        );
        assert_eq!(mapped.handler_range, None, "no handler block survived");
        assert_eq!(mapped.filter_range, None, "no filter block survived");
        assert!(!mapped.has_block_mapping());
    }

    /// A non-monotone remap yields a superset of the survivors rather than an
    /// exact answer, which is the conservative direction for a barrier and for a
    /// region membership set alike.
    #[test]
    fn a_non_monotone_remap_yields_a_superset() {
        let mut mapped = handler(0);
        mapped.protected_range = BlockRange::new(0, 3);
        mapped.handler_range = None;
        mapped.filter_range = None;

        // Blocks 0 and 2 survive, renumbered far apart; block 1 is removed.
        mapped.remap_block_indices(&[Some(5), None, Some(1)]);

        assert_eq!(
            mapped.protected_range,
            BlockRange::new(1, 6),
            "the survivors are 1 and 5, so the range spans them"
        );
    }
}
