//! What a static execution establishes about the function it ran.
//!
//! Every fact is the join of what every visit observed, so it holds on every
//! path the execution explored - and an execution that explored every path
//! there is makes it a fact about every run of the program. One that did not
//! finish established nothing: its facts are withheld, and [`Completion`] says
//! why.

use std::{
    collections::{BTreeMap, BTreeSet, btree_map},
    mem,
    ops::Range,
    sync::Arc,
};

use rustc_hash::FxHashMap;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    analysis::{
        budget::Exhaustion,
        execution::{
            InstructionSources,
            edges::{EdgeObservation, EdgeRecords},
            findings::Finding,
            memory::{ImageContents, ImageWrite, Memory, MemoryRefusal},
            trace::Trace,
            uses::{UseSite, Uses},
            value::{Address, Value},
        },
    },
    ir::{value::ConstValue, variable::SsaVarId},
    target::Target,
};

/// How an execution ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Completion {
    /// A memory operation lacked a required lifetime, extent or address contract.
    Memory {
        /// Block containing the refused operation.
        block: usize,
        /// The missing memory contract.
        reason: MemoryRefusal,
    },
    /// The host cancelled execution; no observations are published.
    Cancelled,
    /// Every path ran to its end: the facts hold on every run.
    Complete,
    /// A budget ran out first.
    Exhausted(Exhaustion),
    /// A path reached something the execution cannot follow, at the end of
    /// this block.
    Stuck {
        /// The block.
        block: usize,
    },
    /// The host could not supply supported semantics for an accessed type.
    ///
    /// No observations are published.
    UnsupportedType,
    /// The host could not prove that the interpreted instruction snapshot stays valid.
    ///
    /// No observations are published.
    UnstableCode,
    /// A body's machine interpretation differs from this execution's context.
    ///
    /// No observations are published.
    IncompatibleMachine,
    /// The configured host refused call effects that execution would leave unevaluated.
    ///
    /// This includes an unanswered call or a termination shortcut that skips
    /// its body or model. No observations are published.
    UnevaluatedEffects,
    /// An execution identity namespace has reached its representable end.
    IdentityExhausted,
}

/// Why borrowed observation payload sizing refused publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationSizeRefusal {
    /// The publication owner cancelled the traversal.
    Cancelled,
    /// Retained payload exceeds the owner bound or its count cannot be represented.
    OutputLimit,
}

/// The actual control action that entered another source body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CodeTransferKind {
    /// An ordinary call entered a new activation.
    Call,
    /// A tail call replaced the current activation.
    TailCall,
    /// A computed branch continued the current activation in another body.
    Branch,
    /// A write changed the code the current activation ran, which went on in
    /// a lift of what the code holds now.
    Rewritten,
}

/// What every visit of each call site resolved its target to, in one trial's
/// body and instruction namespaces: the method a computed target reached,
/// and the one a direct call actually entered, which is not the one it
/// names where the code there was written since (a recovered version).
///
/// A site keeps its method only while every visit resolved to that same
/// method; a visit that resolved to another method or to none leaves the
/// site without one. A visit the trial never made records nothing.
#[derive(Debug, Clone)]
pub struct CallTargets<M> {
    /// The one method each visited `(body, block, instruction)` reached on
    /// every visit, or `None` once visits disagreed or failed to resolve.
    sites: BTreeMap<(usize, usize, usize), Option<M>>,
}

impl<M> Default for CallTargets<M> {
    fn default() -> Self {
        Self {
            sites: BTreeMap::new(),
        }
    }
}

impl<M: Clone + Eq> CallTargets<M> {
    /// Records one visit of the call at `site` of `body`, which resolved to
    /// `method` or to nothing.
    pub fn record(&mut self, body: usize, site: (usize, usize), method: Option<&M>) {
        self.sites
            .entry((body, site.0, site.1))
            .and_modify(|held| {
                if held.as_ref() != method {
                    *held = None;
                }
            })
            .or_insert_with(|| method.cloned());
    }

    /// Returns the method every visit of the call at `block` and
    /// `instruction` of `body` resolved to, when they all resolved to one.
    #[must_use]
    pub fn unique(&self, body: usize, block: usize, instruction: usize) -> Option<&M> {
        self.sites.get(&(body, block, instruction))?.as_ref()
    }
}

/// An admitted source transfer in one root trial's body and instruction namespaces.
///
/// A record means a path in the completed trial took this transfer. Different
/// targets at the same source site remain separate possibilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CodeTransfer {
    /// Source body index in the trial's method table.
    pub from_body: usize,
    /// Source block in that exact SSA body.
    pub block: usize,
    /// Instruction index within the source block.
    pub instruction: usize,
    /// Entered body index in the trial's method table.
    pub to_body: usize,
    /// Actual activation or source action.
    pub kind: CodeTransferKind,
}

/// What proves where an observed run of bytes ends ([`ArgumentBytes::bound`]).
///
/// The run itself may stop earlier, at the first byte that is not known; the
/// bound is how far the storage or its use is proven to reach from the
/// run's address, and a run as long as its bound is the whole of it
/// ([`ArgumentBytes::complete`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ByteBound {
    /// Nothing does: the bytes are the known prefix of storage that may
    /// continue past them, into an adjacent object.
    #[default]
    Prefix,
    /// The use reads or writes no further than `bytes` from the address.
    Use {
        /// The bytes the use reaches from the address.
        bytes: u64,
        /// What states the use.
        by: UseBound,
    },
    /// The address lies in an object of `bytes` bytes, whose first byte is
    /// `start` in the address's root.
    Object {
        /// The offset of the object's first byte in the address's root.
        start: i64,
        /// The object's length in bytes.
        bytes: u64,
        /// What kind of storage the object is.
        kind: ObjectKind,
    },
}

/// What states how far a use reaches ([`ByteBound::Use`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UseBound {
    /// The call's complete memory contract (`CallEffects`).
    Contract,
    /// The library model the call runs ([`CallModel::observation_bound`]).
    ///
    /// [`CallModel::observation_bound`]: crate::analysis::execution::CallModel::observation_bound
    Model,
    /// The callee's own accesses, every one counted, on a complete run.
    Callee,
}

/// The storage an object is ([`ByteBound::Object`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ObjectKind {
    /// A live heap allocation, as long as every path that reached it
    /// allocated.
    Heap,
    /// A live activation's frame, the range the host promises writable
    /// throughout it (`ExecutionHost::stack_extent`), or an object in the
    /// outermost one whose end the host proves.
    Frame,
    /// An image object whose extent the host proves: a sized symbol or a
    /// recovered data object.
    Global,
}

/// Where an object's extent comes from: whether the end it states is the
/// object's own, or one recovered from how the code uses the storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExtentSource {
    /// The allocator that made the object states its size: exact.
    Allocator,
    /// The image's symbol states the object's size, where its container
    /// records one (ELF `st_size`): exact.
    Symbol,
    /// The host attests the range for the run (a live activation's frame,
    /// [`ExecutionHost::stack_extent`]): exact.
    ///
    /// [`ExecutionHost::stack_extent`]: crate::analysis::execution::ExecutionHost::stack_extent
    Attested,
    /// A frame address and what lies above it up to the first cell the
    /// function saves a register into, or up to its entry stack pointer where
    /// none lies above: an end no pointer the program forms into a local
    /// crosses in bounds, whichever local it is into. Exact as a bound from
    /// above: the object may end below it, never above, and may begin below
    /// the address, which only bounds the end ([`StorageObject::holds`]).
    SaveArea,
    /// Recovered from how the code uses the storage (a frame slot, a data
    /// object's layout): inferred, an end the code suggests and nothing
    /// states.
    Recovered,
}

impl ExtentSource {
    /// Whether the extent is one no access in bounds runs past, so an access
    /// past it is one past the object.
    #[must_use]
    pub const fn exact(self) -> bool {
        !matches!(self, Self::Recovered)
    }
}

/// An object of storage the host states an address lies in
/// ([`ObjectExtents::object`]).
///
/// [`ObjectExtents::object`]: crate::analysis::execution::ObjectExtents::object
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StorageObject {
    /// The offset of the object's first byte in the address's root.
    pub start: i64,
    /// The object's length in bytes.
    pub bytes: u64,
    /// What kind of storage the object is.
    pub kind: ObjectKind,
    /// Where its extent comes from.
    pub source: ExtentSource,
}

impl StorageObject {
    /// Whether an access of `len` bytes at `address` stays within the
    /// object: every byte inside it. An extent bounded only from above
    /// ([`ExtentSource::SaveArea`]) holds every access that ends at or below
    /// its end, wherever it starts: the save above a local bounds a pointer
    /// to it, and nothing states where the local begins.
    #[must_use]
    pub fn holds(self, address: Address, len: u64) -> bool {
        if self.source == ExtentSource::SaveArea {
            let end = i128::from(self.start).saturating_add(i128::from(self.bytes));
            return i128::from(address.offset()).saturating_add(i128::from(len)) <= end;
        }
        self.reach(address).is_some_and(|reach| len <= reach)
    }

    /// How many bytes of the object lie at and past `address`, which must
    /// be inside it; `None` where it is not.
    #[must_use]
    pub fn reach(self, address: Address) -> Option<u64> {
        ByteBound::Object {
            start: self.start,
            bytes: self.bytes,
            kind: self.kind,
        }
        .reach(address)
    }
}

impl ByteBound {
    /// The offset in the address's root of the byte past the bound's last,
    /// for a run at `address`; `None` for [`Self::Prefix`].
    fn end(self, address: Address) -> Option<i64> {
        match self {
            Self::Prefix => None,
            Self::Use { bytes, .. } => address.offset().checked_add(i64::try_from(bytes).ok()?),
            Self::Object { start, bytes, .. } => start.checked_add(i64::try_from(bytes).ok()?),
        }
    }

    /// How many bytes the bound reaches from `address`; `None` for
    /// [`Self::Prefix`], and for an object `address` does not lie in.
    #[must_use]
    pub fn reach(self, address: Address) -> Option<u64> {
        let end = self.end(address)?;
        let offset = address.offset();
        let holds = match self {
            Self::Object { start, .. } => (start..end).contains(&offset),
            Self::Prefix | Self::Use { .. } => end >= offset,
        };
        holds.then(|| end.abs_diff(offset))
    }

    /// What holds of two visits' runs, at `mine` and `theirs`, for the joined
    /// run at `at`: the bound reaching further, since either visit may have
    /// read that far, and none where either is unbounded or the object it
    /// names does not hold `at`. A use bound is restated from `at`.
    fn loosest(self, mine: Address, other: Self, theirs: Address, at: Address) -> Self {
        let (Some(mine_end), Some(their_end)) = (self.end(mine), other.end(theirs)) else {
            return Self::Prefix;
        };
        let (bound, end) = if their_end > mine_end {
            (other, their_end)
        } else {
            (self, mine_end)
        };
        match bound {
            Self::Use { by, .. } if end >= at.offset() => Self::Use {
                bytes: end.abs_diff(at.offset()),
                by,
            },
            Self::Object { .. } if bound.reach(at).is_some() => bound,
            Self::Prefix | Self::Use { .. } | Self::Object { .. } => Self::Prefix,
        }
    }

    /// The tighter of a use bound reaching `use_reach` bytes from the
    /// address and the object it lies in, and how many bytes that is; none
    /// for an unbounded use in no known object.
    #[must_use]
    pub fn tightest(
        use_bound: Option<(u64, UseBound)>,
        object: Option<StorageObject>,
        address: Address,
    ) -> (Self, Option<u64>) {
        let object = object.and_then(|object| Some((object.reach(address)?, object)));
        match (use_bound, object) {
            (Some((bytes, by)), Some((reach, _))) if bytes < reach => {
                (Self::Use { bytes, by }, Some(bytes))
            }
            (_, Some((reach, object))) => (
                Self::Object {
                    start: object.start,
                    bytes: object.bytes,
                    kind: object.kind,
                },
                Some(reach),
            ),
            (Some((bytes, by)), None) => (Self::Use { bytes, by }, Some(bytes)),
            (None, None) => (Self::Prefix, None),
        }
    }
}

/// The known linear run behind a call's pointer argument, at the call.
///
/// The run describes memory immediately before the call instruction executes,
/// preceding a tail transfer's frame retirement. Consumers separately establish
/// whether the callee can read those bytes after the transfer. It stops at the
/// first unknown byte, at the tightest proven extent ([`Self::bound`]: the
/// call's stated use, the callee's own accesses, the object it lies in) and
/// before target-coordinate wrap. [`Address::pointer_size`] identifies its
/// coordinate width.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ArgumentBytes {
    /// The argument's index; for another use of a pointer
    /// ([`StaticFacts::uses`]), the operand's position among the
    /// instruction's operands.
    pub argument: usize,
    /// Where it points.
    pub address: Address,
    /// The known bytes from there on, up to the first that is not known.
    pub bytes: Vec<u8>,
    /// The extent the bytes are proven to lie within.
    pub bound: ByteBound,
    /// For a heap address, the call of the outermost activation, `(block,
    /// instruction)`, whose execution allocated the object it lies in, where
    /// every visit agrees on one.
    pub allocated_at: Option<(usize, usize)>,
    /// Which of the bytes the run wrote, on every visit; the rest are the
    /// image's own, unchanged.
    #[cfg_attr(feature = "serde", serde(default))]
    pub written: WrittenRanges,
}

/// Which bytes of an observed run the run wrote, by their offset from its
/// first, ascending and apart: every other byte is the image's own content
/// at its address, unchanged. A run of bytes the image holds tells nothing
/// the file does not.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WrittenRanges(Vec<Range<usize>>);

impl WrittenRanges {
    /// The ranges of the offsets `written` names, which ascend.
    pub fn of_offsets(written: impl IntoIterator<Item = usize>) -> Self {
        let mut ranges: Vec<Range<usize>> = Vec::new();
        for offset in written {
            match ranges.last_mut() {
                Some(last) if last.end == offset => last.end = offset.saturating_add(1),
                _ => ranges.push(offset..offset.saturating_add(1)),
            }
        }
        Self(ranges)
    }

    /// Every byte of a run `len` bytes long.
    #[must_use]
    pub fn whole(len: usize) -> Self {
        Self(
            (len != 0)
                .then_some(Range { start: 0, end: len })
                .into_iter()
                .collect(),
        )
    }

    /// The written ranges, ascending.
    #[must_use]
    pub fn ranges(&self) -> &[Range<usize>] {
        &self.0
    }

    /// Whether the run wrote none of the bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether the run wrote every byte of a run `len` bytes long.
    #[must_use]
    pub fn covers(&self, len: usize) -> bool {
        len == 0 || matches!(self.0.as_slice(), [all] if all.start == 0 && all.end == len)
    }

    /// Whether the run wrote the byte at `offset`.
    #[must_use]
    pub fn contains(&self, offset: usize) -> bool {
        self.0.iter().any(|range| range.contains(&offset))
    }

    /// The ranges of the run's first `len` bytes.
    pub fn truncate(&mut self, len: usize) {
        self.0.retain(|range| range.start < len);
        if let Some(last) = self.0.last_mut() {
            last.end = last.end.min(len);
        }
    }

    /// The ranges of the run's bytes from `start` to `end`, as offsets from
    /// `start`.
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self(
            self.0
                .iter()
                .filter_map(|range| {
                    let from = range.start.max(start);
                    let to = range.end.min(end);
                    (from < to).then(|| from.saturating_sub(start)..to.saturating_sub(start))
                })
                .collect(),
        )
    }

    /// The bytes written on both of two visits of one run, `other`'s first
    /// byte lying `shift` bytes into it.
    #[must_use]
    pub fn meet(&self, other: &Self, shift: usize) -> Self {
        let mut met = Vec::new();
        for mine in &self.0 {
            for theirs in &other.0 {
                let from = mine.start.max(theirs.start.saturating_add(shift));
                let to = mine.end.min(theirs.end.saturating_add(shift));
                if from < to {
                    met.push(from..to);
                }
            }
        }
        Self(met)
    }
}

/// A call's owned argument observations, ordered by its source instruction.
///
/// The values describe the original entry activation's pre-instruction memory.
/// Tail callees inherit the return continuation while retaining their own SSA
/// identities; their instructions never acquire the root's observation indices.
#[derive(Debug)]
pub struct CallObservation {
    /// Caller block index.
    pub block: usize,
    /// Instruction index within the caller block.
    pub instruction: usize,
    /// Retained argument bytes, moved without cloning their payloads.
    pub arguments: Vec<ArgumentBytes>,
}

impl ArgumentBytes {
    /// Whether the bytes are the whole of their bound: every byte the use or
    /// the object reaches from the address is known.
    #[must_use]
    pub fn complete(&self) -> bool {
        self.bound
            .reach(self.address)
            .is_some_and(|reach| u64::try_from(self.bytes.len()).is_ok_and(|len| len == reach))
    }

    /// Restricts the run to a use reaching `bytes` from its address, stated
    /// by `by`, where that is tighter than the bound it has.
    pub fn restrict(&mut self, bytes: u64, by: UseBound) {
        if self
            .bound
            .reach(self.address)
            .is_none_or(|reach| bytes < reach)
        {
            self.bound = ByteBound::Use { bytes, by };
        }
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        self.bytes.truncate(len);
        self.written.truncate(len);
    }

    /// What holds on both of two visits: the bytes both saw, when both point
    /// at one place, within the looser of the two bounds.
    fn join(&self, other: &Self) -> Option<Self> {
        if self.argument != other.argument || self.address != other.address {
            return None;
        }
        let mut common = 0usize;
        for (mine, theirs) in self.bytes.iter().zip(&other.bytes) {
            if mine != theirs {
                break;
            }
            common = common.saturating_add(1);
        }
        if common == 0 {
            return None;
        }
        Some(Self {
            argument: self.argument,
            address: self.address,
            bytes: self.bytes.get(..common)?.to_vec(),
            bound: self
                .bound
                .loosest(self.address, other.bound, other.address, self.address),
            allocated_at: self
                .allocated_at
                .filter(|site| other.allocated_at == Some(*site)),
            written: {
                let mut written = self.written.meet(&other.written, 0);
                written.truncate(common);
                written
            },
        })
    }

    /// What holds on both of two visits of a use that walks a buffer: the
    /// bytes from the lower of the two addresses, where the other visit reads
    /// within them, in the same storage, and sees what they hold there; up to
    /// the first byte the two disagree on. A loop comparing a string a byte at
    /// a time reads one byte further on each visit, and every visit reads the
    /// same string.
    fn walked(&self, other: &Self) -> Option<Self> {
        if self.argument != other.argument
            || self.address.root() != other.address.root()
            || self.address.pointer_size() != other.address.pointer_size()
        {
            return None;
        }
        if other.address.offset() < self.address.offset() {
            return other.walked(self);
        }
        let distance =
            usize::try_from(other.address.offset().checked_sub(self.address.offset())?).ok()?;
        let along = self.bytes.get(distance..)?;
        if along.is_empty() {
            return None;
        }
        let agree = along
            .iter()
            .zip(&other.bytes)
            .take_while(|(mine, theirs)| mine == theirs)
            .count();
        let keep = distance.checked_add(agree)?;
        // A byte before the other visit's first is this visit's alone; from
        // there on both visits read it.
        let mut written = WrittenRanges::of_offsets((0..keep).filter(|offset| {
            self.written.contains(*offset)
                && (*offset < distance || other.written.contains(offset.saturating_sub(distance)))
        }));
        written.truncate(keep);
        (keep > 0).then(|| Self {
            argument: self.argument,
            address: self.address,
            bytes: self.bytes.get(..keep).unwrap_or_default().to_vec(),
            bound: self
                .bound
                .loosest(self.address, other.bound, other.address, self.address),
            allocated_at: self
                .allocated_at
                .filter(|site| other.allocated_at == Some(*site)),
            written,
        })
    }
}

/// Work consumed by one bounded execution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExecutionUsage {
    /// Admitted instruction, control and phi steps charged to this execution.
    ///
    /// A refused step or phi-batch demand appears in [`Exhaustion`], without
    /// increasing consumed usage past the configured step allowance.
    pub steps: usize,
    /// Additional paths explored beyond the initial invocation.
    pub paths: usize,
    /// Instructions proven to trap rather than reaching their next instruction.
    #[cfg_attr(feature = "serde", serde(default))]
    pub traps: usize,
    /// Path-state snapshots created for alternatives and arrival joins.
    #[cfg_attr(feature = "serde", serde(default))]
    pub state_snapshots: usize,
    /// Sides of a fork no input satisfies, given the conditions their path
    /// took before: never explored, and not among [`Self::paths`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub infeasible_paths: usize,
}

/// What one whole run of a loop leaves behind: the bytes it wrote, and what
/// its values hold as control leaves it.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopEffect<T: Target> {
    /// The loop's header block.
    pub header: usize,
    /// The edge control leaves the loop by, as `(from, to)`.
    pub exit: (usize, usize),
    /// The bytes the run leaves written, as runs of consecutive addresses in
    /// address order.
    pub writes: Vec<(Address, Vec<u8>)>,
    /// What each value defined in the loop holds when control leaves it, in
    /// id order.
    pub exit_values: Vec<(SsaVarId, Value<T>)>,
}

/// Evidence that an instruction can be replaced without losing an observable access or trap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplacementProof {
    /// Every visit completed the operation without its arithmetic fault: a
    /// checked operation or conversion did not overflow, a division neither
    /// divided by zero nor overflowed where the machine traps on either, and a
    /// finite check received a finite value.
    Completed,
    /// Every visit made the same aligned ordinary read of lifetime-immutable image bytes.
    ReadOnlyImage {
        /// Address in the host's symbol space.
        address: u64,
        /// Size of the read in bytes.
        len: u64,
    },
    /// Every visit made an ordinary read wholly inside storage the run knows
    /// is live, at an alignment that cannot fault: a live activation's frame,
    /// or a live allocation's extent.
    Accessible,
}

/// What every visit to one loop left behind.
#[derive(Debug, Clone, PartialEq)]
enum LoopRecord<T: Target> {
    /// Every visit so far was one run with this effect.
    Effect(LoopEffect<T>),
    /// Some visit was not, or two visits differed.
    Spoiled,
}

/// Identifies an actual image read in the unchanged source body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReadFootprint {
    /// Source block containing the read.
    pub block: usize,
    /// Source instruction within the block.
    pub instruction: usize,
    /// First byte read in unsigned image coordinates.
    pub address: u64,
    /// Actual read extent, excluding unvisited scan elements.
    pub len: u64,
    /// Whether this span supplies the instruction's scalar result.
    pub produces: bool,
}

/// What an execution observed, joined over every visit.
#[derive(Debug, Clone)]
pub struct Observations<T: Target> {
    /// Each value the outermost activation defined, by what every definition
    /// of it held.
    values: FxHashMap<SsaVarId, Value<T>>,
    /// Replacement evidence common to every visit, including trapping visits.
    replacements: FxHashMap<SsaVarId, Option<ReplacementProof>>,
    /// Each call the outermost activation made, by `(block, instruction)`:
    /// the pointer arguments every visit saw the same bytes behind.
    calls: FxHashMap<(usize, usize), Vec<ArgumentBytes>>,
    /// Each other use of a pointer the outermost activation made, by
    /// `(block, instruction)` and then operand: the bytes it had written that
    /// every visit knowing the pointer saw behind it, or `None` once one saw
    /// other bytes or none it wrote ([`StaticFacts::uses`]).
    uses: FxHashMap<(usize, usize), BTreeMap<usize, Option<ArgumentBytes>>>,
    /// The bytes it had written behind the pointer every return returned,
    /// joined; `None` inside once a return showed none.
    returned_bytes: Option<Option<ArgumentBytes>>,
    /// Requested target-entry bytes and scalars, never definition replacement facts.
    pub edges: EdgeRecords<T>,
    /// Each loop of the outermost activation, by header.
    loops: FxHashMap<usize, LoopRecord<T>>,
    /// Read footprints per loop, sharing witnesses across nested visits.
    loop_reads: FxHashMap<usize, BTreeSet<ReadFootprint>>,
    /// Per-footprint value agreement, independent of loop output agreement.
    reads: FxHashMap<ReadFootprint, Option<ConstValue<T>>>,
    /// Explicit image writes common to normal returns in this source domain.
    exit_memory: Option<ImageContents>,
    /// What every return returned, joined; `None` inside for a return without
    /// a value.
    returned: Option<Option<Value<T>>>,
}

impl<T: Target> Default for Observations<T> {
    fn default() -> Self {
        Self {
            values: FxHashMap::default(),
            replacements: FxHashMap::default(),
            calls: FxHashMap::default(),
            uses: FxHashMap::default(),
            returned_bytes: None,
            edges: EdgeRecords::default(),
            loops: FxHashMap::default(),
            loop_reads: FxHashMap::default(),
            reads: FxHashMap::default(),
            exit_memory: None,
            returned: None,
        }
    }
}

impl<T: Target> Observations<T> {
    /// The facts of an execution that ended as `completion` having observed
    /// these, withholding observations unless it completed.
    pub fn into_facts(self, completion: Completion) -> StaticFacts<T> {
        let observed = match completion {
            Completion::Complete => self,
            Completion::UnsupportedType
            | Completion::UnstableCode
            | Completion::IncompatibleMachine
            | Completion::UnevaluatedEffects
            | Completion::IdentityExhausted
            | Completion::Memory { .. }
            | Completion::Cancelled
            | Completion::Exhausted(_)
            | Completion::Stuck { .. } => Observations::default(),
        };
        StaticFacts {
            completion,
            observed,
            body_methods: Vec::new(),
            code_transfers: Vec::new(),
            code_transfers_exhausted: false,
            call_targets: CallTargets::default(),
            instruction_sources: None,
            invocations: Vec::new(),
            calls_truncated: false,
            edge_output_exhausted: false,
            root_entry: true,
            usage: ExecutionUsage::default(),
            findings: Vec::new(),
        }
    }

    /// Returns whether another visit can still establish common replacement evidence.
    pub fn needs_replacement_evidence(&self, value: SsaVarId) -> bool {
        !matches!(self.replacements.get(&value), Some(None))
    }

    /// Intersects replacement evidence, retaining failed or conflicting visits.
    pub fn replacement(&mut self, value: SsaVarId, proof: Option<ReplacementProof>) {
        self.replacements
            .entry(value)
            .and_modify(|known| {
                if *known != proof {
                    *known = None;
                }
            })
            .or_insert(proof);
    }

    /// Records that a definition of `value` held `held`.
    pub fn value(&mut self, value: SsaVarId, held: &Value<T>) {
        self.values
            .entry(value)
            .and_modify(|known| *known = known.join(held))
            .or_insert_with(|| held.clone());
    }

    /// Returns pointer coordinates retained at one actual call site.
    pub fn call_pointers(&self, site: (usize, usize)) -> Vec<(usize, Address)> {
        self.calls
            .get(&site)
            .into_iter()
            .flatten()
            .map(|argument| (argument.argument, argument.address))
            .collect()
    }

    /// Restricts pre-call snapshots after complete access coverage has been collected.
    pub fn bound_call_uses(&mut self, owner: Option<usize>, uses: &Uses) {
        for (position, arguments) in &mut self.calls {
            uses.restrict(
                UseSite {
                    owner,
                    position: *position,
                },
                arguments,
            );
        }
    }

    /// Records the bytes behind a call's pointer arguments on one visit.
    ///
    /// # Arguments
    ///
    /// * `site` - The call, as `(block, instruction)`.
    /// * `arguments` - What this visit saw behind each pointer argument.
    pub fn call(&mut self, site: (usize, usize), arguments: Vec<ArgumentBytes>) {
        match self.calls.get_mut(&site) {
            None => {
                self.calls.insert(site, arguments);
            }
            Some(known) => {
                // Both records are in argument-index order. Merge once instead
                // of searching every incoming argument for every retained one.
                let mut incoming = arguments.iter().peekable();
                let mut joined = Vec::new();
                for mine in known.iter() {
                    while incoming
                        .peek()
                        .is_some_and(|theirs| theirs.argument < mine.argument)
                    {
                        incoming.next();
                    }
                    if let Some(theirs) = incoming
                        .peek()
                        .filter(|theirs| theirs.argument == mine.argument)
                        && let Some(common) = mine.join(theirs)
                    {
                        joined.push(common);
                    }
                }
                *known = joined;
            }
        }
    }

    /// Records one visit of the pointer use at `site`: each operand whose
    /// pointer the visit knew, with the bytes the activation wrote behind it,
    /// or `None` where it wrote none there, joined with what earlier visits
    /// saw. A use that walks a buffer reads further along it on each visit,
    /// and keeps the buffer from where the walk began
    /// ([`ArgumentBytes::walked`]). An operand a visit does not know the
    /// pointer of, as after paths with different counts merge, is no evidence
    /// either way; one that saw other bytes, or none written, ends it.
    pub fn use_site(&mut self, site: (usize, usize), seen: Vec<(usize, Option<ArgumentBytes>)>) {
        let known = self.uses.entry(site).or_default();
        for (operand, theirs) in seen {
            match known.entry(operand) {
                btree_map::Entry::Vacant(slot) => {
                    slot.insert(theirs);
                }
                btree_map::Entry::Occupied(mut slot) => {
                    let joined = slot
                        .get()
                        .as_ref()
                        .zip(theirs.as_ref())
                        .and_then(|(mine, theirs)| mine.walked(theirs));
                    slot.insert(joined);
                }
            }
        }
    }

    /// Records the bytes the activation wrote behind the pointer one return
    /// returned, or `None` where it returned no such pointer, joined with
    /// what earlier returns returned.
    pub fn returned_pointer(&mut self, bytes: Option<ArgumentBytes>) {
        self.returned_bytes = Some(match self.returned_bytes.take() {
            None => bytes,
            Some(known) => known
                .zip(bytes)
                .and_then(|(mine, theirs)| mine.join(&theirs)),
        });
    }

    /// Retains a performed image access for each enclosing clean loop.
    ///
    /// Read witnesses are unions: differing legitimate source ranges do not
    /// weaken agreement about the loop's writes and live-out values.
    pub fn read(
        &mut self,
        trace: &Trace,
        site: (usize, usize),
        footprint: (Address, usize),
        produces: bool,
        value: Option<&ConstValue<T>>,
    ) {
        let (address, len) = footprint;
        let Some(address) = address.image_address().filter(|_| len != 0) else {
            return;
        };
        let key = ReadFootprint {
            block: site.0,
            instruction: site.1,
            address,
            len: len as u64,
            produces,
        };
        let mut retained = false;
        for header in trace.clean_headers() {
            self.loop_reads.entry(header).or_default().insert(key);
            retained = true;
        }
        if retained {
            self.reads
                .entry(key)
                .and_modify(|known| {
                    if known.as_ref() != value {
                        *known = None;
                    }
                })
                .or_insert_with(|| value.cloned());
        }
    }

    /// Records one visit to the loop headed by `header`: its effect, or
    /// `None` when it had none the execution can state.
    pub fn loop_visit(&mut self, header: usize, effect: Option<LoopEffect<T>>) {
        let record = match (self.loops.get(&header), effect) {
            (_, None) | (Some(LoopRecord::Spoiled), _) => LoopRecord::Spoiled,
            (None, Some(effect)) => LoopRecord::Effect(effect),
            (Some(LoopRecord::Effect(known)), Some(effect)) if known == &effect => {
                LoopRecord::Effect(effect)
            }
            (Some(LoopRecord::Effect(_)), Some(_)) => LoopRecord::Spoiled,
        };
        self.loops.insert(header, record);
    }

    /// Records one return of the observed activation, which activation
    /// `since` entered: the image bytes it and the activations it ran wrote.
    pub fn exit(&mut self, memory: &Memory, returned: Option<Value<T>>, since: u32) {
        match &mut self.exit_memory {
            Some(known) => known.intersect(memory, since),
            None => self.exit_memory = Some(memory.image_contents(since)),
        }
        let returned = match &self.returned {
            None => returned,
            Some(known) => match (known, returned) {
                (Some(known), Some(returned)) => Some(known.join(&returned)),
                _ => None,
            },
        };
        self.returned = Some(returned);
    }
}

/// Facts in the callee's SSA domain at one selected structural call.
///
/// The record is conditional on reaching this call through its parent ancestry.
/// Visits to the same parent, caller, local site and entered target intersect
/// their observations; distinct structural calls retain separate domains.
/// The enclosing execution owns usage and the shared body-method table. These
/// facts carry no replacement witnesses and do not justify rewriting the callee
/// outside this call context. Incomplete enclosing executions expose no records.
#[derive(Debug, Clone)]
pub struct CallFacts<T: Target> {
    /// Immediate parent index in the enclosing invocation list, or the root.
    ///
    /// A parent precedes its children. These execution-local indices do not
    /// define persistent observation identities.
    pub parent: Option<usize>,
    /// Caller index in the enclosing body-method table.
    pub caller_body: usize,
    /// Immediate caller block containing the selected call.
    pub block: usize,
    /// Call instruction within the immediate caller block.
    pub instruction: usize,
    /// Callee index in the enclosing [`StaticFacts::body_methods`] table.
    pub body: usize,
    /// Callee-local values, calls and normal-exit memory.
    ///
    /// Usage is owned by the enclosing execution, and this record's method
    /// table is empty. Tail descendants contribute only to the inherited
    /// return and memory; their scalar identities never enter this domain.
    pub facts: StaticFacts<T>,
}

/// What a completed static execution established.
#[derive(Debug, Clone)]
pub struct StaticFacts<T: Target> {
    /// How it ended.
    completion: Completion,
    /// What it observed; withheld unless it completed.
    observed: Observations<T>,
    /// Method identities for entered bodies, with the root first.
    body_methods: Vec<Option<T::MethodRef>>,
    /// Bounded source transfers selected by the host.
    code_transfers: Vec<CodeTransfer>,
    /// Whether a distinct selected transfer exceeded the owner's limit.
    code_transfers_exhausted: bool,
    /// Every indirect call site's resolution across all of a complete trial's visits.
    call_targets: CallTargets<T::MethodRef>,
    /// Complete trial source events, shared by root and selected call observations.
    instruction_sources: Option<Arc<InstructionSources>>,
    /// Whether these observations begin with unknown root-entry inputs.
    root_entry: bool,
    /// Selected call-site observations in their separate callee SSA domains.
    invocations: Vec<CallFacts<T>>,
    /// Whether a complete execution omitted actual entries at its observation limit.
    calls_truncated: bool,
    /// An edge payload exceeded the enclosing trial reservation; semantics still completed.
    edge_output_exhausted: bool,
    /// Work consumed, including paths that did not establish facts.
    usage: ExecutionUsage,
    /// What the run found wrong with the program, in order.
    findings: Vec<Finding>,
}

impl<T: Target> StaticFacts<T> {
    /// States what the run found wrong with the program.
    #[must_use]
    pub fn with_findings(mut self, findings: Vec<Finding>) -> Self {
        self.findings = findings;
        self
    }

    /// What the run found wrong with the program, on any path, at the
    /// instruction that did it: an access past the extent of its object, an
    /// access to heap storage whose lifetime had ended, a release of storage
    /// that was no live block, a use of bits no write had reached.
    ///
    /// The run goes on past each, with the machine's own semantics, and
    /// publishes its facts as it would without one.
    #[must_use]
    pub fn findings(&self) -> &[Finding] {
        &self.findings
    }

    /// Marks facts as conditional on one selected actual-call context.
    pub const fn in_call_context(mut self) -> Self {
        self.root_entry = false;
        self
    }

    /// Returns whether these facts begin at an unknown-input root entry.
    ///
    /// Call-scoped facts require their enclosing context and cannot drive
    /// unconditional rewriting of the shared callee body.
    #[must_use]
    pub const fn is_root_entry(&self) -> bool {
        self.root_entry
    }

    /// Attaches selected call records only after the enclosing execution completes.
    pub fn with_invocations(mut self, invocations: Vec<CallFacts<T>>, truncated: bool) -> Self {
        if self.is_complete() {
            self.invocations = invocations;
            self.calls_truncated = truncated;
        }
        self
    }

    /// Returns selected calls without merging their SSA domains or charging usage twice.
    #[must_use]
    pub fn invocations(&self) -> &[CallFacts<T>] {
        &self.invocations
    }

    /// Returns whether an actual root call entry exceeded the observation limit.
    ///
    /// Unreachable sites and repeated visits to retained site/body pairs do not
    /// set this flag. Incomplete executions withhold it along with their facts.
    #[must_use]
    pub const fn call_observations_truncated(&self) -> bool {
        self.calls_truncated
    }

    /// Moves selected call records without cloning observations or argument buffers.
    pub fn take_invocations(&mut self) -> Vec<CallFacts<T>> {
        mem::take(&mut self.invocations)
    }

    /// Retains the explorer's body identities without copying metadata.
    pub fn with_body_methods(mut self, methods: Vec<Option<T::MethodRef>>) -> Self {
        if self.is_complete() {
            self.body_methods = methods;
        }
        self
    }

    /// Retains one immutable source union without copying it into every invocation.
    pub fn with_instruction_sources(mut self, sources: Option<Arc<InstructionSources>>) -> Self {
        if self.is_complete() {
            self.instruction_sources = sources;
        }
        self
    }

    /// Returns the completed trial's union of instruction-source entry and write events.
    ///
    /// The union includes normal exits and terminated prefixes. Selected call
    /// observations share their enclosing root trial's history. Events retain
    /// the host's finite source namespace; they are not an execution-order or
    /// instruction-stability proof. The publication owner checks satisfiability
    /// across paths and other execution views. Incomplete trials and trials
    /// with no source events return `None`; no empty history is allocated.
    #[must_use]
    pub fn instruction_sources(&self) -> Option<&InstructionSources> {
        self.instruction_sources.as_deref()
    }

    /// Returns the method identities of bodies entered by this completed execution.
    ///
    /// The root is first and may have no host method identity. Repeated entries
    /// of the same body under the same method share an entry. Distinct method
    /// identities remain distinct even when a host maps them to one body.
    /// Opaque and modelled calls are not executed bodies and are not included.
    /// The table is a union across explored paths, not a call-order witness or
    /// a complete dependency key for reusing facts. Incomplete executions expose
    /// no body identities.
    #[must_use]
    pub fn body_methods(&self) -> &[Option<T::MethodRef>] {
        &self.body_methods
    }

    /// Retains every call site's resolution for a complete trial.
    #[must_use]
    pub fn with_call_targets(mut self, targets: CallTargets<T::MethodRef>) -> Self {
        if self.is_complete() {
            self.call_targets = targets;
        }
        self
    }

    /// Returns what each call site resolved to on every visit of a complete
    /// trial; empty for an incomplete one.
    #[must_use]
    pub const fn call_targets(&self) -> &CallTargets<T::MethodRef> {
        &self.call_targets
    }

    /// Retains complete, bounded control observations from the explorer.
    pub fn with_code_transfers(
        mut self,
        transfers: BTreeSet<CodeTransfer>,
        exhausted: bool,
    ) -> Self {
        if self.is_complete() {
            self.code_transfers_exhausted = exhausted;
            if !exhausted {
                self.code_transfers = transfers.into_iter().collect();
            }
        }
        self
    }

    /// Returns the admitted source transfers in deterministic order.
    ///
    /// These are scoped possible transfers, not assertions that every path
    /// takes each edge. Incomplete trials and exhausted transfer reservations
    /// expose no transfer records.
    #[must_use]
    pub fn code_transfers(&self) -> &[CodeTransfer] {
        &self.code_transfers
    }

    /// Returns whether the owner limit withheld the completed trial's transfer set.
    #[must_use]
    pub const fn code_transfers_exhausted(&self) -> bool {
        self.code_transfers_exhausted
    }

    /// Moves the retained body-method identities without cloning their payloads.
    pub fn take_body_methods(&mut self) -> Vec<Option<T::MethodRef>> {
        mem::take(&mut self.body_methods)
    }

    /// Moves a retained constant out without copying its payload or allocating a list.
    ///
    /// Other observations and replacement evidence remain available. Callers that
    /// need replacement analysis perform it before extracting its constant values.
    fn take_constant(&mut self, value: SsaVarId) -> Option<ConstValue<T>> {
        let held = self.observed.values.get(&value)?;
        held.constant()?;
        self.observed.values.remove(&value)?.into_constant()
    }

    /// Moves retained constants into variable order without cloning payloads.
    pub fn take_constants(&mut self) -> Vec<(SsaVarId, ConstValue<T>)> {
        self.map_constants(|value, constant| (value, constant))
    }

    /// Maps owned constants into variable order without cloning payloads.
    fn map_constants<R>(&mut self, mut map: impl FnMut(SsaVarId, ConstValue<T>) -> R) -> Vec<R> {
        let keys = self.constant_keys();
        let mut result = Vec::with_capacity(keys.len());
        for value in keys {
            if let Some(constant) = self.take_constant(value) {
                result.push(map(value, constant));
            }
        }
        result
    }

    /// Maps owned constants in variable order until the mapper declines a value.
    ///
    /// Returns `None` without publishing a partial collection. Transferred payloads
    /// are consumed; callers publish their outer result only after mapping succeeds.
    pub fn try_map_constants<R>(
        &mut self,
        mut map: impl FnMut(SsaVarId, ConstValue<T>) -> Option<R>,
    ) -> Option<Vec<R>> {
        let keys = self.constant_keys();
        let mut result = Vec::with_capacity(keys.len());
        for value in keys {
            result.push(map(value, self.take_constant(value)?)?);
        }
        Some(result)
    }

    /// Collects constant identities in deterministic variable order.
    fn constant_keys(&self) -> Vec<SsaVarId> {
        let mut keys: Vec<_> = self
            .observed
            .values
            .iter()
            .filter_map(|(value, held)| held.constant().map(|_| *value))
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Moves call observations into instruction order without cloning argument buffers.
    pub fn take_calls(&mut self) -> Vec<CallObservation> {
        let mut calls: Vec<_> = mem::take(&mut self.observed.calls)
            .into_iter()
            .map(|((block, instruction), arguments)| CallObservation {
                block,
                instruction,
                arguments,
            })
            .collect();
        calls.sort_unstable_by_key(|call| (call.block, call.instruction));
        calls
    }

    /// Moves requested edge observations in deterministic source-edge order.
    ///
    /// Records include visited but unavailable prefixes. A never-visited edge is
    /// absent. These point-qualified facts cannot justify context-wide rewrites.
    pub fn take_edges(&mut self) -> Vec<EdgeObservation<T>> {
        mem::take(&mut self.observed.edges).into_observations()
    }

    /// Returns whether the enclosing trial omitted edge payloads at its output bound.
    ///
    /// Execution continues after overflow. Owners must refuse publication under
    /// their output-limit policy rather than treating these as complete observations.
    pub const fn edge_output_exhausted(&self) -> bool {
        self.edge_output_exhausted
    }

    /// Attaches one enclosing-trial output admission result to its observations.
    pub fn with_edge_output_limit(mut self, exhausted: bool) -> Self {
        self.edge_output_exhausted = exhausted;
        self
    }

    /// Records work consumed by the explorer.
    pub const fn with_usage(mut self, usage: ExecutionUsage) -> Self {
        self.usage = usage;
        self
    }

    /// Returns consumed work, even when execution established no facts.
    #[must_use]
    pub const fn usage(&self) -> ExecutionUsage {
        self.usage
    }

    /// How the execution ended.
    #[must_use]
    pub const fn completion(&self) -> Completion {
        self.completion
    }

    /// Whether the execution explored every path, so its facts hold.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self.completion, Completion::Complete)
    }

    /// Returns replacement evidence common to all visits to this definition.
    ///
    /// A known constant alone does not prove this: a later visit can trap before
    /// defining its value. Interrupted or incomplete executions retain no proof.
    #[must_use]
    pub fn replacement_proof(&self, value: SsaVarId) -> Option<ReplacementProof> {
        self.observed.replacements.get(&value).copied().flatten()
    }

    /// What `value` holds on every run that defines it, when that is known.
    #[must_use]
    pub fn value(&self, value: SsaVarId) -> Option<&Value<T>> {
        self.observed.values.get(&value)
    }

    /// The constant `value` is on every run that defines it.
    #[must_use]
    pub fn constant(&self, value: SsaVarId) -> Option<&ConstValue<T>> {
        self.value(value).and_then(Value::constant)
    }

    /// Forgets what is known of each value `forget` names, before it is
    /// published or charged to an output allowance.
    ///
    /// A host withholds facts that tell a reader nothing - a value the program
    /// already states as a literal. A forgotten value is no constant here and
    /// no replacement either, which a literal never needs.
    pub fn forget_values(&mut self, mut forget: impl FnMut(SsaVarId) -> bool) {
        self.observed.values.retain(|value, _| !forget(*value));
    }

    /// Every value that is one constant on every run, in id order.
    #[must_use]
    pub fn constants(&self) -> Vec<(SsaVarId, &ConstValue<T>)> {
        let mut constants: Vec<(SsaVarId, &ConstValue<T>)> = self
            .observed
            .values
            .iter()
            .filter_map(|(value, held)| Some((*value, held.constant()?)))
            .collect();
        constants.sort_by_key(|(value, _)| value.index());
        constants
    }

    /// The known bytes behind the pointer arguments of the call at
    /// `instruction` of `block`, as every run that makes it sees them.
    #[must_use]
    pub fn call_arguments(&self, block: usize, instruction: usize) -> &[ArgumentBytes] {
        self.observed
            .calls
            .get(&(block, instruction))
            .map_or(&[], Vec::as_slice)
    }

    /// The bytes the outermost activation wrote behind the pointer operands of
    /// the other pointer uses it made, by `(block, instruction)` in order, as
    /// every visit saw them: the address a load reads at, the pointer a store
    /// writes into memory, either side of a block compare. Each operand's
    /// index is its position among the instruction's operands
    /// ([`ArgumentBytes::argument`]).
    ///
    /// Only bytes the activation itself wrote: a pointer into bytes the image
    /// holds unchanged shows nothing the image does not. Each run stops at
    /// the end of the object it lies in where one is proven, and states it
    /// ([`ArgumentBytes::bound`]).
    #[must_use]
    pub fn uses(&self) -> Vec<((usize, usize), Vec<&ArgumentBytes>)> {
        let mut uses: Vec<_> = self
            .observed
            .uses
            .iter()
            .map(|(site, operands)| (*site, operands.values().flatten().collect::<Vec<_>>()))
            .filter(|(_, bytes)| !bytes.is_empty())
            .collect();
        uses.sort_by_key(|(site, _)| *site);
        uses
    }

    /// The bytes the outermost activation wrote behind the pointer it
    /// returned, the same on every return; `None` where any return returned
    /// no such pointer.
    #[must_use]
    pub fn returned_bytes(&self) -> Option<&ArgumentBytes> {
        self.observed.returned_bytes.as_ref()?.as_ref()
    }

    /// Returns call observations in block and instruction order.
    ///
    /// Each run states the extent it is proven within ([`ArgumentBytes::bound`]):
    /// a use the call states or the callee's accesses reach, or the object it
    /// lies in; only a run that is the whole of its bound
    /// ([`ArgumentBytes::complete`]) is a complete buffer, and a
    /// [`ByteBound::Prefix`] is never one.
    #[must_use]
    pub fn calls(&self) -> Vec<((usize, usize), &[ArgumentBytes])> {
        let mut calls: Vec<_> = self
            .observed
            .calls
            .iter()
            .map(|(site, bytes)| (*site, bytes.as_slice()))
            .collect();
        calls.sort_by_key(|(site, _)| *site);
        calls
    }

    /// Every loop that leaves the same effect on every run, by header, in
    /// header order.
    #[must_use]
    pub fn decoded_loops(&self) -> Vec<usize> {
        let mut headers: Vec<usize> = self
            .observed
            .loops
            .iter()
            .filter_map(|(header, record)| {
                matches!(record, LoopRecord::Effect(_)).then_some(*header)
            })
            .collect();
        headers.sort_unstable();
        headers
    }

    /// What every whole run of the loop headed by `header` leaves behind,
    /// when every run the execution saw was one the program alone decided and
    /// all of them agree.
    #[must_use]
    pub fn loop_effect(&self, header: usize) -> Option<&LoopEffect<T>> {
        match self.observed.loops.get(&header)? {
            LoopRecord::Effect(effect) => Some(effect),
            LoopRecord::Spoiled => None,
        }
    }

    /// Returns the union of actual reads retained for a decoded loop.
    ///
    /// Positions belong to this execution's immutable source body. Consumers
    /// publish these witnesses only when the corresponding rewrite commits.
    pub fn loop_reads(
        &self,
        header: usize,
    ) -> impl Iterator<Item = (ReadFootprint, Option<&ConstValue<T>>)> + '_ {
        self.observed
            .loop_reads
            .get(&header)
            .into_iter()
            .flatten()
            .map(|key| (*key, self.observed.reads.get(key).and_then(Option::as_ref)))
    }

    /// The image bytes the function leaves written on every return, as runs
    /// by unsigned image address. Runs stop before address-space wrap.
    #[must_use]
    pub fn image_writes(&self) -> Vec<(u64, Vec<u8>)> {
        self.observed
            .exit_memory
            .as_ref()
            .map(|memory| {
                memory
                    .runs(usize::MAX)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|write| (write.address, write.bytes))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The exact image writes in unsigned coordinate order, as
    /// [`ImageWrite`]s; `None` where they total more than `maximum_bytes`.
    pub fn image_writes_within(&self, maximum_bytes: usize) -> Option<Vec<ImageWrite>> {
        match &self.observed.exit_memory {
            Some(memory) => memory.runs(maximum_bytes),
            None => Some(Vec::new()),
        }
    }

    /// Counts this scope's retained publication payload without moving or copying it.
    ///
    /// Includes call argument bytes, requested edge payload, known image-exit bytes,
    /// the bytes each decoded loop leaves written ([`Self::loop_effect`]), and
    /// the inline storage of observed constants. Fixed headers and host identities
    /// are excluded. Child invocation scopes are counted separately by their owner;
    /// a child refusal need not invalidate a previously published parent. The count
    /// uses the same constant eligibility as [`Self::try_map_constants`].
    ///
    /// # Errors
    ///
    /// Returns cancellation or an output limit refusal, including sticky omitted
    /// edge payload. No observations are consumed and no run buffers are allocated.
    pub fn observation_payload_bytes(
        &self,
        maximum_bytes: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<usize, ObservationSizeRefusal> {
        if cancelled() {
            return Err(ObservationSizeRefusal::Cancelled);
        }
        if self.edge_output_exhausted {
            return Err(ObservationSizeRefusal::OutputLimit);
        }
        let mut retained = self
            .observed
            .edges
            .payload_bytes_within(maximum_bytes, cancelled)?;
        for arguments in self.observed.calls.values() {
            if cancelled() {
                return Err(ObservationSizeRefusal::Cancelled);
            }
            for argument in arguments {
                if cancelled() {
                    return Err(ObservationSizeRefusal::Cancelled);
                }
                retained = retained
                    .checked_add(argument.bytes.len())
                    .filter(|bytes| *bytes <= maximum_bytes)
                    .ok_or(ObservationSizeRefusal::OutputLimit)?;
            }
        }
        let returned = self.observed.returned_bytes.iter().flatten();
        let used = self
            .observed
            .uses
            .values()
            .flat_map(|operands| operands.values().flatten());
        for used in used.chain(returned) {
            if cancelled() {
                return Err(ObservationSizeRefusal::Cancelled);
            }
            retained = retained
                .checked_add(used.bytes.len())
                .filter(|bytes| *bytes <= maximum_bytes)
                .ok_or(ObservationSizeRefusal::OutputLimit)?;
        }
        if let Some(memory) = &self.observed.exit_memory {
            let bytes =
                memory.payload_bytes_within(maximum_bytes.saturating_sub(retained), cancelled)?;
            retained = retained
                .checked_add(bytes)
                .filter(|bytes| *bytes <= maximum_bytes)
                .ok_or(ObservationSizeRefusal::OutputLimit)?;
        }
        for record in self.observed.loops.values() {
            if cancelled() {
                return Err(ObservationSizeRefusal::Cancelled);
            }
            if let LoopRecord::Effect(effect) = record {
                for (_, bytes) in &effect.writes {
                    retained = retained
                        .checked_add(bytes.len())
                        .filter(|bytes| *bytes <= maximum_bytes)
                        .ok_or(ObservationSizeRefusal::OutputLimit)?;
                }
            }
        }
        for held in self.observed.values.values() {
            if cancelled() {
                return Err(ObservationSizeRefusal::Cancelled);
            }
            if let Some(constant) = held.constant() {
                retained = retained
                    .checked_add(constant.payload_bytes())
                    .filter(|bytes| *bytes <= maximum_bytes)
                    .ok_or(ObservationSizeRefusal::OutputLimit)?;
            }
        }
        if cancelled() {
            return Err(ObservationSizeRefusal::Cancelled);
        }
        Ok(retained)
    }

    /// What the function returns on every return, when it returns a value.
    #[must_use]
    pub fn returned(&self) -> Option<&Value<T>> {
        self.observed.returned.as_ref()?.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;
    use crate::{
        analysis::execution::{
            WriteSite,
            trace::{LoopVisit, Trace},
            value::Root,
        },
        pointer::PointerSize,
        testing::MockTarget,
    };

    fn bytes(argument: usize, offset: i64, bytes: &[u8]) -> ArgumentBytes {
        ArgumentBytes {
            argument,
            address: Address::new(Root::Stack, offset, PointerSize::Bit64),
            bytes: bytes.to_vec(),
            bound: ByteBound::Prefix,
            allocated_at: None,
            written: WrittenRanges::whole(bytes.len()),
        }
    }

    #[test]
    fn written_ranges_keep_what_every_visit_wrote() {
        let ranges = WrittenRanges::of_offsets([0, 1, 2, 5, 6, 9]);
        assert_eq!(ranges.ranges(), &[0..3, 5..7, 9..10]);
        assert!(ranges.contains(6) && !ranges.contains(4));
        assert!(!ranges.covers(10) && WrittenRanges::whole(10).covers(10));
        assert_eq!(ranges.slice(1, 6).ranges(), &[0..2, 4..5]);
        let mut truncated = ranges.clone();
        truncated.truncate(6);
        assert_eq!(truncated.ranges(), &[0..3, 5..6]);
        // Two visits of one run: a byte the image supplied on either is the
        // image's own.
        let image_tail = ArgumentBytes {
            written: WrittenRanges::of_offsets([0, 1]),
            ..bytes(0, -16, &[7; 4])
        };
        let written = bytes(0, -16, &[7; 4]);
        assert_eq!(
            written.join(&image_tail).map(|run| run.written),
            Some(WrittenRanges::of_offsets([0, 1]))
        );
        // A walk one byte along: the first byte is the first visit's alone.
        let along = ArgumentBytes {
            written: WrittenRanges::of_offsets([1, 2]),
            ..bytes(0, -15, &[7; 3])
        };
        assert_eq!(
            written.walked(&along).map(|run| run.written),
            Some(WrittenRanges::of_offsets([0, 2, 3]))
        );
    }

    /// A frame object of `bytes` bytes beginning `start` past entry.
    const fn frame(start: i64, bytes: u64) -> ByteBound {
        ByteBound::Object {
            start,
            bytes,
            kind: ObjectKind::Frame,
        }
    }

    /// **The tightest proven extent bounds a run, and a run as long as it
    /// is whole.** A use tighter than the object bounds it; an object at
    /// least as tight as the use does, from wherever in it the run begins;
    /// an address outside the object has no object; nothing proven leaves a
    /// prefix.
    #[test]
    fn the_tightest_proven_extent_bounds_a_run() {
        let at = |offset| Address::new(Root::Stack, offset, PointerSize::Bit64);
        let object = StorageObject {
            start: -32,
            bytes: 16,
            kind: ObjectKind::Frame,
            source: ExtentSource::Attested,
        };
        let contract = |bytes| ByteBound::Use {
            bytes,
            by: UseBound::Contract,
        };
        assert_eq!(
            ByteBound::tightest(Some((4, UseBound::Contract)), Some(object), at(-32)),
            (contract(4), Some(4))
        );
        assert_eq!(
            ByteBound::tightest(Some((16, UseBound::Contract)), Some(object), at(-32)),
            (frame(-32, 16), Some(16)),
            "a tie is the object's"
        );
        assert_eq!(
            ByteBound::tightest(Some((64, UseBound::Contract)), Some(object), at(-24)),
            (frame(-32, 16), Some(8)),
            "eight of the object's bytes lie past the run's address"
        );
        assert_eq!(
            ByteBound::tightest(None, Some(object), at(-16)),
            (ByteBound::Prefix, None),
            "the byte past the object is outside it"
        );
        assert_eq!(
            ByteBound::tightest(None, None, at(0)),
            (ByteBound::Prefix, None)
        );
        for outside in [-40, -33, -16, 0] {
            assert_eq!(frame(-32, 16).reach(at(outside)), None, "{outside}");
        }
        let mut run = bytes(0, -24, &[1; 8]);
        run.bound = frame(-32, 16);
        assert!(run.complete());
        run.bytes.pop();
        assert!(!run.complete(), "a byte of the object is unknown");
        run.bound = ByteBound::Prefix;
        assert!(!run.complete(), "a prefix is never whole");
    }

    /// **Joined visits keep the looser bound, and the callee's use only
    /// tightens one.** Two visits bounded to four and eight bytes join
    /// within eight; an unbounded visit leaves the join a prefix; a walk
    /// restates a use bound from where it began and keeps an object as it
    /// is. The callee reading six of a run bounded to eight bounds it to
    /// six, and reading six of a four-byte object leaves the object.
    #[test]
    fn joined_visits_keep_the_looser_bound() {
        let contract = |bytes| ByteBound::Use {
            bytes,
            by: UseBound::Contract,
        };
        let bounded = |offset, data: &[u8], bound| ArgumentBytes {
            bound,
            ..bytes(0, offset, data)
        };
        let four = bounded(-16, &[7; 4], contract(4));
        let eight = bounded(-16, &[7; 8], contract(8));
        assert_eq!(four.join(&eight).map(|run| run.bound), Some(contract(8)));
        assert_eq!(eight.join(&four).map(|run| run.bound), Some(contract(8)));
        assert_eq!(
            four.join(&bounded(-16, &[7; 4], ByteBound::Prefix))
                .map(|run| run.bound),
            Some(ByteBound::Prefix)
        );
        let walked = bounded(-16, &[1, 2, 3, 4], contract(4))
            .walked(&bounded(-14, &[3, 4], contract(4)))
            .map(|run| run.bound);
        assert_eq!(walked, Some(contract(6)), "the second visit reaches -10");
        let object = frame(-16, 4);
        assert_eq!(
            bounded(-16, &[1, 2], object)
                .walked(&bounded(-15, &[2], object))
                .map(|run| run.bound),
            Some(object)
        );
        let mut run = bounded(-16, &[7; 8], contract(8));
        run.restrict(6, UseBound::Callee);
        assert_eq!(
            (run.bytes.len(), run.bound),
            (
                6,
                ByteBound::Use {
                    bytes: 6,
                    by: UseBound::Callee
                }
            )
        );
        let mut run = bounded(-16, &[7; 4], object);
        run.restrict(6, UseBound::Callee);
        assert_eq!((run.bytes.len(), run.bound), (4, object));
    }

    /// **A use keeps the bytes every visit that knew its pointer saw.** A
    /// visit reading further along the same bytes keeps them from where the
    /// walk began; a visit that did not know the pointer abstains; a visit
    /// that saw other bytes, or none the run wrote, ends the observation for
    /// good.
    #[test]
    fn uses_join_over_the_visits_that_know_their_pointer() {
        let shown = |observed: Observations<MockTarget>| {
            observed
                .into_facts(Completion::Complete)
                .uses()
                .into_iter()
                .map(|(site, bytes)| (site, bytes.into_iter().cloned().collect::<Vec<_>>()))
                .collect::<Vec<_>>()
        };
        let mut walked = Observations::<MockTarget>::default();
        walked.use_site((0, 1), vec![(0, Some(bytes(0, -16, b"hi\0")))]);
        walked.use_site((0, 1), Vec::new());
        walked.use_site((0, 1), vec![(0, Some(bytes(0, -15, b"i\0")))]);
        assert_eq!(shown(walked), [((0, 1), vec![bytes(0, -16, b"hi\0")])]);
        for later in [None, Some(bytes(0, -16, b"xy"))] {
            let mut ended = Observations::<MockTarget>::default();
            ended.use_site((0, 1), vec![(0, Some(bytes(0, -16, b"hi\0")))]);
            ended.use_site((0, 1), vec![(0, later)]);
            ended.use_site((0, 1), vec![(0, Some(bytes(0, -16, b"hi\0")))]);
            assert!(shown(ended).is_empty());
        }
    }

    /// A value is a fact only where every visit agrees.
    #[test]
    fn values_join_over_visits() {
        let mut observed = Observations::<MockTarget>::default();
        let (same, differs) = (SsaVarId::from_index(1), SsaVarId::from_index(2));
        observed.value(same, &Value::Const(ConstValue::I32(7)));
        observed.value(same, &Value::Const(ConstValue::I32(7)));
        observed.value(differs, &Value::Const(ConstValue::I32(1)));
        observed.value(differs, &Value::Const(ConstValue::I32(2)));
        let facts = observed.into_facts(Completion::Complete);
        assert_eq!(facts.constant(same), Some(&ConstValue::I32(7)));
        assert_eq!(facts.value(differs), Some(&Value::Unknown));
        assert_eq!(facts.constants().len(), 1);
    }

    /// Argument bytes keep what every visit saw at one address.
    #[test]
    fn call_bytes_keep_the_common_prefix() {
        let mut observed = Observations::<MockTarget>::default();
        observed.call((0, 1), vec![bytes(0, -32, b"attack"), bytes(1, -8, b"x")]);
        observed.call((0, 1), vec![bytes(0, -32, b"attic"), bytes(1, -16, b"x")]);
        let facts = observed.into_facts(Completion::Complete);
        assert_eq!(facts.call_arguments(0, 1), &[bytes(0, -32, b"att")]);
    }

    fn visit(written: &[i64], clean: bool) -> LoopVisit {
        let mut trace = Trace::new();
        trace.enter_loop(1, 0);
        for offset in written {
            trace.record_write(
                Address::new(Root::Stack, *offset, PointerSize::Bit64),
                1,
                PointerSize::Bit64,
            );
        }
        if !clean {
            trace.taint();
        }
        trace
            .leave_loops(|_| true)
            .pop()
            .unwrap_or_else(|| process::abort())
    }

    /// A clean run's effect is its final bytes, in runs; a run with a byte it
    /// left unknown has none.
    #[test]
    fn a_loop_effect_is_its_final_bytes() {
        let mut memory = Memory::default();
        let at = |offset| Address::new(Root::Stack, offset, PointerSize::Bit64);
        memory.store_bytes(at(-8), b"ab");
        memory.store_bytes(at(-4), b"c");
        let effect = visit(&[-8, -7, -4], true)
            .effect::<MockTarget>((1, 2), &memory, vec![])
            .unwrap_or_else(|| process::abort());
        assert_eq!(
            effect.writes,
            vec![(at(-8), b"ab".to_vec()), (at(-4), b"c".to_vec())]
        );
        assert!(
            visit(&[-8, -2], true)
                .effect::<MockTarget>((1, 2), &memory, vec![])
                .is_none()
        );
        assert!(
            visit(&[-8], false)
                .effect::<MockTarget>((1, 2), &memory, vec![])
                .is_none()
        );
    }

    /// Two visits that differ spoil the loop for good.
    #[test]
    fn differing_visits_spoil_a_loop() {
        let effect = |exit| LoopEffect::<MockTarget> {
            header: 1,
            exit,
            writes: vec![],
            exit_values: vec![],
        };
        let mut observed = Observations::<MockTarget>::default();
        observed.loop_visit(1, Some(effect((1, 2))));
        observed.loop_visit(1, Some(effect((1, 2))));
        observed.loop_visit(4, Some(effect((1, 2))));
        observed.loop_visit(4, Some(effect((1, 3))));
        observed.loop_visit(4, Some(effect((1, 2))));
        let facts = observed.into_facts(Completion::Complete);
        assert!(facts.loop_effect(1).is_some());
        assert!(facts.loop_effect(4).is_none());
    }

    /// A decoded loop's bytes are publication payload; a spoiled loop's are
    /// not, having none to publish.
    #[test]
    fn a_decoded_loop_charges_its_bytes_to_the_payload() {
        let at = |offset| Address::new(Root::Stack, offset, PointerSize::Bit64);
        let effect = |header, writes| LoopEffect::<MockTarget> {
            header,
            exit: (header, header.saturating_add(1)),
            writes,
            exit_values: vec![],
        };
        let mut observed = Observations::<MockTarget>::default();
        observed.loop_visit(
            1,
            Some(effect(
                1,
                vec![(at(-8), b"ab".to_vec()), (at(-4), b"c".to_vec())],
            )),
        );
        observed.loop_visit(4, Some(effect(4, vec![(at(-16), b"spoiled".to_vec())])));
        observed.loop_visit(4, None);
        let facts = observed.into_facts(Completion::Complete);
        assert_eq!(facts.observation_payload_bytes(3, &|| false), Ok(3));
        assert_eq!(
            facts.observation_payload_bytes(2, &|| false),
            Err(ObservationSizeRefusal::OutputLimit)
        );
    }

    /// The image writes are what every return leaves written.
    #[test]
    fn image_writes_hold_on_every_return() {
        let at = |offset| Address::new(Root::Symbol, offset, PointerSize::Bit64);
        let mut first = Memory::default();
        first.set_writer(
            1,
            WriteSite {
                body: 0,
                block: 0,
                instruction: 0,
            },
        );
        first.store_bytes(at(0x3000), b"key!");
        let mut second = Memory::default();
        second.set_writer(
            1,
            WriteSite {
                body: 0,
                block: 0,
                instruction: 0,
            },
        );
        second.store_bytes(at(0x3000), b"keys");
        let mut observed = Observations::<MockTarget>::default();
        observed.exit(&first, Some(Value::Const(ConstValue::I32(1))), 0);
        observed.exit(&second, Some(Value::Const(ConstValue::I32(1))), 0);
        let facts = observed.into_facts(Completion::Complete);
        assert_eq!(facts.image_writes(), vec![(0x3000, b"key".to_vec())]);
        assert_eq!(facts.returned(), Some(&Value::Const(ConstValue::I32(1))));
    }

    /// An execution that did not finish establishes nothing.
    #[test]
    fn an_incomplete_execution_withholds_its_facts() {
        let mut observed = Observations::<MockTarget>::default();
        observed.value(SsaVarId::from_index(1), &Value::Const(ConstValue::I32(7)));
        let facts = observed.into_facts(Completion::Stuck { block: 3 });
        assert!(!facts.is_complete());
        assert!(facts.value(SsaVarId::from_index(1)).is_none());
    }

    /// A failed visit or differing image access permanently blocks one definition's proof.
    #[test]
    fn replacement_evidence_intersects_all_visits_and_is_withheld_on_failure() {
        let id = SsaVarId::from_index(0);
        let other = SsaVarId::from_index(1);
        let mut observed = Observations::<MockTarget>::default();
        assert!(observed.needs_replacement_evidence(id));
        observed.replacement(id, Some(ReplacementProof::Completed));
        assert!(observed.needs_replacement_evidence(id));
        observed.replacement(id, Some(ReplacementProof::Completed));
        observed.replacement(
            other,
            Some(ReplacementProof::ReadOnlyImage {
                address: 0x1000,
                len: 1,
            }),
        );
        observed.replacement(
            other,
            Some(ReplacementProof::ReadOnlyImage {
                address: 0x1001,
                len: 1,
            }),
        );
        assert!(!observed.needs_replacement_evidence(other));
        let complete = observed.clone().into_facts(Completion::Complete);
        assert_eq!(
            complete.replacement_proof(id),
            Some(ReplacementProof::Completed)
        );
        assert_eq!(complete.replacement_proof(other), None);
        let refused = observed.clone().into_facts(Completion::Stuck { block: 0 });
        assert_eq!(refused.replacement_proof(id), None);
        observed.replacement(id, None);
        observed.replacement(id, Some(ReplacementProof::Completed));
        assert!(!observed.needs_replacement_evidence(id));
        assert_eq!(
            observed
                .into_facts(Completion::Complete)
                .replacement_proof(id),
            None
        );
    }

    /// Both exit and loop artifacts use numeric order and split the address wrap.
    #[test]
    fn image_artifacts_preserve_linear_ranges_at_every_target_width() {
        for width in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            let top = u64::MAX >> (64 - width.bits().min(64));
            let mut memory = Memory::default();
            memory.set_writer(
                1,
                WriteSite {
                    body: 0,
                    block: 0,
                    instruction: 0,
                },
            );
            let mut trace = Trace::new();
            trace.enter_loop(1, 0);
            let mut expected = vec![(0, vec![1, 2])];
            let mut stored = vec![(0u64, 1u8), (1, 2), (top, 5)];
            if width.bits() >= 64 {
                let middle = i64::MAX.cast_unsigned();
                stored.extend([(middle, 3), (middle + 1, 4)]);
                expected.push((middle, vec![3, 4]));
            }
            expected.push((top, vec![5]));
            for (location, byte) in stored {
                let address = Address::new(Root::Symbol, location.cast_signed(), width);
                memory.store_bytes(address, &[byte]);
                trace.record_write(address, 1, width);
            }
            let mut observed = Observations::<MockTarget>::default();
            observed.exit(&memory, None, 0);
            let facts = observed.into_facts(Completion::Complete);
            assert_eq!(facts.image_writes(), expected);
            let visit = trace
                .leave_loops(|_| true)
                .pop()
                .unwrap_or_else(|| process::abort());
            let effect = visit
                .effect::<MockTarget>((1, 2), &memory, vec![])
                .unwrap_or_else(|| process::abort());
            assert_eq!(
                effect.writes,
                expected
                    .into_iter()
                    .map(|(at, bytes)| (Address::new(Root::Symbol, at.cast_signed(), width), bytes))
                    .collect::<Vec<_>>()
            );
        }
    }

    /// Signed stack displacements can cross zero, but a signed-domain wrap splits runs.
    #[test]
    fn stack_artifacts_keep_signed_ranges_and_split_their_boundary() {
        let width = PointerSize::Bit8;
        let mut memory = Memory::default();
        let mut trace = Trace::new();
        trace.enter_loop(1, 0);
        for (offset, byte) in [(127, 4), (-128, 1), (-1, 2), (0, 3)] {
            let address = Address::new(Root::Stack, offset, width);
            memory.store_bytes(address, &[byte]);
            trace.record_write(address, 1, width);
        }
        let visit = trace
            .leave_loops(|_| true)
            .pop()
            .unwrap_or_else(|| process::abort());
        let effect = visit
            .effect::<MockTarget>((1, 2), &memory, vec![])
            .unwrap_or_else(|| process::abort());
        assert_eq!(
            effect.writes,
            [(-128, vec![1]), (-1, vec![2, 3]), (127, vec![4])]
                .into_iter()
                .map(|(offset, bytes)| (Address::new(Root::Stack, offset, width), bytes))
                .collect::<Vec<_>>()
        );
    }

    /// Projection moves a retained constant out without allocating another copy of its payload.
    #[test]
    fn output_projection_moves_constant_buffers() {
        let id = SsaVarId::from_index(0);
        let text: Box<str> = "decoded".repeat(16_384).into();
        let text_storage = text.as_ptr();
        let mut observed = Observations::<MockTarget>::default();
        observed
            .values
            .insert(id, Value::Const(ConstValue::DecryptedString(text)));
        let mut facts = observed.into_facts(Completion::Complete);
        let ConstValue::DecryptedString(text) = facts.take_constant(id).unwrap() else {
            process::abort()
        };
        assert!(facts.is_complete());
        assert_eq!(text.as_ptr(), text_storage);
        assert_eq!(text.as_ref(), "decoded".repeat(16_384));
        assert!(facts.constants().is_empty());
    }

    /// Ordered extraction preserves sparse ids, instruction sites and original payload storage.
    #[test]
    fn owned_output_collections_preserve_order_and_buffer_identity() {
        let mut observed = Observations::<MockTarget>::default();
        let mut expected = Vec::new();
        for id in [1_000_000usize, 7, 0, 31, 9] {
            let text: Box<str> = format!("decoded-{id}").into();
            expected.push((id, text.as_ptr()));
            observed.values.insert(
                SsaVarId::from_index(id),
                Value::Const(ConstValue::DecryptedString(text)),
            );
        }
        observed
            .values
            .insert(SsaVarId::from_index(3), Value::Unknown);
        let mut call_storage = Vec::new();
        for site in [(5usize, 1usize), (0, 3), (0, 0)] {
            let bytes = vec![site.1 as u8; 256];
            call_storage.push((site, bytes.as_ptr()));
            observed.calls.insert(
                site,
                vec![ArgumentBytes {
                    argument: 0,
                    address: Address::new(Root::Symbol, 0x1000, PointerSize::Bit64),
                    bytes,
                    bound: ByteBound::Prefix,
                    allocated_at: None,
                    written: WrittenRanges::default(),
                }],
            );
        }
        expected.sort_by_key(|(id, _)| *id);
        call_storage.sort_by_key(|(site, _)| *site);
        let mut facts = observed.into_facts(Completion::Complete);
        let constants = facts.take_constants();
        let calls = facts.take_calls();
        assert!(facts.is_complete());
        assert_eq!(constants.len(), expected.len());
        for ((id, constant), (expected_id, pointer)) in constants.iter().zip(expected) {
            assert_eq!(id.index(), expected_id);
            let ConstValue::DecryptedString(text) = constant else {
                process::abort()
            };
            assert_eq!(text.as_ptr(), pointer);
            assert_eq!(text.as_ref(), format!("decoded-{expected_id}"));
        }
        for (call, (expected_site, pointer)) in calls.iter().zip(call_storage) {
            assert_eq!((call.block, call.instruction), expected_site);
            assert_eq!(call.arguments[0].bytes.as_ptr(), pointer);
        }
        assert!(facts.constants().is_empty());
        assert!(facts.calls().is_empty());
    }

    /// Image output retains unsigned order, gaps and target-coordinate boundaries.
    #[test]
    fn bounded_image_writes_keep_the_exact_byte_runs() {
        let mut memory = Memory::default();
        memory.set_writer(
            1,
            WriteSite {
                body: 0,
                block: 0,
                instruction: 0,
            },
        );
        for (address, bytes) in [
            (0u64, b"zero".as_slice()),
            (0x1000, b"ab".as_slice()),
            (0x1003, b"gap".as_slice()),
            (i64::MAX as u64, b"sign".as_slice()),
            (u64::MAX - 1, b"en".as_slice()),
        ] {
            memory.store_bytes(
                Address::new(Root::Symbol, address.cast_signed(), PointerSize::Bit64),
                bytes,
            );
        }
        let observed = Observations::<MockTarget> {
            exit_memory: Some(memory.image_contents(0)),
            ..Observations::default()
        };
        let facts = observed.into_facts(Completion::Complete);
        let expected = facts.image_writes();
        let runs = facts.image_writes_within(usize::MAX).unwrap();
        assert!(facts.is_complete());
        let actual: Vec<_> = runs
            .into_iter()
            .map(|run| (run.address, run.bytes))
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(actual.first(), Some(&(0, b"zero".to_vec())));
        assert_eq!(actual.last(), Some(&(u64::MAX - 1, b"en".to_vec())));
        assert!(actual.contains(&(i64::MAX as u64, b"sign".to_vec())));
    }

    /// A retained-output limit is checked before allocating descriptors or byte buffers.
    #[test]
    fn cooperative_image_output_checks_extent_before_allocation() {
        let mut memory = Memory::default();
        memory.set_writer(
            1,
            WriteSite {
                body: 0,
                block: 0,
                instruction: 0,
            },
        );
        memory.store_bytes(
            Address::new(Root::Symbol, 0x1000, PointerSize::Bit64),
            b"decoded",
        );
        let observed = Observations::<MockTarget> {
            exit_memory: Some(memory.image_contents(0)),
            ..Observations::default()
        };
        let facts = observed.into_facts(Completion::Complete);
        assert!(facts.image_writes_within(3).is_none());
        assert!(facts.is_complete());
    }

    /// **A save-bounded frame region binds only from above.** The region from
    /// 32 bytes below entry up to the save 8 below holds any access that ends
    /// at the save, including one that begins below the region, as a
    /// top-pointer walk with a negative offset does; an access reaching into
    /// the save is outside it. An object stated whole binds both ends.
    #[test]
    fn a_save_bounded_region_binds_only_its_end() {
        let at = |offset| Address::new(Root::Stack, offset, PointerSize::Bit64);
        let region = StorageObject {
            start: -0x20,
            bytes: 0x18,
            kind: ObjectKind::Frame,
            source: ExtentSource::SaveArea,
        };
        assert!(region.holds(at(-0x20), 0x18));
        assert!(region.holds(at(-0x38), 8));
        assert!(!region.holds(at(-0x10), 9));
        let whole = StorageObject {
            source: ExtentSource::Recovered,
            ..region
        };
        assert!(whole.holds(at(-0x20), 0x18));
        assert!(!whole.holds(at(-0x38), 8));
        assert!(!whole.holds(at(-0x10), 9));
    }
}
