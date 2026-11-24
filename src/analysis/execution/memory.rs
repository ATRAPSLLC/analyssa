//! The memory a static execution knows, one byte at a time.
//!
//! A byte is keyed by its [`Root`] and target-width offset. Explicit cells
//! override lazy object initialization. Unknown writes mask initialized
//! ranges, so removing a cell cannot reveal a stale initial byte. An unwritten
//! [`Root::Symbol`] byte reads through only where the image oracle promises
//! lifetime immutability; writable image contents are not entry assumptions.
//!
//! A stored address is not a number the execution can write out: an address
//! under the stack or under an opaque root has no bytes. It is kept as pieces
//! instead, and a load that reads back exactly those pieces in order reads the
//! address again. Anything else that reads a piece reads an unknown byte.
//!
//! # Aliasing
//!
//! A store writes its own bytes and forgets every byte under another root the
//! same memory may be named by:
//!
//! - The image, the stack and each heap allocation never overlap one another.
//! - The stack below the outermost activation's entry pointer is private to
//!   the execution until a stack address escapes. At and above that pointer it
//!   is the caller's memory, which anything else may name.
//! - An integer address the host cannot place, and an opaque value, may name
//!   any memory but the private stack. A [`Root::StackDerived`] identity may
//!   also reach private stack storage; it retains related offsets after an
//!   operation such as alignment loses the original stack coordinate.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    hash::{Hash, Hasher},
    ops::{
        Bound::{Excluded, Included, Unbounded},
        Range,
    },
    ptr,
    sync::Arc,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    analysis::{
        execution::{
            InstructionSources, MemoryWriteGuard, ObjectExtents, ObservationSizeRefusal, Value,
            facts::{ExtentSource, StorageObject, WrittenRanges},
            findings::{
                Decisions, Ending, Finding, FindingKind, Findings, FoundObject, ReleaseFault,
            },
            memory::{
                cells::Cells,
                formed::Formed,
                origins::ImageOrigins,
                unwritten::Unwritten,
                writers::{CurrentWriter, ImageWriters, Writer},
            },
            value::{Address, Root},
            writeguard::check_write,
        },
        image::{ImageData, Mutability},
    },
    pointer::PointerSize,
    target::Target,
};

mod cells;
mod formed;
mod origins;
mod unwritten;
mod writers;

pub use self::{
    formed::{Through, Transfer},
    writers::WriteSite,
};

/// One contiguous image-coordinate byte range a static execution wrote.
#[derive(Debug, PartialEq, Eq)]
pub struct ImageWrite {
    /// Unsigned image coordinate of the first byte.
    pub address: u64,
    /// Exact known bytes, stopping before gaps and coordinate wrap.
    pub bytes: Vec<u8>,
    /// Every instruction that wrote bytes of the run, sorted, all of one
    /// activation; empty when paths that agree on the bytes wrote them at
    /// different instructions.
    pub sites: Vec<WriteSite>,
}
/// An exact prefix of an indefinitely repeated, nonempty byte pattern.
struct PatternBytes<'a> {
    /// The borrowed element in target byte order.
    pattern: &'a [u8],
    /// The next position within the pattern.
    offset: usize,
    /// Bytes still to emit, including a possible partial final element.
    remaining: usize,
}

impl Iterator for PatternBytes<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let byte = *self.pattern.get(self.offset)?;
        let next = self.offset.saturating_add(1);
        self.offset = if next == self.pattern.len() { 0 } else { next };
        self.remaining = self.remaining.saturating_sub(1);
        Some(byte)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for PatternBytes<'_> {}

/// The memory contract needed to continue an execution proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MemoryRefusal {
    /// A physical-state binding is malformed or lacks a compatible source contract.
    MachineState,

    /// The address span exceeds the execution domain's coordinate range.
    AddressRange,
    /// A streamed block has no nonzero element width or a partial reverse element.
    BlockLayout,
    /// A repeated block operation (`rep scas`, `rep cmps`, `rep lods`) ran
    /// with a count of zero where it states its count is never zero, and so
    /// names no value its empty run leaves.
    EmptyBlockInput,
    /// An atomic access lacks an exclusive private span or an exact layout contract.
    AtomicContract,
    /// A write the program may not make intersects image storage the oracle
    /// promises remains immutable: a model's write, which the routine may
    /// hand to the kernel and fail rather than fault, or one that may land
    /// on writable bytes instead (a lane an unknown mask may leave alone,
    /// one element of a span). A store the program makes there faults its
    /// path instead, found as [`FindingKind::ReadOnlyWrite`].
    ReadOnlyImage,
    /// Instruction-source conditions could not be preserved across a write.
    InstructionSource,
    /// The configured write policy requires a footprint that could not be established.
    UnresolvedWrite,
    /// The root's declared entry contract cannot be stated over its stack.
    StackAlignment,
    /// A physical return target or outgoing stack coordinate cannot satisfy its frame's obligation.
    ReturnControl,
}

/// The ordering and indivisible read unit of a modeled block copy.
#[derive(Debug, Clone, Copy)]
pub enum CopyOrder {
    /// Reads the whole source before the first destination write.
    Snapshot,
    /// Reads one complete element before writing it, then advances in the stated direction.
    Stream {
        /// Byte width of each element.
        element_bytes: usize,
        /// Whether successive elements begin at lower addresses.
        reverse: bool,
    },
}

impl CopyOrder {
    /// Returns the lowest address and full linear extent without target-coordinate wrap.
    pub fn span(self, first: Address, len: usize) -> Result<Address, MemoryRefusal> {
        let Self::Stream {
            element_bytes,
            reverse,
        } = self
        else {
            return Ok(first);
        };
        if element_bytes == 0 || !len.is_multiple_of(element_bytes) {
            return Err(MemoryRefusal::BlockLayout);
        }
        if !reverse || len == 0 {
            return Ok(first);
        }
        let delta = len
            .checked_sub(element_bytes)
            .and_then(|delta| i64::try_from(delta).ok())
            .ok_or(MemoryRefusal::AddressRange)?;
        let start = first
            .moved(delta.saturating_neg(), first.pointer_size())
            .filter(|start| first.follows(*start, delta as usize))
            .ok_or(MemoryRefusal::AddressRange)?;
        Ok(start)
    }
}

/// One byte of memory the execution holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Cell {
    /// A known byte.
    Byte(u8),
    /// Unknown bits derived from a stack address on at least one joined path.
    StackBits,
    /// Byte `index` of an address stored `width` bytes wide.
    Piece {
        /// The address stored.
        address: Address,
        /// Which of its bytes this is, from the lowest address up.
        index: u8,
        /// How many bytes it was stored in.
        width: u8,
    },
}

impl Cell {
    /// Returns whether a byte may retain private stack-pointer bits.
    const fn may_reference_stack(self) -> bool {
        matches!(self, Self::StackBits)
            || matches!(self, Self::Piece { address, .. } if address.root().may_reference_stack())
    }
}

/// What a load reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Loaded {
    /// Known bytes, lowest address first.
    Bytes(Vec<u8>),
    /// An address stored whole at exactly the bytes read.
    Address(Address),
    /// Unknown bits retaining stack-pointer provenance.
    StackDerived,
    /// Anything else.
    Unknown,
}

/// What a bounded string scan found, and how many bytes it inspected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringScan<S> {
    /// What came before the first NUL, or `None` where unknown data or the
    /// limit came first.
    pub string: Option<S>,
    /// The bytes the scan inspected, the terminator or the unknown byte that
    /// stopped it included.
    pub inspected: usize,
}

/// The bytes an allocation holds: those it holds on every path, and the most
/// it may hold, which an access is judged against, where its allocator states
/// it.
///
/// `malloc(8)` holds exactly eight bytes; `VirtualAlloc` of 100 bytes holds at
/// least a page, and exactly one where the platform's page size is fixed; a
/// size the call does not state holds no byte known to belong to it and has
/// no extent to be past.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockSize {
    /// The bytes it holds on every path.
    pub guaranteed: u64,
    /// The most bytes it may hold, where they are known.
    pub extent: Option<u64>,
}

impl BlockSize {
    /// A size no allocator states.
    pub const UNSTATED: Self = Self {
        guaranteed: 0,
        extent: None,
    };

    /// Exactly `bytes`.
    #[must_use]
    pub const fn exact(bytes: u64) -> Self {
        Self {
            guaranteed: bytes,
            extent: Some(bytes),
        }
    }

    /// At least `bytes`, with no extent known.
    #[must_use]
    pub const fn at_least(bytes: u64) -> Self {
        Self {
            guaranteed: bytes,
            extent: None,
        }
    }
}

impl From<Option<u64>> for BlockSize {
    /// Exactly the bytes stated, or [`BlockSize::UNSTATED`].
    fn from(bytes: Option<u64>) -> Self {
        bytes.map_or(Self::UNSTATED, Self::exact)
    }
}

/// The allocator a block belongs to: its family, and the heap of that family
/// it was made in where the routine names one (`HeapAlloc`'s handle, the
/// block `HeapCreate` made it). A deallocator ends only a block of its own
/// family, and of its own heap where both name one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Allocator {
    /// The family, where the routine states one.
    pub family: Option<AllocatorFamily>,
    /// The heap: the root of the block its handle names.
    pub heap: Option<Root>,
}

impl Allocator {
    /// Whether a deallocator of this allocator may end a block of `family`
    /// made in `heap`: every part each states agrees.
    #[must_use]
    pub fn admits(self, family: Option<AllocatorFamily>, heap: Option<Root>) -> bool {
        self.family.is_none_or(|wanted| family == Some(wanted))
            && self
                .heap
                .zip(heap)
                .is_none_or(|(wanted, made)| wanted == made)
    }
}

impl From<Option<AllocatorFamily>> for Allocator {
    /// The family alone, of no named heap.
    fn from(family: Option<AllocatorFamily>) -> Self {
        Self { family, heap: None }
    }
}

/// The allocator an object belongs to, by the identity its host gives each
/// family: `malloc`'s, a Windows heap's, C++'s `new` and `new[]`.
///
/// A deallocator of another family does not end an object's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AllocatorFamily(pub u16);

/// How an object's lifetime stands over the paths joined into a state, and
/// what ended it.
///
/// A site is the call of the outermost activation whose execution ended the
/// object, as `(block, instruction)`. An access in every state but
/// [`Self::Live`] is a finding ([`FindingKind::Ended`]); the states apart from it say
/// whether a deallocation the execution stated ended the object on every
/// path, on some, or whether an effect it cannot follow may have.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Lifetime {
    /// Live on every joined path.
    Live,
    /// Ended on every joined path by a stated deallocation at one of `sites`.
    Ended {
        /// The deallocations that ended it.
        sites: BTreeSet<(usize, usize)>,
    },
    /// Ended on some joined paths by a stated deallocation at one of
    /// `sites`, and live on the others.
    EndedOnSome {
        /// The deallocations that ended it where it ended.
        sites: BTreeSet<(usize, usize)>,
    },
    /// Possibly ended by an effect the execution cannot follow (a call
    /// through an unknown pointer, a deallocation of an unknown address), on
    /// a path where it was not already ended; `sites` are the stated
    /// deallocations that ended it on other paths.
    PossiblyEnded {
        /// The stated deallocations that ended it on some path.
        sites: BTreeSet<(usize, usize)>,
    },
}

impl Lifetime {
    /// Whether the object is live on every path.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        matches!(self, Self::Live)
    }

    /// The stated deallocations that ended the object on some path.
    #[must_use]
    pub fn sites(&self) -> Option<&BTreeSet<(usize, usize)>> {
        match self {
            Self::Live => None,
            Self::Ended { sites } | Self::EndedOnSome { sites } | Self::PossiblyEnded { sites } => {
                Some(sites)
            }
        }
    }

    /// The lifetime over both paths: the weaker of the two states, with the
    /// sites of both. Ended on one path and live on the other is ended on
    /// some; an effect that may have ended it on either path may have ended
    /// it on the join.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        let sites = || {
            let mut sites = self.sites().cloned().unwrap_or_default();
            sites.extend(other.sites().into_iter().flatten().copied());
            sites
        };
        match (self, other) {
            (Self::Live, Self::Live) => Self::Live,
            (Self::PossiblyEnded { .. }, _) | (_, Self::PossiblyEnded { .. }) => {
                Self::PossiblyEnded { sites: sites() }
            }
            (Self::Ended { .. }, Self::Ended { .. }) => Self::Ended { sites: sites() },
            _ => Self::EndedOnSome { sites: sites() },
        }
    }

    /// The lifetime once an effect the execution cannot follow may have
    /// ended the object: one already ended on every path stays ended, and
    /// one live on some path may have ended there.
    #[must_use]
    fn possibly_ended(&self) -> Self {
        match self {
            Self::Ended { .. } | Self::PossiblyEnded { .. } => self.clone(),
            Self::Live => Self::PossiblyEnded {
                sites: BTreeSet::new(),
            },
            Self::EndedOnSome { sites } => Self::PossiblyEnded {
                sites: sites.clone(),
            },
        }
    }
}

/// What a deallocation of an address does, classified by what the address
/// names.
///
/// [`Self::Released`] ends a live object and [`Self::Null`] does nothing, as
/// the C library states them; [`Self::Ended`], [`Self::Interior`],
/// [`Self::NotHeap`] and [`Self::OtherFamily`] are deallocations no allocator
/// defines; [`Self::Unknown`] names nothing the execution can follow.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FreeOutcome {
    /// Ended the live object the address starts.
    Released,
    /// A null pointer: no effect.
    Null,
    /// An object whose lifetime is not live on every path: ended, ended on
    /// some, or possibly ended, with the earlier deallocations' sites.
    Ended {
        /// How the object's lifetime stood before this deallocation.
        lifetime: Lifetime,
    },
    /// An address inside a heap object, `offset` bytes past its start.
    Interior {
        /// The distance from the object's first byte.
        offset: i64,
    },
    /// An address of no heap object: the stack or the image.
    NotHeap,
    /// A live object another allocator family made, or one whose family the
    /// joined paths do not agree on (`made_by` is `None`).
    OtherFamily {
        /// The family that made the object, where every path agrees on one.
        made_by: Option<AllocatorFamily>,
    },
    /// An address the execution cannot name.
    Unknown,
}

impl FreeOutcome {
    /// What was wrong with the release, where it was one no allocator
    /// defines; `None` for a release, a null and an address the execution
    /// cannot name.
    #[must_use]
    pub const fn fault(&self) -> Option<ReleaseFault> {
        match self {
            Self::Released | Self::Null | Self::Unknown => None,
            Self::Ended { .. } => Some(ReleaseFault::Repeated),
            Self::Interior { .. } => Some(ReleaseFault::Interior),
            Self::NotHeap => Some(ReleaseFault::NotHeap),
            Self::OtherFamily { .. } => Some(ReleaseFault::OtherFamily),
        }
    }
}

/// The accessible prefix, extent, lazy initial contents and lifetime of one
/// allocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Object {
    /// Bytes guaranteed to belong to the allocation on every joined path.
    size: u64,
    /// The most bytes the allocation holds on any joined path, an access past
    /// which is past it on every path; `None` where a path's allocator did
    /// not state its size.
    extent: Option<u64>,
    /// The allocator that made it, where every joined path agrees on one.
    family: Option<AllocatorFamily>,
    /// The heap it was made in, where its routine names one and every
    /// joined path agrees on it.
    heap: Option<Root>,
    /// The call of the outermost activation whose execution allocated it, as
    /// `(block, instruction)`, where every joined path agrees on one.
    site: Option<(usize, usize)>,
    /// Whether it is live, and what ended it where it is not.
    lifetime: Lifetime,
    /// Initial byte value, when a zeroing allocator established one.
    initial: Option<u8>,
    /// Half-open ranges that no longer have known initial contents.
    unknown: BTreeMap<u64, u64>,
}

impl Object {
    /// A live allocation of `size`, made by allocator `family`, each byte
    /// `initial` where it states one.
    fn fresh(size: BlockSize, initial: Option<u8>, allocator: Allocator) -> Self {
        Self {
            size: size.guaranteed,
            extent: size.extent,
            family: allocator.family,
            heap: allocator.heap,
            site: None,
            lifetime: Lifetime::Live,
            initial,
            unknown: BTreeMap::new(),
        }
    }

    /// Returns an unwritten byte unless an unknown write reached it.
    fn byte(&self, offset: u64) -> Option<u8> {
        if offset >= self.size
            || self
                .unknown
                .range(..=offset)
                .next_back()
                .is_some_and(|(_, end)| offset < *end)
        {
            None
        } else {
            self.initial
        }
    }

    /// Marks an unknown interval, merging overlapping and adjacent ranges.
    fn forget(&mut self, mut start: u64, mut end: u64) {
        end = end.min(self.size);
        if self.initial.is_none() || start >= end {
            return;
        }
        if let Some((&prior, &prior_end)) = self.unknown.range(..=start).next_back()
            && prior_end >= start
        {
            start = prior;
            end = end.max(prior_end);
            self.unknown.remove(&prior);
        }
        while let Some((&next, &next_end)) = self.unknown.range(start..=end).next() {
            end = end.max(next_end);
            self.unknown.remove(&next);
        }

        self.unknown.insert(start, end);
    }

    /// Returns the allocation extent and initial bytes common to both paths.
    fn join(&self, other: &Self) -> Self {
        let mut joined = Self {
            size: self.size.min(other.size),
            extent: self
                .extent
                .zip(other.extent)
                .map(|(left, right)| left.max(right)),
            family: self.family.filter(|family| other.family == Some(*family)),
            heap: self.heap.filter(|heap| other.heap == Some(*heap)),
            site: self.site.filter(|site| other.site == Some(*site)),
            lifetime: self.lifetime.join(&other.lifetime),
            initial: self
                .initial
                .filter(|initial| other.initial == Some(*initial)),
            unknown: BTreeMap::new(),
        };
        for (start, end) in self.unknown.iter().chain(other.unknown.iter()) {
            joined.forget(*start, *end);
        }
        joined
    }
}

/// The image bytes one activation's own execution wrote, retained at an
/// observation independently of execution storage, with the write that
/// produced each.
#[derive(Debug, Clone)]
pub struct ImageContents {
    /// Shared image pages holding only the activation's own bytes; private
    /// stack and heap pages are excluded.
    cells: Cells,
    /// The writer of every retained byte.
    writers: ImageWriters,
}

impl ImageContents {
    /// Counts visible known image bytes without assembling runs or copying payloads.
    pub fn payload_bytes_within(
        &self,
        maximum_bytes: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<usize, ObservationSizeRefusal> {
        let nonnegative = self
            .cells
            .range((Root::Symbol, 0)..=(Root::Symbol, i64::MAX));
        let negative = self
            .cells
            .range((Root::Symbol, i64::MIN)..=(Root::Symbol, -1));
        let mut bytes = 0usize;
        for (_, cell) in nonnegative.chain(negative) {
            if cancelled() {
                return Err(ObservationSizeRefusal::Cancelled);
            }
            if matches!(cell, Cell::Byte(_)) {
                bytes = bytes
                    .checked_add(1)
                    .filter(|bytes| *bytes <= maximum_bytes)
                    .ok_or(ObservationSizeRefusal::OutputLimit)?;
            }
        }
        Ok(bytes)
    }

    /// Keeps the bytes `other` also holds and that activation `since` or a
    /// later one wrote there too, joining their writers; later memory growth
    /// cannot enlarge this observation.
    pub fn intersect(&mut self, other: &Memory, since: u32) {
        self.writers = self.writers.join(&other.writers.owned_since(since));
        let mut removed = Vec::new();
        for page in self.cells.page_keys() {
            let shared = self.cells.shares_page(&other.cells, page);
            for key in self.cells.keys_on_page(page) {
                if (!shared && self.cells.get(&key) != other.cells.get(&key))
                    || self.writers.writer(key.1).is_none()
                {
                    removed.push(key);
                }
            }
        }
        for key in removed {
            self.cells.remove(&key);
        }
    }

    /// Returns known runs in unsigned image order under the caller's output bound.
    pub fn runs(&self, maximum_bytes: usize) -> Option<Vec<ImageWrite>> {
        let nonnegative = self
            .cells
            .range((Root::Symbol, 0)..=(Root::Symbol, i64::MAX));
        let negative = self
            .cells
            .range((Root::Symbol, i64::MIN)..=(Root::Symbol, -1));
        let mut runs: Vec<ImageWrite> = Vec::new();
        let mut bytes = 0usize;
        let mut next = None;
        let mut run_owner = None;
        for ((_, at), cell) in nonnegative.chain(negative) {
            let Cell::Byte(byte) = cell else {
                next = None;
                continue;
            };
            bytes = bytes.saturating_add(1);
            if bytes > maximum_bytes {
                return None;
            }
            // A run is what one activation wrote: where another invocation's
            // bytes begin, so does another run, and two calls' writes left
            // side by side are never one value. One activation's several
            // stores (an unrolled decoder's) stay one run.
            let writer = self.writers.writer(at);
            let owner = writer.map(|writer| (writer.activation, writer.site.is_some()));
            match runs.last_mut() {
                Some(run) if next == Some(at) && run_owner == owner => {
                    run.bytes.push(*byte);
                    if let Some(site) = writer.and_then(|writer| writer.site)
                        && let Err(index) = run.sites.binary_search(&site)
                    {
                        run.sites.insert(index, site);
                    }
                }
                _ => {
                    run_owner = owner;
                    runs.push(ImageWrite {
                        address: at.cast_unsigned(),
                        bytes: vec![*byte],
                        sites: writer.and_then(|writer| writer.site).into_iter().collect(),
                    });
                }
            }
            next = at.cast_unsigned().checked_add(1).map(u64::cast_signed);
        }
        Some(runs)
    }
}

/// The bytes a static execution holds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Memory {
    /// Every byte the execution has an answer for.
    cells: Cells,
    /// Known instruction-source writes and entries belonging to this path.
    sources: InstructionSources,
    /// Initial image bindings invalidated by exact writes or unresolved aliases.
    image_origins: ImageOrigins,
    /// The write that produced each image byte.
    writers: ImageWriters,
    /// The write the running instruction performs.
    current: CurrentWriter,
    /// Live heap objects; absence prevents dereferencing a stale allocation identity.
    objects: BTreeMap<Root, Object>,
    /// Attested numeric ranges in this root's physical address space.
    disjoint_numeric: Option<Arc<[Range<u64>]>>,
    /// A memory operation whose contract could not be established.
    refusal: Option<MemoryRefusal>,
    /// The write that reached storage that may not be written, which the
    /// machine faults: its first byte and how many it writes. The path goes
    /// no further than that write.
    fault: Option<(Address, u64)>,
    /// Whether an address into the stack has escaped where the execution
    /// cannot follow it.
    stack_escaped: bool,
    /// The host's objects outside the heap for this root.
    extents: Option<Extents>,
    /// Where the run records what this memory's accesses find wrong with
    /// the program; `None` outside a run.
    findings: Option<Findings>,
    /// The bits no write has reached since their object began.
    unwritten: Unwritten,
    /// The objects the pointers stored here were formed into.
    formed: Formed,
    /// The objects the running access goes through.
    through: Through,
    /// The decisions of the path this memory runs on.
    decisions: Decisions,
}

/// The host's table of objects outside the heap for one root
/// ([`ObjectExtents`]), shared by every path of the run.
#[derive(Clone)]
struct Extents {
    /// The host's table.
    table: Arc<dyn ObjectExtents>,
}

impl fmt::Debug for Extents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Extents")
            .field("table", &self.table)
            .finish_non_exhaustive()
    }
}

/// One table is one table: every path of a run shares the run's.
impl PartialEq for Extents {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.table, &other.table)
    }
}

impl Eq for Extents {}

impl Hash for Extents {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(Arc::as_ptr(&self.table).cast::<u8>(), state);
    }
}

impl Memory {
    /// Retains the valid numeric disjointness declarations for this root.
    pub fn set_disjoint_numeric(&mut self, ranges: Vec<Range<u64>>, width: PointerSize) {
        let mut ranges = ranges;
        ranges.retain(|range| {
            range.start < range.end
                && width.mask_unsigned(range.start) == range.start
                && width.mask_unsigned(range.end.saturating_sub(1)) == range.end.saturating_sub(1)
        });
        ranges.sort_unstable_by_key(|range| (range.start, range.end));
        self.disjoint_numeric = (!ranges.is_empty()).then(|| Arc::from(ranges));
    }

    /// Checks that the whole numeric write lies in one independently mapped range.
    fn disjoint_numeric_span(&self, address: Address, len: usize) -> bool {
        if address.root() != Root::Number || address.linear_prefix_len(len) != len {
            return false;
        }
        let first = address.offset().cast_unsigned();
        let Some(end) = u64::try_from(len)
            .ok()
            .and_then(|len| first.checked_add(len))
        else {
            return false;
        };
        self.disjoint_numeric.as_ref().is_some_and(|ranges| {
            ranges
                .iter()
                .any(|range| range.start <= first && end <= range.end)
        })
    }

    /// Forgets numeric cells that lack a disjoint physical mapping.
    fn forget_unmapped_numeric(&mut self) {
        let Some(ranges) = &self.disjoint_numeric else {
            self.erase_interval(Root::Number, i64::MIN, i64::MAX);
            return;
        };
        let mut start = Included((Root::Number, i64::MIN));
        let end = Included((Root::Number, i64::MAX));
        loop {
            let key = self.cells.range((start, end)).next().map(|(key, _)| key);
            let Some(key) = key else { break };
            start = Excluded(key);
            if !ranges
                .iter()
                .any(|range| range.contains(&key.1.cast_unsigned()))
            {
                self.cells.remove(&key);
            }
        }
    }

    /// Returns whether paths can merge without mixing distinct instruction-source histories.
    pub fn same_instruction_sources(&self, other: &Self) -> bool {
        self.sources == other.sources
    }

    /// Borrows instruction events without projecting them into ordinary data cells.
    pub const fn instruction_sources(&self) -> &InstructionSources {
        &self.sources
    }

    /// Borrows the path's source history for body and write admission.
    pub fn instruction_sources_mut(&mut self) -> &mut InstructionSources {
        &mut self.sources
    }

    /// Returns whether a complete image cell retains its declared initial binding.
    pub fn initial_image_binding(&self, address: Address, len: usize) -> bool {
        self.refusal.is_none() && self.image_origins.unchanged(address, len)
    }

    /// Returns a memory-contract refusal, when one occurred.
    pub const fn refusal(&self) -> Option<MemoryRefusal> {
        self.refusal
    }

    /// Withholds this execution's facts after an unsupported memory access.
    pub fn refuse(&mut self, reason: MemoryRefusal) {
        self.refusal.get_or_insert(reason);
    }

    /// The write that reached storage that may not be written
    /// ([`Self::deny_write`]), its first byte and how many it writes: the
    /// machine faults it, and the path goes no further. `None` where no
    /// write faulted.
    pub const fn fault(&self) -> Option<(Address, u64)> {
        self.fault
    }

    /// Denies the write of `len` bytes at `address` for `reason`. A write
    /// the run places exactly into storage the image or the write guard
    /// states may not be written ([`MemoryRefusal::ReadOnlyImage`]) is one
    /// the machine faults, where the program's own store makes it
    /// ([`Self::fault`]). One through an address the run cannot place (in an
    /// opaque or stack-derived root, which the guard judges as a write it
    /// cannot place) may land elsewhere, and like any other reason refuses
    /// the execution.
    pub fn deny_write(&mut self, reason: MemoryRefusal, address: Address, len: usize) {
        let placed = !matches!(
            address.root(),
            Root::Opaque { .. } | Root::StackDerived { .. }
        );
        if reason == MemoryRefusal::ReadOnlyImage && placed {
            self.fault
                .get_or_insert((address, u64::try_from(len).unwrap_or(u64::MAX)));
        } else {
            self.refuse(reason);
        }
    }

    /// Admits an access of `len` bytes at `address`, moving bytes as
    /// `transfer` says: records what it finds wrong with the program at the
    /// running instruction, where it lands ([`Self::access_finding`]) and
    /// against the object its pointer was formed into ([`Self::set_through`]),
    /// and returns the refusal where the executor cannot model the access at
    /// all. An access the program should not make is still made.
    pub fn admit(&self, address: Address, len: usize, transfer: Transfer) -> Option<MemoryRefusal> {
        if let Some(kind) = self.access_finding(address, len) {
            let accessed = Some((address, u64::try_from(len).unwrap_or(u64::MAX)));
            self.record(kind, accessed, self.found_object(address));
        }
        if let Some(object) = self.through.object(transfer) {
            self.judge_formed(address, len, object);
        }
        self.access_refusal(address, len)
    }

    /// States the objects the running access goes through: every access
    /// admitted from here until the next statement is judged against the
    /// object its direction names, whatever object it lands in. Whoever
    /// knows the access's pointers states them before it and
    /// [`Through::default`] after it.
    pub const fn set_through(&mut self, through: Through) {
        self.through = through;
    }

    /// Whether an access of `len` bytes at `address` is one the program may
    /// make and the executor can model: nothing found, nothing refused.
    /// Asks; records nothing.
    pub fn is_sound(&self, address: Address, len: usize) -> bool {
        self.access_refusal(address, len).is_none() && self.access_finding(address, len).is_none()
    }

    /// The refusal of an access of `len` bytes at `address` the executor
    /// cannot model: one whose span wraps the address space.
    pub fn access_refusal(&self, address: Address, len: usize) -> Option<MemoryRefusal> {
        (len != 0 && address.offset_at(len.saturating_sub(1)).is_none())
            .then_some(MemoryRefusal::AddressRange)
    }

    /// What is wrong with an access of `len` bytes at `address`, as the
    /// program would make it: heap storage whose lifetime ended, or a span
    /// past the extent of the object its first byte lies in (a heap block's
    /// as its allocator made it, any other's as the host's table states it).
    #[must_use]
    pub fn access_finding(&self, address: Address, len: usize) -> Option<FindingKind> {
        if len == 0 || address.offset_at(len.saturating_sub(1)).is_none() {
            return None;
        }
        if !matches!(address.root(), Root::Heap(_)) {
            let object = self.extents.as_ref()?.table.object(address)?;
            return u64::try_from(len)
                .is_ok_and(|len| !object.holds(address, len))
                .then_some(FindingKind::PastExtent(object.source));
        }
        let object = self.objects.get(&address.root())?;
        let ending = match object.lifetime {
            Lifetime::Live => None,
            Lifetime::Ended { .. } => Some(Ending::Every),
            Lifetime::EndedOnSome { .. } => Some(Ending::Some),
            Lifetime::PossiblyEnded { .. } => Some(Ending::Possibly),
        };
        if let Some(ending) = ending {
            return Some(FindingKind::Ended(ending));
        }
        let extent = object.extent?;
        let within = address
            .offset()
            .cast_unsigned()
            .checked_add(len as u64)
            .is_some_and(|end| end <= extent);
        (!within).then_some(FindingKind::PastExtent(ExtentSource::Allocator))
    }

    /// Records `kind` as found at the running instruction, in the run's
    /// record, with the bytes `access` reached, the `object` it was judged
    /// against and the decisions of the path that found it
    /// ([`Self::set_decisions`]). A memory outside a run, or before its first
    /// instruction, records nothing.
    pub fn record(
        &self,
        kind: FindingKind,
        access: Option<(Address, u64)>,
        object: Option<FoundObject>,
    ) {
        if let (Some(findings), Some(site)) = (
            &self.findings,
            self.current.0.and_then(|writer| writer.site),
        ) && !findings.has(site.body, site.block, site.instruction, kind)
        {
            // The first path to find it gives the evidence: a loop that finds
            // it on every iteration reads its witness out once.
            findings.record(Finding {
                body: site.body,
                block: site.block,
                instruction: site.instruction,
                kind,
                access,
                object,
                witness: self.decisions.witness(),
            });
        }
    }

    /// The object `address` lies in, as a finding names it: the heap block
    /// its root is, or the object outside the heap the host's table states.
    fn found_object(&self, address: Address) -> Option<FoundObject> {
        match address.root() {
            root @ Root::Heap(_) => {
                let object = self.objects.get(&root)?;
                let ended_at = match &object.lifetime {
                    Lifetime::Live => BTreeSet::new(),
                    Lifetime::Ended { sites }
                    | Lifetime::EndedOnSome { sites }
                    | Lifetime::PossiblyEnded { sites } => sites.clone(),
                };
                Some(FoundObject::Block {
                    root,
                    size: object.size,
                    extent: object.extent,
                    family: object.family,
                    allocated_at: object.site,
                    ended_at,
                })
            }
            root => self
                .stated_object(address)
                .map(|object| FoundObject::Stated { root, object }),
        }
    }

    /// The decisions of the path this memory runs on.
    #[must_use]
    pub const fn decisions(&self) -> &Decisions {
        &self.decisions
    }

    /// States the decisions of the path this memory runs on, which every
    /// finding it records from here carries as its witness.
    pub fn set_decisions(&mut self, decisions: Decisions) {
        self.decisions = decisions;
    }

    /// Hands the run the host's table of objects outside the heap, which
    /// every access outside the heap is judged against from here on.
    pub fn set_object_extents(&mut self, table: Arc<dyn ObjectExtents>) {
        self.extents = Some(Extents { table });
    }

    /// Hands the run's record of findings to this memory, which records in
    /// it what its accesses find wrong with the program.
    pub fn set_findings(&mut self, findings: Findings) {
        self.findings = Some(findings);
    }

    /// The object outside the heap `address` lies in, as the host's table
    /// states it, with where its extent comes from.
    #[must_use]
    pub fn stated_object(&self, address: Address) -> Option<StorageObject> {
        self.extents.as_ref()?.table.object(address)
    }

    /// Whether the host states objects outside the heap for this run: what a
    /// pointer outside the heap can be formed into.
    #[must_use]
    pub const fn states_objects(&self) -> bool {
        self.extents.is_some()
    }

    /// Keeps `object`, the start of the object a pointer was formed into, as
    /// that of the pointer just stored in `width` bytes at `at`.
    pub fn keep_formed(&mut self, at: Address, width: usize, object: Address) {
        self.formed.store(at, width, object);
    }

    /// The start of the object the pointer stored in exactly `width` bytes
    /// at `at` was formed into, where the span still holds it.
    #[must_use]
    pub fn formed(&self, at: Address, width: usize) -> Option<Address> {
        self.formed.load(at, width)
    }

    /// Records an access of `len` bytes at `address`, through a pointer
    /// formed into the object starting at `object`, that runs outside that
    /// object as the host's table states it: past its extent, whatever
    /// object the access itself lands in.
    fn judge_formed(&self, address: Address, len: usize, object: Address) {
        // An access in another root than the object's, as through an
        // unknown address, is no access this judges.
        if address.root() != object.root() {
            return;
        }
        let Some(stated) = self.stated_object(object) else {
            return;
        };
        let within = u64::try_from(len).is_ok_and(|len| stated.holds(address, len));
        if !within && len != 0 {
            self.record(
                FindingKind::PastExtent(stated.source),
                Some((address, u64::try_from(len).unwrap_or(u64::MAX))),
                Some(FoundObject::Stated {
                    root: object.root(),
                    object: stated,
                }),
            );
        }
    }

    /// The allocation `root` names, where it is live on every path.
    fn live(&self, root: Root) -> Option<&Object> {
        self.objects
            .get(&root)
            .filter(|object| object.lifetime.is_live())
    }

    /// Checks destination lifetime, coordinates and image protection before any write or escape.
    ///
    /// Target-width wrapping splits the span into linear image queries. A write
    /// covering a complete small address space checks that space only once.
    pub fn write_refusal(
        &self,
        address: Address,
        len: usize,
        image: Option<&dyn ImageData>,
    ) -> Option<MemoryRefusal> {
        if let Some(reason) = self.admit(address, len, Transfer::Write) {
            return Some(reason);
        }
        let (Some(at), Some(image)) = (address.image_address(), image) else {
            return None;
        };
        if len == 0 {
            return None;
        }
        let protected = if address.covers_space(len) {
            image.read_only_overlap(0, 1u64 << address.pointer_size().bits())
        } else {
            let prefix = address.linear_prefix_len(len);
            let suffix = len.saturating_sub(prefix);
            self.denies_image_write(address, at, prefix, image)
                || (suffix != 0 && image.read_only_overlap(0, suffix as u64))
        };
        protected.then_some(MemoryRefusal::ReadOnlyImage)
    }

    /// Whether any of the `len` image bytes from `address` (at `at` in the
    /// image) may not be written now: where a protection change installed
    /// permissions over a byte, those say; elsewhere the image's own.
    fn denies_image_write(
        &self,
        address: Address,
        at: u64,
        len: usize,
        image: &dyn ImageData,
    ) -> bool {
        let Ok(span) = u64::try_from(len) else {
            return true;
        };
        self.sources
            .permissions_over(address, span)
            .into_iter()
            .any(|(piece, bytes, installed)| match installed {
                Some(permissions) => !permissions.writable,
                None => {
                    let offset = piece
                        .offset()
                        .wrapping_sub(address.offset())
                        .cast_unsigned();
                    image.read_only_overlap(at.wrapping_add(offset), bytes)
                }
            })
    }

    /// Whether every one of the `len` image bytes from `address` may not be
    /// written now, so a write anywhere among them faults: where a
    /// protection change installed permissions over a byte, those say;
    /// elsewhere the image's own, a byte no region holds being none it
    /// denies. A span that wraps the address space is not.
    pub fn denies_every_image_write(
        &self,
        address: Address,
        len: usize,
        image: &dyn ImageData,
    ) -> bool {
        let (Some(at), Ok(span)) = (address.image_address(), u64::try_from(len)) else {
            return false;
        };
        if len == 0 || address.linear_prefix_len(len) != len {
            return false;
        }
        self.sources
            .permissions_over(address, span)
            .into_iter()
            .all(|(piece, bytes, installed)| match installed {
                Some(permissions) => !permissions.writable,
                None => {
                    let mut from = at.wrapping_add(
                        piece
                            .offset()
                            .wrapping_sub(address.offset())
                            .cast_unsigned(),
                    );
                    let end = from.saturating_add(bytes);
                    while from < end {
                        match image.region(from) {
                            Some(region)
                                if region.mutability == Mutability::ReadOnly
                                    && region.end > from =>
                            {
                                from = region.end;
                            }
                            _ => return false,
                        }
                    }
                    true
                }
            })
    }

    /// Whether the image byte at `address` still holds what the image binds,
    /// for a byte of read-only image: one no protection change covered on
    /// this path, or one no write since reached.
    fn image_byte_holds(&self, address: Address) -> bool {
        !self.sources.protection_changed(address, 1) || self.image_origins.unchanged(address, 1)
    }

    /// Records a refusal before a write can install a partial memory effect,
    /// and otherwise states that the write reaches its span: none of its
    /// bits is never written from here on.
    fn admit_write(&mut self, address: Address, len: usize) -> bool {
        if let Some(reason) = self.admit(address, len, Transfer::Write) {
            self.refusal.get_or_insert(reason);
            false
        } else {
            let start = address.offset();
            let end = i64::try_from(len).map_or(i64::MAX, |len| start.saturating_add(len));
            self.unwritten.write(address.root(), start, end);
            self.formed.write(address.root(), start, end);
            true
        }
    }

    /// Creates replacement storage carrying only the destination's allocation extent.
    pub fn write_candidate(&self, address: Address) -> Self {
        let mut candidate = Self::default();
        if let Some(object) = self.live(address.root()) {
            candidate.objects.insert(
                address.root(),
                Object::fresh(
                    BlockSize {
                        guaranteed: object.size,
                        extent: object.extent,
                    },
                    None,
                    Allocator {
                        family: object.family,
                        heap: object.heap,
                    },
                ),
            );
        }
        // The candidate states contents only; what was never written is the
        // memory's own.
        candidate.unwritten = Unwritten::default();
        candidate
    }

    /// Registers a fresh allocation of `size` made by `allocator`, without
    /// materializing its initial bytes: each is `initial` where it states
    /// one, else never written. A size of `None` is one the allocator did not
    /// state: no byte is known to belong to it, and no access is past it.
    pub fn allocate(
        &mut self,
        root: Root,
        size: impl Into<BlockSize>,
        initial: Option<u8>,
        allocator: impl Into<Allocator>,
    ) {
        let size = size.into();
        if initial.is_none() {
            self.unwritten
                .begin(root, 0, i64::try_from(size.guaranteed).unwrap_or(i64::MAX));
        } else {
            self.unwritten.write_root(root);
            self.formed.write_root(root);
        }
        self.objects
            .insert(root, Object::fresh(size, initial, allocator.into()));
    }

    /// States the call of the outermost activation, `(block, instruction)`,
    /// whose execution made the live allocation `root`.
    pub fn locate(&mut self, root: Root, site: (usize, usize)) {
        if let Some(object) = self.objects.get_mut(&root) {
            object.site = Some(site);
        }
    }

    /// The call of the outermost activation whose execution made the live
    /// allocation `root`, where every path agrees on one.
    #[must_use]
    pub fn allocation_site(&self, root: Root) -> Option<(usize, usize)> {
        self.objects.get(&root)?.site
    }

    /// Returns the bytes a live allocation is known to hold on every path,
    /// and the allocator that made it where every path agrees on one.
    #[must_use]
    pub fn object(&self, root: Root) -> Option<(u64, Option<AllocatorFamily>)> {
        self.live(root).map(|object| (object.size, object.family))
    }

    /// Returns the allocator that made the live allocation `root`: its
    /// family and its heap, each where every path agrees on it.
    #[must_use]
    pub fn made_by(&self, root: Root) -> Option<Allocator> {
        self.live(root).map(|object| Allocator {
            family: object.family,
            heap: object.heap,
        })
    }

    /// Deallocates `address` by allocator `family` where one is stated, the
    /// call at `site` doing it, and says what the address named.
    ///
    /// Only the start of an object live on every path, made by `family`,
    /// ends: its lifetime becomes ended at `site` and its contents are
    /// discarded ([`FreeOutcome::Released`]). Anything else changes nothing,
    /// and is classified: an object not live on every path, an interior
    /// pointer, another family's object (or one whose family the paths do
    /// not agree on, where a family is stated), the stack or the image, and
    /// an address the execution cannot name.
    ///
    /// # Arguments
    ///
    /// * `address` - The address deallocated.
    /// * `family` - The deallocator's allocator family, where it states one.
    /// * `site` - The call of the outermost activation whose execution
    ///   deallocates, as `(block, instruction)`.
    pub fn free(
        &mut self,
        address: Address,
        allocator: impl Into<Allocator>,
        site: Option<(usize, usize)>,
    ) -> FreeOutcome {
        // The block as the release finds it, before it ends.
        let object = self.found_object(address);
        let outcome = self.release(address, allocator.into(), site);
        if let Some(fault) = outcome.fault() {
            self.record(FindingKind::Release(fault), Some((address, 0)), object);
        }
        outcome
    }

    /// Ends the object `address` starts, where it is a live heap object of
    /// `family`, as [`Self::free`] does, finding nothing.
    fn release(
        &mut self,
        address: Address,
        allocator: Allocator,
        site: Option<(usize, usize)>,
    ) -> FreeOutcome {
        let root = address.root();
        match root {
            Root::Heap(_) => {}
            Root::Symbol | Root::Stack | Root::StackFrame(_) | Root::StackAligned { .. } => {
                return FreeOutcome::NotHeap;
            }
            _ => return FreeOutcome::Unknown,
        }
        let Some(object) = self.objects.get_mut(&root) else {
            return FreeOutcome::Unknown;
        };
        if !object.lifetime.is_live() {
            return FreeOutcome::Ended {
                lifetime: object.lifetime.clone(),
            };
        }
        if address.offset() != 0 {
            return FreeOutcome::Interior {
                offset: address.offset(),
            };
        }
        if !allocator.admits(object.family, object.heap) {
            return FreeOutcome::OtherFamily {
                made_by: object.family,
            };
        }
        object.lifetime = Lifetime::Ended {
            sites: site.into_iter().collect(),
        };
        self.erase_interval(root, i64::MIN, i64::MAX);
        FreeOutcome::Released
    }

    /// Ends every live block made in `heap`, as destroying the heap does,
    /// at `site`; a heap the execution cannot name may be any, so every
    /// lifetime becomes possibly ended ([`Self::invalidate_lifetimes`]).
    pub fn destroy_heap(&mut self, heap: Option<Root>, site: Option<(usize, usize)>) {
        let Some(heap) = heap else {
            self.invalidate_lifetimes();
            return;
        };
        let ended: Vec<Root> = self
            .objects
            .iter_mut()
            .filter(|(_, object)| object.heap == Some(heap) && object.lifetime.is_live())
            .map(|(root, object)| {
                object.lifetime = Lifetime::Ended {
                    sites: site.into_iter().collect(),
                };
                *root
            })
            .collect();
        for root in ended {
            self.erase_interval(root, i64::MIN, i64::MAX);
        }
    }

    /// Marks every heap lifetime an unknown call may end as possibly ended,
    /// and discards the contents of every heap object.
    pub fn invalidate_lifetimes(&mut self) {
        for object in self.objects.values_mut() {
            object.lifetime = object.lifetime.possibly_ended();
        }
        let mut next = self.cells.first_key_value().map(|((root, _), _)| root);
        while let Some(root) = next {
            next = self
                .cells
                .range((Excluded((root, i64::MAX)), Unbounded))
                .next()
                .map(|((root, _), _)| root);
            if matches!(root, Root::Heap(_)) {
                self.erase_interval(root, i64::MIN, i64::MAX);
            }
        }
    }

    /// Returns whether a heap access stays inside a live allocation's proven extent.
    fn accessible(&self, root: Root, offset: i64, len: usize) -> bool {
        if len == 0 || !matches!(root, Root::Heap(_)) {
            return true;
        }
        let Some(object) = self.live(root) else {
            return false;
        };
        offset
            .cast_unsigned()
            .checked_add(len as u64)
            .is_some_and(|end| end <= object.size)
    }

    /// Returns a stored cell or a byte supplied by a live object's initializer.
    fn cell(&self, root: Root, offset: i64) -> Option<Cell> {
        if !self.accessible(root, offset, 1) {
            return None;
        }
        self.cells.get(&(root, offset)).copied().or_else(|| {
            self.objects
                .get(&root)?
                .byte(offset.cast_unsigned())
                .map(Cell::Byte)
        })
    }

    /// Hides backing bytes overwritten by an unknown transfer, which may have
    /// written them.
    fn forget_backing(&mut self, root: Root, offset: i64, len: usize) {
        let end = i64::try_from(len).map_or(i64::MAX, |len| offset.saturating_add(len));
        self.unwritten.write(root, offset, end);
        self.formed.write(root, offset, end);
        if let Some(object) = self.objects.get_mut(&root)
            && let start = offset.cast_unsigned()
            && let Some(end) = start.checked_add(len as u64)
        {
            object.forget(start, end);
        }
    }

    /// Discards initial byte knowledge while retaining the allocation's
    /// lifetime and extent: an unknown effect may have written any of it.
    fn clear_backing(&mut self, root: Root) {
        self.unwritten.write_root(root);
        self.formed.write_root(root);
        if let Some(object) = self.objects.get_mut(&root) {
            object.initial = None;
            object.unknown.clear();
        }
    }

    /// Marks the stack as escaped: from now on any name may reach all of it.
    pub fn escape_stack(&mut self) {
        self.stack_escaped = true;
    }

    /// Whether an address into the stack has escaped.
    #[cfg(test)]
    pub const fn stack_escaped(&self) -> bool {
        self.stack_escaped
    }

    /// Whether the byte at `root + offset` is one only this execution reaches:
    /// the stack below the entry pointer, a callee-local frame, and a frame
    /// realigned from either that lies wholly below it.
    const fn is_private(&self, root: Root, offset: i64) -> bool {
        let private = match root {
            Root::Stack => offset < 0,
            Root::StackFrame(_) | Root::StackAligned { frame: Some(_), .. } => true,
            Root::StackAligned {
                frame: None, at, ..
            } => at.saturating_add(offset) < 0,
            _ => false,
        };
        private && !self.stack_escaped
    }

    /// Writes `bytes` at `address`.
    pub fn store_bytes(&mut self, address: Address, bytes: &[u8]) {
        self.store_encoded(address, bytes.iter().copied());
    }

    /// Stores an encoding after checking its complete destination extent.
    ///
    /// The entire extent is checked before any write or source-byte inspection.
    pub fn store_encoded(
        &mut self,
        address: Address,
        mut bytes: impl ExactSizeIterator<Item = u8>,
    ) {
        let len = bytes.len();
        if !self.admit_write(address, len) {
            return;
        }
        self.forget_aliases(address, len);
        for index in 0..len {
            let Some(byte) = bytes.next() else {
                return;
            };
            if let Some(at) = address.offset_at(index) {
                self.cells.insert((address.root(), at), Cell::Byte(byte));
            }
        }
    }

    /// Repeats a byte pattern directly into memory.
    ///
    /// An empty pattern establishes unknown bytes. Extent checks and source
    /// cancellation share the exact encoding path used by scalar stores.
    pub fn store_pattern(&mut self, address: Address, pattern: &[u8], len: usize) {
        if pattern.is_empty() {
            self.store_unknown(address, len);
            return;
        }
        self.store_encoded(
            address,
            PatternBytes {
                pattern,
                offset: 0,
                remaining: len,
            },
        );
    }

    /// Writes `value`, an address, `width` bytes wide at `address`.
    pub fn store_address(&mut self, address: Address, value: Address, width: u8) {
        self.store_address_prefix(address, value, width, width);
    }

    /// Writes a prefix retaining the pointer's complete representation width.
    pub fn store_address_prefix(&mut self, address: Address, value: Address, width: u8, bytes: u8) {
        if bytes > width {
            self.refuse(MemoryRefusal::BlockLayout);
            return;
        }
        let len = usize::from(bytes);
        if !self.admit_write(address, len) {
            return;
        }
        if value.root().may_reference_stack()
            && len > 0
            && (!self.is_private(address.root(), address.offset())
                || address
                    .offset_at(len.saturating_sub(1))
                    .is_none_or(|end| !self.is_private(address.root(), end)))
        {
            self.escape_stack();
        }
        self.forget_aliases(address, len);
        for index in 0..bytes {
            if let Some(at) = address.offset_at(usize::from(index)) {
                self.cells.insert(
                    (address.root(), at),
                    Cell::Piece {
                        address: value,
                        index,
                        width,
                    },
                );
            }
        }
    }

    /// Writes `len` unknown bytes, including spans crossing the target modulus.
    pub fn store_unknown(&mut self, address: Address, len: usize) {
        if len == 0 {
            return;
        }
        if !self.admit_write(address, len) {
            return;
        }
        self.forget_aliases(address, len);
        let root = address.root();
        let start = address.offset();
        self.forget_backing(root, start, len);
        let end = address.offset_at(len.saturating_sub(1));
        if address.covers_space(len) || end.is_none() {
            self.erase_interval(root, i64::MIN, i64::MAX);
        } else if let Some(end) = end {
            if start <= end {
                self.erase_interval(root, start, end);
            } else {
                self.erase_interval(root, start, i64::MAX);
                self.erase_interval(root, i64::MIN, end);
            }
        }
    }

    /// Erases an inclusive interval without overflowing its final coordinate:
    /// whatever erased it may have written it.
    fn erase_interval(&mut self, root: Root, start: i64, end: i64) {
        if start > end {
            return;
        }
        self.unwritten.write(root, start, end.saturating_add(1));
        self.formed.write(root, start, end.saturating_add(1));
        while let Some(key) = {
            self.cells
                .range((root, start)..=(root, end))
                .next()
                .map(|(key, _)| key)
        } {
            self.cells.remove(&key);
        }
    }

    /// The realigned frames ([`Root::StackAligned`]) of `frame` some cell is
    /// held under.
    fn realigned_roots(&self, frame: Root) -> Vec<Root> {
        let first = Root::StackAligned {
            frame: None,
            at: i64::MIN,
            bits: 0,
        };
        let mut roots = Vec::new();
        let mut next = self
            .cells
            .range((first, i64::MIN)..)
            .next()
            .map(|((root, _), _)| root);
        while let Some(root) = next {
            let Some((of, _)) = root.realigned_window() else {
                break;
            };
            if of == frame {
                roots.push(root);
            }
            next = self
                .cells
                .range((Excluded((root, i64::MAX)), Unbounded))
                .next()
                .map(|((root, _), _)| root);
        }
        roots
    }

    /// Forgets every cell of a frame realigned from `frame` that may lie at
    /// its offsets `start..=end`: a cell at offset `d` of a base realigned
    /// from `at` lies somewhere in `at - 2^bits + 1 + d ..= at + d`.
    fn erase_realigned(&mut self, frame: Root, start: i64, end: i64) {
        for root in self.realigned_roots(frame) {
            let Some((_, window)) = root.realigned_window() else {
                continue;
            };
            self.erase_interval(
                root,
                start.saturating_sub(*window.end()),
                end.saturating_sub(*window.start()),
            );
        }
    }

    /// Forgets every cell of a frame realigned from `frame` that lies below
    /// its offset `entry` wherever the base is: what an activation that has
    /// returned leaves dead there.
    fn release_realigned(&mut self, frame: Root, entry: i64) {
        for root in self.realigned_roots(frame) {
            let Some((_, window)) = root.realigned_window() else {
                continue;
            };
            self.erase_interval(
                root,
                i64::MIN,
                entry.saturating_sub(*window.end()).saturating_sub(1),
            );
        }
    }

    /// Forgets every byte under another root that a store of `len` bytes at
    /// `address` may also write.
    fn forget_aliases(&mut self, address: Address, len: usize) {
        if len == 0 {
            return;
        }
        let disjoint_numeric = self.disjoint_numeric_span(address, len);
        if !disjoint_numeric {
            self.image_origins.write(address, len);
        }
        self.writers.write(address, len, self.current.0);
        let root = address.root();
        let end = i64::try_from(len)
            .ok()
            .and_then(|len| address.offset().checked_add(len));
        let private = !self.stack_escaped
            && match root {
                Root::StackFrame(_) | Root::StackAligned { frame: Some(_), .. } => true,
                Root::Stack => end.is_some_and(|end| end <= 0),
                Root::StackAligned {
                    frame: None, at, ..
                } => end.is_some_and(|end| at.saturating_add(end) <= 0),
                _ => false,
            };
        let escaped = self.stack_escaped;
        let own = root;
        let named_elsewhere = |key: &(Root, i64)| -> bool {
            let (other, at) = *key;
            if other == own {
                return false;
            }
            match (own, other) {
                (Root::StackDerived { .. }, _) | (_, Root::StackDerived { .. }) => true,
                // A realigned frame and the frame it was realigned from alias
                // in a bounded window, forgotten below; two realigned from one
                // frame may overlap anywhere.
                (Root::StackAligned { .. }, Root::StackAligned { .. }) => {
                    own.realigned_window().map(|(frame, _)| frame)
                        == other.realigned_window().map(|(frame, _)| frame)
                }
                (Root::StackAligned { .. }, Root::Stack | Root::StackFrame(_))
                | (Root::Stack | Root::StackFrame(_), Root::StackAligned { .. }) => false,
                (Root::Number | Root::Opaque { .. }, Root::StackAligned { .. }) => escaped,
                (Root::Number, other) if disjoint_numeric => matches!(other, Root::Opaque { .. }),
                // Integer addresses and opaque values may name anything but
                // the private stack.
                (Root::Number | Root::Opaque { .. }, Root::Stack) => at >= 0 || escaped,
                (Root::Number | Root::Opaque { .. }, Root::StackFrame(_)) => escaped,
                (Root::Number | Root::Opaque { .. }, _) => true,
                // Everything else is named by integers and opaque values only.
                (_, Root::Number | Root::Opaque { .. }) => !private,
                _ => false,
            }
        };
        // Skip complete root ranges instead of testing every byte in the
        // destination's growing buffer on each store. Only aliased roots pay
        // for visiting their cells, exactly when those facts are invalidated.
        let mut next = self.cells.first_key_value().map(|((root, _), _)| root);
        while let Some(other) = next {
            next = self
                .cells
                .range((Excluded((other, i64::MAX)), Unbounded))
                .next()
                .map(|((root, _), _)| root);
            let start = if !escaped
                && other == Root::Stack
                && matches!(own, Root::Number | Root::Opaque { .. })
            {
                0
            } else {
                i64::MIN
            };
            if named_elsewhere(&(other, start)) {
                if other == Root::Number
                    && matches!(
                        own,
                        Root::Symbol
                            | Root::Stack
                            | Root::StackFrame(_)
                            | Root::StackAligned { .. }
                            | Root::Heap(_)
                    )
                {
                    self.forget_unmapped_numeric();
                } else {
                    self.erase_interval(other, start, i64::MAX);
                }
            }
        }
        // Storage never written holds no cell, and a store through a name
        // that may alias it may write it as well.
        if matches!(
            own,
            Root::Number
                | Root::Opaque { .. }
                | Root::StackDerived { .. }
                | Root::StackAligned { .. }
        ) {
            let mut roots = self.unwritten.roots();
            roots.extend(self.formed.roots());
            roots.sort_unstable();
            roots.dedup();
            for other in roots {
                let start = if !escaped
                    && other == Root::Stack
                    && matches!(own, Root::Number | Root::Opaque { .. })
                {
                    0
                } else {
                    i64::MIN
                };
                if named_elsewhere(&(other, start)) {
                    self.unwritten.write(other, start, i64::MAX);
                    self.formed.write(other, start, i64::MAX);
                }
            }
        }
        // The window a realigned store may land in, in the frame it was
        // realigned from; and the realigned cells a store into that frame may
        // overlap.
        let last = address
            .offset()
            .saturating_add(i64::try_from(len).unwrap_or(i64::MAX))
            .saturating_sub(1);
        if let Some((frame, window)) = root.realigned_window() {
            self.erase_interval(
                frame,
                window.start().saturating_add(address.offset()),
                window.end().saturating_add(last),
            );
        } else if matches!(root, Root::Stack | Root::StackFrame(_)) {
            self.erase_realigned(root, address.offset(), last);
        }
        if !private {
            let mut next = self.objects.first_key_value().map(|(root, _)| *root);
            while let Some(root) = next {
                next = self
                    .objects
                    .range((Excluded(root), Unbounded))
                    .next()
                    .map(|(root, _)| *root);
                if named_elsewhere(&(root, 0)) {
                    self.clear_backing(root);
                }
            }
        }
    }

    /// Makes unknown every byte a call with no known effect may write: all of
    /// them but the private stack. Such a call may write code as well, so no
    /// instruction fetched since is vouched for, and no image byte is still
    /// the one the image binds.
    pub fn clobber(&mut self) {
        self.sources.dirty_fetch();
        self.image_origins.clobber();
        self.clobber_data();
    }

    /// Makes unknown every byte a call that writes no code may write: all of
    /// them but the private stack, as [`Self::clobber`] does, while the
    /// instructions fetched since stay vouched for and the image keeps the
    /// initial bindings of its code and of its pointer cells that bind code
    /// (an import's slot): the routine writes neither. Writable image bytes
    /// read unknown after it, written or not, since only read-only image is
    /// read from the image where no cell holds it.
    pub fn clobber_data(&mut self) {
        let mut next = self.objects.first_key_value().map(|(root, _)| *root);
        while let Some(root) = next {
            next = self
                .objects
                .range((Excluded(root), Unbounded))
                .next()
                .map(|(root, _)| *root);
            self.clear_backing(root);
        }
        let mut next = self.cells.first_key_value().map(|((root, _), _)| root);
        while let Some(root) = next {
            next = self
                .cells
                .range((Excluded((root, i64::MAX)), Unbounded))
                .next()
                .map(|((root, _), _)| root);
            self.clobber_root(root);
        }
        // Storage never written holds no cell, and the call may write it as
        // well; so may a span holding a stored pointer.
        let mut roots = self.unwritten.roots();
        roots.extend(self.formed.roots());
        roots.sort_unstable();
        roots.dedup();
        for root in roots {
            self.clobber_root(root);
        }
    }

    /// Makes unknown what of `root` a call with no known effect may write.
    fn clobber_root(&mut self, root: Root) {
        match root {
            Root::Stack if !self.stack_escaped => self.erase_interval(root, 0, i64::MAX),
            Root::StackFrame(_) | Root::StackAligned { frame: Some(_), .. }
                if !self.stack_escaped => {}
            // What may lie at or above the entry pointer is the caller's.
            Root::StackAligned {
                frame: None, at, ..
            } if !self.stack_escaped => {
                self.erase_interval(root, at.saturating_neg(), i64::MAX);
            }
            _ => self.erase_interval(root, i64::MIN, i64::MAX),
        }
    }

    /// Forgets the stack below `entry`, the pointer an activation that has
    /// just returned was entered with: those bytes are dead, and the frame
    /// of the next activation entered there begins never written.
    pub fn release_below(&mut self, entry: i64) {
        while let Some(key) = {
            self.cells
                .range((Root::Stack, i64::MIN)..(Root::Stack, entry))
                .next()
                .map(|(key, _)| key)
        } {
            self.cells.remove(&key);
        }
        self.release_realigned(Root::Stack, entry);
        if !self.stack_escaped {
            self.unwritten.begin(Root::Stack, i64::MIN, entry);
        }
    }

    /// States that the frame of an activation entered with its stack pointer
    /// at `entry` of `root` begins never written below it.
    pub fn begin_frame(&mut self, root: Root, entry: i64) {
        if !self.stack_escaped {
            self.unwritten.begin(root, i64::MIN, entry);
        }
    }

    /// The bits of the `len` bytes at `address` that no write has reached
    /// since their object began, byte by byte from the lowest address:
    /// `None` where every bit was written, or may have been.
    #[must_use]
    pub fn unwritten_bits(&self, address: Address, len: usize) -> Option<Vec<u8>> {
        if self.unwritten.is_empty() || len == 0 {
            return None;
        }
        let bits: Vec<u8> = (0..len)
            .map(|index| {
                address
                    .offset_at(index)
                    .map_or(0, |at| self.unwritten.bits(address.root(), at))
            })
            .collect();
        bits.iter().any(|bits| *bits != 0).then_some(bits)
    }

    /// Records the never-written bits a value carried into the `bits.len()`
    /// bytes just stored at `address`, byte by byte from the lowest address.
    pub fn carry_unwritten(&mut self, address: Address, bits: &[u8]) {
        if bits.iter().all(|bits| *bits == 0) {
            return;
        }
        self.unwritten.carry(address.root(), address.offset(), bits);
    }

    /// Releases only storage owned by a detached activation or below its ABI entry.
    pub fn release_activation(&mut self, root: Root, entry: i64, activation: u32) {
        let end = if root == Root::StackFrame(activation) {
            i64::MAX
        } else {
            let Some(end) = entry.checked_sub(1) else {
                return;
            };
            end
        };
        self.erase_interval(root, i64::MIN, end);
        self.release_realigned(root, end.saturating_add(1));
    }

    /// Reads one known byte without allocating a scalar-load buffer.
    pub fn known_byte(&self, address: Address) -> Option<u8> {
        self.byte_at(address, 0, None)
    }

    /// Proves a read span is backed without allocating or retaining its bytes.
    ///
    /// Live heap extents and complete mapped image spans need no byte walk.
    /// Other storage requires a cell or immutable image byte at each coordinate.
    pub fn readable_span(
        &self,
        address: Address,
        len: usize,
        image: Option<&dyn ImageData>,
    ) -> bool {
        if !self.accessible(address.root(), address.offset(), len) {
            return false;
        }
        if len == 0 || matches!(address.root(), Root::Heap(_)) {
            return true;
        }
        if let Some(at) = address.image_address()
            && let Some(image) = image
            && image
                .region(at)
                .is_some_and(|region| region.covers(at, len as u64))
            && image.bytes(at, len).is_some_and(|bytes| bytes.len() == len)
        {
            return true;
        }
        (0..len).all(|index| {
            address
                .offset_at(index)
                .is_some_and(|at| self.cell(address.root(), at).is_some())
                || self.byte_at(address, index, image).is_some()
        })
    }

    /// Borrows a complete immutable image cell's address fact without overriding stored cells.
    ///
    /// Read-only backing is the oracle's lifetime promise. Legal writes cannot
    /// invalidate it; explicit execution cells still take precedence. Mutable,
    /// partial, wrapping, unserved and target-unrepresentable cells supply no fact.
    pub fn unchanged_image_address(
        &self,
        address: Address,
        len: usize,
        image: &dyn ImageData,
    ) -> Option<Address> {
        if self.refusal.is_some()
            || address.root() != Root::Symbol
            || len != address.pointer_size().bytes()
            || address.linear_prefix_len(len) != len
        {
            return None;
        }
        for index in 0..len {
            if self
                .cells
                .get(&(address.root(), address.offset_at(index)?))
                .is_some()
            {
                return None;
            }
        }
        let at = address.image_address()?;
        let extent = u64::try_from(len).ok()?;
        let region = image.region(at)?;
        if region.mutability != Mutability::ReadOnly
            || !region.covers(at, extent)
            || (self.sources.protection_changed(address, extent)
                && !self.image_origins.unchanged(address, len))
        {
            return None;
        }
        if image.bytes(at, len)?.len() != len {
            return None;
        }
        let named = image.address_at(at, extent)?;
        if address.pointer_size().mask_unsigned(named) != named {
            return None;
        }
        Some(Address::new(
            Root::Symbol,
            named.cast_signed(),
            address.pointer_size(),
        ))
    }

    /// Reads `len` bytes at `address`.
    ///
    /// # Arguments
    ///
    /// * `address` - Where the read starts.
    /// * `len` - How many bytes it reads.
    /// * `image` - Where a never-written image byte is read from.
    pub fn load(&self, address: Address, len: usize, image: Option<&dyn ImageData>) -> Loaded {
        if !self.accessible(address.root(), address.offset(), len) {
            return Loaded::Unknown;
        }
        if let Some(address) = self.load_address(address, len) {
            return Loaded::Address(address);
        }
        let mut bytes = Vec::with_capacity(len);
        for index in 0..len {
            match self.byte_at(address, index, image) {
                Some(byte) => bytes.push(byte),
                None => {
                    let stack_bits = (0..len).any(|index| {
                        address
                            .offset_at(index)
                            .and_then(|offset| self.cell(address.root(), offset))
                            .is_some_and(Cell::may_reference_stack)
                    });
                    return if stack_bits {
                        Loaded::StackDerived
                    } else {
                        Loaded::Unknown
                    };
                }
            }
        }
        Loaded::Bytes(bytes)
    }

    /// Copies bytes and symbolic pointer pieces with forward or snapshot semantics.
    ///
    /// Pointer pieces retain their original indices, so a complete transferred
    /// pointer can be loaded as an address without inventing numeric bytes.
    /// Returns whether every actual source read has readable backing, independently
    /// of whether its contents are known. A refused transfer returns `false`.
    pub fn copy(
        &mut self,
        to: Address,
        from: Address,
        len: i64,
        image: Option<&dyn ImageData>,
        buffered: bool,
    ) -> bool {
        self.copy_ordered(
            to,
            from,
            len,
            image,
            if buffered {
                CopyOrder::Snapshot
            } else {
                CopyOrder::Stream {
                    element_bytes: 1,
                    reverse: false,
                }
            },
        )
    }

    /// Copies complete elements in stream order, preserving symbolic pointer pieces.
    ///
    /// Returns whether every source read has readable backing at the instant it
    /// executes. Earlier stream writes can provide later source cells; a snapshot
    /// reads its entire source before any write. A refused transfer returns `false`.
    fn copy_ordered(
        &mut self,
        to: Address,
        from: Address,
        len: i64,
        image: Option<&dyn ImageData>,
        order: CopyOrder,
    ) -> bool {
        self.copy_with_guard(to, from, len, image, order, None)
    }

    /// Preflights a complete transfer and its path-local host constraint before changing memory.
    pub fn copy_with_guard(
        &mut self,
        to: Address,
        from: Address,
        len: i64,
        image: Option<&dyn ImageData>,
        order: CopyOrder,
        guard: Option<&dyn MemoryWriteGuard>,
    ) -> bool {
        let ptr_size = to.pointer_size();
        if ptr_size != from.pointer_size() {
            self.refuse(MemoryRefusal::AddressRange);
            return false;
        }
        let Ok(bytes) = usize::try_from(len) else {
            self.clobber();
            return false;
        };
        let (to, from) = match (order.span(to, bytes), order.span(from, bytes)) {
            (Ok(to), Ok(from)) => (to, from),
            (Err(reason), _) | (_, Err(reason)) => {
                self.refuse(reason);
                return false;
            }
        };
        if let Some(reason) = self.write_refusal(to, bytes, image) {
            self.deny_write(reason, to, bytes);
            return false;
        }
        if let Some(reason) = self.admit(from, bytes, Transfer::Read) {
            self.refusal.get_or_insert(reason);
            return false;
        }
        if bytes != 0
            && let Some(guard) = guard
            && let Err(reason) = check_write(guard, &mut self.sources, to, bytes)
        {
            self.refuse(reason);
            return false;
        }
        let mut readable = true;
        let mut read = |memory: &Self, delta| {
            let Some(offset) = usize::try_from(delta)
                .ok()
                .and_then(|index| from.offset_at(index))
            else {
                readable = false;
                return (None, 0);
            };
            let address = Address::new(from.root(), offset, ptr_size);
            let (cell, backed) = memory.transfer_cell(address, image);
            readable &= backed;
            (cell, memory.unwritten.bits(from.root(), offset))
        };
        // Which aliases a byte's store invalidates depends on the byte only
        // through whether it is private stack, so forgetting them for the
        // whole destination at the first write is the union of the per-byte
        // forgets, at one walk over the roots instead of one per byte. Only an
        // escape the copy itself causes widens that set, and forgets again.
        let mut forgotten_escaped = None;
        // Each byte copied carries the bits of its source no write reached.
        let mut write = |memory: &mut Self, delta, (cell, unwritten): (Option<Cell>, u8)| {
            let Some(offset) = usize::try_from(delta)
                .ok()
                .and_then(|index| to.offset_at(index))
            else {
                memory.clobber();
                return;
            };
            let address = Address::new(to.root(), offset, ptr_size);
            if cell.is_some_and(Cell::may_reference_stack)
                && !memory.is_private(address.root(), address.offset())
            {
                memory.escape_stack();
            }
            if forgotten_escaped != Some(memory.stack_escaped) {
                memory.forget_aliases(to, bytes);
                forgotten_escaped = Some(memory.stack_escaped);
            }
            let key = (address.root(), address.offset());
            match cell {
                Some(cell) => {
                    memory.cells.insert(key, cell);
                }
                None => {
                    memory.cells.remove(&key);
                    memory.forget_backing(address.root(), address.offset(), 1);
                }
            }
            // Written now, with whatever bits of the source were never.
            memory
                .unwritten
                .write(to.root(), offset, offset.saturating_add(1));
            if unwritten != 0 {
                memory.unwritten.carry(to.root(), offset, &[unwritten]);
            }
        };
        if matches!(order, CopyOrder::Snapshot) {
            let mut cells = Vec::with_capacity(bytes);
            for delta in 0..len {
                cells.push(read(self, delta));
            }
            for (delta, cell) in (0..len).zip(cells) {
                write(self, delta, cell);
            }
        } else if let CopyOrder::Stream {
            element_bytes,
            reverse,
        } = order
        {
            // Byte streams need no temporary allocation, including backward streams.
            if element_bytes == 1 {
                for step in 0..len {
                    let delta = if reverse {
                        len.saturating_sub(step).saturating_sub(1)
                    } else {
                        step
                    };
                    let cell = read(self, delta);
                    write(self, delta, cell);
                }
                return readable;
            }
            let mut cells = Vec::with_capacity(element_bytes.min(bytes));
            let Some(count) = bytes.checked_div(element_bytes) else {
                self.refuse(MemoryRefusal::BlockLayout);
                return false;
            };
            for step in 0..count {
                let element = if reverse {
                    count.saturating_sub(step).saturating_sub(1)
                } else {
                    step
                };
                let base = element.saturating_mul(element_bytes);
                cells.clear();
                for offset in 0..element_bytes {
                    cells.push(read(self, base.saturating_add(offset) as i64));
                }
                for (offset, cell) in cells.iter().copied().enumerate() {
                    write(self, base.saturating_add(offset) as i64, cell);
                }
            }
        }
        readable
    }

    /// Reads one transfer cell and proves backing without requiring a known byte.
    fn transfer_cell(
        &self,
        address: Address,
        image: Option<&dyn ImageData>,
    ) -> (Option<Cell>, bool) {
        if let Some(cell) = self.cell(address.root(), address.offset()) {
            return (Some(cell), true);
        }
        if matches!(address.root(), Root::Heap(_)) {
            return (None, self.accessible(address.root(), address.offset(), 1));
        }
        let (Some(at), Some(image)) = (address.image_address(), image) else {
            return (None, false);
        };
        let Some(region) = image.region(at).filter(|region| region.covers(at, 1)) else {
            return (None, false);
        };
        let Some([byte]) = image.bytes(at, 1) else {
            return (None, false);
        };
        let cell = (region.mutability == Mutability::ReadOnly && self.image_byte_holds(address))
            .then_some(Cell::Byte(*byte));
        (cell, true)
    }

    /// The address stored whole at exactly the `len` bytes at `address`.
    fn load_address(&self, address: Address, len: usize) -> Option<Address> {
        let width = u8::try_from(len).ok()?;
        let mut found = None;
        for index in 0..width {
            let at = address.offset_at(usize::from(index))?;
            match self.cells.get(&(address.root(), at))? {
                Cell::Piece {
                    address: stored,
                    index: piece,
                    width: stored_width,
                } if *piece == index && *stored_width == width => {
                    if found.is_some_and(|first| first != *stored) {
                        return None;
                    }
                    found = Some(*stored);
                }
                Cell::Piece { .. } | Cell::Byte(_) | Cell::StackBits => return None,
            }
        }
        found
    }

    /// The known byte `index` bytes past `address`: held, or read-only in the
    /// image and never written.
    fn byte_at(&self, address: Address, index: usize, image: Option<&dyn ImageData>) -> Option<u8> {
        let at = address.offset_at(index)?;
        match self.cell(address.root(), at) {
            Some(Cell::Byte(byte)) => Some(byte),
            Some(Cell::Piece { .. } | Cell::StackBits) => None,
            None if address.root() == Root::Symbol => {
                let image = image?;
                let location = at.cast_unsigned();
                let region = image.region(location)?;
                let byte = Address::new(Root::Symbol, at, address.pointer_size());
                if region.mutability != Mutability::ReadOnly || !self.image_byte_holds(byte) {
                    return None;
                }
                image.bytes(location, 1)?.first().copied()
            }
            None => None,
        }
    }

    /// Collects bytes before a string's first NUL within `limit`.
    ///
    /// Unknown bytes or a missing terminator leave no string, without
    /// exposing a partial buffer. Either way the scan says how many bytes it
    /// inspected.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::analysis::execution::MemoryRefusal`] when a visited
    /// byte has an unrepresentable coordinate or lacks a live object extent.
    pub fn read_string(
        &self,
        address: Address,
        limit: usize,
        image: Option<&dyn ImageData>,
    ) -> Result<StringScan<Vec<u8>>, MemoryRefusal> {
        let mut bytes = Vec::new();
        let scan = self.scan_string(address, limit, image, |byte| {
            bytes.push(byte);
            true
        })?;
        Ok(StringScan {
            string: scan.string.map(|_| bytes),
            inspected: scan.inspected,
        })
    }

    /// Counts bytes before a bounded string's first NUL without retained storage.
    ///
    /// Uses the same byte, lifetime and coordinate checks as
    /// [`Self::read_string`]. Unknown bytes or an absent terminator within
    /// `limit` leave no length.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::analysis::execution::MemoryRefusal`] for a visited
    /// byte outside a live object or a representable address span.
    pub fn string_length(
        &self,
        address: Address,
        limit: usize,
        image: Option<&dyn ImageData>,
    ) -> Result<StringScan<usize>, MemoryRefusal> {
        self.scan_string(address, limit, image, |_| true)
    }

    /// Visits known nonzero bytes and returns the first NUL's offset, with the
    /// bytes inspected.
    ///
    /// A visitor returning `false` stops the scan immediately, as does unknown
    /// data. The terminator is inspected but never passed to the visitor.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::analysis::execution::MemoryRefusal`] before reading
    /// a byte whose address, lifetime or extent cannot be established.
    fn scan_string(
        &self,
        address: Address,
        limit: usize,
        image: Option<&dyn ImageData>,
        mut accept: impl FnMut(u8) -> bool,
    ) -> Result<StringScan<usize>, MemoryRefusal> {
        for index in 0..limit {
            let offset = address
                .offset_at(index)
                .ok_or(MemoryRefusal::AddressRange)?;
            // The coordinate is already normalized; a single-byte admission
            // check cannot advance it into another target coordinate.
            if let Some(reason) = self.admit(
                Address::new(address.root(), offset, PointerSize::Bit64),
                1,
                Transfer::Read,
            ) {
                return Err(reason);
            }
            let inspected = index.saturating_add(1);
            match self.byte_at(address, index, image) {
                Some(0) => {
                    return Ok(StringScan {
                        string: Some(index),
                        inspected,
                    });
                }
                Some(byte) if accept(byte) => {}
                Some(_) | None => {
                    return Ok(StringScan {
                        string: None,
                        inspected,
                    });
                }
            }
        }
        Ok(StringScan {
            string: None,
            inspected: limit,
        })
    }

    /// The known bytes from `address` on, up to `limit` of them, stopping at
    /// the first byte that is not known.
    pub fn known_run(
        &self,
        address: Address,
        limit: usize,
        image: Option<&dyn ImageData>,
    ) -> Vec<u8> {
        (0..limit)
            .map_while(|index| self.byte_at(address, index, image))
            .collect()
    }

    /// Which of the `len` bytes at `address` the run wrote: those memory
    /// holds of its own, rather than the read-only image's unchanged
    /// contents.
    #[must_use]
    pub fn written(&self, address: Address, len: usize) -> WrittenRanges {
        WrittenRanges::of_offsets(
            (0..len).filter(|index| self.byte_at(address, *index, None).is_some()),
        )
    }

    /// Collects a known prefix without materializing a payload byte beyond the retained allowance.
    ///
    /// `None` means an additional known byte would exceed the allowance. Unknown
    /// memory or the requested end completes a shorter prefix successfully.
    pub fn known_run_within(
        &self,
        address: Address,
        limit: usize,
        maximum: usize,
        image: Option<&dyn ImageData>,
    ) -> Option<Vec<u8>> {
        let mut bytes = Vec::new();
        for index in 0..limit {
            let Some(byte) = self.byte_at(address, index, image) else {
                break;
            };
            if bytes.len() >= maximum {
                return None;
            }
            bytes.push(byte);
        }
        Some(bytes)
    }

    /// Shrinks an existing prefix to bytes still known at the same address.
    pub fn intersect_known_run(
        &self,
        address: Address,
        bytes: &mut Vec<u8>,
        image: Option<&dyn ImageData>,
    ) {
        let common = bytes
            .iter()
            .enumerate()
            .take_while(|(index, byte)| self.byte_at(address, *index, image) == Some(**byte))
            .count();
        bytes.truncate(common);
    }

    /// Returns linear runs under `root`, in unsigned-coordinate or signed-displacement order.
    ///
    /// Runs cross a host signed-integer boundary when unsigned addresses remain
    /// consecutive, and stop before address-space wrap.
    #[cfg(test)]
    fn held_runs(&self, root: Root) -> Vec<(i64, Vec<u8>)> {
        let mut runs: Vec<(i64, Vec<u8>)> = Vec::new();
        let mut next: Option<i64> = None;
        let negative = self.cells.range((root, i64::MIN)..=(root, -1));
        let nonnegative = self.cells.range((root, 0)..=(root, i64::MAX));
        let (first, second) = if root.unsigned_offsets() {
            (nonnegative, negative)
        } else {
            (negative, nonnegative)
        };
        for ((_, at), cell) in first.chain(second) {
            let Cell::Byte(byte) = cell else {
                next = None;
                continue;
            };
            match runs.last_mut() {
                Some((_, run)) if next == Some(at) => run.push(*byte),
                _ => runs.push((at, vec![*byte])),
            }
            next = if root.unsigned_offsets() {
                at.cast_unsigned().checked_add(1).map(u64::cast_signed)
            } else {
                at.checked_add(1)
            };
        }
        runs
    }

    /// The image bytes activation `since` and the activations it ran wrote,
    /// for an observation, without retaining private storage.
    pub fn image_contents(&self, since: u32) -> ImageContents {
        let writers = self.writers.owned_since(since);
        let mut cells = self.cells.for_root(Root::Symbol);
        let foreign: Vec<_> = cells
            .iter()
            .filter(|((_, at), _)| writers.writer(*at).is_none())
            .map(|(key, _)| key)
            .collect();
        for key in foreign {
            cells.remove(&key);
        }
        ImageContents { cells, writers }
    }

    /// Makes `site` of `activation` the writer of every store until the next
    /// call: the instruction about to execute.
    pub fn set_writer(&mut self, activation: u32, site: WriteSite) {
        self.current = CurrentWriter(Some(Writer {
            activation,
            site: Some(site),
        }));
    }

    /// Returns what both of two memories hold: the bytes they agree on.
    ///
    /// # Arguments
    ///
    /// * `other` - The memory on the other path.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        if self == other {
            return self.clone();
        }
        let mut objects = BTreeMap::new();
        for (root, object) in &self.objects {
            if let Some(other) = other.objects.get(root) {
                objects.insert(*root, object.join(other));
            }
        }
        let mut joined = Self {
            cells: self.cells.clone(),
            sources: self.sources.union(&other.sources),
            image_origins: self.image_origins.union(&other.image_origins),
            writers: self.writers.join(&other.writers),
            current: self.current,
            objects,
            disjoint_numeric: if self.disjoint_numeric == other.disjoint_numeric {
                self.disjoint_numeric.clone()
            } else {
                None
            },
            refusal: self.refusal.or(other.refusal),
            fault: self.fault.or(other.fault),
            stack_escaped: self.stack_escaped || other.stack_escaped,
            extents: self.extents.clone().or_else(|| other.extents.clone()),
            findings: self.findings.clone().or_else(|| other.findings.clone()),
            unwritten: self.unwritten.join(&other.unwritten),
            formed: self.formed.join(&other.formed),
            through: Through::default(),
            decisions: self.decisions.joined(),
        };
        // A shared leaf proves agreement of its explicit cells. Its extreme
        // coordinates additionally prove that every cell survives a heap meet.
        for (other_side, cells) in [(false, &self.cells), (true, &other.cells)] {
            for page in cells.page_keys() {
                if other_side && self.cells.contains_page(page) {
                    continue;
                }
                if self.cells.shares_page(&other.cells, page)
                    && self.cells.page_limits(page).is_some_and(|(first, last)| {
                        joined.accessible(page.0, first, 1) && joined.accessible(page.0, last, 1)
                    })
                {
                    continue;
                }
                for (other_side, source) in [(false, &self.cells), (true, &other.cells)] {
                    for key in source.keys_on_page(page) {
                        if other_side && self.cells.contains_key(&key) {
                            continue;
                        }
                        let (root, offset) = key;
                        let cell = match (self.cell(root, offset), other.cell(root, offset)) {
                            (Some(left), Some(right))
                                if left == right && joined.accessible(root, offset, 1) =>
                            {
                                Some(left)
                            }
                            (left, right)
                                if (left.is_some_and(Cell::may_reference_stack)
                                    || right.is_some_and(Cell::may_reference_stack))
                                    && joined.accessible(root, offset, 1) =>
                            {
                                Some(Cell::StackBits)
                            }
                            _ => None,
                        };
                        if let Some(cell) = cell {
                            if joined.cells.get(&key) != Some(&cell) {
                                joined.cells.insert(key, cell);
                            }
                        } else {
                            joined.cells.remove(&key);
                            joined.forget_backing(root, offset, 1);
                        }
                    }
                }
            }
        }

        joined
    }

    /// Returns whether every byte belongs exclusively to this execution.
    ///
    /// The complete span must remain linear and private; checking only the
    /// first byte would admit an access reaching caller-owned storage.
    pub fn private_span(&self, address: Address, bytes: usize) -> bool {
        let Some(last) = bytes
            .checked_sub(1)
            .and_then(|last| i64::try_from(last).ok())
        else {
            return false;
        };
        self.holds_privately(address)
            && address
                .moved(last, address.pointer_size())
                .is_some_and(|end| {
                    end.follows(address, bytes.saturating_sub(1)) && self.holds_privately(end)
                })
    }

    /// Retains cells shared by an unchanged span and one possible replacement.
    ///
    /// The candidate contains only the bounded write. No unrelated memory is
    /// copied or compared; each affected cell is compared exactly once.
    pub fn intersect_span(&mut self, candidate: &Self, address: Address, bytes: usize) {
        if !self.admit_write(address, bytes) {
            return;
        }
        self.forget_aliases(address, bytes);
        for index in 0..bytes {
            let Some(at) = address.offset_at(index) else {
                return;
            };
            if self.cell(address.root(), at) != candidate.cell(address.root(), at) {
                self.cells.remove(&(address.root(), at));
            }
        }
        self.stack_escaped |= candidate.stack_escaped;
    }

    /// Whether the byte at `address` is private to the execution.
    pub const fn holds_privately(&self, address: Address) -> bool {
        self.is_private(address.root(), address.offset())
    }

    /// Marks the stack escaped when `value` is an address into it that is
    /// about to be written where the execution cannot follow it.
    pub fn escape_if_stack<T: Target>(&mut self, value: &Value<T>) {
        if value.may_reference_stack() {
            self.escape_stack();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell as Counter, iter};

    use super::*;
    use crate::{
        analysis::{
            execution::{InstructionPermissions, findings::Witness},
            image::ImageRegion,
        },
        ir::variable::SsaVarId,
        pointer::PointerSize,
        target::Endianness,
    };

    /// **Read-only image a path made writable is written, and served only
    /// while unwritten.** A write into read-only image is refused, until a
    /// protection change installs write permission over it; then a written
    /// byte reads as written and an unwritten one as the image binds it, and
    /// once read-only again it refuses writes anew. After a write nobody can
    /// place, the bytes a change covered are no longer the image's. The
    /// control: read-only image no change covered is served after such a
    /// write as before.
    #[test]
    fn read_only_image_a_path_made_writable_is_served_only_while_unwritten() {
        let at = |offset| Address::new(Root::Symbol, offset, PointerSize::Bit64);
        let byte = |memory: &Memory, offset| match memory.load(at(offset), 1, Some(&Image)) {
            Loaded::Bytes(bytes) => bytes.first().copied(),
            _ => None,
        };
        let mut memory = Memory::default();
        assert_eq!(
            memory.write_refusal(at(0x1001), 1, Some(&Image)),
            Some(MemoryRefusal::ReadOnlyImage)
        );
        let mut clobbered = memory.clone();
        clobbered.clobber();
        assert_eq!(byte(&clobbered, 0x1000), Some(0x10));

        memory.instruction_sources_mut().set_permissions(
            at(0x1000),
            4,
            InstructionPermissions {
                writable: true,
                executable: false,
            },
        );
        assert_eq!(memory.write_refusal(at(0x1001), 1, Some(&Image)), None);
        memory.store_bytes(at(0x1001), &[0x55]);
        assert_eq!(byte(&memory, 0x1001), Some(0x55));
        assert_eq!(byte(&memory, 0x1000), Some(0x10));
        memory.instruction_sources_mut().set_permissions(
            at(0x1000),
            4,
            InstructionPermissions {
                writable: false,
                executable: true,
            },
        );
        assert_eq!(
            memory.write_refusal(at(0x1001), 1, Some(&Image)),
            Some(MemoryRefusal::ReadOnlyImage)
        );
        assert_eq!(byte(&memory, 0x1001), Some(0x55));
        memory.clobber();
        assert_eq!(byte(&memory, 0x1000), None);
    }

    /// How the lifetime of the allocation `root` stands in `memory`, and what
    /// ended it where it is not live; `None` where no path made it.
    fn lifetime(memory: &Memory, root: Root) -> Option<&Lifetime> {
        memory.objects.get(&root).map(|object| &object.lifetime)
    }

    /// Read-only bytes `0x10..0x14` at `0x1000`; a writable region at `0x2000`.
    struct Image;

    impl ImageData for Image {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, address: u64) -> Option<ImageRegion> {
            let (start, mutability) = match address {
                0x1000..0x1004 => (0x1000, Mutability::ReadOnly),
                0x2000..0x2004 => (0x2000, Mutability::Writable),
                _ => return None,
            };
            Some(ImageRegion {
                start,
                end: start + 4,
                mutability,
            })
        }

        fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
            const BYTES: [u8; 4] = [0x10, 0x11, 0x12, 0x13];
            let start = usize::try_from(address.checked_sub(0x1000)?).ok()?;
            BYTES.get(start..start.checked_add(len)?)
        }
    }

    fn at(root: Root, offset: i64) -> Address {
        Address::new(root, offset, PointerSize::Bit64)
    }

    fn opaque(index: usize) -> Root {
        Root::Opaque {
            activation: 0,
            value: SsaVarId::from_index(index),
            run: 0,
        }
    }

    /// A never-written image byte reads from the image only where it is
    /// read-only.
    #[test]
    fn only_read_only_image_bytes_read_through() {
        let memory = Memory::default();
        assert_eq!(
            memory.load(at(Root::Symbol, 0x1001), 2, Some(&Image)),
            Loaded::Bytes(vec![0x11, 0x12])
        );
        assert_eq!(
            memory.load(at(Root::Symbol, 0x2000), 1, Some(&Image)),
            Loaded::Unknown
        );
        assert_eq!(
            memory.load(at(Root::Symbol, 0x1000), 1, None),
            Loaded::Unknown
        );
    }

    /// A stored address reads back whole, and only whole.
    #[test]
    fn an_address_reads_back_only_whole() {
        let mut memory = Memory::default();
        let slot = at(Root::Stack, -16);
        let buffer = at(Root::Stack, -48);
        memory.store_address(slot, buffer, 8);
        assert_eq!(memory.load(slot, 8, None), Loaded::Address(buffer));
        assert_eq!(memory.load(slot, 4, None), Loaded::StackDerived);
        assert_eq!(
            memory.load(slot.moved(1, PointerSize::Bit64).unwrap_or(slot), 8, None),
            Loaded::StackDerived
        );
    }

    /// A store through an opaque pointer forgets the image bytes the execution
    /// wrote, but never the private stack.
    #[test]
    fn an_opaque_store_spares_only_the_private_stack() {
        let mut memory = Memory::default();
        memory.store_bytes(at(Root::Symbol, 0x2000), &[1, 2]);
        memory.store_bytes(at(Root::Stack, -8), &[3]);
        memory.store_bytes(at(Root::Stack, 8), &[4]);
        memory.store_bytes(at(opaque(1), 0), &[0]);
        assert_eq!(
            memory.load(at(Root::Symbol, 0x2000), 1, Some(&Image)),
            Loaded::Unknown
        );
        assert_eq!(
            memory.load(at(Root::Stack, -8), 1, None),
            Loaded::Bytes(vec![3])
        );
        assert_eq!(memory.load(at(Root::Stack, 8), 1, None), Loaded::Unknown);
    }

    /// Once the stack escapes, an opaque store may reach all of it.
    #[test]
    fn an_escaped_stack_is_reachable() {
        let mut memory = Memory::default();
        memory.store_bytes(at(Root::Stack, -8), &[3]);
        memory.escape_stack();
        memory.store_bytes(at(opaque(1), 0), &[0]);
        assert_eq!(memory.load(at(Root::Stack, -8), 1, None), Loaded::Unknown);
    }

    /// Two opaque roots may name the same memory; one root's other offsets
    /// are other bytes.
    #[test]
    fn opaque_roots_alias_each_other_but_not_themselves() {
        let mut memory = Memory::default();
        memory.store_bytes(at(opaque(1), 0), &[7]);
        memory.store_bytes(at(opaque(1), 1), &[8]);
        assert_eq!(
            memory.load(at(opaque(1), 0), 1, None),
            Loaded::Bytes(vec![7])
        );
        memory.store_bytes(at(opaque(2), 0), &[9]);
        assert_eq!(memory.load(at(opaque(1), 0), 1, None), Loaded::Unknown);
    }

    /// A store to the image does not reach the private stack or a heap block.
    #[test]
    fn disjoint_spaces_do_not_alias() {
        let mut memory = Memory::default();
        memory.store_bytes(at(Root::Stack, -4), &[1]);
        memory.allocate(Root::Heap(0), Some(1), None, None);
        memory.store_bytes(at(Root::Heap(0), 0), &[2]);
        memory.store_bytes(at(Root::Symbol, 0x2000), &[3]);
        assert_eq!(
            memory.load(at(Root::Stack, -4), 1, None),
            Loaded::Bytes(vec![1])
        );
        assert_eq!(
            memory.load(at(Root::Heap(0), 0), 1, None),
            Loaded::Bytes(vec![2])
        );
    }

    /// Repeated exit observations inspect retained image pages and never absorb later growth.
    #[test]
    fn exit_image_intersection_is_monotone_and_excludes_private_storage() {
        let mut memory = Memory::default();
        memory.set_writer(
            1,
            WriteSite {
                body: 0,
                block: 0,
                instruction: 0,
            },
        );
        memory.store_bytes(at(Root::Stack, -8), &[3]);
        memory.store_bytes(at(Root::Symbol, 0x2000), &[7]);
        let mut image = memory.image_contents(0);
        assert_eq!(image.cells.iter().count(), 1);
        for index in 1..1024 {
            memory.store_bytes(at(Root::Symbol, 0x2000 + index * 128), &[9]);
        }
        image.intersect(&memory, 0);
        assert_eq!(image.cells.iter().count(), 1);
        assert_eq!(image.runs(1).unwrap()[0].bytes, vec![7]);
        memory.store_bytes(at(Root::Symbol, 0x2000), &[8]);
        image.intersect(&memory, 0);
        assert!(image.runs(0).unwrap().is_empty());
        memory.store_bytes(at(Root::Symbol, 0x2000), &[7]);
        image.intersect(&memory, 0);
        assert!(image.runs(0).unwrap().is_empty());
    }

    /// **An observation keeps its own activation's writes, one run per
    /// writing activation.** Bytes a caller wrote before the call are not the
    /// callee's; a later call's write over the start of an earlier call's
    /// leaves two runs, not one value no invocation wrote.
    #[test]
    fn exit_contents_keep_own_writes_split_by_writer() {
        let site = |instruction| WriteSite {
            body: 0,
            block: 0,
            instruction,
        };
        let mut memory = Memory::default();
        memory.set_writer(1, site(1));
        memory.store_bytes(at(Root::Symbol, 0x1000), b"caller");
        memory.set_writer(2, site(2));
        memory.store_bytes(at(Root::Symbol, 0x2000), b"0123456789");
        memory.set_writer(3, site(3));
        memory.store_bytes(at(Root::Symbol, 0x2000), b"abcdefgh");
        let own = memory.image_contents(2);
        let runs = own.runs(usize::MAX).unwrap_or_default();
        assert_eq!(
            runs,
            vec![
                ImageWrite {
                    address: 0x2000,
                    bytes: b"abcdefgh".to_vec(),
                    sites: vec![site(3)],
                },
                ImageWrite {
                    address: 0x2008,
                    bytes: b"89".to_vec(),
                    sites: vec![site(2)],
                },
            ]
        );
        // One activation's two stores are one run that names both.
        memory.set_writer(3, site(4));
        memory.store_bytes(at(Root::Symbol, 0x2008), b"ij");
        assert_eq!(
            memory
                .image_contents(3)
                .runs(usize::MAX)
                .unwrap_or_default(),
            vec![ImageWrite {
                address: 0x2000,
                bytes: b"abcdefghij".to_vec(),
                sites: vec![site(3), site(4)],
            }]
        );
        let whole = memory
            .image_contents(0)
            .runs(usize::MAX)
            .unwrap_or_default();
        assert_eq!(
            whole.first().map(|run| run.bytes.as_slice()),
            Some(&b"caller"[..])
        );
    }

    /// Paths agree only on bytes both hold alike.
    #[test]
    fn join_keeps_only_agreement() {
        let mut left = Memory::default();
        let mut right = Memory::default();
        left.store_bytes(at(Root::Stack, -2), &[1, 2]);
        right.store_bytes(at(Root::Stack, -2), &[1, 5]);
        let joined = left.join(&right);
        assert_eq!(
            joined.load(at(Root::Stack, -2), 1, None),
            Loaded::Bytes(vec![1])
        );
        assert_eq!(joined.load(at(Root::Stack, -1), 1, None), Loaded::Unknown);
    }

    /// A returned activation's frame is dead; its caller's side is not.
    #[test]
    fn release_forgets_only_below_the_entry() {
        let mut memory = Memory::default();
        memory.store_bytes(at(Root::Stack, -40), &[1]);
        memory.store_bytes(at(Root::Stack, -8), &[2]);
        memory.release_below(-16);
        assert_eq!(memory.load(at(Root::Stack, -40), 1, None), Loaded::Unknown);
        assert_eq!(
            memory.load(at(Root::Stack, -8), 1, None),
            Loaded::Bytes(vec![2])
        );
    }

    /// A run stops at the first byte that is not known, and held runs split
    /// where bytes are missing.
    #[test]
    fn runs_stop_at_the_unknown() {
        let mut memory = Memory::default();
        memory.store_bytes(at(Root::Symbol, 0x3000), b"ab");
        memory.store_bytes(at(Root::Symbol, 0x3003), b"c");
        assert_eq!(
            memory.known_run(at(Root::Symbol, 0x3000), 8, None),
            b"ab".to_vec()
        );
        assert_eq!(
            memory.held_runs(Root::Symbol),
            vec![(0x3000, b"ab".to_vec()), (0x3003, b"c".to_vec())]
        );
    }
    /// Detached objects neither overlap each other nor delete their caller's bytes.
    #[test]
    fn detached_stack_lifetimes_and_escape_are_explicit() {
        let mut memory = Memory::default();
        let caller = at(Root::Stack, -(1i64 << 33));
        let first = at(Root::StackFrame(1), -8);
        let second = at(Root::StackFrame(2), -8);
        memory.store_bytes(caller, &[1]);
        memory.store_bytes(first, &[2]);
        memory.store_bytes(second, &[3]);
        memory.release_activation(Root::StackFrame(2), 0, 2);
        assert_eq!(memory.known_run(caller, 1, None), [1]);
        assert_eq!(memory.known_run(first, 1, None), [2]);
        assert!(memory.known_run(second, 1, None).is_empty());
        memory.clobber();
        assert_eq!(memory.known_run(first, 1, None), [2]);
        memory.escape_stack();
        memory.clobber();
        assert!(memory.known_run(first, 1, None).is_empty());
    }
    /// **A realigned frame lives and dies with the stack it was realigned
    /// from.** Its base lies in `at - 15 ..= at` for `and sp, -16`. A call
    /// with no known effect keeps what lies wholly below the entry pointer
    /// and forgets what may lie at or above it; an activation's return
    /// releases what lies wholly below its entry and keeps what may lie above;
    /// a detached frame's realignment dies with the frame; and once the stack
    /// escapes, a call forgets all of it.
    #[test]
    fn a_realigned_frame_follows_its_stack_lifetimes() {
        let realigned = |frame, at| Root::StackAligned { frame, at, bits: 4 };
        // Base in -23..=-8: byte -8 lies in -31..=-16, byte 20 in -3..=12.
        let below = at(realigned(None, -8), -8);
        let straddling = at(realigned(None, -8), 20);
        let mut memory = Memory::default();
        memory.store_bytes(below, &[1]);
        memory.store_bytes(straddling, &[2]);
        memory.clobber();
        assert_eq!(memory.known_run(below, 1, None), [1]);
        assert!(memory.known_run(straddling, 1, None).is_empty());

        // Returning from an activation entered at -40: byte -40 lies in
        // -63..=-48, below it; byte -24 in -47..=-32, which may be above.
        let dead = at(realigned(None, -8), -40);
        let live = at(realigned(None, -8), -24);
        memory.store_bytes(dead, &[3]);
        memory.store_bytes(live, &[4]);
        memory.release_below(-40);
        assert!(memory.known_run(dead, 1, None).is_empty());
        assert_eq!(memory.known_run(live, 1, None), [4]);

        let detached = at(realigned(Some(2), -8), -8);
        let kept = at(realigned(Some(1), -8), -8);
        memory.store_bytes(detached, &[5]);
        memory.store_bytes(kept, &[6]);
        memory.release_activation(Root::StackFrame(2), 0, 2);
        assert!(memory.known_run(detached, 1, None).is_empty());
        assert_eq!(memory.known_run(kept, 1, None), [6]);
        memory.clobber();
        assert_eq!(memory.known_run(kept, 1, None), [6]);
        memory.escape_stack();
        memory.clobber();
        assert!(memory.known_run(kept, 1, None).is_empty());
        assert!(memory.known_run(below, 1, None).is_empty());
    }

    /// A multi-byte access advances in the target space, including its wrap.
    #[test]
    fn absolute_memory_spans_keep_the_target_pointer_width() {
        for (width, start, coordinates) in [
            (PointerSize::Bit8, 254, [254, 255, 0, 1]),
            (PointerSize::Bit16, 65534, [65534, 65535, 0, 1]),
            (
                PointerSize::Bit32,
                4294967294,
                [4294967294, 4294967295, 0, 1],
            ),
            (
                PointerSize::Bit64,
                i64::MAX - 1,
                [i64::MAX - 1, i64::MAX, i64::MIN, i64::MIN + 1],
            ),
            (PointerSize::Bit64, -2, [-2, -1, 0, 1]),
        ] {
            let mut memory = Memory::default();
            let start = Address::new(Root::Symbol, start, width);
            memory.store_bytes(start, &[1, 2, 3, 4]);
            assert_eq!(memory.load(start, 4, None), Loaded::Bytes(vec![1, 2, 3, 4]));
            for (offset, expected) in coordinates.into_iter().zip([1, 2, 3, 4]) {
                assert_eq!(
                    memory.load(Address::new(Root::Symbol, offset, width), 1, None),
                    Loaded::Bytes(vec![expected])
                );
            }
            memory.store_unknown(start, 4);
            for offset in coordinates {
                assert_eq!(
                    memory.load(Address::new(Root::Symbol, offset, width), 1, None),
                    Loaded::Unknown
                );
            }
        }
    }

    /// Pointer fragments survive storage and copying across a target wrap.
    #[test]
    fn symbolic_pointer_storage_can_cross_the_target_wrap() {
        let mut memory = Memory::default();
        let source = Address::new(Root::Symbol, 0xffff_fffc, PointerSize::Bit32);
        let dest = Address::new(Root::Symbol, 64, PointerSize::Bit32);
        let pointer = Address::of_root(Root::Heap(9));
        memory.store_address(source, pointer, 8);
        assert_eq!(memory.load(source, 8, None), Loaded::Address(pointer));
        memory.copy(dest, source, 8, None, true);
        assert_eq!(memory.load(dest, 8, None), Loaded::Address(pointer));
    }

    /// Zero-length writes have no aliases or bytes to invalidate.
    #[test]
    fn zero_length_writes_preserve_other_address_roots() {
        let mut memory = Memory::default();
        let symbol = at(Root::Symbol, 0x1000);
        let number = at(Root::Number, 0x2000);
        memory.store_bytes(symbol, &[17]);
        memory.store_unknown(number, 0);
        memory.store_bytes(number, &[]);
        memory.store_address(number, Address::of_root(Root::Heap(1)), 0);
        assert_eq!(memory.load(symbol, 1, None), Loaded::Bytes(vec![17]));
    }

    /// Protected image storage is checked after modular wrap and across a complete address space.
    #[test]
    fn image_write_protection_covers_wrapped_and_complete_target_spans() {
        struct Tiny;
        impl ImageData for Tiny {
            fn endianness(&self) -> Endianness {
                Endianness::Little
            }
            fn region(&self, at: u64) -> Option<ImageRegion> {
                (0x10..0x20).contains(&at).then_some(ImageRegion {
                    start: 0x10,
                    end: 0x20,
                    mutability: Mutability::ReadOnly,
                })
            }
            fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
                None
            }
        }
        let memory = Memory::default();
        let at = Address::new(Root::Symbol, 0xfc, PointerSize::Bit8);
        assert_eq!(memory.write_refusal(at, 20, Some(&Tiny)), None);
        for len in [21, 256, 1024] {
            assert_eq!(
                memory.write_refusal(at, len, Some(&Tiny)),
                Some(MemoryRefusal::ReadOnlyImage)
            );
        }
        assert_eq!(memory.write_refusal(at, 0, Some(&Tiny)), None);
        assert_eq!(memory.write_refusal(at, 21, None), None);
    }

    /// Erasing an entire small target space includes every coordinate once.
    #[test]
    fn unknown_spans_can_cover_an_entire_address_space() {
        let mut memory = Memory::default();
        for offset in [0, 17, 200, 255] {
            memory.store_bytes(Address::new(Root::Symbol, offset, PointerSize::Bit8), &[3]);
        }
        memory.store_unknown(Address::new(Root::Symbol, 200, PointerSize::Bit8), 256);
        for offset in [0, 17, 200, 255] {
            assert_eq!(
                memory.load(
                    Address::new(Root::Symbol, offset, PointerSize::Bit8),
                    1,
                    None
                ),
                Loaded::Unknown
            );
        }
        let relative = at(Root::Stack, i64::MAX);
        memory.store_bytes(relative, &[4]);
        memory.store_unknown(relative, 1);
        assert_eq!(memory.load(relative, 1, None), Loaded::Unknown);
    }

    /// An unrepresentable wider address must not alias byte zero.
    #[test]
    fn a_wider_pointer_span_does_not_wrap_at_sixty_four_bits() {
        let mut memory = Memory::default();
        let high = Address::new(Root::Number, -1, PointerSize::Bit128);
        let zero = Address::new(Root::Number, 0, PointerSize::Bit128);
        memory.store_bytes(zero, &[9]);
        memory.store_bytes(high, &[1, 2]);
        assert_eq!(memory.load(high, 2, None), Loaded::Unknown);
        assert_eq!(memory.load(zero, 1, None), Loaded::Bytes(vec![9]));
    }

    /// Huge initialized objects consume metadata until individual bytes are written.
    #[test]
    fn lazy_allocations_track_unknown_intervals_against_a_byte_oracle() {
        let mut memory = Memory::default();
        let root = Root::Heap(0);
        memory.allocate(root, Some(1 << 40), Some(0), None);
        assert_eq!(memory.cells.range(..).count(), 0);
        assert_eq!(
            memory
                .read_string(at(root, 0), 1 << 20, None)
                .map(|scan| scan.string),
            Ok(Some(vec![]))
        );
        let mut expected = [Some(0); 32];
        for (start, len) in [(4, 4), (8, 4), (2, 3), (19, 5), (16, 8)] {
            memory.store_unknown(at(root, start as i64), len);
            expected[start..start + len].fill(None);
            for (index, expected) in expected.iter().enumerate() {
                assert_eq!(memory.byte_at(at(root, index as i64), 0, None), *expected);
            }
        }
        memory.store_bytes(at(root, 7), &[17, 19]);
        expected[7..9].copy_from_slice(&[Some(17), Some(19)]);
        for (index, expected) in expected.iter().enumerate() {
            assert_eq!(memory.byte_at(at(root, index as i64), 0, None), *expected);
        }
        assert_eq!(memory.cells.range(..).count(), 2);
        memory.store_bytes(at(opaque(17), 0), &[1]);
        assert_eq!(memory.load(at(root, 0), 1, None), Loaded::Unknown);
        assert_eq!(memory.access_refusal(at(root, 0), 32), None);
    }

    /// A joined default cannot reveal bytes that disagreed or became unknown;
    /// the bytes known to belong to the object are those every path holds,
    /// and an access is past it only past what any path holds.
    #[test]
    fn object_joins_intersect_contents_extents_and_lifetimes() {
        let root = Root::Heap(0);
        let mut left = Memory::default();
        left.allocate(root, Some(16), Some(0), None);
        let mut right = left.clone();
        left.store_bytes(at(root, 1), &[1]);
        right.store_bytes(at(root, 2), &[2]);
        left.store_unknown(at(root, 5), 2);
        left.store_bytes(at(root, 6), &[9]);
        right.store_bytes(at(root, 6), &[9]);
        let joined = left.join(&right);
        for (index, expected) in [
            Some(0),
            None,
            None,
            Some(0),
            Some(0),
            None,
            Some(9),
            Some(0),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(joined.byte_at(at(root, index as i64), 0, None), expected);
        }
        assert_eq!(joined, right.join(&left));
        right.allocate(root, Some(8), Some(0), None);
        assert_eq!(left.join(&right).byte_at(at(root, 8), 0, None), None);
        assert_eq!(left.join(&right).access_finding(at(root, 8), 1), None);
        assert_eq!(
            left.join(&right).access_finding(at(root, 16), 1),
            Some(FindingKind::PastExtent(ExtentSource::Allocator))
        );
        assert_eq!(right.free(at(root, 0), None, None), FreeOutcome::Released);
        assert_eq!(
            left.join(&right).access_finding(at(root, 0), 1),
            Some(FindingKind::Ended(Ending::Some))
        );
    }

    /// **A block whose size no allocator stated has no extent to be past.**
    /// An allocation of unstated size holds no byte known to belong to it,
    /// so a read of it is unknown, and no access is found past it; joined
    /// with a path that states eight bytes, it still has none. The control: a
    /// block of eight stated bytes is past at its ninth.
    #[test]
    fn a_block_of_unstated_size_has_no_extent_to_be_past() {
        let root = Root::Heap(0);
        let mut unstated = Memory::default();
        unstated.allocate(root, None, Some(0), None);
        assert_eq!(unstated.load(at(root, 0), 1, None), Loaded::Unknown);
        assert_eq!(unstated.access_finding(at(root, 8), 1), None);
        let mut stated = Memory::default();
        stated.allocate(root, Some(8), Some(0), None);
        assert_eq!(
            stated.access_finding(at(root, 8), 1),
            Some(FindingKind::PastExtent(ExtentSource::Allocator)),
            "the control: past a stated size is past the block"
        );
        for joined in [unstated.join(&stated), stated.join(&unstated)] {
            assert_eq!(joined.access_finding(at(root, 8), 1), None);
            assert_eq!(joined.load(at(root, 0), 1, None), Loaded::Unknown);
        }
    }

    /// **An object keeps the family every path agrees made it, and only a
    /// deallocator of that family ends it.** Joined with the same family the
    /// object keeps it; joined with another, it has none a stated
    /// deallocator ends, while one stating no family still ends it.
    #[test]
    fn an_object_keeps_the_family_its_paths_agree_on() {
        let root = Root::Heap(0);
        let (libc, heap) = (AllocatorFamily(1), AllocatorFamily(2));
        let mut left = Memory::default();
        left.allocate(root, Some(8), None, Some(libc));
        let same = left.join(&left.clone());
        assert_eq!(same.object(root), Some((8, Some(libc))));
        let mut right = Memory::default();
        right.allocate(root, Some(8), None, Some(heap));
        let mut joined = left.join(&right);
        assert_eq!(joined.object(root), Some((8, None)));
        assert_eq!(
            joined.free(at(root, 0), Some(libc), None),
            FreeOutcome::OtherFamily { made_by: None }
        );
        assert_eq!(joined.free(at(root, 0), None, None), FreeOutcome::Released);
        assert_eq!(
            left.free(at(root, 0), Some(heap), None),
            FreeOutcome::OtherFamily {
                made_by: Some(libc)
            }
        );
        assert_eq!(
            left.free(at(root, 0), Some(libc), None),
            FreeOutcome::Released
        );
        assert_eq!(left.object(root), None);
    }

    /// **A byte reads never written until a write may have reached it.**
    /// `malloc`'s bytes are never written and `calloc`'s are zero; a frame's
    /// bytes below its entry are never written until stored, and after an
    /// unknown call that received the stack they may have been written; a
    /// returned activation's frame is fresh for the next one.
    #[test]
    fn a_byte_reads_never_written_until_a_write_may_reach_it() {
        let heap = Root::Heap(0);
        let mut malloc = Memory::default();
        malloc.allocate(heap, Some(4), None, None);
        assert_eq!(malloc.unwritten_bits(at(heap, 0), 4), Some(vec![0xff; 4]));
        malloc.store_bytes(at(heap, 1), &[7]);
        assert_eq!(
            malloc.unwritten_bits(at(heap, 0), 4),
            Some(vec![0xff, 0, 0xff, 0xff])
        );
        let mut calloc = Memory::default();
        calloc.allocate(heap, Some(4), Some(0), None);
        assert_eq!(calloc.unwritten_bits(at(heap, 0), 4), None);

        let mut frame = Memory::default();
        frame.begin_frame(Root::Stack, 0);
        assert_eq!(
            frame.unwritten_bits(at(Root::Stack, -8), 8),
            Some(vec![0xff; 8])
        );
        assert_eq!(
            frame.unwritten_bits(at(Root::Stack, 0), 8),
            None,
            "the caller's"
        );
        let mut stored = frame.clone();
        stored.store_bytes(at(Root::Stack, -8), &[1; 8]);
        assert_eq!(stored.unwritten_bits(at(Root::Stack, -8), 8), None);
        let mut kept = frame.clone();
        kept.clobber();
        assert_eq!(
            kept.unwritten_bits(at(Root::Stack, -8), 8),
            Some(vec![0xff; 8]),
            "an unknown call cannot reach the private stack"
        );
        let mut escaped = frame.clone();
        escaped.escape_stack();
        escaped.clobber();
        assert_eq!(escaped.unwritten_bits(at(Root::Stack, -8), 8), None);
        let mut aliased = frame.clone();
        aliased.escape_stack();
        aliased.store_unknown(at(Root::Number, 0x5000), 4);
        assert_eq!(aliased.unwritten_bits(at(Root::Stack, -8), 8), None);

        let mut returned = stored.clone();
        returned.store_bytes(at(Root::Stack, -24), &[2; 8]);
        returned.release_below(-16);
        assert_eq!(
            returned.unwritten_bits(at(Root::Stack, -24), 8),
            Some(vec![0xff; 8])
        );
        assert_eq!(returned.unwritten_bits(at(Root::Stack, -8), 8), None);

        assert_eq!(
            frame.join(&stored).unwritten_bits(at(Root::Stack, -8), 8),
            None,
            "written on one path"
        );
        assert_eq!(
            frame
                .join(&frame.clone())
                .unwritten_bits(at(Root::Stack, -8), 8),
            Some(vec![0xff; 8])
        );
    }

    /// **A copy carries what was never written.** Copying never-written
    /// bytes and stored bytes into a frame leaves the copied never-written
    /// bytes never written and the copied stored ones written; a value
    /// stored with never-written bits keeps exactly those.
    #[test]
    fn a_copy_carries_what_was_never_written() {
        let mut memory = Memory::default();
        memory.begin_frame(Root::Stack, 0);
        memory.store_bytes(at(Root::Stack, -16), &[1, 2]);
        memory.carry_unwritten(at(Root::Stack, -16), &[0, 0x0f]);
        assert!(
            !memory.copy(at(Root::Stack, -32), at(Root::Stack, -16), 4, None, true),
            "a never-written byte backs no read"
        );
        assert_eq!(
            memory.unwritten_bits(at(Root::Stack, -32), 4),
            Some(vec![0, 0x0f, 0xff, 0xff])
        );
        memory.store_bytes(at(Root::Stack, -31), &[3]);
        assert_eq!(
            memory.unwritten_bits(at(Root::Stack, -32), 4),
            Some(vec![0, 0, 0xff, 0xff])
        );
    }

    /// **A lifetime joins; it is never replaced.** A free on one arm joined
    /// with a live arm is ended on some path, with the free's site; freed on
    /// both arms it is ended, with both sites; an unknown effect makes a live
    /// object possibly ended, not ended, and leaves an ended one ended. Every
    /// state but live refuses an access.
    #[test]
    fn a_lifetime_joins_what_ended_it() {
        let root = Root::Heap(0);
        let live = || {
            let mut memory = Memory::default();
            memory.allocate(root, Some(8), None, None);
            memory
        };
        let freed = |site| {
            let mut memory = live();
            assert_eq!(
                memory.free(at(root, 0), None, Some(site)),
                FreeOutcome::Released
            );
            memory
        };
        let sites = |list: &[(usize, usize)]| list.iter().copied().collect::<BTreeSet<_>>();

        let one_arm = freed((1, 2)).join(&live());
        assert_eq!(
            lifetime(&one_arm, root),
            Some(&Lifetime::EndedOnSome {
                sites: sites(&[(1, 2)])
            })
        );
        assert_eq!(one_arm, live().join(&freed((1, 2))));
        let both = freed((1, 2)).join(&freed((3, 0)));
        assert_eq!(
            lifetime(&both, root),
            Some(&Lifetime::Ended {
                sites: sites(&[(1, 2), (3, 0)])
            })
        );
        assert_eq!(lifetime(&live().join(&live()), root), Some(&Lifetime::Live));

        let mut unknown = live();
        unknown.invalidate_lifetimes();
        assert_eq!(
            lifetime(&unknown, root),
            Some(&Lifetime::PossiblyEnded { sites: sites(&[]) })
        );
        let mut ended = freed((1, 2));
        ended.invalidate_lifetimes();
        assert_eq!(
            lifetime(&ended, root),
            Some(&Lifetime::Ended {
                sites: sites(&[(1, 2)])
            })
        );
        let mut some = one_arm.clone();
        some.invalidate_lifetimes();
        assert_eq!(
            lifetime(&some, root),
            Some(&Lifetime::PossiblyEnded {
                sites: sites(&[(1, 2)])
            })
        );
        assert_eq!(
            lifetime(&unknown.join(&freed((1, 2))), root),
            Some(&Lifetime::PossiblyEnded {
                sites: sites(&[(1, 2)])
            })
        );

        for memory in [&one_arm, &both, &unknown, &ended, &some] {
            assert!(matches!(
                memory.access_finding(at(root, 0), 1),
                Some(FindingKind::Ended(_))
            ));
            assert_eq!(memory.access_refusal(at(root, 0), 1), None);
            assert_eq!(memory.object(root), None);
        }
        assert_eq!(live().access_finding(at(root, 0), 1), None);
    }

    /// **A deallocation says what its address named.** Each outcome of
    /// [`Memory::free`]: the start of a live object of the family is
    /// released; an ended, ended-on-some or possibly-ended object reports its
    /// lifetime; an interior pointer its offset; the stack and the image are
    /// no heap object; another family's object names the family that made
    /// it; a number and an allocation no path made are
    /// unknown. Only the release changes anything.
    #[test]
    fn a_free_is_classified_by_what_its_address_names() {
        let (libc, heap) = (AllocatorFamily(1), AllocatorFamily(2));
        let root = Root::Heap(0);
        let live = || {
            let mut memory = Memory::default();
            memory.allocate(root, Some(8), Some(0), Some(libc));
            memory
        };
        let unchanged = |address: Address, family, expected: FreeOutcome| {
            let mut memory = live();
            let before = memory.clone();
            assert_eq!(memory.free(address, family, Some((5, 1))), expected);
            assert_eq!(memory, before, "{expected:?} changes nothing");
        };

        let mut released = live();
        assert_eq!(
            released.free(at(root, 0), Some(libc), Some((5, 1))),
            FreeOutcome::Released
        );
        assert_eq!(
            released.free(at(root, 0), Some(libc), Some((6, 0))),
            FreeOutcome::Ended {
                lifetime: Lifetime::Ended {
                    sites: BTreeSet::from([(5, 1)])
                }
            }
        );
        let mut possibly = live();
        possibly.invalidate_lifetimes();
        assert_eq!(
            possibly.free(at(root, 0), Some(libc), None),
            FreeOutcome::Ended {
                lifetime: Lifetime::PossiblyEnded {
                    sites: BTreeSet::new()
                }
            }
        );
        let mut some = released.join(&live());
        assert_eq!(
            some.free(at(root, 0), Some(libc), None),
            FreeOutcome::Ended {
                lifetime: Lifetime::EndedOnSome {
                    sites: BTreeSet::from([(5, 1)])
                }
            }
        );

        unchanged(at(root, 3), Some(libc), FreeOutcome::Interior { offset: 3 });
        unchanged(
            at(root, 0),
            Some(heap),
            FreeOutcome::OtherFamily {
                made_by: Some(libc),
            },
        );
        for not_heap in [Root::Stack, Root::StackFrame(1), Root::Symbol] {
            unchanged(at(not_heap, -16), Some(libc), FreeOutcome::NotHeap);
        }
        unchanged(at(Root::Number, 0x1000), Some(libc), FreeOutcome::Unknown);
        unchanged(at(Root::Heap(7), 0), Some(libc), FreeOutcome::Unknown);
    }

    /// **An access past a block or to an ended one, and a release that is no
    /// live block, are findings at the running instruction; the run goes
    /// on.** A write running past its block lands, and the block reads as
    /// the machine left it within its extent. Stale pointers keep their
    /// identity; what they reach after the block ended reads as unknown.
    #[test]
    fn object_misuse_is_found_and_the_run_goes_on() {
        let findings = Findings::default();
        let mut memory = Memory::default();
        memory.set_findings(findings.clone());
        let site = |instruction| WriteSite {
            body: 0,
            block: 1,
            instruction,
        };
        // Each finding carries the bytes it reached and the block as it
        // stood, ended at the release before a repeated one.
        let found = |instruction, kind, access, ended_at: &[(usize, usize)]| Finding {
            body: 0,
            block: 1,
            instruction,
            kind,
            access: Some(access),
            object: Some(FoundObject::Block {
                root: Root::Heap(0),
                size: 4,
                extent: Some(4),
                family: None,
                allocated_at: None,
                ended_at: ended_at.iter().copied().collect(),
            }),
            witness: Witness::default(),
        };
        let block = at(Root::Heap(0), 0);
        memory.allocate(block.root(), Some(4), Some(0), None);
        memory.store_address(at(Root::Stack, -8), block, 8);
        memory.store_bytes(block, &[1, 2, 3, 4]);
        memory.set_writer(0, site(5));
        memory.store_bytes(at(block.root(), 3), &[5, 6]);
        assert_eq!(memory.refusal(), None);
        assert_eq!(memory.load(block, 4, None), Loaded::Bytes(vec![1, 2, 3, 5]));
        memory.set_writer(0, site(6));
        assert_eq!(
            memory.free(at(block.root(), 1), None, None),
            FreeOutcome::Interior { offset: 1 }
        );
        assert_eq!(
            memory.free(block, None, Some((2, 3))),
            FreeOutcome::Released
        );
        assert_eq!(
            memory.free(block, None, None),
            FreeOutcome::Ended {
                lifetime: Lifetime::Ended {
                    sites: BTreeSet::from([(2, 3)])
                }
            }
        );
        assert_eq!(
            memory.load(at(Root::Stack, -8), 8, None),
            Loaded::Address(block)
        );
        assert_eq!(memory.load(block, 1, None), Loaded::Unknown);
        assert_eq!(
            memory.access_finding(block, 1),
            Some(FindingKind::Ended(Ending::Every))
        );
        assert_eq!(memory.access_finding(block, 0), None);
        memory.allocate(Root::Heap(1), Some(4), Some(0), None);
        assert_eq!(memory.known_run(at(Root::Heap(1), 0), 4, None), [0; 4]);
        memory.invalidate_lifetimes();
        assert_eq!(
            memory.access_finding(at(Root::Heap(1), 0), 1),
            Some(FindingKind::Ended(Ending::Possibly))
        );
        assert_eq!(
            findings.all(),
            [
                found(
                    5,
                    FindingKind::PastExtent(ExtentSource::Allocator),
                    (at(Root::Heap(0), 3), 2),
                    &[]
                ),
                found(
                    6,
                    FindingKind::Release(ReleaseFault::Repeated),
                    (block, 0),
                    &[(2, 3)]
                ),
                found(
                    6,
                    FindingKind::Release(ReleaseFault::Interior),
                    (at(Root::Heap(0), 1), 0),
                    &[]
                ),
            ]
        );
    }

    /// Copying any symbolic stack-pointer piece into public storage publishes the stack.
    #[test]
    fn copied_pointer_fragments_expose_private_stack_storage() {
        for width in [1, 4, 8] {
            let mut memory = Memory::default();
            let slot = at(Root::Stack, -16);
            let buffer = at(Root::Stack, -32);
            memory.store_bytes(buffer, b"secret");
            memory.store_address(slot, buffer, 8);
            assert!(!memory.stack_escaped());
            memory.copy(at(Root::Symbol, 0x2000), slot, width, None, true);
            assert!(memory.stack_escaped());
            memory.clobber();
            assert!(memory.known_run(buffer, 6, None).is_empty());
        }
    }

    /// A string read stops at its terminator instead of requiring the whole
    /// limit, and counts the bytes it inspected whatever stopped it.
    #[test]
    fn string_reads_stop_at_their_terminator() {
        let mut memory = Memory::default();
        let root = Root::Heap(0);
        memory.allocate(root, Some(4), None, None);
        memory.store_bytes(at(root, 0), b"abc\0");
        assert_eq!(
            memory.read_string(at(root, 0), 1 << 20, None),
            Ok(StringScan {
                string: Some(b"abc".to_vec()),
                inspected: 4,
            })
        );
        assert_eq!(
            memory.string_length(at(root, 0), 2, None),
            Ok(StringScan {
                string: None,
                inspected: 2,
            })
        );
        memory.store_bytes(at(root, 3), b"d");
        // Past the block nothing is known: the scan reads its fifth byte as
        // unknown, where the program reads past the block.
        assert_eq!(
            memory.read_string(at(root, 0), 1 << 20, None),
            Ok(StringScan {
                string: None,
                inspected: 5,
            })
        );
        memory.store_unknown(at(root, 2), 1);
        assert_eq!(
            memory.read_string(at(root, 0), 1 << 20, None),
            Ok(StringScan {
                string: None,
                inspected: 3,
            })
        );
    }

    /// Stack and opaque spans normalize bytes across the signed coordinate boundary.
    #[test]
    fn relative_spans_obey_the_same_modulus_as_scalar_arithmetic() {
        for root in [Root::Stack, Root::StackFrame(1), opaque(2)] {
            let mut memory = Memory::default();
            let start = Address::new(root, 126, PointerSize::Bit8);
            memory.store_bytes(start, &[1, 2, 3, 4]);
            for (offset, expected) in [(126, 1), (127, 2), (-128, 3), (-127, 4)] {
                assert_eq!(
                    memory.load(Address::new(root, offset, PointerSize::Bit8), 1, None),
                    Loaded::Bytes(vec![expected])
                );
            }
            memory.store_unknown(start, 4);
            assert!(memory.known_run(start, 4, None).is_empty());
            for offset in [126, 127, -128, -127] {
                assert_eq!(
                    memory.load(Address::new(root, offset, PointerSize::Bit8), 1, None),
                    Loaded::Unknown
                );
            }
        }
        let mut memory = Memory::default();
        let root = Root::Heap(0);
        memory.allocate(root, Some(u64::MAX), Some(0), None);
        let high = Address::new(root, -2, PointerSize::Bit64);
        assert_eq!(memory.load(high, 1, None), Loaded::Bytes(vec![0]));
        memory.store_unknown(high, 1);
        assert_eq!(memory.load(high, 1, None), Loaded::Unknown);
        assert_eq!(memory.access_refusal(high, 2), None);
        assert_eq!(
            memory.access_finding(high, 2),
            Some(FindingKind::PastExtent(ExtentSource::Allocator))
        );
    }

    /// Partial loads and joins retain provenance even when exact pointer identity is lost.
    #[test]
    fn incomplete_and_joined_pointer_pieces_remain_stack_derived() {
        let mut left = Memory::default();
        let slot = at(Root::Stack, -16);
        left.store_address(slot, at(Root::Stack, -32), 8);
        left.store_unknown(slot, 1);
        assert_eq!(left.load(slot, 8, None), Loaded::StackDerived);
        let mut right = Memory::default();
        right.store_address(slot, at(Root::Stack, -48), 8);
        let mut joined = left.join(&right);
        assert_eq!(joined.load(slot, 8, None), Loaded::StackDerived);
        joined.copy(at(Root::Symbol, 0x2000), slot, 1, None, true);
        assert!(joined.stack_escaped());
        assert_eq!(
            joined.load(at(Root::Symbol, 0x2000), 1, None),
            Loaded::StackDerived
        );
    }

    /// Derived identities relate their own offsets but alias both public and private roots.
    #[test]
    fn named_stack_derivatives_keep_offsets_without_hiding_private_aliases() {
        let derived = Root::StackDerived {
            activation: 0,
            value: SsaVarId::from_index(7),
            run: 1,
        };
        let mut memory = Memory::default();
        memory.store_bytes(at(Root::Stack, -16), b"private");
        memory.store_bytes(at(derived, 0), b"aligned");
        memory.store_bytes(at(derived, 8), b"counter");
        assert_eq!(memory.known_run(at(derived, 0), 7, None), b"aligned");
        assert!(memory.known_run(at(Root::Stack, -16), 7, None).is_empty());
        memory.store_bytes(at(Root::Stack, -32), b"other");
        assert!(memory.known_run(at(derived, 0), 7, None).is_empty());
        assert!(memory.known_run(at(derived, 8), 7, None).is_empty());
        assert!(!memory.stack_escaped());
        memory.store_address(at(Root::Symbol, 0x2000), at(derived, 0), 8);
        assert!(memory.stack_escaped());
    }

    /// The alias-space matrix is independent of byte-map traversal and root order.
    #[test]
    fn root_invalidation_matches_the_private_and_escaped_alias_matrix() {
        // Columns and rows: image, numeric, private stack, public stack,
        // detached stack, heap, opaque, stack-derived.
        let private = [
            [false, true, false, false, false, false, true, true],
            [true, false, false, true, false, true, true, true],
            [false, false, false, false, false, false, false, true],
            [false, true, false, false, false, false, true, true],
            [false, false, false, false, false, false, false, true],
            [false, true, false, false, false, false, true, true],
            [true, true, false, true, false, true, false, true],
            [true, true, true, true, true, true, true, false],
        ];
        let escaped = [
            [false, true, false, false, false, false, true, true],
            [true, false, true, true, true, true, true, true],
            [false, true, false, false, false, false, true, true],
            [false, true, false, false, false, false, true, true],
            [false, true, false, false, false, false, true, true],
            [false, true, false, false, false, false, true, true],
            [true, true, true, true, true, true, false, true],
            [true, true, true, true, true, true, true, false],
        ];
        let derived = Root::StackDerived {
            activation: 0,
            value: SsaVarId::from_index(7),
            run: 1,
        };
        let roots = [
            Root::Symbol,
            Root::Number,
            Root::Stack,
            Root::Stack,
            Root::StackFrame(1),
            Root::Heap(0),
            opaque(1),
            derived,
        ];
        let stored_offsets = [0, 0, -8, 8, 0, 0, 0, 0];
        let write_offsets = [64, 64, -32, 0, 64, 16, 64, 64];
        for (escaped_stack, matrix) in [(false, private), (true, escaped)] {
            for (row, writer) in roots.iter().enumerate() {
                let mut memory = Memory::default();
                memory.allocate(Root::Heap(0), Some(32), None, None);
                if escaped_stack {
                    memory.escape_stack();
                }
                for (root, offset) in roots.iter().zip(stored_offsets) {
                    memory.cells.insert((*root, offset), Cell::Byte(55));
                }
                memory.store_bytes(at(*writer, write_offsets[row]), &[99]);
                for (column, (root, offset)) in roots.iter().zip(stored_offsets).enumerate() {
                    let expected = if matrix[row][column] {
                        Loaded::Unknown
                    } else {
                        Loaded::Bytes(vec![55])
                    };
                    assert_eq!(
                        memory.load(at(*root, offset), 1, None),
                        expected,
                        "escaped={escaped_stack} row={row} column={column}"
                    );
                }
            }
        }
    }

    /// Copy coordinates require the same target width even for a zero-byte copy.
    #[test]
    fn copy_rejects_mixed_target_widths() {
        let mut memory = Memory::default();
        memory.copy(
            Address::new(Root::Stack, -1, PointerSize::Bit32),
            Address::new(Root::Symbol, 0, PointerSize::Bit64),
            0,
            None,
            true,
        );
        assert_eq!(memory.refusal(), Some(MemoryRefusal::AddressRange));
    }

    /// **A span refused whole is refused before the first byte of its
    /// encoding is fetched; one running past its block lands.** A store of
    /// two bytes from the last of a 128-bit space wraps it, and reads none of
    /// its bytes.
    /// Seventeen bytes into a 16-byte block are an access past its extent,
    /// which the program makes: all are read, and the block holds the
    /// sixteen within it.
    #[test]
    fn encoded_stores_check_the_whole_span_before_reading() {
        let reads = Counter::new(0);
        let encoding = |count: usize| {
            (0..count).map(|_| {
                reads.set(reads.get() + 1);
                0x5a
            })
        };
        let mut memory = Memory::default();
        memory.store_encoded(
            Address::new(Root::Symbol, -1, PointerSize::Bit128),
            encoding(2),
        );
        assert_eq!(reads.get(), 0);
        assert_eq!(memory.refusal(), Some(MemoryRefusal::AddressRange));

        let mut memory = Memory::default();
        memory.allocate(Root::Heap(0), Some(16), Some(0), None);
        memory.store_encoded(at(Root::Heap(0), 0), encoding(17));
        assert_eq!(reads.get(), 17);
        assert_eq!(memory.refusal(), None);
        assert_eq!(memory.known_run(at(Root::Heap(0), 0), 16, None), [0x5a; 16]);
        assert_eq!(
            memory.access_finding(at(Root::Heap(0), 0), 17),
            Some(FindingKind::PastExtent(ExtentSource::Allocator))
        );
    }

    /// Repeated encodings keep an exact remaining length, including an incomplete last element.
    #[test]
    fn repeated_patterns_are_exact_and_bounded() {
        let expected = [0x12, 0, 0xfe, 0x12, 0, 0xfe, 0x12, 0, 0xfe, 0x12, 0];
        let mut bytes = PatternBytes {
            pattern: &[0x12, 0, 0xfe],
            offset: 0,
            remaining: expected.len(),
        };
        for (index, byte) in expected.iter().enumerate() {
            assert_eq!(bytes.len(), expected.len() - index);
            assert_eq!(bytes.next(), Some(*byte));
        }
        assert_eq!(bytes.len(), 0);
        assert_eq!(bytes.next(), None);
        assert_eq!(bytes.next(), None);
        let mut memory = Memory::default();
        let address = at(Root::Stack, -32);
        memory.store_pattern(address, &[0x12, 0, 0xfe], expected.len());
        assert_eq!(memory.known_run(address, expected.len(), None), expected);
        memory.store_pattern(address, &[], 1);
        assert_eq!(memory.load(address, 1, None), Loaded::Unknown);
        assert_eq!(
            memory.known_run(address.moved(1, PointerSize::Bit64).unwrap(), 10, None),
            expected[1..]
        );
    }

    #[test]
    fn disjoint_numeric_stores_preserve_public_stack_and_image_bindings() {
        let mut memory = Memory::default();
        memory.set_disjoint_numeric(iter::once(0x5000..0x5008).collect(), PointerSize::Bit64);
        let stack = Address::new(Root::Stack, 0, PointerSize::Bit64);
        let symbol = Address::new(Root::Symbol, 0x2000, PointerSize::Bit64);
        let numeric = Address::new(Root::Number, 0x5000, PointerSize::Bit64);
        memory.store_bytes(stack, &[42]);
        memory.store_bytes(numeric, &[3]);
        assert_eq!(memory.known_byte(stack), Some(42));
        assert!(memory.initial_image_binding(symbol, 1));
        memory.store_bytes(symbol, &[11]);
        assert_eq!(memory.known_byte(numeric), Some(3));
        let unmapped = Address::new(Root::Number, 0x5010, PointerSize::Bit64);
        memory.store_bytes(unmapped, &[9]);
        assert_eq!(memory.known_byte(stack), None);
        assert!(!memory.initial_image_binding(symbol, 1));
        memory.clobber();
        assert_eq!(memory.known_byte(numeric), None);
    }

    /// Makes explicit cells, symbolic fragments and lazy heap intervals independently.
    fn copy_fixture(count: usize, intervals: usize) -> Memory {
        let mut memory = Memory {
            stack_escaped: true,
            refusal: Some(MemoryRefusal::BlockLayout),
            ..Memory::default()
        };
        for index in 0..count {
            let cell = match index % 3 {
                0 => Cell::Byte(index as u8),
                1 => Cell::StackBits,
                _ => Cell::Piece {
                    address: Address::new(Root::StackFrame(7), -16, PointerSize::Bit32),
                    index: (index % 4) as u8,
                    width: 4,
                },
            };
            memory.cells.insert((Root::Stack, index as i64), cell);
        }
        memory.objects.insert(
            Root::Heap(3),
            Object {
                size: 16384,
                extent: Some(16384),
                family: Some(AllocatorFamily(5)),
                heap: None,
                site: None,
                lifetime: Lifetime::Live,
                initial: Some(0),
                unknown: (0..intervals)
                    .map(|index| ((index * 2) as u64, (index * 2 + 1) as u64))
                    .collect(),
            },
        );
        memory
    }

    /// Snapshots share cells while independently copied interval metadata compares exactly.
    #[test]
    fn memory_copies_share_cells_and_preserve_interval_metadata() {
        for (cells, intervals) in [(4096, 0), (0, 4096), (4096, 4096)] {
            let memory = copy_fixture(cells, intervals);
            let original = copy_fixture(cells, intervals);
            let copied = memory.clone();
            assert!(copied.cells.shares(&memory.cells));
            assert_eq!(copied, original);
            assert_eq!(memory, original);

            assert!(memory.eq(&original));
        }
    }

    /// Equal counts do not hide a different interval, object, provenance or refusal.
    #[test]
    fn memory_agreement_compares_semantic_metadata_and_map_contents() {
        let original = copy_fixture(8, 4);
        for field in 0..9 {
            let mut changed = original.clone();
            match field {
                0 => changed.stack_escaped = false,
                1 => changed.refusal = None,
                2 => {
                    changed.cells.insert((Root::Stack, 1), Cell::Byte(3));
                }
                3 => changed.objects.get_mut(&Root::Heap(3)).unwrap().size = 16383,
                4 => changed.objects.get_mut(&Root::Heap(3)).unwrap().initial = Some(1),
                7 => changed.objects.get_mut(&Root::Heap(3)).unwrap().family = None,
                8 => changed.objects.get_mut(&Root::Heap(3)).unwrap().extent = None,
                5 => {
                    changed
                        .objects
                        .get_mut(&Root::Heap(3))
                        .unwrap()
                        .unknown
                        .insert(0, 2);
                }
                _ => {
                    let object = changed.objects.remove(&Root::Heap(3)).unwrap();
                    changed.objects.insert(Root::Heap(4), object);
                }
            }
            assert_ne!(original, changed, "field {field}");
        }
    }

    /// Sharing explicit cells does not prove agreement of lifetime, backing or escape rules.
    #[test]
    fn shared_cell_joins_still_meet_all_memory_contracts() {
        let heap = Root::Heap(4);
        let mut left = Memory::default();
        left.allocate(heap, Some(8), Some(0), None);
        left.store_bytes(at(heap, 0), &[0x11]);
        let mut right = left.clone();
        right.objects.get_mut(&heap).unwrap().size = 4;
        right.objects.get_mut(&heap).unwrap().extent = Some(4);
        right.objects.get_mut(&heap).unwrap().initial = Some(1);
        right.stack_escaped = true;
        right.refusal = Some(MemoryRefusal::AddressRange);
        assert!(left.cells.shares(&right.cells));
        for joined in [left.join(&right), right.join(&left)] {
            assert_eq!(joined.load(at(heap, 0), 1, None), Loaded::Bytes(vec![0x11]));
            assert_eq!(joined.load(at(heap, 1), 1, None), Loaded::Unknown);
            // Past the four bytes one path holds is inside the other's eight,
            // so only past the eight is past the block on every path.
            assert_eq!(joined.access_finding(at(heap, 4), 1), None);
            assert_eq!(
                joined.access_finding(at(heap, 8), 1),
                Some(FindingKind::PastExtent(ExtentSource::Allocator))
            );
            assert_eq!(joined.refusal(), Some(MemoryRefusal::AddressRange));
            assert!(joined.stack_escaped());
        }
        right.objects.remove(&heap);
        let joined = left.join(&right);
        assert_eq!(joined.load(at(heap, 0), 1, None), Loaded::Unknown);
        assert_eq!(joined.access_finding(at(heap, 0), 1), None);
    }

    /// A conflicting byte does not require rebuilding every untouched byte of a decoded buffer.
    #[test]
    fn conflicting_memory_joins_reuse_unmodified_pages_with_bounded_copies() {
        let address = at(Root::Stack, -8192);
        let last = at(Root::Stack, -1);
        let mut left = Memory::default();
        left.store_bytes(address, &[0x5a; 8192]);
        let mut right = left.clone();
        right.store_bytes(last, &[0x11]);

        let joined = left.join(&right);

        assert_eq!(joined.known_run(address, 8192, None), vec![0x5a; 8191]);
        assert_eq!(joined.load(last, 1, None), Loaded::Unknown);
        assert_eq!(left.load(last, 1, None), Loaded::Bytes(vec![0x5a]));
        assert_eq!(right.load(last, 1, None), Loaded::Bytes(vec![0x11]));
    }

    /// A shared page survives only inside the common live extent, including high unsigned offsets.
    #[test]
    fn shared_pages_do_not_preserve_cells_outside_the_joined_heap_extent() {
        let root = Root::Heap(3);
        for (offset, size, joined_size, count, retained) in [
            (0, 1024, 128, 256usize, 128usize),
            (-129, u64::MAX, u64::MAX - 64, 128, 64),
        ] {
            let address = at(root, offset);
            let mut left = Memory::default();
            left.allocate(root, Some(size), None, None);
            left.store_bytes(address, &vec![0x42; count]);
            let mut right = left.clone();
            right.objects.get_mut(&root).unwrap().size = joined_size;
            for joined in [left.join(&right), right.join(&left)] {
                assert_eq!(joined.known_run(address, count, None), vec![0x42; retained]);
                let explicit: Vec<_> = joined
                    .held_runs(root)
                    .into_iter()
                    .flat_map(|(_, bytes)| bytes)
                    .collect();
                assert_eq!(explicit, vec![0x42; retained]);
            }
            assert_eq!(left.known_run(address, count, None), vec![0x42; count]);
            right.objects.remove(&root);
            let joined = left.join(&right);
            assert!(joined.held_runs(root).is_empty());
            assert_eq!(joined.access_finding(address, 1), None);
        }
    }

    /// Invalid reverse spans and malformed elements refuse before the destination changes.
    #[test]
    fn streamed_block_spans_refuse_wrap_and_invalid_layout_before_writes() {
        for pointer_size in [
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            for root in [Root::Number, Root::Stack] {
                let minimum = if root == Root::Number {
                    0
                } else {
                    i64::MIN >> 64u32.saturating_sub(pointer_size.bits())
                };
                let valid = Address::new(root, 128, pointer_size);
                let too_low = Address::new(root, minimum + 1, pointer_size);
                for invalid_source in [false, true] {
                    let mut memory = Memory::default();
                    memory.store_bytes(valid, &[0x55; 16]);
                    memory.copy_ordered(
                        if invalid_source { valid } else { too_low },
                        if invalid_source { too_low } else { valid },
                        8,
                        None,
                        CopyOrder::Stream {
                            element_bytes: 2,
                            reverse: true,
                        },
                    );
                    assert_eq!(memory.refusal(), Some(MemoryRefusal::AddressRange));
                    assert_eq!(memory.known_run(valid, 16, None), [0x55; 16]);
                }
            }
        }
        for (width, len) in [(0, 8), (3, 8), (16, 8)] {
            let mut memory = Memory::default();
            let address = at(Root::Stack, -64);
            memory.store_bytes(address, &[0x77; 16]);
            memory.copy_ordered(
                address,
                address,
                len,
                None,
                CopyOrder::Stream {
                    element_bytes: width,
                    reverse: false,
                },
            );
            assert_eq!(memory.refusal(), Some(MemoryRefusal::BlockLayout));
            assert_eq!(memory.known_run(address, 16, None), [0x77; 16]);
        }
    }

    /// Stream copies retain whole pointer identities regardless of element partition or direction.
    #[test]
    fn streamed_block_copies_preserve_pointer_pieces_and_heap_admission() {
        for reverse in [false, true] {
            for width in [1, 2, 4, 8] {
                let mut memory = Memory::default();
                let source = at(Root::Stack, -192);
                let dest = at(Root::Stack, -128);
                let pointer = at(Root::Stack, -24);
                memory.store_address(source, pointer, 8);
                let delta = if reverse { 8 - width } else { 0 };
                memory.copy_ordered(
                    dest.moved(delta as i64, PointerSize::Bit64).unwrap(),
                    source.moved(delta as i64, PointerSize::Bit64).unwrap(),
                    8,
                    None,
                    CopyOrder::Stream {
                        element_bytes: width,
                        reverse,
                    },
                );
                assert_eq!(memory.refusal(), None);
                assert_eq!(memory.load(dest, 8, None), Loaded::Address(pointer));
            }
        }
        for live in [false, true] {
            let mut memory = Memory::default();
            let dest = at(Root::Stack, -64);
            let source = at(Root::Heap(1), 2);
            memory.store_bytes(dest, &[0x88; 8]);
            if live {
                memory.allocate(Root::Heap(1), Some(3), Some(0x11), None);
            }
            memory.copy_ordered(
                dest,
                source,
                4,
                None,
                CopyOrder::Stream {
                    element_bytes: 2,
                    reverse: false,
                },
            );
            // The copy is made: past the block, and from a block the path never
            // made, the source is unknown, and so is what the copy writes.
            assert_eq!(memory.refusal(), None);
            assert_eq!(
                memory.access_finding(source, 4),
                live.then_some(FindingKind::PastExtent(ExtentSource::Allocator))
            );
            assert_eq!(
                memory.known_run(dest.moved(4, PointerSize::Bit64).unwrap(), 4, None),
                [0x88; 4]
            );
            assert_eq!(memory.known_run(dest, 2, None).len(), usize::from(live));
        }
    }

    /// A readable live object need not have known contents; its ended lifetime admits no read.
    #[test]
    fn transfer_read_evidence_distinguishes_unknown_contents_and_lifetime() {
        for width in [1usize, 2, 16] {
            for reverse in [false, true] {
                for buffered in [false, true] {
                    let bytes = 3 * width;
                    let first_offset = if reverse && !buffered {
                        (bytes - width) as i64
                    } else {
                        0
                    };
                    let source = at(Root::Heap(1), first_offset);
                    let destination = at(Root::Stack, -128 + first_offset);
                    let low_destination = at(Root::Stack, -128);
                    let order = if buffered {
                        CopyOrder::Snapshot
                    } else {
                        CopyOrder::Stream {
                            element_bytes: width,
                            reverse,
                        }
                    };
                    let mut memory = Memory::default();
                    memory.allocate(Root::Heap(1), Some(bytes as u64), None, None);
                    memory.store_bytes(low_destination, &vec![0xaa; bytes]);
                    assert!(memory.copy_ordered(destination, source, bytes as i64, None, order));
                    assert_eq!(memory.refusal(), None);
                    assert_eq!(memory.load(low_destination, bytes, None), Loaded::Unknown);
                    assert_eq!(
                        memory.free(at(Root::Heap(1), 0), None, None),
                        FreeOutcome::Released
                    );
                    memory.store_bytes(low_destination, &vec![0xbb; bytes]);
                    // A copy out of the ended block is made, and finds it ended:
                    // what it writes is unknown.
                    assert!(!memory.copy_ordered(destination, source, bytes as i64, None, order));
                    assert_eq!(memory.refusal(), None);
                    assert_eq!(
                        memory.access_finding(source, 1),
                        Some(FindingKind::Ended(Ending::Every))
                    );
                    assert!(memory.known_run(low_destination, bytes, None).is_empty());
                }
            }
        }
        for root in [Root::Symbol, Root::Number, Root::Stack] {
            let mut memory = Memory::default();
            let destination = at(Root::Stack, -128);
            let source = at(root, if root == Root::Stack { -64 } else { 0x3000 });
            memory.store_bytes(destination, &[0xaa; 4]);
            assert!(!memory.copy(destination, source, 4, None, true));
            assert_eq!(memory.refusal(), None);
            assert_eq!(memory.load(destination, 4, None), Loaded::Unknown);
        }
    }

    /// Copying image bytes preserves independent heap objects.
    #[test]
    fn copy_preserves_independent_heap_objects() {
        let used = |roots: u32| {
            let mut memory = Memory::default();
            for heap in 0..roots {
                memory.allocate(Root::Heap(heap), Some(1), None, None);
                memory.store_bytes(at(Root::Heap(heap), 0), &[1]);
            }
            assert_eq!(memory.known_run(at(Root::Heap(roots - 1), 0), 1, None), [1]);
            memory.store_bytes(at(Root::Symbol, 0x1_0000), &[0x5a; 4096]);

            memory.copy_ordered(
                at(Root::Symbol, 0x2_0000),
                at(Root::Symbol, 0x1_0000),
                4096,
                None,
                CopyOrder::Stream {
                    element_bytes: 1,
                    reverse: false,
                },
            );

            assert_eq!(
                memory.known_run(at(Root::Symbol, 0x2_0000), 4096, None),
                [0x5a; 4096]
            );
            assert_eq!(memory.known_run(at(Root::Heap(roots - 1), 0), 1, None), [1]);
        };
        used(1);
        used(256);
    }

    /// A pointer the copy itself carries into named memory escapes the stack
    /// part-way through, and from then on its stores may reach private stack
    /// bytes: those are forgotten exactly as a byte-by-byte copy forgets them.
    #[test]
    fn copy_escaping_the_stack_forgets_private_stack_aliases() {
        let mut memory = Memory::default();
        let private = at(Root::Stack, -8);
        memory.store_bytes(private, &[0xaa]);
        memory.store_bytes(at(Root::Stack, -80), &[0x11; 8]);
        memory.store_address(at(Root::Stack, -72), private, 8);
        assert!(!memory.stack_escaped());
        assert_eq!(memory.known_run(private, 1, None), [0xaa]);
        let dest = at(Root::Number, 0x5000);
        memory.copy_ordered(
            dest,
            at(Root::Stack, -80),
            16,
            None,
            CopyOrder::Stream {
                element_bytes: 1,
                reverse: false,
            },
        );
        assert!(memory.stack_escaped());
        assert_eq!(memory.known_run(dest, 8, None), [0x11; 8]);
        assert!(memory.known_run(private, 1, None).is_empty());
    }
}
