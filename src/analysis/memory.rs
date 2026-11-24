//! Memory SSA (MSSA) for tracking versioned memory locations.
//!
//! This module extends SSA to track state stored in fields, arrays, and heap locations.
//! Memory SSA is essential for precise analysis of obfuscated code that stores state
//! in memory rather than local variables, and for field-based state machines in
//! control flow flattening.
//!
//! # Architecture
//!
//! Memory SSA builds on top of traditional SSA by adding:
//!
//! 1. **Memory Locations**: Abstract representation of memory (fields, arrays, pointers)
//! 2. **Memory Versioning**: Each store creates a new version, each load reads a version
//! 3. **Memory Phi Nodes**: At control flow merges, memory versions are merged
//!
//! ```text
//! Traditional SSA:           Memory SSA:
//!
//!   v1 = x                     v1 = x
//!   obj.field = v1             mem[obj.field]₁ = v1
//!   ...                        ...
//!   v2 = obj.field             v2 = mem[obj.field]₁
//! ```
//!
//! # Construction Algorithm (three-phase)
//!
//! **Phase 1: Operation Identification** (O(N)):
//! Scan all instructions to classify memory operations (LoadField, StoreField,
//! LoadStaticField, StoreStaticField, LoadElement, StoreElement, LoadIndirect,
//! StoreIndirect). Each operation is recorded with its location and block/instr position.
//!
//! **Phase 2: Phi Placement** (O(L * F)):
//! Uses the iterated dominance frontier algorithm from Cytron et al.:
//! 1. For each memory location with stores, find the set of definition blocks
//! 2. Compute the iterated dominance frontier of these blocks
//! 3. Place memory phi nodes at each frontier block
//! 4. Add new phi blocks to the worklist (they're also definitions)
//! 5. Repeat until fixed point
//!
//! **Phase 3: Renaming and instruction indexing**:
//! Traverse the dominator tree in preorder, maintaining a version stack per location:
//! 1. Record entry version for each location at block entry
//! 2. Push new versions for phi nodes and store definitions
//! 3. Process all instructions (loads read top of stack, stores create new versions)
//! 4. Record exit versions for each location at block exit
//! 5. Fill in phi operands for successor blocks with current version stacks
//!
//! Instruction-point queries index existing definitions on first use. The
//! location/version traversal already orders each block's records, so the
//! bounded index needs no sort. Subsequent queries allocate nothing and use
//! binary search; block-boundary consumers build no instruction index.
//!
//! # Memory Location Hierarchy
//!
//! Memory locations form a hierarchy for alias analysis:
//!
//! ```text
//! Unknown (may alias anything)
//!   ├── StaticField(token)      - Specific static field
//!   ├── InstanceField(obj, token) - Specific instance field
//!   ├── ArrayElement(arr, idx)   - Specific array element
//!   │     ├── ArrayElement(arr, Constant(i)) - Known index
//!   │     └── ArrayElement(arr, Variable(v)) - Unknown index (may alias)
//!   └── Indirect{base,index,offset,size} - Decoded pointer dereference
//!         ├── same base    - compare bit extents (disjoint offsets don't alias)
//!         └── other base   - may alias (needs points-to to separate)
//! ```
//!
//! # Alias Analysis
//!
//! The `may_alias` and `must_alias` methods on `MemoryLocation` provide
//! location-based alias analysis:
//!
//! | Location A | Location B | may_alias | must_alias |
//! |------------|------------|-----------|------------|
//! | StaticField(f1) | StaticField(f2) | f1 == f2 | f1 == f2 |
//! | InstanceField(o1, f1) | InstanceField(o2, f2) | o1==o2 && f1==f2 | o1==o2 && f1==f2 |
//! | ArrayElement(a1, i1) | ArrayElement(a2, i2) | a1==a2 && i1.may_overlap(i2) | a1==a2 && i1.must_equal(i2) |
//! | Indirect(a) | Indirect(b) | see below | see below |
//! | Unknown | Anything | true | false |
//! | StaticField | InstanceField/ArrayElement | false | false |
//! | InstanceField | ArrayElement | false | false |
//!
//! An `Indirect` location is a *decoded* address - `base + index*stride +
//! offset` plus the access width - not the SSA id of the address operand. Two
//! of them off a common base compare by overlapping bit extent, so `[rbp-8]`
//! and `[rbp-16]` are provably disjoint; different bases conservatively
//! may-alias, since two SSA pointers can hold one address - except where
//! both bases are storage of their own: two different frame slots, and a
//! frame slot against a *stated* absolute address, which is a link-time or
//! program-stated address and so names no slot of the function being
//! analysed. An address the
//! program states as a number has the absolute base, so two image cells
//! compare by extent too, however their addresses were materialised. See
//! [`IndirectLocation`] for the full rationale, including why keying on the
//! address value id is both imprecise and unstable under GVN.
//!
//! # Regions
//!
//! A location lies in a [`MemoryRegion`]: the image's read-only data, its
//! writable data, the function's own frame, or anywhere else. The region is
//! what decides what a location holds when the function is entered, and one
//! rule decides everything built on it:
//!
//! > A load folds to a constant only if its region is provably unwritten on
//! > every path reaching it.
//!
//! A [read-only image](MemoryRegion::ImageReadOnly) cell gives that for free: a
//! write to it faults, so no execution continues past one, and nothing the
//! function does can have changed it. Such a location is never versioned by a
//! store or a barrier, and its one version is defined by
//! [`MemoryDefSite::Image`] - the image's bytes. Every other region's entry
//! version is [`MemoryDefSite::Entry`], which says nothing about what an
//! earlier function wrote. The regions come from the host through
//! [`AnalysisContext`]; without an image every location is
//! [`MemoryRegion::Unknown`] and nothing is read.
//!
//! # Usage
//!
//! ```rust
//! use analyssa::{
//!     analysis::{
//!         memory::{MemoryLocation, MemorySsa},
//!         AnalysisContext, SsaCfg,
//!     },
//!     ir::SsaVarId,
//!     testing,
//! };
//!
//! // A function that stores to `object.field1`, loads it back, then performs
//! // an indirect store and an atomic exchange.
//! let ssa = testing::memory_effect_fixture();
//! let cfg = SsaCfg::from_ssa(&ssa);
//! let mem_ssa = MemorySsa::build(&ssa, &AnalysisContext::new(testing::MACHINE));
//!
//! // The store and the load are attributed to the same memory location.
//! let object = SsaVarId::from_index(0);
//! let loc = MemoryLocation::InstanceField(object, 1);
//! assert!(mem_ssa.locations().contains(&loc));
//!
//! // Query the memory version entering and leaving a block. Every operation
//! // that may alias this location bumps its version, not just the ones naming
//! // it: the function's indirect store and atomic exchange both classify to
//! // `Unknown`, which may-aliases everything, so the field's version advances
//! // past them too. That is what makes a version comparison a valid test for
//! // "is the value I read here still intact".
//! assert_eq!(mem_ssa.version_at_entry(&loc, 0), Some(0));
//! assert_eq!(mem_ssa.version_at_exit(&loc, 0), Some(4));
//!
//! // A must-alias query holds for the identical location...
//! assert!(loc.must_alias(&loc));
//!
//! // ...but a distinct field of the same object does not alias it.
//! let other = MemoryLocation::InstanceField(object, 2);
//! assert!(!loc.may_alias(&other));
//! ```
//!
//! # References
//!
//! - Chow et al., "Effective Representation of Aliases and Indirect Memory
//!   Operations in SSA Form", CC 1996

use std::{
    borrow::Borrow, cmp::Reverse, collections::VecDeque, num::NonZeroU16, ops::Range, rc::Rc,
    sync::OnceLock,
};

use log::warn;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    analysis::{
        address::{AddressBase, AddressWalk},
        budget::{Budget, Exhaustion},
        cells::MemoryCells,
        cfg::SsaCfg,
        context::AnalysisContext,
        frame::{FrameAnchors, FrameEscape},
        image::{ImageData, Mutability},
        pointsto::{self, Footprint, PointsTo},
    },
    bitset::BitSet,
    graph::{
        GraphBase, NodeId, RootedGraph, Successors,
        algorithms::{DominatorTree, compute_dominance_frontiers, compute_dominators},
    },
    ir::{
        function::SsaFunction,
        ops::{
            AccessMode, CallAccess, CallLocation, FrameOrigin, MemoryEffectLocation, PointeeExtent,
            SsaEffectKind, SsaEffects, SsaOp, SystemOpKind,
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
};

/// Represents an abstract memory location.
///
/// Memory locations are used to track which memory is being accessed by
/// load/store operations. The granularity varies by location type:
///
/// - Static fields are precise (one location per field)
/// - Instance fields depend on object identity (may alias if objects may alias)
/// - Array elements depend on both array identity and index
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum MemoryLocation<T: Target> {
    /// Instance field access: `object.field`
    ///
    /// The `SsaVarId` identifies the object, and `T::FieldRef` identifies the field.
    /// Two instance field locations may alias if the objects may alias.
    InstanceField(SsaVarId, T::FieldRef),

    /// Static field access: `ClassName.field`
    ///
    /// Static fields are uniquely identified by their token. Two static field
    /// locations alias iff they have the same token.
    StaticField(T::FieldRef),

    /// Array element access: `array[index]`
    ///
    /// The `SsaVarId` identifies the array, and `ArrayIndex` identifies the index.
    /// Array element locations may alias based on array identity and index overlap.
    ArrayElement(SsaVarId, ArrayIndex),

    /// Indirect memory access through a pointer: `*(base + index*stride + offset)`
    ///
    /// Decoded from the address expression rather than keyed on the address
    /// value id - see [`IndirectLocation`] for why that distinction is
    /// load-bearing.
    Indirect(IndirectLocation),

    /// Unknown/escaped memory.
    ///
    /// Used when we can't determine the exact location (e.g., after a call
    /// that may modify memory, or for volatile accesses).
    Unknown,

    /// Any memory the program has let escape: every location but the
    /// function's own frame cells whose address it kept to itself.
    ///
    /// What a call that touches memory only through what it was handed, and
    /// carries its stack arguments as operands, reaches
    /// ([`CallLocation::Escaped`]). It
    /// may alias every location [`Self::Unknown`] does, except a frame cell
    /// whose address never left the function.
    Escaped,
}

impl<T: Target> MemoryLocation<T> {
    /// Returns the region this location lies in.
    ///
    /// # Arguments
    ///
    /// * `context` - The host's facts, for its image.
    /// * `entry_stack_pointer` - The value the host declared the stack pointer
    ///   at entry, when it declared one.
    #[must_use]
    pub fn region(
        &self,
        context: &AnalysisContext<'_>,
        entry_stack_pointer: Option<SsaVarId>,
    ) -> MemoryRegion {
        let Self::Indirect(cell) = self else {
            return MemoryRegion::Unknown;
        };
        if cell.base.is_slot() {
            return MemoryRegion::Frame;
        }
        if entry_stack_pointer.is_some_and(|sp| cell.base == AddressBase::Value(sp))
            && cell.index.is_none()
            && cell.address_space.is_none()
        {
            return MemoryRegion::Frame;
        }
        let Some(image) = context.image() else {
            return MemoryRegion::Unknown;
        };
        let Some((address, len)) = cell.image_extent(image, None) else {
            return MemoryRegion::Unknown;
        };
        match image.region(address) {
            Some(region) if region.covers(address, len) => match region.mutability {
                Mutability::ReadOnly => MemoryRegion::ImageReadOnly,
                Mutability::Writable => MemoryRegion::ImageWritable,
            },
            _ => MemoryRegion::Unknown,
        }
    }

    /// Returns the memory an access to this location may touch, as
    /// `relation` places it ([`Footprint`]): escaped memory for
    /// [`Escaped`](Self::Escaped), an indirect cell's
    /// ([`IndirectLocation::footprint`]), and anywhere for the rest.
    ///
    /// # Arguments
    ///
    /// * `relation` - The function's points-to relation.
    /// * `image` - The host's image, which places a stated address in it.
    #[must_use]
    pub fn footprint(&self, relation: &PointsTo, image: Option<&dyn ImageData>) -> Footprint {
        match self {
            Self::Escaped => Footprint::ESCAPED,
            Self::Indirect(cell) => cell.footprint(relation, image),
            Self::InstanceField(..)
            | Self::StaticField(_)
            | Self::ArrayElement(..)
            | Self::Unknown => Footprint::Anywhere,
        }
    }

    /// Returns the base object variable, if any.
    ///
    /// For instance fields and arrays, this is the object/array variable, and
    /// for an indirect access its base value. For static fields, unknown
    /// locations and absolute addresses, returns `None`.
    #[must_use]
    pub fn base_object(&self) -> Option<SsaVarId> {
        match self {
            Self::InstanceField(obj, _) => Some(*obj),
            Self::ArrayElement(arr, _) => Some(*arr),
            Self::Indirect(indirect) => indirect.base.value(),
            Self::StaticField(_) | Self::Unknown | Self::Escaped => None,
        }
    }

    /// Returns `true` if this location may alias the other location.
    ///
    /// Conservative by design: when non-aliasing cannot be proven the answer is
    /// `true`. This predicate drives the *invalidation* half of the memory pass -
    /// a store or barrier bumps the version of every location it may-alias - so a
    /// false negative here would let a stale value survive a clobbering write.
    /// Distinct SSA ids for the same object or array are therefore treated as
    /// possibly equal; only [`must_alias`](Self::must_alias) insists on identical
    /// ids. `Unknown` aliases everything, and an `Indirect` cell conservatively
    /// aliases every concrete managed location.
    ///
    /// # Arguments
    ///
    /// * `other` - The location to test against.
    ///
    /// # Returns
    ///
    /// `true` if the two locations may refer to overlapping memory.
    #[must_use]
    pub fn may_alias(&self, other: &Self) -> bool {
        match (self, other) {
            // A slot of the function's own frame is not a static field.
            (Self::Indirect(cell), Self::StaticField(_))
            | (Self::StaticField(_), Self::Indirect(cell))
                if cell.base.is_slot() =>
            {
                false
            }
            // Unknown aliases everything; Indirect may alias any concrete location.
            // Escaped memory leaves out only frame cells whose address never left
            // the function, which this predicate cannot tell apart, so it aliases
            // everything too; the memory pass, which knows the frame, narrows it.
            (Self::Unknown | Self::Escaped, _)
            | (_, Self::Unknown | Self::Escaped)
            | (
                Self::Indirect(_),
                Self::InstanceField(..) | Self::ArrayElement(..) | Self::StaticField(_),
            )
            | (
                Self::InstanceField(..) | Self::ArrayElement(..) | Self::StaticField(_),
                Self::Indirect(_),
            ) => true,

            // Static fields alias iff same field
            (Self::StaticField(f1), Self::StaticField(f2)) => f1 == f2,

            // Static fields don't alias instance fields or arrays;
            // Instance fields don't alias array elements (different memory types)
            (Self::StaticField(_), Self::InstanceField(..) | Self::ArrayElement(..))
            | (Self::InstanceField(..) | Self::ArrayElement(..), Self::StaticField(_))
            | (Self::InstanceField(..), Self::ArrayElement(..))
            | (Self::ArrayElement(..), Self::InstanceField(..)) => false,

            // Instance fields on the same object alias iff it is the same field.
            //
            // Different *SSA ids* for the object do NOT prove different objects:
            // two un-GVN'd loads of one slot, two arguments, or a value and a phi
            // of it all name one object through distinct ids. Treating distinct
            // ids as disjoint would be a false NoAlias, which is unsound in the
            // direction that matters - `may_alias` drives the invalidation half
            // of the memory pass, so a false NoAlias lets a stale value survive a
            // store to the same cell. This mirrors `Indirect`, which is already
            // may-alias across distinct bases for exactly this reason.
            //
            // `must_alias` keeps requiring identical ids, so forwarding still
            // fires on the common same-id case.
            (Self::InstanceField(obj1, f1), Self::InstanceField(obj2, f2)) => {
                obj1 != obj2 || f1 == f2
            }

            // Array elements: same reasoning for the array reference. Distinct
            // ids may name one array, so only same-id lets the index comparison
            // prove disjointness.
            (Self::ArrayElement(arr1, idx1), Self::ArrayElement(arr2, idx2)) => {
                arr1 != arr2 || idx1.may_overlap(idx2)
            }

            // Two indirect accesses: compare the decoded address expressions.
            (Self::Indirect(a), Self::Indirect(b)) => a.may_alias(b),
        }
    }

    /// Returns `true` if this location must alias the other location.
    ///
    /// The precise counterpart to [`may_alias`](Self::may_alias): it returns
    /// `true` only when the two descriptions provably denote one cell, requiring
    /// identical object/array ids together with identical field tokens or
    /// constant indices. `Unknown` never must-aliases, not even itself. This
    /// drives the *forwarding* half of the memory pass, where a false positive
    /// would be unsound, so it never guesses.
    ///
    /// # Arguments
    ///
    /// * `other` - The location to test against.
    ///
    /// # Returns
    ///
    /// `true` if the two locations provably refer to the same memory cell.
    #[must_use]
    pub fn must_alias(&self, other: &Self) -> bool {
        match (self, other) {
            // Static fields must-alias iff same field
            (Self::StaticField(f1), Self::StaticField(f2)) => f1 == f2,

            // Instance fields must-alias iff same object AND same field
            (Self::InstanceField(obj1, f1), Self::InstanceField(obj2, f2)) => {
                obj1 == obj2 && f1 == f2
            }

            // Array elements must-alias iff same array AND same constant index
            (Self::ArrayElement(arr1, idx1), Self::ArrayElement(arr2, idx2)) => {
                arr1 == arr2 && idx1.must_equal(idx2)
            }

            // Indirect must-alias iff the decoded addresses denote one cell
            (Self::Indirect(a), Self::Indirect(b)) => a.must_alias(b),

            // Unknown never must-aliases (not precise enough)
            _ => false,
        }
    }
}

/// Which part of the program's memory a location lies in.
///
/// Decides what the location holds when the function is entered; see the
/// module documentation for the rule a fold rests on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryRegion {
    /// Image data the host reports read-only: the file's bytes, in every
    /// function, always.
    ImageReadOnly,
    /// Image data the program may write: the file's bytes only until
    /// something writes them, which can happen before this function runs.
    ImageWritable,
    /// The function's own frame, addressed from the stack pointer it was
    /// entered with.
    Frame,
    /// Anywhere else, or somewhere the analysis cannot place: memory reached
    /// through a pointer, the heap, an address the host does not serve.
    Unknown,
}

impl MemoryRegion {
    /// Whether this region and `other` cannot be the same memory.
    ///
    /// The frame is the stack the function was entered on; the image's two
    /// regions are the bytes the host serves at the addresses it states. No two
    /// of those three overlap. [`MemoryRegion::Unknown`] is the answer for memory
    /// reached through a pointer, so it overlaps everything, itself included.
    #[must_use]
    pub const fn is_apart_from(self, other: Self) -> bool {
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => false,
            (Self::Frame, Self::Frame)
            | (Self::ImageReadOnly, Self::ImageReadOnly)
            | (Self::ImageWritable, Self::ImageWritable) => false,
            (Self::Frame, Self::ImageReadOnly | Self::ImageWritable)
            | (Self::ImageReadOnly, Self::Frame | Self::ImageWritable)
            | (Self::ImageWritable, Self::Frame | Self::ImageReadOnly) => true,
        }
    }
}

/// A pointer dereference decoded into `base + index*stride + offset`, together
/// with the width of the access.
///
/// # Why not the address value id
///
/// The obvious encoding for `*ptr` is the `SsaVarId` of the address operand.
/// That is wrong for native code in both directions:
///
/// - It is **imprecise**: `[rbp-8]` and `[rbp-16]` compute two different
///   address values, so nothing relates them - even though they provably do not
///   overlap. Every stack slot looks unrelated to every other.
/// - It is **unstable**: the address is produced by a [`SsaOp::PtrAdd`], which
///   is a pure op, so GVN and LICM freely re-number and hoist it. The identity
///   of a memory cell would then depend on which optimization ran.
///
/// Decoding the address (via [`AddressWalk::normalize`](crate::analysis::address::AddressWalk::normalize))
/// fixes both: the cell is named by what it *is*, not by which instruction computed
/// it, and two accesses off a common base become comparable by offset.
///
/// # Soundness
///
/// Distinct `base` values are treated as **may-alias**. Two unrelated SSA
/// pointers can hold the same address, so nothing in the address expression
/// alone can rule that out; separating them requires a points-to oracle, which
/// is not available to a pure function on locations (see
/// [`pointsto`]). This is deliberately weaker than
/// the managed-code variants above, which key on object identity.
///
/// The absolute bases are the exception, and they are one because they are not
/// values: two addresses in one absolute space are numbers, so their extents
/// are directly comparable. An absolute address against a value base may alias,
/// since the value may hold that number, and the [`Symbol`](AddressBase::Symbol)
/// and [`Number`](AddressBase::Number) spaces may alias each other, since the
/// host's symbol space and the program's need not agree.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct IndirectLocation {
    /// What the address is measured from.
    pub base: AddressBase,
    /// Scaled index term, when the address carries one.
    pub index: Option<SsaVarId>,
    /// Stride applied to [`Self::index`], in bytes.
    ///
    /// Meaningless without an index, and therefore held at `0` whenever
    /// [`Self::index`] is `None` - see [`IndirectLocation::new`]. Keeping that
    /// invariant is what lets the derived `Eq`/`Hash` agree with
    /// [`must_alias`](Self::must_alias): otherwise two locations denoting the
    /// same cell could differ in a field neither alias rule consults, and be
    /// tracked as two separate memory versions.
    pub stride_bytes: u64,
    /// Constant displacement from [`Self::base`], in bits.
    pub offset_bits: i64,
    /// Width of the access in bits; `None` when the host reports no width.
    pub size_bits: Option<u32>,
    /// Address space the access is qualified by, or `None` for the target's
    /// default (flat) space. Distinct spaces never alias.
    pub address_space: Option<NonZeroU16>,
    /// Target pointer width this address was decoded against.
    ///
    /// Address arithmetic wraps here, so the width is part of what the offset
    /// *means*: `+4294967288` and `-8` are one displacement on a 32-bit target
    /// and two distinct ones on a 64-bit target.
    /// [`AddressWalk::normalize`](crate::analysis::address::AddressWalk::normalize)
    /// canonicalises every folded displacement into it, and
    /// `extents_overlap` needs it to notice an access
    /// that runs off the top of the address space and wraps to the bottom.
    ///
    /// Every location within one function shares this, so it does not fragment
    /// the derived `Eq`/`Hash`.
    pub ptr_size: PointerSize,
}

impl IndirectLocation {
    /// Builds a location, normalizing the stride so it is `0` whenever there is
    /// no index (see [`Self::stride_bytes`]).
    #[must_use]
    pub fn new(
        base: impl Into<AddressBase>,
        index: Option<SsaVarId>,
        stride_bytes: u64,
        offset_bits: i64,
        size_bits: Option<u32>,
        address_space: Option<NonZeroU16>,
        ptr_size: PointerSize,
    ) -> Self {
        Self {
            base: base.into(),
            index,
            stride_bytes: if index.is_some() { stride_bytes } else { 0 },
            offset_bits,
            size_bits,
            address_space,
            ptr_size,
        }
    }

    /// Returns the memory this access may touch, as `relation` places it
    /// ([`PointsTo::footprint`]): through a value, the bytes of what the
    /// value points into; at a stated address, those bytes of the image
    /// where `image` places them, and otherwise memory the function lets go
    /// of, which holds no byte of its own frame or allocations.
    ///
    /// Anywhere for an access in a qualified address space, one that wraps
    /// the address space, and one off a slot base, which the relation does
    /// not place.
    ///
    /// # Arguments
    ///
    /// * `relation` - The function's points-to relation.
    /// * `image` - The host's image.
    #[must_use]
    pub fn footprint(&self, relation: &PointsTo, image: Option<&dyn ImageData>) -> Footprint {
        if self.address_space.is_some() || self.crosses_wrap_boundary() {
            return Footprint::Anywhere;
        }
        // Its first byte from its base, and the byte past its last where no
        // index moves it.
        let first = self.offset_bits.div_euclid(8);
        let end = self
            .index
            .is_none()
            .then_some(self.size_bits)
            .flatten()
            .and_then(|size| {
                let end_bits = self.offset_bits.checked_add(i64::from(size))?;
                end_bits
                    .div_euclid(8)
                    .checked_add(i64::from(end_bits.rem_euclid(8) != 0))
            });
        match self.base {
            AddressBase::Value(pointer) => relation.footprint(pointer, first, end, self.ptr_size),
            AddressBase::Symbol | AddressBase::Number => {
                match image.and_then(|image| self.image_extent(image, None)) {
                    Some((address, len)) => Footprint::image(
                        i64::try_from(address).ok().zip(
                            address
                                .checked_add(len)
                                .and_then(|end| i64::try_from(end).ok()),
                        ),
                    ),
                    None => Footprint::ESCAPED,
                }
            }
            AddressBase::Local(_) | AddressBase::Argument(_) => Footprint::Anywhere,
        }
    }

    /// Returns where in `image` this access reads, as `(address, bytes)` in the
    /// host's symbol space, when it names a whole-byte extent of the image:
    /// for an indexed access, every byte an index from `index`'s least to
    /// its greatest value may move it over, a table read at a guarded index
    /// spanning the table.
    ///
    /// `None` for an access through a value, an indexed one `index` does not
    /// bound, one in a qualified address space, one of unknown or sub-byte
    /// width, one that wraps the address space, and a
    /// [`Number`](AddressBase::Number) address when the host cannot relate
    /// the program's integers to its symbols.
    ///
    /// # Arguments
    ///
    /// * `image` - The host's image, which also says how the two absolute
    ///   spaces relate.
    /// * `index` - The least and greatest value of [`Self::index`] where the
    ///   access is made; read only for an indexed access.
    #[must_use]
    pub fn image_extent(
        &self,
        image: &dyn ImageData,
        index: Option<(i64, i64)>,
    ) -> Option<(u64, u64)> {
        if self.address_space.is_some() || self.crosses_wrap_boundary() {
            return None;
        }
        let size_bits = self.size_bits?;
        if size_bits == 0 || size_bits % 8 != 0 || self.offset_bits % 8 != 0 {
            return None;
        }
        let bytes = self.offset_bits / 8;
        let (first, last) = match self.index {
            None => (bytes, bytes),
            Some(_) => {
                let (low, high) = index?;
                let stride = i64::try_from(self.stride_bytes).ok()?;
                (
                    bytes.checked_add(low.checked_mul(stride)?)?,
                    bytes.checked_add(high.checked_mul(stride)?)?,
                )
            }
        };
        let in_symbol_space = |offset: i64| match self.base {
            AddressBase::Symbol => Some(offset),
            AddressBase::Number => offset.checked_add(image.number_offset()?),
            AddressBase::Value(_) | AddressBase::Local(_) | AddressBase::Argument(_) => None,
        };
        let (first, last) = (in_symbol_space(first)?, in_symbol_space(last)?);
        let len = u64::try_from(last.checked_sub(first)?)
            .ok()?
            .checked_add(u64::from(size_bits / 8))?;
        let address = self.ptr_size.mask_unsigned(first.cast_unsigned());
        // An index moving the access across the top of the address space
        // reads at both ends of it, which no one extent says.
        let top = u128::from(self.ptr_size.mask_unsigned(u64::MAX)).saturating_add(1);
        (u128::from(address).saturating_add(u128::from(len)) <= top).then_some((address, len))
    }

    /// Returns where in `image` this access reads, as [`Self::image_extent`]
    /// does, where the image maps the whole of it ([`ImageData::mapped`]):
    /// storage the loader mapped, which no read of it faults on.
    ///
    /// # Arguments
    ///
    /// * `image` - The host's image.
    /// * `index` - The least and greatest value of [`Self::index`] where the
    ///   access is made; read only for an indexed access.
    #[must_use]
    pub fn placed_in_image(
        &self,
        image: &dyn ImageData,
        index: Option<(i64, i64)>,
    ) -> Option<(u64, u64)> {
        self.image_extent(image, index)
            .filter(|(address, extent)| image.mapped(*address, *extent))
    }

    /// Whether this access reads the first word of a symbol the host proves
    /// bound ([`Target::symbol_bound`]): storage the loader mapped before the
    /// function ran, at an address no read of a word there faults on.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function the access is in, whose facts the host reads.
    /// * `word_bits` - The target's word, in bits.
    #[must_use]
    pub fn bound_first_word<T: Target>(&self, ssa: &SsaFunction<T>, word_bits: i64) -> bool {
        let AddressBase::Value(base) = self.base else {
            return false;
        };
        self.index.is_none()
            && self.offset_bits == 0
            && self
                .size_bits
                .is_some_and(|bits| i64::from(bits) <= word_bits)
            && matches!(
                ssa.get_definition(base),
                Some(SsaOp::Const {
                    value: ConstValue::Symbol(symbol),
                    ..
                }) if T::symbol_bound(ssa.facts(), symbol)
            )
    }

    /// Whether no read of this access can fault: the first word of a bound
    /// symbol ([`Self::bound_first_word`]), or an unindexed access the image
    /// maps whole ([`Self::placed_in_image`]) - or, with no image at hand, one
    /// the function's facts hold mapped ([`Target::address_mapped`]).
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function the access is in.
    /// * `image` - The host's image, where it has one.
    /// * `word_bits` - The target's word, in bits.
    #[must_use]
    pub fn reads_mapped<T: Target>(
        &self,
        ssa: &SsaFunction<T>,
        image: Option<&dyn ImageData>,
        word_bits: i64,
    ) -> bool {
        if self.bound_first_word(ssa, word_bits) {
            return true;
        }
        if self.index.is_some() {
            return false;
        }
        match image {
            Some(image) => self.placed_in_image(image, None).is_some(),
            None => self
                .symbol_extent()
                .is_some_and(|(address, len)| T::address_mapped(ssa.facts(), address, len)),
        }
    }

    /// Returns where this unindexed access reads in the host's symbol space,
    /// as `(address, bytes)`, where its address is a symbol's: what
    /// [`Self::image_extent`] says of it without an image.
    #[must_use]
    pub fn symbol_extent(&self) -> Option<(u64, u64)> {
        if self.base != AddressBase::Symbol
            || self.index.is_some()
            || self.address_space.is_some()
            || self.crosses_wrap_boundary()
        {
            return None;
        }
        let size_bits = self.size_bits?;
        if size_bits == 0 || size_bits % 8 != 0 || self.offset_bits % 8 != 0 {
            return None;
        }
        let address = self
            .ptr_size
            .mask_unsigned((self.offset_bits / 8).cast_unsigned());
        Some((address, u64::from(size_bits / 8)))
    }

    /// Returns `true` when this access runs past the top of the address space.
    ///
    /// Displacements are canonicalised into the signed pointer-width range, so
    /// an access whose *end* exceeds that range wraps around to the bottom - and
    /// can then overlap a low-offset access that the linear extent comparison
    /// calls disjoint.
    pub fn crosses_wrap_boundary(&self) -> bool {
        let Some(size) = self.size_bits else {
            return false;
        };
        // Highest representable signed displacement, in bits. Beyond 64-bit
        // pointers this cannot overflow `i64` in a reachable way, and the shift
        // would, so treat those as never wrapping.
        let Some(max_bits) = 1i64
            .checked_shl(self.ptr_size.bits().saturating_sub(1))
            .and_then(|bytes| bytes.checked_mul(8))
        else {
            return false;
        };
        self.offset_bits
            .checked_add(i64::from(size))
            .is_none_or(|end| end > max_bits)
    }

    /// Returns `true` if the two accesses may touch overlapping memory.
    ///
    /// Tries three cheap disjointness proofs before the last-resort extent
    /// comparison: different address spaces never alias; different bases always
    /// may-alias (two SSA pointers can hold one address, and a value can hold an
    /// absolute one); and an index term that
    /// cannot be equated defeats offset reasoning, since the indices could take
    /// the same value. Only when space, base, and index all line up does it
    /// compare the `[offset, offset + size)` bit extents.
    ///
    /// # Arguments
    ///
    /// * `other` - The other decoded access to test against.
    ///
    /// # Returns
    ///
    /// `true` unless the two accesses are provably disjoint.
    #[must_use]
    pub fn may_alias(&self, other: &Self) -> bool {
        // Distinct address spaces are disjoint by construction: a segmented
        // access and a flat one at the same numeric offset name different
        // memory, so this holds even when everything else matches.
        if self.address_space != other.address_space {
            return false;
        }
        // Different roots: two distinct SSA pointers can hold the same
        // address, so they may alias. Two bases that are each storage of
        // their own cannot - which is both slots, and a slot against a
        // *stated* absolute address: the displacement of a symbol or a bare
        // integer is a link-time or program-stated address, and no such
        // address names a frame slot of the function being analysed
        // ([`AddressBase::Local`]). An address a *value* holds is not stated,
        // so it keeps aliasing the frame.
        if self.base != other.base {
            let slot_against_absolute =
                |a: AddressBase, b: AddressBase| a.is_slot() && b.is_absolute();
            return !((self.base.is_slot() && other.base.is_slot())
                || slot_against_absolute(self.base, other.base)
                || slot_against_absolute(other.base, self.base));
        }
        // A scaled index term whose contribution we cannot equate defeats any
        // offset reasoning - the indices could take the same value.
        if !self.index_matches(other) {
            return true;
        }
        self.extents_overlap(other)
    }

    /// Returns `true` if the two accesses provably name exactly one cell.
    ///
    /// Requires everything the extent comparison relies on to be known and equal:
    /// same address space, same base, matching index contribution, equal offset,
    /// and equal known widths. An unknown width can never prove equal extent, so
    /// a width-less access never must-aliases, not even against itself.
    ///
    /// # Arguments
    ///
    /// * `other` - The other decoded access to test against.
    ///
    /// # Returns
    ///
    /// `true` only when both accesses provably denote the same memory cell.
    #[must_use]
    pub fn must_alias(&self, other: &Self) -> bool {
        self.address_space == other.address_space
            && self.base == other.base
            && self.index_matches(other)
            && self.offset_bits == other.offset_bits
            // An unknown width cannot prove equal extent, so it never
            // must-aliases -- not even against itself.
            && matches!(
                (self.size_bits, other.size_bits),
                (Some(a), Some(b)) if a == b
            )
    }

    /// Returns `true` when both addresses carry the same scaled-index
    /// contribution, so their constant offsets are directly comparable.
    ///
    /// Equal index *value ids* with equal strides contribute equally whatever
    /// the runtime value. Anything else - different index values, different
    /// strides, or an index on only one side (it could be zero) - is not
    /// comparable.
    fn index_matches(&self, other: &Self) -> bool {
        match (self.index, other.index) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b && self.stride_bytes == other.stride_bytes,
            _ => false,
        }
    }

    /// Returns `true` when the half-open bit extents `[offset, offset + size)`
    /// intersect. An unknown width on either side overlaps anything.
    fn extents_overlap(&self, other: &Self) -> bool {
        let (Some(self_size), Some(other_size)) = (self.size_bits, other.size_bits) else {
            return true;
        };
        // An access that wraps off the top of the address space reappears at the
        // bottom, where the linear comparison below cannot see it. Refuse to
        // prove disjointness rather than prove it wrongly.
        if self.crosses_wrap_boundary() || other.crosses_wrap_boundary() {
            return true;
        }
        let (Some(self_end), Some(other_end)) = (
            self.offset_bits.checked_add(i64::from(self_size)),
            other.offset_bits.checked_add(i64::from(other_size)),
        ) else {
            return true;
        };
        self.offset_bits < other_end && other.offset_bits < self_end
    }
}

/// Represents an array index for array element memory locations.
///
/// The granularity of the index determines alias precision:
/// - A known constant index allows precise alias analysis (same index = must alias)
/// - A variable index conservatively assumes it may overlap with any other index
/// - Unknown represents a completely indeterminate index (may alias everything)
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ArrayIndex {
    /// A known constant index value. Two constants with the same value must alias.
    Constant(i64),
    /// A variable-driven index. Conservatively may overlap any other variable or unknown index.
    Variable(SsaVarId),
    /// Completely unknown index. May alias any other index (most conservative).
    Unknown,
}

impl ArrayIndex {
    /// Returns `true` if these indices may refer to the same element.
    ///
    /// A variable or unknown index may take any runtime value, so it overlaps
    /// anything; two constants overlap only when equal.
    ///
    /// # Arguments
    ///
    /// * `other` - The index to test against.
    ///
    /// # Returns
    ///
    /// `true` unless both indices are constants with different values.
    #[must_use]
    pub fn may_overlap(&self, other: &Self) -> bool {
        match (self, other) {
            // Unknown overlaps everything; Variable indices may overlap (conservative)
            (Self::Unknown | Self::Variable(_), _) | (_, Self::Unknown | Self::Variable(_)) => true,
            // Constants overlap iff equal
            (Self::Constant(i1), Self::Constant(i2)) => i1 == i2,
        }
    }

    /// Returns `true` if these indices must refer to the same element.
    ///
    /// Provable only for two equal constants or two identical index variables;
    /// any other pairing, including a variable against a constant, is not equal
    /// enough to prove.
    ///
    /// # Arguments
    ///
    /// * `other` - The index to test against.
    ///
    /// # Returns
    ///
    /// `true` only when the indices provably select the same element.
    #[must_use]
    pub fn must_equal(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Constant(i1), Self::Constant(i2)) => i1 == i2,
            (Self::Variable(v1), Self::Variable(v2)) => v1 == v2,
            _ => false,
        }
    }
}

/// A memory operation (load or store).
#[derive(Debug, Clone)]
pub enum MemoryOp<T: Target> {
    /// A memory load operation.
    Load {
        /// The memory location being loaded.
        location: MemoryLocation<T>,
        /// The SSA variable receiving the loaded value.
        dest: SsaVarId,
        /// Block containing this operation.
        block: usize,
        /// Instruction index within the block.
        instr: usize,
    },
    /// A memory store operation.
    Store {
        /// The memory location being stored to.
        location: MemoryLocation<T>,
        /// The SSA variable being stored.
        value: SsaVarId,
        /// Block containing this operation.
        block: usize,
        /// Instruction index within the block.
        instr: usize,
    },
    /// A read of a location that defines no value the location holds: a load
    /// whose result is computed from the bytes it reads (a broadcast of one
    /// element, a deinterleaving of segments, a masked read).
    Read {
        /// The memory location read.
        location: MemoryLocation<T>,
        /// Block containing this operation.
        block: usize,
        /// Instruction index within the block.
        instr: usize,
    },
    /// A memory operation that both reads and writes, or whose value operand is implicit.
    ReadWrite {
        /// The memory location being read and written.
        location: MemoryLocation<T>,
        /// Effect summary attached to this operation.
        effects: SsaEffects,
        /// Block containing this operation.
        block: usize,
        /// Instruction index within the block.
        instr: usize,
    },
    /// A write of a location that stores no value an operand holds and reads
    /// nothing ([`SsaEffectKind::Write`]): a coprocessor register stored to
    /// memory, a move into a special register whose memory effect is not
    /// modelled. What the location holds is unknown after it, and no store
    /// before it is observed by it.
    Write {
        /// The memory location written.
        location: MemoryLocation<T>,
        /// Effect summary attached to this operation.
        effects: SsaEffects,
        /// Block containing this operation.
        block: usize,
        /// Instruction index within the block.
        instr: usize,
    },
    /// A call whose host stated its effects: it reads and may write exactly
    /// these locations ([`CallEffects`](crate::ir::ops::CallEffects)), and no
    /// other memory. A call nobody described is a [`Self::Barrier`].
    Call {
        /// Each location the call touches, and how.
        accesses: Vec<(MemoryLocation<T>, AccessMode)>,
        /// The frame bytes the callee owns for the duration of the call
        /// ([`CallLocation::Frame`]), and
        /// how it treats them. What it does to them is this, whatever escaped
        /// memory it reaches otherwise: no object of the caller lies where a
        /// callee's parameters do.
        owned: Vec<(MemoryLocation<T>, AccessMode)>,
        /// Effect summary attached to this call.
        effects: SsaEffects,
        /// Block containing this operation.
        block: usize,
        /// Instruction index within the block.
        instr: usize,
    },
    /// A memory barrier, call, or opaque operation that constrains memory state.
    Barrier {
        /// Conservative memory location affected by the barrier.
        location: MemoryLocation<T>,
        /// Effect summary attached to this barrier.
        effects: SsaEffects,
        /// Block containing this operation.
        block: usize,
        /// Instruction index within the block.
        instr: usize,
    },
}

impl<T: Target> MemoryOp<T> {
    /// This operation as a volatile access makes it, with `effects`: a
    /// [`Self::ReadWrite`] of its location. A volatile access may be observed
    /// (an earlier store to its cell stays live) and may change what the cell
    /// holds (a device register's read), so no value is forwarded through it
    /// and it is no store a later one makes dead. A call or barrier stays as
    /// it is.
    #[must_use]
    pub fn made_volatile(self, effects: SsaEffects) -> Self {
        match self {
            Self::Load {
                location,
                block,
                instr,
                ..
            }
            | Self::Store {
                location,
                block,
                instr,
                ..
            }
            | Self::Read {
                location,
                block,
                instr,
            } => Self::ReadWrite {
                location,
                effects,
                block,
                instr,
            },
            other => other,
        }
    }

    /// Returns the one memory location this operation accesses, or `None` for
    /// a [`Self::Call`], which may access several ([`Self::accessed_locations`]).
    #[must_use]
    pub fn location(&self) -> Option<&MemoryLocation<T>> {
        match self {
            Self::Load { location, .. }
            | Self::Store { location, .. }
            | Self::Read { location, .. }
            | Self::ReadWrite { location, .. }
            | Self::Write { location, .. }
            | Self::Barrier { location, .. } => Some(location),
            Self::Call { .. } => None,
        }
    }

    /// Returns every location this operation reads or writes.
    pub fn accessed_locations(&self) -> impl Iterator<Item = &MemoryLocation<T>> {
        self.location()
            .into_iter()
            .chain(self.call_accesses().map(|(location, _)| location))
    }

    /// Returns the locations this operation may write: each one gets a new
    /// version here, as does every location it may alias.
    pub fn defined_locations(&self) -> impl Iterator<Item = &MemoryLocation<T>> {
        let single = match self {
            Self::Store { location, .. }
            | Self::ReadWrite { location, .. }
            | Self::Write { location, .. }
            | Self::Barrier { location, .. } => Some(location),
            Self::Load { .. } | Self::Read { .. } | Self::Call { .. } => None,
        };
        single.into_iter().chain(
            self.call_accesses()
                .filter(|(_, access)| access.writes())
                .map(|(location, _)| location),
        )
    }

    /// Returns the locations this operation may read, so that a store to any
    /// location that may alias one of them is observed here.
    pub fn read_locations(&self) -> impl Iterator<Item = &MemoryLocation<T>> {
        let single = match self {
            Self::Load { location, .. }
            | Self::Read { location, .. }
            | Self::ReadWrite { location, .. }
            | Self::Barrier { location, .. } => Some(location),
            Self::Store { .. } | Self::Write { .. } | Self::Call { .. } => None,
        };
        single.into_iter().chain(
            self.call_accesses()
                .filter(|(_, access)| access.reads())
                .map(|(location, _)| location),
        )
    }

    /// Returns every location a described call touches, owned bytes
    /// included, and how; nothing for any other operation.
    fn call_accesses(&self) -> impl Iterator<Item = &(MemoryLocation<T>, AccessMode)> {
        let (accesses, owned) = match self {
            Self::Call {
                accesses, owned, ..
            } => (accesses.as_slice(), owned.as_slice()),
            _ => (&[][..], &[][..]),
        };
        accesses.iter().chain(owned)
    }

    /// Returns the block index containing this operation.
    #[must_use]
    pub fn block(&self) -> usize {
        match self {
            Self::Load { block, .. }
            | Self::Store { block, .. }
            | Self::Read { block, .. }
            | Self::ReadWrite { block, .. }
            | Self::Write { block, .. }
            | Self::Call { block, .. }
            | Self::Barrier { block, .. } => *block,
        }
    }

    /// Returns the instruction index within the block.
    #[must_use]
    pub fn instr(&self) -> usize {
        match self {
            Self::Load { instr, .. }
            | Self::Store { instr, .. }
            | Self::Read { instr, .. }
            | Self::ReadWrite { instr, .. }
            | Self::Write { instr, .. }
            | Self::Call { instr, .. }
            | Self::Barrier { instr, .. } => *instr,
        }
    }

    /// Returns `true` if this is a store operation.
    #[must_use]
    pub fn is_store(&self) -> bool {
        matches!(self, Self::Store { .. })
    }

    /// Returns `true` if this operation defines a new memory version.
    #[must_use]
    pub fn defines_memory(&self) -> bool {
        self.defined_locations().next().is_some()
    }

    /// Returns `true` if this is a load operation.
    #[must_use]
    pub fn is_load(&self) -> bool {
        matches!(self, Self::Load { .. })
    }

    /// Returns the detailed effect summary for classified effectful operations.
    #[must_use]
    pub fn effects(&self) -> Option<SsaEffects> {
        match self {
            Self::ReadWrite { effects, .. }
            | Self::Write { effects, .. }
            | Self::Barrier { effects, .. }
            | Self::Call { effects, .. } => Some(*effects),
            Self::Load { .. } | Self::Store { .. } | Self::Read { .. } => None,
        }
    }
}

/// A phi node for memory locations.
///
/// Memory phi nodes are placed at control flow merge points where different
/// memory versions from different predecessors need to be merged.
#[derive(Debug, Clone)]
pub struct MemoryPhi<T: Target> {
    /// The memory location this phi node is for.
    pub location: MemoryLocation<T>,
    /// The result version number produced by this phi.
    pub result_version: u32,
    /// The operands from each predecessor.
    pub operands: Vec<MemoryPhiOperand>,
}

impl<T: Target> MemoryPhi<T> {
    /// Creates a new memory phi node.
    #[must_use]
    pub fn new(location: MemoryLocation<T>, result_version: u32) -> Self {
        Self {
            location,
            result_version,
            operands: Vec::new(),
        }
    }

    /// Adds an operand from a predecessor block.
    pub fn add_operand(&mut self, predecessor: usize, version: u32) {
        self.operands.push(MemoryPhiOperand {
            predecessor,
            version,
        });
    }

    /// Returns the operand from a specific predecessor, if present.
    #[must_use]
    pub fn operand_from(&self, predecessor: usize) -> Option<&MemoryPhiOperand> {
        self.operands
            .iter()
            .find(|op| op.predecessor == predecessor)
    }
}

/// An operand of a memory phi node, connecting a predecessor block to a memory version.
///
/// Memory phi operands work like regular SSA phi operands: at a control flow merge,
/// each predecessor contributes its current memory version for the location.
#[derive(Debug, Clone)]
pub struct MemoryPhiOperand {
    /// The predecessor block whose memory version this operand represents.
    pub predecessor: usize,
    /// The memory version number from that predecessor at the time of the merge.
    pub version: u32,
}

/// Memory version identifier.
///
/// Combines a memory location with a version number to uniquely identify
/// a specific "value" of that memory location.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MemoryVersion<T: Target> {
    /// The memory location.
    pub location: MemoryLocation<T>,
    /// The version number.
    pub version: u32,
}

impl<T: Target> MemoryVersion<T> {
    /// Creates a new memory version.
    #[must_use]
    pub fn new(location: MemoryLocation<T>, version: u32) -> Self {
        Self { location, version }
    }
}

/// Definition site for a memory version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryDefSite {
    /// Defined at function entry (initial version before any stores).
    ///
    /// Says nothing about the contents: an earlier function may have written
    /// the location, and a caller may have handed any value in.
    Entry,
    /// Defined by the image: the location lies in a region the host reports
    /// read-only, so its contents are the image's bytes and nothing the
    /// function does changes them.
    ///
    /// The one version such a location has. Unlike [`Self::Entry`] it states
    /// the contents, which is what lets a load of it fold to a constant.
    Image,
    /// Defined by a store instruction at a specific block and instruction index.
    Store {
        /// The block containing the store.
        block: usize,
        /// The instruction index within the block.
        instr: usize,
    },
    /// Defined by a memory phi node at a control flow merge point.
    Phi {
        /// The block containing the phi node.
        block: usize,
    },
    /// Defined by entry into an exception handler or filter.
    ///
    /// The runtime dispatches here from *somewhere* inside the protected
    /// region, not from the end of it, so a store the region performs may or
    /// may not have run. This version stands for "unknown, because the region
    /// was interrupted", and it is deliberately not a [`Self::Store`]: the
    /// forwarding in `passes::memory` matches only `Store`, so it declines to
    /// forward across an exceptional edge by construction rather than by a
    /// check somebody has to remember to write.
    ExceptionEntry {
        /// The handler or filter entry block.
        block: usize,
    },
}

/// One step of the dominator-tree walk in
/// [`MemorySsa::rename_memory_versions`].
///
/// The walk is an explicit-stack DFS rather than recursion, so the scope
/// restore that recursion would get for free is modelled as its own step.
enum RenameStep {
    /// Rename this block, then descend into its dominator-tree children.
    Enter(usize),
    /// Leaving a block's dominator subtree: pop one version for each location
    /// listed (by interned id), restoring the rename state that block's
    /// siblings must observe.
    Exit(Vec<usize>),
}

/// The live version of every interned location while the dominator tree is
/// walked: a stack per location, the top being the version that reaches the
/// block being renamed.
#[derive(Debug, Default)]
struct VersionStacks {
    /// One stack per interned location id.
    stacks: Vec<Vec<u32>>,
}

impl VersionStacks {
    /// Creates empty stacks for `locations` interned locations.
    fn new(locations: usize) -> Self {
        Self {
            stacks: vec![Vec::new(); locations],
        }
    }

    /// Makes `version` the live version of `location`.
    fn push(&mut self, location: usize, version: u32) {
        if let Some(stack) = self.stacks.get_mut(location) {
            stack.push(version);
        }
    }

    /// Restores the version that was live before the last [`Self::push`] of
    /// `location`.
    fn pop(&mut self, location: usize) {
        if let Some(stack) = self.stacks.get_mut(location) {
            stack.pop();
        }
    }

    /// The version of `location` that reaches the block being renamed.
    fn live(&self, location: usize) -> Option<u32> {
        self.stacks.get(location)?.last().copied()
    }
}

/// A dense table of memory versions, one cell per `(block, location)`.
///
/// Renaming records the version of every location at every block's entry and
/// exit, so the table holds `blocks x locations` cells by construction. Cells
/// are addressed by an interned location id, not by the location: the lookup
/// is a multiply and an add, where keying a map by location hashes a ~48 byte
/// value for every one of those cells.
#[derive(Debug, Default)]
struct VersionTable {
    /// Locations per row; a row holds one block.
    width: usize,
    /// Row-major cells, [`Self::ABSENT`] where the location had no version.
    cells: Vec<u32>,
}

impl VersionTable {
    /// The cell value of a location that has no live version in the block.
    const ABSENT: u32 = u32::MAX;

    /// Creates a table with every cell absent.
    fn new(blocks: usize, locations: usize) -> Self {
        Self {
            width: locations,
            cells: vec![Self::ABSENT; blocks.saturating_mul(locations)],
        }
    }

    /// The version `location` had in `block`, when it had one.
    fn get(&self, block: usize, location: usize) -> Option<u32> {
        if location >= self.width {
            return None;
        }
        let cell = block.checked_mul(self.width)?.checked_add(location)?;
        self.cells.get(cell).copied().filter(|&v| v != Self::ABSENT)
    }

    /// Records, for `block`, the live version of every location that has one.
    fn record(&mut self, block: usize, live: &VersionStacks) {
        let start = block.saturating_mul(self.width);
        let Some(row) = self.cells.get_mut(start..start.saturating_add(self.width)) else {
            return;
        };
        for (cell, stack) in row.iter_mut().zip(&live.stacks) {
            if let Some(&version) = stack.last() {
                *cell = version;
            }
        }
    }
}

/// A location's new version immediately after one instruction completes.
#[derive(Debug)]
struct VersionWrite {
    /// Interned location whose state changes.
    location: usize,
    /// Instruction making the change within its block.
    instruction: usize,
    /// Version defined by that instruction.
    version: usize,
}

/// A block's instruction extent and classified memory-operation range.
#[derive(Debug)]
struct MemoryBlock {
    /// One past the final instruction, also the block-exit query position.
    instructions: usize,
    /// Classified operations in the shared operation vector.
    operations: Range<usize>,
}

/// Memory SSA representation.
///
/// This structure tracks versioned memory locations throughout a function,
/// enabling precise tracking of memory state for analysis.
///
/// Every location is interned to its index in the first-appearance order while
/// the operations are identified. Everything keyed by location afterwards (the
/// definitions, the live versions during renaming, the entry and exit versions)
/// is a dense vector indexed by that id, and only the public lookups by
/// [`MemoryLocation`] hash one, once per call.
#[derive(Debug)]
pub struct MemorySsa<T: Target> {
    /// Memory phi nodes at each block, indexed by block.
    memory_phis: Vec<Vec<MemoryPhi<T>>>,

    /// The interned location id of each phi in [`Self::memory_phis`], in the
    /// same order, so renaming never looks a phi's location up.
    ///
    /// Phis are placed one location at a time in id order, at most one per
    /// block and location, so each block's row ascends strictly and
    /// [`Self::memory_phi`] finds a location's phi by binary search.
    phi_locations: Vec<Vec<usize>>,

    /// Definition site of each memory version, indexed by interned location id
    /// and then by version number. A location's versions are numbered in the
    /// order they are defined, so the position in its row is the number.
    definitions: Vec<Vec<MemoryDefSite>>,

    /// Memory version at block entry.
    entry_versions: VersionTable,

    /// Memory version at block exit.
    exit_versions: VersionTable,

    /// Instruction-local versions, derived only when an observation requests them.
    write_versions: OnceLock<Vec<Vec<VersionWrite>>>,

    /// Interning table: location to its index in [`Self::ordered_locations`].
    location_ids: FxHashMap<MemoryLocation<T>, usize>,

    /// All identified memory operations, in instruction order.
    operations: Vec<MemoryOp<T>>,

    /// The operations and instruction extent of each block.
    block_operations: Vec<MemoryBlock>,

    /// All unique memory locations in the function.
    locations: FxHashSet<MemoryLocation<T>>,
    /// The same locations in first-appearance order.
    ///
    /// `locations` is a hash set, so its iteration order follows the hasher
    /// rather than the function. Version numbers and the order availability is
    /// seeded both derive from this list instead, and the downstream similarity
    /// pipeline is content-addressed - it requires byte-identical optimized IR.
    /// This list is built from `operations`, which is instruction order.
    ordered_locations: Vec<MemoryLocation<T>>,

    /// Interned ids of the locations in [`MemoryRegion::ImageReadOnly`]: never
    /// versioned by a store or a barrier, and defined by
    /// [`MemoryDefSite::Image`].
    immutable: BitSet,

    /// The region each interned location lies in, by its id - the same answer
    /// [`MemoryLocation::region`] gives, kept because the invalidation half
    /// asks it of every (candidate, defined) pair and the host's context is
    /// not in hand there.
    regions: Vec<MemoryRegion>,

    /// The anchors of the function's own frame, when no address of it leaves
    /// the function ([`FrameEscape`]) and some operation touches escaped
    /// memory: its cells are then none of the memory the program let escape
    /// ([`Self::may_alias`]).
    private_frame: Option<FrameAnchors>,

    /// The memory each interned location may touch
    /// ([`MemoryLocation::footprint`]), by id; empty where no footprint can
    /// set two locations apart ([`Self::footprints_may_separate`]), which
    /// places nothing apart.
    footprints: Vec<Footprint>,

    /// The budget the build ran out of, when it did. Such a structure is empty.
    exhaustion: Option<Exhaustion>,
}

/// Where the memory operations of one function at one pointer width read and
/// write: each classified into the location it touches ([`Self::classify`]),
/// the one [`MemorySsa`] versions.
#[derive(Debug, Clone, Copy)]
pub struct MemoryLocator<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// The target pointer width addresses are read at.
    ptr_size: PointerSize,
}

impl<'a, T: Target> MemoryLocator<'a, T> {
    /// Locates the memory operations of `ssa` at `ptr_size`.
    #[must_use]
    pub const fn new(ssa: &'a SsaFunction<T>, ptr_size: PointerSize) -> Self {
        Self { ssa, ptr_size }
    }

    /// Classifies an SSA operation as a memory operation, if applicable.
    ///
    /// Public because the classification - not the versioning built on top of
    /// it - is what a consumer needs to ask "does this operation touch memory,
    /// and which cell". Everything whose effect is not modelled classifies to a
    /// [`MemoryOp::Barrier`], [`MemoryOp::ReadWrite`] or [`MemoryOp::Write`] over
    /// [`MemoryLocation::Unknown`], which may-aliases every location, so a
    /// consumer that consults [`MemoryLocation::may_alias`] is conservative by
    /// construction.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to classify.
    /// * `block` / `instr` - Position recorded on the returned [`MemoryOp`].
    ///
    /// # Returns
    ///
    /// The classified memory operation, or `None` when the operation touches no
    /// memory. A volatile access classifies as [`MemoryOp::made_volatile`]
    /// makes it.
    #[must_use]
    pub fn classify(&self, op: &SsaOp<T>, block: usize, instr: usize) -> Option<MemoryOp<T>> {
        let classified = if let SsaOp::Call(call) = op
            && let Some(stated) = &call.effects
        {
            let args = &call.args;
            let effects = op.effects();
            // Its memory unknown, or running code that may touch any:
            // the barrier every undescribed call is.
            let Some(accesses) = stated.accesses() else {
                return Some(MemoryOp::Barrier {
                    location: MemoryLocation::Unknown,
                    effects,
                    block,
                    instr,
                });
            };
            if accesses.is_empty() {
                return None;
            }
            let (owned, accesses) = accesses.iter().partition::<Vec<_>, _>(|access| {
                matches!(access.location, CallLocation::Frame { .. })
            });
            let located = |accesses: Vec<&CallAccess<T>>| {
                accesses
                    .into_iter()
                    .map(|access| (self.call_location(access, args), access.access))
                    .collect()
            };
            Some(MemoryOp::Call {
                accesses: located(accesses),
                owned: located(owned),
                effects,
                block,
                instr,
            })
        } else {
            match op {
                SsaOp::LoadField {
                    dest,
                    object,
                    field,
                    ..
                } => {
                    let location = MemoryLocation::InstanceField(*object, field.clone());
                    Some(MemoryOp::Load {
                        location,
                        dest: *dest,
                        block,
                        instr,
                    })
                }
                SsaOp::StoreField {
                    object,
                    field,
                    value,
                    ..
                } => {
                    let location = MemoryLocation::InstanceField(*object, field.clone());
                    Some(MemoryOp::Store {
                        location,
                        value: *value,
                        block,
                        instr,
                    })
                }
                SsaOp::LoadStaticField { dest, field, .. } => {
                    let location = MemoryLocation::StaticField(field.clone());
                    Some(MemoryOp::Load {
                        location,
                        dest: *dest,
                        block,
                        instr,
                    })
                }
                SsaOp::StoreStaticField { field, value, .. } => {
                    let location = MemoryLocation::StaticField(field.clone());
                    Some(MemoryOp::Store {
                        location,
                        value: *value,
                        block,
                        instr,
                    })
                }
                SsaOp::LoadElement {
                    dest, array, index, ..
                } => {
                    let idx = self.resolve_array_index(*index);
                    let location = MemoryLocation::ArrayElement(*array, idx);
                    Some(MemoryOp::Load {
                        location,
                        dest: *dest,
                        block,
                        instr,
                    })
                }
                SsaOp::StoreElement {
                    array,
                    index,
                    value,
                    ..
                } => {
                    let idx = self.resolve_array_index(*index);
                    let location = MemoryLocation::ArrayElement(*array, idx);
                    Some(MemoryOp::Store {
                        location,
                        value: *value,
                        block,
                        instr,
                    })
                }
                SsaOp::LoadIndirect {
                    dest,
                    addr,
                    value_type,
                    access: space,
                } => {
                    let location = self.indirect_location(*addr, value_type, space.address_space);
                    Some(MemoryOp::Load {
                        location,
                        dest: *dest,
                        block,
                        instr,
                    })
                }
                SsaOp::StoreIndirect {
                    addr,
                    value,
                    value_type,
                    access: space,
                } => {
                    let location = self.indirect_location(*addr, value_type, space.address_space);
                    Some(MemoryOp::Store {
                        location,
                        value: *value,
                        block,
                        instr,
                    })
                }
                // A whole vector moves as a value of its type does.
                SsaOp::VectorLoad {
                    dest,
                    addr,
                    vector_type,
                    access: space,
                } => Some(MemoryOp::Load {
                    location: self.indirect_location(*addr, vector_type, space.address_space),
                    dest: *dest,
                    block,
                    instr,
                }),
                SsaOp::VectorStore {
                    addr,
                    value,
                    vector_type,
                    access: space,
                } => Some(MemoryOp::Store {
                    location: self.indirect_location(*addr, vector_type, space.address_space),
                    value: *value,
                    block,
                    instr,
                }),
                // The bytes a vector read computes its lanes from, and a vector
                // write that writes them in a layout of its own or only where its
                // mask says.
                SsaOp::VectorBroadcastLoad {
                    addr,
                    vector_type,
                    access: space,
                    ..
                } => Some(MemoryOp::Read {
                    location: self.indirect_extent(
                        *addr,
                        T::vector_shape(vector_type).map(|shape| shape.lane_bits),
                        space.address_space,
                    ),
                    block,
                    instr,
                }),
                // One segment is the vector it loads.
                SsaOp::VectorSegmentLoad {
                    dests,
                    base,
                    mask: None,
                    vector_type,
                    segments: 1,
                    access: space,
                    ..
                } if dests.len() == 1 => Some(MemoryOp::Load {
                    location: self.indirect_location(*base, vector_type, space.address_space),
                    dest: dests.first().copied()?,
                    block,
                    instr,
                }),
                SsaOp::VectorSegmentLoad {
                    base,
                    vector_type,
                    segments,
                    access: space,
                    ..
                } => Some(MemoryOp::Read {
                    location: self.indirect_extent(
                        *base,
                        T::bit_width(vector_type).and_then(|bits| bits.checked_mul(*segments)),
                        space.address_space,
                    ),
                    block,
                    instr,
                }),
                // A masked read, and an expanding one, which reads the elements
                // its mask enables one after another, read within the vector's
                // extent at the address.
                SsaOp::VectorMaskedLoad {
                    addr,
                    vector_type,
                    access: space,
                    ..
                }
                | SsaOp::VectorPackLoad {
                    addr,
                    vector_type,
                    access: space,
                    ..
                } => Some(MemoryOp::Read {
                    location: self.indirect_location(*addr, vector_type, space.address_space),
                    block,
                    instr,
                }),
                // One segment is the vector it stores.
                SsaOp::VectorSegmentStore {
                    base,
                    values,
                    mask: None,
                    vector_type,
                    segments: 1,
                    access: space,
                    ..
                } if values.len() == 1 => Some(MemoryOp::Store {
                    location: self.indirect_location(*base, vector_type, space.address_space),
                    value: values.first().copied()?,
                    block,
                    instr,
                }),
                SsaOp::VectorSegmentStore {
                    base,
                    vector_type,
                    segments,
                    access: space,
                    ..
                } => Some(MemoryOp::ReadWrite {
                    location: self.indirect_extent(
                        *base,
                        T::bit_width(vector_type).and_then(|bits| bits.checked_mul(*segments)),
                        space.address_space,
                    ),
                    effects: op.effects(),
                    block,
                    instr,
                }),
                // So do a masked write and a compressing one.
                SsaOp::VectorMaskedStore {
                    addr,
                    vector_type,
                    access: space,
                    ..
                }
                | SsaOp::VectorPackStore {
                    addr,
                    vector_type,
                    access: space,
                    ..
                } => Some(MemoryOp::ReadWrite {
                    location: self.indirect_location(*addr, vector_type, space.address_space),
                    effects: op.effects(),
                    block,
                    instr,
                }),
                // A cache operation on one line reads the line its address names,
                // or writes it where it zeroes or discards the line: the bytes at
                // and around the address, in a size the implementation fixes and
                // no operand states. A width left unknown overlaps every access
                // through the same base, below the address as well as above it.
                // Read as memory nothing can place, `clflush [rdi+32]` kept every
                // register save of its function alive.
                SsaOp::SystemOp(data)
                    if matches!(
                        data.kind,
                        SystemOpKind::CacheMaintenance(line) if line.names_a_line()
                    ) && !data.inputs.is_empty() =>
                {
                    let effects = op.effects();
                    let address = data.inputs.first().copied()?;
                    let location = self.indirect_extent(address, None, None);
                    Some(match effects.kind {
                        SsaEffectKind::Read => MemoryOp::Read {
                            location,
                            block,
                            instr,
                        },
                        _ => MemoryOp::ReadWrite {
                            location,
                            effects,
                            block,
                            instr,
                        },
                    })
                }
                // A move to or from a register the machine holds reads no memory.
                // One that writes the register may decide what memory holds at an
                // address (a translation base), which is a write nothing places:
                // `msr basepri, r3` read as a read of every cell kept each register
                // save of a Cortex-M critical section alive.
                SsaOp::SystemOp(data)
                    if matches!(
                        data.kind,
                        SystemOpKind::ControlRegister(register) if !register.moves_memory()
                    ) =>
                {
                    let effects = op.effects();
                    match effects.kind {
                        SsaEffectKind::Write | SsaEffectKind::ReadWrite => Some(MemoryOp::Write {
                            location: MemoryLocation::Unknown,
                            effects,
                            block,
                            instr,
                        }),
                        _ => None,
                    }
                }
                _ => {
                    let effects = op.effects();
                    match effects.kind {
                        SsaEffectKind::ReadWrite | SsaEffectKind::Atomic => {
                            Some(MemoryOp::ReadWrite {
                                location: MemoryLocation::Unknown,
                                effects,
                                block,
                                instr,
                            })
                        }
                        SsaEffectKind::Fence | SsaEffectKind::Call if !effects.is_pure() => {
                            Some(MemoryOp::Barrier {
                                location: MemoryLocation::Unknown,
                                effects,
                                block,
                                instr,
                            })
                        }
                        SsaEffectKind::Opaque
                            if !effects.is_pure()
                                && !matches!(effects.memory, MemoryEffectLocation::None) =>
                        {
                            Some(MemoryOp::Barrier {
                                location: MemoryLocation::Unknown,
                                effects,
                                block,
                                instr,
                            })
                        }
                        // A read without a result (a cache clean or flush) still
                        // observes the memory it reads.
                        SsaEffectKind::Read => Some(op.dest().map_or(
                            MemoryOp::Read {
                                location: MemoryLocation::Unknown,
                                block,
                                instr,
                            },
                            |dest| MemoryOp::Load {
                                location: MemoryLocation::Unknown,
                                dest,
                                block,
                                instr,
                            },
                        )),
                        SsaEffectKind::Write => Some(MemoryOp::Write {
                            location: MemoryLocation::Unknown,
                            effects,
                            block,
                            instr,
                        }),
                        SsaEffectKind::Pure
                        | SsaEffectKind::RuntimeDefined
                        | SsaEffectKind::Fence
                        | SsaEffectKind::Call
                        | SsaEffectKind::Opaque => None,
                    }
                }
            }
        };
        if op.is_volatile() {
            classified.map(|classified| classified.made_volatile(op.effects()))
        } else {
            classified
        }
    }

    /// The memory location a call's stated access names.
    ///
    /// A named field is that static field; escaped memory is
    /// [`MemoryLocation::Escaped`]; frame bytes are the cells an access through
    /// the entry stack pointer names; memory an argument points to is what that
    /// address names, as a load or store through it would see it
    /// ([`MemoryLocator::pointee_location`]).
    fn call_location(&self, access: &CallAccess<T>, args: &[SsaVarId]) -> MemoryLocation<T> {
        match &access.location {
            CallLocation::Field(field) => MemoryLocation::StaticField(field.clone()),
            CallLocation::Escaped => MemoryLocation::Escaped,
            // The bytes an indirect access through the value they are measured
            // from would name at that offset: the entry stack pointer, or the
            // one alignment's result. With neither to measure from, no frame is
            // placed and they may be anywhere.
            CallLocation::Frame {
                origin,
                offset,
                bytes,
            } => self
                .ssa
                .entry_stack_pointer()
                .and_then(|entry| match origin {
                    FrameOrigin::Entry => Some(entry),
                    FrameOrigin::Alignment => AddressWalk::new(self.ssa, self.ptr_size)
                        .frame_anchor(entry)
                        .map(|(anchor, _)| anchor),
                })
                .zip(offset.checked_mul(8))
                .zip(
                    bytes
                        .checked_mul(8)
                        .and_then(|bits| u32::try_from(bits).ok()),
                )
                .map_or(
                    MemoryLocation::Unknown,
                    |((base, offset_bits), size_bits)| {
                        MemoryLocation::Indirect(IndirectLocation::new(
                            base,
                            None,
                            0,
                            offset_bits,
                            Some(size_bits),
                            None,
                            self.ptr_size,
                        ))
                    },
                ),
            CallLocation::Pointee { extent, .. } => {
                access
                    .pointer_argument(args)
                    .map_or(MemoryLocation::Unknown, |addr| {
                        let extent = extent.at_call(args, |count| self.ssa.const_u64(count));
                        self.pointee_location(addr, extent)
                    })
            }
        }
    }

    /// The memory an address names, `extent` wide.
    ///
    /// An address taken of a field names that field: a by-reference argument
    /// built with `LoadStaticFieldAddr` is the static field itself. Any other
    /// address is decoded as an indirect access through it would be, with the
    /// extent's width when it states one and no width - overlapping whatever
    /// lies from the address on - when it does not.
    fn pointee_location(&self, addr: SsaVarId, extent: PointeeExtent) -> MemoryLocation<T> {
        match self.ssa.get_definition(addr) {
            Some(SsaOp::LoadStaticFieldAddr { field, .. }) => {
                return MemoryLocation::StaticField(field.clone());
            }
            Some(SsaOp::LoadFieldAddr { object, field, .. }) => {
                return MemoryLocation::InstanceField(*object, field.clone());
            }
            _ => {}
        }
        let size_bits = match extent {
            PointeeExtent::Bytes(bytes) => bytes
                .checked_mul(8)
                .and_then(|bits| u32::try_from(bits).ok()),
            PointeeExtent::InArgument(_) | PointeeExtent::Unbounded => None,
        };
        let address = AddressWalk::new(self.ssa, self.ptr_size).normalize(addr);
        MemoryLocation::Indirect(IndirectLocation::new(
            address.base,
            address.index,
            address.stride_bytes,
            address.offset_bits,
            size_bits,
            None,
            self.ptr_size,
        ))
    }

    /// Decodes an indirect access into an offset-aware memory location.
    ///
    /// The address is normalized to `base + index*stride + offset` rather than
    /// keyed on `addr` itself, so the resulting cell identity survives the
    /// address computation being re-numbered or hoisted. See
    /// [`IndirectLocation`].
    fn indirect_location(
        &self,
        addr: SsaVarId,
        value_type: &T::Type,
        address_space: Option<NonZeroU16>,
    ) -> MemoryLocation<T> {
        let address = AddressWalk::new(self.ssa, self.ptr_size).normalize(addr);
        MemoryLocation::Indirect(IndirectLocation::new(
            address.base,
            address.index,
            address.stride_bytes,
            address.offset_bits,
            T::bit_width(value_type),
            address_space,
            self.ptr_size,
        ))
    }

    /// The `bits` from `addr` in `address_space`: the extent a vector access
    /// touches, whatever value its lanes make of it.
    fn indirect_extent(
        &self,
        addr: SsaVarId,
        bits: Option<u32>,
        address_space: Option<NonZeroU16>,
    ) -> MemoryLocation<T> {
        let address = AddressWalk::new(self.ssa, self.ptr_size).normalize(addr);
        MemoryLocation::Indirect(IndirectLocation::new(
            address.base,
            address.index,
            address.stride_bytes,
            address.offset_bits,
            bits,
            address_space,
            self.ptr_size,
        ))
    }

    /// Resolves an array index to an [`ArrayIndex`] abstraction, folding a
    /// constant index so that distinct constant elements stop may-aliasing.
    fn resolve_array_index(&self, index_var: SsaVarId) -> ArrayIndex {
        match self.ssa.const_i64(index_var) {
            Some(index) => ArrayIndex::Constant(index),
            None => ArrayIndex::Variable(index_var),
        }
    }
}

impl<T: Target> MemorySsa<T> {
    /// Creates an empty Memory SSA structure.
    #[must_use]
    pub fn new() -> Self {
        Self {
            memory_phis: Vec::new(),
            phi_locations: Vec::new(),
            definitions: Vec::new(),
            entry_versions: VersionTable::default(),
            exit_versions: VersionTable::default(),
            write_versions: OnceLock::new(),
            location_ids: FxHashMap::default(),
            operations: Vec::new(),
            block_operations: Vec::new(),
            locations: FxHashSet::default(),
            ordered_locations: Vec::new(),
            immutable: BitSet::new(0),
            regions: Vec::new(),
            private_frame: None,
            footprints: Vec::new(),
            exhaustion: None,
        }
    }

    /// Builds Memory SSA from an SSA function, with the host's facts: its
    /// pointer width, which canonicalises folded address displacements; its
    /// image, which places locations in their [`MemoryRegion`]; and its work
    /// budget.
    ///
    /// This performs the full Memory SSA construction: it identifies every
    /// memory operation, places memory phi nodes at dominance frontiers, and
    /// renames memory versions by a dominator tree traversal. Address
    /// arithmetic wraps at the pointer width, and the model cannot tell a
    /// sign-extended `-8` from a zero-extended `0xFFFF_FFF8` without it
    /// ([`AddressWalk::normalize`](crate::analysis::address::AddressWalk::normalize)).
    ///
    /// A location in [`MemoryRegion::ImageReadOnly`] is never versioned by a
    /// store or a barrier and its one version is [`MemoryDefSite::Image`] (see
    /// the module documentation). Past the
    /// [`MemorySsaCells`](Budget::MemorySsaCells) budget for block/location cells
    /// plus may-alias instruction definitions, the structure is empty
    /// and records the [`Exhaustion`]: every alias query then falls back to the
    /// conservative answer and a consumer finds nothing to do.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to analyze.
    /// * `context` - The host's machine, image and budget.
    ///
    /// # Returns
    ///
    /// A complete Memory SSA representation, or an empty one that says why.
    #[must_use]
    pub fn build(ssa: &SsaFunction<T>, context: &AnalysisContext<'_>) -> Self {
        let eh = SsaCfg::from_ssa(ssa);
        Self::build_with_cfg(&eh, context, None, || {
            pointsto::analyze_function(ssa, context.budget(), context.models())
        })
    }

    /// Builds canonical memory state while reusing the caller's CFG and optional dominance.
    ///
    /// The function is borrowed from the CFG, so its block coordinates cannot
    /// name a different body. Ordinary construction computes dominance only
    /// after checking the configured retained-cell limit. Dependency analysis
    /// supplies its existing tree.
    ///
    /// `points_to` gives the function's points-to relation under `context`,
    /// asked only where a footprint can set two locations apart: solved for
    /// the build, or the one a function's analyses hold
    /// ([`FunctionAnalyses::memory_ssa`](crate::analysis::cache::FunctionAnalyses::memory_ssa)).
    pub fn build_with_cfg<R: Borrow<PointsTo>>(
        eh: &SsaCfg<'_, T>,
        context: &AnalysisContext<'_>,
        dominators: Option<&DominatorTree>,
        points_to: impl FnOnce() -> R,
    ) -> Self {
        let ssa = eh.ssa();
        let ptr_size = context.machine().ptr_size;
        let mut mem_ssa = Self::new();

        // Phase 1: Identify all memory operations
        mem_ssa.identify_memory_operations(ssa, ptr_size);

        // Renaming records an entry and an exit version for every location in
        // every block, so the retained size is `2 × blocks × locations`. A
        // function that is really misread data yields a distinct `Indirect`
        // location per distinct `base + offset`, so the location count scales
        // with the block count and the product is quadratic.
        //
        // Past the budget, drop the analysis rather than the process.
        let cells = mem_ssa
            .ordered_locations
            .len()
            .saturating_mul(ssa.block_count());
        if let Some(exhaustion) = context.budget().check(Budget::MemorySsaCells, cells) {
            warn!(
                "memory SSA exceeded its size budget ({} locations x {} blocks); {exhaustion}; \
                 skipping memory analysis for this function",
                mem_ssa.ordered_locations.len(),
                ssa.block_count(),
            );
            return Self::refused(exhaustion);
        }

        // No location needs versioning or a dominance frontier.
        if mem_ssa.ordered_locations.is_empty() {
            return mem_ssa;
        }

        // Which locations the image says nothing can write.
        let entry_stack_pointer = ssa.entry_stack_pointer();
        let mut immutable = BitSet::new(mem_ssa.ordered_locations.len());
        let mut regions = Vec::with_capacity(mem_ssa.ordered_locations.len());
        for (id, location) in mem_ssa.ordered_locations.iter().enumerate() {
            let region = location.region(context, entry_stack_pointer);
            if region == MemoryRegion::ImageReadOnly {
                immutable.insert_checked(id);
            }
            regions.push(region);
        }
        mem_ssa.immutable = immutable;
        mem_ssa.regions = regions;
        // What each location may touch, where a footprint can set two apart:
        // the relation is over the function's values, its objects' bytes and
        // what escapes, with the host's call models.
        if mem_ssa.footprints_may_separate() {
            let relation = points_to();
            mem_ssa.footprints = mem_ssa
                .ordered_locations
                .iter()
                .map(|location| location.footprint(relation.borrow(), context.image()))
                .collect();
        }
        // Escaped memory leaves out the frame's own cells while no address of
        // the frame leaves; worked out only where something touches it.
        let touches_escaped = mem_ssa.operations.iter().any(|operation| {
            operation
                .defined_locations()
                .chain(operation.read_locations())
                .any(|location| matches!(location, MemoryLocation::Escaped))
        });
        mem_ssa.private_frame = entry_stack_pointer
            .filter(|_| touches_escaped)
            .filter(|&pointer| !FrameEscape::of(ssa, pointer).escapes())
            .map(|pointer| FrameAnchors::of(ssa, pointer, ptr_size));

        // Both phases walk the same dominator tree, and both version what each
        // defining operation clobbers, which is asked of every location once.
        let computed_dominators;
        let dom_tree = match dominators {
            Some(tree) => tree,
            None => {
                computed_dominators = compute_dominators(eh, eh.entry());
                &computed_dominators
            }
        };
        // Admit each changed-location cell before retaining it. Both temporary
        // clobber lists and canonical definitions obey the combined allowance;
        // a rejected long block must not first allocate its complete write set.
        let mut clobbers: Vec<Rc<[usize]>> = Vec::with_capacity(mem_ssa.operations.len());
        let mut demand = cells;
        // Operations defining the same locations clobber the same ones: each
        // such set is worked out once, and held once.
        let mut shared: FxHashMap<Vec<usize>, Rc<[usize]>> = FxHashMap::default();
        let none: Rc<[usize]> = Rc::from(Vec::new());
        for operation in &mem_ssa.operations {
            if !operation.defines_memory() {
                clobbers.push(Rc::clone(&none));
                continue;
            }
            let defined: Vec<usize> = operation
                .defined_locations()
                .filter_map(|location| mem_ssa.location_ids.get(location).copied())
                .collect();
            let row = match shared.get(&defined) {
                Some(row) => Rc::clone(row),
                None => {
                    let row: Rc<[usize]> =
                        Rc::from(mem_ssa.clobbered_by(operation).collect::<Vec<usize>>());
                    shared.insert(defined, Rc::clone(&row));
                    row
                }
            };
            let Some(next) = demand.checked_add(row.len()) else {
                return Self::refused(Exhaustion {
                    budget: Budget::MemorySsaCells,
                    limit: context.budget().limit(Budget::MemorySsaCells),
                    demand: usize::MAX,
                });
            };
            if let Some(exhaustion) = context.budget().check(Budget::MemorySsaCells, next) {
                warn!(
                    "memory SSA exceeded its definition budget; {exhaustion}; skipping memory analysis for this function"
                );
                return Self::refused(exhaustion);
            }
            demand = next;
            clobbers.push(row);
        }

        // Phase 2: Place memory phi nodes
        mem_ssa.place_memory_phis(eh, dom_tree, &clobbers);

        // Phase 3: Rename memory versions
        mem_ssa.rename_memory_versions(eh, dom_tree, &clobbers);

        mem_ssa
    }

    /// Withholds partial structures when the configured cell limit or index range is exceeded.
    fn refused(exhaustion: Exhaustion) -> Self {
        let mut empty = Self::new();
        empty.exhaustion = Some(exhaustion);
        empty
    }

    /// Returns whether an access to `one` may touch `other`: whether the two
    /// may alias ([`MemoryLocation::may_alias`]), except that escaped memory
    /// is no cell of the function's own frame ([`FrameAnchors::cell`]) where
    /// no address of that frame leaves the function ([`FrameEscape`]), and
    /// that two locations whose footprints do not overlap
    /// ([`Footprint::overlaps`]) touch no byte in common: a pointer into one
    /// object against one into another, or memory the function lets go of
    /// against a frame cell below its least escaped local.
    ///
    /// A call that states it reaches escaped memory touches what it was
    /// handed and what the program let escape, and carries its stack
    /// arguments as operands: a spill kept across it in a frame whose
    /// address it never saw holds what was stored there.
    ///
    /// # Arguments
    ///
    /// * `one` - One accessed location.
    /// * `other` - The other.
    #[must_use]
    pub fn may_alias(&self, one: &MemoryLocation<T>, other: &MemoryLocation<T>) -> bool {
        self.may_alias_interned(
            one,
            self.location_ids.get(one).copied(),
            other,
            self.location_ids.get(other).copied(),
        )
    }

    /// [`Self::may_alias`] of two locations whose interned ids, where they
    /// have one, are `one_id` and `other_id`.
    fn may_alias_interned(
        &self,
        one: &MemoryLocation<T>,
        one_id: Option<usize>,
        other: &MemoryLocation<T>,
        other_id: Option<usize>,
    ) -> bool {
        let own = |location: &MemoryLocation<T>| {
            self.private_frame
                .as_ref()
                .is_some_and(|anchors| anchors.cell(location).is_some())
        };
        let placed = |id: Option<usize>| id.and_then(|id| self.footprints.get(id));
        one.may_alias(other)
            && !(matches!(one, MemoryLocation::Escaped) && own(other))
            && !(matches!(other, MemoryLocation::Escaped) && own(one))
            && placed(one_id)
                .zip(placed(other_id))
                .is_none_or(|(one, other)| one.overlaps(other))
    }

    /// Returns `true` when `location` lies in read-only image data: its
    /// contents are the image's bytes wherever the function reads them.
    #[must_use]
    pub fn is_immutable(&self, location: &MemoryLocation<T>) -> bool {
        self.location_ids
            .get(location)
            .is_some_and(|&id| self.is_immutable_id(id))
    }

    /// Tests a canonical interned cell without hashing host metadata again.
    pub fn is_immutable_id(&self, id: usize) -> bool {
        self.immutable.contains_checked(id)
    }

    /// Returns the budget the build ran out of, when it did.
    #[must_use]
    pub const fn exhaustion(&self) -> Option<Exhaustion> {
        self.exhaustion
    }

    /// Returns the memory phi nodes at a block.
    #[must_use]
    pub fn memory_phis(&self, block: usize) -> &[MemoryPhi<T>] {
        self.memory_phis.get(block).map_or(&[], Vec::as_slice)
    }

    /// Returns the memory phi that merges the location interned as `location`
    /// at `block`, if the block merges it.
    ///
    /// The phi a [`MemoryDefSite::Phi`] names, found without hashing a
    /// location: a walk back through a cell's merges asks this once per phi it
    /// passes.
    #[must_use]
    pub fn memory_phi(&self, block: usize, location: usize) -> Option<&MemoryPhi<T>> {
        let at = self
            .phi_locations
            .get(block)?
            .binary_search(&location)
            .ok()?;
        self.memory_phis.get(block)?.get(at)
    }

    /// Returns all memory operations.
    #[must_use]
    pub fn operations(&self) -> &[MemoryOp<T>] {
        &self.operations
    }

    /// Returns all unique memory locations.
    #[must_use]
    pub fn locations(&self) -> &FxHashSet<MemoryLocation<T>> {
        &self.locations
    }

    /// Returns the memory version at block entry for a location.
    ///
    /// The version every instruction in the block before the first write to the
    /// location reads. Where the block merges the location, that is its memory
    /// phi's version: the phi defines it at the block's entry.
    #[must_use]
    pub fn version_at_entry(&self, location: &MemoryLocation<T>, block: usize) -> Option<u32> {
        let id = *self.location_ids.get(location)?;
        self.entry_versions.get(block, id)
    }

    /// Returns the memory version at block exit for a location.
    #[must_use]
    pub fn version_at_exit(&self, location: &MemoryLocation<T>, block: usize) -> Option<u32> {
        let id = *self.location_ids.get(location)?;
        self.exit_versions.get(block, id)
    }

    /// Returns the version reaching a position immediately before an instruction.
    ///
    /// The instruction at `instruction` has not executed. A position equal to
    /// the block's instruction count denotes its normal exit. Earlier
    /// may-alias stores, calls and barriers contribute their versions, and
    /// block-entry phis and exception-entry unknowns retain their identities.
    /// The first interior mutable-state query derives a bounded index from
    /// existing definitions in one linear pass. Later queries allocate nothing
    /// and binary-search a location/position row. Entry and exit use the
    /// canonical boundary tables directly. A cell unchanged within its block
    /// uses the entry version, including read-only image cells. Block-boundary
    /// consumers build no such index.
    ///
    /// # Arguments
    ///
    /// * `location` - The memory cell whose state is requested.
    /// * `block` - The block containing the observation.
    /// * `instruction` - The position before which state is observed.
    ///
    /// # Returns
    ///
    /// The reaching version, or `None` for an unknown cell, unreachable block,
    /// invalid position, unrepresentable version or exhausted analysis. A
    /// version identifies state; it does not prove that the state's bytes are known.
    #[must_use]
    pub fn version_before(
        &self,
        location: &MemoryLocation<T>,
        block: usize,
        instruction: usize,
    ) -> Option<u32> {
        let id = self.location_ids.get(location).copied()?;
        self.version_before_id(id, block, instruction)
    }

    /// Queries one canonical cell by its analysis-local identity.
    ///
    /// Entry and exit boundaries use the canonical tables. Interior queries
    /// initialize a shared instruction-point index on first use.
    pub fn version_before_id(&self, id: usize, block: usize, instruction: usize) -> Option<u32> {
        let row = self.block_operations.get(block)?;
        if instruction > row.instructions {
            return None;
        }
        let entry = self.entry_versions.get(block, id)?;
        let exit = self.exit_versions.get(block, id);
        if instruction == 0 || exit == Some(entry) {
            return Some(entry);
        }
        if instruction == row.instructions {
            return exit;
        }
        let writes = self.instruction_versions().get(block)?;
        let mut low = 0usize;
        let mut high = writes.len();
        while low < high {
            let middle = low.saturating_add(high.saturating_sub(low) / 2);
            let write = writes.get(middle)?;
            if (write.location, write.instruction) < (id, instruction) {
                low = middle.saturating_add(1);
            } else {
                high = middle;
            }
        }
        if let Some(write) = low.checked_sub(1).and_then(|last| writes.get(last))
            && write.location == id
        {
            return u32::try_from(write.version).ok();
        }
        Some(entry)
    }

    /// Indexes writes by location and instruction position on first use.
    ///
    /// Existing definition order already sorts each block's writes. The index
    /// borrows the same immutable body and needs no second memory representation.
    fn instruction_versions(&self) -> &[Vec<VersionWrite>] {
        if let Some(index) = self.write_versions.get() {
            return index;
        }
        let mut blocks: Vec<Vec<VersionWrite>> = (0..self.block_operations.len())
            .map(|_| Vec::new())
            .collect();
        for (location, definitions) in self.definitions.iter().enumerate() {
            for (version, definition) in definitions.iter().enumerate() {
                if let MemoryDefSite::Store { block, instr } = definition
                    && let Some(writes) = blocks.get_mut(*block)
                {
                    writes.push(VersionWrite {
                        location,
                        instruction: *instr,
                        version,
                    });
                }
            }
        }

        self.write_versions.get_or_init(|| blocks)
    }

    /// Returns every location with the definition site of the version that
    /// enters `block`, in the order the locations first appear in the function.
    ///
    /// The batch form of [`Self::version_at_entry`] followed by
    /// [`Self::definition`] for each location: it never hashes a location. A
    /// location with no version at the block's entry, as in a block the
    /// function's entry does not reach, is left out.
    pub fn entry_definitions(
        &self,
        block: usize,
    ) -> impl Iterator<Item = (&MemoryLocation<T>, MemoryDefSite)> {
        self.ordered_locations
            .iter()
            .enumerate()
            .filter_map(move |(id, location)| {
                let version = self.entry_versions.get(block, id)?;
                Some((location, self.defined_at(id, version)?))
            })
    }

    /// Returns the definition site for a memory version.
    #[must_use]
    pub fn definition(&self, version: &MemoryVersion<T>) -> Option<MemoryDefSite> {
        self.definition_of(&version.location, version.version)
    }

    /// Returns the definition of a borrowed cell's analysis-local version.
    ///
    /// Avoids cloning target field metadata when walking memory dependencies.
    #[must_use]
    pub fn definition_of(
        &self,
        location: &MemoryLocation<T>,
        version: u32,
    ) -> Option<MemoryDefSite> {
        self.defined_at(*self.location_ids.get(location)?, version)
    }

    /// Returns the canonical classified memory operation at an instruction.
    ///
    /// Uses block-local binary search. Unclassified instructions and invalid
    /// positions return `None`.
    #[must_use]
    pub fn operation_at(&self, block: usize, instruction: usize) -> Option<&MemoryOp<T>> {
        let row = self
            .operations
            .get(self.block_operations.get(block)?.operations.clone())?;
        let index = row
            .binary_search_by_key(&instruction, MemoryOp::instr)
            .ok()?;
        row.get(index)
    }

    /// Resolves a borrowed cell to this analysis's interned identity.
    pub fn location_id(&self, location: &MemoryLocation<T>) -> Option<usize> {
        self.location_ids.get(location).copied()
    }

    /// Borrows a canonical cell without copying host metadata.
    pub fn location_by_id(&self, id: usize) -> Option<&MemoryLocation<T>> {
        self.ordered_locations.get(id)
    }

    /// The values the load at `instruction` of `block` of `ssa` may read:
    /// what each store to its location that may reach it wrote, through the
    /// memory phis that merge them, each once, in value order.
    ///
    /// `None` where anything else may define what it reads: the function's
    /// entry or the image, a write that only may be to the location (a call,
    /// a store through an address the alias relation cannot tell from the
    /// location's), an exception handler's entry, which the protected region
    /// may have left at any point, or a store of more than one value; and
    /// where the instruction is no load of one location.
    #[must_use]
    pub fn stored_values(
        &self,
        ssa: &SsaFunction<T>,
        block: usize,
        instruction: usize,
    ) -> Option<Vec<SsaVarId>> {
        let load = self
            .operation_at(block, instruction)
            .filter(|operation| operation.is_load())?;
        let location = self.location_id(load.location()?)?;
        let mut pending = vec![self.version_before_id(location, block, instruction)?];
        let mut seen = FxHashSet::default();
        let mut stored = Vec::new();
        while let Some(version) = pending.pop() {
            if !seen.insert(version) {
                continue;
            }
            match self.defined_at(location, version)? {
                MemoryDefSite::Store { block, instr } => {
                    let store = self
                        .operation_at(block, instr)
                        .filter(|operation| operation.is_store())?;
                    if self.location_id(store.location()?)? != location {
                        return None;
                    }
                    let mut written = Vec::new();
                    ssa.block(block)?
                        .instructions()
                        .get(instr)?
                        .op()
                        .for_each_addressed_access(|access| {
                            if access.writes {
                                written.push(access.value);
                            }
                        });
                    let [Some(value)] = written[..] else {
                        return None;
                    };
                    stored.push(value);
                }
                MemoryDefSite::Phi { block } => pending.extend(
                    self.memory_phi(block, location)?
                        .operands
                        .iter()
                        .map(|operand| operand.version),
                ),
                MemoryDefSite::Entry
                | MemoryDefSite::Image
                | MemoryDefSite::ExceptionEntry { .. } => return None,
            }
        }
        stored.sort_unstable();
        stored.dedup();
        Some(stored)
    }

    /// Returns the definition site of `version` of the location interned as
    /// `location`.
    pub fn defined_at(&self, location: usize, version: u32) -> Option<MemoryDefSite> {
        let version = usize::try_from(version).ok()?;
        self.definitions.get(location)?.get(version).copied()
    }

    /// Defines the next version of the location interned as `location` at
    /// `site`, and returns its number.
    ///
    /// A location's versions are numbered from zero in the order they are
    /// defined, which is the position of their site in its row of
    /// [`Self::definitions`].
    fn define(&mut self, location: usize, site: MemoryDefSite) -> u32 {
        let Some(versions) = self.definitions.get_mut(location) else {
            return 0;
        };
        let version = u32::try_from(versions.len()).unwrap_or(u32::MAX);
        versions.push(site);
        version
    }

    /// The interned ids of the locations `op` invalidates, in location order:
    /// each location it defines and every location those may alias, except the
    /// read-only image cells, which nothing can write.
    fn clobbered_by<'a>(&'a self, op: &'a MemoryOp<T>) -> impl Iterator<Item = usize> + 'a {
        // Each defined location's region, found once: a hash per candidate is
        // what this loop would otherwise spend most of its time on.
        let defined: Vec<(&MemoryLocation<T>, Option<usize>, Option<MemoryRegion>)> = op
            .defined_locations()
            .map(|location| {
                let id = self.location_ids.get(location).copied();
                let region = id.and_then(|id| self.regions.get(id).copied());
                (location, id, region)
            })
            .collect();
        self.ordered_locations
            .iter()
            .enumerate()
            .filter(move |&(id, candidate)| {
                !self.immutable.contains_checked(id)
                    && defined.iter().any(|&(location, defined_id, region)| {
                        !self.regions_are_disjoint(id, region)
                            && self.may_alias_interned(candidate, Some(id), location, defined_id)
                    })
            })
            .map(|(id, _)| id)
    }

    /// Whether footprints ([`MemoryLocation::footprint`]) can set two of the
    /// locations apart that their addresses leave may-aliasing: only two of
    /// different bases do, escaped memory counting as one, where one of them
    /// is reached through a value, which only the points-to relation places.
    /// Every other location's footprint is any memory.
    fn footprints_may_separate(&self) -> bool {
        let mut through_value = false;
        let mut bases: Vec<Option<AddressBase>> = Vec::new();
        for location in &self.ordered_locations {
            let base = match location {
                MemoryLocation::Indirect(cell) => Some(cell.base),
                MemoryLocation::Escaped => None,
                MemoryLocation::InstanceField(..)
                | MemoryLocation::StaticField(_)
                | MemoryLocation::ArrayElement(..)
                | MemoryLocation::Unknown => continue,
            };
            through_value |= matches!(base, Some(AddressBase::Value(_)));
            if !bases.contains(&base) {
                bases.push(base);
            }
            if through_value && bases.len() > 1 {
                return true;
            }
        }
        false
    }

    /// Whether the location interned as `id` and a location in the region
    /// `defined` (`None` for one the analysis has no region for) lie in regions
    /// that cannot be the same memory.
    ///
    /// The address expressions alone cannot say that a stack cell and an image
    /// cell are apart: the frame is addressed from a value and the image cell
    /// from a stated address, which [`IndirectLocation::may_alias`] must call
    /// a may-alias because a value can hold any address. Their *regions* say
    /// it - the frame is the stack the function was entered on and the image's
    /// cells are at addresses the host serves, and no frame of a running
    /// function lies in the image it was loaded from. This is where that is
    /// worth saying: a store to one otherwise defines a version of every
    /// location in the other, which is most of them.
    ///
    /// [`MemoryRegion::Unknown`] is disjoint from nothing - it is the answer
    /// for memory reached through a pointer, which may reach anywhere - so a
    /// location the analysis cannot place keeps clobbering everything.
    fn regions_are_disjoint(&self, id: usize, defined: Option<MemoryRegion>) -> bool {
        match (self.regions.get(id), defined) {
            (Some(&candidate), Some(other)) => candidate.is_apart_from(other),
            _ => false,
        }
    }

    /// Phase 1: Identify all memory operations in the SSA function.
    fn identify_memory_operations(&mut self, ssa: &SsaFunction<T>, ptr_size: PointerSize) {
        self.block_operations = ssa
            .blocks()
            .iter()
            .map(|block| MemoryBlock {
                instructions: block.instructions().len(),
                operations: 0..0,
            })
            .collect();
        for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
            if let Some(mem_op) =
                MemoryLocator::new(ssa, ptr_size).classify(instr.op(), block_idx, instr_idx)
            {
                for location in mem_op.accessed_locations() {
                    if self.locations.insert(location.clone()) {
                        self.location_ids
                            .insert(location.clone(), self.ordered_locations.len());
                        self.ordered_locations.push(location.clone());
                        self.definitions.push(Vec::new());
                    }
                }
                // Instructions are visited block by block, so a block's
                // operations are adjacent.
                let index = self.operations.len();
                if let Some(block) = self.block_operations.get_mut(block_idx) {
                    let range = &mut block.operations;
                    if range.start == range.end {
                        range.start = index;
                    }
                    range.end = index.saturating_add(1);
                }
                self.operations.push(mem_op);
            }
        }
    }

    /// Phase 2: Place memory phi nodes at dominance frontiers.
    ///
    /// Standard iterated-dominance-frontier placement, with the twist specific to
    /// memory: a store to `L` is also a definition of every location that
    /// may-alias `L`, so each defining op seeds the definition-block set of all
    /// its may-aliases, not just its own location. Without that, an overlapping
    /// cell or a barrier's `Unknown` would receive no merge point and its version
    /// could survive a clobber. Locations are processed in `ordered_locations`
    /// order so version numbering does not depend on `HashMap` iteration order.
    ///
    /// # Arguments
    ///
    /// * `eh` - Exception-aware CFG, source of the dominance frontiers.
    /// * `dom_tree` - The dominator tree of `eh`.
    fn place_memory_phis(
        &mut self,
        eh: &SsaCfg<'_, T>,
        dom_tree: &DominatorTree,
        clobbers: &[Rc<[usize]>],
    ) {
        let block_count = eh.node_count();
        if block_count == 0 {
            return;
        }

        let frontiers = compute_dominance_frontiers(eh, dom_tree);
        self.memory_phis = vec![Vec::new(); block_count];
        self.phi_locations = vec![Vec::new(); block_count];

        // A definition of `L` is also a definition of every location it may
        // alias. Memory versions are per-location, so without this a store to
        // `*(p+16)` bumps only its own version and a barrier bumps only
        // `Unknown` - leaving the version of a may-aliasing cell unchanged and
        // therefore still looking forwardable across a block boundary.
        let mut def_blocks: Vec<Vec<usize>> = vec![Vec::new(); self.ordered_locations.len()];
        for (op, clobbered) in self.operations.iter().zip(clobbers) {
            for &id in clobbered.iter() {
                if let Some(blocks) = def_blocks.get_mut(id) {
                    blocks.push(op.block());
                }
            }
        }

        // Standard phi placement algorithm (iterated dominance frontier).
        // Iterated in `ordered_locations` order so version numbering does not
        // depend on `HashMap` iteration order, each location's worklist from
        // its defining blocks in ascending order. Which blocks a location has
        // a phi at, and which it has processed, are stamped with the location:
        // one array of each serves every location, with no set to allocate
        // per location.
        let mut phi_at = vec![usize::MAX; block_count];
        let mut processed_at = vec![usize::MAX; block_count];
        for (id, defs) in def_blocks.iter_mut().enumerate() {
            if defs.is_empty() {
                continue;
            }
            defs.sort_unstable();
            defs.dedup();
            let Some(location) = self.ordered_locations.get(id).cloned() else {
                continue;
            };
            let mut worklist: VecDeque<usize> = defs.iter().copied().collect();

            while let Some(block) = worklist.pop_front() {
                match processed_at.get_mut(block) {
                    Some(stamp) if *stamp != id => *stamp = id,
                    _ => continue,
                }

                let Some(frontier_set) = frontiers.get(block) else {
                    continue;
                };
                for frontier_block in frontier_set.iter() {
                    let fresh = match phi_at.get_mut(frontier_block) {
                        Some(stamp) if *stamp != id => {
                            *stamp = id;
                            true
                        }
                        _ => false,
                    };
                    if fresh {
                        // Add phi at frontier
                        let version = self.define(
                            id,
                            MemoryDefSite::Phi {
                                block: frontier_block,
                            },
                        );
                        if let (Some(phis), Some(locations)) = (
                            self.memory_phis.get_mut(frontier_block),
                            self.phi_locations.get_mut(frontier_block),
                        ) {
                            phis.push(MemoryPhi::new(location.clone(), version));
                            locations.push(id);
                        }
                        worklist.push_back(frontier_block);
                    }
                }
            }
        }
    }

    /// Phase 3: Rename memory versions using dominator tree traversal.
    ///
    /// Implements the renaming half of Cytron et al.: each block pushes the
    /// versions its phis and stores define, recurses into its dominator-tree
    /// children, then **pops exactly what it pushed** so that a sibling subtree
    /// sees the state its own dominator left, not its sibling's.
    ///
    /// The walk is an explicit-stack DFS rather than recursion: dominator trees
    /// on real functions reach thousands of blocks deep, and this crate denies
    /// panics - a blown call stack is not a recoverable error.
    ///
    /// Version *numbering* is an opaque identity, not an ordering: phi versions
    /// are allocated during phi placement (phase 2), so a location carrying phis
    /// receives its entry version here with a number above them. Consumers that
    /// need to recognise the entry version match [`MemoryDefSite::Entry`] via
    /// [`MemorySsa::definition`] rather than comparing against zero.
    ///
    /// # Arguments
    ///
    /// * `eh` - Exception-aware CFG the walk follows.
    /// * `dom_tree` - The dominator tree of `eh`, in whose preorder blocks are
    ///   renamed.
    fn rename_memory_versions(
        &mut self,
        eh: &SsaCfg<'_, T>,
        dom_tree: &DominatorTree,
        clobbers: &[Rc<[usize]>],
    ) {
        let block_count = eh.node_count();
        if block_count == 0 {
            return;
        }

        let locations = self.ordered_locations.len();
        self.entry_versions = VersionTable::new(block_count, locations);
        self.exit_versions = VersionTable::new(block_count, locations);

        // Live version stack per location; the top is the version reaching the
        // block currently being renamed. Every location is seeded with its
        // function-entry version - the image's bytes for a location nothing can
        // write.
        let mut live = VersionStacks::new(locations);
        for id in 0..locations {
            let site = if self.immutable.contains_checked(id) {
                MemoryDefSite::Image
            } else {
                MemoryDefSite::Entry
            };
            let entry_version = self.define(id, site);
            live.push(id, entry_version);
        }

        // Rename in dominator-tree preorder, restoring scope on the way out.
        let mut visited = vec![false; block_count];
        let mut worklist = vec![RenameStep::Enter(eh.entry().index())];

        while let Some(step) = worklist.pop() {
            match step {
                RenameStep::Enter(block_idx) => {
                    match visited.get(block_idx) {
                        Some(true) | None => continue,
                        Some(false) => {}
                    }
                    if let Some(slot) = visited.get_mut(block_idx) {
                        *slot = true;
                    }

                    let pushed = self.rename_block(block_idx, eh, &mut live, clobbers);

                    // Pushed before the children so it pops after all of them:
                    // the subtree runs with this block's definitions live.
                    worklist.push(RenameStep::Exit(pushed));
                    for child in dom_tree.children(NodeId::new(block_idx)) {
                        if visited.get(child.index()).copied() == Some(false) {
                            worklist.push(RenameStep::Enter(child.index()));
                        }
                    }
                }
                RenameStep::Exit(pushed) => {
                    for location in pushed {
                        live.pop(location);
                    }
                }
            }
        }
    }

    /// Gives every location the protected region may have written a fresh
    /// version at an exception root, standing for "unknown".
    ///
    /// Block dominance says the guard executed wholly before the handler; the
    /// runtime leaves it part-way. Without this, renaming pushes the guard's
    /// version into the handler and a later pass reads a store that the throw
    /// may have pre-empted.
    ///
    /// A location whose live version was **not** defined inside the region is
    /// left alone: a store completed before the region started is as visible in
    /// the handler as anywhere else. When the region's extent is unknown, every
    /// location is killed, which is the conservative direction.
    ///
    /// Returns the locations pushed, for the caller to pop on the way out.
    fn kill_at_exception_root(
        &mut self,
        block_idx: usize,
        eh: &SsaCfg<'_, T>,
        live: &mut VersionStacks,
    ) -> Vec<usize> {
        let region = eh.roots().region_of(block_idx);
        let mut killed: Vec<usize> = Vec::new();

        for id in 0..self.ordered_locations.len() {
            // Nothing the region did can have written the image's read-only
            // bytes, interrupted or not.
            if self.immutable.contains_checked(id) {
                continue;
            }
            let Some(current) = live.live(id) else {
                continue;
            };
            let defined_in_region = match self.defined_at(id, current) {
                // An unknown extent could have written anything.
                None => true,
                Some(MemoryDefSite::Store { block, .. } | MemoryDefSite::Phi { block }) => {
                    region.as_ref().is_none_or(|range| range.contains(&block))
                }
                // Already unknown, or the function-entry state, which the
                // region cannot have disturbed without a store.
                Some(
                    MemoryDefSite::ExceptionEntry { .. }
                    | MemoryDefSite::Entry
                    | MemoryDefSite::Image,
                ) => false,
            };
            if !defined_in_region {
                continue;
            }
            killed.push(id);
        }

        for &id in &killed {
            let version = self.define(id, MemoryDefSite::ExceptionEntry { block: block_idx });
            live.push(id, version);
        }

        killed
    }

    /// Renames memory versions within a single block.
    ///
    /// Returns one entry per version pushed onto `live`, which the caller pops
    /// when leaving this block's dominator subtree. A block that stores twice to
    /// the same location pushes twice and so appears twice.
    fn rename_block(
        &mut self,
        block_idx: usize,
        eh: &SsaCfg<'_, T>,
        live: &mut VersionStacks,
        clobbers: &[Rc<[usize]>],
    ) -> Vec<usize> {
        let mut pushed: Vec<usize> = Vec::new();

        // Entering a handler or filter: the region it protects was interrupted
        // part-way, so any store the region made is not known to have run.
        // Those locations get a fresh version standing for "unknown".
        if eh.roots().is_flow_root(block_idx) {
            pushed.extend(self.kill_at_exception_root(block_idx, eh, live));
        }

        // Memory phis define their versions at the block's entry, before
        // anything in it runs.
        let phis = self
            .memory_phis
            .get(block_idx)
            .map_or(&[][..], Vec::as_slice);
        let phi_locations = self
            .phi_locations
            .get(block_idx)
            .map_or(&[][..], Vec::as_slice);
        for (phi, &location) in phis.iter().zip(phi_locations) {
            live.push(location, phi.result_version);
            pushed.push(location);
        }

        // Record entry versions, the phis' included: a location merged here
        // enters on the merge.
        self.entry_versions.record(block_idx, live);

        // The block's operations were classified once, when they were
        // identified, and the function has not changed since.
        for index in self
            .block_operations
            .get(block_idx)
            .map(|block| block.operations.clone())
            .unwrap_or_default()
        {
            let Some(mem_op) = self.operations.get(index) else {
                continue;
            };
            if !mem_op.defines_memory() {
                continue;
            }
            // The op defines its own location *and clobbers every location it
            // may alias*. A store to `*(p+16)` invalidates the overlapping
            // `*(p+0)`; a call, fence, atomic, volatile or opaque op classifies
            // to `Unknown`, which may-aliases everything and so invalidates all
            // of memory. Versioning only the named location is what let a value
            // survive across a block boundary that a barrier or an overlapping
            // store had already destroyed.
            //
            // Matches `place_memory_phis`, which places phis for the same set,
            // so every clobbered location has a merge point.
            //
            // A call its host described defines only what it may write.
            let instr = mem_op.instr();
            for &location in clobbers.get(index).map_or(&[][..], |row| &**row) {
                let new_version = self.define(
                    location,
                    MemoryDefSite::Store {
                        block: block_idx,
                        instr,
                    },
                );
                live.push(location, new_version);
                pushed.push(location);
            }
        }

        // Record exit versions
        self.exit_versions.record(block_idx, live);

        // Fill in phi operands for successors
        for succ_id in eh.successors(NodeId::new(block_idx)) {
            let succ_idx = succ_id.index();
            if let (Some(phis), Some(locations)) = (
                self.memory_phis.get_mut(succ_idx),
                self.phi_locations.get(succ_idx),
            ) {
                for (phi, &location) in phis.iter_mut().zip(locations) {
                    if let Some(version) = live.live(location) {
                        phi.add_operand(block_idx, version);
                    }
                }
            }
        }

        pushed
    }

    /// Returns statistics about the Memory SSA.
    #[must_use]
    pub fn stats(&self) -> MemorySsaStats {
        let total_phis = self.memory_phis.iter().map(Vec::len).sum();
        let store_count = self
            .operations
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    MemoryOp::Store { .. } | MemoryOp::ReadWrite { .. } | MemoryOp::Write { .. }
                )
            })
            .count();
        let load_count = self.operations.iter().filter(|op| op.is_load()).count();
        let barrier_count = self
            .operations
            .iter()
            .filter(|op| matches!(op, MemoryOp::Barrier { .. }))
            .count();
        let call_count = self
            .operations
            .iter()
            .filter(|op| matches!(op, MemoryOp::Call { .. }))
            .count();

        MemorySsaStats {
            location_count: self.locations.len(),
            memory_phi_count: total_phis,
            store_count,
            load_count,
            barrier_count,
            call_count,
            version_count: self.definitions.iter().map(Vec::len).sum(),
        }
    }
}

impl<T: Target> Default for MemorySsa<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about Memory SSA.
#[derive(Debug, Clone, Copy)]
pub struct MemorySsaStats {
    /// Number of unique memory locations tracked.
    pub location_count: usize,
    /// Number of memory phi nodes placed.
    pub memory_phi_count: usize,
    /// Number of store operations.
    pub store_count: usize,
    /// Number of load operations.
    pub load_count: usize,
    /// Number of barriers: calls nobody described and opaque memory
    /// operations ([`MemoryOp::Barrier`]).
    pub barrier_count: usize,
    /// Number of calls whose effects their host stated ([`MemoryOp::Call`]),
    /// each defining the locations it may write.
    pub call_count: usize,
    /// Total number of memory versions.
    pub version_count: usize,
}

/// SSA identities and memory versions retained along one analysis path.
///
/// Stores invalidate every location they may overwrite; loads resolve exact or
/// must-aliasing cells. The payload names an SSA definition rather than copying
/// its current evaluated value. An executor that revisits definitions uses
/// store-time value snapshots, as [`crate::analysis::evaluator::SsaEvaluator`]
/// does, rather than reading this identity's later value.
///
/// # Examples
///
/// ```
/// use analyssa::{
///     analysis::memory::{MemoryLocation, MemoryState},
///     ir::variable::SsaVarId,
///     testing::MockTarget,
/// };
/// let mut state = MemoryState::<MockTarget>::new();
/// let value = SsaVarId::from_index(0);
/// let field = MemoryLocation::StaticField(1);
/// state.store(field.clone(), value, 1);
/// assert_eq!(state.load(&field), Some(value));
/// state.forget(&MemoryLocation::Unknown);
/// assert!(state.load(&field).is_none());
/// ```
#[derive(Debug, Clone)]
pub struct MemoryState<T: Target> {
    /// Current memory values: location -> (version, value as SSA variable).
    values: MemoryCells<T, (u32, SsaVarId)>,
}

impl<T: Target> MemoryState<T> {
    /// Creates a new empty memory state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: MemoryCells::new(),
        }
    }

    /// Records a memory store, overwriting any previous value for `location`.
    ///
    /// # Arguments
    ///
    /// * `location` - The cell being written.
    /// * `value` - SSA variable holding the stored value.
    /// * `version` - Memory version this store produces for the location.
    ///
    /// Whatever the store may overwrite is forgotten first: a cell that only
    /// may alias `location` no longer holds what was stored to it, and keeping
    /// it would answer a later load with a value this store may have replaced.
    pub fn store(&mut self, location: MemoryLocation<T>, value: SsaVarId, version: u32) {
        self.values.store(location, (version, value));
    }

    /// Forgets every value a write to `location` may change: each tracked
    /// cell that may alias it.
    ///
    /// # Arguments
    ///
    /// * `location` - The memory written, or [`MemoryLocation::Unknown`] for
    ///   a write that may reach anything.
    pub fn forget(&mut self, location: &MemoryLocation<T>) {
        self.values.forget(location);
    }

    /// Loads from a memory location, resolving through must-aliasing cells.
    ///
    /// A direct hit on `location` wins immediately. Otherwise the stored cells
    /// that must-alias the query are considered, and because several can hold
    /// different values, the most recently stored version wins, with the lower
    /// SSA id breaking a tie. That rule is deterministic, which matters because
    /// the downstream pipeline is content-addressed and must not depend on the
    /// hasher's per-instance seed.
    ///
    /// # Arguments
    ///
    /// * `location` - The cell to read.
    ///
    /// # Returns
    ///
    /// The SSA variable holding the value, or `None` if nothing is known.
    #[must_use]
    pub fn load(&self, location: &MemoryLocation<T>) -> Option<SsaVarId> {
        self.values
            .load(location, |(version, value)| (*version, Reverse(*value)))
            .map(|(_, value)| *value)
    }

    /// Returns the current version for a location, if known.
    #[must_use]
    pub fn version(&self, location: &MemoryLocation<T>) -> Option<u32> {
        self.values.direct(location).map(|(v, _)| *v)
    }

    /// Checks if any stored location may alias the given location.
    #[must_use]
    pub fn has_may_alias(&self, location: &MemoryLocation<T>) -> bool {
        self.values.has_may_alias(location)
    }

    /// Clears all memory state.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Returns the number of tracked locations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if no memory is being tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl<T: Target> Default for MemoryState<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of alias analysis between two memory locations.
///
/// Three outcomes are possible based on the precision of the location descriptions:
/// - `MustAlias`: Proven to refer to the same memory cell
/// - `MayAlias`: Cannot prove they are different (conservative approximation)
/// - `NoAlias`: Proven to refer to different memory cells
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub enum AliasResult {
    /// The two memory locations definitely refer to different memory cells.
    /// This is the strongest result and enables optimization.
    NoAlias,
    /// The two locations MAY refer to the same memory cell (conservative).
    /// Analysis must assume they could alias.
    MayAlias,
    /// The two locations definitely refer to the same memory cell.
    /// This requires proof (e.g., same static field, same object + same field).
    MustAlias,
}

/// Performs alias analysis between two memory locations.
///
/// Collapses the two boolean predicates into the three-valued [`AliasResult`]:
/// [`must_alias`](MemoryLocation::must_alias) proves `MustAlias`, else
/// [`may_alias`](MemoryLocation::may_alias) gives `MayAlias`, else the locations
/// are proven disjoint and the result is `NoAlias`.
///
/// # Arguments
///
/// * `loc1` - First location.
/// * `loc2` - Second location.
///
/// # Returns
///
/// The three-valued alias relationship between the two locations.
#[must_use]
pub fn analyze_alias<T: Target>(loc1: &MemoryLocation<T>, loc2: &MemoryLocation<T>) -> AliasResult {
    if loc1.must_alias(loc2) {
        AliasResult::MustAlias
    } else if loc1.may_alias(loc2) {
        AliasResult::MayAlias
    } else {
        AliasResult::NoAlias
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use crate::{
        analysis::{
            context::AnalysisContext,
            memory::{MemoryDefSite, MemorySsa},
        },
        ir::{
            function::{SsaDefSpec, SsaFunctionBuilder},
            variable::{DefSite, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    /// Builds actual ordered writes to one mutable cell, followed by return.
    fn writes(count: usize) -> MemorySsa<MockTarget> {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let pointer = builder.function_mut().create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        builder
            .in_block(0, |block| {
                let value = block.const_i32(SsaDefSpec::tmp(MockType::I32), 42)?;
                for _ in 0..count {
                    block.store_indirect(pointer, value, MockType::I32)?;
                }
                block.ret(None)
            })
            .unwrap();
        let ssa = builder.finish().unwrap();
        MemorySsa::build(&ssa, &AnalysisContext::new(MACHINE))
    }

    /// Interior queries initialize the write index once and reuse the exact versions.
    #[test]
    fn canonical_point_index_is_lazy_and_reusable() {
        let memory = writes(4096);
        let location = memory.location_by_id(0).unwrap();
        assert!(memory.write_versions.get().is_none());
        let version = memory.version_before(location, 0, 4096).unwrap();
        assert_eq!(
            memory.definition_of(location, version),
            Some(MemoryDefSite::Store {
                block: 0,
                instr: 4095
            })
        );
        let index = memory.write_versions.get().unwrap();
        assert_eq!(memory.version_before(location, 0, 4096), Some(version));
        assert!(ptr::eq(index, memory.write_versions.get().unwrap()));
    }

    /// Entry and exit queries use canonical tables without constructing a write index.
    #[test]
    fn canonical_point_boundaries_need_no_index_allocation() {
        let memory = writes(4);
        let location = memory.location_by_id(0).unwrap();
        for (instruction, expected) in [
            (0, memory.version_at_entry(location, 0)),
            (6, memory.version_at_exit(location, 0)),
        ] {
            assert_eq!(memory.version_before(location, 0, instruction), expected);
            assert!(memory.write_versions.get().is_none());
        }
    }
}
