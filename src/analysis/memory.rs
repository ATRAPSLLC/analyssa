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
//! **Phase 3: Renaming** (O(B * L)):
//! Traverse the dominator tree in preorder, maintaining a version stack per location:
//! 1. Record entry version for each location at block entry
//! 2. Push new versions for phi nodes and store definitions
//! 3. Process all instructions (loads read top of stack, stores create new versions)
//! 4. Record exit versions for each location at block exit
//! 5. Fill in phi operands for successor blocks with current version stacks
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
//! An `Indirect` location is a *decoded* address — `base + index*stride +
//! offset` plus the access width — not the SSA id of the address operand. Two
//! of them off a common base compare by overlapping bit extent, so `[rbp-8]`
//! and `[rbp-16]` are provably disjoint; different bases conservatively
//! may-alias, since two SSA pointers can hold one address. See
//! [`IndirectLocation`] for the full rationale, including why keying on the
//! address value id is both imprecise and unstable under GVN.
//!
//! # Usage
//!
//! ```rust
//! use analyssa::{
//!     analysis::{
//!         memory::{MemoryLocation, MemorySsa},
//!         SsaCfg,
//!     },
//!     ir::SsaVarId,
//!     testing, PointerSize,
//! };
//!
//! // Fixture that stores to `object.field1`, loads it back, then performs
//! // an indirect store and an atomic exchange.
//! let ssa = testing::memory_effect_fixture();
//! let cfg = SsaCfg::from_ssa(&ssa);
//! let mem_ssa = MemorySsa::build(&ssa, &cfg, PointerSize::Bit64);
//!
//! // The store and the load are attributed to the same memory location.
//! let object = SsaVarId::from_index(0);
//! let loc = MemoryLocation::InstanceField(object, 1);
//! assert!(mem_ssa.locations().contains(&loc));
//!
//! // Query the memory version entering and leaving a block. Every operation
//! // that may alias this location bumps its version, not just the ones naming
//! // it: the fixture's indirect store and atomic exchange both classify to
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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

/// Maximum `locations × blocks` product [`MemorySsa::build`] will construct.
///
/// Renaming records an entry and an exit version per location per block, so the
/// retained allocation is quadratic when the location count scales with the
/// block count — which it does for a "function" that is really data misread as
/// code, where each distinct `base + offset` mints its own `Indirect` location.
/// At 4000 blocks × 4000 locations that is ~16M map entries, several GB.
///
/// 4M cells is far above any real function (a 2000-block function touching 200
/// distinct cells is 400k) and bounds the structure at tens of MB.
const MAX_MEMORY_SSA_CELLS: usize = 4_000_000;

use crate::{
    analysis::{
        address::{const_i64, normalize_address},
        cfg::SsaCfg,
    },
    graph::{
        GraphBase, NodeId, RootedGraph, Successors,
        algorithms::{compute_dominance_frontiers, compute_dominators},
    },
    ir::{
        function::SsaFunction,
        ops::{MemoryEffectLocation, SsaEffectKind, SsaEffects, SsaOp},
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
    /// value id — see [`IndirectLocation`] for why that distinction is
    /// load-bearing.
    Indirect(IndirectLocation),

    /// Unknown/escaped memory.
    ///
    /// Used when we can't determine the exact location (e.g., after a call
    /// that may modify memory, or for volatile accesses).
    Unknown,
}

impl<T: Target> MemoryLocation<T> {
    /// Returns the base object variable, if any.
    ///
    /// For instance fields and arrays, this is the object/array variable.
    /// For static fields and unknown locations, returns `None`.
    #[must_use]
    pub fn base_object(&self) -> Option<SsaVarId> {
        match self {
            Self::InstanceField(obj, _) => Some(*obj),
            Self::ArrayElement(arr, _) => Some(*arr),
            Self::Indirect(indirect) => Some(indirect.base),
            Self::StaticField(_) | Self::Unknown => None,
        }
    }

    /// Returns `true` if this location may alias the other location.
    ///
    /// This is a conservative analysis - if we can't prove non-aliasing,
    /// we assume aliasing is possible.
    #[must_use]
    pub fn may_alias(&self, other: &Self) -> bool {
        match (self, other) {
            // Unknown aliases everything; Indirect may alias any concrete location
            (Self::Unknown, _)
            | (_, Self::Unknown)
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
            // direction that matters — `may_alias` drives the invalidation half
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
    /// This is a more precise analysis - returns `true` only if we can
    /// prove the locations definitely refer to the same memory.
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

/// A pointer dereference decoded into `base + index*stride + offset`, together
/// with the width of the access.
///
/// # Why not the address value id
///
/// The obvious encoding for `*ptr` is the `SsaVarId` of the address operand.
/// That is wrong for native code in both directions:
///
/// - It is **imprecise**: `[rbp-8]` and `[rbp-16]` compute two different
///   address values, so nothing relates them — even though they provably do not
///   overlap. Every stack slot looks unrelated to every other.
/// - It is **unstable**: the address is produced by a [`SsaOp::PtrAdd`], which
///   is a pure op, so GVN and LICM freely re-number and hoist it. The identity
///   of a memory cell would then depend on which optimization ran.
///
/// Decoding the address (via [`crate::analysis::address::normalize_address`])
/// fixes both: the cell is named by what it *is*, not by which instruction computed
/// it, and two accesses off a common base become comparable by offset.
///
/// # Soundness
///
/// Distinct `base` values are treated as **may-alias**. Two unrelated SSA
/// pointers can hold the same address, so nothing in the address expression
/// alone can rule that out; separating them requires a points-to oracle, which
/// is not available to a pure function on locations (see
/// [`pointsto`](crate::analysis::pointsto)). This is deliberately weaker than
/// the managed-code variants above, which key on object identity.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct IndirectLocation {
    /// Root value the address is measured from.
    pub base: SsaVarId,
    /// Scaled index term, when the address carries one.
    pub index: Option<SsaVarId>,
    /// Stride applied to [`Self::index`], in bytes.
    ///
    /// Meaningless without an index, and therefore held at `0` whenever
    /// [`Self::index`] is `None` — see [`IndirectLocation::new`]. Keeping that
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
    pub address_space: Option<u16>,
    /// Target pointer width this address was decoded against.
    ///
    /// Address arithmetic wraps here, so the width is part of what the offset
    /// *means*: `+4294967288` and `-8` are one displacement on a 32-bit target
    /// and two distinct ones on a 64-bit target.
    /// [`normalize_address`]
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
        base: SsaVarId,
        index: Option<SsaVarId>,
        stride_bytes: u64,
        offset_bits: i64,
        size_bits: Option<u32>,
        address_space: Option<u16>,
        ptr_size: PointerSize,
    ) -> Self {
        Self {
            base,
            index,
            stride_bytes: if index.is_some() { stride_bytes } else { 0 },
            offset_bits,
            size_bits,
            address_space,
            ptr_size,
        }
    }

    /// Returns `true` when this access runs past the top of the address space.
    ///
    /// Displacements are canonicalised into the signed pointer-width range, so
    /// an access whose *end* exceeds that range wraps around to the bottom — and
    /// can then overlap a low-offset access that the linear extent comparison
    /// calls disjoint.
    fn crosses_wrap_boundary(&self) -> bool {
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
    #[must_use]
    pub fn may_alias(&self, other: &Self) -> bool {
        // Distinct address spaces are disjoint by construction: a segmented
        // access and a flat one at the same numeric offset name different
        // memory, so this holds even when everything else matches.
        if self.address_space != other.address_space {
            return false;
        }
        // Different roots: two distinct SSA pointers can hold the same address.
        if self.base != other.base {
            return true;
        }
        // A scaled index term whose contribution we cannot equate defeats any
        // offset reasoning — the indices could take the same value.
        if !self.index_matches(other) {
            return true;
        }
        self.extents_overlap(other)
    }

    /// Returns `true` if the two accesses provably name exactly one cell.
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
    /// the runtime value. Anything else — different index values, different
    /// strides, or an index on only one side (it could be zero) — is not
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
    /// Returns the memory location accessed by this operation.
    #[must_use]
    pub fn location(&self) -> &MemoryLocation<T> {
        match self {
            Self::Load { location, .. }
            | Self::Store { location, .. }
            | Self::ReadWrite { location, .. }
            | Self::Barrier { location, .. } => location,
        }
    }

    /// Returns the block index containing this operation.
    #[must_use]
    pub fn block(&self) -> usize {
        match self {
            Self::Load { block, .. }
            | Self::Store { block, .. }
            | Self::ReadWrite { block, .. }
            | Self::Barrier { block, .. } => *block,
        }
    }

    /// Returns the instruction index within the block.
    #[must_use]
    pub fn instr(&self) -> usize {
        match self {
            Self::Load { instr, .. }
            | Self::Store { instr, .. }
            | Self::ReadWrite { instr, .. }
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
        matches!(
            self,
            Self::Store { .. } | Self::ReadWrite { .. } | Self::Barrier { .. }
        )
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
            Self::ReadWrite { effects, .. } | Self::Barrier { effects, .. } => Some(*effects),
            Self::Load { .. } | Self::Store { .. } => None,
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
    Entry,
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
}

/// One step of the dominator-tree walk in
/// [`MemorySsa::rename_memory_versions`].
///
/// The walk is an explicit-stack DFS rather than recursion, so the scope
/// restore that recursion would get for free is modelled as its own step.
enum RenameStep<T: Target> {
    /// Rename this block, then descend into its dominator-tree children.
    Enter(usize),
    /// Leaving a block's dominator subtree: pop one version for each location
    /// listed, restoring the rename state that block's siblings must observe.
    Exit(Vec<MemoryLocation<T>>),
}

/// Memory SSA representation.
///
/// This structure tracks versioned memory locations throughout a function,
/// enabling precise tracking of memory state for analysis.
#[derive(Debug)]
pub struct MemorySsa<T: Target> {
    /// Next version number for each memory location.
    next_version: HashMap<MemoryLocation<T>, u32>,

    /// Memory phi nodes at each block.
    /// Key is block index, value is list of memory phi nodes.
    memory_phis: BTreeMap<usize, Vec<MemoryPhi<T>>>,

    /// Definition sites for each memory version.
    definitions: HashMap<MemoryVersion<T>, MemoryDefSite>,

    /// Memory version at block entry, keyed by `(interned location id, block)`.
    ///
    /// Keyed by an interned id rather than a cloned [`MemoryLocation`]: the map
    /// holds up to `blocks × locations` entries, and an `Indirect` location is
    /// ~48 bytes to clone and hash for every one of them.
    entry_versions: HashMap<(u32, usize), u32>,

    /// Memory version at block exit, keyed by `(interned location id, block)`.
    exit_versions: HashMap<(u32, usize), u32>,

    /// Interning table: location to its index in [`Self::ordered_locations`].
    location_ids: HashMap<MemoryLocation<T>, u32>,

    /// All identified memory operations.
    operations: Vec<MemoryOp<T>>,

    /// All unique memory locations in the function.
    locations: HashSet<MemoryLocation<T>>,
    /// The same locations in first-appearance order.
    ///
    /// `locations` is a `HashSet`, so iterating it is nondeterministic across
    /// runs. Version numbers and the order availability is seeded both derive
    /// from that iteration, and the downstream similarity pipeline is
    /// content-addressed — it requires byte-identical optimized IR. This list is
    /// built from `operations`, which is instruction order.
    ordered_locations: Vec<MemoryLocation<T>>,
}

impl<T: Target> MemorySsa<T> {
    /// Creates an empty Memory SSA structure.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_version: HashMap::new(),
            memory_phis: BTreeMap::new(),
            definitions: HashMap::new(),
            entry_versions: HashMap::new(),
            exit_versions: HashMap::new(),
            location_ids: HashMap::new(),
            operations: Vec::new(),
            locations: HashSet::new(),
            ordered_locations: Vec::new(),
        }
    }

    /// Builds Memory SSA from an SSA function.
    ///
    /// This performs the full Memory SSA construction:
    /// 1. Identify all memory operations
    /// 2. Place memory phi nodes at dominance frontiers
    /// 3. Rename memory versions using dominator tree traversal
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to analyze.
    /// * `cfg` - The control flow graph of the function.
    /// * `ptr_size` - Target pointer width, used to canonicalise folded address
    ///   displacements. Address arithmetic wraps at this width, and the model
    ///   cannot tell a sign-extended `-8` from a zero-extended `0xFFFF_FFF8`
    ///   without it — see
    ///   [`normalize_address`].
    ///
    /// # Returns
    ///
    /// A complete Memory SSA representation.
    #[must_use]
    pub fn build(ssa: &SsaFunction<T>, cfg: &SsaCfg<'_, T>, ptr_size: PointerSize) -> Self {
        let mut mem_ssa = Self::new();

        // Phase 1: Identify all memory operations
        mem_ssa.identify_memory_operations(ssa, ptr_size);

        // Renaming records an entry and an exit version for every location in
        // every block, so the retained size is `2 × blocks × locations`. A
        // function that is really misread data yields a distinct `Indirect`
        // location per distinct `base + offset`, so the location count scales
        // with the block count and the product is quadratic.
        //
        // Past the budget, drop the analysis rather than the process: an empty
        // `MemorySsa` reports no locations, every alias query falls back to the
        // conservative answer, and the memory pass simply finds nothing to do.
        let budget = MAX_MEMORY_SSA_CELLS;
        let cells = mem_ssa
            .ordered_locations
            .len()
            .saturating_mul(ssa.block_count());
        if cells > budget {
            log::warn!(
                "memory SSA exceeded its size budget ({} locations x {} blocks = {cells} cells, \
                 bound {budget}); skipping memory analysis for this function",
                mem_ssa.ordered_locations.len(),
                ssa.block_count(),
            );
            return Self::new();
        }

        // Phase 2: Place memory phi nodes
        mem_ssa.place_memory_phis(cfg);

        // Phase 3: Rename memory versions
        mem_ssa.rename_memory_versions(ssa, cfg, ptr_size);

        mem_ssa
    }

    /// Returns the memory phi nodes at a block.
    #[must_use]
    pub fn memory_phis(&self, block: usize) -> &[MemoryPhi<T>] {
        self.memory_phis.get(&block).map_or(&[], Vec::as_slice)
    }

    /// Returns all memory operations.
    #[must_use]
    pub fn operations(&self) -> &[MemoryOp<T>] {
        &self.operations
    }

    /// Returns all unique memory locations.
    #[must_use]
    pub fn locations(&self) -> &HashSet<MemoryLocation<T>> {
        &self.locations
    }

    /// Returns the memory version at block entry for a location.
    #[must_use]
    pub fn version_at_entry(&self, location: &MemoryLocation<T>, block: usize) -> Option<u32> {
        let id = *self.location_ids.get(location)?;
        self.entry_versions.get(&(id, block)).copied()
    }

    /// Returns the memory version at block exit for a location.
    #[must_use]
    pub fn version_at_exit(&self, location: &MemoryLocation<T>, block: usize) -> Option<u32> {
        let id = *self.location_ids.get(location)?;
        self.exit_versions.get(&(id, block)).copied()
    }

    /// Returns the definition site for a memory version.
    #[must_use]
    pub fn definition(&self, version: &MemoryVersion<T>) -> Option<MemoryDefSite> {
        self.definitions.get(version).copied()
    }

    /// Returns the next version number for a location (and increments it).
    fn allocate_version(&mut self, location: &MemoryLocation<T>) -> u32 {
        let version = self.next_version.entry(location.clone()).or_insert(0);
        let result = *version;
        *version = version.saturating_add(1);
        result
    }

    /// Phase 1: Identify all memory operations in the SSA function.
    fn identify_memory_operations(&mut self, ssa: &SsaFunction<T>, ptr_size: PointerSize) {
        for (block_idx, instr_idx, instr) in ssa.iter_instructions() {
            if let Some(mem_op) =
                Self::classify_memory_operation(ssa, instr.op(), block_idx, instr_idx, ptr_size)
            {
                if self.locations.insert(mem_op.location().clone()) {
                    let id = u32::try_from(self.ordered_locations.len()).unwrap_or(u32::MAX);
                    self.location_ids.insert(mem_op.location().clone(), id);
                    self.ordered_locations.push(mem_op.location().clone());
                }
                self.operations.push(mem_op);
            }
        }
    }

    /// Classifies an SSA operation as a memory operation, if applicable.
    fn classify_memory_operation(
        ssa: &SsaFunction<T>,
        op: &SsaOp<T>,
        block: usize,
        instr: usize,
        ptr_size: PointerSize,
    ) -> Option<MemoryOp<T>> {
        match op {
            SsaOp::LoadField {
                dest,
                object,
                field,
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
            } => {
                let location = MemoryLocation::InstanceField(*object, field.clone());
                Some(MemoryOp::Store {
                    location,
                    value: *value,
                    block,
                    instr,
                })
            }
            SsaOp::LoadStaticField { dest, field } => {
                let location = MemoryLocation::StaticField(field.clone());
                Some(MemoryOp::Load {
                    location,
                    dest: *dest,
                    block,
                    instr,
                })
            }
            SsaOp::StoreStaticField { field, value } => {
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
                let idx = Self::resolve_array_index(ssa, *index);
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
                let idx = Self::resolve_array_index(ssa, *index);
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
                address_space,
            } => {
                let location =
                    Self::indirect_location(ssa, *addr, value_type, *address_space, ptr_size);
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
                address_space,
            } => {
                let location =
                    Self::indirect_location(ssa, *addr, value_type, *address_space, ptr_size);
                Some(MemoryOp::Store {
                    location,
                    value: *value,
                    block,
                    instr,
                })
            }
            _ => {
                let effects = op.effects();
                match effects.kind {
                    SsaEffectKind::ReadWrite | SsaEffectKind::Atomic => Some(MemoryOp::ReadWrite {
                        location: MemoryLocation::Unknown,
                        effects,
                        block,
                        instr,
                    }),
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
                    SsaEffectKind::Read => op.dest().map(|dest| MemoryOp::Load {
                        location: MemoryLocation::Unknown,
                        dest,
                        block,
                        instr,
                    }),
                    SsaEffectKind::Write => Some(MemoryOp::ReadWrite {
                        location: MemoryLocation::Unknown,
                        effects,
                        block,
                        instr,
                    }),
                    SsaEffectKind::Pure
                    | SsaEffectKind::Fence
                    | SsaEffectKind::Call
                    | SsaEffectKind::Opaque => None,
                }
            }
        }
    }

    /// Decodes an indirect access into an offset-aware memory location.
    ///
    /// The address is normalized to `base + index*stride + offset` rather than
    /// keyed on `addr` itself, so the resulting cell identity survives the
    /// address computation being re-numbered or hoisted. See
    /// [`IndirectLocation`].
    fn indirect_location(
        ssa: &SsaFunction<T>,
        addr: SsaVarId,
        value_type: &T::Type,
        address_space: Option<u16>,
        ptr_size: PointerSize,
    ) -> MemoryLocation<T> {
        let address = normalize_address(ssa, addr, ptr_size);
        MemoryLocation::Indirect(IndirectLocation::new(
            address.base,
            address.index,
            address.stride_bytes,
            address.offset_bits,
            T::bit_width(value_type),
            address_space,
            ptr_size,
        ))
    }

    /// Resolves an array index to an [`ArrayIndex`] abstraction, folding a
    /// constant index so that distinct constant elements stop may-aliasing.
    fn resolve_array_index(ssa: &SsaFunction<T>, index_var: SsaVarId) -> ArrayIndex {
        match const_i64(ssa, index_var) {
            Some(index) => ArrayIndex::Constant(index),
            None => ArrayIndex::Variable(index_var),
        }
    }

    /// Phase 2: Place memory phi nodes at dominance frontiers.
    fn place_memory_phis(&mut self, cfg: &SsaCfg<'_, T>) {
        let block_count = cfg.node_count();
        if block_count == 0 {
            return;
        }

        // Compute dominators and dominance frontiers
        let dom_tree = compute_dominators(cfg, cfg.entry());
        let frontiers = compute_dominance_frontiers(cfg, &dom_tree);

        // A definition of `L` is also a definition of every location it may
        // alias. Memory versions are per-location, so without this a store to
        // `*(p+16)` bumps only its own version and a barrier bumps only
        // `Unknown` — leaving the version of a may-aliasing cell unchanged and
        // therefore still looking forwardable across a block boundary.
        let mut def_blocks: HashMap<MemoryLocation<T>, BTreeSet<usize>> = HashMap::new();
        for op in &self.operations {
            if !op.defines_memory() {
                continue;
            }
            let defined = op.location();
            for location in &self.ordered_locations {
                if location.may_alias(defined) {
                    def_blocks
                        .entry(location.clone())
                        .or_default()
                        .insert(op.block());
                }
            }
        }

        // Standard phi placement algorithm (iterated dominance frontier).
        // Iterated in `ordered_locations` order so version numbering does not
        // depend on `HashMap` iteration order.
        let ordered = self.ordered_locations.clone();
        for location in ordered {
            let Some(defs) = def_blocks.get(&location).cloned() else {
                continue;
            };
            let mut phi_blocks: BTreeSet<usize> = BTreeSet::new();
            let mut worklist: VecDeque<usize> = defs.iter().copied().collect();
            let mut processed: BTreeSet<usize> = BTreeSet::new();

            while let Some(block) = worklist.pop_front() {
                if !processed.insert(block) {
                    continue;
                }

                let node_id = NodeId::new(block);
                if node_id.index() >= frontiers.len() {
                    continue;
                }

                let Some(frontier_set) = frontiers.get(node_id.index()) else {
                    continue;
                };
                for frontier_block in frontier_set.iter() {
                    if phi_blocks.insert(frontier_block) {
                        // Add phi at frontier
                        let version = self.allocate_version(&location);
                        let phi = MemoryPhi::new(location.clone(), version);
                        self.memory_phis
                            .entry(frontier_block)
                            .or_default()
                            .push(phi);
                        self.definitions.insert(
                            MemoryVersion::new(location.clone(), version),
                            MemoryDefSite::Phi {
                                block: frontier_block,
                            },
                        );
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
    /// panics — a blown call stack is not a recoverable error.
    ///
    /// Version *numbering* is an opaque identity, not an ordering: phi versions
    /// are allocated during phi placement (phase 2), so a location carrying phis
    /// receives its entry version here with a number above them. Consumers that
    /// need to recognise the entry version match [`MemoryDefSite::Entry`] via
    /// [`MemorySsa::definition`] rather than comparing against zero.
    fn rename_memory_versions(
        &mut self,
        ssa: &SsaFunction<T>,
        cfg: &SsaCfg<'_, T>,
        ptr_size: PointerSize,
    ) {
        let block_count = cfg.node_count();
        if block_count == 0 {
            return;
        }

        // Compute dominators for traversal order
        let dom_tree = compute_dominators(cfg, cfg.entry());

        // Live version stack per location; the top is the version reaching the
        // block currently being renamed.
        let mut version_stacks: HashMap<MemoryLocation<T>, Vec<u32>> = HashMap::new();

        // Seed every location with its function-entry version.
        let locations = self.ordered_locations.clone();
        for location in locations {
            let entry_version = self.allocate_version(&location);
            version_stacks
                .entry(location.clone())
                .or_default()
                .push(entry_version);
            self.definitions.insert(
                MemoryVersion::new(location, entry_version),
                MemoryDefSite::Entry,
            );
        }

        // Rename in dominator-tree preorder, restoring scope on the way out.
        let mut visited = vec![false; block_count];
        let mut worklist = vec![RenameStep::Enter(cfg.entry().index())];

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

                    let pushed =
                        self.rename_block(block_idx, ssa, cfg, &mut version_stacks, ptr_size);

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
                        if let Some(stack) = version_stacks.get_mut(&location) {
                            stack.pop();
                        }
                    }
                }
            }
        }
    }

    /// Renames memory versions within a single block.
    ///
    /// Returns one entry per version pushed onto [`version_stacks`], which the
    /// caller pops when leaving this block's dominator subtree. A block that
    /// stores twice to the same location pushes twice and so appears twice.
    ///
    /// [`version_stacks`]: MemorySsa::rename_memory_versions
    fn rename_block(
        &mut self,
        block_idx: usize,
        ssa: &SsaFunction<T>,
        cfg: &SsaCfg<'_, T>,
        version_stacks: &mut HashMap<MemoryLocation<T>, Vec<u32>>,
        ptr_size: PointerSize,
    ) -> Vec<MemoryLocation<T>> {
        let mut pushed: Vec<MemoryLocation<T>> = Vec::new();

        // Record entry versions. Every location was seeded with a stack, so
        // iterating the stacks covers `self.locations` without cloning it.
        for (location, stack) in version_stacks.iter() {
            if let Some((&version, &id)) = stack.last().zip(self.location_ids.get(location)) {
                self.entry_versions.insert((id, block_idx), version);
            }
        }

        // Process memory phi nodes - they define new versions
        if let Some(phis) = self.memory_phis.get(&block_idx).cloned() {
            for phi in phis {
                version_stacks
                    .entry(phi.location.clone())
                    .or_default()
                    .push(phi.result_version);
                pushed.push(phi.location);
            }
        }

        // Process instructions in the block
        let Some(block) = ssa.block(block_idx) else {
            return pushed;
        };

        for (instr_idx, instr) in block.instructions().iter().enumerate() {
            // Handle stores - create new version
            if let Some(mem_op) =
                Self::classify_memory_operation(ssa, instr.op(), block_idx, instr_idx, ptr_size)
                && mem_op.defines_memory()
            {
                // The op defines its own location *and clobbers every location
                // it may alias*. A store to `*(p+16)` invalidates the overlapping
                // `*(p+0)`; a call, fence, atomic, volatile or opaque op
                // classifies to `Unknown`, which may-aliases everything and so
                // invalidates all of memory. Versioning only the named location
                // is what let a value survive across a block boundary that a
                // barrier or an overlapping store had already destroyed.
                //
                // Matches `place_memory_phis`, which places phis for the same
                // set, so every clobbered location has a merge point.
                let defined = mem_op.location().clone();
                let clobbered: Vec<MemoryLocation<T>> = self
                    .ordered_locations
                    .iter()
                    .filter(|candidate| candidate.may_alias(&defined))
                    .cloned()
                    .collect();
                for location in clobbered {
                    let new_version = self.allocate_version(&location);
                    version_stacks
                        .entry(location.clone())
                        .or_default()
                        .push(new_version);
                    self.definitions.insert(
                        MemoryVersion::new(location.clone(), new_version),
                        MemoryDefSite::Store {
                            block: block_idx,
                            instr: instr_idx,
                        },
                    );
                    pushed.push(location);
                }
            }
        }

        // Record exit versions
        for (location, stack) in version_stacks.iter() {
            if let Some((&version, &id)) = stack.last().zip(self.location_ids.get(location)) {
                self.exit_versions.insert((id, block_idx), version);
            }
        }

        // Fill in phi operands for successors
        for succ_id in cfg.successors(NodeId::new(block_idx)) {
            let succ_idx = succ_id.index();
            if let Some(phis) = self.memory_phis.get_mut(&succ_idx) {
                for phi in phis {
                    if let Some(&version) = version_stacks.get(&phi.location).and_then(|s| s.last())
                    {
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
        let total_phis = self.memory_phis.values().map(Vec::len).sum();
        let store_count = self
            .operations
            .iter()
            .filter(|op| matches!(op, MemoryOp::Store { .. } | MemoryOp::ReadWrite { .. }))
            .count();
        let load_count = self.operations.iter().filter(|op| op.is_load()).count();
        let barrier_count = self
            .operations
            .iter()
            .filter(|op| matches!(op, MemoryOp::Barrier { .. }))
            .count();

        MemorySsaStats {
            location_count: self.locations.len(),
            memory_phi_count: total_phis,
            store_count,
            load_count,
            barrier_count,
            version_count: self.definitions.len(),
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
    /// Number of barrier, call, or opaque memory operations.
    pub barrier_count: usize,
    /// Total number of memory versions.
    pub version_count: usize,
}

/// Memory state tracker for path-aware evaluation.
///
/// This tracks the memory values along a specific execution path, enabling
/// precise tracking of memory contents during symbolic or concrete evaluation.
#[derive(Debug, Clone)]
pub struct MemoryState<T: Target> {
    /// Current memory values: location -> (version, value as SSA variable).
    values: HashMap<MemoryLocation<T>, (u32, SsaVarId)>,
}

impl<T: Target> MemoryState<T> {
    /// Creates a new empty memory state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Records a memory store.
    pub fn store(&mut self, location: MemoryLocation<T>, value: SsaVarId, version: u32) {
        self.values.insert(location, (version, value));
    }

    /// Loads from a memory location.
    ///
    /// Returns the SSA variable holding the value, if known.
    #[must_use]
    pub fn load(&self, location: &MemoryLocation<T>) -> Option<SsaVarId> {
        // Direct match
        if let Some((_, value)) = self.values.get(location) {
            return Some(*value);
        }

        // Check for aliasing locations.
        //
        // More than one stored location can must-alias the query, and they can
        // hold different values. Returning whichever the map yielded first made
        // the answer a function of the hasher's per-instance seed, so the same
        // bytes could lift to different SSA on consecutive runs. The most
        // recently stored version wins, with the lower variable id breaking a
        // tie — a rule that is both deterministic and the one a load actually
        // wants.
        self.values
            .iter()
            .filter(|(loc, _)| location.must_alias(loc))
            .max_by_key(|(_, (version, value))| (*version, std::cmp::Reverse(*value)))
            .map(|(_, (_, value))| *value)
    }

    /// Returns the current version for a location, if known.
    #[must_use]
    pub fn version(&self, location: &MemoryLocation<T>) -> Option<u32> {
        self.values.get(location).map(|(v, _)| *v)
    }

    /// Checks if any stored location may alias the given location.
    #[must_use]
    pub fn has_may_alias(&self, location: &MemoryLocation<T>) -> bool {
        self.values.keys().any(|loc| loc.may_alias(location))
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
