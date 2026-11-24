//! Memory optimization pass - store-to-load forwarding, redundant load
//! elimination, and dead store elimination.
//!
//! Every rewrite here is gated on a Memory SSA alias *proof*
//! ([`MemoryLocation::must_alias`] / [`MemoryLocation::may_alias`]), never on
//! syntactic shape. That is what makes the pass safe on native code, where two
//! textually unrelated addresses routinely name one cell and one address
//! expression routinely names several.
//!
//! # Transforms
//!
//! - **Store-to-load forwarding.** A load whose location must-alias an earlier
//!   store's, with nothing in between that may-alias it, is replaced by the
//!   stored value. A load narrower than the store reads the stored value's
//!   bits at its place: the value narrowed where the two start together, a
//!   slice of it where the load reads a part further in (the high half of a
//!   doubleword, a word of a vector a structure store wrote). A load wider
//!   than any one store that two or more available stores tile exactly,
//!   through one base and contiguous, each holding a value of its own width,
//!   is their composition in the target's byte order (a word read back over
//!   four byte stores), where the byte order is stated.
//! - **Copies in parts rejoined.** Stores that put the parts loads read of
//!   one value stored whole back together at another place, every part once
//!   and nothing reading or writing those bytes between the first and the
//!   last, are one store of the value at the last: a record moved in two
//!   vector registers is the record stored again, whatever its type can be
//!   sliced into.
//! - **Redundant load elimination.** The same rule with an earlier *load* as
//!   the source: the second load is replaced by the first load's result.
//! - **Dead store elimination.** A store overwritten by a later must-aliasing
//!   store, with no possible read in between, is removed.
//! - **Silent stack stores.** A store into the stack (an address derived from
//!   the entry stack pointer) of the value just loaded from the cell, with
//!   nothing that may alias it written since, writes nothing and is removed:
//!   x86-64's stack probe `or qword [rsp], 0` loads a word of the frame it
//!   just allocated and stores it back, in line or in the loop allocating a
//!   large frame. Memory the function was handed stays written: a device
//!   register written back may act on it.
//! - **Image load folding.** A load of a cell the host's image reports
//!   read-only is replaced by the constant the image holds there. Memory SSA
//!   gives such a cell one version, [`MemoryDefSite::Image`], because a write
//!   to it faults and no execution continues past one - so this is
//!   store-to-load forwarding whose store is the image, and needs no proof
//!   about what ran before the function.
//! - **Frame-exit dead store elimination.** A store into the function's own
//!   frame (memory strictly below the stack pointer it was entered with, as the
//!   host declares through
//!   [`VariableRole::EntryStackPointer`](crate::ir::variable::VariableRole::EntryStackPointer))
//!   that nothing can observe before the function leaves is removed. The
//!   caller cannot reach that memory once the function returns, so the only
//!   possible readers are the function's own accesses, and an unread write to
//!   it is a callee-saved register save, a spill nothing reloads, or an
//!   initialisation the program overwrites. The same solve tracks the cells
//!   at or above that pointer a store writes - the stack arguments, the
//!   return address, the caller's frame - as the memory that outlives the
//!   function: live wherever it leaves, by a return or a throw, and read by
//!   any access that may name them, so only an overwrite before it leaves
//!   ends one. A call's own argument bytes are such an overwrite where they
//!   reach that high: an Arm `_start` pushes `stack_end` into the word it
//!   popped `argc` from, and the call passing it owns the word.
//!
//! # Scope
//!
//! Forwarding and redundant-load elimination reach **across blocks**: a load
//! can be served by a store in a dominating block, read out of Memory SSA's
//! block-entry version. Overwrite-based dead store elimination is
//! deliberately **block-local** - proving a store overwritten across a merge
//! would need every path to overwrite it. Frame-exit elimination is the one
//! cross-block form: one backward liveness solve over the frame's cells, and
//! the caller's, covers every path from every store. Any load, read-write or
//! barrier on a path that may alias the cell keeps it, a block whose
//! continuation is not fully known (an unresolved indirect branch, a handler
//! end, a missing terminator) keeps it, and only a must-aliasing overwrite or
//! a real exit ends a path.
//!
//! # Exceptions
//!
//! A throw from any block of a protected region continues at the region's
//! handlers, not only a throw from the block that guards it. Both forms of dead
//! store elimination treat every instruction that may throw inside a protected
//! region as a path to those handlers: an overwrite that a fault can pre-empt
//! does not kill the store before it, and a handler that reads a frame cell
//! keeps every write to it the region can fault after. An exception a callee
//! raises, or a throw, may also leave the function for the caller it unwinds
//! to ([`SsaOp::may_unwind`]), so both forms treat such a call as an exit at
//! which all memory but the function's own frame is live. A fault is no such
//! exit: whether one reaches the caller is the runtime's to say, and like a
//! signal handler's read (below) it is not modelled.
//!
//! # Barriers
//!
//! Calls, fences, atomics, volatile accesses, and opaque native operations
//! classify to a barrier or an `Unknown` location. Both invalidate the whole
//! availability map and every pending dead-store candidate, so no value is
//! carried across an operation whose memory effect is not modelled.
//!
//! # What this does not model
//!
//! Removing a store is safe against anything *in-process*: a call is a barrier,
//! so a callee can never observe a store this pass removed, and a store is only
//! removed once it is proven overwritten before any read, or proven to be a
//! frame cell nothing reads before the function leaves. It is not
//! safe against a concurrent observer of a non-atomic, non-volatile cell - a
//! racing thread or a signal handler could read the intermediate value. That is
//! the same latitude an optimizing compiler takes, and the reason atomics,
//! volatile accesses, and fences are all barriers here.
//!
//! # Faults
//!
//! Removing an access changes *where* a memory fault would be raised, though
//! not whether one is. Forwarding a load past a must-aliasing store cannot
//! introduce a fault: the store already proved the address writable. Dead store
//! elimination drops a write whose address is proven written again later on the
//! same straight-line path, so the surviving store faults where the removed one
//! would have. Frame-exit elimination drops a write into the function's own
//! frame with nothing after it to fault in its place; a fault there is a stack
//! overflow, which this pass does not preserve.

use std::{cmp::Reverse, collections::BTreeSet};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    BitSet, Result,
    analysis::{
        address::AddressBase,
        budget::{Budget, Exhaustion},
        cfg::SsaCfg,
        context::AnalysisContext,
        exceptions::FunctionRoots,
        frame::{CellKey, FrameAnchors, FrameEscape},
        image::{ImageData, ImageRead},
        memory::{IndirectLocation, MemoryDefSite, MemoryLocation, MemoryOp, MemorySsa},
        reach::IndexBounds,
    },
    events::EventKind,
    ir::{
        function::{SsaEditOptions, SsaEditor, SsaFunction, SsaRollbackPolicy},
        instruction::{InstrId, InstructionRole, SsaInstruction},
        ops::{
            AccessMode, AccessQualifiers, SsaEffectKind, SsaOp, VectorMaskMode, VectorPackKind,
            VectorSegmentLayout,
        },
        phi::PhiOperand,
        value::ConstValue,
        variable::{DefSite, SsaVarId, VariableOrigin},
    },
    scheduling::{FunctionPass, ModificationScope, PassEnv},
    target::{Endianness, Target, VectorLaneKind, VectorShape},
};

/// One stored value of a load several stores tile.
#[derive(Debug, Clone, Copy)]
struct Piece {
    /// The value the store wrote.
    value: SsaVarId,
    /// The bit of the load its lowest bit lands on.
    shift: u32,
    /// The width the store wrote.
    width: u32,
}

/// A rewrite the analysis proved safe, applied in one editor session.
enum Rewrite<T: Target> {
    /// Replace every use of `redundant` with `available`, then nop the
    /// instruction at `(block, instr)` that defined it.
    Forward {
        /// Value produced by the redundant access.
        redundant: SsaVarId,
        /// The type the redundant access read, when the function states it:
        /// what the available value is reinterpreted as where its own type
        /// differs.
        read_type: Option<T::Type>,
        /// Value already holding that memory's contents.
        available: SsaVarId,
        /// Block of the access to remove.
        block: usize,
        /// Instruction index of the access to remove.
        instr: usize,
    },
    /// Remove the dead store at `(block, instr)`.
    DeadStore {
        /// Block of the store to remove.
        block: usize,
        /// Instruction index of the store to remove.
        instr: usize,
    },
    /// Remove the load at `(block, instr)`: nothing reads its result, and it
    /// reads a cell the image places, so it cannot fault either.
    ///
    /// The read is reported as a fold's is ([`Target::record_image_read`]):
    /// the program still reads the cell there, and a host recovering which
    /// objects its code touches counts the read with or without the load.
    DeadImageLoad {
        /// Block of the load to remove.
        block: usize,
        /// Instruction index of the load to remove.
        instr: usize,
        /// The address it reads, in the host's symbol space.
        address: u64,
        /// How many bytes it reads.
        len: u64,
        /// The type the bytes are read as: the load's own, or the pointer a
        /// cell the image proves holds an address is, as a fold of it reads it.
        read_type: T::Type,
    },
    /// Replace the load at `(block, instr)` by the value the stores it reads
    /// from compose: several narrower stores tile the bytes it reads exactly,
    /// and each piece is placed at its bits and the whole read as the load's
    /// type.
    Compose {
        /// The load's result, which the composition now defines.
        dest: SsaVarId,
        /// The type the load reads.
        read_type: T::Type,
        /// The load's width in bits.
        bits: u32,
        /// Each stored value, where it lands in the load.
        pieces: Vec<Piece>,
        /// Block of the load.
        block: usize,
        /// Instruction index of the load.
        instr: usize,
    },
    /// Replace the load at `(block, instr)` by the part of one wider stored
    /// value it reads: `bits` from bit `from` of `value`, a lane of a vector
    /// or the shifted bits of an integer.
    Slice {
        /// The load's result, which the slice now defines.
        dest: SsaVarId,
        /// The type the load reads.
        read_type: T::Type,
        /// The stored value the load reads part of.
        value: SsaVarId,
        /// Whether the load reads from where the value was stored whole, so
        /// the host keeps the value there ([`SsaEditor::note_whole`]).
        at_start: bool,
        /// The bit of the value the load's lowest bit is.
        from: u32,
        /// The load's width in bits.
        bits: u32,
        /// Block of the load.
        block: usize,
        /// Instruction index of the load.
        instr: usize,
    },
    /// Remove the load at `(block, instr)`: nothing reads its result, and it
    /// reads the function's own frame, below the stack pointer it was entered
    /// with, or the caller's last stack word at it, which is the stack the
    /// function runs on and cannot fault.
    DeadFrameLoad {
        /// Block of the load to remove.
        block: usize,
        /// Instruction index of the load to remove.
        instr: usize,
    },
    /// Remove the load at `(block, instr)`: nothing reads its result, and it
    /// reads the first word of a symbol the host proves the loader binds
    /// ([`Target::symbol_bound`]), storage mapped before the function ran.
    DeadBoundLoad {
        /// Block of the load to remove.
        block: usize,
        /// Instruction index of the load to remove.
        instr: usize,
    },
    /// Replace the stores at `stores` of `block`, which put the parts of one
    /// stored value back together at another place, by one store of the
    /// value at the last of them: a record copied in two vector moves.
    Rejoin {
        /// The value the parts are of.
        value: SsaVarId,
        /// The address the part at the value's start is stored at.
        addr: SsaVarId,
        /// How that part is stored.
        access: AccessQualifiers,
        /// Block of the stores.
        block: usize,
        /// Instruction index of each store of a part, in block order; the
        /// last becomes the store of the value.
        stores: Vec<usize>,
    },
}

/// A load of read-only image data, and the constants its bytes hold.
struct Fold<T: Target> {
    /// Each value the load defines and the constant that now defines it, the
    /// first in the load's place: one for a load or a broadcast, one per
    /// segment for a segment load.
    values: Vec<(SsaVarId, ConstValue<T>)>,
    /// The type the destination takes, when the constant is an address the
    /// load read as an integer (see [`ImageData::address_at`]).
    dest_type: Option<T::Type>,
    /// The address the load read.
    address: u64,
    /// How many bytes it read.
    len: u64,
    /// The type it read them as.
    read_type: T::Type,
    /// Block of the load.
    block: usize,
    /// Instruction index of the load.
    instr: usize,
}

/// The values the load `read` makes of the image's `bytes` at `address`, the
/// type it reads them as, and the type its destination takes where the image
/// proves a pointer-wide cell holds an address.
type ImageValues<T> = (
    Vec<(SsaVarId, ConstValue<T>)>,
    <T as Target>::Type,
    Option<<T as Target>::Type>,
);

/// A masked read of the image whose mask, and what its inactive lanes keep,
/// are known constants: each lane the mask enables comes from the image, and
/// every other lane is the passthrough's, or zero.
struct MaskedRead {
    /// Whether each lane is enabled, lane 0 first.
    enabled: Vec<bool>,
    /// The bytes every lane holds where the mask disables it: the
    /// passthrough's, or zeros, in the image's byte order.
    inactive: Vec<u8>,
    /// The bytes of one lane.
    lane_bytes: usize,
}

impl MaskedRead {
    /// The read a masked access of `vector_type` makes under the constant
    /// `mask`, inactive lanes keeping the constant `passthrough` where `mode`
    /// merges and zero where it zeroes; `None` where either is unknown, or
    /// the lanes are no whole bytes.
    ///
    /// The mask is one bit per lane, bit `i` governing lane `i` (an AVX-512
    /// opmask), or a vector of the access's lanes, each nonzero where active
    /// (an SVE predicate).
    fn of<T: Target>(
        ssa: &SsaFunction<T>,
        mask: SsaVarId,
        passthrough: Option<SsaVarId>,
        mode: VectorMaskMode,
        vector_type: &T::Type,
        order: Endianness,
        context: &AnalysisContext<'_>,
    ) -> Option<Self> {
        let shape = T::vector_shape(vector_type)?;
        if shape.lane_bits % 8 != 0 {
            return None;
        }
        let lane_bytes = usize::try_from(shape.lane_bits / 8).ok()?;
        let lanes = usize::try_from(shape.lane_count).ok()?;
        let constant = |var: SsaVarId| match ssa.get_definition(var)? {
            SsaOp::Const { value, .. } => Some(value.clone()),
            _ => None,
        };
        let enabled: Vec<bool> = match constant(mask)? {
            ConstValue::Vector(predicate) => {
                if usize::try_from(predicate.lanes()).ok()? != lanes {
                    return None;
                }
                predicate.lane_values().map(|lane| lane != 0).collect()
            }
            bits => {
                let (pattern, width) = bits.integer_pattern(context.machine().ptr_size)?;
                (0..lanes)
                    .map(|lane| {
                        let lane = u32::try_from(lane).ok()?;
                        Some(lane < width && pattern >> lane & 1 == 1)
                    })
                    .collect::<Option<_>>()?
            }
        };
        let inactive = match (mode, passthrough) {
            (VectorMaskMode::Zero, _) => vec![0; lanes.checked_mul(lane_bytes)?],
            (VectorMaskMode::Merge, Some(passthrough)) => {
                constant(passthrough)?.to_bytes(order, context.machine().ptr_size)?
            }
            (VectorMaskMode::Merge, None) => return None,
        };
        (inactive.len() == lanes.checked_mul(lane_bytes)?).then_some(Self {
            enabled,
            inactive,
            lane_bytes,
        })
    }

    /// The bytes of the vector the read leaves, from the image's `bytes`: an
    /// enabled lane is the lane at its own place in them, or, where the read
    /// `expands`, the next of the elements laid out one after another.
    fn compose(&self, bytes: &[u8], expands: bool) -> Option<Vec<u8>> {
        let mut composed = Vec::with_capacity(self.inactive.len());
        let mut next = 0usize;
        for (lane, &enabled) in self.enabled.iter().enumerate() {
            let at = lane.checked_mul(self.lane_bytes)?;
            let source = if !enabled {
                self.inactive.get(at..at.checked_add(self.lane_bytes)?)?
            } else if expands {
                let from = next.checked_mul(self.lane_bytes)?;
                next = next.checked_add(1)?;
                bytes.get(from..from.checked_add(self.lane_bytes)?)?
            } else {
                bytes.get(at..at.checked_add(self.lane_bytes)?)?
            };
            composed.extend_from_slice(source);
        }
        Some(composed)
    }
}

/// The results of the loads nothing reads that cannot fault, which the pass
/// removes: a load of the function's own frame or the caller's word at its
/// entry, of a cell the image places, or of the first word of a symbol the
/// loader binds. Empty where Memory SSA ran out of
/// budget.
///
/// # Arguments
///
/// * `ssa` - The SSA function being analyzed.
/// * `mem_ssa` - Its memory, built under `context`.
/// * `context` - The analysis context the pass runs under.
#[must_use]
pub fn unread_loads<T: Target>(
    ssa: &SsaFunction<T>,
    mem_ssa: &MemorySsa<T>,
    context: &AnalysisContext<'_>,
) -> BTreeSet<SsaVarId> {
    if mem_ssa.exhaustion().is_some() {
        return BTreeSet::new();
    }
    MemoryPlanner::new(ssa, mem_ssa, context)
        .dead_image_loads(&FxHashSet::default(), &FxHashSet::default())
        .iter()
        .filter_map(|rewrite| match rewrite {
            Rewrite::DeadFrameLoad { block, instr }
            | Rewrite::DeadBoundLoad { block, instr }
            | Rewrite::DeadImageLoad { block, instr, .. } => ssa
                .block(*block)
                .and_then(|body| body.instructions().get(*instr))
                .and_then(|instruction| instruction.op().dest()),
            _ => None,
        })
        .collect()
}

/// The stores into the function's own frame that nothing observes before the
/// function leaves, and those into the caller's memory above it that the
/// function overwrites before it leaves: the ones the pass removes at frame
/// exit, by identity. Empty where the host declared no entry stack pointer or
/// Memory SSA ran out of budget.
///
/// # Arguments
///
/// * `ssa` - The SSA function being analyzed.
/// * `mem_ssa` - Its memory, built under `context`.
/// * `context` - The analysis context the pass runs under.
#[must_use]
pub fn dead_frame_stores<T: Target>(
    ssa: &SsaFunction<T>,
    mem_ssa: &MemorySsa<T>,
    context: &AnalysisContext<'_>,
) -> BTreeSet<InstrId> {
    if mem_ssa.exhaustion().is_some() {
        return BTreeSet::new();
    }
    let graph = SsaCfg::from_ssa(ssa);
    let mut rewrites = Vec::new();
    MemoryPlanner::new(ssa, mem_ssa, context).frame_exit_dead_stores(&graph, &mut rewrites);
    rewrites
        .iter()
        .filter_map(|rewrite| match rewrite {
            Rewrite::DeadStore { block, instr } => ssa
                .block(*block)
                .and_then(|body| body.instructions().get(*instr))
                .and_then(SsaInstruction::id),
            _ => None,
        })
        .collect()
}

/// The planning of the rewrites over one function: the function, its Memory
/// SSA, its classified memory operations by position, and the context the
/// pass runs under.
struct MemoryPlanner<'a, T: Target> {
    /// The function planned on.
    ssa: &'a SsaFunction<T>,
    /// Its Memory SSA, which classifies every memory operation.
    mem_ssa: &'a MemorySsa<T>,
    /// The classified memory operations, by `(block, instr)`.
    by_position: FxHashMap<(usize, usize), &'a MemoryOp<T>>,
    /// The machine, image and budget the pass runs under.
    context: &'a AnalysisContext<'a>,
}

impl<'a, T: Target> MemoryPlanner<'a, T> {
    /// Plans over `ssa`, whose Memory SSA is `mem_ssa`, under `context`.
    fn new(
        ssa: &'a SsaFunction<T>,
        mem_ssa: &'a MemorySsa<T>,
        context: &'a AnalysisContext<'a>,
    ) -> Self {
        let by_position = mem_ssa
            .operations()
            .iter()
            .map(|op| ((op.block(), op.instr()), op))
            .collect();
        Self {
            ssa,
            mem_ssa,
            by_position,
            context,
        }
    }

    /// Plans a fold for every load of read-only image data whose bytes the image
    /// serves and whose type a constant can state.
    ///
    /// A load qualifies when Memory SSA reports its location immutable - which it
    /// only does for an absolute, whole-byte, unindexed access the image places in
    /// a read-only region. A scalar or a whole vector reads its type's bytes; a
    /// broadcast reads one element and splats it; a segment load reads its
    /// segments and deinterleaves them, one value each; a masked load, or an
    /// expanding one, under a constant mask reads the lanes the mask enables and
    /// keeps the passthrough, or zero, in the others ([`MaskedRead`]). The bytes
    /// are decoded in
    /// the image's byte order. Folding stops at the
    /// [`ImageBytes`](Budget::ImageBytes) budget, which is then returned.
    ///
    /// # Arguments
    ///
    fn image_folds(&self) -> (Vec<Fold<T>>, Option<Exhaustion>) {
        let Some(image) = self.context.image() else {
            return (Vec::new(), None);
        };
        let mut folds = Vec::new();
        let mut spent = 0usize;
        for op in self.mem_ssa.operations() {
            let location = match op {
                MemoryOp::Load { location, .. } | MemoryOp::Read { location, .. } => location,
                _ => continue,
            };
            let MemoryLocation::Indirect(cell) = location else {
                continue;
            };
            if !self.mem_ssa.is_immutable(location) {
                continue;
            }
            let (block, instr) = (op.block(), op.instr());
            let Some(read) = self
                .ssa
                .block(block)
                .and_then(|body| body.instructions().get(instr))
                .map(|instruction| instruction.op())
            else {
                continue;
            };
            let Some((address, extent)) = cell.image_extent(image, None) else {
                continue;
            };
            let Ok(len) = usize::try_from(extent) else {
                continue;
            };
            let Some(bytes) = image.bytes(address, len) else {
                continue;
            };
            let Some((values, read_type, dest_type)) =
                self.image_values(read, bytes, address, extent, image)
            else {
                continue;
            };
            spent = spent.saturating_add(len);
            if let Some(exhaustion) = self.context.budget().check(Budget::ImageBytes, spent) {
                return (folds, Some(exhaustion));
            }
            folds.push(Fold {
                values,
                read_type,
                dest_type,
                address,
                len: extent,
                block,
                instr,
            });
        }
        (folds, None)
    }

    /// What `read` makes of the image's `bytes`: see [`MemoryPlanner::image_folds`].
    fn image_values(
        &self,
        read: &SsaOp<T>,
        bytes: &[u8],
        address: u64,
        extent: u64,
        image: &dyn ImageData,
    ) -> Option<ImageValues<T>> {
        let order = image.endianness();
        match read {
            SsaOp::LoadIndirect {
                dest, value_type, ..
            } => {
                // A cell the host proves holds an address folds to the location it
                // names. Its width must be the pointer's, which is also what makes
                // the named symbol mean the address and not a slice of it.
                let pointer = (u64::try_from(self.context.machine().ptr_size.bytes()).ok()
                    == Some(extent))
                .then(|| image.address_at(address, extent))
                .flatten()
                .and_then(|named| {
                    Some((
                        T::symbol_at(named)?,
                        u32::try_from(extent).ok().and_then(T::pointer_type),
                    ))
                });
                let (value, dest_type) = match pointer {
                    Some((symbol, pointer_type)) => (
                        ConstValue::Symbol(symbol),
                        pointer_type.filter(|pointer_type| pointer_type != value_type),
                    ),
                    None => (
                        ConstValue::from_typed_bytes(bytes, value_type, order)?,
                        None,
                    ),
                };
                let read_type = dest_type.clone().unwrap_or_else(|| value_type.clone());
                Some((vec![(*dest, value)], read_type, dest_type))
            }
            SsaOp::VectorLoad {
                dest, vector_type, ..
            } => Some((
                vec![(
                    *dest,
                    ConstValue::from_typed_bytes(bytes, vector_type, order)?,
                )],
                vector_type.clone(),
                None,
            )),
            SsaOp::VectorBroadcastLoad {
                dest, vector_type, ..
            } => {
                let shape = T::vector_shape(vector_type)?;
                let lanes = usize::try_from(shape.lane_count).ok()?;
                let splat: Vec<u8> = bytes
                    .iter()
                    .copied()
                    .cycle()
                    .take(bytes.len().checked_mul(lanes)?)
                    .collect();
                Some((
                    vec![(
                        *dest,
                        ConstValue::from_typed_bytes(&splat, vector_type, order)?,
                    )],
                    vector_type.clone(),
                    None,
                ))
            }
            SsaOp::VectorSegmentLoad {
                dests,
                vector_type,
                segments,
                layout,
                mask: None,
                ..
            } => {
                let shape = T::vector_shape(vector_type)?;
                let lane = usize::try_from(shape.lane_bits.checked_div(8)?).ok()?;
                let (lanes, count) = (
                    usize::try_from(shape.lane_count).ok()?,
                    usize::try_from(*segments).ok()?,
                );
                let values = (0..count)
                    .zip(dests)
                    .map(|(segment, dest)| {
                        let gathered = (0..lanes)
                            .map(|index| {
                                let element = match layout {
                                    VectorSegmentLayout::Interleaved => {
                                        index.checked_mul(count)?.checked_add(segment)?
                                    }
                                    VectorSegmentLayout::Consecutive => {
                                        segment.checked_mul(lanes)?.checked_add(index)?
                                    }
                                };
                                let at = element.checked_mul(lane)?;
                                bytes.get(at..at.checked_add(lane)?)
                            })
                            .collect::<Option<Vec<&[u8]>>>()?
                            .concat();
                        Some((
                            *dest,
                            ConstValue::from_typed_bytes(&gathered, vector_type, order)?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?;
                (values.len() == dests.len()).then(|| (values, vector_type.clone(), None))
            }
            SsaOp::VectorMaskedLoad {
                dest,
                mask,
                passthrough,
                vector_type,
                mode,
                ..
            } => {
                let read = MaskedRead::of(
                    self.ssa,
                    *mask,
                    *passthrough,
                    *mode,
                    vector_type,
                    order,
                    self.context,
                )?;
                let lanes = read.compose(bytes, false)?;
                Some((
                    vec![(
                        *dest,
                        ConstValue::from_typed_bytes(&lanes, vector_type, order)?,
                    )],
                    vector_type.clone(),
                    None,
                ))
            }
            SsaOp::VectorPackLoad {
                dest,
                mask,
                passthrough,
                vector_type,
                element_bits,
                kind: VectorPackKind::Expand,
                mode,
                ..
            } => {
                let read = MaskedRead::of(
                    self.ssa,
                    *mask,
                    *passthrough,
                    *mode,
                    vector_type,
                    order,
                    self.context,
                )?;
                if read.lane_bytes.checked_mul(8) != usize::try_from(*element_bits).ok() {
                    return None;
                }
                let lanes = read.compose(bytes, true)?;
                Some((
                    vec![(
                        *dest,
                        ConstValue::from_typed_bytes(&lanes, vector_type, order)?,
                    )],
                    vector_type.clone(),
                    None,
                ))
            }
            _ => None,
        }
    }

    /// Plans the removal of every load nothing reads whose cell the image places,
    /// the frame holds, or a symbol the host proves bound begins.
    ///
    /// A load is kept when its result is dead only because it may fault, and a
    /// cell the image places cannot: the loader maps it. Neither can the first
    /// word of an import the loader must bind ([`Target::symbol_bound`]), whose
    /// storage it mapped before the function ran: a function that never returns
    /// loads `__stack_chk_guard` for a canary no check reads. Kept, a dead load held
    /// up the branch it sat under - `uidget`'s `if (reg_4 == *(0x250b0)) { }`,
    /// whose arm read a cell of `.bss` for a result the function does not return.
    /// An indexed cell is placed where the image places every byte the index's
    /// bounds at the load let it read ([`IndexBounds`]): the entry a dispatch
    /// reads from its table, at the index its guard bounds, once a switch on the
    /// index replaced the jump the entry computed. The value ranges are computed
    /// only for such a load.
    ///
    /// # Arguments
    ///
    /// * `rewritten` - Positions another rewrite or fold already takes.
    /// * `forwarded` - Values another rewrite hands a load's readers, which are
    ///   read once it applies: a load nothing reads yet stays when a later load of
    ///   the same cell is forwarded its result.
    fn dead_image_loads(
        &self,
        rewritten: &FxHashSet<(usize, usize)>,
        forwarded: &FxHashSet<SsaVarId>,
    ) -> Vec<Rewrite<T>> {
        let image = self.context.image();
        let anchors = self
            .ssa
            .entry_stack_pointer()
            .map(|pointer| FrameAnchors::of(self.ssa, pointer, self.context.machine().ptr_size));
        let mut read: FxHashSet<SsaVarId> = forwarded.clone();
        for block in self.ssa.blocks() {
            for phi in block.phi_nodes() {
                read.extend(phi.operands().iter().map(PhiOperand::value));
            }
            for instruction in block.instructions() {
                read.extend(instruction.op().uses());
            }
        }
        let mut dead = Vec::new();
        let mut bounds = IndexBounds::new(
            self.ssa,
            self.context.machine().ptr_size,
            self.context.budget(),
        );
        for op in self.mem_ssa.operations() {
            let MemoryOp::Load {
                location,
                dest,
                block,
                instr,
            } = op
            else {
                continue;
            };
            let MemoryLocation::Indirect(cell) = location else {
                continue;
            };
            if read.contains(dest) || rewritten.contains(&(*block, *instr)) {
                continue;
            }
            // A load of the cell's bytes as one value, scalar or vector: what
            // it reads is its result alone, so unread it reads nothing.
            let Some(
                SsaOp::LoadIndirect {
                    access:
                        AccessQualifiers {
                            address_space: None,
                            ..
                        },
                    value_type,
                    ..
                }
                | SsaOp::VectorLoad {
                    access:
                        AccessQualifiers {
                            address_space: None,
                            ..
                        },
                    vector_type: value_type,
                    ..
                },
            ) = self
                .ssa
                .block(*block)
                .and_then(|body| body.instructions().get(*instr))
                .map(SsaInstruction::op)
            else {
                continue;
            };
            let word_bits = i64::try_from(self.context.machine().ptr_size.bytes())
                .unwrap_or(8)
                .saturating_mul(8);
            if anchors.as_ref().is_some_and(|anchors| {
                anchors.cell(location).is_some() || anchors.in_entry_word(location, word_bits)
            }) {
                dead.push(Rewrite::DeadFrameLoad {
                    block: *block,
                    instr: *instr,
                });
                continue;
            }
            // The first word of a symbol the host proves bound: storage the
            // loader mapped, at an address no read of a word there can fault on.
            if cell.bound_first_word(self.ssa, word_bits) {
                dead.push(Rewrite::DeadBoundLoad {
                    block: *block,
                    instr: *instr,
                });
                continue;
            }
            let Some(image) = image else {
                continue;
            };
            // An indexed read is placed where every address its index may move
            // it to is: a jump table's entry read at the dispatch's guarded
            // index, which the switch that replaced the jump no longer reads.
            let index = cell
                .index
                .filter(|_| cell.base.is_absolute())
                .and_then(|index| bounds.at(index, *block));
            if let Some((address, len)) = cell.placed_in_image(image, index) {
                let pointer = (u64::try_from(self.context.machine().ptr_size.bytes()).ok()
                    == Some(len))
                .then(|| image.address_at(address, len))
                .flatten()
                .and_then(|_| T::pointer_type(u32::try_from(len).ok()?));
                dead.push(Rewrite::DeadImageLoad {
                    block: *block,
                    instr: *instr,
                    address,
                    len,
                    read_type: pointer.unwrap_or_else(|| value_type.clone()),
                });
            }
        }
        dead
    }

    /// Walks every block and returns the rewrites the alias model proves safe.
    ///
    /// Plans each block independently ([`Self::block`]), reading what each
    /// instruction does to memory from the operations indexed by position.
    /// Cross-block forwarding is not lost by this per-block split: a value
    /// reaching a block from a dominating store is recovered by
    /// [`Self::seed_from_dominator`].
    ///
    /// # Returns
    ///
    /// Every proven-safe [`Rewrite`], in block-then-instruction order.
    fn rewrites(&self) -> Vec<Rewrite<T>> {
        let order = self.context.machine().byte_order;

        let graph = SsaCfg::from_ssa(self.ssa);
        let stack = self
            .ssa
            .entry_stack_pointer()
            .map(|entry| FrameEscape::of(self.ssa, entry));
        let mut rewrites = Vec::new();
        for block_idx in 0..self.ssa.block_count() {
            self.block(
                graph.roots(),
                stack.as_ref().map(FrameEscape::derived),
                (block_idx, order),
                &mut rewrites,
            );
        }
        self.frame_exit_dead_stores(&graph, &mut rewrites);
        // A rejoined copy writes every part at its last store, so a part whose
        // own store is dead would be written again: such a copy stays in parts.
        let dead: FxHashSet<(usize, usize)> = rewrites
            .iter()
            .filter_map(|rewrite| match rewrite {
                Rewrite::DeadStore { block, instr } => Some((*block, *instr)),
                _ => None,
            })
            .collect();
        rewrites.retain(|rewrite| {
            !matches!(rewrite, Rewrite::Rejoin { block, stores, .. }
                if stores.iter().any(|instr| dead.contains(&(*block, *instr))))
        });
        rewrites
    }

    /// Plans the removal of every store into the function's own frame that nothing
    /// observes before the function leaves, and of every store into the caller's
    /// memory above it that the function overwrites before it leaves.
    ///
    /// Applies only when the host has declared the entry stack pointer
    /// ([`VariableRole::EntryStackPointer`](crate::ir::variable::VariableRole::EntryStackPointer));
    /// without that fact no location is known to be private to the function, nor
    /// placed in the caller's frame, and nothing is removed.
    ///
    /// One backward liveness solve over the cells answers the question for
    /// every store at once ([`FrameLiveness`]): a store is dead exactly when the
    /// cell it writes is not live immediately after it.
    ///
    /// # Arguments
    ///
    /// * `graph` - The exception-aware CFG of `ssa`.
    /// * `rewrites` - Planned rewrites; frame-exit dead stores are appended, and a
    ///   store an earlier rule already removes is not planned twice.
    fn frame_exit_dead_stores(&self, graph: &SsaCfg<'_, T>, rewrites: &mut Vec<Rewrite<T>>) {
        let ptr_size = self.context.machine().ptr_size;
        let Some(entry_stack_pointer) = self.ssa.entry_stack_pointer() else {
            return;
        };
        let anchors = FrameAnchors::of(self.ssa, entry_stack_pointer, ptr_size);
        let cells = FrameCells::collect(self.ssa, self.mem_ssa, &anchors);
        if cells.is_empty() {
            return;
        }
        let escape = FrameEscape::of(self.ssa, entry_stack_pointer);
        let frame = FramePrivacy {
            anchors: &anchors,
            derived: escape.derived(),
            escapes: escape.escapes(),
        };
        let already_dead: FxHashSet<(usize, usize)> = rewrites
            .iter()
            .filter_map(|rewrite| match rewrite {
                Rewrite::DeadStore { block, instr } => Some((*block, *instr)),
                Rewrite::Forward { .. }
                | Rewrite::Compose { .. }
                | Rewrite::Slice { .. }
                | Rewrite::DeadImageLoad { .. }
                | Rewrite::DeadFrameLoad { .. }
                | Rewrite::DeadBoundLoad { .. }
                | Rewrite::Rejoin { .. } => None,
            })
            .collect();
        let liveness = FrameLiveness::solve(self.ssa, graph, &self.by_position, &cells, frame);
        liveness.for_each_dead_store(|block, instr| {
            if !already_dead.contains(&(block, instr)) {
                rewrites.push(Rewrite::DeadStore { block, instr });
            }
        });
    }

    /// Plans the rewrites for one block.
    ///
    /// Walks the block instruction by instruction while maintaining two pieces of
    /// state: `available`, the value currently held by each location (from a store,
    /// or from an earlier load that could not itself be forwarded), and
    /// `pending_stores`, the unread must-aliasing stores a later store could kill.
    /// A load whose location is available becomes a [`Rewrite::Forward`]; a store
    /// that overwrites an unread must-aliasing store makes it a
    /// [`Rewrite::DeadStore`]. Every load, store, read-write, and barrier then
    /// invalidates whatever it may-aliases so no value is carried across an access
    /// whose effect is not covered. The `available` map is seeded from dominating
    /// stores by [`MemoryPlanner::seed_from_dominator`] before the walk, which is what lets a load
    /// be served across block boundaries.
    ///
    /// # Arguments
    ///
    /// * `roots` - The function's exceptional entries, for the handlers a throw
    ///   from this block reaches.
    /// * `block_idx` - Index of the block to plan.
    /// * `rewrites` - Output sink; proven rewrites for this block are appended.
    fn block(
        &self,
        roots: &FunctionRoots,
        stack: Option<&FxHashSet<SsaVarId>>,
        (block_idx, order): (usize, Option<Endianness>),
        rewrites: &mut Vec<Rewrite<T>>,
    ) {
        let Some(block) = self.ssa.block(block_idx) else {
            return;
        };
        let protected = !roots.exceptional_successors(block_idx).is_empty();
        // Value currently held by each location, and the store that put it there
        // if it has not been read since.
        let mut available: Vec<(MemoryLocation<T>, SsaVarId)> = Vec::new();
        // Cells holding a lane of a stored value, and the bit of the value the
        // cell starts at: the elements a segment store wrote.
        let mut parts: Vec<(MemoryLocation<T>, SsaVarId, u32)> = Vec::new();
        let mut pending_stores: Vec<(MemoryLocation<T>, usize)> = Vec::new();
        let mut rejoining = Rejoining::new();

        self.seed_from_dominator(block_idx, &mut available);

        for (instr_idx, instruction) in block.instructions().iter().enumerate() {
            // A throw from here reaches a handler with every pending store still
            // the cell's contents - including when this instruction is itself the
            // overwriting store, which writes nothing when it faults. The handler
            // may read them, so none of them is dead. So may the caller an
            // exception a callee raises unwinds to, whatever the function
            // catches; the pending stores into the function's own frame, which
            // the caller cannot read, are the frame-exit solve's
            // ([`MemoryPlanner::frame_exit_dead_stores`]).
            if (protected && instruction.may_throw()) || instruction.op().may_unwind() {
                pending_stores.clear();
                rejoining.clear();
            }
            let Some(op) = self.by_position.get(&(block_idx, instr_idx)) else {
                continue;
            };
            match op {
                MemoryOp::Load { location, dest, .. } => {
                    // A load reads whatever may-alias it, so any pending store it
                    // could observe is live.
                    pending_stores.retain(|(pending, _)| !pending.may_alias(location));
                    rejoining.load(self.ssa, &available, location, *dest);

                    if let Some(value) = lookup_available(&available, location)
                        && let Some((read_type, bits)) = narrowed_read(self.ssa, *dest, value)
                        && let Some(width) = self.ssa.bit_width(value)
                    {
                        // A store narrower than its value writes the value's low
                        // bits, and a load of that cell reads them back: the value
                        // narrowed to the width read, whatever the byte order. A
                        // composition of that one piece at the bottom.
                        rewrites.push(Rewrite::Compose {
                            dest: *dest,
                            read_type,
                            bits,
                            pieces: vec![Piece {
                                value,
                                shift: 0,
                                width,
                            }],
                            block: block_idx,
                            instr: instr_idx,
                        });
                    } else if let Some(value) = lookup_available(&available, location) {
                        // A value stored as the width the load reads is those
                        // bits whatever its own type says: `mov r2, #0; str r2`
                        // stores an integer the load reads back as `u32`. The
                        // load's type is carried where the two differ at one
                        // width, and the forward reinterprets.
                        let type_of =
                            |var: SsaVarId| self.ssa.variable(var).map(|v| v.var_type().clone());
                        let read_type = match (type_of(*dest), type_of(value)) {
                            (Some(read), Some(held))
                                if read != held
                                    && T::bit_width(&read).is_some()
                                    && T::bit_width(&read) == T::bit_width(&held) =>
                            {
                                Some(read)
                            }
                            _ => None,
                        };
                        rewrites.push(Rewrite::Forward {
                            redundant: *dest,
                            read_type,
                            available: value,
                            block: block_idx,
                            instr: instr_idx,
                        });
                    } else {
                        // Part of one wider stored value: a word of a vector a
                        // structure store wrote, the high half of a doubleword.
                        if let Some(Sliced {
                            value,
                            from,
                            bits,
                            at_start,
                        }) = order.and_then(|order| {
                            sliced_from(self.ssa, &available, &parts, location, order)
                        }) && let Some(read_type) = self
                            .ssa
                            .variable(*dest)
                            .map(|variable| variable.var_type().clone())
                        {
                            rewrites.push(Rewrite::Slice {
                                dest: *dest,
                                read_type,
                                value,
                                at_start,
                                from,
                                bits,
                                block: block_idx,
                                instr: instr_idx,
                            });
                        } else
                        // Several narrower stores that tile the bytes it reads:
                        // `-O0` x86 copies a `double` argument into a local in two
                        // 32-bit moves and reads the local back whole.
                        if let Some((bits, pieces)) = order
                            .and_then(|order| composed_from(self.ssa, &available, location, order))
                            && let Some(read_type) = self
                                .ssa
                                .variable(*dest)
                                .map(|variable| variable.var_type().clone())
                        {
                            rewrites.push(Rewrite::Compose {
                                dest: *dest,
                                read_type,
                                bits,
                                pieces,
                                block: block_idx,
                                instr: instr_idx,
                            });
                        }
                        // Its result is now the live value for this cell - that is
                        // redundant-load elimination for any later load of the
                        // same cell.
                        set_available(&mut available, location.clone(), *dest);
                    }
                }
                MemoryOp::Store {
                    location, value, ..
                } => {
                    // A store into the stack of the value just loaded from the cell
                    // writes nothing: the cell, and everything the walk knows of
                    // it, is as it was. A stack probe's `or [rsp], 0`, in line or
                    // in the loop that allocates a large frame a page at a time.
                    // The load stands between it and any store before, which it
                    // reads, so no other rule ends that store for this one; a
                    // store of the value a store put there is that store's
                    // overwrite instead, and ends it. The stack is an address
                    // derived from the entry stack pointer; memory the function
                    // was handed may be a device's, whose write-back may act, and
                    // stays.
                    if let MemoryLocation::Indirect(cell) = location
                        && cell
                            .base
                            .value()
                            .is_some_and(|base| stack.is_some_and(|stack| stack.contains(&base)))
                        && lookup_available(&available, location) == Some(*value)
                        && matches!(
                            self.ssa.get_definition(*value),
                            Some(SsaOp::LoadIndirect { .. })
                        )
                    {
                        rewrites.push(Rewrite::DeadStore {
                            block: block_idx,
                            instr: instr_idx,
                        });
                        continue;
                    }
                    // A store that overwrites an unread must-aliasing store kills
                    // it. Only an exact must-alias proves full coverage; a partial
                    // overlap leaves some of the earlier store observable.
                    if let Some(position) = pending_stores
                        .iter()
                        .position(|(pending, _)| pending.must_alias(location))
                    {
                        let (_, dead_instr) = pending_stores.swap_remove(position);
                        rewrites.push(Rewrite::DeadStore {
                            block: block_idx,
                            instr: dead_instr,
                        });
                    }
                    // Anything this store could touch has unknown contents after
                    // it.
                    available.retain(|(known, _)| !known.may_alias(location));
                    parts.retain(|(known, _, _)| !known.may_alias(location));
                    // A store this one only partially overlaps is not provably
                    // dead: part of it survives. Dropping the candidate is the
                    // conservative answer.
                    pending_stores.retain(|(pending, _)| !pending.may_alias(location));

                    set_available(&mut available, location.clone(), *value);
                    pending_stores.push((location.clone(), instr_idx));
                    // The last store of a value's parts, copied: the parts' stores
                    // are one store of the value now, which no later overwrite of
                    // one part's cell makes dead.
                    if let Some(rejoin) =
                        rejoining.store(instruction.op(), location, *value, (block_idx, instr_idx))
                    {
                        if let Rewrite::Rejoin { stores, .. } = &rejoin {
                            pending_stores.retain(|(_, pending)| !stores.contains(pending));
                        }
                        rewrites.push(rejoin);
                    }
                }
                // A read whose result is no value of the location: what it reads
                // observes every pending store that may alias it.
                MemoryOp::Read { location, .. } => {
                    pending_stores.retain(|(pending, _)| !pending.may_alias(location));
                    rejoining.read(location);
                }
                MemoryOp::ReadWrite { location, .. } | MemoryOp::Write { location, .. } => {
                    // Reads and writes an unmodelled amount of the location, or
                    // writes one; a write nothing places is kept from ending a
                    // pending store, as a read would be.
                    pending_stores.retain(|(pending, _)| !pending.may_alias(location));
                    rejoining.read(location);
                    available.retain(|(known, _)| !known.may_alias(location));
                    parts.retain(|(known, _, _)| !known.may_alias(location));
                    // A segment store writes each element of its extent with a
                    // lane of one of its values.
                    parts.extend(segment_parts(self.ssa, instruction.op(), location));
                }
                MemoryOp::Barrier { .. } => {
                    // A call, fence, or opaque native operation: nothing survives.
                    available.clear();
                    parts.clear();
                    pending_stores.clear();
                    rejoining.clear();
                }
                MemoryOp::Call { .. } => {
                    // A call its host described. What it reads observes every
                    // pending store that may alias it, so none of those is dead;
                    // what it may write has unknown contents after it. A stated
                    // write kills no pending store: it may not happen, and one it
                    // does not read stays a candidate for a later overwrite.
                    // Escaped memory is none of the frame's own cells while no
                    // address of the frame leaves ([`MemorySsa::may_alias`]).
                    for read in op.read_locations() {
                        pending_stores
                            .retain(|(pending, _)| !self.mem_ssa.may_alias(pending, read));
                    }
                    rejoining.clear();
                    for written in op.defined_locations() {
                        available.retain(|(known, _)| !self.mem_ssa.may_alias(known, written));
                        parts.retain(|(known, _, _)| !self.mem_ssa.may_alias(known, written));
                    }
                }
            }
        }
    }

    /// Seeds `available` with values reaching this block from a dominating store.
    ///
    /// Safe because a memory definition versions *every location it may alias*, not
    /// only the one it names (see `MemorySsa::rename_block`). So a location's
    /// block-entry version having a store as its definition really does mean that
    /// store's value is intact here: any intervening barrier, call, or overlapping
    /// store would have bumped this location's version and made its definition that
    /// operation instead, which the `MemoryOp::Store` and `must_alias` checks below
    /// then reject.
    ///
    /// Memory SSA renames on the dominator tree, so a location's block-entry
    /// version is the version its dominator left, or the block's own memory phi
    /// where two or more definitions merge. When that version is a store, the
    /// stored value dominates this block and can serve loads here; a phi names no
    /// single value, and is not a store.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - Index of the block being seeded.
    /// * `available` - Availability map to populate with dominating stored values.
    fn seed_from_dominator(
        &self,
        block_idx: usize,
        available: &mut Vec<(MemoryLocation<T>, SsaVarId)>,
    ) {
        for (location, site) in self.mem_ssa.entry_definitions(block_idx) {
            let MemoryDefSite::Store {
                block: store_block,
                instr: store_instr,
            } = site
            else {
                continue;
            };
            // A store in this very block is handled by the walk itself.
            if store_block == block_idx {
                continue;
            }
            let Some(MemoryOp::Store {
                location: stored_location,
                value,
                ..
            }) = self.by_position.get(&(store_block, store_instr)).copied()
            else {
                continue;
            };
            // The recorded version identifies the location, but only an exact
            // must-alias proves the store covers a whole read of it.
            if !stored_location.must_alias(location) {
                continue;
            }
            // Guard against a malformed graph handing back a value nothing
            // defines; the editor's dominance check is the real gate, this is
            // cheap insurance. A phi or the function's entry defines a value as
            // well as an instruction does: an incoming register spilled across a
            // call is one.
            if !self.ssa.is_defined(*value) {
                continue;
            }
            set_available(available, location.clone(), *value);
        }
    }
}

/// Whether `read` may name a cell of the function's own frame.
///
/// The provenance question `IndirectLocation::may_alias` cannot ask: it is a
/// pure function on two locations, so two distinct SSA roots always may-alias.
/// Here the root is available, and with it the fact that a pointer which reached
/// the function from outside was formed before this frame existed.
///
/// A managed location - an instance field, a static field, an array element -
/// names no frame cell, and answering `false` for it is not a relaxation:
/// [`FrameAnchors::cell`] requires the *stored* location to be indirect, and a
/// managed location is not raw memory an indirect access reaches. Every other
/// location may: an indirect one whose base could hold a frame address, and
/// [`MemoryLocation::Unknown`], which is what an access Memory SSA does not
/// place - an atomic, a vector load or gather, a block copy, a string
/// instruction - classifies to.
///
/// # Arguments
///
/// * `read` - The access being tested.
/// * `frame` - What is known about this function's frame.
fn reads_this_frame<T: Target>(read: &MemoryLocation<T>, frame: FramePrivacy<'_>) -> bool {
    match read {
        MemoryLocation::Indirect(cell) => frame.may_name_frame(cell.base),
        MemoryLocation::InstanceField(..)
        | MemoryLocation::StaticField(_)
        | MemoryLocation::ArrayElement(..) => false,
        MemoryLocation::Unknown => true,
        // Everything but the cells whose address the function kept to itself.
        MemoryLocation::Escaped => frame.escapes,
    }
}

/// Every distinct cell of the frame, or of the caller's memory above it, some
/// store of the function writes.
///
/// A cell's position in `cells` is its bit in a [`FrameLiveness`] set.
struct FrameCells {
    /// The cells, by anchor and then in ascending offset order.
    cells: Vec<CellKey>,
    /// One past the last bit of each cell, indexed as `cells`.
    ends: Vec<i64>,
    /// Position of each cell in `cells`.
    positions: FxHashMap<CellKey, usize>,
    /// Width of the widest cell, in bits: no cell of a read's anchor that
    /// starts further than this below the read's offset can reach it.
    widest_bits: i64,
    /// The positions of the cells every store to which the host declared a
    /// save of the caller's state ([`InstructionRole::CallerStateSave`]): the
    /// register saves, which no callee reads.
    saves: BTreeSet<usize>,
    /// The positions of the caller's cells ([`FrameAnchors::caller_cell`]):
    /// live wherever the function leaves, and read by every access that may
    /// name them, the ones through a pointer from outside included.
    callers: BitSet,
}

impl FrameCells {
    /// Collects the frame cells, and the caller's, written by `mem_ssa`'s
    /// stores.
    ///
    /// # Arguments
    ///
    /// * `mem_ssa` - Memory SSA providing the classified operations.
    /// * `anchors` - The values the frame is addressed from.
    fn collect<T: Target>(
        ssa: &SsaFunction<T>,
        mem_ssa: &MemorySsa<T>,
        anchors: &FrameAnchors,
    ) -> Self {
        // Each store's cell, whether the host declared the store a save, and
        // whether the cell is the caller's, whose stores save nothing.
        let stores: Vec<(CellKey, bool, bool)> = mem_ssa
            .operations()
            .iter()
            .filter_map(|op| match op {
                MemoryOp::Store {
                    location,
                    block,
                    instr,
                    ..
                } => anchors
                    .cell(location)
                    .map(|cell| {
                        let save = ssa
                            .block(*block)
                            .and_then(|body| body.instructions().get(*instr))
                            .is_some_and(|store| {
                                store.role() == Some(InstructionRole::CallerStateSave)
                            });
                        (cell, save, false)
                    })
                    .or_else(|| {
                        anchors
                            .caller_cell(location)
                            .map(|cell| (cell, false, true))
                    }),
                _ => None,
            })
            .collect();
        let mut cells: Vec<CellKey> = stores.iter().map(|(cell, _, _)| *cell).collect();
        cells.sort_unstable();
        cells.dedup();
        let ends = cells
            .iter()
            .map(|cell| cell.offset_bits.saturating_add(i64::from(cell.size_bits)))
            .collect();
        let positions: FxHashMap<CellKey, usize> = cells
            .iter()
            .enumerate()
            .map(|(position, cell)| (*cell, position))
            .collect();
        let widest_bits = cells
            .iter()
            .map(|cell| i64::from(cell.size_bits))
            .max()
            .unwrap_or(0);
        let mut holds_data: BTreeSet<usize> = BTreeSet::new();
        let mut callers = BitSet::new(cells.len());
        for (cell, save, caller) in &stores {
            if let Some(&position) = positions.get(cell) {
                if !save {
                    holds_data.insert(position);
                }
                if *caller {
                    callers.insert(position);
                }
            }
        }
        let saves = (0..cells.len())
            .filter(|position| !holds_data.contains(position))
            .collect();
        Self {
            cells,
            ends,
            positions,
            widest_bits,
            saves,
            callers,
        }
    }

    /// The position of the cell a store to `location` writes, of the frame
    /// or of the caller's memory.
    fn written<T: Target>(
        &self,
        location: &MemoryLocation<T>,
        anchors: &FrameAnchors,
    ) -> Option<usize> {
        anchors
            .cell(location)
            .or_else(|| anchors.caller_cell(location))
            .and_then(|key| self.positions.get(&key).copied())
    }

    /// What an access that may name memory outside the function's frame, but
    /// none of the frame itself, reads of the cells: the caller's.
    fn read_outside(&self) -> CellEffect {
        if self.callers.is_empty() {
            CellEffect::None
        } else {
            CellEffect::Reads(self.callers.iter().collect())
        }
    }

    /// The cells wholly inside `location`, when it names bytes at a fixed
    /// offset from an anchor: the cells it covers, measured as cells are
    /// ([`FrameAnchors::canonical`]). A cell of an anchor placed only within
    /// bounds of another is never inside it.
    fn inside<T: Target>(
        &self,
        location: &MemoryLocation<T>,
        frame: FramePrivacy<'_>,
    ) -> Vec<usize> {
        let MemoryLocation::Indirect(access) = location else {
            return Vec::new();
        };
        if access.address_space.is_some() || access.crosses_wrap_boundary() {
            return Vec::new();
        }
        let Some((anchor, start, end)) = frame.anchors.extent(location) else {
            return Vec::new();
        };
        self.cells
            .iter()
            .zip(&self.ends)
            .enumerate()
            .filter(|(_, (cell, cell_end))| {
                cell.anchor == anchor && cell.offset_bits >= start && **cell_end <= end
            })
            .map(|(position, _)| position)
            .collect()
    }

    /// What a call nobody described reads of the frame, and what escaped
    /// memory holds of it: every cell but the register saves.
    fn read_by_call(&self) -> CellEffect {
        if self.saves.is_empty() {
            return CellEffect::ReadsAll;
        }
        CellEffect::Reads(
            (0..self.cells.len())
                .filter(|position| !self.saves.contains(position))
                .collect(),
        )
    }

    /// Whether no store writes the frame at all.
    fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// The number of cells, and so the width of a liveness set.
    fn len(&self) -> usize {
        self.cells.len()
    }

    /// What reading `read` does to the frame cells.
    ///
    /// Exactly the cells `read` may alias, when it may name the frame at all
    /// ([`reads_this_frame`]); when it cannot, it may still name the caller's
    /// memory, through a pointer from outside or as escaped memory, and so
    /// reads every cell of the caller's ([`Self::read_outside`]). A read
    /// through the entry stack pointer with a known extent is answered from
    /// the sorted cells, by the same extent comparison
    /// [`IndirectLocation::may_alias`](crate::analysis::memory::IndirectLocation::may_alias)
    /// makes, among the cells of the read's anchor, and each cell of another
    /// anchor the bounds on the two anchors' places let it reach. A read at no
    /// fixed offset from an anchor (another base, or an index term) reads an
    /// object, and so reads what a call reads ([`Self::read_by_call`]): every
    /// cell but the register saves. Every other read that may name the frame
    /// may alias every cell: an unknown width or an access that wraps the
    /// address space defeats the offset comparison.
    ///
    /// # Arguments
    ///
    /// * `read` - The location read.
    /// * `frame` - What is known about this function's frame.
    fn read_by<T: Target>(&self, read: &MemoryLocation<T>, frame: FramePrivacy<'_>) -> CellEffect {
        if !reads_this_frame(read, frame) {
            return match read {
                MemoryLocation::Indirect(access) if access.address_space.is_some() => {
                    CellEffect::None
                }
                MemoryLocation::Indirect(_) | MemoryLocation::Escaped | MemoryLocation::Unknown => {
                    self.read_outside()
                }
                MemoryLocation::InstanceField(..)
                | MemoryLocation::StaticField(_)
                | MemoryLocation::ArrayElement(..) => CellEffect::None,
            };
        }
        let access = match read {
            MemoryLocation::Indirect(access) => access,
            // Escaped memory is the objects the function let addresses of go,
            // and a register save is none: its restore reads it at a fixed
            // offset from an anchor. So it reads what a call reads.
            MemoryLocation::Escaped => return self.read_by_call(),
            _ => return CellEffect::ReadsAll,
        };
        // Every cell is in the default space, and distinct spaces never alias.
        if access.address_space.is_some() {
            return CellEffect::None;
        }
        // An access not placed at a fixed offset from an anchor reads an
        // object: through an address that escaped, when the function did not
        // form its base from its own stack pointer, or as an element of one or
        // through a pointer walking one, when it did. The register saves are
        // no object - their restores read them at fixed offsets from an anchor
        // - so it reads what a call reads.
        if access
            .base
            .value()
            .filter(|base| frame.anchors.contains(*base) && access.index.is_none())
            .is_none()
        {
            return self.read_by_call();
        }
        if access.crosses_wrap_boundary() {
            return CellEffect::ReadsAll;
        }
        let Some((anchor, start, end)) = frame.anchors.extent(read) else {
            return CellEffect::ReadsAll;
        };
        let floor = start.saturating_sub(self.widest_bits);
        let first = self
            .cells
            .partition_point(|cell| (cell.anchor, cell.offset_bits) <= (anchor, floor));
        let mut read_cells: Vec<usize> = self
            .cells
            .iter()
            .zip(&self.ends)
            .enumerate()
            .skip(first)
            .take_while(|(_, (cell, _))| cell.anchor == anchor && cell.offset_bits < end)
            .filter(|(_, (_, cell_end))| start < **cell_end)
            .map(|(position, _)| position)
            .collect();
        // A cell measured from another anchor lies at a distance from this
        // one no constant states, only bounds.
        read_cells.extend(
            self.cells
                .iter()
                .zip(&self.ends)
                .enumerate()
                .filter(|(_, (cell, cell_end))| {
                    cell.anchor != anchor
                        && frame.anchors.may_overlap(
                            (anchor, start, end),
                            (cell.anchor, cell.offset_bits, **cell_end),
                        )
                })
                .map(|(position, _)| position),
        );
        read_cells.sort_unstable();
        if read_cells.is_empty() {
            CellEffect::None
        } else {
            CellEffect::Reads(read_cells)
        }
    }
}

/// What one instruction does to the frame's cells.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CellEffect {
    /// Neither reads nor overwrites any cell.
    None,
    /// May read the listed cells.
    Reads(Vec<usize>),
    /// May read every cell: a barrier, or an access whose location is not
    /// separable from the frame.
    ReadsAll,
    /// Overwrites exactly this cell, so no earlier value of it is read past
    /// here.
    Overwrites(usize),
}

impl CellEffect {
    /// The same effect with the cells at `positions` read no longer, of
    /// `cells` in all.
    fn without(self, positions: &[usize], cells: usize) -> Self {
        let kept: Vec<usize> = match self {
            Self::ReadsAll if !positions.is_empty() => (0..cells).collect(),
            Self::Reads(read) => read,
            effect => return effect,
        };
        let kept: Vec<usize> = kept
            .into_iter()
            .filter(|position| !positions.contains(position))
            .collect();
        if kept.is_empty() {
            Self::None
        } else {
            Self::Reads(kept)
        }
    }

    /// The effect of doing both: reading every cell either reads. An
    /// overwrite does not survive the union, since the other half may read
    /// the cell first.
    fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::None, effect) | (effect, Self::None) => match effect {
                Self::Overwrites(_) => Self::None,
                effect => effect,
            },
            (Self::ReadsAll, _) | (_, Self::ReadsAll) => Self::ReadsAll,
            (Self::Reads(mut left), Self::Reads(right)) => {
                left.extend(right);
                left.sort_unstable();
                left.dedup();
                Self::Reads(left)
            }
            (Self::Reads(cells), Self::Overwrites(_))
            | (Self::Overwrites(_), Self::Reads(cells)) => Self::Reads(cells),
            (Self::Overwrites(_), Self::Overwrites(_)) => Self::None,
        }
    }
}

/// One instruction the liveness solve cannot skip.
struct Step {
    /// Position of the instruction in its block.
    instr: usize,
    /// What it does to the frame's cells.
    effect: CellEffect,
    /// The cells it replaces after reading what `effect` reads, without
    /// being a store of them.
    replaces: Vec<usize>,
    /// Whether it may throw, to the handlers of a block that has them.
    throws: bool,
    /// Whether it may raise an exception that unwinds out of the function
    /// ([`SsaOp::may_unwind`]), to a caller that may read its memory.
    unwinds: bool,
}

/// How control leaves a block when it runs to its terminator.
enum BlockExit {
    /// Through its terminator's named successors, and nothing else.
    Named,
    /// Out of the function, by a return or a throw, where the caller's
    /// cells are live: past the named successors and handlers it also has.
    Leaves,
    /// Somewhere this analysis cannot see: an unresolved indirect branch, a
    /// handler end, or no terminator at all. Every cell is live there.
    Unknown,
}

/// Backward liveness of the frame's cells over the exception-aware CFG.
///
/// A cell is live at a point when some path from there reads it before the
/// function leaves and before a must-aliasing store overwrites it. Reads are
/// what may alias the cell and may name the frame ([`FrameCells::read_by`]),
/// every barrier (a callee reads its stack arguments from this region), and
/// every exit whose continuation is unknown. A cell of the caller's memory is
/// read by the function leaving as well: every return, every throw, and every
/// call an exception may unwind out of ([`SsaOp::may_unwind`]), but for the
/// bytes that call itself replaces, which hold nothing of this function's
/// however it ends.
///
/// # Exceptional flow
///
/// A throw from *any* block of a protected region continues at the region's
/// handlers ([`FunctionRoots::exceptional_successors`](crate::analysis::exceptions::FunctionRoots::exceptional_successors)),
/// so for such a block:
///
/// - every instruction that may throw makes the handlers' live cells live in
///   front of it. An overwrite after a faulting load does not end the path: the
///   handler runs with the earlier value still in the cell. A faulting store
///   writes nothing, so its own overwrite does not cover the handler either.
/// - the end of the block flows to the handlers as well as to its named
///   successors. That covers a `Throw` or `Rethrow`, which leave the block only
///   for its handlers, and a `Leave` or return that runs a finally handler on
///   the way out.
struct FrameLiveness<'a> {
    /// The cells, whose positions are the bits of every set here.
    cells: &'a FrameCells,
    /// Per block, the instructions that read, overwrite, or may throw to a
    /// handler, in block order.
    steps: Vec<Vec<Step>>,
    /// Per block, the terminator successors.
    successors: Vec<Vec<usize>>,
    /// Per block, the handler and filter entries a throw from it reaches.
    handlers: Vec<Vec<usize>>,
    /// Per block, how it ends.
    exits: Vec<BlockExit>,
    /// Per block, the cells live on entry.
    live_in: Vec<BitSet>,
}

impl<'a> FrameLiveness<'a> {
    /// Solves frame-cell liveness for `ssa` to a fixpoint.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function being analyzed.
    /// * `graph` - The exception-aware CFG of `ssa`.
    /// * `by_position` - Classified memory operations indexed by `(block, instr)`.
    /// * `cells` - The frame cells whose liveness is tracked.
    /// * `frame` - What is known about this function's frame.
    fn solve<T: Target>(
        ssa: &SsaFunction<T>,
        graph: &SsaCfg<'_, T>,
        by_position: &FxHashMap<(usize, usize), &MemoryOp<T>>,
        cells: &'a FrameCells,
        frame: FramePrivacy<'_>,
    ) -> Self {
        let block_count = ssa.block_count();
        let width = cells.len();
        let roots = graph.roots();
        // A load nothing reads is removed in the same round, when it reads the
        // frame (`plan_dead_image_loads`), so it reads nothing here: counted,
        // a restore no instruction consumes kept its save alive for a round.
        let mut read_values: FxHashSet<SsaVarId> = FxHashSet::default();
        for block in ssa.blocks() {
            for phi in block.phi_nodes() {
                read_values.extend(phi.operands().iter().map(PhiOperand::value));
            }
            for instruction in block.instructions() {
                read_values.extend(instruction.op().uses());
            }
        }
        let mut steps = Vec::with_capacity(block_count);
        let mut successors = Vec::with_capacity(block_count);
        let mut handlers = Vec::with_capacity(block_count);
        let mut exits = Vec::with_capacity(block_count);
        let mut live_in = Vec::with_capacity(block_count);
        for block_idx in 0..block_count {
            let block_handlers = roots.exceptional_successors(block_idx).to_vec();
            let Some(block) = ssa.block(block_idx) else {
                // A block the function does not have may continue anywhere.
                steps.push(Vec::new());
                successors.push(Vec::new());
                handlers.push(Vec::new());
                exits.push(BlockExit::Unknown);
                live_in.push(BitSet::full(width));
                continue;
            };
            let block_steps: Vec<Step> = block
                .instructions()
                .iter()
                .enumerate()
                .filter_map(|(instr, instruction)| {
                    // Cells the instruction replaces after its reads without
                    // being a store of them: the bytes a described call's callee
                    // owns and writes.
                    let mut replaces: Vec<usize> = Vec::new();
                    let effect = match by_position.get(&(block_idx, instr)) {
                        Some(MemoryOp::Load { location, dest, .. })
                            if !read_values.contains(dest)
                                && frame.anchors.cell(location).is_some() =>
                        {
                            CellEffect::None
                        }
                        // An access Memory SSA could not place - a block copy, a
                        // string instruction - still reads through its own
                        // address operands, and when the function formed none of
                        // them from its stack pointer it reaches the frame only
                        // as a pointer from outside does.
                        Some(
                            MemoryOp::Load {
                                location: MemoryLocation::Unknown,
                                ..
                            }
                            | MemoryOp::ReadWrite {
                                location: MemoryLocation::Unknown,
                                ..
                            },
                        ) if frame.reads_only_foreign_addresses(instruction.op()) => {
                            if frame.escapes {
                                cells.read_by_call()
                            } else {
                                cells.read_outside()
                            }
                        }
                        Some(
                            MemoryOp::Load { location, .. }
                            | MemoryOp::Read { location, .. }
                            | MemoryOp::ReadWrite { location, .. },
                        ) => cells.read_by(location, frame),
                        // **A call stays a barrier, and the reason is not
                        // escape.** A callee reads its stack arguments from the
                        // caller's frame, by convention and without ever being
                        // handed an address - and a pushed argument sits
                        // *below* the entry stack pointer, which is the same
                        // region `FrameAnchors::cell` claims. Bounding the argument
                        // area apart from the saves needs the stack depth at
                        // the call, which this pass does not have; a host that
                        // has it states the call's memory instead
                        // (`CallLocation::Frame`).
                        //
                        // So "nothing escaped" does not make the frame private
                        // to a call, only to a pointer the function was handed.
                        // What it does make private is the register saves: a
                        // callee reads arguments, and a save above the locals is
                        // none (`InstructionRole::CallerStateSave`).
                        //
                        // A fence is no call: it orders this thread's accesses
                        // and reads no memory itself, so a cell of the frame is
                        // observed past it only by an observer holding an
                        // address of the frame, which nothing has while no
                        // address of it escapes.
                        Some(MemoryOp::Barrier { effects, .. })
                            if effects.kind == SsaEffectKind::Fence =>
                        {
                            if frame.escapes {
                                cells.read_by_call()
                            } else {
                                cells.read_outside()
                            }
                        }
                        Some(MemoryOp::Barrier { .. }) => cells.read_by_call(),
                        // A write that reads nothing observes no store, and at
                        // a location nothing places it overwrites no cell for
                        // certain: a move into a special register (`msr
                        // basepri`) leaves every save before it unread.
                        Some(MemoryOp::Write { .. }) => CellEffect::None,
                        // A call its host described reads what it says it
                        // reads - which, for a callee with arguments on the
                        // stack, the host must state as the argument area or
                        // carry as operands. A write it states of other memory
                        // overwrites nothing for certain: it may not happen.
                        //
                        // The bytes the callee owns are read and written as it
                        // says, whatever its other accesses would reach of
                        // them, and the ones it writes hold nothing of this
                        // function's after it.
                        Some(MemoryOp::Call {
                            accesses, owned, ..
                        }) => {
                            let reads = |accesses: &[(MemoryLocation<T>, AccessMode)]| {
                                accesses
                                    .iter()
                                    .filter(|(_, access)| access.reads())
                                    .map(|(location, _)| cells.read_by(location, frame))
                                    .fold(CellEffect::None, CellEffect::union)
                            };
                            let inside: Vec<usize> = owned
                                .iter()
                                .flat_map(|(location, _)| cells.inside(location, frame))
                                .collect();
                            replaces = owned
                                .iter()
                                .filter(|(_, access)| *access == AccessMode::Write)
                                .flat_map(|(location, _)| cells.inside(location, frame))
                                .collect();
                            // The frame bytes a call names are the stack it
                            // reads as its arguments, and a register save is
                            // none: a call reading every byte above the ones it
                            // passes, as one nothing bounds does, keeps the
                            // saves no more than a barrier does.
                            let saves: Vec<usize> = cells.saves.iter().copied().collect();
                            reads(accesses)
                                .without(&inside, cells.len())
                                .union(reads(owned).without(&saves, cells.len()))
                        }
                        Some(MemoryOp::Store { location, .. }) => cells
                            .written(location, frame.anchors)
                            .map_or(CellEffect::None, CellEffect::Overwrites),
                        None => CellEffect::None,
                    };
                    let throws = instruction.may_throw();
                    let unwinds = instruction.op().may_unwind();
                    (effect != CellEffect::None || !replaces.is_empty() || throws).then_some(Step {
                        instr,
                        effect,
                        replaces,
                        throws,
                        unwinds,
                    })
                })
                .collect();
            let exit = match block.control_terminator() {
                Some(SsaOp::Return { .. } | SsaOp::Throw { .. } | SsaOp::Rethrow) => {
                    BlockExit::Leaves
                }
                Some(
                    SsaOp::Unreachable
                    | SsaOp::Jump { .. }
                    | SsaOp::Leave { .. }
                    | SsaOp::Branch { .. }
                    | SsaOp::BranchCmp { .. }
                    | SsaOp::BranchFlags { .. }
                    | SsaOp::Switch { .. },
                ) => BlockExit::Named,
                _ => BlockExit::Unknown,
            };
            steps.push(block_steps);
            // Terminator successors only: the handlers this block can throw to
            // are tracked separately, just below.
            successors.push(graph.terminator_successors(block_idx).to_vec());
            handlers.push(block_handlers);
            exits.push(exit);
            live_in.push(BitSet::new(width));
        }

        let mut liveness = Self {
            cells,
            steps,
            successors,
            handlers,
            exits,
            live_in,
        };
        liveness.iterate(ssa);
        liveness
    }

    /// Iterates block transfer functions until no block's entry set grows.
    ///
    /// The sets only grow from empty and every transfer is monotone, so a
    /// union into the entry set is the whole update, and the solve ends after
    /// at most one growth per cell per block.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function being analyzed.
    fn iterate<T: Target>(&mut self, ssa: &SsaFunction<T>) {
        let block_count = self.live_in.len();
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); block_count];
        for (block, (named, handlers)) in self.successors.iter().zip(&self.handlers).enumerate() {
            for successor in named.iter().chain(handlers) {
                if let Some(slot) = predecessors.get_mut(*successor) {
                    slot.push(block);
                }
            }
        }
        let mut queued = vec![true; block_count];
        // Popped from the back, so the last block is solved first: the usual
        // order for a backward problem over blocks laid out in program order.
        let mut worklist: Vec<usize> = (0..block_count)
            .filter(|b| ssa.block(*b).is_some())
            .collect();
        while let Some(block) = worklist.pop() {
            if let Some(slot) = queued.get_mut(block) {
                *slot = false;
            }
            let entry = self.block_entry(block);
            let grew = self
                .live_in
                .get_mut(block)
                .is_some_and(|live| live.union_with(&entry));
            if !grew {
                continue;
            }
            for predecessor in predecessors.get(block).map_or(&[][..], Vec::as_slice) {
                if let Some(slot) = queued.get_mut(*predecessor)
                    && !*slot
                {
                    *slot = true;
                    worklist.push(*predecessor);
                }
            }
        }
    }

    /// The union of the entry sets of `blocks`; every cell for a block the
    /// function does not have.
    ///
    /// # Arguments
    ///
    /// * `blocks` - The blocks whose entry sets to join.
    fn join(&self, blocks: &[usize]) -> BitSet {
        let mut joined = BitSet::new(self.cells.len());
        for block in blocks {
            match self.live_in.get(*block) {
                Some(live) => {
                    joined.union_with(live);
                }
                None => joined.fill(),
            }
        }
        joined
    }

    /// The cells live after `block`'s last instruction, and the cells live on
    /// entry to its handlers.
    ///
    /// # Arguments
    ///
    /// * `block` - The block whose exit to evaluate.
    fn block_exit(&self, block: usize) -> (BitSet, BitSet) {
        let handlers = self.join(self.handlers.get(block).map_or(&[][..], Vec::as_slice));
        let mut live = match self.exits.get(block) {
            Some(BlockExit::Named) => {
                self.join(self.successors.get(block).map_or(&[][..], Vec::as_slice))
            }
            Some(BlockExit::Leaves) => {
                let mut live = self.join(self.successors.get(block).map_or(&[][..], Vec::as_slice));
                live.union_with(&self.cells.callers);
                live
            }
            Some(BlockExit::Unknown) | None => BitSet::full(self.cells.len()),
        };
        live.union_with(&handlers);
        (live, handlers)
    }

    /// Runs `block` backward from its exit and returns the cells live on entry.
    ///
    /// Calls `on_store` with each tracked store's position and whether its
    /// cell is live immediately after it.
    ///
    /// # Arguments
    ///
    /// * `block` - The block to run.
    /// * `on_store` - Visitor for each tracked store in the block.
    fn run_block(&self, block: usize, mut on_store: impl FnMut(usize, bool)) -> BitSet {
        let (mut live, handlers) = self.block_exit(block);
        for step in self
            .steps
            .get(block)
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .rev()
        {
            for cell in &step.replaces {
                if *cell < live.len() {
                    live.remove(*cell);
                }
            }
            match &step.effect {
                CellEffect::None => {}
                CellEffect::Reads(read) => {
                    for cell in read {
                        live.insert_checked(*cell);
                    }
                }
                CellEffect::ReadsAll => live.fill(),
                CellEffect::Overwrites(cell) => {
                    on_store(step.instr, live.contains_checked(*cell));
                    if *cell < live.len() {
                        live.remove(*cell);
                    }
                }
            }
            // A throw here leaves before this instruction's write lands, with
            // every earlier value still in place for the handler to read.
            if step.throws {
                live.union_with(&handlers);
            }
            // And for the caller an exception it raises unwinds to: all of the
            // caller's cells but the ones this instruction replaces, which the
            // callee that threw owned.
            if step.unwinds {
                let mut outlive = self.cells.callers.clone();
                for cell in &step.replaces {
                    if *cell < outlive.len() {
                        outlive.remove(*cell);
                    }
                }
                live.union_with(&outlive);
            }
        }
        live
    }

    /// The cells live on entry to `block`, from its successors' current sets.
    ///
    /// # Arguments
    ///
    /// * `block` - The block to evaluate.
    fn block_entry(&self, block: usize) -> BitSet {
        self.run_block(block, |_, _| {})
    }

    /// Calls `dead` with the position of every tracked store whose cell is
    /// not live after it.
    ///
    /// # Arguments
    ///
    /// * `dead` - Visitor receiving `(block, instr)` for each dead store.
    fn for_each_dead_store(&self, mut dead: impl FnMut(usize, usize)) {
        for block in 0..self.live_in.len() {
            self.run_block(block, |instr, live| {
                if !live {
                    dead(block, instr);
                }
            });
        }
    }
}

/// What is known about who can name the function's own frame.
#[derive(Debug, Clone, Copy)]
struct FramePrivacy<'a> {
    /// The values the frame is addressed from.
    anchors: &'a FrameAnchors,
    /// Every value derived from the entry stack pointer.
    derived: &'a FxHashSet<SsaVarId>,
    /// Whether any of them may leave the function.
    escapes: bool,
}

impl FramePrivacy<'_> {
    /// Whether an access through `base` may name a cell of this frame.
    ///
    /// `true` whenever the address could have escaped, and whenever the base is
    /// itself frame-derived - the function naming its own frame is the case this
    /// must not rule out, and it reaches the frame by whatever route, not only as
    /// the entry stack pointer itself. `PtrToInt` and back is one such route, and
    /// testing equality with the entry pointer alone would dismiss it, in the
    /// unsound direction. Otherwise the base is a pointer that reached the
    /// function from outside, and the frame did not exist when it was formed.
    ///
    /// An absolute address is a number fixed before the frame existed too: it
    /// names the frame only as far as anything can once the frame's address
    /// has escaped.
    ///
    /// # Arguments
    ///
    /// * `base` - What the access being tested is measured from.
    fn may_name_frame(self, base: AddressBase) -> bool {
        self.escapes
            || base
                .value()
                .is_some_and(|value| self.derived.contains(&value))
    }

    /// Whether `op` reads memory only through addresses it names as operands,
    /// none of them one the function formed from its own stack pointer.
    ///
    /// Answered for the operations whose read addresses are operands: a block
    /// or object copy reads its source, a string instruction reads through its
    /// inputs (its counts and accumulators among them, which are no frame
    /// address either), and a single-address access through its address. Any
    /// other operation, and one that reads through no address at all, is
    /// `false`.
    ///
    /// # Arguments
    ///
    /// * `op` - The reading operation.
    fn reads_only_foreign_addresses<T: Target>(self, op: &SsaOp<T>) -> bool {
        let addresses: Vec<SsaVarId> = match op {
            SsaOp::CopyBlk { src_addr, .. }
            | SsaOp::MoveBlk { src_addr, .. }
            | SsaOp::CopyObj { src_addr, .. } => vec![*src_addr],
            SsaOp::CompareBlk(data) => vec![data.left, data.right],
            SsaOp::ScanBlk { address, .. } | SsaOp::LoadBlk { address, .. } => vec![*address],
            _ => op
                .memory_effect()
                .filter(|effect| effect.reads)
                .map(|effect| effect.addr)
                .into_iter()
                .collect(),
        };
        !addresses.is_empty()
            && addresses
                .iter()
                .all(|address| !self.derived.contains(address))
    }
}

/// Returns the value available for an exact read of `location`.
///
/// Requires [`MemoryLocation::must_alias`], which rejects unknown-width
/// accesses and partial overlaps - a value is only usable when the recorded
/// write covers exactly what is being read.
///
/// # Arguments
///
/// * `available` - Current availability map of `(location, value)` entries.
/// * `location` - The location being read.
///
/// # Returns
///
/// The available value if one must-aliases `location`, otherwise `None`.
fn lookup_available<T: Target>(
    available: &[(MemoryLocation<T>, SsaVarId)],
    location: &MemoryLocation<T>,
) -> Option<SsaVarId> {
    available
        .iter()
        .find(|(known, _)| known.must_alias(location))
        .map(|(_, value)| *value)
}

/// The type and width of `dest`, a load, when the value `held` that serves it
/// is wider: a store narrower than its value put only the value's low bits in
/// the cell the load reads.
fn narrowed_read<T: Target>(
    ssa: &SsaFunction<T>,
    dest: SsaVarId,
    held: SsaVarId,
) -> Option<(T::Type, u32)> {
    let read = ssa.variable(dest)?.var_type().clone();
    let read_bits = T::bit_width(&read)?;
    let held_bits = ssa.bit_width(held)?;
    (held_bits > read_bits && read_bits <= 64).then_some((read, read_bits))
}

/// Replaces the load at `at` by the composition of `pieces`: each stored value
/// read as an integer of the width its store wrote, widened to `bits`, shifted to the bit
/// its lowest bit lands on, and the pieces or-ed together; the load's result is
/// that integer read as `read_type`.
///
/// Every type is settled before anything is inserted, so a composition either
/// happens whole or not at all.
///
/// # Returns
///
/// Whether the load was replaced.
///
/// # Errors
///
/// Returns an error when the editor refuses an insertion or the replacement.
fn compose_load<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    dest: SsaVarId,
    read_type: &T::Type,
    bits: u32,
    pieces: &[Piece],
    (block, instr): (usize, usize),
) -> Result<bool> {
    let Some(wide) = T::integer_type(bits, false) else {
        return Ok(false);
    };
    let Some(amount_type) = T::integer_type(32, true) else {
        return Ok(false);
    };
    let mut typed = Vec::with_capacity(pieces.len());
    for &Piece {
        value,
        shift,
        width,
    } in pieces
    {
        let Some(held) = editor
            .function()
            .variable(value)
            .map(|variable| variable.var_type().clone())
        else {
            return Ok(false);
        };
        let Some(narrow) = T::integer_type(width, false) else {
            return Ok(false);
        };
        // A value of no stated type is the integer its store wrote.
        let held = if T::is_unknown(&held) {
            narrow.clone()
        } else {
            held
        };
        typed.push((value, shift, held, narrow));
    }
    let mut at = instr;
    let mut emit = |editor: &mut SsaEditor<'_, T>,
                    ty: T::Type,
                    op: &dyn Fn(SsaVarId) -> SsaOp<T>|
     -> Result<SsaVarId> {
        let value = editor.create_fresh_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(block, at),
            ty,
        );
        editor.insert_instruction(block, at, SsaInstruction::synthetic(op(value)))?;
        at = at.saturating_add(1);
        Ok(value)
    };
    let mut combined: Option<SsaVarId> = None;
    for (value, shift, held, narrow) in typed {
        let as_integer = if held == narrow {
            value
        } else {
            emit(editor, narrow.clone(), &|dest| {
                SsaOp::reinterpretation(dest, value, narrow.clone(), &held)
            })?
        };
        let widened = emit(editor, wide.clone(), &|dest| SsaOp::IntConv {
            dest,
            operand: as_integer,
            target: wide.clone(),
            overflow_check: false,
            unsigned: true,
        })?;
        let placed = if shift == 0 {
            widened
        } else {
            let amount = emit(editor, amount_type.clone(), &|dest| SsaOp::Const {
                dest,
                value: ConstValue::I32(i32::try_from(shift).unwrap_or(i32::MAX)),
            })?;
            emit(editor, wide.clone(), &|dest| SsaOp::Shl {
                dest,
                value: widened,
                amount,
                flags: None,
            })?
        };
        combined = Some(match combined {
            None => placed,
            Some(acc) => emit(editor, wide.clone(), &|dest| SsaOp::Or {
                dest,
                left: acc,
                right: placed,
                flags: None,
            })?,
        });
    }
    let Some(combined) = combined else {
        return Ok(false);
    };
    editor.replace_instruction_op(
        block,
        at,
        SsaOp::reinterpretation(dest, combined, read_type.clone(), &wide),
    )?;
    Ok(true)
}

/// The stored values that tile `location` exactly, each with the bit its
/// lowest bit lands on, and the location's width: two or more available cells
/// through the same base, contiguous, covering the load's bytes and no more,
/// each holding a value of its own width or of none, and the width each
/// store wrote.
///
/// `order` says which end of the load a lower address is: its low bits on a
/// little-endian target, its high bits on a big-endian one.
fn composed_from<T: Target>(
    ssa: &SsaFunction<T>,
    available: &[(MemoryLocation<T>, SsaVarId)],
    location: &MemoryLocation<T>,
    order: Endianness,
) -> Option<(u32, Vec<Piece>)> {
    let MemoryLocation::Indirect(load) = location else {
        return None;
    };
    let bits = load.size_bits.filter(|bits| *bits <= 64)?;
    if load.index.is_some() || load.address_space.is_some() {
        return None;
    }
    let start = load.offset_bits;
    let end = start.checked_add(i64::from(bits))?;
    let mut pieces: Vec<(i64, i64, SsaVarId)> = available
        .iter()
        .filter_map(|(known, value)| {
            let MemoryLocation::Indirect(cell) = known else {
                return None;
            };
            let size = i64::from(cell.size_bits?);
            let cell_end = cell.offset_bits.checked_add(size)?;
            (cell.base == load.base
                && cell.index.is_none()
                && cell.address_space.is_none()
                && cell.offset_bits >= start
                && cell_end <= end)
                .then_some((cell.offset_bits, cell_end, *value))
        })
        .collect();
    pieces.sort_unstable_by_key(|(offset, _, _)| *offset);
    if pieces.len() < 2 {
        return None;
    }
    let mut at = start;
    let mut placed = Vec::with_capacity(pieces.len());
    for (offset, piece_end, value) in pieces {
        let width = u32::try_from(piece_end.checked_sub(offset)?).ok()?;
        // A value of no stated type is the bits its store writes: the store
        // states the width, and the value has no other to disagree with it.
        let held = ssa.variable(value)?.var_type().clone();
        if offset != at || T::bit_width(&held).is_some_and(|held| held != width) {
            return None;
        }
        let from_low = u32::try_from(offset.checked_sub(start)?).ok()?;
        let shift = match order {
            Endianness::Little => from_low,
            Endianness::Big => bits.checked_sub(from_low)?.checked_sub(width)?,
        };
        placed.push(Piece {
            value,
            shift,
            width,
        });
        at = piece_end;
    }
    (at == end).then_some((bits, placed))
}

/// Part of one value stored whole that a load read: the bits from `from` of
/// the cell the value was stored to, of the `size` it fills.
#[derive(Debug, Clone, Copy)]
struct HeldPart {
    /// The value stored whole.
    value: SsaVarId,
    /// The bit of the value's cell the part starts at.
    from: i64,
    /// The part's width in bits.
    bits: u32,
    /// The value's width in bits.
    size: u32,
}

/// One value's parts being stored back together, filling `extent`.
struct Reassembly<T: Target> {
    /// The value the parts are of.
    value: SsaVarId,
    /// The bytes the parts fill once every one is stored.
    extent: MemoryLocation<T>,
    /// Each part stored so far: the bit of the extent it starts at, its
    /// width, and the store's instruction index.
    stored: Vec<(i64, u32, usize)>,
    /// The address and access of the store of the part at the extent's
    /// start, once it is stored.
    head: Option<(SsaVarId, AccessQualifiers)>,
}

/// The copies a block makes of stored values in parts ([`Rewrite::Rejoin`]):
/// the loads holding a part of a value stored whole, and the stores putting
/// the parts back together at another place.
///
/// A copy is rejoined only where nothing between its first store and its
/// last reads or writes the bytes it fills, and nothing may leave the block
/// there, so writing all of it at its last store is the same as writing it
/// part by part.
struct Rejoining<T: Target> {
    /// Each load's result that holds part of a value stored whole.
    parts: FxHashMap<SsaVarId, HeldPart>,
    /// The copies whose parts are being stored.
    copies: Vec<Reassembly<T>>,
}

impl<T: Target> Rejoining<T> {
    /// Starts a block with no part known and no copy under way.
    fn new() -> Self {
        Self {
            parts: FxHashMap::default(),
            copies: Vec::new(),
        }
    }

    /// Notes the load of `location` into `dest`: it may read the bytes of a
    /// copy under way, which then stays in parts, and it holds part of a
    /// value an available store wrote whole where that store's cell holds
    /// every bit it reads and more.
    fn load(
        &mut self,
        ssa: &SsaFunction<T>,
        available: &[(MemoryLocation<T>, SsaVarId)],
        location: &MemoryLocation<T>,
        dest: SsaVarId,
    ) {
        self.read(location);
        let MemoryLocation::Indirect(load) = location else {
            return;
        };
        let Some(bits) = load.size_bits.filter(|bits| *bits > 0) else {
            return;
        };
        if load.index.is_some() || load.address_space.is_some() {
            return;
        }
        let Some(end) = load.offset_bits.checked_add(i64::from(bits)) else {
            return;
        };
        let held = available.iter().find_map(|(known, value)| {
            let MemoryLocation::Indirect(cell) = known else {
                return None;
            };
            let size = cell.size_bits?;
            let inside = cell.base == load.base
                && cell.index.is_none()
                && cell.address_space.is_none()
                && cell.offset_bits <= load.offset_bits
                && end <= cell.offset_bits.checked_add(i64::from(size))?
                && size > bits;
            let whole = ssa.bit_width(*value) == Some(size);
            (inside && whole).then_some(HeldPart {
                value: *value,
                from: load.offset_bits.checked_sub(cell.offset_bits)?,
                bits,
                size,
            })
        });
        if let Some(held) = held {
            self.parts.insert(dest, held);
        }
    }

    /// Notes a read of `location`: each copy whose bytes it may read stays in
    /// parts.
    fn read(&mut self, location: &MemoryLocation<T>) {
        self.copies.retain(|copy| !copy.extent.may_alias(location));
    }

    /// Notes that nothing is known past this point: a barrier, or an
    /// instruction a throw may leave the block from.
    fn clear(&mut self) {
        self.copies.clear();
    }

    /// Notes the store of `value` to `location` by `op` at `(block, instr)`:
    /// it continues the copy of the value its part belongs to, or starts one,
    /// and any other copy whose bytes it may write stays in parts.
    ///
    /// # Returns
    ///
    /// The rewrite of a copy this store completes: every part stored, none
    /// twice, and the part at the start stored by a plain or vector store
    /// whose address the rewrite then stores the value at.
    fn store(
        &mut self,
        op: &SsaOp<T>,
        location: &MemoryLocation<T>,
        value: SsaVarId,
        (block, instr): (usize, usize),
    ) -> Option<Rewrite<T>> {
        let part = self.parts.get(&value).copied();
        let placed = part.and_then(|part| {
            let MemoryLocation::Indirect(cell) = location else {
                return None;
            };
            let fits = cell.size_bits == Some(part.bits)
                && cell.index.is_none()
                && cell.address_space.is_none();
            let start = cell.offset_bits.checked_sub(part.from)?;
            let extent = MemoryLocation::Indirect(IndirectLocation {
                offset_bits: start,
                size_bits: Some(part.size),
                ..*cell
            });
            fits.then_some((part, extent))
        });
        let Some((part, extent)) = placed else {
            self.read(location);
            return None;
        };
        let of_this = |copy: &Reassembly<T>| copy.value == part.value && copy.extent == extent;
        self.copies
            .retain(|copy| of_this(copy) || !copy.extent.may_alias(location));
        let index = match self.copies.iter().position(of_this) {
            Some(index) => index,
            None => {
                self.copies.push(Reassembly {
                    value: part.value,
                    extent,
                    stored: Vec::new(),
                    head: None,
                });
                self.copies.len().checked_sub(1)?
            }
        };
        let copy = self.copies.get_mut(index)?;
        let end = part.from.checked_add(i64::from(part.bits))?;
        let overlaps = copy.stored.iter().any(|(from, bits, _)| {
            from.checked_add(i64::from(*bits))
                .is_none_or(|stored_end| *from < end && part.from < stored_end)
        });
        if overlaps {
            self.copies.swap_remove(index);
            return None;
        }
        copy.stored.push((part.from, part.bits, instr));
        if part.from == 0 {
            copy.head = match op {
                SsaOp::StoreIndirect { addr, access, .. }
                | SsaOp::VectorStore { addr, access, .. } => Some((*addr, *access)),
                _ => None,
            };
        }
        let filled: u64 = copy
            .stored
            .iter()
            .map(|(_, bits, _)| u64::from(*bits))
            .sum();
        if filled != u64::from(part.size) {
            return None;
        }
        let done = self.copies.swap_remove(index);
        let (addr, access) = done.head?;
        let mut stores: Vec<usize> = done.stored.iter().map(|(_, _, at)| *at).collect();
        stores.sort_unstable();
        Some(Rewrite::Rejoin {
            value: done.value,
            addr,
            access,
            block,
            stores,
        })
    }
}

/// The part of one wider stored value a load reads ([`sliced_from`]).
struct Sliced {
    /// The stored value.
    value: SsaVarId,
    /// The bit of the value the load's lowest bit is.
    from: u32,
    /// The load's width in bits.
    bits: u32,
    /// Whether the load reads from where the value was stored as a whole:
    /// the cell's start, of a value stored by one store. A lane a segment
    /// store wrote is part of an interleaving, not of a value stored whole.
    at_start: bool,
}

/// The one available stored value that holds every bit `location` reads,
/// wider than it: the value, the bit of it the load's lowest bit is, and the
/// load's width.
///
/// A vector serves a load of a whole lane of its, at the lane's width: a
/// lane of the vector's own on either byte order, any aligned width on a
/// little-endian target, where lane `i` of a narrower view is the bytes from
/// `i` times its width. An integer of at most 64 bits serves any load inside
/// it, its bits counted from the low end the byte order puts there.
fn sliced_from<T: Target>(
    ssa: &SsaFunction<T>,
    available: &[(MemoryLocation<T>, SsaVarId)],
    parts: &[(MemoryLocation<T>, SsaVarId, u32)],
    location: &MemoryLocation<T>,
    order: Endianness,
) -> Option<Sliced> {
    let MemoryLocation::Indirect(load) = location else {
        return None;
    };
    let bits = load.size_bits.filter(|bits| *bits > 0 && *bits <= 128)?;
    if load.index.is_some() || load.address_space.is_some() {
        return None;
    }
    let end = load.offset_bits.checked_add(i64::from(bits))?;
    // Within one lane a segment store wrote: the lane's bits, from the bit
    // the lane starts at, on a little-endian target where they lie in order.
    let in_part = parts.iter().find_map(|(known, value, start)| {
        let MemoryLocation::Indirect(cell) = known else {
            return None;
        };
        let size = cell.size_bits?;
        let cell_end = cell.offset_bits.checked_add(i64::from(size))?;
        let inside = cell.base == load.base
            && cell.index.is_none()
            && cell.address_space.is_none()
            && cell.offset_bits <= load.offset_bits
            && end <= cell_end;
        if !inside || order != Endianness::Little {
            return None;
        }
        let from = start
            .checked_add(u32::try_from(load.offset_bits.checked_sub(cell.offset_bits)?).ok()?)?;
        from.is_multiple_of(bits).then_some(Sliced {
            value: *value,
            from,
            bits,
            at_start: false,
        })
    });
    if in_part.is_some() {
        return in_part;
    }
    available.iter().find_map(|(known, value)| {
        let MemoryLocation::Indirect(cell) = known else {
            return None;
        };
        let size = cell.size_bits?;
        let cell_end = cell.offset_bits.checked_add(i64::from(size))?;
        if cell.base != load.base
            || cell.index.is_some()
            || cell.address_space.is_some()
            || cell.offset_bits > load.offset_bits
            || end > cell_end
            || size <= bits
        {
            return None;
        }
        let from = u32::try_from(load.offset_bits.checked_sub(cell.offset_bits)?).ok()?;
        let at_start = from == 0;
        let held = ssa.variable(*value)?.var_type();
        if T::bit_width(held) != Some(size) {
            return None;
        }
        match T::vector_shape(held) {
            Some(shape) => {
                let aligned = from.is_multiple_of(bits) && size.is_multiple_of(bits);
                let laid = order == Endianness::Little || shape.lane_bits == bits;
                (aligned && laid).then_some(Sliced {
                    value: *value,
                    from,
                    bits,
                    at_start,
                })
            }
            None if size <= 64 && T::is_integer(held) => Some(Sliced {
                value: *value,
                from: match order {
                    Endianness::Little => from,
                    Endianness::Big => size.checked_sub(from)?.checked_sub(bits)?,
                },
                bits,
                at_start,
            }),
            None => None,
        }
    })
}

/// The elements a segment store `op` writes of `extent`, each a cell holding
/// a lane of one of its values: lane `j` of segment `m` is element `j *
/// segments + m` interleaved, `m * lanes + j` consecutive. Nothing for any
/// other operation, a masked store, or an extent of no known place.
fn segment_parts<T: Target>(
    ssa: &SsaFunction<T>,
    op: &SsaOp<T>,
    extent: &MemoryLocation<T>,
) -> Vec<(MemoryLocation<T>, SsaVarId, u32)> {
    let (
        SsaOp::VectorSegmentStore {
            values,
            mask: None,
            vector_type,
            segments,
            layout,
            ..
        },
        MemoryLocation::Indirect(cell),
    ) = (op, extent)
    else {
        return Vec::new();
    };
    let Some(shape) = T::vector_shape(vector_type) else {
        return Vec::new();
    };
    let (lane_bits, lanes) = (shape.lane_bits, shape.lane_count);
    let element = |segment: u32, lane: u32| match layout {
        VectorSegmentLayout::Interleaved => lane.checked_mul(*segments)?.checked_add(segment),
        VectorSegmentLayout::Consecutive => segment.checked_mul(lanes)?.checked_add(lane),
    };
    let mut parts = Vec::new();
    for (segment, value) in (0u32..).zip(values) {
        if ssa
            .variable(*value)
            .map(|variable| T::bit_width(variable.var_type()))
            != Some(T::bit_width(vector_type))
        {
            return Vec::new();
        }
        for lane in 0..lanes {
            let Some(offset) = element(segment, lane)
                .and_then(|element| i64::from(element).checked_mul(i64::from(lane_bits)))
                .and_then(|offset| cell.offset_bits.checked_add(offset))
            else {
                return Vec::new();
            };
            let part = IndirectLocation::new(
                cell.base,
                None,
                0,
                offset,
                Some(lane_bits),
                None,
                cell.ptr_size,
            );
            let Some(start) = lane.checked_mul(lane_bits) else {
                return Vec::new();
            };
            parts.push((MemoryLocation::Indirect(part), *value, start));
        }
    }
    parts
}

/// Replaces the load at `at` by `bits` from bit `from` of `value`
/// ([`sliced_from`]): the lane a reinterpretation of a vector in lanes that
/// wide holds there, or an integer shifted down and narrowed, read as
/// `read_type`.
///
/// # Returns
///
/// Whether the load was replaced.
///
/// # Errors
///
/// Returns an error when the editor refuses an insertion or the replacement.
fn slice_load<T: Target>(
    editor: &mut SsaEditor<'_, T>,
    dest: SsaVarId,
    read_type: &T::Type,
    value: SsaVarId,
    (from, bits): (u32, u32),
    (block, instr): (usize, usize),
) -> Result<bool> {
    let Some(held) = editor
        .function()
        .variable(value)
        .map(|variable| variable.var_type().clone())
    else {
        return Ok(false);
    };
    let Some(size) = T::bit_width(&held) else {
        return Ok(false);
    };
    let mut at = instr;
    let mut emit = |editor: &mut SsaEditor<'_, T>,
                    ty: T::Type,
                    op: &dyn Fn(SsaVarId) -> SsaOp<T>|
     -> Result<SsaVarId> {
        let fresh = editor.create_fresh_variable(
            VariableOrigin::Phi,
            0,
            DefSite::instruction(block, at),
            ty,
        );
        editor.insert_instruction(block, at, SsaInstruction::synthetic(op(fresh)))?;
        at = at.saturating_add(1);
        Ok(fresh)
    };
    if T::vector_shape(&held).is_some() {
        let lanes = size.checked_div(bits).unwrap_or(0);
        let Some(view) =
            VectorShape::new(lanes, VectorLaneKind::Integer, bits, size).and_then(T::vector_type)
        else {
            return Ok(false);
        };
        let Some(lane_type) = VectorShape::new(lanes, VectorLaneKind::Integer, bits, size)
            .and_then(T::vector_lane_type)
        else {
            return Ok(false);
        };
        let relaid = emit(editor, view.clone(), &|dest| SsaOp::VectorReinterpret {
            dest,
            value,
            target_type: view.clone(),
        })?;
        let lane = emit(editor, lane_type.clone(), &|dest| SsaOp::VectorExtract {
            dest,
            vector: relaid,
            lane: from.checked_div(bits).unwrap_or(0),
        })?;
        editor.replace_instruction_op(
            block,
            at,
            SsaOp::reinterpretation(dest, lane, read_type.clone(), &lane_type),
        )?;
        return Ok(true);
    }
    let (Some(wide), Some(narrow), Some(amount_type)) = (
        T::integer_type(size, false),
        T::integer_type(bits, false),
        T::integer_type(32, true),
    ) else {
        return Ok(false);
    };
    let as_integer = if held == wide {
        value
    } else {
        emit(editor, wide.clone(), &|dest| {
            SsaOp::reinterpretation(dest, value, wide.clone(), &held)
        })?
    };
    let shifted = if from == 0 {
        as_integer
    } else {
        let amount = emit(editor, amount_type.clone(), &|dest| SsaOp::Const {
            dest,
            value: ConstValue::I32(i32::try_from(from).unwrap_or(i32::MAX)),
        })?;
        emit(editor, wide.clone(), &|dest| SsaOp::Shr {
            dest,
            value: as_integer,
            amount,
            unsigned: true,
            flags: None,
        })?
    };
    let narrowed = emit(editor, narrow.clone(), &|dest| SsaOp::IntConv {
        dest,
        operand: shifted,
        target: narrow.clone(),
        overflow_check: false,
        unsigned: true,
    })?;
    editor.replace_instruction_op(
        block,
        at,
        SsaOp::reinterpretation(dest, narrowed, read_type.clone(), &narrow),
    )?;
    Ok(true)
}

/// Records `value` as the live contents of `location`, replacing any entry for
/// the same cell.
///
/// Matches on exact equality rather than [`MemoryLocation::must_alias`] so a
/// location keeps a single slot: the same cell is updated in place, and a
/// distinct-but-aliasing cell (already invalidated by the caller) gets its own
/// entry.
///
/// # Arguments
///
/// * `available` - Availability map to update.
/// * `location` - The location whose contents are now known.
/// * `value` - The value the location holds.
fn set_available<T: Target>(
    available: &mut Vec<(MemoryLocation<T>, SsaVarId)>,
    location: MemoryLocation<T>,
    value: SsaVarId,
) {
    if let Some(slot) = available
        .iter_mut()
        .find(|(known, _)| *known == location)
        .map(|(_, existing)| existing)
    {
        *slot = value;
        return;
    }
    available.push((location, value));
}

/// Applies planned rewrites in one checked editor session.
///
/// A forward is only completed when *every* use of the redundant value could be
/// replaced - the editor's dominance check may refuse some - otherwise the
/// access stays and only the replaced uses benefit.
///
/// Runs under [`SsaRollbackPolicy::OnFailure`], so a failed edit unwinds the
/// whole batch and the function is reported unchanged; only fully applied
/// batches emit the [`EventKind::LoadForwarded`] / [`EventKind::DeadStoreRemoved`]
/// summary events.
///
/// # Arguments
///
/// * `ssa` - The SSA function to rewrite in place.
/// * `env` - The pass environment: where events are recorded, naming the method.
/// * `rewrites` - The proven-safe rewrites to apply, from [`MemoryPlanner::rewrites`].
/// * `folds` - The image loads to fold, from [`MemoryPlanner::image_folds`]. Each load is
///   rewritten in place into the constant, so it keeps its value and its
///   readers.
///
/// # Returns
///
/// `true` if any load was forwarded or folded, or any dead store removed.
fn apply_rewrites<T>(
    ssa: &mut SsaFunction<T>,
    env: &PassEnv<'_, T>,
    rewrites: Vec<Rewrite<T>>,
    folds: Vec<Fold<T>>,
) -> bool
where
    T: Target,
{
    let mut forwarded = 0usize;
    let mut dead_stores = 0usize;
    let mut dead_loads = 0usize;
    let mut image_folds = 0usize;
    let mut rejoined = 0usize;
    // Taken before the edit, which moves the loads it folds and removes the
    // dead ones: once each is a constant or gone, only this says the function
    // read the image there, and whether it entered what it read.
    let entered = ssa.entered_values();
    let read_at = |block: usize,
                   instr: usize,
                   address: u64,
                   len: u64,
                   value_type: T::Type,
                   value: Option<ConstValue<T>>| {
        let instruction = ssa
            .block(block)
            .and_then(|body| body.instructions().get(instr));
        ImageRead {
            address,
            len,
            instruction: instruction.and_then(SsaInstruction::id),
            original: instruction.map(|instruction| instruction.original().clone()),
            value_type,
            value,
            entered: instruction
                .and_then(|instruction| instruction.op().dest())
                .is_some_and(|dest| entered.contains(&dest)),
        }
    };
    let reads: Vec<ImageRead<T>> = folds
        .iter()
        .map(|fold| {
            read_at(
                fold.block,
                fold.instr,
                fold.address,
                fold.len,
                fold.read_type.clone(),
                fold.values.first().map(|(_, value)| value.clone()),
            )
        })
        .chain(rewrites.iter().filter_map(|rewrite| match rewrite {
            Rewrite::DeadImageLoad {
                block,
                instr,
                address,
                len,
                read_type,
            } => Some(read_at(
                *block,
                *instr,
                *address,
                *len,
                read_type.clone(),
                None,
            )),
            _ => None,
        }))
        .collect();
    // Each completed forward's redundant value, to the value that replaced it.
    let mut replaced: FxHashMap<SsaVarId, SsaVarId> = FxHashMap::default();
    // The value standing for `value` once the completed forwards are applied:
    // a forward's redundant value has no definition left, so anything built
    // after it - a later forward, a composition's piece - reads what replaced
    // it. A chain is finite: each step replaces a value with one planned
    // earlier, and a completed forward is recorded once. The bound only guards
    // the loop.
    let surviving = |replaced: &FxHashMap<SsaVarId, SsaVarId>, value: SsaVarId| {
        let mut source = value;
        for _ in 0..=replaced.len() {
            match replaced.get(&source) {
                Some(next) => source = *next,
                None => break,
            }
        }
        source
    };

    // A composition inserts its pieces ahead of its load, so it runs after
    // every other rewrite, from the end of each block back: no insertion then
    // moves a position a rewrite still to run names.
    let (mut composes, rewrites): (Vec<Rewrite<T>>, Vec<Rewrite<T>>) = rewrites
        .into_iter()
        .partition(|rewrite| matches!(rewrite, Rewrite::Compose { .. } | Rewrite::Slice { .. }));
    composes.sort_by_key(|rewrite| match rewrite {
        Rewrite::Compose { block, instr, .. } | Rewrite::Slice { block, instr, .. } => {
            Reverse((*block, *instr))
        }
        _ => Reverse((0, 0)),
    });
    // A load into a version of a slot the function reads as storage also
    // writes the slot: forwarded or unread, the write stays.
    let slots = ssa.storage_slots();
    let report = ssa.edit(
        SsaEditOptions::new().with_rollback(SsaRollbackPolicy::OnFailure),
        |editor| {
            let writes_slot = |editor: &SsaEditor<'_, T>, block: usize, instr: usize| {
                editor
                    .function()
                    .block(block)
                    .and_then(|this| this.instruction(instr))
                    .is_some_and(|this| slots.written_by(editor.function(), this.op()))
            };
            // From the end of each block back, so a segment load's further
            // constants, inserted after it, move no position a fold still to
            // run names.
            let mut folds = folds;
            folds.sort_by_key(|fold| Reverse((fold.block, fold.instr)));
            for fold in folds {
                let mut values = fold.values.into_iter();
                let Some((dest, value)) = values.next() else {
                    continue;
                };
                let op = SsaOp::Const { dest, value };
                match fold.dest_type {
                    Some(dest_type) => {
                        editor.replace_instruction_op_as(fold.block, fold.instr, op, dest_type)?;
                    }
                    None => editor.replace_instruction_op(fold.block, fold.instr, op)?,
                }
                for (offset, (dest, value)) in (1usize..).zip(values) {
                    editor.insert_instruction(
                        fold.block,
                        fold.instr.saturating_add(offset),
                        SsaInstruction::synthetic(SsaOp::Const { dest, value }),
                    )?;
                }
                image_folds = image_folds.saturating_add(1);
            }
            for rewrite in rewrites {
                match rewrite {
                    Rewrite::Forward {
                        redundant,
                        read_type,
                        available,
                        block,
                        instr,
                    } => {
                        let source = surviving(&replaced, available);
                        if let Some(read_type) = read_type {
                            // The load becomes the reinterpretation, and its
                            // result keeps its readers and its type.
                            let held = editor
                                .function()
                                .variable(source)
                                .map(|variable| variable.var_type().clone())
                                .unwrap_or_else(T::unknown_type);
                            editor.replace_instruction_op(
                                block,
                                instr,
                                SsaOp::reinterpretation(redundant, source, read_type, &held),
                            )?;
                            forwarded = forwarded.saturating_add(1);
                            continue;
                        }
                        let result = editor.replace_uses_checked(redundant, source);
                        if result.is_complete() {
                            if !writes_slot(editor, block, instr) {
                                editor.nop_instruction(block, instr)?;
                            }
                            replaced.insert(redundant, source);
                            forwarded = forwarded.saturating_add(1);
                        }
                    }
                    Rewrite::DeadStore { block, instr, .. } => {
                        editor.nop_instruction(block, instr)?;
                        dead_stores = dead_stores.saturating_add(1);
                    }
                    Rewrite::DeadImageLoad { block, instr, .. }
                    | Rewrite::DeadFrameLoad { block, instr }
                    | Rewrite::DeadBoundLoad { block, instr } => {
                        if !writes_slot(editor, block, instr) {
                            editor.nop_instruction(block, instr)?;
                            dead_loads = dead_loads.saturating_add(1);
                        }
                    }
                    Rewrite::Rejoin {
                        value,
                        addr,
                        access,
                        block,
                        stores,
                    } => {
                        let Some((last, parts)) = stores.split_last() else {
                            continue;
                        };
                        let value = surviving(&replaced, value);
                        let Some(value_type) = editor
                            .function()
                            .variable(value)
                            .map(|variable| variable.var_type().clone())
                        else {
                            continue;
                        };
                        let op = if T::vector_shape(&value_type).is_some() {
                            SsaOp::VectorStore {
                                addr,
                                value,
                                vector_type: value_type,
                                access,
                            }
                        } else {
                            SsaOp::StoreIndirect {
                                addr,
                                value,
                                value_type,
                                access,
                            }
                        };
                        for part in parts {
                            editor.nop_instruction(block, *part)?;
                        }
                        editor.replace_instruction_op(block, *last, op)?;
                        rejoined = rejoined.saturating_add(1);
                    }
                    Rewrite::Compose { .. } | Rewrite::Slice { .. } => {}
                }
            }
            for compose in composes {
                let done = match compose {
                    Rewrite::Compose {
                        dest,
                        read_type,
                        bits,
                        pieces,
                        block,
                        instr,
                    } => {
                        let pieces: Vec<Piece> = pieces
                            .into_iter()
                            .map(|piece| Piece {
                                value: surviving(&replaced, piece.value),
                                ..piece
                            })
                            .collect();
                        compose_load(editor, dest, &read_type, bits, &pieces, (block, instr))?
                    }
                    Rewrite::Slice {
                        dest,
                        read_type,
                        value,
                        at_start,
                        from,
                        bits,
                        block,
                        instr,
                    } => {
                        let value = surviving(&replaced, value);
                        let sliced = slice_load(
                            editor,
                            dest,
                            &read_type,
                            value,
                            (from, bits),
                            (block, instr),
                        )?;
                        if sliced && at_start {
                            editor.note_whole(value, dest, None);
                        }
                        sliced
                    }
                    _ => false,
                };
                if done {
                    forwarded = forwarded.saturating_add(1);
                }
            }
            Ok(())
        },
    );

    if report.is_err() {
        return false;
    }
    for read in reads {
        T::record_image_read(ssa.facts_mut(), read);
    }
    if forwarded > 0 {
        env.record(EventKind::LoadForwarded).message(format!(
            "forwarded {forwarded} load(s) from available memory"
        ));
    }
    if dead_stores > 0 {
        env.record(EventKind::DeadStoreRemoved)
            .message(format!("removed {dead_stores} dead store(s)"));
    }
    if image_folds > 0 {
        env.record(EventKind::ImageLoadFolded).message(format!(
            "folded {image_folds} load(s) of read-only image data"
        ));
    }
    if dead_loads > 0 {
        env.record(EventKind::InstructionRemoved)
            .message(format!("removed {dead_loads} dead load(s) of image data"));
    }
    if rejoined > 0 {
        env.record(EventKind::CopyPropagated)
            .message(format!("stored {rejoined} value(s) copied in parts whole"));
    }
    forwarded > 0 || dead_stores > 0 || image_folds > 0 || dead_loads > 0 || rejoined > 0
}

/// Memory optimization pass.
///
/// Runs store-to-load forwarding, redundant load elimination, and dead store
/// elimination, all gated on Memory SSA alias proofs.
///
/// # Modification scope
///
/// [`ModificationScope::InstructionsOnly`] - replaces uses and nops
/// instructions, never touching the CFG.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemoryOptimizationPass;

impl<T> FunctionPass<T> for MemoryOptimizationPass
where
    T: Target,
    T::MethodRef: Send + Sync,
{
    /// Returns this pass's unique short name for logging.
    fn name(&self) -> &'static str {
        "memory-optimization"
    }

    /// Returns this pass's human-readable description.
    fn description(&self) -> &'static str {
        "Forward stores to loads, drop redundant loads, and remove dead stores"
    }

    /// Returns the IR scope this pass may modify.
    fn modification_scope(&self) -> ModificationScope {
        ModificationScope::InstructionsOnly
    }

    /// Reports whether this pass leaves the SSA form intact for the next pass.
    fn repairs_ssa(&self) -> bool {
        true
    }

    /// Runs memory optimization on `ssa` with the host's facts.
    ///
    /// Two phases. First the loads of read-only image data fold
    /// (`plan_image_folds`), round after round until none does, since a fold can
    /// form the address the next one reads; each round is planned over a fresh
    /// Memory SSA. Then the rewrites the alias model proves safe (`plan_rewrites`)
    /// and the dead loads of image cells (`plan_dead_image_loads`) are planned over
    /// the folded function, using the Memory SSA of the round that folded nothing,
    /// and applied in one checked editor session (`apply_rewrites`). A clean
    /// function pays for one analysis.
    ///
    /// A budget the analysis runs out of is reported as
    /// [`EventKind::BudgetExhausted`], and the work stops there: an empty Memory
    /// SSA proves nothing, and fewer folds are only less precise.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to rewrite in place.
    /// * `env` - The pass environment: where events are recorded, naming the
    ///   method, and the context, budget and range facts the pass reads.
    ///
    /// # Returns
    ///
    /// `true` if any load was forwarded or folded, or any dead store removed.
    fn run(&self, ssa: &mut SsaFunction<T>, env: &PassEnv<'_, T>) -> Result<bool> {
        let context = env.context();
        // Folding first, to its own fixpoint. A folded load is the constant it
        // read, and an address built from that constant only then names the cell
        // every other copy of it names: planned together, a store through one
        // copy of a literal-pool address and a load through another were never
        // matched, and `uidget` stored `0` and read it back.
        let mut changed = false;
        // The Memory SSA of the function the last round left unfolded, which is the
        // one the rewrites are planned over: the function has not changed since.
        let mem_ssa = loop {
            let mem_ssa = MemorySsa::build(ssa, context);
            if let Some(exhaustion) = mem_ssa.exhaustion() {
                env.exhausted(exhaustion);
                return Ok(changed);
            }
            let (folds, exhaustion) = MemoryPlanner::new(ssa, &mem_ssa, context).image_folds();
            if let Some(exhaustion) = exhaustion {
                env.exhausted(exhaustion);
                return Ok(changed | apply_rewrites(ssa, env, Vec::new(), folds));
            }
            if folds.is_empty() {
                break mem_ssa;
            }
            if !apply_rewrites(ssa, env, Vec::new(), folds) {
                return Ok(changed);
            }
            changed = true;
        };
        // Every read the image maps whole is kept with the function: a later step
        // that no longer has the image knows it cannot fault.
        if let Some(image) = context.image() {
            let mapped: Vec<(u64, u64)> = mem_ssa
                .operations()
                .iter()
                .filter_map(|op| match op {
                    MemoryOp::Load {
                        location: MemoryLocation::Indirect(cell),
                        ..
                    } if cell.index.is_none() => cell.placed_in_image(image, None),
                    _ => None,
                })
                .collect();
            for (address, len) in mapped {
                T::record_mapped_read(ssa.facts_mut(), address, len);
            }
        }
        let rewrites = {
            let planner = MemoryPlanner::new(ssa, &mem_ssa, context);
            let mut rewrites = planner.rewrites();
            let rewritten: FxHashSet<(usize, usize)> = rewrites
                .iter()
                .flat_map(|rewrite| match *rewrite {
                    Rewrite::Forward { block, instr, .. }
                    | Rewrite::Compose { block, instr, .. }
                    | Rewrite::Slice { block, instr, .. }
                    | Rewrite::DeadStore { block, instr }
                    | Rewrite::DeadImageLoad { block, instr, .. }
                    | Rewrite::DeadFrameLoad { block, instr }
                    | Rewrite::DeadBoundLoad { block, instr } => vec![(block, instr)],
                    Rewrite::Rejoin {
                        block, ref stores, ..
                    } => stores.iter().map(|instr| (block, *instr)).collect(),
                })
                .collect();
            let forwarded: FxHashSet<SsaVarId> = rewrites
                .iter()
                .flat_map(|rewrite| match rewrite {
                    Rewrite::Forward { available, .. } => vec![*available],
                    Rewrite::Compose { pieces, .. } => {
                        pieces.iter().map(|piece| piece.value).collect()
                    }
                    Rewrite::Slice { value, .. } | Rewrite::Rejoin { value, .. } => vec![*value],
                    _ => Vec::new(),
                })
                .collect();
            rewrites.extend(planner.dead_image_loads(&rewritten, &forwarded));
            rewrites
        };
        if rewrites.is_empty() {
            return Ok(changed);
        }
        Ok(apply_rewrites(ssa, env, rewrites, Vec::new()) || changed)
    }
}
