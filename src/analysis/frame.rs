//! The function's own frame: where it is addressed from, and whether it is
//! private to the function.
//!
//! A cell wholly below the entry stack pointer belongs to the function
//! ([`FrameAnchors::cell`]), and nothing outside it can name the cell unless
//! an address derived from that pointer leaves: [`FrameEscape`] follows every
//! such address and says whether one does.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    analysis::{address::AddressWalk, memory::MemoryLocation},
    ir::{
        function::SsaFunction,
        integer::BitWidth,
        ops::{SsaEffectKind, SsaOp},
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
    vartables::VarBuckets,
};

/// Where a value is used: an instruction operand, or a phi operand.
#[derive(Debug, Clone, Copy)]
enum UseSite {
    /// An operand of the instruction at `(block, instr)`.
    Instruction {
        /// Block of the using instruction.
        block: usize,
        /// Position of the using instruction in its block.
        instr: usize,
    },
    /// An operand of the phi at `(block, phi)`.
    Phi {
        /// Block of the using phi.
        block: usize,
        /// Position of the phi in its block.
        phi: usize,
    },
}

/// Every use of every value, grouped by value.
///
/// The placeholder is not a value and has no uses; an id past the function's
/// variable count is grouped in a map rather than sizing the index.
struct UseIndex {
    /// The use sites, grouped by value.
    sites: VarBuckets<UseSite>,
}

impl UseIndex {
    /// Indexes every instruction and phi operand of `ssa`.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to index.
    fn build<T: Target>(ssa: &SsaFunction<T>) -> Self {
        let pad = UseSite::Instruction { block: 0, instr: 0 };
        Self {
            sites: VarBuckets::build(ssa.variable_count(), pad, |emit| visit_uses(ssa, emit)),
        }
    }
}

/// What an operation does with a frame-derived operand.
enum FrameUse {
    /// Accesses memory through it, or branches on it: nothing learns the
    /// address.
    Contained,
    /// Computes new values from it, which may carry the address in turn.
    Carries,
    /// Hands it somewhere this function does not control.
    Leaves,
}

impl FrameUse {
    /// Classifies how `op` uses the frame-derived `value`.
    ///
    /// An allow-list: only a memory access's own address operand, a branch
    /// condition, and an argument a call lists everything it touches through and
    /// does not keep ([`Capture::keeps`]) are contained, and only an operation
    /// with no effect beyond its results carries. Every other use leaves: a
    /// store's value, a field or element write, an atomic's operand, an argument
    /// a call may keep or reach in ways it does not state, an intrinsic's
    /// operand, a return or a throw. That is the conservative answer for an
    /// operation this does not recognise.
    ///
    /// A call that lists its accesses ([`CallEffects::accesses`]) and does not
    /// keep an address it is passed reaches that address's memory only through
    /// the accesses it states of it ([`CallLocation::Pointee`]), while it runs:
    /// AAPCS64 hands a `va_list` on as the address of a copy in the caller's
    /// frame, and a callee that only reads it with `va_arg` reads the copy, not
    /// the frame around it.
    ///
    /// [`Capture::keeps`]: crate::ir::ops::Capture::keeps
    /// [`CallEffects::accesses`]: crate::ir::ops::CallEffects::accesses
    /// [`CallLocation::Pointee`]: crate::ir::ops::CallLocation::Pointee
    ///
    /// # Arguments
    ///
    /// * `op` - The using operation.
    /// * `value` - The frame-derived operand.
    fn of<T: Target>(op: &SsaOp<T>, value: SsaVarId) -> Self {
        match op {
            SsaOp::Branch { .. }
            | SsaOp::BranchCmp { .. }
            | SsaOp::BranchFlags { .. }
            | SsaOp::Switch { .. } => return Self::Contained,
            _ => {}
        }
        if matches!(op.effects().kind, SsaEffectKind::Pure) {
            return Self::Carries;
        }
        if let Some(call) = op.as_call() {
            let mut uses = 0usize;
            op.for_each_use(|used| {
                if used == value {
                    uses = uses.saturating_add(1);
                }
            });
            let confined = call
                .effects
                .as_ref()
                .filter(|effects| effects.accesses().is_some())
                .map_or(0, |effects| {
                    (0u32..)
                        .zip(&call.args)
                        .filter(|(position, arg)| {
                            **arg == value && !effects.captures.keeps(*position)
                        })
                        .count()
                });
            return if uses <= confined {
                Self::Contained
            } else {
                Self::Leaves
            };
        }
        let addresses: [Option<SsaVarId>; 2] = match op {
            SsaOp::CopyBlk {
                dest_addr,
                src_addr,
                ..
            }
            | SsaOp::MoveBlk {
                dest_addr,
                src_addr,
                ..
            }
            | SsaOp::CopyObj {
                dest_addr,
                src_addr,
                ..
            } => [Some(*dest_addr), Some(*src_addr)],
            _ => [op.memory_effect().map(|effect| effect.addr), None],
        };
        let as_address = addresses
            .iter()
            .filter(|address| **address == Some(value))
            .count();
        let mut uses = 0usize;
        op.for_each_use(|used| {
            if used == value {
                uses = uses.saturating_add(1);
            }
        });
        if uses <= as_address {
            Self::Contained
        } else {
            Self::Leaves
        }
    }
}

/// The values a function derives from its entry stack pointer, and whether
/// any address among them leaves the function.
///
/// **This is the fact that makes a frame cell private to the function's own
/// pointers.** A cell below the entry stack pointer is the function's own, but
/// "own" is not the same as "unreachable": the question is whether anything
/// *else* can name it. A pointer the function was handed cannot, once this
/// answers `false`: it was formed by the caller, before this frame existed, so
/// it cannot point into it - the region below the entry stack pointer is dead
/// stack the ABI gives the caller no way to address.
/// `IndirectLocation::may_alias` cannot know that: it is a pure function on two
/// locations, sees two different SSA roots, and must answer "may".
///
/// A callee is a different matter. It reads its stack arguments out of this
/// region by convention, without being handed an address, so a call that
/// does not state its memory stays a barrier to the frame whatever this
/// answers.
///
/// # What counts as derived, and what counts as leaving
///
/// Derived is the closure of the entry stack pointer under phis and under
/// operations with no effect beyond their results - address arithmetic, casts,
/// selects, comparisons, vector lane moves. Over-approximating it can only make
/// more values look like addresses, which makes this answer `true` more often,
/// which is the safe direction. A value *loaded* from memory is not derived:
/// for a frame address to be in memory it has to have been stored there as a
/// value, and that store already counts as leaving.
///
/// Leaving is any other use: stored into memory as a value,
/// written to a field or element, passed to a call or intrinsic, returned or
/// thrown. Accessing memory *through* a frame address is not leaving - that is
/// the frame being used - so an access's own address operand is not consulted.
///
/// A function that calls one that returns twice lets its frame escape
/// whatever its addresses do: the second return resumes code no edge leads to
/// from the call that jumps back, which reads the frame as that call leaves it.
///
/// One worklist pass over the def-use index: each derived value's uses are
/// visited once.
///
/// The derived set is held whole whatever the answer, because the frame's
/// liveness needs both: a base that is *not* in it is a pointer from outside,
/// or one that reaches the frame only through an escaped address, and a base
/// that is names the frame whatever route it took there (`PtrToInt` and back
/// is one, and equality with the entry stack pointer alone would miss it - in
/// the unsound direction).
#[derive(Debug, Clone)]
pub struct FrameEscape {
    /// Every value derived from the entry stack pointer, itself included.
    derived: FxHashSet<SsaVarId>,
    /// Whether any of them may leave the function: the conservative answer
    /// every unmodelled shape falls to.
    escapes: bool,
}

impl FrameEscape {
    /// Works out which values `ssa` derives from its entry stack pointer and
    /// whether any of them leaves.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function to examine.
    /// * `entry_stack_pointer` - The value the host declared the entry stack pointer.
    #[must_use]
    pub fn of<T: Target>(ssa: &SsaFunction<T>, entry_stack_pointer: SsaVarId) -> Self {
        let uses = UseIndex::build(ssa);
        let mut derived: FxHashSet<SsaVarId> = FxHashSet::default();
        let mut escapes = false;
        derived.insert(entry_stack_pointer);
        let mut worklist = vec![entry_stack_pointer];
        while let Some(value) = worklist.pop() {
            for site in uses.sites.get(value) {
                match *site {
                    UseSite::Phi { block, phi } => {
                        let Some(node) =
                            ssa.block(block).and_then(|body| body.phi_nodes().get(phi))
                        else {
                            continue;
                        };
                        if derived.insert(node.result()) {
                            worklist.push(node.result());
                        }
                    }
                    UseSite::Instruction { block, instr } => {
                        let Some(instruction) = ssa
                            .block(block)
                            .and_then(|body| body.instructions().get(instr))
                        else {
                            continue;
                        };
                        match FrameUse::of(instruction.op(), value) {
                            FrameUse::Contained => {}
                            FrameUse::Carries => {
                                for def in instruction.defs() {
                                    if derived.insert(def) {
                                        worklist.push(def);
                                    }
                                }
                            }
                            FrameUse::Leaves => escapes = true,
                        }
                    }
                }
            }
        }
        // A call that returns twice (`setjmp`) comes back a second time from
        // whichever later call jumps to it, and the code it resumes reads what
        // the frame holds then: dash's `main` sets `state` before each profile it
        // reads and reads it where `setjmp` returns again. No edge of the function
        // leads from those calls to that code, so its frame is as good as escaped
        // to every call: what each reads is what it may resume.
        let returns_twice = ssa.iter_blocks().any(|(_, body)| {
            body.instructions().iter().any(|instruction| {
                instruction
                    .op()
                    .as_call()
                    .and_then(|call| call.effects.as_ref())
                    .is_some_and(|effects| effects.returns_twice)
            })
        });
        Self {
            derived,
            escapes: escapes || returns_twice,
        }
    }

    /// Returns whether any address derived from the entry stack pointer may
    /// leave the function.
    #[must_use]
    pub fn escapes(&self) -> bool {
        self.escapes
    }

    /// Returns every value derived from the entry stack pointer, itself
    /// included.
    #[must_use]
    pub fn derived(&self) -> &FxHashSet<SsaVarId> {
        &self.derived
    }
}

/// A cell of the function's own frame, by its extent below the value it is
/// measured from: the entry stack pointer wherever the anchor it is addressed
/// from has one place there ([`FrameAnchors::canonical`]), the anchor itself
/// otherwise.
///
/// Two own-frame locations with equal keys name the same bytes: both have an
/// anchor as base, no index and the default address space
/// ([`FrameAnchors::cell`] requires all three), which leaves where the anchor
/// lies, the offset and the width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CellKey {
    /// The anchor the cell is measured from ([`FrameAnchors`]).
    pub anchor: SsaVarId,
    /// Displacement from the anchor, in bits.
    pub offset_bits: i64,
    /// Width of the access, in bits.
    pub size_bits: u32,
}

/// The values the function's own frame is addressed from, each placed
/// relative to the entry stack pointer.
///
/// The entry stack pointer, and each value that rounds an address measured
/// from an anchor down to an alignment (`and esp, -16`). Rounding down moves an
/// address by less than the alignment, so a realigned anchor lies within a
/// known interval of the entry stack pointer, and a cell wholly below the
/// entry stack pointer wherever in that interval its anchor falls is the
/// function's own. A realigned frame - x86 `main`, a function with an
/// over-aligned local - keeps its register saves and locals below such a
/// value, at a distance from the entry stack pointer no constant states.
///
/// And each phi that joins addresses of one place on every path: the stack
/// pointer after an `if` whose arms each made a call, or around a loop that
/// leaves it where it found it.
#[derive(Debug)]
pub struct FrameAnchors {
    /// Each anchor, with the lowest and the highest place it can have,
    /// measured from the entry stack pointer in bits.
    anchors: FxHashMap<SsaVarId, (i64, i64)>,
    /// Each phi anchor, with the value it is measured from and the bits from
    /// it: one place on every path relative to that value, even where the
    /// value has none relative to entry.
    rooted: FxHashMap<SsaVarId, (SsaVarId, i64)>,
    /// The entry stack pointer, which an anchor of one place is measured from.
    entry: SsaVarId,
}

impl FrameAnchors {
    /// Finds the anchors of `ssa`'s frame.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function being analyzed.
    /// * `entry_stack_pointer` - The value the host declared the entry stack pointer.
    /// * `ptr_size` - The target pointer width, which an alignment mask is read at.
    pub fn of<T: Target>(
        ssa: &SsaFunction<T>,
        entry_stack_pointer: SsaVarId,
        ptr_size: PointerSize,
    ) -> Self {
        let bits = u32::try_from(ptr_size.bytes())
            .unwrap_or(8)
            .saturating_mul(8);
        let width = BitWidth::new(bits).mask();
        // An alignment mask is all ones above its alignment and zeros below
        // it, at the pointer's width: `-16` however wide it was stated. The
        // bits below the alignment are how far rounding moves an address.
        let slack_bits = |mask: SsaVarId| {
            ssa.const_i64(mask).and_then(|mask| {
                let below = !mask.cast_unsigned() & width;
                (below != width && below.wrapping_add(1).is_power_of_two())
                    .then(|| i64::try_from(below).ok()?.checked_mul(8))
                    .flatten()
            })
        };
        let realignments: Vec<(SsaVarId, SsaVarId, i64)> = ssa
            .blocks()
            .iter()
            .flat_map(|block| block.instructions())
            .filter_map(|instruction| match instruction.op() {
                SsaOp::And {
                    dest, left, right, ..
                } => slack_bits(*right)
                    .map(|slack| (*dest, *left, slack))
                    .or_else(|| slack_bits(*left).map(|slack| (*dest, *right, slack))),
                _ => None,
            })
            .collect();
        let mut anchors: FxHashMap<SsaVarId, (i64, i64)> =
            FxHashMap::from_iter([(entry_stack_pointer, (0, 0))]);
        let mut rooted = FxHashMap::default();
        loop {
            let before = anchors.len();
            for (phi, (root, offset_bits)) in
                merged_anchors(ssa, &anchors, &rooted, entry_stack_pointer, ptr_size)
            {
                let Some(&(low, high)) = anchors.get(&root) else {
                    continue;
                };
                if let (Some(low), Some(high)) =
                    (low.checked_add(offset_bits), high.checked_add(offset_bits))
                {
                    anchors.insert(phi, (low, high));
                    rooted.insert(phi, (root, offset_bits));
                }
            }
            for &(dest, source, slack) in &realignments {
                if anchors.contains_key(&dest) {
                    continue;
                }
                let address = AddressWalk::new(ssa, ptr_size).normalize(source);
                let Some(&(low, high)) = address
                    .base
                    .value()
                    .filter(|_| address.index.is_none())
                    .and_then(|base| anchors.get(&base))
                else {
                    continue;
                };
                if let (Some(low), Some(high)) = (
                    low.checked_add(address.offset_bits)
                        .and_then(|low| low.checked_sub(slack)),
                    high.checked_add(address.offset_bits),
                ) {
                    anchors.insert(dest, (low, high));
                }
            }
            if anchors.len() == before {
                break;
            }
        }
        Self {
            anchors,
            rooted,
            entry: entry_stack_pointer,
        }
    }

    /// Returns the frame cell `location` names, when it lies wholly below the
    /// entry stack pointer.
    ///
    /// The whole extent, `[offset, offset + size)` from its anchor, has to end
    /// at or below the entry stack pointer wherever the anchor lies: a cell
    /// that straddles it overlaps the return address or the caller's frame,
    /// both of which outlive the function. An access of unknown width, through
    /// an index, or in a qualified address space is never claimed.
    ///
    /// # Arguments
    ///
    /// * `location` - The accessed location.
    pub fn cell<T: Target>(&self, location: &MemoryLocation<T>) -> Option<CellKey> {
        let MemoryLocation::Indirect(cell) = location else {
            return None;
        };
        let anchor = cell.base.value()?;
        let &(_, high) = self.anchors.get(&anchor)?;
        if cell.index.is_some() || cell.address_space.is_some() {
            return None;
        }
        let size_bits = cell.size_bits?;
        let end = cell.offset_bits.checked_add(i64::from(size_bits))?;
        if high.checked_add(end)? > 0 {
            return None;
        }
        let (anchor, offset_bits) = self.canonical(anchor, cell.offset_bits)?;
        Some(CellKey {
            anchor,
            offset_bits,
            size_bits,
        })
    }

    /// Returns the cell of the caller's memory `location` names, when it lies
    /// wholly at or above the entry stack pointer: the stack arguments, the
    /// return address, the caller's frame, all of which outlive the function.
    ///
    /// The whole extent has to start at or above the entry stack pointer
    /// wherever the anchor lies; a cell straddling it is neither the
    /// function's own ([`Self::cell`]) nor the caller's, and is not tracked.
    /// An access of unknown width, through an index, or in a qualified address
    /// space is never claimed.
    ///
    /// # Arguments
    ///
    /// * `location` - The accessed location.
    pub fn caller_cell<T: Target>(&self, location: &MemoryLocation<T>) -> Option<CellKey> {
        let MemoryLocation::Indirect(cell) = location else {
            return None;
        };
        let anchor = cell.base.value()?;
        let &(low, _) = self.anchors.get(&anchor)?;
        if cell.index.is_some() || cell.address_space.is_some() {
            return None;
        }
        let size_bits = cell.size_bits?;
        cell.offset_bits.checked_add(i64::from(size_bits))?;
        if low.checked_add(cell.offset_bits)? < 0 {
            return None;
        }
        let (anchor, offset_bits) = self.canonical(anchor, cell.offset_bits)?;
        Some(CellKey {
            anchor,
            offset_bits,
            size_bits,
        })
    }

    /// Whether `location` lies within the word at the entry stack pointer, at
    /// a fixed offset from an anchor whose every place keeps it there: the
    /// caller's last stack word, which a stack growing downward holds where
    /// the function is entered ([`VariableRole::EntryStackPointer`]) - the
    /// return address an x86 `call` pushes.
    ///
    /// [`VariableRole::EntryStackPointer`]: crate::ir::variable::VariableRole::EntryStackPointer
    pub fn in_entry_word<T: Target>(&self, location: &MemoryLocation<T>, word_bits: i64) -> bool {
        let MemoryLocation::Indirect(cell) = location else {
            return false;
        };
        let Some(&(low, high)) = cell
            .base
            .value()
            .and_then(|anchor| self.anchors.get(&anchor))
        else {
            return false;
        };
        let Some(end) = cell
            .size_bits
            .and_then(|bits| cell.offset_bits.checked_add(i64::from(bits)))
        else {
            return false;
        };
        cell.index.is_none()
            && cell.address_space.is_none()
            && low
                .checked_add(cell.offset_bits)
                .is_some_and(|start| start >= 0)
            && high.checked_add(end).is_some_and(|end| end <= word_bits)
    }

    /// `offset_bits` from `anchor`, measured from the value the anchor is
    /// rooted at where it is a phi, from the entry stack pointer where the
    /// anchor has one place there, and from the anchor itself where it does
    /// not: one byte of the frame has one name wherever it can, so a store
    /// through a merged stack pointer overwrites the cell a push through the
    /// value it merges wrote.
    pub fn canonical(&self, anchor: SsaVarId, offset_bits: i64) -> Option<(SsaVarId, i64)> {
        if let Some(&(root, at)) = self.rooted.get(&anchor) {
            return Some((root, at.checked_add(offset_bits)?));
        }
        match self.place(anchor) {
            Some(place) if anchor != self.entry => {
                Some((self.entry, place.checked_add(offset_bits)?))
            }
            _ => Some((anchor, offset_bits)),
        }
    }

    /// The bytes `location` names in the frame - its anchor and its extent in
    /// bits from it, measured as [`Self::canonical`] measures - when it is an
    /// access at a fixed offset from an anchor with a width.
    pub fn extent<T: Target>(&self, location: &MemoryLocation<T>) -> Option<(SsaVarId, i64, i64)> {
        let MemoryLocation::Indirect(access) = location else {
            return None;
        };
        let anchor = access
            .base
            .value()
            .filter(|base| self.contains(*base) && access.index.is_none())?;
        let end = access
            .size_bits
            .and_then(|size| access.offset_bits.checked_add(i64::from(size)))?;
        let (anchor, start) = self.canonical(anchor, access.offset_bits)?;
        let end = start.checked_add(end.checked_sub(access.offset_bits)?)?;
        Some((anchor, start, end))
    }

    /// Whether `value` is one of the anchors.
    pub fn contains(&self, value: SsaVarId) -> bool {
        self.anchors.contains_key(&value)
    }

    /// The place of `anchor` in bits from the entry stack pointer, when it
    /// has exactly one.
    pub fn place(&self, anchor: SsaVarId) -> Option<i64> {
        self.anchors
            .get(&anchor)
            .and_then(|&(low, high)| (low == high).then_some(low))
    }

    /// Whether `[start, end)` from `anchor` and `[other_start, other_end)` from
    /// `other` may overlap, wherever in their intervals the two anchors lie.
    ///
    /// # Arguments
    ///
    /// * `anchor`, `start`, `end` - One extent, in bits from its anchor.
    /// * `other`, `other_start`, `other_end` - The other, in bits from its.
    pub fn may_overlap(
        &self,
        (anchor, start, end): (SsaVarId, i64, i64),
        (other, other_start, other_end): (SsaVarId, i64, i64),
    ) -> bool {
        let (Some(&(low, high)), Some(&(other_low, other_high))) =
            (self.anchors.get(&anchor), self.anchors.get(&other))
        else {
            return true;
        };
        low.saturating_add(start) < other_high.saturating_add(other_end)
            && other_low.saturating_add(other_start) < high.saturating_add(end)
    }
}

/// The phis not yet in `anchors` that hold one exact place in the frame
/// on every path, each with that place: the value it is measured from, the
/// entry stack pointer or an alignment's result, and the bits from it.
///
/// Where two paths leave the stack pointer at one depth they reach the
/// merge as two values, and the phi that joins them is an address like
/// either: after an `if` whose arm makes a call, every push below it is
/// through the phi. The places are solved optimistically, so a loop that
/// leaves the stack pointer where it found it resolves, a phi of its
/// own header's value included: each phi is taken to hold the place its
/// other operands give, and is dropped once an operand places it
/// elsewhere, cannot be placed, or depends on a dropped phi. What is left
/// holds its place on every path, by induction over the paths.
fn merged_anchors<T: Target>(
    ssa: &SsaFunction<T>,
    anchors: &FxHashMap<SsaVarId, (i64, i64)>,
    rooted: &FxHashMap<SsaVarId, (SsaVarId, i64)>,
    entry: SsaVarId,
    ptr_size: PointerSize,
) -> Vec<(SsaVarId, (SsaVarId, i64))> {
    let placed_at = |base: SsaVarId| {
        rooted.get(&base).copied().or_else(|| {
            anchors
                .get(&base)
                .map(|&(low, high)| if low == high { (entry, low) } else { (base, 0) })
        })
    };
    let phis: FxHashMap<SsaVarId, Vec<SsaVarId>> = ssa
        .blocks()
        .iter()
        .flat_map(|block| block.phi_nodes())
        .filter(|phi| !anchors.contains_key(&phi.result()))
        .map(|phi| {
            (
                phi.result(),
                phi.operands()
                    .iter()
                    .map(|operand| operand.value())
                    .collect(),
            )
        })
        .collect();
    let mut candidates: FxHashMap<SsaVarId, Option<(SsaVarId, i64)>> =
        phis.keys().map(|phi| (*phi, None)).collect();
    loop {
        let mut changed = false;
        for (phi, operands) in &phis {
            let Some(guess) = candidates.get(phi).copied() else {
                continue;
            };
            let mut place: Option<(SsaVarId, i64)> = None;
            let mut placed = true;
            for operand in operands {
                let address = AddressWalk::new(ssa, ptr_size).normalize(*operand);
                let base = address.base.value().filter(|_| address.index.is_none());
                let from = base.and_then(|base| {
                    placed_at(base)
                        .map(Some)
                        .or_else(|| candidates.get(&base).copied())
                });
                match from {
                    // Not placed at one point, or through a phi already
                    // dropped.
                    None => {
                        placed = false;
                        break;
                    }
                    // A phi whose place is not yet known says nothing.
                    Some(None) => {}
                    Some(Some((root, at))) => match at.checked_add(address.offset_bits) {
                        Some(at) if place.is_none_or(|known| known == (root, at)) => {
                            place = Some((root, at));
                        }
                        _ => {
                            placed = false;
                            break;
                        }
                    },
                }
            }
            let keep = placed && (guess.is_none() || place.is_none() || guess == place);
            if !keep {
                candidates.remove(phi);
                changed = true;
            } else if guess.is_none() && place.is_some() {
                candidates.insert(*phi, place);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    candidates
        .into_iter()
        .filter_map(|(phi, place)| Some((phi, place?)))
        .collect()
}

/// Calls `f` with every used value and the site using it.
///
/// # Arguments
///
/// * `ssa` - The function to walk.
/// * `f` - Visitor receiving `(value, site)` per operand.
fn visit_uses<T: Target>(ssa: &SsaFunction<T>, mut f: impl FnMut(SsaVarId, UseSite)) {
    for (block, body) in ssa.iter_blocks() {
        for (phi, node) in body.phi_nodes().iter().enumerate() {
            for operand in node.operands() {
                f(operand.value(), UseSite::Phi { block, phi });
            }
        }
        for (instr, instruction) in body.instructions().iter().enumerate() {
            instruction
                .op()
                .for_each_use(|value| f(value, UseSite::Instruction { block, instr }));
        }
    }
}
