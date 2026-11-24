//! The allocator models: what a routine making, resizing, duplicating or
//! ending a heap object does, read from its arguments by position.
//!
//! Every allocator belongs to a family the host names ([`AllocatorFamily`]),
//! and an object remembers the family that made it, so a deallocator of
//! another family ends nothing. Each model states its success and its failure
//! as separate alternatives of the call: an allocator that cannot allocate
//! returns a null pointer, raises, or returns a status, as its routine does.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    analysis::execution::{
        ModelCall, ModelControl,
        memory::{Allocator, AllocatorFamily, BlockSize, FreeOutcome},
        provenance::Pointer,
        value::{Address, Root, Value},
        writeguard::ProtectionEncoding,
    },
    ir::value::ConstValue,
    target::Target,
};

/// Where a model reads one of its values: an argument, or the pointer-width
/// cell an argument points at.
///
/// `malloc(n)` reads its size from argument 0; `NtAllocateVirtualMemory`
/// reads its size from the `SIZE_T` its fourth argument points at, and writes
/// the size it grants back there. Written as the argument's position, or as
/// `{"through": position}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(untagged))]
pub enum Operand {
    /// The argument at this position.
    Argument(usize),
    /// The pointer-width cell the argument at this position points at.
    Through {
        /// The argument pointing at the cell.
        through: usize,
    },
}

impl Operand {
    /// The argument the operand reads, or whose cell it reads.
    #[must_use]
    pub const fn argument(self) -> usize {
        match self {
            Self::Argument(index) | Self::Through { through: index } => index,
        }
    }

    /// The operand as an unsigned count at the pointer width, when it holds a
    /// known integer.
    pub fn count<T: Target>(self, call: &mut ModelCall<'_, T>) -> Option<u128> {
        match self {
            Self::Argument(index) => call.count(index),
            Self::Through { through } => call.cell(through)?.native_count(call.ptr_size()),
        }
    }

    /// The operand as an address, when it holds one; a null pointer is the
    /// constant zero, which names no address.
    pub fn address<T: Target>(self, call: &mut ModelCall<'_, T>) -> Option<Address> {
        match self {
            Self::Argument(index) => call.address(index),
            Self::Through { through } => {
                let value = call.cell(through)?;
                call.access.address_of(&value, None)
            }
        }
    }

    /// The heap the operand names: the block its handle is, where the
    /// execution made it (`HeapCreate`'s); `None` for a handle it cannot
    /// name, as the process heap's.
    pub fn heap<T: Target>(self, call: &mut ModelCall<'_, T>) -> Option<Root> {
        self.address(call)
            .filter(|handle| matches!(handle.root(), Root::Heap(_)))
            .map(|handle| handle.root())
    }

    /// The cell a [`Self::Through`] operand reads, which a routine writing it
    /// back writes; `None` for an argument.
    pub fn cell<T: Target>(self, call: &ModelCall<'_, T>) -> Option<Pointer> {
        match self {
            Self::Argument(_) => None,
            Self::Through { through } => call.pointer(through),
        }
    }
}

/// What one of a routine's values must be for the model to state what the
/// call does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Test {
    /// Equal to this integer at the pointer width: `VirtualAlloc`'s address
    /// null, `NtAllocateVirtualMemory`'s process the current one's handle,
    /// `-1`.
    Equals(i64),
    /// Every one of these bits set: `MEM_COMMIT`.
    AllSet(u64),
    /// None of these bits set: `MAP_FIXED`.
    NoneSet(u64),
}

/// One requirement a model states of a call: the value `operand` reads must
/// pass `test`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Condition {
    /// Where the value is read.
    pub operand: Operand,
    /// What it must be.
    pub test: Test,
}

impl Condition {
    /// Whether the call's value passes the test; `false` where it is
    /// unknown.
    fn holds<T: Target>(self, call: &mut ModelCall<'_, T>) -> bool {
        let Some(value) = self.operand.count(call) else {
            return false;
        };
        let mask = call.ptr_size().mask_unsigned_128(u128::MAX);
        let value = value & mask;
        match self.test {
            Test::Equals(expected) => value == (expected as i128 as u128) & mask,
            Test::AllSet(bits) => value & u128::from(bits) == u128::from(bits),
            Test::NoneSet(bits) => value & u128::from(bits) == 0,
        }
    }
}

/// The requirements a model states of a call, at most [`Conditions::CAPACITY`]
/// of them, all of which a call must meet for the model to run; a call that
/// fails one, or whose value is unknown, does what no model here states.
/// Written as a list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(into = "Vec<Condition>", try_from = "Vec<Condition>")
)]
pub struct Conditions([Option<Condition>; Conditions::CAPACITY]);

impl Conditions {
    /// The most requirements one model states.
    pub const CAPACITY: usize = 4;

    /// No requirement.
    pub const NONE: Self = Self([None; Self::CAPACITY]);

    /// These requirements; `None` where there are more than
    /// [`Self::CAPACITY`].
    #[must_use]
    pub fn of<const N: usize>(conditions: [Condition; N]) -> Option<Self> {
        Self::try_from(Vec::from(conditions)).ok()
    }

    /// Whether there is no requirement.
    #[must_use]
    pub fn is_none(&self) -> bool {
        self.0.iter().all(Option::is_none)
    }

    /// The requirements, in order.
    pub fn iter(&self) -> impl Iterator<Item = Condition> + '_ {
        self.0.iter().flatten().copied()
    }

    /// Whether a call meets every requirement.
    pub fn hold<T: Target>(&self, call: &mut ModelCall<'_, T>) -> bool {
        self.iter().all(|condition| condition.holds(call))
    }
}

impl TryFrom<Vec<Condition>> for Conditions {
    type Error = &'static str;

    /// The requirements a list states, refused past [`Conditions::CAPACITY`].
    fn try_from(conditions: Vec<Condition>) -> Result<Self, Self::Error> {
        if conditions.len() > Self::CAPACITY {
            return Err("a model states at most four requirements");
        }
        let mut held = [None; Self::CAPACITY];
        for (slot, condition) in held.iter_mut().zip(conditions) {
            *slot = Some(condition);
        }
        Ok(Self(held))
    }
}

impl From<Conditions> for Vec<Condition> {
    /// The requirements, in order.
    fn from(conditions: Conditions) -> Self {
        conditions.iter().collect()
    }
}

/// The pages an allocator grants: a size rounds up to whole pages of `size`
/// bytes, every one of which the block holds; exactly that where the
/// platform's page size is fixed (`exact`), and at least that otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pages {
    /// The smallest page size, a power of two.
    pub size: u64,
    /// Whether the platform's page size is always `size`.
    pub exact: bool,
}

impl Pages {
    /// The block `bytes` requested makes; `None` where rounding overflows.
    fn block(self, bytes: u64) -> Option<BlockSize> {
        let mask = self.size.checked_sub(1)?;
        let granted = bytes.checked_add(mask)? & !mask;
        Some(if self.exact {
            BlockSize::exact(granted)
        } else {
            BlockSize::at_least(granted)
        })
    }
}

/// The protection an allocator installs on what it makes, read from the
/// operand the routine takes it in, spelled as its operating system does
/// (`VirtualAlloc`'s `flProtect`, `mmap`'s `prot`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AllocatedProtection {
    /// Where the protection is read.
    pub operand: Operand,
    /// How it spells an access.
    pub encoding: ProtectionEncoding,
}

/// A flags argument an allocator reads, and what its bits do.
///
/// `HeapAlloc(heap, flags, n)` zeroes its bytes for `HEAP_ZERO_MEMORY` and
/// raises rather than returning null for `HEAP_GENERATE_EXCEPTIONS`;
/// `LocalAlloc(flags, n)` zeroes for `LMEM_ZEROINIT` and returns a movable
/// handle rather than a pointer for `LMEM_MOVEABLE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AllocationFlags {
    /// Which argument holds the flags.
    pub argument: usize,
    /// The bits that zero the bytes the call allocates.
    pub zeroing: u64,
    /// The bits under which the routine does what no model here states: a
    /// call passing any of them is unmodeled.
    pub unmodeled: u64,
}

impl AllocationFlags {
    /// Whether the call's flags zero what it allocates, or `None` where they
    /// are no constant or ask for what no model states.
    fn zeroing<T: Target>(self, call: &ModelCall<'_, T>) -> Option<bool> {
        let flags = u64::try_from(call.count(self.argument)?).ok()?;
        (flags & self.unmodeled == 0).then_some(flags & self.zeroing != 0)
    }
}

/// How an allocator hands over the object it makes, and what it does when it
/// cannot make one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Delivery {
    /// Returns the object, or a null pointer (`malloc`, `operator new` with
    /// `nothrow`).
    Returned,
    /// Returns the object, or raises (`operator new`).
    ReturnedOrRaises,
    /// Returns the object, or ends the program without raising
    /// (`__cxa_allocate_exception`, which calls `std::terminate`).
    ReturnedOrEnds,
    /// Returns the object, or this integer at the pointer width, as a
    /// pointer (`mmap`'s `MAP_FAILED`, minus one).
    ReturnedOr {
        /// The value a failed call returns.
        failure: i64,
    },
    /// Writes the object to the pointer cell argument `out` points at and
    /// returns a signed 32-bit zero, or returns `failure` and writes nothing
    /// (`posix_memalign`, whose failure is `ENOMEM`).
    Out {
        /// Which argument points at the cell receiving the object.
        out: usize,
        /// The status a failed call returns.
        failure: i32,
    },
}

impl Delivery {
    /// Whether the cell an [`Self::Out`] delivery writes is named, so the
    /// call can be modeled at all.
    fn names_its_cell<T: Target>(self, call: &ModelCall<'_, T>) -> bool {
        match self {
            Self::Returned
            | Self::ReturnedOrRaises
            | Self::ReturnedOrEnds
            | Self::ReturnedOr { .. } => true,
            Self::Out { out, .. } => call.address(out).is_some(),
        }
    }

    /// Hands over `object` as a successful call does.
    fn deliver<T: Target>(self, call: &mut ModelCall<'_, T>, object: Address) {
        match self {
            Self::Returned
            | Self::ReturnedOrRaises
            | Self::ReturnedOrEnds
            | Self::ReturnedOr { .. } => {
                call.return_value(Value::Addr(object));
            }
            Self::Out { out, .. } => {
                let Some(cell) = call.pointer(out) else {
                    call.set_control(ModelControl::Unsupported);
                    return;
                };
                call.write_address(cell, object);
                call.return_value(Value::Const(ConstValue::I32(0)));
            }
        }
    }

    /// Ends the call as an allocator that cannot allocate does.
    fn fail<T: Target>(self, call: &mut ModelCall<'_, T>) {
        match self {
            Self::Returned => {
                call.return_value(Value::Const(ConstValue::native(call.ptr_size(), 0)))
            }
            Self::ReturnedOr { failure } => {
                call.return_value(Value::Const(ConstValue::native(call.ptr_size(), failure)))
            }
            Self::ReturnedOrRaises => call.set_control(ModelControl::Trap),
            Self::ReturnedOrEnds => call.set_control(ModelControl::NoReturn),
            Self::Out { failure, .. } => {
                call.return_value(Value::Const(ConstValue::I32(failure)));
            }
        }
    }
}

/// An allocator: `malloc(n)`, `calloc(count, n)`, `aligned_alloc(alignment,
/// n)`, `posix_memalign(&p, alignment, n)`, `HeapAlloc(heap, flags, n)`,
/// `operator new(n)`, `VirtualAlloc(NULL, n, MEM_COMMIT, PAGE_READWRITE)`,
/// `NtAllocateVirtualMemory(-1, &p, 0, &n, MEM_COMMIT, PAGE_READWRITE)`.
///
/// Its object holds the size's bytes, times the count argument's where there
/// is one; a product that overflows the pointer width fails the call. A page
/// allocator rounds the size up to whole pages ([`Pages`]), and a size read
/// through a cell is written back there as the size granted. Its bytes are
/// zero where the allocator zeroes them, always or for a flag, and unknown
/// otherwise. An alignment argument must be a known power of two, as every
/// such routine requires of a call that succeeds. A call failing one of the
/// model's requirements ([`Conditions`]: an address to place the block at, a
/// process other than the current one) is unmodeled, and so is one installing
/// an executable protection, whose block holds code the run does not yet
/// follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Allocation {
    /// The family the object belongs to.
    pub family: AllocatorFamily,
    /// Where the size is read, in bytes or of each element.
    pub size: Operand,
    /// Which argument holds the element count, when there is one.
    pub count: Option<usize>,
    /// Which argument holds the alignment, when there is one.
    pub alignment: Option<usize>,
    /// Whether every object it makes is zero (`calloc`).
    pub zeroed: bool,
    /// The flags argument it reads, when it reads one.
    pub flags: Option<AllocationFlags>,
    /// How it hands the object over, and fails.
    pub delivery: Delivery,
    /// What a call must state for the model to run.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Conditions::is_none")
    )]
    pub requires: Conditions,
    /// The pages it grants, for a page allocator.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub pages: Option<Pages>,
    /// The protection it installs, where it takes one.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub protection: Option<AllocatedProtection>,
    /// The heap handle it allocates in, where it takes one (`HeapAlloc`'s
    /// first argument): its block belongs to that heap.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub heap: Option<Operand>,
}

impl Allocation {
    /// Makes the object, or ends the call unmodeled where its arguments do
    /// not say what the routine does.
    pub fn allocate<T: Target>(self, call: &mut ModelCall<'_, T>) {
        let Some(zeroed) = self.admits(call) else {
            call.set_control(ModelControl::Unsupported);
            return;
        };
        let size = match self.bytes(call) {
            Some(Some(bytes)) => match self.pages {
                Some(pages) => match pages.block(bytes) {
                    Some(block) => block,
                    None => {
                        self.delivery.fail(call);
                        return;
                    }
                },
                None => BlockSize::exact(bytes),
            },
            Some(None) => {
                self.delivery.fail(call);
                return;
            }
            None => BlockSize::UNSTATED,
        };
        let allocator = Allocator {
            family: Some(self.family),
            heap: self.heap.and_then(|heap| heap.heap(call)),
        };
        let object = call.allocate_block(size, zeroed, allocator);
        self.delivery.deliver(call, object);
        // A size read through a cell is the routine's to update: it writes
        // back the size it granted, whole pages for a page allocator.
        if let Some(cell) = self.size.cell(call) {
            match size.extent.filter(|_| size.guaranteed > 0) {
                Some(granted) => call.write_count(cell, granted),
                None => call.write_unknown(cell, call.ptr_size().bytes()),
            }
        }
    }

    /// Fails the call as the routine does when it cannot allocate, or ends it
    /// unmodeled where its arguments do not say what the routine does.
    pub fn fail<T: Target>(self, call: &mut ModelCall<'_, T>) {
        if self.admits(call).is_none() {
            call.set_control(ModelControl::Unsupported);
            return;
        }
        self.delivery.fail(call);
    }

    /// Whether the model states what the call does, and if so whether its
    /// object is zero: the call meets every requirement, installs no
    /// executable protection, passes flags and an alignment the model reads,
    /// and names the cell an out delivery writes.
    fn admits<T: Target>(self, call: &mut ModelCall<'_, T>) -> Option<bool> {
        if !self.requires.hold(call) || !self.aligned(call) || !self.delivery.names_its_cell(call) {
            return None;
        }
        if let Some(protection) = self.protection {
            let installed = protection
                .operand
                .count(call)
                .and_then(|value| u64::try_from(value).ok())
                .and_then(|value| protection.encoding.decode(value))?;
            if installed.executable {
                return None;
            }
        }
        self.zeroing(call)
    }

    /// Whether the object is zero, or `None` where the flags leave the call
    /// unmodeled.
    fn zeroing<T: Target>(self, call: &ModelCall<'_, T>) -> Option<bool> {
        match self.flags {
            Some(flags) => Some(self.zeroed || flags.zeroing(call)?),
            None => Some(self.zeroed),
        }
    }

    /// Whether the alignment argument, where there is one, is a known power
    /// of two.
    fn aligned<T: Target>(self, call: &ModelCall<'_, T>) -> bool {
        self.alignment
            .is_none_or(|index| call.count(index).is_some_and(u128::is_power_of_two))
    }

    /// The object's size: `None` where an argument is unknown, and
    /// `Some(None)` where the product overflows the pointer width, which
    /// fails the call.
    fn bytes<T: Target>(self, call: &mut ModelCall<'_, T>) -> Option<Option<u64>> {
        let size = self.size.count(call)?;
        let count = match self.count {
            Some(index) => call.count(index)?,
            None => 1,
        };
        Some(
            size.checked_mul(count)
                .filter(|bytes| call.ptr_size().mask_unsigned_128(*bytes) == *bytes)
                .map(|bytes| u64::try_from(bytes).unwrap_or(u64::MAX)),
        )
    }
}

/// A reallocator: `realloc(pointer, n)`, `reallocarray(pointer, count, n)`,
/// `HeapReAlloc(heap, flags, pointer, n)`.
///
/// On success the old object ends and a new one holds its bytes up to the
/// smaller of the two sizes, every byte past them unknown; on failure the
/// call returns a null pointer and the old object survives unchanged. A null
/// pointer allocates afresh where the routine says so, as `realloc` does. A
/// size of zero for a live object does what each C library chooses, so is
/// unmodeled; so is a pointer to no live object of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Reallocation {
    /// The family the old and the new object belong to.
    pub family: AllocatorFamily,
    /// Which argument holds the old object.
    pub pointer: usize,
    /// Which argument holds the new size, in bytes or of each element.
    pub size: usize,
    /// Which argument holds the element count, when there is one.
    pub count: Option<usize>,
    /// The flags argument it reads, when it reads one; its zeroing bits are
    /// not read, since the bytes it zeroes past the old object's are unknown
    /// here (a joined object's extent is the least its paths hold).
    pub flags: Option<AllocationFlags>,
    /// Whether a null old pointer makes a new object, as `realloc` does.
    pub null_allocates: bool,
    /// The heap handle it reallocates in, where it takes one
    /// (`HeapReAlloc`'s first argument): the old block must be that heap's,
    /// and the new one is.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub heap: Option<Operand>,
}

/// What a reallocation's old pointer names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Old {
    /// A null pointer, which allocates afresh.
    Null,
    /// A live object of the family, at its start, and the bytes it holds.
    Object(Address, u64),
    /// Anything else: an address the execution cannot name, which may be
    /// any object of the family, or one no allocator defines, which the
    /// deallocation judges ([`ModelCall::deallocate`]).
    Unnamed,
}

impl Reallocation {
    /// What the old pointer names; ends the call unmodeled and returns
    /// `None` where the arguments do not say what the routine does.
    fn old<T: Target>(self, call: &mut ModelCall<'_, T>) -> Option<Old> {
        let null = call
            .argument(self.pointer)
            .and_then(Value::constant)
            .is_some_and(ConstValue::is_zero);
        let old = if self
            .flags
            .is_some_and(|flags| flags.zeroing(call).is_none())
        {
            None
        } else if null {
            self.null_allocates.then_some(Old::Null)
        } else {
            Some(
                call.address(self.pointer)
                    .and_then(|old| {
                        let (size, _) = call.object(old)?;
                        let made = call.made_by(old)?;
                        self.allocator(call)
                            .admits(made.family, made.heap)
                            .then_some(Old::Object(old, size))
                    })
                    .unwrap_or(Old::Unnamed),
            )
        };
        if old.is_none() {
            call.set_control(ModelControl::Unsupported);
        }
        old
    }

    /// Moves the object to one of the new size. An old pointer the
    /// execution cannot name is deallocated as `free` deallocates it, and
    /// every byte of the new object may come from it: written, unknown.
    pub fn reallocate<T: Target>(self, call: &mut ModelCall<'_, T>) {
        let Some(old) = self.old(call) else {
            return;
        };
        let size = match self.as_allocation().bytes(call) {
            Some(Some(0)) if old != Old::Null => {
                call.set_control(ModelControl::Unsupported);
                return;
            }
            Some(Some(size)) => Some(size),
            Some(None) => {
                call.return_value(Value::Const(ConstValue::native(call.ptr_size(), 0)));
                return;
            }
            None => None,
        };
        let allocator = self.allocator(call);
        // The old object ends before the new one is made: one the execution
        // cannot name may be any live object, and the new one is none.
        if old == Old::Unnamed && call.deallocate(self.pointer, allocator) == FreeOutcome::Unknown {
            call.invalidate_lifetimes();
        }
        let object = call.allocate_block(size, false, allocator);
        match old {
            Old::Null => {}
            Old::Object(old, held) => {
                // A new size the call does not state keeps no byte known.
                let kept = usize::try_from(held.min(size.unwrap_or(0))).unwrap_or(usize::MAX);
                if kept > 0 && call.check_write_extent(kept) {
                    call.copy(object, self.pointer, kept);
                }
                call.free(old, allocator);
            }
            Old::Unnamed => {
                if let Some(len) = size.and_then(|size| usize::try_from(size).ok())
                    && len > 0
                    && call.check_write_extent(len)
                {
                    call.write_unknown(object, len);
                }
            }
        }
        call.return_value(Value::Addr(object));
    }

    /// Fails the call, leaving the old object as it was.
    pub fn fail<T: Target>(self, call: &mut ModelCall<'_, T>) {
        if self.old(call).is_some() {
            call.return_value(Value::Const(ConstValue::native(call.ptr_size(), 0)));
        }
    }

    /// The allocator its blocks belong to: its family, in the heap its handle
    /// names where it takes one.
    fn allocator<T: Target>(self, call: &mut ModelCall<'_, T>) -> Allocator {
        Allocator {
            family: Some(self.family),
            heap: self.heap.and_then(|heap| heap.heap(call)),
        }
    }

    /// The allocation its size and count arguments describe.
    const fn as_allocation(self) -> Allocation {
        Allocation {
            family: self.family,
            size: Operand::Argument(self.size),
            count: self.count,
            alignment: None,
            zeroed: false,
            flags: None,
            delivery: Delivery::Returned,
            requires: Conditions::NONE,
            pages: None,
            protection: None,
            heap: None,
        }
    }
}

/// A string duplicator: `strdup(source)`, `strndup(source, limit)`.
///
/// The new object holds the string's bytes, at most `limit` of them where
/// there is a limit, and a terminator; a string the call cannot read makes
/// an object of unknown size. On failure the call returns a null pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Duplication {
    /// The family the copy belongs to.
    pub family: AllocatorFamily,
    /// Which argument holds the string.
    pub source: usize,
    /// Which argument holds the most bytes copied, when there is one.
    pub limit: Option<usize>,
}

impl Duplication {
    /// Copies the string into a new object.
    pub fn duplicate<T: Target>(self, call: &mut ModelCall<'_, T>) {
        if self.limit.is_some_and(|index| call.count(index).is_none()) {
            call.set_control(ModelControl::Unsupported);
            return;
        }
        let string = self.string(call);
        let size = string
            .as_ref()
            .map(|bytes| u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        let object = call.allocate_block(size, false, Some(self.family));
        if let Some(bytes) = string {
            call.write(object, &bytes);
        }
        call.return_value(Value::Addr(object));
    }

    /// The bytes the copy holds, its terminator with them, when the call
    /// can read them.
    fn string<T: Target>(self, call: &mut ModelCall<'_, T>) -> Option<Vec<u8>> {
        let source = call.pointer(self.source)?;
        let limit = match self.limit {
            Some(index) => Some(usize::try_from(call.count(index)?).ok()?),
            None => None,
        };
        let mut bytes = match limit {
            None => call.read_string(source, call.access.string_limit())?,
            Some(limit) => match call.string_length(source, limit.checked_add(1)?) {
                Some(length) => call.read(source, length)?,
                None => call.read(source, limit)?,
            },
        };
        bytes.push(0);
        Some(bytes)
    }
}

/// What a deallocator returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FreeResult {
    /// Nothing (`free`, `operator delete`).
    Nothing,
    /// One, a Windows `BOOL` (`HeapFree`).
    One,
    /// A null pointer (`LocalFree`, `GlobalFree`).
    Null,
    /// A signed 32-bit zero, a success status (`munmap`'s `0`,
    /// `NtFreeVirtualMemory`'s `STATUS_SUCCESS`).
    Zero,
}

/// A deallocator: `free(pointer)`, `free_sized(pointer, n)`,
/// `operator delete(pointer, n)`, `HeapFree(heap, flags, pointer)`,
/// `LocalFree(pointer)`, `VirtualFree(pointer, 0, MEM_RELEASE)`,
/// `munmap(pointer, n)`, `NtFreeVirtualMemory(-1, &pointer, &n, MEM_RELEASE)`.
///
/// It ends the lifetime of the object of its family the pointer names; a
/// null pointer does nothing. A pointer to no live object of the family,
/// another family's object included, leaves the call unmodeled, and an
/// unknown one may end any object. A size argument is read by the
/// deallocators that take one but is not checked against the object here; a
/// routine releasing an extent (`munmap`) releases the whole object or the
/// call is unmodeled, and one reporting the extent it released
/// (`NtFreeVirtualMemory`) writes the object's size back through its cell. A
/// call failing one of the model's requirements ([`Conditions`]) is
/// unmodeled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Deallocation {
    /// The family whose objects it ends.
    pub family: AllocatorFamily,
    /// Where the object is read.
    pub pointer: Operand,
    /// Which argument holds the object's size, when it takes one.
    pub size: Option<usize>,
    /// What it returns.
    pub result: FreeResult,
    /// What a call must state for the model to run.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Conditions::is_none")
    )]
    pub requires: Conditions,
    /// The extent a call releases, which must cover the whole object, in
    /// the pages it is granted in where it has them.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub whole: Option<(Operand, Option<Pages>)>,
    /// The cell the routine writes the released object's size back to.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub released: Option<Operand>,
    /// The heap handle it releases in, where it takes one (`HeapFree`'s
    /// first argument): another heap's block is another allocator's.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub heap: Option<Operand>,
}

impl Deallocation {
    /// Ends the object's lifetime.
    ///
    /// What the pointer names decides the effect ([`FreeOutcome`]): the
    /// start of a live object of the family ends, a null pointer does
    /// nothing, and an address the execution cannot name may end any
    /// object. A deallocation no allocator defines (an object already ended,
    /// an interior pointer, the stack or the image, another family's object)
    /// is a finding ([`FreeOutcome::fault`]) and ends nothing; the call
    /// returns as any other.
    pub fn free<T: Target>(self, call: &mut ModelCall<'_, T>) {
        if !self.requires.hold(call) {
            call.set_control(ModelControl::Unsupported);
            return;
        }
        let object = self.pointer.address(call);
        let held = object.and_then(|address| call.object(address));
        if let Some((operand, pages)) = self.whole {
            let covers = held
                .zip(operand.count(call))
                .is_some_and(|((size, _), len)| {
                    let len = u64::try_from(len).unwrap_or(u64::MAX);
                    let len = pages
                        .and_then(|pages| pages.block(len))
                        .map_or(len, |block| block.guaranteed);
                    len >= size
                });
            if !covers {
                call.set_control(ModelControl::Unsupported);
                return;
            }
        }
        let allocator = Allocator {
            family: Some(self.family),
            heap: self.heap.and_then(|heap| heap.heap(call)),
        };
        let outcome = match self.pointer {
            Operand::Argument(index) => call.deallocate(index, allocator),
            Operand::Through { .. } => match object {
                Some(address) => call.free(address, allocator),
                None => {
                    call.set_control(ModelControl::Unsupported);
                    return;
                }
            },
        };
        if outcome == FreeOutcome::Unknown {
            call.invalidate_lifetimes();
        }
        if let Some(cell) = self.released.and_then(|operand| operand.cell(call)) {
            match held {
                Some((size, _)) => call.write_count(cell, size),
                None => call.write_unknown(cell, call.ptr_size().bytes()),
            }
        }
        match self.result {
            FreeResult::Nothing => {}
            FreeResult::One => call.return_value(Value::Const(ConstValue::I32(1))),
            FreeResult::Null => {
                call.return_value(Value::Const(ConstValue::native(call.ptr_size(), 0)))
            }
            FreeResult::Zero => call.return_value(Value::Const(ConstValue::I32(0))),
        }
    }
}
