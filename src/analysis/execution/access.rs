//! Loads and stores: moving values between what static execution knows about
//! a value and the bytes it knows in memory.
//!
//! A value is written as its bytes in the target's byte order, or, for an
//! address, as the pieces [`Memory`] keeps. A load reads the bytes back as a
//! constant of the loaded type, or the address stored whole where it reads.
//! Where the pointer, the width or the byte order is not known, a load is
//! unknown and a store forgets what it may have overwritten.

use std::num::NonZeroU16;

use crate::{
    analysis::{
        budget::{Budget, Exhaustion},
        consts::MisalignedAccess,
        context::AnalysisContext,
        execution::{
            MemoryWriteGuard,
            facts::{
                ByteBound, ExtentSource, ObjectKind, ReplacementProof, StorageObject, UseBound,
            },
            memory::{CopyOrder, Loaded, Memory, MemoryRefusal, StringScan, Transfer},
            state::State,
            trace::Trace,
            types::OperationLayout,
            value::{Address, Root, Value},
            writeguard::check_write,
        },
        image::Mutability,
    },
    ir::{
        ops::{AccessPermission, AccessQualifiers},
        value::ConstValue,
    },
    pointer::PointerSize,
    target::{Endianness, StorageLayout, Target, TypeLayout},
};

/// Distinguishes an unavailable memory contract from a configured operation limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessError {
    /// A required memory contract is unavailable.
    Refused(MemoryRefusal),
    /// A concrete operation extent exceeds its configured limit.
    Exhausted(Exhaustion),
}

impl From<MemoryRefusal> for AccessError {
    fn from(reason: MemoryRefusal) -> Self {
        Self::Refused(reason)
    }
}

impl From<Exhaustion> for AccessError {
    fn from(exhaustion: Exhaustion) -> Self {
        Self::Exhausted(exhaustion)
    }
}

/// The write family processed by an instruction's footprint preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePreflight {
    /// This preflight does not own the instruction's write family.
    Unrelated,
    /// The instruction is proven to perform no write.
    Empty,
    /// A possible write was processed by its configured admission policy.
    Possible,
}

/// Loads and stores against one host's image and target.
#[derive(Debug, Clone, Copy)]
pub struct Access<'a> {
    /// The pointer width, the byte order and the image.
    context: &'a AnalysisContext<'a>,
    /// Lazy target query shared with proofs for the current memory instruction.
    operation: Option<OperationLayout<'a>>,
    /// Optional constraint shared by all writes of this root trial.
    write_guard: Option<&'a dyn MemoryWriteGuard>,
}

/// A checked scalar read and the source footprint used to obtain it.
#[derive(Debug)]
pub struct ReadValue<T: Target> {
    /// The decoded value, or unknown when no supported representation is available.
    pub value: Value<T>,
    /// The exact checked source span, absent when its address or layout is unresolved.
    pub footprint: Option<(Address, usize)>,
}

impl<'a> Access<'a> {
    /// Accesses memory as `context`'s target lays it out.
    pub const fn new(context: &'a AnalysisContext<'a>) -> Self {
        Self {
            context,

            operation: None,
            write_guard: None,
        }
    }

    /// Applies the root trial's known-write constraint to every access route.
    pub const fn with_write_guard(mut self, guard: Option<&'a dyn MemoryWriteGuard>) -> Self {
        self.write_guard = guard;
        self
    }

    /// Shares the current instruction's target layout with its load or store proof.
    pub const fn with_operation<'op>(self, operation: OperationLayout<'op>) -> Access<'op>
    where
        'a: 'op,
    {
        Access {
            context: self.context,

            operation: Some(operation),
            write_guard: self.write_guard,
        }
    }

    /// Returns whether this execution has a configured write-admission policy.
    pub const fn has_write_guard(self) -> bool {
        self.write_guard.is_some()
    }

    /// Borrows the policy that validates modeled mapping transitions.
    pub const fn write_guard(self) -> Option<&'a dyn MemoryWriteGuard> {
        self.write_guard
    }

    /// Reads exact physical bits from a numeric or mapped module address.
    pub fn physical_address_bits(self, address: Address) -> Option<u64> {
        let width = self.ptr_size();
        if address.pointer_size() != width {
            return None;
        }
        let offset = address.offset().cast_unsigned();
        match address.root() {
            Root::Number => Some(offset),
            Root::Symbol => {
                let shift = self.context.image()?.number_offset()?;
                Some(width.mask_unsigned(offset.wrapping_sub(shift.cast_unsigned())))
            }
            _ => None,
        }
    }

    /// Consults the configured policy before losing an unresolved write footprint.
    pub fn admit_unresolved_write(
        self,
        memory: &mut Memory,
        pointer: Option<Address>,
        len: Option<usize>,
    ) -> bool {
        if len == Some(0) {
            return true;
        }
        if let Some(reason) = self
            .write_guard
            .and_then(|guard| guard.check_unresolved_write(pointer, len).err())
        {
            memory.refuse(reason);
            false
        } else {
            true
        }
    }

    /// Consults the configured policy before a routine that writes no code
    /// loses the footprint of what it writes of escaped data
    /// ([`MemoryWriteGuard::check_unresolved_data_write`]).
    pub fn admit_unresolved_data_write(self, memory: &mut Memory) -> bool {
        if let Some(reason) = self
            .write_guard
            .and_then(|guard| guard.check_unresolved_data_write().err())
        {
            memory.refuse(reason);
            false
        } else {
            true
        }
    }

    /// The bytes from `address` to the end of the object it lies in, where
    /// something bounds that end: a live allocation's size, an object the
    /// host states ([`Memory::stated_object`]), or the end of the image
    /// region holding it. `None` where nothing does.
    #[must_use]
    pub fn rest_of_object(self, memory: &Memory, address: Address) -> Option<usize> {
        let end = if matches!(address.root(), Root::Heap(_)) {
            let (size, _) = memory.object(address.root())?;
            i64::try_from(size).ok()?
        } else if let Some(object) = memory.stated_object(address) {
            object
                .start
                .checked_add(i64::try_from(object.bytes).ok()?)?
        } else {
            let at = address.image_address()?;
            let region = self.context.image()?.region(at)?;
            return usize::try_from(region.end.checked_sub(at)?).ok();
        };
        usize::try_from(end.checked_sub(address.offset())?).ok()
    }

    /// Checks block-write admission before an unsupported operation forgets
    /// memory. A write `certain` to happen in a known direction that may not
    /// be written faults ([`Memory::deny_write`]); one that may not happen
    /// (a lane an unknown mask may leave alone, a direction the operation
    /// may not take) refuses the execution instead.
    pub fn preflight_block_write<T: Target>(
        self,
        memory: &mut Memory,
        pointer: &Value<T>,
        len: usize,
        element_bytes: usize,
        reverse: Option<bool>,
        certain: bool,
    ) -> Result<(), MemoryRefusal> {
        if len == 0 {
            return Ok(());
        }
        let Some(first) = self.address_of(pointer, None) else {
            return if self.admit_unresolved_write(memory, None, Some(len)) {
                Ok(())
            } else {
                Err(memory.refusal().unwrap_or(MemoryRefusal::UnresolvedWrite))
            };
        };
        for direction in [false, true] {
            if reverse.is_some_and(|actual| direction != actual) {
                continue;
            }
            let start = if direction {
                CopyOrder::Stream {
                    element_bytes,
                    reverse: true,
                }
                .span(first, len)?
            } else {
                first
            };
            let admitted = if certain && reverse.is_some() {
                self.admit_write(memory, start, len)
            } else {
                self.admit_possible_write(memory, start, len)
            };
            if !admitted {
                return Err(memory.refusal().unwrap_or(MemoryRefusal::InstructionSource));
            }
        }
        Ok(())
    }

    /// Returns the configured maximum materialized buffer size.
    pub fn buffer_limit(self) -> usize {
        self.context.budget().limit(Budget::ExecutionBufferBytes)
    }

    /// Returns the configured maximum library string scan.
    pub fn string_limit(self) -> usize {
        self.context.budget().limit(Budget::ExecutionStringBytes)
    }

    /// Returns the configured byte extent of one native block-memory operation.
    pub fn block_limit(self) -> usize {
        self.context.budget().limit(Budget::ExecutionBlockBytes)
    }

    /// Returns the target's attested byte order.
    pub fn byte_order(self) -> Option<Endianness> {
        self.context.machine().byte_order
    }

    /// The pointer width.
    pub fn ptr_size(self) -> PointerSize {
        self.context.machine().ptr_size
    }

    /// The known bytes at `address`, up to `limit` of them, stopping at the
    /// first that is not known; never-written bytes read from the image.
    pub fn known_run(self, memory: &Memory, address: Address, limit: usize) -> Vec<u8> {
        memory.known_run(
            address,
            limit.min(self.buffer_limit()),
            self.context.image(),
        )
    }

    /// The known bytes at `address`, up to `maximum` of them and within the
    /// tightest proven extent, stopping at the first that is not known, and
    /// that extent: the use, reaching `use_bound` bytes as it states; the
    /// live heap allocation the address lies in; or `stated`, an object the
    /// host proves. Without one the bytes are a prefix ([`ByteBound::Prefix`]).
    pub fn bounded_run(
        self,
        memory: &Memory,
        address: Address,
        maximum: usize,
        use_bound: Option<(u64, UseBound)>,
        stated: Option<StorageObject>,
    ) -> (Vec<u8>, ByteBound) {
        let object = match address.root() {
            root @ Root::Heap(_) => memory.object(root).map(|(bytes, _)| StorageObject {
                start: 0,
                bytes,
                kind: ObjectKind::Heap,
                source: ExtentSource::Allocator,
            }),
            _ => stated,
        };
        let (bound, reach) = ByteBound::tightest(use_bound, object, address);
        let limit = reach.map_or(maximum, |reach| {
            usize::try_from(reach).map_or(maximum, |reach| reach.min(maximum))
        });
        let bytes = self.known_run(memory, address, address.linear_prefix_len(limit));
        (bytes, bound)
    }

    /// The address `pointer` holds and the known bytes from there on, bounded
    /// as a call argument's are and within the object it lies in (`stated`,
    /// for the address, where the host proves one; [`Self::bounded_run`]),
    /// where its first byte is one the run wrote: a pointer into bytes the
    /// image holds unchanged shows nothing new.
    pub fn written_prefix<T: Target>(
        self,
        memory: &Memory,
        pointer: &Value<T>,
        stated: impl FnOnce(Address) -> Option<StorageObject>,
    ) -> Option<(Address, Vec<u8>, ByteBound)> {
        let address = self.address_of(pointer, None)?;
        if memory.known_run(address, 1, None).is_empty() {
            return None;
        }
        let (bytes, bound) = self.bounded_run(
            memory,
            address,
            self.observation_limit(),
            None,
            stated(address),
        );
        (!bytes.is_empty()).then_some((address, bytes, bound))
    }

    /// Returns the existing per-observation and materialized-buffer bound.
    pub fn observation_limit(self) -> usize {
        self.context
            .budget()
            .limit(Budget::ExecutionArgumentBytes)
            .min(self.buffer_limit())
    }

    /// Reads a bounded prefix under the owner's retained payload reservation.
    pub fn known_run_within(
        self,
        memory: &Memory,
        address: Address,
        limit: usize,
        maximum: usize,
    ) -> Option<Vec<u8>> {
        memory.known_run_within(
            address,
            limit.min(self.observation_limit()),
            maximum,
            self.context.image(),
        )
    }

    /// Meets another visit without cloning an already-retained buffer.
    pub fn intersect_known_run(self, memory: &Memory, address: Address, bytes: &mut Vec<u8>) {
        memory.intersect_known_run(address, bytes, self.context.image());
    }

    /// The address a value names, when it names one.
    ///
    /// An integer is an address in the program's space, which the host may
    /// relate to its symbols.
    ///
    /// # Arguments
    ///
    /// * `value` - The pointer operand.
    /// * `address_space` - The address space the access is qualified by;
    ///   anything but the flat space names nothing the execution models.
    pub fn address_of<T: Target>(
        self,
        value: &Value<T>,
        address_space: Option<NonZeroU16>,
    ) -> Option<Address> {
        if address_space.is_some() {
            return None;
        }
        let ptr_size = self.context.machine().ptr_size;
        match value {
            Value::Addr(address) => Some(*address),
            Value::Const(constant) => {
                let number = constant.scalar_pattern_at_width(ptr_size.bits(), ptr_size)?;
                if let Ok(physical) = u64::try_from(number)
                    && self
                        .context
                        .disjoint_numeric()
                        .iter()
                        .any(|range| range.contains(&physical))
                {
                    return Some(Address::new(Root::Number, physical.cast_signed(), ptr_size));
                }
                let shift = self.context.image().and_then(|image| image.number_offset());
                let (root, coordinate) = match shift {
                    Some(shift) => (
                        Root::Symbol,
                        ptr_size.mask_unsigned_128(
                            number.wrapping_add(i128::from(shift).cast_unsigned()),
                        ),
                    ),
                    None => (Root::Number, number),
                };
                let coordinate = u64::try_from(coordinate).ok()?;
                Some(Address::new(root, coordinate.cast_signed(), ptr_size))
            }

            Value::Integer(_) | Value::StackDerived | Value::Unknown => None,
        }
    }

    /// The bytes an access of type `ty` at `pointer` touches: where they start
    /// and how many there are, when both are known.
    ///
    /// # Arguments
    ///
    /// * `pointer` - The address operand.
    /// * `ty` - The type accessed.
    /// * `address_space` - The access's address-space qualifier.
    pub fn footprint<T: Target>(
        self,
        pointer: &Value<T>,
        ty: &T::Type,
        address_space: Option<NonZeroU16>,
    ) -> Option<(Address, usize)> {
        let width = self.width::<T>(ty)?;
        Some((self.address_of(pointer, address_space)?, width))
    }

    /// A block size the execution writes byte by byte, as a length.
    pub fn block_footprint<T: Target>(
        self,
        pointer: &Value<T>,
        size: &Value<T>,
    ) -> Option<(Address, usize)> {
        let len = self.block_len(size)?;
        Some((self.address_of(pointer, None)?, len))
    }

    /// Returns the full footprint of a stream whose pointer names its first element.
    pub fn stream_footprint<T: Target>(
        self,
        pointer: &Value<T>,
        size: &Value<T>,
        element_bytes: usize,
        reverse: bool,
    ) -> Option<(Address, usize)> {
        let len = self.block_len(size)?;
        let address = self.address_of(pointer, None)?;
        let start = CopyOrder::Stream {
            element_bytes,
            reverse,
        }
        .span(address, len)
        .ok()?;
        Some((start, len))
    }

    /// Projects the type once under the invocation policy.
    pub fn type_layout<T: Target>(self, ty: &T::Type) -> Option<TypeLayout> {
        match self.operation {
            Some(operation) => operation.read(),
            None => T::type_layout(ty, self.ptr_size()),
        }
    }

    /// Returns the fixed byte extent without inventing a runtime vector size.
    fn layout_width(self, layout: TypeLayout) -> Option<usize> {
        layout
            .storage_bytes(self.ptr_size())
            .filter(|bytes| *bytes <= self.buffer_limit())
    }

    /// Returns the fixed extent of the type in bytes.
    fn width<T: Target>(self, ty: &T::Type) -> Option<usize> {
        self.layout_width(self.type_layout::<T>(ty)?)
    }

    /// Reads a value of type `ty` at `pointer`.
    ///
    /// # Arguments
    ///
    /// * `memory` - What the execution knows of memory.
    /// * `pointer` - The address operand.
    /// * `ty` - The type loaded.
    /// * `address_space` - The access's address-space qualifier.
    #[cfg(test)]
    pub fn load<T: Target>(
        self,
        memory: &Memory,
        pointer: &Value<T>,
        ty: &T::Type,
        address_space: Option<NonZeroU16>,
    ) -> Value<T> {
        let Some(layout) = self.type_layout::<T>(ty) else {
            return Value::Unknown;
        };
        let (Some(address), Some(width)) = (
            self.address_of(pointer, address_space),
            self.layout_width(layout),
        ) else {
            return Value::Unknown;
        };
        self.load_at(memory, address, width, layout)
    }

    /// Accepts only a complete scalar integer or pointer declaration at machine pointer width.
    fn named_address_layout(self, layout: TypeLayout) -> bool {
        if layout.is_boolean()
            || layout.is_floating()
            || layout.float_format().is_some()
            || layout.is_reference()
            || layout.is_unknown()
            || layout.vector_descriptor().is_some()
            || layout.vector_mask_descriptor().is_some()
        {
            return false;
        }
        let bits = self.ptr_size().bits();
        if layout.is_pointer() {
            return !layout.is_integer() && layout.bit_width().is_none_or(|width| width == bits);
        }
        layout.is_integer()
            && layout.bit_width() == Some(bits)
            && layout
                .integer_conversion_type()
                .is_some_and(|integer| integer.bits() == bits)
    }

    /// Reads a location using the same layout that determined its extent.
    fn load_at<T: Target>(
        self,
        memory: &Memory,
        address: Address,
        width: usize,
        layout: TypeLayout,
    ) -> Value<T> {
        if self.named_address_layout(layout)
            && let Some(image) = self.context.image()
            && self.context.machine().byte_order == Some(image.endianness())
            && let Some(named) = memory.unchanged_image_address(address, width, image)
        {
            return Value::Addr(named);
        }
        match memory.load(address, width, self.context.image()) {
            Loaded::Address(stored) => Value::Addr(stored),
            Loaded::Bytes(bytes) => self
                .context
                .machine()
                .byte_order
                .or_else(|| (width == 1).then_some(Endianness::Little))
                .and_then(|order| ConstValue::from_layout_bytes(&bytes, layout, order))
                .map_or(Value::Unknown, Value::Const),
            Loaded::StackDerived => Value::StackDerived,
            Loaded::Unknown => Value::Unknown,
        }
    }

    /// Loads a value and retains the exact checked source extent without a second type query.
    ///
    /// # Errors
    ///
    /// Returns a memory refusal for an unrepresentable span or a missing live allocation extent.
    pub fn load_with_footprint<T: Target>(
        self,
        memory: &Memory,
        pointer: &Value<T>,
        ty: &T::Type,
        address_space: Option<NonZeroU16>,
    ) -> Result<ReadValue<T>, MemoryRefusal> {
        let Some(layout) = self.type_layout::<T>(ty) else {
            return Ok(ReadValue {
                value: Value::Unknown,
                footprint: None,
            });
        };
        let (Some(address), Some(width)) = (
            self.address_of(pointer, address_space),
            self.layout_width(layout),
        ) else {
            return Ok(ReadValue {
                value: Value::Unknown,
                footprint: None,
            });
        };
        if let Some(reason) = memory.admit(address, width, Transfer::Read) {
            return Err(reason);
        }
        Ok(ReadValue {
            value: self.load_at(memory, address, width, layout),
            footprint: Some((address, width)),
        })
    }

    /// Checks whether a complete repeated-read span is backed by live, readable storage.
    ///
    /// # Errors
    ///
    /// Returns a memory refusal when the span wraps or exceeds a live allocation's extent.
    pub fn readable_span(
        self,
        memory: &Memory,
        address: Address,
        len: usize,
    ) -> Result<bool, MemoryRefusal> {
        if let Some(reason) = memory.access_refusal(address, len) {
            return Err(reason);
        }
        Ok(memory.readable_span(address, len, self.context.image()))
    }

    /// Reads an already resolved storage representation.
    pub fn load_layout<T: Target>(
        self,
        memory: &Memory,
        address: Address,
        layout: TypeLayout,
    ) -> Result<Value<T>, MemoryRefusal> {
        let width = self
            .layout_width(layout)
            .ok_or(MemoryRefusal::AtomicContract)?;
        if let Some(reason) = memory.admit(address, width, Transfer::Read) {
            return Err(reason);
        }
        Ok(self.load_at(memory, address, width, layout))
    }

    /// Joins a possible store against only its bounded destination footprint.
    ///
    /// # Errors
    ///
    /// Returns a memory-contract refusal or configured store exhaustion.
    pub fn maybe_store<T: Target>(
        self,
        memory: &mut Memory,
        address: Address,
        value: &Value<T>,
        layout: TypeLayout,
    ) -> Result<(), AccessError> {
        let width = layout
            .storage_bytes(self.ptr_size())
            .ok_or(MemoryRefusal::AtomicContract)?;
        if let Some(exhaustion) = self
            .context
            .budget()
            .check(Budget::ExecutionBufferBytes, width)
        {
            return Err(exhaustion.into());
        }
        if !self.admit_possible_write(memory, address, width) {
            return Err(memory
                .refusal()
                .unwrap_or(MemoryRefusal::InstructionSource)
                .into());
        }
        let mut candidate = memory.write_candidate(address);
        self.with_write_guard(None).store_layout(
            &mut candidate,
            &Value::Addr(address),
            value,
            layout,
            None,
        )?;
        if let Some(reason) = candidate.refusal() {
            return Err(reason.into());
        }
        memory.intersect_span(&candidate, address, width);
        Ok(())
    }

    /// Proves an actual nonempty access span cannot fault on its alignment.
    ///
    /// An access stating the alignment it requires
    /// ([`AccessQualifiers::required`]) needs its address aligned to that,
    /// on any machine. Any other access needs none where the machine
    /// tolerates misaligned ordinary accesses ([`MisalignedAccess::Tolerated`]),
    /// and otherwise each element start at its natural alignment, which an
    /// explicit weaker guarantee lowers. Byte accesses need no base fact;
    /// wider accesses require numeric coordinates or a realigned frame, whose
    /// base the mask aligned ([`Root::StackAligned`]). Other relative stack,
    /// heap and opaque offsets establish no base alignment.
    fn aligned_span(
        self,
        address: Address,
        len: usize,
        element_bytes: usize,
        access: AccessQualifiers,
    ) -> bool {
        if element_bytes == 0 || access.address_space.is_some() {
            return false;
        }
        if len == 0 {
            return true;
        }
        let required = match access.required {
            // The address alone is held to a stated requirement.
            Some(bytes) => usize::from(bytes.get()),
            None if self.context.machine().misaligned == MisalignedAccess::Tolerated => {
                return true;
            }
            None => {
                let required = access.alignment.map_or(element_bytes, |alignment| {
                    usize::from(alignment.get()).min(element_bytes)
                });
                if len > element_bytes
                    && required.is_power_of_two()
                    && !element_bytes.is_multiple_of(required)
                {
                    return false;
                }
                required
            }
        };
        if required == 1 {
            return true;
        }
        if !required.is_power_of_two() {
            return false;
        }
        let numeric = match address.root() {
            Root::Number => address.offset().cast_unsigned(),
            Root::Symbol => {
                let Some(shift) = self.context.image().and_then(|image| image.number_offset())
                else {
                    return false;
                };
                address
                    .offset()
                    .cast_unsigned()
                    .wrapping_sub(shift.cast_unsigned())
            }
            Root::StackAligned { bits, .. } => {
                return 1u64
                    .checked_shl(u32::from(bits))
                    .is_some_and(|base| base.is_multiple_of(required as u64))
                    && address
                        .offset()
                        .cast_unsigned()
                        .is_multiple_of(required as u64);
            }
            Root::Stack
            | Root::StackFrame(_)
            | Root::Heap(_)
            | Root::Opaque { .. }
            | Root::StackDerived { .. } => return false,
        };
        // Power-of-two alignment depends only on these low target bits,
        // including when the full numeric pointer is wider than u64.
        self.ptr_size()
            .mask_unsigned(numeric)
            .is_multiple_of(required as u64)
    }

    /// Proves an executed access cannot trap on its alignment: a naturally
    /// aligned span ([`Self::aligned_span`]), or one wholly in a live
    /// activation's frame ([`State::frame_backed`]).
    pub fn proven_span<T: Target>(
        self,
        state: &State<T>,
        address: Address,
        len: usize,
        element_bytes: usize,
        access: AccessQualifiers,
    ) -> bool {
        self.aligned_span(address, len, element_bytes, access)
            || (access.address_space.is_none() && state.frame_backed(address, len))
    }

    /// Proves writable backing for an image span or live private allocation.
    ///
    /// Missing regions establish no permission. Each byte is queried because the
    /// image oracle may refine a surrounding region with a smaller protected span.
    /// Private storage requires a live frame extent or allocation; unnamed storage is unproved.
    fn writable_span<T: Target>(self, state: &State<T>, address: Address, len: usize) -> bool {
        if len == 0 {
            return true;
        }
        match address.root() {
            Root::Heap(_) => return state.memory().is_sound(address, len),
            Root::Stack | Root::StackFrame(_) | Root::StackAligned { .. } => {
                return state.frame_backed(address, len);
            }
            Root::Symbol => {}
            Root::Number | Root::Opaque { .. } | Root::StackDerived { .. } => return false,
        }
        let Some(image) = self.context.image() else {
            return false;
        };
        for offset in 0..len {
            let Some(at) = i64::try_from(offset)
                .ok()
                .and_then(|offset| address.moved(offset, self.ptr_size()))
                .and_then(Address::image_address)
            else {
                return false;
            };
            if !image.region(at).is_some_and(|region| {
                region.mutability == Mutability::Writable && region.covers(at, 1)
            }) {
                return false;
            }
        }
        true
    }

    /// Records a write only after proving backing for every intermediate destination.
    pub fn record_write<T: Target>(
        self,
        state: &State<T>,
        trace: &mut Trace,
        address: Address,
        len: usize,
    ) {
        if trace.clean_headers().next().is_some() && !self.writable_span(state, address, len) {
            trace.taint();
        }
        trace.record_write(address, len, self.ptr_size());
    }

    /// Returns whether every byte of the `len` bytes at `address` is image
    /// storage promised read-only for the whole run, in one region, that no
    /// protection change on the path `memory` holds covered: memory no
    /// observer can write.
    #[must_use]
    pub fn read_only_image(self, memory: &Memory, address: Address, len: usize) -> bool {
        let (Some(at), Some(image)) = (address.image_address(), self.context.image()) else {
            return false;
        };
        image.region(at).is_some_and(|region| {
            region.mutability == Mutability::ReadOnly && region.covers(at, len as u64)
        }) && !memory
            .instruction_sources()
            .protection_changed(address, u64::try_from(len).unwrap_or(u64::MAX))
    }

    /// Proves an aligned ordinary image load cannot fault or observe changing memory.
    fn image_read_proof<T: Target>(
        self,
        pointer: &Value<T>,
        ty: &T::Type,
        space: AccessQualifiers,
    ) -> Option<ReplacementProof> {
        if space.volatile || space.permission != AccessPermission::Current {
            return None;
        }
        let (place, len) = self.footprint::<T>(pointer, ty, space.address_space)?;
        let address = place.image_address()?;
        let image = self.context.image()?;
        if !self.aligned_span(place, len, len, space) {
            return None;
        }
        let region = image.region(address)?;
        if region.mutability != Mutability::ReadOnly
            || !region.covers(address, len as u64)
            || image.bytes(address, len)?.len() != len
        {
            return None;
        }
        Some(ReplacementProof::ReadOnlyImage {
            address,
            len: len as u64,
        })
    }

    /// Proves an ordinary load cannot fault, before it executes in `state`.
    ///
    /// Read-only image bytes give the image's proof
    /// ([`Self::image_read_proof`]). Elsewhere the span must lie wholly in
    /// storage the run knows is live, at an alignment that cannot fault: a
    /// live activation's frame ([`State::frame_backed`]), or a live
    /// allocation's extent at a proven alignment ([`Self::aligned_span`]).
    /// Writable image bytes and numeric addresses prove nothing here: the run
    /// knows no extent for them.
    pub fn read_proof<T: Target>(
        self,
        state: &State<T>,
        pointer: &Value<T>,
        ty: &T::Type,
        space: AccessQualifiers,
    ) -> Option<ReplacementProof> {
        // Image bytes a protection change on this path covered may have
        // been written: no longer the image's, they prove nothing.
        if let Some(proof) = self.image_read_proof(pointer, ty, space) {
            let ReplacementProof::ReadOnlyImage { address, len } = proof else {
                return Some(proof);
            };
            let start = Address::new(Root::Symbol, address.cast_signed(), self.ptr_size());
            return (!state
                .memory()
                .instruction_sources()
                .protection_changed(start, len))
            .then_some(proof);
        }
        if space.volatile || space.permission != AccessPermission::Current {
            return None;
        }
        let (address, len) = self.footprint::<T>(pointer, ty, space.address_space)?;
        let live = match address.root() {
            Root::Heap(_) => {
                state.memory().is_sound(address, len)
                    && state.memory().readable_span(address, len, None)
                    && self.aligned_span(address, len, len, space)
            }
            Root::Stack | Root::StackFrame(_) | Root::StackAligned { .. } => {
                space.address_space.is_none() && state.frame_backed(address, len)
            }
            Root::Symbol | Root::Number | Root::Opaque { .. } | Root::StackDerived { .. } => false,
        };
        (live && len != 0).then_some(ReplacementProof::Accessible)
    }

    /// Reads a bounded NUL-terminated string using the target's memory oracle.
    pub fn read_string(
        self,
        memory: &Memory,
        address: Address,
        limit: usize,
    ) -> Result<StringScan<Vec<u8>>, MemoryRefusal> {
        memory.read_string(
            address,
            limit
                .min(self.string_limit())
                .min(self.buffer_limit().saturating_add(1)),
            self.context.image(),
        )
    }

    /// Counts bytes before a bounded NUL without allocating a string buffer.
    ///
    /// Uses the target image oracle. Unknown data or a missing terminator
    /// yield no length; the scan says how many bytes it inspected either way.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::analysis::execution::MemoryRefusal`] for an invalid
    /// visited coordinate, object lifetime or extent.
    pub fn string_length(
        self,
        memory: &Memory,
        address: Address,
        limit: usize,
    ) -> Result<StringScan<usize>, MemoryRefusal> {
        memory.string_length(
            address,
            limit.min(self.string_limit()),
            self.context.image(),
        )
    }

    /// Writes `value`, of type `ty`, at `pointer`.
    ///
    /// # Arguments
    ///
    /// * `memory` - What the execution knows of memory.
    /// * `pointer` - The address operand.
    /// * `value` - The value stored.
    /// * `ty` - Its type.
    /// * `address_space` - The access's address-space qualifier.
    ///
    /// # Errors
    ///
    /// Returns configured buffer exhaustion before image-address resolution,
    /// protection queries or effects of the write.
    pub fn store<T: Target>(
        self,
        memory: &mut Memory,
        pointer: &Value<T>,
        value: &Value<T>,
        ty: &T::Type,
        address_space: Option<NonZeroU16>,
    ) -> Result<(), Exhaustion> {
        let Some(layout) = self.type_layout::<T>(ty) else {
            if self.has_write_guard()
                && !self.admit_unresolved_write(
                    memory,
                    self.address_of(pointer, address_space),
                    None,
                )
            {
                return Ok(());
            }
            memory.escape_if_stack(value);
            memory.escape_if_stack(pointer);
            memory.clobber();
            return Ok(());
        };
        self.store_layout(memory, pointer, value, layout, address_space)
    }

    /// Writes some element of the `len` bytes from `start`: a store whose
    /// address is not known exactly, only that every byte it may reach lies
    /// there, inside one object. Each of those bytes becomes unknown, and
    /// each of the `stored` values escapes where it holds the private stack,
    /// as any store of it would make it.
    pub fn store_element<T: Target>(
        self,
        memory: &mut Memory,
        start: Address,
        len: usize,
        stored: &[Value<T>],
    ) {
        // The store reaches one element of the span, which faults for
        // certain only where the image denies a write to every byte of it.
        if let Some(reason) = self.write_denial(memory, start, len) {
            if reason == MemoryRefusal::ReadOnlyImage
                && self
                    .context
                    .image()
                    .is_some_and(|image| memory.denies_every_image_write(start, len, image))
            {
                memory.deny_write(reason, start, len);
            } else {
                memory.refuse(reason);
            }
            return;
        }
        if !memory.holds_privately(start) {
            for value in stored {
                memory.escape_if_stack(value);
            }
        }
        memory.store_unknown(start, len);
    }

    /// Writes an already resolved physical layout under the same memory rules.
    ///
    /// # Errors
    ///
    /// Returns configured buffer exhaustion before resolving an address, querying
    /// image protection or changing memory and pointer escape state.
    pub fn store_layout<T: Target>(
        self,
        memory: &mut Memory,
        pointer: &Value<T>,
        value: &Value<T>,
        layout: TypeLayout,
        address_space: Option<NonZeroU16>,
    ) -> Result<(), Exhaustion> {
        let Some(storage) = layout.storage_layout(self.ptr_size()) else {
            if self.has_write_guard()
                && !self.admit_unresolved_write(
                    memory,
                    self.address_of(pointer, address_space),
                    None,
                )
            {
                return Ok(());
            }
            memory.escape_if_stack(value);
            memory.escape_if_stack(pointer);
            memory.clobber();
            return Ok(());
        };
        let width = storage.bytes();
        if width == 0 {
            return Ok(());
        }
        if let Some(exhaustion) = self
            .context
            .budget()
            .check(Budget::ExecutionBufferBytes, width)
        {
            return Err(exhaustion);
        }
        let Some(address) = self.address_of(pointer, address_space) else {
            if !self.admit_unresolved_write(memory, None, Some(width)) {
                return Ok(());
            }
            if self.has_write_guard() && address_space.is_some() {
                memory.refuse(MemoryRefusal::InstructionSource);
                return Ok(());
            }
            memory.escape_if_stack(value);
            memory.escape_if_stack(pointer);
            memory.clobber();
            return Ok(());
        };
        if !self.admit_write(memory, address, width) {
            return Ok(());
        }

        match value {
            Value::Addr(stored) => {
                if stored.root().may_reference_stack() && !memory.holds_privately(address) {
                    memory.escape_stack();
                }
                match u8::try_from(width) {
                    Ok(pieces) if width == self.context.machine().ptr_size.bytes() => {
                        memory.store_address(address, *stored, pieces);
                    }
                    _ => {
                        memory.escape_if_stack(value);
                        memory.store_unknown(address, width);
                    }
                }
            }
            Value::Const(constant) => match self.bytes_of(constant, storage) {
                Some(bytes) => memory.store_encoded(address, bytes),
                None => memory.store_unknown(address, width),
            },
            Value::StackDerived => {
                memory.escape_stack();
                memory.store_unknown(address, width);
            }
            Value::Integer(bits) => {
                if bits.may_reference_stack() {
                    memory.escape_stack();
                }
                memory.store_unknown(address, width);
            }
            Value::Unknown => memory.store_unknown(address, width),
        }
        Ok(())
    }

    /// Why the whole destination may not be written, as the memory, the
    /// image and the write guard say; `None` where it may.
    fn write_denial(
        self,
        memory: &mut Memory,
        address: Address,
        len: usize,
    ) -> Option<MemoryRefusal> {
        memory
            .write_refusal(address, len, self.context.image())
            .or_else(|| {
                (len != 0)
                    .then_some(self.write_guard)
                    .flatten()
                    .and_then(|guard| {
                        check_write(guard, memory.instruction_sources_mut(), address, len).err()
                    })
            })
    }

    /// Checks the whole destination of a write that may not happen: where
    /// it may not be written, the execution is refused, since the machine
    /// may never make the write.
    fn admit_possible_write(self, memory: &mut Memory, address: Address, len: usize) -> bool {
        if let Some(reason) = self.write_denial(memory, address, len) {
            memory.refuse(reason);
            false
        } else {
            true
        }
    }

    /// Checks the whole destination of a write the machine makes before
    /// changing contents, aliases or pointer escape state: one into storage
    /// that may not be written faults ([`Memory::deny_write`]).
    fn admit_write(self, memory: &mut Memory, address: Address, len: usize) -> bool {
        if let Some(reason) = self.write_denial(memory, address, len) {
            memory.deny_write(reason, address, len);
            false
        } else {
            true
        }
    }

    /// Writes model bytes within the configured buffer extent.
    pub fn write_bytes(self, memory: &mut Memory, address: Address, bytes: &[u8]) {
        if !self.admit_write(memory, address, bytes.len()) {
            return;
        }
        if bytes.len() > self.buffer_limit() {
            memory.store_unknown(address, bytes.len());
            return;
        }

        memory.store_bytes(address, bytes);
    }

    /// Fills admitted model memory with one byte without a temporary buffer.
    pub fn fill_byte(self, memory: &mut Memory, address: Address, byte: u8, len: usize) {
        if !self.admit_write(memory, address, len) {
            return;
        }
        if len > self.buffer_limit() {
            memory.store_unknown(address, len);
            return;
        }

        memory.store_pattern(address, &[byte], len);
    }

    /// Writes the pointer `value` at `address`, a pointer's width of bytes,
    /// escaping the private stack where a cell others can read takes a
    /// pointer into it.
    pub fn write_address(self, memory: &mut Memory, address: Address, value: Address) {
        let width = self.ptr_size().bytes();
        if !self.admit_write(memory, address, width) {
            return;
        }
        if value.root().may_reference_stack() && !memory.holds_privately(address) {
            memory.escape_stack();
        }
        match u8::try_from(width) {
            Ok(pieces) => memory.store_address(address, value, pieces),
            Err(_) => memory.store_unknown(address, width),
        }
    }

    /// Forgets model memory contents across the supplied span.
    pub fn write_unknown(self, memory: &mut Memory, address: Address, len: usize) {
        if self.admit_write(memory, address, len) {
            memory.store_unknown(address, len);
        }
    }

    /// A constant's bytes in the target's order, when there are exactly
    /// `width` of them.
    fn bytes_of<'b, T: Target>(
        self,
        constant: &'b ConstValue<T>,
        storage: StorageLayout,
    ) -> Option<impl ExactSizeIterator<Item = u8> + 'b> {
        let width = storage.bytes();
        let order = self
            .context
            .machine()
            .byte_order
            .or_else(|| (width == 1).then_some(Endianness::Little))?;
        constant.encoded_storage_bytes(storage, order, self.context.machine().ptr_size)
    }

    /// Checks a full fill span before encoding or changing memory.
    fn fill_span<T: Target>(
        self,
        memory: &mut Memory,
        pointer: &Value<T>,
        value: &Value<T>,
        element_bytes: Option<usize>,
        size: &Value<T>,
        reverse: bool,
    ) -> Option<(Address, usize)> {
        if size.is_zero_count() {
            return None;
        }
        let first = self.address_of(pointer, None);
        let len = self.block_len(size);
        let (Some(first), Some(len)) = (first, len) else {
            if !self.admit_unresolved_write(memory, first, len) {
                return None;
            }
            memory.escape_if_stack(pointer);
            memory.escape_if_stack(value);
            memory.clobber();
            return None;
        };
        let address = if reverse {
            let Some(element_bytes) = element_bytes else {
                if !self.admit_unresolved_write(memory, Some(first), Some(len)) {
                    return None;
                }
                memory.escape_if_stack(pointer);
                memory.escape_if_stack(value);
                memory.clobber();
                return None;
            };
            match (CopyOrder::Stream {
                element_bytes,
                reverse,
            })
            .span(first, len)
            {
                Ok(start) => start,
                Err(reason) => {
                    memory.refuse(reason);
                    return None;
                }
            }
        } else {
            first
        };
        self.admit_write(memory, address, len)
            .then_some((address, len))
    }

    /// Fills using one storage view for both the extent and encoding.
    pub fn fill_storage<T: Target>(
        self,
        memory: &mut Memory,
        pointer: &Value<T>,
        value: &Value<T>,
        storage: Option<StorageLayout>,
        size: &Value<T>,
        reverse: bool,
    ) {
        let Some((address, len)) = self.fill_span(
            memory,
            pointer,
            value,
            storage.map(StorageLayout::bytes),
            size,
            reverse,
        ) else {
            return;
        };
        let Some(storage) = storage else {
            memory.escape_if_stack(value);
            memory.store_unknown(address, len);
            return;
        };
        match value {
            Value::Const(constant) => match self.bytes_of(constant, storage) {
                Some(bytes) => self.fill_encoded(memory, address, bytes, len),
                None => memory.store_unknown(address, len),
            },
            Value::Addr(stored) if storage.bytes() == self.ptr_size().bytes() => {
                self.fill_address(memory, address, *stored, storage.bytes(), len);
            }
            Value::Addr(_) | Value::Integer(_) | Value::StackDerived | Value::Unknown => {
                memory.escape_if_stack(value);
                memory.store_unknown(address, len);
            }
        }
    }

    /// Encodes one bounded pattern and reuses it across the fill.
    fn fill_encoded(
        self,
        memory: &mut Memory,
        address: Address,
        bytes: impl ExactSizeIterator<Item = u8>,
        len: usize,
    ) {
        let pattern_len = bytes.len().min(len);
        if pattern_len == 0 {
            memory.store_unknown(address, len);
            return;
        }
        // Scalar patterns use stack storage. A wider pattern has one scratch
        // allocation while it is encoded.
        let mut small = [0u8; 16];
        if let Some(pattern) = small.get_mut(..pattern_len) {
            for (slot, byte) in pattern.iter_mut().zip(bytes) {
                *slot = byte;
            }
            memory.store_pattern(address, pattern, len);
            return;
        }
        let pattern: Vec<u8> = bytes.take(pattern_len).collect();
        memory.store_pattern(address, &pattern, len);
    }

    /// Repeats pointer pieces without inventing their numeric bytes.
    fn fill_address(
        self,
        memory: &mut Memory,
        address: Address,
        stored: Address,
        width: usize,
        len: usize,
    ) {
        let Ok(width_bytes) = u8::try_from(width) else {
            if stored.root().may_reference_stack() {
                memory.escape_stack();
            }
            memory.store_unknown(address, len);
            return;
        };
        for offset in (0..len).step_by(width) {
            let Some(at) = i128::try_from(offset)
                .ok()
                .and_then(|offset| address.displaced(offset, self.ptr_size()))
            else {
                memory.refuse(MemoryRefusal::AddressRange);
                return;
            };
            let count = len.saturating_sub(offset).min(width);
            let Ok(count) = u8::try_from(count) else {
                memory.refuse(MemoryRefusal::BlockLayout);
                return;
            };
            memory.store_address_prefix(at, stored, width_bytes, count);
        }
    }

    /// Copies `size` bytes from `source` to `pointer`. Forward (`memcpy`,
    /// and `rep movsb`) reads each byte just before writing it, so a
    /// destination overlapping the source above it repeats the bytes already
    /// copied; `buffered` (`memmove`) reads every byte before writing any, so
    /// overlapping regions receive what the source held.
    /// Returns whether all executed source reads have readable backing; an
    /// unknown or refused transfer returns `false`, and an empty one returns `true`.
    pub fn copy<T: Target>(
        self,
        memory: &mut Memory,
        pointer: &Value<T>,
        source: &Value<T>,
        size: &Value<T>,
        buffered: bool,
    ) -> bool {
        if size.is_zero_count() {
            return true;
        }
        let destination = self.address_of(pointer, None);
        let extent = self.block_len(size);
        let (Some(to), Some(len)) = (destination, extent.and_then(|len| i64::try_from(len).ok()))
        else {
            let admitted = match (destination, extent) {
                (Some(address), Some(len)) => self.admit_write(memory, address, len),
                _ => self.admit_unresolved_write(memory, destination, extent),
            };
            if !admitted {
                return false;
            }
            memory.escape_if_stack(pointer);
            memory.clobber();
            return false;
        };

        let Some(from) = self.address_of(source, None) else {
            if let Ok(bytes) = usize::try_from(len)
                && self.admit_write(memory, to, bytes)
            {
                memory.escape_if_stack(pointer);
                memory.escape_if_stack(source);
                memory.clobber();
            }
            return false;
        };
        let image = self.context.image();
        if self.write_guard.is_none() {
            return memory.copy(to, from, len, image, buffered);
        }
        memory.copy_with_guard(
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
            self.write_guard,
        )
    }

    /// Copies a model's already bounded extent independently of the native block-instruction limit.
    pub fn copy_model<T: Target>(
        self,
        memory: &mut Memory,
        destination: Address,
        source: &Value<T>,
        len: usize,
    ) -> bool {
        if len == 0 {
            return true;
        }
        let Ok(count) = i64::try_from(len) else {
            memory.refuse(MemoryRefusal::AddressRange);
            return false;
        };
        let Some(from) = self.address_of(source, None) else {
            if self.admit_write(memory, destination, len) {
                memory.escape_if_stack(&Value::<T>::Addr(destination));
                memory.escape_if_stack(source);
                memory.clobber();
            }
            return false;
        };
        memory.copy_with_guard(
            destination,
            from,
            count,
            self.context.image(),
            CopyOrder::Snapshot,
            self.write_guard,
        )
    }

    /// Copies a stream whose addresses name the first elements, with one read unit per element.
    ///
    /// Returns whether all executed source reads have readable backing at their
    /// execution points. Earlier writes can supply later source cells. Unknown
    /// or refused transfers return `false`; valid empty ones return `true`.
    pub fn copy_stream<T: Target>(
        self,
        memory: &mut Memory,
        pointer: &Value<T>,
        source: &Value<T>,
        size: &Value<T>,
        element_bytes: usize,
        reverse: bool,
    ) -> bool {
        if element_bytes != 0 && size.is_zero_count() {
            return true;
        }
        let destination = self.address_of(pointer, None);
        let extent = self.block_len(size);
        let (Some(to), Some(len)) = (destination, extent) else {
            if !self.admit_unresolved_write(memory, destination, extent) {
                return false;
            }
            memory.escape_if_stack(pointer);
            memory.clobber();
            return false;
        };

        let Some(from) = self.address_of(source, None) else {
            match (CopyOrder::Stream {
                element_bytes,
                reverse,
            })
            .span(to, len)
            {
                Ok(start) => {
                    if self.admit_write(memory, start, len) {
                        memory.escape_if_stack(pointer);
                        memory.escape_if_stack(source);
                        memory.clobber();
                    }
                }
                Err(reason) => memory.refuse(reason),
            }
            return false;
        };
        memory.copy_with_guard(
            to,
            from,
            len as i64,
            self.context.image(),
            CopyOrder::Stream {
                element_bytes,
                reverse,
            },
            self.write_guard,
        )
    }

    /// A block size the execution writes byte by byte.
    fn block_len<T: Target>(self, size: &Value<T>) -> Option<usize> {
        let len = size.constant()?.as_u64()?;
        if len > self.context.budget().limit(Budget::ExecutionBlockBytes) as u64 {
            return None;
        }
        usize::try_from(len).ok()
    }
}

#[cfg(test)]
mod tests {

    use std::num::NonZeroU8;

    use super::*;
    use crate::{
        analysis::{
            Machine,
            budget::WorkBudget,
            execution::{
                ExtentSource, FindingKind, InstructionSourceId, InstructionSources, frame::Frame,
                memory::Loaded,
            },
            image::{ImageData, ImageRegion, Mutability},
        },
        ir::{FloatBits, exception::HandlerKind, integer::IntegerBits, lanes::VectorConst},
        testing::{MACHINE, MockTarget, MockType},
    };

    /// Builds inline and shared unknown bits that may contain a private stack pointer.
    fn partial_stack_values() -> [Value<MockTarget>; 2] {
        [
            Value::partial_integer(IntegerBits::new(64, 15, 0).unwrap(), true),
            Value::partial_integer(IntegerBits::new(128, 1, 1u128 << 127).unwrap(), true),
        ]
    }

    /// Independently states arbitrary byte-aligned element widths for the memory codec.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct PatternHost;

    impl Target for PatternHost {
        type TypeRef = ();
        type MethodRef = ();
        type FieldRef = ();
        type SigRef = ();
        type SymbolRef = ();
        type ExceptionKind = ();
        type Type = u32;
        type OriginalInstruction = ();
        type LocalSignature = ();
        type VariableFacts = ();
        type InstructionFacts = ();
        type FunctionFacts = ();
        type Capability = ();
        fn ptr_bytes(&self) -> u32 {
            8
        }
        fn synthetic_instruction() {}
        fn unknown_type() -> u32 {
            0
        }
        fn is_integer(_: &u32) -> bool {
            true
        }
        fn is_floating(_: &u32) -> bool {
            false
        }
        fn is_signed(_: &u32) -> bool {
            false
        }
        fn is_pointer(_: &u32) -> bool {
            false
        }
        fn is_reference(_: &u32) -> bool {
            false
        }
        fn is_unknown(bits: &u32) -> bool {
            *bits == 0
        }
        fn bit_width(bits: &u32) -> Option<u32> {
            Some(*bits)
        }
        fn instruction_mnemonic(_: &()) -> &'static str {
            "pattern"
        }
        fn instruction_rva(_: &()) -> u64 {
            0
        }
        fn handler_kind(_: &()) -> HandlerKind {
            HandlerKind::Catch
        }
    }

    /// Declares no image access legal during an oversized scalar store.
    struct NoStoreQueries;

    impl ImageData for NoStoreQueries {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }
        fn number_offset(&self) -> Option<i64> {
            panic!("oversized address query")
        }
        fn region(&self, _: u64) -> Option<ImageRegion> {
            panic!("unexpected region query")
        }
        fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
            panic!("unexpected byte query")
        }
        fn read_only_overlap(&self, _: u64, _: u64) -> bool {
            panic!("oversized protection query")
        }
    }

    /// A large physical store cannot query protection, erase bytes or escape a held pointer.
    #[test]
    fn oversized_scalar_store_preserves_memory_without_image_queries() {
        let context = AnalysisContext::new(MACHINE)
            .with_image(&NoStoreQueries)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionBufferBytes, 16));
        let access = Access::new(&context);
        let at = Address::new(Root::Symbol, 0x1000, PointerSize::Bit64);
        let private = Address::new(Root::Stack, -32, PointerSize::Bit64);
        for (bits, bytes) in [(136, 17usize), (4096, 512), (u32::MAX - 7, 536_870_911)] {
            for pointer in [
                Value::<PatternHost>::Addr(at),
                Value::Addr(private),
                Value::Const(ConstValue::U64(0x1000)),
                Value::Unknown,
            ] {
                for value in [
                    Value::<PatternHost>::Unknown,
                    Value::Const(ConstValue::U64(0x1234)),
                    Value::Addr(private),
                ] {
                    let mut memory = Memory::default();
                    memory.store_bytes(at, &[0xa5; 16]);
                    memory.store_address(private, at, 8);
                    let layout = TypeLayout::of::<PatternHost>(&bits, PointerSize::Bit64);
                    assert_eq!(layout.storage_bytes(PointerSize::Bit64), Some(bytes));
                    assert_eq!(
                        access.store_layout(&mut memory, &pointer, &value, layout, None),
                        Err(Exhaustion {
                            budget: Budget::ExecutionBufferBytes,
                            limit: 16,
                            demand: bytes
                        })
                    );
                    assert_eq!(memory.known_run(at, 16, None), [0xa5; 16]);
                    assert_eq!(memory.load(private, 8, None), Loaded::Address(at));
                    assert!(!memory.stack_escaped());
                    assert_eq!(memory.refusal(), None);
                }
            }
        }
    }

    /// Oversized scalar and block footprints need no numeric image-address mapping.
    #[test]
    fn oversized_footprints_decline_before_address_mapping() {
        let context = AnalysisContext::new(MACHINE)
            .with_image(&NoStoreQueries)
            .with_budget(
                WorkBudget::DEFAULT
                    .with_limit(Budget::ExecutionBufferBytes, 16)
                    .with_limit(Budget::ExecutionBlockBytes, 16),
            );
        let access = Access::new(&context);
        let pointer = Value::<MockTarget>::Const(ConstValue::U64(0x1000));
        assert_eq!(access.footprint(&pointer, &MockType::V2I256, None), None);
        for size in [17, u64::MAX] {
            assert_eq!(
                access.block_footprint(&pointer, &Value::Const(ConstValue::U64(size))),
                None
            );
        }
    }

    /// A host whose only supported type inspection is its coherent layout projection.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct ProjectedHost;

    impl Target for ProjectedHost {
        type TypeRef = u32;
        type MethodRef = u32;
        type FieldRef = u32;
        type SigRef = u32;
        type SymbolRef = u32;
        type ExceptionKind = ();
        type Type = u32;
        type OriginalInstruction = ();
        type LocalSignature = ();
        type VariableFacts = ();
        type InstructionFacts = ();
        type FunctionFacts = ();
        type Capability = ();
        fn ptr_bytes(&self) -> u32 {
            8
        }
        fn synthetic_instruction() {}
        fn unknown_type() -> u32 {
            u32::MAX
        }
        fn instruction_mnemonic(_: &()) -> &'static str {
            "fixture"
        }
        fn instruction_rva(_: &()) -> u64 {
            0
        }
        fn handler_kind(_: &()) -> HandlerKind {
            HandlerKind::Catch
        }
        fn is_integer(_: &u32) -> bool {
            panic!("unrestricted integer query")
        }
        fn is_floating(_: &u32) -> bool {
            panic!("unrestricted floating query")
        }
        fn is_signed(_: &u32) -> bool {
            panic!("unrestricted signedness query")
        }
        fn is_pointer(_: &u32) -> bool {
            panic!("unrestricted pointer query")
        }
        fn is_reference(_: &u32) -> bool {
            panic!("unrestricted reference query")
        }
        fn is_unknown(_: &u32) -> bool {
            panic!("unrestricted unknown query")
        }
        fn bit_width(_: &u32) -> Option<u32> {
            panic!("unrestricted width query")
        }
        fn type_layout(depth: &u32, ptr_size: PointerSize) -> Option<TypeLayout> {
            (*depth != u32::MAX).then(|| TypeLayout::of::<MockTarget>(&MockType::U32, ptr_size))
        }
    }

    /// Loads, stores, footprints and proofs use projected layouts without raw type queries.
    #[test]
    fn memory_type_projection_supplies_coherent_byte_decoding() {
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let pointer = Value::<ProjectedHost>::Const(ConstValue::U64(0x1004));

        let access = Access::new(&context);
        let mut memory = Memory::default();
        assert_eq!(
            access
                .footprint::<ProjectedHost>(&pointer, &4096, None)
                .map(|(_, len)| len),
            Some(4)
        );
        assert_eq!(
            access
                .load_with_footprint(&memory, &pointer, &4096, None)
                .map(|read| read.value),
            Ok(Value::Const(ConstValue::U32(0x0403_0201)))
        );
        let private =
            Value::<ProjectedHost>::Addr(Address::new(Root::Stack, -8, PointerSize::Bit64));
        let held = Value::Const(ConstValue::U32(0x1234_5678));
        assert!(
            access
                .store(&mut memory, &private, &held, &4096, None)
                .is_ok()
        );
        assert_eq!(
            access
                .load_with_footprint(&memory, &private, &4096, None)
                .map(|read| read.value),
            Ok(held)
        );
        assert!(
            access
                .image_read_proof(&pointer, &0, AccessQualifiers::default())
                .is_some()
        );
    }

    /// Unsupported layouts yield no value without invoking raw type queries.
    #[test]
    fn memory_type_projection_refusal_has_no_unrestricted_fallback() {
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let pointer = Value::<ProjectedHost>::Const(ConstValue::U64(0x1004));
        let memory = Memory::default();

        assert_eq!(
            Access::new(&context)
                .load_with_footprint(&memory, &pointer, &u32::MAX, None)
                .map(|read| read.value),
            Ok(Value::Unknown)
        );
    }

    /// Read-only bytes at `0x1000`, and integers that equal symbols.
    struct Image;

    impl ImageData for Image {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, address: u64) -> Option<ImageRegion> {
            (0x1000..0x1008).contains(&address).then_some(ImageRegion {
                start: 0x1000,
                end: 0x1008,
                mutability: Mutability::ReadOnly,
            })
        }

        fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
            const BYTES: [u8; 8] = [0x3b, 0x2e, 0x2e, 0x3b, 1, 2, 3, 4];
            let start = usize::try_from(address.checked_sub(0x1000)?).ok()?;
            BYTES.get(start..start.checked_add(len)?)
        }

        fn number_offset(&self) -> Option<i64> {
            Some(0)
        }
    }

    fn stack(offset: i64) -> Value<MockTarget> {
        Value::Addr(Address::new(Root::Stack, offset, PointerSize::Bit64))
    }

    /// An integer pointer reads the image, in the image's byte order.
    #[test]
    fn an_integer_pointer_reads_the_image() {
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let access = Access::new(&context);
        let memory = Memory::default();
        let pointer = Value::<MockTarget>::Const(ConstValue::U64(0x1004));
        assert_eq!(
            access.load(&memory, &pointer, &MockType::U32, None),
            Value::Const(ConstValue::U32(0x0403_0201))
        );
    }

    /// **A store faults only where it reaches read-only image for
    /// certain.** A write the run places exactly faults where any byte it
    /// writes may not be written, as the machine faults it before the write
    /// completes. A store to some element of a span faults only where every
    /// byte of the span may not be written: one whose span also holds bytes
    /// the image does not deny may land on those, and refuses the execution
    /// instead.
    #[test]
    fn a_store_faults_only_where_it_reaches_read_only_image_for_certain() {
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let access = Access::new(&context);
        let at = |offset| Address::new(Root::Symbol, offset, PointerSize::Bit64);
        for (exact, start, fault, refusal) in [
            (true, 0x0ffc, Some((at(0x0ffc), 8)), None),
            (true, 0x1000, Some((at(0x1000), 8)), None),
            (false, 0x1000, Some((at(0x1000), 8)), None),
            (false, 0x0ffc, None, Some(MemoryRefusal::ReadOnlyImage)),
        ] {
            let mut memory = Memory::default();
            if exact {
                access.write_bytes(&mut memory, at(start), &[0; 8]);
            } else {
                access.store_element::<MockTarget>(&mut memory, at(start), 8, &[]);
            }
            assert_eq!(memory.fault(), fault, "exact={exact} start={start:#x}");
            assert_eq!(memory.refusal(), refusal, "exact={exact} start={start:#x}");
        }
    }

    /// Protects one already-entered source without exposing any image bytes.
    #[derive(Debug)]
    struct SourceGuard;

    impl MemoryWriteGuard for SourceGuard {
        fn check_write(
            &self,
            sources: &mut InstructionSources,
            address: Address,
            len: usize,
        ) -> Result<(), MemoryRefusal> {
            let start = address.image_address().ok_or(MemoryRefusal::AddressRange)?;
            let end = start
                .checked_add(u64::try_from(len).map_err(|_| MemoryRefusal::AddressRange)?)
                .ok_or(MemoryRefusal::AddressRange)?;
            if start < 0x1008 && 0x1000 < end {
                sources.record_write(InstructionSourceId::new(0))?;
            }
            Ok(())
        }
    }

    /// Rejects any attempt to consult a write guard for an empty operation.
    #[derive(Debug)]
    struct NoEmptyGuard;

    impl MemoryWriteGuard for NoEmptyGuard {
        fn check_write(
            &self,
            _: &mut InstructionSources,
            _: Address,
            _: usize,
        ) -> Result<(), MemoryRefusal> {
            panic!("an empty write must not consult the guard")
        }
    }

    /// Image protection is checked across the whole write before bytes or
    /// pointer escapes change: a write that reaches read-only image faults
    /// without writing any of its bytes.
    #[test]
    fn writes_cannot_cross_into_read_only_image_storage() {
        for guarded in [false, true] {
            let context = if guarded {
                AnalysisContext::new(MACHINE)
            } else {
                AnalysisContext::new(MACHINE).with_image(&Image)
            };
            let access = Access::new(&context).with_write_guard(guarded.then_some(&SourceGuard));
            let destination = Address::new(Root::Symbol, 0x0ffc, PointerSize::Bit64);
            let source = Address::new(Root::Symbol, 0x3000, PointerSize::Bit64);
            let pointer = Value::<MockTarget>::Addr(destination);
            let count = Value::Const(ConstValue::U64(16));
            for operation in 0..11 {
                let mut memory = Memory::default();
                memory.store_bytes(destination, &[0xaa; 16]);
                memory.store_bytes(source, &[0xbb; 16]);
                if guarded {
                    assert!(
                        memory
                            .instruction_sources_mut()
                            .enter(InstructionSourceId::new(0))
                    );
                }
                match operation {
                    0 => assert!(
                        access
                            .store(
                                &mut memory,
                                &pointer,
                                &Value::Const(ConstValue::of_lane(128, 0).unwrap()),
                                &MockType::U128,
                                None,
                            )
                            .is_ok()
                    ),
                    1 => assert!(
                        access
                            .store(
                                &mut memory,
                                &pointer,
                                &Value::Unknown,
                                &MockType::U128,
                                None,
                            )
                            .is_ok()
                    ),
                    2 => assert!(
                        access
                            .store(&mut memory, &pointer, &stack(-32), &MockType::Ptr, None)
                            .is_ok()
                    ),
                    3 => access.write_bytes(&mut memory, destination, &[0xcc; 16]),
                    4 => access.fill_byte(&mut memory, destination, 0xcc, 16),
                    5 => access.write_unknown(&mut memory, destination, 16),
                    6 => access.fill_storage(
                        &mut memory,
                        &pointer,
                        &Value::Const(ConstValue::U8(0)),
                        MockTarget::type_layout(&MockType::U8, PointerSize::Bit64)
                            .and_then(|layout| layout.storage_layout(PointerSize::Bit64)),
                        &count,
                        false,
                    ),
                    7 => {
                        access.copy(
                            &mut memory,
                            &pointer,
                            &Value::<MockTarget>::Addr(source),
                            &count,
                            true,
                        );
                    }
                    8 => {
                        access.copy_stream(
                            &mut memory,
                            &pointer,
                            &Value::<MockTarget>::Addr(source),
                            &count,
                            1,
                            false,
                        );
                    }
                    9 => {
                        access.copy_stream(
                            &mut memory,
                            &Value::<MockTarget>::Addr(
                                destination.moved(15, PointerSize::Bit64).unwrap(),
                            ),
                            &Value::<MockTarget>::Addr(
                                source.moved(15, PointerSize::Bit64).unwrap(),
                            ),
                            &count,
                            1,
                            true,
                        );
                    }
                    10 => {
                        access.copy_stream(
                            &mut memory,
                            &Value::<MockTarget>::Addr(
                                destination.moved(14, PointerSize::Bit64).unwrap(),
                            ),
                            &Value::<MockTarget>::Addr(
                                source.moved(14, PointerSize::Bit64).unwrap(),
                            ),
                            &count,
                            2,
                            true,
                        );
                    }
                    _ => unreachable!(),
                }
                // The guard refuses the write; the image's protection makes
                // it one the machine faults.
                assert_eq!(
                    memory.refusal(),
                    guarded.then_some(MemoryRefusal::InstructionSource),
                    "operation={operation}, guarded={guarded}"
                );
                assert_eq!(
                    memory.fault().is_some(),
                    !guarded,
                    "operation={operation}, guarded={guarded}"
                );
                assert_eq!(
                    memory.known_run(destination, 16, None),
                    [0xaa; 16],
                    "operation={operation}"
                );
                assert_eq!(memory.known_run(source, 16, None), [0xbb; 16]);
                assert!(!memory.stack_escaped());
            }
        }
    }

    /// An empty write never consults protection or escapes pointers.
    #[test]
    fn empty_writes_to_read_only_image_storage_have_no_effect() {
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let access = Access::new(&context).with_write_guard(Some(&NoEmptyGuard));
        let destination = Address::new(Root::Symbol, 0x1000, PointerSize::Bit64);
        let pointer = Value::<MockTarget>::Addr(destination);
        let zero = Value::Const(ConstValue::U64(0));
        let mut memory = Memory::default();
        memory.store_bytes(destination, &[0xaa; 8]);
        access.write_bytes(&mut memory, destination, &[]);
        access.fill_byte(&mut memory, destination, 0, 0);
        access.write_unknown(&mut memory, destination, 0);
        access.copy(
            &mut memory,
            &pointer,
            &Value::<MockTarget>::Unknown,
            &zero,
            true,
        );
        access.copy_stream(
            &mut memory,
            &pointer,
            &Value::<MockTarget>::Unknown,
            &zero,
            1,
            true,
        );
        access.fill_storage(&mut memory, &pointer, &stack(-32), None, &zero, true);
        assert_eq!(memory.refusal(), None);
        assert_eq!(memory.known_run(destination, 8, None), [0xaa; 8]);
        assert!(!memory.stack_escaped());
    }

    /// A possible atomic replacement retains the real heap extent without copying unrelated storage.
    #[test]
    fn possible_heap_store_uses_the_real_destination_extent() {
        let context = AnalysisContext::new(Machine {
            byte_order: Some(Endianness::Little),
            ..MACHINE
        });
        let access = Access::new(&context);
        let address = Address::new(Root::Heap(1), 4, PointerSize::Bit64);
        let unrelated = Address::new(Root::Heap(2), 999_999, PointerSize::Bit64);
        let mut memory = Memory::default();
        memory.allocate(Root::Heap(1), Some(16), None, None);
        memory.allocate(Root::Heap(2), Some(1_000_000), Some(0x55), None);
        memory.store_bytes(address, &[4, 3, 2, 1]);
        let layout = MockTarget::type_layout(&MockType::U32, PointerSize::Bit64).unwrap();
        assert!(
            access
                .maybe_store(
                    &mut memory,
                    address,
                    &Value::<MockTarget>::Const(ConstValue::U32(0x0102_0304)),
                    layout
                )
                .is_ok()
        );
        assert_eq!(memory.refusal(), None);
        assert_eq!(memory.known_run(address, 4, None), [4, 3, 2, 1]);
        assert_eq!(memory.known_run(unrelated, 1, None), [0x55]);
    }

    /// A value stored reads back as the type loaded.
    #[test]
    fn a_stored_value_reads_back() {
        let context = AnalysisContext::new(Machine {
            byte_order: Some(Endianness::Big),
            ..MACHINE
        });
        let access = Access::new(&context);
        let mut memory = Memory::default();
        let value = Value::<MockTarget>::Const(ConstValue::U32(0x6174_7461));
        assert!(
            access
                .store(&mut memory, &stack(-8), &value, &MockType::U32, None)
                .is_ok()
        );
        assert_eq!(
            access.load(&memory, &stack(-8), &MockType::U32, None),
            value
        );
        assert_eq!(
            access.load(&memory, &stack(-8), &MockType::U8, None),
            Value::Const(ConstValue::U8(0x61))
        );
    }

    /// A pointer-wide address round-trips; a narrower store of one does not.
    #[test]
    fn an_address_round_trips_only_pointer_wide() {
        let context = AnalysisContext::new(MACHINE);
        let access = Access::new(&context);
        let mut memory = Memory::default();
        assert!(
            access
                .store(&mut memory, &stack(-16), &stack(-48), &MockType::Ptr, None)
                .is_ok()
        );
        assert_eq!(
            access.load(&memory, &stack(-16), &MockType::Ptr, None),
            stack(-48)
        );
        assert!(
            access
                .store(&mut memory, &stack(-24), &stack(-48), &MockType::U32, None)
                .is_ok()
        );
        assert_eq!(
            access.load(&memory, &stack(-24), &MockType::U32, None),
            Value::Unknown
        );
    }

    /// A store through an unknown pointer leaves only the private stack, and a
    /// stack address stored where nothing follows it escapes the stack.
    #[test]
    fn an_unknown_store_clobbers_and_escapes() {
        let context = AnalysisContext::new(MACHINE);
        let access = Access::new(&context);
        let mut memory = Memory::default();
        let byte = Value::<MockTarget>::Const(ConstValue::U8(7));
        assert!(
            access
                .store(&mut memory, &stack(-1), &byte, &MockType::U8, None)
                .is_ok()
        );
        assert!(
            access
                .store(&mut memory, &Value::Unknown, &byte, &MockType::U8, None)
                .is_ok()
        );
        assert_eq!(access.load(&memory, &stack(-1), &MockType::U8, None), byte);
        assert!(
            access
                .store(
                    &mut memory,
                    &Value::Unknown,
                    &stack(-1),
                    &MockType::Ptr,
                    None,
                )
                .is_ok()
        );
        assert!(memory.stack_escaped());
    }

    /// A segment-qualified load is outside the flat space.
    #[test]
    fn a_segment_load_is_unknown() {
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let access = Access::new(&context);
        let pointer = Value::<MockTarget>::Const(ConstValue::U64(0x1000));
        assert_eq!(
            access.load(
                &Memory::default(),
                &pointer,
                &MockType::U8,
                NonZeroU16::new(1)
            ),
            Value::Unknown
        );
    }

    /// `memcpy` copies known bytes and `memset` fills; an unknown size writes
    /// nothing it can name and forgets what it may have reached.
    #[test]
    fn blocks_copy_and_fill() {
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let access = Access::new(&context);
        let mut memory = Memory::default();
        let source = Value::<MockTarget>::Const(ConstValue::U64(0x1000));
        let four = Value::Const(ConstValue::U64(4));
        access.copy(&mut memory, &stack(-8), &source, &four, false);
        assert_eq!(
            memory.known_run(Address::new(Root::Stack, -8, PointerSize::Bit64), 8, None),
            vec![0x3b, 0x2e, 0x2e, 0x3b]
        );
        access.fill_byte(
            &mut memory,
            Address::new(Root::Stack, -4, PointerSize::Bit64),
            0,
            4,
        );
        assert_eq!(
            memory.known_run(Address::new(Root::Stack, -4, PointerSize::Bit64), 8, None),
            vec![0, 0, 0, 0]
        );
        // A wider element repeats whole: `rep stosw` of 0x1234 over six bytes.
        let six = Value::Const(ConstValue::U64(6));
        access.fill_storage(
            &mut memory,
            &stack(-16),
            &Value::Const(ConstValue::U16(0x1234)),
            TypeLayout::of::<MockTarget>(&MockType::U16, PointerSize::Bit64)
                .storage_layout(PointerSize::Bit64),
            &six,
            false,
        );
        assert_eq!(
            memory.known_run(Address::new(Root::Stack, -16, PointerSize::Bit64), 6, None),
            vec![0x34, 0x12, 0x34, 0x12, 0x34, 0x12]
        );
    }

    /// Copying a block one byte above itself: forward (`memcpy`, `rep
    /// movsb`) repeats the first byte down the block, buffered (`memmove`)
    /// shifts the block intact.
    #[test]
    fn an_overlapping_copy_repeats_forward_and_shifts_buffered() {
        let image = Image;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let access = Access::new(&context);
        let three = Value::Const(ConstValue::U64(3));
        let block = Address::new(Root::Stack, -8, PointerSize::Bit64);
        for (buffered, expected) in [(false, [1, 1, 1, 1]), (true, [1, 1, 2, 3])] {
            let mut memory = Memory::default();
            memory.store_bytes(block, &[1, 2, 3, 4]);
            access.copy(&mut memory, &stack(-7), &stack(-8), &three, buffered);
            assert_eq!(
                memory.known_run(block, 4, None),
                expected,
                "buffered {buffered}"
            );
        }
    }
    /// Block transfers preserve symbolic pointers, even when copied in pieces.
    #[test]
    fn block_copy_retains_pointer_pieces() {
        let context = AnalysisContext::new(MACHINE);
        let access = Access::new(&context);
        for (buffered, pointer_root) in [
            (false, Root::Heap(7)),
            (true, Root::Heap(7)),
            (false, Root::Stack),
            (true, Root::Stack),
        ] {
            let mut memory = Memory::default();
            let source = Address::new(Root::Stack, -32, PointerSize::Bit64);
            let dest = Address::new(Root::Stack, -16, PointerSize::Bit64);
            let pointer = Address::new(pointer_root, 3, PointerSize::Bit64);
            memory.store_address(source, pointer, 8);
            for delta in [0, 4] {
                access.copy(
                    &mut memory,
                    &stack(-16 + delta),
                    &stack(-32 + delta),
                    &Value::Const(ConstValue::I64(4)),
                    buffered,
                );
            }
            assert_eq!(memory.load(dest, 8, None), Loaded::Address(pointer));
            // Overwriting one fragment must prevent reconstructing the pointer.
            memory.store_bytes(dest.moved(2, PointerSize::Bit64).unwrap_or(dest), &[0]);
            assert_eq!(
                memory.load(dest, 8, None),
                if pointer_root == Root::Stack {
                    Loaded::StackDerived
                } else {
                    Loaded::Unknown
                }
            );
        }
    }

    /// Wider numeric addresses must not truncate sign extension or loader displacement.
    #[test]
    fn wide_numeric_address_conversion_respects_the_target_domain() {
        let context = AnalysisContext::new(Machine {
            ptr_size: PointerSize::Bit128,
            ..MACHINE
        });
        let access = Access::new(&context);
        assert_eq!(
            access.address_of::<MockTarget>(&Value::Const(ConstValue::I64(-1)), None),
            None
        );
        assert_eq!(
            access.address_of::<MockTarget>(&Value::Const(ConstValue::NativeInt(-1)), None),
            None
        );
        assert_eq!(
            access.address_of::<MockTarget>(&Value::Const(ConstValue::U64(u64::MAX)), None),
            Some(Address::new(Root::Number, -1, PointerSize::Bit128))
        );
        let wide = |raw| Value::<MockTarget>::Const(ConstValue::of_lane(128, raw).unwrap());
        assert_eq!(
            access.address_of(&wide(u128::from(u64::MAX)), None),
            Some(Address::new(Root::Number, -1, PointerSize::Bit128))
        );
        assert_eq!(access.address_of(&wide(1u128 << 64), None), None);
        let image = ProofImage {
            shift: Some(-1),
            mutability: Mutability::ReadOnly,
            complete: true,
        };
        let context = context.with_image(&image);
        let shifted = Access::new(&context);
        assert_eq!(
            shifted.address_of(&wide(1u128 << 64), None),
            Some(Address::new(Root::Symbol, -1, PointerSize::Bit128))
        );
        assert_eq!(shifted.address_of(&wide(0), None), None);
        let image = ProofImage {
            shift: Some(1),
            mutability: Mutability::ReadOnly,
            complete: true,
        };
        let context = AnalysisContext::new(context.machine()).with_image(&image);
        assert_eq!(
            Access::new(&context).address_of(&wide(u128::MAX), None),
            Some(Address::new(Root::Symbol, 0, PointerSize::Bit128))
        );
    }

    /// Image replacement requires actual alignment, permissions, complete bytes and a fixed extent.
    #[test]
    fn image_replacement_evidence_requires_the_complete_access_contract() {
        let context = AnalysisContext::new(MACHINE).with_image(&Image);
        let access = Access::new(&context);
        let ordinary = AccessQualifiers::default();
        let symbol =
            |at| Value::<MockTarget>::Addr(Address::new(Root::Symbol, at, PointerSize::Bit64));
        assert_eq!(
            access.image_read_proof(&symbol(0x1000), &MockType::U16, ordinary),
            Some(ReplacementProof::ReadOnlyImage {
                address: 0x1000,
                len: 2
            })
        );
        for (address, ty) in [
            (0x1001, MockType::U16),
            (0x1008, MockType::U8),
            (0x1000, MockType::F80),
            (0x1000, MockType::U128),
            (0x1000, MockType::Unknown),
        ] {
            assert_eq!(
                access.image_read_proof(&symbol(address), &ty, ordinary),
                None
            );
        }
        assert_eq!(
            access.image_read_proof(
                &symbol(0x1000),
                &MockType::U8,
                AccessQualifiers {
                    address_space: NonZeroU16::new(1),
                    ..ordinary
                }
            ),
            None
        );
        assert_eq!(
            access.image_read_proof(&stack(-8), &MockType::U8, ordinary),
            None
        );
        assert_eq!(
            access.image_read_proof::<MockTarget>(&Value::Unknown, &MockType::U8, ordinary),
            None
        );
        let no_image = AnalysisContext::new(MACHINE);
        assert_eq!(
            Access::new(&no_image).image_read_proof(&symbol(0x1000), &MockType::U8, ordinary),
            None
        );
        for (shift, mutability, complete) in [
            (None, Mutability::ReadOnly, true),
            (Some(1), Mutability::ReadOnly, true),
            (Some(0), Mutability::Writable, true),
            (Some(0), Mutability::ReadOnly, false),
        ] {
            let image = ProofImage {
                shift,
                mutability,
                complete,
            };
            let context = AnalysisContext::new(MACHINE).with_image(&image);
            assert_eq!(
                Access::new(&context).image_read_proof(&symbol(0x1000), &MockType::U16, ordinary),
                None
            );
        }
    }

    /// Alignment uses physical coordinates and checks every element start.
    #[test]
    fn alignment_spans_require_coordinates_and_compatible_strides() {
        let ordinary = AccessQualifiers::default();
        let two = AccessQualifiers {
            alignment: NonZeroU8::new(2),
            ..ordinary
        };
        let byte = AccessQualifiers {
            alignment: NonZeroU8::new(1),
            ..ordinary
        };
        for ptr_size in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            for shift in [None, Some(-1), Some(0), Some(1)] {
                let image = ProofImage {
                    shift,
                    mutability: Mutability::ReadOnly,
                    complete: true,
                };
                let context = AnalysisContext::new(Machine {
                    ptr_size,
                    ..MACHINE
                })
                .with_image(&image);
                let access = Access::new(&context);
                for offset in [-16, -1, 0, 1, 2, 3, 4, 255, 256] {
                    let number = Address::new(Root::Number, offset, ptr_size);
                    let symbol = Address::new(Root::Symbol, offset, ptr_size);
                    assert_eq!(access.aligned_span(number, 8, 4, ordinary), offset % 4 == 0);
                    assert_eq!(
                        access.aligned_span(symbol, 8, 4, ordinary),
                        shift.is_some_and(|shift| (offset - shift) % 4 == 0)
                    );
                    for root in [Root::Stack, Root::StackFrame(1), Root::Heap(1)] {
                        let address = Address::new(root, offset, ptr_size);
                        assert!(!access.aligned_span(address, 4, 4, ordinary));
                        assert!(access.aligned_span(address, 4, 4, byte));
                        assert!(access.aligned_span(address, 0, 4, ordinary));
                        assert!(!access.aligned_span(address, 0, 0, ordinary));
                    }
                }
                let even = Address::new(Root::Number, 2, ptr_size);
                assert!(access.aligned_span(even, 3, 3, two));
                assert!(!access.aligned_span(even, 6, 3, two));
                assert!(!access.aligned_span(even, 3, 3, ordinary));
                assert!(access.aligned_span(even, 6, 3, byte));
                assert!(!access.aligned_span(
                    even,
                    2,
                    2,
                    AccessQualifiers {
                        address_space: NonZeroU16::new(1),
                        ..two
                    }
                ));
            }
        }
    }

    /// **The machine decides an ordinary access, and an access stating the
    /// alignment it requires is held to it on any machine.** A misaligned
    /// ordinary span is proven where misaligned accesses complete (x86) and
    /// not where they may fault (MIPS); a span requiring 16 bytes (`movaps`)
    /// is proven only at a 16-byte-aligned address, and never from a root
    /// that establishes no alignment, on either.
    #[test]
    fn the_machine_decides_an_ordinary_span_and_a_requirement_holds_everywhere() {
        let ordinary = AccessQualifiers::default();
        let sixteen = ordinary.requiring_alignment(NonZeroU8::new(16).expect("nonzero"));
        let at = |root, offset| Address::new(root, offset, PointerSize::Bit64);
        for (rule, tolerated) in [
            (MisalignedAccess::Faults, false),
            (MisalignedAccess::Tolerated, true),
        ] {
            let context = AnalysisContext::new(MACHINE.with_misaligned_access(rule));
            let access = Access::new(&context);
            assert_eq!(
                access.aligned_span(at(Root::Number, 3), 8, 8, ordinary),
                tolerated,
                "{rule:?}"
            );
            assert_eq!(
                access.aligned_span(at(Root::Heap(1), 3), 8, 8, ordinary),
                tolerated,
                "{rule:?}"
            );
            assert!(access.aligned_span(at(Root::Number, 32), 8, 8, ordinary));
            assert!(access.aligned_span(at(Root::Number, 32), 16, 16, sixteen));
            for misaligned in [3, 8] {
                assert!(
                    !access.aligned_span(at(Root::Number, misaligned), 16, 16, sixteen),
                    "{rule:?}: {misaligned}"
                );
            }
            assert!(!access.aligned_span(at(Root::Heap(1), 0), 16, 16, sixteen));
        }
    }

    /// **A misaligned read-only image load is replaced where the machine
    /// tolerates it, and kept where it may fault or the access requires its
    /// alignment.**
    #[test]
    fn a_misaligned_image_load_is_replaced_only_where_it_cannot_fault() {
        let image = ProofImage {
            shift: Some(0),
            mutability: Mutability::ReadOnly,
            complete: true,
        };
        let odd = Value::<MockTarget>::Addr(Address::new(Root::Symbol, 0x1001, PointerSize::Bit64));
        let ordinary = AccessQualifiers::default();
        let required = ordinary.requiring_alignment(NonZeroU8::new(4).expect("nonzero"));
        for (rule, replaced) in [
            (MisalignedAccess::Faults, false),
            (MisalignedAccess::Tolerated, true),
        ] {
            let context =
                AnalysisContext::new(MACHINE.with_misaligned_access(rule)).with_image(&image);
            let access = Access::new(&context);
            assert_eq!(
                access
                    .image_read_proof(&odd, &MockType::U32, ordinary)
                    .is_some(),
                replaced,
                "{rule:?}"
            );
            assert!(
                access
                    .image_read_proof(&odd, &MockType::U32, required)
                    .is_none(),
                "{rule:?}: an access requiring its alignment"
            );
        }
    }

    /// Explicit weaker alignment permits only complete ordinary read-only image loads.
    #[test]
    fn image_replacement_honours_explicit_weaker_alignment() {
        let ordinary = AccessQualifiers::default();
        for shift in [None, Some(0), Some(1)] {
            let image = ProofImage {
                shift,
                mutability: Mutability::ReadOnly,
                complete: true,
            };
            let context = AnalysisContext::new(MACHINE).with_image(&image);
            let access = Access::new(&context);
            let odd =
                Value::<MockTarget>::Addr(Address::new(Root::Symbol, 0x1001, PointerSize::Bit64));
            for ty in [MockType::U16, MockType::U32] {
                assert!(
                    access
                        .image_read_proof(
                            &odd,
                            &ty,
                            AccessQualifiers {
                                alignment: NonZeroU8::new(1),
                                ..ordinary
                            }
                        )
                        .is_some()
                );
                assert_eq!(
                    access.image_read_proof(&odd, &ty, ordinary).is_some(),
                    shift == Some(1)
                );
            }
            let even =
                Value::<MockTarget>::Addr(Address::new(Root::Symbol, 0x1002, PointerSize::Bit64));
            assert_eq!(
                access
                    .image_read_proof(
                        &even,
                        &MockType::U32,
                        AccessQualifiers {
                            alignment: NonZeroU8::new(2),
                            ..ordinary
                        }
                    )
                    .is_some(),
                shift == Some(0)
            );
            assert!(
                access
                    .image_read_proof(&even, &MockType::U32, ordinary)
                    .is_none()
            );
        }
    }

    /// An image oracle whose offset, permission and returned bytes are independently varied.
    struct ProofImage {
        shift: Option<i64>,
        mutability: Mutability,
        complete: bool,
    }

    impl ImageData for ProofImage {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }
        fn number_offset(&self) -> Option<i64> {
            self.shift
        }
        fn region(&self, address: u64) -> Option<ImageRegion> {
            Image.region(address).map(|region| ImageRegion {
                mutability: self.mutability,
                ..region
            })
        }
        fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
            if self.complete {
                Image.bytes(address, len)
            } else {
                Some(&[])
            }
        }
    }
    /// Native stores preserve sign extension through 128 bits and spare adjacent bytes.
    #[test]
    fn wide_native_stores_use_the_full_signed_pattern() {
        for order in [Endianness::Little, Endianness::Big] {
            let context = AnalysisContext::new(Machine {
                ptr_size: PointerSize::Bit128,
                byte_order: Some(order),
                ..MACHINE
            });
            let access = Access::new(&context);
            let address = Address::new(Root::Stack, -32, PointerSize::Bit128);
            for value in [i64::MIN, -1, 0, i64::MAX] {
                let mut memory = Memory::default();
                let before = Address::new(Root::Stack, -33, PointerSize::Bit128);
                let after = Address::new(Root::Stack, -16, PointerSize::Bit128);
                memory.store_bytes(before, &[0xaa]);
                memory.store_bytes(after, &[0xbb]);
                let held = Value::<MockTarget>::Const(ConstValue::NativeInt(value));
                assert!(
                    access
                        .store(
                            &mut memory,
                            &Value::Addr(address),
                            &held,
                            &MockType::Ptr,
                            None,
                        )
                        .is_ok()
                );
                let expected = match order {
                    Endianness::Little => i128::from(value).to_le_bytes(),
                    Endianness::Big => i128::from(value).to_be_bytes(),
                };
                assert_eq!(memory.known_run(address, 16, None), expected);
                assert_eq!(memory.known_run(before, 1, None), [0xaa]);
                assert_eq!(memory.known_run(after, 1, None), [0xbb]);
            }
        }
    }

    /// A mismatched constant writes only the declared unknown width.
    #[test]
    fn mismatched_constant_encoding_spares_adjacent_memory() {
        let context = AnalysisContext::new(MACHINE);

        let access = Access::new(&context);
        let value = Value::<MockTarget>::Const(ConstValue::Vector(
            VectorConst::splat(8, 256, 0x5a).unwrap(),
        ));
        let address = Address::new(Root::Stack, -512, PointerSize::Bit64);
        let mut memory = Memory::default();
        memory.store_bytes(address, &[0xaa; 257]);
        assert!(
            access
                .store(
                    &mut memory,
                    &Value::Addr(address),
                    &value,
                    &MockType::U8,
                    None,
                )
                .is_ok()
        );
        assert_eq!(memory.load(address, 1, None), Loaded::Unknown);
        assert_eq!(
            memory.known_run(address.moved(1, PointerSize::Bit64).unwrap(), 256, None),
            [0xaa; 256]
        );
    }

    /// Wide vector patterns preserve lane byte order and both full and partial repetitions.
    #[test]
    fn typed_fill_wide_vectors_keep_lane_order_and_guards() {
        let lanes = [
            0x0123_4567u128,
            0x89ab_cdef,
            0x1122_3344,
            0x5566_7788,
            0xaabb_ccdd,
            0xeeff_0099,
            0x7654_3210,
            0xfedc_ba98,
        ];
        for order in [Endianness::Little, Endianness::Big] {
            for reverse in [false, true] {
                let context = AnalysisContext::new(Machine {
                    byte_order: Some(order),
                    ..MACHINE
                });
                let access = Access::new(&context);
                let start = Address::new(Root::Stack, -512, PointerSize::Bit64);
                let len = if reverse { 96 } else { 99 };
                let first = start
                    .moved(if reverse { 64 } else { 0 }, PointerSize::Bit64)
                    .unwrap();
                let mut memory = Memory::default();
                memory.store_bytes(start, &[0xcc; 128]);
                let value = Value::<MockTarget>::Const(ConstValue::Vector(
                    VectorConst::from_lanes(32, lanes).unwrap(),
                ));
                access.fill_storage(
                    &mut memory,
                    &Value::Addr(first),
                    &value,
                    TypeLayout::of::<PatternHost>(&256, PointerSize::Bit64)
                        .storage_layout(PointerSize::Bit64),
                    &Value::Const(ConstValue::U64(len as u64)),
                    reverse,
                );
                let mut pattern = Vec::new();
                for lane in lanes {
                    for index in 0..4 {
                        let byte = if order == Endianness::Little {
                            index
                        } else {
                            3 - index
                        };
                        pattern.push((lane >> (byte * 8)) as u8);
                    }
                }
                let mut expected = vec![0xcc; 128];
                for index in 0..len {
                    expected[index] = pattern[index % 32];
                }
                assert_eq!(
                    memory.known_run(start, 128, None),
                    expected,
                    "{order:?} reverse={reverse}"
                );
            }
        }
    }

    /// Floating fills preserve exact NaN payload bits without numeric conversion.
    #[test]
    fn typed_fill_floating_payloads_keep_storage_bits() {
        for (ty, raw, width) in [
            (MockType::BF16, 0xffc1u128, 2usize),
            (MockType::F64, 0x7ff8_0123_4567_89ab, 8),
            (MockType::F80, 0xffff_c123_4567_89ab_cdef, 10),
        ] {
            for order in [Endianness::Little, Endianness::Big] {
                for reverse in [false, true] {
                    let context = AnalysisContext::new(Machine {
                        byte_order: Some(order),
                        ..MACHINE
                    });
                    let access = Access::new(&context);
                    let start = Address::new(Root::Stack, -256, PointerSize::Bit64);
                    let len = width * 2 + usize::from(!reverse);
                    let first = start
                        .moved(if reverse { width as i64 } else { 0 }, PointerSize::Bit64)
                        .unwrap();
                    let mut memory = Memory::default();
                    memory.store_bytes(start, &[0xcc; 64]);
                    let value = Value::<MockTarget>::Const(ConstValue::Float(
                        FloatBits::new(MockTarget::float_format(&ty).unwrap(), raw).unwrap(),
                    ));
                    access.fill_storage(
                        &mut memory,
                        &Value::Addr(first),
                        &value,
                        TypeLayout::of::<MockTarget>(&ty, PointerSize::Bit64)
                            .storage_layout(PointerSize::Bit64),
                        &Value::Const(ConstValue::U64(len as u64)),
                        reverse,
                    );
                    let mut expected = vec![0xcc; 64];
                    for (index, output) in expected.iter_mut().take(len).enumerate() {
                        let byte = if order == Endianness::Little {
                            index % width
                        } else {
                            width - 1 - index % width
                        };
                        *output = (raw >> (byte * 8)) as u8;
                    }
                    assert_eq!(
                        memory.known_run(start, 64, None),
                        expected,
                        "{ty:?} {order:?} reverse={reverse}"
                    );
                }
            }
        }
    }

    /// Pointer repetitions retain full-width piece identities even in a final partial element.
    #[test]
    fn typed_fill_pointer_pieces_repeat_and_preserve_partial_tails() {
        for ptr_size in [PointerSize::Bit32, PointerSize::Bit64, PointerSize::Bit128] {
            for order in [Endianness::Little, Endianness::Big] {
                for public in [false, true] {
                    for stack_value in [false, true] {
                        for (reverse, partial) in [(false, false), (false, true), (true, false)] {
                            let context = AnalysisContext::new(Machine::new(
                                ptr_size,
                                Some(order),
                                Default::default(),
                            ));
                            let access = Access::new(&context);
                            let width = ptr_size.bytes();
                            let len = width * 2 + usize::from(partial);
                            let start = Address::new(
                                if public { Root::Number } else { Root::Stack },
                                if public { 0x2000 } else { -256 },
                                ptr_size,
                            );
                            let first = start
                                .moved(if reverse { width as i64 } else { 0 }, ptr_size)
                                .unwrap();
                            let stored = Address::new(
                                if stack_value {
                                    Root::Stack
                                } else {
                                    Root::Number
                                },
                                if stack_value { -8 } else { 0x1234 },
                                ptr_size,
                            );
                            let mut memory = Memory::default();
                            memory.store_bytes(start, &[0xcc; 128]);
                            let before = start.moved(-1, ptr_size).unwrap();
                            memory.store_bytes(before, &[0x96]);
                            access.fill_storage(
                                &mut memory,
                                &Value::<MockTarget>::Addr(first),
                                &Value::Addr(stored),
                                TypeLayout::of::<PatternHost>(&ptr_size.bits(), ptr_size)
                                    .storage_layout(ptr_size),
                                &Value::Const(ConstValue::U64(len as u64)),
                                reverse,
                            );
                            for offset in [0, width] {
                                assert_eq!(
                                    memory.load(
                                        start.moved(offset as i64, ptr_size).unwrap(),
                                        width,
                                        None
                                    ),
                                    Loaded::Address(stored)
                                );
                            }
                            assert_eq!(memory.known_run(before, 1, None), [0x96]);
                            assert_eq!(
                                memory.known_run(
                                    start.moved(len as i64, ptr_size).unwrap(),
                                    1,
                                    None
                                ),
                                [0xcc]
                            );
                            assert_eq!(
                                memory.holds_privately(Address::new(Root::Stack, -8, ptr_size)),
                                !public || !stack_value
                            );
                            if partial {
                                let tail = start.moved((width * 2) as i64, ptr_size).unwrap();
                                let scratch = start.moved(64, ptr_size).unwrap();
                                assert_eq!(
                                    memory.load(tail, width, None),
                                    if stack_value {
                                        Loaded::StackDerived
                                    } else {
                                        Loaded::Unknown
                                    }
                                );
                                memory.store_address(scratch, stored, width as u8);
                                memory.copy(
                                    tail.moved(1, ptr_size).unwrap(),
                                    scratch.moved(1, ptr_size).unwrap(),
                                    (width - 1) as i64,
                                    None,
                                    true,
                                );
                                assert_eq!(
                                    memory.load(tail, width, None),
                                    Loaded::Address(stored),
                                    "partial pointer pieces {ptr_size:?} public={public} stack={stack_value}"
                                );
                            }
                            assert_eq!(memory.refusal(), None);
                        }
                    }
                }
            }
        }
    }

    /// A pointer fill running past its block is made as the machine makes it:
    /// an access past the block's extent, and every element written.
    #[test]
    fn typed_fill_past_its_block_is_found_and_written() {
        let context = AnalysisContext::new(MACHINE);
        let access = Access::new(&context);
        let root = Root::Heap(1);
        let mut memory = Memory::default();
        memory.allocate(root, Some(12), Some(0xcc), None);
        let start = Address::new(root, 0, PointerSize::Bit64);
        let stored = Address::new(Root::Number, 0x1234, PointerSize::Bit64);
        access.fill_storage(
            &mut memory,
            &Value::<MockTarget>::Addr(start),
            &Value::Addr(stored),
            TypeLayout::of::<MockTarget>(&MockType::Ptr, PointerSize::Bit64)
                .storage_layout(PointerSize::Bit64),
            &Value::Const(ConstValue::U64(16)),
            false,
        );
        assert_eq!(memory.refusal(), None);
        assert_eq!(
            memory.access_finding(start, 16),
            Some(FindingKind::PastExtent(ExtentSource::Allocator))
        );
        assert_eq!(memory.load(start, 8, None), Loaded::Address(stored));
    }

    /// Reverse streams require whole elements while forward fills can end on a prefix.
    #[test]
    fn typed_fill_reverse_partial_element_refuses_without_writing() {
        let context = AnalysisContext::new(MACHINE);
        let access = Access::new(&context);
        let start = Address::new(Root::Stack, -64, PointerSize::Bit64);
        let mut memory = Memory::default();
        memory.store_bytes(start, &[0xcc; 16]);
        access.fill_storage(
            &mut memory,
            &Value::<MockTarget>::Addr(start),
            &Value::Const(ConstValue::U16(0x1234)),
            TypeLayout::of::<MockTarget>(&MockType::U16, PointerSize::Bit64)
                .storage_layout(PointerSize::Bit64),
            &Value::Const(ConstValue::U64(3)),
            true,
        );
        assert_eq!(memory.refusal(), Some(MemoryRefusal::BlockLayout));
        assert_eq!(memory.known_run(start, 16, None), [0xcc; 16]);
    }

    /// An untracked nonempty store can publish stack-derived bits even without a layout.
    #[test]
    fn typed_fill_unplaced_stores_publish_stack_values() {
        let context = AnalysisContext::new(MACHINE);
        let access = Access::new(&context);
        let stack = Address::new(Root::Stack, -8, PointerSize::Bit64);
        let public = Address::new(Root::Number, 0x2000, PointerSize::Bit64);
        let storage = TypeLayout::of::<MockTarget>(&MockType::Ptr, PointerSize::Bit64)
            .storage_layout(PointerSize::Bit64);
        for value in [Value::<MockTarget>::Addr(stack), Value::StackDerived]
            .into_iter()
            .chain(partial_stack_values())
        {
            for (pointer, layout) in [(Value::Unknown, storage), (Value::Addr(public), None)] {
                let mut memory = Memory::default();
                memory.store_bytes(stack, &[0x41]);
                access.fill_storage(
                    &mut memory,
                    &pointer,
                    &value,
                    layout,
                    &Value::Const(ConstValue::U64(8)),
                    false,
                );
                assert!(!memory.holds_privately(stack));
                assert_eq!(memory.load(stack, 1, None), Loaded::Unknown);
            }
        }
    }

    /// Empty fills cannot publish values or discard bytes through a lost address.
    #[test]
    fn typed_fill_empty_spans_keep_private_values() {
        let context = AnalysisContext::new(MACHINE);
        let access = Access::new(&context);
        let stack = Address::new(Root::Stack, -8, PointerSize::Bit64);
        let public = Address::new(Root::Number, 0x2000, PointerSize::Bit64);
        let storage = TypeLayout::of::<MockTarget>(&MockType::Ptr, PointerSize::Bit64)
            .storage_layout(PointerSize::Bit64);
        for reverse in [false, true] {
            for value in [Value::<MockTarget>::Addr(stack), Value::StackDerived]
                .into_iter()
                .chain(partial_stack_values())
            {
                for (pointer, layout) in [
                    (Value::Unknown, storage),
                    (Value::StackDerived, storage),
                    (Value::Addr(public), None),
                ] {
                    let mut memory = Memory::default();
                    memory.store_bytes(stack, &[0x41]);
                    access.fill_storage(
                        &mut memory,
                        &pointer,
                        &value,
                        layout,
                        &Value::Const(ConstValue::U64(0)),
                        reverse,
                    );
                    assert!(memory.holds_privately(stack));
                    assert_eq!(memory.load(stack, 1, None), Loaded::Bytes(vec![0x41]));
                    assert_eq!(memory.refusal(), None);
                }
            }
        }
    }

    /// Empty transfers preserve public and private bytes without spending a memory budget.
    #[test]
    fn streamed_block_zero_access_transfers_do_not_resolve_addresses() {
        let context = AnalysisContext::new(MACHINE);
        let public = Address::new(Root::Number, 0x1000, PointerSize::Bit64);
        let private = Address::new(Root::Stack, -8, PointerSize::Bit64);
        for mode in 0..4 {
            for pointer in [
                Value::<MockTarget>::Unknown,
                Value::StackDerived,
                Value::Addr(public),
            ] {
                for source in [Value::Unknown, Value::StackDerived, Value::Addr(private)]
                    .into_iter()
                    .chain(partial_stack_values())
                {
                    let access = Access::new(&context);
                    let mut memory = Memory::default();
                    memory.store_bytes(public, &[0x96]);
                    memory.store_bytes(private, &[0xa5]);
                    let size = Value::Const(ConstValue::U64(0));
                    if mode < 2 {
                        access.copy(&mut memory, &pointer, &source, &size, mode == 1);
                    } else {
                        access.copy_stream(&mut memory, &pointer, &source, &size, 16, mode == 3);
                    }
                    assert_eq!(memory.known_run(public, 1, None), [0x96], "mode={mode}");
                    assert_eq!(memory.known_run(private, 1, None), [0xa5]);
                    assert!(memory.holds_privately(private));
                    assert_eq!(memory.refusal(), None);
                }
            }
        }
    }

    /// A malformed zero-width descriptor is refused even when it requests no bytes.
    #[test]
    fn streamed_block_zero_count_retains_element_width_validation() {
        let context = AnalysisContext::new(MACHINE);
        let access = Access::new(&context);
        let address = Address::new(Root::Stack, -8, PointerSize::Bit64);
        let mut memory = Memory::default();
        memory.store_bytes(address, &[0x96]);
        access.copy_stream(
            &mut memory,
            &Value::<MockTarget>::Addr(address),
            &Value::Addr(address),
            &Value::Const(ConstValue::U64(0)),
            0,
            false,
        );
        assert_eq!(memory.refusal(), Some(MemoryRefusal::BlockLayout));
        assert_eq!(memory.known_run(address, 1, None), [0x96]);
    }

    /// Every element width repeats in target order and can end partway through an element.
    #[test]
    fn streamed_fills_preserve_element_order_and_partial_tails() {
        for order in [Endianness::Little, Endianness::Big] {
            let context = AnalysisContext::new(Machine {
                byte_order: Some(order),
                ..MACHINE
            });
            let access = Access::new(&context);
            for width in 1..=16usize {
                for raw in [0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128, u128::MAX - 1] {
                    for len in [0, 1, width - 1, width, width + 1, width * 3 + 1] {
                        let mut memory = Memory::default();
                        let address = Address::new(Root::Stack, -256, PointerSize::Bit64);
                        memory.store_bytes(address, &[0xcc; 64]);
                        access.fill_storage(
                            &mut memory,
                            &Value::<MockTarget>::Addr(address),
                            &Value::Const(ConstValue::of_lane((width * 8) as u32, raw).unwrap()),
                            TypeLayout::of::<PatternHost>(
                                &((width * 8) as u32),
                                PointerSize::Bit64,
                            )
                            .storage_layout(PointerSize::Bit64),
                            &Value::Const(ConstValue::U64(len as u64)),
                            false,
                        );
                        let expected: Vec<u8> = (0..len)
                            .map(|index| {
                                let byte = index % width;
                                let byte = if order == Endianness::Big {
                                    width - 1 - byte
                                } else {
                                    byte
                                };
                                (raw >> (byte * 8)) as u8
                            })
                            .collect();
                        assert_eq!(memory.known_run(address, len, None), expected);
                        assert_eq!(
                            memory.known_run(
                                address.moved(len as i64, PointerSize::Bit64).unwrap(),
                                1,
                                None
                            ),
                            [0xcc]
                        );
                    }
                }
            }
        }
    }

    /// Unknown byte order allows byte fills but establishes no multibyte pattern.
    #[test]
    fn streamed_fills_require_their_element_byte_order() {
        let context = AnalysisContext::new(Machine {
            byte_order: None,
            ..MACHINE
        });
        let access = Access::new(&context);
        for width in [1, 2] {
            let mut memory = Memory::default();
            let (constant, ty) = if width == 1 {
                (ConstValue::I8(-2), MockType::I8)
            } else {
                (ConstValue::I16(-2), MockType::I16)
            };
            access.fill_storage(
                &mut memory,
                &stack(-32),
                &Value::Const(constant),
                TypeLayout::of::<MockTarget>(&ty, PointerSize::Bit64)
                    .storage_layout(PointerSize::Bit64),
                &Value::Const(ConstValue::U64(5)),
                false,
            );
            let address = Address::new(Root::Stack, -32, PointerSize::Bit64);
            if width == 1 {
                assert_eq!(memory.known_run(address, 5, None), [0xfe; 5]);
            } else {
                assert_eq!(memory.load(address, 1, None), Loaded::Unknown);
            }
        }
    }

    /// Writable private spans belong to live frames or live allocations, including shared stacks.
    #[test]
    fn writable_backing_tracks_frame_and_allocation_lifetimes() {
        let body = crate::ir::function::SsaFunction::<MockTarget>::new(0, 0);
        let width = PointerSize::Bit64;
        let address = |root, offset| Address::new(root, offset, width);
        let frame = |root, activation, entry, extent| {
            Frame::enter_at(&body, 0, activation, address(root, entry), vec![], width)
                .with_stack_extent(extent)
        };
        let context = AnalysisContext::new(MACHINE);
        let access = Access::new(&context);
        let mut state = State::new(frame(Root::Stack, 0, 0, Some(-16..0)));
        assert!(access.writable_span(&state, address(Root::Stack, -16), 16));
        assert!(!access.writable_span(&state, address(Root::Stack, -16), 17));
        state.call(frame(Root::StackFrame(1), 1, 0, Some(-8..0)));
        assert!(access.writable_span(&state, address(Root::StackFrame(1), -8), 8));
        assert!(access.writable_span(&state, address(Root::Stack, -16), 16));
        assert!(state.return_from().is_some());
        assert!(!access.writable_span(&state, address(Root::StackFrame(1), -8), 1));
        state.call(frame(Root::Stack, 2, -32, Some(-8..0)));
        assert!(access.writable_span(&state, address(Root::Stack, -40), 8));
        assert!(state.return_from().is_some());
        assert!(!access.writable_span(&state, address(Root::Stack, -40), 1));
        assert!(state.release_stack().is_some());
        assert!(
            state
                .tail_call(frame(Root::StackFrame(3), 3, 0, Some(-8..0)))
                .is_some()
        );
        assert!(!access.writable_span(&state, address(Root::Stack, -8), 1));
        assert!(access.writable_span(&state, address(Root::StackFrame(3), -8), 8));
        state
            .memory_mut()
            .allocate(Root::Heap(0), Some(4), None, None);
        assert!(access.writable_span(&state, address(Root::Heap(0), 0), 4));
        assert!(!access.writable_span(&state, address(Root::Heap(0), 0), 5));
        state.memory_mut().invalidate_lifetimes();
        assert!(!access.writable_span(&state, address(Root::Heap(0), 0), 1));
    }
}
