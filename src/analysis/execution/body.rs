//! Owned and borrowed semantic bodies and current-state code byte access.

use std::{cell::Cell, fmt, ops::Deref, sync::Arc};

use crate::{
    PointerSize,
    analysis::{
        budget::Exhaustion,
        context::AnalysisContext,
        execution::{
            Address, InstructionPermissions, MemoryRefusal, Root,
            access::Access,
            memory::{Loaded, Memory, Transfer},
            state::State,
            uses::UseTrace,
            value::Value,
        },
        image::ImageData,
    },
    ir::function::SsaFunction,
    target::{Endianness, Target},
};

/// Storage retaining one immutable semantic body throughout its execution.
#[derive(Debug, Clone)]
pub enum SourceBody<'a, T: Target> {
    /// A body retained by the invocation's immutable inputs.
    Borrowed(&'a SsaFunction<T>),
    /// A recovered body whose allocation is independently owned.
    Owned(Arc<SsaFunction<T>>),
}

impl<T: Target> Deref for SourceBody<'_, T> {
    type Target = SsaFunction<T>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(body) => body,
            Self::Owned(body) => body,
        }
    }
}

/// Identifies live stack backing without exposing executor frames to the host.
trait StackOwners {
    /// Returns the single activation whose backing covers the span.
    fn owner(&self, address: Address, len: usize) -> Option<u32>;
}

impl<T: Target> StackOwners for State<T> {
    fn owner(&self, address: Address, len: usize) -> Option<u32> {
        self.stack_owner(address, len)
    }
}

/// What a code-resolution boundary refused
/// ([`CodeMemory::with_code_refusals`]): the first report stops the
/// enclosing execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeRefusal {
    /// A code-recovery request exceeded its allowance.
    Limit(Exhaustion),
    /// Code a transfer reaches cannot be fetched: the bytes an instruction
    /// there needs are not known.
    Fetch(MemoryRefusal),
}

/// Reads the reaching memory state while a host resolves a code target or body.
///
/// The view borrows the active execution, including object identities and live
/// extents. It does not provide a second execution or modify state. Explicit
/// initial instruction bytes require a host contract and survive only while
/// their memory origins remain unchanged; unknown writes cannot resurrect them.
/// The host still establishes source representation, mode and version identity.
#[derive(Clone, Copy)]
pub struct CodeMemory<'a> {
    /// Current path memory, before call entry or tail retirement.
    memory: &'a Memory,
    /// The reaching activations that can prove stack backing.
    stack_owners: Option<&'a dyn StackOwners>,
    /// Existing checked access and payload bounds of this execution.
    access: Access<'a>,
    /// Active enclosing call-buffer obligations receiving these reads.
    uses: Option<&'a UseTrace>,
    /// Host-attested initial instruction contents; never a data mutability promise.
    initial_instructions: Option<&'a dyn ImageData>,
    /// Synchronous resolution boundary receiving the first reported code
    /// refusal.
    code_refusals: Option<&'a Cell<Option<CodeRefusal>>>,
}

impl fmt::Debug for CodeMemory<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodeMemory")
            .field("access", &self.access)
            .field("initial_instructions", &self.initial_instructions.is_some())
            .finish_non_exhaustive()
    }
}

/// How the executor lends a [`CodeMemory`] view of its own path state.
///
/// The view's host-facing methods are inherent; building it from executor
/// memory, activations and use obligations stays with the executor.
pub trait ExecutorCodeMemory<'a> {
    /// Borrows the current path under its existing machine and resource policy.
    fn new(memory: &'a Memory, context: &'a AnalysisContext<'a>) -> Self;

    /// Borrows the reaching activations for live instruction-source checks.
    #[must_use]
    fn with_stack_owners<T: Target>(self, state: &'a State<T>) -> Self;

    /// Shares enclosing byte-use obligations without cloning execution state.
    #[must_use]
    fn with_uses(self, uses: Option<&'a UseTrace>) -> Self;
}

impl<'a> ExecutorCodeMemory<'a> for CodeMemory<'a> {
    fn new(memory: &'a Memory, context: &'a AnalysisContext<'a>) -> Self {
        Self {
            memory,
            stack_owners: None,
            access: Access::new(context),
            uses: None,
            initial_instructions: None,
            code_refusals: None,
        }
    }

    fn with_stack_owners<T: Target>(mut self, state: &'a State<T>) -> Self {
        self.stack_owners = Some(state);
        self
    }

    fn with_uses(mut self, uses: Option<&'a UseTrace>) -> Self {
        self.uses = uses;
        self
    }
}

impl<'a> CodeMemory<'a> {
    /// Returns the unique live activation backing this nonempty stack span.
    #[must_use]
    pub fn stack_owner(self, address: Address, len: usize) -> Option<u32> {
        if len == 0 || address.pointer_size() != self.pointer_size() {
            return None;
        }
        self.stack_owners?.owner(address, len)
    }

    /// Borrows one synchronous boundary's code-refusal signal.
    pub fn with_code_refusals<'b>(self, refusals: &'b Cell<Option<CodeRefusal>>) -> CodeMemory<'b>
    where
        'a: 'b,
    {
        CodeMemory {
            memory: self.memory,
            stack_owners: self.stack_owners,
            access: self.access,
            uses: self.uses,
            initial_instructions: self.initial_instructions,
            code_refusals: Some(refusals),
        }
    }

    /// Reports a refused code-recovery request to this resolution boundary.
    ///
    /// The first valid report stops the enclosing execution before fallback,
    /// argument projection, source entry or tail retirement, even if the host
    /// also returns a target. No guest state is changed and no report survives
    /// into another callback or execution. Nested instruction-image views keep
    /// the same reporting boundary.
    ///
    /// Returns false for a demand within its allowance or a view outside an
    /// observed code-resolution boundary.
    #[must_use]
    pub fn refuse_code_limit(self, exhaustion: Exhaustion) -> bool {
        if !exhaustion.is_exhausted() {
            return false;
        }
        self.refuse(CodeRefusal::Limit(exhaustion))
    }

    /// Reports to this resolution boundary that code a transfer reaches
    /// cannot be fetched: the bytes an instruction there needs, or its delay
    /// slot, are not known, written with contents this execution does not
    /// know or never there at all.
    ///
    /// The first report stops the enclosing execution with `reason` before
    /// it enters the code, as [`Self::refuse_code_limit`] does. Returns false
    /// for a view outside an observed code-resolution boundary.
    #[must_use]
    pub fn refuse_fetch(self, reason: MemoryRefusal) -> bool {
        self.refuse(CodeRefusal::Fetch(reason))
    }

    /// Records `refusal` at the boundary unless one came first.
    fn refuse(self, refusal: CodeRefusal) -> bool {
        let Some(refusals) = self.code_refusals else {
            return false;
        };
        if refusals.get().is_none() {
            refusals.set(Some(refusal));
        }
        true
    }

    /// Borrows host-attested initial instruction bytes for unchanged image cells.
    ///
    /// The host declares these bytes to describe instruction storage at this
    /// trial's entry. This grants neither execute/write permission nor whole-run
    /// immutability. Reaching stores take precedence byte by byte, and unknown
    /// writes, aliases or clobbers prevent fallback. Ordinary data loads and
    /// pointer-cell lookup retain their separate initial-value rules.
    #[must_use]
    pub fn with_initial_instructions<'b>(self, image: &'b dyn ImageData) -> CodeMemory<'b>
    where
        'a: 'b,
    {
        CodeMemory {
            memory: self.memory,
            stack_owners: self.stack_owners,
            access: self.access,
            uses: self.uses,
            initial_instructions: Some(image),
            code_refusals: self.code_refusals,
        }
    }

    /// Reads reaching instruction bytes under the declared initial-source contract.
    fn instruction_run(self, address: Address, limit: usize) -> Vec<u8> {
        let mut bytes = self.access.known_run(self.memory, address, limit);
        let Some(image) = self.initial_instructions else {
            return bytes;
        };
        for index in bytes.len()..limit {
            let Some(at) = i64::try_from(index)
                .ok()
                .and_then(|delta| address.moved(delta, self.pointer_size()))
            else {
                break;
            };
            let known = self.access.known_run(self.memory, at, 1).first().copied();
            let initial = || {
                if !self.memory.initial_image_binding(at, 1) {
                    return None;
                }
                image.bytes(at.image_address()?, 1)?.first().copied()
            };
            let Some(byte) = known.or_else(initial) else {
                break;
            };
            bytes.push(byte);
        }
        bytes
    }

    /// Returns the pointer width of the current execution's memory domain.
    #[must_use]
    pub fn pointer_size(self) -> PointerSize {
        self.access.ptr_size()
    }

    /// Resolves an established pointer under the current execution's numeric mapping.
    ///
    /// Numeric constants use only the loader relation declared by the image.
    /// Symbolic stack and allocation identities retain their actual coordinates;
    /// unknown and partial values supply no target.
    #[must_use]
    pub fn address_of<T: Target>(self, value: &Value<T>) -> Option<Address> {
        self.access.address_of(value, None)
    }

    /// Returns the current execution's established memory byte order.
    #[must_use]
    pub fn byte_order(self) -> Option<Endianness> {
        self.access.byte_order()
    }

    /// Returns whether a proven local fence covers the reaching instruction writes.
    #[must_use]
    pub const fn instruction_fetch_synchronized(self) -> bool {
        self.memory.instruction_sources().fetch_synchronized()
    }

    /// Returns the permissions the path's protection changes installed in
    /// force at `base`, if any.
    #[must_use]
    pub fn instruction_permissions(self, base: Address) -> Option<InstructionPermissions> {
        self.memory.instruction_sources().permissions(base)
    }

    /// The `len` bytes from `address`, cut where the permissions the path's
    /// protection changes installed change: each piece's start, length and
    /// what was installed over it, `None` where the storage's own hold.
    #[must_use]
    pub fn instruction_permissions_over(
        self,
        address: Address,
        len: u64,
    ) -> Vec<(Address, u64, Option<InstructionPermissions>)> {
        self.memory
            .instruction_sources()
            .permissions_over(address, len)
    }

    /// Whether a protection change on this path covered any of the `len`
    /// bytes from `address`: code there may have been written.
    #[must_use]
    pub fn protection_changed(self, address: Address, len: u64) -> bool {
        self.memory
            .instruction_sources()
            .protection_changed(address, len)
    }

    /// Returns whether the pointer cell retains its host-declared initial image binding.
    ///
    /// This checks writes and aliases in the current execution, including unknown
    /// stores and writes preceding discovery of the binding. It supplies no target
    /// or payload bytes; the host remains responsible for the initial binding.
    ///
    /// # Errors
    /// Returns a memory refusal for invalid lifetime, extent or pointer coordinates.
    pub fn unchanged_image_binding(self, address: Address) -> Result<bool, MemoryRefusal> {
        let len = self.pointer_span(address)?;
        Ok(len <= self.access.buffer_limit() && self.memory.initial_image_binding(address, len))
    }

    /// Whether the `len` image bytes from `address` still hold what the image
    /// binds there: no write of this execution reached them, and no write it
    /// could not place may have.
    #[must_use]
    pub fn image_unwritten(self, address: Address, len: usize) -> bool {
        self.memory.initial_image_binding(address, len)
    }

    /// Reads the current complete pointer while preserving symbolic storage identity.
    ///
    /// Unknown bytes and partial pointers yield no target. Numeric payloads retain
    /// [`Root::Number`] coordinates; converting them into a module address requires
    /// an actual machine mapping supplied by the host.
    ///
    /// # Errors
    /// Returns a memory refusal for invalid lifetime, extent or pointer coordinates.
    pub fn read_pointer(self, address: Address) -> Result<Option<Address>, MemoryRefusal> {
        let len = self.pointer_span(address)?;
        if len > self.access.buffer_limit() {
            if let Some(uses) = self.uses {
                uses.unresolved();
            }
            return Ok(None);
        }
        let pointer = match self.memory.load(address, len, None) {
            Loaded::Address(pointer) if pointer.pointer_size() == self.pointer_size() => {
                Some(pointer)
            }
            Loaded::Bytes(bytes) if len <= 8 => self.byte_order().map(|order| {
                let bits = match order {
                    Endianness::Little => bytes
                        .iter()
                        .rev()
                        .fold(0u64, |bits, &byte| bits.wrapping_shl(8) | u64::from(byte)),
                    Endianness::Big => bytes
                        .iter()
                        .fold(0u64, |bits, &byte| bits.wrapping_shl(8) | u64::from(byte)),
                };
                Address::new(Root::Number, bits.cast_signed(), self.pointer_size())
            }),
            _ => None,
        };
        if pointer.is_none()
            && let Some(uses) = self.uses
        {
            uses.unresolved();
        }
        Ok(pointer)
    }

    /// Checks and records the exact current-machine pointer-cell access.
    fn pointer_span(self, address: Address) -> Result<usize, MemoryRefusal> {
        let len = self.pointer_size().bytes();
        if let Some(uses) = self.uses {
            uses.access(address, len);
        }
        if address.pointer_size() != self.pointer_size() || address.linear_prefix_len(len) != len {
            return Err(MemoryRefusal::AddressRange);
        }
        if let Some(reason) = self.memory.admit(address, len, Transfer::Read) {
            return Err(reason);
        }
        Ok(len)
    }

    /// Inspects a bounded known prefix while planning instruction decoding.
    ///
    /// Speculative decoder probes do not establish executed byte uses. A host
    /// must fetch the exact decoded instruction spans with [`Self::read`] before
    /// entering the body. Unknown suffixes remain absent; this method never
    /// crosses target-coordinate wrap or the execution's buffer allowance.
    ///
    /// # Errors
    /// Returns a memory refusal when the first required byte has an invalid
    /// object lifetime or extent. A zero limit performs no memory access.
    pub fn inspect_prefix(
        self,
        address: Address,
        maximum: usize,
    ) -> Result<Vec<u8>, MemoryRefusal> {
        let maximum = address.linear_prefix_len(maximum.min(self.access.buffer_limit()));
        if maximum == 0 {
            return Ok(Vec::new());
        }
        if let Some(reason) = self.memory.admit(address, 1, Transfer::Read) {
            return Err(reason);
        }
        Ok(self.instruction_run(address, maximum))
    }

    /// Returns an exact known span within the execution's byte-buffer limit.
    ///
    /// Unknown contents and size refusal return `Ok(None)`. No partial span is
    /// exposed as complete code. Ordinary data-image permissions are preserved;
    /// absent executable image bytes are not invented by this view.
    ///
    /// # Errors
    ///
    /// Returns a memory refusal for expired objects, invalid extents, or wrapping
    /// coordinates. A host must propagate a refusal from a required code read.
    pub fn read(self, address: Address, len: usize) -> Result<Option<Vec<u8>>, MemoryRefusal> {
        if address.linear_prefix_len(len) != len {
            return Err(MemoryRefusal::AddressRange);
        }
        if len > self.access.buffer_limit() {
            if let Some(uses) = self.uses {
                uses.unresolved();
            }
            return Ok(None);
        }
        if let Some(uses) = self.uses {
            uses.access(address, len);
        }
        if let Some(reason) = self.memory.admit(address, len, Transfer::Read) {
            return Err(reason);
        }
        let bytes = self.instruction_run(address, len);
        if bytes.len() != len {
            if let Some(uses) = self.uses {
                uses.unresolved();
            }
            return Ok(None);
        }
        Ok(Some(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MACHINE;

    #[test]
    fn pointer_cells_keep_unknown_write_and_join_origins() {
        let context = AnalysisContext::new(MACHINE);
        let slot = Address::new(Root::Symbol, 0x3000, MACHINE.ptr_size);
        let target = Address::new(Root::Symbol, 0x4000, MACHINE.ptr_size);
        let mut memory = Memory::default();
        assert_eq!(
            CodeMemory::new(&memory, &context).unchanged_image_binding(slot),
            Ok(true)
        );
        memory.store_bytes(Address::new(Root::Symbol, 0x2000, MACHINE.ptr_size), &[1]);
        assert_eq!(
            CodeMemory::new(&memory, &context).unchanged_image_binding(slot),
            Ok(true)
        );
        let untouched = memory.clone();
        memory.store_address(slot, target, 8);
        assert_eq!(
            CodeMemory::new(&memory, &context).read_pointer(slot),
            Ok(Some(target))
        );
        assert_eq!(
            CodeMemory::new(&memory, &context).unchanged_image_binding(slot),
            Ok(false)
        );
        memory.store_unknown(slot, 8);
        assert_eq!(
            CodeMemory::new(&memory, &context).read_pointer(slot),
            Ok(None)
        );
        assert_eq!(
            CodeMemory::new(&memory, &context).unchanged_image_binding(slot),
            Ok(false)
        );
        let joined = untouched.join(&memory);
        assert_eq!(
            CodeMemory::new(&joined, &context).unchanged_image_binding(slot),
            Ok(false)
        );
        assert_eq!(
            CodeMemory::new(&joined, &context).read_pointer(slot),
            Ok(None)
        );
        let mut unknown_alias = untouched.clone();
        unknown_alias.store_bytes(Address::of_root(Root::Number), &[1]);
        assert_eq!(
            CodeMemory::new(&unknown_alias, &context).unchanged_image_binding(slot),
            Ok(false)
        );
        let mut clobbered = untouched;
        clobbered.clobber();
        assert_eq!(
            CodeMemory::new(&clobbered, &context).unchanged_image_binding(slot),
            Ok(false)
        );
    }

    #[test]
    fn numeric_pointer_bytes_keep_their_actual_coordinate_domain() {
        let slot = Address::new(Root::Symbol, 0x3000, MACHINE.ptr_size);
        for order in [Endianness::Little, Endianness::Big] {
            let mut machine = MACHINE;
            machine.byte_order = Some(order);
            let context = AnalysisContext::new(machine);
            let mut memory = Memory::default();
            let number = 0x1234_5678_9abc_def0u64;
            let bytes = match order {
                Endianness::Little => number.to_le_bytes(),
                Endianness::Big => number.to_be_bytes(),
            };
            memory.store_bytes(slot, &bytes);
            assert_eq!(
                CodeMemory::new(&memory, &context).read_pointer(slot),
                Ok(Some(Address::new(
                    Root::Number,
                    number.cast_signed(),
                    MACHINE.ptr_size
                )))
            );
            memory.store_unknown(slot, 1);
            assert_eq!(
                CodeMemory::new(&memory, &context).read_pointer(slot),
                Ok(None)
            );
        }
        let context = AnalysisContext::new(MACHINE);
        let mut memory = Memory::default();
        memory.store_bytes(slot, &[0; 8]);
        assert_eq!(
            CodeMemory::new(&memory, &context).read_pointer(slot),
            Ok(None)
        );
    }

    /// A pointer cell in no block the path made, or running past its block,
    /// holds no pointer the run knows; only a span that wraps is refused.
    #[test]
    fn pointer_cells_require_live_complete_nonwrapping_storage() {
        let context = AnalysisContext::new(MACHINE);
        let mut memory = Memory::default();
        let heap = Address::of_root(Root::Heap(1));
        assert_eq!(
            CodeMemory::new(&memory, &context).read_pointer(heap),
            Ok(None)
        );
        memory.allocate(Root::Heap(1), Some(4), None, None);
        assert_eq!(
            CodeMemory::new(&memory, &context).read_pointer(heap),
            Ok(None)
        );
        let wrapping = Address::new(Root::Symbol, -4, MACHINE.ptr_size);
        assert_eq!(
            CodeMemory::new(&memory, &context).read_pointer(wrapping),
            Err(MemoryRefusal::AddressRange)
        );
    }
    struct MutableInstructions;

    impl ImageData for MutableInstructions {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }
        fn region(&self, at: u64) -> Option<crate::analysis::image::ImageRegion> {
            (0x1000..0x1008)
                .contains(&at)
                .then_some(crate::analysis::image::ImageRegion {
                    start: 0x1000,
                    end: 0x1008,
                    mutability: crate::analysis::image::Mutability::Writable,
                })
        }
        fn bytes(&self, at: u64, len: usize) -> Option<&[u8]> {
            let offset = usize::try_from(at.checked_sub(0x1000)?).ok()?;
            [0xb8, 42, 0, 0, 0, 0xc3, 0x90, 0x90].get(offset..offset.checked_add(len)?)
        }
    }

    #[test]
    fn initial_instruction_bytes_follow_origins_and_never_supply_ordinary_data() {
        let image = MutableInstructions;
        let context = AnalysisContext::new(MACHINE).with_image(&image);
        let at = Address::new(Root::Symbol, 0x1000, MACHINE.ptr_size);
        let mut memory = Memory::default();
        assert_eq!(CodeMemory::new(&memory, &context).read(at, 8), Ok(None));
        let code = CodeMemory::new(&memory, &context).with_initial_instructions(&image);
        assert_eq!(
            code.read(at, 8),
            Ok(Some(vec![0xb8, 42, 0, 0, 0, 0xc3, 0x90, 0x90]))
        );
        assert_eq!(code.read_pointer(at), Ok(None));
        assert!(Access::new(&context).known_run(&memory, at, 8).is_empty());
        let changed = at
            .moved(1, MACHINE.ptr_size)
            .expect("instruction immediate");
        memory.store_bytes(changed, &[43]);
        assert_eq!(
            CodeMemory::new(&memory, &context)
                .with_initial_instructions(&image)
                .read(at, 8),
            Ok(Some(vec![0xb8, 43, 0, 0, 0, 0xc3, 0x90, 0x90]))
        );
        memory.store_unknown(changed, 1);
        let code = CodeMemory::new(&memory, &context).with_initial_instructions(&image);
        assert_eq!(code.read(at, 8), Ok(None));
        assert_eq!(code.inspect_prefix(at, 8), Ok(vec![0xb8]));
        let mut memory = Memory::default();
        memory.clobber();
        assert_eq!(
            CodeMemory::new(&memory, &context)
                .with_initial_instructions(&image)
                .read(at, 8),
            Ok(None)
        );
        let mut memory = Memory::default();
        memory.store_bytes(Address::of_root(Root::Number), &[1]);
        assert_eq!(
            CodeMemory::new(&memory, &context)
                .with_initial_instructions(&image)
                .read(at, 8),
            Ok(None)
        );
        assert_eq!(
            CodeMemory::new(&Memory::default(), &context)
                .with_initial_instructions(&image)
                .read(Address::new(Root::Symbol, -1, MACHINE.ptr_size), 2),
            Err(MemoryRefusal::AddressRange)
        );
    }
}
