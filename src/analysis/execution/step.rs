//! Running the instructions of one block.
//!
//! The stepper runs the running activation's block from its next instruction
//! up to the block's terminator, or up to a call, which it leaves to the
//! explorer: what a call does is the host's to say. Everything else it runs in
//! place:
//!
//! - pure operations share the constant evaluator's value semantics; proven
//!   checked-integer overflow ends the path before subsequent observations;
//! - a load, a store, a `memset` and a `memcpy` go through memory, and every
//!   byte a store writes is recorded in the loops the path is in;
//! - any other operation defines what it defines as unknown, and one that may
//!   write memory forgets all of it the execution cannot keep private.
//!
//! An observable effect or an unproven potentially faulting operation spoils
//! loop replacement while its values may still be analyzed. A write the
//! execution cannot place - through an unknown pointer, or of an
//! unknown size - spoils the loops the path is in: their effect is no longer
//! the bytes they are known to have written.

use std::cell::RefCell;

use crate::{
    analysis::{
        budget::{Budget, Exhaustion},
        consts::{Destination, FoldedResults, OperandLayout},
        context::AnalysisContext,
        execution::{
            MemoryWriteGuard,
            access::{Access, AccessError, WritePreflight},
            atomic::Atomics,
            facts::{
                ArgumentBytes, ExtentSource, ObjectKind, Observations, ReplacementProof,
                StorageObject,
            },
            findings::FindingKind,
            frame::{Frame, Resumption},
            host::ExecutionHost,
            marks::Marker,
            memory::{Memory, MemoryRefusal, Through, WriteSite},
            provenance::{Provenance, Served},
            scalar::Scalar,
            state::State,
            trace::Trace,
            types::{OperationLayout, TypeLayouts},
            value::{Address, Root, Value},
            vectorreads::VectorReads,
            vectorwrites::VectorWrites,
        },
        inactive,
    },
    ir::{
        evaluation::Evaluation,
        function::SsaFunction,
        integer::{BitWidth, IntegerType},
        lanes::VectorConst,
        ops::{
            AccessPermission, AccessQualifiers, AtomicOrdering, BarrierOp, CompareBlkData,
            MemoryAccessSemantics, RuntimeValueKind, SsaEffectKind, SsaEffects, SsaOp,
            SystemOpKind, TrapClass, TrapOp,
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    target::{Target, TypeLayout},
};

/// Why the stepper stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// A value or activation identity cannot be represented uniquely.
    IdentityExhausted,
    /// A required type has no supported host interpretation.
    UnsupportedType,
    /// A configured operation bound was exceeded.
    Exhausted(Exhaustion),
    /// An arithmetic check is proven to trap.
    Trap,
    /// A store reached storage that may not be written
    /// ([`Memory::fault`](crate::analysis::execution::memory::Memory::fault)),
    /// and the machine faulted it: found at the store
    /// ([`FindingKind::ReadOnlyWrite`]). Where a handler may receive the
    /// raise, the path before the store is among the raises.
    Faulted,
    /// A memory contract was unavailable.
    Refused(MemoryRefusal),
    /// At the block's terminator, which is not run.
    Terminator,
    /// At the call at this instruction index, which is not run.
    Call(usize),
    /// Before the instruction at this index, which belongs to no source
    /// instruction of the one whose write changed the activation's code, or
    /// which is no pure instruction of a block that entered changed code
    /// ([`Frame::resuming`](crate::analysis::execution::frame::Frame::resuming)):
    /// the activation goes on in a lift of the code as it is now.
    Resume(usize),
    /// After the instruction at this index, which ran and wrote code an
    /// activation had entered
    /// ([`InstructionSources::record_change`](crate::analysis::execution::InstructionSources::record_change)).
    CodeChanged(usize),
    /// The allowance for non-control instructions was consumed.
    Limited,
}

/// What one run of a block executed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ran {
    /// The instructions executed, a trapping one included.
    pub instructions: usize,
    /// The steps they cost: one each, and one more per byte each block
    /// instruction spanned.
    pub steps: usize,
}

/// What a repeated compare leaves ([`SsaOp::CompareBlk`],
/// [`SsaOp::ScanBlk`]).
struct Compared<T: Target> {
    /// The count left.
    remaining: Value<T>,
    /// The pointer past the last element compared in the first block.
    left_next: Value<T>,
    /// The pointer past the last element compared in the second block;
    /// unknown for a scan, which has none.
    right_next: Value<T>,
    /// The flags of the last pair compared, or those given where none was.
    flags: Value<T>,
    /// Elements actually read before the compare stopped.
    elements: u64,
}

/// What a repeated compare compares each element of its first block with:
/// the element of a second block ([`SsaOp::CompareBlk`]) or one value
/// ([`SsaOp::ScanBlk`]).
#[derive(Debug, Clone, Copy)]
enum Against {
    /// The second block's first element.
    Block(SsaVarId),
    /// The value.
    Value(SsaVarId),
}

/// Exact read effects of one successfully executed instruction.
#[derive(Default)]
struct ReadEffect {
    /// Whether a block proved that it performs no memory access.
    empty: bool,
    /// At most two source spans: a block comparison has two inputs.
    footprints: [Option<(Address, usize)>; 2],
    /// Span index and SSA result supplied by its final scalar read.
    value: Option<(usize, SsaVarId)>,
    /// The element reads of a lane-wise vector load.
    lanes: Option<LaneSpans>,
}

/// The elements a lane-wise vector load read, one span each.
struct LaneSpans {
    /// Each element's span.
    spans: Vec<(Address, usize)>,
    /// The bytes of one element, the unit each span's alignment is proven in.
    unit: usize,
    /// The access qualifiers every element was read with.
    access: AccessQualifiers,
    /// Whether the spans are every element the load names.
    complete: bool,
}

impl ReadEffect {
    /// Separates the final result-producing element from the preceding repeated span.
    fn repeated(
        address: Address,
        len: usize,
        element_bytes: u32,
        down: bool,
        value: SsaVarId,
    ) -> Option<Self> {
        let width = element_bytes as usize;
        if width == 0 {
            return None;
        }
        let preceding = len.checked_sub(width)?;
        if preceding == 0 {
            return Some(Self {
                empty: false,
                footprints: [Some((address, width)), None],
                value: Some((0, value)),
                lanes: None,
            });
        }
        let last_offset = if down { 0 } else { preceding };
        let prefix_offset = if down { width } else { 0 };
        let last = Address::new(
            address.root(),
            address.offset_at(last_offset)?,
            address.pointer_size(),
        );
        let prefix = Address::new(
            address.root(),
            address.offset_at(prefix_offset)?,
            address.pointer_size(),
        );
        Some(Self {
            empty: false,
            footprints: [Some((prefix, preceding)), Some((last, width))],
            value: Some((1, value)),
            lanes: None,
        })
    }
}

/// A point where an instruction may raise: the path as it stands before the
/// instruction, which raises there on the paths where it does.
pub struct Raised<T: Target> {
    /// What the path knows before the instruction.
    pub state: State<T>,
    /// How the path got there.
    pub trace: Trace,
    /// The instruction, in the running activation's block.
    pub at: usize,
}

/// Runs instructions of one function.
pub struct Stepper<'a, T: Target> {
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// Loads and stores.
    access: Access<'a>,
    /// Pure operations.
    scalar: Scalar<'a>,
    /// Root-scoped physical mappings for evaluated operands.
    host: Option<&'a dyn ExecutionHost<T>>,
    /// Layouts shared by every path of this body.
    layouts: Option<&'a TypeLayouts<'a, T>>,
    /// Carries never-written bits through the instructions and records
    /// their uses, where the run tracks them.
    marker: Option<Marker<'a, T>>,
    /// Where an instruction that may raise records the path before it,
    /// where a handler may receive the raise.
    raises: Option<&'a RefCell<Vec<Raised<T>>>>,
}

impl<'a, T: Target> Stepper<'a, T> {
    /// Runs `ssa` against `context`'s image and target.
    pub fn new(ssa: &'a SsaFunction<T>, context: &'a AnalysisContext<'a>) -> Self {
        Self {
            ssa,
            access: Access::new(context),
            scalar: Scalar::new(context.machine()).with_image(context.image()),
            host: None,
            layouts: None,
            marker: None,
            raises: None,
        }
    }

    /// Records in `raises` the path before every instruction that may raise
    /// and is not proven to: an access through an address the run does not
    /// know, an operation that traps on operands it does not know. One that
    /// is proven to raise stops the run ([`Stop::Trap`]). A store into
    /// storage the image states is never written stops it too
    /// ([`Stop::Faulted`]), and raises from the path before the store.
    pub const fn with_raises(mut self, raises: &'a RefCell<Vec<Raised<T>>>) -> Self {
        self.raises = Some(raises);
        self
    }

    /// Supplies path-local interpretation of mapped allocation addresses.
    pub const fn with_host(mut self, host: &'a dyn ExecutionHost<T>) -> Self {
        self.host = Some(host);
        self
    }

    /// Shares the root trial's known-write admission constraint.
    pub fn with_write_guard(mut self, guard: Option<&'a dyn MemoryWriteGuard>) -> Self {
        self.access = self.access.with_write_guard(guard);
        self
    }

    /// Shares admitted variable layouts across every activation of this body.
    pub const fn with_layouts(mut self, layouts: &'a TypeLayouts<'a, T>) -> Self {
        self.layouts = Some(layouts);
        self
    }

    /// Carries never-written bits through the instructions run, and records
    /// their uses, with `marker`.
    pub const fn with_marker(mut self, marker: Marker<'a, T>) -> Self {
        self.marker = Some(marker);
        self
    }

    /// Reads a declared variable type, through the shared layouts where
    /// they are installed.
    fn variable_layout(&self, value: SsaVarId) -> Option<TypeLayout> {
        if let Some(layouts) = self.layouts {
            return layouts.get(value);
        }
        self.access
            .type_layout::<T>(self.ssa.variable(value)?.var_type())
    }

    /// Reads a result's declared numeric facts.
    fn destination(&self, op: &SsaOp<T>) -> Destination {
        if let Some(layouts) = self.layouts {
            return layouts.destination(op);
        }
        op.dest()
            .and_then(|var| self.variable_layout(var))
            .map_or_else(Destination::default, Destination::from_type_layout)
    }

    /// Whether a block operation walks downward, as the direction flag it
    /// reads holds here: upward where it reads none, and `None` where the
    /// flag is not known, so the operation clobbers as any unmodelled write.
    fn downward(&self, state: &State<T>, backward: Option<SsaVarId>) -> Option<bool> {
        match backward {
            None => Some(false),
            Some(flag) => state.value(flag).constant().and_then(ConstValue::as_bool),
        }
    }

    /// Element `index` of the elements of `element_bytes` bytes from
    /// `pointer`, walking down where `down`, as an unsigned integer: `None`
    /// where the address or a byte of the element is not known.
    fn element(
        &self,
        memory: &Memory,
        pointer: &Value<T>,
        index: u64,
        element_bytes: u32,
        down: bool,
    ) -> Result<Option<Value<T>>, MemoryRefusal> {
        let Some(address) = self.walked(index, element_bytes, down).and_then(|delta| {
            self.access
                .address_of(pointer, None)?
                .moved(delta, self.access.ptr_size())
        }) else {
            return Ok(None);
        };
        let Some(integer) = IntegerType::new(element_bytes.saturating_mul(8), false) else {
            return Ok(None);
        };
        let loaded = self
            .access
            .load_layout::<T>(memory, address, TypeLayout::integer(integer))?;
        Ok(loaded.constant().is_some().then_some(loaded))
    }

    /// The bytes `elements` elements of `element_bytes` bytes span, negated
    /// where the walk goes `down`; `None` where they overflow.
    fn walked(&self, elements: u64, element_bytes: u32, down: bool) -> Option<i64> {
        let bytes = i64::try_from(elements.checked_mul(u64::from(element_bytes))?).ok()?;
        Some(if down { bytes.wrapping_neg() } else { bytes })
    }

    /// `pointer` past `elements` elements of `element_bytes` bytes, walking
    /// down where `down`.
    fn advanced(
        &self,
        pointer: Value<T>,
        elements: u64,
        element_bytes: u32,
        down: bool,
    ) -> Value<T> {
        match self.walked(elements, element_bytes, down) {
            Some(delta) => self.scalar.pointer_add(pointer, None, 1, delta),
            None => Value::Unknown,
        }
    }

    /// The results of a repeated compare of elements ([`SsaOp::CompareBlk`],
    /// a block of them against a second; [`SsaOp::ScanBlk`], against one
    /// value): how many it compares, stopping after the first pair that is
    /// unequal (`while_equal`) or equal, at most `count`; the count left; the
    /// pointer past the last compared in each block. The flags of the last
    /// pair compared are unknown, as an arithmetic operation's are here, and
    /// those of an empty compare `flags_in` (a refusal,
    /// [`MemoryRefusal::EmptyBlockInput`], where it states none). `None` where the count, a
    /// pointer or a compared element is not known, the element width is zero,
    /// or the walk would visit past the block limit
    /// ([`Budget::ExecutionBlockBytes`]). The limit bounds the elements
    /// visited, not the count: `repne scasb` with a count of all ones, x86's
    /// inline `strlen`, stops at the terminator.
    #[allow(clippy::too_many_arguments)]
    fn compared(
        &self,
        state: &mut State<T>,
        left: SsaVarId,
        right: Against,
        count: SsaVarId,
        flags_in: Option<SsaVarId>,
        element_bytes: u32,
        while_equal: bool,
        down: bool,
    ) -> Result<Option<Compared<T>>, MemoryRefusal> {
        if element_bytes == 0 {
            return Ok(None);
        }
        let mask = BitWidth::new(element_bytes.saturating_mul(8)).mask();

        let left_pointer = state.value(left);
        let Some(count) = state.value(count).into_constant() else {
            return Ok(None);
        };
        let Some(limit) = count.as_u64_bits() else {
            return Ok(None);
        };
        let visitable = u64::try_from(self.access.block_limit())
            .unwrap_or(u64::MAX)
            .checked_div(u64::from(element_bytes))
            .unwrap_or(0);
        // A scan compares against one value, a block compare against a second
        // block's elements. A block compare reads both blocks one way, so it
        // states the object of each block's pointer as it reads that block
        // (`Provenance::before`); a scan's one pointer is stated already.
        let provenance = Provenance::new(self.ssa, self.access);
        let objects = match right {
            Against::Block(block) => Some((
                provenance.pointer_object(state, left),
                provenance.pointer_object(state, block),
            )),
            Against::Value(_) => None,
        };
        let (right_pointer, value) = match right {
            Against::Block(block) => (Some(state.value(block)), None),
            Against::Value(value) => {
                let Some(value) = state.value(value).into_constant() else {
                    return Ok(None);
                };
                (None, Some(value))
            }
        };
        let mut compared = 0u64;
        while compared < limit {
            if compared >= visitable {
                return Ok(None);
            }
            if let Some((reads, _)) = objects {
                state.memory_mut().set_through(Through {
                    reads,
                    writes: None,
                });
            }
            let Some(first) =
                self.element(state.memory(), &left_pointer, compared, element_bytes, down)?
            else {
                return Ok(None);
            };
            if let Some((_, reads)) = objects {
                state.memory_mut().set_through(Through {
                    reads,
                    writes: None,
                });
            }
            let second = match (&right_pointer, &value) {
                (Some(pointer), _) => {
                    self.element(state.memory(), pointer, compared, element_bytes, down)?
                }
                (None, value) => value.clone().map(Value::Const),
            };
            let (Some(first), Some(second)) = (
                first.constant().and_then(ConstValue::as_u64_bits),
                second
                    .as_ref()
                    .and_then(Value::constant)
                    .and_then(ConstValue::as_u64_bits),
            ) else {
                return Ok(None);
            };
            compared = compared.saturating_add(1);
            if (first & mask == second & mask) != while_equal {
                break;
            }
        }
        let Some(remaining) = count.decremented(compared) else {
            return Ok(None);
        };
        let flags = if compared == 0 {
            state.value(flags_in.ok_or(MemoryRefusal::EmptyBlockInput)?)
        } else {
            Value::Unknown
        };
        let left_next = self.advanced(left_pointer, compared, element_bytes, down);
        let right_next = right_pointer.map_or(Value::Unknown, |pointer| {
            self.advanced(pointer, compared, element_bytes, down)
        });
        Ok(Some(Compared {
            remaining: Value::Const(remaining),
            left_next,
            right_next,
            flags,
            elements: compared,
        }))
    }

    /// Resolves the exact linear footprint of completed repeated elements.
    fn repeated_footprint(
        &self,
        pointer: &Value<T>,
        elements: u64,
        width: u32,
        down: bool,
    ) -> Option<(Address, usize)> {
        if elements == 0 || width == 0 {
            return None;
        }
        let bytes = elements.checked_mul(u64::from(width))?;
        self.access.stream_footprint(
            pointer,
            &Value::Const(ConstValue::U64(bytes)),
            width as usize,
            down,
        )
    }

    /// Reads a numeric input's scalar or vector interpretation.
    fn operand_layout(&self, value: SsaVarId) -> OperandLayout {
        if let Some(layouts) = self.layouts {
            return layouts.operand(value);
        }
        self.variable_layout(value)
            .map_or_else(OperandLayout::default, OperandLayout::from_type_layout)
    }

    /// Runs a block without collecting replacement evidence in unit fixtures.
    #[cfg(test)]
    pub fn run(&self, state: &mut State<T>, trace: &mut Trace, allowance: usize) -> (Stop, Ran) {
        self.run_observed(state, trace, allowance, None)
    }

    /// Runs a block, recording replacement evidence for the root activation.
    pub fn run_observed(
        &self,
        state: &mut State<T>,
        trace: &mut Trace,
        allowance: usize,
        mut observed: Option<&mut Observations<T>>,
    ) -> (Stop, Ran) {
        let Some((mut position, activation, body)) = state
            .frame()
            .map(|frame| (frame.position(), frame.activation(), frame.body()))
        else {
            return (Stop::Terminator, Ran::default());
        };
        let instructions = self
            .ssa
            .block(position.block)
            .map_or(&[][..], |block| block.instructions());
        let mut ran = 0usize;
        let mut spanned = 0usize;
        let stop = loop {
            let Some(instruction) = instructions.get(position.next) else {
                break Stop::Terminator;
            };
            // Every store from here to the next instruction is this one's,
            // including a call's model writes after the stop below.
            state.memory_mut().set_writer(
                activation,
                WriteSite {
                    body,
                    block: position.block,
                    instruction: position.next,
                },
            );
            let op = instruction.op();
            // An activation whose code changed ahead of it finishes the source
            // instruction that changed it, and stops before anything else; one
            // that entered changed code runs the pure instructions the block
            // begins with, and stops before anything else.
            let stops = match state.frame().and_then(Frame::resuming) {
                Some(Resumption::After(trigger_block, trigger)) => {
                    op.is_terminator()
                        || trigger_block != position.block
                        || !self.host.is_some_and(|host| {
                            instructions
                                .get(trigger)
                                .is_some_and(|written| host.same_source(written, instruction))
                        })
                }
                Some(Resumption::Entry { .. }) => {
                    op.is_terminator() || matches!(op, SsaOp::Call(_)) || !op.is_pure()
                }
                None => false,
            };
            if stops {
                break Stop::Resume(position.next);
            }
            if op.is_terminator() {
                break Stop::Terminator;
            }
            if matches!(op, SsaOp::Call(_)) {
                break Stop::Call(position.next);
            }
            if ran.saturating_add(spanned) >= allowance {
                break Stop::Limited;
            }

            let buffer = self.access.buffer_limit();
            let payload = match op {
                SsaOp::Const { value, .. } => value.payload_bytes(),
                _ => 0,
            };
            // Memory access checks its own byte extent before materializing a load.
            let dest = if matches!(
                op,
                SsaOp::LoadIndirect { .. }
                    | SsaOp::VectorLoad { .. }
                    | SsaOp::VectorBroadcastLoad { .. }
            ) {
                Destination::default()
            } else {
                self.destination(op)
            };
            let extent = Self::buffer_demand(dest, payload);
            if extent > buffer {
                break Stop::Exhausted(Exhaustion {
                    budget: Budget::ExecutionBufferBytes,
                    limit: buffer,
                    demand: extent,
                });
            }
            if self.layouts.is_some_and(TypeLayouts::is_unsupported) {
                break Stop::UnsupportedType;
            }
            let (block, index) = (position.block, position.next);
            if let Some(observed) = observed.as_deref_mut() {
                self.observe_use(op, state, (block, index), observed);
            }
            let query = || self.layouts?.operation(block, index);
            let operation = (self.layouts.is_some() && op.stated_type().is_some())
                .then(|| OperationLayout::new(&query));
            let operator = Stepper {
                ssa: self.ssa,
                access: match operation {
                    Some(operation) => self.access.with_operation(operation),
                    None => self.access,
                },
                scalar: match operation {
                    Some(operation) => self.scalar.with_operation(operation),
                    None => self.scalar,
                },
                host: self.host,
                layouts: self.layouts,
                marker: self.marker,
                raises: self.raises,
            };
            let control = if matches!(
                op,
                SsaOp::Div { .. }
                    | SsaOp::Rem { .. }
                    | SsaOp::AddOvf { .. }
                    | SsaOp::SubOvf { .. }
                    | SsaOp::MulOvf { .. }
                    | SsaOp::IntConv {
                        overflow_check: true,
                        ..
                    }
                    | SsaOp::FloatToInt {
                        overflow_check: true,
                        ..
                    }
                    | SsaOp::Ckfinite { .. }
            ) || matches!(
                op,
                SsaOp::SystemOp(data)
                    if matches!(data.kind, SystemOpKind::Trap { op, .. } if op != TrapOp::SoftwareInterrupt)
            ) {
                operator.scalar.control(
                    op,
                    operator.destination(op),
                    |var| state.value(var),
                    |var| operator.operand_layout(var),
                )
            } else {
                None
            };
            let control_results = operator.scalar.control_results(
                op,
                |var| state.value(var),
                |var| operator.operand_layout(var),
            );
            if let Some(observed) = observed.as_deref_mut() {
                match op {
                    SsaOp::AddOvf { dest, .. }
                    | SsaOp::SubOvf { dest, .. }
                    | SsaOp::MulOvf { dest, .. }
                    | SsaOp::IntConv {
                        dest,
                        overflow_check: true,
                        ..
                    }
                    | SsaOp::FloatToInt {
                        dest,
                        overflow_check: true,
                        ..
                    }
                    | SsaOp::Ckfinite { dest, .. }
                    | SsaOp::Div { dest, .. }
                    | SsaOp::Rem { dest, .. } => {
                        let proof = matches!(&control, Some((Evaluation::Value(_), _)))
                            .then_some(ReplacementProof::Completed);
                        observed.replacement(*dest, proof);
                    }
                    SsaOp::WideDiv {
                        quotient,
                        remainder,
                        ..
                    } => {
                        let proof = matches!(&control_results, Some((Evaluation::Value(_), _)))
                            .then_some(ReplacementProof::Completed);
                        observed.replacement(*quotient, proof);
                        observed.replacement(*remainder, proof);
                    }
                    // A relaxed atomic load that opens no reservation orders
                    // nothing, so only its read can fault.
                    SsaOp::LoadIndirect {
                        dest,
                        addr,
                        value_type,
                        access,
                    }
                    | SsaOp::VectorLoad {
                        dest,
                        addr,
                        vector_type: value_type,
                        access,
                    }
                    | SsaOp::AtomicLoad {
                        dest,
                        addr,
                        value_type,
                        ordering: AtomicOrdering::Relaxed,
                        reserves: false,
                        access,
                        ..
                    } if observed.needs_replacement_evidence(*dest) => {
                        let proof = operator.access.read_proof(
                            state,
                            &state.value(*addr),
                            value_type,
                            *access,
                        );
                        observed.replacement(*dest, proof);
                    }
                    // A broadcast reads one element, the lane's width.
                    SsaOp::VectorBroadcastLoad {
                        dest,
                        addr,
                        vector_type,
                        access,
                    } if observed.needs_replacement_evidence(*dest) => {
                        let proof = T::vector_shape(vector_type)
                            .and_then(|shape| T::integer_type(shape.lane_bits, false))
                            .and_then(|element| {
                                operator.access.read_proof(
                                    state,
                                    &state.value(*addr),
                                    &element,
                                    *access,
                                )
                            });
                        observed.replacement(*dest, proof);
                    }
                    // A lane-wise load is proven where every element it reads
                    // is: each of its results is its constant only then.
                    _ if op
                        .defs()
                        .any(|dest| observed.needs_replacement_evidence(dest)) =>
                    {
                        let lanes = VectorReads::new(operator.access);
                        if let Ok(Some(plan)) = lanes.plan(
                            op,
                            state,
                            |value| operator.variable_layout(value),
                            |value| operator.operand_layout(value),
                        ) {
                            let proof = plan.and_then(|plan| lanes.proof(&plan, state));
                            for dest in op.defs() {
                                observed.replacement(dest, proof);
                            }
                        }
                    }
                    _ => {}
                }
            }
            if matches!(&control, Some((Evaluation::Trap(_), _)))
                || matches!(&control_results, Some((Evaluation::Trap(_), _)))
            {
                trace.taint();
                ran = ran.saturating_add(1);
                position.next = position.next.saturating_add(1);
                break Stop::Trap;
            }
            // Asked only where a handler may receive a raise: telling whether
            // an access reaches an address the run knows resolves it, which
            // nothing else needs before the access's own limits are checked.
            let may_raise = self.raises.is_some()
                && (matches!(
                    &control,
                    Some((Evaluation::Unknown | Evaluation::Unsupported, _))
                ) || matches!(
                    &control_results,
                    Some((Evaluation::Unknown | Evaluation::Unsupported, _))
                ) || self.reaches_unknown_address(op, state));
            if let Some(raises) = self.raises
                && may_raise
            {
                raises.borrow_mut().push(Raised {
                    state: state.clone(),
                    trace: trace.clone(),
                    at: position.next,
                });
            }
            // A store into the image faults where the image may not be
            // written: where a handler may receive the raise, it raises from
            // the path as it stands before the store.
            let before_store = (self.raises.is_some()
                && !may_raise
                && self.stores_image(op, state))
            .then(|| Raised {
                state: state.clone(),
                trace: trace.clone(),
                at: position.next,
            });
            if let Some(uses) = &mut trace.uses {
                uses.begin();
            }
            let generation = state.memory().instruction_sources().generation();
            let provenance = Provenance::new(self.ssa, operator.access);
            provenance.before(op, state);
            let executed = match control {
                Some((outcome, stack_bits)) => {
                    self.execute_control(op, state, trace, outcome, stack_bits);
                    Ok(ReadEffect::default())
                }
                None => match control_results {
                    Some((outcome, stack_bits)) => {
                        operator.execute_results_control(op, state, trace, outcome, stack_bits);
                        Ok(ReadEffect::default())
                    }
                    None => operator.execute(op, state, trace),
                },
            };
            state.memory_mut().set_through(Through::default());
            ran = ran.saturating_add(1);
            position.next = position.next.saturating_add(1);
            if state.identity_exhausted() {
                break Stop::IdentityExhausted;
            }
            if self.layouts.is_some_and(TypeLayouts::is_unsupported) {
                break Stop::UnsupportedType;
            }
            if let Some(access) = state.memory().fault() {
                state
                    .memory()
                    .record(FindingKind::ReadOnlyWrite, Some(access), None);
                if let (Some(raises), Some(before)) = (self.raises, before_store) {
                    raises.borrow_mut().push(before);
                }
                break Stop::Faulted;
            }
            let reads = match executed {
                Ok(reads) => reads,
                Err(AccessError::Refused(reason)) => break Stop::Refused(reason),
                Err(AccessError::Exhausted(exhaustion)) => break Stop::Exhausted(exhaustion),
            };
            provenance.after(op, state);
            if let Some(marker) = self.marker {
                let loaded = reads.value.and_then(|(slot, defined)| {
                    reads
                        .footprints
                        .get(slot)
                        .copied()
                        .flatten()
                        .map(|span| (span, defined))
                });
                marker.after(op, state, loaded, (body, block, index));
            }
            spanned = spanned.saturating_add(Self::spanned(op, state, &reads));
            if let Some(reason) = state.memory().refusal() {
                break Stop::Refused(reason);
            }
            if matches!(
                op,
                SsaOp::SystemOp(data)
                    if data.kind == SystemOpKind::Barrier(BarrierOp::InstructionSync)
            ) && self
                .host
                .is_some_and(|host| host.synchronizes_instruction_fetch(self.ssa, instruction))
            {
                state
                    .memory_mut()
                    .instruction_sources_mut()
                    .synchronize_fetch();
            }
            operator.record_use_coverage(op, &reads, trace);
            operator.record_read_alignment(state, op, &reads, trace);
            if let Some(observed) = observed.as_deref_mut()
                && trace.clean_headers().next().is_some()
            {
                let value = reads
                    .value
                    .and_then(|(_, value)| state.value(value).into_constant());
                for (slot, footprint) in reads.footprints.into_iter().enumerate() {
                    let Some(footprint) = footprint else {
                        continue;
                    };
                    let produces = reads.value.is_some_and(|(result, _)| result == slot);
                    observed.read(
                        trace,
                        (block, index),
                        footprint,
                        produces,
                        value.as_ref().filter(|_| produces),
                    );
                }
                for span in reads.lanes.iter().flat_map(|lanes| lanes.spans.iter()) {
                    observed.read(trace, (block, index), *span, false, None);
                }
            }
            if state.memory().instruction_sources().generation() != generation {
                break Stop::CodeChanged(index);
            }
        };
        if let Some(frame) = state.frame_mut() {
            frame.move_to(position);
        }
        (
            stop,
            Ran {
                instructions: ran,
                steps: ran.saturating_add(spanned),
            },
        )
    }

    /// Collects exact memory uses while withholding bounds for unsupported footprints.
    fn record_use_coverage(&self, op: &SsaOp<T>, reads: &ReadEffect, trace: &mut Trace) {
        let Some(uses) = &trace.uses else {
            return;
        };
        for (address, length) in reads.footprints.into_iter().flatten().chain(
            reads
                .lanes
                .iter()
                .flat_map(|lanes| lanes.spans.iter().copied()),
        ) {
            uses.access(address, length);
        }
        let effects = op.effects();
        let complete_reads = reads.empty
            || !effects.reads_memory()
            || match op {
                // A fill's pattern and count are SSA values. Its destination
                // is the only data-memory access despite the coarse ReadWrite class.
                SsaOp::InitBlk { .. } => true,
                SsaOp::LoadIndirect { .. }
                | SsaOp::VectorLoad { .. }
                | SsaOp::VectorBroadcastLoad { .. }
                | SsaOp::CopyBlk { .. }
                | SsaOp::MoveBlk { .. }
                | SsaOp::ScanBlk { .. } => reads.footprints[0].is_some(),
                SsaOp::CompareBlk(_) => reads.footprints.iter().all(Option::is_some),
                SsaOp::LoadBlk { .. } => reads.value.is_some_and(|(slot, _)| {
                    reads
                        .footprints
                        .get(..=slot)
                        .is_some_and(|spans| spans.iter().all(Option::is_some))
                }),
                _ => reads.lanes.as_ref().is_some_and(|lanes| lanes.complete),
            };
        let supported_write = matches!(
            op,
            SsaOp::StoreIndirect { .. }
                | SsaOp::VectorStore { .. }
                | SsaOp::InitBlk { .. }
                | SsaOp::CopyBlk { .. }
                | SsaOp::MoveBlk { .. }
        );
        let complete_writes =
            reads.empty || !effects.writes_memory() || (supported_write && uses.wrote());
        if !complete_reads || !complete_writes || matches!(effects.kind, SsaEffectKind::Opaque) {
            uses.unresolved();
        }
    }

    /// Marks visits whose observable effects or access permissions cannot be replayed as byte writes.
    fn record_effects(&self, op: &SsaOp<T>, trace: &mut Trace, empty: bool) {
        let effects = op.effects();
        let memory_transfer = match op {
            SsaOp::LoadIndirect { access, .. }
            | SsaOp::VectorLoad { access, .. }
            | SsaOp::VectorBroadcastLoad { access, .. }
            | SsaOp::StoreIndirect { access, .. }
            | SsaOp::VectorStore { access, .. }
            | SsaOp::InitBlk { access, .. }
            | SsaOp::ScanBlk { access, .. }
            | SsaOp::LoadBlk { access, .. }
            | SsaOp::VectorMaskedLoad { access, .. }
            | SsaOp::VectorGather { access, .. }
            | SsaOp::VectorFaultingLoad { access, .. }
            | SsaOp::VectorSegmentLoad { access, .. }
            | SsaOp::VectorPackLoad { access, .. } => {
                empty || access.permission == AccessPermission::Current
            }
            // A replicating structure load states no qualifiers: an ordinary
            // read of current memory.
            SsaOp::VectorStructLoadReplicate(_) => true,
            SsaOp::CopyBlk {
                src_access,
                dest_access,
                ..
            }
            | SsaOp::MoveBlk {
                src_access,
                dest_access,
                ..
            } => {
                empty
                    || (src_access.permission == AccessPermission::Current
                        && dest_access.permission == AccessPermission::Current)
            }
            SsaOp::CompareBlk(data) => {
                empty
                    || (data.left_access.permission == AccessPermission::Current
                        && data.right_access.permission == AccessPermission::Current)
            }
            _ => false,
        };
        if effects.volatile
            || effects.ordering.is_some()
            || (!memory_transfer && !effects.is_pure())
        {
            trace.taint();
        }
    }

    /// Discharges only a supported arithmetic check proven normal on this visit.
    fn record_control_effects(&self, op: &SsaOp<T>, trace: &mut Trace, normal: bool) {
        let arithmetic = matches!(
            op,
            SsaOp::Div { .. }
                | SsaOp::Rem { .. }
                | SsaOp::WideDiv { .. }
                | SsaOp::AddOvf { .. }
                | SsaOp::SubOvf { .. }
                | SsaOp::MulOvf { .. }
                | SsaOp::IntConv {
                    overflow_check: true,
                    ..
                }
                | SsaOp::FloatToInt {
                    overflow_check: true,
                    ..
                }
                | SsaOp::Ckfinite { .. }
        );
        // The kernel outcome covers this operation's arithmetic fault only.
        // Equality checks every remaining effect field, including memory,
        // volatility, ordering and control, rather than only the coarse kind.
        if normal && arithmetic && op.effects().with_trap(TrapClass::None) == SsaEffects::pure() {
            return;
        }
        if !normal {
            trace.taint();
        }
        self.record_effects(op, trace, false);
    }

    /// The bytes an access through `addr` may reach, one reaching from
    /// `below` bytes under the address it is made at to `len` bytes past
    /// that, as the first of them and their count: where the address is
    /// known exactly, or is a known base plus an offset whose known bits
    /// bound it (an index masked, of a narrower width, scaled by a stride),
    /// and every byte it may reach
    /// lies inside one object whose extent is proven (a live heap
    /// allocation, an object the host states exactly, the frame of the live
    /// activation holding it). `None` where any of that is not known.
    fn element_span(
        &self,
        state: &State<T>,
        addr: SsaVarId,
        below: usize,
        len: usize,
    ) -> Option<(Address, usize)> {
        let ptr_size = self.access.ptr_size();
        let exact = self.exact_address(state, addr);
        let (base, (low, high), displacement) = match exact {
            Some(address) => (Value::Addr(address), (0, 0), 0),
            None => match self.ssa.get_definition(addr)? {
                SsaOp::PtrAdd {
                    base,
                    index: Some(index),
                    stride,
                    offset,
                    ..
                } => {
                    let (low, high) = state.value(*index).non_negative_bounds()?;
                    let stride = *stride;
                    (
                        state.value(*base),
                        (low.checked_mul(stride)?, high.checked_mul(stride)?),
                        *offset,
                    )
                }
                SsaOp::Add { left, right, .. } => match (state.value(*left), state.value(*right)) {
                    (base @ Value::Addr(_), offset) | (offset, base @ Value::Addr(_)) => {
                        (base, offset.non_negative_bounds()?, 0)
                    }
                    _ => return None,
                },
                _ => return None,
            },
        };
        let Value::Addr(base) = base else {
            return None;
        };
        let first = base.moved(
            displacement
                .checked_add(i64::try_from(low).ok()?)?
                .checked_sub(i64::try_from(below).ok()?)?,
            ptr_size,
        )?;
        let bytes = usize::try_from(high.checked_sub(low)?)
            .ok()?
            .checked_add(len)?;
        let object = match first.root() {
            root @ Root::Heap(_) => {
                let (size, _) = state.memory().object(root)?;
                StorageObject {
                    start: 0,
                    bytes: size,
                    kind: ObjectKind::Heap,
                    source: ExtentSource::Allocator,
                }
            }
            _ => state.proven_object(first)?,
        };
        (u64::try_from(bytes).ok()? <= object.reach(first)?).then_some((first, bytes))
    }

    /// The bytes an operation reaching memory through one pointer whose
    /// address is not known exactly may reach, as the first of them and
    /// their count, where its offset is bounded inside one object
    /// ([`Self::element_span`]). The accesses bounded are every write the
    /// operation makes and, for an atomic operation, every access, all
    /// through that one pointer in the flat space and of a width the
    /// operation states: a scalar's or a vector's storage (each lane a
    /// masked or compressing store may write), a structure's segments, an
    /// atomic's
    /// machine width, a block's greatest size in its direction (both
    /// directions where its direction is not known). A gather or a scatter
    /// reaches its base plus each lane's own index, no run from the base,
    /// and is none of these. `None` where any of that is not known, or where
    /// the pointer's address is.
    fn bounded_access(&self, op: &SsaOp<T>, state: &State<T>) -> Option<(Address, usize)> {
        let atomic = op.effects().memory_semantics == MemoryAccessSemantics::Atomic;
        let (mut served, mut foreign, mut width) = (Served::None, false, None);
        op.for_each_addressed_access_by(
            |_| None,
            |access| {
                if !(access.writes || atomic) {
                    return;
                }
                foreign |= access.qualifiers.address_space.is_some();
                served.add(access.addr);
                width = access.width_bits;
            },
        );
        let pointer = served.one().filter(|_| !foreign)?;
        // Whether every size it reaches is known exactly; a block's whose
        // known bits only bound it is not.
        let mut exact = true;
        // A block reaches its size from its pointer upward, or down to it
        // from the element its pointer names where it walks down: at most
        // the greatest size its known bits allow.
        let mut block = |size: SsaVarId, backward: Option<SsaVarId>, element: usize| {
            let size = state.value(size);
            let size = match size.constant().and_then(ConstValue::as_u64_bits) {
                Some(size) => size,
                None => {
                    exact = false;
                    size.non_negative_bounds()?.1
                }
            };
            let size = usize::try_from(size).ok()?;
            let under = size.checked_sub(element)?;
            match self.downward(state, backward) {
                Some(false) => Some((0, size)),
                Some(true) => Some((under, size)),
                None => Some((under, size.checked_add(under)?)),
            }
        };
        let (below, len) = match op {
            SsaOp::InitBlk {
                value,
                size,
                backward,
                ..
            } => {
                let element = self
                    .variable_layout(*value)
                    .and_then(|layout| layout.storage_bytes(self.access.ptr_size()))?;
                block(*size, *backward, element)?
            }
            SsaOp::CopyBlk {
                size,
                element_bytes,
                backward,
                ..
            } => block(*size, *backward, usize::try_from(*element_bytes).ok()?)?,
            SsaOp::MoveBlk { size, .. } => block(*size, None, 1)?,
            // A store spans its type's storage, a structure store its
            // segments'; an atomic or any other access the width it states.
            SsaOp::StoreIndirect { value_type, .. } => (0, self.storage_bytes(value_type)?),
            SsaOp::VectorStore { vector_type, .. }
            | SsaOp::VectorMaskedStore { vector_type, .. }
            | SsaOp::VectorPackStore { vector_type, .. } => (0, self.storage_bytes(vector_type)?),
            SsaOp::VectorSegmentStore {
                vector_type,
                segments,
                ..
            } => (
                0,
                self.storage_bytes(vector_type)?
                    .checked_mul(usize::try_from(*segments).ok()?)?,
            ),
            _ => {
                let bits = width.filter(|bits| bits % 8 == 0)?;
                (0, usize::try_from(bits / 8).ok()?)
            }
        };
        // An access of exact width at an exact address runs as it is.
        if len == 0
            || len > self.access.buffer_limit()
            || (exact && self.exact_address(state, pointer).is_some())
        {
            return None;
        }
        self.element_span(state, pointer, below, len)
    }

    /// The bytes a value of `ty` occupies in memory.
    fn storage_bytes(&self, ty: &T::Type) -> Option<usize> {
        self.access
            .type_layout::<T>(ty)?
            .storage_bytes(self.access.ptr_size())
    }

    /// Whether `op` reaches memory through a pointer whose address the run
    /// does not know exactly, which may fault.
    fn reaches_unknown_address(&self, op: &SsaOp<T>, state: &State<T>) -> bool {
        let mut unknown = false;
        op.for_each_addressed_access_by(
            |_| None,
            |access| unknown |= self.exact_address(state, access.addr).is_none(),
        );
        if let SsaOp::VectorGather { base, .. } | SsaOp::VectorScatter { base, .. } = op {
            unknown |= self.exact_address(state, *base).is_none();
        }
        unknown
    }

    /// Whether `op` stores through an address it knows exactly in the
    /// image, where the machine may fault the store.
    fn stores_image(&self, op: &SsaOp<T>, state: &State<T>) -> bool {
        let mut stores = false;
        op.for_each_addressed_access_by(
            |_| None,
            |access| {
                stores |= access.writes
                    && self
                        .exact_address(state, access.addr)
                        .is_some_and(|address| address.image_address().is_some());
            },
        );
        if let SsaOp::VectorScatter { base, .. } = op {
            stores |= self
                .exact_address(state, *base)
                .is_some_and(|address| address.image_address().is_some());
        }
        stores
    }

    /// The address `pointer` holds, where it is known exactly: one in no
    /// opaque or stack-derived root.
    fn exact_address(&self, state: &State<T>, pointer: SsaVarId) -> Option<Address> {
        self.access
            .address_of(&state.value(pointer), None)
            .filter(|address| {
                !matches!(
                    address.root(),
                    Root::Opaque { .. } | Root::StackDerived { .. }
                )
            })
    }

    /// Runs an operation whose accesses reach some element of one object
    /// ([`Self::bounded_access`]): each byte it may write becomes unknown,
    /// a value it stores escapes where it holds the private stack and the
    /// object is not private, and every result it defines is unknown. An
    /// atomic one runs so only where the bytes are private, as every
    /// atomic does; elsewhere it is no bounded access. Says whether it ran.
    fn execute_bounded(&self, op: &SsaOp<T>, state: &mut State<T>, trace: &mut Trace) -> bool {
        let Some((start, len)) = self.bounded_access(op, state) else {
            return false;
        };
        if op.effects().memory_semantics == MemoryAccessSemantics::Atomic {
            if !state.memory().private_span(start, len) {
                return false;
            }
            state.set_reservation_possible(true);
        }
        // Whether it writes, as its accesses say: an atomic load's effects
        // count it among the writers, but it writes nothing.
        let (mut pointers, mut writes) = (Vec::new(), false);
        op.for_each_addressed_access_by(
            |_| None,
            |access| {
                pointers.push(access.addr);
                writes |= access.writes;
            },
        );
        if writes {
            // What it stores: every operand but the pointers it reaches
            // memory through.
            let stored: Vec<Value<T>> = op
                .uses()
                .into_iter()
                .filter(|value| !pointers.contains(value))
                .map(|value| state.value(value))
                .collect();
            self.access
                .store_element(state.memory_mut(), start, len, &stored);
        }
        let _ =
            op.try_for_each_def(|def| state.define(def, Value::Unknown).then_some(()).ok_or(()));
        trace.taint();
        true
    }

    fn preflight_object_write(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
    ) -> Result<WritePreflight, AccessError> {
        let (destination, ty, space) = match op {
            SsaOp::InitObj {
                dest_addr,
                value_type,
            }
            | SsaOp::CopyObj {
                dest_addr,
                value_type,
                ..
            } => (*dest_addr, value_type, None),
            SsaOp::StoreObj {
                dest_addr,
                value_type,
                access,
                ..
            } => (*dest_addr, value_type, access.address_space),
            _ => return Ok(WritePreflight::Unrelated),
        };
        let Some(ty) = T::value_type_from_ref(ty) else {
            let pointer = state.value(destination);
            if self.access.has_write_guard()
                && !self.access.admit_unresolved_write(
                    state.memory_mut(),
                    self.access.address_of(&pointer, space),
                    None,
                )
            {
                return Err(state
                    .memory()
                    .refusal()
                    .unwrap_or(MemoryRefusal::UnresolvedWrite)
                    .into());
            }
            return Ok(WritePreflight::Possible);
        };
        let Some(bytes) = self
            .access
            .type_layout::<T>(&ty)
            .and_then(|layout| layout.storage_bytes(self.access.ptr_size()))
        else {
            let pointer = state.value(destination);
            if self.access.has_write_guard()
                && !self.access.admit_unresolved_write(
                    state.memory_mut(),
                    self.access.address_of(&pointer, space),
                    None,
                )
            {
                return Err(state
                    .memory()
                    .refusal()
                    .unwrap_or(MemoryRefusal::UnresolvedWrite)
                    .into());
            }
            return Ok(WritePreflight::Possible);
        };
        if bytes == 0 {
            return Ok(WritePreflight::Empty);
        }
        let limit = self.access.buffer_limit();
        if bytes > limit {
            return Err(Exhaustion {
                budget: Budget::ExecutionBufferBytes,
                limit,
                demand: bytes,
            }
            .into());
        }
        if space.is_some() {
            if !self
                .access
                .admit_unresolved_write(state.memory_mut(), None, Some(bytes))
            {
                return Err(state
                    .memory()
                    .refusal()
                    .unwrap_or(MemoryRefusal::UnresolvedWrite)
                    .into());
            }
            return if self.access.has_write_guard() {
                Err(MemoryRefusal::InstructionSource.into())
            } else {
                Ok(WritePreflight::Possible)
            };
        }
        let pointer = state.value(destination);
        self.access.preflight_block_write(
            state.memory_mut(),
            &pointer,
            bytes,
            1,
            Some(false),
            true,
        )?;
        Ok(WritePreflight::Possible)
    }

    /// Preserves known destinations when a guarded block operation otherwise falls back to havoc.
    fn preflight_guarded_block(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
    ) -> Result<WritePreflight, AccessError> {
        if !self.access.has_write_guard() {
            return Ok(WritePreflight::Unrelated);
        }
        let (dest, size, element, direction, source_supported, destination_supported) = match op {
            SsaOp::InitBlk {
                dest_addr,
                value,
                size,
                backward,
                access,
            } => (
                *dest_addr,
                *size,
                self.variable_layout(*value)
                    .and_then(|layout| layout.storage_layout(self.access.ptr_size()))
                    .map(|layout| layout.bytes()),
                self.downward(state, *backward),
                true,
                access.address_space.is_none(),
            ),
            SsaOp::CopyBlk {
                dest_addr,
                size,
                element_bytes,
                backward,
                src_access,
                dest_access,
                ..
            } => (
                *dest_addr,
                *size,
                Some(*element_bytes as usize),
                self.downward(state, *backward),
                src_access.address_space.is_none(),
                dest_access.address_space.is_none(),
            ),
            SsaOp::MoveBlk {
                dest_addr,
                size,
                src_access,
                dest_access,
                ..
            } => (
                *dest_addr,
                *size,
                Some(1),
                Some(false),
                src_access.address_space.is_none(),
                dest_access.address_space.is_none(),
            ),
            _ => return Ok(WritePreflight::Unrelated),
        };
        let value = state.value(size);
        let Some(constant) = value.constant() else {
            let pointer = state.value(dest);
            if !self.access.admit_unresolved_write(
                state.memory_mut(),
                self.access.address_of(&pointer, None),
                None,
            ) {
                return Err(state
                    .memory()
                    .refusal()
                    .unwrap_or(MemoryRefusal::UnresolvedWrite)
                    .into());
            }
            return Ok(WritePreflight::Possible);
        };
        let count = constant.as_u64().ok_or(MemoryRefusal::AddressRange)?;
        let len = usize::try_from(count).map_err(|_| MemoryRefusal::AddressRange)?;
        if len > self.access.block_limit() {
            return Err(AccessError::Exhausted(Exhaustion {
                budget: Budget::ExecutionBlockBytes,
                limit: self.access.block_limit(),
                demand: len,
            }));
        }
        if !destination_supported {
            if !self
                .access
                .admit_unresolved_write(state.memory_mut(), None, Some(len))
            {
                return Err(state
                    .memory()
                    .refusal()
                    .unwrap_or(MemoryRefusal::UnresolvedWrite)
                    .into());
            }
            return Err(MemoryRefusal::InstructionSource.into());
        }
        if !source_supported || direction.is_none() {
            let width = element
                .filter(|width| *width != 0)
                .ok_or(MemoryRefusal::BlockLayout)?;
            let pointer = state.value(dest);
            self.access.preflight_block_write(
                state.memory_mut(),
                &pointer,
                len,
                width,
                direction,
                true,
            )?;
        }
        Ok(WritePreflight::Possible)
    }

    /// Returns the steps a block instruction costs beyond its own once it has
    /// run: one per byte it spanned, as the loop it stands for costs. A block
    /// whose extent is unknown, one that did not run, and any other
    /// instruction, cost none.
    fn spanned(op: &SsaOp<T>, state: &State<T>, reads: &ReadEffect) -> usize {
        let known = |var: SsaVarId| {
            state
                .value(var)
                .into_constant()
                .and_then(|constant| constant.as_u64_bits())
        };
        let bytes = match op {
            SsaOp::InitBlk { size, .. }
            | SsaOp::CopyBlk { size, .. }
            | SsaOp::MoveBlk { size, .. } => known(*size),
            SsaOp::ScanBlk {
                remaining,
                count,
                element_bytes,
                ..
            } => known(*count)
                .zip(known(*remaining))
                .map(|(count, remaining)| {
                    count
                        .saturating_sub(remaining)
                        .saturating_mul(u64::from(*element_bytes))
                }),
            SsaOp::CompareBlk(data) => {
                known(data.count)
                    .zip(known(data.remaining))
                    .map(|(count, remaining)| {
                        count
                            .saturating_sub(remaining)
                            .saturating_mul(u64::from(data.element_bytes))
                    })
            }
            // A repeated load reads its block only where it ran: its span is
            // the preceding elements and the last.
            SsaOp::LoadBlk { .. } => Some(
                reads
                    .footprints
                    .iter()
                    .flatten()
                    .map(|(_, len)| u64::try_from(*len).unwrap_or(u64::MAX))
                    .fold(0, u64::saturating_add),
            ),
            _ => None,
        };
        bytes.map_or(0, |bytes| usize::try_from(bytes).unwrap_or(usize::MAX))
    }

    /// Executes the no-access case before consulting address spaces, permissions or direction.
    ///
    /// A zero-width element is malformed and establishes no no-access proof. Volatile
    /// and ordering effects remain subject to ordinary effect checks. A repeated
    /// compare, scan or load that states its count is never zero names no value
    /// an empty run leaves: run empty, it refuses
    /// ([`MemoryRefusal::EmptyBlockInput`]).
    fn execute_empty_block(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
        trace: &mut Trace,
    ) -> Result<bool, MemoryRefusal> {
        let left = |input: Option<SsaVarId>| {
            input
                .map(|input| state.value(input))
                .ok_or(MemoryRefusal::EmptyBlockInput)
        };
        match op {
            SsaOp::InitBlk { size, .. } | SsaOp::MoveBlk { size, .. }
                if state.value(*size).is_zero_count() => {}
            SsaOp::CopyBlk {
                size,
                element_bytes,
                ..
            } if *element_bytes != 0 && state.value(*size).is_zero_count() => {}
            SsaOp::CompareBlk(data)
                if data.element_bytes != 0 && state.value(data.count).is_zero_count() =>
            {
                let flags_in = left(data.flags_in)?;
                self.define_block_results(
                    op,
                    state,
                    trace,
                    Some([
                        (data.remaining, state.value(data.count)),
                        (data.left_next, state.value(data.left)),
                        (data.right_next, state.value(data.right)),
                        (data.flags, flags_in),
                    ]),
                );
            }
            SsaOp::ScanBlk {
                remaining,
                next,
                flags,
                address,
                count,
                flags_in,
                element_bytes,
                ..
            } if *element_bytes != 0 && state.value(*count).is_zero_count() => {
                let flags_in = left(*flags_in)?;
                self.define_block_results(
                    op,
                    state,
                    trace,
                    Some([
                        (*remaining, state.value(*count)),
                        (*next, state.value(*address)),
                        (*flags, flags_in),
                    ]),
                );
            }
            SsaOp::LoadBlk {
                value,
                next,
                address,
                count,
                value_in,
                element_bytes,
                ..
            } if *element_bytes != 0 && state.value(*count).is_zero_count() => {
                let value_in = left(*value_in)?;
                self.define_block_results(
                    op,
                    state,
                    trace,
                    Some([(*value, value_in), (*next, state.value(*address))]),
                );
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Installs an already evaluated control value without resolving its operands again.
    fn execute_control(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
        trace: &mut Trace,
        outcome: Evaluation<ConstValue<T>>,
        stack_bits: bool,
    ) {
        self.record_control_effects(op, trace, matches!(&outcome, Evaluation::Value(_)));
        let held = match outcome {
            Evaluation::Value(value) => Value::Const(value),
            Evaluation::Unknown | Evaluation::Unsupported if stack_bits => Value::StackDerived,
            Evaluation::Unknown | Evaluation::Unsupported => Value::Unknown,
            Evaluation::Trap(_) => return,
        };
        if let Some(dest) = op.dest() {
            state.define(dest, held);
        }
        self.define_rest_unknown(op, state);
    }

    /// Defines every wide result while preserving a failed control evaluation as unknown.
    fn execute_results_control(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
        trace: &mut Trace,
        outcome: Evaluation<FoldedResults<T>>,
        stack_bits: bool,
    ) {
        self.record_control_effects(op, trace, matches!(&outcome, Evaluation::Value(_)));
        match outcome {
            Evaluation::Value(results) => {
                for (def, held) in results {
                    if !state.define(def, Value::Const(held)) {
                        return;
                    }
                }
            }
            Evaluation::Unknown | Evaluation::Unsupported => {
                let held = if stack_bits {
                    Value::StackDerived
                } else {
                    Value::Unknown
                };
                let _ = op.try_for_each_def(|def| {
                    state.define(def, held.clone()).then_some(()).ok_or(())
                });
            }
            Evaluation::Trap(_) => {}
        }
    }

    /// Returns one result's materialized extent under the existing buffer policy.
    fn buffer_demand(dest: Destination, payload: usize) -> usize {
        dest.bits
            .filter(|bits| {
                dest.lanes.is_some() || dest.integer && !matches!(bits, 1 | 8 | 16 | 32 | 64)
            })
            .and_then(|bits| usize::try_from(bits.div_ceil(8)).ok())
            .unwrap_or(0)
            .max(payload)
    }

    /// Defines every result of a proved inactive computation before admitting effects.
    fn execute_inactive(&self, op: &SsaOp<T>, state: &mut State<T>) -> Result<bool, AccessError> {
        let Some(plan) = inactive::plan(
            op,
            self.access.ptr_size(),
            |var| state.frame()?.held_value(var)?.constant(),
            |var| self.variable_layout(var),
            |left, right| {
                self.ssa
                    .variable(left)
                    .zip(self.ssa.variable(right))
                    .is_some_and(|(left, right)| left.var_type() == right.var_type())
            },
        ) else {
            return Ok(false);
        };
        let limit = self.access.buffer_limit();
        // Admit every declared result and every existing payload before cloning
        // a passthrough or changing a destination. Inline scalars keep their
        // existing policy; a malformed held carrier cannot hide behind its type.
        for result in &plan.results {
            let payload = match &result.definition {
                SsaOp::Copy { src, .. } => state
                    .frame()
                    .and_then(|frame| frame.held_value(*src))
                    .map_or(0, Value::owned_bytes),
                SsaOp::Const { value, .. } => value.payload_bytes(),
                SsaOp::RuntimeValue {
                    kind: RuntimeValueKind::Unspecified,
                    ..
                } => 0,
                _ => return Ok(false),
            };
            let demand = Self::buffer_demand(Destination::from_type_layout(result.layout), payload);
            if demand > limit {
                return Err(Exhaustion {
                    budget: Budget::ExecutionBufferBytes,
                    limit,
                    demand,
                }
                .into());
            }
        }
        // A Compute defines its outputs together. Capture all sources from
        // the pre-instruction state, preserving partial facts and addresses.
        let mut prepared = Vec::with_capacity(plan.results.len());
        for result in &plan.results {
            let (dest, held) = match &result.definition {
                SsaOp::Copy { dest, src } => (*dest, state.value(*src)),
                SsaOp::RuntimeValue {
                    dest,
                    kind: RuntimeValueKind::Unspecified,
                } => (*dest, Value::Unknown),
                SsaOp::Const { dest, .. } => (
                    *dest,
                    self.scalar.evaluate_with_operand_layouts(
                        &result.definition,
                        Destination::from_type_layout(result.layout),
                        |_| Value::Unknown,
                        |_| OperandLayout::default(),
                    ),
                ),
                _ => return Ok(false),
            };
            prepared.push((dest, held));
        }
        for (dest, held) in prepared {
            if !state.define(dest, held) {
                break;
            }
        }
        Ok(true)
    }

    /// Runs one instruction that is neither a terminator nor a call.
    fn execute(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
        trace: &mut Trace,
    ) -> Result<ReadEffect, AccessError> {
        if self.execute_inactive(op, state)? {
            return Ok(ReadEffect {
                empty: true,
                ..ReadEffect::default()
            });
        }
        let mut reads = ReadEffect::default();
        let empty = self.execute_empty_block(op, state, trace)?;
        self.record_effects(op, trace, empty);
        if empty {
            reads.empty = true;
            return Ok(reads);
        }
        if self.execute_bounded(op, state, trace) {
            return Ok(reads);
        }
        let block_write = self.preflight_guarded_block(op, state)?;
        if Atomics::new(self.access, self.scalar)
            .execute(op, state, |value| self.variable_layout(value))?
        {
            return Ok(reads);
        }
        // A lane-wise vector load whose lanes are known reads each active
        // element as a load does; one whose lanes are not stays opaque below.
        let lanes = VectorReads::new(self.access);
        if let Some(Some(plan)) = lanes.plan(
            op,
            state,
            |value| self.variable_layout(value),
            |value| self.operand_layout(value),
        )? {
            let outcome = lanes.execute(&plan, state, |value| self.variable_layout(value))?;
            if !outcome.complete {
                trace.taint();
            }
            for (dest, value) in outcome.results {
                if matches!(value, Value::Unknown | Value::StackDerived) {
                    // An unused unknown read can still fault.
                    trace.taint();
                }
                state.define(dest, value);
            }
            reads.lanes = Some(LaneSpans {
                spans: outcome.spans,
                unit: plan.element_bytes(),
                access: plan.access(),
                complete: outcome.complete,
            });
            return Ok(reads);
        }
        match op {
            // The selector a runtime hands the handler it enters, where the
            // raise delivered one: the clause its personality chose. The
            // exception handed with it reaches only the runtime's routines,
            // which take it from the run's exceptions, so it stays unknown.
            SsaOp::RuntimeValue {
                dest,
                kind: RuntimeValueKind::HandlerEntry,
            } if state.exceptions().selector().is_some() => {
                let bits = self
                    .variable_layout(*dest)
                    .and_then(TypeLayout::bit_width)
                    .unwrap_or_else(|| self.access.ptr_size().bits());
                let value = state
                    .exceptions()
                    .selector()
                    .and_then(|selector| ConstValue::integer(bits, true, selector))
                    .map_or(Value::Unknown, Value::Const);
                state.define(*dest, value);
                return Ok(reads);
            }
            SsaOp::CallClobber { outputs, machine } => {
                state.define_call_outputs(
                    outputs,
                    machine.as_deref(),
                    |output| self.variable_layout(output),
                    self.access.ptr_size(),
                )?;
            }
            SsaOp::LoadIndirect {
                dest,
                addr,
                value_type,
                access: space,
            }
            | SsaOp::VectorLoad {
                dest,
                addr,
                vector_type: value_type,
                access: space,
            } => {
                let loaded = self.access.load_with_footprint(
                    state.memory(),
                    &state.value(*addr),
                    value_type,
                    space.address_space,
                )?;
                reads.footprints[0] = loaded.footprint;
                reads.value = Some((0, *dest));
                if matches!(loaded.value, Value::Unknown | Value::StackDerived) {
                    // An unused unknown read can still fault; known output
                    // bytes do not justify bypassing it.
                    trace.taint();
                }
                state.define(*dest, loaded.value);
            }
            // One element read, repeated in every lane: the element is a
            // lane's width of the bytes at the address.
            SsaOp::VectorBroadcastLoad {
                dest,
                addr,
                vector_type,
                access: space,
            } => {
                // The memory read is one element; the vector's own shape
                // says how wide and how many.
                let element = T::vector_shape(vector_type)
                    .and_then(|shape| Some((shape, T::integer_type(shape.lane_bits, false)?)));
                let Some((shape, element)) = element else {
                    trace.taint();
                    state.define(*dest, Value::Unknown);
                    return Ok(reads);
                };
                let loaded = self.access.load_with_footprint(
                    state.memory(),
                    &state.value(*addr),
                    &element,
                    space.address_space,
                )?;
                reads.footprints[0] = loaded.footprint;
                reads.value = Some((0, *dest));
                let splat = loaded
                    .value
                    .constant()
                    .and_then(|value| value.scalar_integer_pattern(self.access.ptr_size()))
                    .and_then(|(bits, _)| {
                        VectorConst::splat(shape.lane_bits, shape.lane_count, bits)
                    })
                    .map(|vector| Value::Const(ConstValue::Vector(vector)));
                let Some(splat) = splat else {
                    trace.taint();
                    state.define(*dest, Value::Unknown);
                    return Ok(reads);
                };
                state.define(*dest, splat);
            }
            SsaOp::StoreIndirect {
                addr,
                value,
                value_type,
                access: space,
            }
            | SsaOp::VectorStore {
                addr,
                value,
                vector_type: value_type,
                access: space,
            } => {
                let (pointer, stored) = (state.value(*addr), state.value(*value));
                self.record(
                    state,
                    trace,
                    self.access
                        .footprint(&pointer, value_type, space.address_space),
                    None,
                    *space,
                );
                self.access.store(
                    state.memory_mut(),
                    &pointer,
                    &stored,
                    value_type,
                    space.address_space,
                )?;
            }
            // The block forms run here in the flat space alone, which the
            // modelled memory is; one in another space clobbers as any other
            // write does.
            SsaOp::InitBlk {
                dest_addr,
                value,
                size,
                backward,
                access: space,
            } if space.address_space.is_none() && self.downward(state, *backward).is_some() => {
                let reverse = self.downward(state, *backward).unwrap_or_default();
                let size = state.value(*size);
                let pointer = state.value(*dest_addr);
                let fill = state.value(*value);
                let storage = self
                    .variable_layout(*value)
                    .and_then(|layout| layout.storage_layout(self.access.ptr_size()));
                self.record(
                    state,
                    trace,
                    if reverse {
                        storage.and_then(|storage| {
                            self.access
                                .stream_footprint(&pointer, &size, storage.bytes(), true)
                        })
                    } else {
                        self.access.block_footprint(&pointer, &size)
                    },
                    Some(storage.map_or(0, |view| view.bytes())),
                    *space,
                );
                self.access.fill_storage(
                    state.memory_mut(),
                    &pointer,
                    &fill,
                    storage,
                    &size,
                    reverse,
                );
            }
            SsaOp::CopyBlk {
                dest_addr,
                src_addr,
                size,
                element_bytes,
                backward,
                src_access: src_space,
                dest_access: dest_space,
            } if src_space.address_space.is_none()
                && dest_space.address_space.is_none()
                && self.downward(state, *backward).is_some() =>
            {
                let reverse = self.downward(state, *backward).unwrap_or_default();
                let size = state.value(*size);
                let pointer = state.value(*dest_addr);
                let source = state.value(*src_addr);
                let destination =
                    self.access
                        .stream_footprint(&pointer, &size, *element_bytes as usize, reverse);
                reads.footprints[0] =
                    self.access
                        .stream_footprint(&source, &size, *element_bytes as usize, reverse);
                if self.access.copy_stream(
                    state.memory_mut(),
                    &pointer,
                    &source,
                    &size,
                    *element_bytes as usize,
                    reverse,
                ) {
                    self.record(
                        state,
                        trace,
                        destination,
                        Some(*element_bytes as usize),
                        *dest_space,
                    );
                } else {
                    trace.taint();
                }
            }
            // A repeated compare, scan or load runs here in the flat space
            // alone, as a block copy does, and where its count, its pointers
            // and the elements it reads are known; elsewhere its results are
            // unknown, as an unmodelled read's are.
            SsaOp::CompareBlk(data)
                if data.left_access.address_space.is_none()
                    && data.right_access.address_space.is_none()
                    && self.downward(state, data.backward).is_some() =>
            {
                let CompareBlkData {
                    remaining,
                    left_next,
                    right_next,
                    flags,
                    left,
                    right,
                    count,
                    flags_in,
                    backward,
                    element_bytes,
                    while_equal,
                    ..
                } = **data;
                let down = self.downward(state, backward).unwrap_or_default();
                let results = self.compared(
                    state,
                    left,
                    Against::Block(right),
                    count,
                    flags_in,
                    element_bytes,
                    while_equal,
                    down,
                )?;
                if let Some(walked) = &results {
                    reads.footprints = [
                        self.repeated_footprint(
                            &state.value(left),
                            walked.elements,
                            element_bytes,
                            down,
                        ),
                        self.repeated_footprint(
                            &state.value(right),
                            walked.elements,
                            element_bytes,
                            down,
                        ),
                    ];
                }
                self.define_block_results(
                    op,
                    state,
                    trace,
                    results.map(|walked| {
                        [
                            (remaining, walked.remaining),
                            (left_next, walked.left_next),
                            (right_next, walked.right_next),
                            (flags, walked.flags),
                        ]
                    }),
                );
            }
            SsaOp::ScanBlk {
                remaining,
                next,
                flags,
                address,
                value,
                count,
                flags_in,
                backward,
                element_bytes,
                while_equal,
                access,
            } if access.address_space.is_none() && self.downward(state, *backward).is_some() => {
                let down = self.downward(state, *backward).unwrap_or_default();
                let results = self.compared(
                    state,
                    *address,
                    Against::Value(*value),
                    *count,
                    *flags_in,
                    *element_bytes,
                    *while_equal,
                    down,
                )?;
                if let Some(walked) = &results {
                    reads.footprints[0] = self.repeated_footprint(
                        &state.value(*address),
                        walked.elements,
                        *element_bytes,
                        down,
                    );
                }
                self.define_block_results(
                    op,
                    state,
                    trace,
                    results.map(|walked| {
                        [
                            (*remaining, walked.remaining),
                            (*next, walked.left_next),
                            (*flags, walked.flags),
                        ]
                    }),
                );
            }
            // A repeated load leaves the last element it reads, or `value_in`
            // where it reads none.
            SsaOp::LoadBlk {
                value,
                next,
                address,
                count,
                backward,
                element_bytes,
                access,
                ..
            } if access.address_space.is_none() && self.downward(state, *backward).is_some() => {
                let down = self.downward(state, *backward).unwrap_or_default();
                let pointer = state.value(*address);
                let elements = state
                    .value(*count)
                    .constant()
                    .and_then(ConstValue::as_u64_bits);
                let results = match elements {
                    Some(elements) => {
                        match self.repeated_footprint(&pointer, elements, *element_bytes, down) {
                            Some((address, len)) => {
                                if !self.access.readable_span(state.memory(), address, len)? {
                                    trace.taint();
                                }
                                match ReadEffect::repeated(
                                    address,
                                    len,
                                    *element_bytes,
                                    down,
                                    *value,
                                ) {
                                    Some(effect) => reads = effect,
                                    None => trace.taint(),
                                }
                                self.element(
                                    state.memory(),
                                    &pointer,
                                    elements.saturating_sub(1),
                                    *element_bytes,
                                    down,
                                )?
                                .map(|last| {
                                    [
                                        (*value, last),
                                        (
                                            *next,
                                            self.advanced(pointer, elements, *element_bytes, down),
                                        ),
                                    ]
                                })
                            }
                            None => None,
                        }
                    }
                    None => None,
                };
                self.define_block_results(op, state, trace, results);
            }
            SsaOp::MoveBlk {
                dest_addr,
                src_addr,
                size,
                src_access: src_space,
                dest_access: dest_space,
            } if src_space.address_space.is_none() && dest_space.address_space.is_none() => {
                let size = state.value(*size);
                let pointer = state.value(*dest_addr);
                let source = state.value(*src_addr);
                let destination = self.access.block_footprint(&pointer, &size);
                let buffered = matches!(op, SsaOp::MoveBlk { .. });
                reads.footprints[0] = self.access.block_footprint(&source, &size);
                if self
                    .access
                    .copy(state.memory_mut(), &pointer, &source, &size, buffered)
                {
                    self.record(state, trace, destination, Some(1), *dest_space);
                } else {
                    trace.taint();
                }
            }
            _ => {
                let effects = op.effects();
                if matches!(effects.kind, SsaEffectKind::Pure) {
                    if let Some(results) = self.scalar.evaluate_results(
                        op,
                        |var| state.value(var),
                        |var| self.operand_layout(var),
                    ) {
                        for (def, held) in results {
                            if !state.define(def, held) {
                                return Ok(reads);
                            }
                        }
                        return Ok(reads);
                    }
                    if let Some(dest) = op.dest() {
                        let heap_bits = |address| {
                            let host = self.host?;
                            if state.memory().is_sound(address, 1) {
                                host.heap_address_bits(address)
                            } else {
                                None
                            }
                        };
                        let held = self
                            .scalar
                            .with_stack_congruences(&*state)
                            .with_heap_address_bits(&heap_bits)
                            .evaluate_with_operand_layouts(
                                op,
                                self.destination(op),
                                |var| state.value(var),
                                |var| self.operand_layout(var),
                            );
                        state.define(dest, held);
                    }
                    self.define_rest_unknown(op, state);
                    return Ok(reads);
                }
                let vector_write = VectorWrites::new(self.access)
                    .preflight(op, state, |id| self.operand_layout(id))?;
                if vector_write == WritePreflight::Empty {
                    return Ok(reads);
                }
                let object_write = self.preflight_object_write(op, state)?;
                if object_write == WritePreflight::Empty {
                    return Ok(reads);
                }
                if effects.writes_memory()
                    && [block_write, vector_write, object_write]
                        .into_iter()
                        .all(|write| write == WritePreflight::Unrelated)
                    && !matches!(op, SsaOp::SystemOp(data) if matches!(data.kind, SystemOpKind::Barrier(_)))
                    && !self
                        .access
                        .admit_unresolved_write(state.memory_mut(), None, None)
                {
                    return Err(state
                        .memory()
                        .refusal()
                        .unwrap_or(MemoryRefusal::UnresolvedWrite)
                        .into());
                }
                // An unsupported read can fault even when none of its results is used.
                trace.taint();
                state.set_reservation_possible(true);
                let stack_bits = self
                    .scalar
                    .references_stack(op, &|operand| state.value(operand));
                // Opaque architectural state can retain a pointer even when it
                // does not access data memory in this instruction.
                if stack_bits
                    && (effects.writes_memory()
                        || matches!(effects.kind, SsaEffectKind::Opaque | SsaEffectKind::Call))
                {
                    state.memory_mut().escape_stack();
                }
                if effects.writes_memory() {
                    state.memory_mut().clobber();
                    trace.taint();
                }
                let _ = op.try_for_each_def(|def| {
                    state
                        .define(
                            def,
                            if stack_bits {
                                Value::StackDerived
                            } else {
                                Value::Unknown
                            },
                        )
                        .then_some(())
                        .ok_or(())
                });
            }
        }
        Ok(reads)
    }

    /// Records the bytes this run wrote behind the pointers `op` uses, before
    /// it executes ([`StaticFacts::uses`]): the address a load reads at, the
    /// pointer a store writes into memory, either side of a block compare.
    /// A visit whose pointer reaches nothing the run wrote records nothing for
    /// it, which the join then drops.
    ///
    /// [`StaticFacts::uses`]: crate::analysis::execution::StaticFacts::uses
    fn observe_use(
        &self,
        op: &SsaOp<T>,
        state: &State<T>,
        site: (usize, usize),
        observed: &mut Observations<T>,
    ) {
        let pointers: &[SsaVarId] = match op {
            SsaOp::LoadIndirect { addr, access, .. } if access.address_space.is_none() => &[*addr],
            SsaOp::StoreIndirect { value, access, .. } if access.address_space.is_none() => {
                &[*value]
            }
            SsaOp::CompareBlk(data)
                if data.left_access.address_space.is_none()
                    && data.right_access.address_space.is_none() =>
            {
                &[data.left, data.right]
            }
            _ => return,
        };
        let uses = op.uses();
        let seen: Vec<(usize, Option<ArgumentBytes>)> = pointers
            .iter()
            .filter_map(|pointer| {
                let argument = uses.iter().position(|used| used == pointer)?;
                // A pointer this visit does not know the place of says nothing
                // either way: one into the stack at an offset paths with
                // different counts merged away, or into storage no root names.
                let value = state.value(*pointer);
                let address = self.access.address_of(&value, None)?;
                if matches!(
                    address.root(),
                    Root::StackDerived { .. } | Root::Opaque { .. }
                ) {
                    return None;
                }
                let stated = |address| state.proven_object(address);
                let bytes = self
                    .access
                    .written_prefix(state.memory(), &value, stated)
                    .map(|(address, bytes, bound)| ArgumentBytes {
                        argument,
                        address,
                        written: state.memory().written(address, bytes.len()),
                        bytes,
                        bound,
                        allocated_at: state.memory().allocation_site(address.root()),
                    });
                Some((argument, bytes))
            })
            .collect();
        observed.use_site(site, seen);
    }

    /// Defines every value `op` defines besides its primary destination as
    /// unknown: a flags word, a second result.
    fn define_rest_unknown(&self, op: &SsaOp<T>, state: &mut State<T>) {
        let primary = op.dest();
        let _ = op.try_for_each_def(|def| {
            if Some(def) == primary {
                return Ok(());
            }
            state.define(def, Value::Unknown).then_some(()).ok_or(())
        });
    }

    /// Defines the results of a repeated compare, scan or load, or, where
    /// they are not known, every value `op` defines as unknown; a read of
    /// unknown elements can still fault, so the path is spoiled as an unknown
    /// load's is.
    fn define_block_results<const N: usize>(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
        trace: &mut Trace,
        results: Option<[(SsaVarId, Value<T>); N]>,
    ) {
        match results {
            Some(results) => {
                for (def, held) in results {
                    if !state.define(def, held) {
                        return;
                    }
                }
            }
            None => {
                trace.taint();
                let _ = op.try_for_each_def(|def| {
                    state.define(def, Value::Unknown).then_some(()).ok_or(())
                });
            }
        }
    }

    /// Retains replayable writes only with aligned elements and writable destination backing.
    fn record(
        &self,
        state: &State<T>,
        trace: &mut Trace,
        footprint: Option<(Address, usize)>,
        element_bytes: Option<usize>,
        access: AccessQualifiers,
    ) {
        match footprint {
            Some((address, len)) => {
                if trace.clean_headers().next().is_some()
                    && !self.access.proven_span(
                        state,
                        address,
                        len,
                        element_bytes.unwrap_or(len),
                        access,
                    )
                {
                    trace.taint();
                }
                self.access.record_write(state, trace, address, len);
            }
            None => trace.taint(),
        }
    }

    /// Checks only the read elements actually executed, including early stops and the final load.
    fn record_read_alignment(
        &self,
        state: &State<T>,
        op: &SsaOp<T>,
        reads: &ReadEffect,
        trace: &mut Trace,
    ) {
        if reads.empty || trace.clean_headers().next().is_none() {
            return;
        }
        let mut check = |span: Option<(Address, usize)>, unit: Option<usize>, access| {
            if trace.clean_headers().next().is_none() {
                return;
            }
            match span {
                Some((address, len))
                    if self.access.proven_span(
                        state,
                        address,
                        len,
                        unit.unwrap_or(len),
                        access,
                    ) => {}
                _ => trace.taint(),
            }
        };
        match op {
            SsaOp::LoadIndirect { access, .. }
            | SsaOp::VectorLoad { access, .. }
            | SsaOp::VectorBroadcastLoad { access, .. } => {
                check(reads.footprints[0], None, *access);
            }
            SsaOp::CopyBlk {
                element_bytes,
                src_access,
                ..
            } => check(
                reads.footprints[0],
                Some(*element_bytes as usize),
                *src_access,
            ),
            SsaOp::MoveBlk { src_access, .. } => check(reads.footprints[0], Some(1), *src_access),
            SsaOp::CompareBlk(data) => {
                check(
                    reads.footprints[0],
                    Some(data.element_bytes as usize),
                    data.left_access,
                );
                check(
                    reads.footprints[1],
                    Some(data.element_bytes as usize),
                    data.right_access,
                );
            }
            SsaOp::ScanBlk {
                element_bytes,
                access,
                ..
            } => check(reads.footprints[0], Some(*element_bytes as usize), *access),
            SsaOp::LoadBlk {
                element_bytes,
                access,
                ..
            } => {
                if reads.footprints.iter().all(Option::is_none) {
                    check(None, Some(*element_bytes as usize), *access);
                } else {
                    for span in reads.footprints.into_iter().flatten() {
                        check(Some(span), Some(*element_bytes as usize), *access);
                    }
                }
            }
            _ => {
                if let Some(lanes) = &reads.lanes {
                    if !lanes.complete {
                        check(None, Some(lanes.unit), lanes.access);
                    }
                    for span in &lanes.spans {
                        check(Some(*span), Some(lanes.unit), lanes.access);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;
    use crate::{
        analysis::{
            budget::WorkBudget,
            execution::{frame::Frame, value::Root},
            image::{ImageData, ImageRegion, Mutability},
        },
        ir::{
            Callee,
            function::{SsaBlockBuilder, SsaDefSpec, SsaFunctionBuilder},
            integer::IntegerBits,
            ops::{
                AccessQualifiers, AtomicAccessWidth, AtomicOrdering, HintOp, NativeKindedData,
                SysRegOp, SystemOpKind,
            },
            value::ConstValue,
            variable::{DefSite, SsaVarId, VariableOrigin, VariableRole},
        },
        pointer::PointerSize,
        target::Endianness,
        testing::{MACHINE, MockTarget, MockType},
    };

    /// Where [`Zeros`] starts.
    const ZEROS_AT: u64 = 0x1000;

    /// Read-only zero bytes from [`ZEROS_AT`] to the end of the address space.
    struct Zeros;

    impl ImageData for Zeros {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, address: u64) -> Option<ImageRegion> {
            (address >= ZEROS_AT).then_some(ImageRegion {
                start: ZEROS_AT,
                end: u64::MAX,
                mutability: Mutability::ReadOnly,
            })
        }

        fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
            const BYTES: [u8; 16] = [0; 16];
            self.region(address)?;
            BYTES.get(..len)
        }

        fn number_offset(&self) -> Option<i64> {
            Some(0)
        }
    }

    /// Runs a scan of `count` bytes from [`ZEROS_AT`] while they equal zero,
    /// inside a loop: the count it leaves, and whether the loop stays clean.
    fn scan_zeros(count: u64, maximum: usize) -> (Value<MockTarget>, bool) {
        let (mut builder, _, _) = with_inputs();
        let remaining = builder
            .in_block(0, |b| {
                let address =
                    b.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::U64(ZEROS_AT))?;
                let value = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                let count =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(count))?;
                let flags_in = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                let results = b.emit_many_defs(
                    &[
                        SsaDefSpec::tmp(MockType::U64),
                        SsaDefSpec::tmp(MockType::Ptr),
                        SsaDefSpec::tmp(MockType::U64),
                    ],
                    |defs| SsaOp::ScanBlk {
                        remaining: defs[0],
                        next: defs[1],
                        flags: defs[2],
                        address,
                        value,
                        count,
                        flags_in: Some(flags_in),
                        backward: None,
                        element_bytes: 1,
                        while_equal: true,
                        access: AccessQualifiers::default(),
                    },
                )?;
                b.ret(Some(results[0]))?;
                Ok(results[0])
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE)
            .with_image(&Zeros)
            .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionBlockBytes, maximum));
        let mut state = State::new(Frame::enter_at(
            &ssa,
            0,
            0,
            Address::new(Root::Stack, 0, PointerSize::Bit64),
            vec![],
            PointerSize::Bit64,
        ));
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        let (stop, _) = Stepper::new(&ssa, &context).run(&mut state, &mut trace, usize::MAX);
        assert_eq!(stop, Stop::Terminator);
        let clean = trace.loops().iter().all(|visit| visit.clean);
        (state.value(remaining), clean)
    }

    /// A scan that states its count is never zero names no flags for an
    /// empty run, so run with a count of zero it refuses, under a reason of
    /// its own rather than as a scan it could not follow.
    #[test]
    fn an_empty_scan_that_states_no_empty_input_refuses() {
        let (mut builder, _, _) = with_inputs();
        builder
            .in_block(0, |b| {
                let address =
                    b.const_value(SsaDefSpec::tmp(MockType::Ptr), ConstValue::U64(ZEROS_AT))?;
                let value = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                let count = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                let results = b.emit_many_defs(
                    &[
                        SsaDefSpec::tmp(MockType::U64),
                        SsaDefSpec::tmp(MockType::Ptr),
                        SsaDefSpec::tmp(MockType::U64),
                    ],
                    |defs| SsaOp::ScanBlk {
                        remaining: defs[0],
                        next: defs[1],
                        flags: defs[2],
                        address,
                        value,
                        count,
                        flags_in: None,
                        backward: None,
                        element_bytes: 1,
                        while_equal: true,
                        access: AccessQualifiers::default(),
                    },
                )?;
                b.ret(Some(results[2]))?;
                Ok(results[2])
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE).with_image(&Zeros);
        let mut state = State::new(Frame::enter_at(
            &ssa,
            0,
            0,
            Address::new(Root::Stack, 0, PointerSize::Bit64),
            vec![],
            PointerSize::Bit64,
        ));
        let mut trace = Trace::new();
        let (stop, _) = Stepper::new(&ssa, &context).run(&mut state, &mut trace, usize::MAX);
        assert_eq!(stop, Stop::Refused(MemoryRefusal::EmptyBlockInput));
    }

    /// A scan of known bytes compares them all and leaves no count.
    #[test]
    fn a_scan_of_known_bytes_completes() {
        assert_eq!(scan_zeros(16, 16), (Value::Const(ConstValue::U64(0)), true));
    }

    /// A scan longer than one block's limit gives unknown results rather than
    /// comparing without end.
    #[test]
    fn a_scan_past_the_block_limit_is_unknown() {
        for maximum in [0usize, 16] {
            for count in [maximum as u64 + 1, u64::MAX] {
                let (remaining, clean) = scan_zeros(count, maximum);
                assert!(remaining.constant().is_none());
                assert!(!clean);
            }
        }
    }

    /// Builds inline and shared unknown bits that may contain a private stack pointer.
    fn partial_stack_values() -> [Value<MockTarget>; 2] {
        [
            Value::partial_integer(IntegerBits::new(64, 15, 0).unwrap(), true),
            Value::partial_integer(IntegerBits::new(128, 1, 1u128 << 127).unwrap(), true),
        ]
    }

    /// Ends the test: a fixture the builder rejects is a broken test.
    fn abort<E, R>(_: E) -> R {
        process::abort()
    }

    /// A builder with a stack pointer and one parameter live on entry.
    fn with_inputs() -> (SsaFunctionBuilder<MockTarget>, SsaVarId, SsaVarId) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let ssa = builder.function_mut();
        let stack = ssa.create_variable(
            VariableOrigin::EntryLiveIn,
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        if let Some(variable) = ssa.variable_mut(stack) {
            variable.set_role(VariableRole::EntryStackPointer);
        }
        let parameter = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        (builder, stack, parameter)
    }

    /// Accessible guard storage used by the empty-transfer fixtures.
    struct Guard;

    impl ImageData for Guard {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }
        fn region(&self, at: u64) -> Option<ImageRegion> {
            (at == 0x1000).then_some(ImageRegion {
                start: 0x1000,
                end: 0x1001,
                mutability: Mutability::Writable,
            })
        }
        fn bytes(&self, _at: u64, _len: usize) -> Option<&[u8]> {
            None
        }
        fn number_offset(&self) -> Option<i64> {
            Some(0)
        }
    }

    /// Runs block 0 of `ssa` from its start, with `bound` values.
    fn run(
        ssa: &SsaFunction<MockTarget>,
        bound: Vec<(SsaVarId, Value<MockTarget>)>,
        trace: &mut Trace,
    ) -> (State<MockTarget>, Stop, usize) {
        let context = AnalysisContext::new(MACHINE).with_image(&Guard);
        let mut state = State::new(
            Frame::enter_at(
                ssa,
                0,
                0,
                Address::new(Root::Stack, 0, PointerSize::Bit64),
                bound,
                PointerSize::Bit64,
            )
            .with_stack_extent(Some(-64..0)),
        );
        let (stop, ran) = Stepper::new(ssa, &context).run(&mut state, trace, usize::MAX);
        (state, stop, ran.instructions)
    }

    /// A byte decoded into the frame reads back decoded, and the write lands
    /// in the loop the path is in.
    #[test]
    fn a_decoded_byte_reads_back() {
        let (mut builder, stack, _) = with_inputs();
        let loaded = builder
            .in_block(0, |b| {
                let buffer = b.emit_def(SsaDefSpec::tmp(MockType::Ptr), |dest| SsaOp::PtrAdd {
                    dest,
                    base: stack,
                    index: None,
                    stride: 1,
                    offset: -8,
                    result_type: MockType::Ptr,
                })?;
                let encoded = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x3b))?;
                let key = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x5a))?;
                let plain = b.xor(SsaDefSpec::tmp(MockType::U8), encoded, key)?;
                b.store_indirect(buffer, plain, MockType::U8)?;
                let loaded =
                    b.load_indirect(SsaDefSpec::tmp(MockType::U8), buffer, MockType::U8)?;
                b.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        let (state, stop, ran) = run(&ssa, vec![], &mut trace);
        assert_eq!(stop, Stop::Terminator);
        assert_eq!(ran, 6);
        assert_eq!(state.value(loaded), Value::Const(ConstValue::U8(0x61)));
        let visit = trace.loops().first().unwrap_or_else(|| abort(()));
        assert!(visit.clean);
        assert!(
            visit
                .written
                .contains(&Address::new(Root::Stack, -8, PointerSize::Bit64))
        );
    }

    /// Both results of a double-width product of constants are held.
    #[test]
    fn both_wide_results_are_held() {
        let (mut builder, _, _) = with_inputs();
        let (low, high) = builder
            .in_block(0, |b| {
                let left =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(u64::MAX))?;
                let right = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(3))?;
                let halves = b.wide_mul(
                    SsaDefSpec::tmp(MockType::U64),
                    SsaDefSpec::tmp(MockType::U64),
                    left,
                    right,
                    true,
                )?;
                b.ret(Some(halves.1))?;
                Ok(halves)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let mut trace = Trace::new();
        let (state, stop, _) = run(&ssa, vec![], &mut trace);
        assert_eq!(stop, Stop::Terminator);
        assert_eq!(
            state.value(low),
            Value::Const(ConstValue::U64(u64::MAX - 2))
        );
        assert_eq!(state.value(high), Value::Const(ConstValue::U64(2)));
    }

    /// The stepper leaves a call to the explorer, standing on it.
    #[test]
    fn a_call_stops_the_stepper() {
        let (mut builder, _, _) = with_inputs();
        builder
            .in_block(0, |b| {
                let one = b.const_i32(SsaDefSpec::tmp(MockType::I32), 1)?;
                b.call_void(Callee::Direct(7), vec![one])?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let (state, stop, ran) = run(&ssa, vec![], &mut Trace::new());
        assert_eq!(stop, Stop::Call(1));
        assert_eq!(ran, 1);
        assert_eq!(state.frame().map(|frame| frame.position().next), Some(1));
    }

    /// A store the execution cannot place forgets what it may have reached
    /// and spoils the loops the path is in.
    #[test]
    fn an_unplaced_store_spoils_the_loop() {
        let (mut builder, _, parameter) = with_inputs();
        builder
            .in_block(0, |b| {
                let zero = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                b.store_indirect(parameter, zero, MockType::U8)?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        run(&ssa, vec![(parameter, Value::Unknown)], &mut trace);
        assert!(trace.loops().iter().all(|visit| !visit.clean));
    }

    /// A store through a named but unknown pointer is placed: under that
    /// pointer's own root.
    #[test]
    fn a_store_through_a_parameter_is_placed() {
        let (mut builder, _, parameter) = with_inputs();
        builder
            .in_block(0, |b| {
                let zero = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?;
                b.store_indirect(parameter, zero, MockType::U8)?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        let (state, _, _) = run(&ssa, vec![], &mut trace);
        let visit = trace.loops().first().unwrap_or_else(|| abort(()));
        assert!(
            !visit.clean,
            "a named pointer does not prove writable backing"
        );
        let address = state
            .value(parameter)
            .address()
            .unwrap_or_else(|| abort(()));
        assert!(matches!(address.root(), Root::Opaque { .. }));
        assert_eq!(state.memory().known_run(address, 1, None), [0]);
    }

    /// Unmodelled memory effects can follow or publish any stack-derived operand.
    #[test]
    fn unknown_memory_effects_cannot_keep_stack_operands_private() {
        let (mut builder, _, parameter) = with_inputs();
        builder.in_block(0, |b| b.ret(None)).unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE);
        let buffer = Address::new(Root::Stack, -8, PointerSize::Bit64);
        for op in [
            SsaOp::StoreField {
                object: parameter,
                field: 0,
                value: parameter,
                access: crate::ir::ops::AccessQualifiers::FLAT,
            },
            SsaOp::StoreStaticField {
                field: 0,
                value: parameter,
                access: crate::ir::ops::AccessQualifiers::FLAT,
            },
        ] {
            let mut state = State::new(Frame::enter_at(
                &ssa,
                0,
                0,
                Address::new(Root::Stack, 0, PointerSize::Bit64),
                vec![(parameter, Value::Addr(buffer))],
                PointerSize::Bit64,
            ));
            state.memory_mut().store_bytes(buffer, b"private");
            let mut trace = Trace::new();
            trace.enter_loop(0, 0);
            assert_eq!(
                Stepper::new(&ssa, &context)
                    .execute(&op, &mut state, &mut trace)
                    .map(|_| ()),
                Ok(())
            );
            assert!(state.memory().stack_escaped());
            assert!(state.memory().known_run(buffer, 7, None).is_empty());
            assert!(trace.loops().iter().all(|visit| !visit.clean));
        }
    }

    /// Builds a zero-length fill whose address can be lost by a state join.
    fn zero_fill_fixture(reverse: bool) -> (SsaFunction<MockTarget>, SsaVarId, SsaVarId) {
        let (mut builder, _, parameter) = with_inputs();
        let loaded = builder
            .in_block(0, |block| {
                let guard =
                    block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                let byte =
                    block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x96))?;
                block.store_indirect(guard, byte, MockType::U8)?;
                let size = block.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                let backward = if reverse {
                    Some(block.const_value(SsaDefSpec::tmp(MockType::Bool), ConstValue::True)?)
                } else {
                    None
                };
                block.init_blk(parameter, byte, size, backward)?;
                let loaded =
                    block.load_indirect(SsaDefSpec::tmp(MockType::U8), guard, MockType::U8)?;
                block.ret(Some(loaded))?;
                Ok(loaded)
            })
            .unwrap_or_else(abort);
        (builder.finish().unwrap_or_else(abort), parameter, loaded)
    }

    /// A zero-length write through a lost address preserves independently stored bytes.
    #[test]
    fn streamed_block_zero_fill_preserves_unplaced_memory() {
        for reverse in [false, true] {
            for input in [Value::Unknown, Value::StackDerived]
                .into_iter()
                .chain(partial_stack_values())
            {
                let (ssa, parameter, loaded) = zero_fill_fixture(reverse);
                let (state, stop, _) =
                    run(&ssa, vec![(parameter, input.clone())], &mut Trace::new());
                assert_eq!(stop, Stop::Terminator);
                assert_eq!(state.value(parameter), input);
                assert_eq!(
                    state.value(loaded),
                    Value::Const(ConstValue::U8(0x96)),
                    "reverse={reverse}"
                );
            }
        }
    }

    /// A zero-length fill records no unplaced write in the loop's effect proof.
    #[test]
    fn streamed_block_zero_fill_records_no_unplaced_effect() {
        for reverse in [false, true] {
            let (ssa, parameter, _) = zero_fill_fixture(reverse);
            let mut trace = Trace::new();
            trace.enter_loop(0, 0);
            let (_, stop, _) = run(&ssa, vec![(parameter, Value::Unknown)], &mut trace);
            assert_eq!(stop, Stop::Terminator);
            let visit = trace.loops().first().unwrap_or_else(|| abort(()));
            assert!(visit.clean, "reverse={reverse}");
            assert_eq!(visit.written.len(), 1);
        }
    }

    /// A zero-length fill cannot publish a stack-derived address.
    #[test]
    fn streamed_block_zero_fill_keeps_stack_private() {
        for reverse in [false, true] {
            let (ssa, parameter, _) = zero_fill_fixture(reverse);
            let (state, stop, _) = run(
                &ssa,
                vec![(parameter, Value::StackDerived)],
                &mut Trace::new(),
            );
            assert_eq!(stop, Stop::Terminator);
            assert!(
                state
                    .memory()
                    .holds_privately(Address::new(Root::Stack, -8, PointerSize::Bit64)),
                "reverse={reverse}"
            );
        }
    }

    /// Checks an empty transfer after either address was lost by a state join.
    fn zero_transfer_preserves_state(buffered: bool, reverse: bool) {
        for lost_source in [false, true] {
            for input in [Value::Unknown, Value::StackDerived]
                .into_iter()
                .chain(partial_stack_values())
            {
                for width in [1, 2, 16, 256] {
                    let (mut builder, _, parameter) = with_inputs();
                    let loaded = builder
                        .in_block(0, |block| {
                            let guard = block.const_value(
                                SsaDefSpec::tmp(MockType::U64),
                                ConstValue::U64(0x1000),
                            )?;
                            let byte = block
                                .const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x96))?;
                            block.store_indirect(guard, byte, MockType::U8)?;
                            let size = block
                                .const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0))?;
                            let (dest, source) = if lost_source {
                                (guard, parameter)
                            } else {
                                (parameter, guard)
                            };
                            if buffered {
                                block.move_blk(dest, source, size)?;
                            } else {
                                let backward = if reverse {
                                    Some(block.const_value(
                                        SsaDefSpec::tmp(MockType::Bool),
                                        ConstValue::True,
                                    )?)
                                } else {
                                    None
                                };
                                block.copy_blk_elements(dest, source, size, width, backward)?;
                            }
                            let loaded = block.load_indirect(
                                SsaDefSpec::tmp(MockType::U8),
                                guard,
                                MockType::U8,
                            )?;
                            block.ret(Some(loaded))?;
                            Ok(loaded)
                        })
                        .unwrap_or_else(abort);
                    let ssa = builder.finish().unwrap_or_else(abort);
                    let mut trace = Trace::new();
                    trace.enter_loop(0, 0);
                    let (state, stop, _) = run(&ssa, vec![(parameter, input.clone())], &mut trace);
                    assert_eq!(stop, Stop::Terminator);
                    assert_eq!(state.value(parameter), input);
                    assert_eq!(
                        state.value(loaded),
                        Value::Const(ConstValue::U8(0x96)),
                        "source={lost_source} width={width}"
                    );
                    assert!(state.memory().holds_privately(Address::new(
                        Root::Stack,
                        -8,
                        PointerSize::Bit64
                    )));
                    let visit = trace.loops().first().unwrap_or_else(|| abort(()));
                    assert!(visit.clean);
                    assert_eq!(visit.written.len(), 1);
                }
            }
        }
    }

    /// An empty forward stream touches neither a lost source nor a lost destination.
    #[test]
    fn streamed_block_zero_copy_forward_preserves_state() {
        zero_transfer_preserves_state(false, false);
    }

    /// An empty reverse stream needs no address for its first element.
    #[test]
    fn streamed_block_zero_copy_reverse_preserves_state() {
        zero_transfer_preserves_state(false, true);
    }

    /// An empty snapshot move retains independent memory and effect proofs.
    #[test]
    fn streamed_block_zero_move_preserves_state() {
        zero_transfer_preserves_state(true, false);
    }

    /// A fill whose direction flag is not known is neither the upward nor the
    /// downward fill: what it writes is not modelled, as any unmodelled write.
    #[test]
    fn a_fill_of_unknown_direction_is_not_modelled_as_either() {
        let (mut builder, _, flag) = with_inputs();
        builder
            .in_block(0, |b| {
                let dest =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1020))?;
                let value = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0xef))?;
                let size = b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(4))?;
                b.init_blk(dest, value, size, Some(flag))?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE);
        let mut state = State::new(Frame::enter_at(
            &ssa,
            0,
            0,
            Address::new(Root::Stack, 0, PointerSize::Bit64),
            vec![],
            PointerSize::Bit64,
        ));
        let address = Address::new(Root::Number, 0x1000, PointerSize::Bit64);
        state.memory_mut().store_bytes(address, &[0xcc; 96]);
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        let _ = Stepper::new(&ssa, &context).run(&mut state, &mut trace, usize::MAX);
        let held = state.memory().known_run(address, 96, None);
        for start in [0x20usize, 0x1d] {
            let mut filled = vec![0xcc; 96];
            filled[start..start + 4].fill(0xef);
            assert_ne!(held, filled, "modelled as the fill from {start:#x}");
        }
    }

    /// Forward and reverse fills agree on their exact touched bytes and retain their sentinels.
    #[test]
    fn streamed_block_fill_honors_direction_and_element_width() {
        for reverse in [false, true] {
            for (ty, constant, width) in [
                (MockType::U8, ConstValue::U8(0xef), 1usize),
                (MockType::U16, ConstValue::U16(0xcdef), 2),
                (MockType::U32, ConstValue::U32(0x89ab_cdef), 4),
                (MockType::U64, ConstValue::U64(0x0123_4567_89ab_cdef), 8),
            ] {
                let len = width * 3;
                let start = 0x1020usize;
                let first = start + if reverse { len - width } else { 0 };
                let (mut builder, _, _) = with_inputs();
                builder
                    .in_block(0, |b| {
                        let dest = b.const_value(
                            SsaDefSpec::tmp(MockType::U64),
                            ConstValue::U64(first as u64),
                        )?;
                        let value = b.const_value(SsaDefSpec::tmp(ty), constant.clone())?;
                        let size = b.const_value(
                            SsaDefSpec::tmp(MockType::U64),
                            ConstValue::U64(len as u64),
                        )?;
                        let backward = if reverse {
                            Some(b.const_value(SsaDefSpec::tmp(MockType::Bool), ConstValue::True)?)
                        } else {
                            None
                        };
                        b.init_blk(dest, value, size, backward)?;
                        b.ret(None)
                    })
                    .unwrap_or_else(abort);
                let ssa = builder.finish().unwrap_or_else(abort);
                let context = AnalysisContext::new(crate::analysis::Machine {
                    byte_order: Some(crate::target::Endianness::Little),
                    ..MACHINE
                });
                let mut state = State::new(Frame::enter_at(
                    &ssa,
                    0,
                    0,
                    Address::new(Root::Stack, 0, PointerSize::Bit64),
                    vec![],
                    PointerSize::Bit64,
                ));
                let address = Address::new(Root::Number, 0x1000, PointerSize::Bit64);
                state.memory_mut().store_bytes(address, &[0xcc; 96]);
                let mut trace = Trace::new();
                trace.enter_loop(0, 0);
                assert_eq!(
                    Stepper::new(&ssa, &context)
                        .run(&mut state, &mut trace, usize::MAX)
                        .0,
                    Stop::Terminator
                );
                let mut expected = vec![0xcc; 96];
                for index in 0..len {
                    expected[start - 0x1000 + index] =
                        (0x0123_4567_89ab_cdefu64 >> ((index % width) * 8)) as u8;
                }
                assert_eq!(
                    state.memory().known_run(address, 96, None),
                    expected,
                    "reverse={reverse}, width={width}"
                );
            }
        }
    }

    /// What a repeated compare, scan or load leaves, run over bytes `i` at
    /// `0x1000 + i` with `0x1082..0x1088` set to `2..8`: the defined values in
    /// order, each `None` where it is no constant.
    fn walked_block(
        build: impl FnOnce(
            &mut SsaBlockBuilder<'_, MockTarget>,
            SsaVarId,
        ) -> crate::Result<Vec<SsaVarId>>,
    ) -> Vec<Option<u64>> {
        walked(build).0
    }

    /// Runs [`walked_block`]'s block, returning what it defined and what the
    /// run cost.
    fn walked(
        build: impl FnOnce(
            &mut SsaBlockBuilder<'_, MockTarget>,
            SsaVarId,
        ) -> crate::Result<Vec<SsaVarId>>,
    ) -> (Vec<Option<u64>>, Ran) {
        let (mut builder, _, _) = with_inputs();
        let mut defined = Vec::new();
        builder
            .in_block(0, |b| {
                let flags_in =
                    b.const_value(SsaDefSpec::tmp(MockType::U32), ConstValue::U32(0x55))?;
                defined = build(b, flags_in)?;
                b.ret(None)
            })
            .unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(crate::analysis::Machine {
            byte_order: Some(crate::target::Endianness::Little),
            ..MACHINE
        });
        let mut state = State::new(Frame::enter_at(
            &ssa,
            0,
            0,
            Address::new(Root::Stack, 0, PointerSize::Bit64),
            vec![],
            PointerSize::Bit64,
        ));
        let mut bytes: Vec<u8> = (0..=255).collect();
        bytes[0x82..0x88].copy_from_slice(&[2, 3, 4, 5, 6, 7]);
        state.memory_mut().store_bytes(
            Address::new(Root::Number, 0x1000, PointerSize::Bit64),
            &bytes,
        );
        let mut trace = Trace::new();
        let (stop, ran) = Stepper::new(&ssa, &context).run(&mut state, &mut trace, usize::MAX);
        assert_eq!(stop, Stop::Terminator);
        let held = defined
            .into_iter()
            .map(|var| {
                state
                    .value(var)
                    .constant()
                    .and_then(ConstValue::as_u64_bits)
            })
            .collect();
        (held, ran)
    }

    /// **A scan stops where its element lies, whatever its count.** x86's
    /// inline `strlen` scans with a count of all ones (`or ecx, -1; repne
    /// scasb`) and stops at the terminator: the walk is bounded by the
    /// elements it visits, so at either count width it runs to the byte it
    /// looks for, leaving the count less the bytes visited, and costs those
    /// bytes. One that would visit past the block limit stays unknown
    /// ([`a_scan_past_the_block_limit_is_unknown`]).
    #[test]
    fn a_scan_stops_where_its_element_lies_whatever_its_count() {
        for (count, remaining) in [
            (ConstValue::U64(u64::MAX), u64::MAX.saturating_sub(0x11)),
            (
                ConstValue::U32(u32::MAX),
                u64::from(u32::MAX).saturating_sub(0x11),
            ),
        ] {
            let ty = if matches!(count, ConstValue::U64(_)) {
                MockType::U64
            } else {
                MockType::U32
            };
            let (held, ran) = walked(|b, flags_in| {
                let address =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                let value = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x10))?;
                let count = b.const_value(SsaDefSpec::tmp(ty), count.clone())?;
                b.emit_many_defs(
                    &[
                        SsaDefSpec::tmp(ty),
                        SsaDefSpec::tmp(MockType::U64),
                        SsaDefSpec::tmp(MockType::U32),
                    ],
                    |ids| SsaOp::ScanBlk {
                        remaining: ids[0],
                        next: ids[1],
                        flags: ids[2],
                        address,
                        value,
                        count,
                        flags_in: Some(flags_in),
                        backward: None,
                        element_bytes: 1,
                        while_equal: false,
                        access: AccessQualifiers::FLAT,
                    },
                )
                .map(|ids| vec![ids[0]])
            });
            assert_eq!(held, vec![Some(remaining)], "{count:?}");
            assert_eq!(
                ran.steps.saturating_sub(ran.instructions),
                0x11,
                "{count:?}"
            );
        }
    }

    /// Every block instruction costs a step per byte it spans beyond its own,
    /// as the loop it stands for would: a fill, copy or move its size, a scan,
    /// compare or repeated load the elements it visited. One that does not
    /// run, and a scalar load, cost their one step.
    #[test]
    fn block_instructions_cost_a_step_per_byte_they_span() {
        let constant = |b: &mut SsaBlockBuilder<'_, MockTarget>, value: u64| {
            b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(value))
        };
        let fill = walked(|b, _| {
            let dest = constant(b, 0x1100)?;
            let value = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0x11))?;
            let size = constant(b, 32)?;
            b.emit_no_defs(SsaOp::InitBlk {
                dest_addr: dest,
                value,
                size,
                backward: None,
                access: AccessQualifiers::FLAT,
            })?;
            let check = constant(b, 0x111f)?;
            Ok(vec![b.load_indirect(
                SsaDefSpec::tmp(MockType::U8),
                check,
                MockType::U8,
            )?])
        });
        let copy = walked(|b, _| {
            let dest = constant(b, 0x1100)?;
            let source = constant(b, 0x1000)?;
            let size = constant(b, 32)?;
            b.emit_no_defs(SsaOp::CopyBlk {
                dest_addr: dest,
                src_addr: source,
                size,
                element_bytes: 4,
                backward: None,
                src_access: AccessQualifiers::FLAT,
                dest_access: AccessQualifiers::FLAT,
            })?;
            let check = constant(b, 0x111f)?;
            Ok(vec![b.load_indirect(
                SsaDefSpec::tmp(MockType::U8),
                check,
                MockType::U8,
            )?])
        });
        let moved = walked(|b, _| {
            let dest = constant(b, 0x1010)?;
            let source = constant(b, 0x1000)?;
            let size = constant(b, 32)?;
            b.emit_no_defs(SsaOp::MoveBlk {
                dest_addr: dest,
                src_addr: source,
                size,
                src_access: AccessQualifiers::FLAT,
                dest_access: AccessQualifiers::FLAT,
            })?;
            let check = constant(b, 0x102f)?;
            Ok(vec![b.load_indirect(
                SsaDefSpec::tmp(MockType::U8),
                check,
                MockType::U8,
            )?])
        });
        // `repne scasb` for the byte 5 stops on the sixth byte.
        let scan = walked(|b, flags_in| {
            let address = constant(b, 0x1000)?;
            let value = b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(5))?;
            let count = constant(b, 200)?;
            b.emit_many_defs(
                &[
                    SsaDefSpec::tmp(MockType::U64),
                    SsaDefSpec::tmp(MockType::U64),
                    SsaDefSpec::tmp(MockType::U32),
                ],
                |ids| SsaOp::ScanBlk {
                    remaining: ids[0],
                    next: ids[1],
                    flags: ids[2],
                    address,
                    value,
                    count,
                    flags_in: Some(flags_in),
                    backward: None,
                    element_bytes: 1,
                    while_equal: false,
                    access: AccessQualifiers::FLAT,
                },
            )
            .map(|ids| vec![ids[0]])
        });
        let compare = walked(|b, flags_in| {
            let address = constant(b, 0x1000)?;
            let count = constant(b, 16)?;
            b.emit_many_defs(
                &[
                    SsaDefSpec::tmp(MockType::U64),
                    SsaDefSpec::tmp(MockType::U64),
                    SsaDefSpec::tmp(MockType::U64),
                    SsaDefSpec::tmp(MockType::U32),
                ],
                |ids| {
                    SsaOp::CompareBlk(Box::new(CompareBlkData {
                        remaining: ids[0],
                        left_next: ids[1],
                        right_next: ids[2],
                        flags: ids[3],
                        left: address,
                        right: address,
                        count,
                        flags_in: Some(flags_in),
                        backward: None,
                        element_bytes: 2,
                        while_equal: true,
                        left_access: AccessQualifiers::FLAT,
                        right_access: AccessQualifiers::FLAT,
                    }))
                },
            )
            .map(|ids| vec![ids[0]])
        });
        let load = walked(|b, value_in| {
            let address = constant(b, 0x1000)?;
            let count = constant(b, 8)?;
            b.emit_many_defs(
                &[
                    SsaDefSpec::tmp(MockType::U32),
                    SsaDefSpec::tmp(MockType::U64),
                ],
                |ids| SsaOp::LoadBlk {
                    value: ids[0],
                    next: ids[1],
                    address,
                    count,
                    value_in: Some(value_in),
                    backward: None,
                    element_bytes: 4,
                    access: AccessQualifiers::FLAT,
                },
            )
            .map(|ids| vec![ids[0]])
        });
        // A load whose block wraps the address space does not run.
        let unrun = walked(|b, value_in| {
            let address = constant(b, 0x1000)?;
            let count = constant(b, u64::MAX)?;
            b.emit_many_defs(
                &[
                    SsaDefSpec::tmp(MockType::U32),
                    SsaDefSpec::tmp(MockType::U64),
                ],
                |ids| SsaOp::LoadBlk {
                    value: ids[0],
                    next: ids[1],
                    address,
                    count,
                    value_in: Some(value_in),
                    backward: None,
                    element_bytes: 4,
                    access: AccessQualifiers::FLAT,
                },
            )
            .map(|ids| vec![ids[0]])
        });
        assert_eq!(unrun.0, vec![None]);
        assert_eq!(unrun.1.steps, unrun.1.instructions);
        let scalar = walked(|b, _| {
            let address = constant(b, 0x1000)?;
            Ok(vec![b.load_indirect(
                SsaDefSpec::tmp(MockType::U64),
                address,
                MockType::U64,
            )?])
        });
        for (name, (held, ran), value, spanned) in [
            ("fill", fill, 0x11, 32),
            ("copy", copy, 0x1f, 32),
            ("move", moved, 0x1f, 32),
            ("scan", scan, 194, 6),
            ("compare", compare, 0, 32),
            ("load", load, 0x1f1e_1d1c, 32),
            ("scalar", scalar, 0x0706_0504_0302_0100, 0),
        ] {
            assert_eq!(held, vec![Some(value)], "{name}");
            assert_eq!(
                ran.steps.saturating_sub(ran.instructions),
                spanned,
                "{name}"
            );
        }
    }

    /// A malformed zero-width scan or compare establishes no result, even for an empty count.
    #[test]
    fn repeated_zero_width_reads_remain_opaque() {
        for compare in [false, true] {
            for count in [0, 1] {
                for down in [false, true] {
                    let held = walked_block(|b, flags_in| {
                        let address =
                            b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                        let right = if compare {
                            b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1080))?
                        } else {
                            b.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(0))?
                        };
                        let count =
                            b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(count))?;
                        let backward = b.const_value(
                            SsaDefSpec::tmp(MockType::Bool),
                            ConstValue::from_bool(down),
                        )?;
                        let mut defs = vec![
                            SsaDefSpec::tmp(MockType::U64),
                            SsaDefSpec::tmp(MockType::U64),
                        ];
                        if compare {
                            defs.push(SsaDefSpec::tmp(MockType::U64));
                        }
                        defs.push(SsaDefSpec::tmp(MockType::U32));
                        b.emit_many_defs(&defs, |ids| {
                            if compare {
                                SsaOp::CompareBlk(Box::new(CompareBlkData {
                                    remaining: ids[0],
                                    left_next: ids[1],
                                    right_next: ids[2],
                                    flags: ids[3],
                                    left: address,
                                    right,
                                    count,
                                    flags_in: Some(flags_in),
                                    backward: Some(backward),
                                    element_bytes: 0,
                                    while_equal: false,
                                    left_access: AccessQualifiers::FLAT,
                                    right_access: AccessQualifiers::FLAT,
                                }))
                            } else {
                                SsaOp::ScanBlk {
                                    remaining: ids[0],
                                    next: ids[1],
                                    flags: ids[2],
                                    address,
                                    value: right,
                                    count,
                                    flags_in: Some(flags_in),
                                    backward: Some(backward),
                                    element_bytes: 0,
                                    while_equal: false,
                                    access: AccessQualifiers::FLAT,
                                }
                            }
                        })
                    });
                    assert_eq!(held, vec![None; if compare { 4 } else { 3 }]);
                }
            }
        }
    }

    /// A malformed zero-width load does not establish values or a no-access proof.
    #[test]
    fn repeated_zero_width_loads_remain_opaque() {
        for count in [0, 1] {
            let held = walked_block(|b, value_in| {
                let address =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(0x1000))?;
                let count =
                    b.const_value(SsaDefSpec::tmp(MockType::U64), ConstValue::U64(count))?;
                b.emit_many_defs(
                    &[
                        SsaDefSpec::tmp(MockType::U32),
                        SsaDefSpec::tmp(MockType::U64),
                    ],
                    |defs| SsaOp::LoadBlk {
                        value: defs[0],
                        next: defs[1],
                        address,
                        count,
                        value_in: Some(value_in),
                        backward: None,
                        element_bytes: 0,
                        access: AccessQualifiers::FLAT,
                    },
                )
            });
            assert_eq!(held, vec![None, None]);
        }
    }

    /// A repeated scan, compare or load walks up or down as its direction
    /// flag says and stops where the processor does: `repne` after the first
    /// equal element, `repe` after the first unequal pair, a count run out
    /// with unknown flags, an empty one with the flags it was given; a load
    /// leaves its last element. An unknown element leaves every result
    /// unknown.
    #[test]
    fn repeated_compares_scans_and_loads_walk_either_direction() {
        let constant = |b: &mut SsaBlockBuilder<'_, MockTarget>, ty, value| {
            b.const_value(SsaDefSpec::tmp(ty), value)
        };
        let direction = |b: &mut SsaBlockBuilder<'_, MockTarget>, down: bool| {
            b.const_value(SsaDefSpec::tmp(MockType::Bool), ConstValue::from_bool(down))
        };
        let scan_defs = [
            SsaDefSpec::tmp(MockType::U64),
            SsaDefSpec::tmp(MockType::U64),
            SsaDefSpec::tmp(MockType::U32),
        ];
        // repne scas: (start, value, value type, count, width, down) and
        // (remaining, next, flags).
        for ((start, value, count, width, down), expected) in [
            ((0x100a, 20, 50, 1, false), [Some(39), Some(0x1015), None]),
            ((0x101e, 20, 50, 1, true), [Some(39), Some(0x1013), None]),
            (
                (0x1010, 0x1716, 50, 2, false),
                [Some(46), Some(0x1018), None],
            ),
            (
                (0x101c, 0x1716, 50, 2, true),
                [Some(46), Some(0x1014), None],
            ),
            ((0x1000, 0xff, 5, 1, false), [Some(0), Some(0x1005), None]),
            (
                (0x1000, 0, 0, 1, false),
                [Some(0), Some(0x1000), Some(0x55)],
            ),
            ((0x3000, 0, 4, 1, false), [None, None, None]),
        ] {
            let held = walked_block(|b, flags_in| {
                let address = constant(b, MockType::U64, ConstValue::U64(start))?;
                let value = if width == 1 {
                    constant(b, MockType::U8, ConstValue::U8(value as u8))?
                } else {
                    constant(b, MockType::U16, ConstValue::U16(value as u16))?
                };
                let count = constant(b, MockType::U64, ConstValue::U64(count))?;
                let backward = direction(b, down)?;
                b.emit_many_defs(&scan_defs, |defs| SsaOp::ScanBlk {
                    remaining: defs[0],
                    next: defs[1],
                    flags: defs[2],
                    address,
                    value,
                    count,
                    flags_in: Some(flags_in),
                    backward: Some(backward),
                    element_bytes: width,
                    while_equal: false,
                    access: AccessQualifiers::FLAT,
                })
            });
            assert_eq!(
                held, expected,
                "scan from {start:#x} for {value:#x}, down={down}"
            );
        }
        // repe cmps from (left, down): 7 pairs, the last unequal, leaving
        // 3 of 10 and each pointer past the last pair.
        for ((left, down), expected) in [
            ((0x1002, false), [Some(3), Some(0x1009), Some(0x1089), None]),
            ((0x1007, true), [Some(3), Some(0x1000), Some(0x1080), None]),
        ] {
            let held = walked_block(|b, flags_in| {
                let first = constant(b, MockType::U64, ConstValue::U64(left))?;
                let second = constant(b, MockType::U64, ConstValue::U64(left + 0x80))?;
                let count = constant(b, MockType::U64, ConstValue::U64(10))?;
                let backward = direction(b, down)?;
                b.emit_many_defs(
                    &[
                        SsaDefSpec::tmp(MockType::U64),
                        SsaDefSpec::tmp(MockType::U64),
                        SsaDefSpec::tmp(MockType::U64),
                        SsaDefSpec::tmp(MockType::U32),
                    ],
                    |defs| {
                        SsaOp::CompareBlk(Box::new(CompareBlkData {
                            remaining: defs[0],
                            left_next: defs[1],
                            right_next: defs[2],
                            flags: defs[3],
                            left: first,
                            right: second,
                            count,
                            flags_in: Some(flags_in),
                            backward: Some(backward),
                            element_bytes: 1,
                            while_equal: true,
                            left_access: AccessQualifiers::FLAT,
                            right_access: AccessQualifiers::FLAT,
                        }))
                    },
                )
            });
            assert_eq!(held, expected, "compare from {left:#x}, down={down}");
        }
        // rep lods from (start, count, width, down): (value, next).
        for ((start, count, width, down), expected) in [
            ((0x1005, 4, 1, false), [Some(8), Some(0x1009)]),
            ((0x1005, 4, 1, true), [Some(2), Some(0x1001)]),
            ((0x1004, 3, 2, false), [Some(0x0908), Some(0x100a)]),
            ((0x1005, 0, 1, false), [Some(0x77), Some(0x1005)]),
        ] {
            let held = walked_block(|b, _| {
                let address = constant(b, MockType::U64, ConstValue::U64(start))?;
                let count = constant(b, MockType::U64, ConstValue::U64(count))?;
                let (ty, value_in) = if width == 1 {
                    (
                        MockType::U8,
                        constant(b, MockType::U8, ConstValue::U8(0x77))?,
                    )
                } else {
                    (
                        MockType::U16,
                        constant(b, MockType::U16, ConstValue::U16(0x77))?,
                    )
                };
                let backward = direction(b, down)?;
                b.emit_many_defs(
                    &[SsaDefSpec::tmp(ty), SsaDefSpec::tmp(MockType::U64)],
                    |defs| SsaOp::LoadBlk {
                        value: defs[0],
                        next: defs[1],
                        address,
                        count,
                        value_in: Some(value_in),
                        backward: Some(backward),
                        element_bytes: width,
                        access: AccessQualifiers::FLAT,
                    },
                )
            });
            assert_eq!(held, expected, "load from {start:#x}, down={down}");
        }
    }

    /// Each whole element is read before its writes, in either direction and with arbitrary overlap.
    #[test]
    fn streamed_block_copy_honors_direction_and_overlap() {
        for reverse in [false, true] {
            for width in [1usize, 2, 3, 4, 8, 16] {
                for shift in [-49i64, -17, -3, -1, 0, 1, 3, 17, 49] {
                    let from = 0x1060i64;
                    let to = from + shift;
                    let len = width * 3;
                    let delta = if reverse { len - width } else { 0 };
                    let (mut builder, _, _) = with_inputs();
                    builder
                        .in_block(0, |b| {
                            let dest = b.const_value(
                                SsaDefSpec::tmp(MockType::U64),
                                ConstValue::U64(to as u64 + delta as u64),
                            )?;
                            let source = b.const_value(
                                SsaDefSpec::tmp(MockType::U64),
                                ConstValue::U64(from as u64 + delta as u64),
                            )?;
                            let size = b.const_value(
                                SsaDefSpec::tmp(MockType::U64),
                                ConstValue::U64(len as u64),
                            )?;
                            let backward = if reverse {
                                Some(b.const_value(
                                    SsaDefSpec::tmp(MockType::Bool),
                                    ConstValue::True,
                                )?)
                            } else {
                                None
                            };
                            b.emit_no_defs(SsaOp::CopyBlk {
                                dest_addr: dest,
                                src_addr: source,
                                size,
                                element_bytes: width as u32,
                                backward,
                                src_access: AccessQualifiers::FLAT,
                                dest_access: AccessQualifiers::FLAT,
                            })?;
                            b.ret(None)
                        })
                        .unwrap_or_else(abort);
                    let ssa = builder.finish().unwrap_or_else(abort);
                    let context = AnalysisContext::new(crate::analysis::Machine {
                        byte_order: Some(crate::target::Endianness::Little),
                        ..MACHINE
                    });
                    let mut state = State::new(Frame::enter_at(
                        &ssa,
                        0,
                        0,
                        Address::new(Root::Stack, 0, PointerSize::Bit64),
                        vec![],
                        PointerSize::Bit64,
                    ));
                    let address = Address::new(Root::Number, 0x1000, PointerSize::Bit64);
                    let mut expected: Vec<u8> = (0..=255).collect();
                    state.memory_mut().store_bytes(address, &expected);
                    let mut trace = Trace::new();
                    assert_eq!(
                        Stepper::new(&ssa, &context)
                            .run(&mut state, &mut trace, usize::MAX)
                            .0,
                        Stop::Terminator
                    );
                    for step in 0..3 {
                        let index = if reverse { 2 - step } else { step } * width;
                        let source = (from - 0x1000) as usize + index;
                        let dest = (to - 0x1000) as usize + index;
                        let element = expected[source..source + width].to_vec();
                        expected[dest..dest + width].copy_from_slice(&element);
                    }
                    assert_eq!(
                        state.memory().known_run(address, 256, None),
                        expected,
                        "reverse={reverse}, width={width}, shift={shift}"
                    );
                }
            }
        }
    }

    /// Machine state an operation keeps can capture an address it reads: a
    /// hint (no memory) keeps the bytes behind a stack-derived input, a
    /// system register exchange (which may write memory) does not, and each
    /// publishes the address.
    #[test]
    fn memory_contract_non_memory_state_keeps_bytes_and_tracks_escape() {
        for (kind, keeps) in [
            (SystemOpKind::Hint(HintOp::GenericHint), true),
            (
                SystemOpKind::ControlRegister(SysRegOp::ExchangeSystemRegister),
                false,
            ),
        ] {
            let (mut builder, _, parameter) = with_inputs();
            builder
                .in_block(0, |block| {
                    block.emit_no_defs(SsaOp::SystemOp(Box::new(NativeKindedData {
                        kind,
                        mnemonic: "state-observer".into(),
                        metadata: None,
                        inputs: vec![parameter],
                        outputs: vec![],
                    })))?;
                    block.ret(None)
                })
                .unwrap_or_else(abort);
            let ssa = builder.finish().unwrap_or_else(abort);
            let context = AnalysisContext::new(MACHINE);
            let pointer = Address::new(Root::Stack, -8, PointerSize::Bit64);
            let mut state = State::new(Frame::enter_at(
                &ssa,
                0,
                0,
                Address::new(Root::Stack, 0, PointerSize::Bit64),
                vec![(parameter, Value::Addr(pointer))],
                PointerSize::Bit64,
            ));
            state.memory_mut().store_bytes(pointer, &[7, 8, 9]);
            let mut trace = Trace::new();
            trace.enter_loop(0, 0);
            assert_eq!(
                Stepper::new(&ssa, &context)
                    .run(&mut state, &mut trace, usize::MAX)
                    .0,
                Stop::Terminator
            );
            assert!(
                !state.memory().holds_privately(pointer),
                "opaque state can capture an address"
            );
            assert!(!trace.loops().first().unwrap_or_else(|| abort(())).clean);
            let actual = state.memory().known_run(pointer, 3, None);
            assert_eq!(actual, if keeps { vec![7, 8, 9] } else { vec![] });
        }
    }

    /// A private atomic keeps exact pointer contents while spoiling loop deletion proofs.
    #[test]
    fn atomic_private_store_retains_value_and_ordering_effect() {
        let (mut builder, _, parameter) = with_inputs();
        builder.in_block(0, |b| b.ret(None)).unwrap_or_else(abort);
        let ssa = builder.finish().unwrap_or_else(abort);
        let context = AnalysisContext::new(MACHINE);
        let buffer = Address::new(Root::Stack, -16, PointerSize::Bit64);
        let mut state = State::new(Frame::enter_at(
            &ssa,
            0,
            0,
            Address::new(Root::Stack, 0, PointerSize::Bit64),
            vec![(parameter, Value::Addr(buffer))],
            PointerSize::Bit64,
        ));
        let op = SsaOp::AtomicStore {
            addr: parameter,
            value: parameter,
            value_type: MockType::Ptr,
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Pointer,
            access: AccessQualifiers::default(),
        };
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        assert_eq!(
            Stepper::new(&ssa, &context)
                .execute(&op, &mut state, &mut trace)
                .map(|_| ()),
            Ok(())
        );
        assert!(!state.memory().stack_escaped());
        assert_eq!(
            Access::new(&context)
                .load_with_footprint(state.memory(), &Value::Addr(buffer), &MockType::Ptr, None)
                .map(|read| read.value),
            Ok(Value::<MockTarget>::Addr(buffer))
        );
        assert!(trace.loops().iter().all(|visit| !visit.clean));
    }

    /// Admission consumes visit outcomes without erasing earlier faults or other effects.
    #[test]
    fn arithmetic_normality_only_discharges_its_own_trap() {
        let ssa = SsaFunction::<MockTarget>::new(0, 0);
        let context = AnalysisContext::new(MACHINE);
        let stepper = Stepper::new(&ssa, &context);
        let value = SsaVarId::from_index(0);
        let arithmetic = SsaOp::Div {
            dest: value,
            left: value,
            right: value,
            unsigned: true,
            flags: None,
        };
        let mut trace = Trace::new();
        trace.enter_loop(0, 0);
        stepper.record_control_effects(&arithmetic, &mut trace, true);
        assert_eq!(trace.clean_headers().count(), 1);
        stepper.record_control_effects(&arithmetic, &mut trace, false);
        assert_eq!(trace.clean_headers().count(), 0);
        stepper.record_control_effects(&arithmetic, &mut trace, true);
        assert_eq!(trace.clean_headers().count(), 0);

        // These are effect-summary controls, not executions of fabricated operands.
        for effectful in [
            SsaOp::LoadIndirect {
                dest: value,
                addr: value,
                value_type: MockType::U8,
                access: AccessQualifiers::FLAT.made_volatile(),
            },
            SsaOp::AtomicStore {
                addr: value,
                value,
                value_type: MockType::U8,
                ordering: AtomicOrdering::SeqCst,
                width: AtomicAccessWidth::Pointer,
                access: AccessQualifiers::FLAT,
            },
            SsaOp::Unreachable,
        ] {
            let mut trace = Trace::new();
            trace.enter_loop(0, 0);
            stepper.record_control_effects(&effectful, &mut trace, true);
            assert_eq!(trace.clean_headers().count(), 0);
        }
    }
}
