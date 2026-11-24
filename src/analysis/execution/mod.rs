//! Static execution: running a function on what the program alone fixes.
//!
//! The values the program's input cannot change - constants, read-only image
//! bytes, and everything computed from them - are evaluated ahead of time, and
//! everything else is carried as unknown. What an execution establishes on
//! every path it explores is a fact about the program, and a pass may replace
//! the computation that produced it with its result.
//!
//! # Examples
//!
//! ```rust
//! use analyssa::{
//!     analysis::{context::AnalysisContext, execution::StaticExecution},
//!     testing,
//! };
//!
//! let ssa = testing::const_i32_return(42);
//! let context = AnalysisContext::new(testing::MACHINE);
//! let facts = StaticExecution::new(&ssa, &context).run();
//! assert!(facts.is_complete());
//! ```

mod access;
mod alignment;
mod allocation;
mod arguments;
mod arrivals;
mod atomic;
mod bank;
mod body;
mod branch;
mod continuation;
mod control;
mod datawrites;
mod edges;
mod edgescalar;
mod explore;
mod facts;
mod findings;
mod frame;
mod host;
mod inputs;
mod integer;
mod marks;
mod memory;
mod observation;
mod provenance;
mod refine;
mod register;
mod returning;
mod scalar;
mod shape;
mod state;
mod step;
mod thrown;
mod trace;
mod types;
mod uses;
mod value;
mod vectorreads;
mod vectorwrites;
mod writeguard;

pub use self::{
    alignment::{RealignedExtent, StackCongruence},
    allocation::{
        AllocatedProtection, Allocation, AllocationFlags, Condition, Conditions, Deallocation,
        Delivery, Duplication, FreeResult, Operand, Pages, Reallocation, Test,
    },
    arguments::ArgumentBindings,
    bank::{MachineEffects, MachineOutput, Placement},
    body::{CodeMemory, CodeRefusal},
    continuation::{
        BranchContinuation, BranchSite, ContinuationBinding, EnteredCode, EntrySite, FrameWrite,
    },
    control::{ReturnAddressStorage, ReturnContract},
    datawrites::{ArgumentSet, DataWrites, printf_writes},
    edges::{
        EdgeByteRequest, EdgeBytes, EdgeObservation, EdgeObservationPlan, EdgeRequestLimits,
        EdgeScalar, EdgeScalarRequest, EdgeSite,
    },
    facts::{
        ArgumentBytes, ByteBound, CallFacts, CallObservation, CallTargets, CodeTransfer,
        CodeTransferKind, Completion, ExecutionUsage, ExtentSource, LoopEffect, ObjectKind,
        ObservationSizeRefusal, ReadFootprint, ReplacementProof, StaticFacts, StorageObject,
        UseBound, WrittenRanges,
    },
    findings::{
        Decision, Decisions, Ending, Finding, FindingKind, Findings, FoundObject, ReleaseFault,
        Witness,
    },
    host::{
        CallModel, CallSite, CallStack, ExecutionHost, LibraryModel, ModelControl, ModelValue,
        ObjectExtents, ParameterBindings, ProtectionResult,
    },
    inputs::CallValues,
    integer::PartialInteger,
    marks::{Marks, UnwrittenUseKind},
    memory::{
        Allocator, AllocatorFamily, BlockSize, FreeOutcome, ImageWrite, Lifetime, MemoryRefusal,
        StringScan, Through, Transfer, WriteSite,
    },
    provenance::Pointer,
    returning::ReturnBinding,
    thrown::{Exceptions, Thrown},
    value::{Address, Root, Value},
    writeguard::{
        InstructionPermissions, InstructionSourceId, InstructionSources, MemoryWriteGuard,
        ProtectionEncoding,
    },
};
use crate::{
    analysis::{
        budget::{Budget, Exhaustion},
        context::AnalysisContext,
        execution::{
            access::Access, explore::Explorer, inputs::InputIndex, observation::CallSelection,
            state::State, trace::Trace,
        },
    },
    ir::{CallInputKey, function::SsaFunction, integer::IntegerType, value::ConstValue},
    pointer::PointerSize,
    target::{Endianness, Target, TypeLayout},
};

/// A static execution of one function, ready to run.
pub struct StaticExecution<'a, T: Target> {
    /// Optional host cancellation probe.
    cancelled: Option<&'a dyn Fn() -> bool>,
    /// The function.
    ssa: &'a SsaFunction<T>,
    /// The image, the target, and the budgets.
    context: &'a AnalysisContext<'a>,
    /// What the host says about calls.
    host: &'a dyn ExecutionHost<T>,
    /// The method the function is, when the host names it.
    method: Option<T::MethodRef>,
    /// Actual entered bodies selected for bounded separate observations.
    call_selection: CallSelection<'a>,
    /// Validated source requests, without adding execution trials.
    edge_plan: Option<&'a EdgeObservationPlan<'a, T>>,
    /// Owner-supplied retained edge payload reservation for this trial.
    edge_bytes: usize,
    /// Maximum distinct host-selected code transfers, when requested.
    code_transfer_limit: Option<usize>,
}

impl<'a, T: Target> StaticExecution<'a, T> {
    /// An execution of `ssa` against `context`, whose calls are all opaque.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function.
    /// * `context` - The pointer width, the image and the budgets.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>, context: &'a AnalysisContext<'a>) -> Self {
        Self {
            ssa,
            context,
            host: &(),
            method: None,
            cancelled: None,
            call_selection: CallSelection::Sites(&[]),
            edge_plan: None,
            edge_bytes: 0,
            code_transfer_limit: None,
        }
    }

    /// Consults `cancelled` during execution and withholds facts when it fires.
    #[must_use]
    pub fn with_cancellation(mut self, cancelled: &'a dyn Fn() -> bool) -> Self {
        self.cancelled = Some(cancelled);
        self
    }

    /// Returns this execution answering calls through `host`.
    #[must_use]
    pub fn with_host(mut self, host: &'a dyn ExecutionHost<T>) -> Self {
        self.host = host;
        self
    }

    /// Returns this execution of the function the host knows as `method`, so
    /// a call it makes names its caller ([`CallSite::caller_method`]).
    #[must_use]
    pub fn with_method(mut self, method: T::MethodRef) -> Self {
        self.method = Some(method);
        self
    }

    /// Observes actual body entries at selected root call sites in separate SSA domains.
    ///
    /// Positions are `(block, instruction)` in this immutable root body.
    /// Duplicate or unreachable positions produce no duplicate records, and
    /// opaque or modelled calls have no entered body to observe. Execution
    /// still follows every admitted path and preserves preceding memory effects.
    /// A failed enclosing run withholds all records, including completed calls.
    /// This selects explicit sites instead of observing every root call and adds no trials.
    #[must_use]
    pub fn with_call_observations(mut self, sites: &'a [(usize, usize)]) -> Self {
        self.call_selection = CallSelection::Sites(sites);
        self
    }

    /// Observes at most `maximum` distinct actual root call-site and body pairs.
    ///
    /// Admission follows actual body entries in execution order. Repeated visits
    /// intersect an existing record without consuming another slot. Unreachable,
    /// opaque and modelled calls consume none. Execution continues after the
    /// observation limit, so its semantics and usage are unchanged; complete
    /// facts report omitted actual entries through
    /// [`StaticFacts::call_observations_truncated`].
    /// This replaces explicit site selection from [`Self::with_call_observations`].
    #[must_use]
    pub const fn with_root_call_observations(mut self, maximum: usize) -> Self {
        self.call_selection = CallSelection::Roots(maximum);
        self
    }

    /// Observes at most `maximum` distinct structural calls throughout this execution.
    ///
    /// Each record names its immediate retained parent, caller body and local
    /// call site. Ancestors consume the same bound, even without a payload;
    /// omitted ancestors never leave orphan descendants. Repeat visits meet
    /// inputs and observations in one record. Tail calls retain both their
    /// own SSA domain and the retired callers' normal-return obligations.
    /// Execution continues after observation admission stops, and incomplete
    /// enclosing executions withhold every record. This replaces root-only
    /// selection from the other call-observation builders.
    #[must_use]
    pub const fn with_transitive_call_observations(mut self, maximum: usize) -> Self {
        self.call_selection = CallSelection::Descendants(maximum);
        self
    }

    /// Observes requested normal target edges inside already-admitted invocation contexts.
    ///
    /// `maximum_bytes` reserves retained edge payload for this whole trial, across
    /// root and selected calls. Overflow does not stop execution; the result exposes
    /// [`StaticFacts::edge_output_exhausted`] for the owner's publication decision.
    /// Final aggregate output admission must count all channels without subtracting
    /// this reservation twice. Requests never add trials or grant source admission.
    #[must_use]
    pub const fn with_edge_observations(
        mut self,
        plan: &'a EdgeObservationPlan<'a, T>,
        maximum_bytes: usize,
    ) -> Self {
        self.edge_plan = Some(plan);
        self.edge_bytes = maximum_bytes;
        self
    }

    /// Requests bounded observations of the source transfers selected by the host.
    #[must_use]
    pub const fn with_code_transfers(mut self, maximum: usize) -> Self {
        self.code_transfer_limit = Some(maximum);
        self
    }

    /// Explores every path of the function and returns what they established.
    #[must_use]
    pub fn run(self) -> StaticFacts<T> {
        let width = self.context.machine().ptr_size;
        let mut ranges = self.host.disjoint_numeric_ranges();
        ranges.retain(|range| {
            range.start < range.end
                && width.mask_unsigned(range.start) == range.start
                && width.mask_unsigned(range.end.saturating_sub(1)) == range.end.saturating_sub(1)
        });
        let context = (*self.context).with_disjoint_numeric(&ranges);
        Explorer::new(self.ssa, self.method, &context, self.host)
            .with_cancellation(self.cancelled)
            .with_call_selection(self.call_selection)
            .with_edge_observations(self.edge_plan, self.edge_bytes)
            .with_code_transfers(self.code_transfer_limit)
            .run()
    }
}

/// One call, as a [`CallModel`] sees it.
///
/// Actual values are captured before the call transfers control, including
/// before a tail call releases its caller's storage. [`Self::argument`],
/// [`Self::address`] and [`Self::integer`] use ordinary argument indices;
/// [`Self::implicit_input`] reads a separate host-defined semantic key.
///
/// Memory operations act on the current execution state. Return a known value
/// through [`Self::return_value`] and set exceptional or nonreturning control
/// through [`Self::set_control`]. An omitted or explicitly unknown normal
/// return retains possible private-stack aliases from either input domain.
///
/// The executor owns configured call and data limits. Every byte a memory
/// operation reads, writes or inspects costs one step of
/// [`Budget::ExecutionSteps`], as the loop the model stands for would, charged
/// when the model returns: a model that copies a buffer costs what copying it
/// a byte at a time does. Incomplete executions withhold every fact,
/// including modeled results.
pub struct ModelCall<'a, T: Target> {
    /// Ordinary arguments followed by separately keyed implicit values.
    arguments: Vec<Value<T>>,
    /// Length of the ordinary prefix; implicit values never increase language arity.
    ordinary_count: usize,
    /// Borrowed immutable lookup shared by the model's alternatives.
    input_index: Option<&'a InputIndex>,
    /// The start of the object each ordinary argument is a pointer into,
    /// in argument order.
    objects: &'a [Option<Address>],
    /// The path the call runs on.
    state: &'a mut State<T>,
    /// The path's trace, where writes are recorded.
    trace: &'a mut Trace,
    /// Loads and stores.
    access: Access<'a>,
    /// What the call returns, once the model says.
    returned: Option<Value<T>>,
    /// How this alternative transfers control.
    control: ModelControl,
    /// Steps the memory operations have cost: one per byte each spans.
    work: usize,
    /// Whether the model read bytes no write had reached.
    read_unwritten: bool,
}

impl<'a, T: Target> ModelCall<'a, T> {
    /// Opens a call for a model.
    const fn new(
        arguments: Vec<Value<T>>,
        state: &'a mut State<T>,
        trace: &'a mut Trace,
        access: Access<'a>,
    ) -> Self {
        Self {
            ordinary_count: arguments.len(),
            input_index: None,
            objects: &[],
            arguments,
            state,
            trace,
            access,
            returned: None,
            control: ModelControl::Return,
            work: 0,
            read_unwritten: false,
        }
    }

    /// Returns the steps the model's memory operations have cost.
    const fn work(&self) -> usize {
        self.work
    }

    /// Charges the `bytes` one memory operation spans.
    const fn charge(&mut self, bytes: usize) {
        self.work = self.work.saturating_add(bytes);
    }

    /// Sets the ordinary argument boundary and the separately keyed suffix.
    const fn with_implicit_inputs(mut self, ordinary_count: usize, index: &'a InputIndex) -> Self {
        self.ordinary_count = ordinary_count;
        self.input_index = Some(index);
        self
    }

    /// States the start of the object each ordinary argument is a pointer
    /// into, in argument order, which the model's accesses through it are
    /// judged against ([`Self::pointer`]).
    #[must_use]
    pub const fn with_objects(mut self, objects: &'a [Option<Address>]) -> Self {
        self.objects = objects;
        self
    }

    /// States the objects the access the model makes next goes through.
    fn through(&mut self, through: Through) {
        self.state.memory_mut().set_through(through);
    }

    /// Takes the model's result, retaining possible input-derived stack aliases.
    ///
    /// An absent or unknown result can contain bits supplied through ordinary arguments, implicit inputs or staged stack
    /// candidates. Carrying that provenance prevents subsequent unknown
    /// stores from treating the result as disjoint from private stack storage.
    /// Refusal leaves the invocation stopped and publishes no partial result.
    fn take_returned(&mut self) -> Option<Value<T>> {
        if self.control == ModelControl::Return
            && self
                .returned
                .as_ref()
                .is_none_or(|value| matches!(value, Value::Unknown))
            && self.stack_input()
        {
            return Some(Value::StackDerived);
        }
        self.returned.take()
    }

    /// Checks every possible input for private-stack bits under the shared limits.
    fn stack_input(&self) -> bool {
        for value in &self.arguments {
            if value.may_reference_stack() {
                return true;
            }
        }
        false
    }

    /// How many ordinary arguments the call passes: those
    /// [`Self::argument`] indexes, without the implicit inputs.
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.ordinary_count
    }

    /// Returns the control outcome set by the model.
    #[must_use]
    pub const fn control(&self) -> ModelControl {
        self.control
    }

    /// Establishes how this alternative transfers control.
    ///
    /// Non-returning and trapping calls remain observable and prevent loop
    /// replacement. `Unsupported` withholds the whole execution's facts.
    /// Configured exhaustion is latched: subsequent model decisions cannot resume
    /// effects or publish a normal return.
    pub fn set_control(&mut self, control: ModelControl) {
        if matches!(self.control, ModelControl::Exhausted(_)) {
            return;
        }
        self.control = control;
        if control != ModelControl::Return {
            self.trace.taint();
        }
    }

    /// Returns ordinary argument `index`, excluding implicit and staged inputs.
    #[must_use]
    pub fn argument(&self, index: usize) -> Option<&Value<T>> {
        self.arguments.get(..self.ordinary_count)?.get(index)
    }

    /// The C++ exceptions the run has thrown and not yet finished handling.
    #[must_use]
    pub fn exceptions(&self) -> &Exceptions<T> {
        self.state.exceptions()
    }

    /// The C++ exceptions, for a model that throws, catches or finishes with
    /// one.
    pub fn exceptions_mut(&mut self) -> &mut Exceptions<T> {
        self.trace.taint();
        self.state.exceptions_mut()
    }

    /// Returns the last value supplied for the host-defined semantic input `key`.
    ///
    /// Key interpretation belongs to the same host contract as the call's ABI.
    /// An absent key returns `None`.
    /// Ordinary argument indices never address this keyed domain.
    #[must_use]
    pub fn implicit_input(&self, key: CallInputKey) -> Option<&Value<T>> {
        let position = self.input_index?.get(key)?;
        self.arguments
            .get(self.ordinary_count.checked_add(position)?)
    }

    /// Returns argument `index` as an address, when it names one.
    #[must_use]
    pub fn address(&self, index: usize) -> Option<Address> {
        self.access.address_of(self.argument(index)?, None)
    }

    /// Returns argument `index` as a pointer the model reaches memory
    /// through, when it names an address: with the object the caller formed
    /// it into, which every access through it is judged against.
    #[must_use]
    pub fn pointer(&self, index: usize) -> Option<Pointer> {
        let object = self.objects.get(index).copied().flatten();
        Some(Pointer::new(self.address(index)?, object))
    }

    /// Returns argument `index` as an integer, when it is a known one.
    #[must_use]
    pub fn integer(&self, index: usize) -> Option<i64> {
        self.argument(index)?.integer()
    }

    /// Returns argument `index` as an unsigned count at the pointer width,
    /// when it is a known integer ([`Value::native_count`]).
    #[must_use]
    pub fn count(&self, index: usize) -> Option<u128> {
        self.argument(index)?.native_count(self.ptr_size())
    }

    /// Returns the pointer-width value held in the cell argument `index`
    /// points at, as memory holds it: a count, or an address a call wrote
    /// there (`NtAllocateVirtualMemory`'s `BaseAddress` and `RegionSize`).
    /// `None` where the argument names no cell, or memory refuses the read,
    /// which it records.
    #[must_use]
    pub fn cell(&mut self, index: usize) -> Option<Value<T>> {
        let cell = self.pointer(index)?;
        let layout = TypeLayout::integer(IntegerType::new(self.ptr_size().bits(), false)?);
        self.through(Through {
            reads: cell.object(),
            writes: None,
        });
        let len = self.ptr_size().bytes();
        self.charge(len);
        if let Some(uses) = &self.trace.uses {
            uses.access(cell.address(), len);
        }
        match self
            .access
            .load_layout::<T>(self.state.memory(), cell.address(), layout)
        {
            Ok(value) => Some(value),
            Err(reason) => {
                self.state.memory_mut().refuse(reason);
                self.trace.taint();
                None
            }
        }
    }

    /// Writes `value` as an unsigned count at the pointer width where
    /// `pointer` points, in the target's byte order; unknown bytes where the
    /// order is not stated.
    pub fn write_count(&mut self, pointer: impl Into<Pointer>, value: u64) {
        let pointer = pointer.into();
        let len = self.ptr_size().bytes();
        let Some(order) = self.access.byte_order() else {
            self.write_unknown(pointer, len);
            return;
        };
        let bytes = match order {
            Endianness::Little => value.to_le_bytes(),
            Endianness::Big => value.to_be_bytes(),
        };
        let start = match order {
            Endianness::Little => 0,
            Endianness::Big => bytes.len().saturating_sub(len),
        };
        match bytes.get(start..start.saturating_add(len)) {
            Some(cell) => self.write(pointer, cell),
            None => self.write_unknown(pointer, len),
        }
    }

    /// Returns the pointer width.
    #[must_use]
    pub fn ptr_size(&self) -> PointerSize {
        self.access.ptr_size()
    }

    /// Returns the `len` bytes `pointer` points at, when all of them are
    /// known.
    #[must_use]
    pub fn read(&mut self, pointer: impl Into<Pointer>, len: usize) -> Option<Vec<u8>> {
        let pointer = pointer.into();
        let address = pointer.address();
        self.through(Through {
            reads: pointer.object(),
            writes: None,
        });
        if len > self.access.buffer_limit() {
            self.trace.taint();
            if let Some(uses) = &self.trace.uses {
                uses.unresolved();
            }
            return None;
        }
        self.charge(len);
        if let Some(uses) = &self.trace.uses {
            uses.access(address, len);
        }
        if let Some(reason) = self.state.memory().admit(address, len, Transfer::Read) {
            self.state.memory_mut().refuse(reason);
            self.trace.taint();
            return None;
        }
        self.inspect(address, len);
        let run = self.access.known_run(self.state.memory(), address, len);
        if run.len() != len {
            self.trace.taint();
            return None;
        }
        Some(run)
    }

    /// Returns bytes before the first NUL when it occurs within `limit`.
    ///
    /// The configured string bound further limits `limit`. Unknown data or a
    /// missing terminator return `None` and prevent loop replacement. Invalid
    /// object lifetimes or extents withhold the execution's facts.
    #[must_use]
    pub fn read_string(&mut self, pointer: impl Into<Pointer>, limit: usize) -> Option<Vec<u8>> {
        let pointer = pointer.into();
        let address = pointer.address();
        self.through(Through {
            reads: pointer.object(),
            writes: None,
        });
        let scan = self.access.read_string(self.state.memory(), address, limit);
        if let Ok(scan) = &scan {
            self.inspect(address, scan.inspected);
        }
        match scan {
            Ok(StringScan {
                string: Some(bytes),
                inspected,
            }) => {
                self.charge(inspected);
                if let Some(uses) = &self.trace.uses {
                    uses.access(address, bytes.len().saturating_add(1));
                }
                Some(bytes)
            }
            Ok(StringScan {
                string: None,
                inspected,
            }) => {
                self.charge(inspected);
                self.trace.taint();
                if let Some(uses) = &self.trace.uses {
                    uses.unresolved();
                }
                None
            }
            Err(reason) => {
                self.trace.taint();
                self.state.memory_mut().refuse(reason);
                None
            }
        }
    }

    /// Returns the number of bytes before a NUL found within `limit`.
    ///
    /// Uses the configured string bound and the same address and lifetime
    /// checks as [`Self::read_string`] while retaining no byte buffer. The
    /// terminator counts toward `limit`, but not the returned length. Unknown
    /// data or a missing bounded terminator return `None` and prevent loop
    /// replacement. Invalid lifetimes or extents withhold the execution's facts.
    ///
    /// # Examples
    ///
    /// A model callback can inspect an argument's bounded length without
    /// allocating a copy of its contents.
    ///
    /// ```rust
    /// use analyssa::{analysis::execution::ModelCall, target::Target};
    ///
    /// fn argument_length<T: Target>(call: &mut ModelCall<'_, T>, limit: usize) -> Option<usize> {
    ///     let pointer = call.pointer(0)?;
    ///     call.string_length(pointer, limit)
    /// }
    /// ```
    #[must_use]
    pub fn string_length(&mut self, pointer: impl Into<Pointer>, limit: usize) -> Option<usize> {
        let pointer = pointer.into();
        let address = pointer.address();
        self.through(Through {
            reads: pointer.object(),
            writes: None,
        });
        let scan = self
            .access
            .string_length(self.state.memory(), address, limit);
        if let Ok(scan) = &scan {
            self.inspect(address, scan.inspected);
        }
        match scan {
            Ok(StringScan {
                string: Some(length),
                inspected,
            }) => {
                self.charge(inspected);
                if let Some(uses) = &self.trace.uses {
                    uses.access(address, length.saturating_add(1));
                }
                Some(length)
            }
            Ok(StringScan {
                string: None,
                inspected,
            }) => {
                self.charge(inspected);
                self.trace.taint();
                if let Some(uses) = &self.trace.uses {
                    uses.unresolved();
                }
                None
            }
            Err(reason) => {
                self.trace.taint();
                self.state.memory_mut().refuse(reason);
                None
            }
        }
    }

    /// Checks one modeled write extent before image queries or memory effects.
    fn check_write_extent(&mut self, len: usize) -> bool {
        if matches!(self.control, ModelControl::Exhausted(_)) {
            return false;
        }
        let limit = self.access.buffer_limit();
        if len > limit {
            self.set_control(ModelControl::Exhausted(Exhaustion {
                budget: Budget::ExecutionBufferBytes,
                limit,
                demand: len,
            }));
            false
        } else {
            true
        }
    }

    /// Writes `bytes` where `pointer` points.
    pub fn write(&mut self, pointer: impl Into<Pointer>, bytes: &[u8]) {
        let pointer = pointer.into();
        let address = pointer.address();
        self.through(Through {
            reads: None,
            writes: pointer.object(),
        });
        if !self.check_write_extent(bytes.len()) {
            return;
        }
        self.charge(bytes.len());
        self.access
            .record_write(self.state, self.trace, address, bytes.len());
        self.access
            .write_bytes(self.state.memory_mut(), address, bytes);
    }

    /// Fills `len` bytes with `byte`.
    ///
    /// Records loop writes and checks the whole object span before writing.
    /// The repeated byte is streamed without allocating an intermediate buffer.
    pub fn fill(&mut self, pointer: impl Into<Pointer>, byte: u8, len: usize) {
        let pointer = pointer.into();
        let address = pointer.address();
        self.through(Through {
            reads: None,
            writes: pointer.object(),
        });
        if !self.check_write_extent(len) {
            return;
        }
        self.charge(len);
        self.access
            .record_write(self.state, self.trace, address, len);
        self.access
            .fill_byte(self.state.memory_mut(), address, byte, len);
    }

    /// Copies the `len` bytes the pointer argument `source` points at to
    /// where `destination` points, as they are before the call, unknown
    /// bytes included; an unknown source forgets what the destination held.
    /// The source's bytes are judged against the object the caller formed
    /// the argument into.
    pub fn copy(&mut self, destination: impl Into<Pointer>, source: usize, len: usize) {
        let destination = destination.into();
        let source_object = self.objects.get(source).copied().flatten();
        let source = self.argument(source).cloned().unwrap_or(Value::Unknown);
        self.through(Through {
            reads: source_object,
            writes: destination.object(),
        });
        let destination = destination.address();
        self.charge(len);
        if let Some(uses) = &self.trace.uses {
            if let Some(address) = self.access.address_of(&source, None) {
                uses.access(address, len);
            } else {
                uses.unresolved();
            }
        }
        if self
            .access
            .copy_model(self.state.memory_mut(), destination, &source, len)
        {
            self.access
                .record_write(self.state, self.trace, destination, len);
        } else {
            self.trace.taint();
        }
    }

    /// Writes the pointer `value` where `pointer` points.
    pub fn write_address(&mut self, pointer: impl Into<Pointer>, value: Address) {
        let pointer = pointer.into();
        let address = pointer.address();
        self.through(Through {
            reads: None,
            writes: pointer.object(),
        });
        let len = self.ptr_size().bytes();
        if !self.check_write_extent(len) {
            return;
        }
        self.charge(len);
        self.access
            .record_write(self.state, self.trace, address, len);
        self.access
            .write_address(self.state.memory_mut(), address, value);
    }

    /// Writes `len` bytes nobody knows where `pointer` points.
    pub fn write_unknown(&mut self, pointer: impl Into<Pointer>, len: usize) {
        let pointer = pointer.into();
        let address = pointer.address();
        self.through(Through {
            reads: None,
            writes: pointer.object(),
        });
        if !self.check_write_extent(len) {
            return;
        }
        self.charge(len);
        self.access
            .record_write(self.state, self.trace, address, len);
        self.access
            .write_unknown(self.state.memory_mut(), address, len);
    }

    /// Applies one attested instruction-storage permission change, returning
    /// the permissions the extent held before it where the host establishes
    /// them, or `None` where the change was refused.
    ///
    /// The extent must be nonempty and linear. A protection change reads and
    /// writes no byte, so no object's extent or lifetime judges it: a page's
    /// protection reaches past the object it holds. The host guard
    /// establishes the backing and change authority. A missing contract
    /// stops the execution without supplying a modeled call result.
    pub fn protect_code(
        &mut self,
        address: Address,
        len: usize,
        permissions: InstructionPermissions,
    ) -> Option<Option<InstructionPermissions>> {
        let result = if len == 0
            || address.pointer_size() != self.ptr_size()
            || address.linear_prefix_len(len) != len
        {
            Err(MemoryRefusal::AddressRange)
        } else if let Some(reason) = self.state.memory().access_refusal(address, len) {
            Err(reason)
        } else if let Some(guard) = self.access.write_guard() {
            let stack_owner = if matches!(address.root(), Root::Stack | Root::StackFrame(_)) {
                self.state.stack_owner(address, len)
            } else {
                None
            };
            guard.change_instruction_permissions(
                self.state.memory_mut().instruction_sources_mut(),
                address,
                len,
                stack_owner,
                permissions,
            )
        } else {
            Err(MemoryRefusal::InstructionSource)
        };
        match result {
            Ok(held) => {
                self.trace.taint();
                Some(held)
            }
            Err(reason) => {
                self.state.memory_mut().refuse(reason);
                None
            }
        }
    }

    /// Returns a fresh allocation identity without an established usable extent.
    ///
    /// Use [`Self::allocate_block`] when the model establishes accessible bytes.
    #[must_use]
    pub fn allocate(&mut self) -> Address {
        self.allocate_block(BlockSize::UNSTATED, false, None)
    }

    /// Allocates a live object of `size` made by allocator `family`,
    /// optionally initialized to zero ([`BlockSize`]: the bytes it holds on
    /// every path and the most it may hold). A size of `None` is one the call
    /// does not state: no byte is known to belong to the object, and no
    /// access is found past it.
    ///
    /// Initial zero bytes are represented lazily, independent of object size.
    /// The caller models only the successful alternative with this method;
    /// failure must be a separate call alternative.
    #[must_use]
    pub fn allocate_block(
        &mut self,
        size: impl Into<BlockSize>,
        zeroed: bool,
        allocator: impl Into<Allocator>,
    ) -> Address {
        let Some(identity) = self.state.fresh_heap() else {
            return Address::new(Root::Heap(u32::MAX), 0, self.ptr_size());
        };
        let block = Address::new(identity.root(), 0, self.ptr_size());

        self.state
            .memory_mut()
            .allocate(block.root(), size, zeroed.then_some(0), allocator);
        if let Some(site) = self.site() {
            self.state.memory_mut().locate(block.root(), site);
        }
        block
    }

    /// Returns the bytes the live allocation `address` starts is known to
    /// hold, and the allocator that made it where every path agrees on one.
    #[must_use]
    pub fn object(&self, address: Address) -> Option<(u64, Option<AllocatorFamily>)> {
        (address.offset() == 0)
            .then(|| self.state.memory().object(address.root()))
            .flatten()
    }

    /// Returns the allocator that made the live allocation `address` starts:
    /// its family and heap, each where every path agrees on it.
    #[must_use]
    pub fn made_by(&self, address: Address) -> Option<Allocator> {
        (address.offset() == 0)
            .then(|| self.state.memory().made_by(address.root()))
            .flatten()
    }

    /// Deallocates `address` by allocator `family` where one is stated, and
    /// says what it named: only the start of a live allocation of that
    /// family ends ([`FreeOutcome::Released`]), and anything else changes
    /// nothing and is classified by what the address named.
    pub fn free(&mut self, address: Address, allocator: impl Into<Allocator>) -> FreeOutcome {
        self.trace.taint();
        let site = self.site();
        self.state.memory_mut().free(address, allocator, site)
    }

    /// Ends every live block made in the heap `handle` names, as destroying
    /// the heap does; a handle that names no block of this run may be any
    /// heap's, so every live block's lifetime becomes unknown.
    pub fn destroy_heap(&mut self, handle: Option<Address>) {
        self.trace.taint();
        let site = self.site();
        let heap = handle
            .filter(|handle| matches!(handle.root(), Root::Heap(_)))
            .map(|handle| handle.root());
        self.state.memory_mut().destroy_heap(heap, site);
    }

    /// Deallocates the pointer argument `index` by allocator `family` where
    /// one is stated: [`FreeOutcome::Null`] for a null pointer, which changes
    /// nothing, [`FreeOutcome::Unknown`] for a value that names no address,
    /// and otherwise what [`Self::free`] says.
    pub fn deallocate(&mut self, index: usize, allocator: impl Into<Allocator>) -> FreeOutcome {
        if self
            .argument(index)
            .and_then(Value::constant)
            .is_some_and(ConstValue::is_zero)
        {
            return FreeOutcome::Null;
        }
        match self.address(index) {
            Some(address) => self.free(address, allocator),
            None => FreeOutcome::Unknown,
        }
    }

    /// Notes whether the `len` bytes at `address` the model inspects hold
    /// bits no write had reached: a model that reads them uses them.
    fn inspect(&mut self, address: Address, len: usize) {
        if !self.read_unwritten && len != 0 {
            self.read_unwritten = self.state.memory().unwritten_bits(address, len).is_some();
        }
    }

    /// Whether the model inspected bytes no write had reached: read them,
    /// or scanned them for a string. A copy is not an inspection.
    #[must_use]
    pub const fn read_unwritten(&self) -> bool {
        self.read_unwritten
    }

    /// The call of the outermost activation, `(block, instruction)`: the call
    /// whose execution the running model is part of.
    fn site(&self) -> Option<(usize, usize)> {
        self.state
            .frames()
            .first()
            .map(|frame| frame.position())
            .map(|position| (position.block, position.next))
    }

    /// Invalidates lifetimes a call may end through an unknown pointer.
    pub fn invalidate_lifetimes(&mut self) {
        self.trace.taint();
        if let Some(uses) = &self.trace.uses {
            uses.unresolved();
        }
        self.state.memory_mut().invalidate_lifetimes();
    }

    /// Says the call returns `value`.
    pub fn return_value(&mut self, value: Value<T>) {
        self.returned = Some(value);
    }

    /// Invalidates memory a callee can reach, including both domains of pointer inputs.
    ///
    /// A private stack pointer in either domain escapes before clobbering.
    /// A configured write policy can refuse the unresolved footprint before
    /// memory or pointer-escape state changes.
    pub fn clobber(&mut self) {
        if let Some(uses) = &self.trace.uses {
            uses.unresolved();
        }
        if matches!(self.control, ModelControl::Exhausted(_))
            || self.state.memory().refusal().is_some()
        {
            return;
        }
        if !self
            .access
            .admit_unresolved_write(self.state.memory_mut(), None, None)
        {
            return;
        }
        self.trace.taint();
        let stack_input = self.stack_input();
        if stack_input {
            self.state.memory_mut().escape_stack();
        }
        self.state.memory_mut().clobber();
    }

    /// Forgets what the call may have written through argument `index`, a
    /// pointer it was handed ([`Self::forget_from`]). A null pointer, the
    /// constant zero, reaches nothing: a write through it faults, and the
    /// call returned; nor does another known integer naming no address.
    ///
    /// An argument whose value nobody knows reaches what nobody can say: a
    /// write that may reach anything, code included ([`Self::clobber`]),
    /// unless `typed`: a pointer the routine declares to an object of a type
    /// that holds no code (a `FILE *`), which can only be data the program
    /// let escape, for [`Self::forget_escaped_data`] to forget.
    pub fn forget_through(&mut self, index: usize, typed: bool) {
        if matches!(self.control, ModelControl::Exhausted(_))
            || self.state.memory().refusal().is_some()
            || self.integer(index) == Some(0)
        {
            return;
        }
        if let Some(pointer) = self.pointer(index) {
            self.forget_from(pointer, typed);
        } else if !typed
            && self
                .argument(index)
                .is_some_and(|value| !matches!(value, Value::Const(_)))
        {
            self.clobber();
        }
    }

    /// Forgets what a write of a length nobody knows from `pointer` may
    /// change: every byte from where it points to the end of the object it
    /// points into, where something bounds that end, judged by the
    /// configured policy as any write is, so a write into code that ran is a
    /// change to that code. Memory the program can never write keeps its
    /// bytes: a write there faults, and the call returned.
    ///
    /// Where nothing bounds that object (a frame, an object nobody can
    /// place), what of it the write reaches nobody can say. Into an object of
    /// a type that holds no code (`typed`), that is data the program let
    /// escape: a frame escapes, for [`Self::forget_escaped_data`] to forget.
    /// Otherwise it is a write whose footprint is unresolved
    /// ([`MemoryWriteGuard::check_unresolved_write`]): in the stack or an
    /// allocation, which a policy admits only where no code lies, it forgets
    /// their data; anywhere else it may reach anything, code included.
    pub fn forget_from(&mut self, pointer: impl Into<Pointer>, typed: bool) {
        let pointer = pointer.into();
        if matches!(self.control, ModelControl::Exhausted(_))
            || self.state.memory().refusal().is_some()
        {
            return;
        }
        let address = pointer.address();
        let root = address.root();
        let bounded = if root.is_stack_storage() {
            None
        } else {
            self.access.rest_of_object(self.state.memory(), address)
        };
        let Some(len) = bounded else {
            if !typed {
                if let Some(uses) = &self.trace.uses {
                    uses.unresolved();
                }
                if !self
                    .access
                    .admit_unresolved_write(self.state.memory_mut(), Some(address), None)
                {
                    return;
                }
                self.trace.taint();
            }
            if root.is_stack_storage() {
                self.state.memory_mut().escape_stack();
            }
            if !typed {
                if root.is_stack_storage() || matches!(root, Root::Heap(_)) {
                    self.state.memory_mut().clobber_data();
                } else {
                    self.state.memory_mut().clobber();
                }
            }
            return;
        };
        if len == 0
            || self
                .access
                .read_only_image(self.state.memory(), address, len)
        {
            return;
        }
        self.through(Through {
            reads: None,
            writes: pointer.object(),
        });
        self.trace.taint();
        self.access
            .record_write(self.state, self.trace, address, len);
        self.access
            .write_unknown(self.state.memory_mut(), address, len);
    }

    /// Forgets the data a routine that writes no code may write without the
    /// program naming where: anything the program let escape, the stack
    /// included once a pointer into it escaped. The instructions every
    /// running body was checked against stay vouched for, by the routine's
    /// own statement. The configured policy judges the unresolved footprint
    /// ([`MemoryWriteGuard::check_unresolved_data_write`]).
    pub fn forget_escaped_data(&mut self) {
        if let Some(uses) = &self.trace.uses {
            uses.unresolved();
        }
        if matches!(self.control, ModelControl::Exhausted(_))
            || self.state.memory().refusal().is_some()
        {
            return;
        }
        if !self
            .access
            .admit_unresolved_data_write(self.state.memory_mut())
        {
            return;
        }
        self.trace.taint();
        self.state.memory_mut().clobber_data();
    }
}

#[cfg(test)]
mod tests {
    mod inactive;

    use std::{cell::Cell, process};

    use super::*;
    use crate::{
        analysis::{
            budget::WorkBudget,
            image::{ImageData, ImageRegion, Mutability},
        },
        ir::{
            function::{SsaDefSpec, SsaFunctionBuilder},
            value::ConstValue,
        },
        target::Endianness,
        testing::{self, MACHINE, MockTarget, MockType},
    };

    /// Rejects any protection query when a configured write extent already excludes it.
    struct Protection {
        query_forbidden: bool,
    }
    impl ImageData for Protection {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }
        fn region(&self, address: u64) -> Option<ImageRegion> {
            assert!(
                !self.query_forbidden,
                "over-limit write queried image protection"
            );
            (0x1000..0x1010).contains(&address).then_some(ImageRegion {
                start: 0x1000,
                end: 0x1010,
                mutability: Mutability::ReadOnly,
            })
        }
        fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
            None
        }
    }

    /// Exercises each public model write path through the real explorer.
    struct Writes {
        kind: usize,
        len: usize,
    }
    impl CallModel<MockTarget> for Writes {
        fn call(&self, call: &mut ModelCall<'_, MockTarget>) {
            let address = Address::new(Root::Symbol, 0x1000, PointerSize::Bit64);
            match self.kind {
                0 => call.write(address, &[1; 8][..self.len]),
                1 => call.fill(address, 1, self.len),
                2 => call.write_unknown(address, self.len),
                _ => unreachable!(),
            }
            if self.len > 4 {
                // A model cannot erase owner exhaustion or resume effects after it.
                call.set_control(ModelControl::Return);
                call.fill(address, 0, 1);
            }
        }
    }
    impl ExecutionHost<MockTarget> for Writes {
        fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
            (*method == 1).then_some(self)
        }
    }

    /// Model writes honor protection and limits before publishing any prefix observations.
    #[test]
    fn model_writes_withhold_facts_for_readonly_storage_and_configured_extent() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |block| {
                let before =
                    block.const_value(SsaDefSpec::tmp(MockType::U8), ConstValue::U8(42))?;
                block.call_void(crate::ir::Callee::Direct(1), vec![])?;
                block.ret(Some(before))
            })
            .unwrap_or_else(|error| panic!("{error:?}"));
        let ssa = builder.finish().unwrap_or_else(|error| panic!("{error:?}"));
        for excess in [false, true] {
            let image = Protection {
                query_forbidden: excess,
            };
            let context = AnalysisContext::new(MACHINE)
                .with_image(&image)
                .with_budget(WorkBudget::DEFAULT.with_limit(Budget::ExecutionBufferBytes, 4));
            for kind in 0..3 {
                let len = if !excess {
                    4
                } else if kind == 0 {
                    8
                } else {
                    usize::MAX
                };
                let host = Writes { kind, len };
                let facts = StaticExecution::new(&ssa, &context).with_host(&host).run();
                assert_eq!(
                    facts.completion(),
                    if excess {
                        Completion::Exhausted(Exhaustion {
                            budget: Budget::ExecutionBufferBytes,
                            limit: 4,
                            demand: len,
                        })
                    } else {
                        Completion::Memory {
                            block: 0,
                            reason: MemoryRefusal::ReadOnlyImage,
                        }
                    }
                );
                assert!(facts.constants().is_empty());
                assert!(facts.calls().is_empty());
                assert!(facts.image_writes().is_empty());
            }
        }
    }

    /// Enables cancellation only after execution reaches its first modeled call.
    struct Arm(Cell<bool>);
    impl CallModel<MockTarget> for Arm {
        fn call(&self, _: &mut ModelCall<'_, MockTarget>) {
            self.0.set(true);
        }
    }
    impl ExecutionHost<MockTarget> for Arm {
        fn model(&self, method: &u32) -> Option<&dyn CallModel<MockTarget>> {
            (*method == 1).then_some(self)
        }
    }

    /// Cancellation withholds even values established earlier in the same block.
    #[test]
    fn cancellation_inside_a_long_block_withholds_facts() {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(0, 0);
        builder
            .in_block(0, |block| {
                for value in 0..32 {
                    block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(value))?;
                }
                block.call_void(crate::ir::Callee::Direct(1), vec![])?;
                for value in 0..1024 {
                    block.const_value(SsaDefSpec::tmp(MockType::I32), ConstValue::I32(value))?;
                }
                block.ret(None)
            })
            .unwrap_or_else(|_| process::abort());
        let ssa = builder.finish().unwrap_or_else(|_| process::abort());
        let arm = Arm(Cell::new(false));
        let cancelled = || arm.0.get();
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context)
            .with_cancellation(&cancelled)
            .with_host(&arm)
            .run();
        assert_eq!(facts.completion(), Completion::Cancelled);
        assert!(facts.constants().is_empty());
        assert!(facts.calls().is_empty());
        assert!(facts.image_writes().is_empty());
        assert!(facts.usage().steps > 8 && facts.usage().steps < 1024);
    }

    /// A probe that remains clear preserves the ordinary execution result.
    #[test]
    fn a_clear_cancellation_probe_preserves_execution() {
        let ssa = testing::const_i32_return(42);
        let context = AnalysisContext::new(MACHINE);
        let facts = StaticExecution::new(&ssa, &context)
            .with_cancellation(&|| false)
            .run();
        assert!(facts.is_complete());
        assert_eq!(facts.constants().len(), 1);
    }
}
