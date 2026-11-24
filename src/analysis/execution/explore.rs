//! Exploring every path of a function.
//!
//! A path runs concretely while the program decides where it goes, and forks
//! at a branch it cannot decide. Paths advance lowest reverse-postorder first,
//! so the alternatives of one branch reach the block where they meet before
//! either runs past it, and [`Arrivals`] joins them there.

use std::{
    cell::{Cell, RefCell},
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    mem, ptr,
    rc::Rc,
    sync::Arc,
};

use rustc_hash::FxHashMap;

use self::control::CallControlEntry;
use crate::{
    analysis::{
        budget::Budget,
        context::AnalysisContext,
        execution::{
            ArgumentBindings, BranchContinuation, BranchSite, CodeMemory, CodeRefusal,
            CodeTransfer, CodeTransferKind, ContinuationBinding, EnteredCode, EntrySite,
            InstructionSources, MachineEffects, ReturnBinding,
            access::Access,
            alignment::StackCongruences,
            arguments::ExecutorArguments,
            arrivals::{Arrivals, Path},
            body::{ExecutorCodeMemory, SourceBody},
            branch::{Branches, Decision},
            edges::{EdgeAllowance, EdgeObservationPlan},
            facts::{
                ArgumentBytes, CallFacts, CallTargets, Completion, ExecutionUsage, Observations,
                StaticFacts,
            },
            findings::{Finding, FindingKind, Findings},
            frame::{CallScope, Entry, Frame, HeldRegister, Resumption},
            host::{
                CallBinding, CallModel, CallSite, ExecutionHost, ExecutorParameters, ModelControl,
                ParameterBindings,
            },
            inputs::CallInputs,
            marks::{Marker, UnwrittenUseKind},
            memory::{MemoryRefusal, Through},
            observation::{CallLimits, CallSelection},
            provenance::{Origin, Provenance},
            refine::{Conditions, Taken},
            register,
            shape::Shape,
            state::State,
            step::{Stepper, Stop},
            trace::{LoopVisit, Trace},
            types::TypeLayouts,
            uses::{UseSite, Uses},
            value::{Address, Root, Value},
        },
    },
    ir::{
        CallInputKey, Callee,
        function::{SsaFunction, StatedRegister},
        instruction::SsaInstruction,
        ops::{CallEffects, CallTermination, SsaOp},
        value::ConstValue,
        variable::{SsaVarId, VariableRole},
    },
    target::{Endianness, Target, TypeLayout},
};

mod control;

/// Immutable logical and physical return contracts for one selected model.
struct ModelBoundary {
    /// Identity-only logical destination requirement.
    binding: ReturnBinding,
    /// Selected model's actual physical effects.
    machine: MachineEffects,
    /// Whether the call is stated never to return: a model's normal return
    /// then resumes nothing, and the path ends there.
    never_returns: bool,
}

/// Exact structural identity of one retained call domain.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct CallKey {
    /// Immediate retained parent, or the original root.
    parent: Option<usize>,
    /// Actual caller identity in the enclosing body-method table.
    caller: usize,
    /// Call position local to that caller.
    site: (usize, usize),
    /// Actual entered body-method identity.
    body: usize,
}

/// One selected structural call and its callee-local observation domain.
struct CallRecord<T: Target> {
    /// Immediate parent record, or the original root body.
    parent: Option<usize>,
    /// Actual caller in the shared body-method table.
    caller_body: usize,
    /// Retired caller continuation, linked through earlier bounded records.
    returns: Option<usize>,
    /// Mapping from this retired observation to its next retired observation.
    returns_binding: ReturnBinding,
    /// Call position in the immediate caller body.
    site: (usize, usize),
    /// Actual entered target, indexed by body and method identity.
    body: usize,
    /// Entry bindings intersected across every visit, including omitted pairings.
    entries: Option<FxHashMap<SsaVarId, Value<T>>>,
    /// Facts intersected across visits to this site and target.
    observed: Observations<T>,
}

/// How a call enters the body it activates.
#[derive(Debug, Clone, Copy)]
struct EnteredBy {
    /// The call replaces the caller's activation.
    tail: bool,
    /// The call's target was a value the execution computed.
    computed: bool,
}

impl<T: Target> CallRecord<T> {
    /// Intersects supplied inputs independently from conditional body definitions.
    fn enter(&mut self, frame: &Frame<T>, inputs: Vec<SsaVarId>) {
        let current: FxHashMap<_, _> = inputs
            .into_iter()
            .filter_map(|value| Some((value, frame.held_value(value)?.clone())))
            .collect();
        let Some(entries) = &mut self.entries else {
            self.entries = Some(current);
            return;
        };
        for (value, known) in entries.iter_mut() {
            *known = known.join(current.get(value).unwrap_or(&Value::Unknown));
        }
        for value in current.keys() {
            // An input first supplied after another visit was unknown on that visit.
            entries.entry(*value).or_insert(Value::Unknown);
        }
    }

    /// Merges the entry meet into definitions only when sealing this source domain.
    fn into_facts(
        mut self,
        completion: Completion,
        sources: Option<Arc<InstructionSources>>,
    ) -> CallFacts<T> {
        for (value, held) in self.entries.into_iter().flatten() {
            self.observed.value(value, &held);
        }
        CallFacts {
            parent: self.parent,
            caller_body: self.caller_body,
            block: self.site.0,
            instruction: self.site.1,
            body: self.body,
            facts: self
                .observed
                .into_facts(completion)
                .in_call_context()
                .with_instruction_sources(sources),
        }
    }
}

/// One exploration of one function.
pub struct Explorer<'a, T: Target> {
    /// Optional host cancellation probe.
    cancelled: Option<&'a dyn Fn() -> bool>,
    /// The image, the target, and the budgets.
    context: &'a AnalysisContext<'a>,
    /// What the host says about calls.
    host: &'a dyn ExecutionHost<T>,
    /// The function explored.
    root_ssa: &'a SsaFunction<T>,
    /// The root's method identity, moved into its body record.
    root_method: Option<T::MethodRef>,
    /// Every function the exploration runs, the explored one first.
    bodies: Vec<SourceBody<'a, T>>,
    /// The indices of each body, by its address: one body entered as several
    /// methods has an entry per method.
    body_indices: FxHashMap<*const SsaFunction<T>, Vec<usize>>,
    /// The method each body was entered as, by the same index.
    methods: Vec<Option<T::MethodRef>>,
    /// Each body's order and loops, by the same index.
    shapes: Vec<Shape>,
    /// Variable layouts shared across the paths of each body.
    layouts: Vec<TypeLayouts<'a, T>>,
    /// What each body's forks are refined through, by the same index.
    conditions: Vec<Conditions<T>>,
    /// Paths waiting to run, by their order key and then their number.
    queue: BinaryHeap<Reverse<(Vec<usize>, u64)>>,
    /// The waiting paths, by number.
    waiting: FxHashMap<u64, Path<T>>,
    /// The next path number.
    next_path: u64,
    /// Where paths have arrived.
    arrivals: Arrivals<T>,
    /// What the visits observed.
    observed: Observations<T>,
    /// Access coverage shares unions across paths without retaining executed instructions.
    uses: Rc<RefCell<Uses>>,
    /// Finite source events collected when a path completes or is abandoned.
    instruction_sources: InstructionSources,
    /// Selected actual body entries and their retained-record bound.
    call_selection: CallSelection<'a>,
    /// Immutable requested edge sites, compiled under owner request-count bounds.
    edge_plan: Option<&'a EdgeObservationPlan<'a, T>>,
    /// Single trial reservation, separate from execution work and completion.
    edge_allowance: EdgeAllowance,
    /// What the run found wrong with the program.
    findings: Findings,
    /// Maximum distinct selected code transfers, when the owner requested them.
    code_transfer_limit: Option<usize>,
    /// Complete admitted body transfers before the owner limit is reached.
    code_transfers: BTreeSet<CodeTransfer>,
    /// Whether a distinct selected transfer exceeded the limit.
    code_transfers_exhausted: bool,
    /// Every visited indirect call site's resolution, recorded on every visit.
    call_targets: CallTargets<T::MethodRef>,
    /// Indexed explicit root sites, built once for logarithmic membership.
    call_sites: BTreeSet<(usize, usize)>,
    /// Whether an actual selected entry could not retain a new observation domain.
    calls_truncated: bool,
    /// Stable records keyed by ancestry, caller body, local site and entered body.
    call_indices: BTreeMap<CallKey, usize>,
    /// Separately scoped observations for calls that actually entered a body.
    call_records: Vec<CallRecord<T>>,
    /// Instructions run so far.
    steps: usize,
    /// Proven trapping instructions encountered across paths.
    traps: usize,
    /// Paths begun so far.
    paths: usize,
    /// State copies created for independent alternatives.
    alternative_snapshots: usize,
    /// Sides of forks no input satisfies.
    infeasible_paths: usize,
    /// How the exploration ended, once it has.
    ended: Option<Completion>,
}

impl<'a, T: Target> Explorer<'a, T> {
    /// An exploration of `ssa`.
    pub fn new(
        ssa: &'a SsaFunction<T>,
        method: Option<T::MethodRef>,
        context: &'a AnalysisContext<'a>,
        host: &'a dyn ExecutionHost<T>,
    ) -> Self {
        Self {
            context,
            host,
            cancelled: None,
            root_ssa: ssa,
            root_method: method,
            bodies: Vec::new(),
            body_indices: FxHashMap::default(),
            methods: Vec::new(),
            shapes: Vec::new(),
            layouts: Vec::new(),
            conditions: Vec::new(),
            queue: BinaryHeap::new(),
            waiting: FxHashMap::default(),
            next_path: 0,
            arrivals: Arrivals::default(),
            observed: Observations::default(),
            uses: Rc::default(),
            instruction_sources: InstructionSources::default(),
            call_selection: CallSelection::Sites(&[]),
            edge_plan: None,
            edge_allowance: EdgeAllowance::default(),
            findings: Findings::default(),
            code_transfer_limit: None,
            code_transfers: BTreeSet::new(),
            code_transfers_exhausted: false,
            call_targets: CallTargets::default(),
            call_sites: BTreeSet::new(),
            calls_truncated: false,
            call_indices: BTreeMap::new(),
            call_records: Vec::new(),
            steps: 0,
            traps: 0,
            paths: 0,
            alternative_snapshots: 0,
            infeasible_paths: 0,
            ended: None,
        }
    }

    /// Installs the host's cancellation probe.
    pub fn with_cancellation(mut self, cancelled: Option<&'a dyn Fn() -> bool>) -> Self {
        self.cancelled = cancelled;
        self
    }

    /// Selects actual body entries without allocating records for unvisited calls.
    pub fn with_call_selection(mut self, selection: CallSelection<'a>) -> Self {
        self.call_selection = selection;
        if let CallSelection::Sites(sites) = selection {
            self.call_sites.extend(sites.iter().copied());
        }
        self
    }

    /// Installs source-bound requests and one enclosing-trial payload reservation.
    pub fn with_edge_observations(
        mut self,
        plan: Option<&'a EdgeObservationPlan<'a, T>>,
        maximum_bytes: usize,
    ) -> Self {
        self.edge_plan = plan;
        self.edge_allowance = EdgeAllowance::new(maximum_bytes);
        self
    }

    /// Installs the owner's bound for selected source transfers.
    pub const fn with_code_transfers(mut self, maximum: Option<usize>) -> Self {
        self.code_transfer_limit = maximum;
        self
    }

    /// Retains a distinct selected transfer only after the body actually
    /// enters; `computed` when its target came from a computed value.
    fn record_code_transfer(&mut self, transfer: CodeTransfer, computed: bool) {
        let Some(maximum) = self.code_transfer_limit else {
            return;
        };
        if self.code_transfers_exhausted {
            return;
        }
        let from = self
            .methods
            .get(transfer.from_body)
            .and_then(Option::as_ref);
        let Some(to) = self.methods.get(transfer.to_body).and_then(Option::as_ref) else {
            return;
        };
        if !self.host.observes_code_transfer(from, to, computed)
            || self.code_transfers.contains(&transfer)
        {
            return;
        }
        if self.code_transfers.len() == maximum {
            self.code_transfers_exhausted = true;
            return;
        }
        self.code_transfers.insert(transfer);
    }

    /// Records an actual admitted transfer after phis, before any arrival-state join.
    fn observe_edge(&mut self, path: &Path<T>, from: usize, to: usize) {
        let Some(plan) = self.edge_plan else {
            return;
        };
        let Some(frame) = path.state.frame() else {
            return;
        };
        let Some(body) = self.body(frame.body()) else {
            return;
        };
        let observed = if path.is_root() {
            &mut self.observed
        } else {
            let Some(slot) = frame.observation().and_then(|scope| scope.body) else {
                return;
            };
            let Some(record) = self
                .call_records
                .get_mut(slot)
                .filter(|record| record.body == frame.body())
            else {
                return;
            };
            &mut record.observed
        };
        observed.edges.observe(
            plan,
            &body,
            (from, to),
            &path.state,
            self.context,
            &mut self.edge_allowance,
        );
    }

    /// Finds the observation domain of the currently executing source body.
    fn observation(&mut self, path: &Path<T>) -> Option<&mut Observations<T>> {
        if path.is_root() {
            return Some(&mut self.observed);
        }
        let frame = path.state.frame()?;
        let slot = frame.observation()?.body?;
        let record = self.call_records.get_mut(slot)?;
        (frame.body() == record.body).then_some(&mut record.observed)
    }

    /// Stops at an execution boundary when the host cancels this invocation.
    fn check_cancelled(&mut self) -> bool {
        if self.cancelled.is_some_and(|probe| probe()) {
            self.end(Completion::Cancelled);
            true
        } else {
            false
        }
    }

    /// Stops before a synchronous resolution's reported code refusal can
    /// become fallback: a code limit exhausts the trial, and code that cannot
    /// be fetched refuses it at `block`.
    fn check_code_refusal(&mut self, refusals: &Cell<Option<CodeRefusal>>, block: usize) -> bool {
        match refusals.get() {
            None => false,
            Some(CodeRefusal::Limit(exhaustion)) => {
                self.end(Completion::Exhausted(exhaustion));
                true
            }
            Some(CodeRefusal::Fetch(reason)) => {
                self.end(Completion::Memory { block, reason });
                true
            }
        }
    }

    /// Withholds this invocation after a path's semantic failure.
    fn check_path(&mut self, path: &Path<T>, block: usize) -> bool {
        if self.ended.is_some() || self.check_cancelled() {
            return false;
        }
        if path.state.identity_exhausted() {
            self.end(Completion::IdentityExhausted);
            return false;
        }
        if path
            .state
            .frame()
            .and_then(|frame| self.layouts.get(frame.body()))
            .is_some_and(TypeLayouts::is_unsupported)
        {
            self.end(Completion::UnsupportedType);
            return false;
        }
        if let Some(reason) = path.state.memory().refusal() {
            self.end(Completion::Memory { block, reason });
            return false;
        }
        // A write the program's own store did not make (a model's, which may
        // hand the storage to the kernel and fail rather than fault) into
        // storage that may not be written leaves no way on the execution
        // can state.
        if path.state.memory().fault().is_some() {
            self.end(Completion::Memory {
                block,
                reason: MemoryRefusal::ReadOnlyImage,
            });
            return false;
        }
        true
    }

    /// The order key of `path`: each activation's block, by its rank in its
    /// function, outermost first.
    fn order(&self, path: &Path<T>) -> Vec<usize> {
        path.state
            .frames()
            .iter()
            .map(|frame| {
                self.shapes
                    .get(frame.body())
                    .map_or(usize::MAX, |shape| shape.rank(frame.position().block))
            })
            .collect()
    }

    /// Queues `path` to run.
    fn push(&mut self, path: Path<T>) {
        let number = self.next_path;
        let Some(next) = self.next_path.checked_add(1) else {
            self.end(Completion::IdentityExhausted);
            return;
        };
        self.next_path = next;
        self.queue.push(Reverse((self.order(&path), number)));
        self.waiting.insert(number, path);
    }

    /// Takes the path that runs next.
    fn pop(&mut self) -> Option<Path<T>> {
        let Reverse((_, number)) = self.queue.pop()?;
        self.waiting.remove(&number)
    }

    /// Charges `demand` against `budget`, ending the exploration when it is
    /// over. Returns whether the exploration may go on.
    fn check_limit(&mut self, budget: Budget, demand: usize) -> bool {
        if let Some(exhaustion) = self.context.budget().check(budget, demand) {
            self.end(Completion::Exhausted(exhaustion));
            return false;
        }
        true
    }

    /// Ends the exploration as `completion`, unless it has already ended.
    fn end(&mut self, completion: Completion) {
        if self.ended.is_none() {
            self.ended = Some(completion);
        }
    }

    /// The body with number `body`.
    fn body(&self, body: usize) -> Option<SourceBody<'a, T>> {
        self.bodies.get(body).cloned()
    }

    /// Records what the outermost activation's `values` hold on this visit.
    fn observe(&mut self, path: &Path<T>, values: impl IntoIterator<Item = SsaVarId>) {
        let Some(observed) = self.observation(path) else {
            return;
        };
        for value in values {
            observed.value(value, &path.state.value(value));
        }
    }

    /// Records the known argument runs before the call retires or changes
    /// any memory, each within the tightest extent proven for it: the call's
    /// stated use, the object it lies in.
    fn observe_call(
        &mut self,
        path: &Path<T>,
        site: (usize, usize),
        operation: &SsaOp<T>,
        inputs: &CallInputs<T>,
        model: Option<&dyn CallModel<T>>,
    ) -> Option<()> {
        self.observation(path)?;
        let arguments = inputs.ordinary();
        // Source operand indices do not name the host's logical ABI positions.
        let limits = if inputs.is_projected() {
            CallLimits::unstated()
        } else {
            CallLimits::of(operation, arguments)
        };
        let limits = match model {
            Some(model) => limits.with_model(model, arguments, self.context.machine().ptr_size),
            None => limits,
        };
        let mut seen = Vec::new();
        if !limits.no_memory() {
            let access = Access::new(self.context).with_write_guard(self.host.write_guard());
            for (argument, value) in arguments.iter().enumerate() {
                let limit = limits.prefix(
                    argument,
                    self.context.budget().limit(Budget::ExecutionArgumentBytes),
                );
                if limit == 0 {
                    continue;
                }
                let Some(address) = access.address_of(value, None) else {
                    continue;
                };
                let (bytes, bound) = access.bounded_run(
                    path.state.memory(),
                    address,
                    limit,
                    limits.use_bound(argument),
                    path.state.proven_object(address),
                );
                if !bytes.is_empty() {
                    seen.push(ArgumentBytes {
                        argument,
                        address,
                        written: path.state.memory().written(address, bytes.len()),
                        bytes,
                        bound,
                        allocated_at: path.state.memory().allocation_site(address.root()),
                    });
                }
            }
        }
        if let Some(observed) = self.observation(path) {
            observed.call(site, seen);
        }
        Some(())
    }

    /// Runs `path`'s running activation to its block's terminator, answering
    /// every call on the way. Returns `None` when the exploration has ended.
    fn run_block(&mut self, mut path: Path<T>) -> Option<Path<T>> {
        loop {
            // Each stop is itself a call or a control transfer. Charge it
            // before execution, including blocks with no value instructions.

            let demand = self.steps.saturating_add(1);
            if !self.check_limit(Budget::ExecutionSteps, demand) {
                return None;
            }
            self.steps = demand;
            // A finding the block makes carries the decisions of this path.
            let decisions = path.trace.decisions().clone();
            path.state.memory_mut().set_decisions(decisions);
            let frame = path.state.frame()?;
            let ssa = self.body(frame.body())?;
            let position = frame.position();
            let allowance = self
                .context
                .budget()
                .limit(Budget::ExecutionSteps)
                .saturating_sub(self.steps);
            let root = path.is_root();
            let layouts = self.layouts.get(frame.body())?;
            let big_endian = self
                .context
                .machine()
                .byte_order
                .map(|order| order == Endianness::Big);
            // Where a handler may receive a raise, every instruction that may
            // raise leaves the path before it, to raise from.
            let raises = RefCell::new(Vec::new());
            let stepper = Stepper::new(&ssa, self.context)
                .with_host(self.host)
                .with_write_guard(self.host.write_guard())
                .with_layouts(layouts)
                .with_marker(Marker::new(&ssa, big_endian, &self.findings));
            let stepper = if self.raise_reaches_handler(&path) {
                stepper.with_raises(&raises)
            } else {
                stepper
            };
            let (stop, ran) = stepper.run_observed(
                &mut path.state,
                &mut path.trace,
                allowance,
                if root { Some(&mut self.observed) } else { None },
            );
            let raises = raises.into_inner();
            let raised_before = raises.last().map(|raised| raised.at);
            for raised in raises {
                let at = (position.block, raised.at);
                self.raise(
                    Path {
                        state: raised.state,
                        trace: raised.trace,
                    },
                    at,
                );
                if self.ended.is_some() {
                    return None;
                }
            }
            self.steps = self.steps.saturating_add(ran.steps);
            if !self.check_limit(Budget::ExecutionSteps, self.steps) {
                return None;
            }
            let instructions = ssa
                .block(position.block)
                .map_or(&[][..], |block| block.instructions());
            match stop {
                Stop::IdentityExhausted => {
                    self.end(Completion::IdentityExhausted);
                    return None;
                }
                Stop::UnsupportedType => {
                    self.end(Completion::UnsupportedType);
                    return None;
                }
                Stop::Exhausted(exhaustion) => {
                    self.end(Completion::Exhausted(exhaustion));
                    return None;
                }
                Stop::Trap => {
                    self.traps = self.traps.saturating_add(1);
                    self.observe_instructions(
                        &path,
                        instructions,
                        position.next,
                        ran.instructions.saturating_sub(1),
                    );
                    // The instruction that trapped is the one before where the
                    // activation stands.
                    let at = path
                        .state
                        .frame()
                        .map_or(0, |frame| frame.position().next.saturating_sub(1));
                    self.raise(path, (position.block, at));
                    return None;
                }
                Stop::Faulted => {
                    self.traps = self.traps.saturating_add(1);
                    self.observe_instructions(
                        &path,
                        instructions,
                        position.next,
                        ran.instructions.saturating_sub(1),
                    );
                    // The machine faults the store. Where a handler may
                    // receive the raise, the path before the store raised
                    // above; the path the store ran on goes no further. A
                    // store the stepper could not tell faults before it ran
                    // (a stream that runs into storage it may not write)
                    // raised from no state the execution holds.
                    let at = path
                        .state
                        .frame()
                        .map_or(0, |frame| frame.position().next.saturating_sub(1));
                    if self.raise_reaches_handler(&path) && raised_before != Some(at) {
                        self.end(Completion::Memory {
                            block: position.block,
                            reason: MemoryRefusal::ReadOnlyImage,
                        });
                    } else {
                        self.abandon(path);
                    }
                    return None;
                }
                Stop::Refused(reason) => {
                    self.end(Completion::Memory {
                        block: position.block,
                        reason,
                    });
                    return None;
                }
                Stop::Limited => {
                    self.check_limit(Budget::ExecutionSteps, self.steps.saturating_add(1));
                    return None;
                }
                Stop::Terminator => {
                    self.observe_instructions(&path, instructions, position.next, ran.instructions);
                    return Some(path);
                }
                Stop::CodeChanged(index) => {
                    self.observe_instructions(&path, instructions, position.next, ran.instructions);
                    path = self.code_changed(path, (position.block, index))?;
                }
                Stop::Resume(_) => {
                    self.observe_instructions(&path, instructions, position.next, ran.instructions);
                    path = self.resume(path)?;
                }
                Stop::Call(index) => {
                    self.observe_instructions(&path, instructions, position.next, ran.instructions);
                    let op = ssa
                        .block(position.block)
                        .and_then(|block| block.instruction(index))
                        .map(|instruction| instruction.op())?;
                    path = self.call(path, (position.block, index), op)?;
                }
            }
        }
    }

    /// Observes executed definitions directly from their operand stream.
    fn observe_instructions(
        &mut self,
        path: &Path<T>,
        instructions: &[SsaInstruction<T>],
        start: usize,
        count: usize,
    ) {
        if self.observation(path).is_none() {
            return;
        }
        for instruction in instructions.iter().skip(start).take(count) {
            instruction
                .op()
                .for_each_def(|value| self.observe(path, [value]));
        }
    }

    /// Executes a call, suspending its caller or transferring the caller's continuation.
    ///
    /// Tail calls retire stack contents and open loop proofs before the callee
    /// executes. Returns `None` when this path finishes or exploration stops.
    fn call(&mut self, mut path: Path<T>, site: (usize, usize), op: &SsaOp<T>) -> Option<Path<T>> {
        if !self.check_path(&path, site.0) {
            return None;
        }
        let (dest, arguments, implicit, staged, tail) =
            op.as_call()
                .map_or((None, &[][..], &[][..], &[][..], false), |call| {
                    (
                        call.dest,
                        call.args.as_slice(),
                        call.implicit_inputs.as_slice(),
                        call.staged.as_slice(),
                        call.tail,
                    )
                });
        let caller = path.state.frame()?;
        let caller_body = caller.body();
        let source = self.body(caller_body)?;
        // A pointer handed to a call as an argument is a pointer into the
        // object it was formed into, or the one it lands in.
        let provenance = Provenance::new(&source, Access::new(self.context));
        let objects = arguments
            .iter()
            .map(|argument| provenance.argument(&path.state, *argument))
            .collect();
        let mut inputs =
            CallInputs::capture(arguments, implicit, staged, &path.state)?.with_objects(objects);
        let never_returns = op
            .stated_call_effects()
            .is_some_and(|effects| effects.termination == CallTermination::NeverReturns);
        // Resolve one implementation while the caller and its memory are live.
        // A body owns its behavior even when the host also offers a model.
        let refusals = Cell::new(None);
        let method = self.callee(op, &path, &inputs, site, &refusals);
        let computed = op
            .as_call()
            .is_some_and(|call| matches!(call.callee, Callee::Indirect { .. }));
        // What the call entered on this visit: a computed target's method,
        // or a direct call's, which is not the one it names where the code
        // there was written since.
        self.call_targets.record(caller_body, site, method.as_ref());
        if self.check_cancelled() || self.check_code_refusal(&refusals, site.0) {
            return None;
        }
        // A call stated never to return enters no body: the statement already
        // decides that its continuation is never reached. The host's model of
        // it still runs, as it states how the call leaves: a throw raises
        // into the handlers, `exit` ends the path.
        let body = match method.as_ref().filter(|_| !never_returns) {
            Some(method) => self.source_body(&path, method, site)?,
            None => None,
        };
        let model = if body.is_none() {
            method.as_ref().and_then(|method| self.host.model(method))
        } else {
            None
        };
        let caller = path.state.frame()?;
        let caller_source = self.body(caller.body())?;
        let call_site = CallSite {
            caller: &caller_source,
            caller_method: self.methods.get(caller.body()).and_then(Option::as_ref),
            block: site.0,
            instruction: site.1,
        };
        let mut arguments = ArgumentBindings::new(
            &inputs,
            staged,
            &mut path.state,
            Access::new(self.context).with_write_guard(self.host.write_guard()),
            self.context.machine(),
        );
        if body.is_some() || model.is_some() {
            self.host.arguments(
                call_site,
                method.as_ref(),
                body.as_deref(),
                model,
                &mut arguments,
            );
        }
        let projection = arguments.finish();
        if let Some(exhaustion) = projection.exhaustion {
            self.end(Completion::Exhausted(exhaustion));
            return None;
        }
        if let Some(uses) = &path.trace.uses {
            for &(address, bytes) in &projection.accesses {
                uses.access(address, bytes);
            }
            if projection.unresolved {
                uses.unresolved();
            }
        }
        if self.check_cancelled() {
            return None;
        }
        if let Some(reason) = path.state.memory().refusal() {
            self.end(Completion::Memory {
                block: site.0,
                reason,
            });
            return None;
        }
        inputs.project(projection);
        if body.is_none()
            && model.is_none()
            && let Some(uses) = &path.trace.uses
        {
            uses.unresolved();
        }
        let use_site = UseSite {
            owner: path
                .state
                .frame()
                .and_then(|frame| frame.observation())
                .and_then(|scope| scope.body),
            position: site,
        };
        let use_inputs = self.observe_call(&path, site, op, &inputs, model);
        if body.is_none() && model.is_none() && use_inputs.is_some() {
            self.uses.borrow_mut().unresolved(use_site);
        }

        if tail {
            self.retire_loops(&mut path)?;
            if !self.retire_source(&mut path) {
                self.end(Completion::UnstableCode);
                return None;
            }
            path.state.release_stack()?;
        }
        if never_returns && model.is_none() {
            // What the callee did before not returning is seen only where a
            // handler catches its raise. Elsewhere the path ends there, as at
            // `exit`, where the statement stands on the callee's code.
            if self.raise_reaches_handler(&path)
                || !self.never_returns_stands(&mut path, method.as_ref(), site)?
            {
                self.admit_unevaluated_call(&path, site)?;
            }
            path.trace.taint();
            // A call that never returns may raise, as a throw does.
            self.raise_from_call(path, site, tail);
            return None;
        }
        if let Some(method) = method {
            if let Some(body) = body {
                if !self.machine_matches(&body, Some(&method)) {
                    self.end(Completion::IncompatibleMachine);
                    return None;
                }
                if !self.host.admit_code_source(
                    &body,
                    Some(&method),
                    path.state.memory_mut().instruction_sources_mut(),
                ) {
                    self.end(Completion::UnstableCode);
                    return None;
                }
                return self.enter(
                    path,
                    site,
                    &method,
                    body,
                    &inputs,
                    EnteredBy { tail, computed },
                );
            }
            if let Some(model) = model {
                let caller = path.state.frame()?;
                let caller_method = self.methods.get(caller.body()).cloned().flatten();
                let call_site = CallSite {
                    caller: &caller_source,
                    caller_method: caller_method.as_ref(),
                    block: site.0,
                    instruction: site.1,
                };
                let binding = self.host.model_return_binding(call_site, &method, model);
                let machine = self.host.model_machine_effects(call_site, &method, model);
                self.record_argument_use(&path, site, op);
                return self.modelled(
                    path,
                    site,
                    dest,
                    model,
                    inputs,
                    ModelBoundary {
                        binding,
                        machine,
                        never_returns,
                    },
                );
            }
        }
        if self.check_cancelled() {
            return None;
        }
        self.admit_unevaluated_call(&path, site)?;
        path.state.clear_machine();
        self.record_argument_use(&path, site, op);
        let returned = self.unanswered(&mut path, op, inputs.all());
        // A call the execution does not follow may raise.
        if self.raise_reaches_handler(&path) {
            self.raise_from_call(path.clone(), site, tail);
            if self.ended.is_some() {
                return None;
            }
        }
        self.resumed(path, site, dest, Some(returned))
    }

    /// The body `method` runs as on `path`: what the code there holds now,
    /// where the host snapshots it, else the body the host knows. `None`
    /// where the execution ended, refusing the code or the memory it read.
    fn source_body(
        &mut self,
        path: &Path<T>,
        method: &T::MethodRef,
        site: (usize, usize),
    ) -> Option<Option<SourceBody<'a, T>>> {
        let refusals = Cell::new(None);
        let snapshot = self.host.body_snapshot(
            method,
            CodeMemory::new(path.state.memory(), self.context)
                .with_stack_owners(&path.state)
                .with_uses(path.trace.uses.as_ref())
                .with_code_refusals(&refusals),
        );
        if self.check_cancelled() || self.check_code_refusal(&refusals, site.0) {
            return None;
        }
        match snapshot {
            Ok(Some(body)) => Some(Some(SourceBody::Owned(body))),
            Ok(None) => Some(self.host.body(method).map(SourceBody::Borrowed)),
            Err(reason) => {
                self.end(Completion::Memory {
                    block: site.0,
                    reason,
                });
                None
            }
        }
    }

    /// Whether the statement that the call at `site` never returns stands on
    /// the code its callee `method` holds now: the callee's body admits as
    /// the source it was lifted from ([`ExecutionHost::admit_code_source`]),
    /// so the termination the host read in that body holds, whatever the
    /// body writes before it. A call to no body the execution knows has only
    /// the host's word. `None` where the execution ended.
    fn never_returns_stands(
        &mut self,
        path: &mut Path<T>,
        method: Option<&T::MethodRef>,
        site: (usize, usize),
    ) -> Option<bool> {
        let Some(method) = method else {
            return Some(false);
        };
        let Some(body) = self.source_body(path, method, site)? else {
            return Some(false);
        };
        Some(
            self.machine_matches(&body, Some(method))
                && self.host.admit_code_source(
                    &body,
                    Some(method),
                    path.state.memory_mut().instruction_sources_mut(),
                ),
        )
    }

    /// Records a use of never-written bits where the call `op` at `site`
    /// hands any to a model or to a function the execution cannot follow,
    /// either of which may read them. A body run in place is handed them,
    /// and uses them only where it does.
    fn record_argument_use(&self, path: &Path<T>, site: (usize, usize), op: &SsaOp<T>) {
        let Some(body) = path.state.frame().map(Frame::body) else {
            return;
        };
        if op
            .uses()
            .iter()
            .any(|used| path.state.marks(*used).is_some())
        {
            self.findings.record(Finding {
                body,
                block: site.0,
                instruction: site.1,
                kind: FindingKind::Unwritten(UnwrittenUseKind::Argument),
                access: None,
                object: None,
                witness: path.trace.decisions().witness(),
            });
        }
    }

    /// Checks the configured policy before skipping call effects.
    fn admit_unevaluated_call(&mut self, path: &Path<T>, site: (usize, usize)) -> Option<()> {
        let caller = path.state.frame()?;
        let caller_source = self.body(caller.body())?;
        let admitted = self.host.admit_unevaluated_call(CallSite {
            caller: &caller_source,
            caller_method: self.methods.get(caller.body()).and_then(Option::as_ref),
            block: site.0,
            instruction: site.1,
        });
        if admitted {
            Some(())
        } else {
            self.end(Completion::UnevaluatedEffects);
            None
        }
    }

    /// Resolves the method a call reaches: the one a direct call names, the one the
    /// host resolves a virtual call's receiver to, or for a call through a
    /// pointer, the one the host places at the image address it holds. A
    /// constrained call uses a separate host lookup for its type-selected
    /// implementation, preserving the original receiver-address shape.
    /// Dispatch borrows the argument snapshot captured before a tail transfer,
    /// avoiding a second read or payload copy from caller state.
    fn callee(
        &self,
        op: &SsaOp<T>,
        path: &Path<T>,
        inputs: &CallInputs<T>,
        site: (usize, usize),
        refusals: &Cell<Option<CodeRefusal>>,
    ) -> Option<T::MethodRef> {
        let call = op.as_call()?;
        match &call.callee {
            Callee::Direct(method) => {
                let frame = path.state.frame()?;
                let caller = self.body(frame.body())?;
                self.host.method_for_call_with_inputs(
                    CallSite {
                        caller: &caller,
                        caller_method: self.methods.get(frame.body()).and_then(Option::as_ref),
                        block: site.0,
                        instruction: site.1,
                    },
                    method,
                    inputs.view(),
                    CodeMemory::new(path.state.memory(), self.context)
                        .with_stack_owners(&path.state)
                        .with_uses(path.trace.uses.as_ref())
                        .with_code_refusals(refusals),
                )
            }
            Callee::Virtual(method) => self.host.dispatch(method, inputs.ordinary().first()?),
            Callee::Constrained {
                method,
                constraint_type,
                instance,
            } => {
                let receiver = if *instance {
                    Some(inputs.ordinary().first()?)
                } else {
                    None
                };
                self.host
                    .dispatch_constrained(method, constraint_type, receiver)
            }
            Callee::Indirect { fptr, .. } => {
                let target = Access::new(self.context)
                    .with_write_guard(self.host.write_guard())
                    .address_of(&path.state.value(*fptr), None)?;
                let frame = path.state.frame()?;
                let caller = self.body(frame.body())?;
                self.host.method_at_call(
                    CallSite {
                        caller: &caller,
                        caller_method: self.methods.get(frame.body()).and_then(Option::as_ref),
                        block: site.0,
                        instruction: site.1,
                    },
                    target,
                    CodeMemory::new(path.state.memory(), self.context)
                        .with_stack_owners(&path.state)
                        .with_uses(path.trace.uses.as_ref())
                        .with_code_refusals(refusals),
                )
            }
        }
    }

    /// Enters a body using the caller's actual values and its enclosing
    /// continuation, by the call `by` describes.
    ///
    /// A tail caller's stack and loop proofs are already retired. Its values stay
    /// available for ABI binding until the callee replaces the frame;
    /// ordinary calls append a frame. A memory refusal withholds the
    /// execution's facts.
    fn enter(
        &mut self,
        mut path: Path<T>,
        site: (usize, usize),
        method: &T::MethodRef,
        source: SourceBody<'a, T>,
        inputs: &CallInputs<T>,
        by: EnteredBy,
    ) -> Option<Path<T>> {
        let EnteredBy { tail, computed } = by;
        let body = &*source;
        if !self.check_limit(
            Budget::ExecutionDepth,
            path.state.depth().saturating_add(usize::from(!tail)),
        ) {
            return None;
        }
        let caller = path.state.frame()?;
        let caller_body = caller.body();
        let caller_activation = caller.activation();
        let parent = caller.observation().and_then(|scope| scope.body);
        let inherited = if tail {
            caller
                .observation()
                .and_then(|scope| scope.body.or(scope.returns))
        } else {
            None
        };
        let caller_continuation = caller.return_binding();
        let caller_observation = caller.observation();
        let mut selected = self.call_selection.selects(
            path.is_root(),
            parent.is_some(),
            self.call_sites.contains(&site),
        );
        let caller_source = self.body(caller_body)?;
        let caller_method = self.methods.get(caller_body).cloned().flatten();
        let site = CallSite {
            caller: &caller_source,
            caller_method: caller_method.as_ref(),
            block: site.0,
            instruction: site.1,
        };

        let binding = self.host.body_return_binding(site, method, body);
        let continuation = if tail {
            caller_continuation.meet(binding)
        } else {
            binding
        };
        let inherited_binding = if tail {
            caller_observation.map_or(ReturnBinding::Identity, |scope| {
                if scope.body.is_some() {
                    binding
                } else {
                    scope.returns_binding.meet(binding)
                }
            })
        } else {
            ReturnBinding::Identity
        };
        path.state.set_reservation_possible(true);
        let Some(activation) = path.state.fresh_activation() else {
            self.end(Completion::IdentityExhausted);
            return None;
        };
        let caller = path.state.frame()?;
        let shared_entry = match self.host.call_stack(site) {
            Some(stack) => stack
                .stack_pointer
                .map_or(Some(caller.entry_stack_address()), |pointer| {
                    caller.value(pointer).address()
                })
                .and_then(|pointer| {
                    pointer
                        .root()
                        .may_reference_stack()
                        .then(|| pointer.moved(stack.adjustment, self.context.machine().ptr_size))
                        .flatten()
                }),
            None => self
                .host
                .call_stack_from_entry(site)
                .and_then(|adjustment| {
                    caller
                        .entry_stack_address()
                        .moved(adjustment, self.context.machine().ptr_size)
                }),
        };
        let entry = shared_entry.unwrap_or_else(|| {
            Address::new(
                Root::StackFrame(activation),
                0,
                self.context.machine().ptr_size,
            )
        });
        // The callee runs on the stack pointer its caller actually passes, with
        // the low bits the path established for that root. Its convention's
        // entry alignment is not consulted: a compiler that sees every caller
        // of a local function may enter it below that alignment.
        let stack_congruence = path.state.stack_congruence(entry.root());
        let entered = self.enter_control(
            &mut path,
            CallControlEntry {
                body,
                method,
                site,
                entry,
                activation,
                tail,
            },
        )?;
        let position = (site.block, site.instruction);
        let index = self.body_index(source.clone(), method);
        let key = CallKey {
            parent,
            caller: caller_body,
            site: position,
            body: index,
        };
        if selected
            && !self.call_indices.contains_key(&key)
            && self
                .call_selection
                .maximum()
                .is_some_and(|maximum| self.call_records.len() >= maximum)
        {
            self.calls_truncated = true;
            selected = false;
        }
        if let Some(observed) = self.observation(&path) {
            let use_site = UseSite {
                owner: parent,
                position,
            };
            let arguments = observed.call_pointers(position);
            if let Some(uses) = &mut path.trace.uses {
                uses.enter(
                    use_site,
                    path.state.depth().saturating_add(usize::from(!tail)),
                    &arguments,
                );
            }
        }
        let (implicit, input_index) = inputs.implicit();
        let bindings = ParameterBindings::of_call(CallBinding {
            arguments: inputs.ordinary(),
            implicit: Some((implicit, input_index)),
            stack: Some((
                path.state.memory_mut(),
                shared_entry,
                Access::new(self.context).with_write_guard(self.host.write_guard()),
            )),
        });
        let mut bindings = bindings
            .with_physical(inputs.physical())
            .with_uses(path.trace.uses.as_ref());
        let site = CallSite {
            caller: &caller_source,
            caller_method: caller_method.as_ref(),
            block: position.0,
            instruction: position.1,
        };
        self.host.parameters(site, method, body, &mut bindings);
        let exhaustion = bindings.exhaustion();
        let mut bound = bindings.finish();
        if let Some(exhaustion) = exhaustion {
            self.end(Completion::Exhausted(exhaustion));
            return None;
        }
        if let Some(reason) = path.state.memory().refusal() {
            self.end(Completion::Memory {
                block: site.block,
                reason,
            });
            return None;
        }
        let active = if selected {
            let next = self.call_records.len();
            let slot = *self.call_indices.entry(key).or_insert(next);
            if slot == next {
                self.call_records.push(CallRecord {
                    parent,
                    caller_body,
                    returns: inherited,
                    returns_binding: inherited_binding,
                    site: position,
                    body: index,
                    entries: None,
                    observed: Observations::default(),
                });
            }
            if let Some(record) = self.call_records.get_mut(slot) {
                record.returns_binding = record.returns_binding.meet(inherited_binding);
            }
            Some(slot)
        } else {
            None
        };
        let returns_since = match caller_observation {
            Some(scope) if tail && scope.body.is_some() => caller_activation,
            Some(scope) if tail => scope.returns_since,
            _ => 0,
        };
        let observation = (active.is_some() || inherited.is_some()).then_some(CallScope {
            body: active,
            returns: inherited,
            returns_binding: inherited_binding,
            returns_since,
        });
        let observed_inputs: Vec<_> = if selected {
            bound
                .iter()
                .filter_map(|(value, _)| body.variable(*value).map(|_| *value))
                .collect()
        } else {
            Vec::new()
        };
        // A body leaves its caller its result alone: what models left in
        // registers is not the callee's to read.
        path.state.clear_machine();
        // The callee finds the caller's registers as the call leaves them:
        // those the caller's code names as its call hands them on, the rest
        // as the caller found them.
        let mut registers = path
            .state
            .frame()
            .map(|frame| frame.registers().clone())
            .unwrap_or_default();
        if let Some(caller) = path.state.frame() {
            let call = caller_source
                .block(position.0)
                .and_then(|block| block.instruction(position.1));
            let mut handover = call
                .and_then(SsaInstruction::id)
                .map(|at| caller_source.handover_state(at))
                .unwrap_or_default();
            // What the call states as its inputs it hands on as they are.
            handover.extend(
                call.and_then(|call| call.op().as_call())
                    .map_or(&[][..], |call| call.implicit_inputs.as_slice())
                    .iter()
                    .copied()
                    .map(StatedRegister::from),
            );
            self.overlay_registers(
                &mut registers,
                caller_body,
                caller_source.named_registers(),
                &handover,
                caller,
            );
        }
        // A link register nothing else states receives the address the
        // callee returns to.
        if let Some(entered) = &entered
            && let Some(Some(key)) = entered.link
        {
            registers.entry(key).or_insert_with(|| HeldRegister {
                value: entered.expected.clone(),
                layout: None,
            });
        }
        if let Some(entered) = &entered
            && let Some(entry) = entered.entry
            && bound.iter().all(|(value, _)| *value != entry)
        {
            bound.push((entry, entered.expected.clone()));
        }
        // A register the callee reads that its call binds nothing to holds
        // what the caller left there; its stack pointer is the one it is
        // entered on.
        for input in body.register_entries() {
            if body.entry_stack_pointer() != Some(input.value)
                && bound.iter().all(|(value, _)| *value != input.value)
                && let Some(held) = registers.get(&input.key)
            {
                bound.push((input.value, held.value.clone()));
            }
        }
        // The callee's copy of a pointer argument is formed into its object;
        // its own stack pointer is a pointer into no object.
        let formed: Vec<(SsaVarId, Address)> = bound
            .iter()
            .filter(|(value, _)| {
                body.variable(*value)
                    .is_none_or(|variable| variable.role() != Some(VariableRole::EntryStackPointer))
            })
            .filter_map(|(value, held)| Some((*value, inputs.object_of(held)?)))
            .collect();
        let mut frame = Frame::enter_at(
            body,
            index,
            activation,
            entry,
            bound,
            self.context.machine().ptr_size,
        )
        .with_return_binding(continuation)
        .with_stack_congruence(stack_congruence)
        .with_stack_extent(self.host.stack_extent(body, Some(method)))
        .with_realigned_extent(self.host.realigned_stack_extent(body, Some(method)))
        .with_stack_input_aliases(inputs.may_reference_stack())
        .with_observation(observation);
        frame.set_registers(registers);
        frame.set_expected_return(entered.map(|entered| entered.expected));
        // The callee's code is checked as it is now, on entry.
        frame.set_code_generation(path.state.memory().instruction_sources().generation());
        for (value, object) in formed {
            frame.set_formed(value, Some(Origin::Object(object)));
        }
        if selected && let Some(record) = active.and_then(|slot| self.call_records.get_mut(slot)) {
            record.enter(&frame, observed_inputs);
        }
        if tail {
            path.state.tail_call(frame)?;
        } else {
            path.state.call(frame);
        }
        self.record_code_transfer(
            CodeTransfer {
                from_body: caller_body,
                block: position.0,
                instruction: position.1,
                to_body: index,
                kind: if tail {
                    CodeTransferKind::TailCall
                } else {
                    CodeTransferKind::Call
                },
            },
            computed,
        );
        Some(path)
    }

    /// Continues an unresolved indirect branch without adding or retiring a physical frame.
    fn continue_branch(
        &mut self,
        path: Path<T>,
        site: (usize, usize),
        target: SsaVarId,
    ) -> Option<Path<T>> {
        if self.check_cancelled() || !self.check_path(&path, site.0) {
            return None;
        }
        let caller = path.state.frame()?;
        let caller_index = caller.body();
        let caller_body = self.body(caller_index)?;
        let caller = path.state.frame()?;
        let at = BranchSite {
            body: &caller_body,
            method: self.methods.get(caller_index).and_then(Option::as_ref),
            block: site.0,
            instruction: site.1,
        };
        let target = caller.value(target);
        let refusals = Cell::new(None);
        let memory = CodeMemory::new(path.state.memory(), self.context)
            .with_stack_owners(&path.state)
            .with_uses(path.trace.uses.as_ref())
            .with_code_refusals(&refusals);
        let request = self.host.branch_continuation(at, &target, memory);
        if self.check_cancelled() || self.check_code_refusal(&refusals, site.0) {
            return None;
        }
        let plan = match request {
            Ok(Some(plan)) => plan,
            Ok(None) => {
                self.end(Completion::Stuck { block: site.0 });
                return None;
            }
            Err(reason) => {
                self.end(Completion::Memory {
                    block: site.0,
                    reason,
                });
                return None;
            }
        };
        // The branch hands on the registers it states, read by the branch.
        let handed = caller_body
            .block(site.0)
            .and_then(|block| block.instruction(site.1))
            .map(|instruction| match instruction.op() {
                SsaOp::IndirectBranch { state, .. } => {
                    state.iter().copied().map(StatedRegister::from).collect()
                }
                _ => Vec::new(),
            })
            .unwrap_or_default();
        self.continue_source(path, site, plan, CodeTransferKind::Branch, handed)
    }

    /// Checks after `site`, the instruction whose write changed code an
    /// activation had entered, or the call during which a callee did,
    /// whether the code the running activation executes before its block
    /// transfers changed: where it did, the activation finishes `site`'s
    /// source instruction and goes on in a lift of what the code holds now
    /// ([`Self::resume`]); otherwise it goes on in the lift it runs, and the
    /// write is met where control enters code it changed
    /// ([`Self::entered_change`]).
    fn code_changed(&mut self, mut path: Path<T>, site: (usize, usize)) -> Option<Path<T>> {
        let frame = path.state.frame()?;
        if frame.resuming().is_some() {
            return Some(path);
        }
        let index = frame.body();
        let since = frame.code_generation();
        let body = self.body(index)?;
        let method = self.methods.get(index).and_then(Option::as_ref);
        let at = BranchSite {
            body: &body,
            method,
            block: site.0,
            instruction: site.1,
        };
        if self
            .host
            .code_changed(at, path.state.memory().instruction_sources(), since)
        {
            path.state
                .frame_mut()?
                .set_resuming(Some(Resumption::After(site.0, site.1)));
        }
        Some(path)
    }

    /// Checks, as control enters `to` from `from` in the running activation,
    /// whether a write since its lift changed the code it runs on the way:
    /// where one did, the activation goes on in a lift of what the code holds
    /// now, from the host block the host names, once it has run the pure
    /// instructions `to` begins with ([`Resumption::Entry`]). Returns `false`
    /// where the trial ended.
    fn entered_change(&mut self, path: &mut Path<T>, from: usize, edge: bool, to: usize) -> bool {
        let Some(frame) = path.state.frame() else {
            return true;
        };
        let since = frame.code_generation();
        if since == path.state.memory().instruction_sources().generation() {
            return true;
        }
        let index = frame.body();
        let Some(body) = self.body(index) else {
            return true;
        };
        let site = EntrySite {
            body: &body,
            method: self.methods.get(index).and_then(Option::as_ref),
            from,
            edge,
            to,
        };
        let host = match self.host.entered_code_change(
            site,
            path.state.memory().instruction_sources(),
            since,
        ) {
            EnteredCode::Unchanged => return true,
            EnteredCode::Changed(Some(host)) => host,
            EnteredCode::Changed(None) => {
                self.end(Completion::UnstableCode);
                return false;
            }
        };
        if let Some(frame) = path.state.frame_mut() {
            frame.set_resuming(Some(Resumption::Entry {
                from,
                edge,
                block: to,
                host,
            }));
        }
        true
    }

    /// Goes on in a lift of the code as it is now, after the source
    /// instruction whose write changed the running activation's code
    /// ([`ExecutionHost::resume_continuation`]).
    fn resume(&mut self, path: Path<T>) -> Option<Path<T>> {
        let frame = path.state.frame()?;
        let (block, instruction) = match frame.resuming()? {
            Resumption::After(block, instruction) => (block, instruction),
            Resumption::Entry {
                from,
                edge,
                block,
                host,
            } => return self.resume_at_entry(path, (from, edge), block, host),
        };
        let site = (block, instruction);
        let index = frame.body();
        let body = self.body(index)?;
        let method = self.methods.get(index).and_then(Option::as_ref);
        let at = BranchSite {
            body: &body,
            method,
            block: site.0,
            instruction: site.1,
        };
        let refusals = Cell::new(None);
        let memory = CodeMemory::new(path.state.memory(), self.context)
            .with_stack_owners(&path.state)
            .with_uses(path.trace.uses.as_ref())
            .with_code_refusals(&refusals);
        let request = self.host.resume_continuation(at, memory);
        if self.check_cancelled() || self.check_code_refusal(&refusals, site.0) {
            return None;
        }
        let plan = match request {
            Ok(Some(plan)) => plan,
            Ok(None) => {
                self.end(Completion::UnstableCode);
                return None;
            }
            Err(reason) => {
                self.end(Completion::Memory {
                    block: site.0,
                    reason,
                });
                return None;
            }
        };
        // The registers as the source instruction that wrote leaves them.
        let handed = body
            .block(site.0)
            .and_then(|block| block.instruction(site.1))
            .and_then(SsaInstruction::id)
            .map(|at| body.resume_state(at))
            .unwrap_or_default();
        self.continue_source(path, site, plan, CodeTransferKind::Rewritten, handed)
    }

    /// Goes on in a lift of the code as it is now, from where a run of the
    /// host block `host` begins, control having entered `block` from `from`
    /// on its way there ([`ExecutionHost::entry_continuation`]), with the
    /// registers the body states there.
    fn resume_at_entry(
        &mut self,
        path: Path<T>,
        (from, edge): (usize, bool),
        block: usize,
        host: u64,
    ) -> Option<Path<T>> {
        let frame = path.state.frame()?;
        let index = frame.body();
        let next = frame.position().next;
        let body = self.body(index)?;
        let site = EntrySite {
            body: &body,
            method: self.methods.get(index).and_then(Option::as_ref),
            from,
            edge,
            to: block,
        };
        let refusals = Cell::new(None);
        let memory = CodeMemory::new(path.state.memory(), self.context)
            .with_stack_owners(&path.state)
            .with_uses(path.trace.uses.as_ref())
            .with_code_refusals(&refusals);
        let request = self.host.entry_continuation(site, host, memory);
        if self.check_cancelled() || self.check_code_refusal(&refusals, block) {
            return None;
        }
        let plan = match request {
            Ok(Some(plan)) => plan,
            Ok(None) => {
                self.end(Completion::UnstableCode);
                return None;
            }
            Err(reason) => {
                self.end(Completion::Memory { block, reason });
                return None;
            }
        };
        let handed = body.host_entry_state(host);
        self.continue_source(
            path,
            (block, next),
            plan,
            CodeTransferKind::Rewritten,
            handed,
        )
    }

    /// Brings `registers` (what the activation's machine registers hold
    /// where the code running `body` does not name them) over a transfer
    /// that code makes: each register it names holds what `state` says of
    /// it, read from `frame`, and nothing known where `state` says nothing.
    fn overlay_registers(
        &self,
        registers: &mut BTreeMap<CallInputKey, HeldRegister<T>>,
        body: usize,
        named: &[CallInputKey],
        state: &[StatedRegister],
        frame: &Frame<T>,
    ) {
        for key in named {
            registers.remove(key);
        }
        let layouts = self.layouts.get(body);
        // Only registers the code names: the rest of what a transfer states
        // (its target, a frame slot) is no register the activation holds.
        for input in state.iter().filter(|input| named.contains(&input.key)) {
            let layout = layouts.and_then(|layouts| layouts.get(input.value));
            let held = frame
                .held_value(input.value)
                .filter(|value| **value != Value::Unknown)
                .and_then(|value| match input.widened {
                    None => Some((value.clone(), layout)),
                    Some(widening) => {
                        register::widen(value, layout, widening, self.context.machine())
                            .map(|(value, layout)| (value, Some(layout)))
                    }
                });
            match held {
                Some((value, layout)) => {
                    registers.insert(input.key, HeldRegister { value, layout });
                }
                None => {
                    registers.remove(&input.key);
                }
            }
        }
    }

    /// Transfers the current physical activation into a validated source
    /// namespace, the registers the code there finds being `handed` (the
    /// transfer's statement of the registers it names) over what the
    /// activation held.
    fn continue_source(
        &mut self,
        mut path: Path<T>,
        site: (usize, usize),
        plan: BranchContinuation<T>,
        kind: CodeTransferKind,
        handed: Vec<StatedRegister>,
    ) -> Option<Path<T>> {
        let caller_index = path.state.frame()?.body();
        let caller_body = self.body(caller_index)?;
        let refusals = Cell::new(None);
        let memory = CodeMemory::new(path.state.memory(), self.context)
            .with_stack_owners(&path.state)
            .with_uses(path.trace.uses.as_ref())
            .with_code_refusals(&refusals);
        if !self.check_limit(
            Budget::ExecutionBufferBytes,
            plan.bindings
                .len()
                .saturating_mul(mem::size_of::<ContinuationBinding>()),
        ) || self.check_cancelled()
        {
            return None;
        }
        let snapshot = self.host.body_snapshot(&plan.method, memory);
        if self.check_cancelled() || self.check_code_refusal(&refusals, site.0) {
            return None;
        }
        let source = match snapshot {
            Ok(Some(body)) => SourceBody::Owned(body),
            Ok(None) => match self.host.body(&plan.method) {
                Some(body) => SourceBody::Borrowed(body),
                None => {
                    self.end(Completion::Stuck { block: site.0 });
                    return None;
                }
            },
            Err(reason) => {
                self.end(Completion::Memory {
                    block: site.0,
                    reason,
                });
                return None;
            }
        };
        if !self.machine_matches(&source, Some(&plan.method)) {
            self.end(Completion::IncompatibleMachine);
            return None;
        }
        let index = self.body_index(source.clone(), &plan.method);
        let caller = path.state.frame()?;
        let stack = match plan.stack {
            Some(stack) => stack
                .stack_pointer
                .map_or(Some(caller.entry_stack_address()), |pointer| {
                    caller.value(pointer).address()
                })
                .and_then(|address| {
                    address.moved(stack.adjustment, self.context.machine().ptr_size)
                }),
            None => Some(caller.entry_stack_address()),
        };
        let Some(stack) = stack.filter(|stack| {
            stack.root() == caller.stack_root()
                && stack.pointer_size() == self.context.machine().ptr_size
        }) else {
            self.end(Completion::Memory {
                block: site.0,
                reason: MemoryRefusal::AddressRange,
            });
            return None;
        };
        let mut destinations = BTreeSet::new();
        let mut bound = Vec::with_capacity(plan.bindings.len());
        for binding in &plan.bindings {
            let held = caller_body.variable(binding.from).is_some_and(|from| {
                from.is_entry_definition() || caller.held_value(binding.from).is_some()
            });
            // A resume point's value the activation has not defined (a pass
            // moved its definition past the point) leaves that register
            // unknown; a branch's state is read by the branch, so defined.
            if !held && kind == CodeTransferKind::Rewritten {
                continue;
            }
            let valid = held
                && source
                    .variable(binding.into)
                    .is_some_and(|into| into.is_entry_definition())
                && destinations.insert(binding.into);
            let from = self
                .layouts
                .get(caller_index)
                .and_then(|layout| layout.get(binding.from));
            let into = self
                .layouts
                .get(index)
                .and_then(|layout| layout.get(binding.into));
            if !valid {
                self.end(Completion::UnstableCode);
                return None;
            }
            // A value narrower than the register it stands for fills it as
            // the state says it does; one no integer of that width holds
            // leaves the register unknown.
            let (carried, from) = match binding.widened {
                None => (caller.value(binding.from), from),
                Some(widening) => match register::widen(
                    &caller.value(binding.from),
                    from,
                    widening,
                    self.context.machine(),
                ) {
                    Some((value, layout)) => (value, Some(layout)),
                    None => continue,
                },
            };
            // A register's value the entry reads through another view of the
            // register is that view's bits of it.
            let same = from.zip(into).is_some_and(|(from, into)| {
                let from = ReturnBinding::for_layout(from, self.context.machine().ptr_size);
                let into = ReturnBinding::for_layout(into, self.context.machine().ptr_size);
                from != ReturnBinding::Unknown && from == into
            });
            // A representation no view of the register reads is no
            // unchanged value the code there enters with.
            let value = match source.variable(binding.into) {
                Some(entry) if !same => {
                    match register::view(&carried, from, entry.var_type(), self.context.machine()) {
                        Some(value) => value,
                        None => {
                            self.end(Completion::UnstableCode);
                            return None;
                        }
                    }
                }
                _ => carried,
            };
            if source.entry_stack_pointer() == Some(binding.into) && value.address() != Some(stack)
            {
                self.end(Completion::UnstableCode);
                return None;
            }
            bound.push((binding.into, value));
        }
        // The code there finds the registers as the code before left them:
        // those it names as the transfer states them, the rest as the
        // activation held them.
        let mut registers = caller.registers().clone();
        self.overlay_registers(
            &mut registers,
            caller_index,
            caller_body.named_registers(),
            &handed,
            caller,
        );
        for input in source.register_entries() {
            if source.entry_stack_pointer() != Some(input.value)
                && bound.iter().all(|(value, _)| *value != input.value)
                && let Some(held) = registers.get(&input.key)
            {
                bound.push((input.value, held.value.clone()));
            }
        }
        // The frame takes what the code before held in SSA alone.
        let pointer = self.context.machine().ptr_size;
        let entry = Address::new(caller.stack_root(), caller.release_entry_stack(), pointer);
        let writes: Vec<(Address, Value<T>, TypeLayout)> = plan
            .frame
            .iter()
            .filter_map(|write| {
                let held = caller.held_value(write.from)?;
                let layout = self.layouts.get(caller_index)?.get(write.from)?;
                let (value, layout) = match write.widened {
                    None => (held.clone(), layout),
                    Some(widening) => {
                        register::widen(held, Some(layout), widening, self.context.machine())?
                    }
                };
                Some((entry.moved(write.offset, pointer)?, value, layout))
            })
            .collect();
        for (address, value, layout) in writes {
            if let Err(exhaustion) = Access::new(self.context)
                .with_write_guard(self.host.write_guard())
                .store_layout(
                    path.state.memory_mut(),
                    &Value::Addr(address),
                    &value,
                    layout,
                    None,
                )
            {
                self.end(Completion::Exhausted(exhaustion));
                return None;
            }
        }
        path.state.clear_machine();
        if !self.check_path(&path, site.0) || self.check_cancelled() {
            return None;
        }
        self.retire_loops(&mut path)?;
        if !self.retire_source(&mut path)
            || !self.host.admit_code_source(
                &source,
                Some(&plan.method),
                path.state.memory_mut().instruction_sources_mut(),
            )
        {
            self.end(Completion::UnstableCode);
            return None;
        }
        if path
            .state
            .continue_source(&source, index, stack, bound)
            .is_none()
        {
            self.end(Completion::IdentityExhausted);
            return None;
        }
        path.state.frame_mut()?.set_registers(registers);
        self.record_code_transfer(
            CodeTransfer {
                from_body: caller_index,
                block: site.0,
                instruction: site.1,
                to_body: index,
                kind,
            },
            matches!(kind, CodeTransferKind::Branch | CodeTransferKind::Rewritten),
        );
        // The lift entered is of the code as it is now.
        let generation = path.state.memory().instruction_sources().generation();
        path.state.frame_mut()?.set_code_generation(generation);
        Some(path)
    }

    /// The number of `body` in the table of bodies, adding it when new.
    fn body_index(&mut self, body: SourceBody<'a, T>, method: &T::MethodRef) -> usize {
        let key = ptr::from_ref(&*body);
        for &index in self.body_indices.get(&key).map_or(&[][..], Vec::as_slice) {
            if self
                .methods
                .get(index)
                .and_then(Option::as_ref)
                .is_some_and(|known| known == method)
            {
                return index;
            }
        }
        self.register(body, Some(method.clone()))
    }

    /// Adds `body`, entered as `method`, to the table of bodies with what
    /// every body keeps beside it there, and returns its number.
    fn register(&mut self, body: SourceBody<'a, T>, method: Option<T::MethodRef>) -> usize {
        let index = self.bodies.len();
        self.body_indices
            .entry(ptr::from_ref(&*body))
            .or_default()
            .push(index);
        self.shapes.push(Shape::of(&body));
        self.layouts.push(TypeLayouts::of_body(
            body.clone(),
            self.context.machine().ptr_size,
        ));
        self.conditions.push(Conditions::default());
        self.methods.push(method);
        self.bodies.push(body);
        index
    }

    /// Explores every model alternative before resuming the caller.
    fn modelled(
        &mut self,
        path: Path<T>,
        site: (usize, usize),
        dest: Option<SsaVarId>,
        model: &dyn CallModel<T>,
        inputs: CallInputs<T>,
        boundary: ModelBoundary,
    ) -> Option<Path<T>> {
        let mut path = path;
        if !self.check_limit(
            Budget::ExecutionBufferBytes,
            mem::size_of_val(boundary.machine.outputs.as_slice()),
        ) {
            return None;
        }
        // A model leaves its call's outputs alone: nothing earlier is its to
        // keep, a register it preserves being no output of the call.
        path.state.clear_machine();
        if !model.tracks_all_memory_accesses() {
            if let Some(uses) = &path.trace.uses {
                uses.unresolved();
            }
            let owner = path
                .state
                .frame()
                .and_then(|frame| frame.observation())
                .and_then(|scope| scope.body);
            self.uses.borrow_mut().unresolved(UseSite {
                owner,
                position: site,
            });
        }
        let owner = path
            .state
            .frame()
            .and_then(|frame| frame.observation())
            .and_then(|scope| scope.body);
        if let Some(observed) = self.observation(&path) {
            let pointers = observed.call_pointers(site);
            if let Some(uses) = &mut path.trace.uses {
                uses.enter(
                    UseSite {
                        owner,
                        position: site,
                    },
                    path.state.depth().saturating_add(1),
                    &pointers,
                );
            }
        }
        let alternatives = model.alternatives();
        if alternatives == 0 {
            self.end(Completion::Stuck { block: site.0 });
            return None;
        }
        self.paths = self.paths.saturating_add(alternatives.saturating_sub(1));
        if !self.check_limit(Budget::ExecutionPaths, self.paths) {
            return None;
        }
        if alternatives == 1 {
            let mut path = path;
            let (returned, control, read) = self.model(&mut path, model, 0, &inputs);
            self.record_model_read(&path, site, read);
            return self.after_model(path, site, dest, returned, control, &boundary);
        }
        for alternative in 0..alternatives {
            let mut fork = path.clone();
            self.alternative_snapshots = self.alternative_snapshots.saturating_add(1);
            let body = fork.state.frame().map(Frame::body).unwrap_or_default();
            fork.trace.fork_call(body, site, alternative);

            let (returned, control, read) = self.model(&mut fork, model, alternative, &inputs);
            self.record_model_read(&fork, site, read);
            if let Some(fork) = self.after_model(fork, site, dest, returned, control, &boundary) {
                self.push(fork);
            }
            if self.ended.is_some() {
                break;
            }
        }
        None
    }

    /// Applies a model's control outcome without inventing a normal return.
    fn after_model(
        &mut self,
        mut path: Path<T>,
        site: (usize, usize),
        dest: Option<SsaVarId>,
        returned: Value<T>,
        control: ModelControl,
        boundary: &ModelBoundary,
    ) -> Option<Path<T>> {
        if !self.check_path(&path, site.0) {
            return None;
        }
        if let Some(reason) = path.state.memory().refusal() {
            self.end(Completion::Memory {
                block: site.0,
                reason,
            });
            return None;
        }
        if !self.check_limit(Budget::ExecutionBufferBytes, returned.owned_bytes()) {
            return None;
        }
        if control == ModelControl::Return && boundary.never_returns {
            // A call stated never to return resumes nothing, whatever the
            // model's normal outcome: there is no return to validate.
            self.abandon(path);
            return None;
        }
        if let Some(uses) = &mut path.trace.uses {
            uses.returned(path.state.depth().saturating_add(1));
        }
        match control {
            ModelControl::Return => {
                let mut keys = BTreeSet::new();
                for output in &boundary.machine.outputs {
                    if !keys.insert(output.key) {
                        self.end(Completion::Memory {
                            block: site.0,
                            reason: MemoryRefusal::MachineState,
                        });
                        return None;
                    }
                    if !path.state.has_machine(output.key)
                        && !self.check_limit(
                            Budget::ExecutionBufferBytes,
                            path.state.machine_storage_bytes(1),
                        )
                    {
                        return None;
                    }
                    let pointer = self.context.machine().ptr_size;
                    path.state.publish_machine(
                        output.key,
                        output.value(returned.clone(), pointer),
                        output.layout,
                        pointer,
                    );
                }
                let returned = self.bind_returned(boundary.binding, Some(returned));
                self.resumed(path, site, dest, returned)
            }
            ModelControl::NoReturn => {
                self.abandon(path);
                None
            }
            ModelControl::Trap => {
                self.traps = self.traps.saturating_add(1);
                self.raise(path, site);
                None
            }
            ModelControl::Unsupported => {
                self.end(Completion::Stuck { block: site.0 });
                None
            }
            ModelControl::Exhausted(exhaustion) => {
                self.end(Completion::Exhausted(exhaustion));
                None
            }
        }
    }

    /// Runs `model` on `inputs`. A model whose effect is not only memory
    /// spoils the loops the path is in: removing them would remove the call.
    /// The steps its memory operations cost are charged to the run, and a run
    /// they exhaust ends there.
    fn model(
        &mut self,
        path: &mut Path<T>,
        model: &dyn CallModel<T>,
        alternative: usize,
        inputs: &CallInputs<T>,
    ) -> (Value<T>, ModelControl, bool) {
        if self.check_cancelled() {
            return (Value::Unknown, ModelControl::Unsupported, false);
        }
        if !model.only_memory() {
            path.trace.taint();
        }
        path.state.set_reservation_possible(true);
        // A finding the model makes carries the decisions of this alternative.
        let decisions = path.trace.decisions().clone();
        path.state.memory_mut().set_decisions(decisions);
        let mut call = inputs.model_call(
            &mut path.state,
            &mut path.trace,
            Access::new(self.context).with_write_guard(self.host.write_guard()),
        );
        model.call_alternative(alternative, &mut call);
        let read_unwritten = call.read_unwritten();
        let work = call.work();
        let control = call.control();
        let returned = call.take_returned();
        // The objects the model's last access went through describe no
        // access after it.
        path.state.memory_mut().set_through(Through::default());
        self.steps = self.steps.saturating_add(work);
        if let Some(exhaustion) = self
            .context
            .budget()
            .check(Budget::ExecutionSteps, self.steps)
        {
            return (
                Value::Unknown,
                ModelControl::Exhausted(exhaustion),
                read_unwritten,
            );
        }
        (returned.unwrap_or(Value::Unknown), control, read_unwritten)
    }

    /// Records the use a model at `site` made of never-written bits it read
    /// through a pointer it was handed, where `read` says it did.
    fn record_model_read(&self, path: &Path<T>, site: (usize, usize), read: bool) {
        if read && let Some(body) = path.state.frame().map(Frame::body) {
            self.findings.record(Finding {
                body,
                block: site.0,
                instruction: site.1,
                kind: FindingKind::Unwritten(UnwrittenUseKind::Argument),
                access: None,
                object: None,
                witness: path.trace.decisions().witness(),
            });
        }
    }

    /// Resumes an ordinary call or completes a tail model's inherited return.
    ///
    /// A tail return never defines the retired caller's result or executes its
    /// remaining instructions. Body tail calls already replaced that frame.
    fn resumed(
        &mut self,
        mut path: Path<T>,
        site: (usize, usize),
        dest: Option<SsaVarId>,
        returned: Option<Value<T>>,
    ) -> Option<Path<T>> {
        let tail = path
            .state
            .frame()
            .and_then(|frame| self.body(frame.body()))
            .is_some_and(|body| {
                body.block(site.0)
                    .and_then(|block| block.instruction(site.1))
                    .and_then(|instruction| instruction.op().as_call())
                    .is_some_and(|call| call.tail)
            });
        if tail {
            self.finish(path, returned, false, None);
            return None;
        }
        self.resume_after(&mut path, site, dest, returned.unwrap_or(Value::Unknown));
        if !self.check_path(&path, site.0) {
            return None;
        }
        // A callee may have written the code its caller goes on with.
        if path.state.frame().is_some_and(|frame| {
            frame.code_generation() != path.state.memory().instruction_sources().generation()
        }) {
            path = self.code_changed(path, site)?;
        }

        if let Some(dest) = dest {
            self.observe(&path, [dest]);
        }
        Some(path)
    }

    /// Applies an unanswered call's complete purity promise or conservative effects.
    ///
    /// A pure call preserves memory and allocation lifetimes. Its unknown result
    /// can still return stack-derived argument or target bits; keeping that
    /// provenance prevents a later store from treating a private-stack alias as
    /// an unrelated opaque value. Other unanswered calls retain their effects.
    fn unanswered(
        &self,
        path: &mut Path<T>,
        operation: &SsaOp<T>,
        arguments: &[Value<T>],
    ) -> Value<T> {
        let mut stack_bits = false;
        for argument in arguments {
            stack_bits |= argument.may_reference_stack();
        }
        if let Some(fptr) = operation.as_call().and_then(|call| call.callee.fptr()) {
            stack_bits |= path.state.value(fptr).may_reference_stack();
        }
        if operation
            .stated_call_effects()
            .is_some_and(CallEffects::is_pure)
        {
            return if stack_bits {
                Value::StackDerived
            } else {
                Value::Unknown
            };
        }
        path.trace.taint();
        if stack_bits {
            path.state.memory_mut().escape_stack();
        }
        path.state.memory_mut().invalidate_lifetimes();
        path.state.memory_mut().clobber();
        Value::Unknown
    }

    /// Moves the running activation past the call at `site`, defining its
    /// result as `returned`.
    fn resume_after(
        &self,
        path: &mut Path<T>,
        site: (usize, usize),
        dest: Option<SsaVarId>,
        returned: Value<T>,
    ) {
        if let Some(dest) = dest {
            path.state.define(dest, returned);
        }
        // Each value the call returns beside its result is what the return
        // handed back under its key, and unknown where it handed back none.
        let handed = path.state.take_handed();
        let results: Vec<(CallInputKey, SsaVarId)> = path
            .state
            .frame()
            .and_then(|frame| self.body(frame.body()))
            .and_then(|body| {
                let call = body.block(site.0)?.instruction(site.1)?.op().as_call()?;
                Some(
                    call.results
                        .iter()
                        .map(|result| (result.key, result.value))
                        .collect(),
                )
            })
            .unwrap_or_default();
        for (key, value) in results {
            let held = handed
                .iter()
                .find(|(handed_key, _)| *handed_key == key)
                .map_or(Value::Unknown, |(_, held)| held.clone());
            path.state.define(value, held);
        }
        if let Some(frame) = path.state.frame_mut() {
            let mut position = frame.position();
            position.next = site.1.saturating_add(1);
            frame.move_to(position);
        }
    }
    /// Carries `path` into `to` in its running activation, by `entry` (an
    /// edge, or a raise dispatching to a handler), and admits it there.
    /// Returns the path to run on, or `None` when what it could do has
    /// already been explored.
    fn transfer(
        &mut self,
        mut path: Path<T>,
        entry: Entry,
        to: usize,
        join_states: bool,
    ) -> Option<Path<T>> {
        let from = entry.from();
        let frame = path.state.frame()?;
        let (body, activation) = (frame.body(), frame.activation());
        let ssa = self.body(body)?;
        let enters = {
            let shape = self.shapes.get(body)?;
            shape.is_header(to) && !shape.in_loop(to, from)
        };
        loop {
            let shape = self.shapes.get(body)?;
            let Some(visit) = path.trace.pop_loop(|visit| {
                visit.activation == activation && !shape.in_loop(visit.header, to)
            }) else {
                break;
            };
            self.close_visit(&path, &visit, Some((from, to)));
        }
        let phi_count = ssa.block(to).map_or(0, |block| block.phi_count());
        let demand = self.steps.saturating_add(phi_count);
        if !self.check_limit(Budget::ExecutionSteps, demand) {
            return None;
        }
        self.steps = demand;
        if let Some(frame) = path.state.frame_mut() {
            frame.arrive(&ssa, to, entry);
        }

        if !self.check_path(&path, from) {
            return None;
        }
        // Code a write changed since the lift is met as control gets there,
        // the values the block enters with defined.
        if !self.entered_change(&mut path, from, matches!(entry, Entry::Edge(_)), to) {
            return None;
        }
        if enters {
            path.trace.enter_loop(to, activation);
        }
        let phis = ssa
            .block(to)
            .into_iter()
            .flat_map(|block| block.phi_nodes())
            .map(|phi| phi.result());
        self.observe(&path, phis);
        // A raise enters from the middle of its block, by no edge: what it
        // holds is no value on an edge of the function.
        if let Entry::Edge(_) = entry {
            self.observe_edge(&path, from, to);
        }
        if join_states {
            // Paths that disagree on what their conditions establish of a
            // value still read stand apart; a refinement of one no later
            // instruction reads keeps no path from joining another.
            let (conditions, bodies) = (&self.conditions, &self.bodies);
            let live = |body: usize, block: usize, value: SsaVarId| {
                conditions
                    .get(body)
                    .zip(bodies.get(body))
                    .is_none_or(|(conditions, ssa)| conditions.live_in(ssa, block, value))
            };
            self.arrivals.admit(path, &live)
        } else {
            // This execution has never admitted alternatives. Exact loops can
            // advance owned state without cloning growing memory on each byte.
            // Once alternatives exist, retain history even when another path
            // has returned: separate loop headers still need those joins.
            self.arrivals.clear();
            Some(path)
        }
    }

    /// Records the end of `visit` in the outermost activation: its effect when
    /// it left by `exit` as one clean run, and a spoiled loop otherwise.
    fn close_visit(&mut self, path: &Path<T>, visit: &LoopVisit, exit: Option<(usize, usize)>) {
        if !path.is_root() {
            return;
        }
        let effect = exit.filter(|_| visit.clean).and_then(|exit| {
            let ssa = self.body(0)?;
            let shape = self.shapes.first()?;
            let mut values = BTreeMap::new();
            let mut append = |value| {
                let held = path.state.value(value);

                values.insert(value, held).is_none().then_some(())
            };
            for block in 0..ssa.block_count() {
                if !shape.in_loop(visit.header, block) {
                    continue;
                }
                let Some(block) = ssa.block(block) else {
                    continue;
                };
                for phi in block.phi_nodes() {
                    append(phi.result())?;
                }
                for instruction in block.instructions() {
                    instruction
                        .op()
                        .try_for_each_def(|value| append(value).ok_or(()))
                        .ok()?;
                }
            }

            let mut exit_values = Vec::with_capacity(values.len());
            for value in values {
                exit_values.push(value);
            }
            visit.effect(exit, path.state.memory(), exit_values)
        });
        self.observed.loop_visit(visit.header, effect);
    }

    /// Checks every host-declared machine rule against the one used by all frames.
    fn machine_matches(&self, body: &SsaFunction<T>, method: Option<&T::MethodRef>) -> bool {
        self.host
            .body_machine(body, method)
            .is_none_or(|mut declared| {
                declared.byte_order = declared
                    .byte_order
                    .or_else(|| self.context.image().map(|image| image.endianness()));
                declared == self.context.machine()
            })
    }

    /// Explores every path from the root under the configured executor limits.
    pub fn run(mut self) -> StaticFacts<T> {
        if self
            .edge_plan
            .is_some_and(|plan| plan.pointer_size() != self.context.machine().ptr_size)
        {
            return self.facts(Completion::IncompatibleMachine);
        }
        if self.check_cancelled() {
            return self.facts(Completion::Cancelled);
        }
        let ssa = self.root_ssa;
        if !self.machine_matches(ssa, self.root_method.as_ref()) {
            return self.facts(Completion::IncompatibleMachine);
        }
        let mut sources = InstructionSources::default();
        if !self
            .host
            .admit_code_source(ssa, self.root_method.as_ref(), &mut sources)
        {
            return self.facts(Completion::UnstableCode);
        }
        let method = self.root_method.take();
        self.register(SourceBody::Borrowed(ssa), method);
        let ptr_size = self.context.machine().ptr_size;
        let root_stack = Address::new(Root::Stack, 0, ptr_size);
        let stack_congruence = match self
            .host
            .entry_stack_congruence(ssa, self.methods.first().and_then(Option::as_ref))
        {
            Some(entry) => match entry.at_root(root_stack) {
                Some(root) => Some(root),
                None => {
                    return self.facts(Completion::Memory {
                        block: 0,
                        reason: MemoryRefusal::StackAlignment,
                    });
                }
            },
            None => None,
        };
        let entered = self
            .host
            .root_entry_values(ssa, self.methods.first().and_then(Option::as_ref));
        let mut destinations = BTreeSet::new();
        if entered.iter().any(|(value, _)| {
            !destinations.insert(*value)
                || ssa
                    .variable(*value)
                    .is_none_or(|variable| !variable.is_entry_definition())
        }) {
            return self.facts(Completion::Memory {
                block: 0,
                reason: MemoryRefusal::MachineState,
            });
        }
        let entry = Frame::enter_at(ssa, 0, 0, root_stack, entered, ptr_size)
            .with_stack_congruence(stack_congruence)
            .with_stack_extent(
                self.host
                    .stack_extent(ssa, self.methods.first().and_then(Option::as_ref)),
            )
            .with_realigned_extent(
                self.host
                    .realigned_stack_extent(ssa, self.methods.first().and_then(Option::as_ref)),
            );
        let mut state = State::new(entry);
        state
            .memory_mut()
            .set_disjoint_numeric(self.context.disjoint_numeric().to_vec(), ptr_size);
        state.memory_mut().set_findings(self.findings.clone());
        if let Some(extents) = self
            .host
            .object_extents(ssa, self.methods.first().and_then(Option::as_ref))
        {
            state.memory_mut().set_object_extents(extents);
        }
        // The root's frame below the stack pointer it was entered with is
        // its own, and nothing has written it yet.
        state
            .memory_mut()
            .begin_frame(root_stack.root(), root_stack.offset());
        *state.memory_mut().instruction_sources_mut() = sources;
        let mut path = Path {
            state,
            trace: Trace::new().with_uses(self.uses.clone()),
        };
        // The root's code is checked against the memory it starts in, as a
        // callee's is on entry.
        if let Some(method) = self.methods.first().cloned().flatten() {
            let refusals = Cell::new(None);
            let memory = CodeMemory::new(path.state.memory(), self.context)
                .with_stack_owners(&path.state)
                .with_code_refusals(&refusals);
            if let Err(reason) = self.host.body_snapshot(&method, memory) {
                return self.facts(Completion::Memory { block: 0, reason });
            }
        }
        if !self.initialize_root_control(&mut path, ssa) {
            let completion = self.ended.unwrap_or(Completion::Memory {
                block: 0,
                reason: MemoryRefusal::ReturnControl,
            });
            return self.facts(completion);
        }
        self.push(path);
        while self.ended.is_none() {
            if self.check_cancelled() {
                break;
            }
            let Some(path) = self.pop() else {
                break;
            };
            self.advance(path);
        }
        let completion = self.ended.unwrap_or_else(|| {
            if self.layouts.iter().any(TypeLayouts::is_unsupported) {
                Completion::UnsupportedType
            } else {
                Completion::Complete
            }
        });
        self.facts(completion)
    }

    /// Seals observations with the same counters on every completion path.
    fn facts(mut self, completion: Completion) -> StaticFacts<T> {
        if completion == Completion::Complete {
            let uses = self.uses.borrow();
            self.observed.bound_call_uses(None, &uses);
            for (slot, record) in self.call_records.iter_mut().enumerate() {
                record.observed.bound_call_uses(Some(slot), &uses);
            }
        }
        let edge_output_exhausted = self.edge_allowance.exhausted;
        let sources = if completion == Completion::Complete && !self.instruction_sources.is_empty()
        {
            Some(Arc::new(self.instruction_sources))
        } else {
            None
        };

        let invocations = if completion == Completion::Complete {
            let mut records = self.call_records;
            // Structural admission is parent-first. Root-only selection retains
            // its local-site ordering and has no parent indices to remap.
            if !matches!(self.call_selection, CallSelection::Descendants(_)) {
                records.sort_unstable_by_key(|record| (record.site, record.body));
            }
            records
                .into_iter()
                .map(|record| {
                    let mut record = record.into_facts(completion, sources.clone());
                    record.facts = record.facts.with_edge_output_limit(edge_output_exhausted);
                    record
                })
                .collect()
        } else {
            Vec::new()
        };
        self.observed
            .into_facts(completion)
            .with_body_methods(self.methods)
            .with_code_transfers(self.code_transfers, self.code_transfers_exhausted)
            .with_call_targets(self.call_targets)
            .with_instruction_sources(sources)
            .with_invocations(invocations, self.calls_truncated)
            .with_edge_output_limit(edge_output_exhausted)
            .with_usage(ExecutionUsage {
                steps: self.steps,
                paths: self.paths,
                traps: self.traps,
                state_snapshots: self
                    .alternative_snapshots
                    .saturating_add(self.arrivals.snapshots()),
                infeasible_paths: self.infeasible_paths,
            })
            .with_findings(self.findings.all())
    }

    /// Runs `path` through its block and sends it on wherever the block's end
    /// sends it.
    fn advance(&mut self, path: Path<T>) {
        let Some(mut path) = self.run_block(path) else {
            return;
        };
        let Some(frame) = path.state.frame() else {
            return;
        };
        let (body, block) = (frame.body(), frame.position().block);
        let Some(ssa) = self.body(body) else {
            return;
        };
        let Some(ending) = ssa.block(block) else {
            self.end(Completion::Stuck { block });
            return;
        };
        if let Some(SsaOp::IndirectBranch {
            target,
            resolved_targets,
            ..
        }) = ending.control_terminator()
            && resolved_targets.is_empty()
        {
            let site = (block, frame.position().next);
            if let Some(path) = self.continue_branch(path, site, *target) {
                self.push(path);
            }
            return;
        }
        let Some(layouts) = self.layouts.get(body) else {
            return;
        };
        // A decision on never-written bits uses them, and so does the root's
        // return of them; a callee's return hands them on.
        if let Some(terminator) = ending.control_terminator() {
            let kind = match terminator {
                SsaOp::Return { .. } => {
                    (path.state.depth() == 1).then_some(UnwrittenUseKind::Return)
                }
                _ => Some(UnwrittenUseKind::Branch),
            };
            if let Some(kind) = kind
                && terminator
                    .uses()
                    .iter()
                    .any(|used| path.state.marks(*used).is_some())
            {
                self.findings.record(Finding {
                    body,
                    block,
                    instruction: frame.position().next,
                    kind: FindingKind::Unwritten(kind),
                    access: None,
                    object: None,
                    witness: path.trace.decisions().witness(),
                });
            }
        }
        let decision = Branches::new(&ssa, self.context.machine().ptr_size)
            .with_float_rules(self.context.machine().float)
            .with_layouts(layouts)
            .decide(ending, |value| path.state.value(value));
        if !self.check_path(&path, block) {
            return;
        }
        match decision {
            Decision::Goto(target) => {
                if let Some(path) = self.transfer(path, Entry::Edge(block), target, self.paths != 0)
                {
                    self.push(path);
                }
            }
            Decision::Either(targets) => {
                // Each side is refined by the condition it took; a side no
                // input satisfies is no path.
                let definitions = self
                    .conditions
                    .get(body)
                    .map(|conditions| conditions.definitions(&ssa));
                let layouts = self.layouts.get(body);
                let mut forks = Vec::with_capacity(targets.len());
                for target in targets {
                    let mut fork = path.clone();
                    let feasible =
                        match (definitions, layouts, ending.control_terminator()) {
                            (Some(definitions), Some(layouts), Some(terminator)) => {
                                Taken::new(&ssa, definitions, layouts, self.context.machine())
                                    .refine(terminator, target, &mut fork.state)
                            }
                            _ => true,
                        };
                    if feasible {
                        forks.push((target, fork));
                    } else {
                        self.infeasible_paths = self.infeasible_paths.saturating_add(1);
                    }
                }
                self.paths = self.paths.saturating_add(forks.len().saturating_sub(1));
                if !self.check_limit(Budget::ExecutionPaths, self.paths) {
                    return;
                }
                for (target, mut fork) in forks {
                    self.alternative_snapshots = self.alternative_snapshots.saturating_add(1);
                    let body = fork.state.frame().map(Frame::body).unwrap_or_default();
                    fork.trace.fork(body, block, target);
                    if let Some(fork) = self.transfer(fork, Entry::Edge(block), target, true) {
                        self.push(fork);
                    }
                }
            }
            Decision::Return => {
                let target = match ending.control_terminator() {
                    Some(SsaOp::Return {
                        transfer: Some(transfer),
                        ..
                    }) => transfer.target,
                    _ => None,
                };
                if !self.validate_return_control(&path, &ssa, target, block) {
                    return;
                }
                let returned = match ending.control_terminator() {
                    Some(SsaOp::Return {
                        value: Some(value), ..
                    }) => Some(path.state.value(*value)),
                    _ => None,
                };
                // What it hands back beside the result, for the call it
                // returns to: a tail call's caller receives it as its own.
                let handed = match ending.control_terminator() {
                    Some(SsaOp::Return { results, .. }) => results
                        .iter()
                        .map(|result| (result.key, path.state.value(result.value)))
                        .collect(),
                    _ => Vec::new(),
                };
                path.state.hand_back(handed);
                // The caller finds the registers as the callee leaves them.
                let exit = path.state.frame().map(|frame| {
                    let mut registers = frame.registers().clone();
                    let mut handover = ending
                        .instructions()
                        .last()
                        .and_then(SsaInstruction::id)
                        .map(|at| ssa.handover_state(at))
                        .unwrap_or_default();
                    // What the return states it hands back as it is.
                    if let Some(SsaOp::Return {
                        transfer: Some(stated),
                        ..
                    }) = ending.control_terminator()
                    {
                        handover.extend(stated.handover.iter().copied().map(StatedRegister::from));
                    }
                    self.overlay_registers(
                        &mut registers,
                        frame.body(),
                        ssa.named_registers(),
                        &handover,
                        frame,
                    );
                    registers
                });
                self.finish(path, returned, true, exit);
            }
            Decision::Halt => self.abandon(path),
            Decision::Stuck => self.end(Completion::Stuck { block }),
        }
    }

    /// Whether a raise on `path` may reach a handler: whether a clause
    /// protects where one of its activations stands, the running one in its
    /// block or a caller at its call.
    fn raise_reaches_handler(&self, path: &Path<T>) -> bool {
        path.state.frames().iter().any(|frame| {
            self.shapes
                .get(frame.body())
                .and_then(|shape| shape.dispatch(frame.position().block))
                .is_some()
        })
    }

    /// Raises on `path` from the call at `site`, which may raise rather than
    /// return. A tail call raises in its caller's caller: the activation it
    /// replaces is retired already.
    fn raise_from_call(&mut self, mut path: Path<T>, mut site: (usize, usize), tail: bool) {
        if !self.raise_reaches_handler(&path) {
            self.abandon(path);
            return;
        }
        if tail {
            if path.state.depth() <= 1 || path.state.return_from().is_none() {
                self.abandon(path);
                return;
            }
            let Some(caller) = path.state.frame() else {
                return;
            };
            site = (caller.position().block, caller.position().next);
        }
        self.raise(path, site);
    }

    /// Raises on `path` at the instruction `site` of its running activation,
    /// as it stands before the instruction completes.
    ///
    /// The raise enters every handler a clause protecting the instruction's
    /// block names, each a path of its own whose handler phis take the
    /// values the raise brings ([`Entry::Raise`]); a handler that resumes the
    /// function after one that ran elsewhere is entered past that handler's
    /// effects, which the execution does not follow. Since a handler may
    /// decline the raise, it then goes on outward: the activation unwinds and
    /// the raise is its caller's, at its call. Where no activation left
    /// stands where a clause protects it, the raise ends the path, as a trap
    /// does. A clause whose handler runs elsewhere and names
    /// no place the function resumes at may resume it where the execution
    /// cannot say, and ends the run.
    fn raise(&mut self, mut path: Path<T>, mut site: (usize, usize)) {
        loop {
            if self.ended.is_some() {
                return;
            }
            if !self.raise_reaches_handler(&path) {
                self.abandon(path);
                return;
            }
            let Some(body) = path.state.frame().map(Frame::body) else {
                return;
            };
            let dispatch = self
                .shapes
                .get(body)
                .and_then(|shape| shape.dispatch(site.0))
                .cloned();
            if let Some(dispatch) = dispatch {
                if dispatch.is_unresolved() {
                    self.end(Completion::Stuck { block: site.0 });
                    return;
                }
                let Some(unwound) = self.unwind_call(&mut path, site) else {
                    return;
                };
                site = unwound;
                // An exception the run threw is dispatched as the runtime's
                // personality chooses: each entry it may enter, handed the
                // selector of the clause it chose.
                let personality = path
                    .state
                    .exceptions()
                    .in_flight()
                    .and_then(|thrown| dispatch.chosen(thrown));
                let chooses = personality.is_some();
                if let Some((chosen, onward)) = personality {
                    for chosen in chosen {
                        self.paths = self.paths.saturating_add(1);
                        if !self.check_limit(Budget::ExecutionPaths, self.paths) {
                            return;
                        }
                        let mut fork = path.clone();
                        fork.trace.raise(body, site, chosen.entry);
                        fork.state.exceptions_mut().deliver(Some(chosen.selector));
                        let raised = Entry::Raise {
                            block: site.0,
                            at: site.1,
                        };
                        if let Some(fork) = self.transfer(fork, raised, chosen.entry, true) {
                            self.push(fork);
                        }
                    }
                    if !onward {
                        self.abandon(path);
                        return;
                    }
                }
                for &(entry, resumes) in dispatch.entries().iter().filter(|_| !chooses) {
                    self.paths = self.paths.saturating_add(1);
                    if !self.check_limit(Budget::ExecutionPaths, self.paths) {
                        return;
                    }
                    let mut fork = path.clone();
                    fork.trace.raise(body, site, entry);
                    // No personality chose this entry: it is handed no
                    // selector.
                    fork.state.exceptions_mut().deliver(None);
                    if resumes {
                        // The handler ran elsewhere, on this activation's frame.
                        if self.admit_unevaluated_call(&fork, site).is_none() {
                            return;
                        }
                        fork.trace.taint();
                        let memory = fork.state.memory_mut();
                        memory.escape_stack();
                        memory.invalidate_lifetimes();
                        memory.clobber();
                    }
                    let raised = Entry::Raise {
                        block: site.0,
                        at: site.1,
                    };
                    if let Some(fork) = self.transfer(fork, raised, entry, true) {
                        self.push(fork);
                    }
                }
            }
            if path.state.depth() <= 1 {
                self.abandon(path);
                return;
            }
            if !self.retire_source(&mut path) {
                self.end(Completion::UnstableCode);
                return;
            }
            if self.retire_loops(&mut path).is_none() {
                return;
            }
            if let Some(uses) = &mut path.trace.uses {
                uses.returned(path.state.depth());
            }
            if path.state.return_from().is_none() {
                return;
            }
            let Some(caller) = path.state.frame() else {
                return;
            };
            site = (caller.position().block, caller.position().next);
        }
    }

    /// Completes the call at `site` of `path`'s running activation as the
    /// unwinder leaves it where the call raises into a handler of this
    /// activation, and returns the site the raise then stands at.
    ///
    /// The unwinder restores what the call's convention preserves to its
    /// value at the call, the stack pointer to where the call leaves it, and
    /// hands the handler the rest as the callee left it: what a return
    /// leaves, without the value returned. A source body states that as the
    /// call's outputs ([`SsaOp::CallClobber`]), read from the machine cells;
    /// the cells are therefore published again from the values the call took
    /// (whatever a callee or a model did to them since) and the outputs read
    /// from them, and the raise stands past them, so a handler reading them
    /// reads the restored state. The value the call returns stays undefined:
    /// a call that raises returns none. A raise that is no call's, or a call
    /// with no outputs, stands where it is.
    ///
    /// `None` where the outputs cannot be read, which ends the run.
    fn unwind_call(&mut self, path: &mut Path<T>, site: (usize, usize)) -> Option<(usize, usize)> {
        let body = path.state.frame()?.body();
        let source = self.body(body)?;
        let instructions = source.block(site.0)?.instructions();
        if instructions
            .get(site.1)
            .and_then(|instruction| instruction.op().as_call())
            .is_none()
        {
            return Some(site);
        }
        let clobber = site.1.checked_add(1)?;
        let Some(SsaOp::CallClobber { outputs, machine }) =
            instructions.get(clobber).map(SsaInstruction::op)
        else {
            return Some(site);
        };
        let layouts = self.layouts.get(body);
        if let Err(reason) = path.state.define_call_outputs(
            outputs,
            machine.as_deref(),
            |output| layouts.and_then(|layouts| layouts.get(output)),
            self.context.machine().ptr_size,
        ) {
            self.end(Completion::Memory {
                block: site.0,
                reason,
            });
            return None;
        }
        Some((site.0, clobber))
    }

    /// Spoils unfinished root loop proofs when their activation terminates.
    fn retire_loops(&mut self, path: &mut Path<T>) -> Option<()> {
        let activation = path.state.frame()?.activation();
        while let Some(visit) = path.trace.pop_loop(|visit| visit.activation == activation) {
            self.close_visit(path, &visit, None);
        }
        Some(())
    }

    /// Ends `path` at a return of its running activation, leaving its caller
    /// `exit`, where one is given: the machine registers as it returned them, where it
    /// states them: the caller's registers its code does not name hold them,
    /// and the registers its call destroyed take them.
    fn finish(
        &mut self,
        mut path: Path<T>,
        returned: Option<Value<T>>,
        source_live: bool,
        exit: Option<BTreeMap<CallInputKey, HeldRegister<T>>>,
    ) {
        if source_live && !self.retire_source(&mut path) {
            self.end(Completion::UnstableCode);
            return;
        }
        if let Some(uses) = &mut path.trace.uses {
            uses.returned(path.state.depth());
        }
        if let Some(frame) = path.state.frame()
            && let Some(scope) = frame.observation()
        {
            // A record's own writes are those of the activation that entered
            // it and the ones it ran. A record retired by a deeper tail chain
            // takes the first retired one's activation, which is no earlier
            // than its own: it can omit writes it made before its tail call,
            // never claim its caller's.
            if let Some(record) = scope.body.and_then(|slot| self.call_records.get_mut(slot)) {
                record
                    .observed
                    .exit(path.state.memory(), returned.clone(), frame.activation());
            }
            let mut pending = scope.returns;
            let mut binding = scope.returns_binding;
            let mut inherited = returned.clone();
            while let Some(slot) = pending {
                if self.call_records.get(slot).is_none() {
                    break;
                }
                inherited = self.bind_returned(binding, inherited);
                let Some(record) = self.call_records.get_mut(slot) else {
                    break;
                };
                record
                    .observed
                    .exit(path.state.memory(), inherited.clone(), scope.returns_since);
                pending = record.returns;
                binding = record.returns_binding;
            }
        }
        let returned = match path.state.frame() {
            Some(frame) => self.bind_returned(frame.return_binding(), returned),
            None => returned,
        };
        if path.state.depth() == 1 {
            // The bytes the run wrote behind a returned pointer, as a caller
            // receives them ([`StaticFacts::returned_bytes`]).
            let bytes = returned
                .as_ref()
                .and_then(|value| {
                    Access::new(self.context).written_prefix(
                        path.state.memory(),
                        value,
                        |address| path.state.proven_object(address),
                    )
                })
                .map(|(address, bytes, bound)| ArgumentBytes {
                    argument: 0,
                    address,
                    written: path.state.memory().written(address, bytes.len()),
                    bytes,
                    bound,
                    allocated_at: path.state.memory().allocation_site(address.root()),
                });
            self.observed.returned_pointer(bytes);
            // Every write of a root's run is the root's own.
            self.observed.exit(path.state.memory(), returned, 0);
            self.abandon(path);
            return;
        }
        if self.retire_loops(&mut path).is_none() {
            return;
        }
        if path.state.return_from().is_none() {
            return;
        }
        let handed = exit.is_some();
        if let Some(exit) = exit {
            let pointer = self.context.machine().ptr_size;
            for (key, held) in &exit {
                if let Some(layout) = held.layout {
                    path.state
                        .publish_machine(*key, held.value.clone(), layout, pointer);
                }
            }
            if let Some(frame) = path.state.frame_mut() {
                frame.set_registers(exit);
            }
        }
        let Some(caller) = path.state.frame() else {
            return;
        };
        let site = (caller.position().block, caller.position().next);
        let caller_body = self.body(caller.body());
        let dest = caller_body.as_ref().and_then(|ssa| {
            ssa.block(site.0)
                .and_then(|block| block.instruction(site.1))
                .and_then(|instruction| instruction.op().as_call()?.dest)
        });
        // The call's result is what the callee left in the register its
        // convention returns one in, where the callee states it.
        let returned = match (caller_body.as_ref(), dest) {
            (Some(ssa), Some(dest)) if handed => {
                let key = self.host.call_result_register(CallSite {
                    caller: ssa,
                    caller_method: self.methods.get(caller.body()).and_then(Option::as_ref),
                    block: site.0,
                    instruction: site.1,
                });
                match (
                    key.and_then(|key| caller.registers().get(&key)),
                    ssa.variable(dest),
                ) {
                    (Some(held), Some(variable)) => Some(register::read(
                        &held.value,
                        held.layout,
                        variable.var_type(),
                        self.context.machine(),
                    )),
                    _ => returned,
                }
            }
            _ => returned,
        };
        if let Some(path) = self.resumed(path, site, dest, returned) {
            self.push(path);
        }
    }

    /// Applies a call's result contract to what the callee returned
    /// ([`ReturnBinding::apply`]), reading an address result's bits through
    /// the image first where the contract takes its low bits.
    fn bind_returned(
        &self,
        binding: ReturnBinding,
        returned: Option<Value<T>>,
    ) -> Option<Value<T>> {
        let pointer = self.context.machine().ptr_size;
        let returned = match (binding, returned) {
            (ReturnBinding::Low { .. }, Some(Value::Addr(address))) => Some(
                Access::new(self.context)
                    .physical_address_bits(address)
                    .and_then(|bits| ConstValue::of_lane(pointer.bits(), u128::from(bits)))
                    .map_or(Value::Unknown, Value::Const),
            ),
            (_, returned) => returned,
        };
        binding.apply(returned, pointer)
    }

    /// Retires exactly the running activation while preserving suspended callers.
    fn retire_source(&self, path: &mut Path<T>) -> bool {
        let Some(index) = path.state.frame().map(Frame::body) else {
            return false;
        };
        let Some(body) = self.body(index) else {
            return false;
        };
        let method = self.methods.get(index).and_then(Option::as_ref);
        self.host.retire_code_source(
            &body,
            method,
            path.state.memory_mut().instruction_sources_mut(),
        )
    }

    /// Spoils all loop visits still open when the path terminates.
    fn abandon(&mut self, mut path: Path<T>) {
        self.instruction_sources
            .merge(path.state.memory().instruction_sources());
        while let Some(visit) = path.trace.pop_loop(|_| true) {
            self.close_visit(&path, &visit, None);
        }
    }
}
