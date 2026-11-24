//! Everything one path of a static execution knows: its activations, its
//! memory, and the counters that keep the names it mints distinct.
//!
//! Two states are compared on what they know - their activations and their
//! memory - and never on their counters: a counter only has to keep names
//! apart within one path, and two paths that know the same things are the same
//! state whatever they have numbered.
//!
//! A state is copied at each block a path reaches once it has alternatives,
//! and below the running activation nothing changes while it runs. The
//! activations are therefore held shared: a copy shares every frame, and the
//! first write to the running one after a copy copies that frame alone, whose
//! values are themselves held in shared pages.

use std::{collections::BTreeMap, sync::Arc};

use crate::{
    analysis::execution::{
        StackCongruence,
        alignment::StackCongruences,
        bank::Bank,
        facts::StorageObject,
        frame::Frame,
        marks::Marks,
        memory::{Memory, MemoryRefusal},
        provenance::Origin,
        refine::Refined,
        thrown::Exceptions,
        value::{Address, Root, Value},
    },
    ir::{CallInputKey, CallOutput, function::SsaFunction, variable::SsaVarId},
    pointer::PointerSize,
    target::{Target, TypeLayout},
};

/// What one path knows.
#[derive(Debug, Clone)]
pub struct State<T: Target> {
    /// The activations, outermost first, each shared with the copies of this
    /// state until one of them writes it.
    frames: Vec<Arc<Frame<T>>>,
    /// The memory.
    memory: Memory,
    /// Physical state retained independently of source SSA namespaces.
    machine: Option<Bank<T>>,
    /// The next run number a named definition takes.
    next_run: u64,
    /// The next heap block's number.
    next_heap: u64,
    /// The next activation's number.
    next_activation: u64,
    /// Whether the calling environment or an exclusive load may hold a reservation.
    ///
    /// Granule size and asynchronous loss are not attested, so a possible
    /// reservation never establishes successful store-conditional execution.
    reservation_possible: bool,
    /// Whether an identity namespace has reached its representable end.
    identity_exhausted: bool,
    /// The C++ exceptions the run has thrown and not yet finished handling.
    exceptions: Exceptions<T>,
    /// What the last return handed its caller beside its result, under the
    /// keys a call receives each by ([`SsaOp::Return`]'s `results`), until the
    /// call it returns to takes them ([`Self::take_handed`]).
    ///
    /// [`SsaOp::Return`]: crate::ir::SsaOp::Return
    handed: Vec<(CallInputKey, Value<T>)>,
}

impl<T: Target> PartialEq for State<T> {
    fn eq(&self, other: &Self) -> bool {
        self.frames == other.frames
            && self.memory == other.memory
            && self.machine == other.machine
            && self.reservation_possible == other.reservation_possible
            && self.identity_exhausted == other.identity_exhausted
            && self.exceptions == other.exceptions
            && self.handed == other.handed
    }
}

impl<T: Target> State<T> {
    /// A state entering `frame`, the outermost activation, with nothing in
    /// memory.
    pub fn new(frame: Frame<T>) -> Self {
        let next_activation = u64::from(frame.activation()).saturating_add(1);
        Self {
            frames: vec![Arc::new(frame)],
            memory: Memory::default(),
            machine: None,
            next_run: 1,
            next_heap: 0,
            next_activation,
            reservation_possible: true,
            identity_exhausted: false,
            exceptions: Exceptions::default(),
            handed: Vec::new(),
        }
    }

    /// Hands the activation's caller `values` beside its result, under the
    /// keys the call receives each by, replacing what an earlier return
    /// handed.
    pub fn hand_back(&mut self, values: Vec<(CallInputKey, Value<T>)>) {
        self.handed = values;
    }

    /// Takes what the last return handed its caller beside its result
    /// ([`Self::hand_back`]), leaving nothing: a call that reaches no return,
    /// an unanswered or a modelled one, finds none.
    pub fn take_handed(&mut self) -> Vec<(CallInputKey, Value<T>)> {
        std::mem::take(&mut self.handed)
    }

    /// Returns whether execution can no longer mint distinct identities.
    pub const fn identity_exhausted(&self) -> bool {
        self.identity_exhausted
    }

    /// The innermost activation, the one running.
    pub fn frame(&self) -> Option<&Frame<T>> {
        self.frames.last().map(Arc::as_ref)
    }

    /// The innermost activation, to run it: copied first when a copy of this
    /// state shares it.
    pub fn frame_mut(&mut self) -> Option<&mut Frame<T>> {
        self.frames.last_mut().map(Arc::make_mut)
    }

    /// The activations, outermost first.
    pub fn frames(&self) -> &[Arc<Frame<T>>] {
        &self.frames
    }

    /// Returns the unique live activation that backs this entire stack span.
    pub fn stack_owner(&self, address: Address, len: usize) -> Option<u32> {
        let mut matches = self
            .frames
            .iter()
            .filter(|frame| frame.covers_stack(address, len));
        let activation = matches.next()?.activation();
        matches.next().is_none().then_some(activation)
    }

    /// The frame of the one live activation whose promised writable range
    /// holds `address` ([`Frame::frame_object`]); `None` where none or several
    /// do.
    pub fn frame_object(&self, address: Address) -> Option<StorageObject> {
        let activation = self.stack_owner(address, 1)?;
        self.frames
            .iter()
            .find(|frame| frame.activation() == activation)?
            .frame_object()
    }

    /// The object `address` lies in, where its extent is proven: the one the
    /// host's table states with an exact extent
    /// ([`Memory::stated_object`]), or else the frame of the live activation
    /// holding it ([`Self::frame_object`]). A recovered extent proves nothing.
    pub fn proven_object(&self, address: Address) -> Option<StorageObject> {
        self.memory
            .stated_object(address)
            .filter(|object| object.source.exact())
            .or_else(|| self.frame_object(address))
    }

    /// How many activations are open.
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// What `value` holds in the running activation.
    pub fn value(&self, value: SsaVarId) -> Value<T> {
        self.frame()
            .map_or(Value::Unknown, |frame| frame.value(value))
    }

    /// Returns whether a reservation may exist, without assuming its granule or success.
    pub const fn reservation_possible(&self) -> bool {
        self.reservation_possible
    }

    /// Updates the thread's reservation abstraction after exclusive or unknown effects.
    pub fn set_reservation_possible(&mut self, possible: bool) {
        self.reservation_possible = possible;
    }

    /// The memory.
    pub const fn memory(&self) -> &Memory {
        &self.memory
    }

    /// The memory, to write it.
    pub const fn memory_mut(&mut self) -> &mut Memory {
        &mut self.memory
    }

    /// Returns the number of retained physical locations.
    fn machine_len(&self) -> usize {
        self.machine.as_ref().map_or(0, Bank::len)
    }

    /// Bounds storage for the retained cells and the proposed additions.
    pub fn machine_storage_bytes(&self, additional: usize) -> usize {
        self.machine_len()
            .saturating_add(additional)
            .saturating_mul(Bank::<T>::entry_bytes())
    }

    /// Returns whether a physical location already owns bank storage.
    pub fn has_machine(&self, key: CallInputKey) -> bool {
        self.machine.as_ref().is_some_and(|bank| bank.contains(key))
    }

    /// Publishes one actual source value without modifying any existing SSA definition.
    pub fn publish_machine(
        &mut self,
        key: CallInputKey,
        value: Value<T>,
        layout: TypeLayout,
        pointer: PointerSize,
    ) {
        if self.machine.is_none() && value == Value::Unknown {
            return;
        }
        self.machine
            .get_or_insert_with(Bank::default)
            .publish(key, value, layout, pointer);
        if self.machine.as_ref().is_some_and(|bank| bank.len() == 0) {
            self.machine = None;
        }
    }

    /// Returns a physical value only under the same representation, in the reader's type.
    pub fn machine_value(
        &self,
        key: CallInputKey,
        layout: TypeLayout,
        pointer: PointerSize,
    ) -> Option<Value<T>> {
        self.machine.as_ref()?.get(key, layout, pointer)
    }

    /// Forgets physical values when the selected implementation supplies no machine contract.
    pub fn clear_machine(&mut self) {
        self.machine = None;
    }

    /// Returns an unused definition identity, refusing namespace exhaustion.
    pub fn fresh_run(&mut self) -> Option<u32> {
        let Ok(run) = u32::try_from(self.next_run) else {
            self.identity_exhausted = true;
            return None;
        };
        self.next_run = self.next_run.saturating_add(1);
        Some(run)
    }

    /// Returns an unused heap identity, refusing namespace exhaustion.
    pub fn fresh_heap(&mut self) -> Option<Address> {
        let Ok(block) = u32::try_from(self.next_heap) else {
            self.identity_exhausted = true;
            return None;
        };
        self.next_heap = self.next_heap.saturating_add(1);
        Some(Address::of_root(Root::Heap(block)))
    }

    /// Returns an unused activation identity, refusing namespace exhaustion.
    pub fn fresh_activation(&mut self) -> Option<u32> {
        let Ok(activation) = u32::try_from(self.next_activation) else {
            self.identity_exhausted = true;
            return None;
        };
        self.next_activation = self.next_activation.saturating_add(1);
        Some(activation)
    }

    /// Defines a value after admitting its entry and fresh identity.
    pub fn define(&mut self, value: SsaVarId, held: Value<T>) -> bool {
        let Some(run) = self.fresh_run() else {
            return false;
        };
        if let Some(frame) = self.frame_mut() {
            frame.define(value, held, run);
            return true;
        }
        true
    }

    /// Defines a call's outputs (an [`SsaOp::CallClobber`]'s) from the
    /// physical cells `machine` binds them to, each read in its own layout
    /// (`layout_of`); an output no cell binds, or one the bank holds in no
    /// matching representation, is unknown.
    ///
    /// # Errors
    ///
    /// [`MemoryRefusal::MachineState`] where a binding names no output or
    /// binds one twice.
    ///
    /// [`SsaOp::CallClobber`]: crate::ir::SsaOp::CallClobber
    pub fn define_call_outputs(
        &mut self,
        outputs: &[SsaVarId],
        machine: Option<&[CallOutput]>,
        layout_of: impl Fn(SsaVarId) -> Option<TypeLayout>,
        pointer: PointerSize,
    ) -> Result<(), MemoryRefusal> {
        let mut bound = BTreeMap::new();
        for output in machine.into_iter().flatten() {
            let index = usize::try_from(output.index).map_err(|_| MemoryRefusal::MachineState)?;
            let Some(variable) = outputs.get(index) else {
                return Err(MemoryRefusal::MachineState);
            };
            if bound.insert(*variable, output.key).is_some() {
                return Err(MemoryRefusal::MachineState);
            }
        }
        for output in outputs {
            let value = bound
                .get(output)
                .and_then(|key| {
                    layout_of(*output).and_then(|layout| self.machine_value(*key, layout, pointer))
                })
                .unwrap_or(Value::Unknown);
            self.define(*output, value);
        }
        Ok(())
    }

    /// The never-written bits `value` carries in the running activation.
    pub fn marks(&self, value: SsaVarId) -> Option<&Marks> {
        self.frame().and_then(|frame| frame.marks(value))
    }

    /// States the never-written bits `value` carries in the running
    /// activation, `None` for none.
    pub fn set_marks(&mut self, value: SsaVarId, marks: Option<Marks>) {
        if (marks.is_some() || self.marks(value).is_some())
            && let Some(frame) = self.frame_mut()
        {
            frame.set_marks(value, marks);
        }
    }

    /// Where the pointer `value` holds in the running activation was formed,
    /// where it was formed anywhere.
    pub fn formed(&self, value: SsaVarId) -> Option<Origin> {
        self.frame()?.formed(value)
    }

    /// States where the pointer `value` holds in the running activation was
    /// formed, `None` for nowhere.
    pub fn set_formed(&mut self, value: SsaVarId, object: Option<Origin>) {
        if let Some(frame) = self.frame_mut() {
            frame.set_formed(value, object);
        }
    }

    /// States what the conditions this path took establish of `value` in
    /// the running activation, `None` for nothing.
    pub fn refine(&mut self, value: SsaVarId, refined: Option<Refined<T>>) {
        if let Some(frame) = self.frame_mut() {
            frame.refine(value, refined);
        }
    }

    /// Opens `frame`, a callee executed in place, as the running activation.
    pub fn call(&mut self, frame: Frame<T>) {
        self.frames.push(Arc::new(frame));
    }

    /// Replaces a retired caller with its admitted tail callee at the same depth.
    ///
    /// The explorer closes the caller's loop visits and releases its stack before
    /// this transfer. The replacement retains the enclosing caller's continuation
    /// without allocating another frame-vector entry. Returns `None` without a
    /// running activation with the new frame's bound input values.
    pub fn tail_call(&mut self, frame: Frame<T>) -> Option<()> {
        *self.frames.last_mut()? = Arc::new(frame);
        Some(())
    }

    /// Changes source bodies at the same physical depth without releasing any storage.
    pub fn continue_source(
        &mut self,
        ssa: &SsaFunction<T>,
        body: usize,
        stack: Address,
        bound: Vec<(SsaVarId, Value<T>)>,
    ) -> Option<()> {
        let run = self.fresh_run()?;
        self.frame_mut()?
            .continue_source(ssa, body, stack, bound, run);
        Some(())
    }

    /// Releases the running activation's storage while retaining values for ABI binding.
    ///
    /// Detached storage belongs only to its activation. An ABI-shared stack loses
    /// only the bytes below this frame's entry, preserving the enclosing caller's
    /// storage. Policy refusal returns `None` and prevents publication of the path.
    pub fn release_stack(&mut self) -> Option<()> {
        let frame = self.frames.last()?;
        if frame.stack_root() == Root::Stack {
            self.memory.release_below(frame.release_entry_stack());
        } else {
            self.memory.release_activation(
                frame.stack_root(),
                frame.release_entry_stack(),
                frame.activation(),
            );
        }
        Some(())
    }

    /// Closes the running activation after releasing its stack contents.
    ///
    /// Returns the closed frame on success. Policy refusal returns `None` with
    /// the frame still suspended; the explorer discards that path's observations.
    pub fn return_from(&mut self) -> Option<Frame<T>> {
        self.release_stack()?;
        self.frames.pop().map(Arc::unwrap_or_clone)
    }

    /// The C++ exceptions the run has thrown and not yet finished handling.
    #[must_use]
    pub const fn exceptions(&self) -> &Exceptions<T> {
        &self.exceptions
    }

    /// The C++ exceptions, to throw, catch and finish with one.
    pub const fn exceptions_mut(&mut self) -> &mut Exceptions<T> {
        &mut self.exceptions
    }

    /// Returns what both of two states know, when every activation of one
    /// stands where the other's does; `None` when they do not.
    ///
    /// # Arguments
    ///
    /// * `other` - The state on another path.
    pub fn join(&self, other: &Self) -> Option<Self> {
        // Paths handling different exceptions stay apart: a handler's
        // `__cxa_end_catch` acts on the exception its own path caught. So do
        // paths a return handed different values to, which the call it
        // returns to has yet to take.
        if self.frames.len() != other.frames.len()
            || !self.memory.same_instruction_sources(&other.memory)
            || self.exceptions != other.exceptions
            || self.handed != other.handed
        {
            return None;
        }

        let mut frames = Vec::with_capacity(self.frames.len());
        for (left, right) in self.frames.iter().zip(&other.frames) {
            frames.push(Arc::new(left.join(right)?));
        }
        let memory = self.memory.join(&other.memory);

        Some(Self {
            frames,
            memory,
            machine: self
                .machine
                .as_ref()
                .zip(other.machine.as_ref())
                .map(|(left, right)| left.join(right)),
            next_run: self.next_run.max(other.next_run),
            next_heap: self.next_heap.max(other.next_heap),
            next_activation: self.next_activation.max(other.next_activation),
            reservation_possible: self.reservation_possible || other.reservation_possible,
            identity_exhausted: self.identity_exhausted || other.identity_exhausted,
            exceptions: self.exceptions.clone(),
            handed: self.handed.clone(),
        })
    }

    /// Proves a nonempty stack span lies wholly in a live activation's frame:
    /// the extent the host attests or the activation reserved itself, or the
    /// frame it reserved below a realigned base. For a realigned address,
    /// otherwise every place the realigned base may lie: the window's lowest
    /// through its highest, in the frame it was realigned from.
    ///
    /// The function's own frame is storage the IR already treats as never
    /// faulting ([`VariableRole::EntryStackPointer`]): the memory pass deletes
    /// a store there nothing reads, and a load nothing uses. So an access in
    /// it needs no further alignment proof than the frame's.
    ///
    /// [`VariableRole::EntryStackPointer`]: crate::ir::variable::VariableRole::EntryStackPointer
    #[must_use]
    pub fn frame_backed(&self, address: Address, len: usize) -> bool {
        match address.root() {
            Root::Stack | Root::StackFrame(_) => self
                .frames()
                .iter()
                .any(|frame| frame.covers_stack(address, len)),
            Root::StackAligned { .. } => {
                if self
                    .frames()
                    .iter()
                    .any(|frame| frame.covers_realigned(address, len))
                {
                    return true;
                }
                let Some((frame, window)) = address.root().realigned_window() else {
                    return false;
                };
                let spread = window.end().saturating_sub(*window.start());
                let (Some(low), Some(len)) = (
                    window.start().checked_add(address.offset()),
                    usize::try_from(spread)
                        .ok()
                        .and_then(|spread| len.checked_add(spread)),
                ) else {
                    return false;
                };
                self.frames().iter().any(|stack| {
                    stack.covers_stack(Address::new(frame, low, address.pointer_size()), len)
                })
            }
            Root::Symbol
            | Root::Number
            | Root::Heap(_)
            | Root::Opaque { .. }
            | Root::StackDerived { .. } => false,
        }
    }
}

impl<T: Target> StackCongruences for State<T> {
    /// Borrows live root evidence without changing its physical origin or precision.
    fn stack_congruence(&self, root: Root) -> Option<StackCongruence> {
        self.frames
            .iter()
            .rev()
            .find_map(|frame| frame.stack_congruence(root))
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;
    use crate::{
        ir::{function::SsaFunction, value::ConstValue},
        pointer::PointerSize,
        testing::MockTarget,
    };

    /// A function with no body worth running: the frames here only carry
    /// values.
    fn empty() -> SsaFunction<MockTarget> {
        SsaFunction::new(0, 0)
    }

    fn stack(offset: i64) -> Address {
        Address::new(Root::Stack, offset, PointerSize::Bit64)
    }

    /// Every name a path mints is new.
    #[test]
    fn fresh_names_are_distinct() {
        let ssa = empty();
        let mut state = State::new(Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64));
        assert_ne!(state.fresh_run(), state.fresh_run());
        assert_ne!(state.fresh_heap(), state.fresh_heap());
        assert_ne!(state.fresh_activation(), Some(0));
    }

    /// A returned callee's frame is dead; its caller's is not.
    #[test]
    fn returning_releases_the_callee_frame() {
        let ssa = empty();
        let mut state = State::new(Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64));
        state.memory_mut().store_bytes(stack(-8), &[1]);
        let activation = state.fresh_activation().unwrap_or_else(|| process::abort());
        state.call(Frame::enter(
            &ssa,
            0,
            activation,
            -16,
            vec![],
            PointerSize::Bit64,
        ));
        state.memory_mut().store_bytes(stack(-24), &[2]);
        assert_eq!(state.depth(), 2);
        state.return_from();
        assert_eq!(state.depth(), 1);
        assert_eq!(state.memory().known_run(stack(-8), 1, None), vec![1]);
        assert!(state.memory().known_run(stack(-24), 1, None).is_empty());
    }

    /// Two states are the same when they know the same things, whatever they
    /// have numbered; they join only at one depth.
    #[test]
    fn sameness_ignores_counters_and_joins_need_one_depth() {
        let ssa = empty();
        let base = State::new(Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64));
        let mut counted = base.clone();
        counted.fresh_run();
        assert!(base.eq(&counted));
        let mut left = base.clone();
        let mut right = base.clone();
        left.memory_mut().store_bytes(stack(-2), &[1, 2]);
        right.memory_mut().store_bytes(stack(-2), &[1, 3]);
        let joined = left.join(&right).unwrap_or_else(|| process::abort());
        assert_eq!(joined.memory().known_run(stack(-2), 2, None), vec![1]);
        let activation = right.fresh_activation().unwrap_or_else(|| process::abort());
        right.call(Frame::enter(
            &ssa,
            0,
            activation,
            -8,
            vec![],
            PointerSize::Bit64,
        ));
        assert!(left.join(&right).is_none());
    }
    /// Every identity namespace stops at its last representable value instead of reusing it.
    #[test]
    fn identity_namespaces_refuse_wrap() {
        for namespace in 0..3 {
            let ssa = empty();
            let mut state = State::new(Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64));
            let start = u64::from(u32::MAX).saturating_sub(1);
            match namespace {
                0 => state.next_run = start,
                1 => state.next_heap = start,
                _ => state.next_activation = start,
            }

            let mut next = || match namespace {
                0 => state.fresh_run().map(u64::from),
                1 => state.fresh_heap().and_then(|address| match address.root() {
                    Root::Heap(id) => Some(u64::from(id)),
                    _ => None,
                }),
                _ => state.fresh_activation().map(u64::from),
            };
            assert_eq!(next(), Some(start));
            assert_eq!(next(), Some(u64::from(u32::MAX)));
            assert_eq!(next(), None);
            assert_eq!(next(), None);
            assert!(state.identity_exhausted());
        }
    }

    /// State copies walk empty activations too and preserve every naming counter.
    #[test]
    fn state_copies_and_agreement_cooperate_across_empty_frames() {
        let ssa = empty();
        let mut state = State::new(Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64));
        for activation in 1..1024 {
            state.call(Frame::enter(
                &ssa,
                0,
                activation,
                0,
                vec![],
                PointerSize::Bit64,
            ));
        }
        state.next_run = 19;
        state.next_heap = 23;
        state.next_activation = 1024;
        let copied = state.clone();
        assert_eq!(
            (copied.next_run, copied.next_heap, copied.next_activation),
            (19, 23, 1024)
        );
        assert!(state.eq(&copied));
    }

    /// The state copy reaches memory, and its comparison ignores counters, not bytes.
    #[test]
    fn state_copy_and_agreement_include_memory_without_counter_aliasing() {
        let ssa = empty();
        let mut state = State::new(Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64));
        let bytes: Vec<_> = (0..4096).map(|index| index as u8).collect();
        state.memory_mut().store_bytes(stack(-4096), &bytes);
        let copied = state.clone();
        assert_eq!(copied.memory().known_run(stack(-4096), 4096, None), bytes);

        let mut independent = State::new(Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64));
        independent.memory_mut().store_bytes(stack(-4096), &bytes);
        let mut changed = copied;
        changed.next_run = 100;
        changed.next_heap = 100;
        changed.next_activation = 100;
        assert!(state.eq(&changed));
        changed.memory_mut().store_bytes(stack(-1), &[0]);
        assert!(!(state.eq(&changed)));
    }

    /// A copy of a state shares every activation; running the innermost one
    /// copies it alone, and its callers stay shared. The original keeps what
    /// its innermost activation held.
    #[test]
    fn a_copy_shares_its_callers_and_copies_the_frame_it_runs() {
        let ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut state = State::new(Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64));
        for activation in 1..8 {
            state.call(Frame::enter(
                &ssa,
                0,
                activation,
                -16,
                vec![],
                PointerSize::Bit64,
            ));
        }
        let value = SsaVarId::from_index(0);
        assert!(state.define(value, Value::Const(ConstValue::I32(1))));
        let mut copy = state.clone();
        assert!(copy.define(value, Value::Const(ConstValue::I32(2))));

        let (callers, running) = state.frames().split_at(7);
        let (copied_callers, copied_running) = copy.frames().split_at(7);
        assert!(
            callers
                .iter()
                .zip(copied_callers)
                .all(|(left, right)| Arc::ptr_eq(left, right))
        );
        assert!(!Arc::ptr_eq(&running[0], &copied_running[0]));
        assert_eq!(state.value(value).integer(), Some(1));
        assert_eq!(copy.value(value).integer(), Some(2));
    }
}
