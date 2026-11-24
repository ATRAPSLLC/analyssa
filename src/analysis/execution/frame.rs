//! One activation of one function: its values and where it stands.
//!
//! A value read before anything in the activation defined it is one of the
//! function's inputs - a parameter, a register live on entry - and reads as an
//! [`Root::Opaque`] address named by the value itself. The activation's entry
//! stack pointer and the parameters a caller bound are defined when the frame
//! is made, so they read as what they are.
//!
//! A definition whose result the execution cannot compute is still *one*
//! value on any real run, so it is named rather than forgotten: an opaque root
//! numbered by the run that defined it. A pointer an unmodelled call returns
//! can then be stored through and read back. What is left unnamed -
//! [`Value::Unknown`] - is what two paths disagree on after a join.

use std::{collections::BTreeMap, ops::Range};

use crate::{
    analysis::execution::{
        RealignedExtent, ReturnBinding, StackCongruence,
        facts::{ExtentSource, ObjectKind, StorageObject},
        marks::Marks,
        provenance::Origin,
        refine::{Facts, Refined},
        value::{Address, Root, Value},
    },
    ir::{
        CallInputKey,
        function::SsaFunction,
        phi::{PhiNode, PhiOperand},
        variable::SsaVarId,
        varstore::SharedVarMap,
    },
    pointer::PointerSize,
    target::{Target, TypeLayout},
};

/// Where an activation stands: the block it is in, the block it came from,
/// and the next instruction it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    /// The block.
    pub block: usize,
    /// The block control arrived from, or `None` at the function's entry.
    pub previous: Option<usize>,
    /// The index of the next instruction to run in `block`.
    pub next: usize,
}

/// Where an activation whose code a write changed goes on in a lift of what
/// the code holds now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resumption {
    /// After the source instruction of the instruction at this `(block,
    /// index)`, whose write changed the code ahead of it in its block, or
    /// the call during which a callee did.
    After(usize, usize),
    /// Where a run of the host block `host` begins, control having entered
    /// `block` from `from` on its way there, by an edge of `from`'s
    /// terminator or not. The activation first runs the pure instructions
    /// `block` begins with: they may define the values the registers there
    /// are stated by.
    Entry {
        /// The block control left.
        from: usize,
        /// Whether it left by an edge of its terminator.
        edge: bool,
        /// The block control entered.
        block: usize,
        /// The host block the run goes on from.
        host: u64,
    },
}

/// How control enters a block: by a terminator of the block it leaves, or by a
/// raise inside a protected block that dispatches to the block's handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Entry {
    /// A terminator of this block transfers control.
    Edge(usize),
    /// An operation of the protected `block`, its instruction `at`, raises
    /// before it completes.
    Raise {
        /// The protected block.
        block: usize,
        /// The raising instruction.
        at: usize,
    },
}

impl Entry {
    /// The block control comes from.
    #[must_use]
    pub const fn from(self) -> usize {
        match self {
            Self::Edge(block) | Self::Raise { block, .. } => block,
        }
    }

    /// The value of `phi` this entry brings: the operand its edge names, or,
    /// for a raise, the one of the values a throw from the protected block
    /// may bring that the merged storage holds where it raises. That is the
    /// latest the block defines before the raising instruction, its own phi,
    /// or else the one value it is entered with. `None` where none is or
    /// several are: the value is not known.
    #[must_use]
    pub fn operand<T: Target>(self, ssa: &SsaFunction<T>, phi: &PhiNode) -> Option<SsaVarId> {
        let (block, at) = match self {
            Self::Edge(previous) => return phi.operand_from(previous).map(PhiOperand::value),
            Self::Raise { block, at } => (block, at),
        };
        let site = |value: SsaVarId| ssa.variable(value).map(|variable| variable.def_site());
        let candidates: Vec<SsaVarId> = phi
            .exceptional_operands()
            .filter(|operand| operand.predecessor() == block)
            .map(PhiOperand::value)
            .collect();
        // A definition at the block's entry stands before every instruction.
        let latest = candidates
            .iter()
            .filter_map(|value| {
                let site = site(*value)?;
                let order = match site.index() {
                    None => 0,
                    Some(index) => index.checked_add(1)?,
                };
                (site.block() == block && order <= at).then_some((order, *value))
            })
            .max_by_key(|(order, _)| *order);
        if let Some((_, value)) = latest {
            return Some(value);
        }
        match candidates
            .iter()
            .filter(|value| site(**value).is_some_and(|site| site.block() != block))
            .collect::<Vec<_>>()
            .as_slice()
        {
            [value] => Some(**value),
            _ => None,
        }
    }
}

/// Separate source-body observations and retired tail-return obligations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallScope {
    /// Record owning the active body's SSA definitions and call sites.
    pub body: Option<usize>,
    /// First retired caller whose normal return this activation implements.
    pub returns: Option<usize>,
    /// Identity-only mapping to the first retired observation, independent of its admission.
    pub returns_binding: ReturnBinding,
    /// The activation that entered the first retired observation. Its own
    /// writes at the shared return are those of that activation and later ones.
    pub returns_since: u32,
}

/// A machine register's value, with the layout it has where it was read.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldRegister<T: Target> {
    /// The value.
    pub value: Value<T>,
    /// Its layout, where known.
    pub layout: Option<TypeLayout>,
}

/// One activation of one function.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame<T: Target> {
    /// Which function, as the explorer numbers the bodies it runs.
    body: usize,
    /// The activation's number, unique within one execution.
    activation: u32,
    /// Fresh default definition run for inputs in this source namespace.
    entry_run: u32,
    /// Entry coordinate owning the physical stack lifetime across source continuations.
    release_entry_stack: i64,
    /// Selected call observation inheriting this activation's return continuation.
    observation: Option<CallScope>,
    /// Actual result mapping to the live continuation, including retired tail boundaries.
    return_binding: ReturnBinding,
    /// Actual or formal target required to resume this frame's caller.
    expected_return: Option<Value<T>>,
    /// What each machine register holds that the code this activation runs
    /// does not name, under the host's key for it: the caller's registers as
    /// the call left them, a callee's as it returned them, a lift's this one
    /// continued from as it left them. Absent where nothing is known.
    registers: BTreeMap<CallInputKey, HeldRegister<T>>,
    /// Target pointer width for named opaque coordinates.
    ptr_size: PointerSize,
    /// What each value holds. A value with no entry is not yet defined here.
    values: SharedVarMap<Value<T>>,
    /// The bits of each value that came from memory no write had reached;
    /// a value with no entry, or `None`, carries none.
    marks: SharedVarMap<Option<Marks>>,
    /// Where each value, a pointer, was formed ([`Origin`]); a value with no
    /// entry, or `None`, was formed nowhere.
    formed: SharedVarMap<Option<Origin>>,
    /// What the conditions this path took establish of values it defined
    /// before them: read in place of what the value holds, until the value is
    /// defined again. A value with no entry, or `None`, is not                 .filter(|refined| other.refinement(value) == Some(*refined)).
    refined: SharedVarMap<Option<Refined<T>>>,
    /// Whether unnamed incoming values may alias caller-supplied private storage.
    unbound_stack_aliases: bool,
    /// The stack offset the activation was entered with.
    entry_stack: i64,
    /// Object holding this activation's stack storage.
    stack_root: Root,
    /// Established physical low bits of this same stack root's base.
    stack_congruence: Option<StackCongruence>,
    /// Writable backing promised by the host relative to this activation's entry stack.
    stack_extent: Option<Range<i64>>,
    /// Writable backing promised by the host below a realigned base, with the
    /// masked stack pointer in this frame's own coordinates, which a source
    /// continuation keeps.
    realigned_extent: Option<RealignedExtent>,
    /// The number of the last code-changing write
    /// ([`InstructionSources::generation`](crate::analysis::execution::InstructionSources::generation))
    /// before the lift this activation runs was made: a later one may have
    /// changed code it has yet to run.
    code_generation: u64,
    /// Where the activation goes on in a lift of what the code holds now,
    /// once it has finished what it still runs of the lift it runs.
    resuming: Option<Resumption>,
    /// Where it stands.
    position: Position,
}

impl<T: Target> Frame<T> {
    /// Enters a frame on a shared or detached stack object.
    ///
    /// Missing host bindings remain named inputs in the new activation.
    pub fn enter_at(
        ssa: &SsaFunction<T>,
        body: usize,
        activation: u32,
        stack: Address,
        bound: Vec<(SsaVarId, Value<T>)>,
        ptr_size: PointerSize,
    ) -> Self {
        let values = SharedVarMap::new(ssa.var_id_bound());
        let mut frame = Self {
            body,
            activation,
            entry_run: 0,
            release_entry_stack: stack.offset(),
            observation: None,
            return_binding: ReturnBinding::Identity,
            expected_return: None,
            registers: BTreeMap::new(),
            ptr_size,
            values,
            marks: SharedVarMap::new(ssa.var_id_bound()),
            formed: SharedVarMap::new(ssa.var_id_bound()),
            refined: SharedVarMap::new(ssa.var_id_bound()),
            unbound_stack_aliases: false,
            entry_stack: stack.offset(),
            stack_root: stack.root(),
            stack_congruence: None,
            stack_extent: None,
            realigned_extent: None,
            code_generation: 0,
            resuming: None,
            position: Position {
                block: 0,
                previous: None,
                next: 0,
            },
        };
        if let Some(pointer) = ssa.entry_stack_pointer() {
            frame.values.insert(pointer, Value::Addr(stack));
        }
        for (value, held) in bound {
            frame.values.insert(value, held);
        }
        frame
    }

    /// Retains the actual result contract independently of observation selection.
    pub const fn with_return_binding(mut self, binding: ReturnBinding) -> Self {
        self.return_binding = binding;
        self
    }

    /// Returns the result contract carried by this activation's live continuation.
    pub const fn return_binding(&self) -> ReturnBinding {
        self.return_binding
    }

    /// Retains established root-base evidence independently of the entry offset.
    pub const fn with_stack_congruence(mut self, congruence: Option<StackCongruence>) -> Self {
        self.stack_congruence = congruence;
        self
    }

    /// Returns known physical low bits only for this frame's existing stack root.
    pub fn stack_congruence(&self, root: Root) -> Option<StackCongruence> {
        (root == self.stack_root)
            .then_some(self.stack_congruence)
            .flatten()
    }

    /// Retains a host's nonwrapping writable range for this activation's lifetime.
    pub fn with_stack_extent(mut self, extent: Option<Range<i64>>) -> Self {
        self.stack_extent = extent.filter(|range| {
            range.start < range.end
                && [range.start, range.end.saturating_sub(1)]
                    .into_iter()
                    .all(|offset| {
                        self.entry_stack.checked_add(offset).is_some_and(|at| {
                            Address::new(self.stack_root, at, self.ptr_size).offset() == at
                        })
                    })
        });
        self
    }

    /// Retains a host's writable range below one realigned base for this
    /// activation's lifetime, its masked stack pointer moved from entry-relative
    /// into this frame's coordinates.
    pub fn with_realigned_extent(mut self, extent: Option<RealignedExtent>) -> Self {
        self.realigned_extent = extent.and_then(|extent| {
            (extent.range.start < extent.range.end).then_some(RealignedExtent {
                at: self.entry_stack.checked_add(extent.at)?,
                range: extent.range,
            })
        });
        self
    }

    /// Proves a nonempty access under a realigned base is wholly inside the
    /// writable range this activation reserved below it: the base is this
    /// frame's stack pointer realigned where the extent names.
    pub fn covers_realigned(&self, address: Address, len: usize) -> bool {
        let (Some(extent), Root::StackAligned { frame, at, .. }) =
            (&self.realigned_extent, address.root())
        else {
            return false;
        };
        frame.map_or(Root::Stack, Root::StackFrame) == self.stack_root
            && at == extent.at
            && address.pointer_size() == self.ptr_size
            && address.offset() >= extent.range.start
            && i64::try_from(len)
                .ok()
                .and_then(|len| address.offset().checked_add(len))
                .is_some_and(|end| end <= extent.range.end)
    }

    /// This activation's frame as an object of storage in its stack root: the
    /// range the host promises writable throughout it ([`Self::with_stack_extent`]),
    /// which nothing in the frame reaches past; `None` without one.
    pub fn frame_object(&self) -> Option<StorageObject> {
        let extent = self.stack_extent.as_ref()?;
        Some(StorageObject {
            start: self.entry_stack.checked_add(extent.start)?,
            bytes: u64::try_from(extent.end.checked_sub(extent.start)?).ok()?,
            kind: ObjectKind::Frame,
            source: ExtentSource::Attested,
        })
    }

    /// Proves a nonempty access is wholly inside this live activation's writable range.
    pub fn covers_stack(&self, address: Address, len: usize) -> bool {
        let Some(extent) = &self.stack_extent else {
            return false;
        };
        if address.root() != self.stack_root || address.pointer_size() != self.ptr_size {
            return false;
        }
        address
            .offset()
            .checked_sub(self.entry_stack)
            .is_some_and(|start| {
                start >= extent.start
                    && i64::try_from(len)
                        .ok()
                        .and_then(|len| start.checked_add(len))
                        .is_some_and(|end| end <= extent.end)
            })
    }

    /// Associates this continuation with one selected call's observation record.
    pub const fn with_observation(mut self, observation: Option<CallScope>) -> Self {
        self.observation = observation;
        self
    }

    /// Returns the selected call whose return continuation this frame carries.
    pub const fn observation(&self) -> Option<CallScope> {
        self.observation
    }

    /// Retains possible caller-stack aliases in inputs the host did not bind.
    ///
    /// Known bindings remain exact. Missing values retain their individual
    /// activation/value identities and relative offsets, without inventing an
    /// actual pointer. Definitions computed later use their own provenance.
    pub const fn with_stack_input_aliases(mut self, possible: bool) -> Self {
        self.unbound_stack_aliases = possible;
        self
    }

    /// Enters an isolated semantic fixture.
    #[cfg(test)]
    pub fn enter(
        ssa: &SsaFunction<T>,
        body: usize,
        activation: u32,
        entry_stack: i64,
        bound: Vec<(SsaVarId, Value<T>)>,
        ptr_size: PointerSize,
    ) -> Self {
        Self::enter_at(
            ssa,
            body,
            activation,
            Address::new(Root::Stack, entry_stack, ptr_size),
            bound,
            ptr_size,
        )
    }

    /// Returns the object holding this activation's stack.
    pub const fn stack_root(&self) -> Root {
        self.stack_root
    }

    /// Returns the stack coordinate released when the physical activation ends.
    pub const fn release_entry_stack(&self) -> i64 {
        self.release_entry_stack
    }

    /// What each machine register the running code does not name holds.
    #[must_use]
    pub const fn registers(&self) -> &BTreeMap<CallInputKey, HeldRegister<T>> {
        &self.registers
    }

    /// Records what each machine register the running code does not name
    /// holds.
    pub fn set_registers(&mut self, registers: BTreeMap<CallInputKey, HeldRegister<T>>) {
        self.registers = registers;
    }

    /// The number of the last code-changing write before the lift this
    /// activation runs was made.
    #[must_use]
    pub const fn code_generation(&self) -> u64 {
        self.code_generation
    }

    /// Records that the lift this activation runs is of the code as the
    /// code-changing writes up to `generation` left it.
    pub const fn set_code_generation(&mut self, generation: u64) {
        self.code_generation = generation;
    }

    /// Where this activation goes on in a lift of the code as it is now,
    /// once it has finished what it still runs of the lift it runs.
    #[must_use]
    pub const fn resuming(&self) -> Option<Resumption> {
        self.resuming
    }

    /// Records where this activation goes on in a lift of the code as it is
    /// now, or clears it.
    pub const fn set_resuming(&mut self, at: Option<Resumption>) {
        self.resuming = at;
    }

    /// Replaces only this activation's source namespace, retaining memory ownership and return.
    pub fn continue_source(
        &mut self,
        ssa: &SsaFunction<T>,
        body: usize,
        stack: Address,
        bound: Vec<(SsaVarId, Value<T>)>,
        entry_run: u32,
    ) {
        let previous_entry = self.entry_stack;
        self.entry_stack = stack.offset();
        self.stack_extent = self.stack_extent.take().and_then(|extent| {
            Some(
                extent
                    .start
                    .checked_add(previous_entry)?
                    .checked_sub(self.entry_stack)?
                    ..extent
                        .end
                        .checked_add(previous_entry)?
                        .checked_sub(self.entry_stack)?,
            )
        });
        self.body = body;
        self.entry_run = entry_run;
        self.resuming = None;
        // Missing machine state may hold a pointer the prefix formed into this
        // same private stack. Explicitly supplied values retain their own roots.
        self.unbound_stack_aliases = true;
        self.values = SharedVarMap::new(ssa.var_id_bound());
        self.marks = SharedVarMap::new(ssa.var_id_bound());
        self.formed = SharedVarMap::new(ssa.var_id_bound());
        self.refined = SharedVarMap::new(ssa.var_id_bound());
        if let Some(pointer) = ssa.entry_stack_pointer() {
            self.values.insert(pointer, Value::Addr(stack));
        }
        for (value, held) in bound {
            self.values.insert(value, held);
        }
        self.observation = self.observation.map(|scope| CallScope {
            body: None,
            returns: scope.body.or(scope.returns),
            returns_binding: if scope.body.is_some() {
                ReturnBinding::Identity
            } else {
                scope.returns_binding
            },
            returns_since: if scope.body.is_some() {
                self.activation
            } else {
                scope.returns_since
            },
        });
        self.position = Position {
            block: 0,
            previous: None,
            next: 0,
        };
    }

    /// Which function this is an activation of.
    pub const fn body(&self) -> usize {
        self.body
    }

    /// The activation's number.
    pub const fn activation(&self) -> u32 {
        self.activation
    }

    /// Returns the activation's entry coordinate independently of SSA value liveness.
    pub fn entry_stack_address(&self) -> Address {
        Address::new(self.stack_root, self.entry_stack, self.ptr_size)
    }

    /// Where the activation stands.
    pub const fn position(&self) -> Position {
        self.position
    }

    /// Moves the activation to `position`.
    pub const fn move_to(&mut self, position: Position) {
        self.position = position;
    }

    /// Returns whether both frames name the same target-width activation and position.
    fn same_activation(&self, other: &Self) -> bool {
        self.body == other.body
            && self.activation == other.activation
            && self.entry_run == other.entry_run
            && self.release_entry_stack == other.release_entry_stack
            && self.observation == other.observation
            && self.return_binding == other.return_binding
            && self.expected_return == other.expected_return
            && self.ptr_size == other.ptr_size
            && self.stack_root == other.stack_root
            && self.stack_extent == other.stack_extent
            && self.realigned_extent == other.realigned_extent
            && self.entry_stack == other.entry_stack
            && self.position == other.position
    }

    /// Borrows the expected physical return target across source replacements.
    pub fn expected_return(&self) -> Option<&Value<T>> {
        self.expected_return.as_ref()
    }

    /// Installs the obligation established before this physical activation runs.
    pub fn set_expected_return(&mut self, expected: Option<Value<T>>) {
        self.expected_return = expected;
    }

    /// Borrows one supplied value without scanning the body's variable table.
    pub fn held_value(&self, value: SsaVarId) -> Option<&Value<T>> {
        self.refinement(value)
            .map(Refined::value)
            .or_else(|| self.values.get(value))
    }

    /// What `value` holds: as the conditions this path took refine it, as
    /// defined, or the input it names.
    pub fn value(&self, value: SsaVarId) -> Value<T> {
        if let Some(refined) = self.refinement(value) {
            return refined.value().clone();
        }
        self.defined(value)
    }

    /// What `value` holds as defined, or the input it names, whatever the
    /// conditions this path took establish of it.
    fn defined(&self, value: SsaVarId) -> Value<T> {
        match self.values.get(value) {
            Some(held) => held.clone(),
            None => self.named(value, self.entry_run, self.unbound_stack_aliases),
        }
    }

    /// Defines `value` as `held`, naming it by `run` when nothing more is
    /// known about it.
    ///
    /// # Arguments
    ///
    /// * `value` - The value defined.
    /// * `held` - What it holds.
    /// * `run` - A number no other definition in this execution was given.
    pub fn define(&mut self, value: SsaVarId, held: Value<T>, run: u32) {
        let held = match held {
            Value::Unknown => self.named(value, run, false),
            Value::StackDerived => Value::Addr(Address::new(
                Root::StackDerived {
                    activation: self.activation,
                    value,
                    run,
                },
                0,
                self.ptr_size,
            )),
            other => other,
        };
        self.values.insert(value, held);
        // A new definition carries no marks until its operation states them,
        // and no condition taken before it says anything of it.
        self.set_marks(value, None);
        self.set_formed(value, None);
        self.refine(value, None);
    }

    /// Enters `block` by `entry`, defining every phi there from the operand
    /// that entry brings ([`Entry::operand`]), and unknown where it brings
    /// none.
    ///
    /// Every operand is read before any phi is written: the phis of a block
    /// take their values together, so one phi reading another at the same
    /// block reads its value from before the edge.
    pub fn arrive(&mut self, ssa: &SsaFunction<T>, block: usize, entry: Entry) {
        type Arrival<T> = (SsaVarId, Value<T>, Option<Marks>, Option<Origin>);
        let arrivals: Vec<Arrival<T>> = ssa
            .block(block)
            .map(|target| {
                target
                    .phi_nodes()
                    .iter()
                    .map(|phi| {
                        let operand = entry.operand(ssa, phi);
                        let held = operand.map_or(Value::Unknown, |operand| self.value(operand));
                        let marks = operand.and_then(|operand| self.marks(operand).cloned());
                        let formed = operand.and_then(|operand| self.formed(operand));
                        (phi.result(), held, marks, formed)
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (result, held, marks, formed) in arrivals {
            self.values.insert(result, held);
            self.set_marks(result, marks);
            self.set_formed(result, formed);
            self.refine(result, None);
        }
        self.position = Position {
            block,
            previous: Some(entry.from()),
            next: 0,
        };
    }

    /// Returns what both of two activations hold, when they stand at one
    /// place in one function; `None` when they do not.
    ///
    /// # Arguments
    ///
    /// * `other` - The same activation on another path.
    pub fn join(&self, other: &Self) -> Option<Self> {
        if !self.same_activation(other) {
            return None;
        }
        let values = self.values.empty_union(&other.values);
        let marks = self.marks.empty_union(&other.marks);
        let formed = self.formed.empty_union(&other.formed);
        let refined = self.refined.empty_union(&other.refined);
        let mut joined = Self {
            values,
            marks,
            formed,
            refined,
            unbound_stack_aliases: self.unbound_stack_aliases || other.unbound_stack_aliases,
            body: self.body,
            activation: self.activation,
            entry_run: self.entry_run,
            release_entry_stack: self.release_entry_stack,
            observation: self.observation,
            return_binding: self.return_binding,
            expected_return: self.expected_return.clone(),
            registers: self
                .registers
                .iter()
                .filter(|(key, held)| other.registers.get(key) == Some(*held))
                .map(|(key, held)| (*key, held.clone()))
                .collect(),
            ptr_size: self.ptr_size,
            entry_stack: self.entry_stack,
            stack_root: self.stack_root,
            stack_congruence: self
                .stack_congruence
                .zip(other.stack_congruence)
                .and_then(|(left, right)| left.meet(right)),
            stack_extent: self.stack_extent.clone(),
            realigned_extent: self.realigned_extent.clone(),
            // Made before the earlier of the two: a write after it may have
            // changed the code on the other path.
            code_generation: self.code_generation.min(other.code_generation),
            resuming: self.resuming.or(other.resuming),
            position: self.position,
        };
        // What each path defined joins as defined, so a value one path's
        // conditions refined keeps the identity both paths gave it.
        for source in [&self.values, &other.values] {
            for (value, _) in source.iter() {
                let held = self.defined(value).join(&other.defined(value));

                joined.values.insert(value, held);
            }
        }
        // A bit stays marked where both paths marked it.
        for (value, marks) in self.marks.iter() {
            let both = marks
                .as_ref()
                .zip(other.marks(value))
                .and_then(|(left, right)| left.join(right));
            joined.set_marks(value, both);
        }
        // A pointer stays formed into the object both paths formed it into.
        for (value, formed) in self.formed.iter() {
            let both = formed.filter(|object| other.formed(value) == Some(*object));
            joined.set_formed(value, both);
        }
        // A refinement stays where both paths established it.
        for (value, refined) in self.refined.iter() {
            let both = refined
                .as_ref()
                .filter(|refined| other.refinement(value) == Some(*refined))
                .cloned();
            joined.refine(value, both);
        }
        Some(joined)
    }

    /// What the conditions this path took establish of `value`, where they
    /// establish anything.
    pub fn refinement(&self, value: SsaVarId) -> Option<&Refined<T>> {
        self.refined.get(value).and_then(Option::as_ref)
    }

    /// States what the conditions this path took establish of `value`,
    /// `None` for nothing.
    pub fn refine(&mut self, value: SsaVarId, refined: Option<Refined<T>>) {
        if refined.is_some() || self.refinement(value).is_some() {
            self.refined.insert(value, refined);
        }
    }

    /// What the conditions this path took establish, as the facts of each
    /// value they refine, in the order of the values: two paths that agree
    /// here agree on everything their conditions established.
    pub fn refinements(&self) -> Vec<(SsaVarId, Facts)> {
        self.refined
            .iter()
            .filter_map(|(value, refined)| Some((value, refined.as_ref()?.facts().clone())))
            .collect()
    }

    /// Where the pointer `value` holds was formed, where it was formed
    /// anywhere.
    pub fn formed(&self, value: SsaVarId) -> Option<Origin> {
        self.formed.get(value).copied().flatten()
    }

    /// States where the pointer `value` holds was formed, `None` for
    /// nowhere.
    pub fn set_formed(&mut self, value: SsaVarId, object: Option<Origin>) {
        if object.is_some() || self.formed(value).is_some() {
            self.formed.insert(value, object);
        }
    }

    /// The never-written bits `value` carries, where it carries any.
    pub fn marks(&self, value: SsaVarId) -> Option<&Marks> {
        self.marks.get(value).and_then(Option::as_ref)
    }

    /// States the never-written bits `value` carries, `None` for none.
    pub fn set_marks(&mut self, value: SsaVarId, marks: Option<Marks>) {
        if marks.is_some() || self.marks(value).is_some() {
            self.marks.insert(value, marks);
        }
    }

    /// Names unknown bits while preserving their admitted private-stack provenance.
    fn named(&self, value: SsaVarId, run: u32, stack_aliases: bool) -> Value<T> {
        let root = if stack_aliases {
            Root::StackDerived {
                activation: self.activation,
                value,
                run,
            }
        } else {
            Root::Opaque {
                activation: self.activation,
                value,
                run,
            }
        };
        Value::Addr(Address::new(root, 0, self.ptr_size))
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;
    use crate::{
        ir::{
            function::{SsaDefSpec, SsaFunctionBuilder},
            ops::SsaOp,
            value::ConstValue,
            variable::{DefSite, VariableOrigin, VariableRole},
        },
        testing::{MockTarget, MockType},
    };

    /// Ends the test: a fixture the builder rejects is a broken test.
    fn abort<E, R>(_: E) -> R {
        process::abort()
    }

    /// A loop header whose two phis swap on the back edge, with a stack
    /// pointer and a parameter live on entry.
    ///
    /// Returns the function, the stack pointer, the parameter and the phis.
    fn swapping_loop() -> (
        SsaFunction<MockTarget>,
        SsaVarId,
        SsaVarId,
        SsaVarId,
        SsaVarId,
    ) {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        let (stack, parameter) = {
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
                MockType::I32,
            );
            (stack, parameter)
        };
        let (one, two) = builder
            .in_block(0, |b| {
                let one = b.const_i32(SsaDefSpec::tmp(MockType::I32), 1)?;
                let two = b.const_i32(SsaDefSpec::tmp(MockType::I32), 2)?;
                b.jump(1)?;
                Ok((one, two))
            })
            .unwrap_or_else(abort);
        let (first, second) = builder
            .in_block(1, |b| {
                let first = b.empty_phi(SsaDefSpec::tmp(MockType::I32))?;
                let second = b.empty_phi(SsaDefSpec::tmp(MockType::I32))?;
                Ok((first, second))
            })
            .unwrap_or_else(abort);
        for (result, from_entry, from_back_edge) in [(first, one, second), (second, two, first)] {
            builder
                .add_phi_operand(1, result, 0, from_entry)
                .unwrap_or_else(abort);
            builder
                .add_phi_operand(1, result, 1, from_back_edge)
                .unwrap_or_else(abort);
        }
        builder
            .in_block(1, |b| b.branch(parameter, 1, 2))
            .unwrap_or_else(abort);
        builder.in_block(2, |b| b.ret(None)).unwrap_or_else(abort);
        (
            builder.finish().unwrap_or_else(abort),
            stack,
            parameter,
            first,
            second,
        )
    }

    /// The entry stack pointer is the stack at the entry offset; a parameter
    /// nothing bound is named by itself.
    #[test]
    fn inputs_read_as_what_they_are() {
        let (ssa, stack, parameter, _, _) = swapping_loop();
        let frame = Frame::enter(&ssa, 0, 3, -64, vec![], PointerSize::Bit64);
        assert_eq!(
            frame.value(stack),
            Value::Addr(Address::new(Root::Stack, -64, PointerSize::Bit64))
        );
        assert_eq!(
            frame.value(parameter),
            Value::Addr(Address::of_root(Root::Opaque {
                activation: 3,
                value: parameter,
                run: 0,
            }))
        );
        let bound = Frame::enter(
            &ssa,
            0,
            3,
            -64,
            vec![(parameter, Value::Const(ConstValue::I32(9)))],
            PointerSize::Bit64,
        );
        assert_eq!(bound.value(parameter), Value::Const(ConstValue::I32(9)));
    }

    /// Phis take their values together: a swap on the back edge swaps.
    #[test]
    fn phis_arrive_together() {
        let (ssa, _, _, first, second) = swapping_loop();
        let mut frame = Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64);
        frame.move_to(Position {
            block: 0,
            previous: None,
            next: 0,
        });
        for (var, value) in ssa
            .block(0)
            .map(|block| block.instructions().to_vec())
            .unwrap_or_default()
            .iter()
            .filter_map(|instruction| match instruction.op() {
                SsaOp::Const { dest, value } => Some((*dest, value.clone())),
                _ => None,
            })
        {
            frame.define(var, Value::Const(value), 1);
        }
        frame.arrive(&ssa, 1, Entry::Edge(0));
        assert_eq!(frame.value(first), Value::Const(ConstValue::I32(1)));
        assert_eq!(frame.value(second), Value::Const(ConstValue::I32(2)));
        frame.arrive(&ssa, 1, Entry::Edge(1));
        assert_eq!(frame.value(first), Value::Const(ConstValue::I32(2)));
        assert_eq!(frame.value(second), Value::Const(ConstValue::I32(1)));
    }

    /// A definition nothing is known about is named by its run; two
    /// activations at one place agree only where their values do.
    #[test]
    fn unknown_definitions_are_named_and_joins_keep_agreement() {
        let (ssa, _, _, first, second) = swapping_loop();
        let mut left = Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64);
        left.define(first, Value::Unknown, 7);
        assert_eq!(
            left.value(first),
            Value::Addr(Address::of_root(Root::Opaque {
                activation: 0,
                value: first,
                run: 7,
            }))
        );
        let mut right = left.clone();
        left.define(second, Value::Const(ConstValue::I32(1)), 8);
        right.define(second, Value::Const(ConstValue::I32(2)), 8);
        let joined = left.join(&right).unwrap_or_else(|| abort(()));
        assert_eq!(joined.value(first), left.value(first));
        assert_eq!(joined.value(second), Value::Unknown);
        right.move_to(Position {
            block: 2,
            previous: Some(1),
            next: 0,
        });
        assert!(left.join(&right).is_none());
    }

    /// Declares many variables without executable instructions for storage-policy fixtures.
    fn sparse_body(count: usize) -> (SsaFunction<MockTarget>, SsaVarId) {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut last = SsaVarId::PLACEHOLDER;
        for _ in 0..count {
            last = ssa.create_variable(
                VariableOrigin::EntryLiveIn,
                0,
                DefSite::entry(),
                MockType::U8,
            );
        }
        (ssa, last)
    }

    /// Entry role lookup finds the declared pointer even after a large sparse prefix.
    #[test]
    fn entry_role_lookup_finds_the_declared_pointer() {
        let (mut ssa, last) = sparse_body(4096);
        ssa.variable_mut(last)
            .unwrap()
            .set_role(VariableRole::EntryStackPointer);
        assert_eq!(ssa.entry_stack_pointer(), Some(last));
    }

    /// Copies and comparisons stop in sparse storage before reaching its tail payload.
    #[test]
    fn frame_copies_and_agreement_check_sparse_storage() {
        let (ssa, last) = sparse_body(4096);
        let mut frame = Frame::enter(&ssa, 0, 7, -16, vec![], PointerSize::Bit64);
        frame.define(
            last,
            Value::Const(ConstValue::DecryptedString("decoded".into())),
            19,
        );
        let original = frame.clone();
        let copied = frame.clone();
        assert_eq!(copied, original);
        assert_eq!(copied.value(last), frame.value(last));
        assert!(copied.eq(&frame));
    }

    /// Target width is part of frame identity even when both value maps are empty.
    #[test]
    fn frame_identity_includes_target_pointer_width() {
        let ssa = SsaFunction::<MockTarget>::new(0, 0);
        let narrow = Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit32);
        let wide = Frame::enter(&ssa, 0, 0, 0, vec![], PointerSize::Bit64);
        assert!(!(narrow.eq(&wide)));
        assert!(narrow.join(&wide).is_none());
    }
    /// Missing-input alias provenance survives snapshots and joins without
    /// tainting explicit bindings or independently computed unknown definitions.
    #[test]
    fn unbound_stack_input_provenance_survives_frame_copies_and_joins() {
        let mut body = SsaFunction::<MockTarget>::new(1, 0);
        let input = body.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::entry(),
            MockType::Ptr,
        );
        let ordinary = Frame::enter(&body, 0, 1, 0, vec![], PointerSize::Bit64);
        let possible = ordinary.clone().with_stack_input_aliases(true);
        assert!(!ordinary.value(input).may_reference_stack());
        assert!(possible.value(input).may_reference_stack());
        assert!(!(ordinary.eq(&possible)));
        let copied = possible.clone();
        assert!(copied.value(input).may_reference_stack());
        for (left, right) in [(&ordinary, &possible), (&possible, &ordinary)] {
            let mut joined = left.join(right).expect("same activation");
            assert!(joined.value(input).may_reference_stack());
            let address = joined.value(input).address().expect("named input");
            assert_eq!(
                address.root(),
                Root::StackDerived {
                    activation: 1,
                    value: input,
                    run: 0
                }
            );
            joined.define(input, Value::Const(ConstValue::I64(7)), 1);
            assert_eq!(joined.value(input).constant(), Some(&ConstValue::I64(7)));
            joined.define(input, Value::Unknown, 2);
            assert!(!joined.value(input).may_reference_stack());
        }
    }
    /// Physical low-bit evidence participates in equality and meets across alternative paths.
    #[test]
    fn joins_keep_only_common_root_congruence_without_changing_identity() {
        let function = SsaFunction::<MockTarget>::new(0, 0);
        let empty = Frame::enter_at(
            &function,
            0,
            0,
            Address::new(Root::Stack, 0, PointerSize::Bit64),
            vec![],
            PointerSize::Bit64,
        );
        let first = empty
            .clone()
            .with_stack_congruence(StackCongruence::from_low_bits(4, 8));
        let second = empty
            .clone()
            .with_stack_congruence(StackCongruence::from_low_bits(4, 12));
        assert_ne!(first, empty);
        assert_eq!(
            first.join(&first).unwrap().stack_congruence(Root::Stack),
            StackCongruence::from_low_bits(4, 8)
        );
        for (left, right) in [(&first, &second), (&second, &first)] {
            let joined = left.join(right).unwrap();
            assert_eq!(
                joined.stack_congruence(Root::Stack),
                StackCongruence::from_low_bits(2, 0)
            );
            assert_eq!(joined.entry_stack_address().offset(), 0);
            assert!(joined.stack_congruence(Root::StackFrame(0)).is_none());
        }
        for (left, right) in [(&empty, &first), (&first, &empty)] {
            assert!(
                left.join(right)
                    .unwrap()
                    .stack_congruence(Root::Stack)
                    .is_none()
            );
        }
    }

    /// Joins retain a shared live return contract and refuse a different contract.
    #[test]
    fn live_return_bindings_are_preserved_and_distinguish_joins() {
        let (ssa, _, parameter, _, _) = swapping_loop();
        let baseline = Frame::enter(
            &ssa,
            0,
            3,
            0,
            vec![(parameter, Value::Const(ConstValue::I32(7)))],
            PointerSize::Bit64,
        );
        for binding in [
            ReturnBinding::Identity,
            ReturnBinding::Word(32),
            ReturnBinding::Unknown,
            ReturnBinding::Discard,
        ] {
            let first = baseline.clone().with_return_binding(binding);
            let mut alternative = first.clone();
            alternative.define(parameter, Value::Const(ConstValue::I32(9)), 1);
            for (left, right) in [(&first, &alternative), (&alternative, &first)] {
                let joined = left.join(right).unwrap_or_else(|| process::abort());
                assert_eq!(joined.return_binding(), binding);
                assert_eq!(joined.value(parameter), Value::Unknown);
            }
            let different =
                first
                    .clone()
                    .with_return_binding(if binding == ReturnBinding::Unknown {
                        ReturnBinding::Word(32)
                    } else {
                        ReturnBinding::Unknown
                    });
            assert_ne!(first, different);
            assert!(first.join(&different).is_none());
            assert!(different.join(&first).is_none());
        }
    }

    /// Retired observation segments remain distinct even when the live binding agrees.
    #[test]
    fn observation_return_bindings_distinguish_frame_joins() {
        let (ssa, _, _, _, _) = swapping_loop();
        let scope = CallScope {
            body: None,
            returns: Some(2),
            returns_binding: ReturnBinding::Word(32),
            returns_since: 0,
        };
        let first = Frame::enter(&ssa, 0, 3, 0, vec![], PointerSize::Bit64)
            .with_return_binding(ReturnBinding::Unknown)
            .with_observation(Some(scope));
        let same = first.join(&first).unwrap_or_else(|| process::abort());
        assert_eq!(same.return_binding(), ReturnBinding::Unknown);
        assert_eq!(same.observation(), Some(scope));
        let different = first.clone().with_observation(Some(CallScope {
            returns_binding: ReturnBinding::Unknown,
            ..scope
        }));
        assert_ne!(first, different);
        assert!(first.join(&different).is_none());
        assert!(different.join(&first).is_none());
    }

    /// Source namespaces retain stack lifetime and permissions while separating unknown inputs.
    #[test]
    fn continuations_preserve_physical_storage_and_freshen_unbound_value_identity() {
        let (ssa, stack, parameter, _, _) = swapping_loop();
        let mut frame = Frame::enter(&ssa, 0, 3, -64, vec![], PointerSize::Bit64)
            .with_stack_extent(Some(-32..16))
            .with_return_binding(ReturnBinding::Word(32))
            .with_observation(Some(CallScope {
                body: Some(0),
                returns: Some(1),
                returns_binding: ReturnBinding::Word(32),
                returns_since: 0,
            }));
        let original_input = frame.value(parameter);
        for (body, offset, run) in [(7, -80, 9), (8, -48, 10)] {
            let position = Address::new(Root::Stack, offset, PointerSize::Bit64);
            let prior_input = frame.value(parameter);
            frame.continue_source(&ssa, body, position, vec![], run);
            assert_eq!(frame.body(), body);
            assert_eq!(frame.activation(), 3);
            assert_eq!(frame.release_entry_stack(), -64);
            assert_eq!(frame.entry_stack_address(), position);
            assert_eq!(frame.value(stack), Value::Addr(position));
            assert_ne!(frame.value(parameter), original_input);
            assert_ne!(frame.value(parameter), prior_input);
            assert_eq!(frame.return_binding(), ReturnBinding::Word(32));
            assert_eq!(
                frame.observation(),
                // The continued activation's own record is retired into its
                // continuation, entered by this same activation.
                Some(CallScope {
                    body: None,
                    returns: Some(0),
                    returns_binding: ReturnBinding::Identity,
                    returns_since: 3,
                })
            );
            assert!(frame.covers_stack(Address::new(Root::Stack, -96, PointerSize::Bit64), 48));
            for offset in [-97, -48] {
                assert!(
                    !frame.covers_stack(Address::new(Root::Stack, offset, PointerSize::Bit64), 1)
                );
            }
        }
    }
}
