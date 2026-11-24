//! ABI-bound call values derived from one captured caller state.

use crate::{
    analysis::{
        Machine,
        budget::Exhaustion,
        execution::{
            access::Access,
            inputs::CallInputs,
            memory::MemoryRefusal,
            register,
            state::State,
            types::OperationLayout,
            value::{Address, Value},
        },
    },
    ir::{StagedValue, variable::SsaVarId},
    target::{Target, TypeLayout},
};

/// Actual carrier bits retained separately from the logical argument's typed view.
pub struct PhysicalArgument<T: Target> {
    /// Value in the caller's original register or ABI stack word.
    pub value: Value<T>,
    /// Representation established for that carrier, when available.
    pub layout: Option<TypeLayout>,
}

/// One completed optional argument projection and its actual memory reads.
pub struct ArgumentProjection<T: Target> {
    /// Replacement logical arguments; absence preserves the ordinary operand list.
    pub values: Option<Vec<Value<T>>>,
    /// Original physical carriers in the same order as the logical arguments.
    pub physical: Vec<PhysicalArgument<T>>,
    /// Complete ABI reads and source-promoted writes performed before transfer.
    pub accesses: Vec<(Address, usize)>,
    /// Staging exhaustion, retained even when a host ignores the return value.
    pub exhaustion: Option<Exhaustion>,
    /// Whether an ABI read lacked a complete address or representation.
    pub unresolved: bool,
}

/// Binds a call's logical arguments from its captured values and reaching ABI state.
///
/// Hosts establish physical locations and types; the executor owns memory access,
/// lifetime checking and pointer provenance. Bind before tail retirement. Models,
/// callee bindings and pre-call observations consume this same logical argument
/// list. Physical register bits remain available for wider native entry reads.
/// Returning without calling [`Self::begin`] preserves semantic SSA operands.
pub struct ArgumentBindings<'a, T: Target> {
    /// Ordinary, implicit and staged values captured at the call boundary.
    inputs: &'a CallInputs<T>,
    /// Exact staged offset identities belonging to this call's source instruction.
    staged: &'a [StagedValue],
    /// Reaching caller state, kept private from the host.
    state: &'a mut State<T>,
    /// Checked memory and target interpretation in the current execution.
    access: Access<'a>,
    /// Machine used for typed register views.
    machine: Machine,
    /// Locally accumulated argument values and access evidence.
    projection: ArgumentProjection<T>,
}

/// How the executor lends [`ArgumentBindings`] to a host and takes back its result.
///
/// The host-facing pushes are inherent; borrowing the caller's state and
/// reading the completed projection stay with the executor.
pub trait ExecutorArguments<'a, T: Target> {
    /// Borrows one captured call and its live caller state.
    fn new(
        inputs: &'a CallInputs<T>,
        staged: &'a [StagedValue],
        state: &'a mut State<T>,
        access: Access<'a>,
        machine: Machine,
    ) -> Self;

    /// Returns the completed list and actual source-read evidence.
    fn finish(self) -> ArgumentProjection<T>;
}

impl<'a, T: Target> ExecutorArguments<'a, T> for ArgumentBindings<'a, T> {
    fn new(
        inputs: &'a CallInputs<T>,
        staged: &'a [StagedValue],
        state: &'a mut State<T>,
        access: Access<'a>,
        machine: Machine,
    ) -> Self {
        Self {
            inputs,
            staged,
            state,
            access,
            machine,
            projection: ArgumentProjection {
                values: None,
                physical: Vec::new(),
                accesses: Vec::new(),
                unresolved: false,
                exhaustion: None,
            },
        }
    }

    fn finish(self) -> ArgumentProjection<T> {
        self.projection
    }
}

impl<'a, T: Target> ArgumentBindings<'a, T> {
    /// Begins an explicit argument list in the host's proven ABI order.
    ///
    /// Each subsequent push adds one position, including unknown positions.
    /// Calling this again discards the previous list but retains all actual read
    /// evidence and refusals; memory accesses cannot be undone by resetting a list.
    pub fn begin(&mut self) {
        self.projection.values = Some(Vec::new());
        self.projection.physical.clear();
    }

    /// Appends an unknown argument while retaining possible private-stack aliases.
    pub fn push_unknown(&mut self) {
        self.push(Value::StackDerived, None, None);
    }

    /// Appends a typed view of one captured ordinary operand.
    ///
    /// `source` describes its physical representation and `target` the logical
    /// parameter. Unsupported or wider views remain unknown, preserving aliases.
    /// Missing operands occupy an unknown position rather than shifting later ones.
    pub fn push_ordinary(&mut self, argument: usize, source: Option<TypeLayout>, target: &T::Type) {
        let value = self
            .inputs
            .ordinary()
            .get(argument)
            .cloned()
            .unwrap_or(Value::StackDerived);
        self.push(value, source, Some(target));
    }

    /// Appends a typed view of one uniquely staged stack value.
    ///
    /// The host proves that this exact outgoing offset is a parameter. Staged
    /// locals are not arguments by default. Missing or duplicated offsets produce
    /// an unknown position. This reads the captured value, not another SSA execution.
    pub fn push_staged(&mut self, offset: i64, source: Option<TypeLayout>, target: &T::Type) {
        let value = self
            .inputs
            .staged_at(self.staged, offset)
            .cloned()
            .unwrap_or(Value::StackDerived);
        self.push(value, source, Some(target));
    }

    /// Materializes an attested source argument store before observation or transfer.
    ///
    /// The host proves that this captured ordinary operand represents an actual
    /// outgoing stack write. `base` names its caller SSA pointer, or the entry
    /// stack when absent. `offset` is the checked displacement from that base,
    /// and `layout` is the actual stored representation before logical narrowing.
    /// Missing coordinates or layout refuse the invocation. All writes use the
    /// executor's ordinary lifetime, source-admission and storage checks.
    ///
    /// Returns `false` on refusal or exhaustion, which remain latched even if the
    /// host ignores the result. Stage every promoted source store before reading
    /// the logical argument view. Physical stores already present need no staging.
    #[must_use]
    pub fn stage_ordinary_stack(
        &mut self,
        argument: usize,
        base: Option<SsaVarId>,
        offset: Option<i64>,
        layout: Option<TypeLayout>,
    ) -> bool {
        if self.projection.exhaustion.is_some() || self.state.memory().refusal().is_some() {
            return false;
        }
        let base = match base {
            Some(value) => self.state.value(value).address(),
            None => self.state.frame().map(|frame| frame.entry_stack_address()),
        };
        let address = base.and_then(|base| base.moved(offset?, self.machine.ptr_size));
        let Some((address, layout)) = address.zip(layout.filter(|layout| !layout.is_unknown()))
        else {
            self.state
                .memory_mut()
                .refuse(MemoryRefusal::UnresolvedWrite);
            return false;
        };
        let value = self
            .inputs
            .ordinary()
            .get(argument)
            .cloned()
            .unwrap_or(Value::StackDerived);
        if let Err(exhaustion) = self.access.store_layout(
            self.state.memory_mut(),
            &Value::Addr(address),
            &value,
            layout,
            None,
        ) {
            self.projection.exhaustion = Some(exhaustion);
            return false;
        }
        if let Some(storage) = layout.storage_layout(self.machine.ptr_size) {
            self.projection.accesses.push((address, storage.bytes()));
        } else {
            self.projection.unresolved = true;
        }
        self.state.memory().refusal().is_none()
    }

    /// Appends a logical argument read from a proven reaching caller-stack location.
    ///
    /// `base` names an actual caller SSA pointer; `None` selects its entry stack.
    /// `offset` is relative to that base. `physical` states the ABI word read,
    /// and `target` its logical view. The host proves the location and extent.
    /// The executor checks the live span and target layout before reading it.
    /// A refusal withholds the invocation even when the host ignores the result.
    #[must_use]
    pub fn push_stack(
        &mut self,
        base: Option<SsaVarId>,
        offset: i64,
        physical: &T::Type,
        target: &T::Type,
    ) -> bool {
        if self.state.memory().refusal().is_some() {
            return false;
        }
        let base = match base {
            Some(value) => self.state.value(value).address(),
            None => self.state.frame().map(|frame| frame.entry_stack_address()),
        };
        let Some(address) = base.and_then(|base| base.moved(offset, self.machine.ptr_size)) else {
            self.projection.unresolved = true;
            self.push_unknown();
            return true;
        };
        let layout = self.access.type_layout::<T>(physical);
        let query = || layout;
        let access = self.access.with_operation(OperationLayout::new(&query));
        match access.load_with_footprint(self.state.memory(), &Value::Addr(address), physical, None)
        {
            Ok(read) => {
                if let Some(footprint) = read.footprint {
                    self.projection.accesses.push(footprint);
                } else {
                    self.projection.unresolved = true;
                }
                self.push(read.value, layout, Some(target));
                true
            }
            Err(reason) => {
                self.state.memory_mut().refuse(reason);
                false
            }
        }
    }

    /// Adds one carrier and its supported typed view to the explicit list.
    fn push(&mut self, value: Value<T>, layout: Option<TypeLayout>, target: Option<&T::Type>) {
        let Some(values) = &mut self.projection.values else {
            return;
        };
        let value = if value.is_unknown() {
            Value::StackDerived
        } else {
            value
        };
        values.push(target.map_or_else(
            || value.join(&Value::Unknown),
            |target| register::read(&value, layout, target, self.machine),
        ));
        self.projection
            .physical
            .push(PhysicalArgument { value, layout });
    }
}
