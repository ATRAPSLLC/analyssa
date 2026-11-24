//! Pure operations over what static execution knows.
//!
//! An operation over constants is folded exactly as the rest of the crate
//! folds it, by [`Machine::evaluate`]. What this adds is arithmetic on
//! addresses: moving one by a constant, the distance between two under one
//! root, and comparing two for equality. Scalar masks, bitwise logic, shifts
//! and conversions retain partial integer facts through the shared typed bit
//! transfer. Unknown positions stay unknown, and possible private stack aliases
//! remain visible to escape and memory handling.

use crate::{
    analysis::{
        consts::{Destination, FoldedResults, Machine, OperandLayout, TypeQueries},
        execution::{
            alignment::{CongruenceLookup, StackCongruences},
            types::OperationLayout,
            value::{Address, Root, Value},
        },
        image::ImageData,
    },
    ir::{evaluation::Evaluation, ops::SsaOp, value::ConstValue, variable::SsaVarId},
    target::Target,
};

/// What a symbol address adds to read as the number it holds: unknown, or
/// the image's, read only when an operation actually converts a symbol
/// address into a number.
///
/// Execution asks the image lazily so that an operation refused before it
/// reaches an address (an oversized store, for one) never queries the image.
#[derive(Clone, Copy)]
enum NumberOffset<'a> {
    /// No mapping is known.
    Unknown,
    /// The image states the mapping, asked on first use.
    Image(&'a dyn ImageData),
}

impl NumberOffset<'_> {
    /// Returns the mapping, asking the image only now.
    fn get(self) -> Option<i64> {
        match self {
            Self::Unknown => None,
            Self::Image(image) => image.number_offset(),
        }
    }
}

/// Evaluates pure operations as one machine computes them.
#[derive(Clone, Copy)]
pub struct Scalar<'a> {
    /// The machine: the pointer width addresses and native integers wrap at,
    /// and what its float operations give where IEEE leaves it.
    machine: Machine,
    /// Lazy target query scoped to the instruction currently being evaluated.
    operation: Option<OperationLayout<'a>>,
    /// Borrowed low-bit evidence for live concrete stack roots.
    congruences: Option<CongruenceLookup<'a>>,
    /// Attested addition mapping absolute numbers into the module's symbol space.
    number_offset: NumberOffset<'a>,
    /// Path-local projection from a live allocation to physical bits.
    heap_address_bits: Option<&'a dyn Fn(Address) -> Option<u64>>,
}

impl<'a> Scalar<'a> {
    /// An evaluator computing as `machine` does.
    pub const fn new(machine: Machine) -> Self {
        Self {
            machine,

            operation: None,
            congruences: None,
            number_offset: NumberOffset::Unknown,
            heap_address_bits: None,
        }
    }

    /// Reads the number-to-symbol mapping from `image`, asking it only when an
    /// operation converts a symbol address into a number.
    pub const fn with_image(mut self, image: Option<&'a dyn ImageData>) -> Self {
        self.number_offset = match image {
            Some(image) => NumberOffset::Image(image),
            None => NumberOffset::Unknown,
        };
        self
    }

    /// Shares the current instruction's target layout without inspecting it eagerly.
    pub const fn with_operation<'op>(self, operation: OperationLayout<'op>) -> Scalar<'op>
    where
        'a: 'op,
    {
        Scalar {
            machine: self.machine,
            operation: Some(operation),
            congruences: self.congruences,
            number_offset: self.number_offset,
            heap_address_bits: self.heap_address_bits,
        }
    }

    /// Borrows path-local stack evidence for a pure evaluation, before mutating state.
    pub fn with_stack_congruences<'state>(
        self,
        state: &'state dyn StackCongruences,
    ) -> Scalar<'state>
    where
        'a: 'state,
    {
        Scalar {
            machine: self.machine,
            operation: self.operation,
            congruences: Some(CongruenceLookup(state)),
            number_offset: self.number_offset,
            heap_address_bits: self.heap_address_bits,
        }
    }

    /// Borrows physical address evidence for allocations live on this path.
    pub fn with_heap_address_bits<'path>(
        self,
        lookup: &'path dyn Fn(Address) -> Option<u64>,
    ) -> Scalar<'path>
    where
        'a: 'path,
    {
        Scalar {
            machine: self.machine,
            operation: self.operation,
            congruences: self.congruences,
            number_offset: self.number_offset,
            heap_address_bits: Some(lookup),
        }
    }

    /// Runs one kernel entry against the layouts the target states: the
    /// current instruction's, where one is shared.
    fn with_type_queries<T: Target, R>(self, evaluate: impl FnOnce(&TypeQueries<'_, T>) -> R) -> R {
        let inspect = |ty: &T::Type| match self.operation {
            Some(operation) => operation.read(),
            None => T::type_layout(ty, self.machine.ptr_size),
        };
        evaluate(&TypeQueries::projected(self.machine.ptr_size, &inspect))
    }

    /// Returns a numeric control outcome and source stack provenance with one operand read.
    pub fn control<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: Destination,
        value: impl Fn(SsaVarId) -> Value<T>,
        layout_for: impl Fn(SsaVarId) -> OperandLayout,
    ) -> Option<(Evaluation<ConstValue<T>>, bool)> {
        let mut stack_bits = false;
        let outcome = self.with_type_queries::<T, _>(|types| {
            self.machine.evaluate_control_with_type_queries(
                op,
                dest,
                |var| {
                    let resolved = value(var);
                    stack_bits |= resolved.may_reference_stack();
                    self.numeric_constant(resolved, layout_for(var), op)
                },
                &layout_for,
                types,
            )
        })?;
        Some((outcome, stack_bits))
    }

    /// Evaluates a wide division's values and faults using each declared operand layout.
    pub fn control_results<T: Target>(
        self,
        op: &SsaOp<T>,
        value: impl Fn(SsaVarId) -> Value<T>,
        layout_for: impl Fn(SsaVarId) -> OperandLayout,
    ) -> Option<(Evaluation<FoldedResults<T>>, bool)> {
        let mut stack_bits = false;
        let SsaOp::WideDiv {
            low, high, divisor, ..
        } = op
        else {
            return None;
        };
        let mut expected = None;
        let mut mismatch = false;
        for value in [*low, *high, *divisor] {
            match layout_for(value) {
                OperandLayout::Register(layout) if layout.integer && layout.lanes.is_none() => {
                    let width = layout.bits.filter(|width| (1..=64).contains(width));
                    mismatch |=
                        width.is_none() || expected.is_some_and(|known| Some(known) != width);
                    expected = expected.or(width);
                }
                _ => mismatch = true,
            }
        }
        let outcome = self.machine.evaluate_results_control(op, |variable| {
            let held = value(variable);
            stack_bits |= held.may_reference_stack();
            let held = self.numeric_constant(held, layout_for(variable), op)?;
            let read = layout_for(variable).read_numeric(held, self.machine.ptr_size);
            mismatch |= read.is_none();
            read
        })?;
        Some((
            if mismatch {
                Evaluation::Unsupported
            } else {
                outcome
            },
            stack_bits,
        ))
    }

    /// Evaluates `op`, reading its operands through `value`.
    ///
    /// Meant for operations without effects; any other operation is unknown.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation.
    /// * `dest` - What its destination declares ([`Destination`]): its width,
    ///   lanes and kind.
    /// * `value` - What the execution knows about each operand.
    #[cfg(test)]
    fn evaluate<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        value: impl Fn(SsaVarId) -> Value<T>,
    ) -> Value<T> {
        self.evaluate_with_operand_layouts(op, dest, value, |_| OperandLayout::default())
    }

    /// Evaluates pure values using the declared layout of each numeric input.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to evaluate.
    /// * `dest` - Its result layout.
    /// * `value` - Resolves owned values.
    /// * `layout_for` - Returns each input's attested layout when needed.
    pub fn evaluate_with_operand_layouts<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        value: impl Fn(SsaVarId) -> Value<T>,
        layout_for: impl Fn(SsaVarId) -> OperandLayout,
    ) -> Value<T> {
        let dest = dest.into();
        let dest_width = dest.bits;
        if let Some(address) = self.masked_stack_address(op, dest, &value, &layout_for) {
            return Value::Addr(address);
        }
        if let Some(partial) = self.partial_integer(op, dest, &value, &layout_for) {
            return partial;
        }
        let computed = match op {
            SsaOp::Const {
                value: constant, ..
            } => self.constant(constant, dest),
            SsaOp::Copy { src, .. } => value(*src),
            SsaOp::Select {
                condition,
                true_val,
                false_val,
                ..
            } => match value(*condition).constant().and_then(ConstValue::as_bool) {
                Some(true) => value(*true_val),
                Some(false) => value(*false_val),
                None => value(*true_val).join(&value(*false_val)),
            },
            SsaOp::PtrAdd {
                base,
                index,
                stride,
                offset,
                ..
            } => self.pointer_add(value(*base), index.map(&value), *stride, *offset),
            SsaOp::Add { left, right, .. } if self.holds_a_pointer(dest_width) => {
                let (lhs, rhs) = (value(*left), value(*right));
                match (lhs, rhs) {
                    (Value::Addr(address), Value::Const(delta))
                    | (Value::Const(delta), Value::Addr(address))
                        if Self::address_inputs(*left, *right, &layout_for) =>
                    {
                        self.moved(address, &delta, 1)
                    }
                    (lhs, rhs) => self.fold_resolved(
                        op,
                        dest,
                        [Some((*left, lhs)), Some((*right, rhs))],
                        &layout_for,
                    ),
                }
            }
            SsaOp::Sub { left, right, .. } if self.holds_a_pointer(dest_width) => {
                let (lhs, rhs) = (value(*left), value(*right));
                match (lhs, rhs) {
                    (Value::Addr(address), Value::Const(delta))
                        if Self::address_inputs(*left, *right, &layout_for) =>
                    {
                        self.moved(address, &delta, -1)
                    }
                    (Value::Addr(lhs), Value::Addr(rhs))
                        if Self::address_inputs(*left, *right, &layout_for) =>
                    {
                        lhs.distance_bits_from(rhs)
                            .and_then(|distance| self.integer(dest_width, distance))
                            .map_or(Value::Unknown, Value::Const)
                    }
                    (lhs, rhs) => self.fold_resolved(
                        op,
                        dest,
                        [Some((*left, lhs)), Some((*right, rhs))],
                        &layout_for,
                    ),
                }
            }
            SsaOp::Ceq { left, right, .. } => {
                let (lhs, rhs) = (value(*left), value(*right));
                match (lhs, rhs) {
                    (Value::Addr(lhs), Value::Addr(rhs))
                        if lhs.root() == rhs.root()
                            && Self::address_inputs(*left, *right, &layout_for) =>
                    {
                        Value::Const(ConstValue::from_bool(lhs.offset() == rhs.offset()))
                    }
                    (lhs, rhs) => self.fold_resolved(
                        op,
                        dest,
                        [Some((*left, lhs)), Some((*right, rhs))],
                        &layout_for,
                    ),
                }
            }
            SsaOp::Xor { left, right, .. } => {
                let (lhs, rhs) = (value(*left), value(*right));
                match (lhs, rhs) {
                    (Value::Addr(lhs), Value::Addr(rhs))
                        if lhs == rhs
                            && dest.lanes.is_none()
                            && Self::address_inputs(*left, *right, &layout_for) =>
                    {
                        self.machine
                            .integer_zero(dest_width)
                            .map_or(Value::Unknown, Value::Const)
                    }
                    (lhs, rhs) => self.fold_resolved(
                        op,
                        dest,
                        [Some((*left, lhs)), Some((*right, rhs))],
                        &layout_for,
                    ),
                }
            }
            SsaOp::IntConv { operand, .. }
            | SsaOp::IntToPtr { operand, .. }
            | SsaOp::PtrToInt { operand, .. }
            | SsaOp::Bitcast { operand, .. } => match value(*operand) {
                Value::Addr(address)
                    if self.preserves_address(op, dest, address, layout_for(*operand)) =>
                {
                    Value::Addr(address)
                }
                input => self.fold_resolved(op, dest, [Some((*operand, input)), None], &layout_for),
            },
            _ => self.fold(op, dest, &value, &layout_for),
        };
        if computed.is_unknown() && self.references_stack(op, &value) {
            Value::StackDerived
        } else {
            computed
        }
    }

    /// Evaluates an operation whose results fold one by one
    /// ([`SsaOp::folds_results`]: the double-width arithmetic, the exact
    /// float functions of several results, a computation's data and status):
    /// each of its results, a constant where it folds of the operands read as
    /// `layout_for` declares them
    /// ([`Machine::evaluate_results_with_operand_layouts`]) and unknown
    /// otherwise, or `None` for any other operation.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation.
    /// * `value` - What the execution knows about each operand.
    /// * `layout_for` - Each operand's and result's declared layout.
    pub fn evaluate_results<T: Target>(
        self,
        op: &SsaOp<T>,
        value: impl Fn(SsaVarId) -> Value<T>,
        layout_for: impl Fn(SsaVarId) -> OperandLayout,
    ) -> Option<Vec<(SsaVarId, Value<T>)>> {
        if !op.folds_results() {
            return None;
        }
        let results = self
            .machine
            .evaluate_results_with_operand_layouts(op, |var| value(var).into_constant(), layout_for)
            .unwrap_or_default();
        let unknown = if results.len() < op.defs().count() && self.references_stack(op, &value) {
            Value::StackDerived
        } else {
            Value::Unknown
        };
        Some(
            op.defs()
                .map(|def| {
                    let folded = results.iter().find(|(result, _)| *result == def);
                    (
                        def,
                        folded.map_or_else(
                            || unknown.clone(),
                            |(_, constant)| Value::Const(constant.clone()),
                        ),
                    )
                })
                .collect(),
        )
    }

    /// What a constant operation defines: a symbol is the address it names.
    fn constant<T: Target>(self, constant: &ConstValue<T>, dest: Destination) -> Value<T> {
        match constant {
            ConstValue::Symbol(symbol) => T::execution_address(symbol, self.machine.ptr_size)
                .map_or(Value::Unknown, Value::Addr),
            other => self
                .machine
                .normalize_literal(other.clone(), dest)
                .map_or(Value::Unknown, Value::Const),
        }
    }

    /// Returns whether an operand may reference the stack, or `None` on refusal.
    ///
    /// Reads only uses and stops at the first known stack reference.
    pub fn references_stack<T: Target>(
        self,
        op: &SsaOp<T>,
        value: &impl Fn(SsaVarId) -> Value<T>,
    ) -> bool {
        let result = op.try_for_each_use(|var| {
            if value(var).may_reference_stack() {
                Err(())
            } else {
                Ok(())
            }
        });
        result.is_err()
    }

    /// Folds `op`, resolving only operands the shared evaluator reads.
    fn fold<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: Destination,
        value: &impl Fn(SsaVarId) -> Value<T>,
        layout_for: &impl Fn(SsaVarId) -> OperandLayout,
    ) -> Value<T> {
        self.with_type_queries::<T, _>(|types| {
            self.machine
                .evaluate_with_type_queries(
                    op,
                    dest,
                    |var| self.numeric_constant(value(var), layout_for(var), op),
                    layout_for,
                    types,
                )
                .map_or(Value::Unknown, Value::Const)
        })
    }

    /// Retains partial scalar integer facts without revisiting SSA definitions.
    ///
    /// Fully concrete operands continue through the exact semantic kernel.
    /// Each integer input is bounded to 128 bits; unknown pointer bits
    /// retain their stack provenance until the entire result becomes constant.
    fn partial_integer<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: Destination,
        value: &impl Fn(SsaVarId) -> Value<T>,
        layout_for: &impl Fn(SsaVarId) -> OperandLayout,
    ) -> Option<Value<T>> {
        if !matches!(
            op,
            SsaOp::And { .. }
                | SsaOp::Or { .. }
                | SsaOp::Xor { .. }
                | SsaOp::Not { .. }
                | SsaOp::Shl { .. }
                | SsaOp::Shr { .. }
                | SsaOp::IntConv {
                    overflow_check: false,
                    ..
                }
                | SsaOp::Select { .. }
                | SsaOp::Ceq { .. }
                | SsaOp::Clt { .. }
                | SsaOp::Cgt { .. }
                | SsaOp::ReadFlags { .. }
                | SsaOp::MergeFlags { .. }
                | SsaOp::InsertFlag { .. }
        ) {
            return None;
        }
        self.with_type_queries::<T, _>(|types| {
            let mut partial_input = false;
            let mut stack_derived = false;
            let bits = self.machine.evaluate_integer_bits(
                op,
                dest,
                |variable, layout| {
                    let input = value(variable);
                    stack_derived |= input.may_reference_stack();
                    match input {
                        Value::Const(constant) => {
                            self.machine.constant_integer_bits(&constant, layout)
                        }
                        Value::Integer(bits) => {
                            partial_input = true;
                            bits.bits()
                        }
                        Value::Addr(address) => {
                            partial_input = true;
                            if matches!(op, SsaOp::IntConv { .. })
                                && self.preserves_address(op, dest, address, layout)
                            {
                                None
                            } else {
                                self.numeric_constant(Value::<T>::Addr(address), layout, op)
                                    .and_then(|constant| {
                                        self.machine.constant_integer_bits(&constant, layout)
                                    })
                                    .or_else(|| {
                                        self.congruences?
                                            .low_bits(address, layout.scalar_integer()?.bits())
                                    })
                            }
                        }
                        _ => {
                            partial_input = true;
                            None
                        }
                    }
                },
                layout_for,
                types,
            )?;
            if !partial_input || bits.is_unknown() {
                return None;
            }
            if bits.value().is_some() {
                return Some(
                    self.machine
                        .known_result(op, bits, dest)
                        .map_or(Value::Unknown, Value::Const),
                );
            }
            Some(Value::partial_integer(bits, stack_derived))
        })
    }

    /// Preserves a stack coordinate for a fully typed low-bit mask: exactly
    /// where the stack's low bits justify it, and otherwise as the base of the
    /// realigned frame ([`Address::realigned`]).
    fn masked_stack_address<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: Destination,
        value: &impl Fn(SsaVarId) -> Value<T>,
        layout_for: &impl Fn(SsaVarId) -> OperandLayout,
    ) -> Option<Address> {
        let SsaOp::And { left, right, .. } = op else {
            return None;
        };
        let bits = self.machine.ptr_size.bits();
        if dest
            .scalar_integer()
            .is_none_or(|layout| layout.bits() != bits)
            || [*left, *right].into_iter().any(|var| {
                layout_for(var)
                    .scalar_integer()
                    .is_none_or(|layout| layout.bits() != bits)
            })
        {
            return None;
        }
        let (address, mask) = match (value(*left), value(*right)) {
            (Value::Addr(address), Value::Const(mask))
            | (Value::Const(mask), Value::Addr(address)) => (address, mask),
            _ => return None,
        };
        if address.pointer_size() != self.machine.ptr_size {
            return None;
        }
        let mask = mask.scalar_pattern_at_width(bits, self.machine.ptr_size)?;
        self.congruences
            .and_then(|congruences| congruences.masked(address, mask))
            .or_else(|| address.realigned(mask))
    }

    /// Excludes floating declarations from address arithmetic and identity proofs.
    fn address_inputs(
        left: SsaVarId,
        right: SsaVarId,
        layout_for: &impl Fn(SsaVarId) -> OperandLayout,
    ) -> bool {
        !matches!(layout_for(left), OperandLayout::Float(_))
            && !matches!(layout_for(right), OperandLayout::Float(_))
    }

    /// Reads mapped address bits through a native integer, an explicit pointer
    /// conversion, or an operation placing the pointer in a vector's lanes.
    ///
    /// Symbol coordinates use the loader's base relation. Heap coordinates
    /// require a physical mapping for the reaching allocation.
    fn numeric_constant<T: Target>(
        self,
        value: Value<T>,
        layout: OperandLayout,
        operation: &SsaOp<T>,
    ) -> Option<ConstValue<T>> {
        match value {
            Value::Addr(address) => {
                let width = self.machine.ptr_size;
                let full_integer = layout
                    .scalar_integer()
                    .is_some_and(|integer| integer.bits() == width.bits());
                // An explicit conversion reads a whole pointer's bits, and so
                // does an operation that places one in a vector's lanes: a
                // lane holds bits, never an object's identity, as the image's
                // pointer cell holds the number it was loaded with.
                let pointer_read = matches!(
                    operation,
                    SsaOp::PtrToInt { .. }
                        | SsaOp::VectorReinterpret { .. }
                        | SsaOp::VectorInsert { .. }
                        | SsaOp::VectorSplat { .. }
                ) && matches!(
                    layout,
                    // A pointer's type states no width of its own: it is the
                    // machine's.
                    OperandLayout::Register(declared)
                        if declared.lanes.is_none()
                            && declared.bits.is_none_or(|bits| bits == width.bits())
                );
                if address.pointer_size() != width || !(full_integer || pointer_read) {
                    return None;
                }
                let offset = u128::from(address.offset().cast_unsigned());
                let numeric = match address.root() {
                    Root::Number => offset,
                    Root::Symbol => {
                        offset.wrapping_sub(i128::from(self.number_offset.get()?).cast_unsigned())
                    }
                    Root::Heap(_) => {
                        let bits = (self.heap_address_bits?)(address)?;
                        if width.mask_unsigned(bits) != bits {
                            return None;
                        }
                        u128::from(bits)
                    }
                    Root::Stack
                    | Root::StackFrame(_)
                    | Root::StackAligned { .. }
                    | Root::Opaque { .. }
                    | Root::StackDerived { .. } => return None,
                };
                ConstValue::of_lane(width.bits(), width.mask_unsigned_128(numeric))
            }
            value => value.into_constant(),
        }
    }

    /// Consumes already resolved inputs after checking address-specific semantics.
    ///
    /// The fixed array lives on the stack and gives repeated SSA operands
    /// distinct owned slots. A consumed value is never copied or resolved again.
    fn fold_resolved<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: Destination,
        mut inputs: [Option<(SsaVarId, Value<T>)>; 2],
        layout_for: &impl Fn(SsaVarId) -> OperandLayout,
    ) -> Value<T> {
        self.with_type_queries::<T, _>(|types| {
            self.machine
                .evaluate_with_type_queries(
                    op,
                    dest,
                    |variable| {
                        inputs
                            .iter_mut()
                            .find(|slot| slot.as_ref().is_some_and(|(id, _)| *id == variable))
                            .and_then(Option::take)
                            .and_then(|(_, value)| {
                                self.numeric_constant(value, layout_for(variable), op)
                            })
                    },
                    layout_for,
                    types,
                )
                .map_or(Value::Unknown, Value::Const)
        })
    }

    /// Moves a pointer by its scaled index and displacement.
    ///
    /// A unit-stride symbolic index and a numeric base have the same sum as
    /// the corresponding symbolic base and numeric index. Scaling a root or
    /// adding two address identities supplies no representable displacement.
    pub fn pointer_add<T: Target>(
        self,
        base: Value<T>,
        index: Option<Value<T>>,
        stride: u64,
        offset: i64,
    ) -> Value<T> {
        let (base, index) = match (base, index) {
            (Value::Const(base), Some(Value::Addr(address)))
                if stride == 1
                    && address.pointer_size() == self.machine.ptr_size
                    && base
                        .scalar_pattern_at_width(
                            self.machine.ptr_size.bits(),
                            self.machine.ptr_size,
                        )
                        .is_some() =>
            {
                (Value::Addr(address), Some(Value::Const(base)))
            }
            operands => operands,
        };
        let index = match index {
            Some(Value::Const(index)) => Some(index),
            Some(_) => return Value::Unknown,
            None => None,
        };
        match base {
            Value::Addr(address) if address.pointer_size() != self.machine.ptr_size => {
                Value::Unknown
            }
            Value::Addr(address) => ConstValue::<T>::scaled_offset(
                index.as_ref(),
                stride,
                offset,
                self.machine.ptr_size,
            )
            .and_then(|delta| address.displaced_bits(delta, self.machine.ptr_size))
            .map_or(Value::Unknown, Value::Addr),
            Value::Const(base) => base
                .pointer_add(index.as_ref(), stride, offset, self.machine.ptr_size)
                .map_or(Value::Unknown, Value::Const),
            Value::StackDerived => Value::StackDerived,
            Value::Integer(bits) if bits.may_reference_stack() => Value::StackDerived,
            Value::Unknown | Value::Integer(_) => Value::Unknown,
        }
    }

    /// Moves `address` by a constant pattern, subtracting it when `sign` is negative.
    fn moved<T: Target>(self, address: Address, delta: &ConstValue<T>, sign: i64) -> Value<T> {
        delta
            .scalar_pattern_at_width(self.machine.ptr_size.bits(), self.machine.ptr_size)
            .map(|delta| {
                if sign < 0 {
                    delta.wrapping_neg()
                } else {
                    delta
                }
            })
            .and_then(|delta| address.displaced_bits(delta, self.machine.ptr_size))
            .map_or(Value::Unknown, Value::Addr)
    }

    /// Proves a conversion preserves every bit of an address identity.
    ///
    /// A register's width does not override the conversion target's width. An
    /// explicit source width must also match the address domain. With no source
    /// width, established pointer roots and operations that require a pointer or
    /// preserve representation supply that evidence; an opaque integer does not.
    fn preserves_address<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: Destination,
        address: Address,
        source: OperandLayout,
    ) -> bool {
        let pointer_bits = self.machine.ptr_size.bits();
        let (target, bitcast) = match op {
            SsaOp::IntConv {
                target,
                overflow_check: false,
                ..
            }
            | SsaOp::IntToPtr { target, .. }
            | SsaOp::PtrToInt { target, .. } => (target, false),
            SsaOp::Bitcast { target, .. } => (target, true),
            _ => return false,
        };
        if address.pointer_size() != self.machine.ptr_size
            || !self.holds_a_pointer(dest.bits)
            || (!bitcast && dest.lanes.is_some())
        {
            return false;
        }
        let source_preserved = match source {
            OperandLayout::Boolean => false,
            OperandLayout::Float(format) => bitcast && format.bits() == pointer_bits,
            OperandLayout::Register(layout) => {
                if !bitcast && layout.lanes.is_some() {
                    return false;
                }
                match layout.bits {
                    Some(bits) => bits == pointer_bits,
                    None => {
                        !matches!(
                            address.root(),
                            Root::Opaque { .. } | Root::StackDerived { .. }
                        ) || matches!(op, SsaOp::PtrToInt { .. } | SsaOp::Bitcast { .. })
                    }
                }
            }
        };
        if !source_preserved {
            return false;
        }
        let target_bits = self.with_type_queries::<T, _>(|types| {
            if bitcast {
                types.bitcast_width(target, dest.bits)
            } else {
                types.integer_type(target).map(|layout| layout.bits())
            }
        });
        target_bits == Some(pointer_bits)
    }

    /// Whether a destination `dest_width` bits wide holds a whole pointer.
    fn holds_a_pointer(self, dest_width: Option<u32>) -> bool {
        dest_width.is_none_or(|bits| bits == self.machine.ptr_size.bits())
    }

    /// Returns an integer bit pattern at the declared width or the pointer width.
    fn integer<T: Target>(self, bits: Option<u32>, value: u128) -> Option<ConstValue<T>> {
        let bits = bits.unwrap_or_else(|| self.machine.ptr_size.bits());
        if bits <= 64 {
            ConstValue::integer(bits, true, value as u64 as i64)
        } else {
            ConstValue::of_lane(bits, value)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::{
        PointerSize,
        analysis::{execution::StackCongruence, image::ImageRegion},
        ir::{
            float::{FloatBits, FloatExceptionLayout, FloatExceptions, FloatFormat},
            integer::{IntegerBits, IntegerType},
            ops::{FlagsMask, VectorExtendInLaneData},
            value::DecryptedArrayData,
        },
        target::Endianness,
        testing::{MACHINE, MockTarget, MockType},
    };

    /// An image that states only the loader's number-to-symbol mapping.
    struct Offset(Option<i64>);

    impl ImageData for Offset {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }
        fn region(&self, _: u64) -> Option<ImageRegion> {
            None
        }
        fn bytes(&self, _: u64, _: usize) -> Option<&[u8]> {
            None
        }
        fn number_offset(&self) -> Option<i64> {
            self.0
        }
    }

    /// Literal evaluation preserves the exact owned string and array payloads.
    #[test]
    fn literals_preserve_owned_payloads() {
        let values: [ConstValue<MockTarget>; 2] = [
            ConstValue::DecryptedString("é中𐍈".repeat(1024).into()),
            ConstValue::DecryptedArray(Box::new(DecryptedArrayData {
                data: vec![0x5a; 4096],
                element_type_ref: 7,
                element_size: 1,
            })),
        ];
        for value in values {
            assert_eq!(
                Scalar::new(MACHINE).constant(&value, Destination::default()),
                Value::Const(value)
            );
        }
    }

    /// Control evaluation resolves each unary input once and retains stack provenance on refusal.
    #[test]
    fn control_reads_each_operand_once_and_preserves_stack_provenance() {
        for kind in 0..3 {
            let op = match kind {
                0 => SsaOp::IntConv {
                    dest: var(0),
                    operand: var(1),
                    target: MockType::I8,
                    unsigned: false,
                    overflow_check: true,
                },
                1 => SsaOp::FloatToInt {
                    dest: var(0),
                    operand: var(1),
                    target: MockType::I8,
                    unsigned: false,
                    overflow_check: true,
                    source: None,
                },
                _ => SsaOp::Ckfinite {
                    dest: var(0),
                    operand: var(1),
                },
            };
            for input in [
                Value::Unknown,
                Value::StackDerived,
                stack(-8),
                Value::Const(ConstValue::Type(7)),
            ] {
                let reads = Cell::new(0usize);
                let expected_stack = input.may_reference_stack();
                let expected_outcome = if input.constant().is_some() {
                    Evaluation::Unsupported
                } else {
                    Evaluation::Unknown
                };
                let actual = SCALAR.control(
                    &op,
                    Destination::default(),
                    |id| {
                        assert_eq!(id, var(1));
                        reads.set(reads.get() + 1);
                        input.clone()
                    },
                    |_| OperandLayout::default(),
                );
                assert_eq!(actual, Some((expected_outcome, expected_stack)));
                assert_eq!(reads.get(), 1);
            }
        }
    }

    /// Conversion targets and attested source widths must preserve the complete address pattern.
    #[test]
    fn address_conversions_require_lossless_source_and_target_layouts() {
        for kind in 0..4 {
            for root in [
                Root::Stack,
                Root::Symbol,
                Root::StackDerived {
                    activation: 0,
                    value: var(1),
                    run: 0,
                },
                Root::Opaque {
                    activation: 0,
                    value: var(1),
                    run: 0,
                },
            ] {
                for (source_bits, target, preserved) in [
                    (Some(64), MockType::U64, true),
                    (Some(128), MockType::U64, false),
                    (Some(32), MockType::U64, false),
                    (Some(64), MockType::U32, false),
                    (
                        None,
                        MockType::U64,
                        kind >= 2 || matches!(root, Root::Stack | Root::Symbol),
                    ),
                ] {
                    let op: SsaOp<MockTarget> = match kind {
                        0 => SsaOp::IntConv {
                            dest: var(0),
                            operand: var(1),
                            target,
                            unsigned: true,
                            overflow_check: false,
                        },
                        1 => SsaOp::IntToPtr {
                            dest: var(0),
                            operand: var(1),
                            target,
                        },
                        2 => SsaOp::PtrToInt {
                            dest: var(0),
                            operand: var(1),
                            target,
                        },
                        _ => SsaOp::Bitcast {
                            dest: var(0),
                            operand: var(1),
                            target,
                        },
                    };
                    let address = Address::new(root, 7, PointerSize::Bit64);
                    let result = SCALAR.evaluate_with_operand_layouts(
                        &op,
                        Some(64),
                        |_| Value::Addr(address),
                        |_| {
                            OperandLayout::Register(Destination {
                                bits: source_bits,
                                lanes: None,
                                lane_format: None,
                                integer: true,
                                integer_type: None,
                            })
                        },
                    );
                    if preserved {
                        assert_eq!(result, Value::Addr(address));
                    } else {
                        let expected = if root.may_reference_stack() {
                            Value::StackDerived
                        } else {
                            Value::Unknown
                        };
                        assert_eq!(
                            result, expected,
                            "{kind}/{root:?}/{source_bits:?}/{target:?}"
                        );
                        assert_eq!(result.may_reference_stack(), root.may_reference_stack());
                    }
                }
            }
        }
    }

    /// Equal-width bitcasts carry the same unknown bits through floating interpretations.
    #[test]
    fn address_bitcasts_preserve_exact_float_bit_patterns() {
        let address = Address::new(
            Root::Opaque {
                activation: 0,
                value: var(1),
                run: 0,
            },
            0,
            PointerSize::Bit64,
        );
        for (target, layout) in [
            (MockType::U64, OperandLayout::Float(FloatFormat::Double)),
            (
                MockType::F64,
                OperandLayout::Register(Destination {
                    bits: Some(64),
                    lanes: None,
                    lane_format: None,
                    integer: true,
                    integer_type: None,
                }),
            ),
            (
                MockType::Unknown,
                OperandLayout::Register(Destination {
                    bits: Some(64),
                    lanes: None,
                    lane_format: None,
                    integer: true,
                    integer_type: None,
                }),
            ),
        ] {
            let op: SsaOp<MockTarget> = SsaOp::Bitcast {
                dest: var(0),
                operand: var(1),
                target,
            };
            assert_eq!(
                SCALAR.evaluate_with_operand_layouts(
                    &op,
                    Some(64),
                    |_| Value::Addr(address),
                    |_| layout
                ),
                Value::Addr(address)
            );
        }
    }

    /// A table of known operand values.
    fn known(
        values: &[(usize, Value<MockTarget>)],
    ) -> impl Fn(SsaVarId) -> Value<MockTarget> + use<> {
        let values = values.to_vec();
        move |var| {
            values
                .iter()
                .find(|(index, _)| *index == var.index())
                .map_or(Value::Unknown, |(_, value)| value.clone())
        }
    }

    fn var(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    fn stack(offset: i64) -> Value<MockTarget> {
        Value::Addr(Address::new(Root::Stack, offset, PointerSize::Bit64))
    }

    const SCALAR: Scalar = Scalar::new(MACHINE);

    /// Constants fold as the crate folds them.
    #[test]
    fn constants_fold() {
        let op = SsaOp::Xor {
            dest: var(0),
            left: var(1),
            right: var(2),
            flags: None,
        };
        let values = known(&[
            (1, Value::Const(ConstValue::I32(0x3b))),
            (2, Value::Const(ConstValue::I32(0x5a))),
        ]);
        assert_eq!(
            SCALAR.evaluate(&op, Some(32), values),
            Value::Const(ConstValue::I32(0x61))
        );
    }

    /// An address moves by a constant, in either operand order, and by an
    /// unsigned all-ones constant as by minus one.
    #[test]
    fn an_address_moves_by_a_constant() {
        let add = SsaOp::Add {
            dest: var(0),
            left: var(1),
            right: var(2),
            flags: None,
        };
        let values = known(&[
            (1, Value::Const(ConstValue::U64(u64::MAX))),
            (2, stack(-40)),
        ]);
        assert_eq!(SCALAR.evaluate(&add, None, values), stack(-41));
        let sub = SsaOp::Sub {
            dest: var(0),
            left: var(1),
            right: var(2),
            flags: None,
        };
        let values = known(&[(1, stack(0)), (2, Value::Const(ConstValue::I64(0x28)))]);
        assert_eq!(SCALAR.evaluate(&sub, None, values), stack(-0x28));
    }

    /// Two addresses under one root are a distance apart; under two roots
    /// nothing is known.
    #[test]
    fn a_distance_needs_one_root() {
        let sub = SsaOp::Sub {
            dest: var(0),
            left: var(1),
            right: var(2),
            flags: None,
        };
        let values = known(&[(1, stack(-8)), (2, stack(-40))]);
        assert_eq!(
            SCALAR.evaluate(&sub, Some(64), values),
            Value::Const(ConstValue::I64(32))
        );
        let image = Value::Addr(Address::new(Root::Symbol, 0x1000, PointerSize::Bit64));
        let values = known(&[(1, image), (2, stack(-40))]);
        assert_eq!(SCALAR.evaluate(&sub, Some(64), values), Value::StackDerived);
    }

    /// A structural address adds its scaled index and displacement.
    #[test]
    fn a_pointer_add_scales_its_index() {
        let op = SsaOp::PtrAdd {
            dest: var(0),
            base: var(1),
            index: Some(var(2)),
            stride: 4,
            offset: 2,
            result_type: MockType::Ptr,
        };
        let values = known(&[(1, stack(-64)), (2, Value::Const(ConstValue::I64(3)))]);
        assert_eq!(SCALAR.evaluate(&op, None, values), stack(-50));
        let values = known(&[(1, stack(-64))]);
        assert_eq!(SCALAR.evaluate(&op, None, values), Value::StackDerived);
    }

    /// Unit-stride commutation preserves one address identity at its declared target width.
    #[test]
    fn pointer_add_commutes_only_a_numeric_base_and_unscaled_matching_address() {
        for width in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            let scalar = Scalar::new(Machine {
                ptr_size: width,
                ..MACHINE
            });
            for root in [
                Root::Symbol,
                Root::Number,
                Root::Stack,
                Root::StackFrame(1),
                Root::Heap(1),
            ] {
                let address = Address::new(root, 100, width);
                for (number, offset, delta) in [
                    (ConstValue::<MockTarget>::I32(-3), 2, -1),
                    (ConstValue::U64(3), 5, 8),
                ] {
                    let expected = address
                        .moved(delta, width)
                        .map_or(Value::Unknown, Value::Addr);
                    assert_eq!(
                        scalar.pointer_add(
                            Value::Addr(address),
                            Some(Value::Const(number.clone())),
                            1,
                            offset
                        ),
                        expected
                    );
                    assert_eq!(
                        scalar.pointer_add(
                            Value::Const(number.clone()),
                            Some(Value::Addr(address)),
                            1,
                            offset
                        ),
                        expected
                    );
                    assert_eq!(
                        scalar.pointer_add(
                            Value::Const(number),
                            Some(Value::Addr(address)),
                            2,
                            offset
                        ),
                        Value::Unknown
                    );
                }
                assert_eq!(
                    scalar.pointer_add::<MockTarget>(
                        Value::Addr(address),
                        Some(Value::Addr(address)),
                        1,
                        0
                    ),
                    Value::Unknown
                );
                for constant in [
                    ConstValue::<MockTarget>::True,
                    ConstValue::Float(
                        FloatBits::new(FloatFormat::Single, 0).expect("single-precision zero"),
                    ),
                    ConstValue::Symbol(100),
                ] {
                    assert_eq!(
                        scalar.pointer_add(
                            Value::Const(constant),
                            Some(Value::Addr(address)),
                            1,
                            0
                        ),
                        Value::Unknown
                    );
                }
            }
        }
        let address = Address::new(Root::Symbol, 100, PointerSize::Bit32);
        assert_eq!(
            SCALAR.pointer_add(
                Value::<MockTarget>::Addr(address),
                Some(Value::Const(ConstValue::I64(3))),
                1,
                0
            ),
            Value::Unknown
        );
        assert_eq!(
            SCALAR.pointer_add(
                Value::Const(ConstValue::<MockTarget>::I64(3)),
                Some(Value::Addr(address)),
                1,
                0
            ),
            Value::Unknown
        );
    }

    /// A symbol constant is the address it names.
    #[test]
    fn a_symbol_is_its_address() {
        let op = SsaOp::Const {
            dest: var(0),
            value: ConstValue::Symbol(0x2040),
        };
        assert_eq!(
            SCALAR.evaluate(&op, None, known(&[])),
            Value::Addr(Address::new(Root::Symbol, 0x2040, PointerSize::Bit64))
        );
    }

    /// Narrowing an address loses it; keeping its width keeps it.
    #[test]
    fn only_a_pointer_wide_conversion_keeps_an_address() {
        for (target, bits, expected) in [
            (MockType::I32, 32, Value::StackDerived),
            (MockType::I64, 64, stack(-8)),
        ] {
            let op = SsaOp::IntConv {
                dest: var(0),
                operand: var(1),
                target,
                overflow_check: false,
                unsigned: true,
            };
            assert_eq!(
                SCALAR.evaluate(&op, Some(bits), known(&[(1, stack(-8))])),
                expected
            );
        }
    }

    /// An address and anything but a move, a distance or an equality is
    /// unknown.
    #[test]
    fn other_arithmetic_on_an_address_is_unknown() {
        let op = SsaOp::Xor {
            dest: var(0),
            left: var(1),
            right: var(2),
            flags: None,
        };
        let values = known(&[(1, stack(-8)), (2, Value::Const(ConstValue::I64(1)))]);
        assert_eq!(SCALAR.evaluate(&op, None, values), Value::StackDerived);
    }

    /// Two addresses under one root are equal exactly when their offsets are.
    #[test]
    fn equality_of_addresses_under_one_root() {
        let op = SsaOp::Ceq {
            dest: var(0),
            left: var(1),
            right: var(2),
            signaling: false,
        };
        let values = known(&[(1, stack(-8)), (2, stack(-8))]);
        assert_eq!(
            SCALAR.evaluate(&op, None, values),
            Value::Const(ConstValue::from_bool(true))
        );
    }

    /// Equality consumes each owned input once, including repeated SSA operands.
    #[test]
    fn equality_does_not_resolve_constant_operands_twice() {
        for same_operand in [false, true] {
            let reads = Cell::new(0);
            let op = SsaOp::Ceq {
                dest: var(0),
                left: var(1),
                right: var(if same_operand { 1 } else { 2 }),
                signaling: false,
            };
            let result = SCALAR.evaluate(&op, Some(1), |_| {
                reads.set(reads.get() + 1);
                Value::<MockTarget>::Const(ConstValue::I32(42))
            });
            assert_eq!(result, Value::Const(ConstValue::True));
            assert_eq!(reads.get(), 2, "same operand: {same_operand}");
        }
    }

    /// Address-aware fallback dispatch consumes constants only once per operand.
    #[test]
    fn pointer_width_fallbacks_reuse_resolved_inputs() {
        for (op, expected, count) in [
            (
                SsaOp::Add {
                    dest: var(0),
                    left: var(1),
                    right: var(2),
                    flags: None,
                },
                84,
                2,
            ),
            (
                SsaOp::Sub {
                    dest: var(0),
                    left: var(1),
                    right: var(2),
                    flags: None,
                },
                0,
                2,
            ),
            (
                SsaOp::Bitcast {
                    dest: var(0),
                    operand: var(1),
                    target: MockType::I64,
                },
                42,
                1,
            ),
        ] {
            let reads = Cell::new(0);
            let result = SCALAR.evaluate(&op, Some(64), |_| {
                reads.set(reads.get() + 1);
                Value::<MockTarget>::Const(ConstValue::I64(42))
            });
            assert_eq!(result, Value::Const(ConstValue::I64(expected)));
            assert_eq!(reads.get(), count, "{op}");
        }
    }

    /// Unsupported conversions retain the separate stack-provenance scan.
    #[test]
    fn declined_conversion_fallbacks_only_reread_for_stack_provenance() {
        for op in [
            SsaOp::IntConv {
                dest: var(0),
                operand: var(1),
                target: MockType::Unknown,
                overflow_check: false,
                unsigned: false,
            },
            SsaOp::IntToPtr {
                dest: var(0),
                operand: var(1),
                target: MockType::Unknown,
            },
            SsaOp::PtrToInt {
                dest: var(0),
                operand: var(1),
                target: MockType::Unknown,
            },
        ] {
            let reads = Cell::new(0);
            let result = SCALAR.evaluate(&op, Some(64), |_| {
                reads.set(reads.get() + 1);
                Value::<MockTarget>::Const(ConstValue::I64(42))
            });
            assert_eq!(
                result,
                Value::Unknown,
                "unknown target layouts decline conversion"
            );
            assert_eq!(
                reads.get(),
                2,
                "one semantic read and one provenance read: {op}"
            );
        }
    }

    /// A select on an unknown condition is what both arms agree on.
    #[test]
    fn an_undecided_select_joins_its_arms() {
        let op = SsaOp::Select {
            dest: var(0),
            condition: var(1),
            true_val: var(2),
            false_val: var(3),
        };
        let same = known(&[(2, stack(-8)), (3, stack(-8))]);
        assert_eq!(SCALAR.evaluate(&op, None, same), stack(-8));
        let differ = known(&[(2, stack(-8)), (3, stack(-16))]);
        assert_eq!(SCALAR.evaluate(&op, None, differ), Value::StackDerived);
    }

    /// Scaled offsets wrap under every root, including offsets wider than a pointer.
    #[test]
    fn pointer_arithmetic_normalizes_every_root_at_the_target_width() {
        let roots = [
            Root::Symbol,
            Root::Number,
            Root::Stack,
            Root::StackFrame(1),
            Root::Heap(1),
            Root::Opaque {
                activation: 0,
                value: var(7),
                run: 0,
            },
        ];
        for width in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
        ] {
            let scalar = Scalar::new(Machine {
                ptr_size: width,
                ..MACHINE
            });
            for root in roots {
                let base = Address::new(root, 0, width);
                let op = SsaOp::PtrAdd {
                    dest: var(0),
                    base: var(1),
                    index: Some(var(2)),
                    stride: u64::MAX,
                    offset: -1,
                    result_type: MockType::Ptr,
                };
                let values = known(&[
                    (1, Value::Addr(base)),
                    (2, Value::Const(ConstValue::U64(u64::MAX))),
                ]);
                assert_eq!(
                    scalar.evaluate(&op, None, values),
                    Value::Addr(base),
                    "{width:?} {root:?}"
                );
                if width.bits() < 64 {
                    let period = 1u64 << width.bits();
                    let values = known(&[
                        (1, Value::Addr(base)),
                        (2, Value::Const(ConstValue::U64(period))),
                    ]);
                    let op = SsaOp::PtrAdd {
                        dest: var(0),
                        base: var(1),
                        index: Some(var(2)),
                        stride: 1,
                        offset: 0,
                        result_type: MockType::Ptr,
                    };
                    assert_eq!(scalar.evaluate(&op, None, values), Value::Addr(base));
                }
            }
        }
    }

    /// Address arithmetic cannot turn an unrepresentable 128-bit result into a low address.
    #[test]
    fn wide_address_arithmetic_never_uses_a_sixty_four_bit_intermediate() {
        let scalar = Scalar::new(Machine {
            ptr_size: PointerSize::Bit128,
            ..MACHINE
        });
        let op = SsaOp::PtrAdd {
            dest: var(0),
            base: var(1),
            index: Some(var(2)),
            stride: 2,
            offset: 0,
            result_type: MockType::Ptr,
        };
        let values = known(&[
            (
                1,
                Value::Addr(Address::new(Root::Number, 0, PointerSize::Bit128)),
            ),
            (2, Value::Const(ConstValue::U64(1 << 63))),
        ]);
        assert_eq!(scalar.evaluate(&op, None, values), Value::Unknown);
        let op = SsaOp::Add {
            dest: var(0),
            left: var(1),
            right: var(2),
            flags: None,
        };
        let values = known(&[
            (
                1,
                Value::Addr(Address::new(Root::Number, -1, PointerSize::Bit128)),
            ),
            (2, Value::Const(ConstValue::U64(1))),
        ]);
        assert_eq!(scalar.evaluate(&op, None, values), Value::Unknown);
        let narrow = Scalar::new(MACHINE);
        let values = known(&[(1, stack(-8)), (2, Value::Const(ConstValue::I32(1)))]);
        assert_eq!(narrow.evaluate(&op, Some(32), values), Value::StackDerived);
    }

    /// Full-width modular intermediates can resolve to representable root-relative pointers.
    #[test]
    fn wide_pointer_intermediates_wrap_before_coordinate_admission() {
        let scalar = Scalar::new(Machine {
            ptr_size: PointerSize::Bit128,
            ..MACHINE
        });
        let wide = |raw| Value::Const(ConstValue::of_lane(128, raw).unwrap());
        let roots = [
            Root::Symbol,
            Root::Number,
            Root::Stack,
            Root::StackFrame(1),
            Root::Heap(1),
            Root::Opaque {
                activation: 0,
                value: var(1),
                run: 0,
            },
            Root::StackDerived {
                activation: 0,
                value: var(1),
                run: 0,
            },
        ];
        for root in roots {
            let address = |offset| Value::Addr(Address::new(root, offset, PointerSize::Bit128));
            let op = SsaOp::PtrAdd {
                dest: var(0),
                base: var(1),
                index: Some(var(2)),
                stride: 2,
                offset: 0,
                result_type: MockType::Ptr,
            };
            assert_eq!(
                scalar.evaluate(
                    &op,
                    None,
                    known(&[(1, address(8)), (2, wide(i128::MAX.cast_unsigned()))])
                ),
                address(6)
            );
            let add = SsaOp::Add {
                dest: var(0),
                left: var(1),
                right: var(2),
                flags: None,
            };
            let sub = SsaOp::Sub {
                dest: var(0),
                left: var(1),
                right: var(2),
                flags: None,
            };
            assert_eq!(
                scalar.evaluate(&add, None, known(&[(1, address(8)), (2, wide(u128::MAX))])),
                address(7)
            );
            assert_eq!(
                scalar.evaluate(&sub, None, known(&[(1, address(8)), (2, wide(u128::MAX))])),
                address(9)
            );
            let (last, first) = if root.unsigned_offsets() {
                (-1, 0)
            } else {
                (i64::MAX, i64::MIN)
            };
            assert_eq!(
                scalar.evaluate(
                    &sub,
                    None,
                    known(&[(1, address(last)), (2, address(first))])
                ),
                wide(u128::from(u64::MAX))
            );
            assert_eq!(
                scalar.evaluate(
                    &sub,
                    None,
                    known(&[(1, address(first)), (2, address(last))])
                ),
                wide(0u128.wrapping_sub(u128::from(u64::MAX)))
            );
        }
        let op = SsaOp::PtrAdd {
            dest: var(0),
            base: var(1),
            index: Some(var(2)),
            stride: 2,
            offset: 0,
            result_type: MockType::Ptr,
        };
        assert_eq!(
            scalar.evaluate(
                &op,
                None,
                known(&[(1, wide(8)), (2, wide(i128::MAX.cast_unsigned()))])
            ),
            wide(6)
        );
        assert_eq!(
            scalar.evaluate(&op, None, known(&[(1, wide(8)), (2, wide(1u128 << 100))])),
            wide((1u128 << 101) + 8)
        );
        assert_eq!(
            scalar.evaluate(
                &op,
                None,
                known(&[
                    (
                        1,
                        Value::Addr(Address::new(Root::Number, 8, PointerSize::Bit128))
                    ),
                    (2, wide(1u128 << 100))
                ])
            ),
            Value::Unknown
        );
    }
    /// Integer folds outside pointer dispatch resolve each operand once without a precheck.
    #[test]
    fn default_integer_folds_do_not_resolve_or_copy_inputs_twice() {
        let cases = [
            (
                SsaOp::Mul {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                2,
                6,
            ),
            (
                SsaOp::Xor {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                2,
                1,
            ),
            (
                SsaOp::Add {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                2,
                5,
            ),
            (
                SsaOp::Shl {
                    dest: var(9),
                    value: var(0),
                    amount: var(1),
                    flags: None,
                },
                2,
                12,
            ),
            (
                SsaOp::Neg {
                    dest: var(9),
                    operand: var(0),
                    flags: None,
                },
                1,
                -3,
            ),
            (
                SsaOp::Not {
                    dest: var(9),
                    operand: var(0),
                    flags: None,
                },
                1,
                -4,
            ),
        ];
        for (op, expected_reads, expected) in cases {
            let reads = Cell::new(0);
            let value = SCALAR.evaluate(&op, Some(32), |variable| {
                reads.set(reads.get() + 1);
                Value::<MockTarget>::Const(ConstValue::I32(if variable == var(0) { 3 } else { 2 }))
            });
            assert_eq!(value, Value::Const(ConstValue::I32(expected)));
            assert_eq!(reads.get(), expected_reads, "{op}");
        }
    }

    /// Finding a stack pointer stops before an arbitrarily long untouched operand tail.
    #[test]
    fn stack_provenance_scans_stop_at_the_first_stack_reference() {
        let op = SsaOp::<MockTarget>::VectorExtendInLane(Box::new(VectorExtendInLaneData {
            signed: false,
            source_bits: 8,
            element_bits: 32,
            outputs: vec![var(0)],
            inputs: (1..8192).map(var).collect(),
        }));

        let reads = Cell::new(0);
        assert!(SCALAR.references_stack(&op, &|_| {
            reads.set(reads.get() + 1);
            stack(-8)
        }));
        assert_eq!(reads.get(), 1);
    }

    /// Empty exception sets retain the declared wide status width without reading operands.
    #[test]
    fn float_raise_status_retains_wide_register() {
        let op = SsaOp::float_raise(
            var(2),
            SsaOp::Add {
                dest: var(2),
                left: var(0),
                right: var(1),
                flags: None,
            },
            None,
            FloatExceptions::NONE,
            FloatExceptionLayout::IEEE,
        );
        let value = SCALAR
            .evaluate::<MockTarget>(&op, Some(128), |_| panic!("no numeric input is demanded"));
        assert_eq!(
            value
                .constant()
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size)),
            Some((0, 128))
        );
    }

    /// Builds nested exception probes without recursive construction.
    fn nested_stack_probe(depth: usize, mask: bool, large_outputs: bool) -> SsaOp<MockTarget> {
        let mut operation = SsaOp::VectorExtendInLane(Box::new(VectorExtendInLaneData {
            signed: false,
            source_bits: 8,
            element_bits: 32,
            outputs: (0..if large_outputs { 8192 } else { 1 }).map(var).collect(),
            inputs: vec![var(8192)],
        }));
        for _ in 0..depth {
            operation = SsaOp::FloatRaise {
                dest: var(0),
                op: Box::new(operation),
                lanes: mask.then(|| var(8193)),
                raises: FloatExceptions::INEXACT,
                layout: FloatExceptionLayout::IEEE,
            };
        }
        operation
    }

    /// Releases nested probes iteratively before asserting results.
    fn drop_stack_probe(mut operation: SsaOp<MockTarget>) {
        loop {
            operation = match operation {
                SsaOp::FloatRaise { op, .. } => *op,
                _ => break,
            };
        }
    }

    /// A complete nested probe finds a stack input before reading later mask operands.
    #[test]
    fn nested_raise_stack_probes_stop_at_the_first_stack_input() {
        let operation = nested_stack_probe(1024, true, false);

        let reads = Cell::new(0);
        let result = SCALAR.references_stack(&operation, &|_| {
            reads.set(reads.get() + 1);
            stack(-8)
        });
        drop_stack_probe(operation);
        assert!(result);
        assert_eq!(reads.get(), 1);
    }

    /// Partial proofs retain exact wide results and possible private-stack aliases.
    #[test]
    fn partial_integer_proofs_keep_stack_provenance() {
        let dest = Destination {
            bits: Some(128),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(128, false),
        };
        let zero = ConstValue::<MockTarget>::of_lane(128, 0).unwrap();
        let op = SsaOp::And {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        };
        let inputs = |id| {
            if id == var(0) {
                Value::StackDerived
            } else {
                Value::Const(zero.clone())
            }
        };
        let result = SCALAR
            .evaluate_with_operand_layouts(&op, dest, inputs, |_| OperandLayout::Register(dest));
        assert_eq!(
            result
                .constant()
                .and_then(|value| value.scalar_integer_pattern(MACHINE.ptr_size)),
            Some((0, 128))
        );
        assert!(!result.may_reference_stack());
        let partial = |id| {
            if id == var(0) {
                Value::StackDerived
            } else {
                Value::partial_integer(IntegerBits::new(128, 15, 0).unwrap(), false)
            }
        };
        let result = SCALAR
            .evaluate_with_operand_layouts(&op, dest, partial, |_| OperandLayout::Register(dest));
        assert!(result.may_reference_stack());
        assert!(
            result
                .integer_bits()
                .is_some_and(|bits| bits.zeros() == 15 && bits.ones() == 0)
        );
    }

    /// Wide partial results preserve complete masks and private-stack aliases.
    #[test]
    fn partial_integer_shared_storage_preserves_complete_patterns() {
        let dest = Destination {
            bits: Some(128),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(128, false),
        };
        let ends = 1u128 | (1u128 << 127);
        let mask = ConstValue::<MockTarget>::of_lane(128, !ends).unwrap();
        let op = SsaOp::And {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        };
        let inputs = |id| {
            if id == var(0) {
                Value::StackDerived
            } else {
                Value::Const(mask.clone())
            }
        };
        let value = SCALAR
            .evaluate_with_operand_layouts(&op, dest, inputs, |_| OperandLayout::Register(dest));
        assert_eq!(value.integer_bits(), IntegerBits::new(128, ends, 0));
        assert!(value.may_reference_stack());
    }
    /// Supplies a root contract without asserting the absolute physical base.
    struct RootLowBits;

    impl StackCongruences for RootLowBits {
        fn stack_congruence(&self, root: Root) -> Option<StackCongruence> {
            if root == Root::Stack {
                StackCongruence::from_low_bits(4, 8)
            } else {
                None
            }
        }
    }

    /// Coordinate reconstruction needs fully matching scalar declarations and never turns zero into a stack address.
    #[test]
    fn stack_mask_requires_exact_integer_layouts_and_preserves_numeric_zero() {
        let op = SsaOp::And {
            dest: var(0),
            left: var(1),
            right: var(2),
            flags: None,
        };
        let wide = Destination {
            bits: Some(64),
            integer: true,
            integer_type: IntegerType::new(64, true),
            ..Destination::default()
        };
        let scalar = SCALAR.with_stack_congruences(&RootLowBits);
        let values = || known(&[(1, stack(-8)), (2, Value::Const(ConstValue::I64(-16)))]);
        assert_eq!(
            scalar.evaluate_with_operand_layouts(&op, wide, values(), |_| OperandLayout::Register(
                wide
            )),
            stack(-8)
        );
        for destination in [
            Destination::default(),
            Destination {
                integer_type: None,
                ..wide
            },
            Destination {
                lanes: Some((64, 1)),
                ..wide
            },
            Destination {
                bits: Some(32),
                ..wide
            },
            Destination {
                integer: false,
                ..wide
            },
        ] {
            assert!(!matches!(
                scalar.evaluate_with_operand_layouts(&op, destination, values(), |_| {
                    OperandLayout::Register(wide)
                }),
                Value::Addr(_)
            ));
        }
        for operand in [
            OperandLayout::default(),
            OperandLayout::Boolean,
            OperandLayout::Float(FloatFormat::Double),
            OperandLayout::Register(Destination {
                integer_type: None,
                ..wide
            }),
            OperandLayout::Register(Destination {
                lanes: Some((64, 1)),
                ..wide
            }),
            OperandLayout::Register(Destination {
                bits: Some(32),
                ..wide
            }),
        ] {
            assert!(!matches!(
                scalar.evaluate_with_operand_layouts(&op, wide, values(), |_| operand),
                Value::Addr(_)
            ));
        }
        let zero = scalar.evaluate_with_operand_layouts(
            &op,
            wide,
            known(&[(1, stack(-8)), (2, Value::Const(ConstValue::I64(0)))]),
            |_| OperandLayout::Register(wide),
        );
        assert_eq!(zero.constant().and_then(ConstValue::as_u64_bits), Some(0));
        assert!(zero.address().is_none());
        // With no low bits stated, the mask names the realigned frame's base.
        let unaligned = SCALAR
            .evaluate_with_operand_layouts(&op, wide, values(), |_| OperandLayout::Register(wide));
        assert_eq!(
            unaligned.address(),
            Some(Address::new(
                Root::StackAligned {
                    frame: None,
                    at: -8,
                    bits: 4,
                },
                0,
                PointerSize::Bit64,
            ))
        );
    }
    /// A known carry survives unknown surrounding flags without inventing the zero flag.
    #[test]
    fn a_partial_flag_value_decides_only_its_established_boolean_bit() {
        let var = SsaVarId::from_index;
        let word = Destination {
            bits: Some(32),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(32, false),
        };
        let scalar = Scalar::new(MACHINE);
        let insert = SsaOp::<MockTarget>::InsertFlag {
            dest: var(2),
            base: var(0),
            bit: var(1),
            mask: FlagsMask::CARRY,
        };
        let flags = scalar.evaluate_with_operand_layouts(
            &insert,
            word,
            |id| {
                if id == var(1) {
                    Value::Const(ConstValue::from_bool(true))
                } else {
                    Value::Unknown
                }
            },
            |id| {
                if id == var(1) {
                    OperandLayout::Boolean
                } else {
                    OperandLayout::Register(word)
                }
            },
        );
        assert_eq!(
            flags.integer_bits().unwrap().ones(),
            u128::from(FlagsMask::CARRY.bits())
        );
        let boolean = Destination {
            bits: Some(1),
            ..Destination::default()
        };
        for (mask, expected) in [(FlagsMask::CARRY, Some(true)), (FlagsMask::ZERO, None)] {
            let read = SsaOp::ReadFlags {
                dest: var(3),
                flags: var(2),
                mask,
            };
            let value = scalar.evaluate_with_operand_layouts(
                &read,
                boolean,
                |_| flags.clone(),
                |_| OperandLayout::Register(word),
            );
            assert_eq!(value.constant().and_then(ConstValue::as_bool), expected);
        }
    }

    /// A declared loader base gives numeric masks their actual physical bits.
    #[test]
    fn mapped_address_masks_use_physical_bits_at_the_declared_native_width() {
        for pointer in [PointerSize::Bit32, PointerSize::Bit64] {
            let machine = Machine {
                ptr_size: pointer,
                ..MACHINE
            };
            let dest = Destination {
                bits: Some(pointer.bits()),
                lanes: None,
                lane_format: None,
                integer: true,
                integer_type: IntegerType::new(pointer.bits(), false),
            };
            for (root, mapping, expected) in [
                (Root::Symbol, Some(0), 0x2080),
                (Root::Symbol, Some(-0x400000), 0x402080),
                (Root::Symbol, Some(0x1000), 0x1080),
                (Root::Number, None, 0x2080),
                (Root::Number, Some(-0x400000), 0x2080),
            ] {
                let op = SsaOp::And {
                    dest: var(2),
                    left: var(0),
                    right: var(1),
                    flags: None,
                };
                let mask = ConstValue::<MockTarget>::of_lane(
                    pointer.bits(),
                    pointer.mask_unsigned_128(u128::MAX) ^ 1,
                )
                .unwrap();
                let result = Scalar::new(machine)
                    .with_image(Some(&Offset(mapping)))
                    .evaluate_with_operand_layouts(
                        &op,
                        dest,
                        |id| {
                            if id == var(0) {
                                Value::Addr(Address::new(root, 0x2081, pointer))
                            } else {
                                Value::Const(mask.clone())
                            }
                        },
                        |_| OperandLayout::Register(dest),
                    );
                assert_eq!(
                    result.constant().and_then(
                        |constant| constant.scalar_pattern_at_width(pointer.bits(), pointer)
                    ),
                    Some(expected),
                    "{pointer:?} {root:?} {mapping:?}"
                );
            }
        }
    }

    /// Missing bases and object-relative offsets never fabricate physical mask results.
    #[test]
    fn mapped_address_masks_require_base_authority_and_complete_integer_layouts() {
        let native = Destination {
            bits: Some(64),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(64, false),
        };
        let narrow = Destination {
            bits: Some(32),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(32, false),
        };
        let op = SsaOp::And {
            dest: var(2),
            left: var(0),
            right: var(1),
            flags: None,
        };
        for (root, mapping, layout) in [
            (Root::Symbol, None, OperandLayout::Register(native)),
            (Root::Stack, Some(0), OperandLayout::Register(native)),
            (
                Root::StackFrame(3),
                Some(0),
                OperandLayout::Register(native),
            ),
            (Root::Heap(7), Some(0), OperandLayout::Register(native)),
            (
                Root::Opaque {
                    activation: 0,
                    value: var(3),
                    run: 1,
                },
                Some(0),
                OperandLayout::Register(native),
            ),
            (Root::Symbol, Some(0), OperandLayout::Register(narrow)),
            (
                Root::Symbol,
                Some(0),
                OperandLayout::Register(Destination {
                    bits: Some(64),
                    ..Destination::default()
                }),
            ),
            (
                Root::Symbol,
                Some(0),
                OperandLayout::Float(FloatFormat::Double),
            ),
        ] {
            let result = Scalar::new(MACHINE)
                .with_image(Some(&Offset(mapping)))
                .evaluate_with_operand_layouts(
                    &op,
                    native,
                    |id| {
                        if id == var(0) {
                            Value::<MockTarget>::Addr(Address::new(
                                root,
                                0x2081,
                                PointerSize::Bit64,
                            ))
                        } else {
                            Value::Const(ConstValue::U64(!1))
                        }
                    },
                    |id| {
                        if id == var(0) {
                            layout
                        } else {
                            OperandLayout::Register(native)
                        }
                    },
                );
            assert!(
                result.constant().is_none(),
                "{root:?} {mapping:?} {layout:?}: {result:?}"
            );
        }
    }

    /// A mapped narrowing conversion computes bytes while whole-word conversions keep identity.
    #[test]
    fn mapped_address_conversion_truncates_known_bits_and_preserves_whole_word_identity() {
        let image = Offset(Some(-0x400000));
        let scalar = Scalar::new(MACHINE).with_image(Some(&image));
        let native = Destination {
            bits: Some(64),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(64, false),
        };
        let narrow = Destination {
            bits: Some(32),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(32, false),
        };
        let address = Address::new(Root::Symbol, 0x2081, PointerSize::Bit64);
        let convert = SsaOp::PtrToInt {
            dest: var(0),
            operand: var(1),
            target: MockType::U32,
        };
        let result = scalar.evaluate_with_operand_layouts(
            &convert,
            narrow,
            |_| Value::<MockTarget>::Addr(address),
            |_| OperandLayout::Register(native),
        );
        assert_eq!(
            result.constant().and_then(ConstValue::as_u64),
            Some(0x402081)
        );
        let pointer = OperandLayout::Register(Destination {
            bits: Some(64),
            ..Destination::default()
        });
        let pointer_result = scalar.evaluate_with_operand_layouts(
            &convert,
            narrow,
            |_| Value::<MockTarget>::Addr(address),
            |_| pointer,
        );
        assert_eq!(
            pointer_result.constant().and_then(ConstValue::as_u64),
            Some(0x402081)
        );
        let preserve = SsaOp::IntConv {
            dest: var(0),
            operand: var(1),
            target: MockType::U64,
            unsigned: true,
            overflow_check: false,
        };
        assert_eq!(
            scalar.evaluate_with_operand_layouts(
                &preserve,
                native,
                |_| Value::<MockTarget>::Addr(address),
                |_| OperandLayout::Register(native)
            ),
            Value::Addr(address)
        );
        let unknown = Scalar::new(MACHINE).evaluate_with_operand_layouts(
            &convert,
            narrow,
            |_| Value::<MockTarget>::Addr(address),
            |_| OperandLayout::Register(native),
        );
        assert!(unknown.constant().is_none());
    }
}
