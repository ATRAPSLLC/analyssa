//! Atomic values and bounded private-memory effects with retained ordering.
//!
//! Only a complete unescaped private span is interpreted. Observable accesses
//! refuse the invocation, and the stepper retains every atomic/fence effect in
//! its trace. Weak comparisons and exclusive stores join possible writes;
//! neither instruction order nor an exclusive load guarantees hardware success.

use crate::{
    analysis::{
        consts::{Destination, OperandLayout},
        execution::{
            access::{Access, AccessError},
            memory::{MemoryRefusal, Transfer},
            scalar::Scalar,
            state::State,
            value::{Address, Value},
        },
    },
    ir::{
        evaluation::{ArithmeticTrap, Evaluation},
        float::FloatFormat,
        integer::{BitWidth, IntegerType},
        lanes::VectorConst,
        ops::{
            AccessPermission, AtomicRmwOp, AtomicStatus, BarrierOp, CmpKind, CompareCondition,
            FlagCondition, RcwCheck, RcwProtection, SsaOp, SystemOpKind, VectorBinaryKind,
            VectorElement, VectorElementKind,
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    target::{Endianness, Target, TypeLayout},
};

/// Operand positions local to an arithmetic query, never added to the SSA body.
const LEFT: SsaVarId = SsaVarId::from_index(0);
/// The second local arithmetic operand.
const RIGHT: SsaVarId = SsaVarId::from_index(1);
/// The local query's unused result identity.
const RESULT: SsaVarId = SsaVarId::from_index(2);

/// Interprets atomic operations through shared value and memory kernels.
pub struct Atomics<'a> {
    /// Target memory.
    access: Access<'a>,
    /// The same arithmetic evaluator used by ordinary SSA operations.
    scalar: Scalar<'a>,
}

impl<'a> Atomics<'a> {
    /// Shares the current instruction's value and memory rules.
    pub const fn new(access: Access<'a>, scalar: Scalar<'a>) -> Self {
        Self { access, scalar }
    }

    /// Executes an atomic/fence operation, or reports that `op` is another family.
    ///
    /// No refusal publishes a partial invocation. The stepper taints effect
    /// proofs before entering this kernel, including deterministic accesses.
    pub fn execute<T: Target>(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
        layout: impl Fn(SsaVarId) -> Option<TypeLayout>,
    ) -> Result<bool, AccessError> {
        if let SsaOp::SystemOp(data) = op
            && data.kind == SystemOpKind::Barrier(BarrierOp::ClearExclusive)
        {
            state.set_reservation_possible(false);
            return Ok(true);
        }
        if matches!(op, SsaOp::Fence { .. }) {
            state.memory_mut().clobber();
            return Ok(true);
        }
        let (addr, width, space) = match op {
            SsaOp::AtomicLoad {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicStore {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicStoreConditional {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicPairLoad {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicPairStoreConditional {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicExchange {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicLockRmw {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicCondRmw {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicCmpXchg {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicPairCmpXchg {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicPairRmw {
                addr,
                width,
                access,
                ..
            }
            | SsaOp::AtomicCheckedRmw {
                addr,
                width,
                access,
                ..
            } => (*addr, *width, *access),
            _ => return Ok(false),
        };
        let bits = width.bits().unwrap_or(self.access.ptr_size().bits());
        // The integer word of the access's width, which only the arithmetic,
        // the compares and an access of no stated type read: a 64-byte block
        // loads and stores as the vector it is typed, and no integer is that
        // wide.
        let word = self.word(bits, false);
        let bytes = usize::try_from(bits / 8).map_err(|_| MemoryRefusal::AtomicContract)?;
        let pointer = state.value(addr);
        let address = self
            .access
            .address_of(&pointer, space.address_space)
            .ok_or(MemoryRefusal::AtomicContract)?;
        // A store only writes; every other atomic reads what it may write.
        let transfer = if op.memory_effect().is_some_and(|effect| !effect.reads) {
            Transfer::Write
        } else {
            Transfer::Read
        };
        if let Some(reason) = state.memory().admit(address, bytes, transfer) {
            return Err(reason.into());
        }
        // An atomic load that opens no reservation, of memory no observer can
        // write, reads what is there as a plain load does: read-only image
        // storage (`ld64b` or `ldar` of a constant table).
        let reads_immutable = matches!(
            op,
            SsaOp::AtomicLoad {
                reserves: false,
                ..
            }
        ) && self.access.read_only_image(state.memory(), address, bytes);
        if space.volatile
            || space.permission != AccessPermission::Current
            || !(state.memory().private_span(address, bytes) || reads_immutable)
        {
            return Err(MemoryRefusal::AtomicContract.into());
        }
        match op {
            SsaOp::AtomicLoad {
                dest,
                value_type,
                reserves,
                ..
            } => {
                let ty = self.typed::<T>(value_type, bits)?;
                let value = self.load(state, address, ty)?;
                state.define(*dest, value);
                if *reserves {
                    state.set_reservation_possible(true);
                }
            }
            SsaOp::AtomicStore {
                value, value_type, ..
            } => {
                let ty = self.typed::<T>(value_type, bits)?;
                let value = state.value(*value);
                self.store(state, address, &value, ty, Some(true))?;
            }
            SsaOp::AtomicExchange { dest, value, .. }
            | SsaOp::AtomicLockRmw { dest, value, .. } => {
                let kind = match op {
                    SsaOp::AtomicLockRmw { op, .. } => *op,
                    _ => AtomicRmwOp::Xchg,
                };
                let ty = self.rmw_layout(kind, layout(*value).or_else(|| layout(*dest)), word?)?;
                let before = self.load(state, address, ty)?;
                let operand = state.value(*value);
                let after = self.rmw(kind, &before, &operand, ty);
                self.store(state, address, &after, ty, Some(true))?;
                state.define(*dest, before);
                state.set_reservation_possible(true);
            }
            SsaOp::AtomicCondRmw {
                old,
                compare,
                value,
                op,
                condition,
                ..
            } => {
                let before = self.load(state, address, word?)?;
                let compared = state.value(*compare);
                let operand = state.value(*value);
                let selected = self.condition(*condition, &before, &compared, word?);
                let after = self.rmw(*op, &before, &operand, word?);
                self.store(state, address, &after, word?, selected)?;
                state.define(*old, before);
                state.set_reservation_possible(true);
            }
            SsaOp::AtomicCmpXchg {
                old,
                success,
                expected,
                desired,
                weak,
                ..
            } => {
                let ty = self.choose(layout(*expected).or_else(|| layout(*old)), word?)?;
                let before = self.load(state, address, ty)?;
                let compared = state.value(*expected);
                let desired = state.value(*desired);
                let equal = self.equal_bits(&before, &compared, ty);
                let selected = if *weak && equal == Some(true) {
                    None
                } else {
                    equal
                };
                self.store(state, address, &desired, ty, selected)?;
                state.define(*old, before);
                if let Some(success) = success {
                    let result = selected.map_or(Value::Unknown, |value| {
                        Value::Const(ConstValue::from_bool(value))
                    });
                    state.define(*success, result);
                }
                state.set_reservation_possible(true);
            }
            SsaOp::AtomicStoreConditional {
                status,
                status_encoding,
                consumes_reservation,
                value,
                value_type,
                ..
            } => {
                let ty = self.typed::<T>(value_type, bits)?;
                let desired = state.value(*value);
                let possible = state.reservation_possible();
                self.store(
                    state,
                    address,
                    &desired,
                    ty,
                    if possible { None } else { Some(false) },
                )?;
                self.status(state, *status, possible, *status_encoding, layout(*status))?;
                state.set_reservation_possible(possible && !*consumes_reservation);
            }
            SsaOp::AtomicPairLoad {
                first,
                second,
                first_type,
                second_type,
                ..
            } => {
                let (first_ty, second_ty) = (
                    self.typed::<T>(first_type, bits / 2)?,
                    self.typed::<T>(second_type, bits / 2)?,
                );
                let next = self.offset(address, bytes / 2)?;
                let first_value = self.load(state, address, first_ty)?;
                let second_value = self.load(state, next, second_ty)?;
                state.define(*first, first_value);
                state.define(*second, second_value);
                state.set_reservation_possible(true);
            }
            SsaOp::AtomicPairStoreConditional {
                status,
                status_encoding,
                consumes_reservation,
                first_value,
                second_value,
                first_type,
                second_type,
                ..
            } => {
                let (first_ty, second_ty) = (
                    self.typed::<T>(first_type, bits / 2)?,
                    self.typed::<T>(second_type, bits / 2)?,
                );
                let next = self.offset(address, bytes / 2)?;
                let first = state.value(*first_value);
                let second = state.value(*second_value);
                let possible = state.reservation_possible();
                let selected = if possible { None } else { Some(false) };
                self.store(state, address, &first, first_ty, selected)?;
                self.store(state, next, &second, second_ty, selected)?;
                self.status(state, *status, possible, *status_encoding, layout(*status))?;
                state.set_reservation_possible(possible && !*consumes_reservation);
            }
            SsaOp::AtomicPairCmpXchg {
                old_first,
                old_second,
                expected_first,
                expected_second,
                desired_first,
                desired_second,
                weak,
                ..
            } => {
                let half = self.word(bits / 2, false)?;
                let first_ty =
                    self.choose(layout(*expected_first).or_else(|| layout(*old_first)), half)?;
                let second_ty = self.choose(
                    layout(*expected_second).or_else(|| layout(*old_second)),
                    half,
                )?;
                let next = self.offset(address, bytes / 2)?;
                let before_first = self.load(state, address, first_ty)?;
                let before_second = self.load(state, next, second_ty)?;
                let expected_first = state.value(*expected_first);
                let expected_second = state.value(*expected_second);
                let desired_first = state.value(*desired_first);
                let desired_second = state.value(*desired_second);
                let equal = match (
                    self.equal_bits(&before_first, &expected_first, first_ty),
                    self.equal_bits(&before_second, &expected_second, second_ty),
                ) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) if !weak => Some(true),
                    _ => None,
                };
                self.store(state, address, &desired_first, first_ty, equal)?;
                self.store(state, next, &desired_second, second_ty, equal)?;
                state.define(*old_first, before_first);
                state.define(*old_second, before_second);
                state.set_reservation_possible(true);
            }
            SsaOp::AtomicPairRmw {
                old_first,
                old_second,
                value_first,
                value_second,
                op,
                ..
            } => {
                let half = self.word(bits / 2, false)?;
                let second_address = self.offset(address, bytes / 2)?;
                let old_lower = self.load(state, address, half)?;
                let old_higher = self.load(state, second_address, half)?;
                let first = state.value(*value_first);
                let second = state.value(*value_second);
                if matches!(
                    op,
                    AtomicRmwOp::Xchg
                        | AtomicRmwOp::And
                        | AtomicRmwOp::Or
                        | AtomicRmwOp::Xor
                        | AtomicRmwOp::AndNot
                ) {
                    let lower = self.rmw(*op, &old_lower, &first, half);
                    let higher = self.rmw(*op, &old_higher, &second, half);
                    self.store(state, address, &lower, half, Some(true))?;
                    self.store(state, second_address, &higher, half, Some(true))?;
                } else {
                    if op.is_float() {
                        return Err(MemoryRefusal::AtomicContract.into());
                    }
                    // A carry runs from the low-order half, which the byte
                    // order places at the lower address or the higher one.
                    let ((low, high), (operand_low, operand_high)) = match self.access.byte_order()
                    {
                        Some(Endianness::Little) => ((&old_lower, &old_higher), (&first, &second)),
                        Some(Endianness::Big) => ((&old_higher, &old_lower), (&second, &first)),
                        None => return Err(MemoryRefusal::AtomicContract.into()),
                    };
                    let before = self.combine(low, high, bits / 2);
                    let operand = self.combine(operand_low, operand_high, bits / 2);
                    let after = self.rmw(*op, &before, &operand, word?);
                    self.store(state, address, &after, word?, Some(true))?;
                }
                state.define(*old_first, old_lower);
                state.define(*old_second, old_higher);
                state.set_reservation_possible(true);
            }
            SsaOp::AtomicCheckedRmw {
                old,
                status,
                compare,
                value,
                op,
                rcw_mask,
                rcws_mask,
                protection,
                ..
            } => {
                // Whether a 64-bit form's protected bit is in use is the
                // regime's configuration, which the program does not state.
                if *protection == RcwProtection::Regime {
                    return Err(MemoryRefusal::AtomicContract.into());
                }
                let word = word?;
                let ptr_size = self.access.ptr_size();
                let bits_of = |value: &Value<T>| {
                    value
                        .constant()
                        .and_then(|constant| constant.scalar_integer_pattern(ptr_size))
                        .map(|(pattern, _)| pattern)
                        .ok_or(MemoryRefusal::AtomicContract)
                };
                let before = self.load(state, address, word)?;
                let old_bits = bits_of(&before)?;
                let operand = bits_of(&state.value(*value))?;
                let checked_mask = bits_of(&state.value(*rcw_mask))?;
                let soft_mask = match rcws_mask {
                    Some(mask) => Some(bits_of(&state.value(*mask))?),
                    None => None,
                };
                let all = BitWidth::new(bits).mask_u128();
                let (new_bits, compared) = match compare {
                    Some(compare) => {
                        let expected = bits_of(&state.value(*compare))?;
                        if expected == old_bits {
                            (operand & all, true)
                        } else {
                            (old_bits, false)
                        }
                    }
                    None => match op {
                        AtomicRmwOp::AndNot => (old_bits & !operand & all, true),
                        AtomicRmwOp::Or => ((old_bits | operand) & all, true),
                        AtomicRmwOp::Xchg => (operand & all, true),
                        _ => return Err(MemoryRefusal::AtomicContract.into()),
                    },
                };
                let nzcv = if compared {
                    RcwCheck {
                        bits,
                        soft: soft_mask.is_some(),
                        protected: true,
                    }
                    .status(
                        old_bits,
                        new_bits,
                        checked_mask,
                        soft_mask.unwrap_or(0),
                    )
                } else {
                    RcwCheck::COMPARE_FAILED
                };
                // A failed check leaves the write to the implementation, which
                // may make it anyway: where that would change memory, memory is
                // one of two values no state names.
                if nzcv != RcwCheck::PASSED && new_bits != old_bits {
                    return Err(MemoryRefusal::AtomicContract.into());
                }
                if nzcv == RcwCheck::PASSED {
                    let integer = word
                        .integer_conversion_type()
                        .ok_or(MemoryRefusal::AtomicContract)?;
                    let after = integer
                        .constant(new_bits, ptr_size)
                        .map_or(Value::Unknown, Value::Const);
                    self.store(state, address, &after, word, Some(true))?;
                }
                let flags = layout(*status)
                    .filter(|layout| !layout.is_unknown())
                    .unwrap_or(self.word(8, false)?)
                    .integer_conversion_type()
                    .ok_or(MemoryRefusal::AtomicContract)?;
                state.define(
                    *status,
                    flags
                        .constant(u128::from(nzcv), ptr_size)
                        .map_or(Value::Unknown, Value::Const),
                );
                state.define(*old, before);
                state.set_reservation_possible(true);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Resolves an attested word without querying or manufacturing a host type.
    fn word(&self, bits: u32, signed: bool) -> Result<TypeLayout, MemoryRefusal> {
        if !matches!(bits, 8 | 16 | 32 | 64 | 128) {
            return Err(MemoryRefusal::AtomicContract);
        }
        IntegerType::new(bits, signed)
            .map(TypeLayout::integer)
            .ok_or(MemoryRefusal::AtomicContract)
    }

    /// Resolves an explicit access type, retaining a physical width for unknown language types.
    fn typed<T: Target>(&self, ty: &T::Type, bits: u32) -> Result<TypeLayout, MemoryRefusal> {
        match self
            .access
            .type_layout::<T>(ty)
            .filter(|layout| !layout.is_unknown())
        {
            Some(layout) => (layout.storage_bytes(self.access.ptr_size())
                == usize::try_from(bits / 8).ok())
            .then_some(layout)
            .ok_or(MemoryRefusal::AtomicContract),
            None => self.word(bits, false),
        }
    }

    /// Checks exact storage extent before interpreting a scalar carrier.
    fn choose(
        &self,
        layout: Option<TypeLayout>,
        word: TypeLayout,
    ) -> Result<TypeLayout, MemoryRefusal> {
        let layout = layout.filter(|layout| !layout.is_unknown()).unwrap_or(word);
        if layout.storage_bytes(self.access.ptr_size())
            != word.storage_bytes(self.access.ptr_size())
        {
            return Err(MemoryRefusal::AtomicContract);
        }
        Ok(layout)
    }

    /// Requires the actual float format for floating atomic arithmetic.
    fn rmw_layout(
        &self,
        kind: AtomicRmwOp,
        layout: Option<TypeLayout>,
        word: TypeLayout,
    ) -> Result<TypeLayout, MemoryRefusal> {
        if kind.is_float() {
            let layout = self.choose(layout, word)?;
            if layout.float_format().is_none() {
                return Err(MemoryRefusal::AtomicContract);
            }
            Ok(layout)
        } else if kind == AtomicRmwOp::Xchg {
            self.choose(layout, word)
        } else {
            Ok(word)
        }
    }

    /// Reads the span before any result definition overwrites a source.
    fn load<T: Target>(
        &self,
        state: &State<T>,
        address: Address,
        layout: TypeLayout,
    ) -> Result<Value<T>, MemoryRefusal> {
        self.access.load_layout(state.memory(), address, layout)
    }

    /// Writes or joins a possible write without cloning unrelated memory.
    fn store<T: Target>(
        &self,
        state: &mut State<T>,
        address: Address,
        value: &Value<T>,
        layout: TypeLayout,
        selected: Option<bool>,
    ) -> Result<(), AccessError> {
        match selected {
            Some(false) => {}
            Some(true) => self.access.store_layout(
                state.memory_mut(),
                &Value::Addr(address),
                value,
                layout,
                None,
            )?,
            None => self
                .access
                .maybe_store(state.memory_mut(), address, value, layout)?,
        }
        if let Some(reason) = state.memory().refusal() {
            return Err(reason.into());
        }
        Ok(())
    }

    /// Moves within the already checked pair footprint.
    fn offset(&self, address: Address, bytes: usize) -> Result<Address, MemoryRefusal> {
        address
            .moved(
                i64::try_from(bytes).map_err(|_| MemoryRefusal::AddressRange)?,
                self.access.ptr_size(),
            )
            .ok_or(MemoryRefusal::AddressRange)
    }

    /// Projects possible success and mandatory failure through the instruction's encoding.
    fn status<T: Target>(
        &self,
        state: &mut State<T>,
        status: SsaVarId,
        possible: bool,
        encoding: AtomicStatus,
        layout: Option<TypeLayout>,
    ) -> Result<(), MemoryRefusal> {
        let layout = layout
            .filter(|layout| !layout.is_unknown())
            .unwrap_or(self.word(32, false)?);
        let integer = layout
            .integer_conversion_type()
            .ok_or(MemoryRefusal::AtomicContract)?;
        let (success, failure) = match encoding {
            AtomicStatus::Unspecified => (None, None),
            AtomicStatus::ZeroNonzero => (Some(0), None),
            AtomicStatus::ZeroOne => (Some(0), Some(1)),
            AtomicStatus::OneZero => (Some(1), Some(0)),
        };
        let failure = failure
            .and_then(|bits| integer.constant(bits, self.access.ptr_size()))
            .map_or(Value::Unknown, Value::Const);
        let value = if possible {
            let success = success
                .and_then(|bits| integer.constant(bits, self.access.ptr_size()))
                .map_or(Value::Unknown, Value::Const);
            success.join(&failure)
        } else {
            failure
        };
        state.define(status, value);
        Ok(())
    }

    /// Evaluates a local arithmetic query through the ordinary scalar kernel.
    fn evaluate<T: Target>(
        &self,
        op: SsaOp<T>,
        left: &Value<T>,
        right: &Value<T>,
        layout: TypeLayout,
    ) -> Value<T> {
        self.scalar.evaluate_with_operand_layouts(
            &op,
            Destination::from_type_layout(layout),
            |id| {
                if id == LEFT {
                    left.clone()
                } else {
                    right.clone()
                }
            },
            |_| OperandLayout::from_type_layout(layout),
        )
    }

    /// Applies wrapping integer or declared floating arithmetic without a second kernel.
    fn rmw<T: Target>(
        &self,
        kind: AtomicRmwOp,
        left: &Value<T>,
        right: &Value<T>,
        layout: TypeLayout,
    ) -> Value<T> {
        if kind == AtomicRmwOp::Xchg {
            return right.clone();
        }
        let op = match kind {
            AtomicRmwOp::Add | AtomicRmwOp::FAdd => SsaOp::Add {
                dest: RESULT,
                left: LEFT,
                right: RIGHT,
                flags: None,
            },
            AtomicRmwOp::Sub => SsaOp::Sub {
                dest: RESULT,
                left: LEFT,
                right: RIGHT,
                flags: None,
            },
            AtomicRmwOp::And => SsaOp::And {
                dest: RESULT,
                left: LEFT,
                right: RIGHT,
                flags: None,
            },
            AtomicRmwOp::Or => SsaOp::Or {
                dest: RESULT,
                left: LEFT,
                right: RIGHT,
                flags: None,
            },
            AtomicRmwOp::Xor => SsaOp::Xor {
                dest: RESULT,
                left: LEFT,
                right: RIGHT,
                flags: None,
            },
            AtomicRmwOp::Neg => SsaOp::Neg {
                dest: RESULT,
                operand: LEFT,
                flags: None,
            },
            AtomicRmwOp::AndNot => {
                let inverted = self.evaluate(
                    SsaOp::Not {
                        dest: RESULT,
                        operand: LEFT,
                        flags: None,
                    },
                    right,
                    right,
                    layout,
                );
                return self.rmw(AtomicRmwOp::And, left, &inverted, layout);
            }
            AtomicRmwOp::Min | AtomicRmwOp::Max | AtomicRmwOp::MinU | AtomicRmwOp::MaxU => {
                let minimum = matches!(kind, AtomicRmwOp::Min | AtomicRmwOp::MinU);
                let unsigned = matches!(kind, AtomicRmwOp::MinU | AtomicRmwOp::MaxU);
                return match self.relation(
                    if minimum { CmpKind::Lt } else { CmpKind::Gt },
                    unsigned,
                    left,
                    right,
                    layout,
                ) {
                    Some(true) => left.clone(),
                    Some(false) => right.clone(),
                    None => left.join(right),
                };
            }
            AtomicRmwOp::FMin | AtomicRmwOp::FMax | AtomicRmwOp::FMinNum | AtomicRmwOp::FMaxNum => {
                let Some(format) = layout.float_format() else {
                    return Value::Unknown;
                };
                let kind = match kind {
                    AtomicRmwOp::FMin => VectorBinaryKind::Min,
                    AtomicRmwOp::FMax => VectorBinaryKind::Max,
                    AtomicRmwOp::FMinNum => VectorBinaryKind::MinNum,
                    _ => VectorBinaryKind::MaxNum,
                };
                SsaOp::VectorBinary {
                    dest: RESULT,
                    left: LEFT,
                    right: RIGHT,
                    kind,
                    element: VectorElement {
                        kind: if format == FloatFormat::BFloat16 {
                            VectorElementKind::BFloat16
                        } else {
                            VectorElementKind::Float
                        },
                        scalar: true,
                        ..VectorElement::float(format.bits())
                    },
                    environment: None,
                }
            }
            AtomicRmwOp::Xchg => return right.clone(),
        };
        self.evaluate(op, left, right, layout)
    }

    /// Compares stored representations without applying floating numeric equality.
    fn equal_bits<T: Target>(
        &self,
        left: &Value<T>,
        right: &Value<T>,
        layout: TypeLayout,
    ) -> Option<bool> {
        let bits = layout
            .storage_bytes(self.access.ptr_size())?
            .checked_mul(8)?;
        let integer = self.word(u32::try_from(bits).ok()?, false).ok()?;
        let representation = |value: &Value<T>| match value.constant() {
            Some(ConstValue::Float(value)) => {
                ConstValue::of_lane(value.format().bits(), value.bits())
                    .map_or(Value::Unknown, Value::Const)
            }
            _ => value.clone(),
        };
        self.relation(
            CmpKind::Eq,
            true,
            &representation(left),
            &representation(right),
            integer,
        )
    }

    /// Decides a relation through the shared comparison semantics.
    fn relation<T: Target>(
        &self,
        cmp: CmpKind,
        unsigned: bool,
        left: &Value<T>,
        right: &Value<T>,
        layout: TypeLayout,
    ) -> Option<bool> {
        let op = match cmp {
            CmpKind::Eq | CmpKind::Ne => SsaOp::Ceq {
                dest: RESULT,
                left: LEFT,
                right: RIGHT,
                signaling: false,
            },
            CmpKind::Lt | CmpKind::Ge => SsaOp::Clt {
                dest: RESULT,
                left: LEFT,
                right: RIGHT,
                unsigned,
                signaling: false,
            },
            CmpKind::Gt | CmpKind::Le => SsaOp::Cgt {
                dest: RESULT,
                left: LEFT,
                right: RIGHT,
                unsigned,
                signaling: false,
            },
        };
        let compared = self.scalar.evaluate_with_operand_layouts(
            &op,
            Destination::default(),
            |id| {
                if id == LEFT {
                    left.clone()
                } else {
                    right.clone()
                }
            },
            |_| OperandLayout::from_type_layout(layout),
        );
        let result = compared.constant()?.as_bool()?;
        Some(if matches!(cmp, CmpKind::Ne | CmpKind::Ge | CmpKind::Le) {
            !result
        } else {
            result
        })
    }

    /// Evaluates subtraction conditions through shared arithmetic, overflow and comparisons.
    fn condition<T: Target>(
        &self,
        condition: CompareCondition,
        left: &Value<T>,
        right: &Value<T>,
        layout: TypeLayout,
    ) -> Option<bool> {
        let flag = match condition {
            CompareCondition::Relation { cmp, unsigned } => {
                return self.relation(cmp, unsigned, left, right, layout);
            }
            CompareCondition::Flag(flag) => flag,
        };
        match flag {
            FlagCondition::Carry | FlagCondition::NotCarry => self
                .relation(CmpKind::Lt, true, left, right, layout)
                .map(|value| {
                    if flag == FlagCondition::Carry {
                        value
                    } else {
                        !value
                    }
                }),
            FlagCondition::Zero | FlagCondition::NotZero => self
                .relation(CmpKind::Eq, true, left, right, layout)
                .map(|value| {
                    if flag == FlagCondition::Zero {
                        value
                    } else {
                        !value
                    }
                }),
            FlagCondition::Overflow | FlagCondition::NotOverflow => {
                let signed = self.word(layout.bit_width()?, true).ok()?;
                let (outcome, _) = self.scalar.control(
                    &SsaOp::SubOvf {
                        dest: RESULT,
                        left: LEFT,
                        right: RIGHT,
                        unsigned: false,
                        flags: None,
                    },
                    Destination::from_type_layout(signed),
                    |id| {
                        if id == LEFT {
                            left.clone()
                        } else {
                            right.clone()
                        }
                    },
                    |_| OperandLayout::from_type_layout(signed),
                )?;
                let overflow = match outcome {
                    Evaluation::Trap(ArithmeticTrap::Overflow) => true,
                    Evaluation::Value(_) => false,
                    _ => return None,
                };
                Some(if flag == FlagCondition::Overflow {
                    overflow
                } else {
                    !overflow
                })
            }
            _ => {
                let result = self.rmw(AtomicRmwOp::Sub, left, right, layout);
                let integer = layout.integer_conversion_type()?;
                let zero = Value::Const(integer.constant(0, self.access.ptr_size())?);
                match flag {
                    FlagCondition::Negative | FlagCondition::Positive => self
                        .relation(CmpKind::Lt, false, &result, &zero, layout)
                        .map(|value| {
                            if flag == FlagCondition::Negative {
                                value
                            } else {
                                !value
                            }
                        }),
                    FlagCondition::ParityEven | FlagCondition::ParityOdd => {
                        let mask = Value::Const(integer.constant(0xff, self.access.ptr_size())?);
                        let byte = self.rmw(AtomicRmwOp::And, &result, &mask, layout);
                        let count = self.evaluate(
                            SsaOp::Popcount {
                                dest: RESULT,
                                src: LEFT,
                            },
                            &byte,
                            &byte,
                            layout,
                        );
                        let one = Value::Const(integer.constant(1, self.access.ptr_size())?);
                        let parity = self.rmw(AtomicRmwOp::And, &count, &one, layout);
                        self.relation(CmpKind::Eq, true, &parity, &zero, layout)
                            .map(|value| {
                                if flag == FlagCondition::ParityEven {
                                    value
                                } else {
                                    !value
                                }
                            })
                    }
                    _ => None,
                }
            }
        }
    }

    /// Joins mathematical low/high halves for arithmetic on the entire pair.
    fn combine<T: Target>(&self, low: &Value<T>, high: &Value<T>, bits: u32) -> Value<T> {
        let (Some(low), Some(high)) = (low.constant(), high.constant()) else {
            return if low.may_reference_stack() || high.may_reference_stack() {
                Value::StackDerived
            } else {
                Value::Unknown
            };
        };
        let (Some((low, low_bits)), Some((high, high_bits))) = (
            low.scalar_integer_pattern(self.access.ptr_size()),
            high.scalar_integer_pattern(self.access.ptr_size()),
        ) else {
            return Value::Unknown;
        };
        if low_bits != bits || high_bits != bits {
            return Value::Unknown;
        }
        VectorConst::from_lanes(bits.saturating_mul(2), [low | (high << bits)])
            .map_or(Value::Unknown, |value| {
                Value::Const(ConstValue::Vector(value))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        analysis::{
            context::AnalysisContext,
            execution::{frame::Frame, value::Root},
            image::{ImageData, ImageRegion, Mutability},
        },
        ir::{
            function::SsaFunctionBuilder,
            ops::{AccessQualifiers, AtomicAccessWidth, AtomicOrdering, FenceKind},
            variable::{DefSite, VariableOrigin},
        },
        pointer::PointerSize,
        testing::{MACHINE, MockTarget, MockType},
    };

    const ATOMIC_MACHINE: crate::analysis::Machine = crate::analysis::Machine {
        byte_order: Some(Endianness::Little),
        ..MACHINE
    };

    const ADDR: SsaVarId = SsaVarId::from_index(3);
    const INPUT: SsaVarId = SsaVarId::from_index(4);
    const OLD: SsaVarId = SsaVarId::from_index(5);
    const STATUS: SsaVarId = SsaVarId::from_index(6);
    const EXPECTED: SsaVarId = SsaVarId::from_index(7);
    const SECOND: SsaVarId = SsaVarId::from_index(8);
    const OLD_SECOND: SsaVarId = SsaVarId::from_index(9);
    const EXPECTED_SECOND: SsaVarId = SsaVarId::from_index(10);

    fn state() -> State<MockTarget> {
        let mut builder = SsaFunctionBuilder::<MockTarget>::new(1, 0);
        for index in 0..11 {
            builder.function_mut().create_variable(
                VariableOrigin::Argument(index),
                0,
                DefSite::entry(),
                MockType::U64,
            );
        }
        State::new(Frame::enter_at(
            builder.function(),
            0,
            0,
            Address::new(Root::Stack, 0, PointerSize::Bit64),
            vec![],
            PointerSize::Bit64,
        ))
    }

    fn at() -> Address {
        Address::new(Root::Stack, -32, PointerSize::Bit64)
    }

    fn layout(bits: u32) -> TypeLayout {
        TypeLayout::integer(IntegerType::new(bits, false).expect("integer width"))
    }

    fn constant(bits: u32, value: u128) -> Value<MockTarget> {
        Value::Const(
            IntegerType::new(bits, false)
                .expect("integer width")
                .constant(value, PointerSize::Bit64)
                .expect("integer constant"),
        )
    }

    fn pattern(value: &Value<MockTarget>) -> Option<u128> {
        value
            .constant()?
            .scalar_integer_pattern(PointerSize::Bit64)
            .map(|(bits, _)| bits)
    }

    fn seed(
        kernel: &Atomics<'_>,
        state: &mut State<MockTarget>,
        bits: u32,
        old: u128,
        input: u128,
    ) {
        state.define(ADDR, Value::Addr(at()));
        state.define(INPUT, constant(bits, input));
        kernel
            .store(state, at(), &constant(bits, old), layout(bits), Some(true))
            .expect("seed");
    }

    fn width(bits: u32) -> AtomicAccessWidth {
        AtomicAccessWidth::try_from(bits).expect("width")
    }

    /// Independent host arithmetic covers every integer RMW and all physical widths.
    #[test]
    fn atomic_integer_rmw_widths_and_endianness() {
        for order in [Endianness::Little, Endianness::Big] {
            let context = AnalysisContext::new(crate::analysis::Machine {
                byte_order: Some(order),
                ..ATOMIC_MACHINE
            });
            let kernel = Atomics::new(Access::new(&context), Scalar::new(context.machine()));
            for bits in [8, 16, 32, 64, 128] {
                let mask = u128::MAX >> (128 - bits);
                let sign = 1u128 << (bits - 1);
                for (old, input) in [
                    (0, 1),
                    (mask, 1),
                    (sign, sign - 1),
                    (sign - 1, sign),
                    (0x55, 0xaa),
                ] {
                    for op in [
                        AtomicRmwOp::Xchg,
                        AtomicRmwOp::Add,
                        AtomicRmwOp::Sub,
                        AtomicRmwOp::And,
                        AtomicRmwOp::Or,
                        AtomicRmwOp::Xor,
                        AtomicRmwOp::AndNot,
                        AtomicRmwOp::Neg,
                        AtomicRmwOp::Min,
                        AtomicRmwOp::Max,
                        AtomicRmwOp::MinU,
                        AtomicRmwOp::MaxU,
                    ] {
                        let signed_less = (old ^ sign) < (input ^ sign);
                        let expected = match op {
                            AtomicRmwOp::Xchg => input,
                            AtomicRmwOp::Add => old.wrapping_add(input),
                            AtomicRmwOp::Sub => old.wrapping_sub(input),
                            AtomicRmwOp::And => old & input,
                            AtomicRmwOp::Or => old | input,
                            AtomicRmwOp::Xor => old ^ input,
                            AtomicRmwOp::AndNot => old & !input,
                            AtomicRmwOp::Neg => old.wrapping_neg(),
                            AtomicRmwOp::Min => {
                                if signed_less {
                                    old
                                } else {
                                    input
                                }
                            }
                            AtomicRmwOp::Max => {
                                if signed_less {
                                    input
                                } else {
                                    old
                                }
                            }
                            AtomicRmwOp::MinU => old.min(input),
                            AtomicRmwOp::MaxU => old.max(input),
                            _ => unreachable!(),
                        } & mask;
                        let mut state = state();
                        seed(&kernel, &mut state, bits, old, input);
                        let instruction = SsaOp::AtomicLockRmw {
                            dest: OLD,
                            addr: ADDR,
                            value: INPUT,
                            op,
                            ordering: AtomicOrdering::SeqCst,
                            width: width(bits),
                            access: AccessQualifiers::default(),
                        };
                        assert_eq!(
                            kernel.execute(&instruction, &mut state, |_| Some(layout(bits))),
                            Ok(true)
                        );
                        assert_eq!(
                            pattern(&state.value(OLD)),
                            Some(old),
                            "old {op:?}/{bits}/{order:?}"
                        );
                        let after = kernel.load(&state, at(), layout(bits)).expect("load");
                        assert_eq!(
                            pattern(&after),
                            Some(expected),
                            "{op:?}/{bits}/{order:?}: {old:x}, {input:x}"
                        );
                    }
                }
            }
        }
    }

    /// Exchange writes a known replacement even when the old contents are unknown.
    #[test]
    fn atomic_exchange_unknown_old_and_pointer_width() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        let mut state = state();
        state.define(ADDR, Value::Addr(at()));
        state.define(INPUT, constant(64, 0x12345678));
        let op = SsaOp::AtomicExchange {
            dest: OLD,
            addr: ADDR,
            value: INPUT,
            ordering: AtomicOrdering::SeqCst,
            width: AtomicAccessWidth::Pointer,
            access: AccessQualifiers::default(),
        };
        assert_eq!(
            kernel.execute(&op, &mut state, |_| Some(layout(64))),
            Ok(true)
        );
        assert_eq!(pattern(&state.value(OLD)), None);
        assert_eq!(
            pattern(&kernel.load(&state, at(), layout(64)).expect("load")),
            Some(0x12345678)
        );
    }

    /// Weak success joins possible writes, retaining only common physical bytes.
    #[test]
    fn atomic_compare_exchange_strong_weak_and_mismatch() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        for (weak, expected, succeeds) in [
            (false, 0x11223344, Some(true)),
            (false, 0, Some(false)),
            (true, 0x11223344, None),
            (true, 0, Some(false)),
        ] {
            let mut state = state();
            seed(&kernel, &mut state, 32, 0x11223344, 0x11223355);
            state.define(EXPECTED, constant(32, expected));
            let op = SsaOp::AtomicCmpXchg {
                old: OLD,
                success: Some(STATUS),
                addr: ADDR,
                expected: EXPECTED,
                desired: INPUT,
                success_ordering: AtomicOrdering::SeqCst,
                failure_ordering: AtomicOrdering::Relaxed,
                width: width(32),
                weak,
                access: AccessQualifiers::default(),
            };
            assert_eq!(
                kernel.execute(&op, &mut state, |_| Some(layout(32))),
                Ok(true)
            );
            assert_eq!(pattern(&state.value(OLD)), Some(0x11223344));
            assert_eq!(
                state.value(STATUS).constant().and_then(ConstValue::as_bool),
                succeeds
            );
            let after = kernel.load(&state, at(), layout(32)).expect("load");
            assert_eq!(
                pattern(&after),
                succeeds.map(|yes| if yes { 0x11223355 } else { 0x11223344 })
            );
            assert_eq!(
                state
                    .memory()
                    .known_run(at().moved(1, PointerSize::Bit64).expect("byte"), 3, None),
                [0x33, 0x22, 0x11]
            );
        }
    }

    /// Exclusive stores may fail after a load and consume the possible reservation.
    #[test]
    fn atomic_exclusive_load_store_and_consumed_status() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        for reserves in [false, true] {
            let mut state = state();
            seed(&kernel, &mut state, 32, 7, 9);
            let load = SsaOp::AtomicLoad {
                dest: OLD,
                addr: ADDR,
                value_type: MockType::U32,
                ordering: AtomicOrdering::Acquire,
                width: width(32),
                reserves,
                access: AccessQualifiers::default(),
            };
            assert_eq!(
                kernel.execute(&load, &mut state, |_| Some(layout(32))),
                Ok(true)
            );
            assert_eq!(pattern(&state.value(OLD)), Some(7));
            let store = SsaOp::AtomicStoreConditional {
                status: STATUS,
                status_encoding: AtomicStatus::ZeroOne,
                consumes_reservation: true,
                addr: ADDR,
                value: INPUT,
                value_type: MockType::U32,
                success_ordering: AtomicOrdering::Release,
                failure_ordering: AtomicOrdering::Relaxed,
                width: width(32),
                access: AccessQualifiers::default(),
            };
            assert_eq!(
                kernel.execute(&store, &mut state, |_| Some(layout(64))),
                Ok(true)
            );
            assert_eq!(pattern(&state.value(STATUS)), None);
            assert!(!state.reservation_possible());
            kernel
                .store(&mut state, at(), &constant(32, 13), layout(32), Some(true))
                .expect("reset bytes");
            assert_eq!(
                kernel.execute(&store, &mut state, |_| Some(layout(64))),
                Ok(true)
            );
            assert_eq!(pattern(&state.value(STATUS)), Some(1));
            assert_eq!(
                pattern(&kernel.load(&state, at(), layout(32)).expect("load")),
                Some(13)
            );
        }
    }

    /// Signed relations and every subtraction flag agree with independent byte arithmetic.
    #[test]
    fn atomic_conditional_updates_cover_relations_and_flags() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        for (left, right) in [
            (0u8, 0u8),
            (0, 1),
            (127, 255),
            (128, 1),
            (255, 127),
            (85, 170),
        ] {
            let result = left.wrapping_sub(right);
            let (a, b) = (
                constant(8, u128::from(left)),
                constant(8, u128::from(right)),
            );
            for flag in [
                FlagCondition::Carry,
                FlagCondition::NotCarry,
                FlagCondition::Zero,
                FlagCondition::NotZero,
                FlagCondition::Overflow,
                FlagCondition::NotOverflow,
                FlagCondition::Negative,
                FlagCondition::Positive,
                FlagCondition::ParityEven,
                FlagCondition::ParityOdd,
            ] {
                let expected = match flag {
                    FlagCondition::Carry => left < right,
                    FlagCondition::NotCarry => left >= right,
                    FlagCondition::Zero => result == 0,
                    FlagCondition::NotZero => result != 0,
                    FlagCondition::Overflow => (left as i8).overflowing_sub(right as i8).1,
                    FlagCondition::NotOverflow => !(left as i8).overflowing_sub(right as i8).1,
                    FlagCondition::Negative => result & 128 != 0,
                    FlagCondition::Positive => result & 128 == 0,
                    FlagCondition::ParityEven => result.count_ones() % 2 == 0,
                    FlagCondition::ParityOdd => result.count_ones() % 2 != 0,
                };
                assert_eq!(
                    kernel.condition(CompareCondition::Flag(flag), &a, &b, layout(8)),
                    Some(expected),
                    "{flag:?} {left}/{right}"
                );
            }
            for unsigned in [false, true] {
                for cmp in [
                    CmpKind::Eq,
                    CmpKind::Ne,
                    CmpKind::Lt,
                    CmpKind::Le,
                    CmpKind::Gt,
                    CmpKind::Ge,
                ] {
                    let ordering = if unsigned {
                        left.cmp(&right)
                    } else {
                        (left as i8).cmp(&(right as i8))
                    };
                    let expected = match cmp {
                        CmpKind::Eq => ordering.is_eq(),
                        CmpKind::Ne => !ordering.is_eq(),
                        CmpKind::Lt => ordering.is_lt(),
                        CmpKind::Le => !ordering.is_gt(),
                        CmpKind::Gt => ordering.is_gt(),
                        CmpKind::Ge => !ordering.is_lt(),
                    };
                    let condition = CompareCondition::Relation { cmp, unsigned };
                    assert_eq!(
                        kernel.condition(condition, &a, &b, layout(8)),
                        Some(expected),
                        "{cmp:?}/{unsigned}: {left}/{right}"
                    );
                    let mut state = state();
                    seed(&kernel, &mut state, 8, u128::from(left), 3);
                    state.define(EXPECTED, b.clone());
                    let op = SsaOp::AtomicCondRmw {
                        old: OLD,
                        addr: ADDR,
                        compare: EXPECTED,
                        value: INPUT,
                        op: AtomicRmwOp::Add,
                        condition,
                        ordering: AtomicOrdering::SeqCst,
                        width: width(8),
                        access: AccessQualifiers::default(),
                    };
                    assert_eq!(
                        kernel.execute(&op, &mut state, |_| Some(layout(8))),
                        Ok(true)
                    );
                    assert_eq!(
                        pattern(&kernel.load(&state, at(), layout(8)).expect("load")),
                        Some(u128::from(if expected {
                            left.wrapping_add(3)
                        } else {
                            left
                        }))
                    );
                }
            }
        }
    }

    /// A 64-byte block is one atomic access of the vector it is typed as
    /// (AArch64 `ld64b`, `st64b`): what one block load reads, one block store
    /// writes back whole, each word where it was. An arithmetic of that width
    /// has no integer word to compute in, and refuses.
    #[test]
    fn atomic_block_moves_its_whole_extent() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        let mut state = state();
        let from = Address::new(Root::Stack, -192, PointerSize::Bit64);
        let to = Address::new(Root::Stack, -128, PointerSize::Bit64);
        for word in 0..8u8 {
            let at = from
                .moved(i64::from(word) * 8, PointerSize::Bit64)
                .expect("word");
            kernel
                .store(
                    &mut state,
                    at,
                    &constant(64, 0x1111_1111_1111_1111 * u128::from(word + 1)),
                    layout(64),
                    Some(true),
                )
                .expect("seed");
        }
        state.define(ADDR, Value::Addr(from));
        let load = SsaOp::AtomicLoad {
            dest: OLD,
            addr: ADDR,
            value_type: MockType::V2I256,
            ordering: AtomicOrdering::Relaxed,
            width: width(512),
            reserves: false,
            access: AccessQualifiers::default(),
        };
        assert_eq!(kernel.execute(&load, &mut state, |_| None), Ok(true));
        state.define(ADDR, Value::Addr(to));
        let store = SsaOp::AtomicStore {
            addr: ADDR,
            value: OLD,
            value_type: MockType::V2I256,
            ordering: AtomicOrdering::Relaxed,
            width: width(512),
            access: AccessQualifiers::default(),
        };
        assert_eq!(kernel.execute(&store, &mut state, |_| None), Ok(true));
        for word in 0..8u8 {
            let at = to
                .moved(i64::from(word) * 8, PointerSize::Bit64)
                .expect("word");
            assert_eq!(
                pattern(&kernel.load(&state, at, layout(64)).expect("load")),
                Some(0x1111_1111_1111_1111 * u128::from(word + 1)),
                "word {word}"
            );
        }
        let rmw = SsaOp::AtomicLockRmw {
            dest: OLD,
            addr: ADDR,
            value: INPUT,
            op: AtomicRmwOp::Add,
            ordering: AtomicOrdering::Relaxed,
            width: width(512),
            access: AccessQualifiers::default(),
        };
        assert_eq!(
            kernel.execute(&rmw, &mut state, |_| None),
            Err(AccessError::Refused(MemoryRefusal::AtomicContract))
        );
    }

    /// A read-check-write writes what its check passes, states the check's
    /// NZCV, and refuses what it cannot decide: a change a failed check may
    /// or may not write, and a protection the regime configures.
    #[test]
    fn atomic_read_check_write_follows_its_check() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        let valid: u128 = (1 << 114) | 1;
        const MASK: SsaVarId = SsaVarId::from_index(7);
        let checked = |compare: Option<SsaVarId>, op: AtomicRmwOp, protection: RcwProtection| {
            SsaOp::AtomicCheckedRmw {
                old: OLD,
                status: STATUS,
                addr: ADDR,
                compare,
                value: INPUT,
                op,
                rcw_mask: MASK,
                rcws_mask: None,
                protection,
                ordering: AtomicOrdering::AcqRel,
                width: width(128),
                access: AccessQualifiers::default(),
            }
        };
        let layouts = |var: SsaVarId| {
            Some(if var == STATUS {
                layout(8)
            } else {
                layout(128)
            })
        };
        let run = |op: &SsaOp<MockTarget>, input: u128, mask: u128, compare: u128| {
            let mut state = state();
            seed(&kernel, &mut state, 128, valid, input);
            state.define(MASK, constant(128, mask));
            state.define(SECOND, constant(128, compare));
            let result = kernel.execute(op, &mut state, layouts);
            let memory = pattern(&kernel.load(&state, at(), layout(128)).expect("load"));
            (result, pattern(&state.value(STATUS)), memory)
        };
        // A set the mask allows: written, NZCV 0010, the old value returned.
        let set = checked(None, AtomicRmwOp::Or, RcwProtection::Enabled);
        assert_eq!(
            run(&set, 1 << 5, 1 << 5, 0),
            (Ok(true), Some(0b0010), Some(valid | (1 << 5)))
        );
        // A compare that fails: no check, NZCV 1010, memory as it was.
        let cas = checked(Some(SECOND), AtomicRmwOp::Xchg, RcwProtection::Enabled);
        assert_eq!(run(&cas, 0, 0, 7), (Ok(true), Some(0b1010), Some(valid)));
        // A compare that passes, swapping in a value that clears the valid
        // bit: the check fails, and whether it writes is the implementation's.
        assert!(matches!(
            run(&cas, 1 << 114, u128::MAX, valid).0,
            Err(AccessError::Refused(MemoryRefusal::AtomicContract))
        ));
        // A 64-bit form's protection is the regime's to configure.
        let regime = checked(None, AtomicRmwOp::Or, RcwProtection::Regime);
        assert!(matches!(
            run(&regime, 0, 0, 0).0,
            Err(AccessError::Refused(MemoryRefusal::AtomicContract))
        ));
    }

    /// Eight read-only bytes at `0x2000` and eight writable ones at `0x3000`.
    struct Image;

    impl ImageData for Image {
        fn endianness(&self) -> Endianness {
            Endianness::Little
        }

        fn region(&self, address: u64) -> Option<ImageRegion> {
            let (start, mutability) = match address {
                0x2000..0x2008 => (0x2000, Mutability::ReadOnly),
                0x3000..0x3008 => (0x3000, Mutability::Writable),
                _ => return None,
            };
            Some(ImageRegion {
                start,
                end: start + 8,
                mutability,
            })
        }

        fn bytes(&self, address: u64, len: usize) -> Option<&[u8]> {
            const BYTES: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
            let start = usize::try_from(address.checked_sub(0x2000)?).ok()?;
            BYTES.get(start..start.checked_add(len)?)
        }

        fn number_offset(&self) -> Option<i64> {
            Some(0)
        }
    }

    /// An atomic load of memory no observer can write reads what is there:
    /// read-only image bytes. A load that opens a reservation, and one of
    /// writable bytes, which another observer may change, stay refused.
    #[test]
    fn an_atomic_load_of_read_only_image_bytes_reads_them() {
        let context = AnalysisContext::new(ATOMIC_MACHINE).with_image(&Image);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        let run = |address: i64, reserves: bool| {
            let mut state = state();
            state.define(
                ADDR,
                Value::Addr(Address::new(Root::Symbol, address, PointerSize::Bit64)),
            );
            let load = SsaOp::AtomicLoad {
                dest: OLD,
                addr: ADDR,
                value_type: MockType::U64,
                ordering: AtomicOrdering::Acquire,
                width: width(64),
                reserves,
                access: AccessQualifiers::default(),
            };
            let result = kernel.execute(&load, &mut state, |_| Some(layout(64)));
            (result, pattern(&state.value(OLD)))
        };
        assert_eq!(run(0x2000, false), (Ok(true), Some(0x0807_0605_0403_0201)));
        assert!(matches!(
            run(0x2000, true).0,
            Err(AccessError::Refused(MemoryRefusal::AtomicContract))
        ));
        assert!(matches!(
            run(0x3000, false).0,
            Err(AccessError::Refused(MemoryRefusal::AtomicContract))
        ));
    }

    /// Full-footprint admission rejects shared, escaped, volatile and crossing spans.
    #[test]
    fn atomic_private_admission_precedes_mutation() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        for (root, offset, escaped, volatile) in [
            (Root::Stack, -2, false, false),
            (Root::Number, 0x1000, false, false),
            (Root::Stack, -32, true, false),
            (Root::Stack, -32, false, true),
        ] {
            let mut state = state();
            seed(&kernel, &mut state, 32, 0x11223344, 0x55667788);
            state.define(
                ADDR,
                Value::Addr(Address::new(root, offset, PointerSize::Bit64)),
            );
            if escaped {
                state.memory_mut().escape_stack();
            }
            let before = state.memory().known_run(at(), 4, None);
            let op = SsaOp::AtomicStore {
                addr: ADDR,
                value: INPUT,
                value_type: MockType::U32,
                ordering: AtomicOrdering::SeqCst,
                width: width(32),
                access: AccessQualifiers {
                    volatile,
                    ..AccessQualifiers::default()
                },
            };
            assert_eq!(
                kernel.execute(&op, &mut state, |_| Some(layout(32))),
                Err(AccessError::Refused(MemoryRefusal::AtomicContract))
            );
            assert_eq!(state.memory().known_run(at(), 4, None), before);
        }
        assert!(!state().memory().private_span(at(), usize::MAX));
    }

    /// Pair loads and compare-exchanges number halves by increasing memory address.
    #[test]
    fn atomic_pair_memory_order_and_weak_common_bytes() {
        for order in [Endianness::Little, Endianness::Big] {
            let context = AnalysisContext::new(crate::analysis::Machine {
                byte_order: Some(order),
                ..ATOMIC_MACHINE
            });
            let kernel = Atomics::new(Access::new(&context), Scalar::new(context.machine()));
            for weak in [false, true] {
                let mut state = state();
                seed(&kernel, &mut state, 32, 0x11223344, 0x11223355);
                let next = at().moved(4, PointerSize::Bit64).expect("second");
                kernel
                    .store(
                        &mut state,
                        next,
                        &constant(32, 0xaabbccdd),
                        layout(32),
                        Some(true),
                    )
                    .expect("second seed");
                let load = SsaOp::AtomicPairLoad {
                    first: OLD,
                    second: OLD_SECOND,
                    addr: ADDR,
                    first_type: MockType::U32,
                    second_type: MockType::U32,
                    ordering: AtomicOrdering::Acquire,
                    width: width(64),
                    access: AccessQualifiers::default(),
                };
                assert_eq!(
                    kernel.execute(&load, &mut state, |_| Some(layout(32))),
                    Ok(true)
                );
                assert_eq!(pattern(&state.value(OLD)), Some(0x11223344));
                assert_eq!(pattern(&state.value(OLD_SECOND)), Some(0xaabbccdd));
                state.define(EXPECTED, constant(32, 0x11223344));
                state.define(EXPECTED_SECOND, constant(32, 0xaabbccdd));
                state.define(SECOND, constant(32, 0xaabbccff));
                let cas = SsaOp::AtomicPairCmpXchg {
                    old_first: OLD,
                    old_second: OLD_SECOND,
                    addr: ADDR,
                    expected_first: EXPECTED,
                    expected_second: EXPECTED_SECOND,
                    desired_first: INPUT,
                    desired_second: SECOND,
                    success_ordering: AtomicOrdering::SeqCst,
                    failure_ordering: AtomicOrdering::Relaxed,
                    width: width(64),
                    weak,
                    access: AccessQualifiers::default(),
                };
                assert_eq!(
                    kernel.execute(&cas, &mut state, |_| Some(layout(32))),
                    Ok(true)
                );
                assert_eq!(pattern(&state.value(OLD)), Some(0x11223344));
                assert_eq!(pattern(&state.value(OLD_SECOND)), Some(0xaabbccdd));
                assert_eq!(
                    pattern(&kernel.load(&state, at(), layout(32)).expect("first")),
                    (!weak).then_some(0x11223355)
                );
                assert_eq!(
                    pattern(&kernel.load(&state, next, layout(32)).expect("second")),
                    (!weak).then_some(0xaabbccff)
                );
                // A consumed reservation forces both pair halves to stay unchanged.
                state.set_reservation_possible(false);
                let conditional = SsaOp::AtomicPairStoreConditional {
                    status: STATUS,
                    status_encoding: AtomicStatus::ZeroOne,
                    consumes_reservation: true,
                    addr: ADDR,
                    first_value: EXPECTED,
                    second_value: EXPECTED_SECOND,
                    first_type: MockType::U32,
                    second_type: MockType::U32,
                    success_ordering: AtomicOrdering::Release,
                    failure_ordering: AtomicOrdering::Relaxed,
                    width: width(64),
                    access: AccessQualifiers::default(),
                };
                assert_eq!(
                    kernel.execute(&conditional, &mut state, |_| Some(layout(64))),
                    Ok(true)
                );
                assert_eq!(pattern(&state.value(STATUS)), Some(1));
                assert_eq!(
                    pattern(&kernel.load(&state, at(), layout(32)).expect("first")),
                    (!weak).then_some(0x11223355)
                );
            }
        }
    }

    /// Pair arithmetic carries between mathematical low/high halves in either byte order.
    #[test]
    fn atomic_pair_rmw_carry_and_bitwise_halves() {
        for order in [Endianness::Little, Endianness::Big] {
            let context = AnalysisContext::new(crate::analysis::Machine {
                byte_order: Some(order),
                ..ATOMIC_MACHINE
            });
            let kernel = Atomics::new(Access::new(&context), Scalar::new(context.machine()));
            for bits in [32, 64, 128] {
                let half = bits / 2;
                let mask = u128::MAX >> (128 - half);
                let lower_at = at();
                let higher_at = at()
                    .moved(i64::from(half / 8), PointerSize::Bit64)
                    .expect("higher");
                for op in [
                    AtomicRmwOp::Add,
                    AtomicRmwOp::Sub,
                    AtomicRmwOp::Xchg,
                    AtomicRmwOp::AndNot,
                    AtomicRmwOp::Or,
                    AtomicRmwOp::Xor,
                ] {
                    let mut state = state();
                    state.define(ADDR, Value::Addr(at()));
                    state.define(INPUT, constant(half, 1));
                    state.define(SECOND, constant(half, 2));
                    kernel
                        .store(
                            &mut state,
                            lower_at,
                            &constant(half, mask),
                            layout(half),
                            Some(true),
                        )
                        .expect("lower");
                    kernel
                        .store(
                            &mut state,
                            higher_at,
                            &constant(half, 3),
                            layout(half),
                            Some(true),
                        )
                        .expect("higher");
                    let instruction = SsaOp::AtomicPairRmw {
                        old_first: OLD,
                        old_second: OLD_SECOND,
                        addr: ADDR,
                        value_first: INPUT,
                        value_second: SECOND,
                        op,
                        ordering: AtomicOrdering::SeqCst,
                        width: width(bits),
                        access: AccessQualifiers::default(),
                    };
                    assert_eq!(
                        kernel.execute(&instruction, &mut state, |_| Some(layout(half))),
                        Ok(true)
                    );
                    // Arm SWPP/LDSETP: the first register is the half at the
                    // lower address, so the 2x-wide value reads
                    // `higher:lower` little-endian and `lower:higher`
                    // big-endian, for memory and register pair alike.
                    assert_eq!(pattern(&state.value(OLD)), Some(mask));
                    assert_eq!(pattern(&state.value(OLD_SECOND)), Some(3));
                    let (old, input) = if order == Endianness::Little {
                        ((3 << half) | mask, (2 << half) | 1)
                    } else {
                        ((mask << half) | 3, (1 << half) | 2)
                    };
                    let expected = match op {
                        AtomicRmwOp::Add => old.wrapping_add(input),
                        AtomicRmwOp::Sub => old.wrapping_sub(input),
                        AtomicRmwOp::Xchg => input,
                        AtomicRmwOp::AndNot => old & !input,
                        AtomicRmwOp::Or => old | input,
                        AtomicRmwOp::Xor => old ^ input,
                        _ => unreachable!(),
                    };
                    let (lower, higher) = if order == Endianness::Little {
                        (expected & mask, (expected >> half) & mask)
                    } else {
                        ((expected >> half) & mask, expected & mask)
                    };
                    assert_eq!(
                        pattern(&kernel.load(&state, lower_at, layout(half)).expect("lower")),
                        Some(lower),
                        "{op:?}/{bits}/{order:?}"
                    );
                    assert_eq!(
                        pattern(
                            &kernel
                                .load(&state, higher_at, layout(half))
                                .expect("higher")
                        ),
                        Some(higher),
                        "{op:?}/{bits}/{order:?}"
                    );
                }
            }
        }
    }

    /// Floating RMW uses its declared format rather than integer bit arithmetic.
    #[test]
    fn atomic_floating_rmw_declared_formats() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        let ty = TypeLayout::of::<MockTarget>(&MockType::F64, PointerSize::Bit64);
        for (op, expected) in [
            (AtomicRmwOp::FAdd, 3.75),
            (AtomicRmwOp::FMin, 1.25),
            (AtomicRmwOp::FMax, 2.5),
            (AtomicRmwOp::FMinNum, 1.25),
            (AtomicRmwOp::FMaxNum, 2.5),
        ] {
            let mut state = state();
            state.define(ADDR, Value::Addr(at()));
            state.define(INPUT, Value::Const(ConstValue::from(2.5f64)));
            kernel
                .store(
                    &mut state,
                    at(),
                    &Value::Const(ConstValue::from(1.25f64)),
                    ty,
                    Some(true),
                )
                .expect("float seed");
            let instruction = SsaOp::AtomicLockRmw {
                dest: OLD,
                addr: ADDR,
                value: INPUT,
                op,
                ordering: AtomicOrdering::SeqCst,
                width: width(64),
                access: AccessQualifiers::default(),
            };
            assert_eq!(
                kernel.execute(&instruction, &mut state, |_| Some(ty)),
                Ok(true)
            );
            assert_eq!(
                state.value(OLD).constant().and_then(ConstValue::as_f64),
                Some(1.25)
            );
            let after = kernel.load(&state, at(), ty).expect("float load");
            assert_eq!(
                after.constant().and_then(ConstValue::as_f64),
                Some(expected),
                "{op:?}"
            );
        }
    }

    /// A fence preserves private bytes and remains an effect handled by the stepper.
    #[test]
    fn atomic_fence_preserves_private_storage() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        let mut state = state();
        seed(&kernel, &mut state, 32, 7, 9);
        assert_eq!(
            kernel.execute(
                &SsaOp::Fence {
                    kind: FenceKind::Full
                },
                &mut state,
                |_| None
            ),
            Ok(true)
        );
        assert_eq!(
            pattern(&kernel.load(&state, at(), layout(32)).expect("load")),
            Some(7)
        );
    }

    /// An architecture's failure encoding is independent of the access width.
    #[test]
    fn atomic_status_encoding_is_explicit_and_width_preserving() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        for bits in [32, 64] {
            for (encoding, failed) in [
                (AtomicStatus::Unspecified, None),
                (AtomicStatus::ZeroNonzero, None),
                (AtomicStatus::ZeroOne, Some(1)),
                (AtomicStatus::OneZero, Some(0)),
            ] {
                let mut state = state();
                seed(&kernel, &mut state, 32, 7, 9);
                state.set_reservation_possible(false);
                let op = SsaOp::AtomicStoreConditional {
                    status: STATUS,
                    status_encoding: encoding,
                    consumes_reservation: true,
                    addr: ADDR,
                    value: INPUT,
                    value_type: MockType::U32,
                    success_ordering: AtomicOrdering::Release,
                    failure_ordering: AtomicOrdering::Relaxed,
                    width: width(32),
                    access: AccessQualifiers::default(),
                };
                assert_eq!(
                    kernel.execute(&op, &mut state, |_| Some(layout(bits))),
                    Ok(true)
                );
                assert_eq!(pattern(&state.value(STATUS)), failed, "{encoding:?}/{bits}");
                assert_eq!(
                    pattern(&kernel.load(&state, at(), layout(32)).expect("unchanged")),
                    Some(7)
                );
                kernel
                    .status(&mut state, STATUS, true, encoding, Some(layout(bits)))
                    .expect("possible result");
                assert_eq!(pattern(&state.value(STATUS)), None);
            }
        }
    }

    /// Compare-exchange compares float bits, distinguishing signed zero and matching NaNs.
    #[test]
    fn atomic_compare_exchange_float_representation() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        let ty = TypeLayout::of::<MockTarget>(&MockType::F64, PointerSize::Bit64);
        for (old, expected, matches) in [
            (0u128, 1u128 << 63, false),
            (0x7ff8000000000011, 0x7ff8000000000011, true),
            (0x7ff8000000000011, 0x7ff8000000000012, false),
        ] {
            let mut state = state();
            state.define(ADDR, Value::Addr(at()));
            let before = Value::Const(ConstValue::Float(
                crate::ir::float::FloatBits::new(FloatFormat::Double, old).expect("float bits"),
            ));
            let expected = Value::Const(ConstValue::Float(
                crate::ir::float::FloatBits::new(FloatFormat::Double, expected)
                    .expect("float bits"),
            ));
            state.define(EXPECTED, expected);
            state.define(INPUT, Value::Const(ConstValue::from(2.5f64)));
            kernel
                .store(&mut state, at(), &before, ty, Some(true))
                .expect("old");
            let op = SsaOp::AtomicCmpXchg {
                old: OLD,
                success: Some(STATUS),
                addr: ADDR,
                expected: EXPECTED,
                desired: INPUT,
                success_ordering: AtomicOrdering::SeqCst,
                failure_ordering: AtomicOrdering::Relaxed,
                width: width(64),
                weak: false,
                access: AccessQualifiers::default(),
            };
            assert_eq!(kernel.execute(&op, &mut state, |_| Some(ty)), Ok(true));
            assert_eq!(
                state.value(STATUS).constant().and_then(ConstValue::as_bool),
                Some(matches)
            );
            let after = kernel.load(&state, at(), ty).expect("after");
            let desired = ConstValue::from(2.5f64);
            assert_eq!(
                after.constant(),
                if matches {
                    Some(&desired)
                } else {
                    before.constant()
                }
            );
        }
    }

    /// Clearing the monitor prevents a store even when its memory address is private.
    #[test]
    fn atomic_clear_exclusive_consumes_hidden_reservation() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        let mut state = state();
        seed(&kernel, &mut state, 32, 7, 9);
        let clear = SsaOp::SystemOp(Box::new(crate::ir::ops::NativeKindedData {
            kind: crate::ir::ops::SystemOpKind::Barrier(crate::ir::ops::BarrierOp::ClearExclusive),
            mnemonic: String::new(),
            metadata: None,
            outputs: vec![],
            inputs: vec![],
        }));
        assert_eq!(kernel.execute(&clear, &mut state, |_| None), Ok(true));
        assert!(!state.reservation_possible());
        assert_eq!(
            pattern(&kernel.load(&state, at(), layout(32)).expect("load")),
            Some(7)
        );
    }

    /// Every floating RMW also interprets bfloat16 without guessing IEEE binary16.
    #[test]
    fn atomic_bfloat_rmw_uses_declared_format() {
        for order in [Endianness::Little, Endianness::Big] {
            let machine = crate::analysis::Machine {
                byte_order: Some(order),
                ..ATOMIC_MACHINE
            };
            let context = AnalysisContext::new(machine);
            let kernel = Atomics::new(Access::new(&context), Scalar::new(machine));
            let ty = TypeLayout::of::<MockTarget>(&MockType::BF16, PointerSize::Bit64);
            let before = ConstValue::Float(
                crate::ir::float::FloatBits::new(FloatFormat::BFloat16, 0x3fc0).expect("1.5"),
            );
            let input = ConstValue::Float(
                crate::ir::float::FloatBits::new(FloatFormat::BFloat16, 0x4010).expect("2.25"),
            );
            for (op, expected_bits) in [
                (AtomicRmwOp::FAdd, 0x4070),
                (AtomicRmwOp::FMin, 0x3fc0),
                (AtomicRmwOp::FMax, 0x4010),
                (AtomicRmwOp::FMinNum, 0x3fc0),
                (AtomicRmwOp::FMaxNum, 0x4010),
            ] {
                let mut state = state();
                state.define(ADDR, Value::Addr(at()));
                state.define(INPUT, Value::Const(input.clone()));
                kernel
                    .store(
                        &mut state,
                        at(),
                        &Value::Const(before.clone()),
                        ty,
                        Some(true),
                    )
                    .expect("seed");
                let instruction = SsaOp::AtomicLockRmw {
                    dest: OLD,
                    addr: ADDR,
                    value: INPUT,
                    op,
                    ordering: AtomicOrdering::SeqCst,
                    width: width(16),
                    access: AccessQualifiers::default(),
                };
                assert_eq!(
                    kernel.execute(&instruction, &mut state, |_| Some(ty)),
                    Ok(true)
                );
                let after = kernel.load(&state, at(), ty).expect("after");
                let expected = ConstValue::Float(
                    crate::ir::float::FloatBits::new(FloatFormat::BFloat16, expected_bits)
                        .expect("oracle"),
                );
                assert_eq!(after.constant(), Some(&expected), "{op:?}/{order:?}");
            }
        }
    }

    /// An undecided update joins writes; an unknown replacement cannot keep stale bytes.
    #[test]
    fn atomic_unknown_conditions_and_values_forget_only_written_span() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        let mut state = state();
        seed(&kernel, &mut state, 32, 0x11223344, 1);
        state.define(EXPECTED, Value::Unknown);
        let conditional = SsaOp::AtomicCondRmw {
            old: OLD,
            addr: ADDR,
            compare: EXPECTED,
            value: INPUT,
            op: AtomicRmwOp::Add,
            condition: CompareCondition::Relation {
                cmp: CmpKind::Eq,
                unsigned: true,
            },
            ordering: AtomicOrdering::SeqCst,
            width: width(32),
            access: AccessQualifiers::default(),
        };
        assert_eq!(
            kernel.execute(&conditional, &mut state, |_| Some(layout(32))),
            Ok(true)
        );
        assert_eq!(pattern(&state.value(OLD)), Some(0x11223344));
        assert_eq!(
            pattern(&kernel.load(&state, at(), layout(32)).expect("joined")),
            None
        );
        assert_eq!(
            state
                .memory()
                .known_run(at().moved(1, PointerSize::Bit64).expect("tail"), 3, None),
            [0x33, 0x22, 0x11]
        );
        state.define(INPUT, Value::Unknown);
        let exchange = SsaOp::AtomicExchange {
            dest: OLD,
            addr: ADDR,
            value: INPUT,
            ordering: AtomicOrdering::SeqCst,
            width: width(32),
            access: AccessQualifiers::default(),
        };
        assert_eq!(
            kernel.execute(&exchange, &mut state, |_| Some(layout(32))),
            Ok(true)
        );
        assert!(
            state
                .memory()
                .known_run(at().moved(1, PointerSize::Bit64).expect("tail"), 3, None)
                .is_empty()
        );
    }

    /// Status polarity does not imply that a conditional store consumes the monitor.
    #[test]
    fn atomic_reservation_consumption_is_an_independent_contract() {
        let context = AnalysisContext::new(ATOMIC_MACHINE);
        let kernel = Atomics::new(Access::new(&context), Scalar::new(ATOMIC_MACHINE));
        for encoding in [
            AtomicStatus::Unspecified,
            AtomicStatus::ZeroNonzero,
            AtomicStatus::ZeroOne,
            AtomicStatus::OneZero,
        ] {
            for consumes in [false, true] {
                for pair in [false, true] {
                    let mut state = state();
                    seed(&kernel, &mut state, 32, 7, 9);
                    state.define(SECOND, constant(32, 11));
                    let next = at().moved(4, PointerSize::Bit64).expect("second");
                    kernel
                        .store(&mut state, next, &constant(32, 17), layout(32), Some(true))
                        .expect("second seed");
                    let op = if pair {
                        SsaOp::AtomicPairStoreConditional {
                            status: STATUS,
                            status_encoding: encoding,
                            consumes_reservation: consumes,
                            addr: ADDR,
                            first_value: INPUT,
                            second_value: SECOND,
                            first_type: MockType::U32,
                            second_type: MockType::U32,
                            success_ordering: AtomicOrdering::Release,
                            failure_ordering: AtomicOrdering::Relaxed,
                            width: width(64),
                            access: AccessQualifiers::default(),
                        }
                    } else {
                        SsaOp::AtomicStoreConditional {
                            status: STATUS,
                            status_encoding: encoding,
                            consumes_reservation: consumes,
                            addr: ADDR,
                            value: INPUT,
                            value_type: MockType::U32,
                            success_ordering: AtomicOrdering::Release,
                            failure_ordering: AtomicOrdering::Relaxed,
                            width: width(32),
                            access: AccessQualifiers::default(),
                        }
                    };
                    assert_eq!(
                        kernel.execute(&op, &mut state, |_| Some(layout(32))),
                        Ok(true)
                    );
                    assert_eq!(state.reservation_possible(), !consumes);
                    // A plain private store neither creates nor guarantees loss of a monitor.
                    kernel
                        .store(&mut state, at(), &constant(32, 13), layout(32), Some(true))
                        .expect("reset");
                    kernel
                        .store(&mut state, next, &constant(32, 19), layout(32), Some(true))
                        .expect("reset second");
                    assert_eq!(
                        kernel.execute(&op, &mut state, |_| Some(layout(32))),
                        Ok(true)
                    );
                    assert_eq!(
                        pattern(&kernel.load(&state, at(), layout(32)).expect("result")),
                        consumes.then_some(13),
                        "{encoding:?}/{consumes}/{pair}"
                    );
                    if pair {
                        assert_eq!(
                            pattern(
                                &kernel
                                    .load(&state, next, layout(32))
                                    .expect("second result")
                            ),
                            consumes.then_some(19)
                        );
                    }
                }
            }
        }
    }
}
