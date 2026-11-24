//! Lane-wise vector loads: a masked load, a gather, a first-faulting load, a
//! segment load, an expanding load and a replicating structure load, each
//! reading its active lanes through the same footprint, permission and
//! alignment contract as an ordinary load.

use crate::{
    analysis::{
        consts::{OperandLayout, PredicateMask},
        execution::{
            access::{Access, AccessError},
            facts::ReplacementProof,
            memory::MemoryRefusal,
            state::State,
            value::{Address, Value},
        },
    },
    ir::{
        lanes::VectorConst,
        ops::{
            AccessQualifiers, SsaOp, VectorFaultMode, VectorMaskMode, VectorPackKind,
            VectorSegmentLayout,
        },
        value::ConstValue,
        variable::SsaVarId,
    },
    target::{Endianness, Target, TypeLayout},
};

/// The lane-wise loads of one interpretation, by their shared access policy.
pub struct VectorReads<'a> {
    /// Target layout, operation limits and memory admission.
    access: Access<'a>,
}

/// Where one lane-wise load reads, before it reads: its lanes, the element
/// each reads, and how the results are assembled.
pub struct LanePlan<T: Target> {
    /// The integer type of one element read.
    element: T::Type,
    /// The bytes of one element.
    element_bytes: usize,
    /// The access qualifiers every element read is made with.
    access: AccessQualifiers,
    /// Each element read, in order: the result, the lane and the address.
    reads: Vec<LaneRead>,
    /// The results and how each lane not read is filled.
    results: Vec<LaneResult>,
    /// How faults of reads after the first suppress them.
    fault: Option<FaultPlan>,
}

/// One element read into one lane of one result.
#[derive(Debug, Clone, Copy)]
struct LaneRead {
    /// Which result.
    result: usize,
    /// Which lane of it, or `None` for a replicated element filling every lane.
    lane: Option<u32>,
    /// Where it is read.
    address: Address,
}

/// One vector result and what its unread lanes hold.
struct LaneResult {
    /// The variable defined.
    dest: SsaVarId,
    /// Its lane width.
    lane_bits: u32,
    /// Its lane count.
    lanes: u32,
    /// The lane values before any read: a passthrough's lanes, zeros, or
    /// unknown where a merged lane has no known passthrough.
    base: Option<Vec<u128>>,
}

/// A first-faulting load's suppression rule.
#[derive(Debug, Clone, Copy)]
struct FaultPlan {
    /// Whether the first read faults normally rather than suppressing.
    first_faults: bool,
    /// The variable receiving the mask of elements read, where one is stated.
    output: Option<SsaVarId>,
}

/// What a planned load read and defines.
pub struct LaneOutcome<T: Target> {
    /// Each result's value.
    pub results: Vec<(SsaVarId, Value<T>)>,
    /// The span of every element read.
    pub spans: Vec<(Address, usize)>,
    /// Whether every element the plan names was read, so the spans are the
    /// load's whole footprint.
    pub complete: bool,
}

impl<'a> VectorReads<'a> {
    /// Borrows one instruction's access policy.
    pub const fn new(access: Access<'a>) -> Self {
        Self { access }
    }

    /// Plans `op`'s element reads, where it is a lane-wise load: `None` for
    /// any other operation, and `Some(None)` for one whose lanes, mask,
    /// indices or layout are not known, which reads nothing it can name.
    ///
    /// # Errors
    ///
    /// Returns a refusal for a malformed segment or element layout.
    pub fn plan<T: Target>(
        &self,
        op: &SsaOp<T>,
        state: &State<T>,
        layout: impl Fn(SsaVarId) -> Option<TypeLayout>,
        operand: impl Fn(SsaVarId) -> OperandLayout,
    ) -> Result<Option<Option<LanePlan<T>>>, AccessError> {
        let mask = |id: Option<SsaVarId>| -> Option<Option<PredicateMask>> {
            match id {
                None => Some(None),
                Some(id) => {
                    let value = state.value(id).into_constant()?;
                    PredicateMask::read((value, operand(id)), self.access.ptr_size()).map(Some)
                }
            }
        };
        let shape_of = |ty: &T::Type| {
            self.access
                .type_layout::<T>(ty)
                .and_then(|layout| layout.vector_shape())
                .filter(|shape| shape.is_valid() && shape.lane_bits.is_multiple_of(8))
        };
        let passthrough =
            |mode: VectorMaskMode, id: Option<SsaVarId>, lanes: u32, bits: u32| match mode {
                VectorMaskMode::Zero => Some(vec![0; lanes as usize]),
                VectorMaskMode::Merge => match state.value(id?).into_constant()? {
                    ConstValue::Vector(vector)
                        if vector.lanes() == lanes && vector.lane_bits() == bits =>
                    {
                        Some(vector.lane_values().collect())
                    }
                    _ => None,
                },
            };
        let plan = match op {
            SsaOp::VectorMaskedLoad {
                dest,
                addr,
                mask: active,
                passthrough: through,
                vector_type,
                mode,
                access,
            } => {
                let (Some(shape), Some(Some(active))) =
                    (shape_of(vector_type), mask(Some(*active)))
                else {
                    return Ok(Some(None));
                };
                let Some(base) = self.address(state, *addr, access) else {
                    return Ok(Some(None));
                };
                let mut reads = Vec::new();
                for lane in 0..shape.lane_count {
                    if Self::active(Some(&active), lane, shape.lane_bits, shape.lane_count)? {
                        reads.push(LaneRead {
                            result: 0,
                            lane: Some(lane),
                            address: Self::at(base, lane, shape.lane_bits, self.access)?,
                        });
                    }
                }
                self.contiguous(
                    shape.lane_bits,
                    *access,
                    reads,
                    vec![LaneResult::new(
                        *dest,
                        shape.lane_bits,
                        shape.lane_count,
                        passthrough(*mode, *through, shape.lane_count, shape.lane_bits),
                    )],
                    None,
                )
            }
            SsaOp::VectorGather {
                dest,
                base,
                indices,
                mask: active,
                passthrough: through,
                vector_type,
                mode,
                access,
            } => {
                let (Some(shape), Some(Some(active))) =
                    (shape_of(vector_type), mask(Some(*active)))
                else {
                    return Ok(Some(None));
                };
                let (Some(base), Some(ConstValue::Vector(offsets))) = (
                    self.address(state, *base, access),
                    state.value(*indices).into_constant(),
                ) else {
                    return Ok(Some(None));
                };
                if offsets.lanes() < shape.lane_count || offsets.lane_bits() > 64 {
                    return Ok(Some(None));
                }
                let mut reads = Vec::new();
                for lane in 0..shape.lane_count {
                    if Self::active(Some(&active), lane, shape.lane_bits, shape.lane_count)? {
                        let offset = i64::try_from(offsets.signed_lane(lane))
                            .map_err(|_| MemoryRefusal::AddressRange)?;
                        reads.push(LaneRead {
                            result: 0,
                            lane: Some(lane),
                            address: base
                                .moved(offset, self.access.ptr_size())
                                .ok_or(MemoryRefusal::AddressRange)?,
                        });
                    }
                }
                self.contiguous(
                    shape.lane_bits,
                    *access,
                    reads,
                    vec![LaneResult::new(
                        *dest,
                        shape.lane_bits,
                        shape.lane_count,
                        passthrough(*mode, *through, shape.lane_count, shape.lane_bits),
                    )],
                    None,
                )
            }
            SsaOp::VectorFaultingLoad {
                dest,
                fault,
                addr,
                mask: active,
                passthrough: through,
                vector_type,
                fault_mode,
                mask_mode,
                access,
            } => {
                let (Some(shape), Some(active)) = (shape_of(vector_type), mask(*active)) else {
                    return Ok(Some(None));
                };
                let Some(base) = self.address(state, *addr, access) else {
                    return Ok(Some(None));
                };
                let mut reads = Vec::new();
                for lane in 0..shape.lane_count {
                    if Self::active(active.as_ref(), lane, shape.lane_bits, shape.lane_count)? {
                        reads.push(LaneRead {
                            result: 0,
                            lane: Some(lane),
                            address: Self::at(base, lane, shape.lane_bits, self.access)?,
                        });
                    }
                }
                let suppression = match fault_mode {
                    VectorFaultMode::Normal => None,
                    VectorFaultMode::FirstFault => Some(true),
                    // Element zero faults normally only where it is active.
                    VectorFaultMode::FaultOnlyFirst => {
                        Some(reads.first().is_some_and(|read| read.lane == Some(0)))
                    }
                    VectorFaultMode::NoFault => Some(false),
                };
                self.contiguous(
                    shape.lane_bits,
                    *access,
                    reads,
                    vec![LaneResult::new(
                        *dest,
                        shape.lane_bits,
                        shape.lane_count,
                        passthrough(*mask_mode, *through, shape.lane_count, shape.lane_bits),
                    )],
                    suppression.map(|first_faults| FaultPlan {
                        first_faults,
                        output: *fault,
                    }),
                )
            }
            SsaOp::VectorSegmentLoad {
                dests,
                base,
                mask: active,
                vector_type,
                segments,
                layout: arrangement,
                access,
            } => {
                if *segments == 0 || usize::try_from(*segments).ok() != Some(dests.len()) {
                    return Err(MemoryRefusal::BlockLayout.into());
                }
                let (Some(shape), Some(active)) = (shape_of(vector_type), mask(*active)) else {
                    return Ok(Some(None));
                };
                let Some(base) = self.address(state, *base, access) else {
                    return Ok(Some(None));
                };
                let mut reads = Vec::new();
                for lane in 0..shape.lane_count {
                    if !Self::active(active.as_ref(), lane, shape.lane_bits, shape.lane_count)? {
                        continue;
                    }
                    for segment in 0..*segments {
                        let index = match arrangement {
                            VectorSegmentLayout::Interleaved => lane
                                .checked_mul(*segments)
                                .and_then(|index| index.checked_add(segment)),
                            VectorSegmentLayout::Consecutive => segment
                                .checked_mul(shape.lane_count)
                                .and_then(|index| index.checked_add(lane)),
                        }
                        .ok_or(MemoryRefusal::AddressRange)?;
                        reads.push(LaneRead {
                            result: usize::try_from(segment)
                                .map_err(|_| MemoryRefusal::AddressRange)?,
                            lane: Some(lane),
                            address: Self::at(base, index, shape.lane_bits, self.access)?,
                        });
                    }
                }
                // An inactive structure loads zero into each segment's lane.
                let results = dests
                    .iter()
                    .map(|dest| {
                        LaneResult::new(
                            *dest,
                            shape.lane_bits,
                            shape.lane_count,
                            Some(vec![0; shape.lane_count as usize]),
                        )
                    })
                    .collect();
                self.contiguous(shape.lane_bits, *access, reads, results, None)
            }
            SsaOp::VectorPackLoad {
                dest,
                addr,
                mask: active,
                passthrough: through,
                vector_type,
                element_bits,
                kind,
                mode,
                access,
            } => {
                if *kind != VectorPackKind::Expand {
                    return Err(MemoryRefusal::BlockLayout.into());
                }
                let (Some(shape), Some(Some(active))) =
                    (shape_of(vector_type), mask(Some(*active)))
                else {
                    return Ok(Some(None));
                };
                if *element_bits != shape.lane_bits {
                    return Err(MemoryRefusal::BlockLayout.into());
                }
                let Some(base) = self.address(state, *addr, access) else {
                    return Ok(Some(None));
                };
                // The k-th active lane takes the k-th element in memory.
                let mut reads = Vec::new();
                let mut taken = 0u32;
                for lane in 0..shape.lane_count {
                    if Self::active(Some(&active), lane, shape.lane_bits, shape.lane_count)? {
                        reads.push(LaneRead {
                            result: 0,
                            lane: Some(lane),
                            address: Self::at(base, taken, shape.lane_bits, self.access)?,
                        });
                        taken = taken.checked_add(1).ok_or(MemoryRefusal::AddressRange)?;
                    }
                }
                self.contiguous(
                    shape.lane_bits,
                    *access,
                    reads,
                    vec![LaneResult::new(
                        *dest,
                        shape.lane_bits,
                        shape.lane_count,
                        passthrough(*mode, *through, shape.lane_count, shape.lane_bits),
                    )],
                    None,
                )
            }
            SsaOp::VectorStructLoadReplicate(data) => {
                let bits = u32::from(data.element_bits);
                let (Some(&address), true) = (
                    data.inputs.first(),
                    usize::from(data.count) == data.outputs.len()
                        && bits != 0
                        && bits.is_multiple_of(8),
                ) else {
                    return Err(MemoryRefusal::BlockLayout.into());
                };
                let access = AccessQualifiers::default();
                let Some(base) = self.address(state, address, &access) else {
                    return Ok(Some(None));
                };
                let mut reads = Vec::new();
                let mut results = Vec::new();
                for (index, output) in data.outputs.iter().enumerate() {
                    let Some(shape) = layout(*output)
                        .and_then(|layout| layout.vector_shape())
                        .filter(|shape| shape.is_valid() && shape.lane_bits == bits)
                    else {
                        return Ok(Some(None));
                    };
                    reads.push(LaneRead {
                        result: index,
                        lane: None,
                        address: Self::at(
                            base,
                            u32::try_from(index).map_err(|_| MemoryRefusal::AddressRange)?,
                            bits,
                            self.access,
                        )?,
                    });
                    results.push(LaneResult::new(
                        *output,
                        bits,
                        shape.lane_count,
                        Some(vec![0; shape.lane_count as usize]),
                    ));
                }
                self.contiguous(bits, access, reads, results, None)
            }
            _ => return Ok(None),
        };
        Ok(Some(plan))
    }

    /// The access proof of every read a plan names: `None` where one could
    /// fault, or where a suppressed fault decides what the load defines.
    pub fn proof<T: Target>(
        &self,
        plan: &LanePlan<T>,
        state: &State<T>,
    ) -> Option<ReplacementProof> {
        let mut image: Option<(u64, u64)> = None;
        let mut accessible = false;
        for read in &plan.reads {
            match self.access.read_proof(
                state,
                &Value::Addr(read.address),
                &plan.element,
                plan.access,
            )? {
                ReplacementProof::ReadOnlyImage { address, len } => {
                    let end = address.checked_add(len)?;
                    image = Some(match image {
                        Some((start, stop)) => (start.min(address), stop.max(end)),
                        None => (address, end),
                    });
                }
                ReplacementProof::Accessible | ReplacementProof::Completed => accessible = true,
            }
        }
        match (image, accessible) {
            (Some((address, end)), false) => Some(ReplacementProof::ReadOnlyImage {
                address,
                len: end.checked_sub(address)?,
            }),
            // An empty plan reads nothing that could fault.
            (None, _) | (Some(_), true) => Some(ReplacementProof::Accessible),
        }
    }

    /// Reads a plan's elements and assembles its results.
    ///
    /// A suppressing first-faulting load reads up to the first element it
    /// cannot prove readable; where that leaves the elements read unknown,
    /// the results and the fault mask are unknown.
    ///
    /// # Errors
    ///
    /// Returns a refusal where an element read is refused, as an ordinary
    /// load's is.
    pub fn execute<T: Target>(
        &self,
        plan: &LanePlan<T>,
        state: &State<T>,
        output: impl Fn(SsaVarId) -> Option<TypeLayout>,
    ) -> Result<LaneOutcome<T>, AccessError> {
        let mut lanes: Vec<Option<Vec<u128>>> = plan
            .results
            .iter()
            .map(|result| result.base.clone())
            .collect();
        let mut spans = Vec::new();
        let mut complete = true;
        let mut read_mask = 0u128;
        // An element holding private stack-pointer bits keeps that
        // provenance in a result no lane value states.
        let mut stack = false;
        for (position, read) in plan.reads.iter().enumerate() {
            if let Some(fault) = plan.fault
                && (position > 0 || !fault.first_faults)
                && self
                    .access
                    .read_proof(
                        state,
                        &Value::Addr(read.address),
                        &plan.element,
                        plan.access,
                    )
                    .is_none()
            {
                // Whether this element faults, and so ends the load, is not
                // known: neither is what it defines.
                complete = false;
                lanes.iter_mut().for_each(|lanes| *lanes = None);
                break;
            }
            let loaded = self.access.load_with_footprint::<T>(
                state.memory(),
                &Value::Addr(read.address),
                &plan.element,
                plan.access.address_space,
            )?;
            match loaded.footprint {
                Some(span) => spans.push(span),
                None => complete = false,
            }
            stack |= loaded.value.may_reference_stack();
            let bits = loaded
                .value
                .constant()
                .and_then(|value| value.scalar_integer_pattern(self.access.ptr_size()))
                .map(|(bits, _)| bits);
            let (Some(target), Some(bits)) = (lanes.get_mut(read.result), bits) else {
                if let Some(target) = lanes.get_mut(read.result) {
                    *target = None;
                }
                continue;
            };
            if let Some(values) = target {
                match read.lane {
                    Some(lane) => {
                        if let Some(slot) = values.get_mut(lane as usize) {
                            *slot = bits;
                        }
                        read_mask |= 1u128.checked_shl(lane).unwrap_or(0);
                    }
                    None => values.iter_mut().for_each(|slot| *slot = bits),
                }
            }
        }
        let mut results: Vec<(SsaVarId, Value<T>)> = plan
            .results
            .iter()
            .zip(lanes)
            .map(|(result, lanes)| {
                let value = lanes
                    .and_then(|lanes| VectorConst::from_lanes(result.lane_bits, lanes))
                    .filter(|vector| vector.lanes() == result.lanes)
                    .map_or(
                        if stack {
                            Value::StackDerived
                        } else {
                            Value::Unknown
                        },
                        |vector| Value::Const(ConstValue::Vector(vector)),
                    );
                (result.dest, value)
            })
            .collect();
        if let Some(fault_output) = plan.fault.and_then(|fault| fault.output) {
            let value = complete
                .then(|| output(fault_output))
                .flatten()
                .filter(|layout| layout.is_integer())
                .and_then(|layout| {
                    let bytes = usize::try_from(layout.bit_width()?.div_ceil(8)).ok()?;
                    let pattern = read_mask.to_le_bytes();
                    ConstValue::from_layout_bytes(pattern.get(..bytes)?, layout, Endianness::Little)
                })
                .map_or(Value::Unknown, Value::Const);
            results.push((fault_output, value));
        }
        Ok(LaneOutcome {
            results,
            spans,
            complete,
        })
    }

    /// Builds a plan reading elements of `bits` bits.
    fn contiguous<T: Target>(
        &self,
        bits: u32,
        access: AccessQualifiers,
        reads: Vec<LaneRead>,
        results: Vec<LaneResult>,
        fault: Option<FaultPlan>,
    ) -> Option<LanePlan<T>> {
        Some(LanePlan {
            element: T::integer_type(bits, false)?,
            element_bytes: usize::try_from(bits / 8).ok()?,
            access,
            reads,
            results,
            fault,
        })
    }

    /// The address a pointer operand names in the access's space.
    fn address<T: Target>(
        &self,
        state: &State<T>,
        pointer: SsaVarId,
        access: &AccessQualifiers,
    ) -> Option<Address> {
        self.access
            .address_of(&state.value(pointer), access.address_space)
    }

    /// The address of element `index` of `bits` bits from `base`.
    fn at(
        base: Address,
        index: u32,
        bits: u32,
        access: Access<'_>,
    ) -> Result<Address, AccessError> {
        let offset = i64::from(index)
            .checked_mul(i64::from(bits / 8))
            .ok_or(MemoryRefusal::AddressRange)?;
        Ok(base
            .moved(offset, access.ptr_size())
            .ok_or(MemoryRefusal::AddressRange)?)
    }

    /// Whether lane `lane` of `count` lanes of `bits` bits is active.
    fn active(
        mask: Option<&PredicateMask>,
        lane: u32,
        bits: u32,
        count: u32,
    ) -> Result<bool, AccessError> {
        match mask {
            None => Ok(true),
            Some(mask) => Ok(mask
                .active(lane, bits, count)
                .ok_or(MemoryRefusal::BlockLayout)?),
        }
    }
}

impl<T: Target> LanePlan<T> {
    /// The bytes one element read spans.
    pub const fn element_bytes(&self) -> usize {
        self.element_bytes
    }

    /// The access qualifiers every element read is made with.
    pub const fn access(&self) -> AccessQualifiers {
        self.access
    }
}

impl LaneResult {
    /// A result of `lanes` lanes of `bits` bits, its unread lanes `base`.
    const fn new(dest: SsaVarId, lane_bits: u32, lanes: u32, base: Option<Vec<u128>>) -> Self {
        Self {
            dest,
            lane_bits,
            lanes,
            base,
        }
    }
}
