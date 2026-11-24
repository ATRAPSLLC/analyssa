//! Known destinations of vector writes whose values remain uninterpreted.

use crate::{
    analysis::{
        budget::{Budget, Exhaustion},
        consts::{OperandLayout, PredicateMask},
        execution::{
            access::{Access, AccessError, WritePreflight},
            memory::MemoryRefusal,
            state::State,
            value::Value,
        },
    },
    ir::{
        ops::{SsaOp, VectorPackKind, VectorSegmentLayout},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// Retains each placed write before the opaque operation forgets its values.
pub struct VectorWrites<'a> {
    /// Target layout, operation limits and memory admission.
    access: Access<'a>,
}

/// Memory placement of one vector instruction's selected lanes.
enum Placement {
    /// Selected lanes retain their original offsets.
    Masked,
    /// Selected lanes use signed byte offsets supplied by a vector.
    Scatter(SsaVarId),
    /// Selected lanes are packed contiguously at the destination.
    Packed(u32),
    /// Several fields are written for each selected element.
    Segmented {
        /// Number of fields at each selected element.
        count: u32,
        /// Physical arrangement of the fields.
        layout: VectorSegmentLayout,
    },
}

impl<'a> VectorWrites<'a> {
    /// Borrows one instruction's access policy and cached explicit layout.
    pub const fn new(access: Access<'a>) -> Self {
        Self { access }
    }

    /// Checks write admission and identifies a proven empty footprint.
    pub fn preflight<T: Target>(
        &self,
        op: &SsaOp<T>,
        state: &mut State<T>,
        layout: impl Fn(SsaVarId) -> OperandLayout,
    ) -> Result<WritePreflight, AccessError> {
        let (base, ty, mask, space, placement) = match op {
            SsaOp::VectorMaskedStore {
                addr,
                vector_type,
                mask,
                access,
                ..
            } => (
                *addr,
                vector_type,
                Some(*mask),
                access.address_space,
                Placement::Masked,
            ),
            SsaOp::VectorScatter {
                base,
                indices,
                vector_type,
                mask,
                access,
                ..
            } => (
                *base,
                vector_type,
                Some(*mask),
                access.address_space,
                Placement::Scatter(*indices),
            ),
            SsaOp::VectorPackStore {
                addr,
                vector_type,
                mask,
                element_bits,
                kind,
                access,
                ..
            } if *kind == VectorPackKind::Compress => (
                *addr,
                vector_type,
                Some(*mask),
                access.address_space,
                Placement::Packed(*element_bits),
            ),
            SsaOp::VectorPackStore { .. } => return Err(MemoryRefusal::BlockLayout.into()),
            SsaOp::VectorSegmentStore {
                base,
                values,
                vector_type,
                mask,
                segments,
                layout,
                access,
            } => {
                if *segments == 0 || usize::try_from(*segments).ok() != Some(values.len()) {
                    return Err(MemoryRefusal::BlockLayout.into());
                }
                (
                    *base,
                    vector_type,
                    *mask,
                    access.address_space,
                    Placement::Segmented {
                        count: *segments,
                        layout: *layout,
                    },
                )
            }
            _ => return Ok(WritePreflight::Unrelated),
        };
        // A lane is written for certain where the operation has no mask or
        // the run knows it.
        let unmasked = mask.is_none();
        let mask = mask.and_then(|id| {
            let value = state.value(id).into_constant()?;
            PredicateMask::read((value, layout(id)), self.access.ptr_size())
        });
        if mask.as_ref().and_then(PredicateMask::empty) == Some(true) {
            return Ok(WritePreflight::Empty);
        }
        let Some(shape) = self
            .access
            .type_layout::<T>(ty)
            .and_then(|ty| ty.vector_shape())
        else {
            let pointer = state.value(base);
            if !self.access.admit_unresolved_write(
                state.memory_mut(),
                self.access.address_of(&pointer, space),
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
        if !shape.is_valid() || !shape.lane_bits.is_multiple_of(8) {
            return Err(MemoryRefusal::BlockLayout.into());
        }
        if let Placement::Packed(bits) = placement
            && bits != shape.lane_bits
        {
            return Err(MemoryRefusal::BlockLayout.into());
        }
        let segments = match placement {
            Placement::Segmented { count, .. } => count,
            _ => 1,
        };
        let lane_bytes =
            usize::try_from(shape.lane_bits / 8).map_err(|_| MemoryRefusal::AddressRange)?;
        let extent = usize::try_from(shape.total_bits / 8)
            .ok()
            .and_then(|bytes| bytes.checked_mul(usize::try_from(segments).ok()?))
            .ok_or(MemoryRefusal::AddressRange)?;
        let limit = self.access.buffer_limit();
        if extent > limit {
            return Err(Exhaustion {
                budget: Budget::ExecutionBufferBytes,
                limit,
                demand: extent,
            }
            .into());
        }
        if space.is_some() {
            if !self
                .access
                .admit_unresolved_write(state.memory_mut(), None, Some(extent))
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
        let pointer = state.value(base);
        let Some(base) = self.access.address_of(&pointer, None) else {
            if !self
                .access
                .admit_unresolved_write(state.memory_mut(), None, Some(extent))
            {
                return Err(state
                    .memory()
                    .refusal()
                    .unwrap_or(MemoryRefusal::UnresolvedWrite)
                    .into());
            }
            return Ok(WritePreflight::Possible);
        };
        let indices = match placement {
            Placement::Scatter(id) => match state.value(id).into_constant() {
                Some(ConstValue::Vector(indices))
                    if indices.lanes() >= shape.lane_count && indices.lane_bits() <= 128 =>
                {
                    Some(indices)
                }
                _ => {
                    if !self.access.admit_unresolved_write(
                        state.memory_mut(),
                        Some(base),
                        Some(extent),
                    ) {
                        return Err(state
                            .memory()
                            .refusal()
                            .unwrap_or(MemoryRefusal::UnresolvedWrite)
                            .into());
                    }
                    return Ok(WritePreflight::Possible);
                }
            },
            _ => None,
        };
        let mut written = 0u32;
        for lane in 0..shape.lane_count {
            if let Some(mask) = &mask
                && !mask
                    .active(lane, shape.lane_bits, shape.lane_count)
                    .ok_or(MemoryRefusal::BlockLayout)?
            {
                continue;
            }
            for segment in 0..segments {
                let offset = if let Some(indices) = &indices {
                    let shift = 128u32
                        .checked_sub(indices.lane_bits())
                        .ok_or(MemoryRefusal::BlockLayout)?;
                    let signed = indices
                        .lane(lane)
                        .checked_shl(shift)
                        .ok_or(MemoryRefusal::AddressRange)?
                        .cast_signed()
                        >> shift;
                    i64::try_from(signed).map_err(|_| MemoryRefusal::AddressRange)?
                } else {
                    let index = match placement {
                        Placement::Packed(_) => u64::from(written),
                        Placement::Segmented {
                            layout: VectorSegmentLayout::Interleaved,
                            ..
                        } => u64::from(lane)
                            .checked_mul(u64::from(segments))
                            .and_then(|index| index.checked_add(u64::from(segment)))
                            .ok_or(MemoryRefusal::AddressRange)?,
                        Placement::Segmented {
                            layout: VectorSegmentLayout::Consecutive,
                            ..
                        } => u64::from(segment)
                            .checked_mul(u64::from(shape.lane_count))
                            .and_then(|index| index.checked_add(u64::from(lane)))
                            .ok_or(MemoryRefusal::AddressRange)?,
                        _ => u64::from(lane),
                    };
                    index
                        .checked_mul(
                            u64::try_from(lane_bytes).map_err(|_| MemoryRefusal::AddressRange)?,
                        )
                        .and_then(|offset| i64::try_from(offset).ok())
                        .ok_or(MemoryRefusal::AddressRange)?
                };
                let destination = base
                    .moved(offset, self.access.ptr_size())
                    .ok_or(MemoryRefusal::AddressRange)?;
                self.access.preflight_block_write(
                    state.memory_mut(),
                    &Value::<T>::Addr(destination),
                    lane_bytes,
                    1,
                    Some(false),
                    unmasked || mask.is_some(),
                )?;
            }
            written = written.checked_add(1).ok_or(MemoryRefusal::AddressRange)?;
        }
        Ok(if written == 0 {
            WritePreflight::Empty
        } else {
            WritePreflight::Possible
        })
    }
}
