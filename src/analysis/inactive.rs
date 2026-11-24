//! Typed, bounded result plans for lane computations that execute no element.

use crate::{
    PointerSize,
    ir::{
        ops::{Kernel, RuntimeValueKind, SsaOp, VectorFaultMode, VectorRounding, VectorStatus},
        value::ConstValue,
        variable::SsaVarId,
    },
    target::{Target, TypeLayout, VectorLaneKind, VectorMaskDescriptor},
};

/// One original result and its fixed declared storage demand.
#[derive(Debug)]
pub struct InactiveResult<T: Target> {
    /// A pure Copy, Const or Unspecified definition of the original destination.
    pub definition: SsaOp<T>,
    /// Attested result facts reused by the caller's existing materialization limits.
    pub layout: TypeLayout,
}

/// A complete result plan, constructed without reading inactive data operands.
#[derive(Debug)]
pub struct InactivePlan<T: Target> {
    /// Every result in the computation's original order, possibly empty for a store.
    pub results: Vec<InactiveResult<T>>,
}

/// Returns a complete pure result plan when the typed active interval is empty.
///
/// Layouts must be host-attested, and `same_type` must compare declared types,
/// not merely their bit widths. Only the length and, when needed, restart value
/// are read. Runtime-sized or unknown layouts and unsupported status layouts
/// decline. The caller admits result storage before reading passthrough values.
/// This proof neither admits instruction sources nor charges execution work.
#[must_use]
pub fn plan<'value, T: Target + 'value>(
    op: &SsaOp<T>,
    ptr_size: PointerSize,
    constant: impl Fn(SsaVarId) -> Option<&'value ConstValue<T>>,
    layout: impl Fn(SsaVarId) -> Option<TypeLayout>,
    same_type: impl Fn(SsaVarId, SsaVarId) -> bool,
) -> Option<InactivePlan<T>> {
    let SsaOp::Compute(data) = op else {
        return None;
    };
    let kind = data.kind;
    let lanes = kind.lanes?;
    if !kind.is_valid()
        || matches!(kind.kernel, Kernel::MaskSummary { .. })
        || data.inputs.len() != kind.input_count()
        || data.outputs.len() != kind.output_count()
    {
        return None;
    }
    let fault = match kind.kernel {
        Kernel::Memory {
            fault: VectorFaultMode::FaultOnlyFirst,
            ..
        } => true,
        Kernel::Memory {
            fault: VectorFaultMode::FirstFault | VectorFaultMode::NoFault,
            ..
        } => return None,
        _ => false,
    };
    let count = kind.data_outputs();
    let status = kind.status != VectorStatus::None;
    if count
        .checked_add(usize::from(fault))?
        .checked_add(usize::from(status))?
        != data.outputs.len()
    {
        return None;
    }
    // The descriptor bounds the interface (memory fields are u8). Do not
    // allocate proportional to a malformed caller-provided operand vector.
    for (index, dest) in data.outputs.iter().enumerate() {
        if data.outputs.get(..index)?.contains(dest) || data.inputs.contains(dest) {
            return None;
        }
    }
    let restart = *data.inputs.first()?;
    let length = *data.inputs.get(1)?;
    let restart_width = control_width(layout(restart)?)?;
    if restart_width > 64 {
        return None;
    }
    let length_width = control_width(layout(length)?)?;
    if length_width > 64 {
        return None;
    }
    let length_value = read_control(constant(length)?, length_width, ptr_size)?;
    if length_value != 0 {
        let restart_value = read_control(constant(restart)?, restart_width, ptr_size)?;
        if restart_value < length_value {
            return None;
        }
    }

    // Validate the entire interface before constructing any replacement.
    // No values behind addresses, masks, data operands or environments are read.
    let inputs = data
        .inputs
        .iter()
        .map(|&var| layout(var))
        .collect::<Option<Vec<_>>>()?;
    let outputs = data
        .outputs
        .iter()
        .map(|&var| layout(var))
        .collect::<Option<Vec<_>>>()?;
    for operand in inputs
        .iter()
        .skip(kind.control_input_count())
        .chain(outputs.iter())
    {
        storage_bytes(*operand, ptr_size)?;
    }
    if matches!(
        kind.rounding,
        VectorRounding::FloatingEnvironment | VectorRounding::FixedEnvironment
    ) {
        control_width(*inputs.get(2usize.checked_add(usize::from(kind.inactive().is_some()))?)?)?;
    }
    if kind.mask_tail() {
        control_width(*inputs.get(kind.control_input_count().checked_sub(1)?)?)?;
    }
    if kind.inactive().is_some() {
        let predicate = *inputs.get(2)?;
        if !predicate.is_integer() && predicate.vector_mask_descriptor().is_none() {
            return None;
        }
        let capacity = storage_bits(predicate, ptr_size)?;
        let stride = kind.predicate_bit(1).unwrap_or(1);
        for operand in outputs.iter().take(count).chain(
            inputs.iter().skip(
                kind.control_input_count()
                    .checked_add(kind.passthrough_count())?,
            ),
        ) {
            if let Some(shape) = operand.vector_shape()
                && u64::from(shape.lane_count).checked_mul(stride)? > capacity
            {
                return None;
            }
        }
    }
    for (index, &dest) in data.outputs.iter().take(count).enumerate() {
        if !lanes.no_passthrough {
            let source = *data
                .inputs
                .get(kind.control_input_count().checked_add(index)?)?;
            if !same_type(dest, source) {
                return None;
            }
        }
    }
    if fault && !same_type(*data.outputs.get(count)?, length) {
        return None;
    }
    let zero = if status {
        Some(status_zero::<T>(*outputs.last()?, kind.status, ptr_size)?)
    } else {
        None
    };
    let mut results = Vec::with_capacity(data.outputs.len());
    for (index, &dest) in data.outputs.iter().enumerate() {
        // A runtime-sized or unsupported result storage declines the plan.
        storage_bytes(*outputs.get(index)?, ptr_size)?;
        let definition = if index < count {
            if lanes.no_passthrough {
                SsaOp::RuntimeValue {
                    dest,
                    kind: RuntimeValueKind::Unspecified,
                }
            } else {
                SsaOp::Copy {
                    dest,
                    src: *data
                        .inputs
                        .get(kind.control_input_count().checked_add(index)?)?,
                }
            }
        } else if fault && index == count {
            SsaOp::Copy { dest, src: length }
        } else {
            SsaOp::Const {
                dest,
                value: zero.clone()?,
            }
        };
        results.push(InactiveResult {
            definition,
            layout: *outputs.get(index)?,
        });
    }
    Some(InactivePlan { results })
}

/// Returns an attested scalar integer width supported by the integer codec.
fn control_width(layout: TypeLayout) -> Option<u32> {
    if !layout.is_integer()
        || layout.is_boolean()
        || layout.is_pointer()
        || layout.is_reference()
        || layout.is_unknown()
        || layout.is_runtime_sized()
        || layout.vector_descriptor().is_some()
        || layout.vector_mask_descriptor().is_some()
    {
        return None;
    }
    layout.bit_width().filter(|width| (1..=128).contains(width))
}

/// Reads a scalar pattern using the existing typed-width integer semantics.
fn read_control<T: Target>(
    value: &ConstValue<T>,
    width: u32,
    ptr_size: PointerSize,
) -> Option<u64> {
    u64::try_from(value.scalar_pattern_at_width(width, ptr_size)?).ok()
}

/// Returns a fixed numeric, address or predicate extent without guessing geometry.
fn storage_bits(layout: TypeLayout, ptr_size: PointerSize) -> Option<u64> {
    if layout.is_unknown() || layout.is_reference() || layout.is_runtime_sized() {
        return None;
    }
    let width = layout
        .bit_width()
        .or_else(|| layout.is_pointer().then_some(ptr_size.bits()))?;
    if width == 0 {
        return None;
    }
    if let Some(shape) = layout.vector_shape() {
        if !shape.is_valid()
            || shape.lane_count.checked_mul(shape.lane_bits)? != width
            || matches!(shape.lane_kind, VectorLaneKind::Float(format) if format.bits() != shape.lane_bits)
        {
            return None;
        }
    } else if let Some(VectorMaskDescriptor::Fixed(shape)) = layout.vector_mask_descriptor() {
        if !shape.is_valid() || shape.lane_count.checked_mul(shape.lane_bits)? != width {
            return None;
        }
    } else if !layout.is_integer()
        && !layout.is_floating()
        && !layout.is_pointer()
        && !layout.is_boolean()
    {
        return None;
    }
    Some(u64::from(width))
}

/// Returns the bounded byte extent of one logical result, including packed bits.
fn storage_bytes(layout: TypeLayout, ptr_size: PointerSize) -> Option<usize> {
    let bits = storage_bits(layout, ptr_size)?;
    if layout.is_boolean() {
        return Some(usize::from(layout.boolean_storage()?.bytes()));
    }
    usize::try_from(bits.div_ceil(8)).ok()
}

/// Constructs the exact scalar zero for an admitted status representation.
fn status_zero<T: Target>(
    layout: TypeLayout,
    status: VectorStatus,
    ptr_size: PointerSize,
) -> Option<ConstValue<T>> {
    if status == VectorStatus::Saturation && layout.is_boolean() {
        layout.boolean_storage()?;
        return Some(ConstValue::False);
    }
    let bits = control_width(layout)?;
    if status == VectorStatus::FloatingExceptions && bits < 5 {
        return None;
    }
    let integer = layout.integer_conversion_type()?;
    if integer.bits() != bits || integer.signed() != layout.is_signed() {
        return None;
    }
    integer.constant(0, ptr_size)
}
