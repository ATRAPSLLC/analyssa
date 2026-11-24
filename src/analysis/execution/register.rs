//! Typed low-part reads of host-attested physical register arguments.

use crate::{
    analysis::{
        Machine,
        consts::{Destination, OperandLayout},
        execution::{scalar::Scalar, value::Value},
    },
    ir::{
        function::Widening,
        ops::{SsaOp, VectorShuffleSelector},
        variable::SsaVarId,
    },
    target::{
        Target, TypeLayout, VectorLaneKind, VectorShape, VectorShuffleLane, VectorShuffleMask,
    },
};

/// Reads a justified register view, preserving stack provenance on unsupported views.
pub fn read<T: Target>(
    value: &Value<T>,
    source: Option<TypeLayout>,
    target: &T::Type,
    machine: Machine,
) -> Value<T> {
    view(value, source, target, machine).unwrap_or_else(|| value.join(&Value::Unknown))
}

/// Reads `value`, held in a register as `source` lays it out, as `target`
/// views the register: its low bits, or the same bits as another kind of the
/// same width. `None` where no view of the register reads `target` from it.
pub fn view<T: Target>(
    value: &Value<T>,
    source: Option<TypeLayout>,
    target: &T::Type,
    machine: Machine,
) -> Option<Value<T>> {
    source
        .zip(T::type_layout(target, machine.ptr_size))
        .and_then(|(source, destination)| read_known(value, source, target, destination, machine))
}

/// Widens `value`, held in a register view laid out as `source`, by
/// `widening`: the register's whole content, and its layout. `None` where
/// `source` is no integer narrower than the widening, or the target states no
/// integer of its width.
pub fn widen<T: Target>(
    value: &Value<T>,
    source: Option<TypeLayout>,
    widening: Widening,
    machine: Machine,
) -> Option<(Value<T>, TypeLayout)> {
    let source = source.filter(|source| source.is_integer())?;
    if source.bit_width()? >= widening.bits {
        return None;
    }
    let target = T::integer_type(widening.bits, widening.signed)?;
    let destination = T::type_layout(&target, machine.ptr_size)?;
    let operand = SsaVarId::from_index(0);
    let op = SsaOp::IntConv {
        dest: operand,
        operand,
        target,
        unsigned: !widening.signed,
        overflow_check: false,
    };
    Some((
        evaluate(value, source, destination, &op, machine),
        destination,
    ))
}

/// Evaluates one supported representation read without inventing upper bits.
fn read_known<T: Target>(
    value: &Value<T>,
    source: TypeLayout,
    target: &T::Type,
    destination: TypeLayout,
    machine: Machine,
) -> Option<Value<T>> {
    if source.is_unknown()
        || destination.is_unknown()
        || source.is_runtime_sized()
        || destination.is_runtime_sized()
    {
        return None;
    }
    // Native pointer types may leave their width to the machine contract.
    let source_bits = source
        .bit_width()
        .or_else(|| source.is_pointer().then_some(machine.ptr_size.bits()))?;
    let target_bits = destination
        .bit_width()
        .or_else(|| destination.is_pointer().then_some(machine.ptr_size.bits()))?;
    if target_bits == 0 || target_bits > source_bits {
        return None;
    }
    let operand = SsaVarId::from_index(0);
    let op = if source.is_integer() && destination.is_integer() {
        SsaOp::IntConv {
            dest: operand,
            operand,
            target: target.clone(),
            unsigned: true,
            overflow_check: false,
        }
    } else if source.is_pointer() && destination.is_integer() {
        SsaOp::PtrToInt {
            dest: operand,
            operand,
            target: target.clone(),
        }
    } else if source.is_integer() && destination.is_pointer() && target_bits == source_bits {
        SsaOp::IntToPtr {
            dest: operand,
            operand,
            target: target.clone(),
        }
    } else if let Some(shape) = source.vector_shape() {
        if destination.vector_shape().is_some() && target_bits == source_bits {
            SsaOp::VectorReinterpret {
                dest: operand,
                value: operand,
                target_type: target.clone(),
            }
        } else if destination.float_format().is_some() {
            return low_float(value, shape, target, destination, machine);
        } else {
            return None;
        }
    } else if source_bits == target_bits
        && ((source.float_format().is_some()
            && source.float_format() == destination.float_format())
            || (source.is_pointer() && destination.is_pointer())
            || (source.is_reference() && destination.is_reference())
            || (source.is_boolean()
                && destination.is_boolean()
                && source.boolean_storage().is_some()
                && source.boolean_storage() == destination.boolean_storage())
            || (source.vector_mask_descriptor().is_some()
                && source.vector_mask_descriptor() == destination.vector_mask_descriptor()))
    {
        return Some(value.clone());
    } else {
        return None;
    };
    Some(evaluate(value, source, destination, &op, machine))
}

/// Reads the lowest floating lane through the shared reinterpret, shuffle and bitcast kernels.
fn low_float<T: Target>(
    value: &Value<T>,
    source: VectorShape,
    target: &T::Type,
    destination: TypeLayout,
    machine: Machine,
) -> Option<Value<T>> {
    let lane_bits = destination.float_format()?.bits();
    if destination.bit_width() != Some(lane_bits) || !source.total_bits.is_multiple_of(lane_bits) {
        return None;
    }
    let lanes = source.total_bits.checked_div(lane_bits)?;
    let viewed = VectorShape::new(lanes, VectorLaneKind::Integer, lane_bits, source.total_bits)?;
    let low = VectorShape::new(1, VectorLaneKind::Integer, lane_bits, lane_bits)?;
    let source_type = T::vector_type(source)?;
    let viewed_type = T::vector_type(viewed)?;
    let low_type = T::vector_type(low)?;
    let source_layout = T::type_layout(&source_type, machine.ptr_size)?;
    let viewed_layout = T::type_layout(&viewed_type, machine.ptr_size)?;
    let low_layout = T::type_layout(&low_type, machine.ptr_size)?;
    if source_layout.vector_shape() != Some(source)
        || viewed_layout.vector_shape() != Some(viewed)
        || low_layout.vector_shape() != Some(low)
    {
        return None;
    }
    let operand = SsaVarId::from_index(0);
    let viewed_value = evaluate(
        value,
        source_layout,
        viewed_layout,
        &SsaOp::VectorReinterpret {
            dest: operand,
            value: operand,
            target_type: viewed_type,
        },
        machine,
    );
    let low_value = evaluate(
        &viewed_value,
        viewed_layout,
        low_layout,
        &SsaOp::VectorShuffle {
            dest: operand,
            left: operand,
            right: None,
            selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                VectorShuffleLane::Left(0),
            ])),
        },
        machine,
    );
    Some(evaluate(
        &low_value,
        low_layout,
        destination,
        &SsaOp::Bitcast {
            dest: operand,
            operand,
            target: target.clone(),
        },
        machine,
    ))
}

/// Runs a single-input typed operation with no dependence on the caller's SSA namespace.
fn evaluate<T: Target>(
    value: &Value<T>,
    source: TypeLayout,
    destination: TypeLayout,
    op: &SsaOp<T>,
    machine: Machine,
) -> Value<T> {
    Scalar::new(machine).evaluate_with_operand_layouts(
        op,
        Destination::from_type_layout(destination),
        |_| value.clone(),
        |_| OperandLayout::from_type_layout(source),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            exception::HandlerKind, float::FloatFormat, integer::IntegerBits, lanes::VectorConst,
            value::ConstValue,
        },
        pointer::PointerSize,
        testing::{MACHINE, MockTarget, MockType},
    };

    /// A host that names each fixed vector and scalar floating representation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct ViewTarget;

    /// The exact representations this host makes available to register reads.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum ViewType {
        Unknown,
        Float(FloatFormat),
        Vector(VectorShape),
    }

    impl Target for ViewTarget {
        type TypeRef = ();
        type MethodRef = ();
        type FieldRef = ();
        type SigRef = ();
        type SymbolRef = ();
        type ExceptionKind = ();
        type Type = ViewType;
        type OriginalInstruction = ();
        type LocalSignature = ();
        type VariableFacts = ();
        type InstructionFacts = ();
        type FunctionFacts = ();
        type Capability = ();

        fn ptr_bytes(&self) -> u32 {
            8
        }
        fn synthetic_instruction() {}
        fn unknown_type() -> ViewType {
            ViewType::Unknown
        }
        fn is_integer(_: &ViewType) -> bool {
            false
        }
        fn is_floating(ty: &ViewType) -> bool {
            matches!(ty, ViewType::Float(_))
        }
        fn is_signed(_: &ViewType) -> bool {
            false
        }
        fn is_pointer(_: &ViewType) -> bool {
            false
        }
        fn is_reference(_: &ViewType) -> bool {
            false
        }
        fn is_unknown(ty: &ViewType) -> bool {
            matches!(ty, ViewType::Unknown)
        }
        fn bit_width(ty: &ViewType) -> Option<u32> {
            match ty {
                ViewType::Unknown => None,
                ViewType::Float(format) => Some(format.bits()),
                ViewType::Vector(shape) => Some(shape.total_bits),
            }
        }
        fn float_format(ty: &ViewType) -> Option<FloatFormat> {
            if let ViewType::Float(format) = ty {
                Some(*format)
            } else {
                None
            }
        }
        fn vector_shape(ty: &ViewType) -> Option<VectorShape> {
            if let ViewType::Vector(shape) = ty {
                Some(*shape)
            } else {
                None
            }
        }
        fn vector_type(shape: VectorShape) -> Option<ViewType> {
            Some(ViewType::Vector(shape))
        }
        fn type_layout(ty: &ViewType, ptr_size: PointerSize) -> Option<TypeLayout> {
            Some(TypeLayout::of::<Self>(ty, ptr_size))
        }
        fn instruction_mnemonic(_: &()) -> &'static str {
            "test"
        }
        fn instruction_rva(_: &()) -> u64 {
            0
        }
        fn handler_kind(_: &()) -> HandlerKind {
            HandlerKind::Catch
        }
    }

    /// Low floating reads reinterpret lane zero without including the upper lanes.
    #[test]
    fn low_float_reads_exact_low_lane() {
        for (format, bits) in [
            (FloatFormat::Single, 0x3f80_0000u128),
            (FloatFormat::Double, 0x3ff0_0000_0000_0000),
        ] {
            for width in [128, 512] {
                for lane_bits in [format.bits(), width] {
                    let lanes = width.checked_div(format.bits()).unwrap();
                    let mut words = vec![u128::MAX; usize::try_from(lanes).unwrap()];
                    *words.first_mut().unwrap() = bits;
                    let vector = VectorConst::from_lanes(format.bits(), words)
                        .unwrap()
                        .resized(lane_bits, width.checked_div(lane_bits).unwrap())
                        .unwrap();
                    let shape = VectorShape::new(
                        width.checked_div(lane_bits).unwrap(),
                        VectorLaneKind::Integer,
                        lane_bits,
                        width,
                    )
                    .unwrap();
                    let source =
                        ViewTarget::type_layout(&ViewType::Vector(shape), MACHINE.ptr_size);
                    let value = Value::<ViewTarget>::Const(ConstValue::Vector(vector));
                    let actual = read(&value, source, &ViewType::Float(format), MACHINE);
                    let constant = actual.constant().unwrap();
                    assert_eq!(
                        constant.float_bits(),
                        Some((format, bits)),
                        "{format:?} {shape:?}"
                    );
                }
            }
        }
    }

    /// Unknown vectors retain uncertainty and a possible stack escape through low-lane evaluation.
    #[test]
    fn low_float_preserves_unknown_and_stack_provenance() {
        let shape = VectorShape::new(1, VectorLaneKind::Integer, 512, 512).unwrap();
        let source = ViewTarget::type_layout(&ViewType::Vector(shape), MACHINE.ptr_size);
        for format in [FloatFormat::Single, FloatFormat::Double] {
            for value in [Value::<ViewTarget>::Unknown, Value::StackDerived] {
                assert_eq!(
                    read(&value, source, &ViewType::Float(format), MACHINE),
                    value
                );
            }
        }
    }

    /// Partial integer reads retain known low bits and their possible stack provenance.
    #[test]
    fn register_view_keeps_partial_integer_bits() {
        let partial = IntegerBits::new(64, 0xff00, 2).unwrap();
        let value = Value::<MockTarget>::partial_integer(partial, true);
        let source = MockTarget::type_layout(&MockType::U64, MACHINE.ptr_size);
        let actual = read(&value, source, &MockType::U8, MACHINE);
        assert_eq!(actual.integer_bits(), IntegerBits::new(8, 0, 2));
        assert!(actual.may_reference_stack());
    }
}
