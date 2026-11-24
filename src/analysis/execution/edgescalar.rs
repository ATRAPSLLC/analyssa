//! Borrowed exact scalar observations using the existing typed constant readers.

use crate::{
    analysis::{
        Machine,
        consts::{Destination, OperandLayout},
        execution::value::Value,
    },
    ir::{float::FloatFormat, value::ConstValue},
    target::{Target, TypeLayout},
};

/// One attested fixed scalar interpretation, projected during request admission.
#[derive(Debug, Clone, Copy)]
pub enum ScalarLayout {
    /// An integer's read and materialization contracts, including native carriers.
    Integer {
        operand: OperandLayout,
        destination: Destination,
    },
    /// The logical value bit of an attested Boolean storage type.
    Boolean,
    /// One exact floating-point format, never host numeric equality.
    Float(FloatFormat),
}

impl ScalarLayout {
    /// Admits only coherent fixed numeric scalar layouts without target queries.
    pub fn of(layout: TypeLayout) -> Option<Self> {
        if layout.is_unknown()
            || layout.is_runtime_sized()
            || layout.vector_descriptor().is_some()
            || layout.vector_mask_descriptor().is_some()
            || layout.is_pointer()
            || layout.is_reference()
        {
            return None;
        }
        if layout.is_boolean() {
            layout.boolean_storage()?;
            return Some(Self::Boolean);
        }
        if layout.is_integer() {
            let destination = Destination::from_type_layout(layout);
            let operand = OperandLayout::from_type_layout(layout);
            let ty = destination.scalar_integer()?;
            (operand.scalar_integer() == Some(ty)).then_some(Self::Integer {
                operand,
                destination,
            })
        } else {
            let format = layout.float_format()?;
            (layout.bit_width() == Some(format.bits())).then_some(Self::Float(format))
        }
    }

    /// Reads by borrow without cloning malformed carriers or losing stack provenance.
    ///
    /// The only allocated result is an admitted scalar integer carrier of at most
    /// 128 bits. Partial/unknown/provenance-bearing values establish no exact fact.
    pub fn read<T: Target>(self, value: &Value<T>, machine: Machine) -> Option<ConstValue<T>> {
        if value.may_reference_stack() {
            return None;
        }
        match self {
            Self::Integer {
                operand,
                destination,
            } => {
                let bits = match value {
                    Value::Const(constant) => machine.constant_integer_bits(constant, operand)?,
                    Value::Integer(_) => value.integer_bits()?,
                    _ => return None,
                };
                machine.integer_bits_constant(bits, destination)
            }
            Self::Boolean => {
                let boolean = match value {
                    Value::Const(constant) => constant.logical_boolean(machine.ptr_size)?,
                    Value::Integer(_) => {
                        let bits = value.integer_bits()?;
                        if bits.width() != 1 {
                            return None;
                        }
                        bits.value()? != 0
                    }
                    _ => return None,
                };
                Some(ConstValue::from_bool(boolean))
            }
            Self::Float(format) => value
                .constant()?
                .as_float(format, machine.ptr_size)
                .map(ConstValue::Float),
        }
    }
}

#[cfg(test)]
#[path = "edgescalar_tests.rs"]
mod tests;
