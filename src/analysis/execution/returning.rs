//! Identity-only result contracts, composed across retired tail activations.

use crate::{
    analysis::execution::{bank::Placement, value::Value},
    ir::{float::FloatFormat, integer::IntegerBits, value::ConstValue},
    pointer::PointerSize,
    target::{Target, TypeLayout},
};

/// A host's justified interpretation of one selected implementation's return.
///
/// This contract admits an unchanged value, withholds it, or places a narrow
/// integer result in the register its convention extends it to
/// ([`Self::Placed`]). It performs no other arithmetic, widening, bitcast or
/// register selection. A native host must establish the physical return
/// correspondence before admitting a typed view.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ReturnBinding {
    /// Forwards the semantic result under the host's exact calling contract.
    #[default]
    Identity,
    /// Admits an unchanged scalar word or address of this exact width.
    Word(u32),
    /// Admits a floating constant in this exact format.
    Float(FloatFormat),
    /// Admits a vector with this exact lane geometry.
    Vector {
        /// Bits in each lane.
        lane_bits: u32,
        /// Number of lanes.
        lanes: u32,
    },
    /// Admits a canonical logical Boolean.
    Boolean,
    /// Places a narrow integer result in the whole register its convention
    /// writes it to ([`Placement::place`]): RV64 holds an `int` result
    /// sign-extended in its 64-bit register.
    Placed {
        /// Where the result sits in the register.
        placement: Placement,
        /// The register's complete layout.
        layout: TypeLayout,
    },
    /// Admits the result's low `bits`, as a read of its register's narrow
    /// view takes them: a pointer a model returns, read through an `int`
    /// declaration as its low 32 bits. An address result is first its bits
    /// (the explorer reads them through the image); a result of no known bits
    /// is unknown.
    Low {
        /// Width of the bits the destination receives.
        bits: u32,
    },
    /// Withholds exact result knowledge while retaining possible stack aliases.
    Unknown,
    /// Discards a result the enclosing function does not return.
    Discard,
}

impl ReturnBinding {
    /// Returns the exact fixed representation a caller already expects.
    ///
    /// Unknown, scalable and unsupported layouts withhold result knowledge.
    /// Pointer widths follow the target's machine-width pointer contract.
    #[must_use]
    pub fn for_layout(layout: TypeLayout, ptr_size: PointerSize) -> Self {
        if layout.is_unknown() || layout.is_runtime_sized() {
            return Self::Unknown;
        }
        if let Some(shape) = layout.vector_shape() {
            return Self::Vector {
                lane_bits: shape.lane_bits,
                lanes: shape.lane_count,
            };
        }
        if let Some(format) = layout.float_format() {
            return Self::Float(format);
        }
        if layout.is_boolean() {
            return Self::Boolean;
        }
        if layout.is_integer() || layout.is_pointer() || layout.is_reference() {
            let bits = layout
                .bit_width()
                .or_else(|| layout.is_pointer().then_some(ptr_size.bits()));
            return bits
                .filter(|bits| *bits > 0)
                .map_or(Self::Unknown, Self::Word);
        }
        Self::Unknown
    }

    /// Combines identity-only requirements without retaining retired frames.
    pub fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Self::Discard, _) | (_, Self::Discard) => Self::Discard,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Identity, binding) | (binding, Self::Identity) => binding,
            (left, right) if left == right => left,
            _ => Self::Unknown,
        }
    }

    /// Applies a result contract before caller definitions and observations.
    pub fn apply<T: Target>(
        self,
        returned: Option<Value<T>>,
        ptr_size: PointerSize,
    ) -> Option<Value<T>> {
        if self == Self::Discard {
            return None;
        }
        if let Self::Placed { placement, layout } = self {
            return returned.map(|value| placement.place(value, layout, ptr_size));
        }
        if let Self::Low { bits } = self {
            return returned.map(|value| {
                let stack = value.may_reference_stack();
                let known = match &value {
                    Value::Const(constant) => constant
                        .conversion_integer_pattern(ptr_size)
                        .and_then(|(pattern, width)| IntegerBits::constant(width, pattern)),
                    Value::Integer(_) => value.integer_bits(),
                    _ => None,
                };
                known
                    .filter(|known| known.width() >= bits)
                    .and_then(|known| known.converted(bits, true))
                    .map_or(
                        if stack {
                            Value::StackDerived
                        } else {
                            Value::Unknown
                        },
                        |low| {
                            low.value()
                                .and_then(|pattern| ConstValue::of_lane(bits, pattern))
                                .map_or_else(|| Value::partial_integer(low, stack), Value::Const)
                        },
                    )
            });
        }
        returned.map(|value| {
            let admit = match self {
                Self::Identity => true,
                Self::Unknown | Self::Discard => false,
                binding => compatible(&value, binding, ptr_size),
            };
            if admit {
                value
            } else {
                value.join(&Value::Unknown)
            }
        })
    }
}

/// Checks an existing representation without converting its bits or kind.
fn compatible<T: Target>(value: &Value<T>, binding: ReturnBinding, ptr_size: PointerSize) -> bool {
    if matches!(value, Value::Unknown | Value::StackDerived) {
        return true;
    }
    match (binding, value) {
        (ReturnBinding::Word(bits), Value::Addr(address)) => address.pointer_size().bits() == bits,
        (ReturnBinding::Word(bits), Value::Integer(partial)) => {
            partial.bits().is_some_and(|value| value.width() == bits)
        }
        (ReturnBinding::Word(bits), Value::Const(value)) => {
            value.bit_width(ptr_size) == Some(bits)
                || matches!(value, ConstValue::Vector(vector) if bits > 64 && vector.lanes() == 1 && vector.bits() == bits)
        }
        (ReturnBinding::Float(format), Value::Const(ConstValue::Float(value))) => {
            value.format() == format
        }
        (ReturnBinding::Vector { lane_bits, lanes }, Value::Const(ConstValue::Vector(value))) => {
            value.lane_bits() == lane_bits && value.lanes() == lanes
        }
        (ReturnBinding::Boolean, Value::Const(value)) => value.is_bool(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        analysis::execution::{bank::Placement, returning::ReturnBinding, value::Value},
        ir::value::ConstValue,
        target::Target,
        testing::{MACHINE, MockTarget, MockType},
    };

    /// A placed binding extends a narrow result into its register; an exact
    /// word binding admits only its own width.
    #[test]
    fn placed_results_extend_where_word_results_must_match() {
        let layout = MockTarget::type_layout(&MockType::I64, MACHINE.ptr_size).unwrap();
        let result = Some(Value::<MockTarget>::Const(ConstValue::I32(-1)));
        let placed = ReturnBinding::Placed {
            placement: Placement::Extended {
                bits: 32,
                signed: true,
            },
            layout,
        };
        assert_eq!(
            placed.apply(result.clone(), MACHINE.ptr_size),
            Some(Value::Const(ConstValue::I64(-1)))
        );
        assert_eq!(
            ReturnBinding::Word(64).apply(result, MACHINE.ptr_size),
            Some(Value::Unknown)
        );
        assert_eq!(placed.meet(placed), placed);
        assert_eq!(placed.meet(ReturnBinding::Word(64)), ReturnBinding::Unknown);
    }
}
