//! Path-local physical values transported between immutable source namespaces.

use std::{collections::BTreeMap, fmt, mem, rc::Rc};

use crate::{
    analysis::execution::{returning::ReturnBinding, value::Value},
    ir::{CallInputKey, integer::IntegerBits},
    pointer::PointerSize,
    target::{Target, TypeLayout},
};

/// One complete physical location receiving a model's actual result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineOutput {
    /// Host-defined physical storage identity.
    pub key: CallInputKey,
    /// Complete storage representation of the location.
    pub layout: TypeLayout,
    /// How the result occupies the location, under the actual calling contract.
    pub placement: Placement,
}

impl MachineOutput {
    /// Returns the location's value after a call whose result is `result`
    /// ([`Placement::place`]).
    ///
    /// # Arguments
    ///
    /// * `result` - The model's actual result, at the width its contract states.
    /// * `pointer` - The machine's pointer size.
    pub fn value<T: Target>(&self, result: Value<T>, pointer: PointerSize) -> Value<T> {
        self.placement.place(result, self.layout, pointer)
    }
}

/// Where a model's result sits in its physical location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Placement {
    /// The result is the whole location, unchanged.
    Whole,
    /// The result is the location's low `bits`; the convention leaves the rest unstated.
    Low {
        /// Width of the result.
        bits: u32,
    },
    /// The result is the location's low `bits`, extended over the rest.
    Extended {
        /// Width of the result.
        bits: u32,
        /// Whether the rest holds copies of the result's sign rather than zeros.
        signed: bool,
    },
}

impl Placement {
    /// Returns the value of a location of `layout` that holds `result` as
    /// this placement states.
    ///
    /// A narrow result fills the location's low bits. The bits above it are
    /// unknown where the convention leaves them unstated, and copies of its
    /// sign or zeros where the convention extends it. A location whose bits are
    /// all known holds a constant of its own integer type. A result that is not
    /// an integer of the placed width leaves the location unknown, except that
    /// possible stack provenance is kept.
    ///
    /// # Arguments
    ///
    /// * `result` - The actual result, at the width its contract states.
    /// * `layout` - The location's complete layout.
    /// * `pointer` - The machine's pointer size.
    pub fn place<T: Target>(
        self,
        result: Value<T>,
        layout: TypeLayout,
        pointer: PointerSize,
    ) -> Value<T> {
        let (bits, extension) = match self {
            Self::Whole => return result,
            Self::Low { bits } => (bits, None),
            Self::Extended { bits, signed } => (bits, Some(signed)),
        };
        let narrow = match &result {
            Value::Const(value) => value
                .conversion_integer_pattern(pointer)
                .filter(|&(_, width)| width == bits)
                .and_then(|(pattern, _)| IntegerBits::constant(bits, pattern)),
            Value::Integer(_) => result.integer_bits().filter(|known| known.width() == bits),
            _ => None,
        };
        let cell = narrow.zip(layout.bit_width()).and_then(|(narrow, width)| {
            if width <= bits {
                return None;
            }
            match extension {
                Some(signed) => narrow.converted(width, !signed),
                None => IntegerBits::new(width, narrow.zeros(), narrow.ones()),
            }
        });
        let stack = result.may_reference_stack();
        let Some(cell) = cell else {
            return if stack {
                Value::StackDerived
            } else {
                Value::Unknown
            };
        };
        cell.value()
            .filter(|_| !stack)
            .zip(layout.integer_conversion_type())
            .and_then(|(pattern, integer)| integer.constant(pattern, pointer))
            .map_or_else(|| Value::partial_integer(cell, stack), Value::Const)
    }
}

/// The registers a selected model leaves its result in, which its call's
/// keyed outputs read ([`SsaOp::CallClobber`](crate::ir::ops::SsaOp::CallClobber)).
///
/// Every other output of the call is unknown after it. An output receives
/// the result only under an identical representation, or the placement it
/// states.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MachineEffects {
    /// Registers receiving the model's result.
    pub outputs: Vec<MachineOutput>,
}

/// One value whose complete unchanged representation is established.
struct Entry<T: Target> {
    /// Actual value computed by ordinary SSA execution.
    value: Value<T>,
    /// Its supported fixed representation, independent of nominal type names.
    representation: ReturnBinding,
    /// Layout retained only for the immutable model-boundary description.
    layout: TypeLayout,
}

impl<T: Target> Clone for Entry<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            representation: self.representation,
            layout: self.layout,
        }
    }
}

impl<T: Target> PartialEq for Entry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.representation == other.representation && self.value == other.value
    }
}

impl<T: Target> fmt::Debug for Entry<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Entry")
            .field("value", &self.value)
            .field("representation", &self.representation)
            .finish()
    }
}

/// A sparse physical bank shared until a path actually changes a value.
///
/// Absence means unknown. An untouched location survives a source transfer or
/// an executed callee, including when an intermediate body never reads it.
#[derive(Debug)]
pub struct Bank<T: Target> {
    /// Source-established values, in deterministic key order.
    values: Rc<BTreeMap<CallInputKey, Entry<T>>>,
}

impl<T: Target> Clone for Bank<T> {
    fn clone(&self) -> Self {
        Self {
            values: Rc::clone(&self.values),
        }
    }
}

impl<T: Target> PartialEq for Bank<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<T: Target> Default for Bank<T> {
    fn default() -> Self {
        Self {
            values: Rc::new(BTreeMap::new()),
        }
    }
}

impl<T: Target> Bank<T> {
    /// Bounds inline entry storage, including layout facts and tree-node overhead.
    pub fn entry_bytes() -> usize {
        mem::size_of::<Entry<T>>()
            .saturating_add(mem::size_of::<CallInputKey>())
            .saturating_add(mem::size_of::<usize>().saturating_mul(4))
    }

    /// Returns the number of retained physical locations.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether this physical location has an established value.
    pub fn contains(&self, key: CallInputKey) -> bool {
        self.values.contains_key(&key)
    }

    /// Publishes an actual SSA value without computing, widening or reinterpreting it.
    pub fn publish(
        &mut self,
        key: CallInputKey,
        value: Value<T>,
        layout: TypeLayout,
        pointer: PointerSize,
    ) {
        let representation = ReturnBinding::for_layout(layout, pointer);
        let value = representation
            .apply(Some(value), pointer)
            .unwrap_or(Value::Unknown);
        if representation == ReturnBinding::Unknown || value == Value::Unknown {
            if self.values.contains_key(&key) {
                Rc::make_mut(&mut self.values).remove(&key);
            }
            return;
        }
        let entry = Entry {
            value,
            representation,
            layout,
        };
        if self.values.get(&key) != Some(&entry) {
            Rc::make_mut(&mut self.values).insert(key, entry);
        }
    }

    /// Returns a value only when its complete representation matches the requested read.
    ///
    /// The read is of the same bits in the reader's own type: an integer
    /// constant one namespace published at a width is the reader's integer of
    /// that width, with the sign its layout declares, not the publisher's.
    pub fn get(
        &self,
        key: CallInputKey,
        layout: TypeLayout,
        pointer: PointerSize,
    ) -> Option<Value<T>> {
        let entry = self.values.get(&key)?;
        if entry.representation != ReturnBinding::for_layout(layout, pointer) {
            return None;
        }
        let restated = match &entry.value {
            Value::Const(constant) => constant
                .conversion_integer_pattern(pointer)
                .zip(layout.integer_conversion_type())
                .and_then(|((pattern, _), integer)| integer.constant(pattern, pointer)),
            _ => None,
        };
        Some(restated.map_or_else(|| entry.value.clone(), Value::Const))
    }

    /// Retains only facts established on both paths, joining their actual values.
    pub fn join(&self, other: &Self) -> Self {
        if Rc::ptr_eq(&self.values, &other.values) {
            return self.clone();
        }
        let values = self
            .values
            .iter()
            .filter_map(|(key, left)| {
                let right = other.values.get(key)?;
                if left.representation != right.representation {
                    return None;
                }
                let value = left.value.join(&right.value);
                (value != Value::Unknown).then_some((
                    *key,
                    Entry {
                        value,
                        representation: left.representation,
                        layout: left.layout,
                    },
                ))
            })
            .collect();
        Self {
            values: Rc::new(values),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        analysis::execution::{
            bank::{Bank, MachineOutput, Placement},
            value::Value,
        },
        ir::{CallInputKey, ConstValue, integer::IntegerBits},
        target::Target,
        testing::{MACHINE, MockTarget, MockType},
    };

    /// The cell value a 64-bit location holds after a call returning `result`.
    fn placed(placement: Placement, result: ConstValue<MockTarget>) -> Value<MockTarget> {
        let layout = MockTarget::type_layout(&MockType::U64, MACHINE.ptr_size).unwrap();
        MachineOutput {
            key: CallInputKey::new(0),
            layout,
            placement,
        }
        .value(Value::Const(result), MACHINE.ptr_size)
    }

    /// A narrow result fills the low bits; the rest is unknown or extended per the contract.
    #[test]
    fn narrow_results_fill_low_bits_and_extend_only_where_the_contract_says() {
        let low = Placement::Low { bits: 32 };
        let low_mask = (1u128 << 32) - 1;
        assert_eq!(
            placed(low, ConstValue::I32(-1)).integer_bits(),
            IntegerBits::new(64, 0, low_mask)
        );
        assert_eq!(
            placed(low, ConstValue::U32(42)).integer_bits(),
            IntegerBits::new(64, low_mask & !42, 42)
        );
        for (signed, expected) in [(true, u64::MAX), (false, 0xffff_ffff)] {
            let extended = Placement::Extended { bits: 32, signed };
            assert_eq!(
                placed(extended, ConstValue::I32(-1)),
                Value::Const(ConstValue::U64(expected))
            );
        }
        assert_eq!(
            placed(Placement::Whole, ConstValue::U64(7)),
            Value::Const(ConstValue::U64(7))
        );
    }

    /// A read returns the published bits in the reader's integer type, and
    /// nothing under another width.
    #[test]
    fn reads_restate_published_bits_in_the_readers_type() {
        let layout = |ty| MockTarget::type_layout(&ty, MACHINE.ptr_size).unwrap();
        let key = CallInputKey::new(0);
        let mut bank = Bank::<MockTarget>::default();
        bank.publish(
            key,
            Value::Const(ConstValue::I64(-1)),
            layout(MockType::I64),
            MACHINE.ptr_size,
        );
        assert_eq!(
            bank.get(key, layout(MockType::U64), MACHINE.ptr_size),
            Some(Value::Const(ConstValue::U64(u64::MAX)))
        );
        assert_eq!(
            bank.get(key, layout(MockType::I64), MACHINE.ptr_size),
            Some(Value::Const(ConstValue::I64(-1)))
        );
        assert_eq!(bank.get(key, layout(MockType::U32), MACHINE.ptr_size), None);
    }

    /// A result of another width, or a placement as wide as the cell, establishes nothing.
    #[test]
    fn mismatched_widths_leave_the_location_unknown() {
        assert_eq!(
            placed(Placement::Low { bits: 32 }, ConstValue::U64(7)),
            Value::Unknown
        );
        assert_eq!(
            placed(Placement::Low { bits: 64 }, ConstValue::U64(7)),
            Value::Unknown
        );
        assert_eq!(
            placed(
                Placement::Extended {
                    bits: 16,
                    signed: true
                },
                ConstValue::I32(-1)
            ),
            Value::Unknown
        );
    }
}
