//! Concrete operands the symbolic driver may fold.
//!
//! A known operand enters a symbolic expression only as a bounded numeric
//! carrier; opaque host metadata is never copied into the expression.

use crate::{ir::value::ConstValue, target::Target};

/// Copies only bounded numeric carriers, without visiting opaque host metadata.
pub fn operand<T: Target>(value: &ConstValue<T>) -> Option<ConstValue<T>> {
    match value {
        ConstValue::I8(_)
        | ConstValue::I16(_)
        | ConstValue::I32(_)
        | ConstValue::I64(_)
        | ConstValue::U8(_)
        | ConstValue::U16(_)
        | ConstValue::U32(_)
        | ConstValue::U64(_)
        | ConstValue::NativeInt(_)
        | ConstValue::NativeUInt(_)
        | ConstValue::Float(_)
        | ConstValue::True
        | ConstValue::False
        | ConstValue::Null => Some(value.clone()),
        ConstValue::Vector(value) if value.bits() <= 128 => Some(ConstValue::Vector(value.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use crate::{
        analysis::symbolic::concrete::operand,
        ir::value::ConstValue,
        metadata::tests::{DefaultTarget, EqualityProbe},
    };

    /// Numeric admission declines opaque identities without invoking raw host traits.
    #[test]
    fn concrete_numeric_admission_does_not_copy_host_metadata() {
        let calls = Rc::new(Cell::new(0));
        for value in [
            ConstValue::<DefaultTarget>::Type(EqualityProbe::new(Rc::clone(&calls))),
            ConstValue::MethodHandle(EqualityProbe::new(Rc::clone(&calls))),
            ConstValue::FieldHandle(EqualityProbe::new(Rc::clone(&calls))),
            ConstValue::Symbol(EqualityProbe::new(Rc::clone(&calls))),
        ] {
            assert!(operand(&value).is_none());
        }
        assert_eq!(calls.get(), 0);
    }
}
