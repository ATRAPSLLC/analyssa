//! A host's opaque metadata.
//!
//! Types, methods, fields and signatures are a host's own values, which
//! analyssa compares but never interprets. [`MetadataRef`] borrows one without
//! cloning it.

use crate::target::Target;

/// Borrowed host metadata whose representation is owned by a target.
///
/// The variants retain type safety without cloning payloads. Two are equal
/// when they are the same kind of metadata and the host's values are equal.
#[non_exhaustive]
pub enum MetadataRef<'a, T: Target> {
    /// A resolved value type.
    Type(&'a T::Type),
    /// A type identifier.
    TypeRef(&'a T::TypeRef),
    /// A method identifier.
    Method(&'a T::MethodRef),
    /// A field identifier.
    Field(&'a T::FieldRef),
    /// A signature identifier.
    Signature(&'a T::SigRef),
    /// A located or named symbol.
    Symbol(&'a T::SymbolRef),
}

impl<T: Target> Copy for MetadataRef<'_, T> {}

impl<T: Target> Clone for MetadataRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Target> PartialEq for MetadataRef<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Type(left), Self::Type(right)) => left == right,
            (Self::TypeRef(left), Self::TypeRef(right)) => left == right,
            (Self::Method(left), Self::Method(right)) => left == right,
            (Self::Field(left), Self::Field(right)) => left == right,
            (Self::Signature(left), Self::Signature(right)) => left == right,
            (Self::Symbol(left), Self::Symbol(right)) => left == right,
            _ => false,
        }
    }
}

impl<T: Target> Eq for MetadataRef<'_, T> {}

/// Test targets whose host metadata counts every comparison, clone and hash.
#[cfg(test)]
pub mod tests {
    use std::{
        cell::Cell,
        hash::{Hash, Hasher},
        rc::Rc,
    };

    use super::*;
    use crate::{ir::exception::HandlerKind, testing::MockTarget};

    /// Counts unrestricted equality and cloning of host metadata.
    #[derive(Debug)]
    pub struct EqualityProbe(Rc<Cell<usize>>);

    impl EqualityProbe {
        /// Shares a counter for unrestricted clone, equality and hash calls.
        pub fn new(counter: Rc<Cell<usize>>) -> Self {
            Self(counter)
        }
    }

    impl Clone for EqualityProbe {
        fn clone(&self) -> Self {
            self.0.set(self.0.get().saturating_add(1));
            Self(Rc::clone(&self.0))
        }
    }

    impl PartialEq for EqualityProbe {
        fn eq(&self, _other: &Self) -> bool {
            self.0.set(self.0.get().saturating_add(1));
            true
        }
    }

    impl Eq for EqualityProbe {}

    impl Hash for EqualityProbe {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.0.set(self.0.get().saturating_add(1));
            state.write_u8(0);
        }
    }

    /// Exercises the conservative default without supplying a metadata hook.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct DefaultTarget;

    impl Target for DefaultTarget {
        type TypeRef = EqualityProbe;
        type MethodRef = EqualityProbe;
        type FieldRef = EqualityProbe;
        type SigRef = EqualityProbe;
        type SymbolRef = EqualityProbe;
        type ExceptionKind = ();
        type Type = EqualityProbe;
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
        fn unknown_type() -> EqualityProbe {
            EqualityProbe(Rc::new(Cell::new(0)))
        }
        fn is_integer(_ty: &EqualityProbe) -> bool {
            false
        }
        fn is_floating(_ty: &EqualityProbe) -> bool {
            false
        }
        fn is_signed(_ty: &EqualityProbe) -> bool {
            false
        }
        fn is_pointer(_ty: &EqualityProbe) -> bool {
            false
        }
        fn is_reference(_ty: &EqualityProbe) -> bool {
            false
        }
        fn is_unknown(_ty: &EqualityProbe) -> bool {
            true
        }
        fn bit_width(_ty: &EqualityProbe) -> Option<u32> {
            None
        }
        fn instruction_mnemonic(_instruction: &()) -> &'static str {
            ""
        }
        fn instruction_rva(_instruction: &()) -> u64 {
            0
        }
        fn handler_kind(_kind: &()) -> HandlerKind {
            HandlerKind::Catch
        }
    }

    /// Equal payloads remain distinct when their metadata domains differ.
    #[test]
    fn borrowed_metadata_equality_preserves_domains() {
        let value = 7;
        let other = 8;
        let reference = MetadataRef::<MockTarget>::TypeRef(&value);
        assert!(reference == MetadataRef::TypeRef(&value));
        assert!(reference != MetadataRef::TypeRef(&other));
        assert!(reference != MetadataRef::Method(&value));
        assert!(reference != MetadataRef::Field(&value));
        assert!(reference != MetadataRef::Signature(&value));
        assert!(reference != MetadataRef::Symbol(&value));
    }
}
