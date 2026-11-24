//! Flat semantic type facts projected from borrowed host metadata.

use std::num::{NonZeroU8, NonZeroUsize};

use crate::{
    PointerSize,
    encoder::{ContentWriter, Emit},
    ir::{content::ContentSink, float::FloatFormat, integer::IntegerType},
    target::{Target, VectorDescriptor, VectorMaskDescriptor, VectorShape},
};

mod content;

/// An attested Boolean representation using unsigned zero and one.
///
/// The representation occupies one through sixteen whole bytes in target byte
/// order. Only the exact zero and one patterns decode to logical values;
/// other patterns require a different host representation and are refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BooleanStorage {
    /// Nonzero byte extent, bounded by a scalar integer pattern.
    bytes: NonZeroU8,
}

impl BooleanStorage {
    /// Canonical one-byte Boolean storage.
    pub const BYTE: Self = Self {
        bytes: NonZeroU8::MIN,
    };

    /// Creates a canonical Boolean representation of one through sixteen bytes.
    ///
    /// Returns `None` for a zero extent or one wider than the scalar codec.
    #[must_use]
    pub const fn new(bytes: u8) -> Option<Self> {
        if bytes > 16 {
            return None;
        }
        match NonZeroU8::new(bytes) {
            Some(bytes) => Some(Self { bytes }),
            None => None,
        }
    }

    /// Returns the attested storage extent in bytes.
    #[must_use]
    pub const fn bytes(self) -> u8 {
        self.bytes.get()
    }
}

/// The extent and byte interpretation a memory writer needs.
#[derive(Debug, Clone, Copy)]
pub struct StorageLayout {
    /// Exact nonzero extent in addressable bytes.
    bytes: NonZeroUsize,
    /// Canonical Boolean storage, absent for an ordinary bit pattern.
    boolean: Option<BooleanStorage>,
}

impl StorageLayout {
    /// Returns the exact byte extent.
    pub const fn bytes(self) -> usize {
        self.bytes.get()
    }

    /// Returns the attested Boolean interpretation, where present.
    pub const fn boolean(self) -> Option<BooleanStorage> {
        self.boolean
    }
}

/// The fixed-size semantic facts needed to interpret one host type.
///
/// This value retains no names, handles or references into a type graph.
/// Pointer and Boolean classification remain distinct from integer widths;
/// a floating format is explicit even when two formats share one width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeLayout {
    /// Declared value width, where known.
    bits: Option<u32>,
    /// Whether the type is an integer.
    integer: bool,
    /// Whether the type is a scalar float.
    floating: bool,
    /// Whether it stores a Boolean value.
    boolean: bool,
    /// Canonical storage, present only when the host attests it.
    boolean_storage: Option<BooleanStorage>,
    /// Whether an integer is signed.
    signed: bool,
    /// Whether it is a pointer or byref.
    pointer: bool,
    /// Whether it is an object/reference type.
    reference: bool,
    /// Whether the type is intentionally unspecified.
    unknown: bool,
    /// Exact scalar floating format.
    float: Option<FloatFormat>,
    /// Fixed or runtime-scalable vector geometry.
    vector: Option<VectorDescriptor>,
    /// Fixed or runtime-scalable predicate geometry.
    mask: Option<VectorMaskDescriptor>,
    /// Scalar integer conversion range and result representation.
    conversion: Option<IntegerType>,
}

impl TypeLayout {
    /// Writes every semantic layout field through a canonical fallible visitor.
    ///
    /// Layout format version one uses little-endian primitive fields, explicit
    /// option and variant tags, and the complete floating and vector descriptors.
    /// No host metadata query, allocation or temporary serialization is needed.
    /// The caller supplies the surrounding domain when composing this stream.
    ///
    /// # Errors
    ///
    /// Returns the first sink refusal before visiting subsequent fields. The
    /// sink retains any consumed prefix and owns publication or rollback.
    pub fn write_content<T: Target, S: ContentSink<T> + ?Sized>(
        &self,
        sink: &mut S,
    ) -> Result<(), S::Error> {
        let mut writer = ContentWriter::new(sink);
        1u16.emit(&mut writer)?;
        self.emit(&mut writer)
    }

    /// Creates a scalar layout from an operation's validated integer contract.
    ///
    /// This performs no host metadata query. The caller supplies the width and
    /// signedness established by the operation, rather than inferring them
    /// from a value carrier or a nominal parameter declaration.
    #[must_use]
    pub const fn integer(layout: IntegerType) -> Self {
        Self {
            bits: Some(layout.bits()),
            integer: true,
            floating: false,
            boolean: false,
            boolean_storage: None,
            signed: layout.signed(),
            pointer: false,
            reference: false,
            unknown: false,
            float: None,
            vector: None,
            mask: None,
            conversion: Some(layout),
        }
    }

    /// Projects a type through the target's ordinary inspection methods: the
    /// layout a host whose types need no alias resolution states from
    /// [`Target::type_layout`].
    #[must_use]
    pub fn of<T: Target>(ty: &T::Type, ptr_size: PointerSize) -> Self {
        let boolean = T::is_boolean(ty);
        Self {
            bits: T::bit_width(ty),
            integer: T::is_integer(ty),
            floating: T::is_floating(ty),
            boolean,
            boolean_storage: if boolean {
                T::boolean_storage(ty)
            } else {
                None
            },
            signed: T::is_signed(ty),
            pointer: T::is_pointer(ty),
            reference: T::is_reference(ty),
            unknown: T::is_unknown(ty),
            float: T::float_format(ty),
            vector: T::vector_descriptor(ty),
            mask: T::vector_mask_descriptor(ty),
            conversion: T::integer_conversion_type(ty, ptr_size),
        }
    }

    /// Returns the declared value width without guessing aggregate widths.
    ///
    /// A runtime-sized value needs its runtime extent even when the host also
    /// supplies a minimum width; consult [`Self::is_runtime_sized`].
    #[must_use]
    pub const fn bit_width(self) -> Option<u32> {
        self.bits
    }

    /// Returns whether this is an integer type.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        self.integer
    }

    /// Returns whether this is a scalar floating type.
    #[must_use]
    pub const fn is_floating(self) -> bool {
        self.floating
    }

    /// Returns whether this stores Boolean zero or one.
    #[must_use]
    pub const fn is_boolean(self) -> bool {
        self.boolean
    }

    /// Returns the host-attested canonical Boolean storage, where stated.
    #[must_use]
    pub const fn boolean_storage(self) -> Option<BooleanStorage> {
        self.boolean_storage
    }

    /// Returns the fixed storage extent without rounding logical value widths.
    ///
    /// Boolean storage requires an explicit host declaration. Other values
    /// require a nonzero byte-aligned width; pointers without a declared width
    /// use `ptr_size`. Runtime-scalable vectors have no fixed storage extent.
    #[must_use]
    #[inline]
    pub fn storage_bytes(self, ptr_size: PointerSize) -> Option<usize> {
        self.storage_layout(ptr_size).map(StorageLayout::bytes)
    }

    /// Projects one compact writer view without re-inspecting host metadata.
    #[inline]
    pub fn storage_layout(self, ptr_size: PointerSize) -> Option<StorageLayout> {
        if self.is_runtime_sized() {
            return None;
        }
        if self.boolean {
            let boolean = self.boolean_storage?;
            return Some(StorageLayout {
                bytes: NonZeroUsize::new(usize::from(boolean.bytes()))?,
                boolean: Some(boolean),
            });
        }
        let bits = self
            .bits
            .or_else(|| self.pointer.then_some(ptr_size.bits()))?;
        if bits == 0 || !bits.is_multiple_of(8) {
            return None;
        }
        Some(StorageLayout {
            bytes: NonZeroUsize::new(usize::try_from(bits / 8).ok()?)?,
            boolean: None,
        })
    }

    /// Returns whether an integer is signed.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        self.signed
    }

    /// Returns whether this is a pointer or byref.
    #[must_use]
    pub const fn is_pointer(self) -> bool {
        self.pointer
    }

    /// Returns whether this is an object/reference type.
    #[must_use]
    pub const fn is_reference(self) -> bool {
        self.reference
    }

    /// Returns whether the type is intentionally unspecified.
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        self.unknown
    }

    /// Returns the exact scalar floating format, where stated.
    #[must_use]
    pub const fn float_format(self) -> Option<FloatFormat> {
        self.float
    }

    /// Returns fixed vector geometry without inventing a scalable extent.
    #[must_use]
    pub const fn vector_shape(self) -> Option<VectorShape> {
        match self.vector {
            Some(VectorDescriptor::Fixed(shape)) => Some(shape),
            _ => None,
        }
    }

    /// Returns vector geometry without losing a runtime scale.
    #[must_use]
    pub const fn vector_descriptor(self) -> Option<VectorDescriptor> {
        self.vector
    }

    /// Returns predicate geometry without inventing a fixed lane count.
    #[must_use]
    pub const fn vector_mask_descriptor(self) -> Option<VectorMaskDescriptor> {
        self.mask
    }

    /// Returns whether a runtime vector extent is required to materialize the type.
    #[must_use]
    pub const fn is_runtime_sized(self) -> bool {
        matches!(self.vector, Some(VectorDescriptor::Scalable(_)))
            || matches!(self.mask, Some(VectorMaskDescriptor::Scalable(_)))
    }

    /// Returns the scalar conversion range and constant representation.
    #[must_use]
    pub const fn integer_conversion_type(self) -> Option<IntegerType> {
        self.conversion
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        PointerSize,
        ir::{exception::HandlerKind, float::FloatFormat},
        target::{Target, TypeLayout},
        testing::{MockTarget, MockType},
    };

    /// A host whose ordinary metadata getters must never be entered implicitly.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct UnboundedTypeHost;

    impl Target for UnboundedTypeHost {
        type TypeRef = u32;
        type MethodRef = u32;
        type FieldRef = u32;
        type SigRef = u32;
        type SymbolRef = u32;
        type ExceptionKind = ();
        type Type = MockType;
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
        fn unknown_type() -> Self::Type {
            MockType::Unknown
        }
        fn is_integer(_: &Self::Type) -> bool {
            panic!("unrestricted integer query")
        }
        fn is_floating(_: &Self::Type) -> bool {
            panic!("unrestricted floating query")
        }
        fn is_signed(_: &Self::Type) -> bool {
            panic!("unrestricted sign query")
        }
        fn is_pointer(_: &Self::Type) -> bool {
            panic!("unrestricted pointer query")
        }
        fn is_reference(_: &Self::Type) -> bool {
            panic!("unrestricted reference query")
        }
        fn is_unknown(_: &Self::Type) -> bool {
            panic!("unrestricted unknown query")
        }
        fn bit_width(_: &Self::Type) -> Option<u32> {
            panic!("unrestricted width query")
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

    /// A host that states no layout states none: the default never falls back
    /// to the ordinary type getters.
    #[test]
    fn unsupported_type_projection_has_no_unrestricted_fallback() {
        assert_eq!(
            UnboundedTypeHost::type_layout(&MockType::I8, PointerSize::Bit64),
            None
        );
    }

    /// Flat layouts retain integer ranges, exact formats, vectors and pointer semantics.
    #[test]
    fn flat_type_projection_preserves_independent_semantic_facts() {
        let byte = TypeLayout::of::<MockTarget>(&MockType::U8, PointerSize::Bit64);
        assert_eq!(byte.bit_width(), Some(8));
        assert!(byte.is_integer() && !byte.is_signed() && !byte.is_boolean());
        assert_eq!(
            byte.integer_conversion_type()
                .map(|layout| (layout.bits(), layout.signed())),
            Some((8, false))
        );
        for (ty, bits, format) in [
            (MockType::F64, 64, FloatFormat::Double),
            (MockType::BF16, 16, FloatFormat::BFloat16),
            (MockType::F80, 80, FloatFormat::X87Extended),
        ] {
            let layout = TypeLayout::of::<MockTarget>(&ty, PointerSize::Bit64);
            assert!(layout.is_floating() && !layout.is_integer());
            assert_eq!(layout.bit_width(), Some(bits));
            assert_eq!(layout.float_format(), Some(format));
        }
        let pointer = TypeLayout::of::<MockTarget>(&MockType::Ptr, PointerSize::Bit32);
        assert!(pointer.is_pointer() && !pointer.is_reference());
        assert_eq!(
            pointer
                .integer_conversion_type()
                .map(|layout| layout.bits()),
            Some(32)
        );
        let vector = TypeLayout::of::<MockTarget>(&MockType::V4I32, PointerSize::Bit64);
        assert_eq!(
            vector.vector_shape().map(|shape| (
                shape.lane_count,
                shape.lane_bits,
                shape.total_bits
            )),
            Some((4, 32, 128))
        );
        assert!(TypeLayout::of::<MockTarget>(&MockType::Unknown, PointerSize::Bit64).is_unknown());
        for ty in [MockType::NxV4I32, MockType::NxV8I32, MockType::NxMask4] {
            let layout = TypeLayout::of::<MockTarget>(&ty, PointerSize::Bit64);
            assert!(layout.is_runtime_sized());
            assert_eq!(layout.vector_shape(), None);
        }
        let mask = TypeLayout::of::<MockTarget>(&MockType::Mask4, PointerSize::Bit64);
        assert!(!mask.is_runtime_sized());
        assert_eq!(
            mask.vector_mask_descriptor()
                .and_then(|shape| shape.fixed_lane_count()),
            Some(4)
        );
    }
}
