//! Constants as bytes: their encoding at a byte order and layout, and their
//! decoding back.

use std::ops::Range;

use crate::{
    PointerSize,
    ir::{
        float::FloatFormat,
        lanes::{MAX_VECTOR_BITS, VectorConst},
        value::ConstValue,
    },
    target::{
        BooleanStorage, Endianness, StorageLayout, Target, TypeLayout, VectorLaneKind, VectorShape,
    },
};

/// A scalar byte interpretation after vector and floating formats are resolved.
pub enum ScalarByteLayout {
    /// An explicitly attested canonical Boolean representation.
    Boolean(BooleanStorage),
    /// A byte-aligned scalar integer or numeric pointer.
    Integer {
        /// Declared value width.
        bits: u32,
        /// Signed interpretation of a primitive carrier.
        signed: bool,
    },
}

/// An exact byte iterator with inline scalars and borrowed vector storage.
pub struct EncodedBytes<'a> {
    /// The source bit pattern and its byte order.
    source: ByteSource<'a>,
    /// The unread byte indices within the source.
    indices: Range<usize>,
}

/// The bits an encoding reads without cloning its constant.
pub enum ByteSource<'a> {
    /// A scalar bit pattern in target byte order.
    Scalar([u8; 16]),
    /// Vector storage, with byte order applied separately to each lane.
    Vector {
        /// The unchanged source storage.
        vector: &'a VectorConst,
        /// A lane's byte width; sub-byte lanes pack into single bytes.
        lane_bytes: usize,
        /// The byte order inside each lane.
        order: Endianness,
    },
}

impl<'a> EncodedBytes<'a> {
    /// Returns an exact byte encoding without allocating or scanning vector storage.
    ///
    /// Lane zero starts at the lowest address. Byte order reverses bytes within
    /// a lane, including lanes wider than 128 bits. Sub-byte lanes pack into
    /// bytes without changing bit order, and require a whole-byte total width.
    pub fn of<T: Target>(
        value: &'a ConstValue<T>,
        order: Endianness,
        ptr_size: PointerSize,
    ) -> Option<Self> {
        if let ConstValue::Vector(vector) = value {
            let lane_bits = vector.lane_bits();
            if !vector.bits().is_multiple_of(8)
                || lane_bits == 0
                || lane_bits >= 8 && !lane_bits.is_multiple_of(8)
            {
                return None;
            }
            return Some(Self {
                source: ByteSource::Vector {
                    vector,
                    lane_bytes: usize::try_from(lane_bits.max(8) / 8).ok()?,
                    order,
                },
                indices: 0..usize::try_from(vector.bits() / 8).ok()?,
            });
        }
        let (bits, width) = match *value {
            ConstValue::I8(value) => (u128::from(value as u8), 8),
            ConstValue::U8(value) => (u128::from(value), 8),
            ConstValue::I16(value) => (u128::from(value as u16), 16),
            ConstValue::U16(value) => (u128::from(value), 16),
            ConstValue::I32(value) => (u128::from(value as u32), 32),
            ConstValue::U32(value) => (u128::from(value), 32),
            ConstValue::I64(value) => (u128::from(value.cast_unsigned()), 64),
            ConstValue::U64(value) => (u128::from(value), 64),
            ConstValue::NativeInt(_) | ConstValue::NativeUInt(_) => {
                value.scalar_integer_pattern(ptr_size)?
            }
            ConstValue::Float(value) => (value.bits(), value.format().bits()),
            _ => return None,
        };
        if width == 0 || width > 128 || !width.is_multiple_of(8) {
            return None;
        }
        let width = usize::try_from(width / 8).ok()?;
        let indices = match order {
            Endianness::Little => 0..width,
            Endianness::Big => 16usize.checked_sub(width)?..16,
        };
        Some(Self {
            source: ByteSource::Scalar(order.bytes_of_u128(bits)),
            indices,
        })
    }

    /// Encodes exactly the storage layout's extent of `value`, with one
    /// carrier-length check.
    pub fn at_layout<T: Target>(
        value: &'a ConstValue<T>,
        storage: StorageLayout,
        order: Endianness,
        ptr_size: PointerSize,
    ) -> Option<Self> {
        let width = storage.bytes();
        if storage.boolean().is_some() {
            let bits = u128::from(value.logical_boolean(ptr_size)?);
            let indices = match order {
                Endianness::Little => 0..width,
                Endianness::Big => 16usize.checked_sub(width)?..16,
            };
            return Some(Self {
                source: ByteSource::Scalar(order.bytes_of_u128(bits)),
                indices,
            });
        }
        let bytes = Self::of(value, order, ptr_size)?;
        (bytes.len() == width).then_some(bytes)
    }
}

impl Iterator for EncodedBytes<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.indices.next()?;
        match self.source {
            ByteSource::Scalar(bytes) => bytes.get(index).copied(),
            ByteSource::Vector {
                vector,
                lane_bytes,
                order,
            } => {
                let byte = index.checked_rem(lane_bytes)?;
                let at = match order {
                    Endianness::Little => index,
                    Endianness::Big => index
                        .checked_sub(byte)?
                        .checked_add(lane_bytes.checked_sub(1)?.checked_sub(byte)?)?,
                };
                let bit = u32::try_from(at.checked_mul(8)?).ok()?;
                u8::try_from(vector.bits_at(bit, 8)).ok()
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.indices.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for EncodedBytes<'_> {}

impl<T: Target> ConstValue<T> {
    /// Serializes an integer or float constant to bytes in the given endianness.
    ///
    /// A vector is its lanes in order, each in the given byte order - lane 0 at
    /// the lowest address, as a vector register stores. A float is its IEEE 754
    /// bit pattern. Returns `None` for the other variants (strings, handles,
    /// etc.) and for a vector whose layout cannot encode whole bytes. Packed
    /// sub-byte lanes keep their bit order; wider lanes use their full width.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::{ir::value::ConstValue, target::Endianness, MockTarget, PointerSize};
    ///
    /// let val = ConstValue::<MockTarget>::U32(0x01020304);
    /// let le = val.to_bytes(Endianness::Little, PointerSize::Bit64).unwrap();
    /// let be = val.to_bytes(Endianness::Big, PointerSize::Bit64).unwrap();
    /// assert_eq!(le, vec![0x04, 0x03, 0x02, 0x01]);
    /// assert_eq!(be, vec![0x01, 0x02, 0x03, 0x04]);
    ///
    /// // A native-width value follows the target's pointer size, not `u64`.
    /// let native = ConstValue::<MockTarget>::NativeUInt(0x0102_0304);
    /// assert_eq!(native.to_bytes(Endianness::Little, PointerSize::Bit32).unwrap().len(), 4);
    /// assert_eq!(native.to_bytes(Endianness::Little, PointerSize::Bit64).unwrap().len(), 8);
    /// ```
    #[must_use]
    pub fn to_bytes(&self, endianness: Endianness, ptr_size: PointerSize) -> Option<Vec<u8>> {
        EncodedBytes::of(self, endianness, ptr_size).map(Iterator::collect)
    }

    /// Encodes this value using a host-attested storage representation.
    ///
    /// Boolean values require host-attested canonical storage. Runtime-sized
    /// layouts, unsupported carriers and mismatched extents return `None`.
    /// Logical Boolean value bits do not determine their memory byte extent.
    #[must_use]
    pub fn to_layout_bytes(
        &self,
        layout: TypeLayout,
        endianness: Endianness,
        ptr_size: PointerSize,
    ) -> Option<Vec<u8>> {
        EncodedBytes::at_layout(self, layout.storage_layout(ptr_size)?, endianness, ptr_size)
            .map(Iterator::collect)
    }

    /// Encodes an already projected compact writer view without repeating the projection.
    pub fn encoded_storage_bytes(
        &self,
        storage: StorageLayout,
        order: Endianness,
        ptr_size: PointerSize,
    ) -> Option<impl ExactSizeIterator<Item = u8> + '_> {
        EncodedBytes::at_layout(self, storage, order, ptr_size)
    }

    /// Deserializes a byte slice into an unsigned integer constant.
    ///
    /// The byte slice is interpreted as an unsigned integer in the given
    /// endianness. The variant is chosen based on the slice length:
    ///
    /// | Length | Variant |
    /// |--------|---------|
    /// | 1      | `U8`    |
    /// | 2      | `U16`   |
    /// | 4      | `U32`   |
    /// | 8      | `U64`   |
    ///
    /// Returns `None` for other slice lengths.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::{ir::value::ConstValue, target::Endianness, MockTarget};
    ///
    /// let be_bytes = vec![0x01, 0x02, 0x03, 0x04];
    /// let val = ConstValue::<MockTarget>::from_bytes(&be_bytes, Endianness::Big);
    /// assert_eq!(val, Some(ConstValue::U32(0x01020304)));
    /// ```
    #[must_use]
    pub fn from_bytes(bytes: &[u8], endianness: Endianness) -> Option<Self> {
        match bytes.len() {
            1 => Some(Self::U8(bytes.first().copied()?)),
            2 => Some(Self::U16(endianness.read_u16(bytes))),
            4 => Some(Self::U32(endianness.read_u32(bytes))),
            8 => Some(Self::U64(endianness.read_u64(bytes))),
            _ => None,
        }
    }

    /// Reads the constant of type `ty` that `bytes` encode.
    ///
    /// An integer takes its width and signedness from the type; a pointer is
    /// the unsigned integer of its width - the bare number the program reads,
    /// which is an address in the program's space and nothing more; a float is
    /// the IEEE single or double its width names; a vector is its lanes in
    /// order, each lane its bits. Any other type, or a width the bytes do not
    /// match, states nothing. Fixed vectors must have consistent geometry and
    /// fit within [`crate::ir::lanes::MAX_VECTOR_BITS`]. A wider-than-128-bit
    /// lane retains every bit of its representation.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The encoding, exactly as wide as the type.
    /// * `ty` - The type the bytes are read as.
    /// * `endianness` - The byte order they are stored in.
    #[must_use]
    pub fn from_typed_bytes(bytes: &[u8], ty: &T::Type, endianness: Endianness) -> Option<Self> {
        Self::decode_typed_bytes(
            bytes,
            endianness,
            T::vector_shape(ty),
            || T::float_format(ty),
            || {
                if T::is_boolean(ty) {
                    return Some(ScalarByteLayout::Boolean(T::boolean_storage(ty)?));
                }
                let bits = T::bit_width(ty)?;
                let signed = if T::is_integer(ty) {
                    T::is_signed(ty)
                } else if T::is_pointer(ty) {
                    false
                } else {
                    return None;
                };
                Some(ScalarByteLayout::Integer { bits, signed })
            },
        )
    }

    /// Reads bytes using the layout's semantic facts without inspecting host metadata.
    ///
    /// The bytes must exactly match the scalar or fixed vector extent. A
    /// runtime-scalable layout has no fixed extent and is refused. Fixed-vector
    /// geometry must be consistent and fit within
    /// [`crate::ir::lanes::MAX_VECTOR_BITS`]. This method shares decoding with
    /// [`Self::from_typed_bytes`].
    #[must_use]
    pub fn from_layout_bytes(
        bytes: &[u8],
        layout: TypeLayout,
        endianness: Endianness,
    ) -> Option<Self> {
        if layout.is_runtime_sized() {
            return None;
        }
        Self::decode_typed_bytes(
            bytes,
            endianness,
            layout.vector_shape(),
            || layout.float_format(),
            || {
                if layout.is_boolean() {
                    return Some(ScalarByteLayout::Boolean(layout.boolean_storage()?));
                }
                let signed = if layout.is_integer() {
                    layout.is_signed()
                } else if layout.is_pointer() {
                    false
                } else {
                    return None;
                };
                Some(ScalarByteLayout::Integer {
                    bits: layout.bit_width()?,
                    signed,
                })
            },
        )
    }

    /// Decodes one type view, querying scalar facts only after vector and float refusal.
    ///
    /// Validates fixed-vector geometry and its representation bound before
    /// materializing lanes, so refusal cannot allocate a declared lane array.
    fn decode_typed_bytes(
        bytes: &[u8],
        endianness: Endianness,
        shape: Option<VectorShape>,
        float_format: impl FnOnce() -> Option<FloatFormat>,
        scalar: impl FnOnce() -> Option<ScalarByteLayout>,
    ) -> Option<Self> {
        if let Some(shape) = shape {
            let shape = VectorShape::new(
                shape.lane_count,
                shape.lane_kind,
                shape.lane_bits,
                shape.total_bits,
            )?;
            if shape.total_bits > MAX_VECTOR_BITS
                || shape.lane_kind == VectorLaneKind::NativeInteger
            {
                return None;
            }
            // Lanes narrower than a byte unpack from bytes as the register
            // holds them, the lowest lane in the low bits of the first.
            if shape.lane_bits < 8 {
                if u64::try_from(bytes.len()).ok()?.checked_mul(8)? != u64::from(shape.total_bits) {
                    return None;
                }
                return VectorConst::from_lanes(8, bytes.iter().map(|&byte| u128::from(byte)))?
                    .relaned(shape.lane_bits)
                    .map(Self::Vector);
            }
            if shape.lane_bits % 8 != 0 {
                return None;
            }
            let lane_bytes = usize::try_from(shape.lane_bits / 8).ok()?;
            if lane_bytes == 0
                || bytes.len() != lane_bytes.checked_mul(usize::try_from(shape.lane_count).ok()?)?
            {
                return None;
            }
            // A lane wider than any one integer, a whole wide register moved
            // as one (`vec[i512; 1]`), is its bytes in the lane's own order,
            // the lowest byte the lowest bits, laid out in lanes of its width.
            if lane_bytes > 16 {
                let mut ordered: Vec<u128> = Vec::with_capacity(bytes.len());
                for lane in bytes.chunks(lane_bytes) {
                    let lane = lane.iter().map(|&byte| u128::from(byte));
                    match endianness {
                        Endianness::Little => ordered.extend(lane),
                        Endianness::Big => ordered.extend(lane.rev()),
                    }
                }
                return VectorConst::from_lanes(8, ordered)?
                    .relaned(shape.lane_bits)
                    .map(Self::Vector);
            }
            let lanes = bytes
                .chunks(lane_bytes)
                .map(|lane| endianness.read_wide_uint(lane))
                .collect::<Option<Vec<_>>>()?;
            return VectorConst::from_lanes(shape.lane_bits, lanes).map(Self::Vector);
        }
        if let Some(format) = float_format() {
            return Self::float_from_bytes(bytes, format, endianness);
        }
        match scalar()? {
            ScalarByteLayout::Boolean(storage) => {
                if bytes.len() != usize::from(storage.bytes()) {
                    return None;
                }
                match endianness.read_wide_uint(bytes)? {
                    0 => Some(Self::False),
                    1 => Some(Self::True),
                    _ => None,
                }
            }
            ScalarByteLayout::Integer { bits, signed } => {
                Self::scalar_from_bytes(bytes, bits, signed, endianness)
            }
        }
    }

    /// The float of `format` that `bytes` encode, when they are exactly as
    /// wide as the format.
    fn float_from_bytes(bytes: &[u8], format: FloatFormat, endianness: Endianness) -> Option<Self> {
        let width = usize::try_from(format.bits() / 8).ok()?;
        if bytes.len() != width {
            return None;
        }
        let mut padded = [0u8; 16];
        match endianness {
            Endianness::Little => padded.get_mut(..width)?.copy_from_slice(bytes),
            Endianness::Big => padded
                .get_mut(16usize.checked_sub(width)?..)?
                .copy_from_slice(bytes),
        }
        Self::float(format, endianness.read_u128(&padded))
    }

    /// The integer of `bits` and `signed` that `bytes` encode, when they are
    /// exactly that wide; one wider than 64 bits is the one-lane vector the
    /// evaluator computes such an integer as ([`Self::of_lane`]).
    fn scalar_from_bytes(
        bytes: &[u8],
        bits: u32,
        signed: bool,
        endianness: Endianness,
    ) -> Option<Self> {
        if usize::try_from(bits / 8).ok()? != bytes.len() || !bits.is_multiple_of(8) {
            return None;
        }
        Some(match (bits, signed) {
            (8, false) => Self::U8(*bytes.first()?),
            (8, true) => Self::I8(i8::from_ne_bytes([*bytes.first()?])),
            (16, false) => Self::U16(endianness.read_u16(bytes)),
            (16, true) => Self::I16(endianness.read_u16(bytes).cast_signed()),
            (32, false) => Self::U32(endianness.read_u32(bytes)),
            (32, true) => Self::I32(endianness.read_u32(bytes).cast_signed()),
            (64, false) => Self::U64(endianness.read_u64(bytes)),
            (64, true) => Self::I64(endianness.read_u64(bytes).cast_signed()),
            // A 128-bit register read whole from the image: MPX's bound pair.
            _ => Self::of_lane(bits, endianness.read_wide_uint(bytes)?)?,
        })
    }
}
