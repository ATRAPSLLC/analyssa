//! What a value's type declares of it: the width, lanes and integer type of a
//! register ([`Destination`]), and the reading of an operand ([`OperandLayout`]).

use std::borrow::Cow;

use crate::{
    PointerSize,
    ir::{
        float::FloatFormat, function::SsaFunction, integer::IntegerType, ops::SsaOp,
        value::ConstValue, variable::SsaVarId,
    },
    target::{Target, TypeLayout, VectorShape},
};

/// What an operation's destination declares of the value written to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Destination {
    /// The declared width in bits, if the type states one.
    pub bits: Option<u32>,
    /// The width and count of a vector destination's lanes.
    pub lanes: Option<(u32, u32)>,
    /// The format of a vector's float lanes, which their width alone does
    /// not state (binary16 and bfloat16 are both 16 bits wide).
    pub lane_format: Option<FloatFormat>,
    /// Whether the destination's type is an integer, which holds a float
    /// result as its bits.
    pub integer: bool,
    /// The scalar integer range and constant representation, when attested.
    ///
    /// Explicit vectors have no scalar integer representation. A width alone
    /// does not establish signedness or authorize materializing a bit proof.
    pub integer_type: Option<IntegerType>,
}

/// A destination of a known width, or of none, stating no lanes.
impl From<Option<u32>> for Destination {
    fn from(bits: Option<u32>) -> Self {
        Self {
            bits,
            lanes: None,
            lane_format: None,
            integer: false,
            integer_type: None,
        }
    }
}

/// Returns a scalar integer zero at a supported width.
pub fn scalar_zero<T: Target>(bits: u32) -> Option<ConstValue<T>> {
    match bits {
        1 => Some(ConstValue::False),
        8 | 16 | 32 | 64 => ConstValue::integer(bits, true, 0),
        2..=128 => ConstValue::of_lane(bits, 0),
        _ => None,
    }
}

/// The declared numeric interpretation of an SSA operand.
///
/// Register layouts preserve explicit vector geometry. Scalar floating layouts
/// name a format, since a bit width alone cannot distinguish binary16 and bfloat16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperandLayout {
    /// Logical zero or one, distinct from a signed one-bit register.
    Boolean,
    /// A register using the supplied integer and vector declarations.
    Register(Destination),
    /// A scalar floating register in the exact declared format.
    Float(FloatFormat),
}

impl Default for OperandLayout {
    fn default() -> Self {
        Self::Register(Destination::default())
    }
}

/// Preserves the register declarations of a destination.
impl From<Destination> for OperandLayout {
    fn from(value: Destination) -> Self {
        Self::Register(value)
    }
}

impl OperandLayout {
    /// The declared numeric format of the SSA value `value` of `ssa`, read
    /// without cloning its host type.
    pub fn of<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId) -> Self {
        if ssa
            .variable(value)
            .is_some_and(|value| T::is_boolean(value.var_type()))
        {
            return Self::Boolean;
        }
        let register = Destination::of(ssa, value);
        if !register.integer
            && register.lanes.is_none()
            && let Some(format) = ssa
                .variable(value)
                .and_then(|value| T::float_format(value.var_type()))
        {
            Self::Float(format)
        } else {
            Self::Register(register)
        }
    }

    /// Projects one operand's exact scalar or vector interpretation.
    pub fn from_type_layout(layout: TypeLayout) -> Self {
        if layout.is_boolean() {
            return Self::Boolean;
        }
        let register = Destination::from_type_layout(layout);
        if !register.integer
            && register.lanes.is_none()
            && let Some(format) = layout.float_format()
        {
            return Self::Float(format);
        }
        Self::Register(register)
    }

    /// Reads the Boolean value bit without turning it into a signed scalar lane.
    fn read_boolean<T: Target>(
        value: &ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        value.logical_boolean(ptr_size).map(ConstValue::from_bool)
    }

    /// Reads an owned value through its declared numeric interpretation.
    pub fn read_numeric<T: Target>(
        self,
        value: ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        match self {
            Self::Boolean => Self::read_boolean(&value, ptr_size),
            Self::Float(format) => value.as_float(format, ptr_size).map(ConstValue::Float),
            Self::Register(dest) => {
                if dest.integer
                    && dest.lanes.is_none()
                    && let ConstValue::Float(float) = &value
                {
                    if dest.bits.is_some_and(|bits| bits < float.format().bits()) {
                        return None;
                    }
                    let value = ConstValue::of_lane(float.format().bits(), float.bits())?;
                    return Some(match dest.bits {
                        Some(bits) => integer_carrier(value, bits, ptr_size),
                        None => value,
                    });
                }
                Some(value)
            }
        }
    }

    /// Reads a borrowed numeric value without copying packets or opaque metadata.
    pub fn read_numeric_ref<T: Target>(
        self,
        value: &ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<Cow<'_, ConstValue<T>>> {
        match self {
            Self::Boolean if value.is_bool() => Some(Cow::Borrowed(value)),
            Self::Boolean => Self::read_boolean(value, ptr_size).map(Cow::Owned),
            Self::Float(format) => value
                .as_float(format, ptr_size)
                .map(ConstValue::Float)
                .map(Cow::Owned),
            Self::Register(dest) if dest.integer && dest.lanes.is_none() && value.is_float() => {
                if let ConstValue::Float(float) = value {
                    self.read_numeric(ConstValue::Float(*float), ptr_size)
                        .map(Cow::Owned)
                } else {
                    Some(Cow::Borrowed(value))
                }
            }
            Self::Register(_) => Some(Cow::Borrowed(value)),
        }
    }

    /// Reads an integer conversion at its input's own declared scalar width.
    pub fn read_conversion<T: Target>(
        self,
        value: ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        let value = self.read_numeric(value, ptr_size)?;
        Some(match self {
            Self::Register(dest) if dest.integer && dest.lanes.is_none() => {
                dest.read_comparison(value, ptr_size)
            }
            _ => value,
        })
    }

    /// Reads comparison inputs independently of their predicate's output width.
    pub fn read_comparison<T: Target>(
        self,
        value: ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        let value = self.read_numeric(value, ptr_size)?;
        Some(match self {
            Self::Register(dest) => dest.read_comparison(value, ptr_size),
            Self::Float(_) | Self::Boolean => value,
        })
    }

    /// Reads borrowed comparison inputs using their attested numeric format.
    pub fn read_comparison_ref<T: Target>(
        self,
        value: &ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<Cow<'_, ConstValue<T>>> {
        match self.read_numeric_ref(value, ptr_size)? {
            Cow::Owned(value) => self.read_comparison(value, ptr_size).map(Cow::Owned),
            Cow::Borrowed(value) => Some(match self {
                Self::Register(dest) => dest.read_comparison_ref(value, ptr_size),
                Self::Float(_) | Self::Boolean => Cow::Borrowed(value),
            }),
        }
    }
}

/// Reads a scalar integer carrier at an attested width, retaining explicit vector geometry elsewhere.
pub fn integer_carrier<T: Target>(
    value: ConstValue<T>,
    width: u32,
    ptr_size: PointerSize,
) -> ConstValue<T> {
    if !(1..=128).contains(&width) {
        return value;
    }
    if let ConstValue::Vector(vector) = &value {
        // A wide single lane already is the canonical scalar carrier.
        // The resolver owns the packet; normalization needs no second copy.
        if width > 64 && vector.lanes() == 1 && vector.lane_bits() == width {
            return value;
        }
        return ConstValue::of_lane(width, vector.bits_at(0, width)).unwrap_or(value);
    }
    if value.bit_width(ptr_size) == Some(width) {
        return value.mask_native(ptr_size);
    }
    value
        .scalar_integer_at_width(width, ptr_size)
        .unwrap_or(value)
}

/// Returns a bounded scalar replacement, or `None` when the carrier can stay borrowed.
pub fn read_comparison_carrier<T: Target>(
    value: &ConstValue<T>,
    width: u32,
    ptr_size: PointerSize,
) -> Option<ConstValue<T>> {
    if !(1..=128).contains(&width) {
        return None;
    }
    match value {
        ConstValue::True => return ConstValue::of_lane(width, 1),
        ConstValue::False => return ConstValue::of_lane(width, 0),
        _ => {}
    }
    if let ConstValue::Vector(vector) = value {
        return ConstValue::of_lane(width, vector.bits_at(0, width));
    }
    if value.bit_width(ptr_size) == Some(width) {
        return match value {
            ConstValue::NativeInt(value) => {
                Some(ConstValue::NativeInt(ptr_size.mask_signed(*value)))
            }
            ConstValue::NativeUInt(value) => {
                Some(ConstValue::NativeUInt(ptr_size.mask_unsigned(*value)))
            }
            _ => None,
        };
    }
    value.scalar_integer_at_width(width, ptr_size)
}

impl Destination {
    /// What the function's type of `op`'s destination declares: its width
    /// and, for a vector, its lanes; nothing for an operation without one.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function declaring the destination variable.
    /// * `op` - The operation whose destination to look up.
    #[must_use]
    pub fn of_operation<T: Target>(ssa: &SsaFunction<T>, op: &SsaOp<T>) -> Self {
        op.dest()
            .map_or_else(Self::default, |value| Self::of(ssa, value))
    }

    /// The attested layout of the SSA value `value` of `ssa`, read without
    /// cloning its host type.
    pub fn of<T: Target>(ssa: &SsaFunction<T>, value: SsaVarId) -> Self {
        let ty = ssa.variable(value).map(|variable| variable.var_type());
        Self {
            bits: ty.and_then(T::bit_width),
            lanes: ty
                .and_then(T::vector_shape)
                .map(|shape| (shape.lane_bits, shape.lane_count)),
            lane_format: ty
                .and_then(T::vector_shape)
                .and_then(VectorShape::lane_format),
            integer: ty.is_some_and(T::is_integer),
            integer_type: ty
                .filter(|ty| T::is_integer(ty) && T::vector_shape(ty).is_none())
                .and_then(|ty| IntegerType::new(T::bit_width(ty)?, T::is_signed(ty))),
        }
    }

    /// Projects fixed semantic facts without querying host metadata.
    pub fn from_type_layout(layout: TypeLayout) -> Self {
        Self {
            bits: layout.bit_width(),
            lanes: layout
                .vector_shape()
                .map(|shape| (shape.lane_bits, shape.lane_count)),
            lane_format: layout.vector_shape().and_then(VectorShape::lane_format),
            integer: layout.is_integer(),
            integer_type: layout.integer_conversion_type(),
        }
    }

    /// Reads a borrowed comparison input without copying host metadata or complete packets.
    pub fn read_comparison_ref<T: Target>(
        self,
        value: &ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Cow<'_, ConstValue<T>> {
        match (self.integer, self.lanes, self.bits) {
            (true, None, Some(width)) => read_comparison_carrier(value, width, ptr_size)
                .map_or(Cow::Borrowed(value), Cow::Owned),
            _ => Cow::Borrowed(value),
        }
    }

    /// Reads a comparison input using its own attested scalar integer layout.
    pub fn read_comparison<T: Target>(
        self,
        value: ConstValue<T>,
        ptr_size: PointerSize,
    ) -> ConstValue<T> {
        match (self.integer, self.lanes, self.bits) {
            (true, None, Some(width)) if value.is_bool() => {
                read_comparison_carrier(&value, width, ptr_size).unwrap_or(value)
            }
            (true, None, Some(width)) => integer_carrier(value, width, ptr_size),
            _ => value,
        }
    }

    /// Whether this is one integer, its type as wide as the destination, no
    /// lanes.
    #[must_use]
    pub fn is_integer_scalar(self) -> bool {
        self.integer
            && self.lanes.is_none()
            && self
                .integer_type
                .is_some_and(|layout| Some(layout.bits()) == self.bits)
    }
}
