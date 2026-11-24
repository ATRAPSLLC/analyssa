//! Value types: constants, abstract value lattice, and CSE tracking.
//!
//! This module provides the value representation system for SSA variables, enabling
//! constant propagation, abstract interpretation, and common subexpression elimination.
//!
//! # Key Types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`ConstValue`] | Concrete compile-time constants (integers, floats, strings, null, etc.) |
//! | [`AbstractValue`] | Abstract lattice for dataflow analysis (constant, range, non-null, top, bottom) |
//! | [`ComputedValue`] | Represents a pure computation result for CSE |
//! | [`ComputedOp`] | Operation codes tracked for CSE (arithmetic, bitwise, comparison, conversion) |
//!
//! # Abstract Value Lattice
//!
//! The `AbstractValue` type forms a lattice for monotone dataflow analysis:
//!
//! ```text
//!               Top
//!              / | \
//!         Const Range NonNull
//!              \ | /
//!              Bottom
//! ```
//!
//! - **Top**: No information yet (initial state, join identity)
//! - **Constant**: Known compile-time constant value
//! - **Range**: Value in a bounded inclusive range `[min, max]`
//! - **NonNull**: Known non-null reference
//! - **SameAs**: Equal to another SSA variable (for copy propagation)
//! - **Computed**: Result of a pure computation (for CSE)
//! - **Bottom**: Conflicting information (meet identity, unreachable)
//!
//! The lattice operations are:
//! - **Meet** (greatest lower bound): Used at control flow joins to compute
//!   what is definitely true on all incoming paths
//! - **Join** (least upper bound): Used to find what is possibly true on
//!   any incoming path

use std::{
    fmt,
    hash::{Hash, Hasher},
    mem,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    PointerSize,
    ir::{
        float::{FloatBits, FloatFormat, FloatRules},
        integer::BitWidth,
        lanes::VectorConst,
        ops::{FlagsMask, VectorBinaryKind, VectorElement, VectorElementKind},
    },
    target::Target,
};

mod abstracts;
mod arithmetic;
mod bits;
mod computed;
mod convert;
mod emit;
mod encoding;
mod order;

pub use self::{
    abstracts::AbstractValue,
    computed::{ComputedOp, ComputedValue},
};

/// The scalar bit count computed from a complete register pattern.
#[derive(Clone, Copy)]
pub enum RegisterBitCount {
    /// Index of the least significant set bit; zero has no index.
    First,
    /// Index of the most significant set bit; zero has no index.
    Last,
    /// Number of set bits.
    Population,
    /// Number of set bits modulo two.
    Parity,
}

impl RegisterBitCount {
    /// Counts primitive register bits without a 128-bit intermediate.
    #[inline]
    fn narrow(self, pattern: u64) -> Option<u32> {
        match self {
            Self::First if pattern != 0 => Some(pattern.trailing_zeros()),
            Self::Last if pattern != 0 => 63u32.checked_sub(pattern.leading_zeros()),
            Self::Population => Some(pattern.count_ones()),
            Self::Parity => Some(pattern.count_ones() % 2),
            _ => None,
        }
    }

    /// Counts packed or wide register bits over their complete domain.
    fn wide(self, pattern: u128) -> Option<u32> {
        match self {
            Self::First if pattern != 0 => Some(pattern.trailing_zeros()),
            Self::Last if pattern != 0 => 127u32.checked_sub(pattern.leading_zeros()),
            Self::Population => Some(pattern.count_ones()),
            Self::Parity => Some(pattern.count_ones() % 2),
            _ => None,
        }
    }
}

/// Compile-time constant values tracked through SSA analysis.
///
/// Represents all constant forms that can appear in the IR: numeric literals,
/// string references, null, booleans, and runtime handles. Supports constant
/// folding for arithmetic, bitwise, comparison, and conversion operations.
///
/// Generic over `T: Target` so metadata handles (type refs, method refs, field refs)
/// carry the host's reference types rather than generic placeholders.
///
/// # Equality
///
/// [`PartialEq`]/[`Eq`]/[`Hash`] model **structural constant identity**: "are
/// these the same constant?", not "would these compare equal at runtime?".
/// Floating-point arms are therefore compared and hashed **bitwise**, which is
/// what makes [`Eq`] sound (`NaN` is reflexive here) and lets constants be used
/// as hash-map keys - global value numbering keys on exactly this.
///
/// Two consequences follow, both deliberate:
///
/// - a NaN equals the same NaN (identical bit patterns).
/// - `0.0` and `-0.0` differ (distinct bit patterns, and distinguishable at
///   runtime via `1.0 / x`).
///
/// IEEE-754 *semantic* comparison is a separate operation: use [`Self::ceq`]
/// and the `c*_un` family, which implement CIL/IEEE ordering and unordered
/// semantics.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::TypeRef: serde::Serialize, T::MethodRef: serde::Serialize, \
                     T::FieldRef: serde::Serialize, T::SymbolRef: serde::Serialize",
        deserialize = "T::TypeRef: serde::Deserialize<'de>, \
                       T::MethodRef: serde::Deserialize<'de>, \
                       T::FieldRef: serde::Deserialize<'de>, T::SymbolRef: serde::Deserialize<'de>"
    ))
)]
pub enum ConstValue<T: Target> {
    /// Signed 8-bit integer constant (CIL: `int8`).
    I8(i8),

    /// Signed 16-bit integer constant (CIL: `int16`).
    I16(i16),

    /// Signed 32-bit integer constant (CIL: `int32`).
    I32(i32),

    /// Signed 64-bit integer constant (CIL: `int64`).
    I64(i64),

    /// Unsigned 8-bit integer constant (CIL: `uint8`).
    U8(u8),

    /// Unsigned 16-bit integer constant (CIL: `uint16`).
    U16(u16),

    /// Unsigned 32-bit integer constant (CIL: `uint32`).
    U32(u32),

    /// Unsigned 64-bit integer constant (CIL: `uint64`).
    U64(u64),

    /// Native-size signed integer constant.
    /// Width depends on the target pointer size (32-bit or 64-bit).
    /// Automatically masked to pointer width by `mask_native()`.
    NativeInt(i64),

    /// Native-size unsigned integer constant.
    /// Width depends on the target pointer size.
    NativeUInt(u64),

    /// A float constant of any format, by its bits.
    Float(FloatBits),

    /// A vector constant: its bits, laid out in its lanes.
    Vector(VectorConst),

    /// String constant referenced by its metadata token (index into #US heap).
    /// The actual string content is resolved during code generation.
    String(u32),

    /// Decrypted string with inline content (not just a heap index).
    /// Used by deobfuscation passes that decrypt strings at analysis time.
    /// Contains the actual string bytes for codegen to emit. Boxed to keep
    /// the common scalar [`ConstValue`] variants compact.
    DecryptedString(Box<str>),

    /// Null reference constant.
    Null,

    /// Boolean true constant.
    True,

    /// Boolean false constant.
    False,

    /// Runtime type handle (result of `typeof()` or `ldtoken`).
    Type(T::TypeRef),

    /// Runtime method handle (result of `ldtoken` for a method).
    MethodHandle(T::MethodRef),

    /// Runtime field handle (result of `ldtoken` for a field).
    FieldHandle(T::FieldRef),

    /// A located or named entity used as a value: the address of a function, a
    /// global, an import, or a metadata member.
    ///
    /// Distinct from the integer arms so a symbolic address survives constant
    /// propagation, GVN and any lowering **as a symbol**, instead of degrading
    /// to a number a consumer would have to re-resolve. Equal-symbol subtraction
    /// and XOR establish zero without exposing an address. Displacements and
    /// numeric address projection require the address model and
    /// [`Target::symbol_address`].
    Symbol(T::SymbolRef),

    /// Decrypted array data from deobfuscation.
    ///
    /// Stores the raw bytes and element type of an array that was decrypted
    /// at analysis time. Code generation emits `newarr` + element stores
    /// to reconstruct the array in the output. The payload is boxed (see
    /// [`DecryptedArrayData`]) to keep [`ConstValue`] small.
    DecryptedArray(Box<DecryptedArrayData<T>>),
}

impl<T: Target> PartialEq for ConstValue<T> {
    /// Compares two constants for structural identity.
    ///
    /// Constants of different arms are never equal, even when they denote the
    /// same mathematical value (`I32(4) != I64(4)`) - the arm is part of the
    /// constant's identity. Floats compare bitwise; see the type-level
    /// documentation for why.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => a == b,
            (Self::I16(a), Self::I16(b)) => a == b,
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::I64(a), Self::I64(b)) => a == b,
            (Self::U8(a), Self::U8(b)) => a == b,
            (Self::U16(a), Self::U16(b)) => a == b,
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::U64(a), Self::U64(b)) => a == b,
            (Self::NativeInt(a), Self::NativeInt(b)) => a == b,
            (Self::NativeUInt(a), Self::NativeUInt(b)) => a == b,
            // Bitwise, so `Eq` is reflexive for `NaN`.
            (Self::Float(a), Self::Float(b)) => a == b,
            (Self::Vector(a), Self::Vector(b)) => a == b,
            (Self::String(a), Self::String(b)) => a == b,
            (Self::DecryptedString(a), Self::DecryptedString(b)) => a == b,
            (Self::Null, Self::Null) | (Self::True, Self::True) | (Self::False, Self::False) => {
                true
            }
            (Self::Type(a), Self::Type(b)) => a == b,
            (Self::MethodHandle(a), Self::MethodHandle(b)) => a == b,
            (Self::FieldHandle(a), Self::FieldHandle(b)) => a == b,
            (Self::Symbol(a), Self::Symbol(b)) => a == b,
            (Self::DecryptedArray(a), Self::DecryptedArray(b)) => a == b,
            _ => false,
        }
    }
}

/// Structural identity is reflexive: the float arms compare bitwise, so
/// `NaN == NaN` holds here even though IEEE-754 says otherwise.
impl<T: Target> Eq for ConstValue<T> {}

impl<T: Target> Hash for ConstValue<T> {
    /// Hashes the constant consistently with [`PartialEq`]: the arm followed by
    /// its payload, with floats hashed by their bit pattern.
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
        match self {
            Self::I8(v) => v.hash(state),
            Self::I16(v) => v.hash(state),
            Self::I32(v) => v.hash(state),
            Self::I64(v) => v.hash(state),
            Self::U8(v) => v.hash(state),
            Self::U16(v) => v.hash(state),
            Self::U32(v) => v.hash(state),
            Self::U64(v) => v.hash(state),
            Self::NativeInt(v) => v.hash(state),
            Self::NativeUInt(v) => v.hash(state),
            Self::Float(v) => v.hash(state),
            Self::Vector(v) => v.hash(state),
            Self::String(v) => v.hash(state),
            Self::DecryptedString(v) => v.hash(state),
            Self::Null | Self::True | Self::False => {}
            Self::Type(v) => v.hash(state),
            Self::MethodHandle(v) => v.hash(state),
            Self::FieldHandle(v) => v.hash(state),
            Self::Symbol(v) => v.hash(state),
            Self::DecryptedArray(v) => v.hash(state),
        }
    }
}

/// Boxed payload for [`ConstValue::DecryptedArray`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "T::TypeRef: serde::Serialize",
        deserialize = "T::TypeRef: serde::Deserialize<'de>"
    ))
)]
pub struct DecryptedArrayData<T: Target> {
    /// Raw bytes of the array data in little-endian element layout.
    pub data: Vec<u8>,

    /// Reference to the element type in the host's metadata
    /// (e.g., `TypeRef`/`TypeDef` token for CIL).
    pub element_type_ref: T::TypeRef,

    /// Size of each array element in bytes (1 for byte, 4 for int32, etc.).
    pub element_size: usize,
}

/// The checked integer operation, with width supplied by its operands.
#[derive(Clone, Copy)]
pub enum CheckedArithmetic {
    /// Checked addition.
    Add,
    /// Checked subtraction.
    Sub,
    /// Checked multiplication.
    Mul,
}

impl<T: Target> ConstValue<T> {
    /// Returns `true` if this is the null constant.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Returns `true` if this is a boolean constant.
    #[must_use]
    pub const fn is_bool(&self) -> bool {
        matches!(self, Self::True | Self::False)
    }

    /// Returns `true` if this is an integer constant (signed or unsigned).
    #[must_use]
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::I8(_)
                | Self::I16(_)
                | Self::I32(_)
                | Self::I64(_)
                | Self::U8(_)
                | Self::U16(_)
                | Self::U32(_)
                | Self::U64(_)
                | Self::NativeInt(_)
                | Self::NativeUInt(_)
        )
    }

    /// Returns `true` if this is a signed integer constant.
    #[must_use]
    pub const fn is_signed(&self) -> bool {
        matches!(
            self,
            Self::I8(_) | Self::I16(_) | Self::I32(_) | Self::I64(_) | Self::NativeInt(_)
        )
    }

    /// Returns `true` if this is an unsigned integer constant.
    #[must_use]
    pub const fn is_unsigned(&self) -> bool {
        matches!(
            self,
            Self::U8(_) | Self::U16(_) | Self::U32(_) | Self::U64(_) | Self::NativeUInt(_)
        )
    }

    /// This constant's bits as an integer of the same width and the given
    /// sign: `I32(-1)` is `U32(0xffff_ffff)` unsigned. A constant that is no
    /// integer of a fixed width (a native-width integer, a float, a symbol) is
    /// itself.
    ///
    /// # Arguments
    ///
    /// * `signed` - Whether the result is a signed integer.
    #[must_use]
    // A reinterpretation of the same bits is the point.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn with_sign(self, signed: bool) -> Self {
        match (self, signed) {
            (Self::U8(v), true) => Self::I8(v as i8),
            (Self::I8(v), false) => Self::U8(v as u8),
            (Self::U16(v), true) => Self::I16(v as i16),
            (Self::I16(v), false) => Self::U16(v as u16),
            (Self::U32(v), true) => Self::I32(v as i32),
            (Self::I32(v), false) => Self::U32(v as u32),
            (Self::U64(v), true) => Self::I64(v as i64),
            (Self::I64(v), false) => Self::U64(v as u64),
            (value, _) => value,
        }
    }

    /// Returns `true` if this is a floating-point constant.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::Float(_))
    }

    /// Returns `true` if this is a vector constant.
    #[must_use]
    pub const fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(_))
    }

    /// Returns `true` if this value is a string (`String` or `DecryptedString`).
    #[must_use]
    pub const fn is_string_like(&self) -> bool {
        matches!(self, Self::String(_) | Self::DecryptedString(_))
    }

    /// Returns the constant as an i32 if applicable.
    #[must_use]
    pub const fn as_i32(&self) -> Option<i32> {
        match self {
            Self::I8(v) => Some(*v as i32),
            Self::I16(v) => Some(*v as i32),
            Self::I32(v) => Some(*v),
            Self::U8(v) => Some(*v as i32),
            Self::U16(v) => Some(*v as i32),
            Self::True => Some(1),
            Self::False => Some(0),
            _ => None,
        }
    }

    /// Measures inline payload copied with a constant or retained in publication.
    ///
    /// Strings, arrays and packed vectors contribute their actual payload bytes.
    /// Scalar variants and host identities contribute zero; fixed headers and
    /// external storage are excluded. This borrows the value without allocation.
    #[must_use]
    pub fn payload_bytes(&self) -> usize {
        match self {
            Self::Vector(vector) => vector.bits().div_ceil(8) as usize,
            Self::DecryptedString(value) => value.len(),
            Self::DecryptedArray(array) => array.data.len(),
            _ => 0,
        }
    }

    /// Returns this constant as an `i64`, widening smaller integers and
    /// fallibly narrowing larger ones.
    ///
    /// Covers every integer and native-int width, the boolean `True` (`1`) /
    /// `False` (`0`) constants, and a [`ConstValue::Symbol`] that denotes an
    /// address; everything else returns `None`.
    ///
    /// The symbol arm is what keeps this - and therefore
    /// [`AddressWalk::normalize`](crate::analysis::address::AddressWalk::normalize), [`AliasKey`](crate::analysis::address::AliasKey), and every range/interval analysis built
    /// on them - working once a host stops emitting materialised addresses as
    /// integer constants. Without it a symbolic address would simply vanish from
    /// the numeric analyses, which reads as "points-to got worse" rather than as
    /// the representation change it actually is.
    ///
    /// # Returns
    ///
    /// The value as an `i64` (a symbol resolved to its address), or `None` for a
    /// constant that has no `i64` representation.
    #[must_use]
    pub fn resolved_i64(&self) -> Option<i64> {
        match self {
            Self::Symbol(symbol) => T::symbol_address(symbol),
            Self::I8(value) => Some(i64::from(*value)),
            Self::I16(value) => Some(i64::from(*value)),
            Self::I32(value) => Some(i64::from(*value)),
            Self::I64(value) => Some(*value),
            Self::U8(value) => Some(i64::from(*value)),
            Self::U16(value) => Some(i64::from(*value)),
            Self::U32(value) => Some(i64::from(*value)),
            Self::U64(value) => i64::try_from(*value).ok(),
            Self::NativeInt(value) => Some(*value),
            Self::NativeUInt(value) => i64::try_from(*value).ok(),
            Self::True => Some(1),
            Self::False => Some(0),
            _ => None,
        }
    }

    /// Returns the constant as an i64 if applicable.
    ///
    /// The value, not the bit pattern: every arm here widens exactly, so an
    /// unsigned constant answers only while it still names the same number.
    /// That is why `U64` and `NativeUInt` carry a guard and the narrower
    /// unsigned widths do not - a `u32` always fits an `i64`, a `u64` above
    /// [`i64::MAX`] never does, and reinterpreting one as a negative would hand
    /// every caller a different number than the program has.
    ///
    /// The guarded arms are the mirror of the `I64(v) if *v >= 0` arm in
    /// [`as_u64`](Self::as_u64), and they matter far past symmetry: a 64-bit
    /// lift names most of its constants `U64`, so a missing arm here leaves the
    /// range domain, SCCP, the threader and the algebraic recognisers reading
    /// "unknown" for a constant that is sitting right in front of them.
    #[must_use]
    #[allow(clippy::match_same_arms)] // NativeInt is semantically different from I64
    #[allow(clippy::cast_possible_wrap)] // Guarded by the <= i64::MAX checks
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I8(v) => Some(*v as i64),
            Self::I16(v) => Some(*v as i64),
            Self::I32(v) => Some(*v as i64),
            Self::I64(v) => Some(*v),
            Self::U8(v) => Some(*v as i64),
            Self::U16(v) => Some(*v as i64),
            Self::U32(v) => Some(*v as i64),
            Self::U64(v) if *v <= i64::MAX as u64 => Some(*v as i64),
            Self::NativeInt(v) => Some(*v),
            Self::NativeUInt(v) if *v <= i64::MAX as u64 => Some(*v as i64),
            Self::True => Some(1),
            Self::False => Some(0),
            _ => None,
        }
    }

    /// Returns the constant as a u64 if applicable (for unsigned operations).
    #[must_use]
    #[allow(clippy::cast_sign_loss)] // Guarded by >= 0 checks
    #[allow(clippy::match_same_arms)] // NativeUInt is semantically different from U64
    pub const fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::NativeUInt(v) => Some(*v),
            Self::I8(v) if *v >= 0 => Some(*v as u64),
            Self::I16(v) if *v >= 0 => Some(*v as u64),
            Self::I32(v) if *v >= 0 => Some(*v as u64),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            Self::True => Some(1),
            Self::False => Some(0),
            _ => None,
        }
    }

    /// Returns the constant as a u32 if applicable (for unsigned operations).
    #[must_use]
    #[allow(clippy::cast_sign_loss)] // Guarded by >= 0 checks
    pub const fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(v) => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            Self::U32(v) => Some(*v),
            Self::I8(v) if *v >= 0 => Some(*v as u32),
            Self::I16(v) if *v >= 0 => Some(*v as u32),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            Self::True => Some(1),
            Self::False => Some(0),
            _ => None,
        }
    }

    /// Returns the constant as an `f32` when it is a binary32 float.
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float(v) => f32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Returns the constant as an `f64` when it is a binary64 float.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => f64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Returns the constant as a bool if applicable.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::False
            | Self::Null
            | Self::I8(0)
            | Self::I16(0)
            | Self::I32(0)
            | Self::I64(0)
            | Self::U8(0)
            | Self::U16(0)
            | Self::U32(0)
            | Self::U64(0)
            | Self::NativeInt(0)
            | Self::NativeUInt(0) => Some(false),
            Self::True
            | Self::I8(_)
            | Self::I16(_)
            | Self::I32(_)
            | Self::I64(_)
            | Self::U8(_)
            | Self::U16(_)
            | Self::U32(_)
            | Self::U64(_)
            | Self::NativeInt(_)
            | Self::NativeUInt(_) => Some(true),
            _ => None,
        }
    }

    /// Creates a boolean constant from a bool value.
    #[must_use]
    pub const fn from_bool(value: bool) -> Self {
        if value { Self::True } else { Self::False }
    }

    /// Returns the string content if this is a `DecryptedString`.
    ///
    /// `String` variants are heap references and do not carry inline content.
    #[must_use]
    pub fn as_decrypted_string(&self) -> Option<&str> {
        match self {
            Self::DecryptedString(s) => Some(&**s),
            _ => None,
        }
    }

    /// Returns `true` if this constant represents zero.
    ///
    /// This includes all numeric zero values and `False`.
    /// Useful for opaque predicate detection where `x ^ x`, `x - x`, `x * 0`, etc. produce zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        matches!(
            self,
            Self::I8(0)
                | Self::I16(0)
                | Self::I32(0)
                | Self::I64(0)
                | Self::U8(0)
                | Self::U16(0)
                | Self::U32(0)
                | Self::U64(0)
                | Self::NativeInt(0)
                | Self::NativeUInt(0)
                | Self::False
        )
    }

    /// Returns `true` if this constant represents one.
    ///
    /// This includes all numeric one values and `True`.
    /// Useful for identity operations and opaque predicate detection.
    #[must_use]
    pub const fn is_one(&self) -> bool {
        matches!(
            self,
            Self::I8(1)
                | Self::I16(1)
                | Self::I32(1)
                | Self::I64(1)
                | Self::U8(1)
                | Self::U16(1)
                | Self::U32(1)
                | Self::U64(1)
                | Self::NativeInt(1)
                | Self::NativeUInt(1)
                | Self::True
        )
    }

    /// Returns `true` if this constant represents negative one (-1).
    ///
    /// This is useful for detecting `x | -1 = -1` patterns in opaque predicates.
    #[must_use]
    pub const fn is_minus_one(&self) -> bool {
        matches!(
            self,
            Self::I8(-1) | Self::I16(-1) | Self::I32(-1) | Self::I64(-1) | Self::NativeInt(-1)
        )
    }

    /// Returns `true` if this constant has all bits set (e.g., -1 for signed, MAX for unsigned).
    ///
    /// This is useful for detecting `x & -1 = x` and `x | -1 = -1` patterns.
    #[must_use]
    pub const fn is_all_ones(&self) -> bool {
        matches!(
            self,
            Self::I8(-1)
                | Self::I16(-1)
                | Self::I32(-1)
                | Self::I64(-1)
                | Self::NativeInt(-1)
                | Self::U8(u8::MAX)
                | Self::U16(u16::MAX)
                | Self::U32(u32::MAX)
                | Self::U64(u64::MAX)
                | Self::NativeUInt(u64::MAX)
        )
    }

    /// Returns `value` as a constant of this constant's own integer type.
    ///
    /// Returns `None` when `self` is not an integer, or when `value` does not
    /// fit the target variant. Use this when a rewrite must materialise a new
    /// constant *alongside* an existing one - a mask, a shift amount - so the
    /// replacement keeps the width the surrounding IR declares. Hard-coding
    /// [`Self::I32`] there silently narrows the type, and mixed-width folds
    /// then sign-extend the narrowed operand rather than using the value meant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use analyssa::{ir::ConstValue, testing::MockTarget};
    ///
    /// let wide: ConstValue<MockTarget> = ConstValue::I64(1 << 32);
    /// assert_eq!(wide.integer_of_same_type(255), Some(ConstValue::I64(255)));
    ///
    /// // A value that does not fit the variant is rejected rather than wrapped.
    /// let narrow: ConstValue<MockTarget> = ConstValue::I8(1);
    /// assert_eq!(narrow.integer_of_same_type(1 << 20), None);
    /// ```
    #[must_use]
    pub fn integer_of_same_type(&self, value: i64) -> Option<Self> {
        Some(match self {
            Self::I8(_) => Self::I8(i8::try_from(value).ok()?),
            Self::I16(_) => Self::I16(i16::try_from(value).ok()?),
            Self::I32(_) => Self::I32(i32::try_from(value).ok()?),
            Self::I64(_) => Self::I64(value),
            Self::U8(_) => Self::U8(u8::try_from(value).ok()?),
            Self::U16(_) => Self::U16(u16::try_from(value).ok()?),
            Self::U32(_) => Self::U32(u32::try_from(value).ok()?),
            Self::U64(_) => Self::U64(u64::try_from(value).ok()?),
            Self::NativeInt(_) => Self::NativeInt(value),
            Self::NativeUInt(_) => Self::NativeUInt(u64::try_from(value).ok()?),
            _ => return None,
        })
    }

    /// Returns a zero constant of the same type as this constant.
    ///
    /// Useful for algebraic simplifications like `x * 0 = 0` where the result
    /// should preserve the type of the operands.
    #[must_use]
    pub const fn zero_of_same_type(&self) -> Self {
        match self {
            Self::I8(_) => Self::I8(0),
            Self::I16(_) => Self::I16(0),
            Self::I64(_) => Self::I64(0),
            Self::U8(_) => Self::U8(0),
            Self::U16(_) => Self::U16(0),
            Self::U32(_) => Self::U32(0),
            Self::U64(_) => Self::U64(0),
            Self::NativeInt(_) => Self::NativeInt(0),
            Self::NativeUInt(_) => Self::NativeUInt(0),
            Self::Float(v) => match FloatBits::new(v.format(), 0) {
                Some(zero) => Self::Float(zero),
                None => Self::I32(0),
            },
            // For non-numeric types (including I32), default to i32
            _ => Self::I32(0),
        }
    }

    /// Combines the integer lanes of two vector constants of one shape by
    /// `kind`, each lane read at its width, signed or unsigned as `signed`
    /// says where `kind` depends on it.
    fn integer_lanes(
        a: &VectorConst,
        b: &VectorConst,
        kind: VectorBinaryKind,
        signed: Option<bool>,
    ) -> Option<Self> {
        let element = VectorElement {
            kind: VectorElementKind::Integer,
            bits: 0,
            scalar: false,
            signed,
            rounding: None,
        };
        a.binary(b, kind, element, None, FloatRules::default())
            .map(Self::Vector)
    }

    /// Shifts or rotates the integer lanes of `a` by the lanes of `b`, as
    /// [`Self::integer_lanes`] does; `None` where a shift's count is at least
    /// its lane's width, as for a scalar.
    fn shifted_lanes(
        a: &VectorConst,
        b: &VectorConst,
        kind: VectorBinaryKind,
        signed: Option<bool>,
    ) -> Option<Self> {
        let rotates = matches!(kind, VectorBinaryKind::Rol | VectorBinaryKind::Ror);
        if !rotates
            && b.lane_values()
                .any(|count| count >= u128::from(a.lane_bits()))
        {
            return None;
        }
        Self::integer_lanes(a, b, kind, signed)
    }

    /// The lanes a shift of this constant by `amount` works on, with the
    /// count of each: a vector shifted by a vector, and an integer wider than
    /// any scalar constant, held as one lane, shifted by a scalar count - a
    /// 128-bit register's high half is a shift of the whole by 64. `None` for
    /// a scalar shift, which the integer variants compute themselves.
    fn lanes_and_count(
        &self,
        amount: &Self,
        ptr_size: PointerSize,
        rotates: bool,
    ) -> Option<(VectorConst, VectorConst)> {
        match (self, amount) {
            (Self::Vector(a), _) if a.lanes() == 1 => {
                let (count, _) = amount.scalar_integer_pattern(ptr_size)?;
                let width = u128::from(a.lane_bits());
                // Read the complete scalar count before encoding a lane. A
                // count's high bits can affect modulo at a non-power-of-two
                // width, and cannot turn an out-of-range shift into zero.
                let count = if rotates {
                    count.checked_rem(width)?
                } else if count < width {
                    count
                } else {
                    return None;
                };
                Some((a.clone(), VectorConst::from_lanes(a.lane_bits(), [count])?))
            }
            (Self::Vector(a), Self::Vector(b)) => Some((a.clone(), b.clone())),
            _ => None,
        }
    }

    /// Returns the constant as a vector: a vector itself, an integer or float
    /// as one lane of its width.
    ///
    /// # Arguments
    ///
    /// * `ptr_size` - Target pointer size, giving the native variants' width.
    #[must_use]
    pub fn as_lanes(&self, ptr_size: PointerSize) -> Option<VectorConst> {
        if self.is_wide_native(ptr_size) {
            let (bits, width) = self.scalar_integer_pattern(ptr_size)?;
            return VectorConst::from_lanes(width, [bits]);
        }
        match self {
            Self::Vector(vector) => Some(vector.clone()),
            Self::Float(value) => VectorConst::from_lanes(value.format().bits(), [value.bits()]),
            _ => {
                let (bits, width) = self.integer_extended(ptr_size)?;
                VectorConst::from_lanes(width, [u128::from(bits)])
            }
        }
    }

    /// Returns the unsigned integer `bits` wide holding `value`'s low bits, or
    /// a one-lane vector where no integer is that wide.
    #[must_use]
    pub fn of_lane(bits: u32, value: u128) -> Option<Self> {
        Some(match bits {
            8 => Self::U8(u8::try_from(value & 0xff).ok()?),
            16 => Self::U16(u16::try_from(value & 0xffff).ok()?),
            32 => Self::U32(u32::try_from(value & 0xffff_ffff).ok()?),
            64 => Self::U64(u64::try_from(value & u128::from(u64::MAX)).ok()?),
            _ => Self::Vector(VectorConst::from_lanes(bits, [value])?),
        })
    }

    /// Returns the float a float operation reads of the constant: a float
    /// itself, or the integer or one-lane vector whose bits are the float of
    /// the IEEE format of its width (32, 64, 80 or 128 bits). A 16-bit
    /// integer names no format: binary16 and bfloat16 are both that wide.
    ///
    /// # Arguments
    ///
    /// * `ptr_size` - Target pointer size, giving the native variants' width.
    #[must_use]
    pub fn read_float(&self, ptr_size: PointerSize) -> Option<FloatBits> {
        if let Self::Float(value) = self {
            return Some(*value);
        }
        let width = match self {
            Self::Vector(vector) => vector.bits(),
            _ => self.scalar_integer_pattern(ptr_size)?.1,
        };
        if width == 16 {
            return None;
        }
        self.as_float(FloatFormat::of_width(width)?, ptr_size)
    }

    /// Returns the constant as a float of `format`: a float of the format
    /// itself, or an integer or packed register exactly as wide whose bits
    /// are the float's - a float operation reads its operand's bits, however
    /// the constant was typed.
    ///
    /// # Arguments
    ///
    /// * `format` - The format the operation reads.
    /// * `ptr_size` - Target pointer size, giving the native variants' width.
    #[must_use]
    pub fn as_float(&self, format: FloatFormat, ptr_size: PointerSize) -> Option<FloatBits> {
        let bits = match self {
            Self::Float(value) => return (value.format() == format).then_some(*value),
            Self::Vector(vector) => {
                if vector.bits() != format.bits() {
                    return None;
                }
                vector.bits_at(0, format.bits())
            }
            _ => {
                let (bits, width) = self.scalar_integer_pattern(ptr_size)?;
                if width != format.bits() {
                    return None;
                }
                bits
            }
        };
        FloatBits::new(format, bits)
    }

    /// Whether this is a float NaN, which no comparison orders.
    #[must_use]
    pub fn is_nan(&self) -> bool {
        match self {
            Self::Float(value) => value.is_nan(),
            _ => false,
        }
    }

    /// Returns this constant's bit width, for the integer variants.
    ///
    /// `NativeInt`/`NativeUInt` report the target's pointer width; everything
    /// non-integral reports `None`.
    ///
    /// # Arguments
    ///
    /// * `ptr_size` - Target pointer size, giving the native variants' width.
    ///
    /// # Returns
    ///
    /// The width in bits of an integer constant, `None` for any other variant.
    #[must_use]
    pub fn bit_width(&self, ptr_size: PointerSize) -> Option<u32> {
        Some(match self {
            Self::I8(_) | Self::U8(_) => 8,
            Self::I16(_) | Self::U16(_) => 16,
            Self::I32(_) | Self::U32(_) => 32,
            Self::I64(_) | Self::U64(_) => 64,
            Self::NativeInt(_) | Self::NativeUInt(_) => ptr_size.bits(),
            _ => return None,
        })
    }

    /// Returns whether a native register needs more than its 64-bit payload.
    #[inline]
    fn is_wide_native(&self, ptr_size: PointerSize) -> bool {
        ptr_size.bits() > 64 && matches!(self, Self::NativeInt(_) | Self::NativeUInt(_))
    }

    /// Returns a scalar register pattern, including packed bitvectors through 128 bits.
    ///
    /// Bit operations inspect a register's complete bit sequence independently
    /// of its carrier's lane geometry. Booleans carry one bit; floats and
    /// metadata are not integer registers.
    #[inline]
    fn register_pattern(&self, ptr_size: PointerSize) -> Option<(u128, u32)> {
        match self {
            Self::True => Some((1, 1)),
            Self::False => Some((0, 1)),
            Self::Vector(vector) if (1..=128).contains(&vector.bits()) => {
                Some((vector.bits_at(0, vector.bits()), vector.bits()))
            }
            _ => self.scalar_integer_pattern(ptr_size),
        }
    }

    /// The integer's value extended to 64 bits by its own signedness, with its
    /// width - the shape a width- and sign-correct fold needs.
    ///
    /// A signed variant sign-extends (`I8(-1)` → `0xFFFF_FFFF_FFFF_FFFF`), an
    /// unsigned one zero-extends (`U8(0xFF)` → `0xFF`): extending by signedness
    /// is what makes `(int8_t)-1 & x` behave as the `0xFFFF_FFFF & x` C promotes
    /// it to, rather than `0xFF & x`. The two accessors line up with the two
    /// cases - `as_i64` sign-extends every signed variant, `as_u64` zero-extends
    /// every unsigned one, and neither is asked for the sign it cannot give.
    /// Native payloads are masked at the target width before extension.
    /// Only the integer variants have a pattern; `True`/`False` are excluded so a
    /// boolean never silently joins an arithmetic fold.
    ///
    /// # Arguments
    ///
    /// * `ptr_size` - Target pointer size, giving the width of the
    ///   `NativeInt`/`NativeUInt` variants.
    ///
    /// # Returns
    ///
    /// `Some((bits, width))` - the sign/zero-extended 64-bit value and the
    /// variant's width in bits - for an integer constant; `None` for a
    /// non-integer variant (float, boolean, string, handle, …).
    #[must_use]
    fn integer_extended(&self, ptr_size: PointerSize) -> Option<(u64, u32)> {
        Some(match self {
            Self::I8(value) => (i64::from(*value).cast_unsigned(), 8),
            Self::U8(value) => (u64::from(*value), 8),
            Self::I16(value) => (i64::from(*value).cast_unsigned(), 16),
            Self::U16(value) => (u64::from(*value), 16),
            Self::I32(value) => (i64::from(*value).cast_unsigned(), 32),
            Self::U32(value) => (u64::from(*value), 32),
            Self::I64(value) => (value.cast_unsigned(), 64),
            Self::U64(value) => (*value, 64),
            Self::NativeInt(value) => (
                ptr_size.mask_signed(*value).cast_unsigned(),
                ptr_size.bits(),
            ),
            Self::NativeUInt(value) => (ptr_size.mask_unsigned(*value), ptr_size.bits()),
            _ => return None,
        })
    }

    /// Rebuilds a constant of *this value's own variant* from a raw 64-bit
    /// pattern, narrowing it to the variant's width - the reconstruction step of
    /// a fold that has computed its result in 64 bits.
    ///
    /// Keying off the variant (not a width/signedness pair) is what lets a fold
    /// preserve an operand's exact type - a `NativeUInt` stays a `NativeUInt` at
    /// the target's pointer width, a `U32` stays a `U32` - instead of collapsing
    /// every 64-bit result onto `I64`/`U64` and every pointer-width one onto
    /// nothing.
    ///
    /// # Arguments
    ///
    /// * `bits` - The raw 64-bit result pattern to narrow into this variant.
    /// * `ptr_size` - Target pointer size, used to mask a `NativeInt`/`NativeUInt`
    ///   result to its pointer width.
    ///
    /// # Returns
    ///
    /// `Some(constant)` of the same variant as `self` for an integer variant;
    /// `None` when `self` is not an integer (nothing to reconstruct into).
    #[must_use]
    fn with_integer_bits(&self, bits: u64, ptr_size: PointerSize) -> Option<Self> {
        Some(match self {
            Self::I8(_) => Self::I8(bits as u8 as i8),
            Self::U8(_) => Self::U8(bits as u8),
            Self::I16(_) => Self::I16(bits as u16 as i16),
            Self::U16(_) => Self::U16(bits as u16),
            Self::I32(_) => Self::I32(bits as u32 as i32),
            Self::U32(_) => Self::U32(bits as u32),
            Self::I64(_) => Self::I64(bits as i64),
            Self::U64(_) => Self::U64(bits),
            Self::NativeInt(_) => Self::NativeInt(ptr_size.mask_signed(bits as i64)),
            Self::NativeUInt(_) => Self::NativeUInt(ptr_size.mask_unsigned(bits)),
            _ => return None,
        })
    }

    /// Folds a binary operation over scalar integer constants at their common width.
    ///
    /// Primitive operands extend by their signedness. A primitive result keeps
    /// the wider carrier (the left on a tie); wider registers and scalar lanes
    /// use a complete bit pattern. Explicit vector-vector operations retain
    /// their lane semantics before reaching this fallback.
    ///
    /// The shared machine evaluator normalizes attested scalar operands to
    /// their operation width before calling these value kernels.
    ///
    /// # Arguments
    ///
    /// * `other` - The right-hand operand.
    /// * `ptr_size` - The width of native registers.
    /// * `op` - The primitive bit operation, evaluated at up to 64 bits.
    /// * `wide_op` - The corresponding operation on complete scalar patterns.
    #[must_use]
    fn fold_integer_bits(
        &self,
        other: &Self,
        ptr_size: PointerSize,
        op: impl FnOnce(u64, u64) -> u64,
        wide_op: impl FnOnce(u128, u128) -> u128,
    ) -> Option<Self> {
        if let (Some((left, left_width)), Some((right, right_width))) = (
            self.integer_extended(ptr_size),
            other.integer_extended(ptr_size),
        ) && left_width <= 64
            && right_width <= 64
        {
            let prototype = if left_width >= right_width {
                self
            } else {
                other
            };
            return prototype.with_integer_bits(op(left, right), ptr_size);
        }
        self.fold_wide_integer_bits(other, ptr_size, wide_op)
    }

    /// Folds complete scalar patterns, extending narrow primitive operands by signedness.
    fn fold_wide_integer_bits(
        &self,
        other: &Self,
        ptr_size: PointerSize,
        op: impl FnOnce(u128, u128) -> u128,
    ) -> Option<Self> {
        let (_, left_width) = self.scalar_integer_pattern(ptr_size)?;
        let (_, right_width) = other.scalar_integer_pattern(ptr_size)?;
        let width = left_width.max(right_width);
        let left = self.scalar_pattern_at_width(width, ptr_size)?;
        let right = other.scalar_pattern_at_width(width, ptr_size)?;
        Self::of_lane(width, op(left, right))
    }

    /// The two constants as the numbers they name, when both name one.
    ///
    /// A comparison asks which of two *numbers* is smaller, and the variant a
    /// lift chose for each is not part of the question. One architecture's
    /// register value arrives as `U32` and its immediate as `I32` - MIPS
    /// `slt $t0, $t1, 4` is exactly that - so a fold that enumerates variant
    /// pairs answers "unknown" for a comparison whose two sides are sitting
    /// right in front of it.
    ///
    /// `None` when either side is not an integer (a float, a null, a vector) or
    /// names a value with no exact `i64` reading, which is where the exact
    /// same-variant arms above are the only sound answer.
    fn as_signed_pair(&self, other: &Self) -> Option<(i64, i64)> {
        Some((self.as_i64()?, other.as_i64()?))
    }

    /// The two constants as unsigned numbers, when both readings are exact.
    ///
    /// The unsigned counterpart of [`as_signed_pair`](Self::as_signed_pair).
    /// It deliberately refuses a negative operand: reading one as unsigned is a
    /// question about its *width*, which two different variants do not agree
    /// on, and the same-variant arms above are where that is decided.
    fn as_unsigned_pair(&self, other: &Self) -> Option<(u64, u64)> {
        Some((self.as_u64()?, other.as_u64()?))
    }

    /// The integer constant of `ty` whose bits are the low bits of `bits`: the
    /// type's width and signedness ([`Target::bit_width`],
    /// [`Target::is_signed`]) pick the variant, so no host conversion hook is
    /// needed to name a value at a width a pass computed.
    ///
    /// # Arguments
    ///
    /// * `bits` - The value's bits, in the low bits of a 64-bit pattern.
    /// * `ty` - An integer type 8, 16, 32 or 64 bits wide.
    ///
    /// # Returns
    ///
    /// The constant, or `None` for a type that is not such an integer.
    #[must_use]
    pub fn from_bits_of_type(bits: u64, ty: &T::Type) -> Option<Self> {
        if !T::is_integer(ty) {
            return None;
        }
        let signed = T::is_signed(ty);
        Some(match (T::bit_width(ty)?, signed) {
            (8, true) => Self::I8(bits as u8 as i8),
            (8, false) => Self::U8(bits as u8),
            (16, true) => Self::I16(bits as u16 as i16),
            (16, false) => Self::U16(bits as u16),
            (32, true) => Self::I32(bits as u32 as i32),
            (32, false) => Self::U32(bits as u32),
            (64, true) => Self::I64(bits.cast_signed()),
            (64, false) => Self::U64(bits),
            _ => return None,
        })
    }

    /// Whether `self` and `other` are one integer at one width, whatever the
    /// signedness each is written with ([`Self::integer_pattern`]): `I32(-1)`
    /// and `U32(u32::MAX)`, or the `I32(0)` and `U32(0)` a compiler's two
    /// tests of one flag compare it with.
    ///
    /// # Arguments
    ///
    /// * `other` - The constant compared with.
    /// * `ptr_size` - Target pointer size, the width of the native integers.
    #[must_use]
    pub fn same_integer(&self, other: &Self, ptr_size: PointerSize) -> bool {
        matches!(
            (self.integer_pattern(ptr_size), other.integer_pattern(ptr_size)),
            (Some(ours), Some(theirs)) if ours == theirs
        )
    }

    /// The constant's bits at its own width, in the low bits of a 64-bit
    /// pattern, and that width: `None` for a constant that is not an integer
    /// of a width (a float, a boolean, a vector).
    ///
    /// # Arguments
    ///
    /// * `ptr_size` - Target pointer size, the width of the native integers.
    #[must_use]
    pub fn integer_pattern(&self, ptr_size: PointerSize) -> Option<(u64, u32)> {
        let (bits, width) = self.integer_extended(ptr_size)?;
        Some((BitWidth::new(width).low_bits(bits), width))
    }

    /// Returns a scalar integer's bits and width, including one wide integer lane.
    ///
    /// Native signed integers are sign-extended before masking to the target
    /// width. A one-lane integer bitvector carries widths through 128 bits;
    /// multi-lane vectors, floats, booleans and metadata handles return `None`.
    #[must_use]
    pub fn scalar_integer_pattern(&self, ptr_size: PointerSize) -> Option<(u128, u32)> {
        let (raw, width) = match self {
            Self::Vector(vector)
                if vector.lanes() == 1 && (1..=128).contains(&vector.lane_bits()) =>
            {
                (vector.lane(0), vector.lane_bits())
            }
            Self::Vector(_) => return None,
            _ => {
                let width = self.bit_width(ptr_size)?;
                let raw = if self.is_signed() {
                    i128::from(self.as_i64()?).cast_unsigned()
                } else {
                    u128::from(self.as_u64()?)
                };
                (raw, width)
            }
        };
        Some((raw & BitWidth::new(width).checked_mask_u128()?, width))
    }

    /// Reads the low value bit of a declared Boolean without interpreting its sign.
    pub fn logical_boolean(&self, ptr_size: PointerSize) -> Option<bool> {
        self.conversion_integer_pattern(ptr_size)
            .map(|(bits, _)| bits & 1 != 0)
    }

    /// Returns a scalar conversion's full pattern, treating booleans as positive zero or one.
    ///
    /// Boolean conversion uses an eight-bit positive carrier. Integer and native
    /// values retain their declared width; a single scalar lane retains up to
    /// 128 bits. Floating values, explicit vectors and metadata return no pattern.
    #[must_use]
    pub fn conversion_integer_pattern(&self, ptr_size: PointerSize) -> Option<(u128, u32)> {
        match self {
            Self::True => Some((1, 8)),
            Self::False => Some((0, 8)),
            _ => self.scalar_integer_pattern(ptr_size),
        }
    }

    /// Returns a scalar's target-width pattern, extending by its carrier's signedness.
    pub fn scalar_pattern_at_width(&self, width: u32, ptr_size: PointerSize) -> Option<u128> {
        if width == 0 || width > 128 {
            return None;
        }
        let (raw, own) = self.scalar_integer_pattern(ptr_size)?;
        let raw = if own < width && self.is_signed() {
            let shift = 128u32.checked_sub(own)?;
            (raw << shift)
                .cast_signed()
                .wrapping_shr(shift)
                .cast_unsigned()
        } else {
            raw
        };
        Some(raw & BitWidth::new(width).checked_mask_u128()?)
    }

    /// Normalizes a scalar carrier to the operation width, extending by its own signedness.
    pub fn scalar_integer_at_width(&self, width: u32, ptr_size: PointerSize) -> Option<Self> {
        if width == 0 || width > 128 {
            return None;
        }
        let (_, own) = self.scalar_integer_pattern(ptr_size)?;
        if own == width {
            return Some(self.clone().mask_native(ptr_size));
        }
        let raw = self.scalar_pattern_at_width(width, ptr_size)?;
        if matches!(width, 8 | 16 | 32 | 64) && !matches!(self, Self::Vector(_)) {
            Self::integer(width, self.is_signed(), raw as u64 as i64)
        } else {
            Self::of_lane(width, raw)
        }
    }

    /// The flag `mask` names of a constant flags value ([`SsaOp::ReadFlags`]).
    ///
    /// A constant flags value holds each flag at its bit of [`FlagsMask`], as
    /// a fold of an operation's flags states them (`fprem`'s condition codes)
    /// and a reset leaves them (zero, every flag clear).
    ///
    /// # Returns
    ///
    /// Whether the flag is set; `None` for a mask of more or fewer than one
    /// flag, whose read is no one truth value, or a constant that is no
    /// integer.
    ///
    /// [`SsaOp::ReadFlags`]: crate::ir::ops::SsaOp::ReadFlags
    #[must_use]
    pub fn read_flag(&self, mask: FlagsMask) -> Option<bool> {
        if !mask.is_single_bit() {
            return None;
        }
        Some(self.as_u64_bits()? & u64::from(mask.bits()) != 0)
    }

    /// The flags value of `self`'s flags outside `mask` and `update`'s
    /// inside it ([`SsaOp::MergeFlags`]), each held at its bit of
    /// [`FlagsMask`] ([`Self::read_flag`]); of `self`'s variant.
    ///
    /// [`SsaOp::MergeFlags`]: crate::ir::ops::SsaOp::MergeFlags
    #[must_use]
    pub fn merged_flags(&self, update: &Self, mask: FlagsMask) -> Option<Self> {
        let mask = u64::from(mask.bits());
        let merged = self.as_u64_bits()? & !mask | update.as_u64_bits()? & mask;
        self.of_same_type_from_bits(merged, 64)
    }

    /// The flags value `self` with the one flag `mask` names set to `bit`
    /// ([`SsaOp::InsertFlag`]), each held at its bit of [`FlagsMask`]
    /// ([`Self::read_flag`]); of `self`'s variant. `None` for a mask of more
    /// or fewer than one flag.
    ///
    /// [`SsaOp::InsertFlag`]: crate::ir::ops::SsaOp::InsertFlag
    #[must_use]
    pub fn with_flag(&self, mask: FlagsMask, bit: bool) -> Option<Self> {
        if !mask.is_single_bit() {
            return None;
        }
        let mask = u64::from(mask.bits());
        let flags = self.as_u64_bits()?;
        self.of_same_type_from_bits(if bit { flags | mask } else { flags & !mask }, 64)
    }

    /// The `bits`-wide two's-complement integer `raw` holds, sign-extended.
    const fn sign_extend(raw: u64, bits: u32) -> i64 {
        let shift = 64u32.saturating_sub(bits);
        raw.wrapping_shl(shift).cast_signed().wrapping_shr(shift)
    }

    /// An integer's bits, read as the unsigned integer of its width would be:
    /// `-1` of eight bits is `0xff`; a truth value is the one bit it holds.
    /// `None` for a constant that is neither.
    #[must_use]
    pub const fn as_u64_bits(&self) -> Option<u64> {
        Some(match self {
            Self::True => 1,
            Self::False => 0,
            Self::I8(v) => *v as u8 as u64,
            Self::I16(v) => *v as u16 as u64,
            Self::I32(v) => *v as u32 as u64,
            Self::I64(v) | Self::NativeInt(v) => v.cast_unsigned(),
            Self::U8(v) => *v as u64,
            Self::U16(v) => *v as u64,
            Self::U32(v) => *v as u64,
            Self::U64(v) | Self::NativeUInt(v) => *v,
            _ => return None,
        })
    }

    /// A constant of `self`'s variant holding the low `bits` of `raw`.
    fn of_same_type_from_bits(&self, raw: u64, bits: u32) -> Option<Self> {
        let raw = BitWidth::new(bits).low_bits(raw);
        // The bits are the value; each arm keeps the ones its type holds.
        #[allow(clippy::cast_possible_truncation)]
        Some(match self {
            Self::I8(_) => Self::I8(raw as u8 as i8),
            Self::I16(_) => Self::I16(raw as u16 as i16),
            Self::I32(_) => Self::I32(raw as u32 as i32),
            Self::I64(_) => Self::I64(raw.cast_signed()),
            Self::NativeInt(_) => Self::NativeInt(Self::sign_extend(raw, bits)),
            Self::U8(_) => Self::U8(raw as u8),
            Self::U16(_) => Self::U16(raw as u16),
            Self::U32(_) => Self::U32(raw as u32),
            Self::U64(_) => Self::U64(raw),
            Self::NativeUInt(_) => Self::NativeUInt(raw),
            _ => return None,
        })
    }

    /// Masks a `ConstValue` to the target pointer width.
    ///
    /// For `NativeInt`, sign-extends from 32-bit on `Bit32`.
    /// For `NativeUInt`, zero-extends from 32-bit on `Bit32`.
    /// All other variants are returned unchanged.
    #[must_use]
    pub fn mask_native(self, ptr_size: PointerSize) -> Self {
        match self {
            Self::NativeInt(v) => Self::NativeInt(ptr_size.mask_signed(v)),
            Self::NativeUInt(v) => Self::NativeUInt(ptr_size.mask_unsigned(v)),
            other => other,
        }
    }

    /// The integer `value` as a constant of `bits` width and signedness,
    /// wrapped to that width; `None` for a width no integer constant has.
    ///
    /// # Arguments
    ///
    /// * `bits` - 8, 16, 32 or 64.
    /// * `signed` - Whether the constant is signed.
    /// * `value` - The integer, as a two's-complement 64-bit value.
    #[must_use]
    pub fn integer(bits: u32, signed: bool, value: i64) -> Option<Self> {
        // Truncation to the requested width is the point.
        #[allow(clippy::cast_possible_truncation)]
        Some(match (bits, signed) {
            (8, true) => Self::I8(value as i8),
            (8, false) => Self::U8(value as u8),
            (16, true) => Self::I16(value as i16),
            (16, false) => Self::U16(value as u16),
            (32, true) => Self::I32(value as i32),
            (32, false) => Self::U32(value as u32),
            (64, true) => Self::I64(value),
            (64, false) => Self::U64(value.cast_unsigned()),
            _ => return None,
        })
    }

    /// The constant of `format` whose bits are `bits`. `None` for bits wider
    /// than the format.
    #[must_use]
    pub fn float(format: FloatFormat, bits: u128) -> Option<Self> {
        FloatBits::new(format, bits).map(Self::Float)
    }

    /// The format and bits of a float constant.
    #[must_use]
    pub fn float_bits(&self) -> Option<(FloatFormat, u128)> {
        match self {
            Self::Float(v) => Some((v.format(), v.bits())),
            _ => None,
        }
    }
}

/// The binary32 constant of a single.
impl<T: Target> From<f32> for ConstValue<T> {
    fn from(value: f32) -> Self {
        Self::Float(FloatBits::from(value))
    }
}

/// The binary64 constant of a double.
impl<T: Target> From<f64> for ConstValue<T> {
    fn from(value: f64) -> Self {
        Self::Float(FloatBits::from(value))
    }
}

impl<T: Target> fmt::Display for ConstValue<T>
where
    T::TypeRef: fmt::Display,
    T::MethodRef: fmt::Display,
    T::FieldRef: fmt::Display,
    T::SymbolRef: fmt::Display,
{
    /// Formats the constant in the IR's compact textual syntax: numeric arms
    /// carry a type suffix (`42i8`, `10UL`, `1.5f`) while the rest use named
    /// forms (`str@3`, `null`, `typeof(...)`, `sym(...)`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I8(v) => write!(f, "{v}i8"),
            Self::I16(v) => write!(f, "{v}i16"),
            Self::I32(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}L"),
            Self::U8(v) => write!(f, "{v}u8"),
            Self::U16(v) => write!(f, "{v}u16"),
            Self::U32(v) => write!(f, "{v}u"),
            Self::U64(v) => write!(f, "{v}UL"),
            Self::NativeInt(v) => write!(f, "{v}n"),
            Self::NativeUInt(v) => write!(f, "{v}un"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Vector(vector) => write!(f, "{vector}"),
            Self::String(idx) => write!(f, "str@{idx}"),
            Self::DecryptedString(s) => write!(f, "\"{}\"", s.escape_default()),
            Self::DecryptedArray(arr) => {
                write!(
                    f,
                    "array[{}x{}]<{}>",
                    arr.data
                        .len()
                        .checked_div(arr.element_size.max(1))
                        .unwrap_or(arr.data.len()),
                    arr.element_size,
                    arr.element_type_ref
                )
            }
            Self::Null => write!(f, "null"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
            Self::Type(t) => write!(f, "typeof({t})"),
            Self::MethodHandle(m) => write!(f, "methodof({m})"),
            Self::FieldHandle(fl) => write!(f, "fieldof({fl})"),
            Self::Symbol(sym) => write!(f, "sym({sym})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use super::{encoding::EncodedBytes, *};
    use crate::{
        ir::{CmpKind, variable::SsaVarId},
        target::{Endianness, TypeLayout},
        testing::{FactfulTarget, MockTarget, MockType},
    };

    type Cv = ConstValue<MockTarget>;
    type Av = AbstractValue<MockTarget>;

    /// **Two addresses of one image compare as their addresses.** `ldr r0,
    /// =__TMC_END__; ldr r3, =__TMC_END__; cmp r3, r0` reads one address
    /// twice: equal, as the same number twice was. Ordered unsigned they are
    /// ordered as their addresses; signed, a load base may straddle the sign,
    /// and nothing is decided. An import, with no address, is an opaque
    /// identity no comparison of constants decides, itself included.
    #[test]
    fn two_addresses_of_one_image_compare_as_their_addresses() {
        let ptr = PointerSize::Bit32;
        let (low, high) = (Cv::Symbol(0x250ec), Cv::Symbol(0x25100));
        assert_eq!(low.ceq_at(&Cv::Symbol(0x250ec), ptr), Some(Cv::True));
        assert_eq!(low.ceq_at(&high, ptr), Some(Cv::False));
        assert_eq!(low.relation(&high, CmpKind::Lt, true, ptr), Some(true));
        assert_eq!(high.relation(&low, CmpKind::Ge, true, ptr), Some(true));
        assert_eq!(low.relation(&high, CmpKind::Lt, false, ptr), None);
        // `FactfulTarget` gives a symbol at `IMPORTED` and above no address.
        let import = ConstValue::<FactfulTarget>::Symbol(FactfulTarget::IMPORTED);
        assert_eq!(import.ceq_at(&import, ptr), None);
        let local = ConstValue::<FactfulTarget>::Symbol(0x10);
        assert_eq!(import.ceq_at(&local, ptr), None);
    }

    /// Booleans and every integer width read as an `i64`, and an
    /// out-of-range unsigned constant does not.
    #[test]
    fn resolved_i64_covers_booleans_and_integers() {
        assert_eq!(Cv::True.resolved_i64(), Some(1));
        assert_eq!(Cv::False.resolved_i64(), Some(0));
        assert_eq!(Cv::I32(-7).resolved_i64(), Some(-7));
        assert_eq!(
            Cv::U64(u64::MAX).resolved_i64(),
            None,
            "an out-of-range unsigned constant does not convert"
        );
    }

    /// **A float's bytes are its IEEE 754 bits, and read back as an integer
    /// they are that pattern.** `3.0f` is `0x40400000` and `3.0` is
    /// `0x4008000000000000`.
    #[test]
    fn a_float_is_its_bit_pattern() {
        let b64 = PointerSize::Bit64;
        assert_eq!(
            Cv::from(3.0f32).to_bytes(Endianness::Little, b64),
            Some(vec![0x00, 0x00, 0x40, 0x40])
        );
        assert_eq!(
            Cv::from(3.0f64).to_bytes(Endianness::Big, b64),
            Some(vec![0x40, 0x08, 0, 0, 0, 0, 0, 0])
        );
        let bytes = Cv::from(3.0f32).to_bytes(Endianness::Little, b64);
        assert_eq!(
            bytes.and_then(|bytes| Cv::from_typed_bytes(
                &bytes,
                &MockType::I32,
                Endianness::Little
            )),
            Some(Cv::I32(0x4040_0000))
        );
    }

    /// **A double-width product is both halves of the exact product.** At each
    /// width, signed and unsigned, at the extremes where a narrower product
    /// would wrap: `u64::MAX²` is `2^128 - 2^65 + 1`, `i64::MIN²` is `2^126`,
    /// `-1 * -1` is `1`, and the mixed widths and non-integers have no value.
    #[test]
    fn a_wide_product_is_both_halves_of_the_exact_product() {
        let b64 = PointerSize::Bit64;
        assert_eq!(
            Cv::U64(u64::MAX).wide_mul(&Cv::U64(u64::MAX), true, b64),
            Some((Cv::U64(1), Cv::U64(u64::MAX - 1)))
        );
        assert_eq!(
            Cv::I64(i64::MIN).wide_mul(&Cv::I64(i64::MIN), false, b64),
            Some((Cv::I64(0), Cv::I64(0x4000_0000_0000_0000)))
        );
        assert_eq!(
            Cv::I64(-1).wide_mul(&Cv::I64(-1), false, b64),
            Some((Cv::I64(1), Cv::I64(0)))
        );
        // The same bits read unsigned: (2^64 - 1)^2.
        assert_eq!(
            Cv::I64(-1).wide_mul(&Cv::I64(-1), true, b64),
            Some((Cv::I64(1), Cv::I64(-2)))
        );
        assert_eq!(
            Cv::U32(0x8000_0000).wide_mul(&Cv::U32(2), true, b64),
            Some((Cv::U32(0), Cv::U32(1)))
        );
        assert_eq!(
            Cv::I32(-3).wide_mul(&Cv::I32(5), false, b64),
            Some((Cv::I32(-15), Cv::I32(-1)))
        );
        assert_eq!(
            Cv::I8(-128).wide_mul(&Cv::I8(-128), false, b64),
            Some((Cv::I8(0), Cv::I8(0x40)))
        );
        assert_eq!(
            Cv::NativeUInt(u64::MAX).wide_mul(&Cv::NativeUInt(2), true, b64),
            Some((Cv::NativeUInt(u64::MAX - 1), Cv::NativeUInt(1)))
        );
        assert_eq!(Cv::U32(1).wide_mul(&Cv::U64(1), true, b64), None);
        assert_eq!(
            Cv::from(1.0f64).wide_mul(&Cv::from(1.0f64), true, b64),
            None
        );
    }

    /// **A double-width division truncates, and traps where it has no value.**
    /// `2^64 / 2` fills the quotient's top bit; a signed `-7 / 2` is `-3`
    /// remainder `-1`; a quotient that does not fit, and a zero divisor, have
    /// none.
    #[test]
    fn a_wide_division_truncates_and_has_no_value_where_it_traps() {
        let b64 = PointerSize::Bit64;
        assert_eq!(
            Cv::U64(0).wide_div(&Cv::U64(1), &Cv::U64(2), true, b64),
            Some((Cv::U64(0x8000_0000_0000_0000), Cv::U64(0)))
        );
        assert_eq!(
            Cv::U64(7).wide_div(&Cv::U64(0), &Cv::U64(2), true, b64),
            Some((Cv::U64(3), Cv::U64(1)))
        );
        assert_eq!(
            Cv::I64(-7).wide_div(&Cv::I64(-1), &Cv::I64(2), false, b64),
            Some((Cv::I64(-3), Cv::I64(-1)))
        );
        assert_eq!(
            Cv::I32(100).wide_div(&Cv::I32(0), &Cv::I32(-7), false, b64),
            Some((Cv::I32(-14), Cv::I32(2)))
        );
        // The quotient of 2^64 by 1 is 65 bits wide.
        assert_eq!(
            Cv::U64(0).wide_div(&Cv::U64(1), &Cv::U64(1), true, b64),
            None
        );
        // i64::MIN / -1 is 2^63, one past i64::MAX.
        assert_eq!(
            Cv::I64(i64::MIN).wide_div(&Cv::I64(-1), &Cv::I64(-1), false, b64),
            None
        );
        assert_eq!(
            Cv::U64(1).wide_div(&Cv::U64(0), &Cv::U64(0), true, b64),
            None
        );
        assert_eq!(
            Cv::U32(1).wide_div(&Cv::U32(0), &Cv::U64(1), true, b64),
            None
        );
    }

    /// **A constant flags value holds each flag at its bit of `FlagsMask`**:
    /// a read gives that bit, a merge takes the masked flags from the update
    /// and the rest from the base, and an insertion sets or clears one flag,
    /// each keeping the base's variant. A read or insertion of more than one
    /// flag is no one truth value and declines, as does a constant that is no
    /// integer.
    #[test]
    fn a_constant_flags_value_holds_each_flag_at_its_bit() {
        let carry_zero = Cv::U32(u32::from(FlagsMask::CARRY.bits() | FlagsMask::ZERO.bits()));
        assert_eq!(carry_zero.read_flag(FlagsMask::CARRY), Some(true));
        assert_eq!(carry_zero.read_flag(FlagsMask::ZERO), Some(true));
        assert_eq!(carry_zero.read_flag(FlagsMask::PARITY), Some(false));
        assert_eq!(Cv::U32(0).read_flag(FlagsMask::SIGN), Some(false));
        assert_eq!(
            carry_zero.read_flag(FlagsMask::CARRY.union(FlagsMask::ZERO)),
            None
        );
        assert_eq!(Cv::from(1.0f64).read_flag(FlagsMask::CARRY), None);

        let sign = Cv::U32(u32::from(FlagsMask::SIGN.bits()));
        assert_eq!(
            carry_zero.merged_flags(&sign, FlagsMask::ZERO.union(FlagsMask::SIGN)),
            Some(Cv::U32(u32::from(
                FlagsMask::CARRY.bits() | FlagsMask::SIGN.bits()
            )))
        );
        assert_eq!(
            carry_zero.with_flag(FlagsMask::ZERO, false),
            Some(Cv::U32(u32::from(FlagsMask::CARRY.bits())))
        );
        assert_eq!(
            Cv::U8(0).with_flag(FlagsMask::CARRY, true),
            Some(Cv::U8(
                u8::try_from(FlagsMask::CARRY.bits()).expect("a low bit")
            ))
        );
        assert_eq!(
            carry_zero.with_flag(FlagsMask::CARRY.union(FlagsMask::ZERO), true),
            None
        );
    }

    /// Projected decoding keeps exact integer, floating and packed-lane semantics.
    #[test]
    fn projected_byte_decoding_preserves_exact_type_facts() {
        for order in [Endianness::Little, Endianness::Big] {
            for (ty, bytes, expected) in [
                (MockType::I8, vec![0xff], Cv::I8(-1)),
                (MockType::U8, vec![0xff], Cv::U8(255)),
                (MockType::I16, vec![0xff; 2], Cv::I16(-1)),
                (MockType::U16, vec![0xff; 2], Cv::U16(u16::MAX)),
                (MockType::I32, vec![0xff; 4], Cv::I32(-1)),
                (MockType::U32, vec![0xff; 4], Cv::U32(u32::MAX)),
                (MockType::I64, vec![0xff; 8], Cv::I64(-1)),
                (MockType::U64, vec![0xff; 8], Cv::U64(u64::MAX)),
            ] {
                let layout = TypeLayout::of::<MockTarget>(&ty, PointerSize::Bit64);
                assert_eq!(Cv::from_layout_bytes(&bytes, layout, order), Some(expected));
                assert_eq!(
                    Cv::from_layout_bytes(&bytes[..bytes.len() - 1], layout, order),
                    None
                );
            }
            for (ty, format, bits) in [
                (MockType::BF16, FloatFormat::BFloat16, 0x3f80u128),
                (MockType::F64, FloatFormat::Double, 0x3ff0_0000_0000_0000),
                (
                    MockType::F80,
                    FloatFormat::X87Extended,
                    0x3fff_8000_0000_0000_0000,
                ),
            ] {
                let count = usize::try_from(format.bits() / 8).unwrap();
                let bytes = match order {
                    Endianness::Little => bits.to_le_bytes()[..count].to_vec(),
                    Endianness::Big => bits.to_be_bytes()[16 - count..].to_vec(),
                };
                let layout = TypeLayout::of::<MockTarget>(&ty, PointerSize::Bit64);
                assert_eq!(
                    Cv::from_layout_bytes(&bytes, layout, order),
                    Cv::float(format, bits)
                );
            }
            let layout = TypeLayout::of::<MockTarget>(&MockType::V16I4, PointerSize::Bit64);
            assert_eq!(
                Cv::from_layout_bytes(&[0x21; 8], layout, order),
                VectorConst::from_lanes(4, (0..16).map(|i| if i % 2 == 0 { 1 } else { 2 }))
                    .map(Cv::Vector)
            );
        }
        for ty in [MockType::NxV4I32, MockType::NxMask4, MockType::Unknown] {
            let layout = TypeLayout::of::<MockTarget>(&ty, PointerSize::Bit64);
            assert_eq!(
                Cv::from_layout_bytes(&[0; 16], layout, Endianness::Little),
                None
            );
        }
    }

    /// **A float in the image reads as the float it is.** `2.0` as a double,
    /// and a pair of doubles as a vector of them, where both used to state
    /// nothing. The integer read of the same bytes is the control.
    #[test]
    fn image_bytes_read_as_the_float_their_type_names() {
        let two = 2.0f64.to_bits().to_le_bytes();
        assert_eq!(
            Cv::from_typed_bytes(&two, &MockType::F64, Endianness::Little),
            Some(Cv::from(2.0f64))
        );
        let pair: Vec<u8> = [2.0f64, -0.5]
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        assert_eq!(
            Cv::from_typed_bytes(&pair, &MockType::V2F64, Endianness::Little),
            VectorConst::from_lanes(64, [2.0f64, -0.5].map(|value| u128::from(value.to_bits())))
                .map(Cv::Vector)
        );
        assert_eq!(
            Cv::from_typed_bytes(&two, &MockType::U64, Endianness::Little),
            Some(Cv::U64(2.0f64.to_bits()))
        );
    }

    /// **An integer wider than 64 bits reads from the image as the one-lane
    /// vector the evaluator computes it as**, in either byte order: MPX's
    /// bound pair loaded whole folds to the two bounds. Bytes too few for the
    /// type read as nothing.
    #[test]
    fn a_128_bit_integer_reads_from_the_image_as_one_wide_lane() {
        let value = 0x1000_0000_0000_0002_9000_0000_0000_0001u128;
        let wide = Cv::of_lane(128, value);
        assert!(wide.is_some());
        assert_eq!(
            Cv::from_typed_bytes(&value.to_le_bytes(), &MockType::U128, Endianness::Little),
            wide
        );
        assert_eq!(
            Cv::from_typed_bytes(&value.to_be_bytes(), &MockType::U128, Endianness::Big),
            wide
        );
        assert_eq!(
            Cv::from_typed_bytes(
                &value.to_le_bytes()[..8],
                &MockType::U128,
                Endianness::Little
            ),
            None
        );
    }

    /// **A lane wider than any one integer reads as its bits.** Two 256-bit
    /// lanes of the bytes `0..64`: little-endian, the lowest byte the lowest
    /// bits of its lane, so the vector read again in 64-bit lanes is the
    /// little-endian words of the bytes; big-endian, each lane's bytes from
    /// its last, so the first 64-bit lane holds bytes 31 down to 24. The
    /// vector writes back as the bytes it was read from.
    #[test]
    fn a_lane_wider_than_an_integer_reads_as_its_bits() {
        let bytes: Vec<u8> = (0..64u8).collect();
        let word = |at: usize, order: Endianness| {
            let word: [u8; 8] = bytes[at..at + 8].try_into().expect("eight bytes");
            u128::from(match order {
                Endianness::Little => u64::from_le_bytes(word),
                Endianness::Big => u64::from_be_bytes(word),
            })
        };
        for order in [Endianness::Little, Endianness::Big] {
            let read = Cv::from_typed_bytes(&bytes, &MockType::V2I256, order);
            let Some(Cv::Vector(vector)) = &read else {
                panic!("{order:?}: {read:?}");
            };
            assert_eq!((vector.lane_bits(), vector.lanes()), (256, 2), "{order:?}");
            let words: Vec<u128> = vector
                .relaned(64)
                .expect("eight words")
                .lane_values()
                .collect();
            let expected: Vec<u128> = match order {
                Endianness::Little => (0..8).map(|at| word(at * 8, order)).collect(),
                Endianness::Big => [24, 16, 8, 0, 56, 48, 40, 32]
                    .into_iter()
                    .map(|at| word(at, order))
                    .collect(),
            };
            assert_eq!(words, expected, "{order:?}");
            assert_eq!(
                read.as_ref()
                    .and_then(|value| value.to_bytes(order, PointerSize::Bit64)),
                Some(bytes.clone()),
                "{order:?} writes back"
            );
        }
        assert_eq!(
            Cv::from_typed_bytes(&bytes[..32], &MockType::V2I256, Endianness::Little),
            None,
            "too few bytes"
        );
    }

    /// **Lanes narrower than a byte pack into bytes as the register holds
    /// them.** Sixteen nibbles `0..=15`, lane 0 lowest, are the bytes `0x10,
    /// 0x32, ... 0xfe` read from the low end, and those bytes read back as
    /// sixteen nibbles are the same vector; a 64-bit integer of the bytes is
    /// the control. Bytes too few for the vector read as nothing.
    #[test]
    fn lanes_narrower_than_a_byte_pack_into_bytes() {
        let nibbles = VectorConst::from_lanes(4, 0..16u128).expect("sixteen nibbles");
        let bytes = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        assert_eq!(
            Cv::Vector(nibbles.clone()).to_bytes(Endianness::Little, PointerSize::Bit64),
            Some(bytes.to_vec())
        );
        assert_eq!(
            Cv::from_typed_bytes(&bytes, &MockType::V16I4, Endianness::Little),
            Some(Cv::Vector(nibbles))
        );
        assert_eq!(
            Cv::from_typed_bytes(&bytes, &MockType::U64, Endianness::Little),
            Some(Cv::U64(0xfedc_ba98_7654_3210))
        );
        assert_eq!(
            Cv::from_typed_bytes(&bytes[..4], &MockType::V16I4, Endianness::Little),
            None
        );
    }

    /// `ConstValue`'s `Eq`/`Hash` model structural constant identity, which is
    /// what lets GVN use an op (and therefore its constants) as a hash-map key.
    /// The float arms must compare bitwise or `Eq` would not be reflexive.
    /// Rust's `wrapping_shl` masks the distance by the *value's own* type width,
    /// so the `I8` arm masked by 7: `U8(0x81) << 8` folded to `U8(0x81)`, leaving
    /// the value unchanged. No ISA does that - x86 masks by 31/63 regardless of
    /// the destination sub-register, AArch64 and RISC-V by the register width -
    /// and `SsaOp::Shl` defines no out-of-range behaviour at all. Declining to
    /// fold is sound under every convention.
    #[test]
    fn shifts_decline_rather_than_wrap_the_distance() {
        type Cv = ConstValue<MockTarget>;
        let ptr = PointerSize::Bit64;

        for (value, distance) in [
            (Cv::U8(0x81), 8i32),
            (Cv::I8(-1), 8),
            (Cv::U16(0x8001), 16),
            (Cv::I32(1), 32),
            (Cv::I64(1), 64),
        ] {
            assert_eq!(
                value.shl(&Cv::I32(distance), ptr),
                None,
                "{value:?} << {distance} is out of range and must not fold"
            );
            assert_eq!(
                value.shr(&Cv::I32(distance), false, ptr),
                None,
                "{value:?} >> {distance} is out of range and must not fold"
            );
        }

        // A negative distance is meaningless and must not be reinterpreted as a
        // huge unsigned one.
        assert_eq!(Cv::I32(1).shl(&Cv::I32(-1), ptr), None);
        assert_eq!(Cv::I32(1).shr(&Cv::I32(-1), false, ptr), None);
    }

    /// Bitwise and wrapping-arithmetic folds combine operands of *any* width and
    /// signedness, extending each by its own signedness to a common width and
    /// typing the result as the wider operand's exact variant. This is the path a
    /// split constant (`movz`/`movk`, `lui`/`ori`) takes: a narrow half combined
    /// with a wide mask or addend.
    #[test]
    fn mixed_width_and_sign_integer_folds_are_bit_correct() {
        let ptr = PointerSize::Bit64;

        // The AArch64 `movk` shape: a signed 64-bit field masked by an unsigned
        // 64-bit literal, and OR-ed with a narrow half. `U32 | I64` was the pair
        // the explicit arms missed.
        assert_eq!(
            Cv::I64(0x811c).bitwise_and(&Cv::U64(0xffff), ptr),
            Some(Cv::I64(0x811c))
        );
        assert_eq!(
            Cv::U32(0x9dc5).bitwise_or(&Cv::I64(0x811c_0000), ptr),
            Some(Cv::I64(0x811c_9dc5))
        );

        // The MIPS prime shape: `U32 + I64`.
        assert_eq!(
            Cv::U32(0x0100_0000).add(&Cv::I64(0x193), ptr),
            Some(Cv::I64(0x0100_0193))
        );

        // A narrow *signed* operand is promoted by sign-extension, not
        // zero-extension: `(int8_t)-1 & (uint32_t)x` is `0xFFFF_FFFF & x`, so
        // `I8(-1) & U32(0xABCD)` keeps every low bit of the wider operand.
        assert_eq!(
            Cv::I8(-1).bitwise_and(&Cv::U32(0xABCD), ptr),
            Some(Cv::U32(0xABCD))
        );
        // Zero-extension would have masked it to the low 8 bits - the bug this
        // guards against.
        assert_ne!(
            Cv::I8(-1).bitwise_and(&Cv::U32(0xABCD), ptr),
            Some(Cv::U32(0x00CD))
        );

        // The result takes the wider operand's exact variant, `NativeUInt`
        // included - it is not collapsed onto `U64`.
        assert_eq!(
            Cv::U32(0xF0).bitwise_or(&Cv::NativeUInt(0x0F), ptr),
            Some(Cv::NativeUInt(0xFF))
        );

        // Wrapping multiply, mixed sign and width.
        assert_eq!(
            Cv::U32(0x1_0000).mul(&Cv::I64(0x1_0000), ptr),
            Some(Cv::I64(0x1_0000_0000))
        );
    }

    /// A native integer has a truth value like every other integer.
    ///
    /// Answering `None` sent a branch on a constant `native int` down whichever
    /// arm a caller's "not known true" fallback chose.
    #[test]
    fn native_integers_have_a_truth_value() {
        assert_eq!(Cv::NativeInt(0).as_bool(), Some(false));
        assert_eq!(Cv::NativeInt(-1).as_bool(), Some(true));
        assert_eq!(Cv::NativeUInt(0).as_bool(), Some(false));
        assert_eq!(Cv::NativeUInt(7).as_bool(), Some(true));
        assert_eq!(Cv::from(1.0f64).as_bool(), None);
    }

    /// Unsigned division reads both bit patterns as unsigned at their width.
    #[test]
    fn unsigned_division_reads_the_bit_pattern() {
        let ptr = PointerSize::Bit64;
        assert_eq!(
            Cv::I32(-8).div_un(&Cv::I32(2), ptr),
            Some(Cv::I32(0x7FFF_FFFC))
        );
        assert_eq!(Cv::I32(-8).div(&Cv::I32(2), ptr), Some(Cv::I32(-4)));
        assert_eq!(Cv::I32(-1).rem_un(&Cv::I32(10), ptr), Some(Cv::I32(5)));
        assert_eq!(Cv::I8(-1).div_un(&Cv::I8(16), ptr), Some(Cv::I8(15)));
        assert_eq!(
            Cv::NativeInt(-2).div_un(&Cv::NativeInt(2), PointerSize::Bit32),
            Some(Cv::NativeInt(0x7FFF_FFFF))
        );
        assert_eq!(Cv::U32(7).div_un(&Cv::U32(0), ptr), None);
        assert_eq!(Cv::I32(7).div_un(&Cv::I64(2), ptr), None, "mixed widths");
        assert_eq!(Cv::from(1.0f64).div_un(&Cv::from(2.0f64), ptr), None);
    }

    /// A shift count held in any integer variant is honoured.
    #[test]
    fn a_shift_count_of_any_integer_variant_folds() {
        let ptr = PointerSize::Bit64;
        assert_eq!(Cv::I32(1).shl(&Cv::U32(3), ptr), Some(Cv::I32(8)));
        assert_eq!(Cv::I64(1).shl(&Cv::I64(40), ptr), Some(Cv::I64(1 << 40)));
        assert_eq!(
            Cv::U32(0x80).shr(&Cv::NativeUInt(4), true, ptr),
            Some(Cv::U32(0x8))
        );
        assert_eq!(Cv::I32(1).shl(&Cv::U32(32), ptr), None);
    }

    /// **An integer wider than any scalar constant shifts by a scalar count.**
    /// A 128-bit value, one lane of its width, shifted right by 64 is its high
    /// half and left by 64 moves its low half up; an arithmetic shift copies
    /// the top bit and a rotate wraps. A count of the whole width declines as
    /// a scalar's does, and a negative one never shifts.
    #[test]
    fn a_wide_integer_shifts_by_a_scalar_count() {
        let ptr = PointerSize::Bit64;
        let value = 0x8000_0000_0000_0002_0000_0000_0000_0001u128;
        let wide = |bits: u128| Cv::of_lane(128, bits);
        let of = wide(value).expect("a 128-bit lane");
        assert_eq!(of.shr(&Cv::U8(64), true, ptr), wide(0x8000_0000_0000_0002));
        assert_eq!(
            of.shr(&Cv::U8(64), false, ptr),
            wide(0xffff_ffff_ffff_ffff_8000_0000_0000_0002)
        );
        assert_eq!(of.shl(&Cv::I32(64), ptr), wide(1 << 64));
        assert_eq!(
            of.rotate(&Cv::U64(64), false, ptr),
            wide(0x0000_0000_0000_0001_8000_0000_0000_0002)
        );
        assert_eq!(of.shl(&Cv::U32(128), ptr), None);
        assert_eq!(of.shr(&Cv::I64(-1), true, ptr), None);
    }

    /// In-range shifts must still fold, at every width.
    #[test]
    fn in_range_shifts_still_fold() {
        type Cv = ConstValue<MockTarget>;
        let ptr = PointerSize::Bit64;

        assert_eq!(Cv::U8(0x01).shl(&Cv::I32(7), ptr), Some(Cv::U8(0x80)));
        assert_eq!(Cv::I32(1).shl(&Cv::I32(31), ptr), Some(Cv::I32(i32::MIN)));
        assert_eq!(Cv::I64(1).shl(&Cv::I32(63), ptr), Some(Cv::I64(i64::MIN)));
        assert_eq!(Cv::I32(-8).shr(&Cv::I32(1), false, ptr), Some(Cv::I32(-4)));
        assert_eq!(
            Cv::U32(0x8000_0000).shr(&Cv::I32(31), true, ptr),
            Some(Cv::U32(1))
        );
    }

    /// The operation's flag decides the shift, whatever the constant's type: an
    /// arithmetic shift of an unsigned-typed value copies its top bit down, and
    /// a logical shift of a signed one does not.
    #[test]
    fn the_flag_not_the_type_decides_the_right_shift() {
        type Cv = ConstValue<MockTarget>;
        let ptr = PointerSize::Bit64;
        let by = |n| Cv::I32(n);
        assert_eq!(Cv::U8(0x80).shr(&by(4), false, ptr), Some(Cv::U8(0xf8)));
        assert_eq!(
            Cv::U16(0x8000).shr(&by(8), false, ptr),
            Some(Cv::U16(0xff80))
        );
        assert_eq!(
            Cv::U32(0x8000_0000).shr(&by(24), false, ptr),
            Some(Cv::U32(0xffff_ff80))
        );
        assert_eq!(
            Cv::U64(1 << 63).shr(&by(60), false, ptr),
            Some(Cv::U64(0xffff_ffff_ffff_fff8))
        );
        assert_eq!(
            Cv::NativeUInt(1 << 63).shr(&by(60), false, ptr),
            Some(Cv::NativeUInt(0xffff_ffff_ffff_fff8))
        );
        assert_eq!(
            Cv::NativeUInt(0x8000_0000).shr(&by(28), false, PointerSize::Bit32),
            Some(Cv::NativeUInt(0xffff_fff8)),
            "a pointer-width value's sign is its bit at the pointer width"
        );
        assert_eq!(Cv::I8(-128).shr(&by(4), true, ptr), Some(Cv::I8(8)));
        assert_eq!(
            Cv::U32(0x7fff_ffff).shr(&by(4), false, ptr),
            Some(Cv::U32(0x07ff_ffff))
        );
    }

    /// Structural equality keys on the arm and payload and hashes floats
    /// bitwise, so `NaN` is reflexive and `±0.0` stay distinct, while IEEE `ceq`
    /// semantics are left untouched.
    #[test]
    fn const_value_equality_is_structural_and_reflexive_for_floats() {
        fn hash_of(v: &Cv) -> u64 {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }

        // Reflexive even for NaN - required for `Eq` to be sound.
        assert_eq!(Cv::from(f64::NAN), Cv::from(f64::NAN));
        assert_eq!(Cv::from(f32::NAN), Cv::from(f32::NAN));
        assert_eq!(hash_of(&Cv::from(f64::NAN)), hash_of(&Cv::from(f64::NAN)));

        // `+0.0` and `-0.0` are distinct constants: they are distinguishable at
        // runtime (`1.0 / x`), so value numbering must not conflate them.
        assert_ne!(Cv::from(0.0f64), Cv::from(-0.0f64));

        // Equal values hash equally (the `Hash`/`Eq` contract).
        assert_eq!(Cv::from(1.5f64), Cv::from(1.5f64));
        assert_eq!(hash_of(&Cv::from(1.5f64)), hash_of(&Cv::from(1.5f64)));

        // The arm is part of the identity: same number, different type.
        assert_ne!(Cv::I32(4), Cv::I64(4));

        // IEEE semantic comparison is a separate operation and is unaffected:
        // `ceq` still reports NaN as not-equal.
        assert_eq!(
            Cv::from(f64::NAN).ceq(&Cv::from(f64::NAN)),
            Some(Cv::False),
            "ceq keeps IEEE semantics; only structural equality is bitwise"
        );
    }

    /// The `is_*` classifiers and `as_*` extractors report the expected arm and
    /// scalar value across the integer, float, boolean, and string variants.
    #[test]
    fn const_value_classifies_and_extracts_scalars() {
        assert!(Cv::Null.is_null());
        assert!(Cv::True.is_bool());
        assert!(Cv::I32(-7).is_integer());
        assert!(Cv::I64(-7).is_signed());
        assert!(Cv::U64(7).is_unsigned());
        assert!(Cv::from(1.5f64).is_float());
        assert!(Cv::String(3).is_string_like());
        assert!(Cv::DecryptedString("secret".into()).is_string_like());

        assert_eq!(Cv::I16(-2).as_i32(), Some(-2));
        assert_eq!(Cv::U32(9).as_i64(), Some(9));
        assert_eq!(Cv::I32(-1).as_u64(), None);
        assert_eq!(Cv::I32(7).as_u32(), Some(7));
        assert_eq!(Cv::from(1.25f32).as_f32(), Some(1.25));
        assert_eq!(Cv::from(2.5f64).as_f64(), Some(2.5));
        assert_eq!(Cv::False.as_bool(), Some(false));
        assert_eq!(Cv::I32(5).as_bool(), Some(true));
        assert_eq!(
            Cv::DecryptedString("abc".into()).as_decrypted_string(),
            Some("abc")
        );
        assert_eq!(Cv::String(1).as_decrypted_string(), None);
    }

    /// The `is_zero`/`is_one`/`is_minus_one`/`is_all_ones` predicates and the
    /// `zero_of_same_type`/`from_bool` constructors preserve each constant's
    /// numeric type and meaning.
    #[test]
    fn const_value_identity_helpers_preserve_numeric_intent() {
        assert!(Cv::I32(0).is_zero());
        assert!(Cv::NativeUInt(0).is_zero());
        assert!(Cv::True.is_one());
        assert!(Cv::I64(-1).is_minus_one());
        assert!(Cv::U32(u32::MAX).is_all_ones());

        assert_eq!(Cv::U16(99).zero_of_same_type(), Cv::U16(0));
        assert_eq!(Cv::from(99.0f64).zero_of_same_type(), Cv::from(0.0f64));
        assert_eq!(Cv::I32(99).zero_of_same_type(), Cv::I32(0));
        assert_eq!(Cv::from_bool(true), Cv::True);
        assert_eq!(Cv::from_bool(false), Cv::False);
    }

    /// The bitwise folds act on boolean constants as on one-bit values, and a
    /// boolean against an integer declines.
    #[test]
    fn bitwise_operations_fold_booleans_as_one_bit_values() {
        let ptr = PointerSize::Bit64;
        for a in [false, true] {
            assert_eq!(Cv::from_bool(a).bitwise_not(ptr), Some(Cv::from_bool(!a)));
            for b in [false, true] {
                let (l, r) = (Cv::from_bool(a), Cv::from_bool(b));
                assert_eq!(l.bitwise_and(&r, ptr), Some(Cv::from_bool(a && b)));
                assert_eq!(l.bitwise_or(&r, ptr), Some(Cv::from_bool(a || b)));
                assert_eq!(l.bitwise_xor(&r, ptr), Some(Cv::from_bool(a != b)));
            }
        }
        // A boolean against an integer has no common width: no fold.
        assert_eq!(Cv::True.bitwise_and(&Cv::U8(1), ptr), None);
        assert_eq!(Cv::U8(1).bitwise_or(&Cv::False, ptr), None);
    }

    /// The scalar arithmetic, bitwise, shift, and division folds produce the
    /// expected results, while division by zero and a mixed int/float pair
    /// decline to fold.
    #[test]
    fn arithmetic_and_bitwise_operations_fold_constants() {
        let ptr = PointerSize::Bit64;

        assert_eq!(Cv::I32(7).add(&Cv::I32(5), ptr), Some(Cv::I32(12)));
        assert_eq!(Cv::I32(7).sub(&Cv::I32(5), ptr), Some(Cv::I32(2)));
        assert_eq!(Cv::I32(7).mul(&Cv::I32(5), ptr), Some(Cv::I32(35)));
        assert_eq!(Cv::I32(7).div(&Cv::I32(0), ptr), None);
        assert_eq!(Cv::I32(7).div(&Cv::I32(2), ptr), Some(Cv::I32(3)));
        assert_eq!(Cv::I32(7).rem(&Cv::I32(2), ptr), Some(Cv::I32(1)));
        assert_eq!(
            Cv::I32(0b1010).bitwise_and(&Cv::I32(0b1100), ptr),
            Some(Cv::I32(0b1000))
        );
        assert_eq!(
            Cv::I32(0b1010).bitwise_or(&Cv::I32(0b0101), ptr),
            Some(Cv::I32(0b1111))
        );
        assert_eq!(
            Cv::I32(0b1010).bitwise_xor(&Cv::I32(0b0011), ptr),
            Some(Cv::I32(0b1001))
        );
        assert_eq!(Cv::I8(1).shl(&Cv::I32(3), ptr), Some(Cv::I8(8)));
        assert_eq!(Cv::I8(-8).shr(&Cv::I32(1), false, ptr), Some(Cv::I8(-4)));
        assert_eq!(Cv::I8(-8).shr(&Cv::I32(1), true, ptr), Some(Cv::I8(124)));
        assert_eq!(Cv::I32(1).add(&Cv::from(1.0f32), ptr), None);
    }

    /// Vector folds apply lane by lane at the lanes' width and abort wholesale
    /// on a shape mismatch; a whole-vector equality is one truth value and a
    /// whole-vector order is none.
    #[test]
    fn vector_constants_fold_lane_wise() {
        let ptr = PointerSize::Bit64;
        let vec = |xs: &[i32]| {
            Cv::Vector(
                VectorConst::from_lanes(32, xs.iter().map(|x| u128::from(x.cast_unsigned())))
                    .expect("a vector"),
            )
        };

        // Binary ops fold element-by-element, wrapping at the lane.
        assert_eq!(
            vec(&[1, 2, i32::MAX]).add(&vec(&[10, 20, 1]), ptr),
            Some(vec(&[11, 22, i32::MIN]))
        );
        assert_eq!(
            vec(&[0b1010, 0b1100]).bitwise_and(&vec(&[0b1100, 0b1010]), ptr),
            Some(vec(&[0b1000, 0b1000]))
        );
        assert_eq!(vec(&[-7, 7]).div(&vec(&[2, -2]), ptr), Some(vec(&[-3, -3])));
        assert_eq!(vec(&[1, 1]).div(&vec(&[1, 0]), ptr), None);
        // A shift by a lane's width or more is declined, as a scalar's is.
        assert_eq!(
            vec(&[1, 1]).shl(&vec(&[3, 31]), ptr),
            Some(vec(&[8, i32::MIN]))
        );
        assert_eq!(vec(&[1, 1]).shl(&vec(&[3, 32]), ptr), None);
        assert_eq!(
            vec(&[-8, 8]).shr(&vec(&[1, 1]), false, ptr),
            Some(vec(&[-4, 4]))
        );
        // Unary ops fold element-by-element.
        assert_eq!(vec(&[1, -2, 3]).negate(ptr), Some(vec(&[-1, 2, -3])));
        assert_eq!(vec(&[0, -1]).bitwise_not(ptr), Some(vec(&[-1, 0])));

        assert_eq!(vec(&[1, 5]).ceq(&vec(&[1, 5])), Some(Cv::True));
        assert_eq!(vec(&[1, 5]).ceq(&vec(&[1, 4])), Some(Cv::False));
        assert_eq!(vec(&[1, 5]).clt(&vec(&[2, 4]), ptr), None);

        // Mismatched shapes abort the whole fold.
        assert_eq!(vec(&[1, 2]).add(&vec(&[1]), ptr), None);
    }

    /// The checked add/sub/mul folds return the result when it fits and `None`
    /// on overflow, for both signed and unsigned operands.
    #[test]
    fn checked_arithmetic_reports_overflow() {
        let ptr = PointerSize::Bit64;

        assert_eq!(
            Cv::I8(120).add_checked(&Cv::I8(7), false, ptr),
            Some(Cv::I8(127))
        );
        assert_eq!(Cv::I8(120).add_checked(&Cv::I8(8), false, ptr), None);
        assert_eq!(
            Cv::U8(10).sub_checked(&Cv::U8(7), true, ptr),
            Some(Cv::U8(3))
        );
        assert_eq!(Cv::U8(7).sub_checked(&Cv::U8(10), true, ptr), None);
        assert_eq!(
            Cv::I16(12).mul_checked(&Cv::I16(11), false, ptr),
            Some(Cv::I16(132))
        );
        assert_eq!(Cv::I16(i16::MAX).mul_checked(&Cv::I16(2), false, ptr), None);
    }

    /// Native-width constants are masked to the target pointer width, both on
    /// construction via `mask_native` and after arithmetic.
    #[test]
    fn native_values_are_masked_to_pointer_width() {
        assert_eq!(
            Cv::NativeUInt(0x1_0000_0000).mask_native(PointerSize::Bit32),
            Cv::NativeUInt(0)
        );
        assert_eq!(
            Cv::NativeInt(0x0000_0000_8000_0000).mask_native(PointerSize::Bit32),
            Cv::NativeInt(i64::from(i32::MIN))
        );
        assert_eq!(
            Cv::NativeUInt(0x1_0000_0000).add(&Cv::NativeUInt(1), PointerSize::Bit32),
            Some(Cv::NativeUInt(1))
        );
    }

    /// The comparison folds respect signedness, cross-width promotion, and the
    /// unordered (`NaN`) result of the `*_un` variants.
    #[test]
    fn comparisons_handle_signed_unsigned_and_nan_cases() {
        assert_eq!(Cv::I32(4).ceq(&Cv::I64(4)), Some(Cv::True));
        assert_eq!(
            Cv::U32(4).clt(&Cv::U64(5), PointerSize::Bit64),
            Some(Cv::True)
        );
        assert_eq!(
            Cv::I8(-1).clt_un(&Cv::I8(1), PointerSize::Bit64),
            Some(Cv::False)
        );
        assert_eq!(
            Cv::I8(-1).cgt_un(&Cv::I8(1), PointerSize::Bit64),
            Some(Cv::True)
        );
        assert_eq!(
            Cv::from(f32::NAN).clt_un(&Cv::from(1.0f32), PointerSize::Bit64),
            Some(Cv::True)
        );
        assert_eq!(
            Cv::from(f64::NAN).cgt_un(&Cv::from(1.0f64), PointerSize::Bit64),
            Some(Cv::True)
        );
        assert_eq!(Cv::Null.ceq(&Cv::Null), Some(Cv::True));
        assert_eq!(Cv::True.ceq(&Cv::False), Some(Cv::False));
    }

    /// Standard integer layouts share conversion semantics without host arithmetic hooks.
    #[test]
    fn default_integer_conversion_hooks_keep_values_and_refuse_invalid_inputs() {
        assert_eq!(
            Cv::I32(1).convert_to(&MockType::I64, false, 8),
            Some(Cv::I64(1))
        );
        assert_eq!(
            Cv::I32(1).convert_to_checked(&MockType::I64, false, 8),
            Some(Cv::I64(1))
        );
        assert_eq!(Cv::I32(1).convert_to(&MockType::Unknown, false, 8), None);
        assert_eq!(
            Cv::I32(256).convert_to_checked(&MockType::I8, false, 8),
            None
        );
    }

    /// `Display` renders each constant, computed value, and op in its textual
    /// form, and `normalized` orders a commutative op's operands.
    #[test]
    fn display_formats_constants_and_computed_values() {
        assert_eq!(Cv::I8(-1).to_string(), "-1i8");
        assert_eq!(Cv::U64(10).to_string(), "10UL");
        assert_eq!(Cv::DecryptedString("a\nb".into()).to_string(), "\"a\\nb\"");
        assert_eq!(
            Cv::DecryptedArray(Box::new(DecryptedArrayData {
                data: vec![1, 0, 2, 0],
                element_type_ref: 7,
                element_size: 2,
            }))
            .to_string(),
            "array[2x2]<7>"
        );
        assert_eq!(Cv::Type(9).to_string(), "typeof(9)");
        assert_eq!(Cv::MethodHandle(10).to_string(), "methodof(10)");
        assert_eq!(Cv::FieldHandle(11).to_string(), "fieldof(11)");

        let computed = ComputedValue::binary(
            ComputedOp::Add,
            SsaVarId::from_index(2),
            SsaVarId::from_index(1),
        )
        .normalized();
        assert_eq!(
            computed.operands,
            vec![SsaVarId::from_index(1), SsaVarId::from_index(2)]
        );
        assert_eq!(computed.to_string(), "add(v1, v2)");
        assert_eq!(
            ComputedValue::unary(ComputedOp::Neg, SsaVarId::from_index(0)).to_string(),
            "neg(v0)"
        );
        assert!(ComputedOp::Ceq.is_commutative());
        assert!(!ComputedOp::Sub.is_commutative());
        assert!(ComputedOp::Cgt.is_comparison());
        assert!(!ComputedOp::Add.is_comparison());
        assert_eq!(ComputedOp::ConvI4.to_string(), "conv.i4");
    }

    #[test]
    fn abstract_value_lattice_meet_and_join_are_predictable() {
        let c1 = Av::Constant(Cv::I32(1));
        let c2 = Av::Constant(Cv::I32(2));

        assert!(Av::Top.is_top());
        assert!(Av::Bottom.is_bottom());
        assert!(c1.is_constant());
        assert_eq!(c1.as_constant(), Some(&Cv::I32(1)));
        assert!(Av::NonNull.is_non_null());
        assert!(!Av::Constant(Cv::Null).meet(&Av::NonNull).is_non_null());

        assert_eq!(Av::Top.meet(&c1), c1);
        assert_eq!(Av::Constant(Cv::I32(1)).meet(&c2), Av::Bottom);
        assert_eq!(
            Av::Range { min: 1, max: 3 }.meet(&Av::Range { min: -2, max: 2 }),
            Av::Range { min: -2, max: 3 }
        );
        assert_eq!(
            Av::SameAs(SsaVarId::from_index(1)).meet(&Av::SameAs(SsaVarId::from_index(1))),
            Av::SameAs(SsaVarId::from_index(1))
        );
        assert_eq!(Av::Bottom.join(&Av::NonNull), Av::NonNull);
        assert_eq!(Av::Top.join(&Av::NonNull), Av::Top);
        assert_eq!(Av::NonNull.join(&Av::NonNull), Av::NonNull);
        assert_eq!(Av::NonNull.join(&Av::Constant(Cv::I32(1))), Av::Top);
        assert_eq!(Av::Range { min: 1, max: 2 }.to_string(), "[1..2]");
    }

    /// A native int is pointer-width by definition. Emitting eight bytes on a
    /// 32-bit target writes four bytes past the value - into a neighbouring cell
    /// for any caller laying out memory from these.
    #[test]
    fn native_ints_serialise_at_the_target_pointer_width() {
        let value = Cv::NativeUInt(0x0102_0304);
        assert_eq!(
            value.to_bytes(Endianness::Little, PointerSize::Bit32),
            Some(vec![0x04, 0x03, 0x02, 0x01])
        );
        assert_eq!(
            value.to_bytes(Endianness::Little, PointerSize::Bit64),
            Some(vec![0x04, 0x03, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00])
        );
        assert_eq!(
            value.to_bytes(Endianness::Big, PointerSize::Bit32),
            Some(vec![0x01, 0x02, 0x03, 0x04])
        );

        // A signed native int follows the same width.
        assert_eq!(
            Cv::NativeInt(-1)
                .to_bytes(Endianness::Little, PointerSize::Bit16)
                .map(|b| b.len()),
            Some(2)
        );

        // Fixed-width variants ignore the pointer size entirely.
        assert_eq!(
            Cv::U32(1).to_bytes(Endianness::Little, PointerSize::Bit16),
            Cv::U32(1).to_bytes(Endianness::Little, PointerSize::Bit64)
        );
    }

    /// An `i8` serialises to a single byte in either endianness.
    #[test]
    fn to_bytes_i8_is_one_byte_regardless_of_endianness() {
        let val = Cv::I8(-1);
        for endianness in [Endianness::Little, Endianness::Big] {
            assert_eq!(
                val.to_bytes(endianness, PointerSize::Bit64),
                Some(vec![0xFF])
            );
        }
    }

    /// A `u8` serialises to a single byte in either endianness.
    #[test]
    fn to_bytes_u8_is_one_byte_regardless_of_endianness() {
        let val = Cv::U8(0xAB);
        for endianness in [Endianness::Little, Endianness::Big] {
            assert_eq!(
                val.to_bytes(endianness, PointerSize::Bit64),
                Some(vec![0xAB])
            );
        }
    }

    /// A 16-bit value's byte order follows the requested endianness.
    #[test]
    fn to_bytes_i16_depends_on_endianness() {
        let val = Cv::I16(0x0102);
        assert_eq!(
            val.to_bytes(Endianness::Little, PointerSize::Bit64),
            Some(vec![0x02, 0x01]),
        );
        assert_eq!(
            val.to_bytes(Endianness::Big, PointerSize::Bit64),
            Some(vec![0x01, 0x02]),
        );
    }

    /// A 32-bit value's byte order follows the requested endianness.
    #[test]
    fn to_bytes_u32_depends_on_endianness() {
        let val = Cv::U32(0x01020304);
        assert_eq!(
            val.to_bytes(Endianness::Little, PointerSize::Bit64),
            Some(vec![0x04, 0x03, 0x02, 0x01]),
        );
        assert_eq!(
            val.to_bytes(Endianness::Big, PointerSize::Bit64),
            Some(vec![0x01, 0x02, 0x03, 0x04]),
        );
    }

    /// A 64-bit value's byte order follows the requested endianness.
    #[test]
    fn to_bytes_i64_depends_on_endianness() {
        let val = Cv::I64(0x0102030405060708);
        assert_eq!(
            val.to_bytes(Endianness::Little, PointerSize::Bit64),
            Some(vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]),
        );
        assert_eq!(
            val.to_bytes(Endianness::Big, PointerSize::Bit64),
            Some(vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
        );
    }

    /// A native unsigned value's byte order follows the requested endianness.
    #[test]
    fn to_bytes_native_uint_depends_on_endianness() {
        let val = Cv::NativeUInt(0x0102030405060708);
        assert_eq!(
            val.to_bytes(Endianness::Little, PointerSize::Bit64),
            Some(vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]),
        );
        assert_eq!(
            val.to_bytes(Endianness::Big, PointerSize::Bit64),
            Some(vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
        );
    }

    /// A native signed value's byte order follows the requested endianness.
    #[test]
    fn to_bytes_native_int_depends_on_endianness() {
        let val = Cv::NativeInt(0x0102030405060708);
        assert_eq!(
            val.to_bytes(Endianness::Little, PointerSize::Bit64),
            Some(vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]),
        );
        assert_eq!(
            val.to_bytes(Endianness::Big, PointerSize::Bit64),
            Some(vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
        );
    }

    /// A float serialises as its IEEE 754 bits; null, booleans and strings have
    /// no serialisation.
    #[test]
    fn to_bytes_gives_floats_their_bits_and_refuses_the_rest() {
        assert_eq!(
            Cv::from(1.0f32).to_bytes(Endianness::Little, PointerSize::Bit64),
            Some(vec![0x00, 0x00, 0x80, 0x3F])
        );
        assert_eq!(
            Cv::from(1.0f64).to_bytes(Endianness::Big, PointerSize::Bit64),
            Some(vec![0x3F, 0xF0, 0, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            Cv::Null.to_bytes(Endianness::Little, PointerSize::Bit64),
            None
        );
        assert_eq!(
            Cv::True.to_bytes(Endianness::Little, PointerSize::Bit64),
            None
        );
        assert_eq!(
            Cv::False.to_bytes(Endianness::Little, PointerSize::Bit64),
            None
        );
        assert_eq!(
            Cv::DecryptedString("hello".into()).to_bytes(Endianness::Little, PointerSize::Bit64),
            None,
        );
    }

    /// Signed native encoding follows mathematical sign extension at every target width.
    #[test]
    fn native_serialization_matches_primitive_signed_and_unsigned_oracles() {
        for ptr_size in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            let width = ptr_size.bytes();
            for order in [Endianness::Little, Endianness::Big] {
                for value in [
                    i64::MIN,
                    -257,
                    -256,
                    -129,
                    -128,
                    -1,
                    0,
                    1,
                    127,
                    128,
                    255,
                    256,
                    i64::MAX,
                ] {
                    let extended = i128::from(value);
                    let encoded = match order {
                        Endianness::Little => extended.to_le_bytes(),
                        Endianness::Big => extended.to_be_bytes(),
                    };
                    let expected = match order {
                        Endianness::Little => encoded.get(..width),
                        Endianness::Big => encoded.get(encoded.len().saturating_sub(width)..),
                    }
                    .map(<[u8]>::to_vec);
                    assert_eq!(
                        Cv::NativeInt(value).to_bytes(order, ptr_size),
                        expected,
                        "signed value={value} ptr={ptr_size:?} order={order:?}"
                    );
                }
                for value in [0, 1, 127, 128, 255, 256, u64::MAX] {
                    let extended = u128::from(value);
                    let encoded = match order {
                        Endianness::Little => extended.to_le_bytes(),
                        Endianness::Big => extended.to_be_bytes(),
                    };
                    let expected = match order {
                        Endianness::Little => encoded.get(..width),
                        Endianness::Big => encoded.get(encoded.len().saturating_sub(width)..),
                    }
                    .map(<[u8]>::to_vec);
                    assert_eq!(
                        Cv::NativeUInt(value).to_bytes(order, ptr_size),
                        expected,
                        "unsigned value={value} ptr={ptr_size:?} order={order:?}"
                    );
                }
            }
        }
    }

    // ConstValue::from_bytes - all supported byte lengths, both endiannesses

    /// A one-byte slice deserialises to `U8`.
    #[test]
    fn from_bytes_1_byte_produces_u8() {
        assert_eq!(
            Cv::from_bytes(&[0xAB], Endianness::Little),
            Some(Cv::U8(0xAB)),
        );
        assert_eq!(Cv::from_bytes(&[0xAB], Endianness::Big), Some(Cv::U8(0xAB)),);
    }

    /// A two-byte slice deserialises to `U16` in the given endianness.
    #[test]
    fn from_bytes_2_bytes_produces_u16() {
        let le = Cv::from_bytes(&[0x02, 0x01], Endianness::Little);
        assert_eq!(le, Some(Cv::U16(0x0102)));
        let be = Cv::from_bytes(&[0x01, 0x02], Endianness::Big);
        assert_eq!(be, Some(Cv::U16(0x0102)));
    }

    /// A four-byte slice deserialises to `U32` in the given endianness.
    #[test]
    fn from_bytes_4_bytes_produces_u32() {
        let le = Cv::from_bytes(&[0x04, 0x03, 0x02, 0x01], Endianness::Little);
        assert_eq!(le, Some(Cv::U32(0x01020304)));
        let be = Cv::from_bytes(&[0x01, 0x02, 0x03, 0x04], Endianness::Big);
        assert_eq!(be, Some(Cv::U32(0x01020304)));
    }

    /// An eight-byte slice deserialises to `U64` in the given endianness.
    #[test]
    fn from_bytes_8_bytes_produces_u64() {
        let bytes_le = [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01];
        let bytes_be = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(
            Cv::from_bytes(&bytes_le, Endianness::Little),
            Some(Cv::U64(0x0102030405060708)),
        );
        assert_eq!(
            Cv::from_bytes(&bytes_be, Endianness::Big),
            Some(Cv::U64(0x0102030405060708)),
        );
    }

    /// A slice whose length is not 1, 2, 4, or 8 does not deserialise.
    #[test]
    fn from_bytes_unsupported_length_returns_none() {
        assert_eq!(Cv::from_bytes(&[], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0, 0, 0], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 5], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 6], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 7], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 9], Endianness::Little), None);
        assert_eq!(Cv::from_bytes(&[0; 16], Endianness::Little), None);
    }

    /// Serialising then deserialising recovers the value as its unsigned twin.
    #[test]
    fn from_bytes_round_trips_with_to_bytes() {
        let test_values: Vec<(Cv, Cv)> = vec![
            (Cv::U8(0xAB), Cv::U8(0xAB)),
            (Cv::U16(0xABCD), Cv::U16(0xABCD)),
            (Cv::U32(0xDEAD_BEEF), Cv::U32(0xDEAD_BEEF)),
            (
                Cv::U64(0xDEAD_BEEF_CAFE_BABE),
                Cv::U64(0xDEAD_BEEF_CAFE_BABE),
            ),
            (Cv::I8(-1), Cv::U8(0xFF)),
            (Cv::I16(-128), Cv::U16(0xFF80)),
            (Cv::I32(i32::MIN), Cv::U32(i32::MIN as u32)),
            (Cv::I64(i64::MIN), Cv::U64(i64::MIN as u64)),
            // NativeInt/NativeUInt → U64 since from_bytes produces unsigned
            (Cv::NativeInt(42), Cv::U64(42)),
            (Cv::NativeUInt(100), Cv::U64(100)),
        ];

        for endianness in [Endianness::Little, Endianness::Big] {
            for (original, expected) in &test_values {
                let bytes = original.to_bytes(endianness, PointerSize::Bit64).unwrap();
                let restored = Cv::from_bytes(&bytes, endianness).unwrap();
                assert_eq!(
                    restored, *expected,
                    "round-trip failed for {original:?} with {endianness:?}",
                );
            }
        }
    }

    /// A 64-bit unsigned constant answers `as_i64` while it names the same
    /// number, and refuses when it cannot.
    ///
    /// A 64-bit lift names most of its constants `U64`, so this arm decides
    /// whether the range domain, SCCP and the threader can read a constant at
    /// all - and the guard is what keeps a value above [`i64::MAX`] from being
    /// handed back as a negative number the program never held.
    #[test]
    fn a_wide_unsigned_constant_reads_as_itself_or_not_at_all() {
        type Cv = ConstValue<MockTarget>;
        assert_eq!(Cv::U64(0).as_i64(), Some(0));
        assert_eq!(Cv::U64(42).as_i64(), Some(42));
        assert_eq!(
            Cv::U64(i64::MAX as u64).as_i64(),
            Some(i64::MAX),
            "the largest value with a reading has one"
        );
        assert_eq!(
            Cv::U64(i64::MAX as u64 + 1).as_i64(),
            None,
            "one past it has none, and must not wrap to a negative"
        );
        assert_eq!(Cv::U64(u64::MAX).as_i64(), None);

        assert_eq!(Cv::NativeUInt(7).as_i64(), Some(7));
        assert_eq!(Cv::NativeUInt(u64::MAX).as_i64(), None);

        // The mirror of the arm `as_u64` already had: the pair agrees on every
        // value both can name.
        for value in [0u64, 1, 42, i64::MAX as u64] {
            assert_eq!(
                Cv::U64(value).as_i64().and_then(|v| Cv::I64(v).as_u64()),
                Some(value)
            );
        }
    }

    /// A comparison of two integer constants reads each at its own width with
    /// the comparison's signedness, whatever variants carry them.
    ///
    /// A lift picks a variant per operand, and one comparison's two sides need
    /// not agree: MIPS `slt $t0, $t1, 4` arrives as `U32` against `I32`. A fold
    /// that enumerates variant pairs answers "unknown" there, and every guard
    /// written that way stays undecided - which is what kept the MIPS builds of
    /// a flattened dispatcher from being threaded. Reading by the variant's
    /// sign instead of the comparison's answers a signed compare of a `U32`
    /// register holding a negative number as if it were positive.
    #[test]
    fn a_comparison_reads_both_sides_as_numbers() {
        type Cv = ConstValue<MockTarget>;

        assert_eq!(
            Cv::U32(0).clt(&Cv::I32(4), PointerSize::Bit64),
            Some(Cv::True)
        );
        assert_eq!(
            Cv::U32(9).clt(&Cv::I32(4), PointerSize::Bit64),
            Some(Cv::False)
        );
        assert_eq!(
            Cv::I32(4).cgt(&Cv::U32(0), PointerSize::Bit64),
            Some(Cv::True)
        );
        assert_eq!(Cv::U8(5).ceq(&Cv::I64(5)), Some(Cv::True));
        assert_eq!(
            Cv::I8(5).cgt(&Cv::I32(3), PointerSize::Bit64),
            Some(Cv::True)
        );
        assert_eq!(
            Cv::U32(3).clt_un(&Cv::U64(9), PointerSize::Bit64),
            Some(Cv::True)
        );

        // What stays unknown, and must: neither side names a number the other
        // can be read against.
        assert_eq!(
            Cv::from(5.0f64).clt(&Cv::I32(9), PointerSize::Bit64),
            None,
            "float against integer"
        );
        assert_eq!(Cv::Null.ceq(&Cv::I32(0)), None, "null names no number");
        assert_eq!(
            Cv::U64(u64::MAX).clt(&Cv::I32(1), PointerSize::Bit64),
            Some(Cv::True),
            "a signed compare reads all ones as -1"
        );
        assert_eq!(
            Cv::U32(0xFFFF_FFF8).clt(&Cv::U32(0), PointerSize::Bit64),
            Some(Cv::True),
            "a signed compare of an unsigned variant reads its sign bit"
        );
        assert_eq!(
            Cv::I32(-1).clt_un(&Cv::U32(5), PointerSize::Bit64),
            Some(Cv::False),
            "an unsigned compare reads -1 at 32 bits as 0xFFFF_FFFF"
        );
        assert_eq!(
            Cv::I8(-1).cgt_un(&Cv::U64(0x100), PointerSize::Bit64),
            Some(Cv::False),
            "an unsigned reading zero-extends the narrower side"
        );
        assert_eq!(
            Cv::NativeUInt(0xFFFF_FFFF).clt(&Cv::I32(0), PointerSize::Bit32),
            Some(Cv::True),
            "a native integer is read at the pointer width"
        );
        assert_eq!(
            Cv::NativeUInt(0xFFFF_FFFF).clt(&Cv::I32(0), PointerSize::Bit64),
            Some(Cv::False)
        );
    }

    /// A rotation takes its count modulo the value's width whatever the
    /// count's variant, and rotates within that width: a 64-bit count is read
    /// (an RV64 register), 65 rotates by 1, a negative count rotates the other
    /// way, and a native integer rotates at the pointer's width.
    #[test]
    fn a_rotation_reads_its_count_modulo_the_width() {
        let ptr = PointerSize::Bit64;
        assert_eq!(
            Cv::U64(1).rotate(&Cv::U64(63), false, ptr),
            Some(Cv::U64(1 << 63))
        );
        assert_eq!(
            Cv::U64(1).rotate(&Cv::I64(65), false, ptr),
            Some(Cv::U64(2))
        );
        assert_eq!(
            Cv::U64(1).rotate(&Cv::I32(-1), false, ptr),
            Some(Cv::U64(1 << 63)),
            "-1 is 63 modulo 64"
        );
        assert_eq!(
            Cv::I64(1).rotate(&Cv::U64(1), true, ptr),
            Some(Cv::I64(i64::MIN))
        );
        assert_eq!(
            Cv::U8(0x81).rotate(&Cv::U64(1), false, ptr),
            Some(Cv::U8(0x03))
        );
        assert_eq!(
            Cv::I32(-1).rotate(&Cv::U64(7), true, ptr),
            Some(Cv::I32(-1))
        );
        assert_eq!(
            Cv::NativeUInt(1).rotate(&Cv::U64(1), true, PointerSize::Bit32),
            Some(Cv::NativeUInt(0x8000_0000)),
            "a native integer rotates at the pointer's width"
        );
        assert_eq!(Cv::U64(5).rotate(&Cv::U64(0), false, ptr), Some(Cv::U64(5)));
        assert_eq!(Cv::from(1.0f64).rotate(&Cv::U64(1), false, ptr), None);
        assert_eq!(Cv::U64(1).rotate(&Cv::from(1.0f64), false, ptr), None);
    }

    /// Signed division reads both operands as signed integers of their shared
    /// width, whatever the variants, and refuses the one quotient that
    /// overflows the width.
    #[test]
    fn a_signed_division_reads_its_operands_signed_at_their_width() {
        let ptr = PointerSize::Bit64;
        assert_eq!(
            Cv::U32(0xFFFF_FFF8).div(&Cv::U32(3), ptr),
            Some(Cv::U32(0xFFFF_FFFE)),
            "-8 / 3 is -2"
        );
        assert_eq!(
            Cv::U32(0xFFFF_FFF8).rem(&Cv::U32(3), ptr),
            Some(Cv::U32(0xFFFF_FFFE)),
            "-8 % 3 is -2"
        );
        assert_eq!(Cv::I32(-8).div(&Cv::U32(2), ptr), Some(Cv::I32(-4)));
        assert_eq!(Cv::I64(7).div(&Cv::I64(-2), ptr), Some(Cv::I64(-3)));
        assert_eq!(Cv::I8(i8::MIN).div(&Cv::I8(-1), ptr), None);
        assert_eq!(Cv::U8(0x80).div(&Cv::U8(0xFF), ptr), None);
        assert_eq!(Cv::I8(i8::MIN).rem(&Cv::I8(-1), ptr), None);
        assert_eq!(Cv::I64(i64::MIN).div(&Cv::I64(-1), ptr), None);
        assert_eq!(Cv::I32(5).div(&Cv::I32(0), ptr), None);
        assert_eq!(Cv::I32(5).div(&Cv::I64(1), ptr), None, "mixed widths");
        assert_eq!(
            Cv::NativeUInt(0xFFFF_FFFE).div(&Cv::NativeUInt(2), PointerSize::Bit32),
            Some(Cv::NativeUInt(0xFFFF_FFFF)),
            "a native integer is divided at the pointer width"
        );
        assert_eq!(
            Cv::from(1.0f64).div(&Cv::from(0.0f64), ptr),
            Some(Cv::from(f64::INFINITY))
        );
    }

    /// **A float of every format folds through one path, and a NaN result is
    /// not folded.** Its payload is the target's: x86 makes `0.0 / 0.0` the
    /// negative quiet NaN, RISC-V and AArch64 the positive one, so no fold of
    /// it is right everywhere. Each format's zero is the all-zero pattern, and
    /// `1 / 0`, an infinity, folds in each.
    #[test]
    fn a_nan_result_of_any_format_is_not_folded() {
        let ptr = PointerSize::Bit64;
        for format in [
            FloatFormat::Half,
            FloatFormat::BFloat16,
            FloatFormat::Single,
            FloatFormat::Double,
            FloatFormat::X87Extended,
            FloatFormat::Quad,
        ] {
            let zero = Cv::float(format, 0).expect("a zero");
            assert_eq!(zero.div(&zero, ptr), None, "{format:?} 0/0");
            assert_eq!(zero.rem(&zero, ptr), None, "{format:?} 0 rem 0");
            let one = Cv::float(format, format.from_integer(1)).expect("a one");
            let infinity = one.div(&zero, ptr).expect("1/0 folds");
            assert!(!infinity.is_nan(), "{format:?} 1/0 is an infinity");
        }
        let nan = Cv::from(f64::NAN);
        assert_eq!(nan.add(&Cv::from(1.0f64), ptr), None, "a NaN operand");
        assert_eq!(Cv::from(f32::NAN).mul(&Cv::from(2.0f32), ptr), None);
        assert_eq!(
            Cv::from(f64::INFINITY).sub(&Cv::from(f64::INFINITY), ptr),
            None,
            "inf - inf"
        );
    }

    /// **binary32 and binary64 compute as IEEE's.** The oracle is exact
    /// arithmetic: `1.5 + 2.25` is representable, `0.1f32 * 3` is not and
    /// rounds by the mode the program sets, which the mock target leaves
    /// unknown, so it does not fold; `fmod` keeps the dividend's sign, and the
    /// narrow operands of different formats do not combine.
    #[test]
    fn binary32_and_binary64_compute_as_ieee() {
        let ptr = PointerSize::Bit64;
        assert_eq!(
            Cv::from(1.5f64).add(&Cv::from(2.25f64), ptr),
            Some(Cv::from(3.75f64))
        );
        assert_eq!(Cv::from(0.1f32).mul(&Cv::from(3.0f32), ptr), None);
        assert_eq!(
            Cv::from(-7.5f64).rem(&Cv::from(2.0f64), ptr),
            Some(Cv::from(-1.5f64))
        );
        assert_eq!(Cv::from(1.0f32).add(&Cv::from(1.0f64), ptr), None);
        assert_eq!(Cv::from(1.0f64).as_f64(), Some(1.0));
        assert_eq!(Cv::from(1.0f64).as_f32(), None);
        assert_eq!(Cv::from(1.25f32).as_f32(), Some(1.25));
        assert_eq!(Cv::from(1.5f32).to_string(), "1.5f");
        assert_eq!(Cv::from(1.5f64).to_string(), "1.5");
        assert_eq!(Cv::from(1e300f64).to_string(), format!("{}", 1e300f64));
        assert_eq!(
            format!("{:?}", Cv::from(1.5f64)),
            "Float(Double(0x3ff8000000000000))"
        );
    }

    /// **A 128-bit integer orders as one.** Held as one 128-bit lane, it
    /// compares signed and unsigned by its value at that width, beside another
    /// such integer or a narrower scalar lane the reading extends. Vectors of
    /// more lanes have no scalar order.
    #[test]
    fn a_128_bit_integer_orders_as_one() {
        let wide = |bits: u128| Cv::of_lane(128, bits).unwrap();
        let ptr = PointerSize::Bit64;
        let (minus_one, one, top) = (wide(u128::MAX), wide(1), wide(1 << 127));
        assert_eq!(minus_one.clt(&one, ptr), Some(Cv::True), "-1 < 1 signed");
        assert_eq!(
            minus_one.clt_un(&one, ptr),
            Some(Cv::False),
            "2^128-1 > 1 unsigned"
        );
        assert_eq!(
            top.clt(&minus_one, ptr),
            Some(Cv::True),
            "the minimum is least"
        );
        assert_eq!(top.cgt_un(&one, ptr), Some(Cv::True));
        assert_eq!(one.cgt(&wide(0), ptr), Some(Cv::True));
        assert_eq!(wide(1 << 100).cgt(&Cv::I64(-1), ptr), Some(Cv::True));
        assert_eq!(wide(1 << 100).clt_un(&Cv::I64(-1), ptr), Some(Cv::False));
        assert_eq!(Cv::I64(-1).clt(&wide(0), ptr), Some(Cv::True));
        assert_eq!(
            Cv::U64(u64::MAX).clt_un(&wide(1 << 64), ptr),
            Some(Cv::True)
        );
        let pair = Cv::Vector(VectorConst::from_lanes(64, [1, 2]).unwrap());
        assert_eq!(pair.clt(&pair, ptr), None);
        assert_eq!(
            wide(1).clt(&Cv::of_lane(96, 2).unwrap(), ptr),
            Some(Cv::True),
            "1 < 2 across scalar widths"
        );
    }

    /// Every eight-bit input pair is checked against a wider mathematical oracle.
    #[test]
    fn checked_integer_arithmetic_exhausts_eight_bit_patterns_and_signedness() {
        for left in 0..=255u16 {
            for right in 0..=255u16 {
                for unsigned in [false, true] {
                    let (a, b, min, max) = if unsigned {
                        (i32::from(left), i32::from(right), 0, 255)
                    } else {
                        (
                            i32::from(left as u8 as i8),
                            i32::from(right as u8 as i8),
                            -128,
                            127,
                        )
                    };
                    for left_signed in [false, true] {
                        for right_signed in [false, true] {
                            let l = if left_signed {
                                Cv::I8(left as u8 as i8)
                            } else {
                                Cv::U8(left as u8)
                            };
                            let r = if right_signed {
                                Cv::I8(right as u8 as i8)
                            } else {
                                Cv::U8(right as u8)
                            };
                            for (actual, mathematical) in [
                                (l.add_checked(&r, unsigned, PointerSize::Bit64), a + b),
                                (l.sub_checked(&r, unsigned, PointerSize::Bit64), a - b),
                                (l.mul_checked(&r, unsigned, PointerSize::Bit64), a * b),
                            ] {
                                let expected = (min..=max).contains(&mathematical).then_some({
                                    if left_signed {
                                        Cv::I8(mathematical as u8 as i8)
                                    } else {
                                        Cv::U8(mathematical as u8)
                                    }
                                });
                                assert_eq!(
                                    actual, expected,
                                    "left={left} right={right} unsigned={unsigned} left_signed={left_signed} right_signed={right_signed}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Native overflow uses the target width, including sign extension into 128 bits.
    #[test]
    fn native_checked_arithmetic_obeys_every_target_width() {
        for width in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
        ] {
            let bits = width.bits();
            let maximum = if bits == 64 {
                i64::MAX
            } else {
                (1i64 << (bits - 1)) - 1
            };
            let minimum = if bits == 64 {
                i64::MIN
            } else {
                -1i64 << (bits - 1)
            };
            let unsigned_max = width.mask_unsigned(u64::MAX);
            assert_eq!(
                Cv::NativeInt(maximum).add_checked(&Cv::NativeInt(1), false, width),
                None
            );
            assert_eq!(
                Cv::NativeInt(minimum).sub_checked(&Cv::NativeInt(1), false, width),
                None
            );
            assert_eq!(
                Cv::NativeInt(minimum).mul_checked(&Cv::NativeInt(-1), false, width),
                None
            );
            assert_eq!(
                Cv::NativeUInt(unsigned_max).add_checked(&Cv::NativeUInt(1), true, width),
                None
            );
            assert_eq!(
                Cv::NativeUInt(0).sub_checked(&Cv::NativeUInt(1), true, width),
                None
            );
            assert_eq!(
                Cv::NativeUInt(unsigned_max).mul_checked(&Cv::NativeUInt(2), true, width),
                None
            );
            assert_eq!(
                Cv::NativeInt(maximum - 1).add_checked(&Cv::NativeInt(1), false, width),
                Some(Cv::NativeInt(maximum))
            );
            assert_eq!(
                Cv::NativeInt(-2).mul_checked(&Cv::NativeInt(2), false, width),
                Some(Cv::NativeInt(-4))
            );
        }
        let wide = |bits| Cv::of_lane(128, bits).expect("wide scalar integer");
        let max = wide(i128::MAX.cast_unsigned());
        let min = wide(i128::MIN.cast_unsigned());
        assert_eq!(max.add_checked(&wide(1), false, PointerSize::Bit128), None);
        assert_eq!(min.sub_checked(&wide(1), false, PointerSize::Bit128), None);
        assert_eq!(
            min.mul_checked(&wide(u128::MAX), false, PointerSize::Bit128),
            None
        );
        assert_eq!(
            wide(u128::MAX).add_checked(&wide(1), true, PointerSize::Bit128),
            None
        );
        assert_eq!(
            wide(u128::MAX).sub_checked(&wide(1), true, PointerSize::Bit128),
            Some(wide(u128::MAX - 1))
        );
        assert_eq!(
            Cv::NativeInt(-2).mul_checked(&Cv::NativeInt(2), false, PointerSize::Bit128),
            Some(wide((-4i128).cast_unsigned()))
        );
        assert_eq!(
            Cv::NativeUInt(u64::MAX).add_checked(&Cv::NativeUInt(1), true, PointerSize::Bit128),
            Some(wide(1u128 << 64))
        );
        assert_eq!(
            Cv::NativeInt(-1).scalar_integer_pattern(PointerSize::Bit128),
            Some((u128::MAX, 128))
        );
    }

    /// Unsupported operand shapes do not masquerade as overflow.
    #[test]
    fn checked_integer_layouts_reject_non_scalars_and_unequal_widths() {
        let vector = Cv::Vector(VectorConst::from_lanes(8, [1, 2]).expect("two lanes"));
        let too_wide = Cv::Vector(VectorConst::zeroed(256, 1).expect("noncomputing wide lane"));
        for value in [
            Cv::True,
            Cv::Null,
            Cv::float(FloatFormat::Single, 0).expect("float"),
            vector,
            too_wide,
        ] {
            assert_eq!(value.scalar_integer_pattern(PointerSize::Bit64), None);
            assert_eq!(
                value.add_checked(&Cv::U8(1), true, PointerSize::Bit64),
                None
            );
        }
        assert_eq!(
            Cv::I16(1).mul_checked(&Cv::I8(1), false, PointerSize::Bit64),
            None
        );
    }

    /// Scalar width normalization preserves bit patterns and validates its boundaries.
    #[test]
    fn scalar_carriers_normalize_at_declared_widths_without_host_truncation() {
        let wide = |width, value| Cv::of_lane(width, value).expect("scalar bitvector");
        assert_eq!(
            Cv::I8(-1).scalar_integer_at_width(128, PointerSize::Bit64),
            Some(wide(128, u128::MAX))
        );
        assert_eq!(
            Cv::U8(255).scalar_integer_at_width(128, PointerSize::Bit64),
            Some(wide(128, 255))
        );
        assert_eq!(
            Cv::I64(-1).scalar_integer_at_width(1, PointerSize::Bit64),
            Some(wide(1, 1))
        );
        assert_eq!(
            wide(128, u128::MAX).scalar_integer_at_width(5, PointerSize::Bit64),
            Some(wide(5, 31))
        );
        assert_eq!(
            Cv::NativeInt(255).scalar_integer_at_width(8, PointerSize::Bit8),
            Some(Cv::NativeInt(-1))
        );
        assert_eq!(
            Cv::U64(0x1_ffff_ffff).scalar_integer_at_width(32, PointerSize::Bit64),
            Some(Cv::U32(u32::MAX))
        );
        for width in [0, 129, u32::MAX] {
            assert_eq!(
                Cv::U8(1).scalar_integer_at_width(width, PointerSize::Bit64),
                None
            );
        }
        assert_eq!(
            Cv::True.scalar_integer_at_width(32, PointerSize::Bit64),
            None
        );
    }

    /// Numeric pointer sums agree with a wider integer oracle for every byte pair.
    #[test]
    fn pointer_sums_wrap_at_the_declared_width() {
        for base in 0..=255u16 {
            for index in 0..=255u16 {
                for (stride, offset) in [(3, -17), (255, 127)] {
                    let expected = (i64::from(base) + i64::from(index) * stride + offset)
                        .rem_euclid(256) as u8;
                    assert_eq!(
                        Cv::U16(base).pointer_add(
                            Some(&Cv::U16(index)),
                            stride as u64,
                            offset,
                            PointerSize::Bit8
                        ),
                        Some(Cv::U8(expected))
                    );
                }
            }
        }
        let wide = |raw| Cv::of_lane(128, raw).unwrap();
        assert_eq!(
            Cv::U64(8).pointer_add(
                Some(&wide(i128::MAX.cast_unsigned())),
                2,
                0,
                PointerSize::Bit128
            ),
            Some(wide(6))
        );
        assert_eq!(
            Cv::U64(8).pointer_add(Some(&wide(u128::MAX)), 1, 1, PointerSize::Bit128),
            Some(wide(8))
        );
        assert_eq!(
            wide(u128::MAX).pointer_add(None, 0, 1, PointerSize::Bit128),
            Some(wide(0))
        );
        assert_eq!(
            Cv::I8(-1).pointer_add(None, 0, 0, PointerSize::Bit128),
            Some(wide(u128::MAX))
        );
        assert_eq!(
            Cv::U8(255).pointer_add(None, 0, 0, PointerSize::Bit128),
            Some(wide(255))
        );
        assert_eq!(Cv::True.pointer_add(None, 0, 0, PointerSize::Bit64), None);
        assert_eq!(
            Cv::U64(8).pointer_add(Some(&Cv::False), 1, 0, PointerSize::Bit64),
            None
        );
    }

    /// Byte encodings use exact lane-local order, including wide and packed lanes.
    #[test]
    fn borrowed_encodings_cover_packed_and_wide_lanes() {
        for (lane_bits, lanes) in [
            (1, 24),
            (2, 12),
            (3, 8),
            (4, 6),
            (6, 4),
            (7, 8),
            (8, 32),
            (16, 16),
            (24, 8),
            (64, 8),
            (128, 4),
            (256, 4),
            (512, 2),
            (1024, 2),
            (2048, 1),
        ] {
            let count = usize::try_from(lane_bits * lanes / 8).unwrap();
            let raw: Vec<u8> = (0..count)
                .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
                .collect();
            let vector = VectorConst::from_lanes(8, raw.iter().copied().map(u128::from))
                .unwrap()
                .relaned(lane_bits)
                .unwrap();
            let value = Cv::Vector(vector);
            for order in [Endianness::Little, Endianness::Big] {
                let mut expected = raw.clone();
                if order == Endianness::Big && lane_bits >= 8 {
                    for lane in expected.chunks_mut(usize::try_from(lane_bits / 8).unwrap()) {
                        lane.reverse();
                    }
                }
                let mut bytes = EncodedBytes::of(&value, order, PointerSize::Bit64).unwrap();
                assert_eq!(bytes.len(), count);
                assert_eq!(bytes.size_hint(), (count, Some(count)));
                assert_eq!(bytes.next(), expected.first().copied());
                assert_eq!(bytes.len(), count - 1);
                assert_eq!(bytes.collect::<Vec<_>>(), expected[1..]);
                assert_eq!(value.to_bytes(order, PointerSize::Bit64), Some(expected));
            }
        }
        for (lane_bits, lanes) in [(3, 3), (12, 2)] {
            let value = Cv::Vector(VectorConst::zeroed(lane_bits, lanes).unwrap());
            assert!(EncodedBytes::of(&value, Endianness::Little, PointerSize::Bit64).is_none());
            assert_eq!(value.to_bytes(Endianness::Big, PointerSize::Bit64), None);
        }
    }
}
