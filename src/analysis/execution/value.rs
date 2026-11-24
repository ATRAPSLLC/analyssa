//! What static execution knows about one value.
//!
//! A value is a constant, an address, partial scalar integer bits, or unknown.
//! Partial bits retain their declared width and possible stack provenance;
//! signedness comes from the register layout when a result becomes concrete.
//! An address is a [`Root`] plus
//! a byte offset, and the root says which space the offset is measured in: two
//! addresses under one root are equal exactly when their offsets are, and two
//! under different roots are compared by what the roots may share.
//!
//! A value the execution cannot compute but can still *name* - a parameter, a
//! register live on entry, what an unmodelled call returns - is an address
//! under an [`Root::Opaque`] root of its own. It may be an integer or a
//! pointer; either way `p + 1` stays one byte past `p`, and a store through `p`
//! lands somewhere the execution can name even though it cannot say where that
//! is. [`Value::Unknown`] is what is left when even that identity is lost, such
//! as the join of two different values.

use std::{cmp::Ordering, ops::RangeInclusive};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    analysis::execution::integer::PartialInteger,
    ir::{
        integer::{BitWidth, IntegerBits},
        value::ConstValue,
        variable::SsaVarId,
    },
    pointer::PointerSize,
    target::Target,
};

/// Where an address is measured from.
///
/// The derived order groups each kind of root together, so the bytes under one
/// kind form one contiguous range of an ordered map keyed by `(Root, offset)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Root {
    /// The host's symbol space: an address in the image.
    Symbol,
    /// The program's own integer address space, where the host cannot relate
    /// it to its symbols.
    Number,
    /// The stack. Offset zero is the stack pointer the outermost activation
    /// was entered with, so a callee executed in place addresses its frame and
    /// its stack-passed arguments in the same space as its caller.
    Stack,
    /// A callee-local stack object when the host supplies no shared-stack mapping.
    StackFrame(u32),
    /// One allocation a call model made, by its number. Nothing made before it
    /// can point into it.
    Heap(u32),
    /// Unknown bits derived from a stack address, named by their definition.
    ///
    /// Offsets under this identity remain related, for example after stack
    /// alignment. Unlike an ordinary opaque root, it may alias private stack
    /// storage as well as other address spaces.
    StackDerived {
        /// Activation that computed the value.
        activation: u32,
        /// SSA definition naming its bits.
        value: SsaVarId,
        /// Execution of that definition.
        run: u32,
    },
    /// A stack pointer a mask realigned where nothing states its low bits:
    /// `and sp, -2^bits` of the address `at` under the outermost stack
    /// (`frame` `None`) or a callee-local frame ([`Self::StackFrame`]).
    ///
    /// Clearing the low `bits` bits lowers an address by less than `2^bits`,
    /// so offset zero lies in that frame's window `(at - 2^bits, at]`
    /// ([`Self::realigned_window`]): accesses under this identity are exact
    /// among themselves, may alias that window of the frame, and are private
    /// stack storage like the frame's own. Its base is aligned to `2^bits`.
    /// The same mask of the same address is the same physical value, so no
    /// execution names it.
    StackAligned {
        /// The callee-local frame the masked address lay in, `None` for the
        /// outermost stack.
        frame: Option<u32>,
        /// The masked address's offset in that frame.
        at: i64,
        /// How many low bits the mask clears, from 1 below the pointer width.
        bits: u8,
    },
    /// A value the execution names but cannot compute.
    Opaque {
        /// The activation it was produced in.
        activation: u32,
        /// The SSA value.
        value: SsaVarId,
        /// Which run of its definition: a loop redefines a value, and each run
        /// may be a different address. Zero for a value no instruction in the
        /// function defines.
        run: u32,
    },
}

impl Root {
    /// Returns whether this identity may carry private stack-pointer bits.
    #[must_use]
    pub const fn may_reference_stack(self) -> bool {
        matches!(
            self,
            Self::Stack
                | Self::StackFrame(_)
                | Self::StackDerived { .. }
                | Self::StackAligned { .. }
        )
    }

    /// The frame a realigned stack pointer ([`Self::StackAligned`]) lies in
    /// and the offsets of it its base may be, `(at - 2^bits, at]`; `None` for
    /// any other root.
    #[must_use]
    pub fn realigned_window(self) -> Option<(Self, RangeInclusive<i64>)> {
        let Self::StackAligned { frame, at, bits } = self else {
            return None;
        };
        let below = 1i64.checked_shl(u32::from(bits))?.checked_sub(1)?;
        let frame = frame.map_or(Self::Stack, Self::StackFrame);
        Some((frame, at.checked_sub(below)?..=at))
    }

    /// Whether this identity is the stack's own storage: the outermost stack,
    /// a callee-local frame, or a frame realigned from either. A value merely
    /// derived from a stack address ([`Self::StackDerived`]) may name any
    /// storage and is not.
    #[must_use]
    pub const fn is_stack_storage(self) -> bool {
        matches!(
            self,
            Self::Stack | Self::StackFrame(_) | Self::StackAligned { .. }
        )
    }

    /// Returns whether offsets name unsigned coordinates rather than signed displacements.
    pub const fn unsigned_offsets(self) -> bool {
        matches!(self, Self::Symbol | Self::Number | Self::Heap(_))
    }

    /// Whether an offset under this root lies in a flat space the pointer width
    /// wraps.
    #[must_use]
    pub const fn is_absolute(self) -> bool {
        matches!(self, Self::Symbol | Self::Number)
    }
}

/// An address: a root plus a byte offset from it.
///
/// Ordering groups roots, then orders their unsigned coordinates or signed
/// displacements. The target width breaks ties between otherwise equal addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Address {
    /// What the offset is measured from.
    root: Root,
    /// The offset, in bytes.
    offset: i64,
    /// Target width governing arithmetic under every root.
    coordinate_width: PointerSize,
}

impl Ord for Address {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.root, self.linear_coordinate(), self.coordinate_width).cmp(&(
            other.root,
            other.linear_coordinate(),
            other.coordinate_width,
        ))
    }
}

impl PartialOrd for Address {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Address {
    /// Returns `root + offset`, normalized to the target pointer width.
    ///
    /// Stack and opaque offsets use signed coordinates; image, integer and heap
    /// coordinates use unsigned bit patterns. All roots obey target-width arithmetic.
    ///
    /// # Arguments
    ///
    /// * `root` - What the offset is measured from.
    /// * `offset` - The offset, in bytes.
    /// * `ptr_size` - The width every address wraps at.
    #[must_use]
    pub fn new(root: Root, offset: i64, ptr_size: PointerSize) -> Self {
        let offset = match root {
            Root::Stack
            | Root::StackFrame(_)
            | Root::StackDerived { .. }
            | Root::StackAligned { .. }
            | Root::Opaque { .. } => ptr_size.mask_signed(offset),
            Root::Symbol | Root::Number | Root::Heap(_) => {
                ptr_size.mask_unsigned(offset.cast_unsigned()).cast_signed()
            }
        };
        Self {
            root,
            offset,
            coordinate_width: ptr_size,
        }
    }

    /// Returns the address of `root` itself in a 64-bit coordinate space.
    ///
    /// Use [`Self::new`] to specify another target width.
    ///
    /// # Arguments
    ///
    /// * `root` - What the address is.
    #[must_use]
    pub const fn of_root(root: Root) -> Self {
        Self {
            root,
            offset: 0,
            coordinate_width: PointerSize::Bit64,
        }
    }

    /// Returns the root.
    #[must_use]
    pub const fn root(self) -> Root {
        self.root
    }

    /// Returns the offset from the root, in bytes.
    #[must_use]
    pub const fn offset(self) -> i64 {
        self.offset
    }

    /// Returns the target pointer width governing this address.
    #[must_use]
    pub const fn pointer_size(self) -> PointerSize {
        self.coordinate_width
    }

    /// Returns how many bytes, up to `limit`, fit before linear coordinates wrap.
    ///
    /// Publication uses linear ranges even when execution advances modularly.
    /// A 128-bit target is additionally bounded by this domain's representable
    /// unsigned 64-bit coordinates or signed 64-bit displacements.
    #[must_use]
    pub fn linear_prefix_len(self, limit: usize) -> usize {
        if Self::new(self.root, self.offset, self.coordinate_width) != self {
            return 0;
        }
        let shift = 64u32.saturating_sub(self.coordinate_width.bits());
        let last = if self.root.unsigned_offsets() {
            i128::from(BitWidth::new(self.coordinate_width.bits()).mask())
        } else {
            i128::from(i64::MAX >> shift)
        };
        let remaining = last
            .saturating_sub(self.linear_coordinate())
            .saturating_add(1);
        usize::try_from(remaining).unwrap_or(usize::MAX).min(limit)
    }

    /// Returns the coordinate used to order linear byte ranges.
    fn linear_coordinate(self) -> i128 {
        if self.root.unsigned_offsets() {
            i128::from(self.offset.cast_unsigned())
        } else {
            i128::from(self.offset)
        }
    }

    /// Returns whether this byte immediately follows a linear run at `start`.
    pub fn follows(self, start: Self, len: usize) -> bool {
        self.root == start.root
            && self.coordinate_width == start.coordinate_width
            && i128::try_from(len)
                .ok()
                .and_then(|len| start.linear_coordinate().checked_add(len))
                == Some(self.linear_coordinate())
    }

    /// Returns a byte's coordinate, preserving the target address modulus.
    pub fn offset_at(self, index: usize) -> Option<i64> {
        self.displaced(i128::try_from(index).ok()?, self.coordinate_width)
            .map(Self::offset)
    }

    /// Returns whether a span covers an entire finite target address space.
    pub fn covers_space(self, len: usize) -> bool {
        self.coordinate_width.bits() < usize::BITS
            && 1usize
                .checked_shl(self.coordinate_width.bits())
                .is_some_and(|space| len >= space)
    }

    /// Returns this address moved by `delta` bytes at `ptr_size` width.
    ///
    /// Returns `None` when a 128-bit result exceeds the address domain's
    /// representable coordinates: unsigned 64-bit image, integer and heap
    /// offsets, or signed 64-bit stack and opaque offsets. Such a result must
    /// not wrap into a representable address at a narrower width.
    #[must_use]
    pub fn moved(self, delta: i64, ptr_size: PointerSize) -> Option<Self> {
        self.displaced(i128::from(delta), ptr_size)
    }

    /// Applies a signed delta at the target width before coordinate admission.
    pub fn displaced(self, delta: i128, ptr_size: PointerSize) -> Option<Self> {
        self.displaced_bits(delta.cast_unsigned(), ptr_size)
    }

    /// Applies a full-width modular delta before checking the representable address domain.
    pub fn displaced_bits(self, delta: u128, ptr_size: PointerSize) -> Option<Self> {
        if ptr_size.bits() <= 64 {
            return Some(Self::new(
                self.root,
                self.offset.wrapping_add(delta as i64),
                ptr_size,
            ));
        }
        let sum = self.linear_coordinate().cast_unsigned().wrapping_add(delta);
        let offset = if self.root.unsigned_offsets() {
            u64::try_from(sum).ok()?.cast_signed()
        } else {
            i64::try_from(sum.cast_signed()).ok()?
        };
        Some(Self::new(self.root, offset, ptr_size))
    }

    /// The address `self & mask` is where nothing states the stack pointer's
    /// low bits: the base of the realigned frame ([`Root::StackAligned`]) when
    /// `self` lies in the outermost stack or a callee-local frame and `mask`
    /// clears exactly its low bits, at least one and fewer than the pointer
    /// width. `None` for any other address or mask.
    #[must_use]
    pub fn realigned(self, mask: u128) -> Option<Self> {
        let frame = match self.root {
            Root::Stack => None,
            Root::StackFrame(frame) => Some(frame),
            _ => return None,
        };
        let target = self.coordinate_width.mask_unsigned_128(u128::MAX);
        if mask == 0 || mask & !target != 0 {
            return None;
        }
        let cleared = target ^ mask;
        let modulus = cleared.checked_add(1)?;
        if cleared == 0 || !modulus.is_power_of_two() {
            return None;
        }
        let bits = u8::try_from(modulus.trailing_zeros()).ok()?;
        Some(Self::new(
            Root::StackAligned {
                frame,
                at: self.offset,
                bits,
            },
            0,
            self.coordinate_width,
        ))
    }

    /// Returns the address in the host's symbol space, when it is one.
    #[must_use]
    pub const fn image_address(self) -> Option<u64> {
        match self.root {
            Root::Symbol => Some(self.offset.cast_unsigned()),
            Root::Number
            | Root::Stack
            | Root::StackFrame(_)
            | Root::Heap(_)
            | Root::Opaque { .. }
            | Root::StackDerived { .. }
            | Root::StackAligned { .. } => None,
        }
    }

    /// Returns how far `self` lies past `base`, when both share a root.
    ///
    /// Returns `None` for different roots or target widths, and when a
    /// target-width signed distance does not fit a 64-bit displacement.
    ///
    /// # Arguments
    ///
    /// * `base` - The address measured from.
    #[must_use]
    pub fn distance_from(self, base: Self) -> Option<i64> {
        let difference = self.distance_bits_from(base)?;
        i64::try_from(
            self.coordinate_width
                .mask_signed_128(difference.cast_signed()),
        )
        .ok()
    }

    /// Returns a root-relative distance as a full target-width bit pattern.
    pub fn distance_bits_from(self, base: Self) -> Option<u128> {
        if self.root != base.root || self.coordinate_width != base.coordinate_width {
            return None;
        }
        Some(
            self.coordinate_width.mask_unsigned_128(
                self.linear_coordinate()
                    .cast_unsigned()
                    .wrapping_sub(base.linear_coordinate().cast_unsigned()),
            ),
        )
    }

    /// Whether `address` can never be zero.
    #[must_use]
    pub const fn never_null(self) -> bool {
        matches!(self.root(), Root::Heap(_)) && self.offset() == 0
    }

    /// Whether `address` is a value this execution only names: an input,
    /// or what an operation it could not compute left, stack-derived or not.
    #[must_use]
    pub const fn is_named(self) -> bool {
        matches!(self.root(), Root::Opaque { .. } | Root::StackDerived { .. })
    }
}

/// What the execution knows about one value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value<T: Target> {
    /// A constant.
    Const(ConstValue<T>),
    /// An address - which is also what a value the execution names but cannot
    /// compute is, rooted at itself.
    Addr(Address),
    /// Unknown bits derived from a stack address.
    ///
    /// Publication or use as a pointer may expose private stack storage even
    /// when arithmetic lost its exact offset or pointer-width conversion.
    StackDerived,
    /// Partial scalar integer facts, preserving width and possible stack provenance.
    ///
    /// The register layout supplies signedness when these facts become concrete.
    /// Common masks use inline windows; wider patterns share immutable storage.
    Integer(PartialInteger),
    /// Nothing, not even an identity.
    Unknown,
}

impl<T: Target> Value<T> {
    /// Constructs partial integer facts with their declared width and stack provenance.
    ///
    /// Known positions spanning at most 64 bits use an inline window. Other
    /// patterns use shared immutable storage, with no loss of known positions.
    /// Even fully known bits remain [`Self::Integer`]: creating a typed constant
    /// also requires the register's signed representation.
    ///
    /// # Arguments
    ///
    /// * `bits` - Validated zero and one masks with a width from one through 128.
    /// * `stack_derived` - Whether these bits may derive from private stack storage.
    ///   Set this when an operation loses a stack address's exact identity; known
    ///   bits alone do not prove that the value cannot expose that storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use analyssa::{analysis::execution::Value, ir::integer::IntegerBits, testing::MockTarget};
    /// let bits = IntegerBits::new(8, 0xf0, 1).unwrap();
    /// let value = Value::<MockTarget>::partial_integer(bits, false);
    /// assert_eq!(value.integer_bits(), Some(bits));
    /// assert!(value.constant().is_none());
    /// let stack_value = Value::<MockTarget>::partial_integer(bits, true);
    /// assert!(stack_value.may_reference_stack());
    /// ```
    #[must_use]
    pub fn partial_integer(bits: IntegerBits, stack_derived: bool) -> Self {
        PartialInteger::new(bits, stack_derived).map_or(Self::Unknown, Self::Integer)
    }

    /// Returns partial integer facts without allocating or resolving host types.
    ///
    /// Constants and addresses return `None`; reading their bits requires an
    /// attested numeric layout and the machine's pointer width.
    #[must_use]
    pub fn integer_bits(&self) -> Option<IntegerBits> {
        match self {
            Self::Integer(bits) => bits.bits(),
            _ => None,
        }
    }

    /// Returns owned constant payload bytes copied with this value.
    pub fn owned_bytes(&self) -> usize {
        match self {
            Self::Const(value) => value.payload_bytes(),
            _ => 0,
        }
    }

    /// Moves the constant out of this value without cloning its payload.
    #[must_use]
    pub fn into_constant(self) -> Option<ConstValue<T>> {
        match self {
            Self::Const(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the constant, when this is one.
    #[must_use]
    pub const fn constant(&self) -> Option<&ConstValue<T>> {
        match self {
            Self::Const(value) => Some(value),
            Self::Addr(_) | Self::Integer(_) | Self::StackDerived | Self::Unknown => None,
        }
    }

    /// Returns the address, when this is one.
    #[must_use]
    pub const fn address(&self) -> Option<Address> {
        match self {
            Self::Addr(address) => Some(*address),
            Self::Const(_) | Self::Integer(_) | Self::StackDerived | Self::Unknown => None,
        }
    }

    /// Returns the constant as a signed integer, when it is an integer.
    #[must_use]
    pub fn integer(&self) -> Option<i64> {
        self.constant().and_then(ConstValue::as_i64)
    }

    /// Returns the constant as an unsigned count at `ptr_size`, when it is an
    /// integer: a signed one reads as its two's-complement bits at that width.
    #[must_use]
    pub fn native_count(&self, ptr_size: PointerSize) -> Option<u128> {
        let value = self.constant()?;
        let bits = if value.is_signed() {
            value.as_i64().map(|value| value as u128)?
        } else {
            value.as_u64().map(u128::from)?
        };
        Some(ptr_size.mask_unsigned_128(bits))
    }

    /// Returns whether these bits may identify private stack storage.
    #[must_use]
    pub const fn may_reference_stack(&self) -> bool {
        matches!(self, Self::StackDerived)
            || matches!(self, Self::Integer(bits) if bits.may_reference_stack())
            || matches!(self, Self::Addr(address) if address.root().may_reference_stack())
    }

    /// Returns `true` when nothing is known.
    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// Returns facts that hold on both paths.
    ///
    /// Equal constants and addresses retain their value. Partial scalar integers
    /// retain the zero and one positions shared at the same width. Other
    /// disagreements become [`Self::StackDerived`] when either path may contain
    /// a private stack address, or [`Self::Unknown`] otherwise.
    ///
    /// # Arguments
    ///
    /// * `other` - The value on the other path.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        if let (Some(left), Some(right)) = (self.integer_bits(), other.integer_bits())
            && let Some(bits) = left.join(right)
        {
            if left == right {
                let source = if self.may_reference_stack() {
                    self
                } else {
                    other
                };
                return source.clone();
            }
            return Self::partial_integer(
                bits,
                self.may_reference_stack() || other.may_reference_stack(),
            );
        }
        if self == other {
            self.clone()
        } else if self.may_reference_stack() || other.may_reference_stack() {
            Self::StackDerived
        } else {
            Self::Unknown
        }
    }

    /// Returns whether the byte count performs no memory access.
    #[must_use]
    pub fn is_zero_count(&self) -> bool {
        self.constant().and_then(ConstValue::as_u64) == Some(0)
    }

    /// Returns a condition's truth when its bits or a successful allocation establish it.
    #[must_use]
    pub fn truth(&self) -> Option<bool> {
        match self {
            Value::Const(constant) => constant.as_bool(),
            Value::Addr(address) => address.never_null().then_some(true),
            // A bit known set makes it true; one known whole is a constant.
            Value::Integer(_) => self
                .integer_bits()
                .and_then(|bits| (bits.ones() != 0).then_some(true)),
            Value::StackDerived | Value::Unknown => None,
        }
    }

    /// The least and greatest unsigned values an integer's known bits allow,
    /// where its sign bit is known clear: an offset that cannot be negative.
    #[must_use]
    pub fn non_negative_bounds(&self) -> Option<(u64, u64)> {
        let bits = self.integer_bits()?;
        let sign = 1u128.checked_shl(bits.width().checked_sub(1)?)?;
        if bits.possibly_set() & sign != 0 {
            return None;
        }
        Some((
            u64::try_from(bits.ones()).ok()?,
            u64::try_from(bits.possibly_set()).ok()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::value::DecryptedArrayData, testing::MockTarget};

    fn owned_payloads() -> Vec<Value<MockTarget>> {
        vec![
            Value::Const(ConstValue::DecryptedString("x".repeat(4096).into())),
            Value::Const(ConstValue::DecryptedString("aé中𐍈".repeat(512).into())),
            Value::Const(ConstValue::DecryptedArray(Box::new(DecryptedArrayData {
                data: (0..4096).map(|index| index as u8).collect(),
                element_type_ref: 17,
                element_size: 4,
            }))),
        ]
    }

    /// Array metadata and every byte take part in agreement; different kinds disagree.
    #[test]
    fn payload_agreement_checks_metadata_lengths_and_tail_bytes() {
        for value in owned_payloads() {
            assert_ne!(value, Value::Unknown);
            let mut different = value.clone();
            match &mut different {
                Value::Const(ConstValue::DecryptedString(text)) => {
                    *text = format!("{text}!").into()
                }
                Value::Const(ConstValue::DecryptedArray(array)) => {
                    *array.data.last_mut().unwrap() ^= 1
                }
                _ => unreachable!(),
            }
            assert_ne!(value, different);
            assert!(value.join(&different).is_unknown());
        }
        let array =
            Value::<MockTarget>::Const(ConstValue::DecryptedArray(Box::new(DecryptedArrayData {
                data: vec![1, 2, 3],
                element_type_ref: 17,
                element_size: 1,
            })));
        for metadata in 0..2 {
            let mut changed = array.clone();
            if let Value::Const(ConstValue::DecryptedArray(value)) = &mut changed {
                if metadata == 0 {
                    value.element_type_ref = 18;
                } else {
                    value.element_size = 2;
                }
            }
            assert_ne!(array, changed);
        }
    }

    /// Every coordinate uses the target width while retaining its signedness.
    #[test]
    fn every_address_uses_the_target_width() {
        let top = Address::new(Root::Symbol, 0xffff_ffff, PointerSize::Bit32);
        assert_eq!(
            top.moved(1, PointerSize::Bit32).map(Address::offset),
            Some(0)
        );
        let stack = Address::new(Root::Stack, 0, PointerSize::Bit32);
        assert_eq!(
            stack.moved(-8, PointerSize::Bit32).map(Address::offset),
            Some(-8)
        );
    }

    /// A distance is only defined under one root.
    #[test]
    fn distance_needs_one_root() {
        let p = Address::new(Root::Stack, -40, PointerSize::Bit64);
        let q = p.moved(12, PointerSize::Bit64).unwrap_or(p);
        assert_eq!(q.distance_from(p), Some(12));
        let image = Address::new(Root::Symbol, -28, PointerSize::Bit64);
        assert_eq!(image.distance_from(p), None);
    }

    /// Two paths agree only on a value both hold.
    #[test]
    fn join_keeps_only_agreement() {
        let one = Value::<MockTarget>::Const(ConstValue::I32(1));
        let two = Value::<MockTarget>::Const(ConstValue::I32(2));
        assert_eq!(one.join(&one), one);
        assert_eq!(one.join(&two), Value::Unknown);
        assert_eq!(one.join(&Value::Unknown), Value::Unknown);
    }

    /// Small target spaces use signed stack coordinates and unsigned object offsets.
    #[test]
    fn coordinates_and_distances_keep_each_roots_target_modulus() {
        for width in [PointerSize::Bit8, PointerSize::Bit16, PointerSize::Bit32] {
            let period = 1i64 << width.bits();
            for root in [
                Root::Symbol,
                Root::Number,
                Root::Heap(1),
                Root::Stack,
                Root::StackFrame(2),
                Root::Opaque {
                    activation: 0,
                    value: SsaVarId::from_index(0),
                    run: 0,
                },
            ] {
                let zero = Address::new(root, 0, width);
                assert_eq!(zero.moved(period, width), Some(zero));
                let last = Address::new(root, period - 1, width);
                assert_eq!(last.moved(1, width), Some(zero));
                assert_eq!(last.distance_from(zero), Some(-1));
                assert_eq!(zero.distance_from(last), Some(1));
            }
        }
        assert_eq!(
            Address::new(Root::Stack, 255, PointerSize::Bit8).offset(),
            -1
        );
        assert_eq!(
            Address::new(Root::Heap(1), -1, PointerSize::Bit8).offset(),
            255
        );
        assert_eq!(
            Address::new(Root::Number, -1, PointerSize::Bit128).moved(1, PointerSize::Bit128),
            None
        );
        assert_eq!(
            Address::new(Root::Stack, i64::MAX, PointerSize::Bit128).moved(1, PointerSize::Bit128),
            None
        );
        assert_eq!(
            Address::new(Root::Number, 0, PointerSize::Bit128).moved(-1, PointerSize::Bit128),
            None
        );
    }

    /// Linear publication is bounded by each root's signedness and target domain.
    #[test]
    fn linear_address_order_and_extents_do_not_wrap() {
        for width in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            let bits = width.bits().min(64);
            let top = u64::MAX >> (64 - bits);
            for root in [Root::Symbol, Root::Number, Root::Heap(1)] {
                let first = Address::new(root, 0, width);
                let last = Address::new(root, top.cast_signed(), width);
                assert!(first < last);
                assert_eq!(last.pointer_size(), width);
                assert_eq!(last.linear_prefix_len(32), 1);
                assert!(!first.follows(last, 1));
                if bits == 64 {
                    let before = Address::new(root, i64::MAX, width);
                    let after = Address::new(root, i64::MIN, width);
                    assert!(before < after);
                    assert!(after.follows(before, 1));
                }
            }
            for root in [
                Root::Stack,
                Root::StackFrame(1),
                Root::Opaque {
                    activation: 0,
                    value: SsaVarId::from_index(0),
                    run: 0,
                },
                Root::StackDerived {
                    activation: 0,
                    value: SsaVarId::from_index(0),
                    run: 0,
                },
            ] {
                let top = ((1i128 << (bits - 1)) - 1) as i64;
                let last = Address::new(root, top, width);
                let wrapped = Address::new(root, -top - 1, width);
                assert!(wrapped < last);
                assert_eq!(last.linear_prefix_len(32), 1);
                assert!(!wrapped.follows(last, 1));
                let negative = Address::new(root, -1, width);
                let zero = Address::new(root, 0, width);
                assert!(negative < zero);
                assert!(zero.follows(negative, 1));
                assert_eq!(zero.linear_prefix_len(0), 0);
                assert!(!zero.follows(negative, usize::MAX));
            }
        }
        let start = Address::new(Root::Symbol, 0, PointerSize::Bit8);
        assert_eq!(start.linear_prefix_len(1000), 256);
        for (root, offset) in [(Root::Symbol, -1), (Root::Stack, -129)] {
            let invalid = Address {
                root,
                offset,
                coordinate_width: PointerSize::Bit8,
            };
            assert_eq!(invalid.linear_prefix_len(32), 0);
        }
        assert!(!Address::of_root(Root::Symbol).follows(start, 0));
        assert!(!Address::new(Root::Number, 0, PointerSize::Bit8).follows(start, 0));
    }

    /// Joins retain common bits and stack provenance.
    #[test]
    fn partial_integer_joins_preserve_stack_aliases() {
        let left =
            Value::<MockTarget>::partial_integer(IntegerBits::new(8, 0xf0, 1).unwrap(), true);
        let right = Value::partial_integer(IntegerBits::new(8, 0xf0, 2).unwrap(), false);
        let joined = left.join(&right);
        assert!(joined.may_reference_stack());
        assert!(
            joined
                .integer_bits()
                .is_some_and(|bits| bits.zeros() == 0xf0 && bits.ones() == 0)
        );
        /// Measures the complete constant/address/unknown domain, including its tag.
        enum IdentityValue {
            /// A concrete value with its normal carrier.
            Const(ConstValue<MockTarget>),
            /// A root-relative address and its coordinate width.
            Addr(Address),
            /// Unknown bits that may refer to a private stack.
            StackDerived,
            /// An unnamed runtime value.
            Unknown,
        }
        for value in [
            IdentityValue::Const(ConstValue::U8(0)),
            IdentityValue::Addr(Address::new(Root::Symbol, 0, PointerSize::Bit64)),
            IdentityValue::StackDerived,
            IdentityValue::Unknown,
        ] {
            match value {
                IdentityValue::Const(value) => assert_eq!(value.as_u64(), Some(0)),
                IdentityValue::Addr(address) => assert_eq!(address.root(), Root::Symbol),
                IdentityValue::StackDerived | IdentityValue::Unknown => {}
            }
        }
        assert!(
            size_of::<Value<MockTarget>>() <= size_of::<IdentityValue>(),
            "Value={} ConstValue={} Address={} IntegerBits={} value_align={} identity_domain={}",
            size_of::<Value<MockTarget>>(),
            size_of::<ConstValue<MockTarget>>(),
            size_of::<Address>(),
            size_of::<IntegerBits>(),
            align_of::<Value<MockTarget>>(),
            size_of::<IdentityValue>()
        );
    }

    /// Wide intersections retain every common bit and possible private-stack aliases.
    #[test]
    fn partial_integer_shared_joins_preserve_common_patterns() {
        let ends = 1u128 | (1u128 << 127);
        let left_bits = IntegerBits::new(128, ends, 1 << 8).unwrap();
        let right_bits = IntegerBits::new(128, ends, 1 << 9).unwrap();
        let left = Value::<MockTarget>::partial_integer(left_bits, true);
        let same = Value::partial_integer(left_bits, false);
        let joined = left.join(&same);
        assert_eq!(joined.integer_bits(), Some(left_bits));
        assert!(joined.may_reference_stack());
        let joined = left.join(&Value::partial_integer(right_bits, false));
        assert_eq!(joined.integer_bits(), IntegerBits::new(128, ends, 0));
        assert!(joined.may_reference_stack());
    }
}
