//! Comparisons of constants: equality and the signed and unsigned orderings.

use crate::{
    PointerSize,
    ir::{integer::BitWidth, ops::CmpKind, value::ConstValue},
    target::Target,
};

impl<T: Target> ConstValue<T> {
    /// Whether `self cmp other` holds of two symbols, compared as the
    /// addresses [`Target::symbol_address`] gives them; `None` unless both
    /// are symbols with addresses.
    ///
    /// Two addresses of one image move together wherever it lands: equal
    /// where their addresses are, and ordered as them unsigned, since no load
    /// base puts one past the other. Signed, a base may straddle the sign,
    /// and the order is no constant's. A symbol the target gives no address
    /// (an import the loader binds, a metadata token) is an opaque identity,
    /// compared by nothing here. `ldr r0, =__TMC_END__` twice loads one
    /// address, equal to itself, as the same number twice was.
    #[must_use]
    pub fn symbol_relation(&self, other: &Self, cmp: CmpKind, unsigned: bool) -> Option<bool> {
        let (Self::Symbol(left), Self::Symbol(right)) = (self, other) else {
            return None;
        };
        let (left, right) = (T::symbol_address(left)?, T::symbol_address(right)?);
        match cmp {
            CmpKind::Eq => Some(left == right),
            CmpKind::Ne => Some(left != right),
            _ if !unsigned => None,
            CmpKind::Lt => Some(left < right),
            CmpKind::Gt => Some(left > right),
            CmpKind::Le => Some(left <= right),
            CmpKind::Ge => Some(left >= right),
        }
    }

    /// Returns whether `self cmp other` holds, compared as the constants'
    /// own operations compare them (`ceq_at`, `clt`, `cgt` and their unsigned
    /// forms): `<=` and `>=` as the negation of `>` and `<`. `None` where the
    /// operation does not decide the relation.
    ///
    /// A float relation is the caller's to decide first where it has a
    /// policy for unordered or subnormal operands.
    #[must_use]
    pub fn relation(
        &self,
        other: &Self,
        cmp: CmpKind,
        unsigned: bool,
        ptr_size: PointerSize,
    ) -> Option<bool> {
        let holds = |value: Option<Self>| value.and_then(|value| value.as_bool());
        let less = || {
            holds(if unsigned {
                self.clt_un(other, ptr_size)
            } else {
                self.clt(other, ptr_size)
            })
        };
        let greater = || {
            holds(if unsigned {
                self.cgt_un(other, ptr_size)
            } else {
                self.cgt(other, ptr_size)
            })
        };
        match cmp {
            CmpKind::Eq => holds(self.ceq_at(other, ptr_size)),
            CmpKind::Ne => holds(self.ceq_at(other, ptr_size)).map(|equal| !equal),
            CmpKind::Lt => less(),
            CmpKind::Gt => greater(),
            CmpKind::Le => greater().map(|greater| !greater),
            CmpKind::Ge => less().map(|less| !less),
        }
    }

    /// Compares equality using 64-bit native registers.
    ///
    /// Uses [`Self::ceq_at`] for the comparison rules. Target-aware callers
    /// supply their native width to that method directly.
    #[must_use]
    pub fn ceq(&self, other: &Self) -> Option<Self> {
        self.ceq_at(other, PointerSize::Bit64)
    }

    /// Compares integer register patterns at their common width and floats by IEEE equality.
    ///
    /// Narrow primitives extend by their carrier signedness; scalar packets
    /// supply their unsigned bits. Native payloads are read at `ptr_size`.
    /// Equal-width integer patterns are independent of carrier signedness.
    /// Multi-lane vectors compare as whole bit sequences. Null and booleans
    /// compare within their own classes; unrelated classes remain unknown.
    ///
    /// # Arguments
    ///
    /// * `other` - The value to compare.
    /// * `ptr_size` - The target's native register width.
    #[must_use]
    pub fn ceq_at(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        if let (Self::Symbol(_), Self::Symbol(_)) = (self, other) {
            return self
                .symbol_relation(other, CmpKind::Eq, true)
                .map(Self::from_bool);
        }
        if let (Self::Vector(a), Self::Vector(b)) = (self, other)
            && (a.lanes() != 1 || b.lanes() != 1 || a.lane_bits() > 128 || b.lane_bits() > 128)
        {
            return Some(Self::from_bool(
                a.relaned(b.lane_bits()).as_ref() == Some(b),
            ));
        }
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(Self::from_bool(a == b)),
            (Self::I16(a), Self::I16(b)) => Some(Self::from_bool(a == b)),
            (Self::I32(a), Self::I32(b)) => Some(Self::from_bool(a == b)),
            (Self::I64(a), Self::I64(b)) => Some(Self::from_bool(a == b)),
            (Self::U8(a), Self::U8(b)) => Some(Self::from_bool(a == b)),
            (Self::U16(a), Self::U16(b)) => Some(Self::from_bool(a == b)),
            (Self::U32(a), Self::U32(b)) => Some(Self::from_bool(a == b)),
            (Self::U64(a), Self::U64(b)) => Some(Self::from_bool(a == b)),
            (Self::NativeInt(a), Self::NativeInt(b)) => Some(Self::from_bool(
                ptr_size.mask_signed(*a) == ptr_size.mask_signed(*b),
            )),
            (Self::NativeUInt(a), Self::NativeUInt(b)) => Some(Self::from_bool(
                ptr_size.mask_unsigned(*a) == ptr_size.mask_unsigned(*b),
            )),
            (Self::Float(a), Self::Float(b)) if a.format() == b.format() => Some(Self::from_bool(
                a.compare(*b) == Some(core::cmp::Ordering::Equal),
            )),
            (Self::Null, Self::Null) | (Self::True, Self::True) | (Self::False, Self::False) => {
                Some(Self::True)
            }
            (Self::True, Self::False) | (Self::False, Self::True) => Some(Self::False),
            _ => {
                let (_, left_width) = self.scalar_integer_pattern(ptr_size)?;
                let (_, right_width) = other.scalar_integer_pattern(ptr_size)?;
                let width = left_width.max(right_width);
                Some(Self::from_bool(
                    self.scalar_pattern_at_width(width, ptr_size)?
                        == other.scalar_pattern_at_width(width, ptr_size)?,
                ))
            }
        }
    }

    /// Compares two constants for less-than, signed: each integer read as a
    /// signed integer of its own width, whatever its variant, as CIL's `clt`
    /// and a native signed compare read their operands; floats ordered (false
    /// on NaN).
    ///
    /// # Arguments
    ///
    /// * `other` - The right operand.
    /// * `ptr_size` - The width of the native variants.
    ///
    /// # Returns
    ///
    /// The boolean, or `None` for constants with no such order.
    #[must_use]
    pub fn clt(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        self.order_compare(other, false, false, ptr_size)
    }

    /// Compares two constants for less-than, unsigned: each integer read as an
    /// unsigned integer of its own width (`clt.un`); floats unordered or less
    /// (true on NaN).
    ///
    /// # Arguments
    ///
    /// * `other` - The right operand.
    /// * `ptr_size` - The width of the native variants.
    ///
    /// # Returns
    ///
    /// The boolean, or `None` for constants with no such order.
    #[must_use]
    pub fn clt_un(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        self.order_compare(other, true, false, ptr_size)
    }

    /// Compares two constants for greater-than, signed; see [`Self::clt`].
    ///
    /// # Arguments
    ///
    /// * `other` - The right operand.
    /// * `ptr_size` - The width of the native variants.
    ///
    /// # Returns
    ///
    /// The boolean, or `None` for constants with no such order.
    #[must_use]
    pub fn cgt(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        self.order_compare(other, false, true, ptr_size)
    }

    /// Compares two constants for greater-than, unsigned; see
    /// [`Self::clt_un`].
    ///
    /// # Arguments
    ///
    /// * `other` - The right operand.
    /// * `ptr_size` - The width of the native variants.
    ///
    /// # Returns
    ///
    /// The boolean, or `None` for constants with no such order.
    #[must_use]
    pub fn cgt_un(&self, other: &Self, ptr_size: PointerSize) -> Option<Self> {
        self.order_compare(other, true, true, ptr_size)
    }

    /// The order comparisons' one rule: `self < other`, or `self > other` when
    /// `greater`, with the integers read at their own widths as unsigned or
    /// signed integers - a narrower one extended as the reading says - and the
    /// floats compared ordered or, `unsigned`, unordered-or-related. Scalar
    /// lanes retain their integer width; multi-lane vectors have no scalar order.
    fn order_compare(
        &self,
        other: &Self,
        unsigned: bool,
        greater: bool,
        ptr_size: PointerSize,
    ) -> Option<Self> {
        if let (Self::Symbol(_), Self::Symbol(_)) = (self, other) {
            let cmp = if greater { CmpKind::Gt } else { CmpKind::Lt };
            return self
                .symbol_relation(other, cmp, unsigned)
                .map(Self::from_bool);
        }
        if self.is_wide_native(ptr_size)
            || other.is_wide_native(ptr_size)
            || matches!((self, other), (Self::Vector(_), _) | (_, Self::Vector(_)))
        {
            let (left, right) = (
                self.wide_order_key(unsigned, ptr_size)?,
                other.wide_order_key(unsigned, ptr_size)?,
            );
            return Some(Self::from_bool(if greater {
                left > right
            } else {
                left < right
            }));
        }
        match (self, other) {
            (Self::Float(a), Self::Float(b)) if a.format() == b.format() => {
                let wanted = if greater {
                    core::cmp::Ordering::Greater
                } else {
                    core::cmp::Ordering::Less
                };
                return Some(Self::from_bool(match a.compare(*b) {
                    Some(order) => order == wanted,
                    None => unsigned,
                }));
            }
            _ => {}
        }
        if let (Some(a), Some(b)) = (
            self.integer_reading(unsigned, ptr_size),
            other.integer_reading(unsigned, ptr_size),
        ) {
            return Some(Self::from_bool(if greater { a > b } else { a < b }));
        }
        // A constant that is not an integer of a width (a boolean) compares by
        // its number.
        if unsigned {
            self.as_unsigned_pair(other)
                .map(|(a, b)| Self::from_bool(if greater { a > b } else { a < b }))
        } else {
            self.as_signed_pair(other)
                .map(|(a, b)| Self::from_bool(if greater { a > b } else { a < b }))
        }
    }

    /// Returns an order-preserving key for a complete scalar register pattern.
    ///
    /// Signed patterns extend from their own width to 128 bits, then flip the
    /// sign bit so unsigned key ordering follows signed integer ordering.
    fn wide_order_key(&self, unsigned: bool, ptr_size: PointerSize) -> Option<u128> {
        let (bits, width) = self.scalar_integer_pattern(ptr_size)?;
        if unsigned {
            return Some(bits);
        }
        let shift = 128u32.checked_sub(width)?;
        let extended = ((bits << shift).cast_signed() >> shift).cast_unsigned();
        Some(extended ^ (1u128 << 127))
    }

    /// The integer `self` holds read at its own width, as an unsigned or a
    /// signed integer; `None` for a constant that is not an integer of a width.
    fn integer_reading(&self, unsigned: bool, ptr_size: PointerSize) -> Option<i128> {
        let (bits, width) = self.integer_extended(ptr_size)?;
        let raw = BitWidth::new(width).low_bits(bits);
        Some(if unsigned {
            i128::from(raw)
        } else {
            i128::from(Self::sign_extend(raw, width))
        })
    }
}
