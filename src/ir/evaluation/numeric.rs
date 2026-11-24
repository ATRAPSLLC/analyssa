//! Integer operations shared by SSA folding and symbolic expressions.
//!
//! [`NumericOp`] names the arithmetic, bitwise, shift, rotate, bit-count and
//! comparison operations both consumers evaluate, and evaluates them on
//! borrowed constant carriers, so a symbolic expression and an SSA fold of the
//! same operation agree by construction. Carrier widths decide the semantics;
//! an operation without a defined value on its inputs evaluates to nothing.

use std::fmt;

use crate::{
    PointerSize,
    ir::value::{ConstValue, RegisterBitCount},
    target::Target,
};

/// A numeric operation shared by SSA and symbolic expression consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericOp {
    // Arithmetic operations
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Signed division.
    DivS,
    /// Unsigned division.
    DivU,
    /// Signed remainder (modulo).
    RemS,
    /// Unsigned remainder (modulo).
    RemU,
    /// Negation.
    Neg,

    // Rotate operations
    /// Rotate left.
    Rol,
    /// Rotate right.
    Ror,

    // Bit manipulation operations
    /// Byte swap.
    BSwap,
    /// Bit reverse.
    BRev,
    /// Bit scan forward (index of least significant set bit).
    BitScanForward,
    /// Bit scan reverse (index of most significant set bit).
    BitScanReverse,
    /// Population count (number of set bits).
    Popcount,
    /// Parity (1 if odd number of set bits, 0 if even).
    Parity,

    // Bitwise operations
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Bitwise NOT.
    Not,
    /// Shift left.
    Shl,
    /// Arithmetic shift right (preserves sign).
    ShrS,
    /// Logical shift right (zero-fill).
    ShrU,

    // Comparison operations (return 0 or 1)
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Signed less than.
    LtS,
    /// Unsigned less than.
    LtU,
    /// Signed greater than.
    GtS,
    /// Unsigned greater than.
    GtU,
    /// Signed less than or equal.
    LeS,
    /// Unsigned less than or equal.
    LeU,
    /// Signed greater than or equal.
    GeS,
    /// Unsigned greater than or equal.
    GeU,
}

impl NumericOp {
    /// Checks if this operation is commutative.
    ///
    /// Commutative operations produce the same result regardless of operand order:
    /// `a op b == b op a`. This property is useful for expression canonicalization.
    ///
    /// # Returns
    ///
    /// `true` if the operation is commutative (Add, Mul, And, Or, Xor, Eq, Ne).
    #[must_use]
    pub const fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::And | Self::Or | Self::Xor | Self::Eq | Self::Ne
        )
    }

    /// Checks if this operation is a comparison.
    ///
    /// Comparison operations return 0 or 1 based on the relationship between operands.
    ///
    /// # Returns
    ///
    /// `true` if this is a comparison operation (Eq, Ne, Lt*, Gt*, Le*, Ge*).
    #[must_use]
    pub const fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Eq
                | Self::Ne
                | Self::LtS
                | Self::LtU
                | Self::GtS
                | Self::GtU
                | Self::LeS
                | Self::LeU
                | Self::GeS
                | Self::GeU
        )
    }

    /// Checks if this is a unary operation.
    ///
    /// Unary operations take a single operand, unlike binary operations which take two.
    ///
    /// # Returns
    ///
    /// `true` if this is a unary operation (Neg, Not, BSwap, BRev,
    /// BitScanForward, BitScanReverse, Popcount, Parity).
    #[must_use]
    pub const fn is_unary(self) -> bool {
        matches!(
            self,
            Self::Neg
                | Self::Not
                | Self::BSwap
                | Self::BRev
                | Self::BitScanForward
                | Self::BitScanReverse
                | Self::Popcount
                | Self::Parity
        )
    }

    /// Evaluates a unary operation on a borrowed constant carrier.
    ///
    /// Carrier widths determine integer semantics. Unsupported inputs and
    /// operations without a defined value return `None`. Bit counts use the
    /// default 32-bit result carrier; SSA consumers apply their result layout.
    /// Floating operations establish no unspecified target NaN policy.
    #[inline]
    pub fn evaluate_unary<T: Target>(
        self,
        value: &ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        match self {
            Self::Neg => value.negate(ptr_size),
            Self::Not => value.bitwise_not(ptr_size),
            Self::BSwap => value.byte_swap(ptr_size),
            Self::BRev => value.bit_reverse(ptr_size),
            Self::BitScanForward | Self::BitScanReverse | Self::Popcount | Self::Parity => {
                let count = self.bit_count(value, ptr_size)?;
                if self == Self::BitScanReverse {
                    Some(ConstValue::U32(count))
                } else {
                    Some(ConstValue::I32(i32::try_from(count).ok()?))
                }
            }
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::DivS
            | Self::DivU
            | Self::RemS
            | Self::RemU
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Shl
            | Self::ShrS
            | Self::ShrU
            | Self::Rol
            | Self::Ror
            | Self::Eq
            | Self::Ne
            | Self::LtS
            | Self::LtU
            | Self::GtS
            | Self::GtU
            | Self::LeS
            | Self::LeU
            | Self::GeS
            | Self::GeU => None,
        }
    }

    /// Evaluates a binary operation on borrowed constant carriers.
    ///
    /// Widths, signedness, out-of-range shift refusal and rotate counts follow
    /// the shared constant kernels. Comparisons return booleans. No operand
    /// clone, host metadata comparison or temporary SSA operation is needed.
    /// An inclusive relation is the complement of its strict opposite,
    /// including that opposite's ordered or unordered floating convention.
    #[inline]
    pub fn evaluate_binary<T: Target>(
        self,
        left: &ConstValue<T>,
        right: &ConstValue<T>,
        ptr_size: PointerSize,
    ) -> Option<ConstValue<T>> {
        match self {
            Self::Add => left.add(right, ptr_size),
            Self::Sub => left.sub(right, ptr_size),
            Self::Mul => left.mul(right, ptr_size),
            Self::DivS => left.div(right, ptr_size),
            Self::DivU => left.div_un(right, ptr_size),
            Self::RemS => left.rem(right, ptr_size),
            Self::RemU => left.rem_un(right, ptr_size),
            Self::And => left.bitwise_and(right, ptr_size),
            Self::Or => left.bitwise_or(right, ptr_size),
            Self::Xor => left.bitwise_xor(right, ptr_size),
            Self::Shl => left.shl(right, ptr_size),
            Self::ShrS => left.shr(right, false, ptr_size),
            Self::ShrU => left.shr(right, true, ptr_size),
            Self::Rol => left.rotate(right, false, ptr_size),
            Self::Ror => left.rotate(right, true, ptr_size),
            Self::Eq => left.ceq_at(right, ptr_size),
            Self::LtS => left.clt(right, ptr_size),
            Self::LtU => left.clt_un(right, ptr_size),
            Self::GtS => left.cgt(right, ptr_size),
            Self::GtU => left.cgt_un(right, ptr_size),
            Self::Ne => Some(ConstValue::from_bool(
                !left.ceq_at(right, ptr_size)?.as_bool()?,
            )),
            Self::LeS => Some(ConstValue::from_bool(
                !left.cgt(right, ptr_size)?.as_bool()?,
            )),
            Self::LeU => Some(ConstValue::from_bool(
                !left.cgt_un(right, ptr_size)?.as_bool()?,
            )),
            Self::GeS => Some(ConstValue::from_bool(
                !left.clt(right, ptr_size)?.as_bool()?,
            )),
            Self::GeU => Some(ConstValue::from_bool(
                !left.clt_un(right, ptr_size)?.as_bool()?,
            )),
            Self::Neg
            | Self::Not
            | Self::BSwap
            | Self::BRev
            | Self::BitScanForward
            | Self::BitScanReverse
            | Self::Popcount
            | Self::Parity => None,
        }
    }

    /// Computes one register bit count without selecting the result carrier.
    pub fn bit_count<T: Target>(self, value: &ConstValue<T>, ptr_size: PointerSize) -> Option<u32> {
        let kind = match self {
            Self::BitScanForward => RegisterBitCount::First,
            Self::BitScanReverse => RegisterBitCount::Last,
            Self::Popcount => RegisterBitCount::Population,
            Self::Parity => RegisterBitCount::Parity,
            _ => return None,
        };
        value.bit_count(kind, ptr_size)
    }
}

impl fmt::Display for NumericOp {
    #[allow(clippy::match_same_arms)] // Sub and Neg are semantically different (binary vs unary)
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::DivS => write!(f, "/"),
            Self::DivU => write!(f, "/u"),
            Self::RemS => write!(f, "%"),
            Self::RemU => write!(f, "%u"),
            Self::Neg => write!(f, "-"),
            Self::Rol => write!(f, "rol"),
            Self::Ror => write!(f, "ror"),
            Self::BSwap => write!(f, "bswap"),
            Self::BRev => write!(f, "brev"),
            Self::BitScanForward => write!(f, "bsf"),
            Self::BitScanReverse => write!(f, "bsr"),
            Self::Popcount => write!(f, "popcnt"),
            Self::Parity => write!(f, "parity"),
            Self::And => write!(f, "&"),
            Self::Or => write!(f, "|"),
            Self::Xor => write!(f, "^"),
            Self::Not => write!(f, "~"),
            Self::Shl => write!(f, "<<"),
            Self::ShrS => write!(f, ">>"),
            Self::ShrU => write!(f, ">>>"),
            Self::Eq => write!(f, "=="),
            Self::Ne => write!(f, "!="),
            Self::LtS => write!(f, "<"),
            Self::LtU => write!(f, "<u"),
            Self::GtS => write!(f, ">"),
            Self::GtU => write!(f, ">u"),
            Self::LeS => write!(f, "<="),
            Self::LeU => write!(f, "<=u"),
            Self::GeS => write!(f, ">="),
            Self::GeU => write!(f, ">=u"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use crate::{
        PointerSize,
        ir::{evaluation::NumericOp, value::ConstValue},
        metadata::tests::{DefaultTarget, EqualityProbe},
    };

    /// Nonnumeric host identities are refused before unrestricted cloning or equality.
    #[test]
    fn numeric_adapters_do_not_copy_or_compare_opaque_host_metadata() {
        let calls = Rc::new(Cell::new(0));
        for value in [
            ConstValue::<DefaultTarget>::Type(EqualityProbe::new(Rc::clone(&calls))),
            ConstValue::MethodHandle(EqualityProbe::new(Rc::clone(&calls))),
            ConstValue::FieldHandle(EqualityProbe::new(Rc::clone(&calls))),
            ConstValue::Symbol(EqualityProbe::new(Rc::clone(&calls))),
        ] {
            for op in [
                NumericOp::Add,
                NumericOp::Sub,
                NumericOp::Xor,
                NumericOp::Eq,
                NumericOp::Ne,
            ] {
                assert!(
                    op.evaluate_binary(&value, &value, PointerSize::Bit64)
                        .is_none()
                );
            }
            for op in [NumericOp::Neg, NumericOp::Not, NumericOp::Popcount] {
                assert!(op.evaluate_unary(&value, PointerSize::Bit64).is_none());
            }
        }
        assert_eq!(calls.get(), 0);
    }
}
