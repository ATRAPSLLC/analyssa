//! Algebraic identity simplification for SSA operations.
//!
//! Detects algebraic identities in SSA operations, enabling replacement of
//! computations with simpler forms (constants or copy operations) before
//! full constant propagation.
//!
//! # Algorithm
//!
//! The `simplify_op` function pattern-matches on `SsaOp` variants and, for each,
//! checks whether the operands satisfy known algebraic identities:
//!
//! | Operation | Identity | Result |
//! |-----------|----------|--------|
//! | XOR       | `x ^ x`  | `0`    |
//! | XOR       | `x ^ 0`  | `x`    |
//! | OR        | `x \| x` | `x`    |
//! | OR        | `x \| 0` | `x`    |
//! | OR        | `x \| -1`| `-1`   |
//! | AND       | `x & x`  | `x`    |
//! | AND       | `x & 0`  | `0`    |
//! | AND       | `x & -1` | `x`    |
//! | ADD       | `x + 0`  | `x`    |
//! | SUB       | `x - 0`  | `x`    |
//! | SUB       | `x - x`  | `0`    |
//! | MUL       | `x * 0`  | `0`    |
//! | MUL       | `x * 1`  | `x`    |
//! | DIV       | `x / 1`  | `x`    |
//! | DIV       | `0 / x`  | `0`    |
//! | REM       | `0 % x`  | `0`    |
//! | REM       | `x % 1`  | `0`    |
//! | SHL/SHR/ROL/ROR/RCL/RCR | `x op 0` | `x`    |
//! | CEQ       | `x == x` | `1`    |
//! | CLT/CGT   | `x < x`  | `0`    |
//!
//! # Complexity
//!
//! O(1) per operation - purely pattern matching with constant lookups
//! in the provided `BTreeMap`. No recursion or iteration.
//!
//! # Usage
//!
//! ```rust
//! use analyssa::analysis::algebraic::{simplify_op, SimplifyResult};
//! use analyssa::{MockTarget, testing::MockType, ir::{ConstValue, SsaOp, SsaVarId}};
//! use std::collections::BTreeMap;
//!
//! let x = SsaVarId::from_index(0);
//! let zero = SsaVarId::from_index(1);
//! let dest = SsaVarId::from_index(2);
//! let constants = BTreeMap::from([(zero, ConstValue::<MockTarget>::I32(0))]);
//! let op = SsaOp::<MockTarget>::Add { dest, left: x, right: zero, flags: None };
//! // The operand type decides whether a self-cancelling identity is sound
//! // (`x - x` is NaN for a NaN float) and at what width its constant is built.
//! match simplify_op(&op, &constants, Some(&MockType::I32)) {
//!     SimplifyResult::Constant(value) => { /* replace with constant */ }
//!     SimplifyResult::Copy(var) => assert_eq!(var, x),
//!     SimplifyResult::None => { /* no simplification */ }
//! }
//! ```

use std::collections::BTreeMap;

use crate::{
    ir::{ops::SsaOp, value::ConstValue, variable::SsaVarId},
    target::Target,
};

/// Result of checking an operation for algebraic simplification.
#[derive(Debug, Clone, PartialEq)]
pub enum SimplifyResult<T: Target> {
    /// The operation simplifies to a constant value.
    Constant(ConstValue<T>),
    /// The operation simplifies to copying another variable.
    Copy(SsaVarId),
    /// No simplification possible.
    None,
}

impl<T: Target> SimplifyResult<T> {
    /// Returns true if a simplification is possible.
    #[must_use]
    pub fn is_some(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Returns true if no simplification is possible.
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Check if an SSA operation can be algebraically simplified.
///
/// Returns the simpler form (constant or copy) when an algebraic identity
/// applies, or `SimplifyResult::None` when no simplification is recognized.
#[must_use]
pub fn simplify_op<T: Target>(
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    operand_type: Option<&T::Type>,
) -> SimplifyResult<T> {
    // Self-cancelling identities (`x - x`, `x ^ x`, `x == x`, `x < x`) hold for
    // integers and pointers but not for floats: `x - x` is NaN when `x` is NaN or
    // infinite, and `x == x` is *false* for NaN. `Sub`/`Xor`/`Ceq`/`Clt`/`Cgt`
    // carry no float discriminator — `ConstValue` folds `F32`/`F64` pairs for all
    // of them — so the operand's declared type is the only thing that can tell
    // them apart. An unknown type is treated as possibly-float and refused.
    let self_identity_is_sound =
        operand_type.is_some_and(|ty| !T::is_floating(ty) && !T::is_unknown(ty));
    // Self-cancellation produces a zero (or one) *of the operand's type*.
    // Hard-coding `I32` redefines an `I64` or `F64` destination with a 32-bit
    // constant — `xor rax, rax` is the ubiquitous case.
    let typed_zero = |value: i64| -> Option<ConstValue<T>> {
        let ty = operand_type?;
        let width = T::bit_width(ty)?;
        match width {
            8 => Some(ConstValue::I8(i8::try_from(value).ok()?)),
            16 => Some(ConstValue::I16(i16::try_from(value).ok()?)),
            64 => Some(ConstValue::I64(value)),
            _ => Some(ConstValue::I32(i32::try_from(value).ok()?)),
        }
    };
    match op {
        // XOR: x ^ x = 0, x ^ 0 = x
        SsaOp::Xor { left, right, .. } => {
            if left == right
                && self_identity_is_sound
                && let Some(zero) = typed_zero(0)
            {
                return SimplifyResult::Constant(zero);
            }
            if constants.get(right).is_some_and(ConstValue::is_zero) {
                return SimplifyResult::Copy(*left);
            }
            if constants.get(left).is_some_and(ConstValue::is_zero) {
                return SimplifyResult::Copy(*right);
            }
            SimplifyResult::None
        }

        // OR: x | x = x, x | 0 = x, x | -1 = -1
        SsaOp::Or { left, right, .. } => {
            if left == right {
                return SimplifyResult::Copy(*left);
            }
            if let Some(c) = constants.get(right) {
                if c.is_zero() {
                    return SimplifyResult::Copy(*left);
                }
                if c.is_all_ones() {
                    return SimplifyResult::Constant(c.clone());
                }
            }
            if let Some(c) = constants.get(left) {
                if c.is_zero() {
                    return SimplifyResult::Copy(*right);
                }
                if c.is_all_ones() {
                    return SimplifyResult::Constant(c.clone());
                }
            }
            SimplifyResult::None
        }

        // AND: x & x = x, x & 0 = 0, x & -1 = x
        SsaOp::And { left, right, .. } => {
            if left == right {
                return SimplifyResult::Copy(*left);
            }
            if let Some(c) = constants.get(right) {
                if c.is_zero() {
                    return SimplifyResult::Constant(c.zero_of_same_type());
                }
                if c.is_all_ones() {
                    return SimplifyResult::Copy(*left);
                }
            }
            if let Some(c) = constants.get(left) {
                if c.is_zero() {
                    return SimplifyResult::Constant(c.zero_of_same_type());
                }
                if c.is_all_ones() {
                    return SimplifyResult::Copy(*right);
                }
            }
            SimplifyResult::None
        }

        // ADD: x + 0 = x
        SsaOp::Add { left, right, .. } => {
            if constants.get(right).is_some_and(ConstValue::is_zero) {
                return SimplifyResult::Copy(*left);
            }
            if constants.get(left).is_some_and(ConstValue::is_zero) {
                return SimplifyResult::Copy(*right);
            }
            SimplifyResult::None
        }

        // SUB: x - 0 = x, x - x = 0
        SsaOp::Sub { left, right, .. } => {
            if left == right
                && self_identity_is_sound
                && let Some(zero) = typed_zero(0)
            {
                return SimplifyResult::Constant(zero);
            }
            if constants.get(right).is_some_and(ConstValue::is_zero) {
                return SimplifyResult::Copy(*left);
            }
            SimplifyResult::None
        }

        // MUL: x * 0 = 0, x * 1 = x
        SsaOp::Mul { left, right, .. } => {
            if let Some(c) = constants.get(right) {
                if c.is_zero() {
                    return SimplifyResult::Constant(c.clone());
                }
                if c.is_one() {
                    return SimplifyResult::Copy(*left);
                }
            }
            if let Some(c) = constants.get(left) {
                if c.is_zero() {
                    return SimplifyResult::Constant(c.clone());
                }
                if c.is_one() {
                    return SimplifyResult::Copy(*right);
                }
            }
            SimplifyResult::None
        }

        // DIV: x / 1 = x, 0 / x = 0
        SsaOp::Div { left, right, .. } => {
            if constants.get(right).is_some_and(ConstValue::is_one) {
                return SimplifyResult::Copy(*left);
            }
            if let Some(c) = constants.get(left)
                && c.is_zero()
            {
                return SimplifyResult::Constant(c.clone());
            }
            SimplifyResult::None
        }

        // REM: 0 % x = 0, x % 1 = 0
        SsaOp::Rem { left, right, .. } => {
            if let Some(c) = constants.get(left)
                && c.is_zero()
            {
                return SimplifyResult::Constant(c.clone());
            }
            if let Some(c) = constants.get(right)
                && c.is_one()
            {
                return SimplifyResult::Constant(c.zero_of_same_type());
            }
            SimplifyResult::None
        }

        // SHL/SHR/ROL/ROR/RCL/RCR: shift/rotate by 0 = x
        SsaOp::Shl { value, amount, .. }
        | SsaOp::Shr { value, amount, .. }
        | SsaOp::Rol { value, amount, .. }
        | SsaOp::Ror { value, amount, .. }
        | SsaOp::Rcl { value, amount, .. }
        | SsaOp::Rcr { value, amount, .. } => {
            if constants.get(amount).is_some_and(ConstValue::is_zero) {
                return SimplifyResult::Copy(*value);
            }
            SimplifyResult::None
        }

        // Comparisons: x == x → true, x < x → false, x > x → false
        SsaOp::Ceq { left, right, .. } => {
            // A comparison result is a boolean, so its own width is I32 by
            // convention — but the *operands* decide whether the identity holds.
            if left == right && self_identity_is_sound {
                return SimplifyResult::Constant(ConstValue::I32(1));
            }
            SimplifyResult::None
        }

        SsaOp::Clt { left, right, .. } | SsaOp::Cgt { left, right, .. } => {
            if left == right && self_identity_is_sound {
                return SimplifyResult::Constant(ConstValue::I32(0));
            }
            SimplifyResult::None
        }

        _ => SimplifyResult::None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::testing::{MockTarget, MockType};

    fn make_constants(
        pairs: &[(SsaVarId, ConstValue<MockTarget>)],
    ) -> BTreeMap<SsaVarId, ConstValue<MockTarget>> {
        pairs.iter().cloned().collect()
    }

    #[test]
    fn xor_self_cancels() {
        let v1 = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Xor {
            dest,
            left: v1,
            right: v1,
            flags: None,
        };
        assert_eq!(
            simplify_op(&op, &BTreeMap::new(), Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(0))
        );
    }

    #[test]
    fn xor_zero_identity() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::Xor {
            dest,
            left: v1,
            right: v2,
            flags: None,
        };
        let constants = make_constants(&[(v2, ConstValue::I32(0))]);
        assert_eq!(
            simplify_op(&op, &constants, Some(&MockType::I32)),
            SimplifyResult::Copy(v1)
        );
    }

    #[test]
    fn mul_zero_absorbs() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::Mul {
            dest,
            left: v1,
            right: v2,
            flags: None,
        };
        let constants = make_constants(&[(v2, ConstValue::I32(0))]);
        assert_eq!(
            simplify_op(&op, &constants, Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(0))
        );
    }

    #[test]
    fn mul_one_identity() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::Mul {
            dest,
            left: v1,
            right: v2,
            flags: None,
        };
        let constants = make_constants(&[(v2, ConstValue::I32(1))]);
        assert_eq!(
            simplify_op(&op, &constants, Some(&MockType::I32)),
            SimplifyResult::Copy(v1)
        );
    }

    #[test]
    fn add_zero_identity() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest,
            left: v1,
            right: v2,
            flags: None,
        };
        let constants = make_constants(&[(v2, ConstValue::I32(0))]);
        assert_eq!(
            simplify_op(&op, &constants, Some(&MockType::I32)),
            SimplifyResult::Copy(v1)
        );
    }

    #[test]
    fn sub_self_cancels() {
        let v1 = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Sub {
            dest,
            left: v1,
            right: v1,
            flags: None,
        };
        assert_eq!(
            simplify_op(&op, &BTreeMap::new(), Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(0))
        );
    }

    #[test]
    fn and_zero_absorbs() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::And {
            dest,
            left: v1,
            right: v2,
            flags: None,
        };
        let constants = make_constants(&[(v2, ConstValue::I32(0))]);
        assert_eq!(
            simplify_op(&op, &constants, Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(0))
        );
    }

    #[test]
    fn or_zero_identity() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::Or {
            dest,
            left: v1,
            right: v2,
            flags: None,
        };
        let constants = make_constants(&[(v2, ConstValue::I32(0))]);
        assert_eq!(
            simplify_op(&op, &constants, Some(&MockType::I32)),
            SimplifyResult::Copy(v1)
        );
    }

    #[test]
    fn div_one_identity() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::Div {
            dest,
            left: v1,
            right: v2,
            unsigned: false,
            flags: None,
        };
        let constants = make_constants(&[(v2, ConstValue::I32(1))]);
        assert_eq!(
            simplify_op(&op, &constants, Some(&MockType::I32)),
            SimplifyResult::Copy(v1)
        );
    }

    #[test]
    fn shl_zero_identity() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::Shl {
            dest,
            value: v1,
            amount: v2,
            flags: None,
        };
        let constants = make_constants(&[(v2, ConstValue::I32(0))]);
        assert_eq!(
            simplify_op(&op, &constants, Some(&MockType::I32)),
            SimplifyResult::Copy(v1)
        );
    }

    #[test]
    fn no_simplification() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest,
            left: v1,
            right: v2,
            flags: None,
        };
        assert_eq!(
            simplify_op(&op, &BTreeMap::new(), Some(&MockType::I32)),
            SimplifyResult::None
        );
    }

    #[test]
    fn rem_one_zero() {
        let v1 = SsaVarId::from_index(0);
        let v2 = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let op: SsaOp<MockTarget> = SsaOp::Rem {
            dest,
            left: v1,
            right: v2,
            unsigned: false,
            flags: None,
        };
        let constants = make_constants(&[(v2, ConstValue::I32(1))]);
        assert_eq!(
            simplify_op(&op, &constants, Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(0))
        );
    }

    #[test]
    fn ceq_self_true() {
        let v1 = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Ceq {
            dest,
            left: v1,
            right: v1,
        };
        assert_eq!(
            simplify_op(&op, &BTreeMap::new(), Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(1))
        );
    }

    #[test]
    fn clt_self_false() {
        let v1 = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: v1,
            right: v1,
            unsigned: false,
        };
        assert_eq!(
            simplify_op(&op, &BTreeMap::new(), Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(0))
        );
    }

    #[test]
    fn cgt_self_false() {
        let v1 = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Cgt {
            dest,
            left: v1,
            right: v1,
            unsigned: false,
        };
        assert_eq!(
            simplify_op(&op, &BTreeMap::new(), Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(0))
        );
    }
}
