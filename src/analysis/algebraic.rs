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
//! | SELECT    | `c ? x : x` | `x` |
//!
//! A computation over the lanes of a vector that executes no element
//! (restart `>=` length) has several results; [`executing_nothing`] states
//! each.
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
    ir::{
        ops::{
            Kernel, RuntimeValueKind, SsaOp, VectorBinaryKind, VectorElementKind, VectorFaultMode,
            VectorStatus,
        },
        value::ConstValue,
        variable::SsaVarId,
    },
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
    // carry no float discriminator - `ConstValue` folds `F32`/`F64` pairs for all
    // of them - so the operand's declared type is the only thing that can tell
    // them apart. An unknown type is treated as possibly-float and refused.
    let self_identity_is_sound =
        operand_type.is_some_and(|ty| !T::is_floating(ty) && !T::is_unknown(ty));
    // Self-cancellation produces a zero (or one) at the width the operand
    // computes at ([`Target::operand_width`]): its type's own on a machine,
    // where hard-coding `I32` would redefine an `I64` destination with a
    // 32-bit constant (`xor rax, rax` is the ubiquitous case), and `int32` for
    // a CIL `byte`, which the evaluation stack holds as one.
    let typed_zero = |value: i64| -> Option<ConstValue<T>> {
        let ty = operand_type.filter(|ty| !T::is_floating(ty))?;
        let width = T::operand_width(ty)?;
        // A width no scalar integer has - a vector, a one-bit boolean - has no
        // constant here to be materialised as.
        match width {
            8 => Some(ConstValue::I8(i8::try_from(value).ok()?)),
            16 => Some(ConstValue::I16(i16::try_from(value).ok()?)),
            32 => Some(ConstValue::I32(i32::try_from(value).ok()?)),
            64 => Some(ConstValue::I64(value)),
            _ => None,
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
                if c.is_zero()
                    && let Some(zero) = typed_zero(0)
                {
                    return SimplifyResult::Constant(zero);
                }
                if c.is_all_ones() {
                    return SimplifyResult::Copy(*left);
                }
            }
            if let Some(c) = constants.get(left) {
                if c.is_zero()
                    && let Some(zero) = typed_zero(0)
                {
                    return SimplifyResult::Constant(zero);
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
                if c.is_zero()
                    && let Some(zero) = typed_zero(0)
                {
                    return SimplifyResult::Constant(zero);
                }
                if c.is_one() {
                    return SimplifyResult::Copy(*left);
                }
            }
            if let Some(c) = constants.get(left) {
                if c.is_zero()
                    && let Some(zero) = typed_zero(0)
                {
                    return SimplifyResult::Constant(zero);
                }
                if c.is_one() {
                    return SimplifyResult::Copy(*right);
                }
            }
            SimplifyResult::None
        }

        // DIV: x / 1 = x. Not `0 / x = 0`: an integer division by zero
        // raises, and a float `0.0 / 0.0` is NaN, so what `x` is decides it.
        SsaOp::Div { left, right, .. } => {
            if constants.get(right).is_some_and(ConstValue::is_one) {
                return SimplifyResult::Copy(*left);
            }
            SimplifyResult::None
        }

        // REM: x % 1 = 0. Not `0 % x = 0`, for the reason `0 / x` is not.
        SsaOp::Rem { right, .. } => {
            if let Some(c) = constants.get(right)
                && c.is_one()
                && let Some(zero) = typed_zero(0)
            {
                return SimplifyResult::Constant(zero);
            }
            SimplifyResult::None
        }

        // SHL/SHR/ROL/ROR: shift/rotate by 0 = x
        SsaOp::Shl { value, amount, .. }
        | SsaOp::Shr { value, amount, .. }
        | SsaOp::Rol { value, amount, .. }
        | SsaOp::Ror { value, amount, .. } => {
            if constants.get(amount).is_some_and(ConstValue::is_zero) {
                return SimplifyResult::Copy(*value);
            }
            SimplifyResult::None
        }

        // Comparisons: x == x → true, x < x → false, x > x → false
        SsaOp::Ceq { left, right, .. } => {
            // A comparison result is a boolean, so its own width is I32 by
            // convention - but the *operands* decide whether the identity holds.
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

        // The identities of the lane-wise logic: `x & 0` and `x | ~0` are the
        // constant, `x & ~0`, `x | 0`, `x ^ 0` and `x &~ 0` are `x`; and of
        // integer addition, `x + 0` and `x - 0` are `x` (a float lane's `-0.0
        // + 0.0` is `+0.0`).
        SsaOp::VectorBinary {
            left,
            right,
            kind,
            element,
            ..
        } => {
            let vector = |var: &SsaVarId| match constants.get(var) {
                Some(value @ ConstValue::Vector(lanes)) => Some((value, lanes)),
                _ => None,
            };
            let integer = matches!(element.kind, VectorElementKind::Integer);
            match (kind, vector(left), vector(right)) {
                (VectorBinaryKind::And, _, Some((zero, lanes)))
                | (VectorBinaryKind::And, Some((zero, lanes)), _)
                    if lanes.is_zero() =>
                {
                    SimplifyResult::Constant(zero.clone())
                }
                (VectorBinaryKind::Or, _, Some((ones, lanes)))
                | (VectorBinaryKind::Or, Some((ones, lanes)), _)
                    if lanes.is_all_ones() =>
                {
                    SimplifyResult::Constant(ones.clone())
                }
                (VectorBinaryKind::And, _, Some((_, lanes))) if lanes.is_all_ones() => {
                    SimplifyResult::Copy(*left)
                }
                (VectorBinaryKind::And, Some((_, lanes)), _) if lanes.is_all_ones() => {
                    SimplifyResult::Copy(*right)
                }
                (
                    VectorBinaryKind::Or | VectorBinaryKind::Xor | VectorBinaryKind::AndNot,
                    _,
                    Some((_, lanes)),
                ) if lanes.is_zero() => SimplifyResult::Copy(*left),
                (VectorBinaryKind::Or | VectorBinaryKind::Xor, Some((_, lanes)), _)
                    if lanes.is_zero() =>
                {
                    SimplifyResult::Copy(*right)
                }
                (VectorBinaryKind::Add | VectorBinaryKind::Sub, _, Some((_, lanes)))
                    if integer && lanes.is_zero() =>
                {
                    SimplifyResult::Copy(*left)
                }
                (VectorBinaryKind::Add, Some((_, lanes)), _) if integer && lanes.is_zero() => {
                    SimplifyResult::Copy(*right)
                }
                _ => SimplifyResult::None,
            }
        }

        // A choice between one value and itself, or two equal constants, is
        // that value whichever way the condition falls.
        SsaOp::Select {
            true_val,
            false_val,
            ..
        } => {
            if true_val == false_val {
                return SimplifyResult::Copy(*true_val);
            }
            match (constants.get(true_val), constants.get(false_val)) {
                (Some(chosen), Some(otherwise)) if chosen == otherwise => {
                    SimplifyResult::Constant(chosen.clone())
                }
                _ => SimplifyResult::None,
            }
        }

        _ => SimplifyResult::None,
    }
}

/// Returns one definition of each result of a computation over the lanes of a
/// vector whose restart index is at or beyond its active length, both
/// constant, or whose active length is a constant zero, or `None` for any
/// other operation.
///
/// Such a computation executes no element: it changes no element, agnostic
/// tails included, accesses no memory and raises nothing (RISC-V's `vle32.v`
/// with `vl` zero leaves `vd` as it was and touches no address). So each data
/// result is its passthrough, or a value the implementation chooses
/// ([`RuntimeValueKind::Unspecified`]) where the computation states
/// [`LaneControl::no_passthrough`](crate::ir::ops::LaneControl::no_passthrough);
/// a status is zero, the exceptions or saturation of no element; and a
/// fault-only-first load's length is the length, no element having faulted.
/// A computation that writes a scalar whatever the length (a mask summary,
/// `vcpop.m`), states flags or a first-fault predicate, or computes a whole
/// value, has none. `type_of` gives a result's type, which its zero takes.
#[must_use]
pub fn executing_nothing<T: Target>(
    op: &SsaOp<T>,
    constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    type_of: impl Fn(SsaVarId) -> Option<T::Type>,
) -> Option<Vec<SsaOp<T>>> {
    let SsaOp::Compute(data) = op else {
        return None;
    };
    let kind = data.kind;
    let lanes = kind.lanes?;
    if matches!(kind.kernel, Kernel::MaskSummary { .. })
        || data.outputs.len() != kind.output_count()
        || data.inputs.len() != kind.input_count()
    {
        return None;
    }
    let value = |at: usize| {
        data.inputs
            .get(at)
            .and_then(|input| constants.get(input))
            .and_then(ConstValue::as_u64)
    };
    // A zero length leaves nothing to execute whatever the restart is.
    match (value(0), value(1)) {
        (_, Some(0)) => {}
        (Some(restart), Some(length)) if restart >= length => {}
        _ => return None,
    }
    let fault = match kind.kernel {
        Kernel::Memory {
            fault: VectorFaultMode::FaultOnlyFirst,
            ..
        } => true,
        Kernel::Memory {
            fault: VectorFaultMode::FirstFault | VectorFaultMode::NoFault,
            ..
        } => return None,
        _ => false,
    };
    let status = !matches!(kind.status, VectorStatus::None);
    let data_outputs = kind.data_outputs();
    if data_outputs
        .saturating_add(usize::from(fault))
        .saturating_add(usize::from(status))
        != kind.output_count()
    {
        return None;
    }
    let passthrough = kind.control_input_count();
    let mut definitions = Vec::with_capacity(data.outputs.len());
    for (index, &dest) in data.outputs.iter().enumerate() {
        let definition = if index < data_outputs {
            if lanes.no_passthrough {
                SsaOp::RuntimeValue {
                    dest,
                    kind: RuntimeValueKind::Unspecified,
                }
            } else {
                SsaOp::Copy {
                    dest,
                    src: *data.inputs.get(passthrough.checked_add(index)?)?,
                }
            }
        } else if fault && index == data_outputs {
            let length = *data.inputs.get(1)?;
            if type_of(dest)? != type_of(length)? {
                return None;
            }
            SsaOp::Copy { dest, src: length }
        } else {
            let ty = type_of(dest)?;
            SsaOp::Const {
                dest,
                value: if T::is_boolean(&ty) {
                    ConstValue::False
                } else {
                    ConstValue::from_bits_of_type(0, &ty)?
                },
            }
        };
        definitions.push(definition);
    }
    Some(definitions)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        ir::{
            lanes::VectorConst,
            ops::{
                Computation, LaneControl, VectorElement, VectorInactivePolicy,
                VectorPredicateLayout, VectorRounding,
            },
        },
        target::VectorTailPolicy,
        testing::{MockTarget, MockType},
    };

    fn make_constants(
        pairs: &[(SsaVarId, ConstValue<MockTarget>)],
    ) -> BTreeMap<SsaVarId, ConstValue<MockTarget>> {
        pairs.iter().cloned().collect()
    }

    /// `x ^ x` folds to the constant `0`.
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

    /// `x ^ 0` folds to a copy of `x`.
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

    /// `x * 0` folds to the constant `0`.
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

    /// `x * 1` folds to a copy of `x`.
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

    /// The lane-wise identities: `x & 0` and `x | ~0` are the constant,
    /// `x & ~0`, `x | 0`, `x ^ 0`, `x &~ 0` and an integer `x + 0` and `x - 0`
    /// are `x`, whichever side the constant is on where the operation
    /// commutes; a float `x + 0` is not `x` (`-0.0 + 0.0` is `+0.0`), and a
    /// constant of some other lanes decides nothing.
    #[test]
    fn lane_wise_identities_fold_by_their_constant() {
        let [x, c, dest] = [0, 1, 2].map(SsaVarId::from_index);
        let zero = ConstValue::Vector(VectorConst::splat(32, 4, 0).unwrap());
        let ones = ConstValue::Vector(VectorConst::splat(32, 4, u128::MAX).unwrap());
        let other = ConstValue::Vector(VectorConst::from_lanes(32, [0, 1, 2, 3]).unwrap());
        let binary = |kind, left, right, element| SsaOp::<MockTarget>::VectorBinary {
            dest,
            left,
            right,
            kind,
            element,
            environment: None,
        };
        let integer = VectorElement::integer(32, None);
        let float = VectorElement::float(32);
        for (kind, constant, element, right_side, expected) in [
            (
                VectorBinaryKind::And,
                &zero,
                integer,
                true,
                SimplifyResult::Constant(zero.clone()),
            ),
            (
                VectorBinaryKind::And,
                &zero,
                integer,
                false,
                SimplifyResult::Constant(zero.clone()),
            ),
            (
                VectorBinaryKind::Or,
                &ones,
                integer,
                false,
                SimplifyResult::Constant(ones.clone()),
            ),
            (
                VectorBinaryKind::And,
                &ones,
                integer,
                true,
                SimplifyResult::Copy(x),
            ),
            (
                VectorBinaryKind::And,
                &ones,
                integer,
                false,
                SimplifyResult::Copy(x),
            ),
            (
                VectorBinaryKind::Or,
                &zero,
                integer,
                false,
                SimplifyResult::Copy(x),
            ),
            (
                VectorBinaryKind::Xor,
                &zero,
                integer,
                true,
                SimplifyResult::Copy(x),
            ),
            (
                VectorBinaryKind::AndNot,
                &zero,
                integer,
                true,
                SimplifyResult::Copy(x),
            ),
            (
                VectorBinaryKind::AndNot,
                &zero,
                integer,
                false,
                SimplifyResult::None,
            ),
            (
                VectorBinaryKind::Add,
                &zero,
                integer,
                false,
                SimplifyResult::Copy(x),
            ),
            (
                VectorBinaryKind::Sub,
                &zero,
                integer,
                true,
                SimplifyResult::Copy(x),
            ),
            (
                VectorBinaryKind::Sub,
                &zero,
                integer,
                false,
                SimplifyResult::None,
            ),
            (
                VectorBinaryKind::Add,
                &zero,
                float,
                true,
                SimplifyResult::None,
            ),
            (
                VectorBinaryKind::And,
                &other,
                integer,
                true,
                SimplifyResult::None,
            ),
        ] {
            let op = if right_side {
                binary(kind, x, c, element)
            } else {
                binary(kind, c, x, element)
            };
            let constants = make_constants(&[(c, constant.clone())]);
            assert_eq!(
                simplify_op(&op, &constants, Some(&MockType::V4I32)),
                expected,
                "{kind:?} {constant:?} right {right_side}"
            );
        }
    }

    /// `x + 0` folds to a copy of `x`.
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

    /// `x - x` folds to the constant `0`.
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

    /// `x & 0` folds to the constant `0`.
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

    /// `x | 0` folds to a copy of `x`.
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

    /// `x / 1` folds to a copy of `x`.
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

    /// `x << 0` folds to a copy of `x`.
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

    /// A non-identity `x + y` (neither operand constant) yields no simplification.
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

    /// `x % 1` folds to the constant `0`.
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

    /// `x == x` folds to the constant `1` (true).
    #[test]
    fn ceq_self_true() {
        let v1 = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Ceq {
            dest,
            left: v1,
            right: v1,
            signaling: false,
        };
        assert_eq!(
            simplify_op(&op, &BTreeMap::new(), Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(1))
        );
    }

    /// `x < x` folds to the constant `0` (false).
    #[test]
    fn clt_self_false() {
        let v1 = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Clt {
            dest,
            left: v1,
            right: v1,
            unsigned: false,
            signaling: false,
        };
        assert_eq!(
            simplify_op(&op, &BTreeMap::new(), Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(0))
        );
    }

    /// `x > x` folds to the constant `0` (false).
    #[test]
    fn cgt_self_false() {
        let v1 = SsaVarId::from_index(0);
        let dest = SsaVarId::from_index(1);
        let op: SsaOp<MockTarget> = SsaOp::Cgt {
            dest,
            left: v1,
            right: v1,
            unsigned: false,
            signaling: false,
        };
        assert_eq!(
            simplify_op(&op, &BTreeMap::new(), Some(&MockType::I32)),
            SimplifyResult::Constant(ConstValue::I32(0))
        );
    }

    /// `0 / x` and `0 % x` stay: `x` may be zero, which raises for an integer
    /// and is NaN for a float.
    #[test]
    fn a_zero_dividend_does_not_decide_the_quotient() {
        let (zero, x, dest) = (
            SsaVarId::from_index(0),
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
        );
        let constants = make_constants(&[(zero, ConstValue::I32(0))]);
        let div: SsaOp<MockTarget> = SsaOp::Div {
            dest,
            left: zero,
            right: x,
            unsigned: false,
            flags: None,
        };
        let rem: SsaOp<MockTarget> = SsaOp::Rem {
            dest,
            left: zero,
            right: x,
            unsigned: false,
            flags: None,
        };
        for op in [div, rem] {
            assert_eq!(
                simplify_op(&op, &constants, Some(&MockType::I32)),
                SimplifyResult::None
            );
        }
    }

    /// The integer identities `x + 0`, `x * 0` and `x % 1` are not float ones -
    /// `-0.0 + 0.0` is `+0.0`, `NaN * 0.0` is NaN, `2.5 % 1.0` is `0.5` - and a
    /// float constant never takes part in them.
    #[test]
    fn integer_identities_do_not_hold_for_floats() {
        let (x, c, dest) = (
            SsaVarId::from_index(0),
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
        );
        let add: SsaOp<MockTarget> = SsaOp::Add {
            dest,
            left: x,
            right: c,
            flags: None,
        };
        let mul: SsaOp<MockTarget> = SsaOp::Mul {
            dest,
            left: x,
            right: c,
            flags: None,
        };
        let rem: SsaOp<MockTarget> = SsaOp::Rem {
            dest,
            left: x,
            right: c,
            unsigned: false,
            flags: None,
        };
        for (op, value) in [(add, 0.0), (mul, 0.0), (rem, 1.0)] {
            let constants = make_constants(&[(c, ConstValue::from(value))]);
            assert_eq!(
                simplify_op(&op, &constants, Some(&MockType::F64)),
                SimplifyResult::None
            );
        }
    }

    /// An absorbed result has the operation's width, not the constant's: a
    /// 64-bit `x % 1` with a 32-bit `1` is a 64-bit zero, and a vector `x ^ x`
    /// has no scalar zero to be.
    #[test]
    fn an_absorbed_result_has_the_operations_width() {
        let (x, one, dest) = (
            SsaVarId::from_index(0),
            SsaVarId::from_index(1),
            SsaVarId::from_index(2),
        );
        let rem: SsaOp<MockTarget> = SsaOp::Rem {
            dest,
            left: x,
            right: one,
            unsigned: false,
            flags: None,
        };
        let constants = make_constants(&[(one, ConstValue::I32(1))]);
        assert_eq!(
            simplify_op(&rem, &constants, Some(&MockType::I64)),
            SimplifyResult::Constant(ConstValue::I64(0))
        );
        let xor: SsaOp<MockTarget> = SsaOp::Xor {
            dest,
            left: x,
            right: x,
            flags: None,
        };
        assert_eq!(
            simplify_op(&xor, &BTreeMap::new(), Some(&MockType::V4I32)),
            SimplifyResult::None
        );
    }

    /// **A lane computation that executes no element leaves each result as
    /// it was.** Where the restart index is at or beyond the active length,
    /// both constant, a sum over lanes is a copy of its passthrough, found
    /// after the controls (restart, length, and a governing predicate where
    /// one governs), as it is where the length is zero and the restart not
    /// known; and one that states no passthrough is a value the
    /// implementation chooses. A two-field segment load is a copy of each
    /// field's passthrough, a store is replaced by nothing, a saturating sum's
    /// status is false, and a fault-only-first load's length is the length,
    /// where it has the length's type. A restart below the length, a restart
    /// or length not known, a mask summary, which writes its scalar whatever
    /// the length, and a first-fault load's predicate are none.
    #[test]
    fn a_lane_computation_executing_no_element_leaves_each_result() {
        let var = SsaVarId::from_index;
        let lanes = |inactive, no_passthrough| LaneControl {
            mask_tail: false,
            tail: VectorTailPolicy::Agnostic,
            inactive,
            no_passthrough,
        };
        let kind = |kernel, inactive, no_passthrough| Computation {
            kernel,
            element: VectorElement::integer(32, None),
            predicate: VectorPredicateLayout::Packed,
            rounding: VectorRounding::None,
            status: VectorStatus::None,
            canonical_nan: false,
            lanes: Some(lanes(inactive, no_passthrough)),
        };
        // Results from `var(10)`; restart `var(0)`, length `var(1)`, then the
        // predicate, passthroughs and sources.
        let op = |kind: Computation| {
            SsaOp::<MockTarget>::compute(
                kind,
                (10..10 + kind.output_count()).map(var).collect(),
                (0..kind.input_count()).map(var).collect(),
            )
        };
        let known = |restart: u64, length: u64| {
            make_constants(&[
                (var(0), ConstValue::U64(restart)),
                (var(1), ConstValue::U64(length)),
            ])
        };
        // The length and a fault-only-first load's length are 64-bit, a
        // status a boolean.
        let type_of = |of: SsaVarId| {
            Some(match of.index() {
                1 | 11 => MockType::U64,
                12 => MockType::Bool,
                _ => MockType::V4I32,
            })
        };
        let empty = |op: &SsaOp<MockTarget>, constants| executing_nothing(op, &constants, type_of);
        let copy = |dest: usize, src: usize| SsaOp::Copy {
            dest: var(dest),
            src: var(src),
        };
        let add = Kernel::Binary(VectorBinaryKind::Add);
        let unmasked = op(kind(add, None, false));
        assert_eq!(empty(&unmasked, known(2, 2)), Some(vec![copy(10, 2)]));
        assert_eq!(empty(&unmasked, known(3, 0)), Some(vec![copy(10, 2)]));
        let masked = op(kind(add, Some(VectorInactivePolicy::Undisturbed), false));
        assert_eq!(empty(&masked, known(0, 0)), Some(vec![copy(10, 3)]));
        assert_eq!(empty(&unmasked, known(0, 4)), None);
        assert_eq!(
            empty(&unmasked, make_constants(&[(var(0), ConstValue::U64(0))])),
            None
        );
        assert_eq!(
            empty(&unmasked, make_constants(&[(var(1), ConstValue::U64(0))])),
            Some(vec![copy(10, 2)])
        );
        assert_eq!(
            empty(&unmasked, make_constants(&[(var(1), ConstValue::U64(1))])),
            None
        );
        let dropped = op(kind(add, None, true));
        assert_eq!(
            empty(&dropped, known(0, 0)),
            Some(vec![SsaOp::RuntimeValue {
                dest: var(10),
                kind: RuntimeValueKind::Unspecified,
            }])
        );
        let summary = op(kind(Kernel::MaskSummary { first: true }, None, false));
        assert_eq!(empty(&summary, known(0, 0)), None);
        let memory = |load, fields, fault| Kernel::Memory {
            load,
            access_bits: 32,
            sign_extend: false,
            byte_stream: false,
            coalescible: true,
            indexed: false,
            ordered: false,
            fields,
            fault,
        };
        let segments = op(kind(memory(true, 2, VectorFaultMode::Normal), None, false));
        assert_eq!(
            empty(&segments, known(0, 0)),
            Some(vec![copy(10, 2), copy(11, 3)])
        );
        let store = op(kind(memory(false, 1, VectorFaultMode::Normal), None, false));
        assert_eq!(empty(&store, known(0, 0)), Some(Vec::new()));
        let saturating = op(Computation {
            status: VectorStatus::Saturation,
            ..kind(Kernel::Binary(VectorBinaryKind::SatAdd), None, false)
        });
        // The sum, then its status as the third result (`var(12)`).
        let saturating = match saturating {
            SsaOp::Compute(mut data) => {
                data.outputs = vec![var(10), var(12)];
                SsaOp::Compute(data)
            }
            other => other,
        };
        assert_eq!(
            empty(&saturating, known(0, 0)),
            Some(vec![
                copy(10, 2),
                SsaOp::Const {
                    dest: var(12),
                    value: ConstValue::False,
                },
            ])
        );
        let faulting = op(kind(
            memory(true, 1, VectorFaultMode::FaultOnlyFirst),
            None,
            false,
        ));
        assert_eq!(
            empty(&faulting, known(0, 0)),
            Some(vec![copy(10, 2), copy(11, 1)])
        );
        let mistyped = |of: SsaVarId| {
            Some(if of.index() == 11 {
                MockType::U32
            } else {
                MockType::U64
            })
        };
        assert_eq!(executing_nothing(&faulting, &known(0, 0), mistyped), None);
        let predicated = op(kind(
            memory(true, 1, VectorFaultMode::FirstFault),
            None,
            false,
        ));
        assert_eq!(empty(&predicated, known(0, 0)), None);
    }

    /// **A choice of one value either way is that value.** `c ? x : x` is a
    /// copy of `x`, and a choice between two equal constants is the constant;
    /// a choice between two different constants, or values not known, is no
    /// identity.
    #[test]
    fn a_choice_of_one_value_either_way_is_that_value() {
        let var = SsaVarId::from_index;
        let select = |true_val, false_val| SsaOp::<MockTarget>::Select {
            dest: var(9),
            condition: var(0),
            true_val,
            false_val,
        };
        let simplify = |op: &SsaOp<MockTarget>, constants| simplify_op(op, &constants, None);
        assert_eq!(
            simplify(&select(var(1), var(1)), BTreeMap::new()),
            SimplifyResult::Copy(var(1))
        );
        let fives = make_constants(&[(var(1), ConstValue::U64(5)), (var(2), ConstValue::U64(5))]);
        assert_eq!(
            simplify(&select(var(1), var(2)), fives),
            SimplifyResult::Constant(ConstValue::U64(5))
        );
        let differing =
            make_constants(&[(var(1), ConstValue::U64(5)), (var(2), ConstValue::U64(6))]);
        assert_eq!(
            simplify(&select(var(1), var(2)), differing),
            SimplifyResult::None
        );
        assert_eq!(
            simplify(&select(var(1), var(2)), BTreeMap::new()),
            SimplifyResult::None
        );
    }
}
