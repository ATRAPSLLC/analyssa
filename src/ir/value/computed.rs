//! Values computed by an operation a host records ([`ComputedValue`]).

use std::{fmt, hash::Hash};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ir::variable::SsaVarId;

/// Computed value for common subexpression elimination (CSE).
///
/// This represents the result of a computation, enabling recognition
/// of equivalent expressions that can be eliminated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ComputedValue {
    /// The operation that produced this value.
    pub op: ComputedOp,
    /// The operands to the operation.
    pub operands: Vec<SsaVarId>,
}

impl ComputedValue {
    /// Creates a new computed value.
    #[must_use]
    pub fn new(op: ComputedOp, operands: Vec<SsaVarId>) -> Self {
        Self { op, operands }
    }

    /// Creates a unary computed value.
    #[must_use]
    pub fn unary(op: ComputedOp, operand: SsaVarId) -> Self {
        Self {
            op,
            operands: vec![operand],
        }
    }

    /// Creates a binary computed value.
    #[must_use]
    pub fn binary(op: ComputedOp, left: SsaVarId, right: SsaVarId) -> Self {
        Self {
            op,
            operands: vec![left, right],
        }
    }

    /// Normalizes commutative operations for better CSE.
    ///
    /// For commutative ops like add/mul, orders operands consistently
    /// so that `a + b` and `b + a` have the same computed value.
    #[must_use]
    pub fn normalized(self) -> Self {
        if self.op.is_commutative() && self.operands.len() == 2 {
            let mut ops = self.operands;
            if let (Some(a), Some(b)) = (ops.first(), ops.get(1))
                && a.index() > b.index()
            {
                ops.swap(0, 1);
            }
            Self {
                op: self.op,
                operands: ops,
            }
        } else {
            self
        }
    }
}

impl fmt::Display for ComputedValue {
    /// Formats the computation as `op(operand, ...)`, e.g. `add(v1, v2)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.op)?;
        for (i, op) in self.operands.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{op}")?;
        }
        write!(f, ")")
    }
}

/// Operations that can be tracked for CSE.
///
/// These represent the pure operations whose results can be reused
/// when the same operation is performed with the same operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComputedOp {
    // Arithmetic
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Remainder (modulo)
    Rem,
    /// Negation
    Neg,

    // Bitwise
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Bitwise NOT
    Not,
    /// Shift left
    Shl,
    /// Shift right
    Shr,

    // Comparison
    /// Compare equal
    Ceq,
    /// Compare not equal
    Cne,
    /// Compare less than
    Clt,
    /// Compare greater than
    Cgt,
    /// Compare less than or equal
    Cle,
    /// Compare greater than or equal
    Cge,

    // Conversion
    /// Convert to int8
    ConvI1,
    /// Convert to int16
    ConvI2,
    /// Convert to int32
    ConvI4,
    /// Convert to int64
    ConvI8,
    /// Convert to uint8
    ConvU1,
    /// Convert to uint16
    ConvU2,
    /// Convert to uint32
    ConvU4,
    /// Convert to uint64
    ConvU8,
    /// Convert to float32
    ConvR4,
    /// Convert to float64
    ConvR8,
}

impl ComputedOp {
    /// Returns `true` if this operation is commutative.
    #[must_use]
    pub const fn is_commutative(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::And | Self::Or | Self::Xor | Self::Ceq | Self::Cne
        )
    }

    /// Returns `true` if this is a comparison operation.
    #[must_use]
    pub const fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Ceq | Self::Cne | Self::Clt | Self::Cgt | Self::Cle | Self::Cge
        )
    }
}

impl fmt::Display for ComputedOp {
    /// Formats the op as its lowercase IR mnemonic (`add`, `shl`, `conv.i4`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Rem => "rem",
            Self::Neg => "neg",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Not => "not",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::Ceq => "ceq",
            Self::Cne => "cne",
            Self::Clt => "clt",
            Self::Cgt => "cgt",
            Self::Cle => "cle",
            Self::Cge => "cge",
            Self::ConvI1 => "conv.i1",
            Self::ConvI2 => "conv.i2",
            Self::ConvI4 => "conv.i4",
            Self::ConvI8 => "conv.i8",
            Self::ConvU1 => "conv.u1",
            Self::ConvU2 => "conv.u2",
            Self::ConvU4 => "conv.u4",
            Self::ConvU8 => "conv.u8",
            Self::ConvR4 => "conv.r4",
            Self::ConvR8 => "conv.r8",
        };
        write!(f, "{s}")
    }
}
