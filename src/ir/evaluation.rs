//! The outcome of evaluating one operation exactly.
//!
//! An operation evaluates to a value, to a trap its exact semantics establish
//! ([`ArithmeticTrap`]), to unknown when an operand or the environment is not
//! known, or to unsupported ([`Evaluation`]). Keeping these apart lets a fold
//! replace a value while a possible trap still has to execute.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod numeric;

pub use self::numeric::NumericOp;

/// A fault established by exact arithmetic semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum ArithmeticTrap {
    /// An integer division or remainder receives a zero divisor.
    DivideByZero,
    /// A checked arithmetic operation or conversion exceeds its integer range.
    Overflow,
    /// A finite check receives a NaN or an infinity.
    NonFinite,
    /// A conditional trap's condition holds ([`TrapOp::condition`]): MIPS
    /// `teq` checking a divisor, an MPX bound check.
    ///
    /// [`TrapOp::condition`]: crate::ir::ops::TrapOp::condition
    Condition,
}

/// The result of evaluating one operation in a numeric domain.
///
/// An unknown operand or environment does not prove a trap. Unsupported
/// semantics likewise provide no value or control proof.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Evaluation<V> {
    /// Evaluation establishes this exact value.
    Value(V),
    /// Runtime inputs or environment choices prevent a decision.
    Unknown,
    /// The operation or its value/type domain has no supported semantics.
    Unsupported,
    /// Every execution represented by the operands raises this fault.
    Trap(ArithmeticTrap),
}

impl<V> Evaluation<V> {
    /// Returns the established value, withholding every other outcome.
    #[must_use]
    pub fn into_value(self) -> Option<V> {
        match self {
            Self::Value(value) => Some(value),
            _ => None,
        }
    }

    /// Transforms an established value while preserving control and refusal outcomes.
    pub fn map<W>(self, map: impl FnOnce(V) -> W) -> Evaluation<W> {
        match self {
            Self::Value(value) => Evaluation::Value(map(value)),
            Self::Unknown => Evaluation::Unknown,
            Self::Unsupported => Evaluation::Unsupported,
            Self::Trap(trap) => Evaluation::Trap(trap),
        }
    }
}
