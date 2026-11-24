//! Scalar integer division outcomes where instruction sets disagree.

use crate::{
    PointerSize,
    ir::{
        evaluation::{ArithmeticTrap, Evaluation},
        integer::IntegerType,
        value::ConstValue,
    },
    target::Target,
};

/// The outcome of one exceptional integer division case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DivisionFailure {
    /// Raises the established arithmetic fault.
    Trap,
    /// Produces zero at the dividend's width.
    Zero,
    /// Produces the dividend unchanged.
    Dividend,
    /// Produces all one bits at the dividend's width.
    AllOnes,
    /// Establishes neither a value nor a definite fault.
    Unknown,
}

impl DivisionFailure {
    /// Applies a stated outcome even when only the divisor and integer layout are known.
    pub fn evaluate<T: Target>(
        self,
        dividend: Option<&ConstValue<T>>,
        layout: IntegerType,
        trap: ArithmeticTrap,
        ptr_size: PointerSize,
    ) -> Evaluation<ConstValue<T>> {
        let constant = match self {
            Self::Trap => return Evaluation::Trap(trap),
            Self::Unknown => return Evaluation::Unknown,
            Self::Dividend => {
                return dividend
                    .cloned()
                    .map_or(Evaluation::Unknown, Evaluation::Value);
            }
            Self::Zero => layout.constant(0, ptr_size),
            Self::AllOnes => layout.constant(u128::MAX, ptr_size),
        };
        constant.map_or(Evaluation::Unsupported, Evaluation::Value)
    }
}

/// States exceptional outcomes for scalar integer division and remainder.
///
/// These rules do not alter floating arithmetic or the explicit x86-style
/// double-width quotient contract of [`crate::ir::ops::SsaOp::WideDiv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DivisionRules {
    /// Quotient when the divisor is zero.
    pub quotient_zero: DivisionFailure,
    /// Remainder when the divisor is zero.
    pub remainder_zero: DivisionFailure,
    /// Signed quotient when dividing the minimum integer by minus one.
    pub quotient_overflow: DivisionFailure,
    /// Signed remainder when dividing the minimum integer by minus one.
    pub remainder_overflow: DivisionFailure,
}

impl DivisionRules {
    /// Checked scalar arithmetic, with an unstated minimum-integer remainder outcome.
    pub const CHECKED: Self = Self {
        quotient_zero: DivisionFailure::Trap,
        remainder_zero: DivisionFailure::Trap,
        quotient_overflow: DivisionFailure::Trap,
        remainder_overflow: DivisionFailure::Unknown,
    };

    /// ARM scalar division and the remainder derived from its quotient.
    pub const ARM: Self = Self {
        quotient_zero: DivisionFailure::Zero,
        remainder_zero: DivisionFailure::Dividend,
        quotient_overflow: DivisionFailure::Dividend,
        remainder_overflow: DivisionFailure::Zero,
    };

    /// RISC-V scalar integer division and remainder.
    pub const RISCV: Self = Self {
        quotient_zero: DivisionFailure::AllOnes,
        remainder_zero: DivisionFailure::Dividend,
        quotient_overflow: DivisionFailure::Dividend,
        remainder_overflow: DivisionFailure::Zero,
    };

    /// Unstated exceptional outcomes, including unpredictable instruction results.
    pub const UNSTATED: Self = Self {
        quotient_zero: DivisionFailure::Unknown,
        remainder_zero: DivisionFailure::Unknown,
        quotient_overflow: DivisionFailure::Unknown,
        remainder_overflow: DivisionFailure::Unknown,
    };
}
