//! Captured scalar integer semantics for delayed evaluation.
//!
//! Descriptors retain layouts and machine policy without retaining host types
//! or source variable identities. Evaluation projects only the primary value;
//! flags, effects and reachability remain the source driver's responsibility.

use crate::{
    analysis::consts::{Destination, Machine, OperandLayout},
    ir::{
        division::DivisionFailure,
        evaluation::{Evaluation, NumericOp},
        function::SsaFunction,
        integer::IntegerType,
        ops::SsaOp,
        value::ConstValue,
        variable::SsaVarId,
    },
    target::Target,
};

/// Returns an outcome for terminal forms that cannot continue normal scalar execution.
///
/// User exceptions have no established arithmetic trap reason. EH and machine
/// continuation require a scope contract outside these ordered value drivers.
/// This classification does not change conditional CFG successor queries.
pub fn nonreturn_terminal<T: Target>(op: &SsaOp<T>) -> Option<Evaluation<()>> {
    match op {
        SsaOp::Throw { .. } | SsaOp::Rethrow => Some(Evaluation::Unknown),
        SsaOp::Unreachable
        | SsaOp::EndFinally
        | SsaOp::EndFilter { .. }
        | SsaOp::InterruptReturn => Some(Evaluation::Unsupported),
        _ => None,
    }
}

/// One admitted scalar operation, without source identities or owned host metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    /// A scalar integer operation represented directly by the numeric vocabulary.
    Numeric(NumericOp),
    /// Addition, subtraction or multiplication with an explicit overflow check.
    Checked {
        /// The arithmetic operation, limited by admission to add, subtract or multiply.
        op: NumericOp,
        /// Whether the overflow range is unsigned.
        unsigned: bool,
    },
    /// An integer conversion with its projected target representation.
    Convert {
        /// The exact destination range and constant representation.
        target: IntegerType,
        /// Whether the input pattern is interpreted as unsigned.
        unsigned: bool,
        /// Whether an out-of-range value traps.
        checked: bool,
    },
    /// A value copy that still reads and normalizes its declared layouts.
    Copy,
}

/// Immutable scalar semantics captured from an original SSA operation.
///
/// Construction admits complete scalar integer layouts only. Equality includes
/// the machine and every operand/result declaration. A descriptor does not
/// establish that execution reaches the operation or remove its secondary outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarOperation {
    kind: ScalarKind,
    machine: Machine,
    destination: Destination,
    operands: [OperandLayout; 2],
    arity: u8,
}

/// One operation's admitted scalar semantics and destination layout.
///
/// Derived once per evaluated operation and shared by its control, bit and
/// value transfers, so each reads the same declared layouts.
#[derive(Debug, Clone, Copy)]
pub struct OperationLayout {
    /// The admitted scalar semantics and their ordered source operands.
    pub scalar: Option<(ScalarOperation, [Option<SsaVarId>; 2])>,
    /// The primary destination's layout.
    pub destination: Destination,
}

impl OperationLayout {
    /// Captures `op`'s scalar semantics, or its declared destination when it has none.
    #[must_use]
    pub fn of<T: Target>(ssa: &SsaFunction<T>, op: &SsaOp<T>, machine: Machine) -> Self {
        let scalar = ScalarOperation::capture(ssa, op, machine);
        let destination = scalar.map_or_else(
            || Destination::of_operation(ssa, op),
            |(operation, _)| operation.destination(),
        );
        Self {
            scalar,
            destination,
        }
    }

    /// Returns the layout at which this operation reads `variable`.
    #[must_use]
    pub fn operand<T: Target>(self, ssa: &SsaFunction<T>, variable: SsaVarId) -> OperandLayout {
        self.scalar
            .and_then(|(operation, operands)| {
                operands
                    .iter()
                    .position(|operand| *operand == Some(variable))
                    .and_then(|index| operation.operand(index))
            })
            .unwrap_or_else(|| OperandLayout::of(ssa, variable))
    }
}

impl ScalarOperation {
    /// Captures scalar semantics and returns the ordered source operand identities.
    ///
    /// Returns `None` for an unrepresented operation or incomplete, pointer,
    /// floating or vector domain. The returned identities are separate from the
    /// descriptor, so renaming source variables does not change semantic equality.
    #[must_use]
    pub fn capture<T: Target>(
        ssa: &SsaFunction<T>,
        source: &SsaOp<T>,
        machine: Machine,
    ) -> Option<(Self, [Option<SsaVarId>; 2])> {
        let (kind, first, second) = match source {
            SsaOp::Add { left, right, .. } => {
                (ScalarKind::Numeric(NumericOp::Add), *left, Some(*right))
            }
            SsaOp::Sub { left, right, .. } => {
                (ScalarKind::Numeric(NumericOp::Sub), *left, Some(*right))
            }
            SsaOp::Mul { left, right, .. } => {
                (ScalarKind::Numeric(NumericOp::Mul), *left, Some(*right))
            }
            SsaOp::And { left, right, .. } => {
                (ScalarKind::Numeric(NumericOp::And), *left, Some(*right))
            }
            SsaOp::Or { left, right, .. } => {
                (ScalarKind::Numeric(NumericOp::Or), *left, Some(*right))
            }
            SsaOp::Xor { left, right, .. } => {
                (ScalarKind::Numeric(NumericOp::Xor), *left, Some(*right))
            }
            SsaOp::Ceq { left, right, .. } => {
                (ScalarKind::Numeric(NumericOp::Eq), *left, Some(*right))
            }
            SsaOp::Div {
                left,
                right,
                unsigned,
                ..
            } => (
                ScalarKind::Numeric(if *unsigned {
                    NumericOp::DivU
                } else {
                    NumericOp::DivS
                }),
                *left,
                Some(*right),
            ),
            SsaOp::Rem {
                left,
                right,
                unsigned,
                ..
            } => (
                ScalarKind::Numeric(if *unsigned {
                    NumericOp::RemU
                } else {
                    NumericOp::RemS
                }),
                *left,
                Some(*right),
            ),
            SsaOp::Clt {
                left,
                right,
                unsigned,
                ..
            } => (
                ScalarKind::Numeric(if *unsigned {
                    NumericOp::LtU
                } else {
                    NumericOp::LtS
                }),
                *left,
                Some(*right),
            ),
            SsaOp::Cgt {
                left,
                right,
                unsigned,
                ..
            } => (
                ScalarKind::Numeric(if *unsigned {
                    NumericOp::GtU
                } else {
                    NumericOp::GtS
                }),
                *left,
                Some(*right),
            ),
            SsaOp::AddOvf {
                left,
                right,
                unsigned,
                ..
            } => (
                ScalarKind::Checked {
                    op: NumericOp::Add,
                    unsigned: *unsigned,
                },
                *left,
                Some(*right),
            ),
            SsaOp::SubOvf {
                left,
                right,
                unsigned,
                ..
            } => (
                ScalarKind::Checked {
                    op: NumericOp::Sub,
                    unsigned: *unsigned,
                },
                *left,
                Some(*right),
            ),
            SsaOp::MulOvf {
                left,
                right,
                unsigned,
                ..
            } => (
                ScalarKind::Checked {
                    op: NumericOp::Mul,
                    unsigned: *unsigned,
                },
                *left,
                Some(*right),
            ),
            SsaOp::Shl { value, amount, .. } => {
                (ScalarKind::Numeric(NumericOp::Shl), *value, Some(*amount))
            }
            SsaOp::Rol { value, amount, .. } => {
                (ScalarKind::Numeric(NumericOp::Rol), *value, Some(*amount))
            }
            SsaOp::Ror { value, amount, .. } => {
                (ScalarKind::Numeric(NumericOp::Ror), *value, Some(*amount))
            }
            SsaOp::Shr {
                value,
                amount,
                unsigned,
                ..
            } => (
                ScalarKind::Numeric(if *unsigned {
                    NumericOp::ShrU
                } else {
                    NumericOp::ShrS
                }),
                *value,
                Some(*amount),
            ),
            SsaOp::Neg { operand, .. } => (ScalarKind::Numeric(NumericOp::Neg), *operand, None),
            SsaOp::Not { operand, .. } => (ScalarKind::Numeric(NumericOp::Not), *operand, None),
            SsaOp::BSwap { src, .. } => (ScalarKind::Numeric(NumericOp::BSwap), *src, None),
            SsaOp::BRev { src, .. } => (ScalarKind::Numeric(NumericOp::BRev), *src, None),
            SsaOp::BitScanForward { src, .. } => {
                (ScalarKind::Numeric(NumericOp::BitScanForward), *src, None)
            }
            SsaOp::BitScanReverse { src, .. } => {
                (ScalarKind::Numeric(NumericOp::BitScanReverse), *src, None)
            }
            SsaOp::Popcount { src, .. } => (ScalarKind::Numeric(NumericOp::Popcount), *src, None),
            SsaOp::Parity { src, .. } => (ScalarKind::Numeric(NumericOp::Parity), *src, None),
            SsaOp::Copy { src, .. } => (ScalarKind::Copy, *src, None),
            SsaOp::IntConv {
                operand,
                target,
                unsigned,
                overflow_check,
                ..
            } if T::is_integer(target)
                && !T::is_pointer(target)
                && T::vector_shape(target).is_none() =>
            {
                (
                    ScalarKind::Convert {
                        target: T::integer_conversion_type(target, machine.ptr_size)?,
                        unsigned: *unsigned,
                        checked: *overflow_check,
                    },
                    *operand,
                    None,
                )
            }
            _ => return None,
        };
        let (destination, _) = operand_layout(ssa, source.dest()?, machine)?;
        if let ScalarKind::Convert { target, .. } = kind
            && destination.bits != Some(target.bits())
        {
            return None;
        }
        let (_, first_layout) = operand_layout(ssa, first, machine)?;
        let second_layout = match second {
            Some(id) => Some(operand_layout(ssa, id, machine)?.1),
            None => None,
        };
        Some((
            Self {
                kind,
                machine,
                destination,
                operands: [first_layout, second_layout.unwrap_or_default()],
                arity: if second.is_some() { 2 } else { 1 },
            },
            [Some(first), second],
        ))
    }

    /// Returns the captured operation kind.
    #[must_use]
    pub const fn kind(self) -> ScalarKind {
        self.kind
    }

    /// Returns the captured machine policy.
    #[must_use]
    pub const fn machine(self) -> Machine {
        self.machine
    }

    /// Returns the captured destination layout.
    #[must_use]
    pub const fn destination(self) -> Destination {
        self.destination
    }

    /// Returns the captured layout of an operand position, or `None` past its arity.
    #[must_use]
    pub fn operand(self, index: usize) -> Option<OperandLayout> {
        self.operands
            .get(index)
            .copied()
            .filter(|_| index < usize::from(self.arity))
    }

    /// Returns the number of scalar operands.
    #[must_use]
    pub const fn arity(self) -> u8 {
        self.arity
    }

    /// Proves normal completion for every normal input in the captured domains.
    ///
    /// This is separate from knowing the result value. It admits only total
    /// scalar contracts; callers must separately preserve child computations
    /// and any uncertainty about reaching this operation. Missing exceptional
    /// policies, checked arithmetic and unbounded shift counts do not prove
    /// normal completion.
    #[must_use]
    pub fn returns_normally_for_normal_inputs(self) -> bool {
        let same_width = |index| {
            matches!(self.operand(index), Some(OperandLayout::Register(input))
            if input.integer && input.lanes.is_none() && input.bits == self.destination.bits)
        };
        let both_same_width = || same_width(0) && same_width(1);
        let nontrapping = |failure| {
            matches!(
                failure,
                DivisionFailure::Zero | DivisionFailure::Dividend | DivisionFailure::AllOnes
            )
        };
        match self.kind {
            ScalarKind::Convert {
                target,
                checked: false,
                ..
            } => self.destination.bits == Some(target.bits()),
            ScalarKind::Copy => same_width(0),
            ScalarKind::Numeric(
                NumericOp::Add
                | NumericOp::Sub
                | NumericOp::Mul
                | NumericOp::And
                | NumericOp::Or
                | NumericOp::Xor,
            ) => both_same_width(),
            ScalarKind::Numeric(NumericOp::Neg | NumericOp::Not | NumericOp::BRev) => same_width(0),
            ScalarKind::Numeric(NumericOp::BSwap) => {
                same_width(0) && self.destination.bits.is_some_and(|bits| bits % 8 == 0)
            }
            ScalarKind::Numeric(NumericOp::Rol | NumericOp::Ror) => same_width(0),
            ScalarKind::Numeric(
                NumericOp::Eq | NumericOp::LtS | NumericOp::LtU | NumericOp::GtS | NumericOp::GtU,
            ) => self.operands[0] == self.operands[1],
            ScalarKind::Numeric(NumericOp::Popcount | NumericOp::Parity) => {
                self.destination.bits == Some(32)
            }
            ScalarKind::Numeric(NumericOp::DivU) => {
                both_same_width() && nontrapping(self.machine.division.quotient_zero)
            }
            ScalarKind::Numeric(NumericOp::RemU) => {
                both_same_width() && nontrapping(self.machine.division.remainder_zero)
            }
            ScalarKind::Numeric(NumericOp::DivS) => {
                both_same_width()
                    && nontrapping(self.machine.division.quotient_zero)
                    && nontrapping(self.machine.division.quotient_overflow)
            }
            ScalarKind::Numeric(NumericOp::RemS) => {
                both_same_width()
                    && nontrapping(self.machine.division.remainder_zero)
                    && nontrapping(self.machine.division.remainder_overflow)
            }
            _ => false,
        }
    }

    /// Proves normal completion with an admitted known shift count.
    ///
    /// Shift failure depends on the count and declared widths, so the shared
    /// kernel's result for a zero value proves every normal value at that count.
    /// Other operations retain the universal normal-input proof.
    pub fn returns_normally_with_count<T: Target>(self, count: Option<&ConstValue<T>>) -> bool {
        if self.returns_normally_for_normal_inputs() {
            return true;
        }
        if !matches!(
            self.kind,
            ScalarKind::Numeric(NumericOp::Shl | NumericOp::ShrS | NumericOp::ShrU)
        ) {
            return false;
        }
        let Some(OperandLayout::Register(input)) = self.operand(0) else {
            return false;
        };
        if !input.integer || input.lanes.is_some() || input.bits != self.destination.bits {
            return false;
        }
        let Some(zero) = self
            .destination
            .scalar_integer()
            .and_then(|layout| layout.constant(0, self.machine.ptr_size))
        else {
            return false;
        };
        let Some(count) = count.filter(|count| {
            count
                .conversion_integer_pattern(self.machine.ptr_size)
                .is_some()
                && self
                    .operand(1)
                    .and_then(|layout| layout.read_comparison_ref(count, self.machine.ptr_size))
                    .is_some()
        }) else {
            return false;
        };
        matches!(
            self.evaluate(Some(zero), Some(count.clone())),
            Evaluation::Value(_)
        )
    }

    /// Returns the kernel's result for identical admitted integer inputs.
    ///
    /// The typed SSA driver establishes input identity and normal completion.
    /// This method supplies only the width and representation preserving result.
    pub fn identical_result<T: Target>(self) -> Option<ConstValue<T>> {
        if !matches!(
            self.kind,
            ScalarKind::Numeric(NumericOp::Sub | NumericOp::Xor)
        ) || !self.returns_normally_for_normal_inputs()
        {
            return None;
        }
        let zero = self
            .destination
            .scalar_integer()?
            .constant(0, self.machine.ptr_size)?;
        self.evaluate(Some(zero.clone()), Some(zero)).into_value()
    }

    /// Evaluates supplied normal operands through the shared machine kernel.
    ///
    /// `None` denotes an unknown normal scalar input, not a failing or potentially
    /// failing child computation. Expression callers must retain those obligations
    /// before invoking this method. All kernel outcomes remain distinct.
    pub fn evaluate<T: Target>(
        self,
        first: Option<ConstValue<T>>,
        second: Option<ConstValue<T>>,
    ) -> Evaluation<ConstValue<T>> {
        // Reject values outside the admitted scalar domain before an absent
        // sibling could conceal the mismatch as ordinary runtime uncertainty.
        for (layout, value) in self
            .operands
            .into_iter()
            .zip([first.as_ref(), second.as_ref()])
            .take(usize::from(self.arity))
        {
            if value.is_some_and(|value| {
                layout
                    .read_comparison_ref(value, self.machine.ptr_size)
                    .and_then(|value| value.conversion_integer_pattern(self.machine.ptr_size))
                    .is_none()
            }) {
                return Evaluation::Unsupported;
            }
        }
        if let ScalarKind::Convert {
            target,
            unsigned,
            checked,
        } = self.kind
        {
            let Some(value) = first else {
                return Evaluation::Unknown;
            };
            let Some(value) = self.operands[0].read_comparison(value, self.machine.ptr_size) else {
                return Evaluation::Unsupported;
            };
            return match target.convert(&value, unsigned, checked, self.machine.ptr_size) {
                Evaluation::Value(value) => self
                    .machine
                    .normalize_result(value, self.destination)
                    .map_or(Evaluation::Unsupported, Evaluation::Value),
                other => other,
            };
        }
        let Some(source) = self.source() else {
            return Evaluation::Unsupported;
        };
        let get = |id: SsaVarId| {
            if id == SsaVarId::from_index(1) {
                first.clone()
            } else {
                second.clone()
            }
        };
        let layout = |id: SsaVarId| {
            if id == SsaVarId::from_index(1) {
                self.operands[0]
            } else {
                self.operands[1]
            }
        };
        if let Some(outcome) = self.machine.evaluate_control_with_operand_layouts(
            &source,
            self.destination,
            get,
            layout,
        ) {
            return outcome;
        }
        if first.is_none() || (self.arity == 2 && second.is_none()) {
            return Evaluation::Unknown;
        }
        // Every input is supplied and the operation has no unstated policy in
        // this adapter. A refused operation/input combination is unsupported
        // (for example an undefined shift count or a zero bit scan), rather than
        // a missing runtime value. Policy uncertainty stays in control outcomes.
        self.machine
            .evaluate_with_operand_layouts(&source, self.destination, get, layout)
            .map_or(Evaluation::Unsupported, Evaluation::Value)
    }

    /// Reconstructs target-free operations with fixed local operand slots.
    fn source<T: Target>(self) -> Option<SsaOp<T>> {
        let dest = SsaVarId::from_index(0);
        let left = SsaVarId::from_index(1);
        let right = SsaVarId::from_index(2);
        Some(match self.kind {
            ScalarKind::Copy => SsaOp::Copy { dest, src: left },
            ScalarKind::Numeric(NumericOp::Add) => SsaOp::Add {
                dest,
                left,
                right,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::Sub) => SsaOp::Sub {
                dest,
                left,
                right,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::Mul) => SsaOp::Mul {
                dest,
                left,
                right,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::And) => SsaOp::And {
                dest,
                left,
                right,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::Or) => SsaOp::Or {
                dest,
                left,
                right,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::Xor) => SsaOp::Xor {
                dest,
                left,
                right,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::Eq) => SsaOp::Ceq {
                dest,
                left,
                right,
                signaling: false,
            },
            ScalarKind::Numeric(NumericOp::DivS) => SsaOp::Div {
                dest,
                left,
                right,
                unsigned: false,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::DivU) => SsaOp::Div {
                dest,
                left,
                right,
                unsigned: true,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::RemS) => SsaOp::Rem {
                dest,
                left,
                right,
                unsigned: false,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::RemU) => SsaOp::Rem {
                dest,
                left,
                right,
                unsigned: true,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::LtS) => SsaOp::Clt {
                dest,
                left,
                right,
                unsigned: false,
                signaling: false,
            },
            ScalarKind::Numeric(NumericOp::LtU) => SsaOp::Clt {
                dest,
                left,
                right,
                unsigned: true,
                signaling: false,
            },
            ScalarKind::Numeric(NumericOp::GtS) => SsaOp::Cgt {
                dest,
                left,
                right,
                unsigned: false,
                signaling: false,
            },
            ScalarKind::Numeric(NumericOp::GtU) => SsaOp::Cgt {
                dest,
                left,
                right,
                unsigned: true,
                signaling: false,
            },
            ScalarKind::Checked {
                op: NumericOp::Add,
                unsigned,
            } => SsaOp::AddOvf {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            ScalarKind::Checked {
                op: NumericOp::Sub,
                unsigned,
            } => SsaOp::SubOvf {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            ScalarKind::Checked {
                op: NumericOp::Mul,
                unsigned,
            } => SsaOp::MulOvf {
                dest,
                left,
                right,
                unsigned,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::Shl) => SsaOp::Shl {
                dest,
                value: left,
                amount: right,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::Rol) => SsaOp::Rol {
                dest,
                value: left,
                amount: right,
            },
            ScalarKind::Numeric(NumericOp::Ror) => SsaOp::Ror {
                dest,
                value: left,
                amount: right,
            },
            ScalarKind::Numeric(NumericOp::ShrS) => SsaOp::Shr {
                dest,
                value: left,
                amount: right,
                unsigned: false,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::ShrU) => SsaOp::Shr {
                dest,
                value: left,
                amount: right,
                unsigned: true,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::Neg) => SsaOp::Neg {
                dest,
                operand: left,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::Not) => SsaOp::Not {
                dest,
                operand: left,
                flags: None,
            },
            ScalarKind::Numeric(NumericOp::BSwap) => SsaOp::BSwap { dest, src: left },
            ScalarKind::Numeric(NumericOp::BRev) => SsaOp::BRev { dest, src: left },
            ScalarKind::Numeric(NumericOp::BitScanForward) => {
                SsaOp::BitScanForward { dest, src: left }
            }
            ScalarKind::Numeric(NumericOp::BitScanReverse) => {
                SsaOp::BitScanReverse { dest, src: left }
            }
            ScalarKind::Numeric(NumericOp::Popcount) => SsaOp::Popcount { dest, src: left },
            ScalarKind::Numeric(NumericOp::Parity) => SsaOp::Parity { dest, src: left },
            _ => return None,
        })
    }
}

/// Projects one complete machine-specific scalar layout without owning a host type.
fn operand_layout<T: Target>(
    ssa: &SsaFunction<T>,
    value: SsaVarId,
    machine: Machine,
) -> Option<(Destination, OperandLayout)> {
    let ty = ssa.variable(value)?.var_type();
    if T::is_pointer(ty) {
        return None;
    }
    let (destination, boolean) = if let Some(layout) = T::type_layout(ty, machine.ptr_size) {
        if layout.is_runtime_sized() {
            return None;
        }
        (Destination::from_type_layout(layout), layout.is_boolean())
    } else {
        if !T::is_integer(ty) || T::vector_shape(ty).is_some() {
            return None;
        }
        let integer = T::integer_conversion_type(ty, machine.ptr_size)?;
        let bits = T::bit_width(ty).unwrap_or(integer.bits());
        (
            Destination {
                bits: Some(bits),
                lanes: None,
                lane_format: None,
                integer: true,
                integer_type: Some(integer),
            },
            T::is_boolean(ty),
        )
    };
    if !destination.is_integer_scalar() {
        return None;
    }
    Some((
        destination,
        if boolean {
            OperandLayout::Boolean
        } else {
            OperandLayout::Register(destination)
        },
    ))
}

#[cfg(test)]
mod tests {
    use crate::{
        PointerSize,
        analysis::{
            consts::{
                Destination, OperandLayout,
                scalar::{ScalarKind, ScalarOperation},
            },
            symbolic::SymbolicExpr,
        },
        ir::{
            division::DivisionRules,
            evaluation::{ArithmeticTrap, Evaluation, NumericOp},
            function::SsaFunction,
            integer::IntegerType,
            ops::SsaOp,
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    fn operation(kind: ScalarKind, width: u32, signed: bool) -> ScalarOperation {
        let destination = Destination {
            bits: Some(width),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(width, signed),
        };
        ScalarOperation {
            kind,
            machine: MACHINE,
            destination,
            operands: [OperandLayout::Register(destination); 2],
            arity: match kind {
                ScalarKind::Numeric(op) if op.is_unary() => 1,
                ScalarKind::Copy | ScalarKind::Convert { .. } => 1,
                _ => 2,
            },
        }
    }

    #[test]
    fn delayed_division_retains_policy_and_ignores_legacy_pointer_width() {
        let mut descriptor = operation(ScalarKind::Numeric(NumericOp::DivU), 64, false);
        descriptor.machine.division = DivisionRules::RISCV;
        let expr = SymbolicExpr::<MockTarget>::typed_binary(
            descriptor,
            SymbolicExpr::named("input"),
            SymbolicExpr::constant(ConstValue::U64(0)),
        )
        .unwrap();
        assert_eq!(
            expr.evaluate_outcome_with(PointerSize::Bit32, |_| Evaluation::Value(ConstValue::U64(
                7
            ))),
            Evaluation::Value(ConstValue::U64(u64::MAX)),
        );
        assert_eq!(
            expr.evaluate_outcome_with(PointerSize::Bit32, |_| Evaluation::Unknown),
            Evaluation::Value(ConstValue::U64(u64::MAX)),
        );
    }

    #[test]
    fn checked_scalar_faults_and_conversions_remain_outcomes() {
        let checked = operation(
            ScalarKind::Checked {
                op: NumericOp::Add,
                unsigned: false,
            },
            8,
            true,
        );
        assert_eq!(
            checked.evaluate::<MockTarget>(Some(ConstValue::I8(127)), Some(ConstValue::I8(1))),
            Evaluation::Trap(ArithmeticTrap::Overflow),
        );
        let mut conversion = operation(
            ScalarKind::Convert {
                target: IntegerType::new(8, true).unwrap(),
                unsigned: false,
                checked: true,
            },
            8,
            true,
        );
        conversion.operands[0] = operation(ScalarKind::Copy, 32, true).operands[0];
        assert_eq!(
            conversion.evaluate::<MockTarget>(Some(ConstValue::I32(511)), None),
            Evaluation::Trap(ArithmeticTrap::Overflow),
        );
        conversion.kind = ScalarKind::Convert {
            target: IntegerType::new(8, true).unwrap(),
            unsigned: false,
            checked: false,
        };
        assert_eq!(
            conversion.evaluate::<MockTarget>(Some(ConstValue::I32(511)), None),
            Evaluation::Value(ConstValue::I8(-1)),
        );
    }

    #[test]
    fn a_parent_cannot_absorb_a_failed_or_potentially_failing_child() {
        let divide = operation(ScalarKind::Numeric(NumericOp::DivS), 32, true);
        let partial = SymbolicExpr::<MockTarget>::typed_binary(
            divide,
            SymbolicExpr::constant_i32(1),
            SymbolicExpr::constant_i32(0),
        )
        .unwrap();
        let multiply = operation(ScalarKind::Numeric(NumericOp::Mul), 32, true);
        let expression =
            SymbolicExpr::typed_binary(multiply, partial.clone(), SymbolicExpr::constant_i32(0))
                .unwrap();
        assert_eq!(
            expression
                .simplify(PointerSize::Bit64)
                .evaluate_outcome_with(PointerSize::Bit64, |_| Evaluation::Unknown),
            Evaluation::Trap(ArithmeticTrap::DivideByZero),
        );
        let uncertain =
            SymbolicExpr::typed_binary(divide, SymbolicExpr::named("x"), SymbolicExpr::named("y"))
                .unwrap();
        let ambiguous = SymbolicExpr::typed_binary(multiply, partial, uncertain.clone()).unwrap();
        assert_eq!(
            ambiguous.evaluate_outcome_with(PointerSize::Bit64, |_| Evaluation::Unknown),
            Evaluation::Unknown,
        );
        let mut riscv = divide;
        riscv.machine.division = DivisionRules::RISCV;
        let absorbed =
            SymbolicExpr::typed_binary(riscv, uncertain, SymbolicExpr::constant_i32(0)).unwrap();
        assert_eq!(
            absorbed.evaluate_outcome_with(PointerSize::Bit64, |_| Evaluation::Unknown),
            Evaluation::Unknown,
        );
    }

    #[test]
    fn boolean_conversion_and_copy_preserve_interpretation() {
        let mut conversion = operation(
            ScalarKind::Convert {
                target: IntegerType::new(32, true).unwrap(),
                unsigned: false,
                checked: false,
            },
            32,
            true,
        );
        conversion.operands[0] = OperandLayout::Boolean;
        assert_eq!(
            conversion.evaluate::<MockTarget>(Some(ConstValue::True), None),
            Evaluation::Value(ConstValue::I32(1))
        );
        conversion.operands[0] = operation(ScalarKind::Copy, 1, true).operands[0];
        assert_eq!(
            conversion.evaluate::<MockTarget>(ConstValue::of_lane(1, 1), None),
            Evaluation::Value(ConstValue::I32(-1))
        );
        let copy = operation(ScalarKind::Copy, 8, true);
        let copied = copy.evaluate::<MockTarget>(Some(ConstValue::U8(255)), None);
        assert_eq!(copied, Evaluation::Value(ConstValue::I8(-1)));
        let mut widen = conversion;
        widen.operands[0] = copy.operands[0];
        assert_eq!(
            widen.evaluate(copied.into_value(), None),
            Evaluation::Value(ConstValue::I32(-1))
        );
    }

    #[test]
    fn total_unknown_subtrees_retain_normality_without_erasing_checked_children() {
        let total = operation(ScalarKind::Numeric(NumericOp::Add), 64, false);
        let child = SymbolicExpr::<MockTarget>::typed_binary(
            total,
            SymbolicExpr::named("x"),
            SymbolicExpr::constant(ConstValue::U64(1)),
        )
        .unwrap();
        assert!(child.admits_normal_input(PointerSize::Bit64));
        let mut divide = operation(ScalarKind::Numeric(NumericOp::DivU), 64, false);
        divide.machine.division = DivisionRules::RISCV;
        let expression =
            SymbolicExpr::typed_binary(divide, child, SymbolicExpr::constant(ConstValue::U64(0)))
                .unwrap();
        assert_eq!(
            expression.evaluate_outcome_with(PointerSize::Bit64, |_| Evaluation::Unknown),
            Evaluation::Value(ConstValue::U64(u64::MAX))
        );
        let checked = operation(
            ScalarKind::Checked {
                op: NumericOp::Add,
                unsigned: true,
            },
            64,
            false,
        );
        let child = SymbolicExpr::<MockTarget>::typed_binary(
            checked,
            SymbolicExpr::named("x"),
            SymbolicExpr::constant(ConstValue::U64(1)),
        )
        .unwrap();
        assert!(!child.admits_normal_input(PointerSize::Bit64));
        let expression =
            SymbolicExpr::typed_binary(divide, child, SymbolicExpr::constant(ConstValue::U64(0)))
                .unwrap();
        assert_eq!(
            expression.evaluate_outcome_with(PointerSize::Bit64, |_| Evaluation::Unknown),
            Evaluation::Unknown
        );
    }

    #[test]
    fn captured_semantics_reject_float_layouts_and_survive_rebuilding() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let left = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let right = ssa.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let dest = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 2),
            MockType::I32,
        );
        let floating = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 3),
            MockType::F64,
        );
        let source = SsaOp::Add {
            dest,
            left,
            right,
            flags: None,
        };
        let (captured, inputs) = ScalarOperation::capture(&ssa, &source, MACHINE).unwrap();
        assert_eq!(inputs, [Some(left), Some(right)]);
        assert!(
            ScalarOperation::capture(
                &ssa,
                &SsaOp::Add {
                    dest: floating,
                    left,
                    right,
                    flags: None
                },
                MACHINE
            )
            .is_none()
        );
        let expression = SymbolicExpr::<MockTarget>::typed_binary(
            captured,
            SymbolicExpr::variable(left),
            SymbolicExpr::named("right"),
        )
        .unwrap();
        let copy = expression.clone();
        assert_eq!(copy, expression);
        assert_eq!(copy.node_count(), 3);
        assert_eq!(copy.depth(), 1);
        assert!(copy.variables().contains(&left));
        assert!(copy.named_variables().contains("right"));
        let bound = copy
            .substitute(left, &SymbolicExpr::constant_i32(7))
            .substitute_named("right", &SymbolicExpr::constant_i32(4));
        assert_eq!(
            bound.simplify(PointerSize::Bit32).as_constant(),
            Some(&ConstValue::I32(11))
        );
        let mut changed = captured;
        changed.machine.division = DivisionRules::RISCV;
        assert_ne!(captured, changed);
    }
}
