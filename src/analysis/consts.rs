//! Constant evaluation for SSA operations.
//!
//! This module provides unified constant folding capabilities for SSA analysis.
//! The [`ConstEvaluator`] can be used by multiple passes (unflattening, decryption,
//! SCCP, etc.) to evaluate SSA operations to constant values.
//!
//! # Algorithm
//!
//! Evaluation is demand-driven and depth-limited:
//!
//! 1. **Cache check**: Results are memoized per variable for O(1) repeated queries.
//! 2. **Cycle detection**: A `BitSet` tracks variables currently being evaluated.
//!    If a variable is encountered while already on the recursion stack, evaluation
//!    returns `None` (avoiding infinite recursion through phi cycles).
//! 3. **Depth limiting**: A configurable maximum recursion depth (default 20) prevents
//!    stack overflow on long dependency chains.
//! 4. **Operation dispatch**: [`Machine::evaluate`] handles all pure arithmetic,
//!    bitwise, comparison, overflow-checked, and conversion operations by resolving
//!    operands through a caller-provided closure.
//!
//! ## Operations Handled
//!
//! - Arithmetic: Add, Sub, Mul, Div, Rem, Neg
//! - Bitwise: And, Or, Xor, Not, Shl, Shr
//! - Comparisons: Ceq, Clt, Cgt (signed + unsigned)
//! - Overflow-checked: AddOvf, SubOvf, MulOvf
//! - Type conversions: Conv (checked and unchecked)
//! - Copy propagation: trace-through to source
//! - Constants: direct return of Const value
//!
//! ## Operations NOT Handled
//!
//! - Calls, loads, stores, and other side-effecting operations always return `None`.
//! - `Copy` is resolved by [`ConstEvaluator`], not by [`Machine::evaluate`], which
//!   reads operands rather than definitions (callers handle it first).
//!
//! # Complexity
//!
//! O(d) per variable where d is the depth of the definition chain (bounded by
//! `max_depth`). Caching ensures each variable is evaluated at most once.
//!
//! # Example
//!
//! ```rust
//! use analyssa::{
//!     analysis::{ConstEvaluator, Machine},
//!     ir::{ConstValue, FloatRules, SsaVarId},
//!     testing, PointerSize,
//! };
//!
//! let ssa = testing::const_i32_return(42);
//! let machine = Machine::new(PointerSize::Bit64, None, FloatRules::default());
//! let mut evaluator = ConstEvaluator::new(&ssa, machine);
//! let state_var = SsaVarId::from_index(0);
//!
//! // Inject known values from external analysis
//! evaluator.set_known(state_var, ConstValue::I32(42));
//!
//! // Evaluate a variable
//! if let Some(value) = evaluator.evaluate_var(state_var) {
//!     println!("Variable evaluates to: {:?}", value);
//! }
//!
//! // Get all computed constants
//! let constants = evaluator.into_results();
//! ```

use std::{cell::RefCell, cmp::Ordering, collections::BTreeMap};

use smallvec::SmallVec;

pub use self::raises::Mask as PredicateMask;
use crate::{
    PointerSize,
    ir::{
        division::DivisionRules,
        evaluation::{ArithmeticTrap, Evaluation},
        float::{FloatExceptions, FloatRounding, FloatRules},
        function::SsaFunction,
        integer::BitWidth,
        ops::{
            CmpKind, FlagsMask, FloatFunctionKind, Kernel, LaneExecution, LaneOperand, SsaOp,
            SystemOpKind, TrapOp, VectorStatus,
        },
        value::{CheckedArithmetic, ConstValue},
        variable::SsaVarId,
    },
    target::{Endianness, Target},
};

mod bits;

mod raises;
/// Captured scalar integer semantics for delayed evaluation.
pub mod scalar;

mod evaluator;
mod fold;
mod known;
mod layout;
mod types;
mod vectors;

pub use self::{
    evaluator::ConstEvaluator,
    fold::comparison_stable,
    layout::{Destination, OperandLayout},
    types::{ReinterpretShape, TypeQueries},
};
use self::{
    fold::{as_lanes, fold_comparison, fold_op},
    known::{KNOWN_BITS_DEPTH, KnownBits},
    layout::{integer_carrier, scalar_zero},
};

/// The constants an operation of several results folds to, each with the
/// variable it defines, in the order the operation states them
/// ([`Machine::evaluate_results`]).
pub type FoldedResults<T> = SmallVec<[(SsaVarId, ConstValue<T>); 3]>;

/// The value or control result of checked integer arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckedEvaluation<T: Target> {
    /// The operation returns this exact value without integer overflow.
    Value(ConstValue<T>),
    /// At least one operand is unknown, so overflow is not decided.
    Unknown,
    /// Known integer operands overflow the operation's target width.
    Overflow,
}

/// What constant evaluation, and every analysis that computes as the target
/// does, knows of the machine a function runs on.
///
/// The pointer width fixes the wraparound of native-size integers and whether
/// an address sum is an integer sum at its width. The byte order says which of
/// two adjacent words is a value's low half. The float rules say what the
/// target's float operations give where IEEE leaves the result to the
/// processor, by which a constant conversion out of range or an invalid
/// operation folds; a rule the target does not fix is stated as none, and the
/// operation it decides folds nothing.
///
/// Scalar integer division rules state exceptional values or faults explicitly.
/// Missing rules establish neither. A host supplies the machine when it builds
/// its [`AnalysisContext`], and every
/// analysis reads them from there ([`AnalysisContext::machine`]).
///
/// [`AnalysisContext`]: crate::analysis::AnalysisContext
/// [`AnalysisContext::machine`]: crate::analysis::AnalysisContext::machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Machine {
    /// The target's pointer width.
    pub ptr_size: PointerSize,
    /// The target's byte order, where known.
    pub byte_order: Option<Endianness>,
    /// What the target's float operations give where IEEE leaves it to the
    /// processor.
    pub float: FloatRules,
    /// Exceptional scalar integer division and remainder outcomes.
    pub division: DivisionRules,
    /// Whether an ordinary access whose address is not aligned to its size
    /// faults; an access stating the alignment it requires
    /// ([`AccessQualifiers::required`]) faults without it on any machine.
    ///
    /// [`AccessQualifiers::required`]: crate::ir::ops::AccessQualifiers::required
    pub misaligned: MisalignedAccess,
}

/// Whether an ordinary access whose address is not aligned to its size
/// faults on a machine ([`Machine::misaligned`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MisalignedAccess {
    /// It may fault, so an access is proven not to only at its natural
    /// alignment: MIPS raises an address error, RISC-V may trap, and a
    /// machine that states no rule is taken to.
    #[default]
    Faults,
    /// It completes: x86, and AArch64 and ARMv7 ordinary loads and stores to
    /// normal memory, which need no alignment.
    Tolerated,
}

/// Operands whose scalar integer width is the operation's result width.
#[derive(Clone, Copy)]
enum IntegerOperands {
    /// Every numeric operand of arithmetic or a bitwise binary operation.
    All,
    /// The data operand of a unary operation, shift or rotate.
    Value(SsaVarId),
}

impl IntegerOperands {
    /// Returns the width-preserving integer inputs, leaving counts at their own width.
    fn of<T: Target>(op: &SsaOp<T>) -> Option<Self> {
        match op {
            SsaOp::Add { .. }
            | SsaOp::Sub { .. }
            | SsaOp::Mul { .. }
            | SsaOp::Div { .. }
            | SsaOp::Rem { .. }
            | SsaOp::And { .. }
            | SsaOp::Or { .. }
            | SsaOp::Xor { .. } => Some(Self::All),
            SsaOp::Neg { operand, .. } | SsaOp::Not { operand, .. } => Some(Self::Value(*operand)),
            SsaOp::Shl { value, .. }
            | SsaOp::Shr { value, .. }
            | SsaOp::Rol { value, .. }
            | SsaOp::Ror { value, .. } => Some(Self::Value(*value)),
            SsaOp::BSwap { src, .. } | SsaOp::BRev { src, .. } => Some(Self::Value(*src)),
            _ => None,
        }
    }

    /// Returns whether this input is read at the operation width.
    fn contains(self, value: SsaVarId) -> bool {
        match self {
            Self::All => true,
            Self::Value(operand) => operand == value,
        }
    }
}

impl Machine {
    /// Creates a machine with unstated exceptional integer division and the supplied width and float rules.
    ///
    /// Native hosts install their architecture's rules with [`Self::with_division_rules`].
    #[must_use]
    pub const fn new(
        ptr_size: PointerSize,
        byte_order: Option<Endianness>,
        float: FloatRules,
    ) -> Self {
        Self {
            ptr_size,
            byte_order,
            float,
            division: DivisionRules::UNSTATED,
            misaligned: MisalignedAccess::Faults,
        }
    }

    /// Installs the target's exceptional scalar integer division rules.
    #[must_use]
    pub const fn with_division_rules(mut self, rules: DivisionRules) -> Self {
        self.division = rules;
        self
    }

    /// Installs whether the target's ordinary misaligned accesses fault
    /// ([`Self::misaligned`]).
    #[must_use]
    pub const fn with_misaligned_access(mut self, rule: MisalignedAccess) -> Self {
        self.misaligned = rule;
        self
    }

    /// Evaluates an SSA operation to a constant value using the provided
    /// operand resolver.
    ///
    /// This is the shared arithmetic dispatch for constant evaluation. It
    /// handles all pure arithmetic, bitwise, comparison, overflow-checked, and
    /// conversion operations. Callers provide a `get_const` closure that
    /// resolves an [`SsaVarId`] to its constant value (if known).
    ///
    /// A float-to-integer conversion of a constant its integer does not hold
    /// folds by [`FloatRules::out_of_range`], and an invalid float operation
    /// of numbers (infinity times zero) to [`FloatRules::default_nan`], and
    /// one of a NaN operand to the NaN [`FloatRules::nan_operands`] picks,
    /// where this machine states them; where it does not, they fold nothing.
    ///
    /// # Operations not handled
    ///
    /// `Copy` requires variable-level resolution (trace-through), not
    /// arithmetic, so callers handle it before calling this. Calls, loads,
    /// stores, and other side-effecting operations always return `None`.
    ///
    /// # Destination
    ///
    /// Width-preserving scalar integer operations read their numeric operands
    /// at the declared result width, extending primitive carriers by signedness
    /// and treating packed bitvectors as one integer. Count operands keep their
    /// own width. Vector destinations retain independent lanes. Without an
    /// attested integer destination, operands retain their carrier widths and
    /// an integer result is kept only when it fits the declared width.
    /// Primitive integer literals obey the scalar destination width; packed
    /// literals retain lane geometry for consumers that explicitly read lanes.
    ///
    /// Checked add, subtract and multiply normalize operand carriers to the
    /// declared scalar width before evaluating overflow. With no declared width,
    /// they use the wider operand. See [`Machine::evaluate_checked`].
    ///
    /// A vector result is as wide as its destination and laid out in its lanes
    /// ([`Destination::lanes`]): the value a vector variable holds is laid out
    /// as its type says, whatever lanes the operation computed in, so an
    /// operation reading it at the variable's lanes reads them.
    ///
    /// A float result written to an integer destination is its bits
    /// ([`Destination::integer`]): a reduction or an extract of float lanes
    /// computes a float, and a target whose lanes carry no float type declares
    /// the lane, and the value its variable holds, as an integer of the width.
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA operation to evaluate.
    /// * `dest` - What `op`'s destination declares, if known; see
    ///   [`Destination::of_operation`]. A bare width (`Option<u32>`, the
    ///   destination's [`SsaFunction::bit_width`]) states no lanes.
    /// * `get_const` - Closure that resolves a variable to its constant value.
    ///
    /// # Returns
    ///
    /// The constant result if all operands resolve, the operation succeeds and
    /// the result fits the destination, `None` otherwise. Scalar integer literals
    /// and width-preserving operations read numeric carriers at their declared
    /// width; shifts and rotates retain the count operand's own width.
    pub fn evaluate<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
    ) -> Option<ConstValue<T>> {
        self.evaluate_with_types(op, dest, get_const, &TypeQueries::ordinary(self.ptr_size))
    }

    /// Evaluates with the type queries `types` answers.
    fn evaluate_with_types<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        types: &TypeQueries<'_, T>,
    ) -> Option<ConstValue<T>> {
        let dest = dest.into();
        if let Some(outcome) = self.evaluate_control_with_types(op, dest, &mut get_const, types) {
            // An operation that defines nothing has no value, though its
            // control is decided: a trap whose condition is false.
            return op.dest().and(outcome.into_value());
        }
        let integer_width = (dest.integer && dest.lanes.is_none())
            .then_some(dest.bits)
            .flatten()
            .filter(|bits| (1..=128).contains(bits));
        let operands = IntegerOperands::of(op);
        // A fixed-width integer carrier states its operand's width. One of
        // another width is an ill-typed operand of a width-preserving
        // operation, which the kernel refuses; only width-free carriers
        // (native, packed, boolean) are read at the operation's width.
        let get_const = |value| {
            let constant = get_const(value)?;
            Some(match (integer_width, operands) {
                (Some(width), Some(operands))
                    if operands.contains(value)
                        && !(constant.is_integer()
                            && !matches!(
                                constant,
                                ConstValue::NativeInt(_) | ConstValue::NativeUInt(_)
                            )
                            && constant.bit_width(self.ptr_size) != Some(width)) =>
                {
                    self.integer_carrier(constant, width)
                }
                _ => constant,
            })
        };
        let value = fold_op(op, dest, get_const, self.ptr_size, self.float, types)?;
        let value = if matches!(op, SsaOp::Copy { .. }) {
            match dest.integer_type {
                Some(layout) => value.with_sign(layout.signed()),
                None => value,
            }
        } else {
            value
        };
        if matches!(op, SsaOp::Const { .. }) {
            self.normalize_literal(value, dest)
        } else {
            self.normalize_result(value, dest)
        }
    }

    /// Evaluates an operation with the attested layout of each input.
    ///
    /// Numeric operations read scalar floating inputs at their declared format.
    /// Comparisons normalize integer inputs at their own declared widths;
    /// boolean literals supply zero or one in declared integer inputs. Their
    /// predicate result cannot specify the input width. Explicit vector layouts
    /// retain their geometry. Conversion source formats override inferred types.
    /// Layouts are requested only for numeric operands, and resolving a constant
    /// remains a single read. Result widths follow [`Self::evaluate`].
    ///
    /// # Arguments
    ///
    /// * `op` - The operation to evaluate.
    /// * `dest` - Its result layout.
    /// * `get_const` - Resolves an operand to its owned constant.
    /// * `layout_for` - Returns the operand's declared layout, or the default
    ///   layout when the caller has no type declaration.
    pub fn evaluate_with_operand_layouts<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        layout_for: impl FnMut(SsaVarId) -> OperandLayout,
    ) -> Option<ConstValue<T>> {
        self.evaluate_with_type_queries(
            op,
            dest,
            get_const,
            layout_for,
            &TypeQueries::ordinary(self.ptr_size),
        )
    }

    /// Evaluates typed operands and operation targets under one inspection policy.
    pub fn evaluate_with_type_queries<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        mut layout_for: impl FnMut(SsaVarId) -> OperandLayout,
        types: &TypeQueries<'_, T>,
    ) -> Option<ConstValue<T>> {
        if matches!(op, SsaOp::FloatRaise { .. }) {
            let destination = dest.into();
            let value = raises::fold(
                op,
                destination,
                &mut |variable| Some((get_const(variable)?, layout_for(variable))),
                self.ptr_size,
                self.float,
                types,
            )?;
            return self.normalize_result(value, destination);
        }
        if let SsaOp::Compute(data) = op
            && data.kind.lanes.is_some()
            && data.kind.data_outputs() == 1
        {
            // A computation over lanes computes the elements it executes,
            // its data result the one evaluated beside any status it writes,
            // reading each vector at the element width its type declares and
            // each input only where an element needs it.
            let inputs = self.lane_operands(&data.inputs, &mut get_const, &mut layout_for);
            let execution =
                LaneExecution::of(data.kind, &inputs, T::environment_rounding(), self.float)?;
            return self.lane_data(&execution, data.kind.kernel, dest.into());
        }
        if matches!(op, SsaOp::Div { .. } | SsaOp::Rem { .. }) {
            return self
                .evaluate_control_with_type_queries(op, dest, get_const, layout_for, types)?
                .into_value();
        }
        if matches!(
            op,
            SsaOp::Ceq { .. } | SsaOp::Clt { .. } | SsaOp::Cgt { .. }
        ) {
            let value = ConstValue::from_bool(fold_comparison(
                op,
                |variable| {
                    let constant = get_const(variable)?;
                    layout_for(variable).read_comparison(constant, self.ptr_size)
                },
                self.ptr_size,
                self.float,
            )?);
            self.fits(&value, dest.into().bits).then_some(value)
        } else {
            let numeric = matches!(
                op,
                SsaOp::Add { .. }
                    | SsaOp::Sub { .. }
                    | SsaOp::Mul { .. }
                    | SsaOp::Div { .. }
                    | SsaOp::Rem { .. }
                    | SsaOp::Neg { .. }
                    | SsaOp::IntConv { .. }
                    | SsaOp::FloatConv { .. }
                    | SsaOp::FloatToInt { .. }
                    | SsaOp::IntToFloat { .. }
                    | SsaOp::Ckfinite { .. }
            );
            let source_format = match op {
                SsaOp::FloatToInt { source, .. } | SsaOp::FloatConv { source, .. } => *source,
                _ => None,
            };
            self.evaluate_with_types(
                op,
                dest,
                |variable| {
                    let value = get_const(variable)?;
                    if let Some(format) = source_format {
                        OperandLayout::Float(format).read_numeric(value, self.ptr_size)
                    } else if matches!(op, SsaOp::IntConv { .. } | SsaOp::IntToFloat { .. }) {
                        layout_for(variable).read_conversion(value, self.ptr_size)
                    } else if numeric {
                        layout_for(variable).read_numeric(value, self.ptr_size)
                    } else {
                        Some(value)
                    }
                },
                types,
            )
        }
    }

    /// Evaluates division using normalized integer carriers or the machine's floating rules.
    fn division_outcome<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: Destination,
        mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        types: &TypeQueries<'_, T>,
    ) -> Option<Evaluation<ConstValue<T>>> {
        let (left_id, right_id, unsigned, remainder) = match op {
            SsaOp::Div {
                left,
                right,
                unsigned,
                ..
            } => (*left, *right, *unsigned, false),
            SsaOp::Rem {
                left,
                right,
                unsigned,
                ..
            } => (*left, *right, *unsigned, true),
            _ => return None,
        };
        let (left, right) = (get_const(left_id), get_const(right_id));
        let normalize = |value: ConstValue<T>| {
            if dest.integer
                && dest.lanes.is_none()
                && let Some(width) = dest.bits
            {
                if value.is_integer()
                    && !matches!(value, ConstValue::NativeInt(_) | ConstValue::NativeUInt(_))
                    && value.bit_width(self.ptr_size) != Some(width)
                {
                    return None;
                }
                Some(self.integer_carrier(value, width))
            } else {
                Some(value)
            }
        };
        let normalize = |value: Option<ConstValue<T>>| {
            value.map(|value| normalize(value).ok_or(())).transpose()
        };
        let (Ok(left), Ok(right)) = (normalize(left), normalize(right)) else {
            return Some(Evaluation::Unsupported);
        };
        if dest.integer
            && dest.lanes.is_none()
            && let Some(layout) = dest.integer_type
            && right
                .as_ref()
                .and_then(|right| right.scalar_integer_pattern(self.ptr_size))
                == Some((0, layout.bits()))
        {
            let failure = if remainder {
                self.division.remainder_zero
            } else {
                self.division.quotient_zero
            };
            return Some(failure.evaluate(
                left.as_ref(),
                layout,
                ArithmeticTrap::DivideByZero,
                self.ptr_size,
            ));
        }
        let (Some(left), Some(right)) = (left, right) else {
            return Some(Evaluation::Unknown);
        };
        if dest.lanes.is_none()
            && left.scalar_integer_pattern(self.ptr_size).is_some()
            && right.scalar_integer_pattern(self.ptr_size).is_some()
        {
            return Some(left.division_outcome(
                &right,
                unsigned,
                remainder,
                self.ptr_size,
                self.division,
            ));
        }
        Some(
            fold_op(
                op,
                dest,
                |value| {
                    if value == left_id {
                        Some(left.clone())
                    } else if value == right_id {
                        Some(right.clone())
                    } else {
                        None
                    }
                },
                self.ptr_size,
                self.float,
                types,
            )
            .map_or(Evaluation::Unsupported, Evaluation::Value),
        )
    }

    /// Evaluates explicit arithmetic checks, scalar integer conversions and
    /// conditional traps.
    ///
    /// The result separates exact values, proven faults, runtime uncertainty and
    /// unsupported value or type domains. A conditional trap whose condition
    /// holds is a fault ([`ArithmeticTrap::Condition`]); one whose condition
    /// fails is its condition, false, and does nothing; one a flag or memory
    /// decides is unknown. No outcome is returned for a software interrupt,
    /// which always transfers, or for other operation families. Floating-to-integer conversion uses truncation and
    /// the machine's out-of-range policy; finite checks preserve exact bits.
    pub fn evaluate_control<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
    ) -> Option<Evaluation<ConstValue<T>>> {
        self.evaluate_control_with_types(op, dest, get_const, &TypeQueries::ordinary(self.ptr_size))
    }

    /// Evaluates arithmetic control using the caller's target layouts.
    fn evaluate_control_with_types<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        types: &TypeQueries<'_, T>,
    ) -> Option<Evaluation<ConstValue<T>>> {
        let dest = dest.into();
        let outcome = match op {
            SsaOp::Div { .. } | SsaOp::Rem { .. } => {
                self.division_outcome(op, dest, get_const, types)?
            }
            // A conditional trap fires where its inputs satisfy its
            // condition, and does nothing where they do not: its value is
            // its condition, false. One that fires on a flag or on bounds in
            // memory (`into`, `bound`) is undecided here; only a software
            // interrupt always transfers, as a system call does.
            SsaOp::SystemOp(data) => {
                let SystemOpKind::Trap { op: trap, .. } = data.kind else {
                    return None;
                };
                if trap == TrapOp::SoftwareInterrupt {
                    return None;
                }
                let Some((cmp, unsigned)) = trap.condition() else {
                    return Some(Evaluation::Unknown);
                };
                let (&left, &right) = (data.inputs.first()?, data.inputs.get(1)?);
                match (get_const(left), get_const(right)) {
                    (Some(left), Some(right)) => {
                        match self.compare_values(&left, &right, cmp, unsigned) {
                            Some(true) => Evaluation::Trap(ArithmeticTrap::Condition),
                            Some(false) => Evaluation::Value(ConstValue::False),
                            None => Evaluation::Unknown,
                        }
                    }
                    _ => Evaluation::Unknown,
                }
            }
            SsaOp::AddOvf { .. } | SsaOp::SubOvf { .. } | SsaOp::MulOvf { .. } => {
                match self.evaluate_checked(op, dest.bits, get_const) {
                    Some(CheckedEvaluation::Value(value)) => Evaluation::Value(value),
                    Some(CheckedEvaluation::Unknown) => Evaluation::Unknown,
                    Some(CheckedEvaluation::Overflow) => Evaluation::Trap(ArithmeticTrap::Overflow),
                    None => Evaluation::Unsupported,
                }
            }
            SsaOp::IntConv {
                operand,
                target,
                unsigned,
                overflow_check,
                ..
            } => {
                let Some(value) = get_const(*operand) else {
                    return Some(Evaluation::Unknown);
                };
                match types.integer_type(target) {
                    Some(layout) => {
                        layout.convert(&value, *unsigned, *overflow_check, self.ptr_size)
                    }
                    None => Evaluation::Unsupported,
                }
            }
            SsaOp::FloatToInt {
                operand,
                target,
                unsigned,
                overflow_check,
                source,
                ..
            } => {
                let Some(value) = get_const(*operand) else {
                    return Some(Evaluation::Unknown);
                };
                let float = match source {
                    Some(format) => value.as_float(*format, self.ptr_size),
                    None => value.read_float(self.ptr_size),
                };
                let Some(float) = float else {
                    return Some(Evaluation::Unsupported);
                };
                let Some(layout) = types
                    .integer_type(target)
                    .filter(|layout| layout.signed() != *unsigned)
                else {
                    return Some(Evaluation::Unsupported);
                };
                let Some(converted) = float.to_integer_clamped(
                    layout.bits(),
                    layout.signed(),
                    false,
                    FloatRounding::TowardZero,
                ) else {
                    return Some(Evaluation::Unsupported);
                };
                let invalid = converted.exceptions.contains(FloatExceptions::INVALID);
                let bits = if invalid && *overflow_check {
                    return Some(Evaluation::Trap(ArithmeticTrap::Overflow));
                } else if invalid {
                    let Some(rule) = self.float.out_of_range else {
                        return Some(Evaluation::Unknown);
                    };
                    let Some(bits) = rule.convert(float, layout.bits(), layout.signed()) else {
                        return Some(Evaluation::Unsupported);
                    };
                    bits
                } else {
                    converted.bits
                };
                layout
                    .constant(bits, self.ptr_size)
                    .map_or(Evaluation::Unsupported, Evaluation::Value)
            }
            SsaOp::Ckfinite { operand, .. } => {
                let Some(value) = get_const(*operand) else {
                    return Some(Evaluation::Unknown);
                };
                match value.read_float(self.ptr_size) {
                    Some(float) if float.is_finite() => Evaluation::Value(ConstValue::Float(float)),
                    Some(_) => Evaluation::Trap(ArithmeticTrap::NonFinite),
                    None => Evaluation::Unsupported,
                }
            }
            _ => return None,
        };
        Some(match outcome {
            Evaluation::Value(value) => self
                .normalize_result(value, dest)
                .map_or(Evaluation::Unsupported, Evaluation::Value),
            other => other,
        })
    }

    /// Evaluates control outcomes using each operand's attested numeric layout.
    ///
    /// A known layout mismatch is unsupported, rather than an unknown input or
    /// an overflow proof. Explicit floating source formats override inferred
    /// layouts, and integer conversions retain their source register width.
    pub fn evaluate_control_with_operand_layouts<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        layout_for: impl FnMut(SsaVarId) -> OperandLayout,
    ) -> Option<Evaluation<ConstValue<T>>> {
        self.evaluate_control_with_type_queries(
            op,
            dest,
            get_const,
            layout_for,
            &TypeQueries::ordinary(self.ptr_size),
        )
    }

    /// Reads control operands and targets with the caller's shared inspection policy.
    pub fn evaluate_control_with_type_queries<T: Target>(
        self,
        op: &SsaOp<T>,
        dest: impl Into<Destination>,
        mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        mut layout_for: impl FnMut(SsaVarId) -> OperandLayout,
        types: &TypeQueries<'_, T>,
    ) -> Option<Evaluation<ConstValue<T>>> {
        let dest = dest.into();
        let mut mismatch = false;
        let outcome = self.evaluate_control_with_types(
            op,
            dest,
            |variable| {
                let Some(value) = get_const(variable) else {
                    if matches!(op, SsaOp::Div { .. } | SsaOp::Rem { .. }) && dest.integer {
                        mismatch |= !matches!(layout_for(variable), OperandLayout::Register(input)
                            if input.integer && input.lanes.is_none() && input.bits == dest.bits);
                    }
                    return None;
                };
                let format = match op {
                    SsaOp::FloatToInt { source, .. } => *source,
                    _ => None,
                };
                let layout = format.map_or_else(|| layout_for(variable), OperandLayout::Float);
                let result = if matches!(op, SsaOp::IntConv { .. }) {
                    layout.read_conversion(value, self.ptr_size)
                } else {
                    layout.read_numeric(value, self.ptr_size)
                };
                mismatch |= result.is_none();
                result
            },
            types,
        )?;
        Some(if mismatch {
            Evaluation::Unsupported
        } else {
            outcome
        })
    }

    /// Normalizes a copied literal using its declared register and result layout.
    ///
    /// Packed literals retain their lane geometry for explicit lane consumers.
    pub fn normalize_literal<T: Target>(
        self,
        value: ConstValue<T>,
        dest: Destination,
    ) -> Option<ConstValue<T>> {
        let value = match (dest.integer, dest.lanes, dest.bits) {
            (true, None, Some(width)) if !matches!(&value, ConstValue::Vector(_)) => {
                self.integer_carrier(value, width)
            }
            _ => value,
        };
        self.normalize_result(value, dest)
    }

    /// Fits a computed value and applies its declared vector and register view.
    #[inline(always)]
    pub fn normalize_result<T: Target>(
        self,
        value: ConstValue<T>,
        dest: Destination,
    ) -> Option<ConstValue<T>> {
        if !self.fits(&value, dest.bits) {
            return None;
        }
        let needs_view = match (&value, dest.lanes) {
            (ConstValue::Vector(vector), Some((lane_bits, lanes))) => {
                vector.lane_bits() != lane_bits || vector.lanes() != lanes
            }
            (ConstValue::Float(_), _) => dest.integer,
            _ => false,
        };
        if needs_view {
            self.apply_result_view(value, dest)
        } else {
            Some(value)
        }
    }

    /// Applies a changed lane shape or floating-register representation after width admission.
    #[inline(never)]
    fn apply_result_view<T: Target>(
        self,
        value: ConstValue<T>,
        dest: Destination,
    ) -> Option<ConstValue<T>> {
        Some(match (value, dest.lanes) {
            (ConstValue::Vector(vector), Some((lane_bits, lanes))) => {
                ConstValue::Vector(vector.resized(lane_bits, lanes)?)
            }
            (ConstValue::Float(float), _) if dest.integer => {
                let value = ConstValue::of_lane(float.format().bits(), float.bits())?;
                match dest.bits {
                    Some(width) => self.integer_carrier(value, width),
                    None => value,
                }
            }
            (value, _) => value,
        })
    }

    /// Compares constants only when the machine preserves their floating inputs.
    ///
    /// Unordered floating relations use `unsigned` for ordered predicates;
    /// equality is false and inequality true for a NaN. Possible subnormal
    /// flushing withholds the decision, while x87 extended values remain exact.
    pub fn compare_values<T: Target>(
        self,
        left: &ConstValue<T>,
        right: &ConstValue<T>,
        cmp: CmpKind,
        unsigned: bool,
    ) -> Option<bool> {
        if !comparison_stable(left, self.float) || !comparison_stable(right, self.float) {
            return None;
        }
        if let (ConstValue::Float(left), ConstValue::Float(right)) = (left, right) {
            if left.format() != right.format() {
                return None;
            }
            let order = left.compare(*right);
            return Some(match cmp {
                CmpKind::Eq => order == Some(Ordering::Equal),
                CmpKind::Ne => order != Some(Ordering::Equal),
                _ if order.is_none() => unsigned,
                CmpKind::Lt => order == Some(Ordering::Less),
                CmpKind::Le => order != Some(Ordering::Greater),
                CmpKind::Gt => order == Some(Ordering::Greater),
                CmpKind::Ge => order != Some(Ordering::Less),
            });
        }
        left.relation(right, cmp, unsigned, self.ptr_size)
    }

    /// Returns zero at the declared scalar width, or at the native width when unstated.
    pub fn integer_zero<T: Target>(self, bits: Option<u32>) -> Option<ConstValue<T>> {
        scalar_zero(bits.unwrap_or_else(|| self.ptr_size.bits()))
    }

    /// Reads a numeric carrier at an attested scalar integer width.
    ///
    /// Packed bitvectors supply their low bits as one integer. Primitive
    /// integers extend by their signedness; native values obey pointer width.
    /// Other constant classes retain their value and lane operations retain
    /// their own operand geometry.
    fn integer_carrier<T: Target>(self, value: ConstValue<T>, width: u32) -> ConstValue<T> {
        integer_carrier(value, width, self.ptr_size)
    }

    /// Evaluates checked add, subtract or multiply, distinguishing overflow from unknown inputs.
    ///
    /// A declared destination sets the operation width, independently of the
    /// constants' storage variants. Otherwise the wider operand sets it.
    /// Primitive operands extend by their own signedness before their patterns
    /// are read as signed or unsigned for the check. With a declared width,
    /// packed bitvectors supply one scalar pattern. Without it, non-scalar
    /// layouts return `None`, as do other operations. Overflow is a control
    /// outcome, not an unknown value.
    pub fn evaluate_checked<T: Target>(
        self,
        op: &SsaOp<T>,
        dest_width: Option<u32>,
        mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
    ) -> Option<CheckedEvaluation<T>> {
        let (left, right, unsigned, kind) = match op {
            SsaOp::AddOvf {
                left,
                right,
                unsigned,
                ..
            } => (*left, *right, *unsigned, CheckedArithmetic::Add),
            SsaOp::SubOvf {
                left,
                right,
                unsigned,
                ..
            } => (*left, *right, *unsigned, CheckedArithmetic::Sub),
            SsaOp::MulOvf {
                left,
                right,
                unsigned,
                ..
            } => (*left, *right, *unsigned, CheckedArithmetic::Mul),
            _ => return None,
        };
        if dest_width.is_some_and(|width| width == 0 || width > 128) {
            return None;
        }
        let mut operand = |var| {
            let value = get_const(var)?;
            Some(match dest_width {
                Some(width) => self.integer_carrier(value, width),
                None => value,
            })
        };
        let (left, right) = (operand(left), operand(right));
        let mut inferred = None;
        for known in [left.as_ref(), right.as_ref()].into_iter().flatten() {
            let (_, width) = known.scalar_integer_pattern(self.ptr_size)?;
            inferred = Some(inferred.map_or(width, |prior: u32| prior.max(width)));
        }
        let (Some(left), Some(right)) = (left, right) else {
            return Some(CheckedEvaluation::Unknown);
        };
        let width = dest_width.or(inferred)?;
        let left = left.scalar_integer_at_width(width, self.ptr_size)?;
        let right = right.scalar_integer_at_width(width, self.ptr_size)?;
        let result = left.checked_arithmetic(&right, unsigned, self.ptr_size, kind);
        Some(result.map_or(CheckedEvaluation::Overflow, CheckedEvaluation::Value))
    }

    /// The inputs of a computation over lanes, each the constant
    /// `get_const` resolves read as `layout_for` declares it: a vector with
    /// its elements' width and float format, a scalar with its own format
    /// ([`LaneOperand::of`]); `None` where it is not known.
    fn lane_operands<T: Target>(
        self,
        inputs: &[SsaVarId],
        mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        mut layout_for: impl FnMut(SsaVarId) -> OperandLayout,
    ) -> Vec<Option<LaneOperand>> {
        inputs
            .iter()
            .map(|input| {
                let (lane_bits, format) = match layout_for(*input) {
                    OperandLayout::Register(Destination {
                        lanes: Some((bits, _)),
                        lane_format,
                        ..
                    }) => (Some(bits), lane_format),
                    OperandLayout::Float(format) => (None, Some(format)),
                    OperandLayout::Register(_) | OperandLayout::Boolean => (None, None),
                };
                LaneOperand::of(&get_const(*input)?, lane_bits, format, self.ptr_size)
            })
            .collect()
    }

    /// The data result `execution` writes to `destination`: the vector its
    /// elements make, or the scalar a mask summary writes, a count or an
    /// index or all ones.
    fn lane_data<T: Target>(
        self,
        execution: &LaneExecution<'_>,
        kernel: Kernel,
        destination: Destination,
    ) -> Option<ConstValue<T>> {
        let bits = destination.bits?;
        if matches!(kernel, Kernel::MaskSummary { .. }) {
            let value = execution.summary()? & BitWidth::new(bits).checked_mask_u128()?;
            return ConstValue::integer(bits, false, u64::try_from(value).ok()?.cast_signed());
        }
        self.normalize_result(ConstValue::Vector(execution.evaluate(bits)?), destination)
    }

    /// Evaluates wide division without discarding definite arithmetic faults.
    ///
    /// Returns `None` for other operation families. Unknown operands and
    /// unsupported carrier widths remain distinct from proven traps.
    pub fn evaluate_results_control<T: Target>(
        self,
        op: &SsaOp<T>,
        mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
    ) -> Option<Evaluation<FoldedResults<T>>> {
        let SsaOp::WideDiv {
            quotient,
            remainder,
            high,
            low,
            divisor,
            unsigned,
        } = op
        else {
            return None;
        };
        let (low, high, divisor) = (get_const(*low), get_const(*high), get_const(*divisor));
        if divisor
            .as_ref()
            .and_then(|value| value.integer_pattern(self.ptr_size))
            .is_some_and(|(value, width)| value == 0 && (1..=64).contains(&width))
        {
            return Some(Evaluation::Trap(ArithmeticTrap::DivideByZero));
        }
        let (Some(low), Some(high), Some(divisor)) = (low, high, divisor) else {
            return Some(Evaluation::Unknown);
        };
        Some(
            low.wide_division_outcome(&high, &divisor, *unsigned, self.ptr_size)
                .map(|(low, high)| SmallVec::from_iter([(*quotient, low), (*remainder, high)])),
        )
    }

    /// Evaluates an operation that defines more than one value: one constant
    /// per value it defines that folds.
    ///
    /// [`Self::evaluate`] answers for an operation's one result; the
    /// double-width arithmetic defines two, and both are known exactly when the
    /// operands are. [`SsaOp::WideMul`] defines the low and high halves of its
    /// product ([`ConstValue::wide_mul`]), [`SsaOp::WideDiv`] the quotient and
    /// remainder of its double-width dividend ([`ConstValue::wide_div`]). Of
    /// the float functions, `fxtract` defines the exponent and the significand
    /// ([`FloatBits::extracted`]), and `fprem` and `fprem1` the remainder,
    /// the condition codes as an x87 compare's flags lay them out (`C2` clear
    /// at the parity flag for a complete reduction, the quotient's bit 2 at
    /// the carry flag and bit 1 at the zero flag) and the quotient's bit 0
    /// (`C1`) ([`FloatBits::partial_remainder`]).
    ///
    /// [`FloatBits::extracted`]: crate::ir::float::FloatBits::extracted
    /// [`FloatBits::partial_remainder`]: crate::ir::float::FloatBits::partial_remainder
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA operation to evaluate.
    /// * `get_const` - Closure that resolves a variable to its constant value.
    ///
    /// # Returns
    ///
    /// Each defined variable that folds with its constant, in the order the
    /// operation states them: every one of them, except that a computation
    /// over lanes, which reads its operands as their types declare them
    /// ([`Self::evaluate_results_with_operand_layouts`]), folds none here.
    /// `None` for an operation of one result, an operand that is not a
    /// constant, or arithmetic without a value (division by zero, a quotient
    /// too wide, a reduction left incomplete).
    pub fn evaluate_results<T: Target>(
        self,
        op: &SsaOp<T>,
        get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
    ) -> Option<FoldedResults<T>> {
        self.evaluate_results_with_operand_layouts(op, get_const, |_| OperandLayout::default())
    }

    /// [`Self::evaluate_results`] of operands, and results, each read as
    /// `layout_for` declares it.
    ///
    /// A computation over lanes defines its data result and its status (the
    /// exceptions or the saturation its executed elements report), each
    /// listed where it folds and left out where not: an element whose value
    /// its policies leave unspecified folds no data result, yet the status,
    /// which reads only the executed elements, folds all the same
    /// ([`LaneExecution::status`]). A value it leaves out is not a constant
    /// of the operands. Every other operation lists all of its results or
    /// none.
    pub fn evaluate_results_with_operand_layouts<T: Target>(
        self,
        op: &SsaOp<T>,
        mut get_const: impl FnMut(SsaVarId) -> Option<ConstValue<T>>,
        mut layout_for: impl FnMut(SsaVarId) -> OperandLayout,
    ) -> Option<FoldedResults<T>> {
        if !op.folds_results() {
            return None;
        }
        match op {
            SsaOp::WideMul {
                low,
                high,
                left,
                right,
                unsigned,
            } => {
                let (low_value, high_value) =
                    get_const(*left)?.wide_mul(&get_const(*right)?, *unsigned, self.ptr_size)?;
                Some(SmallVec::from_iter([
                    (*low, low_value),
                    (*high, high_value),
                ]))
            }
            SsaOp::FloatFunction(data) => match (data.kind, data.outputs.as_slice()) {
                (FloatFunctionKind::Extract, [exponent, significand]) => {
                    let [value] = data.inputs.as_slice() else {
                        return None;
                    };
                    let (exponent_value, significand_value) =
                        get_const(*value)?.read_float(self.ptr_size)?.extracted()?;
                    Some(SmallVec::from_iter([
                        (*exponent, ConstValue::Float(exponent_value)),
                        (*significand, ConstValue::Float(significand_value)),
                    ]))
                }
                (
                    kind @ (FloatFunctionKind::Rem | FloatFunctionKind::Rem1),
                    [remainder, codes, quotient_bit],
                ) => {
                    let [value, divisor] = data.inputs.as_slice() else {
                        return None;
                    };
                    let (remainder_value, quotient) = get_const(*value)?
                        .read_float(self.ptr_size)?
                        .partial_remainder(
                            get_const(*divisor)?.read_float(self.ptr_size)?,
                            kind == FloatFunctionKind::Rem1,
                        )?;
                    let mut flags = 0u16;
                    if quotient & 0b100 != 0 {
                        flags |= FlagsMask::CARRY.bits();
                    }
                    if quotient & 0b010 != 0 {
                        flags |= FlagsMask::ZERO.bits();
                    }
                    Some(SmallVec::from_iter([
                        (*remainder, ConstValue::Float(remainder_value)),
                        (*codes, ConstValue::U32(u32::from(flags))),
                        (*quotient_bit, ConstValue::U8(quotient & 1)),
                    ]))
                }
                _ => None,
            },
            // A kernel over a whole register of several results: a result of
            // one lane is the integer the lane holds (a mask register).
            SsaOp::Compute(data) if data.kind.lanes.is_none() => {
                let values = data
                    .inputs
                    .iter()
                    .map(|input| get_const(*input))
                    .collect::<Option<Vec<_>>>()?;
                let results = data.kind.kernel.evaluate_registers(
                    &as_lanes(&values, self.ptr_size)?,
                    data.kind.element.bits,
                )?;
                if results.len() != data.outputs.len() {
                    return None;
                }
                data.outputs
                    .iter()
                    .zip(results)
                    .map(|(output, result)| {
                        let value = if result.lanes() == 1 {
                            ConstValue::of_lane(result.lane_bits(), result.lane(0))?
                        } else {
                            ConstValue::Vector(result)
                        };
                        Some((*output, value))
                    })
                    .collect()
            }
            SsaOp::Compute(data) => {
                if data.kind.data_outputs() != 1 {
                    return None;
                }
                let inputs = self.lane_operands(&data.inputs, &mut get_const, &mut layout_for);
                let execution =
                    LaneExecution::of(data.kind, &inputs, T::environment_rounding(), self.float)?;
                let (output, rest) = data.outputs.split_first()?;
                let destination = match layout_for(*output) {
                    OperandLayout::Register(destination) => destination,
                    OperandLayout::Float(format) => Destination::from(Some(format.bits())),
                    OperandLayout::Boolean => Destination::from(Some(1)),
                };
                let mut results = FoldedResults::new();
                if let Some(value) = self.lane_data(&execution, data.kind.kernel, destination) {
                    results.push((*output, value));
                }
                // The status is the last result, after any fault or flags.
                if data.kind.status != VectorStatus::None
                    && let Some(status) = rest.last()
                    && let OperandLayout::Register(Destination {
                        bits: Some(width), ..
                    }) = layout_for(*status)
                    && let Some(bits) = destination.bits
                    && let Some(value) = execution.status(bits)
                    && let Some(value) = match width {
                        1 => Some(ConstValue::from_bool(value != 0)),
                        8 | 16 | 32 | 64 => u64::try_from(value).ok().and_then(|value| {
                            ConstValue::integer(width, false, value.cast_signed())
                        }),
                        _ => ConstValue::of_lane(width, value),
                    }
                {
                    results.push((*status, value));
                }
                (!results.is_empty()).then_some(results)
            }
            SsaOp::WideDiv {
                quotient,
                remainder,
                high,
                low,
                divisor,
                unsigned,
            } => {
                let (quotient_value, remainder_value) = get_const(*low)?.wide_div(
                    &get_const(*high)?,
                    &get_const(*divisor)?,
                    *unsigned,
                    self.ptr_size,
                )?;
                Some(SmallVec::from_iter([
                    (*quotient, quotient_value),
                    (*remainder, remainder_value),
                ]))
            }
            _ => None,
        }
    }

    /// The constant `op`'s result is when its bits are known though the
    /// values it reads are not all constants.
    ///
    /// A vector a lane of which is not known (an argument's, a NaN product's of
    /// another target, the upper quadword `extrq` leaves undefined) is no
    /// constant, so [`Self::evaluate`] folds nothing of it; the bits read out
    /// of it fold here. A value's bits are known when it is a constant (in
    /// `constants`) or its definition takes them from known bits:
    ///
    /// - a lane of a lane-wise operation (`VectorBinary`, `VectorUnary`,
    ///   `VectorTernary`) is that operation on its operands' lane, and a
    ///   bitwise one (`and`, `not`, ...) computes each bit of its operands'
    ///   same bit, so only the bits asked for need be known;
    /// - a scalar operation computes lane zero and keeps its first operand's
    ///   other lanes;
    /// - a select's lane is the lane of the operand its mask lane's top bit
    ///   picks;
    /// - a permute's lane is the table lane its known index selects;
    /// - a reshape or bit field takes its bits from the values it builds on
    ///   ([`reshape::known_slice`](crate::analysis::reshape::known_slice)).
    ///
    /// The search follows a bounded depth of definitions.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The function, for the definitions.
    /// * `op` - The candidate operation.
    /// * `constants` - Known constant values.
    ///
    /// # Returns
    ///
    /// The result, of its destination's type, or `None` when it is not known,
    /// wider than 128 bits, or `op` is a constant already or reads only
    /// constants (which [`Self::evaluate`] decides).
    #[must_use]
    pub fn fold_known_bits<T: Target>(
        self,
        ssa: &SsaFunction<T>,
        op: &SsaOp<T>,
        constants: &BTreeMap<SsaVarId, ConstValue<T>>,
    ) -> Option<ConstValue<T>> {
        if matches!(op, SsaOp::Const { .. })
            || op.defs().count() != 1
            || op.uses().iter().all(|var| constants.contains_key(var))
        {
            return None;
        }
        let dest = op.dest()?;
        let dest_type = ssa.variable(dest)?.var_type();
        let width = T::bit_width(dest_type)?;
        let computed = KnownBits {
            machine: self,
            ssa,
            constants,
            memo: RefCell::default(),
        }
        .defined(op, 0, width, KNOWN_BITS_DEPTH)?;
        // The bits as the destination's type reads them; a width of no
        // whole bytes is the unsigned lane of that width.
        match usize::try_from(width / 8) {
            Ok(bytes) if width % 8 == 0 => ConstValue::from_typed_bytes(
                computed.to_le_bytes().get(..bytes)?,
                dest_type,
                Endianness::Little,
            ),
            _ => ConstValue::of_lane(width, computed),
        }
    }

    /// Whether `value` can be the result written to a destination `dest_width`
    /// bits wide: an integer or a vector exactly that wide. Other constants and
    /// unknown widths always can.
    #[must_use]
    pub fn fits<T: Target>(self, value: &ConstValue<T>, dest_width: Option<u32>) -> bool {
        let width = match value {
            ConstValue::Vector(vector) => Some(vector.bits()),
            _ => value.bit_width(self.ptr_size),
        };
        match (dest_width, width) {
            (Some(expected), Some(actual)) => expected == actual,
            _ => true,
        }
    }
}

/// Folds a lane operation of constant vectors, as [`VectorConst`]'s lane
/// semantics compute it.
///
/// An operand that is no vector is one lane of its width, as a register a
/// scalar instruction writes is the low lane of its vector. A lane operation
/// on scalars only ([`VectorElement::scalar`] of scalar operands) gives a
/// scalar: the float of its format, or the unsigned integer of its width.
///
/// # Returns
///
/// `None` for an operation that is no lane operation, which the scalar folds
/// take; `Some(None)` for a lane operation that does not fold; and
/// `Some(Some(value))` for one that folds to `value`.
#[allow(clippy::option_option)] // "not a lane operation" differs from "no fold"
#[cfg(test)]
mod tests {
    use std::{borrow::Cow, cell::Cell, slice};

    use super::*;
    use crate::{
        ir::{
            DefaultNan, FloatBits, FloatRules, NanPropagation, VariableOrigin,
            block::SsaBlock,
            float::{FloatClassLayout, FloatExceptionLayout, FloatFormat, FloatRounding},
            instruction::SsaInstruction,
            integer::IntegerType,
            lanes::VectorConst,
            ops::{
                CarrylessPart, ComplexProduct, ComplexShape, Computation, Contraction,
                CrcPolynomial, DotShape, LaneControl, ScalarCipherKind, SveComputeKind,
                VectorBinaryKind, VectorCompareKind, VectorElement, VectorElementKind,
                VectorExtendInLaneData, VectorInactivePolicy, VectorMaskMode, VectorPackKind,
                VectorPredicateLayout, VectorPredicateWhileData, VectorReduceKind, VectorRounding,
                VectorShuffleSelector, VectorStatus, VectorSveComputeData, VectorTernaryKind,
                VectorUnaryKind,
            },
            variable::DefSite,
        },
        target::{
            Endianness, TypeLayout,
            VectorShuffleLane::{Left, Right, Zero},
            VectorShuffleMask, VectorTailPolicy,
        },
        testing::{self, MACHINE, MockTarget, MockType, function_of},
    };

    type Cv = ConstValue<MockTarget>;

    fn var(index: usize) -> SsaVarId {
        SsaVarId::from_index(index)
    }

    fn resolve(id: SsaVarId) -> Option<Cv> {
        match id.index() {
            0 => Some(Cv::I32(10)),
            1 => Some(Cv::I32(3)),
            2 => Some(Cv::I32(-1)),
            3 => Some(Cv::I32(i32::MAX)),
            _ => None,
        }
    }

    /// **A copy is its source's constant, in its destination's type.** A copy
    /// restates its source's bits as a bitcast does: of one width, an
    /// `int32_t` `1` copied into a `uint32_t` is the unsigned `1`, and `-1` is
    /// `0xffff_ffff`. A source wider than the destination, or none known,
    /// folds to nothing.
    #[test]
    fn a_copy_is_its_sources_constant() {
        let copy = SsaOp::<MockTarget>::Copy {
            dest: var(1),
            src: var(0),
        };
        let fold = |width: Option<u32>, source: Cv| {
            MACHINE.evaluate(&copy, width, |id: SsaVarId| {
                (id.index() == 0).then(|| source.clone())
            })
        };
        assert_eq!(fold(Some(32), Cv::I32(1)), Some(Cv::I32(1)), "same width");
        assert_eq!(
            fold(None, Cv::I32(-1)),
            Some(Cv::I32(-1)),
            "no width stated"
        );
        assert_eq!(
            fold(Some(32), Cv::I64(1 << 40)),
            None,
            "wider than its destination"
        );
        assert_eq!(
            MACHINE.evaluate(&copy, Some(32), |_: SsaVarId| None::<Cv>),
            None,
            "an unknown source"
        );
        let unsigned = Destination {
            bits: Some(32),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(32, false),
        };
        let restated = |source: Cv| {
            MACHINE.evaluate(&copy, unsigned, |id: SsaVarId| {
                (id.index() == 0).then(|| source.clone())
            })
        };
        assert_eq!(
            restated(Cv::I32(1)),
            Some(Cv::U32(1)),
            "in the destination's sign"
        );
        assert_eq!(
            restated(Cv::I32(-1)),
            Some(Cv::U32(u32::MAX)),
            "the same bits"
        );
    }

    /// A `while` of constant bounds folds to the elements it sets, bit `e`
    /// element `e`'s: `whilelo` from 5 to 8 over eight elements the first
    /// three, `whilele` up to the greatest signed bound the three counters up
    /// to it (the next exceeds it rather than wrapping to the least, which
    /// would set every element), `whilelt` from a bound already reached none,
    /// and the counting down `whilege` (SVE2) the last elements, down to the
    /// bound.
    #[test]
    fn a_while_of_constant_bounds_folds_to_the_elements_it_sets() {
        let fold = |kind: VectorCompareKind, unsigned: bool, first: Cv, last: Cv| {
            let op =
                SsaOp::<MockTarget>::VectorPredicateWhile(Box::new(VectorPredicateWhileData {
                    kind,
                    unsigned,
                    elements: 8,
                    outputs: vec![var(2)],
                    inputs: vec![var(0), var(1)],
                }));
            MACHINE.evaluate(&op, Some(8), |id: SsaVarId| match id.index() {
                0 => Some(first.clone()),
                1 => Some(last.clone()),
                _ => None,
            })
        };
        assert_eq!(
            fold(VectorCompareKind::Lt, true, Cv::U64(5), Cv::U64(8)),
            Some(Cv::U8(0b0000_0111))
        );
        assert_eq!(
            fold(
                VectorCompareKind::Le,
                false,
                Cv::I32(i32::MAX - 2),
                Cv::I32(i32::MAX)
            ),
            Some(Cv::U8(0b0000_0111))
        );
        assert_eq!(
            fold(VectorCompareKind::Lt, false, Cv::I64(3), Cv::I64(-1)),
            Some(Cv::U8(0))
        );
        assert_eq!(
            fold(VectorCompareKind::Ge, false, Cv::I64(10), Cv::I64(8)),
            Some(Cv::U8(0b1110_0000))
        );
        assert_eq!(
            fold(VectorCompareKind::Lt, true, Cv::U32(u32::MAX), Cv::U32(1)),
            Some(Cv::U8(0)),
            "an unsigned base above the bound"
        );
    }

    /// A compress packs the lanes its mask selects low and an expand spreads
    /// its low lanes over them, the others the passthrough's under a merging
    /// mask and zero under a zeroing one, whether the mask is a bit per lane
    /// or a vector of the lanes.
    #[test]
    fn a_compress_or_expand_folds_to_the_lanes_its_mask_selects() {
        let pack = |kind: VectorPackKind, mode: VectorMaskMode| SsaOp::<MockTarget>::VectorPack {
            dest: var(3),
            value: var(0),
            mask: var(1),
            passthrough: Some(var(2)),
            vector_type: MockType::Unknown,
            element_bits: 32,
            kind,
            mode,
        };
        let fold = |op: &SsaOp<MockTarget>, mask: Cv| {
            MACHINE.evaluate(op, None, |id: SsaVarId| match id.index() {
                0 => VectorConst::from_lanes(32, [10, 11, 12, 13]).map(Cv::Vector),
                1 => Some(mask.clone()),
                2 => VectorConst::from_lanes(32, [90, 91, 92, 93]).map(Cv::Vector),
                _ => None,
            })
        };
        let lanes = |values: [u128; 4]| VectorConst::from_lanes(32, values).map(Cv::Vector);
        let bits = Cv::U8(0b1010);
        let vector =
            VectorConst::from_lanes(32, [0, u128::from(u32::MAX), 0, u128::from(u32::MAX)])
                .map(Cv::Vector)
                .unwrap();
        for mask in [bits, vector] {
            assert_eq!(
                fold(
                    &pack(VectorPackKind::Compress, VectorMaskMode::Merge),
                    mask.clone()
                ),
                lanes([11, 13, 92, 93])
            );
            assert_eq!(
                fold(
                    &pack(VectorPackKind::Compress, VectorMaskMode::Zero),
                    mask.clone()
                ),
                lanes([11, 13, 0, 0])
            );
            assert_eq!(
                fold(&pack(VectorPackKind::Expand, VectorMaskMode::Merge), mask),
                lanes([90, 10, 92, 11])
            );
        }
    }

    /// An in-lane extension widens each lane's low bits, and `cnot` is one
    /// where a lane is zero.
    #[test]
    fn a_lane_extension_or_logical_not_folds_lane_by_lane() {
        let source = |id: SsaVarId| match id.index() {
            0 => VectorConst::from_lanes(32, [0x1122_3344, 0x0000_00F0, 0, 0x8000]).map(Cv::Vector),
            _ => None,
        };
        let lanes = |values: [u128; 4]| VectorConst::from_lanes(32, values).map(Cv::Vector);
        let extend = |signed: bool| {
            SsaOp::<MockTarget>::VectorExtendInLane(Box::new(VectorExtendInLaneData {
                signed,
                source_bits: 8,
                element_bits: 32,
                outputs: vec![var(1)],
                inputs: vec![var(0)],
            }))
        };
        assert_eq!(
            MACHINE.evaluate(&extend(true), None, source),
            lanes([0x44, 0xFFFF_FFF0, 0, 0])
        );
        assert_eq!(
            MACHINE.evaluate(&extend(false), None, source),
            lanes([0x44, 0xF0, 0, 0])
        );
        let not = SsaOp::<MockTarget>::VectorSveCompute(Box::new(VectorSveComputeData {
            op: SveComputeKind::LogicalNot,
            element_bits: 32,
            rotation: 0,
            outputs: vec![var(1)],
            inputs: vec![var(0)],
        }));
        assert_eq!(MACHINE.evaluate(&not, None, source), lanes([0, 0, 1, 0]));
    }

    /// SVE2's carrying adds and subtracts work pair by pair: the even element
    /// sums the accumulator, the bottom or top source element (complemented
    /// for a subtract) and the carry in bit 0 of the odd element of the
    /// second source, and the odd element takes the carry out.
    #[test]
    fn a_carrying_add_or_subtract_folds_pair_by_pair() {
        let carry = |op: SveComputeKind| {
            SsaOp::<MockTarget>::VectorSveCompute(Box::new(VectorSveComputeData {
                op,
                element_bits: 32,
                rotation: 0,
                outputs: vec![var(3)],
                inputs: vec![var(0), var(1), var(2)],
            }))
        };
        let fold = |op: SveComputeKind| {
            MACHINE.evaluate(&carry(op), None, |id: SsaVarId| match id.index() {
                0 => VectorConst::from_lanes(32, [0xFFFF_FFFF, 7, 5, 7]).map(Cv::Vector),
                1 => VectorConst::from_lanes(32, [1, 2, 3, 4]).map(Cv::Vector),
                2 => VectorConst::from_lanes(32, [0, 1, 0, 0]).map(Cv::Vector),
                _ => None,
            })
        };
        let lanes = |values: [u128; 4]| VectorConst::from_lanes(32, values).map(Cv::Vector);
        assert_eq!(fold(SveComputeKind::AddCarryBottom), lanes([1, 1, 8, 0]));
        assert_eq!(fold(SveComputeKind::AddCarryTop), lanes([2, 1, 9, 0]));
        assert_eq!(
            fold(SveComputeKind::SubCarryBottom),
            lanes([0xFFFF_FFFE, 1, 1, 1])
        );
    }

    /// SVE2's `sclamp` and `uclamp` bound each lane of the accumulator by
    /// the first source below and the second above: of `[10, -10, 3, 0x80]`
    /// between `-5` and `5`, signed, `[5, -5, 3, 5]`; unsigned between 2 and
    /// 9 the negative lanes are large, `[9, 9, 3, 9]`.
    #[test]
    fn a_clamp_bounds_each_lane_as_signed_or_unsigned() {
        let clamp = |op: SveComputeKind, low: u128, high: u128| {
            let op = SsaOp::<MockTarget>::VectorSveCompute(Box::new(VectorSveComputeData {
                op,
                element_bits: 32,
                rotation: 0,
                outputs: vec![var(3)],
                inputs: vec![var(0), var(1), var(2)],
            }));
            MACHINE.evaluate(&op, None, |id: SsaVarId| match id.index() {
                0 => VectorConst::from_lanes(32, [10, 0xFFFF_FFF6, 3, 0x80]).map(Cv::Vector),
                1 => VectorConst::splat(32, 4, low).map(Cv::Vector),
                2 => VectorConst::splat(32, 4, high).map(Cv::Vector),
                _ => None,
            })
        };
        let lanes = |values: [u128; 4]| VectorConst::from_lanes(32, values).map(Cv::Vector);
        assert_eq!(
            clamp(SveComputeKind::ClampSigned, 0xFFFF_FFFB, 5),
            lanes([5, 0xFFFF_FFFB, 3, 5])
        );
        assert_eq!(
            clamp(SveComputeKind::ClampUnsigned, 2, 9),
            lanes([9, 9, 3, 9])
        );
    }

    /// SVE2's `bdep`, `bext` and `bgrp` deposit, extract and group bits
    /// within each lane, as `pdep` and `pext` do in a scalar
    /// ([`Kernel::evaluate_lanes`]): `bgrp` of `0x0003` by `0x0101` puts the
    /// selected bit 0 lowest and the unselected bit 1 above the two selected
    /// ones, `0b101`.
    #[test]
    fn a_lane_bit_deposit_extract_or_group_folds_to_its_bits() {
        let lanes = |kernel: Kernel| {
            SsaOp::<MockTarget>::compute(
                Computation::whole(
                    kernel,
                    VectorElement::integer(16, None),
                    VectorRounding::None,
                ),
                vec![var(2)],
                vec![var(0), var(1)],
            )
        };
        let vectors = |id: SsaVarId| match id.index() {
            0 => VectorConst::from_lanes(16, [0b11, 0xF0F0]).map(Cv::Vector),
            1 => VectorConst::from_lanes(16, [0x0101, 0xFF00]).map(Cv::Vector),
            _ => None,
        };
        for (kernel, expected) in [
            (Kernel::BitDeposit, [0x0101, 0xF000]),
            (Kernel::BitExtract, [0b01, 0xF0]),
            (Kernel::BitGroup, [0b101, 0xF0F0]),
        ] {
            assert_eq!(
                MACHINE.evaluate(&lanes(kernel), Some(32), vectors),
                VectorConst::from_lanes(16, expected).map(Cv::Vector),
                "{kernel:?}"
            );
        }
    }

    /// A kernel over a whole register of constants folds: a ternary logic
    /// function of one result through [`Machine::evaluate`], and an
    /// intersection's two masks, each the integer of its register, through
    /// [`Machine::evaluate_results`].
    #[test]
    fn a_whole_register_kernel_of_constants_folds_each_result() {
        let words = |values: [u128; 4]| VectorConst::from_lanes(32, values).map(Cv::Vector);
        let inputs = |id: SsaVarId| match id.index() {
            0 => words([1, 2, 3, 4]),
            1 => words([4, 4, 9, 1]),
            2 => words([0xff, 0, 0xff, 0]),
            _ => None,
        };
        let computation = |kernel: Kernel, outputs: Vec<SsaVarId>, inputs: Vec<SsaVarId>| {
            SsaOp::<MockTarget>::compute(
                Computation::whole(
                    kernel,
                    VectorElement::integer(32, None),
                    VectorRounding::None,
                ),
                outputs,
                inputs,
            )
        };
        let xor = computation(
            Kernel::TernaryLogic { table: 0x96 },
            vec![var(3)],
            vec![var(0), var(1), var(2)],
        );
        assert_eq!(
            MACHINE.evaluate(&xor, Some(128), inputs),
            words([1 ^ 4 ^ 0xff, 2 ^ 4, 3 ^ 9 ^ 0xff, 4 ^ 1])
        );
        let intersect = computation(
            Kernel::Intersect,
            vec![var(3), var(4)],
            vec![var(0), var(1)],
        );
        assert!(intersect.folds_results());
        assert_eq!(
            MACHINE
                .evaluate_results(&intersect, inputs)
                .map(|results| results.to_vec()),
            Some(vec![(var(3), Cv::U64(0b1001)), (var(4), Cv::U64(0b1011))])
        );
    }

    /// The whole carry-less product of two quadwords (`pclmulqdq` of the
    /// halves it selects) folds to the 128 bits no integer holds, as a vector
    /// of one lane: `0x8000_0000_0000_0001` squared is `x^126 + 1`.
    #[test]
    fn a_whole_quadword_carryless_product_folds_to_its_128_bits() {
        let op = SsaOp::<MockTarget>::compute(
            Computation::whole(
                Kernel::CarrylessMultiply {
                    part: CarrylessPart::Full,
                },
                VectorElement::integer(64, None).as_scalar(),
                VectorRounding::None,
            ),
            vec![var(2)],
            vec![var(0), var(1)],
        );
        let factor = |id: SsaVarId| (id.index() < 2).then_some(Cv::U64(0x8000_0000_0000_0001));
        assert_eq!(
            MACHINE.evaluate(&op, Some(128), factor),
            VectorConst::from_lanes(128, [(1u128 << 126) | 1]).map(Cv::Vector)
        );
        assert_eq!(MACHINE.evaluate(&op, Some(64), factor), None);
    }

    /// A mixed-width fold keeps the wider operand's width only when the
    /// destination has it.
    ///
    /// `add` into a 32-bit destination with a 64-bit constant wraps at 32 bits
    /// on the machine; a result typed as the 64-bit operand carries a bit the
    /// destination cannot hold, and would decide a sign test the other way.
    #[test]
    fn a_mixed_width_fold_must_fit_its_destination() {
        let op: SsaOp<MockTarget> = SsaOp::Add {
            dest: SsaVarId::from_index(2),
            left: SsaVarId::from_index(0),
            right: SsaVarId::from_index(1),
            flags: None,
        };
        let operands = |var: SsaVarId| match var.index() {
            0 => Some(ConstValue::I32(0x7FFF_FFFF)),
            1 => Some(ConstValue::I64(1)),
            _ => None,
        };
        assert_eq!(MACHINE.evaluate(&op, Some(32), operands), None);
        assert_eq!(
            MACHINE.evaluate(&op, Some(64), operands),
            Some(ConstValue::I64(0x8000_0000))
        );
        assert_eq!(
            MACHINE.evaluate(&op, None, operands),
            Some(ConstValue::I64(0x8000_0000)),
            "a destination of unknown width keeps the fold"
        );
    }

    /// A rotate folds whatever its count's variant: a 64-bit count, as a
    /// 64-bit register supplies one, is read modulo the width.
    #[test]
    fn a_rotate_folds_a_wide_count() {
        let operands = |var: SsaVarId| match var.index() {
            0 => Some(ConstValue::U64(1)),
            1 => Some(ConstValue::U64(63)),
            _ => None,
        };
        let rol = SsaOp::<MockTarget>::Rol {
            dest: SsaVarId::from_index(2),
            value: SsaVarId::from_index(0),
            amount: SsaVarId::from_index(1),
        };
        let ror = SsaOp::<MockTarget>::Ror {
            dest: SsaVarId::from_index(2),
            value: SsaVarId::from_index(0),
            amount: SsaVarId::from_index(1),
        };
        assert_eq!(
            MACHINE.evaluate(&rol, Some(64), operands),
            Some(ConstValue::U64(1 << 63))
        );
        assert_eq!(
            MACHINE.evaluate(&ror, Some(64), operands),
            Some(ConstValue::U64(2))
        );
    }

    /// A scalar float operation folds where every rounding mode gives the same
    /// answer. A directed round is exact and keeps a zero's sign
    /// (`trunc(-0.5)` is `-0.0`), and ties go where its direction says;
    /// negation and absolute value change only the sign bit, a NaN's
    /// included. A round by the current mode, a square root and a round of a
    /// NaN are left alone.
    #[test]
    fn a_scalar_float_operation_folds_where_every_rounding_mode_agrees() {
        let fold = |kind: VectorUnaryKind, value: Cv| {
            let op = SsaOp::<MockTarget>::VectorUnary {
                dest: var(1),
                value: var(0),
                kind,
                element: VectorElement {
                    kind: VectorElementKind::Float,
                    bits: 64,
                    scalar: true,
                    signed: None,
                    rounding: None,
                },
                environment: None,
            };
            MACHINE.evaluate(&op, None, |id| (id == var(0)).then(|| value.clone()))
        };
        let bits = |value: Option<Cv>| value.as_ref().and_then(Cv::as_f64).map(f64::to_bits);
        assert_eq!(
            bits(fold(VectorUnaryKind::Truncate, Cv::from(-0.5f64))),
            Some((-0.0f64).to_bits())
        );
        assert_eq!(
            fold(VectorUnaryKind::Floor, Cv::from(-1.5f64)),
            Some(Cv::from(-2.0f64))
        );
        assert_eq!(
            fold(VectorUnaryKind::Ceil, Cv::from(1.25f64)),
            Some(Cv::from(2.0f64))
        );
        assert_eq!(
            fold(VectorUnaryKind::RoundNearestEven, Cv::from(2.5f64)),
            Some(Cv::from(2.0f64))
        );
        assert_eq!(
            fold(VectorUnaryKind::RoundNearestAway, Cv::from(2.5f64)),
            Some(Cv::from(3.0f64))
        );
        assert_eq!(
            bits(fold(VectorUnaryKind::Neg, Cv::from(0.0f64))),
            Some((-0.0f64).to_bits())
        );
        assert_eq!(
            bits(fold(
                VectorUnaryKind::Abs,
                Cv::from(f64::from_bits(0xfff8_0000_0000_0001))
            )),
            Some(0x7ff8_0000_0000_0001)
        );
        assert_eq!(
            fold(VectorUnaryKind::Round, Cv::from(2.5f64)),
            None,
            "the current mode"
        );
        assert_eq!(
            fold(VectorUnaryKind::Sqrt, Cv::from(2.0f64)),
            None,
            "the current mode"
        );
        assert_eq!(
            fold(VectorUnaryKind::Truncate, Cv::from(f64::NAN)),
            None,
            "a NaN"
        );
        // `sqrt(4)` and `rint(2)` are exact, so every mode agrees on them.
        assert_eq!(
            fold(VectorUnaryKind::Sqrt, Cv::from(4.0f64)),
            Some(Cv::from(2.0f64))
        );
        assert_eq!(
            fold(VectorUnaryKind::Round, Cv::from(2.0f64)),
            Some(Cv::from(2.0f64))
        );
        assert_eq!(
            fold(VectorUnaryKind::AbsArithmetic, Cv::from(-2.0f64)),
            Some(Cv::from(2.0f64))
        );
        assert_eq!(
            bits(fold(VectorUnaryKind::NegArithmetic, Cv::from(0.0f64))),
            Some((-0.0f64).to_bits())
        );
        for kind in [
            VectorUnaryKind::NegArithmetic,
            VectorUnaryKind::AbsArithmetic,
        ] {
            assert_eq!(
                fold(kind, Cv::from(f64::NAN)),
                None,
                "{kind:?} propagates a NaN as arithmetic"
            );
        }
        // An integer operand is the float's bits.
        assert_eq!(
            fold(VectorUnaryKind::Truncate, Cv::U64(2.5f64.to_bits())),
            Some(Cv::from(2.0f64))
        );
    }

    /// A conversion between two float formats that states no rounding
    /// rounds by the program's mode, so it folds where every mode gives one
    /// value: `1/3` as a double is exactly `0x3ffd_aaaa_aaaa_aaaa_a800` in x87
    /// extended (the double's 53 bits under an explicit integer bit), but
    /// `0x3eab` in bfloat16 only to nearest, which an unstated rounding does
    /// not fold. A NaN does not fold.
    #[test]
    fn a_conversion_between_float_formats_folds() {
        let convert = |target: MockType, value: Cv| {
            let op = SsaOp::<MockTarget>::FloatConv {
                dest: var(1),
                operand: var(0),
                target,
                rounding: None,
                source: None,
                environment: None,
            };
            MACHINE.evaluate(&op, None, |id| (id == var(0)).then(|| value.clone()))
        };
        let third = Cv::from(1.0 / 3.0);
        assert_eq!(convert(MockType::BF16, third.clone()), None);
        assert_eq!(
            convert(MockType::BF16, Cv::from(1.5)),
            Cv::float(FloatFormat::BFloat16, 0x3fc0)
        );
        assert_eq!(
            convert(MockType::F80, third),
            Cv::float(FloatFormat::X87Extended, 0x3ffd_aaaa_aaaa_aaaa_a800)
        );
        assert_eq!(convert(MockType::BF16, Cv::from(f64::NAN)), None);
    }

    /// **An operation rounding by the environment folds by the mode it
    /// reads.** `1/3` as a double narrowed to a bfloat16 is `0x3eaa` with the
    /// mode operand 1 (toward zero) or 2 (down), `0x3eab` with 0 (nearest) or 3
    /// (up); the lane division `1 / 3` of doubles is `0x3fd5_5555_5555_5555`
    /// toward zero and `..56` up. A mode the operand does not know folds only
    /// what every mode agrees on, and a mode no rule names (5) folds nothing.
    #[test]
    fn an_operation_rounding_by_the_environment_folds_by_its_mode() {
        let mode = var(7);
        let narrow = |known: Option<i32>| {
            let op = SsaOp::<MockTarget>::FloatConv {
                dest: var(1),
                operand: var(0),
                target: MockType::BF16,
                rounding: None,
                source: None,
                environment: Some(mode),
            };
            MACHINE.evaluate(&op, None, |id| {
                if id == var(0) {
                    Some(Cv::from(1.0 / 3.0))
                } else if id == mode {
                    known.map(Cv::I32)
                } else {
                    None
                }
            })
        };
        let bfloat = |bits| Cv::float(FloatFormat::BFloat16, bits);
        for (known, bits) in [(1, 0x3eaa), (2, 0x3eaa), (0, 0x3eab), (3, 0x3eab)] {
            assert_eq!(narrow(Some(known)), bfloat(bits), "mode {known}");
        }
        assert_eq!(narrow(None), None);
        assert_eq!(narrow(Some(5)), None);
        let divide = |known: i32| {
            let op = SsaOp::<MockTarget>::VectorBinary {
                dest: var(2),
                left: var(0),
                right: var(1),
                kind: VectorBinaryKind::Div,
                element: VectorElement::float(64).as_scalar(),
                environment: Some(mode),
            };
            MACHINE.evaluate(&op, None, |id| {
                if id == var(0) {
                    Some(Cv::from(1.0))
                } else if id == var(1) {
                    Some(Cv::from(3.0))
                } else if id == mode {
                    Some(Cv::I32(known))
                } else {
                    None
                }
            })
        };
        assert_eq!(
            divide(1),
            Some(Cv::from(f64::from_bits(0x3fd5_5555_5555_5555)))
        );
        assert_eq!(
            divide(3),
            Some(Cv::from(f64::from_bits(0x3fd5_5555_5555_5556)))
        );
    }

    /// **A conversion that states its rounding folds by it.** `1/3` as a double
    /// narrowed to a bfloat16 is `0x3eaa` toward zero or down and `0x3eab` up,
    /// to nearest or to odd (the truncation is inexact, so its last bit is set);
    /// `257` as an `int` is `256` (`0x4380`) toward zero, down and to nearest
    /// (a tie, to even) and `258` (`0x4381`) up, while an integer rounds no way
    /// to odd and does not fold.
    #[test]
    fn a_conversion_stating_its_rounding_folds_by_it() {
        let narrow = |rounding| {
            let op = SsaOp::<MockTarget>::FloatConv {
                dest: var(1),
                operand: var(0),
                target: MockType::BF16,
                rounding: Some(rounding),
                source: None,
                environment: None,
            };
            MACHINE.evaluate(&op, None, |id| (id == var(0)).then(|| Cv::from(1.0 / 3.0)))
        };
        let bfloat = |bits| Cv::float(FloatFormat::BFloat16, bits);
        for (rounding, bits) in [
            (FloatRounding::TowardZero, 0x3eaa),
            (FloatRounding::Down, 0x3eaa),
            (FloatRounding::Up, 0x3eab),
            (FloatRounding::NearestEven, 0x3eab),
            (FloatRounding::ToOdd, 0x3eab),
        ] {
            assert_eq!(narrow(rounding), bfloat(bits), "{rounding}");
        }
        let from_int = |rounding| {
            let op = SsaOp::<MockTarget>::IntToFloat {
                dest: var(1),
                operand: var(0),
                target: MockType::BF16,
                unsigned: false,
                rounding: Some(rounding),
                environment: None,
            };
            MACHINE.evaluate(&op, None, |id| (id == var(0)).then_some(Cv::I32(257)))
        };
        for (rounding, bits) in [
            (FloatRounding::TowardZero, 0x4380),
            (FloatRounding::Down, 0x4380),
            (FloatRounding::Up, 0x4381),
            (FloatRounding::NearestEven, 0x4380),
        ] {
            assert_eq!(from_int(rounding), bfloat(bits), "{rounding}");
        }
        assert_eq!(from_int(FloatRounding::ToOdd), None);
    }

    /// **An integer converted by the program's mode folds only exactly.**
    /// `256` as a bfloat16 is `0x4380` in every mode; `257` is `256` or `258`
    /// as the mode says, so it does not fold.
    #[test]
    fn an_integer_converted_by_the_programs_mode_folds_only_exactly() {
        let from_int = |value| {
            let op = SsaOp::<MockTarget>::IntToFloat {
                dest: var(1),
                operand: var(0),
                target: MockType::BF16,
                unsigned: false,
                rounding: None,
                environment: None,
            };
            MACHINE.evaluate(&op, None, |id| (id == var(0)).then_some(Cv::I32(value)))
        };
        assert_eq!(from_int(256), Cv::float(FloatFormat::BFloat16, 0x4380));
        assert_eq!(from_int(257), None);
    }

    /// A float constant of any format is its bytes in the target's order and
    /// back: an x87 extended value is ten bytes, a bfloat16 two.
    #[test]
    fn a_float_constant_round_trips_through_its_bytes() {
        for endianness in [Endianness::Little, Endianness::Big] {
            for (value, ty) in [
                (
                    Cv::float(FloatFormat::X87Extended, 0x4000_c000_0000_0000_0000),
                    MockType::F80,
                ),
                (Cv::float(FloatFormat::BFloat16, 0x3fc0), MockType::BF16),
            ] {
                let value = value.expect("a float");
                let bytes = value
                    .to_bytes(endianness, PointerSize::Bit64)
                    .expect("its bytes");
                assert_eq!(bytes.len(), if ty == MockType::F80 { 10 } else { 2 });
                assert_eq!(Cv::from_typed_bytes(&bytes, &ty, endianness), Some(value));
            }
        }
    }

    /// Folds `op` over the constants `known` gives its operands, `dest` the
    /// destination it declares.
    fn fold_lanes(op: &SsaOp<MockTarget>, dest: Destination, known: &[Cv]) -> Option<Cv> {
        MACHINE.evaluate(op, dest, |id| known.get(id.index()).cloned())
    }

    fn words(values: [u32; 4]) -> Cv {
        Cv::Vector(VectorConst::from_lanes(32, values.map(u128::from)).expect("four words"))
    }

    /// **A classification of a constant folds in its encoding.** `fclass.s`
    /// of one is bit 6 at the destination's width, MIPS's `class.s` of it
    /// bit 7, and `fxam` of it `C2`, the parity flag of a flags word; of a
    /// value not known, nothing.
    #[test]
    fn a_classification_of_a_constant_folds_in_its_encoding() {
        let one = Cv::Float(FloatBits::new(FloatFormat::Single, 0x3f80_0000).expect("one"));
        let classify = |layout| SsaOp::FpClassify {
            dest: var(5),
            operand: var(0),
            layout,
        };
        assert_eq!(
            fold_lanes(
                &classify(FloatClassLayout::RiscV),
                Some(64).into(),
                std::slice::from_ref(&one)
            ),
            Some(Cv::U64(1 << 6))
        );
        assert_eq!(
            fold_lanes(
                &classify(FloatClassLayout::Mips),
                Some(32).into(),
                std::slice::from_ref(&one)
            ),
            Some(Cv::U32(1 << 7))
        );
        assert_eq!(
            fold_lanes(&classify(FloatClassLayout::X87), None.into(), &[one]),
            Some(Cv::U32(u32::from(FlagsMask::PARITY.bits())))
        );
        assert_eq!(
            fold_lanes(&classify(FloatClassLayout::RiscV), Some(64).into(), &[]),
            None
        );
    }

    /// **The length chosen for no application is zero.** `vsetivli zero, 0`
    /// chooses a length of zero whatever the maximum (here not known); a
    /// count of three, the maximum not known, chooses nothing known.
    #[test]
    fn the_length_chosen_for_no_application_is_zero() {
        let choice = SsaOp::compute(
            Computation::whole(
                Kernel::VectorLengthChoice,
                VectorElement::integer(64, None),
                VectorRounding::None,
            ),
            vec![var(5)],
            vec![var(0), var(1)],
        );
        assert_eq!(
            fold_lanes(&choice, Some(64).into(), &[Cv::U64(0)]),
            Some(Cv::U64(0))
        );
        assert_eq!(fold_lanes(&choice, Some(64).into(), &[Cv::U64(3)]), None);
    }

    /// **A mask summary of no element is a count of zero and no first.**
    /// `vcpop.m` and `vfirst.m` with an active length of zero write zero and
    /// all ones whatever the mask holds (here not known); of a length not zero
    /// they fold nothing, the mask's bits unknown.
    #[test]
    fn a_mask_summary_of_no_element_folds() {
        let summary = |first: bool| {
            SsaOp::compute(
                Computation {
                    kernel: Kernel::MaskSummary { first },
                    element: VectorElement::integer(8, None),
                    predicate: VectorPredicateLayout::Packed,
                    rounding: VectorRounding::None,
                    status: VectorStatus::None,
                    canonical_nan: false,
                    lanes: Some(LaneControl {
                        mask_tail: false,
                        tail: VectorTailPolicy::Agnostic,
                        inactive: None,
                        no_passthrough: false,
                    }),
                },
                vec![var(5)],
                vec![var(0), var(1), var(2), var(3)],
            )
        };
        // Restart, length and the scalar kept; the mask `var(3)` not known.
        let of_length = |length: u64| [Cv::U64(0), Cv::U64(length), Cv::U64(0)];
        assert_eq!(
            fold_lanes(&summary(false), Some(64).into(), &of_length(0)),
            Some(Cv::U64(0))
        );
        assert_eq!(
            fold_lanes(&summary(true), Some(64).into(), &of_length(0)),
            Some(Cv::U64(u64::MAX))
        );
        assert_eq!(
            fold_lanes(&summary(true), Some(64).into(), &of_length(4)),
            None
        );
    }

    /// **A computation over lanes of constants is the elements it computes.**
    /// `vadd.vv` of four words, unmasked at the full length, folds without
    /// its old value (`var(7)`, not known); masked by `0b0110` with an
    /// undisturbed old value of 9s, the inactive words keep the 9s. The
    /// vectors are read as the words their layouts declare.
    #[test]
    fn a_computation_over_lanes_folds_its_elements() {
        let add = |inactive: Option<VectorInactivePolicy>, inputs| {
            SsaOp::compute(
                Computation {
                    kernel: Kernel::Binary(VectorBinaryKind::Add),
                    element: VectorElement::integer(32, None),
                    predicate: VectorPredicateLayout::Packed,
                    rounding: VectorRounding::None,
                    status: VectorStatus::None,
                    canonical_nan: false,
                    lanes: Some(LaneControl {
                        mask_tail: false,
                        tail: VectorTailPolicy::Agnostic,
                        inactive,
                        no_passthrough: false,
                    }),
                },
                vec![var(9)],
                inputs,
            )
        };
        let known = [
            Cv::U64(0),
            Cv::U64(4),
            words([9, 9, 9, 9]),
            words([1, 2, 3, u32::MAX]),
            words([10, 20, 30, 2]),
            Cv::U8(0b0110),
        ];
        // The vectors are declared of four words; the controls are scalars.
        let words_layout = Destination {
            bits: Some(128),
            lanes: Some((32, 4)),
            lane_format: None,
            integer: true,
            integer_type: None,
        };
        let fold = |op: &SsaOp<MockTarget>, declared: bool| {
            MACHINE.evaluate_with_operand_layouts(
                op,
                words_layout,
                |id| known.get(id.index()).cloned(),
                |id| {
                    if declared && matches!(id.index(), 2..=4) {
                        OperandLayout::Register(words_layout)
                    } else {
                        OperandLayout::default()
                    }
                },
            )
        };
        let unmasked = add(None, vec![var(0), var(1), var(7), var(3), var(4)]);
        assert_eq!(fold(&unmasked, true), Some(words([11, 22, 33, 1])));
        let masked = add(
            Some(VectorInactivePolicy::Undisturbed),
            vec![var(0), var(1), var(5), var(2), var(3), var(4)],
        );
        assert_eq!(fold(&masked, true), Some(words([9, 22, 33, 9])));
        // Without the declarations the words are not known to be words, and
        // without layouts at all nothing is.
        assert_eq!(fold(&masked, false), None);
        assert_eq!(fold_lanes(&masked, Some(128).into(), &known), None);
    }

    /// **A vector's layout states its float lanes' format.** Two doubles
    /// declare `Double` lanes, which a 64-bit width alone would not say; four
    /// words declare none.
    #[test]
    fn a_vector_layout_states_its_float_format() {
        let (ssa, vars) = function_of(&[MockType::V2F64, MockType::V4I32], |_| Vec::new());
        assert_eq!(
            Destination::of(&ssa, vars[0]).lane_format,
            Some(FloatFormat::Double)
        );
        assert_eq!(Destination::of(&ssa, vars[1]).lane_format, None);
    }

    /// **A widening float computation reads its sources in the formats
    /// their layouts state.** `vfwadd.vv` of the singles `[1.5, -2]` and
    /// `[2.25, 0.5]` is the doubles `[3.75, -1.5]`; of the same bits declared
    /// of no format, nothing.
    #[test]
    fn a_widening_computation_reads_its_declared_formats() {
        let op = SsaOp::compute(
            Computation {
                kernel: Kernel::Binary(VectorBinaryKind::Add),
                element: VectorElement::float(64),
                predicate: VectorPredicateLayout::Packed,
                rounding: VectorRounding::None,
                status: VectorStatus::None,
                canonical_nan: true,
                lanes: Some(LaneControl {
                    mask_tail: false,
                    tail: VectorTailPolicy::Agnostic,
                    inactive: None,
                    no_passthrough: false,
                }),
            },
            vec![var(9)],
            vec![var(0), var(1), var(7), var(3), var(4)],
        );
        let singles = |values: [f32; 2]| {
            Cv::Vector(
                VectorConst::from_lanes(32, values.map(|value| u128::from(value.to_bits())))
                    .unwrap(),
            )
        };
        let known = [
            Cv::U64(0),
            Cv::U64(2),
            Cv::U64(0),
            singles([1.5, -2.0]),
            singles([2.25, 0.5]),
        ];
        let doubles_layout = Destination {
            bits: Some(128),
            lanes: Some((64, 2)),
            lane_format: Some(FloatFormat::Double),
            integer: false,
            integer_type: None,
        };
        let fold = |format: Option<FloatFormat>| {
            MACHINE.evaluate_with_operand_layouts(
                &op,
                doubles_layout,
                |id| known.get(id.index()).cloned(),
                |id| match id.index() {
                    3 | 4 => OperandLayout::Register(Destination {
                        bits: Some(64),
                        lanes: Some((32, 2)),
                        lane_format: format,
                        integer: false,
                        integer_type: None,
                    }),
                    _ => OperandLayout::default(),
                },
            )
        };
        assert_eq!(
            fold(Some(FloatFormat::Single)),
            Some(Cv::Vector(
                VectorConst::from_lanes(
                    64,
                    [3.75f64, -1.5].map(|value| u128::from(value.to_bits()))
                )
                .unwrap()
            ))
        );
        assert_eq!(fold(None), None);
    }

    /// **A computation over lanes folds its data and its status, each where
    /// it is known.** `vfadd.vv` of four singles at the full length is their
    /// sums and no exception; at a length of two its agnostic tail leaves the
    /// data unspecified while the two executed elements report no exception;
    /// `1 + 2^-30` rounds as the program's mode, not known, would round it, so
    /// no sum folds, yet it is inexact in every mode. Without the layouts,
    /// which state the elements single, nothing folds.
    #[test]
    fn a_computation_over_lanes_folds_its_data_and_status_each_where_known() {
        let op = SsaOp::compute(
            Computation {
                kernel: Kernel::Binary(VectorBinaryKind::Add),
                element: VectorElement::float(32),
                predicate: VectorPredicateLayout::Packed,
                rounding: VectorRounding::None,
                status: VectorStatus::FloatingExceptions,
                canonical_nan: true,
                lanes: Some(LaneControl {
                    mask_tail: false,
                    tail: VectorTailPolicy::Agnostic,
                    inactive: None,
                    no_passthrough: false,
                }),
            },
            vec![var(9), var(10)],
            vec![var(0), var(1), var(7), var(3), var(4)],
        );
        assert!(op.folds_results());
        let singles = |values: [f32; 4]| words(values.map(f32::to_bits));
        let singles_layout = Destination {
            bits: Some(128),
            lanes: Some((32, 4)),
            lane_format: Some(FloatFormat::Single),
            integer: false,
            integer_type: None,
        };
        let flags_layout = Destination {
            bits: Some(8),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(8, false),
        };
        let fold = |length: u64, right: [f32; 4]| {
            let known = [
                Cv::U64(0),
                Cv::U64(length),
                Cv::U64(0),
                singles([1.0, 2.0, 3.0, 4.0]),
                singles(right),
            ];
            MACHINE.evaluate_results_with_operand_layouts(
                &op,
                |id| known.get(id.index()).cloned(),
                |id| match id.index() {
                    3 | 4 | 9 => OperandLayout::Register(singles_layout),
                    10 => OperandLayout::Register(flags_layout),
                    _ => OperandLayout::default(),
                },
            )
        };
        let ones = [1.0; 4];
        assert_eq!(
            fold(4, ones).map(|results| results.into_vec()),
            Some(vec![
                (var(9), singles([2.0, 3.0, 4.0, 5.0])),
                (var(10), Cv::U8(0)),
            ])
        );
        assert_eq!(
            fold(2, ones).map(|results| results.into_vec()),
            Some(vec![(var(10), Cv::U8(0))])
        );
        let tiny = 2.0f32.powi(-30);
        assert_eq!(
            fold(4, [tiny, 0.0, 0.0, 0.0]).map(|results| results.into_vec()),
            Some(vec![(var(10), Cv::U8(1))])
        );
        let known = [
            Cv::U64(0),
            Cv::U64(4),
            Cv::U64(0),
            singles(ones),
            singles(ones),
        ];
        assert_eq!(
            MACHINE.evaluate_results(&op, |id| known.get(id.index()).cloned()),
            None
        );
    }

    /// **A predicated lane operation merges its lanes by its mask.** The words
    /// `[1, 2, 3, 4]` negated under the opmask `0b0101` (bit `i` lane `i`) are
    /// `[-1, 20, -3, 40]` with the passthrough `[10, 20, 30, 40]`, and `[-1, 0,
    /// -3, 0]` zeroing; under the lane mask `[!0, 0, 0, !0]` (an SVE
    /// predicate) `[-1, 20, 30, -4]`. A sum with `[100, 200, 300, 400]` and a
    /// select by `[0, !0, 0, !0]` merge the same way. A passthrough not known
    /// folds where every lane is active, and nowhere else; a mask not known
    /// folds nothing.
    #[test]
    fn a_predicated_lane_operation_merges_by_its_mask() {
        let negated = |value: u32| u32::wrapping_neg(value);
        let known = [
            words([1, 2, 3, 4]),
            words([10, 20, 30, 40]),
            Cv::U64(0b0101),
            words([u32::MAX, 0, 0, u32::MAX]),
            words([100, 200, 300, 400]),
            Cv::U64(0b1111),
            words([0, u32::MAX, 0, u32::MAX]),
        ];
        let element = VectorElement::integer(32, Some(false));
        let lanes = Destination {
            bits: Some(128),
            lanes: Some((32, 4)),
            lane_format: None,
            integer: true,
            integer_type: None,
        };
        let unary = |mask: usize, passthrough: Option<usize>, mode| {
            SsaOp::<MockTarget>::VectorPredicatedUnary {
                dest: var(9),
                value: var(0),
                mask: var(mask),
                passthrough: passthrough.map(var),
                kind: VectorUnaryKind::Neg,
                element,
                mode,
                environment: None,
            }
        };
        assert_eq!(
            fold_lanes(&unary(2, Some(1), VectorMaskMode::Merge), lanes, &known),
            Some(words([negated(1), 20, negated(3), 40]))
        );
        assert_eq!(
            fold_lanes(&unary(2, None, VectorMaskMode::Zero), lanes, &known),
            Some(words([negated(1), 0, negated(3), 0]))
        );
        assert_eq!(
            fold_lanes(&unary(3, Some(1), VectorMaskMode::Merge), lanes, &known),
            Some(words([negated(1), 20, 30, negated(4)]))
        );
        let sum = SsaOp::<MockTarget>::VectorPredicatedBinary {
            dest: var(9),
            left: var(0),
            right: var(4),
            mask: var(2),
            passthrough: Some(var(1)),
            kind: VectorBinaryKind::Add,
            element,
            mode: VectorMaskMode::Merge,
            environment: None,
        };
        assert_eq!(
            fold_lanes(&sum, lanes, &known),
            Some(words([101, 20, 303, 40]))
        );
        let select = SsaOp::<MockTarget>::VectorPredicatedTernary {
            dest: var(9),
            first: var(0),
            second: var(4),
            third: var(6),
            mask: var(2),
            passthrough: Some(var(1)),
            kind: VectorTernaryKind::Select,
            element,
            mode: VectorMaskMode::Merge,
            environment: None,
        };
        assert_eq!(
            fold_lanes(&select, lanes, &known),
            Some(words([1, 20, 3, 40]))
        );
        // The passthrough at 8 is not known.
        assert_eq!(
            fold_lanes(&unary(5, Some(8), VectorMaskMode::Merge), lanes, &known),
            Some(words([1, 2, 3, 4].map(negated)))
        );
        assert_eq!(
            fold_lanes(&unary(2, Some(8), VectorMaskMode::Merge), lanes, &known),
            None
        );
        assert_eq!(
            fold_lanes(&unary(8, Some(1), VectorMaskMode::Merge), lanes, &known),
            None,
            "a mask not known"
        );
    }

    /// **A float result in an integer destination is its bits.** The single
    /// pair `[1.5, 2.5]` summed in order is 4.0: into a destination of no
    /// stated kind the float, into an integer one `0x40800000`, the variable a
    /// target whose lanes carry no float type declares as the lane's integer.
    #[test]
    fn a_float_result_in_an_integer_destination_is_its_bits() {
        let lanes = Cv::Vector(
            VectorConst::from_lanes(32, [1.5f32, 2.5].map(|lane| u128::from(lane.to_bits())))
                .expect("two singles"),
        );
        let reduce = SsaOp::<MockTarget>::VectorReduce {
            dest: var(1),
            value: var(0),
            kind: VectorReduceKind::Add,
            element: VectorElement {
                kind: VectorElementKind::Float,
                bits: 32,
                scalar: false,
                signed: None,
                rounding: None,
            },
            initial: None,
            ordered: true,
            groups: None,
            environment: None,
        };
        let known = [lanes];
        let into = |integer: bool| Destination {
            bits: Some(32),
            lanes: None,
            lane_format: None,
            integer,
            integer_type: None,
        };
        assert_eq!(
            fold_lanes(&reduce, into(false), &known),
            Cv::float(FloatFormat::Single, u128::from(4.0f32.to_bits()))
        );
        assert_eq!(
            fold_lanes(&reduce, into(true), &known),
            Some(Cv::U32(4.0f32.to_bits()))
        );
    }

    /// **A reduction folds from its initial value, in the order it states.**
    /// `2^53` plus the lanes `[1, 1]` added in lane order and rounded to
    /// nearest is `2^53`, each one rounded away (SVE `fadda`); in an order the
    /// reduction does not state the ones may meet first and give `2^53 + 2`,
    /// so it folds nothing; without an initial value the lanes alone sum to 2.
    /// Rounded by the program's mode the ordered sum folds nothing either:
    /// rounding up gives `2^53 + 4`.
    #[test]
    fn a_reduction_folds_from_its_initial_value_in_the_order_it_states() {
        let double = |value: f64| {
            Cv::float(FloatFormat::Double, u128::from(value.to_bits())).expect("a double")
        };
        let lanes = Cv::Vector(
            VectorConst::from_lanes(64, [1.0f64, 1.0].map(|lane| u128::from(lane.to_bits())))
                .expect("two doubles"),
        );
        let scalar = Destination {
            bits: Some(64),
            lanes: None,
            lane_format: None,
            integer: false,
            integer_type: None,
        };
        let reduce = |initial: Option<SsaVarId>, ordered: bool| SsaOp::VectorReduce {
            dest: var(2),
            value: var(0),
            kind: VectorReduceKind::Add,
            element: VectorElement::float(64).rounded(FloatRounding::NearestEven),
            initial,
            ordered,
            groups: None,
            environment: None,
        };
        let big = 2.0f64.powi(53);
        let known = [lanes, double(big)];
        assert_eq!(
            fold_lanes(&reduce(Some(var(1)), true), scalar, &known),
            Some(double(big))
        );
        assert_eq!(
            fold_lanes(&reduce(Some(var(1)), false), scalar, &known),
            None
        );
        assert_eq!(
            fold_lanes(&reduce(None, false), scalar, &known),
            Some(double(2.0))
        );
        let by_mode = SsaOp::VectorReduce {
            dest: var(2),
            value: var(0),
            kind: VectorReduceKind::Add,
            element: VectorElement::float(64),
            initial: Some(var(1)),
            ordered: true,
            groups: None,
            environment: None,
        };
        assert_eq!(fold_lanes(&by_mode, scalar, &known), None);
    }

    /// **A bfloat16 reduction is a bfloat16.** The sum of the bfloat16 lanes
    /// `[1.5, 1.0]` is the bfloat16 `2.5` (`0x4020`), which as a binary16 would
    /// be `2.0625`: the format comes from the element, never the width alone.
    #[test]
    fn a_bfloat16_reduction_is_a_bfloat16() {
        let lanes = Cv::Vector(VectorConst::from_lanes(16, [0x3fc0, 0x3f80]).expect("two lanes"));
        let reduce = SsaOp::VectorReduce {
            dest: var(1),
            value: var(0),
            kind: VectorReduceKind::Add,
            element: VectorElement {
                kind: VectorElementKind::BFloat16,
                bits: 16,
                scalar: false,
                signed: None,
                rounding: None,
            },
            initial: None,
            ordered: false,
            groups: None,
            environment: None,
        };
        assert_eq!(
            fold_lanes(&reduce, Some(16).into(), &[lanes]),
            Cv::float(FloatFormat::BFloat16, 0x4020)
        );
    }

    /// **The lane movements fold**: a splat of a scalar, a shuffle of two
    /// vectors in the lanes its destination holds a selector each of, an
    /// insert at the inserted value's width and an extract at the
    /// destination's, and a reinterpret, which is the same bits. A result is
    /// laid out in its destination's lanes, whatever lanes it was computed in.
    #[test]
    fn the_lane_movements_fold() {
        let vector = Destination {
            bits: Some(128),
            lanes: Some((32, 4)),
            lane_format: None,
            integer: false,
            integer_type: None,
        };
        let splat = SsaOp::VectorSplat {
            dest: var(1),
            value: var(0),
            vector_type: MockType::V4I32,
        };
        assert_eq!(
            fold_lanes(&splat, vector, &[Cv::U64(0x1_0000_0007)]),
            Some(words([7, 7, 7, 7]))
        );
        let shuffle = SsaOp::VectorShuffle {
            dest: var(2),
            left: var(0),
            right: Some(var(1)),
            selector: VectorShuffleSelector::Mask(VectorShuffleMask::new(vec![
                Left(3),
                Right(1),
                Zero,
                Left(0),
            ])),
        };
        assert_eq!(
            fold_lanes(
                &shuffle,
                vector,
                &[words([1, 2, 3, 4]), words([5, 6, 7, 8])]
            ),
            Some(words([4, 6, 0, 1]))
        );
        let insert = SsaOp::VectorInsert {
            dest: var(2),
            vector: var(0),
            lane: 5,
            value: var(1),
        };
        // Lane 5 of the halves is the high half of word 2.
        assert_eq!(
            fold_lanes(&insert, vector, &[words([1, 2, 3, 4]), Cv::U16(0xabcd)]),
            Some(words([1, 2, 0xabcd_0003, 4]))
        );
        let extract = SsaOp::VectorExtract {
            dest: var(1),
            vector: var(0),
            lane: 1,
        };
        assert_eq!(
            fold_lanes(&extract, Some(64).into(), &[words([1, 2, 3, 4])]),
            Some(Cv::U64(0x4_0000_0003))
        );
        assert_eq!(
            fold_lanes(&extract, Some(64).into(), &[Cv::U64(9)]),
            None,
            "past the lanes"
        );
        let reinterpret = SsaOp::VectorReinterpret {
            dest: var(1),
            value: var(0),
            target_type: MockType::V2F64,
        };
        let halves = Destination {
            bits: Some(128),
            lanes: Some((64, 2)),
            lane_format: None,
            integer: false,
            integer_type: None,
        };
        assert_eq!(
            fold_lanes(&reinterpret, halves, &[words([1, 2, 3, 4])]),
            VectorConst::from_lanes(64, [0x2_0000_0001, 0x4_0000_0003]).map(Cv::Vector)
        );
        // Laid out in the destination's lanes.
        assert_eq!(
            fold_lanes(
                &shuffle,
                halves,
                &[words([1, 2, 3, 4]), words([5, 6, 7, 8])]
            ),
            VectorConst::from_lanes(64, [0x6_0000_0004, 0x1_0000_0000]).map(Cv::Vector)
        );
        // A lane per selector across the destination: four of 64 bits are
        // halfwords, read from the inputs in halfwords.
        assert_eq!(
            fold_lanes(
                &shuffle,
                Some(64).into(),
                &[words([1, 2, 3, 4]), words([5, 6, 7, 8])]
            ),
            VectorConst::from_lanes(16, [0, 0, 0, 1]).map(Cv::Vector)
        );
        // An input typed as one register-wide lane is read in them too.
        let register = |value: Cv| match value {
            Cv::Vector(vector) => vector.resized(128, 1).map(Cv::Vector),
            other => Some(other),
        };
        assert_eq!(
            fold_lanes(
                &shuffle,
                vector,
                &[
                    register(words([1, 2, 3, 4])).expect("one lane"),
                    register(words([5, 6, 7, 8])).expect("one lane"),
                ]
            ),
            Some(words([4, 6, 0, 1]))
        );
    }

    /// **A float operation reads its operand's bits as the float**, however
    /// the constant was typed: the integer `0x3fc0_0000` converted to a double
    /// is 1.5, not 1069547520.0. A conversion to an integer rounds toward
    /// zero, and one the integer does not hold folds nothing.
    #[test]
    fn a_float_operation_reads_an_integer_as_its_bits() {
        let widen = SsaOp::FloatConv {
            dest: var(1),
            operand: var(0),
            target: MockType::F64,
            rounding: None,
            source: None,
            environment: None,
        };
        assert_eq!(
            fold_lanes(&widen, Some(64).into(), &[Cv::U32(1.5f32.to_bits())]),
            Some(Cv::from(1.5f64))
        );
        let truncate = SsaOp::FloatToInt {
            dest: var(1),
            operand: var(0),
            target: MockType::I32,
            overflow_check: false,
            unsigned: false,
            source: None,
        };
        assert_eq!(
            fold_lanes(&truncate, Some(32).into(), &[Cv::U64((-2.75f64).to_bits())]),
            Some(Cv::I32(-2))
        );
        assert_eq!(
            fold_lanes(&truncate, Some(32).into(), &[Cv::U16(0x3c00)]),
            None,
            "16 bits name no one format"
        );
        for value in [3e9f64, f64::NAN, f64::INFINITY] {
            assert_eq!(
                fold_lanes(&truncate, Some(32).into(), &[Cv::from(value)]),
                None,
                "{value} is each architecture's own"
            );
        }
    }

    /// One float lane's arithmetic folds as the scalar operations do: a
    /// binary128 `1/3` rounded as the element states (`0x...5555` to nearest,
    /// `0x...5556` up) and not at all where it states nothing, the program's
    /// mode deciding it, an exact binary64 product, and the minimum of two
    /// ordered values. A NaN result does not fold, and neither does a `min`
    /// of a NaN, whose rule is the architecture's.
    #[test]
    fn a_scalar_float_lane_folds_its_arithmetic() {
        let rounded = |kind: VectorBinaryKind,
                       left: Cv,
                       right: Cv,
                       bits: u32,
                       rounding: Option<FloatRounding>| {
            let op = SsaOp::<MockTarget>::VectorBinary {
                dest: var(2),
                left: var(0),
                right: var(1),
                kind,
                element: VectorElement {
                    kind: VectorElementKind::Float,
                    bits,
                    scalar: true,
                    signed: None,
                    rounding,
                },
                environment: None,
            };
            MACHINE.evaluate(&op, None, |id| match id.index() {
                0 => Some(left.clone()),
                1 => Some(right.clone()),
                _ => None,
            })
        };
        let fold = |kind, left, right, bits| rounded(kind, left, right, bits, None);
        let quad = |bits: u128| Cv::float(FloatFormat::Quad, bits).expect("a binary128");
        let one = quad(0x3fff_0000_0000_0000_0000_0000_0000_0000);
        let three = quad(0x4000_8000_0000_0000_0000_0000_0000_0000);
        let third = |rounding| {
            rounded(
                VectorBinaryKind::Div,
                one.clone(),
                three.clone(),
                128,
                rounding,
            )
        };
        assert_eq!(
            third(Some(FloatRounding::NearestEven)),
            Some(quad(0x3ffd_5555_5555_5555_5555_5555_5555_5555))
        );
        assert_eq!(
            third(Some(FloatRounding::Up)),
            Some(quad(0x3ffd_5555_5555_5555_5555_5555_5555_5556))
        );
        assert_eq!(third(None), None, "the program's mode decides a third");
        assert_eq!(
            fold(
                VectorBinaryKind::Mul,
                Cv::from(1.5f64),
                Cv::from(4.0f64),
                64
            ),
            Some(Cv::from(6.0f64))
        );
        let infinity = quad(0x7fff_0000_0000_0000_0000_0000_0000_0000);
        assert_eq!(
            fold(VectorBinaryKind::Sub, infinity.clone(), infinity, 128),
            None,
            "inf - inf"
        );
        assert_eq!(
            fold(
                VectorBinaryKind::Min,
                Cv::from(1.0f64),
                Cv::from(2.0f64),
                64
            ),
            Some(Cv::from(1.0f64)),
            "min"
        );
        assert_eq!(
            fold(
                VectorBinaryKind::Min,
                Cv::from(f64::NAN),
                Cv::from(2.0f64),
                64
            ),
            None,
            "min of a NaN"
        );
    }

    /// A comparison with a NaN constant is decided whatever the other operand
    /// holds: equality and the ordered relations are false, the unordered ones
    /// true.
    #[test]
    fn a_comparison_with_a_nan_is_decided() {
        let unknown_and_nan = |id: SsaVarId| (id == var(1)).then_some(Cv::from(f64::NAN));
        let ceq = SsaOp::<MockTarget>::Ceq {
            dest: var(2),
            left: var(0),
            right: var(1),
            signaling: false,
        };
        assert_eq!(
            MACHINE.evaluate(&ceq, None, unknown_and_nan),
            Some(Cv::False)
        );
        for unsigned in [false, true] {
            let clt = SsaOp::<MockTarget>::Clt {
                dest: var(2),
                left: var(1),
                right: var(0),
                unsigned,
                signaling: false,
            };
            let cgt = SsaOp::<MockTarget>::Cgt {
                dest: var(2),
                left: var(0),
                right: var(1),
                unsigned,
                signaling: false,
            };
            for op in [clt, cgt] {
                assert_eq!(
                    MACHINE.evaluate(&op, None, unknown_and_nan),
                    Some(Cv::from_bool(unsigned)),
                    "{op:?}"
                );
            }
        }
        // Without a NaN an unknown operand still decides nothing.
        let unknown_and_one = |id: SsaVarId| (id == var(1)).then_some(Cv::from(1.0f64));
        assert_eq!(MACHINE.evaluate(&ceq, None, unknown_and_one), None);
    }

    /// `div.un` and `rem.un` fold as unsigned.
    #[test]
    fn unsigned_division_folds_as_unsigned() {
        let operands = |var: SsaVarId| match var.index() {
            0 => Some(ConstValue::I32(-8)),
            1 => Some(ConstValue::I32(3)),
            _ => None,
        };
        let div = |unsigned| SsaOp::<MockTarget>::Div {
            dest: SsaVarId::from_index(2),
            left: SsaVarId::from_index(0),
            right: SsaVarId::from_index(1),
            unsigned,
            flags: None,
        };
        let rem = |unsigned| SsaOp::<MockTarget>::Rem {
            dest: SsaVarId::from_index(2),
            left: SsaVarId::from_index(0),
            right: SsaVarId::from_index(1),
            unsigned,
            flags: None,
        };
        let ptr = PointerSize::Bit64;
        assert_eq!(
            Machine::new(ptr, None, FloatRules::default()).evaluate(&div(true), None, operands),
            Some(ConstValue::I32(0x5555_5552))
        );
        assert_eq!(
            Machine::new(ptr, None, FloatRules::default()).evaluate(&div(false), None, operands),
            Some(ConstValue::I32(-2))
        );
        assert_eq!(
            Machine::new(ptr, None, FloatRules::default()).evaluate(&rem(true), None, operands),
            Some(ConstValue::I32(2))
        );
        assert_eq!(
            Machine::new(ptr, None, FloatRules::default()).evaluate(&rem(false), None, operands),
            Some(ConstValue::I32(-2))
        );
    }

    /// A rotate is a pure function of its operands and folds at the operand's
    /// own width.
    #[test]
    fn a_rotate_folds_at_its_operands_width() {
        let value = SsaVarId::from_index(0);
        let amount = SsaVarId::from_index(1);
        let dest = SsaVarId::from_index(2);
        let constants = |var: SsaVarId| -> Option<ConstValue<MockTarget>> {
            match var.index() {
                0 => Some(ConstValue::U32(0x8000_0001)),
                1 => Some(ConstValue::I32(1)),
                _ => None,
            }
        };

        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::<MockTarget>::Rol {
                    dest,
                    value,
                    amount
                },
                None,
                constants
            ),
            Some(ConstValue::U32(0x0000_0003)),
        );
    }

    /// A `Const`-defined var resolves to its literal, `set_known` overrides it,
    /// and `clear_cache` drops the override so the IR value returns.
    #[test]
    fn const_evaluator_resolves_consts_and_injected_values() {
        let ssa = testing::const_i32_return(42);
        let mut evaluator = ConstEvaluator::new(&ssa, MACHINE);

        assert_eq!(evaluator.ssa().block_count(), 1);
        assert_eq!(evaluator.evaluate_var(var(0)), Some(Cv::I32(42)));

        evaluator.set_known(var(0), Cv::I32(7));
        assert_eq!(evaluator.evaluate_var(var(0)), Some(Cv::I32(7)));

        evaluator.clear_cache();
        assert_eq!(evaluator.evaluate_var(var(0)), Some(Cv::I32(42)));
        assert_eq!(evaluator.into_results().get(&var(0)), Some(&Cv::I32(42)));
    }

    /// A copy chain folds to its root constant, and a zero depth budget refuses
    /// to trace it (returns `None`).
    #[test]
    fn const_evaluator_traces_copy_chains_and_honors_depth_limit() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        for i in 0..3 {
            ssa.create_variable(
                VariableOrigin::Local(i),
                0,
                DefSite::instruction(0, usize::from(i)),
                MockType::I32,
            );
        }
        let mut block = SsaBlock::new(0);
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Const {
            dest: var(0),
            value: Cv::I32(5),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
            dest: var(1),
            src: var(0),
        }));
        block.add_instruction(SsaInstruction::synthetic(SsaOp::Copy {
            dest: var(2),
            src: var(1),
        }));
        ssa.add_block(block);

        let mut evaluator = ConstEvaluator::new(&ssa, MACHINE);
        assert_eq!(evaluator.evaluate_var(var(2)), Some(Cv::I32(5)));

        let mut shallow = ConstEvaluator::new(&ssa, MACHINE).with_max_depth(0);
        assert_eq!(shallow.evaluate_var(var(2)), None);
    }

    /// [`Machine::evaluate`] folds arithmetic, bitwise, and shift ops over
    /// constant operands.
    #[test]
    fn the_machine_folds_arithmetic_bitwise_and_shifts() {
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Add {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(13))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Sub {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(7))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Mul {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(30))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Div {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(3))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Rem {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(1))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::And {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(2))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Or {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(11))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Xor {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(9))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Shl {
                    dest: var(9),
                    value: var(1),
                    amount: var(1),
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(24))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Shr {
                    dest: var(9),
                    value: var(2),
                    amount: var(1),
                    unsigned: true,
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(536_870_911))
        );
    }

    /// [`Machine::evaluate`] folds unary, comparison, and checked-arithmetic ops
    /// over constant operands.
    #[test]
    fn the_machine_folds_unary_comparison_and_checked_arithmetic() {
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Neg {
                    dest: var(9),
                    operand: var(1),
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(-3))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Not {
                    dest: var(9),
                    operand: var(1),
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(!3))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Ceq {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    signaling: false,
                },
                None,
                resolve
            ),
            Some(Cv::False)
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Clt {
                    dest: var(9),
                    left: var(1),
                    right: var(0),
                    unsigned: false,
                    signaling: false,
                },
                None,
                resolve
            ),
            Some(Cv::True)
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Cgt {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: true,
                    signaling: false,
                },
                None,
                resolve
            ),
            Some(Cv::True)
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::AddOvf {
                    dest: var(9),
                    left: var(3),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                None,
                resolve
            ),
            None
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::SubOvf {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(7))
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::MulOvf {
                    dest: var(9),
                    left: var(0),
                    right: var(1),
                    unsigned: false,
                    flags: None,
                },
                None,
                resolve
            ),
            Some(Cv::I32(30))
        );
    }

    /// [`Machine::evaluate`] yields `None` when an operand is unknown or the
    /// operation is not foldable.
    /// **A symbol reinterpreted at the pointer's width is the symbol.** An
    /// import's address held in a 32-bit register on a 32-bit target is
    /// `ptrtoint(&stdout)`; kept a conversion, a loop hoists it into a variable
    /// and the access through it no longer reads as `stdout`. The controls: a
    /// narrower integer is a slice of the address and does not fold, and the
    /// way back to a pointer folds as well.
    #[test]
    fn a_symbol_reinterpreted_at_the_pointer_width_is_the_symbol() {
        let symbol = |_: SsaVarId| Some(Cv::Symbol(7));
        let to_int = |target| SsaOp::PtrToInt {
            dest: var(9),
            operand: var(0),
            target,
        };
        assert_eq!(
            MACHINE.evaluate(&to_int(MockType::U64), None, symbol),
            Some(Cv::Symbol(7))
        );
        assert_eq!(
            MACHINE.evaluate(&to_int(MockType::U32), None, symbol),
            None,
            "a slice of the address"
        );
        assert_eq!(
            Machine::new(PointerSize::Bit32, None, FloatRules::default()).evaluate(
                &to_int(MockType::U32),
                None,
                symbol
            ),
            Some(Cv::Symbol(7))
        );
        let to_pointer = SsaOp::IntToPtr {
            dest: var(9),
            operand: var(0),
            target: MockType::Ptr,
        };
        assert_eq!(
            MACHINE.evaluate(&to_pointer, None, symbol),
            Some(Cv::Symbol(7))
        );
    }

    /// **An address adding nothing to a symbol is the symbol.** dash keeps
    /// `&__stack_chk_guard` in a saved register and reads the guard through it
    /// twice; the address of each read adds nothing to it, and kept an
    /// address computation, the guard printed as `*var_1`. The control: a
    /// displacement is arithmetic on an address relocation moves and does
    /// not fold.
    #[test]
    fn an_address_adding_nothing_to_a_symbol_is_the_symbol() {
        let address = |offset| SsaOp::PtrAdd {
            dest: var(9),
            base: var(0),
            index: None,
            stride: 0,
            offset,
            result_type: MockType::Ptr,
        };
        let symbol = |_: SsaVarId| Some(Cv::Symbol(7));
        assert_eq!(
            MACHINE.evaluate(&address(0), None, symbol),
            Some(Cv::Symbol(7))
        );
        assert_eq!(
            MACHINE.evaluate(&address(8), None, symbol),
            None,
            "a displacement is the address model's"
        );
    }

    /// **An address from a plain number is the number it sums to.** A number
    /// is an absolute address nothing relocates: `0x1000 + 3 * 8 + 0x10` is
    /// `0x1028`, and a negative index counts down. A 32-bit machine wraps
    /// the sum at 32 bits (`0xffff_fff8 + 0x10` is `8`). The controls: a
    /// pointer formed from null, and a symbol displaced, fold nothing.
    #[test]
    fn an_address_from_a_number_is_the_number_it_sums_to() {
        let address = |index: Option<SsaVarId>, offset| SsaOp::PtrAdd {
            dest: var(9),
            base: var(0),
            index,
            stride: 8,
            offset,
            result_type: MockType::Ptr,
        };
        let values = |base: Cv, index: Cv| {
            move |var: SsaVarId| match var.index() {
                0 => Some(base.clone()),
                1 => Some(index.clone()),
                _ => None,
            }
        };
        assert_eq!(
            MACHINE.evaluate(
                &address(Some(var(1)), 0x10),
                None,
                values(Cv::U64(0x1000), Cv::U64(3))
            ),
            Some(Cv::NativeUInt(0x1028))
        );
        assert_eq!(
            MACHINE.evaluate(
                &address(Some(var(1)), 0),
                None,
                values(Cv::U64(0x1000), Cv::I64(-2))
            ),
            Some(Cv::NativeUInt(0xff0))
        );
        let narrow = Machine::new(PointerSize::Bit32, None, FloatRules::default());
        assert_eq!(
            narrow.evaluate(
                &address(None, 0x10),
                None,
                values(Cv::U32(0xffff_fff8), Cv::U32(0))
            ),
            Some(Cv::NativeUInt(8))
        );
        assert_eq!(
            MACHINE.evaluate(&address(None, 0x10), None, values(Cv::Null, Cv::U64(0))),
            None,
            "a pointer formed from null"
        );
        assert_eq!(
            MACHINE.evaluate(
                &address(None, 0x10),
                None,
                values(Cv::Symbol(7), Cv::U64(0))
            ),
            None,
            "a symbol displaced"
        );
    }

    #[test]
    fn the_machine_folds_nothing_of_unknown_inputs_and_unsupported_ops() {
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::Add {
                    dest: var(9),
                    left: var(0),
                    right: var(99),
                    flags: None,
                },
                None,
                resolve
            ),
            None
        );
        assert_eq!(
            MACHINE.evaluate(
                &SsaOp::IntConv {
                    dest: var(9),
                    operand: var(0),
                    target: MockType::Unknown,
                    overflow_check: false,
                    unsigned: false
                },
                None,
                resolve
            ),
            None
        );
        assert_eq!(
            MACHINE.evaluate::<MockTarget>(&SsaOp::returns(Some(var(0))), None, resolve,),
            None
        );
    }

    /// **An integer dot product of constants folds.** `vpdpwssd` of `[2, 3]`
    /// by `[5, 7]` into 1 is 32, `pmaddwd` of the same halfwords without the
    /// accumulator 31, and `sdot` of the bytes `[1, 2, 3, 4]` by
    /// `[-1, -1, -1, -1]` into 100 is 90.
    #[test]
    fn an_integer_dot_product_of_constants_folds() {
        let one_word = |value: u32| {
            Cv::Vector(VectorConst::from_lanes(32, [u128::from(value)]).expect("a word"))
        };
        let halves = Cv::Vector(VectorConst::from_lanes(16, [2, 3]).expect("halves"));
        let factors = Cv::Vector(VectorConst::from_lanes(16, [5, 7]).expect("halves"));
        // Signed halfword pairs summed into words, with an accumulator or not.
        let pairs = |accumulates: bool, inputs: Vec<SsaVarId>| {
            SsaOp::<MockTarget>::contract(
                Contraction::Dot(DotShape {
                    source_bits: 16,
                    dest_bits: 32,
                    signed_a: true,
                    signed_b: true,
                    accumulates,
                    saturates: None,
                }),
                vec![var(9)],
                inputs,
            )
        };
        let known = [one_word(1), halves.clone(), factors.clone()];
        assert_eq!(
            fold_lanes(
                &pairs(true, vec![var(1), var(2), var(0)]),
                Some(32).into(),
                &known
            ),
            Some(one_word(32))
        );
        assert_eq!(
            fold_lanes(&pairs(false, vec![var(1), var(2)]), Some(32).into(), &known),
            Some(one_word(31))
        );
        let sdot = SsaOp::<MockTarget>::contract(
            Contraction::Dot(DotShape {
                source_bits: 8,
                dest_bits: 32,
                signed_a: true,
                signed_b: true,
                accumulates: true,
                saturates: None,
            }),
            vec![var(9)],
            vec![var(1), var(2), var(0)],
        );
        let bytes = Cv::Vector(VectorConst::from_lanes(8, [1, 2, 3, 4]).expect("bytes"));
        let minus_ones = Cv::Vector(VectorConst::from_lanes(8, [0xff; 4]).expect("bytes"));
        assert_eq!(
            fold_lanes(&sdot, Some(32).into(), &[one_word(100), bytes, minus_ones]),
            Some(one_word(90))
        );
    }

    /// Region operations fold their explicit bit positions and layout, but a
    /// runtime-scalable result never folds to a minimum-width placeholder.
    #[test]
    fn vector_regions_fold_fixed_bits_and_keep_runtime_widths() {
        let known = [
            words([1, 2, 3, 4]),
            Cv::U64(32),
            Cv::U64(0xaaaa_aaaa_bbbb_bbbb),
            Cv::U64(0x2_0000_0001),
            Cv::U64(0x4_0000_0003),
            Cv::U64(128),
        ];
        let mut ops = [
            SsaOp::VectorSlice {
                dest: var(6),
                vector: var(0),
                offset: var(1),
                length: var(5),
                target_type: MockType::V4I32,
            },
            SsaOp::VectorInsertSlice {
                dest: var(6),
                vector: var(0),
                offset: var(1),
                value: var(2),
                target_type: MockType::V4I32,
            },
            SsaOp::VectorConcat {
                dest: var(6),
                left: var(3),
                right: var(4),
                target_type: MockType::V4I32,
            },
        ];
        let dest = Destination {
            bits: Some(128),
            lanes: Some((32, 4)),
            lane_format: None,
            integer: false,
            integer_type: None,
        };
        for (op, expected) in ops.iter().zip([
            words([2, 3, 4, 0]),
            words([1, 0xbbbb_bbbb, 0xaaaa_aaaa, 4]),
            words([1, 2, 3, 4]),
        ]) {
            assert_eq!(fold_lanes(op, dest, &known), Some(expected));
        }
        for op in &mut ops {
            match op {
                SsaOp::VectorSlice { target_type, .. }
                | SsaOp::VectorInsertSlice { target_type, .. } => *target_type = MockType::NxV4I32,
                SsaOp::VectorConcat { target_type, .. } => *target_type = MockType::NxV4I32,
                _ => unreachable!(),
            }
            assert_eq!(fold_lanes(op, Destination::default(), &known), None, "{op}");
        }
    }

    /// A computation of `kernel` over a whole scalar, defining `outputs` from
    /// `inputs`.
    fn whole(kernel: Kernel, outputs: Vec<SsaVarId>, inputs: Vec<SsaVarId>) -> SsaOp<MockTarget> {
        SsaOp::compute(
            Computation::whole(
                kernel,
                VectorElement::integer(64, Some(false)).as_scalar(),
                VectorRounding::None,
            ),
            outputs,
            inputs,
        )
    }

    /// The constant `kernel` computes of `inputs`, into a destination `bits`
    /// wide.
    fn compute(kernel: Kernel, inputs: &[Cv], bits: u32) -> Option<Cv> {
        let op = whole(
            kernel,
            vec![var(inputs.len())],
            (0..inputs.len()).map(var).collect(),
        );
        MACHINE.evaluate(&op, Some(bits), |id| inputs.get(id.index()).cloned())
    }

    /// The doubleword a cipher step writes of 64-bit sources.
    fn step(kind: ScalarCipherKind, inputs: &[u64]) -> u64 {
        let inputs: Vec<Cv> = inputs.iter().map(|value| Cv::U64(*value)).collect();
        match compute(Kernel::ScalarCipher(kind), &inputs, 64) {
            Some(Cv::U64(value)) => value,
            other => panic!("{kind:?} of {inputs:?} gave {other:?}"),
        }
    }

    /// The word a column step writes of 32-bit sources, at RV32's width.
    fn column(kind: ScalarCipherKind, first: u32, second: u32, select: u8) -> u32 {
        let inputs = [Cv::U32(first), Cv::U32(second), Cv::U8(select)];
        match compute(Kernel::ScalarCipher(kind), &inputs, 32) {
            Some(Cv::U32(value)) => value,
            other => panic!("{kind:?} of {inputs:?} gave {other:?}"),
        }
    }

    /// The bytes a hex string spells.
    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
            .collect()
    }

    /// The two little-endian doublewords of a 16-byte block, low first.
    fn halves(bytes: &[u8]) -> [u64; 2] {
        [
            u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        ]
    }

    /// **A bit deposit and a bit extract of constants fold.** Each pair of
    /// source and mask gives what BMI2's `pdep` and `pext` gave on the host
    /// that wrote this test, at 64 and at 32 bits.
    #[test]
    fn a_bit_permutation_of_constants_folds() {
        for (source, mask, deposit, extract) in [
            (
                0x0123_4567_89ab_cdefu64,
                0xff00_ff00_f0f0_f0f0u64,
                0x8900_ab00_c0d0_e0f0u64,
                0x0145_8aceu64,
            ),
            (
                0xfedc_ba98_7654_3210,
                0x0000_0000_ffff_ffff,
                0x7654_3210,
                0x7654_3210,
            ),
            (
                u64::MAX,
                0xaaaa_aaaa_aaaa_aaaa,
                0xaaaa_aaaa_aaaa_aaaa,
                0xffff_ffff,
            ),
            (0x8000_0000_0000_0001, 0x8000_0000_0000_0001, 1, 3),
        ] {
            let inputs = [Cv::U64(source), Cv::U64(mask)];
            assert_eq!(
                compute(Kernel::BitDeposit, &inputs, 64),
                Some(Cv::U64(deposit))
            );
            assert_eq!(
                compute(Kernel::BitExtract, &inputs, 64),
                Some(Cv::U64(extract))
            );
        }
        for (source, mask, deposit, extract) in [
            (0x89ab_cdefu32, 0x01e1_e1e1u32, 0x00c1_e0e1u32, 0x1bdfu32),
            (0x7654_3210, 0x01ff_ffff, 0x0054_3210, 0x0054_3210),
            (u32::MAX, 0x5555_5555, 0x5555_5555, 0xffff),
            (1, 0, 0, 0),
        ] {
            let inputs = [Cv::U32(source), Cv::U32(mask)];
            assert_eq!(
                compute(Kernel::BitDeposit, &inputs, 32),
                Some(Cv::U32(deposit))
            );
            assert_eq!(
                compute(Kernel::BitExtract, &inputs, 32),
                Some(Cv::U32(extract))
            );
        }
        // The source and the mask are as wide as the result.
        let mixed = [Cv::U32(1), Cv::U64(1)];
        assert_eq!(compute(Kernel::BitDeposit, &mixed, 64), None);
    }

    /// **A checksum step of constants folds, its data as wide as its
    /// type.** Castagnoli's steps of a byte, a halfword, a word and a
    /// doubleword give what SSE4.2's `crc32` gave on the host that wrote this
    /// test; the IEEE steps over the bytes of `123456789`, inverted before and
    /// after, give zlib's check value `0xcbf43926` (Castagnoli's
    /// `0xe3069283`), and a doubleword step gives zlib's checksum of its eight
    /// bytes. A 64-bit accumulator is read as its low word.
    #[test]
    fn a_checksum_step_of_constants_folds() {
        let castagnoli = Kernel::Checksum(CrcPolynomial::Castagnoli);
        let ieee = Kernel::Checksum(CrcPolynomial::Ieee);
        for (crc, data, expected) in [
            (0x1234_5678u32, Cv::U8(0xef), 0x4670_acaau32),
            (0x1234_5678, Cv::U16(0xcdef), 0xb54a_8725),
            (0x1234_5678, Cv::U32(0x89ab_cdef), 0xa360_621e),
            (0x1234_5678, Cv::U64(0x0123_4567_89ab_cdef), 0xa3d2_07be),
            (0x1234_5679, Cv::U8(0x10), 0x1966_7cf8),
            (0x1234_5679, Cv::U16(0x3210), 0xa876_6880),
            (0x1234_5679, Cv::U32(0x7654_3210), 0xc9bd_7c9e),
            (0x1234_5679, Cv::U64(0xfedc_ba98_7654_3210), 0x2ea1_83d4),
            (0x1234_567a, Cv::U8(0xff), 0xb715_1b32),
            (0x1234_567a, Cv::U64(u64::MAX), 0x1c7d_6e14),
            (0x1234_567b, Cv::U64(0x8000_0000_0000_0001), 0x5ac4_ac21),
        ] {
            assert_eq!(
                compute(castagnoli, &[Cv::U32(crc), data.clone()], 32),
                Some(Cv::U32(expected)),
                "{data:?}"
            );
            // x86's `crc32 rax, rcx` reads the low word of the accumulator
            // and writes the checksum zero-extended.
            assert_eq!(
                compute(
                    castagnoli,
                    &[Cv::U64(0xffff_0000_0000_0000 | u64::from(crc)), data],
                    64
                ),
                Some(Cv::U64(u64::from(expected)))
            );
        }
        for (kind, check) in [(ieee, 0xcbf4_3926u32), (castagnoli, 0xe306_9283)] {
            let mut crc = u32::MAX;
            for byte in *b"123456789" {
                match compute(kind, &[Cv::U32(crc), Cv::U8(byte)], 32) {
                    Some(Cv::U32(next)) => crc = next,
                    other => panic!("{other:?}"),
                }
            }
            assert_eq!(!crc, check, "{kind:?}");
        }
        for (data, zlib) in [
            (0x0123_4567_89ab_cdefu64, 0x443b_e247u32),
            (0xfedc_ba98_7654_3210, 0x005d_e232),
        ] {
            assert_eq!(
                compute(ieee, &[Cv::U32(u32::MAX), Cv::U64(data)], 32),
                Some(Cv::U32(!zlib))
            );
        }
    }

    /// **A carry-less product of constants folds.** The low, high and
    /// reversed parts of 64-bit products are the halves of what
    /// `pclmulqdq` gave on the host that wrote this test, the reversed part
    /// its bits 126 to 63; the 32-bit parts are the halves of its 63-bit
    /// product of words.
    #[test]
    fn a_carryless_product_of_constants_folds() {
        let part = |part: CarrylessPart, left: Cv, right: Cv, bits: u32| {
            compute(Kernel::CarrylessMultiply { part }, &[left, right], bits)
        };
        for (left, right, low, high) in [
            (
                0x0123_4567_89ab_cdefu64,
                0xff00_ff00_f0f0_f0f0u64,
                0x0761_da80_7f6f_bf50u64,
                0x00e1_dd00_78ef_b8d0u64,
            ),
            (
                0xfedc_ba98_7654_3210,
                0x0000_0000_ffff_ffff,
                0x7878_7878_2dcc_11f0,
                0x0000_0000_55b4_6988,
            ),
            (
                u64::MAX,
                0xaaaa_aaaa_aaaa_aaaa,
                0x6666_6666_6666_6666,
                0x6666_6666_6666_6666,
            ),
            (
                0x8000_0000_0000_0001,
                0x8000_0000_0000_0001,
                1,
                0x4000_0000_0000_0000,
            ),
        ] {
            let (l, r) = (Cv::U64(left), Cv::U64(right));
            assert_eq!(
                part(CarrylessPart::Low, l.clone(), r.clone(), 64),
                Some(Cv::U64(low))
            );
            assert_eq!(
                part(CarrylessPart::High, l.clone(), r.clone(), 64),
                Some(Cv::U64(high))
            );
            assert_eq!(
                part(CarrylessPart::Reversed, l, r, 64),
                Some(Cv::U64((high << 1) | (low >> 63)))
            );
        }
        for (left, right, product) in [
            (0x89ab_cdefu32, 0xf0f0_f0f0u32, 0x7f6f_bf50_7f6f_bf50u64),
            (u32::MAX, 0xaaaa_aaaa, 0x6666_6666_6666_6666),
            (0x8000_0001, 0x8000_0001, 0x4000_0000_0000_0001),
        ] {
            let (l, r) = (Cv::U32(left), Cv::U32(right));
            let bits = |shift: u32| Some(Cv::U32((product >> shift) as u32));
            assert_eq!(part(CarrylessPart::Low, l.clone(), r.clone(), 32), bits(0));
            assert_eq!(
                part(CarrylessPart::High, l.clone(), r.clone(), 32),
                bits(32)
            );
            assert_eq!(part(CarrylessPart::Reversed, l, r, 32), bits(31));
        }
    }

    /// The AES round keys a 128- or 256-bit key expands to, each as its two
    /// doublewords, by the `aes64ks1i` and `aes64ks2` steps.
    fn aes_schedule(key: &[u8]) -> Vec<[u64; 2]> {
        let mut keys: Vec<[u64; 2]> = key.chunks(16).map(halves).collect();
        let long = keys.len() == 2;
        let mut round = 0;
        while keys.len() < if long { 15 } else { 11 } {
            let previous = keys[keys.len() - 1];
            let base = keys[keys.len() - if long { 2 } else { 1 }];
            // A 256-bit key's odd keys substitute without rotating or adding
            // a constant: round number ten.
            let number = if long && keys.len() % 2 == 1 {
                10
            } else {
                round
            };
            let mixed = step(ScalarCipherKind::Aes64KeyStep1, &[previous[1], number]);
            let low = step(ScalarCipherKind::Aes64KeyStep2, &[mixed, base[0]]);
            let high = step(ScalarCipherKind::Aes64KeyStep2, &[low, base[1]]);
            keys.push([low, high]);
            if number != 10 {
                round += 1;
            }
        }
        keys
    }

    /// **The AES steps compose the cipher.** For keys and blocks OpenSSL
    /// enciphered (`openssl enc -aes-128-ecb`, `-aes-256-ecb`) and FIPS-197's
    /// Appendix C vectors, the 64-bit steps encipher each block to OpenSSL's
    /// ciphertext and decipher it back by the equivalent inverse cipher, its
    /// keys taken through `aes64im`; the 32-bit column steps, RV32's, do the
    /// same column by column.
    #[test]
    fn the_aes_steps_compose_the_cipher() {
        for (key, plain, cipher) in [
            (
                "000102030405060708090a0b0c0d0e0f",
                "00112233445566778899aabbccddeeff",
                "69c4e0d86a7b0430d8cdb78070b4c55a",
            ),
            (
                "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                "00112233445566778899aabbccddeeff",
                "8ea2b7ca516745bfeafc49904b496089",
            ),
            (
                "ce81472e57a080d5d9c26df109823c8d",
                "b75548c71a547c54f8ed436db4fae213",
                "84a0d7a56d5b6dcb4b97533725574dd0",
            ),
            (
                "550c65699018b738d3f2be5ec70bbcf3",
                "16e097b637a591e3c33a91c74a752bce",
                "337f356157d8db044de56fa26783737b",
            ),
            (
                "595feb3a3b24cc4a26e57033e1b4152d",
                "858108ce1f241843135403012e91c764",
                "0d27ff3007b7e218ae4156803facc7d3",
            ),
            (
                "ac707895e971847651520a1dafb3f36b",
                "c984ea9d0513a6351fa650129e913b88",
                "cb1d7f5ef42652e6defe3ae9720877f3",
            ),
            (
                "31f8ae8e0e6269c6ab0e4be9bad11c67f0e253cb40424bf768721af065474167",
                "70d173cb1b7a7a26d25150f771e0f31d",
                "bdc7e5ae0115dcb99bbab8137488a645",
            ),
            (
                "0479f284add9fee2259781d288f8629371444e0b50144902c8626629f3833207",
                "c32018a98eb858d8f61f58238c91d3db",
                "07bd908d59421ec9bb252dfad0f5e497",
            ),
            (
                "e22bf5507abf4cfe6b372143995346a6f4ef494ca59623a1f61d6e509fe862b6",
                "3a50e07514374fb625abdb3c94526eec",
                "64c7f552c364e9d1dc317243944666db",
            ),
            (
                "bef0f06f777a3d37f1abb28e34f53dc705382fec8486f39f7592b2665ad11e77",
                "8e79e334417b856e73e87175c678b3f2",
                "95b34c11fc2b0162e762472d9f38a8d0",
            ),
        ] {
            let keys = aes_schedule(&hex(key));
            let rounds = keys.len() - 1;
            let (plain, cipher) = (halves(&hex(plain)), halves(&hex(cipher)));
            // The doubleword steps.
            let mut state = [plain[0] ^ keys[0][0], plain[1] ^ keys[0][1]];
            for (round, key) in keys.iter().enumerate().skip(1) {
                let kind = if round == rounds {
                    ScalarCipherKind::Aes64EncryptHalf
                } else {
                    ScalarCipherKind::Aes64EncryptHalfMix
                };
                state = [
                    step(kind, &[state[0], state[1]]) ^ key[0],
                    step(kind, &[state[1], state[0]]) ^ key[1],
                ];
            }
            assert_eq!(state, cipher, "{key}: enciphered by aes64es/aes64esm");
            let mut state = [cipher[0] ^ keys[rounds][0], cipher[1] ^ keys[rounds][1]];
            for round in (0..rounds).rev() {
                let (kind, key) = if round == 0 {
                    (ScalarCipherKind::Aes64DecryptHalf, keys[0])
                } else {
                    let key = keys[round];
                    (
                        ScalarCipherKind::Aes64DecryptHalfMix,
                        [
                            step(ScalarCipherKind::Aes64InverseMix, &[key[0]]),
                            step(ScalarCipherKind::Aes64InverseMix, &[key[1]]),
                        ],
                    )
                };
                state = [
                    step(kind, &[state[0], state[1]]) ^ key[0],
                    step(kind, &[state[1], state[0]]) ^ key[1],
                ];
            }
            assert_eq!(state, plain, "{key}: deciphered by aes64ds/aes64dsm");
            // The column steps: output column `j` takes row `r` from column
            // `j + r` enciphering and `j - r` deciphering.
            let words = |halves: [u64; 2]| {
                [
                    halves[0] as u32,
                    (halves[0] >> 32) as u32,
                    halves[1] as u32,
                    (halves[1] >> 32) as u32,
                ]
            };
            let round_keys: Vec<[u32; 4]> = keys.iter().copied().map(words).collect();
            let mut state = words(plain);
            for (column, word) in state.iter_mut().enumerate() {
                *word ^= round_keys[0][column];
            }
            for (round, key) in round_keys.iter().enumerate().skip(1) {
                let kind = if round == rounds {
                    ScalarCipherKind::Aes32EncryptByte
                } else {
                    ScalarCipherKind::Aes32EncryptByteMix
                };
                state = std::array::from_fn(|j| {
                    (0..4).fold(key[j], |sum, r| {
                        column(kind, sum, state[(j + r) % 4], r as u8)
                    })
                });
            }
            assert_eq!(
                state,
                words(cipher),
                "{key}: enciphered by aes32esi/aes32esmi"
            );
            for (column, word) in state.iter_mut().enumerate() {
                *word ^= round_keys[rounds][column];
            }
            for round in (0..rounds).rev() {
                let (kind, key) = if round == 0 {
                    (ScalarCipherKind::Aes32DecryptByte, round_keys[0])
                } else {
                    let key = keys[round];
                    (
                        ScalarCipherKind::Aes32DecryptByteMix,
                        words([
                            step(ScalarCipherKind::Aes64InverseMix, &[key[0]]),
                            step(ScalarCipherKind::Aes64InverseMix, &[key[1]]),
                        ]),
                    )
                };
                state = std::array::from_fn(|j| {
                    (0..4).fold(key[j], |sum, r| {
                        column(kind, sum, state[(j + 4 - r) % 4], r as u8)
                    })
                });
            }
            assert_eq!(
                state,
                words(plain),
                "{key}: deciphered by aes32dsi/aes32dsmi"
            );
        }
        // A reserved round number names no step.
        assert_eq!(
            compute(
                Kernel::ScalarCipher(ScalarCipherKind::Aes64KeyStep1),
                &[Cv::U64(0), Cv::U64(11)],
                64
            ),
            None
        );
    }

    /// **The SM4 steps compose the cipher.** For keys and blocks OpenSSL
    /// enciphered (`openssl enc -sm4-ecb`) and the standard's example, the key
    /// schedule by `sm4ks` and the rounds by `sm4ed`, four byte steps each,
    /// give OpenSSL's ciphertext; at 64 bits a step writes its word
    /// sign-extended.
    #[test]
    fn the_sm4_steps_compose_the_cipher() {
        const FK: [u32; 4] = [0xa3b1_bac6, 0x56aa_3350, 0x677d_9197, 0xb270_22dc];
        let words = |bytes: &[u8]| -> [u32; 4] {
            std::array::from_fn(|at| {
                u32::from_be_bytes(bytes[4 * at..4 * at + 4].try_into().unwrap())
            })
        };
        let round = |kind: ScalarCipherKind, word: u32, mixed: u32| {
            (0..4).fold(word, |sum, select| column(kind, sum, mixed, select))
        };
        for (key, plain, cipher) in [
            (
                "0123456789abcdeffedcba9876543210",
                "0123456789abcdeffedcba9876543210",
                "681edf34d206965e86b3e94f536e4246",
            ),
            (
                "b186f8c1bbc52d411b22cb3e18389bea",
                "928330bc83aa1283b26551a634466d30",
                "182a45b81cbd848756130813af1903e2",
            ),
            (
                "0dfa4c6da98388c27a65ae3fe9810669",
                "caad8cfaedcf5b9ff01755f0695baaf5",
                "aef30ce2808de0850fd66ab9010a4a21",
            ),
            (
                "5c62502d6f22de8ff428ba8eabaa8161",
                "fdf611fbdc9ed69e5b2abcb11be569ac",
                "eef15753bbb6b0aa8dc02e5bf8734cb7",
            ),
            (
                "507af1ec66104c14875ca6eb75174fbe",
                "7e0ed204e1120d6ab0b9735d4cdcf2e9",
                "ffc8429740354393bd7dc0b509102524",
            ),
        ] {
            let mut keys: Vec<u32> = words(&hex(key))
                .iter()
                .zip(FK)
                .map(|(word, fk)| word ^ fk)
                .collect();
            for index in 0..32u32 {
                let ck = u32::from_be_bytes(std::array::from_fn(|byte| {
                    ((4 * index + byte as u32) * 7) as u8
                }));
                let at = keys.len() - 4;
                let mixed = keys[at + 1] ^ keys[at + 2] ^ keys[at + 3] ^ ck;
                keys.push(round(ScalarCipherKind::Sm4KeyByte, keys[at], mixed));
            }
            let mut state: Vec<u32> = words(&hex(plain)).to_vec();
            for key in &keys[4..] {
                let at = state.len() - 4;
                let mixed = state[at + 1] ^ state[at + 2] ^ state[at + 3] ^ key;
                state.push(round(ScalarCipherKind::Sm4EncryptByte, state[at], mixed));
            }
            let enciphered: Vec<u32> = state.iter().rev().take(4).copied().collect();
            assert_eq!(enciphered, words(&hex(cipher)), "{key}");
        }
        assert_eq!(
            step(ScalarCipherKind::Sm4EncryptByte, &[0x8000_0000, 0, 0]),
            i64::from(column(ScalarCipherKind::Sm4EncryptByte, 0x8000_0000, 0, 0).cast_signed())
                .cast_unsigned()
        );
    }

    /// **A modular conversion of a constant float folds.** Each double
    /// converts as an exact integer truncated toward zero and reduced modulo
    /// two to the width does (Python's `int(x) % 2**32` and `% 2**64`), a NaN
    /// and an infinity to zero; a binary128 past the doubleword keeps the low
    /// bits its significand holds.
    #[test]
    fn a_modular_conversion_of_a_constant_folds() {
        let double = |value: f64| Cv::Float(crate::ir::float::FloatBits::from(value));
        for (value, word, doubleword) in [
            (4_294_967_301.0, 0x5u32, 0x1_0000_0005u64),
            (-1.5, 0xffff_ffff, u64::MAX),
            (3e9, 0xb2d0_5e00, 0xb2d0_5e00),
            (
                2f64.powi(60) + 2f64.powi(31),
                0x8000_0000,
                0x1000_0000_8000_0000,
            ),
            (1e300, 0, 0),
            (-1e300, 0, 0),
            (4_294_967_295.9, 0xffff_ffff, 0xffff_ffff),
            (-2_147_483_648.0, 0x8000_0000, 0xffff_ffff_8000_0000),
            (2f64.powi(63), 0, 0x8000_0000_0000_0000),
            (1.5e19, 0xcedc_0000, 0xd02a_b486_cedc_0000),
            (-0.0, 0, 0),
            (f64::NAN, 0, 0),
            (f64::INFINITY, 0, 0),
            (f64::NEG_INFINITY, 0, 0),
        ] {
            assert_eq!(
                compute(Kernel::FloatToIntModular, &[double(value)], 32),
                Some(Cv::I32(word.cast_signed())),
                "{value}"
            );
            assert_eq!(
                compute(Kernel::FloatToIntModular, &[double(value)], 64),
                Some(Cv::I64(doubleword.cast_signed())),
                "{value}"
            );
        }
        let quad = crate::ir::float::FloatFormat::Quad;
        let wide = crate::ir::float::FloatBits::new(
            quad,
            quad.from_integer((1i128 << 100) + (1 << 40) + 7),
        )
        .unwrap();
        assert_eq!(wide.to_integer_modular(64), Some((1 << 40) + 7));
        assert_eq!(
            wide.negate().to_integer_modular(64),
            Some((1u128 << 64) - (1 << 40) - 7)
        );
    }

    /// **A vector length folds where the count determines it**: a count no
    /// greater than the maximum is the length, one at least twice the
    /// maximum gives the maximum; between them the implementation chooses.
    #[test]
    fn a_vector_length_folds_where_the_count_determines_it() {
        let choice = |count: u64, maximum: u64| {
            compute(
                Kernel::VectorLengthChoice,
                &[Cv::U64(count), Cv::U64(maximum)],
                64,
            )
        };
        assert_eq!(choice(5, 8), Some(Cv::U64(5)));
        assert_eq!(choice(8, 8), Some(Cv::U64(8)));
        assert_eq!(choice(16, 8), Some(Cv::U64(8)));
        assert_eq!(choice(u64::MAX, 8), Some(Cv::U64(8)));
        assert_eq!(choice(7, 0), Some(Cv::U64(0)));
        assert_eq!(choice(9, 8), None);
        assert_eq!(choice(15, 8), None);
    }

    /// **A computation constant inputs do not determine stays**: a random
    /// source, a pointer authentication, a memory tag and every support
    /// query, whatever their inputs, and a computation of a value no fold
    /// knows.
    #[test]
    fn a_computation_its_inputs_do_not_determine_stays() {
        let inputs = [Cv::U64(0x1000), Cv::U64(3), Cv::U64(1)];
        let undetermined = Kernel::identities().into_iter().filter(|kind| {
            matches!(
                kind,
                Kernel::Random { .. }
                    | Kernel::PointerAuth(_)
                    | Kernel::MemoryTag(_)
                    | Kernel::VectorShapeSupported
                    | Kernel::VectorFloatSupported
                    | Kernel::VectorCryptoSupported
            )
        });
        for kind in undetermined {
            for count in 0..=inputs.len() {
                assert_eq!(compute(kind, &inputs[..count], 64), None, "{kind:?}");
            }
        }
        let op = whole(Kernel::BitDeposit, vec![var(9)], vec![var(0), var(7)]);
        assert_eq!(MACHINE.evaluate(&op, Some(32), resolve), None);
    }

    /// **A packed float operation of constants folds in every lane.** Four
    /// singles `[1.5, 2.0, -1.0, 0.25]` and `[0.5, 4.0, 3.0, 8.0]` sum to
    /// `[2.0, 6.0, 2.0, 8.25]` and multiply to `[0.75, 8.0, -3.0, 2.0]`, two
    /// doubles likewise. A lane whose result the rounding modes disagree on
    /// (a third, or `-3.0 + 3.0`, a zero whose sign the mode chooses) or a
    /// NaN folds the whole vector to nothing.
    #[test]
    fn a_packed_float_operation_of_constants_folds_in_every_lane() {
        let singles = |values: [f32; 4]| {
            Cv::Vector(
                VectorConst::from_lanes(32, values.map(|value| u128::from(value.to_bits())))
                    .unwrap(),
            )
        };
        let doubles = |values: [f64; 2]| {
            Cv::Vector(
                VectorConst::from_lanes(64, values.map(|value| u128::from(value.to_bits())))
                    .unwrap(),
            )
        };
        let fold = |kind: VectorBinaryKind, bits: u32, left: Cv, right: Cv| {
            let op = SsaOp::<MockTarget>::VectorBinary {
                dest: var(2),
                left: var(0),
                right: var(1),
                kind,
                element: VectorElement {
                    kind: VectorElementKind::Float,
                    bits,
                    scalar: false,
                    signed: None,
                    rounding: None,
                },
                environment: None,
            };
            fold_lanes(&op, Destination::default(), &[left, right])
        };
        let (a, b) = (
            singles([1.5, 2.0, -1.0, 0.25]),
            singles([0.5, 4.0, 3.0, 8.0]),
        );
        assert_eq!(
            fold(VectorBinaryKind::Add, 32, a.clone(), b.clone()),
            Some(singles([2.0, 6.0, 2.0, 8.25]))
        );
        assert_eq!(
            fold(VectorBinaryKind::Mul, 32, a, b),
            Some(singles([0.75, 8.0, -3.0, 2.0]))
        );
        let (a, b) = (doubles([1.5, -2.0]), doubles([0.5, 0.5]));
        assert_eq!(
            fold(VectorBinaryKind::Sub, 64, a.clone(), b.clone()),
            Some(doubles([1.0, -2.5]))
        );
        assert_eq!(
            fold(VectorBinaryKind::Div, 64, a, b),
            Some(doubles([3.0, -4.0]))
        );
        let undecided = [
            fold(
                VectorBinaryKind::Div,
                32,
                singles([1.0, 2.0, 3.0, 4.0]),
                singles([3.0, 1.0, 1.0, 1.0]),
            ),
            fold(
                VectorBinaryKind::Add,
                32,
                singles([-3.0, 2.0, 3.0, 4.0]),
                singles([3.0, 1.0, 1.0, 1.0]),
            ),
            fold(
                VectorBinaryKind::Sub,
                64,
                doubles([f64::INFINITY, 1.0]),
                doubles([f64::INFINITY, 1.0]),
            ),
        ];
        assert_eq!(undecided, [None, None, None]);
    }

    /// **A 52-bit multiply-add of constants folds.** Each 64-bit lane is the
    /// accumulator plus the low, or the high, 52 bits of the product of the
    /// factors' low 52 bits, wrapped: what exact integer arithmetic gives
    /// (Python's), the high bits of each factor ignored.
    #[test]
    fn a_52_bit_multiply_add_of_constants_folds() {
        let pair = |values: [u64; 2]| {
            Cv::Vector(VectorConst::from_lanes(64, values.map(u128::from)).unwrap())
        };
        let madd = |high: bool| {
            SsaOp::<MockTarget>::contract(
                Contraction::MultiplyAdd52 { high },
                vec![var(9)],
                vec![var(1), var(2), var(0)],
            )
        };
        let known = [
            pair([0x0123_4567_89ab_cdef, u64::MAX]),
            pair([0xfedc_ba98_7654_3210, 0x000f_ffff_ffff_ffff]),
            pair([0x0f0f_0f0f_0f0f_0f0f, 0x000f_ffff_ffff_ffff]),
        ];
        assert_eq!(
            fold_lanes(&madd(false), Destination::default(), &known),
            Some(pair([0x012c_e013_4679_acdf, 0]))
        );
        assert_eq!(
            fold_lanes(&madd(true), Destination::default(), &known),
            Some(pair([0x012f_4051_6273_8494, 0x000f_ffff_ffff_fffd]))
        );
        let small = [pair([5, 5]), pair([3, 3]), pair([7, 7])];
        assert_eq!(
            fold_lanes(&madd(false), Destination::default(), &small),
            Some(pair([0x1a, 0x1a]))
        );
        assert_eq!(
            fold_lanes(&madd(true), Destination::default(), &small),
            Some(pair([5, 5]))
        );
    }

    /// **A complex multiply-add of constants folds where it is exact.** x86's
    /// `vfmaddcph` of `1 + 2i` and `3 + 4i` onto `0.5 + 0.25i` is `-4.5 +
    /// 10.25i` (the halves `0xc480`, `0x4920`), the accumulator its last
    /// input; a product binary16 cannot hold folds nothing
    /// ([`VectorConst::complex`]).
    #[test]
    fn a_complex_multiply_add_of_constants_folds_where_exact() {
        let halves = |values: [u16; 2]| {
            Cv::Vector(VectorConst::from_lanes(16, values.map(u128::from)).unwrap())
        };
        let madd = SsaOp::<MockTarget>::contract(
            Contraction::Complex(ComplexShape {
                product: ComplexProduct::Multiply {
                    conjugate: false,
                    accumulates: true,
                },
                element: VectorElement::float(16),
                saturates: false,
            }),
            vec![var(9)],
            vec![var(0), var(1), var(2)],
        );
        let known = [
            halves([0x3c00, 0x4000]),
            halves([0x4200, 0x4400]),
            halves([0x3800, 0x3400]),
        ];
        assert_eq!(
            fold_lanes(&madd, Destination::default(), &known),
            Some(halves([0xc480, 0x4920]))
        );
        let inexact = [
            halves([0x3c01, 0x0000]),
            halves([0x3c01, 0x0000]),
            halves([0x0000, 0x0000]),
        ];
        assert_eq!(fold_lanes(&madd, Destination::default(), &inexact), None);
    }

    /// **A NaN converted between float formats is what the target's rule
    /// makes it.** The signaling double `0x7ff4000000000000` narrowed to a
    /// bfloat16 is, kept as x86 keeps it (quiet, its sign and the top of its
    /// payload), `0x7fe0`, and widened to the x87 format (`fld`) is
    /// `0x7fffe000000000000000`; RISC-V's canonical NaN is `0x7fc0` and the
    /// x87 `0x7fffc000000000000000`. A target that states no rule (Arm, whose
    /// `FPCR.DN` decides) folds nothing, and a number converts alike under
    /// every rule.
    #[test]
    fn a_nan_converted_between_formats_is_the_targets_rule() {
        let double =
            |bits: u64| Cv::Float(FloatBits::new(FloatFormat::Double, u128::from(bits)).unwrap());
        let convert = |target: MockType, operand: &Cv, nan_conversion| {
            let rules = FloatRules {
                out_of_range: None,
                default_nan: Some(DefaultNan::Positive),
                nan_conversion,
                ..FloatRules::default()
            };
            let op = SsaOp::<MockTarget>::FloatConv {
                dest: var(1),
                operand: var(0),
                target,
                rounding: Some(FloatRounding::NearestEven),
                source: None,
                environment: None,
            };
            Machine::new(PointerSize::Bit64, None, rules)
                .evaluate(&op, Destination::default(), |id| {
                    (id == var(0)).then(|| operand.clone())
                })
                .and_then(|value| match value {
                    Cv::Float(value) => Some(value.bits()),
                    _ => None,
                })
        };
        let signaling = double(0x7ff4_0000_0000_0000);
        let operand = Some(NanPropagation::Operand);
        let default = Some(NanPropagation::Default);
        assert_eq!(convert(MockType::BF16, &signaling, operand), Some(0x7fe0));
        assert_eq!(
            convert(MockType::F80, &signaling, operand),
            Some(0x7fff_e000_0000_0000_0000)
        );
        assert_eq!(convert(MockType::BF16, &signaling, default), Some(0x7fc0));
        assert_eq!(
            convert(MockType::F80, &signaling, default),
            Some(0x7fff_c000_0000_0000_0000)
        );
        assert_eq!(convert(MockType::BF16, &signaling, None), None, "no rule");
        let one = double(1.0f64.to_bits());
        for rule in [operand, default, None] {
            assert_eq!(convert(MockType::BF16, &one, rule), Some(0x3f80));
        }
    }

    /// **A scalar float fold honours the flush mode the program may set.**
    /// The smallest subnormal compares above zero and adds to itself in IEEE
    /// arithmetic, and those fold; where the program may flush subnormals it
    /// may read as zero, so neither the comparison nor the sum folds, as the
    /// same operations on one lane do not, and so too a conversion of a
    /// subnormal operand or to a subnormal result. A normal value folds
    /// either way.
    #[test]
    fn a_scalar_float_fold_honours_the_flush_mode() {
        let subnormal = Cv::Float(FloatBits::from(f64::from_bits(1)));
        let zero = Cv::Float(FloatBits::from(0.0f64));
        let two = Cv::Float(FloatBits::from(2.0f64));
        let greater = SsaOp::<MockTarget>::Cgt {
            dest: var(9),
            left: var(0),
            right: var(1),
            unsigned: false,
            signaling: false,
        };
        let sum = SsaOp::<MockTarget>::Add {
            dest: var(9),
            left: var(0),
            right: var(0),
            flags: None,
        };
        let fold = |flushes: bool, op: &SsaOp<MockTarget>, known: &[Cv]| {
            let rules = FloatRules {
                program_flushes_subnormals: flushes,
                ..FloatRules::default()
            };
            Machine::new(PointerSize::Bit64, None, rules).evaluate(
                op,
                Destination::default(),
                |id| known.get(id.index()).cloned(),
            )
        };
        let inputs = [subnormal.clone(), zero.clone()];
        assert_eq!(fold(false, &greater, &inputs), Some(Cv::from_bool(true)));
        assert_eq!(
            fold(true, &greater, &inputs),
            None,
            "a flushed subnormal is zero"
        );
        assert_eq!(
            fold(false, &sum, slice::from_ref(&subnormal)),
            Some(Cv::Float(FloatBits::from(f64::from_bits(2))))
        );
        assert_eq!(
            fold(true, &sum, &[subnormal]),
            None,
            "a flushed sum is zero"
        );
        assert_eq!(
            fold(true, &greater, &[two.clone(), zero]),
            Some(Cv::from_bool(true)),
            "a normal value compares the same in every mode"
        );
        assert_eq!(
            fold(true, &sum, slice::from_ref(&two)),
            Some(Cv::Float(FloatBits::from(4.0f64)))
        );
        // A conversion reads a subnormal operand, and writes a subnormal
        // result, as the mode does: the least single widened, and the double
        // 2^-130 narrowed to a bfloat16's subnormal.
        let convert = |target| SsaOp::<MockTarget>::FloatConv {
            dest: var(9),
            operand: var(0),
            target,
            rounding: None,
            source: None,
            environment: None,
        };
        let least = Cv::Float(FloatBits::from(f32::from_bits(1)));
        let widened = Cv::Float(FloatBits::from(f64::from(f32::from_bits(1))));
        assert_eq!(
            fold(false, &convert(MockType::F64), slice::from_ref(&least)),
            Some(widened)
        );
        assert_eq!(
            fold(true, &convert(MockType::F64), &[least]),
            None,
            "a flushed operand is zero"
        );
        let tiny = Cv::Float(FloatBits::from(2.0f64.powi(-130)));
        assert_eq!(
            fold(false, &convert(MockType::BF16), slice::from_ref(&tiny)),
            FloatBits::new(FloatFormat::BFloat16, 0x0008).map(Cv::Float)
        );
        assert_eq!(
            fold(true, &convert(MockType::BF16), &[tiny]),
            None,
            "a flushed result is zero"
        );
        assert_eq!(
            fold(true, &convert(MockType::BF16), &[two]),
            FloatBits::new(FloatFormat::BFloat16, 0x4000).map(Cv::Float)
        );
    }

    /// **A raise of constant operands folds to the flags it sets.** A raise
    /// of `1.0 / 3.0` is inexact and of `1.0 + 1.0` nothing, in every mode
    /// the program may set, at RISC-V's bits; x87's `C1`, kept at bit 9, is
    /// set where the stated rounding rounds the quotient up and clear where it
    /// rounds down, and with no rounding stated (which decides `C1`) folds
    /// nothing. A subnormal operand, whose input-denormal flag a flush mode
    /// decides, and a non-constant operand fold nothing.
    #[test]
    fn a_raise_of_constant_operands_folds_to_the_flags_it_sets() {
        let doubles = |value: f64| {
            Cv::Vector(VectorConst::from_lanes(64, [u128::from(value.to_bits())]).unwrap())
        };
        let raise = |kind, rounding, layout| SsaOp::<MockTarget>::FloatRaise {
            dest: var(9),
            op: Box::new(SsaOp::VectorBinary {
                dest: var(8),
                left: var(0),
                right: var(1),
                kind,
                element: VectorElement {
                    scalar: true,
                    rounding,
                    ..VectorElement::float(64)
                },
                environment: None,
            }),
            lanes: None,
            raises: FloatExceptions::ALL,
            layout,
        };
        let rules = FloatRules {
            program_flushes_subnormals: true,
            ..FloatRules::default()
        };
        let fold = |op: &SsaOp<MockTarget>, known: &[Cv]| {
            Machine::new(PointerSize::Bit64, None, rules).evaluate(
                op,
                Destination::from(Some(32)),
                |id| known.get(id.index()).cloned(),
            )
        };
        let third = [doubles(1.0), doubles(3.0)];
        let ieee = FloatExceptionLayout::IEEE;
        assert_eq!(
            fold(&raise(VectorBinaryKind::Div, None, ieee), &third),
            Some(Cv::U32(1)),
            "a third is inexact in every mode"
        );
        assert_eq!(
            fold(
                &raise(VectorBinaryKind::Add, None, ieee),
                &[doubles(1.0), doubles(1.0)]
            ),
            Some(Cv::U32(0))
        );
        let x87 = FloatExceptionLayout::new([Some(5), Some(4), Some(3), Some(2), Some(0), Some(1)])
            .rounded_up_at(9);
        assert_eq!(
            fold(
                &raise(VectorBinaryKind::Div, Some(FloatRounding::Up), x87),
                &third
            ),
            Some(Cv::U32(1 << 5 | 1 << 9)),
            "C1 set: rounded up"
        );
        assert_eq!(
            fold(
                &raise(VectorBinaryKind::Div, Some(FloatRounding::Down), x87),
                &third
            ),
            Some(Cv::U32(1 << 5)),
            "C1 clear: rounded down"
        );
        assert_eq!(
            fold(&raise(VectorBinaryKind::Div, None, x87), &third),
            None,
            "the mode decides C1"
        );
        assert_eq!(
            fold(
                &raise(VectorBinaryKind::Add, None, ieee),
                &[doubles(f64::from_bits(1)), doubles(1.0)]
            ),
            None,
            "a subnormal operand"
        );
        assert_eq!(
            fold(&raise(VectorBinaryKind::Div, None, ieee), &[doubles(1.0)]),
            None,
            "control: a non-constant operand"
        );
    }

    /// **An invalid float operation of numbers is the target's default
    /// NaN.** Infinity times zero, infinity minus infinity, zero by zero, the
    /// square root of a negative number, a fused zero times infinity and a
    /// sum meeting both infinities give the NaN the rules state, positive
    /// (Arm, RISC-V), negative (x86) or MIPS's legacy one, lane by lane beside
    /// the lanes that are numbers; with no rule, and where an operand is
    /// already a NaN, whose propagation each target does its own way, they
    /// fold nothing.
    #[test]
    fn an_invalid_float_operation_is_the_targets_default_nan() {
        let doubles = |values: [u64; 2]| {
            Cv::Vector(VectorConst::from_lanes(64, values.map(u128::from)).unwrap())
        };
        let bits = |value: f64| value.to_bits();
        let rules = |nan: Option<DefaultNan>| FloatRules {
            out_of_range: None,
            default_nan: nan,
            ..FloatRules::default()
        };
        let element = VectorElement::float(64);
        let fold = |op: &SsaOp<MockTarget>, known: &[Cv], nan: Option<DefaultNan>| {
            Machine::new(PointerSize::Bit64, None, rules(nan)).evaluate(
                op,
                Destination::default(),
                |id| known.get(id.index()).cloned(),
            )
        };
        let binary = |kind| SsaOp::<MockTarget>::VectorBinary {
            dest: var(9),
            left: var(0),
            right: var(1),
            kind,
            element,
            environment: None,
        };
        let (positive, negative) = (0x7ff8_0000_0000_0000u64, 0xfff8_0000_0000_0000u64);
        let products = [
            doubles([bits(f64::INFINITY), bits(2.0)]),
            doubles([bits(0.0), bits(3.0)]),
        ];
        let mul = binary(VectorBinaryKind::Mul);
        assert_eq!(
            fold(&mul, &products, Some(DefaultNan::Positive)),
            Some(doubles([positive, bits(6.0)]))
        );
        assert_eq!(
            fold(&mul, &products, Some(DefaultNan::Negative)),
            Some(doubles([negative, bits(6.0)]))
        );
        assert_eq!(
            fold(&mul, &products, Some(DefaultNan::MipsLegacy)),
            Some(doubles([0x7ff7_ffff_ffff_ffff, bits(6.0)]))
        );
        assert_eq!(fold(&mul, &products, None), None, "no rule");
        let infinities = [
            doubles([bits(f64::INFINITY), bits(5.0)]),
            doubles([bits(f64::INFINITY), bits(2.0)]),
        ];
        assert_eq!(
            fold(
                &binary(VectorBinaryKind::Sub),
                &infinities,
                Some(DefaultNan::Positive)
            ),
            Some(doubles([positive, bits(3.0)]))
        );
        assert_eq!(
            fold(
                &binary(VectorBinaryKind::Div),
                &infinities,
                Some(DefaultNan::Negative)
            ),
            Some(doubles([negative, bits(2.5)]))
        );
        let with_nan = [
            doubles([bits(f64::NAN), bits(1.0)]),
            doubles([bits(1.0), bits(1.0)]),
        ];
        assert_eq!(
            fold(
                &binary(VectorBinaryKind::Add),
                &with_nan,
                Some(DefaultNan::Positive)
            ),
            None,
            "a NaN operand propagates as each target says"
        );
        let sqrt = SsaOp::<MockTarget>::VectorUnary {
            dest: var(9),
            value: var(0),
            kind: VectorUnaryKind::Sqrt,
            element,
            environment: None,
        };
        assert_eq!(
            fold(
                &sqrt,
                &[doubles([bits(-4.0), bits(4.0)])],
                Some(DefaultNan::Positive)
            ),
            Some(doubles([positive, bits(2.0)]))
        );
        let fused = SsaOp::<MockTarget>::VectorTernary {
            dest: var(9),
            first: var(0),
            second: var(1),
            third: var(2),
            kind: VectorTernaryKind::Fma,
            element,
            environment: None,
        };
        let fma = [
            doubles([bits(f64::INFINITY), bits(2.0)]),
            doubles([bits(0.0), bits(3.0)]),
            doubles([bits(1.0), bits(1.0)]),
        ];
        assert_eq!(
            fold(&fused, &fma, Some(DefaultNan::Negative)),
            Some(doubles([negative, bits(7.0)]))
        );
        assert_eq!(fold(&fused, &fma, None), None);
        let scalar = Destination {
            bits: Some(64),
            lanes: None,
            lane_format: None,
            integer: false,
            integer_type: None,
        };
        let reduce = |ordered: bool| SsaOp::<MockTarget>::VectorReduce {
            dest: var(9),
            value: var(0),
            kind: VectorReduceKind::Add,
            element: element.rounded(FloatRounding::NearestEven),
            initial: None,
            ordered,
            groups: None,
            environment: None,
        };
        let four = Cv::Vector(
            VectorConst::from_lanes(
                64,
                [
                    bits(1.0),
                    bits(f64::INFINITY),
                    bits(f64::NEG_INFINITY),
                    bits(2.0),
                ]
                .map(u128::from),
            )
            .unwrap(),
        );
        for ordered in [true, false] {
            assert_eq!(
                Machine::new(PointerSize::Bit64, None, rules(Some(DefaultNan::Positive))).evaluate(
                    &reduce(ordered),
                    scalar,
                    |id| (id.index() == 0).then(|| four.clone())
                ),
                Some(Cv::float(FloatFormat::Double, u128::from(positive)).unwrap()),
                "ordered {ordered}"
            );
        }
        assert_eq!(
            DefaultNan::Negative.bits(FloatFormat::Single),
            Some(0xffc0_0000)
        );
        assert_eq!(
            DefaultNan::MipsLegacy.bits(FloatFormat::Single),
            Some(0x7fbf_ffff)
        );
        assert_eq!(DefaultNan::MipsLegacy.bits(FloatFormat::Half), None);
    }

    /// **A lane of a lane-wise operation folds where its operands' lane is
    /// known.** Of `[x0, 5, x2, x3] + [10, 20, 30, 40]`, an argument's vector
    /// with a constant inserted at lane 1, lane 1 is 25 and lane 0 is not
    /// known; of a float product, a lane whose operands are an inserted 3.0 and
    /// a constant's 2.0 is 6.0; a scalar operation's lane above zero is its
    /// first operand's.
    #[test]
    fn a_lane_of_a_lane_wise_operation_folds_where_its_lane_is_known() {
        let words = Cv::Vector(VectorConst::from_lanes(32, [10, 20, 30, 40]).unwrap());
        let element = VectorElement {
            kind: VectorElementKind::Integer,
            bits: 32,
            scalar: false,
            signed: None,
            rounding: None,
        };
        let (ssa, v) = function_of(
            &[
                MockType::V4I32,
                MockType::I32,
                MockType::V4I32,
                MockType::V4I32,
                MockType::V4I32,
                MockType::I32,
                MockType::I32,
                MockType::V4I32,
                MockType::I32,
            ],
            |v| {
                vec![
                    SsaOp::Const {
                        dest: v[1],
                        value: Cv::I32(5),
                    },
                    SsaOp::VectorInsert {
                        dest: v[2],
                        vector: v[0],
                        lane: 1,
                        value: v[1],
                    },
                    SsaOp::Const {
                        dest: v[3],
                        value: words.clone(),
                    },
                    SsaOp::VectorBinary {
                        dest: v[4],
                        left: v[2],
                        right: v[3],
                        kind: VectorBinaryKind::Add,
                        element,
                        environment: None,
                    },
                    SsaOp::VectorExtract {
                        dest: v[5],
                        vector: v[4],
                        lane: 1,
                    },
                    SsaOp::VectorExtract {
                        dest: v[6],
                        vector: v[4],
                        lane: 0,
                    },
                    SsaOp::VectorBinary {
                        dest: v[7],
                        left: v[3],
                        right: v[2],
                        kind: VectorBinaryKind::Add,
                        element: VectorElement {
                            scalar: true,
                            ..element
                        },
                        environment: None,
                    },
                    SsaOp::VectorExtract {
                        dest: v[8],
                        vector: v[7],
                        lane: 2,
                    },
                ]
            },
        );
        let constants: BTreeMap<SsaVarId, Cv> = BTreeMap::from([(v[1], Cv::I32(5)), (v[3], words)]);
        let fold = |var: SsaVarId| {
            let op = ssa.get_definition(var).cloned().unwrap();
            MACHINE.fold_known_bits(&ssa, &op, &constants)
        };
        assert_eq!(fold(v[5]), Some(Cv::I32(25)));
        assert_eq!(fold(v[6]), None, "lane 0 is the argument's");
        assert_eq!(
            fold(v[8]),
            Some(Cv::I32(30)),
            "a scalar add keeps lane 2 of its first operand"
        );
        let (ssa, v) = function_of(
            &[
                MockType::V2F64,
                MockType::F64,
                MockType::V2F64,
                MockType::V2F64,
                MockType::V2F64,
                MockType::F64,
            ],
            |v| {
                vec![
                    SsaOp::Const {
                        dest: v[1],
                        value: Cv::from(3.0f64),
                    },
                    SsaOp::VectorInsert {
                        dest: v[2],
                        vector: v[0],
                        lane: 1,
                        value: v[1],
                    },
                    SsaOp::Const {
                        dest: v[3],
                        value: Cv::Vector(
                            VectorConst::from_lanes(64, [0, u128::from(2f64.to_bits())]).unwrap(),
                        ),
                    },
                    SsaOp::VectorBinary {
                        dest: v[4],
                        left: v[2],
                        right: v[3],
                        kind: VectorBinaryKind::Mul,
                        element: VectorElement::float(64),
                        environment: None,
                    },
                    SsaOp::VectorExtract {
                        dest: v[5],
                        vector: v[4],
                        lane: 1,
                    },
                ]
            },
        );
        let op = ssa.get_definition(v[5]).cloned().unwrap();
        assert_eq!(
            MACHINE.fold_known_bits(&ssa, &op, &BTreeMap::new()),
            Some(Cv::from(6.0f64))
        );
    }

    /// **Bits read out of a register merged over an unknown one fold.** The
    /// low quadword of `v0` (an argument) is the constant `3.0`; its high
    /// quadword is the argument's. A bitwise `and` over 128-bit lanes keeps
    /// the known bits known, so its low quadword folds; a select's lane is its chosen operand's, and a
    /// permute's lane the table lane its index selects. What reads the
    /// unknown quadword folds nothing.
    #[test]
    fn bits_of_a_register_merged_over_an_unknown_one_fold() {
        let whole = VectorElement {
            kind: VectorElementKind::Integer,
            bits: 128,
            scalar: false,
            signed: None,
            rounding: None,
        };
        let (ssa, v) = function_of(
            &[
                MockType::V2F64,
                MockType::F64,
                MockType::V2F64,
                MockType::V2F64,
                MockType::V2F64,
                MockType::F64,
                MockType::F64,
                MockType::V2F64,
                MockType::V2F64,
                MockType::F64,
                MockType::F64,
                MockType::V2F64,
                MockType::V2F64,
                MockType::F64,
                MockType::F64,
            ],
            |v| {
                vec![
                    SsaOp::Const {
                        dest: v[1],
                        value: Cv::from(3.0f64),
                    },
                    SsaOp::VectorInsert {
                        dest: v[2],
                        vector: v[0],
                        lane: 0,
                        value: v[1],
                    },
                    SsaOp::Const {
                        dest: v[3],
                        value: Cv::Vector(
                            VectorConst::from_lanes(64, [u128::from(u64::MAX), 0]).unwrap(),
                        ),
                    },
                    SsaOp::VectorBinary {
                        dest: v[4],
                        left: v[2],
                        right: v[3],
                        kind: VectorBinaryKind::And,
                        element: whole,
                        environment: None,
                    },
                    SsaOp::VectorExtract {
                        dest: v[5],
                        vector: v[4],
                        lane: 0,
                    },
                    SsaOp::VectorExtract {
                        dest: v[6],
                        vector: v[4],
                        lane: 1,
                    },
                    // The mask picks the second operand's lane 0 and the
                    // first's lane 1.
                    SsaOp::Const {
                        dest: v[7],
                        value: Cv::Vector(VectorConst::from_lanes(64, [1 << 63, 0]).unwrap()),
                    },
                    SsaOp::VectorTernary {
                        dest: v[8],
                        first: v[0],
                        second: v[2],
                        third: v[7],
                        kind: VectorTernaryKind::Select,
                        element: VectorElement::float(64),
                        environment: None,
                    },
                    SsaOp::VectorExtract {
                        dest: v[9],
                        vector: v[8],
                        lane: 0,
                    },
                    SsaOp::VectorExtract {
                        dest: v[10],
                        vector: v[8],
                        lane: 1,
                    },
                    // Index 2 is the second table's lane 0, 1 the first's
                    // lane 1.
                    SsaOp::Const {
                        dest: v[11],
                        value: Cv::Vector(VectorConst::from_lanes(64, [2, 1]).unwrap()),
                    },
                    SsaOp::VectorShuffle {
                        dest: v[12],
                        left: v[0],
                        right: Some(v[2]),
                        selector: VectorShuffleSelector::Index(v[11]),
                    },
                    SsaOp::VectorExtract {
                        dest: v[13],
                        vector: v[12],
                        lane: 0,
                    },
                    SsaOp::VectorExtract {
                        dest: v[14],
                        vector: v[12],
                        lane: 1,
                    },
                ]
            },
        );
        let constants: BTreeMap<SsaVarId, Cv> = [1, 3, 7, 11]
            .into_iter()
            .map(|index| {
                let op = ssa.get_definition(v[index]).cloned().unwrap();
                let SsaOp::Const { value, .. } = op else {
                    unreachable!()
                };
                (v[index], value)
            })
            .collect();
        let fold = |var: SsaVarId| {
            let op = ssa.get_definition(var).cloned().unwrap();
            MACHINE.fold_known_bits(&ssa, &op, &constants)
        };
        let three = Some(Cv::from(3.0f64));
        assert_eq!(fold(v[5]), three, "the low quadword of the and");
        assert_eq!(fold(v[6]), None, "the argument's quadword of the and");
        assert_eq!(fold(v[4]), None, "the and as a whole");
        assert_eq!(fold(v[9]), three, "the select's second operand's lane");
        assert_eq!(fold(v[10]), None, "the select's first operand's lane");
        assert_eq!(fold(v[13]), three, "the permute's second table's lane");
        assert_eq!(fold(v[14]), None, "the permute's first table's lane");
        assert_eq!(fold(v[1]), None, "a constant folds to nothing new");
    }

    /// Checked evaluation distinguishes values, unknown input, overflow and invalid layouts.
    #[test]
    fn checked_arithmetic_has_explicit_value_and_control_outcomes() {
        for unsigned in [false, true] {
            let ops = [
                SsaOp::AddOvf {
                    dest: var(0),
                    left: var(1),
                    right: var(2),
                    unsigned,
                    flags: None,
                },
                SsaOp::SubOvf {
                    dest: var(0),
                    left: var(1),
                    right: var(2),
                    unsigned,
                    flags: None,
                },
                SsaOp::MulOvf {
                    dest: var(0),
                    left: var(1),
                    right: var(2),
                    unsigned,
                    flags: None,
                },
            ];
            for op in ops {
                let operands = |l: Cv, r: Cv| {
                    move |id| {
                        if id == var(1) {
                            Some(l.clone())
                        } else {
                            Some(r.clone())
                        }
                    }
                };
                let normal = operands(Cv::U8(2), Cv::U8(1));
                let outcome = MACHINE.evaluate_checked(&op, Some(8), normal.clone());
                assert!(matches!(outcome, Some(CheckedEvaluation::Value(_))));
                if let Some(CheckedEvaluation::Value(value)) = outcome {
                    assert_eq!(MACHINE.evaluate(&op, Some(8), normal), Some(value));
                }
                let (left, right) = match op {
                    SsaOp::SubOvf { .. } if unsigned => (0, 1),
                    SsaOp::SubOvf { .. } => (128, 1),
                    _ if unsigned => (255, 2),
                    _ => (127, 2),
                };
                assert_eq!(
                    MACHINE.evaluate_checked(&op, Some(8), operands(Cv::U8(left), Cv::U8(right))),
                    Some(CheckedEvaluation::Overflow)
                );
                assert_eq!(
                    MACHINE.evaluate_checked::<MockTarget>(&op, Some(8), |_| None),
                    Some(CheckedEvaluation::Unknown)
                );
                assert_eq!(
                    MACHINE
                        .evaluate_checked(&op, Some(8), |id| (id == var(2)).then_some(Cv::U8(1))),
                    Some(CheckedEvaluation::Unknown)
                );
                assert_eq!(
                    MACHINE.evaluate_checked(&op, Some(0), operands(Cv::U8(1), Cv::U8(2))),
                    None
                );
                assert_eq!(
                    MACHINE.evaluate_checked(&op, Some(256), operands(Cv::U8(1), Cv::U16(2))),
                    None
                );
                assert_eq!(
                    MACHINE.evaluate_checked(&op, None, operands(Cv::True, Cv::True)),
                    None
                );
            }
        }
        assert_eq!(
            MACHINE.evaluate_checked::<MockTarget>(&SsaOp::Nop, None, |_| None),
            None
        );
    }

    /// Checked operations use SSA width even when literal carriers are wider or narrower.
    #[test]
    fn checked_operations_normalize_carriers_before_overflow_detection() {
        let operands = |l: Cv, r: Cv| {
            move |id| {
                if id == var(1) {
                    Some(l.clone())
                } else {
                    Some(r.clone())
                }
            }
        };
        let ops = [
            SsaOp::AddOvf {
                dest: var(0),
                left: var(1),
                right: var(2),
                unsigned: true,
                flags: None,
            },
            SsaOp::SubOvf {
                dest: var(0),
                left: var(1),
                right: var(2),
                unsigned: true,
                flags: None,
            },
            SsaOp::MulOvf {
                dest: var(0),
                left: var(1),
                right: var(2),
                unsigned: true,
                flags: None,
            },
        ];
        for (op, l, r) in [
            (&ops[0], u64::from(u32::MAX), 1),
            (&ops[1], 0, 1),
            (&ops[2], u64::from(u32::MAX), 2),
        ] {
            let values = operands(Cv::U64(l), Cv::U64(r));
            assert_eq!(
                MACHINE.evaluate_checked(op, Some(32), values.clone()),
                Some(CheckedEvaluation::Overflow)
            );
            assert_eq!(MACHINE.evaluate(op, Some(32), values), None);
        }
        let sum = &ops[0];
        assert_eq!(
            MACHINE.evaluate(sum, Some(32), operands(Cv::I64(7), Cv::I64(11))),
            Some(Cv::I32(18))
        );
        assert_eq!(
            MACHINE.evaluate(sum, None, operands(Cv::U8(255), Cv::U16(1))),
            Some(Cv::U16(256))
        );
        assert_eq!(
            MACHINE.evaluate_checked(sum, Some(1), operands(Cv::I8(-1), Cv::I8(1))),
            Some(CheckedEvaluation::Overflow)
        );
        assert_eq!(
            MACHINE.evaluate(sum, Some(1), operands(Cv::I8(0), Cv::I8(1))),
            Cv::of_lane(1, 1)
        );
        let signed = SsaOp::AddOvf {
            dest: var(0),
            left: var(1),
            right: var(2),
            unsigned: false,
            flags: None,
        };
        assert_eq!(
            MACHINE.evaluate(&signed, Some(16), operands(Cv::I8(-1), Cv::I32(2))),
            Some(Cv::I16(1))
        );
        assert_eq!(
            MACHINE.evaluate_checked(&signed, Some(8), operands(Cv::I64(127), Cv::I64(1))),
            Some(CheckedEvaluation::Overflow)
        );
        assert_eq!(
            MACHINE.evaluate(&signed, Some(128), operands(Cv::I8(-1), Cv::I8(2))),
            Cv::of_lane(128, 1)
        );
    }

    /// Numeric structural addresses share modular arithmetic with static execution.
    #[test]
    fn numeric_pointer_operations_fold_at_machine_width() {
        for width in [
            PointerSize::Bit8,
            PointerSize::Bit16,
            PointerSize::Bit32,
            PointerSize::Bit64,
            PointerSize::Bit128,
        ] {
            let machine = Machine {
                ptr_size: width,
                ..MACHINE
            };
            let op = SsaOp::PtrAdd {
                dest: var(0),
                base: var(1),
                index: Some(var(2)),
                stride: 2,
                offset: -1,
                result_type: MockType::Ptr,
            };
            let result = machine.evaluate(&op, None, |id| {
                Some(if id == var(1) { Cv::U64(8) } else { Cv::I8(-1) })
            });
            assert_eq!(
                result
                    .as_ref()
                    .and_then(|value| value.scalar_integer_pattern(width)),
                Some((5, width.bits())),
            );
            if width.bits() <= 64 {
                assert_eq!(result, Some(Cv::NativeUInt(5)));
            } else {
                let high = 1u128 << 100;
                let result = machine.evaluate(&op, None, |id| {
                    if id == var(1) {
                        Cv::of_lane(128, high | 8)
                    } else {
                        Some(Cv::I8(-1))
                    }
                });
                assert_eq!(
                    result
                        .as_ref()
                        .and_then(|value| value.scalar_integer_pattern(width)),
                    Some((high | 5, 128)),
                    "wide pointers retain bits above a native integer carrier",
                );
            }
            assert_eq!(
                machine.evaluate(&op, None, |id| (id == var(1)).then_some(Cv::U64(8))),
                None
            );
        }
    }

    /// A symbol less itself, or xored with itself, is zero at every scalar
    /// width; two distinct symbols are no such pair.
    #[test]
    fn a_symbol_cancels_against_itself_at_every_scalar_width() {
        for bits in 1..=128 {
            let dest = Destination {
                bits: Some(bits),
                lanes: None,
                lane_format: None,
                integer: true,
                integer_type: None,
            };
            let ops = [
                SsaOp::Sub {
                    dest: var(0),
                    left: var(1),
                    right: var(2),
                    flags: None,
                },
                SsaOp::Xor {
                    dest: var(0),
                    left: var(1),
                    right: var(2),
                    flags: None,
                },
            ];
            for op in ops {
                for (right, same) in [(7, true), (8, false)] {
                    let reads = Cell::new(0usize);
                    let actual = MACHINE.evaluate_with_operand_layouts(
                        &op,
                        dest,
                        |id| {
                            assert!(id == var(1) || id == var(2));
                            reads.set(reads.get() + 1);
                            Some(Cv::Symbol(if id == var(1) { 7 } else { right }))
                        },
                        |_| OperandLayout::default(),
                    );
                    assert_eq!(reads.get(), 2);
                    if same {
                        if bits == 1 {
                            assert_eq!(actual, Some(Cv::False));
                        } else {
                            assert_eq!(
                                actual.as_ref().and_then(
                                    |value| value.scalar_integer_pattern(MACHINE.ptr_size)
                                ),
                                Some((0, bits))
                            );
                        }
                        if bits.is_multiple_of(8) {
                            assert_eq!(
                                actual
                                    .and_then(|value| value
                                        .to_bytes(Endianness::Little, MACHINE.ptr_size)),
                                Some(vec![0; (bits / 8) as usize])
                            );
                        }
                    } else {
                        assert_eq!(actual, None);
                    }
                }
            }
        }
    }

    /// Boolean readers preserve logical values and signed one-bit readers preserve bit patterns.
    #[test]
    fn boolean_operand_layout_preserves_logical_values_and_borrowing() {
        let boolean = OperandLayout::Boolean;
        let integer = OperandLayout::Register(Destination {
            bits: Some(1),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(1, true),
        });
        for (value, expected) in [
            (Cv::True, Cv::True),
            (Cv::False, Cv::False),
            (Cv::U64(1), Cv::True),
            (Cv::U64(0), Cv::False),
            (Cv::of_lane(1, 1).unwrap(), Cv::True),
            (Cv::of_lane(1, 0).unwrap(), Cv::False),
        ] {
            assert_eq!(
                boolean.read_numeric(value.clone(), PointerSize::Bit64),
                Some(expected.clone())
            );
            assert_eq!(
                boolean.read_conversion(value.clone(), PointerSize::Bit64),
                Some(expected.clone())
            );
            assert_eq!(
                boolean.read_comparison(value.clone(), PointerSize::Bit64),
                Some(expected.clone())
            );
            let numeric = boolean
                .read_numeric_ref(&value, PointerSize::Bit64)
                .unwrap();
            let comparison = boolean
                .read_comparison_ref(&value, PointerSize::Bit64)
                .unwrap();
            assert_eq!(numeric.as_ref(), &expected);
            assert_eq!(comparison.as_ref(), &expected);
            assert_eq!(matches!(numeric, Cow::Borrowed(_)), value.is_bool());
        }
        assert_eq!(
            integer.read_conversion(Cv::True, PointerSize::Bit64),
            Cv::of_lane(1, 1)
        );
        assert_eq!(
            boolean.read_numeric(Cv::from(1.0f64), PointerSize::Bit64),
            None
        );
        assert_eq!(
            boolean.read_numeric_ref(&Cv::Symbol(7), PointerSize::Bit64),
            None
        );
        let logical = TypeLayout::of::<MockTarget>(&MockType::Bool, PointerSize::Bit64);
        assert_eq!(
            OperandLayout::from_type_layout(logical),
            OperandLayout::Boolean
        );
        let signed = TypeLayout::of::<MockTarget>(&MockType::I1, PointerSize::Bit64);
        assert_eq!(OperandLayout::from_type_layout(signed), integer);
    }

    /// **A width-preserving operation folds only operands of its width.** A
    /// fixed-width carrier of another width is an ill-typed operand (the
    /// verifier's mismatched operand widths): `int32 % int64` stays unfolded
    /// rather than reading the `int64` at 32 bits. A width-free packed
    /// bitvector is read at the operation width.
    #[test]
    fn width_preserving_operations_refuse_mismatched_integer_carriers() {
        let (dest, left, right) = (var(2), var(0), var(1));
        let int32 = Destination {
            bits: Some(32),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(32, true),
        };
        let family = [
            SsaOp::<MockTarget>::Add {
                dest,
                left,
                right,
                flags: None,
            },
            SsaOp::Sub {
                dest,
                left,
                right,
                flags: None,
            },
            SsaOp::Mul {
                dest,
                left,
                right,
                flags: None,
            },
            SsaOp::Div {
                dest,
                left,
                right,
                unsigned: false,
                flags: None,
            },
            SsaOp::Rem {
                dest,
                left,
                right,
                unsigned: false,
                flags: None,
            },
            SsaOp::And {
                dest,
                left,
                right,
                flags: None,
            },
            SsaOp::Or {
                dest,
                left,
                right,
                flags: None,
            },
            SsaOp::Xor {
                dest,
                left,
                right,
                flags: None,
            },
        ];
        for op in &family {
            let fold = |destination: Destination, left: Cv, right: Cv| {
                MACHINE.evaluate(op, destination, |id: SsaVarId| match id.index() {
                    0 => Some(left.clone()),
                    1 => Some(right.clone()),
                    _ => None,
                })
            };
            // A bare width reads no operand at another width: the kernel's own
            // rules for mismatched carriers are the oracle.
            for (left, right) in [
                (Cv::I32(7), Cv::I64(2)),
                (Cv::I64(7), Cv::I32(2)),
                (Cv::I32(7), Cv::I16(2)),
            ] {
                assert_eq!(
                    fold(int32, left.clone(), right.clone()),
                    fold(Some(32).into(), left, right),
                    "{op:?}: mismatched carriers"
                );
            }
            assert!(
                fold(int32, Cv::I32(7), Cv::I32(2)).is_some(),
                "{op:?}: one width"
            );
            assert_eq!(
                fold(int32, Cv::I32(7), Cv::of_lane(32, 2).unwrap()),
                fold(int32, Cv::I32(7), Cv::I32(2)),
                "{op:?}: packed carrier"
            );
        }
        // Unverifiable `int32 % int64` stays unfolded.
        assert_eq!(
            MACHINE.evaluate(&family[4], int32, |id: SsaVarId| match id.index() {
                0 => Some(Cv::I32(7)),
                1 => Some(Cv::I64(2)),
                _ => None,
            }),
            None
        );
    }

    /// A physical integer result cannot turn an unknown floating input into integer division.
    #[test]
    fn partial_division_requires_the_declared_integer_domain() {
        let op = SsaOp::<MockTarget>::Div {
            dest: var(0),
            left: var(1),
            right: var(2),
            unsigned: false,
            flags: None,
        };
        let dest = Destination {
            bits: Some(32),
            lanes: None,
            lane_format: None,
            integer: true,
            integer_type: IntegerType::new(32, true),
        };
        let machine = MACHINE.with_division_rules(DivisionRules::ARM);
        for floating in [false, true] {
            let value = |value| (value == var(2)).then_some(Cv::I32(0));
            let layout = |value| {
                if floating && value == var(1) {
                    OperandLayout::Float(FloatFormat::Single)
                } else {
                    OperandLayout::Register(dest)
                }
            };
            assert_eq!(
                machine.evaluate_with_operand_layouts(&op, dest, value, layout),
                (!floating).then_some(Cv::I32(0))
            );
            assert_eq!(
                machine.evaluate_control_with_operand_layouts(&op, dest, value, layout),
                Some(if floating {
                    Evaluation::Unsupported
                } else {
                    Evaluation::Value(Cv::I32(0))
                })
            );
        }
    }
}
