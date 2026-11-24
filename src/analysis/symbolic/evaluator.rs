//! Symbolic evaluator for building expression trees from SSA operations.
//!
//! This module provides [`SymbolicEvaluator`], which tracks SSA operations
//! symbolically to build [`crate::analysis::symbolic::SymbolicExpr`] trees. Unlike
//! concrete evaluation (which computes known values), symbolic evaluation
//! preserves the relationship between operations as expression trees, enabling
//! host-side constraint solving.
//!
//! # Algorithm
//!
//! For each SSA operation, the evaluator:
//! 1. Looks up symbolic expressions for the operand variables
//! 2. If an operand has no expression (unknown), creates a `SymbolicExpr::Variable`
//!    reference to the operand's `SsaVarId`
//! 3. Combines operand expressions with the appropriate `NumericOp`
//! 4. Applies `simplify()` for constant folding
//! 5. Stores the result expression for the destination variable
//!
//! Slot loads retain their own unknown identity because memory may change.
//! Concrete inputs use the shared machine with the SSA's declared layouts.
//! Admitted scalar integer inputs retain their full delayed operation semantics.
//!
//! # Relationship to SsaEvaluator
//!
//! Both evaluators share concrete machine semantics and the `SymbolicExpr`
//! representation. `SsaEvaluator` additionally tracks constraints, control flow
//! and analysis state.

use rustc_hash::FxHashMap;

use crate::{
    PointerSize,
    analysis::{
        consts::{
            Destination, Machine, OperandLayout,
            scalar::{OperationLayout, ScalarOperation, nonreturn_terminal},
        },
        symbolic::{concrete, expr::SymbolicExpr},
    },
    ir::{
        evaluation::Evaluation, float::FloatRules, function::SsaFunction, ops::SsaOp,
        value::ConstValue, variable::SsaVarId,
    },
    target::Target,
};

/// Symbolic evaluator that builds expression trees from SSA operations.
///
/// Tracks unknown inputs as expression trees and evaluates concrete inputs
/// through the shared [`Machine`]. Host solvers can consume the resulting trees.
#[derive(Debug)]
pub struct SymbolicEvaluator<'a, T: Target> {
    /// Normal continuation knowledge for the evaluated prefix.
    execution: Evaluation<()>,
    /// A proved fault prevents later definitions even when the first fault is uncertain.
    halted: bool,
    ssa: &'a SsaFunction<T>,
    /// Expressions computed for each variable.
    expressions: FxHashMap<SsaVarId, SymbolicExpr<T>>,
    /// Target layout and explicitly supplied arithmetic policy.
    machine: Machine,
}

impl<'a, T: Target> SymbolicEvaluator<'a, T> {
    /// Creates a new symbolic evaluator for the given SSA function.
    ///
    /// The evaluator starts with no known expressions. Use [`set_symbolic`](Self::set_symbolic)
    /// or [`set_constant`](Self::set_constant) to initialize variable values before
    /// evaluating blocks.
    ///
    /// # Arguments
    ///
    /// * `ssa` - The SSA function to evaluate.
    /// * `ptr_size` - Target pointer size for native int/uint masking.
    ///
    /// # Returns
    ///
    /// A new evaluator with no expressions.
    #[must_use]
    pub fn new(ssa: &'a SsaFunction<T>, ptr_size: PointerSize) -> Self {
        Self::with_machine(ssa, Machine::new(ptr_size, None, FloatRules::default()))
    }

    /// Creates an evaluator using explicit target layout and floating policy.
    ///
    /// Concrete operands use the original SSA operation, its declared result
    /// layout and each input layout through [`Machine`].
    #[must_use]
    pub fn with_machine(ssa: &'a SsaFunction<T>, machine: Machine) -> Self {
        Self {
            ssa,
            execution: Evaluation::Value(()),
            halted: false,
            expressions: FxHashMap::default(),
            machine,
        }
    }

    /// Sets a variable to a named symbolic value.
    ///
    /// Named symbolic values represent external inputs whose concrete values
    /// are unknown. Use this to mark the "state" variable in control flow
    /// unflattening.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to set.
    /// * `name` - The symbolic name (e.g., "state").
    pub fn set_symbolic(&mut self, var: SsaVarId, name: impl Into<String>) {
        self.expressions.insert(var, SymbolicExpr::named(name));
    }

    /// Sets a variable to a typed constant value.
    ///
    /// Use this to provide known initial values for variables with type preservation.
    /// The caller is responsible for providing the correct `ConstValue` type.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to set.
    /// * `value` - The typed constant value.
    pub fn set_constant(&mut self, var: SsaVarId, value: ConstValue<T>) {
        self.expressions.insert(var, SymbolicExpr::constant(value));
    }

    /// Gets the expression for a variable, if known.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to look up.
    ///
    /// # Returns
    ///
    /// The expression for the variable, or `None` if not yet evaluated.
    #[must_use]
    pub fn get_expression(&self, var: SsaVarId) -> Option<&SymbolicExpr<T>> {
        self.expressions.get(&var)
    }

    /// Returns a copy with constant subexpressions folded.
    ///
    /// Returns a copy of the expression with constant folding and constant subexpressions evaluated.
    ///
    /// # Arguments
    ///
    /// * `var` - The SSA variable to look up.
    ///
    /// # Returns
    ///
    /// A simplified copy of the expression, or `None` if not yet evaluated.
    #[must_use]
    pub fn get_simplified(&self, var: SsaVarId) -> Option<SymbolicExpr<T>> {
        self.expressions
            .get(&var)
            .map(|e| e.simplify(self.machine.ptr_size))
    }

    /// Returns all computed expressions.
    ///
    /// # Returns
    ///
    /// A reference to the map from variable IDs to their symbolic expressions.
    #[must_use]
    pub fn expressions(&self) -> &FxHashMap<SsaVarId, SymbolicExpr<T>> {
        &self.expressions
    }

    /// Evaluates all instructions in a block symbolically.
    ///
    /// Processes each instruction in the block, building symbolic expressions
    /// for any variables they define.
    ///
    /// # Arguments
    ///
    /// * `block_idx` - The index of the block to evaluate.
    pub fn evaluate_block(&mut self, block_idx: usize) {
        let Some(block) = self.ssa.block(block_idx) else {
            return;
        };

        for instr in block.instructions() {
            self.evaluate_op(instr.op());
        }
    }

    /// Evaluates a sequence of blocks in order.
    ///
    /// # Arguments
    ///
    /// * `block_indices` - The indices of blocks to evaluate, in order.
    pub fn evaluate_blocks(&mut self, block_indices: &[usize]) {
        for &block_idx in block_indices {
            self.evaluate_block(block_idx);
        }
    }

    /// Evaluates a single SSA operation symbolically.
    ///
    /// Builds a symbolic expression for the operation's result based on its
    /// operands. If operands have known expressions, those are used; otherwise,
    /// the operands are treated as symbolic variables.
    ///
    /// # Arguments
    ///
    /// * `op` - The SSA operation to evaluate.
    pub fn evaluate_op(&mut self, op: &SsaOp<T>) {
        if self.halted {
            return;
        }
        if let Some(outcome) = nonreturn_terminal(op) {
            self.execution = outcome;
            self.halted = true;
            return;
        }
        let layout = OperationLayout::of(self.ssa, op, self.machine);
        let captured = layout.scalar;
        let outcome = self.machine.evaluate_control_with_operand_layouts(
            op,
            layout.destination,
            |var| {
                self.expressions
                    .get(&var)
                    .and_then(|expression| expression.as_constant().cloned())
            },
            |var| layout.operand(self.ssa, var),
        );
        if let Some(outcome) = outcome {
            match outcome {
                Evaluation::Value(_) => {}
                Evaluation::Trap(trap) => {
                    self.execution = if matches!(self.execution, Evaluation::Value(())) {
                        Evaluation::Trap(trap)
                    } else {
                        Evaluation::Unknown
                    };
                    self.halted = true;
                    for def in op.defs() {
                        self.expressions.remove(&def);
                    }
                    return;
                }
                Evaluation::Unknown => {
                    let normal = captured.is_some_and(|(descriptor, operands)| {
                        descriptor.returns_normally_with_count(operands[1].and_then(|var| {
                            self.expressions
                                .get(&var)
                                .and_then(SymbolicExpr::as_constant)
                        })) && operands.into_iter().flatten().all(|var| {
                            self.expressions.get(&var).is_none_or(|expression| {
                                expression.admits_normal_input(self.machine.ptr_size)
                            })
                        })
                    });
                    if !normal && matches!(self.execution, Evaluation::Value(())) {
                        self.execution = Evaluation::Unknown;
                    }
                }
                _ => self.execution = Evaluation::Unsupported,
            }
        } else if op.may_throw() && matches!(self.execution, Evaluation::Value(())) {
            self.execution = Evaluation::Unknown;
        }

        if let Some((descriptor, operands)) = captured {
            self.evaluate_scalar(op, descriptor, operands);
            return;
        }
        if self.evaluate_concrete(op) {
            return;
        }
        if let SsaOp::Const { dest, value } = op {
            self.expressions
                .insert(*dest, SymbolicExpr::constant(value.clone()));
        } else if let Some(dest) = op.dest() {
            // Unrepresented domains retain their own result identity. They
            // cannot be reconstructed using carrier-only integer operators.
            self.expressions.insert(dest, SymbolicExpr::variable(dest));
        }
    }

    /// Returns knowledge about normal continuation after the evaluated prefix.
    ///
    /// Relations after an uncertain operation are conditional on reaching their
    /// source definitions. A proved fault prevents subsequent definitions.
    #[must_use]
    pub fn execution_outcome(&self) -> &Evaluation<()> {
        &self.execution
    }

    /// Evaluates one operation and returns its primary value's full outcome.
    ///
    /// A retained symbolic relation can have an `Unknown` outcome. This value
    /// query does not establish whether subsequent source instructions execute.
    pub fn evaluate_op_outcome(&mut self, op: &SsaOp<T>) -> Evaluation<ConstValue<T>> {
        self.evaluate_op(op);
        if self.halted {
            return match self.execution {
                Evaluation::Trap(trap) => Evaluation::Trap(trap),
                Evaluation::Unsupported => Evaluation::Unsupported,
                _ => Evaluation::Unknown,
            };
        }
        op.dest()
            .and_then(|dest| self.expressions.get(&dest))
            .map_or(Evaluation::Unsupported, |expression| {
                expression.evaluate_outcome_with(self.machine.ptr_size, |_| Evaluation::Unknown)
            })
    }

    /// Builds the primary scalar relation after bounding operands before cloning.
    fn evaluate_scalar(
        &mut self,
        source: &SsaOp<T>,
        descriptor: ScalarOperation,
        operands: [Option<SsaVarId>; 2],
    ) {
        let (Some(dest), Some(first)) = (source.dest(), operands[0]) else {
            return;
        };
        let first_leaf = SymbolicExpr::variable(first);
        let first_expression = self.expressions.get(&first).unwrap_or(&first_leaf);
        let second_leaf = operands[1].map(SymbolicExpr::variable);
        let second_expression = operands[1]
            .and_then(|id| self.expressions.get(&id))
            .or(second_leaf.as_ref());
        let too_large = match second_expression {
            Some(second) => SymbolicExpr::combining_reaches_limits(&[first_expression, second]),
            None => SymbolicExpr::combining_reaches_limits(&[first_expression]),
        };
        if too_large {
            self.expressions.insert(dest, SymbolicExpr::variable(dest));
            return;
        }
        if matches!(source, SsaOp::Copy { .. })
            && self
                .ssa
                .variable(first)
                .zip(self.ssa.variable(dest))
                .is_some_and(|(first, dest)| first.var_type() == dest.var_type())
        {
            self.expressions.insert(dest, first_expression.clone());
            return;
        }
        let expression = match second_expression {
            Some(second) => {
                SymbolicExpr::typed_binary(descriptor, first_expression.clone(), second.clone())
            }
            None => SymbolicExpr::typed_unary(descriptor, first_expression.clone()),
        };
        if let Some(expression) = expression {
            self.expressions
                .insert(dest, expression.simplify(self.machine.ptr_size));
        }
    }

    /// Folds concrete numeric inputs using the original operation and declared layouts.
    ///
    /// A refused concrete fold cannot fall back to carrier arithmetic or a copy.
    /// Symbolic trees remain available when an input is genuinely unknown.
    fn evaluate_concrete(&mut self, op: &SsaOp<T>) -> bool {
        let Some(dest) = op.dest() else {
            return false;
        };
        let mut all_constant = true;
        op.for_each_use(|variable| {
            all_constant &= self
                .expressions
                .get(&variable)
                .and_then(SymbolicExpr::as_constant)
                .is_some();
        });
        if !all_constant {
            return false;
        }
        let result = if let SsaOp::Const { value, .. } = op {
            self.machine
                .normalize_literal(value.clone(), Destination::of_operation(self.ssa, op))
        } else if let SsaOp::Copy { src, .. } = op {
            self.expressions
                .get(src)
                .and_then(SymbolicExpr::as_constant)
                .and_then(concrete::operand)
                .and_then(|value| {
                    self.machine
                        .normalize_literal(value, Destination::of_operation(self.ssa, op))
                })
        } else {
            self.machine.evaluate_with_operand_layouts(
                op,
                Destination::of_operation(self.ssa, op),
                |variable| {
                    self.expressions
                        .get(&variable)
                        .and_then(SymbolicExpr::as_constant)
                        .and_then(concrete::operand)
                },
                |variable| OperandLayout::of(self.ssa, variable),
            )
        };
        self.expressions.insert(
            dest,
            result.map_or_else(|| SymbolicExpr::variable(dest), SymbolicExpr::constant),
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        PointerSize,
        analysis::symbolic::{SymbolicExpr, evaluator::SymbolicEvaluator},
        ir::{
            division::DivisionRules,
            evaluation::{ArithmeticTrap, Evaluation},
            function::SsaFunction,
            ops::SsaOp,
            value::ConstValue,
            variable::{DefSite, VariableOrigin},
        },
        testing::{MACHINE, MockTarget, MockType},
    };

    #[test]
    fn delayed_division_matches_immediate_machine_outcomes() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let left = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::instruction(0, 0),
            MockType::U64,
        );
        let right = ssa.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::instruction(0, 1),
            MockType::U64,
        );
        let dest = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 2),
            MockType::U64,
        );
        let source = SsaOp::Div {
            dest,
            left,
            right,
            unsigned: true,
            flags: None,
        };
        for division in [
            DivisionRules::CHECKED,
            DivisionRules::ARM,
            DivisionRules::RISCV,
            DivisionRules::UNSTATED,
        ] {
            let machine = MACHINE.with_division_rules(division);
            let mut immediate = SymbolicEvaluator::with_machine(&ssa, machine);
            immediate.set_constant(left, ConstValue::U64(7));
            immediate.set_constant(right, ConstValue::U64(0));
            let expected = immediate.evaluate_op_outcome(&source);
            let mut delayed = SymbolicEvaluator::with_machine(&ssa, machine);
            delayed.set_symbolic(left, "dividend");
            delayed.set_constant(right, ConstValue::U64(0));
            let outcome = delayed.evaluate_op_outcome(&source);
            let actual = delayed.get_expression(dest).map_or(outcome, |expression| {
                expression
                    .substitute_named("dividend", &SymbolicExpr::constant(ConstValue::U64(7)))
                    .evaluate_outcome_with(PointerSize::Bit32, |_| Evaluation::Unknown)
            });
            assert_eq!(actual, expected, "{division:?}");
            if division == DivisionRules::RISCV {
                assert_eq!(actual, Evaluation::Value(ConstValue::U64(u64::MAX)));
            }
        }
    }

    #[test]
    fn an_unused_proven_fault_prevents_following_definitions() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let mut variable = |index| {
            ssa.create_variable(
                VariableOrigin::Local(index),
                0,
                DefSite::instruction(0, usize::from(index)),
                MockType::I32,
            )
        };
        let left = variable(0);
        let right = variable(1);
        let dead = variable(2);
        let later = variable(3);
        let mut evaluator = SymbolicEvaluator::with_machine(&ssa, MACHINE);
        evaluator.set_constant(left, ConstValue::I32(1));
        evaluator.set_constant(right, ConstValue::I32(0));
        assert_eq!(
            evaluator.evaluate_op_outcome(&SsaOp::Div {
                dest: dead,
                left,
                right,
                unsigned: false,
                flags: None
            }),
            Evaluation::Trap(ArithmeticTrap::DivideByZero)
        );
        evaluator.evaluate_op(&SsaOp::Const {
            dest: later,
            value: ConstValue::I32(7),
        });
        assert!(evaluator.get_expression(dead).is_none());
        assert!(evaluator.get_expression(later).is_none());
        assert_eq!(
            evaluator.execution_outcome(),
            &Evaluation::Trap(ArithmeticTrap::DivideByZero)
        );
    }

    #[test]
    fn unknown_total_operations_preserve_normal_continuation() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let source = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let other = ssa.create_variable(
            VariableOrigin::Argument(1),
            0,
            DefSite::instruction(0, 1),
            MockType::I32,
        );
        let narrow = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 2),
            MockType::I8,
        );
        let returned = ssa.create_variable(
            VariableOrigin::Local(1),
            0,
            DefSite::instruction(0, 3),
            MockType::I32,
        );
        for checked in [false, true] {
            let mut evaluator = SymbolicEvaluator::with_machine(&ssa, MACHINE);
            evaluator.set_symbolic(source, "input");
            evaluator.evaluate_op(&SsaOp::IntConv {
                dest: narrow,
                operand: source,
                target: MockType::I8,
                unsigned: false,
                overflow_check: checked,
            });
            evaluator.evaluate_op(&SsaOp::Const {
                dest: returned,
                value: ConstValue::I32(7),
            });
            assert_eq!(
                evaluator.execution_outcome(),
                if checked {
                    &Evaluation::Unknown
                } else {
                    &Evaluation::Value(())
                }
            );
        }
        for division in [
            DivisionRules::CHECKED,
            DivisionRules::UNSTATED,
            DivisionRules::ARM,
            DivisionRules::RISCV,
        ] {
            let mut evaluator =
                SymbolicEvaluator::with_machine(&ssa, MACHINE.with_division_rules(division));
            evaluator.set_symbolic(source, "input");
            evaluator.set_symbolic(other, "divisor");
            evaluator.evaluate_op(&SsaOp::Div {
                dest: returned,
                left: source,
                right: other,
                unsigned: false,
                flags: None,
            });
            let total = division == DivisionRules::ARM || division == DivisionRules::RISCV;
            assert_eq!(
                evaluator.execution_outcome(),
                if total {
                    &Evaluation::Value(())
                } else {
                    &Evaluation::Unknown
                }
            );
            evaluator.evaluate_op(&SsaOp::IntConv {
                dest: narrow,
                operand: source,
                target: MockType::I8,
                unsigned: false,
                overflow_check: false,
            });
            assert_eq!(
                evaluator.execution_outcome(),
                if total {
                    &Evaluation::Value(())
                } else {
                    &Evaluation::Unknown
                }
            );
        }
    }

    #[test]
    fn nonreturn_terminal_refuses_continuation_and_preserves_unsupported() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let later = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        for terminal in [
            SsaOp::Unreachable,
            SsaOp::EndFinally,
            SsaOp::InterruptReturn,
        ] {
            let mut evaluator = SymbolicEvaluator::with_machine(&ssa, MACHINE);
            assert_eq!(
                evaluator.evaluate_op_outcome(&terminal),
                Evaluation::Unsupported
            );
            assert_eq!(evaluator.execution_outcome(), &Evaluation::Unsupported);
            assert_eq!(
                evaluator.evaluate_op_outcome(&SsaOp::Const {
                    dest: later,
                    value: ConstValue::I32(7)
                }),
                Evaluation::Unsupported
            );
            assert!(evaluator.get_expression(later).is_none());
        }
        let mut evaluator = SymbolicEvaluator::with_machine(&ssa, MACHINE);
        assert_eq!(
            evaluator.evaluate_op_outcome(&SsaOp::Rethrow),
            Evaluation::Unknown
        );
        evaluator.evaluate_op(&SsaOp::Const {
            dest: later,
            value: ConstValue::I32(7),
        });
        assert_eq!(evaluator.execution_outcome(), &Evaluation::Unknown);
        assert!(evaluator.get_expression(later).is_none());
    }

    #[test]
    fn a_delayed_checked_conversion_retains_the_target_and_fault() {
        let mut ssa = SsaFunction::<MockTarget>::new(0, 0);
        let operand = ssa.create_variable(
            VariableOrigin::Argument(0),
            0,
            DefSite::instruction(0, 0),
            MockType::I32,
        );
        let dest = ssa.create_variable(
            VariableOrigin::Local(0),
            0,
            DefSite::instruction(0, 1),
            MockType::I8,
        );
        let source = SsaOp::IntConv {
            dest,
            operand,
            target: MockType::I8,
            unsigned: false,
            overflow_check: true,
        };
        let mut evaluator = SymbolicEvaluator::with_machine(&ssa, MACHINE);
        evaluator.set_symbolic(operand, "input");
        assert_eq!(evaluator.evaluate_op_outcome(&source), Evaluation::Unknown);
        let expression = evaluator
            .get_expression(dest)
            .unwrap()
            .substitute_named("input", &SymbolicExpr::constant_i32(511));
        assert_eq!(
            expression
                .simplify(PointerSize::Bit64)
                .evaluate_outcome_with(PointerSize::Bit64, |_| Evaluation::Unknown),
            Evaluation::Trap(ArithmeticTrap::Overflow),
        );
    }
}
